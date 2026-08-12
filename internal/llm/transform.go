package llm

import (
	"strings"
	"time"
)

// Port of reference/pi/packages/ai/src/api/transform-messages.ts.

const (
	nonVisionUserImagePlaceholder = "(image omitted: model does not support images)"
	nonVisionToolImagePlaceholder = "(tool image omitted: model does not support images)"
)

func replaceImagesWithPlaceholder(content []ContentBlock, placeholder string) []ContentBlock {
	result := make([]ContentBlock, 0, len(content))
	previousWasPlaceholder := false

	for _, block := range content {
		if _, isImage := block.(ImageContent); isImage {
			if !previousWasPlaceholder {
				result = append(result, TextContent{Type: "text", Text: placeholder})
			}
			previousWasPlaceholder = true
			continue
		}
		result = append(result, block)
		if tc, ok := block.(TextContent); ok && tc.Text == placeholder {
			previousWasPlaceholder = true
		} else {
			previousWasPlaceholder = false
		}
	}
	return result
}

// downgradeUnsupportedImages replaces image blocks with text placeholders when
// the model has no image input modality.
func downgradeUnsupportedImages(messages []Message, model *Model) []Message {
	if model.SupportsImages() {
		return messages
	}
	out := make([]Message, len(messages))
	for i, msg := range messages {
		switch m := msg.(type) {
		case UserMessage:
			if blocks, ok := m.Content.([]ContentBlock); ok {
				m.Content = replaceImagesWithPlaceholder(blocks, nonVisionUserImagePlaceholder)
				out[i] = m
				continue
			}
			out[i] = msg
		case ToolResultMessage:
			m.Content = replaceImagesWithPlaceholder(m.Content, nonVisionToolImagePlaceholder)
			out[i] = m
		default:
			out[i] = msg
		}
	}
	return out
}

// TransformMessages normalizes a conversation for replay: downgrades
// unsupported images, normalizes thinking blocks and tool call IDs for
// cross-model turns, drops errored/aborted assistant messages, and inserts
// synthetic tool results for orphaned tool calls. Port of transformMessages
// in api/transform-messages.ts.
//
// normalizeToolCallId is only consulted for cross-model tool calls.
func TransformMessages(messages []Message, model *Model, normalizeToolCallId func(id string, model *Model, source *AssistantMessage) string) []Message {
	toolCallIDMap := map[string]string{}
	// Normalize nil content from untyped callers so downstream code can rely
	// on the type contract (transform-messages.ts).
	normalized := make([]Message, len(messages))
	for i, msg := range messages {
		switch m := msg.(type) {
		case UserMessage:
			if m.Content == nil {
				m.Content = []ContentBlock{}
			}
			normalized[i] = m
		case AssistantMessage:
			if m.Content == nil {
				m.Content = []ContentBlock{}
			}
			normalized[i] = m
		case ToolResultMessage:
			if m.Content == nil {
				m.Content = []ContentBlock{}
			}
			normalized[i] = m
		default:
			normalized[i] = msg
		}
	}
	imageAware := downgradeUnsupportedImages(normalized, model)

	// First pass: transform messages (unsupported image downgrade, thinking
	// blocks, tool call ID normalization).
	transformed := make([]Message, 0, len(imageAware))
	for _, msg := range imageAware {
		assistant, isAssistant := asAssistant(msg)
		if !isAssistant {
			// Tool results: normalize toolCallId if we have a mapping.
			if tr, ok := msg.(ToolResultMessage); ok {
				if normalizedID, found := toolCallIDMap[tr.ToolCallID]; found && normalizedID != tr.ToolCallID {
					tr.ToolCallID = normalizedID
					transformed = append(transformed, tr)
					continue
				}
			}
			if tr, ok := msg.(*ToolResultMessage); ok {
				if normalizedID, found := toolCallIDMap[tr.ToolCallID]; found && normalizedID != tr.ToolCallID {
					copy := *tr
					copy.ToolCallID = normalizedID
					transformed = append(transformed, copy)
					continue
				}
			}
			transformed = append(transformed, msg)
			continue
		}

		isSameModel := assistant.Provider == model.Provider &&
			assistant.API == model.API &&
			assistant.Model == model.ID

		transformedContent := make([]ContentBlock, 0, len(assistant.Content))
		for _, block := range assistant.Content {
			switch b := block.(type) {
			case ThinkingContent:
				// Redacted thinking is opaque encrypted content, only valid for
				// the same model. Drop it for cross-model to avoid API errors.
				if b.Redacted {
					if isSameModel {
						transformedContent = append(transformedContent, b)
					}
					continue
				}
				// For same model: keep thinking blocks with signatures (needed
				// for replay) even if the thinking text is empty.
				if isSameModel && b.ThinkingSignature != "" {
					transformedContent = append(transformedContent, b)
					continue
				}
				// Skip empty thinking blocks, convert others to plain text.
				if strings.TrimSpace(b.Thinking) == "" {
					continue
				}
				if isSameModel {
					transformedContent = append(transformedContent, b)
				} else {
					transformedContent = append(transformedContent, TextContent{Type: "text", Text: b.Thinking})
				}
			case TextContent:
				transformedContent = append(transformedContent, b)
			case ToolCall:
				tc := b
				if !isSameModel && tc.ThoughtSignature != "" {
					tc.ThoughtSignature = ""
				}
				if !isSameModel && normalizeToolCallId != nil {
					if normalizedID := normalizeToolCallId(tc.ID, model, assistant); normalizedID != tc.ID {
						toolCallIDMap[tc.ID] = normalizedID
						tc.ID = normalizedID
					}
				}
				transformedContent = append(transformedContent, tc)
			default:
				transformedContent = append(transformedContent, block)
			}
		}

		copy := *assistant
		copy.Content = transformedContent
		transformed = append(transformed, copy)
	}

	// Second pass: insert synthetic empty tool results for orphaned tool calls.
	// This preserves thinking signatures and satisfies API requirements.
	result := make([]Message, 0, len(transformed))
	var pendingToolCalls []ToolCall
	existingToolResultIDs := map[string]bool{}
	insertSyntheticToolResults := func() {
		if len(pendingToolCalls) == 0 {
			return
		}
		for _, tc := range pendingToolCalls {
			if !existingToolResultIDs[tc.ID] {
				result = append(result, ToolResultMessage{
					Role:       "toolResult",
					ToolCallID: tc.ID,
					ToolName:   tc.Name,
					Content:    []ContentBlock{TextContent{Type: "text", Text: "No result provided"}},
					IsError:    true,
					Timestamp:  time.Now().UnixMilli(),
				})
			}
		}
		pendingToolCalls = nil
		existingToolResultIDs = map[string]bool{}
	}

	for _, msg := range transformed {
		switch m := msg.(type) {
		case AssistantMessage:
			// If we have pending orphaned tool calls from a previous
			// assistant, insert synthetic results now.
			insertSyntheticToolResults()

			// Skip errored/aborted assistant messages entirely. These are
			// incomplete turns that shouldn't be replayed.
			if m.StopReason == StopError || m.StopReason == StopAborted {
				continue
			}

			// Track tool calls from this assistant message.
			var toolCalls []ToolCall
			for _, block := range m.Content {
				if tc, ok := block.(ToolCall); ok {
					toolCalls = append(toolCalls, tc)
				}
			}
			if len(toolCalls) > 0 {
				pendingToolCalls = toolCalls
				existingToolResultIDs = map[string]bool{}
			}
			result = append(result, m)
		case ToolResultMessage:
			existingToolResultIDs[m.ToolCallID] = true
			result = append(result, m)
		case UserMessage:
			// User message interrupts tool flow - insert synthetic results.
			insertSyntheticToolResults()
			result = append(result, m)
		default:
			result = append(result, msg)
		}
	}

	// If the conversation ends with unresolved tool calls, synthesize results now.
	insertSyntheticToolResults()

	return result
}
