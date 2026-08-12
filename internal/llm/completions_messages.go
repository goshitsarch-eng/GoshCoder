package llm

import (
	"encoding/json"
	"regexp"
	"strings"
)

// Port of convertMessages in reference/pi/packages/ai/src/api/openai-completions.ts.
//
// Outgoing messages are built as map[string]any because the endpoint accepts
// non-standard fields pi sets conditionally (reasoning_content,
// reasoning_details, cache_control, Kimi tool system messages, custom tool
// calls), which the TS expresses via `as any` casts.

var toolCallIDSanitizer = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// convertMessages converts pi messages into OpenAI chat completion message
// params.
func convertMessages(model *Model, conv *Context, compat resolvedCompat) []any {
	var params []any

	normalizeToolCallID := func(id string, m *Model, source *AssistantMessage) string {
		// Handle pipe-separated IDs from OpenAI Responses API
		// ({call_id}|{id}); preserve item-level uniqueness when replaying into
		// Chat Completions, which requires distinct tool call ids.
		if idx := strings.Index(id, "|"); idx >= 0 {
			callID := toolCallIDSanitizer.ReplaceAllString(id[:idx], "_")
			itemID := toolCallIDSanitizer.ReplaceAllString(id[idx+1:], "_")
			combinedID := callID
			if itemID != "" {
				combinedID = callID + "_" + itemID
			}
			if len(combinedID) <= 40 {
				return combinedID
			}
			hash := ShortHash(id)
			if len(hash) > 8 {
				hash = hash[:8]
			}
			prefixLen := max(1, 40-len(hash)-1)
			if len(callID) > prefixLen {
				callID = callID[:prefixLen]
			}
			return callID + "_" + hash
		}
		if m.Provider == "openai" && len(id) > 40 {
			return id[:40]
		}
		return id
	}

	transformedMessages := TransformMessages(conv.Messages, model, normalizeToolCallID)

	if conv.SystemPrompt != "" {
		role := "system"
		if model.Reasoning && compat.SupportsDeveloperRole {
			role = "developer"
		}
		params = append(params, map[string]any{
			"role":    role,
			"content": sanitizeSurrogates(conv.SystemPrompt),
		})
	}

	lastRole := ""

	for i := 0; i < len(transformedMessages); i++ {
		msg := transformedMessages[i]
		role := messageRole(msg)

		// Some providers don't allow user messages directly after tool
		// results; insert a synthetic assistant message to bridge the gap.
		if compat.RequiresAssistantAfterToolResult && lastRole == "toolResult" && role == "user" {
			params = append(params, map[string]any{
				"role":    "assistant",
				"content": "I have processed the tool results.",
			})
		}

		switch m := msg.(type) {
		case UserMessage:
			if s, ok := m.Content.(string); ok {
				params = append(params, map[string]any{
					"role":    "user",
					"content": sanitizeSurrogates(s),
				})
			} else if blocks, ok := m.Content.([]ContentBlock); ok {
				var content []any
				for _, item := range blocks {
					switch b := item.(type) {
					case TextContent:
						content = append(content, map[string]any{
							"type": "text",
							"text": sanitizeSurrogates(b.Text),
						})
					case ImageContent:
						content = append(content, map[string]any{
							"type":      "image_url",
							"image_url": map[string]any{"url": "data:" + b.MimeType + ";base64," + b.Data},
						})
					}
				}
				if len(content) == 0 {
					continue
				}
				params = append(params, map[string]any{"role": "user", "content": content})
			} else {
				continue
			}

		case AssistantMessage:
			params = appendAssistantMessage(params, m, model, compat)

		case ToolResultMessage:
			var imageBlocks []any
			deferredToolNames := []string{}
			seenDeferred := map[string]bool{}
			j := i

			for ; j < len(transformedMessages); j++ {
				toolMsg, ok := transformedMessages[j].(ToolResultMessage)
				if !ok {
					break
				}

				var texts []string
				hasImages := false
				for _, block := range toolMsg.Content {
					switch b := block.(type) {
					case TextContent:
						texts = append(texts, b.Text)
					case ImageContent:
						hasImages = true
						_ = b
					}
				}

				// Always send tool result with text (or placeholder if only images).
				textResult := strings.Join(texts, "\n")
				toolResultText := textResult
				if toolResultText == "" {
					if hasImages {
						toolResultText = "(see attached image)"
					} else {
						toolResultText = "(no tool output)"
					}
				}
				toolResultMsg := map[string]any{
					"role":         "tool",
					"content":      sanitizeSurrogates(toolResultText),
					"tool_call_id": toolMsg.ToolCallID,
				}
				// Some providers require the 'name' field in tool results.
				if compat.RequiresToolResultName && toolMsg.ToolName != "" {
					toolResultMsg["name"] = toolMsg.ToolName
				}
				params = append(params, toolResultMsg)

				if compat.DeferredToolsMode == "kimi" {
					for _, name := range toolMsg.AddedToolNames {
						if !seenDeferred[name] {
							seenDeferred[name] = true
							deferredToolNames = append(deferredToolNames, name)
						}
					}
				}

				if hasImages && model.SupportsImages() {
					for _, block := range toolMsg.Content {
						if img, ok := block.(ImageContent); ok {
							imageBlocks = append(imageBlocks, map[string]any{
								"type":      "image_url",
								"image_url": map[string]any{"url": "data:" + img.MimeType + ";base64," + img.Data},
							})
						}
					}
				}
			}

			i = j - 1

			if len(imageBlocks) > 0 {
				if compat.RequiresAssistantAfterToolResult {
					params = append(params, map[string]any{
						"role":    "assistant",
						"content": "I have processed the tool results.",
					})
				}
				content := []any{map[string]any{"type": "text", "text": "Attached image(s) from tool result:"}}
				content = append(content, imageBlocks...)
				params = append(params, map[string]any{"role": "user", "content": content})
				lastRole = "user"
			} else {
				lastRole = "toolResult"
			}

			if len(deferredToolNames) > 0 {
				if deferredTools := getToolsByName(conv.Tools, deferredToolNames); len(deferredTools) > 0 {
					// Kimi accepts a system message with tools but omits the
					// standard content field.
					params = append(params, map[string]any{
						"role":  "system",
						"tools": convertTools(deferredTools, compat),
					})
				}
			}
			continue
		}

		lastRole = role
	}

	return params
}

// appendAssistantMessage converts one assistant message and appends it (the
// msg.role === "assistant" branch of convertMessages).
func appendAssistantMessage(params []any, msg AssistantMessage, model *Model, compat resolvedCompat) []any {
	// Some providers don't accept null content, use empty string instead.
	assistantMsg := map[string]any{"role": "assistant"}
	var content any // nil = JSON null
	if compat.RequiresAssistantAfterToolResult {
		content = ""
	}
	assistantMsg["content"] = content

	var textParts []string
	for _, block := range msg.Content {
		if tc, ok := block.(TextContent); ok && strings.TrimSpace(tc.Text) != "" {
			textParts = append(textParts, sanitizeSurrogates(tc.Text))
		}
	}
	assistantText := strings.Join(textParts, "")

	var nonEmptyThinking []ThinkingContent
	for _, block := range msg.Content {
		if tc, ok := block.(ThinkingContent); ok && strings.TrimSpace(tc.Thinking) != "" {
			nonEmptyThinking = append(nonEmptyThinking, tc)
		}
	}
	if len(nonEmptyThinking) > 0 {
		if compat.RequiresThinkingAsText {
			// Convert thinking blocks to plain text (no tags to avoid model
			// mimicking them).
			var thinkingTexts []string
			for _, b := range nonEmptyThinking {
				thinkingTexts = append(thinkingTexts, sanitizeSurrogates(b.Thinking))
			}
			parts := []any{map[string]any{"type": "text", "text": strings.Join(thinkingTexts, "\n\n")}}
			for _, t := range textParts {
				parts = append(parts, map[string]any{"type": "text", "text": t})
			}
			assistantMsg["content"] = parts
		} else {
			// Always send assistant content as a plain string (OpenAI Chat
			// Completions API standard format). See openai-completions.ts for
			// why array-form content breaks some providers.
			if len(assistantText) > 0 {
				assistantMsg["content"] = assistantText
			}

			// Use the signature from the first thinking block if available
			// (for llama.cpp server + gpt-oss).
			signature := nonEmptyThinking[0].ThinkingSignature
			if model.Provider == "opencode-go" && signature == "reasoning" {
				signature = "reasoning_content"
			}
			if signature != "" {
				var thinkings []string
				for _, b := range nonEmptyThinking {
					thinkings = append(thinkings, b.Thinking)
				}
				assistantMsg[signature] = strings.Join(thinkings, "\n")
			}
		}
	} else if len(assistantText) > 0 {
		assistantMsg["content"] = assistantText
	}

	var toolCalls []ToolCall
	for _, block := range msg.Content {
		if tc, ok := block.(ToolCall); ok {
			toolCalls = append(toolCalls, tc)
		}
	}
	if len(toolCalls) > 0 {
		wire := make([]any, 0, len(toolCalls))
		for _, tc := range toolCalls {
			args, err := json.Marshal(tc.Arguments)
			if err != nil {
				args = []byte("{}")
			}
			wire = append(wire, map[string]any{
				"id":   tc.ID,
				"type": "function",
				"function": map[string]any{
					"name":      tc.Name,
					"arguments": string(args),
				},
			})
		}
		assistantMsg["tool_calls"] = wire

		var reasoningDetails []any
		for _, tc := range toolCalls {
			if tc.ThoughtSignature == "" {
				continue
			}
			var detail any
			if err := json.Unmarshal([]byte(tc.ThoughtSignature), &detail); err == nil && detail != nil {
				reasoningDetails = append(reasoningDetails, detail)
			}
		}
		if len(reasoningDetails) > 0 {
			assistantMsg["reasoning_details"] = reasoningDetails
		}
	}

	if compat.RequiresReasoningContentOnAssistantMessages && model.Reasoning {
		if _, ok := assistantMsg["reasoning_content"]; !ok {
			assistantMsg["reasoning_content"] = ""
		}
	}

	// Skip assistant messages that have no content and no tool calls. Some
	// providers require "either content or tool_calls, but not none". This
	// handles aborted assistant responses that got no content.
	hasContent := false
	switch c := assistantMsg["content"].(type) {
	case string:
		hasContent = len(c) > 0
	case []any:
		hasContent = len(c) > 0
	}
	_, hasToolCalls := assistantMsg["tool_calls"]
	if !hasContent && !hasToolCalls {
		return params
	}
	return append(params, assistantMsg)
}

func messageRole(m Message) string {
	return m.messageRole()
}

// getToolsByName returns tools matching names, preserving the given order.
func getToolsByName(tools []Tool, names []string) []Tool {
	byName := map[string]Tool{}
	for _, tool := range tools {
		byName[tool.Name] = tool
	}
	var out []Tool
	for _, name := range names {
		if tool, ok := byName[name]; ok {
			out = append(out, tool)
		}
	}
	return out
}
