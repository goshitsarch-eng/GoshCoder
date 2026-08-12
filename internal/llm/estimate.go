package llm

import (
	"encoding/json"
	"math"
)

// Port of reference/pi/packages/ai/src/utils/estimate.ts. Used by
// StreamSimple (buildBaseOptions) to clamp max tokens to the context window.

const charsPerToken = 4
const estimatedImageChars = 4800

// ContextUsageEstimate is a token estimate for a context.
type ContextUsageEstimate struct {
	// Estimated total context tokens.
	Tokens int
	// Tokens reported by the most recent applicable assistant usage block.
	UsageTokens int
	// Estimated tokens after the most recent applicable assistant usage block.
	TrailingTokens int
	// Index of the applicable message that provided usage, or -1 when none exists.
	LastUsageIndex int
}

// CalculateContextTokens sums a usage block (utils/estimate.ts).
func CalculateContextTokens(usage *Usage) int {
	if usage.TotalTokens != 0 {
		return usage.TotalTokens
	}
	return usage.Input + usage.Output + usage.CacheRead + usage.CacheWrite
}

func safeJSONStringify(value any) string {
	data, err := json.Marshal(value)
	if err != nil {
		return "[unserializable]"
	}
	return string(data)
}

// EstimateTextTokens estimates tokens for plain text (chars/4, ceil).
func EstimateTextTokens(text string) int {
	return int(math.Ceil(float64(len(text)) / charsPerToken))
}

func estimateTextAndImageContentTokens(content any) int {
	switch c := content.(type) {
	case string:
		return EstimateTextTokens(c)
	case []ContentBlock:
		chars := 0
		for _, block := range c {
			switch b := block.(type) {
			case TextContent:
				chars += len(b.Text)
			case ImageContent:
				chars += estimatedImageChars
			}
		}
		return int(math.Ceil(float64(chars) / charsPerToken))
	default:
		return 0
	}
}

// EstimateMessageTokens estimates tokens for one message.
func EstimateMessageTokens(message Message) int {
	switch msg := message.(type) {
	case UserMessage:
		return estimateTextAndImageContentTokens(msg.Content)
	case *UserMessage:
		return estimateTextAndImageContentTokens(msg.Content)
	case ToolResultMessage:
		return estimateTextAndImageContentTokens(msg.Content)
	case *ToolResultMessage:
		return estimateTextAndImageContentTokens(msg.Content)
	case AssistantMessage:
		return estimateAssistantMessageTokens(&msg)
	case *AssistantMessage:
		return estimateAssistantMessageTokens(msg)
	default:
		return 0
	}
}

func estimateAssistantMessageTokens(msg *AssistantMessage) int {
	chars := 0
	for _, block := range msg.Content {
		switch b := block.(type) {
		case TextContent:
			chars += len(b.Text)
		case ThinkingContent:
			chars += len(b.Thinking)
		case ToolCall:
			chars += len(b.Name) + len(safeJSONStringify(b.Arguments))
		}
	}
	return int(math.Ceil(float64(chars) / charsPerToken))
}

func lastAssistantUsageInfo(messages []Message) (usage *Usage, index int) {
	latestPrefixTimestamp := int64(math.MinInt64)
	index = -1
	for i, message := range messages {
		if assistant, ok := asAssistant(message); ok {
			// A newer prefix message was inserted after this response, so its
			// usage cannot describe the current prefix.
			usageAppliesToPrefix := assistant.Timestamp >= latestPrefixTimestamp
			if usageAppliesToPrefix &&
				assistant.StopReason != StopAborted &&
				assistant.StopReason != StopError &&
				CalculateContextTokens(&assistant.Usage) > 0 {
				usage = &assistant.Usage
				index = i
			}
		}
		if ts := messageTimestamp(message); ts > latestPrefixTimestamp {
			latestPrefixTimestamp = ts
		}
	}
	return usage, index
}

func asAssistant(m Message) (*AssistantMessage, bool) {
	switch msg := m.(type) {
	case AssistantMessage:
		return &msg, true
	case *AssistantMessage:
		return msg, true
	default:
		return nil, false
	}
}

func messageTimestamp(m Message) int64 {
	switch msg := m.(type) {
	case UserMessage:
		return msg.Timestamp
	case *UserMessage:
		return msg.Timestamp
	case AssistantMessage:
		return msg.Timestamp
	case *AssistantMessage:
		return msg.Timestamp
	case ToolResultMessage:
		return msg.Timestamp
	case *ToolResultMessage:
		return msg.Timestamp
	default:
		return 0
	}
}

func estimateToolsTokens(tools []Tool) int {
	if len(tools) == 0 {
		return 0
	}
	return EstimateTextTokens(safeJSONStringify(tools))
}

// EstimateContextTokens estimates the token count of a context, preferring
// the last reported assistant usage when available (utils/estimate.ts).
func EstimateContextTokens(context *Context) ContextUsageEstimate {
	usage, usageIndex := lastAssistantUsageInfo(context.Messages)
	if usage != nil {
		usageTokens := CalculateContextTokens(usage)
		trailingTokens := 0
		for i := usageIndex + 1; i < len(context.Messages); i++ {
			trailingTokens += EstimateMessageTokens(context.Messages[i])
		}
		addedNames := map[string]bool{}
		for _, message := range context.Messages[usageIndex+1:] {
			if tr, ok := message.(ToolResultMessage); ok {
				for _, n := range tr.AddedToolNames {
					addedNames[n] = true
				}
			}
			if tr, ok := message.(*ToolResultMessage); ok {
				for _, n := range tr.AddedToolNames {
					addedNames[n] = true
				}
			}
		}
		var addedTools []Tool
		for _, tool := range context.Tools {
			if addedNames[tool.Name] {
				addedTools = append(addedTools, tool)
			}
		}
		trailingTokens += estimateToolsTokens(addedTools)
		return ContextUsageEstimate{
			Tokens:         usageTokens + trailingTokens,
			UsageTokens:    usageTokens,
			TrailingTokens: trailingTokens,
			LastUsageIndex: usageIndex,
		}
	}

	tokens := 0
	for _, message := range context.Messages {
		tokens += EstimateMessageTokens(message)
	}
	prefixTokens := estimateToolsTokens(context.Tools)
	if context.SystemPrompt != "" {
		prefixTokens += EstimateTextTokens(context.SystemPrompt)
	}
	return ContextUsageEstimate{
		Tokens:         tokens + prefixTokens,
		UsageTokens:    0,
		TrailingTokens: tokens + prefixTokens,
		LastUsageIndex: -1,
	}
}
