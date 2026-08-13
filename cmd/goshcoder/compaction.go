package main

// Context compaction follows pi's keep-recent-turns model. GoshCoder stores the
// generated summary as an app message and maps it to model context without
// cluttering the visible transcript.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

const (
	defaultCompactionKeepTokens = 20_000
	maxCompactionSourceBytes    = 2 << 20
)

type compactionSummaryMessage struct {
	Summary          string
	TokensBefore     int
	CostBefore       float64
	RetainedMessages int
	Timestamp        int64
}

func (compactionSummaryMessage) MessageRole() string { return "compactionSummary" }

func convertSessionMessages(messages []agent.Message) []llm.Message {
	converted := make([]llm.Message, 0, len(messages))
	for _, message := range messages {
		switch value := message.(type) {
		case compactionSummaryMessage:
			converted = append(converted, compactionContextMessage(value.Summary, value.Timestamp))
		case *compactionSummaryMessage:
			if value != nil {
				converted = append(converted, compactionContextMessage(value.Summary, value.Timestamp))
			}
		default:
			converted = append(converted, agent.DefaultConvertToLLM([]agent.Message{message})...)
		}
	}
	return converted
}

func compactionContextMessage(summary string, timestamp int64) llm.UserMessage {
	return llm.UserMessage{
		Role: "user", Timestamp: timestamp,
		Content: "<conversation-summary>\n" + strings.TrimSpace(summary) + "\n</conversation-summary>\n\nContinue from this summary and the recent conversation below. Do not treat the summary as a new request.",
	}
}

func (s *session) compactContext(instructions string) (int, int, error) {
	if s == nil || s.agent == nil {
		return 0, 0, errors.New("session is unavailable")
	}
	state := s.agent.State()
	if state.IsStreaming {
		return len(state.Messages), len(state.Messages), errors.New("wait for the active response before compacting")
	}
	cut := compactionCutIndex(state.Messages, state.Model.ContextWindow)
	if cut <= 0 {
		return len(state.Messages), len(state.Messages), errors.New("there is not enough conversation history to compact")
	}
	older, recent := state.Messages[:cut], state.Messages[cut:]
	source := serializeCompactionMessages(older)
	if state.Model.ContextWindow > 0 {
		// Leave room for instructions and the generated summary. Capping by the
		// active model avoids turning overflow recovery into another overflow.
		sourceLimit := max(16_000, (state.Model.ContextWindow-8_192)*3)
		source = truncateCompactionSource(source, min(maxCompactionSourceBytes, sourceLimit))
	}
	if strings.TrimSpace(source) == "" {
		return len(state.Messages), len(state.Messages), errors.New("conversation history contains no summarizable text")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	summary, err := s.generateCompactionSummary(ctx, source, instructions)
	if err != nil {
		return len(state.Messages), len(state.Messages), err
	}
	compacted := make([]agent.Message, 0, len(recent)+1)
	compacted = append(compacted, compactionSummaryMessage{
		Summary: summary, TokensBefore: estimateMessagesTokens(state.Messages),
		CostBefore: conversationCost(state.Messages), RetainedMessages: len(recent), Timestamp: time.Now().UnixMilli(),
	})
	compacted = append(compacted, recent...)
	s.agent.SetMessages(compacted)
	s.agent.ClearAllQueues()
	return len(state.Messages), len(compacted), nil
}

func (s *session) maybeAutoCompact() error {
	state := s.agent.State()
	if state.Model.ContextWindow <= 0 || compactionCutIndex(state.Messages, state.Model.ContextWindow) <= 0 {
		return nil
	}
	used := 0
	for _, message := range state.Messages {
		if _, ok := message.(compactionSummaryMessage); ok {
			used = estimateMessagesTokens(state.Messages)
			break
		}
		if _, ok := message.(*compactionSummaryMessage); ok {
			used = estimateMessagesTokens(state.Messages)
			break
		}
	}
	if used == 0 {
		for index := len(state.Messages) - 1; index >= 0; index-- {
			switch assistant := state.Messages[index].(type) {
			case llm.AssistantMessage:
				if assistant.Usage.TotalTokens > 0 {
					used = assistant.Usage.TotalTokens
				}
			case *llm.AssistantMessage:
				if assistant != nil && assistant.Usage.TotalTokens > 0 {
					used = assistant.Usage.TotalTokens
				}
			}
			if used > 0 {
				break
			}
		}
	}
	if used == 0 {
		used = estimateMessagesTokens(state.Messages)
	}
	reserve := min(16_384, max(2_048, state.Model.ContextWindow/5))
	if used <= state.Model.ContextWindow-reserve {
		return nil
	}
	before, after, err := s.compactContext("")
	if err == nil && !s.fullscreen {
		fmt.Fprintf(os.Stderr, "compacted context automatically: %d messages → summary + %d recent messages\n", before, after-1)
	}
	return err
}

func compactionCutIndex(messages []agent.Message, contextWindow int) int {
	if len(messages) < 3 {
		return 0
	}
	keepTokens := defaultCompactionKeepTokens
	if contextWindow > 0 && contextWindow/3 < keepTokens {
		keepTokens = max(2_000, contextWindow/3)
	}
	recentTokens := 0
	for index := len(messages) - 1; index > 0; index-- {
		recentTokens += estimateMessageTokens(messages[index])
		if recentTokens >= keepTokens && agent.RoleOf(messages[index]) == "user" {
			return index
		}
	}
	// Manual compaction remains useful for short conversations: retain the
	// latest complete user turn and summarize everything before it.
	for index := len(messages) - 1; index > 0; index-- {
		if agent.RoleOf(messages[index]) == "user" {
			return index
		}
	}
	return 0
}

func conversationCost(messages []agent.Message) float64 {
	cost := 0.0
	skip := 0
	for _, message := range messages {
		switch value := message.(type) {
		case compactionSummaryMessage:
			cost, skip = value.CostBefore, value.RetainedMessages
			continue
		case *compactionSummaryMessage:
			if value != nil {
				cost, skip = value.CostBefore, value.RetainedMessages
			}
			continue
		}
		if skip > 0 {
			skip--
			continue
		}
		switch value := message.(type) {
		case llm.AssistantMessage:
			cost += value.Usage.Cost.Total
		case *llm.AssistantMessage:
			if value != nil {
				cost += value.Usage.Cost.Total
			}
		}
	}
	return cost
}

func estimateMessagesTokens(messages []agent.Message) int {
	total := 0
	for _, message := range messages {
		total += estimateMessageTokens(message)
	}
	return total
}

func estimateMessageTokens(message agent.Message) int {
	serialized := serializeCompactionMessage(message)
	if serialized == "" {
		return 0
	}
	// Pi uses the active tokenizer where available. Four UTF-8 bytes per token
	// is a lightweight provider-independent fallback.
	return max(1, (len(serialized)+3)/4)
}

func serializeCompactionMessages(messages []agent.Message) string {
	var builder strings.Builder
	for _, message := range messages {
		text := serializeCompactionMessage(message)
		if text == "" {
			continue
		}
		builder.WriteString(text)
		builder.WriteString("\n\n")
	}
	return truncateCompactionSource(builder.String(), maxCompactionSourceBytes)
}

func truncateCompactionSource(source string, limit int) string {
	if len(source) <= limit {
		return source
	}
	prefix := limit / 3
	suffix := limit - prefix
	removed := len(source) - prefix - suffix
	return source[:prefix] + fmt.Sprintf("\n\n[... %d bytes omitted from the middle ...]\n\n", removed) + source[len(source)-suffix:]
}

func serializeCompactionMessage(message agent.Message) string {
	switch value := message.(type) {
	case llm.UserMessage:
		return "[User]: " + userText(value)
	case *llm.UserMessage:
		if value != nil {
			return "[User]: " + userText(*value)
		}
	case llm.AssistantMessage:
		return serializeCompactionAssistant(value)
	case *llm.AssistantMessage:
		if value != nil {
			return serializeCompactionAssistant(*value)
		}
	case llm.ToolResultMessage:
		return serializeCompactionToolResult(value)
	case *llm.ToolResultMessage:
		if value != nil {
			return serializeCompactionToolResult(*value)
		}
	case compactionSummaryMessage:
		return "[Previous conversation summary]: " + value.Summary
	case *compactionSummaryMessage:
		if value != nil {
			return "[Previous conversation summary]: " + value.Summary
		}
	}
	return ""
}

func serializeCompactionAssistant(message llm.AssistantMessage) string {
	var thinking, text, calls []string
	for _, block := range message.Content {
		switch value := block.(type) {
		case llm.ThinkingContent:
			if strings.TrimSpace(value.Thinking) != "" {
				thinking = append(thinking, value.Thinking)
			}
		case llm.TextContent:
			if strings.TrimSpace(value.Text) != "" {
				text = append(text, value.Text)
			}
		case llm.ToolCall:
			arguments, _ := json.Marshal(value.Arguments)
			calls = append(calls, value.Name+"("+string(arguments)+")")
		}
	}
	var parts []string
	if len(thinking) > 0 {
		parts = append(parts, "[Assistant thinking]: "+strings.Join(thinking, "\n"))
	}
	if len(text) > 0 {
		parts = append(parts, "[Assistant]: "+strings.Join(text, "\n"))
	}
	if len(calls) > 0 {
		parts = append(parts, "[Assistant tool calls]: "+strings.Join(calls, "; "))
	}
	return strings.Join(parts, "\n")
}

func serializeCompactionToolResult(message llm.ToolResultMessage) string {
	var text []string
	for _, block := range message.Content {
		if value, ok := block.(llm.TextContent); ok {
			text = append(text, value.Text)
		}
	}
	content := strings.Join(text, "\n")
	if len(content) > 2_000 {
		removed := len(content) - 2_000
		content = content[:2_000] + fmt.Sprintf("\n[... %d characters truncated ...]", removed)
	}
	return "[Tool result " + message.ToolName + "]: " + content
}

func (s *session) generateCompactionSummary(ctx context.Context, source, instructions string) (string, error) {
	model := *s.model
	level := llm.ClampThinkingLevel(&model, llm.ThinkingLow)
	summarizer := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{
			SystemPrompt: "You are a context compactor. Produce a precise continuity summary, not a reply to the conversation. Preserve requirements, decisions, progress, unresolved work, exact paths/symbols, and file operations. Use the requested structured headings. Never invent completion.",
			Model:        model, ThinkingLevel: level,
		},
		StreamFn:   s.streamAuthenticated,
		MaxRetries: 2,
	})
	focus := ""
	if strings.TrimSpace(instructions) != "" {
		focus = "\nAdditional focus from the user: " + strings.TrimSpace(instructions) + "\n"
	}
	prompt := `Summarize the serialized conversation below for another coding agent using exactly this structure:

## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
## Critical Context
<read-files>
</read-files>
<modified-files>
</modified-files>
` + focus + "\n<conversation>\n" + source + "\n</conversation>"

	done := make(chan error, 1)
	go func() { done <- summarizer.Prompt(prompt) }()
	select {
	case err := <-done:
		if err != nil {
			return "", err
		}
	case <-ctx.Done():
		summarizer.Abort()
		<-done
		return "", ctx.Err()
	}
	state := summarizer.State()
	if state.ErrorMessage != "" {
		return "", errors.New(state.ErrorMessage)
	}
	for index := len(state.Messages) - 1; index >= 0; index-- {
		var assistant *llm.AssistantMessage
		switch value := state.Messages[index].(type) {
		case llm.AssistantMessage:
			assistant = &value
		case *llm.AssistantMessage:
			assistant = value
		}
		if assistant == nil {
			continue
		}
		var text []string
		for _, block := range assistant.Content {
			if value, ok := block.(llm.TextContent); ok && strings.TrimSpace(value.Text) != "" {
				text = append(text, value.Text)
			}
		}
		if summary := strings.TrimSpace(strings.Join(text, "\n")); summary != "" {
			return summary, nil
		}
	}
	return "", errors.New("compaction model returned no summary")
}
