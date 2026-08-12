package main

import (
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

func TestCompactionCutKeepsLatestTurn(t *testing.T) {
	messages := []agent.Message{
		userMessage("first request"),
		llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "first response"}}},
		userMessage("latest request"),
		llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "latest response"}}},
	}
	if cut := compactionCutIndex(messages, 200_000); cut != 2 {
		t.Fatalf("cut = %d, want 2", cut)
	}
}

func TestCompactionSerializationTruncatesToolResults(t *testing.T) {
	result := llm.ToolResultMessage{
		Role: "toolResult", ToolName: "read",
		Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: strings.Repeat("x", 3_000)}},
	}
	serialized := serializeCompactionMessage(result)
	if !strings.Contains(serialized, "1000 characters truncated") || len(serialized) > 2_100 {
		t.Fatalf("serialized tool result length/content = %d, %q", len(serialized), serialized[len(serialized)-80:])
	}
}

func TestCompactionSummaryReachesModelButNotDefaultTranscript(t *testing.T) {
	summary := compactionSummaryMessage{Summary: "## Goal\nKeep parity", Timestamp: 42}
	converted := convertSessionMessages([]agent.Message{summary, userMessage("continue")})
	if len(converted) != 2 {
		t.Fatalf("converted = %#v", converted)
	}
	first, ok := converted[0].(llm.UserMessage)
	if !ok || !strings.Contains(first.Content.(string), "<conversation-summary>") {
		t.Fatalf("first = %#v", converted[0])
	}
	if visible := fullscreenMessages([]agent.Message{summary}); len(visible) != 0 {
		t.Fatalf("summary should stay out of transcript: %#v", visible)
	}
}

func TestConversationCostSurvivesCompaction(t *testing.T) {
	before := llm.AssistantMessage{Usage: llm.Usage{Cost: llm.UsageCost{Total: 1.25}}}
	after := llm.AssistantMessage{Usage: llm.Usage{Cost: llm.UsageCost{Total: 0.50}}}
	messages := []agent.Message{
		compactionSummaryMessage{CostBefore: 1.25, RetainedMessages: 1},
		before,
		userMessage("new turn"),
		after,
	}
	if got := conversationCost(messages); got != 1.75 {
		t.Fatalf("cost = %v", got)
	}
}

func TestFullscreenCompactRunsAsynchronously(t *testing.T) {
	if !fullscreenCommandRunsAsync("/compact focus on tests") || fullscreenCommandRunsAsync("/status") {
		t.Fatal("async command classification is wrong")
	}
}
