package main

import (
	"strings"
	"testing"

	"fmt"
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

// TestContextEstimateMatchesAFullMeasurement covers the cache behind the
// sidebar's context readout. It must agree with measuring from scratch after
// every append, and must start over when a compaction replaces the transcript.
func TestContextEstimateMatchesAFullMeasurement(t *testing.T) {
	var estimate contextEstimate

	var messages []agent.Message
	for i := range 40 {
		messages = append(messages, llm.UserMessage{
			Role: "user", Content: fmt.Sprintf("message number %d with some content", i),
		})
		if got, want := estimate.estimate(messages), estimateMessagesTokens(messages); got != want {
			t.Fatalf("after %d messages: cached %d, full measurement %d", len(messages), got, want)
		}
	}

	// A compaction replaces the transcript with a shorter one.
	compacted := []agent.Message{
		llm.UserMessage{Role: "user", Content: "a summary of everything above"},
		llm.UserMessage{Role: "user", Content: "the most recent turn"},
	}
	if got, want := estimate.estimate(compacted), estimateMessagesTokens(compacted); got != want {
		t.Fatalf("after compaction: cached %d, full measurement %d", got, want)
	}

	// Growing again from the compacted base must stay correct.
	compacted = append(compacted, llm.UserMessage{Role: "user", Content: "a new turn after compaction"})
	if got, want := estimate.estimate(compacted), estimateMessagesTokens(compacted); got != want {
		t.Fatalf("after regrowth: cached %d, full measurement %d", got, want)
	}
}

// TestContextEstimateTracksAMutatingFinalMessage covers the streaming case: the
// last message is rewritten in place as deltas arrive, so it must not be cached.
func TestContextEstimateTracksAMutatingFinalMessage(t *testing.T) {
	var estimate contextEstimate
	messages := []agent.Message{
		llm.UserMessage{Role: "user", Content: "question"},
		llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "par"}}},
	}
	first := estimate.estimate(messages)

	messages[1] = llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{
		llm.TextContent{Type: "text", Text: strings.Repeat("a much longer answer ", 50)},
	}}
	second := estimate.estimate(messages)
	if second <= first {
		t.Fatalf("estimate did not grow with the streaming message: %d then %d", first, second)
	}
	if want := estimateMessagesTokens(messages); second != want {
		t.Fatalf("cached %d, full measurement %d", second, want)
	}
}
