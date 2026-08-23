package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"goshcoder/internal/llm"
)

// TestContinueKeepsQueuedFollowUpsWhenSteeringExists covers a queue drain that
// destroyed messages. Continue drained both queues before consuming either, but
// only ever ran one of them, so a follow-up queued alongside a steering message
// was thrown away with no error and no way to recover it.
func TestContinueKeepsQueuedFollowUpsWhenSteeringExists(t *testing.T) {
	streamFn, _ := streamOf(assistantText("done"))
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn: streamFn,
		InitialState: &InitialState{
			Model:    testModel(),
			Messages: []Message{assistantText("previous answer")},
		},
	})

	agent.Steer(llm.UserMessage{Role: "user", Content: "STEER-ME"})
	agent.FollowUp(llm.UserMessage{Role: "user", Content: "FOLLOW-UP-ME"})

	if err := agent.Continue(); err != nil {
		t.Fatalf("Continue: %v", err)
	}

	// The follow-up must either still be queued or have been run; what it must
	// not be is silently dropped.
	if agent.HasQueuedMessages() {
		return
	}
	for _, message := range agent.State().Messages {
		user, ok := message.(llm.UserMessage)
		if !ok {
			continue
		}
		if text, _ := user.Content.(string); strings.Contains(text, "FOLLOW-UP-ME") {
			return
		}
	}
	t.Fatal("the queued follow-up was neither run nor left queued: Continue discarded it")
}

// panicOnEventListener panics the first time it sees the given event type,
// standing in for any subscriber bug -- a renderer indexing past the end of a
// slice on a narrow terminal, say.
func panicOnEventListener(target EventType) func(context.Context, Event) {
	var once sync.Once
	return func(_ context.Context, ev Event) {
		if ev.Type == target {
			once.Do(func() { panic("listener boom") })
		}
	}
}

// TestPanicInParallelToolWorkerDoesNotKillTheProcess covers the one tool path
// with no recover. executePreparedToolCall and finalizeExecutedToolCall each
// install their own, and the run as a whole is wrapped, but the goroutine body
// in the parallel executor was not -- so a panic raised while emitting an event
// from a worker unwound past every handler and terminated the process, losing
// the transcript.
func TestPanicInParallelToolWorkerDoesNotKillTheProcess(t *testing.T) {
	slow := Tool{
		Name:       "slow",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "ok"}}}, nil
		},
	}

	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "slow", nil), toolCall("c2", "slow", nil)),
		assistantText("finished"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{slow}},
	})
	agent.Subscribe(panicOnEventListener(EventToolExecutionEnd))

	done := make(chan error, 1)
	go func() { done <- agent.Prompt("run both") }()

	select {
	case <-done:
		// Reaching here at all means the panic was contained.
	case <-time.After(20 * time.Second):
		t.Fatal("the run never completed")
	}
}

// TestAbortMidBatchStillAnswersEveryToolCall covers the transcript invariant
// every provider enforces: an assistant message requesting N tool calls must be
// followed by N tool results. Both batch executors used to stop at the first
// cancelled call, so an abort left the transcript with an unanswered tool call
// and the next request was rejected by the provider.
func TestAbortMidBatchStillAnswersEveryToolCall(t *testing.T) {
	var agentRef *Agent
	blocking := Tool{
		Name:          "blocking",
		Parameters:    json.RawMessage(`{"type":"object"}`),
		ExecutionMode: ToolExecutionSequential,
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			// Stands in for the planner's submit-plan tool, which blocks on a
			// human in a browser; the user hits Esc while it waits.
			agentRef.Abort()
			<-ctx.Done()
			return ToolResult{}, ctx.Err()
		},
	}

	calls := []llm.ToolCall{
		toolCall("c1", "blocking", nil),
		toolCall("c2", "blocking", nil),
		toolCall("c3", "blocking", nil),
	}
	streamFn, _ := streamOf(assistantToolCalls(calls...), assistantText("after"))
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{blocking}},
	})
	agentRef = agent

	_ = agent.Prompt("run all three")

	answered := map[string]bool{}
	for _, message := range agent.State().Messages {
		result, ok := message.(llm.ToolResultMessage)
		if !ok {
			continue
		}
		answered[result.ToolCallID] = true
	}
	var missing []string
	for _, call := range calls {
		if !answered[call.ID] {
			missing = append(missing, call.ID)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("tool calls left unanswered after an abort: %s\nthe next provider request would be rejected",
			strings.Join(missing, ", "))
	}
}

// TestAbortMidParallelBatchStillAnswersEveryToolCall is the same invariant for
// the parallel executor's preflight loop.
func TestAbortMidParallelBatchStillAnswersEveryToolCall(t *testing.T) {
	var agentRef *Agent
	var once sync.Once
	tool := Tool{
		Name:       "parallel",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			once.Do(func() { agentRef.Abort() })
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "ok"}}}, nil
		},
	}

	calls := []llm.ToolCall{
		toolCall("p1", "parallel", nil),
		toolCall("p2", "parallel", nil),
		toolCall("p3", "parallel", nil),
	}
	streamFn, _ := streamOf(assistantToolCalls(calls...), assistantText("after"))
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{tool}},
	})
	agentRef = agent

	_ = agent.Prompt("run all three")

	answered := map[string]bool{}
	for _, message := range agent.State().Messages {
		if result, ok := message.(llm.ToolResultMessage); ok {
			answered[result.ToolCallID] = true
		}
	}
	for _, call := range calls {
		if !answered[call.ID] {
			t.Fatalf("tool call %s was left unanswered after an abort", call.ID)
		}
	}
}
