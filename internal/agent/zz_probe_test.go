package agent

import (
	"context"
	"encoding/json"
	"testing"

	"goshcoder/internal/llm"
)

func TestProbeAbortPhantomTurn(t *testing.T) {
	var agentRef *Agent
	blocking := Tool{
		Name:          "blocking",
		Parameters:    json.RawMessage(`{"type":"object"}`),
		ExecutionMode: ToolExecutionSequential,
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			agentRef.Abort()
			<-ctx.Done()
			return ToolResult{}, ctx.Err()
		},
	}
	calls := []llm.ToolCall{toolCall("c1", "blocking", nil)}
	streamFn, count := streamOf(assistantToolCalls(calls...), assistantText("PHANTOM-AFTER-ABORT"))
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{blocking}},
	})
	agentRef = agent
	_ = agent.Prompt("go")

	t.Logf("streamFn calls = %d", *count)
	t.Logf("turn_start events = %d", len(rec.ofType(EventTurnStart)))
	for i, m := range agent.State().Messages {
		if am, ok := m.(llm.AssistantMessage); ok {
			txt := ""
			for _, b := range am.Content {
				if tb, ok := b.(llm.TextContent); ok {
					txt += tb.Text
				}
			}
			t.Logf("msg %d assistant stop=%v text=%q err=%q", i, am.StopReason, txt, am.ErrorMessage)
		} else {
			t.Logf("msg %d %T", i, m)
		}
	}
}
