package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"goshcoder/internal/llm"
)

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

func testModel() llm.Model {
	return llm.Model{
		ID:       "test-model",
		Name:     "Test Model",
		API:      "test-api",
		Provider: "test-provider",
	}
}

// assistantText builds a completed assistant message with text content.
func assistantText(text string) llm.AssistantMessage {
	model := testModel()
	return llm.AssistantMessage{
		Role:       "assistant",
		Content:    []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}},
		API:        model.API,
		Provider:   model.Provider,
		Model:      model.ID,
		StopReason: llm.StopStop,
		Timestamp:  time.Now().UnixMilli(),
	}
}

// assistantToolCalls builds an assistant message requesting tool calls.
func assistantToolCalls(calls ...llm.ToolCall) llm.AssistantMessage {
	model := testModel()
	content := make([]llm.ContentBlock, 0, len(calls))
	for _, call := range calls {
		content = append(content, call)
	}
	return llm.AssistantMessage{
		Role:       "assistant",
		Content:    content,
		API:        model.API,
		Provider:   model.Provider,
		Model:      model.ID,
		StopReason: llm.StopToolUse,
		Timestamp:  time.Now().UnixMilli(),
	}
}

func toolCall(id, name string, args map[string]any) llm.ToolCall {
	if args == nil {
		args = map[string]any{}
	}
	return llm.ToolCall{Type: "toolCall", ID: id, Name: name, Arguments: args}
}

// streamOf returns a StreamFn that replays the given messages, one per turn.
// After the list is exhausted it keeps returning the last message, which would
// loop forever, so tests must arrange for termination.
func streamOf(messages ...llm.AssistantMessage) (StreamFn, *int) {
	var mu sync.Mutex
	calls := 0
	fn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		mu.Lock()
		index := calls
		calls++
		mu.Unlock()
		if index >= len(messages) {
			index = len(messages) - 1
		}
		return completedStream(messages[index])
	}
	return fn, &calls
}

// completedStream emits a start event, then a terminal done/error event.
func completedStream(message llm.AssistantMessage) *llm.AssistantMessageEventStream {
	stream := llm.NewAssistantMessageEventStream()
	partial := message
	partial.Content = []llm.ContentBlock{}
	stream.Push(llm.AssistantMessageEvent{Type: llm.EventStart, Partial: &partial})

	final := message
	if message.StopReason == llm.StopError || message.StopReason == llm.StopAborted {
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventError, Reason: message.StopReason, Error: &final})
	} else {
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventDone, Reason: message.StopReason, Message: &final})
	}
	stream.End()
	return stream
}

// recorder collects agent events for assertions.
type recorder struct {
	mu     sync.Mutex
	events []Event
}

func (r *recorder) listener(ctx context.Context, ev Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, ev)
}

func (r *recorder) types() []EventType {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]EventType, len(r.events))
	for i, ev := range r.events {
		out[i] = ev.Type
	}
	return out
}

func (r *recorder) ofType(t EventType) []Event {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []Event
	for _, ev := range r.events {
		if ev.Type == t {
			out = append(out, ev)
		}
	}
	return out
}

// newTestAgent builds an agent with a recorder attached.
func newTestAgent(t *testing.T, opts AgentOptions) (*Agent, *recorder) {
	t.Helper()
	if opts.InitialState == nil {
		opts.InitialState = &InitialState{Model: testModel()}
	} else if opts.InitialState.Model.ID == "" {
		opts.InitialState.Model = testModel()
	}
	agent := NewAgent(opts)
	rec := &recorder{}
	agent.Subscribe(rec.listener)
	return agent, rec
}

// echoTool returns a tool that reports what it was called with.
func echoTool(name string) Tool {
	return Tool{
		Name:        name,
		Description: "echo",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}}}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			value, _ := params["value"].(string)
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "echo:" + value}}}, nil
		},
	}
}

// ---------------------------------------------------------------------------
// Basic lifecycle
// ---------------------------------------------------------------------------

func TestAgentPromptSingleTurn(t *testing.T) {
	streamFn, calls := streamOf(assistantText("hello"))
	agent, rec := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	want := []EventType{
		EventAgentStart, EventTurnStart,
		// the user prompt
		EventMessageStart, EventMessageEnd,
		// the assistant response
		EventMessageStart, EventMessageEnd,
		EventTurnEnd, EventAgentEnd,
	}
	if got := rec.types(); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v,\nwant %v", got, want)
	}
	if *calls != 1 {
		t.Fatalf("stream calls = %d, want 1", *calls)
	}

	state := agent.State()
	if len(state.Messages) != 2 {
		t.Fatalf("messages = %#v", state.Messages)
	}
	if RoleOf(state.Messages[0]) != "user" || RoleOf(state.Messages[1]) != "assistant" {
		t.Fatalf("roles = %q, %q", RoleOf(state.Messages[0]), RoleOf(state.Messages[1]))
	}
	// The run is settled by the time Prompt returns.
	if state.IsStreaming {
		t.Fatal("agent should be idle after Prompt returns")
	}
	if state.StreamingMessage != nil {
		t.Fatalf("streamingMessage = %#v, want nil", state.StreamingMessage)
	}
}

func TestAgentPromptNormalizesInput(t *testing.T) {
	t.Run("string becomes a user message", func(t *testing.T) {
		streamFn, _ := streamOf(assistantText("ok"))
		agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})
		if err := agent.Prompt("hi"); err != nil {
			t.Fatal(err)
		}
		user := agent.State().Messages[0].(llm.UserMessage)
		blocks := user.BlockContent()
		if text := blocks[0].(llm.TextContent); text.Text != "hi" {
			t.Fatalf("content = %#v", blocks)
		}
	})

	t.Run("images are appended to the text block", func(t *testing.T) {
		streamFn, _ := streamOf(assistantText("ok"))
		agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})
		image := llm.ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"}
		if err := agent.Prompt("look", image); err != nil {
			t.Fatal(err)
		}
		user := agent.State().Messages[0].(llm.UserMessage)
		blocks := user.BlockContent()
		if len(blocks) != 2 {
			t.Fatalf("blocks = %#v", blocks)
		}
		if _, ok := blocks[1].(llm.ImageContent); !ok {
			t.Fatalf("second block = %#v, want an image", blocks[1])
		}
	})

	t.Run("a message batch is used as-is", func(t *testing.T) {
		streamFn, _ := streamOf(assistantText("ok"))
		agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})
		batch := []Message{
			llm.UserMessage{Role: "user", Content: "one"},
			llm.UserMessage{Role: "user", Content: "two"},
		}
		if err := agent.Prompt(batch); err != nil {
			t.Fatal(err)
		}
		// Both prompts, plus the assistant reply.
		if got := len(agent.State().Messages); got != 3 {
			t.Fatalf("messages = %d, want 3", got)
		}
	})
}

func TestAgentBusyReturnsError(t *testing.T) {
	release := make(chan struct{})
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		<-release
		return completedStream(assistantText("done"))
	}
	agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	go func() { _ = agent.Prompt("first") }()

	// Wait until the run is registered.
	deadline := time.Now().Add(2 * time.Second)
	for !agent.State().IsStreaming {
		if time.Now().After(deadline) {
			close(release)
			t.Fatal("run never started")
		}
		time.Sleep(time.Millisecond)
	}

	if err := agent.Prompt("second"); err == nil {
		close(release)
		t.Fatal("expected a busy error")
	} else if !strings.Contains(err.Error(), "already processing") {
		close(release)
		t.Fatalf("err = %v", err)
	}

	close(release)
	agent.WaitForIdle()
}

func TestAgentResetClearsState(t *testing.T) {
	streamFn, _ := streamOf(assistantText("hello"))
	agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})
	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	agent.Steer(llm.UserMessage{Role: "user", Content: "queued"})

	if err := agent.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	state := agent.State()
	if len(state.Messages) != 0 {
		t.Fatalf("messages = %#v, want empty", state.Messages)
	}
	if agent.HasQueuedMessages() {
		t.Fatal("queues should be cleared")
	}
}

// ---------------------------------------------------------------------------
// Continue
// ---------------------------------------------------------------------------

func TestAgentContinue(t *testing.T) {
	t.Run("continues from a user message", func(t *testing.T) {
		streamFn, calls := streamOf(assistantText("continued"))
		agent, rec := newTestAgent(t, AgentOptions{
			StreamFn: streamFn,
			InitialState: &InitialState{
				Model:    testModel(),
				Messages: []Message{llm.UserMessage{Role: "user", Content: "earlier"}},
			},
		})

		if err := agent.Continue(); err != nil {
			t.Fatalf("Continue: %v", err)
		}
		if *calls != 1 {
			t.Fatalf("stream calls = %d", *calls)
		}
		// No new prompt message is emitted, unlike Prompt.
		want := []EventType{EventAgentStart, EventTurnStart, EventMessageStart, EventMessageEnd, EventTurnEnd, EventAgentEnd}
		if got := rec.types(); fmt.Sprint(got) != fmt.Sprint(want) {
			t.Fatalf("events = %v,\nwant %v", got, want)
		}
	})

	t.Run("empty transcript errors", func(t *testing.T) {
		agent, _ := newTestAgent(t, AgentOptions{})
		if err := agent.Continue(); err == nil || !strings.Contains(err.Error(), "no messages") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("assistant tail without queued messages errors", func(t *testing.T) {
		agent, _ := newTestAgent(t, AgentOptions{
			InitialState: &InitialState{
				Model:    testModel(),
				Messages: []Message{assistantText("done")},
			},
		})
		if err := agent.Continue(); err == nil || !strings.Contains(err.Error(), "assistant") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("assistant tail uses queued steering messages", func(t *testing.T) {
		streamFn, _ := streamOf(assistantText("after steering"))
		agent, _ := newTestAgent(t, AgentOptions{
			StreamFn: streamFn,
			InitialState: &InitialState{
				Model:    testModel(),
				Messages: []Message{assistantText("done")},
			},
		})
		agent.Steer(llm.UserMessage{Role: "user", Content: "steered"})

		if err := agent.Continue(); err != nil {
			t.Fatalf("Continue: %v", err)
		}
		messages := agent.State().Messages
		// The steering message became the prompt for this run.
		user, ok := messages[1].(llm.UserMessage)
		if !ok {
			t.Fatalf("messages[1] = %#v", messages[1])
		}
		if text, _ := user.StringContent(); text != "steered" {
			t.Fatalf("steering message = %#v", user)
		}
	})
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

func TestAgentExecutesToolCall(t *testing.T) {
	streamFn, calls := streamOf(
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"})),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{echoTool("echo")}},
	})

	if err := agent.Prompt("run it"); err != nil {
		t.Fatal(err)
	}
	// Two turns: the tool call, then the follow-up response.
	if *calls != 2 {
		t.Fatalf("stream calls = %d, want 2", *calls)
	}

	starts := rec.ofType(EventToolExecutionStart)
	ends := rec.ofType(EventToolExecutionEnd)
	if len(starts) != 1 || len(ends) != 1 {
		t.Fatalf("tool events: %d starts, %d ends", len(starts), len(ends))
	}
	if starts[0].ToolCallID != "c1" || starts[0].ToolName != "echo" {
		t.Fatalf("start = %#v", starts[0])
	}
	if ends[0].IsError {
		t.Fatalf("tool reported an error: %#v", ends[0].Result)
	}
	text := ends[0].Result.Content[0].(llm.TextContent)
	if text.Text != "echo:x" {
		t.Fatalf("result = %#v", text)
	}

	// The transcript carries the tool result between the two assistant turns.
	messages := agent.State().Messages
	if len(messages) != 4 {
		t.Fatalf("messages = %d: %#v", len(messages), messages)
	}
	result, ok := messages[2].(llm.ToolResultMessage)
	if !ok {
		t.Fatalf("messages[2] = %#v", messages[2])
	}
	if result.ToolCallID != "c1" || result.IsError {
		t.Fatalf("tool result = %#v", result)
	}
	// Pending tool calls are cleared once the run settles.
	if got := agent.State().PendingToolCalls; len(got) != 0 {
		t.Fatalf("pendingToolCalls = %#v", got)
	}
}

func TestAgentUnknownToolIsAnError(t *testing.T) {
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "missing", nil)),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 || !ends[0].IsError {
		t.Fatalf("tool end = %#v", ends)
	}
	text := ends[0].Result.Content[0].(llm.TextContent)
	if !strings.Contains(text.Text, "not found") {
		t.Fatalf("result text = %q", text.Text)
	}
}

func TestAgentToolErrorBecomesErrorResult(t *testing.T) {
	failing := Tool{
		Name:       "boom",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			return ToolResult{}, fmt.Errorf("tool exploded")
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "boom", nil)),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{failing}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 || !ends[0].IsError {
		t.Fatalf("tool end = %#v", ends)
	}
	if text := ends[0].Result.Content[0].(llm.TextContent); text.Text != "tool exploded" {
		t.Fatalf("result text = %q", text.Text)
	}
}

// TestAgentToolPanicBecomesErrorResult confirms a panicking tool cannot take
// the whole run down.
func TestAgentToolPanicBecomesErrorResult(t *testing.T) {
	panicking := Tool{
		Name:       "panic",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			panic("tool panicked")
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "panic", nil)),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{panicking}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 || !ends[0].IsError {
		t.Fatalf("tool end = %#v", ends)
	}
	if text := ends[0].Result.Content[0].(llm.TextContent); !strings.Contains(text.Text, "tool panicked") {
		t.Fatalf("result text = %q", text.Text)
	}
}

func TestAgentToolValidationFailure(t *testing.T) {
	strict := Tool{
		Name:       "strict",
		Parameters: json.RawMessage(`{"type":"object","properties":{"n":{"type":"array"}},"required":["n"]}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			t.Error("Execute should not run when validation fails")
			return ToolResult{}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "strict", map[string]any{})),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{strict}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 || !ends[0].IsError {
		t.Fatalf("tool end = %#v", ends)
	}
	if text := ends[0].Result.Content[0].(llm.TextContent); !strings.Contains(text.Text, "Validation failed") {
		t.Fatalf("result text = %q", text.Text)
	}
}

func TestAgentToolUpdatesEmitEvents(t *testing.T) {
	streaming := Tool{
		Name:       "streamer",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			onUpdate(ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "partial"}}})
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "final"}}}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "streamer", nil)),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{streaming}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	updates := rec.ofType(EventToolExecutionUpdate)
	if len(updates) != 1 {
		t.Fatalf("updates = %#v", updates)
	}
	if text := updates[0].PartialResult.Content[0].(llm.TextContent); text.Text != "partial" {
		t.Fatalf("partial = %#v", text)
	}
}

// TestAgentToolUpdatesAfterReturnAreIgnored covers the late-callback guard.
func TestAgentToolUpdatesAfterReturnAreIgnored(t *testing.T) {
	var leaked func(ToolResult)
	leaking := Tool{
		Name:       "leaky",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			leaked = onUpdate
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "final"}}}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "leaky", nil)),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{leaking}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	before := len(rec.ofType(EventToolExecutionUpdate))
	leaked(ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "too late"}}})
	if after := len(rec.ofType(EventToolExecutionUpdate)); after != before {
		t.Fatalf("late update was emitted: %d -> %d", before, after)
	}
}

func TestAgentParallelToolExecution(t *testing.T) {
	var mu sync.Mutex
	concurrent, maxConcurrent := 0, 0
	blocker := make(chan struct{})

	tool := Tool{
		Name:       "slow",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			mu.Lock()
			concurrent++
			if concurrent > maxConcurrent {
				maxConcurrent = concurrent
			}
			reached := concurrent
			mu.Unlock()

			// Release once both calls are in flight.
			if reached == 2 {
				close(blocker)
			}
			<-blocker

			mu.Lock()
			concurrent--
			mu.Unlock()
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: id}}}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "slow", nil), toolCall("c2", "slow", nil)),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:      streamFn,
		ToolExecution: ToolExecutionParallel,
		InitialState:  &InitialState{Model: testModel(), Tools: []Tool{tool}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	peak := maxConcurrent
	mu.Unlock()
	if peak != 2 {
		t.Fatalf("peak concurrency = %d, want 2", peak)
	}

	// Tool results are still ordered by the assistant's call order.
	messages := agent.State().Messages
	var ids []string
	for _, m := range messages {
		if result, ok := m.(llm.ToolResultMessage); ok {
			ids = append(ids, result.ToolCallID)
		}
	}
	if fmt.Sprint(ids) != fmt.Sprint([]string{"c1", "c2"}) {
		t.Fatalf("tool result order = %v", ids)
	}
}

func TestAgentSequentialToolExecution(t *testing.T) {
	var mu sync.Mutex
	var order []string

	tool := Tool{
		Name:       "seq",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			mu.Lock()
			order = append(order, "start:"+id)
			mu.Unlock()
			time.Sleep(5 * time.Millisecond)
			mu.Lock()
			order = append(order, "end:"+id)
			mu.Unlock()
			return ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: id}}}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "seq", nil), toolCall("c2", "seq", nil)),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:      streamFn,
		ToolExecution: ToolExecutionSequential,
		InitialState:  &InitialState{Model: testModel(), Tools: []Tool{tool}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	got := fmt.Sprint(order)
	mu.Unlock()
	// Each call finishes before the next starts.
	want := fmt.Sprint([]string{"start:c1", "end:c1", "start:c2", "end:c2"})
	if got != want {
		t.Fatalf("order = %v,\nwant %v", got, want)
	}
}

// TestAgentSequentialToolForcesWholeBatch covers a per-tool override making an
// otherwise parallel batch run sequentially.
func TestAgentSequentialToolForcesWholeBatch(t *testing.T) {
	var mu sync.Mutex
	concurrent, maxConcurrent := 0, 0

	track := func(name string, mode ToolExecutionMode) Tool {
		return Tool{
			Name:          name,
			Parameters:    json.RawMessage(`{"type":"object"}`),
			ExecutionMode: mode,
			Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
				mu.Lock()
				concurrent++
				if concurrent > maxConcurrent {
					maxConcurrent = concurrent
				}
				mu.Unlock()
				time.Sleep(5 * time.Millisecond)
				mu.Lock()
				concurrent--
				mu.Unlock()
				return ToolResult{}, nil
			},
		}
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "parallel", nil), toolCall("c2", "serial", nil)),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:      streamFn,
		ToolExecution: ToolExecutionParallel,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{
			track("parallel", ""),
			track("serial", ToolExecutionSequential),
		}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	peak := maxConcurrent
	mu.Unlock()
	if peak != 1 {
		t.Fatalf("peak concurrency = %d, want 1 (a sequential tool forces the batch)", peak)
	}
}

// TestAgentTruncatedMessageFailsToolCalls covers a "length" stop: arguments may
// be silently incomplete, so no call is executed.
func TestAgentTruncatedMessageFailsToolCalls(t *testing.T) {
	truncated := assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"}))
	truncated.StopReason = llm.StopLength

	executed := false
	tool := echoTool("echo")
	tool.Execute = func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
		executed = true
		return ToolResult{}, nil
	}

	streamFn, _ := streamOf(truncated, assistantText("done"))
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{tool}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("a truncated tool call must not execute")
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 || !ends[0].IsError {
		t.Fatalf("tool end = %#v", ends)
	}
	if text := ends[0].Result.Content[0].(llm.TextContent); !strings.Contains(text.Text, "output token limit") {
		t.Fatalf("result text = %q", text.Text)
	}
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

func TestAgentBeforeToolCallBlocks(t *testing.T) {
	executed := false
	tool := echoTool("echo")
	tool.Execute = func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
		executed = true
		return ToolResult{}, nil
	}

	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"})),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{tool}},
		BeforeToolCall: func(ctx context.Context, c BeforeToolCallContext) *BeforeToolCallResult {
			return &BeforeToolCallResult{Block: true, Reason: "not allowed"}
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if executed {
		t.Fatal("blocked tool must not execute")
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 || !ends[0].IsError {
		t.Fatalf("tool end = %#v", ends)
	}
	if text := ends[0].Result.Content[0].(llm.TextContent); text.Text != "not allowed" {
		t.Fatalf("block reason = %q", text.Text)
	}
}

func TestAgentBeforeToolCallReceivesValidatedArgs(t *testing.T) {
	var seen map[string]any
	tool := Tool{
		Name:       "coerce",
		Parameters: json.RawMessage(`{"type":"object","properties":{"n":{"type":"number"}}}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			return ToolResult{}, nil
		},
	}
	streamFn, _ := streamOf(
		// The model sent a string where the schema wants a number.
		assistantToolCalls(toolCall("c1", "coerce", map[string]any{"n": "5"})),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{tool}},
		BeforeToolCall: func(ctx context.Context, c BeforeToolCallContext) *BeforeToolCallResult {
			seen = c.Args
			return nil
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if seen["n"] != 5.0 {
		t.Fatalf("hook args = %#v, want the coerced number", seen)
	}
}

func TestAgentAfterToolCallOverrides(t *testing.T) {
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"})),
		assistantText("done"),
	)
	isError := true
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{echoTool("echo")}},
		AfterToolCall: func(ctx context.Context, c AfterToolCallContext) *AfterToolCallResult {
			return &AfterToolCallResult{
				Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "overridden"}},
				IsError: &isError,
			}
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	ends := rec.ofType(EventToolExecutionEnd)
	if len(ends) != 1 {
		t.Fatalf("tool end = %#v", ends)
	}
	if !ends[0].IsError {
		t.Fatal("isError override was not applied")
	}
	if text := ends[0].Result.Content[0].(llm.TextContent); text.Text != "overridden" {
		t.Fatalf("content override = %q", text.Text)
	}
}

// TestAgentAfterToolCallPartialOverride confirms nil fields keep the executed
// values rather than clearing them.
func TestAgentAfterToolCallPartialOverride(t *testing.T) {
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "keep"})),
		assistantText("done"),
	)
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{echoTool("echo")}},
		AfterToolCall: func(ctx context.Context, c AfterToolCallContext) *AfterToolCallResult {
			// Only Details is set; Content must survive.
			return &AfterToolCallResult{Details: map[string]any{"tagged": true}}
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	end := rec.ofType(EventToolExecutionEnd)[0]
	if text := end.Result.Content[0].(llm.TextContent); text.Text != "echo:keep" {
		t.Fatalf("content = %q, want the original", text.Text)
	}
	details := end.Result.Details.(map[string]any)
	if details["tagged"] != true {
		t.Fatalf("details = %#v", details)
	}
}

func TestAgentShouldStopAfterTurn(t *testing.T) {
	// Without the hook this would loop forever on tool calls.
	streamFn, calls := streamOf(assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"})))
	agent, rec := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{echoTool("echo")}},
		ShouldStopAfterTurn: func(ctx context.Context, c ShouldStopAfterTurnContext) bool {
			return true
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if *calls != 1 {
		t.Fatalf("stream calls = %d, want 1", *calls)
	}
	if got := len(rec.ofType(EventAgentEnd)); got != 1 {
		t.Fatalf("agent_end events = %d", got)
	}
}

func TestAgentShouldStopAfterTurnContext(t *testing.T) {
	var seen ShouldStopAfterTurnContext
	streamFn, _ := streamOf(assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"})))
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{echoTool("echo")}},
		ShouldStopAfterTurn: func(ctx context.Context, c ShouldStopAfterTurnContext) bool {
			seen = c
			return true
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if len(seen.ToolResults) != 1 || seen.ToolResults[0].ToolCallID != "c1" {
		t.Fatalf("hook toolResults = %#v", seen.ToolResults)
	}
	// The context includes the assistant message and the tool result.
	if len(seen.NewMessages) != 3 {
		t.Fatalf("hook newMessages = %d: %#v", len(seen.NewMessages), seen.NewMessages)
	}
}

func TestAgentPrepareNextTurnReplacesModel(t *testing.T) {
	var models []string
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		models = append(models, model.ID)
		if len(models) == 1 {
			return completedStream(assistantToolCalls(toolCall("c1", "echo", map[string]any{"value": "x"})))
		}
		return completedStream(assistantText("done"))
	}
	replacement := testModel()
	replacement.ID = "swapped-model"

	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{echoTool("echo")}},
		PrepareNextTurn: func(ctx context.Context, c PrepareNextTurnContext) *TurnUpdate {
			return &TurnUpdate{Model: &replacement}
		},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 {
		t.Fatalf("stream calls = %#v", models)
	}
	if models[0] != "test-model" || models[1] != "swapped-model" {
		t.Fatalf("models = %v, want the second turn to use the replacement", models)
	}
}

func TestAgentTransformContextAndConvertToLLM(t *testing.T) {
	var sawMessages int
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		sawMessages = len(ctx.Messages)
		return completedStream(assistantText("ok"))
	}
	transformCalled := false
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn: streamFn,
		TransformContext: func(ctx context.Context, messages []Message) []Message {
			transformCalled = true
			// Drop everything: the LLM should see an empty transcript.
			return nil
		},
	})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if !transformCalled {
		t.Fatal("TransformContext was not called")
	}
	if sawMessages != 0 {
		t.Fatalf("llm context messages = %d, want 0", sawMessages)
	}
}

// customNotification is a custom transcript message type.
type customNotification struct{ text string }

func (customNotification) MessageRole() string { return "notification" }

func TestDefaultConvertToLLMFiltersCustomMessages(t *testing.T) {
	messages := []Message{
		llm.UserMessage{Role: "user", Content: "hi"},
		customNotification{text: "ignored"},
		assistantText("reply"),
		llm.ToolResultMessage{Role: "toolResult", ToolCallID: "c1", ToolName: "t"},
	}
	got := DefaultConvertToLLM(messages)
	// The custom message is dropped; the other three survive.
	if len(got) != 3 {
		t.Fatalf("converted = %d: %#v", len(got), got)
	}
}

func TestRoleOf(t *testing.T) {
	tests := []struct {
		message Message
		want    string
	}{
		{llm.UserMessage{Role: "user"}, "user"},
		{&llm.UserMessage{Role: "user"}, "user"},
		{llm.AssistantMessage{Role: "assistant"}, "assistant"},
		{&llm.AssistantMessage{Role: "assistant"}, "assistant"},
		{llm.ToolResultMessage{Role: "toolResult"}, "toolResult"},
		{&llm.ToolResultMessage{Role: "toolResult"}, "toolResult"},
		{customNotification{}, "notification"},
		{"not a message", ""},
	}
	for _, tt := range tests {
		if got := RoleOf(tt.message); got != tt.want {
			t.Errorf("RoleOf(%T) = %q, want %q", tt.message, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// Queues
// ---------------------------------------------------------------------------

func TestAgentSteeringMessageInjectedNextTurn(t *testing.T) {
	var agent *Agent
	turn := 0
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		turn++
		if turn == 1 {
			// Queue a steering message mid-run, as a user typing would.
			agent.Steer(llm.UserMessage{Role: "user", Content: "steered"})
			return completedStream(assistantText("first"))
		}
		return completedStream(assistantText("second"))
	}
	agent, _ = newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if turn != 2 {
		t.Fatalf("turns = %d, want 2 (the steering message drives another turn)", turn)
	}
	messages := agent.State().Messages
	// prompt, first reply, steering message, second reply
	if len(messages) != 4 {
		t.Fatalf("messages = %d: %#v", len(messages), messages)
	}
	steered := messages[2].(llm.UserMessage)
	if text, _ := steered.StringContent(); text != "steered" {
		t.Fatalf("messages[2] = %#v", steered)
	}
}

func TestAgentFollowUpMessageStartsAnotherTurn(t *testing.T) {
	var agent *Agent
	turn := 0
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		turn++
		if turn == 1 {
			agent.FollowUp(llm.UserMessage{Role: "user", Content: "follow up"})
		}
		return completedStream(assistantText(fmt.Sprintf("reply %d", turn)))
	}
	agent, _ = newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if turn != 2 {
		t.Fatalf("turns = %d, want 2", turn)
	}
	if agent.HasQueuedMessages() {
		t.Fatal("the follow-up queue should be drained")
	}
}

func TestAgentQueueModes(t *testing.T) {
	t.Run("one-at-a-time drains a single message", func(t *testing.T) {
		queue := &pendingMessageQueue{mode: QueueOneAtATime}
		queue.enqueue(llm.UserMessage{Role: "user", Content: "a"})
		queue.enqueue(llm.UserMessage{Role: "user", Content: "b"})

		drained := queue.drain()
		if len(drained) != 1 {
			t.Fatalf("drained = %#v", drained)
		}
		if !queue.hasItems() {
			t.Fatal("the second message should remain queued")
		}
	})

	t.Run("all drains everything", func(t *testing.T) {
		queue := &pendingMessageQueue{mode: QueueAll}
		queue.enqueue(llm.UserMessage{Role: "user", Content: "a"})
		queue.enqueue(llm.UserMessage{Role: "user", Content: "b"})

		drained := queue.drain()
		if len(drained) != 2 {
			t.Fatalf("drained = %#v", drained)
		}
		if queue.hasItems() {
			t.Fatal("the queue should be empty")
		}
	})
}

func TestAgentClearQueues(t *testing.T) {
	agent, _ := newTestAgent(t, AgentOptions{})
	agent.Steer(llm.UserMessage{Role: "user", Content: "s"})
	agent.FollowUp(llm.UserMessage{Role: "user", Content: "f"})
	if !agent.HasQueuedMessages() {
		t.Fatal("expected queued messages")
	}

	agent.ClearSteeringQueue()
	if !agent.HasQueuedMessages() {
		t.Fatal("the follow-up should still be queued")
	}
	agent.ClearFollowUpQueue()
	if agent.HasQueuedMessages() {
		t.Fatal("all queues should be empty")
	}

	agent.Steer(llm.UserMessage{Role: "user", Content: "s"})
	agent.FollowUp(llm.UserMessage{Role: "user", Content: "f"})
	agent.ClearAllQueues()
	if agent.HasQueuedMessages() {
		t.Fatal("ClearAllQueues should empty both queues")
	}
}

// ---------------------------------------------------------------------------
// Errors and abort
// ---------------------------------------------------------------------------

func TestAgentErrorStopEndsRun(t *testing.T) {
	failed := assistantText("")
	failed.StopReason = llm.StopError
	failed.ErrorMessage = "provider blew up"

	streamFn, calls := streamOf(failed)
	agent, rec := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatalf("Prompt should not return provider failures: %v", err)
	}
	// The loop stops instead of retrying.
	if *calls != 1 {
		t.Fatalf("stream calls = %d, want 1", *calls)
	}
	if got := len(rec.ofType(EventAgentEnd)); got != 1 {
		t.Fatalf("agent_end events = %d", got)
	}
	// The failure surfaces through state rather than a returned error.
	if got := agent.State().ErrorMessage; got != "provider blew up" {
		t.Fatalf("errorMessage = %q", got)
	}
}

func TestAgentAbortCancelsRun(t *testing.T) {
	started := make(chan struct{})
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		close(started)
		// Wait for the abort, then report it the way providers do.
		<-opts.Ctx.Done()
		aborted := assistantText("")
		aborted.StopReason = llm.StopAborted
		aborted.ErrorMessage = "Request was aborted"
		return completedStream(aborted)
	}
	agent, rec := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = agent.Prompt("hi")
	}()

	<-started
	agent.Abort()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the run did not settle after Abort")
	}

	if got := len(rec.ofType(EventAgentEnd)); got != 1 {
		t.Fatalf("agent_end events = %d", got)
	}
	if agent.State().IsStreaming {
		t.Fatal("the agent should be idle after an abort")
	}
}

func TestAgentRunContextClearedWhenIdle(t *testing.T) {
	streamFn, _ := streamOf(assistantText("ok"))
	agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if got := agent.RunContext(); got != nil {
		t.Fatalf("RunContext = %#v, want nil while idle", got)
	}
	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if got := agent.RunContext(); got != nil {
		t.Fatalf("RunContext = %#v, want nil after the run", got)
	}
}

// TestAgentDefaultStreamFnUnknownAPI covers the no-streamer fallback: the run
// ends through the normal error path rather than panicking.
func TestAgentDefaultStreamFnUnknownAPI(t *testing.T) {
	agent, rec := newTestAgent(t, AgentOptions{
		InitialState: &InitialState{Model: llm.Model{ID: "x", API: "not-a-real-api", Provider: "p"}},
	})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := len(rec.ofType(EventAgentEnd)); got != 1 {
		t.Fatalf("agent_end events = %d", got)
	}
	if got := agent.State().ErrorMessage; !strings.Contains(got, "no streamer registered") {
		t.Fatalf("errorMessage = %q", got)
	}
}

// ---------------------------------------------------------------------------
// State and subscriptions
// ---------------------------------------------------------------------------

func TestAgentSetters(t *testing.T) {
	agent, _ := newTestAgent(t, AgentOptions{})

	agent.SetSystemPrompt("be brief")
	replacement := testModel()
	replacement.ID = "other"
	agent.SetModel(replacement)
	agent.SetThinkingLevel(llm.ThinkingHigh)
	agent.SetTools([]Tool{echoTool("echo")})
	agent.SetMessages([]Message{llm.UserMessage{Role: "user", Content: "seeded"}})

	state := agent.State()
	if state.SystemPrompt != "be brief" {
		t.Fatalf("systemPrompt = %q", state.SystemPrompt)
	}
	if state.Model.ID != "other" {
		t.Fatalf("model = %#v", state.Model)
	}
	if state.ThinkingLevel != llm.ThinkingHigh {
		t.Fatalf("thinkingLevel = %q", state.ThinkingLevel)
	}
	if len(state.Tools) != 1 || state.Tools[0].Name != "echo" {
		t.Fatalf("tools = %#v", state.Tools)
	}
	if len(state.Messages) != 1 {
		t.Fatalf("messages = %#v", state.Messages)
	}
}

// TestAgentStateReturnsCopies confirms mutating a snapshot cannot corrupt the
// agent's own state.
func TestAgentStateReturnsCopies(t *testing.T) {
	agent, _ := newTestAgent(t, AgentOptions{
		InitialState: &InitialState{
			Model:    testModel(),
			Tools:    []Tool{echoTool("echo")},
			Messages: []Message{llm.UserMessage{Role: "user", Content: "one"}},
		},
	})

	snapshot := agent.State()
	snapshot.Messages[0] = llm.UserMessage{Role: "user", Content: "tampered"}
	snapshot.Tools[0].Name = "tampered"

	fresh := agent.State()
	user := fresh.Messages[0].(llm.UserMessage)
	if text, _ := user.StringContent(); text != "one" {
		t.Fatalf("messages were mutated through the snapshot: %#v", user)
	}
	if fresh.Tools[0].Name != "echo" {
		t.Fatalf("tools were mutated through the snapshot: %#v", fresh.Tools)
	}
}

func TestAgentSubscribeUnsubscribe(t *testing.T) {
	streamFn, _ := streamOf(assistantText("ok"))
	agent, _ := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	second := &recorder{}
	unsubscribe := agent.Subscribe(second.listener)
	unsubscribe()

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if got := len(second.types()); got != 0 {
		t.Fatalf("an unsubscribed listener received %d events", got)
	}
}

// TestAgentMessageUpdateEventsCarryPartials covers streaming: message_update
// events carry the progressively assembled assistant message.
func TestAgentMessageUpdateEventsCarryPartials(t *testing.T) {
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		stream := llm.NewAssistantMessageEventStream()
		final := assistantText("hi there")

		empty := final
		empty.Content = []llm.ContentBlock{}
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventStart, Partial: &empty})

		firstPartial := final
		firstPartial.Content = []llm.ContentBlock{llm.TextContent{Type: "text", Text: "hi"}}
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventTextDelta, ContentIndex: 0, Delta: "hi", Partial: &firstPartial})

		stream.Push(llm.AssistantMessageEvent{Type: llm.EventTextDelta, ContentIndex: 0, Delta: " there", Partial: &final})
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventDone, Reason: llm.StopStop, Message: &final})
		stream.End()
		return stream
	}
	agent, rec := newTestAgent(t, AgentOptions{StreamFn: streamFn})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	updates := rec.ofType(EventMessageUpdate)
	if len(updates) != 2 {
		t.Fatalf("message_update events = %d", len(updates))
	}
	if updates[0].AssistantEvent == nil || updates[0].AssistantEvent.Delta != "hi" {
		t.Fatalf("first update = %#v", updates[0].AssistantEvent)
	}
	// The final transcript holds the completed message.
	final := agent.State().Messages[1].(llm.AssistantMessage)
	if text := final.Content[0].(llm.TextContent); text.Text != "hi there" {
		t.Fatalf("final message = %#v", text)
	}
}

func TestAgentToolResultMessageNormalizesContent(t *testing.T) {
	// A tool returning no content must not put nil into the transcript.
	empty := Tool{
		Name:       "empty",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			return ToolResult{}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "empty", nil)),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{empty}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	for _, m := range agent.State().Messages {
		if result, ok := m.(llm.ToolResultMessage); ok {
			if result.Content == nil {
				t.Fatal("tool result content should be an empty slice, not nil")
			}
			return
		}
	}
	t.Fatal("no tool result message found")
}

func TestAgentTerminateStopsAfterToolBatch(t *testing.T) {
	terminating := Tool{
		Name:       "stop",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			return ToolResult{
				Content:   []llm.ContentBlock{llm.TextContent{Type: "text", Text: "stopping"}},
				Terminate: true,
			}, nil
		},
	}
	streamFn, calls := streamOf(assistantToolCalls(toolCall("c1", "stop", nil)))
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{terminating}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	// Terminate stops the loop instead of running another turn.
	if *calls != 1 {
		t.Fatalf("stream calls = %d, want 1", *calls)
	}
}

// TestAgentTerminateRequiresWholeBatch confirms one terminating tool among
// several does not stop the loop.
func TestAgentTerminateRequiresWholeBatch(t *testing.T) {
	terminating := Tool{
		Name:       "stop",
		Parameters: json.RawMessage(`{"type":"object"}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			return ToolResult{Terminate: true}, nil
		},
	}
	streamFn, calls := streamOf(
		assistantToolCalls(toolCall("c1", "stop", nil), toolCall("c2", "echo", map[string]any{"value": "x"})),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{terminating, echoTool("echo")}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if *calls != 2 {
		t.Fatalf("stream calls = %d, want 2 (only a fully terminating batch stops)", *calls)
	}
}

func TestAgentPrepareArgumentsShim(t *testing.T) {
	var seen map[string]any
	tool := Tool{
		Name:       "shimmed",
		Parameters: json.RawMessage(`{"type":"object","properties":{"canonical":{"type":"string"}},"required":["canonical"]}`),
		PrepareArguments: func(args map[string]any) map[string]any {
			// Rename a legacy key the model still emits.
			if legacy, ok := args["legacy"]; ok {
				return map[string]any{"canonical": legacy}
			}
			return args
		},
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(ToolResult)) (ToolResult, error) {
			seen = params
			return ToolResult{}, nil
		},
	}
	streamFn, _ := streamOf(
		assistantToolCalls(toolCall("c1", "shimmed", map[string]any{"legacy": "value"})),
		assistantText("done"),
	)
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:     streamFn,
		InitialState: &InitialState{Model: testModel(), Tools: []Tool{tool}},
	})

	if err := agent.Prompt("go"); err != nil {
		t.Fatal(err)
	}
	if seen["canonical"] != "value" {
		t.Fatalf("tool params = %#v, want the shimmed key", seen)
	}
}

func TestAgentGetAPIKeyIsCalledPerTurn(t *testing.T) {
	var providers []string
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		if opts.APIKey != "resolved-key" {
			t.Errorf("apiKey = %q, want the resolved key", opts.APIKey)
		}
		return completedStream(assistantText("ok"))
	}
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn: streamFn,
		GetAPIKey: func(provider string) string {
			providers = append(providers, provider)
			return "resolved-key"
		},
	})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if len(providers) != 1 || providers[0] != "test-provider" {
		t.Fatalf("GetAPIKey calls = %#v", providers)
	}
}

func TestAgentForwardsSessionAndThinking(t *testing.T) {
	var seen *llm.SimpleStreamOptions
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		seen = opts
		return completedStream(assistantText("ok"))
	}
	budgets := &llm.ThinkingBudgets{High: 4096}
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn:        streamFn,
		SessionID:       "sess-1",
		ThinkingBudgets: budgets,
		InitialState:    &InitialState{Model: testModel(), ThinkingLevel: llm.ThinkingHigh},
	})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if seen.SessionID != "sess-1" {
		t.Fatalf("sessionID = %q", seen.SessionID)
	}
	if seen.Reasoning != llm.ThinkingHigh {
		t.Fatalf("reasoning = %q", seen.Reasoning)
	}
	if seen.ThinkingBudgets != budgets {
		t.Fatalf("thinkingBudgets = %#v", seen.ThinkingBudgets)
	}
}

// TestAgentSystemPromptAndToolsReachLLM confirms the context handed to the
// stream function carries the prompt and tool schemas.
func TestAgentSystemPromptAndToolsReachLLM(t *testing.T) {
	var seen *llm.Context
	streamFn := func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		seen = ctx
		return completedStream(assistantText("ok"))
	}
	agent, _ := newTestAgent(t, AgentOptions{
		StreamFn: streamFn,
		InitialState: &InitialState{
			Model:        testModel(),
			SystemPrompt: "be brief",
			Tools:        []Tool{echoTool("echo")},
		},
	})

	if err := agent.Prompt("hi"); err != nil {
		t.Fatal(err)
	}
	if seen.SystemPrompt != "be brief" {
		t.Fatalf("systemPrompt = %q", seen.SystemPrompt)
	}
	if len(seen.Tools) != 1 || seen.Tools[0].Name != "echo" {
		t.Fatalf("tools = %#v", seen.Tools)
	}
}
