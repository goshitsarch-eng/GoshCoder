package main

import (
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/sessionlog"
)

// This file drives a whole turn through the real agent loop with a stubbed
// provider. Everything else in the persistence tests calls the recorder
// directly, which proves the codec but not the wiring: it cannot catch a
// listener that was never subscribed, an event that never fires, or a user
// prompt recorded in the wrong order relative to the reply.

// stubStream replays one assistant message as a completed stream.
func stubStream(reply llm.AssistantMessage) agent.StreamFn {
	return func(model *llm.Model, _ *llm.Context, _ *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		stream := llm.NewAssistantMessageEventStream()
		partial := reply
		partial.Content = []llm.ContentBlock{}
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventStart, Partial: &partial})
		final := reply
		final.Provider = model.Provider
		final.Model = model.ID
		stream.Push(llm.AssistantMessageEvent{Type: llm.EventDone, Reason: final.StopReason, Message: &final})
		stream.End()
		return stream
	}
}

// turnFixture is a session whose agent runs a stubbed provider and whose
// conversation is recorded to a real file.
func newTurnFixture(t *testing.T, reply string) *recordingFixture {
	t.Helper()
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())

	store := defaultStore()
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	model := &llm.Model{ID: "test-model", Name: "Test", Provider: "test", API: "test-api", ContextWindow: 128000}
	s := &session{model: model}
	s.log = newSessionRecorder(writer, s.pushNotice)
	s.agent = agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{Model: *model, ThinkingLevel: llm.ThinkingOff},
		StreamFn: stubStream(llm.AssistantMessage{
			Role:       "assistant",
			Content:    []llm.ContentBlock{llm.TextContent{Type: "text", Text: reply}},
			API:        "test-api",
			StopReason: llm.StopStop,
			Usage:      llm.Usage{Cost: llm.UsageCost{Total: 0.02}},
		}),
		ConvertToLLM: convertSessionMessages,
	})
	// Exactly what newSession does, and in the same order.
	s.agent.Subscribe(s.log.listener())

	t.Cleanup(func() { _ = s.log.close() })
	return &recordingFixture{session: s, store: store, cwd: cwd, path: writer.Path()}
}

// TestAWholeTurnIsRecordedInOrder is the wiring test: prompt in, reply out,
// both on disk, in the order they happened.
func TestAWholeTurnIsRecordedInOrder(t *testing.T) {
	f := newTurnFixture(t, "the answer")

	if err := f.session.runTurn("the question"); err != nil {
		t.Fatalf("runTurn: %v", err)
	}

	got := f.reopen(t)
	if len(got.Messages) != 2 {
		t.Fatalf("recorded %d messages, want the prompt and the reply", len(got.Messages))
	}
	user, ok := got.Messages[0].(llm.UserMessage)
	if !ok {
		t.Fatalf("first recorded message is %T, want the user's prompt first", got.Messages[0])
	}
	// Agent.Prompt wraps a string in content blocks, so the round trip has to
	// preserve the block form rather than flattening it back to a string.
	if got := userText(user); got != "the question" {
		t.Fatalf("recorded prompt = %q", got)
	}
	reply, ok := got.Messages[1].(llm.AssistantMessage)
	if !ok {
		t.Fatalf("second recorded message is %T, want the reply", got.Messages[1])
	}
	if blockSummary(reply.Content) != "the answer" {
		t.Fatalf("recorded reply = %q", blockSummary(reply.Content))
	}
	// The provider and model the turn was actually served by have to survive,
	// or a resume cannot restore the model without them.
	if reply.Provider != "test" || reply.Model != "test-model" {
		t.Fatalf("reply lost its provider/model: %s/%s", reply.Provider, reply.Model)
	}
}

// TestSeveralTurnsFormOneChain covers the tree the whole feature rests on:
// every entry must attach to the one before it, with a single root.
func TestSeveralTurnsFormOneChain(t *testing.T) {
	f := newTurnFixture(t, "ok")

	for _, prompt := range []string{"first", "second", "third"} {
		if err := f.session.runTurn(prompt); err != nil {
			t.Fatalf("runTurn(%q): %v", prompt, err)
		}
	}
	f.session.log.sync()

	tree := f.session.log.snapshot()
	var roots int
	for _, entry := range tree.All() {
		if entry.ParentID == nil {
			roots++
			continue
		}
		if !tree.Has(*entry.ParentID) {
			t.Fatalf("entry %s points at a parent that is not in the tree", entry.ID)
		}
	}
	if roots != 1 {
		t.Fatalf("the session has %d roots, want exactly 1", roots)
	}
	if got := len(tree.Path(tree.Leaf())); got != 6 {
		t.Fatalf("the path holds %d entries, want three prompts and three replies", got)
	}
}

// TestCostSurvivesAFullTurnAndResume ties the recorded usage back to what the
// sidebar reports. A resume that lost it would under-report spend silently.
func TestCostSurvivesAFullTurnAndResume(t *testing.T) {
	f := newTurnFixture(t, "an answer")

	for i := 0; i < 3; i++ {
		if err := f.session.runTurn("a question"); err != nil {
			t.Fatalf("runTurn: %v", err)
		}
	}
	before := conversationCost(f.session.agent.State().Messages)
	if before <= 0 {
		t.Fatalf("the stub recorded no cost (%v); the test proves nothing", before)
	}

	got := f.reopen(t)
	if after := conversationCost(got.Messages); after != before {
		t.Fatalf("cost before resume %.4f != after %.4f", before, after)
	}
}

// TestModelSwitchMidConversationIsRecordedBetweenTheTurns covers the ordering
// guarantee the emit seam exists for: the change must land between the turn it
// followed and the turn it affected, not at the end.
func TestModelSwitchMidConversationIsRecordedBetweenTheTurns(t *testing.T) {
	f := newTurnFixture(t, "ok")

	if err := f.session.runTurn("before the switch"); err != nil {
		t.Fatalf("runTurn: %v", err)
	}
	f.session.agent.SetModel(llm.Model{Provider: "anthropic", ID: "claude-sonnet-4-5"})
	if err := f.session.runTurn("after the switch"); err != nil {
		t.Fatalf("runTurn: %v", err)
	}
	f.session.log.sync()

	entries := f.session.log.snapshot().Path(f.session.log.leaf())
	var order []string
	for _, entry := range entries {
		switch entry.Type {
		case sessionlog.TypeModelChange:
			order = append(order, "model_change")
		case sessionlog.TypeMessage:
			order = append(order, entryRole(entry))
		}
	}
	joined := strings.Join(order, ",")
	want := "user,assistant,model_change,user,assistant"
	if joined != want {
		t.Fatalf("recorded order = %q, want %q", joined, want)
	}
}

// TestNoSessionRunsATurnWithoutRecording is the negative control for the whole
// file: with no recorder the same turn must still complete.
func TestNoSessionRunsATurnWithoutRecording(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())

	model := &llm.Model{ID: "test-model", Provider: "test", API: "test-api", ContextWindow: 128000}
	s := &session{model: model}
	s.agent = agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{Model: *model, ThinkingLevel: llm.ThinkingOff},
		StreamFn: stubStream(llm.AssistantMessage{
			Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "fine"}},
			API: "test-api", StopReason: llm.StopStop,
		}),
		ConvertToLLM: convertSessionMessages,
	})

	if err := s.runTurn("a question"); err != nil {
		t.Fatalf("runTurn with no recorder: %v", err)
	}
	if got := len(s.agent.State().Messages); got != 2 {
		t.Fatalf("the turn produced %d messages", got)
	}
	if s.recordingActive() {
		t.Fatal("a session with no recorder reports itself as recording")
	}
}
