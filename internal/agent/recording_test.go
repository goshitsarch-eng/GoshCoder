package agent

import (
	"context"
	"sync"
	"testing"

	"goshcoder/internal/llm"
)

// collect subscribes a listener that records every event type it sees.
func collect(a *Agent) *eventLog {
	log := &eventLog{}
	a.Subscribe(func(_ context.Context, ev Event) { log.add(ev) })
	return log
}

type eventLog struct {
	mu     sync.Mutex
	events []Event
}

func (l *eventLog) add(ev Event) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, ev)
}

func (l *eventLog) ofType(t EventType) []Event {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []Event
	for _, ev := range l.events {
		if ev.Type == t {
			out = append(out, ev)
		}
	}
	return out
}

func idleAgent(t *testing.T) *Agent {
	t.Helper()
	return NewAgent(AgentOptions{
		InitialState: &InitialState{
			SystemPrompt:  "s",
			Model:         llm.Model{Provider: "anthropic", ID: "claude-sonnet-4-5"},
			ThinkingLevel: "off",
		},
	})
}

// TestModelChangeReachesListeners covers the gap that made a Subscribe-only
// recorder wrong: SetModel mutated state and emitted nothing, so a resumed
// session came back on whatever model the CLI defaulted to rather than the one
// the conversation was actually held with.
func TestModelChangeReachesListeners(t *testing.T) {
	a := idleAgent(t)
	log := collect(a)

	a.SetModel(llm.Model{Provider: "openai", ID: "gpt-5.1"})

	events := log.ofType(EventModelChange)
	if len(events) != 1 {
		t.Fatalf("model_change events = %d, want 1", len(events))
	}
	if events[0].Provider != "openai" || events[0].ModelID != "gpt-5.1" {
		t.Fatalf("event = %s/%s, want openai/gpt-5.1", events[0].Provider, events[0].ModelID)
	}
}

// TestThinkingLevelChangeReachesListeners is the same gap for the other
// setting a resume has to restore.
func TestThinkingLevelChangeReachesListeners(t *testing.T) {
	a := idleAgent(t)
	log := collect(a)

	a.SetThinkingLevel("high")

	events := log.ofType(EventThinkingLevelChange)
	if len(events) != 1 {
		t.Fatalf("thinking_level_change events = %d, want 1", len(events))
	}
	if events[0].ThinkingLevel != "high" {
		t.Fatalf("level = %q, want high", events[0].ThinkingLevel)
	}
}

// TestUnchangedValuesEmitNothing is the negative control, and it guards a real
// cost: syncPlanRuntime calls both setters twice per turn with the values they
// already hold. Emitting on every call would append two dead entries to the
// session log for every turn of every session, forever.
func TestUnchangedValuesEmitNothing(t *testing.T) {
	a := idleAgent(t)
	log := collect(a)

	same := llm.Model{Provider: "anthropic", ID: "claude-sonnet-4-5"}
	for i := 0; i < 100; i++ {
		a.SetModel(same)
		a.SetThinkingLevel("off")
	}

	if got := len(log.ofType(EventModelChange)); got != 0 {
		t.Fatalf("model_change events = %d, want 0 for repeated identical values", got)
	}
	if got := len(log.ofType(EventThinkingLevelChange)); got != 0 {
		t.Fatalf("thinking_level_change events = %d, want 0 for repeated identical values", got)
	}
}

// TestCompactReplacesTheTranscriptAndEmits covers the replacement for the
// external SetMessages call. The event has to carry the marker and the kept
// messages, because the recorder writes exactly the cut the agent made.
func TestCompactReplacesTheTranscriptAndEmits(t *testing.T) {
	a := idleAgent(t)
	a.SetMessages([]Message{
		llm.UserMessage{Role: "user", Content: "old one"},
		llm.UserMessage{Role: "user", Content: "old two"},
		llm.UserMessage{Role: "user", Content: "kept"},
	})
	log := collect(a)

	marker := llm.UserMessage{Role: "user", Content: "summary"}
	kept := []Message{llm.UserMessage{Role: "user", Content: "kept"}}
	info := CompactionInfo{Summary: "summary", TokensBefore: 1000, CostBefore: 1.5, RetainedMessages: 1}

	if err := a.Compact(marker, kept, info); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	if got := len(a.State().Messages); got != 2 {
		t.Fatalf("transcript has %d messages, want the marker plus one kept", got)
	}
	events := log.ofType(EventContextCompacted)
	if len(events) != 1 {
		t.Fatalf("context_compacted events = %d, want 1", len(events))
	}
	if events[0].Compaction == nil {
		t.Fatal("event carries no CompactionInfo")
	}
	// Cost and retained count are what the interface derives spend and context
	// usage from; a resume that loses them misreports both without failing.
	if events[0].Compaction.CostBefore != 1.5 || events[0].Compaction.RetainedMessages != 1 {
		t.Fatalf("CompactionInfo = %+v, want the cost and retained count preserved", *events[0].Compaction)
	}
	if len(events[0].Kept) != 1 {
		t.Fatalf("event carries %d kept messages, want 1", len(events[0].Kept))
	}
}

// TestResetEmitsItsReason covers /clear: without an event the log keeps
// growing across a reset, and the next resume replays a conversation the user
// deliberately discarded.
func TestResetEmitsItsReason(t *testing.T) {
	a := idleAgent(t)
	a.SetMessages([]Message{llm.UserMessage{Role: "user", Content: "before"}})
	log := collect(a)

	if err := a.ResetWithReason("/clear"); err != nil {
		t.Fatalf("ResetWithReason: %v", err)
	}

	events := log.ofType(EventTranscriptReset)
	if len(events) != 1 {
		t.Fatalf("transcript_reset events = %d, want 1", len(events))
	}
	if events[0].Reason != "/clear" {
		t.Fatalf("reason = %q, want /clear", events[0].Reason)
	}
	if got := len(a.State().Messages); got != 0 {
		t.Fatalf("transcript still holds %d messages after a reset", got)
	}
}

// TestResetDoesNotEmitWhileHoldingTheStateLock is the deadlock this ordering
// exists to avoid: emit takes a.mu itself, and a listener writes to disk while
// it runs. A listener that reads State() would deadlock if Reset still held
// the lock.
func TestResetDoesNotEmitWhileHoldingTheStateLock(t *testing.T) {
	a := idleAgent(t)
	done := make(chan struct{})
	a.Subscribe(func(_ context.Context, ev Event) {
		if ev.Type != EventTranscriptReset {
			return
		}
		// Would block forever if Reset emitted under a.mu.
		_ = a.State().Messages
		close(done)
	})

	if err := a.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	select {
	case <-done:
	default:
		t.Fatal("the reset listener did not run")
	}
}

// TestConcurrentSetModelAndAppendsAreLinearized reproduces the real shape:
// Ctrl+P cycles the model from the interface goroutine, ungated by whether a
// run is streaming, while the run goroutine appends messages. The recorded
// order has to be the state's linearization order, or the session log claims a
// turn was held on a model it was not.
func TestConcurrentSetModelAndAppendsAreLinearized(t *testing.T) {
	a := idleAgent(t)
	log := collect(a)

	models := []llm.Model{
		{Provider: "anthropic", ID: "claude-sonnet-4-5"},
		{Provider: "openai", ID: "gpt-5.1"},
		{Provider: "google", ID: "gemini-3-pro"},
	}

	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				a.SetModel(models[(worker+i)%len(models)])
				a.SetThinkingLevel(llm.ModelThinkingLevel([]string{"off", "low", "high"}[(worker+i)%3]))
			}
		}(worker)
	}
	wg.Wait()

	// Whatever interleaving happened, the last recorded change must equal the
	// live state -- that is what makes replaying the log reconstruct it.
	changes := log.ofType(EventModelChange)
	if len(changes) == 0 {
		t.Fatal("no model changes were recorded")
	}
	last := changes[len(changes)-1]
	state := a.State()
	if last.Provider != state.Model.Provider || last.ModelID != state.Model.ID {
		t.Fatalf("last recorded model %s/%s != live state %s/%s",
			last.Provider, last.ModelID, state.Model.Provider, state.Model.ID)
	}
	levels := log.ofType(EventThinkingLevelChange)
	if len(levels) == 0 {
		t.Fatal("no thinking level changes were recorded")
	}
	if got := levels[len(levels)-1].ThinkingLevel; got != state.ThinkingLevel {
		t.Fatalf("last recorded level %q != live state %q", got, state.ThinkingLevel)
	}
}
