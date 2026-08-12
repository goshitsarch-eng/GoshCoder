package llm

import (
	"context"
	"testing"
	"time"
)

func TestEventStreamOrderingAndResult(t *testing.T) {
	es := NewAssistantMessageEventStream()
	final := &AssistantMessage{Role: "assistant", StopReason: StopStop}

	go func() {
		es.Push(AssistantMessageEvent{Type: EventStart})
		es.Push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: 0})
		es.Push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: 0, Delta: "hi"})
		es.Push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: 0, Content: "hi"})
		es.Push(AssistantMessageEvent{Type: EventDone, Reason: StopStop, Message: final})
		es.End()
	}()

	var types []EventType
	for event := range es.Events() {
		types = append(types, event.Type)
	}
	want := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if len(types) != len(want) {
		t.Fatalf("got %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Fatalf("got %v, want %v", types, want)
		}
	}

	result, err := es.Result(context.Background())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if result != final {
		t.Fatalf("Result = %p, want the done message %p", result, final)
	}
}

func TestEventStreamErrorTerminal(t *testing.T) {
	es := NewAssistantMessageEventStream()
	fail := &AssistantMessage{Role: "assistant", StopReason: StopError, ErrorMessage: "boom"}

	go func() {
		es.Push(AssistantMessageEvent{Type: EventStart})
		es.Push(AssistantMessageEvent{Type: EventError, Reason: StopError, Error: fail})
		es.End()
	}()

	// Drain via Next.
	for {
		if _, ok := es.Next(context.Background()); !ok {
			break
		}
	}
	result, err := es.Result(context.Background())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if result != fail || result.ErrorMessage != "boom" {
		t.Fatalf("got %#v", result)
	}
}

func TestEventStreamPushAfterTerminalIgnored(t *testing.T) {
	es := NewAssistantMessageEventStream()
	final := &AssistantMessage{Role: "assistant", StopReason: StopStop}
	es.Push(AssistantMessageEvent{Type: EventDone, Reason: StopStop, Message: final})
	es.Push(AssistantMessageEvent{Type: EventTextDelta, Delta: "late"})

	count := 0
	for range es.Events() {
		count++
	}
	if count != 1 {
		t.Fatalf("expected 1 event, got %d", count)
	}
}

func TestEventStreamEndWithoutTerminal(t *testing.T) {
	es := NewAssistantMessageEventStream()
	es.Push(AssistantMessageEvent{Type: EventStart})
	es.End()
	if _, err := es.Result(context.Background()); err != ErrStreamClosedWithoutResult {
		t.Fatalf("got %v", err)
	}
}

func TestEventStreamNextContextCancel(t *testing.T) {
	es := NewAssistantMessageEventStream()
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	if _, ok := es.Next(ctx); ok {
		t.Fatal("expected Next to return false on canceled context")
	}
}

func TestEventStreamProducerDoesNotBlockWithoutConsumer(t *testing.T) {
	// The TS stream is an unbounded queue; pushing many events with no
	// consumer must not block, and Result must still resolve.
	es := NewAssistantMessageEventStream()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 1000; i++ {
			es.Push(AssistantMessageEvent{Type: EventTextDelta, Delta: "x"})
		}
		es.Push(AssistantMessageEvent{Type: EventDone, Reason: StopStop, Message: &AssistantMessage{StopReason: StopStop}})
		es.End()
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("producer blocked")
	}
	if _, err := es.Result(context.Background()); err != nil {
		t.Fatalf("Result: %v", err)
	}
}
