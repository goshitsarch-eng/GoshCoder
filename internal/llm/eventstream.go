package llm

import (
	"context"
	"errors"
	"sync"
)

// AssistantMessageEvent is the Go port of pi's AssistantMessageEvent sum type
// (reference/pi/packages/ai/src/types.ts). Type is the discriminant; the
// remaining fields are populated per event kind:
//
//	start:          Partial
//	text_start:     ContentIndex, Partial
//	text_delta:     ContentIndex, Delta, Partial
//	text_end:       ContentIndex, Content, Partial
//	thinking_*:     same shape as text_*
//	toolcall_start: ContentIndex, Partial
//	toolcall_delta: ContentIndex, Delta, Partial
//	toolcall_end:   ContentIndex, ToolCall, Partial
//	done:           Reason (stop|length|toolUse|deferred), Message
//	error:          Reason (aborted|error), Error (the final AssistantMessage)
type AssistantMessageEvent struct {
	Type         EventType
	ContentIndex int
	Delta        string
	Content      string
	ToolCall     *ToolCall
	Reason       StopReason
	// Partial is the progressively assembled AssistantMessage. The same
	// underlying message is shared across events of one stream; treat it as
	// read-only unless you are the producer.
	Partial *AssistantMessage
	// Message is set on "done".
	Message *AssistantMessage
	// Error is set on "error" (an AssistantMessage carrying the failure).
	Error *AssistantMessage
}

type EventType = string

const (
	EventStart         EventType = "start"
	EventTextStart     EventType = "text_start"
	EventTextDelta     EventType = "text_delta"
	EventTextEnd       EventType = "text_end"
	EventThinkingStart EventType = "thinking_start"
	EventThinkingDelta EventType = "thinking_delta"
	EventThinkingEnd   EventType = "thinking_end"
	EventToolCallStart EventType = "toolcall_start"
	EventToolCallDelta EventType = "toolcall_delta"
	EventToolCallEnd   EventType = "toolcall_end"
	EventDone          EventType = "done"
	EventError         EventType = "error"
)

// ErrStreamClosedWithoutResult is returned by Result when the stream ended
// without a terminal done/error event.
var ErrStreamClosedWithoutResult = errors.New("llm: stream ended without a done or error event")

// AssistantMessageEventStream is the Go port of pi's AssistantMessageEventStream
// (reference/pi/packages/ai/src/utils/event-stream.ts). Like the TS original it
// is an unbounded ordered queue: producers never block, consumers pull events
// with Next or range over Events, and Result resolves with the final
// AssistantMessage carried by the terminal done/error event.
type AssistantMessageEventStream struct {
	mu       sync.Mutex
	queue    []AssistantMessageEvent
	notify   chan struct{} // closed to wake all waiters, then replaced
	finished bool          // a terminal done/error event was pushed
	ended    bool          // End was called
	result   *AssistantMessage
}

func NewAssistantMessageEventStream() *AssistantMessageEventStream {
	return &AssistantMessageEventStream{notify: make(chan struct{})}
}

// Push appends an event. Pushes after a terminal done/error event are ignored,
// matching the TS implementation.
func (s *AssistantMessageEventStream) Push(event AssistantMessageEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished {
		return
	}
	if event.Type == EventDone || event.Type == EventError {
		s.finished = true
		if event.Type == EventDone {
			s.result = event.Message
		} else {
			s.result = event.Error
		}
	}
	s.queue = append(s.queue, event)
	s.signalLocked()
}

// End marks the stream as ended. Consumers drain the remaining queue and then
// see the stream as closed. If no terminal event was pushed, Result resolves
// with ErrStreamClosedWithoutResult.
func (s *AssistantMessageEventStream) End() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.ended = true
	s.signalLocked()
}

func (s *AssistantMessageEventStream) signalLocked() {
	close(s.notify)
	s.notify = make(chan struct{})
}

// Next returns the next event in order. ok is false when the stream is closed
// and drained, or when ctx is canceled while waiting.
func (s *AssistantMessageEventStream) Next(ctx context.Context) (event AssistantMessageEvent, ok bool) {
	for {
		s.mu.Lock()
		if len(s.queue) > 0 {
			event = s.queue[0]
			s.queue = s.queue[1:]
			s.mu.Unlock()
			return event, true
		}
		if s.finished || s.ended {
			s.mu.Unlock()
			return AssistantMessageEvent{}, false
		}
		ch := s.notify
		s.mu.Unlock()

		select {
		case <-ch:
		case <-ctx.Done():
			return AssistantMessageEvent{}, false
		}
	}
}

// Events returns a channel that yields every event in order and closes when
// the stream is closed and drained.
func (s *AssistantMessageEventStream) Events() <-chan AssistantMessageEvent {
	ch := make(chan AssistantMessageEvent)
	go func() {
		defer close(ch)
		for {
			event, ok := s.Next(context.Background())
			if !ok {
				return
			}
			ch <- event
		}
	}()
	return ch
}

// Result blocks until the terminal done/error event (or End) and returns the
// final AssistantMessage, exactly like the TS result() promise.
func (s *AssistantMessageEventStream) Result(ctx context.Context) (*AssistantMessage, error) {
	for {
		s.mu.Lock()
		if s.finished {
			result := s.result
			s.mu.Unlock()
			return result, nil
		}
		if s.ended {
			s.mu.Unlock()
			return nil, ErrStreamClosedWithoutResult
		}
		ch := s.notify
		s.mu.Unlock()

		select {
		case <-ch:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}
