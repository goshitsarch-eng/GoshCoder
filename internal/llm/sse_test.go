package llm

import (
	"errors"
	"io"
	"strings"
	"testing"
)

func collectSSE(t *testing.T, input string) []SSEEvent {
	t.Helper()
	r := NewSSEReader(strings.NewReader(input))
	var events []SSEEvent
	for {
		ev, err := r.Next()
		if err == io.EOF {
			return events
		}
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		events = append(events, ev)
	}
}

func TestSSEReaderRejectsOversizedLine(t *testing.T) {
	reader := NewSSEReader(strings.NewReader(strings.Repeat("x", maxSSELineBytes+1)))
	if _, err := reader.Next(); !errors.Is(err, ErrSSELineTooLarge) {
		t.Fatalf("err = %v", err)
	}
}

func TestSSEBasicDataLines(t *testing.T) {
	events := collectSSE(t, "data: hello\n\ndata: world\n\n")
	if len(events) != 2 || events[0].Data != "hello" || events[1].Data != "world" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSEMultiLineData(t *testing.T) {
	events := collectSSE(t, "data: line1\ndata: line2\n\n")
	if len(events) != 1 || events[0].Data != "line1\nline2" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSEDoneSentinel(t *testing.T) {
	// The parser passes [DONE] through as data; callers handle it.
	events := collectSSE(t, "data: {}\n\ndata: [DONE]\n\n")
	if len(events) != 2 || events[1].Data != "[DONE]" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSECommentsIgnored(t *testing.T) {
	events := collectSSE(t, ": keepalive\n\ndata: x\n\n")
	if len(events) != 1 || events[0].Data != "x" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSEEventAndIDFields(t *testing.T) {
	events := collectSSE(t, "event: message\nid: 42\ndata: payload\n\n")
	if len(events) != 1 || events[0].Event != "message" || events[0].ID != "42" || events[0].Data != "payload" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSECRLFLineEndings(t *testing.T) {
	events := collectSSE(t, "data: a\r\n\r\ndata: b\r\n\r\n")
	if len(events) != 2 || events[0].Data != "a" || events[1].Data != "b" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSELeadingSpaceStripped(t *testing.T) {
	// Exactly one leading space after the colon is stripped.
	events := collectSSE(t, "data:  two-spaces\n\n")
	if len(events) != 1 || events[0].Data != " two-spaces" {
		t.Fatalf("got %#v", events)
	}
	// No space at all also works.
	events = collectSSE(t, "data:nospace\n\n")
	if len(events) != 1 || events[0].Data != "nospace" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSEBlankWithoutDataDispatchNothing(t *testing.T) {
	events := collectSSE(t, "event: nothing\n\ndata: x\n\n")
	if len(events) != 1 || events[0].Data != "x" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSETrailingUnterminatedLine(t *testing.T) {
	// A final data line without a blank-line dispatch is dropped (spec: events
	// dispatch only on blank lines).
	events := collectSSE(t, "data: complete\n\ndata: dangling")
	if len(events) != 1 || events[0].Data != "complete" {
		t.Fatalf("got %#v", events)
	}
}

func TestSSEEmptyStream(t *testing.T) {
	events := collectSSE(t, "")
	if len(events) != 0 {
		t.Fatalf("got %#v", events)
	}
}

// TestSSECapCoversColonLessDataLines covers a bypass of the event-size cap. The
// guard tested the raw line for a "data:" prefix, but accumulation keys off the
// parsed field name -- and a colon-less "data" line is a valid SSE field with an
// empty value, so it appended to the buffer while slipping past the check.
func TestSSECapCoversColonLessDataLines(t *testing.T) {
	// Each bare "data" line contributes a newline to the accumulator, so
	// enough of them exceed the cap without any line carrying a colon.
	var body strings.Builder
	for range maxSSEEventBytes + 16 {
		body.WriteString("data\n")
	}

	reader := NewSSEReader(strings.NewReader(body.String()))
	_, err := reader.Next()
	if !errors.Is(err, ErrSSEEventTooLarge) {
		t.Fatalf("err = %v, want ErrSSEEventTooLarge", err)
	}
}

// TestSSEStillReadsNormalEvents is the negative control for the moved check.
func TestSSEStillReadsNormalEvents(t *testing.T) {
	reader := NewSSEReader(strings.NewReader("event: ping\ndata: one\ndata: two\n\n"))
	event, err := reader.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if event.Event != "ping" || event.Data != "one\ntwo" {
		t.Fatalf("event = %#v", event)
	}
}
