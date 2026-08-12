package llm

import (
	"bufio"
	"io"
	"strings"
)

// SSEEvent is one dispatched Server-Sent-Events record. Only the fields pi's
// providers use are surfaced: OpenAI-style streams send bare "data:" lines.
type SSEEvent struct {
	Event string
	Data  string
	ID    string
}

// SSEReader is a hand-rolled Server-Sent-Events parser over an io.Reader
// (pi's providers rely on the OpenAI SDK's internal SSE handling; there is no
// vendored standalone parser to port, so this follows the WHATWG SSE stream
// interpretation rules):
//   - "data:" lines accumulate; multiple data lines join with "\n"
//   - "event:"/"id:" set the corresponding fields; "retry:" is parsed but unused
//   - lines starting with ':' are comments and ignored
//   - a blank line dispatches the accumulated event (skipped if no data)
//   - line endings may be \n or \r\n
type SSEReader struct {
	r *bufio.Reader
}

func NewSSEReader(r io.Reader) *SSEReader {
	return &SSEReader{r: bufio.NewReader(r)}
}

// Next returns the next dispatched event. It returns io.EOF when the stream
// is exhausted; any partially accumulated event at EOF is discarded, matching
// the SSE spec (dispatch only happens on blank lines).
func (s *SSEReader) Next() (SSEEvent, error) {
	var event SSEEvent
	var data strings.Builder
	haveData := false

	for {
		line, err := s.readLine()
		if err != nil {
			if err == io.EOF && line != "" {
				// Process a final unterminated line, then report EOF on the
				// next call unless it dispatches an event.
				if ev, ok := s.processLine(line, &event, &data, &haveData); ok {
					return ev, nil
				}
			}
			return SSEEvent{}, err
		}
		if ev, ok := s.processLine(line, &event, &data, &haveData); ok {
			return ev, nil
		}
	}
}

func (s *SSEReader) readLine() (string, error) {
	line, err := s.r.ReadString('\n')
	line = strings.TrimSuffix(line, "\n")
	line = strings.TrimSuffix(line, "\r")
	return line, err
}

// processLine handles one line; ok is true when a blank line dispatched an event.
func (s *SSEReader) processLine(line string, event *SSEEvent, data *strings.Builder, haveData *bool) (SSEEvent, bool) {
	if line == "" {
		if !*haveData {
			*event = SSEEvent{}
			return SSEEvent{}, false
		}
		event.Data = data.String()
		ev := *event
		*event = SSEEvent{}
		data.Reset()
		*haveData = false
		return ev, true
	}
	if strings.HasPrefix(line, ":") {
		return SSEEvent{}, false // comment
	}
	field, value, found := strings.Cut(line, ":")
	if found {
		value = strings.TrimPrefix(value, " ") // exactly one leading space is stripped
	} else {
		value = ""
	}
	switch field {
	case "data":
		if *haveData {
			data.WriteByte('\n')
		}
		data.WriteString(value)
		*haveData = true
	case "event":
		event.Event = value
	case "id":
		event.ID = value
	case "retry":
		// reconnection delay: not used for LLM streams
	}
	return SSEEvent{}, false
}
