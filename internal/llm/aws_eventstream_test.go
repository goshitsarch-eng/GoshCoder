package llm

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"io"
	"strings"
	"testing"
)

func TestEventStreamRoundTrip(t *testing.T) {
	frame := encodeEventStreamMessage(
		map[string]string{":event-type": "contentBlockDelta", ":message-type": "event"},
		[]byte(`{"delta":{"text":"hi"}}`),
	)

	reader := newEventStreamReader(bytes.NewReader(frame))
	message, err := reader.Next()
	if err != nil {
		t.Fatalf("Next: %v", err)
	}
	if message.EventType() != "contentBlockDelta" || message.MessageType() != "event" {
		t.Fatalf("headers = %#v", message.Headers)
	}
	if string(message.Payload) != `{"delta":{"text":"hi"}}` {
		t.Fatalf("payload = %q", message.Payload)
	}

	// The stream ends cleanly after the last frame.
	if _, err := reader.Next(); err != io.EOF {
		t.Fatalf("second Next = %v, want io.EOF", err)
	}
}

func TestEventStreamMultipleFrames(t *testing.T) {
	var stream bytes.Buffer
	for _, eventType := range []string{"messageStart", "contentBlockDelta", "messageStop"} {
		stream.Write(encodeEventStreamMessage(map[string]string{":event-type": eventType}, []byte("{}")))
	}

	reader := newEventStreamReader(&stream)
	var seen []string
	for {
		message, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		seen = append(seen, message.EventType())
	}
	if strings.Join(seen, ",") != "messageStart,contentBlockDelta,messageStop" {
		t.Fatalf("events = %v", seen)
	}
}

func TestEventStreamEmptyPayload(t *testing.T) {
	frame := encodeEventStreamMessage(map[string]string{":event-type": "contentBlockStop"}, nil)
	reader := newEventStreamReader(bytes.NewReader(frame))

	message, err := reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if len(message.Payload) != 0 {
		t.Fatalf("payload = %q, want empty", message.Payload)
	}
}

// TestEventStreamPreludeCRCMismatch confirms a corrupted length prefix is
// rejected rather than driving a bogus allocation.
func TestEventStreamPreludeCRCMismatch(t *testing.T) {
	frame := encodeEventStreamMessage(map[string]string{":event-type": "x"}, []byte("{}"))
	// Corrupt the total length without fixing the prelude CRC.
	binary.BigEndian.PutUint32(frame[0:4], 999999)

	_, err := newEventStreamReader(bytes.NewReader(frame)).Next()
	if err == nil || !strings.Contains(err.Error(), "prelude CRC mismatch") {
		t.Fatalf("err = %v", err)
	}
}

// TestEventStreamMessageCRCMismatch confirms payload corruption is caught.
func TestEventStreamMessageCRCMismatch(t *testing.T) {
	frame := encodeEventStreamMessage(map[string]string{":event-type": "x"}, []byte("{}"))
	// Flip a payload byte, leaving the trailing CRC stale.
	frame[len(frame)-6] ^= 0xff

	_, err := newEventStreamReader(bytes.NewReader(frame)).Next()
	if err == nil || !strings.Contains(err.Error(), "message CRC mismatch") {
		t.Fatalf("err = %v", err)
	}
}

func TestEventStreamTruncatedPrelude(t *testing.T) {
	// Fewer than the 12 prelude bytes.
	_, err := newEventStreamReader(bytes.NewReader([]byte{0, 0, 0})).Next()
	if err == nil || !strings.Contains(err.Error(), "truncated message prelude") {
		t.Fatalf("err = %v", err)
	}
}

func TestEventStreamTruncatedBody(t *testing.T) {
	frame := encodeEventStreamMessage(map[string]string{":event-type": "x"}, []byte("payload"))
	// Drop the last few bytes so the declared length overruns the data.
	_, err := newEventStreamReader(bytes.NewReader(frame[:len(frame)-4])).Next()
	if err == nil || !strings.Contains(err.Error(), "truncated message body") {
		t.Fatalf("err = %v", err)
	}
}

// TestEventStreamImplausibleLength covers a length prefix that passes its CRC
// but is absurd, which must not be allocated.
func TestEventStreamImplausibleLength(t *testing.T) {
	prelude := make([]byte, 12)
	binary.BigEndian.PutUint32(prelude[0:4], maxEventStreamMessageSize+1)
	binary.BigEndian.PutUint32(prelude[4:8], 0)
	binary.BigEndian.PutUint32(prelude[8:12], crc32.ChecksumIEEE(prelude[0:8]))

	_, err := newEventStreamReader(bytes.NewReader(prelude)).Next()
	if err == nil || !strings.Contains(err.Error(), "implausible message length") {
		t.Fatalf("err = %v", err)
	}
}

// TestEventStreamHeadersLengthOverrun covers a headers length larger than the
// message that contains it.
func TestEventStreamHeadersLengthOverrun(t *testing.T) {
	prelude := make([]byte, 12)
	binary.BigEndian.PutUint32(prelude[0:4], 32)
	binary.BigEndian.PutUint32(prelude[4:8], 64)
	binary.BigEndian.PutUint32(prelude[8:12], crc32.ChecksumIEEE(prelude[0:8]))

	_, err := newEventStreamReader(bytes.NewReader(prelude)).Next()
	if err == nil || !strings.Contains(err.Error(), "exceeds message length") {
		t.Fatalf("err = %v", err)
	}
}

func TestParseEventStreamHeaderValueTypes(t *testing.T) {
	// Only string headers are surfaced; the other value types are skipped over
	// so a frame carrying them still parses.
	var headers []byte
	appendHeader := func(name string, valueType byte, value []byte) {
		headers = append(headers, byte(len(name)))
		headers = append(headers, name...)
		headers = append(headers, valueType)
		headers = append(headers, value...)
	}

	appendHeader("bool-true", headerTypeBoolTrue, nil)
	appendHeader("a-byte", headerTypeByte, []byte{1})
	appendHeader("a-short", headerTypeShort, []byte{0, 1})
	appendHeader("an-int", headerTypeInteger, []byte{0, 0, 0, 1})
	appendHeader("a-long", headerTypeLong, []byte{0, 0, 0, 0, 0, 0, 0, 1})
	appendHeader("a-uuid", headerTypeUUID, make([]byte, 16))

	stringValue := []byte("event")
	lengthPrefixed := append(binary.BigEndian.AppendUint16(nil, uint16(len(stringValue))), stringValue...)
	appendHeader(":event-type", headerTypeString, lengthPrefixed)

	parsed, err := parseEventStreamHeaders(headers)
	if err != nil {
		t.Fatalf("parseEventStreamHeaders: %v", err)
	}
	if parsed[":event-type"] != "event" {
		t.Fatalf("headers = %#v", parsed)
	}
	// Non-string headers are skipped rather than converted.
	if len(parsed) != 1 {
		t.Fatalf("headers = %#v, want only the string header", parsed)
	}
}

func TestParseEventStreamHeadersRejectsUnknownType(t *testing.T) {
	headers := []byte{1, 'x', 99}
	if _, err := parseEventStreamHeaders(headers); err == nil {
		t.Fatal("an unknown value type must be rejected")
	}
}

func TestParseEventStreamHeadersRejectsTruncation(t *testing.T) {
	tests := []struct {
		name    string
		headers []byte
	}{
		{"truncated name", []byte{10, 'x'}},
		{"missing value type", []byte{1, 'x'}},
		{"truncated value length", []byte{1, 'x', headerTypeString, 0}},
		{"truncated value", append([]byte{1, 'x', headerTypeString}, 0, 10, 'y')},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := parseEventStreamHeaders(tt.headers); err == nil {
				t.Fatalf("expected an error for %s", tt.name)
			}
		})
	}
}

func TestEventStreamExceptionHeaders(t *testing.T) {
	frame := encodeEventStreamMessage(map[string]string{
		":message-type":   "exception",
		":exception-type": "ThrottlingException",
	}, []byte(`{"message":"slow down"}`))

	message, err := newEventStreamReader(bytes.NewReader(frame)).Next()
	if err != nil {
		t.Fatal(err)
	}
	if message.ExceptionType() != "ThrottlingException" || message.MessageType() != "exception" {
		t.Fatalf("headers = %#v", message.Headers)
	}
}
