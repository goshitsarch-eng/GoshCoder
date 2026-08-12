package llm

// AWS binary event-stream (application/vnd.amazon.eventstream) decoding.
//
// DEVIATION: pi gets this from @aws-sdk/eventstream-serde. Bedrock's
// ConverseStream responds with framed binary messages rather than SSE, so the
// framing is implemented here.
//
// Wire format per message:
//
//	prelude: total length (uint32), headers length (uint32), prelude CRC32 (uint32)
//	headers: repeated (name length (uint8), name, value type (uint8), value)
//	payload: total length - headers length - 16 bytes
//	message CRC32 (uint32) over every preceding byte of the message
//
// All integers are big-endian. Both CRCs are IEEE CRC-32.

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
)

// maxEventStreamMessageSize bounds one frame, guarding against a corrupt
// length prefix causing a huge allocation. AWS caps event stream messages at
// 16 MiB.
const maxEventStreamMessageSize = 16 << 20

// eventStreamMessage is one decoded frame.
type eventStreamMessage struct {
	// Headers holds string-valued headers, which is all Bedrock uses. The
	// meaningful ones are ":event-type", ":exception-type", and ":message-type".
	Headers map[string]string
	Payload []byte
}

// EventType returns the ":event-type" header.
func (m *eventStreamMessage) EventType() string { return m.Headers[":event-type"] }

// ExceptionType returns the ":exception-type" header, set on error frames.
func (m *eventStreamMessage) ExceptionType() string { return m.Headers[":exception-type"] }

// MessageType returns the ":message-type" header ("event", "exception", or
// "error").
func (m *eventStreamMessage) MessageType() string { return m.Headers[":message-type"] }

// eventStreamReader decodes framed messages from a response body.
type eventStreamReader struct {
	r io.Reader
}

func newEventStreamReader(r io.Reader) *eventStreamReader {
	return &eventStreamReader{r: r}
}

// Next decodes the next message, returning io.EOF at the end of the stream.
func (s *eventStreamReader) Next() (*eventStreamMessage, error) {
	var prelude [12]byte
	if _, err := io.ReadFull(s.r, prelude[:]); err != nil {
		// A clean end of stream is io.EOF; a partial prelude is corruption.
		if err == io.EOF {
			return nil, io.EOF
		}
		if err == io.ErrUnexpectedEOF {
			return nil, fmt.Errorf("event stream: truncated message prelude")
		}
		return nil, err
	}

	totalLength := binary.BigEndian.Uint32(prelude[0:4])
	headersLength := binary.BigEndian.Uint32(prelude[4:8])
	preludeCRC := binary.BigEndian.Uint32(prelude[8:12])

	if got := crc32.ChecksumIEEE(prelude[0:8]); got != preludeCRC {
		return nil, fmt.Errorf("event stream: prelude CRC mismatch (got %08x, want %08x)", got, preludeCRC)
	}
	if totalLength < 16 || totalLength > maxEventStreamMessageSize {
		return nil, fmt.Errorf("event stream: implausible message length %d", totalLength)
	}
	if uint64(headersLength)+16 > uint64(totalLength) {
		return nil, fmt.Errorf("event stream: headers length %d exceeds message length %d", headersLength, totalLength)
	}

	// The rest of the message: headers, payload, and the trailing CRC.
	remaining := make([]byte, totalLength-12)
	if _, err := io.ReadFull(s.r, remaining); err != nil {
		return nil, fmt.Errorf("event stream: truncated message body: %w", err)
	}

	messageCRC := binary.BigEndian.Uint32(remaining[len(remaining)-4:])
	// The trailing CRC covers the prelude plus everything before itself.
	sum := crc32.ChecksumIEEE(prelude[:])
	sum = crc32.Update(sum, crc32.IEEETable, remaining[:len(remaining)-4])
	if sum != messageCRC {
		return nil, fmt.Errorf("event stream: message CRC mismatch (got %08x, want %08x)", sum, messageCRC)
	}

	headerBytes := remaining[:headersLength]
	payload := remaining[headersLength : len(remaining)-4]

	headers, err := parseEventStreamHeaders(headerBytes)
	if err != nil {
		return nil, err
	}
	return &eventStreamMessage{Headers: headers, Payload: payload}, nil
}

// Header value type identifiers.
const (
	headerTypeBoolTrue  = 0
	headerTypeBoolFalse = 1
	headerTypeByte      = 2
	headerTypeShort     = 3
	headerTypeInteger   = 4
	headerTypeLong      = 5
	headerTypeByteArray = 6
	headerTypeString    = 7
	headerTypeTimestamp = 8
	headerTypeUUID      = 9
)

// parseEventStreamHeaders decodes the header block. Non-string values are
// skipped rather than converted: Bedrock's control headers are all strings.
func parseEventStreamHeaders(data []byte) (map[string]string, error) {
	headers := map[string]string{}
	offset := 0

	for offset < len(data) {
		if offset+1 > len(data) {
			return nil, fmt.Errorf("event stream: truncated header name length")
		}
		nameLength := int(data[offset])
		offset++
		if offset+nameLength > len(data) {
			return nil, fmt.Errorf("event stream: truncated header name")
		}
		name := string(data[offset : offset+nameLength])
		offset += nameLength

		if offset+1 > len(data) {
			return nil, fmt.Errorf("event stream: truncated header value type")
		}
		valueType := data[offset]
		offset++

		switch valueType {
		case headerTypeBoolTrue, headerTypeBoolFalse:
			// No value bytes follow.
		case headerTypeByte:
			offset += 1
		case headerTypeShort:
			offset += 2
		case headerTypeInteger:
			offset += 4
		case headerTypeLong, headerTypeTimestamp:
			offset += 8
		case headerTypeUUID:
			offset += 16
		case headerTypeByteArray, headerTypeString:
			if offset+2 > len(data) {
				return nil, fmt.Errorf("event stream: truncated header value length")
			}
			valueLength := int(binary.BigEndian.Uint16(data[offset : offset+2]))
			offset += 2
			if offset+valueLength > len(data) {
				return nil, fmt.Errorf("event stream: truncated header value")
			}
			if valueType == headerTypeString {
				headers[name] = string(data[offset : offset+valueLength])
			}
			offset += valueLength
		default:
			return nil, fmt.Errorf("event stream: unknown header value type %d", valueType)
		}

		if offset > len(data) {
			return nil, fmt.Errorf("event stream: header block overran its bounds")
		}
	}
	return headers, nil
}

// encodeEventStreamMessage frames a message. Only used by tests, which need to
// produce valid frames for a fake Bedrock endpoint.
func encodeEventStreamMessage(headers map[string]string, payload []byte) []byte {
	var headerBytes []byte
	for name, value := range headers {
		headerBytes = append(headerBytes, byte(len(name)))
		headerBytes = append(headerBytes, name...)
		headerBytes = append(headerBytes, headerTypeString)
		headerBytes = binary.BigEndian.AppendUint16(headerBytes, uint16(len(value)))
		headerBytes = append(headerBytes, value...)
	}

	totalLength := uint32(16 + len(headerBytes) + len(payload))
	message := make([]byte, 0, totalLength)
	message = binary.BigEndian.AppendUint32(message, totalLength)
	message = binary.BigEndian.AppendUint32(message, uint32(len(headerBytes)))
	message = binary.BigEndian.AppendUint32(message, crc32.ChecksumIEEE(message[0:8]))
	message = append(message, headerBytes...)
	message = append(message, payload...)
	return binary.BigEndian.AppendUint32(message, crc32.ChecksumIEEE(message))
}
