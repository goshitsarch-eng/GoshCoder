package llm

import (
	"encoding/json"
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// This file ports reference/pi/packages/ai/src/utils/json-parse.ts plus the
// "partial-json" npm package pi depends on (its dist is not vendored in the
// reference checkout, so the tolerant parser below reimplements its documented
// behavior: complete truncated strings/numbers, close open containers, drop
// trailing incomplete key/value pairs).

var validJSONEscapes = map[byte]bool{
	'"': true, '\\': true, '/': true, 'b': true,
	'f': true, 'n': true, 'r': true, 't': true, 'u': true,
}

func isControlByte(b byte) bool { return b <= 0x1f }

func escapeControlByte(b byte) string {
	switch b {
	case '\b':
		return `\b`
	case '\f':
		return `\f`
	case '\n':
		return `\n`
	case '\r':
		return `\r`
	case '\t':
		return `\t`
	default:
		s := strconv.FormatInt(int64(b), 16)
		return `\u` + strings.Repeat("0", 4-len(s)) + s
	}
}

func isHexDigit(b byte) bool {
	return (b >= '0' && b <= '9') || (b >= 'a' && b <= 'f') || (b >= 'A' && b <= 'F')
}

// RepairJSON repairs malformed JSON string literals by escaping raw control
// characters inside strings and doubling backslashes before invalid escape
// characters. Port of repairJson in json-parse.ts. It operates on bytes,
// which is safe because every marker it looks for is ASCII.
func RepairJSON(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	inString := false

	for i := 0; i < len(s); i++ {
		c := s[i]

		if !inString {
			b.WriteByte(c)
			if c == '"' {
				inString = true
			}
			continue
		}

		if c == '"' {
			b.WriteByte(c)
			inString = false
			continue
		}

		if c == '\\' {
			if i+1 >= len(s) {
				b.WriteString(`\\`)
				continue
			}
			next := s[i+1]

			if next == 'u' && i+6 <= len(s) {
				digits := s[i+2 : i+6]
				if isHexDigit(digits[0]) && isHexDigit(digits[1]) && isHexDigit(digits[2]) && isHexDigit(digits[3]) {
					b.WriteString(`\u` + digits)
					i += 5
					continue
				}
			}

			if validJSONEscapes[next] {
				b.WriteByte('\\')
				b.WriteByte(next)
				i++
				continue
			}

			b.WriteString(`\\`)
			continue
		}

		if isControlByte(c) {
			b.WriteString(escapeControlByte(c))
		} else {
			b.WriteByte(c)
		}
	}

	return b.String()
}

// ParseJSONWithRepair tries a strict parse first, then repairs and retries.
// Port of parseJsonWithRepair in json-parse.ts.
func ParseJSONWithRepair(s string) (any, error) {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err == nil {
		return v, nil
	} else {
		repaired := RepairJSON(s)
		if repaired != s {
			var rv any
			if rerr := json.Unmarshal([]byte(repaired), &rv); rerr == nil {
				return rv, nil
			}
		}
		return nil, err
	}
}

// partialParser is a tolerant JSON parser for truncated streaming input.
type partialParser struct {
	s   string
	pos int
}

type incompleteError struct{}

func (incompleteError) Error() string { return "incomplete JSON" }

// ParsePartialJSON parses potentially truncated JSON, completing strings and
// numbers and closing open containers. It is a reimplementation of the
// "partial-json" npm package's default parse() semantics.
func ParsePartialJSON(s string) (any, error) {
	p := &partialParser{s: s}
	p.skipWS()
	if p.pos >= len(p.s) {
		return nil, incompleteError{}
	}
	v, err := p.parseValue()
	if err != nil {
		return nil, err
	}
	return v, nil
}

func (p *partialParser) skipWS() {
	for p.pos < len(p.s) {
		switch p.s[p.pos] {
		case ' ', '\t', '\n', '\r':
			p.pos++
		default:
			return
		}
	}
}

func (p *partialParser) parseValue() (any, error) {
	p.skipWS()
	if p.pos >= len(p.s) {
		return nil, incompleteError{}
	}
	switch c := p.s[p.pos]; {
	case c == '{':
		return p.parseObject()
	case c == '[':
		return p.parseArray()
	case c == '"':
		return p.parseString()
	case c == 't':
		return p.parseLiteral("true", true)
	case c == 'f':
		return p.parseLiteral("false", false)
	case c == 'n':
		return p.parseLiteral("null", nil)
	case c == '-' || (c >= '0' && c <= '9'):
		return p.parseNumber()
	default:
		return nil, incompleteError{}
	}
}

func (p *partialParser) parseLiteral(word string, value any) (any, error) {
	if p.pos+len(word) <= len(p.s) && p.s[p.pos:p.pos+len(word)] == word {
		p.pos += len(word)
		return value, nil
	}
	return nil, incompleteError{}
}

// parseString parses a JSON string; a truncated string is completed with the
// content seen so far. A trailing lone backslash or incomplete \u escape at
// EOF is dropped.
func (p *partialParser) parseString() (string, error) {
	p.pos++ // opening quote
	var b strings.Builder
	for p.pos < len(p.s) {
		c := p.s[p.pos]
		switch c {
		case '"':
			p.pos++
			return b.String(), nil
		case '\\':
			if p.pos+1 >= len(p.s) {
				// trailing backslash at EOF: drop it
				return b.String(), nil
			}
			next := p.s[p.pos+1]
			if next == 'u' {
				r, size, status := decodeUnicodeEscape(p.s[p.pos:])
				switch status {
				case escapeIncomplete:
					// The rest of the escape (or a surrogate's low half) has
					// not streamed in yet. Dropping it is better than emitting
					// a replacement character the next delta would contradict.
					return b.String(), nil
				case escapeInvalid:
					// Not valid hex: keep the raw bytes rather than inventing
					// a rune the strict parser would never produce.
					b.WriteString(p.s[p.pos : p.pos+6])
					p.pos += 6
				default:
					b.WriteRune(r)
					p.pos += size
				}
				continue
			}
			if decoded, ok := simpleEscape(next); ok {
				b.WriteByte(decoded)
				p.pos += 2
				continue
			}
			// Not an escape JSON defines; keep it verbatim.
			b.WriteByte('\\')
			b.WriteByte(next)
			p.pos += 2
		default:
			b.WriteByte(c)
			p.pos++
		}
	}
	return b.String(), nil
}

// Escape-decoding outcomes for decodeUnicodeEscape.
const (
	escapeOK = iota
	// escapeIncomplete: the escape is cut off by the end of the buffer and
	// will be completed by a later delta.
	escapeIncomplete
	// escapeInvalid: the six bytes are present but are not \uXXXX hex.
	escapeInvalid
)

// simpleEscape maps a one-character JSON escape to the byte it denotes.
func simpleEscape(c byte) (byte, bool) {
	switch c {
	case '"':
		return '"', true
	case '\\':
		return '\\', true
	case '/':
		return '/', true
	case 'b':
		return '\b', true
	case 'f':
		return '\f', true
	case 'n':
		return '\n', true
	case 'r':
		return '\r', true
	case 't':
		return '\t', true
	}
	return 0, false
}

// decodeUnicodeEscape decodes the \uXXXX escape at the start of s, joining a
// surrogate pair when its low half follows, and reports how many bytes of s it
// consumed.
func decodeUnicodeEscape(s string) (rune, int, int) {
	high, ok := parseHex4(s)
	if !ok {
		if len(s) < 6 {
			return 0, 0, escapeIncomplete
		}
		return 0, 0, escapeInvalid
	}
	if !utf16.IsSurrogate(rune(high)) {
		return rune(high), 6, escapeOK
	}
	// A high surrogate alone is not a character. Wait for its partner rather
	// than emitting U+FFFD, which a later delta would have to take back.
	if len(s) < 12 {
		return 0, 0, escapeIncomplete
	}
	low, ok := parseHex4(s[6:])
	if !ok {
		return utf8.RuneError, 6, escapeOK
	}
	if joined := utf16.DecodeRune(rune(high), rune(low)); joined != utf8.RuneError {
		return joined, 12, escapeOK
	}
	return utf8.RuneError, 6, escapeOK
}

// parseHex4 reads the four hex digits of a \uXXXX escape at the start of s.
func parseHex4(s string) (uint16, bool) {
	if len(s) < 6 || s[0] != '\\' || s[1] != 'u' {
		return 0, false
	}
	var value uint16
	for i := 2; i < 6; i++ {
		var digit uint16
		switch c := s[i]; {
		case c >= '0' && c <= '9':
			digit = uint16(c - '0')
		case c >= 'a' && c <= 'f':
			digit = uint16(c-'a') + 10
		case c >= 'A' && c <= 'F':
			digit = uint16(c-'A') + 10
		default:
			return 0, false
		}
		value = value<<4 | digit
	}
	return value, true
}

// parseNumber scans a numeric token, trimming trailing characters that leave
// it invalid (e.g. "12.", "1e", "-").
func (p *partialParser) parseNumber() (any, error) {
	start := p.pos
	for p.pos < len(p.s) {
		c := p.s[p.pos]
		if c == '-' || c == '+' || c == '.' || (c >= '0' && c <= '9') || c == 'e' || c == 'E' {
			p.pos++
		} else {
			break
		}
	}
	tok := p.s[start:p.pos]
	for len(tok) > 0 {
		if f, err := strconv.ParseFloat(tok, 64); err == nil {
			return f, nil
		}
		tok = tok[:len(tok)-1]
	}
	return nil, incompleteError{}
}

func (p *partialParser) parseObject() (any, error) {
	p.pos++ // '{'
	obj := map[string]any{}
	for {
		p.skipWS()
		if p.pos >= len(p.s) {
			return obj, nil // truncated: close the object
		}
		if p.s[p.pos] == '}' {
			p.pos++
			return obj, nil
		}
		if p.s[p.pos] == ',' {
			p.pos++
			continue
		}
		if p.s[p.pos] != '"' {
			// unexpected token: stop and close with what we have
			return obj, nil
		}
		key, err := p.parseString()
		if err != nil || p.pos >= len(p.s) {
			// truncated key (or nothing after it): drop the pair
			return obj, nil
		}
		p.skipWS()
		if p.pos >= len(p.s) || p.s[p.pos] != ':' {
			// missing colon: drop the pair
			return obj, nil
		}
		p.pos++
		val, err := p.parseValue()
		if err != nil {
			// incomplete value: drop the pair
			return obj, nil
		}
		obj[key] = val
		// After a value, expect ',' or '}'; anything else closes the object.
		p.skipWS()
		if p.pos >= len(p.s) {
			return obj, nil
		}
		switch p.s[p.pos] {
		case ',':
			p.pos++
		case '}':
			p.pos++
			return obj, nil
		default:
			return obj, nil
		}
	}
}

func (p *partialParser) parseArray() (any, error) {
	p.pos++ // '['
	arr := []any{}
	for {
		p.skipWS()
		if p.pos >= len(p.s) {
			return arr, nil // truncated: close the array
		}
		if p.s[p.pos] == ']' {
			p.pos++
			return arr, nil
		}
		if p.s[p.pos] == ',' {
			p.pos++
			continue
		}
		val, err := p.parseValue()
		if err != nil {
			// incomplete element: drop it
			return arr, nil
		}
		arr = append(arr, val)
		p.skipWS()
		if p.pos >= len(p.s) {
			return arr, nil
		}
		switch p.s[p.pos] {
		case ',':
			p.pos++
		case ']':
			p.pos++
			return arr, nil
		default:
			return arr, nil
		}
	}
}

// ParseStreamingJSON attempts to parse potentially incomplete JSON during
// streaming. It always returns a valid object, even if the JSON is
// incomplete. Port of parseStreamingJson in json-parse.ts.
//
// Order of attempts: strict parse (with repair) → partial parse → partial
// parse of the repaired input → empty object. The TS version blindly casts
// non-object results to the target type; the Go port returns an empty object
// in that case because tool-call arguments are always objects.
func ParseStreamingJSON(partial string) map[string]any {
	if strings.TrimSpace(partial) == "" {
		return map[string]any{}
	}
	if v, err := ParseJSONWithRepair(partial); err == nil {
		if obj, ok := v.(map[string]any); ok {
			return obj
		}
		return map[string]any{}
	}
	if v, err := ParsePartialJSON(partial); err == nil && v != nil {
		if obj, ok := v.(map[string]any); ok {
			return obj
		}
		return map[string]any{}
	}
	if v, err := ParsePartialJSON(RepairJSON(partial)); err == nil && v != nil {
		if obj, ok := v.(map[string]any); ok {
			return obj
		}
	}
	return map[string]any{}
}

// streamingParseFloor is the smallest buffer growth that justifies re-parsing
// a tool call's arguments mid-stream. It is deliberately tiny: small tool calls
// (a path, a shell command) must still show a live preview from their first few
// deltas, and the quadratic cost this guards against only bites at sizes where
// the geometric term below has already taken over.
const streamingParseFloor = 8

// shouldReparseStreamingJSON reports whether accumulated tool-call arguments
// have grown enough to be worth re-parsing for the live preview.
//
// ParseStreamingJSON is O(n) in the whole accumulated buffer, so running it on
// every delta makes assembling one tool call O(n^2). Measured: a 54 KB `write`
// argument arriving in token-sized chunks cost 5.1 s of CPU and 2.5 GB of
// allocation, stalling the UI for the entire call. Growing the interval
// geometrically keeps the total parsing work linear in the argument size while
// still refreshing the preview often enough to look live -- roughly thirty
// times across a large call, while a small one still updates on nearly every
// delta.
//
// This only affects the intermediate value published in event.Partial. Both
// protocols re-parse authoritatively once the arguments are complete, so the
// final Arguments map is never approximate.
func shouldReparseStreamingJSON(parsedLen, currentLen int) bool {
	next := parsedLen + streamingParseFloor
	if geometric := parsedLen + parsedLen/4; geometric > next {
		next = geometric
	}
	return currentLen >= next
}
