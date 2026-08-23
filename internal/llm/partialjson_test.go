package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestParseStreamingJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  map[string]any
	}{
		{"empty", "", map[string]any{}},
		{"whitespace", "   \n ", map[string]any{}},
		{"complete object", `{"a": 1, "b": "x"}`, map[string]any{"a": 1.0, "b": "x"}},
		{"truncated string value", `{"a": "hel`, map[string]any{"a": "hel"}},
		{"truncated mid-key", `{"a": 1, "b`, map[string]any{"a": 1.0}},
		{"truncated before colon", `{"a": 1, "b"`, map[string]any{"a": 1.0}},
		{"truncated after colon", `{"a": 1, "b":`, map[string]any{"a": 1.0}},
		{"truncated array", `{"a": [1, 2,`, map[string]any{"a": []any{1.0, 2.0}}},
		{"truncated nested object", `{"a": {"b": {"c": tr`, map[string]any{"a": map[string]any{"b": map[string]any{}}}},
		{"integer number", `{"n": 12`, map[string]any{"n": 12.0}},
		{"truncated float", `{"n": 12.`, map[string]any{"n": 12.0}},
		{"truncated exponent", `{"n": 1.5e`, map[string]any{"n": 1.5}},
		{"lone minus dropped", `{"n": -`, map[string]any{}},
		{"complete bool", `{"ok": true`, map[string]any{"ok": true}},
		{"truncated literal dropped", `{"ok": tr`, map[string]any{}},
		{"null value", `{"x": null`, map[string]any{"x": nil}},
		{"trailing backslash in string", `{"a": "x\`, map[string]any{"a": "x"}},
		{"escape sequences", `{"a": "x\ny\t\"z"}`, map[string]any{"a": "x\ny\t\"z"}},
		{"incomplete unicode escape", `{"a": "x\u0`, map[string]any{"a": "x"}},
		{"complete unicode escape", `{"a": "xA"}`, map[string]any{"a": "xA"}},
		{"garbage", `not json at all`, map[string]any{}},
		{"array at top level", `[1, 2`, map[string]any{}}, // non-object result: empty object
		{"truncated after comma", `{"a": 1,`, map[string]any{"a": 1.0}},
		{"deep nesting truncated", `{"a": [{"b": [1, {"c": "d`, map[string]any{"a": []any{map[string]any{"b": []any{1.0, map[string]any{"c": "d"}}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseStreamingJSON(tt.input)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseStreamingJSON(%q) = %#v, want %#v", tt.input, got, tt.want)
			}
		})
	}
}

func TestRepairJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"raw newline in string", "{\"a\": \"x\ny\"}", `{"a": "x\ny"}`},
		{"raw tab in string", "{\"a\": \"x\ty\"}", `{"a": "x\ty"}`},
		{"invalid escape doubled", `{"a": "x\y"}`, `{"a": "x\\y"}`},
		{"trailing backslash doubled", `{"a": "x\`, `{"a": "x\\`},
		{"valid escape kept", `{"a": "x\ny"}`, `{"a": "x\ny"}`},
		{"unicode escape kept", `{"a": "xA"}`, `{"a": "xA"}`},
		// 'u' is a valid escape character; the hex check only applies when 4
		// hex digits follow, so an invalid \u escape passes through unchanged.
		{"short unicode escape kept", `{"a": "x\u0g"}`, `{"a": "x\u0g"}`},
		{"quote outside string untouched", `{"a": 1}`, `{"a": 1}`},
		{"control char outside string untouched", "{\"a\": 1,\n\"b\": 2}", "{\"a\": 1,\n\"b\": 2}"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RepairJSON(tt.input); got != tt.want {
				t.Errorf("RepairJSON(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestParseJSONWithRepair(t *testing.T) {
	// Strict parse succeeds without repair.
	v, err := ParseJSONWithRepair(`{"a": 1}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(v, map[string]any{"a": 1.0}) {
		t.Fatalf("got %#v", v)
	}
	// Raw newline inside a string is invalid JSON; repair fixes it.
	v, err = ParseJSONWithRepair("{\"a\": \"x\ny\"}")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(v, map[string]any{"a": "x\ny"}) {
		t.Fatalf("got %#v", v)
	}
	// Unrepairable input errors.
	if _, err := ParseJSONWithRepair(`{"a": }`); err == nil {
		t.Fatal("expected error")
	}
}

// TestPartialParserDecodesEscapes covers a disagreement between the two
// parsers: the strict path (ParseJSONWithRepair) decodes JSON escapes, while
// the tolerant partial path copied them verbatim. Streaming previews therefore
// showed literal backslash-n and backslash-quote, and any tool call that ended
// up going through the tolerant path was executed with the wrong string.
func TestPartialParserDecodesEscapes(t *testing.T) {
	cases := []struct {
		name    string
		partial string
		key     string
		want    string
	}{
		{"newline", `{"command":"echo one\ntwo`, "command", "echo one\ntwo"},
		{"quote", `{"command":"grep -n \"foo\" bar`, "command", `grep -n "foo" bar`},
		{"tab and backslash", `{"path":"a\tb\\c`, "path", "a\tb\\c"},
		{"forward slash", `{"path":"a\/b`, "path", "a/b"},
		{"unicode", `{"text":"café`, "text", "café"},
		{"surrogate pair", `{"text":"😀 hi`, "text", "\U0001F600 hi"},
		{"carriage return", `{"text":"a\r\nb`, "text", "a\r\nb"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			value, err := ParsePartialJSON(tc.partial)
			if err != nil {
				t.Fatalf("ParsePartialJSON: %v", err)
			}
			object, ok := value.(map[string]any)
			if !ok {
				t.Fatalf("value = %#v", value)
			}
			if got, _ := object[tc.key].(string); got != tc.want {
				t.Fatalf("%s = %q, want %q", tc.key, got, tc.want)
			}
		})
	}
}

// TestPartialParserAgreesWithStrictParser is the property behind the case
// above: for input that is complete and valid, both parsers must produce the
// same value.
func TestPartialParserAgreesWithStrictParser(t *testing.T) {
	complete := `{"command":"echo \"a\"\n\tb\\c","path":"café/😀.txt"}`

	strict, err := ParseJSONWithRepair(complete)
	if err != nil {
		t.Fatalf("ParseJSONWithRepair: %v", err)
	}
	tolerant, err := ParsePartialJSON(complete)
	if err != nil {
		t.Fatalf("ParsePartialJSON: %v", err)
	}
	strictJSON, _ := json.Marshal(strict)
	tolerantJSON, _ := json.Marshal(tolerant)
	if string(strictJSON) != string(tolerantJSON) {
		t.Fatalf("parsers disagree:\n strict:   %s\n tolerant: %s", strictJSON, tolerantJSON)
	}
}

// TestPartialParserWaitsForASurrogatePair keeps a half-arrived pair out of the
// preview instead of emitting a replacement character a later delta contradicts.
func TestPartialParserWaitsForASurrogatePair(t *testing.T) {
	value, err := ParsePartialJSON(`{"text":"hi \ud83d`)
	if err != nil {
		t.Fatalf("ParsePartialJSON: %v", err)
	}
	object, _ := value.(map[string]any)
	if got, _ := object["text"].(string); got != "hi " {
		t.Fatalf("text = %q, want the incomplete pair dropped", got)
	}
}
