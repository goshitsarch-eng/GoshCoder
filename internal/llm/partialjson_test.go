package llm

import (
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
