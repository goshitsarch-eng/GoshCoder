package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// schemaTool builds a tool with the given JSON Schema parameters.
func schemaTool(schema string) Tool {
	return Tool{Name: "t", Description: "d", Parameters: json.RawMessage(schema)}
}

func TestValidateToolArgumentsAcceptsValidArgs(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"name":{"type":"string"},"count":{"type":"integer"}},"required":["name"]}`)
	got, err := validateToolArguments(tool, "t", map[string]any{"name": "x", "count": 2.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got["name"] != "x" || got["count"] != 2.0 {
		t.Fatalf("args = %#v", got)
	}
}

func TestValidateToolArgumentsNoSchema(t *testing.T) {
	// A tool without parameters accepts anything.
	tool := Tool{Name: "t", Description: "d"}
	got, err := validateToolArguments(tool, "t", map[string]any{"anything": 1})
	if err != nil || got["anything"] != 1 {
		t.Fatalf("args = %#v, err = %v", got, err)
	}

	// Nil args normalize to an empty map rather than staying nil.
	got, err = validateToolArguments(tool, "t", nil)
	if err != nil || got == nil || len(got) != 0 {
		t.Fatalf("args = %#v, err = %v", got, err)
	}
}

func TestValidateToolArgumentsMissingRequired(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`)
	_, err := validateToolArguments(tool, "my_tool", map[string]any{})
	if err == nil {
		t.Fatal("expected a validation error")
	}
	// The message keeps pi's shape: tool name, bullet list, received args.
	if !strings.Contains(err.Error(), `Validation failed for tool "my_tool"`) {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(err.Error(), "name: required property missing") {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(err.Error(), "Received arguments:") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsTypeMismatch(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"flag":{"type":"array"}}}`)
	_, err := validateToolArguments(tool, "t", map[string]any{"flag": "not-an-array"})
	if err == nil {
		t.Fatal("expected a validation error")
	}
	if !strings.Contains(err.Error(), "flag: expected array") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsCoercesPrimitives(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		args   map[string]any
		check  func(t *testing.T, got map[string]any)
	}{
		{
			name:   "string to number",
			schema: `{"type":"object","properties":{"n":{"type":"number"}}}`,
			args:   map[string]any{"n": "1.5"},
			check: func(t *testing.T, got map[string]any) {
				if got["n"] != 1.5 {
					t.Fatalf("n = %#v", got["n"])
				}
			},
		},
		{
			name:   "string to integer",
			schema: `{"type":"object","properties":{"n":{"type":"integer"}}}`,
			args:   map[string]any{"n": "42"},
			check: func(t *testing.T, got map[string]any) {
				if got["n"] != 42.0 {
					t.Fatalf("n = %#v", got["n"])
				}
			},
		},
		{
			name:   "string to boolean",
			schema: `{"type":"object","properties":{"b":{"type":"boolean"}}}`,
			args:   map[string]any{"b": "true"},
			check: func(t *testing.T, got map[string]any) {
				if got["b"] != true {
					t.Fatalf("b = %#v", got["b"])
				}
			},
		},
		{
			name:   "number to string",
			schema: `{"type":"object","properties":{"s":{"type":"string"}}}`,
			args:   map[string]any{"s": 7.0},
			check: func(t *testing.T, got map[string]any) {
				if got["s"] != "7" {
					t.Fatalf("s = %#v", got["s"])
				}
			},
		},
		{
			name:   "nested object properties",
			schema: `{"type":"object","properties":{"o":{"type":"object","properties":{"n":{"type":"number"}}}}}`,
			args:   map[string]any{"o": map[string]any{"n": "3"}},
			check: func(t *testing.T, got map[string]any) {
				nested := got["o"].(map[string]any)
				if nested["n"] != 3.0 {
					t.Fatalf("o.n = %#v", nested["n"])
				}
			},
		},
		{
			name:   "array items",
			schema: `{"type":"object","properties":{"a":{"type":"array","items":{"type":"number"}}}}`,
			args:   map[string]any{"a": []any{"1", "2"}},
			check: func(t *testing.T, got map[string]any) {
				items := got["a"].([]any)
				if items[0] != 1.0 || items[1] != 2.0 {
					t.Fatalf("a = %#v", items)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := validateToolArguments(schemaTool(tt.schema), "t", tt.args)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tt.check(t, got)
		})
	}
}

// TestValidateToolArgumentsDoesNotMutateInput confirms coercion works on a
// deep copy, so the caller's arguments are untouched.
func TestValidateToolArgumentsDoesNotMutateInput(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"n":{"type":"number"},"o":{"type":"object","properties":{"m":{"type":"number"}}}}}`)
	args := map[string]any{"n": "1", "o": map[string]any{"m": "2"}}
	got, err := validateToolArguments(tool, "t", args)
	if err != nil {
		t.Fatal(err)
	}
	if got["n"] != 1.0 {
		t.Fatalf("validated n = %#v", got["n"])
	}
	if args["n"] != "1" {
		t.Fatalf("input was mutated: %#v", args["n"])
	}
	if nested := args["o"].(map[string]any); nested["m"] != "2" {
		t.Fatalf("nested input was mutated: %#v", nested["m"])
	}
}

func TestValidateToolArgumentsEnumAndConst(t *testing.T) {
	enumTool := schemaTool(`{"type":"object","properties":{"mode":{"enum":["a","b"]}}}`)
	if _, err := validateToolArguments(enumTool, "t", map[string]any{"mode": "a"}); err != nil {
		t.Fatalf("valid enum value rejected: %v", err)
	}
	_, err := validateToolArguments(enumTool, "t", map[string]any{"mode": "c"})
	if err == nil || !strings.Contains(err.Error(), "must be one of") {
		t.Fatalf("err = %v", err)
	}

	constTool := schemaTool(`{"type":"object","properties":{"v":{"const":"fixed"}}}`)
	if _, err := validateToolArguments(constTool, "t", map[string]any{"v": "fixed"}); err != nil {
		t.Fatalf("valid const value rejected: %v", err)
	}
	_, err = validateToolArguments(constTool, "t", map[string]any{"v": "other"})
	if err == nil || !strings.Contains(err.Error(), "must be fixed") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsUnionTypes(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"v":{"type":["string","number"]}}}`)
	// Both union members are accepted as-is.
	for _, value := range []any{"text", 5.0} {
		if _, err := validateToolArguments(tool, "t", map[string]any{"v": value}); err != nil {
			t.Fatalf("union member %#v rejected: %v", value, err)
		}
	}
	_, err := validateToolArguments(tool, "t", map[string]any{"v": []any{}})
	if err == nil || !strings.Contains(err.Error(), "expected string or number") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsAnyOf(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"v":{"anyOf":[{"type":"string"},{"type":"boolean"}]}}}`)
	if _, err := validateToolArguments(tool, "t", map[string]any{"v": true}); err != nil {
		t.Fatalf("anyOf member rejected: %v", err)
	}
	_, err := validateToolArguments(tool, "t", map[string]any{"v": []any{1}})
	if err == nil || !strings.Contains(err.Error(), "does not match any anyOf schema") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsNestedPaths(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"outer":{"type":"object","properties":{"inner":{"type":"string"}},"required":["inner"]}}}`)
	_, err := validateToolArguments(tool, "t", map[string]any{"outer": map[string]any{}})
	if err == nil {
		t.Fatal("expected a validation error")
	}
	// Nested paths are dotted so the model can see which field failed.
	if !strings.Contains(err.Error(), "outer.inner: required property missing") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsArrayItemPaths(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"a":{"type":"array","items":{"type":"object","properties":{"n":{"type":"number"}}}}}}`)
	_, err := validateToolArguments(tool, "t", map[string]any{"a": []any{map[string]any{"n": []any{}}}})
	if err == nil {
		t.Fatal("expected a validation error")
	}
	if !strings.Contains(err.Error(), "a[0].n: expected number") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsInvalidSchema(t *testing.T) {
	tool := Tool{Name: "t", Parameters: json.RawMessage(`{not json`)}
	_, err := validateToolArguments(tool, "t", map[string]any{})
	if err == nil || !strings.Contains(err.Error(), "invalid parameters schema") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateToolArgumentsSortsErrors(t *testing.T) {
	tool := schemaTool(`{"type":"object","properties":{"b":{"type":"string"},"a":{"type":"string"}},"required":["a","b"]}`)
	_, err := validateToolArguments(tool, "t", map[string]any{})
	if err == nil {
		t.Fatal("expected a validation error")
	}
	// Errors are sorted, so the message is stable across runs.
	message := err.Error()
	aIndex := strings.Index(message, "a: required property missing")
	bIndex := strings.Index(message, "b: required property missing")
	if aIndex < 0 || bIndex < 0 || aIndex > bIndex {
		t.Fatalf("errors are not sorted: %v", message)
	}
}
