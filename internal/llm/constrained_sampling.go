package llm

// Go port of reference/pi/packages/ai/src/api/constrained-sampling.ts.
//
// Tools may declare provider-side constrained sampling: either JSON-schema
// "strict" mode, or a grammar (lark/regex) that OpenAI exposes as a custom
// tool. Grammar tools take a single string input property, so their streamed
// input has to be re-encoded as JSON object deltas for pi's tool-call events.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// constrainedSamplingConfig is the parsed Tool.ConstrainedSampling payload.
type constrainedSamplingConfig struct {
	Type string `json:"type"` // "json_schema" | "grammar"
	// json_schema
	Strict string `json:"strict,omitempty"` // "prefer" | "require"
	// grammar: format name -> definition (openai_lark, openai_regex, ...).
	Variants map[string]string `json:"variants,omitempty"`
}

// parseConstrainedSampling decodes Tool.ConstrainedSampling. ok is false when
// absent or literal `false` (pi models "no config" as `false | Config`).
func parseConstrainedSampling(tool Tool) (constrainedSamplingConfig, bool) {
	raw := strings.TrimSpace(string(tool.ConstrainedSampling))
	if raw == "" || raw == "null" || raw == "false" {
		return constrainedSamplingConfig{}, false
	}
	var config constrainedSamplingConfig
	if err := json.Unmarshal(tool.ConstrainedSampling, &config); err != nil {
		return constrainedSamplingConfig{}, false
	}
	return config, config.Type != ""
}

// GrammarConstrainedSampling is a resolved grammar tool definition.
type GrammarConstrainedSampling struct {
	// Format is "lark" or "regex".
	Format string
	// Definition is the grammar source in that format.
	Definition string
	// InputProperty is the single required string property carrying the input.
	InputProperty string
}

// GrammarToolInputJSONBuffer tracks how much of a grammar tool's input has
// been re-encoded as a JSON object, so deltas stay monotonic.
type GrammarToolInputJSONBuffer struct {
	Input   string
	Started bool
	Closed  bool
}

// GetGrammarToolInput reads the grammar input property off a tool call's
// arguments, which must be a string.
func GetGrammarToolInput(toolName string, arguments map[string]any, inputProperty string) (string, error) {
	input, ok := arguments[inputProperty].(string)
	if !ok {
		return "", fmt.Errorf("Grammar tool call %q requires argument %q to be a string.", toolName, inputProperty)
	}
	return input, nil
}

// AppendGrammarToolInputJSONDelta advances buffer to nextInput and returns the
// JSON text to emit as a tool-call delta, so consumers see a well-formed
// `{"prop":"..."}` object being built. Returns "" (ok, no delta) when nothing
// changed. close emits the closing quote and brace.
func AppendGrammarToolInputJSONDelta(buffer *GrammarToolInputJSONBuffer, inputProperty, nextInput string, close bool) (string, error) {
	if buffer.Closed {
		if close && nextInput == buffer.Input {
			return "", nil
		}
		return "", fmt.Errorf("grammar tool input for property %q changed after it was closed", inputProperty)
	}
	if !strings.HasPrefix(nextInput, buffer.Input) {
		return "", fmt.Errorf("grammar tool input for property %q changed non-monotonically", inputProperty)
	}

	inputDelta := nextInput[len(buffer.Input):]
	if !close && inputDelta == "" {
		return "", nil
	}

	var delta strings.Builder
	if !buffer.Started {
		delta.WriteString("{" + jsonString(inputProperty) + ":\"")
		buffer.Started = true
	}
	// The delta goes inside an already-open JSON string, so strip the quotes
	// the encoder adds.
	encoded := jsonString(inputDelta)
	delta.WriteString(encoded[1 : len(encoded)-1])
	buffer.Input = nextInput

	if close {
		delta.WriteString("\"}")
		buffer.Closed = true
	}
	return delta.String(), nil
}

// jsonString encodes s as a JSON string literal, including quotes. Go escapes
// <, > and & by default in encoding/json; that is valid JSON but differs from
// JSON.stringify, so HTML escaping is disabled to keep deltas byte-identical
// to pi's.
func jsonString(s string) string {
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(s); err != nil {
		// Encoding a string cannot fail.
		return `""`
	}
	return strings.TrimSuffix(sb.String(), "\n")
}

// inferGrammarInputProperty finds the single required string property of the
// tool's object schema.
func inferGrammarInputProperty(tool Tool) (string, error) {
	var schema struct {
		Type       string                     `json:"type"`
		Properties map[string]json.RawMessage `json:"properties"`
		Required   []string                   `json:"required"`
	}
	if len(tool.Parameters) > 0 {
		if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
			return "", fmt.Errorf("grammar constrained sampling requires an object parameter schema")
		}
	}
	if schema.Type != "object" {
		return "", fmt.Errorf("grammar constrained sampling requires an object parameter schema")
	}
	if len(schema.Required) != 1 {
		return "", fmt.Errorf("grammar constrained sampling requires exactly one required string property")
	}
	inputProperty := schema.Required[0]
	propSchema, ok := schema.Properties[inputProperty]
	if !ok {
		return "", fmt.Errorf("grammar constrained sampling requires a properties entry for %s", inputProperty)
	}
	var prop struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(propSchema, &prop); err != nil || prop.Type != "string" {
		return "", fmt.Errorf("grammar constrained sampling property %s must have type string", inputProperty)
	}
	return inputProperty, nil
}

// ResolveJSONSchemaStrictSampling reports the `strict` value for a tool with
// json_schema constrained sampling. ok is false when the field should be
// omitted. Errors when the tool requires strict mode and the model lacks it.
func ResolveJSONSchemaStrictSampling(tool Tool, supportsStrictMode bool) (strict bool, ok bool, err error) {
	config, found := parseConstrainedSampling(tool)
	if !found || config.Type != "json_schema" {
		return false, false, nil
	}
	if supportsStrictMode {
		return true, true, nil
	}
	if config.Strict == "require" {
		return false, false, fmt.Errorf("Tool %q requires JSON-schema constrained sampling, but strict tools are unsupported.", tool.Name)
	}
	return false, false, nil
}

// ResolveGrammarConstrainedSampling resolves a tool's grammar config. ok is
// false when the tool has no grammar config, or when the model does not
// support OpenAI grammar tools (in which case the tool degrades to a plain
// function tool).
func ResolveGrammarConstrainedSampling(tool Tool, supportsOpenAIGrammarTools bool) (GrammarConstrainedSampling, bool, error) {
	config, found := parseConstrainedSampling(tool)
	if !found || config.Type != "grammar" {
		return GrammarConstrainedSampling{}, false, nil
	}
	if !supportsOpenAIGrammarTools {
		return GrammarConstrainedSampling{}, false, nil
	}

	lark := strings.TrimSpace(config.Variants["openai_lark"])
	regex := strings.TrimSpace(config.Variants["openai_regex"])
	if lark == "" && regex == "" {
		return GrammarConstrainedSampling{}, false, fmt.Errorf(
			"Tool %q cannot use grammar constrained sampling: no supported grammar variant was provided.", tool.Name)
	}

	inputProperty, err := inferGrammarInputProperty(tool)
	if err != nil {
		return GrammarConstrainedSampling{}, false, fmt.Errorf(
			"Tool %q cannot use grammar constrained sampling: %v.", tool.Name, err)
	}

	format, definition := "regex", config.Variants["openai_regex"]
	if lark != "" {
		format, definition = "lark", config.Variants["openai_lark"]
	}
	return GrammarConstrainedSampling{Format: format, Definition: definition, InputProperty: inputProperty}, true, nil
}

// CreateGrammarToolInputProperties maps tool name -> grammar input property
// for every grammar tool in tools.
func CreateGrammarToolInputProperties(tools []Tool, supportsOpenAIGrammarTools bool) (map[string]string, error) {
	properties := map[string]string{}
	for _, tool := range tools {
		grammar, ok, err := ResolveGrammarConstrainedSampling(tool, supportsOpenAIGrammarTools)
		if err != nil {
			return nil, err
		}
		if ok {
			properties[tool.Name] = grammar.InputProperty
		}
	}
	return properties, nil
}
