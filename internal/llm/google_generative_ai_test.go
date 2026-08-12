package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testGoogleModel(baseURL string) *Model {
	return &Model{
		ID:            "gemini-2.5-flash",
		Name:          "Gemini 2.5 Flash",
		API:           APIGoogleGenerativeAI,
		Provider:      "google",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 1000000,
		MaxTokens:     65536,
	}
}

// googleSSE formats one Gemini SSE record. The REST stream sends bare data
// lines with whole GenerateContentResponse objects.
func googleSSE(data string) string {
	return "data: " + data + "\n\n"
}

// googleFinish is a minimal terminal chunk carrying a finish reason and usage.
const googleFinish = `{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`

func googleServer(t *testing.T, body string) *Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return testGoogleModel(srv.URL)
}

// captureGoogleRequest runs a stream against a recording server and returns the
// decoded request body, headers, and request URI.
func captureGoogleRequest(t *testing.T, model *Model, conv *Context, opts *GoogleOptions) (map[string]any, http.Header, string) {
	t.Helper()
	var captured map[string]any
	var headers http.Header
	var requestURI string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		requestURI = r.URL.RequestURI()
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, googleSSE(googleFinish))
	}))
	defer srv.Close()
	model.BaseURL = srv.URL
	drainEvents(StreamGoogleGenerativeAI(model, conv, opts))
	return captured, headers, requestURI
}

func TestGoogleStreamTextOnly(t *testing.T) {
	model := googleServer(t,
		googleSSE(`{"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}`)+
			googleSSE(`{"candidates":[{"content":{"parts":[{"text":" world"}]}}]}`)+
			googleSSE(googleFinish))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	want := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	if events[2].Delta != "Hello" || events[3].Delta != " world" {
		t.Fatalf("deltas = %q, %q", events[2].Delta, events[3].Delta)
	}
	if events[4].Content != "Hello world" {
		t.Fatalf("text_end content = %q", events[4].Content)
	}

	final := events[len(events)-1]
	if final.Reason != StopStop {
		t.Fatalf("reason = %q", final.Reason)
	}
	if final.Message.ResponseID != "resp_1" {
		t.Fatalf("responseId = %q", final.Message.ResponseID)
	}
	// promptTokenCount excludes cached tokens here; thoughts count as output.
	if final.Message.Usage.Input != 10 || final.Message.Usage.Output != 5 || final.Message.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %#v", final.Message.Usage)
	}
}

func TestGoogleStreamThinkingThenText(t *testing.T) {
	model := googleServer(t,
		googleSSE(`{"candidates":[{"content":{"parts":[{"text":"pondering","thought":true,"thoughtSignature":"c2ln"}]}}]}`)+
			googleSSE(`{"candidates":[{"content":{"parts":[{"text":"answer"}]}}]}`)+
			googleSSE(googleFinish))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	// The kind change from thinking to text closes the thinking block.
	want := []EventType{EventStart, EventThinkingStart, EventThinkingDelta, EventThinkingEnd, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	final := events[len(events)-1]
	thinking := final.Message.Content[0].(ThinkingContent)
	if thinking.Thinking != "pondering" || thinking.ThinkingSignature != "c2ln" {
		t.Fatalf("thinking = %#v", thinking)
	}
	if text := final.Message.Content[1].(TextContent); text.Text != "answer" {
		t.Fatalf("text = %#v", text)
	}
}

// TestGoogleRetainsThoughtSignatureAcrossDeltas covers backends that only send
// thoughtSignature on the first delta of a part.
func TestGoogleRetainsThoughtSignatureAcrossDeltas(t *testing.T) {
	model := googleServer(t,
		googleSSE(`{"candidates":[{"content":{"parts":[{"text":"a","thought":true,"thoughtSignature":"c2ln"}]}}]}`)+
			googleSSE(`{"candidates":[{"content":{"parts":[{"text":"b","thought":true}]}}]}`)+
			googleSSE(googleFinish))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	thinking := events[len(events)-1].Message.Content[0].(ThinkingContent)
	if thinking.Thinking != "ab" {
		t.Fatalf("thinking = %q", thinking.Thinking)
	}
	// The later signature-less delta must not clear the retained signature.
	if thinking.ThinkingSignature != "c2ln" {
		t.Fatalf("thinkingSignature = %q, want retained", thinking.ThinkingSignature)
	}
}

func TestGoogleStreamToolCall(t *testing.T) {
	model := googleServer(t,
		googleSSE(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"},"id":"call_1"}}]}}]}`)+
			googleSSE(`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "weather?"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	// Google delivers complete calls, so start/delta/end are emitted together.
	want := []EventType{EventStart, EventToolCallStart, EventToolCallDelta, EventToolCallEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	final := events[len(events)-1]
	// A tool call overrides the STOP finish reason.
	if final.Reason != StopToolUse {
		t.Fatalf("reason = %q", final.Reason)
	}
	tc := final.Message.Content[0].(ToolCall)
	if tc.ID != "call_1" || tc.Name != "get_weather" || tc.Arguments["location"] != "Paris" {
		t.Fatalf("toolCall = %#v", tc)
	}
}

// TestGoogleGeneratesMissingToolCallIDs covers Google omitting the id, and
// repeating one, both of which need a fresh unique id.
func TestGoogleGeneratesMissingToolCallIDs(t *testing.T) {
	model := googleServer(t,
		googleSSE(`{"candidates":[{"content":{"parts":[{"functionCall":{"name":"f","args":{}}},{"functionCall":{"name":"g","args":{},"id":"dup"}},{"functionCall":{"name":"h","args":{},"id":"dup"}}]}}]}`)+
			googleSSE(googleFinish))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "go"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	content := events[len(events)-1].Message.Content
	if len(content) != 3 {
		t.Fatalf("content = %#v", content)
	}
	first := content[0].(ToolCall)
	second := content[1].(ToolCall)
	third := content[2].(ToolCall)

	if first.ID == "" || !strings.HasPrefix(first.ID, "f_") {
		t.Fatalf("generated id = %q, want an f_ prefixed id", first.ID)
	}
	if second.ID != "dup" {
		t.Fatalf("first use of an explicit id should be kept, got %q", second.ID)
	}
	// The duplicate is replaced so tool results stay unambiguous.
	if third.ID == "dup" {
		t.Fatalf("duplicate id was not replaced: %q", third.ID)
	}
}

func TestGoogleStopReasons(t *testing.T) {
	tests := []struct {
		finishReason string
		wantReason   StopReason
	}{
		{"STOP", StopStop},
		{"MAX_TOKENS", StopLength},
	}
	for _, tt := range tests {
		t.Run(tt.finishReason, func(t *testing.T) {
			model := googleServer(t, googleSSE(fmt.Sprintf(`{"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":%q}]}`, tt.finishReason)))
			conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
			events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

			final := events[len(events)-1]
			if final.Reason != tt.wantReason {
				t.Fatalf("reason = %q, want %q", final.Reason, tt.wantReason)
			}
			if final.Message.RawStopReason != tt.finishReason {
				t.Fatalf("rawStopReason = %q", final.Message.RawStopReason)
			}
		})
	}
}

// TestGoogleSafetyFinishReasonErrors covers a blocked response: Google reports
// failures as a finishReason, not an error body.
func TestGoogleSafetyFinishReasonErrors(t *testing.T) {
	model := googleServer(t, googleSSE(`{"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}]}`))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError || final.Reason != StopError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "provider stopped with: SAFETY") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestGoogleStreamWithoutFinishReasonErrors(t *testing.T) {
	model := googleServer(t, googleSSE(`{"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}`))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "without a finish reason") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestGoogleUsageSubtractsCachedTokens(t *testing.T) {
	model := googleServer(t,
		googleSSE(`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":40,"candidatesTokenCount":20,"thoughtsTokenCount":8,"totalTokenCount":128}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	usage := events[len(events)-1].Message.Usage
	if usage.Input != 60 || usage.CacheRead != 40 {
		t.Fatalf("input/cacheRead = %d/%d, want 60/40", usage.Input, usage.CacheRead)
	}
	// Thoughts are billed as output.
	if usage.Output != 28 {
		t.Fatalf("output = %d, want 28", usage.Output)
	}
	if usage.Reasoning == nil || *usage.Reasoning != 8 {
		t.Fatalf("reasoning = %#v", usage.Reasoning)
	}
}

func TestGoogleRequestURLAndAuth(t *testing.T) {
	model := testGoogleModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	_, headers, requestURI := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "secret"},
	})

	if !strings.HasPrefix(requestURI, "/models/gemini-2.5-flash:streamGenerateContent") {
		t.Fatalf("request path = %q", requestURI)
	}
	if !strings.Contains(requestURI, "alt=sse") {
		t.Fatalf("request query = %q, want alt=sse", requestURI)
	}
	if headers.Get("x-goog-api-key") != "secret" {
		t.Fatalf("x-goog-api-key = %q", headers.Get("x-goog-api-key"))
	}
}

func TestGoogleBuildParams(t *testing.T) {
	model := testGoogleModel("")
	temperature := 0.4
	conv := &Context{
		SystemPrompt: "be brief",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k", MaxTokens: 2048, Temperature: &temperature},
	})

	// The REST body puts generation settings under generationConfig, not the
	// SDK's "config".
	generationConfig := captured["generationConfig"].(map[string]any)
	if generationConfig["maxOutputTokens"] != 2048.0 || generationConfig["temperature"] != 0.4 {
		t.Fatalf("generationConfig = %#v", generationConfig)
	}
	if _, hasConfig := captured["config"]; hasConfig {
		t.Fatalf("config should be flattened for REST: %#v", captured["config"])
	}

	systemInstruction := captured["systemInstruction"].(map[string]any)
	part := systemInstruction["parts"].([]any)[0].(map[string]any)
	if part["text"] != "be brief" {
		t.Fatalf("systemInstruction = %#v", systemInstruction)
	}

	contents := captured["contents"].([]any)
	user := contents[0].(map[string]any)
	userPart := user["parts"].([]any)[0].(map[string]any)
	if user["role"] != "user" || userPart["text"] != "hi" {
		t.Fatalf("contents = %#v", contents)
	}
}

func TestGoogleToolConversion(t *testing.T) {
	model := testGoogleModel("")
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	tools := captured["tools"].([]any)
	declarations := tools[0].(map[string]any)["functionDeclarations"].([]any)
	declaration := declarations[0].(map[string]any)
	if declaration["name"] != "get_weather" || declaration["description"] != "Get weather" {
		t.Fatalf("declaration = %#v", declaration)
	}
	// Full JSON Schema goes through parametersJsonSchema by default.
	schema, ok := declaration["parametersJsonSchema"].(map[string]any)
	if !ok || schema["type"] != "object" {
		t.Fatalf("parametersJsonSchema = %#v", declaration["parametersJsonSchema"])
	}
	if _, hasLegacy := declaration["parameters"]; hasLegacy {
		t.Fatalf("legacy parameters should be absent: %#v", declaration)
	}
	// No explicit tool choice and no strict tools: the mode is omitted.
	if _, hasToolConfig := captured["toolConfig"]; hasToolConfig {
		t.Fatalf("toolConfig should be omitted: %#v", captured["toolConfig"])
	}
}

func TestGoogleLegacyParametersSanitizesMetaKeywords(t *testing.T) {
	tools := []Tool{{
		Name:        "t",
		Description: "d",
		Parameters:  json.RawMessage(`{"$schema":"https://json-schema.org/draft/2020-12/schema","$defs":{"x":{}},"type":"object","properties":{"a":{"$comment":"drop me","type":"string"}}}`),
	}}
	converted := convertGoogleTools(tools, true)

	declaration := converted[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	schema := declaration["parameters"].(map[string]any)
	for _, key := range []string{"$schema", "$defs"} {
		if _, present := schema[key]; present {
			t.Fatalf("%s should be stripped for OpenAPI: %#v", key, schema)
		}
	}
	if schema["type"] != "object" {
		t.Fatalf("schema = %#v", schema)
	}
	// Sanitizing recurses into nested schemas.
	properties := schema["properties"].(map[string]any)
	nested := properties["a"].(map[string]any)
	if _, present := nested["$comment"]; present {
		t.Fatalf("$comment should be stripped from nested schemas: %#v", nested)
	}
}

func TestGoogleToolChoiceModes(t *testing.T) {
	tests := []struct {
		choice   string
		wantMode string
	}{
		{"auto", googleFunctionCallingAuto},
		{"none", googleFunctionCallingNone},
		{"any", googleFunctionCallingAny},
	}
	for _, tt := range tests {
		t.Run(tt.choice, func(t *testing.T) {
			model := testGoogleModel("")
			conv := &Context{
				Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
				Tools:    []Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
			}
			captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
				StreamOptions: StreamOptions{APIKey: "k"},
				ToolChoice:    tt.choice,
			})

			toolConfig := captured["toolConfig"].(map[string]any)
			functionCallingConfig := toolConfig["functionCallingConfig"].(map[string]any)
			if functionCallingConfig["mode"] != tt.wantMode {
				t.Fatalf("mode = %#v, want %q", functionCallingConfig["mode"], tt.wantMode)
			}
		})
	}
}

// TestGoogleStrictToolUsesValidatedMode covers Gemini 3+, which enforces
// required parameters through VALIDATED mode.
func TestGoogleStrictToolUsesValidatedMode(t *testing.T) {
	model := testGoogleModel("")
	model.ID = "gemini-3-pro"
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:                "t",
			Description:         "d",
			Parameters:          json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}`),
			ConstrainedSampling: json.RawMessage(`{"type":"json_schema","strict":"prefer"}`),
		}},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	functionCallingConfig := captured["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if functionCallingConfig["mode"] != googleFunctionCallingValidated {
		t.Fatalf("mode = %#v, want VALIDATED", functionCallingConfig["mode"])
	}
}

func TestGoogleRequiresToolCallID(t *testing.T) {
	tests := []struct {
		modelID string
		want    bool
	}{
		{"gemini-3-pro", true},
		{"gemini-3.1-pro", true},
		{"gemini-live-3-flash", true},
		{"gemini-2.5-flash", false},
		{"gemini-2.5-pro", false},
		{"claude-sonnet-4-5", true},
		{"gpt-oss-120b", true},
		{"gemma-4-27b", false},
	}
	for _, tt := range tests {
		if got := RequiresToolCallID(tt.modelID); got != tt.want {
			t.Errorf("RequiresToolCallID(%q) = %v, want %v", tt.modelID, got, tt.want)
		}
	}
}

func TestGoogleReplaysAssistantAndToolResult(t *testing.T) {
	model := testGoogleModel("")
	// Gemini 3 requires explicit tool call ids on calls and responses.
	model.ID = "gemini-3-pro"
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "c2ln"},
					ToolCall{Type: "toolCall", ID: "call_1", Name: "do_thing", Arguments: map[string]any{"x": 1.0}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1", ToolName: "do_thing", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	contents := captured["contents"].([]any)
	if len(contents) != 3 {
		t.Fatalf("contents = %#v", contents)
	}

	assistant := contents[1].(map[string]any)
	if assistant["role"] != "model" {
		t.Fatalf("assistant role = %#v", assistant["role"])
	}
	parts := assistant["parts"].([]any)
	thought := parts[0].(map[string]any)
	if thought["thought"] != true || thought["text"] != "hmm" || thought["thoughtSignature"] != "c2ln" {
		t.Fatalf("thinking part = %#v", thought)
	}
	call := parts[1].(map[string]any)["functionCall"].(map[string]any)
	if call["name"] != "do_thing" || call["id"] != "call_1" {
		t.Fatalf("functionCall = %#v", call)
	}

	// Tool results come back as a user turn with a functionResponse.
	result := contents[2].(map[string]any)
	response := result["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if result["role"] != "user" || response["name"] != "do_thing" || response["id"] != "call_1" {
		t.Fatalf("functionResponse = %#v", response)
	}
	if output := response["response"].(map[string]any); output["output"] != "ok" {
		t.Fatalf("response payload = %#v", output)
	}
}

// TestGoogleToolResultErrorUsesErrorKey covers the success/error key split.
func TestGoogleToolResultErrorUsesErrorKey(t *testing.T) {
	model := testGoogleModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content:    []ContentBlock{ToolCall{Type: "toolCall", ID: "c1", Name: "f", Arguments: map[string]any{}}},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "c1", ToolName: "f", IsError: true, Content: []ContentBlock{TextContent{Type: "text", Text: "boom"}}},
		},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	contents := captured["contents"].([]any)
	response := contents[len(contents)-1].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	payload := response["response"].(map[string]any)
	if payload["error"] != "boom" {
		t.Fatalf("payload = %#v, want an error key", payload)
	}
	// Gemini 2.5 does not require ids, so none is sent.
	if _, hasID := response["id"]; hasID {
		t.Fatalf("id should be omitted for this model: %#v", response)
	}
}

// TestGoogleMergesConsecutiveToolResults covers the Cloud Code Assist rule that
// all function responses share one user turn.
func TestGoogleMergesConsecutiveToolResults(t *testing.T) {
	model := testGoogleModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ToolCall{Type: "toolCall", ID: "c1", Name: "f", Arguments: map[string]any{}},
					ToolCall{Type: "toolCall", ID: "c2", Name: "g", Arguments: map[string]any{}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "c1", ToolName: "f", Content: []ContentBlock{TextContent{Type: "text", Text: "one"}}},
			ToolResultMessage{Role: "toolResult", ToolCallID: "c2", ToolName: "g", Content: []ContentBlock{TextContent{Type: "text", Text: "two"}}},
		},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	contents := captured["contents"].([]any)
	// user, model, and a single merged user turn.
	if len(contents) != 3 {
		t.Fatalf("contents = %d turns, want 3: %#v", len(contents), contents)
	}
	parts := contents[2].(map[string]any)["parts"].([]any)
	if len(parts) != 2 {
		t.Fatalf("merged parts = %#v", parts)
	}
}

// TestGoogleCrossModelThinkingBecomesText covers replay from another model:
// the signature is unusable, so thinking is downgraded to plain text.
func TestGoogleCrossModelThinkingBecomesText(t *testing.T) {
	model := testGoogleModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: APIAnthropicMessages, Provider: "anthropic", Model: "claude-test",
				StopReason: StopStop,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "c2ln"},
					TextContent{Type: "text", Text: "answer"},
				},
			},
		},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	parts := captured["contents"].([]any)[1].(map[string]any)["parts"].([]any)
	first := parts[0].(map[string]any)
	if first["text"] != "hmm" {
		t.Fatalf("downgraded thinking = %#v", first)
	}
	// No thought marker and no signature survive a cross-model replay.
	if _, hasThought := first["thought"]; hasThought {
		t.Fatalf("thought marker should be dropped: %#v", first)
	}
	if _, hasSignature := first["thoughtSignature"]; hasSignature {
		t.Fatalf("signature should be dropped: %#v", first)
	}
}

// TestGoogleDropsInvalidThoughtSignature covers the base64 requirement: Google
// types signatures as TYPE_BYTES and rejects anything else.
func TestGoogleDropsInvalidThoughtSignature(t *testing.T) {
	model := testGoogleModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopStop,
				Content: []ContentBlock{
					// "not base64!" is neither a multiple of 4 nor base64.
					TextContent{Type: "text", Text: "answer", TextSignature: "not base64!"},
				},
			},
		},
	}
	captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	part := captured["contents"].([]any)[1].(map[string]any)["parts"].([]any)[0].(map[string]any)
	if _, hasSignature := part["thoughtSignature"]; hasSignature {
		t.Fatalf("invalid signature should be dropped: %#v", part)
	}
}

func TestGoogleThinkingConfig(t *testing.T) {
	t.Run("budget model gets thinkingBudget", func(t *testing.T) {
		model := testGoogleModel("")
		model.Reasoning = true
		budget := 4096
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			Thinking:      &GoogleThinkingConfig{Enabled: true, BudgetTokens: &budget},
		})

		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["includeThoughts"] != true || thinkingConfig["thinkingBudget"] != 4096.0 {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})

	t.Run("level takes precedence over budget", func(t *testing.T) {
		model := testGoogleModel("")
		model.Reasoning = true
		budget := 4096
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			Thinking:      &GoogleThinkingConfig{Enabled: true, Level: GoogleThinkingLevelHigh, BudgetTokens: &budget},
		})

		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["thinkingLevel"] != GoogleThinkingLevelHigh {
			t.Fatalf("thinkingLevel = %#v", thinkingConfig["thinkingLevel"])
		}
		if _, hasBudget := thinkingConfig["thinkingBudget"]; hasBudget {
			t.Fatalf("thinkingBudget should be omitted when a level is set: %#v", thinkingConfig)
		}
	})

	t.Run("non-reasoning model omits thinkingConfig", func(t *testing.T) {
		model := testGoogleModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		captured, _, _ := captureGoogleRequest(t, model, conv, &GoogleOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			Thinking:      &GoogleThinkingConfig{Enabled: true, Level: GoogleThinkingLevelHigh},
		})

		if generationConfig, ok := captured["generationConfig"].(map[string]any); ok {
			if _, hasThinking := generationConfig["thinkingConfig"]; hasThinking {
				t.Fatalf("thinkingConfig should be omitted: %#v", generationConfig)
			}
		}
	})
}

func TestGoogleDisabledThinkingConfig(t *testing.T) {
	tests := []struct {
		modelID  string
		wantKey  string
		wantVal  any
		wantName string
	}{
		// Gemini 3 Pro cannot disable thinking, so it uses the lowest level.
		{"gemini-3-pro", "thinkingLevel", GoogleThinkingLevelLow, "gemini 3 pro"},
		{"gemini-3-flash", "thinkingLevel", GoogleThinkingLevelMinimal, "gemini 3 flash"},
		{"gemini-flash-latest", "thinkingLevel", GoogleThinkingLevelMinimal, "gemini flash latest"},
		{"gemma-4-27b", "thinkingLevel", GoogleThinkingLevelMinimal, "gemma 4"},
		// Gemini 2.x disables via a zero budget.
		{"gemini-2.5-flash", "thinkingBudget", 0, "gemini 2.5"},
	}
	for _, tt := range tests {
		t.Run(tt.wantName, func(t *testing.T) {
			model := testGoogleModel("")
			model.ID = tt.modelID
			got := getDisabledGoogleThinkingConfig(model)
			if got[tt.wantKey] != tt.wantVal {
				t.Fatalf("config = %#v, want %s=%v", got, tt.wantKey, tt.wantVal)
			}
		})
	}
}

func TestGoogleThinkingLevelMapping(t *testing.T) {
	tests := []struct {
		modelID string
		effort  ThinkingLevel
		want    GoogleThinkingLevel
	}{
		// Gemini 3 Pro exposes only LOW and HIGH.
		{"gemini-3-pro", ThinkingMinimal, GoogleThinkingLevelLow},
		{"gemini-3-pro", ThinkingLow, GoogleThinkingLevelLow},
		{"gemini-3-pro", ThinkingMedium, GoogleThinkingLevelHigh},
		{"gemini-3-pro", ThinkingHigh, GoogleThinkingLevelHigh},
		{"gemma-4-27b", ThinkingLow, GoogleThinkingLevelMinimal},
		{"gemma-4-27b", ThinkingMedium, GoogleThinkingLevelHigh},
		// Other models pass levels through one-to-one.
		{"gemini-3-flash", ThinkingMinimal, GoogleThinkingLevelMinimal},
		{"gemini-3-flash", ThinkingMedium, GoogleThinkingLevelMedium},
		{"gemini-3-flash", ThinkingHigh, GoogleThinkingLevelHigh},
	}
	for _, tt := range tests {
		model := testGoogleModel("")
		model.ID = tt.modelID
		if got := getGoogleThinkingLevel(tt.effort, model); got != tt.want {
			t.Errorf("getGoogleThinkingLevel(%q, %q) = %q, want %q", tt.effort, tt.modelID, got, tt.want)
		}
	}
}

func TestGoogleBudgets(t *testing.T) {
	tests := []struct {
		modelID string
		effort  ThinkingLevel
		want    int
	}{
		{"gemini-2.5-pro", ThinkingMinimal, 128},
		{"gemini-2.5-pro", ThinkingHigh, 32768},
		{"gemini-2.5-flash-lite", ThinkingMinimal, 512},
		{"gemini-2.5-flash-lite", ThinkingHigh, 24576},
		{"gemini-2.5-flash", ThinkingMinimal, 128},
		{"gemini-2.5-flash", ThinkingMedium, 8192},
		// Unknown models delegate the budget to the model (-1 = dynamic).
		{"gemini-9-experimental", ThinkingHigh, -1},
	}
	for _, tt := range tests {
		model := testGoogleModel("")
		model.ID = tt.modelID
		if got := getGoogleBudget(model, tt.effort, nil); got != tt.want {
			t.Errorf("getGoogleBudget(%q, %q) = %d, want %d", tt.modelID, tt.effort, got, tt.want)
		}
	}

	// Custom budgets win over the built-in table.
	model := testGoogleModel("")
	model.ID = "gemini-2.5-pro"
	if got := getGoogleBudget(model, ThinkingHigh, &ThinkingBudgets{High: 999}); got != 999 {
		t.Fatalf("custom budget = %d, want 999", got)
	}
}

func TestGoogleStreamSimpleThinking(t *testing.T) {
	t.Run("no reasoning disables thinking", func(t *testing.T) {
		model := testGoogleModel("")
		model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

		var captured map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&captured)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, googleSSE(googleFinish))
		}))
		defer srv.Close()
		model.BaseURL = srv.URL

		drainEvents(StreamGoogleGenerativeAISimple(model, conv, &SimpleStreamOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
		}))

		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		// A 2.5 model disables thinking with a zero budget.
		if thinkingConfig["thinkingBudget"] != 0.0 {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})

	t.Run("gemini 3 uses a thinking level", func(t *testing.T) {
		model := testGoogleModel("")
		model.ID = "gemini-3-pro"
		model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

		var captured map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&captured)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, googleSSE(googleFinish))
		}))
		defer srv.Close()
		model.BaseURL = srv.URL

		drainEvents(StreamGoogleGenerativeAISimple(model, conv, &SimpleStreamOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			Reasoning:     ThinkingHigh,
		}))

		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["includeThoughts"] != true || thinkingConfig["thinkingLevel"] != GoogleThinkingLevelHigh {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})

	t.Run("2.5 model uses a token budget", func(t *testing.T) {
		model := testGoogleModel("")
		model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

		var captured map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&captured)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, googleSSE(googleFinish))
		}))
		defer srv.Close()
		model.BaseURL = srv.URL

		drainEvents(StreamGoogleGenerativeAISimple(model, conv, &SimpleStreamOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			Reasoning:     ThinkingMedium,
		}))

		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["includeThoughts"] != true || thinkingConfig["thinkingBudget"] != 8192.0 {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})
}

func TestGoogleNoAPIKeyErrors(t *testing.T) {
	model := testGoogleModel("http://127.0.0.1:1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleGenerativeAI(model, conv, &GoogleOptions{}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "no API key") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestGoogleRegisteredInRegistry(t *testing.T) {
	funcs, ok := GetStreamer(APIGoogleGenerativeAI)
	if !ok {
		t.Fatal("google-generative-ai is not registered")
	}
	if funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("registration = %#v", funcs)
	}
}

// TestGoogleImageToolResultNestsForGemini3 covers the multimodal split: Gemini
// 3+ nests images in functionResponse.parts, older models need a separate turn.
func TestGoogleImageToolResultNestsForGemini3(t *testing.T) {
	buildConv := func(model *Model) *Context {
		return &Context{
			Messages: []Message{
				UserMessage{Role: "user", Content: "q"},
				AssistantMessage{
					Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
					StopReason: StopToolUse,
					Content:    []ContentBlock{ToolCall{Type: "toolCall", ID: "c1", Name: "shot", Arguments: map[string]any{}}},
				},
				ToolResultMessage{Role: "toolResult", ToolCallID: "c1", ToolName: "shot", Content: []ContentBlock{
					TextContent{Type: "text", Text: "here"},
					ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
				}},
			},
		}
	}

	t.Run("gemini 3 nests the image", func(t *testing.T) {
		model := testGoogleModel("")
		model.ID = "gemini-3-pro"
		model.Input = []string{"text", "image"}
		captured, _, _ := captureGoogleRequest(t, model, buildConv(model), &GoogleOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
		})

		contents := captured["contents"].([]any)
		// user, model, and one merged tool-result turn.
		if len(contents) != 3 {
			t.Fatalf("contents = %d turns, want 3", len(contents))
		}
		response := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
		parts := response["parts"].([]any)
		inlineData := parts[0].(map[string]any)["inlineData"].(map[string]any)
		if inlineData["mimeType"] != "image/png" || inlineData["data"] != "AAAA" {
			t.Fatalf("nested image = %#v", inlineData)
		}
	})

	t.Run("gemini 2.5 adds a separate image turn", func(t *testing.T) {
		model := testGoogleModel("")
		model.Input = []string{"text", "image"}
		captured, _, _ := captureGoogleRequest(t, model, buildConv(model), &GoogleOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
		})

		contents := captured["contents"].([]any)
		// user, model, tool result, and the follow-up image turn.
		if len(contents) != 4 {
			t.Fatalf("contents = %d turns, want 4", len(contents))
		}
		response := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
		if _, nested := response["parts"]; nested {
			t.Fatalf("image should not be nested for gemini 2.5: %#v", response)
		}
		imageTurn := contents[3].(map[string]any)
		parts := imageTurn["parts"].([]any)
		if parts[0].(map[string]any)["text"] != "Tool result image:" {
			t.Fatalf("image turn = %#v", imageTurn)
		}
	})
}
