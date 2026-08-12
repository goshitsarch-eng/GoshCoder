package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testResponsesModel(baseURL string) *Model {
	return &Model{
		ID:            "gpt-test",
		Name:          "GPT Test",
		API:           APIOpenAIResponses,
		Provider:      "openai",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 200000,
		MaxTokens:     8192,
	}
}

// responsesSSE formats one Responses SSE record. The payload carries its own
// "type", matching the real API.
func responsesSSE(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

// responsesCompleted is a minimal terminal event, required for every stream.
const responsesCompleted = `{"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`

func responsesServer(t *testing.T, body string) *Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return testResponsesModel(srv.URL)
}

// captureResponsesRequest runs a stream against a recording server and returns
// the decoded request body and headers.
func captureResponsesRequest(t *testing.T, model *Model, conv *Context, opts *OpenAIResponsesOptions) (map[string]any, http.Header) {
	t.Helper()
	var captured map[string]any
	var headers http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, responsesSSE("response.completed", responsesCompleted))
	}))
	defer srv.Close()
	model.BaseURL = srv.URL
	drainEvents(StreamOpenAIResponses(model, conv, opts))
	return captured, headers
}

func TestResponsesStreamTextOnly(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.created", `{"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}`)+
			responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`)+
			responsesSSE("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"delta":"Hello"}`)+
			responsesSSE("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"delta":" world"}`)+
			responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello world","annotations":[]}]}}`)+
			responsesSSE("response.completed", responsesCompleted))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

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
	// The text signature persists the item id for replay.
	text := final.Message.Content[0].(TextContent)
	if text.TextSignature != `{"v":1,"id":"msg_1"}` {
		t.Fatalf("textSignature = %q", text.TextSignature)
	}
	if final.Message.Usage.Input != 10 || final.Message.Usage.Output != 5 || final.Message.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %#v", final.Message.Usage)
	}
}

func TestResponsesStreamToolCall(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}`)+
			responsesSSE("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"location\":"}`)+
			responsesSSE("response.function_call_arguments.delta", `{"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"Paris\"}"}`)+
			responsesSSE("response.function_call_arguments.done", `{"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"location\":\"Paris\"}"}`)+
			responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}`)+
			responsesSSE("response.completed", responsesCompleted))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "weather?"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	want := []EventType{EventStart, EventToolCallStart, EventToolCallDelta, EventToolCallDelta, EventToolCallEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	// Progressive partial parse, same contract as the other protocol ports.
	first := events[2].Partial.Content[0].(ToolCall)
	if first.Name != "get_weather" || len(first.Arguments) != 0 {
		t.Fatalf("partial after first delta = %#v", first)
	}

	final := events[len(events)-1]
	// Tool calls override the "completed" status.
	if final.Reason != StopToolUse {
		t.Fatalf("reason = %q", final.Reason)
	}
	tc := final.Message.Content[0].(ToolCall)
	// pi encodes Responses tool calls as "callID|itemID".
	if tc.ID != "call_1|fc_1" || tc.Arguments["location"] != "Paris" {
		t.Fatalf("toolCall = %#v", tc)
	}
}

func TestResponsesStreamReasoning(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`)+
			responsesSSE("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"Think"}`)+
			responsesSSE("response.reasoning_summary_part.done", `{"type":"response.reasoning_summary_part.done","output_index":0}`)+
			responsesSSE("response.reasoning_summary_text.delta", `{"type":"response.reasoning_summary_text.delta","output_index":0,"delta":"more"}`)+
			responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think"},{"type":"summary_text","text":"more"}],"encrypted_content":"enc"}}`)+
			responsesSSE("response.completed", responsesCompleted))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	want := []EventType{EventStart, EventThinkingStart, EventThinkingDelta, EventThinkingDelta, EventThinkingDelta, EventThinkingEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	// Summary parts are separated by a blank line.
	if events[3].Delta != "\n\n" {
		t.Fatalf("part separator delta = %q", events[3].Delta)
	}

	thinking := events[len(events)-1].Message.Content[0].(ThinkingContent)
	if thinking.Thinking != "Think\n\nmore" {
		t.Fatalf("thinking = %q", thinking.Thinking)
	}
	// The signature is the whole reasoning item, replayed verbatim next turn.
	var item map[string]any
	if err := json.Unmarshal([]byte(thinking.ThinkingSignature), &item); err != nil {
		t.Fatalf("signature is not JSON: %v", err)
	}
	if item["id"] != "rs_1" || item["encrypted_content"] != "enc" {
		t.Fatalf("signature item = %#v", item)
	}
}

// TestResponsesBackfillsReasoningSignature covers the Azure case where
// encrypted_content only arrives on the terminal response event.
func TestResponsesBackfillsReasoningSignature(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`)+
			responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"t"}]}}`)+
			responsesSSE("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"late-enc"}]}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	thinking := events[len(events)-1].Message.Content[0].(ThinkingContent)
	var item map[string]any
	if err := json.Unmarshal([]byte(thinking.ThinkingSignature), &item); err != nil {
		t.Fatalf("signature is not JSON: %v", err)
	}
	if item["encrypted_content"] != "late-enc" {
		t.Fatalf("encrypted_content = %#v, want backfilled", item["encrypted_content"])
	}
}

func TestResponsesUsageSubtractsCacheTokens(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.completed", `{"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":100,"output_tokens":20,"total_tokens":120,"input_tokens_details":{"cached_tokens":30,"cache_write_tokens":10},"output_tokens_details":{"reasoning_tokens":8}}}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	usage := events[len(events)-1].Message.Usage
	// OpenAI counts cached and cache-write tokens inside input_tokens.
	if usage.Input != 60 || usage.CacheRead != 30 || usage.CacheWrite != 10 {
		t.Fatalf("usage = %#v", usage)
	}
	if usage.Reasoning == nil || *usage.Reasoning != 8 {
		t.Fatalf("reasoning = %#v", usage.Reasoning)
	}
}

func TestResponsesStopReasons(t *testing.T) {
	tests := []struct {
		name       string
		response   string
		wantReason StopReason
		wantRaw    string
	}{
		{
			name:       "max output tokens maps to length",
			response:   `{"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"}}}`,
			wantReason: StopLength,
			wantRaw:    "incomplete.max_output_tokens",
		},
		{
			name:       "completed maps to stop",
			response:   `{"type":"response.completed","response":{"id":"r","status":"completed"}}`,
			wantReason: StopStop,
			wantRaw:    "completed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := responsesServer(t, responsesSSE("x", tt.response))
			conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
			events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))
			final := events[len(events)-1]
			if final.Reason != tt.wantReason {
				t.Fatalf("reason = %q, want %q", final.Reason, tt.wantReason)
			}
			if final.Message.RawStopReason != tt.wantRaw {
				t.Fatalf("rawStopReason = %q, want %q", final.Message.RawStopReason, tt.wantRaw)
			}
		})
	}
}

// TestResponsesIncompleteWithoutMaxTokensErrors covers a non-truncation
// incomplete reason surfacing as an error.
func TestResponsesIncompleteWithoutMaxTokensErrors(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("x", `{"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"content_filter"}}}`))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError || final.Reason != StopError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "content_filter") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestResponsesStreamWithoutTerminalEventErrors(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"m","role":"assistant","status":"in_progress","content":[]}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final event = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "terminal response event") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestResponsesFailedEventErrors(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("x", `{"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"server_error","message":"boom"}}}`))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final event = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "server_error: boom") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestResponsesBuildParamsBasics(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{
		SystemPrompt: "be brief",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	captured, headers := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k", MaxTokens: 2048, SessionID: "sess-1"},
	})

	if captured["model"] != "gpt-test" || captured["stream"] != true || captured["store"] != false {
		t.Fatalf("params = %#v", captured)
	}
	if captured["max_output_tokens"] != 2048.0 {
		t.Fatalf("max_output_tokens = %#v", captured["max_output_tokens"])
	}
	if captured["prompt_cache_key"] != "sess-1" {
		t.Fatalf("prompt_cache_key = %#v", captured["prompt_cache_key"])
	}
	if headers.Get("Authorization") != "Bearer k" {
		t.Fatalf("authorization = %q", headers.Get("Authorization"))
	}
	// Non-reasoning model: "system", not "developer".
	input := captured["input"].([]any)
	system := input[0].(map[string]any)
	if system["role"] != "system" || system["content"] != "be brief" {
		t.Fatalf("system message = %#v", system)
	}
	user := input[1].(map[string]any)
	content := user["content"].([]any)[0].(map[string]any)
	if user["role"] != "user" || content["type"] != "input_text" || content["text"] != "hi" {
		t.Fatalf("user message = %#v", user)
	}
}

func TestResponsesMaxOutputTokensFloor(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	// The API rejects max_output_tokens below 16.
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k", MaxTokens: 4},
	})
	if captured["max_output_tokens"] != 16.0 {
		t.Fatalf("max_output_tokens = %#v, want 16", captured["max_output_tokens"])
	}
}

func TestResponsesDeveloperRoleForReasoningModels(t *testing.T) {
	model := testResponsesModel("")
	model.Reasoning = true
	conv := &Context{SystemPrompt: "sys", Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	system := captured["input"].([]any)[0].(map[string]any)
	if system["role"] != "developer" {
		t.Fatalf("role = %#v, want developer", system["role"])
	}
}

func TestResponsesReasoningParams(t *testing.T) {
	model := testResponsesModel("")
	model.Reasoning = true
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions:   StreamOptions{APIKey: "k"},
		ReasoningEffort: "high",
	})

	reasoning, ok := captured["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" || reasoning["summary"] != "auto" {
		t.Fatalf("reasoning = %#v", captured["reasoning"])
	}
	// Encrypted reasoning must be requested for stateless replay.
	include, ok := captured["include"].([]any)
	if !ok || len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %#v", captured["include"])
	}
}

func TestResponsesToolConversion(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		}},
	}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	tools := captured["tools"].([]any)
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "get_weather" {
		t.Fatalf("tool = %#v", tool)
	}
	// strict is omitted unless the model declares support.
	if _, hasStrict := tool["strict"]; hasStrict {
		t.Fatalf("strict should be omitted without supportsStrictMode: %#v", tool)
	}
	params := tool["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Fatalf("parameters = %#v", params)
	}
}

func TestResponsesStrictModeTools(t *testing.T) {
	model := testResponsesModel("")
	supportsStrict := true
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools:    []Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Compat:        &OpenAIResponsesCompat{SupportsStrictMode: &supportsStrict},
	})

	tool := captured["tools"].([]any)[0].(map[string]any)
	if tool["strict"] != false {
		t.Fatalf("strict = %#v, want false (default when the tool asks for nothing)", tool["strict"])
	}
}

func TestResponsesReplaysAssistantAndToolResult(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: `{"type":"reasoning","id":"rs_1","summary":[]}`},
					TextContent{Type: "text", Text: "answer", TextSignature: `{"v":1,"id":"msg_1"}`},
					ToolCall{Type: "toolCall", ID: "call_1|fc_1", Name: "do_thing", Arguments: map[string]any{"x": 1.0}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1|fc_1", ToolName: "do_thing", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		},
	}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	input := captured["input"].([]any)
	// user, reasoning, message, function_call, function_call_output
	if len(input) != 5 {
		t.Fatalf("input items = %d: %#v", len(input), input)
	}

	reasoning := input[1].(map[string]any)
	if reasoning["type"] != "reasoning" || reasoning["id"] != "rs_1" {
		t.Fatalf("reasoning item = %#v", reasoning)
	}
	message := input[2].(map[string]any)
	if message["type"] != "message" || message["id"] != "msg_1" || message["status"] != "completed" {
		t.Fatalf("message item = %#v", message)
	}
	call := input[3].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "call_1" || call["id"] != "fc_1" {
		t.Fatalf("function_call item = %#v", call)
	}
	if call["arguments"] != `{"x":1}` {
		t.Fatalf("arguments = %#v", call["arguments"])
	}
	result := input[4].(map[string]any)
	if result["type"] != "function_call_output" || result["call_id"] != "call_1" || result["output"] != "ok" {
		t.Fatalf("function_call_output item = %#v", result)
	}
}

// TestResponsesDropsForeignToolCallItemID covers cross-provider replay: the
// foreign item id is meaningless to OpenAI, so it is rewritten to fc_* by
// TransformMessages and then dropped to avoid pairing validation.
func TestResponsesDropsForeignToolCallItemID(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: APIAnthropicMessages, Provider: "anthropic", Model: "claude-test",
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ToolCall{Type: "toolCall", ID: "toolu_abc", Name: "do_thing", Arguments: map[string]any{}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "toolu_abc", ToolName: "do_thing", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		},
	}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	input := captured["input"].([]any)
	call := input[1].(map[string]any)
	if call["type"] != "function_call" || call["call_id"] != "toolu_abc" {
		t.Fatalf("function_call item = %#v", call)
	}
	// No "|" in the source id, so there is no item id to carry over.
	if call["id"] != nil {
		t.Fatalf("id = %#v, want null", call["id"])
	}
	result := input[2].(map[string]any)
	if result["call_id"] != "toolu_abc" {
		t.Fatalf("function_call_output call_id = %#v", result["call_id"])
	}
}

func TestResponsesImageToolResultOutput(t *testing.T) {
	model := testResponsesModel("")
	model.Input = []string{"text", "image"}
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content:    []ContentBlock{ToolCall{Type: "toolCall", ID: "call_1|fc_1", Name: "shot", Arguments: map[string]any{}}},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1|fc_1", ToolName: "shot", Content: []ContentBlock{
				TextContent{Type: "text", Text: "here"},
				ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
			}},
		},
	}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	input := captured["input"].([]any)
	result := input[len(input)-1].(map[string]any)
	output := result["output"].([]any)
	if len(output) != 2 {
		t.Fatalf("output = %#v", output)
	}
	if text := output[0].(map[string]any); text["type"] != "input_text" || text["text"] != "here" {
		t.Fatalf("text part = %#v", output[0])
	}
	image := output[1].(map[string]any)
	if image["type"] != "input_image" || image["image_url"] != "data:image/png;base64,AAAA" {
		t.Fatalf("image part = %#v", image)
	}
}

// TestResponsesTextOnlyToolResultForTextModel covers the downgrade path: for a
// text-only model TransformMessages replaces tool images with a text
// placeholder before conversion, so the output is a plain string.
func TestResponsesTextOnlyToolResultForTextModel(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content:    []ContentBlock{ToolCall{Type: "toolCall", ID: "call_1|fc_1", Name: "shot", Arguments: map[string]any{}}},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1|fc_1", ToolName: "shot", Content: []ContentBlock{
				ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
			}},
		},
	}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	input := captured["input"].([]any)
	result := input[len(input)-1].(map[string]any)
	if result["output"] != "(tool image omitted: model does not support images)" {
		t.Fatalf("output = %#v", result["output"])
	}
}

func TestResponsesSessionAffinityHeaders(t *testing.T) {
	t.Run("openai", func(t *testing.T) {
		model := testResponsesModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		_, headers := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
			StreamOptions: StreamOptions{APIKey: "k", SessionID: "sess-1"},
		})
		if headers.Get("session_id") != "sess-1" || headers.Get("x-client-request-id") != "sess-1" {
			t.Fatalf("headers = session_id=%q x-client-request-id=%q", headers.Get("session_id"), headers.Get("x-client-request-id"))
		}
	})

	t.Run("openrouter", func(t *testing.T) {
		model := testResponsesModel("")
		model.Provider = "openrouter"
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		_, headers := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
			StreamOptions: StreamOptions{APIKey: "k", SessionID: "sess-1"},
		})
		if headers.Get("x-session-id") != "sess-1" {
			t.Fatalf("x-session-id = %q", headers.Get("x-session-id"))
		}
		if headers.Get("session_id") != "" {
			t.Fatalf("session_id should be openai-only, got %q", headers.Get("session_id"))
		}
	})
}

func TestResponsesCacheRetentionNone(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, headers := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k", SessionID: "sess-1", CacheRetention: CacheRetentionNone},
	})

	if _, ok := captured["prompt_cache_key"]; ok {
		t.Fatalf("prompt_cache_key should be omitted: %#v", captured["prompt_cache_key"])
	}
	// Session affinity headers are suppressed along with caching.
	if headers.Get("session_id") != "" {
		t.Fatalf("session_id = %q, want empty", headers.Get("session_id"))
	}
}

func TestResponsesLongCacheRetention(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k", CacheRetention: CacheRetentionLong},
	})
	if captured["prompt_cache_retention"] != "24h" {
		t.Fatalf("prompt_cache_retention = %#v", captured["prompt_cache_retention"])
	}
}

func TestResponsesSamplingParamsOverrideNamedFields(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _ := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{
			APIKey:         "k",
			MaxTokens:      100,
			SamplingParams: map[string]any{"max_output_tokens": 4096, "top_p": 0.5},
		},
	})
	if captured["max_output_tokens"] != 4096.0 {
		t.Fatalf("max_output_tokens = %#v, want the sampling param to win", captured["max_output_tokens"])
	}
	if captured["top_p"] != 0.5 {
		t.Fatalf("top_p = %#v", captured["top_p"])
	}
}

func TestResponsesNoAPIKeyErrors(t *testing.T) {
	model := testResponsesModel("http://127.0.0.1:1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final event = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "No API key") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

// TestResponsesAuthorizationHeaderInsteadOfAPIKey covers callers that sign
// requests themselves.
func TestResponsesAuthorizationHeaderInsteadOfAPIKey(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	auth := "Bearer custom"
	_, headers := captureResponsesRequest(t, model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{Headers: ProviderHeaders{"authorization": &auth}},
	})
	if headers.Get("Authorization") != "Bearer custom" {
		t.Fatalf("authorization = %q", headers.Get("Authorization"))
	}
}

func TestResponsesServiceTierPricing(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("x", `{"type":"response.completed","response":{"id":"r","status":"completed","service_tier":"flex","usage":{"input_tokens":1000,"output_tokens":1000,"total_tokens":2000}}}`))
	model.Cost = ModelCost{ModelCostRates: ModelCostRates{Input: 1, Output: 1}}

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	}))

	usage := events[len(events)-1].Message.Usage
	// flex is half price: 1000/1e6 * 1 * 0.5 per direction.
	wantPerDirection := 1000.0 / 1e6 * 0.5
	if usage.Cost.Input != wantPerDirection || usage.Cost.Output != wantPerDirection {
		t.Fatalf("cost = %#v, want %v per direction", usage.Cost, wantPerDirection)
	}
	if usage.Cost.Total != usage.Cost.Input+usage.Cost.Output {
		t.Fatalf("total = %v, want the sum of components", usage.Cost.Total)
	}
}

func TestResponsesStreamSimpleReasoning(t *testing.T) {
	model := testResponsesModel("")
	model.Reasoning = true
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, responsesSSE("x", responsesCompleted))
	}))
	defer srv.Close()
	model.BaseURL = srv.URL

	drainEvents(StreamOpenAIResponsesSimple(model, conv, &SimpleStreamOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Reasoning:     ThinkingHigh,
	}))

	reasoning, ok := captured["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v", captured["reasoning"])
	}
}

// TestResponsesStreamSimpleReasoningOff covers thinking "off" clamping to an
// omitted effort rather than an explicit level.
func TestResponsesStreamSimpleReasoningOff(t *testing.T) {
	model := testResponsesModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, responsesSSE("x", responsesCompleted))
	}))
	defer srv.Close()
	model.BaseURL = srv.URL

	// Non-reasoning model: ClampThinkingLevel yields "off".
	drainEvents(StreamOpenAIResponsesSimple(model, conv, &SimpleStreamOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Reasoning:     ThinkingHigh,
	}))

	if _, ok := captured["reasoning"]; ok {
		t.Fatalf("reasoning should be omitted for a non-reasoning model: %#v", captured["reasoning"])
	}
}

func TestResponsesRegisteredInRegistry(t *testing.T) {
	funcs, ok := GetStreamer(APIOpenAIResponses)
	if !ok {
		t.Fatal("openai-responses is not registered")
	}
	if funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("registration = %#v", funcs)
	}
}

func TestResponsesGrammarToolCall(t *testing.T) {
	model := responsesServer(t,
		responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"run","input":""}}`)+
			responsesSSE("response.custom_tool_call_input.delta", `{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"echo "}`)+
			responsesSSE("response.custom_tool_call_input.delta", `{"type":"response.custom_tool_call_input.delta","output_index":0,"delta":"hi"}`)+
			responsesSSE("response.custom_tool_call_input.done", `{"type":"response.custom_tool_call_input.done","output_index":0,"input":"echo hi"}`)+
			responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"custom_tool_call","id":"ctc_1","call_id":"call_1","name":"run","input":"echo hi"}}`)+
			responsesSSE("response.completed", responsesCompleted))

	supportsGrammar := true
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "run it"}},
		Tools: []Tool{{
			Name:                "run",
			Description:         "Run a command",
			Parameters:          json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`),
			ConstrainedSampling: json.RawMessage(`{"type":"grammar","variants":{"openai_lark":"start: /.+/"}}`),
		}},
	}
	events := drainEvents(StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Compat:        &OpenAIResponsesCompat{SupportsOpenAIGrammarTools: &supportsGrammar},
	}))

	final := events[len(events)-1]
	if final.Type != EventDone {
		t.Fatalf("final event = %#v (error: %v)", final.Type, final.Error)
	}
	tc := final.Message.Content[0].(ToolCall)
	// The grammar input lands on the tool's single required property.
	if tc.Name != "run" || tc.Arguments["command"] != "echo hi" {
		t.Fatalf("toolCall = %#v", tc)
	}

	// Deltas re-encode the raw input as JSON object text.
	var deltas strings.Builder
	for _, event := range events {
		if event.Type == EventToolCallDelta {
			deltas.WriteString(event.Delta)
		}
	}
	if got := deltas.String(); got != `{"command":"echo hi"}` {
		t.Fatalf("assembled deltas = %q", got)
	}
}
