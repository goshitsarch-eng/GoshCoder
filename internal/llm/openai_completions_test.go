package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func testModel(baseURL string) *Model {
	return &Model{
		ID:            "gpt-test",
		Name:          "GPT Test",
		API:           APIOpenAICompletions,
		Provider:      "test",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 128000,
		MaxTokens:     4096,
	}
}

func sseServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
}

func drainEvents(es *AssistantMessageEventStream) []AssistantMessageEvent {
	var events []AssistantMessageEvent
	for event := range es.Events() {
		events = append(events, event)
	}
	return events
}

func eventTypes(events []AssistantMessageEvent) []EventType {
	out := make([]EventType, len(events))
	for i, e := range events {
		out[i] = e.Type
	}
	return out
}

func TestStreamTextOnlyFixture(t *testing.T) {
	fixture := `data: {"id":"chatcmpl-1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-test","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}

data: [DONE]

`
	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	want := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextDelta, EventTextEnd, EventDone}
	got := eventTypes(events)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	if events[2].Delta != "Hello" || events[3].Delta != " world" {
		t.Fatalf("deltas = %q, %q", events[2].Delta, events[3].Delta)
	}
	if events[2].ContentIndex != 0 || events[4].ContentIndex != 0 {
		t.Fatalf("content indexes wrong: %d, %d", events[2].ContentIndex, events[4].ContentIndex)
	}

	final := events[len(events)-1]
	if final.Reason != StopStop {
		t.Fatalf("reason = %q", final.Reason)
	}
	msg := final.Message
	if msg.StopReason != StopStop || msg.ResponseID != "chatcmpl-1" {
		t.Fatalf("message = %#v", msg)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content = %#v", msg.Content)
	}
	if tc, ok := msg.Content[0].(TextContent); !ok || tc.Text != "Hello world" {
		t.Fatalf("content[0] = %#v", msg.Content[0])
	}
	if msg.Usage.Input != 10 || msg.Usage.Output != 2 || msg.Usage.TotalTokens != 12 {
		t.Fatalf("usage = %#v", msg.Usage)
	}
}

func TestStreamToolCallFixture(t *testing.T) {
	fixture := `data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc"}}]},"finish_reason":null}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"Par"}}]},"finish_reason":null}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"is\"}"}}]},"finish_reason":null}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]

`
	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "weather?"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	want := []EventType{EventStart, EventToolCallStart, EventToolCallDelta, EventToolCallDelta, EventToolCallDelta, EventToolCallEnd, EventDone}
	got := eventTypes(events)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	// Progressive partial parse: the first delta truncates mid-key (dropped),
	// the second completes the key with a truncated string value.
	first := events[2].Partial.Content[0].(ToolCall)
	if first.Name != "get_weather" || len(first.Arguments) != 0 {
		t.Fatalf("partial after first delta = %#v", first)
	}
	second := events[3].Partial.Content[0].(ToolCall)
	if second.Arguments["location"] != "Par" {
		t.Fatalf("partial after second delta = %#v", second)
	}

	end := events[5].ToolCall
	if end == nil || end.ID != "call_1" || end.Name != "get_weather" {
		t.Fatalf("toolcall_end = %#v", end)
	}
	if end.Arguments["location"] != "Paris" {
		t.Fatalf("arguments = %#v", end.Arguments)
	}

	final := events[len(events)-1]
	if final.Reason != StopToolUse || final.Message.StopReason != StopToolUse {
		t.Fatalf("final = %#v", final)
	}
	if final.Message.RawStopReason != "tool_calls" {
		t.Fatalf("rawStopReason = %q", final.Message.RawStopReason)
	}
}

func TestStreamReasoningFixture(t *testing.T) {
	fixture := `data: {"id":"r1","choices":[{"index":0,"delta":{"reasoning_content":"let me "},"finish_reason":null}]}

data: {"id":"r1","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}

data: {"id":"r1","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}

data: {"id":"r1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"completion_tokens_details":{"reasoning_tokens":3}}}

data: [DONE]

`
	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "q"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	want := []EventType{
		EventStart, EventThinkingStart, EventThinkingDelta, EventThinkingDelta,
		EventTextStart, EventTextDelta,
		EventThinkingEnd, EventTextEnd, EventDone,
	}
	got := eventTypes(events)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	if events[1].ContentIndex != 0 || events[4].ContentIndex != 1 {
		t.Fatalf("content indexes: %d, %d", events[1].ContentIndex, events[4].ContentIndex)
	}
	final := events[len(events)-1].Message
	if tk, ok := final.Content[0].(ThinkingContent); !ok || tk.Thinking != "let me think" || tk.ThinkingSignature != "reasoning_content" {
		t.Fatalf("thinking = %#v", final.Content[0])
	}
	if final.Usage.Reasoning == nil || *final.Usage.Reasoning != 3 {
		t.Fatalf("usage = %#v", final.Usage)
	}
}

func TestStreamHTTPErrorBecomesErrorEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"message":"rate limited"}}`)
	}))
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventError || last.Reason != StopError {
		t.Fatalf("last = %#v", last)
	}
	if last.Error.StopReason != StopError {
		t.Fatalf("stopReason = %q", last.Error.StopReason)
	}
	if !strings.Contains(last.Error.ErrorMessage, "429") || !strings.Contains(last.Error.ErrorMessage, "rate limited") {
		t.Fatalf("errorMessage = %q", last.Error.ErrorMessage)
	}
}

func TestStreamRetryOn429ThenSuccess(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("retry-after-ms", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k", MaxRetries: 2}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventDone {
		t.Fatalf("last = %#v (%v)", last, last.Error)
	}
	if calls != 2 {
		t.Fatalf("expected 2 HTTP calls, got %d", calls)
	}
}

func TestStreamRetryAfterCapExceeded(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("retry-after", "120")
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{}`)
	}))
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k", MaxRetries: 2}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventError {
		t.Fatalf("last = %#v", last)
	}
	if !strings.Contains(last.Error.ErrorMessage, "retry delay") {
		t.Fatalf("errorMessage = %q", last.Error.ErrorMessage)
	}
}

func TestStreamMissingFinishReason(t *testing.T) {
	srv := sseServer(t, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
	defer srv.Close()

	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventError || !strings.Contains(last.Error.ErrorMessage, "Stream ended without finish_reason") {
		t.Fatalf("last = %#v (%v)", last, last.Error)
	}
}

func TestStreamNoFinishReasonCompatInferredStop(t *testing.T) {
	srv := sseServer(t, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n")
	defer srv.Close()

	model := testModel(srv.URL)
	model.Compat = &OpenAICompletionsCompat{SupportsFinishReason: boolPtr(false)}
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventDone || last.Reason != StopStop {
		t.Fatalf("last = %#v", last)
	}
}

func TestStreamAbortedContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			fmt.Fprint(w, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
			f.Flush()
		}
		<-r.Context().Done() // hang until the client goes away
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	model := testModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{Ctx: ctx, APIKey: "k"}})

	var events []AssistantMessageEvent
	for event := range es.Events() {
		events = append(events, event)
		if event.Type == EventTextDelta {
			cancel()
		}
	}
	last := events[len(events)-1]
	if last.Type != EventError || last.Reason != StopAborted {
		t.Fatalf("last = %#v", last)
	}
	if last.Error.StopReason != StopAborted {
		t.Fatalf("stopReason = %q", last.Error.StopReason)
	}
}

func boolPtr(b bool) *bool { return &b }
func intPtr(i int) *int    { return &i }

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

func captureRequest(t *testing.T, model *Model, conv *Context, opts *OpenAICompletionsOptions) map[string]any {
	t.Helper()
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	model.BaseURL = srv.URL
	drainEvents(Stream(model, conv, opts))
	return captured
}

func TestBuildParamsDefaults(t *testing.T) {
	model := testModel("") // provider "test": standard compat
	conv := &Context{
		SystemPrompt: "be nice",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k", MaxTokens: 100}}
	captured := captureRequest(t, model, conv, opts)

	if captured["model"] != "gpt-test" || captured["stream"] != true {
		t.Fatalf("captured = %#v", captured)
	}
	if captured["max_completion_tokens"] != 100.0 {
		t.Fatalf("max tokens field = %#v", captured)
	}
	if _, ok := captured["max_tokens"]; ok {
		t.Fatal("max_tokens should not be set for standard providers")
	}
	if captured["store"] != false {
		t.Fatal("store should be false for standard providers")
	}
	so, ok := captured["stream_options"].(map[string]any)
	if !ok || so["include_usage"] != true {
		t.Fatalf("stream_options = %#v", captured["stream_options"])
	}
	msgs := captured["messages"].([]any)
	first := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be nice" {
		t.Fatalf("first message = %#v", first)
	}
	if _, hasKey := captured["prompt_cache_key"]; hasKey {
		t.Fatal("prompt_cache_key should be absent without a session id")
	}
}

func TestBuildParamsDeveloperRoleForReasoningModel(t *testing.T) {
	model := testModel("")
	model.Reasoning = true
	conv := &Context{SystemPrompt: "think", Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}, ReasoningEffort: "high"}
	captured := captureRequest(t, model, conv, opts)

	msgs := captured["messages"].([]any)
	if msgs[0].(map[string]any)["role"] != "developer" {
		t.Fatalf("role = %#v", msgs[0])
	}
	if captured["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort = %#v", captured["reasoning_effort"])
	}
}

func TestBuildParamsThinkingLevelMap(t *testing.T) {
	model := testModel("")
	model.Reasoning = true
	mapped := "4096"
	model.ThinkingLevelMap = ThinkingLevelMap{"low": &mapped}
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}, ReasoningEffort: "low"}
	captured := captureRequest(t, model, conv, opts)
	if captured["reasoning_effort"] != "4096" {
		t.Fatalf("reasoning_effort = %#v", captured["reasoning_effort"])
	}
}

func TestBuildParamsPromptCacheKey(t *testing.T) {
	model := testModel("")
	model.BaseURL = "https://api.openai.com/v1" // rewritten by captureRequest; detection needs it
	// captureRequest overwrites BaseURL, so call buildParams directly instead.
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{SessionID: "sess-1"}}
	params := buildParams(model, conv, opts, getCompat(model), resolveCacheRetention("", nil))
	if params["prompt_cache_key"] != "sess-1" {
		t.Fatalf("prompt_cache_key = %#v", params["prompt_cache_key"])
	}
}

func TestBuildParamsLongCacheRetention(t *testing.T) {
	model := testModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{CacheRetention: CacheRetentionLong, SessionID: "s"}}
	params := buildParams(model, conv, opts, getCompat(model), CacheRetentionLong)
	if params["prompt_cache_retention"] != "24h" {
		t.Fatalf("prompt_cache_retention = %#v", params["prompt_cache_retention"])
	}
}

func TestBuildParamsSamplingParamsOverride(t *testing.T) {
	model := testModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	temp := 0.5
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{
		Temperature:    &temp,
		SamplingParams: map[string]any{"top_p": 0.9, "temperature": 0.1},
	}}
	params := buildParams(model, conv, opts, getCompat(model), CacheRetentionShort)
	// samplingParams merge last and override named fields.
	if params["temperature"] != 0.1 || params["top_p"] != 0.9 {
		t.Fatalf("params = %#v", params)
	}
}

// ---------------------------------------------------------------------------
// Compat detection
// ---------------------------------------------------------------------------

func TestDetectCompat(t *testing.T) {
	tests := []struct {
		name     string
		provider string
		baseURL  string
		modelID  string
		check    func(t *testing.T, c resolvedCompat)
	}{
		{"openai", "openai", "https://api.openai.com/v1", "gpt-5", func(t *testing.T, c resolvedCompat) {
			if !c.SupportsStore || c.MaxTokensField != "max_completion_tokens" || !c.SupportsDeveloperRole || !c.SupportsReasoningEffort {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"deepseek", "deepseek", "https://api.deepseek.com", "deepseek-chat", func(t *testing.T, c resolvedCompat) {
			if c.MaxTokensField != "max_tokens" || c.ThinkingFormat != "deepseek" || !c.RequiresReasoningContentOnAssistantMessages || c.SupportsStore {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"openrouter anthropic", "openrouter", "https://openrouter.ai/api/v1", "anthropic/claude-x", func(t *testing.T, c resolvedCompat) {
			if c.CacheControlFormat != "anthropic" || !c.SupportsDeveloperRole || c.ThinkingFormat != "openrouter" || c.SessionAffinityFormat != "openrouter" {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"openrouter other", "openrouter", "https://openrouter.ai/api/v1", "meta/llama", func(t *testing.T, c resolvedCompat) {
			if c.SupportsDeveloperRole || c.CacheControlFormat != "" {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"cerebras url", "custom", "https://api.cerebras.ai/v1", "llama", func(t *testing.T, c resolvedCompat) {
			if c.SupportsStore || c.SupportsDeveloperRole {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"xai no reasoning effort", "xai", "https://api.x.ai/v1", "grok", func(t *testing.T, c resolvedCompat) {
			if c.SupportsReasoningEffort {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"moonshot no strict", "moonshotai", "https://api.moonshot.ai", "kimi", func(t *testing.T, c resolvedCompat) {
			if c.SupportsStrictMode || c.SupportsReasoningEffort || c.MaxTokensField != "max_tokens" {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"together", "together", "https://api.together.xyz/v1", "llama", func(t *testing.T, c resolvedCompat) {
			if c.ThinkingFormat != "together" || c.SupportsLongCacheRetention {
				t.Fatalf("compat = %#v", c)
			}
		}},
		{"zai", "zai", "https://api.z.ai/api/paas/v4", "glm", func(t *testing.T, c resolvedCompat) {
			if c.ThinkingFormat != "zai" || c.MaxTokensField != "max_tokens" {
				t.Fatalf("compat = %#v", c)
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Model{Provider: tt.provider, BaseURL: tt.baseURL, ID: tt.modelID}
			tt.check(t, detectCompat(m))
		})
	}
}

func TestGetCompatOverrides(t *testing.T) {
	m := &Model{Provider: "openai", BaseURL: "https://api.openai.com/v1", ID: "gpt-5"}
	m.Compat = &OpenAICompletionsCompat{
		SupportsStore:  boolPtr(false),
		MaxTokensField: "max_tokens",
	}
	c := getCompat(m)
	if c.SupportsStore || c.MaxTokensField != "max_tokens" {
		t.Fatalf("compat = %#v", c)
	}
	if !c.SupportsDeveloperRole { // untouched override keeps detected value
		t.Fatalf("compat = %#v", c)
	}
}

// ---------------------------------------------------------------------------
// Message transform
// ---------------------------------------------------------------------------

func convertForTest(t *testing.T, model *Model, conv *Context) []any {
	t.Helper()
	return convertMessages(model, conv, getCompat(model))
}

func TestConvertMessagesAssistantReplay(t *testing.T) {
	model := testModel("")
	model.Reasoning = true
	assistant := AssistantMessage{
		Role:     "assistant",
		API:      model.API,
		Provider: model.Provider,
		Model:    model.ID,
		Content: []ContentBlock{
			ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "reasoning_content"},
			TextContent{Type: "text", Text: "answer"},
			ToolCall{Type: "toolCall", ID: "call_1", Name: "do_thing", Arguments: map[string]any{"x": 1.0}},
		},
		StopReason: StopToolUse,
	}
	conv := &Context{
		SystemPrompt: "sys",
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			assistant,
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1", ToolName: "do_thing", Content: []ContentBlock{TextContent{Type: "text", Text: "done"}}},
		},
	}
	msgs := convertForTest(t, model, conv)
	if len(msgs) != 4 {
		t.Fatalf("messages = %#v", msgs)
	}
	if msgs[0].(map[string]any)["role"] != "developer" {
		t.Fatalf("system role = %#v", msgs[0])
	}
	am := msgs[2].(map[string]any)
	if am["role"] != "assistant" || am["content"] != "answer" {
		t.Fatalf("assistant = %#v", am)
	}
	if am["reasoning_content"] != "hmm" {
		t.Fatalf("reasoning_content = %#v", am)
	}
	toolCalls := am["tool_calls"].([]any)
	tc := toolCalls[0].(map[string]any)
	fn := tc["function"].(map[string]any)
	if tc["id"] != "call_1" || fn["name"] != "do_thing" || fn["arguments"] != `{"x":1}` {
		t.Fatalf("tool_call = %#v", tc)
	}
	tr := msgs[3].(map[string]any)
	if tr["role"] != "tool" || tr["tool_call_id"] != "call_1" || tr["content"] != "done" {
		t.Fatalf("tool result = %#v", tr)
	}
}

func TestConvertMessagesToolResultNameAndBridge(t *testing.T) {
	model := testModel("")
	model.Compat = &OpenAICompletionsCompat{
		RequiresToolResultName:           boolPtr(true),
		RequiresAssistantAfterToolResult: boolPtr(true),
	}
	conv := &Context{Messages: []Message{
		AssistantMessage{Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID, StopReason: StopToolUse, Content: []ContentBlock{
			ToolCall{Type: "toolCall", ID: "c1", Name: "t", Arguments: map[string]any{}},
		}},
		ToolResultMessage{Role: "toolResult", ToolCallID: "c1", ToolName: "t", Content: []ContentBlock{TextContent{Type: "text", Text: "out"}}},
		UserMessage{Role: "user", Content: "next"},
	}}
	msgs := convertForTest(t, model, conv)
	// assistant, tool (with name), synthetic assistant bridge, user
	if len(msgs) != 4 {
		t.Fatalf("messages = %#v", msgs)
	}
	if msgs[1].(map[string]any)["name"] != "t" {
		t.Fatalf("tool msg = %#v", msgs[1])
	}
	bridge := msgs[2].(map[string]any)
	if bridge["role"] != "assistant" || bridge["content"] != "I have processed the tool results." {
		t.Fatalf("bridge = %#v", bridge)
	}
}

func TestConvertMessagesOrphanedToolCallGetsSyntheticResult(t *testing.T) {
	model := testModel("")
	conv := &Context{Messages: []Message{
		AssistantMessage{Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID, StopReason: StopToolUse, Content: []ContentBlock{
			ToolCall{Type: "toolCall", ID: "orphan", Name: "t", Arguments: map[string]any{}},
		}},
		UserMessage{Role: "user", Content: "never mind"},
	}}
	msgs := convertForTest(t, model, conv)
	if len(msgs) != 3 {
		t.Fatalf("messages = %#v", msgs)
	}
	synthetic := msgs[1].(map[string]any)
	if synthetic["role"] != "tool" || synthetic["tool_call_id"] != "orphan" || synthetic["content"] != "No result provided" {
		t.Fatalf("synthetic = %#v", synthetic)
	}
}

func TestConvertMessagesErroredAssistantDropped(t *testing.T) {
	model := testModel("")
	conv := &Context{Messages: []Message{
		UserMessage{Role: "user", Content: "q"},
		AssistantMessage{Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID, StopReason: StopError, ErrorMessage: "boom", Content: []ContentBlock{TextContent{Type: "text", Text: "partial"}}},
		UserMessage{Role: "user", Content: "retry"},
	}}
	msgs := convertForTest(t, model, conv)
	if len(msgs) != 2 {
		t.Fatalf("messages = %#v", msgs)
	}
}

func TestConvertMessagesPipeToolCallIDNormalization(t *testing.T) {
	model := testModel("")
	long := strings.Repeat("x", 50) + "|" + strings.Repeat("y", 50)
	conv := &Context{Messages: []Message{
		AssistantMessage{Role: "assistant", API: "openai-responses", Provider: "other", Model: "other-model", StopReason: StopToolUse, Content: []ContentBlock{
			ToolCall{Type: "toolCall", ID: long, Name: "t", Arguments: map[string]any{}},
		}},
		ToolResultMessage{Role: "toolResult", ToolCallID: long, ToolName: "t", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
	}}
	msgs := convertForTest(t, model, conv)
	toolMsg := msgs[1].(map[string]any)
	normalized := msgs[0].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)["id"].(string)
	if strings.Contains(normalized, "|") || len(normalized) > 40 {
		t.Fatalf("normalized id = %q", normalized)
	}
	if toolMsg["tool_call_id"] != normalized {
		t.Fatalf("tool result id %q does not match normalized %q", toolMsg["tool_call_id"], normalized)
	}
}

func TestConvertMessagesImagesDowngradedForTextOnlyModel(t *testing.T) {
	model := testModel("") // text-only input
	conv := &Context{Messages: []Message{
		UserMessage{Role: "user", Content: []ContentBlock{
			TextContent{Type: "text", Text: "look"},
			ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
		}},
	}}
	msgs := convertForTest(t, model, conv)
	content := msgs[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content = %#v", content)
	}
	second := content[1].(map[string]any)
	if second["type"] != "text" || second["text"] != "(image omitted: model does not support images)" {
		t.Fatalf("second = %#v", second)
	}
}

func TestConvertMessagesImageURLForVisionModel(t *testing.T) {
	model := testModel("")
	model.Input = []string{"text", "image"}
	conv := &Context{Messages: []Message{
		UserMessage{Role: "user", Content: []ContentBlock{
			ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
		}},
	}}
	msgs := convertForTest(t, model, conv)
	content := msgs[0].(map[string]any)["content"].([]any)
	part := content[0].(map[string]any)
	if part["type"] != "image_url" {
		t.Fatalf("part = %#v", part)
	}
	url := part["image_url"].(map[string]any)["url"]
	if url != "data:image/png;base64,AAAA" {
		t.Fatalf("url = %#v", url)
	}
}

func TestConvertToolsStrictMode(t *testing.T) {
	model := testModel("")
	tools := []Tool{{
		Name:        "get_weather",
		Description: "Get weather",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
	}}
	out := convertTools(tools, getCompat(model))
	tool := out[0].(map[string]any)
	if tool["type"] != "function" {
		t.Fatalf("tool = %#v", tool)
	}
	fn := tool["function"].(map[string]any)
	if fn["name"] != "get_weather" || fn["strict"] != false {
		t.Fatalf("fn = %#v", fn)
	}
	// Parameters must pass through as raw JSON schema.
	raw, err := json.Marshal(fn["parameters"])
	if err != nil || string(raw) != `{"type":"object","properties":{"location":{"type":"string"}}}` {
		t.Fatalf("parameters = %s (%v)", raw, err)
	}

	// Moonshot: no strict field.
	model.Provider = "moonshotai"
	model.BaseURL = "https://api.moonshot.ai"
	out = convertTools(tools, getCompat(model))
	fn = out[0].(map[string]any)["function"].(map[string]any)
	if _, ok := fn["strict"]; ok {
		t.Fatalf("strict should be omitted for moonshot: %#v", fn)
	}
}

func TestHasToolHistoryForcesEmptyToolsParam(t *testing.T) {
	model := testModel("")
	conv := &Context{Messages: []Message{
		AssistantMessage{Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID, StopReason: StopToolUse, Content: []ContentBlock{
			ToolCall{Type: "toolCall", ID: "c1", Name: "t", Arguments: map[string]any{}},
		}},
		ToolResultMessage{Role: "toolResult", ToolCallID: "c1", ToolName: "t", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		UserMessage{Role: "user", Content: "go on"},
	}}
	params := buildParams(model, conv, &OpenAICompletionsOptions{}, getCompat(model), CacheRetentionShort)
	tools, ok := params["tools"].([]any)
	if !ok || len(tools) != 0 {
		t.Fatalf("tools = %#v", params["tools"])
	}
}

func TestMapStopReason(t *testing.T) {
	cases := map[string]StopReason{
		"stop":           StopStop,
		"end":            StopStop,
		"length":         StopLength,
		"function_call":  StopToolUse,
		"tool_calls":     StopToolUse,
		"content_filter": StopError,
		"network_error":  StopError,
		"something_new":  StopError,
	}
	for in, want := range cases {
		got, _ := mapStopReason(in)
		if got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestParseChunkUsageCacheTokens(t *testing.T) {
	model := testModel("")
	model.Cost = ModelCost{ModelCostRates: ModelCostRates{Input: 1, Output: 2, CacheRead: 0.1, CacheWrite: 1.25}}
	raw := &chunkUsage{PromptTokens: 100, CompletionTokens: 20}
	raw.PromptTokensDetails = &struct {
		CachedTokens     *int `json:"cached_tokens"`
		CacheWriteTokens int  `json:"cache_write_tokens"`
	}{CachedTokens: intPtr(30), CacheWriteTokens: 10}
	usage := parseChunkUsage(raw, model)
	if usage.Input != 60 || usage.CacheRead != 30 || usage.CacheWrite != 10 || usage.Output != 20 || usage.TotalTokens != 120 {
		t.Fatalf("usage = %#v", usage)
	}
	if usage.Cost.Total <= 0 {
		t.Fatalf("cost = %#v", usage.Cost)
	}
}

func TestShortHashMatchesTS(t *testing.T) {
	// Reference values computed with utils/hash.ts under Node.
	cases := map[string]string{
		"":                  "k4n83c7h0j2b",
		"call_abc|item_xyz": "5r62khfrcycv",
		"héllo wörld|π":     "1ngra4a7tbed7",
	}
	for in, want := range cases {
		if got := ShortHash(in); got != want {
			t.Errorf("ShortHash(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestJSONFieldNames(t *testing.T) {
	// Session serialization depends on pi's exact JSON field names.
	msg := AssistantMessage{
		Role:       "assistant",
		Content:    []ContentBlock{ThinkingContent{Type: "thinking", Thinking: "t", ThinkingSignature: "sig"}, ToolCall{Type: "toolCall", ID: "i", Name: "n", Arguments: map[string]any{}}},
		API:        APIOpenAICompletions,
		Provider:   "openai",
		Model:      "gpt",
		Usage:      Usage{Input: 1, Output: 2, CacheRead: 3, CacheWrite: 4, TotalTokens: 10},
		StopReason: StopStop,
		Timestamp:  123,
	}
	data, err := json.Marshal(msg)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"role", "content", "api", "provider", "model", "usage", "stopReason", "timestamp"} {
		if _, ok := decoded[key]; !ok {
			t.Errorf("missing key %q in %s", key, data)
		}
	}
	blocks := decoded["content"].([]any)
	if blocks[0].(map[string]any)["thinkingSignature"] != "sig" {
		t.Errorf("thinking block = %v", blocks[0])
	}
	if blocks[1].(map[string]any)["type"] != "toolCall" {
		t.Errorf("toolCall block = %v", blocks[1])
	}
	usage := decoded["usage"].(map[string]any)
	for _, key := range []string{"input", "output", "cacheRead", "cacheWrite", "totalTokens", "cost"} {
		if _, ok := usage[key]; !ok {
			t.Errorf("missing usage key %q", key)
		}
	}
	// Round-trip through UnmarshalMessages.
	msgs, err := UnmarshalMessages(json.RawMessage("[" + string(data) + "]"))
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 {
		t.Fatalf("round-trip = %#v", msgs)
	}
	rt, ok := msgs[0].(AssistantMessage)
	if !ok || rt.StopReason != StopStop || len(rt.Content) != 2 {
		t.Fatalf("round-trip = %#v", msgs[0])
	}
}

func TestRetryClassifier(t *testing.T) {
	retryable := &AssistantMessage{StopReason: StopError, ErrorMessage: "429 rate limit exceeded"}
	if !IsRetryableAssistantError(retryable) {
		t.Fatal("should be retryable")
	}
	quota := &AssistantMessage{StopReason: StopError, ErrorMessage: "429 insufficient_quota"}
	if IsRetryableAssistantError(quota) {
		t.Fatal("quota errors are not retryable")
	}
	ok := &AssistantMessage{StopReason: StopStop}
	if IsRetryableAssistantError(ok) {
		t.Fatal("non-error is not retryable")
	}
}

func TestRetryAssistantCall(t *testing.T) {
	attempts := 0
	policy := &RetryPolicy{Enabled: true, MaxRetries: 2, BaseDelayMs: 1}
	var scheduled []int
	result := RetryAssistantCall(context.Background(), func() *AssistantMessage {
		attempts++
		if attempts < 3 {
			return &AssistantMessage{StopReason: StopError, ErrorMessage: "503 service unavailable"}
		}
		return &AssistantMessage{StopReason: StopStop}
	}, policy, &RetryCallbacks{
		OnRetryScheduled: func(attempt, maxAttempts int, delayMs int64, errorMessage string) {
			scheduled = append(scheduled, attempt)
		},
	})
	if result.StopReason != StopStop || attempts != 3 {
		t.Fatalf("result = %#v after %d attempts", result, attempts)
	}
	if len(scheduled) != 2 || scheduled[0] != 1 || scheduled[1] != 2 {
		t.Fatalf("scheduled = %v", scheduled)
	}
}

func TestStreamSessionAffinityHeaders(t *testing.T) {
	var gotHeaders http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	model := testModel(srv.URL)
	model.Compat = &OpenAICompletionsCompat{SendSessionAffinityHeaders: boolPtr(true)}
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k", SessionID: "sess-42"}}
	drainEvents(Stream(model, conv, opts))

	if gotHeaders.Get("session_id") != "sess-42" || gotHeaders.Get("x-client-request-id") != "sess-42" || gotHeaders.Get("x-session-affinity") != "sess-42" {
		t.Fatalf("headers = %v", gotHeaders)
	}
}

func TestStreamNoAPIKeyErrorEvent(t *testing.T) {
	model := testModel("http://127.0.0.1:1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := Stream(model, conv, &OpenAICompletionsOptions{})
	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventError || !strings.Contains(last.Error.ErrorMessage, "no API key for provider") {
		t.Fatalf("last = %#v (%v)", last, last.Error)
	}
}

func TestStreamSimpleClampsReasoning(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()

	model := testModel(srv.URL)
	model.Reasoning = true
	// "high" unsupported (null) → clamps to next supported level.
	model.ThinkingLevelMap = ThinkingLevelMap{"high": nil}
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := StreamSimple(model, conv, &SimpleStreamOptions{StreamOptions: StreamOptions{APIKey: "k"}, Reasoning: "high"})
	drainEvents(es)
	if captured["reasoning_effort"] != "xhigh" {
		// high is null (unsupported) → clamp upward: xhigh is also unsupported
		// (no mapping) → max (unsupported) → falls down to medium.
		if captured["reasoning_effort"] != "medium" {
			t.Fatalf("reasoning_effort = %#v", captured["reasoning_effort"])
		}
	}
	// maxTokens was clamped to the context window and sent.
	if captured["max_completion_tokens"] == nil {
		t.Fatalf("captured = %#v", captured)
	}
}

var _ = time.Now // keep time import if unused in some build permutations
