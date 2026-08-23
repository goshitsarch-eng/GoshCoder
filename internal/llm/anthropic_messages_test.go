package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testAnthropicModel(baseURL string) *Model {
	return &Model{
		ID:            "claude-test",
		Name:          "Claude Test",
		API:           APIAnthropicMessages,
		Provider:      "anthropic",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 200000,
		MaxTokens:     8192,
	}
}

// anthropicSSE formats one Anthropic SSE record.
func anthropicSSE(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

const anthropicMessageStartFixture = `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","usage":{"input_tokens":10,"output_tokens":1}}}`

func TestAnthropicStreamTextOnly(t *testing.T) {
	fixture := anthropicSSE("message_start", anthropicMessageStartFixture) +
		anthropicSSE("ping", `{"type":"ping"}`) +
		anthropicSSE("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}`) +
		anthropicSSE("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		anthropicSSE("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`) +
		anthropicSSE("message_stop", `{"type":"message_stop"}`)

	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	want := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextDelta, EventTextEnd, EventDone}
	got := eventTypes(events)
	if fmt.Sprint(got) != fmt.Sprint(want) {
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
	msg := final.Message
	if msg.StopReason != StopStop || msg.ResponseID != "msg_1" || msg.RawStopReason != "end_turn" {
		t.Fatalf("message = %#v", msg)
	}
	if len(msg.Content) != 1 {
		t.Fatalf("content = %#v", msg.Content)
	}
	if tc, ok := msg.Content[0].(TextContent); !ok || tc.Text != "Hello world" {
		t.Fatalf("content[0] = %#v", msg.Content[0])
	}
	if msg.Usage.Input != 10 || msg.Usage.Output != 5 || msg.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %#v", msg.Usage)
	}
}

func TestAnthropicStreamToolCall(t *testing.T) {
	fixture := anthropicSSE("message_start", anthropicMessageStartFixture) +
		anthropicSSE("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"Par"}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"is\"}"}}`) +
		anthropicSSE("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		anthropicSSE("message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}`) +
		anthropicSSE("message_stop", `{"type":"message_stop"}`)

	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "weather?"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	want := []EventType{EventStart, EventToolCallStart, EventToolCallDelta, EventToolCallDelta, EventToolCallDelta, EventToolCallEnd, EventDone}
	got := eventTypes(events)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	// Progressive partial parse, same contract as the completions port.
	first := events[2].Partial.Content[0].(ToolCall)
	if first.Name != "get_weather" || len(first.Arguments) != 0 {
		t.Fatalf("partial after first delta = %#v", first)
	}
	second := events[3].Partial.Content[0].(ToolCall)
	if second.Arguments["location"] != "Par" {
		t.Fatalf("partial after second delta = %#v", second)
	}

	end := events[5].ToolCall
	if end == nil || end.ID != "toolu_1" || end.Name != "get_weather" {
		t.Fatalf("toolcall_end = %#v", end)
	}
	if end.Arguments["location"] != "Paris" {
		t.Fatalf("arguments = %#v", end.Arguments)
	}

	final := events[len(events)-1]
	if final.Reason != StopToolUse || final.Message.StopReason != StopToolUse {
		t.Fatalf("final = %#v", final)
	}
	if final.Message.RawStopReason != "tool_use" {
		t.Fatalf("rawStopReason = %q", final.Message.RawStopReason)
	}
}

func TestAnthropicStreamThinking(t *testing.T) {
	fixture := anthropicSSE("message_start", anthropicMessageStartFixture) +
		anthropicSSE("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me "}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-part-1"}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"-part-2"}}`) +
		anthropicSSE("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		anthropicSSE("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-redacted-payload"}}`) +
		anthropicSSE("content_block_stop", `{"type":"content_block_stop","index":1}`) +
		anthropicSSE("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9,"output_tokens_details":{"thinking_tokens":7}}}`) +
		anthropicSSE("message_stop", `{"type":"message_stop"}`)

	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "q"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	want := []EventType{
		EventStart,
		EventThinkingStart, EventThinkingDelta, EventThinkingDelta, EventThinkingEnd,
		EventThinkingStart, EventThinkingEnd,
		EventDone,
	}
	got := eventTypes(events)
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	final := events[len(events)-1].Message
	if len(final.Content) != 2 {
		t.Fatalf("content = %#v", final.Content)
	}
	thinking, ok := final.Content[0].(ThinkingContent)
	if !ok || thinking.Thinking != "let me think" {
		t.Fatalf("thinking = %#v", final.Content[0])
	}
	if thinking.ThinkingSignature != "sig-part-1-part-2" {
		t.Fatalf("thinkingSignature = %q", thinking.ThinkingSignature)
	}
	redacted, ok := final.Content[1].(ThinkingContent)
	if !ok || !redacted.Redacted || redacted.Thinking != "[Reasoning redacted]" || redacted.ThinkingSignature != "opaque-redacted-payload" {
		t.Fatalf("redacted = %#v", final.Content[1])
	}
	if final.Usage.Reasoning == nil || *final.Usage.Reasoning != 7 {
		t.Fatalf("usage = %#v", final.Usage)
	}
}

func TestAnthropicStreamCacheUsage(t *testing.T) {
	fixture := anthropicSSE("message_start", `{"type":"message_start","message":{"id":"msg_c","type":"message","role":"assistant","model":"claude-test","usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":30,"cache_creation_input_tokens":20,"cache_creation":{"ephemeral_1h_input_tokens":8}}}}`) +
		anthropicSSE("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`) +
		anthropicSSE("content_block_stop", `{"type":"content_block_stop","index":0}`) +
		anthropicSSE("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`) +
		anthropicSSE("message_stop", `{"type":"message_stop"}`)

	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	model.Cost = ModelCost{ModelCostRates: ModelCostRates{Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75}}
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	final := events[len(events)-1]
	if final.Type != EventDone {
		t.Fatalf("last = %#v (%v)", final, final.Error)
	}
	u := final.Message.Usage
	if u.Input != 100 || u.Output != 5 || u.CacheRead != 30 || u.CacheWrite != 20 {
		t.Fatalf("usage = %#v", u)
	}
	if u.CacheWrite1h == nil || *u.CacheWrite1h != 8 {
		t.Fatalf("cacheWrite1h = %#v", u.CacheWrite1h)
	}
	if u.TotalTokens != 155 {
		t.Fatalf("totalTokens = %d", u.TotalTokens)
	}
	if u.Cost.Total <= 0 {
		t.Fatalf("cost = %#v", u.Cost)
	}
}

func TestAnthropicStreamHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`)
	}))
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	if len(events) != 1 {
		t.Fatalf("events = %v", eventTypes(events))
	}
	last := events[0]
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

func TestAnthropicStreamEndedBeforeMessageStop(t *testing.T) {
	fixture := anthropicSSE("message_start", anthropicMessageStartFixture) +
		anthropicSSE("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`) +
		anthropicSSE("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`)

	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventError || !strings.Contains(last.Error.ErrorMessage, "anthropic stream ended before message_stop") {
		t.Fatalf("last = %#v (%v)", last, last.Error)
	}
}

func TestAnthropicStreamAPIErrorEvent(t *testing.T) {
	fixture := anthropicSSE("message_start", anthropicMessageStartFixture) +
		anthropicSSE("error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)

	srv := sseServer(t, fixture)
	defer srv.Close()

	model := testAnthropicModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	es := StreamAnthropicMessages(model, conv, &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}})

	events := drainEvents(es)
	last := events[len(events)-1]
	if last.Type != EventError || last.Reason != StopError {
		t.Fatalf("last = %#v", last)
	}
	if !strings.Contains(last.Error.ErrorMessage, "Overloaded") {
		t.Fatalf("errorMessage = %q", last.Error.ErrorMessage)
	}
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

const anthropicCaptureResponse = `{"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","usage":{"input_tokens":1,"output_tokens":1}}}`

// captureAnthropicRequest runs one request against an httptest server and
// returns the decoded request body and headers.
func captureAnthropicRequest(t *testing.T, model *Model, conv *Context, opts *AnthropicOptions) (map[string]any, http.Header) {
	t.Helper()
	var captured map[string]any
	var headers http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			anthropicSSE("message_start", anthropicCaptureResponse)+
				anthropicSSE("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)+
				anthropicSSE("message_stop", `{"type":"message_stop"}`))
	}))
	defer srv.Close()
	model.BaseURL = srv.URL
	drainEvents(StreamAnthropicMessages(model, conv, opts))
	return captured, headers
}

func anthropicTestTools() []Tool {
	return []Tool{
		{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		},
		{
			Name:        "get_time",
			Description: "Get time",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"tz":{"type":"string"}}}`),
		},
	}
}

func TestAnthropicRequestHeadersAndTools(t *testing.T) {
	model := testAnthropicModel("")
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools:    anthropicTestTools(),
	}
	opts := &AnthropicOptions{
		StreamOptions: StreamOptions{APIKey: "test-key", SessionID: "sess-1"},
		Compat: &AnthropicMessagesCompat{
			// Legacy beta header instead of per-tool eager_input_streaming.
			SupportsEagerToolInputStreaming: boolPtr(false),
			SendSessionAffinityHeaders:      boolPtr(true),
		},
	}
	captured, headers := captureAnthropicRequest(t, model, conv, opts)

	if headers.Get("x-api-key") != "test-key" {
		t.Fatalf("x-api-key = %q", headers.Get("x-api-key"))
	}
	if headers.Get("anthropic-version") != "2023-06-01" {
		t.Fatalf("anthropic-version = %q", headers.Get("anthropic-version"))
	}
	beta := headers.Get("anthropic-beta")
	if !strings.Contains(beta, "fine-grained-tool-streaming-2025-05-14") {
		t.Fatalf("anthropic-beta = %q", beta)
	}
	// Interleaved thinking is requested by default.
	if !strings.Contains(beta, "interleaved-thinking-2025-05-14") {
		t.Fatalf("anthropic-beta = %q", beta)
	}
	if headers.Get("x-session-affinity") != "sess-1" {
		t.Fatalf("x-session-affinity = %q", headers.Get("x-session-affinity"))
	}

	tools, ok := captured["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("tools = %#v", captured["tools"])
	}
	first := tools[0].(map[string]any)
	if first["name"] != "get_weather" || first["description"] != "Get weather" {
		t.Fatalf("tool = %#v", first)
	}
	// Per-tool eager_input_streaming is omitted when compat disables it.
	if _, ok := first["eager_input_streaming"]; ok {
		t.Fatalf("eager_input_streaming should be omitted: %#v", first)
	}
	schema := first["input_schema"].(map[string]any)
	if schema["type"] != "object" {
		t.Fatalf("input_schema = %#v", schema)
	}
	props := schema["properties"].(map[string]any)
	if _, ok := props["location"]; !ok {
		t.Fatalf("properties = %#v", props)
	}
	required, err := json.Marshal(schema["required"])
	if err != nil || string(required) != `["location"]` {
		t.Fatalf("required = %s", required)
	}
	// cache_control lands on the last tool only (short retention default).
	if _, ok := first["cache_control"]; ok {
		t.Fatalf("first tool should not have cache_control: %#v", first)
	}
	lastTool := tools[1].(map[string]any)
	cc, ok := lastTool["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" {
		t.Fatalf("last tool cache_control = %#v", lastTool)
	}
	if _, ok := cc["ttl"]; ok {
		t.Fatalf("short retention must not set ttl: %#v", cc)
	}
}

func TestAnthropicRequestBodySystemCacheAndThinking(t *testing.T) {
	model := testAnthropicModel("")
	model.Reasoning = true
	temp := 0.7
	thinkingOn := true
	conv := &Context{
		SystemPrompt: "be helpful",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	opts := &AnthropicOptions{
		StreamOptions: StreamOptions{
			APIKey:         "k",
			Temperature:    &temp,
			CacheRetention: CacheRetentionLong,
		},
		ThinkingEnabled:      &thinkingOn,
		ThinkingBudgetTokens: 2048,
	}
	captured, _ := captureAnthropicRequest(t, model, conv, opts)

	if captured["model"] != "claude-test" || captured["stream"] != true || captured["max_tokens"] != 8192.0 {
		t.Fatalf("captured = %#v", captured)
	}

	system, ok := captured["system"].([]any)
	if !ok || len(system) != 1 {
		t.Fatalf("system = %#v", captured["system"])
	}
	sysBlock := system[0].(map[string]any)
	if sysBlock["type"] != "text" || sysBlock["text"] != "be helpful" {
		t.Fatalf("system block = %#v", sysBlock)
	}
	cc, ok := sysBlock["cache_control"].(map[string]any)
	if !ok || cc["type"] != "ephemeral" || cc["ttl"] != "1h" {
		t.Fatalf("system cache_control = %#v", sysBlock)
	}

	msgs := captured["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %#v", msgs)
	}
	user := msgs[0].(map[string]any)
	if user["role"] != "user" {
		t.Fatalf("user message = %#v", user)
	}
	// String content is wrapped into a block to carry cache_control.
	content := user["content"].([]any)
	last := content[len(content)-1].(map[string]any)
	if last["type"] != "text" || last["text"] != "hi" {
		t.Fatalf("user content = %#v", content)
	}
	userCC, ok := last["cache_control"].(map[string]any)
	if !ok || userCC["ttl"] != "1h" {
		t.Fatalf("user cache_control = %#v", last)
	}

	thinking, ok := captured["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"] != 2048.0 || thinking["display"] != "summarized" {
		t.Fatalf("thinking = %#v", captured["thinking"])
	}
	// Temperature is incompatible with extended thinking.
	if _, ok := captured["temperature"]; ok {
		t.Fatalf("temperature should be omitted when thinking is enabled: %#v", captured["temperature"])
	}
}

func TestAnthropicRequestTemperatureWithoutThinking(t *testing.T) {
	model := testAnthropicModel("")
	temp := 0.3
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k", Temperature: &temp}}
	captured, _ := captureAnthropicRequest(t, model, conv, opts)
	if captured["temperature"] != 0.3 {
		t.Fatalf("temperature = %#v", captured["temperature"])
	}
	if _, ok := captured["thinking"]; ok {
		t.Fatalf("thinking should be omitted for a non-reasoning model: %#v", captured["thinking"])
	}

	// compat.SupportsTemperature=false suppresses the field entirely.
	opts.Compat = &AnthropicMessagesCompat{SupportsTemperature: boolPtr(false)}
	captured, _ = captureAnthropicRequest(t, model, conv, opts)
	if _, ok := captured["temperature"]; ok {
		t.Fatalf("temperature should be suppressed by compat: %#v", captured["temperature"])
	}
}

func TestAnthropicRequestAdaptiveThinking(t *testing.T) {
	model := testAnthropicModel("")
	model.Reasoning = true
	thinkingOn := true
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	opts := &AnthropicOptions{
		StreamOptions:   StreamOptions{APIKey: "k"},
		ThinkingEnabled: &thinkingOn,
		Effort:          "high",
		Compat:          &AnthropicMessagesCompat{ForceAdaptiveThinking: boolPtr(true)},
	}
	captured, headers := captureAnthropicRequest(t, model, conv, opts)

	thinking, ok := captured["thinking"].(map[string]any)
	if !ok || thinking["type"] != "adaptive" || thinking["display"] != "summarized" {
		t.Fatalf("thinking = %#v", captured["thinking"])
	}
	outputConfig, ok := captured["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] != "high" {
		t.Fatalf("output_config = %#v", captured["output_config"])
	}
	// Adaptive models have interleaved thinking built in: no beta header.
	if beta := headers.Get("anthropic-beta"); strings.Contains(beta, "interleaved-thinking") {
		t.Fatalf("anthropic-beta = %q", beta)
	}
}

func TestAnthropicConvertMessagesReplay(t *testing.T) {
	model := testAnthropicModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "sig"},
					TextContent{Type: "text", Text: "answer"},
					ToolCall{Type: "toolCall", ID: "call_1|x", Name: "do_thing", Arguments: map[string]any{"x": 1.0}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1|x", ToolName: "do_thing", IsError: true, Content: []ContentBlock{TextContent{Type: "text", Text: "failed"}}},
		},
	}
	opts := &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}}
	captured, _ := captureAnthropicRequest(t, model, conv, opts)

	msgs := captured["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %#v", msgs)
	}
	assistant := msgs[1].(map[string]any)
	if assistant["role"] != "assistant" {
		t.Fatalf("assistant = %#v", assistant)
	}
	blocks := assistant["content"].([]any)
	if len(blocks) != 3 {
		t.Fatalf("assistant blocks = %#v", blocks)
	}
	thinkingBlock := blocks[0].(map[string]any)
	if thinkingBlock["type"] != "thinking" || thinkingBlock["thinking"] != "hmm" || thinkingBlock["signature"] != "sig" {
		t.Fatalf("thinking block = %#v", thinkingBlock)
	}
	toolUse := blocks[2].(map[string]any)
	// Same-model replay: TransformMessages leaves tool call IDs untouched, so
	// the original "|" survives. Normalization is cross-model only and is
	// covered by TestAnthropicConvertMessagesCrossModel.
	if toolUse["type"] != "tool_use" || toolUse["id"] != "call_1|x" || toolUse["name"] != "do_thing" {
		t.Fatalf("tool_use = %#v", toolUse)
	}
	input := toolUse["input"].(map[string]any)
	if input["x"] != 1.0 {
		t.Fatalf("tool_use input = %#v", input)
	}

	toolResultMsg := msgs[2].(map[string]any)
	if toolResultMsg["role"] != "user" {
		t.Fatalf("tool result message = %#v", toolResultMsg)
	}
	trContent := toolResultMsg["content"].([]any)
	tr := trContent[0].(map[string]any)
	if tr["type"] != "tool_result" || tr["tool_use_id"] != "call_1|x" || tr["is_error"] != true || tr["content"] != "failed" {
		t.Fatalf("tool_result = %#v", tr)
	}
}

// TestAnthropicConvertMessagesCrossModel covers the cross-model branch of
// TransformMessages: thinking blocks are downgraded to text (their signatures
// are only valid for the producing model) and tool call IDs are rewritten to
// Anthropic's [a-zA-Z0-9_-] pattern, including the matching tool result.
func TestAnthropicConvertMessagesCrossModel(t *testing.T) {
	model := testAnthropicModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				// A different provider/model than the request target.
				Role: "assistant", API: APIOpenAICompletions, Provider: "openai", Model: "gpt-test",
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "sig"},
					TextContent{Type: "text", Text: "answer"},
					ToolCall{Type: "toolCall", ID: "call_1|x", Name: "do_thing", Arguments: map[string]any{"x": 1.0}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1|x", ToolName: "do_thing", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		},
	}
	opts := &AnthropicOptions{StreamOptions: StreamOptions{APIKey: "k"}}
	captured, _ := captureAnthropicRequest(t, model, conv, opts)

	msgs := captured["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %#v", msgs)
	}
	blocks := msgs[1].(map[string]any)["content"].([]any)
	if len(blocks) != 3 {
		t.Fatalf("assistant blocks = %#v", blocks)
	}
	// Thinking became plain text rather than a thinking block.
	if first := blocks[0].(map[string]any); first["type"] != "text" || first["text"] != "hmm" {
		t.Fatalf("thinking downgrade = %#v", first)
	}
	toolUse := blocks[2].(map[string]any)
	if toolUse["type"] != "tool_use" || toolUse["id"] != "call_1_x" {
		t.Fatalf("tool_use = %#v", toolUse)
	}
	// The tool result must follow the rewritten ID or Anthropic rejects it.
	tr := msgs[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if tr["type"] != "tool_result" || tr["tool_use_id"] != "call_1_x" {
		t.Fatalf("tool_result = %#v", tr)
	}
}

func TestAnthropicStreamSimpleReasoningBudget(t *testing.T) {
	model := testAnthropicModel("")
	model.Reasoning = true
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			anthropicSSE("message_start", anthropicCaptureResponse)+
				anthropicSSE("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`)+
				anthropicSSE("message_stop", `{"type":"message_stop"}`))
	}))
	defer srv.Close()
	// BaseURL must be set before the stream starts, or the request never
	// reaches the test server and nothing is captured.
	model.BaseURL = srv.URL
	drainEvents(StreamAnthropicMessagesSimple(model, conv, &SimpleStreamOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Reasoning:     ThinkingMedium,
	}))

	// medium budget is 8192, but base maxTokens (the model's 8192 cap) leaves
	// no room for an answer, so the budget drops to maxTokens-MinAnswerTokens.
	thinking, ok := captured["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" || thinking["budget_tokens"] != 7168.0 {
		t.Fatalf("thinking = %#v", captured["thinking"])
	}
}

// TestAnthropicUnknownStopReasonKeepsTheTurn covers a stop reason the port has
// not seen. Erroring out discarded a complete, already-paid-for response --
// Anthropic adds stop reasons over time, and model_context_window_exceeded is
// one this port did not handle.
func TestAnthropicUnknownStopReasonKeepsTheTurn(t *testing.T) {
	for _, reason := range []string{"model_context_window_exceeded", "some_future_reason"} {
		t.Run(reason, func(t *testing.T) {
			stop, message, err := mapAnthropicStopReason(reason, "")
			if err != nil {
				t.Fatalf("mapAnthropicStopReason(%q) errored, discarding the turn: %v", reason, err)
			}
			if stop == StopError {
				t.Fatalf("stop = %q, want the response kept", stop)
			}
			if message != "" {
				t.Fatalf("message = %q", message)
			}
		})
	}
	// A context-window overflow is a length stop, so the truncation guard fires.
	if stop, _, _ := mapAnthropicStopReason("model_context_window_exceeded", ""); stop != StopLength {
		t.Fatalf("model_context_window_exceeded mapped to %q, want StopLength", stop)
	}
}

// TestAnthropicThinkingSendsTheEnlargedMaxTokens covers the extended-thinking
// budget. max_tokens is spent on reasoning as well as the answer, which is why
// the adjustment enlarges it -- but the enlarged value was computed and then
// dropped, so the request carried the original and the model ran out of room
// mid-answer.
func TestAnthropicThinkingSendsTheEnlargedMaxTokens(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `event: message_start
data: {"type":"message_start","message":{"id":"m1","model":"claude-test","role":"assistant","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`)
	}))
	defer server.Close()

	model := &Model{
		ID: "claude-test", Name: "Claude Test", API: APIAnthropicMessages, Provider: "anthropic",
		BaseURL: server.URL, Input: []string{"text"}, ContextWindow: 200000, MaxTokens: 64000,
		Reasoning: true,
		ThinkingLevelMap: map[ThinkingLevel]*string{
			ThinkingHigh: func() *string { s := "high"; return &s }(),
		},
	}
	options := &SimpleStreamOptions{Reasoning: ThinkingHigh}
	options.APIKey = "k"
	options.MaxTokens = 4096

	if _, err := StreamAnthropicMessagesSimple(model,
		&Context{Messages: []Message{UserMessage{Role: "user", Content: "think hard"}}},
		options).Result(t.Context()); err != nil {
		t.Fatalf("stream: %v", err)
	}

	sent, ok := captured["max_tokens"].(float64)
	if !ok {
		t.Fatalf("max_tokens missing from the request: %#v", captured)
	}
	budget, ok := captured["thinking"].(map[string]any)["budget_tokens"].(float64)
	if !ok {
		t.Fatalf("thinking budget missing: %#v", captured["thinking"])
	}
	// The answer needs room beyond the reasoning budget; the un-enlarged
	// value would leave none.
	if sent <= budget {
		t.Fatalf("max_tokens = %v is not greater than the thinking budget %v: the enlarged value was not sent",
			sent, budget)
	}
}
