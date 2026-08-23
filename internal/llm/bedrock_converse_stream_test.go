package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

func testBedrockModel(baseURL string) *Model {
	return &Model{
		ID:            "anthropic.claude-sonnet-4-5-20250929-v1:0",
		Name:          "Claude Sonnet 4.5",
		API:           APIBedrockConverseStream,
		Provider:      "amazon-bedrock",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 200000,
		MaxTokens:     8192,
	}
}

// bedrockFrame encodes one ConverseStream event frame.
func bedrockFrame(eventType, payload string) []byte {
	return encodeEventStreamMessage(
		map[string]string{":event-type": eventType, ":message-type": "event"},
		[]byte(payload),
	)
}

// bedrockStream concatenates frames into a response body.
func bedrockStream(frames ...[]byte) []byte {
	var out bytes.Buffer
	for _, frame := range frames {
		out.Write(frame)
	}
	return out.Bytes()
}

// bedrockTextTurn is a minimal complete turn: start, one text delta, stop.
func bedrockTextTurn(text string) []byte {
	encoded, _ := json.Marshal(text)
	return bedrockStream(
		bedrockFrame("messageStart", `{"role":"assistant"}`),
		bedrockFrame("contentBlockDelta", fmt.Sprintf(`{"contentBlockIndex":0,"delta":{"text":%s}}`, encoded)),
		bedrockFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		bedrockFrame("messageStop", `{"stopReason":"end_turn"}`),
		bedrockFrame("metadata", `{"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14}}`),
	)
}

// bedrockServer serves a fixed event stream and records requests.
type bedrockServer struct {
	model    *Model
	requests []map[string]any
	headers  []http.Header
	paths    []string
}

func newBedrockServer(t *testing.T, body []byte) *bedrockServer {
	t.Helper()
	server := &bedrockServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var decoded map[string]any
		_ = json.NewDecoder(r.Body).Decode(&decoded)
		server.requests = append(server.requests, decoded)
		server.headers = append(server.headers, r.Header.Clone())
		server.paths = append(server.paths, r.URL.Path)

		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	server.model = testBedrockModel(srv.URL)
	return server
}

// bedrockTestOptions returns options with credentials that avoid touching the
// host machine's AWS configuration.
func bedrockTestOptions() *BedrockOptions {
	return &BedrockOptions{StreamOptions: StreamOptions{Env: map[string]string{
		"AWS_ACCESS_KEY_ID":     "AKIDTEST",
		"AWS_SECRET_ACCESS_KEY": "secret-test",
		"AWS_REGION":            "us-east-1",
	}}}
}

func TestBedrockStreamTextOnly(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("Hello from Bedrock"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	want := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	final := events[len(events)-1]
	if final.Reason != StopStop {
		t.Fatalf("reason = %q", final.Reason)
	}
	if text := final.Message.Content[0].(TextContent); text.Text != "Hello from Bedrock" {
		t.Fatalf("text = %q", text.Text)
	}
	if final.Message.Usage.Input != 10 || final.Message.Usage.Output != 4 {
		t.Fatalf("usage = %#v", final.Message.Usage)
	}
	// The request is a signed POST to the converse-stream path.
	if !strings.HasSuffix(server.paths[0], "/converse-stream") {
		t.Fatalf("path = %q", server.paths[0])
	}
	if auth := server.headers[0].Get("Authorization"); !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 ") {
		t.Fatalf("authorization = %q", auth)
	}
}

func TestBedrockStreamToolCall(t *testing.T) {
	body := bedrockStream(
		bedrockFrame("messageStart", `{"role":"assistant"}`),
		bedrockFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"get_weather"}}}`),
		bedrockFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"location\":"}}}`),
		bedrockFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"Paris\"}"}}}`),
		bedrockFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		bedrockFrame("messageStop", `{"stopReason":"tool_use"}`),
	)
	server := newBedrockServer(t, body)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "weather?"}}}

	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	want := []EventType{EventStart, EventToolCallStart, EventToolCallDelta, EventToolCallDelta, EventToolCallEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	final := events[len(events)-1]
	if final.Reason != StopToolUse {
		t.Fatalf("reason = %q", final.Reason)
	}
	tc := final.Message.Content[0].(ToolCall)
	if tc.ID != "call_1" || tc.Name != "get_weather" || tc.Arguments["location"] != "Paris" {
		t.Fatalf("toolCall = %#v", tc)
	}
}

func TestBedrockStreamReasoning(t *testing.T) {
	body := bedrockStream(
		bedrockFrame("messageStart", `{"role":"assistant"}`),
		bedrockFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"thinking"}}}`),
		// The signature arrives in its own delta and is not streamed onward.
		bedrockFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig-abc"}}}`),
		bedrockFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		bedrockFrame("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"text":"answer"}}`),
		bedrockFrame("contentBlockStop", `{"contentBlockIndex":1}`),
		bedrockFrame("messageStop", `{"stopReason":"end_turn"}`),
	)
	server := newBedrockServer(t, body)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	want := []EventType{
		EventStart,
		EventThinkingStart, EventThinkingDelta, EventThinkingEnd,
		EventTextStart, EventTextDelta, EventTextEnd,
		EventDone,
	}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	content := events[len(events)-1].Message.Content
	thinking := content[0].(ThinkingContent)
	if thinking.Thinking != "thinking" || thinking.ThinkingSignature != "sig-abc" {
		t.Fatalf("thinking = %#v", thinking)
	}
	if text := content[1].(TextContent); text.Text != "answer" {
		t.Fatalf("text = %#v", text)
	}
}

func TestBedrockStopReasons(t *testing.T) {
	tests := []struct {
		stopReason string
		wantReason StopReason
	}{
		{"end_turn", StopStop},
		{"stop_sequence", StopStop},
		{"max_tokens", StopLength},
		{"model_context_window_exceeded", StopLength},
		{"tool_use", StopToolUse},
	}
	for _, tt := range tests {
		t.Run(tt.stopReason, func(t *testing.T) {
			body := bedrockStream(
				bedrockFrame("messageStart", `{"role":"assistant"}`),
				bedrockFrame("messageStop", fmt.Sprintf(`{"stopReason":%q}`, tt.stopReason)),
			)
			server := newBedrockServer(t, body)
			conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
			events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

			final := events[len(events)-1]
			if final.Reason != tt.wantReason {
				t.Fatalf("reason = %q, want %q", final.Reason, tt.wantReason)
			}
			if final.Message.RawStopReason != tt.stopReason {
				t.Fatalf("rawStopReason = %q", final.Message.RawStopReason)
			}
		})
	}
}

func TestBedrockUnknownStopReasonErrors(t *testing.T) {
	body := bedrockStream(
		bedrockFrame("messageStart", `{"role":"assistant"}`),
		bedrockFrame("messageStop", `{"stopReason":"guardrail_intervened"}`),
	)
	server := newBedrockServer(t, body)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "guardrail_intervened") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

// TestBedrockMidStreamExceptionFrame covers a modeled exception delivered as a
// stream frame rather than an HTTP error.
func TestBedrockMidStreamExceptionFrame(t *testing.T) {
	exception := encodeEventStreamMessage(map[string]string{
		":message-type":   "exception",
		":exception-type": "ThrottlingException",
	}, []byte(`{"message":"Too many requests"}`))

	body := bedrockStream(bedrockFrame("messageStart", `{"role":"assistant"}`), exception)
	server := newBedrockServer(t, body)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	// The human-readable prefix is what the retry logic matches on.
	if !strings.Contains(final.Error.ErrorMessage, "Throttling error: Too many requests") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

// TestBedrockDataRetentionHint covers the docs pointer appended to data
// retention failures.
func TestBedrockDataRetentionHint(t *testing.T) {
	exception := encodeEventStreamMessage(map[string]string{
		":message-type":   "exception",
		":exception-type": "ValidationException",
	}, []byte(`{"message":"data retention mode 'default' is not available for this model"}`))

	server := newBedrockServer(t, bedrockStream(exception))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	message := events[len(events)-1].Error.ErrorMessage
	if !strings.Contains(message, bedrockDataRetentionDocsURL) {
		t.Fatalf("errorMessage = %q, want the docs URL", message)
	}
}

func TestBedrockStreamWithoutStopReasonErrors(t *testing.T) {
	body := bedrockStream(
		bedrockFrame("messageStart", `{"role":"assistant"}`),
		bedrockFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"partial"}}`),
	)
	server := newBedrockServer(t, body)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "without a stop reason") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestBedrockRejectsNonAssistantMessageStart(t *testing.T) {
	server := newBedrockServer(t, bedrockStream(bedrockFrame("messageStart", `{"role":"user"}`)))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "message start") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestBedrockBuildParams(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{
		SystemPrompt: "be brief",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	options := bedrockTestOptions()
	options.MaxTokens = 1024
	temperature := 0.3
	options.Temperature = &temperature

	drainEvents(StreamBedrock(server.model, conv, options))

	request := server.requests[0]
	inference := request["inferenceConfig"].(map[string]any)
	if inference["maxTokens"] != 1024.0 || inference["temperature"] != 0.3 {
		t.Fatalf("inferenceConfig = %#v", inference)
	}

	// The system prompt is its own content block list.
	system := request["system"].([]any)
	if system[0].(map[string]any)["text"] != "be brief" {
		t.Fatalf("system = %#v", system)
	}
	messages := request["messages"].([]any)
	user := messages[0].(map[string]any)
	if user["role"] != "user" {
		t.Fatalf("user message = %#v", user)
	}
	if user["content"].([]any)[0].(map[string]any)["text"] != "hi" {
		t.Fatalf("user content = %#v", user["content"])
	}
}

// TestBedrockClaudeDefaultsMaxTokens covers Claude on Bedrock, which requires
// an explicit output cap.
func TestBedrockClaudeDefaultsMaxTokens(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	inference := server.requests[0]["inferenceConfig"].(map[string]any)
	if inference["maxTokens"] != 8192.0 {
		t.Fatalf("maxTokens = %#v, want the model cap", inference["maxTokens"])
	}
}

func TestBedrockToolConfig(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
		}},
	}
	options := bedrockTestOptions()
	options.ToolChoice = "any"

	drainEvents(StreamBedrock(server.model, conv, options))

	config := server.requests[0]["toolConfig"].(map[string]any)
	spec := config["tools"].([]any)[0].(map[string]any)["toolSpec"].(map[string]any)
	if spec["name"] != "get_weather" || spec["description"] != "Get weather" {
		t.Fatalf("toolSpec = %#v", spec)
	}
	if _, ok := spec["inputSchema"].(map[string]any)["json"]; !ok {
		t.Fatalf("inputSchema = %#v", spec["inputSchema"])
	}
	// strict is omitted unless the tool asks for it.
	if _, hasStrict := spec["strict"]; hasStrict {
		t.Fatalf("strict should be omitted: %#v", spec)
	}
	if _, ok := config["toolChoice"].(map[string]any)["any"]; !ok {
		t.Fatalf("toolChoice = %#v", config["toolChoice"])
	}
}

// TestBedrockToolChoiceNoneOmitsTools confirms "none" drops the whole tool
// config rather than sending an empty choice.
func TestBedrockToolChoiceNoneOmitsTools(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools:    []Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	options := bedrockTestOptions()
	options.ToolChoice = "none"

	drainEvents(StreamBedrock(server.model, conv, options))

	if _, present := server.requests[0]["toolConfig"]; present {
		t.Fatalf("toolConfig should be omitted: %#v", server.requests[0]["toolConfig"])
	}
}

func TestBedrockStrictToolSampling(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	supportsStrict := true
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:                "t",
			Description:         "d",
			Parameters:          json.RawMessage(`{"type":"object"}`),
			ConstrainedSampling: json.RawMessage(`{"type":"json_schema","strict":"prefer"}`),
		}},
	}
	options := bedrockTestOptions()
	options.Compat = &BedrockCompat{SupportsStrictMode: &supportsStrict}

	drainEvents(StreamBedrock(server.model, conv, options))

	spec := server.requests[0]["toolConfig"].(map[string]any)["tools"].([]any)[0].(map[string]any)["toolSpec"].(map[string]any)
	if spec["strict"] != true {
		t.Fatalf("strict = %#v, want true", spec["strict"])
	}
}

func TestBedrockReplaysAssistantAndToolResults(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: server.model.API, Provider: server.model.Provider, Model: server.model.ID,
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "sig"},
					ToolCall{Type: "toolCall", ID: "call_1", Name: "f", Arguments: map[string]any{"x": 1.0}},
					ToolCall{Type: "toolCall", ID: "call_2", Name: "g", Arguments: map[string]any{}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_1", ToolName: "f", Content: []ContentBlock{TextContent{Type: "text", Text: "one"}}},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call_2", ToolName: "g", IsError: true, Content: []ContentBlock{TextContent{Type: "text", Text: "boom"}}},
		},
	}

	drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	messages := server.requests[0]["messages"].([]any)
	// user, assistant, and one merged tool-result turn.
	if len(messages) != 3 {
		t.Fatalf("messages = %d: %#v", len(messages), messages)
	}

	assistant := messages[1].(map[string]any)
	content := assistant["content"].([]any)
	reasoning := content[0].(map[string]any)["reasoningContent"].(map[string]any)["reasoningText"].(map[string]any)
	if reasoning["text"] != "hmm" || reasoning["signature"] != "sig" {
		t.Fatalf("reasoningContent = %#v", reasoning)
	}
	toolUse := content[1].(map[string]any)["toolUse"].(map[string]any)
	if toolUse["toolUseId"] != "call_1" || toolUse["name"] != "f" {
		t.Fatalf("toolUse = %#v", toolUse)
	}

	// Bedrock requires every consecutive tool result in one user turn.
	results := messages[2].(map[string]any)
	if results["role"] != "user" {
		t.Fatalf("tool result role = %#v", results["role"])
	}
	blocks := results["content"].([]any)
	// Two tool results, plus the trailing cache point this Claude model gets on
	// the last user turn.
	if len(blocks) != 3 {
		t.Fatalf("tool result blocks = %#v", blocks)
	}
	if _, hasCachePoint := blocks[2].(map[string]any)["cachePoint"]; !hasCachePoint {
		t.Fatalf("last block = %#v, want a cache point", blocks[2])
	}
	first := blocks[0].(map[string]any)["toolResult"].(map[string]any)
	second := blocks[1].(map[string]any)["toolResult"].(map[string]any)
	if first["status"] != "success" || second["status"] != "error" {
		t.Fatalf("statuses = %#v, %#v", first["status"], second["status"])
	}
}

// TestBedrockUnsignedThinkingBecomesText covers a persisted thinking block with
// no signature: Bedrock rejects it, so it replays as plain text.
func TestBedrockUnsignedThinkingBecomesText(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: server.model.API, Provider: server.model.Provider, Model: server.model.ID,
				StopReason: StopStop,
				Content:    []ContentBlock{ThinkingContent{Type: "thinking", Thinking: "unsigned"}},
			},
		},
	}

	drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	content := server.requests[0]["messages"].([]any)[1].(map[string]any)["content"].([]any)
	block := content[0].(map[string]any)
	if block["text"] != "unsigned" {
		t.Fatalf("block = %#v, want plain text", block)
	}
	if _, hasReasoning := block["reasoningContent"]; hasReasoning {
		t.Fatal("an unsigned thinking block must not replay as reasoningContent")
	}
}

// TestBedrockNonClaudeOmitsThinkingSignature covers models that reject the
// signature field outright.
func TestBedrockNonClaudeOmitsThinkingSignature(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	server.model.ID = "amazon.nova-pro-v1:0"
	server.model.Name = "Nova Pro"
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: server.model.API, Provider: server.model.Provider, Model: server.model.ID,
				StopReason: StopStop,
				Content:    []ContentBlock{ThinkingContent{Type: "thinking", Thinking: "hmm", ThinkingSignature: "sig"}},
			},
		},
	}

	drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	content := server.requests[0]["messages"].([]any)[1].(map[string]any)["content"].([]any)
	reasoning := content[0].(map[string]any)["reasoningContent"].(map[string]any)["reasoningText"].(map[string]any)
	if reasoning["text"] != "hmm" {
		t.Fatalf("reasoningText = %#v", reasoning)
	}
	if _, hasSignature := reasoning["signature"]; hasSignature {
		t.Fatal("a non-Claude model must not receive a signature")
	}
}

// TestBedrockSkipsEmptyAssistantMessage covers an aborted turn, which leaves an
// empty assistant message that Bedrock would reject.
func TestBedrockSkipsEmptyAssistantMessage(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: server.model.API, Provider: server.model.Provider, Model: server.model.ID,
				StopReason: StopAborted,
				Content:    []ContentBlock{},
			},
		},
	}

	drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	messages := server.requests[0]["messages"].([]any)
	if len(messages) != 1 {
		t.Fatalf("messages = %d, want the empty assistant turn dropped", len(messages))
	}
}

// TestBedrockBlankTextGetsPlaceholder covers Bedrock rejecting blank text.
func TestBedrockBlankTextGetsPlaceholder(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "   "}}}

	drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

	content := server.requests[0]["messages"].([]any)[0].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["text"] != bedrockEmptyTextPlaceholder {
		t.Fatalf("content = %#v, want the placeholder", content)
	}
}

func TestBedrockPromptCaching(t *testing.T) {
	t.Run("claude gets cache points", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		conv := &Context{
			SystemPrompt: "be brief",
			Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
		}
		drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

		system := server.requests[0]["system"].([]any)
		if len(system) != 2 {
			t.Fatalf("system = %#v, want a trailing cache point", system)
		}
		if _, ok := system[1].(map[string]any)["cachePoint"]; !ok {
			t.Fatalf("system[1] = %#v", system[1])
		}
		// The last user turn also carries a cache point.
		content := server.requests[0]["messages"].([]any)[0].(map[string]any)["content"].([]any)
		if _, ok := content[len(content)-1].(map[string]any)["cachePoint"]; !ok {
			t.Fatalf("user content = %#v", content)
		}
	})

	t.Run("long retention sets a ttl", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		conv := &Context{SystemPrompt: "s", Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options := bedrockTestOptions()
		options.CacheRetention = CacheRetentionLong

		drainEvents(StreamBedrock(server.model, conv, options))

		point := server.requests[0]["system"].([]any)[1].(map[string]any)["cachePoint"].(map[string]any)
		if point["ttl"] != "1h" {
			t.Fatalf("cachePoint = %#v", point)
		}
	})

	t.Run("retention none omits cache points", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		conv := &Context{SystemPrompt: "s", Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options := bedrockTestOptions()
		options.CacheRetention = CacheRetentionNone

		drainEvents(StreamBedrock(server.model, conv, options))

		if system := server.requests[0]["system"].([]any); len(system) != 1 {
			t.Fatalf("system = %#v, want no cache point", system)
		}
	})

	t.Run("non-claude models get none", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		server.model.ID = "amazon.nova-pro-v1:0"
		server.model.Name = "Nova Pro"
		conv := &Context{SystemPrompt: "s", Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

		drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

		if system := server.requests[0]["system"].([]any); len(system) != 1 {
			t.Fatalf("system = %#v, want no cache point", system)
		}
	})
}

func TestBedrockSupportsPromptCaching(t *testing.T) {
	tests := []struct {
		id   string
		name string
		want bool
	}{
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5", true},
		{"anthropic.claude-3-7-sonnet-20250219-v1:0", "Claude 3.7 Sonnet", true},
		{"anthropic.claude-3-5-haiku-20241022-v1:0", "Claude 3.5 Haiku", true},
		{"anthropic.claude-opus-5", "Claude Opus 5", true},
		{"anthropic.claude-3-sonnet-20240229-v1:0", "Claude 3 Sonnet", false},
		{"amazon.nova-pro-v1:0", "Nova Pro", false},
	}
	for _, tt := range tests {
		model := &Model{ID: tt.id, Name: tt.name}
		if got := bedrockSupportsPromptCaching(model, nil); got != tt.want {
			t.Errorf("bedrockSupportsPromptCaching(%q) = %v, want %v", tt.id, got, tt.want)
		}
	}

	// An application inference profile ARN hides the model name, so caching can
	// be forced through the environment.
	arn := &Model{ID: "arn:aws:bedrock:us-east-1:123:application-inference-profile/abc", Name: "profile"}
	if bedrockSupportsPromptCaching(arn, nil) {
		t.Fatal("an opaque ARN should not enable caching by default")
	}
	if !bedrockSupportsPromptCaching(arn, map[string]string{"AWS_BEDROCK_FORCE_CACHE": "1"}) {
		t.Fatal("AWS_BEDROCK_FORCE_CACHE should enable caching")
	}
}

func TestBedrockAdaptiveThinkingDetection(t *testing.T) {
	tests := []struct {
		id       string
		name     string
		adaptive bool
		xhigh    bool
	}{
		{"anthropic.claude-opus-4-6", "Claude Opus 4.6", true, false},
		{"anthropic.claude-opus-4-7", "Claude Opus 4.7", true, true},
		{"anthropic.claude-sonnet-4-6", "Claude Sonnet 4.6", true, false},
		{"anthropic.claude-sonnet-5", "Claude Sonnet 5", true, true},
		{"anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5", false, false},
	}
	for _, tt := range tests {
		model := &Model{ID: tt.id, Name: tt.name}
		if got := bedrockSupportsAdaptiveThinking(model); got != tt.adaptive {
			t.Errorf("adaptive(%q) = %v, want %v", tt.id, got, tt.adaptive)
		}
		if got := bedrockSupportsNativeXHigh(model); got != tt.xhigh {
			t.Errorf("nativeXHigh(%q) = %v, want %v", tt.id, got, tt.xhigh)
		}
	}
}

func TestBedrockThinkingFields(t *testing.T) {
	t.Run("budget-based claude", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		server.model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options := bedrockTestOptions()
		options.Reasoning = ThinkingMedium

		drainEvents(StreamBedrock(server.model, conv, options))

		fields := server.requests[0]["additionalModelRequestFields"].(map[string]any)
		thinking := fields["thinking"].(map[string]any)
		if thinking["type"] != "enabled" || thinking["budget_tokens"] != 8192.0 {
			t.Fatalf("thinking = %#v", thinking)
		}
		if thinking["display"] != "summarized" {
			t.Fatalf("display = %#v", thinking["display"])
		}
		// Interleaved thinking is on by default for budget-based Claude.
		if fields["anthropic_beta"].([]any)[0] != anthropicInterleavedThinkingBeta {
			t.Fatalf("anthropic_beta = %#v", fields["anthropic_beta"])
		}
	})

	t.Run("adaptive claude uses an effort level", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		server.model.ID = "anthropic.claude-opus-4-7"
		server.model.Name = "Claude Opus 4.7"
		server.model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options := bedrockTestOptions()
		options.Reasoning = ThinkingXHigh

		drainEvents(StreamBedrock(server.model, conv, options))

		fields := server.requests[0]["additionalModelRequestFields"].(map[string]any)
		if fields["thinking"].(map[string]any)["type"] != "adaptive" {
			t.Fatalf("thinking = %#v", fields["thinking"])
		}
		if fields["output_config"].(map[string]any)["effort"] != "xhigh" {
			t.Fatalf("output_config = %#v", fields["output_config"])
		}
		// Adaptive models have interleaved thinking built in.
		if _, present := fields["anthropic_beta"]; present {
			t.Fatalf("anthropic_beta should be omitted: %#v", fields["anthropic_beta"])
		}
	})

	t.Run("govcloud omits display", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		server.model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options := bedrockTestOptions()
		options.Reasoning = ThinkingMedium
		options.Region = "us-gov-west-1"
		options.Env["AWS_REGION"] = "us-gov-west-1"

		drainEvents(StreamBedrock(server.model, conv, options))

		thinking := server.requests[0]["additionalModelRequestFields"].(map[string]any)["thinking"].(map[string]any)
		if _, present := thinking["display"]; present {
			t.Fatalf("display should be omitted on GovCloud: %#v", thinking)
		}
	})

	t.Run("no reasoning omits the field", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		server.model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

		drainEvents(StreamBedrock(server.model, conv, bedrockTestOptions()))

		if _, present := server.requests[0]["additionalModelRequestFields"]; present {
			t.Fatal("additionalModelRequestFields should be omitted")
		}
	})

	t.Run("non-reasoning model omits the field", func(t *testing.T) {
		server := newBedrockServer(t, bedrockTextTurn("ok"))
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options := bedrockTestOptions()
		options.Reasoning = ThinkingHigh

		drainEvents(StreamBedrock(server.model, conv, options))

		if _, present := server.requests[0]["additionalModelRequestFields"]; present {
			t.Fatal("a non-reasoning model must not get thinking fields")
		}
	})
}

func TestBedrockRegionResolution(t *testing.T) {
	clearAmbientAWSEnv(t)
	// An empty AWS config keeps the host machine's profile out of the result.
	emptyFiles := func(t *testing.T) map[string]string {
		return map[string]string{
			"AWS_SHARED_CREDENTIALS_FILE": filepath.Join(t.TempDir(), "credentials"),
			"AWS_CONFIG_FILE":             filepath.Join(t.TempDir(), "config"),
		}
	}

	t.Run("ARN region wins", func(t *testing.T) {
		model := testBedrockModel("")
		model.ID = "arn:aws:bedrock:eu-west-3:123:inference-profile/x"
		options := &BedrockOptions{StreamOptions: StreamOptions{Env: emptyFiles(t)}, Region: "us-east-2"}
		region, _ := resolveBedrockTarget(model, options)
		if region != "eu-west-3" {
			t.Fatalf("region = %q", region)
		}
	})

	t.Run("explicit option beats env", func(t *testing.T) {
		env := emptyFiles(t)
		env["AWS_REGION"] = "us-west-1"
		options := &BedrockOptions{StreamOptions: StreamOptions{Env: env}, Region: "ap-south-1"}
		region, _ := resolveBedrockTarget(testBedrockModel(""), options)
		if region != "ap-south-1" {
			t.Fatalf("region = %q", region)
		}
	})

	t.Run("env region is used", func(t *testing.T) {
		env := emptyFiles(t)
		env["AWS_DEFAULT_REGION"] = "sa-east-1"
		options := &BedrockOptions{StreamOptions: StreamOptions{Env: env}}
		region, endpoint := resolveBedrockTarget(testBedrockModel(""), options)
		if region != "sa-east-1" {
			t.Fatalf("region = %q", region)
		}
		if endpoint != "https://bedrock-runtime.sa-east-1.amazonaws.com" {
			t.Fatalf("endpoint = %q", endpoint)
		}
	})

	t.Run("standard endpoint supplies the region", func(t *testing.T) {
		model := testBedrockModel("https://bedrock-runtime.eu-central-1.amazonaws.com")
		options := &BedrockOptions{StreamOptions: StreamOptions{Env: emptyFiles(t)}}
		region, endpoint := resolveBedrockTarget(model, options)
		if region != "eu-central-1" {
			t.Fatalf("region = %q", region)
		}
		if endpoint != "https://bedrock-runtime.eu-central-1.amazonaws.com" {
			t.Fatalf("endpoint = %q", endpoint)
		}
	})

	t.Run("a custom endpoint is preserved", func(t *testing.T) {
		model := testBedrockModel("https://bedrock.internal.example.com/proxy")
		env := emptyFiles(t)
		env["AWS_REGION"] = "us-east-1"
		options := &BedrockOptions{StreamOptions: StreamOptions{Env: env}}
		_, endpoint := resolveBedrockTarget(model, options)
		if endpoint != "https://bedrock.internal.example.com/proxy" {
			t.Fatalf("endpoint = %q", endpoint)
		}
	})

	t.Run("default region", func(t *testing.T) {
		options := &BedrockOptions{StreamOptions: StreamOptions{Env: emptyFiles(t)}}
		region, _ := resolveBedrockTarget(testBedrockModel(""), options)
		if region != bedrockDefaultRegion {
			t.Fatalf("region = %q", region)
		}
	})
}

func TestStandardBedrockEndpointRegion(t *testing.T) {
	tests := []struct {
		baseURL string
		want    string
	}{
		{"https://bedrock-runtime.us-east-1.amazonaws.com", "us-east-1"},
		{"https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com", "us-gov-west-1"},
		{"https://bedrock-runtime.cn-north-1.amazonaws.com.cn", "cn-north-1"},
		{"https://proxy.example.com", ""},
		{"not-a-url", ""},
		{"", ""},
	}
	for _, tt := range tests {
		if got := standardBedrockEndpointRegion(tt.baseURL); got != tt.want {
			t.Errorf("standardBedrockEndpointRegion(%q) = %q, want %q", tt.baseURL, got, tt.want)
		}
	}
}

func TestBedrockBearerTokenSkipsSigning(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	options := bedrockTestOptions()
	options.BearerToken = "bedrock-api-key"

	drainEvents(StreamBedrock(server.model, conv, options))

	auth := server.headers[0].Get("Authorization")
	if auth != "Bearer bedrock-api-key" {
		t.Fatalf("authorization = %q", auth)
	}
	// A bearer token bypasses SigV4 entirely.
	if server.headers[0].Get("X-Amz-Date") != "" {
		t.Fatal("a bearer request must not be signed")
	}
}

// TestBedrockBearerTokenFromEnv covers AWS_BEARER_TOKEN_BEDROCK.
func TestBedrockBearerTokenFromEnv(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	options := &BedrockOptions{StreamOptions: StreamOptions{Env: map[string]string{
		"AWS_BEARER_TOKEN_BEDROCK": "env-token",
		"AWS_REGION":               "us-east-1",
	}}}

	drainEvents(StreamBedrock(server.model, conv, options))

	if auth := server.headers[0].Get("Authorization"); auth != "Bearer env-token" {
		t.Fatalf("authorization = %q", auth)
	}
}

// TestBedrockIgnoresAmbientSentinel confirms the catalog's ambient marker is
// never sent as a literal token.
func TestBedrockIgnoresAmbientSentinel(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	options := bedrockTestOptions()
	options.APIKey = "<authenticated>"

	drainEvents(StreamBedrock(server.model, conv, options))

	auth := server.headers[0].Get("Authorization")
	if strings.Contains(auth, "<authenticated>") {
		t.Fatalf("the sentinel leaked into the request: %q", auth)
	}
	// It falls through to SigV4 instead.
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 ") {
		t.Fatalf("authorization = %q", auth)
	}
}

func TestBedrockSkipAuthStillSigns(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	options := &BedrockOptions{StreamOptions: StreamOptions{Env: map[string]string{
		"AWS_BEDROCK_SKIP_AUTH": "1",
		"AWS_REGION":            "us-east-1",
	}}}

	drainEvents(StreamBedrock(server.model, conv, options))

	// Unauthenticated proxies still expect a well-formed signature.
	if auth := server.headers[0].Get("Authorization"); !strings.Contains(auth, "dummy-access-key") {
		t.Fatalf("authorization = %q", auth)
	}
}

func TestBedrockMissingCredentialsErrors(t *testing.T) {
	clearAmbientAWSEnv(t)
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	// Empty file paths shadow any ambient credentials on the host.
	options := &BedrockOptions{StreamOptions: StreamOptions{Env: map[string]string{
		"AWS_REGION":                  "us-east-1",
		"AWS_SHARED_CREDENTIALS_FILE": filepath.Join(t.TempDir(), "missing-credentials"),
		"AWS_CONFIG_FILE":             filepath.Join(t.TempDir(), "missing-config"),
	}}}

	events := drainEvents(StreamBedrock(server.model, conv, options))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "no AWS credentials found") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestBedrockCustomHeadersAreSigned(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	options := bedrockTestOptions()
	custom := "custom-value"
	reserved := "should-be-ignored"
	options.Headers = ProviderHeaders{
		"x-custom-header": &custom,
		// Reserved headers must not be overridable.
		"x-amz-date":    &reserved,
		"authorization": &reserved,
	}

	drainEvents(StreamBedrock(server.model, conv, options))

	headers := server.headers[0]
	if headers.Get("x-custom-header") != "custom-value" {
		t.Fatalf("custom header = %q", headers.Get("x-custom-header"))
	}
	if headers.Get("Authorization") == reserved {
		t.Fatal("authorization was overridden by a caller header")
	}
	// The custom header is covered by the signature.
	if !strings.Contains(headers.Get("Authorization"), "x-custom-header") {
		t.Fatalf("custom header was not signed: %q", headers.Get("Authorization"))
	}
}

func TestIsBedrockReservedHeader(t *testing.T) {
	for _, name := range []string{"Authorization", "host", "X-Amz-Date", "x-amz-content-sha256"} {
		if !isBedrockReservedHeader(name) {
			t.Errorf("%q should be reserved", name)
		}
	}
	for _, name := range []string{"x-custom", "content-type", "accept"} {
		if isBedrockReservedHeader(name) {
			t.Errorf("%q should not be reserved", name)
		}
	}
}

func TestSanitizeBedrockDocument(t *testing.T) {
	input := map[string]any{
		"keep": "value",
		"":     "dropped",
		"nested": map[string]any{
			"":     "dropped",
			"deep": []any{map[string]any{"": "dropped", "ok": 1}},
		},
	}
	got := sanitizeBedrockDocument(input).(map[string]any)

	if _, present := got[""]; present {
		t.Fatal("the empty key was not dropped")
	}
	nested := got["nested"].(map[string]any)
	if _, present := nested[""]; present {
		t.Fatal("a nested empty key was not dropped")
	}
	deep := nested["deep"].([]any)[0].(map[string]any)
	if _, present := deep[""]; present {
		t.Fatal("an empty key inside an array was not dropped")
	}
	if deep["ok"] != 1 {
		t.Fatalf("array element = %#v", deep)
	}
}

func TestBedrockImageBlock(t *testing.T) {
	for mimeType, want := range map[string]string{
		"image/jpeg": "jpeg",
		"image/jpg":  "jpeg",
		"image/png":  "png",
		"image/gif":  "gif",
		"image/webp": "webp",
	} {
		block, err := bedrockImageBlock(mimeType, "AAAA")
		if err != nil {
			t.Fatalf("bedrockImageBlock(%q): %v", mimeType, err)
		}
		image := block["image"].(map[string]any)
		if image["format"] != want {
			t.Errorf("format for %q = %#v, want %q", mimeType, image["format"], want)
		}
	}

	if _, err := bedrockImageBlock("image/tiff", "AAAA"); err == nil {
		t.Fatal("an unknown mime type must be rejected")
	}
	if _, err := bedrockImageBlock("image/png", "not base64!!"); err == nil {
		t.Fatal("invalid base64 must be rejected")
	}
}

func TestBedrockHTTPErrorIncludesBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprint(w, `{"message":"not authorized"}`)
	}))
	defer srv.Close()

	model := testBedrockModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamBedrock(model, conv, bedrockTestOptions()))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "not authorized") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestBedrockStreamSimpleBudgetCarveOut(t *testing.T) {
	server := newBedrockServer(t, bedrockTextTurn("ok"))
	server.model.Reasoning = true
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}

	drainEvents(StreamBedrockSimple(server.model, conv, &SimpleStreamOptions{
		StreamOptions: StreamOptions{Env: bedrockTestOptions().Env},
		Reasoning:     ThinkingMedium,
	}))

	request := server.requests[0]
	thinking := request["additionalModelRequestFields"].(map[string]any)["thinking"].(map[string]any)
	maxTokens := request["inferenceConfig"].(map[string]any)["maxTokens"].(float64)
	budget := thinking["budget_tokens"].(float64)

	// The budget must leave room for an answer.
	if budget >= maxTokens {
		t.Fatalf("budget %v must be below maxTokens %v", budget, maxTokens)
	}
	if budget != maxTokens-float64(MinAnswerTokens) && budget != 8192 {
		t.Fatalf("budget = %v with maxTokens = %v", budget, maxTokens)
	}
}

func TestBedrockRegisteredInRegistry(t *testing.T) {
	funcs, ok := GetStreamer(APIBedrockConverseStream)
	if !ok {
		t.Fatal("bedrock-converse-stream is not registered")
	}
	if funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("registration = %#v", funcs)
	}
}
