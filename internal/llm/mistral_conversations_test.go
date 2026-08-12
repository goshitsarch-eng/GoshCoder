package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testMistralModel(baseURL string) *Model {
	return &Model{
		ID:            "mistral-medium-latest",
		Name:          "Mistral Medium",
		API:           APIMistralConversations,
		Provider:      "mistral",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 128000,
		MaxTokens:     8192,
	}
}

// mistralSSE formats one Mistral streaming chunk.
func mistralSSE(payload string) string {
	return "data: " + payload + "\n\n"
}

// mistralFinish is a terminal chunk with a finish reason and usage.
const mistralFinish = `{"id":"c1","choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`

func mistralServer(t *testing.T, body string) *Model {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	t.Cleanup(srv.Close)
	return testMistralModel(srv.URL)
}

// captureMistralRequest runs a stream against a recording server and returns
// the decoded body, headers, and request path.
func captureMistralRequest(t *testing.T, model *Model, conv *Context, opts *MistralOptions) (map[string]any, http.Header, string) {
	t.Helper()
	var captured map[string]any
	var headers http.Header
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		path = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, mistralSSE(mistralFinish))
	}))
	defer srv.Close()
	model.BaseURL = srv.URL
	drainEvents(StreamMistral(model, conv, opts))
	return captured, headers, path
}

func TestMistralStreamTextOnly(t *testing.T) {
	model := mistralServer(t,
		mistralSSE(`{"id":"c1","choices":[{"delta":{"content":"Hello"}}]}`)+
			mistralSSE(`{"id":"c1","choices":[{"delta":{"content":" world"}}]}`)+
			mistralSSE(mistralFinish)+
			mistralSSE("[DONE]"))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	want := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
	final := events[len(events)-1]
	if final.Reason != StopStop {
		t.Fatalf("reason = %q", final.Reason)
	}
	if text := final.Message.Content[0].(TextContent); text.Text != "Hello world" {
		t.Fatalf("text = %q", text.Text)
	}
	if final.Message.ResponseID != "c1" {
		t.Fatalf("responseId = %q", final.Message.ResponseID)
	}
	if final.Message.Usage.Input != 10 || final.Message.Usage.Output != 4 {
		t.Fatalf("usage = %#v", final.Message.Usage)
	}
}

// TestMistralStructuredContentChunks covers the array form of delta.content,
// which is how Mistral streams thinking alongside text.
func TestMistralStructuredContentChunks(t *testing.T) {
	model := mistralServer(t,
		mistralSSE(`{"id":"c1","choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"text":"pondering"}]}]}}]}`)+
			mistralSSE(`{"id":"c1","choices":[{"delta":{"content":[{"type":"text","text":"answer"}]}}]}`)+
			mistralSSE(mistralFinish))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	want := []EventType{EventStart, EventThinkingStart, EventThinkingDelta, EventThinkingEnd, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	content := events[len(events)-1].Message.Content
	if thinking := content[0].(ThinkingContent); thinking.Thinking != "pondering" {
		t.Fatalf("thinking = %#v", thinking)
	}
	if text := content[1].(TextContent); text.Text != "answer" {
		t.Fatalf("text = %#v", text)
	}
}

func TestMistralStreamToolCall(t *testing.T) {
	model := mistralServer(t,
		mistralSSE(`{"id":"c1","choices":[{"delta":{"tool_calls":[{"id":"abc123def","index":0,"function":{"name":"get_weather","arguments":"{\"location\":"}}]}}]}`)+
			mistralSSE(`{"id":"c1","choices":[{"delta":{"tool_calls":[{"id":"abc123def","index":0,"function":{"name":"get_weather","arguments":"\"Paris\"}"}}]}}]}`)+
			mistralSSE(`{"id":"c1","choices":[{"finish_reason":"tool_calls","delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "weather?"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	want := []EventType{EventStart, EventToolCallStart, EventToolCallDelta, EventToolCallDelta, EventToolCallEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	final := events[len(events)-1]
	if final.Reason != StopToolUse {
		t.Fatalf("reason = %q", final.Reason)
	}
	tc := final.Message.Content[0].(ToolCall)
	if tc.ID != "abc123def" || tc.Name != "get_weather" || tc.Arguments["location"] != "Paris" {
		t.Fatalf("toolCall = %#v", tc)
	}
}

func TestMistralStopReasons(t *testing.T) {
	tests := []struct {
		finishReason string
		wantReason   StopReason
	}{
		{"stop", StopStop},
		{"length", StopLength},
		{"model_length", StopLength},
		{"tool_calls", StopToolUse},
	}
	for _, tt := range tests {
		t.Run(tt.finishReason, func(t *testing.T) {
			model := mistralServer(t, mistralSSE(fmt.Sprintf(
				`{"id":"c1","choices":[{"finish_reason":%q,"delta":{"content":"x"}}]}`, tt.finishReason)))
			conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
			events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

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

func TestMistralUnknownStopReasonErrors(t *testing.T) {
	model := mistralServer(t, mistralSSE(`{"id":"c1","choices":[{"finish_reason":"content_filter","delta":{}}]}`))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "Provider stopped with: content_filter") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestMistralStreamWithoutFinishReasonErrors(t *testing.T) {
	model := mistralServer(t, mistralSSE(`{"id":"c1","choices":[{"delta":{"content":"partial"}}]}`))
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "without a finish reason") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestMistralUsageSubtractsCachedTokens(t *testing.T) {
	model := mistralServer(t, mistralSSE(
		`{"id":"c1","choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":30}}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	usage := events[len(events)-1].Message.Usage
	if usage.Input != 70 || usage.CacheRead != 30 || usage.Output != 20 {
		t.Fatalf("usage = %#v", usage)
	}
}

// TestMistralUsageClampsCachedTokens covers a provider reporting more cached
// tokens than prompt tokens.
func TestMistralUsageClampsCachedTokens(t *testing.T) {
	model := mistralServer(t, mistralSSE(
		`{"id":"c1","choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,"num_cached_tokens":999}}`))

	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	usage := events[len(events)-1].Message.Usage
	if usage.Input != 0 || usage.CacheRead != 10 {
		t.Fatalf("usage = %#v, want cacheRead clamped to promptTokens", usage)
	}
}

func TestMistralRequestBasics(t *testing.T) {
	model := testMistralModel("")
	conv := &Context{
		SystemPrompt: "be brief",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	captured, headers, path := captureMistralRequest(t, model, conv, &MistralOptions{
		StreamOptions: StreamOptions{APIKey: "sk-mistral", MaxTokens: 512},
	})

	if path != "/v1/chat/completions" {
		t.Fatalf("path = %q", path)
	}
	if headers.Get("Authorization") != "Bearer sk-mistral" {
		t.Fatalf("authorization = %q", headers.Get("Authorization"))
	}
	if headers.Get("Accept") != "text/event-stream" {
		t.Fatalf("accept = %q", headers.Get("Accept"))
	}
	if captured["model"] != "mistral-medium-latest" || captured["stream"] != true {
		t.Fatalf("params = %#v", captured)
	}
	// maxTokens is remapped to the snake_case wire field.
	if captured["max_tokens"] != 512.0 {
		t.Fatalf("max_tokens = %#v", captured["max_tokens"])
	}

	// The system prompt is prepended as a message.
	messages := captured["messages"].([]any)
	system := messages[0].(map[string]any)
	if system["role"] != "system" || system["content"] != "be brief" {
		t.Fatalf("system message = %#v", system)
	}
}

func TestMistralToolConversion(t *testing.T) {
	model := testMistralModel("")
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get weather",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
		}},
	}
	captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	tool := captured["tools"].([]any)[0].(map[string]any)
	function := tool["function"].(map[string]any)
	if tool["type"] != "function" || function["name"] != "get_weather" {
		t.Fatalf("tool = %#v", tool)
	}
	// Mistral always sends strict, defaulting to false.
	if function["strict"] != false {
		t.Fatalf("strict = %#v", function["strict"])
	}
}

// TestMistralStrictToolSampling covers a json_schema constrained tool, which
// Mistral can always honor.
func TestMistralStrictToolSampling(t *testing.T) {
	model := testMistralModel("")
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:                "t",
			Description:         "d",
			Parameters:          json.RawMessage(`{"type":"object"}`),
			ConstrainedSampling: json.RawMessage(`{"type":"json_schema","strict":"require"}`),
		}},
	}
	captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	function := captured["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if function["strict"] != true {
		t.Fatalf("strict = %#v, want true", function["strict"])
	}
}

func TestMistralToolCallIDNormalization(t *testing.T) {
	// Exactly 9 alphanumeric characters passes through unchanged.
	if got := deriveMistralToolCallID("abc123def", 0); got != "abc123def" {
		t.Fatalf("derive(abc123def) = %q", got)
	}
	// Anything else is hashed to 9 characters.
	got := deriveMistralToolCallID("call_a_very_long_identifier", 0)
	if len(got) != mistralToolCallIDLength {
		t.Fatalf("derived id = %q, want %d characters", got, mistralToolCallIDLength)
	}
	if mistralNonAlphanumeric.MatchString(got) {
		t.Fatalf("derived id %q contains non-alphanumeric characters", got)
	}
	// Different attempts produce different ids, which is how collisions resolve.
	if deriveMistralToolCallID("x", 0) == deriveMistralToolCallID("x", 1) {
		t.Fatal("attempts should produce distinct ids")
	}
}

func TestMistralToolCallIDNormalizerIsStableAndUnique(t *testing.T) {
	normalize := mistralToolCallIDNormalizer()
	model := testMistralModel("")

	first := normalize("call_one", model, nil)
	// The same input maps to the same id.
	if again := normalize("call_one", model, nil); again != first {
		t.Fatalf("normalizer is not stable: %q vs %q", first, again)
	}
	// Distinct inputs never collide.
	second := normalize("call_two", model, nil)
	if second == first {
		t.Fatalf("distinct ids collided: %q", second)
	}
}

func TestMistralReplaysAssistantAndToolResult(t *testing.T) {
	model := testMistralModel("")
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content: []ContentBlock{
					ThinkingContent{Type: "thinking", Thinking: "hmm"},
					TextContent{Type: "text", Text: "calling"},
					ToolCall{Type: "toolCall", ID: "abc123def", Name: "do_thing", Arguments: map[string]any{"x": 1.0}},
				},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "abc123def", ToolName: "do_thing", Content: []ContentBlock{
				TextContent{Type: "text", Text: "ok"},
			}},
		},
	}
	captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	messages := captured["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("messages = %d: %#v", len(messages), messages)
	}

	assistant := messages[1].(map[string]any)
	if assistant["role"] != "assistant" || assistant["prefix"] != false {
		t.Fatalf("assistant = %#v", assistant)
	}
	content := assistant["content"].([]any)
	// Thinking replays as a structured thinking chunk.
	thinking := content[0].(map[string]any)
	if thinking["type"] != "thinking" {
		t.Fatalf("thinking chunk = %#v", thinking)
	}
	inner := thinking["thinking"].([]any)[0].(map[string]any)
	if inner["text"] != "hmm" {
		t.Fatalf("thinking text = %#v", inner)
	}

	call := assistant["tool_calls"].([]any)[0].(map[string]any)
	if call["id"] != "abc123def" {
		t.Fatalf("tool call = %#v", call)
	}
	if call["function"].(map[string]any)["arguments"] != `{"x":1}` {
		t.Fatalf("arguments = %#v", call["function"])
	}

	// Tool results are a content array on a "tool" role message.
	result := messages[2].(map[string]any)
	if result["role"] != "tool" || result["tool_call_id"] != "abc123def" || result["name"] != "do_thing" {
		t.Fatalf("tool result = %#v", result)
	}
	resultText := result["content"].([]any)[0].(map[string]any)
	if resultText["text"] != "ok" {
		t.Fatalf("tool result text = %#v", resultText)
	}
}

func TestMistralToolResultTextVariants(t *testing.T) {
	tests := []struct {
		name           string
		text           string
		hasImages      bool
		supportsImages bool
		isError        bool
		want           string
	}{
		{"plain text", "output", false, false, false, "output"},
		{"error prefix", "boom", false, false, true, "[tool error] boom"},
		{"no output", "", false, false, false, "(no tool output)"},
		{"error without output", "", false, false, true, "[tool error] (no tool output)"},
		{"image on a vision model", "", true, true, false, "(see attached image)"},
		{"image on a text model", "", true, false, false, "(image omitted: model does not support images)"},
		{"text plus dropped image", "output", true, false, false, "output\n[tool image omitted: model does not support images]"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mistralToolResultText(tt.text, tt.hasImages, tt.supportsImages, tt.isError)
			if got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMistralImageHandling(t *testing.T) {
	t.Run("vision model keeps images", func(t *testing.T) {
		model := testMistralModel("")
		model.Input = []string{"text", "image"}
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: []ContentBlock{
			TextContent{Type: "text", Text: "look"},
			ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
		}}}}
		captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
		})

		content := captured["messages"].([]any)[0].(map[string]any)["content"].([]any)
		if len(content) != 2 {
			t.Fatalf("content = %#v", content)
		}
		image := content[1].(map[string]any)
		if image["type"] != "image_url" || image["image_url"] != "data:image/png;base64,AAAA" {
			t.Fatalf("image chunk = %#v", image)
		}
	})

	t.Run("text model drops images with a placeholder", func(t *testing.T) {
		model := testMistralModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: []ContentBlock{
			ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
		}}}}
		captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
		})

		// TransformMessages already replaced the image with placeholder text, so
		// the message survives as a text chunk rather than being dropped.
		messages := captured["messages"].([]any)
		if len(messages) != 1 {
			t.Fatalf("messages = %#v", messages)
		}
	})
}

func TestMistralPromptCaching(t *testing.T) {
	t.Run("session id enables caching and affinity", func(t *testing.T) {
		model := testMistralModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		captured, headers, _ := captureMistralRequest(t, model, conv, &MistralOptions{
			StreamOptions: StreamOptions{APIKey: "k", SessionID: "sess-1"},
		})

		if captured["prompt_cache_key"] != "sess-1" {
			t.Fatalf("prompt_cache_key = %#v", captured["prompt_cache_key"])
		}
		if headers.Get("x-affinity") != "sess-1" {
			t.Fatalf("x-affinity = %q", headers.Get("x-affinity"))
		}
	})

	t.Run("cacheRetention none disables it", func(t *testing.T) {
		model := testMistralModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		captured, headers, _ := captureMistralRequest(t, model, conv, &MistralOptions{
			StreamOptions: StreamOptions{APIKey: "k", SessionID: "sess-1", CacheRetention: CacheRetentionNone},
		})

		if _, present := captured["prompt_cache_key"]; present {
			t.Fatalf("prompt_cache_key should be omitted: %#v", captured["prompt_cache_key"])
		}
		if headers.Get("x-affinity") != "" {
			t.Fatalf("x-affinity = %q, want empty", headers.Get("x-affinity"))
		}
	})

	t.Run("an explicit affinity header wins", func(t *testing.T) {
		model := testMistralModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		custom := "custom-affinity"
		_, headers, _ := captureMistralRequest(t, model, conv, &MistralOptions{
			StreamOptions: StreamOptions{
				APIKey:    "k",
				SessionID: "sess-1",
				Headers:   ProviderHeaders{"x-affinity": &custom},
			},
		})
		if headers.Get("x-affinity") != "custom-affinity" {
			t.Fatalf("x-affinity = %q", headers.Get("x-affinity"))
		}
	})
}

func TestMistralToolChoice(t *testing.T) {
	for _, choice := range []string{"auto", "none", "any", "required"} {
		t.Run(choice, func(t *testing.T) {
			model := testMistralModel("")
			conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
			captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
				StreamOptions: StreamOptions{APIKey: "k"},
				ToolChoice:    choice,
			})
			if captured["tool_choice"] != choice {
				t.Fatalf("tool_choice = %#v", captured["tool_choice"])
			}
		})
	}

	t.Run("unknown string is omitted", func(t *testing.T) {
		model := testMistralModel("")
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			ToolChoice:    "nonsense",
		})
		if _, present := captured["tool_choice"]; present {
			t.Fatalf("tool_choice = %#v, want omitted", captured["tool_choice"])
		}
	})
}

func TestMistralReasoningModeSelection(t *testing.T) {
	runSimple := func(t *testing.T, modelID string, reasoning ThinkingLevel) map[string]any {
		t.Helper()
		var captured map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&captured)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, mistralSSE(mistralFinish))
		}))
		defer srv.Close()

		model := testMistralModel(srv.URL)
		model.ID = modelID
		model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		drainEvents(StreamMistralSimple(model, conv, &SimpleStreamOptions{
			StreamOptions: StreamOptions{APIKey: "k"},
			Reasoning:     reasoning,
		}))
		return captured
	}

	t.Run("effort models use reasoning_effort", func(t *testing.T) {
		captured := runSimple(t, "mistral-small-latest", ThinkingHigh)
		if captured["reasoning_effort"] != "high" {
			t.Fatalf("reasoning_effort = %#v", captured["reasoning_effort"])
		}
		if _, present := captured["prompt_mode"]; present {
			t.Fatalf("prompt_mode should be omitted: %#v", captured["prompt_mode"])
		}
	})

	t.Run("other models use prompt_mode", func(t *testing.T) {
		captured := runSimple(t, "magistral-medium-latest", ThinkingHigh)
		if captured["prompt_mode"] != "reasoning" {
			t.Fatalf("prompt_mode = %#v", captured["prompt_mode"])
		}
		if _, present := captured["reasoning_effort"]; present {
			t.Fatalf("reasoning_effort should be omitted: %#v", captured["reasoning_effort"])
		}
	})

	t.Run("no reasoning omits both", func(t *testing.T) {
		captured := runSimple(t, "magistral-medium-latest", "")
		for _, key := range []string{"prompt_mode", "reasoning_effort"} {
			if _, present := captured[key]; present {
				t.Fatalf("%s should be omitted: %#v", key, captured[key])
			}
		}
	})
}

// TestMistralNonReasoningModelOmitsThinking covers a model without reasoning
// support: an explicit level must not turn thinking on.
func TestMistralNonReasoningModelOmitsThinking(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, mistralSSE(mistralFinish))
	}))
	defer srv.Close()

	model := testMistralModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	drainEvents(StreamMistralSimple(model, conv, &SimpleStreamOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Reasoning:     ThinkingHigh,
	}))

	for _, key := range []string{"prompt_mode", "reasoning_effort"} {
		if _, present := captured[key]; present {
			t.Fatalf("%s should be omitted for a non-reasoning model", key)
		}
	}
}

func TestMistralNoAPIKeyErrors(t *testing.T) {
	model := testMistralModel("http://127.0.0.1:1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "No API key") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestMistralHTTPErrorIncludesBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"message":"invalid model"}`)
	}))
	defer srv.Close()

	model := testMistralModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamMistral(model, conv, &MistralOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "invalid model") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestMistralSamplingParamsOverride(t *testing.T) {
	model := testMistralModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _, _ := captureMistralRequest(t, model, conv, &MistralOptions{
		StreamOptions: StreamOptions{
			APIKey:         "k",
			MaxTokens:      100,
			SamplingParams: map[string]any{"max_tokens": 2048, "top_p": 0.25},
		},
	})
	if captured["max_tokens"] != 2048.0 {
		t.Fatalf("max_tokens = %#v, want the sampling param to win", captured["max_tokens"])
	}
	if captured["top_p"] != 0.25 {
		t.Fatalf("top_p = %#v", captured["top_p"])
	}
}

func TestMistralRegisteredInRegistry(t *testing.T) {
	funcs, ok := GetStreamer(APIMistralConversations)
	if !ok {
		t.Fatal("mistral-conversations is not registered")
	}
	if funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("registration = %#v", funcs)
	}
}
