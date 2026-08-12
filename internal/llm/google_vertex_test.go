package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testVertexModel(baseURL string) *Model {
	return &Model{
		ID:            "gemini-2.5-flash",
		Name:          "Gemini 2.5 Flash",
		API:           APIGoogleVertex,
		Provider:      "google-vertex",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 1000000,
		MaxTokens:     65536,
	}
}

// captureVertexRequest runs a stream against a recording server and returns the
// decoded request body, headers, and request URI. The test server is used as a
// custom base URL, which bypasses project/location host construction.
func captureVertexRequest(t *testing.T, model *Model, conv *Context, opts *GoogleVertexOptions) (map[string]any, http.Header, string) {
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
	// A version segment keeps the port from being mistaken for one.
	model.BaseURL = srv.URL + "/v1"
	drainEvents(StreamGoogleVertex(model, conv, opts))
	return captured, headers, requestURI
}

func TestVertexResolveAPIKey(t *testing.T) {
	tests := []struct {
		name   string
		apiKey string
		want   string
	}{
		{"real key is used", "AIzaSy-real", "AIzaSy-real"},
		{"empty means ambient credentials", "", ""},
		// pi stores this marker when Vertex should use ADC.
		{"credentials marker means ambient", gcpVertexCredentialsMarker, ""},
		{"placeholder means ambient", "<YOUR_API_KEY>", ""},
		{"whitespace is trimmed", "  AIzaSy-real  ", "AIzaSy-real"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveVertexAPIKey(&GoogleVertexOptions{StreamOptions: StreamOptions{APIKey: tt.apiKey}})
			if got != tt.want {
				t.Fatalf("resolveVertexAPIKey(%q) = %q, want %q", tt.apiKey, got, tt.want)
			}
		})
	}
}

func TestVertexResolveProjectAndLocation(t *testing.T) {
	t.Run("explicit options win", func(t *testing.T) {
		opts := &GoogleVertexOptions{
			StreamOptions: StreamOptions{Env: map[string]string{"GOOGLE_CLOUD_PROJECT": "env-proj"}},
			Project:       "opt-proj",
			Location:      "us-central1",
		}
		project, err := resolveVertexProject(opts)
		if err != nil || project != "opt-proj" {
			t.Fatalf("project = %q, err = %v", project, err)
		}
	})

	t.Run("env fallbacks", func(t *testing.T) {
		opts := &GoogleVertexOptions{StreamOptions: StreamOptions{Env: map[string]string{
			"GOOGLE_CLOUD_PROJECT":  "env-proj",
			"GOOGLE_CLOUD_LOCATION": "europe-west4",
		}}}
		project, err := resolveVertexProject(opts)
		if err != nil || project != "env-proj" {
			t.Fatalf("project = %q, err = %v", project, err)
		}
		location, err := resolveVertexLocation(opts)
		if err != nil || location != "europe-west4" {
			t.Fatalf("location = %q, err = %v", location, err)
		}
	})

	t.Run("GCLOUD_PROJECT is also accepted", func(t *testing.T) {
		opts := &GoogleVertexOptions{StreamOptions: StreamOptions{Env: map[string]string{"GCLOUD_PROJECT": "gcloud-proj"}}}
		project, err := resolveVertexProject(opts)
		if err != nil || project != "gcloud-proj" {
			t.Fatalf("project = %q, err = %v", project, err)
		}
	})

	t.Run("missing project errors", func(t *testing.T) {
		// An explicit empty env map keeps os.Getenv from leaking in.
		opts := &GoogleVertexOptions{StreamOptions: StreamOptions{Env: map[string]string{"GOOGLE_CLOUD_PROJECT": ""}}}
		if _, err := resolveVertexProject(opts); err == nil {
			t.Fatal("expected an error when no project is configured")
		} else if !strings.Contains(err.Error(), "requires a project ID") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("missing location errors", func(t *testing.T) {
		opts := &GoogleVertexOptions{}
		if _, err := resolveVertexLocation(opts); err == nil {
			t.Fatal("expected an error when no location is configured")
		} else if !strings.Contains(err.Error(), "requires a location") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestVertexHost(t *testing.T) {
	if got := vertexHost("us-central1"); got != "https://us-central1-aiplatform.googleapis.com" {
		t.Fatalf("regional host = %q", got)
	}
	// The global location uses the unprefixed host.
	if got := vertexHost("global"); got != "https://aiplatform.googleapis.com" {
		t.Fatalf("global host = %q", got)
	}
}

func TestVertexResolveCustomBaseURL(t *testing.T) {
	// The catalog stores a template, which is not a usable endpoint.
	if got := resolveVertexCustomBaseURL("https://{location}-aiplatform.googleapis.com"); got != "" {
		t.Fatalf("template base URL = %q, want empty", got)
	}
	if got := resolveVertexCustomBaseURL("  https://proxy.example.com/v1  "); got != "https://proxy.example.com/v1" {
		t.Fatalf("custom base URL = %q", got)
	}
	if got := resolveVertexCustomBaseURL(""); got != "" {
		t.Fatalf("empty base URL = %q", got)
	}
}

func TestVertexBaseURLIncludesAPIVersion(t *testing.T) {
	tests := []struct {
		baseURL string
		want    bool
	}{
		{"https://proxy.example.com/v1", true},
		{"https://proxy.example.com/v1beta1", true},
		{"https://proxy.example.com/v1/", true},
		{"https://proxy.example.com", false},
		{"https://proxy.example.com/api", false},
	}
	for _, tt := range tests {
		if got := vertexBaseURLIncludesAPIVersion(tt.baseURL); got != tt.want {
			t.Errorf("vertexBaseURLIncludesAPIVersion(%q) = %v, want %v", tt.baseURL, got, tt.want)
		}
	}
}

func TestVertexBuildResourceURL(t *testing.T) {
	t.Run("default regional host", func(t *testing.T) {
		// The catalog's template base URL is ignored in favor of the location.
		model := testVertexModel("https://{location}-aiplatform.googleapis.com")
		got, err := buildVertexResourceURL(model, &GoogleVertexOptions{
			Project:  "my-proj",
			Location: "us-central1",
		})
		if err != nil {
			t.Fatal(err)
		}
		want := "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
		if got != want {
			t.Fatalf("url = %q,\nwant %q", got, want)
		}
	})

	t.Run("custom base URL without a version gets one", func(t *testing.T) {
		model := testVertexModel("https://proxy.example.com")
		got, err := buildVertexResourceURL(model, &GoogleVertexOptions{Project: "p", Location: "us-central1"})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(got, "https://proxy.example.com/v1/projects/p/") {
			t.Fatalf("url = %q", got)
		}
	})

	t.Run("custom base URL with a version is left alone", func(t *testing.T) {
		model := testVertexModel("https://proxy.example.com/v1beta1")
		got, err := buildVertexResourceURL(model, &GoogleVertexOptions{Project: "p", Location: "us-central1"})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(got, "https://proxy.example.com/v1beta1/projects/p/") {
			t.Fatalf("url = %q", got)
		}
		if strings.Contains(got, "/v1beta1/v1/") {
			t.Fatalf("version segment was doubled: %q", got)
		}
	})
}

// TestVertexBuildExpressURL covers express mode, which addresses publisher
// models directly with no project or location.
func TestVertexBuildExpressURL(t *testing.T) {
	model := testVertexModel("https://{location}-aiplatform.googleapis.com")
	got, err := buildVertexExpressURL(model)
	if err != nil {
		t.Fatal(err)
	}
	want := "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
	if got != want {
		t.Fatalf("url = %q,\nwant %q", got, want)
	}
}

func TestVertexExpressModeUsesAPIKey(t *testing.T) {
	model := testVertexModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	_, headers, requestURI := captureVertexRequest(t, model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{APIKey: "AIzaSy-real"},
	})

	// No project or location in the path, and no bearer token.
	if !strings.HasPrefix(requestURI, "/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent") {
		t.Fatalf("request path = %q", requestURI)
	}
	if !strings.Contains(requestURI, "alt=sse") {
		t.Fatalf("request query = %q", requestURI)
	}
	if headers.Get("x-goog-api-key") != "AIzaSy-real" {
		t.Fatalf("x-goog-api-key = %q", headers.Get("x-goog-api-key"))
	}
	if headers.Get("Authorization") != "" {
		t.Fatalf("Authorization should be unset in express mode, got %q", headers.Get("Authorization"))
	}
}

// TestVertexUsesPreFetchedAccessToken covers the GOOGLE_OAUTH_ACCESS_TOKEN
// path, which is how callers supply `gcloud auth print-access-token` output.
func TestVertexUsesPreFetchedAccessToken(t *testing.T) {
	ClearGoogleTokenCache()
	t.Cleanup(ClearGoogleTokenCache)

	model := testVertexModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	_, headers, requestURI := captureVertexRequest(t, model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{Env: map[string]string{"GOOGLE_OAUTH_ACCESS_TOKEN": "ya29.token"}},
		Project:       "my-proj",
		Location:      "us-central1",
	})

	if !strings.Contains(requestURI, "/projects/my-proj/locations/us-central1/publishers/google/models/") {
		t.Fatalf("request path = %q", requestURI)
	}
	if headers.Get("Authorization") != "Bearer ya29.token" {
		t.Fatalf("Authorization = %q", headers.Get("Authorization"))
	}
	if headers.Get("x-goog-api-key") != "" {
		t.Fatalf("x-goog-api-key should be unset, got %q", headers.Get("x-goog-api-key"))
	}
}

func TestVertexMissingCredentialsErrors(t *testing.T) {
	ClearGoogleTokenCache()
	t.Cleanup(ClearGoogleTokenCache)

	model := testVertexModel("https://proxy.example.com/v1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	// Empty env values shadow any ambient credentials on the host machine.
	events := drainEvents(StreamGoogleVertex(model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{Env: map[string]string{
			"GOOGLE_OAUTH_ACCESS_TOKEN":      "",
			"GOOGLE_APPLICATION_CREDENTIALS": "",
		}},
		Project:  "p",
		Location: "us-central1",
	}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	// Either no credentials were found, or an ambient file failed to load.
	if final.Error.ErrorMessage == "" {
		t.Fatal("expected a credential error message")
	}
}

// TestVertexMissingProjectErrors covers the non-express path without a project.
func TestVertexMissingProjectErrors(t *testing.T) {
	model := testVertexModel("https://proxy.example.com/v1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleVertex(model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{Env: map[string]string{
			"GOOGLE_CLOUD_PROJECT": "",
			"GCLOUD_PROJECT":       "",
		}},
		Location: "us-central1",
	}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "requires a project ID") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestVertexStreamsSharedGoogleProtocol(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			googleSSE(`{"responseId":"resp_1","candidates":[{"content":{"parts":[{"text":"think","thought":true,"thoughtSignature":"c2ln"}]}}]}`)+
				googleSSE(`{"candidates":[{"content":{"parts":[{"text":"answer"}]}}]}`)+
				googleSSE(`{"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":3,"totalTokenCount":18}}`))
	}))
	defer srv.Close()

	model := testVertexModel(srv.URL + "/v1")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamGoogleVertex(model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{APIKey: "AIzaSy-real"},
	}))

	want := []EventType{EventStart, EventThinkingStart, EventThinkingDelta, EventThinkingEnd, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	final := events[len(events)-1]
	if final.Message.API != APIGoogleVertex {
		t.Fatalf("api = %q", final.Message.API)
	}
	if final.Message.ResponseID != "resp_1" {
		t.Fatalf("responseId = %q", final.Message.ResponseID)
	}
	thinking := final.Message.Content[0].(ThinkingContent)
	if thinking.Thinking != "think" || thinking.ThinkingSignature != "c2ln" {
		t.Fatalf("thinking = %#v", thinking)
	}
	// Thoughts are billed as output, like the Generative Language API.
	if final.Message.Usage.Output != 8 {
		t.Fatalf("output tokens = %d, want 8", final.Message.Usage.Output)
	}
}

func TestVertexBuildParams(t *testing.T) {
	model := testVertexModel("")
	conv := &Context{
		SystemPrompt: "be brief",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools:        []Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	captured, _, _ := captureVertexRequest(t, model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{APIKey: "AIzaSy-real", MaxTokens: 1024},
		ToolChoice:    "any",
	})

	generationConfig := captured["generationConfig"].(map[string]any)
	if generationConfig["maxOutputTokens"] != 1024.0 {
		t.Fatalf("generationConfig = %#v", generationConfig)
	}
	systemInstruction := captured["systemInstruction"].(map[string]any)
	if systemInstruction["parts"].([]any)[0].(map[string]any)["text"] != "be brief" {
		t.Fatalf("systemInstruction = %#v", systemInstruction)
	}
	functionCallingConfig := captured["toolConfig"].(map[string]any)["functionCallingConfig"].(map[string]any)
	if functionCallingConfig["mode"] != googleFunctionCallingAny {
		t.Fatalf("mode = %#v", functionCallingConfig["mode"])
	}
	if _, hasTools := captured["tools"]; !hasTools {
		t.Fatalf("tools should be present: %#v", captured)
	}
}

func TestVertexThinkingLevelMapping(t *testing.T) {
	tests := []struct {
		modelID string
		effort  ThinkingLevel
		want    GoogleThinkingLevel
	}{
		// Gemini 3 Pro exposes only LOW and HIGH.
		{"gemini-3-pro", ThinkingMinimal, GoogleThinkingLevelLow},
		{"gemini-3-pro", ThinkingHigh, GoogleThinkingLevelHigh},
		// Other models map one-to-one. Vertex has no Gemma branch, so a Gemma
		// id falls through to the pass-through mapping.
		{"gemini-3-flash", ThinkingMinimal, GoogleThinkingLevelMinimal},
		{"gemini-3-flash", ThinkingMedium, GoogleThinkingLevelMedium},
		{"gemma-4-27b", ThinkingLow, GoogleThinkingLevelLow},
	}
	for _, tt := range tests {
		model := testVertexModel("")
		model.ID = tt.modelID
		if got := getVertexThinkingLevel(tt.effort, model); got != tt.want {
			t.Errorf("getVertexThinkingLevel(%q, %q) = %q, want %q", tt.effort, tt.modelID, got, tt.want)
		}
	}
}

func TestVertexBudgets(t *testing.T) {
	tests := []struct {
		modelID string
		effort  ThinkingLevel
		want    int
	}{
		{"gemini-2.5-pro", ThinkingMinimal, 128},
		{"gemini-2.5-pro", ThinkingHigh, 32768},
		// Vertex collapses flash-lite into the flash table, so minimal is 128
		// rather than the Generative Language API's 512.
		{"gemini-2.5-flash-lite", ThinkingMinimal, 128},
		{"gemini-2.5-flash", ThinkingHigh, 24576},
		{"gemini-9-experimental", ThinkingHigh, -1},
	}
	for _, tt := range tests {
		model := testVertexModel("")
		model.ID = tt.modelID
		if got := getVertexBudget(model, tt.effort, nil); got != tt.want {
			t.Errorf("getVertexBudget(%q, %q) = %d, want %d", tt.modelID, tt.effort, got, tt.want)
		}
	}

	model := testVertexModel("")
	model.ID = "gemini-2.5-pro"
	if got := getVertexBudget(model, ThinkingMedium, &ThinkingBudgets{Medium: 777}); got != 777 {
		t.Fatalf("custom budget = %d, want 777", got)
	}
}

func TestVertexStreamSimpleThinking(t *testing.T) {
	runSimple := func(t *testing.T, modelID string, options *SimpleStreamOptions) map[string]any {
		t.Helper()
		var captured map[string]any
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewDecoder(r.Body).Decode(&captured)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, googleSSE(googleFinish))
		}))
		defer srv.Close()

		model := testVertexModel(srv.URL + "/v1")
		model.ID = modelID
		model.Reasoning = true
		conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
		options.APIKey = "AIzaSy-real"
		drainEvents(StreamGoogleVertexSimple(model, conv, options))
		return captured
	}

	t.Run("no reasoning disables thinking", func(t *testing.T) {
		captured := runSimple(t, "gemini-2.5-flash", &SimpleStreamOptions{})
		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["thinkingBudget"] != 0.0 {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})

	t.Run("gemini 3 uses a level", func(t *testing.T) {
		captured := runSimple(t, "gemini-3-pro", &SimpleStreamOptions{Reasoning: ThinkingHigh})
		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["includeThoughts"] != true || thinkingConfig["thinkingLevel"] != GoogleThinkingLevelHigh {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})

	t.Run("2.5 uses a budget", func(t *testing.T) {
		captured := runSimple(t, "gemini-2.5-pro", &SimpleStreamOptions{Reasoning: ThinkingLow})
		thinkingConfig := captured["generationConfig"].(map[string]any)["thinkingConfig"].(map[string]any)
		if thinkingConfig["includeThoughts"] != true || thinkingConfig["thinkingBudget"] != 2048.0 {
			t.Fatalf("thinkingConfig = %#v", thinkingConfig)
		}
	})
}

func TestVertexRegisteredInRegistry(t *testing.T) {
	funcs, ok := GetStreamer(APIGoogleVertex)
	if !ok {
		t.Fatal("google-vertex is not registered")
	}
	if funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("registration = %#v", funcs)
	}
}

// TestVertexToolCallIDsForClaudeModels covers Anthropic models served through
// Vertex, which require explicit tool call ids. Ids are only rewritten on
// cross-model replay, since TransformMessages skips normalization when the
// transcript came from the model being called.
func TestVertexToolCallIDsForClaudeModels(t *testing.T) {
	model := testVertexModel("")
	model.ID = "claude-sonnet-4-5"
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				// A foreign source, so normalization runs.
				Role: "assistant", API: APIAnthropicMessages, Provider: "anthropic", Model: "claude-sonnet-4-5",
				StopReason: StopToolUse,
				Content:    []ContentBlock{ToolCall{Type: "toolCall", ID: "call|1", Name: "f", Arguments: map[string]any{}}},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call|1", ToolName: "f", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		},
	}
	captured, _, _ := captureVertexRequest(t, model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{APIKey: "AIzaSy-real"},
	})

	contents := captured["contents"].([]any)
	call := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	// "|" is not in Google's allowed id charset, so it is normalized.
	if call["id"] != "call_1" {
		t.Fatalf("functionCall id = %#v, want the normalized form", call["id"])
	}
	response := contents[2].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionResponse"].(map[string]any)
	if response["id"] != "call_1" {
		t.Fatalf("functionResponse id = %#v", response["id"])
	}
}

// TestVertexSameModelToolCallIDsPassThrough documents the other side of that
// rule: a same-model transcript replays its ids untouched.
func TestVertexSameModelToolCallIDsPassThrough(t *testing.T) {
	model := testVertexModel("")
	model.ID = "claude-sonnet-4-5"
	conv := &Context{
		Messages: []Message{
			UserMessage{Role: "user", Content: "q"},
			AssistantMessage{
				Role: "assistant", API: model.API, Provider: model.Provider, Model: model.ID,
				StopReason: StopToolUse,
				Content:    []ContentBlock{ToolCall{Type: "toolCall", ID: "call|1", Name: "f", Arguments: map[string]any{}}},
			},
			ToolResultMessage{Role: "toolResult", ToolCallID: "call|1", ToolName: "f", Content: []ContentBlock{TextContent{Type: "text", Text: "ok"}}},
		},
	}
	captured, _, _ := captureVertexRequest(t, model, conv, &GoogleVertexOptions{
		StreamOptions: StreamOptions{APIKey: "AIzaSy-real"},
	})

	contents := captured["contents"].([]any)
	call := contents[1].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	if call["id"] != "call|1" {
		t.Fatalf("functionCall id = %#v, want the original", call["id"])
	}
}
