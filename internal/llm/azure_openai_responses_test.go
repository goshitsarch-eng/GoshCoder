package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func testAzureModel(baseURL string) *Model {
	return &Model{
		ID:            "gpt-4o-mini",
		Name:          "GPT-4o mini",
		API:           APIAzureOpenAIResponses,
		Provider:      "azure-openai-responses",
		BaseURL:       baseURL,
		Input:         []string{"text"},
		ContextWindow: 128000,
		MaxTokens:     16384,
	}
}

// captureAzureRequest runs a stream against a recording server and returns the
// decoded request body, headers, and request URL.
func captureAzureRequest(t *testing.T, model *Model, conv *Context, opts *AzureOpenAIResponsesOptions) (map[string]any, http.Header, string) {
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
		fmt.Fprint(w, responsesSSE("response.completed", responsesCompleted))
	}))
	defer srv.Close()

	// A localhost test server is not an Azure host, so the base URL passes
	// through normalization untouched.
	if opts.AzureBaseURL == "" {
		opts.AzureBaseURL = srv.URL
	} else {
		model.BaseURL = srv.URL
	}
	drainEvents(StreamAzureOpenAIResponses(model, conv, opts))
	return captured, headers, requestURI
}

func TestAzureNormalizeBaseURL(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"cognitive services root", "https://res.cognitiveservices.azure.com", "https://res.cognitiveservices.azure.com/openai/v1"},
		{"foundry root", "https://res.ai.azure.com", "https://res.ai.azure.com/openai/v1"},
		{"azure openai root", "https://res.openai.azure.com", "https://res.openai.azure.com/openai/v1"},
		{"/openai becomes /openai/v1", "https://res.cognitiveservices.azure.com/openai", "https://res.cognitiveservices.azure.com/openai/v1"},
		{"/openai/v1 preserved", "https://res.cognitiveservices.azure.com/openai/v1", "https://res.cognitiveservices.azure.com/openai/v1"},
		{"/openai/v1/responses trimmed", "https://res.services.ai.azure.com/openai/v1/responses", "https://res.services.ai.azure.com/openai/v1"},
		{"non-azure proxy path preserved", "https://proxy.example.com/v1", "https://proxy.example.com/v1"},
		{"query stripped on azure host", "https://res.openai.azure.com/openai?api-version=2024-12-01", "https://res.openai.azure.com/openai/v1"},
		{"query preserved on proxy", "https://proxy.example.com/v1?custom=true", "https://proxy.example.com/v1?custom=true"},
		{"trailing slashes trimmed", "https://proxy.example.com/v1///", "https://proxy.example.com/v1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := normalizeAzureBaseURL(tt.in)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Fatalf("normalizeAzureBaseURL(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestAzureInvalidBaseURLErrors(t *testing.T) {
	if _, err := normalizeAzureBaseURL("not-a-url"); err == nil {
		t.Fatal("expected an error for a non-absolute URL")
	}

	model := testAzureModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamAzureOpenAIResponses(model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		AzureBaseURL:  "not-a-url",
	}))

	final := events[len(events)-1]
	if final.Type != EventError || final.Reason != StopError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "invalid Azure OpenAI base URL") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestAzureResolveConfig(t *testing.T) {
	model := testAzureModel("")

	t.Run("resource name builds the default endpoint", func(t *testing.T) {
		baseURL, apiVersion, err := resolveAzureConfig(model, &AzureOpenAIResponsesOptions{
			AzureResourceName: "my-resource",
		})
		if err != nil {
			t.Fatal(err)
		}
		if baseURL != "https://my-resource.openai.azure.com/openai/v1" {
			t.Fatalf("baseURL = %q", baseURL)
		}
		if apiVersion != defaultAzureAPIVersion {
			t.Fatalf("apiVersion = %q", apiVersion)
		}
	})

	t.Run("env supplies base URL and api version", func(t *testing.T) {
		baseURL, apiVersion, err := resolveAzureConfig(model, &AzureOpenAIResponsesOptions{
			StreamOptions: StreamOptions{Env: map[string]string{
				"AZURE_OPENAI_BASE_URL":    "https://from-env.openai.azure.com",
				"AZURE_OPENAI_API_VERSION": "2025-01-01",
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		if baseURL != "https://from-env.openai.azure.com/openai/v1" {
			t.Fatalf("baseURL = %q", baseURL)
		}
		if apiVersion != "2025-01-01" {
			t.Fatalf("apiVersion = %q", apiVersion)
		}
	})

	t.Run("explicit options beat env", func(t *testing.T) {
		baseURL, _, err := resolveAzureConfig(model, &AzureOpenAIResponsesOptions{
			StreamOptions: StreamOptions{Env: map[string]string{"AZURE_OPENAI_BASE_URL": "https://from-env.openai.azure.com"}},
			AzureBaseURL:  "https://explicit.openai.azure.com",
		})
		if err != nil {
			t.Fatal(err)
		}
		if baseURL != "https://explicit.openai.azure.com/openai/v1" {
			t.Fatalf("baseURL = %q", baseURL)
		}
	})

	t.Run("model base URL is the last resort", func(t *testing.T) {
		baseURL, _, err := resolveAzureConfig(testAzureModel("https://from-model.openai.azure.com"), &AzureOpenAIResponsesOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if baseURL != "https://from-model.openai.azure.com/openai/v1" {
			t.Fatalf("baseURL = %q", baseURL)
		}
	})

	t.Run("no endpoint at all is an error", func(t *testing.T) {
		_, _, err := resolveAzureConfig(testAzureModel(""), &AzureOpenAIResponsesOptions{})
		if err == nil || !strings.Contains(err.Error(), "base URL is required") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestAzureDeploymentNameResolution(t *testing.T) {
	model := testAzureModel("")

	t.Run("defaults to the model id", func(t *testing.T) {
		if got := resolveAzureDeploymentName(model, &AzureOpenAIResponsesOptions{}); got != "gpt-4o-mini" {
			t.Fatalf("deployment = %q", got)
		}
	})

	t.Run("explicit option wins", func(t *testing.T) {
		got := resolveAzureDeploymentName(model, &AzureOpenAIResponsesOptions{AzureDeploymentName: "my-deploy"})
		if got != "my-deploy" {
			t.Fatalf("deployment = %q", got)
		}
	})

	t.Run("env map is consulted", func(t *testing.T) {
		got := resolveAzureDeploymentName(model, &AzureOpenAIResponsesOptions{
			StreamOptions: StreamOptions{Env: map[string]string{
				"AZURE_OPENAI_DEPLOYMENT_NAME_MAP": "other=nope, gpt-4o-mini=mapped-deploy",
			}},
		})
		if got != "mapped-deploy" {
			t.Fatalf("deployment = %q", got)
		}
	})

	t.Run("unmapped model falls back to the id", func(t *testing.T) {
		got := resolveAzureDeploymentName(model, &AzureOpenAIResponsesOptions{
			StreamOptions: StreamOptions{Env: map[string]string{"AZURE_OPENAI_DEPLOYMENT_NAME_MAP": "other=nope"}},
		})
		if got != "gpt-4o-mini" {
			t.Fatalf("deployment = %q", got)
		}
	})
}

func TestAzureParseDeploymentNameMap(t *testing.T) {
	got := parseAzureDeploymentNameMap(" a=1, b=2 ,,malformed, c = 3 ")
	want := map[string]string{"a": "1", "b": "2", "c": "3"}
	if len(got) != len(want) {
		t.Fatalf("map = %#v, want %#v", got, want)
	}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("map[%q] = %q, want %q", k, got[k], v)
		}
	}
	if len(parseAzureDeploymentNameMap("")) != 0 {
		t.Fatal("empty value should yield an empty map")
	}
}

func TestAzureRequestURLAndAuth(t *testing.T) {
	model := testAzureModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	_, headers, requestURI := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions:   StreamOptions{APIKey: "secret"},
		AzureAPIVersion: "2025-04-01",
	})

	if !strings.HasPrefix(requestURI, "/responses?") {
		t.Fatalf("request path = %q", requestURI)
	}
	if !strings.Contains(requestURI, "api-version=2025-04-01") {
		t.Fatalf("request query = %q", requestURI)
	}
	// Azure authenticates with api-key, not a bearer token.
	if headers.Get("api-key") != "secret" {
		t.Fatalf("api-key = %q", headers.Get("api-key"))
	}
	if headers.Get("Authorization") != "" {
		t.Fatalf("Authorization should be unset, got %q", headers.Get("Authorization"))
	}
}

func TestAzureDefaultAPIVersion(t *testing.T) {
	model := testAzureModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	_, _, requestURI := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})
	if !strings.Contains(requestURI, "api-version=v1") {
		t.Fatalf("request query = %q", requestURI)
	}
}

func TestAzureBuildParams(t *testing.T) {
	model := testAzureModel("")
	conv := &Context{
		SystemPrompt: "be brief",
		Messages:     []Message{UserMessage{Role: "user", Content: "hi"}},
	}
	captured, _, _ := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions:       StreamOptions{APIKey: "k", MaxTokens: 1024},
		AzureDeploymentName: "my-deploy",
	})

	// The body's model is the Azure deployment, not the pi model id.
	if captured["model"] != "my-deploy" {
		t.Fatalf("model = %#v, want the deployment name", captured["model"])
	}
	if captured["stream"] != true || captured["store"] != false {
		t.Fatalf("params = %#v", captured)
	}
	if captured["max_output_tokens"] != 1024.0 {
		t.Fatalf("max_output_tokens = %#v", captured["max_output_tokens"])
	}
	input := captured["input"].([]any)
	if system := input[0].(map[string]any); system["role"] != "system" || system["content"] != "be brief" {
		t.Fatalf("system message = %#v", input[0])
	}
}

func TestAzureClampsPromptCacheKey(t *testing.T) {
	model := testAzureModel("")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _, _ := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k", SessionID: strings.Repeat("x", 67)},
	})
	if captured["prompt_cache_key"] != strings.Repeat("x", 64) {
		t.Fatalf("prompt_cache_key = %#v", captured["prompt_cache_key"])
	}
}

func TestAzureStrictModeDefaultsOn(t *testing.T) {
	model := testAzureModel("")
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools:    []Tool{{Name: "t", Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}
	captured, _, _ := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
	})

	tool := captured["tools"].([]any)[0].(map[string]any)
	// Azure defaults supportsStrictMode to true, so the field is present.
	if strict, ok := tool["strict"]; !ok || strict != false {
		t.Fatalf("strict = %#v, want false and present", tool["strict"])
	}
}

// TestAzureHonorsSupportsStrictModeFalse covers a "prefer" constrained tool on
// a model without strict support: the field is omitted rather than erroring.
func TestAzureHonorsSupportsStrictModeFalse(t *testing.T) {
	model := testAzureModel("")
	supportsStrict := false
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:                "preferred",
			Description:         "Preferred constrained tool",
			Parameters:          json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}`),
			ConstrainedSampling: json.RawMessage(`{"type":"json_schema","strict":"prefer"}`),
		}},
	}
	captured, _, _ := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Compat:        &AzureOpenAIResponsesCompat{SupportsStrictMode: &supportsStrict},
	})

	tool := captured["tools"].([]any)[0].(map[string]any)
	if _, hasStrict := tool["strict"]; hasStrict {
		t.Fatalf("strict should be omitted: %#v", tool)
	}
}

// TestAzureRequireStrictWithoutSupportErrors covers the "require" case, which
// must fail rather than silently downgrade.
func TestAzureRequireStrictWithoutSupportErrors(t *testing.T) {
	model := testAzureModel("https://res.openai.azure.com")
	supportsStrict := false
	conv := &Context{
		Messages: []Message{UserMessage{Role: "user", Content: "hi"}},
		Tools: []Tool{{
			Name:                "required",
			Description:         "Required constrained tool",
			Parameters:          json.RawMessage(`{"type":"object"}`),
			ConstrainedSampling: json.RawMessage(`{"type":"json_schema","strict":"require"}`),
		}},
	}
	events := drainEvents(StreamAzureOpenAIResponses(model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		Compat:        &AzureOpenAIResponsesCompat{SupportsStrictMode: &supportsStrict},
	}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "requires JSON-schema constrained sampling") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

func TestAzureReasoningParams(t *testing.T) {
	model := testAzureModel("")
	model.Reasoning = true
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	captured, _, _ := captureAzureRequest(t, model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions:   StreamOptions{APIKey: "k"},
		ReasoningEffort: "high",
	})

	reasoning, ok := captured["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" || reasoning["summary"] != "auto" {
		t.Fatalf("reasoning = %#v", captured["reasoning"])
	}
	include, ok := captured["include"].([]any)
	if !ok || len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %#v", captured["include"])
	}
}

func TestAzureNoAPIKeyErrors(t *testing.T) {
	model := testAzureModel("https://res.openai.azure.com")
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	// Unlike openai-responses, an authorization header is not an alternative.
	auth := "Bearer custom"
	events := drainEvents(StreamAzureOpenAIResponses(model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{Headers: ProviderHeaders{"authorization": &auth}},
	}))

	final := events[len(events)-1]
	if final.Type != EventError {
		t.Fatalf("final = %#v", final)
	}
	if !strings.Contains(final.Error.ErrorMessage, "no API key") {
		t.Fatalf("errorMessage = %q", final.Error.ErrorMessage)
	}
}

// TestAzureStreamsSharedResponsesProtocol confirms the Azure path reuses the
// shared Responses stream assembly, including reasoning replay signatures.
func TestAzureStreamsSharedResponsesProtocol(t *testing.T) {
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}`)+
				responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"think"}]}}`)+
				responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"in_progress","content":[]}}`)+
				responsesSSE("response.output_text.delta", `{"type":"response.output_text.delta","output_index":1,"delta":"hello"}`)+
				responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`)+
				// encrypted_content arrives only here, the Azure quirk the
				// shared layer backfills.
				responsesSSE("response.completed", `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"enc"}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`))
	}))
	defer srv.Close()

	model := testAzureModel(srv.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}
	events := drainEvents(StreamAzureOpenAIResponses(model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions: StreamOptions{APIKey: "k"},
		AzureBaseURL:  srv.URL,
	}))

	want := []EventType{EventStart, EventThinkingStart, EventThinkingEnd, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", got, want)
	}

	final := events[len(events)-1]
	if final.Message.API != APIAzureOpenAIResponses {
		t.Fatalf("api = %q", final.Message.API)
	}
	thinking := final.Message.Content[0].(ThinkingContent)
	var item map[string]any
	if err := json.Unmarshal([]byte(thinking.ThinkingSignature), &item); err != nil {
		t.Fatalf("signature is not JSON: %v", err)
	}
	if item["encrypted_content"] != "enc" {
		t.Fatalf("encrypted_content = %#v, want backfilled", item["encrypted_content"])
	}
}

func TestAzureRegisteredInRegistry(t *testing.T) {
	funcs, ok := GetStreamer(APIAzureOpenAIResponses)
	if !ok {
		t.Fatal("azure-openai-responses is not registered")
	}
	if funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("registration = %#v", funcs)
	}
}
