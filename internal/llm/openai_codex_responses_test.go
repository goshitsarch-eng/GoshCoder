package llm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func codexTestJWT(t *testing.T, accountID string) string {
	t.Helper()
	payload, err := json.Marshal(map[string]any{
		codexJWTClaimPath: map[string]any{"chatgpt_account_id": accountID},
	})
	if err != nil {
		t.Fatal(err)
	}
	return "header." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
}

func codexTestModel(baseURL string) *Model {
	return &Model{
		ID: "gpt-5.4", Name: "GPT-5.4", API: APIOpenAICodexResponses,
		Provider: "openai-codex", BaseURL: baseURL, Reasoning: true,
		Input: []string{"text", "image"}, ContextWindow: 272000, MaxTokens: 128000,
	}
}

func TestCodexResponsesStreamAndRequest(t *testing.T) {
	var captured map[string]any
	var headers http.Header
	var path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		headers = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w,
			responsesSSE("response.created", `{"type":"response.created","response":{"id":"resp_codex","status":"in_progress"}}`)+
				responsesSSE("response.output_item.added", `{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","status":"in_progress","content":[]}}`)+
				responsesSSE("response.output_text.delta", `{"type":"response.output_text.delta","output_index":0,"delta":"Hello"}`)+
				responsesSSE("response.output_item.done", `{"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_1","status":"completed","content":[{"type":"output_text","text":"Hello","annotations":[]}]}}`)+
				responsesSSE("response.done", `{"type":"response.done","response":{"id":"resp_codex","status":"completed","end_turn":true,"usage":{"input_tokens":12,"output_tokens":3,"total_tokens":15}}}`))
	}))
	defer server.Close()

	model := codexTestModel(server.URL)
	conv := &Context{SystemPrompt: "Be concise.", Messages: []Message{UserMessage{Role: "user", Content: "Hi"}}}
	events := drainEvents(StreamOpenAICodexResponses(model, conv, &OpenAICodexResponsesOptions{
		StreamOptions:   StreamOptions{APIKey: codexTestJWT(t, "acct-1"), SessionID: "session-1"},
		ReasoningEffort: "high", TextVerbosity: "medium",
	}))

	wantEvents := []EventType{EventStart, EventTextStart, EventTextDelta, EventTextEnd, EventDone}
	if got := eventTypes(events); fmt.Sprint(got) != fmt.Sprint(wantEvents) {
		t.Fatalf("events = %v, want %v", got, wantEvents)
	}
	final := events[len(events)-1].Message
	if final == nil || final.StopReason != StopStop || final.ResponseID != "resp_codex" || final.EndTurn == nil || !*final.EndTurn {
		t.Fatalf("final = %#v", final)
	}
	if path != "/codex/responses" {
		t.Fatalf("path = %q", path)
	}
	if headers.Get("Authorization") != "Bearer "+codexTestJWT(t, "acct-1") || headers.Get("Chatgpt-Account-Id") != "acct-1" {
		t.Fatalf("auth headers = %#v", headers)
	}
	if headers.Get("OpenAI-Beta") != "responses=experimental" || headers.Get("Session-Id") != "session-1" {
		t.Fatalf("Codex headers = %#v", headers)
	}
	if captured["instructions"] != "Be concise." || captured["store"] != false || captured["stream"] != true {
		t.Fatalf("request = %#v", captured)
	}
	if captured["prompt_cache_key"] != "session-1" || captured["tool_choice"] != "auto" {
		t.Fatalf("request caching/tools = %#v", captured)
	}
	text := captured["text"].(map[string]any)
	if text["verbosity"] != "medium" {
		t.Fatalf("text = %#v", text)
	}
	reasoning := captured["reasoning"].(map[string]any)
	if reasoning["effort"] != "high" || reasoning["summary"] != "auto" {
		t.Fatalf("reasoning = %#v", reasoning)
	}
	input := captured["input"].([]any)
	if len(input) != 1 || input[0].(map[string]any)["role"] != "user" {
		t.Fatalf("input = %#v; system prompt must travel as instructions only", input)
	}
}

func TestResolveCodexURL(t *testing.T) {
	tests := map[string]string{
		"":                                      defaultCodexBaseURL + "/codex/responses",
		"https://example.com/backend-api/":      "https://example.com/backend-api/codex/responses",
		"https://example.com/backend-api/codex": "https://example.com/backend-api/codex/responses",
		"https://example.com/codex/responses":   "https://example.com/codex/responses",
	}
	for input, want := range tests {
		if got := resolveCodexURL(input); got != want {
			t.Errorf("resolveCodexURL(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestCodexResponsesRejectsTokenWithoutAccountID(t *testing.T) {
	events := drainEvents(StreamOpenAICodexResponses(codexTestModel("http://127.0.0.1:1"), &Context{},
		&OpenAICodexResponsesOptions{StreamOptions: StreamOptions{APIKey: "not-a-jwt"}}))
	if len(events) != 1 || events[0].Type != EventError || events[0].Error == nil ||
		!strings.Contains(events[0].Error.ErrorMessage, "accountId") {
		t.Fatalf("events = %#v", events)
	}
}

func TestCodexResponsesNestedErrorEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, responsesSSE("error", `{"type":"error","error":{"code":"usage_limit_reached","message":"limit reached"}}`))
	}))
	defer server.Close()
	events := drainEvents(StreamOpenAICodexResponses(codexTestModel(server.URL), &Context{},
		&OpenAICodexResponsesOptions{StreamOptions: StreamOptions{APIKey: codexTestJWT(t, "acct")}}))
	last := events[len(events)-1]
	if last.Type != EventError || last.Error == nil || !strings.Contains(last.Error.ErrorMessage, "limit reached") {
		t.Fatalf("events = %#v", events)
	}
}

func TestCodexResponsesCompatAndToolStrictness(t *testing.T) {
	model := codexTestModel("")
	model.RawCompat = json.RawMessage(`{"supportsOpenAIGrammarTools":true,"supportsToolSearch":true}`)
	compat := getCodexResponsesCompat(model)
	if !compat.SupportsStrictMode || !compat.SupportsOpenAIGrammarTools || !compat.SupportsToolSearch {
		t.Fatalf("compat = %#v", compat)
	}
	conv := &Context{Tools: []Tool{{Name: "lookup", Description: "Look up data", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}}
	params, err := buildCodexResponsesParams(model, conv, &OpenAICodexResponsesOptions{}, compat, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	tools := params["tools"].([]any)
	tool := tools[0].(map[string]any)
	strict, present := tool["strict"]
	if !present || strict != nil {
		t.Fatalf("tool strict = %#v, present = %v; Codex requires explicit null", strict, present)
	}
}

func TestCodexResponsesRegistration(t *testing.T) {
	funcs, ok := GetStreamer(APIOpenAICodexResponses)
	if !ok || funcs.Stream == nil || funcs.StreamSimple == nil {
		t.Fatalf("streamer = %#v, ok = %v", funcs, ok)
	}
}
