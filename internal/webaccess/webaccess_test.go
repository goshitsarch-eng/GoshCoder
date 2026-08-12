package webaccess

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

func TestExaMCPProvidesZeroConfigSearch(t *testing.T) {
	var seenQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tools") != "web_search_exa" || r.Header.Get("x-exa-source") != "pi-web-access" {
			t.Errorf("request = %s, headers %#v", r.URL.String(), r.Header)
		}
		var request struct {
			Params struct {
				Arguments struct {
					Query string `json:"query"`
				} `json:"arguments"`
			} `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		seenQuery = request.Params.Arguments.Query
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `event: message`)
		fmt.Fprintln(w, `data: {"result":{"content":[{"type":"text","text":"Title: Go\nURL: https://go.dev/\nHighlights:\nThe Go programming language\n---\n\nTitle: Docs\nURL: https://go.dev/doc/\nText: Official documentation"}]},"jsonrpc":"2.0","id":1}`)
	}))
	defer server.Close()

	service := New("", nil)
	service.Client = server.Client()
	service.exaMCPURL = server.URL
	response, err := service.Search(t.Context(), "Go language", Options{
		Provider: "exa", NumResults: 2, DomainFilter: []string{"go.dev", "-old.go.dev"}, RecencyFilter: "year",
	})
	if err != nil {
		t.Fatal(err)
	}
	if response.Provider != "exa" || len(response.Results) != 2 || response.Results[0].URL != "https://go.dev/" {
		t.Fatalf("response = %#v", response)
	}
	for _, expected := range []string{"site:go.dev", "-site:old.go.dev", "past year"} {
		if !strings.Contains(seenQuery, expected) {
			t.Fatalf("query %q does not contain %q", seenQuery, expected)
		}
	}
}

func TestKagiSearchUsesConfigCredentialAndNormalizesResults(t *testing.T) {
	var authorization string
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data": []any{
				map[string]any{"title": "Kagi result", "url": "https://example.com/one", "snippet": "First result"},
				map[string]any{"name": "Second", "href": "https://example.org/two", "description": "Second result"},
			},
		})
	}))
	defer server.Close()

	configPath := filepath.Join(t.TempDir(), "web-search.json")
	if err := os.WriteFile(configPath, []byte(`{"provider":"kagi","kagiApiKey":"kagi-secret"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	service := New(configPath, nil)
	service.Client = server.Client()
	service.kagiSearchURL = server.URL
	response, err := service.Search(t.Context(), "current Go release", Options{NumResults: 2})
	if err != nil {
		t.Fatal(err)
	}
	if authorization != "Bearer kagi-secret" {
		t.Fatalf("Authorization = %q", authorization)
	}
	if request["query"] != "current Go release" || request["limit"] != float64(2) {
		t.Fatalf("request = %#v", request)
	}
	if response.Provider != "kagi" || len(response.Results) != 2 || response.Results[1].Title != "Second" {
		t.Fatalf("response = %#v", response)
	}
}

func TestOpenAICodexSearchReusesOAuthHeadersAndParsesSSE(t *testing.T) {
	jwt := testCodexJWT(t, "acct-search")
	var headers http.Header
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		headers = r.Header.Clone()
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"type":"response.output_item.done","item":{"type":"web_search_call","action":{"sources":[{"url":"https://go.dev/","title":"Go"}]}}}`)
		fmt.Fprintln(w, `data: {"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"Go is a programming language [Go](https://go.dev/).","annotations":[{"type":"url_citation","url":"https://go.dev/?utm_source=openai","title":"The Go Programming Language"}]}]}}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer server.Close()

	service := New("", func(context.Context) (*OpenAIAuth, error) {
		return &OpenAIAuth{Provider: "openai-codex", APIKey: jwt, Model: "gpt-5.6-terra"}, nil
	})
	service.Client = server.Client()
	service.codexURL = server.URL
	response, err := service.Search(t.Context(), "what is Go", Options{Provider: "openai", NumResults: 5})
	if err != nil {
		t.Fatal(err)
	}
	if headers.Get("Chatgpt-Account-Id") != "acct-search" || headers.Get("Originator") != "goshcoder" {
		t.Fatalf("headers = %#v", headers)
	}
	if request["tool_choice"] != "required" || response.Answer == "" || len(response.Results) != 1 {
		t.Fatalf("request/response = %#v / %#v", request, response)
	}
	if response.Results[0].URL != "https://go.dev/" {
		t.Fatalf("clean citation URL = %q", response.Results[0].URL)
	}
}

func TestAutoFallsBackFromOpenAIToExa(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/openai" {
			http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintln(w, `data: {"result":{"content":[{"type":"text","text":"Title: Fallback\nURL: https://example.com/\nHighlights:\nFallback worked"}]},"jsonrpc":"2.0","id":1}`)
	}))
	defer server.Close()
	service := New("", func(context.Context) (*OpenAIAuth, error) {
		return &OpenAIAuth{Provider: "openai", APIKey: "openai-secret"}, nil
	})
	service.Client = server.Client()
	service.openAIURL = server.URL + "/openai"
	service.exaMCPURL = server.URL + "/exa"
	response, err := service.Search(t.Context(), "fallback", Options{NumResults: 5})
	if err != nil {
		t.Fatal(err)
	}
	if response.Provider != "exa" || len(response.Results) != 1 {
		t.Fatalf("response = %#v", response)
	}
}

func TestWebSearchToolAcceptsMultipleQueriesAndReportsProgress(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Params struct {
				Arguments struct {
					Query string `json:"query"`
				} `json:"arguments"`
			} `json:"params"`
		}
		_ = json.NewDecoder(r.Body).Decode(&request)
		text := fmt.Sprintf("Title: %s\nURL: https://example.com/%s\nHighlights:\nanswer", request.Params.Arguments.Query, request.Params.Arguments.Query)
		_ = json.NewEncoder(w).Encode(map[string]any{"result": map[string]any{"content": []any{map[string]any{"type": "text", "text": text}}}})
	}))
	defer server.Close()
	service := New("", nil)
	service.Client = server.Client()
	service.exaMCPURL = server.URL
	tool := service.Tool()
	updates := 0
	result, err := tool.Execute(t.Context(), "call-1", map[string]any{
		"queries": []any{"alpha", "beta"}, "provider": "exa", "numResults": float64(1),
	}, func(agent.ToolResult) { updates++ })
	if err != nil {
		t.Fatal(err)
	}
	if updates != 2 {
		t.Fatalf("updates = %d", updates)
	}
	text := toolResultText(result)
	if !strings.Contains(text, "## alpha") || !strings.Contains(text, "## beta") {
		t.Fatalf("tool output = %q", text)
	}
	if details, ok := result.Details.([]Response); !ok || len(details) != 2 {
		t.Fatalf("details = %#v", result.Details)
	}
}

func testCodexJWT(t *testing.T, accountID string) string {
	t.Helper()
	payload, err := json.Marshal(map[string]any{
		"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": accountID},
	})
	if err != nil {
		t.Fatal(err)
	}
	return "header." + base64.RawURLEncoding.EncodeToString(payload) + ".signature"
}

func toolResultText(result agent.ToolResult) string {
	var parts []string
	for _, block := range result.Content {
		if text, ok := block.(llm.TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n")
}
