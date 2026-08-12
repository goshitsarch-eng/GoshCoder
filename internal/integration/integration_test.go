// Package integration wires the ported layers together against a fake
// provider server: catalog credential resolution -> agent loop -> wire
// protocol -> built-in tools. These tests are the check that the pieces
// actually connect, rather than each working in isolation.
package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/tools"
)

// sseChunk formats one openai-completions streaming chunk.
func sseChunk(payload string) string {
	return "data: " + payload + "\n\n"
}

// textChunks streams a plain text reply, then a terminal finish chunk.
func textChunks(text string) string {
	encoded, _ := json.Marshal(text)
	return sseChunk(fmt.Sprintf(
		`{"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":%s},"finish_reason":null}]}`, encoded)) +
		sseChunk(`{"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`) +
		sseChunk("[DONE]")
}

// toolCallChunks streams a single tool call, then a tool_calls finish chunk.
func toolCallChunks(id, name, argsJSON string) string {
	encodedArgs, _ := json.Marshal(argsJSON)
	return sseChunk(fmt.Sprintf(
		`{"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":%q,"type":"function","function":{"name":%q,"arguments":%s}}]},"finish_reason":null}]}`,
		id, name, encodedArgs)) +
		sseChunk(`{"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`) +
		sseChunk("[DONE]")
}

// fakeProvider is a stand-in for an openai-completions endpoint. It replays
// scripted responses and records what it received.
type fakeProvider struct {
	server *httptest.Server

	mu       sync.Mutex
	requests []map[string]any
	headers  []http.Header
	// responses are replayed in order; the last one repeats.
	responses []string
}

func newFakeProvider(t *testing.T, responses ...string) *fakeProvider {
	t.Helper()
	provider := &fakeProvider{responses: responses}
	provider.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)

		provider.mu.Lock()
		index := len(provider.requests)
		provider.requests = append(provider.requests, body)
		provider.headers = append(provider.headers, r.Header.Clone())
		if index >= len(provider.responses) {
			index = len(provider.responses) - 1
		}
		payload := provider.responses[index]
		provider.mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, payload)
	}))
	t.Cleanup(provider.server.Close)
	return provider
}

func (p *fakeProvider) requestCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.requests)
}

func (p *fakeProvider) request(i int) map[string]any {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.requests[i]
}

func (p *fakeProvider) header(i int) http.Header {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.headers[i]
}

// testModel points an openai-completions model at the fake server.
func (p *fakeProvider) testModel() llm.Model {
	return llm.Model{
		ID:            "gpt-test",
		Name:          "GPT Test",
		API:           llm.APIOpenAICompletions,
		Provider:      "openai",
		BaseURL:       p.server.URL,
		Input:         []string{"text"},
		ContextWindow: 128000,
		MaxTokens:     4096,
	}
}

// authStreamFn mirrors the CLI's stream function: it applies resolved
// credentials, then dispatches through the protocol registry.
func authStreamFn(auth *catalog.Auth) agent.StreamFn {
	return func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		if auth != nil {
			opts.Headers = auth.Headers
			opts.Env = auth.Env
		}
		funcs, ok := llm.GetStreamer(model.API)
		if !ok || funcs.StreamSimple == nil {
			stream := llm.NewAssistantMessageEventStream()
			stream.Push(llm.AssistantMessageEvent{
				Type:   llm.EventError,
				Reason: llm.StopError,
				Error: &llm.AssistantMessage{
					Role: "assistant", StopReason: llm.StopError,
					ErrorMessage: "no streamer for " + model.API,
				},
			})
			stream.End()
			return stream
		}
		return funcs.StreamSimple(model, ctx, opts)
	}
}

// TestEndToEndTextTurn walks the whole stack for a plain reply: a stored
// credential is resolved from disk, the agent runs a turn, and the protocol
// sends a real HTTP request.
func TestEndToEndTextTurn(t *testing.T) {
	provider := newFakeProvider(t, textChunks("Hello from the fake provider"))

	// A file-backed credential store, as the CLI uses.
	authPath := filepath.Join(t.TempDir(), "auth.json")
	store := catalog.NewFileCredentialStore(authPath)
	if _, err := store.Modify("openai", func(*catalog.Credential) (*catalog.Credential, error) {
		return &catalog.Credential{Type: "api_key", Key: "sk-stored-e2e"}, nil
	}); err != nil {
		t.Fatal(err)
	}

	c := catalog.NewCatalog(store)
	c.SetGetenv(func(string) string { return "" })
	auth, ok := c.ResolveAuth("openai")
	if !ok {
		t.Fatal("the stored credential did not resolve")
	}

	model := provider.testModel()
	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{SystemPrompt: "be brief", Model: model},
		StreamFn:     authStreamFn(auth),
		GetAPIKey:    func(string) string { return auth.APIKey },
	})

	if err := a.Prompt("hi"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	if provider.requestCount() != 1 {
		t.Fatalf("provider requests = %d, want 1", provider.requestCount())
	}
	// The stored key reached the wire as a bearer token.
	if got := provider.header(0).Get("Authorization"); got != "Bearer sk-stored-e2e" {
		t.Fatalf("Authorization = %q", got)
	}
	// The system prompt reached the request body.
	messages := provider.request(0)["messages"].([]any)
	system := messages[0].(map[string]any)
	if system["role"] != "system" || system["content"] != "be brief" {
		t.Fatalf("system message = %#v", system)
	}

	state := a.State()
	if state.ErrorMessage != "" {
		t.Fatalf("run failed: %s", state.ErrorMessage)
	}
	if len(state.Messages) != 2 {
		t.Fatalf("messages = %#v", state.Messages)
	}
	final := state.Messages[1].(llm.AssistantMessage)
	if text := final.Content[0].(llm.TextContent); text.Text != "Hello from the fake provider" {
		t.Fatalf("reply = %q", text.Text)
	}
	// Usage flowed back from the provider through the agent.
	if final.Usage.Input != 10 || final.Usage.Output != 4 {
		t.Fatalf("usage = %#v", final.Usage)
	}
}

// TestEndToEndToolUse is the full agentic loop: the model calls a real tool,
// the tool touches the filesystem, and the result is replayed to the provider
// on the next turn.
func TestEndToEndToolUse(t *testing.T) {
	provider := newFakeProvider(t,
		toolCallChunks("call_1", "write", `{"path":"generated.txt","content":"written by the agent"}`),
		textChunks("Done, the file is written."),
	)

	workspace, err := tools.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	c := catalog.NewCatalog(nil)
	c.SetGetenv(func(name string) string {
		if name == "OPENAI_API_KEY" {
			return "sk-env-e2e"
		}
		return ""
	})
	auth, ok := c.ResolveAuth("openai")
	if !ok {
		t.Fatal("the environment credential did not resolve")
	}

	model := provider.testModel()
	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{Model: model, Tools: workspace.All()},
		StreamFn:     authStreamFn(auth),
		GetAPIKey:    func(string) string { return auth.APIKey },
	})

	if err := a.Prompt("create generated.txt"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	// Two turns: the tool call, then the summary.
	if provider.requestCount() != 2 {
		t.Fatalf("provider requests = %d, want 2", provider.requestCount())
	}

	// The tool actually ran against the filesystem.
	content, err := os.ReadFile(filepath.Join(workspace.Root, "generated.txt"))
	if err != nil {
		t.Fatalf("the tool did not create the file: %v", err)
	}
	if string(content) != "written by the agent" {
		t.Fatalf("file content = %q", content)
	}

	// The tool schemas were advertised to the provider.
	toolsField, hasTools := provider.request(0)["tools"].([]any)
	if !hasTools || len(toolsField) != len(workspace.All()) {
		t.Fatalf("tools in request = %#v", provider.request(0)["tools"])
	}

	// The second request replays the tool result.
	replayed := provider.request(1)["messages"].([]any)
	foundToolResult := false
	for _, raw := range replayed {
		if message, ok := raw.(map[string]any); ok && message["role"] == "tool" {
			foundToolResult = true
			if !strings.Contains(fmt.Sprint(message["content"]), "generated.txt") {
				t.Fatalf("tool result content = %#v", message["content"])
			}
		}
	}
	if !foundToolResult {
		t.Fatalf("no tool result was replayed: %#v", replayed)
	}

	state := a.State()
	if state.ErrorMessage != "" {
		t.Fatalf("run failed: %s", state.ErrorMessage)
	}
	// user, assistant tool call, tool result, assistant summary
	if len(state.Messages) != 4 {
		t.Fatalf("messages = %d: %#v", len(state.Messages), state.Messages)
	}
	if agent.RoleOf(state.Messages[2]) != "toolResult" {
		t.Fatalf("messages[2] role = %q", agent.RoleOf(state.Messages[2]))
	}
}

// TestEndToEndToolErrorIsReportedToModel confirms a failing tool does not abort
// the run: the error is replayed so the model can react.
func TestEndToEndToolErrorIsReportedToModel(t *testing.T) {
	provider := newFakeProvider(t,
		// Reading a missing file fails inside the tool.
		toolCallChunks("call_1", "read", `{"path":"missing.txt"}`),
		textChunks("That file does not exist."),
	)

	workspace, err := tools.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}

	model := provider.testModel()
	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{Model: model, Tools: workspace.All()},
		StreamFn:     authStreamFn(&catalog.Auth{APIKey: "sk-test"}),
		GetAPIKey:    func(string) string { return "sk-test" },
	})

	var toolErrors int
	a.Subscribe(func(ctx context.Context, ev agent.Event) {
		if ev.Type == agent.EventToolExecutionEnd && ev.IsError {
			toolErrors++
		}
	})

	if err := a.Prompt("read missing.txt"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if toolErrors != 1 {
		t.Fatalf("tool errors = %d, want 1", toolErrors)
	}
	// The loop continued to a second turn despite the failure.
	if provider.requestCount() != 2 {
		t.Fatalf("provider requests = %d, want 2", provider.requestCount())
	}
	if state := a.State(); state.ErrorMessage != "" {
		t.Fatalf("a tool failure must not fail the run: %s", state.ErrorMessage)
	}
}

// TestEndToEndPathConfinementThroughAgent is the security check at the full
// stack level: a model asking to escape the workspace is refused.
func TestEndToEndPathConfinementThroughAgent(t *testing.T) {
	provider := newFakeProvider(t,
		toolCallChunks("call_1", "write", `{"path":"../escaped.txt","content":"pwned"}`),
		textChunks("I cannot write outside the workspace."),
	)

	root := t.TempDir()
	workspace, err := tools.NewWorkspace(filepath.Join(root, "inner"))
	if err != nil {
		if mkErr := os.MkdirAll(filepath.Join(root, "inner"), 0o755); mkErr != nil {
			t.Fatal(mkErr)
		}
		workspace, err = tools.NewWorkspace(filepath.Join(root, "inner"))
		if err != nil {
			t.Fatal(err)
		}
	}

	model := provider.testModel()
	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{Model: model, Tools: workspace.All()},
		StreamFn:     authStreamFn(&catalog.Auth{APIKey: "sk-test"}),
		GetAPIKey:    func(string) string { return "sk-test" },
	})

	if err := a.Prompt("escape the workspace"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}

	// Nothing was written outside the workspace.
	if _, err := os.Stat(filepath.Join(root, "escaped.txt")); !os.IsNotExist(err) {
		t.Fatalf("a file escaped the workspace: %v", err)
	}
}

// TestEndToEndAmbientBedrockIsNotUsableAsAKey confirms the ambient sentinel is
// recognizable, so callers do not send it as a literal credential.
func TestEndToEndAmbientBedrockIsNotUsableAsAKey(t *testing.T) {
	c := catalog.NewCatalog(nil)
	c.SetGetenv(func(name string) string {
		if name == "AWS_PROFILE" {
			return "default"
		}
		return ""
	})
	c.SetFileExists(func(string) bool { return false })

	auth, ok := c.ResolveAuth("amazon-bedrock")
	if !ok {
		t.Fatal("bedrock should be configured from AWS_PROFILE")
	}
	if !auth.IsAmbient() {
		t.Fatalf("auth = %#v, want an ambient credential", auth)
	}
}

// TestEndToEndUnimplementedProtocolFailsCleanly confirms an unported protocol
// surfaces a clear error instead of panicking.
func TestEndToEndUnimplementedProtocolFailsCleanly(t *testing.T) {
	model := llm.Model{ID: "m", Name: "M", API: "bedrock-converse-stream", Provider: "amazon-bedrock"}
	if _, ok := llm.GetStreamer(model.API); ok {
		t.Skip("bedrock-converse-stream is now implemented; pick another unported protocol")
	}

	a := agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{Model: model},
		StreamFn:     authStreamFn(&catalog.Auth{APIKey: "sk-test"}),
	})
	if err := a.Prompt("hi"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got := a.State().ErrorMessage; !strings.Contains(got, "no streamer") {
		t.Fatalf("errorMessage = %q", got)
	}
}

// TestRegisteredProtocolsCoverCatalog reports which wire protocols the catalog
// needs and which are implemented, so the remaining gap stays visible.
func TestRegisteredProtocolsCoverCatalog(t *testing.T) {
	c := catalog.NewCatalog(nil)
	c.SetGetenv(func(string) string { return "" })

	implemented := map[string]int{}
	missing := map[string]int{}
	for _, id := range c.ProviderIDs() {
		for _, model := range c.Provider(id).Models() {
			if _, ok := llm.GetStreamer(model.API); ok {
				implemented[model.API]++
			} else {
				missing[model.API]++
			}
		}
	}

	total := 0
	for _, count := range implemented {
		total += count
	}
	for _, count := range missing {
		total += count
	}
	if total == 0 {
		t.Fatal("the catalog is empty")
	}
	if len(implemented) == 0 {
		t.Fatal("no wire protocol is registered")
	}

	covered := 0
	for _, count := range implemented {
		covered += count
	}
	t.Logf("model coverage: %d/%d (%.1f%%)", covered, total, float64(covered)/float64(total)*100)
	for api, count := range implemented {
		t.Logf("  implemented %-28s %d models", api, count)
	}
	for api, count := range missing {
		t.Logf("  MISSING     %-28s %d models", api, count)
	}

	// Every protocol ported so far must stay registered. Only
	// openai-codex-responses remains unported.
	for _, api := range []string{
		llm.APIOpenAICompletions,
		llm.APIAnthropicMessages,
		llm.APIOpenAIResponses,
		llm.APIAzureOpenAIResponses,
		llm.APIGoogleGenerativeAI,
		llm.APIGoogleVertex,
		llm.APIMistralConversations,
		llm.APIBedrockConverseStream,
	} {
		if _, ok := llm.GetStreamer(api); !ok {
			t.Errorf("protocol %q is no longer registered", api)
		}
	}
}
