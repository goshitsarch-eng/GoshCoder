package aperture

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"goshcoder/internal/llm"
)

// mcpStub is a minimal Streamable HTTP MCP endpoint: initialize issues a
// session id, tools/list and tools/call answer, and responses can be
// SSE-framed to exercise the parser.
func mcpStub(t *testing.T, sse bool, callResponse func(name string, args map[string]any) (McpCallResult, bool)) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/mcp" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Method string          `json:"method"`
			ID     *int64          `json:"id"`
			Params json.RawMessage `json:"params"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		respond := func(result any) {
			body, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": request.ID, "result": result})
			if sse {
				fmt.Fprintf(w, "event: message\ndata: %s\n\n", body)
				return
			}
			w.Write(body)
		}
		switch request.Method {
		case "initialize":
			w.Header().Set("Mcp-Session-Id", "session-123")
			respond(map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":    map[string]any{},
				"serverInfo":      map[string]any{"name": "stub", "version": "1"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			if r.Header.Get("Mcp-Session-Id") != "session-123" {
				http.Error(w, "missing session", http.StatusBadRequest)
				return
			}
			respond(map[string]any{"tools": []map[string]any{
				{"name": "github_list_repos", "description": "List repositories",
					"inputSchema": map[string]any{"type": "object", "properties": map[string]any{
						"org": map[string]any{"type": "string", "description": "Organization"},
					}, "required": []string{"org"}}},
				{"name": "github_create_issue", "description": "Create an issue"},
				{"name": "slack_send", "description": "Send a Slack message"},
				{"name": "orphan", "description": "No connector prefix"},
			}})
		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			json.Unmarshal(request.Params, &params)
			if callResponse != nil {
				if result, ok := callResponse(params.Name, params.Arguments); ok {
					respond(result)
					return
				}
			}
			respond(McpCallResult{Content: []McpContentItem{{Type: "text", Text: "ok:" + params.Name}}})
		default:
			http.Error(w, "unknown method "+request.Method, http.StatusBadRequest)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func TestMcpSessionSSEAndPlain(t *testing.T) {
	for _, sse := range []bool{false, true} {
		server := mcpStub(t, sse, nil)
		session, err := NewMcpSession(context.Background(), server.URL)
		if err != nil {
			t.Fatalf("sse=%v: %v", sse, err)
		}
		tools, err := session.ListTools(context.Background())
		if err != nil {
			t.Fatalf("sse=%v: %v", sse, err)
		}
		if len(tools) != 4 || tools[0].Name != "github_list_repos" {
			t.Fatalf("sse=%v tools = %+v", sse, tools)
		}
		result, err := session.CallTool(context.Background(), "slack_send", map[string]any{"text": "hi"})
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Content) != 1 || result.Content[0].Text != "ok:slack_send" {
			t.Fatalf("call result = %+v", result)
		}
	}
}

func TestMcpSessionMissingSessionID(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "result": map[string]any{"protocolVersion": "2024-11-05"}})
	}))
	t.Cleanup(server.Close)
	if _, err := NewMcpSession(context.Background(), server.URL); err == nil || !strings.Contains(err.Error(), "Mcp-Session-Id") {
		t.Fatalf("err = %v", err)
	}
}

func connectorFixtures(t *testing.T) (*McpSession, []McpTool, []ConnectorInfo) {
	t.Helper()
	server := mcpStub(t, false, nil)
	session, err := NewMcpSession(context.Background(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	tools, err := session.ListTools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	connectors := []ConnectorInfo{
		{ID: "github", Provider: "GitHub", Status: "connected", Description: "GitHub connector"},
		{ID: "slack", Provider: "Slack", Status: "connected"},
		{ID: "toolless", Provider: "Empty"},
	}
	return session, tools, connectors
}

func TestBuildConnectorToolsSplit(t *testing.T) {
	session, tools, connectors := connectorFixtures(t)
	resolved := Resolved{
		ConnectorsEnabled: true,
		DiscoveryTools:    true,
		PinnedTools: []PinnedConnectorTool{
			{ConnectorID: "github", ToolName: "github_list_repos"},
			{ConnectorID: "github", ToolName: "github_gone"},
		},
	}
	set := BuildConnectorTools(resolved, connectors, tools, func() *McpSession { return session })
	names := make([]string, 0, len(set.Tools))
	for _, tool := range set.Tools {
		names = append(names, tool.Name)
	}
	want := []string{"github_list_repos", "aperture_connector_list", "aperture_connector_tool_search", "aperture_connector_tool_describe", "aperture_connector_tool_call"}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("tools = %v, want %v", names, want)
	}
	if len(set.MissingPins) != 1 || set.MissingPins[0] != "github_gone" {
		t.Errorf("missing pins = %v", set.MissingPins)
	}

	// Discovery off: only pinned tools register.
	resolved.DiscoveryTools = false
	set = BuildConnectorTools(resolved, connectors, tools, func() *McpSession { return session })
	if len(set.Tools) != 1 || set.Tools[0].Name != "github_list_repos" {
		t.Errorf("discovery off tools = %v", set.Tools)
	}
}

func resultText(t *testing.T, blocks []llm.ContentBlock) string {
	t.Helper()
	var parts []string
	for _, block := range blocks {
		if text, ok := block.(llm.TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func TestConnectorListTool(t *testing.T) {
	session, tools, connectors := connectorFixtures(t)
	set := BuildConnectorTools(Resolved{ConnectorsEnabled: true, DiscoveryTools: true}, connectors, tools, func() *McpSession { return session })
	list := set.Tools[0]
	result, err := list.Execute(context.Background(), "id", map[string]any{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	text := resultText(t, result.Content)
	if !strings.Contains(text, "2 connector(s) available") {
		t.Errorf("connectors exposing no tools must be hidden: %s", text)
	}
	if !strings.Contains(text, "**GitHub** (`github`)") || !strings.Contains(text, "2 tools") {
		t.Errorf("list output: %s", text)
	}
}

func TestConnectorSearchTool(t *testing.T) {
	session, tools, connectors := connectorFixtures(t)
	set := BuildConnectorTools(Resolved{ConnectorsEnabled: true, DiscoveryTools: true}, connectors, tools, func() *McpSession { return session })
	search := set.Tools[1]

	result, err := search.Execute(context.Background(), "id", map[string]any{"query": "*"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	text := resultText(t, result.Content)
	if !strings.Contains(text, "### github (2)") || !strings.Contains(text, "### slack (1)") {
		t.Errorf("grouping: %s", text)
	}
	if !strings.HasSuffix(strings.TrimSpace(text), "- `orphan`: No connector prefix") || !strings.Contains(text, "### other (1)") {
		t.Errorf("unknown prefixes sort last under other: %s", text)
	}

	result, _ = search.Execute(context.Background(), "id", map[string]any{"connector": "github", "query": "issue"}, nil)
	text = resultText(t, result.Content)
	if strings.Contains(text, "###") || !strings.Contains(text, "github_create_issue") {
		t.Errorf("connector filter yields a flat list: %s", text)
	}

	result, _ = search.Execute(context.Background(), "id", map[string]any{"query": "nothing-matches-this"}, nil)
	if !strings.Contains(resultText(t, result.Content), "No tools found") {
		t.Error("empty result message missing")
	}
}

func TestConnectorDescribeTool(t *testing.T) {
	session, tools, connectors := connectorFixtures(t)
	set := BuildConnectorTools(Resolved{ConnectorsEnabled: true, DiscoveryTools: true}, connectors, tools, func() *McpSession { return session })
	describe := set.Tools[2]
	result, err := describe.Execute(context.Background(), "id", map[string]any{"tool": "github_list_repos"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	text := resultText(t, result.Content)
	if !strings.Contains(text, "### github_list_repos") || !strings.Contains(text, "`org` (string, required): Organization") {
		t.Errorf("describe output: %s", text)
	}
	result, _ = describe.Execute(context.Background(), "id", map[string]any{"tool": "nope"}, nil)
	if !strings.Contains(resultText(t, result.Content), "not found") {
		t.Error("unknown tool message missing")
	}
}

func TestConnectorCallTool(t *testing.T) {
	session, tools, connectors := connectorFixtures(t)
	set := BuildConnectorTools(Resolved{ConnectorsEnabled: true, DiscoveryTools: true}, connectors, tools, func() *McpSession { return session })
	call := set.Tools[3]

	result, err := call.Execute(context.Background(), "id", map[string]any{"tool": "github_create_issue", "args": `{"title": "x"}`}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resultText(t, result.Content); got != "ok:github_create_issue" {
		t.Errorf("call result = %q", got)
	}

	result, _ = call.Execute(context.Background(), "id", map[string]any{"tool": "github_create_issue", "args": `[1]`}, nil)
	if !strings.Contains(resultText(t, result.Content), "Invalid args JSON") {
		t.Error("non-object args must be rejected with guidance")
	}
	result, _ = call.Execute(context.Background(), "id", map[string]any{"tool": "unknown_tool"}, nil)
	if !strings.Contains(resultText(t, result.Content), "not found") {
		t.Error("unknown tool message missing")
	}

	// A nil session reports the disabled/unreachable state as an error.
	missing := ConnectorToolCallTool([]McpTool{{Name: "x_y"}}, func() *McpSession { return nil })
	if _, err := missing.Execute(context.Background(), "id", map[string]any{"tool": "x_y"}, nil); err == nil {
		t.Error("nil session must fail the call")
	}
}

func TestConnectorCallTruncationOverflow(t *testing.T) {
	large := strings.Repeat("x", maxConnectorOutput+1024)
	server := mcpStub(t, false, func(name string, _ map[string]any) (McpCallResult, bool) {
		if name == "github_big" {
			return McpCallResult{Content: []McpContentItem{{Type: "text", Text: large}}}, true
		}
		return McpCallResult{}, false
	})
	session, err := NewMcpSession(context.Background(), server.URL)
	if err != nil {
		t.Fatal(err)
	}
	result, err := executeConnectorCall(context.Background(), session, "github_big", nil)
	if err != nil {
		t.Fatal(err)
	}
	text := resultText(t, result.Content)
	if len(text) > maxConnectorOutput+256 {
		t.Errorf("output not truncated: %d bytes", len(text))
	}
	if !strings.Contains(text, "Full output: ") {
		t.Fatalf("overflow path missing: %s", text[len(text)-200:])
	}
	path := strings.TrimSuffix(text[strings.Index(text, "Full output: ")+len("Full output: "):], "]")
	content, err := os.ReadFile(strings.TrimSpace(path))
	if err != nil {
		t.Fatalf("overflow file unreadable: %v", err)
	}
	t.Cleanup(func() { os.Remove(strings.TrimSpace(path)) })
	if len(content) != len(large) {
		t.Errorf("overflow file = %d bytes, want %d", len(content), len(large))
	}
}

func TestStandaloneConnectorToolSchema(t *testing.T) {
	schema := json.RawMessage(`{"type": "object", "properties": {"org": {"type": "string"}}}`)
	tool := StandaloneConnectorTool(McpTool{Name: "github_list_repos", InputSchema: schema}, func() *McpSession { return nil })
	if string(tool.Parameters) != string(schema) {
		t.Errorf("real object schemas pass through: %s", tool.Parameters)
	}
	fallback := StandaloneConnectorTool(McpTool{Name: "weird", InputSchema: json.RawMessage(`"nope"`)}, func() *McpSession { return nil })
	if !strings.Contains(string(fallback.Parameters), `"object"`) {
		t.Errorf("unrecognizable schemas coerce to an empty object: %s", fallback.Parameters)
	}
	if !strings.Contains(tool.Description, "github_list_repos") {
		t.Errorf("empty descriptions get a fallback: %q", tool.Description)
	}
}
