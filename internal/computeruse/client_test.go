package computeruse

// The stdio client is exercised against a helper process: the test binary
// re-executes itself and TestHelperMCPServer speaks a canned rmcp-style
// newline-delimited JSON-RPC dialog on stdio.

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

func helperSession(t *testing.T) *Session {
	t.Helper()
	session := &Session{
		binaryPath: os.Args[0],
		argv:       []string{"-test.run=TestHelperMCPServer", "--", "mcp"},
	}
	t.Cleanup(session.Close)
	return session
}

// TestHelperMCPServer is not a real test: it becomes the fake MCP server
// when the test binary is re-executed by helperSession.
func TestHelperMCPServer(t *testing.T) {
	if os.Getenv("GO_COMPUTERUSE_HELPER") != "1" {
		t.Skip("helper process entry point")
	}
	reader := bufio.NewReader(os.Stdin)
	writer := bufio.NewWriter(os.Stdout)
	respond := func(id any, result any) {
		encoded, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
		writer.Write(encoded)
		writer.WriteByte('\n')
		writer.Flush()
	}
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			os.Exit(0)
		}
		var request struct {
			Method string          `json:"method"`
			ID     any             `json:"id"`
			Params json.RawMessage `json:"params"`
		}
		if json.Unmarshal([]byte(line), &request) != nil {
			continue
		}
		switch request.Method {
		case "initialize":
			respond(request.ID, map[string]any{
				"protocolVersion": "2024-11-05",
				"capabilities":    map[string]any{"tools": map[string]any{}},
				"serverInfo":      map[string]any{"name": "fake-cul", "version": "0"},
			})
		case "notifications/initialized":
			// Notification: emit an unrelated server notification to prove
			// the client skips them.
			encoded, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "method": "notifications/progress"})
			writer.Write(encoded)
			writer.WriteByte('\n')
			writer.Flush()
		case "tools/list":
			readOnly := true
			respond(request.ID, map[string]any{"tools": []map[string]any{
				{"name": "doctor", "description": "Readiness report",
					"annotations": map[string]any{"readOnlyHint": readOnly}},
				{"name": "screenshot", "description": "Capture the screen as a bounded PNG or JPEG image"},
				{"name": "click", "description": "Click by element index, selector, or coordinates",
					"annotations": map[string]any{"readOnlyHint": false, "destructiveHint": true}},
			}})
		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			json.Unmarshal(request.Params, &params)
			switch params.Name {
			case "screenshot":
				respond(request.ID, map[string]any{"content": []map[string]any{
					{"type": "text", "text": `{"coordinate_width": 1920, "scale": 1}`},
					{"type": "image", "data": base64.StdEncoding.EncodeToString([]byte("png-bytes")), "mimeType": "image/png"},
				}})
			case "doctor":
				respond(request.ID, map[string]any{"content": []map[string]any{
					{"type": "text", "text": `{"readiness": {"blockers": []}}`},
				}})
			case "click":
				respond(request.ID, map[string]any{
					"isError": true,
					"content": []map[string]any{{"type": "text", "text": "no such element"}},
				})
			default:
				respond(request.ID, map[string]any{"content": []map[string]any{}})
			}
		}
	}
}

func TestSessionToolsAndCall(t *testing.T) {
	t.Setenv("GO_COMPUTERUSE_HELPER", "1")
	session := helperSession(t)
	tools, err := session.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 3 || tools[0].Name != "doctor" {
		t.Fatalf("tools = %+v", tools)
	}
	if tools[0].Annotations == nil || tools[0].Annotations.ReadOnlyHint == nil || !*tools[0].Annotations.ReadOnlyHint {
		t.Error("annotations lost")
	}

	result, err := session.Call(context.Background(), "doctor", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 1 || !strings.Contains(result.Content[0].Text, "readiness") {
		t.Fatalf("doctor result = %+v", result)
	}
}

func TestMCPToolListSearchCall(t *testing.T) {
	t.Setenv("GO_COMPUTERUSE_HELPER", "1")
	session := helperSession(t)
	tool := Tool(session)
	if tool.Name != "mcp" || tool.ExecutionMode != agent.ToolExecutionSequential {
		t.Fatalf("tool identity: %s %s", tool.Name, tool.ExecutionMode)
	}

	// mcp({server: "computer-use-linux"}) lists prefixed tools with
	// mutability markers.
	result, err := tool.Execute(context.Background(), "id", map[string]any{"server": ServerName}, nil)
	if err != nil {
		t.Fatal(err)
	}
	text := toolText(t, result.Content)
	if !strings.Contains(text, "`computer_use_linux_doctor` [read-only]") || !strings.Contains(text, "`computer_use_linux_click` [destructive]") {
		t.Errorf("list output: %s", text)
	}

	// A single search match returns the full description.
	result, _ = tool.Execute(context.Background(), "id", map[string]any{"search": "doctor"}, nil)
	if !strings.Contains(toolText(t, result.Content), "### computer_use_linux_doctor") {
		t.Errorf("search output: %s", toolText(t, result.Content))
	}

	// Screenshots come back as inline images plus the metadata text.
	result, err = tool.Execute(context.Background(), "id", map[string]any{"tool": "computer_use_linux_screenshot"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	var image *llm.ImageContent
	for _, block := range result.Content {
		if img, ok := block.(llm.ImageContent); ok {
			image = &img
		}
	}
	if image == nil || image.MimeType != "image/png" {
		t.Fatalf("screenshot image missing: %+v", result.Content)
	}
	if decoded, _ := base64.StdEncoding.DecodeString(image.Data); string(decoded) != "png-bytes" {
		t.Errorf("image data = %q", image.Data)
	}
	if !strings.Contains(toolText(t, result.Content), "coordinate_width") {
		t.Error("metadata text missing")
	}

	// isError results become tool errors.
	if _, err := tool.Execute(context.Background(), "id", map[string]any{"tool": "click", "args": map[string]any{"x": 1.0}}, nil); err == nil || !strings.Contains(err.Error(), "no such element") {
		t.Errorf("isError must fail the call: %v", err)
	}

	// Unknown tools get guidance, not an error.
	result, err = tool.Execute(context.Background(), "id", map[string]any{"tool": "computer_use_linux_nope"}, nil)
	if err != nil || !strings.Contains(toolText(t, result.Content), "not found") {
		t.Errorf("unknown tool: %v %s", err, toolText(t, result.Content))
	}

	// No parameters yields usage guidance.
	result, _ = tool.Execute(context.Background(), "id", map[string]any{}, nil)
	if !strings.Contains(toolText(t, result.Content), ServerName) {
		t.Error("usage guidance missing")
	}
}

func TestSessionRestartsAfterExit(t *testing.T) {
	t.Setenv("GO_COMPUTERUSE_HELPER", "1")
	session := helperSession(t)
	if _, err := session.Tools(context.Background()); err != nil {
		t.Fatal(err)
	}
	session.Close()
	// A fresh call after Close respawns the server.
	if _, err := session.Call(context.Background(), "doctor", nil); err != nil {
		t.Fatalf("respawn failed: %v", err)
	}
}

func toolText(t *testing.T, blocks []llm.ContentBlock) string {
	t.Helper()
	var parts []string
	for _, block := range blocks {
		if text, ok := block.(llm.TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n")
}
