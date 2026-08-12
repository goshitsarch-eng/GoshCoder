package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

// newTestWorkspace builds a workspace over a fresh temp directory.
func newTestWorkspace(t *testing.T) *Workspace {
	t.Helper()
	workspace, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return workspace
}

// runTool executes a tool and returns its text output.
func runTool(t *testing.T, tool agent.Tool, params map[string]any) (string, error) {
	t.Helper()
	result, err := tool.Execute(context.Background(), "call-1", params, func(agent.ToolResult) {})
	if err != nil {
		return "", err
	}
	var parts []string
	for _, block := range result.Content {
		if text, ok := block.(llm.TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n"), nil
}

func TestNewWorkspaceRejectsFiles(t *testing.T) {
	file := filepath.Join(t.TempDir(), "a.txt")
	if err := os.WriteFile(file, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := NewWorkspace(file); err == nil {
		t.Fatal("a file must not be accepted as a workspace")
	}
	if _, err := NewWorkspace(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("a missing directory must not be accepted")
	}
}

func TestWriteAndReadRoundTrip(t *testing.T) {
	workspace := newTestWorkspace(t)

	out, err := runTool(t, workspace.WriteTool(), map[string]any{
		"path":    "notes/todo.txt",
		"content": "first line\nsecond line",
	})
	if err != nil {
		t.Fatalf("write: %v", err)
	}
	if !strings.Contains(out, "notes/todo.txt") {
		t.Fatalf("write output = %q", out)
	}

	// Parent directories are created on demand.
	if _, err := os.Stat(filepath.Join(workspace.Root, "notes", "todo.txt")); err != nil {
		t.Fatalf("file was not created: %v", err)
	}

	got, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "notes/todo.txt"})
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got != "first line\nsecond line" {
		t.Fatalf("read = %q", got)
	}
}

func TestReadMissingFileErrors(t *testing.T) {
	workspace := newTestWorkspace(t)
	if _, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "nope.txt"}); err == nil {
		t.Fatal("expected an error for a missing file")
	}
}

func TestReadTruncatesLargeFiles(t *testing.T) {
	workspace := newTestWorkspace(t)
	large := strings.Repeat("x", maxReadBytes+500)
	if err := os.WriteFile(filepath.Join(workspace.Root, "big.txt"), []byte(large), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "big.txt"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "[truncated:") {
		t.Fatal("large reads should be marked as truncated")
	}
	if len(got) > maxReadBytes+200 {
		t.Fatalf("truncated read is still %d bytes", len(got))
	}
}

// TestPathConfinement is the security-critical case: tools must not touch
// anything outside the workspace root.
func TestPathConfinement(t *testing.T) {
	workspace := newTestWorkspace(t)
	outside := filepath.Join(filepath.Dir(workspace.Root), "outside.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0o644); err != nil {
		t.Fatal(err)
	}

	escapes := []string{
		"../outside.txt",
		"../../outside.txt",
		"notes/../../outside.txt",
		outside,
	}
	for _, path := range escapes {
		t.Run(path, func(t *testing.T) {
			if _, err := runTool(t, workspace.ReadTool(), map[string]any{"path": path}); err == nil {
				t.Fatalf("read escaped the workspace via %q", path)
			}
			if _, err := runTool(t, workspace.WriteTool(), map[string]any{"path": path, "content": "pwned"}); err == nil {
				t.Fatalf("write escaped the workspace via %q", path)
			}
		})
	}

	// The outside file must be untouched.
	content, err := os.ReadFile(outside)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "secret" {
		t.Fatalf("outside file was modified: %q", content)
	}
}

func TestEmptyPathIsRejected(t *testing.T) {
	workspace := newTestWorkspace(t)
	if _, err := runTool(t, workspace.ReadTool(), map[string]any{"path": ""}); err == nil {
		t.Fatal("an empty path must be rejected")
	}
}

func TestEditReplacesUniqueMatch(t *testing.T) {
	workspace := newTestWorkspace(t)
	path := filepath.Join(workspace.Root, "code.go")
	if err := os.WriteFile(path, []byte("package main\n\nfunc old() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := runTool(t, workspace.EditTool(), map[string]any{
		"path":     "code.go",
		"old_text": "func old() {}",
		"new_text": "func renamed() {}",
	}); err != nil {
		t.Fatalf("edit: %v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), "func renamed() {}") {
		t.Fatalf("file = %q", content)
	}
}

func TestEditRejectsAmbiguousAndMissingMatches(t *testing.T) {
	workspace := newTestWorkspace(t)
	path := filepath.Join(workspace.Root, "dup.txt")
	if err := os.WriteFile(path, []byte("repeat\nrepeat\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Two occurrences: the edit must refuse rather than guess.
	_, err := runTool(t, workspace.EditTool(), map[string]any{
		"path": "dup.txt", "old_text": "repeat", "new_text": "changed",
	})
	if err == nil || !strings.Contains(err.Error(), "appears 2 times") {
		t.Fatalf("err = %v", err)
	}

	_, err = runTool(t, workspace.EditTool(), map[string]any{
		"path": "dup.txt", "old_text": "absent", "new_text": "x",
	})
	if err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("err = %v", err)
	}

	// Neither failure changed the file.
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "repeat\nrepeat\n" {
		t.Fatalf("file was modified: %q", content)
	}
}

func TestEditRejectsEmptyOldText(t *testing.T) {
	workspace := newTestWorkspace(t)
	if err := os.WriteFile(filepath.Join(workspace.Root, "a.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := runTool(t, workspace.EditTool(), map[string]any{
		"path": "a.txt", "old_text": "", "new_text": "y",
	})
	if err == nil || !strings.Contains(err.Error(), "must not be empty") {
		t.Fatalf("err = %v", err)
	}
}

func TestListTool(t *testing.T) {
	workspace := newTestWorkspace(t)
	if err := os.MkdirAll(filepath.Join(workspace.Root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workspace.Root, "a.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := runTool(t, workspace.ListTool(), nil)
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	// Directories are suffixed so the model can tell them apart.
	if !strings.Contains(got, "sub/") || !strings.Contains(got, "a.txt") {
		t.Fatalf("list = %q", got)
	}

	empty, err := runTool(t, workspace.ListTool(), map[string]any{"path": "sub"})
	if err != nil {
		t.Fatal(err)
	}
	if empty != "(empty directory)" {
		t.Fatalf("empty list = %q", empty)
	}
}

func TestBashToolRunsInWorkspace(t *testing.T) {
	workspace := newTestWorkspace(t)
	if err := os.WriteFile(filepath.Join(workspace.Root, "marker.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := runTool(t, workspace.BashTool(), map[string]any{"command": "ls"})
	if err != nil {
		t.Skipf("no usable shell in this environment: %v", err)
	}
	if !strings.Contains(got, "marker.txt") {
		t.Fatalf("bash output = %q, want the workspace listing", got)
	}
}

func TestBashToolReportsFailures(t *testing.T) {
	workspace := newTestWorkspace(t)
	_, err := runTool(t, workspace.BashTool(), map[string]any{"command": "exit 3"})
	if err == nil {
		t.Fatal("a non-zero exit must be reported as an error")
	}
}

func TestBashToolRequiresCommand(t *testing.T) {
	workspace := newTestWorkspace(t)
	if _, err := runTool(t, workspace.BashTool(), map[string]any{"command": "   "}); err == nil {
		t.Fatal("a blank command must be rejected")
	}
}

func TestAllToolsHaveSchemas(t *testing.T) {
	workspace := newTestWorkspace(t)
	all := workspace.All()
	if len(all) != 5 {
		t.Fatalf("tool count = %d, want 5", len(all))
	}
	seen := map[string]bool{}
	for _, tool := range all {
		if tool.Name == "" || tool.Description == "" {
			t.Fatalf("tool is missing metadata: %#v", tool)
		}
		if len(tool.Parameters) == 0 {
			t.Fatalf("tool %s has no parameter schema", tool.Name)
		}
		if tool.Execute == nil {
			t.Fatalf("tool %s has no Execute", tool.Name)
		}
		if seen[tool.Name] {
			t.Fatalf("duplicate tool name %q", tool.Name)
		}
		seen[tool.Name] = true
	}
}
