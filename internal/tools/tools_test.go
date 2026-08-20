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
	t.Cleanup(func() { _ = workspace.Close() })
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

func TestNestedSymlinkCannotEscapeWorkspace(t *testing.T) {
	workspace := newTestWorkspace(t)
	outside := t.TempDir()
	link := filepath.Join(workspace.Root, "link")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	_, err := runTool(t, workspace.WriteTool(), map[string]any{
		"path": "link/new/file.txt", "content": "secret",
	})
	if err == nil {
		t.Fatal("write escaped through a symlinked ancestor")
	}
	if _, statErr := os.Stat(filepath.Join(outside, "new", "file.txt")); !os.IsNotExist(statErr) {
		t.Fatalf("outside file exists or stat failed unexpectedly: %v", statErr)
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

func TestReadToolSupportsOffsetAndLimit(t *testing.T) {
	workspace := newTestWorkspace(t)
	if err := os.WriteFile(filepath.Join(workspace.Root, "lines.txt"), []byte("one\ntwo\nthree\nfour\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "lines.txt", "offset": 2.0, "limit": 2.0})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(got, "two\nthree") || !strings.Contains(got, "offset=4") {
		t.Fatalf("read = %q", got)
	}
}

func TestSearchAndLsTools(t *testing.T) {
	workspace := newTestWorkspace(t)
	if err := os.MkdirAll(filepath.Join(workspace.Root, "src"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workspace.Root, "src", "main.go"), []byte("package main\n// Needle\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	found, err := runTool(t, workspace.FindTool(), map[string]any{"pattern": "**/*.go"})
	if err != nil || !strings.Contains(found, "src/main.go") {
		t.Fatalf("find = %q, %v", found, err)
	}
	matched, err := runTool(t, workspace.GrepTool(), map[string]any{"pattern": "needle", "ignoreCase": true})
	if err != nil || !strings.Contains(matched, "src/main.go:2:") {
		t.Fatalf("grep = %q, %v", matched, err)
	}
	listed, err := runTool(t, workspace.LsTool(), map[string]any{"path": "src"})
	if err != nil || !strings.Contains(listed, "main.go") {
		t.Fatalf("ls = %q, %v", listed, err)
	}
}

func TestFindLimitNoticeFollowsSortedResults(t *testing.T) {
	workspace := newTestWorkspace(t)
	for _, name := range []string{"b.go", "a.go"} {
		if err := os.WriteFile(filepath.Join(workspace.Root, name), []byte("package p"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	found, err := runTool(t, workspace.FindTool(), map[string]any{"pattern": "*.go", "limit": 1.0})
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(found, "\n")
	if lines[0] != "a.go" || !strings.Contains(lines[len(lines)-1], "limit reached") {
		t.Fatalf("limited find = %q", found)
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

func TestFindSkipsVendorAndDependencyTrees(t *testing.T) {
	workspace := newTestWorkspace(t)
	for _, dir := range []string{"src", "vendor/pkg", "node_modules/dep"} {
		if err := os.MkdirAll(filepath.Join(workspace.Root, dir), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(workspace.Root, "src", "main.go"), []byte("package main"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workspace.Root, "vendor", "pkg", "lib.go"), []byte("package pkg"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(workspace.Root, "node_modules", "dep", "index.go"), []byte("package dep"), 0o644); err != nil {
		t.Fatal(err)
	}
	found, err := runTool(t, workspace.FindTool(), map[string]any{"pattern": "**/*.go"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(found, "src/main.go") {
		t.Fatalf("workspace go file missing: %q", found)
	}
	if strings.Contains(found, "vendor") || strings.Contains(found, "node_modules") {
		t.Fatalf("dependency tree was searched: %q", found)
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

func TestCappedOutputDiscardsExcessData(t *testing.T) {
	var output cappedOutput
	output.limit = 4
	if n, err := output.Write([]byte("abcdef")); err != nil || n != 6 {
		t.Fatalf("Write = %d, %v", n, err)
	}
	if got := output.String(); got != "abcd\n[output truncated]" {
		t.Fatalf("output = %q", got)
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
	if len(workspace.Planning()) != 6 {
		t.Fatalf("planning tool count = %d, want 6", len(workspace.Planning()))
	}
	for _, tool := range workspace.Planning() {
		if tool.Name == "bash" {
			t.Fatal("planning tools include bash")
		}
	}
	if len(all) != 7 {
		t.Fatalf("tool count = %d, want 7", len(all))
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

func TestClipUTF8DoesNotSplitRunes(t *testing.T) {
	text := "你"
	if got := clipUTF8(text, 1); got != "" {
		t.Fatalf("partial rune = %q", got)
	}
	if got := clipUTF8(text+text, len(text)+1); got != text {
		t.Fatalf("split pair = %q", got)
	}
}
