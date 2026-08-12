// Package tools provides the built-in coding tools the agent exposes to models.
//
// These mirror the intent of pi's coding-agent tools (read, write, edit, bash,
// list) with the same guardrails: paths are confined to the workspace root, and
// edits require an exact unique match.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

// maxReadBytes caps how much of a file is returned in one read.
const maxReadBytes = 50 * 1024

// maxOutputBytes caps captured command output.
const maxOutputBytes = 30 * 1024

// maxEditBytes prevents an edit request from loading an unexpectedly huge
// file into memory. Large files should be changed with a purpose-built tool.
const maxEditBytes = 10 * 1024 * 1024

// Workspace confines tool file access to a single directory tree.
type Workspace struct {
	// Root is the absolute workspace path. Tools refuse to touch anything
	// outside it.
	Root string
	// BashTimeout bounds command execution. Zero means 120s.
	BashTimeout time.Duration

	root *os.Root
}

// NewWorkspace resolves root to an absolute, symlink-free path.
func NewWorkspace(root string) (*Workspace, error) {
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	if resolved, err := filepath.EvalSymlinks(absolute); err == nil {
		absolute = resolved
	}
	info, err := os.Stat(absolute)
	if err != nil {
		return nil, fmt.Errorf("workspace %s: %w", root, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("workspace %s is not a directory", root)
	}
	confinedRoot, err := os.OpenRoot(absolute)
	if err != nil {
		return nil, fmt.Errorf("open workspace %s: %w", root, err)
	}
	return &Workspace{Root: absolute, root: confinedRoot}, nil
}

// Close releases the operating-system handle used for race-free path
// confinement. A closed workspace must not be reused.
func (w *Workspace) Close() error {
	if w == nil || w.root == nil {
		return nil
	}
	return w.root.Close()
}

// resolve converts a tool path to a root-relative name. os.Root performs the
// actual filesystem operation, preventing symlink and rename races from
// escaping the workspace.
func (w *Workspace) resolve(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	name := filepath.Clean(path)
	if filepath.IsAbs(name) {
		var err error
		name, err = filepath.Rel(w.Root, name)
		if err != nil {
			return "", fmt.Errorf("path %s is outside the workspace", path)
		}
	}
	if name == ".." || strings.HasPrefix(name, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %s is outside the workspace", path)
	}
	return name, nil
}

func (w *Workspace) display(path string) string { return filepath.ToSlash(path) }

func (w *Workspace) readLimited(path string, limit int64) ([]byte, bool, error) {
	file, err := w.root.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, limit+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(content)) > limit {
		return content[:limit], true, nil
	}
	return content, false, nil
}

func textResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}

type cappedOutput struct {
	mu        sync.Mutex
	builder   strings.Builder
	limit     int
	truncated bool
}

func (w *cappedOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	originalLength := len(p)
	remaining := w.limit - w.builder.Len()
	if remaining <= 0 {
		w.truncated = w.truncated || originalLength > 0
		return originalLength, nil
	}
	if len(p) > remaining {
		p = p[:remaining]
		w.truncated = true
	}
	_, _ = w.builder.Write(p)
	return originalLength, nil
}

func (w *cappedOutput) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	text := w.builder.String()
	if w.truncated {
		text += "\n[output truncated]"
	}
	return text
}

// All returns the built-in tool set for a workspace.
func (w *Workspace) All() []agent.Tool {
	return []agent.Tool{
		w.ReadTool(),
		w.WriteTool(),
		w.EditTool(),
		w.ListTool(),
		w.BashTool(),
	}
}

// ReadTool reads a UTF-8 text file.
func (w *Workspace) ReadTool() agent.Tool {
	return agent.Tool{
		Name:        "read",
		Label:       "Read",
		Description: "Read the contents of a text file in the workspace. Output is truncated for very large files.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path, relative to the workspace root"}
			},
			"required": ["path"]
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			path, _ := params["path"].(string)
			resolved, err := w.resolve(path)
			if err != nil {
				return agent.ToolResult{}, err
			}
			content, truncated, err := w.readLimited(resolved, maxReadBytes)
			if err != nil {
				return agent.ToolResult{}, err
			}
			text := string(content)
			if truncated {
				text += fmt.Sprintf("\n\n[truncated: showing first %d bytes]", maxReadBytes)
			}
			return textResult(text), nil
		},
	}
}

// WriteTool creates or overwrites a file.
func (w *Workspace) WriteTool() agent.Tool {
	return agent.Tool{
		Name:        "write",
		Label:       "Write",
		Description: "Write content to a file in the workspace, creating parent directories as needed. Overwrites an existing file.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path, relative to the workspace root"},
				"content": {"type": "string", "description": "Full file content to write"}
			},
			"required": ["path", "content"]
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			path, _ := params["path"].(string)
			content, _ := params["content"].(string)
			resolved, err := w.resolve(path)
			if err != nil {
				return agent.ToolResult{}, err
			}
			if err := w.root.MkdirAll(filepath.Dir(resolved), 0o755); err != nil {
				return agent.ToolResult{}, err
			}
			if err := w.root.WriteFile(resolved, []byte(content), 0o644); err != nil {
				return agent.ToolResult{}, err
			}
			return textResult(fmt.Sprintf("Wrote %d bytes to %s", len(content), w.display(resolved))), nil
		},
	}
}

// EditTool replaces an exact, unique substring in a file.
func (w *Workspace) EditTool() agent.Tool {
	return agent.Tool{
		Name:  "edit",
		Label: "Edit",
		Description: "Replace an exact substring in a file. old_text must appear exactly once, " +
			"so include enough surrounding context to make it unique.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "File path, relative to the workspace root"},
				"old_text": {"type": "string", "description": "Exact text to replace; must be unique in the file"},
				"new_text": {"type": "string", "description": "Replacement text"}
			},
			"required": ["path", "old_text", "new_text"]
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			path, _ := params["path"].(string)
			oldText, _ := params["old_text"].(string)
			newText, _ := params["new_text"].(string)
			if oldText == "" {
				return agent.ToolResult{}, fmt.Errorf("old_text must not be empty")
			}
			resolved, err := w.resolve(path)
			if err != nil {
				return agent.ToolResult{}, err
			}
			content, truncated, err := w.readLimited(resolved, maxEditBytes)
			if err != nil {
				return agent.ToolResult{}, err
			}
			if truncated {
				return agent.ToolResult{}, fmt.Errorf("%s exceeds the %d-byte edit limit", w.display(resolved), maxEditBytes)
			}
			text := string(content)
			switch occurrences := strings.Count(text, oldText); occurrences {
			case 0:
				return agent.ToolResult{}, fmt.Errorf("old_text was not found in %s", w.display(resolved))
			case 1:
			default:
				return agent.ToolResult{}, fmt.Errorf("old_text appears %d times in %s; add more context to make it unique",
					occurrences, w.display(resolved))
			}
			updated := strings.Replace(text, oldText, newText, 1)
			if err := w.root.WriteFile(resolved, []byte(updated), 0o644); err != nil {
				return agent.ToolResult{}, err
			}
			return textResult("Edited " + w.display(resolved)), nil
		},
	}
}

// ListTool lists directory entries.
func (w *Workspace) ListTool() agent.Tool {
	return agent.Tool{
		Name:        "list",
		Label:       "List",
		Description: "List the entries of a directory in the workspace. Directories are suffixed with /.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Directory path, relative to the workspace root. Defaults to the root."}
			}
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			path, _ := params["path"].(string)
			if path == "" {
				path = "."
			}
			resolved, err := w.resolve(path)
			if err != nil {
				return agent.ToolResult{}, err
			}
			directory, err := w.root.Open(resolved)
			if err != nil {
				return agent.ToolResult{}, err
			}
			entries, readErr := directory.ReadDir(-1)
			closeErr := directory.Close()
			if readErr != nil {
				return agent.ToolResult{}, readErr
			}
			if closeErr != nil {
				return agent.ToolResult{}, closeErr
			}
			names := make([]string, 0, len(entries))
			for _, entry := range entries {
				name := entry.Name()
				if entry.IsDir() {
					name += "/"
				}
				names = append(names, name)
			}
			sort.Strings(names)
			if len(names) == 0 {
				return textResult("(empty directory)"), nil
			}
			return textResult(strings.Join(names, "\n")), nil
		},
	}
}

// BashTool runs a shell command in the workspace.
//
// SECURITY: this executes arbitrary commands with the user's privileges. It is
// the reason the CLI requires an explicit opt-in flag before enabling tools.
func (w *Workspace) BashTool() agent.Tool {
	return agent.Tool{
		Name:  "bash",
		Label: "Bash",
		Description: "Run a shell command in the workspace and return its combined output. " +
			"Use for builds, tests, and searches.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"command": {"type": "string", "description": "Shell command to run"}
			},
			"required": ["command"]
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			command, _ := params["command"].(string)
			if strings.TrimSpace(command) == "" {
				return agent.ToolResult{}, fmt.Errorf("command is required")
			}

			timeout := w.BashTimeout
			if timeout <= 0 {
				timeout = 120 * time.Second
			}
			runCtx, cancel := context.WithTimeout(ctx, timeout)
			defer cancel()

			shell, args := "sh", []string{"-c"}
			if runtime.GOOS == "windows" {
				if bash, err := exec.LookPath("bash"); err == nil {
					shell, args = bash, []string{"-c"}
				} else {
					shell, args = "cmd.exe", []string{"/d", "/s", "/c"}
				}
			}

			cmd := exec.CommandContext(runCtx, shell, append(args, command)...)
			cmd.Dir = w.Root
			output := cappedOutput{limit: maxOutputBytes}
			cmd.Stdout = &output
			cmd.Stderr = &output

			runErr := cmd.Run()
			text := output.String()
			if runCtx.Err() == context.DeadlineExceeded {
				return agent.ToolResult{}, fmt.Errorf("command timed out after %s\n%s", timeout, text)
			}
			if runErr != nil {
				// A non-zero exit is reported as an error result so the model
				// can react, with the output still attached.
				return agent.ToolResult{}, fmt.Errorf("%v\n%s", runErr, text)
			}
			if strings.TrimSpace(text) == "" {
				return textResult("(no output)"), nil
			}
			return textResult(text), nil
		},
	}
}
