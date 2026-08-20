// Package tools provides the built-in coding tools the agent exposes to models.
//
// These mirror pi's coding-agent tools (read, write, edit, bash, grep, find,
// and ls) with the same guardrails: paths are confined to the workspace root, and
// edits require an exact unique match.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

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

const (
	maxCandidateBytes = 8 * 1024 * 1024
	maxCandidateFiles = 100_000
)

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

func clipUTF8(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
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

func (w *cappedOutput) Truncated() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.truncated
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

// All returns pi's seven built-in tools for a workspace.
func (w *Workspace) All() []agent.Tool {
	return append(w.Planning(), w.BashTool())
}

// Planning returns the filesystem tools that are safe for plan mode. Bash is
// intentionally excluded because shell commands can bypass markdown-only
// write restrictions.
func (w *Workspace) Planning() []agent.Tool {
	return []agent.Tool{w.ReadTool(), w.WriteTool(), w.EditTool(), w.LsTool(), w.GrepTool(), w.FindTool()}
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
				"path": {"type": "string", "description": "File path, relative to the workspace root"},
				"offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
				"limit": {"type": "number", "description": "Maximum number of lines to read"}
			},
			"required": ["path"]
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			path, _ := params["path"].(string)
			resolved, err := w.resolve(path)
			if err != nil {
				return agent.ToolResult{}, err
			}
			content, _, err := w.readLimited(resolved, maxReadBytes*40)
			if err != nil {
				return agent.ToolResult{}, err
			}
			lines := strings.Split(strings.ReplaceAll(string(content), "\r\n", "\n"), "\n")
			offset := numericParam(params["offset"], 1)
			if offset < 1 {
				offset = 1
			}
			if offset > len(lines) {
				return agent.ToolResult{}, fmt.Errorf("offset %d is beyond end of file (%d lines total)", offset, len(lines))
			}
			end := len(lines)
			if limit := numericParam(params["limit"], 0); limit > 0 {
				end = min(end, offset-1+limit)
			}
			selected := strings.Join(lines[offset-1:end], "\n")
			truncated := false
			if len(selected) > maxReadBytes {
				selected = clipUTF8(selected, maxReadBytes)
				if cut := strings.LastIndexByte(selected, '\n'); cut >= 0 {
					selected = selected[:cut]
				}
				truncated = true
			}
			shownLines := strings.Count(selected, "\n") + 1
			lastLine := offset + shownLines - 1
			if truncated {
				selected += fmt.Sprintf("\n\n[truncated: showing at most %d bytes from line %d]", maxReadBytes, offset)
			} else if lastLine < len(lines) {
				selected += fmt.Sprintf("\n\n[Showing lines %d-%d of %d. Use offset=%d to continue.]", offset, lastLine, len(lines), lastLine+1)
			}
			return textResult(selected), nil
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

// LsTool lists directory entries using pi's built-in tool name.
func (w *Workspace) LsTool() agent.Tool {
	tool := w.ListTool()
	tool.Name = "ls"
	tool.Label = "ls"
	tool.Description = "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles."
	tool.Parameters = json.RawMessage(`{
		"type":"object",
		"properties":{
			"path":{"type":"string","description":"Directory to list (default: current directory)"},
			"limit":{"type":"number","description":"Maximum number of entries (default: 500)"}
		}
	}`)
	original := tool.Execute
	tool.Execute = func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
		result, err := original(ctx, id, params, onUpdate)
		if err != nil {
			return result, err
		}
		limit := numericParam(params["limit"], 500)
		if limit <= 0 {
			limit = 500
		}
		if len(result.Content) == 0 {
			return result, nil
		}
		text, ok := result.Content[0].(llm.TextContent)
		if !ok {
			return result, nil
		}
		lines := strings.Split(text.Text, "\n")
		if len(lines) > limit {
			text.Text = strings.Join(lines[:limit], "\n") + fmt.Sprintf("\n\n[%d entries limit reached]", limit)
			result.Content[0] = text
		}
		return result, nil
	}
	return tool
}

// GrepTool searches text files while using git's file list when available so
// ignored files match pi's behavior without requiring a bundled ripgrep.
func (w *Workspace) GrepTool() agent.Tool {
	return agent.Tool{
		Name: "grep", Label: "grep",
		Description: "Search file contents for a regex or literal string. Returns matching lines with file paths and line numbers. Respects .gitignore in git workspaces. Output is limited to 100 matches by default.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"pattern":{"type":"string","description":"Search pattern (regex or literal string)"},
				"path":{"type":"string","description":"Directory or file to search (default: current directory)"},
				"glob":{"type":"string","description":"Filter files by glob pattern"},
				"ignoreCase":{"type":"boolean","description":"Case-insensitive search"},
				"literal":{"type":"boolean","description":"Treat pattern as a literal string"},
				"context":{"type":"number","description":"Lines before and after each match"},
				"limit":{"type":"number","description":"Maximum matches (default: 100)"}
			},
			"required":["pattern"]
		}`),
		Execute: func(ctx context.Context, _ string, params map[string]any, _ func(agent.ToolResult)) (agent.ToolResult, error) {
			pattern, _ := params["pattern"].(string)
			if pattern == "" {
				return agent.ToolResult{}, errors.New("pattern is required")
			}
			if literal, _ := params["literal"].(bool); literal {
				pattern = regexp.QuoteMeta(pattern)
			}
			if ignoreCase, _ := params["ignoreCase"].(bool); ignoreCase {
				pattern = "(?i)" + pattern
			}
			re, err := regexp.Compile(pattern)
			if err != nil {
				return agent.ToolResult{}, fmt.Errorf("invalid pattern: %w", err)
			}
			searchPath, _ := params["path"].(string)
			if searchPath == "" {
				searchPath = "."
			}
			resolved, err := w.resolve(searchPath)
			if err != nil {
				return agent.ToolResult{}, err
			}
			files, err := w.candidateFiles(ctx, resolved)
			if err != nil {
				return agent.ToolResult{}, err
			}
			glob, _ := params["glob"].(string)
			limit := numericParam(params["limit"], 100)
			if limit <= 0 {
				limit = 100
			}
			contextLines := numericParam(params["context"], 0)
			if contextLines < 0 {
				contextLines = 0
			}
			var output []string
			matches := 0
			for _, file := range files {
				if ctx.Err() != nil {
					return agent.ToolResult{}, ctx.Err()
				}
				if glob != "" && !globMatches(glob, filepath.ToSlash(file)) {
					continue
				}
				content, truncated, readErr := w.readLimited(file, 4*1024*1024)
				if readErr != nil || truncated || strings.IndexByte(string(content), 0) >= 0 {
					continue
				}
				lines := strings.Split(strings.ReplaceAll(string(content), "\r\n", "\n"), "\n")
				for index, line := range lines {
					if !re.MatchString(line) {
						continue
					}
					matches++
					start, end := max(0, index-contextLines), min(len(lines)-1, index+contextLines)
					for row := start; row <= end; row++ {
						separator := "-"
						if row == index {
							separator = ":"
						}
						value := lines[row]
						if len(value) > 2000 {
							value = clipUTF8(value, 2000) + "…"
						}
						output = append(output, fmt.Sprintf("%s%s%d%s %s", filepath.ToSlash(file), separator, row+1, separator, value))
					}
					if matches >= limit {
						output = append(output, fmt.Sprintf("\n[%d matches limit reached. Refine the pattern or increase limit.]", limit))
						return textResult(strings.Join(output, "\n")), nil
					}
				}
			}
			if len(output) == 0 {
				return textResult("No matches found"), nil
			}
			return textResult(strings.Join(output, "\n")), nil
		},
	}
}

// FindTool searches workspace paths by glob and respects gitignore when git
// can provide the candidate list.
func (w *Workspace) FindTool() agent.Tool {
	return agent.Tool{
		Name: "find", Label: "find",
		Description: "Search for files by glob pattern. Returns paths relative to the search directory and respects .gitignore. Output is limited to 1000 results by default.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"pattern":{"type":"string","description":"Glob pattern such as '*.go' or 'internal/**/*.go'"},
				"path":{"type":"string","description":"Directory to search (default: current directory)"},
				"limit":{"type":"number","description":"Maximum results (default: 1000)"}
			},
			"required":["pattern"]
		}`),
		Execute: func(ctx context.Context, _ string, params map[string]any, _ func(agent.ToolResult)) (agent.ToolResult, error) {
			pattern, _ := params["pattern"].(string)
			if pattern == "" {
				return agent.ToolResult{}, errors.New("pattern is required")
			}
			searchPath, _ := params["path"].(string)
			if searchPath == "" {
				searchPath = "."
			}
			resolved, err := w.resolve(searchPath)
			if err != nil {
				return agent.ToolResult{}, err
			}
			files, err := w.candidateFiles(ctx, resolved)
			if err != nil {
				return agent.ToolResult{}, err
			}
			limit := numericParam(params["limit"], 1000)
			if limit <= 0 {
				limit = 1000
			}
			var matches []string
			limited := false
			for _, file := range files {
				relative := filepath.ToSlash(file)
				if resolved != "." {
					relative = strings.TrimPrefix(relative, filepath.ToSlash(resolved)+"/")
				}
				if globMatches(pattern, relative) {
					matches = append(matches, relative)
					if len(matches) >= limit {
						limited = true
						break
					}
				}
			}
			if len(matches) == 0 {
				return textResult("No files found matching pattern"), nil
			}
			sort.Strings(matches)
			if limited {
				matches = append(matches, fmt.Sprintf("\n[%d results limit reached. Refine the pattern or increase limit.]", limit))
			}
			return textResult(strings.Join(matches, "\n")), nil
		},
	}
}

func (w *Workspace) candidateFiles(ctx context.Context, resolved string) ([]string, error) {
	info, err := w.root.Stat(resolved)
	if err != nil {
		return nil, err
	}
	if info.Mode().IsRegular() {
		return []string{resolved}, nil
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("not a regular file or directory: %s", resolved)
	}

	// git ls-files includes tracked and untracked files while applying the
	// repository's complete ignore stack. Fall back to a conservative walk.
	gitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	command := exec.CommandContext(gitCtx, "git", "ls-files", "-co", "--exclude-standard", "--", resolved)
	command.Dir = w.Root
	command.WaitDelay = time.Second
	output := cappedOutput{limit: maxCandidateBytes}
	command.Stdout = &output
	if commandErr := command.Run(); commandErr == nil {
		if output.Truncated() {
			return nil, fmt.Errorf("candidate file list exceeds %d bytes", maxCandidateBytes)
		}
		var files []string
		for _, line := range strings.Split(strings.TrimSpace(output.String()), "\n") {
			line = filepath.Clean(line)
			if line != "." && line != "" {
				files = append(files, line)
				if len(files) > maxCandidateFiles {
					return nil, fmt.Errorf("candidate file list exceeds %d files", maxCandidateFiles)
				}
			}
		}
		sort.Strings(files)
		return files, nil
	}

	var files []string
	start := filepath.Join(w.Root, resolved)
	err = filepath.WalkDir(start, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if entry.IsDir() {
			if path != start && (entry.Name() == ".git" || entry.Name() == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		relative, relErr := filepath.Rel(w.Root, path)
		if relErr == nil {
			files = append(files, relative)
			if len(files) > maxCandidateFiles {
				return fmt.Errorf("candidate file list exceeds %d files", maxCandidateFiles)
			}
		}
		return nil
	})
	sort.Strings(files)
	return files, err
}

func globMatches(pattern, name string) bool {
	pattern = filepath.ToSlash(pattern)
	name = filepath.ToSlash(name)
	var expression strings.Builder
	expression.WriteByte('^')
	for index := 0; index < len(pattern); index++ {
		switch pattern[index] {
		case '*':
			if index+1 < len(pattern) && pattern[index+1] == '*' {
				index++
				if index+1 < len(pattern) && pattern[index+1] == '/' {
					index++
					expression.WriteString("(?:.*/)?")
				} else {
					expression.WriteString(".*")
				}
			} else {
				expression.WriteString("[^/]*")
			}
		case '?':
			expression.WriteString("[^/]")
		default:
			expression.WriteString(regexp.QuoteMeta(string(pattern[index])))
		}
	}
	expression.WriteByte('$')
	matched, _ := regexp.MatchString(expression.String(), name)
	if matched {
		return true
	}
	// A basename-only pattern applies at every depth, matching fd/rg globs.
	return !strings.Contains(pattern, "/") && regexp.MustCompile(expression.String()).MatchString(filepath.Base(name))
}

func numericParam(value any, fallback int) int {
	switch number := value.(type) {
	case int:
		return number
	case float64:
		return int(number)
	case json.Number:
		parsed, err := strconv.Atoi(number.String())
		if err == nil {
			return parsed
		}
	}
	return fallback
}

// BashTool runs a shell command in the workspace.
//
// SECURITY: this executes arbitrary commands with the user's privileges. Chat
// enables coding tools by default; callers can explicitly choose read-only mode.
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
			cmd.WaitDelay = 2 * time.Second
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
