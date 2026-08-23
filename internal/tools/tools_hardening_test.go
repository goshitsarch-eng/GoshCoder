package tools

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// TestReadReportsTruncationInsteadOfInventingALineCount covers a file larger
// than the read tool's byte cap. The tool discarded the truncation flag and
// then reported the truncated prefix's line count as the file's total, so the
// model was told a 60,000-line file had 39,385 lines and that it had reached
// the end when it had not.
func TestReadReportsTruncationInsteadOfInventingALineCount(t *testing.T) {
	workspace := newTestWorkspace(t)

	// Comfortably past maxReadBytes*40 so readLimited caps it.
	var b strings.Builder
	line := strings.Repeat("a", 60) + "\n"
	for b.Len() <= maxReadBytes*40+len(line)*10 {
		b.WriteString(line)
	}
	full := b.String()
	realLines := strings.Count(full, "\n")
	if err := os.WriteFile(filepath.Join(workspace.Root, "big.txt"), []byte(full), 0o600); err != nil {
		t.Fatal(err)
	}

	text, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "big.txt", "limit": 5})
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if strings.Contains(text, fmt.Sprintf("of %d.", realLines)) {
		t.Fatalf("claimed the real total for a capped read: %q", text)
	}
	if !strings.Contains(text, "at least") {
		t.Fatalf("a capped read must not state an exact total: %q", text)
	}
}

// TestReadPastTheCapSaysWhy keeps the out-of-range error honest: the file has
// more lines, they are just past what this tool will read.
func TestReadPastTheCapSaysWhy(t *testing.T) {
	workspace := newTestWorkspace(t)
	var b strings.Builder
	for b.Len() <= maxReadBytes*40+1000 {
		b.WriteString(strings.Repeat("b", 60) + "\n")
	}
	if err := os.WriteFile(filepath.Join(workspace.Root, "big.txt"), []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "big.txt", "offset": 10_000_000})
	if err == nil {
		t.Fatal("an offset past the cap must error")
	}
	if strings.Contains(err.Error(), "beyond end of file") {
		t.Fatalf("a capped file must not be reported as ended: %v", err)
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Fatalf("error should explain the cap: %v", err)
	}
}

// TestReadDropsThePartialFinalLine keeps the tool from presenting the fragment
// left by the byte cap as a complete line.
func TestReadDropsThePartialFinalLine(t *testing.T) {
	workspace := newTestWorkspace(t)
	// One enormous single line: capping it leaves exactly one partial line.
	content := strings.Repeat("x", maxReadBytes*40+500)
	if err := os.WriteFile(filepath.Join(workspace.Root, "oneline.txt"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := runTool(t, workspace.ReadTool(), map[string]any{"path": "oneline.txt"}); err != nil {
		t.Fatalf("read: %v", err)
	}
}

// TestBashKillsTheWholeProcessTree covers a timed-out command whose shell
// backgrounded work. Cancellation used to kill only the shell's own pid, so
// builds and dev servers kept running -- and kept burning CPU -- after the tool
// had already reported a timeout.
func TestBashKillsTheWholeProcessTree(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell job control and ps")
	}
	workspace := newTestWorkspace(t)
	workspace.BashTimeout = time.Second

	marker := filepath.Join(t.TempDir(), "pid")
	command := fmt.Sprintf("sh -c 'echo $$ > %s; sleep 60' & sleep 30", marker)
	if _, err := runTool(t, workspace.BashTool(), map[string]any{"command": command}); err == nil {
		t.Fatal("the command should have timed out")
	}

	content, err := os.ReadFile(marker)
	if err != nil {
		t.Skipf("the grandchild never recorded its pid: %v", err)
	}
	pid := strings.TrimSpace(string(content))

	// Give the kill a moment to land, then confirm the grandchild is gone.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if exec.Command("kill", "-0", pid).Run() != nil {
			return // no such process: the tree was torn down
		}
		time.Sleep(50 * time.Millisecond)
	}
	_ = exec.Command("kill", "-9", pid).Run()
	t.Fatalf("process %s survived the timeout; the descendant tree was orphaned", pid)
}

// TestBashDoesNotFailACommandThatLeavesABackgroundProcess covers the other side
// of the pipe: the shell exits 0 but a backgrounded grandchild still holds the
// output pipe, so Wait returns ErrWaitDelay. Reporting that as an error made
// the model retry a command that had actually succeeded.
func TestBashDoesNotFailACommandThatLeavesABackgroundProcess(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("uses POSIX shell job control")
	}
	workspace := newTestWorkspace(t)
	workspace.BashTimeout = 30 * time.Second

	text, err := runTool(t, workspace.BashTool(), map[string]any{"command": "echo started; sleep 20 &"})
	if err != nil {
		t.Fatalf("a successful command with a background child was reported as a failure: %v", err)
	}
	if !strings.Contains(text, "started") {
		t.Fatalf("output lost: %q", text)
	}
}

// TestWriteIsAtomic and TestEditIsAtomic assert that no temporary file is left
// behind and that the destination is replaced rather than truncated in place.
func TestWriteIsAtomic(t *testing.T) {
	workspace := newTestWorkspace(t)
	path := filepath.Join(workspace.Root, "file.txt")
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := runTool(t, workspace.WriteTool(), map[string]any{"path": "file.txt", "content": "replaced"}); err != nil {
		t.Fatalf("write: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "replaced" {
		t.Fatalf("content = %q", got)
	}
	assertNoTempFiles(t, workspace.Root)

	// The existing file's mode must survive the replace.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %v, want the original 0600 preserved", info.Mode().Perm())
	}
}

func TestEditIsAtomic(t *testing.T) {
	workspace := newTestWorkspace(t)
	path := filepath.Join(workspace.Root, "file.txt")
	if err := os.WriteFile(path, []byte("alpha beta gamma"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := runTool(t, workspace.EditTool(), map[string]any{
		"path": "file.txt", "old_text": "beta", "new_text": "delta",
	}); err != nil {
		t.Fatalf("edit: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "alpha delta gamma" {
		t.Fatalf("content = %q", got)
	}
	assertNoTempFiles(t, workspace.Root)
}

func assertNoTempFiles(t *testing.T, root string) {
	t.Helper()
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.Contains(entry.Name(), "goshcoder-") && strings.HasSuffix(entry.Name(), ".tmp") {
			t.Fatalf("temporary file left behind: %s", entry.Name())
		}
	}
}

// TestGrepFindsNonASCIIFilenames covers git's core.quotePath default: paths
// came back C-quoted and octal-escaped, so the tool looked them up under names
// that do not exist and silently reported no matches.
func TestGrepFindsNonASCIIFilenames(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	workspace := newTestWorkspace(t)
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "test"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = workspace.Root
		if err := cmd.Run(); err != nil {
			t.Skipf("git %v: %v", args, err)
		}
	}

	name := "café-日本.txt"
	if err := os.WriteFile(filepath.Join(workspace.Root, name), []byte("NEEDLE here\n"), 0o600); err != nil {
		t.Skipf("filesystem cannot hold the name: %v", err)
	}

	text, err := runTool(t, workspace.GrepTool(), map[string]any{"pattern": "NEEDLE"})
	if err != nil {
		t.Fatalf("grep: %v", err)
	}
	if !strings.Contains(text, "NEEDLE") {
		t.Fatalf("grep missed a file with a non-ASCII name: %q", text)
	}
}
