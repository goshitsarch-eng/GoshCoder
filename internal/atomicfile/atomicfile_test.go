package atomicfile

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestWriteCreatesWithTheRequestedMode covers the reason the temporary is
// chmodded before the rename rather than after: a credential-bearing file must
// never exist, even briefly, at the process umask.
func TestWriteCreatesWithTheRequestedMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows does not honor Unix permission bits")
	}
	path := filepath.Join(t.TempDir(), "secret.json")

	if err := Write(path, []byte(`{"k":"v"}`), 0o600); err != nil {
		t.Fatalf("Write: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("mode = %o, want 600", got)
	}
}

// TestWriteReplacesExistingContentWholesale is the baseline: a rename over an
// existing file must publish the new bytes, not append to or merge with them.
func TestWriteReplacesExistingContentWholesale(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.json")

	if err := Write(path, []byte("first-and-much-longer"), 0o600); err != nil {
		t.Fatalf("first Write: %v", err)
	}
	if err := Write(path, []byte("second"), 0o600); err != nil {
		t.Fatalf("second Write: %v", err)
	}

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(content) != "second" {
		t.Fatalf("content = %q, want the second write alone", content)
	}
}

// TestWriteCreatesTheParentDirectory covers first-run: every caller writes into
// the agent directory, which may not exist yet on a fresh machine.
func TestWriteCreatesTheParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "deeper", "state.json")

	if err := Write(path, []byte("x"), 0o600); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("stat: %v", err)
	}
}

// TestFailedWriteLeavesTheOriginalIntact covers the crash-safety promise. The
// destination is a directory here, so the rename fails after the temporary is
// fully written -- the same shape as a crash between staging and publishing.
func TestFailedWriteLeavesTheOriginalIntact(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "occupied")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatalf("seed: %v", err)
	}

	if err := Write(path, []byte("new"), 0o600); err == nil {
		t.Fatal("Write over a directory succeeded; want an error")
	}
	info, err := os.Stat(path)
	if err != nil || !info.IsDir() {
		t.Fatalf("destination was damaged by a failed write: %v", err)
	}
}

// TestFailedWriteLeavesNoTemporaryBehind covers the cleanup path: a staged file
// that never got published must not accumulate in the agent directory.
func TestFailedWriteLeavesNoTemporaryBehind(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "occupied")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatalf("seed: %v", err)
	}

	_ = Write(path, []byte("new"), 0o600)

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	for _, entry := range entries {
		if entry.Name() != "occupied" {
			t.Fatalf("a temporary file was left behind: %s", entry.Name())
		}
	}
}
