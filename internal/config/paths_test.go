package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExpandTilde(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home directory in this environment")
	}

	tests := []struct {
		in   string
		want string
	}{
		{"~", home},
		{"~/sub", filepath.Join(home, "sub")},
		// Only a leading "~" segment expands.
		{"/absolute/path", "/absolute/path"},
		{"relative/path", "relative/path"},
		{"~notme/path", "~notme/path"},
		{"", ""},
	}
	for _, tt := range tests {
		if got := ExpandTilde(tt.in); got != tt.want {
			t.Errorf("ExpandTilde(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestAgentDirHonorsEnvOverride(t *testing.T) {
	override := t.TempDir()
	t.Setenv(EnvAgentDir, override)

	if got := AgentDir(); got != override {
		t.Fatalf("AgentDir = %q, want %q", got, override)
	}
}

// TestAgentDirExpandsTildeInOverride covers a "~/..." override, which shells do
// not expand when the value is quoted.
func TestAgentDirExpandsTildeInOverride(t *testing.T) {
	t.Setenv(EnvAgentDir, "~/goshcoder-test-dir")

	got := AgentDir()
	if strings.HasPrefix(got, "~") {
		t.Fatalf("AgentDir = %q, want the tilde expanded", got)
	}
	if !strings.HasSuffix(filepath.ToSlash(got), "/goshcoder-test-dir") {
		t.Fatalf("AgentDir = %q", got)
	}
}

func TestAgentDirDefault(t *testing.T) {
	t.Setenv(EnvAgentDir, "")

	got := AgentDir()
	// The default lives under the per-user config directory.
	if !strings.Contains(filepath.ToSlash(got), DirName+"/agent") {
		t.Fatalf("AgentDir = %q, want it under %s/agent", got, DirName)
	}
	// It must be distinct from pi's directory so the two can coexist.
	if strings.Contains(filepath.ToSlash(got), "/.pi/") {
		t.Fatalf("AgentDir = %q, must not collide with pi", got)
	}
}

func TestAuthPath(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(EnvAgentDir, dir)

	want := filepath.Join(dir, "auth.json")
	if got := AuthPath(); got != want {
		t.Fatalf("AuthPath = %q, want %q", got, want)
	}
}

func TestEnsureAgentDir(t *testing.T) {
	// A nested path exercises directory creation.
	dir := filepath.Join(t.TempDir(), "nested", "agent")
	t.Setenv(EnvAgentDir, dir)

	created, err := EnsureAgentDir()
	if err != nil {
		t.Fatalf("EnsureAgentDir: %v", err)
	}
	if created != dir {
		t.Fatalf("created = %q, want %q", created, dir)
	}
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if !info.IsDir() {
		t.Fatal("EnsureAgentDir did not create a directory")
	}

	// It is idempotent.
	if _, err := EnsureAgentDir(); err != nil {
		t.Fatalf("second EnsureAgentDir: %v", err)
	}
}

// TestEnsureAgentDirPermissions checks the directory is user-only, since it
// holds credentials. Unix permission bits are not meaningful on Windows.
func TestEnsureAgentDirPermissions(t *testing.T) {
	if os.PathSeparator == '\\' {
		t.Skip("POSIX permission bits are not enforced on Windows")
	}
	dir := filepath.Join(t.TempDir(), "agent")
	t.Setenv(EnvAgentDir, dir)

	if _, err := EnsureAgentDir(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o700 {
		t.Fatalf("permissions = %o, want 700 (the directory holds credentials)", perm)
	}
}
