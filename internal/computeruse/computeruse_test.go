package computeruse

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func fakeExecutable(t *testing.T, dir, name string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestFindBinaryEnvOverride(t *testing.T) {
	dir := t.TempDir()
	binary := fakeExecutable(t, dir, "custom-binary")
	getenv := func(name string) string {
		if name == BinaryEnvVar {
			return binary
		}
		return ""
	}
	if got := FindBinary(getenv); got != binary {
		t.Errorf("FindBinary = %q, want %q", got, binary)
	}
	// A missing override path falls through.
	getenv = func(name string) string {
		if name == BinaryEnvVar {
			return filepath.Join(dir, "does-not-exist")
		}
		return ""
	}
	if got := FindBinary(getenv); got != "" {
		t.Errorf("missing override must not resolve, got %q", got)
	}
}

func TestFindBinaryPath(t *testing.T) {
	if runtime.GOOS == "windows" {
		// The server itself is Linux-only; PATH discovery of a mode-bit
		// executable is a POSIX concern.
		t.Skip("executable-bit discovery is POSIX-specific")
	}
	dir := t.TempDir()
	binary := fakeExecutable(t, dir, ServerName)
	getenv := func(name string) string {
		if name == "PATH" {
			return t.TempDir() + string(os.PathListSeparator) + dir
		}
		return ""
	}
	if got := FindBinary(getenv); got != binary {
		t.Errorf("FindBinary via PATH = %q, want %q", got, binary)
	}
	// A non-executable file is skipped.
	os.Chmod(binary, 0o644)
	if got := FindBinary(getenv); got != "" {
		t.Errorf("non-executable candidate must be skipped, got %q", got)
	}
}

func TestToolNamePrefixing(t *testing.T) {
	if got := PrefixedToolName("doctor"); got != "computer_use_linux_doctor" {
		t.Errorf("PrefixedToolName = %q", got)
	}
	if got := RawToolName("computer_use_linux_doctor"); got != "doctor" {
		t.Errorf("RawToolName prefixed = %q", got)
	}
	if got := RawToolName("doctor"); got != "doctor" {
		t.Errorf("RawToolName bare = %q", got)
	}
}

func TestEnsureServerEntryCreatesAndPreserves(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp.json")
	result, err := EnsureServerEntry(path, "/usr/bin/computer-use-linux")
	if err != nil || result != EnsureUpdated {
		t.Fatalf("create: %v %v", result, err)
	}
	// Second call with the same binary is a no-op.
	result, err = EnsureServerEntry(path, "/usr/bin/computer-use-linux")
	if err != nil || result != EnsureUnchanged {
		t.Fatalf("unchanged: %v %v", result, err)
	}
	// A moved binary updates the entry.
	result, err = EnsureServerEntry(path, "/opt/bin/computer-use-linux")
	if err != nil || result != EnsureUpdated {
		t.Fatalf("update: %v %v", result, err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err := json.Unmarshal(raw, &config); err != nil {
		t.Fatal(err)
	}
	servers := config["mcpServers"].(map[string]any)
	entry := servers[ServerName].(map[string]any)
	if entry["command"] != "/opt/bin/computer-use-linux" {
		t.Errorf("entry = %v", entry)
	}
	args := entry["args"].([]any)
	if len(args) != 1 || args[0] != "mcp" {
		t.Errorf("args = %v", args)
	}
}

func TestEnsureServerEntryPreservesOtherKeys(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mcp.json")
	existing := `{
		"someTopLevel": {"nested": [1, 2, 3]},
		"mcpServers": {
			"other-server": {"command": "/bin/other", "args": ["run"], "lifecycle": "eager"}
		}
	}`
	if err := os.WriteFile(path, []byte(existing), 0o600); err != nil {
		t.Fatal(err)
	}
	if result, err := EnsureServerEntry(path, "/usr/bin/computer-use-linux"); err != nil || result != EnsureUpdated {
		t.Fatalf("%v %v", result, err)
	}
	raw, _ := os.ReadFile(path)
	var config map[string]any
	if err := json.Unmarshal(raw, &config); err != nil {
		t.Fatal(err)
	}
	if _, ok := config["someTopLevel"].(map[string]any); !ok {
		t.Error("unrelated top-level keys must survive")
	}
	servers := config["mcpServers"].(map[string]any)
	other, ok := servers["other-server"].(map[string]any)
	if !ok || other["lifecycle"] != "eager" {
		t.Errorf("other server entries must survive verbatim: %v", servers)
	}
	if _, ok := servers[ServerName]; !ok {
		t.Error("our entry must be added")
	}
}

func TestEnsureServerEntryRefusesMalformed(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{
		"broken.json": `{not json`,
		"array.json":  `[1, 2]`,
	} {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		result, err := EnsureServerEntry(path, "/usr/bin/computer-use-linux")
		if result != EnsureFailed || err == nil {
			t.Errorf("%s: result=%v err=%v", name, result, err)
		}
		after, _ := os.ReadFile(path)
		if string(after) != content {
			t.Errorf("%s: malformed config was overwritten", name)
		}
	}
}

func TestFormatSchema(t *testing.T) {
	schema := json.RawMessage(`{"type": "object", "properties": {
		"window_id": {"type": "string", "description": "Target window"},
		"max_width": {"type": "number"}
	}, "required": ["window_id"]}`)
	text := formatSchema(schema)
	if !strings.Contains(text, "window_id (string, required): Target window") {
		t.Errorf("formatSchema: %s", text)
	}
	if !strings.Contains(text, "max_width (number, optional)") {
		t.Errorf("formatSchema: %s", text)
	}
	if got := formatSchema(nil); got != "(no parameters)" {
		t.Errorf("empty schema: %q", got)
	}
}
