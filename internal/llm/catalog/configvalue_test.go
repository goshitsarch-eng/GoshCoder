package catalog

// Port of reference/pi/packages/coding-agent/test/resolve-config-value.test.ts.

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakeEnv builds a Getenv over a map, so tests never touch os.Environ.
func fakeEnv(values map[string]string) Getenv {
	return func(name string) string { return values[name] }
}

func TestResolveConfigValueLiteralsTemplatesEscapes(t *testing.T) {
	getenv := fakeEnv(map[string]string{
		"TEST_CONFIG_LEFT":  "left",
		"TEST_CONFIG_RIGHT": "right",
	})

	tests := []struct {
		config string
		want   string
	}{
		{"literal-key", "literal-key"},
		{"$TEST_CONFIG_LEFT", "left"},
		{"${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT", "left_right"},
		// "$$" escapes a literal "$", so no expansion happens.
		{"$$TEST_CONFIG_LEFT", "$TEST_CONFIG_LEFT"},
		// "$!" escapes a literal "!", which would otherwise mean "command".
		{"$!literal-$TEST_CONFIG_RIGHT", "!literal-right"},
	}
	for _, tt := range tests {
		got, ok := ResolveConfigValueOK(tt.config, nil, getenv)
		if !ok || got != tt.want {
			t.Errorf("ResolveConfigValueOK(%q) = %q, %v; want %q, true", tt.config, got, ok, tt.want)
		}
	}
}

func TestResolveConfigValueScopedEnvBeatsProcessEnv(t *testing.T) {
	getenv := fakeEnv(map[string]string{"TEST_CONFIG_SCOPED": "process"})
	scoped := map[string]string{"TEST_CONFIG_SCOPED": "credential"}
	if got := ResolveConfigValue("$TEST_CONFIG_SCOPED", scoped, getenv); got != "credential" {
		t.Fatalf("got %q, want %q", got, "credential")
	}
	// Falls back to the process environment when the scoped map lacks the var.
	if got := ResolveConfigValue("$TEST_CONFIG_SCOPED", nil, getenv); got != "process" {
		t.Fatalf("got %q, want %q", got, "process")
	}
}

func TestResolveConfigValueUnresolvedEnvVar(t *testing.T) {
	getenv := fakeEnv(nil)
	// An unresolvable env reference yields "not resolved", not empty string.
	if got, ok := ResolveConfigValueOK("$TEST_CONFIG_MISSING", nil, getenv); ok {
		t.Fatalf("got %q, %v; want unresolved", got, ok)
	}
	// A partially resolvable template fails as a whole.
	if _, ok := ResolveConfigValueOK("prefix-$TEST_CONFIG_MISSING", nil, getenv); ok {
		t.Fatal("partially resolvable template should not resolve")
	}
}

func TestConfigValueEnvVarNameHelpers(t *testing.T) {
	if name, ok := ConfigValueEnvVarName("$SOLO"); !ok || name != "SOLO" {
		t.Errorf("ConfigValueEnvVarName($SOLO) = %q, %v", name, ok)
	}
	// Not a single bare reference, so there is no one name.
	if _, ok := ConfigValueEnvVarName("a$SOLO"); ok {
		t.Error("ConfigValueEnvVarName(a$SOLO) should not report a name")
	}
	if _, ok := ConfigValueEnvVarName("!echo hi"); ok {
		t.Error("command values have no env var name")
	}

	names := ConfigValueEnvVarNames("${A}-$B-$A")
	if strings.Join(names, ",") != "A,B" {
		t.Errorf("ConfigValueEnvVarNames = %v, want [A B] (deduped, in order)", names)
	}

	getenv := fakeEnv(map[string]string{"A": "set"})
	missing := MissingConfigValueEnvVarNames("${A}-$B", nil, getenv)
	if strings.Join(missing, ",") != "B" {
		t.Errorf("missing = %v, want [B]", missing)
	}
	if IsConfigValueConfigured("${A}-$B", nil, getenv) {
		t.Error("value with a missing env var should not be configured")
	}
	if !IsConfigValueConfigured("${A}", nil, getenv) {
		t.Error("value with all env vars present should be configured")
	}
}

func TestIsCommandConfigValue(t *testing.T) {
	if !IsCommandConfigValue("!echo hi") {
		t.Error("!echo hi should be a command value")
	}
	// "$!" is an escape, so this is a literal, not a command.
	if IsCommandConfigValue("$!echo hi") {
		t.Error("$!echo hi should be a literal")
	}
	if IsCommandConfigValue("literal") {
		t.Error("literal should not be a command value")
	}
}

// requireShell skips command tests where no POSIX shell is available.
func requireShell(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" {
		if _, err := resolveShellConfig(""); err != nil {
			t.Skip("no bash shell available for command resolution")
		}
	}
}

func TestResolveConfigValueExecutesCommands(t *testing.T) {
	requireShell(t)
	ClearConfigValueCache()
	t.Cleanup(ClearConfigValueCache)

	tests := []struct {
		config string
		want   string
	}{
		{"!echo '  spaced-key  '", "spaced-key"},
		{"!printf 'line1\\nline2'", "line1\nline2"},
		{"!echo 'hello world' | tr ' ' '-'", "hello-world"},
	}
	for _, tt := range tests {
		got, ok := ResolveConfigValueOK(tt.config, nil, nil)
		if !ok || got != tt.want {
			t.Errorf("ResolveConfigValueOK(%q) = %q, %v; want %q, true", tt.config, got, ok, tt.want)
		}
	}
}

func TestResolveConfigValueCommandFailures(t *testing.T) {
	requireShell(t)
	ClearConfigValueCache()
	t.Cleanup(ClearConfigValueCache)

	// Non-zero exit, missing binary, and empty output all resolve to nothing.
	for _, config := range []string{"!exit 1", "!nonexistent-command-12345", "!printf ''"} {
		if got, ok := ResolveConfigValueOK(config, nil, nil); ok {
			t.Errorf("ResolveConfigValueOK(%q) = %q, true; want unresolved", config, got)
		}
	}
}

// shellPath returns path in a form a POSIX shell accepts on this platform.
func shellPath(path string) string {
	return strings.ReplaceAll(path, `\`, "/")
}

func TestResolveConfigValueCachesUntilCleared(t *testing.T) {
	requireShell(t)
	ClearConfigValueCache()
	t.Cleanup(ClearConfigValueCache)

	counter := filepath.Join(t.TempDir(), "counter")
	if err := os.WriteFile(counter, []byte("0"), 0o600); err != nil {
		t.Fatal(err)
	}
	escaped := shellPath(counter)
	readCount := func() string {
		content, err := os.ReadFile(counter)
		if err != nil {
			t.Fatal(err)
		}
		return strings.TrimSpace(string(content))
	}

	success := `!sh -c 'count=$(cat "` + escaped + `"); echo $((count + 1)) > "` + escaped + `"; echo value'`
	if got := ResolveConfigValue(success, nil, nil); got != "value" {
		t.Fatalf("first call = %q", got)
	}
	if got := ResolveConfigValue(success, nil, nil); got != "value" {
		t.Fatalf("second call = %q", got)
	}
	// Cached: the command ran once despite two resolutions.
	if got := readCount(); got != "1" {
		t.Fatalf("run count = %s, want 1", got)
	}

	ClearConfigValueCache()
	if got := ResolveConfigValue(success, nil, nil); got != "value" {
		t.Fatalf("after clear = %q", got)
	}
	if got := readCount(); got != "2" {
		t.Fatalf("run count = %s, want 2", got)
	}

	// Failures are cached too.
	failure := `!sh -c 'count=$(cat "` + escaped + `"); echo $((count + 1)) > "` + escaped + `"; exit 1'`
	for range 2 {
		if got, ok := ResolveConfigValueOK(failure, nil, nil); ok {
			t.Fatalf("failure resolved to %q", got)
		}
	}
	if got := readCount(); got != "3" {
		t.Fatalf("run count = %s, want 3", got)
	}
}

func TestResolveConfigValueDoesNotCacheEnvValues(t *testing.T) {
	value := "first"
	getenv := Getenv(func(string) string { return value })
	if got := ResolveConfigValue("$TEST_CONFIG_DYNAMIC", nil, getenv); got != "first" {
		t.Fatalf("got %q, want first", got)
	}
	value = "second"
	if got := ResolveConfigValue("$TEST_CONFIG_DYNAMIC", nil, getenv); got != "second" {
		t.Fatalf("got %q, want second", got)
	}
}

func TestResolveConfigValueUncachedRunsEveryCall(t *testing.T) {
	requireShell(t)
	ClearConfigValueCache()
	t.Cleanup(ClearConfigValueCache)

	counter := filepath.Join(t.TempDir(), "uncached-counter")
	if err := os.WriteFile(counter, []byte("0"), 0o600); err != nil {
		t.Fatal(err)
	}
	escaped := shellPath(counter)
	command := `!sh -c 'count=$(cat "` + escaped + `"); echo $((count + 1)) > "` + escaped + `"; echo value'`

	for range 2 {
		if got, ok := ResolveConfigValueUncached(command, nil, nil); !ok || got != "value" {
			t.Fatalf("got %q, %v", got, ok)
		}
	}
	content, err := os.ReadFile(counter)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(content)); got != "2" {
		t.Fatalf("run count = %s, want 2", got)
	}
}

func TestResolveConfigValueOrError(t *testing.T) {
	getenv := fakeEnv(map[string]string{"PRESENT": "yes"})

	if got, err := ResolveConfigValueOrError("$PRESENT", "api key", nil, getenv); err != nil || got != "yes" {
		t.Fatalf("got %q, %v", got, err)
	}

	_, err := ResolveConfigValueOrError("$MISSING_ONE", "api key", nil, getenv)
	if err == nil || !strings.Contains(err.Error(), "environment variable: MISSING_ONE") {
		t.Fatalf("single missing var error = %v", err)
	}

	_, err = ResolveConfigValueOrError("$MISSING_ONE-$MISSING_TWO", "api key", nil, getenv)
	if err == nil || !strings.Contains(err.Error(), "environment variables: MISSING_ONE, MISSING_TWO") {
		t.Fatalf("multiple missing vars error = %v", err)
	}

	requireShell(t)
	ClearConfigValueCache()
	t.Cleanup(ClearConfigValueCache)
	_, err = ResolveConfigValueOrError("!exit 1", "api key", nil, nil)
	if err == nil || !strings.Contains(err.Error(), "shell command: exit 1") {
		t.Fatalf("command error = %v", err)
	}
}

func TestResolveHeaders(t *testing.T) {
	getenv := fakeEnv(map[string]string{"TOKEN": "secret"})

	if got := ResolveHeaders(nil, nil, getenv); got != nil {
		t.Fatalf("nil headers = %#v, want nil", got)
	}

	// Unresolvable headers are dropped rather than failing the whole map.
	resolved := ResolveHeaders(map[string]string{
		"Authorization": "Bearer $TOKEN",
		"X-Dropped":     "$MISSING",
	}, nil, getenv)
	if len(resolved) != 1 || resolved["Authorization"] != "Bearer secret" {
		t.Fatalf("resolved = %#v", resolved)
	}

	// When nothing resolves the result is nil, not an empty map.
	if got := ResolveHeaders(map[string]string{"X": "$MISSING"}, nil, getenv); got != nil {
		t.Fatalf("all-unresolved headers = %#v, want nil", got)
	}

	resolved, err := ResolveHeadersOrError(map[string]string{"Authorization": "Bearer $TOKEN"}, "provider", nil, getenv)
	if err != nil || resolved["Authorization"] != "Bearer secret" {
		t.Fatalf("resolved = %#v, err = %v", resolved, err)
	}
	if _, err := ResolveHeadersOrError(map[string]string{"X": "$MISSING"}, "provider", nil, getenv); err == nil {
		t.Fatal("expected an error for an unresolvable header")
	}
}
