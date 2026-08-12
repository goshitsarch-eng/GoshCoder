package catalog

// Port of reference/pi/packages/coding-agent/src/core/resolve-config-value.ts.
//
// A "config value" is a string that may be a literal, an environment-variable
// template ("$VAR" / "${VAR}"), or a shell command ("!cmd"). API keys, header
// values and provider env values all go through this resolution.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

// commandTimeout bounds shell command resolution (TS timeout: 10000).
const commandTimeout = 10 * time.Second
const commandOutputLimit = 1 << 20

type commandOutput struct {
	builder   strings.Builder
	truncated bool
}

func (output *commandOutput) Write(data []byte) (int, error) {
	originalLength := len(data)
	remaining := commandOutputLimit - output.builder.Len()
	if remaining <= 0 {
		output.truncated = output.truncated || originalLength > 0
		return originalLength, nil
	}
	if len(data) > remaining {
		data = data[:remaining]
		output.truncated = true
	}
	_, _ = output.builder.Write(data)
	return originalLength, nil
}

var (
	envVarNameRE       = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
	envVarNamePrefixRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*`)
)

// commandResultCache caches command results (including failures) for the
// process lifetime, mirroring the TS module-level Map.
var (
	commandCacheMu sync.Mutex
	commandCache   = map[string]cachedCommand{}
)

type cachedCommand struct {
	value string
	ok    bool
}

// ClearConfigValueCache drops cached shell command results.
func ClearConfigValueCache() {
	commandCacheMu.Lock()
	defer commandCacheMu.Unlock()
	commandCache = map[string]cachedCommand{}
}

type templatePart struct {
	// isEnv distinguishes an env-var reference from a literal chunk.
	isEnv bool
	value string // literal text, or env var name when isEnv
}

type configValueRef struct {
	isCommand bool
	config    string // full config incl. leading "!" when isCommand
	parts     []templatePart
}

func appendLiteral(parts []templatePart, value string) []templatePart {
	if value == "" {
		return parts
	}
	if n := len(parts); n > 0 && !parts[n-1].isEnv {
		parts[n-1].value += value
		return parts
	}
	return append(parts, templatePart{value: value})
}

func parseConfigValueTemplate(config string) []templatePart {
	var parts []templatePart
	index := 0
	for index < len(config) {
		dollar := strings.Index(config[index:], "$")
		if dollar < 0 {
			parts = appendLiteral(parts, config[index:])
			break
		}
		dollar += index
		parts = appendLiteral(parts, config[index:dollar])

		var next byte
		if dollar+1 < len(config) {
			next = config[dollar+1]
		}

		// "$$" and "$!" escape a literal "$" / "!".
		if next == '$' || next == '!' {
			parts = appendLiteral(parts, string(next))
			index = dollar + 2
			continue
		}

		if next == '{' {
			end := strings.Index(config[dollar+2:], "}")
			if end < 0 {
				parts = appendLiteral(parts, "$")
				index = dollar + 1
				continue
			}
			end += dollar + 2
			name := config[dollar+2 : end]
			if envVarNameRE.MatchString(name) {
				parts = append(parts, templatePart{isEnv: true, value: name})
			} else {
				parts = appendLiteral(parts, config[dollar:end+1])
			}
			index = end + 1
			continue
		}

		if match := envVarNamePrefixRE.FindString(config[dollar+1:]); match != "" {
			parts = append(parts, templatePart{isEnv: true, value: match})
			index = dollar + 1 + len(match)
			continue
		}

		parts = appendLiteral(parts, "$")
		index = dollar + 1
	}
	return parts
}

func parseConfigValueReference(config string) configValueRef {
	if strings.HasPrefix(config, "!") {
		return configValueRef{isCommand: true, config: config}
	}
	return configValueRef{parts: parseConfigValueTemplate(config)}
}

// Getenv is the process-environment lookup used during config value
// resolution. Pass nil to use os.Getenv; tests inject a fake.
type Getenv func(string) string

func (g Getenv) orDefault() Getenv {
	if g == nil {
		return os.Getenv
	}
	return g
}

// resolveEnvConfigValue mirrors `env?.[name] || process.env[name] ||
// undefined`: empty values fall through to the next source.
func resolveEnvConfigValue(name string, env map[string]string, getenv Getenv) (string, bool) {
	if v := env[name]; v != "" {
		return v, true
	}
	if v := getenv.orDefault()(name); v != "" {
		return v, true
	}
	return "", false
}

func templateEnvVarNames(parts []templatePart) []string {
	var names []string
	for _, part := range parts {
		if !part.isEnv {
			continue
		}
		if !slicesContains(names, part.value) {
			names = append(names, part.value)
		}
	}
	return names
}

func slicesContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func resolveTemplate(parts []templatePart, env map[string]string, getenv Getenv) (string, bool) {
	var sb strings.Builder
	for _, part := range parts {
		if !part.isEnv {
			sb.WriteString(part.value)
			continue
		}
		value, ok := resolveEnvConfigValue(part.value, env, getenv)
		if !ok {
			return "", false
		}
		sb.WriteString(value)
	}
	return sb.String(), true
}

// ConfigValueEnvVarName returns the env var name when config is exactly one
// env reference (TS getConfigValueEnvVarName).
func ConfigValueEnvVarName(config string) (string, bool) {
	ref := parseConfigValueReference(config)
	if ref.isCommand {
		return "", false
	}
	if len(ref.parts) == 1 && ref.parts[0].isEnv {
		return ref.parts[0].value, true
	}
	return "", false
}

// ConfigValueEnvVarNames returns every env var referenced by config.
func ConfigValueEnvVarNames(config string) []string {
	ref := parseConfigValueReference(config)
	if ref.isCommand {
		return nil
	}
	return templateEnvVarNames(ref.parts)
}

// MissingConfigValueEnvVarNames returns referenced env vars that resolve to
// nothing.
func MissingConfigValueEnvVarNames(config string, env map[string]string, getenv Getenv) []string {
	var missing []string
	for _, name := range ConfigValueEnvVarNames(config) {
		if _, ok := resolveEnvConfigValue(name, env, getenv); !ok {
			missing = append(missing, name)
		}
	}
	return missing
}

// IsCommandConfigValue reports whether config is a "!command" value.
func IsCommandConfigValue(config string) bool {
	return parseConfigValueReference(config).isCommand
}

// IsConfigValueConfigured reports whether every referenced env var resolves.
// Command values are always considered configured (as in pi).
func IsConfigValueConfigured(config string, env map[string]string, getenv Getenv) bool {
	return len(MissingConfigValueEnvVarNames(config, env, getenv)) == 0
}

// ResolveConfigValue resolves a config value, returning "" when it cannot be
// resolved (the TS `undefined`). Command results are cached per process.
func ResolveConfigValue(config string, env map[string]string, getenv Getenv) string {
	value, _ := ResolveConfigValueOK(config, env, getenv)
	return value
}

// ResolveConfigValueOK is ResolveConfigValue with an explicit resolved flag,
// so callers can tell "unresolved" from "resolved to empty".
func ResolveConfigValueOK(config string, env map[string]string, getenv Getenv) (string, bool) {
	ref := parseConfigValueReference(config)
	if ref.isCommand {
		return executeCommand(ref.config)
	}
	return resolveTemplate(ref.parts, env, getenv)
}

// ResolveConfigValueUncached is ResolveConfigValueOK without the command
// cache: a "!command" value runs on every call.
func ResolveConfigValueUncached(config string, env map[string]string, getenv Getenv) (string, bool) {
	ref := parseConfigValueReference(config)
	if ref.isCommand {
		return executeCommandUncached(ref.config)
	}
	return resolveTemplate(ref.parts, env, getenv)
}

// ResolveConfigValueOrError resolves uncached and reports a descriptive error
// naming the failing command or env vars (TS resolveConfigValueOrThrow).
func ResolveConfigValueOrError(config, description string, env map[string]string, getenv Getenv) (string, error) {
	if value, ok := ResolveConfigValueUncached(config, env, getenv); ok {
		return value, nil
	}
	ref := parseConfigValueReference(config)
	if ref.isCommand {
		return "", fmt.Errorf("failed to resolve %s from shell command: %s", description, strings.TrimPrefix(ref.config, "!"))
	}
	missing := MissingConfigValueEnvVarNames(config, env, getenv)
	switch len(missing) {
	case 0:
	case 1:
		return "", fmt.Errorf("failed to resolve %s from environment variable: %s", description, missing[0])
	default:
		return "", fmt.Errorf("failed to resolve %s from environment variables: %s", description, strings.Join(missing, ", "))
	}
	return "", fmt.Errorf("failed to resolve %s", description)
}

// ResolveHeaders resolves every header value, dropping the ones that resolve
// to nothing. Returns nil when no header resolves.
func ResolveHeaders(headers map[string]string, env map[string]string, getenv Getenv) map[string]string {
	if headers == nil {
		return nil
	}
	resolved := map[string]string{}
	for key, value := range headers {
		if v, ok := ResolveConfigValueOK(value, env, getenv); ok && v != "" {
			resolved[key] = v
		}
	}
	if len(resolved) == 0 {
		return nil
	}
	return resolved
}

// ResolveHeadersOrError resolves every header value, failing on the first
// header that cannot be resolved.
func ResolveHeadersOrError(headers map[string]string, description string, env map[string]string, getenv Getenv) (map[string]string, error) {
	if headers == nil {
		return nil, nil
	}
	resolved := map[string]string{}
	for key, value := range headers {
		v, err := ResolveConfigValueOrError(value, fmt.Sprintf("%s header %q", description, key), env, getenv)
		if err != nil {
			return nil, err
		}
		resolved[key] = v
	}
	if len(resolved) == 0 {
		return nil, nil
	}
	return resolved, nil
}

func executeCommand(commandConfig string) (string, bool) {
	commandCacheMu.Lock()
	if cached, hit := commandCache[commandConfig]; hit {
		commandCacheMu.Unlock()
		return cached.value, cached.ok
	}
	commandCacheMu.Unlock()

	value, ok := executeCommandUncached(commandConfig)

	commandCacheMu.Lock()
	commandCache[commandConfig] = cachedCommand{value: value, ok: ok}
	commandCacheMu.Unlock()
	return value, ok
}

// executeCommandUncached runs the command after the leading "!". On Windows pi
// prefers a configured bash and falls back to the default shell; elsewhere it
// uses the default shell directly.
func executeCommandUncached(commandConfig string) (string, bool) {
	command := strings.TrimPrefix(commandConfig, "!")
	if runtime.GOOS == "windows" {
		if executed, value, ok := executeWithConfiguredShell(command); executed {
			return value, ok
		}
	}
	return executeWithDefaultShell(command)
}

// executeWithConfiguredShell runs command through the resolved bash shell.
// executed is false when no shell could be found/spawned, so the caller can
// fall back.
func executeWithConfiguredShell(command string) (executed bool, value string, ok bool) {
	cfg, err := resolveShellConfig("")
	if err != nil {
		return false, "", false
	}
	args := cfg.args
	stdin := ""
	if cfg.commandViaStdin {
		stdin = command
	} else {
		args = append(append([]string(nil), cfg.args...), command)
	}
	out, runErr, spawned := runShell(cfg.shell, args, stdin)
	if !spawned {
		return false, "", false
	}
	if runErr != nil {
		return true, "", false
	}
	out = strings.TrimSpace(out)
	return true, out, out != ""
}

func executeWithDefaultShell(command string) (string, bool) {
	shell, args := "sh", []string{"-c"}
	if runtime.GOOS == "windows" {
		shell, args = "cmd.exe", []string{"/d", "/s", "/c"}
	}
	out, runErr, spawned := runShell(shell, append(args, command), "")
	if !spawned || runErr != nil {
		return "", false
	}
	out = strings.TrimSpace(out)
	return out, out != ""
}

// runShell executes shell with args, optionally feeding stdin, and returns
// stdout. spawned is false when the executable could not be started.
func runShell(shell string, args []string, stdin string) (out string, runErr error, spawned bool) {
	ctx, cancel := context.WithTimeout(context.Background(), commandTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, shell, args...)
	cmd.WaitDelay = 2 * time.Second
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var stdout commandOutput
	cmd.Stdout = &stdout
	if err := cmd.Start(); err != nil {
		return "", err, false
	}
	err := cmd.Wait()
	if ctx.Err() != nil {
		return "", fmt.Errorf("command timed out after %s", commandTimeout), true
	}
	if stdout.truncated {
		return "", errors.New("command output exceeds 1 MiB"), true
	}
	return stdout.builder.String(), err, true
}

// shellConfig is the resolved command shell (TS ShellConfig).
type shellConfig struct {
	shell string
	args  []string
	// commandViaStdin is set for legacy WSL bash, which needs "-s" + stdin.
	commandViaStdin bool
}

var legacyWSLBashRE = regexp.MustCompile(`(?i)^[a-z]:\\windows\\(system32|sysnative)\\bash\.exe$`)

func bashShellConfig(shell string) shellConfig {
	normalized := strings.ReplaceAll(shell, "/", `\`)
	if legacyWSLBashRE.MatchString(normalized) {
		return shellConfig{shell: shell, args: []string{"-s"}, commandViaStdin: true}
	}
	return shellConfig{shell: shell, args: []string{"-c"}}
}

// resolveShellConfig mirrors pi's getShellConfig: explicit path, then Git Bash
// (Windows) or /bin/bash (Unix), then bash on PATH, then sh.
func resolveShellConfig(customShellPath string) (shellConfig, error) {
	if customShellPath != "" {
		if fileExists(customShellPath) {
			return bashShellConfig(customShellPath), nil
		}
		return shellConfig{}, fmt.Errorf("custom shell path not found: %s", customShellPath)
	}

	if runtime.GOOS == "windows" {
		var candidates []string
		if programFiles := os.Getenv("ProgramFiles"); programFiles != "" {
			candidates = append(candidates, filepath.Join(programFiles, "Git", "bin", "bash.exe"))
		}
		if programFilesX86 := os.Getenv("ProgramFiles(x86)"); programFilesX86 != "" {
			candidates = append(candidates, filepath.Join(programFilesX86, "Git", "bin", "bash.exe"))
		}
		for _, candidate := range candidates {
			if fileExists(candidate) {
				return bashShellConfig(candidate), nil
			}
		}
		if bash, err := exec.LookPath("bash.exe"); err == nil && fileExists(bash) {
			return bashShellConfig(bash), nil
		}
		return shellConfig{}, fmt.Errorf("no bash shell found; searched: %s", strings.Join(candidates, ", "))
	}

	if fileExists("/bin/bash") {
		return bashShellConfig("/bin/bash"), nil
	}
	if bash, err := exec.LookPath("bash"); err == nil {
		return bashShellConfig(bash), nil
	}
	return shellConfig{shell: "sh", args: []string{"-c"}}, nil
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}
