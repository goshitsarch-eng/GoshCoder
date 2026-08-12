// Package config resolves GoshCoder's on-disk locations.
//
// Port of the path helpers in reference/pi/packages/coding-agent/src/config.ts.
// pi stores state under ~/.pi/agent; GoshCoder uses ~/.goshcoder/agent so the
// two can coexist on one machine. File formats (auth.json, models.json) stay
// identical to pi's, so a directory can be pointed at either tool.
package config

import (
	"io"
	"os"
	"path/filepath"
	"strings"
)

// DirName is the per-user configuration directory name.
const DirName = ".goshcoder"

// EnvAgentDir overrides the agent directory wholesale.
const EnvAgentDir = "GOSHCODER_AGENT_DIR"

// ExpandTilde expands a leading "~" to the user's home directory.
func ExpandTilde(path string) string {
	if path != "~" && !strings.HasPrefix(path, "~/") && !strings.HasPrefix(path, `~\`) {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	if path == "~" {
		return home
	}
	return filepath.Join(home, path[2:])
}

// AgentDir returns the agent configuration directory, honoring
// GOSHCODER_AGENT_DIR. It falls back to the working directory when the home
// directory cannot be determined.
func AgentDir() string {
	if dir := os.Getenv(EnvAgentDir); dir != "" {
		return ExpandTilde(dir)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(DirName, "agent")
	}
	return filepath.Join(home, DirName, "agent")
}

// AuthPath returns the credential store path (pi-compatible auth.json).
func AuthPath() string {
	return filepath.Join(AgentDir(), "auth.json")
}

// DefaultModelPath is the small text file used to remember the last model
// selected for interactive chat.
func DefaultModelPath() string { return filepath.Join(AgentDir(), "default-model") }

// ReadDefaultModel returns the remembered model reference, or an empty string
// when no model has been selected yet.
func ReadDefaultModel() string {
	file, err := os.Open(DefaultModelPath())
	if err != nil {
		return ""
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, 4097))
	if err != nil || len(content) > 4096 {
		return ""
	}
	return strings.TrimSpace(string(content))
}

// WriteDefaultModel atomically remembers the model used by interactive chat.
func WriteDefaultModel(model string) error {
	directory, err := EnsureAgentDir()
	if err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".default-model-*.tmp")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if _, err := temporary.WriteString(strings.TrimSpace(model) + "\n"); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(name, DefaultModelPath())
}

// EnsureAgentDir creates the agent directory if it does not exist. The
// directory is user-only (0700) because it holds credentials.
func EnsureAgentDir() (string, error) {
	dir := AgentDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}
