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

	"goshcoder/internal/atomicfile"
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

// WebSearchPath returns the pi-web-access-compatible search configuration
// path used by the native web_search tool.
func WebSearchPath() string { return filepath.Join(AgentDir(), "web-search.json") }

// OmniRoutePath stores the native OmniRoute server and synchronized catalog.
// Credentials remain in auth.json rather than being duplicated in this file.
func OmniRoutePath() string { return filepath.Join(AgentDir(), "omniroute.json") }

// BTWPath stores the native pi-btw model and thinking preferences.
func BTWPath() string { return filepath.Join(AgentDir(), "pi-btw.json") }

// AperturePath stores the native Tailscale Aperture configuration. The file
// lives under extensions/ because that is where @aliou/pi-ts-aperture keeps
// it (~/.pi/agent/extensions/aperture.json), so an agent directory pointed at
// pi reads and writes the same file.
func AperturePath() string { return filepath.Join(AgentDir(), "extensions", "aperture.json") }

// ApertureCachePath stores the synchronized Aperture dedicated catalog and
// gateway snapshot. pi keeps the equivalent snapshot in its models store; the
// shape here is GoshCoder's own, so it uses a sibling file rather than
// overloading aperture.json (upstream migrated cached models out of the
// config file on purpose).
func ApertureCachePath() string {
	return filepath.Join(AgentDir(), "extensions", "aperture-cache.json")
}

// MCPConfigPath is the agent-level MCP server registry
// (pi-mcp-adapter-compatible mcp.json) that the native computer-use-linux
// adaptation maintains.
func MCPConfigPath() string { return filepath.Join(AgentDir(), "mcp.json") }

// SessionsDir returns the root holding persisted session logs. Sessions are
// sharded beneath it by a cwd-derived directory name; see internal/sessionlog.
func SessionsDir() string { return filepath.Join(AgentDir(), "sessions") }

// PromptsDir returns the user-scoped prompt template directory. A project may
// also carry its own under .goshcoder/prompts; see internal/resources.
func PromptsDir() string { return filepath.Join(AgentDir(), "prompts") }

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
	return atomicfile.Write(DefaultModelPath(), []byte(strings.TrimSpace(model)+"\n"), 0o600)
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
