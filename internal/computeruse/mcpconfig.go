package computeruse

// Agent-level mcp.json maintenance (pi/extension/index.ts): the extension's
// core feature is keeping the computer-use-linux server entry registered in
// the pi-mcp-adapter-compatible config so any MCP host reading it can spawn
// the server. The file is preserved verbatim apart from the one entry, and a
// file that exists but cannot be parsed is never overwritten.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const maxMCPConfigBytes = 4 << 20

// EnsureResult reports what EnsureServerEntry did.
type EnsureResult string

const (
	// EnsureUpdated means the entry was written or updated.
	EnsureUpdated EnsureResult = "updated"
	// EnsureUnchanged means the entry already matched; nothing was written.
	EnsureUnchanged EnsureResult = "unchanged"
	// EnsureFailed means the config could not be read or written; the
	// returned error says why.
	EnsureFailed EnsureResult = "failed"
)

// EnsureServerEntry writes or updates the computer-use-linux server entry
// ({command: binaryPath, args: ["mcp"]}) in the MCP config at configPath,
// preserving every other key in the file. A missing file is created; a
// malformed or non-object file is left alone and reported as failed so a
// broken hand-edited config is never clobbered.
func EnsureServerEntry(configPath, binaryPath string) (EnsureResult, error) {
	config, err := readMCPConfig(configPath)
	if err != nil {
		return EnsureFailed, err
	}

	servers, _ := config["mcpServers"].(map[string]any)
	if servers == nil {
		servers = map[string]any{}
	}

	if existing, ok := servers[ServerName].(map[string]any); ok {
		command, _ := existing["command"].(string)
		if args, ok := existing["args"].([]any); ok && command == binaryPath && len(args) == 1 {
			if arg, ok := args[0].(string); ok && arg == "mcp" {
				return EnsureUnchanged, nil
			}
		}
	}

	servers[ServerName] = map[string]any{
		"command": binaryPath,
		"args":    []any{"mcp"},
	}
	config["mcpServers"] = servers

	if err := writeMCPConfig(configPath, config); err != nil {
		return EnsureFailed, err
	}
	return EnsureUpdated, nil
}

func readMCPConfig(path string) (map[string]any, error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Only a genuinely missing file starts empty.
			return map[string]any{}, nil
		}
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("MCP config is not a regular file")
	}
	if info.Size() > maxMCPConfigBytes {
		return nil, fmt.Errorf("MCP config exceeds %d bytes", maxMCPConfigBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxMCPConfigBytes+1))
	if err != nil {
		return nil, err
	}
	var parsed any
	if err := json.Unmarshal(data, &parsed); err != nil {
		return nil, fmt.Errorf("MCP config at %s is not valid JSON; refusing to overwrite it: %w", path, err)
	}
	object, ok := parsed.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("MCP config at %s contains a %T, expected a JSON object; refusing to overwrite it", path, parsed)
	}
	return object, nil
}

func writeMCPConfig(path string, config map[string]any) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".mcp-*.tmp")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}
