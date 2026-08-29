// Package computeruse is GoshCoder's native Go adaptation of
// @agent-sh/computer-use-linux (github.com/agent-sh/computer-use-linux,
// version 0.4.10), the Linux desktop-control MCP server: AT-SPI
// accessibility trees, Wayland/X11 input, screenshots, and compositor window
// targeting.
//
// The npm package ships a Rust MCP server binary plus a pi extension
// (pi/extension/index.ts) that discovers the binary and registers it in
// ~/.pi/agent/mcp.json for pi-mcp-adapter's mcp() proxy tool. GoshCoder has
// no plugin host or separate MCP adapter, so this package carries the whole
// chain natively: the same binary discovery order, the same mcp.json
// maintenance (pi-compatible, so a shared agent directory serves both
// tools), a stdio MCP client speaking the rmcp 2024-11-05 protocol, and an
// mcp proxy tool matching the documented pi usage
// (mcp({server}), mcp({search}), mcp({tool})) scoped to this server.
// Screenshots come back as inline images the model can see.
package computeruse

import (
	"os"
	"path/filepath"
	"strings"
)

// PackageName names the upstream package in user-facing messages, matching
// the extension's notification prefixes.
const PackageName = "@agent-sh/computer-use-linux"

// ServerName is the MCP server key in mcp.json and the server id the proxy
// tool exposes.
const ServerName = "computer-use-linux"

// BinaryEnvVar overrides binary discovery (pi/extension/index.ts findBinary).
const BinaryEnvVar = "COMPUTER_USE_LINUX_BIN"

// FindBinary locates the computer-use-linux binary:
//
//  1. the COMPUTER_USE_LINUX_BIN environment variable,
//  2. PATH,
//  3. ~/.local/bin (where install.sh and the prebuilt-binary instructions
//     place it).
//
// The original's second step probes the npm package's own bundled
// npm/bin/ directory relative to the extension file; a native binary has no
// package directory, so that step has no equivalent here — an npm global
// install still resolves through PATH.
func FindBinary(getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if fromEnv := getenv(BinaryEnvVar); fromEnv != "" {
		if info, err := os.Stat(fromEnv); err == nil && info.Mode().IsRegular() {
			return fromEnv
		}
	}
	for _, dir := range filepath.SplitList(getenv("PATH")) {
		if dir == "" {
			continue
		}
		candidate := filepath.Join(dir, ServerName)
		if isExecutable(candidate) {
			return candidate
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidate := filepath.Join(home, ".local", "bin", ServerName)
		if isExecutable(candidate) {
			return candidate
		}
	}
	return ""
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return false
	}
	return info.Mode().Perm()&0o111 != 0
}

// InstallHint is the guidance shown when the binary is missing, mirroring
// the extension's warning notification.
const InstallHint = "Install it with 'npm install -g @agent-sh/computer-use-linux' " +
	"or 'cargo install computer-use-linux', or set " + BinaryEnvVar + "."

// PrefixedToolName maps a raw MCP tool name to the server-prefixed form the
// proxy tool exposes (pi-mcp-adapter's computer_use_linux_<tool> naming).
func PrefixedToolName(raw string) string {
	return strings.ReplaceAll(ServerName, "-", "_") + "_" + raw
}

// RawToolName accepts either a prefixed or a raw tool name and returns the
// raw MCP name.
func RawToolName(name string) string {
	prefix := strings.ReplaceAll(ServerName, "-", "_") + "_"
	return strings.TrimPrefix(name, prefix)
}
