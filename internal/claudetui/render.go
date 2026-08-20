// Package claudetui is a native adaptation of pi-claude-code-tui for
// GoshCoder's line-oriented and fullscreen terminal interfaces.
package claudetui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/charmbracelet/x/ansi"
)

const Version = "0.1.12-native"

// SessionInfo is the compact, OpenCode-inspired status shown on the right of
// the startup card and by /status. Values are prepared by the CLI so this
// rendering package stays independent of the agent and git implementations.
type ChangedFile struct {
	Path   string
	Status string
}

// SessionInfo contains the live local session metrics used by status views.
type SessionInfo struct {
	Name         string
	Model        string
	ContextUsed  int
	ContextLimit int
	Cost         float64
	Messages     int
	Tools        int
	ChangedFiles int
	Changes      []ChangedFile
	Branch       string
	Mode         string
	Thinking     string
}

func displayWidth(text string) int { return ansi.StringWidth(text) }

func stripANSI(text string) string {
	var out strings.Builder
	for i := 0; i < len(text); {
		if text[i] == 0x1b && i+1 < len(text) && text[i+1] == '[' {
			i += 2
			for i < len(text) && (text[i] < '@' || text[i] > '~') {
				i++
			}
			if i < len(text) {
				i++
			}
			continue
		}
		out.WriteByte(text[i])
		i++
	}
	return out.String()
}

func truncate(text string, width int) string {
	if width <= 0 {
		return ""
	}
	if displayWidth(text) <= width {
		return text
	}
	if width == 1 {
		return "…"
	}
	return ansi.Truncate(text, width, "…")
}

func pad(text string, width int) string {
	text = truncate(text, width)
	return text + strings.Repeat(" ", max(0, width-displayWidth(text)))
}

func center(text string, width int) string {
	text = truncate(text, width)
	return strings.Repeat(" ", max(0, (width-displayWidth(text))/2)) + text
}

func FormatCWD(cwd string) string {
	home, _ := os.UserHomeDir()
	if home != "" && (cwd == home || strings.HasPrefix(cwd, home+string(filepath.Separator))) {
		return "~" + strings.TrimPrefix(cwd, home)
	}
	return cwd
}

// Header renders the extension's Pi-inspired startup card. It is retained for
// callers that do not have live session information.
func Header(width int, appVersion, model, thinking, cwd string, color bool) []string {
	return HeaderWithInfo(width, appVersion, model, thinking, cwd, color, nil)
}

// HeaderWithInfo renders an OpenCode-inspired session sidebar on wide
// terminals. Narrow terminals retain the centered identity block.
func HeaderWithInfo(width int, appVersion, model, thinking, cwd string, color bool, info *SessionInfo) []string {
	if width < 24 {
		return []string{"GoshCoder v" + appVersion}
	}
	accent := func(s string) string {
		if color {
			return "\x1b[38;2;215;119;87m" + s + "\x1b[39m"
		}
		return s
	}
	inner := width - 2
	useTips := inner >= 55
	leftWidth, rightWidth := inner, 0
	if useTips {
		rightWidth = min(28, max(16, int(float64(inner)*0.28)))
		leftWidth = inner - rightWidth - 3
	}
	logo := []string{"  ██████  ", " ██  ███  ", "  ███  ██ ", "  ██   ██ "}
	left := []string{
		center(accent(logo[0]), leftWidth), center(accent(logo[1]), leftWidth),
		center(accent(logo[2]), leftWidth), center(accent(logo[3]), leftWidth),
		center("Let's build something great", leftWidth),
		center(model+" · "+thinking+" effort", leftWidth), center(FormatCWD(cwd), leftWidth), "",
	}
	tips := []string{"", "Getting started", "Ask GoshCoder to build it", "────────────────", "Commands", "/help", "/model", "/planner"}
	if info != nil {
		tips = infoLines(*info)
	}
	for len(tips) < len(left) {
		tips = append(tips, "")
	}
	label := " GoshCoder v" + appVersion + " "
	topFill := max(0, width-2-displayWidth(label)-3)
	lines := []string{accent("╭───") + label + accent(strings.Repeat("─", topFill)+"╮")}
	for i, value := range left {
		content := pad(value, leftWidth)
		if useTips {
			content += " " + accent("│") + " " + pad(tips[min(i, len(tips)-1)], rightWidth)
		}
		lines = append(lines, accent("│")+pad(content, inner)+accent("│"))
	}
	lines = append(lines, accent("╰"+strings.Repeat("─", width-2)+"╯"))
	return lines
}

// Sidebar renders the same information as a standalone card for /status.
func Sidebar(width int, info SessionInfo, color bool) []string {
	width = max(24, width)
	paint := func(s string) string {
		if color {
			return "\x1b[38;2;215;119;87m" + s + "\x1b[39m"
		}
		return s
	}
	inner := width - 2
	lines := []string{paint("╭" + strings.Repeat("─", inner) + "╮")}
	for _, line := range infoLines(info) {
		lines = append(lines, paint("│")+" "+pad(line, inner-2)+" "+paint("│"))
	}
	lines = append(lines, paint("╰"+strings.Repeat("─", inner)+"╯"))
	return lines
}

func infoLines(info SessionInfo) []string {
	contextValue := formatCount(info.ContextUsed)
	if info.ContextLimit > 0 {
		percent := min(100, info.ContextUsed*100/info.ContextLimit)
		contextValue += fmt.Sprintf("/%s · %d%%", formatCount(info.ContextLimit), percent)
	}
	branch := info.Branch
	if branch == "" {
		branch = "not a git repo"
	}
	mode := info.Mode
	if mode == "" {
		mode = "normal"
	}
	if info.Thinking != "" && info.Thinking != "off" {
		mode += " · " + info.Thinking
	}
	return []string{
		"Session", truncate(info.Model, 24),
		"Context  " + contextValue,
		fmt.Sprintf("Cost     $%.4f", info.Cost),
		fmt.Sprintf("Messages %d · Tools %d", info.Messages, info.Tools),
		fmt.Sprintf("Files    %d changed", info.ChangedFiles),
		"Branch   " + branch,
		"Mode     " + mode,
	}
}

func formatCount(value int) string {
	if value >= 1_000_000 {
		if value%1_000_000 == 0 {
			return fmt.Sprintf("%dM", value/1_000_000)
		}
		return fmt.Sprintf("%.1fM", float64(value)/1_000_000)
	}
	if value >= 1_000 {
		if value%1_000 == 0 {
			return fmt.Sprintf("%dk", value/1_000)
		}
		return fmt.Sprintf("%.1fk", float64(value)/1_000)
	}
	return fmt.Sprintf("%d", value)
}

func InputPrompt(width int, color bool) string {
	if width < 8 {
		return "> "
	}
	paint := func(s string) string {
		if color {
			return "\x1b[38;2;215;119;87m" + s + "\x1b[39m"
		}
		return s
	}
	return fmt.Sprintf("%s\n%s ", paint("╭"+strings.Repeat("─", width-2)+"╮"), paint("╰─❯"))
}
