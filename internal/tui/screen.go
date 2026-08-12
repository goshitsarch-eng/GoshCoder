// Package tui renders GoshCoder's Bubble Tea fullscreen interface.
package tui

import (
	"os"
	"strings"
	"unicode"

	"github.com/charmbracelet/lipgloss"
	charmterm "github.com/charmbracelet/x/term"
)

const accent = "\x1b[38;2;139;124;246m"
const cyan = "\x1b[38;2;69;214;181m"
const blue = "\x1b[38;2;88;166;255m"
const amber = "\x1b[38;2;245;184;91m"
const red = "\x1b[38;2;244;112;122m"
const muted = "\x1b[38;2;112;118;145m"
const selected = "\x1b[48;2;61;57;92m\x1b[38;2;242;240;255m"
const bold = "\x1b[1m"
const reset = "\x1b[0m"
const dim = "\x1b[2m"

// Message is one transcript entry rendered in the main viewport.
type Message struct {
	Role string
	Text string
}

// Frame is a complete immutable screen snapshot.
type Frame struct {
	Title              string
	Messages           []Message
	Sidebar            []string
	Input              []rune
	Cursor             int
	Status             string
	Streaming          bool
	Scroll             int
	Suggestions        []string
	SelectedSuggestion int
	SuggestionTitle    string
	Thinking           string
	Mode               string
	VirtualCursor      bool
}

// Supported reports whether Bubble Tea can safely use both files as a TTY.
func Supported(in, out *os.File) bool {
	return in != nil && out != nil && charmterm.IsTerminal(in.Fd()) && charmterm.IsTerminal(out.Fd())
}

// View renders a complete frame for Bubble Tea. The cursor is painted in the
// composer so the view remains stable across terminal implementations.
func View(frame Frame, width, height int) string {
	if width < 20 || height < 8 {
		return accent + bold + "GoshCoder" + reset + "\n" + muted + "Terminal is too small" + reset
	}
	lines, _, _ := renderFrame(frame, width, height)
	return strings.Join(lines, "\n")
}

func renderFrame(frame Frame, width, height int) ([]string, int, int) {
	useSidebar := width >= 96
	sideWidth := 0
	if useSidebar {
		sideWidth = min(34, width/3)
	}
	mainWidth := width
	if useSidebar {
		mainWidth = width - sideWidth - 1
	}
	bodyTop, bodyBottom := 2, height-4
	bodyHeight := max(1, bodyBottom-bodyTop+1)

	suggestionRows := min(len(frame.Suggestions), min(7, max(0, bodyHeight-1)))
	paletteHeight := 0
	if suggestionRows > 0 {
		paletteHeight = suggestionRows + 1
	}
	transcriptHeight := max(0, bodyHeight-paletteHeight)
	transcript := renderMessages(frame.Messages, mainWidth)
	start := max(0, len(transcript)-transcriptHeight-frame.Scroll)
	end := min(len(transcript), start+transcriptHeight)
	visible := transcript[start:end]

	lines := make([]string, height)
	title := frame.Title
	if title == "" {
		title = "interactive session"
	}
	headerLeft := accent + bold + "◆  GOSH CODER" + reset + muted + "  " + safeTerminalText(title) + reset
	mode := frame.Mode
	if mode == "" {
		mode = "normal"
	}
	thinking := frame.Thinking
	if thinking == "" {
		thinking = "off"
	}
	headerRight := cyan + "● " + reset + mode + muted + "  ·  " + reset + accent + thinking + " thinking" + reset
	gap := width - lipgloss.Width(headerLeft) - lipgloss.Width(headerRight)
	if gap >= 2 {
		lines[0] = headerLeft + strings.Repeat(" ", gap) + headerRight
	} else {
		lines[0] = accent + bold + truncate(stripANSI(headerLeft), width) + reset
	}
	lines[1] = muted + strings.Repeat("─", width) + reset

	suggestionOffset := max(0, frame.SelectedSuggestion-suggestionRows+1)
	paletteStart := bodyHeight - paletteHeight
	for row := 0; row < bodyHeight; row++ {
		left := ""
		switch {
		case paletteHeight > 0 && row == paletteStart:
			title := frame.SuggestionTitle
			if title == "" {
				title = "COMMAND PALETTE"
			}
			left = muted + bold + "  " + truncate(title, max(1, mainWidth-2)) + reset
		case paletteHeight > 0 && row > paletteStart:
			index := suggestionOffset + row - paletteStart - 1
			if index < len(frame.Suggestions) {
				parts := strings.SplitN(frame.Suggestions[index], "\t", 2)
				label, description := parts[0], ""
				if len(parts) == 2 {
					description = parts[1]
				}
				if index == frame.SelectedSuggestion {
					left = selected + "  › " + bold + label + reset + muted + "  " + description + reset
				} else {
					left = "    " + accent + label + reset + muted + "  " + description + reset
				}
			}
		case row < len(visible):
			left = visible[row]
		}
		left = pad(left, mainWidth)
		if useSidebar {
			right := ""
			if row < len(frame.Sidebar) {
				right = styleSidebarLine(frame.Sidebar[row])
			}
			lines[bodyTop+row] = left + muted + "│" + reset + pad(right, sideWidth)
		} else {
			lines[bodyTop+row] = left
		}
	}

	status := frame.Status
	hints := "Shift-Tab thinking  ·  PgUp scroll  ·  Ctrl-C exit"
	if frame.Streaming {
		hints = "Esc abort  ·  type to steer"
	}
	leftStatus := cyan + "● " + reset + safeTerminalText(status)
	rightStatus := muted + hints + reset
	statusGap := width - lipgloss.Width(leftStatus) - lipgloss.Width(rightStatus)
	if statusGap >= 2 {
		lines[height-3] = leftStatus + strings.Repeat(" ", statusGap) + rightStatus
	} else {
		lines[height-3] = truncate(status, width)
	}
	composerTitle := " Message "
	lines[height-2] = accent + "╭─" + composerTitle + strings.Repeat("─", max(0, width-lipgloss.Width(composerTitle)-3)) + "╮" + reset

	promptWidth := max(1, width-7)
	cursor := min(max(frame.Cursor, 0), len(frame.Input))
	beforeWidth := runeSliceWidth(frame.Input[:cursor])
	windowStart := 0
	for windowStart < cursor && beforeWidth-runeSliceWidth(frame.Input[:windowStart]) >= promptWidth {
		windowStart++
	}
	shown := truncateRunes(frame.Input[windowStart:], promptWidth)
	editorText := string(shown)
	if len(frame.Input) == 0 {
		placeholder := truncate("Ask anything, or type / for commands", max(1, promptWidth-2))
		if frame.VirtualCursor {
			editorText = "\x1b[7m \x1b[27m" + muted + " " + placeholder + reset
		} else {
			editorText = muted + placeholder + reset
		}
	} else if frame.VirtualCursor {
		relative := cursor - windowStart
		if relative < len(shown) {
			editorText = string(shown[:relative]) + "\x1b[7m" + string(shown[relative:relative+1]) + "\x1b[27m" + string(shown[relative+1:])
		} else {
			editorText += "\x1b[7m \x1b[27m"
		}
	}
	lines[height-1] = accent + "│ " + reset + bold + "❯ " + reset + editorText
	cursorCol := 5 + runeSliceWidth(frame.Input[windowStart:cursor])
	return lines, height, min(width, cursorCol)
}

func renderMessages(messages []Message, width int) []string {
	var lines []string
	for _, message := range messages {
		if strings.TrimSpace(message.Text) == "" {
			continue
		}
		label, color := strings.ToUpper(message.Role), accent
		icon := "•"
		switch message.Role {
		case "user":
			label, color, icon = "YOU", blue, "›"
		case "assistant":
			label, color, icon = "GOSH", accent, "✦"
		case "tool":
			label, color, icon = "TOOL", cyan, "⚙"
		case "thinking":
			label, color, icon = "THINKING", amber, "◌"
		case "Error":
			label, color, icon = "ERROR", red, "!"
		case "Notice":
			label, color, icon = "NOTICE", cyan, "i"
		case "Command":
			label, color, icon = "COMMAND", cyan, "⌘"
		}
		lines = append(lines, color+bold+icon+"  "+label+reset)
		for _, paragraph := range strings.Split(strings.ReplaceAll(message.Text, "\t", "    "), "\n") {
			wrapped := wrap(paragraph, max(1, width-4))
			for _, line := range wrapped {
				if message.Role == "thinking" {
					lines = append(lines, muted+"│  "+line+reset)
				} else {
					lines = append(lines, "   "+line)
				}
			}
		}
		lines = append(lines, "")
	}
	return lines
}

func styleSidebarLine(line string) string {
	plain := safeTerminalText(line)
	trimmed := strings.TrimSpace(plain)
	if trimmed == "SESSION" {
		return accent + bold + "  ◆ SESSION" + reset
	}
	if strings.HasPrefix(plain, " ") && plain == " "+trimmed && (trimmed == "Model" || trimmed == "Context" || trimmed == "Branch" || trimmed == "Mode") {
		return muted + strings.ToUpper(plain) + reset
	}
	if strings.Contains(plain, "changed") || strings.Contains(plain, "Cost") || strings.Contains(plain, "Messages") || strings.Contains(plain, "Tools") {
		return cyan + plain + reset
	}
	return plain
}

func wrap(text string, width int) []string {
	text = safeTerminalText(text)
	if text == "" {
		return []string{""}
	}
	var lines []string
	remaining := []rune(text)
	for len(remaining) > 0 {
		count, cells := 0, 0
		for count < len(remaining) {
			next := runeWidth(remaining[count])
			if cells+next > width {
				break
			}
			cells += next
			count++
		}
		if count == 0 {
			count = 1
		}
		lines = append(lines, string(remaining[:count]))
		remaining = remaining[count:]
	}
	return lines
}

func truncate(text string, width int) string {
	return string(truncateRunes([]rune(safeTerminalText(stripANSI(text))), width))
}
func pad(text string, width int) string {
	plainWidth := lipgloss.Width(text)
	if plainWidth > width {
		return truncate(text, width)
	}
	return text + strings.Repeat(" ", width-plainWidth)
}
func truncateRunes(input []rune, width int) []rune {
	cells, end := 0, 0
	for end < len(input) && cells+runeWidth(input[end]) <= width {
		cells += runeWidth(input[end])
		end++
	}
	return input[:end]
}
func runeSliceWidth(input []rune) int {
	total := 0
	for _, r := range input {
		total += runeWidth(r)
	}
	return total
}
func runeWidth(r rune) int {
	if r == 0 || unicode.Is(unicode.Mn, r) {
		return 0
	}
	if r >= 0x1100 && (r <= 0x115f || r == 0x2329 || r == 0x232a || r >= 0x2e80 && r <= 0xa4cf || r >= 0xac00 && r <= 0xd7a3 || r >= 0xf900 && r <= 0xfaff || r >= 0xfe10 && r <= 0xfe19 || r >= 0xfe30 && r <= 0xfe6f || r >= 0xff00 && r <= 0xff60 || r >= 0xffe0 && r <= 0xffe6 || r >= 0x1f300 && r <= 0x1faff) {
		return 2
	}
	return 1
}
func safeTerminalText(text string) string {
	var output strings.Builder
	for _, r := range text {
		if r == '\n' || r == '\t' || !unicode.IsControl(r) {
			output.WriteRune(r)
		}
	}
	return output.String()
}

func stripANSI(text string) string {
	var out strings.Builder
	for index := 0; index < len(text); {
		if text[index] == 0x1b && index+1 < len(text) && text[index+1] == '[' {
			index += 2
			for index < len(text) && (text[index] < '@' || text[index] > '~') {
				index++
			}
			if index < len(text) {
				index++
			}
			continue
		}
		out.WriteByte(text[index])
		index++
	}
	return out.String()
}
