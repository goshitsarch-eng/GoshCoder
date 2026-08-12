// Package tui renders GoshCoder's Bubble Tea fullscreen interface.
package tui

import (
	"os"
	"strings"
	"unicode"

	"github.com/charmbracelet/lipgloss"
	charmterm "github.com/charmbracelet/x/term"
)

const background = "\x1b[48;2;30;27;33m"
const accent = "\x1b[38;2;255;92;214m"
const violet = "\x1b[38;2;126;92;255m"
const cyan = "\x1b[38;2;55;230;161m"
const blue = "\x1b[38;2;128;170;255m"
const amber = "\x1b[38;2;245;190;92m"
const red = "\x1b[38;2;255;105;120m"
const textColor = "\x1b[38;2;230;225;233m"
const muted = "\x1b[38;2;116;109;124m"
const faint = "\x1b[38;2;76;70;83m"
const selected = "\x1b[48;2;62;49;70m\x1b[38;2;255;240;252m"
const bold = "\x1b[1m"
const dim = "\x1b[2m"
const clearStyle = "\x1b[0m"
const reset = clearStyle + background

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
		return background + accent + bold + "GoshCoder" + reset + "\n" + muted + "Terminal is too small" + clearStyle
	}
	lines, _, _ := renderFrame(frame, width, height)
	return strings.Join(lines, "\n")
}

func renderFrame(frame Frame, width, height int) ([]string, int, int) {
	useSidebar := width >= 100
	sideWidth, gutter := 0, 0
	if useSidebar {
		sideWidth = min(36, width/3)
		gutter = 3
	}
	mainWidth := width - sideWidth - gutter
	mainLines := make([]string, height)

	title := frame.Title
	if title == "" {
		title = "interactive session"
	}
	if useSidebar {
		mainLines[0] = muted + "  " + truncate(title, max(1, mainWidth-2)) + reset
	} else {
		mainLines[0] = accent + bold + "  GOSH" + violet + "CODER" + reset + muted + "  " + truncate(title, max(1, mainWidth-16)) + reset
	}
	mainLines[1] = accent + "  · · · · · ·" + reset

	bodyTop, bodyBottom := 2, height-5
	bodyHeight := max(1, bodyBottom-bodyTop+1)
	suggestionRows := min(len(frame.Suggestions), min(9, max(0, bodyHeight-2)))
	paletteHeight := 0
	if suggestionRows > 0 {
		paletteHeight = suggestionRows + 2
	}
	transcriptHeight := max(0, bodyHeight-paletteHeight)
	transcript := renderMessages(frame.Messages, mainWidth)
	start := max(0, len(transcript)-transcriptHeight-frame.Scroll)
	end := min(len(transcript), start+transcriptHeight)
	visible := transcript[start:end]

	suggestionOffset := max(0, frame.SelectedSuggestion-suggestionRows+1)
	paletteStart := bodyHeight - paletteHeight
	panelWidth := max(8, mainWidth-4)
	for row := 0; row < bodyHeight; row++ {
		line := ""
		switch {
		case paletteHeight > 0 && row == paletteStart:
			title := frame.SuggestionTitle
			if title == "" {
				title = "COMMANDS"
			}
			title = " " + truncate(title, max(1, panelWidth-5)) + " "
			line = "  " + violet + "╭─" + reset + muted + title + reset + violet + strings.Repeat("─", max(0, panelWidth-lipgloss.Width(title)-3)) + "╮" + reset
		case paletteHeight > 0 && row > paletteStart && row < bodyHeight-1:
			index := suggestionOffset + row - paletteStart - 1
			if index < len(frame.Suggestions) {
				parts := strings.SplitN(frame.Suggestions[index], "\t", 2)
				label, description := parts[0], ""
				if len(parts) == 2 {
					description = parts[1]
				}
				contentWidth := max(1, panelWidth-3)
				entry := "  " + label
				remaining := contentWidth - lipgloss.Width(entry) - 2
				if remaining > 8 && description != "" {
					entry += "  " + truncate(description, remaining)
				}
				entry = pad(entry, contentWidth)
				if index == frame.SelectedSuggestion {
					line = "  " + violet + "│" + reset + selected + "▌" + bold + entry + reset + violet + "│" + reset
				} else {
					line = "  " + violet + "│" + reset + " " + textColor + entry + reset + violet + "│" + reset
				}
			}
		case paletteHeight > 0 && row == bodyHeight-1:
			line = "  " + violet + "╰" + strings.Repeat("─", max(0, panelWidth-1)) + "╯" + reset
		case row < len(visible):
			line = visible[row]
		}
		mainLines[bodyTop+row] = line
	}

	composerRow := height - 3
	promptWidth := max(1, mainWidth-8)
	cursor := min(max(frame.Cursor, 0), len(frame.Input))
	beforeWidth := runeSliceWidth(frame.Input[:cursor])
	windowStart := 0
	for windowStart < cursor && beforeWidth-runeSliceWidth(frame.Input[:windowStart]) >= promptWidth {
		windowStart++
	}
	shown := truncateRunes(frame.Input[windowStart:], promptWidth)
	editorText := string(shown)
	if len(frame.Input) == 0 {
		placeholder := truncate("Tell GoshCoder what to build…  / for commands", max(1, promptWidth-2))
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
	mainLines[composerRow] = cyan + bold + "  > " + reset + textColor + editorText + reset
	mainLines[composerRow+1] = cyan + "  ┊" + reset

	status := safeTerminalText(frame.Status)
	leftStatus := cyan + "  ● " + reset + muted + status + reset
	if frame.Streaming {
		leftStatus = accent + "  ● " + reset + textColor + status + reset
	}
	hints := "enter send  ·  shift-tab thinking  ·  pgup scroll  ·  ctrl+c quit"
	if frame.Streaming {
		hints = "esc abort  ·  type to steer"
	}
	rightStatus := faint + hints + reset
	gap := mainWidth - lipgloss.Width(leftStatus) - lipgloss.Width(rightStatus)
	if gap >= 2 {
		mainLines[height-1] = leftStatus + strings.Repeat(" ", gap) + rightStatus
	} else {
		mainLines[height-1] = muted + "  " + truncate(status, max(1, mainWidth-2)) + reset
	}

	sideLines := renderSidebar(frame.Sidebar, sideWidth, height)
	lines := make([]string, height)
	for row := 0; row < height; row++ {
		left := pad(mainLines[row], mainWidth)
		if useSidebar {
			right := ""
			if row < len(sideLines) {
				right = sideLines[row]
			}
			lines[row] = canvasLine(left+strings.Repeat(" ", gutter)+pad(right, sideWidth), width)
		} else {
			lines[row] = canvasLine(left, width)
		}
	}
	cursorCol := 5 + runeSliceWidth(frame.Input[windowStart:cursor])
	return lines, composerRow + 1, min(mainWidth, cursorCol)
}

func renderSidebar(sidebar []string, width, height int) []string {
	if width == 0 {
		return nil
	}
	lines := []string{
		violet + "  ╱╱╱╱╱╱╱╱╱╱╱╱╱╱" + reset,
		accent + bold + "  █▀▀ █▀█ █▀▀ █ █" + reset,
		violet + bold + "  █▄█ █▄█ ▄▄█ █▀█" + reset,
		muted + "             C O D E R" + reset,
		violet + "  ╱╱╱╱╱╱╱╱╱╱╱╱╱╱" + reset,
		"",
	}
	for _, line := range sidebar {
		lines = append(lines, styleSidebarLine(line, width))
	}
	if len(lines) > height {
		lines = lines[:height]
	}
	return lines
}

func renderMessages(messages []Message, width int) []string {
	var lines []string
	for _, message := range messages {
		if strings.TrimSpace(message.Text) == "" {
			continue
		}
		switch message.Role {
		case "user":
			lines = append(lines, accent+"  │ "+reset+muted+"YOU"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-7), "user") {
				lines = append(lines, accent+"  │ "+reset+textColor+line+reset)
			}
		case "assistant":
			lines = append(lines, violet+"  ◆ "+reset+bold+textColor+"GOSHCODER"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-5), "assistant") {
				lines = append(lines, "    "+line)
			}
		case "thinking":
			lines = append(lines, amber+"  ◌ "+reset+muted+"THINKING"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-7), "thinking") {
				lines = append(lines, muted+"  ┊ "+line+reset)
			}
		case "tool":
			lines = append(lines, cyan+"  ✓ "+reset+textColor+"Tool"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-7), "tool") {
				lines = append(lines, faint+"  ┊ "+reset+muted+line+reset)
			}
		case "Error":
			lines = append(lines, red+bold+"  × ERROR"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-5), "error") {
				lines = append(lines, red+"    "+line+reset)
			}
		case "Command":
			lines = append(lines, violet+"  ◇ "+reset+muted+"COMMAND"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-5), "command") {
				lines = append(lines, textColor+"    "+line+reset)
			}
		default:
			lines = append(lines, cyan+"  i "+reset+muted+strings.ToUpper(message.Role)+reset)
			for _, line := range renderRichText(message.Text, max(1, width-5), "notice") {
				lines = append(lines, muted+"    "+line+reset)
			}
		}
		lines = append(lines, "")
	}
	return lines
}

func renderRichText(source string, width int, role string) []string {
	source = safeTerminalText(strings.ReplaceAll(source, "\t", "    "))
	var lines []string
	inCode := false
	for _, raw := range strings.Split(source, "\n") {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "```") {
			inCode = !inCode
			continue
		}
		if inCode {
			for _, line := range wrap(raw, max(1, width-2)) {
				lines = append(lines, faint+"│ "+reset+blue+line+reset)
			}
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			heading := strings.TrimSpace(strings.TrimLeft(trimmed, "#"))
			for _, line := range wrapWords(heading, width) {
				lines = append(lines, accent+bold+line+reset)
			}
			continue
		}
		if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			body := strings.TrimSpace(trimmed[2:])
			wrapped := wrapWords(body, max(1, width-2))
			for index, line := range wrapped {
				prefix := "  "
				if index == 0 {
					prefix = cyan + "• " + reset
				}
				lines = append(lines, prefix+line)
			}
			continue
		}
		if strings.HasPrefix(trimmed, "> ") {
			for _, line := range wrapWords(strings.TrimSpace(trimmed[2:]), max(1, width-2)) {
				lines = append(lines, faint+"│ "+reset+muted+line+reset)
			}
			continue
		}
		if trimmed == "" {
			lines = append(lines, "")
			continue
		}
		for _, line := range wrapWords(raw, width) {
			if role == "thinking" {
				lines = append(lines, muted+line+reset)
			} else {
				lines = append(lines, textColor+line+reset)
			}
		}
	}
	return lines
}

func styleSidebarLine(line string, width int) string {
	plain := safeTerminalText(line)
	trimmed := strings.TrimSpace(plain)
	if trimmed == "" {
		return ""
	}
	if trimmed == strings.ToUpper(trimmed) {
		label := " " + trimmed + " "
		return muted + "  " + label + faint + strings.Repeat("─", max(0, width-lipgloss.Width(label)-4)) + reset
	}
	if strings.HasPrefix(trimmed, "◇") {
		return violet + "  " + trimmed + reset
	}
	if strings.HasPrefix(trimmed, "●") {
		return cyan + "  " + trimmed + reset
	}
	return muted + "  " + truncate(trimmed, max(1, width-4)) + reset
}

func canvasLine(text string, width int) string {
	return background + pad(text, width) + clearStyle
}

func wrapWords(text string, width int) []string {
	text = safeTerminalText(text)
	if text == "" {
		return []string{""}
	}
	words := strings.Fields(text)
	if len(words) == 0 {
		return []string{""}
	}
	var lines []string
	current := ""
	for _, word := range words {
		if runeSliceWidth([]rune(word)) > width {
			if current != "" {
				lines = append(lines, current)
				current = ""
			}
			parts := wrap(word, width)
			lines = append(lines, parts[:len(parts)-1]...)
			current = parts[len(parts)-1]
			continue
		}
		candidate := word
		if current != "" {
			candidate = current + " " + word
		}
		if runeSliceWidth([]rune(candidate)) > width {
			lines = append(lines, current)
			current = word
		} else {
			current = candidate
		}
	}
	if current != "" {
		lines = append(lines, current)
	}
	return lines
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
