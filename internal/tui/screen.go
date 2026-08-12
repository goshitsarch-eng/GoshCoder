// Package tui renders GoshCoder's Bubble Tea fullscreen interface.
package tui

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"unicode"

	"github.com/charmbracelet/lipgloss"
	charmterm "github.com/charmbracelet/x/term"
)

const background = "\x1b[48;2;10;10;10m"
const panelBackground = "\x1b[48;2;20;20;20m"
const userBackground = "\x1b[48;2;30;30;30m"
const toolBackground = "\x1b[48;2;24;24;24m"
const accent = "\x1b[38;2;255;172;92m"
const violet = "\x1b[38;2;73;166;191m"
const cyan = "\x1b[38;2;86;182;194m"
const green = "\x1b[38;2;127;216;143m"
const blue = "\x1b[38;2;112;174;221m"
const amber = "\x1b[38;2;242;201;108m"
const red = "\x1b[38;2;246;116;116m"
const textColor = "\x1b[38;2;220;230;232m"
const muted = "\x1b[38;2;119;143;150m"
const faint = "\x1b[38;2;62;87;96m"
const selected = "\x1b[48;2;27;67;76m\x1b[38;2;239;248;246m"
const bold = "\x1b[1m"
const dim = "\x1b[2m"
const clearStyle = "\x1b[0m"
const reset = clearStyle + background

// Message is one transcript entry rendered in the main viewport.
type Message struct {
	Role    string
	Title   string
	Text    string
	Detail  string
	IsError bool
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
	ToolsExpanded      bool
	HideThinking       bool
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
	useSidebar := width >= 96
	sideWidth, gutter := 0, 0
	if useSidebar {
		sideWidth = min(42, max(32, width/3))
		gutter = 1
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

	promptWidth := max(1, mainWidth-8)
	editor := layoutEditor(frame.Input, frame.Cursor, promptWidth, frame.VirtualCursor)
	composerTop := height - len(editor.lines) - 3
	bodyTop, bodyBottom := 2, composerTop-1
	bodyHeight := max(0, bodyBottom-bodyTop+1)
	suggestionRows := min(len(frame.Suggestions), min(9, max(0, bodyHeight-2)))
	paletteHeight := 0
	if suggestionRows > 0 {
		paletteHeight = suggestionRows + 2
	}
	transcriptHeight := max(0, bodyHeight-paletteHeight)
	transcript := renderMessages(frame.Messages, mainWidth, frame.ToolsExpanded, frame.HideThinking)
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

	borderWidth := max(2, mainWidth-4)
	mainLines[composerTop] = cyan + "  ╭" + strings.Repeat("─", max(0, borderWidth-2)) + "╮" + reset
	for index, editorLine := range editor.lines {
		mainLines[composerTop+1+index] = textColor + "    " + editorLine + reset
	}
	mainLines[composerTop+len(editor.lines)+1] = cyan + "  ╰" + strings.Repeat("─", max(0, borderWidth-2)) + "╯" + reset

	status := safeTerminalText(frame.Status)
	leftStatus := cyan + "  ● " + reset + muted + status + reset
	if frame.Streaming {
		leftStatus = accent + "  ● " + reset + textColor + status + reset
	}
	hints := "enter send · shift+enter newline · ctrl+l model · / commands"
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
			right = strings.ReplaceAll(right, reset, clearStyle+panelBackground)
			right = panelBackground + pad(right, sideWidth) + clearStyle + background
			lines[row] = canvasLine(left+strings.Repeat(" ", gutter)+right, width)
		} else {
			lines[row] = canvasLine(left, width)
		}
	}
	cursorRow := composerTop + 1 + editor.cursorRow
	return lines, cursorRow + 1, min(mainWidth, 5+editor.cursorCol)
}

type editorLayout struct {
	lines     []string
	cursorRow int
	cursorCol int
}

func layoutEditor(input []rune, cursor, width int, virtualCursor bool) editorLayout {
	cursor = min(max(cursor, 0), len(input))
	logical := strings.Split(string(input), "\n")
	cursorLine, cursorInLine := 0, 0
	for index := 0; index < cursor; index++ {
		if input[index] == '\n' {
			cursorLine++
			cursorInLine = 0
		} else {
			cursorInLine++
		}
	}
	if len(logical) == 0 {
		logical = []string{""}
	}
	cursorLine = min(cursorLine, len(logical)-1)
	startLine := max(0, cursorLine-2)
	if len(logical)-startLine < 3 {
		startLine = max(0, len(logical)-3)
	}
	endLine := min(len(logical), startLine+3)
	visible := logical[startLine:endLine]
	layout := editorLayout{lines: make([]string, len(visible)), cursorRow: cursorLine - startLine}
	for visibleIndex, raw := range visible {
		runes := []rune(raw)
		lineIndex := startLine + visibleIndex
		lineCursor := 0
		if lineIndex == cursorLine {
			lineCursor = min(cursorInLine, len(runes))
		}
		windowStart := 0
		if lineIndex == cursorLine {
			beforeWidth := runeSliceWidth(runes[:lineCursor])
			for windowStart < lineCursor && beforeWidth-runeSliceWidth(runes[:windowStart]) >= width {
				windowStart++
			}
		}
		shown := truncateRunes(runes[windowStart:], width)
		text := string(shown)
		if lineIndex == cursorLine {
			relative := lineCursor - windowStart
			layout.cursorCol = runeSliceWidth(runes[windowStart:lineCursor])
			if virtualCursor {
				if relative < len(shown) {
					text = string(shown[:relative]) + accent + "\x1b[7m" + string(shown[relative:relative+1]) + "\x1b[27m" + textColor + string(shown[relative+1:])
				} else {
					text += accent + "\x1b[7m \x1b[27m" + textColor
				}
			}
		}
		layout.lines[visibleIndex] = text
	}
	if len(input) == 0 {
		placeholder := truncate("Tell GoshCoder what to build…  / for commands", max(1, width-2))
		if virtualCursor {
			layout.lines[0] = accent + "\x1b[7m \x1b[27m" + muted + " " + placeholder + reset
		} else {
			layout.lines[0] = muted + placeholder + reset
		}
	}
	return layout
}

func renderSidebar(sidebar []string, width, height int) []string {
	if width == 0 {
		return nil
	}
	lines := []string{""}
	for _, line := range sidebar {
		lines = append(lines, styleSidebarLine(line, width))
	}
	if len(lines) > height {
		lines = lines[:height]
	}
	return lines
}

func renderMessages(messages []Message, width int, toolsExpanded, hideThinking bool) []string {
	var lines []string
	for _, message := range messages {
		if strings.TrimSpace(message.Text) == "" && strings.TrimSpace(message.Title) == "" {
			continue
		}
		switch message.Role {
		case "user":
			contentWidth := max(1, width-6)
			lines = append(lines, "  "+userBackground+strings.Repeat(" ", max(1, width-4))+reset)
			for _, line := range renderRichText(message.Text, contentWidth, "user") {
				line = strings.ReplaceAll(line, reset, clearStyle+userBackground)
				lines = append(lines, "  "+userBackground+" "+pad(line, contentWidth)+" "+reset)
			}
			lines = append(lines, "  "+userBackground+strings.Repeat(" ", max(1, width-4))+reset)
		case "assistant":
			for _, line := range renderRichText(strings.TrimSpace(message.Text), max(1, width-4), "assistant") {
				lines = append(lines, "  "+line)
			}
		case "thinking":
			if hideThinking {
				lines = append(lines, muted+dim+"  Thinking…"+reset)
				break
			}
			for _, line := range renderRichText(strings.TrimSpace(message.Text), max(1, width-4), "thinking") {
				lines = append(lines, muted+"\x1b[3m  "+line+"\x1b[23m"+reset)
			}
		case "tool":
			boxWidth := max(1, width-4)
			icon, titleColor := "✓", cyan
			if message.IsError {
				icon, titleColor = "×", red
			}
			title := strings.TrimSpace(message.Title)
			if title == "" {
				title = "tool"
			}
			header := titleColor + icon + " " + bold + truncate(title, max(1, boxWidth-4)) + reset
			lines = append(lines, "  "+toolBackground+" "+pad(header, max(1, boxWidth-2))+" "+reset)
			detail := strings.TrimSpace(message.Text)
			if toolsExpanded || message.IsError {
				if strings.TrimSpace(message.Detail) != "" {
					detail = strings.TrimSpace(message.Detail)
				}
			}
			if detail != "" {
				maxLines := 3
				if toolsExpanded || message.IsError {
					maxLines = 20
				}
				rendered := renderRichText(detail, max(1, boxWidth-4), "tool")
				for index, line := range rendered {
					line = strings.ReplaceAll(line, reset, clearStyle+toolBackground)
					if index >= maxLines {
						remaining := len(rendered) - index
						more := fmt.Sprintf("… %d more lines (ctrl+o to expand)", remaining)
						lines = append(lines, "  "+toolBackground+" "+muted+pad(more, max(1, boxWidth-2))+reset)
						break
					}
					lines = append(lines, "  "+toolBackground+" "+muted+pad("  "+line, max(1, boxWidth-2))+reset)
				}
			}
		case "Error":
			lines = append(lines, red+bold+"  Error"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-4), "error") {
				lines = append(lines, red+"  "+line+reset)
			}
		case "Command":
			lines = append(lines, violet+"  ◇ "+reset+muted+"Command"+reset)
			for _, line := range renderRichText(message.Text, max(1, width-4), "command") {
				lines = append(lines, textColor+"  "+line+reset)
			}
		default:
			lines = append(lines, cyan+"  i "+reset+muted+strings.ToUpper(message.Role)+reset)
			for _, line := range renderRichText(message.Text, max(1, width-4), "notice") {
				lines = append(lines, muted+"  "+line+reset)
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
	if strings.TrimSpace(plain) == "" {
		return ""
	}
	parts := strings.Split(plain, "\t")
	kind := parts[0]
	value := ""
	if len(parts) > 1 {
		value = strings.TrimSpace(parts[1])
	}
	available := max(1, width-5)
	switch kind {
	case "title":
		return textColor + bold + "  " + truncate(value, available) + reset
	case "section":
		return textColor + bold + "  " + truncate(value, available) + reset
	case "accent":
		return accent + "  " + truncate(value, available) + reset
	case "active":
		return green + "  ● " + reset + textColor + truncate(value, max(1, width-7)) + reset
	case "meta":
		return muted + "  " + truncate(value, available) + reset
	case "path":
		return muted + "  " + truncateLeft(value, available) + reset
	case "brand":
		return green + "  " + truncate(value, available) + reset
	case "bar":
		percent, _ := strconv.Atoi(value)
		cells := max(4, width-9)
		filled := min(cells, max(0, percent*cells/100))
		return accent + "  " + strings.Repeat("━", filled) + faint + strings.Repeat("━", cells-filled) + reset
	case "todo":
		state, text := "open", ""
		if len(parts) > 1 {
			state = parts[1]
		}
		if len(parts) > 2 {
			text = parts[2]
		}
		if state == "done" {
			return green + "  ☑ " + reset + muted + truncate(text, max(1, width-7)) + reset
		}
		return muted + "  ☐ " + reset + textColor + truncate(text, max(1, width-7)) + reset
	case "file":
		status, path := "", ""
		if len(parts) > 1 {
			status = parts[1]
		}
		if len(parts) > 2 {
			path = parts[2]
		}
		statusColor := amber
		if strings.Contains(status, "?") || strings.Contains(status, "A") {
			statusColor = green
		} else if strings.Contains(status, "D") {
			statusColor = red
		}
		return statusColor + "  " + pad(truncate(status, 2), 2) + " " + reset + muted + truncateLeft(path, max(1, width-8)) + reset
	default:
		return muted + "  " + truncate(strings.TrimSpace(plain), available) + reset
	}
}

func truncateLeft(text string, width int) string {
	text = safeTerminalText(stripANSI(text))
	if lipgloss.Width(text) <= width {
		return text
	}
	if width <= 1 {
		return "…"
	}
	runes := []rune(text)
	start := len(runes)
	cells := 1
	for start > 0 {
		next := runeWidth(runes[start-1])
		if cells+next > width {
			break
		}
		cells += next
		start--
	}
	return "…" + string(runes[start:])
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
