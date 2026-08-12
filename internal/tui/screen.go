// Package tui provides GoshCoder's dependency-free fullscreen terminal UI.
package tui

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"unicode"
)

const accent = "\x1b[38;2;215;119;87m"
const reset = "\x1b[0m"
const dim = "\x1b[2m"

// Message is one transcript entry rendered in the main viewport.
type Message struct {
	Role string
	Text string
}

// Frame is a complete immutable screen snapshot.
type Frame struct {
	Title     string
	Messages  []Message
	Sidebar   []string
	Input     []rune
	Cursor    int
	Status    string
	Streaming bool
	Scroll    int
}

// Terminal owns raw mode and the alternate screen.
type Terminal struct {
	in, out *os.File
	state   *terminalState
	mu      sync.Mutex
	closed  bool
}

// Supported reports whether both files are terminals with native raw-mode support.
func Supported(in, out *os.File) bool { return isTerminal(in) && isTerminal(out) }

// Open enters raw mode and switches to the terminal's alternate screen.
func Open(in, out *os.File) (*Terminal, error) {
	if !Supported(in, out) {
		return nil, fmt.Errorf("fullscreen TUI requires a terminal")
	}
	state, err := makeRaw(in)
	if err != nil {
		return nil, err
	}
	terminal := &Terminal{in: in, out: out, state: state}
	if _, err := io.WriteString(out, "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H"); err != nil {
		_ = state.restore()
		return nil, err
	}
	return terminal, nil
}

// Size returns the current terminal dimensions.
func (terminal *Terminal) Size() (int, int) { return terminalSize(terminal.out) }

// Close restores the normal screen and terminal settings.
func (terminal *Terminal) Close() error {
	terminal.mu.Lock()
	defer terminal.mu.Unlock()
	if terminal.closed {
		return nil
	}
	terminal.closed = true
	_, writeErr := io.WriteString(terminal.out, "\x1b[?25h\x1b[0m\x1b[?1049l")
	restoreErr := terminal.state.restore()
	if writeErr != nil {
		return writeErr
	}
	return restoreErr
}

// Render repaints the screen and places the hardware cursor in the editor.
func (terminal *Terminal) Render(frame Frame) error {
	terminal.mu.Lock()
	defer terminal.mu.Unlock()
	if terminal.closed {
		return nil
	}
	width, height := terminal.Size()
	if width < 20 || height < 8 {
		return nil
	}
	lines, cursorRow, cursorCol := renderFrame(frame, width, height)
	var output strings.Builder
	output.WriteString("\x1b[?25l\x1b[H")
	for row := 0; row < height; row++ {
		fmt.Fprintf(&output, "\x1b[%d;1H\x1b[2K", row+1)
		if row < len(lines) {
			output.WriteString(lines[row])
		}
	}
	fmt.Fprintf(&output, "\x1b[%d;%dH\x1b[?25h", cursorRow, cursorCol)
	_, err := io.WriteString(terminal.out, output.String())
	return err
}

func renderFrame(frame Frame, width, height int) ([]string, int, int) {
	useSidebar := width >= 90
	sideWidth := 0
	if useSidebar {
		sideWidth = min(32, width/3)
	}
	mainWidth := width
	if useSidebar {
		mainWidth = width - sideWidth - 1
	}
	bodyTop, bodyBottom := 2, height-4
	bodyHeight := max(1, bodyBottom-bodyTop+1)

	transcript := renderMessages(frame.Messages, mainWidth)
	start := max(0, len(transcript)-bodyHeight-frame.Scroll)
	end := min(len(transcript), start+bodyHeight)
	visible := transcript[start:end]

	lines := make([]string, height)
	title := frame.Title
	if title == "" {
		title = "GoshCoder"
	}
	lines[0] = accent + "● " + reset + truncate(title, width)
	lines[1] = dim + strings.Repeat("─", width) + reset
	for row := 0; row < bodyHeight; row++ {
		left := ""
		if row < len(visible) {
			left = visible[row]
		}
		left = pad(left, mainWidth)
		if useSidebar {
			right := ""
			if row < len(frame.Sidebar) {
				right = frame.Sidebar[row]
			}
			lines[bodyTop+row] = left + dim + "│" + reset + pad(truncate(right, sideWidth), sideWidth)
		} else {
			lines[bodyTop+row] = left
		}
	}
	status := frame.Status
	if frame.Streaming {
		status = "◌ working  ·  Esc/Ctrl-C abort  ·  " + status
	}
	lines[height-3] = dim + truncate(status, width) + reset
	lines[height-2] = accent + "╭" + strings.Repeat("─", width-2) + "╮" + reset

	promptWidth := max(1, width-5)
	cursor := min(max(frame.Cursor, 0), len(frame.Input))
	beforeWidth := runeSliceWidth(frame.Input[:cursor])
	windowStart := 0
	for beforeWidth-runeSliceWidth(frame.Input[:windowStart]) >= promptWidth {
		windowStart++
	}
	shown := truncateRunes(frame.Input[windowStart:], promptWidth)
	lines[height-1] = accent + "╰─❯ " + reset + string(shown)
	cursorCol := 5 + runeSliceWidth(frame.Input[windowStart:cursor])
	return lines, height, min(width, cursorCol)
}

func renderMessages(messages []Message, width int) []string {
	var lines []string
	for _, message := range messages {
		if strings.TrimSpace(message.Text) == "" {
			continue
		}
		label := message.Role
		switch label {
		case "user":
			label = "You"
		case "assistant":
			label = "GoshCoder"
		case "tool":
			label = "Tool"
		case "thinking":
			label = "Thinking"
		}
		lines = append(lines, accent+label+reset)
		for _, paragraph := range strings.Split(strings.ReplaceAll(message.Text, "\t", "    "), "\n") {
			lines = append(lines, wrap(paragraph, max(1, width-2))...)
		}
		lines = append(lines, "")
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
	plainWidth := runeSliceWidth([]rune(stripANSI(text)))
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
