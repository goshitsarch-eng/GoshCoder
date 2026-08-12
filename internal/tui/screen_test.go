package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestRenderFrameUsesResponsiveSidebar(t *testing.T) {
	frame := Frame{Title: "test", Messages: []Message{{Role: "user", Text: strings.Repeat("hello ", 30)}}, Sidebar: []string{"Session", "model"}, Input: []rune("你好"), Cursor: 2}
	lines, row, col := renderFrame(frame, 100, 24)
	if len(lines) != 24 || row != 22 {
		t.Fatalf("dimensions = %d, %d", len(lines), row)
	}
	if !strings.Contains(stripANSI(strings.Join(lines, "\n")), "Session") {
		t.Fatalf("sidebar missing: %q", strings.Join(lines, "\n"))
	}
	if col <= 5 {
		t.Fatalf("cursor column = %d", col)
	}
	for index, line := range lines {
		if runeSliceWidth([]rune(stripANSI(line))) > 100 {
			t.Fatalf("line %d exceeds width: %q", index, line)
		}
	}
}

func TestRenderFrameShowsSlashSuggestions(t *testing.T) {
	frame := Frame{Suggestions: []string{"/help  Show help", "/model  Switch model"}, SelectedSuggestion: 1}
	lines, _, _ := renderFrame(frame, 80, 18)
	output := stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(output, "▌  /model") || !strings.Contains(output, "/help") {
		t.Fatalf("suggestions missing: %s", output)
	}
}

func TestRenderFrameRendersMultilineComposer(t *testing.T) {
	input := []rune("first line\nsecond line\nthird line")
	frame := Frame{Input: input, Cursor: len(input), VirtualCursor: true}
	lines, row, col := renderFrame(frame, 80, 20)
	output := stripANSI(strings.Join(lines, "\n"))
	for _, text := range []string{"first line", "second line", "third line"} {
		if !strings.Contains(output, text) {
			t.Fatalf("composer missing %q: %s", text, output)
		}
	}
	if row != 18 || col <= 5 {
		t.Fatalf("multiline cursor = row %d, col %d", row, col)
	}
}

func TestLayoutEditorKeepsCursorLineInThreeRowWindow(t *testing.T) {
	input := []rune("one\ntwo\nthree\nfour")
	layout := layoutEditor(input, len(input), 20, false)
	if got := strings.Join(layout.lines, "|"); got != "two|three|four" {
		t.Fatalf("visible editor = %q", got)
	}
	if layout.cursorRow != 2 || layout.cursorCol != 4 {
		t.Fatalf("cursor = %d,%d", layout.cursorRow, layout.cursorCol)
	}
}

func TestRenderFrameHidesSidebarWhenNarrow(t *testing.T) {
	frame := Frame{
		Title: "a/very-long-provider-and-model-name", Sidebar: []string{"SHOULD_NOT_APPEAR"},
		VirtualCursor: true,
	}
	lines, _, _ := renderFrame(frame, 30, 16)
	if strings.Contains(strings.Join(lines, "\n"), "SHOULD_NOT_APPEAR") {
		t.Fatal("sidebar visible on narrow screen")
	}
	for index, line := range lines {
		if lipgloss.Width(line) > 30 {
			t.Fatalf("line %d exceeds narrow width: %q", index, line)
		}
	}
}

func TestRenderFrameRemovesUntrustedTerminalEscapes(t *testing.T) {
	frame := Frame{Title: "bad\x1b]52;c;payload\a", Messages: []Message{{Role: "assistant", Text: "hello\x1b[2Jworld"}}}
	lines, _, _ := renderFrame(frame, 80, 16)
	output := strings.Join(lines, "\n")
	if strings.Contains(output, "\x1b]52") || strings.Contains(output, "\x1b[2Jworld") {
		t.Fatalf("untrusted escape survived: %q", output)
	}
}

func TestRuneWidthHandlesCJKAndCombiningMarks(t *testing.T) {
	if got := runeSliceWidth([]rune("A你e\u0301")); got != 4 {
		t.Fatalf("width = %d", got)
	}
}
