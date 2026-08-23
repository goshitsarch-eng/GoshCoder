package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestRenderFrameUsesResponsiveSidebar(t *testing.T) {
	frame := Frame{Title: "test", Messages: []Message{{Role: "user", Text: strings.Repeat("hello ", 30)}}, Sidebar: []string{"Session", "model"}, Input: []rune("你好"), Cursor: 2}
	lines, row, col, _ := renderFrame(frame, 100, 24)
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
	lines, _, _, _ := renderFrame(frame, 80, 18)
	output := stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(output, "▌  /model") || !strings.Contains(output, "/help") {
		t.Fatalf("suggestions missing: %s", output)
	}
}

func TestRenderFrameRendersMultilineComposer(t *testing.T) {
	input := []rune("first line\nsecond line\nthird line")
	frame := Frame{Input: input, Cursor: len(input), VirtualCursor: true}
	lines, row, col, _ := renderFrame(frame, 80, 20)
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
	lines, _, _, _ := renderFrame(frame, 30, 16)
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
	lines, _, _, _ := renderFrame(frame, 80, 16)
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

func TestRenderMessagesKeepsLeadingIndentedCode(t *testing.T) {
	lines := renderMessages([]Message{{Role: "assistant", Text: "    func main() {}"}}, 80, false, false, nil)
	plain := stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(plain, "func main() {}") || !strings.Contains(plain, "│") {
		t.Fatalf("indented assistant code lost: %q", plain)
	}
}

// TestMessageCacheReusesRenderedMessages covers the memoization the fullscreen
// UI relies on: the whole transcript is re-rendered on every frame -- every
// keystroke and every stream delta -- and markdown-parsing hundreds of
// unchanged messages each time is what makes typing lag in a long session.
func TestMessageCacheReusesRenderedMessages(t *testing.T) {
	messages := []Message{
		{Role: "user", Text: "first question"},
		{Role: "assistant", Text: "# Heading\n\nSome **bold** text and `code`."},
		{Role: "assistant", Text: "second answer"},
	}
	cache := NewMessageCache()

	first := renderMessages(messages, 80, false, false, cache)
	second := renderMessages(messages, 80, false, false, cache)
	if strings.Join(first, "\n") != strings.Join(second, "\n") {
		t.Fatal("cached render differs from the first render")
	}
	if len(cache.entries) != len(messages) {
		t.Fatalf("cache holds %d entries for %d messages", len(cache.entries), len(messages))
	}

	// An uncached render must produce exactly the same output.
	uncached := renderMessages(messages, 80, false, false, nil)
	if strings.Join(uncached, "\n") != strings.Join(first, "\n") {
		t.Fatal("cached and uncached renders disagree")
	}
}

// TestMessageCacheInvalidatesOnLayoutChange keeps a resize or a toggle from
// showing lines rendered for the previous layout.
func TestMessageCacheInvalidatesOnLayoutChange(t *testing.T) {
	messages := []Message{{Role: "assistant", Text: strings.Repeat("word ", 40)}}
	cache := NewMessageCache()

	wide := renderMessages(messages, 100, false, false, cache)
	narrow := renderMessages(messages, 40, false, false, cache)
	if strings.Join(wide, "\n") == strings.Join(narrow, "\n") {
		t.Fatal("a width change reused lines wrapped for the old width")
	}
	if want := renderMessages(messages, 40, false, false, nil); strings.Join(narrow, "\n") != strings.Join(want, "\n") {
		t.Fatal("post-resize render does not match an uncached render at the new width")
	}
}
