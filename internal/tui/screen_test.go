package tui

import (
	"strings"
	"testing"
)

func TestRenderFrameUsesResponsiveSidebar(t *testing.T) {
	frame := Frame{Title: "test", Messages: []Message{{Role: "user", Text: strings.Repeat("hello ", 30)}}, Sidebar: []string{"Session", "model"}, Input: []rune("你好"), Cursor: 2}
	lines, row, col := renderFrame(frame, 100, 24)
	if len(lines) != 24 || row != 24 {
		t.Fatalf("dimensions = %d, %d", len(lines), row)
	}
	if !strings.Contains(stripANSI(lines[2]), "Session") {
		t.Fatalf("sidebar missing: %q", lines[2])
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

func TestRenderFrameHidesSidebarWhenNarrow(t *testing.T) {
	lines, _, _ := renderFrame(Frame{Sidebar: []string{"SHOULD_NOT_APPEAR"}}, 60, 16)
	if strings.Contains(strings.Join(lines, "\n"), "SHOULD_NOT_APPEAR") {
		t.Fatal("sidebar visible on narrow screen")
	}
}

func TestRuneWidthHandlesCJKAndCombiningMarks(t *testing.T) {
	if got := runeSliceWidth([]rune("A你e\u0301")); got != 4 {
		t.Fatalf("width = %d", got)
	}
}
