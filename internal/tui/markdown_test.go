package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestRenderRichTextKeepsMarkdownSections(t *testing.T) {
	source := `# Overview

#include <stdio.h>

- Item 1
  - Nested 1.1
  - Nested 1.2
- Item 2

1. alpha
1. beta
1. gamma

- [ ] beep
- [x] boop

| Name | Age |
|------|-----|
| Alice | 12 |
| Bob | 9 |

` + "```go\nfunc main() {}\n```\n"
	lines := renderRichText(source, 80, "assistant")
	plain := stripANSI(strings.Join(lines, "\n"))
	for _, expected := range []string{
		"Overview",
		"#include <stdio.h>",
		"- Item 1",
		"    - Nested 1.1",
		"    - Nested 1.2",
		"- Item 2",
		"1. alpha",
		"2. beta",
		"3. gamma",
		"- [ ] beep",
		"- [x] boop",
		"Alice",
		"Bob",
		"func main() {}",
	} {
		if !strings.Contains(plain, expected) {
			t.Fatalf("missing %q in:\n%s", expected, plain)
		}
	}
	if strings.Contains(plain, "include <stdio.h>") && !strings.Contains(plain, "#include <stdio.h>") {
		t.Fatal("C include was rendered as a heading")
	}
}

func TestRenderRichTextPreservesIndentedSections(t *testing.T) {
	source := "intro\n    aligned block\n    still aligned"
	plain := stripANSI(strings.Join(renderRichText(source, 40, "assistant"), "\n"))
	if !strings.Contains(plain, "aligned block") || strings.Count(plain, "aligned") != 2 {
		t.Fatalf("indented section lost: %q", plain)
	}
	if !strings.Contains(plain, "│") {
		t.Fatal("indented section should render as a code block")
	}
}

func TestPadKeepsANSIWhenTruncating(t *testing.T) {
	styled := accent + bold + strings.Repeat("section-title", 8) + reset
	got := pad(styled, 20)
	if lipgloss.Width(got) != 20 {
		t.Fatalf("width = %d", lipgloss.Width(got))
	}
	if !strings.Contains(got, "\x1b[") {
		t.Fatalf("styles stripped: %q", got)
	}
}

func TestFitSidebarKeepsWorkspaceSection(t *testing.T) {
	var sidebar []string
	sidebar = append(sidebar, "title\tSession", "section\tContext", "meta\t120 tokens")
	for index := 0; index < 30; index++ {
		sidebar = append(sidebar, "todo\topen\tstep")
	}
	sidebar = append(sidebar, "", "section\tWorkspace", "meta\tmain", "path\t~/src", "", "brand\tGoshCoder")
	lines := renderSidebar(sidebar, 32, 12)
	plain := stripANSI(strings.Join(lines, "\n"))
	if !strings.Contains(plain, "Workspace") || !strings.Contains(plain, "GoshCoder") {
		t.Fatalf("workspace section clipped: %q", plain)
	}
	if !strings.Contains(plain, "…") {
		t.Fatal("overflowing sidebar should show an ellipsis")
	}
}

func TestRenderFrameClampsTranscriptScroll(t *testing.T) {
	messages := make([]Message, 0, 20)
	for index := 0; index < 20; index++ {
		messages = append(messages, Message{Role: "assistant", Text: "line"})
	}
	_, _, _, scroll := renderFrame(Frame{Messages: messages, Scroll: 1000}, 80, 16)
	if scroll > 40 {
		t.Fatalf("scroll was not clamped: %d", scroll)
	}
}
