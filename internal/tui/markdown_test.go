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

func TestRenderRichTextLooseOrderedSections(t *testing.T) {
	source := `1. Lorem ipsum dolor sit amet.

   Ut enim ad minim veniam.

2. Duis aute irure dolor.

   Excepteur sint occaecat cupidatat.

3. Beep boop`
	plain := strings.Split(stripANSI(strings.Join(renderRichText(source, 80, "assistant"), "\n")), "\n")
	for i := range plain {
		plain[i] = strings.TrimRight(plain[i], " ")
	}
	want := []string{
		"1. Lorem ipsum dolor sit amet.",
		"",
		"   Ut enim ad minim veniam.",
		"",
		"2. Duis aute irure dolor.",
		"",
		"   Excepteur sint occaecat cupidatat.",
		"",
		"3. Beep boop",
	}
	if strings.Join(plain, "\n") != strings.Join(want, "\n") {
		t.Fatalf("loose list =\n%s", strings.Join(plain, "\n"))
	}
}

func TestRenderRichTextKeepsNumberingAcrossCodeFences(t *testing.T) {
	source := "1. First item\n\n```typescript\n// code block\n```\n\n2. Second item\n\n```typescript\n// another code block\n```\n\n3. Third item"
	plain := stripANSI(strings.Join(renderRichText(source, 80, "assistant"), "\n"))
	var numbered []string
	for _, line := range strings.Split(plain, "\n") {
		trim := strings.TrimSpace(line)
		if len(trim) > 2 && trim[0] >= '1' && trim[0] <= '9' && trim[1] == '.' {
			numbered = append(numbered, trim)
		}
	}
	if len(numbered) != 3 || !strings.HasPrefix(numbered[0], "1.") || !strings.HasPrefix(numbered[1], "2.") || !strings.HasPrefix(numbered[2], "3.") {
		t.Fatalf("numbered items = %#v\n%s", numbered, plain)
	}
}

func TestRenderRichTextSetextAndIndentedStart(t *testing.T) {
	source := "Overview\n========\n\n    func main() {}\n"
	plain := stripANSI(strings.Join(renderRichText(source, 80, "assistant"), "\n"))
	if !strings.Contains(plain, "Overview") || strings.Contains(plain, "====") {
		t.Fatalf("setext heading lost: %q", plain)
	}
	if !strings.Contains(plain, "func main() {}") || !strings.Contains(plain, "│") {
		t.Fatalf("leading indented code lost: %q", plain)
	}
}

func TestRenderRichTextWrapsListHangingIndent(t *testing.T) {
	plain := make([]string, 0, 2)
	for _, line := range renderRichText("- alpha beta gamma delta epsilon", 20, "assistant") {
		plain = append(plain, strings.TrimRight(stripANSI(line), " "))
	}
	want := []string{"- alpha beta gamma", "  delta epsilon"}
	if strings.Join(plain, "\n") != strings.Join(want, "\n") {
		t.Fatalf("wrap =\n%s", strings.Join(plain, "\n"))
	}
}

func TestRenderRichTextNestedQuoteAndListQuote(t *testing.T) {
	plain := stripANSI(strings.Join(renderRichText("> outer\n>> inner\n\n- > quoted item", 80, "assistant"), "\n"))
	if !strings.Contains(plain, "│ │ inner") && !strings.Contains(plain, "│ inner") {
		t.Fatalf("nested quote missing: %q", plain)
	}
	if !strings.Contains(plain, "- │ quoted item") {
		t.Fatalf("list quote missing: %q", plain)
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
