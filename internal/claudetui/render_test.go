package claudetui

import (
	"strings"
	"testing"
)

func TestHeaderFitsWidth(t *testing.T) {
	for _, width := range []int{20, 40, 80, 120} {
		lines := Header(width, "1.0", "anthropic/claude", "high", "/work/project", false)
		if len(lines) == 0 {
			t.Fatalf("width %d produced no header", width)
		}
		for _, line := range lines {
			if displayWidth(line) > width {
				t.Fatalf("width %d: line is %d cells: %q", width, displayWidth(line), line)
			}
		}
	}
}

func TestWideHeaderHasTipsAndNarrowDoesNot(t *testing.T) {
	wide := strings.Join(Header(100, "1", "p/m", "off", "/x", false), "\n")
	if !strings.Contains(wide, "Getting started") || !strings.Contains(wide, "/planner") {
		t.Fatalf("wide header = %q", wide)
	}
	narrow := strings.Join(Header(40, "1", "p/m", "off", "/x", false), "\n")
	if strings.Contains(narrow, "Getting started") {
		t.Fatalf("narrow header unexpectedly has sidebar: %q", narrow)
	}
}

func TestHeaderWithSessionInfo(t *testing.T) {
	info := SessionInfo{Model: "anthropic/claude", ContextUsed: 12500, ContextLimit: 200000, Cost: 0.1234, Messages: 8, Tools: 6, ChangedFiles: 3, Branch: "main", Mode: "planning", Thinking: "high"}
	header := strings.Join(HeaderWithInfo(100, "1", info.Model, "high", "/x", false, &info), "\n")
	for _, expected := range []string{"Session", "Context", "12.5k/200k · 6%", "$0.1234", "3 changed", "Branch", "main", "planning · high"} {
		if !strings.Contains(header, expected) {
			t.Fatalf("header lacks %q: %s", expected, header)
		}
	}
}

func TestStandaloneSidebarFitsWidth(t *testing.T) {
	lines := Sidebar(32, SessionInfo{Model: "provider/a-very-long-model-name", Branch: "feature/long-name"}, false)
	for _, line := range lines {
		if displayWidth(line) > 32 {
			t.Fatalf("line is too wide: %q", line)
		}
	}
}

func TestInputPromptUsesHalfOpenRoundedBox(t *testing.T) {
	prompt := InputPrompt(20, false)
	if !strings.Contains(prompt, "╭") || !strings.Contains(prompt, "╮") || !strings.Contains(prompt, "╰─❯") {
		t.Fatalf("prompt = %q", prompt)
	}
	if strings.Contains(prompt, "│") {
		t.Fatalf("prompt has vertical sides: %q", prompt)
	}
}
