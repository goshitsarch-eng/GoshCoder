package main

import (
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/tools"
)

func TestBubbleTeaEditorUnicodeNavigation(t *testing.T) {
	model := &llm.Model{ID: "chat", Provider: "vendor"}
	tuiModel := &fullscreenModel{session: fullscreenTestSession(model, llm.ThinkingOff)}
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("你好")})
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyLeft})
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyBackspace})
	if got := string(tuiModel.editor.input); got != "好" || tuiModel.editor.cursor != 0 {
		t.Fatalf("editor = %q at %d", got, tuiModel.editor.cursor)
	}

	tuiModel.setInput("/pla")
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyTab})
	if got := string(tuiModel.editor.input); got != "/plannotator" {
		t.Fatalf("tab completion = %q", got)
	}
}

func TestThinkingSuggestionsFollowModelCapabilities(t *testing.T) {
	model := &llm.Model{
		ID: "reasoner", Provider: "vendor", Reasoning: true,
		ThinkingLevelMap: llm.ThinkingLevelMap{
			llm.ThinkingMinimal: nil,
			llm.ThinkingMedium:  nil,
		},
	}
	session := fullscreenTestSession(model, llm.ThinkingHigh)
	items := fullscreenSuggestions(session, "/thinking ")
	var labels []string
	for _, item := range items {
		labels = append(labels, item.label)
	}
	if got, want := strings.Join(labels, ","), "off,low,high"; got != want {
		t.Fatalf("levels = %q, want %q", got, want)
	}
	if !items[2].current {
		t.Fatal("current thinking level is not marked")
	}

	nonReasoning := fullscreenTestSession(&llm.Model{ID: "chat", Provider: "vendor"}, llm.ThinkingOff)
	items = fullscreenSuggestions(nonReasoning, "/thinking ")
	if len(items) != 1 || items[0].label != "off" {
		t.Fatalf("non-reasoning levels = %#v", items)
	}
}

func TestBubbleTeaCommandPaletteAcceptsThinkingChoice(t *testing.T) {
	model := &llm.Model{ID: "reasoner", Provider: "vendor", Reasoning: true}
	session := fullscreenTestSession(model, llm.ThinkingOff)
	tuiModel := &fullscreenModel{session: session, editor: fullscreenEditor{input: []rune("/"), cursor: 1}}
	items := tuiModel.suggestions()
	for index, item := range items {
		if item.label == "/thinking" {
			tuiModel.editor.suggestion = index
			break
		}
	}
	if command := tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyEnter}); command != nil {
		t.Fatal("opening the thinking submenu should not run a command")
	}
	if got := string(tuiModel.editor.input); got != "/thinking " {
		t.Fatalf("input = %q", got)
	}

	items = tuiModel.suggestions()
	for index, item := range items {
		if item.label == "high" {
			tuiModel.editor.suggestion = index
			break
		}
	}
	if command := tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyEnter}); command != nil {
		t.Fatal("thinking command should complete immediately")
	}
	if got := session.agent.State().ThinkingLevel; got != llm.ThinkingHigh {
		t.Fatalf("thinking = %q", got)
	}
}

func TestBubbleTeaCommandPaletteRunsPlannotator(t *testing.T) {
	workspace, err := tools.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer workspace.Close()
	manager, err := plannotator.New(workspace.Root, filepath.Join(t.TempDir(), "plan.json"), nil)
	if err != nil {
		t.Fatal(err)
	}
	model := &llm.Model{ID: "chat", Provider: "vendor"}
	session := fullscreenTestSession(model, llm.ThinkingOff)
	session.workspace = workspace
	session.plan = manager
	session.normalTools = workspace.All()
	session.agent.SetTools(session.normalTools)

	tuiModel := &fullscreenModel{session: session, editor: fullscreenEditor{input: []rune("/pla"), cursor: 4}}
	items := tuiModel.suggestions()
	if len(items) == 0 || items[0].label != "/plannotator" {
		t.Fatalf("suggestions = %#v", items)
	}
	if command := tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyEnter}); command != nil {
		t.Fatal("Plannotator toggle should complete immediately")
	}
	if phase := manager.State().Phase; phase != plannotator.PhasePlanning {
		t.Fatalf("phase = %q", phase)
	}
}

func fullscreenTestSession(model *llm.Model, thinking llm.ModelThinkingLevel) *session {
	return &session{
		model: model,
		agent: agent.NewAgent(agent.AgentOptions{InitialState: &agent.InitialState{
			Model: *model, ThinkingLevel: thinking,
		}}),
	}
}

func TestFullscreenEditorHistory(t *testing.T) {
	editor := fullscreenEditor{history: []string{"first", "second"}, historyIndex: -1, input: []rune("draft")}
	editor.cursor = len(editor.input)
	editorHistory(&editor, -1)
	if string(editor.input) != "second" {
		t.Fatalf("input = %q", editor.input)
	}
	editorHistory(&editor, 1)
	if string(editor.input) != "draft" {
		t.Fatalf("input = %q", editor.input)
	}
}
