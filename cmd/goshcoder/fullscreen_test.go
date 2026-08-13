package main

import (
	"path/filepath"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/ralph"
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

	tuiModel.setInput("hello")
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeySpace, Runes: []rune{' '}})
	if got := string(tuiModel.editor.input); got != "hello " {
		t.Fatalf("space input = %q", got)
	}

	tuiModel.setInput("/mo")
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyTab})
	if got := string(tuiModel.editor.input); got != "/model " {
		t.Fatalf("tab completion = %q", got)
	}
}

func TestBubbleTeaEditorMultilineNavigation(t *testing.T) {
	model := &llm.Model{ID: "chat", Provider: "vendor"}
	tuiModel := &fullscreenModel{session: fullscreenTestSession(model, llm.ThinkingOff)}
	tuiModel.setInput("short\na longer line\nlast")
	tuiModel.editor.cursor = len([]rune("short\na long"))
	if !moveEditorCursorVertical(&tuiModel.editor, -1) || tuiModel.editor.cursor != 5 {
		t.Fatalf("up cursor = %d", tuiModel.editor.cursor)
	}
	if !moveEditorCursorVertical(&tuiModel.editor, 1) || tuiModel.editor.cursor != len([]rune("short\na lon")) {
		t.Fatalf("down cursor = %d", tuiModel.editor.cursor)
	}
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyHome})
	if tuiModel.editor.cursor != len([]rune("short\n")) {
		t.Fatalf("home cursor = %d", tuiModel.editor.cursor)
	}
	tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyEnd})
	if tuiModel.editor.cursor != len([]rune("short\na longer line")) {
		t.Fatalf("end cursor = %d", tuiModel.editor.cursor)
	}
}

func TestOmniNetworkCommandsRunOffTheFullscreenUpdateLoop(t *testing.T) {
	for _, command := range []string{"/omni status", "/omni sync", "/omni dashboard"} {
		if !fullscreenCommandRunsAsync(command) {
			t.Fatalf("%s should run asynchronously", command)
		}
	}
	if fullscreenCommandRunsAsync("/omni setup") {
		t.Fatal("setup uses Tea ExecProcess and should not use the captured command path")
	}
}

func TestFullscreenSuggestionsIncludeNativeOmniAndBTWCommands(t *testing.T) {
	session := &session{}
	omni := fullscreenSuggestionsWithModels(session, "/omni s", nil)
	if len(omni) != 3 || omni[0].label != "status" || omni[1].label != "sync" || omni[2].label != "setup" {
		t.Fatalf("omni suggestions = %#v", omni)
	}
	btw := fullscreenSuggestionsWithModels(session, "/btw r", nil)
	if len(btw) != 1 || btw[0].label != "resume" || btw[0].execute {
		t.Fatalf("btw suggestions = %#v", btw)
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

func TestModelPickerFiltersAndMarksCurrentModel(t *testing.T) {
	model := &llm.Model{ID: "alpha", Name: "Alpha", Provider: "vendor"}
	session := fullscreenTestSession(model, llm.ThinkingOff)
	choices := []fullscreenModelChoice{
		{ref: "vendor/alpha", name: "Alpha", provider: "vendor", current: true},
		{ref: "vendor/beta-code", name: "Beta Coder", provider: "vendor", context: 200_000, reasoning: true},
	}
	items := fullscreenSuggestionsWithModels(session, "/model beta", choices)
	if len(items) != 1 || items[0].label != "Beta Coder" || items[0].value != "/model vendor/beta-code" {
		t.Fatalf("model suggestions = %#v", items)
	}
	items = fullscreenSuggestionsWithModels(session, "/model ", choices)
	if len(items) != 2 || !items[0].current {
		t.Fatalf("unfiltered model suggestions = %#v", items)
	}

	tuiModel := &fullscreenModel{session: session, models: choices, editor: fullscreenEditor{input: []rune("/mo"), cursor: 3}}
	if command := tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyEnter}); command != nil {
		t.Fatal("opening the model picker should not execute a command")
	}
	if got := string(tuiModel.editor.input); got != "/model " {
		t.Fatalf("model picker input = %q", got)
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

	tuiModel.setInput("/thinking low")
	if command := tuiModel.submitInput(true); command != nil {
		t.Fatal("slash commands should execute immediately while streaming")
	}
	if got := session.agent.State().ThinkingLevel; got != llm.ThinkingLow {
		t.Fatalf("streaming slash command set thinking = %q", got)
	}
}

func TestFullscreenActivityUpdatesDuringTools(t *testing.T) {
	model := &llm.Model{ID: "chat", Provider: "vendor"}
	tuiModel := &fullscreenModel{session: fullscreenTestSession(model, llm.ThinkingOff), activity: "Ready"}
	tuiModel.applyAgentEvent(agent.Event{Type: agent.EventToolExecutionStart, ToolName: "edit"})
	if tuiModel.activity != "Running edit" || !strings.Contains(tuiModel.recentTool, "running") {
		t.Fatalf("start activity = %q, %q", tuiModel.activity, tuiModel.recentTool)
	}
	tuiModel.applyAgentEvent(agent.Event{Type: agent.EventToolExecutionEnd, ToolName: "edit"})
	if tuiModel.activity != "edit complete" || !strings.Contains(tuiModel.recentTool, "complete") {
		t.Fatalf("end activity = %q, %q", tuiModel.activity, tuiModel.recentTool)
	}
}

func TestFullscreenTranscriptKeepsToolActivityCompact(t *testing.T) {
	messages := []agent.Message{
		llm.AssistantMessage{Content: []llm.ContentBlock{
			llm.TextContent{Type: "text", Text: "Working on it."},
			llm.ToolCall{Type: "toolCall", ID: "call-1", Name: "edit", Arguments: map[string]any{"path": "secret/file.go"}},
		}},
		llm.ToolResultMessage{Role: "toolResult", ToolName: "edit", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "secret/file.go changed"}}},
	}
	visible := fullscreenMessages(messages)
	if len(visible) != 2 || visible[0].Role != "assistant" || visible[1].Role != "tool" || !strings.Contains(visible[1].Title, "secret/file.go") {
		t.Fatalf("visible transcript = %#v", visible)
	}

	messages = append(messages, llm.ToolResultMessage{Role: "toolResult", ToolName: "edit", IsError: true, Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "permission denied"}}})
	visible = fullscreenMessages(messages)
	if len(visible) != 3 || visible[2].Role != "tool" || !visible[2].IsError {
		t.Fatalf("tool error missing from transcript: %#v", visible)
	}
}

func TestBubbleTeaCommandPaletteShowsAdditiveProviderLogin(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	model := &llm.Model{ID: "chat", Provider: "vendor"}
	session := fullscreenTestSession(model, llm.ThinkingOff)

	root := fullscreenSuggestions(session, "/log")
	if len(root) != 1 || root[0].label != "/login" || root[0].execute || root[0].value != "/login " {
		t.Fatalf("login root suggestion = %#v", root)
	}
	oauth := fullscreenSuggestions(session, "/login anthropic")
	if len(oauth) != 1 || oauth[0].label != "anthropic" || !oauth[0].execute || !strings.Contains(oauth[0].description, "OAuth") {
		t.Fatalf("Anthropic login suggestion = %#v", oauth)
	}
	apiKey := fullscreenSuggestions(session, "/login deepseek")
	if len(apiKey) != 1 || apiKey[0].label != "deepseek" || !strings.Contains(apiKey[0].description, "API key") {
		t.Fatalf("DeepSeek login suggestion = %#v", apiKey)
	}
	if title := fullscreenSuggestionTitle("/login "); !strings.Contains(title, "existing logins are kept") {
		t.Fatalf("login picker title = %q", title)
	}
}

func TestBubbleTeaCommandPaletteShowsRalphControls(t *testing.T) {
	model := &llm.Model{ID: "chat", Provider: "vendor"}
	session := fullscreenTestSession(model, llm.ThinkingOff)
	session.loops = ralph.NewStore(t.TempDir(), "test-session")

	root := fullscreenSuggestions(session, "/")
	found := false
	for _, item := range root {
		if item.label == "/ralph" {
			found = true
			if item.execute || item.value != "/ralph " {
				t.Fatalf("Ralph root suggestion = %#v", item)
			}
		}
	}
	if !found {
		t.Fatalf("Ralph missing from root suggestions: %#v", root)
	}

	controls := fullscreenSuggestions(session, "/ralph ")
	if len(controls) != 5 || controls[0].label != "start" || controls[0].value != "/ralph start " {
		t.Fatalf("Ralph controls = %#v", controls)
	}
	if !fullscreenCommandRunsAsync("/ralph start audit improve test coverage") {
		t.Fatal("Ralph start must not block the Bubble Tea update loop")
	}
}

func TestBubbleTeaCommandPaletteRunsPlanner(t *testing.T) {
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
	if len(items) == 0 || items[0].label != "/planner" {
		t.Fatalf("suggestions = %#v", items)
	}
	if command := tuiModel.handleTeaKey(tea.KeyMsg{Type: tea.KeyEnter}); command != nil {
		t.Fatal("Planner toggle should complete immediately")
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
