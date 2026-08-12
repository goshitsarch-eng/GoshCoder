package main

// The interactive screen is driven by Bubble Tea. Agent work stays in the
// existing goroutines and is reported back as Tea messages, keeping terminal
// input, resize handling, rendering, and command-palette state in one loop.

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"

	"goshcoder/internal/agent"
	"goshcoder/internal/claudetui"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/tui"
)

type fullscreenSuggestion struct {
	label       string
	description string
	value       string
	execute     bool
	current     bool
}

type fullscreenModelChoice struct {
	ref       string
	name      string
	provider  string
	context   int
	reasoning bool
	current   bool
}

type fullscreenAgentEvent struct{ event agent.Event }
type fullscreenTurnFinished struct{ err error }
type fullscreenCommandFinished struct{ result fullscreenResult }
type fullscreenInfoRefreshed struct{ info claudetui.SessionInfo }
type fullscreenTick time.Time

type fullscreenModel struct {
	session       *session
	editor        fullscreenEditor
	notices       []tui.Message
	info          claudetui.SessionInfo
	activity      string
	activityAt    time.Time
	width         int
	height        int
	busyCommand   bool
	spin          int
	updates       <-chan fullscreenAgentEvent
	lifecycle     <-chan fullscreenAgentEvent
	models        []fullscreenModelChoice
	recentTool    string
	toolsExpanded bool
	hideThinking  bool
}

func newFullscreenModel(session *session, updates, lifecycle <-chan fullscreenAgentEvent) *fullscreenModel {
	model := &fullscreenModel{
		session:   session,
		editor:    fullscreenEditor{historyIndex: -1},
		info:      session.sessionInfo(),
		activity:  "Ready",
		width:     80,
		height:    24,
		updates:   updates,
		lifecycle: lifecycle,
		models:    availableFullscreenModels(session),
	}
	for _, tool := range session.agent.State().Tools {
		if tool.Name == "bash" {
			model.notices = append(model.notices, tui.Message{
				Role: "Notice",
				Text: "Coding tools are active in " + session.workspaceRoot() + ". Commands run with your user privileges.",
			})
			break
		}
	}
	return model
}

func runFullscreenChat(session *session) error {
	// Streaming deltas are coalesced, while lifecycle/tool events use a
	// separate queue so a delta flood can never hide activity changes.
	updates := make(chan fullscreenAgentEvent, 1)
	lifecycle := make(chan fullscreenAgentEvent, 64)
	unsubscribe := session.agent.Subscribe(func(_ context.Context, event agent.Event) {
		message := fullscreenAgentEvent{event: event}
		if event.Type == agent.EventMessageUpdate {
			select {
			case updates <- message:
			default:
			}
			return
		}
		lifecycle <- message
	})
	defer unsubscribe()

	program := tea.NewProgram(
		newFullscreenModel(session, updates, lifecycle),
		tea.WithAltScreen(),
		tea.WithInput(os.Stdin),
		tea.WithOutput(os.Stderr),
		tea.WithMouseCellMotion(),
	)
	_, err := program.Run()
	return err
}

func (model *fullscreenModel) Init() tea.Cmd {
	return tea.Batch(tea.HideCursor, waitForFullscreenEvent(model.updates, model.lifecycle), fullscreenTickCommand())
}

func waitForFullscreenEvent(updates, lifecycle <-chan fullscreenAgentEvent) tea.Cmd {
	return func() tea.Msg {
		select {
		case event := <-lifecycle:
			return event
		case event := <-updates:
			return event
		}
	}
}

func fullscreenTickCommand() tea.Cmd {
	return tea.Tick(120*time.Millisecond, func(now time.Time) tea.Msg { return fullscreenTick(now) })
}

func refreshFullscreenInfo(session *session) tea.Cmd {
	return func() tea.Msg { return fullscreenInfoRefreshed{info: session.sessionInfo()} }
}

func (model *fullscreenModel) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	switch value := message.(type) {
	case tea.WindowSizeMsg:
		model.width, model.height = value.Width, value.Height

	case fullscreenAgentEvent:
		model.applyAgentEvent(value.event)
		model.refreshRealtimeInfo()
		wait := waitForFullscreenEvent(model.updates, model.lifecycle)
		if value.event.Type == agent.EventToolExecutionEnd || value.event.Type == agent.EventAgentEnd {
			return model, tea.Batch(wait, refreshFullscreenInfo(model.session))
		}
		return model, wait

	case fullscreenTurnFinished:
		if value.err != nil {
			model.notices = appendNotice(model.notices, "Error", value.err.Error())
		}
		model.info = model.session.sessionInfo()
		model.activity = "Ready"
		model.activityAt = time.Time{}

	case fullscreenCommandFinished:
		if model.applyCommandResult(value.result) {
			return model, tea.Quit
		}

	case fullscreenInfoRefreshed:
		model.info = value.info

	case fullscreenTick:
		model.spin = (model.spin + 1) % 8
		return model, fullscreenTickCommand()

	case tea.KeyMsg:
		return model, model.handleTeaKey(value)

	case tea.MouseMsg:
		switch value.Button {
		case tea.MouseButtonWheelUp:
			model.editor.scroll += 3
		case tea.MouseButtonWheelDown:
			model.editor.scroll = max(0, model.editor.scroll-3)
		}
	}
	return model, nil
}

func (model *fullscreenModel) applyAgentEvent(event agent.Event) {
	switch event.Type {
	case agent.EventMessageUpdate:
		model.activity = "Composing response"
		if model.activityAt.IsZero() {
			model.activityAt = time.Now()
		}
	case agent.EventToolExecutionStart:
		model.activity = "Running " + event.ToolName
		model.recentTool = "● " + event.ToolName + " running"
	case agent.EventToolExecutionEnd:
		if event.IsError {
			model.activity = event.ToolName + " failed"
			model.recentTool = "× " + event.ToolName + " failed"
		} else {
			model.activity = event.ToolName + " complete"
			model.recentTool = "✓ " + event.ToolName + " complete"
		}
	case agent.EventAgentEnd:
		model.activity = "Ready"
		model.activityAt = time.Time{}
	}
}

func (model *fullscreenModel) refreshRealtimeInfo() {
	fresh := model.session.sessionInfoRealtime()
	fresh.Branch = model.info.Branch
	fresh.ChangedFiles = model.info.ChangedFiles
	fresh.Changes = append([]claudetui.ChangedFile(nil), model.info.Changes...)
	model.info = fresh
}

func (model *fullscreenModel) View() string {
	state := model.session.agent.State()
	transcript := state.Messages
	if state.StreamingMessage != nil {
		transcript = append(append([]agent.Message(nil), transcript...), state.StreamingMessage)
	}
	messages := append([]tui.Message(nil), fullscreenMessages(transcript)...)
	messages = append(messages, model.notices...)

	items := model.suggestions()
	model.clampSuggestion(len(items))
	display := make([]string, 0, len(items))
	for _, item := range items {
		description := item.description
		if item.current {
			description = "CURRENT  ·  " + description
		}
		display = append(display, item.label+"\t"+description)
	}

	status := model.activity
	if state.IsStreaming || model.busyCommand {
		spinner := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"}
		status = spinner[model.spin] + "  " + status
		if !model.activityAt.IsZero() {
			status += fmt.Sprintf("  %.1fs", time.Since(model.activityAt).Seconds())
		}
	}

	var planItems []plannotator.ChecklistItem
	if model.session.plan != nil {
		planItems = model.session.plan.State().Items
	}
	frame := tui.Frame{
		Title:              fmt.Sprintf("v%s  ·  %s/%s", Version, state.Model.Provider, state.Model.ID),
		Messages:           messages,
		Sidebar:            fullscreenSidebar(model.info, model.session.workspaceRoot(), model.activity, len(state.PendingToolCalls), model.recentTool, planItems),
		Input:              model.editor.input,
		Cursor:             model.editor.cursor,
		Status:             status,
		Streaming:          state.IsStreaming,
		Scroll:             model.editor.scroll,
		Suggestions:        display,
		SelectedSuggestion: model.editor.suggestion,
		SuggestionTitle:    fullscreenSuggestionTitle(string(model.editor.input)),
		Thinking:           state.ThinkingLevel,
		Mode:               model.info.Mode,
		VirtualCursor:      true,
		ToolsExpanded:      model.toolsExpanded,
		HideThinking:       model.hideThinking,
	}
	return tui.View(frame, model.width, model.height)
}

func (model *fullscreenModel) handleTeaKey(key tea.KeyMsg) tea.Cmd {
	streaming := model.session.agent.State().IsStreaming
	switch key.String() {
	case "ctrl+c":
		if len(model.editor.input) > 0 {
			model.clearInput()
			return nil
		}
		if streaming {
			model.session.agent.Abort()
			model.activity = "Aborting"
			return nil
		}
		return tea.Quit

	case "ctrl+d":
		if len(model.editor.input) == 0 {
			if streaming {
				model.session.agent.Abort()
				model.activity = "Aborting"
				return nil
			}
			return tea.Quit
		}
		model.deleteAtCursor()

	case "esc":
		if streaming {
			model.session.agent.Abort()
			model.activity = "Aborting"
		} else if len(model.editor.input) > 0 {
			model.clearInput()
		}

	case "shift+tab":
		cycleFullscreenThinking(model.session)
		model.info = model.session.sessionInfo()
		model.activity = "Thinking set to " + model.session.agent.State().ThinkingLevel

	case "ctrl+l":
		model.setInput("/model ")

	case "ctrl+p":
		model.cycleModel(1)

	case "ctrl+shift+p", "shift+ctrl+p":
		model.cycleModel(-1)

	case "ctrl+o":
		model.toolsExpanded = !model.toolsExpanded
		if model.toolsExpanded {
			model.activity = "Tool output expanded"
		} else {
			model.activity = "Tool output collapsed"
		}

	case "ctrl+t":
		model.hideThinking = !model.hideThinking
		if model.hideThinking {
			model.activity = "Thinking collapsed"
		} else {
			model.activity = "Thinking expanded"
		}

	case "alt+enter":
		return model.submitFollowUp(streaming)

	case "up":
		if items := model.suggestions(); len(items) > 0 {
			model.editor.suggestion = max(0, model.editor.suggestion-1)
		} else if !moveEditorCursorVertical(&model.editor, -1) {
			editorHistory(&model.editor, -1)
		}

	case "down":
		if items := model.suggestions(); len(items) > 0 {
			model.editor.suggestion = min(len(items)-1, model.editor.suggestion+1)
		} else if !moveEditorCursorVertical(&model.editor, 1) {
			editorHistory(&model.editor, 1)
		}

	case "left", "ctrl+b":
		model.editor.cursor = max(0, model.editor.cursor-1)
	case "right", "ctrl+f":
		model.editor.cursor = min(len(model.editor.input), model.editor.cursor+1)
	case "alt+left", "ctrl+left", "alt+b":
		model.moveWord(-1)
	case "alt+right", "ctrl+right", "alt+f":
		model.moveWord(1)
	case "home", "ctrl+a":
		model.editor.cursor = editorLineStart(model.editor.input, model.editor.cursor)
	case "end", "ctrl+e":
		model.editor.cursor = editorLineEnd(model.editor.input, model.editor.cursor)
	case "pgup":
		model.editor.scroll += 10
	case "pgdown":
		model.editor.scroll = max(0, model.editor.scroll-10)

	case "shift+enter", "ctrl+j":
		model.insertRunes([]rune{'\n'})

	case "backspace":
		if model.editor.cursor > 0 {
			model.editor.input = append(model.editor.input[:model.editor.cursor-1], model.editor.input[model.editor.cursor:]...)
			model.editor.cursor--
			model.editor.suggestion = 0
		}
	case "delete":
		model.deleteAtCursor()
	case "ctrl+k":
		model.editor.input = model.editor.input[:model.editor.cursor]
		model.editor.suggestion = 0
	case "ctrl+u":
		model.editor.input = append([]rune(nil), model.editor.input[model.editor.cursor:]...)
		model.editor.cursor = 0
		model.editor.suggestion = 0
	case "ctrl+w":
		model.deletePreviousWord()

	case "tab":
		if items := model.suggestions(); len(items) > 0 {
			model.setInput(items[model.selectedSuggestion(len(items))].value)
		}

	case "enter":
		if items := model.suggestions(); len(items) > 0 {
			item := items[model.selectedSuggestion(len(items))]
			if !item.execute {
				model.setInput(item.value)
				return nil
			}
			model.setInput(item.value)
		}
		return model.submitInput(streaming)

	default:
		// Bubble Tea emits a dedicated KeySpace event for a single space,
		// while pasted text (including spaces) arrives as KeyRunes.
		if key.Type == tea.KeyRunes || key.Type == tea.KeySpace {
			var input []rune
			for _, r := range key.Runes {
				if !unicode.IsControl(r) {
					input = append(input, r)
				}
			}
			model.insertRunes(input)
		}
	}
	return nil
}

func (model *fullscreenModel) cycleModel(direction int) {
	if len(model.models) < 2 {
		model.activity = "Only one model is available"
		return
	}
	current := 0
	for index, choice := range model.models {
		if choice.current {
			current = index
			break
		}
	}
	next := (current + direction + len(model.models)) % len(model.models)
	if err := model.session.setModel(model.models[next].ref); err != nil {
		model.notices = appendNotice(model.notices, "Error", err.Error())
		return
	}
	for index := range model.models {
		model.models[index].current = index == next
	}
	model.info = model.session.sessionInfo()
	model.activity = "Model set to " + model.models[next].ref
}

func (model *fullscreenModel) submitFollowUp(streaming bool) tea.Cmd {
	if !streaming {
		return model.submitInput(false)
	}
	prompt := strings.TrimSpace(string(model.editor.input))
	if prompt == "" {
		return nil
	}
	model.editor.history = append(model.editor.history, prompt)
	model.editor.historyIndex, model.editor.draft = -1, ""
	model.clearInput()
	if expanded, ok, err := model.session.expandResourceInput(prompt); err != nil {
		model.notices = appendNotice(model.notices, "Error", err.Error())
		return nil
	} else if ok {
		prompt = expanded
	}
	model.session.agent.FollowUp(userMessage(prompt))
	model.activity = "Follow-up message queued"
	return nil
}

func (model *fullscreenModel) submitInput(streaming bool) tea.Cmd {
	prompt := strings.TrimSpace(string(model.editor.input))
	if prompt == "" {
		return nil
	}
	if model.busyCommand && !streaming {
		model.activity = "Wait for the current command to finish"
		return nil
	}
	model.editor.history = append(model.editor.history, prompt)
	model.editor.historyIndex, model.editor.draft = -1, ""
	model.clearInput()

	if expanded, ok, expandErr := model.session.expandResourceInput(prompt); ok {
		if expandErr != nil {
			model.notices = appendNotice(model.notices, "Error", expandErr.Error())
			model.activity = "Ready"
			return nil
		}
		prompt = expanded
	} else if expandErr != nil {
		model.notices = appendNotice(model.notices, "Error", expandErr.Error())
		model.activity = "Ready"
		return nil
	}

	if strings.HasPrefix(prompt, "/") {
		if prompt == "/exit" || prompt == "/quit" {
			return tea.Quit
		}
		model.activity = "Running " + strings.Fields(prompt)[0]
		if fullscreenCommandRunsAsync(prompt) {
			model.busyCommand = true
			model.activityAt = time.Now()
			return func() tea.Msg {
				return fullscreenCommandFinished{result: runFullscreenCommand(model.session, prompt)}
			}
		}
		if model.applyCommandResult(runFullscreenCommand(model.session, prompt)) {
			return tea.Quit
		}
		return nil
	}
	if streaming {
		model.session.agent.Steer(userMessage(prompt))
		model.activity = "Steering message queued"
		return nil
	}

	model.activity = "Sending message"
	model.activityAt = time.Now()
	return func() tea.Msg { return fullscreenTurnFinished{err: model.session.runTurn(prompt)} }
}

func (model *fullscreenModel) applyCommandResult(result fullscreenResult) bool {
	model.busyCommand = false
	if result.output != "" {
		model.notices = appendNotice(model.notices, "Command", result.output)
	}
	if result.err != nil {
		model.notices = appendNotice(model.notices, "Error", result.err.Error())
	}
	model.info = model.session.sessionInfo()
	state := model.session.agent.State()
	currentRef := state.Model.Provider + "/" + state.Model.ID
	for index := range model.models {
		model.models[index].current = model.models[index].ref == currentRef
	}
	if state.IsStreaming {
		model.activity = "Composing response"
		if model.activityAt.IsZero() {
			model.activityAt = time.Now()
		}
	} else {
		model.activity = "Ready"
		model.activityAt = time.Time{}
	}
	return result.exit
}

func fullscreenCommandRunsAsync(prompt string) bool {
	fields := strings.Fields(prompt)
	if len(fields) == 0 {
		return false
	}
	command := fields[0]
	return command == "/compact" || command == "/plannotator-review" || command == "/plannotator-annotate" || command == "/plannotator-last" ||
		(command == "/ralph" && len(fields) > 1 && fields[1] == "start")
}

func (model *fullscreenModel) suggestions() []fullscreenSuggestion {
	return fullscreenSuggestionsWithModels(model.session, string(model.editor.input), model.models)
}

func (model *fullscreenModel) selectedSuggestion(count int) int {
	model.clampSuggestion(count)
	return model.editor.suggestion
}

func (model *fullscreenModel) clampSuggestion(count int) {
	if count <= 0 {
		model.editor.suggestion = 0
		return
	}
	model.editor.suggestion = min(max(0, model.editor.suggestion), count-1)
}

func (model *fullscreenModel) setInput(value string) {
	model.editor.input = []rune(value)
	model.editor.cursor = len(model.editor.input)
	model.editor.suggestion = 0
}

func (model *fullscreenModel) clearInput() {
	model.editor.input = nil
	model.editor.cursor = 0
	model.editor.scroll = 0
	model.editor.suggestion = 0
}

func (model *fullscreenModel) insertRunes(input []rune) {
	if len(input) == 0 {
		return
	}
	at := model.editor.cursor
	model.editor.input = append(model.editor.input, make([]rune, len(input))...)
	copy(model.editor.input[at+len(input):], model.editor.input[at:len(model.editor.input)-len(input)])
	copy(model.editor.input[at:at+len(input)], input)
	model.editor.cursor += len(input)
	model.editor.suggestion = 0
}

func (model *fullscreenModel) moveWord(direction int) {
	if direction < 0 {
		for model.editor.cursor > 0 && unicode.IsSpace(model.editor.input[model.editor.cursor-1]) {
			model.editor.cursor--
		}
		for model.editor.cursor > 0 && !unicode.IsSpace(model.editor.input[model.editor.cursor-1]) {
			model.editor.cursor--
		}
		return
	}
	for model.editor.cursor < len(model.editor.input) && !unicode.IsSpace(model.editor.input[model.editor.cursor]) {
		model.editor.cursor++
	}
	for model.editor.cursor < len(model.editor.input) && unicode.IsSpace(model.editor.input[model.editor.cursor]) {
		model.editor.cursor++
	}
}

func editorLineStart(input []rune, cursor int) int {
	cursor = min(max(cursor, 0), len(input))
	for cursor > 0 && input[cursor-1] != '\n' {
		cursor--
	}
	return cursor
}

func editorLineEnd(input []rune, cursor int) int {
	cursor = min(max(cursor, 0), len(input))
	for cursor < len(input) && input[cursor] != '\n' {
		cursor++
	}
	return cursor
}

func moveEditorCursorVertical(editor *fullscreenEditor, direction int) bool {
	if !strings.ContainsRune(string(editor.input), '\n') {
		return false
	}
	start := editorLineStart(editor.input, editor.cursor)
	column := editor.cursor - start
	if direction < 0 {
		if start == 0 {
			return false
		}
		previousEnd := start - 1
		previousStart := editorLineStart(editor.input, previousEnd)
		editor.cursor = min(previousStart+column, previousEnd)
		return true
	}
	end := editorLineEnd(editor.input, editor.cursor)
	if end >= len(editor.input) {
		return false
	}
	nextStart := end + 1
	nextEnd := editorLineEnd(editor.input, nextStart)
	editor.cursor = min(nextStart+column, nextEnd)
	return true
}

func (model *fullscreenModel) deleteAtCursor() {
	if model.editor.cursor < len(model.editor.input) {
		model.editor.input = append(model.editor.input[:model.editor.cursor], model.editor.input[model.editor.cursor+1:]...)
		model.editor.suggestion = 0
	}
}

func (model *fullscreenModel) deletePreviousWord() {
	for model.editor.cursor > 0 && unicode.IsSpace(model.editor.input[model.editor.cursor-1]) {
		model.editor.input = append(model.editor.input[:model.editor.cursor-1], model.editor.input[model.editor.cursor:]...)
		model.editor.cursor--
	}
	for model.editor.cursor > 0 && !unicode.IsSpace(model.editor.input[model.editor.cursor-1]) {
		model.editor.input = append(model.editor.input[:model.editor.cursor-1], model.editor.input[model.editor.cursor:]...)
		model.editor.cursor--
	}
	model.editor.suggestion = 0
}

func fullscreenSuggestions(session *session, input string) []fullscreenSuggestion {
	return fullscreenSuggestionsWithModels(session, input, availableFullscreenModels(session))
}

func fullscreenSuggestionsWithModels(session *session, input string, models []fullscreenModelChoice) []fullscreenSuggestion {
	lower := strings.ToLower(input)
	if strings.HasPrefix(lower, "/model ") {
		query := strings.TrimSpace(strings.TrimPrefix(lower, "/model "))
		terms := strings.Fields(query)
		var suggestions []fullscreenSuggestion
		for _, choice := range models {
			haystack := strings.ToLower(choice.ref + " " + choice.name + " " + choice.provider)
			matched := true
			for _, term := range terms {
				if !strings.Contains(haystack, term) {
					matched = false
					break
				}
			}
			if !matched {
				continue
			}
			suggestions = append(suggestions, fullscreenSuggestion{
				label: choice.name, description: modelChoiceDescription(choice),
				value: "/model " + choice.ref, execute: true, current: choice.current,
			})
		}
		return suggestions
	}
	if strings.HasPrefix(lower, "/ralph ") {
		query := strings.TrimSpace(strings.TrimPrefix(lower, "/ralph "))
		if strings.ContainsAny(query, " \t\n") {
			return nil
		}
		commands := []fullscreenSuggestion{
			{label: "start", description: "Start a new iterative development loop", value: "/ralph start ", execute: false},
			{label: "status", description: "Show the active loop", value: "/ralph status", execute: true},
			{label: "list", description: "List saved loops", value: "/ralph list", execute: true},
			{label: "resume", description: "Resume a paused loop", value: "/ralph resume ", execute: false},
			{label: "stop", description: "Stop an active loop", value: "/ralph stop ", execute: false},
		}
		var suggestions []fullscreenSuggestion
		for _, command := range commands {
			if strings.HasPrefix(command.label, query) {
				suggestions = append(suggestions, command)
			}
		}
		return suggestions
	}
	if strings.HasPrefix(lower, "/thinking ") {
		query := strings.TrimSpace(strings.TrimPrefix(lower, "/thinking "))
		if strings.ContainsAny(query, " \t\n") {
			return nil
		}
		state := session.agent.State()
		current := state.ThinkingLevel
		var suggestions []fullscreenSuggestion
		for _, level := range llm.GetSupportedThinkingLevels(&state.Model) {
			name := string(level)
			if !strings.HasPrefix(name, query) {
				continue
			}
			suggestions = append(suggestions, fullscreenSuggestion{
				label: name, description: thinkingLevelDescription(name),
				value: "/thinking " + name, execute: true, current: name == current,
			})
		}
		return suggestions
	}
	if !strings.HasPrefix(input, "/") || strings.ContainsAny(input, " \t\n") {
		return nil
	}

	query := strings.ToLower(input)
	var suggestions []fullscreenSuggestion
	for _, command := range fullscreenSlashCommands {
		if !fullscreenCommandAvailable(session, command.name) || !strings.HasPrefix(strings.ToLower(command.name), query) {
			continue
		}
		item := fullscreenSuggestion{
			label: command.name, description: command.description,
			value: command.name, execute: true,
		}
		switch command.name {
		case "/model", "/thinking", "/ralph", "/plannotator-annotate", "/steer", "/followup":
			item.value += " "
			item.execute = false
		}
		suggestions = append(suggestions, item)
	}
	if session.resources != nil {
		for _, template := range session.resources.Templates {
			name := "/" + template.Name
			if strings.HasPrefix(strings.ToLower(name), query) {
				description := template.Description
				if template.ArgumentHint != "" {
					description = template.ArgumentHint + "  ·  " + description
				}
				suggestions = append(suggestions, fullscreenSuggestion{label: name, description: description, value: name, execute: true})
			}
		}
		for _, skill := range session.resources.Skills {
			name := "/skill:" + skill.Name
			if strings.HasPrefix(strings.ToLower(name), query) {
				suggestions = append(suggestions, fullscreenSuggestion{label: name, description: skill.Description, value: name, execute: true})
			}
		}
	}
	return suggestions
}

func fullscreenCommandAvailable(session *session, command string) bool {
	if command == "/ralph" {
		return session.loops != nil
	}
	if strings.HasPrefix(command, "/plannotator") {
		return session.plan != nil
	}
	return true
}

func availableFullscreenModels(session *session) []fullscreenModelChoice {
	state := session.agent.State()
	currentRef := state.Model.Provider + "/" + state.Model.ID
	catalog := newCatalog()
	configured := catalog.ConfiguredProviderIDs()
	if !containsString(configured, state.Model.Provider) {
		configured = append(configured, state.Model.Provider)
	}
	seen := make(map[string]bool)
	var choices []fullscreenModelChoice
	for _, providerID := range configured {
		provider := catalog.Provider(providerID)
		if provider == nil {
			continue
		}
		for _, candidate := range provider.Models() {
			if _, supported := llm.GetStreamer(candidate.API); !supported {
				continue
			}
			ref := providerID + "/" + candidate.ID
			if seen[ref] {
				continue
			}
			seen[ref] = true
			name := candidate.Name
			if name == "" {
				name = candidate.ID
			}
			choices = append(choices, fullscreenModelChoice{
				ref: ref, name: name, provider: providerID, context: candidate.ContextWindow,
				reasoning: candidate.Reasoning, current: ref == currentRef,
			})
		}
	}
	if !seen[currentRef] {
		name := state.Model.Name
		if name == "" {
			name = state.Model.ID
		}
		choices = append(choices, fullscreenModelChoice{
			ref: currentRef, name: name, provider: state.Model.Provider,
			context: state.Model.ContextWindow, reasoning: state.Model.Reasoning, current: true,
		})
	}
	sort.SliceStable(choices, func(left, right int) bool {
		if choices[left].current != choices[right].current {
			return choices[left].current
		}
		if choices[left].provider != choices[right].provider {
			return choices[left].provider < choices[right].provider
		}
		return choices[left].name < choices[right].name
	})
	return choices
}

func modelChoiceDescription(choice fullscreenModelChoice) string {
	parts := []string{choice.ref}
	if choice.context > 0 {
		parts = append(parts, compactNumber(choice.context)+" context")
	}
	if choice.reasoning {
		parts = append(parts, "reasoning")
	}
	return strings.Join(parts, "  ·  ")
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func fullscreenSuggestionTitle(input string) string {
	lower := strings.ToLower(input)
	if strings.HasPrefix(lower, "/model ") {
		return "SELECT MODEL  ·  type to filter authenticated providers"
	}
	if strings.HasPrefix(lower, "/thinking ") {
		return "THINKING LEVEL  ·  model-supported options"
	}
	if strings.HasPrefix(lower, "/ralph ") {
		return "RALPH LOOP  ·  iterative development controls"
	}
	return "COMMANDS  ·  ↑/↓ navigate  ·  Tab complete  ·  Enter select"
}

func thinkingLevelDescription(level string) string {
	switch level {
	case "off":
		return "Fastest responses, no extra reasoning"
	case "minimal":
		return "Very brief reasoning"
	case "low":
		return "Quick tasks and small edits"
	case "medium":
		return "Balanced depth and speed"
	case "high":
		return "Complex implementation work"
	case "xhigh":
		return "Deep analysis for difficult problems"
	case "max":
		return "Maximum available reasoning"
	default:
		return "Reasoning effort"
	}
}
