package main

// The interactive screen is driven by Bubble Tea. Agent work stays in the
// existing goroutines and is reported back as Tea messages, keeping terminal
// input, resize handling, rendering, and command-palette state in one loop.

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"

	"goshcoder/internal/agent"
	"goshcoder/internal/btw"
	"goshcoder/internal/claudetui"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/sessionlog"
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
type fullscreenLoginFinished struct {
	provider string
	err      error
}
type fullscreenInfoRefreshed struct{ info claudetui.SessionInfo }
type fullscreenBTWFinished struct {
	threadID string
	question string
	answer   string
	err      error
}
type fullscreenTick time.Time

type fullscreenBTWState struct {
	thread   *btw.Thread
	running  bool
	question string
	queued   []string
	cancel   context.CancelFunc
}

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
	btw           *fullscreenBTWState
	// renderCache memoizes per-message rendering so a long transcript is not
	// markdown-parsed from scratch on every keystroke.
	renderCache *tui.MessageCache
	// ticking records whether a spinner tick is already scheduled. The tick
	// only animates the spinner and the elapsed-time readout, so it is armed
	// while the agent is working and left off otherwise; re-arming it
	// unconditionally kept the process re-rendering eight times a second at a
	// fully idle prompt.
	ticking bool
	// loginChoices caches the /login picker. Building it resolves credentials
	// for every provider, which reads auth.json once per provider and can
	// trigger an OAuth refresh over the network -- work that must never happen
	// on the render path.
	loginChoices      []fullscreenSuggestion
	loginChoicesQuery string
	loginChoicesValid bool
	// gitRefreshWanted / gitRefreshInFlight schedule the git enrichment of the
	// sidebar off the update loop. sessionInfo forks `git status`, which in a
	// large repository takes seconds; running it inline froze the UI, and
	// holding a key that triggered it forked one git process per repeat.
	gitRefreshWanted   bool
	gitRefreshInFlight bool
	// sessionChoices caches the resume picker's candidates. suggestions() runs
	// inside View(), so an uncached picker would re-read every file in the
	// session shard on every keystroke and on every 120 ms tick -- the same
	// mistake /login had to be rescued from.
	sessionChoices      []sessionlog.Info
	sessionChoicesValid bool
	// quitArmedAt is when Ctrl+C was last pressed at an idle empty prompt.
	// It only arms when nothing is being saved: see handleTeaKey.
	quitArmedAt time.Time
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

		renderCache: tui.NewMessageCache(),
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
	model.ticking = true
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

// quitConfirmWindow is how long a first Ctrl+C stays armed.
const quitConfirmWindow = 3 * time.Second

func fullscreenTickCommand() tea.Cmd {
	return tea.Tick(120*time.Millisecond, func(now time.Time) tea.Msg { return fullscreenTick(now) })
}

// tickIfBusy schedules the next spinner frame only while the agent or a
// command is working. An idle prompt has nothing to animate, and re-arming
// regardless kept View -- which re-renders the whole transcript -- running
// eight times a second forever.
func (model *fullscreenModel) tickIfBusy() tea.Cmd {
	if !model.session.agent.State().IsStreaming && !model.busyCommand {
		model.ticking = false
		return nil
	}
	model.ticking = true
	return fullscreenTickCommand()
}

// resumeTicking restarts the spinner when work begins after an idle period.
func (model *fullscreenModel) resumeTicking() tea.Cmd {
	if model.ticking {
		return nil
	}
	return model.tickIfBusy()
}

func refreshFullscreenInfo(session *session) tea.Cmd {
	return func() tea.Msg { return fullscreenInfoRefreshed{info: session.sessionInfo()} }
}

// Update handles one Bubble Tea message.
//
// The spinner tick is re-armed here rather than at each site that starts work:
// the tick is only scheduled while the agent or a command is busy, so every
// transition into that state needs to restart it, and doing it centrally means
// a new transition cannot forget to.
func (model *fullscreenModel) Update(message tea.Msg) (tea.Model, tea.Cmd) {
	next, cmd := model.update(message)
	// Messages raised from agent-goroutine work (the Planner's review URL, for
	// one) are queued on the session because that goroutine must not touch the
	// screen. Drain them here so they appear like any other notice.
	for _, notice := range model.session.drainNotices() {
		model.notices = appendNotice(model.notices, notice.Kind, notice.Text)
	}
	cmds := []tea.Cmd{cmd}
	if resume := model.resumeTicking(); resume != nil {
		cmds = append(cmds, resume)
	}
	if git := model.pendingGitRefresh(); git != nil {
		cmds = append(cmds, git)
	}
	if len(cmds) == 1 {
		return next, cmd
	}
	return next, tea.Batch(cmds...)
}

func (model *fullscreenModel) update(message tea.Msg) (tea.Model, tea.Cmd) {
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
		model.refreshInfoAsync()
		model.activity = "Ready"
		model.activityAt = time.Time{}

	case fullscreenCommandFinished:
		model.models = availableFullscreenModels(model.session)
		if model.applyCommandResult(value.result) {
			return model, tea.Quit
		}

	case fullscreenLoginFinished:
		model.busyCommand = false
		model.activityAt = time.Time{}
		if value.err != nil {
			model.notices = appendNotice(model.notices, "Error", value.err.Error())
			model.activity = "Ready"
			break
		}
		// The child process wrote one provider entry without touching any of the
		// others. Rebuild the picker so every authenticated provider is visible
		// immediately.
		model.models = availableFullscreenModels(model.session)
		model.invalidateLoginSuggestions()
		model.invalidateSessionSuggestions()
		model.notices = appendNotice(model.notices, "Login", "Added "+value.provider+". Use /model to switch providers.")
		model.activity = "Logged in to " + value.provider

	case fullscreenInfoRefreshed:
		model.gitRefreshInFlight = false
		model.info = value.info

	case fullscreenBTWFinished:
		if model.btw == nil || model.btw.thread.ID != value.threadID {
			break
		}
		model.btw.running = false
		model.btw.cancel = nil
		model.btw.question = ""
		if value.err != nil {
			model.activity = "BTW question failed"
			model.notices = appendNotice(model.notices, "BTW Error", value.err.Error())
		} else {
			model.activity = "BTW answer ready"
		}
		if len(model.btw.queued) > 0 {
			next := model.btw.queued[0]
			model.btw.queued = model.btw.queued[1:]
			return model, model.startBTWQuestion(next)
		}

	case fullscreenTick:
		model.spin = (model.spin + 1) % 8
		return model, model.tickIfBusy()

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

// refreshInfoAsync updates everything the sidebar can show without blocking,
// and asks for the git-derived fields to be filled in on a background command.
func (model *fullscreenModel) refreshInfoAsync() {
	model.refreshRealtimeInfo()
	model.gitRefreshWanted = true
}

// pendingGitRefresh returns the command that enriches the sidebar with git
// state, at most one at a time.
func (model *fullscreenModel) pendingGitRefresh() tea.Cmd {
	if !model.gitRefreshWanted || model.gitRefreshInFlight {
		return nil
	}
	model.gitRefreshWanted = false
	model.gitRefreshInFlight = true
	return refreshFullscreenInfo(model.session)
}

func (model *fullscreenModel) View() string {
	if model.btw != nil {
		return model.viewBTW()
	}
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
		Cache:    model.renderCache,
		Title:    fmt.Sprintf("v%s  ·  %s/%s", Version, state.Model.Provider, state.Model.ID),
		Messages: messages,
		Sidebar: fullscreenSidebar(model.info, model.session.workspaceRoot(), model.activity,
			len(state.PendingToolCalls), model.recentTool, planItems, model.session.sidebarStorageLine()),
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
	view, scroll := tui.Render(frame, model.width, model.height)
	model.editor.scroll = scroll
	return view
}

func (model *fullscreenModel) handleTeaKey(key tea.KeyMsg) tea.Cmd {
	if model.btw != nil {
		return model.handleBTWTeaKey(key)
	}
	streaming := model.session.agent.State().IsStreaming
	if key.String() != "ctrl+c" && !model.quitArmedAt.IsZero() {
		// Anything else means the user did not mean to quit.
		model.quitArmedAt = time.Time{}
		if strings.HasPrefix(model.activity, "Press Ctrl+C again") {
			model.activity = "Ready"
		}
	}
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
		// Confirmation is only warranted when quitting would actually lose
		// something. Once the conversation is on disk, a second keypress is
		// friction with nothing behind it.
		//
		// The branch stays for the cases where the transcript really is only in
		// memory: -no-session, a session opened read-only because another
		// window holds the claim, and a recorder that stopped after a write
		// failure. Those are exactly the states in which this guard earns its
		// keep, so it is made conditional rather than deleted.
		if model.session.recordingActive() {
			return tea.Quit
		}
		if !model.quitArmedAt.IsZero() && time.Since(model.quitArmedAt) < quitConfirmWindow {
			return tea.Quit
		}
		model.quitArmedAt = time.Now()
		model.activity = "Press Ctrl+C again to exit (this session is not being saved)"
		// Keep the spinner loop alive long enough for the hint to expire on
		// its own if the user does nothing.
		return fullscreenTickCommand()

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
		model.refreshInfoAsync()
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

	case " ":
		model.insertRunes([]rune{' '})

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
			if key.Type == tea.KeySpace && len(input) == 0 {
				input = []rune{' '}
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
	model.refreshInfoAsync()
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

	if strings.HasPrefix(prompt, "/btw") && (prompt == "/btw" || strings.HasPrefix(prompt, "/btw ")) {
		if streaming {
			model.activity = "Wait for the main response before opening BTW"
			return nil
		}
		return model.openBTW(strings.TrimSpace(strings.TrimPrefix(prompt, "/btw")))
	}
	if prompt == "/omni setup" {
		model.busyCommand = true
		model.activity = "Opening OmniRoute setup"
		model.activityAt = time.Now()
		return fullscreenOmniSetupCommand()
	}

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
		fields := strings.Fields(prompt)
		if len(fields) == 2 && fields[0] == "/login" {
			model.busyCommand = true
			model.activity = "Logging in to " + fields[1]
			model.activityAt = time.Now()
			return fullscreenLoginCommand(fields[1])
		}
		model.activity = "Running " + fields[0]
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

func (model *fullscreenModel) openBTW(arguments string) tea.Cmd {
	fields := strings.Fields(arguments)
	if len(fields) > 0 {
		switch strings.ToLower(fields[0]) {
		case "settings", "list", "bring":
			result := runFullscreenCommand(model.session, "/btw "+arguments)
			model.applyCommandResult(result)
			return nil
		}
	}
	selection, warnings, err := model.session.resolveBTWSelection()
	if err != nil {
		model.notices = appendNotice(model.notices, "Error", err.Error())
		return nil
	}
	for _, warning := range warnings {
		model.notices = appendNotice(model.notices, "BTW", warning)
	}
	var thread *btw.Thread
	question := arguments
	if len(fields) >= 2 && strings.EqualFold(fields[0], "resume") {
		thread = model.session.btw.Get(fields[1])
		if thread == nil {
			model.notices = appendNotice(model.notices, "Error", "Unknown BTW thread "+fields[1])
			return nil
		}
		question = strings.TrimSpace(strings.Join(fields[2:], " "))
	}
	if thread == nil {
		thread = model.session.btw.New(btw.BuildConversationContext(model.session.agent.State().Messages), selection.thinking)
	}
	model.btw = &fullscreenBTWState{thread: thread}
	model.editor.scroll = 0
	model.activity = "BTW side thread"
	if question != "" {
		return model.startBTWQuestion(question)
	}
	return nil
}

func (model *fullscreenModel) startBTWQuestion(question string) tea.Cmd {
	if model.btw == nil || strings.TrimSpace(question) == "" {
		return nil
	}
	question = strings.TrimSpace(question)
	if model.btw.running {
		model.btw.queued = append(model.btw.queued, question)
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	state := model.btw
	thinking := model.session.btw.Snapshot(state.thread).ThinkingLevel
	state.running, state.question, state.cancel = true, question, cancel
	model.activity = "Answering BTW question"
	return func() tea.Msg {
		answer, err := model.session.askBTW(ctx, state.thread, question, thinking)
		return fullscreenBTWFinished{threadID: state.thread.ID, question: question, answer: answer, err: err}
	}
}

func (model *fullscreenModel) closeBTW() {
	if model.btw != nil && model.btw.cancel != nil {
		model.btw.cancel()
	}
	model.btw = nil
	model.editor.scroll = 0
	model.activity = "Ready"
}

func (model *fullscreenModel) handleBTWTeaKey(key tea.KeyMsg) tea.Cmd {
	state := model.btw
	if state == nil {
		return nil
	}
	switch key.String() {
	case "ctrl+c":
		model.clearInput()
		state.queued = nil
		model.closeBTW()
		return nil
	case "esc":
		if !state.running {
			model.closeBTW()
		}
		return nil
	case "ctrl+r":
		// Read through the manager's lock: the answer for an in-flight question
		// is appended to thread.Turns from a tea.Cmd goroutine, so iterating
		// the live slice here races the append and can pair a new length with
		// the old backing array.
		snapshot := model.session.btw.Snapshot(state.thread)
		segments := btw.LatestSegments(&snapshot)
		if len(segments) == 0 {
			model.activity = "No successful BTW answer to bring"
			return nil
		}
		draft := btw.FormatBringToMain(segments)
		model.closeBTW()
		if len(model.editor.input) > 0 {
			model.insertRunes([]rune("\n\n" + draft))
		} else {
			model.setInput(draft)
		}
		model.activity = fmt.Sprintf("Brought ~%d BTW tokens to the editor", btw.EstimateTokens(segments))
		return nil
	case "shift+tab":
		selection, _, err := model.session.resolveBTWSelection()
		if err != nil {
			return nil
		}
		levels := llm.GetSupportedThinkingLevels(selection.model)
		current := model.session.btw.Snapshot(state.thread).ThinkingLevel
		next := current
		for index, level := range levels {
			if level == current {
				next = levels[(index+1)%len(levels)]
				goto persist
			}
		}
		if len(levels) > 0 {
			next = levels[0]
		}
	persist:
		model.session.btw.SetThinkingLevel(state.thread, next)
		if selection.remember {
			level := next
			if _, err := btw.UpdateSettings(config.BTWPath(), &level, nil); err != nil {
				model.activity = "BTW thinking changed but was not remembered"
			}
		}
		return nil
	case "enter":
		question := strings.TrimSpace(string(model.editor.input))
		if question == "" {
			return nil
		}
		model.clearInput()
		if state.running {
			state.queued = append(state.queued, question)
			model.activity = "BTW steering question queued"
			return nil
		}
		return model.startBTWQuestion(question)
	case "shift+enter", "ctrl+j":
		model.insertRunes([]rune{'\n'})
	case " ":
		model.insertRunes([]rune{' '})
	case "up":
		if !moveEditorCursorVertical(&model.editor, -1) {
			editorHistory(&model.editor, -1)
		}
	case "down":
		if !moveEditorCursorVertical(&model.editor, 1) {
			editorHistory(&model.editor, 1)
		}
	case "left", "ctrl+b":
		model.editor.cursor = max(0, model.editor.cursor-1)
	case "right", "ctrl+f":
		model.editor.cursor = min(len(model.editor.input), model.editor.cursor+1)
	case "home", "ctrl+a":
		model.editor.cursor = editorLineStart(model.editor.input, model.editor.cursor)
	case "end", "ctrl+e":
		model.editor.cursor = editorLineEnd(model.editor.input, model.editor.cursor)
	case "pgup":
		model.editor.scroll += 10
	case "pgdown":
		model.editor.scroll = max(0, model.editor.scroll-10)
	case "backspace":
		if model.editor.cursor > 0 {
			model.editor.input = append(model.editor.input[:model.editor.cursor-1], model.editor.input[model.editor.cursor:]...)
			model.editor.cursor--
		}
	case "delete":
		model.deleteAtCursor()
	case "ctrl+k":
		model.editor.input = model.editor.input[:model.editor.cursor]
	case "ctrl+u":
		model.editor.input = append([]rune(nil), model.editor.input[model.editor.cursor:]...)
		model.editor.cursor = 0
	case "ctrl+w":
		model.deletePreviousWord()
	default:
		if key.Type == tea.KeyRunes || key.Type == tea.KeySpace {
			var input []rune
			for _, r := range key.Runes {
				if !unicode.IsControl(r) {
					input = append(input, r)
				}
			}
			if key.Type == tea.KeySpace && len(input) == 0 {
				input = []rune{' '}
			}
			model.insertRunes(input)
		}
	}
	return nil
}

func (model *fullscreenModel) viewBTW() string {
	state := model.btw
	if state == nil {
		return ""
	}
	thread := model.session.btw.Snapshot(state.thread)
	messages := make([]tui.Message, 0, len(thread.Turns)*2+1)
	for _, turn := range thread.Turns {
		messages = append(messages, tui.Message{Role: "user", Text: turn.Question})
		if turn.Kind == "error" {
			messages = append(messages, tui.Message{Role: "Error", Text: turn.Answer, IsError: true})
		} else {
			messages = append(messages, tui.Message{Role: "assistant", Text: turn.Answer})
		}
	}
	if state.running {
		already := false
		if len(thread.Turns) > 0 && thread.Turns[len(thread.Turns)-1].Question == state.question {
			already = true
		}
		if !already {
			messages = append(messages, tui.Message{Role: "user", Text: state.question})
			messages = append(messages, tui.Message{Role: "thinking", Text: "Answering…"})
		}
	}
	for _, question := range state.queued {
		messages = append(messages, tui.Message{Role: "Notice", Text: "Steering · " + question})
	}
	status := "Type a side question · Esc close · Ctrl+R bring latest to main"
	if state.running {
		status = "Answering… · Enter queues Steering · Ctrl+C cancel and close"
	}
	sidebar := []string{"title\tBTW · side thread", "accent\t" + thread.ID, "meta\t" + thread.ThinkingLevel + " thinking", "", "section\tEphemeral", "meta\tNever added to main context", fmt.Sprintf("meta\t%d recorded question(s)", len(thread.Turns)), "", "section\tKeys", "meta\tShift+Tab  thinking", "meta\tCtrl+R  bring latest", "meta\tEsc  close", "", "brand\t● GoshCoder BTW"}
	view, scroll := tui.Render(tui.Frame{Title: "btw · side thread · " + thread.ID, Messages: messages, Sidebar: sidebar, Input: model.editor.input, Cursor: model.editor.cursor, Status: status, Streaming: state.running, Scroll: model.editor.scroll, Thinking: thread.ThinkingLevel, Mode: "BTW", VirtualCursor: true}, model.width, model.height)
	model.editor.scroll = scroll
	return view
}

func fullscreenOmniSetupCommand() tea.Cmd {
	executable, err := os.Executable()
	if err != nil {
		return func() tea.Msg { return fullscreenCommandFinished{result: fullscreenResult{err: err}} }
	}
	command := exec.Command(executable, "omni", "setup")
	return tea.ExecProcess(command, func(err error) tea.Msg {
		return fullscreenCommandFinished{result: fullscreenResult{err: err, output: map[bool]string{true: "OmniRoute setup complete. Run /omni sync.", false: ""}[err == nil]}}
	})
}

func fullscreenLoginCommand(providerID string) tea.Cmd {
	provider := newCatalog().Provider(providerID)
	if provider == nil {
		return func() tea.Msg {
			return fullscreenLoginFinished{provider: providerID, err: fmt.Errorf("unknown provider %q", providerID)}
		}
	}
	executable, err := os.Executable()
	if err != nil {
		return func() tea.Msg { return fullscreenLoginFinished{provider: providerID, err: err} }
	}
	subcommand := "set"
	if _, ok := catalog.LoginProviderFor(providerID); ok {
		subcommand = "login"
	}
	command := exec.Command(executable, "auth", subcommand, providerID)
	return tea.ExecProcess(command, func(err error) tea.Msg {
		return fullscreenLoginFinished{provider: providerID, err: err}
	})
}

func (model *fullscreenModel) applyCommandResult(result fullscreenResult) bool {
	model.busyCommand = false
	// Any command may have named, switched or removed a session, so the picker's
	// cached candidates are stale from here.
	model.invalidateSessionSuggestions()
	if result.output != "" {
		model.notices = appendNotice(model.notices, "Command", result.output)
	}
	if result.err != nil {
		model.notices = appendNotice(model.notices, "Error", result.err.Error())
	}
	model.refreshInfoAsync()
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
	return command == "/compact" || command == "/sessions" || command == "/resume" ||
		command == "/tree" || command == "/fork" || command == "/clone" ||
		command == "/export" || command == "/import" ||
		(command == "/omni" && len(fields) > 1 && fields[1] != "setup") ||
		command == "/planner-review" || command == "/planner-annotate" || command == "/planner-last" ||
		command == "/plannotator-review" || command == "/plannotator-annotate" || command == "/plannotator-last" ||
		(command == "/ralph" && len(fields) > 1 && fields[1] == "start")
}

func (model *fullscreenModel) suggestions() []fullscreenSuggestion {
	return fullscreenSuggestionsWithModels(model.session, string(model.editor.input), model.models,
		model.cachedLoginSuggestions, model.cachedSessionSuggestions)
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
	return fullscreenSuggestionsWithModels(session, input, availableFullscreenModels(session), nil, nil)
}

// loginSuggestions is how the picker for "/login <query>" is produced. The
// fullscreen model passes a cached implementation, because the uncached one
// resolves credentials for every provider and must not run on the render path.
func fullscreenSuggestionsWithModels(session *session, input string, models []fullscreenModelChoice,
	loginSuggestions func(string) []fullscreenSuggestion,
	sessionSuggestions func(string) []fullscreenSuggestion) []fullscreenSuggestion {
	if loginSuggestions == nil {
		loginSuggestions = fullscreenLoginSuggestions
	}
	lower := strings.ToLower(input)
	if strings.HasPrefix(lower, "/resume ") && sessionSuggestions != nil {
		return sessionSuggestions(strings.TrimSpace(strings.TrimPrefix(input[len("/resume "):], " ")))
	}
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
	if strings.HasPrefix(lower, "/login ") {
		query := strings.TrimSpace(strings.TrimPrefix(lower, "/login "))
		if strings.ContainsAny(query, " \t\n") {
			return nil
		}
		return loginSuggestions(query)
	}
	if strings.HasPrefix(lower, "/omni ") {
		query := strings.TrimSpace(strings.TrimPrefix(lower, "/omni "))
		if strings.ContainsAny(query, " \t\n") {
			return nil
		}
		commands := []fullscreenSuggestion{
			{label: "status", description: "Check gateway health and synchronized models", value: "/omni status", execute: true},
			{label: "sync", description: "Refresh models from /v1/models", value: "/omni sync", execute: true},
			{label: "setup", description: "Configure URL and API key", value: "/omni setup", execute: true},
			{label: "dashboard", description: "Show the management dashboard URL", value: "/omni dashboard", execute: true},
		}
		var suggestions []fullscreenSuggestion
		for _, command := range commands {
			if strings.HasPrefix(command.label, query) {
				suggestions = append(suggestions, command)
			}
		}
		return suggestions
	}
	if strings.HasPrefix(lower, "/btw ") {
		query := strings.TrimSpace(strings.TrimPrefix(lower, "/btw "))
		if strings.ContainsAny(query, " \t\n") {
			return nil
		}
		commands := []fullscreenSuggestion{
			{label: "list", description: "List in-memory side threads", value: "/btw list", execute: true},
			{label: "resume", description: "Resume a side thread by ID", value: "/btw resume ", execute: false},
			{label: "settings", description: "View or change side-thread settings", value: "/btw settings", execute: true},
		}
		var suggestions []fullscreenSuggestion
		for _, command := range commands {
			if strings.HasPrefix(command.label, query) {
				suggestions = append(suggestions, command)
			}
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
		case "/model", "/login", "/omni", "/btw", "/thinking", "/ralph", "/planner-annotate", "/steer", "/followup":
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

// cachedLoginSuggestions returns the /login picker without touching the
// credential store on the render path.
//
// fullscreenLoginSuggestions resolves auth for every provider, which re-reads
// auth.json once each and, for an expired OAuth credential, performs a token
// refresh over the network. View() runs on the render goroutine, so behind a
// captive portal a single "/login " in the composer froze the whole UI for the
// HTTP timeout on every frame.
func (model *fullscreenModel) cachedLoginSuggestions(query string) []fullscreenSuggestion {
	if model.loginChoicesValid && model.loginChoicesQuery == query {
		return model.loginChoices
	}
	model.loginChoices = fullscreenLoginSuggestions(query)
	model.loginChoicesQuery = query
	model.loginChoicesValid = true
	return model.loginChoices
}

// invalidateLoginSuggestions forces the picker to be rebuilt after the set of
// stored credentials may have changed.
func (model *fullscreenModel) invalidateLoginSuggestions() {
	model.loginChoicesValid = false
}

func fullscreenLoginSuggestions(query string) []fullscreenSuggestion {
	models := newCatalog()
	configured := make(map[string]bool)
	for _, providerID := range models.ConfiguredProviderIDs() {
		configured[providerID] = true
	}
	type choice struct {
		id          string
		description string
		oauth       bool
	}
	var choices []choice
	for _, providerID := range models.ProviderIDs() {
		provider := models.Provider(providerID)
		if provider == nil || !strings.Contains(strings.ToLower(providerID+" "+provider.Name), query) {
			continue
		}
		supported := false
		for _, model := range provider.Models() {
			if _, ok := llm.GetStreamer(model.API); ok {
				supported = true
				break
			}
		}
		if !supported {
			continue
		}
		_, oauth := catalog.LoginProviderFor(providerID)
		description := provider.Name + "  ·  API key"
		if oauth {
			description = provider.Name + "  ·  OAuth / subscription"
		}
		if configured[providerID] {
			description = "SIGNED IN  ·  " + description
		}
		choices = append(choices, choice{id: providerID, description: description, oauth: oauth})
	}
	sort.SliceStable(choices, func(left, right int) bool {
		if choices[left].oauth != choices[right].oauth {
			return choices[left].oauth
		}
		return choices[left].id < choices[right].id
	})
	suggestions := make([]fullscreenSuggestion, 0, len(choices))
	for _, choice := range choices {
		suggestions = append(suggestions, fullscreenSuggestion{
			label: choice.id, description: choice.description,
			value: "/login " + choice.id, execute: true,
		})
	}
	return suggestions
}

func fullscreenCommandAvailable(session *session, command string) bool {
	if command == "/ralph" {
		return session.loops != nil
	}
	if strings.HasPrefix(command, "/planner") {
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
	if strings.HasPrefix(lower, "/resume ") {
		return "RESUME SESSION  ·  type to search names and transcripts"
	}
	if strings.HasPrefix(lower, "/model ") {
		return "SELECT MODEL  ·  type to filter authenticated providers"
	}
	if strings.HasPrefix(lower, "/thinking ") {
		return "THINKING LEVEL  ·  model-supported options"
	}
	if strings.HasPrefix(lower, "/login ") {
		return "ADD PROVIDER  ·  OAuth or API key  ·  existing logins are kept"
	}
	if strings.HasPrefix(lower, "/omni ") {
		return "OMNIROUTE  ·  setup, synchronize, and inspect the gateway"
	}
	if strings.HasPrefix(lower, "/btw ") {
		return "BTW  ·  ephemeral side questions that do not pollute main context"
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
