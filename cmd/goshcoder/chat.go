package main

// Interactive chat mode uses the native Bubble Tea fullscreen interface on a
// TTY and falls back to a pipeable line-oriented REPL. Both paths share queued
// steering, follow-ups, aborts, runtime model/thinking changes, local resources,
// compaction, and native extension commands.

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/claudetui"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/ralph"
	"goshcoder/internal/tui"
)

const chatHelp = `Slash commands:
  /help                 Show this help
  /model [ref]          Show or switch the model
  /login [provider]     Add an OAuth or API-key provider (keeps existing logins)
  /omni [command]       Setup, sync, inspect, or open OmniRoute
  /btw [question]       Ask in an ephemeral, context-aware side thread
  /thinking [level]     Show or set the thinking level
  /system [text]        Show or replace the system prompt
  /tools                List the available tools
  /messages             Show the transcript summary
  /status, /session     Show model, context, cost, git, and mode information
  /hotkeys              Show interactive keyboard shortcuts
  /steer <text>         Queue a steering message for the next turn
  /followup <text>      Queue a follow-up message
  /queue                Show whether messages are queued
  /clear, /new          Start a fresh conversation
  /compact [focus]      Summarize older context and keep recent turns
  /reload               Reload context files, prompts, and skills
  /resources            Show loaded local resources
  /ralph <subcommand>   Ralph loops (start, list, status, resume, stop)
  /planner              Toggle native planning mode
  /planner-review [PR URL]  Review git changes or a GitHub PR
  /planner-annotate <target> Annotate a file, folder, or URL
  /planner-last         Annotate the last assistant response
  /use-claude-code-tui  Enable the native startup/editor look
  /use-default-tui      Restore the plain line-oriented look
  /exit                 Leave chat

Anything else is sent to the model. Ctrl-C aborts a running turn; Ctrl-C while
idle exits.
`

func applyChatDefaults(flags *flag.FlagSet, cfg *sessionConfig) {
	// Coding tools and native Ralph loops are available out of the box in
	// interactive mode, matching always-loaded pi extensions. Explicit false
	// flags still provide an opt-out.
	cfg.EnableTools = true
	cfg.EnableRalph = true
	flags.Lookup("tools").DefValue = "true"
	flags.Lookup("ralph").DefValue = "true"
}

func chatCommand(args []string) error {
	flags := flag.NewFlagSet("chat", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	cfg := bindSessionFlags(flags)
	applyChatDefaults(flags, cfg)
	flags.BoolVar(&cfg.Fullscreen, "fullscreen", true, "use the fullscreen interactive TUI")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if err := validateSessionFlags(cfg); err != nil {
		return err
	}
	markModelProvenance(flags, cfg)
	// Planner commands are available in chat even when plan mode was not
	// requested at startup.
	cfg.LoadPlannotator = true
	cfg.Fullscreen = cfg.Fullscreen && tui.Supported(os.Stdin, os.Stderr)
	if cfg.ModelRef == "" {
		modelRef, modelErr := defaultChatModel()
		if modelErr != nil && tui.Supported(os.Stdin, os.Stderr) {
			modelRef, modelErr = onboardChatModel()
		}
		if modelErr != nil {
			return modelErr
		}
		cfg.ModelRef = modelRef
	}

	s, err := newSession(*cfg)
	if err != nil {
		return err
	}
	defer s.close()
	_ = config.WriteDefaultModel(s.model.Provider + "/" + s.model.ID)
	if cfg.Fullscreen {
		return runFullscreenChat(s)
	}
	for _, notice := range s.drainStartupNotices() {
		fmt.Fprintln(os.Stderr, dim("session: "+notice))
	}

	// Ctrl-C aborts a turn; when idle it closes stdin so the read loop ends.
	stop := s.handleInterrupts(func() {
		fmt.Fprintln(os.Stderr, "\n"+dim("exiting"))
		os.Stdin.Close()
	})
	defer stop()

	var lastSidebar claudetui.SessionInfo
	if s.claudeTUI {
		lastSidebar = s.sessionInfo()
		for _, line := range claudetui.HeaderWithInfo(terminalWidth(), Version, s.model.Provider+"/"+s.model.ID, s.agent.State().ThinkingLevel, s.workspaceRoot(), colorEnabled(), &lastSidebar) {
			fmt.Fprintln(os.Stderr, line)
		}
	} else {
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
			"goshcoder %s · %s/%s · /help for commands", Version, s.model.Provider, s.model.ID)))
	}

	if banner := s.resumeBanner; banner != "" {
		renderRestoredTranscript(s.restoredMessages, banner)
	}
	if hint := s.exitHint(); hint != "" {
		fmt.Fprintln(os.Stderr, dim(hint))
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		if s.claudeTUI {
			// The line-oriented UI cannot pin a panel over scrolling output. Refresh
			// it automatically whenever model usage, tools, git, or mode changes.
			currentSidebar := s.sessionInfo()
			if !reflect.DeepEqual(currentSidebar, lastSidebar) {
				fmt.Fprintln(os.Stderr)
				for _, statusLine := range claudetui.Sidebar(min(terminalWidth(), 42), currentSidebar, colorEnabled()) {
					fmt.Fprintln(os.Stderr, statusLine)
				}
				lastSidebar = currentSidebar
			}
			fmt.Fprint(os.Stderr, "\n"+claudetui.InputPrompt(min(terminalWidth(), 88), colorEnabled()))
		} else {
			fmt.Fprint(os.Stderr, bold("\n> "))
		}
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF || errors.Is(err, os.ErrClosed) {
				// A closed stdin is a clean exit, not a failure.
				fmt.Fprintln(os.Stderr)
				return nil
			}
			return err
		}

		input := strings.TrimSpace(line)
		if input == "" {
			continue
		}

		if expanded, ok, expandErr := s.expandResourceInput(input); ok {
			if expandErr != nil {
				fmt.Fprintf(os.Stderr, "error: %v\n", expandErr)
				continue
			}
			input = expanded
		} else if expandErr != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", expandErr)
			continue
		}

		if strings.HasPrefix(input, "/") {
			exit, err := s.handleSlashCommand(input)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s\n", "error: "+err.Error())
			}
			if exit {
				return nil
			}
			continue
		}

		// A prompt while a turn is somehow still in flight becomes steering
		// rather than an error.
		if s.agent.State().IsStreaming {
			s.agent.Steer(userMessage(input))
			fmt.Fprintf(os.Stderr, "%s\n", dim("queued as a steering message"))
			continue
		}

		if err := s.runTurn(input); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", "error: "+err.Error())
		}
	}
}

func onboardChatModel() (string, error) {
	fmt.Fprintln(os.Stderr, "Welcome to GoshCoder. Choose a subscription login:")
	fmt.Fprintln(os.Stderr, "  1. OpenAI Codex (recommended)")
	fmt.Fprintln(os.Stderr, "  2. Anthropic")
	fmt.Fprintln(os.Stderr, "  3. Kimi Coding")
	fmt.Fprint(os.Stderr, "Choice [1]: ")
	choice, err := readTerminalLine(os.Stdin)
	if err != nil {
		return "", err
	}
	provider := "openai-codex"
	switch strings.TrimSpace(choice) {
	case "", "1":
	case "2":
		provider = "anthropic"
	case "3":
		provider = "kimi-coding"
	default:
		return "", fmt.Errorf("unknown login choice %q", strings.TrimSpace(choice))
	}
	if err := authCommand([]string{"login", provider}); err != nil {
		return "", err
	}
	return defaultChatModel()
}

func readTerminalLine(input *os.File) (string, error) {
	var line strings.Builder
	var one [1]byte
	for {
		count, err := input.Read(one[:])
		if count == 1 {
			if one[0] == '\n' {
				return line.String(), nil
			}
			if one[0] != '\r' {
				line.WriteByte(one[0])
			}
		}
		if err != nil {
			return "", err
		}
	}
}

func defaultChatModel() (string, error) {
	if configured := strings.TrimSpace(os.Getenv("GOSHCODER_MODEL")); configured != "" {
		return configured, nil
	}
	if remembered := config.ReadDefaultModel(); remembered != "" {
		if _, _, err := newCatalog().ResolveModel(remembered); err == nil {
			return remembered, nil
		}
	}
	catalog := newCatalog()
	configured := catalog.ConfiguredProviderIDs()
	preferred := []struct{ provider, model string }{
		{"openai-codex", "gpt-5.4"},
		{"anthropic", "claude-sonnet-4-5"},
		{"kimi-coding", "kimi-for-coding"},
		{"openai", "gpt-5.1"},
	}
	available := make(map[string]bool, len(configured))
	for _, provider := range configured {
		available[provider] = true
	}
	for _, candidate := range preferred {
		if available[candidate.provider] && catalog.Model(candidate.provider, candidate.model) != nil {
			return candidate.provider + "/" + candidate.model, nil
		}
	}
	for _, providerID := range configured {
		provider := catalog.Provider(providerID)
		if provider == nil {
			continue
		}
		models := provider.Models()
		if len(models) > 0 {
			return providerID + "/" + models[len(models)-1].ID, nil
		}
	}
	return "", errors.New("no authenticated model is available; run 'goshcoder auth login openai-codex' or pass -m provider/model")
}

// userMessage builds a transcript user message.
func userMessage(text string) llm.UserMessage {
	return llm.UserMessage{Role: "user", Content: text, Timestamp: time.Now().UnixMilli()}
}

// handleSlashCommand runs one slash command. exit is true when chat should end.
func (s *session) handleSlashCommand(input string) (exit bool, err error) {
	command, rest, _ := strings.Cut(input, " ")
	rest = strings.TrimSpace(rest)
	// Preserve the old upstream command names as hidden compatibility aliases.
	switch command {
	case "/plannator", "/plannotator":
		command = "/planner"
	case "/plannotator-review":
		command = "/planner-review"
	case "/plannotator-annotate":
		command = "/planner-annotate"
	case "/plannotator-last":
		command = "/planner-last"
	}

	switch command {
	case "/help", "/?":
		fmt.Fprint(os.Stderr, chatHelp)

	case "/exit", "/quit":
		return true, nil

	case "/model":
		if rest == "" {
			fmt.Fprintf(os.Stderr, "%s/%s\n", s.model.Provider, s.model.ID)
			return false, nil
		}
		if err := s.setModel(rest); err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("model set to %s/%s", s.model.Provider, s.model.ID)))

	case "/login":
		if rest == "" {
			fmt.Fprintln(os.Stderr, "Choose a provider from /login in the command palette, or run /login <provider>.")
			fmt.Fprintln(os.Stderr, "OAuth subscriptions: "+strings.Join(catalog.OAuthProviderIDs(), ", "))
			fmt.Fprintln(os.Stderr, "Other providers prompt for an API key; existing provider credentials are preserved.")
			return false, nil
		}
		fields := strings.Fields(rest)
		if len(fields) != 1 {
			return false, errors.New("usage: /login <provider>")
		}
		providerID := fields[0]
		if newCatalog().Provider(providerID) == nil {
			return false, fmt.Errorf("unknown provider %q", providerID)
		}
		subcommand := "set"
		if _, ok := catalog.LoginProviderFor(providerID); ok {
			subcommand = "login"
		}
		if err := authCommand([]string{subcommand, providerID}); err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "Added %s. Use /model to switch providers.\n", providerID)

	case "/omni":
		output, commandErr := runOmniCommand(context.Background(), strings.Fields(rest), true)
		if output != "" {
			fmt.Fprintln(os.Stderr, output)
		}
		return false, commandErr

	case "/btw":
		if s.agent.State().IsStreaming {
			return false, errors.New("wait for the active response before opening /btw")
		}
		return false, s.handleBTWCommand(rest)

	case "/thinking":
		available := llm.GetSupportedThinkingLevels(s.model)
		availableNames := make([]string, len(available))
		for index, level := range available {
			availableNames[index] = string(level)
		}
		if rest == "" {
			fmt.Fprintf(os.Stderr, "current: %s\navailable for %s/%s: %s\n",
				s.agent.State().ThinkingLevel, s.model.Provider, s.model.ID, strings.Join(availableNames, ", "))
			return false, nil
		}
		rest = strings.ToLower(rest)
		if !isThinkingLevel(rest) {
			return false, fmt.Errorf("unknown thinking level %q (off|minimal|low|medium|high|xhigh|max)", rest)
		}
		supported := false
		for _, level := range available {
			if level == rest {
				supported = true
				break
			}
		}
		if !supported {
			return false, fmt.Errorf("thinking level %q is not supported by %s/%s (available: %s)",
				rest, s.model.Provider, s.model.ID, strings.Join(availableNames, ", "))
		}
		s.agent.SetThinkingLevel(rest)
		fmt.Fprintf(os.Stderr, "%s\n", dim("thinking set to "+rest))

	case "/system":
		if rest == "" {
			current := s.agent.State().SystemPrompt
			if current == "" {
				current = "(none)"
			}
			fmt.Fprintln(os.Stderr, current)
			return false, nil
		}
		s.setExplicitSystemPrompt(rest)
		if err := s.reloadResources(); err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim("system prompt updated"))

	case "/tools":
		state := s.agent.State()
		if len(state.Tools) == 0 {
			fmt.Fprintln(os.Stderr, "No tools enabled. Restart with -tools to enable them.")
			return false, nil
		}
		names := make([]string, 0, len(state.Tools))
		for _, tool := range state.Tools {
			names = append(names, tool.Name)
		}
		sort.Strings(names)
		fmt.Fprintln(os.Stderr, strings.Join(names, ", "))

	case "/messages":
		printTranscriptSummary(s.agent.State().Messages)

	case "/status", "/sidebar":
		for _, line := range claudetui.Sidebar(min(terminalWidth(), 42), s.sessionInfo(), colorEnabled()) {
			fmt.Fprintln(os.Stderr, line)
		}

	case "/session":
		// Extended rather than replaced: the card is what people already reach
		// for, and pi's /session means the same thing.
		for _, line := range claudetui.Sidebar(min(terminalWidth(), 42), s.sessionInfo(), colorEnabled()) {
			fmt.Fprintln(os.Stderr, line)
		}
		for _, line := range s.sessionStorageLines() {
			fmt.Fprintln(os.Stderr, dim(line))
		}

	case "/sessions":
		for _, line := range sessionListLines(s.workspaceRoot()) {
			fmt.Fprintln(os.Stderr, dim(line))
		}

	case "/resume":
		if rest == "" {
			return false, errors.New("/resume needs a session id, prefix, or path")
		}
		if err := s.switchSession(rest); err != nil {
			return false, err
		}
		for _, notice := range s.drainNotices() {
			fmt.Fprintln(os.Stderr, dim(notice.Kind+": "+notice.Text))
		}
		renderRestoredTranscript(s.restoredMessages, "resumed transcript")

	case "/name":
		if rest == "" {
			if name := s.sessionName(); name != "" {
				fmt.Fprintln(os.Stderr, name)
			} else {
				fmt.Fprintln(os.Stderr, dim("this session has no name; /name <text> sets one"))
			}
			return false, nil
		}
		if err := s.setSessionName(rest); err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim("session named "+rest))

	case "/hotkeys":
		fmt.Fprintln(os.Stderr, `Enter       send or accept selection
Shift-Enter insert a newline (Ctrl-J where distinguishable)
Up/Down     move between editor lines, palette items, or history
Alt-←/→     move by word; Home/End move within the current line
Tab         complete the selected command
Shift-Tab   cycle model-supported thinking levels
Ctrl-L      open model picker; Ctrl-O expand tools; Ctrl-T toggle thinking
PgUp/PgDn   scroll transcript
Esc         close palette or abort a response
Ctrl-C      clear input, abort, or quit
Ctrl-D      quit when the editor is empty`)

	case "/steer":
		if rest == "" {
			return false, errors.New("usage: /steer <text>")
		}
		s.agent.Steer(userMessage(rest))
		fmt.Fprintf(os.Stderr, "%s\n", dim("steering message queued"))

	case "/followup":
		if rest == "" {
			return false, errors.New("usage: /followup <text>")
		}
		s.agent.FollowUp(userMessage(rest))
		fmt.Fprintf(os.Stderr, "%s\n", dim("follow-up queued"))

	case "/queue":
		if s.agent.HasQueuedMessages() {
			fmt.Fprintln(os.Stderr, "Messages are queued.")
		} else {
			fmt.Fprintln(os.Stderr, "No queued messages.")
		}

	case "/clear", "/new":
		// The pre-clear transcript stays in the session file behind a reset
		// marker rather than being rotated into a new one, so an accidental
		// /clear is recoverable with `sessions show --full`.
		if err := s.agent.ResetWithReason(command); err != nil {
			return false, err
		}
		s.contextEstimate.reset()
		fmt.Fprintf(os.Stderr, "%s\n", dim("transcript cleared"))

	case "/compact":
		before, after, err := s.compactContext(rest)
		if err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("compacted context: %d messages → summary + %d recent messages", before, after-1)))

	case "/reload":
		if err := s.reloadResources(); err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim("reloaded context files, prompt templates, and skills"))

	case "/resources":
		loaded := s.currentResources()
		if loaded == nil {
			fmt.Fprintln(os.Stderr, "No resources loaded.")
			return false, nil
		}
		fmt.Fprintf(os.Stderr, "Context files: %d\nPrompt templates: %d\nSkills: %d\n", len(loaded.ContextFiles), len(loaded.Templates), len(loaded.Skills))
		for _, file := range loaded.ContextFiles {
			fmt.Fprintln(os.Stderr, "  "+file.Path)
		}
		for _, template := range loaded.Templates {
			fmt.Fprintln(os.Stderr, "  /"+template.Name+" — "+template.Description)
		}
		for _, skill := range loaded.Skills {
			fmt.Fprintln(os.Stderr, "  /skill:"+skill.Name+" — "+skill.Description)
		}

	case "/ralph":
		return false, s.handleRalphSlashCommand(rest)

	case "/planner":
		if s.plan == nil {
			return false, errors.New("planner is unavailable in this session")
		}
		phase, err := s.plan.Toggle()
		if err != nil {
			return false, err
		}
		s.syncPlanRuntime()
		fmt.Fprintln(os.Stderr, dim("Planner: "+string(phase)))

	case "/planner-review":
		return false, s.reviewCode(rest)

	case "/planner-annotate":
		if rest == "" {
			return false, errors.New("usage: /planner-annotate <file>")
		}
		return false, s.annotateFile(rest)

	case "/planner-last":
		return false, s.annotateLastMessage()

	case "/use-claude-code-tui":
		if s.fullscreen {
			fmt.Fprintln(os.Stderr, dim("The fullscreen interface already uses the native sidebar layout; this switch only affects line mode."))
			return false, nil
		}
		s.claudeTUI = true
		fmt.Fprintln(os.Stderr, dim("Using native pi-claude-code-tui look"))

	case "/use-default-tui":
		if s.fullscreen {
			fmt.Fprintln(os.Stderr, dim("The fullscreen layout cannot be changed in-place; restart with -fullscreen=false for the line-mode interface."))
			return false, nil
		}
		s.claudeTUI = false
		fmt.Fprintln(os.Stderr, dim("Using default GoshCoder interface"))

	default:
		return false, fmt.Errorf("unknown command %q; /help lists the commands", command)
	}
	return false, nil
}

// handleRalphSlashCommand exposes loop management inside chat.
func (s *session) handleRalphSlashCommand(rest string) error {
	if s.loops == nil {
		return errors.New("ralph loops are disabled; restart with -ralph to enable them")
	}
	subcommand, argument, _ := strings.Cut(rest, " ")
	argument = strings.TrimSpace(argument)

	switch subcommand {
	case "start":
		name, task, _ := strings.Cut(argument, " ")
		name, task = strings.TrimSpace(name), strings.TrimSpace(task)
		if name == "" || task == "" {
			return errors.New("usage: /ralph start <name> <task>")
		}
		if !strings.HasPrefix(task, "#") {
			task = "# Task\n\n" + task
		}
		state, err := s.loops.Start(name, task, ralph.Options{})
		if err != nil {
			return err
		}
		s.agent.SetSystemPrompt(s.runtimeSystemPrompt())
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("started Ralph loop %s (max %d iterations)", state.Name, state.MaxIterations)))
		return s.runTurn(ralph.BuildPrompt(state, task, false))

	case "", "status":
		state, ok := s.loops.Current()
		if !ok {
			fmt.Fprintln(os.Stderr, "No active loop.")
			return nil
		}
		fmt.Fprintln(os.Stderr, state.Summary())

	case "list":
		states := s.loops.List(false)
		if len(states) == 0 {
			fmt.Fprintln(os.Stderr, "No loops.")
			return nil
		}
		for _, state := range states {
			fmt.Fprintln(os.Stderr, state.Summary())
		}

	case "resume":
		if argument == "" {
			return errors.New("usage: /ralph resume <name>")
		}
		state, err := s.loops.Resume(argument)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("resumed %s at iteration %d", state.Name, state.Iteration)))

	case "stop":
		state, ok := s.loops.Current()
		if !ok {
			if argument == "" {
				return errors.New("no active loop; pass a name to stop a specific loop")
			}
			loaded, found := s.loops.Load(argument, false)
			if !found {
				return fmt.Errorf("no loop named %q", argument)
			}
			state = loaded
		}
		if err := s.loops.Complete(state); err != nil {
			return err
		}
		// A stopped loop should no longer steer the model, while other active
		// modes continue to contribute their instructions.
		s.agent.SetSystemPrompt(s.runtimeSystemPrompt())
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("stopped %s at iteration %d", state.Name, state.Iteration)))

	default:
		return fmt.Errorf("unknown ralph subcommand %q (start, status, list, resume, stop)", subcommand)
	}
	return nil
}

func terminalWidth() int {
	if width, err := strconv.Atoi(os.Getenv("COLUMNS")); err == nil && width > 0 {
		return width
	}
	return 80
}

func (s *session) sessionInfo() claudetui.SessionInfo {
	info := s.sessionInfoRealtime()
	s.enrichSessionGit(&info)
	return info
}

// sessionInfoRealtime avoids subprocesses and is safe to refresh for every
// agent event. Git state is layered in asynchronously by the fullscreen UI.
func (s *session) sessionInfoRealtime() claudetui.SessionInfo {
	state := s.agent.State()
	info := claudetui.SessionInfo{
		Model:        s.model.Provider + "/" + s.model.ID,
		ContextLimit: s.model.ContextWindow,
		Messages:     len(state.Messages),
		Tools:        len(state.Tools),
		Mode:         "normal",
		Thinking:     state.ThinkingLevel,
	}
	if s.plan != nil && s.plan.State().Phase != plannotator.PhaseIdle {
		info.Mode = string(s.plan.State().Phase)
	}
	hasCompaction := false
	for _, message := range state.Messages {
		if info.Name == "" {
			switch value := message.(type) {
			case llm.UserMessage:
				info.Name = sessionTitle(userText(value))
			case *llm.UserMessage:
				if value != nil {
					info.Name = sessionTitle(userText(*value))
				}
			}
		}
		switch message.(type) {
		case compactionSummaryMessage, *compactionSummaryMessage:
			hasCompaction = true
		}
		var assistant *llm.AssistantMessage
		switch value := message.(type) {
		case llm.AssistantMessage:
			assistant = &value
		case *llm.AssistantMessage:
			assistant = value
		}
		if assistant != nil && assistant.Usage.TotalTokens > 0 {
			info.ContextUsed = assistant.Usage.TotalTokens
		}
	}
	info.Cost = conversationCost(state.Messages)
	if hasCompaction {
		info.ContextUsed = s.contextEstimate.estimate(state.Messages)
	}
	return info
}

func sessionTitle(text string) string {
	text = strings.TrimSpace(strings.ReplaceAll(text, "\n", " "))
	if text == "" {
		return ""
	}
	return strings.Join(strings.Fields(text), " ")
}

func (s *session) enrichSessionGit(info *claudetui.SessionInfo) {
	statusContext, cancelStatus := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancelStatus()
	status := exec.CommandContext(statusContext, "git", "status", "--short", "--branch", "--untracked-files=normal")
	status.WaitDelay = time.Second
	status.Dir = s.workspaceRoot()
	if output, _, err := runCommandLimited(status, 1<<20); err == nil {
		lines := strings.Split(strings.TrimSpace(output), "\n")
		for index, line := range lines {
			if index == 0 && strings.HasPrefix(line, "## ") {
				branch := strings.TrimPrefix(line, "## ")
				branch, _, _ = strings.Cut(branch, "...")
				if strings.HasPrefix(branch, "HEAD ") {
					branch = "detached HEAD"
				}
				info.Branch = branch
				continue
			}
			if line != "" {
				info.ChangedFiles++
				status, path := parseGitStatusLine(line)
				if path != "" && len(info.Changes) < 50 {
					info.Changes = append(info.Changes, claudetui.ChangedFile{Path: path, Status: status})
				}
			}
		}
	}
}

func parseGitStatusLine(line string) (status, path string) {
	if len(line) < 3 {
		return "", ""
	}
	status = strings.TrimSpace(line[:2])
	path = strings.TrimSpace(line[3:])
	if _, renamed, ok := strings.Cut(path, " -> "); ok {
		path = renamed
	}
	path = strings.Trim(path, `"`)
	return status, filepath.ToSlash(path)
}

type limitedCommandOutput struct {
	mu        sync.Mutex
	builder   strings.Builder
	limit     int
	truncated bool
}

func (output *limitedCommandOutput) Write(data []byte) (int, error) {
	output.mu.Lock()
	defer output.mu.Unlock()
	originalLength := len(data)
	remaining := output.limit - output.builder.Len()
	if remaining <= 0 {
		output.truncated = output.truncated || originalLength > 0
		return originalLength, nil
	}
	if len(data) > remaining {
		data = data[:remaining]
		output.truncated = true
	}
	_, _ = output.builder.Write(data)
	return originalLength, nil
}

func runCommandLimited(command *exec.Cmd, limit int) (string, bool, error) {
	output := &limitedCommandOutput{limit: limit}
	command.Stdout, command.Stderr = output, output
	err := command.Run()
	output.mu.Lock()
	defer output.mu.Unlock()
	return output.builder.String(), output.truncated, err
}

func (s *session) workspaceRoot() string {
	if s.workspace != nil {
		return s.workspace.Root
	}
	root, err := os.Getwd()
	if err != nil {
		return "."
	}
	return root
}

func (s *session) browserReview(title, content string) (plannotator.Decision, error) {
	reviewer := plannotator.BrowserReviewer{Notify: func(message string) { fmt.Fprintln(os.Stderr, message) }}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	return reviewer.Review(ctx, title, content)
}

func (s *session) deliverReviewFeedback(subject string, decision plannotator.Decision) error {
	feedback := strings.TrimSpace(decision.Feedback)
	if decision.Approved && feedback == "" {
		fmt.Fprintln(os.Stderr, dim(subject+" approved"))
		return nil
	}
	if feedback == "" {
		feedback = subject + " was denied without notes."
	}
	status := "denied"
	if decision.Approved {
		status = "approved with notes"
	}
	return s.runTurn(fmt.Sprintf("Planner review of %s was %s. Address this feedback:\n\n%s", subject, status, feedback))
}

func (s *session) reviewCode(argument string) error {
	target := strings.TrimSpace(argument)
	var cmd *exec.Cmd
	commandContext, cancelCommand := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelCommand()
	if strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") {
		cmd = exec.CommandContext(commandContext, "gh", "pr", "diff", target)
	} else {
		cmd = exec.CommandContext(commandContext, "git", "diff", "--no-ext-diff", "--")
	}
	cmd.WaitDelay = 2 * time.Second
	cmd.Dir = s.workspaceRoot()
	content, truncated, err := runCommandLimited(cmd, 4<<20)
	if err != nil {
		return fmt.Errorf("load review diff: %w: %s", err, strings.TrimSpace(content))
	}
	if truncated {
		return errors.New("review diff exceeds the 4 MiB limit")
	}
	if len(content) == 0 && target == "" {
		cmd = exec.CommandContext(commandContext, "git", "diff", "--cached", "--no-ext-diff", "--")
		cmd.WaitDelay = 2 * time.Second
		cmd.Dir = s.workspaceRoot()
		content, truncated, err = runCommandLimited(cmd, 4<<20)
		if err != nil {
			return fmt.Errorf("git diff --cached: %w", err)
		}
		if truncated {
			return errors.New("review diff exceeds the 4 MiB limit")
		}
	}
	if len(content) == 0 {
		return errors.New("there are no changes to review")
	}
	decision, err := s.browserReview("Review code changes", "```diff\n"+content+"\n```")
	if err != nil {
		return err
	}
	return s.deliverReviewFeedback("the code changes", decision)
}

func (s *session) annotateFile(path string) error {
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		response, err := (&http.Client{Timeout: 30 * time.Second}).Get(path)
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			return fmt.Errorf("fetch %s: status %d", path, response.StatusCode)
		}
		content, err := io.ReadAll(io.LimitReader(response.Body, (2<<20)+1))
		if err != nil {
			return err
		}
		if len(content) > 2<<20 {
			return errors.New("document is too large to annotate (maximum 2MB)")
		}
		decision, err := s.browserReview("Annotate "+path, string(content))
		if err != nil {
			return err
		}
		return s.deliverReviewFeedback(path, decision)
	}

	candidate := path
	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(s.workspaceRoot(), candidate)
	}
	info, err := os.Stat(candidate)
	if err != nil {
		return err
	}
	var content []byte
	if info.IsDir() {
		var combined strings.Builder
		err = filepath.WalkDir(candidate, func(filePath string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if entry.IsDir() {
				name := entry.Name()
				if filePath != candidate && (strings.HasPrefix(name, ".") || name == "node_modules" || name == "vendor" || name == "dist" || name == "build" || name == ".venv") {
					return filepath.SkipDir
				}
				return nil
			}
			ext := strings.ToLower(filepath.Ext(entry.Name()))
			if ext != ".md" && ext != ".mdx" && ext != ".txt" && ext != ".html" && ext != ".htm" {
				return nil
			}
			relative, _ := filepath.Rel(candidate, filePath)
			heading := fmt.Sprintf("\n\n## %s\n\n", filepath.ToSlash(relative))
			remaining := (2 << 20) - combined.Len() - len(heading)
			data, readErr := readFileLimited(filePath, remaining)
			if readErr != nil {
				return readErr
			}
			combined.WriteString(heading)
			combined.Write(data)
			return nil
		})
		if err != nil {
			return err
		}
		content = []byte(combined.String())
		if len(content) == 0 {
			return errors.New("folder has no annotatable text files")
		}
	} else {
		content, err = readFileLimited(candidate, 2<<20)
		if err != nil {
			return err
		}
	}
	decision, err := s.browserReview("Annotate "+filepath.Base(candidate), string(content))
	if err != nil {
		return err
	}
	return s.deliverReviewFeedback(candidate, decision)
}

func readFileLimited(path string, limit int) ([]byte, error) {
	if limit < 0 {
		return nil, errors.New("content exceeds the annotation limit")
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if len(content) > limit {
		return nil, errors.New("content exceeds the 2 MiB annotation limit")
	}
	return content, nil
}

func (s *session) annotateLastMessage() error {
	messages := s.agent.State().Messages
	for index := len(messages) - 1; index >= 0; index-- {
		var content string
		switch message := messages[index].(type) {
		case llm.AssistantMessage:
			content = blockSummary(message.Content)
		case *llm.AssistantMessage:
			if message != nil {
				content = blockSummary(message.Content)
			}
		default:
			continue
		}
		if strings.TrimSpace(content) == "" {
			continue
		}
		decision, err := s.browserReview("Annotate last assistant response", content)
		if err != nil {
			return err
		}
		return s.deliverReviewFeedback("the previous assistant response", decision)
	}
	return errors.New("no assistant response found")
}

// isThinkingLevel reports whether level is a recognized thinking level.
func isThinkingLevel(level string) bool {
	switch level {
	case llm.ThinkingOff, llm.ThinkingMinimal, llm.ThinkingLow,
		llm.ThinkingMedium, llm.ThinkingHigh, llm.ThinkingXHigh, llm.ThinkingMax:
		return true
	default:
		return false
	}
}

// printTranscriptSummary prints one line per transcript message.
func printTranscriptSummary(messages []agent.Message) {
	if len(messages) == 0 {
		fmt.Fprintln(os.Stderr, "The transcript is empty.")
		return
	}
	for i, message := range messages {
		role := agent.RoleOf(message)
		fmt.Fprintf(os.Stderr, "%3d  %-10s %s\n", i+1, role, dim(summarizeMessage(message)))
	}
}

// summarizeMessage renders a one-line preview of a transcript message.
func summarizeMessage(message agent.Message) string {
	switch m := message.(type) {
	case llm.UserMessage:
		if text, ok := m.StringContent(); ok {
			return firstLine(text)
		}
		return firstLine(blockSummary(m.BlockContent()))
	case llm.AssistantMessage:
		return firstLine(blockSummary(m.Content))
	case llm.ToolResultMessage:
		return firstLine(m.ToolName + ": " + blockSummary(m.Content))
	default:
		return ""
	}
}

// blockSummary renders content blocks compactly, naming non-text blocks.
func blockSummary(blocks []llm.ContentBlock) string {
	var parts []string
	for _, block := range blocks {
		switch b := block.(type) {
		case llm.TextContent:
			if strings.TrimSpace(b.Text) != "" {
				parts = append(parts, b.Text)
			}
		case llm.ThinkingContent:
			parts = append(parts, "[thinking]")
		case llm.ToolCall:
			parts = append(parts, "["+b.Name+"]")
		case llm.ImageContent:
			parts = append(parts, "[image]")
		}
	}
	return strings.Join(parts, " ")
}
