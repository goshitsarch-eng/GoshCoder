package main

// Interactive chat mode: the REPL equivalent of pi's TUI, reduced to a
// line-oriented interface.
//
// pi renders a full-screen terminal UI. GoshCoder keeps the same session
// semantics (queued steering, follow-ups, abort, runtime model and thinking
// changes, ralph loops) behind slash commands, so no terminal-control layer is
// needed and output stays pipeable.

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
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/claudetui"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
)

const chatHelp = `Slash commands:
  /help                 Show this help
  /model [ref]          Show or switch the model
  /thinking [level]     Show or set the thinking level
  /system [text]        Show or replace the system prompt
  /tools                List the available tools
  /messages             Show the transcript summary
  /status               Show model, context, cost, git, and mode information
  /steer <text>         Queue a steering message for the next turn
  /followup <text>      Queue a follow-up message
  /queue                Show whether messages are queued
  /clear                Clear the transcript
  /ralph <subcommand>   Manage ralph loops (list, status, resume, stop)
  /plannotator          Toggle native Plannotator planning mode
  /plannotator-review [PR URL]  Review git changes or a GitHub PR
  /plannotator-annotate <target> Annotate a file, folder, or URL
  /plannotator-last     Annotate the last assistant response
  /use-claude-code-tui  Enable the native startup/editor look
  /use-default-tui      Restore the plain line-oriented look
  /exit                 Leave chat

Anything else is sent to the model. Ctrl-C aborts a running turn; Ctrl-C while
idle exits.
`

func chatCommand(args []string) error {
	flags := flag.NewFlagSet("chat", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	cfg := bindSessionFlags(flags)
	if err := flags.Parse(args); err != nil {
		return err
	}
	// Native extension commands are available in chat even when plan mode was
	// not requested at startup.
	cfg.LoadPlannotator = true

	s, err := newSession(*cfg)
	if err != nil {
		return err
	}
	defer s.close()

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

	reader := bufio.NewReader(os.Stdin)
	for {
		if s.claudeTUI {
			// The line-oriented UI cannot pin a panel over scrolling output. Refresh
			// it automatically whenever model usage, tools, git, or mode changes.
			currentSidebar := s.sessionInfo()
			if currentSidebar != lastSidebar {
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

// userMessage builds a transcript user message.
func userMessage(text string) llm.UserMessage {
	return llm.UserMessage{Role: "user", Content: text, Timestamp: time.Now().UnixMilli()}
}

// handleSlashCommand runs one slash command. exit is true when chat should end.
func (s *session) handleSlashCommand(input string) (exit bool, err error) {
	command, rest, _ := strings.Cut(input, " ")
	rest = strings.TrimSpace(rest)

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

	case "/thinking":
		if rest == "" {
			fmt.Fprintf(os.Stderr, "%s\n", s.agent.State().ThinkingLevel)
			return false, nil
		}
		if !isThinkingLevel(rest) {
			return false, fmt.Errorf("unknown thinking level %q (off|minimal|low|medium|high|xhigh|max)", rest)
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
		s.baseSystemPrompt = rest
		s.agent.SetSystemPrompt(rest)
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

	case "/clear":
		if err := s.agent.Reset(); err != nil {
			return false, err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim("transcript cleared"))

	case "/ralph":
		return false, s.handleRalphSlashCommand(rest)

	case "/plannotator":
		if s.plan == nil {
			return false, errors.New("plannotator is unavailable in this session")
		}
		phase, err := s.plan.Toggle()
		if err != nil {
			return false, err
		}
		s.syncPlanRuntime()
		fmt.Fprintln(os.Stderr, dim("Plannotator: "+string(phase)))

	case "/plannotator-review":
		return false, s.reviewCode(rest)

	case "/plannotator-annotate":
		if rest == "" {
			return false, errors.New("usage: /plannotator-annotate <file>")
		}
		return false, s.annotateFile(rest)

	case "/plannotator-last":
		return false, s.annotateLastMessage()

	case "/use-claude-code-tui":
		s.claudeTUI = true
		fmt.Fprintln(os.Stderr, dim("Using native pi-claude-code-tui look"))

	case "/use-default-tui":
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
		// A stopped loop should no longer steer the model.
		s.agent.SetSystemPrompt(s.baseSystemPrompt)
		fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("stopped %s at iteration %d", state.Name, state.Iteration)))

	default:
		return fmt.Errorf("unknown ralph subcommand %q (status, list, resume, stop)", subcommand)
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
	for _, message := range state.Messages {
		var assistant *llm.AssistantMessage
		switch value := message.(type) {
		case llm.AssistantMessage:
			assistant = &value
		case *llm.AssistantMessage:
			assistant = value
		}
		if assistant == nil {
			continue
		}
		info.Cost += assistant.Usage.Cost.Total
		if assistant.Usage.TotalTokens > 0 {
			info.ContextUsed = assistant.Usage.TotalTokens
		}
	}
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
			}
		}
	}
	return info
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
	return s.runTurn(fmt.Sprintf("Plannotator review of %s was %s. Address this feedback:\n\n%s", subject, status, feedback))
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
				if filePath != candidate && strings.HasPrefix(entry.Name(), ".") {
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
		message, ok := messages[index].(llm.AssistantMessage)
		if !ok {
			continue
		}
		content := blockSummary(message.Content)
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
