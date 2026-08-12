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
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

const chatHelp = `Slash commands:
  /help                 Show this help
  /model [ref]          Show or switch the model
  /thinking [level]     Show or set the thinking level
  /system [text]        Show or replace the system prompt
  /tools                List the available tools
  /messages             Show the transcript summary
  /steer <text>         Queue a steering message for the next turn
  /followup <text>      Queue a follow-up message
  /queue                Show whether messages are queued
  /clear                Clear the transcript
  /ralph <subcommand>   Manage ralph loops (list, status, resume, stop)
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

	s, err := newSession(*cfg)
	if err != nil {
		return err
	}

	// Ctrl-C aborts a turn; when idle it closes stdin so the read loop ends.
	stop := s.handleInterrupts(func() {
		fmt.Fprintln(os.Stderr, "\n"+dim("exiting"))
		os.Stdin.Close()
	})
	defer stop()

	fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
		"goshcoder %s · %s/%s · /help for commands", Version, s.model.Provider, s.model.ID)))

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Fprint(os.Stderr, bold("\n> "))
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
