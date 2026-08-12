package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"

	"goshcoder/internal/agent"
	"goshcoder/internal/claudetui"
	"goshcoder/internal/llm"
	"goshcoder/internal/tui"
)

type fullscreenResult struct {
	exit   bool
	output string
	err    error
}

type fullscreenEditor struct {
	input        []rune
	cursor       int
	history      []string
	historyIndex int
	draft        string
	scroll       int
	pending      []byte
}

func runFullscreenChat(session *session) error {
	terminal, err := tui.Open(os.Stdin, os.Stderr)
	if err != nil {
		return err
	}
	defer terminal.Close()

	var activityMu sync.Mutex
	activity := "Ready · Enter submit · /help commands · Ctrl-D exit"
	dirty := make(chan struct{}, 1)
	requestRender := func() {
		select {
		case dirty <- struct{}{}:
		default:
		}
	}
	unsubscribe := session.agent.Subscribe(func(_ context.Context, event agent.Event) {
		activityMu.Lock()
		switch event.Type {
		case agent.EventMessageUpdate:
			activity = "Generating response…"
		case agent.EventToolExecutionStart:
			activity = "Running " + event.ToolName + "…"
		case agent.EventToolExecutionEnd:
			if event.IsError {
				activity = event.ToolName + " failed"
			} else {
				activity = event.ToolName + " complete"
			}
		case agent.EventAgentEnd:
			activity = "Ready · Enter submit · /help commands · Ctrl-D exit"
		}
		activityMu.Unlock()
		requestRender()
	})
	defer unsubscribe()

	inputEvents := make(chan []byte, 32)
	inputErrors := make(chan error, 1)
	go readTerminalInput(os.Stdin, inputEvents, inputErrors)

	interrupts := make(chan os.Signal, 1)
	signal.Notify(interrupts, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(interrupts)

	editor := fullscreenEditor{historyIndex: -1}
	var notices []tui.Message
	info := session.sessionInfo()
	turnDone := make(chan error, 1)
	commandDone := make(chan fullscreenResult, 1)
	busyCommand := false
	lastSize := ""

	render := func() error {
		width, height := terminal.Size()
		size := fmt.Sprintf("%dx%d", width, height)
		lastSize = size
		activityMu.Lock()
		status := activity
		activityMu.Unlock()
		messages := append(fullscreenMessages(session.agent.State().Messages), notices...)
		frame := tui.Frame{
			Title:    fmt.Sprintf("GoshCoder %s  ·  %s/%s", Version, session.model.Provider, session.model.ID),
			Messages: messages, Sidebar: fullscreenSidebar(info),
			Input: editor.input, Cursor: editor.cursor, Status: status,
			Streaming: session.agent.State().IsStreaming || busyCommand, Scroll: editor.scroll,
		}
		return terminal.Render(frame)
	}

	startTurn := func(prompt string) {
		if strings.TrimSpace(prompt) == "" {
			return
		}
		editor.history = append(editor.history, prompt)
		editor.historyIndex, editor.draft = -1, ""
		editor.input, editor.cursor, editor.scroll = nil, 0, 0
		go func() { turnDone <- session.runTurn(prompt) }()
	}

	requestRender()
	ticker := time.NewTicker(80 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-dirty:
			if err := render(); err != nil {
				return err
			}
		case <-ticker.C:
			width, height := terminal.Size()
			if fmt.Sprintf("%dx%d", width, height) != lastSize {
				if err := render(); err != nil {
					return err
				}
			}
		case data := <-inputEvents:
			action := handleFullscreenInput(&editor, data, session.agent.State().IsStreaming)
			switch action {
			case "exit":
				if session.agent.State().IsStreaming {
					session.agent.Abort()
				} else {
					return nil
				}
			case "abort":
				session.agent.Abort()
			case "submit":
				prompt := strings.TrimSpace(string(editor.input))
				if prompt == "" {
					break
				}
				if session.agent.State().IsStreaming {
					session.agent.Steer(userMessage(prompt))
					editor.input, editor.cursor = nil, 0
					activityMu.Lock()
					activity = "Steering message queued"
					activityMu.Unlock()
				} else if !busyCommand && strings.HasPrefix(prompt, "/") {
					editor.history = append(editor.history, prompt)
					editor.input, editor.cursor, editor.historyIndex = nil, 0, -1
					if prompt == "/exit" || prompt == "/quit" {
						return nil
					}
					busyCommand = true
					go func() { commandDone <- runFullscreenCommand(session, prompt) }()
				} else if !busyCommand {
					startTurn(prompt)
				}
			}
			requestRender()
		case err := <-inputErrors:
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		case err := <-turnDone:
			if err != nil {
				notices = appendNotice(notices, "Error", err.Error())
			}
			info = session.sessionInfo()
			requestRender()
		case result := <-commandDone:
			busyCommand = false
			if result.output != "" {
				notices = appendNotice(notices, "Command", result.output)
			}
			if result.err != nil {
				notices = appendNotice(notices, "Error", result.err.Error())
			}
			if result.exit {
				return nil
			}
			info = session.sessionInfo()
			activityMu.Lock()
			activity = "Ready · Enter submit · /help commands · Ctrl-D exit"
			activityMu.Unlock()
			requestRender()
		case <-interrupts:
			if session.agent.State().IsStreaming {
				session.agent.Abort()
				requestRender()
			} else {
				return nil
			}
		}
	}
}

func readTerminalInput(input *os.File, events chan<- []byte, failures chan<- error) {
	reader := bufio.NewReader(input)
	for {
		data := make([]byte, 256)
		count, err := reader.Read(data)
		if count > 0 {
			events <- append([]byte(nil), data[:count]...)
		}
		if err != nil {
			failures <- err
			return
		}
	}
}

func handleFullscreenInput(editor *fullscreenEditor, data []byte, streaming bool) string {
	if len(editor.pending) > 0 {
		data = append(append([]byte(nil), editor.pending...), data...)
		editor.pending = nil
	}
	for len(data) > 0 {
		switch {
		case strings.HasPrefix(string(data), "\x1b[5~"):
			editor.scroll += 10
			data = data[4:]
			continue
		case strings.HasPrefix(string(data), "\x1b[6~"):
			editor.scroll = max(0, editor.scroll-10)
			data = data[4:]
			continue
		case strings.HasPrefix(string(data), "\x1b[D"):
			editor.cursor = max(0, editor.cursor-1)
			data = data[3:]
			continue
		case strings.HasPrefix(string(data), "\x1b[C"):
			editor.cursor = min(len(editor.input), editor.cursor+1)
			data = data[3:]
			continue
		case strings.HasPrefix(string(data), "\x1b[A"):
			editorHistory(editor, -1)
			data = data[3:]
			continue
		case strings.HasPrefix(string(data), "\x1b[B"):
			editorHistory(editor, 1)
			data = data[3:]
			continue
		}
		value := data[0]
		data = data[1:]
		switch value {
		case 3: // Ctrl-C: clear first, then abort/exit.
			if len(editor.input) > 0 {
				editor.input, editor.cursor = nil, 0
				continue
			}
			if streaming {
				return "abort"
			}
			return "exit"
		case 4: // Ctrl-D.
			if len(editor.input) == 0 {
				return "exit"
			}
			if editor.cursor < len(editor.input) {
				editor.input = append(editor.input[:editor.cursor], editor.input[editor.cursor+1:]...)
			}
		case 27:
			if streaming {
				return "abort"
			}
		case 13, 10:
			return "submit"
		case 127, 8:
			if editor.cursor > 0 {
				editor.input = append(editor.input[:editor.cursor-1], editor.input[editor.cursor:]...)
				editor.cursor--
			}
		case 1:
			editor.cursor = 0
		case 5:
			editor.cursor = len(editor.input)
		case 11:
			editor.input = editor.input[:editor.cursor]
		case 21:
			editor.input = append([]rune(nil), editor.input[editor.cursor:]...)
			editor.cursor = 0
		case 23:
			for editor.cursor > 0 && editor.input[editor.cursor-1] == ' ' {
				editor.input = append(editor.input[:editor.cursor-1], editor.input[editor.cursor:]...)
				editor.cursor--
			}
			for editor.cursor > 0 && editor.input[editor.cursor-1] != ' ' {
				editor.input = append(editor.input[:editor.cursor-1], editor.input[editor.cursor:]...)
				editor.cursor--
			}
		default:
			candidate := append([]byte{value}, data...)
			if value >= utf8.RuneSelf && !utf8.FullRune(candidate) {
				editor.pending = append(editor.pending[:0], candidate...)
				return ""
			}
			r, size := utf8.DecodeRune(candidate)
			if r == utf8.RuneError && size == 1 {
				continue
			}
			if size > 1 {
				data = data[size-1:]
			}
			if r >= 32 {
				editor.input = append(editor.input, 0)
				copy(editor.input[editor.cursor+1:], editor.input[editor.cursor:])
				editor.input[editor.cursor] = r
				editor.cursor++
			}
		}
	}
	return ""
}

func editorHistory(editor *fullscreenEditor, direction int) {
	if len(editor.history) == 0 {
		return
	}
	if editor.historyIndex < 0 {
		if direction > 0 {
			return
		}
		editor.draft = string(editor.input)
		editor.historyIndex = len(editor.history) - 1
	} else {
		editor.historyIndex += direction
		if editor.historyIndex < 0 {
			editor.historyIndex = 0
		}
		if editor.historyIndex >= len(editor.history) {
			editor.historyIndex = -1
			editor.input = []rune(editor.draft)
			editor.cursor = len(editor.input)
			return
		}
	}
	editor.input = []rune(editor.history[editor.historyIndex])
	editor.cursor = len(editor.input)
}

func runFullscreenCommand(session *session, input string) fullscreenResult {
	oldStderr := os.Stderr
	reader, writer, err := os.Pipe()
	if err != nil {
		return fullscreenResult{err: err}
	}
	os.Stderr = writer
	captured := make(chan string, 1)
	go func() { data, _ := io.ReadAll(io.LimitReader(reader, 2<<20)); captured <- string(data); reader.Close() }()
	exit, commandErr := session.handleSlashCommand(input)
	writer.Close()
	os.Stderr = oldStderr
	return fullscreenResult{exit: exit, output: strings.TrimSpace(stripTerminalStyles(<-captured)), err: commandErr}
}

func fullscreenMessages(messages []agent.Message) []tui.Message {
	result := make([]tui.Message, 0, len(messages))
	for _, message := range messages {
		switch value := message.(type) {
		case llm.UserMessage:
			result = append(result, tui.Message{Role: "user", Text: userText(value)})
		case *llm.UserMessage:
			if value != nil {
				result = append(result, tui.Message{Role: "user", Text: userText(*value)})
			}
		case llm.AssistantMessage:
			result = append(result, assistantTUIMessages(value)...)
		case *llm.AssistantMessage:
			if value != nil {
				result = append(result, assistantTUIMessages(*value)...)
			}
		case llm.ToolResultMessage:
			result = append(result, tui.Message{Role: "tool", Text: value.ToolName + ": " + blockSummary(value.Content)})
		case *llm.ToolResultMessage:
			if value != nil {
				result = append(result, tui.Message{Role: "tool", Text: value.ToolName + ": " + blockSummary(value.Content)})
			}
		}
	}
	return result
}

func userText(message llm.UserMessage) string {
	if text, ok := message.StringContent(); ok {
		return text
	}
	return blockSummary(message.BlockContent())
}
func assistantTUIMessages(message llm.AssistantMessage) []tui.Message {
	var thinking, text strings.Builder
	var tools []string
	for _, block := range message.Content {
		switch value := block.(type) {
		case llm.ThinkingContent:
			thinking.WriteString(value.Thinking)
		case llm.TextContent:
			text.WriteString(value.Text)
		case llm.ToolCall:
			tools = append(tools, value.Name+" "+summarizeArgs(value.Arguments))
		}
	}
	var result []tui.Message
	if thinking.Len() > 0 {
		result = append(result, tui.Message{Role: "thinking", Text: thinking.String()})
	}
	if text.Len() > 0 {
		result = append(result, tui.Message{Role: "assistant", Text: text.String()})
	}
	if len(tools) > 0 {
		result = append(result, tui.Message{Role: "tool", Text: "Calling " + strings.Join(tools, ", ")})
	}
	if message.ErrorMessage != "" {
		result = append(result, tui.Message{Role: "Error", Text: message.ErrorMessage})
	}
	return result
}

func fullscreenSidebar(info claudetui.SessionInfo) []string {
	contextText := compactNumber(info.ContextUsed)
	if info.ContextLimit > 0 {
		contextText += fmt.Sprintf("/%s · %d%%", compactNumber(info.ContextLimit), min(100, info.ContextUsed*100/info.ContextLimit))
	}
	branch := info.Branch
	if branch == "" {
		branch = "not a git repo"
	}
	mode := info.Mode
	if mode == "" {
		mode = "normal"
	}
	return []string{"", " SESSION", "", " Model", " " + info.Model, "", " Context", " " + contextText, "", fmt.Sprintf(" Cost  $%.4f", info.Cost), fmt.Sprintf(" Messages  %d", info.Messages), fmt.Sprintf(" Tools  %d", info.Tools), fmt.Sprintf(" Files  %d changed", info.ChangedFiles), "", " Branch", " " + branch, "", " Mode", " " + mode + " · " + info.Thinking}
}
func compactNumber(value int) string {
	if value >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(value)/1_000_000)
	}
	if value >= 1_000 {
		return fmt.Sprintf("%.1fk", float64(value)/1_000)
	}
	return fmt.Sprint(value)
}
func appendNotice(messages []tui.Message, role, text string) []tui.Message {
	messages = append(messages, tui.Message{Role: role, Text: text})
	if len(messages) > 20 {
		messages = append([]tui.Message(nil), messages[len(messages)-20:]...)
	}
	return messages
}
func stripTerminalStyles(text string) string {
	var output strings.Builder
	for index := 0; index < len(text); {
		if text[index] == 0x1b && index+1 < len(text) && text[index+1] == '[' {
			index += 2
			for index < len(text) && (text[index] < '@' || text[index] > '~') {
				index++
			}
			if index < len(text) {
				index++
			}
			continue
		}
		output.WriteByte(text[index])
		index++
	}
	return output.String()
}
