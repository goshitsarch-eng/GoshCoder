package main

import (
	"fmt"
	"io"
	"os"
	"strings"

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
	suggestion   int
}

type slashCommand struct {
	name        string
	description string
}

var fullscreenSlashCommands = []slashCommand{
	{"/help", "Show all commands"}, {"/model", "Show or switch model"},
	{"/thinking", "Choose reasoning effort for this model"}, {"/tools", "List active tools"},
	{"/status", "Show session information"}, {"/messages", "Show transcript summary"},
	{"/steer", "Guide the active response"}, {"/followup", "Queue the next message"},
	{"/queue", "Show queued messages"}, {"/clear", "Clear the transcript"},
	{"/plannotator", "Toggle planning mode"}, {"/plannotator-review", "Review code changes"},
	{"/plannotator-annotate", "Annotate a target"}, {"/plannotator-last", "Annotate last response"},
	{"/ralph", "Manage Ralph loops"}, {"/system", "Show or replace system prompt"},
	{"/use-claude-code-tui", "Enable Claude-style UI"}, {"/use-default-tui", "Use default styling"},
	{"/exit", "Exit GoshCoder"},
}

func cycleFullscreenThinking(session *session) {
	model := session.agent.State().Model
	levels := llm.GetSupportedThinkingLevels(&model)
	if len(levels) == 0 {
		return
	}
	current := session.agent.State().ThinkingLevel
	for index, level := range levels {
		if level == current {
			session.agent.SetThinkingLevel(levels[(index+1)%len(levels)])
			return
		}
	}
	session.agent.SetThinkingLevel(levels[0])
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
	go func() {
		data, _ := io.ReadAll(io.LimitReader(reader, 2<<20))
		captured <- string(data)
		_ = reader.Close()
	}()
	exit, commandErr := session.handleSlashCommand(input)
	_ = writer.Close()
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
