package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"goshcoder/internal/agent"
	"goshcoder/internal/claudetui"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
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
	{"/help", "Show all commands"}, {"/model", "Choose from authenticated models"},
	{"/thinking", "Choose reasoning effort for this model"}, {"/tools", "List active tools"},
	{"/status", "Show session information"}, {"/session", "Show session information"},
	{"/sidebar", "Show session information"}, {"/hotkeys", "Show keyboard shortcuts"},
	{"/messages", "Show transcript summary"},
	{"/steer", "Guide the active response"}, {"/followup", "Queue the next message"},
	{"/queue", "Show queued messages"}, {"/clear", "Clear the transcript"},
	{"/new", "Start a fresh conversation"}, {"/compact", "Summarize older context"},
	{"/reload", "Reload local resources"},
	{"/resources", "Show loaded context, prompts, and skills"},
	{"/planner", "Toggle planning mode"}, {"/planner-review", "Review code changes"},
	{"/planner-annotate", "Annotate a target"}, {"/planner-last", "Annotate last response"},
	{"/ralph", "Manage Ralph loops"}, {"/system", "Show or replace system prompt"},
	{"/exit", "Exit GoshCoder"}, {"/quit", "Exit GoshCoder"},
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
	toolResults := make(map[string]llm.ToolResultMessage)
	unnamedResults := make(map[string][]llm.ToolResultMessage)
	callCounts := make(map[string]int)
	for _, message := range messages {
		switch value := message.(type) {
		case llm.AssistantMessage:
			for _, block := range value.Content {
				if call, ok := block.(llm.ToolCall); ok {
					callCounts[call.Name]++
				}
			}
		case *llm.AssistantMessage:
			if value != nil {
				for _, block := range value.Content {
					if call, ok := block.(llm.ToolCall); ok {
						callCounts[call.Name]++
					}
				}
			}
		case llm.ToolResultMessage:
			if value.ToolCallID == "" {
				unnamedResults[value.ToolName] = append(unnamedResults[value.ToolName], value)
			} else {
				toolResults[value.ToolCallID] = value
			}
		case *llm.ToolResultMessage:
			if value != nil {
				if value.ToolCallID == "" {
					unnamedResults[value.ToolName] = append(unnamedResults[value.ToolName], *value)
				} else {
					toolResults[value.ToolCallID] = *value
				}
			}
		}
	}

	result := make([]tui.Message, 0, len(messages))
	for _, message := range messages {
		var assistant *llm.AssistantMessage
		switch value := message.(type) {
		case llm.UserMessage:
			result = append(result, tui.Message{Role: "user", Text: userText(value)})
		case *llm.UserMessage:
			if value != nil {
				result = append(result, tui.Message{Role: "user", Text: userText(*value)})
			}
		case llm.AssistantMessage:
			assistant = &value
		case *llm.AssistantMessage:
			assistant = value
		case llm.ToolResultMessage:
			if value.ToolCallID == "" && callCounts[value.ToolName] == 0 {
				result = append(result, toolTUIMessage(llm.ToolCall{Name: value.ToolName}, &value))
			}
		case *llm.ToolResultMessage:
			if value != nil && value.ToolCallID == "" && callCounts[value.ToolName] == 0 {
				result = append(result, toolTUIMessage(llm.ToolCall{Name: value.ToolName}, value))
			}
		}
		if assistant == nil {
			continue
		}
		result = append(result, assistantTUIMessages(*assistant)...)
		for _, block := range assistant.Content {
			if call, ok := block.(llm.ToolCall); ok {
				toolResult, found := toolResults[call.ID]
				if !found && len(unnamedResults[call.Name]) > 0 {
					toolResult = unnamedResults[call.Name][0]
					unnamedResults[call.Name] = unnamedResults[call.Name][1:]
					found = true
				}
				if found {
					result = append(result, toolTUIMessage(call, &toolResult))
				} else {
					result = append(result, toolTUIMessage(call, nil))
				}
			}
		}
	}
	for name, pending := range unnamedResults {
		for _, toolResult := range pending {
			if toolResult.IsError {
				result = append(result, toolTUIMessage(llm.ToolCall{Name: name}, &toolResult))
			}
		}
	}
	return result
}

func toolTUIMessage(call llm.ToolCall, result *llm.ToolResultMessage) tui.Message {
	title := call.Name
	arg := func(name string) string {
		if call.Arguments == nil {
			return ""
		}
		value, ok := call.Arguments[name]
		if !ok || value == nil {
			return ""
		}
		return strings.TrimSpace(fmt.Sprint(value))
	}
	switch call.Name {
	case "read", "write", "edit", "ls", "list":
		if path := arg("path"); path != "" {
			title += " " + path
		}
	case "grep":
		title += " /" + arg("pattern") + "/"
		if path := arg("path"); path != "" {
			title += " in " + path
		}
	case "find":
		title += " " + arg("pattern")
	case "bash":
		if command := arg("command"); command != "" {
			title += " " + firstLine(command)
		}
	case "planner_submit_plan":
		if path := arg("filePath"); path != "" {
			title = "submit plan " + path
		}
	default:
		if summary := summarizeArgs(call.Arguments); summary != "" {
			title += " " + summary
		}
	}
	message := tui.Message{Role: "tool", Title: title}
	if result == nil {
		message.Text = "running…"
		return message
	}
	message.IsError = result.IsError
	message.Detail = blockSummary(result.Content)
	if result.IsError {
		message.Text = firstLine(message.Detail)
		return message
	}
	switch call.Name {
	case "bash", "grep", "find", "ls", "list", "planner_submit_plan":
		lines := strings.Split(strings.TrimSpace(message.Detail), "\n")
		message.Text = strings.Join(lines[:min(3, len(lines))], "\n")
	}
	return message
}

func userText(message llm.UserMessage) string {
	if text, ok := message.StringContent(); ok {
		return text
	}
	return blockSummary(message.BlockContent())
}

func assistantTUIMessages(message llm.AssistantMessage) []tui.Message {
	var thinking, text strings.Builder
	for _, block := range message.Content {
		switch value := block.(type) {
		case llm.ThinkingContent:
			thinking.WriteString(value.Thinking)
		case llm.TextContent:
			text.WriteString(value.Text)
		}
	}
	var result []tui.Message
	if thinking.Len() > 0 {
		result = append(result, tui.Message{Role: "thinking", Text: thinking.String()})
	}
	if text.Len() > 0 {
		result = append(result, tui.Message{Role: "assistant", Text: text.String()})
	}
	if message.ErrorMessage != "" {
		result = append(result, tui.Message{Role: "Error", Text: message.ErrorMessage})
	}
	return result
}

func fullscreenSidebar(info claudetui.SessionInfo, cwd, activity string, pendingTools int, recentTool string, items []plannotator.ChecklistItem) []string {
	name := info.Name
	if name == "" {
		name = "New Session"
	}
	mode := info.Mode
	if mode == "" {
		mode = "normal"
	}
	percent := 0
	if info.ContextLimit > 0 {
		percent = min(100, info.ContextUsed*100/info.ContextLimit)
	}
	contextText := fmt.Sprintf("%s tokens", compactNumber(info.ContextUsed))
	if info.ContextLimit > 0 {
		contextText = fmt.Sprintf("%s / %s tokens", compactNumber(info.ContextUsed), compactNumber(info.ContextLimit))
	}
	lines := []string{
		"title\t" + name,
		"accent\t" + info.Model,
		"meta\t" + info.Thinking + " thinking · " + mode,
		"",
		"section\tContext",
		fmt.Sprintf("bar\t%d", percent),
		"meta\t" + contextText,
		fmt.Sprintf("meta\t%d%% used · $%.4f spent", percent, info.Cost),
	}
	if activity != "Ready" || pendingTools > 0 {
		lines = append(lines, "", "section\tActivity", "active\t"+activity)
		if pendingTools > 0 {
			lines = append(lines, fmt.Sprintf("meta\t%d tool%s running", pendingTools, map[bool]string{true: "", false: "s"}[pendingTools == 1]))
		}
		if recentTool != "" {
			lines = append(lines, "meta\t"+recentTool)
		}
	}
	if len(items) > 0 {
		lines = append(lines, "", "section\tTodo")
		for _, item := range items {
			state := "open"
			if item.Completed {
				state = "done"
			}
			lines = append(lines, "todo\t"+state+"\t"+item.Text)
		}
	}
	if len(info.Changes) > 0 {
		lines = append(lines, "", "section\tModified Files")
		for index, change := range info.Changes {
			if index >= 10 {
				lines = append(lines, fmt.Sprintf("meta\t… %d more files", len(info.Changes)-index))
				break
			}
			lines = append(lines, "file\t"+change.Status+"\t"+change.Path)
		}
	}
	branch := info.Branch
	if branch == "" {
		branch = "not a git repo"
	}
	lines = append(lines,
		"", "section\tWorkspace", "meta\t"+branch,
		"path\t"+claudetui.FormatCWD(cwd), "",
		"brand\t● GoshCoder "+Version,
	)
	return lines
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
