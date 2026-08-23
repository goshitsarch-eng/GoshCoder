package llm

// Native port of omniroute-pi-ext-integration's chat-only tool fallback.
// Some web-synchronized OmniRoute models return prose but cannot emit OpenAI
// tool_calls. This adapter renders tool schemas into a text protocol, buffers
// the response, parses literal <tool_call> blocks, and emits normal Pi tool
// events so the agent loop remains unchanged.

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync/atomic"
	"time"
)

const APIOmniPromptTools = "omni-prompt-tools"

var omniToolCallSequence atomic.Uint64
var omniToolCallPattern = regexp.MustCompile(`(?s)<tool_call>\s*(.*?)\s*</tool_call>`)
var omniOpeningFence = regexp.MustCompile(`(?i)^\s*` + "```" + `(?:json)?\s*\n?`)
var omniClosingFence = regexp.MustCompile(`\n?\s*` + "```" + `\s*$`)

func init() {
	RegisterStreamer(APIOmniPromptTools, StreamFuncs{
		Stream: func(model *Model, request *Context, options any) *AssistantMessageEventStream {
			simple, _ := options.(*SimpleStreamOptions)
			return streamOmniPromptTools(model, request, simple)
		},
		StreamSimple: streamOmniPromptTools,
	})
}

type omniParsedToolCall struct {
	Name      string
	Arguments map[string]any
}

func streamOmniPromptTools(model *Model, request *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	stream := NewAssistantMessageEventStream()
	go func() {
		output := &AssistantMessage{Role: "assistant", Content: []ContentBlock{}, API: model.API,
			Provider: model.Provider, Model: model.ID, StopReason: StopPending, Timestamp: time.Now().UnixMilli()}
		stream.Push(AssistantMessageEvent{Type: EventStart, Partial: output})
		ctx := context.Background()
		if options != nil && options.Ctx != nil {
			ctx = options.Ctx
		}

		innerRequest := &Context{SystemPrompt: request.SystemPrompt, Messages: flattenOmniMessages(request.Messages)}
		if len(request.Tools) > 0 {
			protocol := renderOmniToolProtocol(request.Tools)
			innerRequest.SystemPrompt = strings.TrimSpace(innerRequest.SystemPrompt + "\n\n" + protocol)
		}
		innerModel := *model
		innerModel.API = APIOpenAICompletions
		inner := StreamSimple(&innerModel, innerRequest, options)
		result, err := inner.Result(ctx)
		if err != nil {
			finishOmniPromptError(stream, output, StopError, err.Error())
			return
		}
		if result == nil {
			finishOmniPromptError(stream, output, StopError, "OmniRoute returned no assistant message")
			return
		}
		output.Usage, output.ResponseID, output.ResponseModel = result.Usage, result.ResponseID, result.ResponseModel
		if result.StopReason == StopError || result.StopReason == StopAborted {
			finishOmniPromptError(stream, output, result.StopReason, result.ErrorMessage)
			return
		}

		raw := omniContentText(result.Content)
		prose, calls, parseErrors := parseOmniToolCalls(raw)
		if len(parseErrors) > 0 {
			note := "[omni-prompt-tools] Could not parse tool call(s). Re-emit each as:\n" +
				`<tool_call>{"name":"tool_name","arguments":{}}</tool_call>` + "\n- " + strings.Join(parseErrors, "\n- ")
			if prose != "" {
				prose += "\n\n"
			}
			prose += note
		}
		if prose != "" {
			index := len(output.Content)
			output.Content = append(output.Content, TextContent{Type: "text", Text: prose})
			stream.Push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: index, Partial: snapshotOmniMessage(output)})
			stream.Push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: index, Delta: prose, Partial: snapshotOmniMessage(output)})
			stream.Push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: index, Content: prose, Partial: snapshotOmniMessage(output)})
		}
		for _, call := range calls {
			index := len(output.Content)
			id := fmt.Sprintf("call_omni_%x_%d", time.Now().UnixMilli(), omniToolCallSequence.Add(1))
			toolCall := ToolCall{Type: "toolCall", ID: id, Name: call.Name, Arguments: call.Arguments}
			output.Content = append(output.Content, toolCall)
			encoded, _ := json.Marshal(call.Arguments)
			stream.Push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: index, Partial: snapshotOmniMessage(output)})
			stream.Push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: index, Delta: string(encoded), Partial: snapshotOmniMessage(output)})
			stream.Push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: index, ToolCall: &toolCall, Partial: snapshotOmniMessage(output)})
		}
		// Keep the inner stream's stop reason. Overwriting StopLength here hid
		// truncation from the agent loop's guard, so tool calls parsed out of
		// a response that was cut off mid-generation were executed as if the
		// model had finished asking for them.
		output.StopReason = result.StopReason
		output.RawStopReason = result.RawStopReason
		switch {
		case output.StopReason == StopLength:
			// leave it: the response was truncated
		case len(calls) > 0:
			output.StopReason = StopToolUse
		default:
			output.StopReason = StopStop
		}
		CalculateCost(model, &output.Usage)
		stream.Push(AssistantMessageEvent{Type: EventDone, Reason: output.StopReason, Message: output})
		stream.End()
	}()
	return stream
}

// snapshotOmniMessage copies the in-flight message for publication in
// event.Partial.
//
// The consumer reads event.Partial on its own goroutine (internal/agent's loop
// pulls events while this producer is still running), so handing out the live
// pointer races every subsequent append to Content and the final StopReason
// write. Every other streamer in this package publishes a copy.
func snapshotOmniMessage(message *AssistantMessage) *AssistantMessage {
	copied := *message
	copied.Content = append([]ContentBlock(nil), message.Content...)
	return &copied
}

func finishOmniPromptError(stream *AssistantMessageEventStream, output *AssistantMessage, reason StopReason, message string) {
	if reason != StopAborted {
		reason = StopError
	}
	output.StopReason, output.ErrorMessage = reason, message
	stream.Push(AssistantMessageEvent{Type: EventError, Reason: reason, Error: output})
	stream.End()
}

func renderOmniToolProtocol(tools []Tool) string {
	lines := []string{
		"# Tool calling protocol", "",
		"You are connected to GoshCoder through a chat-only adapter. Native/internal tool calling is NOT available.",
		"The only valid way to call tools is to write literal text <tool_call> blocks in your assistant message.",
		"If you need a tool, write any reasoning first, then end your message with one or more blocks exactly like:", "",
		"<tool_call>", `{"name":"<tool_name>","arguments":{}}`, "</tool_call>", "",
		"Rules:",
		"- Use only this literal <tool_call> text protocol.",
		"- JSON must be valid; arguments must be an object matching the schema.",
		"- Do not wrap the JSON in Markdown fences.",
		"- Emit multiple blocks for multiple tools, then stop and wait for <tool_result> messages.",
		"- Never claim a tool ran unless a corresponding result was returned.",
		"- Never invent command output, file contents, or tool results.",
		"- If no tool is needed, answer normally with no <tool_call> block.", "", "## Available tools", "",
	}
	for _, tool := range tools {
		lines = append(lines, "### "+tool.Name, tool.Description, "Parameters (JSON Schema):", string(tool.Parameters), "")
	}
	return strings.Join(lines, "\n")
}

func parseOmniToolCalls(text string) (string, []omniParsedToolCall, []string) {
	matches := omniToolCallPattern.FindAllStringSubmatch(text, -1)
	calls := make([]omniParsedToolCall, 0, len(matches))
	var problems []string
	for _, match := range matches {
		body := strings.TrimSpace(omniClosingFence.ReplaceAllString(omniOpeningFence.ReplaceAllString(match[1], ""), ""))
		var raw struct {
			Name      string `json:"name"`
			Arguments any    `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(body), &raw); err != nil {
			problems = append(problems, fmt.Sprintf("invalid JSON in <tool_call> (%v): %s", err, truncateOmni(body, 200)))
			continue
		}
		if raw.Name == "" {
			problems = append(problems, `tool_call missing a string "name": `+truncateOmni(body, 200))
			continue
		}
		arguments := map[string]any{}
		switch value := raw.Arguments.(type) {
		case map[string]any:
			arguments = value
		case string:
			_ = json.Unmarshal([]byte(value), &arguments)
		}
		calls = append(calls, omniParsedToolCall{Name: raw.Name, Arguments: arguments})
	}
	return strings.TrimSpace(omniToolCallPattern.ReplaceAllString(text, "")), calls, problems
}

func flattenOmniMessages(messages []Message) []Message {
	output := make([]Message, 0, len(messages))
	push := func(role, text string) {
		text = strings.TrimSpace(text)
		if text == "" {
			return
		}
		if len(output) > 0 {
			switch previous := output[len(output)-1].(type) {
			case UserMessage:
				if role == "user" {
					previous.Content = omniUserText(previous) + "\n\n" + text
					output[len(output)-1] = previous
					return
				}
			case AssistantMessage:
				if role == "assistant" {
					previous.Content = []ContentBlock{TextContent{Type: "text", Text: omniContentText(previous.Content) + "\n\n" + text}}
					output[len(output)-1] = previous
					return
				}
			}
		}
		if role == "user" {
			output = append(output, UserMessage{Role: "user", Content: text, Timestamp: time.Now().UnixMilli()})
			return
		}
		output = append(output, AssistantMessage{Role: "assistant", Content: []ContentBlock{TextContent{Type: "text", Text: text}}, StopReason: StopStop, Timestamp: time.Now().UnixMilli()})
	}
	for _, message := range messages {
		switch value := message.(type) {
		case UserMessage:
			push("user", omniUserText(value))
		case AssistantMessage:
			parts := []string{omniContentText(value.Content)}
			for _, block := range value.Content {
				if call, ok := block.(ToolCall); ok {
					encoded, _ := json.Marshal(map[string]any{"name": call.Name, "arguments": call.Arguments})
					parts = append(parts, "<tool_call>\n"+string(encoded)+"\n</tool_call>")
				}
			}
			push("assistant", strings.Join(nonEmptyOmni(parts), "\n\n"))
		case ToolResultMessage:
			tag := "tool_result"
			if value.IsError {
				tag += " error"
			}
			push("user", fmt.Sprintf("<%s tool=%q id=%q>\n%s\n</tool_result>", tag, value.ToolName, value.ToolCallID, omniContentText(value.Content)))
		}
	}
	return output
}

func omniUserText(message UserMessage) string {
	if value, ok := message.StringContent(); ok {
		return value
	}
	return omniContentText(message.BlockContent())
}
func omniContentText(content []ContentBlock) string {
	var parts []string
	for _, block := range content {
		if text, ok := block.(TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "")
}
func nonEmptyOmni(values []string) []string {
	output := values[:0]
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			output = append(output, value)
		}
	}
	return output
}
func truncateOmni(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "..."
}
