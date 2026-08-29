package computeruse

// The mcp proxy tool. pi reaches the desktop tools through pi-mcp-adapter's
// mcp() tool; GoshCoder registers a native equivalent scoped to the
// computer-use-linux server, covering the documented call shapes:
//
//	mcp({server: "computer-use-linux"})              list all tools
//	mcp({search: "windows"})                          search tools
//	mcp({tool: "computer_use_linux_doctor"})          call a tool
//	mcp({tool: "...", args: {...}})                   call with arguments
//
// The tool description folds in the operating procedure from the package's
// skill (skills/computer-use-linux/SKILL.md) so the model drives the desktop
// the way the upstream skill teaches: doctor first, semantic targeting over
// pixel coordinates, explicit window targets for input, and re-checking
// state after mutations.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

const maxTextOutput = 50 << 10

// toolDescription summarizes the server surface and the skill's procedure.
const toolDescription = "Observe and control the local Linux desktop through the computer-use-linux MCP server: " +
	"accessibility trees, window targeting, screenshots, and input synthesis. " +
	"Call {\"server\": \"computer-use-linux\"} to list the desktop tools, {\"search\": \"...\"} to find one, " +
	"and {\"tool\": \"computer_use_linux_<name>\", \"args\": {...}} to run it. " +
	"Procedure: start with the doctor tool and fix blockers it reports (setup_accessibility, setup_window_targeting); " +
	"verify the intended window with list_windows or focused_window before targeted input; " +
	"prefer element indices and role/name/text selectors from get_app_state over pixel coordinates; " +
	"pass explicit window or terminal targets to type_text and press_key instead of relying on focus; " +
	"re-check state after mutating actions. " +
	"click, drag, press_key, type_text, perform_action, and set_value change real application state — " +
	"desktop input is stateful, so never issue concurrent calls."

// Tool builds the mcp proxy tool over a lazily-started session. The session
// is shared for the life of the GoshCoder session; the caller owns Close.
func Tool(session *Session) agent.Tool {
	return agent.Tool{
		Name:        "mcp",
		Label:       "Desktop MCP",
		Description: toolDescription,
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"server":{"type":"string","description":"List every tool of this MCP server. The only available server is computer-use-linux."},
			"search":{"type":"string","description":"Search desktop tools by name or description."},
			"tool":{"type":"string","description":"Tool to call, e.g. computer_use_linux_doctor or computer_use_linux_screenshot (the bare MCP name works too)."},
			"args":{"type":"object","description":"Arguments for the tool being called, matching its schema.","additionalProperties":true}
		}}`),
		// Desktop input is stateful; the whole batch runs sequentially.
		ExecutionMode: agent.ToolExecutionSequential,
		Execute: func(ctx context.Context, _ string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			toolName, _ := params["tool"].(string)
			search, _ := params["search"].(string)
			server, _ := params["server"].(string)

			switch {
			case toolName != "":
				args, _ := params["args"].(map[string]any)
				return callTool(ctx, session, toolName, args, onUpdate)
			case search != "":
				return searchTools(ctx, session, search)
			case server != "":
				if server != ServerName {
					return textResult(fmt.Sprintf("Unknown MCP server %q. The only available server is %q.", server, ServerName)), nil
				}
				return listTools(ctx, session)
			default:
				return textResult("Pass {\"server\": \"" + ServerName + "\"} to list the desktop tools, " +
					"{\"search\": \"...\"} to find one, or {\"tool\": \"computer_use_linux_<name>\", \"args\": {...}} to call one."), nil
			}
		},
	}
}

func listTools(ctx context.Context, session *Session) (agent.ToolResult, error) {
	tools, err := session.Tools(ctx)
	if err != nil {
		return agent.ToolResult{}, describeSessionError(err)
	}
	lines := []string{fmt.Sprintf("%d tool(s) from %s:", len(tools), ServerName), ""}
	for _, tool := range tools {
		lines = append(lines, toolLine(tool))
	}
	return textResult(strings.Join(lines, "\n")), nil
}

func searchTools(ctx context.Context, session *Session, query string) (agent.ToolResult, error) {
	tools, err := session.Tools(ctx)
	if err != nil {
		return agent.ToolResult{}, describeSessionError(err)
	}
	needle := strings.ToLower(strings.TrimSpace(query))
	var matches []ToolInfo
	for _, tool := range tools {
		if strings.Contains(strings.ToLower(tool.Name), needle) ||
			strings.Contains(strings.ToLower(tool.Description), needle) {
			matches = append(matches, tool)
		}
	}
	if len(matches) == 0 {
		return textResult(fmt.Sprintf("No desktop tools match %q. Call {\"server\": %q} to list them all.", query, ServerName)), nil
	}
	if len(matches) == 1 {
		return textResult(describeTool(matches[0])), nil
	}
	lines := make([]string, 0, len(matches))
	for _, tool := range matches {
		lines = append(lines, toolLine(tool))
	}
	return textResult(strings.Join(lines, "\n")), nil
}

func toolLine(tool ToolInfo) string {
	description := strings.Join(strings.Fields(tool.Description), " ")
	if len(description) > 120 {
		description = description[:117] + "..."
	}
	if description == "" {
		description = "(no description)"
	}
	mutability := ""
	if tool.Annotations != nil {
		switch {
		case tool.Annotations.ReadOnlyHint != nil && *tool.Annotations.ReadOnlyHint:
			mutability = " [read-only]"
		case tool.Annotations.DestructiveHint != nil && *tool.Annotations.DestructiveHint:
			mutability = " [destructive]"
		default:
			mutability = " [mutating]"
		}
	}
	return fmt.Sprintf("- `%s`%s: %s", PrefixedToolName(tool.Name), mutability, description)
}

func describeTool(tool ToolInfo) string {
	lines := []string{"### " + PrefixedToolName(tool.Name)}
	if tool.Description != "" {
		lines = append(lines, tool.Description)
	}
	lines = append(lines, "", "**Parameters:**", "```", formatSchema(tool.InputSchema), "```")
	return strings.Join(lines, "\n")
}

func formatSchema(raw json.RawMessage) string {
	var schema struct {
		Properties map[string]map[string]any `json:"properties"`
		Required   []string                  `json:"required"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &schema) != nil || len(schema.Properties) == 0 {
		return "(no parameters)"
	}
	required := map[string]bool{}
	for _, name := range schema.Required {
		required[name] = true
	}
	names := make([]string, 0, len(schema.Properties))
	for name := range schema.Properties {
		names = append(names, name)
	}
	sort.Strings(names)
	lines := make([]string, 0, len(names))
	for _, name := range names {
		property := schema.Properties[name]
		propertyType, _ := property["type"].(string)
		if propertyType == "" {
			propertyType = "any"
		}
		requirement := "optional"
		if required[name] {
			requirement = "required"
		}
		description, _ := property["description"].(string)
		line := fmt.Sprintf("- %s (%s, %s)", name, propertyType, requirement)
		if description != "" {
			line += ": " + description
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func callTool(ctx context.Context, session *Session, name string, args map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
	raw := RawToolName(name)
	tools, err := session.Tools(ctx)
	if err != nil {
		return agent.ToolResult{}, describeSessionError(err)
	}
	known := false
	for _, tool := range tools {
		if tool.Name == raw {
			known = true
			break
		}
	}
	if !known {
		return textResult(fmt.Sprintf("Tool %q not found. Call {\"server\": %q} to list the desktop tools.", name, ServerName)), nil
	}
	if onUpdate != nil {
		onUpdate(textResult("Calling " + PrefixedToolName(raw) + "..."))
	}
	result, err := session.Call(ctx, raw, args)
	if err != nil {
		return agent.ToolResult{}, describeSessionError(err)
	}

	var blocks []llm.ContentBlock
	var textParts []string
	for _, item := range result.Content {
		switch item.Type {
		case "image":
			blocks = append(blocks, llm.ImageContent{Type: "image", Data: item.Data, MimeType: item.MimeType})
		default:
			if item.Text != "" {
				textParts = append(textParts, item.Text)
			}
		}
	}
	text := strings.Join(textParts, "\n\n")
	if len(text) > maxTextOutput {
		clipped := text[:maxTextOutput]
		for len(clipped) > 0 && clipped[len(clipped)-1]&0xC0 == 0x80 {
			clipped = clipped[:len(clipped)-1]
		}
		text = clipped + fmt.Sprintf("\n\n[Output truncated: showing the first %d of %d bytes]", len(clipped), len(text))
	}
	if result.IsError {
		if text == "" {
			text = raw + " failed"
		}
		return agent.ToolResult{}, errors.New(text)
	}
	if text != "" {
		blocks = append([]llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}, blocks...)
	}
	if len(blocks) == 0 {
		blocks = []llm.ContentBlock{llm.TextContent{Type: "text", Text: "(no output)"}}
	}
	return agent.ToolResult{Content: blocks, Details: map[string]any{"tool": raw}}, nil
}

func describeSessionError(err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%s did not respond in time; run 'computer-use-linux doctor' to check desktop readiness: %w", ServerName, err)
	}
	return err
}

func textResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}
