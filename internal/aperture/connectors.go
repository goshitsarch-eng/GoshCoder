package aperture

// Connector proxy meta-tools (extensions/connectors/index.ts and
// proxy-tools.ts): instead of registering every gateway tool (high context
// cost), four meta-tools let the model list connectors, search and describe
// their tools, and execute one; a pinned allow-list registers selected tools
// first-class. The original's resource proxy tools were removed upstream and
// are not adapted; the raw resource methods remain on McpSession.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

// maxConnectorOutput bounds what one connector call feeds back into context;
// the full output overflows to a temp file (proxy-tools.ts truncateHead
// behavior with GoshCoder's 50 KiB tool-output convention).
const maxConnectorOutput = 50 << 10

// ContextCostWarningThreshold is the pinned-tool count above which the
// settings surface warns about system-prompt cost.
const ContextCostWarningThreshold = 10

// ConnectorIDFromToolName derives the connector id for a gateway tool name:
// the segment before the first "_", or "other".
func ConnectorIDFromToolName(name string) string {
	if index := strings.Index(name, "_"); index > 0 {
		return name[:index]
	}
	return "other"
}

func truncateDescription(description string, max int) string {
	flat := strings.Join(strings.Fields(description), " ")
	if len(flat) <= max {
		return flat
	}
	if max <= 3 {
		return flat[:max]
	}
	return flat[:max-3] + "..."
}

// formatProperty renders one JSON Schema property as an indented bullet,
// recursing into object properties (proxy-tools.ts formatProperty).
func formatProperty(key string, schema map[string]any, required bool, indent int) string {
	schemaType, _ := schema["type"].(string)
	if schemaType == "" {
		schemaType = "any"
	}
	description, _ := schema["description"].(string)
	requirement := "optional"
	if required {
		requirement = "required"
	}

	typeText := schemaType
	if enum, ok := schema["enum"].([]any); ok && len(enum) > 0 {
		parts := make([]string, 0, len(enum))
		for _, value := range enum {
			encoded, _ := json.Marshal(value)
			parts = append(parts, string(encoded))
		}
		typeText = "enum(" + strings.Join(parts, "|") + ")"
	}
	if schemaType == "array" {
		if items, ok := schema["items"].(map[string]any); ok {
			itemType, _ := items["type"].(string)
			if itemType == "" {
				itemType = "any"
			}
			typeText = "array<" + itemType + ">"
		}
	}

	prefix := strings.Repeat("  ", indent)
	line := fmt.Sprintf("%s- `%s` (%s, %s)", prefix, key, typeText, requirement)
	if description != "" {
		line += ": " + description
	}

	if properties, ok := schema["properties"].(map[string]any); ok && schemaType == "object" {
		requiredSet := requiredSet(schema)
		lines := []string{line}
		for _, nestedKey := range sortedKeys(properties) {
			if nested, ok := properties[nestedKey].(map[string]any); ok {
				lines = append(lines, formatProperty(nestedKey, nested, requiredSet[nestedKey], indent+1))
			}
		}
		return strings.Join(lines, "\n")
	}
	return line
}

func requiredSet(schema map[string]any) map[string]bool {
	out := map[string]bool{}
	if required, ok := schema["required"].([]any); ok {
		for _, value := range required {
			if name, ok := value.(string); ok {
				out[name] = true
			}
		}
	}
	return out
}

func sortedKeys(values map[string]any) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// FormatJSONSchema renders a tool input schema as readable parameter bullets.
func FormatJSONSchema(raw json.RawMessage) string {
	var schema map[string]any
	if len(raw) == 0 || json.Unmarshal(raw, &schema) != nil {
		return "(no parameters)"
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok || len(properties) == 0 {
		return "(no parameters)"
	}
	required := requiredSet(schema)
	lines := make([]string, 0, len(properties))
	for _, key := range sortedKeys(properties) {
		if property, ok := properties[key].(map[string]any); ok {
			lines = append(lines, formatProperty(key, property, required[key], 0))
		}
	}
	return strings.Join(lines, "\n")
}

// SessionSource supplies the live MCP session; nil when the connectors
// feature is disabled or the gateway was unreachable at startup.
type SessionSource func() *McpSession

// executeConnectorCall runs one gateway tool, joins its text content, and
// truncates the head with a temp-file overflow so a huge connector response
// does not flood the context (proxy-tools.ts executeConnectorCall).
func executeConnectorCall(ctx context.Context, session *McpSession, toolName string, args map[string]any) (agent.ToolResult, error) {
	result, err := session.CallTool(ctx, toolName, args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	var textParts []string
	for _, item := range result.Content {
		if item.Text != "" {
			textParts = append(textParts, item.Text)
		}
	}
	fullText := strings.Join(textParts, "\n\n")
	outputText := fullText
	if len(fullText) > maxConnectorOutput {
		clipped := fullText[:maxConnectorOutput]
		// Do not split a UTF-8 sequence at the cut.
		for len(clipped) > 0 && clipped[len(clipped)-1]&0xC0 == 0x80 {
			clipped = clipped[:len(clipped)-1]
		}
		note := fmt.Sprintf("\n\n[Showing the first %d of %d bytes", len(clipped), len(fullText))
		if path, writeErr := writeOverflow(fullText); writeErr == nil {
			note += ". Full output: " + path
		}
		outputText = clipped + note + "]"
	}
	if outputText == "" {
		outputText = "(no text output)"
	}
	if result.IsError {
		return agent.ToolResult{}, errors.New(outputText)
	}
	return agent.ToolResult{
		Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: outputText}},
		Details: map[string]any{"toolName": toolName},
	}, nil
}

func writeOverflow(content string) (string, error) {
	var suffix [8]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		return "", err
	}
	path := filepath.Join(os.TempDir(), "goshcoder-aperture-connector-"+hex.EncodeToString(suffix[:])+".json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// noConnectorSession is returned when the session source yields nil.
var errNoConnectorSession = errors.New("connector session is not available; the connectors feature may be disabled or the Aperture host unreachable")

// ConnectorListTool lists the connectors and their tool counts.
func ConnectorListTool(connectors []ConnectorInfo, tools []McpTool) agent.Tool {
	counts := map[string]int{}
	for _, tool := range tools {
		counts[ConnectorIDFromToolName(tool.Name)]++
	}
	return agent.Tool{
		Name:  "aperture_connector_list",
		Label: "Connector List",
		Description: "List all available Aperture connectors and their metadata. " +
			"Use this to discover which connectors are configured and how many tools each exposes.",
		Parameters: json.RawMessage(`{"type":"object","properties":{}}`),
		Execute: func(context.Context, string, map[string]any, func(agent.ToolResult)) (agent.ToolResult, error) {
			if len(connectors) == 0 {
				return textToolResult("No connectors are currently configured."), nil
			}
			lines := make([]string, 0, len(connectors))
			for _, connector := range connectors {
				display := connector.Provider
				if display == "" {
					display = connector.ID
				}
				description := truncateDescription(connector.Description, 80)
				if description == "" {
					description = "(no description)"
				}
				status := connector.Status
				if status == "" {
					status = "unknown"
				}
				count := counts[connector.ID]
				plural := "s"
				if count == 1 {
					plural = ""
				}
				lines = append(lines, fmt.Sprintf("- **%s** (`%s`): %s — %d tool%s — %s", display, connector.ID, description, count, plural, status))
			}
			return textToolResult(fmt.Sprintf("%d connector(s) available:\n\n%s", len(connectors), strings.Join(lines, "\n"))), nil
		},
	}
}

// ConnectorToolSearchTool searches gateway tools by name or description,
// grouping matches by verified connector id (unknown prefixes go to
// "other").
func ConnectorToolSearchTool(tools []McpTool, connectorIDs []string) agent.Tool {
	knownIDs := map[string]bool{}
	for _, id := range connectorIDs {
		knownIDs[strings.ToLower(id)] = true
	}
	return agent.Tool{
		Name:  "aperture_connector_tool_search",
		Label: "Connector Tool Search",
		Description: "Search for available tools from Aperture connectors by name or description. " +
			"Use this when you need to find a tool to accomplish a task but don't know its exact name. " +
			"Pass an empty query to list all tools.",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"query":{"type":"string","description":"Search query to match tool names or descriptions. Use * or leave empty to list all."},
			"limit":{"type":"number","description":"Maximum results to return (default 15)"},
			"connector":{"type":"string","description":"Filter to a specific connector, e.g. 'github' or 'aperture'"}
		}}`),
		Execute: func(_ context.Context, _ string, params map[string]any, _ func(agent.ToolResult)) (agent.ToolResult, error) {
			query := strings.ToLower(strings.TrimSpace(stringParam(params, "query")))
			limit := intParam(params, "limit", 15)
			connector := strings.TrimSpace(stringParam(params, "connector"))

			matches := tools
			if connector != "" {
				prefix := strings.ToLower(connector) + "_"
				var filtered []McpTool
				for _, tool := range matches {
					if strings.HasPrefix(strings.ToLower(tool.Name), prefix) {
						filtered = append(filtered, tool)
					}
				}
				matches = filtered
			}
			if query != "" && query != "*" {
				var filtered []McpTool
				for _, tool := range matches {
					if strings.Contains(strings.ToLower(tool.Name), query) ||
						strings.Contains(strings.ToLower(tool.Description), query) {
						filtered = append(filtered, tool)
					}
				}
				matches = filtered
			}
			if limit > 0 && len(matches) > limit {
				matches = matches[:limit]
			}
			if len(matches) == 0 {
				message := "No tools found"
				if query != "" {
					message += fmt.Sprintf(" matching %q", query)
				}
				if connector != "" {
					message += fmt.Sprintf(" from connector %q", connector)
				}
				return textToolResult(message + ". Use aperture_connector_list to see available connectors."), nil
			}

			if connector != "" {
				lines := make([]string, 0, len(matches))
				for _, tool := range matches {
					lines = append(lines, toolBullet(tool))
				}
				return textToolResult(strings.Join(lines, "\n")), nil
			}

			groups := map[string][]McpTool{}
			for _, tool := range matches {
				prefix := tool.Name
				if index := strings.Index(tool.Name, "_"); index > 0 {
					prefix = tool.Name[:index]
				}
				key := "other"
				if knownIDs[strings.ToLower(prefix)] {
					key = prefix
				}
				groups[key] = append(groups[key], tool)
			}
			names := make([]string, 0, len(groups))
			for name := range groups {
				names = append(names, name)
			}
			sort.Slice(names, func(i, j int) bool {
				if names[i] == "other" {
					return false
				}
				if names[j] == "other" {
					return true
				}
				return names[i] < names[j]
			})
			var lines []string
			for _, name := range names {
				lines = append(lines, fmt.Sprintf("### %s (%d)", name, len(groups[name])))
				for _, tool := range groups[name] {
					lines = append(lines, toolBullet(tool))
				}
				lines = append(lines, "")
			}
			return textToolResult(strings.TrimRight(strings.Join(lines, "\n"), "\n")), nil
		},
	}
}

func toolBullet(tool McpTool) string {
	description := truncateDescription(tool.Description, 100)
	if description == "" {
		description = "(no description)"
	}
	return fmt.Sprintf("- `%s`: %s", tool.Name, description)
}

// ConnectorToolDescribeTool returns the full description and parameter
// schema for one tool.
func ConnectorToolDescribeTool(tools []McpTool) agent.Tool {
	return agent.Tool{
		Name:  "aperture_connector_tool_describe",
		Label: "Connector Tool Describe",
		Description: "Get the full description and parameter schema for a specific connector tool. " +
			"Call this before aperture_connector_tool_call to understand what arguments the tool expects.",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"tool":{"type":"string","description":"Name of the connector tool to describe"}
		},"required":["tool"]}`),
		Execute: func(_ context.Context, _ string, params map[string]any, _ func(agent.ToolResult)) (agent.ToolResult, error) {
			toolName := stringParam(params, "tool")
			for _, tool := range tools {
				if tool.Name != toolName {
					continue
				}
				description := tool.Description
				if description == "" {
					description = "(no description)"
				}
				text := strings.Join([]string{
					"### " + tool.Name,
					description,
					"",
					"**Parameters:**",
					"```",
					FormatJSONSchema(tool.InputSchema),
					"```",
				}, "\n")
				return textToolResult(text), nil
			}
			return textToolResult(fmt.Sprintf("Tool %q not found. Use aperture_connector_tool_search to find available tools.", toolName)), nil
		},
	}
}

// ConnectorToolCallTool executes a connector tool by name with a JSON
// argument object string.
func ConnectorToolCallTool(tools []McpTool, getSession SessionSource) agent.Tool {
	toolNames := map[string]bool{}
	for _, tool := range tools {
		toolNames[tool.Name] = true
	}
	return agent.Tool{
		Name:  "aperture_connector_tool_call",
		Label: "Connector Tool Call",
		Description: "Execute a connector tool by name with JSON arguments. " +
			"Call aperture_connector_tool_describe first to see the required parameters. " +
			"The args field must be a valid JSON object string matching the tool's schema.",
		Parameters: json.RawMessage(`{"type":"object","properties":{
			"tool":{"type":"string","description":"Name of the connector tool to execute"},
			"args":{"type":"string","description":"Arguments as a JSON object string. Call aperture_connector_tool_describe first to see the expected schema. Omit if the tool takes no arguments."}
		},"required":["tool"]}`),
		Execute: func(ctx context.Context, _ string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			toolName := stringParam(params, "tool")
			if !toolNames[toolName] {
				return textToolResult(fmt.Sprintf("Tool %q not found. Use aperture_connector_tool_search to find available tools.", toolName)), nil
			}
			argsJSON := stringParam(params, "args")
			if argsJSON == "" {
				argsJSON = "{}"
			}
			var args map[string]any
			if err := json.Unmarshal([]byte(argsJSON), &args); err != nil {
				return textToolResult(fmt.Sprintf("Invalid args JSON: %v. Use aperture_connector_tool_describe(%q) to see the expected schema.", err, toolName)), nil
			}
			if onUpdate != nil {
				onUpdate(textToolResult("Calling " + toolName + "..."))
			}
			session := getSession()
			if session == nil {
				return agent.ToolResult{}, errNoConnectorSession
			}
			return executeConnectorCall(ctx, session, toolName, args)
		},
	}
}

// StandaloneConnectorTool registers one pinned gateway tool first-class,
// passing its raw schema through so the model sees the real parameter shape.
// Execution reuses the same truncation and overflow path as the call
// meta-tool.
func StandaloneConnectorTool(tool McpTool, getSession SessionSource) agent.Tool {
	description := strings.TrimSpace(tool.Description)
	if description == "" {
		description = "Aperture connector tool: " + tool.Name
	}
	parameters := coerceInputSchema(tool.InputSchema)
	return agent.Tool{
		Name:        tool.Name,
		Label:       tool.Name,
		Description: description,
		Parameters:  parameters,
		Execute: func(ctx context.Context, _ string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			if onUpdate != nil {
				onUpdate(textToolResult("Calling " + tool.Name + "..."))
			}
			session := getSession()
			if session == nil {
				return agent.ToolResult{}, errNoConnectorSession
			}
			return executeConnectorCall(ctx, session, tool.Name, params)
		},
	}
}

// coerceInputSchema keeps the gateway's schema when it is a recognizable
// object schema, and substitutes an empty object schema otherwise so the
// tool still registers.
func coerceInputSchema(raw json.RawMessage) json.RawMessage {
	var schema map[string]any
	if len(raw) > 0 && json.Unmarshal(raw, &schema) == nil {
		if schemaType, _ := schema["type"].(string); schemaType == "object" {
			return raw
		}
	}
	return json.RawMessage(`{"type":"object","properties":{}}`)
}

// ConnectorToolSet is the registered connector tool surface for one session.
type ConnectorToolSet struct {
	Tools []agent.Tool
	// MissingPins are pinned tool names the gateway no longer exposes; they
	// are skipped silently at registration and reported once as a warning.
	MissingPins []string
}

// BuildConnectorTools splits the gateway tool list into pinned first-class
// tools and proxied tools reachable through the discovery meta-tools
// (extensions/connectors/index.ts session_start). Connectors that expose no
// tools are hidden from the list tool.
func BuildConnectorTools(resolved Resolved, connectors []ConnectorInfo, tools []McpTool, getSession SessionSource) ConnectorToolSet {
	// De-dupe by name, preserving gateway order.
	seen := map[string]bool{}
	var uniqueTools []McpTool
	for _, tool := range tools {
		if seen[tool.Name] {
			continue
		}
		seen[tool.Name] = true
		uniqueTools = append(uniqueTools, tool)
	}

	toolCounts := map[string]int{}
	for _, tool := range uniqueTools {
		toolCounts[ConnectorIDFromToolName(tool.Name)]++
	}
	var visibleConnectors []ConnectorInfo
	var connectorIDs []string
	for _, connector := range connectors {
		if toolCounts[connector.ID] > 0 {
			visibleConnectors = append(visibleConnectors, connector)
			connectorIDs = append(connectorIDs, connector.ID)
		}
	}

	pinnedNames := map[string]bool{}
	for _, pin := range resolved.PinnedTools {
		pinnedNames[pin.ToolName] = true
	}
	var pinned, proxied []McpTool
	if len(pinnedNames) > 0 {
		for _, tool := range uniqueTools {
			if pinnedNames[tool.Name] {
				pinned = append(pinned, tool)
			} else {
				proxied = append(proxied, tool)
			}
		}
	} else {
		proxied = uniqueTools
	}

	set := ConnectorToolSet{}
	for name := range pinnedNames {
		found := false
		for _, tool := range pinned {
			if tool.Name == name {
				found = true
				break
			}
		}
		if !found {
			set.MissingPins = append(set.MissingPins, name)
		}
	}
	sort.Strings(set.MissingPins)

	for _, tool := range pinned {
		set.Tools = append(set.Tools, StandaloneConnectorTool(tool, getSession))
	}
	if resolved.DiscoveryTools {
		set.Tools = append(set.Tools,
			ConnectorListTool(visibleConnectors, proxied),
			ConnectorToolSearchTool(proxied, connectorIDs),
			ConnectorToolDescribeTool(proxied),
			ConnectorToolCallTool(proxied, getSession),
		)
	}
	return set
}

func textToolResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}

func stringParam(params map[string]any, key string) string {
	value, _ := params[key].(string)
	return value
}

func intParam(params map[string]any, key string, fallback int) int {
	if value, ok := params[key].(float64); ok && value > 0 {
		return int(value)
	}
	return fallback
}
