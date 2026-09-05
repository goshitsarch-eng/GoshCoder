//! Aperture connector tools for the agent: the four discovery meta-tools and
//! the pinned first-class tools.
//!
//! Ported from `@aliou/pi-ts-aperture`'s `extensions/connectors/index.ts` and
//! `proxy-tools.ts` (via `internal/aperture/connectors.go`). Instead of
//! registering every gateway tool, which costs context, four meta-tools let the
//! model list connectors, search and describe their tools, and execute one; a
//! pinned allow-list registers selected tools first-class. The original's
//! resource proxy tools were removed upstream and are not adapted.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use crate::{
    agent,
    aperture::{ConnectorInfo, GatewayTool, Resolved, connector_id_from_tool_name},
    aperture_mcp::{McpSession, call_result_to_agent_result, gateway_tool_to_agent_tool},
};

/// Provider tool names are limited to this charset and length (Anthropic and
/// OpenAI both reject anything else), so a gateway tool that would break
/// every later request is left out instead of registered.
pub const MAX_TOOL_NAME_LENGTH: usize = 64;

const DEFAULT_SEARCH_LIMIT: usize = 15;
const MAX_SEARCH_LIMIT: usize = 200;
const MAX_ARGS_JSON_BYTES: usize = 1 << 20;
/// A tool schema larger than this is not worth the context it would cost.
const MAX_SCHEMA_BYTES: usize = 64 << 10;

/// The registered connector tool surface for one session.
#[derive(Default)]
pub struct ConnectorAgentTools {
    pub tools: Vec<agent::Tool>,
    /// Pinned tool names the gateway no longer exposes; skipped silently at
    /// registration and reported once as a warning.
    pub missing_pins: Vec<String>,
    /// Gateway tools whose names a provider would reject, or which collide
    /// with a tool the session already has; reported once as a warning.
    pub rejected: Vec<String>,
}

/// Splits the gateway tool list into pinned first-class tools and proxied
/// tools reachable through the discovery meta-tools. Connectors that expose
/// no tools are hidden from the list tool. `reserved` names the tools the
/// session already registers; a gateway tool with the same name is dropped
/// rather than shadowing a built-in.
pub fn build_connector_agent_tools(
    resolved: &Resolved,
    connectors: &[ConnectorInfo],
    tools: &[GatewayTool],
    reserved: &BTreeSet<String>,
    session: McpSession,
) -> ConnectorAgentTools {
    let mut seen = BTreeSet::new();
    let mut rejected = Vec::new();
    let mut unique_tools = Vec::new();
    for tool in tools {
        if !seen.insert(tool.name.clone()) {
            continue;
        }
        if !is_provider_safe_tool_name(&tool.name)
            || reserved.contains(&tool.name)
            || schema_too_large(&tool.input_schema)
        {
            rejected.push(tool.name.clone());
            continue;
        }
        unique_tools.push(tool.clone());
    }

    let mut tool_counts = BTreeMap::<String, usize>::new();
    for tool in &unique_tools {
        *tool_counts
            .entry(connector_id_from_tool_name(&tool.name))
            .or_default() += 1;
    }
    let visible_connectors = connectors
        .iter()
        .filter(|connector| tool_counts.get(&connector.id).copied().unwrap_or(0) > 0)
        .cloned()
        .collect::<Vec<_>>();
    let connector_ids = visible_connectors
        .iter()
        .map(|connector| connector.id.clone())
        .collect::<Vec<_>>();

    let pinned_names = resolved
        .pinned_tools
        .iter()
        .map(|pin| pin.tool_name.clone())
        .collect::<BTreeSet<_>>();
    let (pinned, proxied): (Vec<_>, Vec<_>) = unique_tools
        .into_iter()
        .partition(|tool| pinned_names.contains(&tool.name));
    let found = pinned
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();
    let missing_pins = pinned_names
        .iter()
        .filter(|name| !found.contains(*name))
        .cloned()
        .collect::<Vec<_>>();

    let mut set = ConnectorAgentTools {
        missing_pins,
        rejected,
        ..ConnectorAgentTools::default()
    };
    for tool in pinned {
        match gateway_tool_to_agent_tool(session.clone(), tool.clone()) {
            Ok(agent_tool) => set.tools.push(agent_tool),
            Err(_) => set.rejected.push(tool.name),
        }
    }
    if resolved.discovery_tools {
        set.tools
            .push(connector_list_tool(visible_connectors, &proxied));
        set.tools
            .push(connector_tool_search_tool(proxied.clone(), connector_ids));
        set.tools
            .push(connector_tool_describe_tool(proxied.clone()));
        set.tools.push(connector_tool_call_tool(proxied, session));
    }
    set
}

/// Whether a name satisfies every provider's tool-name grammar.
pub fn is_provider_safe_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_TOOL_NAME_LENGTH
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn schema_too_large(schema: &Value) -> bool {
    serde_json::to_vec(schema).map_or(true, |encoded| encoded.len() > MAX_SCHEMA_BYTES)
}

fn connector_list_tool(connectors: Vec<ConnectorInfo>, tools: &[GatewayTool]) -> agent::Tool {
    let mut counts = BTreeMap::<String, usize>::new();
    for tool in tools {
        *counts
            .entry(connector_id_from_tool_name(&tool.name))
            .or_default() += 1;
    }
    agent::Tool::new(
        "aperture_connector_list",
        "Connector List",
        "List all available Aperture connectors and their metadata. Use this to discover \
which connectors are configured and how many tools each exposes.",
        json!({"type": "object", "properties": {}}),
        move |_cancellation, _tool_call_id, _arguments, _on_update| {
            if connectors.is_empty() {
                return Ok(agent::ToolResult::text(
                    "No connectors are currently configured.",
                ));
            }
            let lines = connectors
                .iter()
                .map(|connector| {
                    let display = if connector.provider.is_empty() {
                        connector.id.as_str()
                    } else {
                        connector.provider.as_str()
                    };
                    let description = or_placeholder(
                        &truncate_description(&connector.description, 80),
                        "(no description)",
                    );
                    let status = or_placeholder(&connector.status, "unknown");
                    let count = counts.get(&connector.id).copied().unwrap_or(0);
                    let plural = if count == 1 { "" } else { "s" };
                    format!(
                        "- **{display}** (`{}`): {description} — {count} tool{plural} — {status}",
                        connector.id
                    )
                })
                .collect::<Vec<_>>();
            Ok(agent::ToolResult::text(format!(
                "{} connector(s) available:\n\n{}",
                connectors.len(),
                lines.join("\n")
            )))
        },
    )
}

fn connector_tool_search_tool(tools: Vec<GatewayTool>, connector_ids: Vec<String>) -> agent::Tool {
    let known_ids = connector_ids
        .iter()
        .map(|id| id.to_lowercase())
        .collect::<BTreeSet<_>>();
    agent::Tool::new(
        "aperture_connector_tool_search",
        "Connector Tool Search",
        "Search for available tools from Aperture connectors by name or description. Use this \
when you need to find a tool to accomplish a task but don't know its exact name. Pass an \
empty query to list all tools.",
        json!({"type": "object", "properties": {
            "query": {"type": "string", "description": "Search query to match tool names or descriptions. Use * or leave empty to list all."},
            "limit": {"type": "number", "description": "Maximum results to return (default 15)"},
            "connector": {"type": "string", "description": "Filter to a specific connector, e.g. 'github' or 'aperture'"}
        }}),
        move |_cancellation, _tool_call_id, arguments, _on_update| {
            let query = string_argument(&arguments, "query").trim().to_lowercase();
            let limit = limit_argument(&arguments, "limit");
            let connector = string_argument(&arguments, "connector").trim().to_owned();

            let mut matches = tools.iter().collect::<Vec<_>>();
            if !connector.is_empty() {
                let prefix = format!("{}_", connector.to_lowercase());
                matches.retain(|tool| tool.name.to_lowercase().starts_with(&prefix));
            }
            if !query.is_empty() && query != "*" {
                matches.retain(|tool| {
                    tool.name.to_lowercase().contains(&query)
                        || tool.description.to_lowercase().contains(&query)
                });
            }
            matches.truncate(limit);
            if matches.is_empty() {
                let mut message = "No tools found".to_owned();
                if !query.is_empty() {
                    message.push_str(&format!(" matching {query:?}"));
                }
                if !connector.is_empty() {
                    message.push_str(&format!(" from connector {connector:?}"));
                }
                message.push_str(". Use aperture_connector_list to see available connectors.");
                return Ok(agent::ToolResult::text(message));
            }
            if !connector.is_empty() {
                let lines = matches
                    .iter()
                    .map(|tool| tool_bullet(tool))
                    .collect::<Vec<_>>();
                return Ok(agent::ToolResult::text(lines.join("\n")));
            }

            let mut groups = BTreeMap::<String, Vec<&GatewayTool>>::new();
            for tool in matches {
                let prefix = tool
                    .name
                    .split_once('_')
                    .filter(|(prefix, _)| !prefix.is_empty())
                    .map_or(tool.name.as_str(), |(prefix, _)| prefix);
                let key = if known_ids.contains(&prefix.to_lowercase()) {
                    prefix.to_owned()
                } else {
                    "other".to_owned()
                };
                groups.entry(key).or_default().push(tool);
            }
            // "other" sorts last; everything else alphabetically.
            let mut names = groups.keys().cloned().collect::<Vec<_>>();
            names.sort_by(|left, right| match (left.as_str(), right.as_str()) {
                ("other", "other") => std::cmp::Ordering::Equal,
                ("other", _) => std::cmp::Ordering::Greater,
                (_, "other") => std::cmp::Ordering::Less,
                _ => left.cmp(right),
            });
            let mut lines = Vec::new();
            for name in names {
                let group = &groups[&name];
                lines.push(format!("### {name} ({})", group.len()));
                lines.extend(group.iter().map(|tool| tool_bullet(tool)));
                lines.push(String::new());
            }
            Ok(agent::ToolResult::text(
                lines.join("\n").trim_end_matches('\n').to_owned(),
            ))
        },
    )
}

fn connector_tool_describe_tool(tools: Vec<GatewayTool>) -> agent::Tool {
    agent::Tool::new(
        "aperture_connector_tool_describe",
        "Connector Tool Describe",
        "Get the full description and parameter schema for a specific connector tool. Call \
this before aperture_connector_tool_call to understand what arguments the tool expects.",
        json!({"type": "object", "properties": {
            "tool": {"type": "string", "description": "Name of the connector tool to describe"}
        }, "required": ["tool"]}),
        move |_cancellation, _tool_call_id, arguments, _on_update| {
            let tool_name = string_argument(&arguments, "tool");
            let Some(tool) = tools.iter().find(|tool| tool.name == tool_name) else {
                return Ok(agent::ToolResult::text(format!(
                    "Tool {tool_name:?} not found. Use aperture_connector_tool_search to find available tools."
                )));
            };
            let description = or_placeholder(&tool.description, "(no description)");
            Ok(agent::ToolResult::text(
                [
                    format!("### {}", tool.name),
                    description,
                    String::new(),
                    "**Parameters:**".to_owned(),
                    "```".to_owned(),
                    format_json_schema(&tool.input_schema),
                    "```".to_owned(),
                ]
                .join("\n"),
            ))
        },
    )
}

fn connector_tool_call_tool(tools: Vec<GatewayTool>, session: McpSession) -> agent::Tool {
    let tool_names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();
    let mut tool = agent::Tool::new(
        "aperture_connector_tool_call",
        "Connector Tool Call",
        "Execute a connector tool by name with JSON arguments. Call \
aperture_connector_tool_describe first to see the required parameters. The args field must \
be a valid JSON object string matching the tool's schema.",
        json!({"type": "object", "properties": {
            "tool": {"type": "string", "description": "Name of the connector tool to execute"},
            "args": {"type": "string", "description": "Arguments as a JSON object string. Call aperture_connector_tool_describe first to see the expected schema. Omit if the tool takes no arguments."}
        }, "required": ["tool"]}),
        move |cancellation, _tool_call_id, arguments, on_update| {
            let tool_name = string_argument(&arguments, "tool");
            if !tool_names.contains(&tool_name) {
                return Ok(agent::ToolResult::text(format!(
                    "Tool {tool_name:?} not found. Use aperture_connector_tool_search to find available tools."
                )));
            }
            let args = match parse_args(arguments.get("args")) {
                Ok(args) => args,
                Err(error) => {
                    return Ok(agent::ToolResult::text(format!(
                        "Invalid args JSON: {error}. Use aperture_connector_tool_describe({tool_name:?}) to see the expected schema."
                    )));
                }
            };
            if cancellation.is_cancelled() {
                return Err("connector call was cancelled".to_owned());
            }
            on_update(agent::ToolResult::text(format!("Calling {tool_name}...")));
            let result = session
                .call_tool_with(&cancellation, &tool_name, args)
                .map_err(|error| error.to_string())?;
            call_result_to_agent_result(&tool_name, result).map_err(|error| error.to_string())
        },
    );
    // Connector calls talk to one gateway session; keeping them sequential
    // matches the extension, which awaits each call in turn.
    tool.execution_mode = Some(agent::ToolExecutionMode::Sequential);
    tool
}

/// The `args` argument is documented as a JSON object string, but a model
/// that passes a real object is accepted too rather than bounced.
fn parse_args(value: Option<&Value>) -> Result<Map<String, Value>, String> {
    match value {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(object)) => Ok(object.clone()),
        Some(Value::String(text)) => {
            let text = text.trim();
            if text.is_empty() {
                return Ok(Map::new());
            }
            if text.len() > MAX_ARGS_JSON_BYTES {
                return Err(format!("args exceed {MAX_ARGS_JSON_BYTES} bytes"));
            }
            match serde_json::from_str::<Value>(text).map_err(|error| error.to_string())? {
                Value::Object(object) => Ok(object),
                _ => Err("args must be a JSON object".to_owned()),
            }
        }
        Some(_) => Err("args must be a JSON object string".to_owned()),
    }
}

fn string_argument(arguments: &BTreeMap<String, Value>, key: &str) -> String {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn limit_argument(arguments: &BTreeMap<String, Value>, key: &str) -> usize {
    arguments
        .get(key)
        .and_then(Value::as_f64)
        .filter(|value| *value > 0.0)
        .map_or(DEFAULT_SEARCH_LIMIT, |value| {
            (value.min(MAX_SEARCH_LIMIT as f64)) as usize
        })
}

fn or_placeholder(value: &str, placeholder: &str) -> String {
    if value.trim().is_empty() {
        placeholder.to_owned()
    } else {
        value.to_owned()
    }
}

fn tool_bullet(tool: &GatewayTool) -> String {
    let description = or_placeholder(
        &truncate_description(&tool.description, 100),
        "(no description)",
    );
    format!("- `{}`: {description}", tool.name)
}

/// Collapses whitespace and clips to `max` characters with an ellipsis.
pub fn truncate_description(description: &str, max: usize) -> String {
    let flat = description.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    if max <= 3 {
        return flat.chars().take(max).collect();
    }
    let mut clipped = flat.chars().take(max - 3).collect::<String>();
    clipped.push_str("...");
    clipped
}

/// Renders a tool input schema as readable parameter bullets.
pub fn format_json_schema(schema: &Value) -> String {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return "(no parameters)".to_owned();
    };
    if properties.is_empty() {
        return "(no parameters)".to_owned();
    }
    let required = required_set(schema);
    let mut keys = properties.keys().collect::<Vec<_>>();
    keys.sort();
    keys.into_iter()
        .filter_map(|key| {
            properties
                .get(key)
                .filter(|property| property.is_object())
                .map(|property| format_property(key, property, required.contains(key.as_str()), 0))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn required_set(schema: &Value) -> BTreeSet<String> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|required| {
            required
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn format_property(key: &str, schema: &Value, required: bool, indent: usize) -> String {
    let schema_type = schema
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| !kind.is_empty())
        .unwrap_or("any");
    let description = schema
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let requirement = if required { "required" } else { "optional" };

    let mut type_text = schema_type.to_owned();
    if let Some(values) = schema
        .get("enum")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty())
    {
        let parts = values.iter().map(Value::to_string).collect::<Vec<_>>();
        type_text = format!("enum({})", parts.join("|"));
    }
    if schema_type == "array"
        && let Some(items) = schema.get("items").filter(|items| items.is_object())
    {
        let item_type = items
            .get("type")
            .and_then(Value::as_str)
            .filter(|kind| !kind.is_empty())
            .unwrap_or("any");
        type_text = format!("array<{item_type}>");
    }

    let prefix = "  ".repeat(indent);
    let mut line = format!("{prefix}- `{key}` ({type_text}, {requirement})");
    if !description.is_empty() {
        line.push_str(": ");
        line.push_str(description);
    }

    if schema_type == "object"
        && let Some(properties) = schema.get("properties").and_then(Value::as_object)
    {
        let nested_required = required_set(schema);
        let mut lines = vec![line];
        let mut keys = properties.keys().collect::<Vec<_>>();
        keys.sort();
        for nested_key in keys {
            if let Some(nested) = properties
                .get(nested_key)
                .filter(|nested| nested.is_object())
            {
                lines.push(format_property(
                    nested_key,
                    nested,
                    nested_required.contains(nested_key.as_str()),
                    indent + 1,
                ));
            }
        }
        return lines.join("\n");
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aperture::PinnedConnectorTool;
    use std::sync::Arc;

    fn tool(name: &str, description: &str) -> GatewayTool {
        GatewayTool {
            name: name.to_owned(),
            description: description.to_owned(),
            input_schema: json!({"type": "object", "properties": {
                "query": {"type": "string", "description": "Search text"},
                "count": {"type": "number"}
            }, "required": ["query"]}),
        }
    }

    fn connector(id: &str, provider: &str) -> ConnectorInfo {
        ConnectorInfo {
            id: id.to_owned(),
            provider: provider.to_owned(),
            description: "Does things".to_owned(),
            status: "connected".to_owned(),
            ..ConnectorInfo::default()
        }
    }

    fn session() -> McpSession {
        // The tools under test never reach the gateway; the session only
        // needs a well-formed endpoint.
        McpSession::from_parts_for_tests("http://127.0.0.1:9/")
    }

    fn run(tool: &agent::Tool, arguments: Value) -> String {
        let arguments = arguments
            .as_object()
            .expect("object")
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let result = (tool.execute)(
            agent::CancellationToken::default(),
            "call".to_owned(),
            arguments,
            Arc::new(|_| {}),
        )
        .expect("tool succeeds");
        result
            .content
            .iter()
            .filter_map(|block| block.plain_text())
            .collect::<Vec<_>>()
            .join("")
    }

    fn resolved(pinned: &[&str], discovery: bool) -> Resolved {
        Resolved {
            pinned_tools: pinned
                .iter()
                .map(|name| PinnedConnectorTool {
                    connector_id: connector_id_from_tool_name(name),
                    tool_name: (*name).to_owned(),
                })
                .collect(),
            discovery_tools: discovery,
            ..Resolved::default()
        }
    }

    #[test]
    fn discovery_tools_are_registered_and_pins_become_first_class() {
        let tools = [
            tool("github_issues_list", "List issues"),
            tool("github_issues_list", "duplicate is dropped"),
            tool("slack_post", "Post a message"),
            tool("bad name", "rejected by providers"),
            tool("bash", "collides with a built-in"),
        ];
        let set = build_connector_agent_tools(
            &resolved(&["slack_post", "gone_tool"], true),
            &[
                connector("github", "GitHub"),
                connector("slack", ""),
                connector("empty", ""),
            ],
            &tools,
            &["bash".to_owned()].into_iter().collect(),
            session(),
        );
        let names = set
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "slack_post",
                "aperture_connector_list",
                "aperture_connector_tool_search",
                "aperture_connector_tool_describe",
                "aperture_connector_tool_call",
            ]
        );
        assert_eq!(set.missing_pins, ["gone_tool"]);
        assert_eq!(set.rejected, ["bad name", "bash"]);

        let list = run(&set.tools[1], json!({}));
        assert!(list.starts_with("2 connector(s) available:"), "{list}");
        assert!(list.contains("**GitHub** (`github`): Does things — 1 tool — connected"));
        assert!(
            !list.contains("empty"),
            "connectors without tools are hidden"
        );

        let search = run(&set.tools[2], json!({"query": "issues"}));
        assert_eq!(
            search,
            "### github (1)\n- `github_issues_list`: List issues"
        );
        let all = run(&set.tools[2], json!({}));
        assert!(all.contains("### github (1)"));
        assert!(!all.contains("slack_post"), "pinned tools are not proxied");
        let none = run(
            &set.tools[2],
            json!({"query": "zzz", "connector": "github"}),
        );
        assert!(none.starts_with("No tools found matching \"zzz\" from connector \"github\"."));

        let describe = run(&set.tools[3], json!({"tool": "github_issues_list"}));
        assert!(describe.contains("### github_issues_list"));
        assert!(describe.contains("- `count` (number, optional)"));
        assert!(describe.contains("- `query` (string, required): Search text"));
        let missing = run(&set.tools[3], json!({"tool": "nope"}));
        assert!(missing.starts_with("Tool \"nope\" not found."));

        let unknown_call = run(&set.tools[4], json!({"tool": "nope"}));
        assert!(unknown_call.starts_with("Tool \"nope\" not found."));
        let bad_args = run(
            &set.tools[4],
            json!({"tool": "github_issues_list", "args": "[1,2]"}),
        );
        assert!(bad_args.starts_with("Invalid args JSON: args must be a JSON object."));
        assert_eq!(
            set.tools[4].execution_mode,
            Some(agent::ToolExecutionMode::Sequential)
        );
    }

    #[test]
    fn discovery_can_be_disabled_leaving_only_pins() {
        let set = build_connector_agent_tools(
            &resolved(&["slack_post"], false),
            &[],
            &[tool("slack_post", "Post")],
            &BTreeSet::new(),
            session(),
        );
        assert_eq!(set.tools.len(), 1);
        assert_eq!(set.tools[0].name, "slack_post");
    }

    #[test]
    fn schema_rendering_handles_enums_arrays_and_nesting() {
        let schema = json!({"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "filter": {"type": "object", "properties": {
                "since": {"type": "string", "description": "ISO date"}
            }, "required": ["since"]}
        }});
        assert_eq!(
            format_json_schema(&schema),
            "- `filter` (object, optional)\n  - `since` (string, required): ISO date\n- `mode` (enum(\"a\"|\"b\"), optional)\n- `tags` (array<string>, optional)"
        );
        assert_eq!(
            format_json_schema(&json!({"type": "object"})),
            "(no parameters)"
        );
        assert_eq!(truncate_description("  a   b  ", 10), "a b");
        assert_eq!(truncate_description("abcdefghij", 6), "abc...");
    }

    #[test]
    fn provider_safe_names_are_enforced() {
        assert!(is_provider_safe_tool_name("github_issues-list2"));
        assert!(!is_provider_safe_tool_name("github.issues"));
        assert!(!is_provider_safe_tool_name(""));
        assert!(!is_provider_safe_tool_name(&"x".repeat(65)));
    }
}
