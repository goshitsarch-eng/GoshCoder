//! Chat-only tool fallback for OmniRoute models that cannot emit native tool
//! calls.
//!
//! Native port of omniroute-pi-ext-integration's prompt-tools adapter (via
//! `internal/llm/omni_prompt_tools.go`). Some web-synchronized OmniRoute
//! models return prose but cannot produce OpenAI `tool_calls`. This adapter
//! renders the tool schemas into a text protocol, flattens the transcript into
//! plain user/assistant text, and parses literal `<tool_call>` blocks out of
//! the reply so the agent loop sees ordinary tool events. The streaming glue
//! lives in [`crate::providers`]; this module holds the protocol itself so it
//! can be tested without a transport.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};

use crate::llm;

pub use crate::omniroute::PROMPT_TOOLS_API as API;

const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";
const PROBLEM_PREVIEW_CHARS: usize = 200;

static TOOL_CALL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One tool invocation parsed out of a reply.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: BTreeMap<String, Value>,
}

/// The reply split into prose and tool calls, plus the blocks that could not
/// be understood. Those are reported back to the model so it can re-emit them
/// in the documented shape instead of being dropped silently.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedReply {
    pub prose: String,
    pub calls: Vec<ParsedToolCall>,
    pub problems: Vec<String>,
}

impl ParsedReply {
    /// The prose the model sees, with a re-emission note appended when a
    /// block was malformed.
    pub fn prose_with_problems(&self) -> String {
        if self.problems.is_empty() {
            return self.prose.clone();
        }
        let mut text = self.prose.clone();
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(
            "[omni-prompt-tools] Could not parse tool call(s). Re-emit each as:\n\
<tool_call>{\"name\":\"tool_name\",\"arguments\":{}}</tool_call>\n- ",
        );
        text.push_str(&self.problems.join("\n- "));
        text
    }
}

/// Builds the request the underlying chat-completions call receives: the
/// system prompt extended with the text protocol, the transcript flattened
/// to plain text, and no native tools.
pub fn inner_context(context: &llm::Context) -> llm::Context {
    let mut system_prompt = context.system_prompt.clone();
    if !context.tools.is_empty() {
        let protocol = render_tool_protocol(&context.tools);
        system_prompt = format!("{system_prompt}\n\n{protocol}").trim().to_owned();
    }
    llm::Context {
        system_prompt,
        messages: flatten_messages(&context.messages),
        tools: Vec::new(),
    }
}

/// Renders the tool-calling protocol the model must follow, with every tool's
/// JSON Schema inline.
pub fn render_tool_protocol(tools: &[llm::Tool]) -> String {
    let mut lines: Vec<String> = [
        "# Tool calling protocol",
        "",
        "You are connected to GoshCoder through a chat-only adapter. Native/internal tool calling is NOT available.",
        "The only valid way to call tools is to write literal text <tool_call> blocks in your assistant message.",
        "If you need a tool, write any reasoning first, then end your message with one or more blocks exactly like:",
        "",
        "<tool_call>",
        "{\"name\":\"<tool_name>\",\"arguments\":{}}",
        "</tool_call>",
        "",
        "Rules:",
        "- Use only this literal <tool_call> text protocol.",
        "- JSON must be valid; arguments must be an object matching the schema.",
        "- Do not wrap the JSON in Markdown fences.",
        "- Emit multiple blocks for multiple tools, then stop and wait for <tool_result> messages.",
        "- Never claim a tool ran unless a corresponding result was returned.",
        "- Never invent command output, file contents, or tool results.",
        "- If no tool is needed, answer normally with no <tool_call> block.",
        "",
        "## Available tools",
        "",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    for tool in tools {
        lines.push(format!("### {}", tool.name));
        lines.push(tool.description.clone());
        lines.push("Parameters (JSON Schema):".to_owned());
        lines.push(tool.parameters.to_string());
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Extracts every `<tool_call>…</tool_call>` block from a reply. Text outside
/// the blocks is the prose; a block that is not a JSON object with a string
/// `name` becomes a problem report instead of a call.
pub fn parse_tool_calls(text: &str) -> ParsedReply {
    let mut reply = ParsedReply::default();
    let mut prose = String::new();
    let mut rest = text;
    loop {
        let Some(open) = rest.find(OPEN_TAG) else {
            prose.push_str(rest);
            break;
        };
        let after_open = &rest[open + OPEN_TAG.len()..];
        let Some(close) = after_open.find(CLOSE_TAG) else {
            // An unterminated block stays in the prose, as the original's
            // non-greedy pattern leaves it unmatched.
            prose.push_str(rest);
            break;
        };
        prose.push_str(&rest[..open]);
        let body = strip_fences(after_open[..close].trim());
        match parse_call(body) {
            Ok(call) => reply.calls.push(call),
            Err(problem) => reply.problems.push(problem),
        }
        rest = &after_open[close + CLOSE_TAG.len()..];
    }
    reply.prose = prose.trim().to_owned();
    reply
}

fn parse_call(body: &str) -> Result<ParsedToolCall, String> {
    let raw: Value = serde_json::from_str(body)
        .map_err(|error| format!("invalid JSON in <tool_call> ({error}): {}", preview(body)))?;
    let name = raw
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if name.is_empty() {
        return Err(format!(
            "tool_call missing a string \"name\": {}",
            preview(body)
        ));
    }
    let arguments = match raw.get("arguments") {
        Some(Value::Object(object)) => object
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
        Some(Value::String(encoded)) => serde_json::from_str::<Value>(encoded)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .map(|object| object.into_iter().collect())
            .unwrap_or_default(),
        _ => BTreeMap::new(),
    };
    Ok(ParsedToolCall { name, arguments })
}

/// Removes a Markdown code fence a model wrapped the JSON in despite the
/// rules: an opening ``` or ```json line and a closing ``` line.
fn strip_fences(body: &str) -> &str {
    let mut text = body;
    if let Some(after) = text.strip_prefix("```") {
        let after = after
            .get(..4)
            .filter(|prefix| prefix.eq_ignore_ascii_case("json"))
            .map_or(after, |_| &after[4..]);
        text = after.trim_start_matches([' ', '\t']);
        text = text.strip_prefix('\n').unwrap_or(text);
    }
    if let Some(before) = text.trim_end().strip_suffix("```") {
        text = before.trim_end();
    }
    text.trim()
}

fn preview(body: &str) -> String {
    let mut chars = body.chars();
    let clipped: String = chars.by_ref().take(PROBLEM_PREVIEW_CHARS).collect();
    if chars.next().is_some() {
        format!("{clipped}...")
    } else {
        clipped
    }
}

/// Joins the text blocks of a message, ignoring thinking, images, and tool
/// calls.
pub fn content_text(blocks: &[llm::ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(llm::ContentBlock::plain_text)
        .collect::<Vec<_>>()
        .concat()
}

/// A tool-call id unique within the process, in the shape the original used.
pub fn next_call_id() -> String {
    format!(
        "call_omni_{:x}_{}",
        now_millis(),
        TOOL_CALL_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1
    )
}

/// Rewrites the transcript as alternating plain-text turns: assistant tool
/// calls become `<tool_call>` blocks and tool results become `<tool_result>`
/// user turns, with consecutive same-role turns merged.
fn flatten_messages(messages: &[llm::Message]) -> Vec<llm::Message> {
    let mut output: Vec<llm::Message> = Vec::with_capacity(messages.len());
    let mut push = |role: &str, text: String| {
        let text = text.trim().to_owned();
        if text.is_empty() {
            return;
        }
        match (output.last_mut(), role) {
            (Some(llm::Message::User(previous)), "user") => {
                let merged = format!("{}\n\n{text}", user_text(previous));
                previous.content = llm::UserContent::Text(merged);
            }
            (Some(llm::Message::Assistant(previous)), "assistant") => {
                let merged = format!("{}\n\n{text}", content_text(&previous.content));
                previous.content = vec![llm::ContentBlock::text(merged)];
            }
            (_, "user") => output.push(llm::Message::User(llm::UserMessage::text(
                text,
                now_millis(),
            ))),
            _ => output.push(llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![llm::ContentBlock::text(text)],
                stop_reason: "stop".to_owned(),
                timestamp: now_millis(),
                ..llm::AssistantMessage::default()
            }))),
        }
    };
    for message in messages {
        match message {
            llm::Message::User(user) => push("user", user_text(user)),
            llm::Message::Assistant(assistant) => {
                let mut parts = vec![content_text(&assistant.content)];
                for block in &assistant.content {
                    if let llm::ContentBlock::ToolCall(call) = block {
                        let encoded = json!({"name": call.name, "arguments": call.arguments});
                        parts.push(format!("{OPEN_TAG}\n{encoded}\n{CLOSE_TAG}"));
                    }
                }
                let joined = parts
                    .into_iter()
                    .filter(|part| !part.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                push("assistant", joined);
            }
            llm::Message::ToolResult(result) => {
                let tag = if result.is_error {
                    "tool_result error"
                } else {
                    "tool_result"
                };
                push(
                    "user",
                    format!(
                        "<{tag} tool={:?} id={:?}>\n{}\n</tool_result>",
                        result.tool_name,
                        result.tool_call_id,
                        content_text(&result.content)
                    ),
                );
            }
        }
    }
    output
}

fn user_text(message: &llm::UserMessage) -> String {
    match &message.content {
        llm::UserContent::Text(text) => text.clone(),
        llm::UserContent::Blocks(blocks) => content_text(blocks),
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(name: &str, arguments: Value) -> llm::ContentBlock {
        llm::ContentBlock::ToolCall(llm::ToolCall {
            id: "call_1".to_owned(),
            name: name.to_owned(),
            arguments: arguments
                .as_object()
                .expect("object")
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
            thought_signature: String::new(),
            namespace: String::new(),
        })
    }

    #[test]
    fn parses_blocks_and_keeps_the_prose() {
        let reply = parse_tool_calls(
            "Let me look.\n<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"a.rs\"}}\n</tool_call>\n\
<tool_call>```json\n{\"name\":\"ls\",\"arguments\":\"{\\\"path\\\":\\\".\\\"}\"}\n```</tool_call>\nDone.",
        );
        assert_eq!(reply.prose, "Let me look.\n\n\nDone.");
        assert_eq!(reply.calls.len(), 2);
        assert_eq!(reply.calls[0].name, "read");
        assert_eq!(reply.calls[0].arguments["path"], json!("a.rs"));
        assert_eq!(reply.calls[1].name, "ls");
        assert_eq!(
            reply.calls[1].arguments["path"],
            json!("."),
            "a JSON-encoded string of arguments is decoded"
        );
        assert!(reply.problems.is_empty());
    }

    #[test]
    fn malformed_blocks_are_reported_not_dropped() {
        let reply = parse_tool_calls(
            "<tool_call>{not json}</tool_call><tool_call>{\"arguments\":{}}</tool_call><tool_call>unterminated",
        );
        assert!(reply.calls.is_empty());
        assert_eq!(reply.problems.len(), 2);
        assert!(reply.problems[0].starts_with("invalid JSON in <tool_call> ("));
        assert!(reply.problems[1].starts_with("tool_call missing a string \"name\": "));
        assert_eq!(reply.prose, "<tool_call>unterminated");
        let prose = reply.prose_with_problems();
        assert!(prose.contains("[omni-prompt-tools] Could not parse tool call(s)."));
        assert!(prose.contains("\n- invalid JSON"));
    }

    #[test]
    fn preview_is_clipped_on_character_boundaries() {
        let long = "é".repeat(PROBLEM_PREVIEW_CHARS + 5);
        let previewed = preview(&long);
        assert!(previewed.ends_with("..."));
        assert_eq!(previewed.chars().count(), PROBLEM_PREVIEW_CHARS + 3);
    }

    #[test]
    fn flattening_merges_roles_and_renders_calls_and_results() {
        let messages = vec![
            llm::Message::User(llm::UserMessage::text("first", 1)),
            llm::Message::User(llm::UserMessage::text("second", 2)),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![
                    llm::ContentBlock::text("Looking"),
                    tool_call("read", json!({"path": "a.rs"})),
                ],
                ..llm::AssistantMessage::default()
            })),
            llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: "call_1".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![llm::ContentBlock::text("fn main() {}")],
                is_error: true,
                ..llm::ToolResultMessage::default()
            })),
        ];
        let flattened = flatten_messages(&messages);
        assert_eq!(flattened.len(), 3);
        let llm::Message::User(first) = &flattened[0] else {
            panic!("user turn");
        };
        assert_eq!(
            first.content,
            llm::UserContent::Text("first\n\nsecond".to_owned())
        );
        let llm::Message::Assistant(assistant) = &flattened[1] else {
            panic!("assistant turn");
        };
        assert_eq!(
            content_text(&assistant.content),
            "Looking\n\n<tool_call>\n{\"arguments\":{\"path\":\"a.rs\"},\"name\":\"read\"}\n</tool_call>"
        );
        let llm::Message::User(result) = &flattened[2] else {
            panic!("tool result becomes a user turn");
        };
        assert_eq!(
            result.content,
            llm::UserContent::Text(
                "<tool_result error tool=\"read\" id=\"call_1\">\nfn main() {}\n</tool_result>"
                    .to_owned()
            )
        );
    }

    #[test]
    fn inner_context_appends_the_protocol_and_drops_native_tools() {
        let context = llm::Context {
            system_prompt: "Be brief.".to_owned(),
            messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
            tools: vec![llm::Tool {
                name: "read".to_owned(),
                description: "Read a file".to_owned(),
                parameters: json!({"type": "object"}),
                constrained_sampling: None,
            }],
        };
        let inner = inner_context(&context);
        assert!(inner.tools.is_empty());
        assert!(
            inner
                .system_prompt
                .starts_with("Be brief.\n\n# Tool calling protocol")
        );
        assert!(
            inner.system_prompt.contains(
                "### read\nRead a file\nParameters (JSON Schema):\n{\"type\":\"object\"}"
            )
        );
        assert_eq!(inner.messages.len(), 1);

        let no_tools = inner_context(&llm::Context {
            tools: Vec::new(),
            ..context
        });
        assert_eq!(no_tools.system_prompt, "Be brief.");
    }

    #[test]
    fn call_ids_are_unique() {
        let first = next_call_id();
        let second = next_call_id();
        assert!(first.starts_with("call_omni_"));
        assert_ne!(first, second);
    }
}
