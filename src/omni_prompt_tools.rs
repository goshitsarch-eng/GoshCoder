//! Text-protocol tool calling for OmniRoute models without native tools.
//!
//! Some OmniRoute models can produce a normal OpenAI-compatible completion but
//! cannot emit structured `tool_calls`. This module rewrites the transcript
//! into a small XML-like text protocol, then converts completed `<tool_call>`
//! blocks back into native assistant tool-call content.

use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::Deserialize;
use serde_json::Value;

use crate::{
    llm, omniroute,
    providers::{MessageEmitter, Result},
    stream,
};

/// The model API used for chat-only OmniRoute models.
pub const API_OMNI_PROMPT_TOOLS: &str = omniroute::PROMPT_TOOLS_API;

static TOOL_CALL_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const OPEN_TOOL_CALL: &str = "<tool_call>";
const CLOSE_TOOL_CALL: &str = "</tool_call>";

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ParsedToolCall {
    pub name: String,
    pub arguments: BTreeMap<String, Value>,
}

/// Builds the native-tool-free inner request sent over OpenAI Completions.
pub(crate) fn inner_context(context: &llm::Context) -> llm::Context {
    let mut system_prompt = context.system_prompt.clone();
    if !context.tools.is_empty() {
        system_prompt = format!(
            "{system_prompt}\n\n{}",
            render_tool_protocol(&context.tools)
        )
        .trim()
        .to_owned();
    }
    llm::Context {
        system_prompt,
        messages: flatten_messages(&context.messages),
        tools: Vec::new(),
    }
}

/// Re-emits a completed hidden OpenAI completion as native OmniRoute events.
///
/// The outer stream intentionally buffers the inner response before replaying
/// it. Forwarding the inner events would show raw XML tool tags to clients and
/// would expose a second, incorrect assistant message API.
pub(crate) fn replay_response(
    emitter: &mut MessageEmitter,
    response: &llm::AssistantMessage,
) -> Result<()> {
    emitter.message.usage = response.usage.clone();
    emitter.message.response_id = response.response_id.clone();
    emitter.message.response_model = response.response_model.clone();

    let raw = content_text(&response.content);
    let (mut prose, calls, problems) = parse_tool_calls(&raw);
    if !problems.is_empty() {
        let note = format!(
            "[omni-prompt-tools] Could not parse tool call(s). Re-emit each as:\n\
             <tool_call>{{\"name\":\"tool_name\",\"arguments\":{{}}}}</tool_call>\n- {}",
            problems.join("\n- ")
        );
        if !prose.is_empty() {
            prose.push_str("\n\n");
        }
        prose.push_str(&note);
    }

    if !prose.is_empty() {
        emitter.replay_text(&prose)?;
    }
    for call in &calls {
        emitter.replay_tool(llm::ToolCall {
            id: next_tool_call_id(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            ..llm::ToolCall::default()
        })?;
    }

    emitter.message.stop_reason = match response.stop_reason.as_str() {
        stream::STOP_LENGTH => stream::STOP_LENGTH.to_owned(),
        _ if !calls.is_empty() => stream::STOP_TOOL_USE.to_owned(),
        _ => stream::STOP_STOP.to_owned(),
    };
    emitter.message.raw_stop_reason = response.raw_stop_reason.clone();
    Ok(())
}

/// Renders native tool schemas into the strict prompt protocol understood by
/// chat-only OmniRoute models.
pub(crate) fn render_tool_protocol(tools: &[llm::Tool]) -> String {
    let mut lines = vec![
        "# Tool calling protocol".to_owned(),
        String::new(),
        "You are connected to GoshCoder through a chat-only adapter. Native/internal tool calling is NOT available.".to_owned(),
        "The only valid way to call tools is to write literal text <tool_call> blocks in your assistant message.".to_owned(),
        "If you need a tool, write any reasoning first, then end your message with one or more blocks exactly like:".to_owned(),
        String::new(),
        "<tool_call>".to_owned(),
        r#"{"name":"<tool_name>","arguments":{}}"#.to_owned(),
        "</tool_call>".to_owned(),
        String::new(),
        "Rules:".to_owned(),
        "- Use only this literal <tool_call> text protocol.".to_owned(),
        "- JSON must be valid; arguments must be an object matching the schema.".to_owned(),
        "- Do not wrap the JSON in Markdown fences.".to_owned(),
        "- Emit multiple blocks for multiple tools, then stop and wait for <tool_result> messages.".to_owned(),
        "- Never claim a tool ran unless a corresponding result was returned.".to_owned(),
        "- Never invent command output, file contents, or tool results.".to_owned(),
        "- If no tool is needed, answer normally with no <tool_call> block.".to_owned(),
        String::new(),
        "## Available tools".to_owned(),
        String::new(),
    ];
    for tool in tools {
        lines.push(format!("### {}", tool.name));
        lines.push(tool.description.clone());
        lines.push("Parameters (JSON Schema):".to_owned());
        lines.push(
            serde_json::to_string(&tool.parameters)
                .expect("serde_json::Value always serializes to JSON"),
        );
        lines.push(String::new());
    }
    lines.join("\n")
}

/// Extracts every complete literal `<tool_call>` block from a response.
///
/// Invalid blocks become user-visible repair instructions rather than failing
/// the turn, so the model can correct its own syntax on the next tool round.
pub(crate) fn parse_tool_calls(text: &str) -> (String, Vec<ParsedToolCall>, Vec<String>) {
    let mut calls = Vec::new();
    let mut problems = Vec::new();
    let mut prose = String::new();
    let mut remaining = text;

    while let Some(open_index) = remaining.find(OPEN_TOOL_CALL) {
        prose.push_str(&remaining[..open_index]);
        let after_open = &remaining[open_index + OPEN_TOOL_CALL.len()..];
        let Some(close_index) = after_open.find(CLOSE_TOOL_CALL) else {
            prose.push_str(&remaining[open_index..]);
            remaining = "";
            break;
        };

        let body = strip_optional_fence(&after_open[..close_index]);
        parse_tool_call(body, &mut calls, &mut problems);
        remaining = &after_open[close_index + CLOSE_TOOL_CALL.len()..];
    }
    prose.push_str(remaining);
    (prose.trim().to_owned(), calls, problems)
}

fn parse_tool_call(body: &str, calls: &mut Vec<ParsedToolCall>, problems: &mut Vec<String>) {
    #[derive(Deserialize)]
    struct RawToolCall {
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        arguments: Value,
    }

    let raw = match serde_json::from_str::<RawToolCall>(body) {
        Ok(raw) => raw,
        Err(error) => {
            problems.push(format!(
                "invalid JSON in <tool_call> ({error}): {}",
                truncate(body, 200)
            ));
            return;
        }
    };
    let name = raw.name.unwrap_or_default();
    if name.is_empty() {
        problems.push(format!(
            "tool_call missing a string \"name\": {}",
            truncate(body, 200)
        ));
        return;
    }
    let arguments = match raw.arguments {
        Value::Object(arguments) => arguments.into_iter().collect(),
        Value::String(arguments) => serde_json::from_str(&arguments).unwrap_or_default(),
        _ => BTreeMap::new(),
    };
    calls.push(ParsedToolCall { name, arguments });
}

fn strip_optional_fence(body: &str) -> &str {
    let mut value = body.trim();
    let Some(after_ticks) = value.strip_prefix("```") else {
        return value;
    };

    let after_language = if after_ticks
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("json"))
    {
        &after_ticks[4..]
    } else {
        after_ticks
    };
    if !after_language.is_empty()
        && !after_language
            .chars()
            .next()
            .is_some_and(char::is_whitespace)
    {
        return value;
    }
    value = after_language.trim_start();

    let without_trailing_space = value.trim_end();
    if let Some(without_fence) = without_trailing_space.strip_suffix("```") {
        without_fence.trim_end()
    } else {
        value
    }
}

fn flatten_messages(messages: &[llm::Message]) -> Vec<llm::Message> {
    let mut output = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            llm::Message::User(user) => {
                push_message(
                    &mut output,
                    "user",
                    user_content_text(&user.content),
                    user.timestamp,
                );
            }
            llm::Message::Assistant(assistant) => {
                let mut parts = vec![content_text(&assistant.content)];
                for block in &assistant.content {
                    if let llm::ContentBlock::ToolCall(call) = block {
                        let encoded = serde_json::to_string(&serde_json::json!({
                            "name": call.name,
                            "arguments": call.arguments,
                        }))
                        .expect("tool call content always serializes to JSON");
                        parts.push(format!("<tool_call>\n{encoded}\n</tool_call>"));
                    }
                }
                push_message(
                    &mut output,
                    "assistant",
                    non_empty(parts).join("\n\n"),
                    assistant.timestamp,
                );
            }
            llm::Message::ToolResult(result) => {
                let tag = if result.is_error {
                    "tool_result error"
                } else {
                    "tool_result"
                };
                let tool_name =
                    serde_json::to_string(&result.tool_name).expect("string always serializes");
                let tool_call_id =
                    serde_json::to_string(&result.tool_call_id).expect("string always serializes");
                push_message(
                    &mut output,
                    "user",
                    format!(
                        "<{tag} tool={tool_name} id={tool_call_id}>\n{}\n</tool_result>",
                        content_text(&result.content)
                    ),
                    result.timestamp,
                );
            }
        }
    }
    output
}

fn push_message(output: &mut Vec<llm::Message>, role: &str, text: String, timestamp: i64) {
    let text = text.trim().to_owned();
    if text.is_empty() {
        return;
    }
    if let Some(previous) = output.last_mut() {
        match (role, previous) {
            ("user", llm::Message::User(user)) => {
                user.content = llm::UserContent::Text(format!(
                    "{}\n\n{text}",
                    user_content_text(&user.content)
                ));
                return;
            }
            ("assistant", llm::Message::Assistant(assistant)) => {
                assistant.content = vec![llm::ContentBlock::text(format!(
                    "{}\n\n{text}",
                    content_text(&assistant.content)
                ))];
                return;
            }
            _ => {}
        }
    }

    if role == "user" {
        output.push(llm::Message::User(llm::UserMessage::text(text, timestamp)));
    } else {
        output.push(llm::Message::Assistant(Box::new(llm::AssistantMessage {
            role: "assistant".to_owned(),
            content: vec![llm::ContentBlock::text(text)],
            stop_reason: stream::STOP_STOP.to_owned(),
            timestamp,
            ..llm::AssistantMessage::default()
        })));
    }
}

fn user_content_text(content: &llm::UserContent) -> String {
    match content {
        llm::UserContent::Text(text) => text.clone(),
        llm::UserContent::Blocks(blocks) => content_text(blocks),
    }
}

fn content_text(content: &[llm::ContentBlock]) -> String {
    content
        .iter()
        .filter_map(llm::ContentBlock::plain_text)
        .collect()
}

fn non_empty(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect()
}

fn next_tool_call_id() -> String {
    let sequence = TOOL_CALL_SEQUENCE.fetch_add(1, Ordering::Relaxed) + 1;
    format!("call_omni_{:x}_{sequence}", now_millis())
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn truncate(value: &str, limit: usize) -> String {
    let mut characters = value.chars();
    let output = characters.by_ref().take(limit).collect::<String>();
    if characters.next().is_some() {
        format!("{output}...")
    } else {
        output
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parser_accepts_fenced_json_and_string_arguments() {
        let text = "before <tool_call>```json\n{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}\n```</tool_call> after";
        let (prose, calls, problems) = parse_tool_calls(text);

        assert_eq!(prose, "before  after");
        assert!(problems.is_empty());
        assert_eq!(
            calls,
            vec![ParsedToolCall {
                name: "bash".to_owned(),
                arguments: BTreeMap::from([("command".to_owned(), json!("pwd"))]),
            }]
        );
    }

    #[test]
    fn parser_removes_invalid_blocks_and_reports_a_repair_problem() {
        let (prose, calls, problems) =
            parse_tool_calls("before <tool_call>{\"arguments\":{}}</tool_call> after");

        assert_eq!(prose, "before  after");
        assert!(calls.is_empty());
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("missing a string"));
    }

    #[test]
    fn flattened_history_replays_tools_as_prompt_protocol() {
        let messages = vec![
            llm::Message::User(llm::UserMessage::text("inspect", 1)),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![
                    llm::ContentBlock::text("I will inspect it."),
                    llm::ContentBlock::ToolCall(llm::ToolCall {
                        id: "call_1".to_owned(),
                        name: "read".to_owned(),
                        arguments: BTreeMap::from([("path".to_owned(), json!("main.rs"))]),
                        ..llm::ToolCall::default()
                    }),
                ],
                timestamp: 2,
                ..llm::AssistantMessage::default()
            })),
            llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: "call_1".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![llm::ContentBlock::text("fn main() {}")],
                is_error: true,
                timestamp: 3,
                ..llm::ToolResultMessage::default()
            })),
            llm::Message::User(llm::UserMessage::text("continue", 4)),
        ];

        let flattened = flatten_messages(&messages);
        assert_eq!(flattened.len(), 3);
        assert_eq!(flattened[0].text_preview(), "inspect");
        assert_eq!(
            flattened[1].text_preview(),
            "I will inspect it.\n\n<tool_call>\n{\"arguments\":{\"path\":\"main.rs\"},\"name\":\"read\"}\n</tool_call>"
        );
        assert_eq!(
            flattened[2].text_preview(),
            "<tool_result error tool=\"read\" id=\"call_1\">\nfn main() {}\n</tool_result>\n\ncontinue"
        );
    }

    #[test]
    fn protocol_includes_each_tool_schema() {
        let protocol = render_tool_protocol(&[llm::Tool {
            name: "read".to_owned(),
            description: "Read a file".to_owned(),
            parameters: json!({"type": "object"}),
            constrained_sampling: None,
        }]);

        assert!(protocol.contains("# Tool calling protocol"));
        assert!(protocol.contains("### read"));
        assert!(protocol.contains(r#"{"type":"object"}"#));
    }
}
