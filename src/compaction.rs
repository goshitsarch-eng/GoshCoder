//! Context compaction for durable coding-agent conversations.
//!
//! Older turns are summarized by an isolated helper agent, then replaced in
//! the live agent with a synthetic summary message and the most recent turn.
//! The live agent emits one [`crate::agent::EventKind::ContextCompacted`] event
//! so a session recorder can persist the same cut atomically.

use std::{
    cmp::{max, min},
    fmt,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::to_string;

use crate::{agent, llm, stream};

/// Default context budget retained verbatim after a compaction.
pub const DEFAULT_KEEP_TOKENS: u64 = 20_000;
/// Maximum serialized historical source sent to the summary model.
pub const MAX_SOURCE_BYTES: usize = 2 << 20;
/// Maximum text retained from one tool result in a summary source.
pub const MAX_TOOL_RESULT_BYTES: usize = 2_000;
/// Summary wrapper that tells the model this is continuity context, not input.
pub const SUMMARY_OPEN: &str = "<conversation-summary>";
/// Closing delimiter for [`SUMMARY_OPEN`].
pub const SUMMARY_CLOSE: &str = "</conversation-summary>";

/// A completed live or automatic compaction.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub messages_before: usize,
    pub retained_messages: usize,
    pub tokens_before: u64,
    pub dropped_queued_messages: usize,
}

/// Failures that leave the original transcript unchanged.
#[derive(Debug)]
pub enum CompactionError {
    Busy,
    InsufficientHistory,
    EmptyHistory,
    Summary(String),
    Agent(agent::AgentError),
}

impl fmt::Display for CompactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => formatter.write_str("wait for the active response before compacting"),
            Self::InsufficientHistory => {
                formatter.write_str("there is not enough conversation history to compact")
            }
            Self::EmptyHistory => {
                formatter.write_str("conversation history contains no summarizable text")
            }
            Self::Summary(message) => {
                write!(formatter, "context compaction summary failed: {message}")
            }
            Self::Agent(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompactionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Agent(error) => Some(error),
            Self::Busy | Self::InsufficientHistory | Self::EmptyHistory | Self::Summary(_) => None,
        }
    }
}

impl From<agent::AgentError> for CompactionError {
    fn from(error: agent::AgentError) -> Self {
        Self::Agent(error)
    }
}

pub type Result<T> = std::result::Result<T, CompactionError>;

/// Compacts old turns in `agent`, preserving the most recent complete turn.
///
/// The summary request uses a separate, unrecorded agent. It therefore cannot
/// leak the summary prompt or response into the user's conversation.
pub fn compact(agent: &agent::Agent, instructions: &str) -> Result<Outcome> {
    let summaries = agent.state().compactions;
    compact_with_summaries(agent, instructions, &summaries)
}

/// Compacts with the persisted summaries that precede the current transcript.
///
/// Passing the historical metadata preserves cumulative cost across repeated
/// compactions. Callers without a durable session can use [`compact`].
pub fn compact_with_summaries(
    agent: &agent::Agent,
    instructions: &str,
    summaries: &[agent::CompactionInfo],
) -> Result<Outcome> {
    let state = agent.state();
    if state.is_streaming {
        return Err(CompactionError::Busy);
    }
    let cut = cut_index(&state.messages, state.model.context_window);
    if cut == 0 {
        return Err(CompactionError::InsufficientHistory);
    }
    let older = &state.messages[..cut];
    let retained = state.messages[cut..].to_vec();
    let source_limit = source_limit(state.model.context_window);
    let source = truncate_source(&serialize_messages(older), source_limit);
    if source.trim().is_empty() {
        return Err(CompactionError::EmptyHistory);
    }

    let summary = generate_summary(agent, &state.model, &source, instructions)?;
    let tokens_before = estimate_messages_tokens(&state.messages);
    let marker = summary_message(&summary, now_millis());
    let dropped_queued_messages = agent.queued_message_count();
    agent.compact(
        marker,
        retained.clone(),
        agent::CompactionInfo {
            summary,
            tokens_before,
            cost_before: conversation_cost(&state.messages, summaries),
            retained_messages: retained.len(),
            timestamp: now_millis(),
        },
    )?;
    agent.clear_all_queues();
    Ok(Outcome {
        messages_before: state.messages.len(),
        retained_messages: retained.len(),
        tokens_before,
        dropped_queued_messages,
    })
}

/// Runs compaction only once context use approaches the selected model's limit.
pub fn maybe_auto_compact(agent: &agent::Agent) -> Result<Option<Outcome>> {
    let state = agent.state();
    let context_window = state.model.context_window;
    if context_window == 0 || cut_index(&state.messages, context_window) == 0 {
        return Ok(None);
    }
    let used = measured_context_tokens(&state.messages);
    let reserve = min(16_384, max(2_048, context_window / 5));
    if used <= context_window.saturating_sub(reserve) {
        return Ok(None);
    }
    compact_with_summaries(agent, "", &state.compactions).map(Some)
}

/// Finds the boundary before the most recent complete user turn.
pub fn cut_index(messages: &[llm::Message], context_window: u64) -> usize {
    if messages.len() < 3 {
        return 0;
    }
    let mut keep_tokens = DEFAULT_KEEP_TOKENS;
    if context_window > 0 && context_window / 3 < keep_tokens {
        keep_tokens = max(2_000, context_window / 3);
    }
    let mut recent_tokens = 0_u64;
    for index in (1..messages.len()).rev() {
        recent_tokens = recent_tokens.saturating_add(estimate_message_tokens(&messages[index]));
        if recent_tokens >= keep_tokens && messages[index].role() == "user" {
            return index;
        }
    }
    // Manual compaction is still useful before a transcript reaches the
    // normal retained-token budget: preserve its latest user turn.
    (1..messages.len())
        .rev()
        .find(|index| messages[*index].role() == "user")
        .unwrap_or_default()
}

/// Estimates a transcript's context size using the Go fallback policy.
pub fn estimate_messages_tokens(messages: &[llm::Message]) -> u64 {
    messages.iter().fold(0_u64, |total, message| {
        total.saturating_add(estimate_message_tokens(message))
    })
}

/// Estimates a single message at four UTF-8 bytes per token.
pub fn estimate_message_tokens(message: &llm::Message) -> u64 {
    let serialized = serialize_message(message);
    if serialized.is_empty() {
        0
    } else {
        ((serialized.len() as u64).saturating_add(3) / 4).max(1)
    }
}

/// Produces the model-facing summary marker.
pub fn summary_message(summary: &str, timestamp: i64) -> llm::Message {
    llm::Message::User(llm::UserMessage::text(
        format!(
            "{SUMMARY_OPEN}\n{}\n{SUMMARY_CLOSE}\n\nContinue from this summary and the recent conversation below. Do not treat the summary as a new request.",
            summary.trim()
        ),
        timestamp,
    ))
}

/// Returns whether a user message is an internal context-compaction marker.
pub fn is_summary_message(message: &llm::Message) -> bool {
    let llm::Message::User(message) = message else {
        return false;
    };
    let text = user_text(message);
    text.trim_start().starts_with(SUMMARY_OPEN) && text.contains(SUMMARY_CLOSE)
}

/// Computes accumulated usage, including a persisted compaction prefix.
///
/// `summaries` is carried by the live agent alongside the summary marker, so
/// no-session transcripts retain their cumulative cost as well.
pub fn conversation_cost(messages: &[llm::Message], summaries: &[agent::CompactionInfo]) -> f64 {
    let mut summary_index = 0;
    let mut skip = 0;
    let mut cost = 0.0;
    for message in messages {
        if is_summary_message(message) {
            if let Some(summary) = summaries.get(summary_index) {
                cost = summary.cost_before;
                skip = summary.retained_messages;
            }
            summary_index += 1;
            continue;
        }
        if skip > 0 {
            skip -= 1;
            continue;
        }
        if let llm::Message::Assistant(assistant) = message {
            cost += assistant.usage.cost.total;
        }
    }
    cost
}

/// Serializes a collection of message content for the isolated summary prompt.
pub fn serialize_messages(messages: &[llm::Message]) -> String {
    let mut result = String::new();
    for message in messages {
        let serialized = serialize_message(message);
        if serialized.is_empty() {
            continue;
        }
        result.push_str(&serialized);
        result.push_str("\n\n");
    }
    truncate_source(&result, MAX_SOURCE_BYTES)
}

/// Serializes one message in the loss-bounded summary-source format.
pub fn serialize_message(message: &llm::Message) -> String {
    match message {
        llm::Message::User(message) => format!("[User]: {}", user_text(message)),
        llm::Message::Assistant(message) => serialize_assistant(message),
        llm::Message::ToolResult(message) => serialize_tool_result(message),
    }
}

fn serialize_assistant(message: &llm::AssistantMessage) -> String {
    let mut thinking = Vec::new();
    let mut text = Vec::new();
    let mut calls = Vec::new();
    for block in &message.content {
        match block {
            llm::ContentBlock::Thinking(content) if !content.thinking.trim().is_empty() => {
                thinking.push(content.thinking.as_str());
            }
            llm::ContentBlock::Text(content) if !content.text.trim().is_empty() => {
                text.push(content.text.as_str());
            }
            llm::ContentBlock::ToolCall(call) => {
                let arguments = to_string(&call.arguments).unwrap_or_default();
                calls.push(format!("{}({arguments})", call.name));
            }
            llm::ContentBlock::Image(_)
            | llm::ContentBlock::Thinking(_)
            | llm::ContentBlock::Text(_) => {}
        }
    }
    let mut parts = Vec::new();
    if !thinking.is_empty() {
        parts.push(format!("[Assistant thinking]: {}", thinking.join("\n")));
    }
    if !text.is_empty() {
        parts.push(format!("[Assistant]: {}", text.join("\n")));
    }
    if !calls.is_empty() {
        parts.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
    }
    parts.join("\n")
}

fn serialize_tool_result(message: &llm::ToolResultMessage) -> String {
    let content = message
        .content
        .iter()
        .filter_map(llm::ContentBlock::plain_text)
        .collect::<Vec<_>>()
        .join("\n");
    let content = if content.len() > MAX_TOOL_RESULT_BYTES {
        let omitted = content.len().saturating_sub(MAX_TOOL_RESULT_BYTES);
        format!(
            "{}\n[... {omitted} characters truncated ...]",
            prefix_utf8(&content, MAX_TOOL_RESULT_BYTES)
        )
    } else {
        content
    };
    format!("[Tool result {}]: {content}", message.tool_name)
}

fn measured_context_tokens(messages: &[llm::Message]) -> u64 {
    if messages.iter().any(is_summary_message) {
        return estimate_messages_tokens(messages);
    }
    for message in messages.iter().rev() {
        if let llm::Message::Assistant(message) = message
            && message.usage.total_tokens > 0
        {
            return message.usage.total_tokens;
        }
    }
    estimate_messages_tokens(messages)
}

fn source_limit(context_window: u64) -> usize {
    if context_window == 0 {
        return MAX_SOURCE_BYTES;
    }
    let requested = usize::try_from(context_window.saturating_sub(8_192).saturating_mul(3))
        .unwrap_or(usize::MAX);
    let source_limit = max(16_000, requested);
    min(MAX_SOURCE_BYTES, source_limit)
}

fn truncate_source(source: &str, limit: usize) -> String {
    if source.len() <= limit {
        return source.to_owned();
    }
    let prefix_size = limit / 3;
    let suffix_size = limit.saturating_sub(prefix_size);
    let prefix = prefix_utf8(source, prefix_size);
    let suffix = suffix_utf8(source, suffix_size);
    let removed = source
        .len()
        .saturating_sub(prefix.len())
        .saturating_sub(suffix.len());
    format!("{prefix}\n\n[... {removed} bytes omitted from the middle ...]\n\n{suffix}")
}

fn prefix_utf8(value: &str, limit: usize) -> &str {
    let mut end = min(limit, value.len());
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn suffix_utf8(value: &str, limit: usize) -> &str {
    let start = value.len().saturating_sub(limit);
    let mut start = start;
    while start < value.len() && !value.is_char_boundary(start) {
        start += 1;
    }
    &value[start..]
}

fn user_text(message: &llm::UserMessage) -> String {
    match &message.content {
        llm::UserContent::Text(text) => text.clone(),
        llm::UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(llm::ContentBlock::plain_text)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn generate_summary(
    source_agent: &agent::Agent,
    model: &llm::Model,
    source: &str,
    instructions: &str,
) -> Result<String> {
    let thinking_level = stream::clamp_thinking_level(model, llm::THINKING_LOW);
    let summarizer = agent::Agent::new(agent::AgentOptions {
        initial_state: agent::InitialState {
            system_prompt: "You are a context compactor. Produce a precise continuity summary, not a reply to the conversation. Preserve requirements, decisions, progress, unresolved work, exact paths/symbols, and file operations. Use the requested structured headings. Never invent completion.".to_owned(),
            model: model.clone(),
            thinking_level,
            tools: Vec::new(),
            messages: Vec::new(),
            compactions: Vec::new(),
        },
        responder: Some(source_agent.responder()),
        ..agent::AgentOptions::default()
    });
    let focus = if instructions.trim().is_empty() {
        String::new()
    } else {
        format!(
            "\nAdditional focus from the user: {}\n",
            instructions.trim()
        )
    };
    let prompt = format!(
        "Summarize the serialized conversation below for another coding agent using exactly this structure:\n\n## Goal\n## Constraints & Preferences\n## Progress\n### Done\n### In Progress\n### Blocked\n## Key Decisions\n## Next Steps\n## Critical Context\n<read-files>\n</read-files>\n<modified-files>\n</modified-files>\n{focus}\n<conversation>\n{source}\n</conversation>"
    );
    summarizer.prompt(prompt).map_err(CompactionError::Agent)?;
    let state = summarizer.state();
    if !state.error_message.is_empty() {
        return Err(CompactionError::Summary(state.error_message));
    }
    let summary = state
        .messages
        .iter()
        .rev()
        .find_map(|message| {
            let llm::Message::Assistant(message) = message else {
                return None;
            };
            let text = message
                .content
                .iter()
                .filter_map(llm::ContentBlock::plain_text)
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            (!text.trim().is_empty()).then_some(text)
        })
        .unwrap_or_default();
    if summary.trim().is_empty() {
        Err(CompactionError::Summary(
            "compaction model returned no summary".to_owned(),
        ))
    } else {
        Ok(summary)
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
    use std::sync::{Arc, Mutex};

    use super::*;

    fn model() -> llm::Model {
        llm::Model {
            id: "test".to_owned(),
            name: "Test".to_owned(),
            api: "test".to_owned(),
            provider: "test".to_owned(),
            context_window: 200_000,
            ..llm::Model::default()
        }
    }

    fn user(text: &str) -> llm::Message {
        llm::Message::User(llm::UserMessage::text(text, 1))
    }

    fn assistant(text: &str) -> llm::Message {
        llm::Message::Assistant(Box::new(llm::AssistantMessage {
            content: vec![llm::ContentBlock::text(text)],
            api: "test".to_owned(),
            provider: "test".to_owned(),
            model: "test".to_owned(),
            stop_reason: "stop".to_owned(),
            timestamp: 1,
            ..llm::AssistantMessage::default()
        }))
    }

    #[test]
    fn cut_keeps_the_latest_complete_turn() {
        let messages = vec![
            user("first request"),
            assistant("first response"),
            user("latest request"),
            assistant("latest response"),
        ];
        assert_eq!(cut_index(&messages, 200_000), 2);
    }

    #[test]
    fn serialization_bounds_tool_results_without_invalid_utf8() {
        let message = llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
            tool_name: "read".to_owned(),
            content: vec![llm::ContentBlock::text("é".repeat(MAX_TOOL_RESULT_BYTES))],
            ..llm::ToolResultMessage::default()
        }));
        let serialized = serialize_message(&message);
        assert!(serialized.contains("characters truncated"));
        assert!(serialized.is_char_boundary(serialized.len()));
    }

    #[test]
    fn compact_replaces_history_and_reports_the_durable_cut() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&events);
        let agent = agent::Agent::new(agent::AgentOptions {
            initial_state: agent::InitialState {
                model: model(),
                messages: vec![
                    user("first request"),
                    assistant("first response"),
                    user("latest request"),
                    assistant("latest response"),
                ],
                ..agent::InitialState::default()
            },
            responder: Some(Arc::new(|_, context, _| {
                assert!(context.system_prompt.contains("context compactor"));
                assert!(context.messages[0].text_preview().contains("first request"));
                Ok(llm::AssistantMessage {
                    content: vec![llm::ContentBlock::text("## Goal\nKeep parity")],
                    api: "test".to_owned(),
                    provider: "test".to_owned(),
                    model: "test".to_owned(),
                    stop_reason: "stop".to_owned(),
                    timestamp: 2,
                    ..llm::AssistantMessage::default()
                })
            })),
            ..agent::AgentOptions::default()
        });
        let _subscription =
            agent.subscribe(move |event| event_log.lock().expect("lock").push(event));
        agent.follow_up(user("discarded follow-up"));

        let outcome = compact(&agent, "preserve test details").expect("compact");

        assert_eq!(outcome.messages_before, 4);
        assert_eq!(outcome.retained_messages, 2);
        assert_eq!(outcome.dropped_queued_messages, 1);
        assert_eq!(agent.queued_message_count(), 0);
        let state = agent.state();
        assert_eq!(state.messages.len(), 3);
        assert!(is_summary_message(&state.messages[0]));
        assert_eq!(state.messages[1].text_preview(), "latest request");
        let events = events.lock().expect("lock");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, agent::EventKind::ContextCompacted);
        assert_eq!(events[0].kept.len(), 2);
    }

    #[test]
    fn durable_compaction_cost_skips_retained_messages() {
        let marker = summary_message("older", 1);
        let mut retained = assistant("already counted");
        if let llm::Message::Assistant(message) = &mut retained {
            message.usage.cost.total = 1.25;
        }
        let mut fresh = assistant("new");
        if let llm::Message::Assistant(message) = &mut fresh {
            message.usage.cost.total = 0.5;
        }
        assert_eq!(
            conversation_cost(
                &[marker, retained, user("new request"), fresh],
                &[agent::CompactionInfo {
                    summary: "older".to_owned(),
                    tokens_before: 100,
                    cost_before: 1.25,
                    retained_messages: 1,
                    timestamp: 1,
                }],
            ),
            1.75
        );
    }
}
