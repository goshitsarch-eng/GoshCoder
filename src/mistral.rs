//! Mistral Conversations request and streaming adapter.
//!
//! Mistral's `/v1/chat/completions` endpoint resembles OpenAI Chat
//! Completions, but it has distinct replay, tool-ID, thinking, cache-affinity,
//! and streamed-content rules. Keeping those rules here avoids routing Mistral
//! models through a superficially similar but wire-incompatible adapter.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::{SystemTime, UNIX_EPOCH},
};

use reqwest::blocking::Response;
use serde_json::{Map, Value, json};

use crate::{
    agent, llm,
    providers::{MessageEmitter, ProviderAdapterError, Result},
    stream,
};

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;

/// Builds a Mistral Conversations request body from a common agent turn.
pub(crate) fn build_mistral_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.id.clone()));
    body.insert(
        "messages".to_owned(),
        Value::Array(mistral_messages(model, context)),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    if !context.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(mistral_tools(&context.tools)?),
        );
    }
    if let Some(max_tokens) = mistral_requested_max_tokens(model, context, options) {
        body.insert("max_tokens".to_owned(), Value::Number(max_tokens.into()));
    }
    if let Some(temperature) = options.temperature {
        let temperature = serde_json::Number::from_f64(temperature).ok_or_else(|| {
            ProviderAdapterError::Protocol(
                "Mistral temperature must be a finite JSON number".to_owned(),
            )
        })?;
        body.insert("temperature".to_owned(), Value::Number(temperature));
    }
    if let Some(tool_choice) = mistral_tool_choice(options.tool_choice.as_ref()) {
        body.insert("tool_choice".to_owned(), tool_choice);
    }
    if let Some((name, value)) = mistral_reasoning_option(model, options) {
        body.insert(name.to_owned(), Value::String(value));
    }
    if should_cache_prompt(options) {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(options.session_id.clone()),
        );
    }
    if let Some(Value::Object(sampling_params)) = &model.sampling_params {
        for (name, value) in sampling_params {
            body.insert(name.clone(), value.clone());
        }
    }
    Ok(Value::Object(body))
}

fn mistral_requested_max_tokens(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Option<u64> {
    let requested = options
        .max_tokens
        .filter(|tokens| *tokens > 0)
        .unwrap_or(model.max_tokens);
    (requested != 0).then(|| stream::clamp_max_tokens_to_context(model, context, requested))
}

fn mistral_tool_choice(choice: Option<&Value>) -> Option<Value> {
    match choice {
        Some(Value::String(value))
            if matches!(value.as_str(), "auto" | "none" | "any" | "required") =>
        {
            Some(Value::String(value.clone()))
        }
        Some(Value::String(_)) | None => None,
        Some(value) => Some(value.clone()),
    }
}

pub(crate) fn should_cache_prompt(options: &agent::RequestOptions) -> bool {
    options.cache_retention != agent::CacheRetention::None && !options.session_id.is_empty()
}

fn mistral_reasoning_option(
    model: &llm::Model,
    options: &agent::RequestOptions,
) -> Option<(&'static str, String)> {
    if !model.reasoning || options.thinking_level.is_empty() {
        return None;
    }
    let level = stream::clamp_thinking_level(model, &options.thinking_level);
    if level == llm::THINKING_OFF {
        return None;
    }
    if mistral_uses_reasoning_effort(model) {
        let effort = model
            .thinking_level_map
            .get(&level)
            .and_then(|mapped| mapped.as_deref())
            .filter(|mapped| !mapped.is_empty())
            .unwrap_or("high")
            .to_owned();
        Some(("reasoning_effort", effort))
    } else {
        Some(("prompt_mode", "reasoning".to_owned()))
    }
}

fn mistral_uses_reasoning_effort(model: &llm::Model) -> bool {
    matches!(
        model.id.as_str(),
        "mistral-small-2603" | "mistral-small-latest" | "mistral-medium-3.5"
    )
}

fn mistral_tools(tools: &[llm::Tool]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            let strict = tool
                .constrained_sampling
                .as_ref()
                .and_then(Value::as_object)
                .is_some_and(|sampling| {
                    sampling.get("type").and_then(Value::as_str) == Some("json_schema")
                });
            let parameters = if tool.parameters.is_null() {
                Value::Object(Map::new())
            } else {
                tool.parameters.clone()
            };
            Ok(json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                    "strict": strict,
                },
            }))
        })
        .collect()
}

fn mistral_messages(model: &llm::Model, context: &llm::Context) -> Vec<Value> {
    let mut messages = Vec::new();
    if !context.system_prompt.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": context.system_prompt,
        }));
    }
    for message in transform_mistral_messages(&context.messages, model) {
        match message {
            llm::Message::User(user) => {
                if let Some(message) = mistral_user_message(&user.content, model.supports_images())
                {
                    messages.push(message);
                }
            }
            llm::Message::Assistant(assistant) => {
                if let Some(message) = mistral_assistant_message(&assistant) {
                    messages.push(message);
                }
            }
            llm::Message::ToolResult(tool_result) => {
                messages.push(mistral_tool_result_message(
                    &tool_result,
                    model.supports_images(),
                ));
            }
        }
    }
    messages
}

fn mistral_user_message(content: &llm::UserContent, supports_images: bool) -> Option<Value> {
    match content {
        llm::UserContent::Text(text) => Some(json!({"role": "user", "content": text})),
        llm::UserContent::Blocks(blocks) => {
            let mut parts = Vec::new();
            let mut had_images = false;
            for block in blocks {
                match block {
                    llm::ContentBlock::Text(text) => {
                        parts.push(json!({"type": "text", "text": text.text}));
                    }
                    llm::ContentBlock::Image(image) => {
                        had_images = true;
                        if supports_images {
                            parts.push(json!({
                                "type": "image_url",
                                "image_url": mistral_data_uri(image),
                            }));
                        }
                    }
                    llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => {}
                }
            }
            if !parts.is_empty() {
                Some(json!({"role": "user", "content": parts}))
            } else if had_images && !supports_images {
                Some(json!({
                    "role": "user",
                    "content": "(image omitted: model does not support images)",
                }))
            } else {
                None
            }
        }
    }
}

fn mistral_assistant_message(assistant: &llm::AssistantMessage) -> Option<Value> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &assistant.content {
        match block {
            llm::ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                content.push(json!({"type": "text", "text": text.text}));
            }
            llm::ContentBlock::Thinking(thinking) if !thinking.thinking.trim().is_empty() => {
                content.push(json!({
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": thinking.thinking}],
                }));
            }
            llm::ContentBlock::ToolCall(call) => {
                let arguments =
                    serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned());
                tool_calls.push(json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": arguments,
                    },
                    "index": 0,
                }));
            }
            llm::ContentBlock::Text(_)
            | llm::ContentBlock::Thinking(_)
            | llm::ContentBlock::Image(_) => {}
        }
    }
    if content.is_empty() && tool_calls.is_empty() {
        return None;
    }
    let mut message = Map::from_iter([
        ("role".to_owned(), Value::String("assistant".to_owned())),
        ("prefix".to_owned(), Value::Bool(false)),
    ]);
    if !content.is_empty() {
        message.insert("content".to_owned(), Value::Array(content));
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    Some(Value::Object(message))
}

fn mistral_tool_result_message(
    tool_result: &llm::ToolResultMessage,
    supports_images: bool,
) -> Value {
    let mut text = Vec::new();
    let mut images = Vec::new();
    for block in &tool_result.content {
        match block {
            llm::ContentBlock::Text(content) => text.push(content.text.as_str()),
            llm::ContentBlock::Image(image) => images.push(image),
            llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => {}
        }
    }
    let mut content = vec![json!({
        "type": "text",
        "text": mistral_tool_result_text(
            &text.join("\n"),
            !images.is_empty(),
            supports_images,
            tool_result.is_error,
        ),
    })];
    if supports_images {
        for image in images {
            content.push(json!({
                "type": "image_url",
                "image_url": mistral_data_uri(image),
            }));
        }
    }
    json!({
        "role": "tool",
        "tool_call_id": tool_result.tool_call_id,
        "name": tool_result.tool_name,
        "content": content,
    })
}

fn mistral_tool_result_text(
    text: &str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> String {
    let prefix = if is_error { "[tool error] " } else { "" };
    let text = text.trim();
    if !text.is_empty() {
        let suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{prefix}{text}{suffix}");
    }
    if has_images {
        return if supports_images {
            format!("{prefix}(see attached image)")
        } else {
            format!("{prefix}(image omitted: model does not support images)")
        };
    }
    format!("{prefix}(no tool output)")
}

fn mistral_data_uri(image: &llm::ImageContent) -> String {
    format!("data:{};base64,{}", image.mime_type, image.data)
}

/// Normalizes a transcript for Mistral replay.
///
/// The generic transform behavior is shared with Bedrock and Google: drop
/// failed assistant turns, downgrade unsupported images and cross-model
/// thinking, remap incompatible tool IDs, and synthesize missing tool results.
pub(crate) fn transform_mistral_messages(
    messages: &[llm::Message],
    model: &llm::Model,
) -> Vec<llm::Message> {
    let image_aware = messages
        .iter()
        .cloned()
        .map(|message| downgrade_mistral_message_images(message, model))
        .collect::<Vec<_>>();
    let mut normalized_tool_ids = MistralToolCallIdNormalizer::default();
    let mut tool_call_ids = BTreeMap::new();
    let mut transformed = Vec::with_capacity(image_aware.len());

    for message in image_aware {
        match message {
            llm::Message::Assistant(assistant) => {
                let same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;
                let mut copy = (*assistant).clone();
                copy.content = copy
                    .content
                    .into_iter()
                    .filter_map(|block| match block {
                        llm::ContentBlock::Thinking(thinking) => {
                            if thinking.redacted {
                                return same_model.then_some(llm::ContentBlock::Thinking(thinking));
                            }
                            if same_model && !thinking.thinking_signature.is_empty() {
                                return Some(llm::ContentBlock::Thinking(thinking));
                            }
                            if thinking.thinking.trim().is_empty() {
                                return None;
                            }
                            if same_model {
                                Some(llm::ContentBlock::Thinking(thinking))
                            } else {
                                Some(llm::ContentBlock::text(thinking.thinking))
                            }
                        }
                        llm::ContentBlock::ToolCall(mut tool_call) => {
                            if !same_model {
                                tool_call.thought_signature.clear();
                                let normalized = normalized_tool_ids.normalize(&tool_call.id);
                                if normalized != tool_call.id {
                                    tool_call_ids.insert(tool_call.id.clone(), normalized.clone());
                                    tool_call.id = normalized;
                                }
                            }
                            Some(llm::ContentBlock::ToolCall(tool_call))
                        }
                        other => Some(other),
                    })
                    .collect();
                transformed.push(llm::Message::Assistant(Box::new(copy)));
            }
            llm::Message::ToolResult(tool_result) => {
                let mut copy = (*tool_result).clone();
                if let Some(normalized) = tool_call_ids.get(&copy.tool_call_id) {
                    copy.tool_call_id = normalized.clone();
                }
                transformed.push(llm::Message::ToolResult(Box::new(copy)));
            }
            other => transformed.push(other),
        }
    }

    let mut result = Vec::with_capacity(transformed.len());
    let mut pending_tool_calls = Vec::<llm::ToolCall>::new();
    let mut existing_tool_results = BTreeSet::<String>::new();
    for message in transformed {
        match message {
            llm::Message::Assistant(assistant) => {
                flush_missing_mistral_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_results,
                );
                if matches!(
                    assistant.stop_reason.as_str(),
                    stream::STOP_ERROR | stream::STOP_ABORTED
                ) {
                    continue;
                }
                let tool_calls = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        llm::ContentBlock::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_tool_results.clear();
                }
                result.push(llm::Message::Assistant(assistant));
            }
            llm::Message::ToolResult(tool_result) => {
                existing_tool_results.insert(tool_result.tool_call_id.clone());
                result.push(llm::Message::ToolResult(tool_result));
            }
            llm::Message::User(user) => {
                flush_missing_mistral_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_results,
                );
                result.push(llm::Message::User(user));
            }
        }
    }
    flush_missing_mistral_tool_results(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_results,
    );
    result
}

fn downgrade_mistral_message_images(message: llm::Message, model: &llm::Model) -> llm::Message {
    if model.supports_images() {
        return message;
    }
    match message {
        llm::Message::User(mut user) => {
            if let llm::UserContent::Blocks(blocks) = user.content {
                user.content = llm::UserContent::Blocks(replace_images_with_placeholder(
                    blocks,
                    "(image omitted: model does not support images)",
                ));
            }
            llm::Message::User(user)
        }
        llm::Message::ToolResult(tool_result) => {
            let mut copy = (*tool_result).clone();
            copy.content = replace_images_with_placeholder(
                copy.content,
                "(tool image omitted: model does not support images)",
            );
            llm::Message::ToolResult(Box::new(copy))
        }
        other => other,
    }
}

fn replace_images_with_placeholder(
    blocks: Vec<llm::ContentBlock>,
    placeholder: &str,
) -> Vec<llm::ContentBlock> {
    let mut output = Vec::with_capacity(blocks.len());
    let mut previous_was_placeholder = false;
    for block in blocks {
        if matches!(block, llm::ContentBlock::Image(_)) {
            if !previous_was_placeholder {
                output.push(llm::ContentBlock::text(placeholder));
            }
            previous_was_placeholder = true;
            continue;
        }
        previous_was_placeholder =
            matches!(&block, llm::ContentBlock::Text(text) if text.text == placeholder);
        output.push(block);
    }
    output
}

fn flush_missing_mistral_tool_results(
    result: &mut Vec<llm::Message>,
    pending_tool_calls: &mut Vec<llm::ToolCall>,
    existing_tool_results: &mut BTreeSet<String>,
) {
    for tool_call in pending_tool_calls.drain(..) {
        if !existing_tool_results.contains(&tool_call.id) {
            result.push(llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                content: vec![llm::ContentBlock::text("No result provided")],
                is_error: true,
                timestamp: mistral_now_millis(),
                ..llm::ToolResultMessage::default()
            })));
        }
    }
    existing_tool_results.clear();
}

fn mistral_now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[derive(Default)]
struct MistralToolCallIdNormalizer {
    forward: BTreeMap<String, String>,
    reverse: BTreeMap<String, String>,
}

impl MistralToolCallIdNormalizer {
    fn normalize(&mut self, source: &str) -> String {
        if let Some(existing) = self.forward.get(source) {
            return existing.clone();
        }
        let mut attempt = 0;
        loop {
            let candidate = derive_mistral_tool_call_id(source, attempt);
            match self.reverse.get(&candidate) {
                None => {
                    self.forward.insert(source.to_owned(), candidate.clone());
                    self.reverse.insert(candidate.clone(), source.to_owned());
                    return candidate;
                }
                Some(owner) if owner == source => {
                    self.forward.insert(source.to_owned(), candidate.clone());
                    return candidate;
                }
                Some(_) => attempt += 1,
            }
        }
    }
}

fn derive_mistral_tool_call_id(source: &str, attempt: usize) -> String {
    let normalized = source
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>();
    if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let mut seed = if normalized.is_empty() {
        source.to_owned()
    } else {
        normalized
    };
    if attempt > 0 {
        seed.push(':');
        seed.push_str(&attempt.to_string());
    }
    let mut hash = mistral_short_hash(&seed)
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .collect::<String>();
    hash.truncate(MISTRAL_TOOL_CALL_ID_LENGTH);
    hash
}

fn mistral_short_hash(value: &str) -> String {
    let mut first = 0xdead_beef_u32;
    let mut second = 0x41c6_ce57_u32;
    for character in value.encode_utf16() {
        first = (first ^ u32::from(character)).wrapping_mul(2_654_435_761);
        second = (second ^ u32::from(character)).wrapping_mul(1_597_334_677);
    }
    first = (first ^ (first >> 16)).wrapping_mul(2_246_822_507)
        ^ (second ^ (second >> 13)).wrapping_mul(3_266_489_909);
    second = (second ^ (second >> 16)).wrapping_mul(2_246_822_507)
        ^ (first ^ (first >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", mistral_base36(second), mistral_base36(first))
}

fn mistral_base36(mut value: u32) -> String {
    if value == 0 {
        return "0".to_owned();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut characters = [0_u8; 7];
    let mut index = characters.len();
    while value > 0 {
        index -= 1;
        characters[index] = DIGITS[(value % 36) as usize];
        value /= 36;
    }
    String::from_utf8(characters[index..].to_vec()).expect("base36 output is ASCII")
}

struct MistralToolCallState {
    content_index: usize,
    arguments: stream::IncrementalJsonObjectParser,
}

/// Consumes Mistral's SSE response and emits common assistant message events.
pub(crate) fn consume_mistral_conversations(
    response: Response,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let mut reader = stream::SseReader::new(response);
    let mut text_index = None;
    let mut thinking_index = None;
    let mut tool_calls = BTreeMap::<String, MistralToolCallState>::new();
    let mut tool_call_keys_by_index = BTreeMap::<i64, String>::new();
    let mut tool_call_order = Vec::new();
    let mut saw_finish_reason = false;

    while let Some(event) = reader.next_event()? {
        ensure_not_cancelled(cancellation)?;
        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk = serde_json::from_str::<Value>(data)?;
        if emitter.message.response_id.is_empty()
            && let Some(id) = chunk
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
        {
            emitter.message.response_id = id.to_owned();
        }
        if let Some(usage) = chunk.get("usage") {
            apply_mistral_usage(&mut emitter.message.usage, usage);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            continue;
        };
        if let Some(reason) = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            saw_finish_reason = true;
            emitter.message.raw_stop_reason = reason.to_owned();
            let (stop_reason, error_message) = map_mistral_stop_reason(reason);
            emitter.message.stop_reason = stop_reason;
            emitter.message.error_message = error_message;
        }

        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            continue;
        };
        if let Some(content) = delta.get("content").filter(|content| !content.is_null()) {
            if let Some(text) = content.as_str() {
                if !text.is_empty() {
                    append_mistral_text(emitter, &mut text_index, &mut thinking_index, text)?;
                }
            } else if let Some(parts) = content.as_array() {
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("thinking") => {
                            let thinking = part
                                .get("thinking")
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .filter_map(|item| item.get("text").and_then(Value::as_str))
                                .collect::<String>();
                            if !thinking.is_empty() {
                                append_mistral_thinking(
                                    emitter,
                                    &mut text_index,
                                    &mut thinking_index,
                                    &thinking,
                                )?;
                            }
                        }
                        Some("text") => {
                            append_mistral_text(
                                emitter,
                                &mut text_index,
                                &mut thinking_index,
                                part.get("text").and_then(Value::as_str).unwrap_or_default(),
                            )?;
                        }
                        _ => {}
                    }
                }
            } else {
                return Err(ProviderAdapterError::Protocol(
                    "Mistral delta content was neither text nor a content array".to_owned(),
                ));
            }
        }

        let Some(raw_calls) = delta.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for raw_call in raw_calls {
            close_mistral_content(emitter, &mut text_index, &mut thinking_index)?;
            let index = raw_call.get("index").and_then(Value::as_i64).unwrap_or(0);
            let key = if let Some(key) = tool_call_keys_by_index.get(&index) {
                key.clone()
            } else {
                let mut id = raw_call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                if id.is_empty() || id == "null" {
                    id = derive_mistral_tool_call_id(&format!("toolcall:{index}"), 0);
                }
                let key = format!("{id}:{index}");
                tool_call_keys_by_index.insert(index, key.clone());
                key
            };
            let function = raw_call.get("function").and_then(Value::as_object);
            let name = function
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = raw_call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty() && *id != "null")
                .unwrap_or_else(|| key.split(':').next().unwrap_or_default());
            if !tool_calls.contains_key(&key) {
                let content_index = emitter.start_tool(id, name)?;
                tool_calls.insert(
                    key.clone(),
                    MistralToolCallState {
                        content_index,
                        arguments: stream::IncrementalJsonObjectParser::new(),
                    },
                );
                tool_call_order.push(key.clone());
            }
            let arguments = function
                .and_then(|function| function.get("arguments"))
                .filter(|arguments| !arguments.is_null())
                .map(mistral_argument_delta)
                .unwrap_or_default();
            let (content_index, preview) = {
                let state = tool_calls
                    .get_mut(&key)
                    .expect("Mistral tool-call state was just inserted");
                let reparsed = state.arguments.push(&arguments);
                let preview = reparsed.then(|| state.arguments.tool_arguments());
                (state.content_index, preview)
            };
            if let Some(preview) = preview {
                emitter.set_tool_arguments(content_index, preview)?;
            }
            emitter.tool_delta(content_index, &arguments)?;
        }
    }

    close_mistral_content(emitter, &mut text_index, &mut thinking_index)?;
    for key in tool_call_order {
        let state = tool_calls
            .get_mut(&key)
            .expect("Mistral tool-call order references a state");
        let content_index = state.content_index;
        let arguments = state.arguments.finish_tool_arguments();
        emitter.set_tool_arguments(content_index, arguments)?;
        emitter.end_tool(content_index)?;
    }
    if emitter.message.stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message.error_message.clone(),
        ));
    }
    if !saw_finish_reason {
        return Err(ProviderAdapterError::Protocol(
            "Mistral stream ended without a finish_reason".to_owned(),
        ));
    }
    Ok(())
}

fn ensure_not_cancelled(cancellation: &agent::CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(ProviderAdapterError::Cancelled)
    } else {
        Ok(())
    }
}

fn append_mistral_text(
    emitter: &mut MessageEmitter,
    text_index: &mut Option<usize>,
    thinking_index: &mut Option<usize>,
    delta: &str,
) -> Result<()> {
    let index = match *text_index {
        Some(index) => index,
        None => {
            close_mistral_content(emitter, text_index, thinking_index)?;
            let index = emitter.start_text("")?;
            *text_index = Some(index);
            index
        }
    };
    emitter.append_text(index, delta)
}

fn append_mistral_thinking(
    emitter: &mut MessageEmitter,
    text_index: &mut Option<usize>,
    thinking_index: &mut Option<usize>,
    delta: &str,
) -> Result<()> {
    let index = match *thinking_index {
        Some(index) => index,
        None => {
            close_mistral_content(emitter, text_index, thinking_index)?;
            let index = emitter.start_thinking("", "", false)?;
            *thinking_index = Some(index);
            index
        }
    };
    emitter.append_thinking(index, delta)
}

fn close_mistral_content(
    emitter: &mut MessageEmitter,
    text_index: &mut Option<usize>,
    thinking_index: &mut Option<usize>,
) -> Result<()> {
    if let Some(index) = text_index.take() {
        emitter.end_text(index)
    } else if let Some(index) = thinking_index.take() {
        emitter.end_thinking(index)
    } else {
        Ok(())
    }
}

fn mistral_argument_delta(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}

fn apply_mistral_usage(usage: &mut llm::Usage, raw: &Value) {
    let prompt_tokens = mistral_u64(raw, "prompt_tokens").unwrap_or(usage.input);
    let completion_tokens = mistral_u64(raw, "completion_tokens").unwrap_or(usage.output);
    let cached_tokens = raw
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
        .and_then(mistral_value_u64)
        .or_else(|| mistral_u64(raw, "num_cached_tokens"))
        .unwrap_or(usage.cache_read)
        .min(prompt_tokens);
    usage.input = prompt_tokens.saturating_sub(cached_tokens);
    usage.output = completion_tokens;
    usage.cache_read = cached_tokens;
    usage.cache_write = 0;
    usage.total_tokens = mistral_u64(raw, "total_tokens").unwrap_or_else(|| {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
    });
}

fn mistral_u64(value: &Value, name: &str) -> Option<u64> {
    value.get(name).and_then(mistral_value_u64)
}

fn mistral_value_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_i64()
            .filter(|value| *value >= 0)
            .map(|value| value as u64)
    })
}

fn map_mistral_stop_reason(reason: &str) -> (String, String) {
    match reason {
        "" | "stop" => (stream::STOP_STOP.to_owned(), String::new()),
        "length" | "model_length" => (stream::STOP_LENGTH.to_owned(), String::new()),
        "tool_calls" => (stream::STOP_TOOL_USE.to_owned(), String::new()),
        other => (
            stream::STOP_ERROR.to_owned(),
            format!("Provider stopped with: {other}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> llm::Model {
        llm::Model {
            id: "mistral-medium-latest".to_owned(),
            name: "Mistral Medium".to_owned(),
            api: "mistral-conversations".to_owned(),
            provider: "mistral".to_owned(),
            base_url: "https://api.mistral.ai".to_owned(),
            input: vec!["text".to_owned()],
            context_window: 128_000,
            max_tokens: 8_192,
            ..llm::Model::default()
        }
    }

    fn options() -> agent::RequestOptions {
        agent::RequestOptions {
            cancellation: agent::CancellationToken::default(),
            thinking_level: llm::THINKING_HIGH.to_owned(),
            thinking_budgets: None,
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            cache_retention: agent::CacheRetention::Short,
            session_id: "session-1".to_owned(),
            assistant_event_listener: None,
        }
    }

    #[test]
    fn request_maps_mistral_options_and_cache_retention() {
        let mut request_model = model();
        request_model.id = "mistral-small-latest".to_owned();
        request_model.reasoning = true;
        request_model.sampling_params = Some(json!({
            "max_tokens": 2_048,
            "top_p": 0.25,
        }));
        let context = llm::Context {
            system_prompt: "be brief".to_owned(),
            messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
            tools: vec![llm::Tool {
                name: "weather".to_owned(),
                description: "Look up weather".to_owned(),
                parameters: json!({"type": "object"}),
                constrained_sampling: Some(json!({
                    "type": "json_schema",
                    "strict": "require",
                })),
            }],
        };
        let mut request_options = options();
        request_options.temperature = Some(0.3);
        request_options.max_tokens = Some(100);
        request_options.tool_choice = Some(json!("required"));

        let request =
            build_mistral_request(&request_model, &context, &request_options).expect("request");
        assert_eq!(request["model"], "mistral-small-latest");
        assert_eq!(request["stream"], true);
        assert_eq!(request["max_tokens"], 2_048);
        assert_eq!(request["temperature"], 0.3);
        assert_eq!(request["tool_choice"], "required");
        assert_eq!(request["reasoning_effort"], "high");
        assert_eq!(request["prompt_cache_key"], "session-1");
        assert_eq!(request["top_p"], 0.25);
        assert_eq!(request["messages"][0]["role"], "system");
        assert_eq!(request["tools"][0]["function"]["strict"], true);

        request_options.cache_retention = agent::CacheRetention::None;
        request_options.tool_choice = Some(json!("unsupported"));
        let without_cache =
            build_mistral_request(&request_model, &context, &request_options).expect("request");
        assert!(without_cache.get("prompt_cache_key").is_none());
        assert!(without_cache.get("tool_choice").is_none());
    }

    #[test]
    fn replay_normalizes_cross_model_tool_ids_and_preserves_tool_result_linkage() {
        let request_model = model();
        let source_id = "call_not_accepted_by_mistral";
        let context = llm::Context {
            messages: vec![
                llm::Message::Assistant(Box::new(llm::AssistantMessage {
                    api: "openai-completions".to_owned(),
                    provider: "other".to_owned(),
                    model: "other-model".to_owned(),
                    stop_reason: stream::STOP_TOOL_USE.to_owned(),
                    content: vec![
                        llm::ContentBlock::Thinking(llm::ThinkingContent {
                            thinking: "consider options".to_owned(),
                            thinking_signature: "opaque".to_owned(),
                            ..llm::ThinkingContent::default()
                        }),
                        llm::ContentBlock::ToolCall(llm::ToolCall {
                            id: source_id.to_owned(),
                            name: "read".to_owned(),
                            ..llm::ToolCall::default()
                        }),
                    ],
                    timestamp: 1,
                    ..llm::AssistantMessage::default()
                })),
                llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                    tool_call_id: source_id.to_owned(),
                    tool_name: "read".to_owned(),
                    content: vec![llm::ContentBlock::text("ok")],
                    timestamp: 2,
                    ..llm::ToolResultMessage::default()
                })),
            ],
            ..llm::Context::default()
        };

        let request = build_mistral_request(&request_model, &context, &options()).expect("request");
        let tool_call_id = request["messages"][0]["tool_calls"][0]["id"]
            .as_str()
            .expect("normalized tool-call ID");
        assert_eq!(tool_call_id.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
        assert!(
            tool_call_id
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        );
        assert_eq!(request["messages"][1]["tool_call_id"], tool_call_id);
        assert_eq!(
            request["messages"][0]["content"][0],
            json!({"type": "text", "text": "consider options"})
        );
    }

    #[test]
    fn tool_call_id_normalization_is_stable_and_compatible() {
        assert_eq!(derive_mistral_tool_call_id("abc123def", 0), "abc123def");
        let derived = derive_mistral_tool_call_id("call_a_very_long_identifier", 0);
        assert_eq!(derived.len(), MISTRAL_TOOL_CALL_ID_LENGTH);
        assert!(
            derived
                .chars()
                .all(|character| character.is_ascii_alphanumeric())
        );
        assert_ne!(
            derive_mistral_tool_call_id("x", 0),
            derive_mistral_tool_call_id("x", 1)
        );

        let mut normalizer = MistralToolCallIdNormalizer::default();
        let first = normalizer.normalize("call_one");
        assert_eq!(normalizer.normalize("call_one"), first);
        assert_ne!(normalizer.normalize("call_two"), first);
    }

    #[test]
    fn tool_result_text_marks_errors_and_image_fallbacks() {
        assert_eq!(
            mistral_tool_result_text("output", false, false, true),
            "[tool error] output"
        );
        assert_eq!(
            mistral_tool_result_text("", false, false, false),
            "(no tool output)"
        );
        assert_eq!(
            mistral_tool_result_text("output", true, false, false),
            "output\n[tool image omitted: model does not support images]"
        );
    }

    #[test]
    fn stop_reason_and_usage_helpers_follow_mistral_rules() {
        assert_eq!(
            map_mistral_stop_reason("model_length").0,
            stream::STOP_LENGTH
        );
        assert_eq!(
            map_mistral_stop_reason("tool_calls").0,
            stream::STOP_TOOL_USE
        );
        assert_eq!(
            map_mistral_stop_reason("content_filter").0,
            stream::STOP_ERROR
        );

        let mut usage = llm::Usage::default();
        apply_mistral_usage(
            &mut usage,
            &json!({
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
                "num_cached_tokens": 999,
            }),
        );
        assert_eq!(usage.input, 0);
        assert_eq!(usage.cache_read, 10);
        assert_eq!(usage.cache_write, 0);
    }
}
