//! Core pi-compatible LLM data types.
//!
//! These types intentionally preserve the JSON field names used by existing
//! session files and provider payload adapters. Protocol-specific clients are
//! layered on top of this module as they are migrated.

use std::collections::BTreeMap;

use serde::{
    de::{Error as DeError, IntoDeserializer},
    Deserialize, Deserializer, Serialize, Serializer,
};
use serde_json::Value;

pub const THINKING_OFF: &str = "off";
pub const THINKING_MINIMAL: &str = "minimal";
pub const THINKING_LOW: &str = "low";
pub const THINKING_MEDIUM: &str = "medium";
pub const THINKING_HIGH: &str = "high";
pub const THINKING_XHIGH: &str = "xhigh";
pub const THINKING_MAX: &str = "max";

pub type Api = String;
pub type ProviderId = String;
pub type ThinkingLevel = String;
pub type ThinkingLevelMap = BTreeMap<String, Option<String>>;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ThinkingBudgets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub medium: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct TextContent {
    pub text: String,
    #[serde(
        rename = "textSignature",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub text_signature: String,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ThinkingContent {
    pub thinking: String,
    #[serde(
        rename = "thinkingSignature",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub thinking_signature: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub redacted: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ImageContent {
    pub data: String,
    #[serde(rename = "mimeType")]
    pub mime_type: String,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub arguments: BTreeMap<String, Value>,
    #[serde(
        rename = "thoughtSignature",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub thought_signature: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub namespace: String,
}

/// A content block is discriminated by pi's required `type` field.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text(TextContent),
    #[serde(rename = "thinking")]
    Thinking(ThinkingContent),
    #[serde(rename = "image")]
    Image(ImageContent),
    #[serde(rename = "toolCall")]
    ToolCall(ToolCall),
}

impl ContentBlock {
    pub fn text(value: impl Into<String>) -> Self {
        Self::Text(TextContent {
            text: value.into(),
            ..TextContent::default()
        })
    }

    pub fn plain_text(&self) -> Option<&str> {
        match self {
            Self::Text(content) => Some(&content.text),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl UserContent {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Blocks(_) => None,
        }
    }

    pub fn blocks(&self) -> Vec<ContentBlock> {
        match self {
            Self::Text(text) if text.is_empty() => Vec::new(),
            Self::Text(text) => vec![ContentBlock::text(text)],
            Self::Blocks(blocks) => blocks.clone(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UserMessage {
    #[serde(default = "user_role")]
    pub role: String,
    pub content: UserContent,
    pub timestamp: i64,
}

impl UserMessage {
    pub fn text(content: impl Into<String>, timestamp: i64) -> Self {
        Self {
            role: user_role(),
            content: UserContent::Text(content.into()),
            timestamp,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct UsageCost {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(rename = "cacheRead", default)]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite", default)]
    pub cache_write: f64,
    #[serde(default)]
    pub total: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Usage {
    #[serde(default)]
    pub input: u64,
    #[serde(default)]
    pub output: u64,
    #[serde(rename = "cacheRead", default)]
    pub cache_read: u64,
    #[serde(rename = "cacheWrite", default)]
    pub cache_write: u64,
    #[serde(
        rename = "cacheWrite1h",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_write_1h: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    #[serde(rename = "totalTokens", default)]
    pub total_tokens: u64,
    #[serde(default)]
    pub cost: UsageCost,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct DeferredHandle {
    pub provider: String,
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub api: String,
    pub id: String,
    #[serde(rename = "expiresAt", default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<i64>,
    #[serde(
        rename = "pollAfterMs",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub poll_after_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AssistantMessage {
    #[serde(default = "assistant_role")]
    pub role: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    pub api: Api,
    pub provider: ProviderId,
    pub model: String,
    #[serde(
        rename = "responseModel",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub response_model: String,
    #[serde(
        rename = "responseId",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub response_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<Value>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(
        rename = "stopReason",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub stop_reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred: Option<DeferredHandle>,
    #[serde(
        rename = "errorMessage",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub error_message: String,
    #[serde(
        rename = "rawStopReason",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub raw_stop_reason: String,
    #[serde(rename = "endTurn", default, skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
    pub timestamp: i64,
}

impl AssistantMessage {
    pub fn error(
        api: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
        message: impl Into<String>,
        timestamp: i64,
    ) -> Self {
        Self {
            role: assistant_role(),
            content: vec![ContentBlock::text("")],
            api: api.into(),
            provider: provider.into(),
            model: model.into(),
            stop_reason: "error".to_owned(),
            error_message: message.into(),
            timestamp,
            ..Self::default()
        }
    }
}

impl Default for AssistantMessage {
    fn default() -> Self {
        Self {
            role: assistant_role(),
            content: Vec::new(),
            api: String::new(),
            provider: String::new(),
            model: String::new(),
            response_model: String::new(),
            response_id: String::new(),
            diagnostics: Vec::new(),
            usage: Usage::default(),
            stop_reason: String::new(),
            deferred: None,
            error_message: String::new(),
            raw_stop_reason: String::new(),
            end_turn: None,
            timestamp: 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ToolResultMessage {
    #[serde(default = "tool_result_role")]
    pub role: String,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(
        rename = "addedToolNames",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub added_tool_names: Vec<String>,
    #[serde(rename = "isError")]
    pub is_error: bool,
    pub timestamp: i64,
}

impl Default for ToolResultMessage {
    fn default() -> Self {
        Self {
            role: tool_result_role(),
            tool_call_id: String::new(),
            tool_name: String::new(),
            content: Vec::new(),
            details: None,
            usage: None,
            added_tool_names: Vec::new(),
            is_error: false,
            timestamp: 0,
        }
    }
}

/// LLM transcript messages. Deserialization dispatches on `role`, mirroring
/// pi's strict behavior instead of accepting a structurally similar message
/// with the wrong role.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Self::User(_) => "user",
            Self::Assistant(_) => "assistant",
            Self::ToolResult(_) => "toolResult",
        }
    }

    pub fn text_preview(&self) -> String {
        match self {
            Self::User(message) => match &message.content {
                UserContent::Text(text) => text.clone(),
                UserContent::Blocks(blocks) => text_from_blocks(blocks),
            },
            Self::Assistant(message) => text_from_blocks(&message.content),
            Self::ToolResult(message) => text_from_blocks(&message.content),
        }
    }
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::User(message) => message.serialize(serializer),
            Self::Assistant(message) => message.serialize(serializer),
            Self::ToolResult(message) => message.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let role = value
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("llm: message has no role"))?;
        let decode = |value: Value| value.into_deserializer();
        match role {
            "user" => UserMessage::deserialize(decode(value))
                .map(Self::User)
                .map_err(D::Error::custom),
            "assistant" => AssistantMessage::deserialize(decode(value))
                .map(Box::new)
                .map(Self::Assistant)
                .map_err(D::Error::custom),
            "toolResult" => ToolResultMessage::deserialize(decode(value))
                .map(Box::new)
                .map(Self::ToolResult)
                .map_err(D::Error::custom),
            other => Err(D::Error::custom(format!(
                "llm: unknown message role {other:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(
        rename = "constrainedSampling",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub constrained_sampling: Option<Value>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Context {
    #[serde(
        rename = "systemPrompt",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub system_prompt: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelCostRates {
    #[serde(default)]
    pub input: f64,
    #[serde(default)]
    pub output: f64,
    #[serde(rename = "cacheRead", default)]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite", default)]
    pub cache_write: f64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelCostTier {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    #[serde(rename = "inputTokensAbove")]
    pub input_tokens_above: u64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelCost {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

/// Provider/model metadata from the generated catalog.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: ProviderId,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub reasoning: bool,
    #[serde(
        rename = "thinkingLevelMap",
        default,
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub thinking_level_map: ThinkingLevelMap,
    #[serde(default)]
    pub input: Vec<String>,
    #[serde(default)]
    pub cost: ModelCost,
    #[serde(rename = "contextWindow")]
    pub context_window: u64,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u64,
    #[serde(
        rename = "samplingParams",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub sampling_params: Option<Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<Value>,
}

impl Model {
    pub fn supports_images(&self) -> bool {
        self.input.iter().any(|input| input == "image")
    }
}

fn text_from_blocks(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(ContentBlock::plain_text)
        .collect::<Vec<_>>()
        .join(" ")
}

fn user_role() -> String {
    "user".to_owned()
}

fn assistant_role() -> String {
    "assistant".to_owned()
}

fn tool_result_role() -> String {
    "toolResult".to_owned()
}

fn is_false(value: &bool) -> bool {
    !value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_messages_keep_pi_json_shapes() {
        let message = Message::Assistant(Box::new(AssistantMessage {
            api: "anthropic-messages".to_owned(),
            provider: "anthropic".to_owned(),
            model: "claude-sonnet-5".to_owned(),
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "considering".to_owned(),
                    thinking_signature: "opaque".to_owned(),
                    ..ThinkingContent::default()
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "toolu_01".to_owned(),
                    name: "read".to_owned(),
                    arguments: BTreeMap::from([("path".to_owned(), json!("main.rs"))]),
                    ..ToolCall::default()
                }),
            ],
            stop_reason: "toolUse".to_owned(),
            timestamp: 2,
            ..AssistantMessage::default()
        }));

        let encoded = serde_json::to_value(&message).expect("encode");
        assert_eq!(encoded["role"], "assistant");
        assert_eq!(encoded["content"][0]["type"], "thinking");
        assert_eq!(encoded["content"][0]["thinkingSignature"], "opaque");
        assert_eq!(encoded["content"][1]["type"], "toolCall");
        assert_eq!(encoded["content"][1]["arguments"]["path"], "main.rs");

        let decoded: Message = serde_json::from_value(encoded).expect("decode");
        assert_eq!(decoded, message);
    }

    #[test]
    fn unknown_message_roles_are_rejected() {
        let error = serde_json::from_value::<Message>(json!({
            "role": "hookMessage",
            "content": "not a model message"
        }))
        .expect_err("unknown role should not silently enter LLM context");

        assert!(error.to_string().contains("unknown message role"));
    }

    #[test]
    fn model_catalog_metadata_keeps_optional_compat() {
        let model: Model = serde_json::from_value(json!({
            "id": "test",
            "name": "Test",
            "api": "openai-responses",
            "provider": "example",
            "baseUrl": "https://example.test/v1",
            "reasoning": true,
            "input": ["text", "image"],
            "cost": {},
            "contextWindow": 128000,
            "maxTokens": 4096,
            "compat": {"supportsReasoning": true}
        }))
        .expect("decode model");

        assert!(model.supports_images());
        assert_eq!(model.compat, Some(json!({"supportsReasoning": true})));
    }
}
