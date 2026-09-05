//! Amazon Bedrock ConverseStream protocol adapter.
//!
//! This module deliberately owns the Bedrock-specific wire protocol while
//! publishing only the crate's common [`crate::llm`] messages and
//! [`crate::stream`] events.  It includes SigV4, the AWS binary event-stream
//! codec, static AWS credential resolution, request conversion, and the
//! blocking transport required by the current Rust runtime.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fmt, fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use hmac::{Hmac, KeyInit, Mac};
use reqwest::{
    blocking::{Client, Response},
    header::{HeaderName, HeaderValue},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use url::{Host, Url};

use crate::{llm, stream};

/// The catalog API identifier for the Bedrock ConverseStream protocol.
pub const API_BEDROCK_CONVERSE_STREAM: &str = "bedrock-converse-stream";
/// SigV4's Bedrock Runtime service identifier.
pub const BEDROCK_SERVICE: &str = "bedrock";
/// Bedrock rejects blank required text blocks, so they use this value instead.
pub const BEDROCK_EMPTY_TEXT_PLACEHOLDER: &str = "<empty>";
/// Region used when no ARN, option, environment, endpoint, or profile supplies
/// one.
pub const BEDROCK_DEFAULT_REGION: &str = "us-east-1";
/// The documented AWS event-stream per-message limit.
pub const MAX_EVENT_STREAM_MESSAGE_SIZE: usize = 16 << 20;
/// Mirrors the common provider error-body bound used by the Go implementation.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4_000;
/// AWS documentation linked from data-retention validation failures.
pub const BEDROCK_DATA_RETENTION_DOCS_URL: &str =
    "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html";
/// The Anthropic beta identifier needed for non-adaptive interleaved thinking.
pub const ANTHROPIC_INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Compatibility fields supplied by a Bedrock model catalog entry.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
pub struct BedrockCompat {
    #[serde(rename = "supportsStrictMode", default)]
    pub supports_strict_mode: Option<bool>,
}

/// Prompt-cache retention accepted by Bedrock cache points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheRetention {
    None,
    Short,
    Long,
}

/// An optional Bedrock tool-choice directive.
///
/// `Unspecified` leaves Bedrock's choice behavior unchanged; it is different
/// from `Auto`, which explicitly serializes a Bedrock `auto` choice.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum BedrockToolChoice {
    #[default]
    Unspecified,
    Auto,
    Any,
    None,
    Tool(String),
}

/// Cooperative cancellation for a Bedrock request.
///
/// Cancellation interrupts retry sleeps promptly and is checked while parsing
/// event frames.  The current blocking `reqwest` dependency cannot interrupt a
/// socket read already in progress; configure `BedrockOptions::timeout` for a
/// bound on such a read.
#[derive(Clone, Default)]
pub struct BedrockCancellation {
    cancelled: Arc<AtomicBool>,
}

impl fmt::Debug for BedrockCancellation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BedrockCancellation")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl BedrockCancellation {
    /// Requests cancellation. Calling this more than once is harmless.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Reports whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Metadata published to a response callback before its body is consumed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BedrockResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
}

/// A callback that can inspect or replace the exact Bedrock JSON object before
/// it is serialized and signed. Returning `None` preserves the supplied value.
pub type BedrockPayloadHook =
    Arc<dyn Fn(Map<String, Value>, &llm::Model) -> Option<Map<String, Value>> + Send + Sync>;

/// A callback invoked after a successful HTTP response is received and before
/// its event stream is consumed.
pub type BedrockResponseHook = Arc<dyn Fn(BedrockResponse, &llm::Model) + Send + Sync>;

/// Bedrock-specific request options.
///
/// `environment` overlays process environment variables when a value is
/// non-empty. This intentionally matches the existing provider configuration
/// convention: an empty scoped value does not mask an ambient value.
#[derive(Clone)]
pub struct BedrockOptions {
    /// A catalog API key. For Bedrock it is treated as an API bearer token,
    /// except for the `<authenticated>` ambient-credential sentinel.
    pub api_key: Option<String>,
    pub on_payload: Option<BedrockPayloadHook>,
    pub on_response: Option<BedrockResponseHook>,
    /// Custom request headers. `None` is ignored, matching provider header
    /// suppression in the common options type.
    pub headers: BTreeMap<String, Option<String>>,
    pub timeout: Option<Duration>,
    /// Number of retries after the first HTTP attempt.
    pub max_retries: u32,
    /// Cap for a provider-directed retry delay. `None` means the common
    /// 60-second default; `Some(Duration::ZERO)` disables the cap.
    pub max_retry_delay: Option<Duration>,
    /// Optional deterministic retry jitter in `[0, 1]`; omitted uses a small
    /// time-derived jitter like the Go implementation's random jitter.
    pub retry_jitter: Option<f64>,
    pub temperature: Option<f64>,
    /// Applied last, so values here override named Bedrock fields.
    pub sampling_params: BTreeMap<String, Value>,
    pub max_tokens: Option<u64>,
    /// `None` resolves to `PI_CACHE_RETENTION=long` or the default `Short`.
    pub cache_retention: Option<CacheRetention>,
    pub environment: BTreeMap<String, String>,
    pub region: Option<String>,
    pub profile: Option<String>,
    pub tool_choice: BedrockToolChoice,
    pub reasoning: Option<llm::ThinkingLevel>,
    pub thinking_budgets: Option<llm::ThinkingBudgets>,
    pub interleaved_thinking: Option<bool>,
    pub thinking_display: Option<String>,
    pub request_metadata: BTreeMap<String, String>,
    pub bearer_token: Option<String>,
    pub compat: Option<BedrockCompat>,
    pub cancellation: Option<BedrockCancellation>,
}

impl Default for BedrockOptions {
    fn default() -> Self {
        Self {
            api_key: None,
            on_payload: None,
            on_response: None,
            headers: BTreeMap::new(),
            timeout: None,
            max_retries: 0,
            max_retry_delay: None,
            retry_jitter: None,
            temperature: None,
            sampling_params: BTreeMap::new(),
            max_tokens: None,
            cache_retention: None,
            environment: BTreeMap::new(),
            region: None,
            profile: None,
            tool_choice: BedrockToolChoice::Unspecified,
            reasoning: None,
            thinking_budgets: None,
            interleaved_thinking: None,
            thinking_display: None,
            request_metadata: BTreeMap::new(),
            bearer_token: None,
            compat: None,
            cancellation: None,
        }
    }
}

impl fmt::Debug for BedrockOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately omit api_key, bearer_token, scoped environment values,
        // custom headers, and callbacks because they can carry credentials.
        formatter
            .debug_struct("BedrockOptions")
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("max_retry_delay", &self.max_retry_delay)
            .field("retry_jitter", &self.retry_jitter)
            .field("temperature", &self.temperature)
            .field("sampling_param_names", &self.sampling_params.keys())
            .field("max_tokens", &self.max_tokens)
            .field("cache_retention", &self.cache_retention)
            .field("region", &self.region)
            .field("profile", &self.profile)
            .field("tool_choice", &self.tool_choice)
            .field("reasoning", &self.reasoning)
            .field("thinking_budgets", &self.thinking_budgets)
            .field("interleaved_thinking", &self.interleaved_thinking)
            .field("thinking_display", &self.thinking_display)
            .field("request_metadata_keys", &self.request_metadata.keys())
            .field("compat", &self.compat)
            .field("cancellation", &self.cancellation)
            .finish()
    }
}

/// Unified options accepted by [`stream_bedrock_simple`].
#[derive(Clone, Debug, Default)]
pub struct BedrockSimpleOptions {
    /// The common Bedrock request options.
    pub request: BedrockOptions,
    pub reasoning: Option<llm::ThinkingLevel>,
    pub thinking_budgets: Option<llm::ThinkingBudgets>,
}

/// Backwards-friendly spelling for callers that prefer the provider name first.
pub type SimpleBedrockOptions = BedrockSimpleOptions;

/// A static AWS access-key tuple resolved for SigV4.
///
/// This type intentionally does not implement `Debug` to prevent accidental
/// credential disclosure in logs.
#[derive(Clone, Eq, PartialEq)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: String,
    /// Non-secret credential origin suitable for diagnostics.
    pub source: String,
}

/// The resolved region and endpoint for one Bedrock request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BedrockTarget {
    pub region: String,
    pub endpoint: String,
}

/// A request representation used by the signer before it is handed to reqwest.
///
/// It keeps request construction testable without exposing reqwest-specific
/// types to the rest of the LLM runtime.
#[derive(Clone)]
pub struct BedrockHttpRequest {
    pub method: String,
    pub url: Url,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

/// One AWS binary event-stream frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventStreamMessage {
    pub headers: BTreeMap<String, String>,
    pub payload: Vec<u8>,
}

impl EventStreamMessage {
    pub fn event_type(&self) -> Option<&str> {
        self.headers.get(":event-type").map(String::as_str)
    }

    pub fn exception_type(&self) -> Option<&str> {
        self.headers.get(":exception-type").map(String::as_str)
    }

    pub fn message_type(&self) -> Option<&str> {
        self.headers.get(":message-type").map(String::as_str)
    }
}

/// Bounds enforced before event-stream framing allocates provider-controlled
/// memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventStreamLimits {
    pub max_stream_bytes: usize,
    pub max_message_size: usize,
}

impl Default for EventStreamLimits {
    fn default() -> Self {
        Self {
            max_stream_bytes: stream::MAX_PROVIDER_STREAM_BYTES,
            max_message_size: MAX_EVENT_STREAM_MESSAGE_SIZE,
        }
    }
}

/// AWS event-stream decoding failures.
#[derive(Debug)]
pub enum EventStreamError {
    Io(io::Error),
    TruncatedPrelude,
    PreludeCrcMismatch { got: u32, expected: u32 },
    ImplausibleLength(usize),
    HeadersLengthExceedsMessage { headers: usize, message: usize },
    TruncatedBody(io::Error),
    MessageCrcMismatch { got: u32, expected: u32 },
    TruncatedHeaderNameLength,
    TruncatedHeaderName,
    TruncatedHeaderValueType,
    TruncatedHeaderValueLength,
    TruncatedHeaderValue,
    HeaderBlockOverrun,
    UnknownHeaderValueType(u8),
    HeaderNameTooLong(usize),
    HeaderValueTooLong(usize),
}

impl fmt::Display for EventStreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "event stream: {error}"),
            Self::TruncatedPrelude => {
                formatter.write_str("event stream: truncated message prelude")
            }
            Self::PreludeCrcMismatch { got, expected } => write!(
                formatter,
                "event stream: prelude CRC mismatch (got {got:08x}, want {expected:08x})"
            ),
            Self::ImplausibleLength(length) => {
                write!(
                    formatter,
                    "event stream: implausible message length {length}"
                )
            }
            Self::HeadersLengthExceedsMessage { headers, message } => write!(
                formatter,
                "event stream: headers length {headers} exceeds message length {message}"
            ),
            Self::TruncatedBody(error) => {
                write!(formatter, "event stream: truncated message body: {error}")
            }
            Self::MessageCrcMismatch { got, expected } => write!(
                formatter,
                "event stream: message CRC mismatch (got {got:08x}, want {expected:08x})"
            ),
            Self::TruncatedHeaderNameLength => {
                formatter.write_str("event stream: truncated header name length")
            }
            Self::TruncatedHeaderName => formatter.write_str("event stream: truncated header name"),
            Self::TruncatedHeaderValueType => {
                formatter.write_str("event stream: truncated header value type")
            }
            Self::TruncatedHeaderValueLength => {
                formatter.write_str("event stream: truncated header value length")
            }
            Self::TruncatedHeaderValue => {
                formatter.write_str("event stream: truncated header value")
            }
            Self::HeaderBlockOverrun => {
                formatter.write_str("event stream: header block overran its bounds")
            }
            Self::UnknownHeaderValueType(value_type) => {
                write!(
                    formatter,
                    "event stream: unknown header value type {value_type}"
                )
            }
            Self::HeaderNameTooLong(length) => {
                write!(
                    formatter,
                    "event stream: header name is too long ({length} bytes)"
                )
            }
            Self::HeaderValueTooLong(length) => {
                write!(
                    formatter,
                    "event stream: header value is too long ({length} bytes)"
                )
            }
        }
    }
}

impl Error for EventStreamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) | Self::TruncatedBody(error) => Some(error),
            _ => None,
        }
    }
}

/// Errors surfaced by the Bedrock adapter.
#[derive(Debug)]
pub enum BedrockError {
    Aborted,
    Io(io::Error),
    Json(serde_json::Error),
    Url(url::ParseError),
    EventStream(EventStreamError),
    Provider(stream::ProviderError),
    RetryDelay(stream::RetryDelayError),
    Message(String),
}

impl BedrockError {
    /// Returns the HTTP-level error when this failure is retry-classifiable.
    pub fn provider_error(&self) -> Option<&stream::ProviderError> {
        match self {
            Self::Provider(error) => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for BedrockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aborted => formatter.write_str("request aborted"),
            Self::Io(error) => write!(formatter, "{error}"),
            Self::Json(error) => write!(formatter, "{error}"),
            Self::Url(error) => write!(formatter, "{error}"),
            Self::EventStream(error) => write!(formatter, "{error}"),
            Self::Provider(error) => write!(formatter, "{error}"),
            Self::RetryDelay(error) => write!(formatter, "{error}"),
            Self::Message(message) => formatter.write_str(message),
        }
    }
}

impl Error for BedrockError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Url(error) => Some(error),
            Self::EventStream(error) => Some(error),
            Self::Provider(error) => Some(error),
            Self::RetryDelay(error) => Some(error),
            Self::Aborted | Self::Message(_) => None,
        }
    }
}

impl From<io::Error> for BedrockError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for BedrockError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<url::ParseError> for BedrockError {
    fn from(error: url::ParseError) -> Self {
        Self::Url(error)
    }
}

impl From<EventStreamError> for BedrockError {
    fn from(error: EventStreamError) -> Self {
        Self::EventStream(error)
    }
}

impl From<stream::RetryDelayError> for BedrockError {
    fn from(error: stream::RetryDelayError) -> Self {
        Self::RetryDelay(error)
    }
}

/// Returns whether an error can be retried before a response stream begins.
pub fn is_retryable_bedrock_error(error: &BedrockError) -> bool {
    error
        .provider_error()
        .is_some_and(stream::is_retryable_provider_error)
}

/// Formats an adapter failure into the pi-compatible `errorMessage` string.
pub fn format_bedrock_stream_error(error: &BedrockError) -> String {
    let Some(provider) = error.provider_error() else {
        return error.to_string();
    };

    let body = truncate_error_text(provider.body.trim(), MAX_PROVIDER_ERROR_BODY_CHARS);
    let mut message = if provider.message.contains(&body) || provider.status == 0 || body.is_empty()
    {
        if provider.status != 0 {
            format!("{}: {}", provider.status, provider)
        } else {
            provider.to_string()
        }
    } else {
        format!("{}: {}", provider.status, body)
    };

    if let Some(raw) = serde_json::from_str::<Value>(&provider.body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/metadata/raw")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .filter(|raw| !raw.is_empty() && !message.contains(raw.as_str()))
    {
        message.push('\n');
        message.push_str(&raw);
    }
    message
}

/// Truncates text by Unicode scalar value, matching Go's rune-based provider
/// error truncation.
pub fn truncate_error_text(text: &str, maximum_chars: usize) -> String {
    let character_count = text.chars().count();
    if character_count <= maximum_chars {
        return text.to_owned();
    }
    let prefix: String = text.chars().take(maximum_chars).collect();
    format!(
        "{prefix}... [truncated {} chars]",
        character_count - maximum_chars
    )
}

// ---------------------------------------------------------------------------
// AWS event stream
// ---------------------------------------------------------------------------

struct LimitedReader<R> {
    reader: R,
    remaining: usize,
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || buffer.is_empty() {
            return Ok(0);
        }
        let permitted = buffer.len().min(self.remaining);
        let read = self.reader.read(&mut buffer[..permitted])?;
        self.remaining -= read;
        Ok(read)
    }
}

/// Decoder for AWS's binary `application/vnd.amazon.eventstream` response
/// format.
pub struct EventStreamReader<R> {
    reader: LimitedReader<R>,
    limits: EventStreamLimits,
}

impl<R: Read> EventStreamReader<R> {
    pub fn new(reader: R) -> Self {
        Self::with_limits(reader, EventStreamLimits::default())
    }

    pub fn with_limits(reader: R, limits: EventStreamLimits) -> Self {
        Self {
            reader: LimitedReader {
                reader,
                remaining: limits.max_stream_bytes,
            },
            limits,
        }
    }

    /// Decodes one frame. `Ok(None)` represents a clean end of stream.
    pub fn next_message(&mut self) -> Result<Option<EventStreamMessage>, EventStreamError> {
        let mut prelude = [0_u8; 12];
        match read_exact_or_eof(&mut self.reader, &mut prelude) {
            Ok(false) => return Ok(None),
            Ok(true) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Err(EventStreamError::TruncatedPrelude);
            }
            Err(error) => return Err(EventStreamError::Io(error)),
        }

        let total_length =
            u32::from_be_bytes(prelude[0..4].try_into().expect("four bytes")) as usize;
        let headers_length =
            u32::from_be_bytes(prelude[4..8].try_into().expect("four bytes")) as usize;
        let expected_prelude_crc =
            u32::from_be_bytes(prelude[8..12].try_into().expect("four bytes"));
        let actual_prelude_crc = crc32_ieee(&prelude[..8]);
        if actual_prelude_crc != expected_prelude_crc {
            return Err(EventStreamError::PreludeCrcMismatch {
                got: actual_prelude_crc,
                expected: expected_prelude_crc,
            });
        }
        if !(16..=self.limits.max_message_size).contains(&total_length) {
            return Err(EventStreamError::ImplausibleLength(total_length));
        }
        if headers_length.saturating_add(16) > total_length {
            return Err(EventStreamError::HeadersLengthExceedsMessage {
                headers: headers_length,
                message: total_length,
            });
        }

        let mut remaining = vec![0_u8; total_length - prelude.len()];
        if let Err(error) = self.reader.read_exact(&mut remaining) {
            return Err(EventStreamError::TruncatedBody(error));
        }
        let expected_message_crc = u32::from_be_bytes(
            remaining[remaining.len() - 4..]
                .try_into()
                .expect("four bytes"),
        );
        let mut actual_message_crc = crc32_ieee(&prelude);
        actual_message_crc =
            crc32_ieee_update(actual_message_crc, &remaining[..remaining.len() - 4]);
        if actual_message_crc != expected_message_crc {
            return Err(EventStreamError::MessageCrcMismatch {
                got: actual_message_crc,
                expected: expected_message_crc,
            });
        }

        let headers = parse_event_stream_headers(&remaining[..headers_length])?;
        let payload = remaining[headers_length..remaining.len() - 4].to_vec();
        Ok(Some(EventStreamMessage { headers, payload }))
    }
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buffer: &mut [u8]) -> io::Result<bool> {
    let mut offset = 0;
    while offset < buffer.len() {
        let read = reader.read(&mut buffer[offset..])?;
        if read == 0 {
            return if offset == 0 {
                Ok(false)
            } else {
                Err(io::Error::from(io::ErrorKind::UnexpectedEof))
            };
        }
        offset += read;
    }
    Ok(true)
}

/// Decodes the string-valued headers in an AWS event-stream header block.
///
/// Bedrock's meaningful headers are strings. Other valid AWS header types are
/// skipped so an unrelated typed header cannot desynchronize framing.
pub fn parse_event_stream_headers(
    data: &[u8],
) -> Result<BTreeMap<String, String>, EventStreamError> {
    const BOOL_TRUE: u8 = 0;
    const BOOL_FALSE: u8 = 1;
    const BYTE: u8 = 2;
    const SHORT: u8 = 3;
    const INTEGER: u8 = 4;
    const LONG: u8 = 5;
    const BYTE_ARRAY: u8 = 6;
    const STRING: u8 = 7;
    const TIMESTAMP: u8 = 8;
    const UUID: u8 = 9;

    let mut headers = BTreeMap::new();
    let mut offset = 0;
    while offset < data.len() {
        let Some(&name_length) = data.get(offset) else {
            return Err(EventStreamError::TruncatedHeaderNameLength);
        };
        offset += 1;
        let name_length = usize::from(name_length);
        let Some(name_bytes) = data.get(offset..offset.saturating_add(name_length)) else {
            return Err(EventStreamError::TruncatedHeaderName);
        };
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        offset += name_length;

        let Some(&value_type) = data.get(offset) else {
            return Err(EventStreamError::TruncatedHeaderValueType);
        };
        offset += 1;
        match value_type {
            BOOL_TRUE | BOOL_FALSE => {}
            BYTE => advance_header_offset(data, &mut offset, 1)?,
            SHORT => advance_header_offset(data, &mut offset, 2)?,
            INTEGER => advance_header_offset(data, &mut offset, 4)?,
            LONG | TIMESTAMP => advance_header_offset(data, &mut offset, 8)?,
            UUID => advance_header_offset(data, &mut offset, 16)?,
            BYTE_ARRAY | STRING => {
                let Some(length_bytes) = data.get(offset..offset.saturating_add(2)) else {
                    return Err(EventStreamError::TruncatedHeaderValueLength);
                };
                let value_length = usize::from(u16::from_be_bytes(
                    length_bytes.try_into().expect("two bytes"),
                ));
                offset += 2;
                let Some(value_bytes) = data.get(offset..offset.saturating_add(value_length))
                else {
                    return Err(EventStreamError::TruncatedHeaderValue);
                };
                if value_type == STRING {
                    headers.insert(name, String::from_utf8_lossy(value_bytes).into_owned());
                }
                offset += value_length;
            }
            unknown => return Err(EventStreamError::UnknownHeaderValueType(unknown)),
        }
    }
    Ok(headers)
}

fn advance_header_offset(
    data: &[u8],
    offset: &mut usize,
    amount: usize,
) -> Result<(), EventStreamError> {
    let end = offset.saturating_add(amount);
    if end > data.len() {
        return Err(EventStreamError::HeaderBlockOverrun);
    }
    *offset = end;
    Ok(())
}

/// Encodes an AWS event-stream frame. It is useful for local fixtures and
/// protocol tests; Bedrock requests themselves are regular JSON POSTs.
pub fn encode_event_stream_message(
    headers: &BTreeMap<String, String>,
    payload: &[u8],
) -> Result<Vec<u8>, EventStreamError> {
    let mut encoded_headers = Vec::new();
    for (name, value) in headers {
        if name.len() > u8::MAX as usize {
            return Err(EventStreamError::HeaderNameTooLong(name.len()));
        }
        if value.len() > u16::MAX as usize {
            return Err(EventStreamError::HeaderValueTooLong(value.len()));
        }
        encoded_headers.push(name.len() as u8);
        encoded_headers.extend_from_slice(name.as_bytes());
        encoded_headers.push(7); // string
        encoded_headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
        encoded_headers.extend_from_slice(value.as_bytes());
    }

    let total_length = 16_usize
        .checked_add(encoded_headers.len())
        .and_then(|length| length.checked_add(payload.len()))
        .ok_or_else(|| EventStreamError::ImplausibleLength(usize::MAX))?;
    if total_length > u32::MAX as usize {
        return Err(EventStreamError::ImplausibleLength(total_length));
    }

    let mut message = Vec::with_capacity(total_length);
    message.extend_from_slice(&(total_length as u32).to_be_bytes());
    message.extend_from_slice(&(encoded_headers.len() as u32).to_be_bytes());
    message.extend_from_slice(&crc32_ieee(&message[..8]).to_be_bytes());
    message.extend_from_slice(&encoded_headers);
    message.extend_from_slice(payload);
    message.extend_from_slice(&crc32_ieee(&message).to_be_bytes());
    Ok(message)
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    crc32_ieee_update(0, bytes)
}

/// Updates an IEEE CRC-32 with a previously finalized CRC, matching Go's
/// `crc32.Update` behavior.
fn crc32_ieee_update(previous: u32, bytes: &[u8]) -> u32 {
    let mut crc = !previous;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// AWS credentials and Signature Version 4
// ---------------------------------------------------------------------------

/// Resolves a scoped provider environment value, then falls back to the
/// process environment when the scoped value is absent or empty.
pub fn provider_env_value(environment: &BTreeMap<String, String>, name: &str) -> String {
    environment
        .get(name)
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| env::var(name).ok().filter(|value| !value.is_empty()))
        .unwrap_or_default()
}

/// Selects the AWS profile that should take precedence over ambient keys.
pub fn resolve_bedrock_profile(options: &BedrockOptions) -> Option<String> {
    if let Some(profile) = options
        .profile
        .as_deref()
        .filter(|profile| !profile.is_empty())
    {
        return Some(profile.to_owned());
    }
    if let Some(profile) = options
        .environment
        .get("AWS_PROFILE")
        .filter(|profile| !profile.is_empty())
    {
        return Some(profile.clone());
    }
    if options
        .environment
        .get("AWS_ACCESS_KEY_ID")
        .is_some_and(|value| !value.is_empty())
        && options
            .environment
            .get("AWS_SECRET_ACCESS_KEY")
            .is_some_and(|value| !value.is_empty())
    {
        return None;
    }
    env::var("AWS_PROFILE")
        .ok()
        .filter(|profile| !profile.is_empty())
}

fn shared_credentials_path(environment: &BTreeMap<String, String>) -> Option<PathBuf> {
    if let value @ Some(_) = environment
        .get("AWS_SHARED_CREDENTIALS_FILE")
        .filter(|path| !path.is_empty())
        .cloned()
    {
        return value.map(PathBuf::from);
    }
    if let Ok(path) = env::var("AWS_SHARED_CREDENTIALS_FILE") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    home_directory().map(|home| home.join(".aws").join("credentials"))
}

fn shared_config_path(environment: &BTreeMap<String, String>) -> Option<PathBuf> {
    if let value @ Some(_) = environment
        .get("AWS_CONFIG_FILE")
        .filter(|path| !path.is_empty())
        .cloned()
    {
        return value.map(PathBuf::from);
    }
    if let Ok(path) = env::var("AWS_CONFIG_FILE") {
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    home_directory().map(|home| home.join(".aws").join("config"))
}

fn home_directory() -> Option<PathBuf> {
    env::var_os("HOME").map(PathBuf::from)
}

/// Reads one AWS INI section. Keys are normalized to ASCII lowercase.
pub fn read_aws_ini_section(
    path: impl AsRef<Path>,
    section: &str,
) -> Result<BTreeMap<String, String>, BedrockError> {
    let content = fs::read_to_string(path.as_ref())?;
    let mut values = BTreeMap::new();
    let mut in_section = false;
    let mut found_section = false;
    for source_line in content.lines() {
        let line = source_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_section = line[1..line.len() - 1].trim() == section;
            found_section |= in_section;
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
    }
    if !found_section {
        return Err(BedrockError::Message(format!(
            "section {section:?} not found in {}",
            path.as_ref().display()
        )));
    }
    Ok(values)
}

/// Resolves static AWS credentials. An explicit profile wins over environment
/// keys; without a profile, scoped/process keys win over the default profile.
pub fn resolve_aws_credentials(
    profile: Option<&str>,
    environment: &BTreeMap<String, String>,
) -> Result<AwsCredentials, BedrockError> {
    if let Some(profile) = profile.filter(|profile| !profile.is_empty()) {
        return credentials_from_profile(profile, environment);
    }

    let access_key_id = provider_env_value(environment, "AWS_ACCESS_KEY_ID");
    let secret_access_key = provider_env_value(environment, "AWS_SECRET_ACCESS_KEY");
    if !access_key_id.is_empty() && !secret_access_key.is_empty() {
        return Ok(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token: provider_env_value(environment, "AWS_SESSION_TOKEN"),
            source: "AWS_ACCESS_KEY_ID".to_owned(),
        });
    }

    if let Ok(credentials) = credentials_from_profile("default", environment) {
        return Ok(credentials);
    }
    Err(BedrockError::Message(
        "no AWS credentials found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, configure a profile, or set AWS_BEARER_TOKEN_BEDROCK"
            .to_owned(),
    ))
}

fn credentials_from_profile(
    profile: &str,
    environment: &BTreeMap<String, String>,
) -> Result<AwsCredentials, BedrockError> {
    let mut candidates = Vec::new();
    if let Some(path) = shared_credentials_path(environment) {
        candidates.push((path, profile.to_owned()));
    }
    if let Some(path) = shared_config_path(environment) {
        candidates.push((path.clone(), profile.to_owned()));
        candidates.push((path, format!("profile {profile}")));
    }

    for (path, section) in candidates {
        let Ok(values) = read_aws_ini_section(&path, &section) else {
            continue;
        };
        let Some(access_key_id) = values
            .get("aws_access_key_id")
            .filter(|value| !value.is_empty())
            .cloned()
        else {
            continue;
        };
        let Some(secret_access_key) = values
            .get("aws_secret_access_key")
            .filter(|value| !value.is_empty())
            .cloned()
        else {
            continue;
        };
        return Ok(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token: values.get("aws_session_token").cloned().unwrap_or_default(),
            source: format!("profile {profile}"),
        });
    }
    Err(BedrockError::Message(format!(
        "no credentials for AWS profile {profile:?}"
    )))
}

/// Reads the configured region for a profile, defaulting an empty profile name
/// to the standard `default` profile.
pub fn region_from_profile(
    profile: Option<&str>,
    environment: &BTreeMap<String, String>,
) -> String {
    let profile = profile
        .filter(|profile| !profile.is_empty())
        .unwrap_or("default");
    let mut candidates = Vec::new();
    if let Some(path) = shared_config_path(environment) {
        candidates.push((path.clone(), format!("profile {profile}")));
        candidates.push((path, profile.to_owned()));
    }
    if let Some(path) = shared_credentials_path(environment) {
        candidates.push((path, profile.to_owned()));
    }
    for (path, section) in candidates {
        if let Ok(values) = read_aws_ini_section(path, &section) {
            if let Some(region) = values.get("region").filter(|region| !region.is_empty()) {
                return region.clone();
            }
        }
    }
    String::new()
}

/// Percent-encodes a string using AWS's SigV4 URI encoding rules.
pub fn aws_uri_encode(value: &str, encode_slash: bool) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        let unreserved = byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~');
        if unreserved || (byte == b'/' && !encode_slash) {
            output.push(byte as char);
        } else {
            output.push('%');
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    output
}

/// Builds the SigV4 canonical URI from an already escaped wire path.
pub fn aws_canonical_uri(escaped_path: &str) -> String {
    if escaped_path.is_empty() {
        "/".to_owned()
    } else {
        aws_uri_encode(escaped_path, false)
    }
}

/// Builds a canonical SigV4 query from a URL's decoded query pairs.
pub fn aws_canonical_query(url: &Url) -> String {
    let mut values = BTreeMap::<String, Vec<String>>::new();
    for (key, value) in url.query_pairs() {
        values
            .entry(key.into_owned())
            .or_default()
            .push(value.into_owned());
    }
    let mut parts = Vec::new();
    for (key, mut entries) in values {
        entries.sort_unstable();
        let encoded_key = aws_uri_encode(&key, true);
        for value in entries {
            parts.push(format!("{encoded_key}={}", aws_uri_encode(&value, true)));
        }
    }
    parts.join("&")
}

/// Builds canonical headers and the signed-header list for SigV4.
pub fn aws_canonical_headers(headers: &BTreeMap<String, String>, host: &str) -> (String, String) {
    let mut values = BTreeMap::<String, Vec<String>>::new();
    values.insert("host".to_owned(), vec![collapse_whitespace(host)]);
    for (name, value) in headers {
        let lower = name.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "authorization" | "content-length" | "user-agent"
        ) {
            continue;
        }
        values.insert(lower, vec![collapse_whitespace(value)]);
    }

    let signed_headers = values.keys().cloned().collect::<Vec<_>>().join(";");
    let canonical_headers = values
        .into_iter()
        .map(|(name, values)| format!("{name}:{}\n", values.join(",")))
        .collect();
    (canonical_headers, signed_headers)
}

/// Trims a header value and collapses every internal run of Unicode whitespace.
pub fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Signs a request in place with AWS Signature Version 4.
pub fn sign_aws_request(
    request: &mut BedrockHttpRequest,
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
    now: OffsetDateTime,
) -> Result<(), BedrockError> {
    if credentials.access_key_id.is_empty() || credentials.secret_access_key.is_empty() {
        return Err(BedrockError::Message(
            "no AWS credentials to sign with".to_owned(),
        ));
    }
    let now = now.to_offset(time::UtcOffset::UTC);
    let date_stamp = format!("{:04}{:02}{:02}", now.year(), now.month() as u8, now.day());
    let amz_date = format!(
        "{date_stamp}T{:02}{:02}{:02}Z",
        now.hour(),
        now.minute(),
        now.second()
    );
    let payload_hash = sha256_hex(&request.body);

    set_header(&mut request.headers, "X-Amz-Date", amz_date.clone());
    set_header(
        &mut request.headers,
        "X-Amz-Content-Sha256",
        payload_hash.clone(),
    );
    if !credentials.session_token.is_empty() {
        set_header(
            &mut request.headers,
            "X-Amz-Security-Token",
            credentials.session_token.clone(),
        );
    }

    let host = url_host_with_port(&request.url)?;
    let (canonical_headers, signed_headers) = aws_canonical_headers(&request.headers, &host);
    let canonical_request = [
        request.method.as_str(),
        &aws_canonical_uri(request.url.path()),
        &aws_canonical_query(&request.url),
        &canonical_headers,
        &signed_headers,
        &payload_hash,
    ]
    .join("\n");
    let credential_scope = format!("{date_stamp}/{region}/{service}/aws4_request");
    let string_to_sign = [
        "AWS4-HMAC-SHA256",
        &amz_date,
        &credential_scope,
        &sha256_hex(canonical_request.as_bytes()),
    ]
    .join("\n");
    let signing_key =
        derive_signing_key(&credentials.secret_access_key, &date_stamp, region, service);
    let signature = hex_encode(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
    set_header(
        &mut request.headers,
        "Authorization",
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
            credentials.access_key_id
        ),
    );
    Ok(())
}

fn url_host_with_port(url: &Url) -> Result<String, BedrockError> {
    let mut host = match url.host() {
        Some(Host::Ipv6(address)) => format!("[{address}]"),
        Some(host) => host.to_string(),
        None => {
            return Err(BedrockError::Message(
                "AWS request URL has no host".to_owned(),
            ));
        }
    };
    if let Some(port) = url.port() {
        host.push(':');
        host.push_str(&port.to_string());
    }
    Ok(host)
}

fn set_header(headers: &mut BTreeMap<String, String>, name: &str, value: String) {
    headers.retain(|existing, _| !existing.eq_ignore_ascii_case(name));
    headers.insert(name.to_owned(), value);
}

fn derive_signing_key(secret: &str, date_stamp: &str, region: &str, service: &str) -> Vec<u8> {
    let date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date_stamp.as_bytes());
    let region = hmac_sha256(&date, region.as_bytes());
    let service = hmac_sha256(&region, service.as_bytes());
    hmac_sha256(&service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts arbitrary key lengths");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256_hex(data: &[u8]) -> String {
    hex_encode(&Sha256::digest(data))
}

fn hex_encode(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(data.len() * 2);
    for byte in data {
        text.push(HEX[(byte >> 4) as usize] as char);
        text.push(HEX[(byte & 0x0f) as usize] as char);
    }
    text
}

// ---------------------------------------------------------------------------
// Endpoint and model capability resolution
// ---------------------------------------------------------------------------

/// Resolves a Bedrock endpoint and signing region with the same precedence as
/// the Go protocol: ARN region, explicit/configured region, standard endpoint,
/// profile, then `us-east-1`.
pub fn resolve_bedrock_target(model: &llm::Model, options: &BedrockOptions) -> BedrockTarget {
    let profile = resolve_bedrock_profile(options);
    let configured_region = configured_bedrock_region(options);
    let endpoint_region = standard_bedrock_endpoint_region(&model.base_url);
    let use_explicit_endpoint =
        endpoint_region.is_none() || (configured_region.is_empty() && profile.is_none());

    let region = arn_bedrock_region(&model.id)
        .or_else(|| (!configured_region.is_empty()).then_some(configured_region.clone()))
        .or_else(|| {
            (use_explicit_endpoint)
                .then_some(endpoint_region.clone())
                .flatten()
        })
        .unwrap_or_else(|| {
            let profile_region = region_from_profile(profile.as_deref(), &options.environment);
            if profile_region.is_empty() {
                BEDROCK_DEFAULT_REGION.to_owned()
            } else {
                profile_region
            }
        });

    let endpoint = if use_explicit_endpoint && !model.base_url.is_empty() {
        model.base_url.trim_end_matches('/').to_owned()
    } else {
        format!("https://bedrock-runtime.{region}.amazonaws.com")
    };
    BedrockTarget { region, endpoint }
}

fn configured_bedrock_region(options: &BedrockOptions) -> String {
    options
        .region
        .as_deref()
        .filter(|region| !region.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| {
            let region = provider_env_value(&options.environment, "AWS_REGION");
            if region.is_empty() {
                provider_env_value(&options.environment, "AWS_DEFAULT_REGION")
            } else {
                region
            }
        })
}

/// Extracts a region from a standard Bedrock Runtime endpoint. Custom endpoints
/// deliberately return `None`.
pub fn standard_bedrock_endpoint_region(base_url: &str) -> Option<String> {
    let parsed = Url::parse(base_url).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    let prefix = ["bedrock-runtime.", "bedrock-runtime-fips."]
        .iter()
        .find_map(|prefix| host.strip_prefix(prefix))?;
    let region = prefix
        .strip_suffix(".amazonaws.com")
        .or_else(|| prefix.strip_suffix(".amazonaws.com.cn"))?;
    (!region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'))
    .then_some(region.to_owned())
}

fn arn_bedrock_region(model_id: &str) -> Option<String> {
    let parts: Vec<_> = model_id.split(':').collect();
    let [arn, partition, service, region, _account, ..] = parts.as_slice() else {
        return None;
    };
    let valid_partition = *partition == "aws"
        || partition.strip_prefix("aws-").is_some_and(|suffix| {
            !suffix.is_empty()
                && suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        });
    if *arn != "arn"
        || !valid_partition
        || !partition
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || *service != "bedrock"
        || region.is_empty()
        || !region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return None;
    }
    Some((*region).to_owned())
}

fn model_match_candidates(model: &llm::Model) -> Vec<String> {
    [model.id.as_str(), model.name.as_str()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .flat_map(|value| {
            let lower = value.to_ascii_lowercase();
            let normalized = normalize_bedrock_candidate(&lower);
            [lower, normalized]
        })
        .collect()
}

fn normalize_bedrock_candidate(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut previous_separator = false;
    for character in value.chars() {
        let separator = character.is_whitespace() || matches!(character, '_' | '.' | ':');
        if separator {
            if !previous_separator {
                normalized.push('-');
            }
        } else {
            normalized.push(character);
        }
        previous_separator = separator;
    }
    normalized
}

fn model_candidates_contain(model: &llm::Model, needles: &[&str]) -> bool {
    model_match_candidates(model)
        .iter()
        .any(|candidate| needles.iter().any(|needle| candidate.contains(needle)))
}

/// Returns whether a model is an Anthropic Claude Bedrock model.
pub fn is_anthropic_claude_bedrock_model(model: &llm::Model) -> bool {
    let id = model.id.to_ascii_lowercase();
    let name = model.name.to_ascii_lowercase();
    id.contains("anthropic.claude")
        || id.contains("anthropic/claude")
        || name.contains("anthropic.claude")
        || name.contains("anthropic/claude")
        || name.contains("claude")
}

/// Returns whether the model uses Bedrock's adaptive thinking configuration.
pub fn bedrock_supports_adaptive_thinking(model: &llm::Model) -> bool {
    model_candidates_contain(
        model,
        &[
            "opus-4-6",
            "opus-4-7",
            "opus-4-8",
            "opus-5",
            "sonnet-4-6",
            "sonnet-5",
            "fable-5",
        ],
    )
}

/// Returns whether `xhigh` is a native Bedrock effort level for a model.
pub fn bedrock_supports_native_xhigh(model: &llm::Model) -> bool {
    model_candidates_contain(
        model,
        &["opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5"],
    )
}

/// Returns whether explicit prompt cache points are supported.
pub fn bedrock_supports_prompt_caching(
    model: &llm::Model,
    environment: &BTreeMap<String, String>,
) -> bool {
    if !model_candidates_contain(model, &["claude"]) {
        return provider_env_value(environment, "AWS_BEDROCK_FORCE_CACHE") == "1";
    }
    model_candidates_contain(model, &["fable-5", "opus-5", "sonnet-5"])
        || model_candidates_contain(model, &["-4-"])
        || model_candidates_contain(model, &["claude-3-7-sonnet"])
        || model_candidates_contain(model, &["claude-3-5-haiku"])
}

fn map_bedrock_thinking_level_to_effort(model: &llm::Model, level: &str) -> String {
    if level == llm::THINKING_XHIGH && bedrock_supports_native_xhigh(model) {
        return llm::THINKING_XHIGH.to_owned();
    }
    if let Some(Some(mapped)) = model.thinking_level_map.get(level) {
        return mapped.clone();
    }
    match level {
        llm::THINKING_MINIMAL | llm::THINKING_LOW => "low".to_owned(),
        llm::THINKING_MEDIUM => "medium".to_owned(),
        _ => "high".to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Bedrock request conversion
// ---------------------------------------------------------------------------

/// Converts a pi-compatible transcript into the JSON object accepted by
/// Bedrock's ConverseStream endpoint.
pub fn build_bedrock_params(
    model: &llm::Model,
    context: &llm::Context,
    options: &BedrockOptions,
) -> Result<Map<String, Value>, BedrockError> {
    let cache_retention = resolve_cache_retention(options);
    let messages = convert_bedrock_messages(model, context, cache_retention, &options.environment)?;
    let mut params = Map::new();
    params.insert("messages".to_owned(), Value::Array(messages));

    if let Some(system) = build_bedrock_system(
        &context.system_prompt,
        model,
        cache_retention,
        &options.environment,
    ) {
        params.insert("system".to_owned(), Value::Array(system));
    }

    let mut inference = Map::new();
    let max_tokens = options.max_tokens.filter(|value| *value != 0).or_else(|| {
        is_anthropic_claude_bedrock_model(model)
            .then_some(model.max_tokens)
            .filter(|value| *value != 0)
    });
    if let Some(max_tokens) = max_tokens {
        inference.insert("maxTokens".to_owned(), Value::from(max_tokens));
    }
    if let Some(temperature) = options.temperature {
        inference.insert("temperature".to_owned(), Value::from(temperature));
    }
    if !inference.is_empty() {
        params.insert("inferenceConfig".to_owned(), Value::Object(inference));
    }

    let compat = options.compat.clone().unwrap_or_else(|| {
        model
            .compat
            .as_ref()
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default()
    });
    let tool_config = convert_bedrock_tool_config(
        &context.tools,
        &options.tool_choice,
        compat.supports_strict_mode.unwrap_or(false),
    )?;
    if let Some(tool_config) = tool_config {
        params.insert("toolConfig".to_owned(), Value::Object(tool_config));
    }

    if let Some(fields) = bedrock_additional_model_request_fields(model, options) {
        params.insert(
            "additionalModelRequestFields".to_owned(),
            Value::Object(fields),
        );
    }
    if !options.request_metadata.is_empty() {
        params.insert(
            "requestMetadata".to_owned(),
            Value::Object(string_map_to_json(&options.request_metadata)),
        );
    }

    // Sampling parameters are applied last by design.
    for (key, value) in &options.sampling_params {
        params.insert(key.clone(), value.clone());
    }
    Ok(params)
}

fn resolve_cache_retention(options: &BedrockOptions) -> CacheRetention {
    options.cache_retention.unwrap_or_else(|| {
        if provider_env_value(&options.environment, "PI_CACHE_RETENTION") == "long" {
            CacheRetention::Long
        } else {
            CacheRetention::Short
        }
    })
}

fn build_bedrock_system(
    system_prompt: &str,
    model: &llm::Model,
    cache_retention: CacheRetention,
    environment: &BTreeMap<String, String>,
) -> Option<Vec<Value>> {
    if system_prompt.is_empty() {
        return None;
    }
    let mut blocks = vec![json!({"text": sanitize_surrogates(system_prompt)})];
    if cache_retention != CacheRetention::None
        && bedrock_supports_prompt_caching(model, environment)
    {
        blocks.push(bedrock_cache_point(cache_retention));
    }
    Some(blocks)
}

fn bedrock_cache_point(cache_retention: CacheRetention) -> Value {
    let mut point = Map::from_iter([("type".to_owned(), Value::String("default".to_owned()))]);
    if cache_retention == CacheRetention::Long {
        point.insert("ttl".to_owned(), Value::String("1h".to_owned()));
    }
    json!({"cachePoint": point})
}

fn convert_bedrock_messages(
    model: &llm::Model,
    context: &llm::Context,
    cache_retention: CacheRetention,
    environment: &BTreeMap<String, String>,
) -> Result<Vec<Value>, BedrockError> {
    let transformed = transform_bedrock_messages(&context.messages, model);
    let mut result = Vec::new();
    let mut index = 0;
    while index < transformed.len() {
        match &transformed[index] {
            llm::Message::User(message) => {
                let mut content = Vec::new();
                match &message.content {
                    llm::UserContent::Text(text) => content.push(bedrock_required_text_block(text)),
                    llm::UserContent::Blocks(blocks) => {
                        for block in blocks {
                            match block {
                                llm::ContentBlock::Text(text) => {
                                    if let Some(text) = bedrock_text_block(&text.text) {
                                        content.push(text);
                                    }
                                }
                                llm::ContentBlock::Image(image) => {
                                    content
                                        .push(bedrock_image_block(&image.mime_type, &image.data)?);
                                }
                                llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => {
                                }
                            }
                        }
                        if content.is_empty() {
                            content.push(bedrock_required_text_block(""));
                        }
                    }
                }
                result.push(json!({"role": "user", "content": content}));
            }
            llm::Message::Assistant(message) => {
                if message.content.is_empty() {
                    index += 1;
                    continue;
                }
                let mut content = Vec::new();
                for block in &message.content {
                    match block {
                        llm::ContentBlock::Text(text) => {
                            if let Some(text) = bedrock_text_block(&text.text) {
                                content.push(text);
                            }
                        }
                        llm::ContentBlock::ToolCall(tool_call) => {
                            let input = Value::Object(
                                tool_call
                                    .arguments
                                    .clone()
                                    .into_iter()
                                    .collect::<Map<_, _>>(),
                            );
                            content.push(json!({
                                "toolUse": {
                                    "toolUseId": tool_call.id,
                                    "name": tool_call.name,
                                    "input": sanitize_bedrock_document(&input),
                                }
                            }));
                        }
                        llm::ContentBlock::Thinking(thinking) => {
                            let thought = sanitize_surrogates(&thinking.thinking);
                            if thought.trim().is_empty() {
                                continue;
                            }
                            if !is_anthropic_claude_bedrock_model(model) {
                                content.push(json!({
                                    "reasoningContent": {"reasoningText": {"text": thought}}
                                }));
                            } else if thinking.thinking_signature.trim().is_empty() {
                                content.push(json!({"text": thought}));
                            } else {
                                content.push(json!({
                                    "reasoningContent": {
                                        "reasoningText": {
                                            "text": thought,
                                            "signature": thinking.thinking_signature,
                                        }
                                    }
                                }));
                            }
                        }
                        llm::ContentBlock::Image(_) => {}
                    }
                }
                if !content.is_empty() {
                    result.push(json!({"role": "assistant", "content": content}));
                }
            }
            llm::Message::ToolResult(_) => {
                let mut results = Vec::new();
                loop {
                    let llm::Message::ToolResult(next) = &transformed[index] else {
                        break;
                    };
                    results.push(json!({
                        "toolResult": {
                            "toolUseId": next.tool_call_id,
                            "content": bedrock_tool_result_content(&next.content)?,
                            "status": if next.is_error { "error" } else { "success" },
                        }
                    }));
                    index += 1;
                    if index == transformed.len()
                        || !matches!(&transformed[index], llm::Message::ToolResult(_))
                    {
                        break;
                    }
                }
                result.push(json!({"role": "user", "content": results}));
                continue;
            }
        }
        index += 1;
    }

    if cache_retention != CacheRetention::None
        && bedrock_supports_prompt_caching(model, environment)
        && matches!(
            result
                .last()
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str),
            Some("user")
        )
    {
        if let Some(content) = result
            .last_mut()
            .and_then(Value::as_object_mut)
            .and_then(|message| message.get_mut("content"))
            .and_then(Value::as_array_mut)
        {
            content.push(bedrock_cache_point(cache_retention));
        }
    }
    Ok(result)
}

/// Normalizes a transcript exactly as Bedrock replay requires: unsupported
/// images become placeholders, cross-model tool IDs are normalized, unsafe
/// thinking is downgraded, aborted/error messages are dropped, and unresolved
/// tool calls receive synthetic error results.
pub fn transform_bedrock_messages(
    messages: &[llm::Message],
    model: &llm::Model,
) -> Vec<llm::Message> {
    let image_aware = messages
        .iter()
        .cloned()
        .map(|message| downgrade_message_images(message, model))
        .collect::<Vec<_>>();
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
                                let normalized = normalize_bedrock_tool_call_id(&tool_call.id);
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
                flush_synthetic_tool_results(
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
                flush_synthetic_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_results,
                );
                result.push(llm::Message::User(user));
            }
        }
    }
    flush_synthetic_tool_results(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_results,
    );
    result
}

fn downgrade_message_images(message: llm::Message, model: &llm::Model) -> llm::Message {
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

fn flush_synthetic_tool_results(
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
                timestamp: now_millis(),
                ..llm::ToolResultMessage::default()
            })));
        }
    }
    existing_tool_results.clear();
}

/// Converts a cross-model tool-call ID into Bedrock's allowed identifier
/// alphabet and 64-byte maximum.
pub fn normalize_bedrock_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn bedrock_text_block(text: &str) -> Option<Value> {
    let sanitized = sanitize_surrogates(text);
    (!sanitized.trim().is_empty()).then(|| json!({"text": sanitized}))
}

fn bedrock_required_text_block(text: &str) -> Value {
    bedrock_text_block(text).unwrap_or_else(|| json!({"text": BEDROCK_EMPTY_TEXT_PLACEHOLDER}))
}

fn bedrock_image_block(mime_type: &str, data: &str) -> Result<Value, BedrockError> {
    let format = match mime_type {
        "image/jpeg" | "image/jpg" => "jpeg",
        "image/png" => "png",
        "image/gif" => "gif",
        "image/webp" => "webp",
        _ => {
            return Err(BedrockError::Message(format!(
                "unknown image type: {mime_type}"
            )));
        }
    };
    // Go's base64 decoder accepts line breaks, so validate an equivalent
    // normalized representation while preserving the original JSON payload.
    let normalized = data
        .bytes()
        .filter(|byte| !matches!(byte, b'\r' | b'\n'))
        .collect::<Vec<_>>();
    STANDARD
        .decode(normalized)
        .map_err(|error| BedrockError::Message(format!("invalid base64 image data: {error}")))?;
    Ok(json!({"image": {"format": format, "source": {"bytes": data}}}))
}

fn bedrock_tool_result_content(content: &[llm::ContentBlock]) -> Result<Vec<Value>, BedrockError> {
    let mut result = Vec::new();
    for block in content {
        match block {
            llm::ContentBlock::Text(text) => {
                if let Some(text) = bedrock_text_block(&text.text) {
                    result.push(text);
                }
            }
            llm::ContentBlock::Image(image) => {
                result.push(bedrock_image_block(&image.mime_type, &image.data)?);
            }
            llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => {}
        }
    }
    if result.is_empty() {
        result.push(bedrock_required_text_block(""));
    }
    Ok(result)
}

/// Removes empty object keys recursively because Bedrock's document type
/// rejects them.
pub fn sanitize_bedrock_document(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .filter(|(key, _)| !key.is_empty())
                .map(|(key, nested)| (key.clone(), sanitize_bedrock_document(nested)))
                .collect(),
        ),
        Value::Array(values) => {
            Value::Array(values.iter().map(sanitize_bedrock_document).collect())
        }
        other => other.clone(),
    }
}

fn convert_bedrock_tool_config(
    tools: &[llm::Tool],
    tool_choice: &BedrockToolChoice,
    supports_strict_mode: bool,
) -> Result<Option<Map<String, Value>>, BedrockError> {
    if tools.is_empty() || *tool_choice == BedrockToolChoice::None {
        return Ok(None);
    }
    let mut specs = Vec::with_capacity(tools.len());
    for tool in tools {
        let strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)?;
        let parameters = if tool.parameters.is_null() {
            Value::Object(Map::new())
        } else {
            tool.parameters.clone()
        };
        let mut spec = Map::from_iter([
            ("name".to_owned(), Value::String(tool.name.clone())),
            (
                "description".to_owned(),
                Value::String(tool.description.clone()),
            ),
            ("inputSchema".to_owned(), json!({"json": parameters})),
        ]);
        if strict {
            spec.insert("strict".to_owned(), Value::Bool(true));
        }
        specs.push(json!({"toolSpec": spec}));
    }

    let mut config = Map::from_iter([("tools".to_owned(), Value::Array(specs))]);
    match tool_choice {
        BedrockToolChoice::Unspecified | BedrockToolChoice::None => {}
        BedrockToolChoice::Auto => {
            config.insert("toolChoice".to_owned(), json!({"auto": {}}));
        }
        BedrockToolChoice::Any => {
            config.insert("toolChoice".to_owned(), json!({"any": {}}));
        }
        BedrockToolChoice::Tool(name) => {
            config.insert("toolChoice".to_owned(), json!({"tool": {"name": name}}));
        }
    }
    Ok(Some(config))
}

fn resolve_json_schema_strict_sampling(
    tool: &llm::Tool,
    supports_strict_mode: bool,
) -> Result<bool, BedrockError> {
    let Some(config) = tool
        .constrained_sampling
        .as_ref()
        .and_then(Value::as_object)
    else {
        return Ok(false);
    };
    if config.get("type").and_then(Value::as_str) != Some("json_schema") {
        return Ok(false);
    }
    if supports_strict_mode {
        return Ok(true);
    }
    if config.get("strict").and_then(Value::as_str) == Some("require") {
        return Err(BedrockError::Message(format!(
            "Tool {:?} requires JSON-schema constrained sampling, but strict tools are unsupported",
            tool.name
        )));
    }
    Ok(false)
}

fn bedrock_additional_model_request_fields(
    model: &llm::Model,
    options: &BedrockOptions,
) -> Option<Map<String, Value>> {
    let reasoning = options
        .reasoning
        .as_deref()
        .filter(|reasoning| !reasoning.is_empty())?;
    if !model.reasoning || !is_anthropic_claude_bedrock_model(model) {
        return None;
    }

    let mut display = options
        .thinking_display
        .as_deref()
        .filter(|display| !display.is_empty())
        .unwrap_or("summarized")
        .to_owned();
    if is_govcloud_bedrock_target(model, options) {
        display.clear();
    }

    let mut result = Map::new();
    if bedrock_supports_adaptive_thinking(model) {
        let mut thinking =
            Map::from_iter([("type".to_owned(), Value::String("adaptive".to_owned()))]);
        if !display.is_empty() {
            thinking.insert("display".to_owned(), Value::String(display));
        }
        result.insert("thinking".to_owned(), Value::Object(thinking));
        result.insert(
            "output_config".to_owned(),
            json!({"effort": map_bedrock_thinking_level_to_effort(model, reasoning)}),
        );
        return Some(result);
    }

    let level = stream::clamp_reasoning_level(reasoning);
    let mut budgets = llm::ThinkingBudgets {
        minimal: Some(1_024),
        low: Some(2_048),
        medium: Some(8_192),
        high: Some(16_384),
    };
    if let Some(overrides) = &options.thinking_budgets {
        merge_thinking_budgets(&mut budgets, overrides);
    }
    let budget = thinking_budget_for_level(&budgets, &level);
    let mut thinking = Map::from_iter([
        ("type".to_owned(), Value::String("enabled".to_owned())),
        ("budget_tokens".to_owned(), Value::from(budget)),
    ]);
    if !display.is_empty() {
        thinking.insert("display".to_owned(), Value::String(display));
    }
    result.insert("thinking".to_owned(), Value::Object(thinking));
    if options.interleaved_thinking != Some(false) {
        result.insert(
            "anthropic_beta".to_owned(),
            Value::Array(vec![Value::String(
                ANTHROPIC_INTERLEAVED_THINKING_BETA.to_owned(),
            )]),
        );
    }
    Some(result)
}

fn is_govcloud_bedrock_target(model: &llm::Model, options: &BedrockOptions) -> bool {
    configured_bedrock_region(options)
        .to_ascii_lowercase()
        .starts_with("us-gov-")
        || model.id.to_ascii_lowercase().starts_with("us-gov.")
        || model.id.to_ascii_lowercase().starts_with("arn:aws-us-gov:")
}

fn merge_thinking_budgets(target: &mut llm::ThinkingBudgets, source: &llm::ThinkingBudgets) {
    if source.minimal.is_some_and(|value| value != 0) {
        target.minimal = source.minimal;
    }
    if source.low.is_some_and(|value| value != 0) {
        target.low = source.low;
    }
    if source.medium.is_some_and(|value| value != 0) {
        target.medium = source.medium;
    }
    if source.high.is_some_and(|value| value != 0) {
        target.high = source.high;
    }
}

fn thinking_budget_for_level(budgets: &llm::ThinkingBudgets, level: &str) -> u64 {
    match level {
        llm::THINKING_MINIMAL => budgets.minimal,
        llm::THINKING_LOW => budgets.low,
        llm::THINKING_MEDIUM => budgets.medium,
        _ => budgets.high,
    }
    .unwrap_or(0)
    .into()
}

fn set_thinking_budget(budgets: &mut llm::ThinkingBudgets, level: &str, value: u64) {
    let value = u32::try_from(value).unwrap_or(u32::MAX);
    match level {
        llm::THINKING_MINIMAL => budgets.minimal = Some(value),
        llm::THINKING_LOW => budgets.low = Some(value),
        llm::THINKING_MEDIUM => budgets.medium = Some(value),
        _ => budgets.high = Some(value),
    }
}

fn string_map_to_json(values: &BTreeMap<String, String>) -> Map<String, Value> {
    values
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
        .collect()
}

fn sanitize_surrogates(text: &str) -> String {
    // Rust `str` is guaranteed to contain Unicode scalar values; unpaired
    // UTF-16 surrogates cannot reach this point.
    text.to_owned()
}

// ---------------------------------------------------------------------------
// HTTP request construction and streamed response conversion
// ---------------------------------------------------------------------------

/// Starts a Bedrock ConverseStream operation on a producer thread.
///
/// The returned stream emits the existing normalized `stream` event types and
/// resolves with an existing `llm::AssistantMessage`.
pub fn stream_bedrock(
    model: llm::Model,
    context: llm::Context,
    options: BedrockOptions,
) -> stream::AssistantMessageEventStream {
    let event_stream = stream::AssistantMessageEventStream::new();
    let producer_stream = event_stream.clone();
    thread::spawn(move || {
        BedrockStreamer::new(model, context, options, producer_stream).run();
    });
    event_stream
}

/// Maps unified simple-stream options onto Bedrock's full streaming operation.
pub fn stream_bedrock_simple(
    model: llm::Model,
    context: llm::Context,
    options: BedrockSimpleOptions,
) -> stream::AssistantMessageEventStream {
    let mut request = options.request;
    let mut sampling_params = model
        .sampling_params
        .as_ref()
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    sampling_params.extend(request.sampling_params.clone());
    request.sampling_params = sampling_params;

    let base_max_tokens = request
        .max_tokens
        .filter(|tokens| *tokens != 0)
        .unwrap_or(model.max_tokens);
    request.max_tokens = Some(stream::clamp_max_tokens_to_context(
        &model,
        &context,
        base_max_tokens,
    ));

    let reasoning = options.reasoning.filter(|reasoning| !reasoning.is_empty());
    if let Some(reasoning) = reasoning {
        if is_anthropic_claude_bedrock_model(&model) && !bedrock_supports_adaptive_thinking(&model)
        {
            let (max_tokens, thinking_budget) = adjust_anthropic_max_tokens_for_thinking(
                request.max_tokens.unwrap_or_default(),
                model.max_tokens,
                &reasoning,
                options.thinking_budgets.as_ref(),
            );
            let max_tokens = stream::clamp_max_tokens_to_context(&model, &context, max_tokens);
            let mut budgets = options.thinking_budgets.unwrap_or_default();
            let clamped_budget =
                thinking_budget.min(max_tokens.saturating_sub(stream::MIN_ANSWER_TOKENS));
            set_thinking_budget(
                &mut budgets,
                &stream::clamp_reasoning_level(&reasoning),
                clamped_budget,
            );
            request.max_tokens = Some(max_tokens);
            request.reasoning = Some(reasoning);
            request.thinking_budgets = Some(budgets);
        } else {
            request.reasoning = Some(reasoning);
            request.thinking_budgets = options.thinking_budgets;
        }
    } else {
        // BedrockSimpleOptions owns these two simple-stream settings. Ignore
        // any full-protocol values carried in `request`, just as Go's
        // SimpleStreamOptions has no reasoning fields of its own.
        request.reasoning = None;
        request.thinking_budgets = None;
    }
    stream_bedrock(model, context, request)
}

fn adjust_anthropic_max_tokens_for_thinking(
    base_max_tokens: u64,
    model_max_tokens: u64,
    reasoning: &str,
    custom: Option<&llm::ThinkingBudgets>,
) -> (u64, u64) {
    let mut budgets = llm::ThinkingBudgets {
        minimal: Some(1_024),
        low: Some(2_048),
        medium: Some(8_192),
        high: Some(16_384),
    };
    if let Some(custom) = custom {
        merge_thinking_budgets(&mut budgets, custom);
    }
    let mut thinking_budget =
        thinking_budget_for_level(&budgets, &stream::clamp_reasoning_level(reasoning));
    let max_tokens = base_max_tokens
        .saturating_add(thinking_budget)
        .min(model_max_tokens);
    if max_tokens <= thinking_budget {
        thinking_budget = max_tokens.saturating_sub(stream::MIN_ANSWER_TOKENS);
    }
    (max_tokens, thinking_budget)
}

struct BedrockStreamer {
    model: llm::Model,
    context: llm::Context,
    options: BedrockOptions,
    event_stream: stream::AssistantMessageEventStream,
    output: llm::AssistantMessage,
}

impl BedrockStreamer {
    fn new(
        model: llm::Model,
        context: llm::Context,
        options: BedrockOptions,
        event_stream: stream::AssistantMessageEventStream,
    ) -> Self {
        let output = llm::AssistantMessage {
            api: API_BEDROCK_CONVERSE_STREAM.to_owned(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            stop_reason: stream::STOP_PENDING.to_owned(),
            timestamp: now_millis(),
            ..llm::AssistantMessage::default()
        };
        Self {
            model,
            context,
            options,
            event_stream,
            output,
        }
    }

    fn run(mut self) {
        if let Err(error) = self.stream_once() {
            self.output.stop_reason =
                if self.is_cancelled() || matches!(error, BedrockError::Aborted) {
                    stream::STOP_ABORTED.to_owned()
                } else {
                    stream::STOP_ERROR.to_owned()
                };
            self.output.error_message = format_bedrock_stream_error(&error);
            let _ = self.event_stream.push(stream::AssistantMessageEvent::error(
                self.output.stop_reason.clone(),
                Arc::new(self.output.clone()),
            ));
            self.event_stream.end();
        }
    }

    fn stream_once(&mut self) -> Result<(), BedrockError> {
        self.check_cancelled()?;
        let mut params = build_bedrock_params(&self.model, &self.context, &self.options)?;
        if let Some(callback) = &self.options.on_payload {
            if let Some(replacement) = callback(params.clone(), &self.model) {
                params = replacement;
            }
        }

        let mut response = self.retry_request(&params)?;
        if let Some(callback) = &self.options.on_response {
            callback(
                BedrockResponse {
                    status: response.status().as_u16(),
                    headers: response_headers(response.headers()),
                },
                &self.model,
            );
        }
        self.consume_stream(&mut response)?;
        self.check_cancelled()?;

        if self.output.stop_reason == stream::STOP_PENDING {
            return Err(BedrockError::Message(
                "bedrock stream ended without a stop reason".to_owned(),
            ));
        }
        if matches!(
            self.output.stop_reason.as_str(),
            stream::STOP_ABORTED | stream::STOP_ERROR
        ) {
            return Err(BedrockError::Message(
                if self.output.error_message.is_empty() {
                    "an unknown error occurred".to_owned()
                } else {
                    self.output.error_message.clone()
                },
            ));
        }
        self.event_stream
            .push(stream::AssistantMessageEvent::done(
                self.output.stop_reason.clone(),
                Arc::new(self.output.clone()),
            ))
            .map_err(|error| BedrockError::Message(error.to_string()))?;
        self.event_stream.end();
        Ok(())
    }

    fn retry_request(&self, params: &Map<String, Value>) -> Result<Response, BedrockError> {
        let mut builder = Client::builder();
        if let Some(timeout) = self.options.timeout.filter(|timeout| !timeout.is_zero()) {
            builder = builder.timeout(timeout);
        }
        let client = builder.build().map_err(|error| {
            BedrockError::Message(format!("build Bedrock HTTP client: {error}"))
        })?;
        let mut retries_remaining = self.options.max_retries;
        loop {
            self.check_cancelled()?;
            match self.do_request(&client, params) {
                Ok(response) => return Ok(response),
                Err(error) => {
                    if self.is_cancelled() {
                        return Err(BedrockError::Aborted);
                    }
                    let Some(provider) = error.provider_error() else {
                        return Err(error);
                    };
                    if retries_remaining == 0 || !stream::is_retryable_provider_error(provider) {
                        return Err(error);
                    }
                    let retry_index = self.options.max_retries - retries_remaining;
                    retries_remaining -= 1;
                    let delay = stream::retry_delay_with_jitter(
                        provider,
                        retry_index,
                        SystemTime::now(),
                        self.retry_delay_limit(),
                        self.retry_jitter(),
                    )?;
                    self.sleep_with_cancellation(delay)?;
                }
            }
        }
    }

    fn retry_delay_limit(&self) -> stream::RetryDelayLimit {
        match self.options.max_retry_delay {
            None => stream::RetryDelayLimit::Default,
            Some(delay) if delay.is_zero() => stream::RetryDelayLimit::Unlimited,
            Some(delay) => stream::RetryDelayLimit::Maximum(delay),
        }
    }

    fn retry_jitter(&self) -> f64 {
        self.options.retry_jitter.unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| f64::from(duration.subsec_nanos()) / 1_000_000_000.0)
                .unwrap_or(0.0)
        })
    }

    fn sleep_with_cancellation(&self, delay: Duration) -> Result<(), BedrockError> {
        let mut remaining = delay;
        let quantum = Duration::from_millis(50);
        while !remaining.is_zero() {
            self.check_cancelled()?;
            let step = remaining.min(quantum);
            thread::sleep(step);
            remaining = remaining.saturating_sub(step);
        }
        self.check_cancelled()
    }

    fn do_request(
        &self,
        client: &Client,
        params: &Map<String, Value>,
    ) -> Result<Response, BedrockError> {
        let request = build_bedrock_http_request(
            &self.model,
            &self.options,
            params,
            OffsetDateTime::now_utc(),
        )?;
        let request_url = request.url.to_string();
        let mut builder = client
            .request(reqwest::Method::POST, request.url)
            .body(request.body);
        for (name, value) in request.headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
                BedrockError::Message("Bedrock request contains an invalid header name".to_owned())
            })?;
            let value = HeaderValue::from_str(&value).map_err(|_| {
                BedrockError::Message("Bedrock request contains an invalid header value".to_owned())
            })?;
            builder = builder.header(name, value);
        }
        let mut response = builder.send().map_err(|error| {
            if self.is_cancelled() {
                BedrockError::Aborted
            } else {
                BedrockError::Provider(stream::ProviderError {
                    status: 0,
                    message: error.to_string(),
                    ..stream::ProviderError::default()
                })
            }
        })?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            let headers = response_headers(response.headers());
            let body = read_bounded_body(&mut response, MAX_PROVIDER_ERROR_BODY_CHARS + 1)?;
            return Err(BedrockError::Provider(stream::ProviderError {
                status,
                headers,
                body: truncate_error_text(body.trim(), MAX_PROVIDER_ERROR_BODY_CHARS),
                message: format!("POST {request_url} failed with status {status}"),
            }));
        }
        Ok(response)
    }

    fn consume_stream<R: Read>(&mut self, body: R) -> Result<(), BedrockError> {
        let mut reader = EventStreamReader::new(body);
        let mut blocks = Vec::<BedrockBlock>::new();
        while let Some(message) = reader.next_message()? {
            self.check_cancelled()?;
            if let Some(exception_type) = message.exception_type().filter(|value| !value.is_empty())
            {
                return Err(bedrock_exception_error(exception_type, &message.payload));
            }
            if matches!(message.message_type(), Some("exception" | "error")) {
                return Err(bedrock_exception_error(
                    message
                        .headers
                        .get(":error-code")
                        .map(String::as_str)
                        .unwrap_or_default(),
                    &message.payload,
                ));
            }

            let event: BedrockStreamEvent = if message.payload.is_empty() {
                BedrockStreamEvent::default()
            } else {
                serde_json::from_slice(&message.payload).map_err(|error| {
                    BedrockError::Message(format!("parsing bedrock event: {error}"))
                })?
            };
            match message.event_type().unwrap_or_default() {
                "messageStart" => {
                    if event.role != "assistant" {
                        return Err(BedrockError::Message(format!(
                            "unexpected assistant message start but got {} message start instead",
                            event.role
                        )));
                    }
                    self.push_progress(stream::AssistantMessageEvent::new(stream::EVENT_START))?;
                }
                "contentBlockStart" => {
                    let Some(wire_index) = event.content_block_index else {
                        continue;
                    };
                    let Some(tool_use) = event.start.and_then(|start| start.tool_use) else {
                        continue;
                    };
                    let content_index = self.output.content.len();
                    self.output
                        .content
                        .push(llm::ContentBlock::ToolCall(llm::ToolCall {
                            id: tool_use.tool_use_id,
                            name: tool_use.name,
                            arguments: BTreeMap::new(),
                            ..llm::ToolCall::default()
                        }));
                    blocks.push(BedrockBlock {
                        kind: BedrockBlockKind::ToolCall,
                        wire_index,
                        content_index,
                        tool_json: stream::IncrementalToolArguments::new(),
                    });
                    self.push_progress(stream::AssistantMessageEvent {
                        event_type: stream::EVENT_TOOLCALL_START.to_owned(),
                        content_index: Some(content_index),
                        ..stream::AssistantMessageEvent::default()
                    })?;
                }
                "contentBlockDelta" => {
                    let (Some(wire_index), Some(delta)) = (event.content_block_index, event.delta)
                    else {
                        continue;
                    };
                    if let Some(text) = delta.text {
                        self.consume_text_delta(&mut blocks, wire_index, &text)?;
                    } else if let Some(tool_use) = delta.tool_use {
                        self.consume_tool_delta(&mut blocks, wire_index, &tool_use.input)?;
                    } else if let Some(reasoning) = delta.reasoning_content {
                        self.consume_reasoning_delta(
                            &mut blocks,
                            wire_index,
                            &reasoning.text,
                            &reasoning.signature,
                        )?;
                    }
                }
                "contentBlockStop" => {
                    if let Some(wire_index) = event.content_block_index {
                        self.consume_block_stop(&mut blocks, wire_index)?;
                    }
                }
                "messageStop" => {
                    self.output.raw_stop_reason = event.stop_reason.clone();
                    let (stop_reason, error_message) = map_bedrock_stop_reason(&event.stop_reason);
                    self.output.stop_reason = stop_reason;
                    if let Some(error_message) = error_message {
                        self.output.error_message = error_message;
                    }
                }
                "metadata" => {
                    if let Some(usage) = event.usage {
                        self.output.usage.input = usage.input_tokens;
                        self.output.usage.output = usage.output_tokens;
                        self.output.usage.cache_read = usage.cache_read_input_tokens;
                        self.output.usage.cache_write = usage.cache_write_input_tokens;
                        self.output.usage.total_tokens = usage.total_tokens;
                        if self.output.usage.total_tokens == 0 {
                            self.output.usage.total_tokens = self
                                .output
                                .usage
                                .input
                                .saturating_add(self.output.usage.output);
                        }
                        stream::calculate_usage_cost(&self.model, &mut self.output.usage);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn consume_text_delta(
        &mut self,
        blocks: &mut Vec<BedrockBlock>,
        wire_index: usize,
        delta: &str,
    ) -> Result<(), BedrockError> {
        let position = match find_block(blocks, wire_index) {
            Some(position) => position,
            None => {
                let content_index = self.output.content.len();
                self.output
                    .content
                    .push(llm::ContentBlock::Text(llm::TextContent::default()));
                blocks.push(BedrockBlock {
                    kind: BedrockBlockKind::Text,
                    wire_index,
                    content_index,
                    tool_json: stream::IncrementalToolArguments::new(),
                });
                self.push_progress(stream::AssistantMessageEvent {
                    event_type: stream::EVENT_TEXT_START.to_owned(),
                    content_index: Some(content_index),
                    ..stream::AssistantMessageEvent::default()
                })?;
                blocks.len() - 1
            }
        };
        if blocks[position].kind != BedrockBlockKind::Text {
            return Ok(());
        }
        let content_index = blocks[position].content_index;
        if let Some(llm::ContentBlock::Text(text)) = self.output.content.get_mut(content_index) {
            text.text.push_str(delta);
        }
        self.push_progress(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TEXT_DELTA.to_owned(),
            content_index: Some(content_index),
            delta: delta.to_owned(),
            ..stream::AssistantMessageEvent::default()
        })
    }

    fn consume_tool_delta(
        &mut self,
        blocks: &mut [BedrockBlock],
        wire_index: usize,
        delta: &str,
    ) -> Result<(), BedrockError> {
        let Some(position) = find_block(blocks, wire_index) else {
            return Ok(());
        };
        if blocks[position].kind != BedrockBlockKind::ToolCall {
            return Ok(());
        }
        let (content_index, preview) = {
            let block = &mut blocks[position];
            let reparsed = block.tool_json.push(delta);
            (
                block.content_index,
                reparsed.then(|| block.tool_json.tool_arguments()),
            )
        };
        if let Some(arguments) = preview {
            if let Some(llm::ContentBlock::ToolCall(tool_call)) =
                self.output.content.get_mut(content_index)
            {
                tool_call.arguments = arguments;
            }
        }
        self.push_progress(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TOOLCALL_DELTA.to_owned(),
            content_index: Some(content_index),
            delta: delta.to_owned(),
            ..stream::AssistantMessageEvent::default()
        })
    }

    fn consume_reasoning_delta(
        &mut self,
        blocks: &mut Vec<BedrockBlock>,
        wire_index: usize,
        text_delta: &str,
        signature: &str,
    ) -> Result<(), BedrockError> {
        let position = match find_block(blocks, wire_index) {
            Some(position) => position,
            None => {
                let content_index = self.output.content.len();
                self.output
                    .content
                    .push(llm::ContentBlock::Thinking(llm::ThinkingContent::default()));
                blocks.push(BedrockBlock {
                    kind: BedrockBlockKind::Thinking,
                    wire_index,
                    content_index,
                    tool_json: stream::IncrementalToolArguments::new(),
                });
                self.push_progress(stream::AssistantMessageEvent {
                    event_type: stream::EVENT_THINKING_START.to_owned(),
                    content_index: Some(content_index),
                    ..stream::AssistantMessageEvent::default()
                })?;
                blocks.len() - 1
            }
        };
        if blocks[position].kind != BedrockBlockKind::Thinking {
            return Ok(());
        }
        let content_index = blocks[position].content_index;
        if !text_delta.is_empty() {
            if let Some(llm::ContentBlock::Thinking(thinking)) =
                self.output.content.get_mut(content_index)
            {
                thinking.thinking.push_str(text_delta);
            }
            self.push_progress(stream::AssistantMessageEvent {
                event_type: stream::EVENT_THINKING_DELTA.to_owned(),
                content_index: Some(content_index),
                delta: text_delta.to_owned(),
                ..stream::AssistantMessageEvent::default()
            })?;
        }
        if !signature.is_empty() {
            if let Some(llm::ContentBlock::Thinking(thinking)) =
                self.output.content.get_mut(content_index)
            {
                thinking.thinking_signature.push_str(signature);
            }
        }
        Ok(())
    }

    fn consume_block_stop(
        &mut self,
        blocks: &mut [BedrockBlock],
        wire_index: usize,
    ) -> Result<(), BedrockError> {
        let Some(position) = find_block(blocks, wire_index) else {
            return Ok(());
        };
        let block = &mut blocks[position];
        let content_index = block.content_index;
        match block.kind {
            BedrockBlockKind::Text => {
                let content = match self.output.content.get(content_index) {
                    Some(llm::ContentBlock::Text(text)) => text.text.clone(),
                    _ => String::new(),
                };
                self.push_progress(stream::AssistantMessageEvent {
                    event_type: stream::EVENT_TEXT_END.to_owned(),
                    content_index: Some(content_index),
                    content,
                    ..stream::AssistantMessageEvent::default()
                })
            }
            BedrockBlockKind::Thinking => {
                let content = match self.output.content.get(content_index) {
                    Some(llm::ContentBlock::Thinking(thinking)) => thinking.thinking.clone(),
                    _ => String::new(),
                };
                self.push_progress(stream::AssistantMessageEvent {
                    event_type: stream::EVENT_THINKING_END.to_owned(),
                    content_index: Some(content_index),
                    content,
                    ..stream::AssistantMessageEvent::default()
                })
            }
            BedrockBlockKind::ToolCall => {
                let arguments = block.tool_json.finish_tool_arguments();
                let tool_call = match self.output.content.get_mut(content_index) {
                    Some(llm::ContentBlock::ToolCall(tool_call)) => {
                        tool_call.arguments = arguments;
                        tool_call.clone()
                    }
                    _ => return Ok(()),
                };
                self.push_progress(stream::AssistantMessageEvent {
                    event_type: stream::EVENT_TOOLCALL_END.to_owned(),
                    content_index: Some(content_index),
                    tool_call: Some(tool_call),
                    ..stream::AssistantMessageEvent::default()
                })
            }
        }
    }

    fn push_progress(&self, mut event: stream::AssistantMessageEvent) -> Result<(), BedrockError> {
        event.partial = Some(Arc::new(self.output.clone()));
        self.event_stream
            .push(event)
            .map_err(|error| BedrockError::Message(error.to_string()))
    }

    fn check_cancelled(&self) -> Result<(), BedrockError> {
        if self.is_cancelled() {
            Err(BedrockError::Aborted)
        } else {
            Ok(())
        }
    }

    fn is_cancelled(&self) -> bool {
        self.options
            .cancellation
            .as_ref()
            .is_some_and(BedrockCancellation::is_cancelled)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BedrockBlockKind {
    Text,
    Thinking,
    ToolCall,
}

struct BedrockBlock {
    kind: BedrockBlockKind,
    wire_index: usize,
    content_index: usize,
    tool_json: stream::IncrementalToolArguments,
}

fn find_block(blocks: &[BedrockBlock], wire_index: usize) -> Option<usize> {
    blocks
        .iter()
        .position(|block| block.wire_index == wire_index)
}

#[derive(Default, Deserialize)]
struct BedrockStreamEvent {
    #[serde(default)]
    role: String,
    #[serde(default)]
    start: Option<BedrockContentBlockStart>,
    #[serde(default)]
    delta: Option<BedrockContentBlockDelta>,
    #[serde(rename = "contentBlockIndex", default)]
    content_block_index: Option<usize>,
    #[serde(rename = "stopReason", default)]
    stop_reason: String,
    #[serde(default)]
    usage: Option<BedrockUsage>,
}

#[derive(Deserialize)]
struct BedrockContentBlockStart {
    #[serde(rename = "toolUse", default)]
    tool_use: Option<BedrockToolUseStart>,
}

#[derive(Deserialize)]
struct BedrockToolUseStart {
    #[serde(rename = "toolUseId", default)]
    tool_use_id: String,
    #[serde(default)]
    name: String,
}

#[derive(Deserialize)]
struct BedrockContentBlockDelta {
    #[serde(default)]
    text: Option<String>,
    #[serde(rename = "toolUse", default)]
    tool_use: Option<BedrockToolUseDelta>,
    #[serde(rename = "reasoningContent", default)]
    reasoning_content: Option<BedrockReasoningDelta>,
}

#[derive(Deserialize)]
struct BedrockToolUseDelta {
    #[serde(default)]
    input: String,
}

#[derive(Deserialize)]
struct BedrockReasoningDelta {
    #[serde(default)]
    text: String,
    #[serde(default)]
    signature: String,
}

#[derive(Deserialize)]
struct BedrockUsage {
    #[serde(rename = "inputTokens", default)]
    input_tokens: u64,
    #[serde(rename = "outputTokens", default)]
    output_tokens: u64,
    #[serde(rename = "totalTokens", default)]
    total_tokens: u64,
    #[serde(rename = "cacheReadInputTokens", default)]
    cache_read_input_tokens: u64,
    #[serde(rename = "cacheWriteInputTokens", default)]
    cache_write_input_tokens: u64,
}

/// Builds, authenticates, and signs a request without performing network I/O.
pub fn build_bedrock_http_request(
    model: &llm::Model,
    options: &BedrockOptions,
    params: &Map<String, Value>,
    now: OffsetDateTime,
) -> Result<BedrockHttpRequest, BedrockError> {
    let body = serde_json::to_vec(params)?;
    let target = resolve_bedrock_target(model, options);
    let url = bedrock_request_url(&target.endpoint, &model.id)?;
    let mut request = BedrockHttpRequest {
        method: "POST".to_owned(),
        url,
        headers: BTreeMap::from([
            ("Content-Type".to_owned(), "application/json".to_owned()),
            (
                "Accept".to_owned(),
                "application/vnd.amazon.eventstream".to_owned(),
            ),
        ]),
        body,
    };
    for (name, value) in &options.headers {
        if let Some(value) = value.as_ref().filter(|_| !is_bedrock_reserved_header(name)) {
            set_header(&mut request.headers, name, value.clone());
        }
    }

    let skip_auth = provider_env_value(&options.environment, "AWS_BEDROCK_SKIP_AUTH") == "1";
    let bearer_token = options
        .bearer_token
        .as_deref()
        .filter(|value| !value.is_empty())
        .or_else(|| options.api_key.as_deref().filter(|value| !value.is_empty()))
        .map(str::to_owned)
        .or_else(|| {
            let value = provider_env_value(&options.environment, "AWS_BEARER_TOKEN_BEDROCK");
            (!value.is_empty()).then_some(value)
        })
        .filter(|value| value != "<authenticated>");
    if skip_auth {
        sign_aws_request(
            &mut request,
            &AwsCredentials {
                access_key_id: "dummy-access-key".to_owned(),
                secret_access_key: "dummy-secret-key".to_owned(),
                session_token: String::new(),
                source: "AWS_BEDROCK_SKIP_AUTH".to_owned(),
            },
            &target.region,
            BEDROCK_SERVICE,
            now,
        )?;
    } else if let Some(token) = bearer_token {
        set_header(
            &mut request.headers,
            "Authorization",
            format!("Bearer {token}"),
        );
    } else {
        let credentials = resolve_aws_credentials(
            resolve_bedrock_profile(options).as_deref(),
            &options.environment,
        )?;
        sign_aws_request(
            &mut request,
            &credentials,
            &target.region,
            BEDROCK_SERVICE,
            now,
        )?;
    }
    Ok(request)
}

fn bedrock_request_url(endpoint: &str, model_id: &str) -> Result<Url, BedrockError> {
    let endpoint = endpoint.trim_end_matches('/');
    // Go's url.PathEscape allows RFC 3986 path-segment pchars such as ':'.
    // Leaving ':' on the wire lets canonical URI construction encode it once,
    // instead of incorrectly double-encoding an already escaped colon.
    let model_id = go_path_escape(model_id);
    Ok(Url::parse(&format!(
        "{endpoint}/model/{model_id}/converse-stream"
    ))?)
}

fn go_path_escape(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut output = String::with_capacity(value.len());
    for byte in value.bytes() {
        let allowed = byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'-' | b'.' | b'_' | b'~' | b'$' | b'&' | b'+' | b'=' | b':' | b'@'
            );
        if allowed {
            output.push(byte as char);
        } else {
            output.push('%');
            output.push(HEX[(byte >> 4) as usize] as char);
            output.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    output
}

/// Returns whether callers must not override this header before SigV4.
pub fn is_bedrock_reserved_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.starts_with("x-amz-") || matches!(name.as_str(), "authorization" | "host")
}

fn response_headers(headers: &reqwest::header::HeaderMap) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    for (name, value) in headers {
        let text = value.to_str().unwrap_or_default();
        result
            .entry(name.as_str().to_owned())
            .and_modify(|current: &mut String| {
                current.push(',');
                current.push_str(text);
            })
            .or_insert_with(|| text.to_owned());
    }
    result
}

fn read_bounded_body(response: &mut Response, maximum: usize) -> Result<String, BedrockError> {
    let mut body = Vec::new();
    response
        .take(maximum as u64)
        .read_to_end(&mut body)
        .map_err(BedrockError::Io)?;
    Ok(String::from_utf8_lossy(&body).into_owned())
}

fn bedrock_exception_error(exception_type: &str, payload: &[u8]) -> BedrockError {
    let mut message = serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|value| {
            value
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| String::from_utf8_lossy(payload).trim().to_owned());
    if message.is_empty() {
        message = "Unknown error".to_owned();
    }
    if message.to_ascii_lowercase().contains("data retention mode") {
        message.push_str(" See ");
        message.push_str(BEDROCK_DATA_RETENTION_DOCS_URL);
        message.push_str(" for supported data retention modes.");
    }
    let prefix = match exception_type {
        "InternalServerException" => "Internal server error",
        "ModelStreamErrorException" => "Model stream error",
        "ValidationException" => "Validation error",
        "ThrottlingException" => "Throttling error",
        "ServiceUnavailableException" => "Service unavailable",
        _ => exception_type,
    };
    if prefix.is_empty() {
        BedrockError::Message(message)
    } else {
        BedrockError::Message(format!("{prefix}: {message}"))
    }
}

/// Maps a Converse stop reason into the common stream reason, retaining an
/// explanatory error for an unrecognized provider reason.
pub fn map_bedrock_stop_reason(reason: &str) -> (String, Option<String>) {
    match reason {
        "end_turn" | "stop_sequence" => (stream::STOP_STOP.to_owned(), None),
        "max_tokens" | "model_context_window_exceeded" => (stream::STOP_LENGTH.to_owned(), None),
        "tool_use" => (stream::STOP_TOOL_USE.to_owned(), None),
        "" => (stream::STOP_ERROR.to_owned(), None),
        other => (
            stream::STOP_ERROR.to_owned(),
            Some(format!("Provider stopped with: {other}")),
        ),
    }
}

fn now_millis() -> i64 {
    let milliseconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    milliseconds.min(i64::MAX as u128) as i64
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        io::{Cursor, Read, Write},
        net::{TcpListener, TcpStream},
        path::PathBuf,
        sync::{Arc, Mutex},
        thread::{self, JoinHandle},
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::{Value, json};

    use super::*;

    fn test_model(base_url: impl Into<String>) -> llm::Model {
        llm::Model {
            id: "anthropic.claude-sonnet-4-5-20250929-v1:0".to_owned(),
            name: "Claude Sonnet 4.5".to_owned(),
            api: API_BEDROCK_CONVERSE_STREAM.to_owned(),
            provider: "amazon-bedrock".to_owned(),
            base_url: base_url.into(),
            input: vec!["text".to_owned()],
            context_window: 200_000,
            max_tokens: 8_192,
            ..llm::Model::default()
        }
    }

    fn test_options() -> BedrockOptions {
        BedrockOptions {
            environment: BTreeMap::from([
                ("AWS_ACCESS_KEY_ID".to_owned(), "AKIDTEST".to_owned()),
                ("AWS_SECRET_ACCESS_KEY".to_owned(), "secret-test".to_owned()),
                ("AWS_REGION".to_owned(), "us-east-1".to_owned()),
            ]),
            retry_jitter: Some(0.0),
            ..BedrockOptions::default()
        }
    }

    fn event_frame(event_type: &str, payload: &str) -> Vec<u8> {
        encode_event_stream_message(
            &BTreeMap::from([
                (":event-type".to_owned(), event_type.to_owned()),
                (":message-type".to_owned(), "event".to_owned()),
            ]),
            payload.as_bytes(),
        )
        .expect("fixture frame")
    }

    fn stream_frames(frames: impl IntoIterator<Item = Vec<u8>>) -> Vec<u8> {
        frames.into_iter().flatten().collect()
    }

    fn text_turn(text: &str) -> Vec<u8> {
        let text = serde_json::to_string(text).expect("text JSON");
        stream_frames([
            event_frame("messageStart", r#"{"role":"assistant"}"#),
            event_frame(
                "contentBlockDelta",
                &format!(r#"{{"contentBlockIndex":0,"delta":{{"text":{text}}}}}"#),
            ),
            event_frame("contentBlockStop", r#"{"contentBlockIndex":0}"#),
            event_frame("messageStop", r#"{"stopReason":"end_turn"}"#),
            event_frame(
                "metadata",
                r#"{"usage":{"inputTokens":10,"outputTokens":4,"totalTokens":14}}"#,
            ),
        ])
    }

    fn consume_fixture(
        body: Vec<u8>,
    ) -> (llm::AssistantMessage, Vec<stream::AssistantMessageEvent>) {
        let model = test_model("");
        let events = stream::AssistantMessageEventStream::new();
        let mut converter = BedrockStreamer::new(
            model,
            llm::Context::default(),
            test_options(),
            events.clone(),
        );
        converter
            .consume_stream(Cursor::new(body))
            .expect("fixture should decode");
        let mut emitted = Vec::new();
        while let Some(event) = events.try_next() {
            emitted.push(event);
        }
        (converter.output, emitted)
    }

    #[test]
    fn event_stream_round_trip_and_multiple_frames_match_aws_fixtures() {
        let frame = encode_event_stream_message(
            &BTreeMap::from([
                (":event-type".to_owned(), "contentBlockDelta".to_owned()),
                (":message-type".to_owned(), "event".to_owned()),
            ]),
            br#"{"delta":{"text":"hi"}}"#,
        )
        .expect("encode");
        let mut reader = EventStreamReader::new(Cursor::new(frame));
        let message = reader.next_message().expect("decode").expect("one frame");
        assert_eq!(message.event_type(), Some("contentBlockDelta"));
        assert_eq!(message.message_type(), Some("event"));
        assert_eq!(message.payload, br#"{"delta":{"text":"hi"}}"#);
        assert!(reader.next_message().expect("clean EOF").is_none());

        let body = stream_frames([
            event_frame("messageStart", "{}"),
            event_frame("contentBlockDelta", "{}"),
            event_frame("messageStop", "{}"),
        ]);
        let mut reader = EventStreamReader::new(Cursor::new(body));
        let mut seen = Vec::new();
        while let Some(message) = reader.next_message().expect("valid frame") {
            seen.push(message.event_type().unwrap_or_default().to_owned());
        }
        assert_eq!(seen, ["messageStart", "contentBlockDelta", "messageStop"]);
    }

    #[test]
    fn event_stream_rejects_corruption_truncation_and_unsupported_headers() {
        let mut bad_prelude = event_frame("x", "{}");
        bad_prelude[0..4].copy_from_slice(&999_999_u32.to_be_bytes());
        assert!(matches!(
            EventStreamReader::new(Cursor::new(bad_prelude)).next_message(),
            Err(EventStreamError::PreludeCrcMismatch { .. })
        ));

        let mut bad_message = event_frame("x", "{}");
        let last_payload_byte = bad_message.len() - 6;
        bad_message[last_payload_byte] ^= 0xff;
        assert!(matches!(
            EventStreamReader::new(Cursor::new(bad_message)).next_message(),
            Err(EventStreamError::MessageCrcMismatch { .. })
        ));
        assert!(matches!(
            EventStreamReader::new(Cursor::new(vec![0, 0, 0])).next_message(),
            Err(EventStreamError::TruncatedPrelude)
        ));

        let mut implausible = [0_u8; 12];
        implausible[..4]
            .copy_from_slice(&((MAX_EVENT_STREAM_MESSAGE_SIZE + 1) as u32).to_be_bytes());
        let prelude_crc = crc32_ieee(&implausible[..8]);
        implausible[8..].copy_from_slice(&prelude_crc.to_be_bytes());
        assert!(matches!(
            EventStreamReader::new(Cursor::new(implausible)).next_message(),
            Err(EventStreamError::ImplausibleLength(_))
        ));
        assert!(matches!(
            parse_event_stream_headers(&[1, b'x', 99]),
            Err(EventStreamError::UnknownHeaderValueType(99))
        ));
    }

    #[test]
    fn event_stream_skips_non_string_header_types() {
        let mut headers = Vec::new();
        let mut append = |name: &str, value_type: u8, value: &[u8]| {
            headers.push(name.len() as u8);
            headers.extend_from_slice(name.as_bytes());
            headers.push(value_type);
            headers.extend_from_slice(value);
        };
        append("bool", 0, &[]);
        append("byte", 2, &[1]);
        append("short", 3, &[0, 1]);
        append("integer", 4, &[0, 0, 0, 1]);
        append("long", 5, &[0; 8]);
        append("uuid", 9, &[0; 16]);
        append(":event-type", 7, &[0, 5, b'e', b'v', b'e', b'n', b't']);
        assert_eq!(
            parse_event_stream_headers(&headers).expect("valid headers"),
            BTreeMap::from([(":event-type".to_owned(), "event".to_owned())])
        );
    }

    #[test]
    fn sigv4_signing_canonicalizes_and_binds_the_payload() {
        let when = OffsetDateTime::from_unix_timestamp(1_767_323_045).expect("known timestamp");
        let credentials = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".to_owned(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_owned(),
            session_token: String::new(),
            source: "test".to_owned(),
        };
        let make_request = |body: &[u8]| BedrockHttpRequest {
            method: "POST".to_owned(),
            url: Url::parse(
                "https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/converse-stream",
            )
            .expect("URL"),
            headers: BTreeMap::from([(
                "Content-Type".to_owned(),
                "  application/json  ".to_owned(),
            )]),
            body: body.to_vec(),
        };
        let mut first = make_request(br#"{"messages":[]}"#);
        sign_aws_request(&mut first, &credentials, "us-east-1", BEDROCK_SERVICE, when)
            .expect("sign");
        let authorization = first.headers["Authorization"].clone();
        assert!(authorization.starts_with(
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260102/us-east-1/bedrock/aws4_request"
        ));
        assert!(
            authorization
                .contains("SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date")
        );
        assert_eq!(first.headers["X-Amz-Date"], "20260102T030405Z");
        assert_eq!(
            first.headers["X-Amz-Content-Sha256"],
            sha256_hex(br#"{"messages":[]}"#)
        );

        let mut same = make_request(br#"{"messages":[]}"#);
        sign_aws_request(&mut same, &credentials, "us-east-1", BEDROCK_SERVICE, when)
            .expect("sign same");
        assert_eq!(same.headers["Authorization"], authorization);

        let mut changed = make_request(br#"{"messages":[1]}"#);
        sign_aws_request(
            &mut changed,
            &credentials,
            "us-east-1",
            BEDROCK_SERVICE,
            when,
        )
        .expect("sign changed");
        assert_ne!(changed.headers["Authorization"], authorization);
    }

    #[test]
    fn sigv4_uri_and_query_encoding_matches_go_rules() {
        assert_eq!(aws_canonical_uri(""), "/");
        assert_eq!(
            aws_canonical_uri(
                "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse-stream"
            ),
            "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream"
        );
        assert_eq!(
            aws_canonical_uri("/model/x%2Fy/converse-stream"),
            "/model/x%252Fy/converse-stream"
        );
        let url =
            Url::parse("https://example.test/?b=two&a=one+1&a=one+2&c%3Akey=v%2Fv").expect("URL");
        assert_eq!(
            aws_canonical_query(&url),
            "a=one%201&a=one%202&b=two&c%3Akey=v%2Fv"
        );
        let request_url = bedrock_request_url(
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        .expect("request URL");
        assert!(request_url.path().contains(":0"));
        assert!(aws_canonical_uri(request_url.path()).contains("%3A0"));
    }

    #[test]
    fn credentials_profile_region_and_explicit_key_precedence_match_go() {
        let directory = temporary_directory("bedrock-credentials");
        let credentials_path = directory.join("credentials");
        let config_path = directory.join("config");
        fs::write(
            &credentials_path,
            "[default]\naws_access_key_id = AKIDDEFAULT\naws_secret_access_key = secret-default\n\n[work]\naws_access_key_id = AKIDWORK\naws_secret_access_key = secret-work\naws_session_token = token-work\n",
        )
        .expect("credentials fixture");
        fs::write(
            &config_path,
            "[default]\nregion = us-west-2\n\n[profile work]\nregion = eu-central-1\n",
        )
        .expect("config fixture");
        let environment = BTreeMap::from([
            (
                "AWS_SHARED_CREDENTIALS_FILE".to_owned(),
                credentials_path.display().to_string(),
            ),
            (
                "AWS_CONFIG_FILE".to_owned(),
                config_path.display().to_string(),
            ),
            ("AWS_ACCESS_KEY_ID".to_owned(), "AKIDENV".to_owned()),
            ("AWS_SECRET_ACCESS_KEY".to_owned(), "secret-env".to_owned()),
        ]);
        let profile = resolve_aws_credentials(Some("work"), &environment).expect("profile");
        assert_eq!(profile.access_key_id, "AKIDWORK");
        assert_eq!(profile.session_token, "token-work");
        assert_eq!(
            region_from_profile(Some("work"), &environment),
            "eu-central-1"
        );
        assert_eq!(region_from_profile(None, &environment), "us-west-2");
        let ambient = resolve_aws_credentials(None, &environment).expect("static environment");
        assert_eq!(ambient.access_key_id, "AKIDENV");
        fs::remove_dir_all(directory).expect("remove fixture");
    }

    #[test]
    fn region_resolution_and_model_capability_detection_match_go() {
        let isolated = isolated_environment();
        let options = BedrockOptions {
            environment: isolated.clone(),
            region: Some("ap-south-1".to_owned()),
            ..BedrockOptions::default()
        };
        let mut arn = test_model("");
        arn.id = "arn:aws:bedrock:eu-west-3:123:inference-profile/x".to_owned();
        assert_eq!(resolve_bedrock_target(&arn, &options).region, "eu-west-3");
        assert_eq!(
            standard_bedrock_endpoint_region(
                "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com"
            ),
            Some("us-gov-west-1".to_owned())
        );
        let endpoint_model = test_model("https://bedrock-runtime.eu-central-1.amazonaws.com");
        let endpoint_options = BedrockOptions {
            environment: isolated,
            ..BedrockOptions::default()
        };
        let target = resolve_bedrock_target(&endpoint_model, &endpoint_options);
        assert_eq!(target.region, "eu-central-1");
        assert_eq!(
            target.endpoint,
            "https://bedrock-runtime.eu-central-1.amazonaws.com"
        );

        assert!(bedrock_supports_prompt_caching(
            &test_model(""),
            &BTreeMap::new()
        ));
        let nova = llm::Model {
            id: "amazon.nova-pro-v1:0".to_owned(),
            name: "Nova Pro".to_owned(),
            ..llm::Model::default()
        };
        assert!(!bedrock_supports_prompt_caching(&nova, &BTreeMap::new()));
        assert!(bedrock_supports_adaptive_thinking(&llm::Model {
            id: "anthropic.claude-opus-4-7".to_owned(),
            name: "Claude Opus 4.7".to_owned(),
            ..llm::Model::default()
        }));
        assert!(bedrock_supports_native_xhigh(&llm::Model {
            id: "anthropic.claude-sonnet-5".to_owned(),
            ..llm::Model::default()
        }));
    }

    #[test]
    fn request_conversion_handles_messages_tools_caching_and_thinking() {
        let mut model = test_model("");
        model.reasoning = true;
        let context = llm::Context {
            system_prompt: "be brief".to_owned(),
            messages: vec![
                llm::Message::User(llm::UserMessage::text("q", 1)),
                llm::Message::Assistant(Box::new(llm::AssistantMessage {
                    api: model.api.clone(),
                    provider: model.provider.clone(),
                    model: model.id.clone(),
                    stop_reason: stream::STOP_TOOL_USE.to_owned(),
                    timestamp: 2,
                    content: vec![
                        llm::ContentBlock::Thinking(llm::ThinkingContent {
                            thinking: "hmm".to_owned(),
                            thinking_signature: "sig".to_owned(),
                            ..llm::ThinkingContent::default()
                        }),
                        llm::ContentBlock::ToolCall(llm::ToolCall {
                            id: "call_1".to_owned(),
                            name: "weather".to_owned(),
                            arguments: BTreeMap::from([
                                ("".to_owned(), json!("drop")),
                                ("nested".to_owned(), json!({"": "drop", "keep": true})),
                            ]),
                            ..llm::ToolCall::default()
                        }),
                    ],
                    ..llm::AssistantMessage::default()
                })),
                llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                    tool_call_id: "call_1".to_owned(),
                    tool_name: "weather".to_owned(),
                    content: vec![llm::ContentBlock::text("sunny")],
                    timestamp: 3,
                    ..llm::ToolResultMessage::default()
                })),
            ],
            tools: vec![llm::Tool {
                name: "weather".to_owned(),
                description: "Get weather".to_owned(),
                parameters: json!({"type": "object"}),
                constrained_sampling: Some(json!({"type": "json_schema", "strict": "prefer"})),
            }],
        };
        let mut options = test_options();
        options.max_tokens = Some(1_024);
        options.temperature = Some(0.3);
        options.tool_choice = BedrockToolChoice::Any;
        options.compat = Some(BedrockCompat {
            supports_strict_mode: Some(true),
        });
        options.reasoning = Some(llm::THINKING_MEDIUM.to_owned());
        options.request_metadata = BTreeMap::from([("cost-center".to_owned(), "test".to_owned())]);
        options
            .sampling_params
            .insert("customSampling".to_owned(), json!(0.7));
        let params = build_bedrock_params(&model, &context, &options).expect("params");
        assert_eq!(params["inferenceConfig"]["maxTokens"], 1_024);
        assert_eq!(params["inferenceConfig"]["temperature"], 0.3);
        assert_eq!(params["customSampling"], 0.7);
        assert_eq!(params["system"][0]["text"], "be brief");
        assert_eq!(params["system"][1]["cachePoint"]["type"], "default");
        assert_eq!(params["messages"].as_array().expect("messages").len(), 3);
        assert_eq!(
            params["messages"][1]["content"][0]["reasoningContent"]["reasoningText"]["signature"],
            "sig"
        );
        assert!(
            params["messages"][1]["content"][1]["toolUse"]["input"]
                .get("")
                .is_none()
        );
        assert_eq!(
            params["messages"][2]["content"][0]["toolResult"]["status"],
            "success"
        );
        assert!(
            params["messages"][2]["content"]
                .as_array()
                .expect("content")
                .last()
                .expect("cache point")
                .get("cachePoint")
                .is_some()
        );
        assert_eq!(params["toolConfig"]["toolChoice"]["any"], json!({}));
        assert_eq!(params["toolConfig"]["tools"][0]["toolSpec"]["strict"], true);
        assert_eq!(
            params["additionalModelRequestFields"]["thinking"]["budget_tokens"],
            8_192
        );
    }

    #[test]
    fn conversion_protects_blank_text_images_and_unsigned_thinking() {
        let image = bedrock_image_block("image/png", "AAAA").expect("valid image");
        assert_eq!(image["image"]["format"], "png");
        assert!(bedrock_image_block("image/tiff", "AAAA").is_err());
        assert!(bedrock_image_block("image/png", "not base64!!").is_err());
        assert_eq!(
            bedrock_required_text_block("   ")["text"],
            BEDROCK_EMPTY_TEXT_PLACEHOLDER
        );

        let model = test_model("");
        let context = llm::Context {
            messages: vec![
                llm::Message::User(llm::UserMessage::text("q", 1)),
                llm::Message::Assistant(Box::new(llm::AssistantMessage {
                    api: model.api.clone(),
                    provider: model.provider.clone(),
                    model: model.id.clone(),
                    timestamp: 2,
                    content: vec![llm::ContentBlock::Thinking(llm::ThinkingContent {
                        thinking: "unsigned".to_owned(),
                        ..llm::ThinkingContent::default()
                    })],
                    ..llm::AssistantMessage::default()
                })),
            ],
            ..llm::Context::default()
        };
        let params = build_bedrock_params(&model, &context, &test_options()).expect("params");
        assert_eq!(params["messages"][1]["content"][0]["text"], "unsigned");
        assert!(
            params["messages"][1]["content"][0]
                .get("reasoningContent")
                .is_none()
        );

        let non_claude = llm::Model {
            id: "amazon.nova-pro-v1:0".to_owned(),
            name: "Nova Pro".to_owned(),
            ..model.clone()
        };
        let mut non_claude_context = context.clone();
        if let llm::Message::Assistant(assistant) = &mut non_claude_context.messages[1] {
            assistant.api = non_claude.api.clone();
            assistant.provider = non_claude.provider.clone();
            assistant.model = non_claude.id.clone();
            if let llm::ContentBlock::Thinking(thinking) = &mut assistant.content[0] {
                thinking.thinking_signature = "ignored".to_owned();
            }
        }
        let params = build_bedrock_params(&non_claude, &non_claude_context, &test_options())
            .expect("params");
        assert_eq!(
            params["messages"][1]["content"][0]["reasoningContent"]["reasoningText"]["text"],
            "unsigned"
        );
        assert!(
            params["messages"][1]["content"][0]["reasoningContent"]["reasoningText"]
                .get("signature")
                .is_none()
        );
    }

    #[test]
    fn adaptive_and_govcloud_thinking_fields_follow_model_rules() {
        let mut adaptive = test_model("");
        adaptive.id = "anthropic.claude-opus-4-7".to_owned();
        adaptive.name = "Claude Opus 4.7".to_owned();
        adaptive.reasoning = true;
        let mut options = test_options();
        options.reasoning = Some(llm::THINKING_XHIGH.to_owned());
        let params =
            build_bedrock_params(&adaptive, &llm::Context::default(), &options).expect("params");
        assert_eq!(
            params["additionalModelRequestFields"]["thinking"]["type"],
            "adaptive"
        );
        assert_eq!(
            params["additionalModelRequestFields"]["output_config"]["effort"],
            "xhigh"
        );
        assert!(
            params["additionalModelRequestFields"]
                .get("anthropic_beta")
                .is_none()
        );

        let mut budget = test_model("");
        budget.reasoning = true;
        let mut options = test_options();
        options.reasoning = Some(llm::THINKING_MEDIUM.to_owned());
        options.region = Some("us-gov-west-1".to_owned());
        let params =
            build_bedrock_params(&budget, &llm::Context::default(), &options).expect("params");
        assert!(
            params["additionalModelRequestFields"]["thinking"]
                .get("display")
                .is_none()
        );
        assert_eq!(
            params["additionalModelRequestFields"]["anthropic_beta"][0],
            ANTHROPIC_INTERLEAVED_THINKING_BETA
        );
    }

    #[test]
    fn stream_conversion_emits_text_tool_and_reasoning_events() {
        let tool_then_reasoning = stream_frames([
            event_frame("messageStart", r#"{"role":"assistant"}"#),
            event_frame(
                "contentBlockStart",
                r#"{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call_1","name":"weather"}}}"#,
            ),
            event_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"location\":"}}}"#,
            ),
            event_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"Paris\"}"}}}"#,
            ),
            event_frame("contentBlockStop", r#"{"contentBlockIndex":0}"#),
            event_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":1,"delta":{"reasoningContent":{"text":"thinking"}}}"#,
            ),
            event_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":1,"delta":{"reasoningContent":{"signature":"sig"}}}"#,
            ),
            event_frame("contentBlockStop", r#"{"contentBlockIndex":1}"#),
            event_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":2,"delta":{"text":"answer"}}"#,
            ),
            event_frame("contentBlockStop", r#"{"contentBlockIndex":2}"#),
            event_frame("messageStop", r#"{"stopReason":"tool_use"}"#),
        ]);
        let (message, events) = consume_fixture(tool_then_reasoning);
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                stream::EVENT_START,
                stream::EVENT_TOOLCALL_START,
                stream::EVENT_TOOLCALL_DELTA,
                stream::EVENT_TOOLCALL_DELTA,
                stream::EVENT_TOOLCALL_END,
                stream::EVENT_THINKING_START,
                stream::EVENT_THINKING_DELTA,
                stream::EVENT_THINKING_END,
                stream::EVENT_TEXT_START,
                stream::EVENT_TEXT_DELTA,
                stream::EVENT_TEXT_END,
            ]
        );
        assert_eq!(message.stop_reason, stream::STOP_TOOL_USE);
        match &message.content[0] {
            llm::ContentBlock::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "call_1");
                assert_eq!(tool_call.arguments["location"], "Paris");
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        match &message.content[1] {
            llm::ContentBlock::Thinking(thinking) => {
                assert_eq!(thinking.thinking, "thinking");
                assert_eq!(thinking.thinking_signature, "sig");
            }
            other => panic!("expected thinking, got {other:?}"),
        }
        assert_eq!(message.content[2].plain_text(), Some("answer"));
    }

    #[test]
    fn stream_exception_stop_and_data_retention_errors_match_go() {
        let exception = encode_event_stream_message(
            &BTreeMap::from([
                (":message-type".to_owned(), "exception".to_owned()),
                (
                    ":exception-type".to_owned(),
                    "ThrottlingException".to_owned(),
                ),
            ]),
            br#"{"message":"Too many requests"}"#,
        )
        .expect("exception fixture");
        let model = test_model("");
        let events = stream::AssistantMessageEventStream::new();
        let mut converter =
            BedrockStreamer::new(model, llm::Context::default(), test_options(), events);
        let error = converter
            .consume_stream(Cursor::new(exception))
            .expect_err("exception should fail");
        assert!(
            error
                .to_string()
                .contains("Throttling error: Too many requests")
        );

        let retention = bedrock_exception_error(
            "ValidationException",
            br#"{"message":"data retention mode is unavailable"}"#,
        );
        assert!(
            retention
                .to_string()
                .contains(BEDROCK_DATA_RETENTION_DOCS_URL)
        );
        assert_eq!(
            map_bedrock_stop_reason("max_tokens"),
            (stream::STOP_LENGTH.to_owned(), None)
        );
        assert_eq!(
            map_bedrock_stop_reason("guardrail_intervened"),
            (
                stream::STOP_ERROR.to_owned(),
                Some("Provider stopped with: guardrail_intervened".to_owned())
            )
        );
    }

    #[test]
    fn building_requests_uses_bearer_or_sigv4_and_protects_reserved_headers() {
        let model = test_model("https://bedrock.internal.example.com/proxy");
        let params = build_bedrock_params(
            &model,
            &llm::Context {
                messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
                ..llm::Context::default()
            },
            &test_options(),
        )
        .expect("params");
        let now = OffsetDateTime::from_unix_timestamp(1_767_323_045).expect("time");

        let mut bearer = test_options();
        bearer.bearer_token = Some("bedrock-api-key".to_owned());
        let request =
            build_bedrock_http_request(&model, &bearer, &params, now).expect("bearer request");
        assert_eq!(request.headers["Authorization"], "Bearer bedrock-api-key");
        assert!(!request.headers.contains_key("X-Amz-Date"));
        assert!(
            request.url.path().ends_with(
                "/proxy/model/anthropic.claude-sonnet-4-5-20250929-v1:0/converse-stream"
            )
        );

        let mut signed = test_options();
        signed.headers = BTreeMap::from([
            (
                "x-custom-header".to_owned(),
                Some("custom-value".to_owned()),
            ),
            ("x-amz-date".to_owned(), Some("override".to_owned())),
            ("authorization".to_owned(), Some("override".to_owned())),
        ]);
        let request =
            build_bedrock_http_request(&model, &signed, &params, now).expect("signed request");
        assert_eq!(request.headers["x-custom-header"], "custom-value");
        assert_ne!(request.headers["Authorization"], "override");
        assert!(request.headers["Authorization"].contains("x-custom-header"));
        assert!(is_bedrock_reserved_header("X-Amz-Date"));
        assert!(is_bedrock_reserved_header("host"));
        assert!(!is_bedrock_reserved_header("content-type"));
    }

    #[test]
    fn network_streams_a_signed_text_turn_and_exposes_usage() {
        let (endpoint, captured, server) =
            serve_responses(vec![(200, text_turn("Hello from Bedrock"))]);
        let model = test_model(endpoint);
        let context = llm::Context {
            messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
            ..llm::Context::default()
        };
        let events = stream_bedrock(model, context, test_options())
            .iter()
            .collect::<Vec<_>>();
        server.join().expect("server thread");
        assert_eq!(
            events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                stream::EVENT_START,
                stream::EVENT_TEXT_START,
                stream::EVENT_TEXT_DELTA,
                stream::EVENT_TEXT_END,
                stream::EVENT_DONE,
            ]
        );
        let final_message = events
            .last()
            .and_then(|event| event.message.as_ref())
            .expect("done message");
        assert_eq!(final_message.stop_reason, stream::STOP_STOP);
        assert_eq!(
            final_message.content[0].plain_text(),
            Some("Hello from Bedrock")
        );
        assert_eq!(final_message.usage.input, 10);
        assert_eq!(final_message.usage.output, 4);
        let captured = captured.lock().expect("captures");
        assert!(captured[0].path.ends_with("/converse-stream"));
        assert!(
            captured[0]
                .headers
                .get("authorization")
                .is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256 "))
        );
    }

    #[test]
    fn network_retries_retryable_http_errors_and_simple_budget_leaves_answer_room() {
        let (endpoint, captured, server) = serve_responses(vec![
            (503, br#"{"message":"try again"}"#.to_vec()),
            (200, text_turn("ok")),
        ]);
        let mut model = test_model(endpoint);
        model.reasoning = true;
        let mut request = test_options();
        request.max_retries = 1;
        let simple = BedrockSimpleOptions {
            request,
            reasoning: Some(llm::THINKING_MEDIUM.to_owned()),
            ..BedrockSimpleOptions::default()
        };
        let events = stream_bedrock_simple(
            model,
            llm::Context {
                messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
                ..llm::Context::default()
            },
            simple,
        )
        .iter()
        .collect::<Vec<_>>();
        server.join().expect("server thread");
        assert_eq!(
            events.last().expect("terminal event").event_type,
            stream::EVENT_DONE
        );
        let captures = captured.lock().expect("captures");
        assert_eq!(captures.len(), 2);
        let request: Value =
            serde_json::from_slice(&captures[1].body).expect("second request body JSON");
        let budget = request["additionalModelRequestFields"]["thinking"]["budget_tokens"]
            .as_u64()
            .expect("budget");
        let max_tokens = request["inferenceConfig"]["maxTokens"]
            .as_u64()
            .expect("max tokens");
        assert!(budget < max_tokens);
        assert_eq!(max_tokens.saturating_sub(budget), stream::MIN_ANSWER_TOKENS);
    }

    #[test]
    fn transform_normalizes_cross_model_tool_flow_and_unsupported_images() {
        let target = test_model("");
        let source_assistant = llm::AssistantMessage {
            api: "other-api".to_owned(),
            provider: "other-provider".to_owned(),
            model: "other-model".to_owned(),
            timestamp: 2,
            content: vec![
                llm::ContentBlock::Thinking(llm::ThinkingContent {
                    thinking: "cross-model thought".to_owned(),
                    thinking_signature: "must-not-leak".to_owned(),
                    ..llm::ThinkingContent::default()
                }),
                llm::ContentBlock::ToolCall(llm::ToolCall {
                    id: "call/with spaces!".to_owned(),
                    name: "weather".to_owned(),
                    thought_signature: "must-not-leak".to_owned(),
                    ..llm::ToolCall::default()
                }),
            ],
            ..llm::AssistantMessage::default()
        };
        let transformed = transform_bedrock_messages(
            &[
                llm::Message::User(llm::UserMessage {
                    content: llm::UserContent::Blocks(vec![
                        llm::ContentBlock::Image(llm::ImageContent {
                            mime_type: "image/png".to_owned(),
                            data: "AAAA".to_owned(),
                        }),
                        llm::ContentBlock::Image(llm::ImageContent {
                            mime_type: "image/png".to_owned(),
                            data: "AAAA".to_owned(),
                        }),
                    ]),
                    timestamp: 1,
                    ..llm::UserMessage::text("", 1)
                }),
                llm::Message::Assistant(Box::new(source_assistant)),
                llm::Message::User(llm::UserMessage::text("next user turn", 3)),
            ],
            &target,
        );
        assert_eq!(transformed.len(), 4);
        match &transformed[0] {
            llm::Message::User(user) => assert_eq!(
                user.content.blocks()[0].plain_text(),
                Some("(image omitted: model does not support images)")
            ),
            other => panic!("expected transformed user, got {other:?}"),
        }
        match &transformed[1] {
            llm::Message::Assistant(assistant) => {
                assert_eq!(
                    assistant.content[0].plain_text(),
                    Some("cross-model thought")
                );
                match &assistant.content[1] {
                    llm::ContentBlock::ToolCall(tool_call) => {
                        assert_eq!(tool_call.id, "call_with_spaces_");
                        assert!(tool_call.thought_signature.is_empty());
                    }
                    other => panic!("expected normalized tool call, got {other:?}"),
                }
            }
            other => panic!("expected transformed assistant, got {other:?}"),
        }
        match &transformed[2] {
            llm::Message::ToolResult(result) => {
                assert_eq!(result.tool_call_id, "call_with_spaces_");
                assert!(result.is_error);
                assert_eq!(result.content[0].plain_text(), Some("No result provided"));
            }
            other => panic!("expected synthesized result, got {other:?}"),
        }
    }

    #[test]
    fn tool_choice_and_strict_requirement_follow_bedrock_contract() {
        let tool = llm::Tool {
            name: "strict-tool".to_owned(),
            description: "strict".to_owned(),
            parameters: json!({"type": "object"}),
            constrained_sampling: Some(json!({"type": "json_schema", "strict": "require"})),
        };
        assert!(
            convert_bedrock_tool_config(&[tool.clone()], &BedrockToolChoice::None, false)
                .expect("none omits tools")
                .is_none()
        );
        assert!(convert_bedrock_tool_config(&[tool], &BedrockToolChoice::Auto, false).is_err());
    }

    #[test]
    fn terminal_stream_errors_include_protocol_and_http_context() {
        let incomplete = stream_frames([
            event_frame("messageStart", r#"{"role":"assistant"}"#),
            event_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"text":"partial"}}"#,
            ),
        ]);
        let (endpoint, _, server) = serve_responses(vec![(200, incomplete)]);
        let events = stream_bedrock(
            test_model(endpoint),
            llm::Context {
                messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
                ..llm::Context::default()
            },
            test_options(),
        )
        .iter()
        .collect::<Vec<_>>();
        server.join().expect("server thread");
        let error = events
            .last()
            .and_then(|event| event.error.as_ref())
            .expect("error");
        assert_eq!(
            events.last().expect("event").event_type,
            stream::EVENT_ERROR
        );
        assert!(error.error_message.contains("without a stop reason"));

        let (endpoint, _, server) =
            serve_responses(vec![(403, br#"{"message":"not authorized"}"#.to_vec())]);
        let events = stream_bedrock(
            test_model(endpoint),
            llm::Context {
                messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
                ..llm::Context::default()
            },
            test_options(),
        )
        .iter()
        .collect::<Vec<_>>();
        server.join().expect("server thread");
        let error = events
            .last()
            .and_then(|event| event.error.as_ref())
            .expect("HTTP error");
        assert!(error.error_message.contains("not authorized"));
    }

    #[test]
    fn environment_bearer_sentinel_session_and_cancellation_behave_as_expected() {
        let model = test_model("");
        let params = build_bedrock_params(&model, &llm::Context::default(), &test_options())
            .expect("params");
        let now = OffsetDateTime::from_unix_timestamp(1_767_323_045).expect("time");
        let bearer = BedrockOptions {
            environment: BTreeMap::from([
                (
                    "AWS_BEARER_TOKEN_BEDROCK".to_owned(),
                    "env-token".to_owned(),
                ),
                ("AWS_REGION".to_owned(), "us-east-1".to_owned()),
            ]),
            ..BedrockOptions::default()
        };
        let request =
            build_bedrock_http_request(&model, &bearer, &params, now).expect("env bearer");
        assert_eq!(request.headers["Authorization"], "Bearer env-token");

        let mut session = test_options();
        session
            .environment
            .insert("AWS_SESSION_TOKEN".to_owned(), "session-token".to_owned());
        session.api_key = Some("<authenticated>".to_owned());
        session.environment.insert(
            "AWS_BEARER_TOKEN_BEDROCK".to_owned(),
            "must-not-leak".to_owned(),
        );
        let request =
            build_bedrock_http_request(&model, &session, &params, now).expect("SigV4 request");
        assert_eq!(request.headers["X-Amz-Security-Token"], "session-token");
        assert!(request.headers["Authorization"].contains("x-amz-security-token"));
        assert!(!request.headers["Authorization"].contains("must-not-leak"));

        let cancellation = BedrockCancellation::default();
        cancellation.cancel();
        let cancelled = BedrockOptions {
            cancellation: Some(cancellation),
            ..BedrockOptions::default()
        };
        let events = stream_bedrock(model, llm::Context::default(), cancelled)
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(
            events.last().expect("terminal event").event_type,
            stream::EVENT_ERROR
        );
        let failure = events
            .last()
            .and_then(|event| event.error.as_ref())
            .expect("error message");
        assert_eq!(failure.stop_reason, stream::STOP_ABORTED);
    }

    #[test]
    fn payload_and_response_hooks_run_at_the_protocol_boundaries() {
        let (endpoint, captured, server) = serve_responses(vec![(200, text_turn("ok"))]);
        let payload_calls = Arc::new(Mutex::new(0_usize));
        let response_status = Arc::new(Mutex::new(None));
        let mut options = test_options();
        {
            let payload_calls = Arc::clone(&payload_calls);
            options.on_payload = Some(Arc::new(move |mut payload, _model| {
                *payload_calls.lock().expect("payload count") += 1;
                payload.insert("hooked".to_owned(), Value::Bool(true));
                Some(payload)
            }));
        }
        {
            let response_status = Arc::clone(&response_status);
            options.on_response = Some(Arc::new(move |response, _model| {
                *response_status.lock().expect("response status") = Some(response.status);
            }));
        }
        let events = stream_bedrock(
            test_model(endpoint),
            llm::Context {
                messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
                ..llm::Context::default()
            },
            options,
        )
        .iter()
        .collect::<Vec<_>>();
        server.join().expect("server thread");
        assert_eq!(
            events.last().expect("terminal event").event_type,
            stream::EVENT_DONE
        );
        assert_eq!(*payload_calls.lock().expect("payload count"), 1);
        assert_eq!(*response_status.lock().expect("response status"), Some(200));
        let payload: Value =
            serde_json::from_slice(&captured.lock().expect("captures")[0].body).expect("JSON");
        assert_eq!(payload["hooked"], true);
    }

    #[derive(Clone, Debug, Default)]
    struct CapturedRequest {
        path: String,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
    }

    fn serve_responses(
        responses: Vec<(u16, Vec<u8>)>,
    ) -> (String, Arc<Mutex<Vec<CapturedRequest>>>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let endpoint = format!("http://{}", listener.local_addr().expect("address"));
        let captured = Arc::new(Mutex::new(Vec::new()));
        let captures = Arc::clone(&captured);
        let server = thread::spawn(move || {
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().expect("connection");
                captures
                    .lock()
                    .expect("captures")
                    .push(read_captured_request(&mut socket));
                let reason = match status {
                    200 => "OK",
                    503 => "Service Unavailable",
                    _ => "Test Response",
                };
                let headers = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/vnd.amazon.eventstream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                socket
                    .write_all(headers.as_bytes())
                    .expect("response headers");
                socket.write_all(&body).expect("response body");
                socket.flush().expect("response flush");
            }
        });
        (endpoint, captured, server)
    }

    fn read_captured_request(socket: &mut TcpStream) -> CapturedRequest {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 4_096];
        let header_end = loop {
            let read = socket.read(&mut buffer).expect("request bytes");
            assert_ne!(read, 0, "request ended before headers");
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(position) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break position + 4;
            }
        };
        let header_text = String::from_utf8_lossy(&bytes[..header_end]);
        let mut lines = header_text.split("\r\n");
        let path = lines
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_owned();
        let mut headers = BTreeMap::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
            }
        }
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_default();
        let total = header_end + content_length;
        while bytes.len() < total {
            let read = socket.read(&mut buffer).expect("request body");
            assert_ne!(read, 0, "request ended before full body");
            bytes.extend_from_slice(&buffer[..read]);
        }
        CapturedRequest {
            path,
            headers,
            body: bytes[header_end..total].to_vec(),
        }
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "goshcoder-{label}-{}-{entropy}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    fn isolated_environment() -> BTreeMap<String, String> {
        let directory = temporary_directory("bedrock-isolated");
        let environment = BTreeMap::from([
            (
                "AWS_SHARED_CREDENTIALS_FILE".to_owned(),
                directory.join("missing-credentials").display().to_string(),
            ),
            (
                "AWS_CONFIG_FILE".to_owned(),
                directory.join("missing-config").display().to_string(),
            ),
        ]);
        // The paths need not persist: profile lookup only treats their absence
        // as "no configured profile". Remove the empty parent immediately.
        fs::remove_dir_all(directory).expect("remove isolated directory");
        environment
    }
}
