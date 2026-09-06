//! Blocking provider protocol adapters for the Rust agent runtime.
//!
//! This module deliberately has no dependency on an async runtime.  It turns
//! the streaming HTTP protocols into [`stream::AssistantMessageEventStream`]
//! events on a worker thread and exposes a synchronous
//! [`agent::AssistantResponder`] for the existing agent loop.  Add
//! `pub mod providers;` to the crate root when the command surface is ready to
//! wire a provider responder; this file is kept independent so it can be
//! compiled and exercised before that integration step.
//!
//! The implemented wire protocols are:
//! - OpenAI Chat Completions (`openai-completions`)
//! - OpenAI Responses (`openai-responses`)
//! - Azure OpenAI Responses (`azure-openai-responses`)
//! - OpenAI Codex Responses (`openai-codex-responses`)
//! - Anthropic Messages (`anthropic-messages`)
//! - Google Generative AI (`google-generative-ai`)
//! - Google Vertex AI (`google-vertex`)
//! - Mistral Conversations (`mistral-conversations`)
//!
//! The implementation shares the existing SSE framing, bounded incremental
//! JSON parser, retry classification, token accounting, and normalized
//! `llm::AssistantMessage` types rather than introducing protocol-local
//! equivalents.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    error::Error,
    fmt,
    io::{self, Read},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::{
    blocking::{Client, Response},
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde_json::{Map, Value, json};
use url::Url;

use crate::{
    agent, aperture, bedrock, catalog, google_auth, llm, mistral, oauth, omni_prompt_tools,
    omniroute, stream,
};

pub const API_OPENAI_COMPLETIONS: &str = "openai-completions";
pub const API_OPENAI_RESPONSES: &str = "openai-responses";
pub const API_AZURE_OPENAI_RESPONSES: &str = "azure-openai-responses";
pub const API_OPENAI_CODEX_RESPONSES: &str = "openai-codex-responses";
pub const API_ANTHROPIC_MESSAGES: &str = "anthropic-messages";
pub const API_GOOGLE_GENERATIVE_AI: &str = "google-generative-ai";
pub const API_GOOGLE_VERTEX: &str = "google-vertex";
pub const API_MISTRAL_CONVERSATIONS: &str = "mistral-conversations";

const DEFAULT_AZURE_OPENAI_API_VERSION: &str = "v1";
const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const DEFAULT_GOOGLE_GENERATIVE_AI_BASE_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_VERTEX_API_VERSION: &str = "v1";
const DEFAULT_MISTRAL_RESPONSE_HEADER_TIMEOUT: Duration = Duration::from_secs(60);
const VERTEX_AMBIENT_CREDENTIALS_MARKER: &str = "gcp-vertex-credentials";
const CODEX_JWT_AUTH_CLAIM: &str = "https://api.openai.com/auth";
const AZURE_MANAGED_HOST_SUFFIXES: &[&str] = &[
    ".openai.azure.com",
    ".cognitiveservices.azure.com",
    ".ai.azure.com",
];
static GOOGLE_TOOL_CALL_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// The maximum request attempts after the initial request that callers may
/// configure.  A bounded value prevents a bad configuration from creating an
/// unbounded agent turn.
pub const MAX_REQUEST_RETRIES: u32 = 8;
pub const DEFAULT_EVENT_BUFFER_CAPACITY: usize = 1_024;
pub const DEFAULT_MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
/// Characters of a response error body kept in an error message, as pi's
/// `MAX_PROVIDER_ERROR_BODY_CHARS`.
pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4_000;
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Default idle deadline; see [`ProviderConfig::read_timeout`].
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(300);
/// How often a blocked producer or consumer re-checks cancellation.
const STREAM_POLL_INTERVAL: Duration = Duration::from_millis(25);
/// How long a terminal event may wait for a slow consumer before the worker
/// gives up on delivering it, and how long a cancelled consumer waits for
/// the worker's own aborted partial before synthesizing one.
const TERMINAL_DELIVERY_BUDGET: Duration = Duration::from_secs(1);
/// Body chunks buffered between the socket reader thread and the stream
/// worker before the reader applies backpressure.
const BODY_CHUNK_QUEUE: usize = 64;

/// The supported provider wire protocol chosen from `llm::Model::api`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderProtocol {
    OpenAiCompletions,
    OpenAiResponses,
    AzureOpenAiResponses,
    OpenAiCodexResponses,
    AnthropicMessages,
    GoogleGenerativeAi,
    GoogleVertex,
    MistralConversations,
    BedrockConverseStream,
}

/// Whether the live responder can serve a model with this `api`. The
/// prompt-emulated OmniRoute protocol is layered over chat completions rather
/// than being a wire protocol of its own, so it is not a [`ProviderProtocol`].
pub fn supports_api(api: &str) -> bool {
    api == omniroute::PROMPT_TOOLS_API || ProviderProtocol::from_api(api).is_ok()
}

impl ProviderProtocol {
    pub fn from_api(api: &str) -> Result<Self> {
        match api {
            API_OPENAI_COMPLETIONS => Ok(Self::OpenAiCompletions),
            API_OPENAI_RESPONSES => Ok(Self::OpenAiResponses),
            API_AZURE_OPENAI_RESPONSES => Ok(Self::AzureOpenAiResponses),
            API_OPENAI_CODEX_RESPONSES => Ok(Self::OpenAiCodexResponses),
            API_ANTHROPIC_MESSAGES => Ok(Self::AnthropicMessages),
            API_GOOGLE_GENERATIVE_AI => Ok(Self::GoogleGenerativeAi),
            API_GOOGLE_VERTEX => Ok(Self::GoogleVertex),
            API_MISTRAL_CONVERSATIONS => Ok(Self::MistralConversations),
            bedrock::API_BEDROCK_CONVERSE_STREAM => Ok(Self::BedrockConverseStream),
            other => Err(ProviderAdapterError::UnsupportedApi(other.to_owned())),
        }
    }

    fn endpoint_suffix(self) -> &'static str {
        match self {
            Self::OpenAiCompletions => "chat/completions",
            Self::OpenAiResponses => "responses",
            Self::AzureOpenAiResponses => "responses",
            Self::OpenAiCodexResponses => "codex/responses",
            Self::AnthropicMessages => "v1/messages",
            Self::MistralConversations => "v1/chat/completions",
            Self::GoogleGenerativeAi => {
                unreachable!("Google uses a model-scoped GenerateContent endpoint")
            }
            Self::GoogleVertex => {
                unreachable!("Vertex uses a model-scoped GenerateContent endpoint")
            }
            Self::BedrockConverseStream => {
                unreachable!("Bedrock uses its own signed request builder")
            }
        }
    }
}

/// Errors from request construction, transport, or protocol decoding.
///
/// API keys and header values are intentionally not included in any display
/// message generated by this type.
#[derive(Debug)]
pub enum ProviderAdapterError {
    UnsupportedApi(String),
    MissingBaseUrl,
    InvalidBaseUrl,
    InvalidHeaderName(String),
    InvalidHeaderValue(String),
    MissingCredential { provider: String },
    MissingApiKey { provider: String },
    AmbientCredentialsUnsupported { provider: String },
    AzureBaseUrlRequired,
    InvalidCodexToken,
    InvalidConfiguration(&'static str),
    Request(reqwest::Error),
    Provider(stream::ProviderError),
    Sse(stream::SseError),
    Json(serde_json::Error),
    Protocol(String),
    Cancelled,
    EventStream(String),
    StreamClosed,
}

impl fmt::Display for ProviderAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedApi(api) => {
                write!(formatter, "provider API {api:?} is not implemented")
            }
            Self::MissingBaseUrl => formatter.write_str("provider model has no base URL"),
            Self::InvalidBaseUrl => formatter.write_str("provider model has an invalid base URL"),
            Self::InvalidHeaderName(name) => write!(
                formatter,
                "provider configured an invalid header name {name:?}"
            ),
            Self::InvalidHeaderValue(name) => {
                write!(
                    formatter,
                    "provider configured an invalid value for header {name:?}"
                )
            }
            Self::MissingCredential { provider } => {
                write!(
                    formatter,
                    "no API key or authorization header for provider {provider:?}"
                )
            }
            Self::MissingApiKey { provider } => {
                write!(formatter, "no API key for provider {provider:?}")
            }
            Self::AmbientCredentialsUnsupported { provider } => write!(
                formatter,
                "provider {provider:?} requires ambient credentials, which this HTTP adapter does not implement"
            ),
            Self::AzureBaseUrlRequired => formatter.write_str(
                "Azure OpenAI base URL is required; set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or configure model.baseUrl",
            ),
            Self::InvalidCodexToken => formatter.write_str("failed to extract accountId from token"),
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::Request(error) => write!(formatter, "provider request failed: {error}"),
            Self::Provider(error) => error.fmt(formatter),
            Self::Sse(error) => error.fmt(formatter),
            Self::Json(error) => write!(formatter, "provider sent invalid JSON: {error}"),
            Self::Protocol(message) => write!(formatter, "provider protocol error: {message}"),
            Self::Cancelled => formatter.write_str("request aborted"),
            Self::EventStream(message) => {
                write!(formatter, "assistant event stream error: {message}")
            }
            Self::StreamClosed => {
                formatter.write_str("assistant stream ended without a terminal result")
            }
        }
    }
}

impl Error for ProviderAdapterError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Request(error) => Some(error),
            Self::Provider(error) => Some(error),
            Self::Sse(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::UnsupportedApi(_)
            | Self::MissingBaseUrl
            | Self::InvalidBaseUrl
            | Self::InvalidHeaderName(_)
            | Self::InvalidHeaderValue(_)
            | Self::MissingCredential { .. }
            | Self::MissingApiKey { .. }
            | Self::AmbientCredentialsUnsupported { .. }
            | Self::AzureBaseUrlRequired
            | Self::InvalidCodexToken
            | Self::InvalidConfiguration(_)
            | Self::Protocol(_)
            | Self::Cancelled
            | Self::EventStream(_)
            | Self::StreamClosed => None,
        }
    }
}

impl From<reqwest::Error> for ProviderAdapterError {
    fn from(error: reqwest::Error) -> Self {
        Self::Request(error)
    }
}

impl From<serde_json::Error> for ProviderAdapterError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<stream::SseError> for ProviderAdapterError {
    fn from(error: stream::SseError) -> Self {
        match error {
            // `CancellableBody` reports cancellation as a read error carrying
            // the abort marker; it is the turn's cancellation, not a fault.
            stream::SseError::Io(error)
                if error.get_ref().is_some_and(|inner| {
                    inner.downcast_ref::<stream::RequestAborted>().is_some()
                }) =>
            {
                Self::Cancelled
            }
            other => Self::Sse(other),
        }
    }
}

pub type Result<T> = std::result::Result<T, ProviderAdapterError>;

/// Authentication and explicit header overrides for one provider responder.
///
/// `None` header values deliberately suppress protocol defaults.  This keeps
/// catalog-derived authentication such as `Authorization: Bearer …` plus a
/// suppressed `x-api-key` intact for Anthropic-compatible providers.
#[derive(Clone, Default)]
pub struct ProviderCredentials {
    api_key: Option<String>,
    headers: BTreeMap<String, Option<String>>,
    environment: BTreeMap<String, String>,
}

impl ProviderCredentials {
    pub fn api_key(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            headers: BTreeMap::new(),
            environment: BTreeMap::new(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        set_header_override(&mut self.headers, name.into(), Some(value.into()));
        self
    }

    /// Suppresses a protocol default header case-insensitively.
    pub fn without_header(mut self, name: impl Into<String>) -> Self {
        set_header_override(&mut self.headers, name.into(), None);
        self
    }

    pub fn api_key_value(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn headers(&self) -> &BTreeMap<String, Option<String>> {
        &self.headers
    }

    /// Copies catalog-resolved auth without exposing it through debug output.
    pub fn from_resolved_model(resolved: &catalog::ResolvedModel) -> Self {
        Self {
            api_key: resolved.auth().api_key().map(str::to_owned),
            headers: resolved.effective_headers().clone(),
            environment: resolved.auth().environment().clone(),
        }
    }
}

/// Transport policy for a provider responder.
#[derive(Clone, Debug)]
pub struct ProviderConfig {
    /// Retries after the initial HTTP request.  Stream decoding is never
    /// retried because replaying a partially consumed completion is unsafe.
    pub max_retries: u32,
    /// Maximum accepted server-directed retry delay.
    pub retry_delay_limit: stream::RetryDelayLimit,
    /// Deadline for establishing a connection, including TLS.
    pub connect_timeout: Option<Duration>,
    /// Idle deadline: it bounds the wait for response headers and then every
    /// individual body read, re-arming after each read.
    ///
    /// Streaming completions legitimately run for many minutes, so no
    /// whole-request deadline is ever applied; only a silent connection
    /// times out. A caller-provided client retains its own timeout policy.
    pub read_timeout: Option<Duration>,
    /// Deadline for Mistral to return HTTP response headers.
    ///
    /// This bounds time to first byte without truncating an active SSE
    /// response. `None` disables the deadline.
    pub mistral_response_header_timeout: Option<Duration>,
    /// Capacity of the externally visible normalized event stream.
    pub event_buffer_capacity: usize,
    /// Maximum response-error body read from a failed request; the text
    /// kept in the error message is further capped at
    /// [`MAX_PROVIDER_ERROR_BODY_CHARS`].
    pub max_error_body_bytes: usize,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_delay_limit: stream::RetryDelayLimit::Default,
            connect_timeout: Some(DEFAULT_CONNECT_TIMEOUT),
            read_timeout: Some(DEFAULT_READ_TIMEOUT),
            mistral_response_header_timeout: Some(DEFAULT_MISTRAL_RESPONSE_HEADER_TIMEOUT),
            event_buffer_capacity: DEFAULT_EVENT_BUFFER_CAPACITY,
            max_error_body_bytes: DEFAULT_MAX_ERROR_BODY_BYTES,
        }
    }
}

impl ProviderConfig {
    fn validate(&self) -> Result<()> {
        if self.max_retries > MAX_REQUEST_RETRIES {
            return Err(ProviderAdapterError::InvalidConfiguration(
                "provider max_retries exceeds MAX_REQUEST_RETRIES",
            ));
        }
        if self.event_buffer_capacity == 0 {
            return Err(ProviderAdapterError::InvalidConfiguration(
                "provider event buffer capacity must be greater than zero",
            ));
        }
        if self.max_error_body_bytes == 0 {
            return Err(ProviderAdapterError::InvalidConfiguration(
                "provider max error body bytes must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// A reusable blocking HTTP provider factory.
///
/// It is cloneable and safe to share between agent turns because reqwest's
/// blocking client owns a connection pool and the per-request state is moved
/// into a dedicated stream worker.
#[derive(Clone)]
pub struct ProviderResponderFactory {
    client: Client,
    credentials: ProviderCredentials,
    config: ProviderConfig,
}

impl ProviderResponderFactory {
    /// Builds a factory using an API key and the default bounded policy.
    pub fn new(api_key: impl Into<String>) -> Result<Self> {
        Self::configured(
            ProviderCredentials::api_key(api_key),
            ProviderConfig::default(),
        )
    }

    /// Builds a factory with explicit credentials and policy.
    pub fn configured(credentials: ProviderCredentials, config: ProviderConfig) -> Result<Self> {
        config.validate()?;
        // The blocking client's own timeout bounds the wait for response
        // headers and then re-arms for every body read, which makes it an
        // idle deadline. A per-request timeout would instead become a total
        // deadline that cuts long streams off mid-response, so none is set.
        let mut builder = Client::builder()
            .user_agent(oauth::OAUTH_USER_AGENT)
            .timeout(config.read_timeout);
        if let Some(timeout) = config.connect_timeout {
            builder = builder.connect_timeout(timeout);
        }
        let client = builder.build().map_err(ProviderAdapterError::Request)?;
        Ok(Self {
            client,
            credentials,
            config,
        })
    }

    /// Uses a caller-owned reqwest client.  This is useful for proxies,
    /// certificates, or a custom timeout policy.
    pub fn with_client(
        client: Client,
        credentials: ProviderCredentials,
        config: ProviderConfig,
    ) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            client,
            credentials,
            config,
        })
    }

    /// Creates a factory from a catalog result, preserving API-key/header
    /// precedence resolved by `src/catalog.rs`.
    pub fn from_resolved_model(
        resolved: &catalog::ResolvedModel,
        config: ProviderConfig,
    ) -> Result<Self> {
        Self::configured(ProviderCredentials::from_resolved_model(resolved), config)
    }

    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Returns a responder matching `agent::AssistantResponder`.
    ///
    /// Provider errors are returned as normalized assistant error messages so
    /// the agent retains the selected API/provider/model metadata.
    pub fn assistant_responder(&self) -> agent::AssistantResponder {
        let factory = self.clone();
        Arc::new(move |model, context, options| {
            factory
                .respond(model, context, options)
                .map_err(|error| error.to_string())
        })
    }

    /// Starts an externally consumable normalized event stream.
    pub fn stream(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
    ) -> stream::AssistantMessageEventStream {
        self.stream_with_credentials(model, context, options, self.credentials.clone())
    }

    /// Runs a provider stream to a terminal normalized assistant message.
    ///
    /// This drains event delivery as it waits so the bounded stream cannot
    /// deadlock a synchronous `AssistantResponder` on a large completion.
    pub fn respond(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
    ) -> Result<llm::AssistantMessage> {
        self.respond_with_credentials(model, context, options, self.credentials.clone())
    }

    fn respond_with_credentials(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
        credentials: ProviderCredentials,
    ) -> Result<llm::AssistantMessage> {
        let assistant_event_listener = options.assistant_event_listener.clone();
        let cancellation = options.cancellation.clone();
        let events = self.stream_with_credentials(model, context, options, credentials);
        // Cancellation is honoured by polling rather than blocking on the
        // next event. The worker checks the token at every point it can
        // block, so it normally publishes the exact aborted partial itself;
        // the budget below only guards against a wedged worker.
        let mut abort_deadline = None;
        loop {
            match events.next_timeout(STREAM_POLL_INTERVAL) {
                Ok(Some(event)) => {
                    let terminal_message = event.terminal_message();
                    if let Some(listener) = &assistant_event_listener {
                        listener(event);
                    }
                    if let Some(message) = terminal_message {
                        return Ok((*message).clone());
                    }
                }
                Ok(None) => return Err(ProviderAdapterError::StreamClosed),
                Err(stream::EventStreamWaitError::TimedOut) => {
                    if !cancellation.is_cancelled() {
                        continue;
                    }
                    let deadline = *abort_deadline.get_or_insert_with(|| {
                        Instant::now()
                            .checked_add(TERMINAL_DELIVERY_BUDGET)
                            .unwrap_or_else(Instant::now)
                    });
                    if Instant::now() >= deadline {
                        return Ok(aborted_message(model));
                    }
                }
            }
        }
    }

    fn stream_with_credentials(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
        credentials: ProviderCredentials,
    ) -> stream::AssistantMessageEventStream {
        if model.api == bedrock::API_BEDROCK_CONVERSE_STREAM {
            return self.stream_bedrock_with_credentials(model, context, options, credentials);
        }
        let prompt_tools = model.api == omniroute::PROMPT_TOOLS_API;
        let events =
            stream::AssistantMessageEventStream::with_capacity(self.config.event_buffer_capacity)
                .expect("validated provider event buffer capacity");
        let worker_events = events.clone();
        let factory = self.clone();
        let model = model.clone();
        let context = context.clone();
        thread::spawn(move || {
            let mut emitter =
                MessageEmitter::new(worker_events, &model, options.cancellation.clone());
            let outcome = if prompt_tools {
                factory.run_omni_prompt_tools(&model, &context, options, credentials, &mut emitter)
            } else {
                factory.run_stream(&model, &context, options, credentials, &mut emitter)
            };
            if let Err(error) = outcome {
                let _ = emitter.fail(error);
            }
        });
        events
    }

    /// Serves a chat-only OmniRoute model: the tools are rendered into the
    /// prompt, the transcript is flattened, the reply is fetched over plain
    /// chat completions, and `<tool_call>` blocks are re-emitted as ordinary
    /// tool events. The reply is buffered rather than streamed because the
    /// blocks can only be parsed once complete.
    fn run_omni_prompt_tools(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
        credentials: ProviderCredentials,
        emitter: &mut MessageEmitter,
    ) -> Result<()> {
        ensure_not_cancelled(&options.cancellation)?;
        emitter.start()?;
        let inner_model = llm::Model {
            api: omniroute::OPENAI_COMPLETIONS_API.to_owned(),
            ..model.clone()
        };
        let inner_context = omni_prompt_tools::inner_context(context);
        let inner_options = agent::RequestOptions {
            assistant_event_listener: None,
            ..options.clone()
        };
        let result = self.respond_with_credentials(
            &inner_model,
            &inner_context,
            inner_options,
            credentials,
        )?;
        {
            let message = emitter.message_mut();
            message.usage = result.usage.clone();
            message.response_id = result.response_id.clone();
            message.response_model = result.response_model.clone();
            message.raw_stop_reason = result.raw_stop_reason.clone();
        }
        if result.stop_reason == stream::STOP_ERROR || result.stop_reason == stream::STOP_ABORTED {
            let message = if result.error_message.is_empty() {
                "OmniRoute returned no assistant message".to_owned()
            } else {
                result.error_message
            };
            return Err(ProviderAdapterError::Protocol(message));
        }
        ensure_not_cancelled(&options.cancellation)?;

        let parsed =
            omni_prompt_tools::parse_tool_calls(&omni_prompt_tools::content_text(&result.content));
        let prose = parsed.prose_with_problems();
        if !prose.is_empty() {
            let index = emitter.start_text("")?;
            emitter.append_text(index, &prose)?;
            emitter.end_text(index)?;
        }
        let had_calls = !parsed.calls.is_empty();
        for call in parsed.calls {
            let id = omni_prompt_tools::next_call_id();
            let index = emitter.start_tool(&id, &call.name)?;
            let encoded = serde_json::to_string(&call.arguments)?;
            emitter.set_tool_arguments(index, call.arguments)?;
            emitter.tool_delta(index, &encoded)?;
            emitter.end_tool(index)?;
        }
        // Keep a truncated reply's stop reason: tool calls parsed out of a
        // response cut off mid-generation must not run as if the model had
        // finished asking for them.
        emitter.message_mut().stop_reason = if result.stop_reason == stream::STOP_LENGTH {
            stream::STOP_LENGTH.to_owned()
        } else if had_calls {
            stream::STOP_TOOL_USE.to_owned()
        } else {
            stream::STOP_STOP.to_owned()
        };
        emitter.finish()
    }

    fn stream_bedrock_with_credentials(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
        credentials: ProviderCredentials,
    ) -> stream::AssistantMessageEventStream {
        let bedrock_cancellation = bedrock::BedrockCancellation::default();
        let agent_cancellation = options.cancellation.clone();
        let reasoning = options.thinking_level;
        let thinking_budgets = options.thinking_budgets;
        let events = bedrock::stream_bedrock_simple(
            model.clone(),
            context.clone(),
            bedrock::BedrockSimpleOptions {
                request: bedrock::BedrockOptions {
                    api_key: credentials.api_key,
                    headers: credentials.headers,
                    // Bedrock owns its client and still applies this as its
                    // own request deadline.
                    timeout: self.config.read_timeout,
                    max_retries: self.config.max_retries,
                    max_retry_delay: bedrock_retry_delay_limit(self.config.retry_delay_limit),
                    environment: credentials.environment,
                    cancellation: Some(bedrock_cancellation.clone()),
                    ..bedrock::BedrockOptions::default()
                },
                reasoning: Some(reasoning),
                thinking_budgets,
            },
        );

        // Bedrock owns its blocking HTTP reader, so translate the agent's
        // cancellation token on a short polling interval while preserving the
        // protocol adapter's bounded socket timeout.
        let monitor = events.clone();
        thread::spawn(move || {
            while !monitor.is_closed() {
                if agent_cancellation.is_cancelled() {
                    bedrock_cancellation.cancel();
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
        });
        events
    }

    fn run_stream(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
        credentials: ProviderCredentials,
        emitter: &mut MessageEmitter,
    ) -> Result<()> {
        ensure_not_cancelled(&options.cancellation)?;
        let protocol = ProviderProtocol::from_api(&model.api)?;
        let anthropic_shape = anthropic_request_shape(model, &credentials, &options);
        let responses_grammar_tool_input_properties = match protocol {
            ProviderProtocol::AzureOpenAiResponses | ProviderProtocol::OpenAiCodexResponses => {
                grammar_tool_input_properties(
                    &context.tools,
                    compat_bool(model, "supportsOpenAIGrammarTools", false),
                )?
            }
            ProviderProtocol::OpenAiCompletions
            | ProviderProtocol::OpenAiResponses
            | ProviderProtocol::AnthropicMessages
            | ProviderProtocol::GoogleGenerativeAi
            | ProviderProtocol::GoogleVertex
            | ProviderProtocol::MistralConversations
            | ProviderProtocol::BedrockConverseStream => BTreeMap::new(),
        };
        let payload = match protocol {
            ProviderProtocol::OpenAiCompletions => {
                build_openai_completions_request(model, context, &options)
            }
            ProviderProtocol::OpenAiResponses => {
                build_openai_responses_request(model, context, &options)
            }
            ProviderProtocol::AzureOpenAiResponses => build_azure_openai_responses_request(
                model,
                context,
                &options,
                &credentials,
                &responses_grammar_tool_input_properties,
            ),
            ProviderProtocol::OpenAiCodexResponses => build_openai_codex_responses_request(
                model,
                context,
                &options,
                &responses_grammar_tool_input_properties,
            ),
            ProviderProtocol::AnthropicMessages => {
                build_anthropic_messages_request(model, context, &options, &anthropic_shape)
            }
            ProviderProtocol::GoogleGenerativeAi => Ok(build_google_generate_content_request(
                model, context, &options,
            )),
            ProviderProtocol::GoogleVertex => {
                Ok(build_google_vertex_request(model, context, &options))
            }
            ProviderProtocol::MistralConversations => {
                mistral::build_mistral_request(model, context, &options)
            }
            ProviderProtocol::BedrockConverseStream => {
                unreachable!("Bedrock is dispatched before the generic HTTP adapter")
            }
        }?;
        let anthropic_beta = (protocol == ProviderProtocol::AnthropicMessages)
            .then(|| anthropic_beta_features(model, context, &anthropic_shape).join(","))
            .filter(|features| !features.is_empty());
        let response = self.send_streaming_request(
            protocol,
            model,
            &payload,
            &credentials,
            &options,
            anthropic_beta.as_deref(),
        )?;
        let response = CancellableBody::spawn(response, options.cancellation.clone());

        emitter.start()?;
        match protocol {
            ProviderProtocol::OpenAiCompletions => {
                consume_openai_completions(response, model, &options.cancellation, emitter)?
            }
            ProviderProtocol::OpenAiResponses => consume_openai_responses(
                response,
                model,
                &options.cancellation,
                emitter,
                &responses_grammar_tool_input_properties,
            )?,
            ProviderProtocol::AzureOpenAiResponses => consume_openai_responses(
                response,
                model,
                &options.cancellation,
                emitter,
                &responses_grammar_tool_input_properties,
            )?,
            ProviderProtocol::OpenAiCodexResponses => consume_codex_responses(
                response,
                model,
                &options.cancellation,
                emitter,
                &responses_grammar_tool_input_properties,
            )?,
            ProviderProtocol::AnthropicMessages => consume_anthropic_messages(
                response,
                &context.tools,
                anthropic_shape.oauth,
                &options.cancellation,
                emitter,
            )?,
            ProviderProtocol::GoogleGenerativeAi => {
                consume_google_generate_content(response, model, &options.cancellation, emitter)?
            }
            ProviderProtocol::GoogleVertex => {
                consume_google_generate_content(response, model, &options.cancellation, emitter)?
            }
            ProviderProtocol::MistralConversations => {
                mistral::consume_mistral_conversations(response, &options.cancellation, emitter)?
            }
            ProviderProtocol::BedrockConverseStream => {
                unreachable!("Bedrock is dispatched before the generic HTTP adapter")
            }
        }
        ensure_not_cancelled(&options.cancellation)?;
        if emitter.message().stop_reason.is_empty()
            || emitter.message().stop_reason == stream::STOP_PENDING
        {
            return Err(ProviderAdapterError::Protocol(
                "stream ended without a terminal stop reason".to_owned(),
            ));
        }
        if emitter.message().stop_reason == stream::STOP_ERROR
            || emitter.message().stop_reason == stream::STOP_ABORTED
        {
            return Err(ProviderAdapterError::Protocol(
                emitter.message().error_message.clone(),
            ));
        }
        emitter.finish()
    }

    fn send_streaming_request(
        &self,
        protocol: ProviderProtocol,
        model: &llm::Model,
        payload: &Value,
        credentials: &ProviderCredentials,
        options: &agent::RequestOptions,
        anthropic_beta: Option<&str>,
    ) -> Result<Response> {
        let endpoint = protocol_endpoint(model, protocol, credentials)?;
        let headers = build_request_headers(
            protocol,
            model,
            credentials,
            &options.session_id,
            options.cache_retention,
            &options.cancellation,
            anthropic_beta,
        )?;
        let body = serde_json::to_vec(payload)?;

        let mut retry_index = 0;
        loop {
            ensure_not_cancelled(&options.cancellation)?;
            // Only Mistral has a separate time-to-headers deadline; the
            // client's idle deadline bounds the wait for every protocol.
            let header_deadline = (protocol == ProviderProtocol::MistralConversations)
                .then_some(self.config.mistral_response_header_timeout)
                .flatten();
            let sent = self.send_request(
                endpoint.clone(),
                headers.clone(),
                body.clone(),
                &options.cancellation,
                header_deadline,
            );
            match sent {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let error =
                        provider_error_from_response(response, self.config.max_error_body_bytes);
                    if !stream::is_retryable_provider_error(&error)
                        || retry_index >= self.config.max_retries
                    {
                        return Err(ProviderAdapterError::Provider(error));
                    }
                    wait_for_retry(
                        &error,
                        retry_index,
                        self.config.retry_delay_limit,
                        &options.cancellation,
                    )?;
                }
                Err(ProviderAdapterError::Provider(error)) => {
                    if !stream::is_retryable_provider_error(&error)
                        || retry_index >= self.config.max_retries
                    {
                        return Err(ProviderAdapterError::Provider(error));
                    }
                    wait_for_retry(
                        &error,
                        retry_index,
                        self.config.retry_delay_limit,
                        &options.cancellation,
                    )?;
                }
                Err(error) => return Err(error),
            }
            retry_index += 1;
        }
    }

    /// Sends on a helper thread and polls for the response so cancellation is
    /// noticed while waiting for headers. No per-request timeout is set: it
    /// would be a whole-request deadline, whereas the client-level deadline
    /// configured in `configured` only bounds idle time.
    fn send_request(
        &self,
        endpoint: Url,
        headers: HeaderMap,
        body: Vec<u8>,
        cancellation: &agent::CancellationToken,
        header_deadline: Option<Duration>,
    ) -> Result<Response> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let client = self.client.clone();
        thread::spawn(move || {
            let response = client.post(endpoint).headers(headers).body(body).send();
            let _ = sender.send(response);
        });

        let deadline = header_deadline.and_then(|timeout| Instant::now().checked_add(timeout));
        loop {
            ensure_not_cancelled(cancellation)?;
            if let Some(deadline) = deadline
                && Instant::now() >= deadline
            {
                return Err(ProviderAdapterError::Provider(stream::ProviderError::new(
                    0,
                    "Mistral response headers timed out",
                )));
            }
            match receiver.recv_timeout(STREAM_POLL_INTERVAL) {
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(error)) => return Err(provider_network_error(error)),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(ProviderAdapterError::Protocol(
                        "request worker exited before returning a response".to_owned(),
                    ));
                }
            }
        }
    }
}

/// Creates a responder which resolves authentication from the catalog before
/// each turn.  This keeps updated credential-store values and configured
/// headers visible to long-running agents.
pub fn assistant_responder_from_catalog(
    catalog: Arc<catalog::Catalog>,
    config: ProviderConfig,
) -> Result<agent::AssistantResponder> {
    let transport = ProviderResponderFactory::configured(ProviderCredentials::default(), config)?;
    Ok(transport.catalog_assistant_responder(catalog))
}

impl ProviderResponderFactory {
    /// Like [`assistant_responder_from_catalog`], but reuses this factory's
    /// configured reqwest client and transport policy.
    pub fn catalog_assistant_responder(
        &self,
        catalog: Arc<catalog::Catalog>,
    ) -> agent::AssistantResponder {
        let transport = self.clone();
        Arc::new(move |model, context, options| {
            let reference = format!("{}/{}", model.provider, model.id);
            let resolved = catalog
                .resolve_model(&reference)
                .map_err(|error| error.to_string())?;
            let credentials = ProviderCredentials::from_resolved_model(&resolved);
            // Native Aperture adaptation: a gateway-routed request carries the
            // provider-qualified model id and the provenance headers, and a
            // transient gateway restart is tagged so the retry classifier
            // recognizes it (cmd/goshcoder/aperture_session.go).
            let routed = catalog.aperture_request_model(model, &options.session_id);
            let gateway_routed = matches!(routed, Cow::Owned(_));
            let result = transport
                .respond_with_credentials(&routed, context, options, credentials)
                .map_err(|error| error.to_string());
            if !gateway_routed {
                return result;
            }
            match result {
                Ok(mut message) => {
                    if let Some(tagged) = aperture::mark_retryable_error(&message.error_message) {
                        message.error_message = tagged;
                    }
                    Ok(message)
                }
                Err(error) => Err(aperture::mark_retryable_error(&error).unwrap_or(error)),
            }
        })
    }
}

fn provider_network_error(error: reqwest::Error) -> ProviderAdapterError {
    ProviderAdapterError::Provider(stream::ProviderError::new(
        0,
        format!("provider network request failed: {error}"),
    ))
}

fn ensure_not_cancelled(cancellation: &agent::CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(ProviderAdapterError::Cancelled)
    } else {
        Ok(())
    }
}

/// The aborted message a cancelled consumer falls back to when the worker
/// never delivers its own.
fn aborted_message(model: &llm::Model) -> llm::AssistantMessage {
    let mut message = initial_assistant_message(model);
    message.stop_reason = stream::STOP_ABORTED.to_owned();
    message.error_message = ProviderAdapterError::Cancelled.to_string();
    message
}

/// A response body whose blocking socket reads happen on a helper thread, so
/// the stream worker can keep checking cancellation between chunks.
///
/// The helper thread finishes on EOF, on a read error (including the idle
/// deadline), or when the worker stops consuming; cancellation therefore
/// never waits on the socket.
struct CancellableBody {
    chunks: mpsc::Receiver<io::Result<Vec<u8>>>,
    pending: Vec<u8>,
    offset: usize,
    cancellation: agent::CancellationToken,
    finished: bool,
}

impl CancellableBody {
    fn spawn(mut response: Response, cancellation: agent::CancellationToken) -> Self {
        let (sender, chunks) = mpsc::sync_channel(BODY_CHUNK_QUEUE);
        thread::spawn(move || {
            let mut buffer = vec![0_u8; 16 * 1024];
            loop {
                let chunk = match response.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => Ok(buffer[..read].to_vec()),
                    Err(error) => Err(describe_body_read_error(error)),
                };
                let failed = chunk.is_err();
                if sender.send(chunk).is_err() || failed {
                    break;
                }
            }
        });
        Self {
            chunks,
            pending: Vec::new(),
            offset: 0,
            cancellation,
            finished: false,
        }
    }
}

/// reqwest reports the idle deadline as an opaque "error decoding response
/// body"; naming the timeout keeps the failure readable and lets pi's retry
/// classifier recognize it as transient.
fn describe_body_read_error(error: io::Error) -> io::Error {
    let timed_out = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<reqwest::Error>())
        .is_some_and(reqwest::Error::is_timeout);
    if timed_out {
        io::Error::new(
            io::ErrorKind::TimedOut,
            format!("provider stream timed out waiting for data: {error}"),
        )
    } else {
        error
    }
}

impl Read for CancellableBody {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.offset < self.pending.len() {
                let length = (self.pending.len() - self.offset).min(buffer.len());
                buffer[..length].copy_from_slice(&self.pending[self.offset..self.offset + length]);
                self.offset += length;
                return Ok(length);
            }
            if self.finished {
                return Ok(0);
            }
            if self.cancellation.is_cancelled() {
                return Err(io::Error::other(stream::RequestAborted));
            }
            match self.chunks.recv_timeout(STREAM_POLL_INTERVAL) {
                Ok(Ok(chunk)) => {
                    self.pending = chunk;
                    self.offset = 0;
                }
                Ok(Err(error)) => {
                    self.finished = true;
                    return Err(error);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.finished = true;
                    return Ok(0);
                }
            }
        }
    }
}

fn initial_assistant_message(model: &llm::Model) -> llm::AssistantMessage {
    llm::AssistantMessage {
        role: "assistant".to_owned(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        stop_reason: stream::STOP_PENDING.to_owned(),
        timestamp: now_millis(),
        ..llm::AssistantMessage::default()
    }
}

fn wait_for_retry(
    error: &stream::ProviderError,
    retry_index: u32,
    limit: stream::RetryDelayLimit,
    cancellation: &agent::CancellationToken,
) -> Result<()> {
    let delay = stream::retry_delay(error, retry_index, SystemTime::now(), limit)
        .map_err(|error| ProviderAdapterError::Protocol(error.to_string()))?;
    let deadline = Instant::now()
        .checked_add(delay)
        .unwrap_or_else(Instant::now);
    loop {
        ensure_not_cancelled(cancellation)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        thread::sleep(remaining.min(Duration::from_millis(25)));
    }
}

fn bedrock_retry_delay_limit(limit: stream::RetryDelayLimit) -> Option<Duration> {
    match limit {
        stream::RetryDelayLimit::Default => None,
        stream::RetryDelayLimit::Unlimited => Some(Duration::ZERO),
        stream::RetryDelayLimit::Maximum(delay) => Some(delay),
    }
}

fn protocol_endpoint(
    model: &llm::Model,
    protocol: ProviderProtocol,
    credentials: &ProviderCredentials,
) -> Result<Url> {
    match protocol {
        ProviderProtocol::AzureOpenAiResponses => {
            return azure_openai_responses_endpoint(model, credentials);
        }
        ProviderProtocol::OpenAiCodexResponses => {
            return codex_responses_endpoint(&model.base_url);
        }
        ProviderProtocol::GoogleGenerativeAi => {
            return google_generate_content_endpoint(model);
        }
        ProviderProtocol::GoogleVertex => {
            return google_vertex_endpoint(model, credentials);
        }
        ProviderProtocol::MistralConversations => {}
        ProviderProtocol::BedrockConverseStream => {
            unreachable!("Bedrock uses its own signed request builder");
        }
        ProviderProtocol::OpenAiCompletions
        | ProviderProtocol::OpenAiResponses
        | ProviderProtocol::AnthropicMessages => {}
    }
    if model.base_url.trim().is_empty() {
        return Err(ProviderAdapterError::MissingBaseUrl);
    }
    let mut endpoint =
        Url::parse(model.base_url.trim()).map_err(|_| ProviderAdapterError::InvalidBaseUrl)?;
    append_endpoint_suffix(&mut endpoint, protocol.endpoint_suffix());
    Ok(endpoint)
}

fn append_endpoint_suffix(endpoint: &mut Url, suffix: &str) {
    let prefix = endpoint.path().trim_end_matches('/');
    endpoint.set_path(&format!("{prefix}/{suffix}"));
}

fn google_generate_content_endpoint(model: &llm::Model) -> Result<Url> {
    let base_url = if model.base_url.trim().is_empty() {
        DEFAULT_GOOGLE_GENERATIVE_AI_BASE_URL
    } else {
        model.base_url.trim()
    };
    let mut endpoint = Url::parse(base_url).map_err(|_| ProviderAdapterError::InvalidBaseUrl)?;
    if endpoint.host_str().is_none() {
        return Err(ProviderAdapterError::InvalidBaseUrl);
    }
    append_endpoint_suffix(
        &mut endpoint,
        &format!("models/{}:streamGenerateContent", model.id),
    );
    endpoint.set_query(Some("alt=sse"));
    Ok(endpoint)
}

fn google_vertex_endpoint(model: &llm::Model, credentials: &ProviderCredentials) -> Result<Url> {
    let api_key = resolve_vertex_api_key(credentials.api_key_value().unwrap_or_default());
    if !api_key.is_empty() {
        return google_vertex_express_endpoint(model);
    }
    google_vertex_resource_endpoint(model, &credentials.environment)
}

fn google_vertex_express_endpoint(model: &llm::Model) -> Result<Url> {
    let mut endpoint = google_vertex_base_endpoint(model, None)?;
    append_endpoint_suffix(
        &mut endpoint,
        &format!(
            "publishers/google/models/{}:streamGenerateContent",
            model.id
        ),
    );
    endpoint.set_query(Some("alt=sse"));
    Ok(endpoint)
}

fn google_vertex_resource_endpoint(
    model: &llm::Model,
    environment: &BTreeMap<String, String>,
) -> Result<Url> {
    let project = vertex_project(environment)?;
    let location = vertex_location(environment)?;
    let mut endpoint = google_vertex_base_endpoint(model, Some(&location))?;
    append_endpoint_suffix(
        &mut endpoint,
        &format!(
            "projects/{project}/locations/{location}/publishers/google/models/{}:streamGenerateContent",
            model.id
        ),
    );
    endpoint.set_query(Some("alt=sse"));
    Ok(endpoint)
}

fn google_vertex_base_endpoint(model: &llm::Model, location: Option<&str>) -> Result<Url> {
    let custom = vertex_custom_base_url(&model.base_url);
    let mut endpoint = if custom.is_empty() {
        let host = match location {
            Some("global") => "https://aiplatform.googleapis.com".to_owned(),
            Some(location) => format!("https://{location}-aiplatform.googleapis.com"),
            None => "https://aiplatform.googleapis.com".to_owned(),
        };
        Url::parse(&format!("{host}/{DEFAULT_VERTEX_API_VERSION}"))
    } else {
        Url::parse(&custom)
    }
    .map_err(|_| ProviderAdapterError::InvalidBaseUrl)?;
    if endpoint.host_str().is_none() {
        return Err(ProviderAdapterError::InvalidBaseUrl);
    }
    if !vertex_base_url_includes_api_version(&endpoint) {
        append_endpoint_suffix(&mut endpoint, DEFAULT_VERTEX_API_VERSION);
    }
    Ok(endpoint)
}

fn resolve_vertex_api_key(api_key: &str) -> String {
    let api_key = api_key.trim();
    if api_key.is_empty()
        || api_key == VERTEX_AMBIENT_CREDENTIALS_MARKER
        || (api_key.starts_with('<') && api_key.ends_with('>'))
    {
        String::new()
    } else {
        api_key.to_owned()
    }
}

fn vertex_project(environment: &BTreeMap<String, String>) -> Result<String> {
    let project = bedrock::provider_env_value(environment, "GOOGLE_CLOUD_PROJECT");
    let project = if project.is_empty() {
        bedrock::provider_env_value(environment, "GCLOUD_PROJECT")
    } else {
        project
    };
    if project.is_empty() {
        Err(ProviderAdapterError::Protocol(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT".to_owned(),
        ))
    } else {
        Ok(project)
    }
}

fn vertex_location(environment: &BTreeMap<String, String>) -> Result<String> {
    let location = bedrock::provider_env_value(environment, "GOOGLE_CLOUD_LOCATION");
    if location.is_empty() {
        Err(ProviderAdapterError::Protocol(
            "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION".to_owned(),
        ))
    } else {
        Ok(location)
    }
}

fn vertex_custom_base_url(base_url: &str) -> String {
    let base_url = base_url.trim();
    if base_url.is_empty() || base_url.contains("{location}") {
        String::new()
    } else {
        base_url.trim_end_matches('/').to_owned()
    }
}

fn vertex_base_url_includes_api_version(endpoint: &Url) -> bool {
    endpoint
        .path_segments()
        .is_some_and(|mut segments| segments.any(vertex_api_version_segment))
}

fn vertex_api_version_segment(segment: &str) -> bool {
    let Some(version) = segment.strip_prefix('v') else {
        return false;
    };
    let digits = version.chars().take_while(char::is_ascii_digit).count();
    if digits == 0 {
        return false;
    }
    let suffix = &version[digits..];
    suffix.is_empty()
        || suffix
            .strip_prefix("beta")
            .is_some_and(|suffix| suffix.chars().all(|character| character.is_ascii_digit()))
}

fn azure_openai_responses_endpoint(
    model: &llm::Model,
    credentials: &ProviderCredentials,
) -> Result<Url> {
    let (mut endpoint, api_version) = resolve_azure_openai_config(model, credentials)?;
    append_endpoint_suffix(&mut endpoint, "responses");

    let mut query = endpoint
        .query_pairs()
        .filter(|(name, _)| name != "api-version")
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    query.push(("api-version".to_owned(), api_version));
    endpoint.set_query(None);
    {
        let mut serializer = endpoint.query_pairs_mut();
        for (name, value) in query {
            serializer.append_pair(&name, &value);
        }
    }
    Ok(endpoint)
}

fn resolve_azure_openai_config(
    model: &llm::Model,
    credentials: &ProviderCredentials,
) -> Result<(Url, String)> {
    let api_version =
        configured_provider_environment_value(credentials, "AZURE_OPENAI_API_VERSION");
    let api_version = if api_version.is_empty() {
        DEFAULT_AZURE_OPENAI_API_VERSION.to_owned()
    } else {
        api_version
    };

    let configured_base_url =
        configured_provider_environment_value(credentials, "AZURE_OPENAI_BASE_URL");
    let base_url = if !configured_base_url.is_empty() {
        configured_base_url
    } else {
        let resource_name =
            configured_provider_environment_value(credentials, "AZURE_OPENAI_RESOURCE_NAME");
        if !resource_name.is_empty() {
            format!(
                "https://{}.openai.azure.com/openai/v1",
                resource_name.trim()
            )
        } else {
            model.base_url.trim().to_owned()
        }
    };
    if base_url.is_empty() {
        return Err(ProviderAdapterError::AzureBaseUrlRequired);
    }
    Ok((normalize_azure_openai_base_url(&base_url)?, api_version))
}

fn configured_provider_environment_value(credentials: &ProviderCredentials, name: &str) -> String {
    bedrock::provider_env_value(&credentials.environment, name)
}

fn normalize_azure_openai_base_url(base_url: &str) -> Result<Url> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let mut endpoint = Url::parse(trimmed).map_err(|_| ProviderAdapterError::InvalidBaseUrl)?;
    if endpoint.host_str().is_none() {
        return Err(ProviderAdapterError::InvalidBaseUrl);
    }

    let is_azure_managed = endpoint.host_str().is_some_and(|host| {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        AZURE_MANAGED_HOST_SUFFIXES
            .iter()
            .any(|suffix| host.ends_with(suffix))
    });
    let path = endpoint.path().trim_end_matches('/');
    if is_azure_managed && matches!(path, "" | "/openai" | "/openai/v1/responses") {
        endpoint.set_path("/openai/v1");
        endpoint.set_query(None);
    }
    Ok(endpoint)
}

fn parse_azure_deployment_name_map(value: &str) -> BTreeMap<String, String> {
    value
        .split(',')
        .filter_map(|entry| {
            let (model_id, deployment_name) = entry.split_once('=')?;
            let model_id = model_id.trim();
            let deployment_name = deployment_name.trim();
            (!model_id.is_empty() && !deployment_name.is_empty())
                .then(|| (model_id.to_owned(), deployment_name.to_owned()))
        })
        .collect()
}

fn azure_deployment_name(model: &llm::Model, credentials: &ProviderCredentials) -> String {
    let deployments = parse_azure_deployment_name_map(&configured_provider_environment_value(
        credentials,
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
    ));
    deployments
        .get(&model.id)
        .cloned()
        .unwrap_or_else(|| model.id.clone())
}

fn codex_responses_endpoint(base_url: &str) -> Result<Url> {
    let normalized = base_url.trim().trim_end_matches('/');
    let base_url = if normalized.is_empty() {
        DEFAULT_CODEX_BASE_URL
    } else {
        normalized
    };
    let mut endpoint = Url::parse(base_url).map_err(|_| ProviderAdapterError::InvalidBaseUrl)?;
    if endpoint.host_str().is_none() {
        return Err(ProviderAdapterError::InvalidBaseUrl);
    }
    let path = endpoint.path().trim_end_matches('/');
    let path = if path.ends_with("/codex/responses") {
        path.to_owned()
    } else if path.ends_with("/codex") {
        format!("{path}/responses")
    } else {
        format!("{path}/codex/responses")
    };
    endpoint.set_path(&path);
    Ok(endpoint)
}

fn extract_codex_account_id(token: &str) -> Result<String> {
    let parts = token.split('.').collect::<Vec<_>>();
    let [_, payload, _] = parts.as_slice() else {
        return Err(ProviderAdapterError::InvalidCodexToken);
    };
    let payload = URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .map_err(|_| ProviderAdapterError::InvalidCodexToken)?;
    let payload = serde_json::from_slice::<Value>(&payload)
        .map_err(|_| ProviderAdapterError::InvalidCodexToken)?;
    payload
        .get(CODEX_JWT_AUTH_CLAIM)
        .and_then(Value::as_object)
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .filter(|account_id| !account_id.is_empty())
        .map(str::to_owned)
        .ok_or(ProviderAdapterError::InvalidCodexToken)
}

fn provider_error_from_response(
    mut response: Response,
    maximum_body_bytes: usize,
) -> stream::ProviderError {
    let status = response.status().as_u16();
    let headers = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.as_str().to_owned(), value.to_owned()))
        })
        .collect();
    let mut body = Vec::new();
    let _ = response
        .by_ref()
        .take(maximum_body_bytes.saturating_add(1) as u64)
        .read_to_end(&mut body);
    let read_capped = body.len() > maximum_body_bytes;
    body.truncate(maximum_body_bytes);
    let text = String::from_utf8_lossy(&body);
    let text = text.trim();
    let body = if text.chars().count() > MAX_PROVIDER_ERROR_BODY_CHARS {
        truncate_error_text(text, MAX_PROVIDER_ERROR_BODY_CHARS)
    } else if read_capped {
        format!("{text}…")
    } else {
        text.to_owned()
    };
    let suffix = if body.is_empty() {
        String::new()
    } else {
        format!(": {body}")
    };
    stream::ProviderError {
        status,
        headers,
        body,
        message: format!("provider request failed with status {status}{suffix}"),
    }
}

/// pi's `truncateErrorText`: keeps the first `max_chars` characters and says
/// how much was cut.
fn truncate_error_text(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_owned();
    }
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str(&format!("... [truncated {} chars]", total - max_chars));
    truncated
}

fn build_request_headers(
    protocol: ProviderProtocol,
    model: &llm::Model,
    credentials: &ProviderCredentials,
    session_id: &str,
    cache_retention: agent::CacheRetention,
    cancellation: &agent::CancellationToken,
    anthropic_beta: Option<&str>,
) -> Result<HeaderMap> {
    let mut overrides = BTreeMap::<String, Option<String>>::new();
    set_header_override(
        &mut overrides,
        "content-type".to_owned(),
        Some("application/json".to_owned()),
    );
    match protocol {
        ProviderProtocol::OpenAiCompletions
        | ProviderProtocol::OpenAiResponses
        | ProviderProtocol::AzureOpenAiResponses
        | ProviderProtocol::OpenAiCodexResponses
        | ProviderProtocol::GoogleGenerativeAi
        | ProviderProtocol::GoogleVertex
        | ProviderProtocol::MistralConversations => {
            set_header_override(
                &mut overrides,
                "accept".to_owned(),
                Some("text/event-stream".to_owned()),
            );
        }
        ProviderProtocol::AnthropicMessages => {
            set_header_override(
                &mut overrides,
                "accept".to_owned(),
                Some("application/json".to_owned()),
            );
            set_header_override(
                &mut overrides,
                "anthropic-version".to_owned(),
                Some("2023-06-01".to_owned()),
            );
            set_header_override(
                &mut overrides,
                "anthropic-dangerous-direct-browser-access".to_owned(),
                Some("true".to_owned()),
            );
            if let Some(features) = anthropic_beta {
                set_header_override(
                    &mut overrides,
                    "anthropic-beta".to_owned(),
                    Some(features.to_owned()),
                );
            }
        }
        ProviderProtocol::BedrockConverseStream => {
            unreachable!("Bedrock uses its own signed request builder")
        }
    }

    let api_key = credentials.api_key_value().unwrap_or_default();
    let ambient = api_key == catalog::AUTHENTICATED_SENTINEL;
    match protocol {
        ProviderProtocol::AzureOpenAiResponses
        | ProviderProtocol::OpenAiCodexResponses
        | ProviderProtocol::GoogleGenerativeAi
            if ambient =>
        {
            return Err(ProviderAdapterError::AmbientCredentialsUnsupported {
                provider: model.provider.clone(),
            });
        }
        ProviderProtocol::AzureOpenAiResponses
        | ProviderProtocol::OpenAiCodexResponses
        | ProviderProtocol::GoogleGenerativeAi
            if api_key.is_empty() =>
        {
            return Err(ProviderAdapterError::MissingApiKey {
                provider: model.provider.clone(),
            });
        }
        _ => {}
    }
    if matches!(protocol, ProviderProtocol::MistralConversations) && (ambient || api_key.is_empty())
    {
        return Err(ProviderAdapterError::MissingApiKey {
            provider: model.provider.clone(),
        });
    }
    let codex_account_id = match protocol {
        ProviderProtocol::OpenAiCodexResponses => Some(extract_codex_account_id(api_key)?),
        _ => None,
    };
    if protocol == ProviderProtocol::AzureOpenAiResponses {
        // The Azure default must be installed before configured model/auth
        // headers so a proxy can intentionally replace or suppress it.
        set_header_override(
            &mut overrides,
            "api-key".to_owned(),
            Some(api_key.to_owned()),
        );
    }
    let vertex_api_key = if protocol == ProviderProtocol::GoogleVertex {
        resolve_vertex_api_key(api_key)
    } else {
        String::new()
    };
    let vertex_access_token =
        if protocol == ProviderProtocol::GoogleVertex && vertex_api_key.is_empty() {
            Some(
                match google_auth::resolve_access_token_with_cancellation(
                    &credentials.environment,
                    cancellation,
                ) {
                    Ok(token) => token,
                    Err(google_auth::GoogleAuthError::Cancelled) => {
                        return Err(ProviderAdapterError::Cancelled);
                    }
                    Err(error) => return Err(ProviderAdapterError::Protocol(error.to_string())),
                },
            )
        } else {
            None
        };
    match protocol {
        ProviderProtocol::GoogleVertex => {
            if vertex_api_key.is_empty() {
                set_header_override(
                    &mut overrides,
                    "authorization".to_owned(),
                    vertex_access_token.map(|token| format!("Bearer {token}")),
                );
            } else {
                set_header_override(
                    &mut overrides,
                    "x-goog-api-key".to_owned(),
                    Some(vertex_api_key),
                );
            }
        }
        _ if !ambient && !api_key.is_empty() => match protocol {
            ProviderProtocol::OpenAiCompletions
            | ProviderProtocol::OpenAiResponses
            | ProviderProtocol::MistralConversations => {
                set_header_override(
                    &mut overrides,
                    "authorization".to_owned(),
                    Some(format!("Bearer {api_key}")),
                );
            }
            ProviderProtocol::AnthropicMessages => {
                if anthropic_is_oauth_token(api_key) {
                    // pi sends an OAuth token as a bearer credential under
                    // the Claude Code client identity.
                    set_header_override(
                        &mut overrides,
                        "authorization".to_owned(),
                        Some(format!("Bearer {api_key}")),
                    );
                    set_header_override(
                        &mut overrides,
                        "user-agent".to_owned(),
                        Some(format!("claude-cli/{CLAUDE_CODE_VERSION}")),
                    );
                    set_header_override(&mut overrides, "x-app".to_owned(), Some("cli".to_owned()));
                } else {
                    set_header_override(
                        &mut overrides,
                        "x-api-key".to_owned(),
                        Some(api_key.to_owned()),
                    );
                }
            }
            ProviderProtocol::GoogleGenerativeAi => {
                set_header_override(
                    &mut overrides,
                    "x-goog-api-key".to_owned(),
                    Some(api_key.to_owned()),
                );
            }
            ProviderProtocol::AzureOpenAiResponses | ProviderProtocol::OpenAiCodexResponses => {}
            ProviderProtocol::GoogleVertex => unreachable!("Vertex is handled above"),
            ProviderProtocol::BedrockConverseStream => {
                unreachable!("Bedrock uses its own signed request builder")
            }
        },
        _ => {}
    }

    for (name, value) in &model.headers {
        set_header_override(&mut overrides, name.clone(), Some(value.clone()));
    }
    for (name, value) in credentials.headers() {
        set_header_override(&mut overrides, name.clone(), value.clone());
    }
    if protocol == ProviderProtocol::MistralConversations
        && cache_retention != agent::CacheRetention::None
        && !session_id.is_empty()
        && !has_header_override(&overrides, "x-affinity")
    {
        set_header_override(
            &mut overrides,
            "x-affinity".to_owned(),
            Some(session_id.to_owned()),
        );
    }

    match protocol {
        ProviderProtocol::OpenAiCodexResponses => {
            let account_id = codex_account_id.expect("Codex account ID was validated");
            set_header_override(
                &mut overrides,
                "authorization".to_owned(),
                Some(format!("Bearer {api_key}")),
            );
            set_header_override(
                &mut overrides,
                "chatgpt-account-id".to_owned(),
                Some(account_id),
            );
            set_header_override(
                &mut overrides,
                "originator".to_owned(),
                Some("goshcoder".to_owned()),
            );
            set_header_override(
                &mut overrides,
                "user-agent".to_owned(),
                Some(format!(
                    "goshcoder ({}; {})",
                    std::env::consts::OS,
                    std::env::consts::ARCH
                )),
            );
            set_header_override(
                &mut overrides,
                "openai-beta".to_owned(),
                Some("responses=experimental".to_owned()),
            );
            set_header_override(
                &mut overrides,
                "accept".to_owned(),
                Some("text/event-stream".to_owned()),
            );
            set_header_override(
                &mut overrides,
                "content-type".to_owned(),
                Some("application/json".to_owned()),
            );
            let session_id = clamp_prompt_cache_key(session_id);
            if !session_id.is_empty() {
                set_header_override(
                    &mut overrides,
                    "session-id".to_owned(),
                    Some(session_id.clone()),
                );
                set_header_override(
                    &mut overrides,
                    "x-client-request-id".to_owned(),
                    Some(session_id),
                );
            }
        }
        ProviderProtocol::OpenAiCompletions
        | ProviderProtocol::OpenAiResponses
        | ProviderProtocol::AzureOpenAiResponses
        | ProviderProtocol::AnthropicMessages
        | ProviderProtocol::GoogleGenerativeAi
        | ProviderProtocol::GoogleVertex
        | ProviderProtocol::MistralConversations
        | ProviderProtocol::BedrockConverseStream => {}
    }

    let has_authorization = has_nonempty_header(&overrides, "authorization")
        || has_nonempty_header(&overrides, "cf-aig-authorization");
    let has_api_key_header = has_nonempty_header(&overrides, "x-api-key");
    let authenticated = match protocol {
        ProviderProtocol::OpenAiCompletions | ProviderProtocol::OpenAiResponses => {
            has_authorization
        }
        // Mistral explicitly requires an API key before configured headers
        // are applied. Those headers may intentionally suppress or replace
        // the default bearer credential for a gateway, matching its native
        // client behavior.
        ProviderProtocol::MistralConversations => true,
        ProviderProtocol::AzureOpenAiResponses
        | ProviderProtocol::OpenAiCodexResponses
        | ProviderProtocol::GoogleGenerativeAi
        | ProviderProtocol::GoogleVertex => true,
        ProviderProtocol::AnthropicMessages => has_authorization || has_api_key_header,
        ProviderProtocol::BedrockConverseStream => {
            unreachable!("Bedrock uses its own signed request builder")
        }
    };
    if !authenticated {
        if ambient {
            return Err(ProviderAdapterError::AmbientCredentialsUnsupported {
                provider: model.provider.clone(),
            });
        }
        return Err(ProviderAdapterError::MissingCredential {
            provider: model.provider.clone(),
        });
    }

    let mut headers = HeaderMap::new();
    for (name, value) in overrides {
        let Some(value) = value else {
            continue;
        };
        let name = HeaderName::from_bytes(name.as_bytes())
            .map_err(|_| ProviderAdapterError::InvalidHeaderName(name))?;
        let value = HeaderValue::from_str(&value)
            .map_err(|_| ProviderAdapterError::InvalidHeaderValue(name.as_str().to_owned()))?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn set_header_override(
    headers: &mut BTreeMap<String, Option<String>>,
    name: String,
    value: Option<String>,
) {
    let existing = headers
        .keys()
        .filter(|candidate| candidate.eq_ignore_ascii_case(&name))
        .cloned()
        .collect::<Vec<_>>();
    for existing in existing {
        headers.remove(&existing);
    }
    headers.insert(name, value);
}

fn has_nonempty_header(headers: &BTreeMap<String, Option<String>>, name: &str) -> bool {
    headers.iter().any(|(candidate, value)| {
        candidate.eq_ignore_ascii_case(name)
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
}

fn has_header_override(headers: &BTreeMap<String, Option<String>>, name: &str) -> bool {
    headers
        .keys()
        .any(|candidate| candidate.eq_ignore_ascii_case(name))
}

/// pi's resolved `OpenAICompletionsCompat`: detected from the provider and
/// base URL, then overridden by explicit `model.compat` entries.
#[derive(Clone, Debug)]
struct OpenAiCompletionsCompat {
    supports_store: bool,
    supports_developer_role: bool,
    supports_reasoning_effort: bool,
    supports_usage_in_streaming: bool,
    supports_finish_reason: bool,
    max_tokens_field: &'static str,
    requires_tool_result_name: bool,
    requires_assistant_after_tool_result: bool,
    requires_thinking_as_text: bool,
    requires_reasoning_content_on_assistant_messages: bool,
    thinking_format: String,
    chat_template_kwargs: Map<String, Value>,
    chat_template_args: Map<String, Value>,
    supports_thinking_token_budget: bool,
    thinking_token_budget_field: Option<String>,
    supports_strict_mode: bool,
}

impl OpenAiCompletionsCompat {
    fn from_model(model: &llm::Model) -> Self {
        let provider = model.provider.as_str();
        let base_url = model.base_url.as_str();
        let is_zai = matches!(provider, "zai" | "zai-coding-cn")
            || base_url.contains("api.z.ai")
            || base_url.contains("open.bigmodel.cn");
        let is_together = provider == "together"
            || base_url.contains("api.together.ai")
            || base_url.contains("api.together.xyz");
        let is_moonshot = matches!(provider, "moonshotai" | "moonshotai-cn")
            || base_url.contains("api.moonshot.");
        let is_openrouter = provider == "openrouter" || base_url.contains("openrouter.ai");
        let is_cloudflare_workers_ai =
            provider == "cloudflare-workers-ai" || base_url.contains("api.cloudflare.com");
        let is_cloudflare_ai_gateway =
            provider == "cloudflare-ai-gateway" || base_url.contains("gateway.ai.cloudflare.com");
        let is_nvidia = provider == "nvidia" || base_url.contains("integrate.api.nvidia.com");
        let is_ant_ling = provider == "ant-ling" || base_url.contains("api.ant-ling.com");
        let is_deepseek =
            provider == "deepseek" || base_url.to_ascii_lowercase().contains("deepseek.com");
        let is_grok = provider == "xai" || base_url.contains("api.x.ai");
        let non_standard = is_nvidia
            || provider == "cerebras"
            || base_url.contains("cerebras.ai")
            || is_grok
            || is_together
            || base_url.contains("chutes.ai")
            || is_deepseek
            || is_zai
            || is_moonshot
            || provider == "opencode"
            || base_url.contains("opencode.ai")
            || is_cloudflare_workers_ai
            || is_cloudflare_ai_gateway
            || is_ant_ling;
        let use_max_tokens = base_url.contains("chutes.ai")
            || is_deepseek
            || is_moonshot
            || is_cloudflare_ai_gateway
            || is_together
            || is_nvidia
            || is_ant_ling
            || is_zai;
        let openrouter_developer_role_model = is_openrouter
            && (model.id.starts_with("anthropic/") || model.id.starts_with("openai/"));
        let detected_thinking_format = if is_deepseek {
            "deepseek"
        } else if is_zai {
            "zai"
        } else if is_together {
            "together"
        } else if is_ant_ling {
            "ant-ling"
        } else if is_openrouter {
            "openrouter"
        } else {
            "openai"
        };
        let max_tokens_field = match compat_string(model, "maxTokensField").as_deref() {
            Some("max_tokens") => "max_tokens",
            Some("max_completion_tokens") => "max_completion_tokens",
            _ if use_max_tokens => "max_tokens",
            _ => "max_completion_tokens",
        };
        Self {
            supports_store: compat_bool(model, "supportsStore", !non_standard),
            supports_developer_role: compat_bool(
                model,
                "supportsDeveloperRole",
                openrouter_developer_role_model || (!non_standard && !is_openrouter),
            ),
            supports_reasoning_effort: compat_bool(
                model,
                "supportsReasoningEffort",
                !is_grok
                    && !is_zai
                    && !is_moonshot
                    && !is_together
                    && !is_cloudflare_ai_gateway
                    && !is_nvidia
                    && !is_ant_ling,
            ),
            supports_usage_in_streaming: compat_bool(model, "supportsUsageInStreaming", true),
            supports_finish_reason: compat_bool(model, "supportsFinishReason", true),
            max_tokens_field,
            requires_tool_result_name: compat_bool(model, "requiresToolResultName", false),
            requires_assistant_after_tool_result: compat_bool(
                model,
                "requiresAssistantAfterToolResult",
                false,
            ),
            requires_thinking_as_text: compat_bool(model, "requiresThinkingAsText", false),
            requires_reasoning_content_on_assistant_messages: compat_bool(
                model,
                "requiresReasoningContentOnAssistantMessages",
                is_deepseek,
            ),
            thinking_format: compat_string(model, "thinkingFormat")
                .unwrap_or_else(|| detected_thinking_format.to_owned()),
            chat_template_kwargs: compat_map(model, "chatTemplateKwargs"),
            chat_template_args: compat_map(model, "chatTemplateArgs"),
            supports_thinking_token_budget: compat_bool(
                model,
                "supportsThinkingTokenBudget",
                false,
            ),
            thinking_token_budget_field: compat_string(model, "thinkingTokenBudgetField")
                .filter(|field| !field.is_empty()),
            supports_strict_mode: compat_bool(
                model,
                "supportsStrictMode",
                !is_moonshot && !is_together && !is_cloudflare_ai_gateway && !is_nvidia,
            ),
        }
    }

    /// pi's `resolveThinkingTokenBudgetField`.
    fn thinking_token_budget_field(&self) -> Option<&str> {
        self.thinking_token_budget_field.as_deref().or_else(|| {
            self.supports_thinking_token_budget
                .then_some("thinking_token_budget")
        })
    }
}

/// A `thinkingLevelMap` lookup keeping JavaScript's three outcomes, because
/// pi's thinking formats treat an explicit `null` differently from an absent
/// entry.
#[derive(Clone, Copy)]
enum LevelMapping<'a> {
    Undefined,
    Null,
    Value(&'a str),
}

fn level_mapping<'a>(model: &'a llm::Model, level: &str) -> LevelMapping<'a> {
    match model.thinking_level_map.get(level) {
        None => LevelMapping::Undefined,
        Some(None) => LevelMapping::Null,
        Some(Some(value)) => LevelMapping::Value(value),
    }
}

/// `model.thinkingLevelMap?.[level] ?? level`.
fn level_or_mapped(model: &llm::Model, level: &str) -> String {
    match level_mapping(model, level) {
        LevelMapping::Value(value) => value.to_owned(),
        LevelMapping::Undefined | LevelMapping::Null => level.to_owned(),
    }
}

fn compat_object(model: &llm::Model) -> Option<&Map<String, Value>> {
    let root = model.compat.as_ref()?.as_object()?;
    root.get(&model.api)
        .and_then(Value::as_object)
        .or(Some(root))
}

fn compat_bool(model: &llm::Model, name: &str, default: bool) -> bool {
    compat_object(model)
        .and_then(|object| object.get(name))
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn compat_string(model: &llm::Model, name: &str) -> Option<String> {
    compat_object(model)
        .and_then(|object| object.get(name))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn compat_map(model: &llm::Model, name: &str) -> Map<String, Value> {
    compat_object(model)
        .and_then(|object| object.get(name))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn requested_max_tokens(model: &llm::Model, context: &llm::Context) -> Option<u64> {
    (model.max_tokens != 0)
        .then(|| stream::clamp_max_tokens_to_context(model, context, model.max_tokens))
}

fn mapped_thinking_level(model: &llm::Model, requested: &str) -> Option<String> {
    if !model.reasoning {
        return None;
    }
    let level = stream::clamp_thinking_level(model, requested);
    if level == llm::THINKING_OFF {
        return None;
    }
    match model.thinking_level_map.get(&level) {
        Some(Some(mapped)) if !mapped.is_empty() => Some(mapped.clone()),
        Some(None) => None,
        _ => Some(level),
    }
}

fn build_openai_completions_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Result<Value> {
    let compat = OpenAiCompletionsCompat::from_model(model);
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.id.clone()));
    body.insert(
        "messages".to_owned(),
        Value::Array(openai_chat_messages(model, context, &compat)),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    if compat.supports_usage_in_streaming {
        body.insert("stream_options".to_owned(), json!({"include_usage": true}));
    }
    if compat.supports_store {
        body.insert("store".to_owned(), Value::Bool(false));
    }
    let max_tokens = requested_max_tokens(model, context);
    if let Some(max_tokens) = max_tokens {
        body.insert(
            compat.max_tokens_field.to_owned(),
            Value::Number(max_tokens.into()),
        );
    }
    if !options.session_id.is_empty() && model.provider == "openai" {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(clamp_prompt_cache_key(&options.session_id)),
        );
    }
    if !context.tools.is_empty() || context_has_tool_history(context) {
        body.insert(
            "tools".to_owned(),
            Value::Array(openai_chat_tools(&context.tools, &compat)?),
        );
    }
    let reasoning_effort = completions_reasoning_effort(model, &options.thinking_level);
    let thinking_budget = resolve_clamped_thinking_budget(
        model,
        reasoning_effort.as_deref(),
        options.thinking_budgets.as_ref(),
        max_tokens,
    );
    apply_completions_thinking(
        &mut body,
        model,
        &compat,
        reasoning_effort.as_deref(),
        thinking_budget,
    );
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

/// pi's `streamSimple` reasoning: the clamped level, absent when off.
fn completions_reasoning_effort(model: &llm::Model, requested: &str) -> Option<String> {
    let level = stream::clamp_thinking_level(model, requested);
    (level != llm::THINKING_OFF).then_some(level)
}

/// pi's `resolveClampedThinkingBudget`: reasoning and the answer share the
/// response ceiling here, so the budget always leaves answer room.
fn resolve_clamped_thinking_budget(
    model: &llm::Model,
    reasoning_effort: Option<&str>,
    custom_budgets: Option<&llm::ThinkingBudgets>,
    max_tokens: Option<u64>,
) -> Option<u64> {
    let level = reasoning_effort?;
    if !model.reasoning {
        return None;
    }
    let ceiling = max_tokens.unwrap_or(model.max_tokens);
    let budget = thinking_budget(level, custom_budgets)
        .min(ceiling.saturating_sub(stream::MIN_ANSWER_TOKENS));
    (budget > 0).then_some(budget)
}

/// Ports the `thinkingFormat` chain of pi's Chat Completions `buildParams`.
fn apply_completions_thinking(
    body: &mut Map<String, Value>,
    model: &llm::Model,
    compat: &OpenAiCompletionsCompat,
    effort: Option<&str>,
    thinking_budget: Option<u64>,
) {
    let reasoning = model.reasoning;
    let off_is_null = matches!(level_mapping(model, llm::THINKING_OFF), LevelMapping::Null);
    match compat.thinking_format.as_str() {
        "zai" if reasoning => {
            body.insert(
                "thinking".to_owned(),
                if effort.is_some() {
                    json!({"type": "enabled", "clear_thinking": false})
                } else {
                    json!({"type": "disabled"})
                },
            );
            if let Some(effort) = effort
                && compat.supports_reasoning_effort
            {
                // An explicit null mapping suppresses the field here.
                let value = match level_mapping(model, effort) {
                    LevelMapping::Undefined => Some(effort.to_owned()),
                    LevelMapping::Null => None,
                    LevelMapping::Value(value) => Some(value.to_owned()),
                };
                if let Some(value) = value {
                    body.insert("reasoning_effort".to_owned(), Value::String(value));
                }
            }
        }
        "qwen" if reasoning => {
            body.insert("enable_thinking".to_owned(), Value::Bool(effort.is_some()));
            if let Some(effort) = effort
                && compat.supports_reasoning_effort
            {
                body.insert(
                    "reasoning_effort".to_owned(),
                    Value::String(level_or_mapped(model, effort)),
                );
            }
        }
        "qwen-chat-template" if reasoning => {
            body.insert(
                "chat_template_kwargs".to_owned(),
                json!({"enable_thinking": effort.is_some(), "preserve_thinking": true}),
            );
        }
        "chat-template" if reasoning => {
            if let Some(values) = build_chat_template_values(
                model,
                effort,
                &compat.chat_template_kwargs,
                thinking_budget,
            ) {
                body.insert("chat_template_kwargs".to_owned(), Value::Object(values));
            }
        }
        "baseten" if reasoning => {
            if let Some(values) = build_chat_template_values(
                model,
                effort,
                &compat.chat_template_args,
                thinking_budget,
            ) {
                body.insert("chat_template_args".to_owned(), Value::Object(values));
            }
            if compat.supports_reasoning_effort {
                let mapped = level_mapping(model, effort.unwrap_or(llm::THINKING_OFF));
                let value = match mapped {
                    LevelMapping::Undefined => effort.map(str::to_owned),
                    LevelMapping::Null => None,
                    LevelMapping::Value(value) => Some(value.to_owned()),
                };
                if let Some(value) = value {
                    body.insert("reasoning_effort".to_owned(), Value::String(value));
                }
            }
        }
        "deepseek" if reasoning => {
            if effort.is_some() {
                body.insert("thinking".to_owned(), json!({"type": "enabled"}));
            } else if !off_is_null {
                body.insert("thinking".to_owned(), json!({"type": "disabled"}));
            }
            if let Some(effort) = effort
                && compat.supports_reasoning_effort
            {
                body.insert(
                    "reasoning_effort".to_owned(),
                    Value::String(level_or_mapped(model, effort)),
                );
            }
        }
        "openrouter" if reasoning => {
            if let Some(effort) = effort {
                body.insert(
                    "reasoning".to_owned(),
                    json!({"effort": level_or_mapped(model, effort)}),
                );
            } else if !off_is_null {
                let off = match level_mapping(model, llm::THINKING_OFF) {
                    LevelMapping::Value(value) => value,
                    LevelMapping::Undefined | LevelMapping::Null => "none",
                };
                body.insert("reasoning".to_owned(), json!({"effort": off}));
            }
        }
        "ant-ling" if reasoning && effort.is_some() => {
            if let Some(effort) = effort
                && let LevelMapping::Value(value) = level_mapping(model, effort)
            {
                body.insert("reasoning".to_owned(), json!({"effort": value}));
            }
        }
        "together" if reasoning => {
            body.insert("reasoning".to_owned(), json!({"enabled": effort.is_some()}));
            if let Some(effort) = effort
                && compat.supports_reasoning_effort
            {
                body.insert(
                    "reasoning_effort".to_owned(),
                    Value::String(level_or_mapped(model, effort)),
                );
            }
        }
        "string-thinking" if reasoning => {
            if let Some(effort) = effort {
                body.insert(
                    "thinking".to_owned(),
                    Value::String(level_or_mapped(model, effort)),
                );
            } else if !off_is_null {
                let off = match level_mapping(model, llm::THINKING_OFF) {
                    LevelMapping::Value(value) => value,
                    LevelMapping::Undefined | LevelMapping::Null => "none",
                };
                body.insert("thinking".to_owned(), Value::String(off.to_owned()));
            }
        }
        _ => {
            if reasoning && compat.supports_reasoning_effort {
                match effort {
                    Some(effort) => {
                        body.insert(
                            "reasoning_effort".to_owned(),
                            Value::String(level_or_mapped(model, effort)),
                        );
                    }
                    None => {
                        if let LevelMapping::Value(off) = level_mapping(model, llm::THINKING_OFF) {
                            body.insert(
                                "reasoning_effort".to_owned(),
                                Value::String(off.to_owned()),
                            );
                        }
                    }
                }
            }
        }
    }
    // Independent of the format: the same server can host several model
    // families, and an uncapped reasoning phase could consume the whole
    // response.
    if let Some(field) = compat.thinking_token_budget_field()
        && let Some(budget) = thinking_budget
    {
        body.insert(field.to_owned(), Value::Number(budget.into()));
    }
}

/// pi's `buildChatTemplateValues`.
fn build_chat_template_values(
    model: &llm::Model,
    effort: Option<&str>,
    values: &Map<String, Value>,
    thinking_budget: Option<u64>,
) -> Option<Map<String, Value>> {
    let mut resolved = Map::new();
    for (key, value) in values {
        if let Some(value) = resolve_chat_template_value(model, effort, value, thinking_budget) {
            resolved.insert(key.clone(), value);
        }
    }
    (!resolved.is_empty()).then_some(resolved)
}

fn resolve_chat_template_value(
    model: &llm::Model,
    effort: Option<&str>,
    value: &Value,
    thinking_budget: Option<u64>,
) -> Option<Value> {
    let Some(object) = value.as_object() else {
        return Some(value.clone());
    };
    if effort.is_none() && object.get("omitWhenOff").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    match object.get("$var").and_then(Value::as_str) {
        Some("thinking.enabled") => return Some(Value::Bool(effort.is_some())),
        Some("thinking.budget") => {
            return thinking_budget.map(|budget| Value::Number(budget.into()));
        }
        _ => {}
    }
    match level_mapping(model, effort.unwrap_or(llm::THINKING_OFF)) {
        LevelMapping::Undefined => effort.map(|effort| Value::String(effort.to_owned())),
        LevelMapping::Null => None,
        LevelMapping::Value(value) => Some(Value::String(value.to_owned())),
    }
}

fn openai_chat_tools(tools: &[llm::Tool], compat: &OpenAiCompletionsCompat) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            let strict = requested_json_schema_strict(tool, compat.supports_strict_mode)?;
            let mut function = json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": schema_or_empty(&tool.parameters),
            });
            // Providers that do not know `strict` reject unknown fields.
            if compat.supports_strict_mode {
                function
                    .as_object_mut()
                    .expect("JSON object")
                    .insert("strict".to_owned(), Value::Bool(strict.unwrap_or(false)));
            }
            Ok(json!({"type": "function", "function": function}))
        })
        .collect()
}

fn openai_chat_messages(
    model: &llm::Model,
    context: &llm::Context,
    compat: &OpenAiCompletionsCompat,
) -> Vec<Value> {
    const BRIDGE_MESSAGE: &str = "I have processed the tool results.";
    let mut normalize =
        |id: &str, _: &llm::AssistantMessage| normalize_completions_tool_call_id(id, model);
    let transformed = transform_messages(&context.messages, model, Some(&mut normalize));

    let mut messages = Vec::new();
    if !context.system_prompt.is_empty() {
        let role = if model.reasoning && compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        messages.push(json!({"role": role, "content": context.system_prompt}));
    }

    let mut last_role = None;
    let mut index = 0;
    while index < transformed.len() {
        let message = &transformed[index];
        // Some providers reject a user message directly after tool results.
        if compat.requires_assistant_after_tool_result
            && last_role == Some("toolResult")
            && matches!(message, llm::Message::User(_))
        {
            messages.push(json!({"role": "assistant", "content": BRIDGE_MESSAGE}));
        }
        match message {
            llm::Message::User(user) => {
                let content = openai_user_content(&user.content, model.supports_images());
                if content.as_array().is_some_and(Vec::is_empty) {
                    index += 1;
                    continue;
                }
                messages.push(json!({"role": "user", "content": content}));
            }
            llm::Message::Assistant(assistant) => {
                if let Some(item) = openai_chat_assistant_message(model, compat, assistant) {
                    messages.push(item);
                } else {
                    index += 1;
                    continue;
                }
            }
            llm::Message::ToolResult(_) => {
                let mut image_blocks = Vec::new();
                while let Some(llm::Message::ToolResult(result)) = transformed.get(index) {
                    let text = text_from_blocks(&result.content);
                    let has_images = result
                        .content
                        .iter()
                        .any(|block| matches!(block, llm::ContentBlock::Image(_)));
                    let content = if !text.is_empty() {
                        text
                    } else if has_images {
                        "(see attached image)".to_owned()
                    } else {
                        "(no tool output)".to_owned()
                    };
                    let mut item = Map::new();
                    item.insert("role".to_owned(), Value::String("tool".to_owned()));
                    item.insert("content".to_owned(), Value::String(content));
                    item.insert(
                        "tool_call_id".to_owned(),
                        Value::String(result.tool_call_id.clone()),
                    );
                    if compat.requires_tool_result_name && !result.tool_name.is_empty() {
                        item.insert("name".to_owned(), Value::String(result.tool_name.clone()));
                    }
                    messages.push(Value::Object(item));
                    if has_images && model.supports_images() {
                        image_blocks.extend(result.content.iter().filter_map(
                            |block| match block {
                                llm::ContentBlock::Image(image) => Some(json!({
                                    "type": "image_url",
                                    "image_url": {"url": data_uri(image)},
                                })),
                                _ => None,
                            },
                        ));
                    }
                    index += 1;
                }
                // Tool messages cannot carry images, so they follow as a user
                // turn.
                if image_blocks.is_empty() {
                    last_role = Some("toolResult");
                } else {
                    if compat.requires_assistant_after_tool_result {
                        messages.push(json!({"role": "assistant", "content": BRIDGE_MESSAGE}));
                    }
                    let mut content = vec![json!({
                        "type": "text",
                        "text": "Attached image(s) from tool result:",
                    })];
                    content.extend(image_blocks);
                    messages.push(json!({"role": "user", "content": content}));
                    last_role = Some("user");
                }
                continue;
            }
        }
        last_role = Some(message.role());
        index += 1;
    }
    messages
}

/// Converts one assistant turn; `None` when it has neither content nor tool
/// calls, which several providers reject.
fn openai_chat_assistant_message(
    model: &llm::Model,
    compat: &OpenAiCompletionsCompat,
    assistant: &llm::AssistantMessage,
) -> Option<Value> {
    const REASONING_FIELDS: [&str; 3] = ["reasoning", "reasoning_content", "reasoning_text"];
    let mut item = Map::new();
    item.insert("role".to_owned(), Value::String("assistant".to_owned()));
    // Providers needing a bridge message also refuse null content.
    let mut content = if compat.requires_assistant_after_tool_result {
        Value::String(String::new())
    } else {
        Value::Null
    };
    let text_parts = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                Some(text.text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let assistant_text = text_parts.concat();
    let thinking_blocks = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Thinking(thinking) if !thinking.thinking.trim().is_empty() => {
                Some(thinking)
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if !thinking_blocks.is_empty() {
        if compat.requires_thinking_as_text {
            // Plain text without tags so the model does not mimic them.
            let thinking = thinking_blocks
                .iter()
                .map(|block| block.thinking.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            let mut parts = vec![json!({"type": "text", "text": thinking})];
            parts.extend(
                text_parts
                    .iter()
                    .map(|text| json!({"type": "text", "text": text})),
            );
            content = Value::Array(parts);
        } else {
            // Assistant text always goes as a plain string: some models mirror
            // a content-block array literally in their output.
            if !assistant_text.is_empty() {
                content = Value::String(assistant_text.clone());
            }
            // The signature records which reasoning field the provider used,
            // so the same field carries it back.
            let mut signature = thinking_blocks[0].thinking_signature.as_str();
            if model.provider == "opencode-go" && signature == "reasoning" {
                signature = "reasoning_content";
            }
            if REASONING_FIELDS.contains(&signature) {
                let thinking = thinking_blocks
                    .iter()
                    .map(|block| block.thinking.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                item.insert(signature.to_owned(), Value::String(thinking));
            }
        }
    } else if !assistant_text.is_empty() {
        content = Value::String(assistant_text);
    }
    let tool_calls = assistant
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::ToolCall(call) => Some(json!({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": serde_json::to_string(&call.arguments)
                        .unwrap_or_else(|_| "{}".to_owned()),
                }
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !tool_calls.is_empty() {
        item.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    if compat.requires_reasoning_content_on_assistant_messages
        && model.reasoning
        && !item.contains_key("reasoning_content")
    {
        item.insert("reasoning_content".to_owned(), Value::String(String::new()));
    }
    let has_content = match &content {
        Value::String(text) => !text.is_empty(),
        Value::Array(parts) => !parts.is_empty(),
        _ => false,
    };
    if !has_content && !item.contains_key("tool_calls") {
        return None;
    }
    item.insert("content".to_owned(), content);
    Some(Value::Object(item))
}

fn openai_user_content(content: &llm::UserContent, supports_images: bool) -> Value {
    match content {
        llm::UserContent::Text(text) => Value::String(text.clone()),
        llm::UserContent::Blocks(blocks) => Value::Array(
            blocks
                .iter()
                .filter_map(|block| match block {
                    llm::ContentBlock::Text(text) => Some(json!({
                        "type": "text",
                        "text": text.text,
                    })),
                    llm::ContentBlock::Image(image) if supports_images => Some(json!({
                        "type": "image_url",
                        "image_url": {"url": data_uri(image)},
                    })),
                    llm::ContentBlock::Image(_) => Some(json!({
                        "type": "text",
                        "text": "(image omitted: model does not support images)",
                    })),
                    llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => None,
                })
                .collect(),
        ),
    }
}

/// Builds the REST payload for Gemini's model-scoped
/// `streamGenerateContent` endpoint.
///
/// Unlike the Google SDK, the REST endpoint expects generation settings at
/// the top level (or nested in `generationConfig`), not an SDK `config`
/// object. Keeping this shape here makes proxy endpoints work too.
fn build_google_generate_content_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Value {
    build_google_request(model, context, options, GoogleApiVariant::Generative)
}

fn build_google_vertex_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Value {
    build_google_request(model, context, options, GoogleApiVariant::Vertex)
}

#[derive(Clone, Copy)]
enum GoogleApiVariant {
    Generative,
    Vertex,
}

fn build_google_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
    variant: GoogleApiVariant,
) -> Value {
    let mut body = Map::new();
    body.insert(
        "contents".to_owned(),
        Value::Array(google_contents(model, context)),
    );
    if !context.system_prompt.is_empty() {
        body.insert(
            "systemInstruction".to_owned(),
            json!({"parts": [{"text": context.system_prompt}]}),
        );
    }
    if !context.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(google_function_tools(&context.tools)),
        );
        if google_uses_validated_tool_mode(model, &context.tools) {
            body.insert(
                "toolConfig".to_owned(),
                json!({"functionCallingConfig": {"mode": "VALIDATED"}}),
            );
        }
    }

    let mut generation_config = Map::new();
    if let Some(max_tokens) = requested_max_tokens(model, context) {
        generation_config.insert(
            "maxOutputTokens".to_owned(),
            Value::Number(max_tokens.into()),
        );
    }
    if let Some(thinking_config) = google_thinking_config(
        model,
        &options.thinking_level,
        options.thinking_budgets.as_ref(),
        variant,
    ) {
        generation_config.insert("thinkingConfig".to_owned(), thinking_config);
    }
    if !generation_config.is_empty() {
        body.insert(
            "generationConfig".to_owned(),
            Value::Object(generation_config),
        );
    }
    Value::Object(body)
}

fn google_function_tools(tools: &[llm::Tool]) -> Vec<Value> {
    let declarations = tools
        .iter()
        .map(|tool| {
            let mut declaration = Map::from_iter([
                ("name".to_owned(), Value::String(tool.name.clone())),
                (
                    "description".to_owned(),
                    Value::String(tool.description.clone()),
                ),
            ]);
            if !tool.parameters.is_null() {
                declaration.insert("parametersJsonSchema".to_owned(), tool.parameters.clone());
            }
            Value::Object(declaration)
        })
        .collect::<Vec<_>>();
    vec![json!({"functionDeclarations": declarations})]
}

fn google_uses_validated_tool_mode(model: &llm::Model, tools: &[llm::Tool]) -> bool {
    let supports_strict = google_supports_strict_tool_sampling(&model.id);
    for tool in tools {
        // The source client treats an unsupported `strict: "require"` hint
        // as a best-effort request for Google models that lack VALIDATED mode.
        if matches!(
            requested_json_schema_strict(tool, supports_strict),
            Ok(Some(true))
        ) {
            return true;
        }
    }
    false
}

fn google_contents(model: &llm::Model, context: &llm::Context) -> Vec<Value> {
    let requires_tool_call_id = google_requires_tool_call_id(&model.id);
    let mut contents = Vec::new();
    for message in transform_google_messages(&context.messages, model) {
        match message {
            llm::Message::User(user) => {
                let parts = google_user_parts(&user.content, model.supports_images());
                if !parts.is_empty() {
                    contents.push(json!({"role": "user", "parts": parts}));
                }
            }
            llm::Message::Assistant(assistant)
                if matches!(
                    assistant.stop_reason.as_str(),
                    stream::STOP_ERROR | stream::STOP_ABORTED
                ) => {}
            llm::Message::Assistant(assistant) => {
                let parts = google_assistant_parts(model, &assistant, requires_tool_call_id);
                if !parts.is_empty() {
                    contents.push(json!({"role": "model", "parts": parts}));
                }
            }
            llm::Message::ToolResult(result) => {
                google_append_tool_result(&mut contents, model, &result, requires_tool_call_id);
            }
        }
    }
    contents
}

/// Replays history in the form accepted by Gemini and Google-hosted models.
///
/// This is intentionally separate from the OpenAI-shaped conversion: Google
/// thought signatures are opaque, only valid for an identical source model,
/// and Google requires a complete function-response sequence.
fn transform_google_messages(messages: &[llm::Message], model: &llm::Model) -> Vec<llm::Message> {
    let requires_tool_call_id = google_requires_tool_call_id(&model.id);
    let mut normalize = |id: &str, _: &llm::AssistantMessage| {
        if requires_tool_call_id {
            sanitize_tool_call_id(id)
        } else {
            id.to_owned()
        }
    };
    transform_messages(messages, model, Some(&mut normalize))
}

/// Rewrites a cross-model tool-call id for the target protocol; the source
/// turn is provided for provider and API checks.
type ToolCallIdNormalizer<'a> = &'a mut dyn FnMut(&str, &llm::AssistantMessage) -> String;

/// pi's `transformMessages`: the one history pass every protocol applies
/// before converting to its wire format.
///
/// Cross-model turns lose provider-specific replay data (thinking becomes
/// plain text, text and thought signatures are dropped, tool-call ids are
/// normalized); same-model turns replay verbatim. Errored or aborted turns
/// are skipped entirely, and tool calls left without a result get a
/// synthetic error result so every protocol sees a well-formed transcript.
pub(crate) fn transform_messages(
    messages: &[llm::Message],
    model: &llm::Model,
    mut normalize_tool_call_id: Option<ToolCallIdNormalizer<'_>>,
) -> Vec<llm::Message> {
    let mut tool_call_ids = BTreeMap::new();
    let mut transformed = Vec::with_capacity(messages.len());

    for message in messages {
        match downgrade_unsupported_images(message.clone(), model) {
            llm::Message::Assistant(assistant) => {
                let same_model = assistant.provider == model.provider
                    && assistant.api == model.api
                    && assistant.model == model.id;
                let mut copy = *assistant;
                let content = std::mem::take(&mut copy.content)
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
                        llm::ContentBlock::Text(text) => Some(if same_model {
                            llm::ContentBlock::Text(text)
                        } else {
                            llm::ContentBlock::text(text.text)
                        }),
                        llm::ContentBlock::ToolCall(mut tool_call) => {
                            if !same_model {
                                tool_call.thought_signature.clear();
                                if let Some(normalize) = normalize_tool_call_id.as_deref_mut() {
                                    let normalized = normalize(&tool_call.id, &copy);
                                    if normalized != tool_call.id {
                                        tool_call_ids
                                            .insert(tool_call.id.clone(), normalized.clone());
                                        tool_call.id = normalized;
                                    }
                                }
                            }
                            Some(llm::ContentBlock::ToolCall(tool_call))
                        }
                        other => Some(other),
                    })
                    .collect();
                copy.content = content;
                transformed.push(llm::Message::Assistant(Box::new(copy)));
            }
            llm::Message::ToolResult(mut tool_result) => {
                if let Some(normalized) = tool_call_ids.get(&tool_result.tool_call_id) {
                    tool_result.tool_call_id = normalized.clone();
                }
                transformed.push(llm::Message::ToolResult(tool_result));
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
                flush_missing_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_results,
                );
                // Incomplete turns are not replayed: partial reasoning or
                // half-finished tool calls make providers reject the request.
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
                flush_missing_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_results,
                );
                result.push(llm::Message::User(user));
            }
        }
    }
    flush_missing_tool_results(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_results,
    );
    result
}

fn downgrade_unsupported_images(message: llm::Message, model: &llm::Model) -> llm::Message {
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
        llm::Message::ToolResult(mut tool_result) => {
            tool_result.content = replace_images_with_placeholder(
                tool_result.content,
                "(tool image omitted: model does not support images)",
            );
            llm::Message::ToolResult(tool_result)
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

fn flush_missing_tool_results(
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

/// Replaces every character outside `[A-Za-z0-9_-]` with an underscore.
fn sanitize_id_part(part: &str) -> String {
    part.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

/// The Anthropic and Google tool-call id rule: sanitized and at most 64
/// characters.
fn sanitize_tool_call_id(id: &str) -> String {
    sanitize_id_part(id).chars().take(64).collect()
}

/// pi's Chat Completions normalizer. Responses-style `call_id|item_id` pairs
/// collapse into one id within OpenAI's 40-character limit, keeping the
/// item part so parallel calls sharing a `call_id` stay distinct.
fn normalize_completions_tool_call_id(id: &str, model: &llm::Model) -> String {
    const MAX_LENGTH: usize = 40;
    if let Some((call_id, item_id)) = id.split_once('|') {
        let call_id = sanitize_id_part(call_id);
        let item_id = sanitize_id_part(item_id);
        let combined = if item_id.is_empty() {
            call_id.clone()
        } else {
            format!("{call_id}_{item_id}")
        };
        if combined.len() <= MAX_LENGTH {
            return combined;
        }
        let hash = responses_short_hash(id).chars().take(8).collect::<String>();
        let prefix = call_id
            .chars()
            .take((MAX_LENGTH - hash.len() - 1).max(1))
            .collect::<String>();
        return format!("{prefix}_{hash}");
    }
    if model.provider == "openai" && id.chars().count() > MAX_LENGTH {
        return id.chars().take(MAX_LENGTH).collect();
    }
    id.to_owned()
}

/// pi's Responses normalizer. Providers that validate `fc_` item ids get a
/// deterministic `fc_` item derived from foreign ids; everything else is
/// simply sanitized.
fn normalize_responses_tool_call_id(
    id: &str,
    model: &llm::Model,
    source: &llm::AssistantMessage,
    allowed_tool_call_providers: &[&str],
) -> String {
    fn normalize_part(part: &str) -> String {
        sanitize_id_part(part)
            .chars()
            .take(64)
            .collect::<String>()
            .trim_end_matches('_')
            .to_owned()
    }
    if !allowed_tool_call_providers.contains(&model.provider.as_str()) {
        return normalize_part(id);
    }
    let Some((call_id, remainder)) = id.split_once('|') else {
        return normalize_part(id);
    };
    let item_id = remainder.split('|').next().unwrap_or_default();
    let call_id = normalize_part(call_id);
    let foreign = source.provider != model.provider || source.api != model.api;
    let mut item_id = if foreign {
        format!("fc_{}", responses_short_hash(item_id))
            .chars()
            .take(64)
            .collect::<String>()
    } else {
        normalize_part(item_id)
    };
    if !item_id.starts_with("fc_") {
        item_id = normalize_part(&format!("fc_{item_id}"));
    }
    format!("{call_id}|{item_id}")
}

fn google_user_parts(content: &llm::UserContent, supports_images: bool) -> Vec<Value> {
    match content {
        llm::UserContent::Text(text) => vec![google_text_part(text, false, None)],
        llm::UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                llm::ContentBlock::Text(text) => Some(google_text_part(&text.text, false, None)),
                llm::ContentBlock::Image(image) if supports_images => Some(json!({
                    "inlineData": {"mimeType": image.mime_type, "data": image.data}
                })),
                llm::ContentBlock::Image(_)
                | llm::ContentBlock::Thinking(_)
                | llm::ContentBlock::ToolCall(_) => None,
            })
            .collect(),
    }
}

fn google_assistant_parts(
    model: &llm::Model,
    assistant: &llm::AssistantMessage,
    requires_tool_call_id: bool,
) -> Vec<Value> {
    let same_model = assistant.provider == model.provider && assistant.model == model.id;
    let mut parts = Vec::new();
    for block in &assistant.content {
        match block {
            llm::ContentBlock::Text(text) => {
                let signature = google_replay_signature(same_model, &text.text_signature);
                if text.text.trim().is_empty() && signature.is_none() {
                    continue;
                }
                parts.push(google_text_part(&text.text, false, signature));
            }
            llm::ContentBlock::Thinking(thinking) if same_model => {
                let signature = google_replay_signature(same_model, &thinking.thinking_signature);
                if thinking.thinking.trim().is_empty() && signature.is_none() {
                    continue;
                }
                parts.push(google_text_part(&thinking.thinking, true, signature));
            }
            llm::ContentBlock::Thinking(thinking) if !thinking.thinking.trim().is_empty() => {
                parts.push(google_text_part(&thinking.thinking, false, None));
            }
            llm::ContentBlock::ToolCall(call) => {
                let id = google_replay_tool_call_id(requires_tool_call_id, &call.id);
                let mut function_call = Map::from_iter([
                    ("name".to_owned(), Value::String(call.name.clone())),
                    (
                        "args".to_owned(),
                        Value::Object(
                            call.arguments
                                .iter()
                                .map(|(name, value)| (name.clone(), value.clone()))
                                .collect(),
                        ),
                    ),
                ]);
                if let Some(id) = id {
                    function_call.insert("id".to_owned(), Value::String(id));
                }
                let mut part =
                    Map::from_iter([("functionCall".to_owned(), Value::Object(function_call))]);
                if let Some(signature) =
                    google_replay_signature(same_model, &call.thought_signature)
                {
                    part.insert("thoughtSignature".to_owned(), Value::String(signature));
                }
                parts.push(Value::Object(part));
            }
            llm::ContentBlock::Image(_) | llm::ContentBlock::Thinking(_) => {}
        }
    }
    parts
}

fn google_append_tool_result(
    contents: &mut Vec<Value>,
    model: &llm::Model,
    result: &llm::ToolResultMessage,
    requires_tool_call_id: bool,
) {
    let mut text = Vec::new();
    let mut images = Vec::new();
    for block in &result.content {
        match block {
            llm::ContentBlock::Text(content) => text.push(content.text.as_str()),
            llm::ContentBlock::Image(image) if model.supports_images() => {
                images.push(json!({
                    "inlineData": {"mimeType": image.mime_type, "data": image.data}
                }));
            }
            llm::ContentBlock::Image(_)
            | llm::ContentBlock::Thinking(_)
            | llm::ContentBlock::ToolCall(_) => {}
        }
    }
    let text = text.join("\n");
    let response_value = if text.is_empty() && !images.is_empty() {
        "(see attached image)".to_owned()
    } else {
        text
    };
    let mut response = Map::new();
    response.insert(
        if result.is_error {
            "error".to_owned()
        } else {
            "output".to_owned()
        },
        Value::String(response_value),
    );
    let mut function_response = Map::from_iter([
        ("name".to_owned(), Value::String(result.tool_name.clone())),
        ("response".to_owned(), Value::Object(response)),
    ]);
    if requires_tool_call_id {
        function_response.insert("id".to_owned(), Value::String(result.tool_call_id.clone()));
    }
    let nests_images = google_supports_multimodal_function_response(&model.id);
    if nests_images && !images.is_empty() {
        function_response.insert("parts".to_owned(), Value::Array(images.clone()));
    }
    let part = Value::Object(Map::from_iter([(
        "functionResponse".to_owned(),
        Value::Object(function_response),
    )]));

    let mut merged = false;
    if let Some(previous) = contents.last_mut()
        && previous.get("role").and_then(Value::as_str) == Some("user")
        && previous
            .get_mut("parts")
            .and_then(Value::as_array_mut)
            .is_some_and(|parts| {
                let has_function_response = parts
                    .iter()
                    .any(|part| part.get("functionResponse").is_some());
                if has_function_response {
                    parts.push(part.clone());
                }
                has_function_response
            })
    {
        merged = true;
    }
    if !merged {
        contents.push(json!({"role": "user", "parts": [part]}));
    }
    if !nests_images && !images.is_empty() {
        let mut image_turn = vec![google_text_part("Tool result image:", false, None)];
        image_turn.extend(images);
        contents.push(json!({"role": "user", "parts": image_turn}));
    }
}

fn google_text_part(text: &str, thought: bool, signature: Option<String>) -> Value {
    let mut part = Map::from_iter([("text".to_owned(), Value::String(text.to_owned()))]);
    if thought {
        part.insert("thought".to_owned(), Value::Bool(true));
    }
    if let Some(signature) = signature {
        part.insert("thoughtSignature".to_owned(), Value::String(signature));
    }
    Value::Object(part)
}

fn google_replay_signature(same_model: bool, signature: &str) -> Option<String> {
    (same_model && google_valid_thought_signature(signature)).then(|| signature.to_owned())
}

fn google_valid_thought_signature(signature: &str) -> bool {
    !signature.is_empty()
        && signature.len().is_multiple_of(4)
        && base64::engine::general_purpose::STANDARD
            .decode(signature)
            .is_ok()
}

fn google_replay_tool_call_id(requires_tool_call_id: bool, id: &str) -> Option<String> {
    requires_tool_call_id.then(|| id.to_owned())
}

fn google_requires_tool_call_id(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    model_id.starts_with("claude-")
        || model_id.starts_with("gpt-oss-")
        || google_gemini_major_version(&model_id).is_some_and(|version| version >= 3)
}

fn google_supports_multimodal_function_response(model_id: &str) -> bool {
    google_gemini_major_version(&model_id.to_ascii_lowercase()).is_none_or(|version| version >= 3)
}

fn google_supports_strict_tool_sampling(model_id: &str) -> bool {
    google_gemini_major_version(&model_id.to_ascii_lowercase()).is_some_and(|version| version >= 3)
}

fn google_gemini_major_version(model_id: &str) -> Option<u32> {
    let model_id = model_id.to_ascii_lowercase();
    let version = model_id
        .strip_prefix("gemini-live-")
        .or_else(|| model_id.strip_prefix("gemini-"))?
        .split('-')
        .next()?
        .split('.')
        .next()?;
    version.parse().ok()
}

fn google_thinking_config(
    model: &llm::Model,
    requested: &str,
    custom_budgets: Option<&llm::ThinkingBudgets>,
    variant: GoogleApiVariant,
) -> Option<Value> {
    if !model.reasoning {
        return None;
    }
    let level = stream::clamp_thinking_level(model, requested);
    if level == llm::THINKING_OFF {
        return Some(Value::Object(google_disabled_thinking_config(
            &model.id, variant,
        )));
    }
    let level = stream::clamp_reasoning_level(&level);
    let mut config = Map::from_iter([("includeThoughts".to_owned(), Value::Bool(true))]);
    if google_uses_thinking_level(&model.id, variant) {
        config.insert(
            "thinkingLevel".to_owned(),
            Value::String(google_thinking_level(&model.id, &level, variant).to_owned()),
        );
    } else {
        config.insert(
            "thinkingBudget".to_owned(),
            Value::Number(google_thinking_budget(model, &level, custom_budgets, variant).into()),
        );
    }
    Some(Value::Object(config))
}

fn google_uses_thinking_level(model_id: &str, variant: GoogleApiVariant) -> bool {
    google_is_gemini_three_pro(model_id)
        || google_is_gemini_three_flash(model_id)
        || matches!(variant, GoogleApiVariant::Generative) && google_is_gemma_four(model_id)
}

fn google_disabled_thinking_config(
    model_id: &str,
    variant: GoogleApiVariant,
) -> Map<String, Value> {
    if google_is_gemini_three_pro(model_id) {
        return Map::from_iter([("thinkingLevel".to_owned(), Value::String("LOW".to_owned()))]);
    }
    if google_is_gemini_three_flash(model_id)
        || matches!(variant, GoogleApiVariant::Generative) && google_is_gemma_four(model_id)
    {
        return Map::from_iter([(
            "thinkingLevel".to_owned(),
            Value::String("MINIMAL".to_owned()),
        )]);
    }
    Map::from_iter([("thinkingBudget".to_owned(), Value::Number(0.into()))])
}

fn google_thinking_level<'a>(model_id: &str, level: &'a str, variant: GoogleApiVariant) -> &'a str {
    if google_is_gemini_three_pro(model_id) {
        return match level {
            llm::THINKING_MINIMAL | llm::THINKING_LOW => "LOW",
            _ => "HIGH",
        };
    }
    if matches!(variant, GoogleApiVariant::Generative) && google_is_gemma_four(model_id) {
        return match level {
            llm::THINKING_MINIMAL | llm::THINKING_LOW => "MINIMAL",
            _ => "HIGH",
        };
    }
    match level {
        llm::THINKING_MINIMAL => "MINIMAL",
        llm::THINKING_LOW => "LOW",
        llm::THINKING_MEDIUM => "MEDIUM",
        _ => "HIGH",
    }
}

fn google_thinking_budget(
    model: &llm::Model,
    level: &str,
    custom_budgets: Option<&llm::ThinkingBudgets>,
    variant: GoogleApiVariant,
) -> i64 {
    if let Some(budget) =
        custom_budgets.and_then(|budgets| google_custom_thinking_budget(budgets, level))
    {
        return i64::from(budget);
    }
    let id = model.id.to_ascii_lowercase();
    match () {
        _ if id.contains("2.5-pro") => google_budget_for_level(level, 128, 2_048, 8_192, 32_768),
        _ if matches!(variant, GoogleApiVariant::Generative) && id.contains("2.5-flash-lite") => {
            google_budget_for_level(level, 512, 2_048, 8_192, 24_576)
        }
        _ if id.contains("2.5-flash-lite") => {
            google_budget_for_level(level, 128, 2_048, 8_192, 24_576)
        }
        _ if id.contains("2.5-flash") => google_budget_for_level(level, 128, 2_048, 8_192, 24_576),
        _ => -1,
    }
}

fn google_custom_thinking_budget(budgets: &llm::ThinkingBudgets, level: &str) -> Option<u32> {
    let budget = match level {
        llm::THINKING_MINIMAL => budgets.minimal,
        llm::THINKING_LOW => budgets.low,
        llm::THINKING_MEDIUM => budgets.medium,
        _ => budgets.high,
    };
    budget.filter(|budget| *budget != 0)
}

fn google_budget_for_level(level: &str, minimal: i64, low: i64, medium: i64, high: i64) -> i64 {
    match level {
        llm::THINKING_MINIMAL => minimal,
        llm::THINKING_LOW => low,
        llm::THINKING_MEDIUM => medium,
        _ => high,
    }
}

fn google_is_gemini_three_pro(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    let Some(version) = model_id.strip_prefix("gemini-") else {
        return false;
    };
    version.strip_prefix('3').is_some_and(|rest| {
        rest.starts_with("-pro") || rest.starts_with(".") && rest.contains("-pro")
    })
}

fn google_is_gemini_three_flash(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    model_id.strip_prefix("gemini-").is_some_and(|version| {
        version.strip_prefix('3').is_some_and(|rest| {
            rest.starts_with("-flash") || rest.starts_with(".") && rest.contains("-flash")
        })
    }) || matches!(
        model_id.as_str(),
        "gemini-flash-latest" | "gemini-flash-lite-latest"
    )
}

fn google_is_gemma_four(model_id: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    model_id.contains("gemma-4") || model_id.contains("gemma4")
}

fn build_openai_responses_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.id.clone()));
    body.insert(
        "input".to_owned(),
        Value::Array(openai_responses_input(model, context)?),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    body.insert("store".to_owned(), Value::Bool(false));
    if let Some(max_tokens) = requested_max_tokens(model, context) {
        body.insert(
            "max_output_tokens".to_owned(),
            Value::Number(max_tokens.max(16).into()),
        );
    }
    if let Some(effort) = mapped_thinking_level(model, &options.thinking_level) {
        body.insert(
            "reasoning".to_owned(),
            json!({"effort": effort, "summary": "auto"}),
        );
        body.insert(
            "include".to_owned(),
            Value::Array(vec![Value::String(
                "reasoning.encrypted_content".to_owned(),
            )]),
        );
    }
    if !options.session_id.is_empty() {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(clamp_prompt_cache_key(&options.session_id)),
        );
    }
    if !context.tools.is_empty() {
        let supports_strict = compat_bool(model, "supportsStrictMode", false);
        body.insert(
            "tools".to_owned(),
            Value::Array(
                context
                    .tools
                    .iter()
                    .map(|tool| {
                        let mut value = json!({
                            "type": "function",
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": schema_or_empty(&tool.parameters),
                        });
                        if supports_strict {
                            value
                                .as_object_mut()
                                .expect("JSON object")
                                .insert("strict".to_owned(), Value::Bool(false));
                        }
                        value
                    })
                    .collect(),
            ),
        );
    }
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

fn build_azure_openai_responses_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
    credentials: &ProviderCredentials,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Value> {
    let tool_options = ResponsesToolOptions {
        supports_strict_mode: compat_bool(model, "supportsStrictMode", true),
        strict_null: false,
        supports_openai_grammar_tools: compat_bool(model, "supportsOpenAIGrammarTools", false),
        defer_loading: false,
    };
    let deferred_tools = BTreeMap::new();
    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(azure_deployment_name(model, credentials)),
    );
    body.insert(
        "input".to_owned(),
        Value::Array(responses_input(
            model,
            context,
            ResponsesInputOptions {
                include_system_prompt: true,
                supports_developer_role: compat_bool(model, "supportsDeveloperRole", true),
                allowed_tool_call_providers: AZURE_TOOL_CALL_PROVIDERS,
                grammar_tool_input_properties,
                deferred_tools: &deferred_tools,
                deferred_tools_mode: None,
                tool_options,
            },
        )?),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    body.insert("store".to_owned(), Value::Bool(false));
    if let Some(max_tokens) = requested_max_tokens(model, context) {
        body.insert(
            "max_output_tokens".to_owned(),
            Value::Number(max_tokens.max(16).into()),
        );
    }
    if !options.session_id.is_empty() {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(clamp_prompt_cache_key(&options.session_id)),
        );
    }
    if !context.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(responses_function_tools(&context.tools, tool_options)?),
        );
    }
    if let Some(effort) = azure_reasoning_effort(model, &options.thinking_level) {
        body.insert(
            "reasoning".to_owned(),
            json!({"effort": effort, "summary": "auto"}),
        );
        body.insert(
            "include".to_owned(),
            Value::Array(vec![Value::String(
                "reasoning.encrypted_content".to_owned(),
            )]),
        );
    }
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

fn azure_reasoning_effort(model: &llm::Model, requested: &str) -> Option<String> {
    if !model.reasoning {
        return None;
    }
    let level = stream::clamp_thinking_level(model, requested);
    if level != llm::THINKING_OFF {
        return Some(
            model
                .thinking_level_map
                .get(&level)
                .and_then(|mapped| mapped.clone())
                .unwrap_or(level),
        );
    }
    match model.thinking_level_map.get(llm::THINKING_OFF) {
        Some(None) => None,
        Some(Some(mapped)) => Some(mapped.clone()),
        None => Some("none".to_owned()),
    }
}

fn build_openai_codex_responses_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<Value> {
    let tool_options = ResponsesToolOptions {
        supports_strict_mode: compat_bool(model, "supportsStrictMode", true),
        strict_null: true,
        supports_openai_grammar_tools: compat_bool(model, "supportsOpenAIGrammarTools", false),
        defer_loading: false,
    };
    let deferred_tools_mode = if compat_bool(model, "supportsAdditionalTools", false) {
        Some(ResponsesDeferredToolsMode::AdditionalTools)
    } else if compat_bool(model, "supportsToolSearch", false) {
        Some(ResponsesDeferredToolsMode::ToolSearch)
    } else {
        None
    };
    let (immediate_tools, deferred_tools) =
        split_responses_deferred_tools(context, deferred_tools_mode.is_some());
    let instructions = if context.system_prompt.is_empty() {
        "You are a helpful assistant.".to_owned()
    } else {
        context.system_prompt.clone()
    };
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.id.clone()));
    body.insert("instructions".to_owned(), Value::String(instructions));
    body.insert(
        "input".to_owned(),
        Value::Array(responses_input(
            model,
            context,
            ResponsesInputOptions {
                include_system_prompt: false,
                supports_developer_role: compat_bool(model, "supportsDeveloperRole", true),
                allowed_tool_call_providers: CODEX_TOOL_CALL_PROVIDERS,
                grammar_tool_input_properties,
                deferred_tools: &deferred_tools,
                deferred_tools_mode,
                tool_options,
            },
        )?),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    body.insert("store".to_owned(), Value::Bool(false));
    body.insert("text".to_owned(), json!({"verbosity": "low"}));
    body.insert(
        "include".to_owned(),
        Value::Array(vec![Value::String(
            "reasoning.encrypted_content".to_owned(),
        )]),
    );
    body.insert("tool_choice".to_owned(), Value::String("auto".to_owned()));
    body.insert("parallel_tool_calls".to_owned(), Value::Bool(true));
    if !options.session_id.is_empty() {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(clamp_prompt_cache_key(&options.session_id)),
        );
    }
    if !immediate_tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(responses_function_tools(&immediate_tools, tool_options)?),
        );
    }
    if let Some(effort) = mapped_thinking_level(model, &options.thinking_level) {
        body.insert(
            "reasoning".to_owned(),
            json!({"effort": effort, "summary": "auto"}),
        );
    }
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

#[derive(Clone, Copy)]
struct ResponsesToolOptions {
    supports_strict_mode: bool,
    strict_null: bool,
    supports_openai_grammar_tools: bool,
    defer_loading: bool,
}

#[derive(Clone, Copy)]
enum ResponsesDeferredToolsMode {
    AdditionalTools,
    ToolSearch,
}

struct ResponsesInputOptions<'a> {
    include_system_prompt: bool,
    supports_developer_role: bool,
    /// Providers whose Responses endpoint validates `fc_` item ids.
    allowed_tool_call_providers: &'static [&'static str],
    grammar_tool_input_properties: &'a BTreeMap<String, String>,
    deferred_tools: &'a BTreeMap<String, llm::Tool>,
    deferred_tools_mode: Option<ResponsesDeferredToolsMode>,
    tool_options: ResponsesToolOptions,
}

fn responses_function_tools(
    tools: &[llm::Tool],
    options: ResponsesToolOptions,
) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) =
                grammar_constrained_sampling(tool, options.supports_openai_grammar_tools)?
            {
                let mut value = json!({
                    "type": "custom",
                    "name": tool.name,
                    "description": tool.description,
                    "format": {
                        "type": "grammar",
                        "syntax": grammar.syntax,
                        "definition": grammar.definition,
                    },
                });
                if options.defer_loading {
                    value
                        .as_object_mut()
                        .expect("Responses custom tool is an object")
                        .insert("defer_loading".to_owned(), Value::Bool(true));
                }
                return Ok(value);
            }
            let requested_strict =
                requested_json_schema_strict(tool, options.supports_strict_mode)?;
            let mut value = json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": schema_or_empty(&tool.parameters),
            });
            if options.defer_loading {
                value
                    .as_object_mut()
                    .expect("Responses function tool is an object")
                    .insert("defer_loading".to_owned(), Value::Bool(true));
            }
            if options.supports_strict_mode {
                value
                    .as_object_mut()
                    .expect("Responses function tool is an object")
                    .insert(
                        "strict".to_owned(),
                        requested_strict.map(Value::Bool).unwrap_or_else(|| {
                            if options.strict_null {
                                Value::Null
                            } else {
                                Value::Bool(false)
                            }
                        }),
                    );
            }
            Ok(value)
        })
        .collect()
}

fn requested_json_schema_strict(
    tool: &llm::Tool,
    supports_strict_mode: bool,
) -> Result<Option<bool>> {
    let Some(config) = tool
        .constrained_sampling
        .as_ref()
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    if config.get("type").and_then(Value::as_str) != Some("json_schema") {
        return Ok(None);
    }
    if supports_strict_mode {
        return Ok(Some(true));
    }
    if config.get("strict").and_then(Value::as_str) == Some("require") {
        return Err(ProviderAdapterError::Protocol(format!(
            "Tool {:?} requires JSON-schema constrained sampling, but strict tools are unsupported",
            tool.name
        )));
    }
    Ok(None)
}

struct GrammarConstrainedSampling {
    syntax: String,
    definition: String,
    input_property: String,
}

fn grammar_constrained_sampling(
    tool: &llm::Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<GrammarConstrainedSampling>> {
    let Some(config) = tool
        .constrained_sampling
        .as_ref()
        .and_then(Value::as_object)
    else {
        return Ok(None);
    };
    if config.get("type").and_then(Value::as_str) != Some("grammar")
        || !supports_openai_grammar_tools
    {
        return Ok(None);
    }
    let error = |message: &str| {
        ProviderAdapterError::Protocol(format!(
            "Tool {:?} cannot use grammar constrained sampling: {message}",
            tool.name
        ))
    };
    let variants = config
        .get("variants")
        .and_then(Value::as_object)
        .ok_or_else(|| error("no supported grammar variant was provided"))?;
    let lark = variants
        .get("openai_lark")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let regex = variants
        .get("openai_regex")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let (syntax, definition) = match (lark, regex) {
        (Some(definition), _) => ("lark", definition),
        (None, Some(definition)) => ("regex", definition),
        (None, None) => {
            return Err(error("no supported grammar variant was provided"));
        }
    };
    let schema = tool
        .parameters
        .as_object()
        .filter(|schema| schema.get("type").and_then(Value::as_str) == Some("object"))
        .ok_or_else(|| error("grammar constrained sampling requires an object parameter schema"))?;
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .filter(|required| required.len() == 1)
        .ok_or_else(|| {
            error("grammar constrained sampling requires exactly one required string property")
        })?;
    let input_property = required
        .first()
        .and_then(Value::as_str)
        .filter(|property| !property.is_empty())
        .ok_or_else(|| {
            error("grammar constrained sampling requires exactly one required string property")
        })?;
    let is_string = schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(input_property))
        .and_then(Value::as_object)
        .and_then(|property| property.get("type"))
        .and_then(Value::as_str)
        == Some("string");
    if !is_string {
        return Err(error(&format!(
            "grammar constrained sampling property {input_property} must have type string"
        )));
    }
    Ok(Some(GrammarConstrainedSampling {
        syntax: syntax.to_owned(),
        definition: definition.to_owned(),
        input_property: input_property.to_owned(),
    }))
}

fn grammar_tool_input_properties(
    tools: &[llm::Tool],
    supports_openai_grammar_tools: bool,
) -> Result<BTreeMap<String, String>> {
    let mut properties = BTreeMap::new();
    for tool in tools {
        if let Some(grammar) = grammar_constrained_sampling(tool, supports_openai_grammar_tools)? {
            properties.insert(tool.name.clone(), grammar.input_property);
        }
    }
    Ok(properties)
}

fn split_responses_deferred_tools(
    context: &llm::Context,
    enabled: bool,
) -> (Vec<llm::Tool>, BTreeMap<String, llm::Tool>) {
    let mut order = Vec::new();
    let mut tools = BTreeMap::new();
    for tool in &context.tools {
        if !tools.contains_key(&tool.name) {
            order.push(tool.name.clone());
        }
        tools.insert(tool.name.clone(), tool.clone());
    }
    if !enabled {
        return (
            order
                .into_iter()
                .filter_map(|name| tools.get(&name).cloned())
                .collect(),
            BTreeMap::new(),
        );
    }

    let mut deferred_names = BTreeSet::new();
    let mut used_names = BTreeSet::new();
    for message in &context.messages {
        match message {
            llm::Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let llm::ContentBlock::ToolCall(call) = block {
                        used_names.insert(call.name.clone());
                    }
                }
            }
            llm::Message::ToolResult(result) => {
                for name in &result.added_tool_names {
                    if !used_names.contains(name) {
                        deferred_names.insert(name.clone());
                    }
                }
            }
            llm::Message::User(_) => {}
        }
    }

    let mut immediate = Vec::new();
    let mut deferred = BTreeMap::new();
    for name in order {
        let Some(tool) = tools.get(&name) else {
            continue;
        };
        if deferred_names.contains(&name) {
            deferred.insert(name, tool.clone());
        } else {
            immediate.push(tool.clone());
        }
    }
    (immediate, deferred)
}

fn responses_short_hash(value: &str) -> String {
    let mut first = 0xdead_beefu32;
    let mut second = 0x41c6_ce57u32;
    for character in value.encode_utf16() {
        first = (first ^ u32::from(character)).wrapping_mul(2_654_435_761);
        second = (second ^ u32::from(character)).wrapping_mul(1_597_334_677);
    }
    first = (first ^ (first >> 16)).wrapping_mul(2_246_822_507)
        ^ (second ^ (second >> 13)).wrapping_mul(3_266_489_909);
    second = (second ^ (second >> 16)).wrapping_mul(2_246_822_507)
        ^ (first ^ (first >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", responses_base36(second), responses_base36(first))
}

fn responses_base36(mut value: u32) -> String {
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

const OPENAI_TOOL_CALL_PROVIDERS: &[&str] = &["openai", "openai-codex", "opencode"];
const AZURE_TOOL_CALL_PROVIDERS: &[&str] = &[
    "openai",
    "openai-codex",
    "opencode",
    "azure-openai-responses",
];
const CODEX_TOOL_CALL_PROVIDERS: &[&str] = &["openai", "openai-codex", "opencode"];

fn openai_responses_input(model: &llm::Model, context: &llm::Context) -> Result<Vec<Value>> {
    let grammar_tool_input_properties = BTreeMap::new();
    let deferred_tools = BTreeMap::new();
    responses_input(
        model,
        context,
        ResponsesInputOptions {
            include_system_prompt: true,
            supports_developer_role: true,
            allowed_tool_call_providers: OPENAI_TOOL_CALL_PROVIDERS,
            grammar_tool_input_properties: &grammar_tool_input_properties,
            deferred_tools: &deferred_tools,
            deferred_tools_mode: None,
            tool_options: ResponsesToolOptions {
                supports_strict_mode: false,
                strict_null: false,
                supports_openai_grammar_tools: false,
                defer_loading: false,
            },
        },
    )
}

fn responses_input(
    model: &llm::Model,
    context: &llm::Context,
    options: ResponsesInputOptions<'_>,
) -> Result<Vec<Value>> {
    let ResponsesInputOptions {
        include_system_prompt,
        supports_developer_role,
        allowed_tool_call_providers,
        grammar_tool_input_properties,
        deferred_tools,
        deferred_tools_mode,
        tool_options,
    } = options;
    let mut normalize = |id: &str, source: &llm::AssistantMessage| {
        normalize_responses_tool_call_id(id, model, source, allowed_tool_call_providers)
    };
    let transformed = transform_messages(&context.messages, model, Some(&mut normalize));
    let mut input = Vec::new();
    if include_system_prompt && !context.system_prompt.is_empty() {
        input.push(json!({
            "role": if model.reasoning && supports_developer_role {
                "developer"
            } else {
                "system"
            },
            "content": context.system_prompt,
        }));
    }

    let mut loaded_tools = BTreeSet::new();
    for (message_index, message) in transformed.iter().enumerate() {
        match message {
            llm::Message::User(user) => {
                let content = responses_user_content(&user.content, model.supports_images());
                if !content.is_empty() {
                    input.push(json!({"role": "user", "content": content}));
                }
            }
            llm::Message::Assistant(assistant) => {
                let same_protocol =
                    assistant.provider == model.provider && assistant.api == model.api;
                let same_model = same_protocol && assistant.model == model.id;
                let different_model = same_protocol && !same_model;
                let mut text_block_index = 0;
                for block in &assistant.content {
                    match block {
                        // The transform keeps signatures only for the same
                        // model, so every surviving one is a replayable item.
                        llm::ContentBlock::Thinking(thinking)
                            if !thinking.thinking_signature.is_empty() =>
                        {
                            if let Ok(item) =
                                serde_json::from_str::<Value>(&thinking.thinking_signature)
                                && item.is_object()
                            {
                                input.push(item);
                            }
                        }
                        llm::ContentBlock::Text(text) => {
                            let (signature_id, phase) = parse_text_signature(&text.text_signature);
                            let fallback = if text_block_index == 0 {
                                format!("msg_pi_{message_index}")
                            } else {
                                format!("msg_pi_{message_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            let id = signature_id.map_or(fallback, |id| {
                                if id.len() > 64 {
                                    format!("msg_{}", responses_short_hash(&id))
                                } else {
                                    id
                                }
                            });
                            let mut item = json!({
                                "type": "message",
                                "id": id,
                                "role": "assistant",
                                "status": "completed",
                                "content": [{
                                    "type": "output_text",
                                    "text": text.text,
                                    "annotations": [],
                                }],
                            });
                            if let Some(phase) = phase {
                                item.as_object_mut()
                                    .expect("JSON object")
                                    .insert("phase".to_owned(), Value::String(phase));
                            }
                            input.push(item);
                        }
                        llm::ContentBlock::ToolCall(call) => {
                            let (call_id, item_id) = split_responses_tool_id(&call.id);
                            let grammar_input_property =
                                grammar_tool_input_properties.get(&call.name);
                            // OpenAI pairs `fc_` items with the reasoning items
                            // of the model that produced them, so another
                            // model's ids are omitted to skip that validation;
                            // a function_call item id must also be `fc_`.
                            let item_id = item_id.filter(|item_id| {
                                let function_call_id = item_id.starts_with("fc_");
                                !((different_model && function_call_id)
                                    || (grammar_input_property.is_none() && !function_call_id))
                            });
                            let mut item = if let Some(input_property) = grammar_input_property {
                                json!({
                                    "type": "custom_tool_call",
                                    "call_id": call_id,
                                    "id": item_id,
                                    "name": call.name,
                                    "input": grammar_tool_input(call, input_property)?,
                                })
                            } else {
                                json!({
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "id": item_id,
                                    "name": call.name,
                                    "arguments": serde_json::to_string(&call.arguments)
                                        .unwrap_or_else(|_| "{}".to_owned()),
                                })
                            };
                            if (same_model || deferred_tools.contains_key(&call.name))
                                && !call.namespace.is_empty()
                            {
                                item.as_object_mut()
                                    .expect("Responses tool call is an object")
                                    .insert(
                                        "namespace".to_owned(),
                                        Value::String(call.namespace.clone()),
                                    );
                            }
                            input.push(item);
                        }
                        llm::ContentBlock::Image(_) | llm::ContentBlock::Thinking(_) => {}
                    }
                }
            }
            llm::Message::ToolResult(result) => {
                let (call_id, _) = split_responses_tool_id(&result.tool_call_id);
                input.push(json!({
                    "type": if grammar_tool_input_properties.contains_key(&result.tool_name) {
                        "custom_tool_call_output"
                    } else {
                        "function_call_output"
                    },
                    "call_id": call_id,
                    "output": responses_tool_result_output(model, &result.content),
                }));
                let Some(deferred_tools_mode) = deferred_tools_mode else {
                    continue;
                };
                let mut announced_tools = Vec::new();
                for name in &result.added_tool_names {
                    if let Some(tool) = deferred_tools.get(name)
                        && loaded_tools.insert(name.clone())
                    {
                        announced_tools.push(tool.clone());
                    }
                }
                if announced_tools.is_empty() {
                    continue;
                }
                match deferred_tools_mode {
                    ResponsesDeferredToolsMode::AdditionalTools => {
                        input.push(json!({
                            "type": "additional_tools",
                            "role": "developer",
                            "tools": responses_function_tools(&announced_tools, tool_options)?,
                        }));
                    }
                    ResponsesDeferredToolsMode::ToolSearch => {
                        let names = announced_tools
                            .iter()
                            .map(|tool| tool.name.as_str())
                            .collect::<Vec<_>>();
                        let call_id = format!(
                            "pi_tool_load_{}",
                            responses_short_hash(&format!(
                                "{}:{}",
                                result.tool_call_id,
                                names.join(",")
                            ))
                        );
                        input.push(json!({
                            "type": "tool_search_call",
                            "call_id": call_id,
                            "execution": "client",
                            "status": "completed",
                            "arguments": {"query": names.join(" "), "limit": names.len()},
                        }));
                        input.push(json!({
                            "type": "tool_search_output",
                            "call_id": call_id,
                            "execution": "client",
                            "status": "completed",
                            "tools": responses_function_tools(
                                &announced_tools,
                                ResponsesToolOptions {
                                    defer_loading: true,
                                    ..tool_options
                                },
                            )?,
                        }));
                    }
                }
            }
        }
    }
    Ok(input)
}

fn grammar_tool_input(call: &llm::ToolCall, input_property: &str) -> Result<String> {
    call.arguments
        .get(input_property)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            ProviderAdapterError::Protocol(format!(
                "grammar tool call {:?} requires argument {:?} to be a string",
                call.name, input_property
            ))
        })
}

fn responses_user_content(content: &llm::UserContent, supports_images: bool) -> Vec<Value> {
    let blocks = content.blocks();
    blocks
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text(text) => Some(json!({
                "type": "input_text",
                "text": text.text,
            })),
            llm::ContentBlock::Image(image) if supports_images => Some(json!({
                "type": "input_image",
                "detail": "auto",
                "image_url": data_uri(image),
            })),
            llm::ContentBlock::Image(_) => Some(json!({
                "type": "input_text",
                "text": "(image omitted: model does not support images)",
            })),
            llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => None,
        })
        .collect()
}

fn responses_tool_result_output(model: &llm::Model, blocks: &[llm::ContentBlock]) -> Value {
    let text = text_from_blocks(blocks);
    let images = blocks
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Image(image) => Some(image),
            _ => None,
        })
        .collect::<Vec<_>>();
    if images.is_empty() || !model.supports_images() {
        return if text.is_empty() {
            if images.is_empty() {
                Value::String("(no tool output)".to_owned())
            } else {
                Value::String("(tool image omitted: model does not support images)".to_owned())
            }
        } else {
            Value::String(text)
        };
    }
    let mut output = Vec::new();
    if !text.is_empty() {
        output.push(json!({"type": "input_text", "text": text}));
    }
    output.extend(images.into_iter().map(|image| {
        json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": data_uri(image),
        })
    }));
    Value::Array(output)
}

const CLAUDE_CODE_VERSION: &str = "2.1.251";
const CLAUDE_CODE_IDENTITY: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
/// Claude Code 2.x tool names; an OAuth session mirrors their casing.
const CLAUDE_CODE_TOOLS: &[&str] = &[
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];
const ANTHROPIC_OAUTH_BETAS: &[&str] = &["claude-code-20250219", "oauth-2025-04-20"];
const ANTHROPIC_FINE_GRAINED_TOOL_STREAMING_BETA: &str = "fine-grained-tool-streaming-2025-05-14";
const ANTHROPIC_INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// Anthropic request shape derived from the credential and turn options.
struct AnthropicRequestShape {
    /// The API key is a Claude Code OAuth token, which pi sends as a bearer
    /// credential with the Claude Code identity and tool naming.
    oauth: bool,
    cache_control: Option<Value>,
    allow_empty_signature: bool,
    /// The clamped thinking level when thinking is on for this request.
    thinking_level: Option<String>,
}

fn anthropic_request_shape(
    model: &llm::Model,
    credentials: &ProviderCredentials,
    options: &agent::RequestOptions,
) -> AnthropicRequestShape {
    let api_key = credentials.api_key_value().unwrap_or_default();
    AnthropicRequestShape {
        oauth: api_key != catalog::AUTHENTICATED_SENTINEL && anthropic_is_oauth_token(api_key),
        cache_control: anthropic_cache_control(model, options.cache_retention),
        allow_empty_signature: compat_bool(model, "allowEmptySignature", false),
        thinking_level: anthropic_thinking_level(model, &options.thinking_level),
    }
}

fn anthropic_is_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

fn to_claude_code_name(name: &str) -> String {
    CLAUDE_CODE_TOOLS
        .iter()
        .find(|tool| tool.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |tool| (*tool).to_owned())
}

fn from_claude_code_name(name: &str, tools: &[llm::Tool]) -> String {
    tools
        .iter()
        .find(|tool| tool.name.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |tool| tool.name.clone())
}

fn anthropic_cache_control(model: &llm::Model, retention: agent::CacheRetention) -> Option<Value> {
    match retention {
        agent::CacheRetention::None => None,
        agent::CacheRetention::Long if compat_bool(model, "supportsLongCacheRetention", true) => {
            Some(json!({"type": "ephemeral", "ttl": "1h"}))
        }
        agent::CacheRetention::Short | agent::CacheRetention::Long => {
            Some(json!({"type": "ephemeral"}))
        }
    }
}

fn anthropic_thinking_level(model: &llm::Model, requested: &str) -> Option<String> {
    if !model.reasoning {
        return None;
    }
    let level = stream::clamp_thinking_level(model, requested);
    (level != llm::THINKING_OFF).then_some(level)
}

/// pi's `mapThinkingLevelToEffort` for adaptive-thinking models.
fn anthropic_effort(model: &llm::Model, level: &str) -> String {
    if let LevelMapping::Value(mapped) = level_mapping(model, level) {
        return mapped.to_owned();
    }
    match level {
        llm::THINKING_MINIMAL | llm::THINKING_LOW => "low",
        llm::THINKING_MEDIUM => "medium",
        _ => "high",
    }
    .to_owned()
}

/// pi's `getBetaFeatures`, less any `anthropic-beta` header configured on the
/// model or credential, which replaces this default wholesale.
fn anthropic_beta_features(
    model: &llm::Model,
    context: &llm::Context,
    shape: &AnthropicRequestShape,
) -> Vec<&'static str> {
    let mut features = Vec::new();
    if shape.oauth {
        features.extend_from_slice(ANTHROPIC_OAUTH_BETAS);
    }
    if !context.tools.is_empty() && !compat_bool(model, "supportsEagerToolInputStreaming", true) {
        features.push(ANTHROPIC_FINE_GRAINED_TOOL_STREAMING_BETA);
    }
    // Adaptive-thinking models interleave thinking without the beta.
    if model.reasoning
        && shape.thinking_level.is_some()
        && !compat_bool(model, "forceAdaptiveThinking", false)
    {
        features.push(ANTHROPIC_INTERLEAVED_THINKING_BETA);
    }
    features
}

fn build_anthropic_messages_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
    shape: &AnthropicRequestShape,
) -> Result<Value> {
    let max_tokens = requested_max_tokens(model, context)
        .or_else(|| (!model.max_tokens.eq(&0)).then_some(model.max_tokens))
        .unwrap_or(1);
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.id.clone()));
    body.insert(
        "messages".to_owned(),
        Value::Array(anthropic_messages(model, context, shape)),
    );
    body.insert("max_tokens".to_owned(), Value::Number(max_tokens.into()));
    body.insert("stream".to_owned(), Value::Bool(true));

    let system_block = |text: &str| {
        let mut block = json!({"type": "text", "text": text});
        if let Some(cache_control) = &shape.cache_control {
            block
                .as_object_mut()
                .expect("JSON object")
                .insert("cache_control".to_owned(), cache_control.clone());
        }
        block
    };
    let mut system = Vec::new();
    // An OAuth token is only accepted with the Claude Code identity.
    if shape.oauth {
        system.push(system_block(CLAUDE_CODE_IDENTITY));
    }
    if !context.system_prompt.is_empty() {
        system.push(system_block(&context.system_prompt));
    }
    if !system.is_empty() {
        body.insert("system".to_owned(), Value::Array(system));
    }

    let thinking_enabled = shape.thinking_level.is_some();
    // Temperature is incompatible with extended thinking.
    if let Some(temperature) = options.temperature
        && !thinking_enabled
        && compat_bool(model, "supportsTemperature", true)
        && let Some(temperature) = serde_json::Number::from_f64(temperature)
    {
        body.insert("temperature".to_owned(), Value::Number(temperature));
    }
    if !context.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(anthropic_tools(&context.tools, model, shape)),
        );
    }
    if model.reasoning {
        if let Some(level) = &shape.thinking_level {
            if compat_bool(model, "forceAdaptiveThinking", false) {
                // Adaptive thinking: Claude decides when and how much to think.
                body.insert(
                    "thinking".to_owned(),
                    json!({"type": "adaptive", "display": "summarized"}),
                );
                body.insert(
                    "output_config".to_owned(),
                    json!({"effort": anthropic_effort(model, level)}),
                );
            } else {
                let budget = thinking_budget(level, options.thinking_budgets.as_ref())
                    .min(max_tokens.saturating_sub(stream::MIN_ANSWER_TOKENS));
                body.insert(
                    "thinking".to_owned(),
                    json!({
                        "type": "enabled",
                        "budget_tokens": if budget == 0 { 1_024 } else { budget },
                        "display": "summarized",
                    }),
                );
            }
        } else if !matches!(level_mapping(model, llm::THINKING_OFF), LevelMapping::Null) {
            body.insert("thinking".to_owned(), json!({"type": "disabled"}));
        }
    }
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

fn anthropic_tools(
    tools: &[llm::Tool],
    model: &llm::Model,
    shape: &AnthropicRequestShape,
) -> Vec<Value> {
    let eager = compat_bool(model, "supportsEagerToolInputStreaming", true);
    let cache_control = compat_bool(model, "supportsCacheControlOnTools", true)
        .then(|| shape.cache_control.clone())
        .flatten();
    tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let mut item = json!({
                "name": if shape.oauth { to_claude_code_name(&tool.name) } else { tool.name.clone() },
                "description": tool.description,
                "input_schema": schema_or_empty(&tool.parameters),
            });
            let object = item.as_object_mut().expect("JSON object");
            if eager {
                object.insert("eager_input_streaming".to_owned(), Value::Bool(true));
            }
            // The cache breakpoint on the last tool covers the whole set.
            if let Some(cache_control) = &cache_control
                && index + 1 == tools.len()
            {
                object.insert("cache_control".to_owned(), cache_control.clone());
            }
            item
        })
        .collect()
}

/// pi's `thinkingBudgetForLevel`: the caller's budgets override the defaults.
fn thinking_budget(level: &str, custom: Option<&llm::ThinkingBudgets>) -> u64 {
    let level = stream::clamp_reasoning_level(level);
    let (default, custom) = match level.as_str() {
        llm::THINKING_MINIMAL => (1_024, custom.and_then(|budgets| budgets.minimal)),
        llm::THINKING_LOW => (2_048, custom.and_then(|budgets| budgets.low)),
        llm::THINKING_MEDIUM => (8_192, custom.and_then(|budgets| budgets.medium)),
        _ => (16_384, custom.and_then(|budgets| budgets.high)),
    };
    custom.map_or(default, u64::from)
}

fn anthropic_messages(
    model: &llm::Model,
    context: &llm::Context,
    shape: &AnthropicRequestShape,
) -> Vec<Value> {
    let mut normalize = |id: &str, _: &llm::AssistantMessage| sanitize_tool_call_id(id);
    let transformed = transform_messages(&context.messages, model, Some(&mut normalize));
    let mut messages = Vec::new();
    let mut index = 0;
    while index < transformed.len() {
        match &transformed[index] {
            llm::Message::User(user) => {
                if let Some(content) =
                    anthropic_user_content(&user.content, model.supports_images())
                {
                    messages.push(json!({"role": "user", "content": content}));
                }
                index += 1;
            }
            llm::Message::Assistant(assistant) => {
                let mut blocks = Vec::new();
                for block in &assistant.content {
                    match block {
                        llm::ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                            blocks.push(json!({"type": "text", "text": text.text}));
                        }
                        llm::ContentBlock::Thinking(thinking) if thinking.redacted => {
                            blocks.push(json!({
                                "type": "redacted_thinking",
                                "data": thinking.thinking_signature,
                            }));
                        }
                        llm::ContentBlock::Thinking(thinking) => {
                            let has_signature = !thinking.thinking_signature.trim().is_empty();
                            if thinking.thinking.trim().is_empty() && !has_signature {
                                continue;
                            }
                            // The transform keeps signatures only for the
                            // same model. A missing one (an aborted stream,
                            // say) is replayed as text unless the provider
                            // accepts empty signatures.
                            if has_signature {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": thinking.thinking,
                                    "signature": thinking.thinking_signature,
                                }));
                            } else if shape.allow_empty_signature {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": thinking.thinking,
                                    "signature": "",
                                }));
                            } else {
                                blocks.push(json!({"type": "text", "text": thinking.thinking}));
                            }
                        }
                        llm::ContentBlock::ToolCall(call) => {
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": call.id,
                                "name": if shape.oauth {
                                    to_claude_code_name(&call.name)
                                } else {
                                    call.name.clone()
                                },
                                "input": call.arguments,
                            }));
                        }
                        llm::ContentBlock::Image(_) | llm::ContentBlock::Text(_) => {}
                    }
                }
                if !blocks.is_empty() {
                    messages.push(json!({"role": "assistant", "content": blocks}));
                }
                index += 1;
            }
            llm::Message::ToolResult(_) => {
                let mut blocks = Vec::new();
                while let Some(llm::Message::ToolResult(result)) = transformed.get(index) {
                    blocks.push(json!({
                        "type": "tool_result",
                        "tool_use_id": result.tool_call_id,
                        "content": anthropic_tool_result_content(
                            &result.content,
                            model.supports_images(),
                        ),
                        "is_error": result.is_error,
                    }));
                    index += 1;
                }
                messages.push(json!({"role": "user", "content": blocks}));
            }
        }
    }
    // A cache breakpoint on the last user block caches the conversation so
    // far for the next turn.
    if let Some(cache_control) = &shape.cache_control
        && let Some(last) = messages.last_mut()
        && last["role"] == "user"
    {
        match &mut last["content"] {
            Value::Array(blocks) => {
                if let Some(block) = blocks.last_mut()
                    && matches!(
                        block["type"].as_str(),
                        Some("text" | "image" | "tool_result")
                    )
                    && let Some(block) = block.as_object_mut()
                {
                    block.insert("cache_control".to_owned(), cache_control.clone());
                }
            }
            Value::String(text) => {
                let text = std::mem::take(text);
                last["content"] = json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": cache_control,
                }]);
            }
            _ => {}
        }
    }
    messages
}

fn anthropic_user_content(content: &llm::UserContent, supports_images: bool) -> Option<Value> {
    match content {
        llm::UserContent::Text(text) if !text.trim().is_empty() => {
            Some(Value::String(text.clone()))
        }
        llm::UserContent::Text(_) => None,
        llm::UserContent::Blocks(blocks) => {
            let blocks = blocks
                .iter()
                .filter_map(|block| match block {
                    llm::ContentBlock::Text(text) if !text.text.trim().is_empty() => {
                        Some(json!({"type": "text", "text": text.text}))
                    }
                    llm::ContentBlock::Image(image) if supports_images => Some(json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.mime_type,
                            "data": image.data,
                        }
                    })),
                    llm::ContentBlock::Image(_) => Some(json!({
                        "type": "text",
                        "text": "(image omitted: model does not support images)",
                    })),
                    llm::ContentBlock::Thinking(_)
                    | llm::ContentBlock::ToolCall(_)
                    | llm::ContentBlock::Text(_) => None,
                })
                .collect::<Vec<_>>();
            (!blocks.is_empty()).then_some(Value::Array(blocks))
        }
    }
}

fn anthropic_tool_result_content(blocks: &[llm::ContentBlock], supports_images: bool) -> Value {
    let text = text_from_blocks(blocks);
    let images = blocks
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Image(image) => Some(image),
            _ => None,
        })
        .collect::<Vec<_>>();
    if images.is_empty() {
        return Value::String(text);
    }
    let mut content = Vec::new();
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    if supports_images {
        content.extend(images.into_iter().map(|image| {
            json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": image.data,
                }
            })
        }));
    } else {
        content.push(json!({
            "type": "text",
            "text": "(tool image omitted: model does not support images)",
        }));
    }
    Value::Array(content)
}

fn schema_or_empty(schema: &Value) -> Value {
    if schema.is_null() {
        json!({"type": "object", "properties": {}})
    } else {
        schema.clone()
    }
}

fn merge_sampling_params(target: &mut Map<String, Value>, model: &llm::Model) {
    if let Some(Value::Object(parameters)) = &model.sampling_params {
        for (name, value) in parameters {
            target.insert(name.clone(), value.clone());
        }
    }
}

fn clamp_prompt_cache_key(session_id: &str) -> String {
    session_id.chars().take(64).collect()
}

fn context_has_tool_history(context: &llm::Context) -> bool {
    context.messages.iter().any(|message| match message {
        llm::Message::ToolResult(_) => true,
        llm::Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::ToolCall(_))),
        llm::Message::User(_) => false,
    })
}

fn text_from_blocks(blocks: &[llm::ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn data_uri(image: &llm::ImageContent) -> String {
    format!("data:{};base64,{}", image.mime_type, image.data)
}

fn parse_text_signature(signature: &str) -> (Option<String>, Option<String>) {
    if signature.is_empty() {
        return (None, None);
    }
    if let Ok(Value::Object(value)) = serde_json::from_str::<Value>(signature)
        && value.get("v").and_then(Value::as_u64) == Some(1)
        && let Some(id) = value.get("id").and_then(Value::as_str)
        && !id.is_empty()
    {
        let phase = value
            .get("phase")
            .and_then(Value::as_str)
            .filter(|phase| matches!(*phase, "commentary" | "final_answer"))
            .map(str::to_owned);
        return (Some(id.to_owned()), phase);
    }
    (Some(signature.to_owned()), None)
}

fn encode_text_signature(id: &str, phase: Option<&str>) -> String {
    let mut value = Map::new();
    value.insert("v".to_owned(), Value::Number(1.into()));
    value.insert("id".to_owned(), Value::String(id.to_owned()));
    if let Some(phase) = phase.filter(|phase| matches!(*phase, "commentary" | "final_answer")) {
        value.insert("phase".to_owned(), Value::String(phase.to_owned()));
    }
    Value::Object(value).to_string()
}

fn split_responses_tool_id(id: &str) -> (String, Option<String>) {
    match id.split_once('|') {
        Some((call_id, item_id)) => (
            call_id.to_owned(),
            (!item_id.is_empty()).then(|| item_id.to_owned()),
        ),
        None => (id.to_owned(), None),
    }
}

/// Mutable output plus safe snapshot publication for one provider turn.
pub(crate) struct MessageEmitter {
    events: stream::AssistantMessageEventStream,
    model: llm::Model,
    cancellation: agent::CancellationToken,
    /// Shared with every published snapshot. Mutation goes through
    /// `Arc::make_mut`, so the message is copied only while a consumer still
    /// holds the previous snapshot rather than on every delta.
    message: Arc<llm::AssistantMessage>,
    usage_cost_multiplier: f64,
}

impl MessageEmitter {
    fn new(
        events: stream::AssistantMessageEventStream,
        model: &llm::Model,
        cancellation: agent::CancellationToken,
    ) -> Self {
        Self {
            events,
            model: model.clone(),
            cancellation,
            message: Arc::new(initial_assistant_message(model)),
            usage_cost_multiplier: 1.0,
        }
    }

    pub(crate) fn message(&self) -> &llm::AssistantMessage {
        &self.message
    }

    pub(crate) fn message_mut(&mut self) -> &mut llm::AssistantMessage {
        Arc::make_mut(&mut self.message)
    }

    fn snapshot(&self) -> Arc<llm::AssistantMessage> {
        Arc::clone(&self.message)
    }

    pub(crate) fn start(&mut self) -> Result<()> {
        self.publish(stream::AssistantMessageEvent::start(self.snapshot()))
    }

    pub(crate) fn start_text(&mut self, initial: &str) -> Result<usize> {
        let index = self.message.content.len();
        self.message_mut()
            .content
            .push(llm::ContentBlock::Text(llm::TextContent {
                text: initial.to_owned(),
                ..llm::TextContent::default()
            }));
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TEXT_START.to_owned(),
            content_index: Some(index),
            ..stream::AssistantMessageEvent::default()
        })?;
        Ok(index)
    }

    pub(crate) fn append_text(&mut self, index: usize, delta: &str) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Text(text)) => text.text.push_str(delta),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "text delta did not match a text content block".to_owned(),
                ));
            }
        }
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TEXT_DELTA.to_owned(),
            content_index: Some(index),
            delta: delta.to_owned(),
            ..stream::AssistantMessageEvent::default()
        })
    }

    fn replace_text(&mut self, index: usize, text: &str) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Text(content)) => content.text = text.to_owned(),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "text completion did not match a text content block".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn set_text_signature(&mut self, index: usize, signature: String) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Text(content)) => content.text_signature = signature,
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "text signature did not match a text content block".to_owned(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn end_text(&mut self, index: usize) -> Result<()> {
        let content = match self.message.content.get(index) {
            Some(llm::ContentBlock::Text(text)) => text.text.clone(),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "text completion did not match a text content block".to_owned(),
                ));
            }
        };
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TEXT_END.to_owned(),
            content_index: Some(index),
            content,
            ..stream::AssistantMessageEvent::default()
        })
    }

    pub(crate) fn start_thinking(
        &mut self,
        initial: &str,
        signature: &str,
        redacted: bool,
    ) -> Result<usize> {
        let index = self.message.content.len();
        self.message_mut()
            .content
            .push(llm::ContentBlock::Thinking(llm::ThinkingContent {
                thinking: initial.to_owned(),
                thinking_signature: signature.to_owned(),
                redacted,
            }));
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_THINKING_START.to_owned(),
            content_index: Some(index),
            ..stream::AssistantMessageEvent::default()
        })?;
        Ok(index)
    }

    pub(crate) fn append_thinking(&mut self, index: usize, delta: &str) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Thinking(thinking)) => thinking.thinking.push_str(delta),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "thinking delta did not match a thinking content block".to_owned(),
                ));
            }
        }
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_THINKING_DELTA.to_owned(),
            content_index: Some(index),
            delta: delta.to_owned(),
            ..stream::AssistantMessageEvent::default()
        })
    }

    fn replace_thinking(&mut self, index: usize, text: &str) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Thinking(thinking)) => thinking.thinking = text.to_owned(),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "thinking completion did not match a thinking content block".to_owned(),
                ));
            }
        }
        Ok(())
    }

    fn append_thinking_signature(&mut self, index: usize, delta: &str) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Thinking(thinking)) => {
                thinking.thinking_signature.push_str(delta);
                Ok(())
            }
            _ => Err(ProviderAdapterError::Protocol(
                "thinking signature did not match a thinking content block".to_owned(),
            )),
        }
    }

    fn set_thinking_signature(&mut self, index: usize, signature: String) -> Result<()> {
        match self.message_mut().content.get_mut(index) {
            Some(llm::ContentBlock::Thinking(thinking)) => {
                thinking.thinking_signature = signature;
                Ok(())
            }
            _ => Err(ProviderAdapterError::Protocol(
                "thinking signature did not match a thinking content block".to_owned(),
            )),
        }
    }

    pub(crate) fn end_thinking(&mut self, index: usize) -> Result<()> {
        let content = match self.message.content.get(index) {
            Some(llm::ContentBlock::Thinking(thinking)) => thinking.thinking.clone(),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "thinking completion did not match a thinking content block".to_owned(),
                ));
            }
        };
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_THINKING_END.to_owned(),
            content_index: Some(index),
            content,
            ..stream::AssistantMessageEvent::default()
        })
    }

    pub(crate) fn start_tool(&mut self, id: &str, name: &str) -> Result<usize> {
        let index = self.message.content.len();
        self.message_mut()
            .content
            .push(llm::ContentBlock::ToolCall(llm::ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments: BTreeMap::new(),
                thought_signature: String::new(),
                namespace: String::new(),
            }));
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TOOLCALL_START.to_owned(),
            content_index: Some(index),
            ..stream::AssistantMessageEvent::default()
        })?;
        Ok(index)
    }

    fn set_tool_metadata(
        &mut self,
        index: usize,
        id: Option<&str>,
        name: Option<&str>,
        namespace: Option<&str>,
    ) -> Result<()> {
        let Some(llm::ContentBlock::ToolCall(call)) = self.message_mut().content.get_mut(index)
        else {
            return Err(ProviderAdapterError::Protocol(
                "tool metadata did not match a tool-call content block".to_owned(),
            ));
        };
        if let Some(id) = id.filter(|id| !id.is_empty()) {
            call.id = id.to_owned();
        }
        if let Some(name) = name.filter(|name| !name.is_empty()) {
            call.name = name.to_owned();
        }
        if let Some(namespace) = namespace {
            call.namespace = namespace.to_owned();
        }
        Ok(())
    }

    fn set_tool_thought_signature(&mut self, index: usize, signature: &str) -> Result<()> {
        let Some(llm::ContentBlock::ToolCall(call)) = self.message_mut().content.get_mut(index)
        else {
            return Err(ProviderAdapterError::Protocol(
                "tool thought signature did not match a tool-call content block".to_owned(),
            ));
        };
        if !signature.is_empty() {
            call.thought_signature = signature.to_owned();
        }
        Ok(())
    }

    pub(crate) fn set_tool_arguments(
        &mut self,
        index: usize,
        arguments: BTreeMap<String, Value>,
    ) -> Result<()> {
        let Some(llm::ContentBlock::ToolCall(call)) = self.message_mut().content.get_mut(index)
        else {
            return Err(ProviderAdapterError::Protocol(
                "tool arguments did not match a tool-call content block".to_owned(),
            ));
        };
        call.arguments = arguments;
        Ok(())
    }

    pub(crate) fn tool_delta(&mut self, index: usize, delta: &str) -> Result<()> {
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TOOLCALL_DELTA.to_owned(),
            content_index: Some(index),
            delta: delta.to_owned(),
            ..stream::AssistantMessageEvent::default()
        })
    }

    pub(crate) fn end_tool(&mut self, index: usize) -> Result<()> {
        let call = match self.message.content.get(index) {
            Some(llm::ContentBlock::ToolCall(call)) => call.clone(),
            _ => {
                return Err(ProviderAdapterError::Protocol(
                    "tool completion did not match a tool-call content block".to_owned(),
                ));
            }
        };
        self.publish(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TOOLCALL_END.to_owned(),
            content_index: Some(index),
            tool_call: Some(call),
            ..stream::AssistantMessageEvent::default()
        })
    }

    fn set_usage_cost_multiplier(&mut self, multiplier: f64) {
        self.usage_cost_multiplier = multiplier;
    }

    fn calculate_usage_cost(&mut self) {
        let Self {
            model,
            message,
            usage_cost_multiplier,
            ..
        } = self;
        let usage = &mut Arc::make_mut(message).usage;
        stream::calculate_usage_cost(model, usage);
        if *usage_cost_multiplier != 1.0 {
            let cost = &mut usage.cost;
            cost.input *= *usage_cost_multiplier;
            cost.output *= *usage_cost_multiplier;
            cost.cache_read *= *usage_cost_multiplier;
            cost.cache_write *= *usage_cost_multiplier;
            cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
        }
    }

    fn finish(&mut self) -> Result<()> {
        self.calculate_usage_cost();
        let event =
            stream::AssistantMessageEvent::done(self.message.stop_reason.clone(), self.snapshot());
        let result = self.deliver_terminal(event);
        self.events.end();
        result
    }

    fn fail(&mut self, error: ProviderAdapterError) -> Result<()> {
        let aborted =
            self.cancellation.is_cancelled() || matches!(&error, ProviderAdapterError::Cancelled);
        {
            let message = self.message_mut();
            message.stop_reason = if aborted {
                stream::STOP_ABORTED.to_owned()
            } else {
                stream::STOP_ERROR.to_owned()
            };
            message.error_message = error.to_string();
        }
        self.calculate_usage_cost();
        let event =
            stream::AssistantMessageEvent::error(self.message.stop_reason.clone(), self.snapshot());
        let result = self.deliver_terminal(event);
        self.events.end();
        result
    }

    fn publish(&self, mut event: stream::AssistantMessageEvent) -> Result<()> {
        event.partial = Some(self.snapshot());
        let mut pending = event;
        loop {
            match self.events.push_timeout(pending, STREAM_POLL_INTERVAL) {
                Ok(()) => return Ok(()),
                Err(stream::EventStreamPushError::TimedOut(event)) => {
                    // A consumer that stopped draining must not pin this
                    // worker: stop once the turn is cancelled or the last
                    // consumer handle is gone.
                    if self.cancellation.is_cancelled() {
                        return Err(ProviderAdapterError::Cancelled);
                    }
                    if self.events.is_orphaned() {
                        return Err(ProviderAdapterError::EventStream(
                            "no consumer remains for the assistant event stream".to_owned(),
                        ));
                    }
                    pending = *event;
                }
                Err(error) => return Err(ProviderAdapterError::EventStream(error.to_string())),
            }
        }
    }

    /// Terminal events are still delivered after cancellation so a consumer
    /// that is draining sees the final message, but a stalled one only holds
    /// this worker for a bounded time.
    fn deliver_terminal(&self, event: stream::AssistantMessageEvent) -> Result<()> {
        let deadline = Instant::now()
            .checked_add(TERMINAL_DELIVERY_BUDGET)
            .unwrap_or_else(Instant::now);
        let mut pending = event;
        loop {
            match self.events.push_timeout(pending, STREAM_POLL_INTERVAL) {
                Ok(()) => return Ok(()),
                Err(stream::EventStreamPushError::TimedOut(event)) => {
                    if self.events.is_orphaned() || Instant::now() >= deadline {
                        return Err(ProviderAdapterError::EventStream(
                            "consumer did not accept the terminal assistant event".to_owned(),
                        ));
                    }
                    pending = *event;
                }
                Err(error) => return Err(ProviderAdapterError::EventStream(error.to_string())),
            }
        }
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

fn value_string<'a>(object: &'a Value, field: &str) -> Option<&'a str> {
    object.get(field).and_then(Value::as_str)
}

fn value_u64(object: &Value, field: &str) -> Option<u64> {
    object.get(field).and_then(Value::as_u64).or_else(|| {
        object
            .get(field)
            .and_then(Value::as_i64)
            .filter(|value| *value >= 0)
            .map(|value| value as u64)
    })
}

fn value_object<'a>(object: &'a Value, field: &str) -> Option<&'a Map<String, Value>> {
    object.get(field).and_then(Value::as_object)
}

fn btree_arguments(value: Option<&Map<String, Value>>) -> BTreeMap<String, Value> {
    value
        .into_iter()
        .flat_map(|value| value.iter())
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn apply_openai_usage(usage: &mut llm::Usage, raw: &Value) {
    let prompt_tokens = value_u64(raw, "prompt_tokens").unwrap_or(usage.input);
    let output_tokens = value_u64(raw, "completion_tokens").unwrap_or(usage.output);
    let details = value_object(raw, "prompt_tokens_details");
    let cache_read = details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| value_u64(raw, "prompt_cache_hit_tokens"))
        // Kimi documents cache hits as a top-level `cached_tokens`.
        .or_else(|| value_u64(raw, "cached_tokens"))
        .unwrap_or(usage.cache_read);
    let cache_write = details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_write);
    usage.input = prompt_tokens.saturating_sub(cache_read.saturating_add(cache_write));
    usage.output = output_tokens;
    usage.cache_read = cache_read;
    usage.cache_write = cache_write;
    if let Some(reasoning) = value_object(raw, "completion_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64)
    {
        usage.reasoning = Some(reasoning);
    }
    usage.total_tokens = value_u64(raw, "total_tokens").unwrap_or_else(|| {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write)
    });
}

fn apply_responses_usage(usage: &mut llm::Usage, raw: &Value) {
    let input_tokens = value_u64(raw, "input_tokens").unwrap_or(usage.input);
    let output_tokens = value_u64(raw, "output_tokens").unwrap_or(usage.output);
    let details = value_object(raw, "input_tokens_details");
    let cache_read = details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_read);
    let cache_write = details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(usage.cache_write);
    usage.input = input_tokens.saturating_sub(cache_read.saturating_add(cache_write));
    usage.output = output_tokens;
    usage.cache_read = cache_read;
    usage.cache_write = cache_write;
    if let Some(reasoning) = value_object(raw, "output_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_u64)
    {
        usage.reasoning = Some(reasoning);
    }
    usage.total_tokens = value_u64(raw, "total_tokens").unwrap_or_else(|| {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write)
    });
}

fn apply_anthropic_usage(usage: &mut llm::Usage, raw: &Value) {
    if let Some(value) = value_u64(raw, "input_tokens") {
        usage.input = value;
    }
    if let Some(value) = value_u64(raw, "output_tokens") {
        usage.output = value;
    }
    if let Some(value) = value_u64(raw, "cache_read_input_tokens") {
        usage.cache_read = value;
    }
    if let Some(value) = value_u64(raw, "cache_creation_input_tokens") {
        usage.cache_write = value;
    }
    if let Some(value) = value_object(raw, "cache_creation")
        .and_then(|value| value.get("ephemeral_1h_input_tokens"))
        .and_then(Value::as_u64)
    {
        usage.cache_write_1h = Some(value);
    }
    if let Some(value) = value_object(raw, "output_tokens_details")
        .and_then(|value| value.get("thinking_tokens"))
        .and_then(Value::as_u64)
    {
        usage.reasoning = Some(value);
    }
    usage.total_tokens = usage
        .input
        .saturating_add(usage.output)
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
}

fn map_openai_stop_reason(raw: &str) -> (String, String) {
    match raw {
        "stop" | "end" => (stream::STOP_STOP.to_owned(), String::new()),
        "length" => (stream::STOP_LENGTH.to_owned(), String::new()),
        "function_call" | "tool_calls" => (stream::STOP_TOOL_USE.to_owned(), String::new()),
        "content_filter" | "network_error" => (
            stream::STOP_ERROR.to_owned(),
            format!("Provider finish_reason: {raw}"),
        ),
        other => (
            stream::STOP_ERROR.to_owned(),
            format!("Provider finish_reason: {other}"),
        ),
    }
}

struct OpenAiToolState {
    content_index: usize,
    arguments: stream::IncrementalJsonObjectParser,
}

/// Formats a Chat Completions `{"error": …}` chunk the way pi surfaces it:
/// the provider's message, plus OpenRouter's raw upstream metadata when it
/// adds something.
fn openai_stream_error_message(error: &Value) -> String {
    let mut message = value_string(error, "message")
        .filter(|message| !message.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| error.to_string());
    if let Some(raw) = value_object(error, "metadata")
        .and_then(|metadata| metadata.get("raw"))
        .and_then(Value::as_str)
        .filter(|raw| !raw.is_empty() && !message.contains(raw))
    {
        message.push('\n');
        message.push_str(raw);
    }
    message
}

fn consume_openai_completions(
    body: impl Read,
    model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let compat = OpenAiCompletionsCompat::from_model(model);
    let mut reader = stream::SseReader::new(body);
    let mut text_index = None;
    let mut thinking_index = None;
    let mut tool_calls = BTreeMap::<usize, OpenAiToolState>::new();
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
        // OpenAI-compatible gateways can inject non-JSON keepalives; their
        // SDKs ignore those records, so keep that tolerant behavior.
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        // The OpenAI SDK raises a chunk carrying an `error` object as the
        // request's failure; treating it as a truncated stream would hide the
        // provider's message behind a retryable "ended without finish_reason".
        if let Some(error) = chunk.get("error").filter(|error| !error.is_null()) {
            return Err(ProviderAdapterError::Protocol(openai_stream_error_message(
                error,
            )));
        }
        if emitter.message().response_id.is_empty()
            && let Some(id) = value_string(&chunk, "id")
        {
            emitter.message_mut().response_id = id.to_owned();
        }
        if let Some(response_model) = value_string(&chunk, "model")
            && response_model != model.id
            && !response_model.is_empty()
            && emitter.message().response_model.is_empty()
        {
            emitter.message_mut().response_model = response_model.to_owned();
        }
        if let Some(usage) = chunk.get("usage") {
            apply_openai_usage(&mut emitter.message_mut().usage, usage);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            continue;
        };
        if let Some(usage) = choice.get("usage") {
            apply_openai_usage(&mut emitter.message_mut().usage, usage);
        }
        if let Some(reason) =
            value_string(choice, "finish_reason").filter(|reason| !reason.is_empty())
        {
            saw_finish_reason = true;
            emitter.message_mut().raw_stop_reason = reason.to_owned();
            let (stop_reason, error_message) = map_openai_stop_reason(reason);
            emitter.message_mut().stop_reason = stop_reason;
            if !error_message.is_empty() {
                emitter.message_mut().error_message = error_message;
            }
        }

        let Some(delta) = value_object(choice, "delta") else {
            continue;
        };
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
        {
            let index = match text_index {
                Some(index) => index,
                None => {
                    let index = emitter.start_text("")?;
                    text_index = Some(index);
                    index
                }
            };
            emitter.append_text(index, text)?;
        }
        let reasoning = ["reasoning_content", "reasoning", "reasoning_text"]
            .iter()
            .find_map(|field| {
                delta
                    .get(*field)
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .map(|value| (*field, value))
            });
        if let Some((source, delta)) = reasoning {
            let index = match thinking_index {
                Some(index) => index,
                None => {
                    let index = emitter.start_thinking("", source, false)?;
                    thinking_index = Some(index);
                    index
                }
            };
            emitter.append_thinking(index, delta)?;
        }
        let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for (ordinal, call) in calls.iter().enumerate() {
            let key = call
                .get("index")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(ordinal);
            let function = value_object(call, "function");
            let custom = value_object(call, "custom");
            let name = function
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .or_else(|| {
                    custom
                        .and_then(|custom| custom.get("name"))
                        .and_then(Value::as_str)
                })
                .unwrap_or_default();
            let id = value_string(call, "id").unwrap_or_default();
            if let Entry::Vacant(entry) = tool_calls.entry(key) {
                let content_index = emitter.start_tool(id, name)?;
                entry.insert(OpenAiToolState {
                    content_index,
                    arguments: stream::IncrementalJsonObjectParser::new(),
                });
            }
            let (content_index, arguments) = {
                let state = tool_calls
                    .get_mut(&key)
                    .expect("state inserted for an OpenAI tool call");
                let arguments = function
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .or_else(|| {
                        custom
                            .and_then(|custom| custom.get("input"))
                            .and_then(Value::as_str)
                    })
                    .unwrap_or_default();
                if !arguments.is_empty() {
                    state.arguments.push(arguments);
                }
                (state.content_index, arguments.to_owned())
            };
            emitter.set_tool_metadata(content_index, Some(id), Some(name), None)?;
            if !arguments.is_empty() {
                let preview = tool_calls
                    .get(&key)
                    .expect("state present")
                    .arguments
                    .tool_arguments();
                emitter.set_tool_arguments(content_index, preview)?;
                emitter.tool_delta(content_index, &arguments)?;
            }
        }
    }

    if let Some(index) = text_index {
        emitter.end_text(index)?;
    }
    if let Some(index) = thinking_index {
        emitter.end_thinking(index)?;
    }
    for state in tool_calls.values_mut() {
        emitter.set_tool_arguments(state.content_index, state.arguments.finish_tool_arguments())?;
        emitter.end_tool(state.content_index)?;
    }
    if !saw_finish_reason && !compat.supports_finish_reason {
        emitter.message_mut().stop_reason = if tool_calls.is_empty() {
            stream::STOP_STOP.to_owned()
        } else {
            stream::STOP_TOOL_USE.to_owned()
        };
    }
    if emitter.message().stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message().error_message.clone(),
        ));
    }
    if !saw_finish_reason && compat.supports_finish_reason {
        return Err(ProviderAdapterError::Protocol(
            "OpenAI stream ended without finish_reason".to_owned(),
        ));
    }
    Ok(())
}

fn consume_google_generate_content(
    body: impl Read,
    _model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let mut reader = stream::SseReader::new(body);
    let mut text_index = None;
    let mut thinking_index = None;
    let mut used_tool_call_ids = BTreeSet::new();
    let mut saw_finish_reason = false;

    while let Some(event) = reader.next_event()? {
        ensure_not_cancelled(cancellation)?;
        let data = event.data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            continue;
        }
        let chunk = serde_json::from_str::<Value>(data)?;
        if emitter.message().response_id.is_empty()
            && let Some(response_id) =
                value_string(&chunk, "responseId").filter(|response_id| !response_id.is_empty())
        {
            emitter.message_mut().response_id = response_id.to_owned();
        }
        if let Some(usage) = chunk.get("usageMetadata") {
            apply_google_usage(&mut emitter.message_mut().usage, usage);
        }

        let Some(candidate) = chunk
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        else {
            continue;
        };
        if let Some(reason) =
            value_string(candidate, "finishReason").filter(|reason| !reason.is_empty())
        {
            saw_finish_reason = true;
            emitter.message_mut().raw_stop_reason = reason.to_owned();
            let (stop_reason, error_message) = map_google_stop_reason(reason);
            emitter.message_mut().stop_reason = stop_reason;
            emitter.message_mut().error_message = if error_message.is_empty() {
                String::new()
            } else {
                format!("provider stopped with: {reason}")
            };
        }
        let Some(parts) = candidate
            .get("content")
            .and_then(Value::as_object)
            .and_then(|content| content.get("parts"))
            .and_then(Value::as_array)
        else {
            continue;
        };

        for part in parts {
            let signature = value_string(part, "thoughtSignature").unwrap_or_default();
            if let Some(text) = value_string(part, "text") {
                let thinking = part
                    .get("thought")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                google_append_stream_text(
                    emitter,
                    &mut text_index,
                    &mut thinking_index,
                    text,
                    thinking,
                    signature,
                )?;
            }
            let Some(function_call) = value_object(part, "functionCall") else {
                continue;
            };
            google_finish_stream_block(emitter, &mut text_index, &mut thinking_index)?;
            let name = function_call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let requested_id = function_call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = google_stream_tool_call_id(name, requested_id, &mut used_tool_call_ids);
            let arguments = btree_arguments(function_call.get("args").and_then(Value::as_object));
            let content_index = emitter.start_tool(&id, name)?;
            emitter.set_tool_arguments(content_index, arguments.clone())?;
            emitter.set_tool_thought_signature(content_index, signature)?;
            let serialized = Value::Object(
                arguments
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect(),
            )
            .to_string();
            emitter.tool_delta(content_index, &serialized)?;
            emitter.end_tool(content_index)?;
        }
    }

    google_finish_stream_block(emitter, &mut text_index, &mut thinking_index)?;
    if !saw_finish_reason {
        return Err(ProviderAdapterError::Protocol(
            "Google stream ended without finishReason".to_owned(),
        ));
    }
    if emitter.message().stop_reason == stream::STOP_STOP
        && emitter
            .message
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::ToolCall(_)))
    {
        emitter.message_mut().stop_reason = stream::STOP_TOOL_USE.to_owned();
    }
    if emitter.message().stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message().error_message.clone(),
        ));
    }
    Ok(())
}

fn google_append_stream_text(
    emitter: &mut MessageEmitter,
    text_index: &mut Option<usize>,
    thinking_index: &mut Option<usize>,
    delta: &str,
    thinking: bool,
    signature: &str,
) -> Result<()> {
    if thinking {
        if let Some(index) = text_index.take() {
            emitter.end_text(index)?;
        }
        let index = match *thinking_index {
            Some(index) => index,
            None => {
                let index = emitter.start_thinking("", "", false)?;
                *thinking_index = Some(index);
                index
            }
        };
        emitter.append_thinking(index, delta)?;
        if !signature.is_empty() {
            emitter.set_thinking_signature(index, signature.to_owned())?;
        }
    } else {
        if let Some(index) = thinking_index.take() {
            emitter.end_thinking(index)?;
        }
        let index = match *text_index {
            Some(index) => index,
            None => {
                let index = emitter.start_text("")?;
                *text_index = Some(index);
                index
            }
        };
        emitter.append_text(index, delta)?;
        if !signature.is_empty() {
            emitter.set_text_signature(index, signature.to_owned())?;
        }
    }
    Ok(())
}

fn google_finish_stream_block(
    emitter: &mut MessageEmitter,
    text_index: &mut Option<usize>,
    thinking_index: &mut Option<usize>,
) -> Result<()> {
    if let Some(index) = text_index.take() {
        emitter.end_text(index)?;
    }
    if let Some(index) = thinking_index.take() {
        emitter.end_thinking(index)?;
    }
    Ok(())
}

fn google_stream_tool_call_id(name: &str, requested: &str, used: &mut BTreeSet<String>) -> String {
    if !requested.is_empty() && requested != "null" && used.insert(requested.to_owned()) {
        return requested.to_owned();
    }
    let name = if name.is_empty() { "tool" } else { name };
    loop {
        let sequence = GOOGLE_TOOL_CALL_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let generated = format!("{name}_{}_{}", now_millis(), sequence);
        if used.insert(generated.clone()) {
            return generated;
        }
    }
}

fn apply_google_usage(usage: &mut llm::Usage, raw: &Value) {
    let prompt = value_u64(raw, "promptTokenCount").unwrap_or_default();
    let cached = value_u64(raw, "cachedContentTokenCount").unwrap_or_default();
    let output = value_u64(raw, "candidatesTokenCount").unwrap_or_default();
    let thinking = value_u64(raw, "thoughtsTokenCount").unwrap_or(0);
    usage.input = prompt.saturating_sub(cached);
    usage.output = output.saturating_add(thinking);
    usage.cache_read = cached;
    usage.cache_write = 0;
    usage.reasoning = Some(thinking);
    usage.total_tokens = value_u64(raw, "totalTokenCount").unwrap_or_default();
}

fn map_google_stop_reason(raw: &str) -> (String, String) {
    match raw.to_ascii_uppercase().as_str() {
        "STOP" => (stream::STOP_STOP.to_owned(), String::new()),
        "MAX_TOKENS" => (stream::STOP_LENGTH.to_owned(), String::new()),
        "MALFORMED_FUNCTION_CALL" => (
            stream::STOP_ERROR.to_owned(),
            "Google stopped due to a malformed function call".to_owned(),
        ),
        other => (
            stream::STOP_ERROR.to_owned(),
            format!("Google finishReason: {other}"),
        ),
    }
}

enum ResponsesSlot {
    Text {
        content_index: usize,
    },
    Thinking {
        content_index: usize,
        item_id: String,
    },
    Function {
        content_index: usize,
        arguments: stream::IncrementalJsonObjectParser,
    },
    Custom {
        content_index: usize,
        input: String,
        input_property: String,
        buffer: GrammarToolInputBuffer,
    },
}

#[derive(Default)]
struct GrammarToolInputBuffer {
    input: String,
    started: bool,
    closed: bool,
}

fn append_grammar_tool_input_json_delta(
    buffer: &mut GrammarToolInputBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<String> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(String::new());
        }
        return Err(ProviderAdapterError::Protocol(format!(
            "grammar tool input for property {input_property:?} changed after it was closed"
        )));
    }
    if !next_input.starts_with(&buffer.input) {
        return Err(ProviderAdapterError::Protocol(format!(
            "grammar tool input for property {input_property:?} changed non-monotonically"
        )));
    }
    let input_delta = &next_input[buffer.input.len()..];
    if !close && input_delta.is_empty() {
        return Ok(String::new());
    }
    let mut delta = String::new();
    if !buffer.started {
        delta.push('{');
        delta.push_str(
            &serde_json::to_string(input_property)
                .expect("serializing a grammar tool property cannot fail"),
        );
        delta.push_str(":\"");
        buffer.started = true;
    }
    let escaped = serde_json::to_string(input_delta)
        .expect("serializing a grammar tool input delta cannot fail");
    delta.push_str(&escaped[1..escaped.len() - 1]);
    buffer.input = next_input.to_owned();
    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(delta)
}

fn response_item_type(item: &Value) -> &str {
    value_string(item, "type").unwrap_or_default()
}

fn response_tool_call_id(item: &Value) -> String {
    let call_id = value_string(item, "call_id").unwrap_or_default();
    let item_id = value_string(item, "id").unwrap_or_default();
    if item_id.is_empty() {
        call_id.to_owned()
    } else {
        format!("{call_id}|{item_id}")
    }
}

fn start_responses_slot(
    output_index: usize,
    item: &Value,
    slots: &mut BTreeMap<usize, ResponsesSlot>,
    emitter: &mut MessageEmitter,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<()> {
    if slots.contains_key(&output_index) {
        return Ok(());
    }
    let slot = match response_item_type(item) {
        "message" => ResponsesSlot::Text {
            content_index: emitter.start_text("")?,
        },
        "reasoning" => ResponsesSlot::Thinking {
            content_index: emitter.start_thinking("", "", false)?,
            item_id: value_string(item, "id").unwrap_or_default().to_owned(),
        },
        "function_call" => {
            let content_index = emitter.start_tool(
                &response_tool_call_id(item),
                value_string(item, "name").unwrap_or_default(),
            )?;
            if let Some(namespace) = value_string(item, "namespace") {
                emitter.set_tool_metadata(content_index, None, None, Some(namespace))?;
            }
            let initial = value_string(item, "arguments").unwrap_or_default();
            let mut arguments = stream::IncrementalJsonObjectParser::new();
            if !initial.is_empty() {
                arguments.push(initial);
                emitter.set_tool_arguments(content_index, arguments.tool_arguments())?;
            }
            ResponsesSlot::Function {
                content_index,
                arguments,
            }
        }
        "custom_tool_call" => {
            let input = value_string(item, "input").unwrap_or_default().to_owned();
            let input_property = grammar_tool_input_properties
                .get(value_string(item, "name").unwrap_or_default())
                .cloned()
                .unwrap_or_else(|| "input".to_owned());
            let content_index = emitter.start_tool(
                &response_tool_call_id(item),
                value_string(item, "name").unwrap_or_default(),
            )?;
            let mut arguments = BTreeMap::new();
            arguments.insert(input_property.clone(), Value::String(input.clone()));
            emitter.set_tool_arguments(content_index, arguments)?;
            if let Some(namespace) = value_string(item, "namespace") {
                emitter.set_tool_metadata(content_index, None, None, Some(namespace))?;
            }
            ResponsesSlot::Custom {
                content_index,
                input,
                input_property,
                buffer: GrammarToolInputBuffer::default(),
            }
        }
        _ => return Ok(()),
    };
    slots.insert(output_index, slot);
    Ok(())
}

fn response_item_text(item: &Value) -> String {
    item.get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| {
                    value_string(part, "text")
                        .or_else(|| value_string(part, "refusal"))
                        .filter(|text| !text.is_empty())
                })
                .collect::<Vec<_>>()
                .join("")
        })
        .unwrap_or_default()
}

fn response_item_reasoning_text(item: &Value) -> String {
    let summary = item
        .get("summary")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| value_string(part, "text"))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default();
    if !summary.is_empty() {
        return summary;
    }
    item.get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| value_string(part, "text"))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default()
}

fn finish_responses_item(
    output_index: usize,
    item: &Value,
    slots: &mut BTreeMap<usize, ResponsesSlot>,
    reasoning_blocks: &mut BTreeMap<String, usize>,
    emitter: &mut MessageEmitter,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<()> {
    start_responses_slot(
        output_index,
        item,
        slots,
        emitter,
        grammar_tool_input_properties,
    )?;
    let Some(slot) = slots.remove(&output_index) else {
        return Ok(());
    };
    match (response_item_type(item), slot) {
        ("message", ResponsesSlot::Text { content_index }) => {
            emitter.replace_text(content_index, &response_item_text(item))?;
            let id = value_string(item, "id").unwrap_or_default();
            emitter.set_text_signature(
                content_index,
                encode_text_signature(id, value_string(item, "phase")),
            )?;
            emitter.end_text(content_index)?;
        }
        (
            "reasoning",
            ResponsesSlot::Thinking {
                content_index,
                item_id,
            },
        ) => {
            let text = response_item_reasoning_text(item);
            if !text.is_empty() {
                emitter.replace_thinking(content_index, &text)?;
            }
            emitter.set_thinking_signature(content_index, item.to_string())?;
            if !item_id.is_empty() {
                reasoning_blocks.insert(item_id, content_index);
            }
            emitter.end_thinking(content_index)?;
        }
        (
            "function_call",
            ResponsesSlot::Function {
                content_index,
                arguments,
            },
        ) => {
            let raw = value_string(item, "arguments")
                .filter(|arguments| !arguments.is_empty())
                .unwrap_or(arguments.raw());
            let mut final_arguments = stream::IncrementalJsonObjectParser::new();
            final_arguments.push(raw);
            emitter.set_tool_metadata(
                content_index,
                Some(&response_tool_call_id(item)),
                value_string(item, "name"),
                value_string(item, "namespace"),
            )?;
            emitter.set_tool_arguments(content_index, final_arguments.finish_tool_arguments())?;
            // The slot parser owns malformed-prefix state until the
            // authoritative item arrives.  The final item is parsed anew.
            let _ = arguments;
            emitter.end_tool(content_index)?;
        }
        (
            "custom_tool_call",
            ResponsesSlot::Custom {
                content_index,
                input,
                input_property,
                mut buffer,
            },
        ) => {
            let input = value_string(item, "input").unwrap_or(&input);
            let delta =
                append_grammar_tool_input_json_delta(&mut buffer, &input_property, input, true)?;
            let mut arguments = BTreeMap::new();
            arguments.insert(input_property, Value::String(input.to_owned()));
            emitter.set_tool_metadata(
                content_index,
                Some(&response_tool_call_id(item)),
                value_string(item, "name"),
                value_string(item, "namespace"),
            )?;
            emitter.set_tool_arguments(content_index, arguments)?;
            if !delta.is_empty() {
                emitter.tool_delta(content_index, &delta)?;
            }
            emitter.end_tool(content_index)?;
        }
        _ => {}
    }
    Ok(())
}

fn close_responses_slots(
    slots: &mut BTreeMap<usize, ResponsesSlot>,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    for (_, slot) in std::mem::take(slots) {
        match slot {
            ResponsesSlot::Text { content_index } => emitter.end_text(content_index)?,
            ResponsesSlot::Thinking { content_index, .. } => emitter.end_thinking(content_index)?,
            ResponsesSlot::Function {
                content_index,
                mut arguments,
            } => {
                emitter.set_tool_arguments(content_index, arguments.finish_tool_arguments())?;
                emitter.end_tool(content_index)?;
            }
            ResponsesSlot::Custom {
                content_index,
                input,
                input_property,
                mut buffer,
            } => {
                let delta = append_grammar_tool_input_json_delta(
                    &mut buffer,
                    &input_property,
                    &input,
                    true,
                )?;
                let mut arguments = BTreeMap::new();
                arguments.insert(input_property, Value::String(input));
                emitter.set_tool_arguments(content_index, arguments)?;
                if !delta.is_empty() {
                    emitter.tool_delta(content_index, &delta)?;
                }
                emitter.end_tool(content_index)?;
            }
        }
    }
    Ok(())
}

fn backfill_responses_reasoning_signatures(
    response: &Value,
    reasoning_blocks: &BTreeMap<String, usize>,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let Some(output) = response.get("output").and_then(Value::as_array) else {
        return Ok(());
    };
    for item in output {
        if response_item_type(item) != "reasoning" {
            continue;
        }
        let Some(encrypted_content) =
            value_string(item, "encrypted_content").filter(|value| !value.is_empty())
        else {
            continue;
        };
        let Some(id) = value_string(item, "id") else {
            continue;
        };
        let Some(index) = reasoning_blocks.get(id).copied() else {
            continue;
        };
        let previous = match emitter.message().content.get(index) {
            Some(llm::ContentBlock::Thinking(thinking)) => thinking.thinking_signature.clone(),
            _ => continue,
        };
        let Ok(Value::Object(mut signature)) = serde_json::from_str::<Value>(&previous) else {
            continue;
        };
        if signature
            .get("encrypted_content")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
        {
            continue;
        }
        signature.insert(
            "encrypted_content".to_owned(),
            Value::String(encrypted_content.to_owned()),
        );
        emitter.set_thinking_signature(index, Value::Object(signature).to_string())?;
    }
    Ok(())
}

fn finalize_responses_response(
    response: &Value,
    reasoning_blocks: &BTreeMap<String, usize>,
    emitter: &mut MessageEmitter,
    codex_requested_service_tier: Option<&str>,
) -> Result<()> {
    if let Some(id) = value_string(response, "id").filter(|id| !id.is_empty()) {
        emitter.message_mut().response_id = id.to_owned();
    }
    if let Some(usage) = response.get("usage") {
        apply_responses_usage(&mut emitter.message_mut().usage, usage);
    }
    if let Some(end_turn) = response.get("end_turn").and_then(Value::as_bool) {
        emitter.message_mut().end_turn = Some(end_turn);
    }
    if let Some(requested_service_tier) = codex_requested_service_tier {
        let service_tier = resolve_codex_service_tier(
            value_string(response, "service_tier").unwrap_or_default(),
            requested_service_tier,
        );
        emitter.set_usage_cost_multiplier(responses_service_tier_cost_multiplier(
            &emitter.model,
            &service_tier,
        ));
    }
    backfill_responses_reasoning_signatures(response, reasoning_blocks, emitter)?;
    let status = value_string(response, "status").unwrap_or_default();
    let incomplete_reason = value_object(response, "incomplete_details")
        .and_then(|details| details.get("reason"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    emitter.message_mut().raw_stop_reason = if incomplete_reason.is_empty() {
        status.to_owned()
    } else {
        format!("{status}.{incomplete_reason}")
    };
    match status {
        "" | "completed" => {
            emitter.message_mut().stop_reason = stream::STOP_STOP.to_owned();
            emitter.message_mut().error_message.clear();
        }
        "incomplete" if incomplete_reason == "max_output_tokens" => {
            emitter.message_mut().stop_reason = stream::STOP_LENGTH.to_owned();
            emitter.message_mut().error_message.clear();
        }
        "incomplete" => {
            emitter.message_mut().stop_reason = stream::STOP_ERROR.to_owned();
            emitter.message_mut().error_message = if incomplete_reason.is_empty() {
                "Response incomplete without a provider reason".to_owned()
            } else {
                format!("Response incomplete: {incomplete_reason}")
            };
        }
        "failed" | "cancelled" => {
            emitter.message_mut().stop_reason = stream::STOP_ERROR.to_owned();
            emitter.message_mut().error_message = value_object(response, "error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Response {status}"));
        }
        "in_progress" | "queued" => {
            emitter.message_mut().stop_reason = stream::STOP_STOP.to_owned();
        }
        other => {
            emitter.message_mut().stop_reason = stream::STOP_ERROR.to_owned();
            emitter.message_mut().error_message = format!("Unhandled response status: {other}");
        }
    }
    if emitter.message().stop_reason == stream::STOP_STOP
        && emitter
            .message
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::ToolCall(_)))
    {
        emitter.message_mut().stop_reason = stream::STOP_TOOL_USE.to_owned();
    }
    Ok(())
}

fn consume_openai_responses(
    body: impl Read,
    _model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<()> {
    consume_responses(
        body,
        cancellation,
        emitter,
        false,
        grammar_tool_input_properties,
        None,
    )
}

fn consume_codex_responses(
    body: impl Read,
    model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<()> {
    let requested_service_tier = requested_responses_service_tier(model);
    consume_responses(
        body,
        cancellation,
        emitter,
        true,
        grammar_tool_input_properties,
        Some(requested_service_tier),
    )
}

fn consume_responses(
    body: impl Read,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
    codex: bool,
    grammar_tool_input_properties: &BTreeMap<String, String>,
    codex_requested_service_tier: Option<&str>,
) -> Result<()> {
    let mut reader = stream::SseReader::new(body);
    let mut slots = BTreeMap::<usize, ResponsesSlot>::new();
    let mut reasoning_blocks = BTreeMap::<String, usize>::new();
    let mut saw_terminal_response = false;

    while let Some(sse) = reader.next_event()? {
        ensure_not_cancelled(cancellation)?;
        let data = sse.data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let mut payload: Value = serde_json::from_str(data)?;
        let mut event_type = value_string(&payload, "type")
            .filter(|value| !value.is_empty())
            .unwrap_or(&sse.event)
            .to_owned();
        if codex
            && matches!(
                event_type.as_str(),
                "response.done" | "response.completed" | "response.incomplete"
            )
        {
            event_type = "response.completed".to_owned();
            if let Some(response) = payload.get_mut("response")
                && let Some(response) = response.as_object_mut()
                && let Some(status) = response.get("status").and_then(Value::as_str)
            {
                response.insert(
                    "status".to_owned(),
                    Value::String(normalize_codex_response_status(status).to_owned()),
                );
            }
        }
        let output_index = value_u64(&payload, "output_index")
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        match event_type.as_str() {
            "response.created" => {
                if let Some(response) = payload.get("response")
                    && let Some(id) = value_string(response, "id")
                {
                    emitter.message_mut().response_id = id.to_owned();
                }
            }
            "response.output_item.added" => {
                if let Some(item) = payload.get("item") {
                    start_responses_slot(
                        output_index,
                        item,
                        &mut slots,
                        emitter,
                        grammar_tool_input_properties,
                    )?;
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = value_string(&payload, "delta").unwrap_or_default();
                if let Some(ResponsesSlot::Thinking { content_index, .. }) =
                    slots.get(&output_index)
                {
                    emitter.append_thinking(*content_index, delta)?;
                }
            }
            "response.reasoning_summary_part.done" => {
                if let Some(ResponsesSlot::Thinking { content_index, .. }) =
                    slots.get(&output_index)
                {
                    emitter.append_thinking(*content_index, "\n\n")?;
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let delta = value_string(&payload, "delta").unwrap_or_default();
                if let Some(ResponsesSlot::Text { content_index }) = slots.get(&output_index) {
                    emitter.append_text(*content_index, delta)?;
                }
            }
            "response.function_call_arguments.delta" => {
                let delta = value_string(&payload, "delta").unwrap_or_default();
                let (content_index, arguments) = match slots.get_mut(&output_index) {
                    Some(ResponsesSlot::Function {
                        content_index,
                        arguments,
                    }) => {
                        arguments.push(delta);
                        (*content_index, arguments.tool_arguments())
                    }
                    _ => continue,
                };
                emitter.set_tool_arguments(content_index, arguments)?;
                if !delta.is_empty() {
                    emitter.tool_delta(content_index, delta)?;
                }
            }
            "response.function_call_arguments.done" => {
                let final_arguments = value_string(&payload, "arguments").unwrap_or_default();
                let (content_index, unseen_delta, arguments) = match slots.get_mut(&output_index) {
                    Some(ResponsesSlot::Function {
                        content_index,
                        arguments,
                    }) => {
                        let previous = arguments.raw().to_owned();
                        let source = if final_arguments.is_empty() {
                            previous.as_str()
                        } else {
                            final_arguments
                        };
                        let mut authoritative = stream::IncrementalJsonObjectParser::new();
                        authoritative.push(source);
                        let preview = authoritative.tool_arguments();
                        *arguments = authoritative;
                        let unseen_delta = final_arguments
                            .strip_prefix(&previous)
                            .unwrap_or_default()
                            .to_owned();
                        (*content_index, unseen_delta, preview)
                    }
                    _ => continue,
                };
                emitter.set_tool_arguments(content_index, arguments)?;
                if !unseen_delta.is_empty() {
                    emitter.tool_delta(content_index, &unseen_delta)?;
                }
            }
            "response.custom_tool_call_input.delta" => {
                let delta = value_string(&payload, "delta").unwrap_or_default();
                let (content_index, arguments, json_delta) = match slots.get_mut(&output_index) {
                    Some(ResponsesSlot::Custom {
                        content_index,
                        input,
                        input_property,
                        buffer,
                    }) => {
                        input.push_str(delta);
                        let json_delta = append_grammar_tool_input_json_delta(
                            buffer,
                            input_property,
                            input,
                            false,
                        )?;
                        let mut arguments = BTreeMap::new();
                        arguments.insert(input_property.clone(), Value::String(input.clone()));
                        (*content_index, arguments, json_delta)
                    }
                    _ => continue,
                };
                emitter.set_tool_arguments(content_index, arguments)?;
                if !json_delta.is_empty() {
                    emitter.tool_delta(content_index, &json_delta)?;
                }
            }
            "response.custom_tool_call_input.done" => {
                let final_input = value_string(&payload, "input").map(str::to_owned);
                let (content_index, arguments, json_delta) = match slots.get_mut(&output_index) {
                    Some(ResponsesSlot::Custom {
                        content_index,
                        input: current,
                        input_property,
                        buffer,
                    }) => {
                        let input = final_input.clone().unwrap_or_else(|| current.clone());
                        let json_delta = append_grammar_tool_input_json_delta(
                            buffer,
                            input_property,
                            &input,
                            true,
                        )?;
                        *current = input.clone();
                        let mut arguments = BTreeMap::new();
                        arguments.insert(input_property.clone(), Value::String(input));
                        (*content_index, arguments, json_delta)
                    }
                    _ => continue,
                };
                emitter.set_tool_arguments(content_index, arguments)?;
                if !json_delta.is_empty() {
                    emitter.tool_delta(content_index, &json_delta)?;
                }
            }
            "response.output_item.done" => {
                if let Some(item) = payload.get("item") {
                    finish_responses_item(
                        output_index,
                        item,
                        &mut slots,
                        &mut reasoning_blocks,
                        emitter,
                        grammar_tool_input_properties,
                    )?;
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                saw_terminal_response = true;
                let Some(response) = payload.get("response") else {
                    return Err(ProviderAdapterError::Protocol(
                        "terminal Responses event omitted response".to_owned(),
                    ));
                };
                finalize_responses_response(
                    response,
                    &reasoning_blocks,
                    emitter,
                    codex_requested_service_tier,
                )?;
            }
            "error" => {
                let nested_error = codex.then(|| value_object(&payload, "error")).flatten();
                let code = value_string(&payload, "code")
                    .or_else(|| {
                        nested_error
                            .and_then(|error| error.get("code"))
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("unknown");
                let message = value_string(&payload, "message")
                    .or_else(|| {
                        nested_error
                            .and_then(|error| error.get("message"))
                            .and_then(Value::as_str)
                    })
                    .unwrap_or("no message");
                return Err(ProviderAdapterError::Protocol(format!("{code}: {message}")));
            }
            _ => {}
        }
        if codex && saw_terminal_response {
            break;
        }
    }
    if !saw_terminal_response {
        return Err(ProviderAdapterError::Protocol(
            if codex {
                "Codex Responses stream ended before a terminal response event"
            } else {
                "OpenAI Responses stream ended before a terminal response event"
            }
            .to_owned(),
        ));
    }
    close_responses_slots(&mut slots, emitter)?;
    if emitter.message().stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message().error_message.clone(),
        ));
    }
    Ok(())
}

fn normalize_codex_response_status(status: &str) -> &str {
    match status {
        "completed" | "incomplete" | "failed" | "cancelled" | "queued" | "in_progress" => status,
        _ => "",
    }
}

fn requested_responses_service_tier(model: &llm::Model) -> &str {
    model
        .sampling_params
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|parameters| parameters.get("service_tier"))
        .and_then(Value::as_str)
        .unwrap_or_default()
}

fn resolve_codex_service_tier(response_tier: &str, request_tier: &str) -> String {
    if response_tier == "default" && matches!(request_tier, "flex" | "priority") {
        return request_tier.to_owned();
    }
    if response_tier.is_empty() {
        request_tier.to_owned()
    } else {
        response_tier.to_owned()
    }
}

fn responses_service_tier_cost_multiplier(model: &llm::Model, service_tier: &str) -> f64 {
    match service_tier {
        "flex" => 0.5,
        "priority" if model.id == "gpt-5.5" => 2.5,
        "priority" => 2.0,
        _ => 1.0,
    }
}

enum AnthropicSlot {
    Text {
        content_index: usize,
    },
    Thinking {
        content_index: usize,
    },
    Tool {
        content_index: usize,
        arguments: stream::IncrementalJsonObjectParser,
        received_delta: bool,
    },
}

fn map_anthropic_stop_reason(raw: &str, refusal_explanation: &str) -> (String, String) {
    match raw {
        "end_turn" | "pause_turn" | "stop_sequence" => {
            (stream::STOP_STOP.to_owned(), String::new())
        }
        "max_tokens" | "model_context_window_exceeded" => {
            (stream::STOP_LENGTH.to_owned(), String::new())
        }
        "tool_use" => (stream::STOP_TOOL_USE.to_owned(), String::new()),
        "refusal" => (
            stream::STOP_ERROR.to_owned(),
            if refusal_explanation.is_empty() {
                "The model refused to complete the request".to_owned()
            } else {
                refusal_explanation.to_owned()
            },
        ),
        "sensitive" => (
            stream::STOP_ERROR.to_owned(),
            "Provider stopped with: sensitive".to_owned(),
        ),
        // Anthropic adds stop reasons over time.  Preserve a complete paid
        // response rather than dropping it merely because a new status is not
        // yet named here.
        _ => (stream::STOP_STOP.to_owned(), String::new()),
    }
}

fn anthropic_error_message(payload: &Value) -> String {
    value_object(payload, "error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| value_string(payload, "message"))
        .unwrap_or("Anthropic returned an error event")
        .to_owned()
}

fn consume_anthropic_messages(
    body: impl Read,
    tools: &[llm::Tool],
    oauth: bool,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let mut reader = stream::SseReader::new(body);
    let mut slots = BTreeMap::<usize, AnthropicSlot>::new();
    let mut saw_message_start = false;
    let mut saw_message_stop = false;

    while let Some(sse) = reader.next_event()? {
        ensure_not_cancelled(cancellation)?;
        let event_name = sse.event.as_str();
        if event_name == "error" {
            let payload = serde_json::from_str::<Value>(&sse.data)?;
            return Err(ProviderAdapterError::Protocol(anthropic_error_message(
                &payload,
            )));
        }
        if !matches!(
            event_name,
            "message_start"
                | "message_delta"
                | "message_stop"
                | "content_block_start"
                | "content_block_delta"
                | "content_block_stop"
        ) {
            continue;
        }
        if sse.data.trim().is_empty() {
            continue;
        }
        let payload: Value = serde_json::from_str(&sse.data)?;
        let api_type = value_string(&payload, "type").unwrap_or(event_name);
        let index = value_u64(&payload, "index")
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(0);
        match api_type {
            "message_start" => {
                saw_message_start = true;
                if let Some(message) = payload.get("message") {
                    if let Some(id) = value_string(message, "id") {
                        emitter.message_mut().response_id = id.to_owned();
                    }
                    if let Some(response_model) = value_string(message, "model")
                        && !response_model.is_empty()
                    {
                        emitter.message_mut().response_model = response_model.to_owned();
                    }
                    if let Some(usage) = message.get("usage") {
                        apply_anthropic_usage(&mut emitter.message_mut().usage, usage);
                    }
                }
            }
            "content_block_start" => {
                let Some(block) = payload.get("content_block") else {
                    continue;
                };
                let slot = match value_string(block, "type").unwrap_or_default() {
                    "text" => AnthropicSlot::Text {
                        content_index: emitter
                            .start_text(value_string(block, "text").unwrap_or_default())?,
                    },
                    "thinking" => AnthropicSlot::Thinking {
                        content_index: emitter.start_thinking(
                            value_string(block, "thinking").unwrap_or_default(),
                            value_string(block, "signature").unwrap_or_default(),
                            false,
                        )?,
                    },
                    "redacted_thinking" => AnthropicSlot::Thinking {
                        content_index: emitter.start_thinking(
                            "[Reasoning redacted]",
                            value_string(block, "data").unwrap_or_default(),
                            true,
                        )?,
                    },
                    "tool_use" => {
                        let name = value_string(block, "name").unwrap_or_default();
                        let name = if oauth {
                            from_claude_code_name(name, tools)
                        } else {
                            name.to_owned()
                        };
                        let content_index = emitter
                            .start_tool(value_string(block, "id").unwrap_or_default(), &name)?;
                        emitter.set_tool_arguments(
                            content_index,
                            btree_arguments(block.get("input").and_then(Value::as_object)),
                        )?;
                        AnthropicSlot::Tool {
                            content_index,
                            arguments: stream::IncrementalJsonObjectParser::new(),
                            received_delta: false,
                        }
                    }
                    _ => continue,
                };
                slots.insert(index, slot);
            }
            "content_block_delta" => {
                let Some(delta) = payload.get("delta") else {
                    continue;
                };
                match value_string(delta, "type").unwrap_or_default() {
                    "text_delta" => {
                        if let Some(AnthropicSlot::Text { content_index }) = slots.get(&index) {
                            emitter.append_text(
                                *content_index,
                                value_string(delta, "text").unwrap_or_default(),
                            )?;
                        }
                    }
                    "thinking_delta" => {
                        if let Some(AnthropicSlot::Thinking { content_index }) = slots.get(&index) {
                            emitter.append_thinking(
                                *content_index,
                                value_string(delta, "thinking").unwrap_or_default(),
                            )?;
                        }
                    }
                    "signature_delta" => {
                        if let Some(AnthropicSlot::Thinking { content_index }) = slots.get(&index) {
                            emitter.append_thinking_signature(
                                *content_index,
                                value_string(delta, "signature").unwrap_or_default(),
                            )?;
                        }
                    }
                    "input_json_delta" => {
                        let partial = value_string(delta, "partial_json").unwrap_or_default();
                        let (content_index, arguments) = match slots.get_mut(&index) {
                            Some(AnthropicSlot::Tool {
                                content_index,
                                arguments,
                                received_delta,
                            }) => {
                                *received_delta = true;
                                arguments.push(partial);
                                (*content_index, arguments.tool_arguments())
                            }
                            _ => continue,
                        };
                        emitter.set_tool_arguments(content_index, arguments)?;
                        if !partial.is_empty() {
                            emitter.tool_delta(content_index, partial)?;
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let Some(slot) = slots.remove(&index) else {
                    continue;
                };
                match slot {
                    AnthropicSlot::Text { content_index } => emitter.end_text(content_index)?,
                    AnthropicSlot::Thinking { content_index } => {
                        emitter.end_thinking(content_index)?
                    }
                    AnthropicSlot::Tool {
                        content_index,
                        mut arguments,
                        received_delta,
                    } => {
                        if received_delta {
                            emitter.set_tool_arguments(
                                content_index,
                                arguments.finish_tool_arguments(),
                            )?;
                        }
                        emitter.end_tool(content_index)?;
                    }
                }
            }
            "message_delta" => {
                if let Some(delta) = payload.get("delta")
                    && let Some(reason) =
                        value_string(delta, "stop_reason").filter(|reason| !reason.is_empty())
                {
                    emitter.message_mut().raw_stop_reason = reason.to_owned();
                    let refusal_explanation = value_object(delta, "stop_details")
                        .and_then(|details| details.get("explanation"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let (stop_reason, error_message) =
                        map_anthropic_stop_reason(reason, refusal_explanation);
                    emitter.message_mut().stop_reason = stop_reason;
                    if !error_message.is_empty() {
                        emitter.message_mut().error_message = error_message;
                    }
                }
                if let Some(usage) = payload.get("usage") {
                    apply_anthropic_usage(&mut emitter.message_mut().usage, usage);
                }
            }
            "message_stop" => saw_message_stop = true,
            _ => {}
        }
    }
    if saw_message_start && !saw_message_stop {
        return Err(ProviderAdapterError::Protocol(
            "Anthropic stream ended before message_stop".to_owned(),
        ));
    }
    if !saw_message_start {
        return Err(ProviderAdapterError::Protocol(
            "Anthropic stream ended before message_start".to_owned(),
        ));
    }
    if emitter.message().stop_reason == stream::STOP_PENDING {
        return Err(ProviderAdapterError::Protocol(
            "Anthropic stream ended without a stop reason".to_owned(),
        ));
    }
    if emitter.message().stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message().error_message.clone(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        sync::{
            Arc, Mutex,
            mpsc::{self, Receiver},
        },
        thread::{self, JoinHandle},
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use serde_json::{Value, json};

    use super::*;

    struct CapturedRequest {
        target: String,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
    }

    fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set test read timeout");
        let mut raw = Vec::new();
        let mut buffer = [0_u8; 4_096];
        let mut header_end = None;
        let mut content_length = 0_usize;
        loop {
            let read = stream.read(&mut buffer).expect("read request");
            assert_ne!(read, 0, "client closed request before sending headers");
            raw.extend_from_slice(&buffer[..read]);
            if header_end.is_none()
                && let Some(index) = raw.windows(4).position(|window| window == b"\r\n\r\n")
            {
                let end = index + 4;
                let header = String::from_utf8_lossy(&raw[..end]);
                content_length = header
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(name, value)| {
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .unwrap_or_default();
                header_end = Some(end);
            }
            if let Some(end) = header_end
                && raw.len() >= end.saturating_add(content_length)
            {
                let header = String::from_utf8_lossy(&raw[..end]);
                let mut lines = header.lines();
                let target = lines
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or_default()
                    .to_owned();
                let headers = lines
                    .filter_map(|line| {
                        line.split_once(':').map(|(name, value)| {
                            (name.to_ascii_lowercase(), value.trim().to_owned())
                        })
                    })
                    .collect();
                return CapturedRequest {
                    target,
                    headers,
                    body: raw[end..end + content_length].to_vec(),
                };
            }
        }
    }

    fn http_response(status: u16, body: &str) -> Vec<u8> {
        http_response_bytes(status, body.as_bytes(), &[])
    }

    fn http_response_with_headers(status: u16, body: &str, headers: &[(&str, &str)]) -> Vec<u8> {
        http_response_bytes(status, body.as_bytes(), headers)
    }

    fn http_response_bytes(status: u16, body: &[u8], headers: &[(&str, &str)]) -> Vec<u8> {
        let reason = match status {
            200 => "OK",
            429 => "Too Many Requests",
            _ => "Test Response",
        };
        let mut response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("\r\n");
        let mut bytes = response.into_bytes();
        bytes.extend_from_slice(body);
        bytes
    }

    fn test_server(responses: Vec<Vec<u8>>) -> (String, Receiver<CapturedRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test HTTP server");
        let address = listener.local_addr().expect("test listener address");
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("accept provider request");
                let request = read_request(&mut stream);
                sender.send(request).expect("send captured request");
                stream
                    .write_all(&response)
                    .expect("write provider response");
                stream.flush().expect("flush provider response");
            }
        });
        (format!("http://{address}"), receiver, handle)
    }

    fn factory(max_retries: u32) -> ProviderResponderFactory {
        factory_with_credentials(max_retries, ProviderCredentials::api_key("test-key"))
    }

    fn factory_with_credentials(
        max_retries: u32,
        credentials: ProviderCredentials,
    ) -> ProviderResponderFactory {
        ProviderResponderFactory::configured(
            credentials,
            ProviderConfig {
                max_retries,
                read_timeout: Some(Duration::from_secs(2)),
                ..ProviderConfig::default()
            },
        )
        .expect("provider factory")
    }

    fn model(api: &str, base_url: String) -> llm::Model {
        llm::Model {
            id: "test-model".to_owned(),
            name: "Test model".to_owned(),
            api: api.to_owned(),
            provider: match api {
                API_ANTHROPIC_MESSAGES => "anthropic".to_owned(),
                API_AZURE_OPENAI_RESPONSES => "azure-openai-responses".to_owned(),
                API_OPENAI_CODEX_RESPONSES => "openai-codex".to_owned(),
                API_GOOGLE_GENERATIVE_AI => "google".to_owned(),
                API_GOOGLE_VERTEX => "google-vertex".to_owned(),
                API_MISTRAL_CONVERSATIONS => "mistral".to_owned(),
                _ => "openai".to_owned(),
            },
            base_url,
            input: vec!["text".to_owned()],
            context_window: 128_000,
            max_tokens: 4_096,
            ..llm::Model::default()
        }
    }

    fn codex_test_jwt(account_id: &str) -> String {
        let payload = serde_json::to_vec(&json!({
            CODEX_JWT_AUTH_CLAIM: {"chatgpt_account_id": account_id},
        }))
        .expect("serialize Codex JWT payload");
        format!("header.{}.signature", URL_SAFE_NO_PAD.encode(payload))
    }

    fn options(cancellation: agent::CancellationToken) -> agent::RequestOptions {
        agent::RequestOptions {
            cancellation,
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

    fn text_context() -> llm::Context {
        llm::Context {
            system_prompt: "be concise".to_owned(),
            messages: vec![llm::Message::User(llm::UserMessage::text("weather?", 1))],
            tools: vec![llm::Tool {
                name: "weather".to_owned(),
                description: "Look up weather".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }),
                constrained_sampling: None,
            }],
        }
    }

    fn bedrock_frame(event_type: &str, payload: &str) -> Vec<u8> {
        bedrock::encode_event_stream_message(
            &BTreeMap::from([
                (":message-type".to_owned(), "event".to_owned()),
                (":event-type".to_owned(), event_type.to_owned()),
                (":content-type".to_owned(), "application/json".to_owned()),
            ]),
            payload.as_bytes(),
        )
        .expect("encode Bedrock frame")
    }

    #[test]
    fn bedrock_converse_stream_uses_the_native_signed_protocol_adapter() {
        let body = [
            bedrock_frame("messageStart", r#"{"role":"assistant"}"#),
            bedrock_frame(
                "contentBlockDelta",
                r#"{"contentBlockIndex":0,"delta":{"text":"Hello from Bedrock"}}"#,
            ),
            bedrock_frame("contentBlockStop", r#"{"contentBlockIndex":0}"#),
            bedrock_frame("messageStop", r#"{"stopReason":"end_turn"}"#),
            bedrock_frame(
                "metadata",
                r#"{"usage":{"inputTokens":12,"outputTokens":3,"totalTokens":15}}"#,
            ),
        ]
        .concat();
        let (base_url, requests, server) = test_server(vec![http_response_bytes(
            200,
            &body,
            &[("Content-Type", "application/vnd.amazon.eventstream")],
        )]);
        let mut request_model = model(bedrock::API_BEDROCK_CONVERSE_STREAM, base_url);
        request_model.provider = "amazon-bedrock".to_owned();
        let response = factory(0)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("Bedrock response");
        let request = requests.recv().expect("captured Bedrock request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/model/test-model/converse-stream");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-key")
        );
        assert_eq!(response.stop_reason, stream::STOP_STOP);
        assert_eq!(
            response
                .content
                .first()
                .and_then(llm::ContentBlock::plain_text),
            Some("Hello from Bedrock")
        );
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn completions_serialization_and_sse_decoding_normalize_tool_usage() {
        let body = concat!(
            "data: {\"id\":\"chat_1\",\"model\":\"test-model\",\"choices\":[{\"delta\":{\"content\":\"Checking \"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chat_1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Pa\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chat_1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ris\\\"}\"}}]},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":4,\"total_tokens\":16,\"prompt_tokens_details\":{\"cached_tokens\":2,\"cache_write_tokens\":1}}}\n\n",
            "data: {\"id\":\"chat_1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut model = model(API_OPENAI_COMPLETIONS, base_url);
        model.reasoning = true;
        let response = factory(0)
            .respond(
                &model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("completion response");
        let request = requests.recv().expect("captured completion request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/chat/completions");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-key")
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("completion JSON body");
        assert_eq!(sent["model"], "test-model");
        assert_eq!(sent["stream"], true);
        assert_eq!(sent["stream_options"]["include_usage"], true);
        assert_eq!(sent["messages"][0]["role"], "developer");
        assert_eq!(sent["messages"][1]["content"], "weather?");
        assert_eq!(sent["tools"][0]["type"], "function");
        assert_eq!(sent["tools"][0]["function"]["name"], "weather");
        assert_eq!(sent["reasoning_effort"], "high");
        assert_eq!(sent["prompt_cache_key"], "session-1");

        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.response_id, "chat_1");
        assert_eq!(response.usage.input, 9);
        assert_eq!(response.usage.output, 4);
        assert_eq!(response.usage.cache_read, 2);
        assert_eq!(response.usage.cache_write, 1);
        assert_eq!(response.usage.total_tokens, 16);
        assert_eq!(response.content[0].plain_text(), Some("Checking "));
        let llm::ContentBlock::ToolCall(call) = &response.content[1] else {
            panic!("expected normalized tool call, got {:?}", response.content);
        };
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "weather");
        assert_eq!(call.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn responses_serialization_replays_items_and_decodes_streamed_function() {
        let body = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
            "event: response.function_call_arguments.delta\ndata: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":20,\"output_tokens\":6,\"total_tokens\":26,\"input_tokens_details\":{\"cached_tokens\":3,\"cache_write_tokens\":2},\"output_tokens_details\":{\"reasoning_tokens\":4}}}}\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut context = text_context();
        context
            .messages
            .push(llm::Message::Assistant(Box::new(llm::AssistantMessage {
                api: API_OPENAI_RESPONSES.to_owned(),
                provider: "openai".to_owned(),
                model: "test-model".to_owned(),
                stop_reason: stream::STOP_STOP.to_owned(),
                content: vec![llm::ContentBlock::Text(llm::TextContent {
                    text: "Earlier answer".to_owned(),
                    text_signature: r#"{"v":1,"id":"msg_previous"}"#.to_owned(),
                })],
                ..llm::AssistantMessage::default()
            })));
        let mut request_model = model(API_OPENAI_RESPONSES, base_url);
        request_model.reasoning = true;
        let response = factory(0)
            .respond(
                &request_model,
                &context,
                options(agent::CancellationToken::default()),
            )
            .expect("Responses response");
        let request = requests.recv().expect("captured Responses request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/responses");
        let sent: Value = serde_json::from_slice(&request.body).expect("Responses JSON body");
        assert_eq!(sent["stream"], true);
        assert_eq!(sent["store"], false);
        assert_eq!(sent["input"][0]["role"], "developer");
        assert_eq!(sent["input"][1]["content"][0]["type"], "input_text");
        assert_eq!(sent["input"][2]["type"], "message");
        assert_eq!(sent["input"][2]["id"], "msg_previous");
        assert_eq!(sent["tools"][0]["name"], "weather");
        assert_eq!(sent["max_output_tokens"], 4096);
        assert_eq!(sent["reasoning"]["summary"], "auto");

        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.response_id, "resp_1");
        assert_eq!(response.usage.input, 15);
        assert_eq!(response.usage.output, 6);
        assert_eq!(response.usage.cache_read, 3);
        assert_eq!(response.usage.cache_write, 2);
        assert_eq!(response.usage.reasoning, Some(4));
        let llm::ContentBlock::ToolCall(call) = &response.content[0] else {
            panic!("expected normalized Responses tool call");
        };
        assert_eq!(call.id, "call_1|fc_1");
        assert_eq!(call.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn anthropic_serialization_and_sse_decoding_normalize_tool_call() {
        let body = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":11,\"cache_read_input_tokens\":2}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"weather\",\"input\":{}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"Paris\\\"}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3,\"output_tokens_details\":{\"thinking_tokens\":1}}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut request_model = model(API_ANTHROPIC_MESSAGES, base_url);
        request_model.reasoning = true;
        let response = factory(0)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("Anthropic response");
        let request = requests.recv().expect("captured Anthropic request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/v1/messages");
        assert_eq!(
            request.headers.get("x-api-key").map(String::as_str),
            Some("test-key")
        );
        assert_eq!(
            request.headers.get("anthropic-version").map(String::as_str),
            Some("2023-06-01")
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("Anthropic JSON body");
        assert_eq!(sent["system"][0]["text"], "be concise");
        assert_eq!(sent["messages"][0]["role"], "user");
        assert_eq!(sent["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(sent["tools"][0]["eager_input_streaming"], true);
        assert_eq!(sent["thinking"]["type"], "enabled");

        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.response_id, "msg_1");
        assert_eq!(response.response_model, "claude-test");
        assert_eq!(response.usage.input, 11);
        assert_eq!(response.usage.output, 3);
        assert_eq!(response.usage.cache_read, 2);
        assert_eq!(response.usage.total_tokens, 16);
        assert_eq!(response.usage.reasoning, Some(1));
        let llm::ContentBlock::ToolCall(call) = &response.content[0] else {
            panic!("expected normalized Anthropic tool call");
        };
        assert_eq!(call.id, "toolu_1");
        assert_eq!(call.name, "weather");
        assert_eq!(call.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn google_generate_content_serialization_and_sse_decoding_preserve_thoughts_and_tools() {
        let body = "data: {\"responseId\":\"google_1\",\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"Considering \",\"thoughtSignature\":\"c2ln\"},{\"text\":\"sunny\"},{\"functionCall\":{\"name\":\"weather\",\"id\":\"call_1\",\"args\":{\"city\":\"Paris\"}},\"thoughtSignature\":\"c2ln\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"cachedContentTokenCount\":2,\"candidatesTokenCount\":3,\"thoughtsTokenCount\":4,\"totalTokenCount\":17}}\n\n";
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut request_model = model(API_GOOGLE_GENERATIVE_AI, base_url);
        request_model.id = "gemini-3-pro".to_owned();
        request_model.reasoning = true;
        let observed_events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&observed_events);
        let mut request_options = options(agent::CancellationToken::default());
        request_options.assistant_event_listener = Some(Arc::new(move |event| {
            event_log
                .lock()
                .expect("Google event log lock")
                .push(event.event_type);
        }));
        let response = factory(0)
            .respond(&request_model, &text_context(), request_options)
            .expect("Google response");
        let request = requests.recv().expect("captured Google request");
        server.join().expect("test server finishes");

        assert_eq!(
            request.target,
            "/models/gemini-3-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            request.headers.get("x-goog-api-key").map(String::as_str),
            Some("test-key")
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("Google JSON body");
        assert_eq!(sent["systemInstruction"]["parts"][0]["text"], "be concise");
        assert_eq!(sent["contents"][0]["role"], "user");
        assert_eq!(sent["contents"][0]["parts"][0]["text"], "weather?");
        assert_eq!(
            sent["tools"][0]["functionDeclarations"][0]["name"],
            "weather"
        );
        assert_eq!(sent["generationConfig"]["maxOutputTokens"], 4096);
        assert_eq!(
            sent["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "HIGH"
        );

        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.response_id, "google_1");
        assert_eq!(response.usage.input, 8);
        assert_eq!(response.usage.output, 7);
        assert_eq!(response.usage.cache_read, 2);
        assert_eq!(response.usage.total_tokens, 17);
        assert_eq!(response.usage.reasoning, Some(4));
        let llm::ContentBlock::Thinking(thinking) = &response.content[0] else {
            panic!("expected Google thought content");
        };
        assert_eq!(thinking.thinking, "Considering ");
        assert_eq!(thinking.thinking_signature, "c2ln");
        assert_eq!(response.content[1].plain_text(), Some("sunny"));
        let llm::ContentBlock::ToolCall(call) = &response.content[2] else {
            panic!("expected Google function call");
        };
        assert_eq!(call.id, "call_1");
        assert_eq!(call.name, "weather");
        assert_eq!(call.arguments.get("city"), Some(&json!("Paris")));
        assert_eq!(call.thought_signature, "c2ln");
        assert_eq!(
            *observed_events.lock().expect("Google event log lock"),
            vec![
                stream::EVENT_START.to_owned(),
                stream::EVENT_THINKING_START.to_owned(),
                stream::EVENT_THINKING_DELTA.to_owned(),
                stream::EVENT_THINKING_END.to_owned(),
                stream::EVENT_TEXT_START.to_owned(),
                stream::EVENT_TEXT_DELTA.to_owned(),
                stream::EVENT_TEXT_END.to_owned(),
                stream::EVENT_TOOLCALL_START.to_owned(),
                stream::EVENT_TOOLCALL_DELTA.to_owned(),
                stream::EVENT_TOOLCALL_END.to_owned(),
                stream::EVENT_DONE.to_owned(),
            ]
        );
    }

    #[test]
    fn google_generate_content_replays_signatures_tool_results_and_strict_tools() {
        let mut context = text_context();
        context
            .messages
            .push(llm::Message::Assistant(Box::new(llm::AssistantMessage {
                api: API_GOOGLE_GENERATIVE_AI.to_owned(),
                provider: "google".to_owned(),
                model: "gemini-3-pro".to_owned(),
                stop_reason: stream::STOP_TOOL_USE.to_owned(),
                content: vec![
                    llm::ContentBlock::Thinking(llm::ThinkingContent {
                        thinking: "reasoning".to_owned(),
                        thinking_signature: "c2ln".to_owned(),
                        ..llm::ThinkingContent::default()
                    }),
                    llm::ContentBlock::ToolCall(llm::ToolCall {
                        id: "call_1".to_owned(),
                        name: "weather".to_owned(),
                        arguments: BTreeMap::from([("city".to_owned(), json!("Paris"))]),
                        thought_signature: "c2ln".to_owned(),
                        ..llm::ToolCall::default()
                    }),
                ],
                ..llm::AssistantMessage::default()
            })));
        context
            .messages
            .push(llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: "call_1".to_owned(),
                tool_name: "weather".to_owned(),
                content: vec![llm::ContentBlock::text("18 C")],
                timestamp: 2,
                ..llm::ToolResultMessage::default()
            })));
        context.tools[0].constrained_sampling =
            Some(json!({"type": "json_schema", "strict": "require"}));

        let mut request_model = model(
            API_GOOGLE_GENERATIVE_AI,
            "https://generativelanguage.googleapis.com/v1beta".to_owned(),
        );
        request_model.id = "gemini-3-pro".to_owned();
        request_model.reasoning = true;
        let body = build_google_generate_content_request(
            &request_model,
            &context,
            &options(agent::CancellationToken::default()),
        );

        assert_eq!(body["contents"][1]["role"], "model");
        assert_eq!(body["contents"][1]["parts"][0]["thought"], true);
        assert_eq!(body["contents"][1]["parts"][0]["thoughtSignature"], "c2ln");
        assert_eq!(
            body["contents"][1]["parts"][1]["functionCall"]["id"],
            "call_1"
        );
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"]["id"],
            "call_1"
        );
        assert_eq!(
            body["contents"][2]["parts"][0]["functionResponse"]["response"]["output"],
            "18 C"
        );
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["mode"],
            "VALIDATED"
        );

        request_model.id = "gemini-2.5-pro".to_owned();
        let non_strict = build_google_generate_content_request(
            &request_model,
            &context,
            &options(agent::CancellationToken::default()),
        );
        assert!(non_strict.get("toolConfig").is_none());
    }

    #[test]
    fn google_history_transform_downgrades_images_and_remaps_cross_model_tool_ids() {
        let source_call = llm::ToolCall {
            id: "call with spaces!".to_owned(),
            name: "weather".to_owned(),
            arguments: BTreeMap::from([("city".to_owned(), json!("Paris"))]),
            thought_signature: "c2ln".to_owned(),
            ..llm::ToolCall::default()
        };
        let messages = vec![
            llm::Message::User(llm::UserMessage {
                role: "user".to_owned(),
                content: llm::UserContent::Blocks(vec![
                    llm::ContentBlock::text("Look"),
                    llm::ContentBlock::Image(llm::ImageContent {
                        data: "image-data".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ]),
                timestamp: 1,
            }),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                api: API_OPENAI_COMPLETIONS.to_owned(),
                provider: "openai".to_owned(),
                model: "other-model".to_owned(),
                stop_reason: stream::STOP_TOOL_USE.to_owned(),
                content: vec![
                    llm::ContentBlock::Thinking(llm::ThinkingContent {
                        thinking: "cross-model thought".to_owned(),
                        thinking_signature: "c2ln".to_owned(),
                        ..llm::ThinkingContent::default()
                    }),
                    llm::ContentBlock::ToolCall(source_call),
                ],
                ..llm::AssistantMessage::default()
            })),
            llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: "call with spaces!".to_owned(),
                tool_name: "weather".to_owned(),
                content: vec![llm::ContentBlock::text("sunny")],
                timestamp: 2,
                ..llm::ToolResultMessage::default()
            })),
        ];
        let request_model = llm::Model {
            id: "gemini-3-pro".to_owned(),
            api: API_GOOGLE_GENERATIVE_AI.to_owned(),
            provider: "google".to_owned(),
            input: vec!["text".to_owned()],
            ..llm::Model::default()
        };

        let transformed = transform_google_messages(&messages, &request_model);
        let llm::Message::User(user) = &transformed[0] else {
            panic!("expected transformed user message");
        };
        let llm::UserContent::Blocks(parts) = &user.content else {
            panic!("expected block user content");
        };
        assert_eq!(
            parts[1].plain_text(),
            Some("(image omitted: model does not support images)")
        );
        let llm::Message::Assistant(assistant) = &transformed[1] else {
            panic!("expected transformed assistant message");
        };
        assert_eq!(
            assistant.content[0].plain_text(),
            Some("cross-model thought")
        );
        let llm::ContentBlock::ToolCall(call) = &assistant.content[1] else {
            panic!("expected transformed tool call");
        };
        assert_eq!(call.id, "call_with_spaces_");
        assert!(call.thought_signature.is_empty());
        let llm::Message::ToolResult(result) = &transformed[2] else {
            panic!("expected transformed tool result");
        };
        assert_eq!(result.tool_call_id, "call_with_spaces_");
    }

    #[test]
    fn google_endpoint_usage_and_stop_helpers_match_google_wire_behavior() {
        let endpoint = google_generate_content_endpoint(&llm::Model {
            id: "gemini-2.5-pro".to_owned(),
            ..llm::Model::default()
        })
        .expect("default Google endpoint");
        assert_eq!(
            endpoint.as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
        let mut usage = llm::Usage::default();
        apply_google_usage(
            &mut usage,
            &json!({
                "promptTokenCount": 3,
                "cachedContentTokenCount": 9,
                "candidatesTokenCount": 2,
                "thoughtsTokenCount": 1,
                "totalTokenCount": 12,
            }),
        );
        assert_eq!(usage.input, 0);
        assert_eq!(usage.output, 3);
        assert_eq!(usage.cache_read, 9);
        assert_eq!(usage.total_tokens, 12);
        assert_eq!(map_google_stop_reason("STOP").0, stream::STOP_STOP);
        assert_eq!(map_google_stop_reason("MAX_TOKENS").0, stream::STOP_LENGTH);
        assert_eq!(map_google_stop_reason("SAFETY").0, stream::STOP_ERROR);

        let mut request_model = llm::Model {
            id: "gemini-2.5-pro".to_owned(),
            reasoning: true,
            max_tokens: 512,
            ..llm::Model::default()
        };
        let mut request_options = options(agent::CancellationToken::default());
        request_options.thinking_budgets = Some(llm::ThinkingBudgets {
            high: Some(999),
            ..llm::ThinkingBudgets::default()
        });
        let body = build_google_generate_content_request(
            &request_model,
            &llm::Context::default(),
            &request_options,
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            999
        );
        request_model.id = "gemini-3-pro".to_owned();
        let body = build_google_generate_content_request(
            &request_model,
            &llm::Context::default(),
            &request_options,
        );
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "HIGH"
        );
    }

    #[test]
    fn vertex_express_mode_uses_an_api_key_and_shared_google_streaming() {
        let body = concat!(
            "data: {\"responseId\":\"vertex_1\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"think\",\"thought\":true,\"thoughtSignature\":\"c2ln\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"answer\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"thoughtsTokenCount\":3,\"totalTokenCount\":18}}\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut request_model = model(API_GOOGLE_VERTEX, format!("{base_url}/v1"));
        request_model.id = "gemini-3-pro".to_owned();
        request_model.reasoning = true;
        let response = factory(0)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("Vertex response");
        let request = requests.recv().expect("captured Vertex request");
        server.join().expect("Vertex test server finishes");

        assert_eq!(
            request.target,
            "/v1/publishers/google/models/gemini-3-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            request.headers.get("x-goog-api-key").map(String::as_str),
            Some("test-key")
        );
        assert!(!request.headers.contains_key("authorization"));
        let sent: Value = serde_json::from_slice(&request.body).expect("Vertex JSON body");
        assert_eq!(
            sent["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "HIGH"
        );
        assert_eq!(response.stop_reason, stream::STOP_STOP);
        assert_eq!(response.response_id, "vertex_1");
        assert_eq!(response.usage.output, 8);
        let llm::ContentBlock::Thinking(thinking) = &response.content[0] else {
            panic!("expected Vertex thought content");
        };
        assert_eq!(thinking.thinking, "think");
        assert_eq!(thinking.thinking_signature, "c2ln");
        assert_eq!(response.content[1].plain_text(), Some("answer"));
    }

    #[test]
    fn vertex_adc_mode_uses_a_bearer_token_and_regional_resource_path() {
        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut credentials = ProviderCredentials::api_key(catalog::AUTHENTICATED_SENTINEL);
        credentials.environment = BTreeMap::from([
            (
                "GOOGLE_OAUTH_ACCESS_TOKEN".to_owned(),
                "ya29.token".to_owned(),
            ),
            ("GOOGLE_CLOUD_PROJECT".to_owned(), "my-project".to_owned()),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "us-central1".to_owned()),
        ]);
        let request_model = model(API_GOOGLE_VERTEX, format!("{base_url}/v1"));
        let response = factory_with_credentials(0, credentials)
            .respond(
                &request_model,
                &llm::Context {
                    messages: vec![llm::Message::User(llm::UserMessage::text("hi", 1))],
                    ..llm::Context::default()
                },
                options(agent::CancellationToken::default()),
            )
            .expect("Vertex ADC response");
        let request = requests.recv().expect("captured Vertex request");
        server.join().expect("Vertex ADC server finishes");

        assert_eq!(
            request.target,
            "/v1/projects/my-project/locations/us-central1/publishers/google/models/test-model:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer ya29.token")
        );
        assert!(!request.headers.contains_key("x-goog-api-key"));
        assert_eq!(response.stop_reason, stream::STOP_STOP);
    }

    #[test]
    fn catalog_vertex_prefetched_token_reaches_the_resource_endpoint() {
        let body = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n";
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let environment = BTreeMap::from([
            (
                "GOOGLE_OAUTH_ACCESS_TOKEN".to_owned(),
                "ya29.prefetched".to_owned(),
            ),
            (
                "GOOGLE_CLOUD_PROJECT".to_owned(),
                "catalog-project".to_owned(),
            ),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "us-central1".to_owned()),
        ]);
        let catalog = Arc::new(
            catalog::Catalog::with_environment(
                None,
                Arc::new(move |name| environment.get(name).cloned()),
            )
            .expect("catalog"),
        );
        let mut request_model = catalog
            .provider("google-vertex")
            .expect("Vertex provider")
            .models()
            .into_iter()
            .next()
            .expect("Vertex model");
        request_model.base_url = format!("{base_url}/v1");
        let response = factory(0).catalog_assistant_responder(catalog)(
            &request_model,
            &text_context(),
            options(agent::CancellationToken::default()),
        )
        .expect("catalog-backed Vertex response");
        let request = requests.recv().expect("captured Vertex request");
        server.join().expect("Vertex test server finishes");

        assert_eq!(
            request.target,
            format!(
                "/v1/projects/catalog-project/locations/us-central1/publishers/google/models/{}:streamGenerateContent?alt=sse",
                request_model.id
            )
        );
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer ya29.prefetched")
        );
        assert_eq!(response.stop_reason, stream::STOP_STOP);
    }

    #[test]
    fn vertex_token_exchange_cancellation_returns_an_aborted_message_promptly() {
        google_auth::clear_token_cache();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind token server");
        let address = listener.local_addr().expect("token server address");
        let (started_sender, started_receiver) = mpsc::channel();
        let (release_sender, release_receiver) = mpsc::channel();
        let token_server = thread::spawn(move || {
            let (mut connection, _) = listener.accept().expect("accept token request");
            let _request = read_request(&mut connection);
            started_sender.send(()).expect("signal token request");
            release_receiver
                .recv_timeout(Duration::from_secs(2))
                .expect("release token response");
            let body = br#"{"access_token":"ya29.refreshed","expires_in":3600}"#;
            connection
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .expect("write token response headers");
            connection
                .write_all(body)
                .expect("write token response body");
        });
        let credential_path = std::env::temp_dir().join(format!(
            "goshcoder-vertex-cancellation-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("current time")
                .as_nanos()
        ));
        fs::write(
            &credential_path,
            json!({
                "type": "authorized_user",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token",
                "token_uri": format!("http://{address}"),
            })
            .to_string(),
        )
        .expect("write credential fixture");

        let mut credentials = ProviderCredentials::api_key(catalog::AUTHENTICATED_SENTINEL);
        credentials.environment = BTreeMap::from([
            (
                "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
                credential_path.display().to_string(),
            ),
            ("GOOGLE_CLOUD_PROJECT".to_owned(), "project".to_owned()),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "us-central1".to_owned()),
        ]);
        let cancellation = agent::CancellationToken::default();
        let request_cancellation = cancellation.clone();
        let (result_sender, result_receiver) = mpsc::channel();
        thread::spawn(move || {
            let response = factory_with_credentials(0, credentials).respond(
                &model(API_GOOGLE_VERTEX, "http://127.0.0.1:1/v1".to_owned()),
                &llm::Context::default(),
                options(request_cancellation),
            );
            let _ = result_sender.send(response);
        });

        started_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("token exchange starts");
        cancellation.cancel();
        let response = result_receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("cancellation interrupts token exchange")
            .expect("normalized cancelled response");
        release_sender.send(()).expect("release token server");
        token_server.join().expect("token server finishes");
        let _ = fs::remove_file(&credential_path);
        google_auth::clear_token_cache();

        assert_eq!(response.stop_reason, stream::STOP_ABORTED);
        assert!(response.error_message.contains("request aborted"));
    }

    #[test]
    fn vertex_endpoint_and_thinking_helpers_follow_vertex_specific_rules() {
        let environment = BTreeMap::from([
            ("GOOGLE_CLOUD_PROJECT".to_owned(), "project".to_owned()),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "global".to_owned()),
        ]);
        let credentials = ProviderCredentials {
            api_key: Some(catalog::AUTHENTICATED_SENTINEL.to_owned()),
            environment,
            ..ProviderCredentials::default()
        };
        let endpoint = google_vertex_endpoint(
            &llm::Model {
                id: "gemini-2.5-flash".to_owned(),
                base_url: "https://{location}-aiplatform.googleapis.com".to_owned(),
                ..llm::Model::default()
            },
            &credentials,
        )
        .expect("regional Vertex endpoint");
        assert_eq!(
            endpoint.as_str(),
            "https://aiplatform.googleapis.com/v1/projects/project/locations/global/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
        assert!(vertex_base_url_includes_api_version(
            &Url::parse("https://example.test/v1beta1").expect("versioned URL")
        ));
        assert!(!vertex_base_url_includes_api_version(
            &Url::parse("https://example.test/api").expect("non-versioned URL")
        ));

        let model = llm::Model {
            id: "gemma-4-27b".to_owned(),
            reasoning: true,
            ..llm::Model::default()
        };
        let config =
            google_thinking_config(&model, llm::THINKING_LOW, None, GoogleApiVariant::Vertex)
                .expect("Vertex thinking config");
        assert_eq!(config["thinkingBudget"], -1);
    }

    #[test]
    fn retry_is_bounded_and_pre_request_cancellation_becomes_aborted_message() {
        let success = concat!(
            "data: {\"id\":\"chat_retry\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![
            http_response_with_headers(
                429,
                r#"{"error":{"message":"slow down"}}"#,
                &[("retry-after-ms", "0")],
            ),
            http_response(200, success),
        ]);
        let response = factory(1)
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("retried response");
        let first = requests.recv().expect("first request");
        let second = requests.recv().expect("second request");
        server.join().expect("test server finishes");
        assert_eq!(first.target, "/chat/completions");
        assert_eq!(second.target, "/chat/completions");
        assert_eq!(response.stop_reason, stream::STOP_STOP);

        let cancellation = agent::CancellationToken::default();
        cancellation.cancel();
        let cancelled = factory(0)
            .respond(
                &model(API_OPENAI_COMPLETIONS, "http://127.0.0.1:1".to_owned()),
                &text_context(),
                options(cancellation),
            )
            .expect("cancellation is a normalized terminal assistant message");
        assert_eq!(cancelled.stop_reason, stream::STOP_ABORTED);
        assert!(cancelled.error_message.contains("request aborted"));
    }

    #[test]
    fn protocol_variants_include_google_vertex_azure_and_codex_responses() {
        assert_eq!(
            ProviderProtocol::from_api(API_GOOGLE_GENERATIVE_AI).expect("Google protocol"),
            ProviderProtocol::GoogleGenerativeAi
        );
        assert_eq!(
            ProviderProtocol::from_api(API_GOOGLE_VERTEX).expect("Vertex protocol"),
            ProviderProtocol::GoogleVertex
        );
        assert_eq!(
            ProviderProtocol::from_api(API_AZURE_OPENAI_RESPONSES).expect("Azure protocol"),
            ProviderProtocol::AzureOpenAiResponses
        );
        assert_eq!(
            ProviderProtocol::from_api(API_OPENAI_CODEX_RESPONSES).expect("Codex protocol"),
            ProviderProtocol::OpenAiCodexResponses
        );
    }

    #[test]
    fn azure_endpoint_normalization_and_scoped_configuration_match_the_protocol() {
        let cases = [
            (
                "https://resource.cognitiveservices.azure.com",
                "https://resource.cognitiveservices.azure.com/openai/v1",
            ),
            (
                "https://resource.ai.azure.com/openai",
                "https://resource.ai.azure.com/openai/v1",
            ),
            (
                "https://resource.openai.azure.com/openai/v1/responses",
                "https://resource.openai.azure.com/openai/v1",
            ),
            (
                "https://resource.openai.azure.com/openai?api-version=old",
                "https://resource.openai.azure.com/openai/v1",
            ),
            (
                "https://proxy.example.test/v1?custom=true",
                "https://proxy.example.test/v1?custom=true",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(
                normalize_azure_openai_base_url(input)
                    .expect("normalize Azure base URL")
                    .as_str(),
                expected
            );
        }
        let invalid = normalize_azure_openai_base_url("not-a-url").expect_err("invalid URL");
        assert!(!invalid.to_string().contains("not-a-url"));

        let mut credentials = ProviderCredentials::api_key("azure-key");
        credentials.environment = BTreeMap::from([
            (
                "AZURE_OPENAI_BASE_URL".to_owned(),
                "https://override.openai.azure.com".to_owned(),
            ),
            (
                "AZURE_OPENAI_RESOURCE_NAME".to_owned(),
                "ignored-resource".to_owned(),
            ),
            (
                "AZURE_OPENAI_API_VERSION".to_owned(),
                "2025-04-01".to_owned(),
            ),
        ]);
        let model = model(
            API_AZURE_OPENAI_RESPONSES,
            "https://model.openai.azure.com".to_owned(),
        );
        let (base_url, api_version) =
            resolve_azure_openai_config(&model, &credentials).expect("Azure config");
        assert_eq!(
            base_url.as_str(),
            "https://override.openai.azure.com/openai/v1"
        );
        assert_eq!(api_version, "2025-04-01");

        credentials.environment.remove("AZURE_OPENAI_BASE_URL");
        let (base_url, _) =
            resolve_azure_openai_config(&model, &credentials).expect("resource Azure config");
        assert_eq!(
            base_url.as_str(),
            "https://ignored-resource.openai.azure.com/openai/v1"
        );
        assert_eq!(
            parse_azure_deployment_name_map(" other=nope, test-model=deployment-1, malformed "),
            BTreeMap::from([
                ("other".to_owned(), "nope".to_owned()),
                ("test-model".to_owned(), "deployment-1".to_owned()),
            ])
        );
    }

    #[test]
    fn azure_responses_use_deployment_api_key_and_shared_responses_stream() {
        let body = concat!(
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_azure\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Azure says hi\"}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_azure\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Azure says hi\",\"annotations\":[]}]}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_azure\",\"status\":\"completed\",\"usage\":{\"input_tokens\":5,\"output_tokens\":3,\"total_tokens\":8}}}\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut credentials = ProviderCredentials::api_key("azure-test-key");
        credentials.environment = BTreeMap::from([
            (
                "AZURE_OPENAI_BASE_URL".to_owned(),
                format!("{base_url}/gateway?custom=true"),
            ),
            (
                "AZURE_OPENAI_API_VERSION".to_owned(),
                "2025-04-01".to_owned(),
            ),
            (
                "AZURE_OPENAI_DEPLOYMENT_NAME_MAP".to_owned(),
                "test-model=azure-deployment".to_owned(),
            ),
        ]);
        let mut request_model = model(API_AZURE_OPENAI_RESPONSES, String::new());
        request_model.reasoning = true;
        let observed_events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&observed_events);
        let mut request_options = options(agent::CancellationToken::default());
        request_options.assistant_event_listener = Some(Arc::new(move |event| {
            event_log
                .lock()
                .expect("event log lock")
                .push(event.event_type);
        }));
        let response = factory_with_credentials(0, credentials)
            .respond(&request_model, &text_context(), request_options)
            .expect("Azure Responses response");
        let request = requests.recv().expect("captured Azure request");
        server.join().expect("Azure test server finishes");

        assert_eq!(
            request.target,
            "/gateway/responses?custom=true&api-version=2025-04-01"
        );
        assert_eq!(
            request.headers.get("api-key").map(String::as_str),
            Some("azure-test-key")
        );
        assert!(!request.headers.contains_key("authorization"));
        let sent: Value = serde_json::from_slice(&request.body).expect("Azure JSON body");
        assert_eq!(sent["model"], "azure-deployment");
        assert_eq!(sent["stream"], true);
        assert_eq!(sent["store"], false);
        assert_eq!(sent["input"][0]["role"], "developer");
        assert_eq!(sent["tools"][0]["strict"], false);
        assert_eq!(sent["reasoning"]["effort"], "high");
        assert_eq!(sent["reasoning"]["summary"], "auto");
        assert_eq!(sent["include"][0], "reasoning.encrypted_content");
        assert_eq!(response.stop_reason, stream::STOP_STOP);
        assert_eq!(response.response_id, "resp_azure");
        assert_eq!(response.content[0].plain_text(), Some("Azure says hi"));
        assert_eq!(response.usage.total_tokens, 8);
        let observed_events = observed_events.lock().expect("event log lock");
        assert!(observed_events.contains(&stream::EVENT_START.to_owned()));
        assert!(observed_events.contains(&stream::EVENT_TEXT_DELTA.to_owned()));
        assert!(observed_events.contains(&stream::EVENT_DONE.to_owned()));
    }

    #[test]
    fn omni_prompt_tools_render_the_protocol_and_reemit_parsed_tool_calls() {
        let body = concat!(
            r#"data: {"id":"omni_1","choices":[{"delta":{"content":"Checking. "}}]}"#,
            "\n\n",
            r#"data: {"id":"omni_1","choices":[{"delta":{"content":"<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"}}]}"#,
            "\n\n",
            r#"data: {"id":"omni_1","choices":[{"finish_reason":"stop","delta":{}}],"usage":{"prompt_tokens":20,"completion_tokens":9,"total_tokens":29}}"#,
            "\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut request_model = model(API_OPENAI_COMPLETIONS, base_url);
        request_model.api = omniroute::PROMPT_TOOLS_API.to_owned();
        request_model.provider = "omni".to_owned();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let mut request_options = options(agent::CancellationToken::default());
        let log = Arc::clone(&observed);
        request_options.assistant_event_listener = Some(Arc::new(move |event| {
            log.lock()
                .expect("event log")
                .push(event.event_type.clone());
        }));

        let response = factory(0)
            .respond(&request_model, &text_context(), request_options)
            .expect("prompt-tools response");
        let request = requests.recv().expect("captured chat request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/chat/completions");
        let sent: Value = serde_json::from_slice(&request.body).expect("JSON body");
        assert!(sent.get("tools").is_none(), "native tools are never sent");
        let system = sent["messages"][0]["content"]
            .as_str()
            .expect("system prompt");
        assert!(system.starts_with("be concise\n\n# Tool calling protocol"));
        assert!(system.contains("### weather\nLook up weather"));
        assert_eq!(sent["messages"][1]["content"], "weather?");

        assert_eq!(response.api, omniroute::PROMPT_TOOLS_API);
        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.content[0].plain_text(), Some("Checking."));
        let llm::ContentBlock::ToolCall(call) = &response.content[1] else {
            panic!("second block is the parsed tool call");
        };
        assert_eq!(call.name, "weather");
        assert_eq!(call.arguments["city"], json!("Paris"));
        assert!(call.id.starts_with("call_omni_"));
        assert_eq!(response.usage.total_tokens, 29);
        let observed = observed.lock().expect("event log");
        assert_eq!(
            observed.first().map(String::as_str),
            Some(stream::EVENT_START)
        );
        assert!(observed.contains(&stream::EVENT_TEXT_END.to_owned()));
        assert!(observed.contains(&stream::EVENT_TOOLCALL_END.to_owned()));
        assert_eq!(
            observed.last().map(String::as_str),
            Some(stream::EVENT_DONE)
        );
    }

    #[test]
    fn omni_prompt_tools_keep_a_truncated_reply_from_running_its_tool_calls() {
        let body = concat!(
            r#"data: {"id":"omni_2","choices":[{"delta":{"content":"<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>"}}]}"#,
            "\n\n",
            r#"data: {"id":"omni_2","choices":[{"finish_reason":"length","delta":{}}]}"#,
            "\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, _requests, server) = test_server(vec![http_response(200, body)]);
        let mut request_model = model(API_OPENAI_COMPLETIONS, base_url);
        request_model.api = omniroute::PROMPT_TOOLS_API.to_owned();
        let response = factory(0)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("prompt-tools response");
        server.join().expect("test server finishes");
        assert_eq!(response.stop_reason, stream::STOP_LENGTH);
        assert!(matches!(
            response.content[0],
            llm::ContentBlock::ToolCall(_)
        ));
    }

    #[test]
    fn mistral_conversations_serialization_and_sse_decoding_preserve_protocol_rules() {
        let body = concat!(
            r#"data: {"id":"mistral_1","choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"text":"consider "}]}]}}]}"#,
            "\n\n",
            r#"data: {"id":"mistral_1","choices":[{"delta":{"content":[{"type":"text","text":"answer"}]}}]}"#,
            "\n\n",
            r#"data: {"id":"mistral_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"abcdefghi","function":{"name":"weather","arguments":"{\"city\":"}}]}}]}"#,
            "\n\n",
            r#"data: {"id":"mistral_1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}}]}"#,
            "\n\n",
            r#"data: {"id":"mistral_1","choices":[{"finish_reason":"tool_calls","delta":{}}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":2}}}"#,
            "\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let mut request_model = model(API_MISTRAL_CONVERSATIONS, base_url);
        request_model.id = "mistral-small-latest".to_owned();
        request_model.reasoning = true;
        let mut request_options = options(agent::CancellationToken::default());
        request_options.temperature = Some(0.3);
        request_options.max_tokens = Some(512);
        request_options.tool_choice = Some(json!("required"));

        let response = factory(0)
            .respond(&request_model, &text_context(), request_options)
            .expect("Mistral response");
        let request = requests.recv().expect("captured Mistral request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/v1/chat/completions");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-key")
        );
        assert_eq!(
            request.headers.get("x-affinity").map(String::as_str),
            Some("session-1")
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("Mistral JSON body");
        assert_eq!(sent["model"], "mistral-small-latest");
        assert_eq!(sent["stream"], true);
        assert_eq!(sent["max_tokens"], 512);
        assert_eq!(sent["temperature"], 0.3);
        assert_eq!(sent["tool_choice"], "required");
        assert_eq!(sent["reasoning_effort"], "high");
        assert_eq!(sent["prompt_cache_key"], "session-1");
        assert_eq!(sent["tools"][0]["function"]["strict"], false);

        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.response_id, "mistral_1");
        assert_eq!(response.usage.input, 10);
        assert_eq!(response.usage.output, 4);
        assert_eq!(response.usage.cache_read, 2);
        assert_eq!(response.usage.total_tokens, 16);
        let llm::ContentBlock::Thinking(thinking) = &response.content[0] else {
            panic!("expected Mistral thinking content");
        };
        assert_eq!(thinking.thinking, "consider ");
        assert_eq!(response.content[1].plain_text(), Some("answer"));
        let llm::ContentBlock::ToolCall(call) = &response.content[2] else {
            panic!("expected Mistral tool call");
        };
        assert_eq!(call.id, "abcdefghi");
        assert_eq!(call.name, "weather");
        assert_eq!(call.arguments.get("city"), Some(&json!("Paris")));
    }

    #[test]
    fn mistral_header_overrides_and_cache_retention_match_native_behavior() {
        let request_model = model(
            API_MISTRAL_CONVERSATIONS,
            "https://api.mistral.ai".to_owned(),
        );
        let cancellation = agent::CancellationToken::default();
        let suppressed_authorization = build_request_headers(
            ProviderProtocol::MistralConversations,
            &request_model,
            &ProviderCredentials::api_key("mistral-key").without_header("authorization"),
            "session-1",
            agent::CacheRetention::Short,
            &cancellation,
            None,
        )
        .expect("Mistral accepts an intentional bearer-header suppression");
        assert!(!suppressed_authorization.contains_key("authorization"));
        assert_eq!(
            suppressed_authorization
                .get("x-affinity")
                .and_then(|value| value.to_str().ok()),
            Some("session-1")
        );

        let no_cache = build_request_headers(
            ProviderProtocol::MistralConversations,
            &request_model,
            &ProviderCredentials::api_key("mistral-key"),
            "session-1",
            agent::CacheRetention::None,
            &cancellation,
            None,
        )
        .expect("Mistral headers without prompt caching");
        assert!(!no_cache.contains_key("x-affinity"));
    }

    #[test]
    fn mistral_response_header_deadline_does_not_truncate_an_active_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind Mistral server");
        let address = listener.local_addr().expect("Mistral server address");
        let (request_sender, request_receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept Mistral request");
            request_sender
                .send(read_request(&mut stream))
                .expect("capture Mistral request");
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
                )
                .expect("write Mistral response headers");
            stream.flush().expect("flush Mistral response headers");
            for chunk in [
                r#"data: {"id":"mistral_1","choices":[{"delta":{"content":"one "}}]}"#,
                r#"data: {"id":"mistral_1","choices":[{"delta":{"content":"two"}}]}"#,
                r#"data: {"id":"mistral_1","choices":[{"finish_reason":"stop","delta":{}}]}"#,
            ] {
                thread::sleep(Duration::from_millis(60));
                stream
                    .write_all(chunk.as_bytes())
                    .expect("write Mistral chunk");
                stream.write_all(b"\n\n").expect("terminate Mistral chunk");
                stream.flush().expect("flush Mistral chunk");
            }
        });
        let factory = ProviderResponderFactory::configured(
            ProviderCredentials::api_key("mistral-key"),
            ProviderConfig {
                max_retries: 0,
                read_timeout: Some(Duration::from_secs(2)),
                mistral_response_header_timeout: Some(Duration::from_millis(50)),
                ..ProviderConfig::default()
            },
        )
        .expect("Mistral provider factory");
        let request_model = model(API_MISTRAL_CONVERSATIONS, format!("http://{address}"));

        let response = factory
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("Mistral stream survives past standard request deadline");
        let request = request_receiver.recv().expect("captured Mistral request");
        server.join().expect("Mistral server finishes");

        assert_eq!(request.target, "/v1/chat/completions");
        assert_eq!(response.stop_reason, stream::STOP_STOP);
        assert_eq!(response.content[0].plain_text(), Some("one two"));
    }

    #[test]
    fn azure_requires_an_api_key_and_never_uses_a_bearer_header_as_a_fallback() {
        let header_secret = "Bearer header-only-secret";
        let response = factory_with_credentials(
            0,
            ProviderCredentials::default().with_header("authorization", header_secret),
        )
        .respond(
            &model(API_AZURE_OPENAI_RESPONSES, "http://127.0.0.1:1".to_owned()),
            &llm::Context::default(),
            options(agent::CancellationToken::default()),
        )
        .expect("normalized Azure auth failure");
        assert_eq!(response.stop_reason, stream::STOP_ERROR);
        assert!(response.error_message.contains("no API key"));
        assert!(!response.error_message.contains(header_secret));
    }

    #[test]
    fn azure_configured_api_key_headers_override_or_suppress_the_default() {
        let request_model = model(
            API_AZURE_OPENAI_RESPONSES,
            "https://example.openai.azure.com/openai/v1".to_owned(),
        );
        let cancellation = agent::CancellationToken::default();
        let overridden = build_request_headers(
            ProviderProtocol::AzureOpenAiResponses,
            &request_model,
            &ProviderCredentials::api_key("azure-default").with_header("api-key", "proxy-key"),
            "",
            agent::CacheRetention::Short,
            &cancellation,
            None,
        )
        .expect("override Azure headers");
        assert_eq!(
            overridden
                .get("api-key")
                .and_then(|value| value.to_str().ok()),
            Some("proxy-key")
        );

        let suppressed = build_request_headers(
            ProviderProtocol::AzureOpenAiResponses,
            &request_model,
            &ProviderCredentials::api_key("azure-default").without_header("api-key"),
            "",
            agent::CacheRetention::Short,
            &cancellation,
            None,
        )
        .expect("suppress Azure header");
        assert!(!suppressed.contains_key("api-key"));
    }

    #[test]
    fn codex_responses_send_required_headers_and_normalize_response_done() {
        let body = concat!(
            "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_codex\",\"status\":\"in_progress\"}}\n\n",
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_codex\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
            "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Codex says hi\"}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_codex\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Codex says hi\",\"annotations\":[]}]}}\n\n",
            "event: response.done\ndata: {\"type\":\"response.done\",\"response\":{\"id\":\"resp_codex\",\"status\":\"completed\",\"service_tier\":\"default\",\"end_turn\":true,\"usage\":{\"input_tokens\":12,\"output_tokens\":3,\"total_tokens\":15}}}\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let token = codex_test_jwt("account-1");
        let mut request_model = model(API_OPENAI_CODEX_RESPONSES, base_url);
        request_model.id = "gpt-5.4".to_owned();
        request_model.reasoning = true;
        request_model.sampling_params = Some(json!({"service_tier": "flex"}));
        request_model.cost.rates = llm::ModelCostRates {
            input: 1_000_000.0,
            output: 1_000_000.0,
            ..llm::ModelCostRates::default()
        };
        let response = factory_with_credentials(0, ProviderCredentials::api_key(token.clone()))
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("Codex Responses response");
        let request = requests.recv().expect("captured Codex request");
        server.join().expect("Codex test server finishes");

        assert_eq!(request.target, "/codex/responses");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some(format!("Bearer {token}").as_str())
        );
        assert_eq!(
            request
                .headers
                .get("chatgpt-account-id")
                .map(String::as_str),
            Some("account-1")
        );
        assert_eq!(
            request.headers.get("originator").map(String::as_str),
            Some("goshcoder")
        );
        assert_eq!(
            request.headers.get("user-agent").map(String::as_str),
            Some(
                format!(
                    "goshcoder ({}; {})",
                    std::env::consts::OS,
                    std::env::consts::ARCH
                )
                .as_str()
            )
        );
        assert_eq!(
            request.headers.get("openai-beta").map(String::as_str),
            Some("responses=experimental")
        );
        assert_eq!(
            request.headers.get("session-id").map(String::as_str),
            Some("session-1")
        );
        assert_eq!(
            request
                .headers
                .get("x-client-request-id")
                .map(String::as_str),
            Some("session-1")
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("Codex JSON body");
        assert_eq!(sent["model"], "gpt-5.4");
        assert_eq!(sent["instructions"], "be concise");
        assert_eq!(sent["input"].as_array().map(Vec::len), Some(1));
        assert_eq!(sent["input"][0]["role"], "user");
        assert_eq!(sent["text"]["verbosity"], "low");
        assert_eq!(sent["include"][0], "reasoning.encrypted_content");
        assert_eq!(sent["tool_choice"], "auto");
        assert_eq!(sent["parallel_tool_calls"], true);
        assert_eq!(sent["prompt_cache_key"], "session-1");
        assert_eq!(sent["service_tier"], "flex");
        assert_eq!(sent["tools"][0]["strict"], Value::Null);
        assert_eq!(sent["reasoning"]["effort"], "high");
        assert_eq!(response.stop_reason, stream::STOP_STOP);
        assert_eq!(response.response_id, "resp_codex");
        assert_eq!(response.end_turn, Some(true));
        assert_eq!(response.content[0].plain_text(), Some("Codex says hi"));
        assert_eq!(response.usage.total_tokens, 15);
        assert_eq!(response.usage.cost.total, 7.5);
    }

    #[test]
    fn codex_endpoint_token_errors_retries_and_nested_sse_errors_are_handled() {
        assert_eq!(
            codex_responses_endpoint("")
                .expect("default Codex URL")
                .as_str(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            codex_responses_endpoint("https://example.test/backend-api/codex/")
                .expect("normalized Codex URL")
                .as_str(),
            "https://example.test/backend-api/codex/responses"
        );
        let token_error = extract_codex_account_id("not-a-jwt").expect_err("invalid JWT");
        assert_eq!(
            token_error.to_string(),
            "failed to extract accountId from token"
        );

        let success = "event: response.done\ndata: {\"type\":\"response.done\",\"response\":{\"id\":\"resp_retry\",\"status\":\"completed\"}}\n\n";
        let (base_url, requests, server) = test_server(vec![
            http_response_with_headers(
                429,
                r#"{"error":{"message":"slow down"}}"#,
                &[("retry-after-ms", "0")],
            ),
            http_response(200, success),
        ]);
        let token = codex_test_jwt("retry-account");
        let response = factory_with_credentials(1, ProviderCredentials::api_key(token.clone()))
            .respond(
                &model(API_OPENAI_CODEX_RESPONSES, base_url),
                &llm::Context::default(),
                options(agent::CancellationToken::default()),
            )
            .expect("retried Codex response");
        let first = requests.recv().expect("first Codex request");
        let second = requests.recv().expect("second Codex request");
        server.join().expect("Codex retry server finishes");
        assert_eq!(first.target, "/codex/responses");
        assert_eq!(second.target, "/codex/responses");
        assert_eq!(response.stop_reason, stream::STOP_STOP);

        let (base_url, requests, server) = test_server(vec![http_response(
            200,
            "event: error\ndata: {\"type\":\"error\",\"error\":{\"code\":\"usage_limit_reached\",\"message\":\"limit reached\"}}\n\n",
        )]);
        let error_response = factory_with_credentials(0, ProviderCredentials::api_key(token))
            .respond(
                &model(API_OPENAI_CODEX_RESPONSES, base_url),
                &llm::Context::default(),
                options(agent::CancellationToken::default()),
            )
            .expect("normalized Codex SSE error");
        let _request = requests.recv().expect("nested-error Codex request");
        server.join().expect("Codex nested-error server finishes");
        assert_eq!(error_response.stop_reason, stream::STOP_ERROR);
        assert!(error_response.error_message.contains("limit reached"));

        let cancellation = agent::CancellationToken::default();
        cancellation.cancel();
        let cancelled = factory_with_credentials(
            0,
            ProviderCredentials::api_key(codex_test_jwt("cancel-account")),
        )
        .respond(
            &model(API_OPENAI_CODEX_RESPONSES, String::new()),
            &llm::Context::default(),
            options(cancellation),
        )
        .expect("normalized Codex cancellation");
        assert_eq!(cancelled.stop_reason, stream::STOP_ABORTED);
        assert!(cancelled.error_message.contains("request aborted"));
    }

    #[test]
    fn azure_grammar_custom_tools_use_schema_property_in_requests_and_streams() {
        let body = concat!(
            "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_azure\",\"call_id\":\"call_azure\",\"name\":\"pattern\",\"input\":\"\"}}\n\n",
            "event: response.custom_tool_call_input.delta\ndata: {\"type\":\"response.custom_tool_call_input.delta\",\"output_index\":0,\"delta\":\"hello\"}\n\n",
            "event: response.custom_tool_call_input.done\ndata: {\"type\":\"response.custom_tool_call_input.done\",\"output_index\":0,\"input\":\"hello\"}\n\n",
            "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"custom_tool_call\",\"id\":\"ctc_azure\",\"call_id\":\"call_azure\",\"name\":\"pattern\",\"input\":\"hello\"}}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_grammar\",\"status\":\"completed\"}}\n\n",
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let grammar_tool = llm::Tool {
            name: "pattern".to_owned(),
            description: "Match a grammar".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            }),
            constrained_sampling: Some(json!({
                "type": "grammar",
                "variants": {"openai_regex": "[a-z]+"},
            })),
        };
        let context = llm::Context {
            system_prompt: "use the grammar".to_owned(),
            messages: vec![
                llm::Message::User(llm::UserMessage::text("first", 1)),
                llm::Message::Assistant(Box::new(llm::AssistantMessage {
                    api: API_AZURE_OPENAI_RESPONSES.to_owned(),
                    provider: "azure-openai-responses".to_owned(),
                    model: "test-model".to_owned(),
                    stop_reason: stream::STOP_TOOL_USE.to_owned(),
                    content: vec![llm::ContentBlock::ToolCall(llm::ToolCall {
                        id: "call_previous|ctc_previous".to_owned(),
                        name: grammar_tool.name.clone(),
                        arguments: BTreeMap::from([("query".to_owned(), json!("prior"))]),
                        namespace: "tools".to_owned(),
                        ..llm::ToolCall::default()
                    })],
                    ..llm::AssistantMessage::default()
                })),
                llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                    tool_call_id: "call_previous|ctc_previous".to_owned(),
                    tool_name: grammar_tool.name.clone(),
                    content: vec![llm::ContentBlock::text("matched")],
                    timestamp: 2,
                    ..llm::ToolResultMessage::default()
                })),
            ],
            tools: vec![grammar_tool],
        };
        let mut request_model = model(API_AZURE_OPENAI_RESPONSES, base_url);
        request_model.reasoning = true;
        request_model.compat = Some(json!({
            "supportsDeveloperRole": false,
            "supportsOpenAIGrammarTools": true,
        }));
        let events = factory_with_credentials(0, ProviderCredentials::api_key("azure-key"))
            .stream(
                &request_model,
                &context,
                options(agent::CancellationToken::default()),
            )
            .iter()
            .collect::<Vec<_>>();
        let request = requests.recv().expect("captured Azure grammar request");
        server.join().expect("Azure grammar server finishes");

        let sent: Value = serde_json::from_slice(&request.body).expect("Azure grammar JSON");
        assert_eq!(sent["input"][0]["role"], "system");
        assert_eq!(sent["input"][2]["type"], "custom_tool_call");
        assert_eq!(sent["input"][2]["id"], "ctc_previous");
        assert_eq!(sent["input"][2]["input"], "prior");
        assert_eq!(sent["input"][2]["namespace"], "tools");
        assert_eq!(sent["input"][3]["type"], "custom_tool_call_output");
        assert_eq!(sent["tools"][0]["type"], "custom");
        assert_eq!(sent["tools"][0]["format"]["type"], "grammar");
        assert_eq!(sent["tools"][0]["format"]["syntax"], "regex");
        assert_eq!(sent["tools"][0]["format"]["definition"], "[a-z]+");
        assert!(sent["tools"][0].get("strict").is_none());

        let deltas = events
            .iter()
            .filter(|event| event.event_type == stream::EVENT_TOOLCALL_DELTA)
            .map(|event| event.delta.as_str())
            .collect::<Vec<_>>();
        assert_eq!(deltas, vec!["{\"query\":\"hello", "\"}"]);
        let response = events
            .last()
            .and_then(stream::AssistantMessageEvent::terminal_message)
            .expect("terminal grammar response");
        let llm::ContentBlock::ToolCall(call) = &response.content[0] else {
            panic!("expected grammar tool call, got {:?}", response.content);
        };
        assert_eq!(call.id, "call_azure|ctc_azure");
        assert_eq!(call.arguments.get("query"), Some(&json!("hello")));
    }

    #[test]
    fn codex_deferred_tools_follow_additional_and_tool_search_compatibility() {
        let body = "event: response.done\ndata: {\"type\":\"response.done\",\"response\":{\"id\":\"resp_deferred\",\"status\":\"completed\"}}\n\n";
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let immediate = llm::Tool {
            name: "immediate".to_owned(),
            description: "Available now".to_owned(),
            parameters: json!({"type": "object", "properties": {}}),
            constrained_sampling: None,
        };
        let deferred = llm::Tool {
            name: "later".to_owned(),
            description: "Available after the first result".to_owned(),
            parameters: json!({"type": "object", "properties": {}}),
            constrained_sampling: None,
        };
        let context = llm::Context {
            messages: vec![
                llm::Message::User(llm::UserMessage::text("start", 1)),
                llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                    tool_call_id: "call_immediate|fc_immediate".to_owned(),
                    tool_name: immediate.name.clone(),
                    content: vec![llm::ContentBlock::text("result")],
                    added_tool_names: vec![deferred.name.clone()],
                    timestamp: 2,
                    ..llm::ToolResultMessage::default()
                })),
            ],
            tools: vec![immediate.clone(), deferred.clone()],
            ..llm::Context::default()
        };
        let mut request_model = model(API_OPENAI_CODEX_RESPONSES, base_url);
        request_model.compat = Some(json!({"supportsAdditionalTools": true}));
        let response = factory_with_credentials(
            0,
            ProviderCredentials::api_key(codex_test_jwt("deferred-account")),
        )
        .respond(
            &request_model,
            &context,
            options(agent::CancellationToken::default()),
        )
        .expect("Codex deferred response");
        let request = requests.recv().expect("captured Codex deferred request");
        server.join().expect("Codex deferred server finishes");

        let sent: Value = serde_json::from_slice(&request.body).expect("Codex deferred JSON");
        assert_eq!(sent["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(sent["tools"][0]["name"], "immediate");
        assert_eq!(sent["input"][1]["type"], "function_call_output");
        assert_eq!(sent["input"][2]["type"], "additional_tools");
        assert_eq!(sent["input"][2]["tools"][0]["name"], "later");
        assert_eq!(response.stop_reason, stream::STOP_STOP);

        let mut search_model = model(API_OPENAI_CODEX_RESPONSES, String::new());
        search_model.compat = Some(json!({"supportsToolSearch": true}));
        let grammar_properties =
            grammar_tool_input_properties(&context.tools, false).expect("grammar properties");
        let params = build_openai_codex_responses_request(
            &search_model,
            &context,
            &options(agent::CancellationToken::default()),
            &grammar_properties,
        )
        .expect("Codex tool-search params");
        assert_eq!(params["tools"].as_array().map(Vec::len), Some(1));
        assert_eq!(params["tools"][0]["name"], "immediate");
        assert_eq!(params["input"][2]["type"], "tool_search_call");
        assert_eq!(params["input"][3]["type"], "tool_search_output");
        assert_eq!(params["input"][3]["tools"][0]["name"], "later");
        assert_eq!(params["input"][3]["tools"][0]["defer_loading"], true);
    }

    /// Streams a chunked response with pauses so idle and total deadlines can
    /// be told apart. The client may drop a stalled stream, so writes are
    /// allowed to fail.
    fn paced_sse_server(
        chunks: Vec<(Duration, Vec<u8>)>,
    ) -> (String, Receiver<CapturedRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind paced server");
        let address = listener.local_addr().expect("paced server address");
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept paced request");
            sender
                .send(read_request(&mut stream))
                .expect("capture paced request");
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            );
            for (pause, chunk) in chunks {
                thread::sleep(pause);
                if stream
                    .write_all(&chunk)
                    .and_then(|()| stream.flush())
                    .is_err()
                {
                    break;
                }
            }
        });
        (format!("http://{address}"), receiver, handle)
    }

    fn completions_text_chunk(text: &str) -> Vec<u8> {
        format!(
            "data: {}\n\n",
            json!({
                "id": "chat_paced",
                "choices": [{"delta": {"content": text}, "finish_reason": null}],
            })
        )
        .into_bytes()
    }

    fn completions_finish_chunks() -> Vec<u8> {
        b"data: {\"id\":\"chat_paced\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
            .to_vec()
    }

    fn factory_with_read_timeout(read_timeout: Duration) -> ProviderResponderFactory {
        ProviderResponderFactory::configured(
            ProviderCredentials::api_key("test-key"),
            ProviderConfig {
                max_retries: 0,
                read_timeout: Some(read_timeout),
                ..ProviderConfig::default()
            },
        )
        .expect("provider factory")
    }

    fn assistant_turn(
        source: &llm::Model,
        stop_reason: &str,
        content: Vec<llm::ContentBlock>,
    ) -> llm::Message {
        llm::Message::Assistant(Box::new(llm::AssistantMessage {
            api: source.api.clone(),
            provider: source.provider.clone(),
            model: source.id.clone(),
            stop_reason: stop_reason.to_owned(),
            content,
            timestamp: 2,
            ..llm::AssistantMessage::default()
        }))
    }

    fn weather_call(id: &str) -> llm::ContentBlock {
        llm::ContentBlock::ToolCall(llm::ToolCall {
            id: id.to_owned(),
            name: "weather".to_owned(),
            arguments: BTreeMap::from([("city".to_owned(), json!("Paris"))]),
            ..llm::ToolCall::default()
        })
    }

    fn weather_result(id: &str) -> llm::Message {
        llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
            tool_call_id: id.to_owned(),
            tool_name: "weather".to_owned(),
            content: vec![llm::ContentBlock::text("sunny")],
            timestamp: 3,
            ..llm::ToolResultMessage::default()
        }))
    }

    fn thinking(text: &str, signature: &str) -> llm::ContentBlock {
        llm::ContentBlock::Thinking(llm::ThinkingContent {
            thinking: text.to_owned(),
            thinking_signature: signature.to_owned(),
            redacted: false,
        })
    }

    #[test]
    fn streaming_has_an_idle_read_deadline_but_no_whole_request_deadline() {
        let pause = Duration::from_millis(120);
        let (base_url, requests, server) = paced_sse_server(vec![
            (pause, completions_text_chunk("one ")),
            (pause, completions_text_chunk("two ")),
            (pause, completions_text_chunk("three")),
            (pause, completions_finish_chunks()),
        ]);
        // The pauses add up to more than the idle deadline; no single gap does.
        let response = factory_with_read_timeout(Duration::from_millis(300))
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("a flowing stream outlives the idle deadline");
        requests.recv().expect("captured paced request");
        server.join().expect("paced server finishes");
        assert_eq!(response.stop_reason, stream::STOP_STOP);
        assert_eq!(response.content[0].plain_text(), Some("one two three"));

        let (base_url, requests, server) = paced_sse_server(vec![
            (Duration::ZERO, completions_text_chunk("partial ")),
            (Duration::from_millis(1_500), completions_finish_chunks()),
        ]);
        let started = Instant::now();
        let stalled = factory_with_read_timeout(Duration::from_millis(200))
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("an idle stream becomes a normalized error message");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "the idle deadline fires before the server resumes"
        );
        requests.recv().expect("captured stalled request");
        assert_eq!(stalled.stop_reason, stream::STOP_ERROR);
        assert!(
            stalled.error_message.contains("timed out"),
            "{}",
            stalled.error_message
        );
        assert!(stream::is_retryable_assistant_error(&stalled));
        server.join().expect("stalled server finishes");
    }

    #[test]
    fn cancellation_yields_the_exact_partial_while_the_socket_read_is_blocked() {
        let (base_url, requests, server) = paced_sse_server(vec![
            (Duration::ZERO, completions_text_chunk("partial ")),
            (Duration::from_millis(1_200), completions_finish_chunks()),
        ]);
        let cancellation = agent::CancellationToken::default();
        let canceller = {
            let cancellation = cancellation.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(200));
                cancellation.cancel();
            })
        };
        let started = Instant::now();
        let aborted = factory_with_read_timeout(Duration::from_secs(5))
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(cancellation),
            )
            .expect("cancellation is a normalized aborted message");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "cancellation must not wait for the socket read"
        );
        assert_eq!(aborted.stop_reason, stream::STOP_ABORTED);
        assert_eq!(aborted.error_message, "request aborted");
        assert_eq!(aborted.content[0].plain_text(), Some("partial "));
        canceller.join().expect("canceller finishes");
        requests.recv().expect("captured request");
        server.join().expect("paced server finishes");
    }

    #[test]
    fn history_transform_skips_incomplete_turns_and_synthesizes_missing_tool_results() {
        let request_options = options(agent::CancellationToken::default());
        for api in [
            API_OPENAI_COMPLETIONS,
            API_OPENAI_RESPONSES,
            API_ANTHROPIC_MESSAGES,
        ] {
            let request_model = model(api, "https://example.test".to_owned());
            let mut context = text_context();
            context.messages = vec![
                llm::Message::User(llm::UserMessage::text("start", 1)),
                assistant_turn(
                    &request_model,
                    stream::STOP_TOOL_USE,
                    vec![weather_call("call_orphan")],
                ),
                llm::Message::User(llm::UserMessage::text("next", 3)),
                assistant_turn(
                    &request_model,
                    stream::STOP_ABORTED,
                    vec![
                        llm::ContentBlock::text("half"),
                        weather_call("call_aborted"),
                    ],
                ),
                assistant_turn(
                    &request_model,
                    stream::STOP_ERROR,
                    vec![llm::ContentBlock::text("failed")],
                ),
                llm::Message::User(llm::UserMessage::text("final", 5)),
            ];
            let payload = match api {
                API_OPENAI_COMPLETIONS => {
                    build_openai_completions_request(&request_model, &context, &request_options)
                }
                API_OPENAI_RESPONSES => {
                    build_openai_responses_request(&request_model, &context, &request_options)
                }
                _ => build_anthropic_messages_request(
                    &request_model,
                    &context,
                    &request_options,
                    &anthropic_request_shape(
                        &request_model,
                        &ProviderCredentials::api_key("sk-ant-api"),
                        &request_options,
                    ),
                ),
            }
            .expect("request body");
            let serialized = payload.to_string();
            assert!(
                !serialized.contains("call_aborted"),
                "{api}: aborted turn replayed: {serialized}"
            );
            assert!(
                !serialized.contains("\"half\"") && !serialized.contains("\"failed\""),
                "{api}: incomplete turns replayed: {serialized}"
            );
            let synthetic = serialized
                .find("No result provided")
                .unwrap_or_else(|| panic!("{api}: orphan tool call lacks a synthetic result"));
            let next = serialized.find("\"next\"").expect("interrupting user turn");
            assert!(
                synthetic < next,
                "{api}: the synthetic result must precede the interrupting user turn"
            );
        }
    }

    #[test]
    fn cross_model_replay_normalizes_ids_and_thinking_while_same_model_replays_verbatim() {
        let request_options = options(agent::CancellationToken::default());
        let long_id = format!("call_{}", "x".repeat(60));
        let foreign = llm::Model {
            id: "gpt-foreign".to_owned(),
            api: API_OPENAI_CODEX_RESPONSES.to_owned(),
            provider: "openai-codex".to_owned(),
            ..llm::Model::default()
        };
        let history = |source: &llm::Model, id: &str| {
            vec![
                llm::Message::User(llm::UserMessage::text("go", 1)),
                assistant_turn(
                    source,
                    stream::STOP_TOOL_USE,
                    vec![
                        thinking(
                            "deliberating",
                            r#"{"type":"reasoning","id":"rs_1","summary":[]}"#,
                        ),
                        llm::ContentBlock::Text(llm::TextContent {
                            text: "calling".to_owned(),
                            text_signature: r#"{"v":1,"id":"msg_signed"}"#.to_owned(),
                        }),
                        weather_call(id),
                    ],
                ),
                weather_result(id),
            ]
        };

        // Chat Completions: pipe ids collapse within 40 characters and
        // cross-model thinking becomes text.
        let completions_model = model(API_OPENAI_COMPLETIONS, "https://example.test".to_owned());
        let mut context = text_context();
        context.messages = history(&foreign, &format!("call_1|fc_{}", "y".repeat(50)));
        let sent = build_openai_completions_request(&completions_model, &context, &request_options)
            .expect("completions body");
        let assistant = &sent["messages"][2];
        let call_id = assistant["tool_calls"][0]["id"].as_str().expect("tool id");
        assert!(
            call_id.len() <= 40 && call_id.starts_with("call_1_"),
            "{call_id}"
        );
        assert_eq!(sent["messages"][3]["tool_call_id"], call_id);
        assert_eq!(assistant["content"], "deliberatingcalling");
        assert!(assistant.get("reasoning_content").is_none());

        context.messages = history(&completions_model, &long_id);
        let sent = build_openai_completions_request(&completions_model, &context, &request_options)
            .expect("same-model completions body");
        assert_eq!(sent["messages"][2]["tool_calls"][0]["id"], long_id);

        // Responses: a different model of the same provider loses its fc_ item
        // id and reasoning item; the same model replays both.
        let responses_model = model(API_OPENAI_RESPONSES, "https://example.test".to_owned());
        let mut sibling = responses_model.clone();
        sibling.id = "other-model".to_owned();
        context.messages = history(&sibling, "call_1|fc_1");
        let sent = build_openai_responses_request(&responses_model, &context, &request_options)
            .expect("responses body");
        let items = sent["input"].as_array().expect("input items");
        assert!(items.iter().all(|item| item["type"] != "reasoning"));
        let call = items
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("function call item");
        assert!(call["id"].is_null(), "{call}");
        assert_eq!(call["call_id"], "call_1");
        let message = items
            .iter()
            .find(|item| item["type"] == "message")
            .expect("message item");
        assert_eq!(message["content"][0]["text"], "deliberating");
        assert!(
            message["id"]
                .as_str()
                .is_some_and(|id| id.starts_with("msg_pi_"))
        );

        context.messages = history(&responses_model, "call_1|fc_1");
        let sent = build_openai_responses_request(&responses_model, &context, &request_options)
            .expect("same-model responses body");
        let items = sent["input"].as_array().expect("input items");
        assert!(
            items
                .iter()
                .any(|item| item["type"] == "reasoning" && item["id"] == "rs_1")
        );
        let call = items
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("function call item");
        assert_eq!(call["id"], "fc_1");
        assert!(
            items
                .iter()
                .any(|item| item["type"] == "message" && item["id"] == "msg_signed")
        );

        // Anthropic: foreign ids are sanitized to its alphabet, foreign
        // thinking is text, and same-model signatures replay.
        let anthropic_model = model(API_ANTHROPIC_MESSAGES, "https://example.test".to_owned());
        let shape = anthropic_request_shape(
            &anthropic_model,
            &ProviderCredentials::api_key("sk-ant-api"),
            &request_options,
        );
        context.messages = history(&foreign, "call with spaces|fc_1");
        let sent =
            build_anthropic_messages_request(&anthropic_model, &context, &request_options, &shape)
                .expect("anthropic body");
        let blocks = sent["messages"][1]["content"]
            .as_array()
            .expect("assistant blocks");
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "deliberating");
        assert_eq!(blocks[2]["id"], "call_with_spaces_fc_1");
        assert_eq!(
            sent["messages"][2]["content"][0]["tool_use_id"],
            "call_with_spaces_fc_1"
        );

        context.messages = history(&anthropic_model, "toolu_1");
        let sent =
            build_anthropic_messages_request(&anthropic_model, &context, &request_options, &shape)
                .expect("same-model anthropic body");
        let blocks = sent["messages"][1]["content"]
            .as_array()
            .expect("assistant blocks");
        assert_eq!(blocks[0]["type"], "thinking");
        assert_eq!(
            blocks[0]["signature"],
            r#"{"type":"reasoning","id":"rs_1","summary":[]}"#
        );
        assert_eq!(blocks[2]["id"], "toolu_1");
    }

    #[test]
    fn anthropic_requests_carry_cache_control_beta_features_and_adaptive_thinking() {
        let mut request_model = model(API_ANTHROPIC_MESSAGES, "https://example.test".to_owned());
        request_model.reasoning = true;
        let credentials = ProviderCredentials::api_key("sk-ant-api-key");
        let context = text_context();
        let request_options = options(agent::CancellationToken::default());
        let ephemeral = json!({"type": "ephemeral"});

        let shape = anthropic_request_shape(&request_model, &credentials, &request_options);
        assert!(!shape.oauth);
        assert_eq!(
            anthropic_beta_features(&request_model, &context, &shape),
            vec![ANTHROPIC_INTERLEAVED_THINKING_BETA]
        );
        let sent =
            build_anthropic_messages_request(&request_model, &context, &request_options, &shape)
                .expect("body");
        assert_eq!(sent["system"][0]["cache_control"], ephemeral);
        assert_eq!(sent["tools"][0]["cache_control"], ephemeral);
        assert_eq!(sent["messages"][0]["content"][0]["text"], "weather?");
        assert_eq!(
            sent["messages"][0]["content"][0]["cache_control"],
            ephemeral
        );
        assert_eq!(sent["thinking"]["type"], "enabled");
        assert_eq!(sent["thinking"]["budget_tokens"], 3_072);

        let mut long_options = request_options.clone();
        long_options.cache_retention = agent::CacheRetention::Long;
        let shape = anthropic_request_shape(&request_model, &credentials, &long_options);
        assert_eq!(
            shape.cache_control,
            Some(json!({"type": "ephemeral", "ttl": "1h"}))
        );
        let mut limited = request_model.clone();
        limited.compat = Some(json!({
            "supportsLongCacheRetention": false,
            "supportsEagerToolInputStreaming": false,
            "supportsCacheControlOnTools": false,
        }));
        let shape = anthropic_request_shape(&limited, &credentials, &long_options);
        assert_eq!(shape.cache_control, Some(ephemeral.clone()));
        assert_eq!(
            anthropic_beta_features(&limited, &context, &shape),
            vec![
                ANTHROPIC_FINE_GRAINED_TOOL_STREAMING_BETA,
                ANTHROPIC_INTERLEAVED_THINKING_BETA
            ]
        );
        let sent = build_anthropic_messages_request(&limited, &context, &long_options, &shape)
            .expect("body");
        assert!(sent["tools"][0].get("cache_control").is_none());
        assert!(sent["tools"][0].get("eager_input_streaming").is_none());

        let mut none_options = request_options.clone();
        none_options.cache_retention = agent::CacheRetention::None;
        let shape = anthropic_request_shape(&request_model, &credentials, &none_options);
        let sent =
            build_anthropic_messages_request(&request_model, &context, &none_options, &shape)
                .expect("body");
        assert!(!sent.to_string().contains("cache_control"));
        assert_eq!(sent["messages"][0]["content"], "weather?");

        let mut adaptive = request_model.clone();
        adaptive.compat = Some(json!({"forceAdaptiveThinking": true}));
        adaptive.thinking_level_map =
            BTreeMap::from([(llm::THINKING_XHIGH.to_owned(), Some("xhigh".to_owned()))]);
        let mut xhigh_options = request_options.clone();
        xhigh_options.thinking_level = llm::THINKING_XHIGH.to_owned();
        let shape = anthropic_request_shape(&adaptive, &credentials, &xhigh_options);
        assert!(anthropic_beta_features(&adaptive, &context, &shape).is_empty());
        let sent = build_anthropic_messages_request(&adaptive, &context, &xhigh_options, &shape)
            .expect("body");
        assert_eq!(
            sent["thinking"],
            json!({"type": "adaptive", "display": "summarized"})
        );
        assert_eq!(sent["output_config"]["effort"], "xhigh");
        let mut minimal_options = request_options.clone();
        minimal_options.thinking_level = llm::THINKING_MINIMAL.to_owned();
        let shape = anthropic_request_shape(&adaptive, &credentials, &minimal_options);
        let sent = build_anthropic_messages_request(&adaptive, &context, &minimal_options, &shape)
            .expect("body");
        assert_eq!(sent["output_config"]["effort"], "low");

        let mut off_options = request_options.clone();
        off_options.thinking_level = llm::THINKING_OFF.to_owned();
        let shape = anthropic_request_shape(&request_model, &credentials, &off_options);
        assert!(anthropic_beta_features(&request_model, &context, &shape).is_empty());
        let sent = build_anthropic_messages_request(&request_model, &context, &off_options, &shape)
            .expect("body");
        assert_eq!(sent["thinking"], json!({"type": "disabled"}));
    }

    #[test]
    fn anthropic_oauth_tokens_use_the_claude_code_request_shape() {
        let body = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_oauth\",\"model\":\"claude-test\",\"usage\":{\"input_tokens\":5}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"Read\",\"input\":{\"path\":\"a.rs\"}}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let request_model = model(API_ANTHROPIC_MESSAGES, base_url);
        let mut context = text_context();
        context.tools[0].name = "read".to_owned();
        context.messages.push(assistant_turn(
            &request_model,
            stream::STOP_TOOL_USE,
            vec![llm::ContentBlock::ToolCall(llm::ToolCall {
                id: "toolu_0".to_owned(),
                name: "read".to_owned(),
                ..llm::ToolCall::default()
            })],
        ));
        context
            .messages
            .push(llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: "toolu_0".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![llm::ContentBlock::text("contents")],
                timestamp: 3,
                ..llm::ToolResultMessage::default()
            })));
        let response =
            factory_with_credentials(0, ProviderCredentials::api_key("sk-ant-oat01-secret"))
                .respond(
                    &request_model,
                    &context,
                    options(agent::CancellationToken::default()),
                )
                .expect("Anthropic OAuth response");
        let request = requests.recv().expect("captured OAuth request");
        server.join().expect("test server finishes");

        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer sk-ant-oat01-secret")
        );
        assert!(!request.headers.contains_key("x-api-key"));
        assert_eq!(
            request.headers.get("user-agent").map(String::as_str),
            Some("claude-cli/2.1.251")
        );
        assert_eq!(
            request.headers.get("x-app").map(String::as_str),
            Some("cli")
        );
        let betas = request.headers.get("anthropic-beta").expect("beta header");
        assert!(
            betas.contains("claude-code-20250219") && betas.contains("oauth-2025-04-20"),
            "{betas}"
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("Anthropic JSON body");
        assert_eq!(sent["system"][0]["text"], CLAUDE_CODE_IDENTITY);
        assert_eq!(sent["system"][1]["text"], "be concise");
        assert_eq!(sent["tools"][0]["name"], "Read");
        assert_eq!(sent["messages"][1]["content"][0]["name"], "Read");
        let llm::ContentBlock::ToolCall(call) = &response.content[0] else {
            panic!("expected a tool call");
        };
        assert_eq!(call.name, "read");
    }

    #[test]
    fn completions_error_chunks_surface_the_provider_message() {
        let body = concat!(
            "data: {\"id\":\"chat_err\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"error\":{\"message\":\"Provider returned error\",\"code\":502,\"metadata\":{\"raw\":\"upstream exploded\"}}}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let failed = factory(0)
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("normalized error message");
        requests.recv().expect("captured request");
        server.join().expect("test server finishes");
        assert_eq!(failed.stop_reason, stream::STOP_ERROR);
        assert_eq!(
            failed.error_message,
            "provider protocol error: Provider returned error\nupstream exploded"
        );
        assert_eq!(failed.content[0].plain_text(), Some("hi"));
    }

    #[test]
    fn emitter_snapshots_are_copied_only_while_a_consumer_holds_one() {
        let events = stream::AssistantMessageEventStream::with_capacity(8).expect("stream");
        let request_model = model(API_OPENAI_COMPLETIONS, "https://example.test".to_owned());
        let mut emitter = MessageEmitter::new(
            events.clone(),
            &request_model,
            agent::CancellationToken::default(),
        );
        let index = emitter.start_text("").expect("text start");
        drop(events.try_next().expect("text_start event"));
        let before = Arc::as_ptr(&emitter.message);
        emitter.append_text(index, "a").expect("delta");
        assert_eq!(
            Arc::as_ptr(&emitter.message),
            before,
            "no consumer held the snapshot, so it is mutated in place"
        );
        let held = events
            .try_next()
            .expect("delta event")
            .partial
            .expect("partial snapshot");
        emitter.append_text(index, "b").expect("delta");
        assert_ne!(
            Arc::as_ptr(&emitter.message),
            Arc::as_ptr(&held),
            "a held snapshot forces a copy"
        );
        assert_eq!(held.content[0].plain_text(), Some("a"));
        assert_eq!(emitter.message().content[0].plain_text(), Some("ab"));
    }

    #[test]
    fn publishing_into_a_stalled_stream_stops_on_cancellation_or_when_orphaned() {
        let request_model = model(API_OPENAI_COMPLETIONS, "https://example.test".to_owned());
        let events = stream::AssistantMessageEventStream::with_capacity(1).expect("stream");
        let cancellation = agent::CancellationToken::default();
        let mut emitter = MessageEmitter::new(events.clone(), &request_model, cancellation.clone());
        emitter.start().expect("first event fills the queue");
        let canceller = {
            let cancellation = cancellation.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(100));
                cancellation.cancel();
            })
        };
        let started = Instant::now();
        let error = emitter
            .start_text("")
            .expect_err("a full queue must not pin the worker once cancelled");
        assert!(matches!(error, ProviderAdapterError::Cancelled), "{error}");
        assert!(started.elapsed() < Duration::from_secs(1));
        canceller.join().expect("canceller finishes");

        // With the last consumer handle gone the worker gives up on its own.
        let orphaned = stream::AssistantMessageEventStream::with_capacity(1).expect("stream");
        let mut emitter = MessageEmitter::new(
            orphaned,
            &request_model,
            agent::CancellationToken::default(),
        );
        emitter.start().expect("first event fills the queue");
        let error = emitter.start_text("").expect_err("orphaned stream");
        assert!(
            matches!(error, ProviderAdapterError::EventStream(_)),
            "{error}"
        );
    }

    #[test]
    fn completions_compat_ports_thinking_formats_reasoning_effort_and_strict_mode() {
        let request_options = options(agent::CancellationToken::default());
        let turn = |provider: &str, content: Vec<llm::ContentBlock>| {
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                api: API_OPENAI_COMPLETIONS.to_owned(),
                provider: provider.to_owned(),
                model: "test-model".to_owned(),
                stop_reason: stream::STOP_STOP.to_owned(),
                content,
                timestamp: 2,
                ..llm::AssistantMessage::default()
            }))
        };
        let build =
            |provider: &str, compat: Option<Value>, level: &str, extra: Vec<llm::Message>| {
                let mut request_model =
                    model(API_OPENAI_COMPLETIONS, "https://example.test/v1".to_owned());
                request_model.provider = provider.to_owned();
                request_model.reasoning = true;
                request_model.compat = compat;
                let mut context = text_context();
                context.messages.extend(extra);
                let mut request_options = request_options.clone();
                request_options.thinking_level = level.to_owned();
                build_openai_completions_request(&request_model, &context, &request_options)
                    .expect("completions body")
            };

        let zai = build("zai", None, llm::THINKING_HIGH, vec![]);
        assert_eq!(
            zai["thinking"],
            json!({"type": "enabled", "clear_thinking": false})
        );
        assert!(zai.get("reasoning_effort").is_none());
        assert!(zai.get("store").is_none());
        assert_eq!(zai["max_tokens"], 4_096);
        assert_eq!(zai["tools"][0]["function"]["strict"], false);
        let zai_off = build("zai", None, llm::THINKING_OFF, vec![]);
        assert_eq!(zai_off["thinking"], json!({"type": "disabled"}));

        let deepseek = build(
            "deepseek",
            None,
            llm::THINKING_HIGH,
            vec![turn("deepseek", vec![llm::ContentBlock::text("earlier")])],
        );
        assert_eq!(deepseek["thinking"], json!({"type": "enabled"}));
        assert_eq!(deepseek["reasoning_effort"], "high");
        assert_eq!(deepseek["messages"][2]["content"], "earlier");
        assert_eq!(deepseek["messages"][2]["reasoning_content"], "");

        let moonshot = build("moonshotai", None, llm::THINKING_HIGH, vec![]);
        assert!(moonshot["tools"][0]["function"].get("strict").is_none());
        assert!(moonshot.get("reasoning_effort").is_none());

        let openrouter = build("openrouter", None, llm::THINKING_MEDIUM, vec![]);
        assert_eq!(openrouter["reasoning"], json!({"effort": "medium"}));
        assert!(openrouter.get("reasoning_effort").is_none());
        let openrouter_off = build("openrouter", None, llm::THINKING_OFF, vec![]);
        assert_eq!(openrouter_off["reasoning"], json!({"effort": "none"}));

        let no_effort = build(
            "openai",
            Some(json!({"supportsReasoningEffort": false})),
            llm::THINKING_HIGH,
            vec![],
        );
        assert!(no_effort.get("reasoning_effort").is_none());
        let qwen = build(
            "openai",
            Some(json!({"thinkingFormat": "qwen"})),
            llm::THINKING_LOW,
            vec![],
        );
        assert_eq!(qwen["enable_thinking"], true);
        assert_eq!(qwen["reasoning_effort"], "low");
        let budgeted = build(
            "openai",
            Some(json!({
                "thinkingFormat": "chat-template",
                "thinkingTokenBudgetField": "thinking_budget",
                "chatTemplateKwargs": {
                    "enable_thinking": {"$var": "thinking.enabled"},
                    "budget": {"$var": "thinking.budget"},
                    "mode": {"omitWhenOff": true},
                },
            })),
            llm::THINKING_LOW,
            vec![],
        );
        assert_eq!(budgeted["thinking_budget"], 2_048);
        assert_eq!(
            budgeted["chat_template_kwargs"],
            json!({"enable_thinking": true, "budget": 2_048, "mode": "low"})
        );

        let as_text = build(
            "openai",
            Some(json!({"requiresThinkingAsText": true})),
            llm::THINKING_HIGH,
            vec![turn(
                "openai",
                vec![
                    thinking("why", "reasoning_content"),
                    llm::ContentBlock::text("answer"),
                ],
            )],
        );
        assert_eq!(
            as_text["messages"][2]["content"],
            json!([{"type": "text", "text": "why"}, {"type": "text", "text": "answer"}])
        );
        let field_replay = build(
            "openai",
            None,
            llm::THINKING_HIGH,
            vec![turn(
                "openai",
                vec![
                    thinking("why", "reasoning_content"),
                    llm::ContentBlock::text("answer"),
                ],
            )],
        );
        assert_eq!(field_replay["messages"][2]["reasoning_content"], "why");
        assert_eq!(field_replay["messages"][2]["content"], "answer");
        let opaque = build(
            "openai",
            None,
            llm::THINKING_HIGH,
            vec![turn(
                "openai",
                vec![thinking("why", "opaque"), llm::ContentBlock::text("answer")],
            )],
        );
        assert!(opaque["messages"][2].get("reasoning_content").is_none());
    }

    #[test]
    fn provider_error_bodies_are_capped_like_pi() {
        let body = "x".repeat(10_000);
        let (base_url, requests, server) = test_server(vec![http_response(500, &body)]);
        let failed = factory(0)
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("normalized error");
        requests.recv().expect("captured request");
        server.join().expect("test server finishes");
        assert_eq!(failed.stop_reason, stream::STOP_ERROR);
        assert!(
            failed.error_message.ends_with("... [truncated 6000 chars]"),
            "{}",
            &failed.error_message[failed.error_message.len().saturating_sub(60)..]
        );
        assert!(failed.error_message.len() < MAX_PROVIDER_ERROR_BODY_CHARS + 100);
    }

    #[test]
    fn completions_usage_accepts_top_level_cached_tokens() {
        let body = concat!(
            "data: {\"id\":\"chat_kimi\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"cached_tokens\":4}}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let response = factory(0)
            .respond(
                &model(API_OPENAI_COMPLETIONS, base_url),
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("completion response");
        requests.recv().expect("captured request");
        server.join().expect("test server finishes");
        assert_eq!(response.usage.cache_read, 4);
        assert_eq!(response.usage.input, 6);
        assert_eq!(response.usage.output, 2);
        assert_eq!(response.usage.total_tokens, 12);
    }
}
