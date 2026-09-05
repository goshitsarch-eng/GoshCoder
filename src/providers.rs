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
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    error::Error,
    fmt,
    io::Read,
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
    agent, aperture, bedrock, catalog, google_auth, llm, mistral, omni_prompt_tools, stream,
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
    OmniPromptTools,
    BedrockConverseStream,
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
            omni_prompt_tools::API_OMNI_PROMPT_TOOLS => Ok(Self::OmniPromptTools),
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
            Self::OmniPromptTools => {
                unreachable!("Omni prompt tools wraps the OpenAI Completions adapter")
            }
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
        Self::Sse(error)
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
    /// Whole-request timeout for standard request/response protocols.
    ///
    /// Mistral streaming intentionally does not use a whole-request deadline:
    /// it may stream longer than this setting. A caller-provided client
    /// retains its own timeout policy.
    pub request_timeout: Option<Duration>,
    /// Deadline for Mistral to return HTTP response headers.
    ///
    /// This bounds time to first byte without truncating an active SSE
    /// response. `None` disables the deadline.
    pub mistral_response_header_timeout: Option<Duration>,
    /// Capacity of the externally visible normalized event stream.
    pub event_buffer_capacity: usize,
    /// Maximum response-error body retained in an error message.
    pub max_error_body_bytes: usize,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_delay_limit: stream::RetryDelayLimit::Default,
            request_timeout: Some(Duration::from_secs(120)),
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
        // A blocking reqwest client applies its timeout to the entire
        // response body. Keep the client deadline-free so Mistral's active
        // SSE stream cannot be cut off; standard protocols install their
        // configured deadline on each individual request below.
        let mut builder = Client::builder().timeout(None);
        if let Some(timeout) = config.request_timeout {
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
        let events = self.stream_with_credentials(model, context, options, credentials);
        while let Some(event) = events.next() {
            let terminal_message = event.terminal_message();
            if let Some(listener) = &assistant_event_listener {
                listener(event);
            }
            if let Some(message) = terminal_message {
                return Ok((*message).clone());
            }
        }
        Err(ProviderAdapterError::StreamClosed)
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
        if model.api == omni_prompt_tools::API_OMNI_PROMPT_TOOLS {
            return self.stream_omni_prompt_tools_with_credentials(
                model,
                context,
                options,
                credentials,
            );
        }
        let events =
            stream::AssistantMessageEventStream::with_capacity(self.config.event_buffer_capacity)
                .expect("validated provider event buffer capacity");
        let worker_events = events.clone();
        let factory = self.clone();
        let model = model.clone();
        let context = context.clone();
        thread::spawn(move || {
            let cancellation = options.cancellation.clone();
            let mut emitter = MessageEmitter::new(worker_events, &model);
            if let Err(error) =
                factory.run_stream(&model, &context, options, credentials, &mut emitter)
            {
                let _ = emitter.fail(error, &cancellation);
            }
        });
        events
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
                    timeout: self.config.request_timeout,
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

    fn stream_omni_prompt_tools_with_credentials(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        options: agent::RequestOptions,
        credentials: ProviderCredentials,
    ) -> stream::AssistantMessageEventStream {
        let events =
            stream::AssistantMessageEventStream::with_capacity(self.config.event_buffer_capacity)
                .expect("validated provider event buffer capacity");
        let worker_events = events.clone();
        let factory = self.clone();
        let model = model.clone();
        let context = context.clone();
        thread::spawn(move || {
            let cancellation = options.cancellation.clone();
            let mut emitter = MessageEmitter::new(worker_events, &model);
            if let Err(error) =
                factory.run_omni_prompt_tools(&model, &context, options, credentials, &mut emitter)
            {
                let _ = emitter.fail(error, &cancellation);
            }
        });
        events
    }

    fn run_omni_prompt_tools(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        mut options: agent::RequestOptions,
        credentials: ProviderCredentials,
        emitter: &mut MessageEmitter,
    ) -> Result<()> {
        ensure_not_cancelled(&options.cancellation)?;
        emitter.start()?;

        let inner_context = omni_prompt_tools::inner_context(context);
        let mut inner_model = model.clone();
        inner_model.api = API_OPENAI_COMPLETIONS.to_owned();
        // The adapter owns the externally visible event stream. The hidden
        // completion must be drained silently before native events are
        // replayed, otherwise callers would receive duplicate raw XML events.
        options.assistant_event_listener = None;
        let response =
            self.respond_with_credentials(&inner_model, &inner_context, options, credentials)?;

        emitter.message.usage = response.usage.clone();
        emitter.message.response_id = response.response_id.clone();
        emitter.message.response_model = response.response_model.clone();
        if matches!(
            response.stop_reason.as_str(),
            stream::STOP_ERROR | stream::STOP_ABORTED
        ) {
            return emitter.finish_error(&response.stop_reason, response.error_message);
        }

        omni_prompt_tools::replay_response(emitter, &response)?;
        emitter.finish()
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
            | ProviderProtocol::OmniPromptTools
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
                build_anthropic_messages_request(model, context, &options)
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
            ProviderProtocol::OmniPromptTools => {
                unreachable!("Omni prompt tools is dispatched before the generic HTTP adapter")
            }
            ProviderProtocol::BedrockConverseStream => {
                unreachable!("Bedrock is dispatched before the generic HTTP adapter")
            }
        }?;
        let response =
            self.send_streaming_request(protocol, model, &payload, &credentials, &options)?;

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
            ProviderProtocol::AnthropicMessages => {
                consume_anthropic_messages(response, model, &options.cancellation, emitter)?
            }
            ProviderProtocol::GoogleGenerativeAi => {
                consume_google_generate_content(response, model, &options.cancellation, emitter)?
            }
            ProviderProtocol::GoogleVertex => {
                consume_google_generate_content(response, model, &options.cancellation, emitter)?
            }
            ProviderProtocol::MistralConversations => {
                mistral::consume_mistral_conversations(response, &options.cancellation, emitter)?
            }
            ProviderProtocol::OmniPromptTools => {
                unreachable!("Omni prompt tools is dispatched before the generic HTTP adapter")
            }
            ProviderProtocol::BedrockConverseStream => {
                unreachable!("Bedrock is dispatched before the generic HTTP adapter")
            }
        }
        ensure_not_cancelled(&options.cancellation)?;
        if emitter.message.stop_reason.is_empty()
            || emitter.message.stop_reason == stream::STOP_PENDING
        {
            return Err(ProviderAdapterError::Protocol(
                "stream ended without a terminal stop reason".to_owned(),
            ));
        }
        if emitter.message.stop_reason == stream::STOP_ERROR
            || emitter.message.stop_reason == stream::STOP_ABORTED
        {
            return Err(ProviderAdapterError::Protocol(
                emitter.message.error_message.clone(),
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
    ) -> Result<Response> {
        let endpoint = protocol_endpoint(model, protocol, credentials)?;
        let headers = build_request_headers(
            protocol,
            model,
            credentials,
            &options.session_id,
            options.cache_retention,
            &options.cancellation,
        )?;
        let body = serde_json::to_vec(payload)?;

        let mut retry_index = 0;
        loop {
            ensure_not_cancelled(&options.cancellation)?;
            let sent = if protocol == ProviderProtocol::MistralConversations {
                self.send_mistral_request(
                    endpoint.clone(),
                    headers.clone(),
                    body.clone(),
                    &options.cancellation,
                )
            } else {
                self.send_standard_request(endpoint.clone(), headers.clone(), body.clone())
            };
            match sent {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let error = mark_aperture_retryable_provider_error(
                        model,
                        provider_error_from_response(response, self.config.max_error_body_bytes),
                    );
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
                    let error = mark_aperture_retryable_provider_error(model, error);
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

    fn send_standard_request(
        &self,
        endpoint: Url,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<Response> {
        let mut request = self.client.post(endpoint).headers(headers).body(body);
        if let Some(timeout) = self.config.request_timeout {
            request = request.timeout(timeout);
        }
        request.send().map_err(provider_network_error)
    }

    fn send_mistral_request(
        &self,
        endpoint: Url,
        headers: HeaderMap,
        body: Vec<u8>,
        cancellation: &agent::CancellationToken,
    ) -> Result<Response> {
        let (sender, receiver) = mpsc::sync_channel(1);
        let client = self.client.clone();
        thread::spawn(move || {
            let response = client.post(endpoint).headers(headers).body(body).send();
            let _ = sender.send(response);
        });

        let deadline = self
            .config
            .mistral_response_header_timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));
        loop {
            ensure_not_cancelled(cancellation)?;
            let wait = deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::from_millis(25))
                .min(Duration::from_millis(25));
            if let Some(deadline) = deadline
                && wait.is_zero()
                && Instant::now() >= deadline
            {
                return Err(ProviderAdapterError::Provider(stream::ProviderError::new(
                    0,
                    "Mistral response headers timed out",
                )));
            }
            match receiver.recv_timeout(wait) {
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(error)) => return Err(provider_network_error(error)),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(ProviderAdapterError::Protocol(
                        "Mistral request worker exited before returning a response".to_owned(),
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
            let aperture_state = catalog.aperture_state();
            let is_aperture_routed = aperture_state.configured
                && ((model.provider == aperture::DEDICATED_PROVIDER_ID
                    && aperture_state.resolved.dedicated_enabled)
                    || aperture_state.routes.contains_key(&model.provider));
            // Keep explicit caller overrides (notably test/enterprise base
            // URLs) for native providers. Aperture must instead use the
            // current catalog model so changed gateway routing takes effect.
            let request_model = if is_aperture_routed {
                &resolved.model
            } else {
                model
            };
            let routed_model = aperture::rewrite_request_model(
                Some(&aperture_state),
                request_model,
                &options.session_id,
            );
            let aperture_routed = matches!(&routed_model, std::borrow::Cow::Owned(_));
            transport
                .respond_with_credentials(routed_model.as_ref(), context, options, credentials)
                .map_err(|error| {
                    let message = error.to_string();
                    if aperture_routed {
                        aperture::mark_retryable_error(&message).unwrap_or(message)
                    } else {
                        message
                    }
                })
        })
    }
}

fn mark_aperture_retryable_provider_error(
    model: &llm::Model,
    mut error: stream::ProviderError,
) -> stream::ProviderError {
    let routed_through_aperture = model.headers.iter().any(|(name, value)| {
        name.eq_ignore_ascii_case("referer") && value == aperture::APERTURE_REFERER
    });
    if routed_through_aperture && let Some(message) = aperture::mark_retryable_error(&error.message)
    {
        error.message = message;
        error
            .headers
            .insert("x-should-retry".to_owned(), "true".to_owned());
    }
    error
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
        ProviderProtocol::OmniPromptTools | ProviderProtocol::BedrockConverseStream => {
            unreachable!("adapter uses its own request builder");
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
    let truncated = body.len() > maximum_body_bytes;
    body.truncate(maximum_body_bytes);
    let body = String::from_utf8_lossy(&body).trim().to_owned();
    let suffix = if body.is_empty() {
        String::new()
    } else if truncated {
        format!(": {body}…")
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

fn build_request_headers(
    protocol: ProviderProtocol,
    model: &llm::Model,
    credentials: &ProviderCredentials,
    session_id: &str,
    cache_retention: agent::CacheRetention,
    cancellation: &agent::CancellationToken,
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
        }
        ProviderProtocol::BedrockConverseStream => {
            unreachable!("Bedrock uses its own signed request builder")
        }
        ProviderProtocol::OmniPromptTools => {
            unreachable!("Omni prompt tools wraps the OpenAI Completions adapter")
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
                set_header_override(
                    &mut overrides,
                    "x-api-key".to_owned(),
                    Some(api_key.to_owned()),
                );
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
            ProviderProtocol::OmniPromptTools | ProviderProtocol::BedrockConverseStream => {
                unreachable!("adapter uses its own request builder")
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
        ProviderProtocol::OmniPromptTools => {
            unreachable!("Omni prompt tools wraps the OpenAI Completions adapter")
        }
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
        ProviderProtocol::OmniPromptTools | ProviderProtocol::BedrockConverseStream => {
            unreachable!("adapter uses its own request builder")
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

#[derive(Clone, Copy)]
struct OpenAiCompletionsCompat {
    supports_store: bool,
    supports_usage_in_streaming: bool,
    supports_finish_reason: bool,
    supports_developer_role: bool,
    requires_tool_result_name: bool,
    requires_assistant_after_tool_result: bool,
    max_tokens_field: &'static str,
}

impl OpenAiCompletionsCompat {
    fn from_model(model: &llm::Model) -> Self {
        let base_url = model.base_url.as_str();
        let non_standard = matches!(
            model.provider.as_str(),
            "cerebras"
                | "cloudflare-ai-gateway"
                | "cloudflare-workers-ai"
                | "deepseek"
                | "moonshotai"
                | "moonshotai-cn"
                | "nvidia"
                | "together"
                | "zai"
                | "zai-coding-cn"
        ) || base_url.contains("cerebras.ai")
            || base_url.contains("deepseek.com")
            || base_url.contains("openrouter.ai");
        let default_max_tokens = if matches!(
            model.provider.as_str(),
            "deepseek" | "moonshotai" | "moonshotai-cn" | "together" | "zai" | "zai-coding-cn"
        ) {
            "max_tokens"
        } else {
            "max_completion_tokens"
        };
        let max_tokens_field = match compat_string(model, "maxTokensField").as_deref() {
            Some("max_tokens") => "max_tokens",
            _ => default_max_tokens,
        };
        Self {
            supports_store: compat_bool(model, "supportsStore", !non_standard),
            supports_usage_in_streaming: compat_bool(model, "supportsUsageInStreaming", true),
            supports_finish_reason: compat_bool(model, "supportsFinishReason", true),
            supports_developer_role: compat_bool(
                model,
                "supportsDeveloperRole",
                !non_standard || model.provider == "openrouter",
            ),
            requires_tool_result_name: compat_bool(model, "requiresToolResultName", false),
            requires_assistant_after_tool_result: compat_bool(
                model,
                "requiresAssistantAfterToolResult",
                false,
            ),
            max_tokens_field,
        }
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
        Value::Array(openai_chat_messages(model, context, compat)),
    );
    body.insert("stream".to_owned(), Value::Bool(true));
    if compat.supports_usage_in_streaming {
        body.insert("stream_options".to_owned(), json!({"include_usage": true}));
    }
    if compat.supports_store {
        body.insert("store".to_owned(), Value::Bool(false));
    }
    if let Some(max_tokens) = requested_max_tokens(model, context) {
        body.insert(
            compat.max_tokens_field.to_owned(),
            Value::Number(max_tokens.into()),
        );
    }
    if let Some(effort) = mapped_thinking_level(model, &options.thinking_level) {
        body.insert("reasoning_effort".to_owned(), Value::String(effort));
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
            Value::Array(openai_chat_tools(&context.tools)),
        );
    }
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

fn openai_chat_tools(tools: &[llm::Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema_or_empty(&tool.parameters),
                    "strict": false,
                }
            })
        })
        .collect()
}

fn openai_chat_messages(
    model: &llm::Model,
    context: &llm::Context,
    compat: OpenAiCompletionsCompat,
) -> Vec<Value> {
    let mut messages = Vec::new();
    if !context.system_prompt.is_empty() {
        let role = if model.reasoning && compat.supports_developer_role {
            "developer"
        } else {
            "system"
        };
        messages.push(json!({"role": role, "content": context.system_prompt}));
    }

    let mut ids = WireIdNormalizer::default();
    let mut pending = BTreeMap::<String, String>::new();
    for message in &context.messages {
        if matches!(message, llm::Message::User(_)) {
            flush_missing_openai_tool_results(&mut messages, &mut pending, compat);
        }
        match message {
            llm::Message::User(user) => {
                messages.push(json!({
                    "role": "user",
                    "content": openai_user_content(&user.content, model.supports_images()),
                }));
            }
            llm::Message::Assistant(assistant) => {
                if assistant.stop_reason == stream::STOP_ERROR {
                    continue;
                }
                let mut item = Map::new();
                item.insert("role".to_owned(), Value::String("assistant".to_owned()));
                let text = text_from_blocks(&assistant.content);
                item.insert(
                    "content".to_owned(),
                    if text.is_empty() {
                        Value::Null
                    } else {
                        Value::String(text)
                    },
                );
                let thinking = thinking_from_blocks(&assistant.content);
                if !thinking.is_empty() {
                    item.insert("reasoning_content".to_owned(), Value::String(thinking));
                }
                let calls = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        llm::ContentBlock::ToolCall(call) => Some(call),
                        _ => None,
                    })
                    .map(|call| {
                        let id = ids.normalize(&call.id, 64);
                        pending.insert(id.clone(), call.name.clone());
                        json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .unwrap_or_else(|_| "{}".to_owned()),
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                if !calls.is_empty() {
                    item.insert("tool_calls".to_owned(), Value::Array(calls));
                }
                if item.get("content").is_some_and(|content| content.is_null())
                    && !item.contains_key("tool_calls")
                {
                    continue;
                }
                messages.push(Value::Object(item));
            }
            llm::Message::ToolResult(result) => {
                let id = ids.normalize(&result.tool_call_id, 64);
                pending.remove(&id);
                let mut item = Map::new();
                item.insert("role".to_owned(), Value::String("tool".to_owned()));
                item.insert("tool_call_id".to_owned(), Value::String(id));
                item.insert(
                    "content".to_owned(),
                    Value::String(tool_result_text(&result.content)),
                );
                if compat.requires_tool_result_name {
                    item.insert("name".to_owned(), Value::String(result.tool_name.clone()));
                }
                messages.push(Value::Object(item));
                if compat.requires_assistant_after_tool_result {
                    messages.push(json!({
                        "role": "assistant",
                        "content": "I have processed the tool results."
                    }));
                }
            }
        }
    }
    messages
}

fn flush_missing_openai_tool_results(
    messages: &mut Vec<Value>,
    pending: &mut BTreeMap<String, String>,
    compat: OpenAiCompletionsCompat,
) {
    for (id, name) in std::mem::take(pending) {
        let mut item = Map::new();
        item.insert("role".to_owned(), Value::String("tool".to_owned()));
        item.insert("tool_call_id".to_owned(), Value::String(id));
        item.insert(
            "content".to_owned(),
            Value::String("No result provided".to_owned()),
        );
        if compat.requires_tool_result_name {
            item.insert("name".to_owned(), Value::String(name));
        }
        messages.push(Value::Object(item));
    }
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
    let image_aware = messages
        .iter()
        .cloned()
        .map(|message| downgrade_google_message_images(message, model))
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
                                if google_requires_tool_call_id(&model.id) {
                                    let normalized = google_normalize_tool_call_id(&tool_call.id);
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
                flush_missing_google_tool_results(
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
                flush_missing_google_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_results,
                );
                result.push(llm::Message::User(user));
            }
        }
    }
    flush_missing_google_tool_results(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_results,
    );
    result
}

fn downgrade_google_message_images(message: llm::Message, model: &llm::Model) -> llm::Message {
    if model.supports_images() {
        return message;
    }
    match message {
        llm::Message::User(mut user) => {
            if let llm::UserContent::Blocks(blocks) = user.content {
                user.content = llm::UserContent::Blocks(google_replace_images_with_placeholder(
                    blocks,
                    "(image omitted: model does not support images)",
                ));
            }
            llm::Message::User(user)
        }
        llm::Message::ToolResult(tool_result) => {
            let mut copy = (*tool_result).clone();
            copy.content = google_replace_images_with_placeholder(
                copy.content,
                "(tool image omitted: model does not support images)",
            );
            llm::Message::ToolResult(Box::new(copy))
        }
        other => other,
    }
}

fn google_replace_images_with_placeholder(
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

fn flush_missing_google_tool_results(
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

fn google_normalize_tool_call_id(id: &str) -> String {
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

fn openai_responses_input(model: &llm::Model, context: &llm::Context) -> Result<Vec<Value>> {
    let grammar_tool_input_properties = BTreeMap::new();
    let deferred_tools = BTreeMap::new();
    responses_input(
        model,
        context,
        ResponsesInputOptions {
            include_system_prompt: true,
            supports_developer_role: true,
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
        grammar_tool_input_properties,
        deferred_tools,
        deferred_tools_mode,
        tool_options,
    } = options;
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
    for (message_index, message) in context.messages.iter().enumerate() {
        match message {
            llm::Message::User(user) => {
                let content = responses_user_content(&user.content, model.supports_images());
                if !content.is_empty() {
                    input.push(json!({"role": "user", "content": content}));
                }
            }
            llm::Message::Assistant(assistant) => {
                if assistant.stop_reason == stream::STOP_ERROR {
                    continue;
                }
                for (block_index, block) in assistant.content.iter().enumerate() {
                    match block {
                        llm::ContentBlock::Thinking(thinking)
                            if assistant.provider == model.provider
                                && assistant.api == model.api
                                && !thinking.thinking_signature.is_empty() =>
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
                            let fallback = if block_index == 0 {
                                format!("msg_pi_{message_index}")
                            } else {
                                format!("msg_pi_{message_index}_{block_index}")
                            };
                            let id = signature_id.filter(|id| id.len() <= 64).unwrap_or(fallback);
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
                            let same_protocol =
                                assistant.provider == model.provider && assistant.api == model.api;
                            let same_model = same_protocol && assistant.model == model.id;
                            let grammar_input_property =
                                grammar_tool_input_properties.get(&call.name);
                            let item_id = if grammar_input_property.is_some() {
                                same_protocol.then_some(item_id).flatten()
                            } else {
                                same_protocol
                                    .then_some(item_id)
                                    .flatten()
                                    .filter(|id| id.starts_with("fc_"))
                            };
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

fn build_anthropic_messages_request(
    model: &llm::Model,
    context: &llm::Context,
    options: &agent::RequestOptions,
) -> Result<Value> {
    let max_tokens = requested_max_tokens(model, context)
        .or_else(|| (!model.max_tokens.eq(&0)).then_some(model.max_tokens))
        .unwrap_or(1);
    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(model.id.clone()));
    body.insert(
        "messages".to_owned(),
        Value::Array(anthropic_messages(model, context)),
    );
    body.insert("max_tokens".to_owned(), Value::Number(max_tokens.into()));
    body.insert("stream".to_owned(), Value::Bool(true));
    if !context.system_prompt.is_empty() {
        body.insert(
            "system".to_owned(),
            Value::Array(vec![json!({
                "type": "text",
                "text": context.system_prompt,
            })]),
        );
    }
    if !context.tools.is_empty() {
        let eager = compat_bool(model, "supportsEagerToolInputStreaming", true);
        body.insert(
            "tools".to_owned(),
            Value::Array(
                context
                    .tools
                    .iter()
                    .map(|tool| {
                        let mut item = json!({
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": schema_or_empty(&tool.parameters),
                        });
                        if eager {
                            item.as_object_mut()
                                .expect("JSON object")
                                .insert("eager_input_streaming".to_owned(), Value::Bool(true));
                        }
                        item
                    })
                    .collect(),
            ),
        );
    }
    if let Some(level) = mapped_thinking_level(model, &options.thinking_level) {
        let budget =
            thinking_budget(&level).min(max_tokens.saturating_sub(stream::MIN_ANSWER_TOKENS));
        if budget > 0 {
            body.insert(
                "thinking".to_owned(),
                json!({"type": "enabled", "budget_tokens": budget, "display": "summarized"}),
            );
        }
    }
    merge_sampling_params(&mut body, model);
    Ok(Value::Object(body))
}

fn thinking_budget(level: &str) -> u64 {
    match level {
        llm::THINKING_MINIMAL => 1_024,
        llm::THINKING_LOW => 2_048,
        llm::THINKING_MEDIUM => 8_192,
        _ => 16_384,
    }
}

fn anthropic_messages(model: &llm::Model, context: &llm::Context) -> Vec<Value> {
    let mut messages = Vec::new();
    let mut ids = WireIdNormalizer::default();
    let mut index = 0;
    while index < context.messages.len() {
        match &context.messages[index] {
            llm::Message::User(user) => {
                if let Some(content) =
                    anthropic_user_content(&user.content, model.supports_images())
                {
                    messages.push(json!({"role": "user", "content": content}));
                }
                index += 1;
            }
            llm::Message::Assistant(assistant) => {
                if assistant.stop_reason != stream::STOP_ERROR {
                    let same_protocol =
                        assistant.provider == model.provider && assistant.api == model.api;
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
                            llm::ContentBlock::Thinking(thinking)
                                if same_protocol && !thinking.thinking_signature.is_empty() =>
                            {
                                blocks.push(json!({
                                    "type": "thinking",
                                    "thinking": thinking.thinking,
                                    "signature": thinking.thinking_signature,
                                }));
                            }
                            llm::ContentBlock::Thinking(thinking)
                                if !thinking.thinking.trim().is_empty() =>
                            {
                                blocks.push(json!({"type": "text", "text": thinking.thinking}));
                            }
                            llm::ContentBlock::ToolCall(call) => {
                                blocks.push(json!({
                                    "type": "tool_use",
                                    "id": ids.normalize(&call.id, 64),
                                    "name": call.name,
                                    "input": call.arguments,
                                }));
                            }
                            llm::ContentBlock::Image(_)
                            | llm::ContentBlock::Text(_)
                            | llm::ContentBlock::Thinking(_) => {}
                        }
                    }
                    if !blocks.is_empty() {
                        messages.push(json!({"role": "assistant", "content": blocks}));
                    }
                }
                index += 1;
            }
            llm::Message::ToolResult(_) => {
                let mut blocks = Vec::new();
                while let Some(llm::Message::ToolResult(result)) = context.messages.get(index) {
                    blocks.push(json!({
                        "type": "tool_result",
                        "tool_use_id": ids.normalize(&result.tool_call_id, 64),
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

fn thinking_from_blocks(blocks: &[llm::ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Thinking(thinking) if !thinking.redacted => {
                Some(thinking.thinking.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_text(blocks: &[llm::ContentBlock]) -> String {
    let text = text_from_blocks(blocks);
    if !text.is_empty() {
        text
    } else if blocks
        .iter()
        .any(|block| matches!(block, llm::ContentBlock::Image(_)))
    {
        "(see attached image)".to_owned()
    } else {
        "(no tool output)".to_owned()
    }
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

#[derive(Default)]
struct WireIdNormalizer {
    values: BTreeMap<String, String>,
    used: BTreeSet<String>,
}

impl WireIdNormalizer {
    fn normalize(&mut self, source: &str, maximum_length: usize) -> String {
        if let Some(value) = self.values.get(source) {
            return value.clone();
        }
        let mut value = source
            .bytes()
            .map(|byte| match byte {
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' => byte as char,
                _ => '_',
            })
            .collect::<String>();
        if value.is_empty() {
            value = "call".to_owned();
        }
        value.truncate(maximum_length.max(1));
        let base = value.clone();
        let mut attempt = 1_u64;
        while self.used.contains(&value) {
            let suffix = format!("_{attempt}");
            let prefix_length = maximum_length.saturating_sub(suffix.len()).max(1);
            value = base.chars().take(prefix_length).collect::<String>();
            value.push_str(&suffix);
            attempt += 1;
        }
        self.used.insert(value.clone());
        self.values.insert(source.to_owned(), value.clone());
        value
    }
}

/// Mutable output plus safe snapshot publication for one provider turn.
pub(crate) struct MessageEmitter {
    events: stream::AssistantMessageEventStream,
    model: llm::Model,
    pub(crate) message: llm::AssistantMessage,
    usage_cost_multiplier: f64,
}

impl MessageEmitter {
    fn new(events: stream::AssistantMessageEventStream, model: &llm::Model) -> Self {
        Self {
            events,
            model: model.clone(),
            message: llm::AssistantMessage {
                role: "assistant".to_owned(),
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                stop_reason: stream::STOP_PENDING.to_owned(),
                timestamp: now_millis(),
                ..llm::AssistantMessage::default()
            },
            usage_cost_multiplier: 1.0,
        }
    }

    fn snapshot(&self) -> Arc<llm::AssistantMessage> {
        Arc::new(self.message.clone())
    }

    pub(crate) fn start(&mut self) -> Result<()> {
        self.publish(stream::AssistantMessageEvent::start(self.snapshot()))
    }

    pub(crate) fn start_text(&mut self, initial: &str) -> Result<usize> {
        let index = self.message.content.len();
        self.message
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
        match self.message.content.get_mut(index) {
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
        match self.message.content.get_mut(index) {
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
        match self.message.content.get_mut(index) {
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
        self.message
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
        match self.message.content.get_mut(index) {
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
        match self.message.content.get_mut(index) {
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
        match self.message.content.get_mut(index) {
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
        match self.message.content.get_mut(index) {
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
        self.message
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
        let Some(llm::ContentBlock::ToolCall(call)) = self.message.content.get_mut(index) else {
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
        let Some(llm::ContentBlock::ToolCall(call)) = self.message.content.get_mut(index) else {
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
        let Some(llm::ContentBlock::ToolCall(call)) = self.message.content.get_mut(index) else {
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
        stream::calculate_usage_cost(&self.model, &mut self.message.usage);
        if self.usage_cost_multiplier != 1.0 {
            let cost = &mut self.message.usage.cost;
            cost.input *= self.usage_cost_multiplier;
            cost.output *= self.usage_cost_multiplier;
            cost.cache_read *= self.usage_cost_multiplier;
            cost.cache_write *= self.usage_cost_multiplier;
            cost.total = cost.input + cost.output + cost.cache_read + cost.cache_write;
        }
    }

    fn finish(&mut self) -> Result<()> {
        self.calculate_usage_cost();
        self.events
            .push(stream::AssistantMessageEvent::done(
                self.message.stop_reason.clone(),
                self.snapshot(),
            ))
            .map_err(|error| ProviderAdapterError::EventStream(error.to_string()))?;
        self.events.end();
        Ok(())
    }

    fn fail(
        &mut self,
        error: ProviderAdapterError,
        cancellation: &agent::CancellationToken,
    ) -> Result<()> {
        let reason =
            if cancellation.is_cancelled() || matches!(&error, ProviderAdapterError::Cancelled) {
                stream::STOP_ABORTED
            } else {
                stream::STOP_ERROR
            };
        self.finish_error(reason, error.to_string())
    }

    fn finish_error(
        &mut self,
        requested_reason: &str,
        error_message: impl Into<String>,
    ) -> Result<()> {
        self.message.stop_reason = if requested_reason == stream::STOP_ABORTED {
            stream::STOP_ABORTED.to_owned()
        } else {
            stream::STOP_ERROR.to_owned()
        };
        self.message.error_message = error_message.into();
        self.calculate_usage_cost();
        let result = self.events.push(stream::AssistantMessageEvent::error(
            self.message.stop_reason.clone(),
            self.snapshot(),
        ));
        self.events.end();
        result.map_err(|error| ProviderAdapterError::EventStream(error.to_string()))
    }

    fn publish(&self, mut event: stream::AssistantMessageEvent) -> Result<()> {
        event.partial = Some(self.snapshot());
        self.events
            .push(event)
            .map_err(|error| ProviderAdapterError::EventStream(error.to_string()))
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

fn consume_openai_completions(
    response: Response,
    model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let compat = OpenAiCompletionsCompat::from_model(model);
    let mut reader = stream::SseReader::new(response);
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
        if emitter.message.response_id.is_empty()
            && let Some(id) = value_string(&chunk, "id")
        {
            emitter.message.response_id = id.to_owned();
        }
        if let Some(response_model) = value_string(&chunk, "model")
            && response_model != model.id
            && !response_model.is_empty()
            && emitter.message.response_model.is_empty()
        {
            emitter.message.response_model = response_model.to_owned();
        }
        if let Some(usage) = chunk.get("usage") {
            apply_openai_usage(&mut emitter.message.usage, usage);
        }

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            continue;
        };
        if let Some(usage) = choice.get("usage") {
            apply_openai_usage(&mut emitter.message.usage, usage);
        }
        if let Some(reason) =
            value_string(choice, "finish_reason").filter(|reason| !reason.is_empty())
        {
            saw_finish_reason = true;
            emitter.message.raw_stop_reason = reason.to_owned();
            let (stop_reason, error_message) = map_openai_stop_reason(reason);
            emitter.message.stop_reason = stop_reason;
            if !error_message.is_empty() {
                emitter.message.error_message = error_message;
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
        emitter.message.stop_reason = if tool_calls.is_empty() {
            stream::STOP_STOP.to_owned()
        } else {
            stream::STOP_TOOL_USE.to_owned()
        };
    }
    if emitter.message.stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message.error_message.clone(),
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
    response: Response,
    _model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let mut reader = stream::SseReader::new(response);
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
        if emitter.message.response_id.is_empty()
            && let Some(response_id) =
                value_string(&chunk, "responseId").filter(|response_id| !response_id.is_empty())
        {
            emitter.message.response_id = response_id.to_owned();
        }
        if let Some(usage) = chunk.get("usageMetadata") {
            apply_google_usage(&mut emitter.message.usage, usage);
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
            emitter.message.raw_stop_reason = reason.to_owned();
            let (stop_reason, error_message) = map_google_stop_reason(reason);
            emitter.message.stop_reason = stop_reason;
            emitter.message.error_message = if error_message.is_empty() {
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
    if emitter.message.stop_reason == stream::STOP_STOP
        && emitter
            .message
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::ToolCall(_)))
    {
        emitter.message.stop_reason = stream::STOP_TOOL_USE.to_owned();
    }
    if emitter.message.stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message.error_message.clone(),
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
        let previous = match emitter.message.content.get(index) {
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
        emitter.message.response_id = id.to_owned();
    }
    if let Some(usage) = response.get("usage") {
        apply_responses_usage(&mut emitter.message.usage, usage);
    }
    if let Some(end_turn) = response.get("end_turn").and_then(Value::as_bool) {
        emitter.message.end_turn = Some(end_turn);
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
    emitter.message.raw_stop_reason = if incomplete_reason.is_empty() {
        status.to_owned()
    } else {
        format!("{status}.{incomplete_reason}")
    };
    match status {
        "" | "completed" => {
            emitter.message.stop_reason = stream::STOP_STOP.to_owned();
            emitter.message.error_message.clear();
        }
        "incomplete" if incomplete_reason == "max_output_tokens" => {
            emitter.message.stop_reason = stream::STOP_LENGTH.to_owned();
            emitter.message.error_message.clear();
        }
        "incomplete" => {
            emitter.message.stop_reason = stream::STOP_ERROR.to_owned();
            emitter.message.error_message = if incomplete_reason.is_empty() {
                "Response incomplete without a provider reason".to_owned()
            } else {
                format!("Response incomplete: {incomplete_reason}")
            };
        }
        "failed" | "cancelled" => {
            emitter.message.stop_reason = stream::STOP_ERROR.to_owned();
            emitter.message.error_message = value_object(response, "error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Response {status}"));
        }
        "in_progress" | "queued" => {
            emitter.message.stop_reason = stream::STOP_STOP.to_owned();
        }
        other => {
            emitter.message.stop_reason = stream::STOP_ERROR.to_owned();
            emitter.message.error_message = format!("Unhandled response status: {other}");
        }
    }
    if emitter.message.stop_reason == stream::STOP_STOP
        && emitter
            .message
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::ToolCall(_)))
    {
        emitter.message.stop_reason = stream::STOP_TOOL_USE.to_owned();
    }
    Ok(())
}

fn consume_openai_responses(
    response: Response,
    _model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<()> {
    consume_responses(
        response,
        cancellation,
        emitter,
        false,
        grammar_tool_input_properties,
        None,
    )
}

fn consume_codex_responses(
    response: Response,
    model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
    grammar_tool_input_properties: &BTreeMap<String, String>,
) -> Result<()> {
    let requested_service_tier = requested_responses_service_tier(model);
    consume_responses(
        response,
        cancellation,
        emitter,
        true,
        grammar_tool_input_properties,
        Some(requested_service_tier),
    )
}

fn consume_responses(
    response: Response,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
    codex: bool,
    grammar_tool_input_properties: &BTreeMap<String, String>,
    codex_requested_service_tier: Option<&str>,
) -> Result<()> {
    let mut reader = stream::SseReader::new(response);
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
                    emitter.message.response_id = id.to_owned();
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
    if emitter.message.stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message.error_message.clone(),
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
    response: Response,
    _model: &llm::Model,
    cancellation: &agent::CancellationToken,
    emitter: &mut MessageEmitter,
) -> Result<()> {
    let mut reader = stream::SseReader::new(response);
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
                        emitter.message.response_id = id.to_owned();
                    }
                    if let Some(response_model) = value_string(message, "model")
                        && !response_model.is_empty()
                    {
                        emitter.message.response_model = response_model.to_owned();
                    }
                    if let Some(usage) = message.get("usage") {
                        apply_anthropic_usage(&mut emitter.message.usage, usage);
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
                        let content_index = emitter.start_tool(
                            value_string(block, "id").unwrap_or_default(),
                            value_string(block, "name").unwrap_or_default(),
                        )?;
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
                    emitter.message.raw_stop_reason = reason.to_owned();
                    let refusal_explanation = value_object(delta, "stop_details")
                        .and_then(|details| details.get("explanation"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let (stop_reason, error_message) =
                        map_anthropic_stop_reason(reason, refusal_explanation);
                    emitter.message.stop_reason = stop_reason;
                    if !error_message.is_empty() {
                        emitter.message.error_message = error_message;
                    }
                }
                if let Some(usage) = payload.get("usage") {
                    apply_anthropic_usage(&mut emitter.message.usage, usage);
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
    if emitter.message.stop_reason == stream::STOP_PENDING {
        return Err(ProviderAdapterError::Protocol(
            "Anthropic stream ended without a stop reason".to_owned(),
        ));
    }
    if emitter.message.stop_reason == stream::STOP_ERROR {
        return Err(ProviderAdapterError::Protocol(
            emitter.message.error_message.clone(),
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
                request_timeout: Some(Duration::from_secs(2)),
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

    fn omni_prompt_tools_sse(content: &str, finish_reason: &str) -> String {
        let first = json!({
            "id": "omni_1",
            "model": "gateway/chat-web",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": Value::Null,
            }],
        });
        let terminal = json!({
            "id": "omni_1",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
        });
        format!("data: {first}\n\ndata: {terminal}\n\ndata: [DONE]\n\n")
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
    fn omni_prompt_tools_converts_text_blocks_to_native_calls_without_inner_events() {
        let body = omni_prompt_tools_sse(
            "I will inspect it.\n<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>",
            "stop",
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, &body)]);
        let mut request_model = model(omni_prompt_tools::API_OMNI_PROMPT_TOOLS, base_url);
        request_model.provider = "omni".to_owned();
        request_model.id = "gateway/chat-web".to_owned();
        let observed_events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&observed_events);
        let mut request_options = options(agent::CancellationToken::default());
        request_options.assistant_event_listener = Some(Arc::new(move |event| {
            event_log
                .lock()
                .expect("OmniRoute event log lock")
                .push(event);
        }));

        let response = factory(0)
            .respond(&request_model, &text_context(), request_options)
            .expect("prompt-protocol response");
        let request = requests.recv().expect("captured OmniRoute request");
        server.join().expect("test server finishes");

        assert_eq!(request.target, "/chat/completions");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer test-key")
        );
        let sent: Value = serde_json::from_slice(&request.body).expect("completion JSON body");
        assert_eq!(sent["model"], "gateway/chat-web");
        assert!(
            sent.get("tools").is_none(),
            "the inner completion must not receive native tool definitions"
        );
        assert_eq!(sent["messages"][0]["role"], "system");
        assert_eq!(sent["messages"][1]["content"], "weather?");
        let system_prompt = sent["messages"][0]["content"]
            .as_str()
            .expect("system prompt");
        assert!(system_prompt.contains("be concise"));
        assert!(system_prompt.contains("# Tool calling protocol"));
        assert!(system_prompt.contains("### weather"));

        assert_eq!(response.api, omni_prompt_tools::API_OMNI_PROMPT_TOOLS);
        assert_eq!(response.stop_reason, stream::STOP_TOOL_USE);
        assert_eq!(response.response_id, "omni_1");
        assert_eq!(response.usage.total_tokens, 18);
        assert_eq!(response.content[0].plain_text(), Some("I will inspect it."));
        let llm::ContentBlock::ToolCall(call) = &response.content[1] else {
            panic!("expected native tool call, got {:?}", response.content);
        };
        assert!(call.id.starts_with("call_omni_"));
        assert_eq!(call.name, "weather");
        assert_eq!(call.arguments.get("city"), Some(&json!("Paris")));

        let observed_events = observed_events
            .lock()
            .expect("OmniRoute event log lock")
            .clone();
        assert_eq!(
            observed_events
                .iter()
                .map(|event| event.event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                stream::EVENT_START,
                stream::EVENT_TEXT_START,
                stream::EVENT_TEXT_DELTA,
                stream::EVENT_TEXT_END,
                stream::EVENT_TOOLCALL_START,
                stream::EVENT_TOOLCALL_DELTA,
                stream::EVENT_TOOLCALL_END,
                stream::EVENT_DONE,
            ]
        );
        let text_start = observed_events
            .iter()
            .find(|event| event.event_type == stream::EVENT_TEXT_START)
            .expect("text start event");
        let partial = text_start.partial.as_ref().expect("text start snapshot");
        assert!(
            partial
                .content
                .iter()
                .all(|block| !matches!(block, llm::ContentBlock::ToolCall(_))),
            "a published text snapshot must not mutate with future tool calls"
        );
        assert_ne!(partial.stop_reason, stream::STOP_TOOL_USE);
    }

    #[test]
    fn omni_prompt_tools_preserves_truncation_and_reports_tool_use_when_complete() {
        let truncated = omni_prompt_tools_sse(
            "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
            "length",
        );
        let complete = omni_prompt_tools_sse(
            "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
            "stop",
        );
        let (base_url, requests, server) = test_server(vec![
            http_response(200, &truncated),
            http_response(200, &complete),
        ]);
        let mut request_model = model(omni_prompt_tools::API_OMNI_PROMPT_TOOLS, base_url);
        request_model.provider = "omni".to_owned();

        let truncated = factory(0)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("truncated prompt-protocol response");
        let complete = factory(0)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("complete prompt-protocol response");
        let _ = requests.recv().expect("first OmniRoute request");
        let _ = requests.recv().expect("second OmniRoute request");
        server.join().expect("test server finishes");

        assert_eq!(truncated.stop_reason, stream::STOP_LENGTH);
        assert_eq!(complete.stop_reason, stream::STOP_TOOL_USE);
        assert!(matches!(
            truncated.content.first(),
            Some(llm::ContentBlock::ToolCall(_))
        ));
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
        let aperture_root = std::env::temp_dir().join(format!(
            "goshcoder-catalog-isolated-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("current time")
                .as_nanos()
        ));
        let catalog = Arc::new(
            catalog::Catalog::with_environment(
                None,
                Arc::new(move |name| environment.get(name).cloned()),
            )
            .expect("catalog")
            .with_aperture_paths(
                aperture_root.join("aperture.json"),
                aperture_root.join("aperture-cache.json"),
            ),
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
    fn catalog_responder_rewrites_dedicated_aperture_models_and_provenance_headers() {
        let body = concat!(
            "data: {\"id\":\"chat_1\",\"model\":\"route-model\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chat_1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![http_response(200, body)]);
        let root = std::env::temp_dir().join(format!(
            "goshcoder-aperture-request-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("current time")
                .as_nanos()
        ));
        let configuration = aperture::Config {
            base_url: base_url.clone(),
            onboarding_done: Some(true),
            dedicated: Some(aperture::DedicatedConfig {
                enabled: Some(true),
                ..aperture::DedicatedConfig::default()
            }),
            ..aperture::Config::default()
        };
        let cached_model = llm::Model {
            id: "openai/route-model".to_owned(),
            name: "Routed model".to_owned(),
            api: API_OPENAI_COMPLETIONS.to_owned(),
            provider: aperture::DEDICATED_PROVIDER_ID.to_owned(),
            base_url: format!("{base_url}/v1"),
            input: vec!["text".to_owned()],
            context_window: 128_000,
            max_tokens: 4_096,
            ..llm::Model::default()
        };
        let cache = aperture::Cache {
            catalog_key: aperture::build_catalog_key(
                &aperture::gateway_url(&configuration.base_url),
                &configuration.resolve(),
            ),
            models: vec![aperture::CachedModel {
                model: cached_model,
                raw_compat: None,
            }],
            ..aperture::Cache::default()
        };
        let configuration_path = root.join("extensions").join("aperture.json");
        let cache_path = root.join("extensions").join("aperture-cache.json");
        aperture::save_config(&configuration_path, &configuration).expect("save Aperture config");
        aperture::save_cache(&cache_path, &cache).expect("save Aperture cache");
        let catalog = Arc::new(
            catalog::Catalog::with_environment(None, Arc::new(|_| None))
                .expect("catalog")
                .with_aperture_paths(configuration_path, cache_path),
        );
        let request_model = catalog
            .provider(aperture::DEDICATED_PROVIDER_ID)
            .expect("Aperture provider")
            .models()
            .into_iter()
            .next()
            .expect("Aperture model");

        let response = factory(0).catalog_assistant_responder(catalog)(
            &request_model,
            &text_context(),
            options(agent::CancellationToken::default()),
        )
        .expect("Aperture response");
        let request = requests.recv().expect("captured Aperture request");
        server.join().expect("Aperture server finishes");
        let payload = serde_json::from_slice::<Value>(&request.body).expect("request JSON");

        assert_eq!(request.target, "/v1/chat/completions");
        assert_eq!(payload["model"], "openai/route-model");
        assert_eq!(
            request.headers.get("referer").map(String::as_str),
            Some(aperture::APERTURE_REFERER)
        );
        assert_eq!(
            request.headers.get("x-session-id").map(String::as_str),
            Some("session-1")
        );
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer -")
        );
        assert_eq!(response.stop_reason, stream::STOP_STOP);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn routed_aperture_restart_errors_are_retried() {
        let success = concat!(
            "data: {\"id\":\"chat_1\",\"model\":\"route-model\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"chat_1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let (base_url, requests, server) = test_server(vec![
            http_response_with_headers(400, "Aperture is restarting", &[("retry-after-ms", "0")]),
            http_response(200, success),
        ]);
        let mut request_model = model(API_OPENAI_COMPLETIONS, format!("{base_url}/v1"));
        request_model
            .headers
            .insert("Referer".to_owned(), aperture::APERTURE_REFERER.to_owned());

        let response = factory(1)
            .respond(
                &request_model,
                &text_context(),
                options(agent::CancellationToken::default()),
            )
            .expect("restart retry succeeds");
        let first = requests.recv().expect("first routed request");
        let second = requests.recv().expect("retried routed request");
        server.join().expect("Aperture server finishes");

        assert_eq!(first.target, "/v1/chat/completions");
        assert_eq!(second.target, "/v1/chat/completions");
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
                request_timeout: Some(Duration::from_millis(50)),
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
}
