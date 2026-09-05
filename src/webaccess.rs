//! Native cited web search for the Rust/Ratatui runtime.
//!
//! This is a Rust adaptation of `internal/webaccess`'s Go implementation and
//! of the `pi-web-access` provider shapes it supports.  It intentionally owns
//! only search transport and normalization: callers supply OpenAI/Codex
//! credentials through [`ResolveOpenAIAuth`], then register [`Service::tool`]
//! with the public [`crate::agent`] runtime.
//!
//! The module reads the existing `web-search.json` path from
//! [`crate::config::web_search_path`] by default.  Its default JSON mode accepts
//! the established camelCase document and snake_case aliases so a partially
//! migrated Rust configuration remains usable.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fmt,
    fs::File,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::{Method, blocking::Client};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use time::{Duration as TimeDuration, OffsetDateTime, format_description::well_known::Rfc3339};
use url::Url;

use crate::{agent, config, llm};

pub const DEFAULT_OPENAI_RESPONSES_URL: &str = "https://api.openai.com/v1/responses";
pub const DEFAULT_CODEX_RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
pub const DEFAULT_EXA_MCP_URL: &str = "https://mcp.exa.ai/mcp";
pub const DEFAULT_EXA_SEARCH_URL: &str = "https://api.exa.ai/search";
pub const DEFAULT_KAGI_SEARCH_URL: &str = "https://kagi.com/api/v1/search";
pub const DEFAULT_SEARCH_MODEL: &str = "gpt-5.6-terra";

/// Maximum size of `web-search.json`.
pub const MAX_CONFIG_BYTES: usize = 1 << 20;
/// Maximum size accepted from a search provider, including an SSE stream.
pub const MAX_RESPONSE_BODY_BYTES: usize = 8 << 20;
/// Maximum text returned to a model for one `web_search` invocation.
pub const MAX_TOOL_OUTPUT_BYTES: usize = 50 << 10;
/// Total timeout for one blocking provider request.
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(60);

const MAX_QUERIES: usize = 8;
const MAX_QUERY_BYTES: usize = 2_000;
const MAX_RESULTS: usize = 20;
const DEFAULT_RESULTS: usize = 5;
const MAX_SNIPPET_BYTES: usize = 1_000;
const MAX_DISPLAY_SNIPPET_BYTES: usize = 500;
const USER_AGENT: &str = "goshcoder-web-access/1.0 (+https://github.com/goshitsarch-eng/GoshCoder)";

/// A non-secret error returned by the web-search service.
///
/// Provider credentials are intentionally never retained in this value.  HTTP
/// response text is redacted against all request credential/header values
/// before it can become an error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebAccessError {
    message: String,
}

impl WebAccessError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// The safe, model-visible diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WebAccessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for WebAccessError {}

/// Result type used by this module.  It is named separately from [`Result`],
/// which is the public cited-search result shape retained for Go/pi parity.
pub type WebAccessResult<T> = std::result::Result<T, WebAccessError>;

/// Authentication headers can override or suppress the OpenAI defaults.
///
/// `None` removes a case-insensitive matching default header, matching the
/// catalog's public auth-header convention without making this module depend
/// on the catalog implementation.
pub type ProviderHeaders = BTreeMap<String, Option<String>>;

/// OpenAI or Codex credentials supplied by the active runtime/session.
///
/// This intentionally does not implement `Debug` or `Display`: all fields
/// except `provider` can contain credentials or credential-derived values.
#[derive(Clone, Default)]
pub struct OpenAIAuth {
    /// Usually `openai` or `openai-codex`.
    pub provider: String,
    /// API key or OAuth access token.  Never log or render this value.
    pub api_key: String,
    /// Search-capable model selected by the credential resolver.
    pub model: String,
    /// Optional request-header overrides from the authenticated provider.
    pub headers: ProviderHeaders,
}

/// Resolves current OpenAI/Codex credentials for a single search.
///
/// The resolver is called for every search, allowing a surrounding session to
/// refresh OAuth tokens without coupling this module to any non-public runtime
/// state.  Resolver failures are deliberately reduced to a safe generic
/// diagnostic; an arbitrary resolver error must not be able to leak a token.
pub type ResolveOpenAIAuth = Arc<
    dyn Fn(&agent::CancellationToken) -> std::result::Result<Option<OpenAIAuth>, String>
        + Send
        + Sync
        + 'static,
>;

/// Process-environment lookup used for credentials configured as `$NAME`.
pub type EnvironmentLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync + 'static>;

/// One normalized, cited web result.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Result {
    pub title: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub snippet: String,
}

/// A normalized response from a selected search route.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Response {
    pub provider: String,
    pub query: String,
    pub answer: String,
    pub results: Vec<Result>,
}

/// Controls one search.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    pub num_results: usize,
    pub recency_filter: String,
    pub domain_filter: Vec<String>,
    pub include_content: bool,
    pub provider: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            num_results: DEFAULT_RESULTS,
            recency_filter: String::new(),
            domain_filter: Vec::new(),
            include_content: false,
            provider: String::new(),
        }
    }
}

/// Chooses whether legacy Rust snake_case JSON aliases are accepted.
///
/// [`JsonCompatibility::Compatible`] is the default because the persistent
/// document is shared with the prior Go runtime, whose canonical keys are
/// camelCase.  Strict mode remains useful for callers that want to reject
/// accidental noncanonical configuration before rollout.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum JsonCompatibility {
    /// Accept only established pi/Go camelCase parameter and config names.
    Strict,
    /// Accept camelCase plus documented snake_case aliases.
    #[default]
    Compatible,
}

/// Provider endpoint overrides, primarily useful for self-hosted proxies and
/// deterministic tests.  The configured OpenAI Responses URL remains a
/// per-user `web-search.json` override, as in the Go implementation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Endpoints {
    pub openai_responses_url: String,
    pub codex_responses_url: String,
    pub exa_mcp_url: String,
    pub exa_search_url: String,
    pub kagi_search_url: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            openai_responses_url: DEFAULT_OPENAI_RESPONSES_URL.to_owned(),
            codex_responses_url: DEFAULT_CODEX_RESPONSES_URL.to_owned(),
            exa_mcp_url: DEFAULT_EXA_MCP_URL.to_owned(),
            exa_search_url: DEFAULT_EXA_SEARCH_URL.to_owned(),
            kagi_search_url: DEFAULT_KAGI_SEARCH_URL.to_owned(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConfigFile {
    #[serde(default)]
    provider: String,
    #[serde(default, alias = "search_provider")]
    search_provider: String,
    #[serde(default, alias = "openai_api_key")]
    openai_api_key: String,
    #[serde(default, alias = "openai_responses_url")]
    openai_responses_url: String,
    #[serde(default, alias = "openai_search_model")]
    openai_search_model: String,
    #[serde(default, alias = "exa_api_key")]
    exa_api_key: String,
    #[serde(default, alias = "kagi_api_key")]
    kagi_api_key: String,
}

/// A service that routes one query to OpenAI/Codex, Exa, or Kagi.
///
/// No background task is started.  Configuration is re-read for each query so
/// edits to `web-search.json` take effect without restarting Ratatui.
#[derive(Clone)]
pub struct Service {
    config_path: PathBuf,
    client: Client,
    resolve_openai: Option<ResolveOpenAIAuth>,
    environment: EnvironmentLookup,
    endpoints: Endpoints,
    json_compatibility: JsonCompatibility,
}

impl Service {
    /// Constructs a service using the stable agent configuration path.
    pub fn new(resolve_openai: Option<ResolveOpenAIAuth>) -> WebAccessResult<Self> {
        Self::with_config_path(config::web_search_path(), resolve_openai)
    }

    /// Constructs a service with a caller-selected config file.
    pub fn with_config_path(
        config_path: impl Into<PathBuf>,
        resolve_openai: Option<ResolveOpenAIAuth>,
    ) -> WebAccessResult<Self> {
        let client = Client::builder()
            .timeout(SEARCH_TIMEOUT)
            .build()
            .map_err(|_| WebAccessError::new("initialize web-search HTTP client"))?;
        Ok(Self {
            config_path: config_path.into(),
            client,
            resolve_openai,
            environment: Arc::new(|name| env::var(name).ok()),
            endpoints: Endpoints::default(),
            json_compatibility: JsonCompatibility::default(),
        })
    }

    /// Replaces endpoint defaults.  Endpoints are validated before requests
    /// are sent so a malformed configured URL never reaches reqwest.
    pub fn set_endpoints(&mut self, endpoints: Endpoints) {
        self.endpoints = endpoints;
    }

    pub fn endpoints(&self) -> &Endpoints {
        &self.endpoints
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn json_compatibility(&self) -> JsonCompatibility {
        self.json_compatibility
    }

    pub fn set_json_compatibility(&mut self, compatibility: JsonCompatibility) {
        self.json_compatibility = compatibility;
    }

    /// Configures JSON compatibility while constructing an embedded service.
    pub fn with_json_compatibility(mut self, compatibility: JsonCompatibility) -> Self {
        self.set_json_compatibility(compatibility);
        self
    }

    /// Replaces environment lookup for embedding applications and tests.
    pub fn set_environment_lookup(&mut self, environment: EnvironmentLookup) {
        self.environment = environment;
    }

    /// Returns the native `web_search` agent tool.
    pub fn tool(&self) -> agent::Tool {
        let service = self.clone();
        agent::Tool::new(
            "web_search",
            "Web Search",
            "Search the web with cited sources. Supports OpenAI/Codex authentication, \
             zero-config Exa, and Kagi. For broad research prefer 2-4 varied queries. \
             The configured provider is used when provider is omitted or auto.",
            web_search_schema(),
            move |cancellation, _call_id, parameters, on_update| {
                service
                    .execute_tool(&cancellation, &parameters, &on_update)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Executes a model-facing tool invocation.  This is public so a runtime
    /// that wraps agent tools can retain the same validation/progress behavior.
    pub fn execute_tool(
        &self,
        cancellation: &agent::CancellationToken,
        parameters: &BTreeMap<String, Value>,
        on_update: &agent::ToolUpdate,
    ) -> WebAccessResult<agent::ToolResult> {
        let queries = query_params(parameters, self.json_compatibility)?;
        let options = options_from_params(parameters, self.json_compatibility)?;
        let mut responses = Vec::with_capacity(queries.len());

        for (index, query) in queries.iter().enumerate() {
            check_cancelled(cancellation)?;
            on_update(agent::ToolResult {
                content: vec![llm::ContentBlock::text(format!(
                    "Searching {}/{}: {:?}...",
                    index + 1,
                    queries.len(),
                    query
                ))],
                ..agent::ToolResult::default()
            });
            responses.push(self.search_with_cancellation(cancellation, query, options.clone())?);
        }

        let mut output = format_responses(&responses);
        if output.len() > MAX_TOOL_OUTPUT_BYTES {
            output = format!(
                "{}\n\n[web search output truncated at 50 KiB]",
                clip_utf8(&output, MAX_TOOL_OUTPUT_BYTES)
            );
        }
        let details = serde_json::to_value(&responses)
            .map_err(|_| WebAccessError::new("serialize web-search result details"))?;
        Ok(agent::ToolResult {
            content: vec![llm::ContentBlock::text(output)],
            details: Some(details),
            ..agent::ToolResult::default()
        })
    }

    /// Executes one query without a caller-owned cancellation token.
    pub fn search(&self, query: impl AsRef<str>, options: Options) -> WebAccessResult<Response> {
        self.search_with_cancellation(&agent::CancellationToken::default(), query, options)
    }

    /// Executes one query with the public agent cancellation token.
    pub fn search_with_cancellation(
        &self,
        cancellation: &agent::CancellationToken,
        query: impl AsRef<str>,
        mut options: Options,
    ) -> WebAccessResult<Response> {
        check_cancelled(cancellation)?;
        let query = query.as_ref().trim();
        if query.is_empty() {
            return Err(WebAccessError::new("query is required"));
        }
        if query.len() > MAX_QUERY_BYTES {
            return Err(WebAccessError::new(format!(
                "query exceeds {MAX_QUERY_BYTES} characters"
            )));
        }
        validate_options(&options)?;
        options.num_results = normalize_count(options.num_results);

        let configuration = self.load_config()?;
        let provider = options.provider.trim().to_ascii_lowercase();
        let provider = if provider.is_empty() || provider == "auto" {
            first_nonempty(&[&configuration.search_provider, &configuration.provider])
                .trim()
                .to_ascii_lowercase()
        } else {
            provider
        };
        if provider.is_empty() || provider == "auto" {
            return self.search_auto(cancellation, &configuration, query, &options);
        }
        self.search_provider(cancellation, &configuration, &provider, query, &options)
    }

    fn search_auto(
        &self,
        cancellation: &agent::CancellationToken,
        configuration: &ConfigFile,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Response> {
        let mut diagnostics = Vec::new();
        // pi-web-access's defaults prefer OpenAI only in its ordinary shape;
        // Exa better represents result count and recency filtering.
        if options.recency_filter.is_empty() && options.num_results == DEFAULT_RESULTS {
            match self.search_openai(cancellation, configuration, query, options) {
                Ok(Some(response)) => return Ok(response),
                Ok(None) => {}
                Err(error) => {
                    check_cancelled(cancellation)?;
                    diagnostics.push(format!("OpenAI: {error}"));
                }
            }
        }

        match self.search_exa(cancellation, configuration, query, options) {
            Ok(response) => Ok(response),
            Err(error) => {
                check_cancelled(cancellation)?;
                diagnostics.push(format!("Exa: {error}"));
                Err(WebAccessError::new(format!(
                    "automatic web search failed:\n  - {}",
                    diagnostics.join("\n  - ")
                )))
            }
        }
    }

    fn search_provider(
        &self,
        cancellation: &agent::CancellationToken,
        configuration: &ConfigFile,
        provider: &str,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Response> {
        match provider {
            "openai" => match self.search_openai(cancellation, configuration, query, options)? {
                Some(response) => Ok(response),
                None => Err(WebAccessError::new(format!(
                    "OpenAI web search is unavailable; log in to openai-codex, set \
                     OPENAI_API_KEY, or configure openaiApiKey in {}",
                    self.config_path.display()
                ))),
            },
            "exa" => self.search_exa(cancellation, configuration, query, options),
            "kagi" => self.search_kagi(cancellation, configuration, query, options),
            other => Err(WebAccessError::new(format!(
                "unsupported web search provider {other:?} (use auto, openai, exa, or kagi)"
            ))),
        }
    }

    fn search_exa(
        &self,
        cancellation: &agent::CancellationToken,
        configuration: &ConfigFile,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Response> {
        let api_key = self.credential(&configuration.exa_api_key, "EXA_API_KEY");
        if api_key.is_empty() {
            self.search_exa_mcp(cancellation, query, options)
        } else {
            self.search_exa_api(cancellation, &api_key, query, options)
        }
    }

    fn search_exa_mcp(
        &self,
        cancellation: &agent::CancellationToken,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Response> {
        let mut endpoint = parse_http_url(&self.endpoints.exa_mcp_url, "Exa MCP URL")?;
        endpoint
            .query_pairs_mut()
            .append_pair("tools", "web_search_exa");
        let constrained = constrained_query(query, options);
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {
                    "query": constrained,
                    "numResults": options.num_results,
                }
            }
        });
        let headers = BTreeMap::from([
            (
                "Accept".to_owned(),
                "application/json, text/event-stream".to_owned(),
            ),
            ("x-exa-source".to_owned(), "pi-web-access".to_owned()),
        ]);
        let raw = self.do_json(
            cancellation,
            Method::POST,
            endpoint.as_str(),
            &body,
            &headers,
            &[],
        )?;
        let text = parse_exa_mcp_envelope(&raw)?;
        let results = parse_exa_text_results(&text, options.num_results);
        if results.is_empty() {
            return Err(WebAccessError::new("no parseable results from Exa MCP"));
        }
        Ok(Response {
            provider: "exa".to_owned(),
            query: query.to_owned(),
            answer: answer_from_results(&results),
            results,
        })
    }

    fn search_exa_api(
        &self,
        cancellation: &agent::CancellationToken,
        api_key: &str,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Response> {
        let mut body = Map::from_iter([
            ("query".to_owned(), Value::String(query.to_owned())),
            ("type".to_owned(), Value::String("auto".to_owned())),
            ("numResults".to_owned(), Value::from(options.num_results)),
            ("contents".to_owned(), json!({"highlights": true})),
        ]);
        if options.include_content {
            body.insert(
                "contents".to_owned(),
                json!({"highlights": true, "text": true}),
            );
        }
        apply_exa_filters(&mut body, options);
        let headers = BTreeMap::from([
            ("x-api-key".to_owned(), api_key.to_owned()),
            ("x-exa-integration".to_owned(), "pi-web-access".to_owned()),
        ]);
        let secrets = secrets_for(&headers, Some(api_key));
        let raw = self.do_json(
            cancellation,
            Method::POST,
            &self.endpoints.exa_search_url,
            &Value::Object(body),
            &headers,
            &secrets,
        )?;
        let payload: Value = serde_json::from_slice(&raw)
            .map_err(|_| WebAccessError::new("invalid JSON from the Exa API"))?;
        let mut results = Vec::new();
        for item in payload
            .get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(record) = item.as_object() else {
                continue;
            };
            let snippet = record
                .get("highlights")
                .and_then(Value::as_array)
                .map(|highlights| {
                    highlights
                        .iter()
                        .filter_map(Value::as_str)
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .filter(|highlights| !highlights.trim().is_empty())
                .unwrap_or_else(|| string_value(record.get("text")));
            if let Some(result) = normalize_result(
                string_value(record.get("title")),
                string_value(record.get("url")),
                snippet,
                MAX_SNIPPET_BYTES,
            ) {
                results.push(result);
                if results.len() >= options.num_results {
                    break;
                }
            }
        }
        if results.is_empty() {
            return Err(WebAccessError::new("no results from the Exa API"));
        }
        Ok(Response {
            provider: "exa".to_owned(),
            query: query.to_owned(),
            answer: answer_from_results(&results),
            results,
        })
    }

    fn search_kagi(
        &self,
        cancellation: &agent::CancellationToken,
        configuration: &ConfigFile,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Response> {
        let api_key = self.credential(&configuration.kagi_api_key, "KAGI_API_KEY");
        if api_key.is_empty() {
            return Err(WebAccessError::new(format!(
                "no Kagi API key; set KAGI_API_KEY or kagiApiKey in {}",
                self.config_path.display()
            )));
        }
        let headers = BTreeMap::from([
            ("Authorization".to_owned(), format!("Bearer {api_key}")),
            ("Accept".to_owned(), "application/json".to_owned()),
        ]);
        let secrets = secrets_for(&headers, Some(&api_key));
        let raw = self.do_json(
            cancellation,
            Method::POST,
            &self.endpoints.kagi_search_url,
            &json!({"query": query, "limit": options.num_results}),
            &headers,
            &secrets,
        )?;
        let payload: Value = serde_json::from_slice(&raw)
            .map_err(|_| WebAccessError::new("invalid JSON from the Kagi API"))?;
        if let Some(error) = kagi_error(&payload).filter(|error| !error.is_empty()) {
            return Err(WebAccessError::new(format!(
                "error from the Kagi API: {}",
                redact_many(&error, &secrets)
            )));
        }
        let mut results = Vec::new();
        if let Some(data) = kagi_data(&payload) {
            append_kagi_results(data, &mut results, options.num_results);
        }
        if results.is_empty() {
            return Err(WebAccessError::new("no results from the Kagi API"));
        }
        Ok(Response {
            provider: "kagi".to_owned(),
            query: query.to_owned(),
            answer: answer_from_results(&results),
            results,
        })
    }

    /// Returns `None` when no usable OpenAI/Codex credential is available.
    fn search_openai(
        &self,
        cancellation: &agent::CancellationToken,
        configuration: &ConfigFile,
        query: &str,
        options: &Options,
    ) -> WebAccessResult<Option<Response>> {
        let Some(mut auth) = self.resolve_openai(cancellation, configuration)? else {
            return Ok(None);
        };
        auth.api_key = auth.api_key.trim().to_owned();
        if auth.api_key.is_empty() {
            return Ok(None);
        }

        let codex = auth.provider.trim().eq_ignore_ascii_case("openai-codex")
            || is_codex_jwt(&auth.api_key);
        let endpoint = if codex {
            self.endpoints.codex_responses_url.clone()
        } else if configuration.openai_responses_url.trim().is_empty() {
            self.endpoints.openai_responses_url.clone()
        } else {
            parse_http_url(
                configuration.openai_responses_url.trim(),
                "openaiResponsesUrl",
            )?
            .to_string()
        };
        let model = first_nonempty(&[
            &configuration.openai_search_model,
            &auth.model,
            DEFAULT_SEARCH_MODEL,
        ])
        .to_owned();
        let mut headers = BTreeMap::from([
            (
                "Authorization".to_owned(),
                format!("Bearer {}", auth.api_key),
            ),
            (
                "OpenAI-Beta".to_owned(),
                "responses=experimental".to_owned(),
            ),
        ]);
        for (name, value) in &auth.headers {
            match value {
                Some(value) => insert_header_fold(&mut headers, name, value),
                None => delete_header_fold(&mut headers, name),
            }
        }
        if codex {
            if let Some(account_id) = codex_account_id(&auth.api_key) {
                insert_header_fold(&mut headers, "chatgpt-account-id", &account_id);
            }
            insert_header_fold(&mut headers, "originator", "goshcoder");
        }
        let body = json!({
            "model": model,
            "instructions": build_openai_instructions(options),
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            }],
            "tools": [build_openai_web_tool(options)],
            "include": ["web_search_call.action.sources"],
            "store": false,
            "stream": true,
            "tool_choice": "required",
            "parallel_tool_calls": true,
        });
        let secrets = secrets_for(&headers, Some(&auth.api_key));
        let raw = self
            .do_json(
                cancellation,
                Method::POST,
                &endpoint,
                &body,
                &headers,
                &secrets,
            )
            .map_err(|error| WebAccessError::new(format!("OpenAI web search: {error}")))?;
        let output = parse_openai_output(&raw)?;
        let answer = openai_answer(&output);
        let results = openai_results(&output, options.num_results);
        if answer.is_empty() && results.is_empty() {
            return Err(WebAccessError::new(
                "OpenAI web search returned no answer or sources",
            ));
        }
        Ok(Some(Response {
            provider: "openai".to_owned(),
            query: query.to_owned(),
            answer,
            results,
        }))
    }

    fn resolve_openai(
        &self,
        cancellation: &agent::CancellationToken,
        configuration: &ConfigFile,
    ) -> WebAccessResult<Option<OpenAIAuth>> {
        check_cancelled(cancellation)?;
        let mut resolver_failed = false;
        if let Some(resolve_openai) = &self.resolve_openai {
            match resolve_openai(cancellation) {
                Ok(Some(auth)) if !auth.api_key.trim().is_empty() => return Ok(Some(auth)),
                Ok(_) => {}
                // A resolver owns arbitrary external/provider errors.  Do not
                // repeat their text because it could contain a bearer token.
                Err(_) => resolver_failed = true,
            }
        }
        let api_key = self.credential(&configuration.openai_api_key, "OPENAI_API_KEY");
        if !api_key.is_empty() {
            return Ok(Some(OpenAIAuth {
                provider: "openai".to_owned(),
                api_key,
                model: first_nonempty(&[&configuration.openai_search_model, DEFAULT_SEARCH_MODEL])
                    .to_owned(),
                headers: ProviderHeaders::new(),
            }));
        }
        if resolver_failed {
            return Err(WebAccessError::new("resolve OpenAI credentials failed"));
        }
        Ok(None)
    }

    fn load_config(&self) -> WebAccessResult<ConfigFile> {
        if self.config_path.as_os_str().is_empty() {
            return Ok(ConfigFile::default());
        }
        let mut file = match File::open(&self.config_path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(ConfigFile::default());
            }
            Err(_) => {
                return Err(WebAccessError::new(format!(
                    "read web search config: {}",
                    self.config_path.display()
                )));
            }
        };
        let contents = read_bounded(&mut file, MAX_CONFIG_BYTES)
            .map_err(|_| WebAccessError::new("read web search config"))?;
        if contents.len() > MAX_CONFIG_BYTES {
            return Err(WebAccessError::new("web search config exceeds 1 MiB"));
        }
        if contents.iter().all(u8::is_ascii_whitespace) {
            return Ok(ConfigFile::default());
        }
        let document: Value = serde_json::from_slice(&contents).map_err(|_| {
            WebAccessError::new(format!(
                "parse {}: invalid JSON",
                self.config_path.display()
            ))
        })?;
        let Some(object) = document.as_object() else {
            return Err(WebAccessError::new(format!(
                "parse {}: web search config must be a JSON object",
                self.config_path.display()
            )));
        };
        if self.json_compatibility == JsonCompatibility::Strict {
            for alias in [
                "search_provider",
                "openai_api_key",
                "openai_responses_url",
                "openai_search_model",
                "exa_api_key",
                "kagi_api_key",
            ] {
                if object.contains_key(alias) {
                    return Err(WebAccessError::new(format!(
                        "parse {}: {alias:?} requires compatible JSON mode",
                        self.config_path.display()
                    )));
                }
            }
        }
        serde_json::from_value(document).map_err(|_| {
            WebAccessError::new(format!(
                "parse {}: invalid web search configuration",
                self.config_path.display()
            ))
        })
    }

    fn credential(&self, configured: &str, environment_name: &str) -> String {
        let mut value = configured.trim().to_owned();
        if value.starts_with("${") && value.ends_with('}') {
            value = self.environment(&value[2..value.len().saturating_sub(1)]);
        } else if let Some(name) = value.strip_prefix('$') {
            value = self.environment(name);
        }
        if value.trim().is_empty() {
            value = self.environment(environment_name);
        }
        value.trim().to_owned()
    }

    fn environment(&self, name: &str) -> String {
        (self.environment)(name).unwrap_or_default()
    }

    fn do_json(
        &self,
        cancellation: &agent::CancellationToken,
        method: Method,
        endpoint: &str,
        body: &Value,
        headers: &BTreeMap<String, String>,
        secrets: &[String],
    ) -> WebAccessResult<Vec<u8>> {
        check_cancelled(cancellation)?;
        let endpoint = parse_http_url(endpoint, "search provider URL")?;
        let encoded = serde_json::to_vec(body)
            .map_err(|_| WebAccessError::new("serialize web-search request"))?;
        let mut request = self
            .client
            .request(method, endpoint)
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .body(encoded);
        for (name, value) in headers {
            request = request.header(name, value);
        }
        let response = request
            .send()
            // reqwest can include the endpoint in its error.  Do not expose a
            // configured endpoint's query string, which may itself be secret.
            .map_err(|_| WebAccessError::new("web search request failed"))?;
        let status = response.status().as_u16();
        let raw = read_response_bounded(response, cancellation)?;
        if !(200..300).contains(&status) {
            let message = compact_text(&redact_many(&String::from_utf8_lossy(&raw), secrets), 300);
            return Err(WebAccessError::new(if message.is_empty() {
                format!("HTTP {status}")
            } else {
                format!("HTTP {status}: {message}")
            }));
        }
        Ok(raw)
    }
}

/// JSON Schema exposed by the native `web_search` tool.
pub fn web_search_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Single search query. Prefer queries for broad research."
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple varied search queries."
            },
            "numResults": {
                "type": "number",
                "description": "Results per query (default 5, max 20)"
            },
            "includeContent": {
                "type": "boolean",
                "description": "Ask providers that support it to include fuller result text"
            },
            "recencyFilter": {
                "type": "string",
                "enum": ["day", "week", "month", "year"],
                "description": "Filter by recency"
            },
            "domainFilter": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Limit to domains; prefix a domain with - to exclude it"
            },
            "provider": {
                "type": "string",
                "enum": ["auto", "openai", "exa", "kagi"],
                "description": "Override the configured search provider"
            },
            "workflow": {
                "type": "string",
                "enum": ["none", "summary-review", "auto-summary"],
                "description": "Accepted for pi-web-access compatibility; native GoshCoder returns results directly"
            }
        }
    })
}

/// Parses the Exa MCP JSON-RPC result from either a direct JSON response or
/// SSE `data:` records.
pub fn parse_exa_mcp_envelope(raw: &[u8]) -> WebAccessResult<String> {
    let mut candidates = Vec::new();
    for line in String::from_utf8_lossy(raw).lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if !data.is_empty() {
                candidates.push(data.as_bytes().to_vec());
            }
        }
    }
    candidates.push(raw.to_vec());

    for candidate in candidates {
        let Ok(envelope) = serde_json::from_slice::<Value>(&candidate) else {
            continue;
        };
        if let Some(error) = envelope.get("error").and_then(Value::as_object) {
            let code = error
                .get("code")
                .map(Value::to_string)
                .unwrap_or_else(|| "unknown".to_owned())
                .trim_matches('"')
                .to_owned();
            let message = string_value(error.get("message"));
            return Err(WebAccessError::new(format!(
                "error {code} from Exa MCP: {message}"
            )));
        }
        let Some(result) = envelope.get("result").and_then(Value::as_object) else {
            continue;
        };
        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        for content in result
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(content) = content.as_object() else {
                continue;
            };
            if content.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            let text = string_value(content.get("text"));
            if text.is_empty() {
                continue;
            }
            if is_error {
                return Err(WebAccessError::new(compact_text(&text, 500)));
            }
            return Ok(text);
        }
    }
    Err(WebAccessError::new("empty response from Exa MCP"))
}

/// Parses Exa MCP's text representation into safe normalized citations.
pub fn parse_exa_text_results(text: &str, limit: usize) -> Vec<Result> {
    if limit == 0 {
        return Vec::new();
    }
    let starts = title_line_starts(text);
    let mut results = Vec::with_capacity(starts.len().min(limit));
    for (index, start) in starts.iter().copied().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(text.len());
        let block = text[start..end].trim();
        let lines = block.lines().collect::<Vec<_>>();
        let Some(first_line) = lines.first() else {
            continue;
        };
        let title = first_line
            .strip_prefix("Title:")
            .unwrap_or(first_line)
            .trim()
            .to_owned();
        let mut url = String::new();
        let mut content_at = None;
        for (line_index, line) in lines.iter().enumerate().skip(1) {
            let line = line.trim();
            if let Some(value) = line.strip_prefix("URL:") {
                url = value.trim().to_owned();
            } else if line.starts_with("Text:") || line.starts_with("Highlights:") {
                content_at = Some(line_index + 1);
            }
        }
        let snippet = content_at
            .filter(|index| *index < lines.len())
            .map(|index| lines[index..].join(" "))
            .unwrap_or_default();
        if let Some(result) = normalize_result(title, url, snippet, MAX_SNIPPET_BYTES) {
            results.push(result);
            if results.len() >= limit {
                break;
            }
        }
    }
    results
}

/// Parses direct JSON or the Responses API's SSE output records.
pub fn parse_openai_output(raw: &[u8]) -> WebAccessResult<Vec<Value>> {
    let trimmed = trim_ascii(raw);
    if trimmed.starts_with(b"{")
        && let Ok(Value::Object(response)) = serde_json::from_slice::<Value>(trimmed)
        && let Some(output) = response.get("output").and_then(Value::as_array)
    {
        return Ok(output.clone());
    }

    let mut output = Vec::new();
    let mut completed = Vec::new();
    for line in String::from_utf8_lossy(raw).lines() {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("response.output_item.done") => {
                if let Some(item) = event.get("item") {
                    output.push(item.clone());
                }
            }
            Some("response.done" | "response.completed") => {
                if let Some(items) = event
                    .get("response")
                    .and_then(Value::as_object)
                    .and_then(|response| response.get("output"))
                    .and_then(Value::as_array)
                    .filter(|items| !items.is_empty())
                {
                    completed = items.clone();
                }
            }
            _ => {}
        }
    }
    if !completed.is_empty() {
        return Ok(completed);
    }
    if !output.is_empty() {
        return Ok(output);
    }
    Err(WebAccessError::new(
        "OpenAI API returned no parseable response output",
    ))
}

/// Normalizes one OpenAI citation URL and removes OpenAI's tracking parameter.
///
/// Only absolute HTTP(S) URLs are returned.  That restriction prevents
/// provider text from creating a `javascript:` or relative Markdown citation.
pub fn normalize_citation_url(value: &str) -> Option<String> {
    let value = value.trim();
    let mut parsed = Url::parse(value).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return None;
    }
    let has_openai_tracking = parsed
        .query_pairs()
        .any(|(name, value)| name == "utm_source" && value == "openai");
    if !has_openai_tracking {
        return Some(value.to_owned());
    }
    let retained = parsed
        .query_pairs()
        .filter(|(name, value)| !(name == "utm_source" && value == "openai"))
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    if retained.is_empty() {
        // `query_pairs_mut().clear()` preserves an empty `?` delimiter.  Go's
        // `url.Values.Encode()` clears RawQuery entirely, so remove it too.
        parsed.set_query(None);
    } else {
        let mut query = parsed.query_pairs_mut();
        query.clear();
        query.extend_pairs(
            retained
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
        );
    }
    Some(parsed.to_string())
}

/// Parses the account id encoded in an OpenAI Codex JWT's payload.
///
/// Invalid/malformed tokens are intentionally indistinguishable from
/// non-Codex tokens and return `None`.
pub fn codex_account_id(token: &str) -> Option<String> {
    let mut parts = token.split('.');
    let _header = parts.next()?;
    let payload = parts.next()?;
    if parts.next().is_none() || parts.next().is_some() {
        return None;
    }
    let payload = URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('=').as_bytes())
        .ok()?;
    let claims: Value = serde_json::from_slice(&payload).ok()?;
    claims
        .get("https://api.openai.com/auth")
        .and_then(Value::as_object)
        .and_then(|authentication| authentication.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|account_id| !account_id.is_empty())
        .map(str::to_owned)
}

/// Replaces every exact occurrence of one known credential.
pub fn redact(value: &str, secret: &str) -> String {
    if secret.is_empty() {
        value.to_owned()
    } else {
        value.replace(secret, "[REDACTED]")
    }
}

fn query_params(
    parameters: &BTreeMap<String, Value>,
    compatibility: JsonCompatibility,
) -> WebAccessResult<Vec<String>> {
    let mut queries = Vec::new();
    if let Some(value) = parameter(parameters, "queries", &[], compatibility) {
        let values = value
            .as_array()
            .ok_or_else(|| WebAccessError::new("queries must be an array of strings"))?;
        for value in values {
            let query = value
                .as_str()
                .ok_or_else(|| WebAccessError::new("queries must contain only strings"))?
                .trim();
            if !query.is_empty() {
                queries.push(query.to_owned());
            }
        }
    }
    if queries.is_empty()
        && let Some(value) = parameter(parameters, "query", &[], compatibility)
    {
        let query = value
            .as_str()
            .ok_or_else(|| WebAccessError::new("query must be a string"))?
            .trim();
        if !query.is_empty() {
            queries.push(query.to_owned());
        }
    }
    if queries.is_empty() {
        return Err(WebAccessError::new(
            "no query provided; use query or queries",
        ));
    }
    if queries.len() > MAX_QUERIES {
        return Err(WebAccessError::new(format!(
            "at most {MAX_QUERIES} queries may be searched at once"
        )));
    }
    for (index, query) in queries.iter().enumerate() {
        if query.len() > MAX_QUERY_BYTES {
            return Err(WebAccessError::new(format!(
                "query {} exceeds {MAX_QUERY_BYTES} characters",
                index + 1
            )));
        }
    }
    Ok(queries)
}

fn options_from_params(
    parameters: &BTreeMap<String, Value>,
    compatibility: JsonCompatibility,
) -> WebAccessResult<Options> {
    if compatibility == JsonCompatibility::Strict {
        for alias in [
            "num_results",
            "recency_filter",
            "include_content",
            "domain_filter",
        ] {
            if parameters.contains_key(alias) {
                return Err(WebAccessError::new(format!(
                    "{alias:?} requires compatible JSON mode"
                )));
            }
        }
    }
    let mut options = Options::default();
    if let Some(value) = parameter(parameters, "numResults", &["num_results"], compatibility) {
        options.num_results = parse_result_count(value)?;
    }
    if let Some(value) = parameter(
        parameters,
        "recencyFilter",
        &["recency_filter"],
        compatibility,
    ) {
        options.recency_filter = required_string(value, "recencyFilter")?.trim().to_owned();
    }
    if let Some(value) = parameter(parameters, "provider", &[], compatibility) {
        options.provider = required_string(value, "provider")?.trim().to_owned();
    }
    if let Some(value) = parameter(
        parameters,
        "includeContent",
        &["include_content"],
        compatibility,
    ) {
        options.include_content = value
            .as_bool()
            .ok_or_else(|| WebAccessError::new("includeContent must be a boolean"))?;
    }
    if let Some(value) = parameter(
        parameters,
        "domainFilter",
        &["domain_filter"],
        compatibility,
    ) {
        let values = value
            .as_array()
            .ok_or_else(|| WebAccessError::new("domainFilter must be an array of strings"))?;
        for value in values {
            let domain = value
                .as_str()
                .ok_or_else(|| WebAccessError::new("domainFilter must contain only strings"))?
                .trim();
            if !domain.is_empty() {
                options.domain_filter.push(domain.to_owned());
            }
        }
    }
    if let Some(value) = parameter(parameters, "workflow", &[], compatibility) {
        let workflow = required_string(value, "workflow")?;
        if !matches!(workflow, "none" | "summary-review" | "auto-summary") {
            return Err(WebAccessError::new(format!(
                "invalid workflow {workflow:?}"
            )));
        }
    }
    validate_options(&options)?;
    options.num_results = normalize_count(options.num_results);
    Ok(options)
}

fn parameter<'a>(
    parameters: &'a BTreeMap<String, Value>,
    canonical: &str,
    aliases: &[&str],
    compatibility: JsonCompatibility,
) -> Option<&'a Value> {
    parameters.get(canonical).or_else(|| {
        (compatibility == JsonCompatibility::Compatible)
            .then(|| aliases.iter().find_map(|alias| parameters.get(*alias)))
            .flatten()
    })
}

fn required_string<'a>(value: &'a Value, name: &str) -> WebAccessResult<&'a str> {
    value
        .as_str()
        .ok_or_else(|| WebAccessError::new(format!("{name} must be a string")))
}

fn parse_result_count(value: &Value) -> WebAccessResult<usize> {
    let number = if let Some(value) = value.as_i64() {
        value as f64
    } else if let Some(value) = value.as_u64() {
        value as f64
    } else {
        value
            .as_f64()
            .ok_or_else(|| WebAccessError::new("numResults must be a number"))?
    };
    if !number.is_finite() || number.fract() != 0.0 || number.abs() > i64::MAX as f64 {
        return Err(WebAccessError::new("numResults must be an integer"));
    }
    if number <= 0.0 {
        return Ok(DEFAULT_RESULTS);
    }
    Ok((number as u64).min(MAX_RESULTS as u64) as usize)
}

fn validate_options(options: &Options) -> WebAccessResult<()> {
    if !options.recency_filter.is_empty()
        && !matches!(
            options.recency_filter.as_str(),
            "day" | "week" | "month" | "year"
        )
    {
        return Err(WebAccessError::new(format!(
            "invalid recencyFilter {:?}",
            options.recency_filter
        )));
    }
    let provider = options.provider.trim().to_ascii_lowercase();
    if !provider.is_empty() && !matches!(provider.as_str(), "auto" | "openai" | "exa" | "kagi") {
        return Err(WebAccessError::new(format!(
            "unsupported web search provider {:?} (use auto, openai, exa, or kagi)",
            options.provider
        )));
    }
    Ok(())
}

fn normalize_count(value: usize) -> usize {
    if value == 0 {
        DEFAULT_RESULTS
    } else {
        value.min(MAX_RESULTS)
    }
}

fn apply_exa_filters(body: &mut Map<String, Value>, options: &Options) {
    let mut included = Vec::new();
    let mut excluded = Vec::new();
    for raw in &options.domain_filter {
        let (domain, blocked) = normalize_domain(raw);
        if domain.is_empty() {
            continue;
        }
        if blocked {
            excluded.push(domain);
        } else {
            included.push(domain);
        }
    }
    if !included.is_empty() {
        body.insert("includeDomains".to_owned(), json!(included));
    }
    if !excluded.is_empty() {
        body.insert("excludeDomains".to_owned(), json!(excluded));
    }
    if !options.recency_filter.is_empty() {
        body.insert(
            "startPublishedDate".to_owned(),
            Value::String(recency_start(&options.recency_filter)),
        );
    }
}

fn constrained_query(query: &str, options: &Options) -> String {
    let mut parts = vec![query.to_owned()];
    for raw in &options.domain_filter {
        let (domain, excluded) = normalize_domain(raw);
        if domain.is_empty() {
            continue;
        }
        parts.push(if excluded {
            format!("-site:{domain}")
        } else {
            format!("site:{domain}")
        });
    }
    if !options.recency_filter.is_empty() {
        parts.push(format!("past {}", options.recency_filter));
    }
    parts.join(" ")
}

fn normalize_domain(raw: &str) -> (String, bool) {
    let mut value = raw.trim().to_ascii_lowercase();
    let excluded = value.starts_with('-');
    if excluded {
        value = value.trim_start_matches('-').trim().to_owned();
    }
    if value.is_empty() {
        return (String::new(), excluded);
    }
    if !value.contains("://") {
        value = format!("https://{value}");
    }
    let Ok(parsed) = Url::parse(&value) else {
        return (String::new(), excluded);
    };
    let Some(host) = parsed.host_str() else {
        return (String::new(), excluded);
    };
    (host.trim_matches('.').to_owned(), excluded)
}

fn recency_start(filter: &str) -> String {
    let days = match filter {
        "day" => 1,
        "week" => 7,
        "month" => 30,
        "year" => 365,
        _ => 0,
    };
    let now = OffsetDateTime::now_utc()
        .replace_nanosecond(0)
        .unwrap_or_else(|_| OffsetDateTime::now_utc());
    (now - TimeDuration::days(days))
        .format(&Rfc3339)
        .unwrap_or_default()
}

fn build_openai_instructions(options: &Options) -> String {
    let mut lines = vec![
        "Search the web and return a concise answer grounded only in the web results.".to_owned(),
        "Include clickable source citations in the response text when possible.".to_owned(),
    ];
    if !options.recency_filter.is_empty() {
        lines.push(format!(
            "Prefer sources from the past {}.",
            options.recency_filter
        ));
    }
    lines.push(format!(
        "Prefer around {} distinct sources.",
        options.num_results
    ));
    let (allowed, blocked) = split_domains(&options.domain_filter);
    if !allowed.is_empty() {
        lines.push(format!("Only use sources from: {}.", allowed.join(", ")));
    }
    if !blocked.is_empty() {
        lines.push(format!("Do not use sources from: {}.", blocked.join(", ")));
    }
    lines.join(" ")
}

fn build_openai_web_tool(options: &Options) -> Value {
    let (allowed, blocked) = split_domains(&options.domain_filter);
    let mut tool = Map::from_iter([("type".to_owned(), Value::String("web_search".to_owned()))]);
    if !allowed.is_empty() || !blocked.is_empty() {
        let mut filters = Map::new();
        if !allowed.is_empty() {
            filters.insert("allowed_domains".to_owned(), json!(allowed));
        }
        if !blocked.is_empty() {
            filters.insert("blocked_domains".to_owned(), json!(blocked));
        }
        tool.insert("filters".to_owned(), Value::Object(filters));
    }
    Value::Object(tool)
}

fn split_domains(domains: &[String]) -> (Vec<String>, Vec<String>) {
    let mut allowed = Vec::new();
    let mut blocked = Vec::new();
    for raw in domains {
        let (domain, excluded) = normalize_domain(raw);
        if domain.is_empty() {
            continue;
        }
        if excluded {
            blocked.push(domain);
        } else {
            allowed.push(domain);
        }
    }
    (allowed, blocked)
}

fn openai_answer(output: &[Value]) -> String {
    let mut parts = Vec::new();
    for item in output {
        let Some(item) = item.as_object() else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        for content in item
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(text) = content
                .as_object()
                .and_then(|content| content.get("text"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
            {
                parts.push(text.to_owned());
            }
        }
    }
    parts.join("\n").trim().to_owned()
}

fn openai_results(output: &[Value], limit: usize) -> Vec<Result> {
    let mut results = Vec::new();
    let mut seen = BTreeSet::new();
    let mut add = |url: &Value, title: &Value, snippet: &Value| {
        let url = string_value(Some(url));
        let title = string_value(Some(title));
        let snippet = string_value(Some(snippet));
        let Some(url) = normalize_citation_url(&url) else {
            return;
        };
        if !seen.insert(url.clone()) || results.len() >= limit {
            return;
        }
        results.push(Result {
            title,
            url,
            snippet: compact_text(&snippet, MAX_DISPLAY_SNIPPET_BYTES),
        });
    };

    for item in output {
        let Some(item) = item.as_object() else {
            continue;
        };
        if item.get("type").and_then(Value::as_str) == Some("message") {
            for content in item
                .get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                for annotation in content
                    .as_object()
                    .and_then(|content| content.get("annotations"))
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let Some(annotation) = annotation.as_object() else {
                        continue;
                    };
                    if annotation.get("type").and_then(Value::as_str) == Some("url_citation") {
                        add(
                            annotation.get("url").unwrap_or(&Value::Null),
                            annotation.get("title").unwrap_or(&Value::Null),
                            &Value::Null,
                        );
                    }
                }
            }
        }
        if item.get("type").and_then(Value::as_str) == Some("web_search_call") {
            let mut groups = Vec::new();
            if let Some(sources) = item.get("sources") {
                groups.push(sources);
            }
            if let Some(results) = item.get("results") {
                groups.push(results);
            }
            if let Some(sources) = item
                .get("action")
                .and_then(Value::as_object)
                .and_then(|action| action.get("sources"))
            {
                groups.push(sources);
            }
            for group in groups {
                for source in group.as_array().into_iter().flatten() {
                    let Some(source) = source.as_object() else {
                        continue;
                    };
                    let url = first_value(source, &["url", "source_website_url"]);
                    let title = first_value(source, &["title", "caption"]);
                    add(url, title, &Value::Null);
                }
            }
        }
    }
    results
}

fn first_value<'a>(record: &'a Map<String, Value>, names: &[&str]) -> &'a Value {
    names
        .iter()
        .filter_map(|name| record.get(*name))
        .find(|value| !string_value(Some(value)).is_empty())
        .unwrap_or(&Value::Null)
}

fn kagi_data(payload: &Value) -> Option<&Value> {
    let record = payload.as_object()?;
    let data = record.get("data")?;
    data.as_object()
        .and_then(|data| data.get("search"))
        .or(Some(data))
}

fn kagi_error(payload: &Value) -> Option<String> {
    let record = payload.as_object()?;
    let value = record.get("errors").or_else(|| record.get("error"))?;
    let mut messages = Vec::new();
    append_error_messages(value, &mut messages);
    Some(messages.join("; "))
}

fn append_error_messages(value: &Value, messages: &mut Vec<String>) {
    match value {
        Value::Array(values) => {
            for value in values {
                append_error_messages(value, messages);
            }
        }
        Value::Object(record) => {
            for field in ["message", "msg", "code"] {
                let message = string_value(record.get(field));
                if !message.is_empty() {
                    messages.push(message);
                    return;
                }
            }
        }
        Value::String(value) if !value.trim().is_empty() => messages.push(value.trim().to_owned()),
        _ => {}
    }
}

fn append_kagi_results(value: &Value, results: &mut Vec<Result>, limit: usize) {
    if results.len() >= limit {
        return;
    }
    match value {
        Value::Array(values) => {
            for value in values {
                append_kagi_results(value, results, limit);
                if results.len() >= limit {
                    return;
                }
            }
        }
        Value::Object(record) => {
            let url = first_string(record, &["url", "href", "link"]);
            if url.is_empty() {
                return;
            }
            let mut title = first_string(record, &["title", "name"]);
            if title.is_empty() {
                title = url.clone();
            }
            let snippet = first_string(
                record,
                &[
                    "snippet",
                    "description",
                    "summary",
                    "content",
                    "markdown",
                    "text",
                ],
            );
            if let Some(result) = normalize_result(title, url, snippet, MAX_SNIPPET_BYTES) {
                results.push(result);
            }
        }
        _ => {}
    }
}

fn first_string(record: &Map<String, Value>, names: &[&str]) -> String {
    for name in names {
        let value = string_value(record.get(*name));
        if !value.is_empty() {
            return value;
        }
    }
    String::new()
}

fn normalize_result(
    title: String,
    url: String,
    snippet: String,
    snippet_limit: usize,
) -> Option<Result> {
    let url = normalize_citation_url(&url)?;
    Some(Result {
        title: title.trim().to_owned(),
        url,
        snippet: compact_text(&snippet, snippet_limit),
    })
}

/// Renders normalized provider results as model-readable Markdown.
pub fn format_responses(responses: &[Response]) -> String {
    let mut sections = Vec::new();
    for response in responses {
        let mut section = String::new();
        if responses.len() > 1 {
            section.push_str(&format!("## {}\n\n", response.query));
        }
        section.push_str(&format!("Provider: {}\n", response.provider));
        if !response.answer.trim().is_empty() {
            section.push('\n');
            section.push_str(response.answer.trim());
            section.push('\n');
        }
        if !response.results.is_empty() {
            section.push_str("\nSources:\n");
            for (index, result) in response.results.iter().enumerate() {
                section.push_str(&format!(
                    "{}. [{}]({})",
                    index + 1,
                    fallback_title(result),
                    result.url
                ));
                let snippet = compact_text(&result.snippet, MAX_DISPLAY_SNIPPET_BYTES);
                if !snippet.is_empty() {
                    section.push_str(&format!(" — {snippet}"));
                }
                section.push('\n');
            }
        }
        sections.push(section.trim().to_owned());
    }
    sections.join("\n\n---\n\n")
}

fn answer_from_results(results: &[Result]) -> String {
    results
        .iter()
        .map(|result| {
            if result.snippet.is_empty() {
                format!("Source: {} ({})", fallback_title(result), result.url)
            } else {
                format!(
                    "{}\nSource: {} ({})",
                    result.snippet,
                    fallback_title(result),
                    result.url
                )
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn fallback_title(result: &Result) -> String {
    if result.title.trim().is_empty() {
        result.url.clone()
    } else {
        result.title.trim().replace(']', "\\]")
    }
}

fn compact_text(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= limit {
        compact
    } else {
        format!("{}...", clip_utf8(&compact, limit.saturating_sub(3)))
    }
}

fn clip_utf8(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn first_nonempty<'a>(values: &[&'a str]) -> &'a str {
    values
        .iter()
        .copied()
        .find(|value| !value.trim().is_empty())
        .unwrap_or_default()
}

fn string_value(value: Option<&Value>) -> String {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_owned()
}

fn title_line_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']).starts_with("Title:") {
            starts.push(offset);
        }
        offset += line.len();
    }
    // `split_inclusive` skips an empty final segment but not a non-newline
    // final line.  It also yields no segment for an empty input.
    if offset < text.len() && text[offset..].starts_with("Title:") {
        starts.push(offset);
    }
    starts
}

fn parse_http_url(value: &str, label: &str) -> WebAccessResult<Url> {
    let parsed = Url::parse(value.trim()).map_err(|_| {
        WebAccessError::new(format!("invalid {label}: must be an absolute HTTP(S) URL"))
    })?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err(WebAccessError::new(format!(
            "invalid {label}: must be an absolute HTTP(S) URL"
        )));
    }
    Ok(parsed)
}

fn is_codex_jwt(token: &str) -> bool {
    codex_account_id(token).is_some()
}

fn delete_header_fold(headers: &mut BTreeMap<String, String>, target: &str) {
    headers.retain(|name, _| !name.eq_ignore_ascii_case(target));
}

fn insert_header_fold(headers: &mut BTreeMap<String, String>, name: &str, value: &str) {
    delete_header_fold(headers, name);
    headers.insert(name.to_owned(), value.to_owned());
}

fn secrets_for(headers: &BTreeMap<String, String>, api_key: Option<&str>) -> Vec<String> {
    let mut secrets = BTreeSet::new();
    if let Some(api_key) = api_key.filter(|key| !key.is_empty()) {
        secrets.insert(api_key.to_owned());
    }
    for value in headers.values().filter(|value| !value.is_empty()) {
        secrets.insert(value.clone());
    }
    let mut secrets = secrets.into_iter().collect::<Vec<_>>();
    secrets.sort_unstable_by_key(|value| std::cmp::Reverse(value.len()));
    secrets
}

fn redact_many(value: &str, secrets: &[String]) -> String {
    secrets
        .iter()
        .fold(value.to_owned(), |value, secret| redact(&value, secret))
}

fn check_cancelled(cancellation: &agent::CancellationToken) -> WebAccessResult<()> {
    if cancellation.is_cancelled() {
        Err(WebAccessError::new("web search was cancelled"))
    } else {
        Ok(())
    }
}

fn read_bounded(reader: &mut impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let mut contents = Vec::with_capacity(limit.min(8_192));
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut contents)?;
    Ok(contents)
}

fn read_response_bounded(
    response: reqwest::blocking::Response,
    cancellation: &agent::CancellationToken,
) -> WebAccessResult<Vec<u8>> {
    let mut reader = response.take((MAX_RESPONSE_BODY_BYTES + 1) as u64);
    let mut contents = Vec::with_capacity(8_192);
    let mut buffer = [0_u8; 8_192];
    loop {
        check_cancelled(cancellation)?;
        let read = reader
            .read(&mut buffer)
            .map_err(|_| WebAccessError::new("read search provider response failed"))?;
        if read == 0 {
            break;
        }
        contents.extend_from_slice(&buffer[..read]);
        if contents.len() > MAX_RESPONSE_BODY_BYTES {
            return Err(WebAccessError::new(
                "search provider response exceeds 8 MiB",
            ));
        }
    }
    check_cancelled(cancellation)?;
    Ok(contents)
}

fn trim_ascii(mut value: &[u8]) -> &[u8] {
    while value.first().is_some_and(u8::is_ascii_whitespace) {
        value = &value[1..];
    }
    while value.last().is_some_and(u8::is_ascii_whitespace) {
        value = &value[..value.len() - 1];
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Cursor, Write},
        net::{Shutdown, TcpListener, TcpStream},
        sync::{
            Mutex,
            atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        },
        thread,
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        path: String,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
    }

    impl RecordedRequest {
        fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .get(&name.to_ascii_lowercase())
                .map(String::as_str)
        }

        fn json(&self) -> Value {
            serde_json::from_slice(&self.body).expect("request JSON")
        }
    }

    #[derive(Clone)]
    struct TestResponse {
        status: u16,
        content_type: &'static str,
        body: String,
    }

    impl TestResponse {
        fn ok_json(body: impl Into<String>) -> Self {
            Self {
                status: 200,
                content_type: "application/json",
                body: body.into(),
            }
        }

        fn sse(body: impl Into<String>) -> Self {
            Self {
                status: 200,
                content_type: "text/event-stream",
                body: body.into(),
            }
        }
    }

    struct TestServer {
        address: String,
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
        stop: Arc<AtomicBool>,
        worker: Option<thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn new(handler: impl Fn(&RecordedRequest) -> TestResponse + Send + Sync + 'static) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
            listener
                .set_nonblocking(true)
                .expect("nonblocking listener");
            let address = listener.local_addr().expect("local address").to_string();
            let requests = Arc::new(Mutex::new(Vec::new()));
            let request_log = requests.clone();
            let stop = Arc::new(AtomicBool::new(false));
            let worker_stop = stop.clone();
            let handler = Arc::new(handler);
            let worker = thread::spawn(move || {
                while !worker_stop.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            let Some(request) = read_request(&mut stream) else {
                                continue;
                            };
                            request_log
                                .lock()
                                .unwrap_or_else(|error| error.into_inner())
                                .push(request.clone());
                            let response = handler(&request);
                            write_response(&mut stream, response);
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(_) => break,
                    }
                }
            });
            Self {
                address,
                requests,
                stop,
                worker: Some(worker),
            }
        }

        fn url(&self) -> String {
            format!("http://{}", self.address)
        }

        fn requests(&self) -> Vec<RecordedRequest> {
            self.requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Ok(mut stream) = TcpStream::connect(&self.address) {
                let _ = stream.write_all(b"GET / HTTP/1.1\r\nContent-Length: 0\r\n\r\n");
                let _ = stream.shutdown(Shutdown::Both);
            }
            if let Some(worker) = self.worker.take() {
                worker.join().expect("join test server");
            }
        }
    }

    fn read_request(stream: &mut TcpStream) -> Option<RecordedRequest> {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("request read timeout");
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 4_096];
        let header_end = loop {
            let read = stream.read(&mut buffer).ok()?;
            if read == 0 {
                return None;
            }
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                break end + 4;
            }
        };
        let header = std::str::from_utf8(&bytes[..header_end]).ok()?;
        let mut lines = header.split("\r\n");
        let request_line = lines.next()?;
        let path = request_line.split_whitespace().nth(1)?.to_owned();
        let mut headers = BTreeMap::new();
        for line in lines {
            let Some((name, value)) = line.split_once(':') else {
                continue;
            };
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
        let content_length = headers
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_default();
        while bytes.len() < header_end.saturating_add(content_length) {
            let read = stream.read(&mut buffer).ok()?;
            if read == 0 {
                return None;
            }
            bytes.extend_from_slice(&buffer[..read]);
        }
        Some(RecordedRequest {
            path,
            headers,
            body: bytes[header_end..header_end + content_length].to_vec(),
        })
    }

    fn write_response(stream: &mut TcpStream, response: TestResponse) {
        let status_text = if response.status < 300 { "OK" } else { "Error" };
        let head = format!(
            "HTTP/1.1 {} {status_text}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            response.status,
            response.content_type,
            response.body.len()
        );
        stream
            .write_all(head.as_bytes())
            .expect("write test headers");
        stream
            .write_all(response.body.as_bytes())
            .expect("write test body");
        stream.flush().expect("flush test response");
    }

    fn test_service(resolve_openai: Option<ResolveOpenAIAuth>) -> Service {
        Service::with_config_path(PathBuf::new(), resolve_openai).expect("create service")
    }

    fn endpoints_with(update: impl FnOnce(&mut Endpoints)) -> Endpoints {
        let mut endpoints = Endpoints::default();
        update(&mut endpoints);
        endpoints
    }

    fn scratch_file(label: &str, contents: &str) -> PathBuf {
        let path = env::temp_dir().join(format!(
            "goshcoder-webaccess-{label}-{}-{}",
            std::process::id(),
            TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, contents).expect("write test config");
        path
    }

    fn test_codex_jwt(account_id: &str) -> String {
        let payload = serde_json::to_vec(&json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id}
        }))
        .expect("encode JWT payload");
        format!("header.{}.signature", URL_SAFE_NO_PAD.encode(payload))
    }

    fn tool_text(result: &agent::ToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(llm::ContentBlock::plain_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn exa_mcp_provides_zero_config_search() {
        let server = TestServer::new(|_| {
            TestResponse::sse(
                "event: message\n\
                 data: {\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"Title: Go\\nURL: https://go.dev/\\nHighlights:\\nThe Go programming language\\n---\\n\\nTitle: Docs\\nURL: https://go.dev/doc/\\nText: Official documentation\"}]},\"jsonrpc\":\"2.0\",\"id\":1}\n",
            )
        });
        let mut service = test_service(None);
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.exa_mcp_url = server.url();
        }));
        let response = service
            .search(
                "Go language",
                Options {
                    provider: "exa".to_owned(),
                    num_results: 2,
                    domain_filter: vec!["go.dev".to_owned(), "-old.go.dev".to_owned()],
                    recency_filter: "year".to_owned(),
                    ..Options::default()
                },
            )
            .expect("Exa MCP search");

        assert_eq!(response.provider, "exa");
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].url, "https://go.dev/");
        let request = server.requests().pop().expect("MCP request");
        let url = Url::parse(&format!("{}{}", server.url(), request.path)).expect("request URL");
        assert_eq!(
            url.query_pairs().find(|(name, _)| name == "tools"),
            Some(("tools".into(), "web_search_exa".into()))
        );
        assert_eq!(request.header("x-exa-source"), Some("pi-web-access"));
        let request_body = request.json();
        let query = request_body["params"]["arguments"]["query"]
            .as_str()
            .expect("MCP query");
        for expected in ["site:go.dev", "-site:old.go.dev", "past year"] {
            assert!(
                query.contains(expected),
                "{query:?} should contain {expected:?}"
            );
        }
    }

    #[test]
    fn kagi_search_uses_config_credential_and_normalizes_results() {
        let server = TestServer::new(|_| {
            TestResponse::ok_json(
                json!({
                    "data": [
                        {"title": "Kagi result", "url": "https://example.com/one", "snippet": "First result"},
                        {"name": "Second", "href": "https://example.org/two", "description": "Second result"}
                    ]
                })
                .to_string(),
            )
        });
        let config = scratch_file("kagi", r#"{"provider":"kagi","kagiApiKey":"kagi-secret"}"#);
        let mut service = Service::with_config_path(&config, None).expect("service");
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.kagi_search_url = server.url();
        }));
        let response = service
            .search(
                "current Go release",
                Options {
                    num_results: 2,
                    ..Options::default()
                },
            )
            .expect("Kagi search");
        let _ = std::fs::remove_file(config);

        assert_eq!(response.provider, "kagi");
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[1].title, "Second");
        let request = server.requests().pop().expect("Kagi request");
        assert_eq!(request.header("authorization"), Some("Bearer kagi-secret"));
        assert_eq!(request.json()["query"], "current Go release");
        assert_eq!(request.json()["limit"], 2);
    }

    #[test]
    fn openai_codex_search_reuses_oauth_headers_and_parses_sse() {
        let jwt = test_codex_jwt("acct-search");
        let server = TestServer::new(|_| {
            TestResponse::sse(
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"web_search_call\",\"action\":{\"sources\":[{\"url\":\"https://go.dev/\",\"title\":\"Go\"}]}}}\n\
                 data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Go is a programming language [Go](https://go.dev/).\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://go.dev/?utm_source=openai\",\"title\":\"The Go Programming Language\"}]}]}}\n\
                 data: [DONE]\n",
            )
        });
        let resolver: ResolveOpenAIAuth = Arc::new(move |_| {
            Ok(Some(OpenAIAuth {
                provider: "openai-codex".to_owned(),
                api_key: jwt.clone(),
                model: "gpt-5.6-terra".to_owned(),
                headers: ProviderHeaders::new(),
            }))
        });
        let mut service = test_service(Some(resolver));
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.codex_responses_url = server.url();
        }));
        let response = service
            .search(
                "what is Go",
                Options {
                    provider: "openai".to_owned(),
                    ..Options::default()
                },
            )
            .expect("Codex search");

        assert!(!response.answer.is_empty());
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].url, "https://go.dev/");
        let request = server.requests().pop().expect("Codex request");
        assert_eq!(request.header("chatgpt-account-id"), Some("acct-search"));
        assert_eq!(request.header("originator"), Some("goshcoder"));
        assert_eq!(request.json()["tool_choice"], "required");
    }

    #[test]
    fn auto_falls_back_from_openai_to_exa() {
        let server = TestServer::new(|request| {
            if request.path.starts_with("/openai") {
                TestResponse {
                    status: 503,
                    content_type: "text/plain",
                    body: "temporarily unavailable".to_owned(),
                }
            } else {
                TestResponse::sse(
                    "data: {\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"Title: Fallback\\nURL: https://example.com/\\nHighlights:\\nFallback worked\"}]},\"jsonrpc\":\"2.0\",\"id\":1}\n",
                )
            }
        });
        let resolver: ResolveOpenAIAuth = Arc::new(|_| {
            Ok(Some(OpenAIAuth {
                provider: "openai".to_owned(),
                api_key: "openai-secret".to_owned(),
                ..OpenAIAuth::default()
            }))
        });
        let mut service = test_service(Some(resolver));
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.openai_responses_url = format!("{}/openai", server.url());
            endpoints.exa_mcp_url = format!("{}/exa", server.url());
        }));
        let response = service
            .search("fallback", Options::default())
            .expect("auto fallback");

        assert_eq!(response.provider, "exa");
        assert_eq!(response.results.len(), 1);
        assert_eq!(server.requests().len(), 2);
    }

    #[test]
    fn web_search_tool_accepts_multiple_queries_and_reports_progress() {
        let server = TestServer::new(|request| {
            let query = request.json()["params"]["arguments"]["query"]
                .as_str()
                .expect("query")
                .to_owned();
            TestResponse::ok_json(
                json!({
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": format!("Title: {query}\nURL: https://example.com/{query}\nHighlights:\nanswer")
                        }]
                    }
                })
                .to_string(),
            )
        });
        let mut service = test_service(None);
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.exa_mcp_url = server.url();
        }));
        let tool = service.tool();
        let updates = Arc::new(AtomicUsize::new(0));
        let updates_seen = updates.clone();
        let result = (tool.execute)(
            agent::CancellationToken::default(),
            "call-1".to_owned(),
            BTreeMap::from([
                ("queries".to_owned(), json!(["alpha", "beta"])),
                ("provider".to_owned(), json!("exa")),
                ("numResults".to_owned(), json!(1)),
            ]),
            Arc::new(move |_| {
                updates_seen.fetch_add(1, Ordering::Relaxed);
            }),
        )
        .expect("tool execution");

        assert_eq!(updates.load(Ordering::Relaxed), 2);
        let text = tool_text(&result);
        assert!(text.contains("## alpha"), "{text}");
        assert!(text.contains("## beta"), "{text}");
        assert_eq!(
            result
                .details
                .as_ref()
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn input_validation_limits_queries_and_rejects_invalid_fields() {
        let service = test_service(None);
        let update: agent::ToolUpdate = Arc::new(|_| {});
        let no_query = service.execute_tool(
            &agent::CancellationToken::default(),
            &BTreeMap::new(),
            &update,
        );
        assert!(no_query.unwrap_err().to_string().contains("no query"));

        let too_many = BTreeMap::from([(
            "queries".to_owned(),
            Value::Array(
                (0..=MAX_QUERIES)
                    .map(|index| json!(format!("q{index}")))
                    .collect(),
            ),
        )]);
        assert!(
            service
                .execute_tool(&agent::CancellationToken::default(), &too_many, &update)
                .unwrap_err()
                .to_string()
                .contains("at most")
        );

        let invalid_recency = BTreeMap::from([
            ("query".to_owned(), json!("q")),
            ("recencyFilter".to_owned(), json!("tomorrow")),
        ]);
        assert!(
            service
                .execute_tool(
                    &agent::CancellationToken::default(),
                    &invalid_recency,
                    &update
                )
                .unwrap_err()
                .to_string()
                .contains("invalid recencyFilter")
        );

        let fractional_count = BTreeMap::from([
            ("query".to_owned(), json!("q")),
            ("numResults".to_owned(), json!(1.5)),
        ]);
        assert!(
            service
                .execute_tool(
                    &agent::CancellationToken::default(),
                    &fractional_count,
                    &update
                )
                .unwrap_err()
                .to_string()
                .contains("integer")
        );
    }

    #[test]
    fn compatibility_mode_accepts_snake_case_config_and_parameters() {
        let config = scratch_file(
            "compatible",
            r#"{"search_provider":"kagi","kagi_api_key":"compatible-key"}"#,
        );
        let mut service = Service::with_config_path(&config, None).expect("service");
        let configuration = service.load_config().expect("compatible config");
        assert_eq!(configuration.search_provider, "kagi");
        assert_eq!(configuration.kagi_api_key, "compatible-key");

        let options = options_from_params(
            &BTreeMap::from([
                ("num_results".to_owned(), json!(2)),
                ("recency_filter".to_owned(), json!("week")),
                ("include_content".to_owned(), json!(true)),
                ("domain_filter".to_owned(), json!(["example.com"])),
            ]),
            JsonCompatibility::Compatible,
        )
        .expect("compatible parameters");
        assert_eq!(options.num_results, 2);
        assert_eq!(options.recency_filter, "week");
        assert!(options.include_content);

        service.set_json_compatibility(JsonCompatibility::Strict);
        assert!(
            service
                .load_config()
                .unwrap_err()
                .to_string()
                .contains("compatible JSON mode")
        );
        let _ = std::fs::remove_file(config);
    }

    #[test]
    fn citations_jwt_and_provider_errors_are_normalized_and_redacted() {
        assert_eq!(
            normalize_citation_url("https://go.dev/?q=rust&utm_source=openai"),
            Some("https://go.dev/?q=rust".to_owned())
        );
        assert_eq!(normalize_citation_url("javascript:alert(1)"), None);
        assert_eq!(
            codex_account_id(&test_codex_jwt("acct")),
            Some("acct".to_owned())
        );
        assert_eq!(codex_account_id("not.a.jwt.extra"), None);
        assert_eq!(redact("key=secret", "secret"), "key=[REDACTED]");

        let server = TestServer::new(|_| TestResponse {
            status: 401,
            content_type: "text/plain",
            body: "invalid key secret-token and Bearer secret-token".to_owned(),
        });
        let config = scratch_file(
            "redaction",
            r#"{"provider":"kagi","kagiApiKey":"secret-token"}"#,
        );
        let mut service = Service::with_config_path(&config, None).expect("service");
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.kagi_search_url = server.url();
        }));
        let error = service
            .search("redaction", Options::default())
            .expect_err("Kagi response should fail");
        let _ = std::fs::remove_file(config);
        assert!(error.to_string().contains("[REDACTED]"));
        assert!(!error.to_string().contains("secret-token"));
    }

    #[test]
    fn provider_response_body_is_bounded_before_parsing() {
        let oversized = "x".repeat(MAX_RESPONSE_BODY_BYTES + 1);
        let server = TestServer::new(move |_| TestResponse::ok_json(oversized.clone()));
        let mut service = test_service(None);
        service.set_endpoints(endpoints_with(|endpoints| {
            endpoints.exa_mcp_url = server.url();
        }));

        let error = service
            .search(
                "bounded",
                Options {
                    provider: "exa".to_owned(),
                    ..Options::default()
                },
            )
            .expect_err("oversized response must be rejected");
        assert!(error.to_string().contains("exceeds 8 MiB"));
    }

    #[test]
    fn bounded_reads_cancel_and_preserve_utf8_output_boundaries() {
        let mut input = Cursor::new(vec![b'x'; 17]);
        assert_eq!(
            read_bounded(&mut input, 16).expect("bounded read").len(),
            17
        );
        assert_eq!(clip_utf8("éé", 3), "é");
        assert_eq!(compact_text(" one\n two  three ", 9), "one tw...");

        let cancellation = agent::CancellationToken::default();
        cancellation.cancel();
        assert!(
            check_cancelled(&cancellation)
                .unwrap_err()
                .to_string()
                .contains("cancelled")
        );
    }

    #[test]
    fn parses_direct_openai_and_kagi_error_shapes() {
        let output = parse_openai_output(
            br#"{"output":[{"type":"message","content":[{"text":"answer","annotations":[{"type":"url_citation","url":"https://example.com/","title":"Example"}]}]}]}"#,
        )
        .expect("direct response");
        assert_eq!(openai_answer(&output), "answer");
        assert_eq!(openai_results(&output, 5)[0].title, "Example");

        let payload = json!({"errors": [{"message": "first"}, {"code": "second"}]});
        assert_eq!(kagi_error(&payload), Some("first; second".to_owned()));
    }
}
