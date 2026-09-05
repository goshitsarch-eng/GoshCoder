//! Blocking Streamable HTTP MCP support for Tailscale Aperture connectors.
//!
//! The Aperture configuration module deliberately only models connector
//! metadata. This module owns the live `/v1/mcp` JSON-RPC session used to
//! discover and invoke connector tools. It mirrors the Go implementation's
//! protocol version, deadlines, bounded response handling, SSE compatibility,
//! and best-effort `notifications/initialized` behavior without requiring a
//! runtime integration point.
//!
//! `reqwest::blocking` cannot interrupt a socket read from another thread.
//! Cancellation is therefore checked before dispatch, after headers arrive,
//! between response-body reads, and before decoding. The per-request timeout
//! remains the hard upper bound while a read is blocked in the transport.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::PathBuf,
    process,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use reqwest::{
    blocking::{Client, Response},
    header::{ACCEPT, CONTENT_TYPE},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use url::Url;

use crate::agent;

/// The Streamable HTTP MCP protocol version supported by Aperture.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
/// Maximum time allotted to the `initialize` request.
pub const MCP_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(10);
/// Maximum time allotted to ordinary MCP requests.
pub const MCP_CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// Maximum time spent on the best-effort initialized notification.
pub const MCP_NOTIFICATION_TIMEOUT: Duration = Duration::from_secs(5);
/// Maximum accepted JSON-RPC response size, matching Aperture's Go client.
pub const MAX_RESPONSE_BYTES: usize = 16 << 20;
/// Maximum encoded JSON-RPC request size accepted from an agent tool call.
pub const MAX_REQUEST_BYTES: usize = 8 << 20;
/// Maximum connector output included directly in an agent tool result.
pub const MAX_CONNECTOR_OUTPUT_BYTES: usize = 50 << 10;
/// Maximum accepted connector tool-name size.
pub const MAX_TOOL_NAME_BYTES: usize = 512;
/// Maximum accepted resource URI size.
pub const MAX_RESOURCE_URI_BYTES: usize = 8 << 10;
/// Maximum accepted session-id header size.
pub const MAX_SESSION_ID_BYTES: usize = 4 << 10;

const CLIENT_NAME: &str = "goshcoder-aperture";
const CLIENT_VERSION: &str = "1";
const MAX_JSON_DIAGNOSTIC_BYTES: usize = 4 << 10;
const MAX_INVALID_JSON_PREVIEW_BYTES: usize = 200;
const MAX_OVERFLOW_FILE_ATTEMPTS: u64 = 32;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// The state-only gateway tool representation shared with `crate::aperture`.
///
/// Using this type directly means output from [`McpSession::list_tools`] can
/// be passed into `aperture::build_connector_tool_set` without translation.
pub use crate::aperture::GatewayTool;

/// Go-compatible name for a tool returned by MCP `tools/list`.
pub type McpTool = GatewayTool;

/// One content item returned by MCP `tools/call`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct McpContentItem {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

/// The useful subset of an MCP `tools/call` response.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct McpCallResult {
    #[serde(default, deserialize_with = "null_to_default")]
    pub content: Vec<McpContentItem>,
    #[serde(rename = "isError")]
    pub is_error: bool,
    #[serde(rename = "structuredContent", skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
}

/// One entry from MCP `resources/list`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(rename = "mimeType", skip_serializing_if = "String::is_empty")]
    pub mime_type: String,
}

/// One entry from MCP `resources/templates/list`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct McpResourceTemplate {
    #[serde(rename = "uriTemplate")]
    pub uri_template: String,
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(rename = "mimeType", skip_serializing_if = "String::is_empty")]
    pub mime_type: String,
}

/// One item returned by MCP `resources/read`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct McpResourceContent {
    pub uri: String,
    #[serde(rename = "mimeType", skip_serializing_if = "String::is_empty")]
    pub mime_type: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub text: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub blob: String,
}

/// Explicit request time bounds for an MCP client.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct McpTimeouts {
    pub initialization: Duration,
    pub call: Duration,
    pub initialized_notification: Duration,
}

impl Default for McpTimeouts {
    fn default() -> Self {
        Self {
            initialization: MCP_INITIALIZATION_TIMEOUT,
            call: MCP_CALL_TIMEOUT,
            initialized_notification: MCP_NOTIFICATION_TIMEOUT,
        }
    }
}

impl McpTimeouts {
    fn validate(self) -> Result<()> {
        for (name, value) in [
            ("initialization", self.initialization),
            ("call", self.call),
            ("initialized notification", self.initialized_notification),
        ] {
            if value.is_zero() {
                return Err(McpError::InvalidInput(format!(
                    "MCP {name} timeout must be greater than zero"
                )));
            }
        }
        Ok(())
    }
}

/// Errors returned by the Aperture MCP transport or adapter.
#[derive(Debug)]
pub enum McpError {
    /// A caller supplied an invalid endpoint, tool name, URI, or arguments.
    InvalidInput(String),
    /// The cooperative cancellation token was observed.
    Cancelled,
    /// Creating the blocking HTTP client failed.
    HttpClient(reqwest::Error),
    /// Sending an HTTP request failed.
    Request(reqwest::Error),
    /// Reading a successful HTTP response failed.
    ResponseRead(io::Error),
    /// A successful response exceeded the fixed response cap.
    ResponseTooLarge { limit: usize },
    /// The gateway replied with a non-success HTTP status.
    HttpStatus { status: u16, reason: String },
    /// A JSON-RPC envelope or decoded MCP result was invalid.
    Protocol(String),
    /// The gateway returned a JSON-RPC error object.
    Rpc { code: i64, message: String },
}

impl fmt::Display for McpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) | Self::Protocol(message) => formatter.write_str(message),
            Self::Cancelled => formatter.write_str("MCP request was cancelled"),
            Self::HttpClient(source) => {
                write!(formatter, "failed to create MCP HTTP client: {source}")
            }
            Self::Request(source) => source.fmt(formatter),
            Self::ResponseRead(source) => source.fmt(formatter),
            Self::ResponseTooLarge { limit } => {
                write!(formatter, "MCP response exceeds {limit} bytes")
            }
            Self::HttpStatus { status, reason } => {
                write!(formatter, "MCP request failed: HTTP {status} {reason}")
            }
            Self::Rpc { code, message } => write!(formatter, "MCP error: {message} (code {code})"),
        }
    }
}

impl StdError for McpError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::HttpClient(source) | Self::Request(source) => Some(source),
            Self::ResponseRead(source) => Some(source),
            Self::InvalidInput(_)
            | Self::Cancelled
            | Self::ResponseTooLarge { .. }
            | Self::HttpStatus { .. }
            | Self::Protocol(_)
            | Self::Rpc { .. } => None,
        }
    }
}

/// Result type used by the Aperture MCP transport.
pub type Result<T> = std::result::Result<T, McpError>;

/// Cooperative cancellation abstraction for callers outside the agent loop.
pub trait Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

impl Cancellation for agent::CancellationToken {
    fn is_cancelled(&self) -> bool {
        agent::CancellationToken::is_cancelled(self)
    }
}

/// A cancellation implementation for direct, non-cancellable calls.
#[derive(Debug, Default)]
pub struct NeverCancelled;

impl Cancellation for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// A prepared Aperture Streamable HTTP client.
///
/// Constructing this type performs endpoint validation but does not contact
/// the gateway. Call [`McpClient::initialize`] to obtain a live session.
#[derive(Clone)]
pub struct McpClient {
    endpoint: Url,
    client: Client,
    timeouts: McpTimeouts,
}

impl fmt::Debug for McpClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpClient")
            .field("endpoint", &self.endpoint)
            .field("timeouts", &self.timeouts)
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Builds a client for `base_url`, whose MCP endpoint is `/v1/mcp`.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self> {
        let client = Client::builder()
            .timeout(MCP_CALL_TIMEOUT)
            .build()
            .map_err(McpError::HttpClient)?;
        Self::with_client(base_url, client)
    }

    /// Builds a client around an injected HTTP transport, useful for embedding
    /// applications and controlled tests.
    pub fn with_client(base_url: impl AsRef<str>, client: Client) -> Result<Self> {
        Ok(Self {
            endpoint: mcp_endpoint(base_url.as_ref())?,
            client,
            timeouts: McpTimeouts::default(),
        })
    }

    /// Replaces the request time bounds.
    pub fn with_timeouts(mut self, timeouts: McpTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// The normalized `/v1/mcp` endpoint.
    pub fn endpoint(&self) -> &Url {
        &self.endpoint
    }

    /// Initializes a session without caller-provided cancellation.
    pub fn initialize(&self) -> Result<McpSession> {
        self.initialize_with(&NeverCancelled)
    }

    /// Initializes a session and sends the best-effort initialized notification.
    pub fn initialize_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
    ) -> Result<McpSession> {
        self.timeouts.validate()?;
        ensure_not_cancelled(cancellation)?;

        let reply = self.post_json_rpc(
            cancellation,
            JsonRpcRequest::request(
                1,
                "initialize",
                Some(json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": CLIENT_NAME,
                        "version": CLIENT_VERSION,
                    },
                })),
            ),
            None,
            self.timeouts.initialization,
        )?;

        let session_id = reply.session_id.ok_or_else(|| {
            McpError::Protocol("MCP initialize response missing Mcp-Session-Id header".to_owned())
        })?;
        validate_session_id(&session_id)?;

        let protocol_version = reply.result.get("protocolVersion").and_then(Value::as_str);
        if protocol_version != Some(MCP_PROTOCOL_VERSION) {
            return Err(McpError::Protocol(format!(
                "MCP initialize returned unexpected result: {}",
                json_diagnostic(&reply.result)
            )));
        }

        let session = McpSession {
            client: self.clone(),
            session_id,
            // The initialize request used id 1. The Go client stores 1 and
            // uses atomic Add for later calls, so the first normal call is 2.
            next_id: Arc::new(AtomicU64::new(1)),
        };

        // The notification is intentionally fire-and-forget: a gateway that
        // accepts initialization remains usable if it declines or times out
        // while receiving this optional notification.
        session.send_initialized_notification(cancellation);
        Ok(session)
    }

    fn post_json_rpc<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        body: JsonRpcRequest,
        session_id: Option<&str>,
        timeout: Duration,
    ) -> Result<JsonRpcReply> {
        ensure_not_cancelled(cancellation)?;
        if timeout.is_zero() {
            return Err(McpError::InvalidInput(
                "MCP request timeout must be greater than zero".to_owned(),
            ));
        }

        let encoded = serde_json::to_vec(&body).map_err(|error| {
            McpError::Protocol(format!("failed to serialize MCP request: {error}"))
        })?;
        if encoded.len() > MAX_REQUEST_BYTES {
            return Err(McpError::InvalidInput(format!(
                "MCP request exceeds {MAX_REQUEST_BYTES} bytes"
            )));
        }

        let mut request = self
            .client
            .post(self.endpoint.clone())
            .timeout(timeout)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json, text/event-stream")
            .body(encoded);
        if let Some(session_id) = session_id {
            request = request.header("Mcp-Session-Id", session_id);
        }

        let mut response = match request.send() {
            Ok(response) => response,
            Err(_error) if cancellation.is_cancelled() => return Err(McpError::Cancelled),
            Err(error) => return Err(McpError::Request(error)),
        };
        ensure_not_cancelled(cancellation)?;

        if !response.status().is_success() {
            let status = response.status();
            return Err(McpError::HttpStatus {
                status: status.as_u16(),
                reason: status.canonical_reason().unwrap_or_default().to_owned(),
            });
        }

        let response_session_id = response
            .headers()
            .get("Mcp-Session-Id")
            .and_then(|value| value.to_str().ok())
            .map(ToOwned::to_owned);
        let payload = read_response_body(&mut response, cancellation)?;
        ensure_not_cancelled(cancellation)?;

        Ok(JsonRpcReply {
            result: decode_json_rpc_response(&payload, body.id)?,
            session_id: response_session_id,
        })
    }
}

/// An initialized, cloneable Aperture MCP session.
///
/// Clones share the monotonically increasing JSON-RPC request-id counter but
/// may issue independent blocking HTTP requests concurrently.
#[derive(Clone)]
pub struct McpSession {
    client: McpClient,
    session_id: String,
    next_id: Arc<AtomicU64>,
}

impl fmt::Debug for McpSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpSession")
            .field("endpoint", self.client.endpoint())
            .field("session_id_present", &!self.session_id.is_empty())
            .finish_non_exhaustive()
    }
}

impl McpSession {
    /// Builds a client and initializes an MCP session in one operation.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self> {
        McpClient::new(base_url)?.initialize()
    }

    /// Alias for [`McpSession::new`] that emphasizes the network operation.
    pub fn connect(base_url: impl AsRef<str>) -> Result<Self> {
        Self::new(base_url)
    }

    /// Initializes a session using an injected blocking HTTP client.
    pub fn with_client(base_url: impl AsRef<str>, client: Client) -> Result<Self> {
        McpClient::with_client(base_url, client)?.initialize()
    }

    /// Initializes a session using an injected client and cancellation token.
    pub fn connect_with<C: Cancellation + ?Sized>(
        base_url: impl AsRef<str>,
        cancellation: &C,
    ) -> Result<Self> {
        McpClient::new(base_url)?.initialize_with(cancellation)
    }

    /// Returns the normalized gateway MCP endpoint.
    pub fn endpoint(&self) -> &Url {
        self.client.endpoint()
    }

    /// Returns the opaque session ID issued during initialization.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Lists connector tools without caller-provided cancellation.
    pub fn list_tools(&self) -> Result<Vec<GatewayTool>> {
        self.list_tools_with(&NeverCancelled)
    }

    /// Lists connector tools, checking cancellation while the response is read.
    pub fn list_tools_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
    ) -> Result<Vec<GatewayTool>> {
        let result = self.call_with(cancellation, "tools/list", None)?;
        let decoded: ToolsListResponse = serde_json::from_value(result)
            .map_err(|error| McpError::Protocol(format!("decode MCP tools/list: {error}")))?;
        let tools = decoded.tools.unwrap_or_default();
        for tool in &tools {
            validate_gateway_tool(tool)
                .map_err(|error| McpError::Protocol(format!("decode MCP tools/list: {error}")))?;
        }
        Ok(tools)
    }

    /// Calls a connector tool with a JSON object of arguments.
    pub fn call_tool(&self, name: &str, arguments: Map<String, Value>) -> Result<McpCallResult> {
        self.call_tool_with(&NeverCancelled, name, arguments)
    }

    /// Calls a connector tool, checking cancellation while the response is read.
    pub fn call_tool_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        name: &str,
        arguments: Map<String, Value>,
    ) -> Result<McpCallResult> {
        validate_tool_name(name)?;
        let result = self.call_with(
            cancellation,
            "tools/call",
            Some(json!({"name": name, "arguments": arguments})),
        )?;
        if result.is_null() {
            return Err(McpError::Protocol(format!(
                "MCP tools/call returned empty result for {name}"
            )));
        }
        serde_json::from_value(result)
            .map_err(|error| McpError::Protocol(format!("decode MCP tools/call: {error}")))
    }

    /// Calls a connector tool from the current agent runtime's argument map.
    pub fn call_tool_from_agent_arguments(
        &self,
        name: &str,
        arguments: BTreeMap<String, Value>,
    ) -> Result<McpCallResult> {
        self.call_tool(name, arguments.into_iter().collect::<Map<String, Value>>())
    }

    /// Strictly validates an arbitrary JSON arguments value before calling a
    /// connector tool. Only JSON objects are valid MCP `arguments` values.
    pub fn call_tool_value(&self, name: &str, arguments: &Value) -> Result<McpCallResult> {
        self.call_tool_value_with(&NeverCancelled, name, arguments)
    }

    /// Cancellation-aware form of [`McpSession::call_tool_value`].
    pub fn call_tool_value_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        name: &str,
        arguments: &Value,
    ) -> Result<McpCallResult> {
        let arguments = arguments.as_object().ok_or_else(|| {
            McpError::InvalidInput("MCP tool arguments must be a JSON object".to_owned())
        })?;
        self.call_tool_with(cancellation, name, arguments.clone())
    }

    /// Lists gateway resources without caller-provided cancellation.
    pub fn list_resources(&self) -> Result<Vec<McpResource>> {
        self.list_resources_with(&NeverCancelled)
    }

    /// Lists gateway resources with cooperative cancellation.
    pub fn list_resources_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
    ) -> Result<Vec<McpResource>> {
        let result = self.call_with(cancellation, "resources/list", None)?;
        let decoded: ResourcesListResponse = serde_json::from_value(result)
            .map_err(|error| McpError::Protocol(format!("decode MCP resources/list: {error}")))?;
        Ok(decoded.resources.unwrap_or_default())
    }

    /// Lists gateway resource templates without caller-provided cancellation.
    pub fn list_resource_templates(&self) -> Result<Vec<McpResourceTemplate>> {
        self.list_resource_templates_with(&NeverCancelled)
    }

    /// Lists gateway resource templates with cooperative cancellation.
    pub fn list_resource_templates_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
    ) -> Result<Vec<McpResourceTemplate>> {
        let result = self.call_with(cancellation, "resources/templates/list", None)?;
        let decoded: ResourceTemplatesListResponse =
            serde_json::from_value(result).map_err(|error| {
                McpError::Protocol(format!("decode MCP resources/templates/list: {error}"))
            })?;
        Ok(decoded.resource_templates.unwrap_or_default())
    }

    /// Reads one gateway resource without caller-provided cancellation.
    pub fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>> {
        self.read_resource_with(&NeverCancelled, uri)
    }

    /// Reads one gateway resource with cooperative cancellation.
    pub fn read_resource_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        uri: &str,
    ) -> Result<Vec<McpResourceContent>> {
        validate_resource_uri(uri)?;
        let result = self.call_with(cancellation, "resources/read", Some(json!({"uri": uri})))?;
        if result.is_null() {
            return Err(McpError::Protocol(format!(
                "MCP resources/read returned empty result for {uri}"
            )));
        }
        let decoded: ResourcesReadResponse = serde_json::from_value(result)
            .map_err(|error| McpError::Protocol(format!("decode MCP resources/read: {error}")))?;
        Ok(decoded.contents.unwrap_or_default())
    }

    fn call_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        method: &'static str,
        params: Option<Value>,
    ) -> Result<Value> {
        ensure_not_cancelled(cancellation)?;
        let id = self.next_request_id()?;
        self.client
            .post_json_rpc(
                cancellation,
                JsonRpcRequest::request(id, method, params),
                Some(&self.session_id),
                self.client.timeouts.call,
            )
            .map(|reply| reply.result)
    }

    fn next_request_id(&self) -> Result<u64> {
        self.next_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map(|previous| previous + 1)
            .map_err(|_| McpError::Protocol("MCP JSON-RPC request id overflowed".to_owned()))
    }

    fn send_initialized_notification<C: Cancellation + ?Sized>(&self, cancellation: &C) {
        if cancellation.is_cancelled() {
            return;
        }
        let body = JsonRpcRequest::notification("notifications/initialized");
        let encoded = match serde_json::to_vec(&body) {
            Ok(encoded) if encoded.len() <= MAX_REQUEST_BYTES => encoded,
            _ => return,
        };
        let request = self
            .client
            .client
            .post(self.client.endpoint.clone())
            .timeout(self.client.timeouts.initialized_notification)
            .header(CONTENT_TYPE, "application/json")
            .header(ACCEPT, "application/json, text/event-stream")
            .header("Mcp-Session-Id", &self.session_id)
            .body(encoded);
        let Ok(mut response) = request.send() else {
            return;
        };

        // Match the Go client's bounded drain and intentionally ignore every
        // HTTP/read error from this best-effort notification.
        let mut limited = (&mut response).take(1 << 20);
        let mut buffer = [0_u8; 8 << 10];
        while limited.read(&mut buffer).is_ok_and(|read| read > 0) {}
    }
}

/// Validates and turns one gateway tool into an executable current
/// [`agent::Tool`]. Unknown or malformed input schemas become an empty object
/// schema, matching the original connector adapter's safe fallback.
pub fn gateway_tool_to_agent_tool(session: McpSession, tool: GatewayTool) -> Result<agent::Tool> {
    validate_gateway_tool(&tool)?;
    let tool_name = tool.name;
    let description = {
        let description = tool.description.trim();
        if description.is_empty() {
            format!("Aperture connector tool: {tool_name}")
        } else {
            description.to_owned()
        }
    };
    let parameters = coerce_input_schema(&tool.input_schema);
    let session_for_execute = session;
    let execute_name = tool_name.clone();

    Ok(agent::Tool::new(
        tool_name.clone(),
        tool_name,
        description,
        parameters,
        move |cancellation, _tool_call_id, arguments, on_update| {
            if cancellation.is_cancelled() {
                return Err(McpError::Cancelled.to_string());
            }
            on_update(agent::ToolResult::text(format!(
                "Calling {execute_name}..."
            )));
            let arguments = arguments.into_iter().collect::<Map<String, Value>>();
            let result = session_for_execute
                .call_tool_with(&cancellation, &execute_name, arguments)
                .map_err(|error| error.to_string())?;
            call_result_to_agent_result(&execute_name, result).map_err(|error| error.to_string())
        },
    ))
}

/// Converts a gateway tool list to first-class agent tools.
///
/// Duplicate names are discarded after their first occurrence, preserving
/// gateway order like the Go connector registration path.
pub fn gateway_tools_to_agent_tools<I>(session: McpSession, tools: I) -> Result<Vec<agent::Tool>>
where
    I: IntoIterator<Item = GatewayTool>,
{
    let mut names = BTreeSet::new();
    let mut agent_tools = Vec::new();
    for tool in tools {
        validate_gateway_tool(&tool)?;
        if names.insert(tool.name.clone()) {
            agent_tools.push(gateway_tool_to_agent_tool(session.clone(), tool)?);
        }
    }
    Ok(agent_tools)
}

/// Converts a tool call result to the current agent result shape.
///
/// Text is capped at 50 KiB. When possible, the complete output is retained
/// in a user-private temporary file, as the Go connector proxy does.
pub fn call_result_to_agent_result(
    tool_name: &str,
    result: McpCallResult,
) -> Result<agent::ToolResult> {
    validate_tool_name(tool_name)?;
    let full_text = result
        .content
        .iter()
        .filter_map(|item| (!item.text.is_empty()).then_some(item.text.as_str()))
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut output = truncate_connector_output(&full_text);
    if output.is_empty() {
        output = "(no text output)".to_owned();
    }
    if result.is_error {
        return Err(McpError::Protocol(output));
    }
    Ok(agent::ToolResult {
        content: vec![crate::llm::ContentBlock::text(output)],
        details: Some(json!({"toolName": tool_name})),
        ..agent::ToolResult::default()
    })
}

/// Retains only recognizable JSON object schemas for model-facing tools.
pub fn coerce_input_schema(schema: &Value) -> Value {
    if schema
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind == "object")
        && schema.is_object()
    {
        return schema.clone();
    }
    json!({"type": "object", "properties": {}})
}

/// Validates the base URL and derives its `/v1/mcp` endpoint.
pub fn mcp_endpoint(base_url: &str) -> Result<Url> {
    let base_url = base_url.trim();
    if base_url.is_empty() {
        return Err(McpError::InvalidInput(
            "Aperture MCP base URL cannot be empty".to_owned(),
        ));
    }
    if base_url.len() > MAX_RESOURCE_URI_BYTES {
        return Err(McpError::InvalidInput(format!(
            "Aperture MCP base URL exceeds {MAX_RESOURCE_URI_BYTES} bytes"
        )));
    }

    let base = Url::parse(base_url).map_err(|error| {
        McpError::InvalidInput(format!("invalid Aperture MCP base URL: {error}"))
    })?;
    if !matches!(base.scheme(), "http" | "https") {
        return Err(McpError::InvalidInput(
            "Aperture MCP base URL must use http or https".to_owned(),
        ));
    }
    if base.host_str().is_none() {
        return Err(McpError::InvalidInput(
            "Aperture MCP base URL must include a host".to_owned(),
        ));
    }
    if !base.username().is_empty() || base.password().is_some() {
        return Err(McpError::InvalidInput(
            "Aperture MCP base URL must not include credentials".to_owned(),
        ));
    }
    if base.query().is_some() || base.fragment().is_some() {
        return Err(McpError::InvalidInput(
            "Aperture MCP base URL must not include a query or fragment".to_owned(),
        ));
    }

    // Append textually instead of Url::join so a gateway installed below a
    // path prefix behaves like the Go strings.TrimRight(...)+"/v1/mcp" code.
    let endpoint = format!("{}/v1/mcp", base.as_str().trim_end_matches('/'));
    Url::parse(&endpoint)
        .map_err(|error| McpError::InvalidInput(format!("invalid Aperture MCP endpoint: {error}")))
}

fn ensure_not_cancelled<C: Cancellation + ?Sized>(cancellation: &C) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(McpError::Cancelled)
    } else {
        Ok(())
    }
}

fn validate_gateway_tool(tool: &GatewayTool) -> Result<()> {
    validate_tool_name(&tool.name)
}

fn validate_tool_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        return Err(McpError::InvalidInput(
            "MCP tool name cannot be empty".to_owned(),
        ));
    }
    if name != name.trim() {
        return Err(McpError::InvalidInput(
            "MCP tool name must not have leading or trailing whitespace".to_owned(),
        ));
    }
    if name.len() > MAX_TOOL_NAME_BYTES {
        return Err(McpError::InvalidInput(format!(
            "MCP tool name exceeds {MAX_TOOL_NAME_BYTES} bytes"
        )));
    }
    if name.chars().any(char::is_control) {
        return Err(McpError::InvalidInput(
            "MCP tool name must not contain control characters".to_owned(),
        ));
    }
    Ok(())
}

fn validate_resource_uri(uri: &str) -> Result<()> {
    if uri.trim().is_empty() {
        return Err(McpError::InvalidInput(
            "MCP resource URI cannot be empty".to_owned(),
        ));
    }
    if uri.len() > MAX_RESOURCE_URI_BYTES {
        return Err(McpError::InvalidInput(format!(
            "MCP resource URI exceeds {MAX_RESOURCE_URI_BYTES} bytes"
        )));
    }
    if uri.chars().any(char::is_control) {
        return Err(McpError::InvalidInput(
            "MCP resource URI must not contain control characters".to_owned(),
        ));
    }
    Ok(())
}

fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.trim().is_empty() {
        return Err(McpError::Protocol(
            "MCP initialize response missing Mcp-Session-Id header".to_owned(),
        ));
    }
    if session_id.len() > MAX_SESSION_ID_BYTES {
        return Err(McpError::Protocol(format!(
            "MCP initialize response Mcp-Session-Id exceeds {MAX_SESSION_ID_BYTES} bytes"
        )));
    }
    if session_id.chars().any(char::is_control) {
        return Err(McpError::Protocol(
            "MCP initialize response has an invalid Mcp-Session-Id header".to_owned(),
        ));
    }
    Ok(())
}

fn read_response_body<C: Cancellation + ?Sized>(
    response: &mut Response,
    cancellation: &C,
) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_RESPONSE_BYTES as u64)
    {
        return Err(McpError::ResponseTooLarge {
            limit: MAX_RESPONSE_BYTES,
        });
    }
    read_limited(response, MAX_RESPONSE_BYTES, cancellation)
}

fn read_limited<R: Read, C: Cancellation + ?Sized>(
    reader: &mut R,
    limit: usize,
    cancellation: &C,
) -> Result<Vec<u8>> {
    let maximum = limit.checked_add(1).ok_or_else(|| {
        McpError::InvalidInput("MCP response limit cannot be represented".to_owned())
    })?;
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 8 << 10];

    while bytes.len() < maximum {
        ensure_not_cancelled(cancellation)?;
        let remaining = maximum - bytes.len();
        let chunk_len = remaining.min(buffer.len());
        let read = match reader.read(&mut buffer[..chunk_len]) {
            Ok(read) => read,
            Err(_error) if cancellation.is_cancelled() => return Err(McpError::Cancelled),
            Err(error) => return Err(McpError::ResponseRead(error)),
        };
        if read == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..read]);
    }
    ensure_not_cancelled(cancellation)?;

    if bytes.len() > limit {
        return Err(McpError::ResponseTooLarge { limit });
    }
    Ok(bytes)
}

fn truncate_connector_output(full_text: &str) -> String {
    if full_text.len() <= MAX_CONNECTOR_OUTPUT_BYTES {
        return full_text.to_owned();
    }
    let mut end = MAX_CONNECTOR_OUTPUT_BYTES;
    while end > 0 && !full_text.is_char_boundary(end) {
        end -= 1;
    }
    let clipped = &full_text[..end];
    let mut note = format!(
        "\n\n[Showing the first {} of {} bytes",
        clipped.len(),
        full_text.len()
    );
    if let Ok(path) = write_overflow(full_text) {
        note.push_str(". Full output: ");
        note.push_str(&path.display().to_string());
    }
    note.push(']');
    format!("{clipped}{note}")
}

fn write_overflow(contents: &str) -> io::Result<PathBuf> {
    let directory = std::env::temp_dir();
    for _ in 0..MAX_OVERFLOW_FILE_ATTEMPTS {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "goshcoder-aperture-connector-{}-{sequence}.json",
            process::id()
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let result = (|| {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
            }
            file.write_all(contents.as_bytes())?;
            file.sync_all()
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&path);
            return Err(error);
        }
        return Ok(path);
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate connector output overflow file",
    ))
}

fn json_diagnostic(value: &Value) -> String {
    let encoded = value.to_string();
    clip_utf8(&encoded, MAX_JSON_DIAGNOSTIC_BYTES).to_owned()
}

fn invalid_json_error(payload: &[u8]) -> McpError {
    let preview = String::from_utf8_lossy(payload);
    let preview = preview.trim();
    McpError::Protocol(format!(
        "MCP response is not valid JSON: {}",
        clip_utf8(preview, MAX_INVALID_JSON_PREVIEW_BYTES)
    ))
}

fn clip_utf8(value: &str, limit: usize) -> &str {
    if value.len() <= limit {
        return value;
    }
    let mut end = limit;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn decode_json_rpc_response(payload: &[u8], expected_id: Option<u64>) -> Result<Value> {
    let (decoded, diagnostic_payload) = match serde_json::from_slice::<Value>(payload) {
        Ok(value) => (value, Cow::Borrowed(payload)),
        Err(_) => {
            let Some(sse_payload) = extract_sse_data(payload) else {
                return Err(invalid_json_error(payload));
            };
            let decoded = serde_json::from_slice::<Value>(&sse_payload)
                .map_err(|_| invalid_json_error(&sse_payload))?;
            (decoded, Cow::Owned(sse_payload))
        }
    };

    let object = decoded
        .as_object()
        .ok_or_else(|| invalid_json_error(diagnostic_payload.as_ref()))?;
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(McpError::Protocol(
            "MCP response did not declare JSON-RPC 2.0".to_owned(),
        ));
    }
    if let Some(expected_id) = expected_id
        && object.get("id").and_then(Value::as_u64) != Some(expected_id)
    {
        return Err(McpError::Protocol(format!(
            "MCP response id did not match request id {expected_id}"
        )));
    }
    if let Some(error) = object.get("error").filter(|error| !error.is_null()) {
        let error = error
            .as_object()
            .ok_or_else(|| McpError::Protocol("MCP response error was not an object".to_owned()))?;
        let code = error.get("code").and_then(Value::as_i64).ok_or_else(|| {
            McpError::Protocol("MCP response error had no numeric code".to_owned())
        })?;
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                McpError::Protocol("MCP response error had no string message".to_owned())
            })?;
        return Err(McpError::Rpc {
            code,
            message: clip_utf8(message, MAX_JSON_DIAGNOSTIC_BYTES).to_owned(),
        });
    }
    object
        .get("result")
        .cloned()
        .ok_or_else(|| McpError::Protocol("MCP response had neither result nor error".to_owned()))
}

/// Extracts the data payload of the first SSE event, accepting standard
/// multi-line `data:` framing as well as the Go client's usual single line.
fn extract_sse_data(payload: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(payload).ok()?;
    let mut lines = Vec::new();
    let mut saw_data = false;
    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            if saw_data {
                break;
            }
            continue;
        }
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        saw_data = true;
        lines.push(data.strip_prefix(' ').unwrap_or(data));
    }
    saw_data.then(|| lines.join("\n").into_bytes())
}

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<u64>,
}

impl JsonRpcRequest {
    fn request(id: u64, method: &'static str, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            method,
            params,
            id: Some(id),
        }
    }

    fn notification(method: &'static str) -> Self {
        Self {
            jsonrpc: "2.0",
            method,
            params: None,
            id: None,
        }
    }
}

struct JsonRpcReply {
    result: Value,
    session_id: Option<String>,
}

#[derive(Deserialize)]
struct ToolsListResponse {
    #[serde(default)]
    tools: Option<Vec<GatewayTool>>,
}

#[derive(Deserialize)]
struct ResourcesListResponse {
    #[serde(default)]
    resources: Option<Vec<McpResource>>,
}

#[derive(Deserialize)]
struct ResourceTemplatesListResponse {
    #[serde(rename = "resourceTemplates", default)]
    resource_templates: Option<Vec<McpResourceTemplate>>,
}

#[derive(Deserialize)]
struct ResourcesReadResponse {
    #[serde(default)]
    contents: Option<Vec<McpResourceContent>>,
}

fn null_to_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(|value| value.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{BufRead, BufReader},
        net::{TcpListener, TcpStream},
        sync::mpsc::{self, Receiver},
        thread::{self, JoinHandle},
    };

    use super::*;

    #[derive(Debug)]
    struct CapturedRequest {
        target: String,
        headers: BTreeMap<String, String>,
        body: Value,
    }

    fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .expect("read request line");
        let target = request_line
            .split_whitespace()
            .nth(1)
            .expect("request target")
            .to_owned();
        let mut headers = BTreeMap::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).expect("read request header");
            if line == "\r\n" || line.is_empty() {
                break;
            }
            let (name, value) = line.split_once(':').expect("header separator");
            headers.insert(name.to_ascii_lowercase(), value.trim().to_owned());
        }
        let content_length = headers
            .get("content-length")
            .expect("content length")
            .parse::<usize>()
            .expect("numeric content length");
        let mut body = vec![0; content_length];
        reader.read_exact(&mut body).expect("read request body");
        CapturedRequest {
            target,
            headers,
            body: serde_json::from_slice(&body).expect("JSON-RPC request"),
        }
    }

    fn response(status: u16, body: &str, headers: &[(&str, &str)]) -> Vec<u8> {
        let reason = match status {
            200 => "OK",
            202 => "Accepted",
            _ => "Test Response",
        };
        let mut output = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            output.push_str(name);
            output.push_str(": ");
            output.push_str(value);
            output.push_str("\r\n");
        }
        output.push_str("\r\n");
        let mut output = output.into_bytes();
        output.extend_from_slice(body.as_bytes());
        output
    }

    fn rpc_response(id: u64, result: Value, sse: bool, session_id: Option<&str>) -> Vec<u8> {
        let body = json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string();
        let body = if sse {
            format!("event: message\ndata: {body}\n\n")
        } else {
            body
        };
        let mut headers = Vec::new();
        if let Some(session_id) = session_id {
            headers.push(("Mcp-Session-Id", session_id));
        }
        response(200, &body, &headers)
    }

    fn test_server(responses: Vec<Vec<u8>>) -> (String, Receiver<CapturedRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let address = listener.local_addr().expect("test address");
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("accept request");
                let request = read_request(&mut stream);
                sender.send(request).expect("send request");
                stream.write_all(&response).expect("write response");
                stream.flush().expect("flush response");
            }
        });
        (format!("http://{address}"), receiver, worker)
    }

    #[test]
    fn initializes_lists_and_calls_with_plain_or_sse_responses() {
        for sse in [false, true] {
            let (base_url, requests, worker) = test_server(vec![
                rpc_response(
                    1,
                    json!({"protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {}}),
                    sse,
                    Some("session-123"),
                ),
                response(202, "", &[]),
                rpc_response(
                    2,
                    json!({"tools": [{
                        "name": "github_list_repos",
                        "description": "List repositories",
                        "inputSchema": {"type": "object", "properties": {"org": {"type": "string"}}}
                    }]}),
                    sse,
                    None,
                ),
                rpc_response(
                    3,
                    json!({"content": [{"type": "text", "text": "ok:github_list_repos"}]}),
                    sse,
                    None,
                ),
            ]);

            let session = McpSession::new(&base_url).expect("initialize session");
            assert_eq!(session.endpoint().path(), "/v1/mcp");
            assert_eq!(session.session_id(), "session-123");
            let tools = session.list_tools().expect("list tools");
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0].name, "github_list_repos");
            let result = session
                .call_tool(
                    "github_list_repos",
                    Map::from_iter([("org".to_owned(), Value::String("acme".to_owned()))]),
                )
                .expect("call tool");
            assert_eq!(result.content[0].text, "ok:github_list_repos");

            let captured = (0..4)
                .map(|_| requests.recv().expect("captured request"))
                .collect::<Vec<_>>();
            worker.join().expect("server worker");
            assert!(captured.iter().all(|request| request.target == "/v1/mcp"));
            assert_eq!(captured[0].body["method"], "initialize");
            assert_eq!(captured[0].body["id"], 1);
            assert_eq!(
                captured[0].body["params"]["protocolVersion"],
                MCP_PROTOCOL_VERSION
            );
            assert_eq!(captured[1].body["method"], "notifications/initialized");
            assert!(captured[1].body.get("id").is_none());
            assert_eq!(captured[2].body["id"], 2);
            assert_eq!(captured[3].body["id"], 3);
            assert_eq!(
                captured[2].headers.get("mcp-session-id"),
                Some(&"session-123".to_owned())
            );
            assert_eq!(
                captured[3].body["params"]["arguments"]["org"],
                Value::String("acme".to_owned())
            );
        }
    }

    #[test]
    fn initialize_requires_a_session_header() {
        let (base_url, requests, worker) = test_server(vec![rpc_response(
            1,
            json!({"protocolVersion": MCP_PROTOCOL_VERSION}),
            false,
            None,
        )]);
        let error = McpSession::new(&base_url).expect_err("session header must be required");
        assert!(
            error.to_string().contains("Mcp-Session-Id"),
            "unexpected error: {error}"
        );
        requests.recv().expect("initialize request");
        worker.join().expect("server worker");
    }

    #[test]
    fn validates_inputs_and_enforces_response_bound() {
        assert!(matches!(
            validate_tool_name(""),
            Err(McpError::InvalidInput(message)) if message.contains("cannot be empty")
        ));
        assert!(matches!(
            validate_tool_name(" bad"),
            Err(McpError::InvalidInput(message)) if message.contains("whitespace")
        ));
        assert!(matches!(
            mcp_endpoint("ftp://gateway.example"),
            Err(McpError::InvalidInput(message)) if message.contains("http or https")
        ));

        let mut bytes = io::repeat(b'x').take((MAX_RESPONSE_BYTES + 1) as u64);
        assert!(matches!(
            read_limited(&mut bytes, MAX_RESPONSE_BYTES, &NeverCancelled),
            Err(McpError::ResponseTooLarge { limit }) if limit == MAX_RESPONSE_BYTES
        ));
    }

    #[test]
    fn preserves_rpc_diagnostics_and_observes_cancellation_before_dispatch() {
        let rpc_error = decode_json_rpc_response(
            br#"{"jsonrpc":"2.0","id":2,"error":{"code":-32001,"message":"connector unavailable"}}"#,
            Some(2),
        )
        .expect_err("JSON-RPC error must be returned");
        assert_eq!(
            rpc_error.to_string(),
            "MCP error: connector unavailable (code -32001)"
        );
        let invalid = decode_json_rpc_response(b"not JSON", Some(2))
            .expect_err("invalid JSON must be returned");
        assert_eq!(
            invalid.to_string(),
            "MCP response is not valid JSON: not JSON"
        );

        struct Cancelled;
        impl Cancellation for Cancelled {
            fn is_cancelled(&self) -> bool {
                true
            }
        }
        let client = McpClient::new("http://127.0.0.1:1").expect("build client");
        assert!(matches!(
            client.initialize_with(&Cancelled),
            Err(McpError::Cancelled)
        ));
    }

    #[test]
    fn agent_result_is_utf8_bounded_and_preserves_overflow() {
        let complete = "é".repeat(MAX_CONNECTOR_OUTPUT_BYTES);
        let result = call_result_to_agent_result(
            "github_large",
            McpCallResult {
                content: vec![McpContentItem {
                    kind: "text".to_owned(),
                    text: complete.clone(),
                }],
                ..McpCallResult::default()
            },
        )
        .expect("convert result");
        let text = result.content[0].plain_text().expect("text result");
        assert!(text.len() <= MAX_CONNECTOR_OUTPUT_BYTES + 256);
        assert!(text.contains("Showing the first"));
        let path = text
            .split("Full output: ")
            .nth(1)
            .expect("overflow path")
            .trim_end_matches(']')
            .trim();
        assert_eq!(
            fs::read_to_string(path).expect("overflow content"),
            complete
        );
        fs::remove_file(path).expect("remove overflow file");
    }

    #[test]
    fn gateway_tool_becomes_an_executable_agent_tool() {
        let (base_url, requests, worker) = test_server(vec![
            rpc_response(
                1,
                json!({"protocolVersion": MCP_PROTOCOL_VERSION}),
                false,
                Some("session-456"),
            ),
            response(202, "", &[]),
            rpc_response(
                2,
                json!({"content": [{"type": "text", "text": "created"}]}),
                false,
                None,
            ),
        ]);
        let session = McpSession::new(&base_url).expect("initialize session");
        let tool = gateway_tool_to_agent_tool(
            session,
            GatewayTool {
                name: "github_create_issue".to_owned(),
                description: String::new(),
                input_schema: json!({"type": "array"}),
            },
        )
        .expect("create agent tool");
        assert_eq!(
            tool.description,
            "Aperture connector tool: github_create_issue"
        );
        assert_eq!(tool.parameters, json!({"type": "object", "properties": {}}));

        let updates = Arc::new(std::sync::Mutex::new(Vec::new()));
        let updates_for_callback = Arc::clone(&updates);
        let on_update: agent::ToolUpdate = Arc::new(move |update| {
            updates_for_callback
                .lock()
                .expect("updates lock")
                .push(update);
        });
        let result = (tool.execute)(
            agent::CancellationToken::default(),
            "tool-call-1".to_owned(),
            BTreeMap::from([("title".to_owned(), Value::String("Issue".to_owned()))]),
            on_update,
        )
        .expect("execute agent tool");
        assert_eq!(result.content[0].plain_text(), Some("created"));
        assert_eq!(updates.lock().expect("updates lock").len(), 1);

        let captured = (0..3)
            .map(|_| requests.recv().expect("captured request"))
            .collect::<Vec<_>>();
        worker.join().expect("server worker");
        assert_eq!(captured[2].body["method"], "tools/call");
        assert_eq!(
            captured[2].body["params"]["name"],
            Value::String("github_create_issue".to_owned())
        );
        assert_eq!(
            captured[2].body["params"]["arguments"]["title"],
            Value::String("Issue".to_owned())
        );
    }

    #[test]
    fn gateway_tool_adapter_coerces_untrusted_schema() {
        assert_eq!(
            coerce_input_schema(&json!({"type": "array"})),
            json!({"type": "object", "properties": {}})
        );
        assert_eq!(
            coerce_input_schema(
                &json!({"type": "object", "properties": {"x": {"type": "string"}}})
            ),
            json!({"type": "object", "properties": {"x": {"type": "string"}}})
        );
    }
}
