//! Local `computer-use-linux` MCP integration.
//!
//! This module deliberately owns configuration maintenance, process lifetime,
//! JSON-RPC framing, and the model-facing proxy behavior.  The application
//! entry point can register [`agent_tool`] when its extension runtime is ready;
//! keeping that registration outside this module makes the transport usable by
//! other local callers as well.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use serde::Deserialize;
use serde_json::{Map, Value, json};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

/// Upstream package name used in installation guidance.
pub const PACKAGE_NAME: &str = "@agent-sh/computer-use-linux";
/// MCP server key and executable name.
pub const SERVER_NAME: &str = "computer-use-linux";
/// Environment override checked before `PATH`.
pub const BINARY_ENV_VAR: &str = "COMPUTER_USE_LINUX_BIN";
/// Model-visible proxy name prefix used by pi-compatible hosts.
pub const TOOL_NAME_PREFIX: &str = "computer_use_linux_";
/// Installation guidance when binary discovery fails.
pub const INSTALL_HINT: &str = "Install it with 'npm install -g @agent-sh/computer-use-linux' \
or 'cargo install computer-use-linux', or set COMPUTER_USE_LINUX_BIN.";

/// Maximum accepted size of an existing `mcp.json`.
pub const MAX_MCP_CONFIG_BYTES: usize = 4 << 20;
/// Maximum encoded JSON-RPC request size.
pub const MAX_REQUEST_BYTES: usize = 8 << 20;
/// Maximum size of one newline-delimited server response.
pub const MAX_RESPONSE_LINE_BYTES: usize = 64 << 20;
/// Maximum text copied into a model tool result.
pub const MAX_TEXT_OUTPUT_BYTES: usize = 50 << 10;

const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
const MAX_SKIPPED_MESSAGES: usize = 1_024;
const RESPONSE_POLL_INTERVAL: Duration = Duration::from_millis(25);
const READER_SHUTDOWN_GRACE: Duration = Duration::from_millis(250);
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Errors from configuration, process control, or the MCP protocol.
#[derive(Debug)]
pub enum ComputerUseError {
    Io(io::Error),
    Json(serde_json::Error),
    InvalidConfiguration(String),
    InvalidInput(String),
    Protocol(String),
    Rpc { code: i64, message: String },
    Timeout { operation: String },
    Cancelled,
    ToolFailed(String),
}

impl fmt::Display for ComputerUseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "I/O error: {error}"),
            Self::Json(error) => write!(formatter, "JSON error: {error}"),
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Protocol(message) => write!(formatter, "{SERVER_NAME}: {message}"),
            Self::Rpc { code, message } => {
                write!(formatter, "{SERVER_NAME}: {message} (code {code})")
            }
            Self::Timeout { operation } => {
                write!(formatter, "{SERVER_NAME} timed out during {operation}")
            }
            Self::Cancelled => formatter.write_str("computer-use operation was cancelled"),
            Self::ToolFailed(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ComputerUseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ComputerUseError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ComputerUseError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub type Result<T> = std::result::Result<T, ComputerUseError>;

/// Locates the server binary using the upstream extension's discovery order.
///
/// The explicit override need only point at a regular file, matching the
/// upstream behavior. `PATH` and `~/.local/bin` candidates must be executable
/// on Unix.
pub fn find_binary() -> Option<PathBuf> {
    find_binary_from(|name| env::var_os(name))
}

/// Testable variant of [`find_binary`] with an injected environment reader.
pub fn find_binary_from<F>(getenv: F) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(override_path) = getenv(BINARY_ENV_VAR)
        .filter(|value| !value.as_os_str().is_empty())
        .map(PathBuf::from)
        && is_regular_file(&override_path)
    {
        return Some(override_path);
    }

    if let Some(path) = getenv("PATH") {
        for directory in env::split_paths(&path) {
            if directory.as_os_str().is_empty() {
                continue;
            }
            let candidate = directory.join(SERVER_NAME);
            if is_executable_file(&candidate) {
                return Some(candidate);
            }
        }
    }

    let home = getenv("HOME").or_else(|| getenv("USERPROFILE"));
    if let Some(home) = home.filter(|value| !value.as_os_str().is_empty()) {
        let candidate = PathBuf::from(home)
            .join(".local")
            .join("bin")
            .join(SERVER_NAME);
        if is_executable_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_regular_file(path: &Path) -> bool {
    fs::metadata(path).is_ok_and(|metadata| metadata.is_file())
}

fn is_executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Converts a raw MCP name such as `doctor` into the model-facing name.
pub fn prefixed_tool_name(raw_name: &str) -> String {
    format!("{TOOL_NAME_PREFIX}{raw_name}")
}

/// Accepts either a model-facing prefixed tool name or a raw MCP name.
pub fn raw_tool_name(name: &str) -> &str {
    name.strip_prefix(TOOL_NAME_PREFIX).unwrap_or(name)
}

/// Outcome of a safe `mcp.json` update.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnsureResult {
    Updated,
    Unchanged,
}

/// Adds or updates this server's entry in an MCP config file.
///
/// A missing file is created. Existing JSON must be a top-level object and
/// `mcpServers`, when present, must also be an object. Invalid files are never
/// overwritten. Unknown top-level fields, other server entries, and unrelated
/// fields on this server's object are retained.
pub fn ensure_server_entry(
    config_path: impl AsRef<Path>,
    binary_path: impl AsRef<Path>,
) -> Result<EnsureResult> {
    let config_path = config_path.as_ref();
    let command = binary_path
        .as_ref()
        .to_str()
        .ok_or_else(|| {
            ComputerUseError::InvalidConfiguration(
                "computer-use binary path is not valid UTF-8 and cannot be represented in mcp.json"
                    .to_owned(),
            )
        })?
        .to_owned();
    let mut config = read_mcp_config(config_path)?;

    let expected_args = Value::Array(vec![Value::String("mcp".to_owned())]);
    let servers = match config.get_mut("mcpServers") {
        Some(Value::Object(servers)) => servers,
        Some(_) => {
            return Err(ComputerUseError::InvalidConfiguration(format!(
                "MCP config at {} has a non-object mcpServers value; refusing to overwrite it",
                config_path.display()
            )));
        }
        None => {
            config.insert("mcpServers".to_owned(), Value::Object(Map::new()));
            config
                .get_mut("mcpServers")
                .and_then(Value::as_object_mut)
                .expect("mcpServers object was just inserted")
        }
    };

    if let Some(Value::Object(existing)) = servers.get(SERVER_NAME)
        && existing.get("command") == Some(&Value::String(command.clone()))
        && existing.get("args") == Some(&expected_args)
    {
        return Ok(EnsureResult::Unchanged);
    }

    // The target entry is ours, so it may be repaired. Preserve any extra
    // metadata it already has instead of discarding it during an update.
    let mut entry = servers
        .get(SERVER_NAME)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    entry.insert("command".to_owned(), Value::String(command));
    entry.insert("args".to_owned(), expected_args);
    servers.insert(SERVER_NAME.to_owned(), Value::Object(entry));

    write_mcp_config(config_path, &config)?;
    Ok(EnsureResult::Updated)
}

fn read_mcp_config(path: &Path) -> Result<Map<String, Value>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Map::new()),
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(ComputerUseError::InvalidConfiguration(format!(
            "MCP config at {} is not a regular file",
            path.display()
        )));
    }
    if metadata.len() > MAX_MCP_CONFIG_BYTES as u64 {
        return Err(ComputerUseError::InvalidConfiguration(format!(
            "MCP config at {} exceeds {} bytes",
            path.display(),
            MAX_MCP_CONFIG_BYTES
        )));
    }

    let mut bytes = Vec::new();
    let mut limited = file.take(MAX_MCP_CONFIG_BYTES as u64 + 1);
    limited.read_to_end(&mut bytes)?;
    if bytes.len() > MAX_MCP_CONFIG_BYTES {
        return Err(ComputerUseError::InvalidConfiguration(format!(
            "MCP config at {} exceeds {} bytes",
            path.display(),
            MAX_MCP_CONFIG_BYTES
        )));
    }

    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        ComputerUseError::InvalidConfiguration(format!(
            "MCP config at {} is not valid JSON; refusing to overwrite it: {error}",
            path.display()
        ))
    })?;
    match value {
        Value::Object(object) => Ok(object),
        value => Err(ComputerUseError::InvalidConfiguration(format!(
            "MCP config at {} contains {}, expected a JSON object; refusing to overwrite it",
            path.display(),
            json_type_name(&value)
        ))),
    }
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

fn write_mcp_config(path: &Path, config: &Map<String, Value>) -> Result<()> {
    let mut contents = serde_json::to_vec_pretty(config)?;
    contents.push(b'\n');
    atomic_write(path, &contents)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path.file_name().ok_or_else(|| {
        ComputerUseError::InvalidConfiguration(format!("{} has no file name", path.display()))
    })?;

    for _ in 0..32 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            std::process::id(),
            sequence
        ));

        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = match options.open(&temporary) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };

        let result: io::Result<()> = (|| {
            #[cfg(unix)]
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
            file.write_all(contents)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_file(&temporary);
            return Err(error.into());
        }
        return Ok(());
    }

    Err(ComputerUseError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "could not allocate a temporary MCP config next to {}",
            path.display()
        ),
    )))
}

/// Cooperative cancellation abstraction for callers outside the agent runtime.
pub trait Cancellation: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

/// Cancellation implementation for callers that do not need cancellation.
#[derive(Debug, Default)]
pub struct NeverCancelled;

impl Cancellation for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Process command used to launch an MCP server.
#[derive(Clone, Debug)]
pub struct ServerCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub environment: BTreeMap<OsString, OsString>,
}

impl ServerCommand {
    pub fn new<I, A>(program: impl Into<PathBuf>, args: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<OsString>,
    {
        Self {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            environment: BTreeMap::new(),
        }
    }

    /// Adds or replaces one inherited-environment override.
    pub fn with_env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }
}

/// Time bounds for initialization and regular tool calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionTimeouts {
    pub initialization: Duration,
    pub call: Duration,
}

impl Default for SessionTimeouts {
    fn default() -> Self {
        Self {
            initialization: Duration::from_secs(15),
            call: Duration::from_secs(120),
        }
    }
}

/// One MCP entry returned by `tools/list`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
    #[serde(default)]
    pub annotations: Option<ToolAnnotations>,
}

/// Optional MCP safety annotations attached to a tool.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct ToolAnnotations {
    #[serde(default, rename = "readOnlyHint")]
    pub read_only_hint: Option<bool>,
    #[serde(default, rename = "destructiveHint")]
    pub destructive_hint: Option<bool>,
    #[serde(default, rename = "idempotentHint")]
    pub idempotent_hint: Option<bool>,
    #[serde(default, rename = "openWorldHint")]
    pub open_world_hint: Option<bool>,
}

/// One content item returned by `tools/call`.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct McpContentItem {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default, rename = "mimeType")]
    pub mime_type: Option<String>,
}

/// Decoded MCP `tools/call` result.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct McpCallResult {
    #[serde(default)]
    pub content: Vec<McpContentItem>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
    #[serde(default, rename = "structuredContent")]
    pub structured_content: Option<Value>,
}

#[derive(Default)]
struct SessionState {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    next_id: u64,
    tools: Option<Vec<McpTool>>,
}

/// Lazily spawned, serialized stdio connection to one MCP server process.
///
/// Clones share one mutex and therefore one process. A full request/response
/// cycle holds that mutex, which is intentional: desktop input is stateful and
/// MCP request ids must not be read concurrently from a single stdout stream.
#[derive(Clone)]
pub struct McpSession {
    command: ServerCommand,
    timeouts: SessionTimeouts,
    state: Arc<Mutex<SessionState>>,
}

impl McpSession {
    /// Prepares a normal `computer-use-linux mcp` session without spawning it.
    pub fn new(binary_path: impl Into<PathBuf>) -> Self {
        Self::from_command(ServerCommand::new(binary_path, ["mcp"]))
    }

    /// Prepares a session from a fully specified process command.
    pub fn from_command(command: ServerCommand) -> Self {
        Self {
            command,
            timeouts: SessionTimeouts::default(),
            state: Arc::new(Mutex::new(SessionState::default())),
        }
    }

    /// Replaces the request deadlines. Useful for controlled test or UI bounds.
    pub fn with_timeouts(mut self, timeouts: SessionTimeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// Explicitly terminates the child process and clears the tool cache.
    pub fn close(&self) {
        let mut state = lock(&self.state);
        self.stop_locked(&mut state);
    }

    /// Lists and caches server tools using a non-cancellable request.
    pub fn tools(&self) -> Result<Vec<McpTool>> {
        self.tools_with(&NeverCancelled)
    }

    /// Lists and caches server tools, stopping the child if cancelled or timed out.
    pub fn tools_with<C: Cancellation + ?Sized>(&self, cancellation: &C) -> Result<Vec<McpTool>> {
        let mut state = lock(&self.state);
        self.ensure_started_locked(&mut state, cancellation)?;
        if let Some(tools) = &state.tools {
            return Ok(tools.clone());
        }

        let response = self.round_trip_locked(
            &mut state,
            "tools/list",
            None,
            self.timeouts.initialization,
            cancellation,
        )?;
        let decoded: ToolsListResponse = match serde_json::from_value(response) {
            Ok(decoded) => decoded,
            Err(error) => {
                self.stop_locked(&mut state);
                return Err(ComputerUseError::Protocol(format!(
                    "decode tools/list: {error}"
                )));
            }
        };
        if decoded.tools.iter().any(|tool| tool.name.trim().is_empty()) {
            self.stop_locked(&mut state);
            return Err(ComputerUseError::Protocol(
                "tools/list returned a tool with an empty name".to_owned(),
            ));
        }
        state.tools = Some(decoded.tools.clone());
        Ok(decoded.tools)
    }

    /// Calls a tool using a non-cancellable request.
    pub fn call(&self, name: &str, arguments: Map<String, Value>) -> Result<McpCallResult> {
        self.call_with(&NeverCancelled, name, arguments)
    }

    /// Calls one MCP tool. Calls are serialized across all session clones.
    pub fn call_with<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        name: &str,
        arguments: Map<String, Value>,
    ) -> Result<McpCallResult> {
        if name.trim().is_empty() {
            return Err(ComputerUseError::InvalidInput(
                "MCP tool name cannot be empty".to_owned(),
            ));
        }

        let mut state = lock(&self.state);
        self.ensure_started_locked(&mut state, cancellation)?;
        let response = self.round_trip_locked(
            &mut state,
            "tools/call",
            Some(json!({"name": name, "arguments": arguments})),
            self.timeouts.call,
            cancellation,
        )?;
        match serde_json::from_value(response) {
            Ok(decoded) => Ok(decoded),
            Err(error) => {
                self.stop_locked(&mut state);
                Err(ComputerUseError::Protocol(format!(
                    "decode tools/call: {error}"
                )))
            }
        }
    }

    fn ensure_started_locked<C: Cancellation + ?Sized>(
        &self,
        state: &mut SessionState,
        cancellation: &C,
    ) -> Result<()> {
        if cancellation.is_cancelled() {
            return Err(ComputerUseError::Cancelled);
        }

        if let Some(child) = state.child.as_mut() {
            match child.try_wait() {
                Ok(None) => return Ok(()),
                Ok(Some(_)) => self.stop_locked(state),
                Err(error) => {
                    self.stop_locked(state);
                    return Err(error.into());
                }
            }
        }

        let mut command = Command::new(&self.command.program);
        command
            .args(&self.command.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // A full stderr pipe can deadlock a long-running server when no UI
            // is attached to drain diagnostics.
            .stderr(Stdio::null());
        for (key, value) in &self.command.environment {
            command.env(key, value);
        }

        let mut child = command.spawn()?;
        let Some(stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ComputerUseError::Protocol(
                "spawned server did not provide stdin".to_owned(),
            ));
        };
        let Some(stdout) = child.stdout.take() else {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            return Err(ComputerUseError::Protocol(
                "spawned server did not provide stdout".to_owned(),
            ));
        };

        state.child = Some(child);
        state.stdin = Some(stdin);
        state.stdout = Some(BufReader::with_capacity(64 << 10, stdout));
        state.next_id = 0;
        state.tools = None;

        let initialized = self.round_trip_locked(
            state,
            "initialize",
            Some(json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "goshcoder", "version": "1"},
            })),
            self.timeouts.initialization,
            cancellation,
        );
        let initialized = match initialized {
            Ok(result) => result,
            Err(error) => {
                self.stop_locked(state);
                return Err(error);
            }
        };
        if initialized
            .get("protocolVersion")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            self.stop_locked(state);
            return Err(ComputerUseError::Protocol(
                "initialize returned no protocolVersion".to_owned(),
            ));
        }
        if let Err(error) = self.write_notification_locked(state, "notifications/initialized") {
            self.stop_locked(state);
            return Err(error);
        }
        Ok(())
    }

    fn round_trip_locked<C: Cancellation + ?Sized>(
        &self,
        state: &mut SessionState,
        method: &str,
        params: Option<Value>,
        timeout: Duration,
        cancellation: &C,
    ) -> Result<Value> {
        if cancellation.is_cancelled() {
            self.stop_locked(state);
            return Err(ComputerUseError::Cancelled);
        }
        let id = state.next_id.checked_add(1).ok_or_else(|| {
            ComputerUseError::Protocol("JSON-RPC request id overflowed".to_owned())
        })?;
        state.next_id = id;

        if let Err(error) = self.write_request_locked(state, id, method, params) {
            self.stop_locked(state);
            return Err(error);
        }

        let Some(mut reader) = state.stdout.take() else {
            self.stop_locked(state);
            return Err(ComputerUseError::Protocol(
                "server stdout was unavailable".to_owned(),
            ));
        };
        let (sender, receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("computer-use-mcp-reader".to_owned())
            .spawn(move || {
                let result = read_matching_response(&mut reader, id);
                let _ = sender.send(ReadOutcome { reader, result });
            });
        if let Err(error) = worker {
            self.stop_locked(state);
            return Err(error.into());
        }

        let deadline = Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(Instant::now);
        loop {
            if cancellation.is_cancelled() {
                self.abort_reader_locked(state, &receiver);
                return Err(ComputerUseError::Cancelled);
            }
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                self.abort_reader_locked(state, &receiver);
                return Err(ComputerUseError::Timeout {
                    operation: method.to_owned(),
                });
            };

            match receiver.recv_timeout(remaining.min(RESPONSE_POLL_INTERVAL)) {
                Ok(outcome) => {
                    state.stdout = Some(outcome.reader);
                    match outcome.result {
                        Ok(result) => return Ok(result),
                        Err(error) => {
                            // JSON-RPC tool errors are valid responses and do
                            // not make a healthy session unusable.
                            if !matches!(&error, ComputerUseError::Rpc { .. }) {
                                self.stop_locked(state);
                            }
                            return Err(error);
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.stop_locked(state);
                    return Err(ComputerUseError::Protocol(
                        "MCP stdout reader stopped unexpectedly".to_owned(),
                    ));
                }
            }
        }
    }

    fn write_request_locked(
        &self,
        state: &mut SessionState,
        id: u64,
        method: &str,
        params: Option<Value>,
    ) -> Result<()> {
        let mut request = Map::new();
        request.insert("jsonrpc".to_owned(), Value::String("2.0".to_owned()));
        request.insert("id".to_owned(), Value::from(id));
        request.insert("method".to_owned(), Value::String(method.to_owned()));
        if let Some(params) = params {
            request.insert("params".to_owned(), params);
        }
        self.write_value_locked(state, Value::Object(request))
    }

    fn write_notification_locked(&self, state: &mut SessionState, method: &str) -> Result<()> {
        self.write_value_locked(state, json!({"jsonrpc": "2.0", "method": method}))
    }

    fn write_value_locked(&self, state: &mut SessionState, value: Value) -> Result<()> {
        let mut encoded = serde_json::to_vec(&value)?;
        if encoded.len() > MAX_REQUEST_BYTES {
            return Err(ComputerUseError::InvalidInput(format!(
                "MCP request exceeds {} bytes",
                MAX_REQUEST_BYTES
            )));
        }
        encoded.push(b'\n');
        let stdin = state
            .stdin
            .as_mut()
            .ok_or_else(|| ComputerUseError::Protocol("server stdin was unavailable".to_owned()))?;
        stdin.write_all(&encoded)?;
        stdin.flush()?;
        Ok(())
    }

    fn abort_reader_locked(
        &self,
        state: &mut SessionState,
        receiver: &mpsc::Receiver<ReadOutcome>,
    ) {
        self.stop_locked(state);
        // Killing the process closes stdout and lets the reader return. Do not
        // wait indefinitely if a platform delays that close; the detached
        // worker only owns the already-dead pipe.
        let _ = receiver.recv_timeout(READER_SHUTDOWN_GRACE);
    }

    fn stop_locked(&self, state: &mut SessionState) {
        state.tools = None;
        state.next_id = 0;
        drop(state.stdin.take());
        drop(state.stdout.take());
        if let Some(mut child) = state.child.take() {
            match child.try_wait() {
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        }
    }
}

impl Drop for McpSession {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) == 1 {
            let mut state = lock(&self.state);
            self.stop_locked(&mut state);
        }
    }
}

#[derive(Deserialize)]
struct ToolsListResponse {
    #[serde(default)]
    tools: Vec<McpTool>,
}

struct ReadOutcome {
    reader: BufReader<ChildStdout>,
    result: Result<Value>,
}

fn read_matching_response<R: BufRead>(reader: &mut R, expected_id: u64) -> Result<Value> {
    let expected_id = Value::from(expected_id);
    let mut skipped = 0;
    loop {
        let Some(line) = read_bounded_line(reader, MAX_RESPONSE_LINE_BYTES)? else {
            return Err(ComputerUseError::Protocol(
                "server closed stdout before returning a response".to_owned(),
            ));
        };
        if line.is_empty() {
            continue;
        }
        let response: Value = match serde_json::from_slice(&line) {
            Ok(response) => response,
            Err(_) => {
                skipped += 1;
                if skipped > MAX_SKIPPED_MESSAGES {
                    return Err(ComputerUseError::Protocol(
                        "too many malformed or unrelated stdout messages".to_owned(),
                    ));
                }
                continue;
            }
        };
        let Value::Object(object) = response else {
            skipped += 1;
            if skipped > MAX_SKIPPED_MESSAGES {
                return Err(ComputerUseError::Protocol(
                    "too many malformed or unrelated stdout messages".to_owned(),
                ));
            }
            continue;
        };

        let Some(response_id) = object.get("id") else {
            skipped += 1;
            if skipped > MAX_SKIPPED_MESSAGES {
                return Err(ComputerUseError::Protocol(
                    "too many malformed or unrelated stdout messages".to_owned(),
                ));
            }
            continue;
        };
        if response_id != &expected_id {
            skipped += 1;
            if skipped > MAX_SKIPPED_MESSAGES {
                return Err(ComputerUseError::Protocol(
                    "too many malformed or unrelated stdout messages".to_owned(),
                ));
            }
            continue;
        }
        if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
            return Err(ComputerUseError::Protocol(
                "matching response did not declare JSON-RPC 2.0".to_owned(),
            ));
        }
        if let Some(error) = object.get("error") {
            let code = error
                .get("code")
                .and_then(Value::as_i64)
                .unwrap_or_default();
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| error.to_string(), ToOwned::to_owned);
            return Err(ComputerUseError::Rpc { code, message });
        }
        return object.get("result").cloned().ok_or_else(|| {
            ComputerUseError::Protocol("matching response had neither result nor error".to_owned())
        });
    }
}

/// Reads exactly one newline-delimited frame without allocating beyond `limit`.
fn read_bounded_line<R: BufRead>(reader: &mut R, limit: usize) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    loop {
        let (consumed, completed) = {
            let buffer = reader.fill_buf()?;
            if buffer.is_empty() {
                if line.is_empty() {
                    return Ok(None);
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "MCP response ended before its newline delimiter",
                ));
            }
            let newline = buffer.iter().position(|byte| *byte == b'\n');
            let consumed = newline.map_or(buffer.len(), |index| index + 1);
            if line.len().saturating_add(consumed) > limit {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("MCP response line exceeds the {limit}-byte limit"),
                ));
            }
            line.extend_from_slice(&buffer[..consumed]);
            (consumed, newline.is_some())
        };
        reader.consume(consumed);
        if completed {
            line.pop(); // newline
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Some(line));
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

/// Model-facing description for the `mcp` proxy tool.
pub const TOOL_DESCRIPTION: &str = "Observe and control the local Linux desktop through the \
computer-use-linux MCP server: accessibility trees, window targeting, screenshots, and input \
synthesis. Call {\"server\": \"computer-use-linux\"} to list the desktop tools, {\"search\": \
\"...\"} to find one, and {\"tool\": \"computer_use_linux_<name>\", \"args\": {...}} to run it. \
Start with doctor and fix setup_accessibility or setup_window_targeting blockers it reports. \
Verify the intended window with list_windows or focused_window before targeted input. Prefer \
element indices and role/name/text selectors from get_app_state over pixel coordinates; pass \
explicit window or terminal targets to type_text and press_key; and re-check state after \
mutating actions. Desktop input is stateful, so never issue concurrent calls.";

/// JSON schema for the proxy's model-facing parameters.
pub fn proxy_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "List every tool of this MCP server. The only available server is computer-use-linux."
            },
            "search": {
                "type": "string",
                "description": "Search desktop tools by name or description."
            },
            "tool": {
                "type": "string",
                "description": "Tool to call, for example computer_use_linux_doctor or computer_use_linux_screenshot. The bare MCP name also works."
            },
            "args": {
                "type": "object",
                "description": "Arguments for the selected tool, matching its MCP schema.",
                "additionalProperties": true
            }
        }
    })
}

/// Parsed model request to the MCP proxy.
#[derive(Clone, Debug, PartialEq)]
pub enum ProxyRequest {
    List {
        server: String,
    },
    Search(String),
    Call {
        tool: String,
        arguments: Map<String, Value>,
    },
    Help,
}

/// Parses the documented `mcp({server|search|tool,args})` input shapes.
pub fn parse_proxy_request(parameters: &Value) -> Result<ProxyRequest> {
    let object = parameters.as_object().ok_or_else(|| {
        ComputerUseError::InvalidInput("computer-use MCP parameters must be an object".to_owned())
    })?;

    if let Some(tool) = object.get("tool") {
        let tool = tool.as_str().ok_or_else(|| {
            ComputerUseError::InvalidInput("\"tool\" must be a string".to_owned())
        })?;
        if !tool.trim().is_empty() {
            let arguments = match object.get("args") {
                None | Some(Value::Null) => Map::new(),
                Some(Value::Object(arguments)) => arguments.clone(),
                Some(_) => {
                    return Err(ComputerUseError::InvalidInput(
                        "\"args\" must be an object".to_owned(),
                    ));
                }
            };
            return Ok(ProxyRequest::Call {
                tool: tool.to_owned(),
                arguments,
            });
        }
    }

    if let Some(search) = object.get("search") {
        let search = search.as_str().ok_or_else(|| {
            ComputerUseError::InvalidInput("\"search\" must be a string".to_owned())
        })?;
        if !search.trim().is_empty() {
            return Ok(ProxyRequest::Search(search.to_owned()));
        }
    }

    if let Some(server) = object.get("server") {
        let server = server.as_str().ok_or_else(|| {
            ComputerUseError::InvalidInput("\"server\" must be a string".to_owned())
        })?;
        if !server.trim().is_empty() {
            return Ok(ProxyRequest::List {
                server: server.to_owned(),
            });
        }
    }
    Ok(ProxyRequest::Help)
}

/// Image content extracted from an MCP tool response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageResult {
    pub data: String,
    pub mime_type: String,
}

/// Integration-neutral proxy result. The agent adapter converts this to
/// [`crate::llm::ContentBlock`] only when the host is ready to register it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProxyResult {
    pub text: String,
    pub images: Vec<ImageResult>,
    pub details: Option<Value>,
}

impl ProxyResult {
    fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Converts the result into the current application's LLM content types.
    pub fn llm_content(&self) -> Vec<crate::llm::ContentBlock> {
        let mut content = Vec::new();
        if !self.text.is_empty() {
            content.push(crate::llm::ContentBlock::text(self.text.clone()));
        }
        content.extend(self.images.iter().map(|image| {
            crate::llm::ContentBlock::Image(crate::llm::ImageContent {
                data: image.data.clone(),
                mime_type: image.mime_type.clone(),
            })
        }));
        if content.is_empty() {
            content.push(crate::llm::ContentBlock::text("(no output)"));
        }
        content
    }
}

/// Native equivalent of pi-mcp-adapter's `mcp()` proxy, scoped to this server.
#[derive(Clone)]
pub struct ComputerUseTool {
    session: McpSession,
}

impl ComputerUseTool {
    pub fn new(session: McpSession) -> Self {
        Self { session }
    }

    /// Executes a proxy request without cancellation.
    pub fn execute(&self, parameters: &Value) -> Result<ProxyResult> {
        self.execute_with(parameters, &NeverCancelled, None)
    }

    /// Executes a request with caller-controlled cancellation and progress.
    pub fn execute_with<C: Cancellation + ?Sized>(
        &self,
        parameters: &Value,
        cancellation: &C,
        on_update: Option<&dyn Fn(String)>,
    ) -> Result<ProxyResult> {
        match parse_proxy_request(parameters)? {
            ProxyRequest::Call { tool, arguments } => {
                self.call_tool(cancellation, &tool, arguments, on_update)
            }
            ProxyRequest::Search(query) => self.search_tools(cancellation, &query),
            ProxyRequest::List { server } if server == SERVER_NAME => self.list_tools(cancellation),
            ProxyRequest::List { server } => Ok(ProxyResult::text(format!(
                "Unknown MCP server {server:?}. The only available server is {SERVER_NAME:?}."
            ))),
            ProxyRequest::Help => Ok(ProxyResult::text(format!(
                "Pass {{\"server\": \"{SERVER_NAME}\"}} to list the desktop tools, \
{{\"search\": \"...\"}} to find one, or {{\"tool\": \"computer_use_linux_<name>\", \
\"args\": {{...}}}} to call one."
            ))),
        }
    }

    fn list_tools<C: Cancellation + ?Sized>(&self, cancellation: &C) -> Result<ProxyResult> {
        let tools = self.session.tools_with(cancellation)?;
        let mut lines = vec![
            format!("{0} tool(s) from {SERVER_NAME}:", tools.len()),
            String::new(),
        ];
        lines.extend(tools.iter().map(tool_line));
        Ok(ProxyResult::text(lines.join("\n")))
    }

    fn search_tools<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        query: &str,
    ) -> Result<ProxyResult> {
        let tools = self.session.tools_with(cancellation)?;
        let needle = query.trim().to_lowercase();
        let matches = tools
            .into_iter()
            .filter(|tool| {
                tool.name.to_lowercase().contains(&needle)
                    || tool.description.to_lowercase().contains(&needle)
            })
            .collect::<Vec<_>>();
        match matches.len() {
            0 => Ok(ProxyResult::text(format!(
                "No desktop tools match {query:?}. Call {{\"server\": {SERVER_NAME:?}}} to list them all."
            ))),
            1 => Ok(ProxyResult::text(describe_tool(&matches[0]))),
            _ => Ok(ProxyResult::text(
                matches.iter().map(tool_line).collect::<Vec<_>>().join("\n"),
            )),
        }
    }

    fn call_tool<C: Cancellation + ?Sized>(
        &self,
        cancellation: &C,
        requested_name: &str,
        arguments: Map<String, Value>,
        on_update: Option<&dyn Fn(String)>,
    ) -> Result<ProxyResult> {
        let raw_name = raw_tool_name(requested_name).to_owned();
        let tools = self.session.tools_with(cancellation)?;
        if !tools.iter().any(|tool| tool.name == raw_name) {
            return Ok(ProxyResult::text(format!(
                "Tool {requested_name:?} not found. Call {{\"server\": {SERVER_NAME:?}}} to list the desktop tools."
            )));
        }
        if let Some(on_update) = on_update {
            on_update(format!("Calling {}...", prefixed_tool_name(&raw_name)));
        }

        let result = self.session.call_with(cancellation, &raw_name, arguments)?;
        call_result_to_proxy_result(&raw_name, result)
    }
}

fn tool_line(tool: &McpTool) -> String {
    let description = compact_description(&tool.description);
    let mutability = tool.annotations.as_ref().map_or("", |annotations| {
        if annotations.read_only_hint == Some(true) {
            " [read-only]"
        } else if annotations.destructive_hint == Some(true) {
            " [destructive]"
        } else {
            " [mutating]"
        }
    });
    format!(
        "- `{}`{mutability}: {description}",
        prefixed_tool_name(&tool.name)
    )
}

fn compact_description(description: &str) -> String {
    let compact = description.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        return "(no description)".to_owned();
    }
    if compact.chars().count() <= 120 {
        return compact;
    }
    format!("{}...", compact.chars().take(117).collect::<String>())
}

fn describe_tool(tool: &McpTool) -> String {
    let mut lines = vec![format!("### {}", prefixed_tool_name(&tool.name))];
    if !tool.description.is_empty() {
        lines.push(tool.description.clone());
    }
    lines.extend([
        String::new(),
        "**Parameters:**".to_owned(),
        "```".to_owned(),
        format_input_schema(&tool.input_schema),
        "```".to_owned(),
    ]);
    lines.join("\n")
}

/// Renders the useful object-property subset of an MCP JSON schema.
pub fn format_input_schema(schema: &Value) -> String {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return "(no parameters)".to_owned();
    };
    if properties.is_empty() {
        return "(no parameters)".to_owned();
    }
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<BTreeSet<_>>();
    let mut names = properties.keys().collect::<Vec<_>>();
    names.sort_unstable();

    names
        .into_iter()
        .map(|name| {
            let property = &properties[name];
            let property_type = property
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("any");
            let requirement = if required.contains(name.as_str()) {
                "required"
            } else {
                "optional"
            };
            let mut line = format!("- {name} ({property_type}, {requirement})");
            if let Some(description) = property.get("description").and_then(Value::as_str)
                && !description.is_empty()
            {
                line.push_str(": ");
                line.push_str(description);
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn call_result_to_proxy_result(raw_name: &str, result: McpCallResult) -> Result<ProxyResult> {
    let mut text_parts = Vec::new();
    let mut images = Vec::new();
    for item in result.content {
        if item.kind == "image" {
            let (Some(data), Some(mime_type)) = (item.data, item.mime_type) else {
                return Err(ComputerUseError::Protocol(format!(
                    "tools/call {raw_name:?} returned image content without data and mimeType"
                )));
            };
            if data.is_empty() || mime_type.is_empty() {
                return Err(ComputerUseError::Protocol(format!(
                    "tools/call {raw_name:?} returned empty image data or mimeType"
                )));
            }
            images.push(ImageResult { data, mime_type });
        } else if let Some(text) = item.text.filter(|text| !text.is_empty()) {
            text_parts.push(text);
        }
    }

    let text = clip_text_output(text_parts.join("\n\n"));
    if result.is_error {
        return Err(ComputerUseError::ToolFailed(if text.is_empty() {
            format!("{raw_name} failed")
        } else {
            text
        }));
    }
    Ok(ProxyResult {
        text,
        images,
        details: Some(json!({"tool": raw_name})),
    })
}

fn clip_text_output(text: String) -> String {
    if text.len() <= MAX_TEXT_OUTPUT_BYTES {
        return text;
    }
    let original_len = text.len();
    let mut end = MAX_TEXT_OUTPUT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[Output truncated: showing the first {} of {} bytes]",
        &text[..end],
        end,
        original_len
    )
}

/// Converts a transport error into guidance suitable for a model-facing tool.
pub fn describe_session_error(error: &ComputerUseError) -> String {
    match error {
        ComputerUseError::Timeout { .. } => format!(
            "{SERVER_NAME} did not respond in time; run 'computer-use-linux doctor' to check desktop readiness: {error}"
        ),
        _ => error.to_string(),
    }
}

struct AgentCancellation<'a>(&'a crate::agent::CancellationToken);

impl Cancellation for AgentCancellation<'_> {
    fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }
}

/// Builds the current agent runtime's sequential `mcp` tool.
///
/// This is an adapter only; the caller remains responsible for adding the
/// returned tool to `agent::InitialState.tools` and for retaining or closing
/// the session as appropriate.
pub fn agent_tool(session: McpSession) -> crate::agent::Tool {
    let proxy = ComputerUseTool::new(session);
    let mut tool = crate::agent::Tool::new(
        "mcp",
        "Desktop MCP",
        TOOL_DESCRIPTION,
        proxy_parameters(),
        move |cancellation, _tool_call_id, arguments, on_update| {
            let parameters = Value::Object(arguments.into_iter().collect());
            let agent_cancellation = AgentCancellation(&cancellation);
            let update = |message| on_update(crate::agent::ToolResult::text(message));
            let result = proxy
                .execute_with(&parameters, &agent_cancellation, Some(&update))
                .map_err(|error| describe_session_error(&error))?;
            let content = result.llm_content();
            Ok(crate::agent::ToolResult {
                content,
                details: result.details,
                ..crate::agent::ToolResult::default()
            })
        },
    );
    tool.execution_mode = Some(crate::agent::ToolExecutionMode::Sequential);
    tool
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        env,
        ffi::OsString,
        io::Cursor,
        process,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    const HELPER_ENV: &str = "GOSHCODER_COMPUTERUSE_TEST_HELPER";

    fn test_dir(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        env::temp_dir().join(format!(
            "goshcoder-computeruse-{label}-{}-{sequence}",
            process::id()
        ))
    }

    fn executable(path: &Path) {
        fs::create_dir_all(path.parent().expect("test path parent"))
            .expect("create test directory");
        fs::write(path, b"#!/bin/sh\n").expect("write executable");
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o755)).expect("make executable");
    }

    #[test]
    fn binary_discovery_prefers_override_then_path_then_local_bin() {
        let root = test_dir("discovery");
        let override_path = root.join("override");
        let path_dir = root.join("path");
        let home = root.join("home");
        executable(&override_path);
        executable(&path_dir.join(SERVER_NAME));
        executable(&home.join(".local/bin").join(SERVER_NAME));

        let values = BTreeMap::from([
            (BINARY_ENV_VAR.to_owned(), OsString::from(&override_path)),
            (
                "PATH".to_owned(),
                env::join_paths([&path_dir]).expect("PATH"),
            ),
            ("HOME".to_owned(), OsString::from(&home)),
        ]);
        assert_eq!(
            find_binary_from(|name| values.get(name).cloned()),
            Some(override_path.clone())
        );

        fs::remove_file(&override_path).expect("remove override");
        assert_eq!(
            find_binary_from(|name| values.get(name).cloned()),
            Some(path_dir.join(SERVER_NAME))
        );

        fs::remove_file(path_dir.join(SERVER_NAME)).expect("remove PATH binary");
        assert_eq!(
            find_binary_from(|name| values.get(name).cloned()),
            Some(home.join(".local/bin").join(SERVER_NAME))
        );
        fs::remove_dir_all(root).expect("clean test directory");
    }

    #[cfg(unix)]
    #[test]
    fn path_discovery_rejects_non_executable_files_but_override_allows_them() {
        let root = test_dir("permissions");
        fs::create_dir_all(&root).expect("create test directory");
        let candidate = root.join(SERVER_NAME);
        fs::write(&candidate, b"not executable").expect("write candidate");
        fs::set_permissions(&candidate, fs::Permissions::from_mode(0o644))
            .expect("remove executable bit");

        let path_values =
            BTreeMap::from([("PATH".to_owned(), env::join_paths([&root]).expect("PATH"))]);
        assert_eq!(
            find_binary_from(|name| path_values.get(name).cloned()),
            None
        );

        let override_values =
            BTreeMap::from([(BINARY_ENV_VAR.to_owned(), OsString::from(&candidate))]);
        assert_eq!(
            find_binary_from(|name| override_values.get(name).cloned()),
            Some(candidate)
        );
        fs::remove_dir_all(root).expect("clean test directory");
    }

    #[test]
    fn config_update_is_atomic_semantic_and_preserves_unknown_entries() {
        let root = test_dir("config");
        let path = root.join("nested/mcp.json");
        let first = ensure_server_entry(&path, "/usr/bin/computer-use-linux").expect("create");
        assert_eq!(first, EnsureResult::Updated);
        assert_eq!(
            ensure_server_entry(&path, "/usr/bin/computer-use-linux").expect("unchanged"),
            EnsureResult::Unchanged
        );

        let mut config: Value =
            serde_json::from_slice(&fs::read(&path).expect("read config")).expect("parse config");
        config["unknown"] = json!({"nested": [1, 2, 3]});
        config["mcpServers"]["other"] =
            json!({"command": "/bin/other", "args": ["run"], "lifecycle": "eager"});
        config["mcpServers"][SERVER_NAME]["custom"] = json!("retained");
        fs::write(&path, serde_json::to_vec(&config).expect("encode config"))
            .expect("write config");

        assert_eq!(
            ensure_server_entry(&path, "/opt/computer-use-linux").expect("update"),
            EnsureResult::Updated
        );
        let after: Value =
            serde_json::from_slice(&fs::read(&path).expect("read updated config")).expect("parse");
        assert_eq!(after["unknown"]["nested"][2], 3);
        assert_eq!(after["mcpServers"]["other"]["lifecycle"], "eager");
        assert_eq!(after["mcpServers"][SERVER_NAME]["custom"], "retained");
        assert_eq!(
            after["mcpServers"][SERVER_NAME]["command"],
            "/opt/computer-use-linux"
        );
        assert_eq!(after["mcpServers"][SERVER_NAME]["args"], json!(["mcp"]));
        fs::remove_dir_all(root).expect("clean test directory");
    }

    #[test]
    fn config_refuses_malformed_non_object_and_oversized_files_without_clobbering() {
        let root = test_dir("unsafe-config");
        fs::create_dir_all(&root).expect("create test directory");
        for (name, contents) in [
            ("broken.json", b"{not json".as_slice()),
            ("array.json", b"[]".as_slice()),
            ("servers.json", br#"{"mcpServers": []}"#.as_slice()),
        ] {
            let path = root.join(name);
            fs::write(&path, contents).expect("write config");
            let error = ensure_server_entry(&path, "/bin/computer-use-linux").expect_err("refuse");
            assert!(matches!(error, ComputerUseError::InvalidConfiguration(_)));
            assert_eq!(fs::read(&path).expect("read untouched config"), contents);
        }

        let oversized = root.join("oversized.json");
        fs::write(&oversized, vec![b'x'; MAX_MCP_CONFIG_BYTES + 1]).expect("write oversized");
        assert!(matches!(
            ensure_server_entry(&oversized, "/bin/computer-use-linux"),
            Err(ComputerUseError::InvalidConfiguration(_))
        ));
        fs::remove_dir_all(root).expect("clean test directory");
    }

    #[test]
    fn names_schema_and_parameter_parser_cover_documented_shapes() {
        assert_eq!(prefixed_tool_name("doctor"), "computer_use_linux_doctor");
        assert_eq!(raw_tool_name("computer_use_linux_doctor"), "doctor");
        assert_eq!(raw_tool_name("doctor"), "doctor");

        let schema = json!({
            "type": "object",
            "properties": {
                "window_id": {"type": "string", "description": "Target window"},
                "max_width": {"type": "number"}
            },
            "required": ["window_id"]
        });
        let rendered = format_input_schema(&schema);
        assert!(rendered.contains("window_id (string, required): Target window"));
        assert!(rendered.contains("max_width (number, optional)"));

        assert_eq!(
            parse_proxy_request(&json!({"tool": "doctor", "args": {"verbose": true}}))
                .expect("call request"),
            ProxyRequest::Call {
                tool: "doctor".to_owned(),
                arguments: Map::from_iter([("verbose".to_owned(), Value::Bool(true))]),
            }
        );
        assert_eq!(
            parse_proxy_request(&json!({"search": "windows"})).expect("search request"),
            ProxyRequest::Search("windows".to_owned())
        );
        assert_eq!(
            parse_proxy_request(&json!({"server": SERVER_NAME})).expect("list request"),
            ProxyRequest::List {
                server: SERVER_NAME.to_owned()
            }
        );
        assert!(matches!(
            parse_proxy_request(&json!({"tool": "doctor", "args": []})),
            Err(ComputerUseError::InvalidInput(_))
        ));
    }

    #[test]
    fn bounded_reader_rejects_oversized_and_unterminated_frames() {
        let mut oversized = BufReader::new(Cursor::new(b"12345\n".to_vec()));
        assert!(read_bounded_line(&mut oversized, 4).is_err());

        let mut unterminated = BufReader::new(Cursor::new(b"{}".to_vec()));
        assert!(matches!(
            read_bounded_line(&mut unterminated, 10),
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof
        ));
    }

    fn helper_session() -> McpSession {
        let command = ServerCommand::new(
            env::current_exe().expect("current test executable"),
            [
                OsString::from("--exact"),
                OsString::from("computeruse::tests::helper_mcp_server"),
                OsString::from("--nocapture"),
            ],
        )
        .with_env(HELPER_ENV, "1");
        McpSession::from_command(command).with_timeouts(SessionTimeouts {
            initialization: Duration::from_secs(2),
            call: Duration::from_secs(2),
        })
    }

    fn write_rpc_line(writer: &mut impl Write, value: Value) {
        serde_json::to_writer(&mut *writer, &value).expect("encode helper response");
        writer.write_all(b"\n").expect("delimiter");
        writer.flush().expect("flush helper response");
    }

    fn reply(writer: &mut impl Write, id: &Value, result: Value) {
        write_rpc_line(
            writer,
            json!({"jsonrpc": "2.0", "id": id, "result": result}),
        );
    }

    /// Runs only in a re-executed test process and behaves like a small,
    /// deliberately noisy newline-delimited MCP server.
    #[test]
    fn helper_mcp_server() {
        if env::var_os(HELPER_ENV).is_none() {
            return;
        }
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut reader = BufReader::new(stdin.lock());
        let mut writer = stdout.lock();
        let mut line = String::new();

        loop {
            line.clear();
            if reader.read_line(&mut line).expect("read helper request") == 0 {
                return;
            }
            let Ok(request) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            let method = request
                .get("method")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = request.get("id").cloned();
            match method {
                "initialize" => {
                    let id = id.expect("initialize id");
                    let unrelated = id.as_u64().unwrap_or_default().saturating_add(10_000);
                    reply(
                        &mut writer,
                        &Value::from(unrelated),
                        json!({"ignored": true}),
                    );
                    write_rpc_line(
                        &mut writer,
                        json!({"jsonrpc": "2.0", "method": "notifications/progress"}),
                    );
                    reply(
                        &mut writer,
                        &id,
                        json!({
                            "protocolVersion": MCP_PROTOCOL_VERSION,
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "fake-computer-use", "version": "0"}
                        }),
                    );
                }
                "notifications/initialized" => write_rpc_line(
                    &mut writer,
                    json!({"jsonrpc": "2.0", "method": "notifications/progress"}),
                ),
                "tools/list" => {
                    let id = id.expect("tools/list id");
                    reply(
                        &mut writer,
                        &id,
                        json!({
                            "tools": [
                                {
                                    "name": "doctor",
                                    "description": "Readiness report",
                                    "annotations": {"readOnlyHint": true},
                                    "inputSchema": {"type": "object", "properties": {}}
                                },
                                {
                                    "name": "screenshot",
                                    "description": "Capture the screen as a bounded image"
                                },
                                {
                                    "name": "click",
                                    "description": "Click an element",
                                    "annotations": {
                                        "readOnlyHint": false,
                                        "destructiveHint": true
                                    }
                                },
                                {
                                    "name": "wait",
                                    "description": "Wait briefly for serialization testing"
                                }
                            ]
                        }),
                    );
                }
                "tools/call" => {
                    let id = id.expect("tools/call id");
                    let name = request
                        .get("params")
                        .and_then(|params| params.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let result = match name {
                        "doctor" => json!({
                            "content": [{"type": "text", "text": "{\"readiness\":{\"blockers\":[]}}"}]
                        }),
                        "screenshot" => json!({
                            "content": [
                                {"type": "text", "text": "{\"coordinate_width\":1920,\"scale\":1}"},
                                {"type": "image", "data": "cG5nLWJ5dGVz", "mimeType": "image/png"}
                            ]
                        }),
                        "click" => json!({
                            "isError": true,
                            "content": [{"type": "text", "text": "no such element"}]
                        }),
                        "wait" => {
                            thread::sleep(Duration::from_millis(100));
                            json!({"content": [{"type": "text", "text": "waited"}]})
                        }
                        "hang" => {
                            thread::sleep(Duration::from_millis(500));
                            json!({"content": [{"type": "text", "text": "too late"}]})
                        }
                        _ => json!({"content": []}),
                    };
                    reply(&mut writer, &id, result);
                }
                _ => {}
            }
        }
    }

    #[test]
    fn stdio_session_initializes_matches_ids_and_decodes_calls() {
        let session = helper_session();
        let tools = session.tools().expect("tools");
        assert_eq!(tools.len(), 4);
        assert_eq!(tools[0].name, "doctor");
        assert_eq!(
            tools[0].annotations.as_ref().unwrap().read_only_hint,
            Some(true)
        );
        assert_eq!(session.tools().expect("cached tools").len(), 4);

        let result = session.call("doctor", Map::new()).expect("doctor");
        assert!(
            result.content[0]
                .text
                .as_deref()
                .is_some_and(|text| text.contains("readiness"))
        );
        session.close();
    }

    #[test]
    fn timeout_stops_the_process_and_a_later_call_restarts_it() {
        let session = helper_session().with_timeouts(SessionTimeouts {
            initialization: Duration::from_secs(1),
            call: Duration::from_millis(40),
        });
        assert!(matches!(
            session.call("hang", Map::new()),
            Err(ComputerUseError::Timeout { operation }) if operation == "tools/call"
        ));
        assert!(
            session.call("doctor", Map::new()).is_ok(),
            "a timed-out session should be usable after lazy restart"
        );
        session.close();
    }

    #[test]
    fn proxy_lists_searches_calls_and_extracts_images() {
        let proxy = ComputerUseTool::new(helper_session());
        let listed = proxy
            .execute(&json!({"server": SERVER_NAME}))
            .expect("list tools");
        assert!(
            listed
                .text
                .contains("`computer_use_linux_doctor` [read-only]")
        );
        assert!(
            listed
                .text
                .contains("`computer_use_linux_click` [destructive]")
        );

        let searched = proxy
            .execute(&json!({"search": "doctor"}))
            .expect("search tools");
        assert!(searched.text.contains("### computer_use_linux_doctor"));

        let screenshot = proxy
            .execute(&json!({"tool": "computer_use_linux_screenshot"}))
            .expect("screenshot");
        assert!(screenshot.text.contains("coordinate_width"));
        assert_eq!(
            screenshot.images,
            vec![ImageResult {
                data: "cG5nLWJ5dGVz".to_owned(),
                mime_type: "image/png".to_owned()
            }]
        );
        assert_eq!(screenshot.details, Some(json!({"tool": "screenshot"})));

        assert!(matches!(
            proxy.execute(&json!({"tool": "click", "args": {"x": 1}})),
            Err(ComputerUseError::ToolFailed(message)) if message.contains("no such element")
        ));
        let unknown = proxy
            .execute(&json!({"tool": "computer_use_linux_missing"}))
            .expect("unknown is guidance");
        assert!(unknown.text.contains("not found"));
    }

    #[test]
    fn session_serializes_stateful_calls_across_clones() {
        let session = helper_session();
        let first = session.clone();
        let second = session.clone();
        let started = Instant::now();
        let left = thread::spawn(move || first.call("wait", Map::new()).expect("first call"));
        let right = thread::spawn(move || second.call("wait", Map::new()).expect("second call"));
        left.join().expect("first thread");
        right.join().expect("second thread");
        assert!(
            started.elapsed() >= Duration::from_millis(180),
            "two 100ms calls should not overlap"
        );
        session.close();
    }

    #[test]
    fn agent_adapter_is_sequential_and_emits_llm_images() {
        let tool = agent_tool(helper_session());
        assert_eq!(
            tool.execution_mode,
            Some(crate::agent::ToolExecutionMode::Sequential)
        );
        let result = (tool.execute)(
            crate::agent::CancellationToken::default(),
            "call-id".to_owned(),
            BTreeMap::from([("tool".to_owned(), Value::String("screenshot".to_owned()))]),
            Arc::new(|_: crate::agent::ToolResult| {}),
        )
        .expect("agent screenshot");
        assert!(result.content.iter().any(|block| {
            matches!(
                block,
                crate::llm::ContentBlock::Image(image) if image.mime_type == "image/png"
            )
        }));
    }
}
