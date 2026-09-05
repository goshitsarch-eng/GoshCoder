//! OmniRoute configuration, model discovery, and command-layer helpers.
//!
//! This module intentionally has no concrete HTTP implementation.  A host
//! supplies [`HttpTransport`] so this code can build and validate OmniRoute
//! requests without pretending that a network stack is available.  API keys
//! are accepted only for requests and are never persisted in [`Config`].

use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    net::Ipv6Addr,
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:20128";
pub const OMNI_PROVIDER_ID: &str = "omni";
pub const OMNI_PROVIDER_NAME: &str = "OmniRoute";
pub const OPENAI_COMPLETIONS_API: &str = "openai-completions";
pub const PROMPT_TOOLS_API: &str = "omni-prompt-tools";
pub const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
pub const DEFAULT_MAX_TOKENS: u64 = 16_384;
pub const MAX_CONFIG_BYTES: usize = 4 << 20;
pub const MAX_RESPONSE_BYTES: usize = 16 << 20;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub type Result<T> = std::result::Result<T, OmniRouteError>;
pub type ThinkingLevelMap = BTreeMap<String, Option<String>>;

/// Errors returned while reading, validating, synchronizing, or writing an
/// OmniRoute setup.
#[derive(Debug)]
pub enum OmniRouteError {
    InvalidUrl { value: String, reason: &'static str },
    ConfigTooLarge { limit: usize },
    ConfigNotRegularFile { path: PathBuf },
    InvalidConfig(serde_json::Error),
    ConfigEncode(serde_json::Error),
    ResponseTooLarge { limit: usize },
    InvalidModelsPayload(serde_json::Error),
    Transport(HttpTransportError),
    HttpStatus { status: u16, body: String },
    Io(io::Error),
    UnknownCommand(String),
    SetupInputRequired,
    Unconfigured { command: &'static str },
    ClockBeforeUnixEpoch,
    ClockOutOfRange,
}

impl OmniRouteError {
    /// Reports whether this error is an absent configuration file.
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::Io(error) if error.kind() == io::ErrorKind::NotFound)
    }
}

impl fmt::Display for OmniRouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUrl { value, reason } => write!(
                formatter,
                "invalid OmniRoute URL {value:?} (expected http:// or https://): {reason}"
            ),
            Self::ConfigTooLarge { limit } => {
                write!(formatter, "OmniRoute config exceeds {limit} bytes")
            }
            Self::ConfigNotRegularFile { path } => write!(
                formatter,
                "OmniRoute config is not a regular file: {}",
                path.display()
            ),
            Self::InvalidConfig(error) => write!(formatter, "invalid OmniRoute config: {error}"),
            Self::ConfigEncode(error) => write!(formatter, "encode OmniRoute config: {error}"),
            Self::ResponseTooLarge { limit } => {
                write!(formatter, "OmniRoute response exceeds {limit} bytes")
            }
            Self::InvalidModelsPayload(error) => {
                write!(formatter, "decode OmniRoute models: {error}")
            }
            Self::Transport(error) => write!(formatter, "OmniRoute transport: {error}"),
            Self::HttpStatus { status, body } => {
                write!(formatter, "OmniRoute returned {status}: {body}")
            }
            Self::Io(error) => error.fmt(formatter),
            Self::UnknownCommand(command) => write!(
                formatter,
                "unknown OmniRoute command {command:?}; use status, sync, setup, or dashboard"
            ),
            Self::SetupInputRequired => {
                formatter.write_str("OmniRoute setup requires URL and API-key input")
            }
            Self::Unconfigured { command } => {
                write!(
                    formatter,
                    "OmniRoute is unconfigured; run /omni setup before {command}"
                )
            }
            Self::ClockBeforeUnixEpoch => {
                formatter.write_str("system clock is before the Unix epoch")
            }
            Self::ClockOutOfRange => {
                formatter.write_str("system clock does not fit Unix milliseconds")
            }
        }
    }
}

impl Error for OmniRouteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(error)
            | Self::ConfigEncode(error)
            | Self::InvalidModelsPayload(error) => Some(error),
            Self::Transport(error) => Some(error),
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for OmniRouteError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The non-secret, durable part of an OmniRoute setup.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Config {
    #[serde(rename = "serverUrl")]
    pub server_url: String,
    #[serde(
        rename = "dashboardUrl",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub dashboard_url: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<Model>,
    #[serde(rename = "syncedAt", default, skip_serializing_if = "Option::is_none")]
    pub synced_at: Option<i64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            server_url: DEFAULT_SERVER_URL.to_owned(),
            dashboard_url: String::new(),
            models: Vec::new(),
            synced_at: None,
        }
    }
}

impl Config {
    pub fn new(server_url: impl AsRef<str>) -> Result<Self> {
        Ok(Self {
            server_url: normalize_server_url(server_url.as_ref())?,
            ..Self::default()
        })
    }

    /// Returns the OpenAI-compatible API root for this gateway.
    pub fn api_base_url(&self) -> String {
        format!("{}/v1", self.server_url.trim_end_matches('/'))
    }

    /// Returns the explicit dashboard URL, or the normalized server root.
    pub fn dashboard(&self) -> String {
        let dashboard = self.dashboard_url.trim();
        if dashboard.is_empty() {
            self.server_url.trim_end_matches('/').to_owned()
        } else {
            dashboard.trim_end_matches('/').to_owned()
        }
    }

    /// Reads a bounded JSON configuration and normalizes its server URL.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.file_type().is_file() {
            return Err(OmniRouteError::ConfigNotRegularFile {
                path: path.to_path_buf(),
            });
        }
        if metadata.len() > MAX_CONFIG_BYTES as u64 {
            return Err(OmniRouteError::ConfigTooLarge {
                limit: MAX_CONFIG_BYTES,
            });
        }

        let contents = read_bounded(&mut file, MAX_CONFIG_BYTES)?;
        if contents.len() > MAX_CONFIG_BYTES {
            return Err(OmniRouteError::ConfigTooLarge {
                limit: MAX_CONFIG_BYTES,
            });
        }

        let mut config: Self =
            serde_json::from_slice(&contents).map_err(OmniRouteError::InvalidConfig)?;
        config.server_url = normalize_server_url(&config.server_url)?;
        Ok(config)
    }

    /// Atomically saves this config. The temporary and final files are 0600 on
    /// Unix; a newly created parent directory is 0700 there as well.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let normalized = Self {
            server_url: normalize_server_url(&self.server_url)?,
            dashboard_url: self.dashboard_url.clone(),
            models: self.models.clone(),
            synced_at: self.synced_at,
        };
        let mut contents =
            serde_json::to_vec_pretty(&normalized).map_err(OmniRouteError::ConfigEncode)?;
        contents.push(b'\n');
        if contents.len() > MAX_CONFIG_BYTES {
            return Err(OmniRouteError::ConfigTooLarge {
                limit: MAX_CONFIG_BYTES,
            });
        }

        let directory = parent_directory(path);
        let directory_existed = directory.exists();
        fs::create_dir_all(directory)?;
        #[cfg(unix)]
        if !directory_existed {
            fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
        }

        let (temporary_path, mut temporary) = create_temporary_file(directory)?;
        let write_result = (|| -> Result<()> {
            temporary.write_all(&contents)?;
            temporary.sync_all()?;
            Ok(())
        })();
        drop(temporary);

        if let Err(error) = write_result {
            let _ = fs::remove_file(&temporary_path);
            return Err(error);
        }
        if let Err(error) = fs::rename(&temporary_path, path) {
            let _ = fs::remove_file(&temporary_path);
            return Err(error.into());
        }

        #[cfg(unix)]
        {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
            sync_parent_directory(directory)?;
        }
        Ok(())
    }

    /// Converts all stored gateway metadata into the live provider catalog
    /// shape used by an OpenAI-completions runtime.
    pub fn live_catalog(&self) -> LiveCatalog {
        LiveCatalog::from_config(self)
    }
}

pub fn default_config() -> Config {
    Config::default()
}

pub fn load_config(path: impl AsRef<Path>) -> Result<Config> {
    Config::load(path)
}

pub fn save_config(path: impl AsRef<Path>, config: &Config) -> Result<()> {
    config.save(path)
}

/// Validates a gateway root URL and removes an optional OpenAI `/v1` suffix,
/// query, fragment, and trailing slashes.
pub fn normalize_server_url(raw: &str) -> Result<String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(invalid_url(value, "a URL is required"));
    }
    if value
        .chars()
        .any(|character| character.is_control() || character.is_whitespace())
    {
        return Err(invalid_url(
            value,
            "whitespace and control characters are not allowed",
        ));
    }
    if !has_valid_percent_encoding(value) {
        return Err(invalid_url(value, "invalid percent encoding"));
    }

    let Some(colon) = value.find(':') else {
        return Err(invalid_url(value, "missing URL scheme"));
    };
    let scheme = &value[..colon];
    if !is_valid_scheme(scheme) {
        return Err(invalid_url(value, "invalid URL scheme"));
    }
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(invalid_url(value, "only HTTP and HTTPS URLs are supported"));
    }

    let after_scheme = &value[colon + 1..];
    let Some(after_authority_marker) = after_scheme.strip_prefix("//") else {
        return Err(invalid_url(value, "a host is required"));
    };
    let authority_end = after_authority_marker
        .find(['/', '?', '#'])
        .unwrap_or(after_authority_marker.len());
    let authority = &after_authority_marker[..authority_end];
    if let Some(reason) = invalid_authority_reason(authority) {
        return Err(invalid_url(value, reason));
    }

    let remainder = &after_authority_marker[authority_end..];
    let path_end = remainder.find(['?', '#']).unwrap_or(remainder.len());
    let path = &remainder[..path_end];
    if !path.is_empty() && !path.starts_with('/') {
        return Err(invalid_url(value, "invalid URL path"));
    }

    let path = path.trim_end_matches('/');
    let path = path.strip_suffix("/v1").unwrap_or(path);
    let path = path.trim_end_matches('/');
    Ok(format!(
        "{}://{}{}",
        scheme.to_ascii_lowercase(),
        authority,
        path
    ))
}

fn invalid_url(value: &str, reason: &'static str) -> OmniRouteError {
    OmniRouteError::InvalidUrl {
        value: value.to_owned(),
        reason,
    }
}

fn is_valid_scheme(value: &str) -> bool {
    let mut characters = value.chars();
    matches!(characters.next(), Some(character) if character.is_ascii_alphabetic())
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
}

fn has_valid_percent_encoding(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return false;
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    true
}

fn invalid_authority_reason(authority: &str) -> Option<&'static str> {
    if authority.is_empty() {
        return Some("a host is required");
    }
    if authority.contains('@') {
        return Some("embedded credentials are not allowed");
    }
    if authority.starts_with('[') {
        let Some(closing) = authority.find(']') else {
            return Some("invalid IPv6 host");
        };
        let host = &authority[1..closing];
        if host.is_empty() || host.parse::<Ipv6Addr>().is_err() {
            return Some("invalid IPv6 host");
        }
        let remainder = &authority[closing + 1..];
        if remainder.is_empty() {
            return None;
        }
        if !remainder.starts_with(':') || !is_valid_port(&remainder[1..]) {
            return Some("invalid port");
        }
        return None;
    }
    if authority.contains(['[', ']']) {
        return Some("invalid host");
    }

    let colon_count = authority.bytes().filter(|byte| *byte == b':').count();
    if colon_count > 1 {
        return Some("IPv6 hosts must be enclosed in brackets");
    }
    let (host, port) = match authority.split_once(':') {
        Some((host, port)) => (host, Some(port)),
        None => (authority, None),
    };
    if host.is_empty() {
        return Some("a host is required");
    }
    if let Some(port) = port {
        if !is_valid_port(port) {
            return Some("invalid port");
        }
    }
    None
}

fn is_valid_port(port: &str) -> bool {
    !port.is_empty()
        && port.bytes().all(|byte| byte.is_ascii_digit())
        && port.parse::<u16>().is_ok()
}

fn read_bounded(reader: &mut impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let mut contents = Vec::with_capacity(limit.min(64 * 1024));
    reader
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut contents)?;
    Ok(contents)
}

fn parent_directory(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn create_temporary_file(directory: &Path) -> Result<(PathBuf, File)> {
    for _ in 0..256 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".omniroute-{}-{sequence}.tmp", process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique OmniRoute config temporary file",
    )
    .into())
}

#[cfg(unix)]
fn sync_parent_directory(directory: &Path) -> Result<()> {
    File::open(directory)?.sync_all()?;
    Ok(())
}

/// Price metadata in dollars per million tokens.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelCostRates {
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub input: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub output: f64,
    #[serde(rename = "cacheRead", default, skip_serializing_if = "is_zero_f64")]
    pub cache_read: f64,
    #[serde(rename = "cacheWrite", default, skip_serializing_if = "is_zero_f64")]
    pub cache_write: f64,
}

/// A price tier selected above an input-token threshold.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelCostTier {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    #[serde(rename = "inputTokensAbove", default)]
    pub input_tokens_above: i64,
}

/// Optional model pricing metadata preserved from a synchronized config.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelCost {
    #[serde(flatten)]
    pub rates: ModelCostRates,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

impl ModelCost {
    pub fn has_values(&self) -> bool {
        self.rates.input != 0.0
            || self.rates.output != 0.0
            || self.rates.cache_read != 0.0
            || self.rates.cache_write != 0.0
            || !self.tiers.is_empty()
    }

    pub fn is_empty(&self) -> bool {
        !self.has_values()
    }
}

/// Metadata for one OmniRoute model. `tool_calling == Some(false)` opts into
/// the prompt-emulated tool adapter for web/chat-only models.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Model {
    #[serde(default)]
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(rename = "ownedBy", default, skip_serializing_if = "String::is_empty")]
    pub owned_by: String,
    #[serde(rename = "contextWindow", default, skip_serializing_if = "is_zero_i64")]
    pub context_window: i64,
    #[serde(rename = "maxTokens", default, skip_serializing_if = "is_zero_i64")]
    pub max_tokens: i64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub reasoning: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input: Vec<String>,
    #[serde(
        rename = "toolCalling",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_calling: Option<bool>,
    #[serde(default, skip_serializing_if = "ModelCost::is_empty")]
    pub cost: ModelCost,
    #[serde(
        rename = "thinkingLevelMap",
        default,
        skip_serializing_if = "thinking_level_map_is_empty"
    )]
    pub thinking_level_map: Option<ThinkingLevelMap>,
}

impl Model {
    /// Converts persisted metadata into the live model shape consumed by a
    /// catalog. Missing or non-positive limits use the same safe defaults as
    /// the Go implementation.
    pub fn live_model(&self, config: &Config) -> LiveModel {
        let input = if self.input.is_empty() {
            vec!["text".to_owned()]
        } else {
            self.input.clone()
        };
        LiveModel {
            id: self.id.clone(),
            name: first_nonempty(&self.name, &human_name(&self.id)).to_owned(),
            api: if self.tool_calling == Some(false) {
                PROMPT_TOOLS_API.to_owned()
            } else {
                OPENAI_COMPLETIONS_API.to_owned()
            },
            provider: OMNI_PROVIDER_ID.to_owned(),
            base_url: config.api_base_url(),
            reasoning: self.reasoning,
            thinking_level_map: self.thinking_level_map.clone().unwrap_or_default(),
            input,
            cost: self.cost.clone(),
            context_window: positive_or_default(self.context_window, DEFAULT_CONTEXT_WINDOW),
            max_tokens: positive_or_default(self.max_tokens, DEFAULT_MAX_TOKENS),
        }
    }
}

fn is_zero_f64(value: &f64) -> bool {
    *value == 0.0
}

fn is_zero_i64(value: &i64) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !value
}

fn thinking_level_map_is_empty(value: &Option<ThinkingLevelMap>) -> bool {
    match value {
        None => true,
        Some(map) => map.is_empty(),
    }
}

fn positive_or_default(value: i64, default: u64) -> u64 {
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// A runtime-ready OmniRoute model. Its JSON field names are compatible with
/// the application's general model catalog shape.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LiveModel {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
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
}

/// The dynamic provider entry produced from the persisted OmniRoute config.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LiveCatalog {
    pub provider_id: String,
    pub provider_name: String,
    pub base_url: String,
    pub models: Vec<LiveModel>,
}

impl LiveCatalog {
    pub fn from_config(config: &Config) -> Self {
        Self {
            provider_id: OMNI_PROVIDER_ID.to_owned(),
            provider_name: OMNI_PROVIDER_NAME.to_owned(),
            base_url: config.api_base_url(),
            models: config
                .models
                .iter()
                .map(|model| model.live_model(config))
                .collect(),
        }
    }
}

/// A catalog owner can implement this to atomically replace its dynamic Omni
/// provider snapshot after a successful config synchronization.
pub trait LiveCatalogSink {
    fn replace_omniroute_catalog(&mut self, catalog: LiveCatalog);
}

/// Decodes, filters, merges, and sorts a `/v1/models` response.
pub fn parse_models_payload(payload: &[u8]) -> Result<Vec<Model>> {
    if payload.len() > MAX_RESPONSE_BYTES {
        return Err(OmniRouteError::ResponseTooLarge {
            limit: MAX_RESPONSE_BYTES,
        });
    }

    #[derive(Deserialize)]
    struct ModelsEnvelope {
        #[serde(default)]
        data: Option<Vec<Value>>,
    }

    let envelope: ModelsEnvelope =
        serde_json::from_slice(payload).map_err(OmniRouteError::InvalidModelsPayload)?;
    let mut by_id = BTreeMap::new();
    for value in envelope.data.unwrap_or_default() {
        let Some(mut model) = decode_model(&value) else {
            continue;
        };
        if let Some(previous) = by_id.remove(&model.id) {
            model = merge_model(previous, model);
        }
        by_id.insert(model.id.clone(), model);
    }

    let mut models: Vec<Model> = by_id.into_values().collect();
    models.sort_by(|left, right| {
        let left_owner = if left.owned_by.is_empty() {
            "zz"
        } else {
            &left.owned_by
        };
        let right_owner = if right.owned_by.is_empty() {
            "zz"
        } else {
            &right.owned_by
        };
        left_owner
            .cmp(right_owner)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(models)
}

/// Decodes a single string or object entry in a `/v1/models` payload.
pub fn decode_model(value: &Value) -> Option<Model> {
    if let Some(id) = value.as_str() {
        if id.trim().is_empty() {
            return None;
        }
        return Some(Model {
            id: id.to_owned(),
            name: human_name(id),
            input: vec!["text".to_owned()],
            ..Model::default()
        });
    }

    let source: GatewayModel = serde_json::from_value(value.clone()).ok()?;
    let GatewayModel {
        id,
        name,
        owned_by,
        provider,
        kind,
        input_modalities,
        input,
        output_modalities,
        output,
        context_length,
        max_input_tokens,
        max_output_tokens,
        max_tokens,
        tool_calling,
        capabilities,
    } = source;
    let id = id.unwrap_or_default();
    if id.trim().is_empty() {
        return None;
    }
    let name = name.unwrap_or_default();
    let owned_by = owned_by.unwrap_or_default();
    let provider = provider.unwrap_or_default();
    let kind = kind.unwrap_or_default();
    let output = normalize_modalities(&first_nonempty_vec(output_modalities, output));
    if kind.eq_ignore_ascii_case("image") || (!output.is_empty() && !contains(&output, "text")) {
        return None;
    }

    let mut input = normalize_modalities(&first_nonempty_vec(input_modalities, input));
    if input.is_empty() {
        input.push("text".to_owned());
    }
    let capabilities = capabilities.unwrap_or_default();
    let mut tool_calling = tool_calling.or(capabilities.tool_calling);
    if is_web_synced([
        id.as_str(),
        name.as_str(),
        owned_by.as_str(),
        provider.as_str(),
    ]) {
        tool_calling = Some(false);
    }

    let context_length = context_length.unwrap_or(0);
    let max_input_tokens = max_input_tokens.unwrap_or(0);
    let max_output_tokens = max_output_tokens.unwrap_or(0);
    let max_tokens = max_tokens.unwrap_or(0);
    Some(Model {
        id: id.clone(),
        name: if name.is_empty() {
            human_name(&id)
        } else {
            name
        },
        owned_by,
        context_window: if context_length == 0 {
            max_input_tokens
        } else {
            context_length
        },
        max_tokens: if max_output_tokens == 0 {
            max_tokens
        } else {
            max_output_tokens
        },
        reasoning: capabilities.reasoning.unwrap_or(false)
            || capabilities.thinking.unwrap_or(false),
        input,
        tool_calling,
        ..Model::default()
    })
}

#[derive(Default, Deserialize)]
struct GatewayModel {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "owned_by")]
    owned_by: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default, rename = "input_modalities")]
    input_modalities: Option<Vec<String>>,
    #[serde(default)]
    input: Option<Vec<String>>,
    #[serde(default, rename = "output_modalities")]
    output_modalities: Option<Vec<String>>,
    #[serde(default)]
    output: Option<Vec<String>>,
    #[serde(default, rename = "context_length")]
    context_length: Option<i64>,
    #[serde(default, rename = "max_input_tokens")]
    max_input_tokens: Option<i64>,
    #[serde(default, rename = "max_output_tokens")]
    max_output_tokens: Option<i64>,
    #[serde(default, rename = "max_tokens")]
    max_tokens: Option<i64>,
    #[serde(default, rename = "tool_calling")]
    tool_calling: Option<bool>,
    #[serde(default)]
    capabilities: Option<GatewayCapabilities>,
}

#[derive(Default, Deserialize)]
struct GatewayCapabilities {
    #[serde(default)]
    reasoning: Option<bool>,
    #[serde(default)]
    thinking: Option<bool>,
    #[serde(default, rename = "tool_calling")]
    tool_calling: Option<bool>,
}

/// Keeps only the text/image modalities, lowercased and deduplicated in
/// encounter order.
pub fn normalize_modalities(values: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    for value in values {
        let value = value.trim().to_ascii_lowercase();
        if matches!(value.as_str(), "text" | "image") && !contains(&normalized, &value) {
            normalized.push(value);
        }
    }
    normalized
}

pub fn merge_model(mut old: Model, next: Model) -> Model {
    if !next.name.is_empty() {
        old.name = next.name;
    }
    if !next.owned_by.is_empty() {
        old.owned_by = next.owned_by;
    }
    if next.context_window != 0 {
        old.context_window = next.context_window;
    }
    if next.max_tokens != 0 {
        old.max_tokens = next.max_tokens;
    }
    old.reasoning |= next.reasoning;
    old.input = union_modalities(&old.input, &next.input);
    if next.tool_calling.is_some() {
        old.tool_calling = next.tool_calling;
    }
    if next.cost.has_values() {
        old.cost = next.cost;
    }
    if next.thinking_level_map.is_some() {
        old.thinking_level_map = next.thinking_level_map;
    }
    old
}

pub fn human_name(id: &str) -> String {
    let name = id.rsplit('/').next().unwrap_or(id);
    name.replace(['-', '_'], " ")
        .split_whitespace()
        .map(capitalize_word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn capitalize_word(word: &str) -> String {
    let mut characters = word.chars();
    match characters.next() {
        Some(first) => first.to_uppercase().collect::<String>() + characters.as_str(),
        None => String::new(),
    }
}

fn first_nonempty<'a>(first: &'a str, second: &'a str) -> &'a str {
    if first.is_empty() { second } else { first }
}

fn first_nonempty_vec(first: Option<Vec<String>>, second: Option<Vec<String>>) -> Vec<String> {
    let first = first.unwrap_or_default();
    if first.is_empty() {
        second.unwrap_or_default()
    } else {
        first
    }
}

fn union_modalities(left: &[String], right: &[String]) -> Vec<String> {
    let mut combined = Vec::with_capacity(left.len() + right.len());
    combined.extend(left.iter().cloned());
    combined.extend(right.iter().cloned());
    normalize_modalities(&combined)
}

fn contains(values: &[String], target: &str) -> bool {
    values.iter().any(|value| value == target)
}

fn is_web_synced<'a>(values: impl IntoIterator<Item = &'a str>) -> bool {
    values.into_iter().any(|value| {
        value
            .as_bytes()
            .windows(4)
            .any(|window| window.eq_ignore_ascii_case(b"-web"))
    })
}

/// An abstract request method. This module only needs GET today, but keeping
/// it typed prevents a transport adapter from inferring semantics from text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HttpMethod {
    Get,
}

impl HttpMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
        }
    }
}

/// A fully built request for an injected HTTP transport. It deliberately does
/// not implement `Debug`, so accidental logs do not expose Authorization.
#[derive(Clone, Eq, PartialEq)]
pub struct HttpRequest {
    pub method: HttpMethod,
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Error text returned by a concrete transport adapter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpTransportError {
    message: String,
}

impl HttpTransportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for HttpTransportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HttpTransportError {}

/// Supplies actual HTTP I/O. A host can implement this with its chosen HTTP
/// stack; no no-op, mock, or network-emulating implementation is provided.
pub trait HttpTransport {
    fn execute(
        &self,
        request: HttpRequest,
    ) -> std::result::Result<HttpResponse, HttpTransportError>;
}

/// Client for the public OpenAI-compatible OmniRoute API.
pub struct Client<'a, T: HttpTransport + ?Sized> {
    pub config: Config,
    api_key: String,
    transport: &'a T,
}

impl<'a, T: HttpTransport + ?Sized> Client<'a, T> {
    pub fn new(config: Config, api_key: impl Into<String>, transport: &'a T) -> Self {
        Self {
            config,
            api_key: api_key.into(),
            transport,
        }
    }

    /// Probes `/v1/models` and returns only transport/protocol errors.
    pub fn health(&self) -> Result<()> {
        self.request(HttpMethod::Get, "/models", None).map(|_| ())
    }

    /// Fetches, normalizes, deduplicates, and sorts the public model catalog.
    pub fn fetch_models(&self) -> Result<Vec<Model>> {
        let payload = self.request(HttpMethod::Get, "/models", None)?;
        parse_models_payload(&payload)
    }

    fn request(
        &self,
        method: HttpMethod,
        endpoint: &str,
        body: Option<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let mut headers = BTreeMap::from([("Accept".to_owned(), "application/json".to_owned())]);
        if body.is_some() {
            headers.insert("Content-Type".to_owned(), "application/json".to_owned());
        }
        let api_key = self.api_key.trim();
        if !api_key.is_empty() && api_key != "omniroute-public" {
            headers.insert("Authorization".to_owned(), format!("Bearer {api_key}"));
        }

        let response = self
            .transport
            .execute(HttpRequest {
                method,
                url: format!("{}{}", self.config.api_base_url(), endpoint),
                headers,
                body,
            })
            .map_err(OmniRouteError::Transport)?;
        if response.body.len() > MAX_RESPONSE_BYTES {
            return Err(OmniRouteError::ResponseTooLarge {
                limit: MAX_RESPONSE_BYTES,
            });
        }
        if !(200..300).contains(&response.status) {
            let body = String::from_utf8_lossy(&response.body);
            return Err(OmniRouteError::HttpStatus {
                status: response.status,
                body: truncate(body.trim(), 500),
            });
        }
        Ok(response.body)
    }
}

fn truncate(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_owned();
    }
    let mut boundary = limit;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    format!("{}...", &value[..boundary])
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Down(String),
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => formatter.write_str("healthy"),
            Self::Down(error) => write!(formatter, "DOWN ({error})"),
        }
    }
}

/// The data rendered by `/omni status`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StatusReport {
    pub configured: bool,
    pub health: Option<HealthStatus>,
    pub server_url: Option<String>,
    pub model_count: usize,
    pub dashboard_url: Option<String>,
}

impl StatusReport {
    pub fn render(&self) -> String {
        if !self.configured {
            return "OmniRoute is unconfigured. Run /omni setup.".to_owned();
        }
        let health = self
            .health
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "DOWN (health unavailable)".to_owned());
        format!(
            "OmniRoute: {health}\nServer: {}\nModels: {}\nDashboard: {}",
            self.server_url.as_deref().unwrap_or_default(),
            self.model_count,
            self.dashboard_url.as_deref().unwrap_or_default(),
        )
    }
}

/// Implements the non-fatal health behavior of `/omni status`.
pub fn status_command<T: HttpTransport + ?Sized>(
    config_path: impl AsRef<Path>,
    api_key: &str,
    transport: &T,
) -> Result<StatusReport> {
    let config = match Config::load(config_path) {
        Ok(config) => config,
        Err(error) if error.is_not_found() => {
            return Ok(StatusReport {
                configured: false,
                health: None,
                server_url: None,
                model_count: 0,
                dashboard_url: None,
            });
        }
        Err(error) => return Err(error),
    };
    let health = match Client::new(config.clone(), api_key, transport).health() {
        Ok(()) => HealthStatus::Healthy,
        Err(error) => HealthStatus::Down(error.to_string()),
    };
    Ok(StatusReport {
        configured: true,
        health: Some(health),
        server_url: Some(config.server_url.clone()),
        model_count: config.models.len(),
        dashboard_url: Some(config.dashboard()),
    })
}

/// Implements `/omni dashboard` without opening a browser.
pub fn dashboard_command(config_path: impl AsRef<Path>) -> Result<String> {
    let config = load_configured(config_path.as_ref(), "dashboard")?;
    Ok(format!(
        "OmniRoute Dashboard: {}\nOpen it to manage combos, providers, usage, and request logs.",
        config.dashboard()
    ))
}

/// Result of a successful `/omni sync`.
#[derive(Clone, Debug, PartialEq)]
pub struct SyncOutcome {
    pub previous_model_count: usize,
    pub model_count: usize,
    pub config: Config,
    pub catalog: LiveCatalog,
}

impl SyncOutcome {
    pub fn render(&self) -> String {
        format!(
            "Synced {} OmniRoute models (was {}). They are available under /model immediately.",
            self.model_count, self.previous_model_count
        )
    }

    pub fn publish_to<S: LiveCatalogSink>(&self, sink: &mut S) {
        sink.replace_omniroute_catalog(self.catalog.clone());
    }
}

/// Fetches a live gateway catalog, saves it, and returns its runtime
/// conversion. The config is unchanged if the request or payload fails.
pub fn sync_command<T: HttpTransport + ?Sized>(
    config_path: impl AsRef<Path>,
    api_key: &str,
    transport: &T,
    synced_at_millis: i64,
) -> Result<SyncOutcome> {
    let config_path = config_path.as_ref();
    let mut config = load_configured(config_path, "sync")?;
    let models = Client::new(config.clone(), api_key, transport).fetch_models()?;
    let previous_model_count = config.models.len();
    config.models = models;
    config.synced_at = Some(synced_at_millis);
    config.save(config_path)?;
    let catalog = config.live_catalog();
    Ok(SyncOutcome {
        previous_model_count,
        model_count: config.models.len(),
        config,
        catalog,
    })
}

pub fn sync_command_now<T: HttpTransport + ?Sized>(
    config_path: impl AsRef<Path>,
    api_key: &str,
    transport: &T,
) -> Result<SyncOutcome> {
    sync_command(config_path, api_key, transport, unix_millis_now()?)
}

/// Inputs collected by a CLI or fullscreen setup wizard. The API key remains
/// separate from config and is returned only as `credential_to_store`.
#[derive(Clone, Eq, PartialEq)]
pub struct SetupRequest {
    pub server_url: String,
    pub api_key: String,
    pub allow_default: bool,
}

/// A completed setup. Store `credential_to_store` in the application's
/// credential store, never in `omniroute.json`.
#[derive(Clone, PartialEq)]
pub struct SetupOutcome {
    pub config: Config,
    pub credential_to_store: String,
}

impl SetupOutcome {
    pub fn render(&self) -> &'static str {
        "OmniRoute setup complete. Run /omni sync to load models."
    }
}

/// Validates a setup against the live gateway before writing config. A caller
/// owns durable credential storage and must persist `credential_to_store`
/// separately after this succeeds.
pub fn setup_command<T: HttpTransport + ?Sized>(
    config_path: impl AsRef<Path>,
    request: SetupRequest,
    transport: &T,
) -> Result<SetupOutcome> {
    let config_path = config_path.as_ref();
    let requested_url = if request.server_url.trim().is_empty() && request.allow_default {
        DEFAULT_SERVER_URL
    } else {
        request.server_url.as_str()
    };
    let server_url = normalize_server_url(requested_url)?;
    let mut config = Config {
        server_url: server_url.clone(),
        ..Config::default()
    };

    if let Ok(current) = Config::load(config_path) {
        if current.server_url == server_url {
            config.models = current.models;
            config.synced_at = current.synced_at;
        }
    }
    Client::new(config.clone(), request.api_key.as_str(), transport).health()?;
    config.save(config_path)?;

    let api_key = request.api_key.trim();
    Ok(SetupOutcome {
        config,
        credential_to_store: if api_key.is_empty() {
            "omniroute-public".to_owned()
        } else {
            api_key.to_owned()
        },
    })
}

/// The command word parsed from `/omni [command]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliCommand {
    Status,
    Dashboard,
    Sync,
    Setup,
}

impl CliCommand {
    /// Parses the first argument. Missing or blank commands default to status.
    pub fn parse<I, S>(arguments: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let command = arguments
            .into_iter()
            .next()
            .map(|argument| argument.as_ref().trim().to_ascii_lowercase())
            .unwrap_or_else(|| "status".to_owned());
        match command.as_str() {
            "" | "status" => Ok(Self::Status),
            "dashboard" | "dash" => Ok(Self::Dashboard),
            "sync" => Ok(Self::Sync),
            "setup" => Ok(Self::Setup),
            _ => Err(OmniRouteError::UnknownCommand(command)),
        }
    }
}

/// A command invocation with setup inputs already gathered by the caller.
pub enum CommandInvocation {
    Status,
    Dashboard,
    Sync,
    Setup(SetupRequest),
}

impl From<CliCommand> for CommandInvocation {
    fn from(command: CliCommand) -> Self {
        match command {
            CliCommand::Status => Self::Status,
            CliCommand::Dashboard => Self::Dashboard,
            CliCommand::Sync => Self::Sync,
            CliCommand::Setup => Self::Setup(SetupRequest {
                server_url: String::new(),
                api_key: String::new(),
                allow_default: false,
            }),
        }
    }
}

/// Dependencies supplied by a CLI frontend. They are explicit to keep clock,
/// transport, and credential acquisition outside this standalone module.
pub struct CommandContext<'a, T: HttpTransport + ?Sized> {
    pub config_path: &'a Path,
    pub api_key: &'a str,
    pub transport: &'a T,
    pub synced_at_millis: i64,
}

/// Result of a CLI-oriented command without writing to stdout or stderr.
pub enum CommandOutput {
    Status(StatusReport),
    Dashboard(String),
    Sync(SyncOutcome),
    Setup(SetupOutcome),
}

impl CommandOutput {
    pub fn render(&self) -> String {
        match self {
            Self::Status(status) => status.render(),
            Self::Dashboard(output) => output.clone(),
            Self::Sync(outcome) => outcome.render(),
            Self::Setup(outcome) => outcome.render().to_owned(),
        }
    }
}

/// Dispatches an already-parsed command. The caller supplies the real
/// transport and any setup input; this function performs no terminal I/O.
pub fn execute_command<T: HttpTransport + ?Sized>(
    context: CommandContext<'_, T>,
    invocation: CommandInvocation,
) -> Result<CommandOutput> {
    match invocation {
        CommandInvocation::Status => {
            status_command(context.config_path, context.api_key, context.transport)
                .map(CommandOutput::Status)
        }
        CommandInvocation::Dashboard => {
            dashboard_command(context.config_path).map(CommandOutput::Dashboard)
        }
        CommandInvocation::Sync => sync_command(
            context.config_path,
            context.api_key,
            context.transport,
            context.synced_at_millis,
        )
        .map(CommandOutput::Sync),
        CommandInvocation::Setup(request) => {
            if request.server_url.trim().is_empty() && !request.allow_default {
                return Err(OmniRouteError::SetupInputRequired);
            }
            setup_command(context.config_path, request, context.transport).map(CommandOutput::Setup)
        }
    }
}

/// Returns the current Unix timestamp in milliseconds for callers that do not
/// need deterministic timestamp injection.
pub fn unix_millis_now() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| OmniRouteError::ClockBeforeUnixEpoch)?;
    i64::try_from(duration.as_millis()).map_err(|_| OmniRouteError::ClockOutOfRange)
}

fn load_configured(path: &Path, command: &'static str) -> Result<Config> {
    match Config::load(path) {
        Ok(config) => Ok(config),
        Err(error) if error.is_not_found() => Err(OmniRouteError::Unconfigured { command }),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_directory(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after Unix epoch")
            .as_nanos();
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "goshcoder-omniroute-{label}-{}-{nanos}-{sequence}",
            process::id()
        ));
        fs::create_dir_all(&path).expect("create test directory");
        path
    }

    #[test]
    fn config_validation_normalizes_and_persists_atomically() {
        assert_eq!(
            normalize_server_url(" https://example.com/gateway/v1/?ignored=true#fragment ")
                .expect("normalize URL"),
            "https://example.com/gateway"
        );
        for invalid in [
            "",
            "gateway.example",
            "ftp://gateway.example",
            "https:///v1",
            "https://user:secret@gateway.example",
            "https://gateway.example:70000",
        ] {
            assert!(
                normalize_server_url(invalid).is_err(),
                "{invalid:?} should be rejected"
            );
        }

        let directory = test_directory("config");
        let path = directory.join("omniroute.json");
        let config = Config {
            server_url: "https://example.com/gateway/v1".to_owned(),
            models: vec![Model {
                id: "combo/code".to_owned(),
                input: vec!["text".to_owned()],
                ..Model::default()
            }],
            ..Config::default()
        };
        config.save(&path).expect("save config");
        Config {
            server_url: "https://example.com/replaced".to_owned(),
            ..config.clone()
        }
        .save(&path)
        .expect("atomically replace config");

        let loaded = Config::load(&path).expect("load config");
        assert_eq!(loaded.server_url, "https://example.com/replaced");
        assert_eq!(loaded.models.len(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            assert_eq!(
                fs::metadata(&path)
                    .expect("config metadata")
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn model_conversion_preserves_metadata_and_selects_prompt_tools() {
        let config = Config {
            server_url: "https://gateway.example".to_owned(),
            ..Config::default()
        };
        let model = Model {
            id: "vendor/chat-web".to_owned(),
            tool_calling: Some(false),
            reasoning: true,
            thinking_level_map: Some(BTreeMap::from([(
                "high".to_owned(),
                Some("extended".to_owned()),
            )])),
            cost: ModelCost {
                rates: ModelCostRates {
                    input: 1.5,
                    output: 2.5,
                    ..ModelCostRates::default()
                },
                ..ModelCost::default()
            },
            ..Model::default()
        };

        let live = model.live_model(&config);
        assert_eq!(live.id, "vendor/chat-web");
        assert_eq!(live.name, "Chat Web");
        assert_eq!(live.api, PROMPT_TOOLS_API);
        assert_eq!(live.provider, OMNI_PROVIDER_ID);
        assert_eq!(live.base_url, "https://gateway.example/v1");
        assert_eq!(live.input, vec!["text"]);
        assert_eq!(live.context_window, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(live.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(live.reasoning);
        assert_eq!(
            live.thinking_level_map.get("high"),
            Some(&Some("extended".to_owned()))
        );
        assert_eq!(live.cost.rates.input, 1.5);
    }

    #[test]
    fn payload_parsing_filters_merges_and_normalizes_models() {
        let models = parse_models_payload(
            br#"{
                "data": [
                    {
                        "id": "cgpt-web/code",
                        "name": "Code Web",
                        "owned_by": "web",
                        "input_modalities": ["TEXT", "IMAGE", "audio", "text"],
                        "context_length": 400000,
                        "capabilities": {"reasoning": true}
                    },
                    {
                        "id": "cgpt-web/code",
                        "owned_by": "web",
                        "max_output_tokens": 32000
                    },
                    {
                        "id": "image-only",
                        "type": "image",
                        "output_modalities": ["image"]
                    },
                    {
                        "id": "direct-no-tools",
                        "tool_calling": false,
                        "output_modalities": ["text"]
                    },
                    "plain-model"
                ]
            }"#,
        )
        .expect("parse models payload");

        assert_eq!(models.len(), 3);
        let web = models
            .iter()
            .find(|model| model.id == "cgpt-web/code")
            .expect("merged web model");
        assert_eq!(web.context_window, 400_000);
        assert_eq!(web.max_tokens, 32_000);
        assert!(web.reasoning);
        assert_eq!(web.input, vec!["text", "image"]);
        assert_eq!(web.tool_calling, Some(false));
        assert_eq!(
            models
                .iter()
                .find(|model| model.id == "direct-no-tools")
                .and_then(|model| model.tool_calling),
            Some(false)
        );
        assert_eq!(
            models
                .iter()
                .find(|model| model.id == "plain-model")
                .map(|model| model.name.as_str()),
            Some("Plain Model")
        );
        assert_eq!(models[0].id, "cgpt-web/code");
    }
}
