//! Tailscale Aperture configuration and routing primitives.
//!
//! This is a standalone port of the non-UI Aperture core.  It intentionally
//! depends only on the crate's LLM model representation so a later runtime
//! integration can load configuration, build a catalog, apply proxy routes,
//! and rewrite requests without reimplementing the compatibility rules.

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    error::Error as StdError,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::Path,
    process,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

use crate::llm;

/// The provider ID reserved for the gateway's dedicated model catalog.
pub const DEDICATED_PROVIDER_ID: &str = "aperture";
/// Provenance sent to Aperture for every routed request.
pub const APERTURE_REFERER: &str = "https://github.com/goshitsarch-eng/goshcoder";
/// Maximum accepted `extensions/aperture.json` size.
pub const MAX_CONFIG_BYTES: usize = 4 << 20;
/// Maximum accepted gateway response and cache size.
pub const MAX_RESPONSE_BYTES: usize = 16 << 20;
/// Pinned connector tools become expensive in the system prompt past this count.
pub const CONTEXT_COST_WARNING_THRESHOLD: usize = 10;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A gateway API usable by a pi-compatible model client.
pub type RoutableApi = String;

/// Errors from Aperture persistence, gateway access, and pin validation.
#[derive(Debug)]
pub enum ApertureError {
    Io {
        operation: &'static str,
        source: io::Error,
    },
    ConfigNotRegularFile,
    CacheNotRegularFile,
    ConfigTooLarge,
    CacheTooLarge,
    GatewayResponseTooLarge,
    InvalidConfig(serde_json::Error),
    InvalidCache(serde_json::Error),
    InvalidGatewayResponse(String),
    SerializeConfig(serde_json::Error),
    SerializeCache(serde_json::Error),
    HttpClient(reqwest::Error),
    Request(reqwest::Error),
    Http(HttpError),
    ToolNotFound(String),
    ToolAlreadyPinned(String),
    ToolNotPinned(String),
}

impl fmt::Display for ApertureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, source } => write!(formatter, "failed to {operation}: {source}"),
            Self::ConfigNotRegularFile => {
                formatter.write_str("aperture config is not a regular file")
            }
            Self::CacheNotRegularFile => {
                formatter.write_str("aperture cache is not a regular file")
            }
            Self::ConfigTooLarge => write!(
                formatter,
                "aperture config exceeds {MAX_CONFIG_BYTES} bytes"
            ),
            Self::CacheTooLarge => write!(
                formatter,
                "aperture cache exceeds {MAX_RESPONSE_BYTES} bytes"
            ),
            Self::GatewayResponseTooLarge => {
                write!(
                    formatter,
                    "aperture response exceeds {MAX_RESPONSE_BYTES} bytes"
                )
            }
            Self::InvalidConfig(source) => write!(formatter, "invalid aperture config: {source}"),
            Self::InvalidCache(source) => write!(formatter, "invalid aperture cache: {source}"),
            Self::InvalidGatewayResponse(message) => {
                write!(formatter, "invalid Aperture gateway response: {message}")
            }
            Self::SerializeConfig(source) => {
                write!(formatter, "failed to serialize aperture config: {source}")
            }
            Self::SerializeCache(source) => {
                write!(formatter, "failed to serialize aperture cache: {source}")
            }
            Self::HttpClient(source) => {
                write!(formatter, "failed to create Aperture HTTP client: {source}")
            }
            Self::Request(source) => write!(formatter, "Aperture request failed: {source}"),
            Self::Http(error) => error.fmt(formatter),
            Self::ToolNotFound(name) => write!(formatter, "tool {name:?} not found on the gateway"),
            Self::ToolAlreadyPinned(name) => write!(formatter, "{name} is already pinned"),
            Self::ToolNotPinned(name) => write!(formatter, "{name} is not pinned"),
        }
    }
}

impl StdError for ApertureError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidConfig(source)
            | Self::InvalidCache(source)
            | Self::SerializeConfig(source)
            | Self::SerializeCache(source) => Some(source),
            Self::HttpClient(source) | Self::Request(source) => Some(source),
            Self::Http(error) => Some(error),
            Self::ConfigNotRegularFile
            | Self::CacheNotRegularFile
            | Self::ConfigTooLarge
            | Self::CacheTooLarge
            | Self::GatewayResponseTooLarge
            | Self::InvalidGatewayResponse(_)
            | Self::ToolNotFound(_)
            | Self::ToolAlreadyPinned(_)
            | Self::ToolNotPinned(_) => None,
        }
    }
}

/// A non-success response from an Aperture endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HttpError {
    pub method: String,
    pub path: String,
    pub status: u16,
}

impl fmt::Display for HttpError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = reqwest::StatusCode::from_u16(self.status)
            .ok()
            .and_then(|status| status.canonical_reason())
            .unwrap_or("Unknown Status");
        write!(
            formatter,
            "[Aperture] {} {}: -> {} {}",
            self.method, self.path, self.status, reason
        )
    }
}

impl StdError for HttpError {}

pub type Result<T> = std::result::Result<T, ApertureError>;

/// One configured upstream provider to proxy through Aperture.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ProxiedProviderConfig {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(rename = "shouldCheckGatewayModels", skip_serializing_if = "is_false")]
    pub should_check_gateway_models: bool,
    #[serde(rename = "keepGatewayModelsOnly", skip_serializing_if = "is_false")]
    pub keep_gateway_models_only: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub api: RoutableApi,
}

impl ProxiedProviderConfig {
    /// Providers are enabled unless their persisted optional boolean is false.
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }
}

/// One gateway provider selected for the dedicated Aperture provider.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct DedicatedProviderConfig {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    pub enabled: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub api: RoutableApi,
}

/// A connector tool selected to register as a first-class tool.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct PinnedConnectorTool {
    #[serde(rename = "connectorId")]
    pub connector_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
}

/// Connector capability settings retained in `aperture.json`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ConnectorsConfig {
    #[serde(skip_serializing_if = "is_false")]
    pub enabled: bool,
    #[serde(rename = "pinnedTools", skip_serializing_if = "option_vec_is_empty")]
    pub pinned_tools: Option<Vec<PinnedConnectorTool>>,
    #[serde(rename = "discoveryTools", skip_serializing_if = "Option::is_none")]
    pub discovery_tools: Option<bool>,
}

/// Onboarding state retained in `aperture.json`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct OnboardingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// Proxy capability settings. `upstreamProviders` intentionally serializes
/// even when absent, matching Go's non-`omitempty` JSON tag.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ProxyConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(rename = "upstreamProviders")]
    pub upstream_providers: Option<Vec<ProxiedProviderConfig>>,
}

/// Dedicated capability settings. `cachedModels` is accepted only for the
/// content-gated migration and never emitted after [`Config::migrate`].
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct DedicatedConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    pub providers: Option<Vec<DedicatedProviderConfig>>,
    #[serde(rename = "cachedModels", skip_serializing_if = "option_vec_is_empty")]
    pub cached_models: Option<Vec<Value>>,
}

/// The persisted pi-compatible `extensions/aperture.json` schema.
///
/// Fields ending in `legacy_*` deserialize the old pre-v0.8 shapes solely so
/// [`Config::migrate`] can normalize them. They are deliberately never
/// serialized.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct Config {
    #[serde(rename = "$schema", skip_serializing_if = "String::is_empty")]
    pub schema: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub version: String,
    #[serde(rename = "baseUrl", skip_serializing_if = "String::is_empty")]
    pub base_url: String,
    #[serde(rename = "onboardingDone", skip_serializing_if = "Option::is_none")]
    pub onboarding_done: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onboarding: Option<OnboardingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<ProxyConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dedicated: Option<DedicatedConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connectors: Option<ConnectorsConfig>,

    #[serde(rename = "mode", skip_serializing)]
    pub legacy_mode: String,
    #[serde(rename = "providers", skip_serializing)]
    pub legacy_providers: Option<Vec<String>>,
    #[serde(rename = "checkGatewayModels", skip_serializing)]
    pub legacy_check_gateway_models: Option<Vec<String>>,
    #[serde(rename = "apertureProvider", skip_serializing)]
    pub legacy_aperture_provider: Option<bool>,
}

/// The fully defaulted Aperture configuration used by catalog and request code.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Resolved {
    pub base_url: String,
    pub onboarding_done: bool,
    pub onboarding_enabled: bool,
    pub proxy_enabled: bool,
    pub upstream_providers: Vec<ProxiedProviderConfig>,
    pub dedicated_enabled: bool,
    pub dedicated_providers: Vec<DedicatedProviderConfig>,
    pub connectors_enabled: bool,
    pub pinned_tools: Vec<PinnedConnectorTool>,
    pub discovery_tools: bool,
}

impl Config {
    /// Applies the original three content-gated migrations in release order.
    ///
    /// The returned flag says whether any migration changed the in-memory
    /// representation; callers may use it to decide whether a save is useful.
    pub fn migrate(&mut self) -> bool {
        let mut changed = false;

        // 001 legacy -> v0.6: providers/checkGatewayModels become proxy
        // settings; a previously configured URL implies onboarding completed.
        let has_legacy_lists =
            self.legacy_providers.is_some() || self.legacy_check_gateway_models.is_some();
        if has_legacy_lists
            || self.legacy_aperture_provider.is_some()
            || (self.onboarding_done.is_none() && !self.base_url.is_empty())
        {
            if has_legacy_lists {
                let checked = self
                    .legacy_check_gateway_models
                    .take()
                    .unwrap_or_default()
                    .into_iter()
                    .collect::<BTreeSet<_>>();
                let upstream = self
                    .legacy_providers
                    .take()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|id| ProxiedProviderConfig {
                        should_check_gateway_models: checked.contains(&id),
                        id,
                        ..ProxiedProviderConfig::default()
                    })
                    .collect();
                let proxy = self.proxy.get_or_insert_with(ProxyConfig::default);
                proxy.enabled = Some(true);
                proxy.upstream_providers = Some(upstream);
            }
            if let Some(enabled) = self.legacy_aperture_provider.take() {
                self.dedicated
                    .get_or_insert_with(DedicatedConfig::default)
                    .enabled = Some(enabled);
            }
            if self.onboarding_done.is_none() && !self.base_url.is_empty() {
                self.onboarding_done = Some(true);
            }
            self.version = "0.6.0".to_owned();
            changed = true;
        }

        // 002 mode -> capabilities. It intentionally runs after 001 so an
        // explicit legacy `mode` wins over `apertureProvider`.
        if !self.legacy_mode.is_empty() {
            let proxy = self.proxy.get_or_insert_with(ProxyConfig::default);
            let dedicated = self.dedicated.get_or_insert_with(DedicatedConfig::default);
            match self.legacy_mode.as_str() {
                "proxy" => {
                    proxy.enabled = Some(true);
                    dedicated.enabled = Some(false);
                }
                "dedicated" => {
                    dedicated.enabled = Some(true);
                    proxy.enabled = Some(false);
                }
                _ => {}
            }
            self.legacy_mode.clear();
            self.version = "0.7.0".to_owned();
            changed = true;
        }

        // 003 makes the two capability blocks explicit and drops the old
        // cached catalog embedded in dedicated settings.
        let needs_normalization = self
            .proxy
            .as_ref()
            .is_none_or(|proxy| proxy.enabled.is_none() || proxy.upstream_providers.is_none())
            || self.dedicated.as_ref().is_none_or(|dedicated| {
                dedicated.enabled.is_none()
                    || dedicated.providers.is_none()
                    || dedicated.cached_models.is_some()
            });
        if needs_normalization {
            let proxy = self.proxy.get_or_insert_with(ProxyConfig::default);
            if proxy.enabled.is_none() {
                proxy.enabled = Some(false);
            }
            if proxy.upstream_providers.is_none() {
                proxy.upstream_providers = Some(Vec::new());
            }

            let dedicated = self.dedicated.get_or_insert_with(DedicatedConfig::default);
            if dedicated.enabled.is_none() {
                dedicated.enabled = Some(true);
            }
            if dedicated.providers.is_none() {
                dedicated.providers = Some(Vec::new());
            }
            dedicated.cached_models = None;

            self.version = "0.8.0".to_owned();
            changed = true;
        }

        changed
    }

    /// Resolves documented defaults: dedicated on, proxy and connectors off,
    /// discovery tools on, and onboarding on until it is explicitly complete.
    pub fn resolve(&self) -> Resolved {
        let mut resolved = Resolved {
            base_url: self.base_url.clone(),
            dedicated_enabled: true,
            discovery_tools: true,
            ..Resolved::default()
        };
        if let Some(done) = self.onboarding_done {
            resolved.onboarding_done = done;
        }
        resolved.onboarding_enabled = !resolved.onboarding_done;
        if let Some(enabled) = self
            .onboarding
            .as_ref()
            .and_then(|onboarding| onboarding.enabled)
        {
            resolved.onboarding_enabled = enabled;
        }
        if let Some(proxy) = &self.proxy {
            resolved.proxy_enabled = proxy.enabled.unwrap_or(false);
            resolved.upstream_providers = proxy.upstream_providers.clone().unwrap_or_default();
        }
        if let Some(dedicated) = &self.dedicated {
            resolved.dedicated_enabled = dedicated.enabled.unwrap_or(true);
            resolved.dedicated_providers = dedicated.providers.clone().unwrap_or_default();
        }
        if let Some(connectors) = &self.connectors {
            resolved.connectors_enabled = connectors.enabled;
            resolved.pinned_tools = connectors.pinned_tools.clone().unwrap_or_default();
            resolved.discovery_tools = connectors.discovery_tools.unwrap_or(true);
        }
        resolved
    }
}

impl Resolved {
    /// Returns only proxy providers that retain the enabled-unless-disabled
    /// default.
    pub fn enabled_upstream_providers(&self) -> Vec<ProxiedProviderConfig> {
        self.upstream_providers
            .iter()
            .filter(|provider| provider.is_enabled())
            .cloned()
            .collect()
    }
}

/// Reads, validates, and migrates an Aperture config. Missing files retain
/// their standard `NotFound` I/O error so callers can distinguish unconfigured
/// from malformed configuration.
pub fn load_config(path: impl AsRef<Path>) -> Result<Config> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| ApertureError::Io {
        operation: "open aperture config",
        source,
    })?;
    let metadata = file.metadata().map_err(|source| ApertureError::Io {
        operation: "stat aperture config",
        source,
    })?;
    if !metadata.is_file() {
        return Err(ApertureError::ConfigNotRegularFile);
    }
    if metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err(ApertureError::ConfigTooLarge);
    }
    let bytes = read_limited(&mut file, MAX_CONFIG_BYTES, ApertureError::ConfigTooLarge)?;
    let mut config: Config =
        serde_json::from_slice(&bytes).map_err(ApertureError::InvalidConfig)?;
    config.migrate();
    Ok(config)
}

/// Atomically persists a migrated Aperture config with user-only file
/// permissions on Unix.
pub fn save_config(path: impl AsRef<Path>, config: &Config) -> Result<()> {
    let mut migrated = config.clone();
    migrated.migrate();
    let mut bytes = serde_json::to_vec_pretty(&migrated).map_err(ApertureError::SerializeConfig)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_CONFIG_BYTES {
        return Err(ApertureError::ConfigTooLarge);
    }
    atomic_private_write(path.as_ref(), &bytes, "write aperture config")
}

fn option_vec_is_empty<T>(value: &Option<Vec<T>>) -> bool {
    value.as_ref().is_none_or(Vec::is_empty)
}

fn is_false(value: &bool) -> bool {
    !value
}

fn read_limited(reader: &mut impl Read, limit: usize, too_large: ApertureError) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader
        .take((limit + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|source| ApertureError::Io {
            operation: "read Aperture data",
            source,
        })?;
    if bytes.len() > limit {
        return Err(too_large);
    }
    Ok(bytes)
}

fn atomic_private_write(path: &Path, bytes: &[u8], operation: &'static str) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let parent_existed = parent.exists();
    fs::create_dir_all(parent).map_err(|source| ApertureError::Io {
        operation: "create Aperture directory",
        source,
    })?;
    #[cfg(unix)]
    if !parent_existed {
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700)).map_err(|source| {
            ApertureError::Io {
                operation: "secure Aperture directory",
                source,
            }
        })?;
    }

    let file_name = path.file_name().ok_or_else(|| ApertureError::Io {
        operation,
        source: io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"),
    })?;
    let mut temporary = None;
    let mut file = None;
    for _ in 0..32 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}-{}-{}.tmp",
            file_name.to_string_lossy(),
            process::id(),
            sequence
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&candidate) {
            Ok(opened) => {
                temporary = Some(candidate);
                file = Some(opened);
                break;
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(ApertureError::Io {
                    operation: "create Aperture temporary file",
                    source,
                });
            }
        }
    }
    let temporary = temporary.ok_or_else(|| ApertureError::Io {
        operation: "create unique Aperture temporary file",
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "temporary file name collision",
        ),
    })?;
    let mut file = file.ok_or_else(|| ApertureError::Io {
        operation: "retain Aperture temporary file",
        source: io::Error::other("temporary file was not retained"),
    })?;

    let result = (|| {
        file.write_all(bytes)
            .map_err(|source| ApertureError::Io { operation, source })?;
        file.sync_all().map_err(|source| ApertureError::Io {
            operation: "sync Aperture temporary file",
            source,
        })?;
        drop(file);
        fs::rename(&temporary, path).map_err(|source| ApertureError::Io {
            operation: "atomically replace Aperture file",
            source,
        })?;
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|source| {
            ApertureError::Io {
                operation: "secure Aperture file",
                source,
            }
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

/// Normalizes human-entered gateway URLs to an origin. Inputs lacking a scheme
/// default to HTTP; a full API path is discarded.
pub fn normalize_input_url(raw: &str) -> String {
    let mut value = raw.trim().to_owned();
    if value.is_empty() {
        return value;
    }
    if !value.starts_with("http://") && !value.starts_with("https://") {
        value = format!("http://{value}");
    }
    match Url::parse(&value) {
        Ok(url) if url.has_host() => parsed_origin(&value, &url),
        _ => trim_gateway_suffix(&value),
    }
}

/// Returns the configured gateway root, without a trailing slash or `/v1`.
pub fn gateway_url(base_url: &str) -> String {
    if base_url.is_empty() {
        String::new()
    } else {
        trim_gateway_suffix(base_url)
    }
}

/// Returns the OpenAI-shaped `/v1` endpoint for a configured gateway.
pub fn provider_base_url(base_url: &str) -> String {
    let gateway = gateway_url(base_url);
    if gateway.is_empty() {
        String::new()
    } else {
        format!("{gateway}/v1")
    }
}

fn trim_gateway_suffix(value: &str) -> String {
    let without_v1 = value
        .strip_suffix("/v1/")
        .or_else(|| value.strip_suffix("/v1"))
        .unwrap_or(value);
    without_v1.trim_end_matches('/').to_owned()
}

// Go's net/url preserves an explicitly written default port in URL.Host.
// url::Url's Origin representation may normalize it away, so retain the
// parsed input authority (without user info) after URL validation.
fn parsed_origin(value: &str, parsed: &Url) -> String {
    let authority = value
        .split_once("://")
        .map(|(_, remainder)| {
            remainder
                .split(['/', '?', '#'])
                .next()
                .unwrap_or_default()
                .rsplit('@')
                .next()
                .unwrap_or_default()
        })
        .filter(|authority| !authority.is_empty());
    authority
        .map(|authority| format!("{}://{authority}", parsed.scheme()))
        .unwrap_or_else(|| parsed.origin().ascii_serialization())
}

const COMPATIBILITY_APIS: &[(&str, &str)] = &[
    ("openai_chat", "openai-completions"),
    ("anthropic_messages", "anthropic-messages"),
    ("openai_responses", "openai-responses"),
    ("gemini_generate_content", "google-generative-ai"),
    ("google_generate_content", "google-vertex"),
    ("bedrock_converse", "bedrock-converse-stream"),
];

/// Returns every dispatchable API in the gateway's auto-selection order.
pub fn selectable_apis(compatibility: &BTreeMap<String, bool>) -> Vec<RoutableApi> {
    COMPATIBILITY_APIS
        .iter()
        .filter(|(flag, _)| compatibility.get(*flag).copied().unwrap_or(false))
        .map(|(_, api)| (*api).to_owned())
        .collect()
}

/// Selects Aperture's preferred API, defaulting to OpenAI chat completions.
pub fn api_for_compatibility(compatibility: &BTreeMap<String, bool>) -> RoutableApi {
    selectable_apis(compatibility)
        .into_iter()
        .next()
        .unwrap_or_else(|| "openai-completions".to_owned())
}

/// Checks whether a configured API override is still served by a provider.
pub fn is_selectable_api(api: &str, compatibility: &BTreeMap<String, bool>) -> bool {
    selectable_apis(compatibility)
        .iter()
        .any(|candidate| candidate == api)
}

/// Whether a client API embeds model IDs in request URLs instead of JSON.
pub fn embeds_model_id_in_path(api: &str) -> bool {
    matches!(
        api,
        "google-generative-ai" | "google-vertex" | "bedrock-converse-stream"
    )
}

/// Whether the client must register against the gateway root rather than
/// `gateway/v1`.
pub fn should_use_gateway_root(api: &str, upstream_base_url: &str) -> bool {
    if matches!(api, "anthropic-messages" | "openai-codex-responses") {
        return true;
    }
    matches!(api, "openai-completions" | "openai-responses")
        && has_non_v1_version_path(upstream_base_url)
}

fn has_non_v1_version_path(base_url: &str) -> bool {
    let Ok(parsed) = Url::parse(base_url) else {
        return false;
    };
    if !parsed.has_host() {
        return false;
    }
    let segment = parsed.path().trim_end_matches('/').rsplit('/').next();
    let Some(segment) = segment.filter(|segment| !segment.is_empty()) else {
        return false;
    };
    let Some(rest) = segment.strip_prefix('v') else {
        return false;
    };
    let mut characters = rest.chars();
    if !characters
        .next()
        .is_some_and(|character| character.is_ascii_digit())
    {
        return false;
    }
    rest.chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_')
        && segment != "v1"
}

/// Computes the API-specific gateway base URL used by proxy and dedicated
/// model registration.
pub fn base_url_for_api(
    api: &str,
    gateway: &str,
    openai_base: &str,
    upstream_base_url: &str,
) -> String {
    match api {
        "anthropic-messages" => gateway.to_owned(),
        "google-generative-ai" => format!("{gateway}/v1beta"),
        "google-vertex" => format!("{gateway}/v1"),
        "bedrock-converse-stream" => format!("{gateway}/bedrock"),
        _ if should_use_gateway_root(api, upstream_base_url) => gateway.to_owned(),
        _ => openai_base.to_owned(),
    }
}

/// Qualifies a body-carried model ID with its gateway provider. APIs whose
/// model IDs appear in URLs must retain a bare ID.
pub fn qualify_model_id(provider_id: &str, api: &str, model_id: &str) -> String {
    if embeds_model_id_in_path(api) {
        model_id.to_owned()
    } else {
        format!("{provider_id}/{model_id}")
    }
}

/// Removes precisely one catalog provider prefix for URL-path APIs.
pub fn strip_catalog_prefix(api: &str, model_id: &str) -> String {
    if !embeds_model_id_in_path(api) {
        return model_id.to_owned();
    }
    model_id
        .split_once('/')
        .map(|(_, bare)| bare.to_owned())
        .unwrap_or_else(|| model_id.to_owned())
}

/// Pricing reported by `/v1/models`, expressed as per-token decimal strings.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelPricing {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub input: String,
    #[serde(rename = "input_cache_read", skip_serializing_if = "String::is_empty")]
    pub input_cache_read: String,
    #[serde(rename = "input_cache_write", skip_serializing_if = "String::is_empty")]
    pub input_cache_write: String,
    #[serde(
        rename = "input_cache_write_1h",
        skip_serializing_if = "String::is_empty"
    )]
    pub input_cache_write_1h: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub output: String,
    #[serde(rename = "web_search", skip_serializing_if = "String::is_empty")]
    pub web_search: String,
}

/// The price metadata attached to an enabled gateway model.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<ModelPricing>,
}

/// A provider returned by `/api/providers`, enriched with enabled model data.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct GatewayProvider {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub models: Vec<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub compatibility: BTreeMap<String, bool>,
    #[serde(rename = "requires_client_auth", skip_serializing_if = "is_false")]
    pub requires_client_auth: bool,
    #[serde(
        rename = "modelInfoById",
        default,
        skip_serializing_if = "BTreeMap::is_empty",
        deserialize_with = "null_to_default"
    )]
    pub model_info_by_id: BTreeMap<String, ModelInfo>,
}

/// One entry from `/api/connectors`.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ConnectorInfo {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub protocol: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub provider: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub category: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub status: String,
    #[serde(rename = "auth_type", skip_serializing_if = "String::is_empty")]
    pub auth_type: String,
}

/// A connector tool as reported by MCP `tools/list`. The state builder keeps
/// its full JSON Schema opaque; the later MCP runtime can register it verbatim.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct GatewayTool {
    pub name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(rename = "inputSchema", skip_serializing_if = "Value::is_null")]
    pub input_schema: Value,
}

fn null_to_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(|value| value.unwrap_or_default())
}

/// Parses the flexible `/api/providers` response shape: an array,
/// `{"providers": [...]}`, or `{"providers": {"id": {...}}}`. Invalid
/// entries are skipped just as the Go decoder skips non-provider values.
pub fn parse_gateway_providers_json(payload: &[u8]) -> Vec<GatewayProvider> {
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return Vec::new();
    };
    match root {
        Value::Array(entries) => entries
            .iter()
            .filter_map(|entry| decode_gateway_provider(entry, None))
            .collect(),
        Value::Null => Vec::new(),
        Value::Object(envelope) => match envelope.get("providers") {
            Some(Value::Array(entries)) => entries
                .iter()
                .filter_map(|entry| decode_gateway_provider(entry, None))
                .collect(),
            Some(Value::Object(entries)) => entries
                .iter()
                .filter_map(|(id, entry)| decode_gateway_provider(entry, Some(id)))
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn decode_gateway_provider(value: &Value, fallback_id: Option<&str>) -> Option<GatewayProvider> {
    let mut provider = serde_json::from_value::<GatewayProvider>(value.clone()).ok()?;
    if provider.id.is_empty() {
        provider.id = fallback_id.unwrap_or_default().to_owned();
    }
    if provider.id.is_empty() {
        return None;
    }
    if provider.name.is_empty() {
        provider.name = provider.id.clone();
    }
    Some(provider)
}

/// Parses the `data` array from `/v1/models`. A malformed body intentionally
/// becomes an empty map; callers then preserve the providers response
/// unfiltered as a fail-open fallback.
pub fn parse_enabled_models_json(payload: &[u8]) -> BTreeMap<String, ModelInfo> {
    let Ok(root) = serde_json::from_slice::<Value>(payload) else {
        return BTreeMap::new();
    };
    let Some(entries) = root.get("data").and_then(Value::as_array) else {
        return BTreeMap::new();
    };
    let mut models = BTreeMap::new();
    for entry in entries {
        let Ok(model) = serde_json::from_value::<ModelInfo>(entry.clone()) else {
            return BTreeMap::new();
        };
        if !model.id.is_empty() {
            models.insert(model.id.clone(), model);
        }
    }
    models
}

/// Applies `/v1/models` as the source of truth for enabled models. An empty
/// model map is intentionally fail-open to retain usable provider information
/// when the secondary endpoint is unavailable or malformed.
pub fn filter_providers_by_enabled_models(
    providers: Vec<GatewayProvider>,
    enabled_models: &BTreeMap<String, ModelInfo>,
) -> Vec<GatewayProvider> {
    if enabled_models.is_empty() {
        return providers;
    }
    providers
        .into_iter()
        .filter_map(|mut provider| {
            let mut model_info_by_id = BTreeMap::new();
            let models = std::mem::take(&mut provider.models)
                .into_iter()
                .filter(|id| {
                    let Some(info) = enabled_models.get(id) else {
                        return false;
                    };
                    model_info_by_id.insert(id.clone(), info.clone());
                    true
                })
                .collect::<Vec<_>>();
            (!models.is_empty()).then(|| {
                provider.models = models;
                provider.model_info_by_id = model_info_by_id;
                provider
            })
        })
        .collect()
}

/// Parses connector metadata and removes unusable no-ID entries.
pub fn parse_connectors_json(payload: &[u8]) -> Result<Vec<ConnectorInfo>> {
    let root = serde_json::from_slice::<Value>(payload).map_err(|error| {
        ApertureError::InvalidGatewayResponse(format!("decode connectors: {error}"))
    })?;
    if root.is_null() {
        return Ok(Vec::new());
    }
    let object = root.as_object().ok_or_else(|| {
        ApertureError::InvalidGatewayResponse("decode connectors: expected an object".to_owned())
    })?;
    let Some(value) = object.get("connectors") else {
        return Ok(Vec::new());
    };
    if value.is_null() {
        return Ok(Vec::new());
    }
    let entries = value.as_array().ok_or_else(|| {
        ApertureError::InvalidGatewayResponse(
            "decode connectors: connectors is not an array".to_owned(),
        )
    })?;
    let mut connectors = Vec::new();
    for entry in entries {
        let connector =
            serde_json::from_value::<ConnectorInfo>(entry.clone()).map_err(|error| {
                ApertureError::InvalidGatewayResponse(format!("decode connectors: {error}"))
            })?;
        if !connector.id.is_empty() {
            connectors.push(connector);
        }
    }
    Ok(connectors)
}

/// Blocking HTTP client for the three Aperture catalog endpoints.
pub struct GatewayClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl GatewayClient {
    /// Builds a bounded-timeout gateway client. The input may include trailing
    /// slashes, which are removed before endpoint paths are joined.
    pub fn new(base_url: impl AsRef<str>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(ApertureError::HttpClient)?;
        Ok(Self {
            base_url: base_url.as_ref().trim_end_matches('/').to_owned(),
            client,
        })
    }

    /// Creates a client around an injected transport for controlled embedding
    /// or tests.
    pub fn with_client(base_url: impl AsRef<str>, client: reqwest::blocking::Client) -> Self {
        Self {
            base_url: base_url.as_ref().trim_end_matches('/').to_owned(),
            client,
        }
    }

    fn fetch(&self, path: &str) -> Result<Vec<u8>> {
        let url = format!("{}{path}", self.base_url);
        let mut response = self
            .client
            .get(url)
            .send()
            .map_err(ApertureError::Request)?;
        if !response.status().is_success() {
            return Err(ApertureError::Http(HttpError {
                method: "GET".to_owned(),
                path: path.to_owned(),
                status: response.status().as_u16(),
            }));
        }
        read_limited(
            &mut response,
            MAX_RESPONSE_BYTES,
            ApertureError::GatewayResponseTooLarge,
        )
    }

    /// Returns gateway providers, filtering models through `/v1/models` when
    /// that endpoint succeeds and decodes. Secondary endpoint failure is
    /// deliberately fail-open.
    pub fn providers(&self) -> Result<Vec<GatewayProvider>> {
        let providers = parse_gateway_providers_json(&self.fetch("/api/providers")?);
        let enabled = self
            .fetch("/v1/models")
            .map(|payload| parse_enabled_models_json(&payload))
            .unwrap_or_default();
        Ok(filter_providers_by_enabled_models(providers, &enabled))
    }

    /// Returns usable connector metadata from `/api/connectors`.
    pub fn connectors(&self) -> Result<Vec<ConnectorInfo>> {
        parse_connectors_json(&self.fetch("/api/connectors")?)
    }

    /// Probes gateway health by listing providers.
    pub fn health(&self) -> Result<()> {
        self.providers().map(|_| ())
    }
}

/// A compact cache snapshot used by proxy routing while the gateway is offline.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct GatewaySnapshot {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub compatibility: BTreeMap<String, bool>,
    #[serde(rename = "requires_client_auth", skip_serializing_if = "is_false")]
    pub requires_client_auth: bool,
}

/// A persisted dedicated model. `rawCompat` mirrors the Go cache shape while
/// `llm::Model::compat` keeps the current Rust representation usable.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CachedModel {
    #[serde(flatten)]
    pub model: llm::Model,
    #[serde(rename = "rawCompat", default, skip_serializing_if = "Option::is_none")]
    pub raw_compat: Option<Value>,
}

/// The persisted `extensions/aperture-cache.json` shape.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct Cache {
    #[serde(rename = "catalogKey")]
    pub catalog_key: String,
    #[serde(rename = "checkedAt")]
    pub checked_at: i64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<CachedModel>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gateway: Vec<GatewaySnapshot>,
}

impl Cache {
    /// Returns cached dedicated models only when their complete catalog
    /// identity matches the active configuration.
    pub fn catalog_models(&self, catalog_key: &str) -> Option<Vec<llm::Model>> {
        (self.catalog_key == catalog_key).then(|| {
            self.models
                .iter()
                .map(|cached| {
                    let mut model = cached.model.clone();
                    if let Some(raw_compat) = &cached.raw_compat {
                        model.compat = Some(raw_compat.clone());
                    }
                    if model.provider.is_empty() {
                        model.provider = DEDICATED_PROVIDER_ID.to_owned();
                    }
                    model
                })
                .collect()
        })
    }
}

/// Builds a cache snapshot after a successful gateway refresh.
pub fn new_cache(
    catalog_key: impl Into<String>,
    models: &[llm::Model],
    gateway: &[GatewayProvider],
) -> Cache {
    Cache {
        catalog_key: catalog_key.into(),
        checked_at: now_millis(),
        models: models
            .iter()
            .cloned()
            .map(|model| CachedModel {
                raw_compat: model.compat.clone(),
                model,
            })
            .collect(),
        gateway: gateway
            .iter()
            .map(|provider| GatewaySnapshot {
                id: provider.id.clone(),
                name: provider.name.clone(),
                models: provider.models.clone(),
                compatibility: provider.compatibility.clone(),
                requires_client_auth: provider.requires_client_auth,
            })
            .collect(),
    }
}

/// Reads the persisted Aperture snapshot without applying migrations.
pub fn load_cache(path: impl AsRef<Path>) -> Result<Cache> {
    let path = path.as_ref();
    let mut file = File::open(path).map_err(|source| ApertureError::Io {
        operation: "open aperture cache",
        source,
    })?;
    let metadata = file.metadata().map_err(|source| ApertureError::Io {
        operation: "stat aperture cache",
        source,
    })?;
    if !metadata.is_file() {
        return Err(ApertureError::CacheNotRegularFile);
    }
    if metadata.len() > MAX_RESPONSE_BYTES as u64 {
        return Err(ApertureError::CacheTooLarge);
    }
    let bytes = read_limited(&mut file, MAX_RESPONSE_BYTES, ApertureError::CacheTooLarge)?;
    serde_json::from_slice(&bytes).map_err(ApertureError::InvalidCache)
}

/// Atomically saves a cache snapshot with private Unix file permissions.
pub fn save_cache(path: impl AsRef<Path>, cache: &Cache) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(cache).map_err(ApertureError::SerializeCache)?;
    bytes.push(b'\n');
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(ApertureError::CacheTooLarge);
    }
    atomic_private_write(path.as_ref(), &bytes, "write aperture cache")
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

/// Stable catalog identity: gateway origin, selected dedicated providers and
/// their API overrides, plus the catalog version.
pub fn build_catalog_key(gateway: &str, resolved: &Resolved) -> String {
    let origin = Url::parse(gateway)
        .ok()
        .filter(|url| url.has_host())
        .map(|url| parsed_origin(gateway, &url))
        .unwrap_or_else(|| gateway.to_owned());
    let mut selected = resolved
        .dedicated_providers
        .iter()
        .filter(|provider| provider.enabled)
        .map(|provider| {
            if provider.api.is_empty() {
                provider.id.clone()
            } else {
                format!("{}@{}", provider.id, provider.api)
            }
        })
        .collect::<Vec<_>>();
    selected.sort();
    let selection = if resolved.dedicated_providers.is_empty() {
        "*".to_owned()
    } else {
        selected.join(",")
    };
    format!("{origin} {selection} v2")
}

/// Filters gateway providers according to the dedicated provider selection.
/// An empty selection means all providers.
pub fn filter_dedicated_providers(
    providers: &[GatewayProvider],
    resolved: &Resolved,
) -> Vec<GatewayProvider> {
    if resolved.dedicated_providers.is_empty() {
        return providers.to_vec();
    }
    let enabled = resolved
        .dedicated_providers
        .iter()
        .filter(|provider| provider.enabled)
        .map(|provider| provider.id.as_str())
        .collect::<BTreeSet<_>>();
    providers
        .iter()
        .filter(|provider| enabled.contains(provider.id.as_str()))
        .cloned()
        .collect()
}

/// Model metadata available from models.dev, intentionally limited to the
/// fields Aperture uses while enriching a gateway model.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelsDevModel {
    pub name: String,
    pub reasoning: Option<bool>,
    pub modalities: ModelsDevModalities,
    pub limit: ModelsDevLimit,
    pub cost: Option<ModelsDevCost>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelsDevModalities {
    pub input: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelsDevLimit {
    pub context: u64,
    pub output: u64,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelsDevCost {
    pub input: Option<f64>,
    pub output: Option<f64>,
    #[serde(rename = "cache_read")]
    pub cache_read: Option<f64>,
    #[serde(rename = "cache_write")]
    pub cache_write: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct ModelsDevProvider {
    pub models: BTreeMap<String, ModelsDevModel>,
}

/// The models.dev document, keyed by provider then model ID.
pub type ModelsDevCatalog = BTreeMap<String, ModelsDevProvider>;

/// Resolved metadata from models.dev and the native static catalog.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelMetadata {
    pub name: String,
    pub reasoning: Option<bool>,
    pub thinking_level_map: llm::ThinkingLevelMap,
    pub input: Vec<String>,
    pub context_window: u64,
    pub max_tokens: u64,
    pub cost: Option<llm::ModelCost>,
    pub compat: Option<Value>,
}

/// Applies models.dev first, then native catalog metadata on top. A
/// model-ID-only fallback gets capabilities but never provider-specific cost.
pub fn resolve_model_metadata(
    provider_id: &str,
    model_id: &str,
    catalog_models: &[llm::Model],
    models_dev: Option<&ModelsDevCatalog>,
) -> ModelMetadata {
    let mut metadata = ModelMetadata::default();
    if let Some(models_dev) = models_dev {
        apply_models_dev_metadata(&mut metadata, models_dev, provider_id, model_id);
    }
    apply_catalog_metadata(&mut metadata, catalog_models, provider_id, model_id);
    metadata
}

fn find_models_dev_match<'a>(
    catalog: &'a ModelsDevCatalog,
    provider_id: &str,
    model_id: &str,
) -> Option<(&'a ModelsDevModel, bool)> {
    if let Some(model) = catalog
        .get(provider_id)
        .and_then(|provider| provider.models.get(model_id))
    {
        return Some((model, true));
    }
    let mut found = None;
    for provider in catalog.values() {
        let Some(model) = provider.models.get(model_id) else {
            continue;
        };
        if found.is_some() {
            return None;
        }
        found = Some(model);
    }
    found.map(|model| (model, false))
}

fn apply_models_dev_metadata(
    metadata: &mut ModelMetadata,
    catalog: &ModelsDevCatalog,
    provider_id: &str,
    model_id: &str,
) {
    let Some((model, provider_exact)) = find_models_dev_match(catalog, provider_id, model_id)
    else {
        return;
    };
    if !model.name.is_empty() {
        metadata.name = model.name.clone();
    }
    if model.reasoning.is_some() {
        metadata.reasoning = model.reasoning;
    }
    let input = normalize_input_modalities(&model.modalities.input);
    if !input.is_empty() {
        metadata.input = input;
    }
    if model.limit.context > 0 {
        metadata.context_window = model.limit.context;
    }
    if model.limit.output > 0 {
        metadata.max_tokens = model.limit.output;
    }
    if provider_exact
        && model
            .cost
            .as_ref()
            .is_some_and(|cost| cost.input.is_some() || cost.output.is_some())
    {
        let cost = model.cost.as_ref().expect("checked above");
        metadata.cost = Some(llm::ModelCost {
            rates: llm::ModelCostRates {
                input: cost.input.unwrap_or_default(),
                output: cost.output.unwrap_or_default(),
                cache_read: cost.cache_read.unwrap_or_default(),
                cache_write: cost.cache_write.unwrap_or_default(),
            },
            ..llm::ModelCost::default()
        });
    }
}

const INTRINSIC_COMPAT_KEYS: &[&str] = &[
    "supportsDeveloperRole",
    "maxTokensField",
    "requiresReasoningContentOnAssistantMessages",
];

fn apply_catalog_metadata(
    metadata: &mut ModelMetadata,
    catalog_models: &[llm::Model],
    provider_id: &str,
    model_id: &str,
) {
    let exact = catalog_models
        .iter()
        .find(|model| model.provider == provider_id && model.id == model_id);
    let (model, provider_exact) =
        match exact.or_else(|| catalog_models.iter().find(|model| model.id == model_id)) {
            Some(model) => (model, exact.is_some()),
            None => return,
        };
    if !model.name.is_empty() {
        metadata.name = model.name.clone();
    }
    metadata.reasoning = Some(model.reasoning);
    if !model.thinking_level_map.is_empty() {
        metadata.thinking_level_map = model.thinking_level_map.clone();
    }
    if !model.input.is_empty() {
        metadata.input = model.input.clone();
    }
    if model.context_window > 0 {
        metadata.context_window = model.context_window;
    }
    if model.max_tokens > 0 {
        metadata.max_tokens = model.max_tokens;
    }
    if provider_exact {
        if model.cost.rates.input != 0.0 || model.cost.rates.output != 0.0 {
            metadata.cost = Some(model.cost.clone());
        }
        if model
            .compat
            .as_ref()
            .is_some_and(|compat| !compat.is_null())
        {
            metadata.compat = model.compat.clone();
        }
        return;
    }

    let Some(Value::Object(compat)) = &model.compat else {
        return;
    };
    let mut intrinsic = Map::new();
    for key in INTRINSIC_COMPAT_KEYS {
        if let Some(value) = compat.get(*key) {
            intrinsic.insert((*key).to_owned(), value.clone());
        }
    }
    if !intrinsic.is_empty() {
        metadata.compat = Some(Value::Object(intrinsic));
    }
}

fn normalize_input_modalities(values: &[String]) -> Vec<String> {
    values
        .iter()
        .filter(|value| value.as_str() == "text" || value.as_str() == "image")
        .cloned()
        .collect()
}

/// Merges gateway pricing over previously resolved per-million-token rates.
/// A present but malformed price follows the Go behavior and becomes zero.
pub fn merge_cost(pricing: Option<&ModelPricing>, base: Option<&llm::ModelCost>) -> llm::ModelCost {
    let mut cost = base.cloned().unwrap_or_default();
    let Some(pricing) = pricing else {
        return cost;
    };
    if !pricing.input.is_empty() {
        cost.rates.input = parse_price(&pricing.input);
    }
    if !pricing.output.is_empty() {
        cost.rates.output = parse_price(&pricing.output);
    }
    if !pricing.input_cache_read.is_empty() {
        cost.rates.cache_read = parse_price(&pricing.input_cache_read);
    }
    if !pricing.input_cache_write.is_empty() {
        cost.rates.cache_write = parse_price(&pricing.input_cache_write);
    }
    cost
}

fn parse_price(value: &str) -> f64 {
    value.parse::<f64>().unwrap_or_default() * 1_000_000.0
}

/// A rebuilt dedicated catalog plus warnings for stale API overrides.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DedicatedBuild {
    pub models: Vec<llm::Model>,
    pub warnings: Vec<String>,
}

/// Builds model entries for the dedicated `aperture` provider. Static catalog
/// data wins over models.dev metadata; gateway pricing wins over both.
pub fn build_dedicated_models(
    providers: &[GatewayProvider],
    gateway: &str,
    openai_base: &str,
    catalog_models: &[llm::Model],
    models_dev: Option<&ModelsDevCatalog>,
    api_overrides: &BTreeMap<String, RoutableApi>,
) -> DedicatedBuild {
    let mut upstream_by_provider = BTreeMap::new();
    let mut upstream_by_model = BTreeMap::new();
    for model in catalog_models {
        if model.base_url.is_empty() || model.base_url == gateway || model.base_url == openai_base {
            continue;
        }
        if !model.provider.is_empty() {
            upstream_by_provider
                .entry(model.provider.clone())
                .or_insert_with(|| model.base_url.clone());
        }
        upstream_by_model
            .entry(model.id.clone())
            .or_insert_with(|| model.base_url.clone());
    }
    let metadata_catalog = catalog_models
        .iter()
        .filter(|model| model.provider != DEDICATED_PROVIDER_ID)
        .cloned()
        .collect::<Vec<_>>();

    let mut result = DedicatedBuild::default();
    for provider in providers {
        let mut api = api_for_compatibility(&provider.compatibility);
        if let Some(override_api) = api_overrides.get(&provider.id) {
            if is_selectable_api(override_api, &provider.compatibility) {
                api = override_api.clone();
            } else {
                result.warnings.push(format!(
                    "[aperture] api override {:?} for dedicated provider {} is not served by the gateway; using the auto-picked api.",
                    override_api, provider.id
                ));
            }
        }
        let provider_upstream = upstream_by_provider.get(&provider.id);
        for model_id in &provider.models {
            let upstream = provider_upstream
                .or_else(|| upstream_by_model.get(model_id))
                .map(String::as_str)
                .unwrap_or_default();
            let metadata =
                resolve_model_metadata(&provider.id, model_id, &metadata_catalog, models_dev);
            let pricing = provider
                .model_info_by_id
                .get(model_id)
                .and_then(|info| info.pricing.as_ref());
            result.models.push(build_default_model(
                provider,
                model_id,
                &api,
                gateway,
                openai_base,
                upstream,
                pricing,
                metadata,
            ));
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn build_default_model(
    provider: &GatewayProvider,
    model_id: &str,
    api: &str,
    gateway: &str,
    openai_base: &str,
    upstream_base_url: &str,
    pricing: Option<&ModelPricing>,
    metadata: ModelMetadata,
) -> llm::Model {
    let ModelMetadata {
        name,
        reasoning,
        thinking_level_map,
        input,
        context_window,
        max_tokens,
        cost,
        compat,
    } = metadata;
    let name = if name.is_empty() {
        model_id.to_owned()
    } else {
        name
    };
    let input = if input.is_empty() {
        vec!["text".to_owned()]
    } else {
        input
    };
    let context_window = if context_window == 0 {
        128_000
    } else {
        context_window
    };
    let max_tokens = if max_tokens == 0 { 8_192 } else { max_tokens };
    llm::Model {
        id: qualify_model_id(&provider.id, api, model_id),
        name,
        api: api.to_owned(),
        provider: DEDICATED_PROVIDER_ID.to_owned(),
        base_url: base_url_for_api(api, gateway, openai_base, upstream_base_url),
        reasoning: reasoning.unwrap_or(false),
        thinking_level_map,
        input,
        cost: merge_cost(pricing, cost.as_ref()),
        context_window,
        max_tokens,
        compat,
        ..llm::Model::default()
    }
}

/// A local provider's immutable catalog facts used to construct a proxy route.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeProviderInfo {
    pub api: RoutableApi,
    pub base_url: String,
    pub model_ids: Vec<String>,
}

/// The catalog rewrite for one proxied provider.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProxyRoute {
    pub provider_id: String,
    pub api: RoutableApi,
    pub api_overridden: bool,
    pub base_url: String,
    /// `None` retains every local model; `Some` hides models absent from the
    /// gateway's snapshot.
    pub served_model_ids: Option<BTreeSet<String>>,
    /// Passthrough providers retain native credentials for the gateway to
    /// forward upstream.
    pub passthrough: bool,
}

/// Plans every active proxy route. `native_info` must describe the immutable
/// local catalog, before a prior Aperture rewrite can influence URL inference.
pub fn plan_proxy_routes<F>(
    resolved: &Resolved,
    gateway: &[GatewaySnapshot],
    mut native_info: F,
) -> BTreeMap<String, ProxyRoute>
where
    F: FnMut(&str) -> Option<NativeProviderInfo>,
{
    if !resolved.proxy_enabled || resolved.base_url.is_empty() {
        return BTreeMap::new();
    }
    let gateway_root = gateway_url(&resolved.base_url);
    let openai_base = provider_base_url(&resolved.base_url);
    if gateway_root.is_empty() || openai_base.is_empty() {
        return BTreeMap::new();
    }
    let snapshots = gateway
        .iter()
        .map(|provider| (provider.id.as_str(), provider))
        .collect::<BTreeMap<_, _>>();
    let mut routes = BTreeMap::new();
    for configured in resolved.enabled_upstream_providers() {
        if configured.id == DEDICATED_PROVIDER_ID {
            continue;
        }
        let Some(native) = native_info(&configured.id) else {
            continue;
        };
        if native.model_ids.is_empty() {
            continue;
        }
        let snapshot = snapshots.get(configured.id.as_str()).copied();
        let mut api = native.api.clone();
        let mut api_overridden = false;
        if !configured.api.is_empty()
            && snapshot
                .is_some_and(|snapshot| is_selectable_api(&configured.api, &snapshot.compatibility))
        {
            api = configured.api.clone();
            api_overridden = true;
        }

        let served_model_ids = if configured.keep_gateway_models_only {
            snapshot.map(|snapshot| snapshot.models.iter().cloned().collect::<BTreeSet<_>>())
        } else {
            None
        };
        if served_model_ids
            .as_ref()
            .is_some_and(|served| !native.model_ids.iter().any(|model| served.contains(model)))
        {
            continue;
        }
        routes.insert(
            configured.id.clone(),
            ProxyRoute {
                provider_id: configured.id,
                api: api.clone(),
                api_overridden,
                base_url: base_url_for_api(&api, &gateway_root, &openai_base, &native.base_url),
                served_model_ids,
                passthrough: snapshot.is_some_and(|snapshot| snapshot.requires_client_auth),
            },
        );
    }
    routes
}

/// Applies a planned proxy route to one catalog model. `None` indicates that
/// the model is filtered out by `keepGatewayModelsOnly`.
pub fn apply_proxy_route(model: &llm::Model, route: &ProxyRoute) -> Option<llm::Model> {
    if route
        .served_model_ids
        .as_ref()
        .is_some_and(|served| !served.contains(&model.id))
    {
        return None;
    }
    let mut rewritten = model.clone();
    rewritten.base_url = route.base_url.clone();
    if route.api_overridden {
        rewritten.api = route.api.clone();
    }
    Some(rewritten)
}

/// Reports configured proxy API overrides that the current gateway no longer
/// serves. The route itself safely falls back to the local native API.
pub fn proxy_override_warnings(resolved: &Resolved, gateway: &[GatewayProvider]) -> Vec<String> {
    let compatibility = gateway
        .iter()
        .map(|provider| (provider.id.as_str(), &provider.compatibility))
        .collect::<BTreeMap<_, _>>();
    resolved
        .enabled_upstream_providers()
        .into_iter()
        .filter(|configured| !configured.api.is_empty())
        .filter_map(|configured| {
            compatibility
                .get(configured.id.as_str())
                .filter(|compatibility| !is_selectable_api(&configured.api, compatibility))
                .map(|_| {
                    format!(
                        "[aperture] api override {:?} for proxied provider {} is not served by the gateway; falling back to the provider's own api.",
                        configured.api, configured.id
                    )
                })
        })
        .collect()
}

/// Summarizes missing local models for checked proxy providers, retaining at
/// most five IDs per provider to avoid overwhelming startup notices.
pub fn missing_models_summary(
    resolved: &Resolved,
    gateway: &[GatewayProvider],
    local_models: &[llm::Model],
) -> String {
    let checked = resolved
        .enabled_upstream_providers()
        .into_iter()
        .filter(|provider| provider.should_check_gateway_models)
        .map(|provider| provider.id)
        .collect::<BTreeSet<_>>();
    if checked.is_empty() || gateway.is_empty() {
        return String::new();
    }
    let served = gateway
        .iter()
        .map(|provider| {
            (
                provider.id.as_str(),
                provider
                    .models
                    .iter()
                    .map(String::as_str)
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut missing = BTreeMap::<String, Vec<String>>::new();
    for model in local_models {
        if !checked.contains(&model.provider)
            || served
                .get(model.provider.as_str())
                .is_some_and(|models| models.contains(model.id.as_str()))
        {
            continue;
        }
        missing
            .entry(model.provider.clone())
            .or_default()
            .push(model.id.clone());
    }
    if missing.is_empty() {
        return String::new();
    }
    let details = missing
        .into_iter()
        .map(|(provider, models)| {
            let shown = models.iter().take(5).cloned().collect::<Vec<_>>();
            let extra = (models.len() > shown.len())
                .then(|| format!(", {} more", models.len() - shown.len()));
            format!(
                "{provider}: {}{}",
                shown.join(", "),
                extra.unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "[aperture] models not available on gateway: {details}. Add them to the gateway configuration."
    )
}

/// One selectable proxy-provider row for onboarding or settings UIs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MappedProxyProvider {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub should_check_gateway_models: bool,
    pub keep_gateway_models_only: bool,
    pub api: RoutableApi,
}

/// Matches locally known providers to grant-scoped gateway providers, retaining
/// existing per-provider configuration.
pub fn map_proxy_providers(
    local_models: &[llm::Model],
    gateway: &[GatewayProvider],
    existing: &[ProxiedProviderConfig],
) -> Vec<MappedProxyProvider> {
    let gateway_by_id = gateway
        .iter()
        .map(|provider| (provider.id.as_str(), provider))
        .collect::<BTreeMap<_, _>>();
    let existing_by_id = existing
        .iter()
        .map(|provider| (provider.id.as_str(), provider))
        .collect::<BTreeMap<_, _>>();
    let ids = local_models
        .iter()
        .filter_map(|model| {
            (model.provider != DEDICATED_PROVIDER_ID
                && gateway_by_id.contains_key(model.provider.as_str()))
            .then_some(model.provider.clone())
        })
        .collect::<BTreeSet<_>>();
    ids.into_iter()
        .filter_map(|id| {
            let gateway = gateway_by_id.get(id.as_str())?;
            let existing = existing_by_id.get(id.as_str()).copied();
            Some(MappedProxyProvider {
                id,
                name: gateway.name.clone(),
                enabled: existing.is_some_and(|provider| provider.is_enabled()),
                should_check_gateway_models: existing
                    .map(|provider| provider.should_check_gateway_models)
                    .unwrap_or(true),
                keep_gateway_models_only: existing
                    .is_some_and(|provider| provider.keep_gateway_models_only),
                api: existing
                    .map(|provider| provider.api.clone())
                    .unwrap_or_default(),
            })
        })
        .collect()
}

/// Maps current gateway providers to dedicated selection rows. New gateway
/// providers default enabled; existing settings win by provider ID.
pub fn map_dedicated_providers(
    gateway: &[GatewayProvider],
    existing: &[DedicatedProviderConfig],
) -> Vec<DedicatedProviderConfig> {
    let existing = existing
        .iter()
        .map(|provider| (provider.id.as_str(), provider))
        .collect::<BTreeMap<_, _>>();
    gateway
        .iter()
        .map(|provider| {
            let existing = existing.get(provider.id.as_str()).copied();
            DedicatedProviderConfig {
                id: provider.id.clone(),
                name: provider.name.clone(),
                enabled: existing.map(|provider| provider.enabled).unwrap_or(true),
                api: existing
                    .map(|provider| provider.api.clone())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

/// The four proxy discovery tool names registered when discovery is enabled.
pub const DISCOVERY_TOOL_NAMES: [&str; 4] = [
    "aperture_connector_list",
    "aperture_connector_tool_search",
    "aperture_connector_tool_describe",
    "aperture_connector_tool_call",
];

/// The connector tool surface derived from a fetched MCP tool list. Actual MCP
/// invocation is intentionally separate from this state-only construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConnectorToolSet {
    pub pinned_tools: Vec<GatewayTool>,
    pub proxied_tools: Vec<GatewayTool>,
    pub visible_connectors: Vec<ConnectorInfo>,
    pub missing_pins: Vec<String>,
    pub discovery_enabled: bool,
}

/// Derives a connector ID from the part of a tool name before its first `_`.
/// Tools without a nonempty prefix belong to the synthetic `other` group.
pub fn connector_id_from_tool_name(name: &str) -> String {
    name.split_once('_')
        .filter(|(prefix, _)| !prefix.is_empty())
        .map(|(prefix, _)| prefix.to_owned())
        .unwrap_or_else(|| "other".to_owned())
}

/// Splits de-duplicated gateway tools into pinned first-class tools and
/// discovery-proxied tools. Stale pins are returned sorted and harmless.
pub fn build_connector_tool_set(
    resolved: &Resolved,
    connectors: &[ConnectorInfo],
    tools: &[GatewayTool],
) -> ConnectorToolSet {
    let mut names = BTreeSet::new();
    let unique_tools = tools
        .iter()
        .filter(|tool| names.insert(tool.name.clone()))
        .cloned()
        .collect::<Vec<_>>();
    let tool_counts = unique_tools
        .iter()
        .fold(BTreeMap::new(), |mut counts, tool| {
            *counts
                .entry(connector_id_from_tool_name(&tool.name))
                .or_insert(0_usize) += 1;
            counts
        });
    let visible_connectors = connectors
        .iter()
        .filter(|connector| tool_counts.get(&connector.id).copied().unwrap_or_default() > 0)
        .cloned()
        .collect::<Vec<_>>();

    let pinned_names = resolved
        .pinned_tools
        .iter()
        .map(|pin| pin.tool_name.clone())
        .collect::<BTreeSet<_>>();
    let (pinned_tools, proxied_tools): (Vec<_>, Vec<_>) = unique_tools
        .into_iter()
        .partition(|tool| pinned_names.contains(&tool.name));
    let found_pins = pinned_tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect::<BTreeSet<_>>();
    let missing_pins = pinned_names
        .into_iter()
        .filter(|name| !found_pins.contains(name.as_str()))
        .collect();
    ConnectorToolSet {
        pinned_tools,
        proxied_tools,
        visible_connectors,
        missing_pins,
        discovery_enabled: resolved.discovery_tools,
    }
}

/// Outcome of a state-only pin or unpin change.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PinChange {
    pub connector_id: String,
    pub pin_count: usize,
    pub context_cost_warning: bool,
}

/// Validates and adds one gateway tool to persisted pins.
pub fn pin_connector_tool(
    config: &mut Config,
    tool_name: &str,
    available_tools: &[GatewayTool],
) -> Result<PinChange> {
    if !available_tools.iter().any(|tool| tool.name == tool_name) {
        return Err(ApertureError::ToolNotFound(tool_name.to_owned()));
    }
    let mut pins = config.resolve().pinned_tools;
    if pins.iter().any(|pin| pin.tool_name == tool_name) {
        return Err(ApertureError::ToolAlreadyPinned(tool_name.to_owned()));
    }
    let connector_id = connector_id_from_tool_name(tool_name);
    pins.push(PinnedConnectorTool {
        connector_id: connector_id.clone(),
        tool_name: tool_name.to_owned(),
    });
    config
        .connectors
        .get_or_insert_with(ConnectorsConfig::default)
        .pinned_tools = Some(pins.clone());
    Ok(PinChange {
        connector_id,
        pin_count: pins.len(),
        context_cost_warning: pins.len() > CONTEXT_COST_WARNING_THRESHOLD,
    })
}

/// Removes one persisted pin. An empty post-removal list is deliberately
/// retained as `[]`, matching the Go unpin command.
pub fn unpin_connector_tool(config: &mut Config, tool_name: &str) -> Result<PinChange> {
    let pins = config.resolve().pinned_tools;
    let Some(removed) = pins.iter().find(|pin| pin.tool_name == tool_name) else {
        return Err(ApertureError::ToolNotPinned(tool_name.to_owned()));
    };
    let connector_id = removed.connector_id.clone();
    let remaining = pins
        .into_iter()
        .filter(|pin| pin.tool_name != tool_name)
        .collect::<Vec<_>>();
    config
        .connectors
        .get_or_insert_with(ConnectorsConfig::default)
        .pinned_tools = Some(remaining.clone());
    Ok(PinChange {
        connector_id,
        pin_count: remaining.len(),
        context_cost_warning: remaining.len() > CONTEXT_COST_WARNING_THRESHOLD,
    })
}

/// The resolved routing state for a single catalog view.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ApertureState {
    pub configured: bool,
    pub resolved: Resolved,
    pub dedicated_models: Vec<llm::Model>,
    pub routes: BTreeMap<String, ProxyRoute>,
}

/// Builds the non-networked catalog state from a loaded config and optional
/// cache. Callers provide the immutable native catalog lookup for proxy plans.
pub fn build_aperture_state<F>(
    config: &Config,
    cache: Option<&Cache>,
    native_info: F,
) -> ApertureState
where
    F: FnMut(&str) -> Option<NativeProviderInfo>,
{
    let resolved = config.resolve();
    if resolved.base_url.is_empty() {
        return ApertureState::default();
    }
    let gateway = gateway_url(&resolved.base_url);
    let cache = cache.cloned().unwrap_or_default();
    let dedicated_models = resolved
        .dedicated_enabled
        .then(|| cache.catalog_models(&build_catalog_key(&gateway, &resolved)))
        .flatten()
        .unwrap_or_default();
    let routes = if resolved.proxy_enabled {
        plan_proxy_routes(&resolved, &cache.gateway, native_info)
    } else {
        BTreeMap::new()
    };
    ApertureState {
        configured: true,
        resolved,
        dedicated_models,
        routes,
    }
}

/// Rewrites a selected model for a live Aperture request. Unrouted models are
/// borrowed unchanged; routed models get model-ID normalization and provenance
/// headers without mutating the catalog entry.
pub fn rewrite_request_model<'a>(
    state: Option<&ApertureState>,
    model: &'a llm::Model,
    session_id: &str,
) -> Cow<'a, llm::Model> {
    let Some(state) = state.filter(|state| state.configured) else {
        return Cow::Borrowed(model);
    };
    let mut rewritten = model.clone();
    let routed = if model.provider == DEDICATED_PROVIDER_ID {
        rewritten.id = strip_catalog_prefix(&model.api, &model.id);
        true
    } else if state.routes.contains_key(&model.provider) {
        rewritten.id = qualify_model_id(&model.provider, &model.api, &model.id);
        true
    } else {
        false
    };
    if !routed {
        return Cow::Borrowed(model);
    }
    let mut headers = model.headers.clone();
    headers.insert("Referer".to_owned(), APERTURE_REFERER.to_owned());
    if !session_id.is_empty() {
        headers.insert("x-session-id".to_owned(), session_id.to_owned());
    }
    rewritten.headers = headers;
    Cow::Owned(rewritten)
}

/// The Aperture-specific contribution to retry classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryClassification {
    NotApertureTransient,
    AlreadyRetryable,
    MarkRetryable,
}

/// Classifies the single known transient gateway failure. The main retry
/// classifier already recognizes "service unavailable", so this only needs to
/// append that marker to a restarting error that lacks it.
pub fn classify_retryable_error(message: &str) -> RetryClassification {
    if !contains_case_insensitive(message, "aperture is restarting") {
        RetryClassification::NotApertureTransient
    } else if contains_case_insensitive(message, "service unavailable") {
        RetryClassification::AlreadyRetryable
    } else {
        RetryClassification::MarkRetryable
    }
}

/// Returns a retry-classifier-compatible replacement error message, or `None`
/// when this error should remain untouched.
pub fn mark_retryable_error(message: &str) -> Option<String> {
    (classify_retryable_error(message) == RetryClassification::MarkRetryable)
        .then(|| format!("{message} (service unavailable)"))
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    haystack.to_lowercase().contains(&needle.to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::{
        fs,
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        thread,
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_path(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "goshcoder-aperture-{label}-{}-{sequence}",
            process::id()
        ))
    }

    fn write_config(content: &str) -> PathBuf {
        let path = temporary_path("config").join("aperture.json");
        fs::create_dir_all(path.parent().expect("parent")).expect("create parent");
        fs::write(&path, content).expect("write config");
        path
    }

    fn compat(entries: &[(&str, bool)]) -> BTreeMap<String, bool> {
        entries
            .iter()
            .map(|(key, value)| ((*key).to_owned(), *value))
            .collect()
    }

    fn model(provider: &str, id: &str, api: &str, base_url: &str) -> llm::Model {
        llm::Model {
            id: id.to_owned(),
            name: id.to_owned(),
            api: api.to_owned(),
            provider: provider.to_owned(),
            base_url: base_url.to_owned(),
            input: vec!["text".to_owned()],
            context_window: 128_000,
            max_tokens: 8_192,
            ..llm::Model::default()
        }
    }

    fn gateway_server(
        responses: BTreeMap<String, (u16, String)>,
        request_count: usize,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind gateway stub");
        let address = listener.local_addr().expect("gateway address");
        let thread = thread::spawn(move || {
            for _ in 0..request_count {
                let (mut stream, _) = listener.accept().expect("accept request");
                let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                let mut request = String::new();
                reader.read_line(&mut request).expect("read request line");
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).expect("read request header");
                    if header == "\r\n" || header.is_empty() {
                        break;
                    }
                }
                let path = request
                    .split_whitespace()
                    .nth(1)
                    .expect("HTTP request path");
                let (status, body) = responses.get(path).cloned().unwrap_or((404, String::new()));
                let reason = match status {
                    200 => "OK",
                    502 => "Bad Gateway",
                    _ => "Not Found",
                };
                write!(
                    stream,
                    "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .expect("write response");
                stream.flush().expect("flush response");
            }
        });
        (format!("http://{address}"), thread)
    }

    #[test]
    fn config_resolve_defaults_and_json_field_names_are_compatible() {
        let config: Config = serde_json::from_str(
            r#"{
                "$schema":"https://pi.dev/schema.json",
                "baseUrl":"http://gw.example",
                "proxy":{"upstreamProviders":[{"id":"anthropic","enabled":false}]},
                "connectors":{"pinnedTools":[{"connectorId":"github","toolName":"github_issues"}]}
            }"#,
        )
        .expect("decode config");
        let resolved = config.resolve();
        assert_eq!(resolved.base_url, "http://gw.example");
        assert!(resolved.dedicated_enabled);
        assert!(!resolved.proxy_enabled);
        assert!(!resolved.connectors_enabled);
        assert!(resolved.discovery_tools);
        assert!(resolved.onboarding_enabled);
        assert!(!resolved.upstream_providers[0].is_enabled());
        assert_eq!(resolved.pinned_tools[0].connector_id, "github");

        let encoded = serde_json::to_value(&config).expect("encode config");
        assert_eq!(encoded["$schema"], "https://pi.dev/schema.json");
        assert!(encoded["proxy"]["upstreamProviders"].is_array());
        assert_eq!(
            encoded["connectors"]["pinnedTools"][0]["toolName"],
            "github_issues"
        );
    }

    #[test]
    fn config_migrations_follow_go_order_and_normalize_blocks() {
        let path = write_config(
            r#"{
                "baseUrl":"http://gw.example",
                "providers":["anthropic","openai"],
                "checkGatewayModels":["anthropic"],
                "mode":"proxy",
                "apertureProvider":true
            }"#,
        );
        let config = load_config(&path).expect("load");
        let resolved = config.resolve();
        assert!(resolved.proxy_enabled);
        assert!(!resolved.dedicated_enabled, "mode wins after 001");
        assert!(resolved.onboarding_done);
        assert_eq!(config.version, "0.8.0");
        assert_eq!(resolved.upstream_providers.len(), 2);
        assert!(resolved.upstream_providers[0].should_check_gateway_models);
        assert!(!resolved.upstream_providers[1].should_check_gateway_models);
        assert!(config.legacy_providers.is_none());
        assert!(config.legacy_check_gateway_models.is_none());
        assert!(config.legacy_aperture_provider.is_none());
        assert!(config.legacy_mode.is_empty());
        assert_eq!(
            config.proxy.as_ref().and_then(|proxy| proxy.enabled),
            Some(true)
        );
        assert_eq!(
            config
                .dedicated
                .as_ref()
                .and_then(|dedicated| dedicated.enabled),
            Some(false)
        );
        let _ = fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn normalization_migration_drops_embedded_cached_models() {
        let path = write_config(
            r#"{
                "baseUrl":"http://gw.example",
                "onboardingDone":true,
                "dedicated":{"enabled":true,"cachedModels":[{"id":"legacy"}]}
            }"#,
        );
        let config = load_config(&path).expect("load");
        assert_eq!(config.version, "0.8.0");
        assert!(
            config
                .dedicated
                .as_ref()
                .expect("dedicated")
                .cached_models
                .is_none()
        );
        assert_eq!(
            config.proxy.as_ref().and_then(|proxy| proxy.enabled),
            Some(false)
        );
        assert_eq!(
            config
                .proxy
                .as_ref()
                .and_then(|proxy| proxy.upstream_providers.as_ref())
                .expect("explicit provider array")
                .len(),
            0
        );
        let _ = fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn config_save_round_trips_schema_and_never_writes_legacy_fields() {
        let path = temporary_path("save")
            .join("extensions")
            .join("aperture.json");
        let config = Config {
            schema: "https://pi.dev/schema.json".to_owned(),
            base_url: "http://gw.example".to_owned(),
            legacy_mode: "proxy".to_owned(),
            legacy_providers: Some(vec!["anthropic".to_owned()]),
            ..Config::default()
        };
        save_config(&path, &config).expect("save");
        let text = fs::read_to_string(&path).expect("read");
        let value: Value = serde_json::from_str(&text).expect("JSON");
        assert_eq!(value["$schema"], "https://pi.dev/schema.json");
        for field in [
            "mode",
            "providers",
            "checkGatewayModels",
            "apertureProvider",
        ] {
            assert!(
                value.get(field).is_none(),
                "legacy field {field} leaked: {text}"
            );
        }
        let resolved = load_config(&path).expect("reload").resolve();
        assert!(resolved.proxy_enabled);
        assert_eq!(resolved.upstream_providers.len(), 1);
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
            0o600
        );
        let _ = fs::remove_dir_all(path.ancestors().nth(2).expect("root"));
    }

    #[test]
    fn config_load_rejects_bad_shapes_and_preserves_not_found() {
        let malformed = write_config(r#"{"baseUrl":[1]}"#);
        assert!(matches!(
            load_config(&malformed),
            Err(ApertureError::InvalidConfig(_))
        ));
        let missing = temporary_path("missing").join("aperture.json");
        assert!(matches!(
            load_config(missing),
            Err(ApertureError::Io { source, .. }) if source.kind() == io::ErrorKind::NotFound
        ));
        let _ = fs::remove_dir_all(malformed.parent().expect("parent"));
    }

    #[test]
    fn url_and_api_selection_follow_gateway_rules() {
        assert_eq!(
            normalize_input_url("  ai.host.ts.net/v1/models  "),
            "http://ai.host.ts.net"
        );
        assert_eq!(
            normalize_input_url("https://gateway.example:8443/a?x=1#f"),
            "https://gateway.example:8443"
        );
        assert_eq!(
            normalize_input_url("http://gateway.example:80/v1/models"),
            "http://gateway.example:80",
            "an explicitly configured default port must round-trip"
        );
        assert_eq!(gateway_url("http://gw.example/v1/"), "http://gw.example");
        assert_eq!(
            provider_base_url("http://gw.example"),
            "http://gw.example/v1"
        );
        assert_eq!(provider_base_url(""), "");

        let compatibility = compat(&[
            ("anthropic_messages", true),
            ("openai_responses", true),
            ("openai_chat", true),
            ("google_raw_predict", true),
        ]);
        assert_eq!(
            selectable_apis(&compatibility),
            vec![
                "openai-completions".to_owned(),
                "anthropic-messages".to_owned(),
                "openai-responses".to_owned()
            ]
        );
        assert_eq!(api_for_compatibility(&compatibility), "openai-completions");
        assert!(is_selectable_api("anthropic-messages", &compatibility));
        assert!(!is_selectable_api("google-vertex", &compatibility));
    }

    #[test]
    fn api_specific_routing_and_model_id_rules_are_preserved() {
        let gateway = "http://gw.example";
        let base = "http://gw.example/v1";
        assert!(should_use_gateway_root("anthropic-messages", ""));
        assert!(should_use_gateway_root("openai-codex-responses", ""));
        assert!(!should_use_gateway_root(
            "openai-completions",
            "https://api.openai.com/v1"
        ));
        assert!(should_use_gateway_root(
            "openai-completions",
            "https://api.z.ai/api/coding/paas/v4"
        ));
        assert!(should_use_gateway_root(
            "openai-responses",
            "https://example.test/v4beta"
        ));
        assert!(!should_use_gateway_root("openai-completions", "://broken"));
        assert_eq!(
            base_url_for_api("anthropic-messages", gateway, base, ""),
            gateway
        );
        assert_eq!(
            base_url_for_api("google-generative-ai", gateway, base, ""),
            "http://gw.example/v1beta"
        );
        assert_eq!(
            base_url_for_api("bedrock-converse-stream", gateway, base, ""),
            "http://gw.example/bedrock"
        );
        assert_eq!(
            base_url_for_api(
                "openai-completions",
                gateway,
                base,
                "https://api.z.ai/api/coding/paas/v4"
            ),
            gateway
        );
        assert_eq!(
            qualify_model_id("anthropic", "anthropic-messages", "claude"),
            "anthropic/claude"
        );
        assert_eq!(
            qualify_model_id("google", "google-generative-ai", "gemini"),
            "gemini"
        );
        assert_eq!(
            strip_catalog_prefix("google-vertex", "acme/hf:org/model"),
            "hf:org/model"
        );
        assert_eq!(
            strip_catalog_prefix("anthropic-messages", "anthropic/claude"),
            "anthropic/claude"
        );
    }

    #[test]
    fn gateway_provider_and_connector_parsing_handles_all_supported_shapes() {
        let array = br#"[
            {"id":"anthropic","name":"Anthropic","models":["claude","gone"],
             "compatibility":{"anthropic_messages":true}},
            {"id":"openai","models":["gpt"],"requires_client_auth":true},
            {"id":"disabled","models":["gone"]}
        ]"#;
        let enabled = parse_enabled_models_json(
            br#"{"data":[
                {"id":"claude","pricing":{"input":"0.000003","output":"0.000015"}},
                {"id":"gpt"}
            ]}"#,
        );
        let providers =
            filter_providers_by_enabled_models(parse_gateway_providers_json(array), &enabled);
        assert_eq!(providers.len(), 2);
        assert_eq!(providers[0].models, vec!["claude"]);
        assert_eq!(
            providers[0].model_info_by_id["claude"]
                .pricing
                .as_ref()
                .expect("pricing")
                .input,
            "0.000003"
        );
        assert!(providers[1].requires_client_auth);
        assert_eq!(providers[1].name, "openai", "missing names default to ID");

        let mapped = parse_gateway_providers_json(
            br#"{"providers":{"zeta":{"name":"Zeta","models":["m1"]},"alpha":{"models":["m2"]}}}"#,
        );
        assert_eq!(
            mapped
                .iter()
                .map(|provider| provider.id.as_str())
                .collect::<Vec<_>>(),
            vec!["alpha", "zeta"]
        );
        assert_eq!(mapped[0].name, "alpha");

        let fail_open = filter_providers_by_enabled_models(
            parse_gateway_providers_json(array),
            &parse_enabled_models_json(br#"{"data":"bad"}"#),
        );
        assert_eq!(fail_open.len(), 3);
        assert_eq!(fail_open[0].models.len(), 2);

        let connectors = parse_connectors_json(
            br#"{"connectors":[
                {"id":"github","provider":"GitHub","status":"connected"},
                {"description":"missing ID"}
            ]}"#,
        )
        .expect("connectors");
        assert_eq!(connectors.len(), 1);
        assert_eq!(connectors[0].id, "github");
    }

    #[test]
    fn gateway_client_fetches_filters_and_reports_http_failures() {
        let (base_url, server) = gateway_server(
            BTreeMap::from([
                (
                    "/api/providers".to_owned(),
                    (
                        200,
                        r#"[{"id":"anthropic","models":["claude","gone"]}]"#.to_owned(),
                    ),
                ),
                (
                    "/v1/models".to_owned(),
                    (
                        200,
                        r#"{"data":[{"id":"claude","pricing":{"input":"0.000003"}}]}"#.to_owned(),
                    ),
                ),
                (
                    "/api/connectors".to_owned(),
                    (
                        200,
                        r#"{"connectors":[{"id":"github","provider":"GitHub"}]}"#.to_owned(),
                    ),
                ),
            ]),
            3,
        );
        let client = GatewayClient::new(&base_url).expect("client");
        let providers = client.providers().expect("providers");
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].models, vec!["claude"]);
        assert_eq!(client.connectors().expect("connectors")[0].id, "github");
        server.join().expect("gateway server");

        let (base_url, server) = gateway_server(
            BTreeMap::from([(
                "/api/providers".to_owned(),
                (502, "gateway unavailable".to_owned()),
            )]),
            1,
        );
        let error = GatewayClient::new(&base_url)
            .expect("client")
            .health()
            .expect_err("HTTP failure");
        assert!(matches!(
            error,
            ApertureError::Http(HttpError { status: 502, .. })
        ));
        server.join().expect("failure gateway server");
    }

    fn proxy_resolved(providers: Vec<ProxiedProviderConfig>) -> Resolved {
        Resolved {
            base_url: "http://gw.example".to_owned(),
            proxy_enabled: true,
            upstream_providers: providers,
            ..Resolved::default()
        }
    }

    fn native_info(id: &str) -> Option<NativeProviderInfo> {
        match id {
            "anthropic" => Some(NativeProviderInfo {
                api: "anthropic-messages".to_owned(),
                base_url: "https://api.anthropic.com".to_owned(),
                model_ids: vec!["claude-sonnet".to_owned(), "claude-haiku".to_owned()],
            }),
            "zai" => Some(NativeProviderInfo {
                api: "openai-completions".to_owned(),
                base_url: "https://api.z.ai/api/coding/paas/v4".to_owned(),
                model_ids: vec!["glm".to_owned()],
            }),
            "openai" => Some(NativeProviderInfo {
                api: "openai-responses".to_owned(),
                base_url: "https://api.openai.com/v1".to_owned(),
                model_ids: vec!["gpt".to_owned()],
            }),
            _ => None,
        }
    }

    fn proxy_snapshots() -> Vec<GatewaySnapshot> {
        vec![
            GatewaySnapshot {
                id: "anthropic".to_owned(),
                models: vec!["claude-sonnet".to_owned()],
                compatibility: compat(&[("anthropic_messages", true), ("openai_chat", true)]),
                ..GatewaySnapshot::default()
            },
            GatewaySnapshot {
                id: "zai".to_owned(),
                models: vec!["glm".to_owned()],
                compatibility: compat(&[("openai_chat", true)]),
                ..GatewaySnapshot::default()
            },
            GatewaySnapshot {
                id: "openai".to_owned(),
                models: vec!["gpt".to_owned()],
                compatibility: compat(&[("openai_responses", true)]),
                requires_client_auth: true,
                ..GatewaySnapshot::default()
            },
        ]
    }

    #[test]
    fn proxy_routes_apply_override_filter_and_passthrough_rules() {
        let routes = plan_proxy_routes(
            &proxy_resolved(vec![
                ProxiedProviderConfig {
                    id: "anthropic".to_owned(),
                    keep_gateway_models_only: true,
                    ..ProxiedProviderConfig::default()
                },
                ProxiedProviderConfig {
                    id: "zai".to_owned(),
                    ..ProxiedProviderConfig::default()
                },
                ProxiedProviderConfig {
                    id: "openai".to_owned(),
                    ..ProxiedProviderConfig::default()
                },
                ProxiedProviderConfig {
                    id: DEDICATED_PROVIDER_ID.to_owned(),
                    ..ProxiedProviderConfig::default()
                },
            ]),
            &proxy_snapshots(),
            native_info,
        );
        assert_eq!(routes.len(), 3);
        let anthropic = &routes["anthropic"];
        assert_eq!(anthropic.base_url, "http://gw.example");
        assert_eq!(
            anthropic.served_model_ids.as_ref().expect("filter"),
            &BTreeSet::from(["claude-sonnet".to_owned()])
        );
        assert!(!anthropic.passthrough);
        assert_eq!(routes["zai"].base_url, "http://gw.example");
        assert!(routes["openai"].passthrough);
        assert_eq!(routes["openai"].base_url, "http://gw.example/v1");

        let overridden = plan_proxy_routes(
            &proxy_resolved(vec![ProxiedProviderConfig {
                id: "anthropic".to_owned(),
                api: "openai-completions".to_owned(),
                ..ProxiedProviderConfig::default()
            }]),
            &proxy_snapshots(),
            native_info,
        );
        assert!(overridden["anthropic"].api_overridden);
        assert_eq!(overridden["anthropic"].api, "openai-completions");
        assert_eq!(overridden["anthropic"].base_url, "http://gw.example/v1");

        let missing = plan_proxy_routes(
            &proxy_resolved(vec![ProxiedProviderConfig {
                id: "anthropic".to_owned(),
                keep_gateway_models_only: true,
                ..ProxiedProviderConfig::default()
            }]),
            &[GatewaySnapshot {
                id: "anthropic".to_owned(),
                models: vec!["not-local".to_owned()],
                ..GatewaySnapshot::default()
            }],
            native_info,
        );
        assert!(missing.is_empty());
    }

    #[test]
    fn proxy_warnings_missing_summary_mapping_and_apply_are_stable() {
        let resolved = proxy_resolved(vec![
            ProxiedProviderConfig {
                id: "anthropic".to_owned(),
                should_check_gateway_models: true,
                api: "openai-completions".to_owned(),
                ..ProxiedProviderConfig::default()
            },
            ProxiedProviderConfig {
                id: "zai".to_owned(),
                ..ProxiedProviderConfig::default()
            },
        ]);
        let gateway = vec![
            GatewayProvider {
                id: "anthropic".to_owned(),
                name: "Anthropic".to_owned(),
                models: vec!["claude-sonnet".to_owned()],
                compatibility: compat(&[("anthropic_messages", true)]),
                ..GatewayProvider::default()
            },
            GatewayProvider {
                id: "zai".to_owned(),
                name: "Z.AI".to_owned(),
                ..GatewayProvider::default()
            },
        ];
        assert_eq!(proxy_override_warnings(&resolved, &gateway).len(), 1);
        let local = vec![
            model("anthropic", "claude-sonnet", "anthropic-messages", ""),
            model("anthropic", "one", "anthropic-messages", ""),
            model("anthropic", "two", "anthropic-messages", ""),
            model("anthropic", "three", "anthropic-messages", ""),
            model("anthropic", "four", "anthropic-messages", ""),
            model("anthropic", "five", "anthropic-messages", ""),
            model("anthropic", "six", "anthropic-messages", ""),
            model("zai", "glm", "openai-completions", ""),
        ];
        let summary = missing_models_summary(&resolved, &gateway, &local);
        assert!(summary.contains("1 more"));
        assert!(!summary.contains("claude-sonnet"));
        assert!(!summary.contains("glm"));

        let mapped = map_proxy_providers(
            &[
                model("anthropic", "claude", "anthropic-messages", ""),
                model("openai", "gpt", "openai-responses", ""),
                model(
                    DEDICATED_PROVIDER_ID,
                    "anthropic/claude",
                    "anthropic-messages",
                    "",
                ),
            ],
            &[
                GatewayProvider {
                    id: "openai".to_owned(),
                    name: "OpenAI".to_owned(),
                    ..GatewayProvider::default()
                },
                gateway[0].clone(),
            ],
            &[ProxiedProviderConfig {
                id: "anthropic".to_owned(),
                keep_gateway_models_only: true,
                ..ProxiedProviderConfig::default()
            }],
        );
        assert_eq!(
            mapped
                .iter()
                .map(|provider| provider.id.as_str())
                .collect::<Vec<_>>(),
            vec!["anthropic", "openai"]
        );
        assert!(mapped[0].enabled);
        assert!(mapped[0].keep_gateway_models_only);
        assert!(!mapped[1].enabled);
        assert!(mapped[1].should_check_gateway_models);

        let routed = apply_proxy_route(
            &model("anthropic", "claude-sonnet", "anthropic-messages", "native"),
            &ProxyRoute {
                base_url: "http://gw.example".to_owned(),
                api: "openai-completions".to_owned(),
                api_overridden: true,
                served_model_ids: Some(BTreeSet::from(["claude-sonnet".to_owned()])),
                ..ProxyRoute::default()
            },
        )
        .expect("served model");
        assert_eq!(routed.base_url, "http://gw.example");
        assert_eq!(routed.api, "openai-completions");
        assert!(
            apply_proxy_route(
                &model("anthropic", "absent", "anthropic-messages", ""),
                &ProxyRoute {
                    served_model_ids: Some(BTreeSet::from(["claude-sonnet".to_owned()])),
                    ..ProxyRoute::default()
                }
            )
            .is_none()
        );
    }

    fn dedicated_providers() -> Vec<GatewayProvider> {
        vec![
            GatewayProvider {
                id: "anthropic".to_owned(),
                name: "Anthropic".to_owned(),
                models: vec!["claude".to_owned()],
                compatibility: compat(&[("anthropic_messages", true)]),
                model_info_by_id: BTreeMap::from([(
                    "claude".to_owned(),
                    ModelInfo {
                        id: "claude".to_owned(),
                        pricing: Some(ModelPricing {
                            input: "0.000003".to_owned(),
                            output: "0.000015".to_owned(),
                            ..ModelPricing::default()
                        }),
                    },
                )]),
                ..GatewayProvider::default()
            },
            GatewayProvider {
                id: "google".to_owned(),
                name: "Google".to_owned(),
                models: vec!["gemini".to_owned()],
                compatibility: compat(&[("gemini_generate_content", true)]),
                ..GatewayProvider::default()
            },
        ]
    }

    #[test]
    fn dedicated_models_merge_metadata_pricing_and_api_overrides() {
        let mut claude = model(
            "anthropic",
            "claude",
            "anthropic-messages",
            "https://api.anthropic.com",
        );
        claude.name = "Claude".to_owned();
        claude.reasoning = true;
        claude.input = vec!["text".to_owned(), "image".to_owned()];
        claude.context_window = 200_000;
        claude.max_tokens = 64_000;
        claude.cost.rates.input = 3.0;
        claude.cost.rates.output = 15.0;
        claude.compat = Some(serde_json::json!({"supportsDeveloperRole":false}));
        let built = build_dedicated_models(
            &dedicated_providers(),
            "http://gw.example",
            "http://gw.example/v1",
            &[claude],
            None,
            &BTreeMap::new(),
        );
        assert!(built.warnings.is_empty());
        assert_eq!(built.models.len(), 2);
        let claude = &built.models[0];
        assert_eq!(claude.id, "anthropic/claude");
        assert_eq!(claude.provider, DEDICATED_PROVIDER_ID);
        assert_eq!(claude.base_url, "http://gw.example");
        assert!(claude.reasoning);
        assert_eq!(claude.context_window, 200_000);
        assert_eq!(claude.cost.rates.input, 3.0);
        assert_eq!(claude.cost.rates.output, 15.0);
        assert!(claude.compat.is_some());
        let gemini = &built.models[1];
        assert_eq!(gemini.id, "gemini");
        assert_eq!(gemini.api, "google-generative-ai");
        assert_eq!(gemini.base_url, "http://gw.example/v1beta");
        assert_eq!(gemini.context_window, 128_000);
        assert_eq!(gemini.max_tokens, 8_192);

        let mut overrides =
            BTreeMap::from([("anthropic".to_owned(), "anthropic-messages".to_owned())]);
        let mut providers = dedicated_providers();
        providers[0].compatibility = compat(&[("anthropic_messages", true), ("openai_chat", true)]);
        let overridden = build_dedicated_models(
            &providers,
            "http://gw.example",
            "http://gw.example/v1",
            &[],
            None,
            &overrides,
        );
        assert_eq!(overridden.models[0].api, "anthropic-messages");
        overrides.insert("google".to_owned(), "anthropic-messages".to_owned());
        let invalid = build_dedicated_models(
            &providers,
            "http://gw.example",
            "http://gw.example/v1",
            &[],
            None,
            &overrides,
        );
        assert_eq!(invalid.models[1].api, "google-generative-ai");
        assert_eq!(invalid.warnings.len(), 1);
    }

    #[test]
    fn models_dev_and_catalog_metadata_obey_precedence_and_fallback_rules() {
        let models_dev = BTreeMap::from([(
            "google".to_owned(),
            ModelsDevProvider {
                models: BTreeMap::from([(
                    "gemini".to_owned(),
                    ModelsDevModel {
                        name: "Gemini 2.5 Pro".to_owned(),
                        reasoning: Some(true),
                        modalities: ModelsDevModalities {
                            input: vec!["text".to_owned(), "image".to_owned(), "audio".to_owned()],
                        },
                        limit: ModelsDevLimit {
                            context: 1_048_576,
                            output: 65_536,
                        },
                        ..ModelsDevModel::default()
                    },
                )]),
            },
        )]);
        let built = build_dedicated_models(
            &dedicated_providers(),
            "http://gw.example",
            "http://gw.example/v1",
            &[],
            Some(&models_dev),
            &BTreeMap::new(),
        );
        let gemini = &built.models[1];
        assert_eq!(gemini.name, "Gemini 2.5 Pro");
        assert!(gemini.reasoning);
        assert_eq!(gemini.input, vec!["text", "image"]);
        assert_eq!(gemini.context_window, 1_048_576);
        assert_eq!(gemini.max_tokens, 65_536);

        let fallback = model("other", "shared", "openai-completions", "");
        let mut fallback = fallback;
        fallback.compat = Some(serde_json::json!({
            "supportsDeveloperRole": true,
            "deferredToolsMode": "kimi"
        }));
        fallback.cost.rates.input = 99.0;
        let metadata = resolve_model_metadata("different", "shared", &[fallback], None);
        assert!(metadata.cost.is_none(), "cross-provider cost is unsafe");
        assert_eq!(
            metadata.compat,
            Some(serde_json::json!({"supportsDeveloperRole":true}))
        );
    }

    #[test]
    fn dedicated_selection_key_cache_and_cost_are_guarded() {
        let resolved = Resolved {
            dedicated_providers: vec![
                DedicatedProviderConfig {
                    id: "anthropic".to_owned(),
                    enabled: true,
                    ..DedicatedProviderConfig::default()
                },
                DedicatedProviderConfig {
                    id: "google".to_owned(),
                    enabled: true,
                    api: "google-generative-ai".to_owned(),
                    ..DedicatedProviderConfig::default()
                },
            ],
            ..Resolved::default()
        };
        let key = build_catalog_key("http://gw.example/path", &resolved);
        assert_eq!(
            key,
            "http://gw.example anthropic,google@google-generative-ai v2"
        );
        assert_ne!(
            key,
            build_catalog_key("http://gw.example.evil", &resolved),
            "origin equality must not use a string prefix"
        );
        assert!(build_catalog_key("http://gw.example", &Resolved::default()).contains(" * "));
        assert_eq!(
            filter_dedicated_providers(
                &dedicated_providers(),
                &Resolved {
                    dedicated_providers: vec![DedicatedProviderConfig {
                        id: "google".to_owned(),
                        enabled: true,
                        ..DedicatedProviderConfig::default()
                    }],
                    ..Resolved::default()
                }
            )[0]
            .id,
            "google"
        );
        let mapped = map_dedicated_providers(
            &[
                GatewayProvider {
                    id: "anthropic".to_owned(),
                    name: "Anthropic".to_owned(),
                    ..GatewayProvider::default()
                },
                GatewayProvider {
                    id: "google".to_owned(),
                    name: "Google".to_owned(),
                    ..GatewayProvider::default()
                },
            ],
            &[DedicatedProviderConfig {
                id: "google".to_owned(),
                enabled: false,
                api: "google-generative-ai".to_owned(),
                ..DedicatedProviderConfig::default()
            }],
        );
        assert!(mapped[0].enabled, "new providers default enabled");
        assert!(!mapped[1].enabled);
        assert_eq!(mapped[1].api, "google-generative-ai");

        let base = llm::ModelCost {
            rates: llm::ModelCostRates {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
            ..llm::ModelCost::default()
        };
        let cost = merge_cost(
            Some(&ModelPricing {
                input_cache_read: "0.0000001".to_owned(),
                ..ModelPricing::default()
            }),
            Some(&base),
        );
        assert_eq!(cost.rates.input, 3.0);
        assert_eq!(cost.rates.output, 15.0);
        assert!((cost.rates.cache_read - 0.1).abs() < 1e-9);

        let path = temporary_path("cache").join("aperture-cache.json");
        let cached_model = llm::Model {
            id: "anthropic/claude".to_owned(),
            provider: DEDICATED_PROVIDER_ID.to_owned(),
            compat: Some(serde_json::json!({"forceAdaptiveThinking":true})),
            ..llm::Model::default()
        };
        let cache = new_cache("key-1", &[cached_model], &dedicated_providers());
        save_cache(&path, &cache).expect("save cache");
        let loaded = load_cache(&path).expect("load cache");
        let restored = loaded.catalog_models("key-1").expect("matching cache");
        assert_eq!(
            restored[0].compat,
            Some(serde_json::json!({"forceAdaptiveThinking":true}))
        );
        assert!(loaded.catalog_models("other-key").is_none());
        assert_eq!(loaded.gateway.len(), 2);
        let _ = fs::remove_dir_all(path.parent().expect("parent"));
    }

    #[test]
    fn connector_state_deduplicates_splits_and_persists_pins() {
        let tools = vec![
            GatewayTool {
                name: "github_list_repos".to_owned(),
                ..GatewayTool::default()
            },
            GatewayTool {
                name: "github_create_issue".to_owned(),
                ..GatewayTool::default()
            },
            GatewayTool {
                name: "slack_send".to_owned(),
                ..GatewayTool::default()
            },
            GatewayTool {
                name: "orphan".to_owned(),
                ..GatewayTool::default()
            },
            GatewayTool {
                name: "github_list_repos".to_owned(),
                description: "duplicate".to_owned(),
                ..GatewayTool::default()
            },
        ];
        let resolved = Resolved {
            discovery_tools: true,
            pinned_tools: vec![
                PinnedConnectorTool {
                    connector_id: "github".to_owned(),
                    tool_name: "github_list_repos".to_owned(),
                },
                PinnedConnectorTool {
                    connector_id: "github".to_owned(),
                    tool_name: "gone".to_owned(),
                },
            ],
            ..Resolved::default()
        };
        let set = build_connector_tool_set(
            &resolved,
            &[
                ConnectorInfo {
                    id: "github".to_owned(),
                    provider: "GitHub".to_owned(),
                    ..ConnectorInfo::default()
                },
                ConnectorInfo {
                    id: "slack".to_owned(),
                    provider: "Slack".to_owned(),
                    ..ConnectorInfo::default()
                },
                ConnectorInfo {
                    id: "empty".to_owned(),
                    ..ConnectorInfo::default()
                },
            ],
            &tools,
        );
        assert_eq!(
            set.pinned_tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            vec!["github_list_repos"]
        );
        assert_eq!(set.proxied_tools.len(), 3);
        assert_eq!(set.missing_pins, vec!["gone"]);
        assert_eq!(set.visible_connectors.len(), 2);
        assert!(set.discovery_enabled);
        assert_eq!(connector_id_from_tool_name("_bad"), "other");
        assert_eq!(connector_id_from_tool_name("github_"), "github");

        let mut config = Config::default();
        let change = pin_connector_tool(&mut config, "github_list_repos", &tools).expect("pin");
        assert_eq!(change.connector_id, "github");
        assert_eq!(change.pin_count, 1);
        assert!(matches!(
            pin_connector_tool(&mut config, "github_list_repos", &tools),
            Err(ApertureError::ToolAlreadyPinned(_))
        ));
        assert!(matches!(
            pin_connector_tool(&mut config, "missing", &tools),
            Err(ApertureError::ToolNotFound(_))
        ));
        let change = unpin_connector_tool(&mut config, "github_list_repos").expect("unpin");
        assert_eq!(change.pin_count, 0);
        assert_eq!(
            config
                .connectors
                .as_ref()
                .and_then(|connectors| connectors.pinned_tools.as_ref())
                .expect("explicit empty pins")
                .len(),
            0
        );
    }

    #[test]
    fn aperture_state_and_request_rewrite_preserve_input_models() {
        let state = ApertureState {
            configured: true,
            routes: BTreeMap::from([(
                "anthropic".to_owned(),
                ProxyRoute {
                    provider_id: "anthropic".to_owned(),
                    api: "anthropic-messages".to_owned(),
                    base_url: "http://gw.example".to_owned(),
                    ..ProxyRoute::default()
                },
            )]),
            ..ApertureState::default()
        };
        let mut proxied = model("anthropic", "claude", "anthropic-messages", "native");
        proxied
            .headers
            .insert("anthropic-beta".to_owned(), "x".to_owned());
        let rewritten = rewrite_request_model(Some(&state), &proxied, "session-1");
        let Cow::Owned(rewritten) = rewritten else {
            panic!("proxied model must be rewritten");
        };
        assert_eq!(rewritten.id, "anthropic/claude");
        assert_eq!(rewritten.headers["Referer"], APERTURE_REFERER);
        assert_eq!(rewritten.headers["x-session-id"], "session-1");
        assert_eq!(rewritten.headers["anthropic-beta"], "x");
        assert_eq!(proxied.id, "claude");
        assert!(!proxied.headers.contains_key("x-session-id"));

        let dedicated = model(
            DEDICATED_PROVIDER_ID,
            "google/gemini",
            "google-generative-ai",
            "http://gw.example/v1beta",
        );
        assert_eq!(
            rewrite_request_model(Some(&state), &dedicated, "session")
                .into_owned()
                .id,
            "gemini"
        );
        let body_dedicated = model(
            DEDICATED_PROVIDER_ID,
            "anthropic/claude",
            "anthropic-messages",
            "http://gw.example",
        );
        assert_eq!(
            rewrite_request_model(Some(&state), &body_dedicated, "")
                .into_owned()
                .id,
            "anthropic/claude"
        );
        let plain = model("openai", "gpt", "openai-responses", "native");
        assert!(matches!(
            rewrite_request_model(Some(&state), &plain, "session"),
            Cow::Borrowed(_)
        ));
        assert!(matches!(
            rewrite_request_model(Some(&ApertureState::default()), &proxied, "session"),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn retry_classification_marks_only_unmarked_restarts() {
        assert_eq!(
            classify_retryable_error("Aperture is restarting, please hold"),
            RetryClassification::MarkRetryable
        );
        assert_eq!(
            mark_retryable_error("Aperture is restarting, please hold").as_deref(),
            Some("Aperture is restarting, please hold (service unavailable)")
        );
        assert_eq!(
            classify_retryable_error("APERTURE IS RESTARTING: service unavailable"),
            RetryClassification::AlreadyRetryable
        );
        assert!(mark_retryable_error("upstream rejected the key").is_none());
    }

    #[test]
    fn cached_state_replays_only_matching_models_and_plans_proxy_routes() {
        let config = Config {
            base_url: "http://gw.example".to_owned(),
            proxy: Some(ProxyConfig {
                enabled: Some(true),
                upstream_providers: Some(vec![ProxiedProviderConfig {
                    id: "anthropic".to_owned(),
                    ..ProxiedProviderConfig::default()
                }]),
            }),
            dedicated: Some(DedicatedConfig {
                enabled: Some(true),
                providers: Some(Vec::new()),
                ..DedicatedConfig::default()
            }),
            ..Config::default()
        };
        let resolved = config.resolve();
        let cache = Cache {
            catalog_key: build_catalog_key("http://gw.example", &resolved),
            models: vec![CachedModel {
                model: model(
                    DEDICATED_PROVIDER_ID,
                    "anthropic/claude",
                    "anthropic-messages",
                    "",
                ),
                raw_compat: None,
            }],
            gateway: proxy_snapshots(),
            ..Cache::default()
        };
        let state = build_aperture_state(&config, Some(&cache), native_info);
        assert!(state.configured);
        assert_eq!(state.dedicated_models.len(), 1);
        assert!(state.routes.contains_key("anthropic"));
    }
}
