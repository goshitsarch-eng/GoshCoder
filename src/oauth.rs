//! OAuth credential flows for the Rust runtime.
//!
//! This module deliberately has no dependency on the terminal UI.  It exposes
//! blocking, testable primitives for browser/loopback and device-code OAuth
//! flows, plus provider-specific implementations ported from
//! `internal/llm/catalog/oauth*.go`.
//!
//! `src/main.rs` does not declare this module yet, by design: the migration can
//! add `mod oauth;` only when the runtime is ready to wire the APIs below.
//!
//! # Catalog/runtime integration contract
//!
//! Once this module is declared, the catalog should:
//!
//! 1. Keep an `OAuthClient` alongside its `CredentialStore`.
//! 2. Call [`OAuthClient::resolve_stored_oauth`] before falling back to
//!    environment credentials. A stored OAuth credential deliberately owns its
//!    provider; refresh failure must not silently use an ambient API key.
//! 3. Call [`OAuthClient::login_and_persist`] from `auth login`, supplying a UI
//!    implementation of [`OAuthInteraction`].
//! 4. Cache a refresh error per catalog instance and clear it after a
//!    successful login, so provider-picker rebuilds do not repeatedly spend a
//!    token-request timeout on a known-bad endpoint.
//! 5. Convert the resulting [`OAuthAuth`] into `catalog::Auth` inside
//!    `catalog.rs`, where `Auth::with_api_key` and `Auth::without_api_key` are
//!    visible. [`OAuthAuthAdapter`] makes that boundary explicit without
//!    exposing `Auth` constructors or secrets from this module.
//!
//! The existing catalog's `Auth` constructors are private to `catalog.rs`, so
//! Rust's privacy rules make it impossible for this sibling module to create
//! `Auth` directly without changing that file. This is an intentional
//! integration seam rather than a duplicated or less-safe auth type.

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fmt,
    io::{self, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, TryRecvError},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::{
    Method,
    blocking::Client,
    header::{HeaderMap, HeaderName, HeaderValue},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use url::Url;
use uuid::Uuid;

use crate::catalog::{
    Auth, AuthHeaders, CatalogError, Credential, CredentialKind, CredentialStore, EnvironmentLookup,
};

/// A token request is bounded independently from a model request.
pub const DEFAULT_TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// xAI's discovery document is advisory and must not hold a store lock long.
pub const XAI_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(5);
/// OAuth tokens are refreshed before they enter their final validity window.
pub const MINIMUM_VALIDITY: Duration = Duration::from_secs(5 * 60);
/// The maximum number of Kimi retries after its initial refresh request.
pub const KIMI_REFRESH_MAX_RETRIES: u32 = 3;
/// The maximum token response body retained in memory.
pub const MAX_TOKEN_RESPONSE_BYTES: usize = 1024 * 1024;

const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const ANTHROPIC_CALLBACK_PORT: u16 = 53692;
const ANTHROPIC_CALLBACK_PATH: &str = "/callback";
const ANTHROPIC_REDIRECT_URI: &str = "http://localhost:53692/callback";
const ANTHROPIC_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";
const ANTHROPIC_REFRESH_SKEW: Duration = Duration::from_secs(5 * 60);

const KIMI_CLIENT_ID: &str = "17e5f671-d194-4dfb-9706-5516cb48c098";
const KIMI_DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";

const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_CALLBACK_PORT: u16 = 1455;
const CODEX_CALLBACK_PATH: &str = "/auth/callback";
const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
const CODEX_DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
const CODEX_SCOPE: &str = "openid profile email offline_access";
const CODEX_AUTH_CLAIM: &str = "https://api.openai.com/auth";

const XAI_DEFAULT_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const XAI_CALLBACK_PORT: u16 = 56121;
const XAI_CALLBACK_PATH: &str = "/callback";
const XAI_REDIRECT_URI: &str = "http://127.0.0.1:56121/callback";
const XAI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const XAI_DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";

const META_DEFAULT_CLIENT_ID: &str = "1031625952748946";
const META_DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const META_API_VERSION: &str = "1.0.0";
const META_KEY_VALIDITY: Duration = Duration::from_secs(20 * 60 * 60);
const META_REFRESH_TOKEN_EXTRA: &str = "metaRefreshToken";
const META_IDENTITY_EXPIRES_EXTRA: &str = "metaIdentityExpires";

const DEFAULT_DEVICE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const DEFAULT_DEVICE_INTERVAL: Duration = Duration::from_secs(5);
const DEVICE_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Result type used by this module.
pub type Result<T> = std::result::Result<T, OAuthError>;

/// Errors intentionally avoid including credential values.
#[derive(Debug)]
pub enum OAuthError {
    Cancelled,
    Unauthorized {
        provider: &'static str,
        operation: &'static str,
        detail: Option<String>,
    },
    UnsupportedFlow {
        provider: String,
    },
    InvalidConfiguration(String),
    InvalidUrl(String),
    Transport(String),
    TokenFailure {
        provider: &'static str,
        operation: &'static str,
        status: u16,
        detail: String,
    },
    InvalidTokenResponse {
        provider: &'static str,
        operation: &'static str,
    },
    InvalidAuthorizationInput,
    StateMismatch,
    Callback(String),
    DeviceExpired {
        provider: &'static str,
    },
    DeviceDenied {
        provider: &'static str,
    },
    DeviceTimedOut,
    Jwt(String),
    Storage(CatalogError),
}

impl OAuthError {
    /// Whether retrying cannot repair this credential and a fresh login is
    /// required.
    pub fn is_unauthorized(&self) -> bool {
        matches!(self, Self::Unauthorized { .. })
    }
}

impl fmt::Display for OAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("OAuth login was cancelled"),
            Self::Unauthorized {
                provider,
                operation,
                detail,
            } => {
                write!(
                    formatter,
                    "{provider} token {operation} is no longer authorized; log in again"
                )?;
                if let Some(detail) = detail.as_ref().filter(|detail| !detail.is_empty()) {
                    write!(formatter, ": {detail}")?;
                }
                Ok(())
            }
            Self::UnsupportedFlow { provider } => {
                write!(
                    formatter,
                    "no OAuth login or refresh flow is available for {provider:?}"
                )
            }
            Self::InvalidConfiguration(message) => formatter.write_str(message),
            Self::InvalidUrl(message) => write!(formatter, "invalid OAuth URL: {message}"),
            Self::Transport(message) => {
                write!(formatter, "OAuth network request failed: {message}")
            }
            Self::TokenFailure {
                provider,
                operation,
                status,
                detail,
            } => write!(
                formatter,
                "{provider} token {operation} failed (status {status}): {detail}"
            ),
            Self::InvalidTokenResponse {
                provider,
                operation,
            } => write!(
                formatter,
                "{provider} token {operation} returned an incomplete or invalid response"
            ),
            Self::InvalidAuthorizationInput => {
                formatter.write_str("no authorization code was provided")
            }
            Self::StateMismatch => formatter.write_str("OAuth state mismatch"),
            Self::Callback(message) => {
                write!(formatter, "OAuth loopback callback failed: {message}")
            }
            Self::DeviceExpired { provider } => {
                write!(
                    formatter,
                    "{provider} device authorization expired; restart login"
                )
            }
            Self::DeviceDenied { provider } => write!(formatter, "{provider} login was denied"),
            Self::DeviceTimedOut => formatter.write_str("OAuth device flow timed out"),
            Self::Jwt(message) => write!(
                formatter,
                "failed to extract account ID from token: {message}"
            ),
            Self::Storage(error) => write!(formatter, "OAuth credential storage failed: {error}"),
        }
    }
}

impl Error for OAuthError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Storage(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CatalogError> for OAuthError {
    fn from(error: CatalogError) -> Self {
        Self::Storage(error)
    }
}

/// Cooperative cancellation for device polling, backoff, prompts, and
/// loopback waiting.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(OAuthError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// A clock whose sleeps can be replaced in tests.
pub trait OAuthClock: Send + Sync {
    fn now_ms(&self) -> i64;
    fn sleep(&self, duration: Duration, cancellation: &CancellationToken) -> Result<()>;
}

/// The production clock. Long waits are split into short slices so
/// cancellation is observed promptly.
#[derive(Default)]
pub struct SystemClock;

impl OAuthClock for SystemClock {
    fn now_ms(&self) -> i64 {
        unix_millis(SystemTime::now())
    }

    fn sleep(&self, duration: Duration, cancellation: &CancellationToken) -> Result<()> {
        let deadline = Instant::now()
            .checked_add(duration)
            .unwrap_or_else(Instant::now);
        loop {
            cancellation.check()?;
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(());
            }
            thread::sleep(remaining.min(Duration::from_millis(25)));
        }
    }
}

fn unix_millis(time: SystemTime) -> i64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn duration_millis(duration: Duration) -> i64 {
    duration.as_millis().min(i64::MAX as u128) as i64
}

/// Environment lookup used by provider endpoint/client-ID overrides.
pub trait OAuthEnvironment: Send + Sync {
    fn value(&self, name: &str) -> Option<String>;
}

impl OAuthEnvironment for BTreeMap<String, String> {
    fn value(&self, name: &str) -> Option<String> {
        self.get(name).filter(|value| !value.is_empty()).cloned()
    }
}

/// Process-backed environment lookup for command-line use.
#[derive(Default)]
pub struct ProcessEnvironment;

impl OAuthEnvironment for ProcessEnvironment {
    fn value(&self, name: &str) -> Option<String> {
        env::var(name).ok().filter(|value| !value.is_empty())
    }
}

/// Adapter for the catalog's injectable environment lookup.
#[derive(Clone)]
pub struct CatalogEnvironment {
    lookup: EnvironmentLookup,
}

impl CatalogEnvironment {
    pub fn new(lookup: EnvironmentLookup) -> Self {
        Self { lookup }
    }
}

impl OAuthEnvironment for CatalogEnvironment {
    fn value(&self, name: &str) -> Option<String> {
        (self.lookup)(name).filter(|value| !value.is_empty())
    }
}

/// Provider IDs which either have a Go flow or are marked OAuth-capable in the
/// Go provider catalog.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum OAuthProviderId {
    Anthropic,
    KimiCoding,
    Meta,
    OpenAiCodex,
    OpenRouter,
    Xai,
    GithubCopilot,
    Radius,
}

impl OAuthProviderId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::KimiCoding => "kimi-coding",
            Self::Meta => "meta",
            Self::OpenAiCodex => "openai-codex",
            Self::OpenRouter => "openrouter",
            Self::Xai => "xai",
            Self::GithubCopilot => "github-copilot",
            Self::Radius => "radius",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "anthropic" => Some(Self::Anthropic),
            "kimi-coding" => Some(Self::KimiCoding),
            "meta" => Some(Self::Meta),
            "openai-codex" => Some(Self::OpenAiCodex),
            "openrouter" => Some(Self::OpenRouter),
            "xai" => Some(Self::Xai),
            "github-copilot" => Some(Self::GithubCopilot),
            "radius" => Some(Self::Radius),
            _ => None,
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic (Claude Pro/Max)",
            Self::KimiCoding => "Kimi Code (subscription)",
            Self::Meta => "Meta (Model API)",
            Self::OpenAiCodex => "OpenAI (ChatGPT Plus/Pro)",
            Self::OpenRouter => "OpenRouter",
            Self::Xai => "xAI (Grok subscription)",
            Self::GithubCopilot => "GitHub Copilot",
            Self::Radius => "Radius",
        }
    }
}

/// The interaction method a provider makes available.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoginMethod {
    BrowserPkce,
    DeviceCode,
    ApiKeyOnly,
}

/// Whether this repository's Go implementation actually supplied a flow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OAuthFlowSupport {
    Implemented,
    MetadataOnly,
}

/// Static provider metadata for login selectors and status views.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProviderMetadata {
    pub id: OAuthProviderId,
    pub display_name: &'static str,
    pub methods: &'static [LoginMethod],
    pub flow_support: OAuthFlowSupport,
}

const BROWSER_METHOD: &[LoginMethod] = &[LoginMethod::BrowserPkce];
const DEVICE_METHOD: &[LoginMethod] = &[LoginMethod::DeviceCode];
const CODEX_METHODS: &[LoginMethod] = &[LoginMethod::BrowserPkce, LoginMethod::DeviceCode];
const XAI_METHODS: &[LoginMethod] = &[LoginMethod::DeviceCode, LoginMethod::BrowserPkce];
const API_KEY_METHOD: &[LoginMethod] = &[LoginMethod::ApiKeyOnly];

const PROVIDER_METADATA: &[ProviderMetadata] = &[
    ProviderMetadata {
        id: OAuthProviderId::Anthropic,
        display_name: "Anthropic (Claude Pro/Max)",
        methods: BROWSER_METHOD,
        flow_support: OAuthFlowSupport::Implemented,
    },
    ProviderMetadata {
        id: OAuthProviderId::KimiCoding,
        display_name: "Kimi Code (subscription)",
        methods: DEVICE_METHOD,
        flow_support: OAuthFlowSupport::Implemented,
    },
    ProviderMetadata {
        id: OAuthProviderId::Meta,
        display_name: "Meta (Model API)",
        methods: DEVICE_METHOD,
        flow_support: OAuthFlowSupport::Implemented,
    },
    ProviderMetadata {
        id: OAuthProviderId::OpenAiCodex,
        display_name: "OpenAI (ChatGPT Plus/Pro)",
        methods: CODEX_METHODS,
        flow_support: OAuthFlowSupport::Implemented,
    },
    // The Go provider metadata marks OpenRouter as OAuth-capable, but no
    // OpenRouter authorization, device, or refresh implementation exists in
    // any `oauth*.go` file. Preserve that fact instead of inventing endpoints.
    ProviderMetadata {
        id: OAuthProviderId::OpenRouter,
        display_name: "OpenRouter",
        methods: API_KEY_METHOD,
        flow_support: OAuthFlowSupport::MetadataOnly,
    },
    ProviderMetadata {
        id: OAuthProviderId::Xai,
        display_name: "xAI (Grok subscription)",
        methods: XAI_METHODS,
        flow_support: OAuthFlowSupport::Implemented,
    },
    // These are also `oauth: true` in Go provider metadata, but Go has no
    // oauth*.go implementations for them.
    ProviderMetadata {
        id: OAuthProviderId::GithubCopilot,
        display_name: "GitHub Copilot",
        methods: API_KEY_METHOD,
        flow_support: OAuthFlowSupport::MetadataOnly,
    },
    ProviderMetadata {
        id: OAuthProviderId::Radius,
        display_name: "Radius",
        methods: API_KEY_METHOD,
        flow_support: OAuthFlowSupport::MetadataOnly,
    },
];

/// Returns every OAuth-marked Go provider, including metadata-only entries.
pub fn provider_metadata() -> &'static [ProviderMetadata] {
    PROVIDER_METADATA
}

/// Looks up OAuth metadata by provider ID.
pub fn metadata_for(provider: OAuthProviderId) -> &'static ProviderMetadata {
    PROVIDER_METADATA
        .iter()
        .find(|metadata| metadata.id == provider)
        .expect("every OAuthProviderId has static metadata")
}

/// Returns the providers which have a concrete Go OAuth flow to port.
pub fn implemented_provider_ids() -> Vec<&'static str> {
    PROVIDER_METADATA
        .iter()
        .filter(|metadata| metadata.flow_support == OAuthFlowSupport::Implemented)
        .map(|metadata| metadata.id.as_str())
        .collect()
}

/// Provider-specific auth shape that can be converted to `catalog::Auth`.
///
/// It intentionally does not implement `Debug` or `Display` because it can
/// contain an access token or a bearer header.
#[derive(Clone, Eq, PartialEq)]
pub struct OAuthAuth {
    api_key: Option<String>,
    headers: AuthHeaders,
    source: String,
}

impl OAuthAuth {
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn headers(&self) -> &AuthHeaders {
        &self.headers
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    /// Moves the secret-bearing parts across the catalog integration seam.
    pub fn into_parts(self) -> (Option<String>, AuthHeaders, String) {
        (self.api_key, self.headers, self.source)
    }

    /// Converts into catalog auth through a factory implemented inside
    /// `catalog.rs`, where the private `Auth` constructors are accessible.
    pub fn into_catalog_auth<A: OAuthAuthAdapter>(self, adapter: &A) -> Auth {
        adapter.build_catalog_auth(self)
    }
}

/// The one small bridge `catalog.rs` must implement to create its private
/// `Auth` type from an OAuth result.
pub trait OAuthAuthAdapter {
    fn build_catalog_auth(&self, auth: OAuthAuth) -> Auth;
}

/// Derives request auth for a usable stored OAuth credential.
pub fn auth_from_credential(
    provider: OAuthProviderId,
    credential: &Credential,
) -> Result<OAuthAuth> {
    let access = credential.access();
    if access.is_empty() {
        return Err(OAuthError::InvalidTokenResponse {
            provider: provider.display_name(),
            operation: "use",
        });
    }

    let mut headers = AuthHeaders::new();
    let api_key = match provider {
        OAuthProviderId::KimiCoding => {
            headers.insert("Authorization".to_owned(), Some(format!("Bearer {access}")));
            Some(access.to_owned())
        }
        OAuthProviderId::Meta => {
            headers.insert("Authorization".to_owned(), Some(format!("Bearer {access}")));
            None
        }
        OAuthProviderId::Anthropic
        | OAuthProviderId::OpenAiCodex
        | OAuthProviderId::OpenRouter
        | OAuthProviderId::Xai
        | OAuthProviderId::GithubCopilot
        | OAuthProviderId::Radius => Some(access.to_owned()),
    };
    Ok(OAuthAuth {
        api_key,
        headers,
        source: "OAuth".to_owned(),
    })
}

/// Returns whether an OAuth credential needs a refresh at `now_ms`.
pub fn credential_expires_soon_at(
    credential: &Credential,
    now_ms: i64,
    minimum_validity: Duration,
) -> bool {
    now_ms.saturating_add(duration_millis(minimum_validity)) >= credential.expires_at_ms()
}

/// Returns whether an OAuth credential needs a refresh according to `clock`.
pub fn credential_expires_soon(credential: &Credential, clock: &dyn OAuthClock) -> bool {
    credential_expires_soon_at(credential, clock.now_ms(), MINIMUM_VALIDITY)
}

/// A PKCE verifier and its RFC 7636 S256 challenge.
///
/// The verifier is secret enough to authorize a code exchange and is therefore
/// intentionally not `Debug`.
#[derive(Clone, Eq, PartialEq)]
pub struct PkcePair {
    verifier: String,
    challenge: String,
}

impl PkcePair {
    pub fn verifier(&self) -> &str {
        &self.verifier
    }

    pub fn challenge(&self) -> &str {
        &self.challenge
    }
}

/// Generates a 32-byte PKCE verifier and the corresponding S256 challenge.
///
/// `uuid` is already a direct dependency with its `v7` feature enabled.
/// Eight v7 UUID samples are hashed to retain at least 256 bits of fresh
/// native CSPRNG material while avoiding a new random-number dependency.
pub fn generate_pkce() -> PkcePair {
    let verifier = URL_SAFE_NO_PAD.encode(random_material(b"goshcoder/oauth/pkce"));
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    PkcePair {
        verifier,
        challenge,
    }
}

/// Generates an opaque state value for CSRF protection.
pub fn random_state() -> String {
    URL_SAFE_NO_PAD.encode(random_material(b"goshcoder/oauth/state"))
}

fn random_material(domain: &[u8]) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(domain);
    for _ in 0..8 {
        digest.update(Uuid::now_v7().as_bytes());
    }
    digest.finalize().into()
}

/// Input returned by either a loopback callback or a manual browser paste.
///
/// This intentionally avoids `Debug` because `code` is exchangeable for
/// tokens.
#[derive(Clone, Eq, PartialEq)]
pub struct AuthorizationResponse {
    pub code: String,
    pub state: String,
}

/// Parses a complete redirect URI, a `code#state` pair, a query fragment, or a
/// bare authorization code.
pub fn parse_authorization_input(input: &str) -> Option<AuthorizationResponse> {
    let value = input.trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(url) = Url::parse(value)
        && !url.scheme().is_empty()
        && url.host_str().is_some()
    {
        return url
            .query_pairs()
            .find(|(name, _)| name == "code")
            .map(|(_, code)| AuthorizationResponse {
                code: code.into_owned(),
                state: url
                    .query_pairs()
                    .find(|(name, _)| name == "state")
                    .map(|(_, state)| state.into_owned())
                    .unwrap_or_default(),
            })
            .filter(|response| !response.code.is_empty());
    }

    if let Some((code, state)) = value.split_once('#') {
        return (!code.is_empty()).then(|| AuthorizationResponse {
            code: code.to_owned(),
            state: state.to_owned(),
        });
    }

    if value.contains("code=") {
        let pairs = url::form_urlencoded::parse(value.as_bytes());
        let mut code = None;
        let mut state = None;
        for (name, value) in pairs {
            match name.as_ref() {
                "code" => code = Some(value.into_owned()),
                "state" => state = Some(value.into_owned()),
                _ => {}
            }
        }
        return code
            .filter(|code| !code.is_empty())
            .map(|code| AuthorizationResponse {
                code,
                state: state.unwrap_or_default(),
            });
    }

    Some(AuthorizationResponse {
        code: value.to_owned(),
        state: String::new(),
    })
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.as_bytes()
        .iter()
        .zip(right.as_bytes())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

/// A prompt a terminal UI must render.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OAuthPromptKind {
    Select,
    ManualCode,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OAuthPromptOption {
    pub id: String,
    pub label: String,
    pub description: String,
}

/// Prompt data plus a cancellation token which a blocking UI must observe.
#[derive(Clone)]
pub struct OAuthPrompt {
    pub kind: OAuthPromptKind,
    pub message: String,
    pub placeholder: String,
    pub options: Vec<OAuthPromptOption>,
    pub cancellation: CancellationToken,
}

/// Progress data a terminal UI must display without blocking the flow.
///
/// It intentionally does not implement `Debug`: authorization URLs contain a
/// CSRF state value.
#[derive(Clone)]
pub struct OAuthEvent {
    pub kind: OAuthEventKind,
    pub message: String,
    pub authorization_url: Option<String>,
    pub instructions: String,
    pub user_code: String,
    pub verification_uri: String,
    pub interval_seconds: u64,
    pub expires_in_seconds: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OAuthEventKind {
    Info,
    AuthorizationUrl,
    DeviceCode,
    Progress,
}

impl OAuthEvent {
    fn info(message: impl Into<String>) -> Self {
        Self {
            kind: OAuthEventKind::Info,
            message: message.into(),
            authorization_url: None,
            instructions: String::new(),
            user_code: String::new(),
            verification_uri: String::new(),
            interval_seconds: 0,
            expires_in_seconds: 0,
        }
    }

    fn authorization_url(url: &Url) -> Self {
        Self {
            kind: OAuthEventKind::AuthorizationUrl,
            message: String::new(),
            authorization_url: Some(url.as_str().to_owned()),
            instructions: "Complete login in your browser. If the browser is on another machine, paste the final redirect URL here.".to_owned(),
            user_code: String::new(),
            verification_uri: String::new(),
            interval_seconds: 0,
            expires_in_seconds: 0,
        }
    }

    fn device_code(
        user_code: impl Into<String>,
        verification_uri: impl Into<String>,
        interval: Duration,
        timeout: Duration,
    ) -> Self {
        Self {
            kind: OAuthEventKind::DeviceCode,
            message: String::new(),
            authorization_url: None,
            instructions: String::new(),
            user_code: user_code.into(),
            verification_uri: verification_uri.into(),
            interval_seconds: interval.as_secs(),
            expires_in_seconds: timeout.as_secs(),
        }
    }

    fn progress(message: impl Into<String>) -> Self {
        Self {
            kind: OAuthEventKind::Progress,
            message: message.into(),
            authorization_url: None,
            instructions: String::new(),
            user_code: String::new(),
            verification_uri: String::new(),
            interval_seconds: 0,
            expires_in_seconds: 0,
        }
    }
}

/// Abstraction used by OAuth flows to interact with Ratatui, a CLI, or tests.
///
/// `prompt` must return promptly after `prompt.cancellation` is cancelled.
/// That lets a loopback callback win its race with a manual paste prompt.
pub trait OAuthInteraction: Send + Sync {
    fn prompt(&self, prompt: OAuthPrompt) -> Result<String>;
    fn notify(&self, event: OAuthEvent);
}

/// Browser launcher abstraction. Browser launch failure is intentionally
/// non-fatal because the URL is always shown and may be opened elsewhere.
pub trait BrowserOpener: Send + Sync {
    fn open(&self, url: &Url) -> Result<()>;
}

/// Production browser opener using the platform's URL launcher.
#[derive(Default)]
pub struct SystemBrowser;

impl BrowserOpener for SystemBrowser {
    fn open(&self, url: &Url) -> Result<()> {
        if !matches!(url.scheme(), "http" | "https") {
            return Err(OAuthError::InvalidUrl(
                "refusing to open a non-HTTP authorization URL".to_owned(),
            ));
        }

        #[cfg(target_os = "windows")]
        let result = Command::new("rundll32")
            .args(["url.dll,FileProtocolHandler", url.as_str()])
            .spawn();
        #[cfg(target_os = "macos")]
        let result = Command::new("open").arg(url.as_str()).spawn();
        #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
        let result = Command::new("xdg-open").arg(url.as_str()).spawn();

        result
            .map(|_| ())
            .map_err(|error| OAuthError::Transport(format!("could not open browser: {error}")))
    }
}

/// Browser opener suitable for headless use and unit tests.
#[derive(Default)]
pub struct NoopBrowser;

impl BrowserOpener for NoopBrowser {
    fn open(&self, _: &Url) -> Result<()> {
        Ok(())
    }
}

/// A validated loopback callback listener. It accepts only the expected path
/// and CSRF state, and returns an escaped, no-store browser response.
pub struct LoopbackCallbackServer {
    listener: TcpListener,
    expected_path: String,
    expected_state: String,
}

impl LoopbackCallbackServer {
    /// Binds a loopback-only callback listener. `localhost` and literal
    /// loopback IPs are accepted; wildcard and remote interfaces are refused.
    pub fn bind(host: &str, port: u16, path: &str, expected_state: &str) -> Result<Self> {
        if !is_loopback_host(host) {
            return Err(OAuthError::Callback(format!(
                "callback host must be loopback, got {host:?}"
            )));
        }
        if !path.starts_with('/') {
            return Err(OAuthError::Callback(
                "callback path must begin with '/'".to_owned(),
            ));
        }
        let listener = TcpListener::bind((host, port)).map_err(|error| {
            OAuthError::Callback(format!("cannot listen on {host}:{port}: {error}"))
        })?;
        if !listener
            .local_addr()
            .map(|address| address.ip().is_loopback())
            .unwrap_or(false)
        {
            return Err(OAuthError::Callback(
                "callback listener did not bind to loopback".to_owned(),
            ));
        }
        listener.set_nonblocking(true).map_err(|error| {
            OAuthError::Callback(format!("cannot configure callback listener: {error}"))
        })?;
        Ok(Self {
            listener,
            expected_path: path.to_owned(),
            expected_state: expected_state.to_owned(),
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(|error| {
            OAuthError::Callback(format!("cannot inspect callback address: {error}"))
        })
    }

    /// Accepts and validates at most one pending callback.
    pub fn try_accept(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<Option<AuthorizationResponse>> {
        let (stream, _) = match self.listener.accept() {
            Ok(pair) => pair,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(None),
            Err(error) => {
                return Err(OAuthError::Callback(format!(
                    "cannot accept callback connection: {error}"
                )));
            }
        };
        self.handle_connection(stream, cancellation)
    }

    fn handle_connection(
        &self,
        mut stream: TcpStream,
        cancellation: &CancellationToken,
    ) -> Result<Option<AuthorizationResponse>> {
        let request = read_callback_request(&mut stream, cancellation)?;
        let Some(request) = request else {
            let _ = write_callback_page(&mut stream, 408, "Callback request timed out.");
            return Ok(None);
        };
        let mut fields = request.split_whitespace();
        let method = fields.next().unwrap_or_default();
        let target = fields.next().unwrap_or_default();
        if method != "GET" {
            let _ = write_callback_page(&mut stream, 405, "Only GET callbacks are supported.");
            return Ok(None);
        }

        let parsed = Url::parse(&format!("http://localhost{target}")).map_err(|_| {
            OAuthError::Callback("callback request target was not a valid URL".to_owned())
        })?;
        if parsed.path() != self.expected_path {
            let _ = write_callback_page(&mut stream, 404, "Callback route not found.");
            return Ok(None);
        }
        let error = query_value(&parsed, "error");
        if !error.is_empty() {
            let _ = write_callback_page(
                &mut stream,
                400,
                &format!("Authentication did not complete: {error}"),
            );
            return Ok(None);
        }
        let code = query_value(&parsed, "code");
        let state = query_value(&parsed, "state");
        if code.is_empty() || state.is_empty() {
            let _ = write_callback_page(&mut stream, 400, "Missing code or state parameter.");
            return Ok(None);
        }
        if !constant_time_eq(&state, &self.expected_state) {
            let _ = write_callback_page(&mut stream, 400, "State mismatch.");
            return Ok(None);
        }
        let _ = write_callback_page(
            &mut stream,
            200,
            "Authentication completed. You can close this window.",
        );
        Ok(Some(AuthorizationResponse { code, state }))
    }
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false)
}

fn read_callback_request(
    stream: &mut TcpStream,
    cancellation: &CancellationToken,
) -> Result<Option<String>> {
    stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .map_err(|error| {
            OAuthError::Callback(format!("cannot configure callback socket: {error}"))
        })?;
    let started = Instant::now();
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        cancellation.check()?;
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => {
                bytes.extend_from_slice(&buffer[..read]);
                if bytes.len() > 16 * 1024 {
                    let _ =
                        write_callback_page(stream, 431, "Callback request headers are too large.");
                    return Ok(None);
                }
                if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                if started.elapsed() >= Duration::from_secs(10) {
                    return Ok(None);
                }
            }
            Err(error) => {
                return Err(OAuthError::Callback(format!(
                    "cannot read callback request: {error}"
                )));
            }
        }
    }
    let request = String::from_utf8(bytes)
        .map_err(|_| OAuthError::Callback("callback request was not UTF-8".to_owned()))?;
    Ok(request.lines().next().map(str::to_owned))
}

fn write_callback_page(stream: &mut TcpStream, status: u16, message: &str) -> io::Result<()> {
    let escaped = escape_html(message);
    let body = format!(
        "<!doctype html><meta charset=utf-8><title>GoshCoder</title>\
         <body style=\"font:16px system-ui;padding:3rem\"><p>{escaped}</p>"
    );
    write!(
        stream,
        "HTTP/1.1 {status} {}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Cache-Control: no-store\r\nContent-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        if status < 400 { "OK" } else { "Bad Request" },
        body.len()
    )?;
    stream.flush()
}

fn escape_html(value: &str) -> String {
    value
        .chars()
        .flat_map(|character| match character {
            '&' => "&amp;".chars().collect::<Vec<_>>(),
            '<' => "&lt;".chars().collect::<Vec<_>>(),
            '>' => "&gt;".chars().collect::<Vec<_>>(),
            '"' => "&quot;".chars().collect::<Vec<_>>(),
            '\'' => "&#39;".chars().collect::<Vec<_>>(),
            character => vec![character],
        })
        .collect()
}

fn query_value(url: &Url, name: &str) -> String {
    url.query_pairs()
        .find(|(candidate, _)| candidate == name)
        .map(|(_, value)| value.into_owned())
        .unwrap_or_default()
}

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Inputs for a PKCE browser flow's loopback/manual-code race.
///
/// This intentionally does not implement `Debug`: the authorization URL
/// contains a state value and the request keeps the matching value.
pub struct LoopbackLoginRequest {
    pub authorization_url: Url,
    pub redirect_uri: String,
    pub expected_state: String,
    pub callback_host: String,
    pub callback_port: u16,
    pub callback_path: String,
}

/// Runs a browser authorization flow, racing a loopback callback against a
/// cancellable manual-paste prompt.
pub fn run_loopback_login(
    interaction: Arc<dyn OAuthInteraction>,
    browser: Arc<dyn BrowserOpener>,
    cancellation: &CancellationToken,
    request: LoopbackLoginRequest,
) -> Result<String> {
    let server = match LoopbackCallbackServer::bind(
        &request.callback_host,
        request.callback_port,
        &request.callback_path,
        &request.expected_state,
    ) {
        Ok(server) => Some(server),
        Err(error) => {
            interaction.notify(OAuthEvent::info(format!(
                "Could not listen on {}:{} for the browser callback ({error}). Complete login in the browser and paste the redirect URL below.",
                request.callback_host, request.callback_port
            )));
            None
        }
    };
    interaction.notify(OAuthEvent::authorization_url(&request.authorization_url));
    let _ = browser.open(&request.authorization_url);

    let manual_cancellation = CancellationToken::new();
    let _cancel_manual = CancelOnDrop(manual_cancellation.clone());
    let prompt = OAuthPrompt {
        kind: OAuthPromptKind::ManualCode,
        message:
            "Complete login in your browser, or paste the authorization code / redirect URL here:"
                .to_owned(),
        placeholder: request.redirect_uri,
        options: Vec::new(),
        cancellation: manual_cancellation,
    };
    let (manual_sender, manual_receiver) = mpsc::sync_channel(1);
    thread::Builder::new()
        .name("oauth-manual-code".to_owned())
        .spawn(move || {
            let result = interaction.prompt(prompt);
            let _ = manual_sender.send(result);
        })
        .map_err(|error| {
            OAuthError::Callback(format!("cannot start manual-code prompt: {error}"))
        })?;

    loop {
        cancellation.check()?;
        if let Some(server) = &server
            && let Some(callback) = server.try_accept(cancellation)?
        {
            return Ok(callback.code);
        }
        match manual_receiver.try_recv() {
            Ok(result) => {
                let parsed = parse_authorization_input(&result?)
                    .ok_or(OAuthError::InvalidAuthorizationInput)?;
                if !parsed.state.is_empty()
                    && !constant_time_eq(&parsed.state, &request.expected_state)
                {
                    return Err(OAuthError::StateMismatch);
                }
                return Ok(parsed.code);
            }
            Err(TryRecvError::Empty) => thread::sleep(Duration::from_millis(10)),
            Err(TryRecvError::Disconnected) => return Err(OAuthError::Cancelled),
        }
    }
}

/// A request the OAuth transport receives. It deliberately has no `Debug`
/// implementation because form and JSON bodies may contain refresh tokens.
#[derive(Clone)]
pub struct OAuthRequest {
    method: Method,
    url: Url,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
    timeout: Duration,
}

impl OAuthRequest {
    pub fn method(&self) -> &Method {
        &self.method
    }

    pub fn url(&self) -> &Url {
        &self.url
    }

    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    pub fn body(&self) -> &[u8] {
        &self.body
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// A bounded OAuth HTTP response.
pub struct OAuthResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Injectable token/discovery transport.
///
/// A runtime that has an async executor may implement this trait with true
/// in-flight cancellation. The included blocking reqwest transport checks
/// cancellation before and after a request and bounds every request by its
/// configured timeout.
pub trait OAuthTransport: Send + Sync {
    fn execute(
        &self,
        request: OAuthRequest,
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse>;
}

/// Production transport using the existing blocking `reqwest` dependency.
#[derive(Clone)]
pub struct ReqwestOAuthTransport {
    client: Client,
}

impl ReqwestOAuthTransport {
    pub fn new() -> Result<Self> {
        Client::builder()
            .build()
            .map(|client| Self { client })
            .map_err(|error| {
                OAuthError::Transport(format!("cannot build OAuth HTTP client: {error}"))
            })
    }

    pub fn with_client(client: Client) -> Self {
        Self { client }
    }
}

impl OAuthTransport for ReqwestOAuthTransport {
    fn execute(
        &self,
        request: OAuthRequest,
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse> {
        cancellation.check()?;
        let mut headers = HeaderMap::new();
        for (name, value) in &request.headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
                OAuthError::InvalidConfiguration(format!("invalid OAuth request header {name:?}"))
            })?;
            let value = HeaderValue::from_str(value).map_err(|_| {
                OAuthError::InvalidConfiguration(format!(
                    "invalid OAuth request value for header {:?}",
                    name.as_str()
                ))
            })?;
            headers.insert(name, value);
        }
        let mut response = self
            .client
            .request(request.method, request.url)
            .headers(headers)
            .body(request.body)
            .timeout(request.timeout)
            .send()
            .map_err(|error| OAuthError::Transport(error.to_string()))?;
        let status = response.status().as_u16();
        let mut body = Vec::new();
        response
            .by_ref()
            .take((MAX_TOKEN_RESPONSE_BYTES + 1) as u64)
            .read_to_end(&mut body)
            .map_err(|error| {
                OAuthError::Transport(format!("cannot read OAuth response: {error}"))
            })?;
        body.truncate(MAX_TOKEN_RESPONSE_BYTES);
        cancellation.check()?;
        Ok(OAuthResponse { status, body })
    }
}

#[derive(Clone, Debug)]
pub struct DevicePollingPolicy {
    pub interval: Duration,
    pub timeout: Duration,
    pub wait_before_first_poll: bool,
}

impl DevicePollingPolicy {
    pub fn new(interval: Duration, timeout: Duration) -> Self {
        Self {
            interval: normalize_device_interval(interval),
            timeout,
            wait_before_first_poll: false,
        }
    }

    pub fn wait_before_first_poll(mut self) -> Self {
        self.wait_before_first_poll = true;
        self
    }
}

fn normalize_device_interval(interval: Duration) -> Duration {
    if interval < DEVICE_MIN_INTERVAL {
        DEFAULT_DEVICE_INTERVAL
    } else {
        interval
    }
}

/// Result of one device-code token poll.
pub enum DevicePoll<T> {
    Pending,
    /// `None` means use RFC 8628's extra five-second slowdown.
    SlowDown(Option<Duration>),
    Complete(T),
}

/// Polls an OAuth device endpoint using an injectable clock and cooperative
/// cancellation.
pub fn poll_device_code<T>(
    clock: &dyn OAuthClock,
    cancellation: &CancellationToken,
    policy: DevicePollingPolicy,
    mut poll: impl FnMut() -> Result<DevicePoll<T>>,
) -> Result<T> {
    let deadline = clock
        .now_ms()
        .saturating_add(duration_millis(policy.timeout));
    let mut interval = normalize_device_interval(policy.interval);
    if policy.wait_before_first_poll {
        sleep_before_deadline(clock, cancellation, deadline, interval)?;
    }

    loop {
        cancellation.check()?;
        if clock.now_ms() >= deadline {
            return Err(OAuthError::DeviceTimedOut);
        }
        match poll()? {
            DevicePoll::Complete(value) => return Ok(value),
            DevicePoll::Pending => {
                sleep_before_deadline(clock, cancellation, deadline, interval)?;
            }
            DevicePoll::SlowDown(Some(suggested)) => {
                interval = normalize_device_interval(suggested);
                sleep_before_deadline(clock, cancellation, deadline, interval)?;
            }
            DevicePoll::SlowDown(None) => {
                interval = interval.saturating_add(Duration::from_secs(5));
                sleep_before_deadline(clock, cancellation, deadline, interval)?;
            }
        }
    }
}

fn sleep_before_deadline(
    clock: &dyn OAuthClock,
    cancellation: &CancellationToken,
    deadline_ms: i64,
    requested: Duration,
) -> Result<()> {
    let remaining = deadline_ms.saturating_sub(clock.now_ms());
    if remaining <= 0 {
        return Err(OAuthError::DeviceTimedOut);
    }
    let wait = requested.min(Duration::from_millis(remaining as u64));
    clock.sleep(wait, cancellation)
}

#[derive(Default, Deserialize)]
struct TokenResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    refresh_token: String,
    #[serde(default)]
    expires_in: i64,
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
}

fn parse_token_response(body: &[u8]) -> Option<TokenResponse> {
    serde_json::from_slice(body).ok()
}

fn token_error(
    provider: &'static str,
    operation: &'static str,
    response: &OAuthResponse,
) -> OAuthError {
    let parsed = parse_token_response(&response.body);
    let detail = parsed
        .as_ref()
        .map(|response| response.error_description.trim().to_owned())
        .filter(|detail| !detail.is_empty())
        .unwrap_or_else(|| truncate_response(&response.body, 500));
    if response.status == 401
        || response.status == 403
        || parsed
            .as_ref()
            .is_some_and(|response| response.error == "invalid_grant")
    {
        OAuthError::Unauthorized {
            provider,
            operation,
            detail: (!detail.is_empty()).then_some(detail),
        }
    } else {
        OAuthError::TokenFailure {
            provider,
            operation,
            status: response.status,
            detail,
        }
    }
}

fn truncate_response(body: &[u8], limit: usize) -> String {
    let body = String::from_utf8_lossy(body);
    let mut text = body.trim().chars();
    let truncated: String = text.by_ref().take(limit).collect();
    if text.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn is_retryable_status(status: u16) -> bool {
    status == 429 || status >= 500
}

fn credential_from_token(
    provider: &'static str,
    operation: &'static str,
    token: TokenResponse,
    now_ms: i64,
    skew: Duration,
) -> Result<Credential> {
    if token.access_token.is_empty() || token.refresh_token.is_empty() || token.expires_in <= 0 {
        return Err(OAuthError::InvalidTokenResponse {
            provider,
            operation,
        });
    }
    let expires_at_ms = now_ms
        .saturating_add(token.expires_in.saturating_mul(1_000))
        .saturating_sub(duration_millis(skew));
    Ok(Credential::oauth(
        token.access_token,
        token.refresh_token,
        expires_at_ms,
    ))
}

fn trusted_http_url(value: &str) -> Option<Url> {
    Url::parse(value)
        .ok()
        .filter(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
}

fn duration_from_seconds(seconds: f64) -> Option<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }
    let milliseconds = (seconds * 1_000.0).ceil();
    if milliseconds > u64::MAX as f64 {
        return None;
    }
    Some(Duration::from_millis(milliseconds as u64))
}

fn append_query(url: &Url, fields: &[(&str, &str)]) -> Url {
    let mut url = url.clone();
    {
        let mut query = url.query_pairs_mut();
        query.clear();
        for (name, value) in fields {
            query.append_pair(name, value);
        }
    }
    url
}

fn endpoint(base: &Url, path: &str) -> Result<Url> {
    if !path.starts_with('/') {
        return Err(OAuthError::InvalidConfiguration(format!(
            "OAuth endpoint path {path:?} must start with '/'"
        )));
    }
    let mut url = base.clone();
    // The Go flows append paths after trimming only the base's trailing slash.
    // Retaining a configured path keeps local proxy/test overrides compatible.
    let prefix = base.path().trim_end_matches('/');
    let path = if prefix.is_empty() || prefix == "/" {
        path.to_owned()
    } else {
        format!("{prefix}{path}")
    };
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// URLs and public client IDs used by the five concrete Go OAuth flows.
///
/// Every field is public so integration tests can point a flow at a local
/// fake server without changing global process state.
#[derive(Clone)]
pub struct OAuthEndpoints {
    pub anthropic_authorize_url: Url,
    pub anthropic_token_url: Url,
    pub kimi_default_oauth_host: Url,
    pub codex_authorize_url: Url,
    pub codex_token_url: Url,
    pub codex_device_user_code_url: Url,
    pub codex_device_token_url: Url,
    pub codex_device_verify_url: Url,
    pub xai_issuer_url: Url,
    pub meta_auth_base_url: Url,
    pub meta_api_base_url: Url,
}

impl Default for OAuthEndpoints {
    fn default() -> Self {
        Self {
            anthropic_authorize_url: fixed_url("https://claude.ai/oauth/authorize"),
            anthropic_token_url: fixed_url("https://platform.claude.com/v1/oauth/token"),
            kimi_default_oauth_host: fixed_url("https://auth.kimi.com"),
            codex_authorize_url: fixed_url("https://auth.openai.com/oauth/authorize"),
            codex_token_url: fixed_url("https://auth.openai.com/oauth/token"),
            codex_device_user_code_url: fixed_url(
                "https://auth.openai.com/api/accounts/deviceauth/usercode",
            ),
            codex_device_token_url: fixed_url(
                "https://auth.openai.com/api/accounts/deviceauth/token",
            ),
            codex_device_verify_url: fixed_url("https://auth.openai.com/codex/device"),
            xai_issuer_url: fixed_url("https://auth.x.ai"),
            meta_auth_base_url: fixed_url("https://auth.meta.com"),
            meta_api_base_url: fixed_url("https://api.meta.ai"),
        }
    }
}

fn fixed_url(value: &str) -> Url {
    Url::parse(value).expect("compiled OAuth endpoint must be a valid URL")
}

/// xAI endpoints obtained from its OIDC discovery document.
#[derive(Clone, Eq, PartialEq)]
pub struct XaiEndpoints {
    pub authorize: Url,
    pub token: Url,
    pub device: Url,
}

#[derive(Clone)]
struct CachedXaiEndpoints {
    issuer: String,
    endpoints: XaiEndpoints,
}

/// Reusable blocking OAuth flow client.
///
/// It has no global mutable endpoint state: tests can create a separate
/// instance with a fake transport and endpoint configuration.
pub struct OAuthClient {
    transport: Arc<dyn OAuthTransport>,
    clock: Arc<dyn OAuthClock>,
    browser: Arc<dyn BrowserOpener>,
    endpoints: OAuthEndpoints,
    token_request_timeout: Duration,
    xai_discovery_timeout: Duration,
    xai_discovery: Mutex<Option<CachedXaiEndpoints>>,
}

impl OAuthClient {
    pub fn new(
        transport: Arc<dyn OAuthTransport>,
        clock: Arc<dyn OAuthClock>,
        browser: Arc<dyn BrowserOpener>,
        endpoints: OAuthEndpoints,
    ) -> Self {
        Self {
            transport,
            clock,
            browser,
            endpoints,
            token_request_timeout: DEFAULT_TOKEN_REQUEST_TIMEOUT,
            xai_discovery_timeout: XAI_DISCOVERY_TIMEOUT,
            xai_discovery: Mutex::new(None),
        }
    }

    /// Builds the production client using only existing dependencies.
    pub fn system() -> Result<Self> {
        Ok(Self::new(
            Arc::new(ReqwestOAuthTransport::new()?),
            Arc::new(SystemClock),
            Arc::new(SystemBrowser),
            OAuthEndpoints::default(),
        ))
    }

    /// Overrides token and discovery timeouts for a bounded embedding or test.
    pub fn with_timeouts(
        mut self,
        token_request_timeout: Duration,
        xai_discovery_timeout: Duration,
    ) -> Self {
        self.token_request_timeout = token_request_timeout;
        self.xai_discovery_timeout = xai_discovery_timeout;
        self
    }

    pub fn endpoints(&self) -> &OAuthEndpoints {
        &self.endpoints
    }

    /// Starts a provider login but leaves persistence under caller control.
    pub fn login(
        &self,
        provider: OAuthProviderId,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        cancellation.check()?;
        match provider {
            OAuthProviderId::Anthropic => {
                self.login_anthropic(interaction, environment, cancellation)
            }
            OAuthProviderId::KimiCoding => self.login_kimi(interaction, environment, cancellation),
            OAuthProviderId::Meta => self.login_meta(interaction, environment, cancellation),
            OAuthProviderId::OpenAiCodex => {
                self.login_codex(interaction, environment, cancellation)
            }
            OAuthProviderId::Xai => self.login_xai(interaction, environment, cancellation),
            OAuthProviderId::OpenRouter
            | OAuthProviderId::GithubCopilot
            | OAuthProviderId::Radius => Err(OAuthError::UnsupportedFlow {
                provider: provider.as_str().to_owned(),
            }),
        }
    }

    /// Runs login and persists the exact `auth.json` credential shape through
    /// the existing store.
    pub fn login_and_persist(
        &self,
        provider: OAuthProviderId,
        store: &CredentialStore,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let credential = self.login(provider, interaction, environment, cancellation)?;
        store
            .put(provider.as_str(), credential)
            .map_err(OAuthError::Storage)
    }

    /// Refreshes an already persisted OAuth credential without storing it.
    pub fn refresh(
        &self,
        provider: OAuthProviderId,
        current: &Credential,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        cancellation.check()?;
        match provider {
            OAuthProviderId::Anthropic => self.refresh_anthropic(current, cancellation),
            OAuthProviderId::KimiCoding => self.refresh_kimi(current, environment, cancellation),
            OAuthProviderId::Meta => self.refresh_meta(current, environment, cancellation),
            OAuthProviderId::OpenAiCodex => self.refresh_codex(current, cancellation),
            OAuthProviderId::Xai => self.refresh_xai(current, environment, cancellation),
            OAuthProviderId::OpenRouter
            | OAuthProviderId::GithubCopilot
            | OAuthProviderId::Radius => Err(OAuthError::UnsupportedFlow {
                provider: provider.as_str().to_owned(),
            }),
        }
    }

    /// Resolves a stored OAuth credential, refreshing it under
    /// `CredentialStore::modify`'s lock when necessary.
    ///
    /// This is the OAuth half of catalog resolution. `Ok(None)` means no
    /// stored OAuth credential is present; an error means one was present but
    /// could not safely resolve and callers must not fall back to ambient auth.
    pub fn resolve_stored_oauth(
        &self,
        provider: OAuthProviderId,
        store: &CredentialStore,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Option<OAuthAuth>> {
        let Some(initial) = store
            .read_raw(provider.as_str())
            .map_err(OAuthError::Storage)?
        else {
            return Ok(None);
        };
        if initial.kind() != &CredentialKind::OAuth {
            return Ok(None);
        }
        if !credential_expires_soon(&initial, self.clock.as_ref()) {
            return auth_from_credential(provider, &initial).map(Some);
        }

        let mut refresh_error = None;
        let post = store
            .modify(provider.as_str(), |current| {
                let Some(current) = current else {
                    return Ok(None);
                };
                if current.kind() != &CredentialKind::OAuth {
                    return Ok(None);
                }
                // Recheck under the credential-store lock. Another process
                // might have rotated this token while this process waited.
                if !credential_expires_soon(&current, self.clock.as_ref()) {
                    return Ok(None);
                }
                match self.refresh(provider, &current, environment, cancellation) {
                    Ok(refreshed) => Ok(Some(refreshed)),
                    Err(error) => {
                        refresh_error = Some(error);
                        // `None` preserves the recoverable credential exactly
                        // as the Go store's Modify callback does.
                        Ok(None)
                    }
                }
            })
            .map_err(OAuthError::Storage)?;
        if let Some(error) = refresh_error {
            return Err(error);
        }
        let Some(post) = post else {
            return Err(OAuthError::Unauthorized {
                provider: provider.display_name(),
                operation: "refresh",
                detail: None,
            });
        };
        if post.kind() != &CredentialKind::OAuth {
            return Err(OAuthError::Unauthorized {
                provider: provider.display_name(),
                operation: "refresh",
                detail: None,
            });
        }
        auth_from_credential(provider, &post).map(Some)
    }

    fn post_form(
        &self,
        url: &Url,
        fields: BTreeMap<String, String>,
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse> {
        let mut encoder = url::form_urlencoded::Serializer::new(String::new());
        for (name, value) in fields {
            encoder.append_pair(&name, &value);
        }
        self.transport.execute(
            OAuthRequest {
                method: Method::POST,
                url: url.clone(),
                headers: BTreeMap::from([
                    (
                        "Content-Type".to_owned(),
                        "application/x-www-form-urlencoded".to_owned(),
                    ),
                    ("Accept".to_owned(), "application/json".to_owned()),
                ]),
                body: encoder.finish().into_bytes(),
                timeout: self.token_request_timeout,
            },
            cancellation,
        )
    }

    fn post_json(
        &self,
        url: &Url,
        payload: Value,
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse> {
        let body = serde_json::to_vec(&payload).map_err(|_| {
            OAuthError::InvalidConfiguration("could not encode OAuth JSON request".to_owned())
        })?;
        self.transport.execute(
            OAuthRequest {
                method: Method::POST,
                url: url.clone(),
                headers: BTreeMap::from([
                    ("Content-Type".to_owned(), "application/json".to_owned()),
                    ("Accept".to_owned(), "application/json".to_owned()),
                ]),
                body,
                timeout: self.token_request_timeout,
            },
            cancellation,
        )
    }

    fn post_json_with_headers(
        &self,
        url: &Url,
        payload: Value,
        mut headers: BTreeMap<String, String>,
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse> {
        headers.insert("Content-Type".to_owned(), "application/json".to_owned());
        headers.insert("Accept".to_owned(), "application/json".to_owned());
        let body = serde_json::to_vec(&payload).map_err(|_| {
            OAuthError::InvalidConfiguration("could not encode OAuth JSON request".to_owned())
        })?;
        self.transport.execute(
            OAuthRequest {
                method: Method::POST,
                url: url.clone(),
                headers,
                body,
                timeout: self.token_request_timeout,
            },
            cancellation,
        )
    }

    fn get_json(
        &self,
        url: &Url,
        timeout: Duration,
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse> {
        self.transport.execute(
            OAuthRequest {
                method: Method::GET,
                url: url.clone(),
                headers: BTreeMap::from([("Accept".to_owned(), "application/json".to_owned())]),
                body: Vec::new(),
                timeout,
            },
            cancellation,
        )
    }

    fn successful_token(
        &self,
        provider: OAuthProviderId,
        operation: &'static str,
        response: OAuthResponse,
    ) -> Result<TokenResponse> {
        if !(200..300).contains(&response.status) {
            return Err(token_error(provider.display_name(), operation, &response));
        }
        serde_json::from_slice(&response.body).map_err(|_| OAuthError::InvalidTokenResponse {
            provider: provider.display_name(),
            operation,
        })
    }
}

fn fields(items: impl IntoIterator<Item = (&'static str, String)>) -> BTreeMap<String, String> {
    items
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect()
}

fn callback_host(environment: &dyn OAuthEnvironment) -> String {
    environment
        .value("GOSHCODER_OAUTH_CALLBACK_HOST")
        .or_else(|| environment.value("PI_OAUTH_CALLBACK_HOST"))
        .unwrap_or_else(|| "127.0.0.1".to_owned())
}

fn kimi_oauth_host(environment: &dyn OAuthEnvironment, endpoints: &OAuthEndpoints) -> Result<Url> {
    let configured = environment
        .value("KIMI_CODE_OAUTH_HOST")
        .or_else(|| environment.value("KIMI_OAUTH_HOST"));
    let host = match configured {
        Some(host) => Url::parse(host.trim_end_matches('/'))
            .map_err(|error| OAuthError::InvalidUrl(error.to_string()))?,
        None => endpoints.kimi_default_oauth_host.clone(),
    };
    if !matches!(host.scheme(), "http" | "https") || host.host_str().is_none() {
        return Err(OAuthError::InvalidUrl(
            "Kimi OAuth host must be an absolute HTTP(S) URL".to_owned(),
        ));
    }
    Ok(host)
}

fn xai_client_id(environment: &dyn OAuthEnvironment) -> String {
    environment
        .value("GOSHCODER_XAI_OAUTH_CLIENT_ID")
        .unwrap_or_else(|| XAI_DEFAULT_CLIENT_ID.to_owned())
}

fn meta_client_id(environment: &dyn OAuthEnvironment) -> String {
    environment
        .value("GOSHCODER_META_OAUTH_CLIENT_ID")
        .unwrap_or_else(|| META_DEFAULT_CLIENT_ID.to_owned())
}

impl OAuthClient {
    /// Builds Anthropic's registered PKCE authorization URL.
    pub fn anthropic_authorization_url(&self, pkce: &PkcePair) -> Url {
        append_query(
            &self.endpoints.anthropic_authorize_url,
            &[
                ("code", "true"),
                ("client_id", ANTHROPIC_CLIENT_ID),
                ("response_type", "code"),
                ("redirect_uri", ANTHROPIC_REDIRECT_URI),
                ("scope", ANTHROPIC_SCOPES),
                ("code_challenge", pkce.challenge()),
                ("code_challenge_method", "S256"),
                // Anthropic's flow uses the verifier as state and returns it
                // to its token endpoint as well.
                ("state", pkce.verifier()),
            ],
        )
    }

    fn login_anthropic(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let pkce = generate_pkce();
        let authorization_url = self.anthropic_authorization_url(&pkce);
        let code = run_loopback_login(
            interaction.clone(),
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                authorization_url,
                redirect_uri: ANTHROPIC_REDIRECT_URI.to_owned(),
                expected_state: pkce.verifier().to_owned(),
                callback_host: callback_host(environment),
                callback_port: ANTHROPIC_CALLBACK_PORT,
                callback_path: ANTHROPIC_CALLBACK_PATH.to_owned(),
            },
        )?;
        interaction.notify(OAuthEvent::progress(
            "Exchanging the authorization code for tokens...",
        ));
        let response = self.post_json(
            &self.endpoints.anthropic_token_url,
            json!({
                "grant_type": "authorization_code",
                "client_id": ANTHROPIC_CLIENT_ID,
                "code": code,
                "state": pkce.verifier(),
                "redirect_uri": ANTHROPIC_REDIRECT_URI,
                "code_verifier": pkce.verifier(),
            }),
            cancellation,
        )?;
        let token = self.successful_token(OAuthProviderId::Anthropic, "exchange", response)?;
        credential_from_token(
            OAuthProviderId::Anthropic.display_name(),
            "exchange",
            token,
            self.clock.now_ms(),
            ANTHROPIC_REFRESH_SKEW,
        )
    }

    fn refresh_anthropic(
        &self,
        current: &Credential,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let response = self.post_json(
            &self.endpoints.anthropic_token_url,
            json!({
                "grant_type": "refresh_token",
                "client_id": ANTHROPIC_CLIENT_ID,
                "refresh_token": current.refresh(),
            }),
            cancellation,
        )?;
        let token = self.successful_token(OAuthProviderId::Anthropic, "refresh", response)?;
        credential_from_token(
            OAuthProviderId::Anthropic.display_name(),
            "refresh",
            token,
            self.clock.now_ms(),
            ANTHROPIC_REFRESH_SKEW,
        )
    }

    fn login_kimi(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let host = kimi_oauth_host(environment, &self.endpoints)?;
        let response = self.post_form(
            &endpoint(&host, "/api/oauth/device_authorization")?,
            fields([("client_id", KIMI_CLIENT_ID.to_owned())]),
            cancellation,
        )?;
        if !(200..300).contains(&response.status) {
            return Err(operation_failure(
                OAuthProviderId::KimiCoding.display_name(),
                "device authorization",
                &response,
            ));
        }
        let device: DeviceAuthorization = serde_json::from_slice(&response.body).map_err(|_| {
            OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::KimiCoding.display_name(),
                operation: "device authorization",
            }
        })?;
        let verification = trusted_http_url(&device.verification_uri);
        let complete = trusted_http_url(&device.verification_uri_complete);
        if device.device_code.is_empty()
            || device.user_code.is_empty()
            || verification.is_none()
            || complete.is_none()
        {
            return Err(OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::KimiCoding.display_name(),
                operation: "device authorization",
            });
        }
        let interval = reported_device_interval(device.interval, DEFAULT_DEVICE_INTERVAL);
        let timeout = reported_device_timeout(device.expires_in, DEFAULT_DEVICE_TIMEOUT);
        let verification = complete.expect("checked above");
        interaction.notify(OAuthEvent::device_code(
            device.user_code.clone(),
            verification.to_string(),
            interval,
            timeout,
        ));
        self.poll_kimi_device_token(&host, &device.device_code, interval, timeout, cancellation)
    }

    fn poll_kimi_device_token(
        &self,
        host: &Url,
        device_code: &str,
        interval: Duration,
        timeout: Duration,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let token_url = endpoint(host, "/api/oauth/token")?;
        let device_code = device_code.to_owned();
        poll_device_code(
            self.clock.as_ref(),
            cancellation,
            DevicePollingPolicy::new(interval, timeout).wait_before_first_poll(),
            || {
                let response = self.post_form(
                    &token_url,
                    fields([
                        ("client_id", KIMI_CLIENT_ID.to_owned()),
                        ("device_code", device_code.clone()),
                        ("grant_type", KIMI_DEVICE_GRANT.to_owned()),
                    ]),
                    cancellation,
                )?;
                if response.status >= 500 {
                    return Err(operation_failure(
                        OAuthProviderId::KimiCoding.display_name(),
                        "device token request",
                        &response,
                    ));
                }
                if (200..300).contains(&response.status) {
                    let token = self.successful_token(
                        OAuthProviderId::KimiCoding,
                        "device poll",
                        response,
                    )?;
                    return credential_from_token(
                        OAuthProviderId::KimiCoding.display_name(),
                        "device poll",
                        token,
                        self.clock.now_ms(),
                        Duration::ZERO,
                    )
                    .map(DevicePoll::Complete);
                }
                let failure = parse_device_failure(&response);
                match failure.error.as_str() {
                    "authorization_pending" => Ok(DevicePoll::Pending),
                    "slow_down" => Ok(DevicePoll::SlowDown(
                        failure
                            .interval
                            .and_then(duration_from_seconds)
                            .map(|interval| interval.max(DEVICE_MIN_INTERVAL)),
                    )),
                    "expired_token" => Err(OAuthError::DeviceExpired {
                        provider: OAuthProviderId::KimiCoding.display_name(),
                    }),
                    "access_denied" => Err(OAuthError::DeviceDenied {
                        provider: OAuthProviderId::KimiCoding.display_name(),
                    }),
                    _ => Err(operation_failure(
                        OAuthProviderId::KimiCoding.display_name(),
                        "device token request",
                        &response,
                    )),
                }
            },
        )
    }

    fn refresh_kimi(
        &self,
        current: &Credential,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let host = kimi_oauth_host(environment, &self.endpoints)?;
        let token_url = endpoint(&host, "/api/oauth/token")?;
        let mut last_error = None;
        for attempt in 0..=KIMI_REFRESH_MAX_RETRIES {
            cancellation.check()?;
            if attempt > 0 {
                self.clock
                    .sleep(Duration::from_secs(1_u64 << (attempt - 1)), cancellation)?;
            }
            match self.post_form(
                &token_url,
                fields([
                    ("client_id", KIMI_CLIENT_ID.to_owned()),
                    ("grant_type", "refresh_token".to_owned()),
                    ("refresh_token", current.refresh().to_owned()),
                ]),
                cancellation,
            ) {
                Ok(response) if (200..300).contains(&response.status) => {
                    let token =
                        self.successful_token(OAuthProviderId::KimiCoding, "refresh", response)?;
                    return credential_from_token(
                        OAuthProviderId::KimiCoding.display_name(),
                        "refresh",
                        token,
                        self.clock.now_ms(),
                        Duration::ZERO,
                    );
                }
                Ok(response) => {
                    let error = token_error(
                        OAuthProviderId::KimiCoding.display_name(),
                        "refresh",
                        &response,
                    );
                    if error.is_unauthorized() {
                        return Err(error);
                    }
                    if is_retryable_status(response.status) && attempt < KIMI_REFRESH_MAX_RETRIES {
                        last_error = Some(error);
                        continue;
                    }
                    return Err(error);
                }
                Err(error) if matches!(error, OAuthError::Cancelled) => return Err(error),
                Err(error) if attempt < KIMI_REFRESH_MAX_RETRIES => {
                    last_error = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        Err(last_error.unwrap_or_else(|| {
            OAuthError::Transport("Kimi token refresh ended without a response".to_owned())
        }))
    }
}

#[derive(Default, Deserialize)]
struct DeviceAuthorization {
    #[serde(default)]
    device_code: String,
    #[serde(default)]
    user_code: String,
    #[serde(default)]
    verification_uri: String,
    #[serde(default)]
    verification_uri_complete: String,
    #[serde(default)]
    interval: f64,
    #[serde(default)]
    expires_in: f64,
}

#[derive(Default, Deserialize)]
struct DeviceFailure {
    #[serde(default)]
    error: String,
    #[serde(default)]
    interval: Option<f64>,
}

fn parse_device_failure(response: &OAuthResponse) -> DeviceFailure {
    serde_json::from_slice(&response.body).unwrap_or_default()
}

fn reported_device_interval(reported_seconds: f64, fallback: Duration) -> Duration {
    duration_from_seconds(reported_seconds)
        .map(|duration| duration.max(DEVICE_MIN_INTERVAL))
        .unwrap_or(fallback)
}

fn reported_device_timeout(reported_seconds: f64, fallback: Duration) -> Duration {
    duration_from_seconds(reported_seconds).unwrap_or(fallback)
}

fn operation_failure(
    provider: &'static str,
    operation: &'static str,
    response: &OAuthResponse,
) -> OAuthError {
    OAuthError::TokenFailure {
        provider,
        operation,
        status: response.status,
        detail: truncate_response(&response.body, 300),
    }
}

impl OAuthClient {
    /// Builds OpenAI Codex's registered browser PKCE authorization URL.
    pub fn codex_authorization_url(&self, pkce: &PkcePair, state: &str) -> Url {
        append_query(
            &self.endpoints.codex_authorize_url,
            &[
                ("response_type", "code"),
                ("client_id", CODEX_CLIENT_ID),
                ("redirect_uri", CODEX_REDIRECT_URI),
                ("scope", CODEX_SCOPE),
                ("code_challenge", pkce.challenge()),
                ("code_challenge_method", "S256"),
                ("state", state),
                ("id_token_add_organizations", "true"),
                ("codex_cli_simplified_flow", "true"),
                ("originator", "goshcoder"),
            ],
        )
    }

    fn login_codex(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let method = interaction.prompt(OAuthPrompt {
            kind: OAuthPromptKind::Select,
            message: "Select the OpenAI Codex login method:".to_owned(),
            placeholder: String::new(),
            options: vec![
                OAuthPromptOption {
                    id: "browser".to_owned(),
                    label: "Browser login (default)".to_owned(),
                    description: String::new(),
                },
                OAuthPromptOption {
                    id: "device_code".to_owned(),
                    label: "Device code login (headless)".to_owned(),
                    description: String::new(),
                },
            ],
            cancellation: cancellation.clone(),
        })?;
        cancellation.check()?;
        match method.as_str() {
            "" | "browser" => self.login_codex_browser(interaction, environment, cancellation),
            "device_code" => self.login_codex_device(interaction, cancellation),
            _ => Err(OAuthError::InvalidConfiguration(format!(
                "unknown OpenAI Codex login method {method:?}"
            ))),
        }
    }

    fn login_codex_browser(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let pkce = generate_pkce();
        let state = random_state();
        let code = run_loopback_login(
            interaction.clone(),
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                authorization_url: self.codex_authorization_url(&pkce, &state),
                redirect_uri: CODEX_REDIRECT_URI.to_owned(),
                expected_state: state,
                callback_host: callback_host(environment),
                callback_port: CODEX_CALLBACK_PORT,
                callback_path: CODEX_CALLBACK_PATH.to_owned(),
            },
        )?;
        interaction.notify(OAuthEvent::progress(
            "Exchanging the authorization code for tokens...",
        ));
        self.exchange_codex_code(&code, pkce.verifier(), CODEX_REDIRECT_URI, cancellation)
    }

    fn login_codex_device(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let response = self.post_json(
            &self.endpoints.codex_device_user_code_url,
            json!({"client_id": CODEX_CLIENT_ID}),
            cancellation,
        )?;
        if response.status == 404 {
            return Err(OAuthError::InvalidConfiguration(
                "OpenAI Codex device-code login is not enabled; use browser login".to_owned(),
            ));
        }
        if !(200..300).contains(&response.status) {
            return Err(operation_failure(
                OAuthProviderId::OpenAiCodex.display_name(),
                "device code request",
                &response,
            ));
        }
        let device = parse_codex_device_authorization(&response.body)?;
        interaction.notify(OAuthEvent::device_code(
            device.user_code.clone(),
            self.endpoints.codex_device_verify_url.to_string(),
            Duration::from_secs(device.interval_seconds),
            DEFAULT_DEVICE_TIMEOUT,
        ));
        let (code, verifier) = self.poll_codex_device_token(&device, cancellation)?;
        interaction.notify(OAuthEvent::progress(
            "Exchanging the authorization code for tokens...",
        ));
        self.exchange_codex_code(&code, &verifier, CODEX_DEVICE_REDIRECT_URI, cancellation)
    }

    fn poll_codex_device_token(
        &self,
        device: &CodexDeviceAuthorization,
        cancellation: &CancellationToken,
    ) -> Result<(String, String)> {
        let device_auth_id = device.device_auth_id.clone();
        let user_code = device.user_code.clone();
        poll_device_code(
            self.clock.as_ref(),
            cancellation,
            DevicePollingPolicy::new(
                Duration::from_secs(device.interval_seconds),
                DEFAULT_DEVICE_TIMEOUT,
            ),
            || {
                let response = self.post_json(
                    &self.endpoints.codex_device_token_url,
                    json!({
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    }),
                    cancellation,
                )?;
                if (200..300).contains(&response.status) {
                    let value: Value = serde_json::from_slice(&response.body).map_err(|_| {
                        OAuthError::InvalidTokenResponse {
                            provider: OAuthProviderId::OpenAiCodex.display_name(),
                            operation: "device poll",
                        }
                    })?;
                    let code = json_string(&value, "authorization_code").ok_or(
                        OAuthError::InvalidTokenResponse {
                            provider: OAuthProviderId::OpenAiCodex.display_name(),
                            operation: "device poll",
                        },
                    )?;
                    let verifier = json_string(&value, "code_verifier").ok_or(
                        OAuthError::InvalidTokenResponse {
                            provider: OAuthProviderId::OpenAiCodex.display_name(),
                            operation: "device poll",
                        },
                    )?;
                    return Ok(DevicePoll::Complete((code, verifier)));
                }
                if response.status == 403
                    || response.status == 404
                    || codex_device_pending(&response.body)
                {
                    return Ok(DevicePoll::Pending);
                }
                Err(operation_failure(
                    OAuthProviderId::OpenAiCodex.display_name(),
                    "device token request",
                    &response,
                ))
            },
        )
    }

    fn exchange_codex_code(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let response = self.post_form(
            &self.endpoints.codex_token_url,
            fields([
                ("grant_type", "authorization_code".to_owned()),
                ("client_id", CODEX_CLIENT_ID.to_owned()),
                ("code", code.to_owned()),
                ("redirect_uri", redirect_uri.to_owned()),
                ("code_verifier", verifier.to_owned()),
            ]),
            cancellation,
        )?;
        let token = self.successful_token(OAuthProviderId::OpenAiCodex, "exchange", response)?;
        self.codex_credential_from_token("exchange", token)
    }

    fn refresh_codex(
        &self,
        current: &Credential,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let response = self.post_form(
            &self.endpoints.codex_token_url,
            fields([
                ("grant_type", "refresh_token".to_owned()),
                ("refresh_token", current.refresh().to_owned()),
                ("client_id", CODEX_CLIENT_ID.to_owned()),
            ]),
            cancellation,
        )?;
        let token = self.successful_token(OAuthProviderId::OpenAiCodex, "refresh", response)?;
        self.codex_credential_from_token("refresh", token)
    }

    fn codex_credential_from_token(
        &self,
        operation: &'static str,
        token: TokenResponse,
    ) -> Result<Credential> {
        let mut credential = credential_from_token(
            OAuthProviderId::OpenAiCodex.display_name(),
            operation,
            token,
            self.clock.now_ms(),
            Duration::ZERO,
        )?;
        let account_id = codex_account_id(credential.access())?;
        credential
            .set_extra("accountId", Value::String(account_id))
            .map_err(OAuthError::Storage)?;
        Ok(credential)
    }
}

struct CodexDeviceAuthorization {
    device_auth_id: String,
    user_code: String,
    interval_seconds: u64,
}

fn parse_codex_device_authorization(body: &[u8]) -> Result<CodexDeviceAuthorization> {
    let value: Value =
        serde_json::from_slice(body).map_err(|_| OAuthError::InvalidTokenResponse {
            provider: OAuthProviderId::OpenAiCodex.display_name(),
            operation: "device code request",
        })?;
    let device_auth_id =
        json_string(&value, "device_auth_id").ok_or(OAuthError::InvalidTokenResponse {
            provider: OAuthProviderId::OpenAiCodex.display_name(),
            operation: "device code request",
        })?;
    let user_code = json_string(&value, "user_code").ok_or(OAuthError::InvalidTokenResponse {
        provider: OAuthProviderId::OpenAiCodex.display_name(),
        operation: "device code request",
    })?;
    let interval_seconds = value.get("interval").and_then(json_nonnegative_u64).ok_or(
        OAuthError::InvalidTokenResponse {
            provider: OAuthProviderId::OpenAiCodex.display_name(),
            operation: "device code request",
        },
    )?;
    Ok(CodexDeviceAuthorization {
        device_auth_id,
        user_code,
        interval_seconds,
    })
}

fn json_string(value: &Value, name: &str) -> Option<String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn json_nonnegative_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_str()?.parse::<u64>().ok())
}

fn codex_device_pending(body: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return false;
    };
    match value.get("error") {
        Some(Value::String(error)) => {
            matches!(
                error.as_str(),
                "deviceauth_authorization_pending" | "authorization_pending"
            )
        }
        Some(Value::Object(error)) => matches!(
            error.get("code").and_then(Value::as_str),
            Some("deviceauth_authorization_pending" | "authorization_pending")
        ),
        _ => false,
    }
}

/// Extracts OpenAI Codex's `chatgpt_account_id` from an unsigned JWT payload.
///
/// The token is acquired over TLS and is not trusted for authorization here;
/// this only reads the account ID needed by Codex's request protocol.
pub fn codex_account_id(access_token: &str) -> Result<String> {
    let mut parts = access_token.split('.');
    let _header = parts.next();
    let payload = parts
        .next()
        .ok_or_else(|| OAuthError::Jwt("not a JWT".to_owned()))?;
    let _signature = parts
        .next()
        .ok_or_else(|| OAuthError::Jwt("not a JWT".to_owned()))?;
    if parts.next().is_some() {
        return Err(OAuthError::Jwt("not a JWT".to_owned()));
    }
    let decoded = URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .map_err(|_| OAuthError::Jwt("invalid JWT payload encoding".to_owned()))?;
    let value: Value = serde_json::from_slice(&decoded)
        .map_err(|_| OAuthError::Jwt("invalid JWT payload JSON".to_owned()))?;
    value
        .get(CODEX_AUTH_CLAIM)
        .and_then(Value::as_object)
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .filter(|account_id| !account_id.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| OAuthError::Jwt("no chatgpt_account_id claim".to_owned()))
}

impl OAuthClient {
    fn xai_fallback_endpoints(&self) -> Result<XaiEndpoints> {
        Ok(XaiEndpoints {
            authorize: endpoint(&self.endpoints.xai_issuer_url, "/oauth2/authorize")?,
            token: endpoint(&self.endpoints.xai_issuer_url, "/oauth2/token")?,
            device: endpoint(&self.endpoints.xai_issuer_url, "/oauth2/device/code")?,
        })
    }

    /// Discovers xAI's OIDC endpoints, preserving documented endpoint
    /// fallbacks when discovery is unavailable and refusing cross-origin
    /// endpoints when it is available.
    pub fn discover_xai_endpoints(&self, cancellation: &CancellationToken) -> Result<XaiEndpoints> {
        let issuer = self.endpoints.xai_issuer_url.to_string();
        if let Some(cached) = lock_unpoisoned(&self.xai_discovery).as_ref()
            && cached.issuer == issuer
        {
            return Ok(cached.endpoints.clone());
        }

        let fallback = self.xai_fallback_endpoints()?;
        let discovery_url = endpoint(
            &self.endpoints.xai_issuer_url,
            "/.well-known/openid-configuration",
        )?;
        let response = match self.get_json(&discovery_url, self.xai_discovery_timeout, cancellation)
        {
            Ok(response) => response,
            Err(OAuthError::Cancelled) => return Err(OAuthError::Cancelled),
            Err(_) => return Ok(fallback),
        };
        if !(200..300).contains(&response.status) {
            return Ok(fallback);
        }
        let document: XaiDiscoveryDocument = match serde_json::from_slice(&response.body) {
            Ok(document) => document,
            Err(_) => return Ok(fallback),
        };
        let mut resolved = fallback;
        if let Some(value) = pin_xai_endpoint(
            &self.endpoints.xai_issuer_url,
            &document.authorization_endpoint,
        ) {
            resolved.authorize = value;
        }
        if let Some(value) =
            pin_xai_endpoint(&self.endpoints.xai_issuer_url, &document.token_endpoint)
        {
            resolved.token = value;
        }
        if let Some(value) = pin_xai_endpoint(
            &self.endpoints.xai_issuer_url,
            &document.device_authorization_endpoint,
        ) {
            resolved.device = value;
        }

        *lock_unpoisoned(&self.xai_discovery) = Some(CachedXaiEndpoints {
            issuer,
            endpoints: resolved.clone(),
        });
        Ok(resolved)
    }

    /// Builds an xAI browser PKCE authorization URL from discovered endpoints.
    pub fn xai_authorization_url(
        &self,
        endpoints: &XaiEndpoints,
        client_id: &str,
        pkce: &PkcePair,
        state: &str,
        nonce: &str,
    ) -> Url {
        append_query(
            &endpoints.authorize,
            &[
                ("response_type", "code"),
                ("client_id", client_id),
                ("redirect_uri", XAI_REDIRECT_URI),
                ("scope", XAI_SCOPE),
                ("code_challenge", pkce.challenge()),
                ("code_challenge_method", "S256"),
                ("state", state),
                ("nonce", nonce),
                ("referrer", "goshcoder"),
            ],
        )
    }

    fn login_xai(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let method = interaction.prompt(OAuthPrompt {
            kind: OAuthPromptKind::Select,
            message: "Select the xAI (Grok) login method:".to_owned(),
            placeholder: String::new(),
            options: vec![
                OAuthPromptOption {
                    id: "device_code".to_owned(),
                    label: "Device code login (default)".to_owned(),
                    description: "Shows a code to enter at accounts.x.ai; works headless."
                        .to_owned(),
                },
                OAuthPromptOption {
                    id: "browser".to_owned(),
                    label: "Browser login".to_owned(),
                    description: "Opens a browser and waits on a loopback callback.".to_owned(),
                },
            ],
            cancellation: cancellation.clone(),
        })?;
        cancellation.check()?;
        match method.as_str() {
            "" | "device_code" => self.login_xai_device(interaction, environment, cancellation),
            "browser" => self.login_xai_browser(interaction, environment, cancellation),
            _ => Err(OAuthError::InvalidConfiguration(format!(
                "unknown xAI login method {method:?}"
            ))),
        }
    }

    fn login_xai_browser(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let endpoints = self.discover_xai_endpoints(cancellation)?;
        let client_id = xai_client_id(environment);
        let pkce = generate_pkce();
        let state = random_state();
        let nonce = random_state();
        let code = run_loopback_login(
            interaction.clone(),
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                authorization_url: self
                    .xai_authorization_url(&endpoints, &client_id, &pkce, &state, &nonce),
                redirect_uri: XAI_REDIRECT_URI.to_owned(),
                expected_state: state,
                callback_host: callback_host(environment),
                callback_port: XAI_CALLBACK_PORT,
                callback_path: XAI_CALLBACK_PATH.to_owned(),
            },
        )?;
        interaction.notify(OAuthEvent::progress(
            "Exchanging the authorization code for tokens...",
        ));
        let response = self.post_form(
            &endpoints.token,
            fields([
                ("grant_type", "authorization_code".to_owned()),
                ("client_id", client_id),
                ("code", code),
                ("redirect_uri", XAI_REDIRECT_URI.to_owned()),
                ("code_verifier", pkce.verifier().to_owned()),
                ("code_challenge", pkce.challenge().to_owned()),
                ("code_challenge_method", "S256".to_owned()),
            ]),
            cancellation,
        )?;
        let token = self.successful_token(OAuthProviderId::Xai, "exchange", response)?;
        self.xai_credential("exchange", token, "")
    }

    fn login_xai_device(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let endpoints = self.discover_xai_endpoints(cancellation)?;
        let client_id = xai_client_id(environment);
        let response = self.post_form(
            &endpoints.device,
            fields([
                ("client_id", client_id.clone()),
                ("scope", XAI_SCOPE.to_owned()),
            ]),
            cancellation,
        )?;
        if !(200..300).contains(&response.status) {
            return Err(operation_failure(
                OAuthProviderId::Xai.display_name(),
                "device authorization",
                &response,
            ));
        }
        let device: DeviceAuthorization = serde_json::from_slice(&response.body).map_err(|_| {
            OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Xai.display_name(),
                operation: "device authorization",
            }
        })?;
        let verification = trusted_http_url(&device.verification_uri);
        if device.device_code.is_empty() || device.user_code.is_empty() || verification.is_none() {
            return Err(OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Xai.display_name(),
                operation: "device authorization",
            });
        }
        let verification = trusted_http_url(&device.verification_uri_complete)
            .or(verification)
            .expect("verification URL was checked above");
        let interval = reported_device_interval(device.interval, DEFAULT_DEVICE_INTERVAL);
        let timeout = reported_device_timeout(device.expires_in, DEFAULT_DEVICE_TIMEOUT);
        interaction.notify(OAuthEvent::device_code(
            device.user_code.clone(),
            verification.to_string(),
            interval,
            timeout,
        ));
        self.poll_xai_device_token(
            &endpoints.token,
            &client_id,
            &device.device_code,
            interval,
            timeout,
            cancellation,
        )
    }

    fn poll_xai_device_token(
        &self,
        token_url: &Url,
        client_id: &str,
        device_code: &str,
        interval: Duration,
        timeout: Duration,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let token_url = token_url.clone();
        let client_id = client_id.to_owned();
        let device_code = device_code.to_owned();
        poll_device_code(
            self.clock.as_ref(),
            cancellation,
            DevicePollingPolicy::new(interval, timeout),
            || {
                let response = self.post_form(
                    &token_url,
                    fields([
                        ("grant_type", XAI_DEVICE_GRANT.to_owned()),
                        ("client_id", client_id.clone()),
                        ("device_code", device_code.clone()),
                    ]),
                    cancellation,
                )?;
                if (200..300).contains(&response.status) {
                    let token =
                        self.successful_token(OAuthProviderId::Xai, "device poll", response)?;
                    return self
                        .xai_credential("device poll", token, "")
                        .map(DevicePoll::Complete);
                }
                let failure = parse_device_failure(&response);
                match failure.error.as_str() {
                    "authorization_pending" => Ok(DevicePoll::Pending),
                    "slow_down" => Ok(DevicePoll::SlowDown(None)),
                    "expired_token" => Err(OAuthError::DeviceExpired {
                        provider: OAuthProviderId::Xai.display_name(),
                    }),
                    "access_denied" => Err(OAuthError::DeviceDenied {
                        provider: OAuthProviderId::Xai.display_name(),
                    }),
                    _ => Err(operation_failure(
                        OAuthProviderId::Xai.display_name(),
                        "device token request",
                        &response,
                    )),
                }
            },
        )
    }

    fn refresh_xai(
        &self,
        current: &Credential,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let endpoints = self.discover_xai_endpoints(cancellation)?;
        let response = self.post_form(
            &endpoints.token,
            fields([
                ("grant_type", "refresh_token".to_owned()),
                ("client_id", xai_client_id(environment)),
                ("refresh_token", current.refresh().to_owned()),
            ]),
            cancellation,
        )?;
        let token = self.successful_token(OAuthProviderId::Xai, "refresh", response)?;
        self.xai_credential("refresh", token, current.refresh())
    }

    fn xai_credential(
        &self,
        operation: &'static str,
        token: TokenResponse,
        previous_refresh: &str,
    ) -> Result<Credential> {
        if token.access_token.is_empty() {
            return Err(OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Xai.display_name(),
                operation,
            });
        }
        let refresh = if token.refresh_token.is_empty() {
            previous_refresh.to_owned()
        } else {
            token.refresh_token
        };
        if refresh.is_empty() {
            return Err(OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Xai.display_name(),
                operation,
            });
        }
        let expires_in = if token.expires_in > 0 {
            token.expires_in
        } else {
            3_600
        };
        Ok(Credential::oauth(
            token.access_token,
            refresh,
            self.clock
                .now_ms()
                .saturating_add(expires_in.saturating_mul(1_000)),
        ))
    }
}

#[derive(Default, Deserialize)]
struct XaiDiscoveryDocument {
    #[serde(default)]
    authorization_endpoint: String,
    #[serde(default)]
    token_endpoint: String,
    #[serde(default)]
    device_authorization_endpoint: String,
}

fn pin_xai_endpoint(issuer: &Url, value: &str) -> Option<Url> {
    if value.is_empty() {
        return None;
    }
    let parsed = Url::parse(value).ok()?;
    if !same_authority(&parsed, issuer) {
        return None;
    }
    if parsed.scheme() != "https" && !is_loopback_hostname(parsed.host_str()?) {
        return None;
    }
    Some(parsed)
}

fn same_authority(left: &Url, right: &Url) -> bool {
    let Some(left_host) = left.host_str() else {
        return false;
    };
    let Some(right_host) = right.host_str() else {
        return false;
    };
    // Pin discovery to the same effective network origin, including a
    // non-default port used by local test/development issuers.
    left_host.eq_ignore_ascii_case(right_host)
        && left.port_or_known_default() == right.port_or_known_default()
}

fn is_loopback_hostname(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false)
}

impl OAuthClient {
    fn meta_device_authorization_url(&self) -> Result<Url> {
        endpoint(
            &self.endpoints.meta_auth_base_url,
            "/oidc/device/authorization/",
        )
    }

    fn meta_token_url(&self) -> Result<Url> {
        endpoint(&self.endpoints.meta_auth_base_url, "/oidc/device/token/")
    }

    fn meta_mint_url(&self) -> Result<Url> {
        endpoint(&self.endpoints.meta_api_base_url, "/muse-code/key")
    }

    fn login_meta(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let client_id = meta_client_id(environment);
        let response = self.post_form(
            &self.meta_device_authorization_url()?,
            fields([("client_id", client_id.clone())]),
            cancellation,
        )?;
        if !(200..300).contains(&response.status) {
            return Err(operation_failure(
                OAuthProviderId::Meta.display_name(),
                "device authorization",
                &response,
            ));
        }
        let device: DeviceAuthorization = serde_json::from_slice(&response.body).map_err(|_| {
            OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "device authorization",
            }
        })?;
        let verification = trusted_http_url(&device.verification_uri);
        if device.device_code.is_empty() || device.user_code.is_empty() || verification.is_none() {
            return Err(OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "device authorization",
            });
        }
        let verification = trusted_http_url(&device.verification_uri_complete)
            .or(verification)
            .expect("verification URL was checked above");
        let interval = reported_device_interval(device.interval, DEFAULT_DEVICE_INTERVAL);
        let timeout = reported_device_timeout(device.expires_in, DEFAULT_DEVICE_TIMEOUT);
        interaction.notify(OAuthEvent::device_code(
            device.user_code.clone(),
            verification.to_string(),
            interval,
            timeout,
        ));
        let grant = self.poll_meta_device_token(
            &client_id,
            &device.device_code,
            interval,
            timeout,
            cancellation,
        )?;
        interaction.notify(OAuthEvent::progress("Requesting a Meta Model API key..."));
        let minted = self.mint_meta(&grant.access_token, cancellation)?;
        let identity_expires_at_ms = grant.expires_at_ms(self.clock.now_ms());
        self.meta_credential(
            grant.access_token,
            grant.refresh_token,
            identity_expires_at_ms,
            minted.api_key,
        )
    }

    fn poll_meta_device_token(
        &self,
        client_id: &str,
        device_code: &str,
        interval: Duration,
        timeout: Duration,
        cancellation: &CancellationToken,
    ) -> Result<MetaTokenGrant> {
        let token_url = self.meta_token_url()?;
        let client_id = client_id.to_owned();
        let device_code = device_code.to_owned();
        poll_device_code(
            self.clock.as_ref(),
            cancellation,
            DevicePollingPolicy::new(interval, timeout),
            || {
                let response = self.post_form(
                    &token_url,
                    fields([
                        ("grant_type", META_DEVICE_GRANT.to_owned()),
                        ("device_code", device_code.clone()),
                        ("client_id", client_id.clone()),
                    ]),
                    cancellation,
                )?;
                if (200..300).contains(&response.status) {
                    return parse_meta_token_grant(
                        &response.body,
                        OAuthProviderId::Meta.display_name(),
                        "device poll",
                    )
                    .map(DevicePoll::Complete);
                }
                let failure = parse_device_failure(&response);
                match failure.error.as_str() {
                    "authorization_pending" => Ok(DevicePoll::Pending),
                    "slow_down" => Ok(DevicePoll::SlowDown(None)),
                    "expired_token" => Err(OAuthError::DeviceExpired {
                        provider: OAuthProviderId::Meta.display_name(),
                    }),
                    "access_denied" => Err(OAuthError::DeviceDenied {
                        provider: OAuthProviderId::Meta.display_name(),
                    }),
                    _ => Err(operation_failure(
                        OAuthProviderId::Meta.display_name(),
                        "device token request",
                        &response,
                    )),
                }
            },
        )
    }

    fn refresh_meta(
        &self,
        current: &Credential,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let identity = current.refresh();
        let refresh_token = current
            .extra_string(META_REFRESH_TOKEN_EXTRA)
            .unwrap_or_default()
            .to_owned();
        let identity_expires = meta_identity_expiry(current);
        let identity_usable = !identity.is_empty()
            && (identity_expires == 0
                || self
                    .clock
                    .now_ms()
                    .saturating_add(duration_millis(MINIMUM_VALIDITY))
                    < identity_expires);
        if identity_usable {
            match self.mint_meta(identity, cancellation) {
                Ok(minted) => {
                    return self.meta_credential(
                        identity.to_owned(),
                        refresh_token,
                        identity_expires,
                        minted.api_key,
                    );
                }
                Err(error) if !error.is_unauthorized() => return Err(error),
                Err(_) => {}
            }
        }

        if refresh_token.is_empty() {
            return Err(OAuthError::Unauthorized {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "refresh",
                detail: Some("saved identity cannot be renewed".to_owned()),
            });
        }
        let grant = self.meta_exchange(
            environment,
            fields([
                ("grant_type", "refresh_token".to_owned()),
                ("refresh_token", refresh_token.clone()),
            ]),
            "refresh",
            cancellation,
        )?;
        let minted = self.mint_meta(&grant.access_token, cancellation)?;
        let identity_expires_at_ms = grant.expires_at_ms(self.clock.now_ms());
        self.meta_credential(
            grant.access_token,
            if grant.refresh_token.is_empty() {
                refresh_token
            } else {
                grant.refresh_token
            },
            identity_expires_at_ms,
            minted.api_key,
        )
    }

    fn meta_exchange(
        &self,
        environment: &dyn OAuthEnvironment,
        mut fields: BTreeMap<String, String>,
        operation: &'static str,
        cancellation: &CancellationToken,
    ) -> Result<MetaTokenGrant> {
        fields.insert("client_id".to_owned(), meta_client_id(environment));
        let response = self.post_form(&self.meta_token_url()?, fields, cancellation)?;
        if response.status == 404 {
            // Meta uses a bare 404 for a dead refresh token.
            return Err(OAuthError::Unauthorized {
                provider: OAuthProviderId::Meta.display_name(),
                operation,
                detail: None,
            });
        }
        if !(200..300).contains(&response.status) {
            return Err(token_error(
                OAuthProviderId::Meta.display_name(),
                operation,
                &response,
            ));
        }
        parse_meta_token_grant(
            &response.body,
            OAuthProviderId::Meta.display_name(),
            operation,
        )
    }

    fn mint_meta(&self, identity: &str, cancellation: &CancellationToken) -> Result<MetaMintedKey> {
        if identity.is_empty() {
            return Err(OAuthError::Unauthorized {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "mint",
                detail: Some("saved credential has no identity token".to_owned()),
            });
        }
        let response = self.post_json_with_headers(
            &self.meta_mint_url()?,
            json!({"dca_token": identity}),
            BTreeMap::from([
                ("Authorization".to_owned(), format!("Bearer {identity}")),
                ("x-api-version".to_owned(), META_API_VERSION.to_owned()),
            ]),
            cancellation,
        )?;
        if response.status == 401 || response.status == 403 {
            return Err(OAuthError::Unauthorized {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "mint",
                detail: None,
            });
        }
        if !(200..300).contains(&response.status) {
            let problem: MetaProblem = serde_json::from_slice(&response.body).unwrap_or_default();
            let detail = if !problem.detail.trim().is_empty() {
                problem.detail
            } else {
                truncate_response(&response.body, 300)
            };
            return Err(OAuthError::TokenFailure {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "mint",
                status: response.status,
                detail,
            });
        }
        let minted: MetaMintedKey = serde_json::from_slice(&response.body).map_err(|_| {
            OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "mint",
            }
        })?;
        if minted.require_payment {
            let detail = if minted.action_url.is_empty() {
                "this Meta account is not set up for the Model API yet".to_owned()
            } else {
                format!(
                    "this Meta account is not set up for the Model API yet; finish setup at {}",
                    minted.action_url
                )
            };
            return Err(OAuthError::TokenFailure {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "mint",
                status: 200,
                detail,
            });
        }
        if minted.api_key.is_empty() {
            return Err(OAuthError::InvalidTokenResponse {
                provider: OAuthProviderId::Meta.display_name(),
                operation: "mint",
            });
        }
        Ok(minted)
    }

    fn meta_credential(
        &self,
        identity: String,
        refresh_token: String,
        identity_expires_at_ms: i64,
        api_key: String,
    ) -> Result<Credential> {
        let mut expires_at_ms = self
            .clock
            .now_ms()
            .saturating_add(duration_millis(META_KEY_VALIDITY));
        if identity_expires_at_ms > 0 && identity_expires_at_ms < expires_at_ms {
            expires_at_ms = identity_expires_at_ms;
        }
        let mut credential = Credential::oauth(api_key, identity, expires_at_ms);
        if !refresh_token.is_empty() {
            credential
                .set_extra(META_REFRESH_TOKEN_EXTRA, Value::String(refresh_token))
                .map_err(OAuthError::Storage)?;
        }
        if identity_expires_at_ms > 0 {
            credential
                .set_extra(
                    META_IDENTITY_EXPIRES_EXTRA,
                    Value::String(identity_expires_at_ms.to_string()),
                )
                .map_err(OAuthError::Storage)?;
        }
        Ok(credential)
    }
}

#[derive(Default, Deserialize)]
struct MetaTokenGrant {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    refresh_token: String,
    #[serde(default)]
    expires_in: i64,
}

impl MetaTokenGrant {
    fn expires_at_ms(&self, now_ms: i64) -> i64 {
        if self.expires_in <= 0 {
            0
        } else {
            now_ms.saturating_add(self.expires_in.saturating_mul(1_000))
        }
    }
}

fn parse_meta_token_grant(
    body: &[u8],
    provider: &'static str,
    operation: &'static str,
) -> Result<MetaTokenGrant> {
    let grant: MetaTokenGrant =
        serde_json::from_slice(body).map_err(|_| OAuthError::InvalidTokenResponse {
            provider,
            operation,
        })?;
    if grant.access_token.is_empty() {
        return Err(OAuthError::InvalidTokenResponse {
            provider,
            operation,
        });
    }
    Ok(grant)
}

fn meta_identity_expiry(credential: &Credential) -> i64 {
    credential
        .extra_string(META_IDENTITY_EXPIRES_EXTRA)
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or_default()
}

#[derive(Default, Deserialize)]
struct MetaMintedKey {
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    require_payment: bool,
    #[serde(default)]
    action_url: String,
}

#[derive(Default, Deserialize)]
struct MetaProblem {
    #[serde(default)]
    detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::VecDeque, net::TcpStream, sync::Mutex};

    struct FakeClock {
        now: Mutex<i64>,
        sleeps: Mutex<Vec<Duration>>,
    }

    impl FakeClock {
        fn new(now: i64) -> Self {
            Self {
                now: Mutex::new(now),
                sleeps: Mutex::new(Vec::new()),
            }
        }

        fn sleeps(&self) -> Vec<Duration> {
            lock_unpoisoned(&self.sleeps).clone()
        }
    }

    impl OAuthClock for FakeClock {
        fn now_ms(&self) -> i64 {
            *lock_unpoisoned(&self.now)
        }

        fn sleep(&self, duration: Duration, cancellation: &CancellationToken) -> Result<()> {
            cancellation.check()?;
            lock_unpoisoned(&self.sleeps).push(duration);
            let mut now = lock_unpoisoned(&self.now);
            *now = now.saturating_add(duration_millis(duration));
            Ok(())
        }
    }

    struct FakeTransport {
        requests: Mutex<Vec<OAuthRequest>>,
        responses: Mutex<VecDeque<Result<OAuthResponse>>>,
    }

    impl FakeTransport {
        fn with_responses(responses: impl IntoIterator<Item = OAuthResponse>) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                responses: Mutex::new(responses.into_iter().map(Ok).collect()),
            }
        }

        fn requests(&self) -> Vec<OAuthRequest> {
            lock_unpoisoned(&self.requests).clone()
        }
    }

    impl OAuthTransport for FakeTransport {
        fn execute(
            &self,
            request: OAuthRequest,
            cancellation: &CancellationToken,
        ) -> Result<OAuthResponse> {
            cancellation.check()?;
            lock_unpoisoned(&self.requests).push(request);
            lock_unpoisoned(&self.responses)
                .pop_front()
                .unwrap_or_else(|| {
                    Err(OAuthError::Transport(
                        "test transport received an unexpected request".to_owned(),
                    ))
                })
        }
    }

    fn response(status: u16, body: impl Into<Vec<u8>>) -> OAuthResponse {
        OAuthResponse {
            status,
            body: body.into(),
        }
    }

    fn test_client(
        transport: Arc<FakeTransport>,
        clock: Arc<FakeClock>,
        endpoints: OAuthEndpoints,
    ) -> OAuthClient {
        OAuthClient::new(transport, clock, Arc::new(NoopBrowser), endpoints)
    }

    fn form(request: &OAuthRequest) -> BTreeMap<String, String> {
        url::form_urlencoded::parse(request.body())
            .into_owned()
            .collect()
    }

    fn codex_jwt(account_id: &str) -> String {
        let payload = json!({
            CODEX_AUTH_CLAIM: {"chatgpt_account_id": account_id},
        })
        .to_string();
        format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(payload.as_bytes())
        )
    }

    #[test]
    fn pkce_challenge_matches_verifier() {
        let pair = generate_pkce();
        let raw = URL_SAFE_NO_PAD
            .decode(pair.verifier())
            .expect("verifier is URL-safe base64");
        assert_eq!(raw.len(), 32);
        assert_eq!(
            pair.challenge(),
            URL_SAFE_NO_PAD.encode(Sha256::digest(pair.verifier().as_bytes()))
        );
        assert_ne!(random_state(), random_state());
    }

    #[test]
    fn parses_manual_authorization_inputs() {
        let cases = [
            (
                "http://localhost/callback?code=url-code&state=url-state",
                "url-code",
                "url-state",
            ),
            ("hash-code#hash-state", "hash-code", "hash-state"),
            (
                "code=query-code&state=query-state",
                "query-code",
                "query-state",
            ),
            ("  bare-code ", "bare-code", ""),
        ];
        for (input, code, state) in cases {
            let parsed = parse_authorization_input(input).expect("authorization input");
            assert_eq!(parsed.code, code);
            assert_eq!(parsed.state, state);
        }
        assert!(parse_authorization_input("  ").is_none());
    }

    #[test]
    fn provider_registry_is_explicit_about_metadata_only_entries() {
        assert_eq!(
            implemented_provider_ids(),
            vec!["anthropic", "kimi-coding", "meta", "openai-codex", "xai"]
        );
        let openrouter = metadata_for(OAuthProviderId::OpenRouter);
        assert_eq!(openrouter.flow_support, OAuthFlowSupport::MetadataOnly);
        assert_eq!(openrouter.methods, &[LoginMethod::ApiKeyOnly]);
        assert_eq!(
            OAuthProviderId::parse("openai-codex"),
            Some(OAuthProviderId::OpenAiCodex)
        );
    }

    #[test]
    fn oauth_credentials_keep_auth_json_shape_and_provider_extras() {
        let mut credential = Credential::oauth("access", "refresh", 123_456);
        credential
            .set_extra("accountId", Value::String("acct-1".to_owned()))
            .expect("set extra");
        let value = serde_json::to_value(&credential).expect("serialize credential");
        assert_eq!(value["type"], "oauth");
        assert_eq!(value["access"], "access");
        assert_eq!(value["refresh"], "refresh");
        assert_eq!(value["expires"], 123_456);
        assert_eq!(value["accountId"], "acct-1");

        let meta = Credential::oauth("model-key", "identity", 1);
        let auth = auth_from_credential(OAuthProviderId::Meta, &meta).expect("meta auth");
        assert_eq!(auth.api_key(), None);
        assert_eq!(
            auth.headers()
                .get("Authorization")
                .and_then(Option::as_deref),
            Some("Bearer model-key")
        );
        let kimi = auth_from_credential(OAuthProviderId::KimiCoding, &meta).expect("Kimi auth");
        assert_eq!(kimi.api_key(), Some("model-key"));
        assert_eq!(
            kimi.headers()
                .get("Authorization")
                .and_then(Option::as_deref),
            Some("Bearer model-key")
        );
    }

    #[test]
    fn expiry_window_matches_go_behavior() {
        let credential = Credential::oauth("a", "r", 1_000_000);
        assert!(!credential_expires_soon_at(
            &credential,
            1_000_000 - duration_millis(MINIMUM_VALIDITY) - 1,
            MINIMUM_VALIDITY
        ));
        assert!(credential_expires_soon_at(
            &credential,
            1_000_000 - duration_millis(MINIMUM_VALIDITY),
            MINIMUM_VALIDITY
        ));
    }

    #[test]
    fn authorization_urls_include_pkce_and_provider_fields() {
        let transport = Arc::new(FakeTransport::with_responses([]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport, clock, OAuthEndpoints::default());
        let pkce = generate_pkce();

        let anthropic = client.anthropic_authorization_url(&pkce);
        assert_eq!(query_value(&anthropic, "client_id"), ANTHROPIC_CLIENT_ID);
        assert_eq!(
            query_value(&anthropic, "redirect_uri"),
            ANTHROPIC_REDIRECT_URI
        );
        assert_eq!(query_value(&anthropic, "state"), pkce.verifier());
        assert_eq!(query_value(&anthropic, "code_challenge"), pkce.challenge());

        let codex = client.codex_authorization_url(&pkce, "csrf-state");
        assert_eq!(query_value(&codex, "client_id"), CODEX_CLIENT_ID);
        assert_eq!(query_value(&codex, "state"), "csrf-state");
        assert_eq!(query_value(&codex, "codex_cli_simplified_flow"), "true");

        let endpoints = client.xai_fallback_endpoints().expect("xAI fallback");
        let xai = client.xai_authorization_url(
            &endpoints,
            XAI_DEFAULT_CLIENT_ID,
            &pkce,
            "state",
            "nonce",
        );
        assert_eq!(query_value(&xai, "redirect_uri"), XAI_REDIRECT_URI);
        assert_eq!(query_value(&xai, "referrer"), "goshcoder");
        assert_eq!(query_value(&xai, "nonce"), "nonce");
    }

    #[test]
    fn endpoint_overrides_preserve_configured_path_and_xai_pinning_is_strict() {
        let base = fixed_url("http://127.0.0.1:4010/proxy/");
        assert_eq!(
            endpoint(&base, "/api/oauth/token")
                .expect("endpoint")
                .as_str(),
            "http://127.0.0.1:4010/proxy/api/oauth/token"
        );
        assert!(!same_authority(
            &fixed_url("https://auth.x.ai:444/oauth2/token"),
            &fixed_url("https://auth.x.ai")
        ));
    }

    #[test]
    fn loopback_server_rejects_remote_hosts_and_escapes_messages() {
        assert!(LoopbackCallbackServer::bind("0.0.0.0", 0, "/callback", "state").is_err());
        assert_eq!(
            escape_html("<script>alert('x')</script>"),
            "&lt;script&gt;alert(&#39;x&#39;)&lt;/script&gt;"
        );
        assert!(!constant_time_eq("state", "other"));
        assert!(constant_time_eq("state", "state"));
    }

    struct CallbackInteraction {
        port: u16,
        state: String,
        events: Mutex<Vec<OAuthEvent>>,
    }

    impl OAuthInteraction for CallbackInteraction {
        fn prompt(&self, prompt: OAuthPrompt) -> Result<String> {
            while !prompt.cancellation.is_cancelled() {
                thread::sleep(Duration::from_millis(1));
            }
            Err(OAuthError::Cancelled)
        }

        fn notify(&self, event: OAuthEvent) {
            if event.kind == OAuthEventKind::AuthorizationUrl {
                let port = self.port;
                let state = self.state.clone();
                thread::spawn(move || {
                    let mut stream =
                        TcpStream::connect(("127.0.0.1", port)).expect("connect callback listener");
                    write!(
                        stream,
                        "GET /callback?code=callback-code&state={state} HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                    .expect("write callback");
                    let mut body = String::new();
                    let _ = stream.read_to_string(&mut body);
                });
            }
            lock_unpoisoned(&self.events).push(event);
        }
    }

    fn unused_loopback_port() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind unused port");
        let port = listener.local_addr().expect("address").port();
        drop(listener);
        port
    }

    #[test]
    fn loopback_login_accepts_callback_and_cancels_manual_prompt() {
        let port = unused_loopback_port();
        let interaction = Arc::new(CallbackInteraction {
            port,
            state: "expected-state".to_owned(),
            events: Mutex::new(Vec::new()),
        });
        let result = run_loopback_login(
            interaction.clone(),
            Arc::new(NoopBrowser),
            &CancellationToken::new(),
            LoopbackLoginRequest {
                authorization_url: fixed_url("https://example.test/authorize"),
                redirect_uri: "http://localhost/callback".to_owned(),
                expected_state: "expected-state".to_owned(),
                callback_host: "127.0.0.1".to_owned(),
                callback_port: port,
                callback_path: "/callback".to_owned(),
            },
        )
        .expect("callback completes login");
        assert_eq!(result, "callback-code");
        assert!(
            lock_unpoisoned(&interaction.events)
                .iter()
                .any(|event| event.kind == OAuthEventKind::AuthorizationUrl)
        );
    }

    #[test]
    fn device_polling_is_cancellable_and_testable_without_real_sleep() {
        let clock = FakeClock::new(0);
        let cancellation = CancellationToken::new();
        let mut calls = 0;
        let value = poll_device_code(
            &clock,
            &cancellation,
            DevicePollingPolicy::new(Duration::from_secs(1), Duration::from_secs(5)),
            || {
                calls += 1;
                if calls == 1 {
                    Ok(DevicePoll::Pending)
                } else {
                    Ok(DevicePoll::Complete("done"))
                }
            },
        )
        .expect("device poll");
        assert_eq!(value, "done");
        assert_eq!(calls, 2);
        assert_eq!(clock.sleeps(), vec![Duration::from_secs(1)]);

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let mut called = false;
        assert!(matches!(
            poll_device_code(
                &clock,
                &cancelled,
                DevicePollingPolicy::new(Duration::from_secs(1), Duration::from_secs(5)),
                || {
                    called = true;
                    Ok(DevicePoll::<()>::Pending)
                }
            ),
            Err(OAuthError::Cancelled)
        ));
        assert!(!called);
    }

    #[test]
    fn anthropic_refresh_applies_skew_and_posts_json() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#
                .to_vec(),
        )]));
        let clock = Arc::new(FakeClock::new(1_000_000));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let credential = client
            .refresh(
                OAuthProviderId::Anthropic,
                &Credential::oauth("old", "old-refresh", 0),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(credential.access(), "new-access");
        assert_eq!(credential.refresh(), "new-refresh");
        assert_eq!(
            credential.expires_at_ms(),
            1_000_000 + 3_600_000 - duration_millis(ANTHROPIC_REFRESH_SKEW)
        );
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(requests[0].body()).expect("JSON body");
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["refresh_token"], "old-refresh");
        assert_eq!(body["client_id"], ANTHROPIC_CLIENT_ID);
    }

    #[test]
    fn kimi_refresh_retries_transient_failure_and_honors_backoff_clock() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(500, b"{}".to_vec()),
            response(
                200,
                br#"{"access_token":"after-retry","refresh_token":"new-refresh","expires_in":3600}"#
                    .to_vec(),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock.clone(), OAuthEndpoints::default());
        let credential = client
            .refresh(
                OAuthProviderId::KimiCoding,
                &Credential::oauth("old", "refresh", 0),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("refresh");
        assert_eq!(credential.access(), "after-retry");
        assert_eq!(transport.requests().len(), 2);
        assert_eq!(clock.sleeps(), vec![Duration::from_secs(1)]);
        let request = transport.requests().pop().expect("refresh request");
        assert_eq!(
            form(&request).get("client_id"),
            Some(&KIMI_CLIENT_ID.to_owned())
        );
    }

    #[test]
    fn codex_refresh_extracts_account_id_and_rejects_missing_claim() {
        let access = codex_jwt("acct-codex");
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            format!(r#"{{"access_token":{access:?},"refresh_token":"rotated","expires_in":3600}}"#)
                .into_bytes(),
        )]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport, clock, OAuthEndpoints::default());
        let credential = client
            .refresh(
                OAuthProviderId::OpenAiCodex,
                &Credential::oauth("old", "refresh", 0),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("Codex refresh");
        assert_eq!(credential.extra_string("accountId"), Some("acct-codex"));
        assert!(codex_account_id("not-a-jwt").is_err());
    }

    #[test]
    fn xai_discovery_is_pinned_and_keeps_nonrotating_refresh_tokens() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(
                200,
                br#"{
                    "authorization_endpoint":"https://auth.x.ai/oauth2/custom-authorize",
                    "token_endpoint":"https://elsewhere.test/stolen-token",
                    "device_authorization_endpoint":"https://auth.x.ai/oauth2/custom-device"
                }"#
                .to_vec(),
            ),
            response(
                200,
                br#"{"access_token":"fresh","expires_in":3600}"#.to_vec(),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let credential = client
            .refresh(
                OAuthProviderId::Xai,
                &Credential::oauth("stale", "long-lived-refresh", 0),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("xAI refresh");
        assert_eq!(credential.access(), "fresh");
        assert_eq!(credential.refresh(), "long-lived-refresh");
        let requests = transport.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].url().path(), "/oauth2/token");
        assert_ne!(requests[1].url().host_str(), Some("elsewhere.test"));
    }

    #[test]
    fn meta_refresh_remints_without_spending_identity_refresh_token() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"api_key":"new-model-key","require_payment":false}"#.to_vec(),
        )]));
        let clock = Arc::new(FakeClock::new(1_000));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let mut current = Credential::oauth("old-model-key", "identity", 24 * 60 * 60 * 1_000);
        current
            .set_extra(
                META_REFRESH_TOKEN_EXTRA,
                Value::String("meta-refresh".to_owned()),
            )
            .expect("meta refresh extra");
        let refreshed = client
            .refresh(
                OAuthProviderId::Meta,
                &current,
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("re-mint");
        assert_eq!(refreshed.access(), "new-model-key");
        assert_eq!(refreshed.refresh(), "identity");
        assert_eq!(
            refreshed.extra_string(META_REFRESH_TOKEN_EXTRA),
            Some("meta-refresh")
        );
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0]
                .headers()
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer identity")
        );
    }

    #[test]
    fn stored_oauth_refreshes_under_the_store_lock_and_persists_rotation() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"access_token":"fresh","refresh_token":"rotated","expires_in":3600}"#.to_vec(),
        )]));
        let clock = Arc::new(FakeClock::new(1_000_000));
        let client = test_client(transport, clock.clone(), OAuthEndpoints::default());
        let store = CredentialStore::in_memory();
        store
            .put(
                "anthropic",
                Credential::oauth("stale", "old-refresh", clock.now_ms() - 1),
            )
            .expect("store credential");
        let auth = client
            .resolve_stored_oauth(
                OAuthProviderId::Anthropic,
                &store,
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("resolve OAuth")
            .expect("OAuth auth");
        assert_eq!(auth.api_key(), Some("fresh"));
        let stored = store
            .read_raw("anthropic")
            .expect("read store")
            .expect("stored credential");
        assert_eq!(stored.access(), "fresh");
        assert_eq!(stored.refresh(), "rotated");
    }

    struct PromptInteraction {
        answers: Mutex<VecDeque<String>>,
        events: Mutex<Vec<OAuthEvent>>,
    }

    impl PromptInteraction {
        fn answers(answers: impl IntoIterator<Item = &'static str>) -> Self {
            Self {
                answers: Mutex::new(answers.into_iter().map(str::to_owned).collect()),
                events: Mutex::new(Vec::new()),
            }
        }

        fn event_kinds(&self) -> Vec<OAuthEventKind> {
            lock_unpoisoned(&self.events)
                .iter()
                .map(|event| event.kind)
                .collect()
        }
    }

    impl OAuthInteraction for PromptInteraction {
        fn prompt(&self, _: OAuthPrompt) -> Result<String> {
            lock_unpoisoned(&self.answers)
                .pop_front()
                .ok_or(OAuthError::Cancelled)
        }

        fn notify(&self, event: OAuthEvent) {
            lock_unpoisoned(&self.events).push(event);
        }
    }

    #[test]
    fn kimi_device_login_waits_then_persists_an_oauth_compatible_credential() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(
                200,
                br#"{
                    "device_code":"device-1",
                    "user_code":"ABCD",
                    "verification_uri":"https://auth.example/device",
                    "verification_uri_complete":"https://auth.example/device?code=ABCD",
                    "interval":0.01,
                    "expires_in":600
                }"#
                .to_vec(),
            ),
            response(
                200,
                br#"{"access_token":"kimi-access","refresh_token":"kimi-refresh","expires_in":3600}"#
                    .to_vec(),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock.clone(), OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers([]));
        let credential = client
            .login(
                OAuthProviderId::KimiCoding,
                interaction.clone(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("Kimi device login");
        assert_eq!(credential.access(), "kimi-access");
        assert_eq!(credential.refresh(), "kimi-refresh");
        assert_eq!(clock.sleeps(), vec![Duration::from_secs(1)]);
        assert_eq!(interaction.event_kinds(), vec![OAuthEventKind::DeviceCode]);
        let requests = transport.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            form(&requests[1]).get("device_code"),
            Some(&"device-1".to_owned())
        );
        assert_eq!(
            form(&requests[1]).get("grant_type"),
            Some(&KIMI_DEVICE_GRANT.to_owned())
        );
    }

    #[test]
    fn codex_device_login_exchanges_device_code_and_stores_account_id() {
        let access = codex_jwt("acct-device");
        let transport = Arc::new(FakeTransport::with_responses([
            response(
                200,
                br#"{"device_auth_id":"device-auth","user_code":"DEVICE","interval":"0"}"#.to_vec(),
            ),
            response(
                200,
                br#"{"authorization_code":"authorization-code","code_verifier":"device-verifier"}"#
                    .to_vec(),
            ),
            response(
                200,
                serde_json::to_vec(&json!({
                    "access_token": access,
                    "refresh_token": "codex-refresh",
                    "expires_in": 3600,
                }))
                .expect("token JSON"),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers(["device_code"]));
        let credential = client
            .login(
                OAuthProviderId::OpenAiCodex,
                interaction.clone(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("Codex device login");
        assert_eq!(credential.extra_string("accountId"), Some("acct-device"));
        assert_eq!(
            interaction.event_kinds(),
            vec![OAuthEventKind::DeviceCode, OAuthEventKind::Progress]
        );
        let requests = transport.requests();
        assert_eq!(requests.len(), 3);
        let device_poll: Value =
            serde_json::from_slice(requests[1].body()).expect("device poll JSON");
        assert_eq!(device_poll["device_auth_id"], "device-auth");
        assert_eq!(
            form(&requests[2]).get("code"),
            Some(&"authorization-code".to_owned())
        );
        assert_eq!(
            form(&requests[2]).get("code_verifier"),
            Some(&"device-verifier".to_owned())
        );
    }

    #[test]
    fn xai_device_login_uses_discovery_and_device_grant() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(
                200,
                br#"{
                    "authorization_endpoint":"https://auth.x.ai/oauth2/authorize",
                    "token_endpoint":"https://auth.x.ai/oauth2/token",
                    "device_authorization_endpoint":"https://auth.x.ai/oauth2/device/code"
                }"#
                .to_vec(),
            ),
            response(
                200,
                br#"{
                    "device_code":"xai-device",
                    "user_code":"AAAA-BBBB",
                    "verification_uri":"https://accounts.x.ai/oauth2/device",
                    "expires_in":600,
                    "interval":1
                }"#
                .to_vec(),
            ),
            response(
                200,
                br#"{"access_token":"xai-access","refresh_token":"xai-refresh","expires_in":3600}"#
                    .to_vec(),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers(["device_code"]));
        let credential = client
            .login(
                OAuthProviderId::Xai,
                interaction.clone(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("xAI device login");
        assert_eq!(credential.access(), "xai-access");
        assert_eq!(credential.refresh(), "xai-refresh");
        assert_eq!(interaction.event_kinds(), vec![OAuthEventKind::DeviceCode]);
        let requests = transport.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            form(&requests[1]).get("client_id"),
            Some(&XAI_DEFAULT_CLIENT_ID.to_owned())
        );
        assert_eq!(
            form(&requests[2]).get("grant_type"),
            Some(&XAI_DEVICE_GRANT.to_owned())
        );
    }

    #[test]
    fn meta_device_login_mints_a_model_api_key_and_keeps_identity_as_refresh() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(
                200,
                br#"{
                    "device_code":"meta-device",
                    "user_code":"VGGF-VLQT",
                    "verification_uri":"https://auth.meta.com/oauth/device/",
                    "verification_uri_complete":"https://auth.meta.com/oauth/device/?code=VGGF-VLQT",
                    "expires_in":600,
                    "interval":1
                }"#
                .to_vec(),
            ),
            response(
                200,
                br#"{"access_token":"identity-token","refresh_token":"meta-refresh","expires_in":3600}"#
                    .to_vec(),
            ),
            response(
                200,
                br#"{"api_key":"model-api-key","require_payment":false}"#.to_vec(),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers([]));
        let credential = client
            .login(
                OAuthProviderId::Meta,
                interaction.clone(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("Meta device login");
        assert_eq!(credential.access(), "model-api-key");
        assert_eq!(credential.refresh(), "identity-token");
        assert_eq!(
            credential.extra_string(META_REFRESH_TOKEN_EXTRA),
            Some("meta-refresh")
        );
        assert_eq!(
            interaction.event_kinds(),
            vec![OAuthEventKind::DeviceCode, OAuthEventKind::Progress]
        );
        let requests = transport.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(
            requests[2]
                .headers()
                .get("x-api-version")
                .map(String::as_str),
            Some(META_API_VERSION)
        );
        let mint: Value = serde_json::from_slice(requests[2].body()).expect("mint JSON");
        assert_eq!(mint["dca_token"], "identity-token");
    }

    #[test]
    fn terminal_refresh_failure_keeps_the_existing_stored_credential() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            401,
            br#"{"error":"invalid_grant","error_description":"refresh token revoked"}"#.to_vec(),
        )]));
        let clock = Arc::new(FakeClock::new(1_000_000));
        let client = test_client(transport, clock.clone(), OAuthEndpoints::default());
        let store = CredentialStore::in_memory();
        store
            .put(
                "anthropic",
                Credential::oauth("stale", "still-on-disk", clock.now_ms() - 1),
            )
            .expect("store credential");
        let error = match client.resolve_stored_oauth(
            OAuthProviderId::Anthropic,
            &store,
            &BTreeMap::new(),
            &CancellationToken::new(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("refresh should be terminal"),
        };
        assert!(error.is_unauthorized());
        let preserved = store
            .read_raw("anthropic")
            .expect("read credential")
            .expect("credential is retained");
        assert_eq!(preserved.access(), "stale");
        assert_eq!(preserved.refresh(), "still-on-disk");
    }
}
