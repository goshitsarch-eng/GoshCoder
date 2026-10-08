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

use crate::{
    catalog::{
        Auth, AuthHeaders, CatalogError, Credential, CredentialKind, CredentialStore,
        EnvironmentLookup,
    },
    meta_muse,
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

/// Sent on every OAuth request. Meta's device-authorization endpoint answers a
/// request without a `User-Agent` with an empty 302 instead of JSON, which
/// surfaced as "returned an incomplete or invalid response" at login.
pub const OAUTH_USER_AGENT: &str = concat!("goshcoder/", env!("CARGO_PKG_VERSION"));

const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
const ANTHROPIC_CALLBACK_PORT: u16 = 53692;
const ANTHROPIC_CALLBACK_PATH: &str = "/callback";
const ANTHROPIC_REDIRECT_URI: &str = "http://localhost:53692/callback";
const ANTHROPIC_COPY_CODE_REDIRECT_URI: &str = "https://platform.claude.com/oauth/code/callback";
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
    /// A [`CancellationToken`] deadline elapsed before the flow finished.
    TimedOut,
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
    /// A provider-specific failure whose wording is the provider's own (the
    /// Meta Muse port keeps the extension's messages verbatim).
    /// `unauthorized` marks one only a new login can fix.
    Message {
        message: String,
        unauthorized: bool,
    },
}

impl OAuthError {
    /// Whether retrying cannot repair this credential and a fresh login is
    /// required.
    pub fn is_unauthorized(&self) -> bool {
        matches!(
            self,
            Self::Unauthorized { .. }
                | Self::Message {
                    unauthorized: true,
                    ..
                }
        )
    }
}

impl fmt::Display for OAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("OAuth login was cancelled"),
            Self::TimedOut => formatter.write_str("OAuth request exceeded its time budget"),
            Self::Unauthorized {
                provider,
                operation,
                detail,
            } => {
                // Only a refresh can lose an authorization the user had; a
                // rejected login step just has to be started over.
                if *operation == "refresh" {
                    write!(
                        formatter,
                        "{provider} token refresh is no longer authorized; log in again"
                    )?;
                } else {
                    write!(
                        formatter,
                        "{provider} rejected the token {operation}; start the login again"
                    )?;
                }
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
            Self::Message { message, .. } => formatter.write_str(message),
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
///
/// A token may also carry a deadline. A refresh that nobody asked for
/// interactively (provider enumeration, model resolution) uses one so a slow
/// token endpoint bounds the caller's wait instead of the transport's full
/// retry allowance.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a token that reports itself cancelled once `timeout` elapses.
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            cancelled: Arc::default(),
            deadline: Instant::now().checked_add(timeout),
        }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn deadline_passed(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire) || self.deadline_passed()
    }

    /// Time left before the deadline; `None` for a token without one.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    pub fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(OAuthError::Cancelled)
        } else if self.deadline_passed() {
            Err(OAuthError::TimedOut)
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
    MetaMuse,
    OpenAiCodex,
    OpenRouter,
    Xai,
    GrokCli,
    GithubCopilot,
    Radius,
}

impl OAuthProviderId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::KimiCoding => "kimi-coding",
            Self::Meta => "meta",
            Self::MetaMuse => "meta-muse",
            Self::OpenAiCodex => "openai-codex",
            Self::OpenRouter => "openrouter",
            Self::Xai => "xai",
            Self::GrokCli => "grok-cli",
            Self::GithubCopilot => "github-copilot",
            Self::Radius => "radius",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "anthropic" => Some(Self::Anthropic),
            "kimi-coding" => Some(Self::KimiCoding),
            "meta" => Some(Self::Meta),
            "meta-muse" => Some(Self::MetaMuse),
            "openai-codex" => Some(Self::OpenAiCodex),
            "openrouter" => Some(Self::OpenRouter),
            "xai" => Some(Self::Xai),
            "grok-cli" => Some(Self::GrokCli),
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
            Self::MetaMuse => "Meta Muse Code (subscription)",
            Self::OpenAiCodex => "OpenAI (ChatGPT Plus/Pro)",
            Self::OpenRouter => "OpenRouter",
            Self::Xai => "xAI (Grok subscription)",
            Self::GrokCli => "Grok CLI",
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
const GROK_CLI_METHODS: &[LoginMethod] = &[LoginMethod::BrowserPkce, LoginMethod::DeviceCode];
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
    // The pi-meta-muse-auth extension: the same auth.meta.com device flow,
    // but the key is minted against the Muse Code subscription.
    ProviderMetadata {
        id: OAuthProviderId::MetaMuse,
        display_name: "Meta Muse Code (subscription)",
        methods: DEVICE_METHOD,
        flow_support: OAuthFlowSupport::Implemented,
    },
    ProviderMetadata {
        id: OAuthProviderId::OpenAiCodex,
        display_name: "OpenAI (ChatGPT Plus/Pro)",
        methods: CODEX_METHODS,
        flow_support: OAuthFlowSupport::Implemented,
    },
    // pi's `auth/oauth/openrouter.ts`: a PKCE browser flow that mints a
    // permanent, user-controlled API key rather than a refreshable token.
    ProviderMetadata {
        id: OAuthProviderId::OpenRouter,
        display_name: "OpenRouter",
        methods: BROWSER_METHOD,
        flow_support: OAuthFlowSupport::Implemented,
    },
    ProviderMetadata {
        id: OAuthProviderId::Xai,
        display_name: "xAI (Grok subscription)",
        methods: XAI_METHODS,
        flow_support: OAuthFlowSupport::Implemented,
    },
    // pi-grok-cli: the xAI flow with the official Grok CLI's parameters.
    ProviderMetadata {
        id: OAuthProviderId::GrokCli,
        display_name: "Grok CLI",
        methods: GROK_CLI_METHODS,
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
    base_url: Option<String>,
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

    /// The API base URL the credential itself names, which replaces the
    /// provider default. Only Meta Muse sets one: Meta returns it with the
    /// minted key.
    pub fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
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
    let mut base_url = None;
    let api_key = match provider {
        OAuthProviderId::KimiCoding => {
            headers.insert("Authorization".to_owned(), Some(format!("Bearer {access}")));
            Some(access.to_owned())
        }
        OAuthProviderId::Meta => {
            headers.insert("Authorization".to_owned(), Some(format!("Bearer {access}")));
            None
        }
        // An openai-responses provider, so the key is an ordinary bearer API
        // key; the stored base URL is re-validated on every use (`toAuth`).
        OAuthProviderId::MetaMuse => {
            let (api_key, sanctioned) = meta_muse::request_auth(credential)?;
            base_url = Some(sanctioned);
            Some(api_key)
        }
        OAuthProviderId::Anthropic
        | OAuthProviderId::OpenAiCodex
        | OAuthProviderId::OpenRouter
        | OAuthProviderId::Xai
        | OAuthProviderId::GrokCli
        | OAuthProviderId::GithubCopilot
        | OAuthProviderId::Radius => Some(access.to_owned()),
    };
    Ok(OAuthAuth {
        api_key,
        headers,
        source: "OAuth".to_owned(),
        base_url,
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

/// The provider's reason when a pasted redirect URL carries `error=` rather
/// than a code.
fn redirect_error(input: &str) -> Option<String> {
    let url = Url::parse(input.trim()).ok()?;
    let error = query_value(&url, "error");
    if error.is_empty() {
        return None;
    }
    let description = query_value(&url, "error_description");
    Some(if description.is_empty() {
        error
    } else {
        description
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

    pub(crate) fn device_code(
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

    pub(crate) fn progress(message: impl Into<String>) -> Self {
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

/// A single callback connection may take this long to deliver its request
/// line. Browsers redirect in milliseconds; only a stuck or hostile client
/// needs more.
const CALLBACK_REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const CALLBACK_REQUEST_LIMIT: usize = 16 * 1024;

/// A validated loopback callback listener. It accepts only the expected path
/// and CSRF state, and returns an escaped, no-store browser response.
///
/// An empty expected state means the provider sends none (OpenRouter); its
/// random callback path then keeps stray requests from completing a login.
///
/// Anything one connection does wrong (a malformed or oversized request, a
/// reset socket, a client that never finishes) is answered or dropped and
/// then forgotten: the login keeps waiting for the browser's real callback or
/// the manual paste. Only cancellation ends the wait from here.
pub struct LoopbackCallbackServer {
    listener: TcpListener,
    expected_path: String,
    expected_state: String,
    request_timeout: Duration,
    /// Browser origins whose CORS preflight is answered. Empty for every
    /// provider except Grok CLI, whose authorization pages probe the
    /// loopback listener from their own origin before redirecting.
    cors_origins: Vec<String>,
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
            request_timeout: CALLBACK_REQUEST_TIMEOUT,
            cors_origins: Vec::new(),
        })
    }

    /// Answers `OPTIONS` preflights with `204` and grants these origins CORS
    /// access (including Chrome's private-network access) on every reply,
    /// as pi-grok-cli's callback server does for xAI's account pages.
    #[must_use]
    pub fn with_cors_origins(mut self, origins: &[&str]) -> Self {
        self.cors_origins = origins.iter().map(|origin| (*origin).to_owned()).collect();
        self
    }

    /// Bounds how long one connection may take to send its request line.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(|error| {
            OAuthError::Callback(format!("cannot inspect callback address: {error}"))
        })
    }

    /// Accepts and validates at most one pending callback.
    pub fn try_accept(&self, cancellation: &CancellationToken) -> Result<Option<CallbackOutcome>> {
        let (stream, _) = match self.listener.accept() {
            Ok(pair) => pair,
            // A peer that vanished between connect and accept, or a passing
            // resource shortage, is not a reason to abandon the login.
            Err(_) => return Ok(None),
        };
        self.handle_connection(stream, cancellation)
    }

    fn handle_connection(
        &self,
        mut stream: TcpStream,
        cancellation: &CancellationToken,
    ) -> Result<Option<CallbackOutcome>> {
        // macOS and Windows hand out the accepted socket in the listener's
        // non-blocking mode, which would turn the timed reads below into a
        // busy loop and could drop the confirmation page on a full buffer.
        let _ = stream.set_nonblocking(false);
        let (request, origin) =
            match read_callback_request(&mut stream, self.request_timeout, cancellation)? {
                CallbackRead::Request { line, host, origin } => {
                    // A page on another site can reach a loopback port through
                    // DNS rebinding; its requests then carry that site's name.
                    if host
                        .as_deref()
                        .is_some_and(|host| !is_loopback_authority(host))
                    {
                        let _ =
                            write_callback_page(&mut stream, 400, "Unexpected Host header.", &[]);
                        return Ok(None);
                    }
                    (line, origin)
                }
                CallbackRead::Reject { status, message } => {
                    let _ = write_callback_page(&mut stream, status, message, &[]);
                    return Ok(None);
                }
            };
        let cors = origin
            .filter(|origin| self.cors_origins.iter().any(|allowed| allowed == origin))
            .map(|origin| cors_headers(&origin))
            .unwrap_or_default();
        let mut fields = request.split_whitespace();
        let method = fields.next().unwrap_or_default();
        let target = fields.next().unwrap_or_default();
        if method == "OPTIONS" && !self.cors_origins.is_empty() {
            let _ = write_empty_response(&mut stream, 204, &cors);
            return Ok(None);
        }
        if method != "GET" {
            let _ =
                write_callback_page(&mut stream, 405, "Only GET callbacks are supported.", &cors);
            return Ok(None);
        }

        let Ok(parsed) = Url::parse(&format!("http://localhost{target}")) else {
            let _ = write_callback_page(
                &mut stream,
                400,
                "Callback request target is invalid.",
                &cors,
            );
            return Ok(None);
        };
        if parsed.path() != self.expected_path {
            let _ = write_callback_page(&mut stream, 404, "Callback route not found.", &cors);
            return Ok(None);
        }
        let state = query_value(&parsed, "state");
        // The state is checked before anything else is believed, so a forged
        // request can neither complete nor abort someone else's login.
        if !self.expected_state.is_empty() && !constant_time_eq(&state, &self.expected_state) {
            let _ = write_callback_page(&mut stream, 400, "State mismatch.", &cors);
            return Ok(None);
        }
        let error = query_value(&parsed, "error");
        if !error.is_empty() {
            let description = query_value(&parsed, "error_description");
            let reason = if description.is_empty() {
                error
            } else {
                description
            };
            let _ = write_callback_page(
                &mut stream,
                400,
                &format!("Sign-in did not complete: {reason}"),
                &cors,
            );
            return Ok(Some(CallbackOutcome::Denied(reason)));
        }
        let code = query_value(&parsed, "code");
        if code.is_empty() {
            let _ = write_callback_page(&mut stream, 400, "Missing authorization code.", &cors);
            return Ok(None);
        }
        // The browser waits on this connection while the code is exchanged,
        // so its page reports how the sign-in actually ended.
        Ok(Some(CallbackOutcome::Authorized(PendingCallback {
            response: AuthorizationResponse { code, state },
            stream,
            cors,
        })))
    }
}

/// What one valid callback request carried.
pub enum CallbackOutcome {
    /// The provider redirected with a code; the browser is still waiting for
    /// its page.
    Authorized(PendingCallback),
    /// The provider redirected with an error, already shown in the browser.
    Denied(String),
}

/// An authorization code whose browser connection has not been answered yet.
/// Dropping it closes the connection without a page.
pub struct PendingCallback {
    pub response: AuthorizationResponse,
    stream: TcpStream,
    cors: Vec<(String, String)>,
}

impl PendingCallback {
    /// Answers the browser with the outcome of the code exchange.
    pub fn finish<T>(mut self, provider: &str, outcome: &Result<T>) {
        let _ = match outcome {
            Ok(_) => write_callback_page(
                &mut self.stream,
                200,
                &format!(
                    "Signed in to {provider}. You can close this window and return to GoshCoder."
                ),
                &self.cors,
            ),
            // Every error already names its provider.
            Err(error) => {
                write_callback_page(&mut self.stream, 502, &error.to_string(), &self.cors)
            }
        };
    }
}

/// Whether a `Host` header value names this machine, with or without a port.
fn is_loopback_authority(authority: &str) -> bool {
    let host = match authority.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or_default(),
        None => authority
            .rsplit_once(':')
            .map_or(authority, |(host, _)| host),
    };
    is_loopback_host(host)
}

fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false)
}

/// Outcome of reading one callback connection's request line. A rejection
/// carries the response the caller sends, so exactly one response is written
/// per connection.
enum CallbackRead {
    Request {
        line: String,
        /// The `Host` header, checked against DNS rebinding.
        host: Option<String>,
        /// The `Origin` header, which only a CORS-enabled server reads.
        origin: Option<String>,
    },
    Reject {
        status: u16,
        message: &'static str,
    },
}

fn read_callback_request(
    stream: &mut TcpStream,
    timeout: Duration,
    cancellation: &CancellationToken,
) -> Result<CallbackRead> {
    if stream
        .set_read_timeout(Some(Duration::from_millis(100)))
        .is_err()
    {
        return Ok(CallbackRead::Reject {
            status: 500,
            message: "Callback socket could not be configured.",
        });
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .unwrap_or_else(Instant::now);
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 1024];
    loop {
        cancellation.check()?;
        // Checked on every pass, not only on idle reads: a client trickling a
        // byte at a time would otherwise keep the login's accept loop busy for
        // as long as it liked.
        if Instant::now() >= deadline {
            return Ok(CallbackRead::Reject {
                status: 408,
                message: "Callback request timed out.",
            });
        }
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => {
                bytes.extend_from_slice(&buffer[..read]);
                if bytes.len() > CALLBACK_REQUEST_LIMIT {
                    return Ok(CallbackRead::Reject {
                        status: 431,
                        message: "Callback request headers are too large.",
                    });
                }
                if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) => {}
            Err(_) => {
                return Ok(CallbackRead::Reject {
                    status: 400,
                    message: "Callback request could not be read.",
                });
            }
        }
    }
    let Ok(request) = String::from_utf8(bytes) else {
        return Ok(CallbackRead::Reject {
            status: 400,
            message: "Callback request was not UTF-8.",
        });
    };
    let origin = request
        .lines()
        .skip(1)
        .take_while(|line| !line.is_empty())
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("origin")
                .then(|| value.trim().to_owned())
        });
    match request
        .lines()
        .next()
        .filter(|line| !line.trim().is_empty())
    {
        Some(line) => Ok(CallbackRead::Request {
            line: line.to_owned(),
            host: request.lines().skip(1).find_map(|header| {
                let (name, value) = header.split_once(':')?;
                name.trim()
                    .eq_ignore_ascii_case("host")
                    .then(|| value.trim().to_owned())
            }),
            origin,
        }),
        None => Ok(CallbackRead::Reject {
            status: 400,
            message: "Callback request was empty.",
        }),
    }
}

/// pi-grok-cli's CORS grant for a trusted xAI origin.
fn cors_headers(origin: &str) -> Vec<(String, String)> {
    [
        ("Access-Control-Allow-Origin", origin),
        ("Access-Control-Allow-Methods", "GET, OPTIONS"),
        ("Access-Control-Allow-Headers", "Content-Type"),
        ("Access-Control-Allow-Private-Network", "true"),
        ("Vary", "Origin"),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value.to_owned()))
    .collect()
}

fn write_header_lines(headers: &[(String, String)]) -> String {
    headers
        .iter()
        .map(|(name, value)| format!("{name}: {value}\r\n"))
        .collect()
}

fn write_empty_response(
    stream: &mut TcpStream,
    status: u16,
    headers: &[(String, String)],
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} No Content\r\n{}Content-Length: 0\r\nConnection: close\r\n\r\n",
        write_header_lines(headers)
    )?;
    stream.flush()
}

fn write_callback_page(
    stream: &mut TcpStream,
    status: u16,
    message: &str,
    headers: &[(String, String)],
) -> io::Result<()> {
    let escaped = escape_html(message);
    let (heading, accent) = if status < 400 {
        ("You're signed in", "#16a34a")
    } else {
        ("Sign-in did not finish", "#dc2626")
    };
    let body = format!(
        "<!doctype html><html lang=en><meta charset=utf-8>\
         <meta name=viewport content=\"width=device-width,initial-scale=1\">\
         <title>GoshCoder sign-in</title><style>\
         :root{{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--fg:#1f2328;--muted:#59636e}}\
         @media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--muted:#9198a1}}}}\
         body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);\
         color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}\
         main{{max-width:30rem;margin:1rem;padding:2rem 2.25rem;background:var(--card);\
         border-radius:12px;border-top:4px solid {accent};box-shadow:0 1px 3px #0002}}\
         .brand{{font-size:.75rem;letter-spacing:.12em;font-weight:700;color:var(--muted)}}\
         h1{{font-size:1.35rem;margin:.4rem 0 .6rem}}p{{margin:0;color:var(--muted);overflow-wrap:anywhere}}\
         </style><main><div class=brand>GOSHCODER</div><h1>{heading}</h1><p>{escaped}</p></main></html>"
    );
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        431 => "Request Header Fields Too Large",
        502 => "Bad Gateway",
        status if status >= 500 => "Internal Server Error",
        _ => "Bad Request",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/html; charset=utf-8\r\n\
         Cache-Control: no-store\r\nContent-Security-Policy: default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'\r\n\
         {}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        write_header_lines(headers),
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
    /// Names the provider on the browser page and in errors.
    pub provider_name: &'static str,
    pub authorization_url: Url,
    pub redirect_uri: String,
    pub expected_state: String,
    pub callback_host: String,
    pub callback_port: u16,
    pub callback_path: String,
    /// How a pasted redirect or code is judged.
    pub manual_input: ManualInputPolicy,
}

/// How [`run_loopback_login`] treats what the user pastes instead of
/// letting the browser reach the loopback listener.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ManualInputPolicy {
    /// Any code is accepted; a state, when one is pasted, must match.
    #[default]
    Lenient,
    /// pi-grok-cli's rule: a pasted callback URL or query must carry the
    /// matching state, and a bare value is accepted only when it looks like
    /// xAI's one-time code (`^[A-Za-z0-9._~-]{32,2048}$`). Anything else is
    /// reported and ignored while the browser callback keeps waiting.
    StateOrOneTimeCode,
}

/// What a strictly judged paste carried.
#[derive(Debug, Eq, PartialEq)]
pub enum ManualCallback {
    Code(String),
    Denied(String),
}

fn is_one_time_code(value: &str) -> bool {
    (32..=2048).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'~' | b'-'))
}

fn strict_callback_params<'a>(
    pairs: impl Iterator<Item = (std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>)>,
    expected_state: &str,
) -> std::result::Result<ManualCallback, String> {
    let mut fields = BTreeMap::new();
    for (name, value) in pairs {
        fields
            .entry(name.into_owned())
            .or_insert_with(|| value.into_owned());
    }
    let state = fields.get("state").filter(|state| !state.is_empty());
    let Some(state) = state else {
        return Err("OAuth state is missing.".to_owned());
    };
    if !constant_time_eq(state, expected_state) {
        return Err("OAuth state did not match.".to_owned());
    }
    let non_empty = |name: &str| fields.get(name).filter(|value| !value.is_empty()).cloned();
    if let Some(error) = non_empty("error") {
        return Ok(ManualCallback::Denied(
            non_empty("error_description").unwrap_or(error),
        ));
    }
    non_empty("code")
        .map(ManualCallback::Code)
        .ok_or_else(|| "Callback did not include an authorization code or OAuth error.".to_owned())
}

/// pi-grok-cli `parseManualCallback`.
pub fn parse_strict_manual_input(
    input: &str,
    expected_state: &str,
    callback_path: &str,
) -> std::result::Result<ManualCallback, String> {
    let value = input.trim();
    if value.is_empty() {
        return Err("Pasted callback was empty.".to_owned());
    }
    if let Ok(url) = Url::parse(value) {
        if url.path() != callback_path {
            return Err("Callback URL path was not recognized.".to_owned());
        }
        return strict_callback_params(url.query_pairs(), expected_state);
    }
    if !value.contains('=') && is_one_time_code(value) {
        return Ok(ManualCallback::Code(value.to_owned()));
    }
    strict_callback_params(
        url::form_urlencoded::parse(value.trim_start_matches('?').as_bytes()),
        expected_state,
    )
}

fn spawn_manual_prompt(
    interaction: &Arc<dyn OAuthInteraction>,
    prompt: OAuthPrompt,
) -> Result<mpsc::Receiver<Result<String>>> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let prompter = interaction.clone();
    thread::Builder::new()
        .name("oauth-manual-code".to_owned())
        .spawn(move || {
            let result = prompter.prompt(prompt);
            let _ = sender.send(result);
        })
        .map_err(|error| {
            OAuthError::Callback(format!("cannot start manual-code prompt: {error}"))
        })?;
    Ok(receiver)
}

/// Runs a browser authorization flow, racing a loopback callback against a
/// cancellable manual-paste prompt, and finishes it with `complete`, which
/// exchanges the code. As in pi's callback server, the browser is answered
/// only after the exchange, so its page reports failures too.
pub fn run_loopback_login<T>(
    interaction: Arc<dyn OAuthInteraction>,
    browser: Arc<dyn BrowserOpener>,
    cancellation: &CancellationToken,
    request: LoopbackLoginRequest,
    complete: impl FnOnce(String) -> Result<T>,
) -> Result<T> {
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
    run_loopback_login_with_server(
        interaction,
        browser,
        cancellation,
        request,
        server,
        complete,
    )
}

/// [`run_loopback_login`] with a listener the caller already bound, for a
/// provider whose authorization URL names the port the system picked.
pub fn run_loopback_login_with_server<T>(
    interaction: Arc<dyn OAuthInteraction>,
    browser: Arc<dyn BrowserOpener>,
    cancellation: &CancellationToken,
    request: LoopbackLoginRequest,
    server: Option<LoopbackCallbackServer>,
    complete: impl FnOnce(String) -> Result<T>,
) -> Result<T> {
    interaction.notify(OAuthEvent::authorization_url(&request.authorization_url));
    let _ = browser.open(&request.authorization_url);

    let manual_cancellation = CancellationToken::new();
    let _cancel_manual = CancelOnDrop(manual_cancellation.clone());
    let prompt = OAuthPrompt {
        kind: OAuthPromptKind::ManualCode,
        message: "Waiting for the browser. Or paste the redirect URL / authorization code:"
            .to_owned(),
        placeholder: request.redirect_uri.clone(),
        options: Vec::new(),
        cancellation: manual_cancellation,
    };
    // `None` once a strictly judged paste was ignored: like upstream, the
    // prompt is asked once and the browser callback keeps the login alive.
    let mut manual_receiver = Some(spawn_manual_prompt(&interaction, prompt)?);

    let code = loop {
        cancellation.check()?;
        match server
            .as_ref()
            .map(|server| server.try_accept(cancellation))
            .transpose()?
            .flatten()
        {
            Some(CallbackOutcome::Authorized(pending)) => {
                // The manual prompt is pointless now; stop it before the
                // exchange prints its progress over it.
                drop(_cancel_manual);
                interaction.notify(OAuthEvent::progress(
                    "Browser sign-in received; exchanging the authorization code...",
                ));
                let outcome = complete(pending.response.code.clone());
                pending.finish(request.provider_name, &outcome);
                return outcome;
            }
            Some(CallbackOutcome::Denied(reason)) => {
                return Err(OAuthError::Callback(format!(
                    "{} sign-in was not authorized in the browser: {reason}",
                    request.provider_name
                )));
            }
            None => {}
        }
        let manual = match manual_receiver.as_ref().map(mpsc::Receiver::try_recv) {
            None | Some(Err(TryRecvError::Empty)) => None,
            Some(Ok(result)) => Some(result?),
            Some(Err(TryRecvError::Disconnected)) => {
                if server.is_none() {
                    return Err(OAuthError::Cancelled);
                }
                manual_receiver = None;
                None
            }
        };
        let Some(input) = manual else {
            thread::sleep(Duration::from_millis(10));
            continue;
        };
        // An empty line, or a stdin that is not a terminal, must not abandon
        // a browser sign-in that is still on its way back.
        if input.trim().is_empty() && server.is_some() {
            manual_receiver = None;
            interaction.notify(OAuthEvent::info(
                "Still waiting for the browser sign-in; press Ctrl-C to cancel.",
            ));
            continue;
        }
        if request.manual_input == ManualInputPolicy::StateOrOneTimeCode {
            match parse_strict_manual_input(&input, &request.expected_state, &request.callback_path)
            {
                Ok(ManualCallback::Code(code)) => break code,
                Ok(ManualCallback::Denied(reason)) => {
                    return Err(OAuthError::Callback(format!(
                        "{} sign-in was not authorized in the browser: {reason}",
                        request.provider_name
                    )));
                }
                // A stray paste must not end a login the browser can still
                // complete. Without a listener nothing else can.
                Err(reason) => {
                    if server.is_none() {
                        return Err(OAuthError::Callback(format!(
                            "the pasted callback was not accepted: {reason}"
                        )));
                    }
                    interaction.notify(OAuthEvent::progress(format!(
                        "Ignored pasted callback: {reason} Paste the complete callback URL or xAI's one-time code."
                    )));
                    manual_receiver = None;
                    continue;
                }
            }
        }
        if let Some(reason) = redirect_error(&input) {
            return Err(OAuthError::Callback(format!(
                "{} sign-in was not authorized in the browser: {reason}",
                request.provider_name
            )));
        }
        let parsed =
            parse_authorization_input(&input).ok_or(OAuthError::InvalidAuthorizationInput)?;
        if !parsed.state.is_empty()
            && !request.expected_state.is_empty()
            && !constant_time_eq(&parsed.state, &request.expected_state)
        {
            return Err(OAuthError::StateMismatch);
        }
        break parsed.code;
    };
    interaction.notify(OAuthEvent::progress(
        "Exchanging the authorization code for tokens...",
    ));
    complete(code)
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
    /// Builds a request for a flow implemented outside this module.
    pub(crate) fn new(
        method: Method,
        url: Url,
        headers: BTreeMap<String, String>,
        body: Vec<u8>,
        timeout: Duration,
    ) -> Self {
        Self {
            method,
            url,
            headers,
            body,
            timeout,
        }
    }

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
            .user_agent(OAUTH_USER_AGENT)
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
            .timeout(bounded_request_timeout(request.timeout, cancellation))
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

/// A blocking request cannot be interrupted mid-flight, so its own timeout is
/// the only way a token deadline can cut it short.
fn bounded_request_timeout(timeout: Duration, cancellation: &CancellationToken) -> Duration {
    cancellation
        .remaining()
        .map_or(timeout, |remaining| remaining.min(timeout))
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
}

fn token_error(
    provider: &'static str,
    operation: &'static str,
    response: &OAuthResponse,
) -> OAuthError {
    let detail = truncate_response(&response.body, 500);
    let code = error_code(&response.body);
    if response.status == 401
        || response.status == 403
        || matches!(
            code.as_deref(),
            Some("invalid_grant" | "invalid_refresh_token" | "token_expired")
        )
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

/// The human-readable part of an error body. Providers disagree on the
/// shape (`error_description`, `{"error":{"message"}}`, `message`, a bare
/// `error` string), so they are tried in the order pi's `errorDetail` uses,
/// and anything else falls back to the raw text.
fn error_detail(body: &[u8]) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    let text = |value: Option<&Value>| {
        value
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    text(value.get("error_description"))
        .or_else(|| text(value.get("error").and_then(|error| error.get("message"))))
        .or_else(|| text(value.get("message")))
        .or_else(|| text(value.get("error")))
}

/// The machine-readable error code, top-level or nested under `error`.
fn error_code(body: &[u8]) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    let error = value.get("error")?;
    error
        .as_str()
        .or_else(|| error.get("code").and_then(Value::as_str))
        .or_else(|| error.get("type").and_then(Value::as_str))
        .map(str::to_owned)
}

fn truncate_response(body: &[u8], limit: usize) -> String {
    let detail = error_detail(body);
    let body = match &detail {
        Some(detail) => detail.clone(),
        None => String::from_utf8_lossy(body).into_owned(),
    };
    // A multi-line HTML or JSON dump reads as noise on one terminal line.
    let body = body.split_whitespace().collect::<Vec<_>>().join(" ");
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

pub(crate) fn endpoint(base: &Url, path: &str) -> Result<Url> {
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
    pub openrouter_authorize_url: Url,
    pub openrouter_key_url: Url,
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
            openrouter_authorize_url: fixed_url("https://openrouter.ai/auth"),
            openrouter_key_url: fixed_url("https://openrouter.ai/api/v1/auth/keys"),
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

    /// Reports whether a stored credential is inside its refresh window
    /// according to this client's clock.
    pub fn credential_needs_refresh(&self, credential: &Credential) -> bool {
        credential_expires_soon(credential, self.clock.as_ref())
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
            OAuthProviderId::MetaMuse => self
                .meta_muse_flow()
                .login(interaction.as_ref(), cancellation),
            OAuthProviderId::OpenAiCodex => {
                self.login_codex(interaction, environment, cancellation)
            }
            OAuthProviderId::Xai => self.login_xai(interaction, environment, cancellation),
            OAuthProviderId::GrokCli => self.login_grok_cli(interaction, environment, cancellation),
            OAuthProviderId::OpenRouter => {
                self.login_openrouter(interaction, environment, cancellation)
            }
            OAuthProviderId::GithubCopilot | OAuthProviderId::Radius => {
                Err(OAuthError::UnsupportedFlow {
                    provider: provider.as_str().to_owned(),
                })
            }
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
            OAuthProviderId::MetaMuse => self.meta_muse_flow().refresh(current, cancellation),
            OAuthProviderId::OpenAiCodex => self.refresh_codex(current, cancellation),
            OAuthProviderId::Xai => self.refresh_xai(current, environment, cancellation),
            OAuthProviderId::GrokCli => self.refresh_grok_cli(current, environment, cancellation),
            // The minted key does not expire; pi's refresh returns it as is.
            OAuthProviderId::OpenRouter => Ok(current.clone()),
            OAuthProviderId::GithubCopilot | OAuthProviderId::Radius => {
                Err(OAuthError::UnsupportedFlow {
                    provider: provider.as_str().to_owned(),
                })
            }
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
                    ("User-Agent".to_owned(), OAUTH_USER_AGENT.to_owned()),
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
                    ("User-Agent".to_owned(), OAUTH_USER_AGENT.to_owned()),
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
        headers
            .entry("User-Agent".to_owned())
            .or_insert_with(|| OAUTH_USER_AGENT.to_owned());
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
                headers: BTreeMap::from([
                    ("Accept".to_owned(), "application/json".to_owned()),
                    ("User-Agent".to_owned(), OAUTH_USER_AGENT.to_owned()),
                ]),
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
    pub fn anthropic_authorization_url(&self, pkce: &PkcePair, redirect_uri: &str) -> Url {
        append_query(
            &self.endpoints.anthropic_authorize_url,
            &[
                ("code", "true"),
                ("client_id", ANTHROPIC_CLIENT_ID),
                ("response_type", "code"),
                ("redirect_uri", redirect_uri),
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
        let method = interaction.prompt(OAuthPrompt {
            kind: OAuthPromptKind::Select,
            message: "Select the Anthropic login method:".to_owned(),
            placeholder: String::new(),
            options: vec![
                OAuthPromptOption {
                    id: "browser".to_owned(),
                    label: "Browser login (default)".to_owned(),
                    description: String::new(),
                },
                OAuthPromptOption {
                    id: "copy_code".to_owned(),
                    label: "Copy code login (headless)".to_owned(),
                    description: "for a machine whose browser cannot reach this one".to_owned(),
                },
            ],
            cancellation: cancellation.clone(),
        })?;
        cancellation.check()?;
        match method.as_str() {
            "" | "browser" => self.login_anthropic_browser(interaction, environment, cancellation),
            "copy_code" => self.login_anthropic_copy_code(interaction, cancellation),
            _ => Err(OAuthError::InvalidConfiguration(format!(
                "unknown Anthropic login method {method:?}"
            ))),
        }
    }

    /// pi's `loginAnthropicCopyCode`: Anthropic's own page shows the code,
    /// so a login on a remote or headless machine needs no loopback at all.
    fn login_anthropic_copy_code(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let pkce = generate_pkce();
        interaction.notify(OAuthEvent {
            instructions:
                "Complete login in your browser, then copy the code Anthropic shows and paste it here."
                    .to_owned(),
            ..OAuthEvent::authorization_url(
                &self.anthropic_authorization_url(&pkce, ANTHROPIC_COPY_CODE_REDIRECT_URI),
            )
        });
        let _ = self
            .browser
            .open(&self.anthropic_authorization_url(&pkce, ANTHROPIC_COPY_CODE_REDIRECT_URI));
        let input = interaction.prompt(OAuthPrompt {
            kind: OAuthPromptKind::ManualCode,
            message: "Paste the code Anthropic shows after you sign in:".to_owned(),
            placeholder: "code#state".to_owned(),
            options: Vec::new(),
            cancellation: cancellation.clone(),
        })?;
        let parsed =
            parse_authorization_input(&input).ok_or(OAuthError::InvalidAuthorizationInput)?;
        if !parsed.state.is_empty() && !constant_time_eq(&parsed.state, pkce.verifier()) {
            return Err(OAuthError::StateMismatch);
        }
        interaction.notify(OAuthEvent::progress(
            "Exchanging the authorization code for tokens...",
        ));
        self.exchange_anthropic_code(
            &parsed.code,
            pkce.verifier(),
            ANTHROPIC_COPY_CODE_REDIRECT_URI,
            cancellation,
        )
    }

    fn exchange_anthropic_code(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let response = self.post_json(
            &self.endpoints.anthropic_token_url,
            json!({
                "grant_type": "authorization_code",
                "client_id": ANTHROPIC_CLIENT_ID,
                "code": code,
                "state": verifier,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
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

    fn login_anthropic_browser(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let pkce = generate_pkce();
        let host = callback_host(environment);
        // pi's anthropic.ts: Anthropic accepts any loopback port, so a busy
        // 53692 falls back to a free one rather than sending the browser's
        // code to whatever holds the port.
        let server = LoopbackCallbackServer::bind(
            &host,
            ANTHROPIC_CALLBACK_PORT,
            ANTHROPIC_CALLBACK_PATH,
            pkce.verifier(),
        )
        .or_else(|_| {
            LoopbackCallbackServer::bind(&host, 0, ANTHROPIC_CALLBACK_PATH, pkce.verifier())
        });
        let redirect_uri = match server.as_ref().map(LoopbackCallbackServer::local_addr) {
            Ok(Ok(address)) if address.port() != ANTHROPIC_CALLBACK_PORT => {
                format!(
                    "http://localhost:{}{ANTHROPIC_CALLBACK_PATH}",
                    address.port()
                )
            }
            _ => ANTHROPIC_REDIRECT_URI.to_owned(),
        };
        let server = match server {
            Ok(server) => Some(server),
            Err(error) => {
                interaction.notify(OAuthEvent::info(format!(
                    "Could not listen for the browser callback ({error}). Complete login in the browser and paste the redirect URL below."
                )));
                None
            }
        };
        let authorization_url = self.anthropic_authorization_url(&pkce, &redirect_uri);
        let exchange_redirect_uri = redirect_uri.clone();
        run_loopback_login_with_server(
            interaction,
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                provider_name: OAuthProviderId::Anthropic.display_name(),
                authorization_url,
                redirect_uri,
                expected_state: pkce.verifier().to_owned(),
                callback_host: host,
                callback_port: ANTHROPIC_CALLBACK_PORT,
                callback_path: ANTHROPIC_CALLBACK_PATH.to_owned(),
                manual_input: ManualInputPolicy::Lenient,
            },
            server,
            |code| {
                self.exchange_anthropic_code(
                    &code,
                    pkce.verifier(),
                    &exchange_redirect_uri,
                    cancellation,
                )
            },
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

/// pi stores OpenRouter's permanent key with `Number.MAX_SAFE_INTEGER` as its
/// expiry; the same value keeps `auth.json` interchangeable.
const OPENROUTER_KEY_EXPIRES_MS: i64 = 9_007_199_254_740_991;

impl OAuthClient {
    fn login_openrouter(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let pkce = generate_pkce();
        // OpenRouter sends no `state`; the random path keeps stray requests
        // from completing the sign-in, and the port is whatever is free.
        let path = format!("/oauth/callback/{}", random_state());
        let host = callback_host(environment);
        let server = LoopbackCallbackServer::bind(&host, 0, &path, "")?;
        let port = server.local_addr()?.port();
        let redirect_host = if host.contains(':') {
            format!("[{host}]")
        } else {
            host.clone()
        };
        let redirect_uri = format!("http://{redirect_host}:{port}{path}");
        let authorization_url = append_query(
            &self.endpoints.openrouter_authorize_url,
            &[
                ("callback_url", redirect_uri.as_str()),
                ("code_challenge", pkce.challenge()),
                ("code_challenge_method", "S256"),
            ],
        );
        run_loopback_login_with_server(
            interaction,
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                provider_name: OAuthProviderId::OpenRouter.display_name(),
                authorization_url,
                redirect_uri,
                expected_state: String::new(),
                callback_host: host,
                callback_port: port,
                callback_path: path,
                manual_input: ManualInputPolicy::Lenient,
            },
            Some(server),
            |code| self.exchange_openrouter_code(&code, pkce.verifier(), cancellation),
        )
    }

    fn exchange_openrouter_code(
        &self,
        code: &str,
        verifier: &str,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let provider = OAuthProviderId::OpenRouter.display_name();
        let response = self.post_json(
            &self.endpoints.openrouter_key_url,
            json!({
                "code": code,
                "code_verifier": verifier,
                "code_challenge_method": "S256",
            }),
            cancellation,
        )?;
        if !(200..300).contains(&response.status) {
            return Err(operation_failure(provider, "exchange", &response));
        }
        let key = serde_json::from_slice::<Value>(&response.body)
            .ok()
            .and_then(|body| body.get("key").and_then(Value::as_str).map(str::to_owned))
            .filter(|key| !key.is_empty())
            .ok_or(OAuthError::InvalidTokenResponse {
                provider,
                operation: "exchange",
            })?;
        Ok(Credential::oauth(key, "", OPENROUTER_KEY_EXPIRES_MS))
    }

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
        run_loopback_login(
            interaction,
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                provider_name: OAuthProviderId::OpenAiCodex.display_name(),
                authorization_url: self.codex_authorization_url(&pkce, &state),
                redirect_uri: CODEX_REDIRECT_URI.to_owned(),
                expected_state: state,
                callback_host: callback_host(environment),
                callback_port: CODEX_CALLBACK_PORT,
                callback_path: CODEX_CALLBACK_PATH.to_owned(),
                manual_input: ManualInputPolicy::Lenient,
            },
            |code| {
                self.exchange_codex_code(&code, pkce.verifier(), CODEX_REDIRECT_URI, cancellation)
            },
        )
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
            device.interval,
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
            DevicePollingPolicy::new(device.interval, DEFAULT_DEVICE_TIMEOUT),
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
                if response.status == 403 || response.status == 404 {
                    return Ok(DevicePoll::Pending);
                }
                match codex_device_error_code(&response.body).as_deref() {
                    Some("deviceauth_authorization_pending" | "authorization_pending") => {
                        Ok(DevicePoll::Pending)
                    }
                    // RFC 8628 section 3.5: back off instead of giving up.
                    Some("slow_down") => Ok(DevicePoll::SlowDown(None)),
                    _ => Err(operation_failure(
                        OAuthProviderId::OpenAiCodex.display_name(),
                        "device token request",
                        &response,
                    )),
                }
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
    interval: Duration,
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
    // The server reports the interval as a number or a numeric string, and a
    // fractional value is a real interval rather than a malformed one.
    let interval_seconds = value
        .get("interval")
        .and_then(json_seconds)
        .filter(|seconds| *seconds >= 0.0)
        .ok_or(OAuthError::InvalidTokenResponse {
            provider: OAuthProviderId::OpenAiCodex.display_name(),
            operation: "device code request",
        })?;
    Ok(CodexDeviceAuthorization {
        device_auth_id,
        user_code,
        interval: reported_device_interval(interval_seconds, DEFAULT_DEVICE_INTERVAL),
    })
}

fn json_string(value: &Value, name: &str) -> Option<String> {
    value
        .get(name)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn json_seconds(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str()?.trim().parse::<f64>().ok())
        .filter(|seconds| seconds.is_finite())
}

/// Codex reports device-poll errors either as `{"error": "code"}` or as
/// `{"error": {"code": "code"}}`.
fn codex_device_error_code(body: &[u8]) -> Option<String> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    match value.get("error")? {
        Value::String(error) => Some(error.clone()),
        Value::Object(error) => error.get("code").and_then(Value::as_str).map(str::to_owned),
        _ => None,
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
        let authorization_url =
            self.xai_authorization_url(&endpoints, &client_id, &pkce, &state, &nonce);
        run_loopback_login(
            interaction,
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                provider_name: OAuthProviderId::Xai.display_name(),
                authorization_url,
                redirect_uri: XAI_REDIRECT_URI.to_owned(),
                expected_state: state,
                callback_host: callback_host(environment),
                callback_port: XAI_CALLBACK_PORT,
                callback_path: XAI_CALLBACK_PATH.to_owned(),
                manual_input: ManualInputPolicy::Lenient,
            },
            |code| {
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
            },
        )
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

// ---------------------------------------------------------------------------
// Grok CLI (pi-grok-cli v0.9.3 `src/auth/oauth.ts` and `src/auth/config.ts`)
//
// The same auth.x.ai issuer and discovery as xAI above, with the official
// Grok CLI's parameters: its own callback port with an ephemeral fallback,
// `plan=generic` on the authorization URL, a PKCE exchange that sends only
// the verifier, a CORS-answering callback listener, a stricter manual paste,
// and 400/401/403 refresh failures treated as a revoked login.

const GROK_CLI_DEFAULT_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
const GROK_CLI_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
const GROK_CLI_CALLBACK_PORT: u16 = 56122;
const GROK_CLI_CALLBACK_PATH: &str = "/callback";
/// Stored expiries run this far ahead of the token's real one, as
/// upstream's `REFRESH_SKEW_MS` makes them.
const GROK_CLI_REFRESH_SKEW: Duration = Duration::from_secs(120);
const GROK_CLI_DEVICE_TIMEOUT: Duration = Duration::from_secs(1800);
/// xAI's account pages probe the loopback listener from these origins.
pub const GROK_CLI_CORS_ORIGINS: &[&str] = &["https://accounts.x.ai", "https://auth.x.ai"];
const GROK_CLI_TOKEN_ENV_MESSAGE: &str =
    "Unset GROK_CLI_OAUTH_TOKEN before logging in to Grok CLI.";

fn grok_cli_client_id(environment: &dyn OAuthEnvironment) -> String {
    environment
        .value("PI_GROK_CLI_OAUTH_CLIENT_ID")
        .unwrap_or_else(|| GROK_CLI_DEFAULT_CLIENT_ID.to_owned())
}

fn grok_cli_scope(environment: &dyn OAuthEnvironment) -> String {
    environment
        .value("PI_GROK_CLI_OAUTH_SCOPE")
        .unwrap_or_else(|| GROK_CLI_SCOPE.to_owned())
}

fn grok_cli_callback_host(environment: &dyn OAuthEnvironment) -> String {
    environment
        .value("PI_GROK_CLI_CALLBACK_HOST")
        .unwrap_or_else(|| callback_host(environment))
}

fn grok_cli_callback_port(environment: &dyn OAuthEnvironment) -> u16 {
    environment
        .value("PI_GROK_CLI_CALLBACK_PORT")
        .and_then(|port| port.trim().parse().ok())
        .unwrap_or(GROK_CLI_CALLBACK_PORT)
}

/// upstream `validateEndpoint`: HTTPS on x.ai or a subdomain. An endpoint
/// on the configured issuer's own origin is accepted too, so a test issuer
/// on loopback works the way the xAI flow's pinning allows.
fn grok_cli_trusted_endpoint(issuer: &Url, value: &str) -> Option<Url> {
    if let Some(pinned) = pin_xai_endpoint(issuer, value) {
        return Some(pinned);
    }
    let url = Url::parse(value).ok()?;
    let host = url.host_str()?.to_ascii_lowercase();
    (url.scheme() == "https" && (host == "x.ai" || host.ends_with(".x.ai"))).then_some(url)
}

/// `String(payload[field])` for the scalar shapes a token endpoint returns.
fn grok_cli_token_field(payload: &Value, field: &str) -> Option<String> {
    match payload.get(field)? {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn grok_cli_discovery(endpoints: &XaiEndpoints) -> Value {
    json!({
        "authorization_endpoint": endpoints.authorize.as_str(),
        "token_endpoint": endpoints.token.as_str(),
        "device_authorization_endpoint": endpoints.device.as_str(),
    })
}

fn grok_cli_base_url(environment: &dyn OAuthEnvironment) -> String {
    crate::grok_cli::base_url(|name| environment.value(name))
}

impl OAuthClient {
    /// Builds the Grok CLI browser authorization URL. `referrer` names
    /// GoshCoder (upstream names its own package), as the xAI flow does.
    #[allow(clippy::too_many_arguments)]
    pub fn grok_cli_authorization_url(
        &self,
        endpoints: &XaiEndpoints,
        client_id: &str,
        scope: &str,
        redirect_uri: &str,
        pkce: &PkcePair,
        state: &str,
        nonce: &str,
    ) -> Url {
        append_query(
            &endpoints.authorize,
            &[
                ("response_type", "code"),
                ("client_id", client_id),
                ("redirect_uri", redirect_uri),
                ("scope", scope),
                ("code_challenge", pkce.challenge()),
                ("code_challenge_method", "S256"),
                ("state", state),
                ("nonce", nonce),
                ("plan", "generic"),
                ("referrer", "goshcoder"),
            ],
        )
    }

    fn login_grok_cli(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        // The environment token would shadow whatever this login stores.
        if environment.value(crate::grok_cli::TOKEN_ENV).is_some() {
            return Err(OAuthError::InvalidConfiguration(
                GROK_CLI_TOKEN_ENV_MESSAGE.to_owned(),
            ));
        }
        let endpoints = self.discover_xai_endpoints(cancellation)?;
        let method = interaction.prompt(OAuthPrompt {
            kind: OAuthPromptKind::Select,
            message: "Select Grok CLI login method:".to_owned(),
            placeholder: String::new(),
            options: vec![
                OAuthPromptOption {
                    id: "browser".to_owned(),
                    label: "Browser login (default)".to_owned(),
                    description: "Opens accounts.x.ai and waits on a loopback callback.".to_owned(),
                },
                OAuthPromptOption {
                    id: "device".to_owned(),
                    label: "Device code login (headless)".to_owned(),
                    description: "Shows a code to enter at accounts.x.ai.".to_owned(),
                },
            ],
            cancellation: cancellation.clone(),
        })?;
        cancellation.check()?;
        match method.as_str() {
            "" | "browser" => {
                self.login_grok_cli_browser(interaction, environment, &endpoints, cancellation)
            }
            "device" | "device_code" => {
                self.login_grok_cli_device(interaction, environment, &endpoints, cancellation)
            }
            _ => Err(OAuthError::InvalidConfiguration(format!(
                "unknown Grok CLI login method {method:?}"
            ))),
        }
    }

    fn login_grok_cli_browser(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        endpoints: &XaiEndpoints,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let client_id = grok_cli_client_id(environment);
        let scope = grok_cli_scope(environment);
        let host = grok_cli_callback_host(environment);
        let port = grok_cli_callback_port(environment);
        let pkce = generate_pkce();
        let state = random_state();
        let nonce = random_state();
        // The registered port first; when another login holds it, any free
        // port, since the redirect URI names whichever one was bound.
        let bound = LoopbackCallbackServer::bind(&host, port, GROK_CLI_CALLBACK_PATH, &state)
            .or_else(|first| {
                LoopbackCallbackServer::bind(&host, 0, GROK_CLI_CALLBACK_PATH, &state).map_err(
                    |second| {
                        OAuthError::Callback(format!(
                            "could not bind {host}:{port} or an ephemeral port: {second} (initial error: {first})"
                        ))
                    },
                )
            });
        let (server, bound_port) = match bound {
            Ok(server) => {
                let bound_port = server.local_addr()?.port();
                (
                    Some(server.with_cors_origins(GROK_CLI_CORS_ORIGINS)),
                    bound_port,
                )
            }
            Err(error) => {
                interaction.notify(OAuthEvent::info(format!(
                    "{error}. Complete login in the browser and paste the callback URL or one-time code below."
                )));
                (None, port)
            }
        };
        let redirect_host = if host.contains(':') {
            format!("[{host}]")
        } else {
            host.clone()
        };
        let redirect_uri = format!("http://{redirect_host}:{bound_port}{GROK_CLI_CALLBACK_PATH}");
        let authorization_url = self.grok_cli_authorization_url(
            endpoints,
            &client_id,
            &scope,
            &redirect_uri,
            &pkce,
            &state,
            &nonce,
        );
        run_loopback_login_with_server(
            interaction,
            self.browser.clone(),
            cancellation,
            LoopbackLoginRequest {
                provider_name: OAuthProviderId::GrokCli.display_name(),
                authorization_url,
                redirect_uri: redirect_uri.clone(),
                expected_state: state,
                callback_host: host,
                callback_port: bound_port,
                callback_path: GROK_CLI_CALLBACK_PATH.to_owned(),
                manual_input: ManualInputPolicy::StateOrOneTimeCode,
            },
            server,
            |code| {
                let response = self.post_form(
                    &endpoints.token,
                    fields([
                        ("grant_type", "authorization_code".to_owned()),
                        ("client_id", client_id.clone()),
                        ("code", code),
                        ("redirect_uri", redirect_uri.clone()),
                        ("code_verifier", pkce.verifier().to_owned()),
                    ]),
                    cancellation,
                )?;
                let payload = self.grok_cli_payload("exchange", response)?;
                self.grok_cli_login_credential("exchange", &payload, endpoints, environment)
            },
        )
    }

    fn login_grok_cli_device(
        &self,
        interaction: Arc<dyn OAuthInteraction>,
        environment: &dyn OAuthEnvironment,
        endpoints: &XaiEndpoints,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let provider = OAuthProviderId::GrokCli.display_name();
        let client_id = grok_cli_client_id(environment);
        let response = self.post_form(
            &endpoints.device,
            fields([
                ("client_id", client_id.clone()),
                ("scope", grok_cli_scope(environment)),
            ]),
            cancellation,
        )?;
        if !(200..300).contains(&response.status) {
            return Err(operation_failure(
                provider,
                "device authorization",
                &response,
            ));
        }
        let invalid = || OAuthError::InvalidTokenResponse {
            provider,
            operation: "device authorization",
        };
        let device: DeviceAuthorization =
            serde_json::from_slice(&response.body).map_err(|_| invalid())?;
        let verification = if device.verification_uri_complete.is_empty() {
            device.verification_uri.as_str()
        } else {
            device.verification_uri_complete.as_str()
        };
        if device.device_code.is_empty() || device.user_code.is_empty() || verification.is_empty() {
            return Err(invalid());
        }
        let verification = grok_cli_trusted_endpoint(&self.endpoints.xai_issuer_url, verification)
            .ok_or_else(|| {
                OAuthError::InvalidConfiguration(format!(
                    "Refusing non-xAI OAuth verification_uri: {verification}"
                ))
            })?;
        let interval = reported_device_interval(device.interval, DEFAULT_DEVICE_INTERVAL);
        let timeout = reported_device_timeout(device.expires_in, GROK_CLI_DEVICE_TIMEOUT);
        interaction.notify(OAuthEvent::device_code(
            device.user_code.clone(),
            verification.to_string(),
            interval,
            timeout,
        ));
        interaction.notify(OAuthEvent::progress(
            "Waiting for xAI device authorization...",
        ));
        // Upstream waits one interval before every poll, the first included.
        poll_device_code(
            self.clock.as_ref(),
            cancellation,
            DevicePollingPolicy::new(interval, timeout).wait_before_first_poll(),
            || {
                let response = self.post_form(
                    &endpoints.token,
                    fields([
                        ("grant_type", XAI_DEVICE_GRANT.to_owned()),
                        ("client_id", client_id.clone()),
                        ("device_code", device.device_code.clone()),
                    ]),
                    cancellation,
                )?;
                if (200..300).contains(&response.status) {
                    let payload = self.grok_cli_payload("device poll", response)?;
                    return self
                        .grok_cli_login_credential("device poll", &payload, endpoints, environment)
                        .map(DevicePoll::Complete);
                }
                match parse_device_failure(&response).error.as_str() {
                    "authorization_pending" => Ok(DevicePoll::Pending),
                    "slow_down" => Ok(DevicePoll::SlowDown(None)),
                    "expired_token" => Err(OAuthError::DeviceExpired { provider }),
                    "access_denied" => Err(OAuthError::DeviceDenied { provider }),
                    _ => Err(operation_failure(
                        provider,
                        "device token request",
                        &response,
                    )),
                }
            },
        )
    }

    fn refresh_grok_cli(
        &self,
        current: &Credential,
        environment: &dyn OAuthEnvironment,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let provider = OAuthProviderId::GrokCli.display_name();
        let stored = current
            .extra_string("tokenEndpoint")
            .filter(|endpoint| !endpoint.is_empty())
            .map(str::to_owned)
            .or_else(|| {
                current
                    .extra("discovery")
                    .and_then(|discovery| json_string(discovery, "token_endpoint"))
            });
        let token_endpoint = match stored {
            Some(endpoint) => grok_cli_trusted_endpoint(&self.endpoints.xai_issuer_url, &endpoint)
                .ok_or_else(|| {
                    OAuthError::InvalidConfiguration(format!(
                        "Refusing non-xAI OAuth token_endpoint: {endpoint}"
                    ))
                })?,
            None => self.discover_xai_endpoints(cancellation)?.token,
        };
        if current.refresh().is_empty() {
            return Err(OAuthError::Unauthorized {
                provider,
                operation: "refresh",
                detail: Some("Missing refresh_token. Re-login required.".to_owned()),
            });
        }
        let response = self.post_form(
            &token_endpoint,
            fields([
                ("grant_type", "refresh_token".to_owned()),
                ("client_id", grok_cli_client_id(environment)),
                ("refresh_token", current.refresh().to_owned()),
            ]),
            cancellation,
        )?;
        // Upstream treats 400 like 401 and 403: the refresh token is spent
        // or revoked and only a new login helps.
        if matches!(response.status, 400 | 401 | 403) {
            let detail = truncate_response(&response.body, 500);
            return Err(OAuthError::Unauthorized {
                provider,
                operation: "refresh",
                detail: (!detail.is_empty()).then_some(detail),
            });
        }
        let payload = self.grok_cli_payload("refresh", response)?;
        let access = grok_cli_token_field(&payload, "access_token").unwrap_or_default();
        if access.is_empty() {
            return Err(OAuthError::Unauthorized {
                provider,
                operation: "refresh",
                detail: Some("the refresh returned no access_token".to_owned()),
            });
        }
        let refresh = grok_cli_token_field(&payload, "refresh_token")
            .filter(|refresh| !refresh.is_empty())
            .unwrap_or_else(|| current.refresh().to_owned());
        let mut refreshed = current.clone();
        refreshed.set_access(access);
        refreshed.set_refresh(refresh);
        refreshed.set_expires_at_ms(self.grok_cli_expiry(&payload));
        let id_token = grok_cli_token_field(&payload, "id_token")
            .or_else(|| current.extra_string("idToken").map(str::to_owned))
            .unwrap_or_default();
        let token_type = grok_cli_token_field(&payload, "token_type")
            .or_else(|| current.extra_string("tokenType").map(str::to_owned))
            .unwrap_or_else(|| "Bearer".to_owned());
        for (name, value) in [
            ("tokenEndpoint", Value::String(token_endpoint.to_string())),
            ("idToken", Value::String(id_token)),
            ("tokenType", Value::String(token_type)),
            ("baseUrl", Value::String(grok_cli_base_url(environment))),
        ] {
            refreshed.set_extra(name, value)?;
        }
        Ok(refreshed)
    }

    fn grok_cli_payload(&self, operation: &'static str, response: OAuthResponse) -> Result<Value> {
        let provider = OAuthProviderId::GrokCli.display_name();
        if !(200..300).contains(&response.status) {
            return Err(token_error(provider, operation, &response));
        }
        serde_json::from_slice::<Value>(&response.body)
            .ok()
            .filter(Value::is_object)
            .ok_or(OAuthError::InvalidTokenResponse {
                provider,
                operation,
            })
    }

    fn grok_cli_expiry(&self, payload: &Value) -> i64 {
        let seconds = payload
            .get("expires_in")
            .and_then(json_seconds)
            .unwrap_or(3_600.0);
        self.clock
            .now_ms()
            .saturating_add((seconds * 1_000.0) as i64)
            .saturating_sub(duration_millis(GROK_CLI_REFRESH_SKEW))
    }

    /// upstream `credentialsFromLoginPayload` plus the discovery document a
    /// login records: both tokens are required, and the extras keep the
    /// endpoint a later refresh uses.
    fn grok_cli_login_credential(
        &self,
        operation: &'static str,
        payload: &Value,
        endpoints: &XaiEndpoints,
        environment: &dyn OAuthEnvironment,
    ) -> Result<Credential> {
        let provider = OAuthProviderId::GrokCli.display_name();
        let access = grok_cli_token_field(payload, "access_token").unwrap_or_default();
        let refresh = grok_cli_token_field(payload, "refresh_token").unwrap_or_default();
        if access.is_empty() || refresh.is_empty() {
            return Err(OAuthError::InvalidTokenResponse {
                provider,
                operation,
            });
        }
        let mut credential = Credential::oauth(access, refresh, self.grok_cli_expiry(payload));
        for (name, value) in [
            ("tokenEndpoint", Value::String(endpoints.token.to_string())),
            ("discovery", grok_cli_discovery(endpoints)),
            (
                "idToken",
                Value::String(grok_cli_token_field(payload, "id_token").unwrap_or_default()),
            ),
            (
                "tokenType",
                Value::String(
                    grok_cli_token_field(payload, "token_type")
                        .unwrap_or_else(|| "Bearer".to_owned()),
                ),
            ),
            ("baseUrl", Value::String(grok_cli_base_url(environment))),
        ] {
            credential.set_extra(name, value)?;
        }
        Ok(credential)
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
    /// The Meta Muse flow lives in `meta_muse.rs`; it shares Meta's two
    /// hosts, so a test pointing them at a fake server covers both flows.
    fn meta_muse_flow(&self) -> meta_muse::Flow<'_> {
        meta_muse::Flow {
            transport: self.transport.as_ref(),
            clock: self.clock.as_ref(),
            auth_base_url: &self.endpoints.meta_auth_base_url,
            api_base_url: &self.endpoints.meta_api_base_url,
            timeout: self.token_request_timeout,
        }
    }

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
    use std::{
        collections::VecDeque,
        net::{Shutdown, TcpStream},
        sync::Mutex,
    };

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
            vec![
                "anthropic",
                "kimi-coding",
                "meta",
                "meta-muse",
                "openai-codex",
                "openrouter",
                "xai",
                "grok-cli"
            ]
        );
        let openrouter = metadata_for(OAuthProviderId::OpenRouter);
        assert_eq!(openrouter.flow_support, OAuthFlowSupport::Implemented);
        assert_eq!(openrouter.methods, &[LoginMethod::BrowserPkce]);
        let radius = metadata_for(OAuthProviderId::Radius);
        assert_eq!(radius.flow_support, OAuthFlowSupport::MetadataOnly);
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

        let anthropic = client.anthropic_authorization_url(&pkce, ANTHROPIC_REDIRECT_URI);
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
        /// The query the fake browser sends to the callback.
        query: String,
        events: Mutex<Vec<OAuthEvent>>,
        /// The page the fake browser received.
        page: Arc<Mutex<String>>,
    }

    impl CallbackInteraction {
        fn new(port: u16, query: &str) -> Arc<Self> {
            Arc::new(Self {
                port,
                query: query.to_owned(),
                events: Mutex::new(Vec::new()),
                page: Arc::new(Mutex::new(String::new())),
            })
        }
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
                let query = self.query.clone();
                let page = self.page.clone();
                thread::spawn(move || {
                    let mut stream =
                        TcpStream::connect(("127.0.0.1", port)).expect("connect callback listener");
                    write!(
                        stream,
                        "GET /callback?{query} HTTP/1.1\r\nHost: localhost\r\n\r\n"
                    )
                    .expect("write callback");
                    let mut body = String::new();
                    let _ = stream.read_to_string(&mut body);
                    *lock_unpoisoned(&page) = body;
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

    fn test_loopback_request(port: u16, expected_state: &str) -> LoopbackLoginRequest {
        LoopbackLoginRequest {
            provider_name: "Example",
            authorization_url: fixed_url("https://example.test/authorize"),
            redirect_uri: "http://localhost/callback".to_owned(),
            expected_state: expected_state.to_owned(),
            callback_host: "127.0.0.1".to_owned(),
            callback_port: port,
            callback_path: "/callback".to_owned(),
            manual_input: ManualInputPolicy::Lenient,
        }
    }

    /// Waits for the fake browser thread to store the page it was sent.
    fn received_page(interaction: &CallbackInteraction) -> String {
        let started = Instant::now();
        loop {
            let page = lock_unpoisoned(&interaction.page).clone();
            if !page.is_empty() || started.elapsed() > Duration::from_secs(5) {
                return page;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn loopback_login_answers_the_browser_only_after_the_exchange() {
        let port = unused_loopback_port();
        let interaction = CallbackInteraction::new(port, "code=callback-code&state=expected-state");
        let exchanged = Arc::new(Mutex::new(false));
        let result = run_loopback_login(
            interaction.clone(),
            Arc::new(NoopBrowser),
            &CancellationToken::new(),
            test_loopback_request(port, "expected-state"),
            |code| {
                // The browser has no page yet while the exchange runs.
                assert!(lock_unpoisoned(&interaction.page).is_empty());
                *lock_unpoisoned(&exchanged) = true;
                Ok(format!("exchanged {code}"))
            },
        )
        .expect("callback completes login");
        assert_eq!(result, "exchanged callback-code");
        assert!(*lock_unpoisoned(&exchanged));
        let page = received_page(&interaction);
        assert!(page.starts_with("HTTP/1.1 200"), "{page}");
        assert!(page.contains("Signed in to Example"), "{page}");
        assert!(
            lock_unpoisoned(&interaction.events)
                .iter()
                .any(|event| event.kind == OAuthEventKind::AuthorizationUrl)
        );
    }

    /// Leaves the manual-paste prompt open until the login cancels it.
    struct WaitingInteraction;

    impl OAuthInteraction for WaitingInteraction {
        fn prompt(&self, prompt: OAuthPrompt) -> Result<String> {
            while !prompt.cancellation.is_cancelled() {
                thread::sleep(Duration::from_millis(1));
            }
            Err(OAuthError::Cancelled)
        }

        fn notify(&self, _: OAuthEvent) {}
    }

    /// A browser that follows OpenRouter's authorize URL straight back to its
    /// `callback_url`, as a user who approves the request would.
    struct ApprovingOpenRouterBrowser {
        page: Arc<Mutex<String>>,
    }

    impl BrowserOpener for ApprovingOpenRouterBrowser {
        fn open(&self, url: &Url) -> Result<()> {
            let callback = Url::parse(&query_value(url, "callback_url")).expect("callback URL");
            let page = self.page.clone();
            thread::spawn(move || {
                let port = callback.port().expect("callback port");
                let mut stream =
                    TcpStream::connect(("127.0.0.1", port)).expect("connect callback listener");
                write!(
                    stream,
                    "GET {}?code=granted HTTP/1.1\r\nHost: localhost\r\n\r\n",
                    callback.path()
                )
                .expect("write callback");
                let mut body = String::new();
                let _ = stream.read_to_string(&mut body);
                *lock_unpoisoned(&page) = body;
            });
            Ok(())
        }
    }

    #[test]
    fn openrouter_login_mints_a_permanent_key_through_an_ephemeral_callback() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"key":"sk-or-v1-minted"}"#.to_vec(),
        )]));
        let page = Arc::new(Mutex::new(String::new()));
        let client = OAuthClient::new(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            Arc::new(ApprovingOpenRouterBrowser { page: page.clone() }),
            OAuthEndpoints::default(),
        );
        let interaction: Arc<dyn OAuthInteraction> = Arc::new(WaitingInteraction);
        let credential = client
            .login(
                OAuthProviderId::OpenRouter,
                interaction,
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("OpenRouter login");
        assert_eq!(credential.access(), "sk-or-v1-minted");
        assert_eq!(credential.expires_at_ms(), OPENROUTER_KEY_EXPIRES_MS);

        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].url().as_str(),
            "https://openrouter.ai/api/v1/auth/keys"
        );
        let body: Value = serde_json::from_slice(requests[0].body()).expect("JSON body");
        assert_eq!(body["code"], "granted");
        assert_eq!(body["code_challenge_method"], "S256");
        assert!(
            body["code_verifier"]
                .as_str()
                .is_some_and(|verifier| verifier.len() >= 43)
        );

        let started = Instant::now();
        while lock_unpoisoned(&page).is_empty() && started.elapsed() < Duration::from_secs(5) {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(lock_unpoisoned(&page).contains("Signed in to OpenRouter"));
        // The minted key is never refreshed away.
        assert_eq!(
            client
                .refresh(
                    OAuthProviderId::OpenRouter,
                    &credential,
                    &BTreeMap::new(),
                    &CancellationToken::new()
                )
                .expect("refresh")
                .access(),
            "sk-or-v1-minted"
        );
    }

    #[test]
    fn anthropic_copy_code_login_needs_no_loopback() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"access_token":"sk-ant-oat-new","refresh_token":"rt","expires_in":3600}"#.to_vec(),
        )]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        let interaction = Arc::new(PromptInteraction::answers(["copy_code", "pasted-code"]));
        let credential = client
            .login(
                OAuthProviderId::Anthropic,
                interaction.clone(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("copy-code login");
        assert_eq!(credential.access(), "sk-ant-oat-new");

        let shown = lock_unpoisoned(&interaction.events)
            .iter()
            .find_map(|event| event.authorization_url.clone())
            .expect("authorization URL shown");
        assert_eq!(
            query_value(&Url::parse(&shown).expect("URL"), "redirect_uri"),
            ANTHROPIC_COPY_CODE_REDIRECT_URI
        );
        let requests = transport.requests();
        let body: Value = serde_json::from_slice(requests[0].body()).expect("JSON body");
        assert_eq!(body["code"], "pasted-code");
        assert_eq!(body["redirect_uri"], ANTHROPIC_COPY_CODE_REDIRECT_URI);
        assert_eq!(body["state"], body["code_verifier"]);
    }

    #[test]
    fn callback_host_header_must_name_this_machine() {
        assert!(is_loopback_authority("localhost:53692"));
        assert!(is_loopback_authority("127.0.0.1:1455"));
        assert!(is_loopback_authority("[::1]:1455"));
        assert!(is_loopback_authority("localhost"));
        assert!(!is_loopback_authority("evil.example:53692"));
        assert!(!is_loopback_authority("127.0.0.1.evil.example"));

        let server = LoopbackCallbackServer::bind("127.0.0.1", 0, "/callback", "state")
            .expect("bind callback listener");
        let port = server.local_addr().expect("address").port();
        let rebound = callback_client(port, |stream| {
            let _ = stream.write_all(
                b"GET /callback?code=x&state=state HTTP/1.1\r\nHost: evil.example\r\n\r\n",
            );
        });
        let (accepted, response) = serve_until_done(&server, rebound);
        assert!(accepted.is_none());
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    }

    #[test]
    fn error_bodies_are_reduced_to_their_message() {
        let nested = br#"{
  "error": {
    "message": "Could not validate your token.",
    "type": "invalid_request_error",
    "code": "token_expired"
  }
}"#;
        assert_eq!(
            truncate_response(nested, 500),
            "Could not validate your token."
        );
        assert_eq!(error_code(nested).as_deref(), Some("token_expired"));
        assert!(
            token_error("Example", "refresh", &response(400, nested.to_vec())).is_unauthorized(),
            "an expired token code means log in again, whatever the status"
        );
        assert_eq!(
            truncate_response(
                br#"{"error":"invalid_grant","error_description":"Bad code"}"#,
                500
            ),
            "Bad code"
        );
        assert_eq!(
            truncate_response(b"<html>\n  <b>oops</b>\n</html>", 500),
            "<html> <b>oops</b> </html>"
        );
        // A transient failure stays retryable.
        assert!(
            !token_error(
                "Example",
                "refresh",
                &response(400, br#"{"error":"invalid_request"}"#.to_vec())
            )
            .is_unauthorized()
        );
    }

    #[test]
    fn loopback_login_shows_a_failed_exchange_in_the_browser() {
        let port = unused_loopback_port();
        let interaction = CallbackInteraction::new(port, "code=callback-code&state=expected-state");
        let error = run_loopback_login(
            interaction.clone(),
            Arc::new(NoopBrowser),
            &CancellationToken::new(),
            test_loopback_request(port, "expected-state"),
            |_| -> Result<()> {
                Err(OAuthError::Unauthorized {
                    provider: "Example",
                    operation: "exchange",
                    detail: Some("Invalid 'code' in request.".to_owned()),
                })
            },
        )
        .expect_err("a rejected code fails the login");
        assert!(error.to_string().contains("Invalid 'code'"), "{error}");
        assert!(
            error.to_string().contains("start the login again"),
            "{error}"
        );
        let page = received_page(&interaction);
        assert!(page.starts_with("HTTP/1.1 502"), "{page}");
        assert!(page.contains("Sign-in did not finish"), "{page}");
        assert!(page.contains("Invalid &#39;code&#39;"), "{page}");
        assert!(!page.contains("Signed in"), "{page}");
    }

    #[test]
    fn loopback_login_ends_when_the_provider_redirects_with_an_error() {
        let port = unused_loopback_port();
        let interaction = CallbackInteraction::new(
            port,
            "error=access_denied&error_description=User+declined&state=expected-state",
        );
        let error = run_loopback_login(
            interaction.clone(),
            Arc::new(NoopBrowser),
            &CancellationToken::new(),
            test_loopback_request(port, "expected-state"),
            |_| -> Result<()> { panic!("a denied login has no code to exchange") },
        )
        .expect_err("a denied login fails instead of waiting forever");
        assert!(error.to_string().contains("User declined"), "{error}");
        assert!(received_page(&interaction).contains("User declined"));
    }

    #[test]
    fn loopback_server_without_state_accepts_any_callback_on_its_path() {
        let server = LoopbackCallbackServer::bind("127.0.0.1", 0, "/callback", "")
            .expect("bind callback listener");
        let port = server.local_addr().expect("address").port();
        let client = callback_client(port, |stream| {
            let _ = stream.write_all(b"GET /callback?code=abc HTTP/1.1\r\nHost: localhost\r\n\r\n");
        });
        let (accepted, response) = serve_until_done(&server, client);
        assert_eq!(
            accepted.map(|callback| callback.code).as_deref(),
            Some("abc")
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
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
        // Meta redirects a request without a User-Agent to an HTML page.
        for request in &requests {
            assert_eq!(
                request.headers().get("User-Agent").map(String::as_str),
                Some(OAUTH_USER_AGENT)
            );
        }
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

    /// Connects to the callback listener, runs `send`, and returns whatever
    /// the server answered.
    fn callback_client(
        port: u16,
        send: impl FnOnce(&mut TcpStream) + Send + 'static,
    ) -> thread::JoinHandle<String> {
        thread::spawn(move || {
            let mut stream =
                TcpStream::connect(("127.0.0.1", port)).expect("connect callback listener");
            send(&mut stream);
            let mut response = Vec::new();
            let _ = stream.read_to_end(&mut response);
            String::from_utf8_lossy(&response).into_owned()
        })
    }

    /// Drives the non-blocking listener until the client has its answer.
    fn serve_until_done(
        server: &LoopbackCallbackServer,
        client: thread::JoinHandle<String>,
    ) -> (Option<AuthorizationResponse>, String) {
        let cancellation = CancellationToken::new();
        let started = Instant::now();
        let mut accepted = None;
        while !client.is_finished() {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "callback client never finished"
            );
            match server
                .try_accept(&cancellation)
                .expect("a misbehaving connection is not a login failure")
            {
                Some(CallbackOutcome::Authorized(pending)) => {
                    accepted = Some(pending.response.clone());
                    pending.finish("Example", &Ok(()));
                }
                Some(CallbackOutcome::Denied(_)) | None => {}
            }
            thread::sleep(Duration::from_millis(5));
        }
        (accepted, client.join().expect("client thread"))
    }

    #[test]
    fn loopback_server_survives_bad_connections_and_answers_each_once() {
        let server = LoopbackCallbackServer::bind("127.0.0.1", 0, "/callback", "state")
            .expect("bind callback listener")
            .with_request_timeout(Duration::from_millis(200));
        let port = server.local_addr().expect("address").port();

        let garbage = callback_client(port, |stream| {
            let _ = stream.write_all(b"\xff\xfe\x00 GARBAGE\r\n\r\n");
        });
        let (accepted, response) = serve_until_done(&server, garbage);
        assert!(accepted.is_none());
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");
        assert_eq!(response.matches("HTTP/1.1").count(), 1);

        let silent = callback_client(port, |stream| {
            let _ = stream.shutdown(Shutdown::Write);
        });
        let (accepted, response) = serve_until_done(&server, silent);
        assert!(accepted.is_none());
        assert!(response.starts_with("HTTP/1.1 400"), "{response}");

        // Exactly the read that tips over the limit consumes the last byte
        // sent, so the reply is not lost to a reset.
        let oversized = callback_client(port, |stream| {
            let _ = stream.write_all(&vec![b'a'; 17 * 1024]);
        });
        let (accepted, response) = serve_until_done(&server, oversized);
        assert!(accepted.is_none());
        assert!(response.starts_with("HTTP/1.1 431"), "{response}");
        assert_eq!(
            response.matches("HTTP/1.1").count(),
            1,
            "an oversized request gets one response, not a 431 and a 408"
        );

        // A client trickling one byte per read never idles long enough to hit
        // the socket timeout; only the per-connection deadline stops it.
        let slow = thread::spawn(move || {
            let mut stream =
                TcpStream::connect(("127.0.0.1", port)).expect("connect callback listener");
            stream
                .set_read_timeout(Some(Duration::from_millis(30)))
                .expect("client read timeout");
            let mut response = Vec::new();
            let mut buffer = [0_u8; 1024];
            let mut bytes = b"GET /callback?code=x&state=state HTTP/1.1"
                .iter()
                .cycle()
                .take(200);
            let mut idle_reads = 0;
            loop {
                // Once the reply starts, stop trickling: on Windows a write
                // after the server's close fails at once and would cut the
                // read short, whereas Linux buffers it.
                if response.is_empty() {
                    match bytes.next() {
                        Some(byte) if stream.write_all(&[*byte]).is_ok() => {}
                        _ => break,
                    }
                }
                match stream.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(read) => response.extend_from_slice(&buffer[..read]),
                    Err(error)
                        if matches!(
                            error.kind(),
                            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                        ) =>
                    {
                        idle_reads += 1;
                        if idle_reads > 100 {
                            break;
                        }
                    }
                    Err(_) => break,
                }
                if response.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            String::from_utf8_lossy(&response).into_owned()
        });
        let started = Instant::now();
        let (accepted, response) = serve_until_done(&server, slow);
        assert!(accepted.is_none());
        assert!(response.starts_with("HTTP/1.1 408"), "{response}");
        assert!(started.elapsed() < Duration::from_secs(3));

        let genuine = callback_client(port, |stream| {
            let _ = stream.write_all(
                b"GET /callback?code=callback-code&state=state HTTP/1.1\r\nHost: localhost\r\n\r\n",
            );
        });
        let (accepted, response) = serve_until_done(&server, genuine);
        assert_eq!(
            accepted.map(|callback| callback.code).as_deref(),
            Some("callback-code"),
            "the listener still serves the real callback afterwards"
        );
        assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    }

    #[test]
    fn codex_device_poll_honors_slow_down_and_fractional_intervals() {
        let access = codex_jwt("acct-slow");
        let transport = Arc::new(FakeTransport::with_responses([
            response(
                200,
                br#"{"device_auth_id":"device-auth","user_code":"DEVICE","interval":2.5}"#.to_vec(),
            ),
            response(400, br#"{"error":"slow_down"}"#.to_vec()),
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
        let client = test_client(transport.clone(), clock.clone(), OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers(["device_code"]));
        client
            .login(
                OAuthProviderId::OpenAiCodex,
                interaction,
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("Codex device login survives slow_down");
        // RFC 8628: slow_down adds five seconds to the reported 2.5 s interval.
        assert_eq!(clock.sleeps(), vec![Duration::from_millis(7_500)]);
        assert_eq!(transport.requests().len(), 4);
    }

    #[test]
    fn cancellation_deadline_reports_timeout_and_bounds_request_timeouts() {
        let unbounded = CancellationToken::new();
        assert_eq!(unbounded.remaining(), None);
        assert_eq!(
            bounded_request_timeout(Duration::from_secs(30), &unbounded),
            Duration::from_secs(30)
        );

        let generous = CancellationToken::with_timeout(Duration::from_secs(3600));
        assert!(generous.check().is_ok());
        assert_eq!(
            bounded_request_timeout(Duration::from_secs(30), &generous),
            Duration::from_secs(30)
        );
        let tight = CancellationToken::with_timeout(Duration::from_secs(5));
        assert!(bounded_request_timeout(Duration::from_secs(30), &tight) <= Duration::from_secs(5));

        let expired = CancellationToken::with_timeout(Duration::ZERO);
        assert!(expired.is_cancelled());
        assert!(matches!(expired.check(), Err(OAuthError::TimedOut)));
        assert_eq!(expired.remaining(), Some(Duration::ZERO));
        // An explicit cancellation still wins over an elapsed deadline.
        expired.cancel();
        assert!(matches!(expired.check(), Err(OAuthError::Cancelled)));

        // The production clock's sleep observes the deadline mid-wait.
        let started = Instant::now();
        assert!(matches!(
            SystemClock.sleep(
                Duration::from_secs(5),
                &CancellationToken::with_timeout(Duration::from_millis(50)),
            ),
            Err(OAuthError::TimedOut)
        ));
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    // --- Meta Muse Code (pi-meta-muse-auth) --------------------------------

    fn muse_device_response(body: &str) -> OAuthResponse {
        response(200, body.as_bytes().to_vec())
    }

    fn muse_standard_device() -> OAuthResponse {
        muse_device_response(
            r#"{
                "device_code":"device-secret",
                "user_code":"ABCD-EFGH",
                "verification_uri":"https://auth.meta.com/oauth/device/",
                "verification_uri_complete":"https://auth.meta.com/oauth/device/?code=ABCD-EFGH",
                "expires_in":900,
                "interval":1
            }"#,
        )
    }

    fn muse_identity() -> OAuthResponse {
        response(
            200,
            br#"{"access_token":"identity-token","token_type":"Bearer","refresh_token":"unused"}"#
                .to_vec(),
        )
    }

    type MuseLogin = (
        Result<Credential>,
        Arc<FakeTransport>,
        Arc<FakeClock>,
        Arc<PromptInteraction>,
    );

    fn muse_login(responses: Vec<OAuthResponse>) -> MuseLogin {
        let transport = Arc::new(FakeTransport::with_responses(responses));
        let clock = Arc::new(FakeClock::new(1_000));
        let client = test_client(transport.clone(), clock.clone(), OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers([]));
        let result = client.login(
            OAuthProviderId::MetaMuse,
            interaction.clone(),
            &BTreeMap::new(),
            &CancellationToken::new(),
        );
        (result, transport, clock, interaction)
    }

    fn device_events(interaction: &PromptInteraction) -> Vec<(String, String, u64, u64)> {
        lock_unpoisoned(&interaction.events)
            .iter()
            .filter(|event| event.kind == OAuthEventKind::DeviceCode)
            .map(|event| {
                (
                    event.user_code.clone(),
                    event.verification_uri.clone(),
                    event.interval_seconds,
                    event.expires_in_seconds,
                )
            })
            .collect()
    }

    fn header_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn meta_muse_device_login_mints_a_subscription_key_and_stores_pis_shape() {
        let directory = env::temp_dir().join(format!(
            "goshcoder-oauth-meta-muse-{}-{}",
            std::process::id(),
            Uuid::now_v7()
        ));
        let store = CredentialStore::file(directory.join("auth.json"));
        let transport = Arc::new(FakeTransport::with_responses([
            muse_standard_device(),
            response(400, br#"{"error":"authorization_pending"}"#.to_vec()),
            muse_identity(),
            response(
                200,
                br#"{
                    "api_key":"model-api-key",
                    "base_url":"https://api.meta.ai/v1/",
                    "is_subs_active":true,
                    "subs_tier_name":"Everyday Usage"
                }"#
                .to_vec(),
            ),
        ]));
        let clock = Arc::new(FakeClock::new(1_000));
        let client = test_client(transport.clone(), clock.clone(), OAuthEndpoints::default());
        let interaction = Arc::new(PromptInteraction::answers([]));
        client
            .login_and_persist(
                OAuthProviderId::MetaMuse,
                &store,
                interaction.clone(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("Meta Muse login");

        // One interval before the first poll and one after "pending", as the
        // extension sleeps before every poll.
        assert_eq!(
            clock.sleeps(),
            vec![Duration::from_secs(1), Duration::from_secs(1)]
        );
        assert_eq!(
            device_events(&interaction),
            vec![(
                "ABCD-EFGH".to_owned(),
                "https://auth.meta.com/oauth/device/?code=ABCD-EFGH".to_owned(),
                1,
                900
            )]
        );

        let requests = transport.requests();
        assert_eq!(requests.len(), 4);
        let launcher = header_map(&[
            ("Accept", "application/json"),
            ("Content-Type", "application/x-www-form-urlencoded"),
            ("User-Agent", "muse-code/launcher-2"),
        ]);
        assert_eq!(requests[0].method(), &Method::POST);
        assert_eq!(
            requests[0].url().as_str(),
            "https://auth.meta.com/oidc/device/authorization/"
        );
        assert_eq!(requests[0].headers(), &launcher);
        assert_eq!(
            form(&requests[0]),
            BTreeMap::from([("client_id".to_owned(), "1031625952748946".to_owned())])
        );
        for poll in &requests[1..3] {
            assert_eq!(poll.method(), &Method::POST);
            assert_eq!(
                poll.url().as_str(),
                "https://auth.meta.com/oidc/device/token/"
            );
            assert_eq!(poll.headers(), &launcher);
            assert_eq!(
                form(poll),
                BTreeMap::from([
                    (
                        "grant_type".to_owned(),
                        "urn:ietf:params:oauth:grant-type:device_code".to_owned()
                    ),
                    ("device_code".to_owned(), "device-secret".to_owned()),
                    ("client_id".to_owned(), "1031625952748946".to_owned()),
                ])
            );
        }
        let mint = &requests[3];
        assert_eq!(mint.method(), &Method::POST);
        assert_eq!(mint.url().as_str(), "https://api.meta.ai/muse-code/key");
        // No User-Agent of its own: only the launcher requests carry one.
        assert_eq!(
            mint.headers(),
            &header_map(&[
                ("Accept", "application/json"),
                ("Authorization", "Bearer identity-token"),
                ("Content-Type", "application/json"),
                ("x-api-version", "1.0.0"),
            ])
        );
        let body: Value = serde_json::from_slice(mint.body()).expect("mint JSON");
        assert_eq!(body, json!({"show_subs_upsell": false}));

        // auth.json holds exactly the extension's credential shape, with the
        // trailing slash of Meta's base URL removed.
        let stored: Value =
            serde_json::from_slice(&std::fs::read(directory.join("auth.json")).expect("auth.json"))
                .expect("auth.json JSON");
        assert_eq!(
            stored,
            json!({"meta-muse": {
                "type": "oauth",
                "access": "model-api-key",
                "refresh": "identity-token",
                // Twelve hours from the mint, after the two one-second waits.
                "expires": 3_000 + 12 * 60 * 60 * 1_000,
                "baseUrl": "https://api.meta.ai/v1",
                "subscriptionActive": true,
                "subscriptionTier": "Everyday Usage"
            }})
        );
        let credential = store
            .read_raw("meta-muse")
            .expect("read")
            .expect("stored credential");
        let auth = auth_from_credential(OAuthProviderId::MetaMuse, &credential).expect("auth");
        assert_eq!(auth.api_key(), Some("model-api-key"));
        assert_eq!(auth.base_url(), Some("https://api.meta.ai/v1"));
        assert!(auth.headers().is_empty());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn meta_muse_refuses_a_key_without_a_confirmed_subscription() {
        for mint in [
            r#"{"api_key":"paygo-key","base_url":"https://api.meta.ai/v1","is_subs_active":false}"#,
            r#"{"api_key":"paygo-key","is_subs_active":"true"}"#,
            r#"{"api_key":"paygo-key"}"#,
        ] {
            let store = CredentialStore::in_memory();
            let transport = Arc::new(FakeTransport::with_responses([
                muse_standard_device(),
                muse_identity(),
                response(200, mint.as_bytes().to_vec()),
            ]));
            let client = test_client(
                transport,
                Arc::new(FakeClock::new(0)),
                OAuthEndpoints::default(),
            );
            let error = client
                .login_and_persist(
                    OAuthProviderId::MetaMuse,
                    &store,
                    Arc::new(PromptInteraction::answers([])),
                    &BTreeMap::new(),
                    &CancellationToken::new(),
                )
                .err()
                .expect("pay-as-you-go key refused");
            assert_eq!(
                error.to_string(),
                "Meta did not confirm an active Muse Code subscription; refusing to use a potentially pay-as-you-go key",
                "{mint}"
            );
            assert!(
                store.read_raw("meta-muse").expect("read").is_none(),
                "{mint}"
            );
        }

        // Control: the same exchange with the subscription confirmed works.
        let (result, ..) = muse_login(vec![
            muse_standard_device(),
            muse_identity(),
            response(
                200,
                br#"{"api_key":"model-api-key","is_subs_active":true}"#.to_vec(),
            ),
        ]);
        let credential = result.expect("subscription confirmed");
        // Without base_url or tier: the default URL and no tier field.
        assert_eq!(
            credential.extra_string("baseUrl"),
            Some("https://api.meta.ai/v1")
        );
        assert!(credential.extra("subscriptionTier").is_none());
    }

    #[test]
    fn meta_muse_mint_failures_keep_the_extension_wording() {
        let cases: [(OAuthResponse, &str, bool); 4] = [
            (
                response(200, br#"{"api_key":"","is_subs_active":true}"#.to_vec()),
                "Meta Muse key exchange returned no API key",
                false,
            ),
            (
                response(403, b"{}".to_vec()),
                "Meta Muse subscription key exchange failed with status 403",
                true,
            ),
            (
                response(503, b"busy".to_vec()),
                "Meta Muse subscription key exchange failed with status 503",
                false,
            ),
            (
                response(
                    200,
                    br#"{"api_key":"k","is_subs_active":true,"base_url":"https://evil.example/v1"}"#
                        .to_vec(),
                ),
                "Meta returned an untrusted Model API base URL",
                false,
            ),
        ];
        for (mint, message, unauthorized) in cases {
            let (result, ..) = muse_login(vec![muse_standard_device(), muse_identity(), mint]);
            let error = result.err().expect(message);
            assert_eq!(error.to_string(), message);
            assert_eq!(error.is_unauthorized(), unauthorized, "{message}");
        }
    }

    #[test]
    fn meta_muse_device_responses_are_validated_and_clamped() {
        // An untrusted complete URI falls back to a trusted plain one.
        let (result, transport, _, interaction) = muse_login(vec![
            muse_device_response(
                r#"{"device_code":"d","user_code":"U","verification_uri":"https://auth.meta.com/device",
                    "verification_uri_complete":"http://auth.meta.com/device?code=U","expires_in":5}"#,
            ),
            response(400, br#"{"error":"access_denied"}"#.to_vec()),
        ]);
        assert_eq!(
            result.err().expect("denied").to_string(),
            "Meta Muse login was denied"
        );
        // expires_in is clamped to at least a minute; interval defaults to 5.
        assert_eq!(
            device_events(&interaction),
            vec![(
                "U".to_owned(),
                "https://auth.meta.com/device".to_owned(),
                5,
                60
            )]
        );
        assert_eq!(transport.requests().len(), 2);

        let (result, _, _, interaction) = muse_login(vec![
            muse_device_response(
                r#"{"device_code":"d","user_code":"U","verification_uri":"https://auth.meta.com/device","expires_in":99999,"interval":"7"}"#,
            ),
            response(400, br#"{"error":"expired_token"}"#.to_vec()),
        ]);
        assert_eq!(
            result.err().expect("expired").to_string(),
            "Meta Muse device authorization expired; run /login again"
        );
        assert_eq!(device_events(&interaction)[0].2, 5, "a string interval");
        assert_eq!(
            device_events(&interaction)[0].3,
            1_800,
            "clamped to 30 minutes"
        );

        // A page anywhere but https://auth.meta.com is refused before polling.
        for device in [
            r#"{"device_code":"d","user_code":"U","verification_uri":"https://auth.meta.com.example/device"}"#,
            r#"{"device_code":"d","user_code":"U","verification_uri":"http://auth.meta.com/device"}"#,
            r#"{"device_code":"","user_code":"U","verification_uri":"https://auth.meta.com/device"}"#,
            r#"{"user_code":"U","verification_uri":"https://auth.meta.com/device"}"#,
        ] {
            let (result, transport, _, interaction) =
                muse_login(vec![muse_device_response(device)]);
            assert_eq!(
                result.err().expect(device).to_string(),
                "Meta Muse returned an invalid device authorization response",
                "{device}"
            );
            assert_eq!(transport.requests().len(), 1, "{device}");
            assert!(device_events(&interaction).is_empty(), "{device}");
        }

        let (result, ..) = muse_login(vec![response(500, b"down".to_vec())]);
        assert_eq!(
            result.err().expect("status").to_string(),
            "Meta Muse device authorization failed with status 500"
        );
    }

    #[test]
    fn meta_muse_polling_backs_off_and_expires_with_the_extension_messages() {
        // slow_down adds five seconds to every later wait.
        let (result, _, clock, _) = muse_login(vec![
            muse_device_response(
                r#"{"device_code":"d","user_code":"U","verification_uri":"https://auth.meta.com/device"}"#,
            ),
            response(400, br#"{"error":"slow_down"}"#.to_vec()),
            response(400, br#"{"error":"authorization_pending"}"#.to_vec()),
            response(400, br#"{"error":"server_error"}"#.to_vec()),
        ]);
        assert_eq!(
            result.err().expect("unknown error").to_string(),
            "Meta Muse device token request failed with status 400"
        );
        assert_eq!(
            clock.sleeps(),
            vec![
                Duration::from_secs(5),
                Duration::from_secs(10),
                Duration::from_secs(10)
            ]
        );

        let (result, transport, ..) = muse_login(vec![
            muse_device_response(
                r#"{"device_code":"d","user_code":"U","verification_uri":"https://auth.meta.com/device","expires_in":60,"interval":30}"#,
            ),
            response(400, br#"{"error":"authorization_pending"}"#.to_vec()),
        ]);
        assert_eq!(
            result.err().expect("deadline").to_string(),
            "Meta Muse device authorization expired; run /login again"
        );
        assert_eq!(transport.requests().len(), 2);

        for token in [
            r#"{"access_token":"identity","token_type":"mac"}"#,
            r#"{"access_token":"identity","token_type":null}"#,
            r#"{"access_token":"","token_type":"Bearer"}"#,
        ] {
            let (result, ..) = muse_login(vec![
                muse_standard_device(),
                response(200, token.as_bytes().to_vec()),
            ]);
            assert_eq!(
                result.err().expect(token).to_string(),
                "Meta Muse returned an invalid device token response",
                "{token}"
            );
        }
        // Control: an absent token_type is accepted.
        let (result, ..) = muse_login(vec![
            muse_standard_device(),
            response(200, br#"{"access_token":"identity"}"#.to_vec()),
            response(
                200,
                br#"{"api_key":"model-api-key","is_subs_active":true}"#.to_vec(),
            ),
        ]);
        assert_eq!(result.expect("logged in").refresh(), "identity");
    }

    fn complete_muse_credential() -> Credential {
        let mut credential = Credential::oauth("old-model-key", "identity-token", 0);
        credential
            .set_extra("baseUrl", json!("https://api.meta.ai/v1"))
            .expect("extra");
        credential
            .set_extra("subscriptionActive", json!(true))
            .expect("extra");
        credential
    }

    #[test]
    fn meta_muse_refresh_re_mints_from_the_stored_identity_token() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"api_key":"renewed-model-key","base_url":"https://api.meta.ai/v2","is_subs_active":true,"subs_tier_name":"Everyday Usage"}"#
                .to_vec(),
        )]));
        let clock = Arc::new(FakeClock::new(50_000));
        let client = test_client(transport.clone(), clock, OAuthEndpoints::default());
        let refreshed = client
            .refresh(
                OAuthProviderId::MetaMuse,
                &complete_muse_credential(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .expect("refresh");
        let requests = transport.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].url().as_str(),
            "https://api.meta.ai/muse-code/key"
        );
        assert_eq!(
            requests[0]
                .headers()
                .get("Authorization")
                .map(String::as_str),
            Some("Bearer identity-token")
        );
        assert_eq!(refreshed.access(), "renewed-model-key");
        assert_eq!(refreshed.refresh(), "identity-token");
        assert_eq!(refreshed.expires_at_ms(), 50_000 + 12 * 60 * 60 * 1_000);
        assert_eq!(
            refreshed.extra_string("baseUrl"),
            Some("https://api.meta.ai/v2")
        );
        assert_eq!(
            refreshed.extra_string("subscriptionTier"),
            Some("Everyday Usage")
        );

        // A rejected identity token needs a new login, not a retry.
        let transport = Arc::new(FakeTransport::with_responses([response(
            401,
            b"{}".to_vec(),
        )]));
        let client = test_client(
            transport,
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        let error = client
            .refresh(
                OAuthProviderId::MetaMuse,
                &complete_muse_credential(),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .err()
            .expect("rejected");
        assert!(error.is_unauthorized());
    }

    #[test]
    fn an_incomplete_meta_muse_credential_is_refused_without_a_request() {
        let transport = Arc::new(FakeTransport::with_responses([]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        let mut incomplete = Credential::oauth("model-key", "identity-token", 0);
        incomplete
            .set_extra("baseUrl", json!("https://api.meta.ai/v1"))
            .expect("extra");
        let error = client
            .refresh(
                OAuthProviderId::MetaMuse,
                &incomplete,
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .err()
            .expect("incomplete");
        assert_eq!(
            error.to_string(),
            "Stored Meta Muse credential is incomplete; run /login and configure Meta Muse Code again"
        );
        assert!(error.is_unauthorized());
        assert!(transport.requests().is_empty());
        let error = auth_from_credential(OAuthProviderId::MetaMuse, &incomplete)
            .err()
            .expect("unusable");
        assert!(
            error
                .to_string()
                .starts_with("Stored Meta Muse credential is incomplete")
        );
        // The `meta` provider has no such requirement for the same entry.
        assert!(auth_from_credential(OAuthProviderId::Meta, &incomplete).is_ok());
    }

    // -- Grok CLI (pi-grok-cli) ------------------------------------------------

    /// A browser that answers xAI's preflight probe and then follows the
    /// authorization URL back to the redirect URI it names, as a user who
    /// approves the login would.
    struct ApprovingGrokBrowser {
        preflight: Arc<Mutex<String>>,
        page: Arc<Mutex<String>>,
        authorization_url: Arc<Mutex<Option<Url>>>,
        /// How long the user takes to approve.
        delay: Duration,
    }

    impl ApprovingGrokBrowser {
        fn new() -> Arc<Self> {
            Self::delayed(Duration::ZERO)
        }

        fn delayed(delay: Duration) -> Arc<Self> {
            Arc::new(Self {
                preflight: Arc::new(Mutex::new(String::new())),
                page: Arc::new(Mutex::new(String::new())),
                authorization_url: Arc::new(Mutex::new(None)),
                delay,
            })
        }
    }

    impl BrowserOpener for ApprovingGrokBrowser {
        fn open(&self, url: &Url) -> Result<()> {
            *lock_unpoisoned(&self.authorization_url) = Some(url.clone());
            let redirect = Url::parse(&query_value(url, "redirect_uri")).expect("redirect URI");
            let state = query_value(url, "state");
            let preflight = self.preflight.clone();
            let page = self.page.clone();
            let delay = self.delay;
            thread::spawn(move || {
                thread::sleep(delay);
                let port = redirect.port().expect("redirect port");
                let mut stream =
                    TcpStream::connect(("127.0.0.1", port)).expect("connect for preflight");
                write!(
                    stream,
                    "OPTIONS {} HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://auth.x.ai\r\nAccess-Control-Request-Method: GET\r\nAccess-Control-Request-Private-Network: true\r\n\r\n",
                    redirect.path()
                )
                .expect("write preflight");
                let mut answer = String::new();
                let _ = stream.read_to_string(&mut answer);
                *lock_unpoisoned(&preflight) = answer;

                let mut stream =
                    TcpStream::connect(("127.0.0.1", port)).expect("connect for callback");
                write!(
                    stream,
                    "GET {}?code=granted-code&state={state} HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://auth.x.ai\r\n\r\n",
                    redirect.path()
                )
                .expect("write callback");
                let mut body = String::new();
                let _ = stream.read_to_string(&mut body);
                *lock_unpoisoned(&page) = body;
            });
            Ok(())
        }
    }

    /// Chooses a login method, then either waits out the manual prompt or
    /// pastes the given inputs one prompt at a time.
    struct GrokInteraction {
        method: &'static str,
        pastes: Mutex<VecDeque<String>>,
        authorization_url: Mutex<Option<Url>>,
        events: Mutex<Vec<OAuthEvent>>,
    }

    impl GrokInteraction {
        fn new(method: &'static str, pastes: &[&str]) -> Arc<Self> {
            Arc::new(Self {
                method,
                pastes: Mutex::new(pastes.iter().map(|paste| (*paste).to_owned()).collect()),
                authorization_url: Mutex::new(None),
                events: Mutex::new(Vec::new()),
            })
        }

        fn progress(&self) -> Vec<String> {
            lock_unpoisoned(&self.events)
                .iter()
                .filter(|event| event.kind == OAuthEventKind::Progress)
                .map(|event| event.message.clone())
                .collect()
        }
    }

    impl OAuthInteraction for GrokInteraction {
        fn prompt(&self, prompt: OAuthPrompt) -> Result<String> {
            if prompt.kind == OAuthPromptKind::Select {
                assert_eq!(prompt.message, "Select Grok CLI login method:");
                assert_eq!(prompt.options[0].label, "Browser login (default)");
                assert_eq!(prompt.options[1].label, "Device code login (headless)");
                return Ok(self.method.to_owned());
            }
            let next = lock_unpoisoned(&self.pastes).pop_front();
            match next {
                Some(paste) => {
                    // A paste can only name the state the URL carried.
                    let state = lock_unpoisoned(&self.authorization_url)
                        .as_ref()
                        .map(|url| query_value(url, "state"))
                        .unwrap_or_default();
                    Ok(paste.replace("{state}", &state))
                }
                None => {
                    while !prompt.cancellation.is_cancelled() {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(OAuthError::Cancelled)
                }
            }
        }

        fn notify(&self, event: OAuthEvent) {
            if let Some(url) = event.authorization_url.as_deref() {
                *lock_unpoisoned(&self.authorization_url) = Url::parse(url).ok();
            }
            lock_unpoisoned(&self.events).push(event);
        }
    }

    fn grok_token_body() -> Vec<u8> {
        br#"{"access_token":"grok-access","refresh_token":"grok-refresh","expires_in":3600,"id_token":"grok-id","token_type":"Bearer"}"#
            .to_vec()
    }

    fn grok_environment(port: u16) -> BTreeMap<String, String> {
        BTreeMap::from([("PI_GROK_CLI_CALLBACK_PORT".to_owned(), port.to_string())])
    }

    fn wait_for(cell: &Mutex<String>) -> String {
        let started = Instant::now();
        loop {
            let value = lock_unpoisoned(cell).clone();
            if !value.is_empty() || started.elapsed() > Duration::from_secs(5) {
                return value;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn grok_cli_browser_login_uses_its_own_port_plan_and_verifier_only_exchange() {
        let port = unused_loopback_port();
        let transport = Arc::new(FakeTransport::with_responses([
            // Discovery is unavailable; the documented endpoints are used.
            response(404, b"no discovery".to_vec()),
            response(200, grok_token_body()),
        ]));
        let browser = ApprovingGrokBrowser::new();
        let client = OAuthClient::new(
            transport.clone(),
            Arc::new(FakeClock::new(1_000_000)),
            browser.clone(),
            OAuthEndpoints::default(),
        );
        let interaction = GrokInteraction::new("browser", &[]);
        let credential = client
            .login(
                OAuthProviderId::GrokCli,
                interaction.clone(),
                &grok_environment(port),
                &CancellationToken::new(),
            )
            .expect("Grok CLI browser login");

        let url = lock_unpoisoned(&browser.authorization_url)
            .clone()
            .expect("browser opened");
        assert_eq!(url.host_str(), Some("auth.x.ai"));
        assert_eq!(url.path(), "/oauth2/authorize");
        let redirect_uri = format!("http://127.0.0.1:{port}/callback");
        assert_eq!(query_value(&url, "redirect_uri"), redirect_uri);
        assert_eq!(query_value(&url, "client_id"), GROK_CLI_DEFAULT_CLIENT_ID);
        assert_eq!(query_value(&url, "scope"), GROK_CLI_SCOPE);
        assert_eq!(query_value(&url, "plan"), "generic");
        assert_eq!(query_value(&url, "referrer"), "goshcoder");
        assert_eq!(query_value(&url, "code_challenge_method"), "S256");
        assert!(!query_value(&url, "nonce").is_empty());

        let requests = transport.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].url().as_str(), "https://auth.x.ai/oauth2/token");
        let exchange = form(&requests[1]);
        assert_eq!(exchange["grant_type"], "authorization_code");
        assert_eq!(exchange["code"], "granted-code");
        assert_eq!(exchange["redirect_uri"], redirect_uri);
        assert_eq!(exchange["client_id"], GROK_CLI_DEFAULT_CLIENT_ID);
        assert_eq!(
            URL_SAFE_NO_PAD.encode(Sha256::digest(exchange["code_verifier"].as_bytes())),
            query_value(&url, "code_challenge")
        );
        // Unlike the xAI flow, the exchange carries the verifier alone.
        assert!(!exchange.contains_key("code_challenge"));
        assert!(!exchange.contains_key("code_challenge_method"));

        assert_eq!(credential.access(), "grok-access");
        assert_eq!(credential.refresh(), "grok-refresh");
        // Expiry runs 120 s ahead of the token's own, as upstream stores it.
        assert_eq!(credential.expires_at_ms(), 1_000_000 + 3_600_000 - 120_000);
        assert_eq!(
            credential.extra_string("tokenEndpoint"),
            Some("https://auth.x.ai/oauth2/token")
        );
        assert_eq!(credential.extra_string("idToken"), Some("grok-id"));
        assert_eq!(credential.extra_string("tokenType"), Some("Bearer"));
        assert_eq!(
            credential.extra_string("baseUrl"),
            Some("https://cli-chat-proxy.grok.com/v1")
        );
        assert_eq!(
            credential
                .extra("discovery")
                .and_then(|discovery| json_string(discovery, "device_authorization_endpoint")),
            Some("https://auth.x.ai/oauth2/device/code".to_owned())
        );

        let preflight = wait_for(&browser.preflight);
        assert!(preflight.starts_with("HTTP/1.1 204"), "{preflight}");
        assert!(
            preflight.contains("Access-Control-Allow-Origin: https://auth.x.ai\r\n"),
            "{preflight}"
        );
        assert!(
            preflight.contains("Access-Control-Allow-Private-Network: true\r\n"),
            "{preflight}"
        );
        let page = wait_for(&browser.page);
        assert!(page.starts_with("HTTP/1.1 200"), "{page}");
        assert!(page.contains("Signed in to Grok CLI"), "{page}");
        assert!(
            page.contains("Access-Control-Allow-Origin: https://auth.x.ai\r\n"),
            "{page}"
        );
    }

    #[test]
    fn grok_cli_browser_login_falls_back_to_a_free_port_when_its_own_is_taken() {
        let occupied = TcpListener::bind(("127.0.0.1", 0)).expect("occupy a port");
        let port = occupied.local_addr().expect("address").port();
        let transport = Arc::new(FakeTransport::with_responses([
            response(404, b"no discovery".to_vec()),
            response(200, grok_token_body()),
        ]));
        let browser = ApprovingGrokBrowser::new();
        let client = OAuthClient::new(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            browser.clone(),
            OAuthEndpoints::default(),
        );
        client
            .login(
                OAuthProviderId::GrokCli,
                GrokInteraction::new("", &[]),
                &grok_environment(port),
                &CancellationToken::new(),
            )
            .expect("login on the fallback port");
        let url = lock_unpoisoned(&browser.authorization_url)
            .clone()
            .expect("browser opened");
        let redirect = Url::parse(&query_value(&url, "redirect_uri")).expect("redirect URI");
        assert_ne!(redirect.port(), Some(port));
        assert_eq!(redirect.path(), "/callback");
        // The exchange names the port that was actually bound.
        assert_eq!(
            form(&transport.requests()[1])["redirect_uri"],
            redirect.as_str()
        );
        drop(occupied);
    }

    #[test]
    fn grok_cli_an_ignored_paste_leaves_the_browser_callback_to_finish_the_login() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(404, b"no discovery".to_vec()),
            response(200, grok_token_body()),
        ]));
        let browser = ApprovingGrokBrowser::delayed(Duration::from_millis(300));
        let client = OAuthClient::new(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            browser.clone(),
            OAuthEndpoints::default(),
        );
        // A URL without the state is not trusted, however plausible.
        let interaction =
            GrokInteraction::new("browser", &["http://127.0.0.1:1/callback?code=pasted-code"]);
        client
            .login(
                OAuthProviderId::GrokCli,
                interaction.clone(),
                &grok_environment(unused_loopback_port()),
                &CancellationToken::new(),
            )
            .expect("the browser callback completes the login");
        assert_eq!(form(&transport.requests()[1])["code"], "granted-code");
        let progress = interaction.progress();
        assert!(
            progress.iter().any(|message| message
                == "Ignored pasted callback: OAuth state is missing. Paste the complete callback URL or xAI's one-time code."),
            "{progress:?}"
        );
        assert!(wait_for(&browser.page).contains("Signed in to Grok CLI"));
    }

    #[test]
    fn grok_cli_a_pasted_callback_with_the_matching_state_completes_the_login() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(404, b"no discovery".to_vec()),
            response(200, grok_token_body()),
        ]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        let interaction = GrokInteraction::new(
            "browser",
            &["http://127.0.0.1:1/callback?code=pasted-code&state={state}"],
        );
        client
            .login(
                OAuthProviderId::GrokCli,
                interaction.clone(),
                &grok_environment(unused_loopback_port()),
                &CancellationToken::new(),
            )
            .expect("the matching paste completes the login");
        assert_eq!(form(&transport.requests()[1])["code"], "pasted-code");
        assert!(
            !interaction
                .progress()
                .iter()
                .any(|message| message.starts_with("Ignored"))
        );
    }

    #[test]
    fn grok_cli_strict_manual_input_matches_upstream_rules() {
        let code = "a".repeat(32);
        assert_eq!(
            parse_strict_manual_input(&code, "s", "/callback"),
            Ok(ManualCallback::Code(code.clone()))
        );
        // One character short of a one-time code is neither code nor query.
        assert!(parse_strict_manual_input(&"a".repeat(31), "s", "/callback").is_err());
        assert_eq!(
            parse_strict_manual_input("?code=abc&state=s", "s", "/callback"),
            Ok(ManualCallback::Code("abc".to_owned()))
        );
        assert_eq!(
            parse_strict_manual_input(
                "http://127.0.0.1:56122/callback?error=access_denied&error_description=No&state=s",
                "s",
                "/callback"
            ),
            Ok(ManualCallback::Denied("No".to_owned()))
        );
        assert_eq!(
            parse_strict_manual_input("http://127.0.0.1/other?code=abc&state=s", "s", "/callback"),
            Err("Callback URL path was not recognized.".to_owned())
        );
        assert_eq!(
            parse_strict_manual_input("code=abc&state=other", "s", "/callback"),
            Err("OAuth state did not match.".to_owned())
        );
        assert_eq!(
            parse_strict_manual_input("code=abc", "s", "/callback"),
            Err("OAuth state is missing.".to_owned())
        );
        assert_eq!(
            parse_strict_manual_input("  ", "s", "/callback"),
            Err("Pasted callback was empty.".to_owned())
        );
        // The lenient parser other providers use takes any bare code.
        assert!(parse_authorization_input("too-short").is_some());
    }

    #[test]
    fn cors_preflight_is_answered_only_where_enabled_and_only_for_xai_origins() {
        let preflight = |server: &LoopbackCallbackServer, origin: &'static str| {
            let port = server.local_addr().expect("address").port();
            let client = callback_client(port, move |stream| {
                let _ = write!(
                    stream,
                    "OPTIONS /callback HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: {origin}\r\n\r\n"
                );
            });
            serve_until_done(server, client).1
        };
        let grok = LoopbackCallbackServer::bind("127.0.0.1", 0, "/callback", "state")
            .expect("bind")
            .with_cors_origins(GROK_CLI_CORS_ORIGINS);
        let trusted = preflight(&grok, "https://accounts.x.ai");
        assert!(trusted.starts_with("HTTP/1.1 204"), "{trusted}");
        for header in [
            "Access-Control-Allow-Origin: https://accounts.x.ai",
            "Access-Control-Allow-Methods: GET, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type",
            "Access-Control-Allow-Private-Network: true",
            "Vary: Origin",
        ] {
            assert!(trusted.contains(&format!("{header}\r\n")), "{trusted}");
        }
        let untrusted = preflight(&grok, "https://evil.example");
        assert!(untrusted.starts_with("HTTP/1.1 204"), "{untrusted}");
        assert!(
            !untrusted.contains("Access-Control-Allow-Origin"),
            "{untrusted}"
        );

        // Every other provider's listener still refuses a non-GET request.
        let plain =
            LoopbackCallbackServer::bind("127.0.0.1", 0, "/callback", "state").expect("bind");
        let refused = preflight(&plain, "https://accounts.x.ai");
        assert!(refused.starts_with("HTTP/1.1 405"), "{refused}");
        assert!(
            !refused.contains("Access-Control-Allow-Origin"),
            "{refused}"
        );
    }

    #[test]
    fn grok_cli_device_login_waits_before_polling_and_records_the_grant() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(404, b"no discovery".to_vec()),
            response(
                200,
                br#"{
                    "device_code":"grok-device",
                    "user_code":"GROK-CODE",
                    "verification_uri":"https://accounts.x.ai/device",
                    "verification_uri_complete":"https://accounts.x.ai/device?code=GROK-CODE",
                    "interval":2
                }"#
                .to_vec(),
            ),
            response(400, br#"{"error":"authorization_pending"}"#.to_vec()),
            response(400, br#"{"error":"slow_down"}"#.to_vec()),
            response(200, grok_token_body()),
        ]));
        let clock = Arc::new(FakeClock::new(0));
        let client = test_client(transport.clone(), clock.clone(), OAuthEndpoints::default());
        let interaction = GrokInteraction::new("device", &[]);
        let environment = BTreeMap::from([(
            "PI_GROK_CLI_OAUTH_SCOPE".to_owned(),
            "openid custom".to_owned(),
        )]);
        let credential = client
            .login(
                OAuthProviderId::GrokCli,
                interaction.clone(),
                &environment,
                &CancellationToken::new(),
            )
            .expect("device login");
        assert_eq!(credential.access(), "grok-access");
        let requests = transport.requests();
        assert_eq!(requests.len(), 5);
        let device = form(&requests[1]);
        assert_eq!(device["scope"], "openid custom");
        assert_eq!(device["client_id"], GROK_CLI_DEFAULT_CLIENT_ID);
        assert_eq!(form(&requests[2])["grant_type"], XAI_DEVICE_GRANT);
        assert_eq!(form(&requests[2])["device_code"], "grok-device");
        // One interval before the first poll, then slow_down adds five.
        assert_eq!(
            clock.sleeps(),
            vec![
                Duration::from_secs(2),
                Duration::from_secs(2),
                Duration::from_secs(7)
            ]
        );
        let events = lock_unpoisoned(&interaction.events);
        let code = events
            .iter()
            .find(|event| event.kind == OAuthEventKind::DeviceCode)
            .expect("device code shown");
        assert_eq!(code.user_code, "GROK-CODE");
        assert_eq!(
            code.verification_uri,
            "https://accounts.x.ai/device?code=GROK-CODE"
        );
        // Upstream's default device window is 30 minutes.
        assert_eq!(code.expires_in_seconds, 1800);
    }

    #[test]
    fn grok_cli_device_login_refuses_a_verification_page_off_xai() {
        let transport = Arc::new(FakeTransport::with_responses([
            response(404, b"no discovery".to_vec()),
            response(
                200,
                br#"{"device_code":"d","user_code":"u","verification_uri":"https://phish.example/device"}"#
                    .to_vec(),
            ),
        ]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        let error = client
            .login(
                OAuthProviderId::GrokCli,
                GrokInteraction::new("device", &[]),
                &BTreeMap::new(),
                &CancellationToken::new(),
            )
            .err()
            .expect("a foreign verification page is refused");
        assert!(error.to_string().contains("phish.example"), "{error}");
        assert_eq!(transport.requests().len(), 2, "nothing was polled");
    }

    #[test]
    fn grok_cli_login_refuses_while_the_environment_token_is_set() {
        let transport = Arc::new(FakeTransport::with_responses([]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        let environment = BTreeMap::from([(
            crate::grok_cli::TOKEN_ENV.to_owned(),
            "env-token".to_owned(),
        )]);
        let error = client
            .login(
                OAuthProviderId::GrokCli,
                GrokInteraction::new("browser", &[]),
                &environment,
                &CancellationToken::new(),
            )
            .err()
            .expect("login refuses");
        assert_eq!(
            error.to_string(),
            "Unset GROK_CLI_OAUTH_TOKEN before logging in to Grok CLI."
        );
        assert!(transport.requests().is_empty());
        // The xAI login is unaffected by the Grok CLI token.
        assert!(matches!(
            test_client(
                Arc::new(FakeTransport::with_responses([])),
                Arc::new(FakeClock::new(0)),
                OAuthEndpoints::default()
            )
            .login(
                OAuthProviderId::Xai,
                Arc::new(PromptInteraction::answers([])),
                &environment,
                &CancellationToken::new(),
            ),
            Err(OAuthError::Cancelled)
        ));
    }

    fn stored_grok_credential(token_endpoint: &str) -> Credential {
        let mut credential = Credential::oauth("old-access", "old-refresh", 0);
        credential
            .set_extra("tokenEndpoint", Value::String(token_endpoint.to_owned()))
            .expect("extra");
        credential
            .set_extra("idToken", Value::String("old-id".to_owned()))
            .expect("extra");
        credential
    }

    #[test]
    fn grok_cli_refresh_uses_the_stored_endpoint_and_keeps_a_nonrotating_token() {
        let transport = Arc::new(FakeTransport::with_responses([response(
            200,
            br#"{"access_token":"new-access","expires_in":"60"}"#.to_vec(),
        )]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(5_000)),
            OAuthEndpoints::default(),
        );
        let environment = BTreeMap::from([(
            "PI_GROK_CLI_BASE_URL".to_owned(),
            "https://proxy.example/v1/".to_owned(),
        )]);
        let refreshed = client
            .refresh(
                OAuthProviderId::GrokCli,
                &stored_grok_credential("https://auth.x.ai/oauth2/token"),
                &environment,
                &CancellationToken::new(),
            )
            .expect("refresh");
        let requests = transport.requests();
        assert_eq!(requests.len(), 1, "no discovery request");
        assert_eq!(requests[0].url().as_str(), "https://auth.x.ai/oauth2/token");
        let body = form(&requests[0]);
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["refresh_token"], "old-refresh");
        assert_eq!(body["client_id"], GROK_CLI_DEFAULT_CLIENT_ID);
        assert_eq!(refreshed.access(), "new-access");
        assert_eq!(refreshed.refresh(), "old-refresh");
        assert_eq!(refreshed.expires_at_ms(), 5_000 + 60_000 - 120_000);
        assert_eq!(refreshed.extra_string("idToken"), Some("old-id"));
        assert_eq!(refreshed.extra_string("tokenType"), Some("Bearer"));
        assert_eq!(
            refreshed.extra_string("baseUrl"),
            Some("https://proxy.example/v1")
        );
    }

    #[test]
    fn grok_cli_refresh_treats_400_401_403_as_a_lost_login_and_5xx_as_transient() {
        for (status, unauthorized) in [(400, true), (401, true), (403, true), (500, false)] {
            let transport = Arc::new(FakeTransport::with_responses([response(
                status,
                br#"{"error":"invalid_request"}"#.to_vec(),
            )]));
            let client = test_client(
                transport,
                Arc::new(FakeClock::new(0)),
                OAuthEndpoints::default(),
            );
            let error = client
                .refresh(
                    OAuthProviderId::GrokCli,
                    &stored_grok_credential("https://auth.x.ai/oauth2/token"),
                    &BTreeMap::new(),
                    &CancellationToken::new(),
                )
                .err()
                .expect("refresh fails");
            assert_eq!(
                error.is_unauthorized(),
                unauthorized,
                "status {status}: {error}"
            );
        }
    }

    #[test]
    fn grok_cli_refresh_refuses_a_stored_endpoint_off_xai_before_sending_the_token() {
        let transport = Arc::new(FakeTransport::with_responses([]));
        let client = test_client(
            transport.clone(),
            Arc::new(FakeClock::new(0)),
            OAuthEndpoints::default(),
        );
        for endpoint in [
            "https://evil.example/token",
            "http://auth.x.ai/oauth2/token",
        ] {
            let error = client
                .refresh(
                    OAuthProviderId::GrokCli,
                    &stored_grok_credential(endpoint),
                    &BTreeMap::new(),
                    &CancellationToken::new(),
                )
                .err()
                .expect("untrusted endpoint");
            assert!(error.to_string().contains("Refusing non-xAI"), "{error}");
        }
        assert!(transport.requests().is_empty());
    }
}
