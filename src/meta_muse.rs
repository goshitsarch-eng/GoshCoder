//! Meta Muse Code subscription provider (`meta-muse`).
//!
//! Native adaptation of the pi extension `pi-meta-muse-auth` 0.1.2 by Sadik
//! Saifi (MIT, <https://github.com/sadiksaifi/pi-meta-muse-auth>), written
//! against its `src/oauth.ts` (device login, key mint, refresh, `toAuth`),
//! `src/models.ts` (bundled and live model catalogs) and `src/index.ts`
//! (provider registration and stored-credential validation). No upstream code
//! is bundled.
//!
//! It sits beside the `meta` provider rather than replacing it. Both sign in
//! at auth.meta.com, but this one mints its Model API key with the Muse Code
//! subscription request and refuses any key Meta does not confirm as
//! subscription-backed, so a login can never quietly start billing
//! pay-as-you-go. The `auth.json` entry has the extension's exact shape, so pi
//! and GoshCoder share one `meta-muse` login.
//!
//! Where this differs from the extension, and why:
//!
//! - Polling reuses [`oauth::poll_device_code`], which treats a reported
//!   interval under one second as absent (five seconds); the extension would
//!   poll that fast.
//! - pi keeps an extension's `fetchModels` result in its models store. Here
//!   the fetched catalog is written to `extensions/meta-muse-models.json`
//!   after a login and again in the background at session start, and the
//!   catalog's dynamic layer serves it; the bundled models in
//!   `catalog_extra.json` remain the fallback.
//! - A mint rejected with 401/403, a refused subscription, and an incomplete
//!   stored credential are reported as needing a new login, so the catalog
//!   remembers them instead of retrying every few seconds.
//! - The catalog refreshes the key five minutes before its stored expiry, as
//!   it does for every OAuth provider; pi waits for the expiry itself.

use std::{collections::BTreeMap, fs, io::Read, path::Path, time::Duration};

use reqwest::Method;
use serde_json::{Map, Value, json};
use url::Url;

use crate::{
    catalog::{Catalog, Credential, CredentialKind},
    config, llm,
    oauth::{
        self, CancellationToken, DevicePoll, DevicePollingPolicy, OAuthClock, OAuthError,
        OAuthEvent, OAuthInteraction, OAuthRequest, OAuthResponse, OAuthTransport,
    },
};

pub const PROVIDER_ID: &str = "meta-muse";
pub const CLIENT_ID: &str = "1031625952748946";
pub const DEFAULT_API_BASE_URL: &str = "https://api.meta.ai/v1";
pub const MODELS_URL: &str = "https://api.meta.ai/v1/models";
/// Sent by the device-flow requests, as the Muse Code launcher does.
pub const LAUNCHER_USER_AGENT: &str = "muse-code/launcher-2";
/// The client identity Meta requires before it honours the `max` effort.
pub const MUSE_USER_AGENT: &str = "muse-build/pi-meta-muse-auth";
pub const FALLBACK_MODEL_IDS: [&str; 5] = [
    "muse-spark-1.3-contributor",
    "muse-spark-1.3",
    "muse-spark-1.2-contributor",
    "muse-spark-1.2",
    "muse-spark-1.1",
];

/// Extra `auth.json` fields of a `meta-muse` credential, named as pi stores
/// them.
pub const BASE_URL_EXTRA: &str = "baseUrl";
pub const SUBSCRIPTION_ACTIVE_EXTRA: &str = "subscriptionActive";
pub const SUBSCRIPTION_TIER_EXTRA: &str = "subscriptionTier";

/// Bounds a model-catalog request, as the extension's `AbortSignal.timeout`.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

const TRUSTED_AUTH_HOST: &str = "auth.meta.com";
const TRUSTED_API_HOST: &str = "api.meta.ai";
const DEVICE_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const API_VERSION: &str = "1.0.0";
const DEFAULT_POLL_INTERVAL_SECONDS: f64 = 5.0;
const DEFAULT_DEVICE_EXPIRY_SECONDS: f64 = 15.0 * 60.0;
const MIN_DEVICE_EXPIRY_SECONDS: f64 = 60.0;
const MAX_DEVICE_EXPIRY_SECONDS: f64 = 1_800.0;
const CREDENTIAL_LIFETIME: Duration = Duration::from_secs(12 * 60 * 60);
const DEFAULT_CONTEXT_WINDOW: u64 = 1_007_997;
const DEFAULT_MAX_TOKENS: u64 = 128_000;
/// `Number.MAX_SAFE_INTEGER`: the extension accepts only safe integers.
const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
/// A cached catalog larger than this is ignored rather than parsed at every
/// startup; Meta's list is a few kilobytes.
const MAX_CACHE_BYTES: u64 = 1024 * 1024;
const THINKING_LEVELS: [&str; 6] = ["minimal", "low", "medium", "high", "xhigh", "max"];

const INCOMPLETE_CREDENTIAL: &str =
    "Stored Meta Muse credential is incomplete; run /login and configure Meta Muse Code again";
const SUBSCRIPTION_REFUSED: &str = "Meta did not confirm an active Muse Code subscription; refusing to use a potentially pay-as-you-go key";
const DEVICE_EXPIRED: &str = "Meta Muse device authorization expired; run /login again";

type Result<T> = oauth::Result<T>;

/// A failure that retrying later may cure.
fn failure(message: impl Into<String>) -> OAuthError {
    OAuthError::Message {
        message: message.into(),
        unauthorized: false,
    }
}

/// A failure only a new login cures.
fn rejection(message: impl Into<String>) -> OAuthError {
    OAuthError::Message {
        message: message.into(),
        unauthorized: true,
    }
}

fn is_success(status: u16) -> bool {
    (200..300).contains(&status)
}

/// `readJson`: anything but a JSON object reads as absent.
fn read_json(body: &[u8]) -> Option<Map<String, Value>> {
    match serde_json::from_slice(body) {
        Ok(Value::Object(object)) => Some(object),
        _ => None,
    }
}

fn non_empty_string<'a>(object: Option<&'a Map<String, Value>>, name: &str) -> Option<&'a str> {
    object
        .and_then(|object| object.get(name))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

/// `positiveNumber`: a finite positive JSON number, else `fallback`.
fn positive_number(value: Option<&Value>, fallback: f64) -> f64 {
    value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number > 0.0)
        .unwrap_or(fallback)
}

/// `trustedMetaUrl`: the device page must be Meta's own, so a tampered
/// response cannot send the user to type their code into someone else's site.
fn trusted_meta_url(value: Option<&Value>) -> Option<String> {
    let text = value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())?;
    let url = Url::parse(text).ok()?;
    (url.scheme() == "https" && url.host_str() == Some(TRUSTED_AUTH_HOST))
        .then(|| url.as_str().to_owned())
}

/// `sanctionedApiBaseUrl`: the minted key must only ever be sent to Meta's
/// Model API over HTTPS. A missing or empty value means the default; anything
/// else must parse, and loses its credentials, query, fragment and one
/// trailing slash.
pub fn sanctioned_api_base_url(value: Option<&Value>) -> Result<String> {
    let Some(text) = value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    else {
        return Ok(DEFAULT_API_BASE_URL.to_owned());
    };
    let mut url =
        Url::parse(text).map_err(|_| failure("Meta returned an invalid Model API base URL"))?;
    if url.scheme() != "https" || url.host_str() != Some(TRUSTED_API_HOST) {
        return Err(failure("Meta returned an untrusted Model API base URL"));
    }
    // Neither can fail on an https URL with a host.
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    let href = url.as_str();
    Ok(href.strip_suffix('/').unwrap_or(href).to_owned())
}

/// `asMuseCredential`: a `meta-muse` entry pi or an older build wrote without
/// the subscription fields cannot be used or refreshed safely.
fn require_complete(credential: &Credential) -> Result<()> {
    let complete = matches!(credential.extra(BASE_URL_EXTRA), Some(Value::String(_)))
        && matches!(
            credential.extra(SUBSCRIPTION_ACTIVE_EXTRA),
            Some(Value::Bool(_))
        )
        && !credential.refresh().is_empty();
    if complete {
        Ok(())
    } else {
        Err(rejection(INCOMPLETE_CREDENTIAL))
    }
}

/// `toMuseAuth`: the API key and the base URL a stored credential sends it
/// to. The stored URL is validated again because auth.json is user-editable.
pub fn request_auth(credential: &Credential) -> Result<(String, String)> {
    require_complete(credential)?;
    let base_url = sanctioned_api_base_url(credential.extra(BASE_URL_EXTRA)).map_err(|error| {
        // Only a new login replaces a stored URL that fails validation.
        rejection(error.to_string())
    })?;
    Ok((credential.access().to_owned(), base_url))
}

/// A subscription-backed key as Meta returned it.
struct MintedKey {
    api_key: String,
    base_url: String,
    subscription_tier: Option<String>,
}

struct DeviceAuthorization {
    device_code: String,
    user_code: String,
    verification_uri: String,
    expires_in: Duration,
    interval: Duration,
}

/// The device login, mint and refresh, run over the OAuth client's transport
/// and clock so tests drive them without a network or real waits.
pub(crate) struct Flow<'a> {
    pub transport: &'a dyn OAuthTransport,
    pub clock: &'a dyn OAuthClock,
    /// `https://auth.meta.com` outside tests.
    pub auth_base_url: &'a Url,
    /// `https://api.meta.ai` outside tests.
    pub api_base_url: &'a Url,
    pub timeout: Duration,
}

impl Flow<'_> {
    /// `loginMetaMuse`.
    pub(crate) fn login(
        &self,
        interaction: &dyn OAuthInteraction,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        let device = self.start_device_authorization(cancellation)?;
        interaction.notify(OAuthEvent::device_code(
            device.user_code.clone(),
            device.verification_uri.clone(),
            device.interval,
            device.expires_in,
        ));
        let identity = self.poll_for_device_token(&device, cancellation)?;
        interaction.notify(OAuthEvent::progress(
            "Confirming the Muse Code subscription...",
        ));
        let minted = self.mint(&identity, cancellation)?;
        self.credential(identity, minted)
    }

    /// `refreshMetaMuse`: the key is re-minted from the stored identity
    /// token; there is no refresh-token exchange.
    pub(crate) fn refresh(
        &self,
        current: &Credential,
        cancellation: &CancellationToken,
    ) -> Result<Credential> {
        require_complete(current)?;
        let identity = current.refresh().to_owned();
        let minted = self.mint(&identity, cancellation)?;
        self.credential(identity, minted)
    }

    fn launcher_form(
        &self,
        url: Url,
        fields: &[(&str, &str)],
        cancellation: &CancellationToken,
    ) -> Result<OAuthResponse> {
        let mut encoder = url::form_urlencoded::Serializer::new(String::new());
        for (name, value) in fields {
            encoder.append_pair(name, value);
        }
        let headers = BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            (
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
            ("User-Agent".to_owned(), LAUNCHER_USER_AGENT.to_owned()),
        ]);
        self.transport.execute(
            OAuthRequest::new(
                Method::POST,
                url,
                headers,
                encoder.finish().into_bytes(),
                self.timeout,
            ),
            cancellation,
        )
    }

    /// `startDeviceAuthorization`.
    fn start_device_authorization(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<DeviceAuthorization> {
        let url = oauth::endpoint(self.auth_base_url, "/oidc/device/authorization/")?;
        let response = self.launcher_form(url, &[("client_id", CLIENT_ID)], cancellation)?;
        if !is_success(response.status) {
            return Err(failure(format!(
                "Meta Muse device authorization failed with status {}",
                response.status
            )));
        }
        let json = read_json(&response.body);
        let json = json.as_ref();
        let device_code = non_empty_string(json, "device_code");
        let user_code = non_empty_string(json, "user_code");
        let verification_uri =
            trusted_meta_url(json.and_then(|json| json.get("verification_uri_complete")))
                .or_else(|| trusted_meta_url(json.and_then(|json| json.get("verification_uri"))));
        let (Some(device_code), Some(user_code), Some(verification_uri)) =
            (device_code, user_code, verification_uri)
        else {
            return Err(failure(
                "Meta Muse returned an invalid device authorization response",
            ));
        };
        let expires_in = positive_number(
            json.and_then(|json| json.get("expires_in")),
            DEFAULT_DEVICE_EXPIRY_SECONDS,
        )
        .clamp(MIN_DEVICE_EXPIRY_SECONDS, MAX_DEVICE_EXPIRY_SECONDS);
        let interval = positive_number(
            json.and_then(|json| json.get("interval")),
            DEFAULT_POLL_INTERVAL_SECONDS,
        );
        Ok(DeviceAuthorization {
            device_code: device_code.to_owned(),
            user_code: user_code.to_owned(),
            verification_uri,
            expires_in: Duration::from_secs_f64(expires_in),
            interval: Duration::try_from_secs_f64(interval)
                .unwrap_or(Duration::from_secs_f64(DEFAULT_POLL_INTERVAL_SECONDS)),
        })
    }

    /// `pollForDeviceToken`: waits one interval before the first poll, as
    /// the extension does, and returns the identity access token.
    fn poll_for_device_token(
        &self,
        device: &DeviceAuthorization,
        cancellation: &CancellationToken,
    ) -> Result<String> {
        let url = oauth::endpoint(self.auth_base_url, "/oidc/device/token/")?;
        let policy =
            DevicePollingPolicy::new(device.interval, device.expires_in).wait_before_first_poll();
        let polled = oauth::poll_device_code(self.clock, cancellation, policy, || {
            let response = self.launcher_form(
                url.clone(),
                &[
                    ("grant_type", DEVICE_CODE_GRANT),
                    ("device_code", &device.device_code),
                    ("client_id", CLIENT_ID),
                ],
                cancellation,
            )?;
            let json = read_json(&response.body);
            if is_success(response.status) {
                let access_token = non_empty_string(json.as_ref(), "access_token");
                // Present but not "Bearer" (null included) is refused, as an
                // `!== undefined` check refuses it upstream.
                let bearer = json
                    .as_ref()
                    .and_then(|json| json.get("token_type"))
                    .is_none_or(|token_type| token_type.as_str() == Some("Bearer"));
                return match access_token {
                    Some(access_token) if bearer => {
                        Ok(DevicePoll::Complete(access_token.to_owned()))
                    }
                    _ => Err(failure(
                        "Meta Muse returned an invalid device token response",
                    )),
                };
            }
            match json
                .as_ref()
                .and_then(|json| json.get("error"))
                .and_then(Value::as_str)
            {
                Some("authorization_pending") => Ok(DevicePoll::Pending),
                // Five more seconds per slow_down, RFC 8628 section 3.5.
                Some("slow_down") => Ok(DevicePoll::SlowDown(None)),
                Some("access_denied") => Err(failure("Meta Muse login was denied")),
                Some("expired_token") => Err(failure(DEVICE_EXPIRED)),
                _ => Err(failure(format!(
                    "Meta Muse device token request failed with status {}",
                    response.status
                ))),
            }
        });
        match polled {
            Err(OAuthError::DeviceTimedOut) => Err(failure(DEVICE_EXPIRED)),
            other => other,
        }
    }

    /// `mintMuseKey`. Unlike the device requests it carries no launcher
    /// User-Agent, only the transport's default one, as upstream's fetch does.
    fn mint(&self, identity: &str, cancellation: &CancellationToken) -> Result<MintedKey> {
        let url = oauth::endpoint(self.api_base_url, "/muse-code/key")?;
        let headers = BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            ("Authorization".to_owned(), format!("Bearer {identity}")),
            ("Content-Type".to_owned(), "application/json".to_owned()),
            ("x-api-version".to_owned(), API_VERSION.to_owned()),
        ]);
        let body = json!({"show_subs_upsell": false}).to_string().into_bytes();
        let response = self.transport.execute(
            OAuthRequest::new(Method::POST, url, headers, body, self.timeout),
            cancellation,
        )?;
        if !is_success(response.status) {
            let message = format!(
                "Meta Muse subscription key exchange failed with status {}",
                response.status
            );
            // A rejected identity token stays rejected; retrying cannot help.
            return Err(if matches!(response.status, 401 | 403) {
                rejection(message)
            } else {
                failure(message)
            });
        }
        let json = read_json(&response.body);
        let Some(api_key) = non_empty_string(json.as_ref(), "api_key") else {
            return Err(failure("Meta Muse key exchange returned no API key"));
        };
        // The whole point of the provider: without Meta's explicit
        // confirmation the key may bill pay-as-you-go, so it is never stored.
        if json.as_ref().and_then(|json| json.get("is_subs_active")) != Some(&Value::Bool(true)) {
            return Err(rejection(SUBSCRIPTION_REFUSED));
        }
        Ok(MintedKey {
            api_key: api_key.to_owned(),
            base_url: sanctioned_api_base_url(json.as_ref().and_then(|json| json.get("base_url")))?,
            subscription_tier: json
                .as_ref()
                .and_then(|json| json.get("subs_tier_name"))
                .and_then(Value::as_str)
                .map(str::to_owned),
        })
    }

    /// `credentialFrom`: pi's `{type, access, refresh, expires, baseUrl,
    /// subscriptionActive, subscriptionTier?}`, with the minted key as the
    /// access token and the identity token as the refresh token.
    fn credential(&self, identity: String, minted: MintedKey) -> Result<Credential> {
        let expires = self.clock.now_ms().saturating_add(
            i64::try_from(CREDENTIAL_LIFETIME.as_millis()).expect("twelve hours fit in i64"),
        );
        let mut credential = Credential::oauth(minted.api_key, identity, expires);
        credential
            .set_extra(BASE_URL_EXTRA, Value::String(minted.base_url))
            .map_err(OAuthError::Storage)?;
        credential
            .set_extra(SUBSCRIPTION_ACTIVE_EXTRA, Value::Bool(true))
            .map_err(OAuthError::Storage)?;
        if let Some(tier) = minted.subscription_tier {
            credential
                .set_extra(SUBSCRIPTION_TIER_EXTRA, Value::String(tier))
                .map_err(OAuthError::Storage)?;
        }
        Ok(credential)
    }
}

// ---------------------------------------------------------------------------
// Models (`src/models.ts`)

/// `displayName`: `muse-spark-1.3-contributor` reads "Muse Spark 1.3
/// Contributor".
fn display_name(id: &str) -> String {
    let suffix = id.get("muse-spark-".len()..).unwrap_or_default();
    let words = suffix
        .split('-')
        .map(|part| {
            if part == "contributor" {
                "Contributor"
            } else {
                part
            }
        })
        .collect::<Vec<_>>();
    format!("Muse Spark {}", words.join(" "))
}

/// The id pattern `/^muse-spark-[a-z0-9][a-z0-9._-]*$/i`, which keeps voice
/// and other non-chat models out of the picker.
fn is_muse_spark_id(id: &str) -> bool {
    let Some(prefix) = id.get(.."muse-spark-".len()) else {
        return false;
    };
    if !prefix.eq_ignore_ascii_case("muse-spark-") {
        return false;
    }
    let mut rest = id["muse-spark-".len()..].chars();
    rest.next()
        .is_some_and(|first| first.is_ascii_alphanumeric())
        && rest.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
}

fn muse_code_metadata(source: Option<&Map<String, Value>>) -> Option<&Map<String, Value>> {
    source?
        .get("metadata")?
        .as_object()?
        .get("muse-code")?
        .as_object()
}

/// `fallbackThinkingLevelMap`: thinking cannot be switched off, and only the
/// 1.3 family offers `max`.
fn fallback_thinking_level_map(id: &str) -> llm::ThinkingLevelMap {
    let supports_max = id
        .strip_prefix("muse-spark-1.3")
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'));
    let mut map = llm::ThinkingLevelMap::new();
    map.insert(llm::THINKING_OFF.to_owned(), None);
    for level in &THINKING_LEVELS[..5] {
        map.insert((*level).to_owned(), Some((*level).to_owned()));
    }
    map.insert(
        llm::THINKING_MAX.to_owned(),
        supports_max.then(|| llm::THINKING_MAX.to_owned()),
    );
    map
}

/// `thinkingLevelMap`: the reasoning efforts Meta lists as variants, or the
/// fallback when it lists none this build recognises.
fn thinking_level_map(id: &str, metadata: Option<&Map<String, Value>>) -> llm::ThinkingLevelMap {
    let Some(variants) = metadata
        .and_then(|metadata| metadata.get("variants"))
        .and_then(Value::as_object)
    else {
        return fallback_thinking_level_map(id);
    };
    let mut map = llm::ThinkingLevelMap::new();
    map.insert(llm::THINKING_OFF.to_owned(), None);
    let mut recognized = false;
    for level in THINKING_LEVELS {
        let effort = variants
            .get(level)
            .and_then(Value::as_object)
            .and_then(|variant| variant.get("reasoningEffort"))
            .and_then(Value::as_str)
            .filter(|effort| !effort.is_empty());
        recognized |= effort.is_some();
        map.insert(level.to_owned(), effort.map(str::to_owned));
    }
    if recognized {
        map
    } else {
        fallback_thinking_level_map(id)
    }
}

/// `modelInput`: text and image unless Meta narrows it.
fn model_input(metadata: Option<&Map<String, Value>>) -> Vec<String> {
    let mut input = Vec::new();
    if let Some(listed) = metadata
        .and_then(|metadata| metadata.get("modalities"))
        .and_then(Value::as_object)
        .and_then(|modalities| modalities.get("input"))
        .and_then(Value::as_array)
    {
        for value in listed.iter().filter_map(Value::as_str) {
            if matches!(value, "text" | "image") && !input.iter().any(|seen| seen == value) {
                input.push(value.to_owned());
            }
        }
    }
    if input.is_empty() {
        vec!["text".to_owned(), "image".to_owned()]
    } else {
        input
    }
}

/// `positiveInteger`, applied to the first of the candidates that is not
/// null (the extension's `??`).
fn positive_integer(candidates: [Option<&Value>; 2], fallback: u64) -> u64 {
    candidates
        .into_iter()
        .flatten()
        .find(|value| !value.is_null())
        .and_then(Value::as_f64)
        .filter(|number| *number > 0.0 && number.fract() == 0.0 && *number <= MAX_SAFE_INTEGER)
        .map_or(fallback, |number| number as u64)
}

/// `createMuseModel`: a `meta-muse` model from its id and, for a live
/// catalog entry, Meta's `metadata["muse-code"]`.
pub fn create_muse_model(id: &str, source: Option<&Map<String, Value>>) -> llm::Model {
    let metadata = muse_code_metadata(source);
    let limits = metadata
        .and_then(|metadata| metadata.get("limit"))
        .and_then(Value::as_object);
    let name = metadata
        .and_then(|metadata| metadata.get("name"))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .or_else(|| {
            source
                .and_then(|source| source.get("display_name"))
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
        })
        .map_or_else(|| display_name(id), str::to_owned);
    llm::Model {
        id: id.to_owned(),
        name,
        api: "openai-responses".to_owned(),
        provider: PROVIDER_ID.to_owned(),
        base_url: DEFAULT_API_BASE_URL.to_owned(),
        reasoning: true,
        thinking_level_map: thinking_level_map(id, metadata),
        input: model_input(metadata),
        cost: llm::ModelCost::default(),
        context_window: positive_integer(
            [
                limits.and_then(|limits| limits.get("context")),
                source.and_then(|source| source.get("context_window")),
            ],
            DEFAULT_CONTEXT_WINDOW,
        ),
        max_tokens: positive_integer(
            [
                limits.and_then(|limits| limits.get("output")),
                source.and_then(|source| source.get("max_output_tokens")),
            ],
            DEFAULT_MAX_TOKENS,
        ),
        sampling_params: None,
        headers: BTreeMap::from([("User-Agent".to_owned(), MUSE_USER_AGENT.to_owned())]),
        // Muse rejects the developer role, so the system prompt is sent as
        // a system message.
        compat: Some(json!({
            "supportsDeveloperRole": false,
            "supportsStrictMode": true,
            "supportsOpenAIGrammarTools": false,
            "supportsMaxOutputTokens": true,
            "supportsLongCacheRetention": false,
        })),
    }
}

/// `FALLBACK_MODELS`. The same five are bundled in `catalog_extra.json`; a
/// test keeps the two identical.
pub fn fallback_models() -> Vec<llm::Model> {
    FALLBACK_MODEL_IDS
        .iter()
        .map(|id| create_muse_model(id, None))
        .collect()
}

/// `parseMuseModels`: Muse Spark chat models from Meta's `/v1/models`
/// payload, deduplicated and in Meta's order.
pub fn parse_muse_models(payload: &Value) -> Result<Vec<llm::Model>> {
    let Some(entries) = payload
        .as_object()
        .and_then(|payload| payload.get("data"))
        .and_then(Value::as_array)
    else {
        return Err(failure("Meta returned an invalid model catalog"));
    };
    let mut seen = Vec::<&str>::new();
    let mut models = Vec::new();
    for entry in entries {
        let Some(entry) = entry.as_object() else {
            continue;
        };
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            continue;
        };
        if !is_muse_spark_id(id) || seen.contains(&id) {
            continue;
        }
        seen.push(id);
        models.push(create_muse_model(id, Some(entry)));
    }
    if models.is_empty() {
        return Err(failure("Meta model catalog contained no Muse Spark models"));
    }
    Ok(models)
}

/// A fetched and validated `/v1/models` payload.
pub struct LiveCatalog {
    payload: Value,
    models: Vec<llm::Model>,
}

impl LiveCatalog {
    pub fn models(&self) -> &[llm::Model] {
        &self.models
    }
}

/// `fetchMuseModels`. The transport must not follow redirects (see
/// [`models_transport`]): the request carries the API key.
pub fn fetch_live_catalog(
    transport: &dyn OAuthTransport,
    api_key: &str,
    cancellation: &CancellationToken,
) -> Result<LiveCatalog> {
    let url = Url::parse(MODELS_URL).expect("compiled models URL is valid");
    let headers = BTreeMap::from([
        ("Accept".to_owned(), "application/json".to_owned()),
        ("Authorization".to_owned(), format!("Bearer {api_key}")),
        ("User-Agent".to_owned(), MUSE_USER_AGENT.to_owned()),
        ("x-api-version".to_owned(), API_VERSION.to_owned()),
    ]);
    let response = transport.execute(
        OAuthRequest::new(Method::GET, url, headers, Vec::new(), REQUEST_TIMEOUT),
        cancellation,
    )?;
    if !is_success(response.status) {
        return Err(failure(format!(
            "Meta model catalog request failed with status {}",
            response.status
        )));
    }
    let payload: Value = serde_json::from_slice(&response.body)
        .map_err(|_| failure("Meta returned a non-JSON model catalog"))?;
    let models = parse_muse_models(&payload)?;
    Ok(LiveCatalog { payload, models })
}

/// The production transport for [`fetch_live_catalog`]: redirects are
/// returned as responses (and so fail the status check) instead of being
/// followed, matching the extension's `redirect: "error"`.
pub fn models_transport() -> Result<oauth::ReqwestOAuthTransport> {
    reqwest::blocking::Client::builder()
        .user_agent(oauth::OAUTH_USER_AGENT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map(oauth::ReqwestOAuthTransport::with_client)
        .map_err(|error| OAuthError::Transport(format!("cannot build HTTP client: {error}")))
}

/// Stores Meta's payload as received. It is parsed again on load, so the
/// cache can never hold a model the parser would not have produced.
pub fn write_cache(path: &Path, catalog: &LiveCatalog) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(&catalog.payload).map_err(std::io::Error::other)?;
    config::atomic_write(path, &bytes, 0o600)
}

/// The cached live catalog, or `None` when it is missing, oversized or no
/// longer parses; the bundled models then apply.
pub fn load_cache(path: &Path) -> Option<Vec<llm::Model>> {
    let file = fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_CACHE_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_CACHE_BYTES {
        return None;
    }
    let payload: Value = serde_json::from_slice(&bytes).ok()?;
    parse_muse_models(&payload).ok()
}

/// Fetches the live model list with the stored subscription key and caches
/// it where the catalog's dynamic layer reads it. `Ok(None)` means there is
/// nothing to do: no cache location or no `meta-muse` login.
pub fn refresh_model_cache(
    catalog: &Catalog,
    transport: &dyn OAuthTransport,
    cancellation: &CancellationToken,
) -> Result<Option<usize>> {
    let Some(path) = catalog.dynamic_paths().meta_muse_models.clone() else {
        return Ok(None);
    };
    // The extension lists models only for an OAuth credential; anything else
    // keeps the bundled list.
    let stored_login = catalog
        .credentials()
        .and_then(|store| store.read_raw(PROVIDER_ID).ok().flatten())
        .is_some_and(|credential| credential.kind() == &CredentialKind::OAuth);
    if !stored_login {
        return Ok(None);
    }
    let auth = catalog
        .resolve_auth(PROVIDER_ID)
        .map_err(|error| failure(error.to_string()))?;
    let Some(api_key) = auth.as_ref().and_then(|auth| auth.api_key()) else {
        return Ok(None);
    };
    let live = fetch_live_catalog(transport, api_key, cancellation)?;
    write_cache(&path, &live).map_err(|error| {
        failure(format!(
            "could not write the Muse Spark model list to {}: {error}",
            path.display()
        ))
    })?;
    // Seen at once even where file timestamps are coarse.
    catalog.refresh_dynamic();
    Ok(Some(live.models.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct RecordingTransport {
        requests: Mutex<Vec<OAuthRequest>>,
        response: Mutex<Option<OAuthResponse>>,
    }

    impl RecordingTransport {
        fn answering(status: u16, body: &str) -> Self {
            Self {
                requests: Mutex::new(Vec::new()),
                response: Mutex::new(Some(OAuthResponse {
                    status,
                    body: body.as_bytes().to_vec(),
                })),
            }
        }
    }

    impl OAuthTransport for RecordingTransport {
        fn execute(&self, request: OAuthRequest, _: &CancellationToken) -> Result<OAuthResponse> {
            self.requests.lock().expect("requests").push(request);
            self.response
                .lock()
                .expect("response")
                .take()
                .ok_or_else(|| OAuthError::Transport("unexpected second request".to_owned()))
        }
    }

    fn levels(model: &llm::Model) -> Vec<String> {
        crate::stream::supported_thinking_levels(model)
    }

    #[test]
    fn base_urls_are_sanitised_and_confined_to_the_model_api() {
        let sanitise = |value: Value| sanctioned_api_base_url(Some(&value));
        assert_eq!(
            sanitise(json!("https://user:secret@api.meta.ai/v2/?debug=1#frag")).expect("trusted"),
            "https://api.meta.ai/v2"
        );
        assert_eq!(
            sanitise(json!("https://API.META.AI")).expect("host is case-insensitive"),
            "https://api.meta.ai"
        );
        // Absent, empty and non-string values mean the default.
        assert_eq!(
            sanctioned_api_base_url(None).expect("absent"),
            DEFAULT_API_BASE_URL
        );
        assert_eq!(sanitise(json!("")).expect("empty"), DEFAULT_API_BASE_URL);
        assert_eq!(sanitise(json!(42)).expect("number"), DEFAULT_API_BASE_URL);
        for untrusted in [
            "http://api.meta.ai/v1",
            "https://api.meta.ai.example.com/v1",
            "https://api.meta.ai@example.com/v1",
            "https://graph.meta.ai/v1",
        ] {
            let error = sanitise(json!(untrusted)).expect_err(untrusted).to_string();
            assert_eq!(
                error, "Meta returned an untrusted Model API base URL",
                "{untrusted}"
            );
        }
        assert_eq!(
            sanitise(json!("not a url"))
                .expect_err("relative")
                .to_string(),
            "Meta returned an invalid Model API base URL"
        );
    }

    #[test]
    fn verification_pages_must_be_meta_https_pages() {
        let trusted = |value: &str| trusted_meta_url(Some(&json!(value)));
        assert_eq!(
            trusted("https://auth.meta.com/oauth/device/?code=AB").as_deref(),
            Some("https://auth.meta.com/oauth/device/?code=AB")
        );
        assert_eq!(trusted("http://auth.meta.com/oauth/device/"), None);
        assert_eq!(trusted("https://auth.meta.com.example/device"), None);
        assert_eq!(trusted("https://meta.com/device"), None);
        assert_eq!(trusted(""), None);
    }

    #[test]
    fn fallback_models_expose_only_known_reasoning_variants() {
        let models = fallback_models();
        assert_eq!(
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            FALLBACK_MODEL_IDS
        );
        let spark13 = &models[1];
        assert_eq!(spark13.name, "Muse Spark 1.3");
        assert_eq!(models[0].name, "Muse Spark 1.3 Contributor");
        assert_eq!(
            levels(spark13),
            ["minimal", "low", "medium", "high", "xhigh", "max"]
        );
        let spark12 = &models[3];
        assert_eq!(
            levels(spark12),
            ["minimal", "low", "medium", "high", "xhigh"]
        );
        assert_eq!(spark12.thinking_level_map.get("off"), Some(&None));
        assert_eq!(spark12.context_window, 1_007_997);
        assert_eq!(spark12.max_tokens, 128_000);
        assert_eq!(
            spark12.headers.get("User-Agent").map(String::as_str),
            Some(MUSE_USER_AGENT)
        );
    }

    #[test]
    fn bundled_catalog_models_match_the_extension_fallbacks() {
        let catalog = Catalog::with_environment(None, std::sync::Arc::new(|_| None))
            .expect("catalog")
            .with_dynamic_paths(crate::catalog::DynamicPaths::disabled());
        let provider = catalog.provider(PROVIDER_ID).expect("meta-muse provider");
        assert_eq!(provider.base_url, DEFAULT_API_BASE_URL);
        for expected in fallback_models() {
            let bundled = provider
                .model(&expected.id)
                .unwrap_or_else(|| panic!("{} is not bundled", expected.id));
            assert_eq!(
                serde_json::to_value(&bundled).expect("bundled JSON"),
                serde_json::to_value(&expected).expect("expected JSON"),
                "{}",
                expected.id
            );
        }
        assert_eq!(provider.models().len(), FALLBACK_MODEL_IDS.len());
    }

    #[test]
    fn parses_muse_code_capabilities_and_excludes_non_chat_models() {
        let models = parse_muse_models(&json!({
            "object": "list",
            "data": [
                {
                    "id": "muse-spark-1.4-contributor",
                    "metadata": {"muse-code": {
                        "name": "Muse Spark Preview",
                        "modalities": {"input": ["text", "audio", "text"]},
                        "limit": {"context": 500_000, "output": 64_000},
                        "variants": {
                            "minimal": {"reasoningEffort": "minimal"},
                            "high": {"reasoningEffort": "high"},
                            "max": {"reasoningEffort": "max"}
                        }
                    }}
                },
                {"id": "muse-voice-transcribe-1.0"},
                {"id": "other-model"},
                {"id": "muse-spark-1.4-contributor"},
                {"id": "muse-spark-1.4", "display_name": "Spark Four", "context_window": 2048, "max_output_tokens": 1.5},
                {"id": "muse-spark-"},
                {"id": "muse-spark-1.5 beta"},
                "not an object",
                {"id": 7}
            ]
        }))
        .expect("parsed");
        assert_eq!(
            models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            ["muse-spark-1.4-contributor", "muse-spark-1.4"]
        );
        let preview = &models[0];
        assert_eq!(preview.name, "Muse Spark Preview");
        assert_eq!(preview.provider, PROVIDER_ID);
        assert_eq!(preview.api, "openai-responses");
        assert!(preview.reasoning);
        assert_eq!(
            preview.thinking_level_map,
            BTreeMap::from([
                ("off".to_owned(), None),
                ("minimal".to_owned(), Some("minimal".to_owned())),
                ("low".to_owned(), None),
                ("medium".to_owned(), None),
                ("high".to_owned(), Some("high".to_owned())),
                ("xhigh".to_owned(), None),
                ("max".to_owned(), Some("max".to_owned())),
            ])
        );
        assert_eq!(preview.input, ["text"]);
        assert_eq!(preview.context_window, 500_000);
        assert_eq!(preview.max_tokens, 64_000);

        // Without metadata: display_name, the top-level limits, and a
        // non-integer limit falling back to the default.
        let plain = &models[1];
        assert_eq!(plain.name, "Spark Four");
        assert_eq!(plain.context_window, 2048);
        assert_eq!(plain.max_tokens, 128_000);
        assert_eq!(plain.input, ["text", "image"]);
        assert_eq!(levels(plain), ["minimal", "low", "medium", "high", "xhigh"]);
    }

    #[test]
    fn unrecognised_variants_fall_back_to_the_known_levels() {
        let models = parse_muse_models(&json!({
            "data": [{"id": "muse-spark-1.3", "metadata": {"muse-code": {"variants": {}}}}]
        }))
        .expect("parsed");
        assert_eq!(
            levels(&models[0]),
            ["minimal", "low", "medium", "high", "xhigh", "max"]
        );
    }

    #[test]
    fn a_catalog_without_muse_spark_models_is_refused() {
        let error = parse_muse_models(&json!({
            "object": "list",
            "data": [{"id": "muse-voice-transcribe-1.0"}]
        }))
        .expect_err("nothing usable");
        assert!(
            error.to_string().contains("no Muse Spark models"),
            "{error}"
        );
        assert_eq!(
            parse_muse_models(&json!({"data": "nope"}))
                .expect_err("not a list")
                .to_string(),
            "Meta returned an invalid model catalog"
        );
    }

    #[test]
    fn fetches_the_authenticated_model_catalog() {
        let transport = RecordingTransport::answering(200, r#"{"data":[{"id":"muse-spark-1.3"}]}"#);
        let live = fetch_live_catalog(&transport, "model-api-key", &CancellationToken::new())
            .expect("catalog");
        assert_eq!(live.models().len(), 1);
        let requests = transport.requests.lock().expect("requests");
        let request = &requests[0];
        assert_eq!(request.method(), &Method::GET);
        assert_eq!(request.url().as_str(), MODELS_URL);
        assert_eq!(request.timeout(), Duration::from_secs(30));
        assert_eq!(
            request.headers(),
            &BTreeMap::from([
                ("Accept".to_owned(), "application/json".to_owned()),
                (
                    "Authorization".to_owned(),
                    "Bearer model-api-key".to_owned()
                ),
                ("User-Agent".to_owned(), MUSE_USER_AGENT.to_owned()),
                ("x-api-version".to_owned(), "1.0.0".to_owned()),
            ])
        );
    }

    #[test]
    fn catalog_request_failures_name_the_status_or_shape() {
        // A redirect is a failure: the no-redirect transport hands it back.
        let redirected = RecordingTransport::answering(302, "");
        assert_eq!(
            fetch_live_catalog(&redirected, "key", &CancellationToken::new())
                .err()
                .expect("redirect refused")
                .to_string(),
            "Meta model catalog request failed with status 302"
        );
        let html = RecordingTransport::answering(200, "<html>");
        assert_eq!(
            fetch_live_catalog(&html, "key", &CancellationToken::new())
                .err()
                .expect("HTML refused")
                .to_string(),
            "Meta returned a non-JSON model catalog"
        );
    }

    #[test]
    fn the_cache_round_trips_and_ignores_unusable_files() {
        let directory = std::env::temp_dir().join(format!(
            "goshcoder-meta-muse-cache-{}-{}",
            std::process::id(),
            uuid::Uuid::now_v7()
        ));
        let path = config::meta_muse_models_path_in(&directory);
        assert!(load_cache(&path).is_none(), "missing file");

        let transport = RecordingTransport::answering(
            200,
            r#"{"data":[{"id":"muse-spark-2.0","metadata":{"muse-code":{"name":"Muse Spark Two"}}}]}"#,
        );
        let live =
            fetch_live_catalog(&transport, "key", &CancellationToken::new()).expect("catalog");
        write_cache(&path, &live).expect("write cache");
        let cached = load_cache(&path).expect("cached models");
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].name, "Muse Spark Two");

        fs::write(&path, r#"{"data":[{"id":"muse-voice-1"}]}"#).expect("overwrite");
        assert!(load_cache(&path).is_none(), "no usable model");
        fs::write(&path, vec![b' '; (MAX_CACHE_BYTES + 1) as usize]).expect("oversize");
        assert!(load_cache(&path).is_none(), "oversized cache");
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn stored_credentials_without_subscription_fields_are_incomplete() {
        let mut credential = Credential::oauth("model-key", "identity", i64::MAX);
        let error = request_auth(&credential).expect_err("incomplete");
        assert_eq!(error.to_string(), INCOMPLETE_CREDENTIAL);
        assert!(error.is_unauthorized());

        credential
            .set_extra(BASE_URL_EXTRA, json!("https://api.meta.ai/v1"))
            .expect("extra");
        assert!(
            request_auth(&credential).is_err(),
            "subscriptionActive missing"
        );
        credential
            .set_extra(SUBSCRIPTION_ACTIVE_EXTRA, json!("yes"))
            .expect("extra");
        assert!(request_auth(&credential).is_err(), "not a boolean");
        credential
            .set_extra(SUBSCRIPTION_ACTIVE_EXTRA, json!(true))
            .expect("extra");
        assert_eq!(
            request_auth(&credential).ok(),
            Some(("model-key".to_owned(), "https://api.meta.ai/v1".to_owned()))
        );

        credential
            .set_extra(BASE_URL_EXTRA, json!("https://example.com/v1"))
            .expect("extra");
        let error = request_auth(&credential).expect_err("untrusted");
        assert!(error.is_unauthorized());
        assert!(error.to_string().contains("untrusted"), "{error}");
    }
}
