//! Grok CLI provider: an X Premium / SuperGrok subscription used through the
//! endpoint the official Grok CLI talks to.
//!
//! Native adaptation of the pi extension pi-grok-cli v0.9.3 (J Liew /
//! kenryu42, MIT, https://github.com/kenryu42/pi-grok-cli), written against
//! its `src/provider/register.ts`, `src/provider/stream.ts`,
//! `src/provider/proxyRetry.ts`, `src/provider/sessionConvId.ts`,
//! `src/payload/sanitize.ts`, `src/models/catalog.ts` and
//! `src/auth/config.ts`. No grok binary is involved: requests impersonate the
//! official client's identification headers against
//! `cli-chat-proxy.grok.com`, which accepts subscription tokens that
//! `api.x.ai` often refuses with 403.
//!
//! The pieces pi wires through extension hooks live here as plain functions
//! the request path calls: the static and per-request headers, the client
//! version lookup with its HTTP 426 refresh, the per-session conversation id
//! with its rotation on proxy rejections, and the Responses payload
//! sanitisation. The OAuth flow is the xAI one with the deltas in
//! `oauth.rs`; usage and Grok Imagine are further down this file and in
//! `grok_imagine.rs`.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use serde_json::{Map, Value, json};

use crate::{llm, session};

pub const PROVIDER_ID: &str = "grok-cli";
pub const PROVIDER_NAME: &str = "Grok CLI";
pub const DEFAULT_BASE_URL: &str = "https://cli-chat-proxy.grok.com/v1";
/// Base URL overrides, first match wins. The first two are upstream's; the
/// third follows GoshCoder's own naming.
pub const BASE_URL_ENV: [&str; 3] = [
    "PI_GROK_CLI_BASE_URL",
    "GROK_CLI_BASE_URL",
    "GOSHCODER_GROK_CLI_BASE_URL",
];
/// A bearer token that bypasses login: no refresh, and it wins over a stored
/// login (upstream resolves it before the account vault).
pub const TOKEN_ENV: &str = "GROK_CLI_OAUTH_TOKEN";
pub const MODELS_ENV: &str = "PI_GROK_CLI_MODELS";
pub const VERSION_URL: &str = "https://x.ai/cli/stable";
pub const VERSION_URL_ENV: &str = "PI_GROK_CLI_VERSION_URL";
/// Grok CLI release sent when the latest stable release cannot be looked up.
/// Upstream keeps it at a current official release so it stays above the
/// endpoint's minimum.
pub const FALLBACK_VERSION: &str = "1.0.46";
const VERSION_TIMEOUT: Duration = Duration::from_secs(5);
/// Session custom entry holding the conversation-id generation; the latest
/// valid one on the current branch wins.
pub const CONV_ENTRY: &str = "grok-cli-conv-id-v1";
const MAX_CONV_ROTATIONS: u32 = 2;
const ENCRYPTED_REASONING_INCLUDE: &str = "reasoning.encrypted_content";
/// Models whose names start with one of these accept `reasoning.effort`.
const EFFORT_CAPABLE_PREFIXES: &[&str] = &[
    "grok-3-mini",
    "grok-4.20-multi-agent",
    "grok-4.3",
    "grok-4.5",
    "grok-4.6",
    "grok-4.7",
];

/// The configured base URL without trailing slashes.
pub fn base_url(lookup: impl Fn(&str) -> Option<String>) -> String {
    BASE_URL_ENV
        .iter()
        .find_map(|name| lookup(name).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
        .trim()
        .trim_end_matches('/')
        .to_owned()
}

/// Static identification headers carried by every model definition, so they
/// reach the wire on every request. The version headers are per request.
pub fn model_headers(model_id: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "x-grok-client-identifier".to_owned(),
            "grok-shell".to_owned(),
        ),
        ("x-xai-token-auth".to_owned(), "xai-grok-cli".to_owned()),
        ("x-grok-model-override".to_owned(), model_id.to_owned()),
    ])
}

/// The official client's own version headers. The endpoint gates on
/// `x-grok-client-version` (HTTP 426 when missing or too old) and ignores
/// the user agent, which is sent in the same shape for fidelity.
pub fn version_headers(version: &str) -> [(&'static str, String); 2] {
    [
        (
            "user-agent",
            format!("grok-shell/{version} (macos; aarch64)"),
        ),
        ("x-grok-client-version", version.to_owned()),
    ]
}

/// `^\d+\.\d+\.\d+(-[\w.]+)?$`, without a regex dependency.
pub fn is_valid_version(value: &str) -> bool {
    let (core, suffix) = match value.split_once('-') {
        Some((core, suffix)) => (core, Some(suffix)),
        None => (value, None),
    };
    let parts = core.split('.').collect::<Vec<_>>();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
        && suffix.is_none_or(|suffix| {
            !suffix.is_empty()
                && suffix
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'.')
        })
}

// ---------------------------------------------------------------------------
// Models

/// Applies `PI_GROK_CLI_MODELS`: a comma-separated list that filters and
/// reorders the catalog models. An id the catalog does not carry gets
/// upstream's generic definition, so a newly launched model is usable before
/// a release names it.
pub fn filter_models(
    models: Vec<llm::Model>,
    spec: Option<&str>,
    base_url: &str,
) -> Vec<llm::Model> {
    let requested = spec
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .collect::<Vec<_>>();
    if requested.is_empty() {
        return models;
    }
    let mut seen = std::collections::BTreeSet::new();
    requested
        .into_iter()
        .filter(|id| seen.insert(*id))
        .map(|id| {
            models
                .iter()
                .find(|model| model.id == id)
                .cloned()
                .unwrap_or_else(|| generic_model(id, base_url))
        })
        .collect()
}

fn generic_model(id: &str, base_url: &str) -> llm::Model {
    llm::Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: "openai-responses".to_owned(),
        provider: PROVIDER_ID.to_owned(),
        base_url: base_url.to_owned(),
        reasoning: true,
        input: vec!["text".to_owned()],
        cost: llm::ModelCost {
            rates: llm::ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.2,
                cache_write: 0.2,
            },
            tiers: Vec::new(),
        },
        context_window: 1_000_000,
        max_tokens: 30_000,
        headers: model_headers(id),
        ..llm::Model::default()
    }
}

fn normalized_model_name(model_id: &str) -> String {
    model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .to_ascii_lowercase()
}

/// upstream `supportsReasoningEffort`: the name must carry an effort-capable
/// prefix, the model must reason, and a level map (when present) must map
/// at least one level to a real effort.
pub fn supports_reasoning_effort(model: &llm::Model) -> bool {
    let name = normalized_model_name(&model.id);
    if !EFFORT_CAPABLE_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
    {
        return false;
    }
    if !model.reasoning {
        return false;
    }
    model.thinking_level_map.is_empty()
        || model
            .thinking_level_map
            .values()
            .any(|level| level.as_deref().is_some_and(|level| level != "none"))
}

// ---------------------------------------------------------------------------
// Client version

/// One cached stable-release lookup per pointer URL. Keying by URL keeps a
/// test's loopback pointer from sharing state with the real one.
#[derive(Default)]
struct VersionCell {
    current: Mutex<Option<String>>,
}

fn version_cell(url: &str) -> Arc<VersionCell> {
    static CELLS: OnceLock<Mutex<HashMap<String, Arc<VersionCell>>>> = OnceLock::new();
    let mut cells = lock(CELLS.get_or_init(Default::default));
    Arc::clone(cells.entry(url.to_owned()).or_default())
}

/// The latest stable Grok CLI release, read once per process from the
/// pointer the official installer uses, so requests stay above a raised
/// minimum without a GoshCoder release. Holding the cell's lock across the
/// lookup makes concurrent first requests share it.
pub fn resolve_version(url: &str) -> String {
    let cell = version_cell(url);
    let mut current = lock(&cell.current);
    if let Some(version) = current.as_ref() {
        return version.clone();
    }
    let version = fetch_version(url);
    *current = Some(version.clone());
    version
}

/// Looks the release up again after the gate rejected `rejected`. When a
/// concurrent request already replaced it, that newer value is used instead
/// of a second lookup, so a failed lookup cannot overwrite a good one.
pub fn refresh_version(url: &str, rejected: &str) -> String {
    let cell = version_cell(url);
    let mut current = lock(&cell.current);
    if let Some(version) = current.as_ref().filter(|version| *version != rejected) {
        return version.clone();
    }
    let version = fetch_version(url);
    *current = Some(version.clone());
    version
}

fn fetch_version(url: &str) -> String {
    let fetched = reqwest::blocking::Client::builder()
        .timeout(VERSION_TIMEOUT)
        .user_agent(crate::oauth::OAUTH_USER_AGENT)
        .build()
        .and_then(|client| client.get(url).send())
        .and_then(|response| response.error_for_status())
        .and_then(|response| response.text());
    match fetched {
        Ok(text) if is_valid_version(text.trim()) => text.trim().to_owned(),
        _ => FALLBACK_VERSION.to_owned(),
    }
}

// ---------------------------------------------------------------------------
// Conversation id

/// Where a session's conversation-id generation lives. The session-backed
/// implementation reads the latest entry on the current branch, so resume,
/// branch navigation and forks pick up the right generation without the
/// session_start/session_tree hooks upstream needs.
pub trait ConvStore: Send + Sync {
    /// The id of the session currently open, which may differ from the id
    /// the store was registered under after `/resume` or `/new`.
    fn session_id(&self) -> Option<String>;
    fn generation(&self) -> Option<u64>;
    /// Persists `generation`. `Ok(false)` means nothing is being recorded
    /// (no session file, read-only), in which case the rotation lives in
    /// memory for this process, as upstream's in-memory session manager does.
    fn record(&self, generation: u64) -> Result<bool, String>;
}

impl ConvStore for session::SessionCustomRecorder {
    fn session_id(&self) -> Option<String> {
        self.handle().map(|handle| handle.id)
    }

    fn generation(&self) -> Option<u64> {
        self.latest_custom(CONV_ENTRY, stored_generation)
    }

    fn record(&self, generation: u64) -> Result<bool, String> {
        if !self.recording() {
            return Ok(false);
        }
        self.record(CONV_ENTRY, json!({ "generation": generation }))
            .map(|_| true)
            .map_err(|error| error.to_string())
    }
}

/// upstream `storedGeneration`: a positive safe integer under `generation`.
pub fn stored_generation(data: &Value) -> Option<u64> {
    const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
    data.as_object()?
        .get("generation")?
        .as_u64()
        .filter(|generation| (1..=MAX_SAFE_INTEGER).contains(generation))
}

#[derive(Default)]
struct ConvRegistry {
    stores: HashMap<String, Arc<dyn ConvStore>>,
    /// Rotations that could not be recorded, by effective session id.
    memory: HashMap<String, u64>,
}

fn conv_registry() -> &'static Mutex<ConvRegistry> {
    static REGISTRY: OnceLock<Mutex<ConvRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

/// Keeps a session's store registered for as long as the session lives.
pub struct ConvRegistration {
    key: String,
}

impl ConvRegistration {
    /// The request session id the store answers for.
    pub fn key(&self) -> &str {
        &self.key
    }
}

impl Drop for ConvRegistration {
    fn drop(&mut self) {
        lock(conv_registry()).stores.remove(&self.key);
    }
}

/// Registers the store behind requests whose `session_id` is `key`.
pub fn register_conv_store(key: &str, store: Arc<dyn ConvStore>) -> ConvRegistration {
    lock(conv_registry()).stores.insert(key.to_owned(), store);
    ConvRegistration {
        key: key.to_owned(),
    }
}

/// The session id requests for `request_session` belong to now.
pub fn effective_session_id(request_session: &str) -> String {
    let store = lock(conv_registry()).stores.get(request_session).cloned();
    store
        .and_then(|store| store.session_id())
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| request_session.to_owned())
}

fn current_generation(request_session: &str) -> (String, u64) {
    let (store, remembered) = {
        let registry = lock(conv_registry());
        let store = registry.stores.get(request_session).cloned();
        let effective = store
            .as_ref()
            .and_then(|store| store.session_id())
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| request_session.to_owned());
        let remembered = registry.memory.get(&effective).copied();
        (store.map(|store| (effective, store)), remembered)
    };
    match store {
        Some((effective, store)) => {
            let generation = remembered.or_else(|| store.generation()).unwrap_or(0);
            (effective, generation)
        }
        None => (request_session.to_owned(), remembered.unwrap_or(0)),
    }
}

fn format_conv_id(session: &str, generation: u64) -> String {
    if generation == 0 {
        session.to_owned()
    } else {
        format!("{session}:{generation}")
    }
}

/// `x-grok-conv-id`: the session id, then `<id>:<generation>` once a
/// rotation has happened. `None` without a session, as upstream sends the
/// header only when pi supplies one.
pub fn conv_id(request_session: &str) -> Option<String> {
    if request_session.is_empty() {
        return None;
    }
    let (session, generation) = current_generation(request_session);
    Some(format_conv_id(&session, generation))
}

/// Starts a new conversation on the proxy side by bumping the generation.
pub fn rotate_conv(request_session: &str) -> Result<String, String> {
    if request_session.is_empty() {
        return Err("no session is active".to_owned());
    }
    let store = lock(conv_registry()).stores.get(request_session).cloned();
    let (session, generation) = current_generation(request_session);
    let next = generation.saturating_add(1);
    let recorded = match store {
        Some(store) => store.record(next)?,
        None => false,
    };
    let mut registry = lock(conv_registry());
    if recorded {
        registry.memory.remove(&session);
    } else {
        registry.memory.insert(session.clone(), next);
    }
    Ok(format_conv_id(&session, next))
}

/// `/grok-cli-conv [status|rotate]`.
pub fn conv_command(request_session: &str, argument: &str) -> Result<String, String> {
    let argument = argument.trim().to_ascii_lowercase();
    match argument.as_str() {
        "" | "status" => Ok(format!(
            "Grok CLI conversation ID: {}",
            conv_id(request_session).unwrap_or_else(|| "(no session)".to_owned())
        )),
        "rotate" => rotate_conv(request_session)
            .map(|id| format!("Grok CLI conversation ID rotated to {id}")),
        _ => Err("Usage: /grok-cli-conv [status|rotate]".to_owned()),
    }
}

// ---------------------------------------------------------------------------
// Request path

/// Settings the request path reads from the catalog's injected environment,
/// so tests point the version lookup at a loopback server.
#[derive(Clone, Debug, Default)]
pub struct RequestSettings {
    pub version_url: Option<String>,
}

impl RequestSettings {
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            version_url: lookup(VERSION_URL_ENV).filter(|url| !url.trim().is_empty()),
        }
    }

    pub fn from_process_environment() -> Self {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn version_url(&self) -> String {
        self.version_url
            .clone()
            .unwrap_or_else(|| VERSION_URL.to_owned())
    }
}

/// One logical request: its version and conversation-id state across the
/// retries proxyRetry.ts allows before any stream event. The generic
/// provider retry is off for this provider (upstream passes `maxRetries: 0`)
/// because it would resend the rejected conversation id.
pub struct RequestAttempt {
    version_url: String,
    session: String,
    version: String,
    version_refreshed: bool,
    rotations: u32,
}

impl RequestAttempt {
    pub fn begin(settings: &RequestSettings, request_session: &str) -> Self {
        let version_url = settings.version_url();
        let version = resolve_version(&version_url);
        Self {
            version_url,
            session: request_session.to_owned(),
            version,
            version_refreshed: false,
            rotations: 0,
        }
    }

    /// Headers for the next send; the conversation id is read fresh so a
    /// rotation, or one made by `/grok-cli-conv rotate`, applies at once.
    pub fn headers(&self) -> Vec<(&'static str, String)> {
        let mut headers = version_headers(&self.version).to_vec();
        if let Some(conv_id) = conv_id(&self.session) {
            headers.push(("x-grok-conv-id", conv_id));
        }
        headers
    }

    /// Decides whether a rejection that arrived before any stream event is
    /// retried: 426 refreshes the version once; 401, 502 and 520 (a stale or
    /// poisoned proxy conversation) rotate the conversation id up to twice.
    /// A failed session write keeps the original error, as upstream does.
    pub fn retry_after(&mut self, status: u16) -> bool {
        match status {
            426 if !self.version_refreshed => {
                self.version_refreshed = true;
                self.version = refresh_version(&self.version_url, &self.version);
                true
            }
            401 | 502 | 520 if self.rotations < MAX_CONV_ROTATIONS && !self.session.is_empty() => {
                match rotate_conv(&self.session) {
                    Ok(_) => {
                        self.rotations += 1;
                        true
                    }
                    Err(_) => false,
                }
            }
            _ => false,
        }
    }
}

/// Brings the generic Responses body to what pi's builder sends for a
/// reasoning model with thinking off (`reasoning.effort` from the model's
/// `off` mapping, else `"none"`), which GoshCoder's shared builder omits,
/// and then applies upstream's sanitisation.
pub fn prepare_payload(
    payload: &mut Value,
    model: &llm::Model,
    thinking_level: &str,
    request_session: &str,
) {
    if let Some(body) = payload.as_object_mut()
        && model.reasoning
        && !body.contains_key("reasoning")
        && crate::stream::clamp_thinking_level(model, thinking_level) == llm::THINKING_OFF
        && model.thinking_level_map.get(llm::THINKING_OFF) != Some(&None)
    {
        let effort = model
            .thinking_level_map
            .get(llm::THINKING_OFF)
            .cloned()
            .flatten()
            .unwrap_or_else(|| "none".to_owned());
        body.insert("reasoning".to_owned(), json!({ "effort": effort }));
    }
    let session = (!request_session.is_empty()).then(|| effective_session_id(request_session));
    sanitize_payload(payload, model, session.as_deref());
}

// ---------------------------------------------------------------------------
// Payload sanitisation (upstream src/payload/sanitize.ts)

fn text_from_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| match part {
                Value::String(text) => Some(text.clone()),
                Value::Object(item) => {
                    let kind = item.get("type").and_then(Value::as_str).unwrap_or_default();
                    matches!(kind, "text" | "input_text" | "output_text")
                        .then(|| item.get("text").and_then(Value::as_str))
                        .flatten()
                        .map(str::to_owned)
                }
                _ => None,
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn strip_shell_quotes(value: &str) -> &str {
    let trimmed = value.trim();
    if trimmed.len() >= 2
        && ((trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'')))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

/// Upstream also resolves local file paths to data URIs (confined to the
/// workspace). GoshCoder's request builders only ever emit data URIs, so
/// the path branch has nothing to do here and is not ported; a value that
/// is not a URL is passed through for the endpoint to judge.
fn normalize_image_input(value: Option<&Value>) -> Option<String> {
    let value = value?.as_str()?;
    if value.trim().is_empty() {
        return None;
    }
    let cleaned = strip_shell_quotes(value);
    let lower = cleaned.to_ascii_lowercase();
    (lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("data:image/"))
    .then(|| cleaned.to_owned())
}

fn image_url_and_detail(object: &Map<String, Value>) -> (Option<Value>, Option<Value>) {
    match object.get("image_url") {
        Some(Value::Object(image_url)) => (
            image_url.get("url").cloned(),
            image_url.get("detail").cloned(),
        ),
        other => (other.cloned(), object.get("detail").cloned()),
    }
}

fn non_empty_string(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn normalize_image_parts(value: Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.into_iter().map(normalize_image_parts).collect()),
        Value::Object(mut object) => {
            let kind = object
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_owned);
            if kind.as_deref() == Some("image")
                && let (Some(Value::String(data)), Some(Value::String(mime))) =
                    (object.get("data"), object.get("mimeType"))
            {
                let detail = non_empty_string(object.get("detail")).unwrap_or("auto");
                return json!({
                    "type": "input_image",
                    "image_url": format!("data:{mime};base64,{data}"),
                    "detail": detail,
                });
            }
            if kind.as_deref() == Some("image_url") {
                let (url, detail) = image_url_and_detail(&object);
                object.insert("type".to_owned(), Value::String("input_image".to_owned()));
                object.insert("image_url".to_owned(), url.unwrap_or(Value::Null));
                if let Some(detail) = non_empty_string(detail.as_ref()) {
                    object.insert("detail".to_owned(), Value::String(detail.to_owned()));
                }
            }
            if object.get("type").and_then(Value::as_str) == Some("input_image") {
                let (url, detail) = image_url_and_detail(&object);
                if let Some(normalized) = normalize_image_input(url.as_ref()) {
                    object.insert("image_url".to_owned(), Value::String(normalized));
                }
                if let Some(detail) = non_empty_string(detail.as_ref()) {
                    object.insert("detail".to_owned(), Value::String(detail.to_owned()));
                }
                if non_empty_string(object.get("detail")).is_none() {
                    object.insert("detail".to_owned(), Value::String("auto".to_owned()));
                }
            }
            for field in ["content", "output"] {
                if let Some(Value::Array(_)) = object.get(field) {
                    let nested = object.remove(field).unwrap_or(Value::Null);
                    object.insert(field.to_owned(), normalize_image_parts(nested));
                }
            }
            Value::Object(object)
        }
        other => other,
    }
}

fn is_input_image(value: &Value) -> bool {
    value.get("type").and_then(Value::as_str) == Some("input_image")
}

/// The endpoint rejects image arrays in `function_call_output.output`: the
/// text stays there and the images move into a user message right after it.
fn rewrite_function_call_output(input: Vec<Value>) -> Vec<Value> {
    let mut rewritten = Vec::with_capacity(input.len());
    for item in input {
        let is_rewrite = item.get("type").and_then(Value::as_str) == Some("function_call_output")
            && item.get("output").is_some_and(Value::is_array);
        if !is_rewrite {
            rewritten.push(item);
            continue;
        }
        let Value::Object(mut object) = item else {
            unreachable!("checked above");
        };
        let Some(Value::Array(parts)) = object.remove("output") else {
            unreachable!("checked above");
        };
        let (images, texts): (Vec<Value>, Vec<Value>) = parts.into_iter().partition(is_input_image);
        let text = texts
            .iter()
            .filter_map(|part| match part {
                Value::String(text) => Some(text.as_str()),
                Value::Object(part) => part.get("text").and_then(Value::as_str),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let text = if text.is_empty() {
            "(tool returned no text output)".to_owned()
        } else {
            text
        };
        let call_id = match object.get("call_id") {
            Some(Value::String(id)) if !id.is_empty() => format!(" ({id})"),
            Some(Value::Number(id)) => format!(" ({id})"),
            Some(Value::Bool(true)) => " (true)".to_owned(),
            _ => String::new(),
        };
        object.insert("output".to_owned(), Value::String(text));
        rewritten.push(Value::Object(object));
        if !images.is_empty() {
            let plural = if images.len() == 1 { "" } else { "s" };
            let label = format!(
                "The previous tool result{call_id} included {} image{plural}. Use the attached image{plural} as the visual output from that tool.",
                images.len()
            );
            let mut content = vec![json!({"type": "input_text", "text": label})];
            content.extend(images);
            rewritten.push(json!({"role": "user", "content": content}));
        }
    }
    rewritten
}

/// Replayed reasoning items must carry typed `reasoning_text` content.
fn normalize_reasoning_content(content: Option<&Value>) -> Option<Value> {
    let normalized = match content? {
        Value::String(text) if text.is_empty() => return None,
        Value::String(text) => vec![json!({"type": "reasoning_text", "text": text})],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| match part {
                Value::String(text) if !text.is_empty() => {
                    Some(json!({"type": "reasoning_text", "text": text}))
                }
                // A typed part is kept only when it is already
                // `reasoning_text`; an untyped one with text is typed.
                Value::Object(object) if object.get("text").is_some_and(Value::is_string) => {
                    match object.get("type") {
                        Some(Value::String(kind)) if kind == "reasoning_text" => Some(part.clone()),
                        None => {
                            let mut object = object.clone();
                            object.insert(
                                "type".to_owned(),
                                Value::String("reasoning_text".to_owned()),
                            );
                            Some(Value::Object(object))
                        }
                        Some(_) => None,
                    }
                }
                _ => None,
            })
            .collect(),
        _ => return None,
    };
    (!normalized.is_empty()).then_some(Value::Array(normalized))
}

/// JavaScript truthiness for the values `.filter(Boolean)` drops.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64().is_some_and(|value| value != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(_) | Value::Object(_) => true,
    }
}

/// Rewrites a Responses body into the shape the Grok CLI endpoint accepts.
/// Every step mirrors upstream `sanitizePayload`, in order.
pub fn sanitize_payload(payload: &mut Value, model: &llm::Model, session_id: Option<&str>) {
    let Some(body) = payload.as_object_mut() else {
        return;
    };

    // A plain-string `input` is valid and stays string-shaped.
    if let Some(Value::Array(items)) = body
        .get("input")
        .is_some_and(Value::is_array)
        .then(|| body.remove("input"))
        .flatten()
    {
        let mut input = items
            .into_iter()
            .filter_map(|item| {
                let Value::Object(mut object) = item else {
                    return truthy(&item).then_some(item);
                };
                if object.get("type").and_then(Value::as_str) == Some("reasoning") {
                    object.remove("status");
                    match normalize_reasoning_content(object.get("content")) {
                        Some(content) => {
                            object.insert("content".to_owned(), content);
                        }
                        None => {
                            object.remove("content");
                        }
                    }
                }
                // The endpoint fails validation on an empty-string content.
                if object.get("content").and_then(Value::as_str) == Some("") {
                    return None;
                }
                Some(Value::Object(object))
            })
            .collect::<Vec<_>>();

        // `role: system|developer` is rejected inside `input`; it moves to
        // the top-level instructions instead.
        let mut instructions = Vec::new();
        input.retain(|item| {
            let role = item.get("role").and_then(Value::as_str);
            if !matches!(role, Some("developer" | "system")) {
                return true;
            }
            let text = text_from_content(item.get("content"));
            let text = text.trim();
            if !text.is_empty() {
                instructions.push(text.to_owned());
            }
            false
        });
        if !instructions.is_empty() {
            let existing = body
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let merged = std::iter::once(existing)
                .chain(instructions)
                .filter(|part| !part.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            body.insert("instructions".to_owned(), Value::String(merged));
        }

        let Value::Array(input) = normalize_image_parts(Value::Array(input)) else {
            unreachable!("an array normalizes to an array");
        };
        body.insert(
            "input".to_owned(),
            Value::Array(rewrite_function_call_output(input)),
        );
    }

    // xAI reads `text.format`, not OpenAI's `response_format`.
    if body.get("response_format").is_some_and(truthy) {
        let format = body.remove("response_format").unwrap_or(Value::Null);
        if !body.get("text").is_some_and(truthy) {
            body.insert("text".to_owned(), json!({ "format": format }));
        }
    }

    let reasoning_supported = model.reasoning;
    let reasoning = match body.remove("reasoning") {
        Some(Value::Object(reasoning)) => Some(reasoning),
        _ => None,
    };
    body.remove("reasoningEffort");
    if let Some(mut reasoning) = reasoning.filter(|_| reasoning_supported) {
        if supports_reasoning_effort(model) {
            if reasoning.get("effort").and_then(Value::as_str) == Some("minimal") {
                reasoning.insert("effort".to_owned(), Value::String("low".to_owned()));
            }
        } else {
            reasoning.remove("effort");
        }
        if !reasoning.is_empty() {
            body.insert("reasoning".to_owned(), Value::Object(reasoning));
        }
    }

    let has_reasoning = body.contains_key("reasoning");
    match body.get("include") {
        Some(Value::Array(include)) => {
            let mut kept_encrypted = false;
            let mut filtered = include
                .iter()
                .filter(|item| {
                    if item.as_str() != Some(ENCRYPTED_REASONING_INCLUDE) {
                        return true;
                    }
                    if !reasoning_supported || kept_encrypted {
                        return false;
                    }
                    kept_encrypted = true;
                    true
                })
                .cloned()
                .collect::<Vec<_>>();
            if has_reasoning && !kept_encrypted {
                filtered.push(Value::String(ENCRYPTED_REASONING_INCLUDE.to_owned()));
            }
            if filtered.is_empty() {
                body.remove("include");
            } else {
                body.insert("include".to_owned(), Value::Array(filtered));
            }
        }
        _ if has_reasoning => {
            body.insert("include".to_owned(), json!([ENCRYPTED_REASONING_INCLUDE]));
        }
        _ => {}
    }

    body.remove("prompt_cache_retention");

    // Conversation caching routes a session to the same server.
    if let Some(session_id) = session_id.filter(|id| !id.is_empty())
        && !body.get("prompt_cache_key").is_some_and(truthy)
    {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(session_id.to_owned()),
        );
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::{Arc, Mutex},
        thread,
    };

    use serde_json::{Value, json};

    use super::*;
    use crate::catalog::Catalog;

    fn model(id: &str) -> llm::Model {
        Catalog::with_environment(None, Arc::new(|_| None))
            .expect("catalog")
            .model(PROVIDER_ID, id)
            .unwrap_or_else(|| panic!("grok-cli/{id} is in the catalog"))
    }

    fn sanitized(mut payload: Value, model_id: &str, session: Option<&str>) -> Value {
        sanitize_payload(&mut payload, &model(model_id), session);
        payload
    }

    #[test]
    fn instructions_reasoning_and_unsupported_fields_follow_upstream() {
        let payload = sanitized(
            json!({
                "instructions": "existing instruction",
                "input": [
                    {"role": "system", "content": "system instruction"},
                    {"role": "developer", "content": [
                        {"type": "input_text", "text": "developer instruction"},
                        {"type": "output_text", "text": "output text instruction"}
                    ]},
                    {"type": "reasoning", "content": "cached reasoning", "status": "completed"},
                    {"role": "user", "content": ""},
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": "later system instruction"}
                ],
                "include": ["reasoning.encrypted_content", "message.output_text"],
                "prompt_cache_retention": "24h",
                "reasoning": {"effort": "minimal", "summary": "auto"},
                "response_format": {"type": "json_object"}
            }),
            "grok-4.3",
            Some("session-123"),
        );
        assert_eq!(
            payload["instructions"],
            "existing instruction\n\nsystem instruction\n\ndeveloper instruction\noutput text instruction\n\nlater system instruction"
        );
        assert_eq!(
            payload["input"],
            json!([
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "cached reasoning"}]},
                {"role": "user", "content": "hello"}
            ])
        );
        assert_eq!(
            payload["include"],
            json!(["reasoning.encrypted_content", "message.output_text"])
        );
        assert!(payload.get("prompt_cache_retention").is_none());
        assert_eq!(
            payload["reasoning"],
            json!({"effort": "low", "summary": "auto"})
        );
        assert_eq!(payload["text"], json!({"format": {"type": "json_object"}}));
        assert!(payload.get("response_format").is_none());
        assert_eq!(payload["prompt_cache_key"], "session-123");
    }

    #[test]
    fn reasoning_items_keep_encrypted_content_and_lose_untyped_parts() {
        let payload = sanitized(
            json!({
                "input": [{
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "summary": [{"type": "summary_text", "text": "summary"}],
                    "content": [
                        "plain text", null, 42, ["nested"], {"ignored": true},
                        {"text": "missing discriminator"},
                        {"type": "future_reasoning_type", "text": "keep discriminator"},
                        {"type": null, "text": "null discriminator"}
                    ],
                    "encrypted_content": "encrypted-reasoning",
                    "status": "completed",
                    "future_field": {"keep": true}
                }],
                "include": ["reasoning.encrypted_content"]
            }),
            "grok-build",
            Some("session-123"),
        );
        assert_eq!(
            payload["input"],
            json!([{
                "type": "reasoning",
                "id": "reasoning-1",
                "summary": [{"type": "summary_text", "text": "summary"}],
                "content": [
                    {"type": "reasoning_text", "text": "plain text"},
                    {"type": "reasoning_text", "text": "missing discriminator"}
                ],
                "encrypted_content": "encrypted-reasoning",
                "future_field": {"keep": true}
            }])
        );
        assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn cumulative_turns_keep_a_stable_serialized_prefix() {
        let sanitize_input = |input: Value| {
            sanitized(
                json!({"input": input, "include": ["reasoning.encrypted_content"]}),
                "grok-build",
                Some("session-123"),
            )["input"]
                .as_array()
                .cloned()
                .expect("input array")
        };
        let first = json!([{"role": "user", "content": "turn one"}]);
        let mut second = first.as_array().cloned().expect("array");
        second.extend([
            json!({"type": "reasoning", "id": "r1", "encrypted_content": "e1", "status": "completed"}),
            json!({"role": "assistant", "content": "answer one"}),
            json!({"role": "user", "content": "turn two"}),
        ]);
        let first = sanitize_input(first);
        let second = sanitize_input(Value::Array(second));
        assert_eq!(
            serde_json::to_string(&second[..first.len()]).expect("json"),
            serde_json::to_string(&first).expect("json")
        );
        assert!(second[1].get("status").is_none());
    }

    #[test]
    fn effort_is_kept_only_for_effort_capable_models() {
        let request = |model_id: &str, include: Value| {
            sanitized(
                json!({
                    "input": "plain prompt",
                    "include": include,
                    "reasoning": {"effort": "high", "summary": "auto", "future_option": "keep"},
                    "reasoningEffort": "high",
                    "prompt_cache_key": "existing-session"
                }),
                model_id,
                Some("new-session"),
            )
        };
        let build = request(
            "grok-build",
            json!([
                "message.output_text",
                "reasoning.encrypted_content",
                "reasoning.encrypted_content"
            ]),
        );
        assert_eq!(build["input"], "plain prompt");
        assert_eq!(
            build["reasoning"],
            json!({"summary": "auto", "future_option": "keep"})
        );
        assert!(build.get("reasoningEffort").is_none());
        assert_eq!(
            build["include"],
            json!(["message.output_text", "reasoning.encrypted_content"])
        );
        // An existing cache key is never replaced.
        assert_eq!(build["prompt_cache_key"], "existing-session");

        for capable in [
            "grok-4.3",
            "grok-4.5",
            "grok-4.6",
            "grok-4.7",
            "grok-4.7-build-fast",
            "grok-4.20-multi-agent-0309",
        ] {
            assert!(supports_reasoning_effort(&model(capable)), "{capable}");
            assert_eq!(
                request(capable, json!([]))["reasoning"]["effort"],
                "high",
                "{capable}"
            );
        }
        for incapable in [
            "grok-build",
            "grok-4.20-0309-reasoning",
            "grok-composer-2.5-fast",
            "grok-4.20-0309-non-reasoning",
        ] {
            assert!(!supports_reasoning_effort(&model(incapable)), "{incapable}");
        }

        // A non-reasoning model loses reasoning and its include entirely.
        let composer = request(
            "grok-composer-2.5-fast",
            json!(["message.output_text", "reasoning.encrypted_content"]),
        );
        assert!(composer.get("reasoning").is_none());
        assert_eq!(composer["include"], json!(["message.output_text"]));
    }

    #[test]
    fn empty_reasoning_is_dropped_and_active_reasoning_requests_encrypted_content() {
        let dropped = sanitized(
            json!({"input": "x", "reasoning": {"effort": "none"}, "include": ["message.output_text"]}),
            "grok-build",
            None,
        );
        assert!(dropped.get("reasoning").is_none());
        assert_eq!(dropped["include"], json!(["message.output_text"]));
        assert!(dropped.get("prompt_cache_key").is_none());

        let added = sanitized(
            json!({"input": "x", "reasoning": {"effort": "high", "summary": "detailed"}}),
            "grok-build",
            None,
        );
        assert_eq!(added["reasoning"], json!({"summary": "detailed"}));
        assert_eq!(added["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn images_are_normalized_and_tool_images_move_to_a_user_message() {
        let payload = sanitized(
            json!({"input": [
                {"role": "user", "content": [
                    {"type": "image", "data": "ZmFrZQ==", "mimeType": "image/png"},
                    {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png", "detail": "high"}},
                    {"type": "input_image", "image_url": "'data:image/png;base64,cXVvdGVk'"}
                ]},
                {"type": "function_call_output", "call_id": "call_1", "output": [
                    {"type": "input_text", "text": "tool text"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aW1n"}
                ]},
                {"type": "function_call_output", "call_id": "call_2", "output": [
                    "plain string output", {"type": "input_text", "text": "object output"}
                ]},
                {"type": "function_call_output", "call_id": "call_3", "output": [
                    {"type": "input_image", "image_url": "data:image/png;base64,YQ==", "detail": "low"},
                    {"type": "input_image", "image_url": "data:image/png;base64,Yg=="}
                ]}
            ]}),
            "grok-composer-2.5-fast",
            None,
        );
        assert_eq!(
            payload["input"],
            json!([
                {"role": "user", "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,ZmFrZQ==", "detail": "auto"},
                    {"type": "input_image", "image_url": "https://example.invalid/image.png", "detail": "high"},
                    {"type": "input_image", "image_url": "data:image/png;base64,cXVvdGVk", "detail": "auto"}
                ]},
                {"type": "function_call_output", "call_id": "call_1", "output": "tool text"},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "The previous tool result (call_1) included 1 image. Use the attached image as the visual output from that tool."},
                    {"type": "input_image", "image_url": "data:image/png;base64,aW1n", "detail": "auto"}
                ]},
                {"type": "function_call_output", "call_id": "call_2", "output": "plain string output\nobject output"},
                {"type": "function_call_output", "call_id": "call_3", "output": "(tool returned no text output)"},
                {"role": "user", "content": [
                    {"type": "input_text", "text": "The previous tool result (call_3) included 2 images. Use the attached images as the visual output from that tool."},
                    {"type": "input_image", "image_url": "data:image/png;base64,YQ==", "detail": "low"},
                    {"type": "input_image", "image_url": "data:image/png;base64,Yg==", "detail": "auto"}
                ]}
            ])
        );
    }

    #[test]
    fn thinking_off_sends_the_models_off_effort_like_pi() {
        let mut payload = json!({"input": "x"});
        prepare_payload(&mut payload, &model("grok-4.3"), llm::THINKING_OFF, "");
        assert_eq!(payload["reasoning"], json!({"effort": "none"}));
        assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));

        // grok-build takes no effort, so an off request leaves nothing.
        let mut payload = json!({"input": "x"});
        prepare_payload(&mut payload, &model("grok-build"), llm::THINKING_OFF, "");
        assert!(payload.get("reasoning").is_none());
        assert!(payload.get("include").is_none());

        // A non-reasoning model never gets a reasoning block.
        let mut payload = json!({"input": "x"});
        prepare_payload(
            &mut payload,
            &model("grok-composer-2.5-fast"),
            llm::THINKING_OFF,
            "",
        );
        assert!(payload.get("reasoning").is_none());
    }

    #[test]
    fn versions_and_base_urls_follow_upstream_rules() {
        for valid in ["1.0.46", "10.20.30", "1.2.3-beta.1", "1.2.3-rc_2"] {
            assert!(is_valid_version(valid), "{valid}");
        }
        for invalid in [
            "",
            "1.2",
            "1.2.3.4",
            "v1.2.3",
            "1.2.3-",
            "1.2.3-a b",
            "<html>",
        ] {
            assert!(!is_valid_version(invalid), "{invalid}");
        }
        assert_eq!(base_url(|_| None), DEFAULT_BASE_URL);
        let lookup = |values: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                values
                    .iter()
                    .find(|(key, _)| *key == name)
                    .map(|(_, value)| (*value).to_owned())
            }
        };
        assert_eq!(
            base_url(lookup(&[
                ("GROK_CLI_BASE_URL", "https://second.example/v1"),
                ("PI_GROK_CLI_BASE_URL", "https://first.example/v1//"),
            ])),
            "https://first.example/v1"
        );
        assert_eq!(
            base_url(lookup(&[(
                "GOSHCODER_GROK_CLI_BASE_URL",
                "https://alias.example/api/"
            )])),
            "https://alias.example/api"
        );
        let headers = version_headers("9.8.7");
        assert_eq!(headers[0].1, "grok-shell/9.8.7 (macos; aarch64)");
        assert_eq!(headers[1], ("x-grok-client-version", "9.8.7".to_owned()));
    }

    /// Serves `bodies` to successive GETs and counts them.
    fn version_server(bodies: Vec<(u16, &'static str)>) -> (String, Arc<Mutex<usize>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind version server");
        let url = format!(
            "http://{}/cli/stable",
            listener.local_addr().expect("address")
        );
        let served = Arc::new(Mutex::new(0));
        let counter = Arc::clone(&served);
        thread::spawn(move || {
            for (status, body) in bodies {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut buffer = [0_u8; 2048];
                let _ = stream.read(&mut buffer);
                *lock(&counter) += 1;
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (url, served)
    }

    #[test]
    fn the_stable_version_is_looked_up_once_and_refreshed_once_per_rejection() {
        let (url, served) = version_server(vec![(200, " 1.2.3\n"), (200, "1.2.4")]);
        assert_eq!(resolve_version(&url), "1.2.3");
        assert_eq!(resolve_version(&url), "1.2.3");
        assert_eq!(*lock(&served), 1, "cached for the process");
        assert_eq!(refresh_version(&url, "1.2.3"), "1.2.4");
        // A second rejection of the old value reuses the newer release.
        assert_eq!(refresh_version(&url, "1.2.3"), "1.2.4");
        assert_eq!(*lock(&served), 2);
    }

    #[test]
    fn an_unusable_version_answer_falls_back_to_the_pinned_release() {
        let (garbage, _) = version_server(vec![(200, "<html>nope</html>")]);
        assert_eq!(resolve_version(&garbage), FALLBACK_VERSION);
        let (missing, _) = version_server(vec![(404, "1.9.9")]);
        assert_eq!(resolve_version(&missing), FALLBACK_VERSION);
        let closed = TcpListener::bind("127.0.0.1:0").expect("bind");
        let unreachable = format!(
            "http://{}/cli/stable",
            closed.local_addr().expect("address")
        );
        drop(closed);
        assert_eq!(resolve_version(&unreachable), FALLBACK_VERSION);
    }

    /// A session log stand-in: entries recorded in order, the newest valid
    /// one wins.
    struct FakeStore {
        session: String,
        recording: bool,
        recorded: Mutex<Vec<Value>>,
    }

    impl ConvStore for FakeStore {
        fn session_id(&self) -> Option<String> {
            Some(self.session.clone())
        }

        fn generation(&self) -> Option<u64> {
            lock(&self.recorded)
                .iter()
                .rev()
                .find_map(stored_generation)
        }

        fn record(&self, generation: u64) -> Result<bool, String> {
            if !self.recording {
                return Ok(false);
            }
            lock(&self.recorded).push(json!({ "generation": generation }));
            Ok(true)
        }
    }

    #[test]
    fn conversation_ids_rotate_into_the_session_log_and_follow_the_open_session() {
        let store = Arc::new(FakeStore {
            session: "file-session".to_owned(),
            recording: true,
            recorded: Mutex::new(vec![json!({"generation": 3}), json!({"generation": "bad"})]),
        });
        let registration = register_conv_store("agent-session-a", store.clone());
        // The latest *valid* entry wins, and the id is the open session's.
        assert_eq!(
            conv_id("agent-session-a").as_deref(),
            Some("file-session:3")
        );
        assert_eq!(
            conv_command("agent-session-a", "rotate"),
            Ok("Grok CLI conversation ID rotated to file-session:4".to_owned())
        );
        assert_eq!(
            lock(&store.recorded).last(),
            Some(&json!({"generation": 4}))
        );
        assert_eq!(
            conv_command("agent-session-a", " STATUS "),
            Ok("Grok CLI conversation ID: file-session:4".to_owned())
        );
        assert_eq!(effective_session_id("agent-session-a"), "file-session");
        assert_eq!(
            conv_command("agent-session-a", "bogus"),
            Err("Usage: /grok-cli-conv [status|rotate]".to_owned())
        );
        drop(registration);
        // Once the session is gone, the request id stands alone again.
        assert_eq!(
            conv_id("agent-session-a").as_deref(),
            Some("agent-session-a")
        );
        assert_eq!(conv_id(""), None);
    }

    #[test]
    fn a_session_without_a_log_rotates_in_memory() {
        let store = Arc::new(FakeStore {
            session: "unsaved-session".to_owned(),
            recording: false,
            recorded: Mutex::new(Vec::new()),
        });
        let _registration = register_conv_store("agent-session-b", store.clone());
        assert_eq!(
            conv_id("agent-session-b").as_deref(),
            Some("unsaved-session")
        );
        assert_eq!(
            rotate_conv("agent-session-b").as_deref(),
            Ok("unsaved-session:1")
        );
        assert_eq!(
            rotate_conv("agent-session-b").as_deref(),
            Ok("unsaved-session:2")
        );
        assert!(lock(&store.recorded).is_empty());
        assert_eq!(
            conv_id("agent-session-b").as_deref(),
            Some("unsaved-session:2")
        );
    }

    #[test]
    fn stored_generations_must_be_positive_safe_integers() {
        assert_eq!(stored_generation(&json!({"generation": 2})), Some(2));
        for invalid in [
            json!({"generation": 0}),
            json!({"generation": -1}),
            json!({"generation": 1.5}),
            json!({"generation": 9_007_199_254_740_992_u64}),
            json!({"generation": "2"}),
            json!([2]),
        ] {
            assert_eq!(stored_generation(&invalid), None, "{invalid}");
        }
    }

    #[test]
    fn the_models_variable_filters_reorders_and_adds_generic_entries() {
        let models = vec![model("grok-4.3"), model("grok-build")];
        let filtered = filter_models(
            models.clone(),
            Some(" grok-build, brand-new ,grok-build,,"),
            "https://proxy.example/v1",
        );
        let ids = filtered
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, ["grok-build", "brand-new"]);
        let generic = &filtered[1];
        assert!(generic.reasoning);
        assert_eq!(generic.input, ["text"]);
        assert_eq!(generic.context_window, 1_000_000);
        assert_eq!(generic.max_tokens, 30_000);
        assert_eq!(generic.base_url, "https://proxy.example/v1");
        assert_eq!(generic.headers["x-grok-model-override"], "brand-new");
        assert_eq!(filter_models(models.clone(), Some(" , "), "x").len(), 2);
        assert_eq!(filter_models(models, None, "x").len(), 2);
    }
}
