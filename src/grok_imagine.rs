//! Grok Imagine: image generation and editing with a Grok CLI login.
//!
//! Native adaptation of pi-grok-cli v0.9.3's Imagine feature (J Liew /
//! kenryu42, MIT), written against its `src/imagine/aspect.ts`,
//! `generate.ts`, `imageUrl.ts`, `parseArgs.ts`, `register.ts`, `save.ts`,
//! `tool.ts` and `workflow.ts`, and the imagine half of `src/config.ts`.
//!
//! Deviations, each for a reason:
//! - No PNG preview is converted or drawn: the interface shows the saved
//!   path as text, which is what upstream falls back to without a capable
//!   terminal.
//! - The `image_gen` tool reads its source image through the workspace
//!   confinement every GoshCoder file tool uses; upstream resolves any path
//!   against the working directory. `/grok-cli-imagine --image` is typed by
//!   the user and keeps upstream's behaviour.
//! - Upstream's migration of its earlier config files is not ported; there
//!   is nothing in GoshCoder to migrate from.

use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, mpsc},
    thread,
    time::{Duration, Instant},
};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};

use crate::{agent, catalog::Catalog, grok_cli, session::SessionCustomRecorder, tools};

pub const TOOL_NAME: &str = "image_gen";
/// Session custom entry recording a `/grok-cli-imagine` result, as upstream
/// appends for its entry renderer.
pub const ENTRY_TYPE: &str = "grok-cli-imagine";
pub const AUTH_ERROR: &str =
    "Imagine requires Grok CLI authentication. Run /login grok-cli or set GROK_CLI_OAUTH_TOKEN.";
pub const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
pub const BASE_URL_ENV: &str = "PI_GROK_CLI_IMAGINE_BASE_URL";
pub const MODEL_ENV: &str = "PI_GROK_CLI_IMAGINE_MODEL";
pub const DEFAULT_MODEL: &str = "grok-imagine-image-quality";
pub const ASPECT_RATIOS: [&str; 14] = [
    "auto", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "2:1", "1:2", "19.5:9", "9:19.5",
    "20:9", "9:20",
];
const CONFIG_VERSION: u64 = 3;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ATTEMPTS: u32 = 3;
const RETRY_BASE_DELAY: Duration = Duration::from_millis(500);
const MAX_SOURCE_BYTES: u64 = 400 * 1024;
const MAX_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(25);

// ---------------------------------------------------------------------------
// Configuration (upstream src/config.ts)

pub fn config_path(agent_dir: &Path) -> PathBuf {
    grok_cli::state_dir(agent_dir).join("config.json")
}

/// Whether `image_gen` should be offered, plus anything wrong with the file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedConfig {
    pub enabled: bool,
    pub warning: Option<String>,
}

/// Reads `{"version":3,"imagine":{"enabled":bool}}`; versions 1 and 2 are
/// read the same way. Anything unusable means the default, enabled, with a
/// warning naming the problem.
pub fn load_config(path: &Path) -> LoadedConfig {
    let enabled = |warning: Option<String>| LoadedConfig {
        enabled: true,
        warning,
    };
    let raw = match fs::read(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return enabled(None),
        Err(error) => {
            return enabled(Some(format!(
                "Could not read {}: {error}. Using defaults.",
                path.display()
            )));
        }
    };
    let parsed = match serde_json::from_slice::<Value>(&raw) {
        Ok(Value::Object(parsed)) => parsed,
        Ok(_) => {
            return enabled(Some(format!(
                "Config {} must be a JSON object. Using defaults.",
                path.display()
            )));
        }
        Err(error) => {
            return enabled(Some(format!(
                "Could not read {}: {error}. Using defaults.",
                path.display()
            )));
        }
    };
    let version = parsed.get("version").and_then(Value::as_u64);
    if !matches!(version, Some(1..=CONFIG_VERSION)) {
        let shown = parsed
            .get("version")
            .map_or_else(|| "undefined".to_owned(), Value::to_string);
        return enabled(Some(format!(
            "Unsupported config version {shown} in {}. Using defaults.",
            path.display()
        )));
    }
    match parsed.get("imagine") {
        None => enabled(None),
        Some(Value::Object(imagine)) => match imagine.get("enabled") {
            Some(Value::Bool(value)) => LoadedConfig {
                enabled: *value,
                warning: None,
            },
            None => enabled(None),
            Some(_) => enabled(Some(format!(
                "Invalid {}: imagine.enabled must be true or false. Using enabled=true.",
                path.display()
            ))),
        },
        Some(_) => enabled(Some(format!(
            "Invalid {}: imagine must be a JSON object. Using defaults.",
            path.display()
        ))),
    }
}

pub fn save_config(path: &Path, enabled: bool) -> io::Result<()> {
    let mut contents = serde_json::to_vec_pretty(&json!({
        "version": CONFIG_VERSION,
        "imagine": { "enabled": enabled },
    }))
    .map_err(io::Error::other)?;
    contents.push(b'\n');
    crate::config::write_atomic(path, &contents, 0o600)
}

// ---------------------------------------------------------------------------
// Arguments (upstream src/imagine/aspect.ts and parseArgs.ts)

pub fn normalize_aspect_ratio(value: Option<&str>) -> Result<String, String> {
    let requested = value.unwrap_or("auto");
    let normalized = requested.trim();
    if ASPECT_RATIOS.contains(&normalized) {
        return Ok(normalized.to_owned());
    }
    Err(format!(
        "Unsupported aspect ratio \"{requested}\". Use one of: {}",
        ASPECT_RATIOS.join(", ")
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImagineArgs {
    pub prompt: String,
    pub aspect_ratio: String,
    pub out_path: Option<String>,
    pub image_path: Option<String>,
    pub resolution: String,
}

/// upstream's `/(?:[^\s"']+|"[^"]*"|'[^']*')+/g` tokenizer: a token joins
/// unquoted runs and complete quoted runs; whitespace and an unterminated
/// quote separate tokens. One pair of surrounding quotes is then stripped.
fn tokenize(args: &str) -> Vec<String> {
    let chars = args.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let mut token = String::new();
        while let Some(&character) = chars.get(index) {
            if character == '"' || character == '\'' {
                let Some(close) = chars[index + 1..]
                    .iter()
                    .position(|candidate| *candidate == character)
                else {
                    break;
                };
                token.extend(&chars[index..=index + 1 + close]);
                index += close + 2;
            } else if character.is_whitespace() {
                break;
            } else {
                token.push(character);
                index += 1;
            }
        }
        if token.is_empty() {
            index += 1;
            continue;
        }
        let quoted = token.chars().count() >= 2
            && ((token.starts_with('"') && token.ends_with('"'))
                || (token.starts_with('\'') && token.ends_with('\'')));
        tokens.push(if quoted {
            token[1..token.len() - 1].to_owned()
        } else {
            token
        });
    }
    tokens
}

/// `/grok-cli-imagine <prompt> [--image|--edit <path>] [--aspect <ratio>]
/// [--out|-o <path>]`.
pub fn parse_args(args: &str) -> Result<ImagineArgs, String> {
    let tokens = tokenize(args);
    let mut options = BTreeMap::new();
    let mut prompt = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        let token = &tokens[index];
        if !token.starts_with('-') {
            prompt.push(token.clone());
            index += 1;
            continue;
        }
        let option = match token.as_str() {
            "--aspect" | "--aspect-ratio" => "aspect",
            "--out" | "-o" => "out",
            "--resolution" => "resolution",
            "--image" | "--edit" => "image",
            _ => return Err(format!("Unknown option: {token}")),
        };
        let value = tokens
            .get(index + 1)
            .filter(|value| !value.is_empty() && !value.starts_with('-'))
            .ok_or_else(|| format!("{token} requires a value"))?;
        options.insert(option, value.clone());
        index += 2;
    }
    if prompt.is_empty() {
        return Err("Prompt is required".to_owned());
    }
    let resolution = options
        .get("resolution")
        .cloned()
        .unwrap_or_else(|| "1k".to_owned());
    if resolution != "1k" {
        return Err("Unsupported resolution. Only 1k is available.".to_owned());
    }
    Ok(ImagineArgs {
        prompt: prompt.join(" "),
        aspect_ratio: normalize_aspect_ratio(options.get("aspect").map(String::as_str))?,
        out_path: options.get("out").cloned(),
        image_path: options.get("image").cloned(),
        resolution,
    })
}

// ---------------------------------------------------------------------------
// Request (upstream src/imagine/generate.ts)

/// Where requests go, from the catalog's injected environment.
#[derive(Clone, Debug)]
pub struct Settings {
    pub base_url: String,
    pub model: String,
}

impl Settings {
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            base_url: lookup(BASE_URL_ENV)
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned())
                .trim_end_matches('/')
                .to_owned(),
            model: lookup(MODEL_ENV).unwrap_or_else(|| DEFAULT_MODEL.to_owned()),
        }
    }
}

fn retryable(status: u16) -> bool {
    matches!(status, 408 | 409 | 425 | 429) || status >= 500
}

fn error_detail(body: &str) -> String {
    match serde_json::from_str::<Value>(body) {
        Ok(json) => json
            .get("error")
            .and_then(|error| error.get("message"))
            .or_else(|| json.get("message"))
            .and_then(Value::as_str)
            .map_or_else(|| body.to_owned(), str::to_owned),
        Err(_) => body.to_owned(),
    }
}

fn http_error(status: u16, detail: &str) -> String {
    let suffix = if detail.is_empty() {
        String::new()
    } else {
        format!(": {}", detail.chars().take(500).collect::<String>())
    };
    match status {
        401 | 403 => format!(
            "Imagine rejected the API key (HTTP {status}). Re-run /login grok-cli or set GROK_CLI_OAUTH_TOKEN{suffix}"
        ),
        400 => format!("Imagine rejected the request (HTTP 400){suffix}"),
        429 => format!("Imagine rate limited the request (HTTP 429). Try again later{suffix}"),
        status if status >= 500 => {
            format!("Imagine service error (HTTP {status}) after automatic retries{suffix}")
        }
        status => format!("Imagine request failed (HTTP {status}){suffix}"),
    }
}

enum Attempt {
    Success(Vec<u8>),
    Status(u16, String),
    Network(String),
}

const CANCELLED: &str = "Imagine request was cancelled";

/// Sends on a helper thread so cancellation is noticed while the request is
/// in flight, the way the provider adapter waits for a response.
fn send_once(
    url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    cancellation: &agent::CancellationToken,
) -> Result<Attempt, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|error| format!("Imagine network request failed: {error}"))?;
    let mut request = client.post(url).body(body.to_vec());
    for (name, value) in headers {
        request = request.header(*name, value);
    }
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let outcome = request.send().map(|response| {
            let status = response.status().as_u16();
            let mut bytes = Vec::new();
            let read = response
                .take(MAX_RESPONSE_BYTES)
                .read_to_end(&mut bytes)
                .map(|_| ());
            (status, bytes, read)
        });
        let _ = sender.send(outcome);
    });
    loop {
        if cancellation.is_cancelled() {
            return Err(CANCELLED.to_owned());
        }
        match receiver.recv_timeout(POLL_INTERVAL) {
            Ok(Ok((status, bytes, read))) => {
                if let Err(error) = read {
                    return Ok(Attempt::Network(error.to_string()));
                }
                if (200..300).contains(&status) {
                    return Ok(Attempt::Success(bytes));
                }
                return Ok(Attempt::Status(
                    status,
                    error_detail(String::from_utf8_lossy(&bytes).trim()),
                ));
            }
            Ok(Err(error)) if error.is_timeout() => {
                return Ok(Attempt::Network("request timed out".to_owned()));
            }
            Ok(Err(error)) => return Ok(Attempt::Network(error.to_string())),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Ok(Attempt::Network("request worker exited".to_owned()));
            }
        }
    }
}

fn wait(delay: Duration, cancellation: &agent::CancellationToken) -> Result<(), String> {
    let deadline = Instant::now() + delay;
    while Instant::now() < deadline {
        if cancellation.is_cancelled() {
            return Err(CANCELLED.to_owned());
        }
        thread::sleep(POLL_INTERVAL.min(deadline.saturating_duration_since(Instant::now())));
    }
    Ok(())
}

/// One image request to `/images/generations`, or `/images/edits` when a
/// source image is given. Returns the base64 JPEG.
pub fn generate_image(
    settings: &Settings,
    token: &str,
    prompt: &str,
    aspect_ratio: &str,
    source_image: Option<&str>,
    cancellation: &agent::CancellationToken,
) -> Result<String, String> {
    let endpoint = if source_image.is_some() {
        "edits"
    } else {
        "generations"
    };
    let url = format!("{}/images/{endpoint}", settings.base_url);
    let mut body = json!({
        "model": settings.model,
        "prompt": prompt,
        "n": 1,
        "aspect_ratio": normalize_aspect_ratio(Some(aspect_ratio))?,
        "resolution": "1k",
        "response_format": "b64_json",
    });
    if let Some(url) = source_image {
        body["image"] = json!({ "url": url, "type": "image_url" });
    }
    let body = serde_json::to_vec(&body).map_err(|error| error.to_string())?;
    // Upstream sends its own package name and the pinned client release here,
    // not the looked-up one.
    let headers = [
        ("authorization", format!("Bearer {token}")),
        ("content-type", "application/json".to_owned()),
        ("accept", "application/json".to_owned()),
        ("user-agent", crate::oauth::OAUTH_USER_AGENT.to_owned()),
        (
            "x-grok-client-version",
            grok_cli::FALLBACK_VERSION.to_owned(),
        ),
    ];
    let mut attempt = 1;
    let response = loop {
        match send_once(&url, &headers, &body, cancellation)? {
            Attempt::Success(bytes) => break bytes,
            Attempt::Status(status, detail) => {
                if !retryable(status) || attempt == MAX_ATTEMPTS {
                    return Err(http_error(status, &detail));
                }
            }
            Attempt::Network(message) => {
                if attempt == MAX_ATTEMPTS {
                    return Err(if message.contains("timed out") {
                        "Imagine request timed out after 60s".to_owned()
                    } else {
                        format!("Imagine network request failed: {message}")
                    });
                }
            }
        }
        wait(RETRY_BASE_DELAY * 2_u32.pow(attempt - 1), cancellation)?;
        attempt += 1;
    };
    serde_json::from_slice::<Value>(&response)
        .ok()
        .and_then(|json| {
            json.get("data")?
                .get(0)?
                .get("b64_json")?
                .as_str()
                .filter(|b64| !b64.is_empty())
                .map(str::to_owned)
        })
        .ok_or_else(|| "Imagine returned a malformed response: missing image data".to_owned())
}

// ---------------------------------------------------------------------------
// Source images (upstream src/imagine/imageUrl.ts)

const SOURCE_TOO_LARGE: &str =
    "Source image exceeds the 400 KiB limit. Resize or compress it before editing.";

/// The media type the bytes declare, judged by magic number rather than
/// file name.
fn sniff_image(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[137, 80, 78, 71, 13, 10, 26, 10]) {
        Some("image/png")
    } else if bytes.starts_with(&[255, 216, 255]) {
        Some("image/jpeg")
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

fn be16(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_be_bytes([
        *bytes.get(at)?,
        *bytes.get(at + 1)?,
    ])))
}

fn le16(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from(u16::from_le_bytes([
        *bytes.get(at)?,
        *bytes.get(at + 1)?,
    ])))
}

fn le24(bytes: &[u8], at: usize) -> Option<u32> {
    Some(
        u32::from(*bytes.get(at)?)
            | u32::from(*bytes.get(at + 1)?) << 8
            | u32::from(*bytes.get(at + 2)?) << 16,
    )
}

/// Width and height read from the header, as pi-tui's
/// `getImageDimensions` does; `None` for a file that only looks like an
/// image.
fn image_dimensions(bytes: &[u8], mime: &str) -> Option<(u32, u32)> {
    match mime {
        "image/png" => {
            if bytes.get(12..16)? != b"IHDR" {
                return None;
            }
            let width = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
            let height = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
            Some((width, height))
        }
        "image/jpeg" => {
            let mut at = 2;
            while at + 4 <= bytes.len() {
                if bytes[at] != 0xFF {
                    return None;
                }
                let marker = bytes[at + 1];
                if marker == 0xFF {
                    at += 1;
                    continue;
                }
                if matches!(marker, 0x01 | 0xD0..=0xD9) {
                    at += 2;
                    continue;
                }
                let length = be16(bytes, at + 2)? as usize;
                let start_of_frame =
                    matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
                if start_of_frame {
                    return Some((be16(bytes, at + 7)?, be16(bytes, at + 5)?));
                }
                at += 2 + length;
            }
            None
        }
        "image/webp" => match bytes.get(12..16)? {
            b"VP8 " => Some((le16(bytes, 26)? & 0x3FFF, le16(bytes, 28)? & 0x3FFF)),
            b"VP8L" => {
                let bits = u32::from_le_bytes(bytes.get(21..25)?.try_into().ok()?);
                Some(((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
            }
            b"VP8X" => Some((le24(bytes, 24)? + 1, le24(bytes, 27)? + 1)),
            _ => None,
        },
        _ => None,
    }
}

/// Reads an edit source (at most 400 KiB) into a data URI.
pub fn image_file_to_data_uri(path: &Path) -> Result<String, String> {
    let metadata = fs::metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if metadata.len() > MAX_SOURCE_BYTES {
        return Err(SOURCE_TOO_LARGE.to_owned());
    }
    // At most one byte beyond the limit, even if the file grows after stat.
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|file| file.take(MAX_SOURCE_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_SOURCE_BYTES {
        return Err(SOURCE_TOO_LARGE.to_owned());
    }
    let mime = sniff_image(&bytes)
        .filter(|mime| {
            image_dimensions(&bytes, mime).is_some_and(|(width, height)| width >= 1 && height >= 1)
        })
        .ok_or_else(|| {
            format!(
                "Unsupported image file: {}. Use a PNG, JPEG, or WebP image.",
                path.display()
            )
        })?;
    Ok(format!("data:{mime};base64,{}", STANDARD.encode(&bytes)))
}

// ---------------------------------------------------------------------------
// Saving (upstream src/imagine/save.ts)

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SavedImage {
    pub absolute_path: PathBuf,
    pub relative_path: String,
    pub filename: String,
    pub used_fallback: bool,
}

/// Where a session's images go: `<session dir>/<session id>/images`.
pub fn session_image_dir(session_file: &Path, session_id: &str) -> Option<PathBuf> {
    Some(session_file.parent()?.join(session_id).join("images"))
}

pub fn fallback_image_dir() -> PathBuf {
    std::env::temp_dir()
        .join("goshcoder-grok-cli")
        .join("images")
}

/// Writes `N.jpg`, one past the highest number already there; a name taken
/// by a concurrent writer moves on to the next.
fn write_numbered_image(directory: &Path, bytes: &[u8]) -> io::Result<PathBuf> {
    fs::create_dir_all(directory)?;
    loop {
        let highest = fs::read_dir(directory)?
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
                let stem = name
                    .strip_suffix(".jpg")
                    .or_else(|| name.strip_suffix(".jpeg"))?;
                (!stem.is_empty() && stem.bytes().all(|byte| byte.is_ascii_digit()))
                    .then(|| stem.parse::<u64>().ok())
                    .flatten()
            })
            .max()
            .unwrap_or(0);
        let path = directory.join(format!("{}.jpg", highest + 1));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(bytes)?;
                return Ok(path);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}

/// Decodes and stores a generated image. `out_path`, already resolved by
/// the caller, is overwritten; otherwise the image is numbered in the
/// session's directory, or in `fallback_dir` without a session.
pub fn save_image(
    b64: &str,
    session_dir: Option<&Path>,
    out_path: Option<&Path>,
    fallback_dir: &Path,
) -> Result<SavedImage, String> {
    let bytes = STANDARD
        .decode(b64.trim())
        .map_err(|_| "Imagine did not return valid JPEG data".to_owned())?;
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return Err("Imagine did not return valid JPEG data".to_owned());
    }
    let used_fallback = out_path.is_none() && session_dir.is_none();
    let absolute_path = match out_path {
        Some(out_path) => {
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent).map_err(|error| error.to_string())?;
            }
            fs::write(out_path, &bytes).map_err(|error| error.to_string())?;
            out_path.to_path_buf()
        }
        None => write_numbered_image(session_dir.unwrap_or(fallback_dir), &bytes)
            .map_err(|error| error.to_string())?,
    };
    let filename = absolute_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let relative_path = match out_path {
        Some(out_path) => out_path.display().to_string(),
        None => format!("images/{filename}"),
    };
    Ok(SavedImage {
        absolute_path,
        relative_path,
        filename,
        used_fallback,
    })
}

// ---------------------------------------------------------------------------
// Workflow, command and tool (upstream workflow.ts, register.ts, tool.ts)

/// Everything a generation needs from the session that runs it.
#[derive(Clone)]
pub struct Context {
    pub catalog: Catalog,
    /// The session id requests carry, which picks the account whose token
    /// pays for the image, as upstream resolves the session's route.
    pub request_session: String,
    pub recorder: SessionCustomRecorder,
    pub cwd: PathBuf,
    /// Confines the tool's source images; `None` without workspace tools.
    pub workspace: Option<tools::Workspace>,
    pub fallback_dir: PathBuf,
}

impl Context {
    fn token(&self) -> Result<String, String> {
        crate::grok_accounts::Accounts::new(&self.catalog)
            .usage_route(&self.request_session)
            .ok()
            .map(|(_, token)| token)
            .filter(|token| !token.is_empty())
            .ok_or_else(|| AUTH_ERROR.to_owned())
    }

    fn settings(&self) -> Settings {
        Settings::from_lookup(|name| self.catalog.environment_value(name))
    }

    fn session_dir(&self) -> Option<PathBuf> {
        let handle = self.recorder.handle()?;
        session_image_dir(&handle.path, &handle.id)
    }

    fn resolve(&self, path: &str) -> PathBuf {
        let path = Path::new(path);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }
}

/// upstream `generateAndSaveImage`: token, optional edit source, request,
/// then the file.
pub fn generate_and_save(
    context: &Context,
    prompt: &str,
    aspect_ratio: &str,
    source: Option<&Path>,
    out_path: Option<&Path>,
    cancellation: &agent::CancellationToken,
) -> Result<SavedImage, String> {
    let token = context.token()?;
    let source = source.map(image_file_to_data_uri).transpose()?;
    let b64 = generate_image(
        &context.settings(),
        &token,
        prompt,
        aspect_ratio,
        source.as_deref(),
        cancellation,
    )?;
    let session_dir = context.session_dir();
    save_image(
        &b64,
        session_dir.as_deref(),
        out_path,
        &context.fallback_dir,
    )
}

/// `/grok-cli-imagine`: the lines to show, or the error.
pub fn run_command(context: &Context, args: &str) -> Result<Vec<String>, String> {
    let parsed = parse_args(args)?;
    let source = parsed
        .image_path
        .as_deref()
        .map(|path| context.resolve(path));
    let out_path = parsed.out_path.as_deref().map(|path| context.resolve(path));
    let saved = generate_and_save(
        context,
        &parsed.prompt,
        &parsed.aspect_ratio,
        source.as_deref(),
        out_path.as_deref(),
        &agent::CancellationToken::default(),
    )?;
    let mut lines = Vec::new();
    if context.recorder.recording() {
        let entry = json!({
            "path": saved.absolute_path.display().to_string(),
            "relativePath": saved.relative_path,
            "prompt": parsed.prompt,
        });
        if let Err(error) = context.recorder.record(ENTRY_TYPE, entry) {
            lines.push(format!(
                "Could not record the image in the session: {error}"
            ));
        }
    }
    if saved.used_fallback {
        lines.push("Session storage unavailable; saved image in temporary storage.".to_owned());
    }
    lines.push(format!(
        "Image saved to {} ({})",
        saved.relative_path,
        saved.absolute_path.display()
    ));
    Ok(lines)
}

const TOOL_DESCRIPTION: &str = "Generate or edit an image with Grok Imagine; returns the saved image's absolute path. Pass image to edit an existing local file. For a request for one image, call this tool exactly once. Call it multiple times only when the user explicitly requests multiple images. Do not re-read or re-display the image unless the user asks.\n\nGuidelines:\n- For a request for one image, call image_gen exactly once. Call it multiple times only when the user explicitly requests multiple images.\n- Do not repeat the saved path unless the user asks for it; the image_gen result already displays a copyable path.";

pub fn tool_parameters() -> Value {
    json!({
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Describe the image to generate, or the changes to apply to the source image."
            },
            "image": {
                "type": "string",
                "description": "Local PNG, JPEG, or WebP path to edit. Relative paths use the session working directory."
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Aspect ratio of the generated image. Defaults to 'auto'. Examples: 1:1, 16:9, 9:16, 3:2, 2:3.",
                "default": "auto"
            }
        },
        "required": ["prompt"]
    })
}

fn run_tool(
    context: &Context,
    parameters: &BTreeMap<String, Value>,
    cancellation: &agent::CancellationToken,
) -> Result<SavedImage, String> {
    let prompt = parameters
        .get("prompt")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    if prompt.is_empty() {
        return Err("Prompt is required".to_owned());
    }
    let image = match parameters.get("image") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let path = value.as_str().unwrap_or_default().trim();
            if path.is_empty() {
                return Err("Image path is required".to_owned());
            }
            Some(match context.workspace.as_ref() {
                Some(workspace) => workspace.resolve_existing(path)?,
                None => context.resolve(path),
            })
        }
    };
    let aspect_ratio =
        normalize_aspect_ratio(parameters.get("aspect_ratio").and_then(Value::as_str))?;
    generate_and_save(
        context,
        &prompt,
        &aspect_ratio,
        image.as_deref(),
        None,
        cancellation,
    )
}

/// The model-facing `image_gen` tool. Failures come back as a result the
/// model reads (`Image Gen error: …`), as upstream returns them.
pub fn tool(context: Arc<Context>) -> agent::Tool {
    agent::Tool::new(
        TOOL_NAME,
        "Image Gen",
        TOOL_DESCRIPTION,
        tool_parameters(),
        move |cancellation, _call_id, parameters, _update| {
            Ok(match run_tool(&context, &parameters, &cancellation) {
                Ok(saved) => agent::ToolResult {
                    content: vec![crate::llm::ContentBlock::text(
                        json!({
                            "path": saved.absolute_path.display().to_string(),
                            "filename": saved.filename,
                            "relative_path": saved.relative_path,
                            "message": "Image generated successfully. Do not repeat the saved path unless the user asks.",
                        })
                        .to_string(),
                    )],
                    details: Some(json!({
                        "path": saved.absolute_path.display().to_string(),
                        "relativePath": saved.relative_path,
                        "filename": saved.filename,
                    })),
                    ..agent::ToolResult::default()
                },
                Err(message) => agent::ToolResult {
                    content: vec![crate::llm::ContentBlock::text(format!(
                        "Image Gen error: {message}"
                    ))],
                    details: Some(json!({ "error": message })),
                    ..agent::ToolResult::default()
                },
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use std::{
        net::TcpListener,
        sync::mpsc::{Receiver, channel},
        thread::JoinHandle,
    };

    use super::*;
    use crate::session::{SessionOptions, SessionRuntime, SessionSelection};

    /// A JPEG the size of a pixel: SOI, a baseline frame header, EOI.
    const JPEG: &[u8] = &[
        0xFF, 0xD8, 0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x02, 0x00, 0x03, 0x03, 0x01, 0x11, 0x00,
        0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xFF, 0xD9,
    ];

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = vec![137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13];
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 2, 0, 0, 0]);
        bytes
    }

    fn temp_dir(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "goshcoder-imagine-{label}-{}-{}",
            std::process::id(),
            time::OffsetDateTime::now_utc().unix_timestamp_nanos()
        ));
        fs::create_dir_all(&directory).expect("temp dir");
        directory
    }

    struct Seen {
        target: String,
        headers: BTreeMap<String, String>,
        body: Value,
    }

    fn read_request(stream: &mut std::net::TcpStream) -> Seen {
        let mut raw = Vec::new();
        let mut buffer = [0_u8; 8192];
        let (head_end, length) = loop {
            let read = stream.read(&mut buffer).expect("read");
            assert_ne!(read, 0, "client hung up");
            raw.extend_from_slice(&buffer[..read]);
            if let Some(end) = raw.windows(4).position(|window| window == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&raw[..end]).to_ascii_lowercase();
                let length = head
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                break (end + 4, length);
            }
        };
        while raw.len() < head_end + length {
            let read = stream.read(&mut buffer).expect("read body");
            raw.extend_from_slice(&buffer[..read]);
        }
        let head = String::from_utf8_lossy(&raw[..head_end]).into_owned();
        let mut lines = head.lines();
        let target = lines
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap_or_default()
            .to_owned();
        let headers = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_owned()))
            .collect();
        Seen {
            target,
            headers,
            body: serde_json::from_slice(&raw[head_end..]).unwrap_or(Value::Null),
        }
    }

    fn server(responses: Vec<(u16, String)>) -> (String, Receiver<Seen>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let base = format!("http://{}/v1", listener.local_addr().expect("address"));
        let (sender, receiver) = channel();
        let handle = thread::spawn(move || {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().expect("accept");
                let _ = sender.send(read_request(&mut stream));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        (base, receiver, handle)
    }

    fn ok_body() -> String {
        json!({"data": [{"b64_json": STANDARD.encode(JPEG)}]}).to_string()
    }

    fn settings(base: &str) -> Settings {
        Settings::from_lookup(|name| (name == BASE_URL_ENV).then(|| format!("{base}/")))
    }

    #[test]
    fn arguments_follow_upstream_parsing() {
        assert_eq!(
            parse_args("--aspect 16:9 --out \"./my cat.jpg\" --resolution 1k a fluffy cat"),
            Ok(ImagineArgs {
                prompt: "a fluffy cat".to_owned(),
                aspect_ratio: "16:9".to_owned(),
                out_path: Some("./my cat.jpg".to_owned()),
                image_path: None,
                resolution: "1k".to_owned(),
            })
        );
        let aliases =
            parse_args("-o cat.jpg --aspect-ratio 1:1 --edit 'in put.png' cat").expect("aliases");
        assert_eq!(aliases.out_path.as_deref(), Some("cat.jpg"));
        assert_eq!(aliases.image_path.as_deref(), Some("in put.png"));
        assert_eq!(aliases.aspect_ratio, "1:1");
        assert_eq!(parse_args("cat").expect("defaults").aspect_ratio, "auto");
        // An unterminated quote separates rather than opens a run.
        assert_eq!(parse_args("say \"hi").expect("quote").prompt, "say hi");
        assert_eq!(parse_args(""), Err("Prompt is required".to_owned()));
        assert_eq!(
            parse_args("--wat cat"),
            Err("Unknown option: --wat".to_owned())
        );
        assert_eq!(
            parse_args("--out"),
            Err("--out requires a value".to_owned())
        );
        assert_eq!(
            parse_args("--resolution 2k cat"),
            Err("Unsupported resolution. Only 1k is available.".to_owned())
        );
        for ratio in ASPECT_RATIOS {
            assert_eq!(normalize_aspect_ratio(Some(ratio)).as_deref(), Ok(ratio));
        }
        assert!(
            normalize_aspect_ratio(Some("5:4"))
                .expect_err("unsupported")
                .starts_with("Unsupported aspect ratio \"5:4\". Use one of: auto, 1:1,")
        );
    }

    #[test]
    fn generation_requests_carry_upstreams_body_and_headers() {
        let (base, seen, handle) = server(vec![(200, ok_body()), (200, ok_body())]);
        let settings = settings(&base);
        let cancel = agent::CancellationToken::default();
        let b64 = generate_image(&settings, "secret", "a cat", "16:9", None, &cancel)
            .expect("generation");
        assert_eq!(STANDARD.decode(b64).expect("base64"), JPEG);
        generate_image(
            &settings,
            "secret",
            "Make it blue",
            "auto",
            Some("data:image/png;base64,c291cmNl"),
            &cancel,
        )
        .expect("edit");
        handle.join().expect("server");
        let requests = seen.try_iter().collect::<Vec<_>>();
        assert_eq!(requests[0].target, "/v1/images/generations");
        assert_eq!(requests[0].headers["authorization"], "Bearer secret");
        assert_eq!(requests[0].headers["content-type"], "application/json");
        assert_eq!(requests[0].headers["accept"], "application/json");
        assert_eq!(
            requests[0].headers["x-grok-client-version"],
            grok_cli::FALLBACK_VERSION
        );
        assert!(requests[0].headers["user-agent"].starts_with("goshcoder/"));
        assert_eq!(
            requests[0].body,
            json!({
                "model": "grok-imagine-image-quality",
                "prompt": "a cat",
                "n": 1,
                "aspect_ratio": "16:9",
                "resolution": "1k",
                "response_format": "b64_json"
            })
        );
        assert_eq!(requests[1].target, "/v1/images/edits");
        assert_eq!(
            requests[1].body["image"],
            json!({"url": "data:image/png;base64,c291cmNl", "type": "image_url"})
        );
        assert_eq!(
            Settings::from_lookup(|name| (name == MODEL_ENV).then(|| "custom-model".to_owned()))
                .model,
            "custom-model"
        );
    }

    #[test]
    fn retryable_failures_are_retried_three_times_and_others_are_not() {
        let (base, seen, handle) = server(vec![
            (503, "busy".to_owned()),
            (429, json!({"error": {"message": "slow down"}}).to_string()),
            (200, ok_body()),
        ]);
        let cancel = agent::CancellationToken::default();
        generate_image(&settings(&base), "t", "cat", "auto", None, &cancel).expect("third try");
        handle.join().expect("server");
        assert_eq!(seen.try_iter().count(), 3);

        let (base, seen, handle) = server(vec![
            (503, "busy".to_owned()),
            (503, "busy".to_owned()),
            (503, json!({"message": "still busy"}).to_string()),
        ]);
        let error = generate_image(&settings(&base), "t", "cat", "auto", None, &cancel)
            .expect_err("gives up");
        handle.join().expect("server");
        assert_eq!(seen.try_iter().count(), 3);
        assert_eq!(
            error,
            "Imagine service error (HTTP 503) after automatic retries: still busy"
        );

        // A rejected key is final on the first answer.
        let (base, seen, handle) = server(vec![(401, "denied".to_owned())]);
        let error = generate_image(&settings(&base), "t", "cat", "auto", None, &cancel)
            .expect_err("unauthorized");
        handle.join().expect("server");
        assert_eq!(seen.try_iter().count(), 1);
        assert_eq!(
            error,
            "Imagine rejected the API key (HTTP 401). Re-run /login grok-cli or set GROK_CLI_OAUTH_TOKEN: denied"
        );

        let (base, _seen, handle) = server(vec![(200, json!({"data": []}).to_string())]);
        let error = generate_image(&settings(&base), "t", "cat", "auto", None, &cancel)
            .expect_err("malformed");
        handle.join().expect("server");
        assert_eq!(
            error,
            "Imagine returned a malformed response: missing image data"
        );

        let cancelled = agent::CancellationToken::default();
        cancelled.cancel();
        let (base, _seen, _handle) = server(Vec::new());
        assert_eq!(
            generate_image(&settings(&base), "t", "cat", "auto", None, &cancelled),
            Err("Imagine request was cancelled".to_owned())
        );
    }

    #[test]
    fn edit_sources_are_typed_by_their_bytes_and_bounded() {
        let directory = temp_dir("sources");
        // The name says JPEG; the bytes say PNG, and the bytes win.
        let misnamed = directory.join("photo.jpg");
        fs::write(&misnamed, png(3, 2)).expect("write");
        assert!(
            image_file_to_data_uri(&misnamed)
                .expect("png")
                .starts_with("data:image/png;base64,iVBORw0KGgo")
        );
        let jpeg = directory.join("image.bin");
        fs::write(&jpeg, JPEG).expect("write");
        assert!(
            image_file_to_data_uri(&jpeg)
                .expect("jpeg")
                .starts_with("data:image/jpeg;base64,")
        );
        let mut webp = b"RIFF\0\0\0\0WEBPVP8X".to_vec();
        webp.extend_from_slice(&[0; 8]);
        webp.extend_from_slice(&[9, 0, 0, 4, 0, 0]);
        let webp_path = directory.join("image.webp");
        fs::write(&webp_path, &webp).expect("write");
        assert!(
            image_file_to_data_uri(&webp_path)
                .expect("webp")
                .starts_with("data:image/webp;base64,")
        );

        for (name, bytes) in [
            ("fake.png", b"not an image".to_vec()),
            ("empty.png", png(0, 5)),
        ] {
            let path = directory.join(name);
            fs::write(&path, bytes).expect("write");
            assert_eq!(
                image_file_to_data_uri(&path),
                Err(format!(
                    "Unsupported image file: {}. Use a PNG, JPEG, or WebP image.",
                    path.display()
                ))
            );
        }
        let large = directory.join("large.png");
        let mut bytes = png(1, 1);
        bytes.resize(400 * 1024 + 1, 0);
        fs::write(&large, &bytes).expect("write");
        assert_eq!(
            image_file_to_data_uri(&large),
            Err(SOURCE_TOO_LARGE.to_owned())
        );
        bytes.truncate(400 * 1024);
        fs::write(&large, &bytes).expect("write");
        assert!(
            image_file_to_data_uri(&large).is_ok(),
            "exactly 400 KiB is allowed"
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn images_are_numbered_checked_and_written_where_asked() {
        let directory = temp_dir("save");
        let images = directory.join("images");
        fs::create_dir_all(&images).expect("dir");
        for name in ["1.jpg", "7.JPEG", "x.jpg", "12.png"] {
            fs::write(images.join(name), b"").expect("seed");
        }
        let b64 = STANDARD.encode(JPEG);
        let saved =
            save_image(&b64, Some(&images), None, &directory.join("fallback")).expect("save");
        assert_eq!(saved.filename, "8.jpg");
        assert_eq!(saved.relative_path, "images/8.jpg");
        assert!(!saved.used_fallback);
        assert_eq!(fs::read(&saved.absolute_path).expect("read"), JPEG);

        let fallback = save_image(&b64, None, None, &directory.join("fallback")).expect("fallback");
        assert!(fallback.used_fallback);
        assert_eq!(
            fallback.absolute_path,
            directory.join("fallback").join("1.jpg")
        );

        let out = directory.join("nested").join("cat.jpg");
        let explicit = save_image(&b64, None, Some(&out), &directory).expect("out path");
        assert!(!explicit.used_fallback);
        assert_eq!(explicit.relative_path, out.display().to_string());
        assert_eq!(fs::read(&out).expect("read"), JPEG);

        assert_eq!(
            save_image(&STANDARD.encode(png(1, 1)), Some(&images), None, &directory),
            Err("Imagine did not return valid JPEG data".to_owned())
        );
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn the_switch_defaults_on_and_survives_a_bad_file_with_a_warning() {
        let directory = temp_dir("config");
        let path = config_path(&directory);
        assert_eq!(path, directory.join("grok-cli").join("config.json"));
        assert_eq!(
            load_config(&path),
            LoadedConfig {
                enabled: true,
                warning: None
            }
        );
        save_config(&path, false).expect("save");
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(&path).expect("read")).expect("json"),
            json!({"version": 3, "imagine": {"enabled": false}})
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(!load_config(&path).enabled);
        for (contents, warning) in [
            (r#"{"version":2,"imagine":{"enabled":false}}"#, None),
            (
                r#"{"version":4,"imagine":{"enabled":false}}"#,
                Some("Unsupported config version 4"),
            ),
            (
                r#"{"version":3,"imagine":{"enabled":"no"}}"#,
                Some("imagine.enabled must be true or false"),
            ),
            (
                r#"{"version":3,"imagine":[]}"#,
                Some("imagine must be a JSON object"),
            ),
            ("[]", Some("must be a JSON object")),
            ("{", Some("Could not read")),
        ] {
            fs::write(&path, contents).expect("write");
            let loaded = load_config(&path);
            match warning {
                None => assert_eq!(
                    loaded,
                    LoadedConfig {
                        enabled: false,
                        warning: None
                    }
                ),
                Some(warning) => {
                    assert!(loaded.enabled, "{contents}");
                    assert!(
                        loaded
                            .warning
                            .as_deref()
                            .is_some_and(|text| text.contains(warning)),
                        "{contents}: {loaded:?}"
                    );
                }
            }
        }
        let _ = fs::remove_dir_all(directory);
    }

    fn context(base: &str, recorder: SessionCustomRecorder, cwd: &Path, token: bool) -> Context {
        let mut environment = BTreeMap::from([(BASE_URL_ENV.to_owned(), base.to_owned())]);
        if token {
            environment.insert(grok_cli::TOKEN_ENV.to_owned(), "imagine-token".to_owned());
        }
        Context {
            catalog: Catalog::with_environment(
                None,
                Arc::new(move |name| environment.get(name).cloned()),
            )
            .expect("catalog"),
            request_session: String::new(),
            recorder,
            cwd: cwd.to_path_buf(),
            workspace: Some(tools::Workspace::new(cwd).expect("workspace")),
            fallback_dir: cwd.join("fallback-images"),
        }
    }

    #[test]
    fn the_command_saves_beside_the_session_and_records_the_image() {
        let root = temp_dir("command");
        let cwd = root.join("workspace");
        fs::create_dir_all(&cwd).expect("cwd");
        fs::write(cwd.join("source.png"), png(2, 2)).expect("source");
        let runtime = SessionRuntime::open(SessionOptions {
            cwd: cwd.clone(),
            sessions_dir: Some(root.join("sessions")),
            ..SessionOptions::default()
        })
        .expect("session");
        let handle = runtime.handle().expect("recording session");
        let (base, seen, server_handle) = server(vec![(200, ok_body())]);
        let context = context(&base, runtime.custom_recorder(), &cwd, true);
        let lines = run_command(&context, "a cat --edit source.png --aspect 1:1").expect("command");
        server_handle.join().expect("server");
        let expected = handle
            .path
            .parent()
            .expect("shard")
            .join(&handle.id)
            .join("images")
            .join("1.jpg");
        assert_eq!(
            lines,
            [format!(
                "Image saved to images/1.jpg ({})",
                expected.display()
            )]
        );
        assert_eq!(fs::read(&expected).expect("image"), JPEG);
        let request = seen.recv().expect("request");
        assert_eq!(request.headers["authorization"], "Bearer imagine-token");
        assert!(
            request.body["image"]["url"]
                .as_str()
                .is_some_and(|url| url.starts_with("data:image/png;base64,"))
        );
        assert_eq!(request.body["aspect_ratio"], "1:1");
        let recorded = runtime
            .custom_recorder()
            .latest_custom(ENTRY_TYPE, |data| Some(data.clone()))
            .expect("custom entry");
        assert_eq!(recorded["relativePath"], "images/1.jpg");
        assert_eq!(recorded["prompt"], "a cat");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn the_tool_reports_errors_to_the_model_and_keeps_to_the_workspace() {
        let root = temp_dir("tool");
        let cwd = root.join("workspace");
        fs::create_dir_all(&cwd).expect("cwd");
        fs::write(root.join("outside.png"), png(1, 1)).expect("outside");
        let runtime = SessionRuntime::open(SessionOptions {
            cwd: cwd.clone(),
            selection: SessionSelection::NoSession,
            ..SessionOptions::default()
        })
        .expect("session");
        let run = |context: Context, parameters: Value| {
            let tool = tool(Arc::new(context));
            let parameters = parameters
                .as_object()
                .expect("object")
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect();
            (tool.execute)(
                agent::CancellationToken::default(),
                "call-1".to_owned(),
                parameters,
                Arc::new(|_| {}),
            )
            .expect("tool result")
        };
        assert_eq!(
            tool(Arc::new(context(
                "http://127.0.0.1:1/v1",
                runtime.custom_recorder(),
                &cwd,
                false
            )))
            .name,
            TOOL_NAME
        );

        let missing = run(
            context(
                "http://127.0.0.1:1/v1",
                runtime.custom_recorder(),
                &cwd,
                false,
            ),
            json!({"prompt": "cat"}),
        );
        assert_eq!(
            missing.content[0].plain_text(),
            Some(format!("Image Gen error: {AUTH_ERROR}").as_str())
        );
        assert_eq!(missing.details, Some(json!({"error": AUTH_ERROR})));

        let escaped = run(
            context(
                "http://127.0.0.1:1/v1",
                runtime.custom_recorder(),
                &cwd,
                true,
            ),
            json!({"prompt": "cat", "image": "../outside.png"}),
        );
        let text = escaped.content[0].plain_text().expect("text");
        assert!(
            text.starts_with("Image Gen error: path ../outside.png is outside the workspace"),
            "{text}"
        );

        for (parameters, error) in [
            (json!({"prompt": "  "}), "Prompt is required"),
            (
                json!({"prompt": "cat", "image": " "}),
                "Image path is required",
            ),
        ] {
            let result = run(
                context(
                    "http://127.0.0.1:1/v1",
                    runtime.custom_recorder(),
                    &cwd,
                    true,
                ),
                parameters,
            );
            assert_eq!(result.details, Some(json!({"error": error})));
        }

        // Without a session file the image lands in the fallback directory.
        let (base, _seen, server_handle) = server(vec![(200, ok_body())]);
        let saved = run(
            context(&base, runtime.custom_recorder(), &cwd, true),
            json!({"prompt": "cat"}),
        );
        server_handle.join().expect("server");
        let details = saved.details.expect("details");
        assert_eq!(details["relativePath"], "images/1.jpg");
        assert_eq!(
            details["path"],
            cwd.join("fallback-images")
                .join("1.jpg")
                .display()
                .to_string()
        );
        let reply: Value =
            serde_json::from_str(saved.content[0].plain_text().expect("text")).expect("json");
        assert_eq!(reply["filename"], "1.jpg");
        assert_eq!(
            reply["message"],
            "Image generated successfully. Do not repeat the saved path unless the user asks."
        );
        let _ = fs::remove_dir_all(root);
    }
}
