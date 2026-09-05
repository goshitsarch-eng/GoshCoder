//! Google Application Default Credential helpers for Vertex AI.
//!
//! The Vertex REST API accepts a short-lived OAuth bearer token when no
//! express-mode API key is configured. This module supports the credential
//! sources available to the previous client: an explicit access token, a
//! service-account JSON key, and an authorized-user (gcloud) JSON file.

use std::{
    collections::BTreeMap,
    error::Error,
    fmt, fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock, mpsc},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use reqwest::blocking::Client;
use rsa::{
    RsaPrivateKey,
    pkcs1::DecodeRsaPrivateKey,
    pkcs1v15::SigningKey,
    pkcs8::DecodePrivateKey,
    rand_core::OsRng,
    sha2::Sha256,
    signature::{RandomizedSigner, SignatureEncoding},
};
use serde::Deserialize;
use serde_json::json;
use url::form_urlencoded;

const GOOGLE_CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
const GOOGLE_TOKEN_SKEW: Duration = Duration::from_secs(60);
const GOOGLE_TOKEN_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_CREDENTIAL_FILE_BYTES: usize = 2 * 1024 * 1024;
const MAX_TOKEN_RESPONSE_BYTES: usize = 64 * 1024;

static TOKEN_CACHE: OnceLock<Mutex<BTreeMap<PathBuf, AccessToken>>> = OnceLock::new();

#[derive(Clone, Debug)]
struct AccessToken {
    value: String,
    expires_at: Instant,
}

impl AccessToken {
    fn is_valid(&self) -> bool {
        !self.value.is_empty()
            && Instant::now()
                .checked_add(GOOGLE_TOKEN_SKEW)
                .is_some_and(|renew_at| renew_at < self.expires_at)
    }
}

#[derive(Debug)]
pub enum GoogleAuthError {
    Message(String),
    Cancelled,
    Io(std::io::Error),
    Http(reqwest::Error),
    Json(serde_json::Error),
}

impl fmt::Display for GoogleAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(message) => formatter.write_str(message),
            Self::Cancelled => formatter.write_str("Google token exchange aborted"),
            Self::Io(error) => write!(formatter, "Google credential I/O failed: {error}"),
            Self::Http(error) => write!(formatter, "Google token exchange failed: {error}"),
            Self::Json(error) => write!(formatter, "parsing Google credentials failed: {error}"),
        }
    }
}

impl Error for GoogleAuthError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Http(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Message(_) | Self::Cancelled => None,
        }
    }
}

impl From<std::io::Error> for GoogleAuthError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<reqwest::Error> for GoogleAuthError {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl From<serde_json::Error> for GoogleAuthError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

#[derive(Deserialize)]
struct CredentialFile {
    #[serde(rename = "type", default)]
    kind: String,
    #[serde(default)]
    client_email: String,
    #[serde(default)]
    private_key: String,
    #[serde(default)]
    private_key_id: String,
    #[serde(default)]
    token_uri: String,
    #[serde(default)]
    client_id: String,
    #[serde(default)]
    client_secret: String,
    #[serde(default)]
    refresh_token: String,
}

#[derive(Deserialize)]
struct TokenResponse {
    #[serde(default)]
    access_token: String,
    #[serde(default)]
    expires_in: i64,
    #[serde(default)]
    error: String,
    #[serde(default)]
    error_description: String,
}

/// Obtains a bearer token using explicit, application-default, or gcloud
/// credentials. Scoped environment values take precedence over process values.
pub fn resolve_access_token(
    environment: &BTreeMap<String, String>,
) -> Result<String, GoogleAuthError> {
    resolve_access_token_with_cancellation(environment, &crate::agent::CancellationToken::default())
}

/// Cancellation-aware form of [`resolve_access_token`].
///
/// `reqwest::blocking` cannot interrupt an in-flight socket read directly, so
/// a token exchange is isolated in a short-lived worker and this caller races
/// its result against the agent cancellation token. This keeps the agent turn
/// responsive while the bounded exchange worker winds down.
pub fn resolve_access_token_with_cancellation(
    environment: &BTreeMap<String, String>,
    cancellation: &crate::agent::CancellationToken,
) -> Result<String, GoogleAuthError> {
    ensure_not_cancelled(cancellation)?;
    if let Some(token) = provider_env_value(environment, "GOOGLE_OAUTH_ACCESS_TOKEN") {
        return Ok(token);
    }

    let path = provider_env_value(environment, "GOOGLE_APPLICATION_CREDENTIALS")
        .map(PathBuf::from)
        .or_else(well_known_credentials_path)
        .ok_or_else(|| {
            GoogleAuthError::Message(
                "no Google credentials found. Set GOOGLE_APPLICATION_CREDENTIALS to a service account or authorized user key file, or set GOOGLE_OAUTH_ACCESS_TOKEN".to_owned(),
            )
        })?;
    if let Some(token) = token_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&path)
        .filter(|token| token.is_valid())
        .map(|token| token.value.clone())
    {
        ensure_not_cancelled(cancellation)?;
        return Ok(token);
    }

    let credentials = load_credential_file(&path)?;
    ensure_not_cancelled(cancellation)?;
    let token = exchange_credentials(&credentials, cancellation)?;
    ensure_not_cancelled(cancellation)?;
    let value = token.value.clone();
    token_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(path, token);
    Ok(value)
}

fn ensure_not_cancelled(
    cancellation: &crate::agent::CancellationToken,
) -> Result<(), GoogleAuthError> {
    if cancellation.is_cancelled() {
        Err(GoogleAuthError::Cancelled)
    } else {
        Ok(())
    }
}

fn provider_env_value(environment: &BTreeMap<String, String>, name: &str) -> Option<String> {
    environment
        .get(name)
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| std::env::var(name).ok().filter(|value| !value.is_empty()))
}

fn well_known_credentials_path() -> Option<PathBuf> {
    let directory = std::env::var_os("APPDATA").map(PathBuf::from).or_else(|| {
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|home| PathBuf::from(home).join(".config"))
    })?;
    let path = directory
        .join("gcloud")
        .join("application_default_credentials.json");
    path.is_file().then_some(path)
}

fn token_cache() -> &'static Mutex<BTreeMap<PathBuf, AccessToken>> {
    TOKEN_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn load_credential_file(path: &Path) -> Result<CredentialFile, GoogleAuthError> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(GoogleAuthError::Message(format!(
            "Google credentials {} are not a file",
            path.display()
        )));
    }
    if metadata.len() > MAX_CREDENTIAL_FILE_BYTES as u64 {
        return Err(GoogleAuthError::Message(format!(
            "Google credentials {} exceed 2 MiB",
            path.display()
        )));
    }
    let content = fs::read(path)?;
    if content.len() > MAX_CREDENTIAL_FILE_BYTES {
        return Err(GoogleAuthError::Message(format!(
            "Google credentials {} exceed 2 MiB",
            path.display()
        )));
    }
    Ok(serde_json::from_slice(&content)?)
}

fn exchange_credentials(
    credentials: &CredentialFile,
    cancellation: &crate::agent::CancellationToken,
) -> Result<AccessToken, GoogleAuthError> {
    ensure_not_cancelled(cancellation)?;
    match credentials.kind.as_str() {
        "service_account" => exchange_service_account(credentials, cancellation),
        "authorized_user" => exchange_refresh_token(credentials, cancellation),
        _ if !credentials.private_key.is_empty() && !credentials.client_email.is_empty() => {
            exchange_service_account(credentials, cancellation)
        }
        _ if !credentials.refresh_token.is_empty() => {
            exchange_refresh_token(credentials, cancellation)
        }
        _ => Err(GoogleAuthError::Message(format!(
            "unsupported Google credential type {:?}; set GOOGLE_OAUTH_ACCESS_TOKEN instead",
            credentials.kind
        ))),
    }
}

fn exchange_service_account(
    credentials: &CredentialFile,
    cancellation: &crate::agent::CancellationToken,
) -> Result<AccessToken, GoogleAuthError> {
    ensure_not_cancelled(cancellation)?;
    let key = parse_rsa_private_key(&credentials.private_key)?;
    let token_uri = token_uri(credentials);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut header = serde_json::Map::from_iter([
        ("alg".to_owned(), json!("RS256")),
        ("typ".to_owned(), json!("JWT")),
    ]);
    if !credentials.private_key_id.is_empty() {
        header.insert("kid".to_owned(), json!(credentials.private_key_id));
    }
    let claims = json!({
        "iss": credentials.client_email,
        "scope": GOOGLE_CLOUD_PLATFORM_SCOPE,
        "aud": token_uri,
        "iat": now,
        "exp": now.saturating_add(3_600),
    });
    let assertion = sign_jwt(&serde_json::Value::Object(header), &claims, &key)?;
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer")
        .append_pair("assertion", &assertion)
        .finish();
    post_token_request(&token_uri, body, cancellation)
}

fn exchange_refresh_token(
    credentials: &CredentialFile,
    cancellation: &crate::agent::CancellationToken,
) -> Result<AccessToken, GoogleAuthError> {
    ensure_not_cancelled(cancellation)?;
    if credentials.client_id.is_empty()
        || credentials.client_secret.is_empty()
        || credentials.refresh_token.is_empty()
    {
        return Err(GoogleAuthError::Message(
            "incomplete authorized_user credentials: client_id, client_secret and refresh_token are required"
                .to_owned(),
        ));
    }
    let body = form_urlencoded::Serializer::new(String::new())
        .append_pair("grant_type", "refresh_token")
        .append_pair("client_id", &credentials.client_id)
        .append_pair("client_secret", &credentials.client_secret)
        .append_pair("refresh_token", &credentials.refresh_token)
        .finish();
    post_token_request(&token_uri(credentials), body, cancellation)
}

fn token_uri(credentials: &CredentialFile) -> String {
    if credentials.token_uri.is_empty() {
        GOOGLE_TOKEN_ENDPOINT.to_owned()
    } else {
        credentials.token_uri.clone()
    }
}

fn post_token_request(
    token_uri: &str,
    body: String,
    cancellation: &crate::agent::CancellationToken,
) -> Result<AccessToken, GoogleAuthError> {
    ensure_not_cancelled(cancellation)?;
    let token_uri = token_uri.to_owned();
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let _ = sender.send(post_token_request_blocking(&token_uri, body));
    });
    loop {
        ensure_not_cancelled(cancellation)?;
        match receiver.recv_timeout(Duration::from_millis(25)) {
            Ok(result) => {
                ensure_not_cancelled(cancellation)?;
                return result;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(GoogleAuthError::Message(
                    "Google token exchange worker stopped unexpectedly".to_owned(),
                ));
            }
        }
    }
}

fn post_token_request_blocking(
    token_uri: &str,
    body: String,
) -> Result<AccessToken, GoogleAuthError> {
    let client = Client::builder()
        .timeout(GOOGLE_TOKEN_REQUEST_TIMEOUT)
        .build()?;
    let mut response = client
        .post(token_uri)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body)
        .send()?;
    let status = response.status();
    let mut body = Vec::new();
    response
        .by_ref()
        .take((MAX_TOKEN_RESPONSE_BYTES + 1) as u64)
        .read_to_end(&mut body)?;
    if body.len() > MAX_TOKEN_RESPONSE_BYTES {
        return Err(GoogleAuthError::Message(
            "Google token response exceeds 64 KiB".to_owned(),
        ));
    }
    let payload = serde_json::from_slice::<TokenResponse>(&body)?;
    if !status.is_success() || payload.access_token.is_empty() {
        let detail = if !payload.error_description.is_empty() {
            payload.error_description
        } else if !payload.error.is_empty() {
            payload.error
        } else {
            format!("status {}", status.as_u16())
        };
        return Err(GoogleAuthError::Message(format!(
            "Google token exchange failed: {detail}"
        )));
    }
    let expires_in = u64::try_from(payload.expires_in)
        .ok()
        .filter(|expires_in| *expires_in != 0)
        .unwrap_or(3_600);
    Ok(AccessToken {
        value: payload.access_token,
        expires_at: Instant::now()
            .checked_add(Duration::from_secs(expires_in))
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3_600)),
    })
}

fn parse_rsa_private_key(value: &str) -> Result<RsaPrivateKey, GoogleAuthError> {
    RsaPrivateKey::from_pkcs1_pem(value)
        .or_else(|_| RsaPrivateKey::from_pkcs8_pem(value))
        .map_err(|error| {
            GoogleAuthError::Message(format!(
                "service account private_key is not a valid RSA PEM key: {error}"
            ))
        })
}

fn sign_jwt(
    header: &serde_json::Value,
    claims: &serde_json::Value,
    key: &RsaPrivateKey,
) -> Result<String, GoogleAuthError> {
    let encode = |value: &serde_json::Value| {
        serde_json::to_vec(value)
            .map(|encoded| URL_SAFE_NO_PAD.encode(encoded))
            .map_err(GoogleAuthError::Json)
    };
    let header = encode(header)?;
    let claims = encode(claims)?;
    let signing_input = format!("{header}.{claims}");
    let signer = SigningKey::<Sha256>::new(key.clone());
    let signature = signer.sign_with_rng(&mut OsRng, signing_input.as_bytes());
    Ok(format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    ))
}

#[cfg(test)]
pub(crate) fn clear_token_cache() {
    token_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        sync::mpsc,
        thread,
    };

    use super::*;

    #[test]
    fn explicit_access_token_precedes_file_credentials() {
        clear_token_cache();
        let token = resolve_access_token(&BTreeMap::from([(
            "GOOGLE_OAUTH_ACCESS_TOKEN".to_owned(),
            "ya29.explicit".to_owned(),
        )]))
        .expect("explicit token");
        assert_eq!(token, "ya29.explicit");
    }

    #[test]
    fn incomplete_authorized_user_credentials_are_rejected() {
        let error = exchange_credentials(
            &CredentialFile {
                kind: "authorized_user".to_owned(),
                client_id: "client".to_owned(),
                client_secret: String::new(),
                refresh_token: "refresh".to_owned(),
                private_key: String::new(),
                client_email: String::new(),
                private_key_id: String::new(),
                token_uri: String::new(),
            },
            &crate::agent::CancellationToken::default(),
        )
        .expect_err("incomplete authorized-user credentials");
        assert!(
            error
                .to_string()
                .contains("incomplete authorized_user credentials")
        );
    }

    #[test]
    fn service_account_jwt_has_the_expected_compact_shape() {
        let key = RsaPrivateKey::new(&mut OsRng, 2_048).expect("generate test RSA key");
        let token = sign_jwt(
            &json!({"alg": "RS256", "typ": "JWT"}),
            &json!({"iss": "service@example.test"}),
            &key,
        )
        .expect("sign JWT");
        let parts = token.split('.').collect::<Vec<_>>();
        assert_eq!(parts.len(), 3);
        let claims = URL_SAFE_NO_PAD.decode(parts[1]).expect("decode JWT claims");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&claims).expect("decode JWT claims JSON")["iss"],
            "service@example.test"
        );
    }

    #[test]
    fn authorized_user_credentials_exchange_and_cache_a_token() {
        clear_token_cache();
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind token server");
        let address = listener.local_addr().expect("token server address");
        let (sender, receiver) = mpsc::channel();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept token request");
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4_096];
            let mut body_start = None;
            let mut content_length = 0_usize;
            loop {
                let read = stream.read(&mut buffer).expect("read token request");
                assert_ne!(read, 0, "token request closed prematurely");
                request.extend_from_slice(&buffer[..read]);
                if body_start.is_none()
                    && let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n")
                {
                    body_start = Some(index + 4);
                    let headers = String::from_utf8_lossy(&request[..index + 4]);
                    content_length = headers
                        .lines()
                        .find_map(|line| {
                            line.split_once(':').and_then(|(name, value)| {
                                name.eq_ignore_ascii_case("content-length")
                                    .then(|| value.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                        })
                        .expect("token request content length");
                }
                if body_start.is_some_and(|start| request.len() >= start + content_length) {
                    break;
                }
            }
            sender
                .send(String::from_utf8(request).expect("UTF-8 token request"))
                .expect("send captured token request");
            let body = br#"{"access_token":"ya29.refreshed","expires_in":3600}"#;
            stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .as_bytes(),
                )
                .expect("write token response");
            stream.write_all(body).expect("write token response body");
        });

        let path = std::env::temp_dir().join(format!(
            "goshcoder-google-auth-{}-{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("current time")
                .as_nanos()
        ));
        fs::write(
            &path,
            json!({
                "type": "authorized_user",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token",
                "token_uri": format!("http://{address}"),
            })
            .to_string(),
        )
        .expect("write authorized-user credential fixture");
        let environment = BTreeMap::from([(
            "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
            path.display().to_string(),
        )]);
        let first = resolve_access_token(&environment).expect("exchange refresh token");
        let request = receiver.recv().expect("captured token request");
        server.join().expect("token server joins");
        let second = resolve_access_token(&environment).expect("cached refresh token");
        let _ = fs::remove_file(&path);
        clear_token_cache();

        assert_eq!(first, "ya29.refreshed");
        assert_eq!(second, "ya29.refreshed");
        assert!(request.contains("grant_type=refresh_token"));
        assert!(request.contains("client_id=client-id"));
        assert!(request.contains("client_secret=client-secret"));
        assert!(request.contains("refresh_token=refresh-token"));
    }
}
