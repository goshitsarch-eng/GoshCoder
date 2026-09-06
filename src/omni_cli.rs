//! Terminal command adapter for the OmniRoute gateway integration.

use std::{
    error::Error,
    io::{self, BufRead, IsTerminal, Read, Write},
    time::Duration,
};

use reqwest::{
    blocking::Client,
    header::{HeaderName, HeaderValue},
};

use crate::{
    catalog::{Catalog, Credential, CredentialStore},
    config, omniroute,
    provider_cli::read_secret,
};

const HEALTH_TIMEOUT: Duration = Duration::from_secs(omniroute::HEALTH_TIMEOUT_SECS);
const TEST_TIMEOUT: Duration = Duration::from_secs(omniroute::TEST_TIMEOUT_SECS);

/// Executes `goshcoder omni [status|setup|sync|models|test|dashboard|config|help]`.
pub fn command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let output = execute(arguments)?;
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}

/// Whether a subcommand talks to the user through the terminal and so must
/// not run behind an alternate screen.
pub fn needs_terminal(arguments: &[String]) -> bool {
    matches!(
        omniroute::CliCommand::parse(arguments),
        Ok(omniroute::CliCommand::Setup)
    )
}

/// Runs one OmniRoute subcommand and returns what it reports. Gateway text
/// travels through the result, so control characters are stripped before it
/// can reach a terminal.
pub fn execute(arguments: &[String]) -> Result<String, Box<dyn Error>> {
    let command = omniroute::CliCommand::parse(arguments)?;
    let timeout = match command {
        omniroute::CliCommand::Test(_) => TEST_TIMEOUT,
        _ => HEALTH_TIMEOUT,
    };
    let transport = ReqwestTransport::new(timeout)?;
    let output = match command {
        omniroute::CliCommand::Setup => setup(&transport)?,
        command => {
            let config_path = config::omni_route_path();
            let url_override = server_url_override();
            let (api_key, credential_source) = resolved_key()?;
            let context = omniroute::CommandContext {
                config_path: &config_path,
                url_override: url_override.as_deref(),
                api_key: &api_key,
                credential_source: &credential_source,
                transport: &transport,
                synced_at_millis: omniroute::unix_millis_now()?,
            };
            omniroute::execute_command(context, command.into())?.render()
        }
    };
    Ok(output
        .chars()
        .filter(|character| !character.is_control() || *character == '\n')
        .collect())
}

/// Probes a configured gateway for the session-start check. The error text
/// is what `/omni status` would show.
pub fn probe_health(config: &omniroute::Config, api_key: &str) -> Result<(), String> {
    let transport = ReqwestTransport::new(HEALTH_TIMEOUT).map_err(|error| error.to_string())?;
    omniroute::Client::new(config.clone(), api_key, &transport)
        .health()
        .map_err(|error| error.to_string())
}

/// `OMNIROUTE_URL`, when set to something other than whitespace.
fn server_url_override() -> Option<String> {
    std::env::var(omniroute::ENV_SERVER_URL)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn setup(transport: &ReqwestTransport) -> Result<String, Box<dyn Error>> {
    let stdin = io::stdin();
    if !stdin.is_terminal() {
        return Err(command_error(
            "OmniRoute setup requires an interactive terminal",
        ));
    }
    let current = omniroute::Config::load(config::omni_route_path())
        .map(|config| config.server_url)
        .unwrap_or_else(|_| omniroute::DEFAULT_SERVER_URL.to_owned());
    eprint!("OmniRoute URL [{current}]: ");
    io::stderr().flush()?;
    let mut url = String::new();
    stdin.lock().read_line(&mut url)?;
    let url = if url.trim().is_empty() {
        current
    } else {
        url.trim().to_owned()
    };
    let key = read_secret("OmniRoute API key (blank for local/public): ")?;
    let key = if key.trim().is_empty() {
        // Keep an already stored key when the prompt is skipped, as the
        // upstream setup does; the public placeholder is replaced too.
        stored_key().unwrap_or_default()
    } else {
        key
    };
    let result = omniroute::setup_command(
        config::omni_route_path(),
        omniroute::SetupRequest {
            server_url: url,
            api_key: key,
            allow_default: true,
        },
        transport,
    )?;
    let message = result.render();
    config::ensure_agent_dir()?;
    CredentialStore::default_file().put("omni", Credential::api_key(result.credential_to_store))?;
    Ok(message)
}

/// The key stored for `omni` in auth.json, unless it is the public
/// placeholder.
fn stored_key() -> Option<String> {
    let catalog = Catalog::with_default_credentials().ok()?;
    let credentials = catalog.credentials()?;
    let credential = credentials.read("omni").ok()??;
    let key = credential.key().trim().to_owned();
    (!key.is_empty() && key != omniroute::PUBLIC_API_KEY).then_some(key)
}

/// The key the catalog resolves for `omni` and where it came from. A
/// configured gateway without a key resolves to the public placeholder, so
/// an empty key means the gateway is unconfigured.
fn resolved_key() -> Result<(String, String), Box<dyn Error>> {
    let catalog = Catalog::with_default_credentials()?;
    Ok(catalog
        .resolve_auth("omni")?
        .and_then(|authentication| {
            authentication
                .api_key()
                .map(|key| (key.to_owned(), authentication.source().to_owned()))
        })
        .unwrap_or_default())
}

fn command_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

struct ReqwestTransport {
    client: Client,
}

impl ReqwestTransport {
    fn new(timeout: Duration) -> Result<Self, Box<dyn Error>> {
        // Requests carry the bearer key; a redirect must not replay it to
        // another host, so redirects are returned as ordinary statuses.
        Ok(Self {
            client: Client::builder()
                .timeout(timeout)
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
        })
    }
}

impl omniroute::HttpTransport for ReqwestTransport {
    fn execute(
        &self,
        request: omniroute::HttpRequest,
    ) -> std::result::Result<omniroute::HttpResponse, omniroute::HttpTransportError> {
        let method = match request.method {
            omniroute::HttpMethod::Get => reqwest::Method::GET,
            omniroute::HttpMethod::Post => reqwest::Method::POST,
        };
        let mut builder = self.client.request(method, &request.url);
        for (name, value) in &request.headers {
            let name = HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
                omniroute::HttpTransportError::new("request contains an invalid header name")
            })?;
            let value = HeaderValue::from_str(value).map_err(|_| {
                omniroute::HttpTransportError::new("request contains an invalid header value")
            })?;
            builder = builder.header(name, value);
        }
        if let Some(body) = request.body {
            builder = builder.body(body);
        }
        let response = builder.send().map_err(|error| {
            // The cause is what a user needs to tell a stopped gateway from
            // a slow one; the error's own text is dropped because reqwest
            // includes the request URL in it.
            omniroute::HttpTransportError::new(if error.is_timeout() {
                "request timed out"
            } else if error.is_connect() {
                "connection failed"
            } else {
                "request failed"
            })
        })?;
        let status = response.status().as_u16();
        let mut body = Vec::new();
        response
            .take((omniroute::MAX_RESPONSE_BYTES + 1) as u64)
            .read_to_end(&mut body)
            .map_err(|_| omniroute::HttpTransportError::new("read response failed"))?;
        Ok(omniroute::HttpResponse { status, body })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_keeps_the_response_bound_without_exposing_request_secrets() {
        let request = omniroute::HttpRequest {
            method: omniroute::HttpMethod::Get,
            url: "http://127.0.0.1:9/never-connect".to_owned(),
            headers: [("Authorization".to_owned(), "Bearer private".to_owned())]
                .into_iter()
                .collect(),
            body: None,
        };
        let transport = ReqwestTransport::new(Duration::from_millis(1)).expect("transport");
        let error = omniroute::HttpTransport::execute(&transport, request)
            .expect_err("connection should fail");
        assert!(
            matches!(
                error.message(),
                "request failed" | "connection failed" | "request timed out"
            ),
            "unexpected transport error: {}",
            error.message()
        );
        assert!(!error.message().contains("private"));
    }

    #[test]
    fn transport_returns_redirects_instead_of_following_them() {
        use std::{
            io::{BufRead, BufReader},
            net::TcpListener,
            thread,
        };

        let victim = TcpListener::bind("127.0.0.1:0").expect("bind victim listener");
        victim
            .set_nonblocking(true)
            .expect("nonblocking victim listener");
        let victim_url = format!(
            "http://{}/v1/models",
            victim.local_addr().expect("victim address")
        );
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind redirecting gateway");
        let address = listener.local_addr().expect("gateway address");
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept request");
            let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).expect("read request");
                if line == "\r\n" || line.is_empty() {
                    break;
                }
            }
            write!(
                stream,
                "HTTP/1.1 302 Found\r\nLocation: {victim_url}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .expect("write redirect");
            stream.flush().expect("flush redirect");
        });

        let transport = ReqwestTransport::new(Duration::from_secs(5)).expect("transport");
        let response = omniroute::HttpTransport::execute(
            &transport,
            omniroute::HttpRequest {
                method: omniroute::HttpMethod::Get,
                url: format!("http://{address}/v1/models"),
                headers: [("Authorization".to_owned(), "Bearer private".to_owned())]
                    .into_iter()
                    .collect(),
                body: None,
            },
        )
        .expect("the redirect itself is a complete response");
        assert_eq!(response.status, 302);
        worker.join().expect("gateway worker");
        assert!(
            matches!(victim.accept(), Err(error) if error.kind() == io::ErrorKind::WouldBlock),
            "the redirect target must never receive the bearer key"
        );
    }

    #[test]
    fn setup_outside_a_terminal_has_a_clear_remedy() {
        let error = command_error("OmniRoute setup requires an interactive terminal");
        assert!(error.to_string().contains("interactive terminal"));
    }
}
