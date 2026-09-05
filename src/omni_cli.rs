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

const HEALTH_TIMEOUT: Duration = Duration::from_secs(4);

/// Executes `goshcoder omni [status|sync|setup|dashboard]`.
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
    let transport = ReqwestTransport::new(HEALTH_TIMEOUT)?;
    let output = match command {
        omniroute::CliCommand::Status => {
            let key = resolved_key().unwrap_or_default();
            let report = omniroute::status_command(config::omni_route_path(), &key, &transport)?;
            if report.configured && key.is_empty() {
                return Err(command_error(
                    "OmniRoute credentials are missing; run `goshcoder omni setup`",
                ));
            }
            report.render()
        }
        omniroute::CliCommand::Dashboard => {
            omniroute::dashboard_command(config::omni_route_path())?
        }
        omniroute::CliCommand::Sync => {
            let key = resolved_key_required()?;
            let result = omniroute::sync_command_now(config::omni_route_path(), &key, &transport)?;
            result.render()
        }
        omniroute::CliCommand::Setup => setup(&transport)?,
    };
    Ok(output
        .chars()
        .filter(|character| !character.is_control() || *character == '\n')
        .collect())
}

fn setup(transport: &ReqwestTransport) -> Result<String, Box<dyn Error>> {
    let stdin = io::stdin();
    if !stdin.is_terminal() {
        return Err(command_error(
            "OmniRoute setup requires an interactive terminal",
        ));
    }
    eprint!("OmniRoute URL [{}]: ", omniroute::DEFAULT_SERVER_URL);
    io::stderr().flush()?;
    let mut url = String::new();
    stdin.lock().read_line(&mut url)?;
    let key = read_secret("OmniRoute API key (blank for local/public): ")?;
    let result = omniroute::setup_command(
        config::omni_route_path(),
        omniroute::SetupRequest {
            server_url: url.trim().to_owned(),
            api_key: key,
            allow_default: true,
        },
        transport,
    )?;
    let message = result.render();
    config::ensure_agent_dir()?;
    CredentialStore::default_file().put("omni", Credential::api_key(result.credential_to_store))?;
    Ok(message.to_owned())
}

fn resolved_key() -> Result<String, Box<dyn Error>> {
    let catalog = Catalog::with_default_credentials()?;
    Ok(catalog
        .resolve_auth("omni")?
        .and_then(|authentication| authentication.api_key().map(str::to_owned))
        .unwrap_or_default())
}

fn resolved_key_required() -> Result<String, Box<dyn Error>> {
    let key = resolved_key()?;
    if key.is_empty() {
        return Err(command_error(
            "OmniRoute credentials are missing; run `goshcoder omni setup`",
        ));
    }
    Ok(key)
}

fn command_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

struct ReqwestTransport {
    client: Client,
}

impl ReqwestTransport {
    fn new(timeout: Duration) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            client: Client::builder().timeout(timeout).build()?,
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
        let response = builder
            .send()
            .map_err(|_| omniroute::HttpTransportError::new("request failed"))?;
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
        assert_eq!(error.message(), "request failed");
    }

    #[test]
    fn missing_key_has_a_clear_setup_remedy() {
        let error = command_error("OmniRoute credentials are missing; run `goshcoder omni setup`");
        assert!(error.to_string().contains("goshcoder omni setup"));
    }
}
