//! Terminal command adapter for the OmniRoute gateway integration.

use std::{
    error::Error,
    io::{self, BufRead, IsTerminal, Read, Write},
    path::Path,
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
    let catalog = Catalog::with_default_credentials()?;
    let output = execute(
        arguments,
        &catalog,
        io::stdin().is_terminal() && io::stderr().is_terminal(),
    )?;
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}

/// Executes an OmniRoute command and returns terminal-ready output.
///
/// The interactive frontend calls this instead of writing directly to stdout,
/// so status, synchronization, and dashboard results remain visible in its
/// transcript. Setup intentionally requires a line-oriented terminal because
/// it prompts for a gateway URL and API key.
pub fn execute(
    arguments: &[String],
    catalog: &Catalog,
    interactive: bool,
) -> Result<String, Box<dyn Error>> {
    let transport = ReqwestTransport::new(HEALTH_TIMEOUT)?;
    execute_with_transport(arguments, catalog, interactive, &transport)
}

fn execute_with_transport<T: omniroute::HttpTransport + ?Sized>(
    arguments: &[String],
    catalog: &Catalog,
    interactive: bool,
    transport: &T,
) -> Result<String, Box<dyn Error>> {
    let command = omniroute::CliCommand::parse(arguments)?;
    let config_path = catalog.omniroute_config_path();
    match command {
        omniroute::CliCommand::Status => {
            let key = resolved_key(catalog).unwrap_or_default();
            let report = omniroute::status_command(&config_path, &key, transport)?;
            if report.configured && key.is_empty() {
                return Err(command_error(
                    "OmniRoute credentials are missing; run `goshcoder omni setup`",
                ));
            }
            Ok(report.render())
        }
        omniroute::CliCommand::Dashboard => omniroute::dashboard_command(&config_path)
            .map_err(|error| Box::new(error) as Box<dyn Error>),
        omniroute::CliCommand::Sync => {
            let key = resolved_key_required(catalog)?;
            let result = omniroute::sync_command_now(&config_path, &key, transport)?;
            Ok(result.render())
        }
        omniroute::CliCommand::Setup => {
            if !interactive {
                return Err(command_error(
                    "OmniRoute setup requires a line-oriented interactive terminal; run `goshcoder omni setup` outside fullscreen chat",
                ));
            }
            setup(transport, &config_path)
        }
    }
}

fn setup<T: omniroute::HttpTransport + ?Sized>(
    transport: &T,
    config_path: &Path,
) -> Result<String, Box<dyn Error>> {
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
        config_path,
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

fn resolved_key(catalog: &Catalog) -> Result<String, Box<dyn Error>> {
    Ok(catalog
        .resolve_auth("omni")?
        .and_then(|authentication| authentication.api_key().map(str::to_owned))
        .unwrap_or_default())
}

fn resolved_key_required(catalog: &Catalog) -> Result<String, Box<dyn Error>> {
    let key = resolved_key(catalog)?;
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
    use std::{
        fs,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct GatewayTransport;

    impl omniroute::HttpTransport for GatewayTransport {
        fn execute(
            &self,
            request: omniroute::HttpRequest,
        ) -> std::result::Result<omniroute::HttpResponse, omniroute::HttpTransportError> {
            assert_eq!(request.method, omniroute::HttpMethod::Get);
            assert!(request.url.ends_with("/v1/models"));
            Ok(omniroute::HttpResponse {
                status: 200,
                body: br#"{"data":[{"id":"gateway/chat","name":"Gateway Chat"}]}"#.to_vec(),
            })
        }
    }

    fn test_config_path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "goshcoder-omni-cli-{}-{}",
            std::process::id(),
            TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

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

    #[test]
    fn interactive_sync_uses_the_active_catalog_path_and_refreshes_models() {
        let path = test_config_path().join("omniroute.json");
        omniroute::Config::new("https://omni.example.test")
            .expect("valid config")
            .save(&path)
            .expect("save config");
        let catalog = Catalog::with_environment(
            None,
            Arc::new(|name: &str| (name == "OMNIROUTE_API_KEY").then(|| "gateway-key".to_owned())),
        )
        .expect("catalog")
        .with_omniroute_path(path.clone());

        let output =
            execute_with_transport(&["sync".to_owned()], &catalog, false, &GatewayTransport)
                .expect("sync command");

        assert!(output.contains("Synced 1 OmniRoute models"));
        let model = catalog
            .model(omniroute::OMNI_PROVIDER_ID, "gateway/chat")
            .expect("refreshed model");
        assert_eq!(model.base_url, "https://omni.example.test/v1");
        assert_eq!(model.api, omniroute::OPENAI_COMPLETIONS_API);
        assert_eq!(
            catalog
                .resolve_model("omni/gateway/chat")
                .expect("resolved refreshed model")
                .auth()
                .api_key(),
            Some("gateway-key")
        );

        let _ = fs::remove_dir_all(path.parent().expect("parent directory"));
    }

    #[test]
    fn fullscreen_setup_reports_a_safe_terminal_requirement() {
        let path = test_config_path().join("omniroute.json");
        let catalog = Catalog::with_environment(None, Arc::new(|_: &str| None))
            .expect("catalog")
            .with_omniroute_path(path);
        let error =
            execute_with_transport(&["setup".to_owned()], &catalog, false, &GatewayTransport)
                .expect_err("fullscreen setup must not consume terminal input");
        assert!(
            error
                .to_string()
                .contains("line-oriented interactive terminal")
        );
    }
}
