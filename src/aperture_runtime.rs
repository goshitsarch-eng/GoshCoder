//! Session-lifecycle integration for Aperture.
//!
//! The core Aperture module owns persistence and routing, while this adapter
//! performs non-blocking session-start refresh work: it refreshes cached
//! gateway models, reports recoverable failures to the active frontend, and
//! makes configured connector tools available to the live agent.

use std::{collections::BTreeSet, io, sync::Arc, thread};

use crate::{agent, aperture, aperture_mcp, catalog::Catalog, session::SessionNoticeSender};

/// Callback used to add dynamically discovered tools to a live session.
pub type ToolInstaller = Arc<dyn Fn(Vec<agent::Tool>) + Send + Sync + 'static>;

/// Owns cancellation for one session's background Aperture startup work.
///
/// No network activity is started unless `extensions/aperture.json` exists
/// and names a gateway. Dropping the prepared session cancels any startup
/// work that has not yet reached a blocking transport boundary.
pub struct ApertureRuntime {
    cancellation: agent::CancellationToken,
}

impl ApertureRuntime {
    /// Starts background Aperture synchronization and connector registration.
    ///
    /// Configuration read errors are shown once as session notices. A missing
    /// config intentionally remains silent, so ordinary sessions do not gain
    /// an Aperture dependency.
    pub fn start(
        catalog: Catalog,
        notices: SessionNoticeSender,
        tools_enabled: bool,
        install_tools: ToolInstaller,
    ) -> Option<Self> {
        let configuration_path = catalog.aperture_config_path();
        let cache_path = catalog.aperture_cache_path();
        let configuration = match aperture::load_config(&configuration_path) {
            Ok(configuration) => configuration,
            Err(aperture::ApertureError::Io { source, .. })
                if source.kind() == io::ErrorKind::NotFound =>
            {
                return None;
            }
            Err(error) => {
                notices.push("Aperture", error.to_string());
                return None;
            }
        };
        let resolved = configuration.resolve();
        if resolved.onboarding_enabled && !resolved.onboarding_done {
            notices.push(
                "Aperture",
                "extension installed. Run /aperture onboarding to configure.",
            );
        }
        let gateway = aperture::gateway_url(&resolved.base_url);
        if gateway.is_empty() {
            return None;
        }

        let cancellation = agent::CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let worker_notices = notices.clone();
        let worker = move || {
            session_start(
                catalog,
                configuration,
                cache_path,
                gateway,
                resolved,
                tools_enabled,
                install_tools,
                worker_notices,
                worker_cancellation,
            );
        };
        if thread::Builder::new()
            .name("goshcoder-aperture-start".to_owned())
            .spawn(worker)
            .is_err()
        {
            notices.push(
                "Aperture",
                "could not start background gateway synchronization",
            );
            return None;
        }
        Some(Self { cancellation })
    }
}

impl Drop for ApertureRuntime {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

/// Builds an installer for sessions that do not use Planner's phase-aware
/// tool rebuilding. Existing tools win on name collision, matching the
/// previous runtime's normal-tools-before-extras merge order.
pub fn agent_tool_installer(agent: agent::Agent) -> ToolInstaller {
    Arc::new(move |additions| {
        if additions.is_empty() {
            return;
        }
        let mut tools = agent.state().tools;
        append_unique_tools(&mut tools, additions);
        agent.set_tools(tools);
    })
}

#[allow(clippy::too_many_arguments)]
fn session_start(
    catalog: Catalog,
    configuration: aperture::Config,
    cache_path: std::path::PathBuf,
    gateway: String,
    resolved: aperture::Resolved,
    tools_enabled: bool,
    install_tools: ToolInstaller,
    notices: SessionNoticeSender,
    cancellation: agent::CancellationToken,
) {
    let local_models = catalog
        .providers()
        .into_iter()
        .flat_map(|provider| provider.models())
        .collect::<Vec<_>>();
    match aperture::sync(&configuration, &cache_path, &local_models) {
        Ok(result) => {
            if cancellation.is_cancelled() {
                return;
            }
            catalog.reload_aperture_state();
            for warning in result.warnings {
                notices.push("Aperture", warning);
            }
        }
        Err(error) => {
            if cancellation.is_cancelled() {
                return;
            }
            notices.push("Aperture", format!("model refresh failed: {error}"));
        }
    }

    if cancellation.is_cancelled() || !tools_enabled || !resolved.connectors_enabled {
        return;
    }

    let timeouts = aperture_mcp::McpTimeouts {
        initialization: aperture_mcp::MCP_INITIALIZATION_TIMEOUT,
        call: std::time::Duration::from_secs(15),
        initialized_notification: aperture_mcp::MCP_NOTIFICATION_TIMEOUT,
    };
    let session = match aperture_mcp::McpClient::new(&gateway)
        .map(|client| client.with_timeouts(timeouts))
        .and_then(|client| client.initialize_with(&cancellation))
    {
        Ok(session) => session,
        Err(aperture_mcp::McpError::Cancelled) => return,
        Err(error) => {
            notices.push(
                "Aperture",
                format!("[connectors] connector session failed: {error}"),
            );
            return;
        }
    };
    let tools = match session.list_tools_with(&cancellation) {
        Ok(tools) => tools,
        Err(aperture_mcp::McpError::Cancelled) => return,
        Err(error) => {
            notices.push(
                "Aperture",
                format!("[connectors] connector tools/list failed: {error}"),
            );
            return;
        }
    };
    if cancellation.is_cancelled() {
        return;
    }

    // Connector metadata only improves grouping. The MCP tool list remains
    // authoritative and useful even if this auxiliary endpoint is unavailable.
    let connectors = aperture::GatewayClient::new(&gateway)
        .and_then(|client| client.connectors())
        .unwrap_or_default();
    if cancellation.is_cancelled() {
        return;
    }
    let connector_tools =
        match aperture_mcp::build_connector_agent_tools(&resolved, &connectors, &tools, session) {
            Ok(connector_tools) => connector_tools,
            Err(error) => {
                notices.push(
                    "Aperture",
                    format!("[connectors] could not register connector tools: {error}"),
                );
                return;
            }
        };
    if !connector_tools.missing_pins.is_empty() {
        notices.push(
            "Aperture",
            format!(
                "[connectors] pinned tool(s) not found on gateway: {}",
                connector_tools.missing_pins.join(", ")
            ),
        );
    }
    if !connector_tools.tools.is_empty() && !cancellation.is_cancelled() {
        install_tools(connector_tools.tools);
    }
}

fn append_unique_tools(tools: &mut Vec<agent::Tool>, additions: Vec<agent::Tool>) {
    let mut names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();
    tools.extend(
        additions
            .into_iter()
            .filter(|tool| names.insert(tool.name.clone())),
    );
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{BufRead, BufReader, Read, Write},
        net::{TcpListener, TcpStream},
        path::PathBuf,
        process,
        sync::{
            atomic::{AtomicU64, Ordering},
            mpsc::{self, Receiver},
        },
        thread::{self, JoinHandle},
        time::{Duration, Instant},
    };

    use serde_json::{Value, json};

    use super::*;
    use crate::{
        catalog::Catalog,
        plannotator,
        runtime::{self, SessionConfig},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[derive(Debug)]
    struct CapturedRequest {
        method: String,
        target: String,
        body: Value,
    }

    fn temporary_directory() -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "goshcoder-aperture-runtime-{}-{sequence}",
            process::id()
        ));
        fs::create_dir_all(&directory).expect("create temporary directory");
        directory
    }

    fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        let mut reader = BufReader::new(stream.try_clone().expect("clone test stream"));
        let mut request_line = String::new();
        reader
            .read_line(&mut request_line)
            .expect("read request line");
        let mut parts = request_line.split_whitespace();
        let method = parts.next().expect("request method").to_owned();
        let target = parts.next().expect("request target").to_owned();
        let mut content_length = 0;
        loop {
            let mut header = String::new();
            reader.read_line(&mut header).expect("read request header");
            if header == "\r\n" || header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                content_length = value.trim().parse().expect("numeric content length");
            }
        }
        let mut body = vec![0; content_length];
        reader.read_exact(&mut body).expect("read request body");
        CapturedRequest {
            method,
            target,
            body: (!body.is_empty())
                .then(|| serde_json::from_slice(&body).expect("JSON request"))
                .unwrap_or(Value::Null),
        }
    }

    fn response(status: u16, body: &str, headers: &[(&str, &str)]) -> Vec<u8> {
        let reason = match status {
            200 => "OK",
            202 => "Accepted",
            _ => "Test Response",
        };
        let mut response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("\r\n");
        let mut response = response.into_bytes();
        response.extend_from_slice(body.as_bytes());
        response
    }

    fn connector_gateway() -> (String, Receiver<CapturedRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind gateway");
        let address = listener.local_addr().expect("gateway address");
        let (sender, receiver) = mpsc::channel();
        let worker = thread::spawn(move || {
            for _ in 0..6 {
                let (mut stream, _) = listener.accept().expect("accept gateway request");
                let request = read_request(&mut stream);
                let response = match (request.method.as_str(), request.target.as_str()) {
                    ("GET", "/api/providers") => response(200, r#"{"providers":[]}"#, &[]),
                    ("GET", "/v1/models") => response(200, r#"{"data":[]}"#, &[]),
                    ("GET", "/api/connectors") => response(
                        200,
                        r#"{"connectors":[{"id":"github","provider":"GitHub","status":"connected"}]}"#,
                        &[],
                    ),
                    ("POST", "/v1/mcp")
                        if request.body["method"] == Value::String("initialize".to_owned()) =>
                    {
                        response(
                            200,
                            &json!({
                                "jsonrpc": "2.0",
                                "id": 1,
                                "result": {"protocolVersion": aperture_mcp::MCP_PROTOCOL_VERSION}
                            })
                            .to_string(),
                            &[("Mcp-Session-Id", "session-runtime")],
                        )
                    }
                    ("POST", "/v1/mcp")
                        if request.body["method"]
                            == Value::String("notifications/initialized".to_owned()) =>
                    {
                        response(202, "", &[])
                    }
                    ("POST", "/v1/mcp")
                        if request.body["method"] == Value::String("tools/list".to_owned()) =>
                    {
                        response(
                            200,
                            &json!({
                                "jsonrpc": "2.0",
                                "id": 2,
                                "result": {
                                    "tools": [{
                                        "name": "github_search",
                                        "description": "Search GitHub",
                                        "inputSchema": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string"}
                                            }
                                        }
                                    }]
                                }
                            })
                            .to_string(),
                            &[],
                        )
                    }
                    _ => panic!("unexpected gateway request: {request:?}"),
                };
                sender.send(request).expect("capture request");
                stream.write_all(&response).expect("write gateway response");
                stream.flush().expect("flush gateway response");
            }
        });
        (format!("http://{address}"), receiver, worker)
    }

    #[test]
    fn session_start_registers_connectors_and_planner_keeps_them() {
        let directory = temporary_directory();
        let (gateway, requests, server) = connector_gateway();
        let configuration_path = directory.join("extensions").join("aperture.json");
        let cache_path = directory.join("extensions").join("aperture-cache.json");
        aperture::save_config(
            &configuration_path,
            &aperture::Config {
                base_url: gateway,
                onboarding_done: Some(true),
                dedicated: Some(aperture::DedicatedConfig {
                    enabled: Some(false),
                    providers: Some(Vec::new()),
                    cached_models: None,
                }),
                connectors: Some(aperture::ConnectorsConfig {
                    enabled: true,
                    pinned_tools: Some(vec![aperture::PinnedConnectorTool {
                        connector_id: "github".to_owned(),
                        tool_name: "github_search".to_owned(),
                    }]),
                    discovery_tools: Some(true),
                }),
                ..aperture::Config::default()
            },
        )
        .expect("save Aperture configuration");
        let catalog = Catalog::with_environment(
            None,
            Arc::new(|name| (name == "OPENAI_API_KEY").then(|| "test-key".to_owned())),
        )
        .expect("catalog")
        .with_aperture_paths(&configuration_path, &cache_path);
        let model_id = catalog
            .provider("openai")
            .and_then(|provider| provider.models().last().map(|model| model.id.clone()))
            .expect("OpenAI model");
        let mut prepared = runtime::prepare_session(
            &catalog,
            SessionConfig {
                model_ref: format!("openai/{model_id}"),
                workdir: directory.clone(),
                enable_tools: true,
                load_planner: true,
                no_session: true,
                ..SessionConfig::default()
            },
            None,
            Vec::new(),
        )
        .expect("prepare session");
        let agent = prepared.runtime.agent().clone();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline
            && !agent
                .state()
                .tools
                .iter()
                .any(|tool| tool.name == "github_search")
        {
            thread::sleep(Duration::from_millis(10));
        }
        for name in [
            "github_search",
            "aperture_connector_list",
            "aperture_connector_tool_search",
            "aperture_connector_tool_describe",
            "aperture_connector_tool_call",
        ] {
            assert!(
                agent.state().tools.iter().any(|tool| tool.name == name),
                "missing {name} after connector registration"
            );
        }

        assert_eq!(
            prepared.toggle_planner().expect("enter planning"),
            plannotator::Phase::Planning
        );
        assert!(
            agent
                .state()
                .tools
                .iter()
                .any(|tool| tool.name == "github_search"),
            "planner rebuild discarded dynamically registered connector"
        );

        let captured = (0..6)
            .map(|_| requests.recv().expect("captured gateway request"))
            .collect::<Vec<_>>();
        server.join().expect("gateway server");
        assert_eq!(
            captured
                .iter()
                .map(|request| (request.method.as_str(), request.target.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("GET", "/api/providers"),
                ("GET", "/v1/models"),
                ("POST", "/v1/mcp"),
                ("POST", "/v1/mcp"),
                ("POST", "/v1/mcp"),
                ("GET", "/api/connectors"),
            ]
        );
        assert_eq!(captured[2].body["method"], "initialize");
        assert_eq!(captured[4].body["method"], "tools/list");

        prepared.runtime.close().expect("close session");
        drop(prepared);
        fs::remove_dir_all(directory).expect("remove temporary directory");
    }
}
