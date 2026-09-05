//! Command-line adapters for the built-in model catalog and credential store.

use std::{
    error::Error,
    io::{self, BufRead, IsTerminal, Write},
    sync::Arc,
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};

use crate::{
    catalog::{Catalog, Credential, CredentialStore, Provider},
    config, oauth,
};

/// Executes `goshcoder providers`.
pub fn providers_command() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::with_default_credentials()?;
    for provider in catalog.providers() {
        let (status, detail) = match catalog.resolve_auth(&provider.id)? {
            Some(authentication) => {
                let ambient = if authentication.is_ambient() {
                    " (ambient)"
                } else {
                    ""
                };
                ("✓", format!("{}{}", authentication.source(), ambient))
            }
            None => ("-", provider_setup_hint(&provider)),
        };
        println!(
            "{status} {:<24} {:<22} {detail}",
            provider.id, provider.name
        );
    }
    Ok(())
}

/// Executes `goshcoder models [provider]`.
pub fn models_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::with_default_credentials()?;
    if let Some(provider_id) = arguments.first() {
        let Some(provider) = catalog.provider(provider_id) else {
            return Err(command_error(format!("unknown provider {provider_id:?}")));
        };
        print_models(&provider);
        return Ok(());
    }

    let configured = catalog.configured_provider_ids()?;
    if configured.is_empty() {
        eprintln!("No providers are configured. Run 'goshcoder providers' to see the options.");
        return Ok(());
    }
    for provider_id in configured {
        if let Some(provider) = catalog.provider(&provider_id) {
            print_models(&provider);
        }
    }
    Ok(())
}

/// Executes `goshcoder auth set|login|list|logout`.
pub fn auth_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let Some(subcommand) = arguments.first().map(String::as_str) else {
        return Err(command_error("usage: goshcoder auth set|login|list|logout"));
    };
    let store = CredentialStore::default_file();
    match subcommand {
        "list" => {
            let credentials = store.list()?;
            if credentials.is_empty() {
                println!("No stored credentials.");
            } else {
                for credential in credentials {
                    println!(
                        "{:<24} {}",
                        credential.provider_id,
                        credential.kind.as_str()
                    );
                }
            }
            Ok(())
        }
        "set" => {
            let Some(provider_id) = arguments.get(1) else {
                return Err(command_error("usage: goshcoder auth set <provider>"));
            };
            let catalog = Catalog::with_default_credentials()?;
            if catalog.provider(provider_id).is_none() {
                return Err(command_error(format!("unknown provider {provider_id:?}")));
            }
            config::ensure_agent_dir()?;
            let key = read_secret(&format!("Enter the API key for {provider_id}: "))?;
            if key.is_empty() {
                return Err(command_error("no key provided"));
            }
            store.put(provider_id, Credential::api_key(key))?;
            println!(
                "Stored an API key for {provider_id} in {}",
                config::auth_path().display()
            );
            Ok(())
        }
        "login" => {
            let Some(provider_id) = arguments.get(1) else {
                return Err(command_error("usage: goshcoder auth login <provider>"));
            };
            let catalog = Catalog::with_default_credentials()?;
            if catalog.provider(provider_id).is_none() {
                return Err(command_error(format!("unknown provider {provider_id:?}")));
            }
            let Some(provider) = oauth::OAuthProviderId::parse(provider_id) else {
                return Err(command_error(format!(
                    "{provider_id:?} does not support OAuth login; use `goshcoder auth set {provider_id}`"
                )));
            };
            if oauth::metadata_for(provider).flow_support == oauth::OAuthFlowSupport::MetadataOnly {
                return Err(command_error(format!(
                    "no OAuth login flow is available for {provider_id:?}; use `goshcoder auth set {provider_id}`"
                )));
            }
            config::ensure_agent_dir()?;
            let client = oauth::OAuthClient::system()?;
            let cancellation = oauth::CancellationToken::new();
            client.login_and_persist(
                provider,
                &store,
                Arc::new(TerminalOAuthInteraction),
                &oauth::ProcessEnvironment,
                &cancellation,
            )?;
            catalog.clear_oauth_refresh_failure(provider_id);
            println!(
                "Logged in to {provider_id} with OAuth; credentials are stored in {}",
                config::auth_path().display()
            );
            Ok(())
        }
        "logout" => {
            let Some(provider_id) = arguments.get(1) else {
                return Err(command_error("usage: goshcoder auth logout <provider>"));
            };
            store.delete(provider_id)?;
            println!("Removed the stored credential for {provider_id}");
            Ok(())
        }
        _ => Err(command_error(format!(
            "unknown auth subcommand {subcommand:?}"
        ))),
    }
}

/// Blocking CLI presentation for the provider-neutral OAuth flow.
///
/// Browser and device authorization remain owned by `oauth`; this adapter only
/// renders their safe instructions and turns a user selection or pasted
/// redirect into the value expected by the selected provider.
struct TerminalOAuthInteraction;

impl oauth::OAuthInteraction for TerminalOAuthInteraction {
    fn prompt(&self, prompt: oauth::OAuthPrompt) -> oauth::Result<String> {
        prompt.cancellation.check()?;
        eprintln!();
        eprintln!("{}", prompt.message);
        for (index, option) in prompt.options.iter().enumerate() {
            if option.description.is_empty() {
                eprintln!("  {}. {}", index + 1, option.label);
            } else {
                eprintln!("  {}. {} — {}", index + 1, option.label, option.description);
            }
        }
        if prompt.placeholder.is_empty() {
            eprint!("> ");
        } else {
            eprint!("{}: ", prompt.placeholder);
        }
        io::stderr()
            .flush()
            .map_err(|error| oauth::OAuthError::Callback(format!("write prompt: {error}")))?;

        let mut input = String::new();
        io::stdin()
            .lock()
            .read_line(&mut input)
            .map_err(|error| oauth::OAuthError::Callback(format!("read prompt: {error}")))?;
        prompt.cancellation.check()?;
        let input = input.trim().to_owned();
        if prompt.kind == oauth::OAuthPromptKind::Select {
            return Ok(select_oauth_option(&input, &prompt.options));
        }
        Ok(input)
    }

    fn notify(&self, event: oauth::OAuthEvent) {
        match event.kind {
            oauth::OAuthEventKind::AuthorizationUrl => {
                if let Some(url) = event.authorization_url {
                    eprintln!("Open this URL to continue login:\n{url}");
                }
                if !event.instructions.is_empty() {
                    eprintln!("{}", event.instructions);
                }
            }
            oauth::OAuthEventKind::DeviceCode => {
                eprintln!(
                    "Open {} and enter code {} (expires in {} minutes).",
                    event.verification_uri,
                    event.user_code,
                    event.expires_in_seconds.div_ceil(60)
                );
            }
            oauth::OAuthEventKind::Info | oauth::OAuthEventKind::Progress => {
                if !event.message.is_empty() {
                    eprintln!("{}", event.message);
                }
            }
        }
    }
}

fn select_oauth_option(input: &str, options: &[oauth::OAuthPromptOption]) -> String {
    if let Ok(index) = input.parse::<usize>()
        && let Some(option) = index.checked_sub(1).and_then(|index| options.get(index))
    {
        return option.id.clone();
    }
    input.to_owned()
}

fn print_models(provider: &Provider) {
    if provider.models().is_empty() {
        return;
    }
    println!("\n{} ({})", provider.name, provider.id);
    for model in provider.models() {
        println!("  {}/{}", provider.id, model.id);
    }
}

fn provider_setup_hint(provider: &Provider) -> String {
    let environment = provider.env_keys.join(" or ");
    if provider.supports_oauth {
        if environment.is_empty() {
            return format!("run: goshcoder auth login {}", provider.id);
        }
        return format!(
            "run: goshcoder auth login {} (or set {environment})",
            provider.id
        );
    }
    if !environment.is_empty() {
        return format!(
            "set {environment}, or run: goshcoder auth set {}",
            provider.id
        );
    }
    match provider.id.as_str() {
        "amazon-bedrock" => {
            "set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, AWS_PROFILE, or AWS_BEARER_TOKEN_BEDROCK"
                .to_owned()
        }
        "google-vertex" | "google-vertex-anthropic" => {
            "set GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION (or GOOGLE_CLOUD_API_KEY for express mode)"
                .to_owned()
        }
        "azure" | "azure-openai-responses" => {
            "set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT".to_owned()
        }
        _ => format!("run: goshcoder auth set {}", provider.id),
    }
}

pub(crate) fn read_secret(prompt: &str) -> io::Result<String> {
    let stdin = io::stdin();
    if !stdin.is_terminal() {
        let mut secret = String::new();
        stdin.lock().read_line(&mut secret)?;
        return Ok(secret.trim().to_owned());
    }

    eprint!("{prompt}");
    io::stderr().flush()?;
    let guard = RawModeGuard::enter()?;
    let result = (|| {
        let mut secret = String::new();
        loop {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind == KeyEventKind::Release {
                continue;
            }
            match key.code {
                KeyCode::Enter => return Ok(secret),
                KeyCode::Backspace => {
                    secret.pop();
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "input cancelled",
                    ));
                }
                KeyCode::Char(character)
                    if !key.modifiers.contains(KeyModifiers::CONTROL)
                        && !key.modifiers.contains(KeyModifiers::ALT) =>
                {
                    secret.push(character);
                }
                _ => {}
            }
        }
    })();
    drop(guard);
    eprintln!();
    result.map(|secret| secret.trim().to_owned())
}

struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

fn command_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Catalog, CredentialStore};
    use std::sync::Arc;

    #[test]
    fn setup_hints_explain_environment_oauth_and_ambient_providers() {
        let aperture_root = std::env::temp_dir().join(format!(
            "goshcoder-provider-cli-hints-{}",
            std::process::id()
        ));
        let catalog = Catalog::with_environment(
            Some(Arc::new(CredentialStore::in_memory())),
            Arc::new(|_| None),
        )
        .expect("catalog")
        .with_aperture_paths(
            aperture_root.join("aperture.json"),
            aperture_root.join("aperture-cache.json"),
        );
        assert!(
            provider_setup_hint(&catalog.provider("openai").expect("OpenAI"))
                .contains("OPENAI_API_KEY")
        );
        assert!(
            provider_setup_hint(&catalog.provider("openrouter").expect("OpenRouter"))
                .contains("auth login openrouter")
        );
        assert!(
            provider_setup_hint(&catalog.provider("amazon-bedrock").expect("Bedrock"))
                .contains("AWS_ACCESS_KEY_ID")
        );
    }

    #[test]
    fn model_listing_only_uses_configured_provider_ids_without_leaking_keys() {
        let credentials = Arc::new(CredentialStore::in_memory());
        credentials
            .put("openai", Credential::api_key("secret"))
            .expect("store credential");
        let aperture_root = std::env::temp_dir().join(format!(
            "goshcoder-provider-cli-models-{}",
            std::process::id()
        ));
        let catalog = Catalog::with_environment(
            Some(credentials),
            Arc::new(|name| (name == "UNUSED").then_some("unused".to_owned())),
        )
        .expect("catalog")
        .with_aperture_paths(
            aperture_root.join("aperture.json"),
            aperture_root.join("aperture-cache.json"),
        );
        assert_eq!(
            catalog
                .configured_provider_ids()
                .expect("configured providers"),
            vec!["openai"]
        );
        assert!(
            catalog
                .resolve_auth("openai")
                .expect("resolve auth")
                .is_some()
        );
    }

    #[test]
    fn oauth_numbered_selections_resolve_to_provider_method_ids() {
        let options = vec![
            oauth::OAuthPromptOption {
                id: "browser".to_owned(),
                label: "Browser".to_owned(),
                description: String::new(),
            },
            oauth::OAuthPromptOption {
                id: "device_code".to_owned(),
                label: "Device code".to_owned(),
                description: String::new(),
            },
        ];
        assert_eq!(select_oauth_option("2", &options), "device_code");
        assert_eq!(select_oauth_option("browser", &options), "browser");
        assert_eq!(select_oauth_option("3", &options), "3");
    }
}
