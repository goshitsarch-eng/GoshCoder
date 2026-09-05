//! Command-line adapters for the built-in model catalog and credential store.

use std::{
    error::Error,
    io::{self, BufRead, IsTerminal, Write},
};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};

use crate::{
    catalog::{Catalog, Credential, CredentialStore, Provider},
    config,
};

/// Executes `goshcoder providers`.
pub fn providers_command() -> Result<(), Box<dyn Error>> {
    let catalog = Catalog::with_default_credentials()?;
    for provider in catalog.providers() {
        let (status, detail) = match catalog.resolve_auth(&provider.id)? {
            Some(authentication) => {
                let ambient = authentication
                    .is_ambient()
                    .then_some(" (ambient)")
                    .unwrap_or("");
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
            Err(command_error(
                "OAuth login has not yet been migrated; use `goshcoder auth set <provider>` with an API key where supported",
            ))
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
            "set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT".to_owned()
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
        let catalog = Catalog::with_environment(
            Some(Arc::new(CredentialStore::in_memory())),
            Arc::new(|_| None),
        )
        .expect("catalog");
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
        let catalog = Catalog::with_environment(
            Some(credentials),
            Arc::new(|name| (name == "UNUSED").then_some("unused".to_owned())),
        )
        .expect("catalog");
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
}
