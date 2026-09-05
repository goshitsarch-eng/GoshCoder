//! Terminal command adapter for the Tailscale Aperture integration.
//!
//! The Aperture core deliberately owns persistence, routing, and MCP protocol
//! details. This module owns only the user-facing command surface, keeping it
//! ready for a later `main.rs` dispatch entry without duplicating those
//! security-sensitive primitives.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    io::{self, BufRead, IsTerminal, Write},
    path::Path,
    time::Duration,
};

use url::Url;

use crate::{aperture, aperture_mcp, catalog::Catalog, config, llm};

const CONNECTOR_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECTOR_CALL_TIMEOUT: Duration = Duration::from_secs(15);
const CONNECTOR_NOTIFICATION_TIMEOUT: Duration = Duration::from_secs(5);

// Keep an accidental API embedding from turning a command invocation into an
// oversized persisted config or terminal response. Gateway and MCP response
// limits are separately enforced by their respective primitives.
const MAX_ARGUMENTS: usize = 64;
const MAX_ARGUMENT_BYTES: usize = 8 << 10;
const MAX_RENDERED_ITEMS: usize = 100;
const MAX_DISPLAY_BYTES: usize = 512;
const MAX_OUTPUT_BYTES: usize = 64 << 10;

type CliResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Subcommand {
    Status,
    Onboarding,
    Settings,
    Sync,
    Providers,
    Connectors,
    Pin,
    Unpin,
    Help,
}

#[derive(Clone, Debug)]
struct IndexedChoice {
    id: String,
    label: String,
}

/// Executes `goshcoder aperture [subcommand]`.
///
/// Onboarding is intentionally unavailable when either standard input or
/// standard error is not a terminal: it never consumes piped input or writes a
/// partial configuration from an unattended process. All non-interactive
/// diagnostics and mutations remain available for automation.
pub fn command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
    let output = execute(arguments, interactive)?;
    if !output.is_empty() {
        write_output(&output)?;
    }
    Ok(())
}

/// Executes an Aperture command and returns its safe, terminal-ready output.
///
/// The interactive frontend uses this instead of writing directly to stdout,
/// so status, sync, and settings results remain visible in its transcript.
pub fn execute(arguments: &[String], interactive: bool) -> Result<String, Box<dyn Error>> {
    let (subcommand, remaining) = parse_invocation(arguments)?;
    match subcommand {
        Subcommand::Status => status_command(),
        Subcommand::Onboarding => onboarding_command(interactive),
        Subcommand::Settings => settings_command(remaining),
        Subcommand::Sync => sync_command(),
        Subcommand::Providers => providers_command(),
        Subcommand::Connectors => connectors_command(),
        Subcommand::Pin => {
            let tool_name = exactly_one_argument(remaining, "goshcoder aperture pin <toolName>")?;
            pin_command(tool_name)
        }
        Subcommand::Unpin => {
            let tool_name = exactly_one_argument(remaining, "goshcoder aperture unpin <toolName>")?;
            unpin_command(tool_name)
        }
        Subcommand::Help => Ok(help_text().to_owned()),
    }
}

fn parse_invocation(arguments: &[String]) -> CliResult<(Subcommand, &[String])> {
    validate_arguments(arguments)?;
    let Some(first) = arguments
        .first()
        .filter(|argument| !argument.trim().is_empty())
    else {
        return Ok((Subcommand::Status, &[]));
    };

    // Accept a complete slash command too. This is useful to callers shared
    // with chat command routing, while normal CLI wiring passes only the
    // arguments after `aperture`.
    let (name, remaining) = if is_aperture_root(first) {
        match arguments.get(1) {
            Some(next) if !next.trim().is_empty() => (next.as_str(), &arguments[2..]),
            _ => return Ok((Subcommand::Status, &[])),
        }
    } else {
        (first.as_str(), &arguments[1..])
    };

    let subcommand = parse_subcommand(name).ok_or_else(|| {
        command_error(format!(
            "unknown Aperture command {:?}; use status, onboarding, settings, sync, providers, connectors, pin, or unpin",
            display_value(name, MAX_DISPLAY_BYTES)
        ))
    })?;
    Ok((subcommand, remaining))
}

fn validate_arguments(arguments: &[String]) -> CliResult<()> {
    if arguments.len() > MAX_ARGUMENTS {
        return Err(command_error(format!(
            "Aperture accepts at most {MAX_ARGUMENTS} arguments"
        )));
    }
    if arguments
        .iter()
        .any(|argument| argument.len() > MAX_ARGUMENT_BYTES)
    {
        return Err(command_error(format!(
            "an Aperture argument exceeds {MAX_ARGUMENT_BYTES} bytes"
        )));
    }
    Ok(())
}

fn is_aperture_root(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "aperture" | "/aperture"
    )
}

fn parse_subcommand(value: &str) -> Option<Subcommand> {
    let normalized = value.trim().to_ascii_lowercase();
    let normalized = normalized.strip_prefix('/').unwrap_or(&normalized);
    let normalized = normalized.strip_prefix("aperture:").unwrap_or(normalized);
    match normalized {
        "" | "status" | "show" | "info" => Some(Subcommand::Status),
        "onboarding" | "setup" | "configure" | ":onboarding" => Some(Subcommand::Onboarding),
        "settings" | "setting" | "config" | ":settings" => Some(Subcommand::Settings),
        "sync" | "refresh" => Some(Subcommand::Sync),
        "providers" | "provider" => Some(Subcommand::Providers),
        "connectors" | "connector" | "tools" => Some(Subcommand::Connectors),
        "pin" | "add" => Some(Subcommand::Pin),
        "unpin" | "remove" => Some(Subcommand::Unpin),
        "help" | "-h" | "--help" | "?" => Some(Subcommand::Help),
        _ => None,
    }
}

fn exactly_one_argument<'a>(arguments: &'a [String], usage: &str) -> CliResult<&'a str> {
    if arguments.len() != 1 {
        return Err(command_error(format!("usage: {usage}")));
    }
    Ok(&arguments[0])
}

fn help_text() -> &'static str {
    r#"Usage: goshcoder aperture [subcommand]

Subcommands:
  status                         Show gateway and cached configuration status
  onboarding | setup             Configure an Aperture gateway interactively
  settings [<key> <value>]       Show or change configuration settings
  sync | refresh                 Refresh the gateway model snapshot
  providers                      List gateway providers and routing APIs
  connectors | tools             List gateway connector tools
  pin <toolName>                 Pin a connector tool for the next session
  unpin <toolName>               Remove a pinned connector tool

Compatibility aliases: /aperture:onboarding and /aperture:settings.
"#
}

fn load_configuration_at(path: &Path) -> CliResult<(aperture::Config, bool)> {
    match aperture::load_config(path) {
        Ok(configuration) => Ok((configuration, true)),
        Err(aperture::ApertureError::Io { source, .. })
            if source.kind() == io::ErrorKind::NotFound =>
        {
            Ok((aperture::Config::default(), false))
        }
        Err(error) => Err(Box::new(error)),
    }
}

fn status_command() -> CliResult<String> {
    status_at(&config::aperture_path(), &config::aperture_cache_path())
}

fn status_at(configuration_path: &Path, cache_path: &Path) -> CliResult<String> {
    let (configuration, exists) = load_configuration_at(configuration_path)?;
    let resolved = configuration.resolve();
    if !exists || resolved.base_url.trim().is_empty() {
        return Ok("Aperture is unconfigured. Run `goshcoder aperture onboarding`.".to_owned());
    }

    let gateway = gateway_from_resolved(&resolved)?;
    let health = match aperture::GatewayClient::new(&gateway).and_then(|client| client.health()) {
        Ok(()) => "healthy".to_owned(),
        Err(error) => format!("DOWN ({})", gateway_error_message(&error)),
    };

    let mut lines = vec![
        format!("Aperture: {health}"),
        format!("Gateway: {gateway}"),
        format!(
            "Dedicated provider: {}",
            enabled_word(resolved.dedicated_enabled)
        ),
    ];
    if resolved.dedicated_enabled {
        let count = aperture::load_cache(cache_path)
            .ok()
            .and_then(|cache| {
                cache.catalog_models(&aperture::build_catalog_key(&gateway, &resolved))
            })
            .map_or(0, |models| models.len());
        lines.push(format!(
            "  Synchronized models: {count} (refresh with `goshcoder aperture sync`)"
        ));
    }

    lines.push(format!("Proxy: {}", enabled_word(resolved.proxy_enabled)));
    if resolved.proxy_enabled {
        let providers = resolved
            .enabled_upstream_providers()
            .into_iter()
            .map(|provider| provider.id);
        lines.push(format!(
            "  Upstream providers: {}",
            display_join(providers, "provider")
        ));
    }

    lines.push(format!(
        "Connectors: {}",
        enabled_word(resolved.connectors_enabled)
    ));
    if resolved.connectors_enabled {
        lines.push(format!(
            "  Discovery tools: {} · pinned: {}",
            enabled_word(resolved.discovery_tools),
            resolved.pinned_tools.len()
        ));
    }
    Ok(lines.join("\n"))
}

fn onboarding_command(interactive: bool) -> CliResult<String> {
    if !interactive {
        return Err(command_error(
            "Aperture onboarding requires an interactive terminal",
        ));
    }

    let stdin = io::stdin();
    let stderr = io::stderr();
    onboarding_at(
        &config::aperture_path(),
        &config::aperture_cache_path(),
        &mut stdin.lock(),
        &mut stderr.lock(),
    )
}

fn onboarding_at<R: BufRead, W: Write>(
    configuration_path: &Path,
    cache_path: &Path,
    input: &mut R,
    output: &mut W,
) -> CliResult<String> {
    let (existing, _) = load_configuration_at(configuration_path)?;
    let resolved = existing.resolve();

    writeln!(
        output,
        "Aperture lets you route LLM traffic through your Tailscale tailnet."
    )?;
    writeln!(output, "You can use it two ways:")?;
    writeln!(
        output,
        "  - Dedicated provider: a standalone \"aperture\" provider with all models from your gateway"
    )?;
    writeln!(
        output,
        "  - Proxy: reroute existing providers (for example anthropic and openai) through Aperture"
    )?;
    writeln!(
        output,
        "You can change these settings later with `goshcoder aperture settings`."
    )?;
    writeln!(output)?;

    let gateway = loop {
        let default = if resolved.base_url.is_empty() {
            String::new()
        } else {
            format!(
                " [{}]",
                display_value(&resolved.base_url, MAX_DISPLAY_BYTES)
            )
        };
        let entered = prompt_line(
            input,
            output,
            &format!("Aperture base URL (for example ai.pango-lin.ts.net){default}: "),
        )?;
        let entered = if entered.is_empty() {
            resolved.base_url.clone()
        } else {
            entered
        };
        if entered.trim().is_empty() {
            writeln!(output, "A gateway URL is required.")?;
            continue;
        }
        let candidate = match normalize_gateway_input(&entered) {
            Ok(candidate) => candidate,
            Err(error) => {
                writeln!(
                    output,
                    "Could not use that URL: {}. Fix the URL and press Enter to retry.",
                    display_value(&error.to_string(), MAX_DISPLAY_BYTES)
                )?;
                continue;
            }
        };
        writeln!(output, "Checking connection...")?;
        match aperture::GatewayClient::new(&candidate).and_then(|client| client.health()) {
            Ok(()) => {
                writeln!(output, "Connected.")?;
                break candidate;
            }
            Err(error) => writeln!(
                output,
                "Could not connect: {}. Fix the URL and press Enter to retry.",
                gateway_error_message(&error)
            )?,
        }
    };

    writeln!(output, "\nHow do you want to use Aperture?")?;
    writeln!(
        output,
        "  1. Dedicated only — all gateway models under one aperture provider"
    )?;
    writeln!(
        output,
        "  2. Proxy only — reroute existing providers, keeping their model definitions"
    )?;
    writeln!(output, "  3. Both")?;
    let choice = prompt_line(input, output, "Choice [1]: ")?;
    let (dedicated_enabled, proxy_enabled) = match choice.trim() {
        "" | "1" => (true, false),
        "2" => (false, true),
        "3" => (true, true),
        other => {
            return Err(command_error(format!(
                "unknown choice {:?}",
                display_value(other, 64)
            )));
        }
    };

    let gateway_providers = fetch_gateway_providers(&gateway)?;

    let mut dedicated_providers = Vec::new();
    if dedicated_enabled {
        dedicated_providers =
            aperture::map_dedicated_providers(&gateway_providers, &resolved.dedicated_providers);
        if dedicated_providers.is_empty() {
            writeln!(output, "\nNo providers found on the Aperture gateway.")?;
        } else {
            writeln!(output, "\nSelect Aperture providers to include:")?;
            let choices = dedicated_providers
                .iter()
                .map(|provider| IndexedChoice {
                    id: provider.id.clone(),
                    label: display_name(&provider.name, &provider.id),
                })
                .collect::<Vec<_>>();
            let mut checked = dedicated_providers
                .iter()
                .map(|provider| (provider.id.clone(), provider.enabled))
                .collect::<BTreeMap<_, _>>();
            select_by_index(input, output, &choices, &mut checked)?;
            for provider in &mut dedicated_providers {
                provider.enabled = checked.get(&provider.id).copied().unwrap_or(false);
            }
        }
    }

    let mut upstream_providers = Vec::new();
    if proxy_enabled {
        let local_models = all_catalog_models()?;
        let mapped = aperture::map_proxy_providers(
            &local_models,
            &gateway_providers,
            &resolved.upstream_providers,
        );
        if mapped.is_empty() {
            writeln!(
                output,
                "\nNo local providers match the Aperture gateway providers."
            )?;
            writeln!(
                output,
                "You can add proxy providers later with `goshcoder aperture settings`."
            )?;
        } else {
            writeln!(output, "\nSelect providers to route through Aperture:")?;
            let choices = mapped
                .iter()
                .map(|provider| IndexedChoice {
                    id: provider.id.clone(),
                    label: display_name(&provider.name, &provider.id),
                })
                .collect::<Vec<_>>();
            let mut checked = mapped
                .iter()
                .map(|provider| (provider.id.clone(), provider.enabled))
                .collect::<BTreeMap<_, _>>();
            select_by_index(input, output, &choices, &mut checked)?;
            let check_answer = prompt_line(
                input,
                output,
                "Warn when local models are missing from the gateway? [Y/n]: ",
            )?;
            let should_check_gateway_models = !check_answer.trim().eq_ignore_ascii_case("n");
            for provider in mapped {
                if checked.get(&provider.id).copied().unwrap_or(false) {
                    upstream_providers.push(aperture::ProxiedProviderConfig {
                        id: provider.id,
                        should_check_gateway_models,
                        ..aperture::ProxiedProviderConfig::default()
                    });
                }
            }
        }
    }

    writeln!(output, "\nRecap:")?;
    writeln!(output, "  URL: {gateway}")?;
    let capabilities = match (dedicated_enabled, proxy_enabled) {
        (true, true) => "Dedicated provider and proxy",
        (true, false) => "Dedicated provider",
        (false, true) => "Proxy existing providers",
        (false, false) => "None",
    };
    writeln!(output, "  Capabilities: {capabilities}")?;
    if proxy_enabled {
        let selected = upstream_providers
            .iter()
            .map(|provider| provider.id.clone());
        writeln!(
            output,
            "  Upstream providers: {}",
            display_join(selected, "provider")
        )?;
    }
    if dedicated_enabled {
        if dedicated_providers.is_empty() {
            writeln!(output, "  Aperture providers: all (no filter)")?;
        } else {
            let enabled = dedicated_providers
                .iter()
                .filter(|provider| provider.enabled)
                .count();
            writeln!(
                output,
                "  Aperture providers: {enabled}/{} enabled",
                dedicated_providers.len()
            )?;
        }
    }

    let confirmation = prompt_line(input, output, "Save? [Y/n]: ")?;
    if confirmation.trim().eq_ignore_ascii_case("n") {
        return Ok("Aperture onboarding cancelled.".to_owned());
    }

    let mut updated = existing;
    updated.base_url = gateway;
    updated.onboarding_done = Some(true);
    updated.onboarding = Some(aperture::OnboardingConfig {
        enabled: Some(false),
    });
    set_proxy_config(&mut updated, proxy_enabled, upstream_providers);
    set_dedicated_config(&mut updated, dedicated_enabled, dedicated_providers);
    aperture::save_config(configuration_path, &updated)?;

    match sync_configuration(&updated, cache_path) {
        Ok(result) => Ok(format!(
            "Aperture onboarding completed. {}",
            render_sync_result(&result)
        )),
        Err(error) => Ok(format!(
            "Aperture onboarding completed, but the first sync failed: {}",
            display_value(&error.to_string(), MAX_DISPLAY_BYTES)
        )),
    }
}

fn prompt_line<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    prompt: &str,
) -> CliResult<String> {
    write!(output, "{prompt}")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err(command_error(
            "Aperture onboarding cancelled because standard input was closed",
        ));
    }
    Ok(line.trim().to_owned())
}

fn select_by_index<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    choices: &[IndexedChoice],
    checked: &mut BTreeMap<String, bool>,
) -> CliResult<()> {
    for (index, choice) in choices.iter().enumerate() {
        let marker = if checked.get(&choice.id).copied().unwrap_or(false) {
            'x'
        } else {
            ' '
        };
        writeln!(
            output,
            "  {:>2}. [{marker}] {}",
            index + 1,
            display_value(&choice.label, MAX_DISPLAY_BYTES)
        )?;
    }
    let selection = prompt_line(
        input,
        output,
        "Toggle by number (for example \"1 3\"), or \"all\"/\"none\"; Enter keeps the selection: ",
    )?;
    match selection.trim().to_ascii_lowercase().as_str() {
        "" => {}
        "all" => {
            for choice in choices {
                checked.insert(choice.id.clone(), true);
            }
        }
        "none" => {
            for choice in choices {
                checked.insert(choice.id.clone(), false);
            }
        }
        selection => {
            for field in selection
                .split(|character: char| character == ',' || character.is_ascii_whitespace())
            {
                if field.is_empty() {
                    continue;
                }
                let index = field.parse::<usize>().map_err(|_| {
                    command_error(format!("invalid selection {:?}", display_value(field, 64)))
                })?;
                let Some(choice) = index.checked_sub(1).and_then(|index| choices.get(index)) else {
                    return Err(command_error(format!(
                        "invalid selection {:?}",
                        display_value(field, 64)
                    )));
                };
                let enabled = checked.get(&choice.id).copied().unwrap_or(false);
                checked.insert(choice.id.clone(), !enabled);
            }
        }
    }
    Ok(())
}

fn settings_command(arguments: &[String]) -> CliResult<String> {
    settings_at(
        arguments,
        &config::aperture_path(),
        &config::aperture_cache_path(),
    )
}

fn settings_at(
    arguments: &[String],
    configuration_path: &Path,
    cache_path: &Path,
) -> CliResult<String> {
    let (mut configuration, exists) = load_configuration_at(configuration_path)?;
    if arguments.is_empty() {
        if !exists {
            return Ok(
                "Aperture is unconfigured. Run `goshcoder aperture onboarding`, or set a URL with `goshcoder aperture settings baseUrl <url>`."
                    .to_owned(),
            );
        }
        return Ok(settings_summary(&configuration));
    }
    if arguments.len() != 2 {
        return Err(command_error(
            "usage: goshcoder aperture settings [<key> <value>]; run `goshcoder aperture settings` to list keys",
        ));
    }

    let key = &arguments[0];
    let value = &arguments[1];
    apply_setting(&mut configuration, key, value)?;
    aperture::save_config(configuration_path, &configuration)?;

    let set_message = format!(
        "Set {} = {}.",
        display_value(key, MAX_DISPLAY_BYTES),
        display_value(value, MAX_DISPLAY_BYTES)
    );
    match sync_configuration(&configuration, cache_path) {
        Ok(result) => Ok(format!("{set_message}\n{}", render_sync_result(&result))),
        Err(_) => Ok(format!(
            "{set_message} Run `goshcoder aperture sync` to refresh the catalog."
        )),
    }
}

fn settings_summary(configuration: &aperture::Config) -> String {
    let resolved = configuration.resolve();
    let mut lines = vec![
        "Aperture settings (change with `goshcoder aperture settings <key> <value>`):".to_owned(),
        format!(
            "  baseUrl                     = {}",
            or_unset(&resolved.base_url)
        ),
        format!(
            "  onboardingDone              = {}",
            resolved.onboarding_done
        ),
        format!(
            "  onboarding.enabled          = {}",
            resolved.onboarding_enabled
        ),
        format!("  proxy.enabled               = {}", resolved.proxy_enabled),
        format!(
            "  dedicated.enabled           = {}",
            resolved.dedicated_enabled
        ),
        format!(
            "  connectors.enabled          = {}",
            resolved.connectors_enabled
        ),
        format!(
            "  connectors.discoveryTools   = {}",
            resolved.discovery_tools
        ),
    ];

    if !resolved.upstream_providers.is_empty() {
        lines.push(
            "Proxy providers (proxy.provider.<id>.enabled|check|gatewayModelsOnly|api):".to_owned(),
        );
        for provider in resolved.upstream_providers.iter().take(MAX_RENDERED_ITEMS) {
            let api = if provider.api.is_empty() {
                "auto"
            } else {
                provider.api.as_str()
            };
            lines.push(format!(
                "  {:<16} enabled={} check={} gatewayModelsOnly={} api={}",
                display_value(&provider.id, 64),
                provider.is_enabled(),
                provider.should_check_gateway_models,
                provider.keep_gateway_models_only,
                display_value(api, 64),
            ));
        }
        append_omitted(
            &mut lines,
            resolved.upstream_providers.len(),
            "proxy provider",
        );
    }

    if !resolved.dedicated_providers.is_empty() {
        lines.push("Dedicated providers (dedicated.provider.<id>.enabled|api):".to_owned());
        for provider in resolved.dedicated_providers.iter().take(MAX_RENDERED_ITEMS) {
            let api = if provider.api.is_empty() {
                "auto"
            } else {
                provider.api.as_str()
            };
            lines.push(format!(
                "  {:<16} enabled={} api={}",
                display_value(&provider.id, 64),
                provider.enabled,
                display_value(api, 64),
            ));
        }
        append_omitted(
            &mut lines,
            resolved.dedicated_providers.len(),
            "dedicated provider",
        );
    } else if resolved.dedicated_enabled {
        lines.push("Dedicated providers: all (no filter)".to_owned());
    }

    if !resolved.pinned_tools.is_empty() {
        let mut names = resolved
            .pinned_tools
            .iter()
            .map(|pin| pin.tool_name.clone())
            .collect::<Vec<_>>();
        names.sort();
        lines.push(format!(
            "Pinned connector tools: {}",
            display_join(names, "tool")
        ));
    }
    lines.join("\n")
}

fn apply_setting(configuration: &mut aperture::Config, key: &str, value: &str) -> CliResult<()> {
    let resolved = configuration.resolve();
    match key {
        "baseUrl" => {
            configuration.base_url = if value.trim().is_empty() {
                String::new()
            } else {
                normalize_gateway_input(value)?
            };
            return Ok(());
        }
        "onboardingDone" => {
            let done = parse_setting_bool(value)?;
            configuration.onboarding_done = Some(done);
            configuration.onboarding = Some(aperture::OnboardingConfig {
                enabled: Some(!done),
            });
            return Ok(());
        }
        "onboarding.enabled" => {
            configuration.onboarding = Some(aperture::OnboardingConfig {
                enabled: Some(parse_setting_bool(value)?),
            });
            return Ok(());
        }
        "proxy.enabled" => {
            set_proxy_config(
                configuration,
                parse_setting_bool(value)?,
                resolved.upstream_providers,
            );
            return Ok(());
        }
        "dedicated.enabled" => {
            set_dedicated_config(
                configuration,
                parse_setting_bool(value)?,
                resolved.dedicated_providers,
            );
            return Ok(());
        }
        "connectors.enabled" => {
            configuration
                .connectors
                .get_or_insert_with(aperture::ConnectorsConfig::default)
                .enabled = parse_setting_bool(value)?;
            return Ok(());
        }
        "connectors.discoveryTools" => {
            configuration
                .connectors
                .get_or_insert_with(aperture::ConnectorsConfig::default)
                .discovery_tools = Some(parse_setting_bool(value)?);
            return Ok(());
        }
        _ => {}
    }

    if let Some((id, field)) = provider_setting_key(key, "proxy.provider.") {
        let mut providers = resolved.upstream_providers;
        let index = providers
            .iter()
            .position(|provider| provider.id == id)
            .unwrap_or_else(|| {
                providers.push(aperture::ProxiedProviderConfig {
                    id: id.to_owned(),
                    should_check_gateway_models: true,
                    ..aperture::ProxiedProviderConfig::default()
                });
                providers.len() - 1
            });
        match field {
            "enabled" => providers[index].enabled = Some(parse_setting_bool(value)?),
            "check" | "shouldCheckGatewayModels" => {
                providers[index].should_check_gateway_models = parse_setting_bool(value)?;
            }
            "gatewayModelsOnly" | "keepGatewayModelsOnly" => {
                providers[index].keep_gateway_models_only = parse_setting_bool(value)?;
            }
            "api" => providers[index].api = parse_setting_api(value)?,
            _ => {
                return Err(command_error(format!(
                    "unknown proxy provider setting {:?} (enabled, check, gatewayModelsOnly, api)",
                    display_value(field, 64)
                )));
            }
        }
        set_proxy_config(configuration, resolved.proxy_enabled, providers);
        return Ok(());
    }

    if let Some((id, field)) = provider_setting_key(key, "dedicated.provider.") {
        let mut providers = resolved.dedicated_providers;
        let index = providers
            .iter()
            .position(|provider| provider.id == id)
            .unwrap_or_else(|| {
                providers.push(aperture::DedicatedProviderConfig {
                    id: id.to_owned(),
                    enabled: true,
                    ..aperture::DedicatedProviderConfig::default()
                });
                providers.len() - 1
            });
        match field {
            "enabled" => providers[index].enabled = parse_setting_bool(value)?,
            "api" => providers[index].api = parse_setting_api(value)?,
            _ => {
                return Err(command_error(format!(
                    "unknown dedicated provider setting {:?} (enabled, api)",
                    display_value(field, 64)
                )));
            }
        }
        set_dedicated_config(configuration, resolved.dedicated_enabled, providers);
        return Ok(());
    }

    Err(command_error(format!(
        "unknown Aperture setting {:?}; run `goshcoder aperture settings` to list keys",
        display_value(key, MAX_DISPLAY_BYTES)
    )))
}

fn parse_setting_bool(value: &str) -> CliResult<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "enabled" | "on" | "yes" | "completed" => Ok(true),
        "false" | "disabled" | "off" | "no" | "pending" => Ok(false),
        _ => Err(command_error(format!(
            "expected enabled/disabled, got {:?}",
            display_value(value, 64)
        ))),
    }
}

fn parse_setting_api(value: &str) -> CliResult<aperture::RoutableApi> {
    let value = value.trim();
    if value.is_empty() || value == "auto" {
        return Ok(String::new());
    }
    if matches!(
        value,
        "openai-completions"
            | "anthropic-messages"
            | "openai-responses"
            | "google-generative-ai"
            | "google-vertex"
            | "bedrock-converse-stream"
    ) {
        return Ok(value.to_owned());
    }
    Err(command_error(format!(
        "unknown api {:?} (auto, openai-completions, anthropic-messages, openai-responses, google-generative-ai, google-vertex, bedrock-converse-stream)",
        display_value(value, 64)
    )))
}

fn provider_setting_key<'a>(key: &'a str, prefix: &str) -> Option<(&'a str, &'a str)> {
    let remainder = key.strip_prefix(prefix)?;
    let (id, field) = remainder.rsplit_once('.')?;
    (!id.is_empty() && !field.is_empty()).then_some((id, field))
}

fn set_proxy_config(
    configuration: &mut aperture::Config,
    enabled: bool,
    providers: Vec<aperture::ProxiedProviderConfig>,
) {
    configuration.proxy = Some(aperture::ProxyConfig {
        enabled: Some(enabled),
        upstream_providers: Some(providers),
    });
}

fn set_dedicated_config(
    configuration: &mut aperture::Config,
    enabled: bool,
    providers: Vec<aperture::DedicatedProviderConfig>,
) {
    configuration.dedicated = Some(aperture::DedicatedConfig {
        enabled: Some(enabled),
        providers: Some(providers),
        cached_models: None,
    });
}

fn sync_command() -> CliResult<String> {
    let (configuration, exists) = load_configuration_at(&config::aperture_path())?;
    if !exists {
        return Err(unconfigured_error());
    }
    let result = sync_configuration(&configuration, &config::aperture_cache_path())?;
    Ok(render_sync_result(&result))
}

fn sync_configuration(
    configuration: &aperture::Config,
    cache_path: &Path,
) -> CliResult<aperture::SyncResult> {
    let local_models = all_catalog_models()?;
    aperture::sync(configuration, cache_path, &local_models).map_err(Into::into)
}

#[cfg(test)]
fn sync_configuration_with_metadata(
    configuration: &aperture::Config,
    cache_path: &Path,
    local_models: &[llm::Model],
    models_dev: Option<&aperture::ModelsDevCatalog>,
) -> CliResult<aperture::SyncResult> {
    aperture::sync_with_models_dev(configuration, cache_path, local_models, models_dev)
        .map_err(Into::into)
}

fn render_sync_result(result: &aperture::SyncResult) -> String {
    let mut lines = vec![format!(
        "Synchronized {} Aperture models from {} gateway provider(s). They are available under /model immediately.",
        result.models.len(),
        result.gateway.len()
    )];
    for warning in result.warnings.iter().take(MAX_RENDERED_ITEMS) {
        lines.push(display_value(warning, MAX_DISPLAY_BYTES));
    }
    append_omitted(&mut lines, result.warnings.len(), "sync warning");
    lines.join("\n")
}

fn all_catalog_models() -> CliResult<Vec<llm::Model>> {
    // Constructing the normal catalog keeps the static model view aligned with
    // the user's durable credential setup without resolving, logging, or
    // transmitting any credential values.
    let catalog = Catalog::with_default_credentials()?;
    Ok(catalog
        .providers()
        .into_iter()
        .flat_map(|provider| provider.models())
        .collect())
}

fn providers_command() -> CliResult<String> {
    let (configuration, exists) = load_configuration_at(&config::aperture_path())?;
    let resolved = configuration.resolve();
    if !exists || resolved.base_url.trim().is_empty() {
        return Err(unconfigured_error());
    }
    let providers = fetch_gateway_providers(&gateway_from_resolved(&resolved)?)?;
    if providers.is_empty() {
        return Ok("No providers found on the Aperture gateway.".to_owned());
    }

    let mut lines = vec![format!("{} gateway provider(s):", providers.len())];
    for provider in providers.iter().take(MAX_RENDERED_ITEMS) {
        let apis = aperture::selectable_apis(&provider.compatibility);
        let routing = match apis.split_first() {
            None => "no routable api".to_owned(),
            Some((first, [])) => format!("auto: {first}"),
            Some((first, rest)) => format!("auto: {first} (also {})", rest.join(", ")),
        };
        let authentication = if provider.requires_client_auth {
            " · client auth required"
        } else {
            ""
        };
        lines.push(format!(
            "  {} ({}) — {} model(s) — {}{}",
            display_name(&provider.name, &provider.id),
            display_value(&provider.id, 128),
            provider.models.len(),
            routing,
            authentication
        ));
    }
    append_omitted(&mut lines, providers.len(), "gateway provider");
    Ok(lines.join("\n"))
}

fn connectors_command() -> CliResult<String> {
    let (configuration, exists) = load_configuration_at(&config::aperture_path())?;
    let resolved = configuration.resolve();
    if !exists || resolved.base_url.trim().is_empty() {
        return Err(unconfigured_error());
    }
    let gateway = gateway_from_resolved(&resolved)?;
    let tools = list_connector_tools(&gateway)?;
    // Connector metadata improves grouping but is deliberately non-fatal:
    // `tools/list` remains the authoritative usable-tool response.
    let connectors = aperture::GatewayClient::new(&gateway)
        .ok()
        .and_then(|client| client.connectors().ok())
        .unwrap_or_default();
    Ok(render_connectors(&resolved, &connectors, &tools))
}

fn render_connectors(
    resolved: &aperture::Resolved,
    connectors: &[aperture::ConnectorInfo],
    tools: &[aperture::GatewayTool],
) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for tool in tools {
        let count = counts
            .entry(aperture::connector_id_from_tool_name(&tool.name))
            .or_default();
        *count = count.saturating_add(1);
    }
    let pinned = resolved
        .pinned_tools
        .iter()
        .map(|pin| pin.tool_name.as_str())
        .collect::<BTreeSet<_>>();

    let mut lines = vec![format!(
        "Connectors feature: {} · discovery tools: {} · {} pinned",
        enabled_word(resolved.connectors_enabled),
        enabled_word(resolved.discovery_tools),
        resolved.pinned_tools.len()
    )];
    let visible_connectors = connectors
        .iter()
        .filter(|connector| counts.get(&connector.id).copied().unwrap_or_default() > 0)
        .collect::<Vec<_>>();
    for connector in visible_connectors.iter().take(MAX_RENDERED_ITEMS) {
        lines.push(format!(
            "  {} ({}): {} tool(s) — {}",
            display_name(&connector.provider, &connector.id),
            display_value(&connector.id, 128),
            counts.get(&connector.id).copied().unwrap_or_default(),
            display_value(&connector.status, 128)
        ));
    }
    append_omitted(&mut lines, visible_connectors.len(), "connector");

    lines.push(format!("Gateway tools ({}):", tools.len()));
    for tool in tools.iter().take(MAX_RENDERED_ITEMS) {
        let marker = if pinned.contains(tool.name.as_str()) {
            " [pinned]"
        } else {
            ""
        };
        lines.push(format!(
            "  {}{marker}",
            display_value(&tool.name, MAX_DISPLAY_BYTES)
        ));
    }
    append_omitted(&mut lines, tools.len(), "gateway tool");
    lines.push(
        "Pin a tool with `goshcoder aperture pin <toolName>`; enable connectors with `goshcoder aperture settings connectors.enabled enabled`."
            .to_owned(),
    );
    lines.join("\n")
}

fn list_connector_tools(gateway: &str) -> CliResult<Vec<aperture::GatewayTool>> {
    let timeouts = aperture_mcp::McpTimeouts {
        initialization: CONNECTOR_INITIALIZATION_TIMEOUT,
        call: CONNECTOR_CALL_TIMEOUT,
        initialized_notification: CONNECTOR_NOTIFICATION_TIMEOUT,
    };
    let session = aperture_mcp::McpClient::new(gateway)
        .map_err(|error| mcp_command_error("connector session failed", &error))?
        .with_timeouts(timeouts)
        .initialize()
        .map_err(|error| mcp_command_error("connector session failed", &error))?;
    session
        .list_tools()
        .map_err(|error| mcp_command_error("connector tools/list failed", &error))
}

fn pin_command(tool_name: &str) -> CliResult<String> {
    validate_tool_name(tool_name)?;
    let configuration_path = config::aperture_path();
    let (mut configuration, exists) = load_configuration_at(&configuration_path)?;
    let resolved = configuration.resolve();
    if !exists || resolved.base_url.trim().is_empty() {
        return Err(unconfigured_error());
    }
    if resolved
        .pinned_tools
        .iter()
        .any(|pin| pin.tool_name == tool_name)
    {
        return Ok(format!(
            "{} is already pinned.",
            display_value(tool_name, MAX_DISPLAY_BYTES)
        ));
    }

    let gateway = gateway_from_resolved(&resolved)?;
    let tools = list_connector_tools(&gateway)?;
    let change = match aperture::pin_connector_tool(&mut configuration, tool_name, &tools) {
        Ok(change) => change,
        Err(aperture::ApertureError::ToolNotFound(_)) => {
            return Err(command_error(format!(
                "tool {:?} not found on the gateway; `goshcoder aperture connectors` lists the available tools",
                display_value(tool_name, MAX_DISPLAY_BYTES)
            )));
        }
        Err(error) => {
            return Err(command_error(format!(
                "pin connector tool: {}",
                display_value(&error.to_string(), MAX_DISPLAY_BYTES)
            )));
        }
    };
    aperture::save_config(&configuration_path, &configuration)?;

    let mut message = format!(
        "Pinned {} ({} pinned). Pin changes take effect on the next session.",
        display_value(tool_name, MAX_DISPLAY_BYTES),
        change.pin_count
    );
    if change.context_cost_warning {
        message.push_str(
            " Each pinned tool adds its full schema to the system prompt; prefer pinning only the few tools you use every session.",
        );
    }
    Ok(message)
}

fn unpin_command(tool_name: &str) -> CliResult<String> {
    validate_tool_name(tool_name)?;
    let configuration_path = config::aperture_path();
    let (mut configuration, exists) = load_configuration_at(&configuration_path)?;
    if !exists {
        return Err(unconfigured_error());
    }
    let change = match aperture::unpin_connector_tool(&mut configuration, tool_name) {
        Ok(change) => change,
        Err(aperture::ApertureError::ToolNotPinned(_)) => {
            return Err(command_error(format!(
                "{} is not pinned",
                display_value(tool_name, MAX_DISPLAY_BYTES)
            )));
        }
        Err(error) => {
            return Err(command_error(format!(
                "unpin connector tool: {}",
                display_value(&error.to_string(), MAX_DISPLAY_BYTES)
            )));
        }
    };
    aperture::save_config(&configuration_path, &configuration)?;
    Ok(format!(
        "Unpinned {} ({} pinned). Pin changes take effect on the next session.",
        display_value(tool_name, MAX_DISPLAY_BYTES),
        change.pin_count
    ))
}

fn validate_tool_name(name: &str) -> CliResult<()> {
    if name.trim().is_empty() {
        return Err(command_error("MCP tool name cannot be empty"));
    }
    if name != name.trim() {
        return Err(command_error(
            "MCP tool name must not have leading or trailing whitespace",
        ));
    }
    if name.len() > aperture_mcp::MAX_TOOL_NAME_BYTES {
        return Err(command_error(format!(
            "MCP tool name exceeds {} bytes",
            aperture_mcp::MAX_TOOL_NAME_BYTES
        )));
    }
    if name.chars().any(char::is_control) {
        return Err(command_error(
            "MCP tool name must not contain control characters",
        ));
    }
    Ok(())
}

fn fetch_gateway_providers(gateway: &str) -> CliResult<Vec<aperture::GatewayProvider>> {
    let client = aperture::GatewayClient::new(gateway)
        .map_err(|error| gateway_command_error("create Aperture gateway client", &error))?;
    client
        .providers()
        .map_err(|error| gateway_command_error("fetch Aperture providers", &error))
}

fn gateway_from_resolved(resolved: &aperture::Resolved) -> CliResult<String> {
    let gateway = aperture::gateway_url(&resolved.base_url);
    if gateway.is_empty() {
        return Err(command_error(
            "Aperture gateway URL is not configured; run `goshcoder aperture onboarding`",
        ));
    }
    validate_gateway_url(&gateway)?;
    Ok(gateway)
}

fn normalize_gateway_input(value: &str) -> CliResult<String> {
    let gateway = aperture::normalize_input_url(value);
    if gateway.is_empty() {
        return Err(command_error("a gateway URL is required"));
    }
    validate_gateway_url(&gateway)?;
    Ok(gateway)
}

fn validate_gateway_url(value: &str) -> CliResult<()> {
    let Ok(parsed) = Url::parse(value) else {
        return Err(command_error(
            "Aperture gateway URL must be a valid http(s) origin",
        ));
    };
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || !matches!(parsed.path(), "" | "/")
    {
        return Err(command_error(
            "Aperture gateway URL must be an http(s) origin without credentials or a path",
        ));
    }
    Ok(())
}

fn gateway_command_error(context: &str, error: &aperture::ApertureError) -> Box<dyn Error> {
    command_error(format!("{context}: {}", gateway_error_message(error)))
}

fn gateway_error_message(error: &aperture::ApertureError) -> String {
    match error {
        aperture::ApertureError::Http(error) => error.to_string(),
        aperture::ApertureError::GatewayResponseTooLarge
        | aperture::ApertureError::InvalidGatewayResponse(_) => {
            display_value(&error.to_string(), MAX_DISPLAY_BYTES)
        }
        aperture::ApertureError::Request(_) | aperture::ApertureError::HttpClient(_) => {
            "could not reach the Aperture gateway; check the URL and tailnet connectivity"
                .to_owned()
        }
        _ => display_value(&error.to_string(), MAX_DISPLAY_BYTES),
    }
}

fn mcp_command_error(context: &str, error: &aperture_mcp::McpError) -> Box<dyn Error> {
    command_error(format!("{context}: {}", mcp_error_message(error)))
}

fn mcp_error_message(error: &aperture_mcp::McpError) -> String {
    match error {
        aperture_mcp::McpError::Request(_)
        | aperture_mcp::McpError::HttpClient(_)
        | aperture_mcp::McpError::ResponseRead(_) => {
            "could not reach the Aperture gateway; check the URL and tailnet connectivity"
                .to_owned()
        }
        aperture_mcp::McpError::InvalidInput(message)
        | aperture_mcp::McpError::Protocol(message) => display_value(message, MAX_DISPLAY_BYTES),
        aperture_mcp::McpError::Rpc { code, message } => format!(
            "MCP error: {} (code {code})",
            display_value(message, MAX_DISPLAY_BYTES)
        ),
        _ => display_value(&error.to_string(), MAX_DISPLAY_BYTES),
    }
}

fn unconfigured_error() -> Box<dyn Error> {
    command_error("Aperture is unconfigured; run `goshcoder aperture onboarding`")
}

fn enabled_word(enabled: bool) -> &'static str {
    if enabled { "enabled" } else { "disabled" }
}

fn or_unset(value: &str) -> String {
    if value.is_empty() {
        "(not set)".to_owned()
    } else {
        display_value(value, MAX_DISPLAY_BYTES)
    }
}

fn display_name(name: &str, fallback: &str) -> String {
    if name.is_empty() {
        display_value(fallback, MAX_DISPLAY_BYTES)
    } else {
        display_value(name, MAX_DISPLAY_BYTES)
    }
}

fn display_join(values: impl IntoIterator<Item = String>, noun: &str) -> String {
    let mut values = values.into_iter();
    let mut shown = Vec::new();
    for _ in 0..MAX_RENDERED_ITEMS {
        let Some(value) = values.next() else {
            return if shown.is_empty() {
                "none".to_owned()
            } else {
                shown.join(", ")
            };
        };
        shown.push(display_value(&value, 128));
    }
    let omitted = values.count();
    if omitted > 0 {
        shown.push(format!("… {omitted} more {noun}(s)"));
    }
    shown.join(", ")
}

fn append_omitted(lines: &mut Vec<String>, count: usize, noun: &str) {
    if count > MAX_RENDERED_ITEMS {
        lines.push(format!(
            "  … {} additional {noun}(s) omitted.",
            count - MAX_RENDERED_ITEMS
        ));
    }
}

fn display_value(value: &str, limit: usize) -> String {
    let mut output = String::new();
    for character in value.chars() {
        let character = if character.is_control() {
            ' '
        } else {
            character
        };
        if output.len() + character.len_utf8() > limit {
            if output.len() + '…'.len_utf8() <= limit {
                output.push('…');
            }
            break;
        }
        output.push(character);
    }
    output
}

fn write_output(output: &str) -> CliResult<()> {
    let output = bounded_output(output, MAX_OUTPUT_BYTES);
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    stdout.write_all(output.as_bytes())?;
    if !output.ends_with('\n') {
        stdout.write_all(b"\n")?;
    }
    stdout.flush()?;
    Ok(())
}

fn bounded_output(value: &str, limit: usize) -> String {
    const TRUNCATED: &str = "\n… output truncated.";
    let budget = limit.saturating_sub(TRUNCATED.len());
    let mut output = String::new();
    for character in value.chars() {
        let character = match character {
            '\n' => '\n',
            character if character.is_control() => ' ',
            character => character,
        };
        if output.len() + character.len_utf8() > budget {
            append_within_limit(&mut output, TRUNCATED, limit);
            return output;
        }
        output.push(character);
    }
    output
}

fn append_within_limit(output: &mut String, suffix: &str, limit: usize) {
    for character in suffix.chars() {
        if output.len() + character.len_utf8() > limit {
            break;
        }
        output.push(character);
    }
}

fn command_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::{BufRead, BufReader, Write},
        net::TcpListener,
        path::PathBuf,
        process,
        sync::atomic::{AtomicU64, Ordering},
        thread,
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "goshcoder-aperture-cli-{label}-{}-{sequence}",
            process::id()
        ));
        fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    #[test]
    fn command_aliases_preserve_legacy_and_cli_forms() {
        let empty = arguments(&[]);
        assert_eq!(
            parse_invocation(&empty).expect("empty command").0,
            Subcommand::Status
        );

        let setup = arguments(&["setup"]);
        assert_eq!(
            parse_invocation(&setup).expect("setup alias").0,
            Subcommand::Onboarding
        );
        let colon = arguments(&["/aperture:settings"]);
        assert_eq!(
            parse_invocation(&colon).expect("colon alias").0,
            Subcommand::Settings
        );
        let full = arguments(&["/aperture", "refresh"]);
        assert_eq!(
            parse_invocation(&full).expect("full slash command").0,
            Subcommand::Sync
        );
        let unknown = arguments(&["surprise"]);
        assert!(
            parse_invocation(&unknown)
                .expect_err("unknown command")
                .to_string()
                .contains("unknown Aperture command")
        );
    }

    #[test]
    fn onboarding_is_rejected_before_it_can_consume_noninteractive_input() {
        let setup = arguments(&["setup"]);
        let error = execute(&setup, false).expect_err("noninteractive setup must fail");
        assert!(error.to_string().contains("interactive terminal"));
    }

    #[test]
    fn settings_update_all_capability_and_provider_controls() {
        let mut configuration = aperture::Config::default();
        for (key, value) in [
            ("baseUrl", "ai.host.ts.net/v1"),
            ("proxy.enabled", "enabled"),
            ("dedicated.enabled", "disabled"),
            ("connectors.enabled", "enabled"),
            ("connectors.discoveryTools", "disabled"),
            ("onboardingDone", "completed"),
            ("proxy.provider.anthropic.enabled", "enabled"),
            ("proxy.provider.anthropic.gatewayModelsOnly", "on"),
            ("proxy.provider.anthropic.api", "anthropic-messages"),
            ("dedicated.provider.google.enabled", "disabled"),
            ("dedicated.provider.google.api", "google-generative-ai"),
        ] {
            apply_setting(&mut configuration, key, value).expect("apply setting");
        }

        let resolved = configuration.resolve();
        assert_eq!(resolved.base_url, "http://ai.host.ts.net");
        assert!(resolved.proxy_enabled);
        assert!(!resolved.dedicated_enabled);
        assert!(resolved.connectors_enabled);
        assert!(!resolved.discovery_tools);
        assert!(resolved.onboarding_done);
        assert!(!resolved.onboarding_enabled);
        assert_eq!(resolved.upstream_providers.len(), 1);
        assert!(resolved.upstream_providers[0].is_enabled());
        assert!(resolved.upstream_providers[0].keep_gateway_models_only);
        assert_eq!(resolved.upstream_providers[0].api, "anthropic-messages");
        assert_eq!(resolved.dedicated_providers.len(), 1);
        assert!(!resolved.dedicated_providers[0].enabled);
        assert_eq!(resolved.dedicated_providers[0].api, "google-generative-ai");

        apply_setting(&mut configuration, "proxy.provider.anthropic.api", "auto")
            .expect("auto clears API override");
        assert!(configuration.resolve().upstream_providers[0].api.is_empty());
        assert!(apply_setting(&mut configuration, "unknown", "value").is_err());
        assert!(
            apply_setting(
                &mut configuration,
                "proxy.provider.anthropic.api",
                "not-an-api"
            )
            .is_err()
        );
    }

    #[test]
    fn provider_setting_key_uses_the_last_dot_for_hyphenated_ids() {
        assert_eq!(
            provider_setting_key("proxy.provider.qwen-token-plan.api", "proxy.provider."),
            Some(("qwen-token-plan", "api"))
        );
        assert_eq!(
            provider_setting_key("proxy.provider.x", "proxy.provider."),
            None
        );
        assert_eq!(provider_setting_key("other.key", "proxy.provider."), None);
    }

    #[test]
    fn sync_builds_and_persists_a_bounded_gateway_snapshot() {
        let (gateway, server) = gateway_server();
        let directory = temporary_directory("sync");
        let cache_path = directory.join("extensions").join("aperture-cache.json");
        let configuration = aperture::Config {
            base_url: gateway.clone(),
            dedicated: Some(aperture::DedicatedConfig {
                enabled: Some(true),
                providers: Some(Vec::new()),
                cached_models: None,
            }),
            proxy: Some(aperture::ProxyConfig {
                enabled: Some(false),
                upstream_providers: Some(Vec::new()),
            }),
            ..aperture::Config::default()
        };

        let result = sync_configuration_with_metadata(&configuration, &cache_path, &[], None)
            .expect("synchronize gateway");
        assert_eq!(result.gateway.len(), 1);
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].id, "anthropic/claude");
        let cache = aperture::load_cache(&cache_path).expect("read saved cache");
        assert_eq!(cache.gateway.len(), 1);
        assert_eq!(
            cache
                .catalog_models(&aperture::build_catalog_key(
                    &gateway,
                    &configuration.resolve()
                ))
                .expect("matching catalog cache")
                .len(),
            1
        );
        server.join().expect("gateway server");
        fs::remove_dir_all(directory).expect("remove temporary directory");
    }

    #[test]
    fn rendered_output_is_utf8_safe_bounded_and_terminal_safe() {
        let rendered = bounded_output(&format!("{}\x1b[31m", "é".repeat(MAX_OUTPUT_BYTES)), 128);
        assert!(rendered.len() <= 128);
        assert!(rendered.contains("output truncated"));
        assert!(!rendered.contains('\x1b'));
        assert!(rendered.is_char_boundary(rendered.len()));
        assert_eq!(display_value("line\nnext", 32), "line next");
    }

    fn gateway_server() -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind gateway stub");
        let address = listener.local_addr().expect("gateway address");
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept request");
                let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                let mut request = String::new();
                reader.read_line(&mut request).expect("read request line");
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).expect("read request header");
                    if header == "\r\n" || header.is_empty() {
                        break;
                    }
                }
                let path = request.split_whitespace().nth(1).expect("request path");
                let body = match path {
                    "/api/providers" => {
                        r#"[{"id":"anthropic","name":"Anthropic","models":["claude"],"compatibility":{"anthropic_messages":true}}]"#
                    }
                    "/v1/models" => r#"{"data":[{"id":"claude"}]}"#,
                    _ => panic!("unexpected request path {path}"),
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .expect("write response");
                stream.flush().expect("flush response");
            }
        });
        (format!("http://{address}"), server)
    }
}
