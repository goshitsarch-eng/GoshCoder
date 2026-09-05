//! Shared command-line and session construction for `run` and `chat`.
//!
//! This module intentionally keeps terminal rendering out of the command
//! parser.  Both the pipeable runner and the Ratatui frontend therefore use
//! the same flag semantics, model-selection rules, local-resource snapshot,
//! workspace tools, and pi-compatible session lifecycle.

use std::{
    env, fmt, fs,
    path::{Path, PathBuf},
};

use crate::{
    agent,
    catalog::Catalog,
    config, llm,
    resources::{self, ResourcePaths, ResourceSet},
    session::{SessionOptions, SessionRuntime, SessionSelection},
    stream,
    tools::Workspace,
};

/// The command path that supplied a shared session configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvocationKind {
    Run,
    Chat,
}

/// Parsed command-line inputs shared by the pipeable and interactive runners.
///
/// The field names intentionally track the previous Go configuration, making
/// the policy visible without coupling it to a particular terminal frontend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionConfig {
    pub model_ref: String,
    pub system_prompt: String,
    pub thinking: String,
    pub workdir: PathBuf,
    pub enable_tools: bool,
    pub enable_ralph: bool,
    pub enable_planner: bool,
    pub load_planner: bool,
    pub claude_tui: bool,
    pub fullscreen: bool,
    pub quiet: bool,
    pub session_ref: Option<String>,
    pub continue_session: bool,
    pub resume: bool,
    pub no_session: bool,
    pub read_only: bool,
    pub session_name: Option<String>,
    pub sessions_dir: Option<PathBuf>,
    /// Distinguishes an explicit `-m` override from a model filled in from
    /// the remembered default. Resuming is allowed to restore the latter.
    pub model_from_flag: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model_ref: String::new(),
            system_prompt: String::new(),
            thinking: llm::THINKING_OFF.to_owned(),
            workdir: PathBuf::from("."),
            enable_tools: false,
            enable_ralph: false,
            enable_planner: false,
            load_planner: false,
            claude_tui: true,
            fullscreen: false,
            quiet: false,
            session_ref: None,
            continue_session: false,
            resume: false,
            no_session: false,
            read_only: false,
            session_name: None,
            sessions_dir: None,
            model_from_flag: false,
        }
    }
}

/// The parsed invocation before its model or terminal is initialized.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Invocation {
    pub kind: InvocationKind,
    pub config: SessionConfig,
    /// The joined one-shot prompt for `run`.
    pub prompt: Option<String>,
}

/// An initialized session plus the objects whose lifetime must outlast tools.
pub struct PreparedSession {
    pub runtime: SessionRuntime,
    pub workspace: Option<Workspace>,
    pub resource_paths: ResourcePaths,
    pub resources: ResourceSet,
    pub config: SessionConfig,
}

#[derive(Debug)]
pub enum RuntimeError {
    Usage(String),
    Catalog(String),
    Resource(String),
    Session(String),
    Io(std::io::Error),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage(message)
            | Self::Catalog(message)
            | Self::Resource(message)
            | Self::Session(message) => formatter.write_str(message),
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Usage(_) | Self::Catalog(_) | Self::Resource(_) | Self::Session(_) => None,
        }
    }
}

impl From<std::io::Error> for RuntimeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Parses `goshcoder run [flags] <prompt>`.
///
/// Like Go's `flag` package, parsing stops at the first non-flag argument.
/// That keeps prompt text that happens to begin with a dash intact.
pub fn parse_run(arguments: &[String]) -> Result<Invocation> {
    let mut config = SessionConfig::default();
    let positionals = parse_session_flags(arguments, &mut config)?;
    validate_session_config(&config)?;

    if config.resume {
        return Err(RuntimeError::Usage(
            "-resume needs an interactive session; use `goshcoder chat -resume`, or -session <id> with run"
                .to_owned(),
        ));
    }

    let prompt = positionals.join(" ").trim().to_owned();
    if prompt.is_empty() {
        return Err(RuntimeError::Usage("a prompt is required".to_owned()));
    }

    // One-shot commands are commonly used from scripts. Persist only when the
    // caller explicitly selected or named a session.
    if !config.continue_session && config.session_ref.is_none() && config.session_name.is_none() {
        config.no_session = true;
    }

    Ok(Invocation {
        kind: InvocationKind::Run,
        config,
        prompt: Some(prompt),
    })
}

/// Parses `goshcoder chat [flags]` and its implicit no-subcommand form.
pub fn parse_chat(arguments: &[String]) -> Result<Invocation> {
    let mut config = SessionConfig {
        enable_tools: true,
        enable_ralph: true,
        load_planner: true,
        fullscreen: true,
        ..SessionConfig::default()
    };
    let positionals = parse_session_flags(arguments, &mut config)?;
    validate_session_config(&config)?;
    if !positionals.is_empty() {
        return Err(RuntimeError::Usage(format!(
            "unexpected chat argument {}; use a quoted prompt with `goshcoder run`",
            positionals[0]
        )));
    }
    Ok(Invocation {
        kind: InvocationKind::Chat,
        config,
        prompt: None,
    })
}

/// Validates combinations that otherwise silently change persistence policy.
pub fn validate_session_config(config: &SessionConfig) -> Result<()> {
    if config.no_session {
        if config.continue_session {
            return Err(RuntimeError::Usage(
                "-no-session and -continue ask for opposite things; drop one".to_owned(),
            ));
        }
        if config.resume {
            return Err(RuntimeError::Usage(
                "-no-session and -resume ask for opposite things; drop one".to_owned(),
            ));
        }
        if config.session_ref.is_some() {
            return Err(RuntimeError::Usage(
                "-no-session and -session ask for opposite things; drop one".to_owned(),
            ));
        }
        if config.session_name.is_some() {
            return Err(RuntimeError::Usage(
                "-name has nothing to name when -no-session is given".to_owned(),
            ));
        }
    }
    if config.continue_session && config.session_ref.is_some() {
        return Err(RuntimeError::Usage(
            "-continue and -session both choose a session; use one".to_owned(),
        ));
    }
    if config.resume && config.session_ref.is_some() {
        return Err(RuntimeError::Usage(
            "-resume and -session both choose a session; use one".to_owned(),
        ));
    }
    if config.resume && config.continue_session {
        return Err(RuntimeError::Usage(
            "-resume and -continue both choose a session; use one".to_owned(),
        ));
    }
    if !thinking_level_names().contains(&config.thinking.as_str()) {
        return Err(RuntimeError::Usage(format!(
            "unknown -thinking level {:?}; use one of: {}",
            config.thinking,
            thinking_level_names().join(", ")
        )));
    }
    Ok(())
}

/// Converts persisted-session switches into the lifecycle module's selection.
///
/// `-resume` is intentionally excluded: it needs an interactive picker before
/// a session can be opened.
pub fn session_selection(config: &SessionConfig) -> Result<SessionSelection> {
    if config.resume {
        return Err(RuntimeError::Usage(
            "-resume requires an interactive session picker".to_owned(),
        ));
    }
    if config.no_session {
        return Ok(SessionSelection::NoSession);
    }
    if config.continue_session {
        return Ok(SessionSelection::Continue);
    }
    if let Some(reference) = &config.session_ref {
        return Ok(SessionSelection::Session(reference.clone()));
    }
    Ok(SessionSelection::New)
}

/// Resolves the remembered/default chat model using the same preference order
/// as the previous implementation.
pub fn default_chat_model_reference(
    catalog: &Catalog,
    environment_model: Option<&str>,
    remembered_model: &str,
) -> Result<String> {
    if let Some(model) = environment_model
        .map(str::trim)
        .filter(|model| !model.is_empty())
    {
        return Ok(model.to_owned());
    }

    let remembered = remembered_model.trim();
    if !remembered.is_empty() && catalog.resolve_model(remembered).is_ok() {
        return Ok(remembered.to_owned());
    }

    let configured = catalog
        .configured_provider_ids()
        .map_err(|error| RuntimeError::Catalog(error.to_string()))?;
    let preferred = [
        ("openai-codex", "gpt-5.6-sol"),
        ("anthropic", "claude-sonnet-5"),
        ("kimi-coding", "kimi-for-coding"),
        ("openai", "gpt-5.6-terra"),
    ];
    for (provider, model) in preferred {
        if configured.iter().any(|configured| configured == provider)
            && catalog.model(provider, model).is_some()
        {
            return Ok(format!("{provider}/{model}"));
        }
    }

    for provider_id in configured {
        if let Some(provider) = catalog.provider(&provider_id)
            && let Some(model) = provider.models().last()
        {
            return Ok(format!("{provider_id}/{}", model.id));
        }
    }

    Err(RuntimeError::Catalog(
        "no authenticated model is available; run `goshcoder auth login openai-codex` or pass -m provider/model"
            .to_owned(),
    ))
}

/// Reads process-level default-model sources. It is split from
/// [`default_chat_model_reference`] so tests and embedded frontends can supply
/// explicit values without modifying global environment state.
pub fn process_default_chat_model_reference(catalog: &Catalog) -> Result<String> {
    let environment_model = env::var("GOSHCODER_MODEL").ok();
    default_chat_model_reference(
        catalog,
        environment_model.as_deref(),
        &config::read_default_model(),
    )
}

/// Returns independently owned catalog model records for session restoration.
pub fn available_models(catalog: &Catalog) -> Vec<llm::Model> {
    catalog
        .providers()
        .into_iter()
        .flat_map(|provider| provider.models())
        .collect()
}

/// Resolves a model reference and turns it into an agent session option set.
///
/// Authentication is deliberately owned by the supplied responder: this
/// function never stores or logs the secret-bearing catalog `Auth`.
pub fn session_options(
    catalog: &Catalog,
    config: &SessionConfig,
    cwd: PathBuf,
    system_prompt: String,
    tools: Vec<agent::Tool>,
    responder: Option<agent::AssistantResponder>,
) -> Result<SessionOptions> {
    let model_ref = if config.model_ref.trim().is_empty() {
        process_default_chat_model_reference(catalog)?
    } else {
        config.model_ref.clone()
    };
    let resolved = catalog
        .resolve_model(&model_ref)
        .map_err(|error| RuntimeError::Catalog(error.to_string()))?;
    let (model, _) = resolved.into_parts();
    let thinking_level = stream::clamp_thinking_level(&model, &config.thinking);

    Ok(SessionOptions {
        cwd,
        sessions_dir: config.sessions_dir.clone(),
        selection: session_selection(config)?,
        read_only: config.read_only,
        name: config.session_name.clone(),
        system_prompt,
        model,
        model_is_explicit: config.model_from_flag,
        available_models: available_models(catalog),
        model_resolver: None,
        thinking_level,
        tools,
        responder,
        steering_mode: agent::QueueMode::OneAtATime,
        follow_up_mode: agent::QueueMode::OneAtATime,
        tool_execution: agent::ToolExecutionMode::Parallel,
        on_notice: None,
    })
}

/// Builds a session with a resource-derived prompt and confined workspace
/// tools. Extra tools (Ralph, planner, web search, MCP proxies) are supplied by
/// their integration modules and stay live for the returned session's lifetime.
pub fn prepare_session(
    catalog: &Catalog,
    config: SessionConfig,
    responder: Option<agent::AssistantResponder>,
    extra_tools: Vec<agent::Tool>,
) -> Result<PreparedSession> {
    let cwd = absolute_workdir(&config.workdir)?;
    let resource_paths = ResourcePaths::new(&cwd, config::agent_dir())
        .map_err(|error| RuntimeError::Resource(error.to_string()))?;
    let resources = resources::discover(&resource_paths)
        .map_err(|error| RuntimeError::Resource(error.to_string()))?;

    let needs_workspace = config.enable_tools || config.enable_planner || config.load_planner;
    let workspace = needs_workspace
        .then(|| Workspace::new(&cwd))
        .transpose()
        .map_err(|error| RuntimeError::Resource(error.to_string()))?;

    let mut tools = workspace
        .as_ref()
        .filter(|_| config.enable_tools)
        .map(Workspace::all)
        .unwrap_or_default();
    tools.extend(extra_tools);
    let tool_names = tools.iter().map(|tool| tool.name.as_str());
    let system_prompt = resources.build_system_prompt(&config.system_prompt, &cwd, tool_names);
    let options = session_options(catalog, &config, cwd, system_prompt, tools, responder)?;
    let runtime =
        SessionRuntime::open(options).map_err(|error| RuntimeError::Session(error.to_string()))?;

    Ok(PreparedSession {
        runtime,
        workspace,
        resource_paths,
        resources,
        config,
    })
}

/// Applies a model change after resolving the exact catalog reference.
///
/// The caller normally persists this through `SessionRuntime`'s agent event
/// subscription, so direct state changes still round-trip in session logs.
pub fn set_model(
    runtime: &SessionRuntime,
    catalog: &Catalog,
    reference: &str,
) -> Result<llm::Model> {
    let resolved = catalog
        .resolve_model(reference)
        .map_err(|error| RuntimeError::Catalog(error.to_string()))?;
    let (model, _) = resolved.into_parts();
    runtime.agent().set_model(model.clone());
    let _ = config::write_default_model(&format!("{}/{}", model.provider, model.id));
    Ok(model)
}

/// Resolves a resource invocation before it reaches the agent.
pub fn expand_resource_input(resources: &ResourceSet, input: &str) -> Result<Option<String>> {
    match resources
        .expand_input(input)
        .map_err(|error| RuntimeError::Resource(error.to_string()))?
    {
        resources::Expansion::NotResource => Ok(None),
        resources::Expansion::Template { text, .. } | resources::Expansion::Skill { text, .. } => {
            Ok(Some(text))
        }
    }
}

/// Builds the short status text shown after opening a session.
pub fn session_banner(runtime: &SessionRuntime) -> Option<String> {
    runtime.handle().map(|handle| {
        let action = if runtime.resumed() {
            "resumed"
        } else {
            "recording"
        };
        format!("{action} session {}", short_id(&handle.id))
    })
}

/// Drains non-fatal lifecycle notices into display strings.
pub fn drain_session_notices(runtime: &SessionRuntime) -> Vec<String> {
    runtime
        .drain_notices()
        .into_iter()
        .map(|notice| format!("{}: {}", notice.kind, notice.text))
        .collect()
}

fn parse_session_flags(arguments: &[String], config: &mut SessionConfig) -> Result<Vec<String>> {
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "--" {
            return Ok(arguments[index + 1..].to_vec());
        }
        if argument == "-" || !argument.starts_with('-') {
            return Ok(arguments[index..].to_vec());
        }

        let raw = argument.trim_start_matches('-');
        if raw.is_empty() {
            return Ok(arguments[index..].to_vec());
        }
        let (name, inline_value) = raw
            .split_once('=')
            .map_or((raw, None), |(name, value)| (name, Some(value)));

        let mut next_value = || {
            if let Some(value) = inline_value {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .cloned()
                .ok_or_else(|| RuntimeError::Usage(format!("-{name} needs a value")))
        };

        match name {
            "m" | "model" => {
                config.model_ref = next_value()?;
                config.model_from_flag = true;
            }
            "s" | "system" => config.system_prompt = next_value()?,
            "thinking" => config.thinking = next_value()?,
            "C" => config.workdir = PathBuf::from(next_value()?),
            "session" => config.session_ref = Some(next_value()?),
            "name" => config.session_name = Some(next_value()?),
            "sessions-dir" => config.sessions_dir = Some(PathBuf::from(next_value()?)),
            "tools" => config.enable_tools = bool_value(name, inline_value)?,
            "ralph" => config.enable_ralph = bool_value(name, inline_value)?,
            "planner" | "plan" => config.enable_planner = bool_value(name, inline_value)?,
            "claude-tui" => config.claude_tui = bool_value(name, inline_value)?,
            "fullscreen" => config.fullscreen = bool_value(name, inline_value)?,
            "continue" => config.continue_session = bool_value(name, inline_value)?,
            "resume" => config.resume = bool_value(name, inline_value)?,
            "no-session" => config.no_session = bool_value(name, inline_value)?,
            "read-only" => config.read_only = bool_value(name, inline_value)?,
            "quiet" => config.quiet = bool_value(name, inline_value)?,
            unknown => {
                return Err(RuntimeError::Usage(format!("unknown flag -{unknown}")));
            }
        }
        index += 1;
    }
    Ok(Vec::new())
}

fn bool_value(name: &str, inline: Option<&str>) -> Result<bool> {
    let Some(value) = inline else {
        return Ok(true);
    };
    match value {
        "1" | "t" | "T" | "TRUE" | "true" | "True" => Ok(true),
        "0" | "f" | "F" | "FALSE" | "false" | "False" => Ok(false),
        _ => Err(RuntimeError::Usage(format!(
            "invalid boolean value {value:?} for -{name}"
        ))),
    }
}

fn thinking_level_names() -> [&'static str; 7] {
    [
        llm::THINKING_OFF,
        llm::THINKING_MINIMAL,
        llm::THINKING_LOW,
        llm::THINKING_MEDIUM,
        llm::THINKING_HIGH,
        llm::THINKING_XHIGH,
        llm::THINKING_MAX,
    ]
}

fn absolute_workdir(workdir: &Path) -> Result<PathBuf> {
    if workdir.is_absolute() {
        return Ok(workdir.to_path_buf());
    }
    Ok(env::current_dir()?.join(workdir))
}

fn short_id(id: &str) -> &str {
    let mut end = id.len().min(8);
    while end > 0 && !id.is_char_boundary(end) {
        end -= 1;
    }
    &id[..end]
}

/// Removes the old standalone-model file only when it is blank. This helper
/// makes startup cleanup safe to call without ever destroying a remembered
/// model selected by another process.
pub fn remove_empty_default_model_file() -> Result<bool> {
    let path = config::default_model_path();
    let Ok(contents) = fs::read(&path) else {
        return Ok(false);
    };
    if !contents.iter().all(u8::is_ascii_whitespace) {
        return Ok(false);
    }
    fs::remove_file(path)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn run_parses_shared_flags_and_keeps_positional_prompt() {
        let invocation = parse_run(&arguments(&[
            "-m",
            "openai/gpt-5.6-terra",
            "-s",
            "be brief",
            "-thinking",
            "high",
            "-tools",
            "-ralph",
            "-planner",
            "-claude-tui",
            "-C",
            "/tmp/work",
            "trailing",
            "prompt",
        ]))
        .expect("parse");

        assert_eq!(invocation.kind, InvocationKind::Run);
        assert_eq!(invocation.config.model_ref, "openai/gpt-5.6-terra");
        assert!(invocation.config.model_from_flag);
        assert_eq!(invocation.config.system_prompt, "be brief");
        assert_eq!(invocation.config.thinking, "high");
        assert!(invocation.config.enable_tools);
        assert!(invocation.config.enable_ralph);
        assert!(invocation.config.enable_planner);
        assert!(invocation.config.claude_tui);
        assert_eq!(invocation.config.workdir, PathBuf::from("/tmp/work"));
        assert_eq!(invocation.prompt.as_deref(), Some("trailing prompt"));
        assert!(invocation.config.no_session);
    }

    #[test]
    fn chat_defaults_enable_tools_ralph_and_fullscreen_with_opt_out() {
        let invocation = parse_chat(&arguments(&[
            "-tools=false",
            "-ralph=false",
            "-fullscreen=false",
        ]))
        .expect("parse");

        assert!(!invocation.config.enable_tools);
        assert!(!invocation.config.enable_ralph);
        assert!(!invocation.config.fullscreen);
        assert!(invocation.config.load_planner);
        assert!(invocation.config.claude_tui);
    }

    #[test]
    fn run_does_not_disable_requested_session_recording() {
        let invocation =
            parse_run(&arguments(&["-continue", "-name", "nightly", "build this"])).expect("parse");

        assert!(invocation.config.continue_session);
        assert_eq!(invocation.config.session_name.as_deref(), Some("nightly"));
        assert!(!invocation.config.no_session);
        assert_eq!(
            session_selection(&invocation.config).expect("selection"),
            SessionSelection::Continue
        );
    }

    #[test]
    fn contradictions_and_unknown_thinking_levels_are_rejected() {
        for arguments in [
            arguments(&["-no-session", "-continue", "prompt"]),
            arguments(&["-no-session", "-session", "abc", "prompt"]),
            arguments(&["-continue", "-session", "abc", "prompt"]),
            arguments(&["-resume", "-continue", "prompt"]),
            arguments(&["-thinking", "hihg", "prompt"]),
        ] {
            assert!(parse_run(&arguments).is_err(), "{arguments:?}");
        }
    }

    #[test]
    fn run_refuses_an_interactive_resume_picker() {
        let error = parse_run(&arguments(&["-resume", "prompt"])).expect_err("must fail");
        assert!(error.to_string().contains("interactive"));
    }

    #[test]
    fn double_dash_keeps_dash_prefixed_prompt_text() {
        let invocation = parse_run(&arguments(&["--", "--not-a-flag", "text"])).expect("parse");
        assert_eq!(invocation.prompt.as_deref(), Some("--not-a-flag text"));
    }

    #[test]
    fn bool_parser_matches_go_flag_spellings() {
        assert!(bool_value("tools", Some("TRUE")).expect("true"));
        assert!(!bool_value("tools", Some("0")).expect("false"));
        assert!(bool_value("tools", None).expect("implicit"));
        assert!(bool_value("tools", Some("perhaps")).is_err());
    }

    #[test]
    fn session_selection_favors_no_session_before_other_defaulting() {
        assert_eq!(
            session_selection(&SessionConfig {
                no_session: true,
                ..SessionConfig::default()
            })
            .expect("selection"),
            SessionSelection::NoSession
        );
        assert_eq!(
            session_selection(&SessionConfig {
                session_ref: Some("abc".to_owned()),
                ..SessionConfig::default()
            })
            .expect("selection"),
            SessionSelection::Session("abc".to_owned())
        );
    }

    #[test]
    fn short_ids_do_not_split_utf8() {
        assert_eq!(short_id("你好世界"), "你好");
        assert_eq!(short_id("abc"), "abc");
    }
}
