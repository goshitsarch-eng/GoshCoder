pub mod agent;
pub mod aperture;
pub mod aperture_cli;
pub mod aperture_mcp;
pub mod aperture_runtime;
pub mod bedrock;
pub mod btw;
pub mod btw_runtime;
pub mod catalog;
pub mod compaction;
pub mod computeruse;
pub mod config;
pub mod google_auth;
pub mod llm;
pub mod markdown;
pub mod mistral;
pub mod oauth;
pub mod omni_cli;
pub mod omni_prompt_tools;
pub mod omniroute;
pub mod planner_runtime;
pub mod plannotator;
pub mod prompts;
pub mod provider_cli;
pub mod providers;
pub mod ralph;
pub mod ralph_cli;
pub mod ralph_runtime;
pub mod resources;
pub mod runtime;
pub mod session;
pub mod session_picker;
pub mod sessionlog;
pub mod sessions;
mod state;
pub mod stream;
pub mod tools;
mod ui;
pub mod webaccess;

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    io::{self, BufRead, IsTerminal, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyEventKind, MouseEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::state::{Action, App, Message, MessageRole};

const USAGE: &str = r#"GoshCoder - a Rust coding agent

Usage:
  goshcoder                         Start fullscreen interactive chat
  goshcoder [chat flags]            Start chat without typing the subcommand
  goshcoder run [flags] <prompt>    Run a single prompt
  goshcoder chat [flags]            Interactive session (slash commands, /help)
  goshcoder providers                List providers and credential status
  goshcoder models [provider]        List available models
  goshcoder auth <subcommand>        Manage credentials
  goshcoder omni <subcommand>        Manage an OmniRoute gateway
  goshcoder aperture <subcommand>    Manage Tailscale Aperture
  goshcoder ralph <subcommand>       Manage Ralph loops
  goshcoder sessions [--sessions-dir <dir>] [subcommand]
                                     List, inspect, export, import, or remove sessions
  goshcoder prompts <subcommand>     Manage prompt templates
  goshcoder version                  Print the version

The Ratatui frontend, persistent-session, prompt, planner, Ralph, provider,
model, credential, and context-compaction foundations are active. `run`
supports `openai-completions`, `openai-responses`, `azure-openai-responses`,
`openai-codex-responses`, `anthropic-messages`, `google-generative-ai`, and
`google-vertex`, `mistral-conversations`, `omni-prompt-tools`, and
`bedrock-converse-stream`;
the remaining provider extensions and interactive commands are still being
migrated from the previous implementation.
"#;

const SIDEBAR_GIT_TIMEOUT: Duration = Duration::from_secs(2);
const SIDEBAR_GIT_REFRESH_INTERVAL: Duration = Duration::from_secs(1);
const SIDEBAR_GIT_MAX_OUTPUT_BYTES: usize = 1 << 20;
const SIDEBAR_GIT_MAX_CHANGES: usize = 50;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version" | "-v" | "version") => {
            print_version();
            Ok(())
        }
        Some("--help" | "-h" | "help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("run") => run_command(&args[1..]),
        Some("providers") => provider_cli::providers_command(),
        Some("models") => provider_cli::models_command(&args[1..]),
        Some("auth") => provider_cli::auth_command(&args[1..]),
        Some("omni") => omni_cli::command(&args[1..]),
        Some("aperture") => aperture_cli::command(&args[1..]),
        Some("sessions") => sessions::command(&args[1..]),
        Some("prompts") => prompts::command(&args[1..]),
        Some("ralph") => ralph_cli::command(&args[1..]),
        Some("chat") => run_interactive(&args[1..]),
        None => run_interactive(&[]),
        Some(argument) if argument.starts_with('-') => run_interactive(&args),
        Some(command) => Err(format!(
            "{command} is queued for runtime migration; use `goshcoder chat` to exercise the Ratatui frontend"
        )
        .into()),
    }
}

fn print_version() {
    println!("goshcoder {}", build_version());
}

/// Release automation supplies a VCS-derived version through `GOSHCODER_VERSION`;
/// ordinary Cargo builds retain the manifest version without requiring a build
/// script or a Git checkout.
fn build_version() -> &'static str {
    option_env!("GOSHCODER_VERSION")
        .filter(|version| !version.is_empty())
        .unwrap_or(env!("CARGO_PKG_VERSION"))
}

fn ui_version() -> &'static str {
    build_version()
        .strip_prefix('v')
        .unwrap_or_else(build_version)
}

/// Executes the pipeable one-shot command on the same durable session stack
/// used by future Ratatui chat sessions.
///
/// Assistant text remains on stdout while reasoning and tool activity use
/// stderr. This preserves the original command's scripting contract even
/// while the interactive frontend is still being connected to this runtime.
fn run_command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let invocation = runtime::parse_run(arguments)?;
    let prompt = invocation
        .prompt
        .ok_or_else(|| io::Error::other("run invocation did not include a prompt"))?;
    let quiet = invocation.config.quiet;
    let catalog = Arc::new(catalog::Catalog::with_default_credentials()?);
    let responder = providers::assistant_responder_from_catalog(
        Arc::clone(&catalog),
        providers::ProviderConfig::default(),
    )?;
    let prepared = runtime::prepare_session(
        catalog.as_ref(),
        invocation.config,
        Some(responder),
        Vec::new(),
    )?;

    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            eprintln!("{}", dim(&format!("session: {notice}"), color_enabled()));
        }
        if let Some(banner) = runtime::session_banner(&prepared.runtime) {
            if prepared.runtime.resumed() {
                let messages = prepared.runtime.restored().messages;
                let mut stderr = io::stderr().lock();
                render_restored_transcript(&messages, &banner, &mut stderr, color_enabled())?;
            } else {
                eprintln!("{}", dim(&banner, color_enabled()));
            }
        }
    }

    let render_lock = Arc::new(Mutex::new(()));
    let color = color_enabled();
    let agent = prepared.runtime.agent().clone();
    let _render_subscription = agent.subscribe({
        let render_lock = Arc::clone(&render_lock);
        move |event| {
            let _guard = render_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut stdout = io::stdout().lock();
            let mut stderr = io::stderr().lock();
            let _ = render_run_event(&event, &mut stdout, &mut stderr, color);
        }
    });

    prepared.sync_extensions()?;
    if let Some(outcome) = compaction::maybe_auto_compact(&agent)? {
        print_compaction_outcome(&outcome, true, color);
    }
    agent.prompt(prompt)?;
    prepared.runtime.sync()?;
    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            eprintln!("{}", dim(&format!("session: {notice}"), color));
        }
    }
    Ok(())
}

/// Renders an agent lifecycle event using the original command's stdout/stderr
/// separation. It is intentionally independent of terminal state so tests and
/// future line-mode chat can reuse it.
fn render_run_event<Out: Write, Err: Write>(
    event: &agent::Event,
    stdout: &mut Out,
    stderr: &mut Err,
    color: bool,
) -> io::Result<()> {
    match event.kind {
        agent::EventKind::MessageUpdate => {
            if let Some(update) = event.assistant_event.as_ref() {
                match update.event_type.as_str() {
                    stream::EVENT_TEXT_DELTA => write!(stdout, "{}", update.delta)?,
                    stream::EVENT_THINKING_DELTA => {
                        write!(stderr, "{}", dim(&update.delta, color))?;
                    }
                    stream::EVENT_THINKING_END => writeln!(stderr)?,
                    _ => {}
                }
            }
        }
        agent::EventKind::MessageEnd => {
            if let Some(llm::Message::Assistant(message)) = event.message.as_ref() {
                if !event.assistant_was_streamed {
                    for content in &message.content {
                        match content {
                            llm::ContentBlock::Text(text) => write!(stdout, "{}", text.text)?,
                            llm::ContentBlock::Thinking(thinking) => {
                                write!(stderr, "{}", dim(&thinking.thinking, color))?;
                            }
                            llm::ContentBlock::Image(_) | llm::ContentBlock::ToolCall(_) => {}
                        }
                    }
                }
                if !message.error_message.is_empty() {
                    writeln!(stderr, "{} {}", dim("error:", color), message.error_message)?;
                }
            }
        }
        agent::EventKind::ToolExecutionStart => {
            let arguments = summarize_tool_arguments(&event.arguments);
            let activity = if arguments.is_empty() {
                event.tool_name.clone()
            } else {
                format!("{} {arguments}", event.tool_name)
            };
            writeln!(stderr, "\n{} {}", dim("→", color), bold(&activity, color))?;
        }
        agent::EventKind::ToolExecutionEnd => {
            let status = if event.is_error { "✗" } else { "✓" };
            writeln!(
                stderr,
                "{} {}",
                dim(status, color),
                dim(&first_line(&tool_result_text(event.result.as_ref())), color)
            )?;
        }
        agent::EventKind::TurnEnd => writeln!(stdout)?,
        agent::EventKind::AgentEnd => {
            if let Some(message) = last_assistant(&event.messages)
                && message.usage.total_tokens > 0
            {
                writeln!(
                    stderr,
                    "{}",
                    dim(
                        &format!(
                            "tokens: {} in / {} out  cost: ${:.4}",
                            message.usage.input, message.usage.output, message.usage.cost.total
                        ),
                        color
                    )
                )?;
            }
        }
        agent::EventKind::AgentStart
        | agent::EventKind::TurnStart
        | agent::EventKind::MessageStart
        | agent::EventKind::ToolExecutionUpdate
        | agent::EventKind::ModelChange
        | agent::EventKind::ThinkingLevelChange
        | agent::EventKind::ContextCompacted
        | agent::EventKind::TranscriptReset => {}
    }
    stdout.flush()?;
    stderr.flush()
}

fn last_assistant(messages: &[llm::Message]) -> Option<&llm::AssistantMessage> {
    messages.iter().rev().find_map(|message| match message {
        llm::Message::Assistant(message) => Some(message.as_ref()),
        llm::Message::User(_) | llm::Message::ToolResult(_) => None,
    })
}

fn restored_transcript_text(messages: &[llm::Message], header: &str) -> String {
    if messages.is_empty() {
        return String::new();
    }
    let mut lines = vec![header.to_owned()];
    for message in messages {
        if let Some(summary) = compaction::summary_text(message) {
            lines.push(format!("· compacted {}", first_line(&summary)));
            continue;
        }
        let text = first_line(&message.text_preview());
        if !text.trim().is_empty() {
            lines.push(format!("· {} {text}", message.role()));
        }
    }
    lines.push("─".repeat(40));
    lines.join("\n")
}

fn render_restored_transcript<Err: Write>(
    messages: &[llm::Message],
    header: &str,
    stderr: &mut Err,
    color: bool,
) -> io::Result<()> {
    let transcript = restored_transcript_text(messages, header);
    if !transcript.is_empty() {
        writeln!(stderr, "{}", dim(&transcript, color))?;
    }
    Ok(())
}

fn tool_result_text(result: Option<&agent::ToolResult>) -> String {
    result
        .map(|result| {
            result
                .content
                .iter()
                .filter_map(llm::ContentBlock::plain_text)
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn first_line(text: &str) -> String {
    let text = text.trim();
    let (first, elided) = text
        .split_once('\n')
        .map_or((text, false), |(first, _)| (first, true));
    let clipped: String = first.chars().take(120).collect();
    if clipped.len() < first.len() || elided {
        format!("{clipped} ...")
    } else {
        clipped
    }
}

fn summarize_tool_arguments(
    arguments: &std::collections::BTreeMap<String, serde_json::Value>,
) -> String {
    arguments
        .iter()
        .map(|(key, value)| {
            let value = value.to_string().replace('\n', " ");
            format!("{key}={}", clip_characters(&value, 60))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn clip_characters(text: &str, limit: usize) -> String {
    let mut characters = text.chars();
    let clipped: String = characters.by_ref().take(limit).collect::<String>();
    if characters.next().is_some() {
        format!("{clipped}...")
    } else {
        clipped
    }
}

fn color_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none() && io::stderr().is_terminal()
}

fn dim(text: &str, color: bool) -> String {
    if text.is_empty() || !color {
        text.to_owned()
    } else {
        format!("\x1b[2m{text}\x1b[0m")
    }
}

fn bold(text: &str, color: bool) -> String {
    if text.is_empty() || !color {
        text.to_owned()
    } else {
        format!("\x1b[1m{text}\x1b[0m")
    }
}

fn terminal_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|width| *width > 0)
        .unwrap_or(80)
}

fn native_accent(text: &str, color: bool) -> String {
    if color {
        format!("\x1b[38;2;215;119;87m{text}\x1b[39m")
    } else {
        text.to_owned()
    }
}

fn native_display_width(text: &str) -> usize {
    UnicodeWidthStr::width(text)
}

fn native_truncate(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    if native_display_width(text) <= width {
        return text.to_owned();
    }
    if width == 1 {
        return "…".to_owned();
    }
    let mut output = String::new();
    let mut used = 0usize;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if used + character_width > width - 1 {
            break;
        }
        output.push(character);
        used += character_width;
    }
    output.push('…');
    output
}

fn native_pad(text: &str, width: usize) -> String {
    let text = native_truncate(text, width);
    let padding = width.saturating_sub(native_display_width(&text));
    format!("{text}{}", " ".repeat(padding))
}

fn native_center(text: &str, width: usize) -> String {
    let text = native_truncate(text, width);
    format!(
        "{}{text}",
        " ".repeat(width.saturating_sub(native_display_width(&text)) / 2)
    )
}

fn native_format_cwd(cwd: &Path) -> String {
    let cwd = cwd.display().to_string();
    let Ok(home) = std::env::var("HOME") else {
        return cwd;
    };
    if cwd == home {
        return "~".to_owned();
    }
    let separator = std::path::MAIN_SEPARATOR.to_string();
    cwd.strip_prefix(&(home + &separator))
        .map_or(cwd.clone(), |suffix| format!("~{separator}{suffix}"))
}

fn native_line_info(prepared: &runtime::PreparedSession) -> NativeLineInfo {
    let state = prepared.runtime.agent().state();
    let git = scan_sidebar_git(&sidebar_workspace_path(prepared));
    let planner_state = prepared
        .planner
        .as_ref()
        .map(|planner| planner.manager().state());
    let (mode, _) = planner_sidebar_details(planner_state);
    NativeLineInfo {
        model: format!("{}/{}", state.model.provider, state.model.id),
        context_used: compaction::measured_context_tokens(&state.messages),
        context_limit: state.model.context_window,
        cost: compaction::conversation_cost(&state.messages, &state.compactions),
        messages: state.messages.len(),
        tools: state.tools.len(),
        changed_files: git.changed_files,
        branch: git.branch.unwrap_or_else(|| "not a git repo".to_owned()),
        mode,
        thinking: state.thinking_level,
    }
}

fn native_info_lines(info: &NativeLineInfo) -> Vec<String> {
    let mut context = compact_number(info.context_used);
    if info.context_limit > 0 {
        let percent = info
            .context_used
            .saturating_mul(100)
            .saturating_div(info.context_limit)
            .min(100);
        context.push_str(&format!(
            "/{} · {percent}%",
            compact_number(info.context_limit)
        ));
    }
    let mode = if info.thinking.is_empty() || info.thinking == llm::THINKING_OFF {
        info.mode.clone()
    } else {
        format!("{} · {}", info.mode, info.thinking)
    };
    vec![
        "Session".to_owned(),
        native_truncate(&info.model, 24),
        format!("Context  {context}"),
        format!("Cost     ${:.4}", info.cost),
        format!("Messages {} · Tools {}", info.messages, info.tools),
        format!("Files    {} changed", info.changed_files),
        format!("Branch   {}", info.branch),
        format!("Mode     {mode}"),
    ]
}

fn native_line_sidebar(info: &NativeLineInfo, width: usize, color: bool) -> String {
    let width = width.max(24);
    let inner = width.saturating_sub(2);
    let mut lines = vec![native_accent(&format!("╭{}╮", "─".repeat(inner)), color)];
    lines.extend(native_info_lines(info).into_iter().map(|line| {
        format!(
            "{} {} {}",
            native_accent("│", color),
            native_pad(&line, inner.saturating_sub(2)),
            native_accent("│", color),
        )
    }));
    lines.push(native_accent(&format!("╰{}╯", "─".repeat(inner)), color));
    lines.join("\n")
}

fn native_line_header(
    width: usize,
    app_version: &str,
    cwd: &Path,
    info: &NativeLineInfo,
    color: bool,
) -> String {
    if width < 24 {
        return format!("GoshCoder v{app_version}");
    }
    let inner = width.saturating_sub(2);
    let use_tips = inner >= 55;
    let right_width = if use_tips {
        (inner.saturating_mul(28) / 100).clamp(16, 28)
    } else {
        0
    };
    let left_width = if use_tips {
        inner.saturating_sub(right_width + 3)
    } else {
        inner
    };
    let logo = ["  ██████  ", " ██  ███  ", "  ███  ██ ", "  ██   ██ "];
    let mut left = logo
        .iter()
        .map(|line| native_center(line, left_width))
        .collect::<Vec<_>>();
    left.extend([
        native_center("Let's build something great", left_width),
        native_center(
            &format!("{} · {} effort", info.model, info.thinking),
            left_width,
        ),
        native_center(&native_format_cwd(cwd), left_width),
        String::new(),
    ]);
    let mut right = native_info_lines(info);
    while right.len() < left.len() {
        right.push(String::new());
    }
    let label = format!(" GoshCoder v{app_version} ");
    let fill = width
        .saturating_sub(2)
        .saturating_sub(native_display_width(&label))
        .saturating_sub(3);
    let mut lines = vec![native_accent(
        &format!("╭───{label}{}╮", "─".repeat(fill)),
        color,
    )];
    for (index, line) in left.iter().enumerate() {
        let content = if use_tips {
            format!(
                "{} {} {}",
                native_pad(line, left_width),
                "│",
                native_pad(&right[index], right_width),
            )
        } else {
            native_pad(line, left_width)
        };
        lines.push(format!(
            "{}{}{}",
            native_accent("│", color),
            native_pad(&content, inner),
            native_accent("│", color)
        ));
    }
    lines.push(native_accent(&format!("╰{}╯", "─".repeat(inner)), color));
    lines.join("\n")
}

fn native_line_input_prompt(width: usize, color: bool) -> String {
    if width < 8 {
        return "> ".to_owned();
    }
    format!(
        "{}\n{} ",
        native_accent(&format!("╭{}╮", "─".repeat(width.saturating_sub(2))), color),
        native_accent("╰─❯", color),
    )
}

fn run_interactive(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut invocation = runtime::parse_chat(arguments)?;
    choose_resume_session(&mut invocation.config)?;
    if !invocation.config.fullscreen || !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return run_line_interactive(invocation);
    }

    let quiet = invocation.config.quiet;
    let catalog = Arc::new(catalog::Catalog::with_default_credentials()?);
    ensure_interactive_model(&mut invocation.config, catalog.as_ref(), true)?;
    let responder = providers::assistant_responder_from_catalog(
        Arc::clone(&catalog),
        providers::ProviderConfig::default(),
    )?;
    let mut prepared = runtime::prepare_session(
        catalog.as_ref(),
        invocation.config,
        Some(responder),
        Vec::new(),
    )?;
    remember_default_model(&prepared);
    let agent = prepared.runtime.agent().clone();
    let (agent_event_sender, agent_event_receiver) = mpsc::sync_channel(64);
    let (turn_sender, turn_receiver) = mpsc::channel();
    let _agent_event_subscription = agent.subscribe(move |event| {
        let _ = agent_event_sender.try_send(event);
    });

    enable_raw_mode()?;
    let mut stderr = io::stderr();
    if let Err(error) = execute!(stderr, EnterAlternateScreen, EnableMouseCapture) {
        let _ = disable_raw_mode();
        return Err(error.into());
    }
    let backend = CrosstermBackend::new(stderr);
    let mut terminal = match Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(error) => {
            let mut cleanup = io::stderr();
            let _ = execute!(cleanup, LeaveAlternateScreen, DisableMouseCapture);
            let _ = disable_raw_mode();
            return Err(error.into());
        }
    };
    let result = event_loop(
        &mut terminal,
        &prepared,
        catalog.as_ref(),
        agent_event_receiver,
        turn_sender,
        turn_receiver,
        quiet,
    );

    let terminal_cleanup = (|| -> io::Result<()> {
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()
    })();
    let session_cleanup = prepared.runtime.close();
    terminal_cleanup?;
    session_cleanup?;
    result
}

/// Resolves `chat -resume` before the frontend creates a session runtime.
///
/// Session selection must happen before opening the log so a picked existing
/// session follows the ordinary durable-session lifecycle, including its
/// existing-model and read-only/busy handling.
fn choose_resume_session(config: &mut runtime::SessionConfig) -> Result<(), Box<dyn Error>> {
    if !config.resume {
        return Ok(());
    }
    let cwd = runtime::absolute_workdir(&config.workdir)?;
    let store = sessionlog::Store::new(
        config
            .sessions_dir
            .clone()
            .unwrap_or_else(config::sessions_dir),
    );
    let stdin = io::stdin();
    let stderr = io::stderr();
    let mut input = stdin.lock();
    let mut output = stderr.lock();
    let selected = session_picker::choose_session(&store, &cwd, &mut input, &mut output)?;
    config.resume = false;
    if let Some(selected) = selected {
        config.session_ref = Some(selected.id);
    }
    Ok(())
}

/// Runs the pipe-friendly chat fallback used when the alternate-screen
/// Ratatui frontend was explicitly disabled or cannot safely own the terminal.
///
/// It deliberately shares the live session, responder, slash-command
/// dispatcher, compaction, and event renderer with fullscreen chat. The only
/// difference is presentation: prompts and command notices are line-oriented.
fn run_line_interactive(mut invocation: runtime::Invocation) -> Result<(), Box<dyn Error>> {
    let quiet = invocation.config.quiet;
    let catalog = Arc::new(catalog::Catalog::with_default_credentials()?);
    let interactive_terminal = io::stdin().is_terminal() && io::stderr().is_terminal();
    ensure_interactive_model(
        &mut invocation.config,
        catalog.as_ref(),
        interactive_terminal,
    )?;
    let responder = providers::assistant_responder_from_catalog(
        Arc::clone(&catalog),
        providers::ProviderConfig::default(),
    )?;
    let mut prepared = runtime::prepare_session(
        catalog.as_ref(),
        invocation.config,
        Some(responder),
        Vec::new(),
    )?;
    remember_default_model(&prepared);
    let agent = prepared.runtime.agent().clone();
    let render_lock = Arc::new(Mutex::new(()));
    let color = color_enabled();
    let _render_subscription = agent.subscribe({
        let render_lock = Arc::clone(&render_lock);
        move |event| {
            let _guard = render_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut stdout = io::stdout().lock();
            let mut stderr = io::stderr().lock();
            let _ = render_run_event(&event, &mut stdout, &mut stderr, color);
        }
    });

    let result = line_interactive_loop(&prepared, catalog.as_ref(), quiet);
    let close_result = prepared.runtime.close();
    result?;
    close_result?;
    Ok(())
}

/// Makes the first interactive session self-contained: a terminal user with
/// no configured model can authenticate before session construction rather
/// than being sent back to an external `auth` command.
fn ensure_interactive_model(
    config: &mut runtime::SessionConfig,
    catalog: &catalog::Catalog,
    can_onboard: bool,
) -> Result<(), Box<dyn Error>> {
    if !config.model_ref.trim().is_empty() {
        return Ok(());
    }
    match runtime::process_default_chat_model_reference(catalog) {
        Ok(model) => {
            config.model_ref = model;
            Ok(())
        }
        Err(_) if can_onboard => {
            config.model_ref = onboard_chat_model(catalog)?;
            Ok(())
        }
        Err(error) => Err(Box::new(error)),
    }
}

fn onboard_chat_model(catalog: &catalog::Catalog) -> Result<String, Box<dyn Error>> {
    eprintln!("Welcome to GoshCoder. Choose a subscription login:");
    eprintln!("  1. OpenAI Codex (recommended)");
    eprintln!("  2. Anthropic");
    eprintln!("  3. Kimi Coding");
    eprint!("Choice [1]: ");
    io::stderr().flush()?;

    let mut choice = String::new();
    io::stdin().lock().read_line(&mut choice)?;
    let provider_id = onboarding_provider(&choice)?;
    let outcome = provider_cli::auth_for_provider(catalog, provider_id)?;
    eprintln!("{}", outcome.notice());
    runtime::process_default_chat_model_reference(catalog).map_err(|error| Box::new(error) as _)
}

fn onboarding_provider(choice: &str) -> Result<&'static str, Box<dyn Error>> {
    match choice.trim() {
        "" | "1" => Ok("openai-codex"),
        "2" => Ok("anthropic"),
        "3" => Ok("kimi-coding"),
        choice => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown login choice {choice:?}"),
        )
        .into()),
    }
}

fn line_interactive_loop(
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    quiet: bool,
) -> Result<(), Box<dyn Error>> {
    let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
    let color = color_enabled();
    let mut claude_tui = prepared.config.claude_tui;
    let mut last_native_sidebar = None;
    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            eprintln!("{}", dim(&format!("session: {notice}"), color));
        }
        if let Some(banner) = runtime::session_banner(&prepared.runtime) {
            if prepared.runtime.resumed() {
                let messages = prepared.runtime.restored().messages;
                let mut stderr = io::stderr().lock();
                render_restored_transcript(&messages, &banner, &mut stderr, color)?;
            } else {
                eprintln!("{}", dim(&banner, color));
            }
        }
        if let Some(hint) = session_continue_hint(&prepared.runtime) {
            eprintln!("{}", dim(&hint, color));
        }
        if interactive {
            if claude_tui {
                let info = native_line_info(prepared);
                let sidebar = native_line_sidebar(&info, terminal_width().min(42), color);
                eprintln!(
                    "{}",
                    native_line_header(
                        terminal_width(),
                        build_version(),
                        &sidebar_workspace_path(prepared),
                        &info,
                        color,
                    )
                );
                last_native_sidebar = Some(sidebar);
            } else {
                let state = prepared.runtime.agent().state();
                eprintln!(
                    "{}",
                    dim(
                        &format!(
                            "goshcoder {} · {}/{} · /help for commands",
                            build_version(),
                            state.model.provider,
                            state.model.id
                        ),
                        color
                    )
                );
            }
        }
    }

    let stdin = io::stdin();
    let mut raw = String::new();
    loop {
        raw.clear();
        if interactive {
            let mut stderr = io::stderr().lock();
            if claude_tui {
                let sidebar = native_line_sidebar(
                    &native_line_info(prepared),
                    terminal_width().min(42),
                    color,
                );
                if last_native_sidebar.as_deref() != Some(sidebar.as_str()) {
                    writeln!(stderr)?;
                    writeln!(stderr, "{sidebar}")?;
                    last_native_sidebar = Some(sidebar);
                }
                write!(
                    stderr,
                    "\n{}",
                    native_line_input_prompt(terminal_width().min(88), color)
                )?;
            } else {
                write!(stderr, "\n> ")?;
            }
            stderr.flush()?;
        }
        if stdin.lock().read_line(&mut raw)? == 0 {
            if interactive {
                eprintln!();
            }
            break;
        }

        let input = raw.trim();
        if input.is_empty() {
            continue;
        }
        let input = match prepared.expand_resource_input(input) {
            Ok(Some(expanded)) => expanded,
            Ok(None) => input.to_owned(),
            Err(error) => {
                eprintln!("error: {error}");
                continue;
            }
        };

        if input.starts_with('/') {
            let mut app = App::new();
            app.streaming = prepared.runtime.agent().state().is_streaming;
            let mut view = InteractiveView {
                line_claude_tui: Some(claude_tui),
                ..InteractiveView::default()
            };
            let (turn_sender, turn_receiver) = mpsc::channel();
            let prior_claude_tui = claude_tui;
            let outcome = dispatch_runtime_slash_command(
                &mut app,
                &mut view,
                prepared,
                catalog,
                turn_sender,
                &input,
                false,
            );
            claude_tui = view.line_claude_tui.unwrap_or(claude_tui);
            if claude_tui != prior_claude_tui {
                last_native_sidebar = None;
            }
            if view.turn_pending {
                match turn_receiver.recv() {
                    Ok(completion) => finish_interactive_task(&mut view, prepared, completion),
                    Err(_) => {
                        view.turn_pending = false;
                        append_view_message(
                            &mut view,
                            MessageRole::Error,
                            "interactive command worker stopped unexpectedly",
                        );
                    }
                }
            }
            render_line_view(&mut view);
            if matches!(outcome, CommandDispatch::Quit) {
                break;
            }
        } else {
            if let Err(error) = prepared.sync_extensions() {
                eprintln!("error: {error}");
                continue;
            }
            match compaction::maybe_auto_compact(prepared.runtime.agent()) {
                Ok(Some(outcome)) => print_compaction_outcome(&outcome, true, color_enabled()),
                Ok(None) => {}
                Err(error) => {
                    eprintln!("error: {error}");
                    continue;
                }
            }
            if let Err(error) = prepared.runtime.agent().prompt(input) {
                eprintln!("error: {error}");
            }
        }

        prepared.runtime.sync()?;
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            eprintln!("{}", dim(&format!("session: {notice}"), color_enabled()));
        }
    }
    Ok(())
}

/// Remembers the resolved chat model only after its session is successfully
/// constructed, mirroring the established chat startup behavior. A failed
/// configuration write must not prevent an otherwise usable conversation.
fn remember_default_model(prepared: &runtime::PreparedSession) {
    let model = prepared.runtime.agent().state().model;
    let _ = config::write_default_model(&model_reference(&model));
}

fn model_reference(model: &llm::Model) -> String {
    format!("{}/{}", model.provider, model.id)
}

fn session_continue_hint(runtime: &session::SessionRuntime) -> Option<String> {
    continue_session_hint(runtime.recording(), runtime.id().as_deref())
}

fn continue_session_hint(recording: bool, session_id: Option<&str>) -> Option<String> {
    recording.then(|| {
        session_id.map(|id| {
            format!(
                "session {} · resume with: goshcoder chat -continue",
                short_id(id)
            )
        })
    })?
}

fn render_line_view(view: &mut InteractiveView) {
    let notices = std::mem::take(&mut view.notices);
    let had_notices = !notices.is_empty();
    for message in notices {
        match message.role {
            MessageRole::Error => eprintln!("error: {}", message.text),
            MessageRole::Command | MessageRole::Notice => eprintln!("{}", message.text),
            MessageRole::User
            | MessageRole::Assistant
            | MessageRole::Thinking
            | MessageRole::Tool => {
                eprintln!("{}", message.text)
            }
        }
    }
    if !had_notices && view.activity != "Ready" {
        eprintln!("{}", view.activity);
    }
}

struct InteractiveView {
    notices: Vec<Message>,
    activity: String,
    recent_tool: String,
    activity_since: Option<Instant>,
    context_estimate: compaction::ContextEstimate,
    turn_pending: bool,
    pending_btw_thread: Option<String>,
    pending_btw_turn_start: Option<usize>,
    resume_sessions: Option<Vec<sessionlog::SessionInfo>>,
    resume_scan: Option<Receiver<Result<Vec<sessionlog::SessionInfo>, String>>>,
    git_status: Option<SidebarGitInfo>,
    git_status_root: Option<PathBuf>,
    git_status_scan: Option<Receiver<SidebarGitInfo>>,
    git_status_last_requested: Option<Instant>,
    /// Present only for line mode, where slash commands can toggle between
    /// the native session card and the plain prompt without a restart.
    line_claude_tui: Option<bool>,
}

#[derive(Clone, Debug, Default)]
struct SidebarGitInfo {
    branch: Option<String>,
    changed_files: usize,
    changes: Vec<SidebarGitChange>,
}

#[derive(Clone, Debug)]
struct SidebarGitChange {
    status: state::FileStatus,
    path: String,
}

#[derive(Clone, Debug)]
struct NativeLineInfo {
    model: String,
    context_used: u64,
    context_limit: u64,
    cost: f64,
    messages: usize,
    tools: usize,
    changed_files: usize,
    branch: String,
    mode: String,
    thinking: String,
}

impl Default for InteractiveView {
    fn default() -> Self {
        Self {
            notices: Vec::new(),
            activity: "Ready".to_owned(),
            recent_tool: String::new(),
            activity_since: None,
            context_estimate: compaction::ContextEstimate::default(),
            turn_pending: false,
            pending_btw_thread: None,
            pending_btw_turn_start: None,
            resume_sessions: None,
            resume_scan: None,
            git_status: None,
            git_status_root: None,
            git_status_scan: None,
            git_status_last_requested: None,
            line_claude_tui: None,
        }
    }
}

impl InteractiveView {
    fn invalidate_resume_sessions(&mut self) {
        self.resume_sessions = None;
        self.resume_scan = None;
    }
}

/// Completion reported by a fullscreen background task.
///
/// Compaction has user-visible outcomes beyond success or failure, so it must
/// not be flattened into the ordinary turn result channel.
enum InteractiveTaskResult {
    Finished {
        result: Result<(), String>,
        automatic_compaction: Option<compaction::Outcome>,
    },
    Compaction(Result<compaction::Outcome, String>),
}

impl InteractiveTaskResult {
    fn finished(result: Result<(), String>) -> Self {
        Self::Finished {
            result,
            automatic_compaction: None,
        }
    }
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    agent_events: Receiver<agent::Event>,
    turn_sender: Sender<InteractiveTaskResult>,
    turn_results: Receiver<InteractiveTaskResult>,
    quiet: bool,
) -> Result<(), Box<dyn Error>> {
    let mut app = App::new();
    app.replace_messages(Vec::new());
    let mut view = InteractiveView::default();
    let mut render_cache = ui::MessageCache::default();
    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            append_view_message(&mut view, MessageRole::Notice, notice);
        }
        if let Some(banner) = runtime::session_banner(&prepared.runtime) {
            append_view_message(&mut view, MessageRole::Notice, banner);
        }
    }

    loop {
        drain_interactive_events(&mut view, prepared, &agent_events, &turn_results);
        refresh_runtime_app(&mut app, prepared, catalog, &mut view);
        terminal.draw(|frame| ui::draw(frame, &app, &mut render_cache))?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match app.handle_key(key) {
                Action::None => {}
                Action::Quit => {
                    if let Some(thread) = view.pending_btw_thread.as_deref() {
                        let _ = prepared.btw.cancel(thread);
                    }
                    prepared.runtime.agent().abort();
                    if let Some(planner) = prepared.planner.as_ref() {
                        planner.abort_review();
                    }
                    return Ok(());
                }
                Action::Abort => {
                    if let Some(thread) = view.pending_btw_thread.as_deref() {
                        let _ = prepared.btw.cancel(thread);
                    }
                    prepared.runtime.agent().abort();
                    if let Some(planner) = prepared.planner.as_ref() {
                        planner.abort_review();
                    }
                    view.activity = "Aborting".to_owned();
                }
                Action::CycleModel { direction } => {
                    match cycle_interactive_model(&prepared.runtime, catalog, direction) {
                        Ok(model) => {
                            app.invalidate_model_suggestions();
                            app.invalidate_thinking_suggestions();
                            view.activity = format!("Model set to {model}");
                        }
                        Err(error) => append_view_message(&mut view, MessageRole::Error, error),
                    }
                }
                Action::CycleThinking => match cycle_interactive_thinking(&prepared.runtime) {
                    Some(level) => {
                        app.invalidate_thinking_suggestions();
                        view.activity = format!("Thinking set to {level}");
                    }
                    None => append_view_message(
                        &mut view,
                        MessageRole::Notice,
                        "This model only supports thinking off.",
                    ),
                },
                Action::Submit(input) => {
                    match submit_interactive_input(
                        &mut app,
                        &mut view,
                        prepared,
                        catalog,
                        turn_sender.clone(),
                        input,
                        false,
                    ) {
                        CommandDispatch::Quit => {
                            prepared.runtime.agent().abort();
                            return Ok(());
                        }
                        CommandDispatch::TerminalCommand(command) => {
                            complete_fullscreen_terminal_command(
                                terminal, &mut app, &mut view, catalog, command,
                            )
                        }
                        CommandDispatch::Handled | CommandDispatch::NotCommand => {}
                    }
                }
                Action::FollowUp(input) => {
                    match submit_interactive_input(
                        &mut app,
                        &mut view,
                        prepared,
                        catalog,
                        turn_sender.clone(),
                        input,
                        true,
                    ) {
                        CommandDispatch::Quit => {
                            prepared.runtime.agent().abort();
                            return Ok(());
                        }
                        CommandDispatch::TerminalCommand(command) => {
                            complete_fullscreen_terminal_command(
                                terminal, &mut app, &mut view, catalog, command,
                            )
                        }
                        CommandDispatch::Handled | CommandDispatch::NotCommand => {}
                    }
                }
            },
            Event::Paste(text) => app.paste(&text),
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => app.scroll = app.scroll.saturating_add(3),
                MouseEventKind::ScrollDown => app.scroll = app.scroll.saturating_sub(3),
                _ => {}
            },
            _ => {}
        }
    }
}

/// Temporarily gives the real terminal to a command that prompts for input,
/// then restores Ratatui. OAuth, gateway setup, and secret entry cannot
/// safely run while the alternate screen and raw mode are active.
fn complete_fullscreen_terminal_command(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    app: &mut App,
    view: &mut InteractiveView,
    catalog: &catalog::Catalog,
    command: FullscreenTerminalCommand,
) {
    view.activity = command.activity();
    view.activity_since = Some(Instant::now());
    let result = run_fullscreen_terminal_command(terminal, &command);
    match result {
        Ok(()) => complete_fullscreen_terminal_success(app, view, catalog, command),
        Err(error) => {
            view.activity = format!("{} failed", command.name());
            view.activity_since = None;
            append_view_message(view, MessageRole::Error, error);
        }
    }
}

fn run_fullscreen_terminal_command(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    command: &FullscreenTerminalCommand,
) -> Result<(), String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    suspend_fullscreen_terminal(terminal)?;
    let status = Command::new(executable)
        .args(command.arguments())
        .status()
        .map_err(|error| format!("start {}: {error}", command.name()));
    let restored = restore_fullscreen_terminal(terminal);
    match (status, restored) {
        (_, Err(error)) => Err(format!(
            "restore terminal after {}: {error}",
            command.name()
        )),
        (Err(error), Ok(())) => Err(error),
        (Ok(status), Ok(())) if status.success() => Ok(()),
        (Ok(status), Ok(())) => Err(format!("{} exited with {status}", command.name())),
    }
}

fn complete_fullscreen_terminal_success(
    app: &mut App,
    view: &mut InteractiveView,
    catalog: &catalog::Catalog,
    command: FullscreenTerminalCommand,
) {
    app.invalidate_model_suggestions();
    app.invalidate_login_suggestions();
    app.invalidate_thinking_suggestions();
    view.activity_since = None;
    match command {
        FullscreenTerminalCommand::Login { provider_id, .. } => {
            catalog.clear_oauth_refresh_failure(&provider_id);
            view.activity = format!("Added {provider_id}");
            append_view_message(
                view,
                MessageRole::Notice,
                format!("Added {provider_id}. Use /model to switch providers."),
            );
        }
        FullscreenTerminalCommand::OmniSetup => {
            view.activity = "OmniRoute setup completed".to_owned();
            append_view_message(
                view,
                MessageRole::Notice,
                "OmniRoute setup completed. Run /omni sync to refresh models.",
            );
        }
        FullscreenTerminalCommand::ApertureOnboarding => {
            catalog.reload_aperture_state();
            view.activity = "Aperture onboarding completed".to_owned();
            append_view_message(view, MessageRole::Notice, "Aperture onboarding completed.");
        }
    }
}

fn suspend_fullscreen_terminal(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
) -> Result<(), String> {
    terminal.show_cursor().map_err(|error| error.to_string())?;
    disable_raw_mode().map_err(|error| error.to_string())?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .map_err(|error| error.to_string())
}

fn restore_fullscreen_terminal(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
) -> io::Result<()> {
    enable_raw_mode()?;
    if let Err(error) = execute!(
        terminal.backend_mut(),
        EnterAlternateScreen,
        EnableMouseCapture
    ) {
        let _ = disable_raw_mode();
        return Err(error);
    }
    terminal.clear()
}

fn drain_interactive_events(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    agent_events: &Receiver<agent::Event>,
    turn_results: &Receiver<InteractiveTaskResult>,
) {
    while let Ok(event) = agent_events.try_recv() {
        match event.kind {
            agent::EventKind::AgentStart => {
                view.turn_pending = true;
                view.activity = "Composing response".to_owned();
                view.activity_since = Some(Instant::now());
            }
            agent::EventKind::MessageUpdate => {
                view.activity = "Composing response".to_owned();
                view.activity_since.get_or_insert_with(Instant::now);
            }
            agent::EventKind::ToolExecutionStart => {
                view.activity = format!("Running {}", event.tool_name);
                view.recent_tool = format!("● {} running", event.tool_name);
                view.activity_since.get_or_insert_with(Instant::now);
            }
            agent::EventKind::ToolExecutionEnd => {
                if event.is_error {
                    view.activity = format!("{} failed", event.tool_name);
                    view.recent_tool = format!("× {} failed", event.tool_name);
                } else {
                    view.activity = format!("{} complete", event.tool_name);
                    view.recent_tool = format!("✓ {} complete", event.tool_name);
                }
            }
            agent::EventKind::AgentEnd => {
                view.turn_pending = false;
                view.activity = "Ready".to_owned();
                view.activity_since = None;
            }
            agent::EventKind::ContextCompacted => {
                if event.compaction.is_some() {
                    view.context_estimate.reset();
                    view.activity = "Context compacted".to_owned();
                    view.activity_since = None;
                }
            }
            agent::EventKind::TurnStart
            | agent::EventKind::TurnEnd
            | agent::EventKind::MessageStart
            | agent::EventKind::MessageEnd
            | agent::EventKind::ToolExecutionUpdate
            | agent::EventKind::ModelChange
            | agent::EventKind::ThinkingLevelChange => {}
            agent::EventKind::TranscriptReset => view.context_estimate.reset(),
        }
    }
    while let Ok(completion) = turn_results.try_recv() {
        finish_interactive_task(view, prepared, completion);
    }
    for notice in prepared.runtime.drain_notices() {
        append_view_message(
            view,
            MessageRole::Notice,
            format!("{}: {}", notice.kind, notice.text),
        );
    }
}

fn compaction_notice_text(outcome: &compaction::Outcome, automatic: bool) -> String {
    let action = if automatic {
        "compacted context automatically"
    } else {
        "compacted context"
    };
    format!(
        "{action}: {} messages → summary + {} recent messages",
        outcome.messages_before, outcome.retained_messages
    )
}

fn compaction_discard_notice(outcome: &compaction::Outcome) -> Option<String> {
    (outcome.dropped_queued_messages > 0).then(|| {
        format!(
            "{} queued message(s) were discarded: they were written against the transcript that was just compacted",
            outcome.dropped_queued_messages
        )
    })
}

fn print_compaction_outcome(outcome: &compaction::Outcome, automatic: bool, color: bool) {
    eprintln!(
        "{}",
        dim(&compaction_notice_text(outcome, automatic), color)
    );
    if let Some(notice) = compaction_discard_notice(outcome) {
        eprintln!("{}", dim(&notice, color));
    }
}

fn report_interactive_compaction(
    view: &mut InteractiveView,
    outcome: &compaction::Outcome,
    automatic: bool,
) {
    view.activity = "Context compacted".to_owned();
    view.activity_since = None;
    append_view_message(
        view,
        MessageRole::Notice,
        compaction_notice_text(outcome, automatic),
    );
    if let Some(notice) = compaction_discard_notice(outcome) {
        append_view_message(view, MessageRole::Notice, notice);
    }
}

fn finish_interactive_task(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    completion: InteractiveTaskResult,
) {
    match completion {
        InteractiveTaskResult::Finished {
            result,
            automatic_compaction,
        } => {
            if view.pending_btw_thread.is_some() {
                let _ = finish_pending_btw(view, prepared, result);
                return;
            }
            view.turn_pending = false;
            if let Some(outcome) = automatic_compaction {
                report_interactive_compaction(view, &outcome, true);
            }
            if let Err(error) = result {
                append_view_message(view, MessageRole::Error, error);
            }
        }
        InteractiveTaskResult::Compaction(result) => {
            view.turn_pending = false;
            match result {
                Ok(outcome) => report_interactive_compaction(view, &outcome, false),
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
        }
    }
}

/// Turns a completed asynchronous side-thread request into a visible
/// transcript card. It returns false for ordinary agent/planner work so the
/// caller can retain its existing completion behavior.
fn finish_pending_btw(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    result: Result<(), String>,
) -> bool {
    let Some(thread_id) = view.pending_btw_thread.take() else {
        return false;
    };
    let turn_index = view.pending_btw_turn_start.take();
    view.turn_pending = false;
    view.activity_since = None;
    match result {
        Err(error) => {
            view.activity = "BTW side thread failed".to_owned();
            append_view_message(view, MessageRole::Error, error);
        }
        Ok(()) => match prepared.btw.thread(&thread_id) {
            Err(error) => {
                view.activity = "BTW side thread failed".to_owned();
                append_view_message(view, MessageRole::Error, error.to_string());
            }
            Ok(thread) => match turn_index.and_then(|index| thread.turns.get(index)) {
                Some(turn) if turn.kind == btw::TurnKind::Answered => {
                    view.activity = format!("BTW {} answered", thread.id);
                    append_view_message(
                        view,
                        MessageRole::Assistant,
                        format!("BTW · {}\n{}", thread.id, turn.answer),
                    );
                }
                Some(turn) if turn.kind == btw::TurnKind::Error => {
                    view.activity = "BTW side thread failed".to_owned();
                    append_view_message(
                        view,
                        MessageRole::Error,
                        format!("BTW · {}\n{}", thread.id, turn.answer),
                    );
                }
                Some(_) => {
                    view.activity = "BTW side thread completed".to_owned();
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!(
                            "BTW · {} completed without a displayable answer.",
                            thread.id
                        ),
                    );
                }
                None => {
                    view.activity = "BTW side thread cancelled".to_owned();
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!("BTW · {} was cancelled.", thread.id),
                    );
                }
            },
        },
    }
    true
}

fn submit_interactive_input(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    turn_sender: Sender<InteractiveTaskResult>,
    input: String,
    follow_up: bool,
) -> CommandDispatch {
    app.record_submission(&input);
    let input = match prepared.expand_resource_input(&input) {
        Ok(Some(expanded)) => expanded,
        Ok(None) => input,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return CommandDispatch::Handled;
        }
    };
    if input.starts_with('/') {
        view.invalidate_resume_sessions();
        return dispatch_runtime_slash_command(
            app,
            view,
            prepared,
            catalog,
            turn_sender,
            &input,
            true,
        );
    }

    let agent = prepared.runtime.agent().clone();
    if follow_up {
        agent.follow_up(llm::Message::User(llm::UserMessage::text(
            input,
            now_millis(),
        )));
        view.activity = "Follow-up queued".to_owned();
        return CommandDispatch::Handled;
    }
    if app.streaming || view.turn_pending || agent.state().is_streaming {
        agent.steer(llm::Message::User(llm::UserMessage::text(
            input,
            now_millis(),
        )));
        view.activity = "Steering response".to_owned();
        return CommandDispatch::Handled;
    }

    if let Err(error) = prepared.sync_extensions() {
        append_view_message(view, MessageRole::Error, error.to_string());
        return CommandDispatch::Handled;
    }
    begin_interactive_turn(view, agent, turn_sender, input, "Starting response");
    CommandDispatch::NotCommand
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CommandDispatch {
    NotCommand,
    Handled,
    TerminalCommand(FullscreenTerminalCommand),
    Quit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FullscreenTerminalCommand {
    Login {
        provider_id: String,
        command: provider_cli::InteractiveAuthCommand,
    },
    OmniSetup,
    ApertureOnboarding,
}

impl FullscreenTerminalCommand {
    fn arguments(&self) -> Vec<String> {
        match self {
            Self::Login {
                provider_id,
                command,
            } => vec![
                "auth".to_owned(),
                command.as_auth_subcommand().to_owned(),
                provider_id.clone(),
            ],
            Self::OmniSetup => vec!["omni".to_owned(), "setup".to_owned()],
            Self::ApertureOnboarding => {
                vec!["aperture".to_owned(), "onboarding".to_owned()]
            }
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Login { .. } => "credential flow",
            Self::OmniSetup => "OmniRoute setup",
            Self::ApertureOnboarding => "Aperture onboarding",
        }
    }

    fn activity(&self) -> String {
        match self {
            Self::Login { provider_id, .. } => format!("Logging in to {provider_id}"),
            Self::OmniSetup => "Configuring OmniRoute".to_owned(),
            Self::ApertureOnboarding => "Configuring Aperture".to_owned(),
        }
    }
}

fn begin_interactive_turn(
    view: &mut InteractiveView,
    agent: agent::Agent,
    turn_sender: Sender<InteractiveTaskResult>,
    prompt: String,
    activity: &str,
) {
    view.turn_pending = true;
    view.activity = activity.to_owned();
    view.activity_since = Some(Instant::now());
    thread::spawn(move || {
        let automatic_compaction = match compaction::maybe_auto_compact(&agent) {
            Ok(outcome) => outcome,
            Err(error) => {
                let _ = turn_sender.send(InteractiveTaskResult::finished(Err(error.to_string())));
                return;
            }
        };
        let result = agent.prompt(prompt).map_err(|error| error.to_string());
        let _ = turn_sender.send(InteractiveTaskResult::Finished {
            result,
            automatic_compaction,
        });
    });
}

fn dispatch_ralph_slash_command(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    turn_sender: Sender<InteractiveTaskResult>,
    rest: &str,
) -> CommandDispatch {
    let Some(ralph_runtime) = prepared.ralph.as_ref() else {
        append_view_message(
            view,
            MessageRole::Error,
            "ralph loops are disabled; restart with -ralph to enable them",
        );
        return CommandDispatch::Handled;
    };

    let mut command = match if rest.is_empty() {
        Ok(ralph::RalphCommand::Status)
    } else {
        ralph::parse_command(rest)
    } {
        Ok(command) => command,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return CommandDispatch::Handled;
        }
    };
    if let ralph::RalphCommand::Start { task_content, .. } = &mut command
        && !task_content.starts_with('#')
    {
        *task_content = format!("# Task\n\n{task_content}");
    }
    let mutates = !matches!(
        &command,
        ralph::RalphCommand::List { .. } | ralph::RalphCommand::Status
    );
    if mutates
        && (app.streaming || view.turn_pending || prepared.runtime.agent().state().is_streaming)
    {
        append_view_message(
            view,
            MessageRole::Error,
            "Wait for the current response before changing a Ralph loop.",
        );
        return CommandDispatch::Handled;
    }

    match ralph_runtime.execute(command) {
        Ok(ralph::CommandResult::Started(state)) => {
            let task = match ralph_runtime.store().read_task(&state) {
                Ok(task) => task,
                Err(error) => {
                    append_view_message(view, MessageRole::Error, error.to_string());
                    return CommandDispatch::Handled;
                }
            };
            append_view_message(
                view,
                MessageRole::Notice,
                format!(
                    "started Ralph loop {} (max {} iterations)",
                    state.name, state.max_iterations
                ),
            );
            begin_interactive_turn(
                view,
                prepared.runtime.agent().clone(),
                turn_sender,
                ralph::build_prompt(&state, &task, false),
                "Starting Ralph iteration",
            );
        }
        Ok(ralph::CommandResult::Listed(states)) => {
            let text = if states.is_empty() {
                "No loops.".to_owned()
            } else {
                states
                    .into_iter()
                    .map(|state| state.summary())
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            append_view_message(view, MessageRole::Command, text);
        }
        Ok(ralph::CommandResult::Status(Some(state))) => {
            append_view_message(view, MessageRole::Command, state.summary());
        }
        Ok(ralph::CommandResult::Status(None)) => {
            append_view_message(view, MessageRole::Command, "No active loop.");
        }
        Ok(ralph::CommandResult::Resumed(state)) => {
            append_view_message(
                view,
                MessageRole::Notice,
                format!("resumed {} at iteration {}", state.name, state.iteration),
            );
        }
        Ok(ralph::CommandResult::Stopped(state)) => {
            append_view_message(
                view,
                MessageRole::Notice,
                format!("stopped {} at iteration {}", state.name, state.iteration),
            );
        }
        Ok(ralph::CommandResult::Archived(name)) => {
            append_view_message(view, MessageRole::Notice, format!("archived {name}."));
        }
        Ok(ralph::CommandResult::Deleted(name)) => {
            append_view_message(view, MessageRole::Notice, format!("deleted {name}."));
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
    CommandDispatch::Handled
}

/// Handles independent, in-memory side discussions without adding their turns
/// to the main session transcript.
fn dispatch_btw_slash_command(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    turn_sender: Sender<InteractiveTaskResult>,
    rest: &str,
) -> CommandDispatch {
    let (action, argument) = split_prompt_action(rest);
    match action.to_ascii_lowercase().as_str() {
        "" | "list" => {
            append_view_message(view, MessageRole::Command, list_btw_threads(prepared));
        }
        "resume" => {
            let (thread_id, question) = split_prompt_action(argument);
            if thread_id.is_empty() || question.is_empty() {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "usage: /btw resume <thread-id> <question>",
                );
            } else if let Err(error) = prepared.btw.resume_thread(thread_id) {
                append_view_message(view, MessageRole::Error, error.to_string());
            } else {
                report_btw_selection_warnings(view, prepared);
                start_btw_question(
                    app,
                    view,
                    prepared,
                    turn_sender,
                    thread_id.to_owned(),
                    question.to_owned(),
                );
            }
        }
        "bring" => {
            let (thread_id, scope) = split_prompt_action(argument);
            if thread_id.is_empty() {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "usage: /btw bring <thread-id> [latest|all|from:N]",
                );
            } else {
                match btw::parse_bring_selection(scope)
                    .map_err(|error| error.to_string())
                    .and_then(|scope| {
                        prepared
                            .btw
                            .bring_to_main(thread_id, scope)
                            .map_err(|error| error.to_string())
                    }) {
                    Ok(output) => append_view_message(view, MessageRole::Command, output.text),
                    Err(error) => append_view_message(view, MessageRole::Error, error),
                }
            }
        }
        "settings" => dispatch_btw_settings(view, prepared, argument),
        _ => {
            let state = prepared.runtime.agent().state();
            let created = prepared.btw.create_thread(&state);
            for warning in &created.selection.warnings {
                append_view_message(view, MessageRole::Notice, warning.clone());
            }
            start_btw_question(
                app,
                view,
                prepared,
                turn_sender,
                created.thread.id,
                rest.trim().to_owned(),
            );
        }
    }
    CommandDispatch::Handled
}

fn list_btw_threads(prepared: &runtime::PreparedSession) -> String {
    let mut lines = vec![
        "BTW side threads (in memory only):".to_owned(),
        "  /btw <question>                  start a fresh side thread".to_owned(),
        "  /btw resume <id> <question>      continue one".to_owned(),
        "  /btw bring <id> [latest|all|from:N]  show side context".to_owned(),
        "  /btw settings [level|remember]   view/change preferences".to_owned(),
    ];
    for summary in prepared.btw.list_threads() {
        lines.push(format!(
            "  {}  {} question(s)  {}",
            summary.id, summary.questions, summary.title
        ));
    }
    lines.join("\n")
}

fn dispatch_btw_settings(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    argument: &str,
) {
    let (setting, value) = split_prompt_action(argument);
    if setting.is_empty() {
        let settings = prepared.btw.read_settings();
        if settings.kind == btw::SettingsKind::Invalid {
            append_view_message(view, MessageRole::Error, settings.reason);
            return;
        }
        let model = if settings.settings.model.is_empty() {
            "current session model"
        } else {
            &settings.settings.model
        };
        let thinking = if settings.settings.thinking_level.is_empty() {
            "current session level"
        } else {
            &settings.settings.thinking_level
        };
        append_view_message(
            view,
            MessageRole::Command,
            format!(
                "pi-btw settings ({})\n  model: {model}\n  thinking: {thinking}\n  remember changes: {}",
                prepared.btw.settings_path().display(),
                settings.settings.effective_remember()
            ),
        );
        return;
    }

    let patch = match setting.to_ascii_lowercase().as_str() {
        "remember" => match value.to_ascii_lowercase().as_str() {
            "on" | "true" => btw::SettingsPatch {
                remember_thinking_level_changes: btw::SettingChange::Set(true),
                ..btw::SettingsPatch::default()
            },
            "off" | "false" => btw::SettingsPatch {
                remember_thinking_level_changes: btw::SettingChange::Set(false),
                ..btw::SettingsPatch::default()
            },
            _ => {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "usage: /btw settings remember <on|off>",
                );
                return;
            }
        },
        "model" => {
            if value.is_empty() {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "usage: /btw settings model <provider/model>",
                );
                return;
            }
            btw::SettingsPatch {
                model: btw::SettingChange::Set(value.to_owned()),
                ..btw::SettingsPatch::default()
            }
        }
        level if value.is_empty() => btw::SettingsPatch {
            thinking_level: btw::SettingChange::Set(level.to_owned()),
            ..btw::SettingsPatch::default()
        },
        _ => {
            append_view_message(
                view,
                MessageRole::Error,
                "usage: /btw settings [level|remember <on|off>|model <provider/model>]",
            );
            return;
        }
    };
    match prepared.btw.update_settings(patch) {
        Ok(_) => {
            view.activity = "BTW settings saved".to_owned();
            append_view_message(view, MessageRole::Notice, "BTW settings saved.");
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
}

fn report_btw_selection_warnings(view: &mut InteractiveView, prepared: &runtime::PreparedSession) {
    let state = prepared.runtime.agent().state();
    for warning in prepared.btw.resolve_selection(&state).warnings {
        append_view_message(view, MessageRole::Notice, warning);
    }
}

fn start_btw_question(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    turn_sender: Sender<InteractiveTaskResult>,
    thread_id: String,
    question: String,
) {
    if question.trim().is_empty() {
        append_view_message(view, MessageRole::Error, "a BTW question cannot be empty");
        return;
    }
    if app.streaming || view.turn_pending || prepared.runtime.agent().state().is_streaming {
        append_view_message(
            view,
            MessageRole::Error,
            "Wait for the active response before opening /btw.",
        );
        return;
    }
    let turns_before = match prepared.btw.thread(&thread_id) {
        Ok(thread) => thread.turns.len(),
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return;
        }
    };
    let queued = match prepared.btw.enqueue_prompt(&thread_id, question) {
        Ok(status) => status,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return;
        }
    };
    if queued.running {
        append_view_message(
            view,
            MessageRole::Notice,
            format!("Queued side question for {}.", queued.thread_id),
        );
        return;
    }

    let side_runtime = prepared.btw.clone();
    let state = prepared.runtime.agent().state();
    let worker_thread_id = thread_id.clone();
    view.pending_btw_thread = Some(thread_id);
    view.pending_btw_turn_start = Some(turns_before);
    view.turn_pending = true;
    view.activity = "BTW side thread is answering".to_owned();
    view.activity_since = Some(Instant::now());
    thread::spawn(move || {
        let result = match side_runtime.run_next(&state, &worker_thread_id) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err("BTW side thread did not have a queued question".to_owned()),
            Err(error) => Err(error.to_string()),
        };
        let _ = turn_sender.send(InteractiveTaskResult::finished(result));
    });
}

/// Executes slash commands that can be served without leaving the fullscreen
/// Ratatui program. Commands with an unavailable integration report that fact
/// in the transcript rather than pretending they changed runtime state.
fn dispatch_runtime_slash_command(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    turn_sender: Sender<InteractiveTaskResult>,
    input: &str,
    fullscreen: bool,
) -> CommandDispatch {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();
    match command {
        "/exit" | "/quit" => {
            if let Some(thread) = view.pending_btw_thread.as_deref() {
                let _ = prepared.btw.cancel(thread);
            }
            CommandDispatch::Quit
        }
        "/help" | "/?" => {
            append_view_message(
                view,
                MessageRole::Command,
                "Slash commands:\n  /help                 Show this help\n  /model [ref]          List or choose an authenticated model\n  /login <provider>     Add an OAuth or API-key provider\n  /thinking [level]     List or choose reasoning effort\n  /tools                List active tools\n  /status, /session     Show live session information\n  /messages             Show transcript summary\n  /queue                Show queued steering/follow-up messages\n  /steer <text>         Guide an active response\n  /followup <text>      Queue the next turn\n  /clear, /new          Reset this transcript\n  /compact [focus]      Summarize older context and keep recent turns\n  /name <text>          Set the persisted session name\n  /sessions             List saved sessions\n  /resume <id>          Switch to a saved session\n  /tree, /fork, /label  Inspect or rewind saved-session branches\n  /clone                Duplicate the current saved session\n  /export [--md] [path] Export the current session\n  /import <path>        Copy a session into this workspace\n  /omni [command]       Manage an OmniRoute gateway\n  /aperture [command]   Manage gateway routing and connectors\n  /prompt <action>      List, save, edit, remove, back up, or restore prompts\n  /reload               Reload local context, prompts, and skills\n  /resources            Show loaded context, prompts, and skills\n  /ralph <subcommand>   Manage Ralph loops\n  /planner              Toggle planning mode\n  /planner-review [URL] Review local changes or a GitHub PR\n  /planner-annotate <target>  Annotate a file, folder, or URL\n  /planner-last         Annotate the latest assistant response\n  /use-claude-code-tui  Enable the native startup/editor look in line mode\n  /use-default-tui      Restore the plain line-oriented look\n  /hotkeys              Show keyboard shortcuts\n  /exit                 Leave chat"
                    .to_owned(),
            );
            CommandDispatch::Handled
        }
        "/hotkeys" => {
            append_view_message(
                view,
                MessageRole::Command,
                "Enter       send or accept selection; steer while a response is active\nAlt-Enter   queue a follow-up\nShift-Enter insert a newline\nUp/Down     navigate palette, editor lines, or history\nAlt-←/→     move by word; Home/End move within a line\nTab         complete the selected command; Shift-Tab cycle thinking\nCtrl-L      open model selector; Ctrl-P cycle models; Ctrl-O expand tools\nCtrl-T      toggle displayed thinking; PgUp/PgDn scroll transcript\nEsc         clear input or abort a response; Ctrl-C abort or quit\nCtrl-D      quit when the editor is empty"
                    .to_owned(),
            );
            CommandDispatch::Handled
        }
        "/clear" | "/new" => {
            match prepared
                .runtime
                .agent()
                .reset_with_reason(if command == "/new" {
                    "new session"
                } else {
                    "clear"
                }) {
                Ok(()) => {
                    app.scroll = 0;
                    view.activity = "Transcript cleared".to_owned();
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        "Transcript reset and recorded in the active session.",
                    );
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/status" | "/session" | "/sidebar" => {
            append_view_message(
                view,
                MessageRole::Command,
                session_status(prepared, &view.activity),
            );
            CommandDispatch::Handled
        }
        "/messages" => {
            let state = prepared.runtime.agent().state();
            let summary = transcript_summary(&state.messages);
            append_view_message(
                view,
                MessageRole::Command,
                if summary.is_empty() {
                    "The transcript is empty.".to_owned()
                } else {
                    summary
                },
            );
            CommandDispatch::Handled
        }
        "/model" if rest.is_empty() => {
            let choices = configured_model_references(catalog);
            append_view_message(
                view,
                MessageRole::Command,
                if choices.is_empty() {
                    "No authenticated models are available. Run `goshcoder auth set <provider>` outside the fullscreen interface, then reopen chat."
                        .to_owned()
                } else {
                    format!("Available models:\n{}", choices.join("\n"))
                },
            );
            CommandDispatch::Handled
        }
        "/model" => {
            match runtime::set_model(&prepared.runtime, catalog, rest) {
                Ok(model) => {
                    app.invalidate_model_suggestions();
                    app.invalidate_thinking_suggestions();
                    view.activity = format!("Model set to {}/{}", model.provider, model.id)
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/thinking" if rest.is_empty() => {
            let state = prepared.runtime.agent().state();
            let levels = stream::supported_thinking_levels(&state.model);
            append_view_message(
                view,
                MessageRole::Command,
                format!(
                    "Thinking levels for {}/{}:\n{}",
                    state.model.provider,
                    state.model.id,
                    levels.join("\n")
                ),
            );
            CommandDispatch::Handled
        }
        "/thinking" => {
            let state = prepared.runtime.agent().state();
            let levels = stream::supported_thinking_levels(&state.model);
            if levels.iter().any(|level| level == rest) {
                prepared.runtime.agent().set_thinking_level(rest);
                app.invalidate_thinking_suggestions();
                view.activity = format!("Thinking set to {rest}");
            } else {
                append_view_message(
                    view,
                    MessageRole::Error,
                    format!(
                        "{rest:?} is not supported by {}/{}; choose: {}",
                        state.model.provider,
                        state.model.id,
                        levels.join(", ")
                    ),
                );
            }
            CommandDispatch::Handled
        }
        "/tools" => {
            let tools = prepared.runtime.agent().state().tools;
            append_view_message(
                view,
                MessageRole::Command,
                if tools.is_empty() {
                    "No tools are active for this session.".to_owned()
                } else {
                    tools
                        .iter()
                        .map(|tool| format!("{} — {}", tool.name, tool.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                },
            );
            CommandDispatch::Handled
        }
        "/queue" => {
            let agent = prepared.runtime.agent();
            append_view_message(
                view,
                MessageRole::Command,
                format!("{} message(s) queued.", agent.queued_message_count()),
            );
            CommandDispatch::Handled
        }
        "/steer" if rest.is_empty() => {
            append_view_message(view, MessageRole::Error, "usage: /steer <text>");
            CommandDispatch::Handled
        }
        "/steer" => {
            prepared
                .runtime
                .agent()
                .steer(llm::Message::User(llm::UserMessage::text(
                    rest,
                    now_millis(),
                )));
            view.activity = "Steering response".to_owned();
            CommandDispatch::Handled
        }
        "/followup" if rest.is_empty() => {
            append_view_message(view, MessageRole::Error, "usage: /followup <text>");
            CommandDispatch::Handled
        }
        "/followup" => {
            prepared
                .runtime
                .agent()
                .follow_up(llm::Message::User(llm::UserMessage::text(
                    rest,
                    now_millis(),
                )));
            view.activity = "Follow-up queued".to_owned();
            CommandDispatch::Handled
        }
        "/name" if rest.is_empty() => {
            append_view_message(
                view,
                MessageRole::Command,
                prepared
                    .runtime
                    .name()
                    .unwrap_or_else(|| "This session has no name.".to_owned()),
            );
            CommandDispatch::Handled
        }
        "/name" => {
            match prepared.runtime.set_name(rest) {
                Ok(()) => view.activity = format!("Session named {rest:?}"),
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/tree" => {
            let points = prepared.runtime.branch_points();
            append_view_message(
                view,
                MessageRole::Command,
                if points.is_empty() {
                    "No session rewind points are available.".to_owned()
                } else {
                    points
                        .iter()
                        .map(|point| {
                            let label = point
                                .label
                                .as_deref()
                                .map(|label| format!(" [{label}]"))
                                .unwrap_or_default();
                            let current = if point.current { " *" } else { "" };
                            format!(
                                "{}. {}{}{}",
                                point.index,
                                first_line(&point.text),
                                label,
                                current
                            )
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                },
            );
            CommandDispatch::Handled
        }
        "/fork" => {
            match parse_branch_index(rest).and_then(|index| {
                prepared
                    .runtime
                    .fork_to(index)
                    .map_err(|error| error.to_string())
            }) {
                Ok(point) => {
                    view.activity = format!("Rewound to {}", first_line(&point.text));
                    app.scroll = 0;
                }
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/label" => {
            let Some((index, label)) = rest.split_once(char::is_whitespace) else {
                append_view_message(view, MessageRole::Error, "usage: /label <point> <name>");
                return CommandDispatch::Handled;
            };
            match index
                .parse::<usize>()
                .map_err(|_| "branch point must be a positive number".to_owned())
                .and_then(|index| {
                    prepared
                        .runtime
                        .label(index, label)
                        .map_err(|error| error.to_string())
                }) {
                Ok(()) => view.activity = "Branch label saved".to_owned(),
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/clone" => {
            match prepared.runtime.clone_session() {
                Ok(handle) => {
                    view.activity = format!("Cloned session {}", short_id(&handle.id));
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!("Created session {}.", handle.id),
                    );
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/export" => dispatch_session_export_slash_command(view, prepared, rest, fullscreen),
        "/import" => dispatch_session_import_slash_command(view, prepared, rest, fullscreen),
        "/omni" if fullscreen && is_omni_setup(rest) => dispatch_fullscreen_terminal_command(
            app,
            view,
            prepared,
            FullscreenTerminalCommand::OmniSetup,
        ),
        "/omni" => dispatch_omni_slash_command(view, catalog, rest, fullscreen),
        "/aperture" if fullscreen && is_aperture_onboarding(rest) => {
            dispatch_fullscreen_terminal_command(
                app,
                view,
                prepared,
                FullscreenTerminalCommand::ApertureOnboarding,
            )
        }
        "/aperture" => dispatch_aperture_slash_command(view, catalog, rest, None, fullscreen),
        "/aperture:onboarding" if fullscreen && rest.trim().is_empty() => {
            dispatch_fullscreen_terminal_command(
                app,
                view,
                prepared,
                FullscreenTerminalCommand::ApertureOnboarding,
            )
        }
        "/aperture:onboarding" => {
            dispatch_aperture_slash_command(view, catalog, rest, Some("onboarding"), fullscreen)
        }
        "/aperture:settings" => {
            dispatch_aperture_slash_command(view, catalog, rest, Some("settings"), fullscreen)
        }
        "/resources" => {
            let resources = prepared.resources();
            append_view_message(
                view,
                MessageRole::Command,
                resources.report(&prepared.resource_paths).render(),
            );
            CommandDispatch::Handled
        }
        "/reload" => {
            match prepared.reload_resources() {
                Ok(resources) => {
                    view.activity = "Local resources reloaded".to_owned();
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!(
                            "Reloaded local resources. They apply to future turns.\n\n{}",
                            resources.report(&prepared.resource_paths).render()
                        ),
                    );
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/sessions" => {
            match list_interactive_sessions(prepared) {
                Ok(output) => append_view_message(view, MessageRole::Command, output),
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/resume" if rest.is_empty() => {
            append_view_message(
                view,
                MessageRole::Error,
                "/resume needs a session id, prefix, or path",
            );
            CommandDispatch::Handled
        }
        "/resume" => {
            match prepared.runtime.switch_to(rest) {
                Ok(handle) => {
                    app.scroll = 0;
                    view.context_estimate.reset();
                    view.activity = format!("Resumed session {}", short_id(&handle.id));
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!("Switched to session {}.", handle.id),
                    );
                    if !fullscreen {
                        let restored = prepared.runtime.restored();
                        let transcript =
                            restored_transcript_text(&restored.messages, "resumed transcript");
                        if !transcript.is_empty() {
                            append_view_message(view, MessageRole::Command, transcript);
                        }
                    }
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/btw" => dispatch_btw_slash_command(app, view, prepared, turn_sender, rest),
        "/prompt" | "/prompts" => dispatch_prompt_slash_command(view, prepared, rest, fullscreen),
        "/ralph" => dispatch_ralph_slash_command(app, view, prepared, turn_sender, rest),
        "/planner" | "/plannator" | "/plannotator" => {
            match prepared.toggle_planner() {
                Ok(phase) => view.activity = format!("Planner: {}", phase.as_str()),
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/planner-review" | "/plannotator-review" => {
            let Some(planner) = prepared.planner.as_ref() else {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "Planner is unavailable in this session. Reopen chat with planner support enabled.",
                );
                return CommandDispatch::Handled;
            };
            let workspace = planner.workspace_root().to_path_buf();
            let target = rest.to_owned();
            start_planner_review(
                app,
                view,
                prepared,
                turn_sender,
                "Loading code review",
                move || {
                    planner_runtime::load_diff_review(workspace, &target)
                        .map(|request| ("the code changes".to_owned(), request))
                },
            )
        }
        "/planner-annotate" | "/plannotator-annotate" => {
            if rest.is_empty() {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "usage: /planner-annotate <target>",
                );
                return CommandDispatch::Handled;
            }
            let Some(planner) = prepared.planner.as_ref() else {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "Planner is unavailable in this session. Reopen chat with planner support enabled.",
                );
                return CommandDispatch::Handled;
            };
            let workspace = planner.workspace_root().to_path_buf();
            let target = rest.to_owned();
            start_planner_review(
                app,
                view,
                prepared,
                turn_sender,
                "Collecting annotation",
                move || {
                    let collector = plannotator::TextCollector::new(&workspace)
                        .map_err(|error| error.to_string())?;
                    let collected = collector
                        .collect(&target)
                        .map_err(|error| error.to_string())?;
                    let request = collected.review_request();
                    Ok((collected.feedback_subject, request))
                },
            )
        }
        "/planner-last" | "/plannotator-last" => {
            let messages = prepared.runtime.agent().state().messages;
            start_planner_review(
                app,
                view,
                prepared,
                turn_sender,
                "Opening annotation",
                move || {
                    let collected = plannotator::collect_last_assistant_response(&messages)
                        .map_err(|error| error.to_string())?;
                    let request = collected.review_request();
                    Ok((collected.feedback_subject, request))
                },
            )
        }
        "/compact" => {
            if app.streaming || view.turn_pending || prepared.runtime.agent().state().is_streaming {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "Wait for the current response before compacting.",
                );
                return CommandDispatch::Handled;
            }
            let agent = prepared.runtime.agent().clone();
            let instructions = rest.to_owned();
            view.turn_pending = true;
            view.activity = "Compacting context".to_owned();
            view.activity_since = Some(Instant::now());
            thread::spawn(move || {
                let result =
                    compaction::compact(&agent, &instructions).map_err(|error| error.to_string());
                let _ = turn_sender.send(InteractiveTaskResult::Compaction(result));
            });
            CommandDispatch::Handled
        }
        "/system" if rest.is_empty() => {
            append_view_message(
                view,
                MessageRole::Command,
                prepared.runtime.agent().state().system_prompt,
            );
            CommandDispatch::Handled
        }
        "/system" => {
            match prepared.set_base_system_prompt(rest) {
                Ok(()) => {
                    view.activity = "System prompt updated for this session".to_owned();
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        "The new system prompt applies to future turns in this session.",
                    );
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/use-claude-code-tui" => {
            if fullscreen {
                append_view_message(
                    view,
                    MessageRole::Notice,
                    "The fullscreen interface already uses the native sidebar layout; this switch only affects line mode.",
                );
            } else if let Some(claude_tui) = view.line_claude_tui.as_mut() {
                *claude_tui = true;
                append_view_message(
                    view,
                    MessageRole::Notice,
                    "Using native pi-claude-code-tui look.",
                );
            }
            CommandDispatch::Handled
        }
        "/use-default-tui" => {
            if fullscreen {
                append_view_message(
                    view,
                    MessageRole::Notice,
                    "The fullscreen layout cannot be changed in-place; restart with -fullscreen=false for the line-mode interface.",
                );
            } else if let Some(claude_tui) = view.line_claude_tui.as_mut() {
                *claude_tui = false;
                append_view_message(
                    view,
                    MessageRole::Notice,
                    "Using default GoshCoder interface.",
                );
            }
            CommandDispatch::Handled
        }
        "/login" => dispatch_login_slash_command(app, view, prepared, catalog, rest, fullscreen),
        _ if command.starts_with('/') => {
            append_view_message(
                view,
                MessageRole::Error,
                format!("{command} has not yet been migrated to the Rust frontend."),
            );
            CommandDispatch::Handled
        }
        _ => CommandDispatch::NotCommand,
    }
}

fn dispatch_login_slash_command(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch {
    let providers = rest.split_whitespace().collect::<Vec<_>>();
    let [provider_id] = providers.as_slice() else {
        if providers.is_empty() {
            append_view_message(view, MessageRole::Command, login_command_help());
        } else {
            append_view_message(view, MessageRole::Error, "usage: /login <provider>");
        }
        return CommandDispatch::Handled;
    };

    if app.streaming || view.turn_pending || prepared.runtime.agent().state().is_streaming {
        append_view_message(
            view,
            MessageRole::Error,
            "Wait for the current response before changing credentials.",
        );
        return CommandDispatch::Handled;
    }

    let command = match provider_cli::interactive_auth_command(catalog, provider_id) {
        Ok(command) => command,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return CommandDispatch::Handled;
        }
    };
    if fullscreen {
        view.activity = format!("Logging in to {provider_id}");
        view.activity_since = Some(Instant::now());
        return CommandDispatch::TerminalCommand(FullscreenTerminalCommand::Login {
            provider_id: (*provider_id).to_owned(),
            command,
        });
    }

    match provider_cli::auth_for_provider(catalog, provider_id) {
        Ok(outcome) => {
            app.invalidate_model_suggestions();
            app.invalidate_login_suggestions();
            view.activity = format!("Added {}", outcome.provider_id);
            append_view_message(view, MessageRole::Notice, outcome.notice());
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
    CommandDispatch::Handled
}

fn login_command_help() -> String {
    format!(
        "Add a provider with /login <provider>.\nOAuth providers: {}\nAPI-key providers prompt for a key; existing logins are kept.",
        oauth::implemented_provider_ids().join(", ")
    )
}

fn transcript_summary(messages: &[llm::Message]) -> String {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| {
            let (role, preview) = compaction::summary_text(message).map_or_else(
                || (message.role(), first_line(&message.text_preview())),
                |summary| ("compacted", first_line(&summary)),
            );
            format!("{:>3}  {:<10} {}", index + 1, role, preview)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn dispatch_session_export_slash_command(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch {
    let arguments = rest
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let options = sessions::parse_export_options(&arguments);
    let destination = match options.values.as_slice() {
        [] => None,
        [destination] => Some(destination.clone()),
        _ => {
            append_view_message(
                view,
                MessageRole::Error,
                "usage: /export [--md|--jsonl] [output-path]",
            );
            return CommandDispatch::Handled;
        }
    };
    if fullscreen && destination.as_deref().is_none_or(|path| path == "-") {
        append_view_message(
            view,
            MessageRole::Error,
            "give /export a destination path; the fullscreen interface has no console to print to",
        );
        return CommandDispatch::Handled;
    }

    let result = match destination.as_deref() {
        None | Some("-") => {
            let content = prepared.runtime.export(options.format);
            content.and_then(|content| {
                let mut stdout = io::stdout().lock();
                stdout.write_all(&content)?;
                stdout.flush()?;
                Ok(())
            })
        }
        Some(destination) => prepared.runtime.export_to(options.format, destination),
    };
    match result {
        Ok(()) => {
            view.activity = "Session exported".to_owned();
            if let Some(destination) = destination.filter(|destination| destination != "-") {
                let id = prepared
                    .runtime
                    .id()
                    .map(|id| short_id(&id).to_owned())
                    .unwrap_or_else(|| "session".to_owned());
                append_view_message(
                    view,
                    MessageRole::Notice,
                    format!("Exported {id} to {destination}."),
                );
            }
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
    CommandDispatch::Handled
}

fn dispatch_session_import_slash_command(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch {
    let arguments = rest.split_whitespace().collect::<Vec<_>>();
    let [source] = arguments.as_slice() else {
        append_view_message(
            view,
            MessageRole::Error,
            "/import needs exactly one .jsonl session path",
        );
        return CommandDispatch::Handled;
    };
    match prepared.runtime.import_copy(source) {
        Ok(handle) => {
            if !fullscreen {
                let write_result = (|| -> io::Result<()> {
                    let mut stdout = io::stdout().lock();
                    writeln!(stdout, "{}", handle.path.display())?;
                    stdout.flush()
                })();
                if let Err(error) = write_result {
                    append_view_message(view, MessageRole::Error, error.to_string());
                    return CommandDispatch::Handled;
                }
            }
            view.activity = format!("Imported session {}", short_id(&handle.id));
            let location = if fullscreen {
                format!("\n{}", handle.path.display())
            } else {
                String::new()
            };
            append_view_message(
                view,
                MessageRole::Notice,
                format!(
                    "Imported as {}.{location}\nUse /resume {} to switch to it.",
                    short_id(&handle.id),
                    handle.id
                ),
            );
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
    CommandDispatch::Handled
}

fn dispatch_fullscreen_terminal_command(
    app: &App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    command: FullscreenTerminalCommand,
) -> CommandDispatch {
    if app.streaming || view.turn_pending || prepared.runtime.agent().state().is_streaming {
        append_view_message(
            view,
            MessageRole::Error,
            format!("Wait for the current response before {}.", command.name()),
        );
        return CommandDispatch::Handled;
    }
    view.activity = command.activity();
    view.activity_since = Some(Instant::now());
    CommandDispatch::TerminalCommand(command)
}

fn is_omni_setup(rest: &str) -> bool {
    rest.trim().eq_ignore_ascii_case("setup")
}

fn is_aperture_onboarding(rest: &str) -> bool {
    matches!(
        rest.trim().to_ascii_lowercase().as_str(),
        "onboarding" | "setup" | "configure"
    )
}

fn dispatch_omni_slash_command(
    view: &mut InteractiveView,
    catalog: &catalog::Catalog,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch {
    let arguments = rest
        .split_whitespace()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let interactive = !fullscreen && io::stdin().is_terminal() && io::stderr().is_terminal();
    match omni_cli::execute(&arguments, catalog, interactive) {
        Ok(output) => {
            view.activity = "OmniRoute command completed".to_owned();
            append_view_message(
                view,
                MessageRole::Command,
                if output.is_empty() {
                    "OmniRoute command completed.".to_owned()
                } else {
                    output
                },
            );
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
    CommandDispatch::Handled
}

fn dispatch_aperture_slash_command(
    view: &mut InteractiveView,
    catalog: &catalog::Catalog,
    rest: &str,
    alias: Option<&str>,
    fullscreen: bool,
) -> CommandDispatch {
    let mut arguments = alias.into_iter().map(ToOwned::to_owned).collect::<Vec<_>>();
    arguments.extend(rest.split_whitespace().map(ToOwned::to_owned));
    match aperture_cli::execute(&arguments, !fullscreen) {
        Ok(output) => {
            catalog.reload_aperture_state();
            view.activity = "Aperture command completed".to_owned();
            append_view_message(
                view,
                MessageRole::Command,
                if output.is_empty() {
                    "Aperture command completed.".to_owned()
                } else {
                    output
                },
            );
        }
        Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
    }
    CommandDispatch::Handled
}

/// Handles `/prompt` and its compatibility alias without writing directly to
/// the terminal. This keeps prompt management usable in both line mode and
/// the Ratatui alternate screen.
fn dispatch_prompt_slash_command(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch {
    let result = prompt_slash_command(prepared, rest, fullscreen);
    match result {
        Ok(output) if !output.is_empty() => append_view_message(view, MessageRole::Command, output),
        Ok(_) => {}
        Err(error) => append_view_message(view, MessageRole::Error, error),
    }
    CommandDispatch::Handled
}

fn prompt_slash_command(
    prepared: &runtime::PreparedSession,
    rest: &str,
    fullscreen: bool,
) -> Result<String, String> {
    let (action, argument) = split_prompt_action(rest);
    match action {
        "" | "list" => prompt_list(prepared),
        "save" => prompt_save(prepared, argument),
        "rm" | "remove" | "delete" => prompt_remove(prepared, argument),
        "edit" => prompt_edit(prepared, argument, fullscreen),
        "backup" => prompt_backup(prepared, argument),
        "restore" => prompt_restore(prepared, argument),
        action => Err(format!(
            "unknown /prompt action {action:?}; use list, save, edit, rm, backup or restore"
        )),
    }
}

fn split_prompt_action(input: &str) -> (&str, &str) {
    let input = input.trim();
    let Some(index) = input.find(char::is_whitespace) else {
        return (input, "");
    };
    (&input[..index], input[index..].trim())
}

fn prompt_list(prepared: &runtime::PreparedSession) -> Result<String, String> {
    let resources = prepared.resources();
    if resources.templates.is_empty() {
        return Ok(
            "no saved prompts; /prompt save <name> stores the last thing you asked".to_owned(),
        );
    }
    Ok(resources
        .templates
        .iter()
        .map(|template| {
            let description = if template.description.is_empty() {
                String::new()
            } else {
                format!("  {}", template.description)
            };
            format!(
                "/{}{}\n    {}",
                template.name,
                description,
                template.path.display()
            )
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

fn prompt_save(prepared: &runtime::PreparedSession, argument: &str) -> Result<String, String> {
    let (first, after_first) = split_prompt_action(argument);
    let (scope, name_and_body) = match first {
        "--project" | "-project" => (resources::PromptScope::Workspace, after_first),
        _ => (resources::PromptScope::User, argument.trim()),
    };
    let (name, inline) = split_prompt_action(name_and_body);
    if name.is_empty() {
        return Err("/prompt save needs a name, e.g. /prompt save review".to_owned());
    }

    let captured = inline.trim().is_empty();
    let body = if captured {
        last_user_prompt(&prepared.runtime.agent().state().messages).ok_or_else(|| {
            "there is no previous message to save; pass the text after the name".to_owned()
        })?
    } else {
        inline.to_owned()
    };
    let resources = prepared.resources();
    let save = resources::save_template(
        &prepared.resource_paths,
        name,
        &body,
        resources::SaveTemplateOptions {
            scope,
            reserved_names: reserved_prompt_names(&resources),
            literal: captured,
            ..resources::SaveTemplateOptions::default()
        },
    )
    .map_err(|error| match error {
        resources::ResourceError::TemplateExists { .. } => {
            format!("{error}; /prompt rm {name} first if you meant to replace it")
        }
        _ => error.to_string(),
    })?;
    prepared
        .reload_templates()
        .map_err(|error| format!("saved /{name}, but could not reload prompts: {error}"))?;

    let mut lines = vec![format!("saved /{name} to {}", save.path.display())];
    if captured && resources::has_placeholders(&body) {
        lines.push(
            "the captured text contains $ placeholders; they were escaped so it expands exactly as written"
                .to_owned(),
        );
    }
    if let Some(shadowed_by) = save.shadowed_by {
        lines.push(format!(
            "note: /{name} already resolves to {}, which takes precedence",
            shadowed_by.display()
        ));
    }
    Ok(lines.join("\n"))
}

fn prompt_remove(prepared: &runtime::PreparedSession, argument: &str) -> Result<String, String> {
    let (name, scope) = parse_prompt_scope(argument);
    if name.is_empty() {
        return Err("/prompt rm needs a name".to_owned());
    }
    let removed = resources::remove_template(&prepared.resource_paths, &name, scope)
        .map_err(|error| error.to_string())?;
    prepared.reload_templates().map_err(|error| {
        format!(
            "removed {}, but could not reload prompts: {error}",
            removed.path.display()
        )
    })?;
    let mut message = format!("removed {}", removed.path.display());
    if removed.removed_symbolic_link {
        message.push_str("\nnote: removed the symbolic link itself, not its target");
    }
    Ok(message)
}

fn prompt_edit(
    prepared: &runtime::PreparedSession,
    argument: &str,
    fullscreen: bool,
) -> Result<String, String> {
    let (name, _) = parse_prompt_scope(argument);
    if name.is_empty() {
        return Err("/prompt edit needs a name".to_owned());
    }
    let resources = prepared.resources();
    let template = resources
        .find_template(&name)
        .ok_or_else(|| format!("no prompt named {name:?}; /prompt list shows what is saved"))?;
    let path = template.path.clone();
    let editor = ["VISUAL", "EDITOR"].iter().find_map(|name| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    });
    let Some(editor) = editor else {
        return Ok(format!(
            "set $EDITOR to edit in place; the file is {}",
            path.display()
        ));
    };
    if fullscreen {
        return Ok(format!(
            "edit it outside the app: {editor} {}",
            path.display()
        ));
    }
    let mut fields = editor.split_whitespace();
    let program = fields
        .next()
        .ok_or_else(|| "$EDITOR does not contain an executable".to_owned())?;
    let status = std::process::Command::new(program)
        .args(fields)
        .arg(&path)
        .status()
        .map_err(|error| format!("run {editor}: {error}"))?;
    if !status.success() {
        return Err(format!("run {editor}: exited with {status}"));
    }
    prepared
        .reload_templates()
        .map_err(|error| format!("edited /{name}, but could not reload prompts: {error}"))?;
    Ok(format!("reloaded /{name}"))
}

fn prompt_backup(prepared: &runtime::PreparedSession, argument: &str) -> Result<String, String> {
    let output = (!argument.trim().is_empty()).then(|| std::path::Path::new(argument.trim()));
    let (path, warnings) =
        prompts::backup_at(&prepared.resource_paths, output).map_err(|error| error.to_string())?;
    let mut lines = warnings
        .into_iter()
        .map(|warning| format!("warning: {warning}"))
        .collect::<Vec<_>>();
    lines.push(format!("backed up prompts to {}", path.display()));
    Ok(lines.join("\n"))
}

fn prompt_restore(prepared: &runtime::PreparedSession, argument: &str) -> Result<String, String> {
    let (archive_path, flags) = split_prompt_action(argument);
    if archive_path.is_empty() {
        return Err("/prompt restore needs an archive path".to_owned());
    }
    let options = resources::RestoreOptions {
        overwrite: flags
            .split_whitespace()
            .any(|flag| matches!(flag, "--overwrite" | "-overwrite")),
        dry_run: flags
            .split_whitespace()
            .any(|flag| matches!(flag, "--dry-run" | "-dry-run")),
        reserved_names: reserved_prompt_names(&prepared.resources()),
        ..resources::RestoreOptions::default()
    };
    let (archive, outcomes) = prompts::restore_at(
        &prepared.resource_paths,
        std::path::Path::new(archive_path),
        &options,
    )
    .map_err(|error| error.to_string())?;
    prepared
        .reload_templates()
        .map_err(|error| format!("restored prompts, but could not reload them: {error}"))?;

    let mut lines = archive
        .warnings
        .into_iter()
        .map(|warning| format!("warning: {warning}"))
        .collect::<Vec<_>>();
    if !archive.manifest.tool.is_empty() && archive.manifest.tool != "goshcoder" {
        lines.push(format!(
            "note: this archive was written by {}",
            archive.manifest.tool
        ));
    }
    lines.extend(prompts::describe_restore(&outcomes));
    Ok(lines.join("\n"))
}

fn parse_prompt_scope(argument: &str) -> (String, resources::PromptScope) {
    let mut scope = resources::PromptScope::User;
    let mut name = String::new();
    for field in argument.split_whitespace() {
        match field {
            "--project" | "-project" => scope = resources::PromptScope::Workspace,
            "--user" | "-user" => scope = resources::PromptScope::User,
            _ if name.is_empty() => name = field.to_owned(),
            _ => {}
        }
    }
    (name, scope)
}

fn reserved_prompt_names(resources: &resources::ResourceSet) -> Vec<String> {
    let mut names = [
        "exit",
        "quit",
        "help",
        "?",
        "model",
        "login",
        "logout",
        "omni",
        "aperture",
        "aperture:onboarding",
        "aperture:settings",
        "btw",
        "thinking",
        "system",
        "tools",
        "messages",
        "status",
        "sidebar",
        "session",
        "tree",
        "fork",
        "label",
        "clone",
        "export",
        "import",
        "prompt",
        "prompts",
        "sessions",
        "resume",
        "name",
        "hotkeys",
        "steer",
        "followup",
        "queue",
        "clear",
        "new",
        "compact",
        "reload",
        "resources",
        "ralph",
        "planner",
        "plannator",
        "plannotator",
        "planner-review",
        "plannotator-review",
        "planner-annotate",
        "plannotator-annotate",
        "planner-last",
        "plannotator-last",
        "use-claude-code-tui",
        "use-default-tui",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    names.extend(
        resources
            .skills
            .iter()
            .map(|skill| format!("skill:{}", skill.name)),
    );
    names.into_iter().collect()
}

fn last_user_prompt(messages: &[llm::Message]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        let llm::Message::User(user) = message else {
            return None;
        };
        let text = user_message_text(user).trim().to_owned();
        (!text.is_empty()).then_some(text)
    })
}

fn list_interactive_sessions(prepared: &runtime::PreparedSession) -> Result<String, String> {
    let cwd =
        runtime::absolute_workdir(&prepared.config.workdir).map_err(|error| error.to_string())?;
    let store = sessionlog::Store::new(
        prepared
            .config
            .sessions_dir
            .clone()
            .unwrap_or_else(config::sessions_dir),
    );
    let sessions = session_picker::list_sessions_for_picker(&store, &cwd)
        .map_err(|error| error.to_string())?;
    Ok(render_interactive_session_list(&sessions))
}

fn refresh_resume_palette(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
) {
    if !app.resume_palette_active() {
        app.invalidate_resume_suggestions();
        return;
    }

    let completed_scan = view
        .resume_scan
        .as_ref()
        .and_then(|receiver| match receiver.try_recv() {
            Ok(result) => Some(result),
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                Some(Err("saved-session scan stopped unexpectedly".to_owned()))
            }
        });
    if let Some(result) = completed_scan {
        view.resume_scan = None;
        match result {
            Ok(sessions) => view.resume_sessions = Some(sessions),
            Err(error) => {
                view.resume_sessions = Some(Vec::new());
                append_view_message(
                    view,
                    MessageRole::Error,
                    format!("could not read saved sessions: {error}"),
                );
            }
        }
    }

    if view.resume_sessions.is_none() {
        if view.resume_scan.is_none() {
            let cwd = match runtime::absolute_workdir(&prepared.config.workdir) {
                Ok(cwd) => cwd,
                Err(error) => {
                    view.resume_sessions = Some(Vec::new());
                    append_view_message(view, MessageRole::Error, error.to_string());
                    return;
                }
            };
            let store = sessionlog::Store::new(
                prepared
                    .config
                    .sessions_dir
                    .clone()
                    .unwrap_or_else(config::sessions_dir),
            );
            let (sender, receiver) = mpsc::sync_channel(1);
            thread::spawn(move || {
                let result = session_picker::list_sessions_for_picker(&store, &cwd)
                    .map_err(|error| error.to_string());
                let _ = sender.send(result);
            });
            view.resume_scan = Some(receiver);
        }
        app.set_resume_suggestions_loading();
        return;
    }

    let current_path = prepared.runtime.path();
    let sessions = view.resume_sessions.as_deref().unwrap_or_default();
    app.refresh_resume_suggestions(|query| {
        resume_picker_suggestions(sessions, query, current_path.as_deref())
    });
}

fn refresh_sidebar_git(view: &mut InteractiveView, workspace: &Path) {
    if view.git_status_root.as_deref() != Some(workspace) {
        view.git_status = None;
        view.git_status_root = Some(workspace.to_path_buf());
        view.git_status_scan = None;
        view.git_status_last_requested = None;
    }

    let completed_scan =
        view.git_status_scan
            .as_ref()
            .and_then(|receiver| match receiver.try_recv() {
                Ok(result) => Some(Some(result)),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => Some(None),
            });
    if let Some(result) = completed_scan {
        view.git_status_scan = None;
        if let Some(result) = result {
            view.git_status = Some(result);
        }
    }

    let refresh_due = view
        .git_status_last_requested
        .is_none_or(|requested| requested.elapsed() >= SIDEBAR_GIT_REFRESH_INTERVAL);
    if view.git_status_scan.is_none() && refresh_due {
        let workspace = workspace.to_path_buf();
        let (sender, receiver) = mpsc::sync_channel(1);
        thread::spawn(move || {
            let _ = sender.send(scan_sidebar_git(&workspace));
        });
        view.git_status_scan = Some(receiver);
        view.git_status_last_requested = Some(Instant::now());
    }
}

fn scan_sidebar_git(workspace: &Path) -> SidebarGitInfo {
    let mut command = Command::new("git");
    command
        .args(["status", "--short", "--branch", "--untracked-files=normal"])
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let Ok(mut child) = command.spawn() else {
        return SidebarGitInfo::default();
    };
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return SidebarGitInfo::default();
    };
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::spawn(move || {
        let mut reader = stdout;
        let mut output = Vec::with_capacity(SIDEBAR_GIT_MAX_OUTPUT_BYTES);
        let mut buffer = [0_u8; 8_192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) if output.len() < SIDEBAR_GIT_MAX_OUTPUT_BYTES => {
                    let remaining = SIDEBAR_GIT_MAX_OUTPUT_BYTES - output.len();
                    output.extend_from_slice(&buffer[..read.min(remaining)]);
                }
                Ok(_) => {}
            }
        }
        let _ = sender.send(output);
    });

    let deadline = Instant::now() + SIDEBAR_GIT_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                break child.wait().ok();
            }
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(_) => break None,
        }
    };
    let output = receiver
        .recv_timeout(Duration::from_millis(250))
        .unwrap_or_default();
    if !status.is_some_and(|status| status.success()) {
        return SidebarGitInfo::default();
    }
    parse_sidebar_git_status(&String::from_utf8_lossy(&output))
}

fn parse_sidebar_git_status(output: &str) -> SidebarGitInfo {
    let mut info = SidebarGitInfo::default();
    for (index, line) in output.lines().enumerate() {
        if index == 0
            && let Some(branch) = line.strip_prefix("## ")
        {
            let branch = branch.split_once("...").map_or(branch, |(name, _)| name);
            info.branch = Some(if branch.starts_with("HEAD ") {
                "detached HEAD".to_owned()
            } else {
                branch.to_owned()
            });
            continue;
        }
        let Some((status, path)) = parse_sidebar_git_change(line) else {
            continue;
        };
        info.changed_files += 1;
        if info.changes.len() >= SIDEBAR_GIT_MAX_CHANGES {
            continue;
        }
        info.changes.push(SidebarGitChange {
            status: state::FileStatus::Raw(status),
            path,
        });
    }
    info
}

fn parse_sidebar_git_change(line: &str) -> Option<(String, String)> {
    if line.len() < 3 {
        return None;
    }
    let status = line[..2].trim().to_owned();
    let path = line[3..]
        .trim()
        .split_once(" -> ")
        .map_or_else(|| line[3..].trim(), |(_, renamed)| renamed)
        .trim_matches('"')
        .replace('\\', "/");
    (!path.is_empty()).then_some((status, path))
}

fn resume_picker_suggestions(
    sessions: &[sessionlog::SessionInfo],
    query: &str,
    current_path: Option<&std::path::Path>,
) -> Vec<state::Suggestion> {
    let terms = query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let labels = sessionlog::short_ids(sessions);
    sessions
        .iter()
        .zip(labels)
        .filter(|(session, _)| {
            current_path.is_none_or(|current| session.path != current)
                && session_picker::matches_session(session, &terms)
        })
        .map(|(session, label)| {
            let (label, description) = session_picker::describe_session(session, &label, false);
            state::Suggestion {
                value: format!("/resume {label}"),
                label,
                description,
                execute: true,
            }
        })
        .collect()
}

fn render_interactive_session_list(sessions: &[sessionlog::SessionInfo]) -> String {
    if sessions.is_empty() {
        return "No saved sessions for this workspace.".to_owned();
    }
    const SHOWN: usize = 10;
    let labels = sessionlog::short_ids(sessions);
    let mut lines = Vec::with_capacity(SHOWN + 2);
    for (index, (session, label)) in sessions.iter().zip(labels).enumerate() {
        if index >= SHOWN {
            lines.push(format!(
                "… {} more · goshcoder sessions list",
                sessions.len() - SHOWN
            ));
            break;
        }
        let (label, description) = session_picker::describe_session(session, &label, false);
        lines.push(format!("{label}  {description}"));
    }
    lines.join("\n")
}

fn start_planner_review<F>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    turn_sender: Sender<InteractiveTaskResult>,
    activity: &str,
    request: F,
) -> CommandDispatch
where
    F: FnOnce() -> Result<(String, plannotator::ReviewRequest), String> + Send + 'static,
{
    if app.streaming || view.turn_pending || prepared.runtime.agent().state().is_streaming {
        append_view_message(
            view,
            MessageRole::Error,
            "Wait for the current response or Planner review to finish.",
        );
        return CommandDispatch::Handled;
    }
    let Some(planner) = prepared.planner.as_ref() else {
        append_view_message(
            view,
            MessageRole::Error,
            "Planner is unavailable in this session. Reopen chat with planner support enabled.",
        );
        return CommandDispatch::Handled;
    };
    let review = planner.review_handle();
    let agent = prepared.runtime.agent().clone();
    view.turn_pending = true;
    view.activity = activity.to_owned();
    view.activity_since = Some(Instant::now());
    thread::spawn(move || {
        let result = request().and_then(|(subject, request)| {
            let decision = review.review(&request).map_err(|error| error.to_string())?;
            if let Some(feedback) = plannotator::review_feedback_prompt(&subject, &decision) {
                agent.prompt(feedback).map_err(|error| error.to_string())
            } else {
                review.notify(format!("{subject} approved"));
                Ok(())
            }
        });
        let _ = turn_sender.send(InteractiveTaskResult::finished(result));
    });
    CommandDispatch::Handled
}

fn append_view_message(view: &mut InteractiveView, role: MessageRole, text: impl Into<String>) {
    view.notices.push(Message {
        role,
        text: text.into(),
        is_error: role == MessageRole::Error,
        ..Message::default()
    });
    const MAX_NOTICES: usize = 20;
    if view.notices.len() > MAX_NOTICES {
        view.notices.drain(..view.notices.len() - MAX_NOTICES);
    }
}

fn refresh_runtime_app(
    app: &mut App,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    view: &mut InteractiveView,
) {
    let state = prepared.runtime.agent().state();
    let model_reference = format!("{}/{}", state.model.provider, state.model.id);
    app.set_command_availability(prepared.ralph.is_some(), prepared.planner.is_some());
    app.refresh_model_suggestions(&model_reference, |query| {
        model_picker_suggestions(catalog, &state.model, query)
    });
    app.refresh_login_suggestions(|query| login_provider_suggestions(catalog, query));
    app.refresh_thinking_suggestions(&model_reference, &state.thinking_level, |query| {
        thinking_picker_suggestions(&state.model, &state.thinking_level, query)
    });
    app.refresh_resource_suggestions(|query| {
        resource_palette_suggestions(&prepared.resources(), query)
    });
    refresh_resume_palette(app, view, prepared);
    let mut messages = agent_messages(&state.messages);
    if let Some(message) = state.streaming_message.as_ref() {
        messages.extend(agent_messages(std::slice::from_ref(message)));
    }
    messages.extend(view.notices.clone());
    app.replace_messages(messages);
    app.streaming = state.is_streaming || view.turn_pending;
    app.set_recording_active(prepared.runtime.recording());
    app.title = format!(
        "v{}  ·  {}/{}",
        ui_version(),
        state.model.provider,
        state.model.id
    );
    app.status = interactive_status(view, app.streaming);
    refresh_sidebar_git(view, &sidebar_workspace_path(prepared));
    app.sidebar = runtime_sidebar(prepared, &state, view);
}

fn interactive_status(view: &InteractiveView, busy: bool) -> String {
    if !busy {
        return view.activity.clone();
    }
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
    let elapsed = view
        .activity_since
        .map(|started| started.elapsed())
        .unwrap_or_default();
    let index = (elapsed.as_millis() / 120) as usize % spinner.len();
    format!(
        "{}  {}  {:.1}s",
        spinner[index],
        view.activity,
        elapsed.as_secs_f32()
    )
}

fn agent_messages(messages: &[llm::Message]) -> Vec<Message> {
    let mut tool_results = BTreeMap::<String, &llm::ToolResultMessage>::new();
    let mut unnamed_results = BTreeMap::<String, Vec<&llm::ToolResultMessage>>::new();
    for message in messages {
        if let llm::Message::ToolResult(result) = message {
            if result.tool_call_id.is_empty() {
                unnamed_results
                    .entry(result.tool_name.clone())
                    .or_default()
                    .push(result);
            } else {
                tool_results.insert(result.tool_call_id.clone(), result);
            }
        }
    }

    let mut matched_results = BTreeSet::new();
    let mut unnamed_positions = BTreeMap::<String, usize>::new();
    let mut result = Vec::new();
    for message in messages {
        match message {
            llm::Message::User(user) => {
                if compaction::is_summary_message(message) {
                    continue;
                }
                result.push(Message {
                    role: MessageRole::User,
                    text: user_message_text(user),
                    ..Message::default()
                });
            }
            llm::Message::Assistant(assistant) => {
                let mut thinking = String::new();
                let mut text = String::new();
                for content in &assistant.content {
                    match content {
                        llm::ContentBlock::Thinking(content) => {
                            thinking.push_str(&content.thinking)
                        }
                        llm::ContentBlock::Text(content) => text.push_str(&content.text),
                        llm::ContentBlock::Image(_) | llm::ContentBlock::ToolCall(_) => {}
                    }
                }
                if !thinking.is_empty() {
                    result.push(Message {
                        role: MessageRole::Thinking,
                        text: thinking,
                        ..Message::default()
                    });
                }
                if !text.is_empty() {
                    result.push(Message {
                        role: MessageRole::Assistant,
                        text,
                        ..Message::default()
                    });
                }
                if !assistant.error_message.is_empty() {
                    result.push(Message {
                        role: MessageRole::Error,
                        text: assistant.error_message.clone(),
                        is_error: true,
                        ..Message::default()
                    });
                }
                for content in &assistant.content {
                    let llm::ContentBlock::ToolCall(call) = content else {
                        continue;
                    };
                    let matched = if call.id.is_empty() {
                        let position = unnamed_positions.entry(call.name.clone()).or_default();
                        let selected = unnamed_results
                            .get(&call.name)
                            .and_then(|results| results.get(*position).copied());
                        if selected.is_some() {
                            *position += 1;
                        }
                        selected
                    } else {
                        tool_results.get(&call.id).copied()
                    };
                    if let Some(tool_result) = matched {
                        matched_results.insert(tool_result.tool_call_id.clone());
                    }
                    result.push(tool_view_message(call, matched));
                }
            }
            llm::Message::ToolResult(tool_result) => {
                if !tool_result.tool_call_id.is_empty()
                    && matched_results.contains(&tool_result.tool_call_id)
                {
                    continue;
                }
                result.push(unmatched_tool_view_message(tool_result));
            }
        }
    }
    result
}

fn user_message_text(message: &llm::UserMessage) -> String {
    match &message.content {
        llm::UserContent::Text(text) => text.clone(),
        llm::UserContent::Blocks(blocks) => content_text(blocks),
    }
}

fn content_text(content: &[llm::ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text(text) => Some(text.text.as_str()),
            llm::ContentBlock::Thinking(thinking) => Some(thinking.thinking.as_str()),
            llm::ContentBlock::Image(_) | llm::ContentBlock::ToolCall(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_view_message(call: &llm::ToolCall, result: Option<&llm::ToolResultMessage>) -> Message {
    let detail = result.map_or_else(String::new, |result| content_text(&result.content));
    let is_error = result.is_some_and(|result| result.is_error);
    let text = match result {
        None => "running…".to_owned(),
        Some(_) if is_error => first_line(&detail),
        Some(_)
            if matches!(
                call.name.as_str(),
                "bash" | "grep" | "find" | "ls" | "list" | "planner_submit_plan" | "web_search"
            ) =>
        {
            detail.lines().take(3).collect::<Vec<_>>().join("\n")
        }
        Some(_) => first_line(&detail),
    };
    Message {
        role: MessageRole::Tool,
        title: tool_title(call),
        text,
        detail,
        is_error,
    }
}

fn unmatched_tool_view_message(result: &llm::ToolResultMessage) -> Message {
    let detail = content_text(&result.content);
    Message {
        role: MessageRole::Tool,
        title: result.tool_name.clone(),
        text: first_line(&detail),
        detail,
        is_error: result.is_error,
    }
}

fn tool_title(call: &llm::ToolCall) -> String {
    let argument = |name: &str| {
        call.arguments
            .get(name)
            .map(|value| {
                value
                    .as_str()
                    .map_or_else(|| value.to_string(), str::to_owned)
            })
            .unwrap_or_default()
    };
    match call.name.as_str() {
        "read" | "write" | "edit" | "ls" | "list" => {
            let path = argument("path");
            if path.is_empty() {
                call.name.clone()
            } else {
                format!("{} {path}", call.name)
            }
        }
        "grep" => {
            let pattern = argument("pattern");
            let path = argument("path");
            if path.is_empty() {
                format!("grep /{pattern}/")
            } else {
                format!("grep /{pattern}/ in {path}")
            }
        }
        "find" => format!("find {}", argument("pattern")),
        "bash" => format!("bash {}", first_line(&argument("command"))),
        "planner_submit_plan" => format!("submit plan {}", argument("filePath")),
        _ => {
            let arguments = summarize_tool_arguments(&call.arguments);
            if arguments.is_empty() {
                call.name.clone()
            } else {
                format!("{} {arguments}", call.name)
            }
        }
    }
}

fn runtime_sidebar(
    prepared: &runtime::PreparedSession,
    state: &agent::State,
    view: &mut InteractiveView,
) -> Vec<state::SidebarLine> {
    let context_tokens = view.context_estimate.measure(&state.messages);
    let limit = state.model.context_window;
    let percent = if limit == 0 {
        0
    } else {
        context_tokens
            .saturating_mul(100)
            .saturating_div(limit)
            .min(100) as u8
    };
    let cost = compaction::conversation_cost(&state.messages, &state.compactions);
    let name = prepared
        .runtime
        .name()
        .or_else(|| prepared.runtime.title())
        .unwrap_or_else(|| "New Session".to_owned());
    let storage = if prepared.runtime.recording() {
        prepared.runtime.id().map_or_else(
            || "recording".to_owned(),
            |id| format!("recording {}", short_id(&id)),
        )
    } else if prepared.runtime.read_only() {
        "read-only session".to_owned()
    } else {
        "not recording".to_owned()
    };
    let cwd = sidebar_workspace_path(prepared).display().to_string();
    let planner_state = prepared
        .planner
        .as_ref()
        .map(|planner| planner.manager().state());
    let (mode, todo_items) = planner_sidebar_details(planner_state);
    let mut lines = vec![
        state::SidebarLine::title(name),
        state::SidebarLine::accent(format!("{}/{}", state.model.provider, state.model.id)),
        state::SidebarLine::meta(format!("{} thinking · {mode}", state.thinking_level)),
        state::SidebarLine::meta(storage),
        state::SidebarLine::blank(),
        state::SidebarLine::section("Context"),
        state::SidebarLine::progress(percent),
        state::SidebarLine::meta(if limit == 0 {
            format!("{} tokens", compact_number(context_tokens))
        } else {
            format!(
                "{} / {} tokens",
                compact_number(context_tokens),
                compact_number(limit)
            )
        }),
        state::SidebarLine::meta(format!("{percent}% used · ${cost:.4} spent")),
    ];
    if state.is_streaming || view.turn_pending || !state.pending_tool_calls.is_empty() {
        lines.extend([
            state::SidebarLine::blank(),
            state::SidebarLine::section("Activity"),
            state::SidebarLine {
                kind: state::SidebarKind::Active,
                value: view.activity.clone(),
            },
        ]);
        if !state.pending_tool_calls.is_empty() {
            lines.push(state::SidebarLine::meta(format!(
                "{} tool{} running",
                state.pending_tool_calls.len(),
                if state.pending_tool_calls.len() == 1 {
                    ""
                } else {
                    "s"
                }
            )));
        }
        if !view.recent_tool.is_empty() {
            lines.push(state::SidebarLine::meta(view.recent_tool.clone()));
        }
    }
    if !todo_items.is_empty() {
        lines.extend([
            state::SidebarLine::blank(),
            state::SidebarLine::section("Todo"),
        ]);
        lines.extend(
            todo_items
                .into_iter()
                .map(|item| state::SidebarLine::todo(item.completed, item.text)),
        );
    }
    if let Some(git) = view.git_status.as_ref()
        && !git.changes.is_empty()
    {
        lines.extend([
            state::SidebarLine::blank(),
            state::SidebarLine::section("Modified Files"),
        ]);
        for (index, change) in git.changes.iter().enumerate() {
            if index >= 10 {
                lines.push(state::SidebarLine::meta(format!(
                    "… {} more files",
                    git.changes.len() - index
                )));
                break;
            }
            lines.push(state::SidebarLine::file(
                change.status.clone(),
                change.path.clone(),
            ));
        }
    }
    let branch = view
        .git_status
        .as_ref()
        .and_then(|git| git.branch.as_deref())
        .unwrap_or("not a git repo");
    lines.extend([
        state::SidebarLine::blank(),
        state::SidebarLine::section("Workspace"),
        state::SidebarLine::meta(branch),
        state::SidebarLine::path(cwd),
        state::SidebarLine::blank(),
        state::SidebarLine::brand(format!("● GoshCoder v{}", ui_version())),
    ]);
    lines
}

fn sidebar_workspace_path(prepared: &runtime::PreparedSession) -> PathBuf {
    prepared
        .workspace
        .as_ref()
        .map(|workspace| workspace.root().to_path_buf())
        .unwrap_or_else(|| prepared.config.workdir.clone())
}

fn planner_sidebar_details(
    planner_state: Option<plannotator::State>,
) -> (String, Vec<plannotator::ChecklistItem>) {
    let Some(planner_state) = planner_state else {
        return ("normal".to_owned(), Vec::new());
    };
    let mode = match planner_state.phase {
        plannotator::Phase::Planning => "planning",
        plannotator::Phase::Executing => "executing",
        plannotator::Phase::Idle | plannotator::Phase::Unknown(_) => "normal",
    };
    (mode.to_owned(), planner_state.items)
}

fn session_status(prepared: &runtime::PreparedSession, activity: &str) -> String {
    let state = prepared.runtime.agent().state();
    let context_tokens = compaction::measured_context_tokens(&state.messages);
    let storage = if prepared.runtime.recording() {
        prepared.runtime.id().map_or_else(
            || "recording".to_owned(),
            |id| format!("recording {}", short_id(&id)),
        )
    } else if prepared.runtime.read_only() {
        "read-only".to_owned()
    } else {
        "not recording".to_owned()
    };
    let context_limit = state.model.context_window;
    let context = if context_limit == 0 {
        format!("{} tokens", compact_number(context_tokens))
    } else {
        format!(
            "{} / {} tokens",
            compact_number(context_tokens),
            compact_number(context_limit)
        )
    };
    let planner = prepared.planner.as_ref().map_or_else(
        || "Planner: unavailable".to_owned(),
        planner_runtime::PlannerRuntime::status_line,
    );
    let ralph = match prepared.ralph.as_ref() {
        Some(ralph_runtime) => match ralph_runtime.current() {
            Ok(Some(state)) => format!("Ralph: {}", state.summary()),
            Ok(None) => "Ralph: no active loop".to_owned(),
            Err(error) => format!("Ralph: unavailable ({error})"),
        },
        None => "Ralph: disabled".to_owned(),
    };
    format!(
        "Session: {}\nModel: {}/{}\nThinking: {}\n{planner}\n{ralph}\nContext: {context}\nActivity: {activity}\nStorage: {storage}",
        prepared
            .runtime
            .id()
            .map_or_else(|| "temporary".to_owned(), |id| short_id(&id).to_owned()),
        state.model.provider,
        state.model.id,
        state.thinking_level,
    )
}

fn configured_model_references(catalog: &catalog::Catalog) -> Vec<String> {
    let configured = catalog
        .configured_provider_ids()
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeSet<_>>();
    catalog
        .providers()
        .into_iter()
        .filter(|provider| configured.contains(&provider.id))
        .flat_map(|provider| {
            provider.models().into_iter().map(move |model| {
                let reference = format!("{}/{}", provider.id, model.id);
                if providers::ProviderProtocol::from_api(&model.api).is_ok() {
                    reference
                } else {
                    format!("{reference}\n  [protocol not yet migrated]")
                }
            })
        })
        .collect()
}

fn model_picker_suggestions(
    catalog: &catalog::Catalog,
    active_model: &llm::Model,
    query: &str,
) -> Vec<state::Suggestion> {
    let active_reference = format!("{}/{}", active_model.provider, active_model.id);
    let mut models = interactive_models(catalog).unwrap_or_default();
    if !models
        .iter()
        .any(|model| format!("{}/{}", model.provider, model.id) == active_reference)
    {
        models.push(active_model.clone());
    }
    let mut seen = BTreeSet::new();
    models.retain(|model| seen.insert(format!("{}/{}", model.provider, model.id)));

    let terms = query
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    let mut choices = models
        .into_iter()
        .filter_map(|model| {
            let reference = format!("{}/{}", model.provider, model.id);
            let label = if model.name.is_empty() {
                model.id.clone()
            } else {
                model.name.clone()
            };
            let haystack = format!(
                "{} {} {}",
                reference,
                model.provider.to_ascii_lowercase(),
                label.to_ascii_lowercase()
            );
            if !terms.iter().all(|term| haystack.contains(term)) {
                return None;
            }
            let current = reference == active_reference;
            let mut details = vec![reference.clone()];
            if model.context_window > 0 {
                details.push(format!("{} context", compact_number(model.context_window)));
            }
            if model.reasoning {
                details.push("reasoning".to_owned());
            }
            let description = details.join(" · ");
            Some((
                current,
                model.provider.clone(),
                label.clone(),
                state::Suggestion {
                    label,
                    description: if current {
                        format!("CURRENT · {description}")
                    } else {
                        description
                    },
                    value: format!("/model {reference}"),
                    execute: true,
                },
            ))
        })
        .collect::<Vec<_>>();
    choices.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    choices
        .into_iter()
        .map(|(_, _, _, suggestion)| suggestion)
        .collect()
}

fn resource_palette_suggestions(
    resources: &resources::ResourceSet,
    query: &str,
) -> Vec<state::Suggestion> {
    let query = query.to_ascii_lowercase();
    let mut suggestions = Vec::new();
    for template in &resources.templates {
        let label = format!("/{}", template.name);
        if !label.to_ascii_lowercase().starts_with(&query) {
            continue;
        }
        let description = if template.argument_hint.is_empty() {
            template.description.clone()
        } else if template.description.is_empty() {
            template.argument_hint.clone()
        } else {
            format!("{} · {}", template.argument_hint, template.description)
        };
        suggestions.push(state::Suggestion {
            label: label.clone(),
            description,
            value: label,
            execute: true,
        });
    }
    for skill in &resources.skills {
        let label = format!("/skill:{}", skill.name);
        if !label.to_ascii_lowercase().starts_with(&query) {
            continue;
        }
        suggestions.push(state::Suggestion {
            label: label.clone(),
            description: skill.description.clone(),
            value: label,
            execute: true,
        });
    }
    suggestions
}

fn thinking_picker_suggestions(
    model: &llm::Model,
    current_level: &str,
    query: &str,
) -> Vec<state::Suggestion> {
    let query = query.to_ascii_lowercase();
    stream::supported_thinking_levels(model)
        .into_iter()
        .filter(|level| level.starts_with(&query))
        .map(|level| {
            let description = thinking_level_description(&level);
            state::Suggestion {
                label: level.clone(),
                description: if level == current_level {
                    format!("CURRENT · {description}")
                } else {
                    description.to_owned()
                },
                value: format!("/thinking {level}"),
                execute: true,
            }
        })
        .collect()
}

fn thinking_level_description(level: &str) -> &'static str {
    match level {
        "off" => "Fastest responses, no extra reasoning",
        "minimal" => "Very brief reasoning",
        "low" => "Quick tasks and small edits",
        "medium" => "Balanced depth and speed",
        "high" => "Complex implementation work",
        "xhigh" => "Deep analysis for difficult problems",
        "max" => "Maximum available reasoning",
        _ => "Reasoning effort",
    }
}

fn login_provider_suggestions(catalog: &catalog::Catalog, query: &str) -> Vec<state::Suggestion> {
    let configured = catalog
        .configured_provider_ids()
        .unwrap_or_default()
        .into_iter()
        .collect::<BTreeSet<_>>();
    let query = query.to_ascii_lowercase();
    let mut choices = catalog
        .providers()
        .into_iter()
        .filter(|provider| {
            provider
                .models()
                .iter()
                .any(|model| providers::ProviderProtocol::from_api(&model.api).is_ok())
        })
        .filter(|provider| {
            query.is_empty()
                || provider.id.to_ascii_lowercase().contains(&query)
                || provider.name.to_ascii_lowercase().contains(&query)
        })
        .map(|provider| {
            let oauth = oauth::OAuthProviderId::parse(&provider.id).is_some_and(|provider_id| {
                oauth::metadata_for(provider_id).flow_support
                    == oauth::OAuthFlowSupport::Implemented
            });
            let method = if oauth {
                "OAuth / subscription"
            } else {
                "API key"
            };
            let signed_in = configured.contains(&provider.id);
            let prefix = if signed_in { "SIGNED IN · " } else { "" };
            (
                oauth,
                provider.id.clone(),
                state::Suggestion {
                    label: provider.id.clone(),
                    description: format!("{prefix}{} · {method}", provider.name),
                    value: format!("/login {}", provider.id),
                    execute: true,
                },
            )
        })
        .collect::<Vec<_>>();
    choices.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    choices
        .into_iter()
        .map(|(_, _, suggestion)| suggestion)
        .collect()
}

fn interactive_models(catalog: &catalog::Catalog) -> Result<Vec<llm::Model>, String> {
    let configured = catalog
        .configured_provider_ids()
        .map_err(|error| error.to_string())?
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut models = catalog
        .providers()
        .into_iter()
        .filter(|provider| configured.contains(&provider.id))
        .flat_map(|provider| provider.models())
        .filter(|model| providers::ProviderProtocol::from_api(&model.api).is_ok())
        .collect::<Vec<_>>();
    models.sort_by(|left, right| (&left.provider, &left.id).cmp(&(&right.provider, &right.id)));
    Ok(models)
}

fn cycle_interactive_model(
    runtime: &session::SessionRuntime,
    catalog: &catalog::Catalog,
    direction: i8,
) -> Result<String, String> {
    let models = interactive_models(catalog)?;
    if models.is_empty() {
        return Err(
            "No authenticated model with a migrated provider protocol is available.".to_owned(),
        );
    }
    let state = runtime.agent().state();
    let current = models
        .iter()
        .position(|model| model.provider == state.model.provider && model.id == state.model.id);
    let next = match (current, direction < 0) {
        (Some(index), true) => (index + models.len() - 1) % models.len(),
        (Some(index), false) => (index + 1) % models.len(),
        (None, _) => 0,
    };
    let model = &models[next];
    let reference = format!("{}/{}", model.provider, model.id);
    runtime::set_model(runtime, catalog, &reference).map_err(|error| error.to_string())?;
    Ok(reference)
}

fn cycle_interactive_thinking(runtime: &session::SessionRuntime) -> Option<String> {
    let state = runtime.agent().state();
    let levels = stream::supported_thinking_levels(&state.model);
    if levels.len() <= 1 {
        return None;
    }
    let current = levels
        .iter()
        .position(|level| level == &state.thinking_level)
        .unwrap_or(levels.len() - 1);
    let next = levels[(current + 1) % levels.len()].clone();
    runtime.agent().set_thinking_level(next.clone());
    Some(next)
}

fn parse_branch_index(value: &str) -> Result<usize, String> {
    let index = value
        .trim()
        .parse::<usize>()
        .map_err(|_| "usage: /fork <positive branch point>".to_owned())?;
    if index == 0 {
        return Err("branch point must be at least 1".to_owned());
    }
    Ok(index)
}

fn compact_number(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}k", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn short_id(value: &str) -> &str {
    let mut end = value.len().min(8);
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn event(kind: agent::EventKind) -> agent::Event {
        agent::Event {
            kind,
            message: None,
            assistant_event: None,
            assistant_was_streamed: false,
            messages: Vec::new(),
            kept: Vec::new(),
            compaction: None,
            tool_call_id: String::new(),
            tool_name: String::new(),
            arguments: BTreeMap::new(),
            result: None,
            is_error: false,
            provider: String::new(),
            model_id: String::new(),
            thinking_level: String::new(),
            reason: String::new(),
        }
    }

    #[test]
    fn help_and_version_are_non_interactive() {
        assert!(USAGE.contains("Ratatui"));
        assert!(env!("CARGO_PKG_VERSION").starts_with("0."));
    }

    #[test]
    fn login_help_lists_supported_oauth_and_additive_api_key_flow() {
        let help = login_command_help();

        assert!(help.contains("anthropic"));
        assert!(help.contains("API-key providers prompt for a key"));
        assert!(help.contains("existing logins are kept"));
    }

    #[test]
    fn onboarding_selection_matches_the_subscription_login_choices() {
        assert_eq!(
            onboarding_provider("").expect("default choice"),
            "openai-codex"
        );
        assert_eq!(
            onboarding_provider("1").expect("first choice"),
            "openai-codex"
        );
        assert_eq!(
            onboarding_provider(" 2 ").expect("second choice"),
            "anthropic"
        );
        assert_eq!(
            onboarding_provider("3").expect("third choice"),
            "kimi-coding"
        );
        assert!(onboarding_provider("4").is_err());
    }

    #[test]
    fn continue_hint_only_promises_a_durable_session() {
        assert_eq!(
            continue_session_hint(true, Some("12345678-1234-7000-8000-000000000000")),
            Some("session 12345678 · resume with: goshcoder chat -continue".to_owned())
        );
        assert_eq!(continue_session_hint(false, Some("saved")), None);
        assert_eq!(continue_session_hint(true, None), None);
    }

    #[test]
    fn model_reference_matches_the_persisted_chat_default_format() {
        assert_eq!(
            model_reference(&llm::Model {
                provider: "anthropic".to_owned(),
                id: "claude".to_owned(),
                ..llm::Model::default()
            }),
            "anthropic/claude"
        );
    }

    #[test]
    fn fullscreen_terminal_commands_preserve_safe_cli_invocations() {
        assert_eq!(
            FullscreenTerminalCommand::Login {
                provider_id: "anthropic".to_owned(),
                command: provider_cli::InteractiveAuthCommand::Login,
            }
            .arguments(),
            vec!["auth", "login", "anthropic"]
        );
        assert_eq!(
            FullscreenTerminalCommand::OmniSetup.arguments(),
            vec!["omni", "setup"]
        );
        assert_eq!(
            FullscreenTerminalCommand::ApertureOnboarding.arguments(),
            vec!["aperture", "onboarding"]
        );
        assert!(is_omni_setup(" SETUP "));
        assert!(is_aperture_onboarding("configure"));
        assert!(!is_aperture_onboarding("settings"));
    }

    #[test]
    fn login_picker_prioritizes_oauth_and_marks_signed_in_providers() {
        let credentials = Arc::new(catalog::CredentialStore::in_memory());
        credentials
            .put("openai", catalog::Credential::api_key("test-key"))
            .expect("store credential");
        let aperture_root =
            std::env::temp_dir().join(format!("goshcoder-login-picker-{}", std::process::id()));
        let catalog =
            catalog::Catalog::with_environment(Some(Arc::clone(&credentials)), Arc::new(|_| None))
                .expect("catalog")
                .with_aperture_paths(
                    aperture_root.join("aperture.json"),
                    aperture_root.join("aperture-cache.json"),
                );

        let choices = login_provider_suggestions(&catalog, "");
        let anthropic = choices
            .iter()
            .find(|choice| choice.value == "/login anthropic")
            .expect("Anthropic choice");
        let openai = choices
            .iter()
            .find(|choice| choice.value == "/login openai")
            .expect("OpenAI choice");

        assert!(anthropic.description.contains("OAuth / subscription"));
        assert!(openai.description.contains("SIGNED IN ·"));
        assert!(openai.description.contains("API key"));
        assert!(
            choices
                .iter()
                .position(|choice| choice.value == "/login anthropic")
                .expect("Anthropic position")
                < choices
                    .iter()
                    .position(|choice| choice.value == "/login openai")
                    .expect("OpenAI position")
        );
        assert_eq!(
            login_provider_suggestions(&catalog, "anth").len(),
            1,
            "query should filter provider choices"
        );
    }

    #[test]
    fn model_picker_filters_models_and_keeps_an_active_unconfigured_model() {
        let credentials = Arc::new(catalog::CredentialStore::in_memory());
        credentials
            .put("openai", catalog::Credential::api_key("test-key"))
            .expect("store credential");
        let aperture_root =
            std::env::temp_dir().join(format!("goshcoder-model-picker-{}", std::process::id()));
        let catalog =
            catalog::Catalog::with_environment(Some(Arc::clone(&credentials)), Arc::new(|_| None))
                .expect("catalog")
                .with_aperture_paths(
                    aperture_root.join("aperture.json"),
                    aperture_root.join("aperture-cache.json"),
                );
        let active = llm::Model {
            id: "detached".to_owned(),
            name: "Detached Model".to_owned(),
            provider: "local".to_owned(),
            context_window: 8_192,
            reasoning: true,
            ..llm::Model::default()
        };

        let choices = model_picker_suggestions(&catalog, &active, "");

        assert_eq!(choices[0].value, "/model local/detached");
        assert!(choices[0].description.contains("CURRENT"));
        assert!(choices[0].description.contains("8.2k context"));
        assert!(
            choices
                .iter()
                .any(|choice| choice.value.starts_with("/model openai/"))
        );
        assert_eq!(
            model_picker_suggestions(&catalog, &active, "detached")
                .into_iter()
                .map(|choice| choice.value)
                .collect::<Vec<_>>(),
            vec!["/model local/detached"]
        );
    }

    #[test]
    fn resource_picker_includes_templates_and_skills_with_native_descriptions() {
        let resources = resources::ResourceSet {
            templates: vec![resources::Template {
                name: "review".to_owned(),
                description: "Review a diff".to_owned(),
                argument_hint: "<path>".to_owned(),
                path: std::path::PathBuf::from("/tmp/review.md"),
                body: String::new(),
            }],
            skills: vec![resources::Skill {
                name: "deploy".to_owned(),
                description: "Deploy safely".to_owned(),
                path: std::path::PathBuf::from("/tmp/deploy/SKILL.md"),
                body: String::new(),
                disable_model_invocation: false,
            }],
            ..resources::ResourceSet::default()
        };

        assert_eq!(
            resource_palette_suggestions(&resources, "/rev"),
            vec![state::Suggestion {
                label: "/review".to_owned(),
                description: "<path> · Review a diff".to_owned(),
                value: "/review".to_owned(),
                execute: true,
            }]
        );
        assert_eq!(
            resource_palette_suggestions(&resources, "/skill"),
            vec![state::Suggestion {
                label: "/skill:deploy".to_owned(),
                description: "Deploy safely".to_owned(),
                value: "/skill:deploy".to_owned(),
                execute: true,
            }]
        );
    }

    #[test]
    fn thinking_picker_marks_the_current_supported_level() {
        let model = llm::Model {
            reasoning: true,
            ..llm::Model::default()
        };

        let choices = thinking_picker_suggestions(&model, "high", "h");

        assert_eq!(
            choices,
            vec![state::Suggestion {
                label: "high".to_owned(),
                description: "CURRENT · Complex implementation work".to_owned(),
                value: "/thinking high".to_owned(),
                execute: true,
            }]
        );
        assert_eq!(
            thinking_level_description("xhigh"),
            "Deep analysis for difficult problems"
        );
    }

    #[test]
    fn resume_picker_searches_transcript_and_hides_the_active_session() {
        let active_path = std::path::PathBuf::from("/sessions/active.jsonl");
        let sessions = vec![
            sessionlog::SessionInfo {
                id: "aaaa1111-0000-7000-8000-000000000001".to_owned(),
                path: active_path.clone(),
                cwd: "/workspace".to_owned(),
                name: "Active work".to_owned(),
                first_message: "current".to_owned(),
                created: None,
                modified: UNIX_EPOCH,
                messages: 2,
                cleared: 0,
                size: 0,
                search_text: "current implementation".to_owned(),
                locked: false,
                owner: sessionlog::LockOwner::default(),
            },
            sessionlog::SessionInfo {
                id: "bbbb2222-0000-7000-8000-000000000002".to_owned(),
                path: std::path::PathBuf::from("/sessions/older.jsonl"),
                cwd: "/workspace".to_owned(),
                name: "Streaming fix".to_owned(),
                first_message: "investigate".to_owned(),
                created: None,
                modified: UNIX_EPOCH,
                messages: 4,
                cleared: 0,
                size: 0,
                search_text: "retry streaming response".to_owned(),
                locked: false,
                owner: sessionlog::LockOwner::default(),
            },
        ];

        assert_eq!(
            resume_picker_suggestions(&sessions, "retry", Some(&active_path)),
            vec![state::Suggestion {
                label: "bbbb2222".to_owned(),
                description: "Jan 1 00:00 · 4 msg · Streaming fix".to_owned(),
                value: "/resume bbbb2222".to_owned(),
                execute: true,
            }]
        );
        assert!(
            resume_picker_suggestions(&sessions, "", Some(&active_path))
                .iter()
                .all(|suggestion| suggestion.value != "/resume aaaa1111")
        );
    }

    #[test]
    fn prompt_command_parsing_preserves_inline_text_and_scopes() {
        assert_eq!(
            split_prompt_action("save review   inspect the parser carefully"),
            ("save", "review   inspect the parser carefully")
        );
        assert_eq!(split_prompt_action("  list  "), ("list", ""));
        assert_eq!(
            parse_prompt_scope("review --project"),
            ("review".to_owned(), resources::PromptScope::Workspace)
        );
        assert_eq!(
            parse_prompt_scope("--user review"),
            ("review".to_owned(), resources::PromptScope::User)
        );
    }

    #[test]
    fn prompt_names_reserve_all_builtin_and_skill_commands() {
        let resources = resources::ResourceSet {
            skills: vec![resources::Skill {
                name: "deploy".to_owned(),
                description: "Deploy safely".to_owned(),
                path: std::path::PathBuf::from("/tmp/deploy/SKILL.md"),
                body: String::new(),
                disable_model_invocation: false,
            }],
            ..resources::ResourceSet::default()
        };
        let names = reserved_prompt_names(&resources);
        for name in [
            "model",
            "prompt",
            "aperture:onboarding",
            "plannotator-review",
            "skill:deploy",
        ] {
            assert!(names.iter().any(|reserved| reserved == name), "{name}");
        }
    }

    #[test]
    fn prompt_capture_uses_the_last_nonempty_user_message() {
        let messages = vec![
            llm::Message::User(llm::UserMessage::text("first", 1)),
            llm::Message::Assistant(Box::default()),
            llm::Message::User(llm::UserMessage::text("  final request  ", 2)),
        ];
        assert_eq!(
            last_user_prompt(&messages),
            Some("final request".to_owned())
        );
    }

    #[test]
    fn interactive_session_list_is_bounded_and_actionable() {
        let sessions = (0..12)
            .map(|index| sessionlog::SessionInfo {
                id: format!("session-{index:02}-0000-7000-8000-000000000000"),
                path: std::path::PathBuf::from(format!("/sessions/{index}.jsonl")),
                cwd: "/workspace".to_owned(),
                name: format!("session {index}"),
                first_message: String::new(),
                created: None,
                modified: UNIX_EPOCH,
                messages: index as usize + 1,
                cleared: 0,
                size: 0,
                search_text: String::new(),
                locked: false,
                owner: sessionlog::LockOwner::default(),
            })
            .collect::<Vec<_>>();

        let rendered = render_interactive_session_list(&sessions);

        assert!(rendered.contains("session 0"));
        assert!(rendered.contains("… 2 more · goshcoder sessions list"));
        assert!(!rendered.contains("session 10"));
    }

    #[test]
    fn live_submission_updates_history_without_placeholder_messages() {
        let mut app = App::new();
        let prior_messages = app.messages.clone();
        app.set_input("build this");

        app.record_submission("build this");

        assert_eq!(app.messages, prior_messages);
        assert_eq!(app.history, ["build this"]);
        assert!(app.input.is_empty());
    }

    #[test]
    fn run_renderer_keeps_assistant_text_pipeable() {
        let assistant = llm::AssistantMessage {
            content: vec![
                llm::ContentBlock::Thinking(llm::ThinkingContent {
                    thinking: "reasoning".to_owned(),
                    ..llm::ThinkingContent::default()
                }),
                llm::ContentBlock::text("answer"),
            ],
            usage: llm::Usage {
                input: 12,
                output: 4,
                total_tokens: 16,
                cost: llm::UsageCost {
                    total: 0.0123,
                    ..llm::UsageCost::default()
                },
                ..llm::Usage::default()
            },
            ..llm::AssistantMessage::default()
        };
        let mut completed = event(agent::EventKind::MessageEnd);
        completed.message = Some(llm::Message::Assistant(Box::new(assistant.clone())));
        completed.assistant_was_streamed = true;
        let mut ended = event(agent::EventKind::AgentEnd);
        ended.messages = vec![llm::Message::Assistant(Box::new(assistant))];
        let mut thinking_delta = event(agent::EventKind::MessageUpdate);
        thinking_delta.assistant_event = Some(stream::AssistantMessageEvent {
            event_type: stream::EVENT_THINKING_DELTA.to_owned(),
            delta: "reasoning".to_owned(),
            ..stream::AssistantMessageEvent::default()
        });
        let mut thinking_end = event(agent::EventKind::MessageUpdate);
        thinking_end.assistant_event = Some(stream::AssistantMessageEvent {
            event_type: stream::EVENT_THINKING_END.to_owned(),
            ..stream::AssistantMessageEvent::default()
        });
        let mut text_delta = event(agent::EventKind::MessageUpdate);
        text_delta.assistant_event = Some(stream::AssistantMessageEvent {
            event_type: stream::EVENT_TEXT_DELTA.to_owned(),
            delta: "answer".to_owned(),
            ..stream::AssistantMessageEvent::default()
        });

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        render_run_event(&thinking_delta, &mut stdout, &mut stderr, false)
            .expect("render thinking delta");
        render_run_event(&thinking_end, &mut stdout, &mut stderr, false)
            .expect("render thinking end");
        render_run_event(&text_delta, &mut stdout, &mut stderr, false).expect("render text delta");
        render_run_event(&completed, &mut stdout, &mut stderr, false).expect("render message");
        render_run_event(
            &event(agent::EventKind::TurnEnd),
            &mut stdout,
            &mut stderr,
            false,
        )
        .expect("render turn end");
        render_run_event(&ended, &mut stdout, &mut stderr, false).expect("render agent end");

        assert_eq!(String::from_utf8(stdout).expect("stdout"), "answer\n");
        let stderr = String::from_utf8(stderr).expect("stderr");
        assert!(stderr.contains("reasoning"));
        assert!(stderr.contains("tokens: 12 in / 4 out  cost: $0.0123"));
    }

    #[test]
    fn restored_transcript_includes_a_compaction_preview() {
        let messages = vec![
            compaction::summary_message("Prior work\nwith details", 1),
            llm::Message::User(llm::UserMessage::text("latest request", 2)),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![llm::ContentBlock::text("latest response")],
                ..llm::AssistantMessage::default()
            })),
        ];

        assert_eq!(
            restored_transcript_text(&messages, "resumed 3 message(s) from session"),
            "resumed 3 message(s) from session\n\
             · compacted Prior work ...\n\
             · user latest request\n\
             · assistant latest response\n\
             ────────────────────────────────────────"
        );
    }

    #[test]
    fn transcript_summary_hides_the_raw_compaction_wrapper() {
        let messages = vec![
            compaction::summary_message("Prior work\nwith details", 1),
            llm::Message::User(llm::UserMessage::text("latest request", 2)),
        ];

        let summary = transcript_summary(&messages);

        assert!(summary.contains("compacted  Prior work"));
        assert!(summary.contains("latest request"));
        assert!(!summary.contains(compaction::SUMMARY_OPEN));
        assert!(!summary.contains(compaction::SUMMARY_CLOSE));
    }

    #[test]
    fn tool_summary_is_stable_and_bounded() {
        let arguments = BTreeMap::from([
            ("a".to_owned(), serde_json::json!("value")),
            ("z".to_owned(), serde_json::json!("x".repeat(80))),
        ]);

        let summary = summarize_tool_arguments(&arguments);
        assert!(summary.starts_with("a=\"value\" z=\""));
        assert!(summary.ends_with("..."));
        assert!(summary.len() <= "a=\"value\" z=".len() + 63);
    }

    #[test]
    fn compaction_notices_report_message_counts_and_dropped_queue() {
        let outcome = compaction::Outcome {
            messages_before: 12,
            retained_messages: 4,
            tokens_before: 40_000,
            dropped_queued_messages: 2,
        };

        assert_eq!(
            compaction_notice_text(&outcome, false),
            "compacted context: 12 messages → summary + 4 recent messages"
        );
        assert_eq!(
            compaction_notice_text(&outcome, true),
            "compacted context automatically: 12 messages → summary + 4 recent messages"
        );
        assert_eq!(
            compaction_discard_notice(&outcome),
            Some(
                "2 queued message(s) were discarded: they were written against the transcript that was just compacted"
                    .to_owned()
            )
        );
    }

    #[test]
    fn sidebar_git_status_keeps_branch_rename_and_porcelain_state() {
        let info = parse_sidebar_git_status(
            "## feature/sidebar...origin/feature/sidebar [ahead 1]\n\
             MM src/main.rs\n\
             ?? notes.txt\n\
             R  old name.rs -> new name.rs\n",
        );

        assert_eq!(info.branch.as_deref(), Some("feature/sidebar"));
        assert_eq!(info.changes.len(), 3);
        assert_eq!(
            info.changes[0].status,
            state::FileStatus::Raw("MM".to_owned())
        );
        assert_eq!(info.changes[0].path, "src/main.rs");
        assert_eq!(
            info.changes[1].status,
            state::FileStatus::Raw("??".to_owned())
        );
        assert_eq!(
            info.changes[2].status,
            state::FileStatus::Raw("R".to_owned())
        );
        assert_eq!(info.changes[2].path, "new name.rs");
    }

    #[test]
    fn sidebar_git_status_labels_detached_heads() {
        let info = parse_sidebar_git_status("## HEAD (detached at 1234567)\n");

        assert_eq!(info.branch.as_deref(), Some("detached HEAD"));
    }

    #[test]
    fn native_line_ui_renders_session_card_and_prompt() {
        let info = NativeLineInfo {
            model: "openai/gpt-test".to_owned(),
            context_used: 1_000,
            context_limit: 4_000,
            cost: 0.0123,
            messages: 4,
            tools: 7,
            changed_files: 2,
            branch: "feature/native-ui".to_owned(),
            mode: "executing".to_owned(),
            thinking: "high".to_owned(),
        };

        let sidebar = native_line_sidebar(&info, 32, false);
        assert!(sidebar.contains("Context  1.0k/4.0k · 25%"));
        assert!(sidebar.contains("Files    2 changed"));
        assert!(sidebar.contains("Mode     executing · high"));
        let header = native_line_header(80, "test", Path::new("/workspace"), &info, false);
        assert!(header.contains("GoshCoder vtest"));
        assert!(header.contains("openai/gpt-test"));
        assert_eq!(native_line_input_prompt(8, false), "╭──────╮\n╰─❯ ");
    }

    #[test]
    fn planner_sidebar_uses_compact_mode_and_keeps_checklist_items() {
        let items = vec![
            plannotator::ChecklistItem {
                step: 1,
                text: "inspect the workspace".to_owned(),
                completed: true,
            },
            plannotator::ChecklistItem {
                step: 2,
                text: "implement the change".to_owned(),
                completed: false,
            },
        ];
        let (mode, visible_items) = planner_sidebar_details(Some(plannotator::State {
            phase: plannotator::Phase::Executing,
            items: items.clone(),
            ..plannotator::State::default()
        }));
        assert_eq!(mode, "executing");
        assert_eq!(visible_items, items);

        let (idle_mode, retained_items) = planner_sidebar_details(Some(plannotator::State {
            phase: plannotator::Phase::Idle,
            items,
            ..plannotator::State::default()
        }));
        assert_eq!(idle_mode, "normal");
        assert_eq!(retained_items.len(), 2);
        assert!(retained_items[0].completed);
        assert!(!retained_items[1].completed);
    }
}
