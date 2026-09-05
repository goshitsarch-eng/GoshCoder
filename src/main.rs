pub mod agent;
pub mod btw;
pub mod catalog;
pub mod computeruse;
pub mod config;
pub mod llm;
pub mod markdown;
pub mod omni_cli;
pub mod omniroute;
pub mod prompts;
pub mod provider_cli;
pub mod providers;
pub mod ralph;
pub mod ralph_cli;
pub mod resources;
pub mod runtime;
pub mod session;
pub mod sessionlog;
pub mod sessions;
mod state;
pub mod stream;
pub mod tools;
mod ui;

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    io::{self, IsTerminal, Write},
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
  goshcoder sessions [subcommand]    List, inspect, export, import, or remove sessions
  goshcoder prompts <subcommand>     Manage prompt templates
  goshcoder version                  Print the version

The Ratatui frontend, persistent-session, prompt, Ralph, provider, model, and
credential CLIs are active. `run` supports the OpenAI and Anthropic provider
protocols; interactive runtime integration and remaining provider extensions
are still being migrated from the previous implementation.
"#;

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
    println!("goshcoder {}", env!("CARGO_PKG_VERSION"));
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
            eprintln!("{}", dim(&banner, color_enabled()));
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
        agent::EventKind::MessageEnd => {
            if let Some(llm::Message::Assistant(message)) = event.message.as_ref() {
                for content in &message.content {
                    match content {
                        llm::ContentBlock::Text(text) => write!(stdout, "{}", text.text)?,
                        llm::ContentBlock::Thinking(thinking) => {
                            write!(stderr, "{}", dim(&thinking.thinking, color))?;
                        }
                        llm::ContentBlock::Image(_) | llm::ContentBlock::ToolCall(_) => {}
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
        | agent::EventKind::MessageUpdate
        | agent::EventKind::ToolExecutionUpdate
        | agent::EventKind::ModelChange
        | agent::EventKind::ThinkingLevelChange
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

fn run_interactive(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let invocation = runtime::parse_chat(arguments)?;
    if !invocation.config.fullscreen {
        return Err(io::Error::other(
            "line-mode chat is still being migrated; omit -fullscreen=false to use the Ratatui frontend",
        )
        .into());
    }
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Err(io::Error::other(
            "interactive chat requires a terminal; use `goshcoder run` for pipeable requests",
        )
        .into());
    }

    let quiet = invocation.config.quiet;
    let catalog = Arc::new(catalog::Catalog::with_default_credentials()?);
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

struct InteractiveView {
    notices: Vec<Message>,
    activity: String,
    recent_tool: String,
    activity_since: Option<Instant>,
    turn_pending: bool,
}

impl Default for InteractiveView {
    fn default() -> Self {
        Self {
            notices: Vec::new(),
            activity: "Ready".to_owned(),
            recent_tool: String::new(),
            activity_since: None,
            turn_pending: false,
        }
    }
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    agent_events: Receiver<agent::Event>,
    turn_sender: Sender<Result<(), String>>,
    turn_results: Receiver<Result<(), String>>,
    quiet: bool,
) -> Result<(), Box<dyn Error>> {
    let mut app = App::new();
    app.replace_messages(Vec::new());
    let mut view = InteractiveView::default();
    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            append_view_message(
                &mut view,
                MessageRole::Notice,
                format!("{}: {}", notice.kind, notice.text),
            );
        }
        if let Some(banner) = runtime::session_banner(&prepared.runtime) {
            append_view_message(&mut view, MessageRole::Notice, banner);
        }
    }

    loop {
        drain_interactive_events(&mut view, &prepared.runtime, &agent_events, &turn_results);
        refresh_runtime_app(&mut app, prepared, &view);
        terminal.draw(|frame| ui::draw(frame, &app))?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match app.handle_key(key) {
                Action::None => {}
                Action::Quit => {
                    prepared.runtime.agent().abort();
                    return Ok(());
                }
                Action::Abort => {
                    prepared.runtime.agent().abort();
                    view.activity = "Aborting".to_owned();
                }
                Action::CycleModel { direction } => {
                    match cycle_interactive_model(&prepared.runtime, catalog, direction) {
                        Ok(model) => view.activity = format!("Model set to {model}"),
                        Err(error) => append_view_message(&mut view, MessageRole::Error, error),
                    }
                }
                Action::CycleThinking => match cycle_interactive_thinking(&prepared.runtime) {
                    Some(level) => view.activity = format!("Thinking set to {level}"),
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

fn drain_interactive_events(
    view: &mut InteractiveView,
    runtime: &session::SessionRuntime,
    agent_events: &Receiver<agent::Event>,
    turn_results: &Receiver<Result<(), String>>,
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
            agent::EventKind::TurnStart
            | agent::EventKind::TurnEnd
            | agent::EventKind::MessageStart
            | agent::EventKind::MessageEnd
            | agent::EventKind::ToolExecutionUpdate
            | agent::EventKind::ModelChange
            | agent::EventKind::ThinkingLevelChange
            | agent::EventKind::TranscriptReset => {}
        }
    }
    while let Ok(result) = turn_results.try_recv() {
        view.turn_pending = false;
        if let Err(error) = result {
            append_view_message(view, MessageRole::Error, error);
        }
    }
    for notice in runtime::drain_notices() {
        append_view_message(
            view,
            MessageRole::Notice,
            format!("{}: {}", notice.kind, notice.text),
        );
    }
}

fn submit_interactive_input(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    turn_sender: Sender<Result<(), String>>,
    input: String,
    follow_up: bool,
) -> CommandDispatch {
    app.record_submission(&input);
    let input = match runtime::expand_resource_input(&prepared.resources, &input) {
        Ok(Some(expanded)) => expanded,
        Ok(None) => input,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return CommandDispatch::Handled;
        }
    };
    if input.starts_with('/') {
        return dispatch_runtime_slash_command(app, view, prepared, catalog, &input);
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

    view.turn_pending = true;
    view.activity = "Starting response".to_owned();
    view.activity_since = Some(Instant::now());
    app.streaming = true;
    thread::spawn(move || {
        let result = agent.prompt(input).map_err(|error| error.to_string());
        let _ = turn_sender.send(result);
    });
    CommandDispatch::NotCommand
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommandDispatch {
    NotCommand,
    Handled,
    Quit,
}

/// Executes slash commands that can be served without leaving the fullscreen
/// Ratatui program. Commands with an unavailable integration report that fact
/// in the transcript rather than pretending they changed runtime state.
fn dispatch_runtime_slash_command(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    input: &str,
) -> CommandDispatch {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();
    match command {
        "/exit" | "/quit" => CommandDispatch::Quit,
        "/help" | "/?" => {
            append_view_message(
                view,
                MessageRole::Command,
                "Slash commands:\n  /help                 Show this help\n  /model [ref]          List or choose an authenticated model\n  /thinking [level]     List or choose reasoning effort\n  /tools                List active tools\n  /status, /session     Show live session information\n  /messages             Show transcript summary\n  /queue                Show queued steering/follow-up messages\n  /steer <text>         Guide an active response\n  /followup <text>      Queue the next turn\n  /clear, /new          Reset this transcript\n  /name <text>          Set the persisted session name\n  /tree, /fork, /label  Inspect or rewind saved-session branches\n  /clone                Duplicate the current saved session\n  /resources            Show loaded context, prompts, and skills\n  /hotkeys              Show keyboard shortcuts\n  /exit                 Leave chat\n\nOAuth login, session picking, prompt editing, Ralph, planner, BTW, compaction, OmniRoute, and Aperture commands are still being migrated."
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
            let messages = &state.messages;
            let summary = messages
                .iter()
                .enumerate()
                .map(|(index, message)| {
                    format!(
                        "{:>3}  {:<10} {}",
                        index + 1,
                        message.role(),
                        first_line(&message.text_preview())
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
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
        "/resources" => {
            let resources = &prepared.resources;
            append_view_message(
                view,
                MessageRole::Command,
                format!(
                    "Context files: {}\nPrompt templates: {}\nSkills: {}\nWarnings: {}",
                    resources.context_files.len(),
                    resources.templates.len(),
                    resources.skills.len(),
                    resources.warnings.len()
                ),
            );
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
            prepared.runtime.agent().set_system_prompt(rest);
            view.activity = "System prompt updated for this session".to_owned();
            append_view_message(
                view,
                MessageRole::Notice,
                "The new system prompt applies to future turns in this session.",
            );
            CommandDispatch::Handled
        }
        "/login" => {
            append_view_message(
                view,
                MessageRole::Error,
                "OAuth login is not available in the Rust frontend yet; use `goshcoder auth set <provider>` outside chat.",
            );
            CommandDispatch::Handled
        }
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

fn refresh_runtime_app(app: &mut App, prepared: &runtime::PreparedSession, view: &InteractiveView) {
    let state = prepared.runtime.agent().state();
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
        env!("CARGO_PKG_VERSION"),
        state.model.provider,
        state.model.id
    );
    app.status = interactive_status(view, app.streaming);
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
            llm::Message::User(user) => result.push(Message {
                role: MessageRole::User,
                text: user_message_text(user),
                ..Message::default()
            }),
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
    view: &InteractiveView,
) -> Vec<state::SidebarLine> {
    let context = llm::Context {
        system_prompt: state.system_prompt.clone(),
        messages: state.messages.clone(),
        tools: state.tools.iter().map(agent::Tool::llm_tool).collect(),
    };
    let estimate = stream::estimate_context_tokens(&context);
    let limit = state.model.context_window;
    let percent = if limit == 0 {
        0
    } else {
        estimate
            .tokens
            .saturating_mul(100)
            .saturating_div(limit)
            .min(100) as u8
    };
    let cost = state
        .messages
        .iter()
        .filter_map(|message| match message {
            llm::Message::Assistant(message) => Some(message.usage.cost.total),
            llm::Message::User(_) | llm::Message::ToolResult(_) => None,
        })
        .sum::<f64>();
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
    let cwd = prepared
        .workspace
        .as_ref()
        .map(|workspace| workspace.root().display().to_string())
        .unwrap_or_else(|| prepared.config.workdir.display().to_string());
    let mut lines = vec![
        state::SidebarLine::title(name),
        state::SidebarLine::accent(format!("{}/{}", state.model.provider, state.model.id)),
        state::SidebarLine::meta(format!("{} thinking · normal", state.thinking_level)),
        state::SidebarLine::meta(storage),
        state::SidebarLine::blank(),
        state::SidebarLine::section("Context"),
        state::SidebarLine::progress(percent),
        state::SidebarLine::meta(if limit == 0 {
            format!("{} tokens", compact_number(estimate.tokens))
        } else {
            format!(
                "{} / {} tokens",
                compact_number(estimate.tokens),
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
    lines.extend([
        state::SidebarLine::blank(),
        state::SidebarLine::section("Workspace"),
        state::SidebarLine::path(cwd),
        state::SidebarLine::blank(),
        state::SidebarLine::brand(format!("● GoshCoder v{}", env!("CARGO_PKG_VERSION"))),
    ]);
    lines
}

fn session_status(prepared: &runtime::PreparedSession, activity: &str) -> String {
    let state = prepared.runtime.agent().state();
    let context = llm::Context {
        system_prompt: state.system_prompt.clone(),
        messages: state.messages.clone(),
        tools: state.tools.iter().map(agent::Tool::llm_tool).collect(),
    };
    let estimate = stream::estimate_context_tokens(&context);
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
        format!("{} tokens", compact_number(estimate.tokens))
    } else {
        format!(
            "{} / {} tokens",
            compact_number(estimate.tokens),
            compact_number(context_limit)
        )
    };
    format!(
        "Session: {}\nModel: {}/{}\nThinking: {}\nContext: {context}\nActivity: {activity}\nStorage: {storage}",
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
        (Some(index), true) if index > 0 => index - 1,
        (Some(0), true) => models.len() - 1,
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
            messages: Vec::new(),
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
        let mut ended = event(agent::EventKind::AgentEnd);
        ended.messages = vec![llm::Message::Assistant(Box::new(assistant))];

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
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
}
