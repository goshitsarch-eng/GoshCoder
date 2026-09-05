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
    error::Error,
    io::{self, IsTerminal, Write},
    sync::{Arc, Mutex},
    time::Duration,
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
        Some("chat") | None => run_interactive(),
        Some(argument) if argument.starts_with('-') => run_interactive(),
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
                            "tokens: {} in / {} out  cost: ${:.4f}",
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

fn run_interactive() -> Result<(), Box<dyn Error>> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return Err(
            "interactive chat requires a terminal; pipeable command execution is being migrated"
                .into(),
        );
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = event_loop(&mut terminal);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<(), Box<dyn Error>> {
    let mut app = App::new();
    // The temporary standalone frontend has no recorder yet. The integrated
    // session runtime updates this as soon as it owns the terminal loop.
    app.set_recording_active(false);
    loop {
        terminal.draw(|frame| ui::draw(frame, &app))?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match app.handle_key(key) {
                Action::None => {}
                Action::Quit => return Ok(()),
                Action::Abort => {
                    app.streaming = false;
                    app.status = "No active response to abort".to_owned();
                }
                Action::CycleModel { direction } => {
                    app.status = if direction < 0 {
                        "Previous-model selection requested".to_owned()
                    } else {
                        "Next-model selection requested".to_owned()
                    };
                }
                Action::CycleThinking => {
                    app.status = "Thinking-level selection requested".to_owned();
                }
                Action::Submit(prompt) => match dispatch_slash_command(&mut app, &prompt) {
                    CommandDispatch::Quit => return Ok(()),
                    CommandDispatch::Handled => {}
                    CommandDispatch::NotCommand => {
                        app.accept_submission(prompt, false);
                    }
                },
                Action::FollowUp(prompt) => {
                    app.accept_submission(prompt, true);
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommandDispatch {
    NotCommand,
    Handled,
    Quit,
}

/// Records a known slash command and reports whether it was handled.
///
/// This small command set is deliberately self-contained: it gives the
/// Ratatui frontend useful behavior while the complete provider, session, and
/// extension command implementations are ported behind it.
fn dispatch_slash_command(app: &mut App, input: &str) -> CommandDispatch {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();
    match command {
        "/exit" | "/quit" => CommandDispatch::Quit,
        "/help" | "/?" => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Command,
                text: "Slash commands:\n  /help                 Show this help\n  /model [ref]          Choose a model\n  /login [provider]     Add a provider\n  /thinking [level]     Choose reasoning effort\n  /status, /session     Show session information\n  /messages             Show transcript summary\n  /clear, /new          Clear the transcript\n  /hotkeys              Show keyboard shortcuts\n  /exit                 Leave chat\n\nThe remaining commands are being migrated with their current behavior."
                    .to_owned(),
                ..Message::default()
            });
            CommandDispatch::Handled
        }
        "/hotkeys" => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Command,
                text: "Enter       send or accept selection\nShift-Enter insert a newline\nUp/Down     navigate palette, editor lines, or history\nAlt-←/→     move by word; Home/End move within a line\nTab         complete the selected command\nCtrl-L      open model picker; Ctrl-O expand tools; Ctrl-T toggle thinking\nPgUp/PgDn   scroll transcript\nEsc         clear input or abort a response\nCtrl-C      clear input, abort, or quit\nCtrl-D      quit when the editor is empty"
                    .to_owned(),
                ..Message::default()
            });
            CommandDispatch::Handled
        }
        "/clear" | "/new" => {
            app.messages.clear();
            app.add_message(Message {
                role: MessageRole::Notice,
                text: "Transcript cleared. Persistent session reset markers are being ported."
                    .to_owned(),
                ..Message::default()
            });
            app.status = "Transcript cleared".to_owned();
            CommandDispatch::Handled
        }
        "/status" | "/session" | "/sidebar" => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Command,
                text: "Session: Rust/Ratatui migration\nModel: not selected\nContext: 0 / 0 tokens\nMode: normal\nStorage: session persistence is being ported"
                    .to_owned(),
                ..Message::default()
            });
            CommandDispatch::Handled
        }
        "/messages" => {
            app.accept_submission(input.to_owned(), false);
            let summary = app
                .messages
                .iter()
                .enumerate()
                .map(|(index, message)| format!("{:>3}  {:?}", index + 1, message.role))
                .collect::<Vec<_>>()
                .join("\n");
            app.add_message(Message {
                role: MessageRole::Command,
                text: if summary.is_empty() {
                    "The transcript is empty.".to_owned()
                } else {
                    summary
                },
                ..Message::default()
            });
            CommandDispatch::Handled
        }
        "/model" | "/login" | "/thinking" | "/tools" | "/resources" if rest.is_empty() => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Notice,
                text: format!("{command} is wired into the Ratatui command palette; its runtime behavior is being ported."),
                ..Message::default()
            });
            CommandDispatch::Handled
        }
        _ if command.starts_with('/') => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Error,
                text: format!(
                    "{command} is not available until its Rust runtime implementation is complete."
                ),
                is_error: true,
                ..Message::default()
            });
            CommandDispatch::Handled
        }
        _ => CommandDispatch::NotCommand,
    }
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
    fn clear_command_replaces_the_transcript() {
        let mut app = App::new();
        app.messages.push(Message {
            role: MessageRole::Assistant,
            text: "old".to_owned(),
            ..Message::default()
        });

        assert_eq!(
            dispatch_slash_command(&mut app, "/clear"),
            CommandDispatch::Handled
        );
        assert_eq!(app.messages.len(), 1);
        assert!(app.messages[0].text.contains("cleared"));
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
