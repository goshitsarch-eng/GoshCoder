mod state;
mod ui;

use std::{
    error::Error,
    io::{self, IsTerminal},
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
  goshcoder sessions <subcommand>    Manage saved sessions
  goshcoder prompts <subcommand>     Manage prompt templates
  goshcoder version                  Print the version

The Ratatui frontend is active. Runtime, provider, persistence, and extension
features are being migrated from the previous implementation.
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
                Action::Submit(prompt) => {
                    if dispatch_slash_command(&mut app, &prompt) {
                        return Ok(());
                    }
                    app.accept_submission(prompt, false);
                }
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

/// Returns true when the command ends the interactive session.
///
/// This small command set is deliberately self-contained: it gives the
/// Ratatui frontend useful behavior while the complete provider, session, and
/// extension command implementations are ported behind it.
fn dispatch_slash_command(app: &mut App, input: &str) -> bool {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();
    match command {
        "/exit" | "/quit" => true,
        "/help" | "/?" => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Command,
                text: "Slash commands:\n  /help                 Show this help\n  /model [ref]          Choose a model\n  /login [provider]     Add a provider\n  /thinking [level]     Choose reasoning effort\n  /status, /session     Show session information\n  /messages             Show transcript summary\n  /clear, /new          Clear the transcript\n  /hotkeys              Show keyboard shortcuts\n  /exit                 Leave chat\n\nThe remaining commands are being migrated with their current behavior."
                    .to_owned(),
                ..Message::default()
            });
            false
        }
        "/hotkeys" => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Command,
                text: "Enter       send or accept selection\nShift-Enter insert a newline\nUp/Down     navigate palette, editor lines, or history\nAlt-←/→     move by word; Home/End move within a line\nTab         complete the selected command\nCtrl-L      open model picker; Ctrl-O expand tools; Ctrl-T toggle thinking\nPgUp/PgDn   scroll transcript\nEsc         clear input or abort a response\nCtrl-C      clear input, abort, or quit\nCtrl-D      quit when the editor is empty"
                    .to_owned(),
                ..Message::default()
            });
            false
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
            false
        }
        "/status" | "/session" | "/sidebar" => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Command,
                text: "Session: Rust/Ratatui migration\nModel: not selected\nContext: 0 / 0 tokens\nMode: normal\nStorage: session persistence is being ported"
                    .to_owned(),
                ..Message::default()
            });
            false
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
            false
        }
        "/model" | "/login" | "/thinking" | "/tools" | "/resources" if rest.is_empty() => {
            app.accept_submission(input.to_owned(), false);
            app.add_message(Message {
                role: MessageRole::Notice,
                text: format!("{command} is wired into the Ratatui command palette; its runtime behavior is being ported."),
                ..Message::default()
            });
            false
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
            false
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert!(!dispatch_slash_command(&mut app, "/clear"));
        assert_eq!(app.messages.len(), 1);
        assert!(app.messages[0].text.contains("cleared"));
    }
}
