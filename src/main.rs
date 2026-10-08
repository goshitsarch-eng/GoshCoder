pub mod agent;
pub mod aperture;
pub mod aperture_cli;
pub mod aperture_mcp;
pub mod aperture_tools;
pub mod bedrock;
pub mod btw;
pub mod btw_runtime;
pub mod catalog;
pub mod compaction;
pub mod computeruse;
pub mod config;
pub mod export_html;
pub mod google_auth;
pub mod grok_accounts;
pub mod grok_cli;
pub mod grok_imagine;
mod line_editor;
pub mod llm;
pub mod markdown;
pub mod meta_muse;
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
mod tui_login;
pub mod turns;
mod ui;
pub mod webaccess;

use std::{
    collections::BTreeSet,
    error::Error,
    io::{self, BufRead, IsTerminal, Write},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, Sender},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        Event, KeyEventKind, KeyboardEnhancementFlags, MouseEventKind, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};

use crate::state::{Action, App, Message, MessageRole};

const SESSION_FLAGS: &str = r#"Flags:
  -m, -model <ref>      Model as provider/model, or a bare id when unambiguous
  -s, -system <text>    System prompt
  -thinking <level>     off, minimal, low, medium (default), high, xhigh, max
  -tools[=false]        Built-in file and shell tools (on in chat)
  -ralph[=false]        Long-running Ralph loops (on in chat)
  -planner              Start in Planner review mode (-plan is an alias)
  -C <dir>              Workspace directory for tools
  -continue             Reopen the most recent session for this workspace
  -resume               Choose a session to resume (chat only)
  -session <ref>        Session id, id prefix, or path
  -name <text>          Display name for the session
  -no-session           Do not record this session
  -read-only            Open a session without claiming it
  -sessions-dir <dir>   Session storage root
  -fullscreen[=false]   Full-screen interface (chat; default on a terminal)
  -claude-tui[=false]   pi-claude-code-tui look in line mode (chat)
  -quiet                Suppress session notices
"#;

/// `goshcoder <command> --help`. Commands with their own `help` keep it; the
/// rest get their usage here instead of an "unknown flag" error.
fn subcommand_help(args: &[String]) -> Option<String> {
    let command = args.first()?.as_str();
    let asks = |argument: &String| matches!(argument.as_str(), "-h" | "-help" | "--help");
    let asked = args.iter().skip(1).any(asks)
        || (args.get(1).is_some_and(|argument| argument == "help")
            && matches!(command, "sessions" | "prompts" | "ralph" | "auth"));
    // A bare `goshcoder -h` is the top-level usage, handled by `run`.
    if !asked {
        return None;
    }
    Some(match command {
        "run" => format!(
            "Usage: goshcoder run [flags] <prompt>\n\nRuns one prompt and exits; non-zero when the turn fails.\nRecords a session only with -continue, -session or -name.\n\n{SESSION_FLAGS}"
        ),
        "chat" => format!(
            "Usage: goshcoder [chat] [flags]\n\nInteractive session. Type / inside chat for commands.\n\n{SESSION_FLAGS}"
        ),
        "sessions" => "Usage: goshcoder sessions <subcommand>

  list [--all]                 Saved sessions for this workspace
  show <id>                    Print a session
  export <id> [--md] [path]    Save as HTML, Markdown (.md) or JSONL;
                               without a path, JSONL goes to stdout
  import <path>                Adopt a session file
  share <id> --yes             Upload as a secret GitHub gist (needs gh)
  rm <id>                      Delete a session
  gc --older-than 30d [--keep-named] [--yes]
                               Delete old sessions (a dry run without --yes)
"
        .to_owned(),
        "prompts" => "Usage: goshcoder prompts <subcommand>

  list                 Saved prompt templates
  backup [path]        Archive every template to a .tar.gz
  restore <archive>    Restore templates from a backup
"
        .to_owned(),
        "ralph" => "Usage: goshcoder ralph <subcommand>

  start <name> <task>  Start a loop
  list                 Loops in this workspace
  status [name]        Progress of a loop
  resume <name>        Resume a paused loop
  stop <name>          Stop a loop
  archive <name>       Archive a finished loop
  delete <name>        Delete a loop
"
        .to_owned(),
        "models" => "Usage: goshcoder models [provider]\n\nLists models for configured providers, or every model of one provider.\n".to_owned(),
        "providers" => "Usage: goshcoder providers\n\nLists providers, whether each is configured, and how to set up the rest.\n".to_owned(),
        "auth" => return None,
        "omni" | "aperture" => return None,
        _ => return None,
    })
}

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
  goshcoder grok-cli <subcommand>    Grok CLI usage and accounts
  goshcoder ralph <subcommand>       Manage Ralph loops
  goshcoder sessions [subcommand]    List, inspect, export, import, or remove sessions
  goshcoder prompts <subcommand>     Manage prompt templates
  goshcoder version                  Print the version

`run` and `chat` speak `openai-completions`, `openai-responses`,
`azure-openai-responses`, `openai-codex-responses`, `anthropic-messages`,
`google-generative-ai`, `google-vertex`, `mistral-conversations`,
`bedrock-converse-stream`, and the OmniRoute prompt-tools adapter. Gateways:
`omni setup` or OMNIROUTE_URL for OmniRoute, `aperture onboarding` for
Tailscale Aperture. Type /help inside chat for the slash commands.
"#;

/// Set by [`run_self_subprocess`]: where a child reports why it failed, since
/// the fullscreen interface redraws over whatever the child printed.
const CHILD_ERROR_FILE_ENV: &str = "GOSHCODER_CHILD_ERROR_FILE";

/// SIGTERM and SIGHUP while chat owns the terminal. Their default action
/// kills the process on the spot, which leaves the user's shell on the
/// alternate screen in raw mode with mouse reporting on, and leaves the
/// session file claimed. The handler only records the signal; the interface
/// loops notice it within one poll and leave the way `/exit` does.
pub mod termination {
    use std::sync::atomic::{AtomicI32, Ordering};

    static RECEIVED: AtomicI32 = AtomicI32::new(0);
    static INSTALLED: std::sync::Once = std::sync::Once::new();

    static HOOKS: std::sync::Mutex<Vec<Box<dyn Fn() + Send>>> = std::sync::Mutex::new(Vec::new());

    /// How long an orderly exit may take before the process ends anyway.
    /// After a hangup crossterm can spin forever reading the dead terminal
    /// inside its own poll, where the interface loop never gets to look at
    /// the request; this keeps that from outliving the terminal. The
    /// terminal is certainly gone after a hangup, so it waits less there.
    #[cfg(unix)]
    fn grace(signal: i32) -> std::time::Duration {
        std::time::Duration::from_millis(if signal == 1 { 500 } else { 5000 })
    }

    /// Routes SIGTERM and SIGHUP to [`requested`]. A no-op off Unix.
    pub fn install() {
        INSTALLED.call_once(install_handlers);
    }

    /// Runs `hook` off the interface thread as soon as a signal arrives, so
    /// work that must not depend on that thread (aborting a reply, which
    /// then records itself) happens even when it is stuck.
    pub fn on_request(hook: impl Fn() + Send + 'static) {
        HOOKS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(Box::new(hook));
    }

    fn install_handlers() {
        #[cfg(unix)]
        {
            use std::os::raw::c_int;
            const SIGHUP: c_int = 1;
            const SIGTERM: c_int = 15;
            unsafe extern "C" {
                // The handler as an address, as every other declaration of
                // this symbol in the crate has it.
                fn signal(signum: c_int, handler: usize) -> usize;
            }
            extern "C" fn record(signum: c_int) {
                // An atomic store is async-signal-safe.
                RECEIVED.store(signum, Ordering::SeqCst);
            }
            // SAFETY: installs a handler that only stores to an atomic.
            unsafe {
                signal(SIGTERM, record as extern "C" fn(c_int) as usize);
                signal(SIGHUP, record as extern "C" fn(c_int) as usize);
            }
            std::thread::spawn(|| {
                let signal = loop {
                    if let Some(signal) = requested() {
                        break signal;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                };
                for hook in HOOKS
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .iter()
                {
                    hook();
                }
                std::thread::sleep(grace(signal));
                exit_if_requested();
            });
        }
    }

    /// The signal that asked chat to end, if one has.
    pub fn requested() -> Option<i32> {
        match RECEIVED.load(Ordering::SeqCst) {
            0 => None,
            signal => Some(signal),
        }
    }

    /// Ends the process the way the signal would have, once cleanup is done:
    /// the shell sees the usual 128 + signal status.
    pub fn exit_if_requested() {
        if let Some(signal) = requested() {
            std::process::exit(128 + signal);
        }
    }
}

/// Ctrl-C while a child owns the terminal. With raw mode off the terminal
/// turns it into SIGINT for the whole foreground process group, which would
/// take the interface down with a login the user only meant to back out of.
mod interrupt {
    #[cfg(unix)]
    mod sys {
        use std::os::raw::c_int;

        const SIGINT: c_int = 2;
        const SIG_DFL: usize = 0;
        const SIG_IGN: usize = 1;

        unsafe extern "C" {
            fn signal(signum: c_int, handler: usize) -> usize;
        }

        pub fn ignore() -> usize {
            // SAFETY: installs a disposition constant, not a handler.
            unsafe { signal(SIGINT, SIG_IGN) }
        }

        pub fn restore(previous: usize) {
            // SAFETY: reinstates the disposition `ignore` returned.
            unsafe {
                signal(SIGINT, previous);
            }
        }

        pub fn reset_default() {
            // SAFETY: installs a disposition constant, not a handler.
            unsafe {
                signal(SIGINT, SIG_DFL);
            }
        }
    }

    #[cfg(windows)]
    mod sys {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn SetConsoleCtrlHandler(handler: usize, add: i32) -> i32;
        }

        pub fn ignore() -> usize {
            // SAFETY: a null handler toggles the process's Ctrl-C flag.
            unsafe {
                SetConsoleCtrlHandler(0, 1);
            }
            0
        }

        pub fn restore(_: usize) {
            reset_default();
        }

        pub fn reset_default() {
            // SAFETY: a null handler toggles the process's Ctrl-C flag.
            unsafe {
                SetConsoleCtrlHandler(0, 0);
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    mod sys {
        pub fn ignore() -> usize {
            0
        }
        pub fn restore(_: usize) {}
        pub fn reset_default() {}
    }

    /// Ignores Ctrl-C in this process until the guard drops.
    pub struct Ignored(usize);

    pub fn ignore() -> Ignored {
        Ignored(sys::ignore())
    }

    impl Drop for Ignored {
        fn drop(&mut self) {
            sys::restore(self.0);
        }
    }

    /// An ignored disposition is inherited, so a child started by
    /// [`super::run_self_subprocess`] takes Ctrl-C back for itself.
    pub fn reset_default() {
        sys::reset_default();
    }
}

/// Whether a child ended because the user pressed Ctrl-C.
fn interrupted(status: &std::process::ExitStatus) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.signal() == Some(2)
    }
    #[cfg(windows)]
    {
        // STATUS_CONTROL_C_EXIT
        status.code() == Some(0xC000_013A_u32 as i32)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = status;
        false
    }
}

/// A failure whose message the command already printed; the process only
/// needs to exit non-zero.
#[derive(Debug)]
struct AlreadyReported;

impl std::fmt::Display for AlreadyReported {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("the run failed")
    }
}

impl Error for AlreadyReported {}

fn main() {
    if std::env::var_os(CHILD_ERROR_FILE_ENV).is_some() {
        interrupt::reset_default();
    }
    if let Err(error) = run() {
        if !error.is::<AlreadyReported>() {
            eprintln!("error: {error}");
        }
        if let Some(path) = std::env::var_os(CHILD_ERROR_FILE_ENV) {
            let _ = std::fs::write(path, error.to_string());
        }
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(help) = subcommand_help(&args) {
        print!("{help}");
        return Ok(());
    }
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
        // OmniRoute names its help `help`; accept the usual flags too.
        Some("omni")
            if args
                .get(1)
                .is_some_and(|flag| matches!(flag.as_str(), "-h" | "-help" | "--help")) =>
        {
            omni_cli::command(&["help".to_owned()])
        }
        Some("omni") => omni_cli::command(&args[1..]),
        Some("aperture") => aperture_cli::command(&args[1..]),
        Some("grok-cli") => provider_cli::grok_cli_command(&args[1..]),
        Some("sessions") => sessions::command(&args[1..]),
        Some("prompts") => prompts::command(&args[1..]),
        Some("ralph") => ralph_cli::command(&args[1..]),
        Some("chat") => run_interactive(&args[1..]),
        None => run_interactive(&[]),
        Some(argument) if argument.starts_with('-') => run_interactive(&args),
        Some(command) => Err(unknown_command_message(command).into()),
    }
}

/// Names the subcommand that was probably meant (`provider` for `providers`)
/// before pointing at the usage text.
fn unknown_command_message(command: &str) -> String {
    const COMMANDS: &[&str] = &[
        "run",
        "chat",
        "providers",
        "models",
        "auth",
        "omni",
        "aperture",
        "grok-cli",
        "ralph",
        "sessions",
        "prompts",
        "version",
        "help",
    ];
    let lowered = command.to_ascii_lowercase();
    let suggestion = COMMANDS.iter().find(|candidate| {
        !lowered.is_empty() && (candidate.starts_with(&lowered) || lowered.starts_with(*candidate))
    });
    match suggestion {
        Some(suggestion) => {
            format!(
                "unknown command {command:?}; did you mean `goshcoder {suggestion}`? Run `goshcoder help` for usage"
            )
        }
        None => format!("unknown command {command:?}; run `goshcoder help` for usage"),
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
    let mut config = invocation.config;
    config.live_notices = !quiet;
    let prepared = runtime::prepare_session(catalog.as_ref(), config, Some(responder), Vec::new())?;

    // Notices were printed as they arrived (`live_notices`); only clear them.
    let _ = runtime::drain_session_notices(&prepared.runtime);
    if !quiet && let Some(banner) = runtime::session_banner(&prepared.runtime) {
        eprintln!("{}", dim(&banner, color_enabled()));
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
            let _ = render_run_event(event, &mut stdout, &mut stderr, color);
        }
    });

    prepared.sync_extensions()?;
    turns::run_prompt(
        &agent,
        prompt,
        &turns::RetryPolicy::default(),
        Some(&prepared.runtime.notice_sender()),
    )?;
    prepared.runtime.sync()?;
    // Notices were printed as they arrived (`live_notices`); only clear them.
    let _ = runtime::drain_session_notices(&prepared.runtime);
    // pi's print mode exits 1 when the final turn failed or was aborted, so
    // scripts can tell a provider error from an answer.
    let failed = agent
        .state()
        .messages
        .iter()
        .rev()
        .find_map(|message| match message {
            llm::Message::Assistant(message) => Some(
                message.stop_reason == stream::STOP_ERROR
                    || message.stop_reason == stream::STOP_ABORTED,
            ),
            _ => None,
        });
    if failed == Some(true) {
        return Err(Box::new(AlreadyReported));
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
                if message.stop_reason == stream::STOP_ABORTED {
                    // Asked for, so not an error; the fullscreen interface
                    // says the same with its "Interrupted" line.
                    writeln!(stderr, "\n{}", dim("(interrupted)", color))?;
                } else if !message.error_message.is_empty() {
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
            // Named, because several calls in a row would otherwise print
            // results nobody can match to their call.
            writeln!(
                stderr,
                "{} {}",
                dim(&format!("{status} {}:", event.tool_name), color),
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
        agent::EventKind::ContextCompacted => {
            if let Some(info) = event.compaction.as_ref() {
                writeln!(
                    stderr,
                    "{}",
                    dim(
                        &format!(
                            "context compacted: {} tokens → summary + {} recent messages",
                            info.tokens_before, info.retained_messages
                        ),
                        color,
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
    let mut invocation = runtime::parse_chat(arguments)?;
    choose_resume_session(&mut invocation.config)?;
    // Chat opens even before any provider is authenticated: the interface
    // walks the user through /login and /model instead of refusing to start.
    invocation.config.allow_unselected_model = true;
    if !invocation.config.fullscreen || !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        return run_line_interactive(invocation);
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
    // Unbounded on purpose: `emit` never runs under the agent's state lock,
    // and dropping the last events of a burst (a tool's end, the turn's end)
    // would leave the status line stuck on stale activity.
    let (agent_event_sender, agent_event_receiver) = mpsc::channel();
    let (turn_sender, turn_receiver) = mpsc::channel();
    let _agent_event_subscription = agent.subscribe(move |event| {
        let _ = agent_event_sender.send(event.clone());
    });

    // A panic anywhere on the UI thread must not leave the shell in raw mode
    // on the alternate screen; the hook restores the terminal before the
    // default hook prints the message.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal_modes();
        previous_hook(info);
    }));

    termination::install();
    termination::on_request({
        let agent = agent.clone();
        move || agent.abort()
    });
    if let Err(error) = enter_terminal_modes() {
        restore_terminal_modes();
        return Err(error.into());
    }
    let backend = CrosstermBackend::new(io::stderr());
    let mut terminal = match Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(error) => {
            restore_terminal_modes();
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

    restore_terminal_modes();
    let terminal_cleanup = terminal.show_cursor();
    // A session without a reply is discarded on close, so only one with an
    // answer in it is worth pointing back to.
    let resumable = prepared.runtime.recording()
        && prepared
            .runtime
            .agent()
            .state()
            .messages
            .iter()
            .any(|message| matches!(message, llm::Message::Assistant(_)));
    let session_cleanup = prepared.runtime.close();
    // After a hangup the terminal is gone: writing to it would fail, so
    // nothing more is said once the session is safely closed.
    termination::exit_if_requested();
    terminal_cleanup?;
    session_cleanup?;
    if result.is_ok() && resumable {
        eprintln!("Resume with: goshcoder chat -continue");
    }
    result
}

/// Raw mode, the alternate screen, mouse capture, and bracketed paste. The
/// last one is what makes a multi-line paste arrive as one `Event::Paste`
/// instead of a burst of Enter keys that submit line by line.
fn enter_terminal_modes() -> io::Result<()> {
    enable_raw_mode()?;
    execute!(
        io::stderr(),
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste
    )?;
    // The kitty keyboard protocol's first level reports Shift-Enter as
    // itself instead of a bare Enter, which is what makes "Shift-Enter
    // inserts a newline" true. Terminals without it are left alone, and
    // Ctrl-J keeps working everywhere.
    if crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false)
        && execute!(
            io::stderr(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )
        .is_ok()
    {
        KEYBOARD_ENHANCED.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(())
}

/// Undoes [`enter_terminal_modes`]. Safe to call more than once and from a
/// panic hook: every step is best effort.
fn restore_terminal_modes() {
    if KEYBOARD_ENHANCED.swap(false, std::sync::atomic::Ordering::Relaxed) {
        let _ = execute!(io::stderr(), PopKeyboardEnhancementFlags);
    }
    let _ = disable_raw_mode();
    let _ = execute!(
        io::stderr(),
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
}

/// Steps the fullscreen interface aside for a command that needs the real
/// terminal (an OAuth login, onboarding prompts, `$EDITOR`), then takes the
/// screen back and redraws from scratch.
fn with_suspended_terminal<T>(
    terminal: &mut Terminal<CrosstermBackend<io::Stderr>>,
    run: impl FnOnce() -> T,
) -> io::Result<T> {
    restore_terminal_modes();
    let _ = terminal.show_cursor();
    let outcome = run();
    enter_terminal_modes()?;
    terminal.clear()?;
    Ok(outcome)
}

/// Runs this executable again with `arguments`, sharing the terminal, and
/// reports a non-zero exit as an error. Subcommands that prompt (`auth login`,
/// `aperture onboarding`, `omni setup`) are reused this way rather than
/// re-implemented against the alternate screen.
fn run_self_subprocess(arguments: &[&str]) -> Result<(), String> {
    let executable =
        std::env::current_exe().map_err(|error| format!("locate goshcoder: {error}"))?;
    let error_file = std::env::temp_dir().join(format!(
        "goshcoder-child-error-{}-{}",
        std::process::id(),
        uuid::Uuid::now_v7()
    ));
    let status = {
        let _interrupt = interrupt::ignore();
        std::process::Command::new(&executable)
            .args(arguments)
            .env(CHILD_ERROR_FILE_ENV, &error_file)
            .status()
            .map_err(|error| format!("run goshcoder {}: {error}", arguments.join(" ")))?
    };
    let reported = std::fs::read_to_string(&error_file).ok();
    let _ = std::fs::remove_file(&error_file);
    if status.success() {
        return Ok(());
    }
    // Ctrl-C at a prompt is the user backing out, not a failure to dwell on.
    if interrupted(&status) {
        return Err(format!("goshcoder {} was cancelled", arguments.join(" ")));
    }
    let cancelled = reported
        .as_deref()
        .is_some_and(|message| message.contains("cancelled"));
    if !cancelled {
        eprint!("\nPress Enter to return to GoshCoder. ");
        let _ = io::stderr().flush();
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
    }
    Err(match reported {
        Some(message) if !message.trim().is_empty() => message.trim().to_owned(),
        _ => format!("goshcoder {} exited with {status}", arguments.join(" ")),
    })
}

/// Runs a network-bound command off the terminal thread. Its output arrives
/// through [`drain_interactive_events`], so the interface keeps drawing while
/// a gateway answers slowly.
fn start_background_command(
    view: &mut InteractiveView,
    label: &str,
    job: impl FnOnce() -> Result<String, String> + Send + 'static,
) {
    if let Some(running) = view.background.as_ref() {
        append_view_message(
            view,
            MessageRole::Error,
            format!(
                "wait for {} to finish before running {label}",
                running.label
            ),
        );
        return;
    }
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(job());
    });
    view.background = Some(BackgroundCommand {
        label: label.to_owned(),
        receiver,
    });
    view.activity = format!("Running {label}");
    view.activity_since = Some(Instant::now());
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
fn run_line_interactive(invocation: runtime::Invocation) -> Result<(), Box<dyn Error>> {
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
            let _ = render_run_event(event, &mut stdout, &mut stderr, color);
        }
    });

    let result = line_interactive_loop(&prepared, catalog.as_ref(), quiet);
    let close_result = prepared.runtime.close();
    termination::exit_if_requested();
    result?;
    close_result?;
    Ok(())
}

fn line_interactive_loop(
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    quiet: bool,
) -> Result<(), Box<dyn Error>> {
    let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            eprintln!("{}", dim(&format!("session: {notice}"), color_enabled()));
        }
        if let Some(banner) = runtime::session_banner(&prepared.runtime) {
            eprintln!("{}", dim(&banner, color_enabled()));
        }
        if interactive {
            let state = prepared.runtime.agent().state();
            eprintln!(
                "{}",
                dim(
                    &format!(
                        "goshcoder {} · {} · /help for commands",
                        build_version(),
                        model_label(&state.model)
                    ),
                    color_enabled()
                )
            );
            if !runtime::model_is_selected(&state.model) {
                eprintln!(
                    "{}",
                    dim(
                        "No provider is authenticated yet: run /login <provider> (see /login for the list); the first login also selects a model.",
                        color_enabled()
                    )
                );
            }
        }
    }

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut raw = String::new();
    let mut history: Vec<String> = Vec::new();
    if interactive {
        // Ctrl-C during a reply aborts the turn instead of killing chat; at
        // the prompt it is a key the editor handles (twice to exit).
        line_editor::install_interrupt_handler();
        termination::install();
        let agent = prepared.runtime.agent().clone();
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_millis(50));
                // Not only while streaming: a retry's backoff is part of the
                // turn too, and the prompt reads keys in raw mode, so a
                // SIGINT here always means "stop the response".
                if termination::requested().is_some() {
                    let _ = agent.take_queued_messages();
                    agent.abort();
                }
                if line_editor::take_interrupts() > 0 {
                    eprintln!("\n^C aborting the response");
                    let _ = agent.take_queued_messages();
                    agent.abort();
                }
            }
        });
    }
    loop {
        raw.clear();
        if interactive {
            eprintln!();
            match line_editor::read_line("> ", &history)? {
                line_editor::LineInput::Line(line) => {
                    if !line.trim().is_empty() && history.last() != Some(&line) {
                        history.push(line.clone());
                    }
                    raw = line;
                }
                line_editor::LineInput::Exit | line_editor::LineInput::Eof => break,
            }
        } else if reader.read_line(&mut raw)? == 0 {
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
            let mut view = InteractiveView::default();
            let (turn_sender, turn_receiver) = mpsc::channel();
            let outcome = dispatch_runtime_slash_command(
                &mut app,
                &mut view,
                prepared,
                catalog,
                turn_sender,
                &input,
                false,
            );
            let outcome = match outcome {
                CommandDispatch::Suspended(run) => {
                    match run() {
                        Ok(output) if !output.is_empty() => {
                            append_view_message(&mut view, MessageRole::Command, output);
                        }
                        Ok(_) => {}
                        Err(error) => append_view_message(&mut view, MessageRole::Error, error),
                    }
                    CommandDispatch::Handled
                }
                other => other,
            };
            if let Some(background) = view.background.take() {
                match background.receiver.recv() {
                    Ok(Ok(output)) if !output.is_empty() => {
                        append_view_message(&mut view, MessageRole::Command, output);
                    }
                    Ok(Ok(_)) => {}
                    Ok(Err(error)) => append_view_message(&mut view, MessageRole::Error, error),
                    Err(_) => append_view_message(
                        &mut view,
                        MessageRole::Error,
                        format!("{} stopped without reporting a result", background.label),
                    ),
                }
            }
            if view.turn_pending {
                match turn_receiver.recv() {
                    Ok(result) if view.pending_btw_thread.is_some() => {
                        let _ = finish_pending_btw(&mut view, prepared, result);
                    }
                    Ok(Ok(())) => {
                        view.turn_pending = false;
                        view.activity = "Ready".to_owned();
                    }
                    Ok(Err(error)) => {
                        view.turn_pending = false;
                        append_view_message(&mut view, MessageRole::Error, error);
                    }
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
            if !runtime::model_is_selected(&prepared.runtime.agent().state().model) {
                eprintln!("error: {NO_MODEL_PROMPT_REFUSED}");
                continue;
            }
            if let Err(error) = prepared.sync_extensions() {
                eprintln!("error: {error}");
                continue;
            }
            if let Err(error) = turns::run_prompt(
                prepared.runtime.agent(),
                input,
                &turns::RetryPolicy::default(),
                Some(&prepared.runtime.notice_sender()),
            ) {
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

fn render_line_view(view: &mut InteractiveView) {
    let notices = std::mem::take(&mut view.notices);
    let had_notices = !notices.is_empty();
    for AnchoredNotice { message, .. } in notices {
        match message.role {
            MessageRole::Error => eprintln!("error: {}", message.text),
            // Line mode already shows what was typed at its own prompt.
            MessageRole::Command => {}
            MessageRole::Notice => eprintln!("{}", message.text),
            MessageRole::User
            | MessageRole::Assistant
            | MessageRole::Thinking
            | MessageRole::Tool
            | MessageRole::Summary => {
                eprintln!("{}", message.text)
            }
        }
    }
    if !had_notices && view.activity != "Ready" {
        eprintln!("{}", view.activity);
    }
}

struct InteractiveView {
    /// Command output and notices, each anchored to the number of agent
    /// messages that existed when it was added. The transcript splices them
    /// in at that point, so a reply that arrives later renders below an
    /// older `/help` instead of above it.
    notices: Vec<AnchoredNotice>,
    /// Agent messages at the last refresh: the anchor for a new notice.
    message_count: usize,
    /// A provider retry being waited out: when it fires, and which attempt.
    retry: Option<RetryWait>,
    /// Steering and follow-up messages waiting, at the last refresh.
    queued_count: usize,
    activity: String,
    recent_tool: String,
    activity_since: Option<Instant>,
    turn_pending: bool,
    pending_btw_thread: Option<String>,
    pending_btw_turn_start: Option<usize>,
    /// A gateway command running off the terminal thread.
    background: Option<BackgroundCommand>,
    /// Palette entries cached while `/model ` or `/login ` is being typed.
    model_choices: Option<Vec<state::Suggestion>>,
    login_choices: Option<Vec<state::Suggestion>>,
    /// Saved sessions, cached while `/resume ` is being typed.
    resume_choices: Option<Vec<state::Suggestion>>,
    /// A `/login` running inside the interface.
    login: Option<tui_login::LoginFlow>,
}

/// Reports a worker's outcome exactly once, including when the worker
/// panics: without this a panic leaves the turn marked pending forever and
/// every later Enter becomes a steering message for an idle agent.
struct TurnCompletion {
    sender: Option<Sender<Result<(), String>>>,
}

impl TurnCompletion {
    fn new(sender: Sender<Result<(), String>>) -> Self {
        Self {
            sender: Some(sender),
        }
    }

    fn finish(mut self, result: Result<(), String>) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(result);
        }
    }
}

impl Drop for TurnCompletion {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err("the worker stopped unexpectedly".to_owned()));
        }
    }
}

/// A transcript notice and the agent message count it follows.
struct AnchoredNotice {
    anchor: usize,
    message: Message,
}

/// The retry `turns::run_prompt` announced and is sleeping before.
struct RetryWait {
    attempt: u32,
    attempts: u32,
    fires_at: Instant,
}

/// A command whose result is still on its way from a worker thread.
struct BackgroundCommand {
    label: String,
    receiver: Receiver<Result<String, String>>,
}

impl Default for InteractiveView {
    fn default() -> Self {
        Self {
            notices: Vec::new(),
            message_count: 0,
            retry: None,
            queued_count: 0,
            activity: "Ready".to_owned(),
            recent_tool: String::new(),
            activity_since: None,
            turn_pending: false,
            pending_btw_thread: None,
            pending_btw_turn_start: None,
            background: None,
            model_choices: None,
            login_choices: None,
            resume_choices: None,
            login: None,
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
    // A resumed transcript is already there; startup notices follow it.
    let mut view = InteractiveView {
        message_count: prepared.runtime.agent().state().messages.len(),
        ..InteractiveView::default()
    };
    if !quiet {
        for notice in runtime::drain_session_notices(&prepared.runtime) {
            append_view_message(&mut view, MessageRole::Notice, notice);
        }
        // The sidebar already says a new session is recording; only a
        // resumed one is worth a line in the transcript.
        if prepared.runtime.resumed()
            && let Some(banner) = runtime::session_banner(&prepared.runtime)
        {
            append_view_message(&mut view, MessageRole::Notice, banner);
        }
    }
    if !quiet && prepared.config.no_session {
        append_view_message(
            &mut view,
            MessageRole::Notice,
            format!(
                "-no-session: this conversation is kept in memory only and will not be written to {}.",
                home_relative(
                    &prepared
                        .config
                        .sessions_dir
                        .clone()
                        .unwrap_or_else(config::sessions_dir)
                        .display()
                        .to_string()
                )
            ),
        );
    }
    if !runtime::model_is_selected(&prepared.runtime.agent().state().model) {
        append_view_message(&mut view, MessageRole::Notice, NO_MODEL_WELCOME);
        app.set_input("/login ");
    }

    // Rebuilding the view means cloning the agent transcript and re-rendering
    // it, so it happens only when something changed or a spinner is running,
    // not on every poll timeout of an idle session.
    let mut dirty = true;
    loop {
        if termination::requested().is_some() {
            leave_chat(prepared, &view, &turn_results);
            return Ok(());
        }
        if app.expire_quit_arm() {
            if view.activity.starts_with("Press Ctrl+C again") {
                view.activity = "Ready".to_owned();
            }
            dirty = true;
        }
        if drain_interactive_events(&mut view, prepared, &agent_events, &turn_results)
            | drain_login_events(&mut app, &mut view, prepared, catalog)
        {
            dirty = true;
        }
        let animating = app.streaming || view.turn_pending || view.background.is_some();
        if dirty || animating {
            refresh_runtime_app(&mut app, prepared, catalog, &mut view);
            terminal.draw(|frame| ui::draw(frame, &app))?;
            dirty = false;
        }

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        // Every event already queued (a key-repeat burst, a large paste) is
        // applied before the next draw instead of costing one frame each.
        loop {
            dirty = true;
            let status_before = app.status.clone();
            let action = match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => app.handle_key(key),
                Event::Paste(text) => {
                    app.paste(&text);
                    Action::None
                }
                Event::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => app.scroll_up(3),
                        MouseEventKind::ScrollDown => app.scroll_down(3),
                        _ => {}
                    }
                    Action::None
                }
                _ => Action::None,
            };
            // Feedback the editor set for this key (Ctrl-C confirmation, a
            // toggle) survives the next refresh, which would otherwise
            // overwrite it with the idle activity text.
            if app.status != status_before {
                view.activity = app.status.clone();
            }
            let submission = match &action {
                Action::FollowUp(_) => Submission::FollowUp,
                _ => Submission::Prompt,
            };
            match action {
                Action::None => {}
                Action::Answer(answer) => {
                    answer_composer_prompt(&mut app, &mut view, prepared, catalog, answer)
                }
                Action::CancelPrompt => cancel_composer_prompt(&mut app, &mut view),
                Action::Quit => {
                    leave_chat(prepared, &view, &turn_results);
                    return Ok(());
                }
                Action::Abort if view.login.is_some() && !view.turn_pending => {
                    cancel_composer_prompt(&mut app, &mut view);
                }
                Action::Abort => {
                    if let Some(thread) = view.pending_btw_thread.as_deref() {
                        let _ = prepared.btw.cancel(thread);
                    }
                    // pi restores queued messages to the editor before the
                    // abort, so nothing queued runs after the interrupted
                    // turn and no text is lost.
                    let restored =
                        restore_queued_messages_to_editor(&mut app, prepared.runtime.agent());
                    prepared.runtime.agent().abort();
                    if let Some(planner) = prepared.planner.as_ref() {
                        planner.abort_review();
                    }
                    view.activity = if restored > 0 {
                        format!("Aborting; {restored} queued message(s) restored to the editor")
                    } else {
                        "Aborting".to_owned()
                    };
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
                Action::Submit(input) | Action::FollowUp(input) => {
                    let follow_up = matches!(submission, Submission::FollowUp);
                    match submit_interactive_input(
                        &mut app,
                        &mut view,
                        prepared,
                        catalog,
                        turn_sender.clone(),
                        input,
                        follow_up,
                    ) {
                        CommandDispatch::Quit => {
                            leave_chat(prepared, &view, &turn_results);
                            return Ok(());
                        }
                        CommandDispatch::Suspended(run) => {
                            match with_suspended_terminal(terminal, run)? {
                                Ok(output) if !output.is_empty() => {
                                    append_view_message(&mut view, MessageRole::Command, output);
                                }
                                Ok(_) => {}
                                Err(error) => {
                                    append_view_message(&mut view, MessageRole::Error, error);
                                }
                            }
                            // A login that could not choose a model on its
                            // own hands over to the picker, now that there
                            // is something to pick from.
                            if !runtime::model_is_selected(&prepared.runtime.agent().state().model)
                                && interactive_models(catalog)
                                    .is_ok_and(|models| !models.is_empty())
                            {
                                app.set_input("/model ");
                            }
                            // Whatever was typed at the child process is
                            // not input for this screen.
                            break;
                        }
                        CommandDispatch::Handled | CommandDispatch::NotCommand => {}
                    }
                }
            }
            if !event::poll(Duration::ZERO)? {
                break;
            }
        }
    }
}

/// `/queue`: what will run after the active response, in order.
fn queue_report(queued: &[String]) -> String {
    if queued.is_empty() {
        return "Nothing is queued. While a response runs, Enter steers it and Alt-Enter queues a follow-up.".to_owned();
    }
    let mut report = format!("{} queued:", plural(queued.len(), "message", "messages"));
    for (index, text) in queued.iter().enumerate() {
        report.push_str(&format!("\n{:>3}. {}", index + 1, first_line(text)));
    }
    report
}

/// Puts `text` on the clipboard through the terminal (OSC 52), which works
/// over SSH and needs no clipboard tool; a terminal that does not support
/// it ignores the sequence. Written between frames, so the screen is left
/// as it was.
fn copy_to_terminal_clipboard(text: &str) {
    use base64::Engine as _;
    let encoded = base64::engine::general_purpose::STANDARD.encode(text);
    let mut stderr = io::stderr();
    let _ = write!(stderr, "\x1b]52;c;{encoded}\x07");
    let _ = stderr.flush();
}

/// Stops everything still running before the interface closes the session.
/// An interrupted turn gets a moment to record its partial reply: closing
/// first would find no assistant message and discard the session, taking
/// the user's prompt with it.
fn leave_chat(
    prepared: &runtime::PreparedSession,
    view: &InteractiveView,
    turn_results: &Receiver<Result<(), String>>,
) {
    if let Some(thread) = view.pending_btw_thread.as_deref() {
        let _ = prepared.btw.cancel(thread);
    }
    prepared.runtime.agent().abort();
    if let Some(planner) = prepared.planner.as_ref() {
        planner.abort_review();
    }
    if view.turn_pending {
        let _ = turn_results.recv_timeout(Duration::from_secs(3));
    }
}

/// Whether a submitted line starts a turn or queues a follow-up.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Submission {
    Prompt,
    FollowUp,
}

fn drain_interactive_events(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    agent_events: &Receiver<agent::Event>,
    turn_results: &Receiver<Result<(), String>>,
) -> bool {
    let mut changed = false;
    while let Ok(event) = agent_events.try_recv() {
        changed = true;
        match event.kind {
            agent::EventKind::AgentStart => {
                view.retry = None;
                view.turn_pending = true;
                view.activity = "Composing response".to_owned();
                view.activity_since = Some(Instant::now());
            }
            agent::EventKind::MessageUpdate => {
                view.retry = None;
                view.activity = "Composing response".to_owned();
                view.activity_since.get_or_insert_with(Instant::now);
            }
            agent::EventKind::ToolExecutionStart => {
                // "Running bash · cargo test": the tool and what it was
                // asked to do, as its card title says it.
                let call = llm::ToolCall {
                    name: event.tool_name.clone(),
                    arguments: event.arguments.clone(),
                    ..llm::ToolCall::default()
                };
                let title = tool_title(&call);
                let detail = title
                    .strip_prefix(event.tool_name.as_str())
                    .map(str::trim)
                    .unwrap_or_default();
                view.activity = if detail.is_empty() {
                    format!("Running {}", event.tool_name)
                } else {
                    format!(
                        "Running {} · {}",
                        event.tool_name,
                        clip_characters(&first_line(detail), 48)
                    )
                };
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
                view.retry = None;
                view.turn_pending = false;
                view.activity = "Ready".to_owned();
                view.activity_since = None;
            }
            agent::EventKind::ContextCompacted => {
                if let Some(info) = event.compaction {
                    view.activity = "Context compacted".to_owned();
                    view.activity_since = None;
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!(
                            "Context compacted: {} tokens → summary + {} recent messages.",
                            info.tokens_before, info.retained_messages
                        ),
                    );
                }
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
        view.retry = None;
        changed = true;
        if view.pending_btw_thread.is_some() {
            let _ = finish_pending_btw(view, prepared, result);
            continue;
        }
        view.turn_pending = false;
        view.activity = "Ready".to_owned();
        view.activity_since = None;
        if let Err(error) = result {
            append_view_message(view, MessageRole::Error, error);
        }
    }
    if let Some(background) = view.background.as_ref() {
        match background.receiver.try_recv() {
            Ok(result) => {
                changed = true;
                let label = background.label.clone();
                view.background = None;
                if !view.turn_pending {
                    view.activity = "Ready".to_owned();
                    view.activity_since = None;
                }
                match result {
                    Ok(output) if !output.is_empty() => {
                        append_view_message(view, MessageRole::Command, output);
                    }
                    Ok(_) => {
                        append_view_message(view, MessageRole::Notice, format!("{label} finished"))
                    }
                    Err(error) => append_view_message(view, MessageRole::Error, error),
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                changed = true;
                let label = background.label.clone();
                view.background = None;
                append_view_message(
                    view,
                    MessageRole::Error,
                    format!("{label} stopped without reporting a result"),
                );
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    let notices = prepared.runtime.drain_notices();
    if !notices.is_empty() {
        changed = true;
        // Anchor them after whatever the agent added since the last frame
        // (the prompt a retry notice is about, for one).
        view.message_count = prepared.runtime.agent().state().messages.len();
    }
    for notice in notices {
        if notice.kind == "retry"
            && let Some((retry, summary)) = parse_retry_notice(&notice.text)
        {
            append_view_message(
                view,
                MessageRole::Notice,
                format!(
                    "{summary}. Retrying in {}s (attempt {} of {}).",
                    retry
                        .fires_at
                        .saturating_duration_since(Instant::now())
                        .as_secs_f64()
                        .round(),
                    retry.attempt,
                    retry.attempts
                ),
            );
            view.retry = Some(retry);
            // The failed attempt ended the agent run, but the turn is still
            // going until the retry succeeds or gives up.
            view.turn_pending = true;
            view.activity_since.get_or_insert_with(Instant::now);
            continue;
        }
        append_view_message(
            view,
            MessageRole::Notice,
            format!("{}: {}", notice.kind, notice.text),
        );
    }
    changed
}

/// Reads `turns`' retry notice ("attempt 1 of 3 in 2s: <error>") into the
/// countdown the status bar shows. `turns` numbers retries; the interface
/// numbers attempts, the first request being attempt 1.
fn parse_retry_notice(text: &str) -> Option<(RetryWait, String)> {
    let rest = text.strip_prefix("attempt ")?;
    let (retry, rest) = rest.split_once(" of ")?;
    let (retries, rest) = rest.split_once(" in ")?;
    let (seconds, summary) = rest.split_once("s: ")?;
    let retry = retry.trim().parse::<u32>().ok()?;
    let retries = retries.trim().parse::<u32>().ok()?;
    let seconds = seconds.trim().parse::<f64>().ok()?;
    Some((
        RetryWait {
            attempt: retry + 1,
            attempts: retries + 1,
            fires_at: Instant::now() + Duration::from_secs_f64(seconds.max(0.0)),
        },
        summary.trim().trim_end_matches('.').to_owned(),
    ))
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

/// Applies whatever an in-interface login reported since the last frame.
fn drain_login_events(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
) -> bool {
    let mut changed = false;
    while let Some(flow) = view.login.as_mut() {
        let Some(event) = flow.poll() else {
            break;
        };
        changed = true;
        let provider = flow.provider().to_owned();
        match event {
            tui_login::LoginEvent::Notify(event) => {
                match event.kind {
                    oauth::OAuthEventKind::DeviceCode => {
                        view.activity = format!("Waiting for {provider} approval");
                    }
                    oauth::OAuthEventKind::AuthorizationUrl => {
                        view.activity = format!("Waiting for {provider} sign-in");
                    }
                    oauth::OAuthEventKind::Progress if !event.message.is_empty() => {
                        view.activity = first_line(&event.message);
                    }
                    _ => {}
                }
                if let Some(mut notice) = tui_login::event_notice(&provider, &event) {
                    if let Some(text) = tui_login::clipboard_text(&event) {
                        copy_to_terminal_clipboard(text);
                        notice.push_str(if event.kind == oauth::OAuthEventKind::DeviceCode {
                            "\nThe code is also on your clipboard, where the terminal allows it."
                        } else {
                            "\nThe address is also on your clipboard, where the terminal allows it."
                        });
                    }
                    append_view_message(view, MessageRole::Notice, notice);
                }
            }
            tui_login::LoginEvent::Prompt { prompt, .. } => {
                // A choice is made in the palette, like any other picker;
                // a paste goes into the composer.
                let options = if prompt.select {
                    prompt
                        .options
                        .iter()
                        .map(|option| state::Suggestion {
                            label: option.label.clone(),
                            description: option.description.clone(),
                            value: option.id.clone(),
                            execute: true,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                append_view_message(
                    view,
                    MessageRole::Notice,
                    if prompt.select {
                        prompt.message.clone()
                    } else {
                        tui_login::prompt_notice(&prompt)
                    },
                );
                app.prompt = Some(state::ComposerPrompt {
                    label: if prompt.select {
                        format!("{provider} login method")
                    } else {
                        format!("{provider} login")
                    },
                    secret: false,
                    placeholder: prompt.placeholder.clone(),
                    options,
                });
                view.activity = "Waiting for your answer".to_owned();
            }
            tui_login::LoginEvent::Finished(result) => {
                finish_login(app, view, prepared, catalog, &provider, result, true);
            }
        }
    }
    changed
}

/// Reports a finished login and puts the new provider to use.
fn finish_login(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    provider: &str,
    result: Result<(), String>,
    oauth: bool,
) {
    view.login = None;
    app.prompt = None;
    view.activity = "Ready".to_owned();
    view.activity_since = None;
    match result {
        Ok(()) => {
            catalog.clear_oauth_refresh_failure(provider);
            let saved = home_relative(&config::auth_path().display().to_string());
            let first = if oauth {
                format!("Logged in to {provider}. Credentials saved to {saved} (mode 0600).")
            } else {
                format!("Stored an API key for {provider} in {saved} (mode 0600).")
            };
            append_view_message(
                view,
                MessageRole::Notice,
                format!(
                    "{first}\n{}",
                    after_login_message(prepared, catalog, provider)
                ),
            );
            if !runtime::model_is_selected(&prepared.runtime.agent().state().model)
                && interactive_models(catalog).is_ok_and(|models| !models.is_empty())
            {
                app.set_input("/model ");
            }
        }
        Err(error) if error.contains("cancelled") => {
            append_view_message(
                view,
                MessageRole::Notice,
                format!("{provider} login cancelled."),
            );
        }
        Err(error) => append_view_message(view, MessageRole::Error, error),
    }
}

/// Enter while the composer is asking a login question.
fn answer_composer_prompt(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    answer: String,
) {
    let Some(flow) = view.login.as_mut() else {
        app.prompt = None;
        return;
    };
    let provider = flow.provider().to_owned();
    let oauth = flow.started().is_some();
    match flow.answer(answer) {
        Some(result) => finish_login(app, view, prepared, catalog, &provider, result, oauth),
        None => {
            app.prompt = flow
                .api_key_question()
                .map(|(label, secret)| state::ComposerPrompt::text(label, secret));
            if oauth {
                view.activity = format!("Signing in to {provider}");
            }
        }
    }
}

/// Esc while a login is running or asking.
fn cancel_composer_prompt(app: &mut App, view: &mut InteractiveView) {
    app.prompt = None;
    if let Some(mut flow) = view.login.take() {
        let provider = flow.provider().to_owned();
        flow.cancel();
        append_view_message(
            view,
            MessageRole::Notice,
            format!("{provider} login cancelled."),
        );
    }
    view.activity = "Ready".to_owned();
    view.activity_since = None;
}

fn submit_interactive_input<'a>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    catalog: &'a catalog::Catalog,
    turn_sender: Sender<Result<(), String>>,
    input: String,
    follow_up: bool,
) -> CommandDispatch<'a> {
    app.record_submission(&input);
    // The last command's "Model set to …" or "Transcript cleared" is not
    // what is happening any more once something new starts.
    if !app.streaming && !view.turn_pending && view.background.is_none() && view.login.is_none() {
        view.activity = "Ready".to_owned();
    }
    let typed = input.clone();
    let input = match prepared.expand_resource_input(&input) {
        Ok(Some(expanded)) => expanded,
        Ok(None) => input,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.to_string());
            return CommandDispatch::Handled;
        }
    };
    if input.starts_with('/') {
        if echoes_command(&input) {
            echo_command(view, &typed);
        }
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

    if !ready_for_prompt(app, view, prepared) {
        return CommandDispatch::Handled;
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

    start_interactive_prompt(view, prepared, turn_sender, input)
}

/// Whether a prompt may start: no login is waiting for an answer and a
/// model is selected. Says why not in the transcript.
fn ready_for_prompt(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
) -> bool {
    if let Some(flow) = view.login.as_ref() {
        append_view_message(
            view,
            MessageRole::Error,
            format!(
                "Finish the {} login first, or press Esc to cancel it.",
                flow.provider()
            ),
        );
        return false;
    }
    if !runtime::model_is_selected(&prepared.runtime.agent().state().model) {
        append_view_message(view, MessageRole::Error, NO_MODEL_PROMPT_REFUSED);
        app.set_input("/login ");
        return false;
    }
    true
}

/// Starts a turn for `input` on an idle agent.
fn start_interactive_prompt<'a>(
    view: &mut InteractiveView,
    prepared: &runtime::PreparedSession,
    turn_sender: Sender<Result<(), String>>,
    input: String,
) -> CommandDispatch<'a> {
    if let Err(error) = prepared.sync_extensions() {
        append_view_message(view, MessageRole::Error, error.to_string());
        return CommandDispatch::Handled;
    }
    begin_interactive_turn(
        view,
        prepared.runtime.agent().clone(),
        prepared.runtime.notice_sender(),
        turn_sender,
        input,
        "Starting response",
    );
    CommandDispatch::NotCommand
}

/// Set once the terminal accepted the kitty keyboard protocol flags, which
/// is what lets Shift-Enter arrive as something other than Enter.
static KEYBOARD_ENHANCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `/hotkeys` for the interface in use: line mode has none of the editor.
fn hotkeys_text(fullscreen: bool) -> String {
    if !fullscreen {
        return "Enter       send the line\n←/→         move in the line; Home/End or Ctrl-A/Ctrl-E jump to its ends\nUp/Down     recall earlier lines\nCtrl-U/K/W  delete to the start, to the end, or the previous word\nCtrl-C      abort the active response; at the prompt clear it, twice to exit\nCtrl-D      exit at an empty prompt"
            .to_owned();
    }
    let newline = if KEYBOARD_ENHANCED.load(std::sync::atomic::Ordering::Relaxed) {
        "Ctrl-J      insert a newline (Shift-Enter also works in this terminal)"
    } else {
        "Ctrl-J      insert a newline (Shift-Enter needs a terminal with the kitty keyboard protocol)"
    };
    format!(
        "Enter       send or accept selection; steer while a response is active\nAlt-Enter   queue a follow-up\n{newline}\nUp/Down     navigate palette, editor lines, or history\nAlt-←/→     move by word; Home/End or Ctrl-A/Ctrl-E move within a line\nCtrl-U/K/W  delete to the line's start, to its end, or the previous word\nTab         complete the selected command; Shift-Tab cycle thinking\nCtrl-L      open model selector; Ctrl-P cycle models; Ctrl-O expand tools\nCtrl-T      toggle displayed thinking\nPgUp/PgDn   scroll the transcript; Ctrl-Home/Ctrl-End jump to top/bottom\nEsc         close the palette, clear input (Up brings it back), or abort a response\nCtrl-C      abort, or quit (asks twice when the session is not saved)\nCtrl-D      quit when the editor is empty"
    )
}

/// Whether a typed command is echoed as a "◇ Command" card. Pickers and
/// their selections (`/model x`, `/thinking high`) announce their result in
/// a notice instead, and `/clear` and `/new` empty the screen anyway.
fn echoes_command(input: &str) -> bool {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    match command {
        "/model" | "/thinking" | "/clear" | "/new" | "/exit" | "/quit" => false,
        "/login" => !rest.trim().is_empty(),
        _ => true,
    }
}

/// A command the fullscreen interface must step aside for: it talks to the
/// user through the raw terminal (an OAuth login, onboarding prompts,
/// `$EDITOR`). The closure runs once the screen has been handed back.
type SuspendedCommand<'a> = Box<dyn FnOnce() -> Result<String, String> + 'a>;

enum CommandDispatch<'a> {
    NotCommand,
    Handled,
    Quit,
    Suspended(SuspendedCommand<'a>),
}

fn begin_interactive_turn(
    view: &mut InteractiveView,
    agent: agent::Agent,
    prepared_notices: session::SessionNoticeSender,
    turn_sender: Sender<Result<(), String>>,
    prompt: String,
    activity: &str,
) {
    view.turn_pending = true;
    view.activity = activity.to_owned();
    view.activity_since = Some(Instant::now());
    let notices = prepared_notices;
    thread::spawn(move || {
        let completion = TurnCompletion::new(turn_sender);
        let result = turns::run_prompt(
            &agent,
            prompt,
            &turns::RetryPolicy::default(),
            Some(&notices),
        )
        .map(|_| ());
        completion.finish(result);
    });
}

fn dispatch_ralph_slash_command<'a>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    turn_sender: Sender<Result<(), String>>,
    rest: &str,
) -> CommandDispatch<'a> {
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
                prepared.runtime.notice_sender(),
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
fn dispatch_btw_slash_command<'a>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    turn_sender: Sender<Result<(), String>>,
    rest: &str,
) -> CommandDispatch<'a> {
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
                    // "Bring to main" means into the main conversation: the
                    // context goes into the composer to edit and send, as
                    // the original extension does.
                    Ok(output) => {
                        app.set_input(&output.text);
                        append_view_message(
                            view,
                            MessageRole::Notice,
                            format!(
                                "Brought {} from {} (about {} tokens) into the editor. Edit it, then press Enter to send it to the main conversation.",
                                plural(output.segments.len(), "message", "messages"),
                                output.thread_id,
                                output.estimated_tokens
                            ),
                        );
                    }
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
    // Threads are saved as session entries, so only an unrecorded session
    // loses them on exit.
    let storage = if prepared.runtime.recording() {
        "saved with this session"
    } else {
        "kept in memory only; this session is not recorded"
    };
    let mut lines = vec![
        format!("BTW side threads ({storage}):"),
        "  /btw <question>                       start a fresh side thread".to_owned(),
        "  /btw resume <id> <question>           continue one".to_owned(),
        "  /btw bring <id> [latest|all|from:N]   show side context".to_owned(),
        "  /btw settings [level|remember]        view/change preferences".to_owned(),
    ];
    let threads = prepared.btw.list_threads();
    if threads.is_empty() {
        lines.push("No side threads yet.".to_owned());
    }
    for summary in threads {
        lines.push(format!(
            "  {}  {}  {}",
            summary.id,
            plural(summary.questions, "question", "questions"),
            summary.title
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
    turn_sender: Sender<Result<(), String>>,
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
        let completion = TurnCompletion::new(turn_sender);
        let result = match side_runtime.run_next(&state, &worker_thread_id) {
            Ok(Some(_)) => Ok(()),
            Ok(None) => Err("BTW side thread did not have a queued question".to_owned()),
            Err(error) => Err(error.to_string()),
        };
        completion.finish(result);
    });
}

/// Executes slash commands that can be served without leaving the fullscreen
/// Ratatui program. Commands with an unavailable integration report that fact
/// in the transcript rather than pretending they changed runtime state.
/// `/login [provider]`: adds an OAuth subscription or an API key without
/// replacing any other stored credential. The prompts run in a child process
/// that owns the terminal, exactly as the previous interface did.
fn dispatch_login_command<'a>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    catalog: &'a catalog::Catalog,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch<'a> {
    if rest.is_empty() {
        if fullscreen {
            // Bare `/login` opens the provider picker; the argument palette
            // lists every provider with its login method.
            app.set_input("/login ");
            return CommandDispatch::Handled;
        }
        append_view_message(
            view,
            MessageRole::Command,
            format!(
                "Usage: /login <provider> [key]\nBrowser or device sign-in: {}\nAdd `key` to store an API key instead; other providers always prompt for one. Existing credentials are preserved.",
                oauth::implemented_provider_ids().join(", ")
            ),
        );
        return CommandDispatch::Handled;
    }
    let fields = rest.split_whitespace().collect::<Vec<_>>();
    let (provider_id, use_key) = match fields.as_slice() {
        [provider_id] => (provider_id, false),
        [provider_id, "key" | "api-key" | "apikey"] => (provider_id, true),
        _ => {
            append_view_message(view, MessageRole::Error, "usage: /login <provider> [key]");
            return CommandDispatch::Handled;
        }
    };
    if catalog.provider(provider_id).is_none() {
        append_view_message(
            view,
            MessageRole::Error,
            format!("unknown provider {provider_id:?}"),
        );
        return CommandDispatch::Handled;
    }
    if let Some(setup) = gateway_setup_command(provider_id) {
        append_view_message(
            view,
            MessageRole::Command,
            format!(
                "{provider_id} is a gateway, not an API-key provider; run {setup} to configure it."
            ),
        );
        return CommandDispatch::Handled;
    }
    let provider_id = (*provider_id).to_owned();
    if use_key && !api_key_login_available(catalog, &provider_id) {
        let (key_provider, key_api) = api_key_alternative(&provider_id);
        append_view_message(
            view,
            MessageRole::Error,
            format!(
                "{provider_id} has no API key; use /login {provider_id}, or /login {key_provider} key for the {key_api}"
            ),
        );
        return CommandDispatch::Handled;
    }
    let oauth = login_flow_available(&provider_id) && !use_key;
    if fullscreen {
        if view.login.is_some() {
            append_view_message(
                view,
                MessageRole::Error,
                "A login is already in progress; finish it or press Esc to cancel it.",
            );
            return CommandDispatch::Handled;
        }
        // The login runs inside the interface: notices in the transcript,
        // questions in the composer, nothing printed over the screen.
        let store = catalog::CredentialStore::default_file();
        if oauth {
            match tui_login::LoginFlow::start_oauth(&provider_id, store) {
                Ok(flow) => {
                    view.activity = format!("{provider_id} login");
                    view.activity_since = Some(Instant::now());
                    view.login = Some(flow);
                }
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
        } else {
            let flow = tui_login::LoginFlow::start_api_key(
                &provider_id,
                provider_cli::cloudflare_credential_fields(&provider_id),
                store,
            );
            append_view_message(
                view,
                MessageRole::Notice,
                format!(
                    "Paste the API key for {provider_id} and press Enter; Esc cancels. It is stored in {} (mode 0600) and not shown on screen.",
                    home_relative(&config::auth_path().display().to_string())
                ),
            );
            app.prompt = flow
                .api_key_question()
                .map(|(label, secret)| state::ComposerPrompt::text(label, secret));
            view.login = Some(flow);
        }
        return CommandDispatch::Handled;
    }
    let subcommand = if oauth { "login" } else { "set" };
    CommandDispatch::Suspended(Box::new(move || {
        run_self_subprocess(&["auth", subcommand, &provider_id])?;
        catalog.clear_oauth_refresh_failure(&provider_id);
        // A Grok CLI login is what makes `image_gen` available.
        prepared.sync_image_tool();
        Ok(after_login_message(prepared, catalog, &provider_id))
    }))
}

/// `/grok-cli-imagine:tool [on|off|status]`: the persisted `image_gen`
/// switch. With no argument it toggles, as upstream does.
fn image_tool_command(prepared: &runtime::PreparedSession, rest: &str) -> Result<String, String> {
    let argument = rest.trim().to_ascii_lowercase();
    if !matches!(argument.as_str(), "" | "on" | "off" | "status") {
        return Err("Usage: /grok-cli-imagine:tool [on|off|status]".to_owned());
    }
    let path = prepared.imagine_config_path();
    let loaded = grok_imagine::load_config(&path);
    let mut lines = loaded.warning.into_iter().collect::<Vec<_>>();
    let on_off = |value: bool| if value { "on" } else { "off" };
    if argument == "status" {
        lines.push(format!(
            "image_gen persisted: {}; active: {}",
            on_off(loaded.enabled),
            on_off(prepared.image_tool_active())
        ));
        return Ok(lines.join("\n"));
    }
    let enabled = if argument.is_empty() {
        !loaded.enabled
    } else {
        argument == "on"
    };
    grok_imagine::save_config(&path, enabled)
        .map_err(|error| format!("Could not save image_gen setting: {error}"))?;
    let active = prepared.sync_image_tool();
    lines.push(format!("image_gen: {}", on_off(enabled)));
    if enabled && !active {
        lines.push(
            "It becomes available once Grok CLI is signed in (/login grok-cli) and tools are on."
                .to_owned(),
        );
    }
    Ok(lines.join("\n"))
}

/// The in-chat command that configures a gateway provider; `/login` would
/// only store a key that the gateway never reads.
fn gateway_setup_command(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        "aperture" => Some("/aperture onboarding"),
        "omni" => Some("/omni setup"),
        _ => None,
    }
}

/// Puts a freshly authenticated provider to use. A session that has no model
/// yet switches to the provider's curated model straight away, so the first
/// login is the whole onboarding; a provider without a curated model is left
/// to the picker, which the event loop opens; an existing selection is left
/// alone.
fn after_login_message(
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    provider_id: &str,
) -> String {
    if !catalog.is_configured(provider_id).unwrap_or(false) {
        let hint = catalog
            .provider(provider_id)
            .map(|provider| provider_cli::provider_setup_hint(&provider))
            .unwrap_or_default();
        return format!(
            "Stored a credential for {provider_id}, but the provider is still not usable: {hint}"
        );
    }
    if runtime::model_is_selected(&prepared.runtime.agent().state().model) {
        return "Use /model or ctrl+l to switch to one of its models.".to_owned();
    }
    let Some(reference) = runtime::curated_model_reference(catalog, &[provider_id.to_owned()])
    else {
        return format!("Pick one of the {provider_id} models to start.");
    };
    match runtime::set_model(&prepared.runtime, catalog, &reference) {
        Ok(model) => format!(
            "Model set to {}/{}. Use /model or ctrl+l to switch.",
            model.provider, model.id
        ),
        Err(error) => format!("{reference} could not be selected: {error}. Pick a model to start."),
    }
}

/// Whether a provider with a login flow also takes an API key. Subscription
/// providers (a ChatGPT plan, Grok CLI, Meta Muse Code, which refuses any key
/// Meta does not confirm as subscription-backed) have none.
fn api_key_login_available(catalog: &catalog::Catalog, provider_id: &str) -> bool {
    catalog
        .provider(provider_id)
        .is_some_and(|provider| provider.auth_kind != catalog::AuthKind::OAuthOnly)
}

/// The provider that takes an API key for the same models as a
/// subscription-only one, and what that key is called.
fn api_key_alternative(provider_id: &str) -> (&'static str, &'static str) {
    match provider_id {
        "meta-muse" => ("meta", "Meta Model API"),
        "grok-cli" => ("xai", "xAI API"),
        _ => ("openai", "OpenAI API"),
    }
}

fn login_flow_available(provider_id: &str) -> bool {
    oauth::OAuthProviderId::parse(provider_id).is_some_and(|provider| {
        oauth::metadata_for(provider).flow_support != oauth::OAuthFlowSupport::MetadataOnly
    })
}

/// `/omni [status|setup|sync|models|test|dashboard|config|help]`. Setup prompts for a URL and key,
/// so it owns the terminal; the rest talk to the gateway off the UI thread.
fn dispatch_omni_command<'a>(
    view: &mut InteractiveView,
    catalog: &'a catalog::Catalog,
    rest: &str,
) -> CommandDispatch<'a> {
    let arguments = rest
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if let Err(error) = omniroute::CliCommand::parse(&arguments) {
        append_view_message(view, MessageRole::Error, error.to_string());
        return CommandDispatch::Handled;
    }
    if omni_cli::needs_terminal(&arguments) {
        return CommandDispatch::Suspended(Box::new(move || {
            run_self_subprocess(&["omni", "setup"])?;
            catalog.refresh_dynamic();
            Ok("OmniRoute setup finished; its models are available under /model. Use /omni status to verify the gateway.".to_owned())
        }));
    }
    let label = format!("/omni {rest}").trim_end().to_owned();
    let catalog = catalog.clone();
    start_background_command(view, &label, move || {
        let output = omni_cli::execute(&arguments).map_err(|error| error.to_string());
        catalog.refresh_dynamic();
        output
    });
    CommandDispatch::Handled
}

/// `/aperture [subcommand]`. Onboarding prompts through the terminal; every
/// other subcommand runs off the UI thread and the catalog re-reads the
/// gateway state once it finishes.
fn dispatch_aperture_command<'a>(
    view: &mut InteractiveView,
    catalog: &'a catalog::Catalog,
    rest: &str,
) -> CommandDispatch<'a> {
    let arguments = rest
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if aperture_cli::needs_terminal(&arguments) {
        return CommandDispatch::Suspended(Box::new(move || {
            let mut child_arguments = vec!["aperture"];
            child_arguments.extend(arguments.iter().map(String::as_str));
            run_self_subprocess(&child_arguments)?;
            catalog.refresh_dynamic();
            Ok("Aperture onboarding finished.".to_owned())
        }));
    }
    let label = format!("/aperture {rest}").trim_end().to_owned();
    let catalog = catalog.clone();
    start_background_command(view, &label, move || {
        let output = aperture_cli::execute(&arguments, false)
            .map(|text| in_chat_terms(&text))
            .map_err(|error| in_chat_terms(&error.to_string()));
        catalog.refresh_dynamic();
        output
    });
    CommandDispatch::Handled
}

/// The Aperture module's text names its shell commands; inside chat the
/// same command is a slash command away, so that is what it should say.
fn in_chat_terms(text: &str) -> String {
    text.replace("`goshcoder aperture ", "`/aperture ")
}

fn dispatch_runtime_slash_command<'a>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    catalog: &'a catalog::Catalog,
    turn_sender: Sender<Result<(), String>>,
    input: &str,
    fullscreen: bool,
) -> CommandDispatch<'a> {
    let (command, rest) = input.split_once(' ').unwrap_or((input, ""));
    let rest = rest.trim();
    // pi's extension registers `/aperture:onboarding`-style commands as well
    // as the spaced form; both reach the same handler.
    let aperture_alias;
    let (command, rest) = match command.strip_prefix("/aperture:") {
        Some(subcommand) => {
            aperture_alias = format!("{subcommand} {rest}");
            ("/aperture", aperture_alias.trim())
        }
        None => (command, rest),
    };
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
                "Slash commands:\n  /help                 Show this help\n  /model [ref]          Open the model picker, or switch to provider/model\n  /thinking [level]     List or choose reasoning effort\n  /tools                List active tools\n  /status, /session     Show live session information\n  /messages             Show transcript summary\n  /queue                Show queued steering/follow-up messages\n  /steer <text>         Guide an active response\n  /followup <text>      Queue the next turn\n  /clear, /new          Reset this transcript\n  /compact [focus]      Summarize older context and keep recent turns\n  /name <text>          Set the persisted session name\n  /sessions             List saved sessions\n  /resume <id>          Switch to a saved session\n  /tree, /fork, /label  Inspect or rewind saved-session branches\n  /clone                Duplicate the current saved session\n  /export [path]        Save this session as HTML (.md or .jsonl by extension)\n  /import <path>        Adopt a session file and switch to it\n  /share [confirm]      Upload this session as a secret GitHub gist\n  /prompt <action>      List, save, edit, remove, back up, or restore prompts\n  /reload               Reload local context, prompts, and skills\n  /resources            Show loaded context, prompts, and skills\n  /ralph <subcommand>   Manage Ralph loops\n  /planner              Toggle planning mode\n  /planner-review [URL] Review local changes or a GitHub PR\n  /planner-annotate <target>\n                        Annotate a file, folder, or URL\n  /planner-last         Annotate the latest assistant response\n  /login [provider]     Open the provider picker, or log in to one (keeps existing logins)\n  /grok-cli-usage       Show the Grok CLI subscription's weekly usage\n  /grok-cli-imagine <prompt> [--image <path>] [--aspect <r>] [--out <path>]\n                        Generate or edit an image with Grok Imagine\n  /grok-cli-imagine:tool [on|off|status]\n                        Offer the image_gen tool to the model, or not\n  /grok-cli-accounts [list|use|add|login|logout|rename|remove]\n                        Manage several Grok CLI accounts\n  /grok-cli-conv [status|rotate]\n                        Show or rotate the Grok CLI conversation ID\n  /omni [command]       Set up, sync, or inspect an OmniRoute gateway\n  /aperture [command]   Manage a Tailscale Aperture gateway\n  /btw <question>       Ask a side question without touching the transcript\n  /hotkeys              Show keyboard shortcuts\n  /exit                 Leave chat"
                    .to_owned(),
            );
            CommandDispatch::Handled
        }
        "/hotkeys" => {
            append_view_message(view, MessageRole::Command, hotkeys_text(fullscreen));
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
                    // The old conversation's notices go with it, so the
                    // screen really is empty apart from this line.
                    reset_view_transcript(view, prepared);
                    app.scroll_to_bottom();
                    view.activity = "Ready".to_owned();
                    append_view_message(view, MessageRole::Notice, "Started a fresh conversation.");
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
            if choices.is_empty() {
                append_view_message(view, MessageRole::Command, NO_MODEL_PICKER_EMPTY);
                if fullscreen {
                    app.set_input("/login ");
                }
            } else if fullscreen {
                // Bare `/model` opens the picker, as it does in pi; the
                // argument palette lists every authenticated model.
                app.set_input("/model ");
            } else {
                append_view_message(
                    view,
                    MessageRole::Command,
                    format!("Available models:\n{}", choices.join("\n")),
                );
            }
            CommandDispatch::Handled
        }
        "/model" => {
            match runtime::set_model(&prepared.runtime, catalog, rest) {
                Ok(model) => {
                    view.activity = format!("Model set to {}/{}", model.provider, model.id);
                    append_view_message(
                        view,
                        MessageRole::Notice,
                        format!("Model set to {}/{}.", model.provider, model.id),
                    );
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
        }
        "/thinking" if rest.is_empty() && fullscreen => {
            // Like `/model`, the bare command opens its picker.
            app.set_input("/thinking ");
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
            let queued = prepared
                .runtime
                .agent()
                .queued_messages()
                .iter()
                .filter_map(|message| match message {
                    llm::Message::User(user) => Some(user_message_text(user)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            append_view_message(view, MessageRole::Command, queue_report(&queued));
            CommandDispatch::Handled
        }
        "/steer" if rest.is_empty() => {
            append_view_message(view, MessageRole::Error, "usage: /steer <text>");
            CommandDispatch::Handled
        }
        // With nothing running there is nothing to steer or follow: the
        // text is simply the next prompt, instead of waiting in a queue
        // that the next prompt (or a compaction) would surprise.
        "/steer" | "/followup"
            if !rest.is_empty()
                && !app.streaming
                && !view.turn_pending
                && !prepared.runtime.agent().state().is_streaming =>
        {
            if !ready_for_prompt(app, view, prepared) {
                return CommandDispatch::Handled;
            }
            start_interactive_prompt(view, prepared, turn_sender, rest.to_owned())
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
                    render_session_tree(&points)
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
                    reset_view_transcript(view, prepared);
                    app.scroll_to_bottom();
                    if point.on_path {
                        // pi hands the rewound message back for editing
                        // rather than leaving it as a turn without a reply.
                        if fullscreen {
                            app.set_input(&point.prompt);
                        }
                        view.activity = format!("Rewound to before {}", first_line(&point.text));
                        append_view_message(
                            view,
                            MessageRole::Notice,
                            if fullscreen {
                                "Rewound. The message is back in the editor: edit it and press Enter to take the conversation another way. /tree lists every branch."
                            } else {
                                "Rewound. Send the message again (or a new one) to take the conversation another way. /tree lists every branch."
                            },
                        );
                    } else {
                        view.activity =
                            format!("Back on the branch at {}", first_line(&point.text));
                        append_view_message(
                            view,
                            MessageRole::Notice,
                            format!("Returned to the branch at \"{}\".", first_line(&point.text)),
                        );
                    }
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
        "/export" => dispatch_export_command(view, prepared, rest),
        "/import" => dispatch_import_command(app, view, prepared, rest),
        "/share" => dispatch_share_command(view, prepared, rest),
        "/clone" => {
            match prepared.runtime.clone_session() {
                Ok(handle) => {
                    // The runtime reports "cloned as <id>" itself.
                    view.activity = format!("Cloned session {}", short_id(&handle.id));
                }
                Err(error) => append_view_message(view, MessageRole::Error, error.to_string()),
            }
            CommandDispatch::Handled
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
            match list_interactive_sessions(prepared) {
                Ok(output) => append_view_message(
                    view,
                    MessageRole::Command,
                    format!("{output}\n/resume <id> switches to a listed session."),
                ),
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/resume" => {
            match prepared.runtime.switch_to(rest) {
                Ok(handle) => {
                    // The previous session's notices stay with it; the
                    // runtime's own "switched to" notice is the one
                    // line this needs.
                    reset_view_transcript(view, prepared);
                    app.scroll_to_bottom();
                    view.activity = format!("Resumed session {}", short_id(&handle.id));
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
            let notices = prepared.runtime.notice_sender();
            let instructions = rest.to_owned();
            view.turn_pending = true;
            view.activity = "Compacting context".to_owned();
            view.activity_since = Some(Instant::now());
            thread::spawn(move || {
                let completion = TurnCompletion::new(turn_sender);
                let result = compaction::compact(&agent, &instructions)
                    .map(|outcome| turns::report_dropped_queue(Some(&notices), Some(&outcome)))
                    .map_err(|error| error.to_string());
                completion.finish(result);
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
        "/login" => dispatch_login_command(app, view, prepared, catalog, rest, fullscreen),
        "/grok-cli-imagine" => {
            if rest.is_empty() {
                append_view_message(
                    view,
                    MessageRole::Error,
                    "Usage: /grok-cli-imagine <prompt> [--image|--edit <path>] [--aspect <ratio>] [--out|-o <path>]",
                );
                return CommandDispatch::Handled;
            }
            let context = prepared.imagine_context();
            let arguments = rest.to_owned();
            start_background_command(view, "/grok-cli-imagine", move || {
                grok_imagine::run_command(&context, &arguments).map(|lines| lines.join("\n"))
            });
            CommandDispatch::Handled
        }
        "/grok-cli-imagine:tool" => {
            match image_tool_command(prepared, rest) {
                Ok(message) => append_view_message(view, MessageRole::Command, message),
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/grok-cli-usage" => {
            let catalog = catalog.clone();
            let agent_dir = catalog
                .dynamic_paths()
                .agent_dir
                .clone()
                .unwrap_or_else(config::agent_dir);
            let session = prepared.request_session_id();
            start_background_command(view, "/grok-cli-usage", move || {
                Ok(grok_cli::usage_report(&catalog, &agent_dir, &session).join("\n\n"))
            });
            CommandDispatch::Handled
        }
        "/grok-cli-accounts" => {
            let accounts = grok_accounts::Accounts::new(catalog);
            match grok_accounts::chat_command(&accounts, &prepared.request_session_id(), rest) {
                Ok(grok_accounts::AccountsCommand::Done(message)) => {
                    append_view_message(view, MessageRole::Command, message);
                }
                Ok(grok_accounts::AccountsCommand::Terminal(arguments)) => {
                    // The OAuth login owns the terminal, as /login's does.
                    return CommandDispatch::Suspended(Box::new(move || {
                        let mut child = vec!["grok-cli"];
                        child.extend(arguments.iter().map(String::as_str));
                        run_self_subprocess(&child)?;
                        catalog.clear_oauth_refresh_failure(grok_cli::PROVIDER_ID);
                        prepared.sync_image_tool();
                        Ok("Grok CLI accounts updated; /grok-cli-accounts lists them.".to_owned())
                    }));
                }
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/grok-cli-conv" => {
            match grok_cli::conv_command(&prepared.request_session_id(), rest) {
                Ok(message) => append_view_message(view, MessageRole::Command, message),
                Err(error) => append_view_message(view, MessageRole::Error, error),
            }
            CommandDispatch::Handled
        }
        "/omni" => dispatch_omni_command(view, catalog, rest),
        "/aperture" => dispatch_aperture_command(view, catalog, rest),
        _ if command.starts_with('/') => {
            append_view_message(
                view,
                MessageRole::Error,
                format!("unknown command {command}; /help lists the available commands"),
            );
            CommandDispatch::Handled
        }
        _ => CommandDispatch::NotCommand,
    }
}

/// Handles `/prompt` and its compatibility alias without writing directly to
/// the terminal. This keeps prompt management usable in both line mode and
/// the Ratatui alternate screen.
fn dispatch_prompt_slash_command<'a>(
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    rest: &str,
    fullscreen: bool,
) -> CommandDispatch<'a> {
    match prompt_slash_command(prepared, rest, fullscreen) {
        Ok(PromptOutcome::Text(output)) if !output.is_empty() => {
            append_view_message(view, MessageRole::Command, output);
        }
        Ok(PromptOutcome::Text(_)) => {}
        Ok(PromptOutcome::Suspend(run)) => return CommandDispatch::Suspended(run),
        Err(error) => append_view_message(view, MessageRole::Error, error),
    }
    CommandDispatch::Handled
}

/// What a `/prompt` action produced: text to show, or an editor session the
/// fullscreen interface must hand the terminal to first.
enum PromptOutcome<'a> {
    Text(String),
    Suspend(SuspendedCommand<'a>),
}

fn prompt_slash_command<'a>(
    prepared: &'a runtime::PreparedSession,
    rest: &str,
    fullscreen: bool,
) -> Result<PromptOutcome<'a>, String> {
    let (action, argument) = split_prompt_action(rest);
    let text = match action {
        "" | "list" => prompt_list(prepared),
        "save" => prompt_save(prepared, argument),
        "rm" | "remove" | "delete" => prompt_remove(prepared, argument),
        "edit" => return prompt_edit(prepared, argument, fullscreen),
        "backup" => prompt_backup(prepared, argument),
        "restore" => prompt_restore(prepared, argument),
        action => Err(format!(
            "unknown /prompt action {action:?}; use list, save, edit, rm, backup or restore"
        )),
    }?;
    Ok(PromptOutcome::Text(text))
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
            // Two spaces, not four: four mark a line to stand out (a
            // sign-in URL), which a template's path is not.
            format!(
                "/{}{}\n  {}",
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

fn prompt_edit<'a>(
    prepared: &'a runtime::PreparedSession,
    argument: &str,
    fullscreen: bool,
) -> Result<PromptOutcome<'a>, String> {
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
        return Ok(PromptOutcome::Text(format!(
            "set $EDITOR to edit in place; the file is {}",
            path.display()
        )));
    };
    let mut fields = editor.split_whitespace().map(str::to_owned);
    let program = fields
        .next()
        .ok_or_else(|| "$EDITOR does not contain an executable".to_owned())?;
    let arguments = fields.collect::<Vec<_>>();
    let run = move || -> Result<String, String> {
        let status = std::process::Command::new(&program)
            .args(&arguments)
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
    };
    if fullscreen {
        // The editor needs the real terminal; the event loop steps aside.
        Ok(PromptOutcome::Suspend(Box::new(run)))
    } else {
        run().map(PromptOutcome::Text)
    }
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
        "grok-cli-accounts",
        "grok-cli-conv",
        "grok-cli-imagine",
        "grok-cli-imagine:tool",
        "grok-cli-usage",
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
        "share",
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

/// The session store and workspace the chat commands resolve sessions in.
fn interactive_session_store(
    prepared: &runtime::PreparedSession,
) -> Result<(sessionlog::Store, std::path::PathBuf), String> {
    let cwd =
        runtime::absolute_workdir(&prepared.config.workdir).map_err(|error| error.to_string())?;
    let store = sessionlog::Store::new(
        prepared
            .config
            .sessions_dir
            .clone()
            .unwrap_or_else(config::sessions_dir),
    );
    Ok((store, cwd))
}

/// The saved session behind this chat, or why there is none to export.
fn current_session_info(
    prepared: &runtime::PreparedSession,
) -> Result<
    (
        sessionlog::Store,
        std::path::PathBuf,
        sessionlog::SessionInfo,
    ),
    String,
> {
    let Some(path) = prepared.runtime.path() else {
        return Err(
            "Nothing to export: this session is not being saved (chat started with -no-session)."
                .to_owned(),
        );
    };
    let (store, cwd) = interactive_session_store(prepared)?;
    let info = store
        .resolve(&cwd, &path.to_string_lossy())
        .map_err(|error| error.to_string())?;
    Ok((store, cwd, info))
}

/// `/export [path]`: `.jsonl` keeps the lossless log, `.md` writes Markdown,
/// anything else (and no path at all) writes the self-contained HTML page
/// into the workspace, as pi's `/export` does.
fn dispatch_export_command<'a>(
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    rest: &str,
) -> CommandDispatch<'a> {
    let outcome = current_session_info(prepared).and_then(|(store, cwd, info)| {
        let destination = if rest.trim().is_empty() {
            cwd.join(sessions::default_export_name(
                &info,
                sessions::ExportFormat::Html,
            ))
        } else {
            let requested = config::expand_tilde(rest.trim());
            if requested.is_absolute() {
                requested
            } else {
                cwd.join(requested)
            }
        };
        let format = sessions::ExportFormat::for_destination(&destination);
        sessions::export_to_file(&store, &info, format, &destination)
            .map(|()| destination)
            .map_err(|error| error.to_string())
    });
    match outcome {
        Ok(destination) => {
            view.activity = "Session exported".to_owned();
            append_view_message(
                view,
                MessageRole::Notice,
                format!("Session exported to: {}", destination.display()),
            );
        }
        Err(error) => append_view_message(view, MessageRole::Error, error),
    }
    CommandDispatch::Handled
}

/// `/import <path.jsonl>`: adopts a session file into the store and switches
/// to the copy, leaving the current session on disk to resume later.
fn dispatch_import_command<'a>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    rest: &str,
) -> CommandDispatch<'a> {
    let source = rest.trim();
    if source.is_empty() {
        append_view_message(view, MessageRole::Error, "usage: /import <path.jsonl>");
        return CommandDispatch::Handled;
    }
    let outcome = interactive_session_store(prepared).and_then(|(store, cwd)| {
        let requested = config::expand_tilde(source);
        let requested = if requested.is_absolute() {
            requested
        } else {
            cwd.join(requested)
        };
        let info = store
            .resolve(&cwd, &requested.to_string_lossy())
            .map_err(|error| format!("read {source}: {error}"))?;
        let mut writer = store
            .fork(&info, None, &cwd)
            .map_err(|error| error.to_string())?;
        let id = writer.id().to_owned();
        writer.close().map_err(|error| error.to_string())?;
        prepared.runtime.switch_to(&id).map_err(|error| {
            format!(
                "imported as {} but could not switch: {error}",
                short_id(&id)
            )
        })
    });
    match outcome {
        Ok(handle) => {
            reset_view_transcript(view, prepared);
            app.scroll_to_bottom();
            view.activity = format!("Imported session {}", short_id(&handle.id));
            append_view_message(
                view,
                MessageRole::Notice,
                format!(
                    "Session imported from {source} as {} and switched to it.",
                    short_id(&handle.id)
                ),
            );
        }
        Err(error) => append_view_message(view, MessageRole::Error, error),
    }
    CommandDispatch::Handled
}

/// `/share`: explains what would leave the machine; `/share confirm` uploads
/// the HTML export as a secret gist through `gh`, off the UI thread.
fn dispatch_share_command<'a>(
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    rest: &str,
) -> CommandDispatch<'a> {
    let (store, _cwd, info) = match current_session_info(prepared) {
        Ok(found) => found,
        Err(error) => {
            append_view_message(view, MessageRole::Error, error.replace("export", "share"));
            return CommandDispatch::Handled;
        }
    };
    if let Some(refusal) = sessions::share_refusal(&info) {
        append_view_message(view, MessageRole::Error, refusal);
        return CommandDispatch::Handled;
    }
    match rest.trim().to_ascii_lowercase().as_str() {
        "" => append_view_message(
            view,
            MessageRole::Notice,
            format!(
                "{}\nType /share confirm to upload.",
                sessions::share_warning(&info)
            ),
        ),
        "confirm" | "yes" => {
            start_background_command(view, "/share", move || {
                sessions::share_session(&store, &info)
                    .map(|outcome| outcome.render())
                    .map_err(|error| error.to_string())
            });
        }
        _ => append_view_message(view, MessageRole::Error, "usage: /share [confirm]"),
    }
    CommandDispatch::Handled
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
    Ok(render_interactive_session_list(
        &sessions,
        prepared.runtime.id().as_deref(),
    ))
}

/// The `/resume ` palette: this workspace's other saved sessions, newest
/// first, so a session can be picked instead of its id remembered.
fn resume_choices(prepared: &runtime::PreparedSession) -> Vec<state::Suggestion> {
    let Ok(cwd) = runtime::absolute_workdir(&prepared.config.workdir) else {
        return Vec::new();
    };
    let store = sessionlog::Store::new(
        prepared
            .config
            .sessions_dir
            .clone()
            .unwrap_or_else(config::sessions_dir),
    );
    let Ok(sessions) = session_picker::list_sessions_for_picker(&store, &cwd) else {
        return Vec::new();
    };
    let current = prepared.runtime.id();
    let labels = sessionlog::short_ids(&sessions);
    sessions
        .iter()
        .zip(labels)
        .filter(|(session, _)| current.as_deref() != Some(session.id.as_str()))
        .map(|(session, label)| {
            let (label, description) =
                session_picker::describe_session(session, &label, false, false);
            state::Suggestion {
                value: format!("/resume {label}"),
                label,
                description,
                execute: true,
            }
        })
        .collect()
}

fn render_interactive_session_list(
    sessions: &[sessionlog::SessionInfo],
    current: Option<&str>,
) -> String {
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
        let is_current = current == Some(session.id.as_str());
        let (label, description) =
            session_picker::describe_session(session, &label, false, is_current);
        lines.push(format!("{label}  {description}"));
    }
    lines.join("\n")
}

fn start_planner_review<'a, F>(
    app: &mut App,
    view: &mut InteractiveView,
    prepared: &'a runtime::PreparedSession,
    turn_sender: Sender<Result<(), String>>,
    activity: &str,
    request: F,
) -> CommandDispatch<'a>
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
        let completion = TurnCompletion::new(turn_sender);
        let result = request().and_then(|(subject, request)| {
            let decision = review.review(&request).map_err(|error| error.to_string())?;
            if let Some(feedback) = plannotator::review_feedback_prompt(&subject, &decision) {
                agent.prompt(feedback).map_err(|error| error.to_string())
            } else {
                review.notify(format!("{subject} approved"));
                Ok(())
            }
        });
        completion.finish(result);
    });
    CommandDispatch::Handled
}

/// Adds command output or a notice at the current end of the transcript.
/// Command output shows as a notice: the "◇ Command" card is the echo of
/// what was typed ([`echo_command`]).
fn append_view_message(view: &mut InteractiveView, role: MessageRole, text: impl Into<String>) {
    let role = match role {
        MessageRole::Command => MessageRole::Notice,
        role => role,
    };
    push_notice(
        view,
        Message {
            role,
            text: text.into(),
            is_error: role == MessageRole::Error,
            ..Message::default()
        },
    );
}

/// Shows a typed slash command as a "◇ Command" card, so its output below it
/// reads as an answer to something.
fn echo_command(view: &mut InteractiveView, input: &str) {
    push_notice(
        view,
        Message {
            role: MessageRole::Command,
            text: input.to_owned(),
            ..Message::default()
        },
    );
}

fn push_notice(view: &mut InteractiveView, message: Message) {
    view.notices.push(AnchoredNotice {
        anchor: view.message_count,
        message,
    });
    // Bounded so a long session of commands does not grow without limit;
    // the oldest are the ones furthest up the transcript.
    const MAX_NOTICES: usize = 200;
    if view.notices.len() > MAX_NOTICES {
        view.notices.drain(..view.notices.len() - MAX_NOTICES);
    }
}

/// Forgets the notices of the conversation being left (`/clear`, `/new`,
/// `/resume`, `/fork`), so they do not trail into the next one.
fn reset_view_transcript(view: &mut InteractiveView, prepared: &runtime::PreparedSession) {
    view.notices.clear();
    view.retry = None;
    view.recent_tool.clear();
    view.message_count = prepared.runtime.agent().state().messages.len();
}

/// Interleaves agent messages and anchored notices in the order they
/// happened. A notice anchored past the end (the transcript was compacted or
/// rewound underneath it) stays at the end.
fn splice_transcript(agent: Vec<(usize, Message)>, notices: &[AnchoredNotice]) -> Vec<Message> {
    let mut result = Vec::with_capacity(agent.len() + notices.len());
    let mut pending = notices.iter().peekable();
    for (index, message) in agent {
        while let Some(notice) = pending.next_if(|notice| notice.anchor <= index) {
            result.push(notice.message.clone());
        }
        result.push(message);
    }
    result.extend(pending.map(|notice| notice.message.clone()));
    result
}

fn refresh_runtime_app(
    app: &mut App,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    view: &mut InteractiveView,
) {
    let state = prepared.runtime.agent().state();
    view.message_count = state.messages.len();
    app.dynamic_suggestions = palette_suggestions(app, prepared, catalog, &state, view);
    // Cloning the resource set is not free; only the slash palette needs it.
    app.command_suggestions = if app.input.starts_with('/') {
        resource_command_suggestions(prepared)
    } else {
        Vec::new()
    };
    let mut messages = agent_messages(&state.messages);
    if let Some(message) = state.streaming_message.as_ref() {
        let index = state.messages.len();
        messages.extend(
            agent_messages(std::slice::from_ref(message))
                .into_iter()
                .map(|(_, mut message)| {
                    message.streaming = message.role == MessageRole::Assistant;
                    (index, message)
                }),
        );
    }
    let mut messages = splice_transcript(messages, &view.notices);
    for message in &mut messages {
        if message.role == MessageRole::User {
            message.text = summarize_large_pastes(&message.text, &app.pasted_blocks);
        }
    }
    app.replace_messages(messages);
    // A browser or device login is waited on like a reply: spinner, Esc
    // to abort.
    app.streaming = state.is_streaming
        || view.turn_pending
        || view
            .login
            .as_ref()
            .is_some_and(|flow| flow.started().is_some());
    app.set_recording_active(prepared.runtime.recording());
    app.title = session_title(prepared).unwrap_or_else(|| "interactive session".to_owned());
    app.queued = prepared
        .runtime
        .agent()
        .queued_messages()
        .iter()
        .filter_map(|message| match message {
            llm::Message::User(user) => Some(user_message_text(user)),
            _ => None,
        })
        .collect();
    view.queued_count = app.queued.len();
    app.status = interactive_status(view, &state, app.streaming || view.background.is_some());
    let context = context_usage(&state);
    app.context_hint = format!("{}% context", context.percent);
    app.sidebar = runtime_sidebar(prepared, &state, view, &context);
}

/// The session's name, else its first message. The runtime's title falls
/// back to the session id, which says nothing to a person, so that case is
/// left to the caller's placeholder.
fn session_title(prepared: &runtime::PreparedSession) -> Option<String> {
    let id = prepared.runtime.id();
    prepared
        .runtime
        .name()
        .or_else(|| prepared.runtime.title())
        .filter(|title| Some(title) != id.as_ref())
        .map(|title| first_line(&title))
}

/// Replaces a large pasted block with a one-line marker for display; the
/// message itself, as the model received it, is unchanged.
fn summarize_large_pastes(text: &str, pasted: &[String]) -> String {
    let mut text = text.to_owned();
    for block in pasted {
        let trimmed = block.trim();
        if !trimmed.is_empty() && text.contains(trimmed) {
            let marker = format!("[pasted {} lines]", trimmed.lines().count());
            text = text.replacen(trimmed, &marker, 1);
        }
    }
    text
}

/// Prompt templates and invocable skills as slash-palette entries.
fn resource_command_suggestions(prepared: &runtime::PreparedSession) -> Vec<state::Suggestion> {
    let resources = prepared.resources();
    let templates = resources.templates.iter().map(|template| {
        let description = if template.description.is_empty() {
            "Prompt template".to_owned()
        } else {
            template.description.clone()
        };
        let description = if template.argument_hint.is_empty() {
            description
        } else {
            format!("{description} · {}", template.argument_hint)
        };
        state::Suggestion {
            label: format!("/{}", template.name),
            description,
            value: format!("/{} ", template.name),
            execute: false,
        }
    });
    // A skill hidden from the model (`disable-model-invocation`) is still
    // the user's to run, so every skill is listed.
    let skills = resources.skills.iter().map(|skill| state::Suggestion {
        label: format!("/skill:{}", skill.name),
        description: if skill.description.is_empty() {
            "Skill".to_owned()
        } else {
            format!("Skill · {}", skill.description)
        },
        value: format!("/skill:{} ", skill.name),
        execute: false,
    });
    templates.chain(skills).collect()
}

/// Fills the argument palette for `/model `, `/thinking `, and `/login `.
///
/// Listing configured providers resolves credentials, which can refresh an
/// OAuth token over the network, so those two lists are computed once per
/// palette opening and reused until the composer leaves the command.
fn palette_suggestions(
    app: &App,
    prepared: &runtime::PreparedSession,
    catalog: &catalog::Catalog,
    state: &agent::State,
    view: &mut InteractiveView,
) -> Vec<state::Suggestion> {
    let input = app.input.as_str();
    if !input.starts_with("/model ") {
        view.model_choices = None;
    }
    if !input.starts_with("/login ") {
        view.login_choices = None;
    }
    if !input.starts_with("/resume ") {
        view.resume_choices = None;
    }
    if state::dynamic_palette_argument(input).is_none() {
        return Vec::new();
    }
    if input.starts_with("/resume ") {
        return view
            .resume_choices
            .get_or_insert_with(|| resume_choices(prepared))
            .clone();
    }
    if input.starts_with("/thinking ") {
        return stream::supported_thinking_levels(&state.model)
            .into_iter()
            .map(|level| {
                let mut description = thinking_level_description(&level).to_owned();
                if level == effective_thinking_level(state) {
                    description.push_str(" · current");
                }
                state::Suggestion {
                    value: format!("/thinking {level}"),
                    label: level,
                    description,
                    execute: true,
                }
            })
            .collect();
    }
    if input.starts_with("/model ") {
        return view
            .model_choices
            .get_or_insert_with(|| {
                let choices = interactive_models(catalog)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|model| {
                        let current =
                            model.provider == state.model.provider && model.id == state.model.id;
                        state::Suggestion {
                            label: format!("{}/{}", model.provider, model.id),
                            description: model_picker_description(&model, current),
                            value: format!("/model {}/{}", model.provider, model.id),
                            execute: true,
                        }
                    })
                    .collect::<Vec<_>>();
                if choices.is_empty() {
                    // An empty picker is a dead end; steer to the login
                    // picker instead.
                    vec![state::Suggestion {
                        label: "/login".to_owned(),
                        description: "No authenticated models yet; add a provider".to_owned(),
                        value: "/login ".to_owned(),
                        execute: false,
                    }]
                } else {
                    choices
                }
            })
            .clone();
    }
    view.login_choices
        .get_or_insert_with(|| login_choices(catalog))
        .clone()
}

/// What each reasoning level means, for the `/thinking` picker.
fn thinking_level_description(level: &str) -> &'static str {
    match level {
        "off" => "No extended reasoning",
        "minimal" => "Barely any reasoning",
        "low" => "Short reasoning budget",
        "medium" => "Balanced",
        "high" => "Larger budget for hard problems",
        "xhigh" => "Very large budget",
        "max" => "Largest budget this model accepts",
        _ => "",
    }
}

/// The level requests actually use: a model without reasoning runs with
/// thinking off whatever level the session last chose.
fn effective_thinking_level(state: &agent::State) -> String {
    let levels = stream::supported_thinking_levels(&state.model);
    if runtime::model_is_selected(&state.model) && levels.contains(&state.thinking_level) {
        state.thinking_level.clone()
    } else {
        llm::THINKING_OFF.to_owned()
    }
}

/// "Claude Sonnet 5 · 1M ctx · current"; gateway models say which gateway.
fn model_picker_description(model: &llm::Model, current: bool) -> String {
    let mut parts = Vec::new();
    match model.provider.as_str() {
        "omni" => parts.push("via OmniRoute".to_owned()),
        "aperture" => parts.push("via Aperture gateway".to_owned()),
        _ => {
            if !model.name.is_empty() && model.name != model.id {
                parts.push(model.name.clone());
            }
            if model.context_window > 0 {
                parts.push(format!("{} ctx", short_token_count(model.context_window)));
            }
        }
    }
    if current {
        parts.push("current".to_owned());
    }
    parts.join(" · ")
}

/// 1M, 200k, 1.5M: a context window as people say it.
fn short_token_count(tokens: u64) -> String {
    if tokens >= 1_000_000 {
        if tokens.is_multiple_of(1_000_000) {
            format!("{}M", tokens / 1_000_000)
        } else {
            format!("{:.1}M", tokens as f64 / 1_000_000.0)
        }
    } else if tokens >= 1_000 {
        format!("{}k", tokens / 1_000)
    } else {
        tokens.to_string()
    }
}

/// Providers most people arrive with, listed first in the login picker; the
/// rest follow alphabetically.
const FEATURED_PROVIDERS: &[&str] = &[
    "anthropic",
    "openai-codex",
    "grok-cli",
    "xai",
    "meta",
    "meta-muse",
    "kimi-coding",
    "openai",
    "google",
    "mistral",
    "omni",
    "openrouter",
    "deepseek",
    "moonshotai",
    "zai",
    "groq",
];

/// The `/login` picker: one row per way in. A provider with a browser or
/// device login and an API key gets a second, key row further down, so a
/// developer key is never hidden behind a subscription flow.
fn login_choices(catalog: &catalog::Catalog) -> Vec<state::Suggestion> {
    let mut providers = catalog
        .providers()
        .into_iter()
        .filter(|provider| {
            provider
                .models()
                .iter()
                .any(|model| providers::supports_api(&model.api))
        })
        .collect::<Vec<_>>();
    providers.sort_by_key(|provider| {
        (
            FEATURED_PROVIDERS
                .iter()
                .position(|featured| *featured == provider.id)
                .unwrap_or(usize::MAX),
            provider.id.clone(),
        )
    });
    let mut primary = Vec::new();
    let mut key_rows = Vec::new();
    for provider in providers {
        let configured = catalog.is_configured(&provider.id).unwrap_or(false);
        let mark = |description: String| {
            if configured {
                format!("{description} · ✓ configured")
            } else {
                description
            }
        };
        let api_key = provider
            .env_keys
            .first()
            .map(|key| format!("API key · {key}"))
            .unwrap_or_else(|| "API key".to_owned());
        if let Some(setup) = gateway_setup_command(&provider.id) {
            primary.push(state::Suggestion {
                description: mark(format!("{} gateway · {setup}", gateway_name(&provider))),
                value: format!("/login {}", provider.id),
                label: provider.id,
                execute: true,
            });
            continue;
        }
        if login_flow_available(&provider.id) {
            primary.push(state::Suggestion {
                description: mark(oauth_login_description(&provider.id, &provider.name)),
                value: format!("/login {}", provider.id),
                label: provider.id.clone(),
                execute: true,
            });
            if api_key_login_available(catalog, &provider.id) {
                let api_key = match provider.id.as_str() {
                    "anthropic" => "API key · ANTHROPIC_API_KEY".to_owned(),
                    _ => api_key,
                };
                key_rows.push(state::Suggestion {
                    description: mark(api_key),
                    value: format!("/login {} key", provider.id),
                    label: provider.id,
                    execute: true,
                });
            }
            continue;
        }
        primary.push(state::Suggestion {
            description: mark(api_key),
            value: format!("/login {}", provider.id),
            label: provider.id,
            execute: true,
        });
    }
    primary.extend(key_rows);
    primary
}

fn gateway_name(provider: &catalog::Provider) -> String {
    match provider.id.as_str() {
        "omni" => "OmniRoute".to_owned(),
        "aperture" => "Tailscale Aperture".to_owned(),
        _ => provider.name.clone(),
    }
}

/// How a browser or device login signs in, in the picker's words. Unknown
/// providers (new logins added later) fall back to their display name.
fn oauth_login_description(id: &str, name: &str) -> String {
    match id {
        "anthropic" => "Claude Pro / Max subscription · OAuth".to_owned(),
        "openai-codex" => "ChatGPT Plus / Pro · OAuth".to_owned(),
        // Grok CLI is the route a consumer Grok subscription works on;
        // xAI's own login reaches api.x.ai, which often refuses one.
        "grok-cli" => "X Premium / SuperGrok subscription · OAuth".to_owned(),
        "xai" => "xAI account · device code or browser".to_owned(),
        "meta" => "Meta account · mints a Model API key".to_owned(),
        "meta-muse" => "Muse Code subscription · OAuth".to_owned(),
        "openrouter" => "OpenRouter account · OAuth".to_owned(),
        "kimi-coding" => "Kimi Code · OAuth".to_owned(),
        _ if name.is_empty() => "Browser sign-in · OAuth".to_owned(),
        _ => format!("{} · OAuth", name),
    }
}

const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

/// "12s" under a minute, "1:05" after.
fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}:{:02}", seconds / 60, seconds % 60)
    }
}

/// The status bar's left side: what is happening and for how long, such as
/// "⠦ Running bash · cargo test · 12s" or "⠸ Retrying in 4s · attempt 2/4".
fn interactive_status(view: &InteractiveView, state: &agent::State, busy: bool) -> String {
    if !busy {
        return view.activity.clone();
    }
    let elapsed = view
        .activity_since
        .map(|started| started.elapsed())
        .unwrap_or_default();
    let spinner = SPINNER[(elapsed.as_millis() / 120) as usize % SPINNER.len()];
    if let Some(retry) = view.retry.as_ref() {
        let remaining = retry.fires_at.saturating_duration_since(Instant::now());
        if !remaining.is_zero() {
            return format!(
                "{spinner} Retrying in {}s · attempt {}/{}",
                remaining.as_secs() + u64::from(remaining.subsec_nanos() > 0),
                retry.attempt,
                retry.attempts
            );
        }
    }
    let activity = live_activity(view, state);
    let queued = state_queue_suffix(view);
    format!("{spinner} {activity}{queued} · {}", format_elapsed(elapsed))
}

/// "· 1 queued" while steering or follow-up messages wait.
fn state_queue_suffix(view: &InteractiveView) -> String {
    if view.queued_count == 0 {
        String::new()
    } else {
        format!(" · {} queued", view.queued_count)
    }
}

/// The current step in words. A streaming reply reads "Thinking" until its
/// first visible text and "Responding" after.
fn live_activity(view: &InteractiveView, state: &agent::State) -> String {
    if view.activity == "Composing response"
        && let Some(llm::Message::Assistant(message)) = state.streaming_message.as_ref()
    {
        let has_text = message
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::Text(text) if !text.text.is_empty()));
        let has_thinking = message
            .content
            .iter()
            .any(|block| matches!(block, llm::ContentBlock::Thinking(_)));
        return if has_text || !has_thinking {
            "Responding".to_owned()
        } else {
            "Thinking".to_owned()
        };
    }
    if view.activity == "Composing response" {
        return "Waiting for the model".to_owned();
    }
    view.activity.clone()
}

fn agent_messages(messages: &[llm::Message]) -> Vec<(usize, Message)> {
    let pairing = llm::ToolPairing::new(messages.iter().map(Some));
    let mut result = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let mut push = |message: Message| result.push((index, message));
        match message {
            llm::Message::User(user) => {
                if compaction::is_summary_message(message) {
                    continue;
                }
                let text = user_message_text(user);
                // pi renders a branch summary as its own compact entry, not
                // as something the user said.
                if let Some(summary) = session::branch_summary_text(&text) {
                    push(Message {
                        role: MessageRole::Summary,
                        title: "branch summary".to_owned(),
                        text: summary.to_owned(),
                        ..Message::default()
                    });
                    continue;
                }
                push(Message {
                    role: MessageRole::User,
                    text,
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
                    push(Message {
                        role: MessageRole::Thinking,
                        text: thinking,
                        ..Message::default()
                    });
                }
                if !text.is_empty() {
                    push(Message {
                        role: MessageRole::Assistant,
                        text,
                        ..Message::default()
                    });
                }
                if assistant.stop_reason == stream::STOP_ABORTED {
                    // The user stopped it; that is not an error to alarm
                    // anyone with.
                    push(Message {
                        role: MessageRole::Summary,
                        title: "Interrupted".to_owned(),
                        text: "stopped before the reply finished".to_owned(),
                        ..Message::default()
                    });
                } else if !assistant.error_message.is_empty() {
                    push(Message {
                        role: MessageRole::Error,
                        text: assistant.error_message.clone(),
                        is_error: true,
                        ..Message::default()
                    });
                }
                for (block, content) in assistant.content.iter().enumerate() {
                    let llm::ContentBlock::ToolCall(call) = content else {
                        continue;
                    };
                    let matched =
                        pairing.result_for(index, block).and_then(|answer| {
                            match &messages[answer] {
                                llm::Message::ToolResult(result) => Some(result.as_ref()),
                                _ => None,
                            }
                        });
                    push(tool_view_message(call, matched));
                }
            }
            llm::Message::ToolResult(tool_result) => {
                if pairing.is_answer(index) {
                    continue;
                }
                push(unmatched_tool_view_message(tool_result));
            }
        }
    }
    result
}

/// Moves every queued steering and follow-up message back into the editor,
/// ahead of whatever was being typed, and returns how many there were.
fn restore_queued_messages_to_editor(app: &mut App, agent: &agent::Agent) -> usize {
    let queued = agent
        .take_queued_messages()
        .iter()
        .filter_map(|message| match message {
            llm::Message::User(user) => Some(user_message_text(user)),
            _ => None,
        })
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>();
    if queued.is_empty() {
        return 0;
    }
    let count = queued.len();
    let mut parts = queued;
    let current = app.input.trim();
    if !current.is_empty() {
        parts.push(current.to_owned());
    }
    app.set_input(&parts.join("\n\n"));
    count
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
    // A successful edit shows the change itself, as pi's edit renderer does,
    // rather than the one-line "Edited path" the model receives.
    if call.name == "edit"
        && !is_error
        && let Some(result) = result
        && let Some(diff) = edit_diff(call, result)
    {
        return Message {
            role: MessageRole::Tool,
            title: tool_title(call),
            text: diff.lines().take(3).collect::<Vec<_>>().join("\n"),
            detail: diff,
            is_error,
            ..Message::default()
        };
    }
    let text = match result {
        None => "running…".to_owned(),
        Some(_) if is_error => first_line(&detail),
        Some(_)
            if matches!(
                call.name.as_str(),
                "bash"
                    | "read"
                    | "grep"
                    | "find"
                    | "ls"
                    | "list"
                    | "planner_submit_plan"
                    | "web_search"
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
        ..Message::default()
    }
}

/// A unified-style diff of an edit call: a hunk header when the tool
/// reported where the change landed (`firstChangedLine`, as in pi's edit
/// details), then the removed and added lines with the unchanged lines at
/// either end left out.
fn edit_diff(call: &llm::ToolCall, result: &llm::ToolResultMessage) -> Option<String> {
    let text = |name: &str| call.arguments.get(name).and_then(serde_json::Value::as_str);
    let old_text = text("old_text").or_else(|| text("oldText"))?;
    let new_text = text("new_text").or_else(|| text("newText"))?;
    let old_lines = old_text.lines().collect::<Vec<_>>();
    let new_lines = new_text.lines().collect::<Vec<_>>();
    let prefix = old_lines
        .iter()
        .zip(&new_lines)
        .take_while(|(old, new)| old == new)
        .count();
    let suffix = old_lines[prefix..]
        .iter()
        .rev()
        .zip(new_lines[prefix..].iter().rev())
        .take_while(|(old, new)| old == new)
        .count();
    let removed = &old_lines[prefix..old_lines.len() - suffix];
    let added = &new_lines[prefix..new_lines.len() - suffix];
    let mut diff = Vec::new();
    if let Some(first) = result
        .details
        .as_ref()
        .and_then(|details| details.get("firstChangedLine"))
        .and_then(serde_json::Value::as_u64)
    {
        let start = first as usize + prefix;
        diff.push(format!(
            "@@ -{start},{} +{start},{} @@",
            removed.len(),
            added.len()
        ));
    }
    diff.extend(removed.iter().map(|line| format!("-{line}")));
    diff.extend(added.iter().map(|line| format!("+{line}")));
    (!diff.is_empty()).then(|| diff.join("\n"))
}

fn unmatched_tool_view_message(result: &llm::ToolResultMessage) -> Message {
    let detail = content_text(&result.content);
    Message {
        role: MessageRole::Tool,
        title: result.tool_name.clone(),
        text: first_line(&detail),
        detail,
        is_error: result.is_error,
        ..Message::default()
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

/// Estimated context use of the next request, against the model's window.
struct ContextUsage {
    tokens: u64,
    limit: u64,
    percent: u8,
    cost: f64,
}

fn context_usage(state: &agent::State) -> ContextUsage {
    let context = llm::Context {
        system_prompt: state.system_prompt.clone(),
        messages: state.messages.clone(),
        tools: state.tools.iter().map(agent::Tool::llm_tool).collect(),
    };
    // Before the first message nothing has been sent; the system prompt
    // alone is not "context used" to anyone reading the bar.
    let tokens = if state.messages.is_empty() {
        0
    } else {
        stream::estimate_context_tokens(&context).tokens
    };
    let limit = state.model.context_window;
    let percent = if limit == 0 {
        0
    } else {
        tokens.saturating_mul(100).saturating_div(limit).min(100) as u8
    };
    ContextUsage {
        tokens,
        limit,
        percent,
        cost: compaction::conversation_cost(&state.messages, &state.compactions),
    }
}

/// What the session has done so far, for the sidebar's Activity section.
#[derive(Default)]
struct ActivitySummary {
    turns: usize,
    tools: usize,
    failed: usize,
    last_tool: Option<String>,
    files: Vec<(String, state::FileStatus)>,
}

fn activity_summary(messages: &[llm::Message]) -> ActivitySummary {
    let mut summary = ActivitySummary::default();
    let pairing = llm::ToolPairing::new(messages.iter().map(Some));
    let mut seen_paths = BTreeSet::new();
    for (index, message) in messages.iter().enumerate() {
        match message {
            llm::Message::User(user) => {
                if !compaction::is_summary_message(message)
                    && session::branch_summary_text(&user_message_text(user)).is_none()
                {
                    summary.turns += 1;
                }
            }
            llm::Message::Assistant(assistant) => {
                for (block, content) in assistant.content.iter().enumerate() {
                    let llm::ContentBlock::ToolCall(call) = content else {
                        continue;
                    };
                    summary.tools += 1;
                    summary.last_tool = Some(tool_title(call));
                    let failed = pairing.result_for(index, block).is_some_and(|answer| {
                        matches!(&messages[answer], llm::Message::ToolResult(result) if result.is_error)
                    });
                    if failed {
                        summary.failed += 1;
                    }
                    let path = call
                        .arguments
                        .get("path")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    if path.is_empty() {
                        continue;
                    }
                    let known = !seen_paths.insert(path.clone());
                    if failed || !matches!(call.name.as_str(), "edit" | "write") {
                        continue;
                    }
                    // A write to a path the session never touched before is
                    // most likely a new file; anything else is a change.
                    let status = if call.name == "write" && !known {
                        state::FileStatus::Added
                    } else {
                        state::FileStatus::Modified
                    };
                    match summary.files.iter_mut().find(|(file, _)| *file == path) {
                        Some(_) => {}
                        None => summary.files.push((path, status)),
                    }
                }
            }
            llm::Message::ToolResult(_) => {}
        }
    }
    summary
}

fn plural(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

/// 12,480: a token count as the sidebar prints it.
fn grouped_number(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

/// A path with the home directory written as `~`.
fn home_relative(path: &str) -> String {
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() && path.starts_with(&home) => {
            let rest = &path[home.len()..];
            if rest.is_empty() || rest.starts_with('/') {
                format!("~{rest}")
            } else {
                path.to_owned()
            }
        }
        _ => path.to_owned(),
    }
}

fn runtime_sidebar(
    prepared: &runtime::PreparedSession,
    state: &agent::State,
    view: &InteractiveView,
    context: &ContextUsage,
) -> Vec<state::SidebarLine> {
    let name = session_title(prepared).unwrap_or_else(|| "New Session".to_owned());
    let storage = if prepared.runtime.recording() {
        prepared.runtime.id().map_or_else(
            || "recording".to_owned(),
            |id| format!("recording · {}", short_id(&id)),
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
    let planner_state = prepared
        .planner
        .as_ref()
        .map(|planner| planner.manager().state());
    let mode = match planner_state.as_ref().map(|state| &state.phase) {
        Some(plannotator::Phase::Planning | plannotator::Phase::Executing) => "planner",
        _ => "normal",
    };
    let model = if runtime::model_is_selected(&state.model) {
        model_label(&state.model)
    } else {
        "no model".to_owned()
    };
    let mut lines = vec![
        state::SidebarLine::title(name),
        state::SidebarLine::accent(model),
        state::SidebarLine::meta(format!(
            "{} thinking · {mode}",
            effective_thinking_level(state)
        )),
        state::SidebarLine::meta(storage),
        state::SidebarLine::blank(),
        state::SidebarLine::section("Context"),
        state::SidebarLine::progress(context.percent),
        state::SidebarLine::meta(format!(
            "{} / {} tokens",
            grouped_number(context.tokens),
            grouped_number(context.limit)
        )),
        state::SidebarLine::meta(format!(
            "{}% used · ${:.4} spent",
            context.percent, context.cost
        )),
    ];

    let busy = state.is_streaming || view.turn_pending || view.background.is_some();
    let mut transcript = state.messages.clone();
    if let Some(message) = state.streaming_message.as_ref() {
        transcript.push(message.clone());
    }
    let summary = activity_summary(&transcript);
    if busy || summary.tools > 0 {
        lines.extend([
            state::SidebarLine::blank(),
            state::SidebarLine::section("Activity"),
        ]);
        if busy {
            let activity = if view.retry.is_some() {
                "Retrying".to_owned()
            } else {
                live_activity(view, state)
            };
            lines.push(state::SidebarLine {
                kind: state::SidebarKind::Active,
                value: activity,
            });
        }
        let mut counts = vec![if busy {
            format!("turn {}", summary.turns.max(1))
        } else {
            plural(summary.turns, "turn", "turns")
        }];
        counts.push(plural(summary.tools, "tool", "tools"));
        if !summary.files.is_empty() {
            counts.push(format!(
                "{} changed",
                plural(summary.files.len(), "file", "files")
            ));
        }
        if summary.failed > 0 {
            counts.push(format!("{} failed", summary.failed));
        }
        lines.push(state::SidebarLine::meta(counts.join(" · ")));
        if view.queued_count > 0 {
            lines.push(state::SidebarLine::meta(format!(
                "follow-up queued: {}",
                view.queued_count
            )));
        }
        if let Some(tool) = summary.last_tool {
            lines.push(state::SidebarLine::meta(first_line(&tool)));
        }
        const SHOWN_FILES: usize = 6;
        for (path, status) in summary.files.iter().rev().take(SHOWN_FILES) {
            lines.push(state::SidebarLine {
                kind: state::SidebarKind::File { status: *status },
                value: path.clone(),
            });
        }
        if summary.files.len() > SHOWN_FILES {
            lines.push(state::SidebarLine::meta(format!(
                "… {} more",
                summary.files.len() - SHOWN_FILES
            )));
        }
    }
    if let Some(planner_state) = planner_state
        && !planner_state.items.is_empty()
    {
        lines.extend([
            state::SidebarLine::blank(),
            state::SidebarLine::section("Plan"),
        ]);
        lines.extend(planner_state.items.iter().map(|item| state::SidebarLine {
            kind: state::SidebarKind::Todo {
                complete: item.completed,
            },
            value: item.text.clone(),
        }));
    }
    lines.extend([
        state::SidebarLine::blank(),
        state::SidebarLine::section("Workspace"),
        state::SidebarLine::path(home_relative(&cwd)),
        state::SidebarLine::blank(),
        state::SidebarLine::brand(format!("● GoshCoder v{}", ui_version())),
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

/// First-run guidance shown when chat opens without an authenticated provider.
const NO_MODEL_WELCOME: &str = "No provider is authenticated yet. Choose one and press Enter; the OAuth or API-key flow runs here, and the first login selects that provider's default model.";

const NO_MODEL_PROMPT_REFUSED: &str = "No model is selected yet. Choose a provider with /login first; the first login also selects a model.";

const NO_MODEL_PICKER_EMPTY: &str =
    "No authenticated models are available yet. Use /login to add a provider.";

/// `provider/id`, or a placeholder for the unselected model.
fn model_label(model: &llm::Model) -> String {
    if runtime::model_is_selected(model) {
        format!("{}/{}", model.provider, model.id)
    } else {
        "no model selected".to_owned()
    }
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
                if providers::supports_api(&model.api) {
                    reference
                } else {
                    format!("{reference}\n  [unsupported protocol {}]", model.api)
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
        .filter(|model| providers::supports_api(&model.api))
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

/// `/tree`: every user message, branches indented under the point they
/// split from, the current path marked, abandoned branches dimmed by a
/// trailing note.
fn render_session_tree(points: &[session::BranchPoint]) -> String {
    let mut lines = points
        .iter()
        .map(|point| {
            let label = point
                .label
                .as_deref()
                .map(|label| format!(" [{label}]"))
                .unwrap_or_default();
            let marker = if point.current {
                "  ← current"
            } else if point.on_path {
                ""
            } else {
                "  (other branch)"
            };
            format!(
                "{}{:>2}. {}{label}{marker}",
                "  ".repeat(point.depth),
                point.index,
                first_line(&point.text),
            )
        })
        .collect::<Vec<_>>();
    lines.push(
        "/fork N rewinds to before message N on this branch, or returns to another branch."
            .to_owned(),
    );
    lines.join("\n")
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

fn short_id(id: &str) -> &str {
    sessionlog::short_id(id)
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
    fn unknown_commands_suggest_the_nearest_subcommand() {
        assert_eq!(
            unknown_command_message("provider"),
            "unknown command \"provider\"; did you mean `goshcoder providers`? Run `goshcoder help` for usage"
        );
        assert!(unknown_command_message("sessionss").contains("`goshcoder sessions`"));
        assert_eq!(
            unknown_command_message("bogus"),
            "unknown command \"bogus\"; run `goshcoder help` for usage"
        );
    }

    #[test]
    fn the_login_picker_offers_muse_code_as_a_subscription_without_a_key_row() {
        let catalog = catalog::Catalog::with_environment(
            Some(std::sync::Arc::new(catalog::CredentialStore::in_memory())),
            std::sync::Arc::new(|_| None),
        )
        .expect("catalog")
        .with_dynamic_paths(catalog::DynamicPaths::disabled());
        let choices = login_choices(&catalog);
        let values = choices
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>();
        let muse = choices
            .iter()
            .find(|choice| choice.value == "/login meta-muse")
            .expect("Muse Code sign-in row");
        assert_eq!(muse.description, "Muse Code subscription · OAuth");
        assert!(!values.contains(&"/login meta-muse key"));
        // The Meta Model API keeps both ways in: its sign-in row comes just
        // before Muse Code, its key row with the other key rows below.
        let position = |value: &str| values.iter().position(|candidate| *candidate == value);
        let meta = position("/login meta").expect("Meta sign-in row");
        assert_eq!(values[meta + 1], "/login meta-muse");
        assert!(position("/login meta key").expect("Meta API-key row") > meta + 1);
        assert!(!api_key_login_available(&catalog, "meta-muse"));
        assert!(api_key_login_available(&catalog, "meta"));
        assert_eq!(api_key_alternative("meta-muse"), ("meta", "Meta Model API"));
    }

    #[test]
    fn the_unselected_model_is_labelled_instead_of_rendering_a_bare_slash() {
        assert_eq!(
            model_label(&runtime::unselected_model()),
            "no model selected"
        );
        let model = llm::Model {
            provider: "anthropic".to_owned(),
            id: "claude-sonnet-5".to_owned(),
            ..llm::Model::default()
        };
        assert_eq!(model_label(&model), "anthropic/claude-sonnet-5");
    }

    #[test]
    fn aborting_restores_queued_messages_to_the_editor_ahead_of_the_draft() {
        let agent = agent::Agent::new(agent::AgentOptions::default());
        let mut app = App::new();
        assert_eq!(restore_queued_messages_to_editor(&mut app, &agent), 0);
        assert!(app.input.is_empty());

        agent.steer(llm::Message::User(llm::UserMessage::text("steer this", 1)));
        agent.follow_up(llm::Message::User(llm::UserMessage::text("then this", 2)));
        agent.follow_up(llm::Message::User(llm::UserMessage::text("   ", 3)));
        app.set_input("half-typed draft");
        assert_eq!(restore_queued_messages_to_editor(&mut app, &agent), 2);
        assert_eq!(app.input, "steer this\n\nthen this\n\nhalf-typed draft");
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn help_and_version_are_non_interactive() {
        assert!(USAGE.contains("goshcoder omni <subcommand>"));
        assert!(USAGE.contains("OMNIROUTE_URL"));
        assert!(env!("CARGO_PKG_VERSION").starts_with("0."));
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

        let rendered = render_interactive_session_list(&sessions, None);

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

        // A reply the user stopped is an interruption, not an error.
        let mut aborted = event(agent::EventKind::MessageEnd);
        aborted.message = Some(llm::Message::Assistant(Box::new(llm::AssistantMessage {
            stop_reason: stream::STOP_ABORTED.to_owned(),
            error_message: "request aborted".to_owned(),
            ..llm::AssistantMessage::default()
        })));
        aborted.assistant_was_streamed = true;
        let mut stderr = Vec::new();
        render_run_event(&aborted, &mut Vec::new(), &mut stderr, false).expect("render abort");
        let stderr = String::from_utf8(stderr).expect("stderr");
        assert!(
            stderr.contains("(interrupted)") && !stderr.contains("error"),
            "{stderr}"
        );
    }

    fn notice(anchor: usize, text: &str) -> AnchoredNotice {
        AnchoredNotice {
            anchor,
            message: Message {
                role: MessageRole::Notice,
                text: text.to_owned(),
                ..Message::default()
            },
        }
    }

    #[test]
    fn notices_interleave_with_the_transcript_in_the_order_they_happened() {
        let agent = vec![
            (
                0,
                Message {
                    role: MessageRole::User,
                    text: "first".to_owned(),
                    ..Message::default()
                },
            ),
            (
                1,
                Message {
                    text: "first reply".to_owned(),
                    ..Message::default()
                },
            ),
            (
                2,
                Message {
                    role: MessageRole::User,
                    text: "second".to_owned(),
                    ..Message::default()
                },
            ),
            (
                3,
                Message {
                    text: "second reply".to_owned(),
                    ..Message::default()
                },
            ),
        ];
        let notices = [
            notice(0, "startup"),
            notice(2, "/help output"),
            notice(9, "anchored past a rewind"),
        ];
        let order = splice_transcript(agent, &notices)
            .into_iter()
            .map(|message| message.text)
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [
                "startup",
                "first",
                "first reply",
                // /help ran after the first exchange: the later reply comes
                // below it, not above.
                "/help output",
                "second",
                "second reply",
                "anchored past a rewind",
            ]
        );
    }

    #[test]
    fn command_output_is_a_notice_and_the_typed_command_its_own_card() {
        let mut view = InteractiveView {
            message_count: 4,
            ..InteractiveView::default()
        };
        echo_command(&mut view, "/hotkeys");
        append_view_message(&mut view, MessageRole::Command, "Enter send");
        append_view_message(&mut view, MessageRole::Error, "boom");
        let roles = view
            .notices
            .iter()
            .map(|notice| (notice.anchor, notice.message.role))
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            [
                (4, MessageRole::Command),
                (4, MessageRole::Notice),
                (4, MessageRole::Error)
            ]
        );
        assert!(echoes_command("/help"));
        assert!(echoes_command("/login xai"));
        assert!(!echoes_command("/login"));
        assert!(!echoes_command("/model omni/auto"));
        assert!(!echoes_command("/clear"));
    }

    #[test]
    fn retry_notices_become_a_countdown_with_attempt_numbers() {
        let (retry, summary) =
            parse_retry_notice("attempt 1 of 3 in 2s: anthropic returned 529 overloaded")
                .expect("parsed");
        assert_eq!((retry.attempt, retry.attempts), (2, 4));
        assert_eq!(summary, "anthropic returned 529 overloaded");
        let remaining = retry.fires_at.saturating_duration_since(Instant::now());
        assert!(remaining <= Duration::from_secs(2) && remaining > Duration::from_secs(1));
        assert!(parse_retry_notice("giving up after 3 attempt(s): boom").is_none());
    }

    #[test]
    fn edits_render_as_a_diff_with_the_reported_line() {
        let call = llm::ToolCall {
            name: "edit".to_owned(),
            arguments: BTreeMap::from([
                ("path".to_owned(), serde_json::json!("src/session.rs")),
                (
                    "old_text".to_owned(),
                    serde_json::json!("fn a() {\n    old();\n}"),
                ),
                (
                    "new_text".to_owned(),
                    serde_json::json!("fn a() {\n    new();\n    more();\n}"),
                ),
            ]),
            ..llm::ToolCall::default()
        };
        let result = llm::ToolResultMessage {
            details: Some(serde_json::json!({ "firstChangedLine": 410 })),
            content: vec![llm::ContentBlock::text("Edited src/session.rs")],
            ..llm::ToolResultMessage::default()
        };
        let card = tool_view_message(&call, Some(&result));
        assert_eq!(
            card.detail,
            "@@ -411,1 +411,2 @@\n-    old();\n+    new();\n+    more();"
        );
        assert_eq!(card.title, "edit src/session.rs");
        // Without the line the diff still shows; a failed edit shows its error.
        let plain = llm::ToolResultMessage {
            details: None,
            ..result.clone()
        };
        assert!(
            tool_view_message(&call, Some(&plain))
                .detail
                .starts_with("-    old();")
        );
        let failed = llm::ToolResultMessage {
            is_error: true,
            content: vec![llm::ContentBlock::text("old_text was not found")],
            ..result
        };
        assert_eq!(
            tool_view_message(&call, Some(&failed)).detail,
            "old_text was not found"
        );
    }

    #[test]
    fn branch_summaries_and_aborts_are_not_shown_as_turns_or_errors() {
        let summary = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\nrewound to before \"x\"</summary>";
        let aborted = llm::AssistantMessage {
            stop_reason: stream::STOP_ABORTED.to_owned(),
            error_message: "request aborted".to_owned(),
            ..llm::AssistantMessage::default()
        };
        let failed = llm::AssistantMessage {
            stop_reason: stream::STOP_ERROR.to_owned(),
            error_message: "529 overloaded".to_owned(),
            ..llm::AssistantMessage::default()
        };
        let view = agent_messages(&[
            llm::Message::User(llm::UserMessage::text(summary, 1)),
            llm::Message::Assistant(Box::new(aborted)),
            llm::Message::Assistant(Box::new(failed)),
        ]);
        let shown = view
            .iter()
            .map(|(index, message)| (*index, message.role, message.title.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(
            shown,
            [
                (0, MessageRole::Summary, "branch summary"),
                (1, MessageRole::Summary, "Interrupted"),
                (2, MessageRole::Error, ""),
            ]
        );
        assert_eq!(view[0].1.text, "rewound to before \"x\"");
    }

    #[test]
    fn activity_summary_counts_turns_tools_failures_and_changed_files() {
        let call = |id: &str, name: &str, path: &str| {
            llm::ContentBlock::ToolCall(llm::ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments: BTreeMap::from([("path".to_owned(), serde_json::json!(path))]),
                ..llm::ToolCall::default()
            })
        };
        let result = |id: &str, is_error: bool| {
            llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: id.to_owned(),
                is_error,
                ..llm::ToolResultMessage::default()
            }))
        };
        let messages = vec![
            llm::Message::User(llm::UserMessage::text("go", 1)),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![
                    call("1", "read", "src/a.rs"),
                    call("2", "edit", "src/a.rs"),
                    call("3", "write", "src/new.rs"),
                    call("4", "edit", "src/broken.rs"),
                ],
                ..llm::AssistantMessage::default()
            })),
            result("1", false),
            result("2", false),
            result("3", false),
            result("4", true),
        ];
        let summary = activity_summary(&messages);
        assert_eq!((summary.turns, summary.tools, summary.failed), (1, 4, 1));
        assert_eq!(
            summary.files,
            [
                ("src/a.rs".to_owned(), state::FileStatus::Modified),
                ("src/new.rs".to_owned(), state::FileStatus::Added),
            ]
        );
        assert_eq!(summary.last_tool.as_deref(), Some("edit src/broken.rs"));
    }

    #[test]
    fn queue_report_lists_what_will_run_next() {
        assert!(queue_report(&[]).starts_with("Nothing is queued."));
        assert_eq!(
            queue_report(&[
                "fix the test\nand rerun".to_owned(),
                "then commit".to_owned()
            ]),
            "2 messages queued:\n  1. fix the test ...\n  2. then commit"
        );
    }

    #[test]
    fn tool_cards_pair_results_by_turn_when_a_server_reuses_call_ids() {
        // Some OpenAI-compatible servers number calls per message, so the
        // same id comes back every turn; each card must keep its own output.
        let turn = |command: &str| {
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![llm::ContentBlock::ToolCall(llm::ToolCall {
                    id: "call_0".to_owned(),
                    name: "bash".to_owned(),
                    arguments: BTreeMap::from([("command".to_owned(), serde_json::json!(command))]),
                    ..llm::ToolCall::default()
                })],
                ..llm::AssistantMessage::default()
            }))
        };
        let result = |output: &str, is_error: bool| {
            llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                tool_call_id: "call_0".to_owned(),
                tool_name: "bash".to_owned(),
                content: vec![llm::ContentBlock::Text(llm::TextContent {
                    text: output.to_owned(),
                    ..llm::TextContent::default()
                })],
                is_error,
                ..llm::ToolResultMessage::default()
            }))
        };
        let messages = vec![
            llm::Message::User(llm::UserMessage::text("one", 1)),
            turn("echo first"),
            result("first output", false),
            llm::Message::User(llm::UserMessage::text("two", 2)),
            turn("false"),
            result("second failure", true),
        ];
        let cards = agent_messages(&messages)
            .into_iter()
            .map(|(_, message)| message)
            .filter(|message| message.role == MessageRole::Tool)
            .collect::<Vec<_>>();
        assert_eq!(cards.len(), 2, "no result may render as an orphan");
        assert!(cards[0].detail.contains("first output"), "{:?}", cards[0]);
        assert!(!cards[0].is_error);
        assert!(cards[1].detail.contains("second failure"), "{:?}", cards[1]);
        assert!(cards[1].is_error);

        let summary = activity_summary(&messages);
        assert_eq!((summary.tools, summary.failed), (2, 1));
    }

    #[test]
    fn session_tree_indents_branches_and_marks_the_current_path() {
        let point = |index, text: &str, depth, on_path, current| session::BranchPoint {
            index,
            id: format!("id{index}"),
            text: text.to_owned(),
            prompt: text.to_owned(),
            label: None,
            children: 0,
            current,
            on_path,
            depth,
        };
        let rendered = render_session_tree(&[
            point(1, "first", 0, true, false),
            point(2, "abandoned", 1, false, false),
            point(3, "new direction", 1, true, true),
        ]);
        let lines = rendered.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], " 1. first");
        assert_eq!(lines[1], "   2. abandoned  (other branch)");
        assert_eq!(lines[2], "   3. new direction  ← current");
        assert!(lines[3].starts_with("/fork N"));
    }

    #[test]
    fn line_mode_hotkeys_list_only_line_mode_keys() {
        let line = hotkeys_text(false);
        assert!(line.contains("Ctrl-C"));
        assert!(!line.contains("Ctrl-L"));
        assert!(!line.contains("Shift-Tab"));
        let fullscreen = hotkeys_text(true);
        assert!(fullscreen.contains("Ctrl-J"));
        assert!(fullscreen.contains("Ctrl-Home/Ctrl-End"));
        assert!(
            !fullscreen.contains("Shift-Enter insert"),
            "Shift-Enter is only promised where the terminal reports it"
        );
    }

    #[test]
    fn numbers_and_durations_read_as_in_the_sidebar() {
        assert_eq!(grouped_number(0), "0");
        assert_eq!(grouped_number(12_480), "12,480");
        assert_eq!(grouped_number(1_000_000), "1,000,000");
        assert_eq!(format_elapsed(Duration::from_secs(12)), "12s");
        assert_eq!(format_elapsed(Duration::from_secs(65)), "1:05");
        assert_eq!(short_token_count(1_000_000), "1M");
        assert_eq!(short_token_count(200_000), "200k");
        assert_eq!(
            summarize_large_pastes("see\nA\nB\nend", &["A\nB".to_owned()]),
            "see\n[pasted 2 lines]\nend"
        );
    }

    #[test]
    fn unknown_providers_get_a_generic_login_description() {
        assert_eq!(
            oauth_login_description("anthropic", "Anthropic"),
            "Claude Pro / Max subscription · OAuth"
        );
        assert_eq!(
            oauth_login_description("acme-cloud", "Acme Cloud"),
            "Acme Cloud · OAuth"
        );
        assert_eq!(thinking_level_description("medium"), "Balanced");
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
    fn the_login_picker_offers_grok_cli_as_a_subscription_without_a_key_row() {
        let catalog =
            catalog::Catalog::with_environment(None, Arc::new(|_| None)).expect("catalog");
        let choices = login_choices(&catalog);
        let values = choices
            .iter()
            .map(|choice| choice.value.as_str())
            .collect::<Vec<_>>();
        let grok = choices
            .iter()
            .find(|choice| choice.value == "/login grok-cli")
            .expect("grok-cli row");
        assert_eq!(
            grok.description,
            "X Premium / SuperGrok subscription · OAuth"
        );
        assert!(!values.contains(&"/login grok-cli key"));
        // xAI keeps both of its ways in, and Grok CLI is listed before it.
        assert!(values.contains(&"/login xai key"));
        let position = |value: &str| values.iter().position(|candidate| *candidate == value);
        assert!(position("/login grok-cli") < position("/login xai"));
    }
}
