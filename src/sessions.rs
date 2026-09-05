//! Command-line session inspection, export, import, and garbage collection.
//!
//! These commands use the same pi-compatible store as the interactive runtime,
//! so a session created by the Rust client remains inspectable even when it is
//! opened from another process or originated in an older pi installation.

use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use time::{OffsetDateTime, format_description::FormatItem, macros::format_description};

use crate::{
    config,
    llm::{self, ContentBlock},
    sessionlog::{self, ListOptions, SessionInfo, Store, Tree},
};

const MINUTE_FORMAT: &[FormatItem<'static>] =
    format_description!("[year]-[month]-[day] [hour]:[minute] UTC");
const SECOND_FORMAT: &[FormatItem<'static>] =
    format_description!("[year]-[month]-[day] [hour]:[minute]:[second] UTC");

pub fn command(args: &[String]) -> Result<(), Box<dyn Error>> {
    let cwd = std::env::current_dir()?;
    let store = Store::new(config::sessions_dir());
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    run(&store, &cwd, args, &mut stdout, &mut stderr)
}

/// Runs a session command against an explicit store. Keeping the command
/// dependency-injected makes its data-loss protections directly testable.
pub fn run(
    store: &Store,
    cwd: &Path,
    args: &[String],
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    match args.split_first() {
        None => list(store, cwd, &[], output, diagnostics),
        Some((subcommand, rest)) => match subcommand.as_str() {
            "list" => list(store, cwd, rest, output, diagnostics),
            "show" => show(store, cwd, rest, output, diagnostics),
            "rm" | "remove" | "delete" => remove(store, cwd, rest, diagnostics),
            "gc" => gc(store, cwd, rest, output, diagnostics),
            "export" => export(store, cwd, rest, output, diagnostics),
            "import" => import(store, cwd, rest, output, diagnostics),
            other => Err(format!(
                "unknown sessions subcommand {other:?}; use list, show, rm, gc, export or import"
            )
            .into()),
        },
    }
}

fn list(
    store: &Store,
    cwd: &Path,
    args: &[String],
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut all_workspaces = false;
    for argument in args {
        match argument.as_str() {
            "--all" | "-all" | "-a" => all_workspaces = true,
            _ => return Err(format!("unknown flag {argument:?} for sessions list").into()),
        }
    }
    let sessions = store.list(
        cwd,
        ListOptions {
            all_workspaces,
            ..ListOptions::default()
        },
    )?;
    if sessions.is_empty() {
        if all_workspaces {
            writeln!(diagnostics, "no sessions recorded yet")?;
        } else {
            writeln!(
                diagnostics,
                "no sessions for {} (use --all to see every workspace)",
                cwd.display()
            )?;
        }
        return Ok(());
    }

    for (info, short_id) in sessions.iter().zip(sessionlog::short_ids(&sessions)) {
        let marker = if info.locked { "*" } else { " " };
        write!(
            output,
            "{marker} {short_id}  {}  {:>3} msg",
            format_time(info.modified, false),
            info.messages,
        )?;
        if info.cleared > 0 {
            write!(output, " (+{} cleared)", info.cleared)?;
        }
        writeln!(output, "  {}", info.title())?;
        if all_workspaces {
            writeln!(output, "     {}", info.cwd)?;
        }
    }
    if sessions.iter().any(|info| info.locked) {
        writeln!(diagnostics, "* open in another window")?;
    }
    Ok(())
}

fn show(
    store: &Store,
    cwd: &Path,
    args: &[String],
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut full = false;
    let mut reference = None;
    for argument in args {
        match argument.as_str() {
            "--full" | "-full" => full = true,
            _ if reference.is_none() => reference = Some(argument.as_str()),
            _ => return Err("sessions show takes one session".into()),
        }
    }
    let reference =
        reference.ok_or_else(|| "sessions show needs a session id, prefix, or path".to_owned())?;
    let info = store.resolve(cwd, reference)?;
    let (tree, header, report) = store.load(&info.path)?;

    writeln!(output, "id        {}", info.id)?;
    writeln!(output, "path      {}", info.path.display())?;
    writeln!(output, "workspace {}", header.cwd)?;
    if !info.name.is_empty() {
        writeln!(output, "name      {}", info.name)?;
    }
    if let Some(parent) = header.parent_session.filter(|parent| !parent.is_empty()) {
        writeln!(output, "forked    {parent}")?;
    }
    writeln!(
        output,
        "created   {}",
        info.created
            .map(|created| created.format(SECOND_FORMAT).unwrap_or_default())
            .unwrap_or_else(|| format_time(info.modified, true))
    )?;
    writeln!(output, "modified  {}", format_time(info.modified, true))?;
    write!(output, "messages  {}", info.messages)?;
    if info.cleared > 0 {
        write!(output, " (+{} before the last clear)", info.cleared)?;
    }
    writeln!(output)?;
    writeln!(output, "size      {}", human_bytes(info.size))?;
    if info.locked {
        writeln!(output, "state     open in {}", info.owner)?;
    }
    if report.migrated {
        writeln!(
            output,
            "format    v{}, read as v{} (fork it to continue)",
            report.source_version,
            sessionlog::FORMAT_VERSION
        )?;
    }
    for warning in report.warnings {
        writeln!(diagnostics, "warning: {warning}")?;
    }
    if full {
        writeln!(
            diagnostics,
            "{} entries on the full path",
            tree.path(None).len()
        )?;
    }
    writeln!(output)?;
    render_transcript(&tree, full, output)?;
    Ok(())
}

fn remove(
    store: &Store,
    cwd: &Path,
    args: &[String],
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    if args.is_empty() {
        return Err("sessions rm needs at least one session id, prefix, or path".into());
    }
    let mut failures = Vec::new();
    for reference in args {
        match store.resolve(cwd, reference).and_then(|info| {
            store.remove(&info)?;
            Ok(info)
        }) {
            Ok(info) => writeln!(diagnostics, "removed {}  {}", info.short_id(), info.title())?,
            Err(error) => failures.push(format!("{reference}: {error}")),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; ").into())
    }
}

fn export(
    store: &Store,
    cwd: &Path,
    args: &[String],
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut format = ExportFormat::Jsonl;
    let mut reference = None;
    let mut destination = None;
    for argument in args {
        match argument.as_str() {
            "--jsonl" | "-jsonl" => format = ExportFormat::Jsonl,
            "--md" | "-md" | "--markdown" => format = ExportFormat::Markdown,
            _ if reference.is_none() => reference = Some(argument.as_str()),
            _ if destination.is_none() => destination = Some(argument.as_str()),
            _ => return Err("sessions export takes one session and one output path".into()),
        }
    }
    let reference = reference
        .ok_or_else(|| "sessions export needs a session id, prefix, or path".to_owned())?;
    let info = store.resolve(cwd, reference)?;
    let content = match format {
        ExportFormat::Jsonl => fs::read(&info.path)?,
        ExportFormat::Markdown => {
            let (tree, header, _) = store.load(&info.path)?;
            export_markdown(&info, &header, &tree).into_bytes()
        }
    };
    match destination {
        None | Some("-") => output.write_all(&content)?,
        Some(destination) => {
            fs::write(destination, &content)?;
            writeln!(diagnostics, "exported {} to {destination}", info.short_id())?;
        }
    }
    Ok(())
}

fn import(
    store: &Store,
    cwd: &Path,
    args: &[String],
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let [source] = args else {
        return Err("sessions import needs exactly one .jsonl path".into());
    };
    let source = store
        .resolve(cwd, source)
        .map_err(|error| format!("read {source}: {error}"))?;
    let mut writer = store.fork(&source, None, cwd)?;
    let path = writer.path().to_path_buf();
    let id = writer.id().to_owned();
    writer.close()?;
    writeln!(diagnostics, "imported as {}", short_id(&id))?;
    writeln!(output, "{}", path.display())?;
    Ok(())
}

fn gc(
    store: &Store,
    cwd: &Path,
    args: &[String],
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), Box<dyn Error>> {
    let mut older_than = None;
    let mut keep_named = false;
    let mut apply = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--older-than" | "-older-than" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("--older-than needs a duration like 30d".into());
                };
                older_than = Some(value.as_str());
            }
            "--keep-named" | "-keep-named" => keep_named = true,
            "--yes" | "-yes" | "-y" => apply = true,
            other => return Err(format!("unknown flag {other:?} for sessions gc").into()),
        }
        index += 1;
    }
    let older_than = older_than
        .ok_or_else(|| "sessions gc needs --older-than, e.g. --older-than 30d".to_owned())?;
    let cutoff = age_cutoff(older_than)?;
    let doomed = store
        .list(
            cwd,
            ListOptions {
                all_workspaces: true,
                ..ListOptions::default()
            },
        )?
        .into_iter()
        .filter(|info| info.modified <= cutoff && !info.locked)
        .filter(|info| !keep_named || info.name.is_empty())
        .collect::<Vec<_>>();
    if doomed.is_empty() {
        writeln!(diagnostics, "nothing to remove")?;
        return Ok(());
    }
    let freed = doomed.iter().map(|info| info.size).sum::<u64>();
    for info in &doomed {
        writeln!(
            output,
            "{}  {}  {}",
            info.short_id(),
            format_time(info.modified, false),
            info.title()
        )?;
    }
    writeln!(
        diagnostics,
        "{} session(s), {}",
        doomed.len(),
        human_bytes(freed)
    )?;
    if !apply {
        writeln!(
            diagnostics,
            "nothing was deleted; re-run with --yes to apply"
        )?;
        return Ok(());
    }
    for info in &doomed {
        if let Err(error) = store.remove(info) {
            writeln!(diagnostics, "skipped {}: {error}", info.short_id())?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ExportFormat {
    Jsonl,
    Markdown,
}

fn render_transcript(tree: &Tree, full: bool, output: &mut dyn Write) -> io::Result<()> {
    let entries = if full {
        tree.path(None)
    } else {
        tree.context_path(None)
    };
    if entries.is_empty() {
        writeln!(output, "The transcript is empty.")?;
        return Ok(());
    }
    writeln!(output, "transcript")?;
    for entry in entries {
        match entry.kind.as_str() {
            sessionlog::TYPE_COMPACTION => {
                writeln!(output, "· compacted {}", one_line(&entry.summary, 120))?;
            }
            sessionlog::TYPE_MESSAGE => {
                let Some(value) = entry.message.as_ref() else {
                    continue;
                };
                match serde_json::from_value::<llm::Message>(value.clone()) {
                    Ok(message) => writeln!(
                        output,
                        "· {:<10} {}",
                        message.role(),
                        one_line(&message.text_preview(), 120)
                    )?,
                    Err(error) => writeln!(output, "· unreadable entry {}: {error}", entry.id)?,
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn export_markdown(info: &SessionInfo, header: &sessionlog::Header, tree: &Tree) -> String {
    let mut markdown = format!(
        "# {}\n\n- Session: `{}`\n- Workspace: `{}`\n- Started: {}\n\n",
        info.title(),
        info.id,
        header.cwd,
        info.created
            .map(|time| time.format(SECOND_FORMAT).unwrap_or_default())
            .unwrap_or_else(|| format_time(info.modified, true))
    );
    for entry in tree.context_path(None) {
        match entry.kind.as_str() {
            sessionlog::TYPE_COMPACTION => {
                markdown.push_str(&format!(
                    "---\n\n**Context compacted** ({} tokens before)\n\n{}\n\n",
                    entry.tokens_before, entry.summary
                ));
            }
            sessionlog::TYPE_MESSAGE => {
                let Some(value) = entry.message.as_ref() else {
                    continue;
                };
                let Ok(message) = serde_json::from_value::<llm::Message>(value.clone()) else {
                    continue;
                };
                match message {
                    llm::Message::User(message) => {
                        markdown.push_str("## User\n\n");
                        markdown.push_str(&message.text_preview());
                        markdown.push_str("\n\n");
                    }
                    llm::Message::Assistant(message) => {
                        markdown.push_str("## Assistant\n\n");
                        for block in &message.content {
                            match block {
                                ContentBlock::Text(text) => {
                                    markdown.push_str(text.text.trim());
                                    markdown.push_str("\n\n");
                                }
                                ContentBlock::ToolCall(call) => {
                                    markdown.push_str(&format!("> called `{}`\n\n", call.name));
                                }
                                ContentBlock::Thinking(_) | ContentBlock::Image(_) => {}
                            }
                        }
                    }
                    llm::Message::ToolResult(message) => {
                        let suffix = if message.is_error { " (error)" } else { "" };
                        markdown
                            .push_str(&format!("> `{}` returned{suffix}\n\n", message.tool_name));
                    }
                }
            }
            _ => {}
        }
    }
    markdown
}

fn age_cutoff(value: &str) -> Result<SystemTime, Box<dyn Error>> {
    let value = value.trim();
    let mut characters = value.chars();
    let Some(unit) = characters.next_back() else {
        return Err("empty duration".into());
    };
    let number = characters
        .as_str()
        .parse::<u64>()
        .map_err(|_| format!("cannot read {value:?} as an age; use a form like 30d or 6w"))?;
    let seconds = match unit {
        'd' => number.checked_mul(24 * 60 * 60),
        'w' => number.checked_mul(7 * 24 * 60 * 60),
        _ => None,
    }
    .ok_or_else(|| format!("cannot read {value:?} as an age; use a form like 30d or 6w"))?;
    SystemTime::now()
        .checked_sub(Duration::from_secs(seconds))
        .ok_or_else(|| format!("cannot read {value:?} as an age; use a form like 30d or 6w").into())
}

fn format_time(value: SystemTime, seconds: bool) -> String {
    let value = OffsetDateTime::from(value);
    value
        .format(if seconds {
            SECOND_FORMAT
        } else {
            MINUTE_FORMAT
        })
        .unwrap_or_else(|_| "unknown time".to_owned())
}

fn short_id(id: &str) -> &str {
    let mut end = id.len().min(8);
    while end > 0 && !id.is_char_boundary(end) {
        end -= 1;
    }
    &id[..end]
}

fn one_line(value: &str, limit: usize) -> String {
    let value = value.split(['\r', '\n']).next().unwrap_or_default().trim();
    let mut characters = value.chars();
    let short = characters.by_ref().take(limit).collect::<String>();
    if characters.next().is_some() {
        format!("{short}…")
    } else {
        short
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sessionlog::{Entry, TYPE_MESSAGE};

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "goshcoder-rust-sessions-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn assistant(text: &str) -> Entry {
        Entry {
            kind: TYPE_MESSAGE.to_owned(),
            message: Some(serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "api": "test",
                "provider": "test",
                "model": "test",
                "usage": {},
                "stopReason": "stop",
                "timestamp": 1
            })),
            ..Entry::default()
        }
    }

    #[test]
    fn list_show_and_markdown_export_use_the_same_session_store() {
        let root = temp_root("commands");
        let workspace = root.join("workspace");
        let store = Store::new(root.join("sessions"));
        let mut writer = store
            .create_with_id(&workspace, None, "session-command")
            .expect("create");
        writer.append(assistant("completed work")).expect("append");
        writer.close().expect("close");

        let mut output = Vec::new();
        let mut diagnostics = Vec::new();
        run(
            &store,
            &workspace,
            &["list".to_owned()],
            &mut output,
            &mut diagnostics,
        )
        .expect("list");
        assert!(String::from_utf8_lossy(&output).contains("completed work"));

        output.clear();
        run(
            &store,
            &workspace,
            &["show".to_owned(), "session-command".to_owned()],
            &mut output,
            &mut diagnostics,
        )
        .expect("show");
        assert!(String::from_utf8_lossy(&output).contains("transcript"));

        let export_path = root.join("session.md");
        run(
            &store,
            &workspace,
            &[
                "export".to_owned(),
                "--md".to_owned(),
                "session-command".to_owned(),
                export_path.display().to_string(),
            ],
            &mut output,
            &mut diagnostics,
        )
        .expect("export");
        assert!(
            fs::read_to_string(export_path)
                .expect("read export")
                .contains("completed work")
        );
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn age_cutoff_requires_explicit_day_or_week_units() {
        assert!(age_cutoff("30d").is_ok());
        assert!(age_cutoff("6w").is_ok());
        assert!(age_cutoff("30").is_err());
        assert!(age_cutoff("tomorrow").is_err());
    }
}
