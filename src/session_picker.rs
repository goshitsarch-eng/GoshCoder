//! Shared, line-oriented session selection for interactive chat startup.
//!
//! The picker deliberately runs before either terminal frontend owns the
//! screen. Keeping its input/output injected makes it usable for Ratatui,
//! line-mode chat, and tests without coupling session discovery to a UI loop.

use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

use time::{OffsetDateTime, macros::format_description};

use crate::sessionlog::{self, ListOptions, SessionInfo, Store};

/// Bounds the startup choice list; transcript search exists for older work.
pub const PICKER_LIMIT: usize = 50;

const PICKER_TIME_FORMAT: &[time::format_description::FormatItem<'static>] =
    format_description!("[month repr:short] [day padding:none] [hour padding:zero]:[minute]");

/// Lists the sessions suitable for `chat -resume`.
///
/// Transcript text is included so future interactive filtering can identify a
/// session by the work discussed in it instead of only its metadata. If the
/// current directory has no sessions, retry at the enclosing repository root:
/// sessions are sharded by workspace and chat is commonly launched from a
/// subdirectory.
pub fn list_sessions_for_picker(
    store: &Store,
    workdir: &Path,
) -> Result<Vec<SessionInfo>, Box<dyn Error>> {
    let options = ListOptions {
        limit: PICKER_LIMIT,
        with_text: true,
        ..ListOptions::default()
    };
    let sessions = store.list(workdir, options)?;
    if !sessions.is_empty() {
        return Ok(sessions);
    }

    let canonical_workdir = workdir
        .canonicalize()
        .unwrap_or_else(|_| workdir.to_path_buf());
    let Some(root) = repository_root(workdir) else {
        return Ok(sessions);
    };
    if root == canonical_workdir {
        return Ok(sessions);
    }
    Ok(store.list(root, options)?)
}

/// Returns the enclosing Git worktree root when it can be determined.
///
/// Failure to inspect Git must never prevent a new chat session from opening,
/// so this intentionally treats an unavailable or non-repository directory as
/// no fallback rather than surfacing an error.
pub fn repository_root(workdir: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(workdir)
        .args(["rev-parse", "--show-toplevel"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let root = String::from_utf8(output.stdout).ok()?;
    let root = root.trim();
    if root.is_empty() {
        return None;
    }
    let root = PathBuf::from(root);
    Some(root.canonicalize().unwrap_or(root))
}

/// Tests whether every normalized search term occurs in a session's metadata
/// or bounded transcript text.
pub fn matches_session(info: &SessionInfo, terms: &[String]) -> bool {
    let haystack = [
        info.id.as_str(),
        info.name.as_str(),
        info.first_message.as_str(),
        info.cwd.as_str(),
        info.search_text.as_str(),
    ]
    .join(" ")
    .to_ascii_lowercase();
    terms.iter().all(|term| haystack.contains(term))
}

/// Returns the readable label and metadata shown for one session.
pub fn describe_session(info: &SessionInfo, label: &str, show_cwd: bool) -> (String, String) {
    let mut parts = vec![format_modified(info.modified)];
    if info.messages > 0 {
        parts.push(format!("{} msg", info.messages));
    }
    if info.locked {
        parts.push("open elsewhere".to_owned());
    }
    if show_cwd {
        let cwd = Path::new(&info.cwd);
        if let Some(name) = cwd.file_name().filter(|name| !name.is_empty()) {
            parts.push(name.to_string_lossy().into_owned());
        }
    }
    let title = info.title();
    if !title.is_empty() && title != info.id {
        parts.push(title.to_owned());
    }
    (label.to_owned(), parts.join(" · "))
}

/// Renders the startup picker and returns the selected session, if any.
///
/// Pressing Enter is an intentional request for a fresh session. The caller
/// owns configuration mutation and converts a selected ID into the normal
/// `SessionSelection::Session` lifecycle path.
pub fn choose_session<R: BufRead, W: Write>(
    store: &Store,
    workdir: &Path,
    input: &mut R,
    output: &mut W,
) -> Result<Option<SessionInfo>, Box<dyn Error>> {
    let sessions = list_sessions_for_picker(store, workdir)?;
    choose_from_sessions(&sessions, input, output)
}

fn choose_from_sessions<R: BufRead, W: Write>(
    sessions: &[SessionInfo],
    input: &mut R,
    output: &mut W,
) -> Result<Option<SessionInfo>, Box<dyn Error>> {
    if sessions.is_empty() {
        writeln!(
            output,
            "no saved sessions for this workspace; starting a new one"
        )?;
        return Ok(None);
    }

    writeln!(output, "saved sessions for this workspace:")?;
    let labels = sessionlog::short_ids(sessions);
    for (index, (session, label)) in sessions.iter().zip(labels).enumerate() {
        let (label, description) = describe_session(session, &label, false);
        writeln!(output, "{:>3}  {label}  {description}", index + 1)?;
    }
    writeln!(
        output,
        "choose a number, or press Enter to start a new session:"
    )?;
    output.flush()?;

    let mut choice = String::new();
    if input.read_line(&mut choice)? == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "session picker input ended before a choice was made",
        )
        .into());
    }
    let choice = choice.trim();
    if choice.is_empty() {
        return Ok(None);
    }
    let selected = choice.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("choose a number between 1 and {}", sessions.len()),
        )
    })?;
    if !(1..=sessions.len()).contains(&selected) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("choose a number between 1 and {}", sessions.len()),
        )
        .into());
    }
    Ok(Some(sessions[selected - 1].clone()))
}

fn format_modified(modified: SystemTime) -> String {
    let seconds = modified
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs().min(i64::MAX as u64) as i64);
    OffsetDateTime::from_unix_timestamp(seconds)
        .ok()
        .and_then(|time| time.format(PICKER_TIME_FORMAT).ok())
        .unwrap_or_else(|| "unknown time".to_owned())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::sessionlog::LockOwner;

    fn session(id: &str, name: &str, first_message: &str, search_text: &str) -> SessionInfo {
        SessionInfo {
            id: id.to_owned(),
            path: PathBuf::from(format!("/sessions/{id}.jsonl")),
            cwd: "/work/project".to_owned(),
            name: name.to_owned(),
            first_message: first_message.to_owned(),
            created: None,
            modified: UNIX_EPOCH,
            messages: 4,
            cleared: 0,
            size: 0,
            search_text: search_text.to_owned(),
            locked: false,
            owner: LockOwner::default(),
        }
    }

    #[test]
    fn transcript_search_requires_each_term() {
        let matching = session(
            "aaaa1111-0000-7000-8000-000000000000",
            "SSE fix",
            "hello",
            "we fixed the streaming reader and retry loop",
        );
        let unrelated = session(
            "bbbb2222-0000-7000-8000-000000000000",
            "other",
            "unrelated",
            "tax returns",
        );
        let terms = ["streaming", "retry"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        assert!(matches_session(&matching, &terms));
        assert!(!matches_session(&unrelated, &terms));
    }

    #[test]
    fn picker_renders_unique_labels_and_returns_selected_session() {
        let sessions = vec![
            session("01a02d09-4d21-7f0a-9b55-000000000001", "first", "", ""),
            session("01a02d09-4d21-7f0a-9b55-000000000002", "second", "", ""),
        ];
        let mut input = Cursor::new("2\n");
        let mut output = Vec::new();

        let selected =
            choose_from_sessions(&sessions, &mut input, &mut output).expect("choose session");

        assert_eq!(selected.expect("selection").id, sessions[1].id);
        let rendered = String::from_utf8(output).expect("picker output");
        assert!(rendered.contains("saved sessions for this workspace"));
        assert!(rendered.contains("01a02d09-4d"));
        assert!(rendered.contains("choose a number"));
    }

    #[test]
    fn picker_allows_a_new_session_or_rejects_bad_indexes() {
        let sessions = vec![session("aaaa1111-0000-7000-8000-000000000000", "", "", "")];
        let mut output = Vec::new();
        assert!(
            choose_from_sessions(&sessions, &mut Cursor::new("\n"), &mut output)
                .expect("new session choice")
                .is_none()
        );
        let error = choose_from_sessions(&sessions, &mut Cursor::new("0\n"), &mut Vec::new())
            .expect_err("invalid index");
        assert!(error.to_string().contains("between 1 and 1"));
    }

    #[test]
    fn session_description_includes_state_and_title() {
        let mut info = session("aaaa1111-0000-7000-8000-000000000000", "SSE fix", "", "");
        info.locked = true;
        let (label, description) = describe_session(&info, "aaaa1111", false);

        assert_eq!(label, "aaaa1111");
        for expected in ["SSE fix", "4 msg", "open elsewhere"] {
            assert!(description.contains(expected), "{description}");
        }
    }
}
