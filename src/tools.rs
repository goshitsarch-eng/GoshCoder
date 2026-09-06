//! Pi-compatible coding tools for a confined workspace.
//!
//! This module intentionally uses only the standard library plus the crate's
//! existing `serde_json` dependency.  It is self-contained so the runtime can
//! opt into it by adding `mod tools;` and constructing a [`Workspace`].
//!
//! ## Confinement model
//!
//! Filesystem tools reject lexical escapes, canonicalize the workspace root,
//! validate every existing path component, and reject symlinked components.
//! New parent directories are created one component at a time and checked
//! after creation.  That prevents ordinary traversal and symlink escapes.
//!
//! Rust's standard library does not expose descriptor-relative `openat` /
//! `O_NOFOLLOW` operations, so it cannot make this guarantee race-free
//! against a malicious concurrent process replacing a checked directory with
//! a symlink.  Run these tools in a workspace not writable by an untrusted
//! concurrent principal.  The Go implementation can use `os.Root` for that
//! stronger platform primitive; this is the closest dependency-free Rust
//! implementation.

use std::{
    collections::{BTreeMap, VecDeque},
    ffi::OsStr,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};

use crate::agent::{self, CancellationToken};

/// Maximum bytes returned by one `read` invocation before its notice.
pub const MAX_READ_BYTES: usize = 50 * 1024;
/// Maximum file prefix inspected to implement line-oriented reads.
///
/// `read` addresses lines, so it must scan from the start of the file to find
/// `offset`; this cap bounds that scan (2 MiB) instead of loading a multi-GiB
/// log into memory. Lines past the cap are not reachable through `read`: the
/// tool says so in its trailing notice and in the `offset` error, and `bash`
/// (`sed -n`, `tail`) remains available for the rest of such a file.
pub const MAX_READ_SCAN_BYTES: usize = MAX_READ_BYTES * 40;
/// Maximum bytes held for an exact `edit`.
pub const MAX_EDIT_BYTES: usize = 10 * 1024 * 1024;
/// Maximum captured output from `bash`.
pub const MAX_OUTPUT_BYTES: usize = 30 * 1024;
/// Maximum bytes accepted from `git ls-files`; a longer list is truncated.
pub const MAX_CANDIDATE_BYTES: usize = 8 * 1024 * 1024;
/// Maximum candidate paths a search inspects; the rest are dropped with a
/// notice rather than failing the whole search.
pub const MAX_CANDIDATE_FILES: usize = 100_000;
/// Maximum bytes read from a single file by `grep`.
pub const MAX_SEARCH_FILE_BYTES: usize = 4 * 1024 * 1024;
/// Maximum bytes shown from a matching source line.
pub const MAX_GREP_LINE_BYTES: usize = 2_000;
/// Maximum entries read from one directory listing.
pub const MAX_DIRECTORY_ENTRIES: usize = 10_000;
/// Maximum bytes rendered by a search or directory listing before its notice.
pub const MAX_LIST_OUTPUT_BYTES: usize = MAX_OUTPUT_BYTES;
/// Default timeout for the shell-backed `bash` tool.
pub const DEFAULT_BASH_TIMEOUT: Duration = Duration::from_secs(120);

const GIT_TIMEOUT: Duration = Duration::from_secs(5);
const OUTPUT_DRAIN_GRACE: Duration = Duration::from_secs(2);
/// Drain grace once the whole process group is known dead: only a writer that
/// escaped the group can still hold the pipe, and it will not close it on our
/// account, so collect what is buffered and stop.
const KILLED_DRAIN_GRACE: Duration = Duration::from_millis(250);
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const MUTATION_WAIT_POLL: Duration = Duration::from_millis(25);
const MAX_CONTEXT_LINES: usize = 100;
const MAX_GLOB_PATTERN_CHARS: usize = 4_096;
const MAX_GLOB_ALTERNATIVES: usize = 256;
const MAX_REGEX_PATTERN_CHARS: usize = 4_096;
const MAX_REGEX_REPEAT: usize = 1_024;
const MAX_REGEX_STATES: usize = 8_192;
const MAX_REGEX_STEPS: usize = 20_000_000;
/// Longest destination-name prefix embedded in a temporary file name, so the
/// `.<name>.goshcoder-<nonce>.tmp` sibling stays under common 255-byte limits.
const TEMP_NAME_PREFIX_BYTES: usize = 48;
#[cfg(unix)]
const SIGKILL: i32 = 9;
#[cfg(windows)]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Serialises mutations of one file, mirroring pi's file-mutation-queue: a
/// `write` or `edit` that races another call for the same path waits for it,
/// so neither read-modify-write can overwrite the other's result. Different
/// files still proceed in parallel.
static FILE_MUTATIONS: FileMutationQueue = FileMutationQueue {
    busy: Mutex::new(Vec::new()),
    released: Condvar::new(),
};

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

struct FileMutationQueue {
    busy: Mutex<Vec<PathBuf>>,
    released: Condvar,
}

/// Holds one path in the mutation queue until dropped.
struct FileMutationGuard {
    key: PathBuf,
}

impl FileMutationQueue {
    fn acquire(&self, key: PathBuf, cancellation: &CancellationToken) -> Result<FileMutationGuard> {
        let mut busy = self.busy.lock().unwrap_or_else(|error| error.into_inner());
        while busy.contains(&key) {
            // Poll instead of blocking indefinitely so a cancelled tool stops
            // queueing behind a slow mutation of the same file.
            check_cancelled(cancellation)?;
            let (guard, _) = self
                .released
                .wait_timeout(busy, MUTATION_WAIT_POLL)
                .unwrap_or_else(|error| error.into_inner());
            busy = guard;
        }
        busy.push(key.clone());
        Ok(FileMutationGuard { key })
    }

    fn release(&self, key: &Path) {
        let mut busy = self.busy.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(index) = busy.iter().position(|busy_key| busy_key == key) {
            busy.swap_remove(index);
        }
        self.released.notify_all();
    }
}

impl Drop for FileMutationGuard {
    fn drop(&mut self) {
        FILE_MUTATIONS.release(&self.key);
    }
}

/// Error returned by workspace construction and tool helpers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolError {
    message: String,
}

impl ToolError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn io(action: &str, path: &Path, error: io::Error) -> Self {
        Self::new(format!("{action} {}: {error}", path.display()))
    }
}

impl fmt::Display for ToolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ToolError {}

/// Result type used by this module's public helpers.
pub type Result<T> = std::result::Result<T, ToolError>;

/// A workspace to which filesystem tool operations are confined.
///
/// `bash` intentionally remains an arbitrary-command tool: its working
/// directory is this root, but a shell command can access the user's wider
/// machine permissions.  Filesystem confinement applies to `read`, `write`,
/// `edit`, `grep`, `find`, `ls`, and `list`.
#[derive(Clone, Debug)]
pub struct Workspace {
    /// Canonical absolute workspace root.
    pub root: PathBuf,
    /// Per-command shell timeout. A zero duration selects
    /// [`DEFAULT_BASH_TIMEOUT`].
    pub bash_timeout: Duration,
}

impl Workspace {
    /// Creates a workspace rooted at an existing directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let requested = root.as_ref();
        let canonical = canonicalize(requested)
            .map_err(|error| ToolError::io("resolve workspace", requested, error))?;
        let metadata = fs::metadata(&canonical)
            .map_err(|error| ToolError::io("inspect workspace", &canonical, error))?;
        if !metadata.is_dir() {
            return Err(ToolError::new(format!(
                "workspace {} is not a directory",
                requested.display()
            )));
        }

        Ok(Self {
            root: canonical,
            bash_timeout: DEFAULT_BASH_TIMEOUT,
        })
    }

    /// Returns the canonical workspace root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Sets a shell timeout while retaining a fluent construction style.
    pub fn with_bash_timeout(mut self, timeout: Duration) -> Self {
        self.bash_timeout = timeout;
        self
    }

    /// Replaces the timeout used by [`Self::bash_tool`].
    pub fn set_bash_timeout(&mut self, timeout: Duration) {
        self.bash_timeout = timeout;
    }

    /// Returns pi's seven active built-in tools.
    pub fn all(&self) -> Vec<agent::Tool> {
        let mut tools = self.planning();
        tools.push(self.bash_tool());
        tools
    }

    /// Returns the six filesystem tools appropriate for planning mode.
    pub fn planning(&self) -> Vec<agent::Tool> {
        vec![
            self.read_tool(),
            self.write_tool(),
            self.edit_tool(),
            self.ls_tool(),
            self.grep_tool(),
            self.find_tool(),
        ]
    }

    /// Builds the `read` tool.
    pub fn read_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "read",
            "Read",
            "Read the contents of a UTF-8 text file in the workspace. Output is truncated for very large files.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root"},
                    "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
                    "limit": {"type": "number", "description": "Maximum number of lines to read"}
                },
                "required": ["path"]
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_read(&cancellation, &parameters)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds the `write` tool.
    pub fn write_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "write",
            "Write",
            "Write content to a file in the workspace, creating parent directories as needed. Overwrites an existing file.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root"},
                    "content": {"type": "string", "description": "Full file content to write"}
                },
                "required": ["path", "content"]
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_write(&cancellation, &parameters)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds the `edit` tool.
    pub fn edit_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "edit",
            "Edit",
            "Replace an exact substring in a file. old_text must appear exactly once, so include enough surrounding context to make it unique.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path, relative to the workspace root"},
                    "old_text": {"type": "string", "description": "Exact text to replace; must be unique in the file"},
                    "new_text": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_text", "new_text"]
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_edit(&cancellation, &parameters)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds the legacy `list` alias. It is not included in [`Self::all`];
    /// pi's active seven-tool set calls this capability `ls`.
    pub fn list_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "list",
            "List",
            "List the entries of a directory in the workspace. Directories are suffixed with '/'.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, relative to the workspace root. Defaults to the root."}
                }
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_list(&cancellation, &parameters, None)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds pi's `ls` tool.
    pub fn ls_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "ls",
            "ls",
            "List directory contents. Returns entries sorted alphabetically, with '/' suffix for directories. Includes dotfiles.",
            json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum number of entries (default: 500)"}
                }
            }),
            move |cancellation, _, parameters, _| {
                let limit = positive_limit(&parameters, "limit", 500, MAX_DIRECTORY_ENTRIES);
                workspace
                    .run_list(&cancellation, &parameters, Some(limit))
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds the `grep` tool.
    pub fn grep_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "grep",
            "grep",
            "Search file contents for a regex or literal string. Returns matching lines with file paths and line numbers. Respects .gitignore in git workspaces. Output is limited to 100 matches by default.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                    "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
                    "glob": {"type": "string", "description": "Filter files by glob pattern"},
                    "ignoreCase": {"type": "boolean", "description": "Case-insensitive search"},
                    "literal": {"type": "boolean", "description": "Treat pattern as a literal string"},
                    "context": {"type": "number", "description": "Lines before and after each match"},
                    "limit": {"type": "number", "description": "Maximum matches (default: 100)"}
                },
                "required": ["pattern"]
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_grep(&cancellation, &parameters)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds the `find` tool.
    pub fn find_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "find",
            "find",
            "Search for files by glob pattern. Returns paths relative to the search directory and respects .gitignore. Output is limited to 1000 results by default.",
            json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern such as '*.rs' or 'src/**/*.rs'"},
                    "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
                    "limit": {"type": "number", "description": "Maximum results (default: 1000)"}
                },
                "required": ["pattern"]
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_find(&cancellation, &parameters)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    /// Builds the `bash` tool.
    pub fn bash_tool(&self) -> agent::Tool {
        let workspace = self.clone();
        agent::Tool::new(
            "bash",
            "Bash",
            "Run a shell command in the workspace and return its combined output. Use for builds, tests, and searches.",
            json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"}
                },
                "required": ["command"]
            }),
            move |cancellation, _, parameters, _| {
                workspace
                    .run_bash(&cancellation, &parameters)
                    .map(agent::ToolResult::text)
                    .map_err(|error| error.to_string())
            },
        )
    }

    fn run_read(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let requested = required_string(parameters, "path")?;
        let relative = self.resolve(requested)?;
        let (bytes, capped) = self.read_limited(&relative, MAX_READ_SCAN_BYTES, cancellation)?;
        let content = decode_text(&bytes, capped, &self.display(&relative))?;
        let normalized = content.replace("\r\n", "\n");
        let mut lines = normalized.split('\n').collect::<Vec<_>>();
        if capped && lines.len() > 1 {
            // A capped byte prefix commonly ends partway through a line. Do
            // not present that fragment as a complete line or claim its line
            // count is the source file's total.
            lines.pop();
        }

        let offset = number_param(parameters, "offset", 1).max(1) as usize;
        if offset > lines.len() {
            if capped {
                return Err(ToolError::new(format!(
                    "offset {offset} is beyond the first {} lines, which is all of {} this tool can read (the file exceeds {MAX_READ_SCAN_BYTES} bytes)",
                    lines.len(),
                    self.display(&relative)
                )));
            }
            return Err(ToolError::new(format!(
                "offset {offset} is beyond end of file ({} lines total)",
                lines.len()
            )));
        }

        let mut end = lines.len();
        let line_limit = number_param(parameters, "limit", 0);
        if line_limit > 0 {
            end = end.min(offset.saturating_sub(1).saturating_add(line_limit as usize));
        }
        let mut selected = lines[offset - 1..end].join("\n");
        let mut truncated = false;
        if selected.len() > MAX_READ_BYTES {
            selected = clip_utf8(&selected, MAX_READ_BYTES);
            if let Some(last_newline) = selected.rfind('\n') {
                selected.truncate(last_newline);
            }
            truncated = true;
        }

        let shown_lines = selected.bytes().filter(|byte| *byte == b'\n').count() + 1;
        let last_line = offset.saturating_add(shown_lines).saturating_sub(1);
        if truncated {
            selected.push_str(&format!(
                "\n\n[truncated: showing at most {MAX_READ_BYTES} bytes from line {offset}]"
            ));
        } else if last_line < lines.len() {
            let total = if capped {
                format!("at least {}", lines.len())
            } else {
                lines.len().to_string()
            };
            selected.push_str(&format!(
                "\n\n[Showing lines {offset}-{last_line} of {total}. Use offset={} to continue.]",
                last_line.saturating_add(1)
            ));
        } else if capped {
            selected.push_str(&format!(
                "\n\n[{} exceeds {MAX_READ_SCAN_BYTES} bytes; only its first {} lines are readable with this tool]",
                self.display(&relative),
                lines.len()
            ));
        }
        Ok(selected)
    }

    fn run_write(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let requested = required_string(parameters, "path")?;
        let content = required_string(parameters, "content")?;
        let relative = self.resolve(requested)?;
        let _mutation = FILE_MUTATIONS.acquire(self.mutation_key(&relative), cancellation)?;
        self.write_file_atomic(&relative, content.as_bytes(), cancellation)?;
        Ok(format!(
            "Wrote {} bytes to {}",
            content.len(),
            self.display(&relative)
        ))
    }

    /// Identifies a file for the mutation queue the way pi does: by its real
    /// path when it exists, else by the resolved path it will be created at.
    fn mutation_key(&self, relative: &Path) -> PathBuf {
        let absolute = self.root.join(relative);
        canonicalize(&absolute).unwrap_or(absolute)
    }

    fn run_edit(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let requested = required_string(parameters, "path")?;
        let old_text = required_string(parameters, "old_text")?;
        let new_text = required_string(parameters, "new_text")?;
        if old_text.is_empty() {
            return Err(ToolError::new("old_text must not be empty"));
        }

        let relative = self.resolve(requested)?;
        let _mutation = FILE_MUTATIONS.acquire(self.mutation_key(&relative), cancellation)?;
        let (bytes, truncated) = self.read_limited(&relative, MAX_EDIT_BYTES, cancellation)?;
        if truncated {
            return Err(ToolError::new(format!(
                "{} exceeds the {MAX_EDIT_BYTES}-byte edit limit",
                self.display(&relative)
            )));
        }
        let raw_content = decode_text(&bytes, false, &self.display(&relative))?;

        // Match the way pi does: the model never includes an invisible BOM
        // in old_text and writes LF regardless of the file's line endings, so
        // compare on LF-normalised text and restore the original ending.
        let (bom, content) = split_bom(&raw_content);
        let line_ending = detect_line_ending(content);
        let content = normalize_to_lf(content);
        let old_text = normalize_to_lf(old_text);
        let new_text = normalize_to_lf(new_text);

        let mut occurrences = 0usize;
        for _ in content.match_indices(old_text.as_str()) {
            occurrences += 1;
            if occurrences.is_multiple_of(1_024) {
                check_cancelled(cancellation)?;
            }
        }
        match occurrences {
            0 => {
                return Err(ToolError::new(format!(
                    "old_text was not found in {}",
                    self.display(&relative)
                )));
            }
            1 => {}
            count => {
                return Err(ToolError::new(format!(
                    "old_text appears {count} times in {}; add more context to make it unique",
                    self.display(&relative)
                )));
            }
        }

        let index = content
            .find(old_text.as_str())
            .expect("the unique old_text occurrence was counted");
        let mut updated = String::with_capacity(
            content
                .len()
                .saturating_sub(old_text.len())
                .saturating_add(new_text.len()),
        );
        updated.push_str(&content[..index]);
        updated.push_str(&new_text);
        updated.push_str(&content[index + old_text.len()..]);
        let mut restored = String::with_capacity(bom.len() + updated.len());
        restored.push_str(bom);
        restored.push_str(&restore_line_endings(&updated, line_ending));
        self.write_file_atomic(&relative, restored.as_bytes(), cancellation)?;
        Ok(format!("Edited {}", self.display(&relative)))
    }

    fn run_list(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
        limit: Option<usize>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let requested = optional_string(parameters, "path")?.unwrap_or(".");
        let relative = if requested.is_empty() {
            PathBuf::from(".")
        } else {
            self.resolve(requested)?
        };
        let absolute = self.existing_path(&relative)?;
        let metadata = fs::symlink_metadata(&absolute)
            .map_err(|error| ToolError::io("inspect directory", &absolute, error))?;
        if !metadata.is_dir() {
            return Err(ToolError::new(format!(
                "{} is not a directory",
                self.display(&relative)
            )));
        }

        let (mut entries, overflowed) =
            read_directory_entries(&absolute, MAX_DIRECTORY_ENTRIES, cancellation)?;
        entries.sort();
        if entries.is_empty() {
            return Ok("(empty directory)".to_owned());
        }

        let reached_limit = match limit {
            Some(entry_limit) if entries.len() > entry_limit => {
                entries.truncate(entry_limit);
                Some(entry_limit)
            }
            _ => None,
        };
        let mut output = render_lines(&entries, MAX_LIST_OUTPUT_BYTES);
        if let Some(entry_limit) = reached_limit {
            output.push_str(&format!("\n\n[{entry_limit} entries limit reached]"));
        } else if overflowed {
            output.push_str(&format!(
                "\n\n[{} has more than {MAX_DIRECTORY_ENTRIES} entries; only {MAX_DIRECTORY_ENTRIES} were listed]",
                self.display(&relative)
            ));
        }
        Ok(output)
    }

    fn run_grep(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let pattern = required_string(parameters, "pattern")?;
        if pattern.is_empty() {
            return Err(ToolError::new("pattern is required"));
        }
        let ignore_case = bool_param(parameters, "ignoreCase", false)?;
        let literal = bool_param(parameters, "literal", false)?;
        let expression = if literal {
            SearchRegex::literal(pattern, ignore_case)?
        } else {
            SearchRegex::compile(pattern, ignore_case)?
        };

        let requested = optional_string(parameters, "path")?.unwrap_or(".");
        let search_root = if requested.is_empty() {
            PathBuf::from(".")
        } else {
            self.resolve(requested)?
        };
        let candidates = self.candidate_files(cancellation, &search_root)?;
        let glob = optional_string(parameters, "glob")?
            .filter(|value| !value.is_empty())
            .map(compile_glob)
            .transpose()?;
        let match_limit = positive_limit(parameters, "limit", 100, MAX_CANDIDATE_FILES);
        let context = number_param(parameters, "context", 0).max(0) as usize;
        let context = context.min(MAX_CONTEXT_LINES);

        let mut output = BoundedText::new(MAX_LIST_OUTPUT_BYTES);
        let mut matches = 0usize;
        let candidate_notice = candidate_truncation_notice(candidates.truncated);
        for candidate in candidates.files {
            check_cancelled(cancellation)?;
            let display = match path_to_slash(&candidate) {
                Ok(display) => display,
                Err(_) => continue,
            };
            if glob.as_ref().is_some_and(|glob| !glob.is_match(&display)) {
                continue;
            }

            let (bytes, truncated) =
                match self.read_limited(&candidate, MAX_SEARCH_FILE_BYTES, cancellation) {
                    Ok(read) => read,
                    // Candidate enumeration is best effort: a removed,
                    // unreadable, special, or symlinked file must not abort
                    // an otherwise useful search.
                    Err(_) => continue,
                };
            if truncated || bytes.contains(&0) {
                continue;
            }
            let text = match decode_text(&bytes, false, &display) {
                Ok(text) => text,
                Err(_) => continue,
            };
            let normalized = text.replace("\r\n", "\n");
            let lines = normalized.split('\n').collect::<Vec<_>>();

            for (index, line) in lines.iter().enumerate() {
                if index % 256 == 0 {
                    check_cancelled(cancellation)?;
                }
                if !expression.is_match(line, cancellation)? {
                    continue;
                }
                matches += 1;
                let start = index.saturating_sub(context);
                let end = index
                    .saturating_add(context)
                    .min(lines.len().saturating_sub(1));
                for (row, source_line) in lines
                    .iter()
                    .enumerate()
                    .take(end.saturating_add(1))
                    .skip(start)
                {
                    let separator = if row == index { ':' } else { '-' };
                    let mut value = (*source_line).to_owned();
                    if value.len() > MAX_GREP_LINE_BYTES {
                        value = clip_utf8(&value, MAX_GREP_LINE_BYTES);
                        value.push('…');
                    }
                    output.push_line(&format!(
                        "{display}{separator}{}{separator} {value}",
                        row + 1
                    ));
                }
                if matches >= match_limit {
                    output.push("\n\n");
                    output.push(&format!(
                        "[{match_limit} matches limit reached. Refine the pattern or increase limit.]"
                    ));
                    return Ok(output.finish() + &candidate_notice);
                }
            }
        }
        if matches == 0 {
            Ok(format!("No matches found{candidate_notice}"))
        } else {
            Ok(output.finish() + &candidate_notice)
        }
    }

    fn run_find(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let pattern = required_string(parameters, "pattern")?;
        if pattern.is_empty() {
            return Err(ToolError::new("pattern is required"));
        }
        let glob = compile_glob(pattern)?;
        let requested = optional_string(parameters, "path")?.unwrap_or(".");
        let search_root = if requested.is_empty() {
            PathBuf::from(".")
        } else {
            self.resolve(requested)?
        };
        let candidates = self.candidate_files(cancellation, &search_root)?;
        let limit = positive_limit(parameters, "limit", 1_000, MAX_CANDIDATE_FILES);
        let candidate_notice = candidate_truncation_notice(candidates.truncated);

        let mut matches = Vec::new();
        let mut limited = false;
        for candidate in candidates.files {
            check_cancelled(cancellation)?;
            let relative = relative_to_search(&candidate, &search_root);
            let display = match path_to_slash(&relative) {
                Ok(display) => display,
                Err(_) => continue,
            };
            if !glob.is_match(&display) {
                continue;
            }
            matches.push(display);
            if matches.len() >= limit {
                limited = true;
                break;
            }
        }
        if matches.is_empty() {
            return Ok(format!("No files found matching pattern{candidate_notice}"));
        }
        matches.sort();
        let mut output = render_lines(&matches, MAX_LIST_OUTPUT_BYTES);
        if limited {
            output.push_str(&format!(
                "\n\n[{limit} results limit reached. Refine the pattern or increase limit.]"
            ));
        }
        output.push_str(&candidate_notice);
        Ok(output)
    }

    fn run_bash(
        &self,
        cancellation: &CancellationToken,
        parameters: &BTreeMap<String, Value>,
    ) -> Result<String> {
        check_cancelled(cancellation)?;
        let source = required_string(parameters, "command")?;
        if source.trim().is_empty() {
            return Err(ToolError::new("command is required"));
        }

        #[cfg(windows)]
        let mut command = match find_bash_on_path() {
            // Like the Go port, prefer a real bash (Git for Windows, MSYS2)
            // so commands written for pi behave the same on every platform.
            Some(bash) => {
                let mut command = Command::new(bash);
                command.args(["-c", source]);
                command
            }
            None => {
                use std::os::windows::process::CommandExt;

                // cmd.exe has no argv: std's per-argument quoting would
                // mangle the command, so hand it the exact `/s /c "..."`
                // command line that cmd itself documents.
                let mut command = Command::new("cmd.exe");
                command
                    .args(["/d", "/s", "/c"])
                    .raw_arg(format!("\"{source}\""));
                command
            }
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", source]);
            command
        };
        command.current_dir(&self.root);

        let timeout = if self.bash_timeout.is_zero() {
            DEFAULT_BASH_TIMEOUT
        } else {
            self.bash_timeout
        };
        let result = run_process(&mut command, cancellation, timeout, MAX_OUTPUT_BYTES, true)
            .map_err(|error| ToolError::io("run command", Path::new(source), error))?;
        let output = result.output.render_text();

        if result.cancelled || cancellation.is_cancelled() {
            return Err(with_output("command cancelled", &output));
        }
        if result.timed_out {
            return Err(with_output(
                &format!("command timed out after {timeout:?}"),
                &output,
            ));
        }
        if let Some(error) = result.reader_error {
            return Err(with_output(
                &format!("failed to read command output: {error}"),
                &output,
            ));
        }
        if !result.status.success() {
            return Err(with_output(
                &format!("command failed with {}", result.status),
                &output,
            ));
        }
        if result.output_open {
            if output.trim().is_empty() {
                return Ok("(no output; a background process is still running)".to_owned());
            }
            return Ok(format!(
                "{output}\n\n[a background process started by this command is still running]"
            ));
        }
        if output.trim().is_empty() {
            Ok("(no output)".to_owned())
        } else {
            Ok(output)
        }
    }

    /// Resolves a user supplied path to a normalized path relative to root.
    fn resolve(&self, requested: &str) -> Result<PathBuf> {
        if requested.is_empty() {
            return Err(ToolError::new("path is required"));
        }
        if requested.contains('\0') {
            return Err(ToolError::new("path must not contain a NUL byte"));
        }

        let path = Path::new(requested);
        let relative = if path.is_absolute() {
            // Canonicalize an existing absolute path first so `root/link`
            // cannot disguise an outside target. For a new path, lexical
            // prefix validation is followed by checked parent creation.
            if let Ok(canonical) = canonicalize(path) {
                canonical
                    .strip_prefix(&self.root)
                    .map_err(|_| {
                        ToolError::new(format!("path {requested} is outside the workspace"))
                    })?
                    .to_path_buf()
            } else {
                path.strip_prefix(&self.root)
                    .map_err(|_| {
                        ToolError::new(format!("path {requested} is outside the workspace"))
                    })?
                    .to_path_buf()
            }
        } else {
            path.to_path_buf()
        };
        normalize_relative(&relative).map_err(|error| {
            ToolError::new(format!(
                "path {requested} is outside the workspace: {error}"
            ))
        })
    }

    /// Checks an existing path and rejects every symlinked component.
    fn existing_path(&self, relative: &Path) -> Result<PathBuf> {
        let absolute = self.root.join(relative);
        self.reject_symlink_components(relative)?;
        let canonical = canonicalize(&absolute)
            .map_err(|error| ToolError::io("resolve path", &absolute, error))?;
        if !canonical.starts_with(&self.root) {
            return Err(ToolError::new(format!(
                "path {} is outside the workspace",
                self.display(relative)
            )));
        }
        Ok(absolute)
    }

    fn reject_symlink_components(&self, relative: &Path) -> Result<()> {
        let mut current = self.root.clone();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                continue;
            };
            current.push(name);
            let metadata = fs::symlink_metadata(&current)
                .map_err(|error| ToolError::io("inspect path", &current, error))?;
            if metadata.file_type().is_symlink() {
                return Err(ToolError::new(format!(
                    "path {} contains a symlinked component",
                    self.display(relative)
                )));
            }
        }
        Ok(())
    }

    /// Creates checked parent directories and returns the destination path.
    fn prepare_write_path(&self, relative: &Path) -> Result<PathBuf> {
        if relative == Path::new(".") {
            return Err(ToolError::new("path must name a file"));
        }
        let parent = relative.parent().unwrap_or_else(|| Path::new("."));
        let mut current = self.root.clone();
        for component in parent.components() {
            let Component::Normal(name) = component else {
                continue;
            };
            current.push(name);
            match fs::symlink_metadata(&current) {
                Ok(metadata) => {
                    if metadata.file_type().is_symlink() {
                        return Err(ToolError::new(format!(
                            "path {} contains a symlinked component",
                            self.display(relative)
                        )));
                    }
                    if !metadata.is_dir() {
                        return Err(ToolError::new(format!(
                            "parent {} is not a directory",
                            current.display()
                        )));
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    match fs::create_dir(&current) {
                        Ok(()) => {}
                        Err(create_error)
                            if create_error.kind() == io::ErrorKind::AlreadyExists => {}
                        Err(create_error) => {
                            return Err(ToolError::io(
                                "create parent directory",
                                &current,
                                create_error,
                            ));
                        }
                    }
                    let metadata = fs::symlink_metadata(&current).map_err(|error| {
                        ToolError::io("inspect parent directory", &current, error)
                    })?;
                    if metadata.file_type().is_symlink() || !metadata.is_dir() {
                        return Err(ToolError::new(format!(
                            "parent {} is not a safe directory",
                            current.display()
                        )));
                    }
                }
                Err(error) => {
                    return Err(ToolError::io("inspect parent directory", &current, error));
                }
            }
            let canonical = canonicalize(&current)
                .map_err(|error| ToolError::io("resolve parent directory", &current, error))?;
            if !canonical.starts_with(&self.root) {
                return Err(ToolError::new(format!(
                    "path {} is outside the workspace",
                    self.display(relative)
                )));
            }
        }

        let destination = self.root.join(relative);
        match fs::symlink_metadata(&destination) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(ToolError::new(format!(
                        "path {} is a symlink",
                        self.display(relative)
                    )));
                }
                if !metadata.is_file() {
                    return Err(ToolError::new(format!(
                        "path {} is not a regular file",
                        self.display(relative)
                    )));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(ToolError::io("inspect destination", &destination, error)),
        }
        Ok(destination)
    }

    fn read_limited(
        &self,
        relative: &Path,
        limit: usize,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<u8>, bool)> {
        check_cancelled(cancellation)?;
        let absolute = self.existing_path(relative)?;
        let metadata = fs::metadata(&absolute)
            .map_err(|error| ToolError::io("inspect file", &absolute, error))?;
        if !metadata.is_file() {
            return Err(ToolError::new(format!(
                "{} is not a regular file",
                self.display(relative)
            )));
        }
        let mut file =
            File::open(&absolute).map_err(|error| ToolError::io("open file", &absolute, error))?;
        let mut bytes = Vec::with_capacity(limit.saturating_add(1).min(64 * 1024));
        Read::by_ref(&mut file)
            .take(limit.saturating_add(1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|error| ToolError::io("read file", &absolute, error))?;
        check_cancelled(cancellation)?;
        let capped = bytes.len() > limit;
        if capped {
            bytes.truncate(limit);
        }
        Ok((bytes, capped))
    }

    /// Writes a sibling temporary file, syncs it, then atomically renames it.
    fn write_file_atomic(
        &self,
        relative: &Path,
        content: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<()> {
        check_cancelled(cancellation)?;
        let destination = self.prepare_write_path(relative)?;
        let inherited_permissions = match fs::symlink_metadata(&destination) {
            Ok(metadata) => Some(inheritable_permissions(metadata.permissions())),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(ToolError::io("inspect destination", &destination, error)),
        };
        let parent = destination
            .parent()
            .ok_or_else(|| ToolError::new("destination has no parent directory"))?;

        for attempt in 0..32_u32 {
            let temporary = temporary_path(parent, &destination, attempt);
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let temporary_file = match options.open(&temporary) {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(ToolError::io("create temporary file", &temporary, error));
                }
            };
            let mut temporary_file = Some(temporary_file);

            let write_result = (|| -> Result<()> {
                {
                    let file = temporary_file
                        .as_mut()
                        .expect("temporary file is available until the rename");
                    file.write_all(content).map_err(|error| {
                        ToolError::io("write temporary file", &temporary, error)
                    })?;
                    check_cancelled(cancellation)?;
                    if let Some(permissions) = inherited_permissions.as_ref() {
                        file.set_permissions(permissions.clone()).map_err(|error| {
                            ToolError::io("preserve destination permissions", &temporary, error)
                        })?;
                    } else {
                        set_new_file_permissions(file, &temporary)?;
                    }
                    file.sync_all()
                        .map_err(|error| ToolError::io("sync temporary file", &temporary, error))?;
                }
                check_cancelled(cancellation)?;
                drop(temporary_file.take());
                fs::rename(&temporary, &destination)
                    .map_err(|error| ToolError::io("replace destination", &destination, error))?;
                // Syncing the directory improves crash durability on Unix.
                // It is best effort because not every filesystem permits it.
                sync_directory_best_effort(parent);
                Ok(())
            })();

            if write_result.is_err() {
                // The file handle has been dropped before cleanup on every
                // error path except write/sync. Dropping it here is harmless.
                drop(temporary_file.take());
                let _ = fs::remove_file(&temporary);
            }
            return write_result;
        }

        Err(ToolError::new(
            "could not allocate a unique temporary file after 32 attempts",
        ))
    }

    fn candidate_files(
        &self,
        cancellation: &CancellationToken,
        search_root: &Path,
    ) -> Result<Candidates> {
        self.collect_candidates(cancellation, search_root, MAX_CANDIDATE_FILES)
    }

    fn collect_candidates(
        &self,
        cancellation: &CancellationToken,
        search_root: &Path,
        maximum_files: usize,
    ) -> Result<Candidates> {
        check_cancelled(cancellation)?;
        let absolute = self.existing_path(search_root)?;
        let metadata = fs::symlink_metadata(&absolute)
            .map_err(|error| ToolError::io("inspect search path", &absolute, error))?;
        if metadata.is_file() {
            return Ok(Candidates {
                files: vec![search_root.to_path_buf()],
                truncated: false,
            });
        }
        if !metadata.is_dir() {
            return Err(ToolError::new(format!(
                "not a regular file or directory: {}",
                self.display(search_root)
            )));
        }

        if let Some(candidates) =
            self.git_candidate_files(cancellation, search_root, maximum_files)?
        {
            return Ok(candidates);
        }
        self.walk_candidate_files(cancellation, search_root, maximum_files)
    }

    /// Uses git's ignored-file-aware index when it is available.
    fn git_candidate_files(
        &self,
        cancellation: &CancellationToken,
        search_root: &Path,
        maximum_files: usize,
    ) -> Result<Option<Candidates>> {
        let search_argument = search_root.as_os_str();
        let mut command = Command::new("git");
        command
            .current_dir(&self.root)
            .arg("-C")
            .arg(&self.root)
            .args(["ls-files", "-z", "-co", "--exclude-standard", "--"])
            .arg(search_argument);
        for key in [
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_COMMON_DIR",
        ] {
            command.env_remove(key);
        }

        let result = match run_process(
            &mut command,
            cancellation,
            GIT_TIMEOUT,
            MAX_CANDIDATE_BYTES,
            false,
        ) {
            Ok(result) => result,
            // Git may be absent, the workspace may not be a repository, or
            // the executable may fail to start. A bounded recursive walk is
            // the safe fallback in each case.
            Err(_) => return Ok(None),
        };
        if result.cancelled || cancellation.is_cancelled() {
            return Err(ToolError::new("operation cancelled"));
        }
        if result.timed_out || result.output_open || !result.status.success() {
            return Ok(None);
        }
        if let Some(error) = result.reader_error {
            return Err(ToolError::new(format!(
                "could not read git candidate list: {error}"
            )));
        }
        let mut listing = result.output.bytes.as_slice();
        let mut truncated = false;
        if result.output.truncated {
            // The byte cap cut the list mid-entry; keep every complete
            // NUL-terminated entry and report the rest as dropped.
            truncated = true;
            let complete = listing
                .iter()
                .rposition(|byte| *byte == 0)
                .map_or(0, |i| i + 1);
            listing = &listing[..complete];
        }

        let mut files = Vec::new();
        for entry in listing.split(|byte| *byte == 0) {
            if entry.is_empty() {
                continue;
            }
            let entry = std::str::from_utf8(entry).map_err(|_| {
                ToolError::new(
                    "git returned a non-UTF-8 candidate path, which this text tool cannot display",
                )
            })?;
            let relative = normalize_relative(Path::new(entry))
                .map_err(|_| ToolError::new("git returned an unsafe candidate path"))?;
            if relative == Path::new(".") {
                continue;
            }
            if !is_under_search_root(&relative, search_root) {
                // Do not accept a surprising file list even if git was
                // launched from a valid workspace. Fall back to a walk that
                // has an independently checked root.
                return Ok(None);
            }
            // A tracked file deleted from the worktree stays in the index
            // until the deletion is staged; listing it would send the model
            // to a path that no longer exists.
            if fs::symlink_metadata(self.root.join(&relative))
                .is_err_and(|error| error.kind() == io::ErrorKind::NotFound)
            {
                continue;
            }
            if files.len() >= maximum_files {
                truncated = true;
                break;
            }
            files.push(relative);
        }
        files.sort();
        Ok(Some(Candidates { files, truncated }))
    }

    /// Conservative, non-symlink-following candidate discovery.
    fn walk_candidate_files(
        &self,
        cancellation: &CancellationToken,
        search_root: &Path,
        maximum_files: usize,
    ) -> Result<Candidates> {
        let mut directories = VecDeque::from([search_root.to_path_buf()]);
        let mut files = Vec::new();
        let mut truncated = false;
        'directories: while let Some(directory_relative) = directories.pop_front() {
            check_cancelled(cancellation)?;
            let directory_absolute = self.root.join(&directory_relative);
            let entries = match fs::read_dir(&directory_absolute) {
                Ok(entries) => entries,
                // The search root itself must be readable; a subdirectory
                // that is not (permissions, removed mid-walk) is skipped so
                // the rest of the tree still gets searched.
                Err(error) if directory_relative == search_root => {
                    return Err(ToolError::io(
                        "read search directory",
                        &directory_absolute,
                        error,
                    ));
                }
                Err(_) => continue,
            };
            for entry in entries {
                check_cancelled(cancellation)?;
                let Ok(entry) = entry else {
                    continue;
                };
                let name = entry.file_name();
                let relative = join_relative(&directory_relative, &name);
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if file_type.is_dir() {
                    if !skip_search_dir(&name) {
                        let Ok(canonical) = canonicalize(&entry.path()) else {
                            continue;
                        };
                        if !canonical.starts_with(&self.root) {
                            return Err(ToolError::new(format!(
                                "search directory {} is outside the workspace",
                                relative.display()
                            )));
                        }
                        directories.push_back(relative);
                    }
                    continue;
                }
                if files.len() >= maximum_files {
                    truncated = true;
                    break 'directories;
                }
                files.push(relative);
            }
        }
        files.sort();
        Ok(Candidates { files, truncated })
    }

    fn display(&self, relative: &Path) -> String {
        path_to_slash(relative).unwrap_or_else(|_| relative.to_string_lossy().into_owned())
    }
}

/// Convenience constructor mirroring the Go package's `NewWorkspace`.
pub fn new_workspace(root: impl AsRef<Path>) -> Result<Workspace> {
    Workspace::new(root)
}

/// Candidate paths for a search plus whether the discovery cap dropped some.
struct Candidates {
    files: Vec<PathBuf>,
    truncated: bool,
}

fn candidate_truncation_notice(truncated: bool) -> String {
    if truncated {
        format!(
            "\n\n[only the first {MAX_CANDIDATE_FILES} candidate files were searched; narrow the path to search the rest]"
        )
    } else {
        String::new()
    }
}

/// Reads at most `maximum` names from a directory, reporting whether more
/// remained so a huge directory is listed partially instead of not at all.
fn read_directory_entries(
    absolute: &Path,
    maximum: usize,
    cancellation: &CancellationToken,
) -> Result<(Vec<String>, bool)> {
    let mut entries = Vec::new();
    let directory =
        fs::read_dir(absolute).map_err(|error| ToolError::io("read directory", absolute, error))?;
    for entry in directory {
        check_cancelled(cancellation)?;
        if entries.len() >= maximum {
            return Ok((entries, true));
        }
        let entry =
            entry.map_err(|error| ToolError::io("read directory entry", absolute, error))?;
        let mut name = entry.file_name().to_string_lossy().into_owned();
        if entry
            .file_type()
            .map_err(|error| ToolError::io("inspect directory entry", &entry.path(), error))?
            .is_dir()
        {
            name.push('/');
        }
        entries.push(name);
    }
    Ok((entries, false))
}

/// `fs::canonicalize` with Windows verbatim prefixes (`\\?\C:\...`) reduced
/// to ordinary paths, so canonical results compare with lexically built ones
/// and a new file under `C:\ws` is not reported as outside `\\?\C:\ws`.
pub(crate) fn canonicalize(path: &Path) -> io::Result<PathBuf> {
    let canonical = fs::canonicalize(path)?;
    #[cfg(windows)]
    {
        Ok(simplify_verbatim(&canonical))
    }
    #[cfg(not(windows))]
    {
        Ok(canonical)
    }
}

#[cfg(any(windows, test))]
fn simplify_verbatim(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\?\")
        && let Some((drive, colon)) = rest.chars().next().zip(rest.chars().nth(1))
        && drive.is_ascii_alphabetic()
        && colon == ':'
    {
        return PathBuf::from(rest);
    }
    path.to_path_buf()
}

/// Locates a usable `bash.exe` on `PATH`, ignoring the legacy WSL launcher
/// in `System32`, which pi also special-cases because it does not accept a
/// command as `-c` argv the way a real bash does.
#[cfg(windows)]
fn find_bash_on_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|directory| directory.join("bash.exe"))
        .find(|candidate| {
            candidate.is_file()
                && candidate
                    .to_str()
                    .is_none_or(|text| !is_legacy_wsl_bash(text))
        })
}

#[cfg(any(windows, test))]
fn is_legacy_wsl_bash(path: &str) -> bool {
    let normalized = path.replace('/', "\\").to_ascii_lowercase();
    let mut characters = normalized.chars();
    let Some(drive) = characters.next() else {
        return false;
    };
    if !drive.is_ascii_alphabetic() {
        return false;
    }
    let rest = characters.as_str();
    rest == r":\windows\system32\bash.exe" || rest == r":\windows\sysnative\bash.exe"
}

/// Permission bits carried over from a replaced file. The setuid, setgid and
/// sticky bits are deliberately dropped: an atomic replace creates a new
/// inode, and silently re-applying them would let an edit mint a setuid
/// binary with the model's content.
fn inheritable_permissions(permissions: fs::Permissions) -> fs::Permissions {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = permissions;
        permissions.set_mode(permissions.mode() & 0o777);
        permissions
    }
    #[cfg(not(unix))]
    {
        permissions
    }
}

/// Splits a leading byte-order mark off decoded text, as pi does before
/// matching `old_text`.
fn split_bom(content: &str) -> (&str, &str) {
    match content.strip_prefix('\u{FEFF}') {
        Some(text) => ("\u{FEFF}", text),
        None => ("", content),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LineEnding {
    Lf,
    CrLf,
}

/// pi's rule: the file uses CRLF when its first line break is one.
fn detect_line_ending(content: &str) -> LineEnding {
    match (content.find("\r\n"), content.find('\n')) {
        (Some(crlf), Some(lf)) if crlf < lf => LineEnding::CrLf,
        _ => LineEnding::Lf,
    }
}

fn normalize_to_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn restore_line_endings(text: &str, ending: LineEnding) -> String {
    match ending {
        LineEnding::CrLf => text.replace('\n', "\r\n"),
        LineEnding::Lf => text.to_owned(),
    }
}

/// Kills the child's whole process tree, as pi's `killProcessTree` does, and
/// returns whether the tree (not just the immediate child) is known dead.
fn kill_process_tree(child: &mut Child) -> bool {
    #[cfg(unix)]
    let group_killed = {
        // `process_group(0)` made the child lead a group whose id equals its
        // pid, so one negative-pid signal reaches every descendant.
        let pid = i32::try_from(child.id()).unwrap_or(0);
        // SAFETY: kill(2) takes two plain integers and has no memory
        // preconditions; a stale pid only yields ESRCH.
        pid > 0 && unsafe { kill(-pid, SIGKILL) } == 0
    };
    #[cfg(windows)]
    let group_killed = {
        // Windows has no process-group signal; taskkill /T walks the tree.
        // Use the System32 copy so cleanup does not depend on PATH.
        let system_root =
            std::env::var_os("SystemRoot").unwrap_or_else(|| OsStr::new(r"C:\Windows").to_owned());
        let taskkill = Path::new(&system_root)
            .join("System32")
            .join("taskkill.exe");
        Command::new(taskkill)
            .args(["/F", "/T", "/PID", &child.id().to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    };
    #[cfg(not(any(unix, windows)))]
    let group_killed = false;
    // Always also kill the immediate child: harmless when the group kill
    // already reached it, and the only fallback when it did not.
    let _ = child.kill();
    group_killed
}

fn required_string<'a>(parameters: &'a BTreeMap<String, Value>, name: &str) -> Result<&'a str> {
    match parameters.get(name) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(ToolError::new(format!("{name:?} must be a string"))),
        None => Err(ToolError::new(format!("{name:?} is required"))),
    }
}

fn optional_string<'a>(
    parameters: &'a BTreeMap<String, Value>,
    name: &str,
) -> Result<Option<&'a str>> {
    match parameters.get(name) {
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(ToolError::new(format!("{name:?} must be a string"))),
        None => Ok(None),
    }
}

fn bool_param(parameters: &BTreeMap<String, Value>, name: &str, fallback: bool) -> Result<bool> {
    match parameters.get(name) {
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(ToolError::new(format!("{name:?} must be a boolean"))),
        None => Ok(fallback),
    }
}

fn number_param(parameters: &BTreeMap<String, Value>, name: &str, fallback: i64) -> i64 {
    let Some(value) = parameters.get(name) else {
        return fallback;
    };
    if let Some(value) = value.as_i64() {
        return value;
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value).unwrap_or(i64::MAX);
    }
    if let Some(value) = value.as_f64()
        && value.is_finite()
    {
        return value.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
    }
    value
        .as_str()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(fallback)
}

fn positive_limit(
    parameters: &BTreeMap<String, Value>,
    name: &str,
    fallback: usize,
    maximum: usize,
) -> usize {
    let value = number_param(parameters, name, fallback as i64);
    if value <= 0 {
        fallback
    } else {
        usize::try_from(value).unwrap_or(maximum).min(maximum)
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(ToolError::new("operation cancelled"))
    } else {
        Ok(())
    }
}

fn normalize_relative(path: &Path) -> std::result::Result<PathBuf, &'static str> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => normalized.push(name),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err("parent traversal is not allowed");
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err("absolute paths are not allowed here");
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        Ok(PathBuf::from("."))
    } else {
        Ok(normalized)
    }
}

fn path_to_slash(path: &Path) -> Result<String> {
    let text = path
        .to_str()
        .ok_or_else(|| ToolError::new("path is not valid UTF-8"))?;
    #[cfg(windows)]
    {
        Ok(text.replace('\\', "/"))
    }
    #[cfg(not(windows))]
    {
        Ok(text.to_owned())
    }
}

fn join_relative(parent: &Path, child: &OsStr) -> PathBuf {
    if parent == Path::new(".") {
        PathBuf::from(child)
    } else {
        parent.join(child)
    }
}

fn is_under_search_root(candidate: &Path, search_root: &Path) -> bool {
    search_root == Path::new(".") || candidate == search_root || candidate.starts_with(search_root)
}

fn relative_to_search(candidate: &Path, search_root: &Path) -> PathBuf {
    if search_root == Path::new(".") {
        return candidate.to_path_buf();
    }
    match candidate.strip_prefix(search_root) {
        Ok(relative) if !relative.as_os_str().is_empty() => relative.to_path_buf(),
        _ => candidate.to_path_buf(),
    }
}

fn skip_search_dir(name: &OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(
            ".git"
                | "node_modules"
                | "vendor"
                | "dist"
                | "build"
                | ".venv"
                | "target"
                | "__pycache__"
        )
    )
}

/// Clips a string at a Unicode scalar boundary.
pub fn clip_utf8(value: &str, maximum_bytes: usize) -> String {
    utf8_prefix(value, maximum_bytes).to_owned()
}

fn utf8_prefix(value: &str, maximum_bytes: usize) -> &str {
    if value.len() <= maximum_bytes {
        return value;
    }
    let mut end = maximum_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn decode_text(bytes: &[u8], truncated_at_end: bool, display: &str) -> Result<String> {
    match std::str::from_utf8(bytes) {
        Ok(text) => Ok(text.to_owned()),
        Err(error) if truncated_at_end && error.error_len().is_none() => {
            Ok(std::str::from_utf8(&bytes[..error.valid_up_to()])
                .expect("the UTF-8 error's valid prefix must be valid")
                .to_owned())
        }
        Err(_) => Err(ToolError::new(format!("{display} is not valid UTF-8 text"))),
    }
}

fn temporary_path(parent: &Path, destination: &Path, attempt: u32) -> PathBuf {
    let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let base = destination
        .file_name()
        .unwrap_or_else(|| OsStr::new("file"))
        .to_string_lossy();
    // The nonce already makes the name unique; the destination name is only
    // a hint, so clip it before a long name pushes the sibling past
    // NAME_MAX and the write fails with ENAMETOOLONG.
    let base = utf8_prefix(&base, TEMP_NAME_PREFIX_BYTES);
    parent.join(format!(
        ".{base}.goshcoder-{:x}-{timestamp:x}-{sequence:x}-{attempt:x}.tmp",
        std::process::id()
    ))
}

fn set_new_file_permissions(file: &File, path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o644))
            .map_err(|error| ToolError::io("set new file permissions", path, error))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (file, path);
    }
    Ok(())
}

fn sync_directory_best_effort(path: &Path) {
    #[cfg(unix)]
    {
        if let Ok(directory) = File::open(path) {
            let _ = directory.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

fn with_output(prefix: &str, output: &str) -> ToolError {
    if output.is_empty() {
        ToolError::new(prefix)
    } else {
        ToolError::new(format!("{prefix}\n{output}"))
    }
}

/// A byte sink that always returns the caller's full write count while
/// retaining only a bounded prefix.
#[derive(Clone, Debug)]
struct CappedBytes {
    bytes: Vec<u8>,
    limit: usize,
    truncated: bool,
}

impl CappedBytes {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            truncated: false,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        let available = self.limit.saturating_sub(self.bytes.len());
        let copied = available.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..copied]);
        if copied < bytes.len() {
            self.truncated = true;
        }
    }

    fn render_text(&self) -> String {
        let prefix = match std::str::from_utf8(&self.bytes) {
            Ok(_) => self.bytes.as_slice(),
            Err(error) if self.truncated && error.error_len().is_none() => {
                &self.bytes[..error.valid_up_to()]
            }
            Err(_) => self.bytes.as_slice(),
        };
        let mut text = String::from_utf8_lossy(prefix).into_owned();
        if self.truncated {
            text.push_str("\n[output truncated]");
        }
        text
    }
}

/// A UTF-8 text sink with a bounded prefix and an explicit truncation notice.
struct BoundedText {
    text: String,
    limit: usize,
    truncated: bool,
}

impl BoundedText {
    fn new(limit: usize) -> Self {
        Self {
            text: String::new(),
            limit,
            truncated: false,
        }
    }

    fn push(&mut self, value: &str) {
        let available = self.limit.saturating_sub(self.text.len());
        if available == 0 {
            self.truncated |= !value.is_empty();
            return;
        }
        let prefix = utf8_prefix(value, available);
        self.text.push_str(prefix);
        if prefix.len() != value.len() {
            self.truncated = true;
        }
    }

    fn push_line(&mut self, value: &str) {
        if !self.text.is_empty() {
            self.push("\n");
        }
        self.push(value);
    }

    fn finish(mut self) -> String {
        if self.truncated {
            if !self.text.is_empty() {
                self.text.push('\n');
            }
            self.text.push_str("[output truncated]");
        }
        self.text
    }
}

fn render_lines(lines: &[String], limit: usize) -> String {
    let mut output = BoundedText::new(limit);
    for line in lines {
        output.push_line(line);
    }
    output.finish()
}

struct Reader {
    done: Receiver<std::result::Result<(), String>>,
    handle: Option<thread::JoinHandle<()>>,
}

fn spawn_reader<R>(mut reader: R, captured: Arc<Mutex<CappedBytes>>) -> Reader
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::sync_channel(1);
    let handle = thread::spawn(move || {
        let mut buffer = [0_u8; 8_192];
        let result = loop {
            match reader.read(&mut buffer) {
                Ok(0) => break Ok(()),
                Ok(read) => {
                    let mut output = captured.lock().unwrap_or_else(|error| error.into_inner());
                    output.push(&buffer[..read]);
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => break Err(error.to_string()),
            }
        };
        let _ = sender.send(result);
    });
    Reader {
        done: receiver,
        handle: Some(handle),
    }
}

fn await_reader(mut reader: Reader, deadline: Instant) -> (bool, Option<String>) {
    let remaining = deadline.saturating_duration_since(Instant::now());
    let received = if remaining.is_zero() {
        reader.done.try_recv().ok()
    } else {
        reader.done.recv_timeout(remaining).ok()
    };
    match received {
        Some(result) => {
            if let Some(handle) = reader.handle.take() {
                let _ = handle.join();
            }
            (false, result.err())
        }
        None => {
            // Dropping a JoinHandle detaches the reader. It keeps draining a
            // pipe held by a background grandchild without retaining
            // unbounded output or blocking this tool result forever.
            (true, None)
        }
    }
}

struct ProcessResult {
    status: ExitStatus,
    output: CappedBytes,
    timed_out: bool,
    cancelled: bool,
    output_open: bool,
    reader_error: Option<String>,
}

/// Runs a child while polling the agent cancellation token and a deadline.
///
/// The child leads its own process group so a timeout or cancellation tears
/// down everything it spawned, not just the shell. A command that exits
/// normally while a descendant it backgrounded still holds the pipe is
/// detected by the bounded drain grace and reported as a background-process
/// notice instead of hanging.
fn run_process(
    command: &mut Command,
    cancellation: &CancellationToken,
    timeout: Duration,
    output_limit: usize,
    combine_streams: bool,
) -> io::Result<ProcessResult> {
    // A tool command must never read the agent's own terminal: an inherited
    // stdin lets `cat` or an interactive prompt hang until the timeout and
    // steal keystrokes meant for the UI.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Detach from this console's Ctrl+C group so a Ctrl+C aimed at the
        // agent does not also interrupt the command the model is running.
        command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .expect("piped stdout must be available after spawn");
    let stderr = child
        .stderr
        .take()
        .expect("piped stderr must be available after spawn");
    let output = Arc::new(Mutex::new(CappedBytes::new(output_limit)));
    let stderr_output = if combine_streams {
        output.clone()
    } else {
        Arc::new(Mutex::new(CappedBytes::new(output_limit)))
    };
    let stdout_reader = spawn_reader(stdout, output.clone());
    let stderr_reader = spawn_reader(stderr, stderr_output);

    let started = Instant::now();
    let (status, timed_out, cancelled, group_killed) = loop {
        if let Some(status) = child.try_wait()? {
            break (status, false, cancellation.is_cancelled(), false);
        }
        if cancellation.is_cancelled() {
            let group_killed = kill_process_tree(&mut child);
            break (child.wait()?, false, true, group_killed);
        }
        if started.elapsed() >= timeout {
            let group_killed = kill_process_tree(&mut child);
            break (child.wait()?, true, false, group_killed);
        }
        thread::sleep(PROCESS_POLL_INTERVAL);
    };

    let drain_grace = if group_killed {
        KILLED_DRAIN_GRACE
    } else {
        OUTPUT_DRAIN_GRACE
    };
    let drain_deadline = Instant::now() + drain_grace;
    let (stdout_open, stdout_error) = await_reader(stdout_reader, drain_deadline);
    let (stderr_open, stderr_error) = await_reader(stderr_reader, drain_deadline);
    let output = output
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone();

    Ok(ProcessResult {
        status,
        output,
        timed_out,
        cancelled,
        output_open: stdout_open || stderr_open,
        reader_error: stdout_error.or(stderr_error),
    })
}

/// A compiled glob with pi-style basename fallback and Unicode-safe matching.
///
/// `{a,b}` groups are expanded up front into alternative token lists, which
/// is what fd and ripgrep (pi's find/grep backends) accept.
#[derive(Clone, Debug)]
pub struct GlobPattern {
    alternatives: Vec<Vec<GlobToken>>,
    basename_too: bool,
}

#[derive(Clone, Debug)]
enum GlobToken {
    Literal(char),
    One,
    Star,
    GlobStar,
    GlobStarSlash,
}

/// Compiles `*`, `**`, `**/`, `?`, and `{a,b}` glob syntax.
pub fn compile_glob(pattern: &str) -> Result<GlobPattern> {
    if pattern.chars().count() > MAX_GLOB_PATTERN_CHARS {
        return Err(ToolError::new(format!(
            "glob pattern exceeds {MAX_GLOB_PATTERN_CHARS} characters"
        )));
    }
    let normalized = normalize_glob_separators(pattern);
    let mut expanded = Vec::new();
    expand_glob_braces(&normalized, &mut expanded)?;
    Ok(GlobPattern {
        alternatives: expanded.iter().map(|text| tokenize_glob(text)).collect(),
        basename_too: !normalized.contains('/'),
    })
}

/// Expands the first balanced `{...}` group and recurses into the results so
/// nested groups and several groups in one pattern all multiply out. An
/// unbalanced brace is an ordinary character.
fn expand_glob_braces(pattern: &str, output: &mut Vec<String>) -> Result<()> {
    let Some((prefix, body, suffix)) = split_first_brace_group(pattern) else {
        output.push(pattern.to_owned());
        return Ok(());
    };
    for alternative in split_top_level_commas(body) {
        expand_glob_braces(&format!("{prefix}{alternative}{suffix}"), output)?;
        if output.len() > MAX_GLOB_ALTERNATIVES {
            return Err(ToolError::new(format!(
                "glob pattern expands to more than {MAX_GLOB_ALTERNATIVES} alternatives"
            )));
        }
    }
    Ok(())
}

fn split_first_brace_group(pattern: &str) -> Option<(&str, &str, &str)> {
    let open = pattern.find('{')?;
    let mut depth = 0usize;
    for (offset, character) in pattern[open..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let close = open + offset;
                    return Some((
                        &pattern[..open],
                        &pattern[open + 1..close],
                        &pattern[close + 1..],
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_commas(body: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (offset, character) in body.char_indices() {
        match character {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&body[start..offset]);
                start = offset + 1;
            }
            _ => {}
        }
    }
    parts.push(&body[start..]);
    parts
}

fn tokenize_glob(pattern: &str) -> Vec<GlobToken> {
    let characters = pattern.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < characters.len() {
        match characters[index] {
            '*' if characters.get(index + 1) == Some(&'*') => {
                index += 2;
                if characters.get(index) == Some(&'/') {
                    index += 1;
                    tokens.push(GlobToken::GlobStarSlash);
                } else {
                    tokens.push(GlobToken::GlobStar);
                }
            }
            '*' => {
                index += 1;
                tokens.push(GlobToken::Star);
            }
            '?' => {
                index += 1;
                tokens.push(GlobToken::One);
            }
            literal => {
                index += 1;
                tokens.push(GlobToken::Literal(literal));
            }
        }
    }
    tokens
}

impl GlobPattern {
    /// Returns whether a slash-separated candidate path matches this glob.
    pub fn is_match(&self, name: &str) -> bool {
        let normalized = normalize_glob_separators(name);
        self.matches_full(&normalized)
            || (self.basename_too
                && normalized
                    .rsplit('/')
                    .next()
                    .is_some_and(|base| self.matches_full(base)))
    }

    fn matches_full(&self, name: &str) -> bool {
        self.alternatives
            .iter()
            .any(|tokens| glob_tokens_match(tokens, name))
    }
}

fn glob_tokens_match(tokens: &[GlobToken], name: &str) -> bool {
    let mut current = vec![false; tokens.len() + 1];
    add_glob_closure(tokens, &mut current, 0);

    for character in name.chars() {
        let mut next = vec![false; tokens.len() + 1];
        for (index, active) in current.iter().enumerate().take(tokens.len()) {
            if !*active {
                continue;
            }
            match tokens[index] {
                GlobToken::Literal(expected) if expected == character => {
                    add_glob_closure(tokens, &mut next, index + 1);
                }
                GlobToken::One if character != '/' => {
                    add_glob_closure(tokens, &mut next, index + 1);
                }
                GlobToken::Star if character != '/' => {
                    add_glob_closure(tokens, &mut next, index);
                }
                GlobToken::GlobStar => {
                    add_glob_closure(tokens, &mut next, index);
                }
                GlobToken::GlobStarSlash => {
                    add_glob_closure(tokens, &mut next, index);
                    if character == '/' {
                        add_glob_closure(tokens, &mut next, index + 1);
                    }
                }
                _ => {}
            }
        }
        current = next;
    }
    current[tokens.len()]
}

fn add_glob_closure(tokens: &[GlobToken], states: &mut [bool], start: usize) {
    let mut pending = vec![start];
    while let Some(index) = pending.pop() {
        if states[index] {
            continue;
        }
        states[index] = true;
        if let Some(GlobToken::Star | GlobToken::GlobStar | GlobToken::GlobStarSlash) =
            tokens.get(index)
        {
            pending.push(index + 1);
        }
    }
}

fn normalize_glob_separators(value: &str) -> String {
    #[cfg(windows)]
    {
        value.replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        value.to_owned()
    }
}

// A small Thompson-NFA regex implementation avoids a new dependency while
// supporting the practical grep subset: literals, '.', grouping, alternation,
// anchors, word boundaries, character classes, escapes, and normal (or lazy)
// quantifiers. It is bounded by state and work limits so model-provided
// patterns cannot backtrack exponentially or consume unbounded CPU.

#[derive(Clone, Debug)]
struct SearchRegex {
    states: Vec<RegexState>,
    start: usize,
    ignore_case: bool,
}

#[derive(Clone, Debug)]
enum RegexState {
    Consume { matcher: CharMatcher, next: usize },
    Split { left: usize, right: usize },
    Jump { next: usize },
    Start { next: usize },
    End { next: usize },
    WordBoundary { negated: bool, next: usize },
    Accept,
}

#[derive(Clone, Debug)]
enum CharMatcher {
    Literal(char),
    Any,
    Class(CharClass),
}

impl CharMatcher {
    fn matches(&self, character: char, ignore_case: bool) -> bool {
        match self {
            Self::Literal(expected) => chars_equal(*expected, character, ignore_case),
            Self::Any => true,
            Self::Class(class) => class.matches(character, ignore_case),
        }
    }
}

#[derive(Clone, Debug)]
struct CharClass {
    negated: bool,
    items: Vec<ClassItem>,
}

#[derive(Clone, Debug)]
enum ClassItem {
    Char(char),
    Range(char, char),
    Kind(CharacterKind, bool),
}

#[derive(Clone, Copy, Debug)]
enum CharacterKind {
    Digit,
    Whitespace,
    Word,
}

impl CharClass {
    fn matches(&self, character: char, ignore_case: bool) -> bool {
        let matched = self.items.iter().any(|item| match item {
            ClassItem::Char(expected) => chars_equal(*expected, character, ignore_case),
            ClassItem::Range(start, end) => char_in_range(character, *start, *end, ignore_case),
            ClassItem::Kind(kind, inverted) => {
                let matched = match kind {
                    CharacterKind::Digit => character.is_ascii_digit(),
                    // Keep the Perl-style escapes aligned with Go's RE2
                    // subset rather than treating every Unicode separator or
                    // letter as `\s`/`\w`.
                    CharacterKind::Whitespace => {
                        matches!(character, '\t' | '\n' | '\u{000C}' | '\r' | ' ')
                    }
                    CharacterKind::Word => character.is_ascii_alphanumeric() || character == '_',
                };
                if *inverted { !matched } else { matched }
            }
        });
        if self.negated { !matched } else { matched }
    }
}

fn chars_equal(left: char, right: char, ignore_case: bool) -> bool {
    left == right || (ignore_case && left.to_lowercase().eq(right.to_lowercase()))
}

fn char_in_range(character: char, start: char, end: char, ignore_case: bool) -> bool {
    if ignore_case && character.is_ascii() && start.is_ascii() && end.is_ascii() {
        let character = character.to_ascii_lowercase();
        let start = start.to_ascii_lowercase();
        let end = end.to_ascii_lowercase();
        return start <= character && character <= end;
    }
    start <= character && character <= end
}

#[derive(Clone, Debug)]
enum RegexExpr {
    Empty,
    Consume(CharMatcher),
    Concat(Vec<RegexExpr>),
    Alternation(Vec<RegexExpr>),
    Repeat(Box<RegexExpr>, Repetition),
    Start,
    End,
    WordBoundary { negated: bool },
}

fn is_word_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || character == '_'
}

/// `\b` in RE2's sense: a `\w` on exactly one side of the position.
fn at_word_boundary(characters: &[char], position: usize) -> bool {
    let before = position
        .checked_sub(1)
        .and_then(|index| characters.get(index))
        .is_some_and(|character| is_word_char(*character));
    let after = characters
        .get(position)
        .is_some_and(|character| is_word_char(*character));
    before != after
}

#[derive(Clone, Copy, Debug)]
enum Repetition {
    ZeroOrMore,
    OneOrMore,
    ZeroOrOne,
    Counted {
        minimum: usize,
        maximum: Option<usize>,
    },
}

impl SearchRegex {
    fn literal(pattern: &str, ignore_case: bool) -> Result<Self> {
        if pattern.chars().count() > MAX_REGEX_PATTERN_CHARS {
            return Err(ToolError::new(format!(
                "pattern exceeds {MAX_REGEX_PATTERN_CHARS} characters"
            )));
        }
        let expression = RegexExpr::Concat(
            pattern
                .chars()
                .map(|character| RegexExpr::Consume(CharMatcher::Literal(character)))
                .collect(),
        );
        Self::from_expression(expression, ignore_case)
    }

    fn compile(pattern: &str, ignore_case: bool) -> Result<Self> {
        let mut source = pattern;
        let mut ignore_case = ignore_case;
        while let Some(rest) = source.strip_prefix("(?i)") {
            ignore_case = true;
            source = rest;
        }
        while let Some(rest) = source.strip_prefix("(?-i)") {
            ignore_case = false;
            source = rest;
        }
        if source.chars().count() > MAX_REGEX_PATTERN_CHARS {
            return Err(ToolError::new(format!(
                "pattern exceeds {MAX_REGEX_PATTERN_CHARS} characters"
            )));
        }
        let expression = RegexParser::new(source).parse()?;
        Self::from_expression(expression, ignore_case)
    }

    fn from_expression(expression: RegexExpr, ignore_case: bool) -> Result<Self> {
        let (states, start) = RegexCompiler::default().compile(&expression)?;
        Ok(Self {
            states,
            start,
            ignore_case,
        })
    }

    fn is_match(&self, text: &str, cancellation: &CancellationToken) -> Result<bool> {
        let characters = text.chars().collect::<Vec<_>>();
        let mut marks = vec![0usize; self.states.len()];
        let mut generation = next_generation(&mut marks, 0);
        let mut current = Vec::new();
        let mut work = 0usize;

        for position in 0..=characters.len() {
            if position % 256 == 0 {
                check_cancelled(cancellation)?;
            }
            self.add_closure(
                self.start,
                position,
                &characters,
                generation,
                &mut marks,
                &mut current,
                &mut work,
                cancellation,
            )?;
            if current
                .iter()
                .any(|state| matches!(self.states[*state], RegexState::Accept))
            {
                return Ok(true);
            }
            if position == characters.len() {
                break;
            }

            let mut next = Vec::new();
            generation = next_generation(&mut marks, generation);
            for state in &current {
                work = work.saturating_add(1);
                if work > MAX_REGEX_STEPS {
                    return Err(ToolError::new(format!(
                        "regex exceeded the {MAX_REGEX_STEPS}-step safety limit"
                    )));
                }
                if work.is_multiple_of(1_024) {
                    check_cancelled(cancellation)?;
                }
                if let RegexState::Consume {
                    matcher,
                    next: target,
                } = &self.states[*state]
                    && matcher.matches(characters[position], self.ignore_case)
                {
                    self.add_closure(
                        *target,
                        position + 1,
                        &characters,
                        generation,
                        &mut marks,
                        &mut next,
                        &mut work,
                        cancellation,
                    )?;
                }
            }
            current = next;
        }
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_closure(
        &self,
        start: usize,
        position: usize,
        characters: &[char],
        generation: usize,
        marks: &mut [usize],
        destination: &mut Vec<usize>,
        work: &mut usize,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        let mut pending = vec![start];
        while let Some(state) = pending.pop() {
            *work = work.saturating_add(1);
            if *work > MAX_REGEX_STEPS {
                return Err(ToolError::new(format!(
                    "regex exceeded the {MAX_REGEX_STEPS}-step safety limit"
                )));
            }
            if (*work).is_multiple_of(1_024) {
                check_cancelled(cancellation)?;
            }
            if marks[state] == generation {
                continue;
            }
            marks[state] = generation;
            match self.states[state] {
                RegexState::Split { left, right } => {
                    pending.push(left);
                    pending.push(right);
                }
                RegexState::Jump { next } => pending.push(next),
                RegexState::Start { next } if position == 0 => pending.push(next),
                RegexState::End { next } if position == characters.len() => pending.push(next),
                RegexState::WordBoundary { negated, next }
                    if at_word_boundary(characters, position) != negated =>
                {
                    pending.push(next);
                }
                RegexState::Start { .. }
                | RegexState::End { .. }
                | RegexState::WordBoundary { .. } => {}
                RegexState::Consume { .. } | RegexState::Accept => destination.push(state),
            }
        }
        Ok(())
    }
}

fn next_generation(marks: &mut [usize], current: usize) -> usize {
    let next = current.wrapping_add(1);
    if next == 0 {
        marks.fill(0);
        1
    } else {
        next
    }
}

struct RegexParser {
    characters: Vec<char>,
    cursor: usize,
    group_depth: usize,
}

impl RegexParser {
    fn new(pattern: &str) -> Self {
        Self {
            characters: pattern.chars().collect(),
            cursor: 0,
            group_depth: 0,
        }
    }

    fn parse(mut self) -> Result<RegexExpr> {
        let expression = self.parse_alternation()?;
        if let Some(character) = self.peek() {
            return Err(ToolError::new(format!(
                "unexpected regex character {character:?}"
            )));
        }
        Ok(expression)
    }

    fn parse_alternation(&mut self) -> Result<RegexExpr> {
        let mut alternatives = vec![self.parse_concat()?];
        while self.consume_if('|') {
            alternatives.push(self.parse_concat()?);
        }
        if alternatives.len() == 1 {
            Ok(alternatives.pop().expect("one alternative"))
        } else {
            Ok(RegexExpr::Alternation(alternatives))
        }
    }

    fn parse_concat(&mut self) -> Result<RegexExpr> {
        let mut expressions = Vec::new();
        while !matches!(self.peek(), None | Some(')') | Some('|')) {
            expressions.push(self.parse_repetition()?);
        }
        match expressions.len() {
            0 => Ok(RegexExpr::Empty),
            1 => Ok(expressions.pop().expect("one expression")),
            _ => Ok(RegexExpr::Concat(expressions)),
        }
    }

    fn parse_repetition(&mut self) -> Result<RegexExpr> {
        let mut expression = self.parse_atom()?;
        let mut quantified = false;
        loop {
            let repetition = match self.peek() {
                Some('*') => {
                    self.cursor += 1;
                    Some(Repetition::ZeroOrMore)
                }
                Some('+') => {
                    self.cursor += 1;
                    Some(Repetition::OneOrMore)
                }
                Some('?') => {
                    self.cursor += 1;
                    Some(Repetition::ZeroOrOne)
                }
                Some('{') => self.parse_counted_repetition()?,
                _ => None,
            };
            let Some(repetition) = repetition else {
                break;
            };
            if quantified {
                return Err(ToolError::new("repeated regex quantifier"));
            }
            quantified = true;
            // A lazy suffix (`*?`, `+?`, `??`, `{n,m}?`) changes which match
            // is preferred, not whether a line matches, so accept and ignore
            // it rather than rejecting a pattern any other grep takes.
            self.consume_if('?');
            expression = RegexExpr::Repeat(Box::new(expression), repetition);
        }
        Ok(expression)
    }

    fn parse_atom(&mut self) -> Result<RegexExpr> {
        let character = self
            .next()
            .ok_or_else(|| ToolError::new("expected a regex atom"))?;
        match character {
            '(' => {
                self.group_depth += 1;
                if self.group_depth > 64 {
                    return Err(ToolError::new("regex nesting exceeds 64 groups"));
                }
                if self.consume_if('?') && !self.consume_if(':') {
                    return Err(ToolError::new("unsupported regex group syntax"));
                }
                let expression = self.parse_alternation()?;
                if !self.consume_if(')') {
                    return Err(ToolError::new("unclosed regex group"));
                }
                self.group_depth -= 1;
                Ok(expression)
            }
            '[' => self
                .parse_class()
                .map(|class| RegexExpr::Consume(CharMatcher::Class(class))),
            '\\' => self.parse_escape(),
            '.' => Ok(RegexExpr::Consume(CharMatcher::Any)),
            '^' => Ok(RegexExpr::Start),
            '$' => Ok(RegexExpr::End),
            '*' | '+' | '?' | ')' | '|' => Err(ToolError::new(format!(
                "regex quantifier or delimiter {character:?} has no target"
            ))),
            literal => Ok(RegexExpr::Consume(CharMatcher::Literal(literal))),
        }
    }

    fn parse_escape(&mut self) -> Result<RegexExpr> {
        let character = self
            .next()
            .ok_or_else(|| ToolError::new("trailing regex escape"))?;
        match character {
            'b' => Ok(RegexExpr::WordBoundary { negated: false }),
            'B' => Ok(RegexExpr::WordBoundary { negated: true }),
            'A' => Ok(RegexExpr::Start),
            'z' => Ok(RegexExpr::End),
            escaped => parse_class_escape(escaped).map(|item| {
                RegexExpr::Consume(match item {
                    ClassItem::Kind(kind, inverted) => {
                        CharMatcher::Class(kind_class(kind, inverted))
                    }
                    ClassItem::Char(literal) | ClassItem::Range(literal, _) => {
                        CharMatcher::Literal(literal)
                    }
                })
            }),
        }
    }

    fn parse_class(&mut self) -> Result<CharClass> {
        let negated = self.consume_if('^');
        let mut items = Vec::new();
        let mut closed = false;
        while let Some(character) = self.peek() {
            if character == ']' && !items.is_empty() {
                self.cursor += 1;
                closed = true;
                break;
            }
            let left = self.parse_class_item()?;
            if let ClassItem::Char(start) = left
                && self.peek() == Some('-')
                && self.characters.get(self.cursor + 1) != Some(&']')
            {
                self.cursor += 1;
                let right = self.parse_class_item()?;
                let ClassItem::Char(end) = right else {
                    return Err(ToolError::new("regex range endpoint must be a character"));
                };
                if end < start {
                    return Err(ToolError::new("regex range is descending"));
                }
                items.push(ClassItem::Range(start, end));
            } else {
                items.push(left);
            }
        }
        if !closed {
            return Err(ToolError::new("unclosed regex character class"));
        }
        Ok(CharClass { negated, items })
    }

    fn parse_class_item(&mut self) -> Result<ClassItem> {
        let character = self
            .next()
            .ok_or_else(|| ToolError::new("unclosed regex character class"))?;
        if character != '\\' {
            return Ok(ClassItem::Char(character));
        }
        let escaped = self
            .next()
            .ok_or_else(|| ToolError::new("trailing regex escape in character class"))?;
        parse_class_escape(escaped)
    }

    fn parse_counted_repetition(&mut self) -> Result<Option<Repetition>> {
        let original = self.cursor;
        self.cursor += 1; // '{'
        if !self
            .peek()
            .is_some_and(|character| character.is_ascii_digit())
        {
            self.cursor = original;
            return Ok(None);
        }
        let minimum = self.parse_decimal()?;
        if minimum > MAX_REGEX_REPEAT {
            return Err(ToolError::new(format!(
                "regex repetition exceeds {MAX_REGEX_REPEAT}"
            )));
        }
        let maximum = if self.consume_if('}') {
            Some(minimum)
        } else if self.consume_if(',') {
            if self.consume_if('}') {
                None
            } else {
                let maximum = self.parse_decimal()?;
                if maximum > MAX_REGEX_REPEAT {
                    return Err(ToolError::new(format!(
                        "regex repetition exceeds {MAX_REGEX_REPEAT}"
                    )));
                }
                if !self.consume_if('}') {
                    return Err(ToolError::new("unclosed regex repetition"));
                }
                if maximum < minimum {
                    return Err(ToolError::new("regex repetition maximum is below minimum"));
                }
                Some(maximum)
            }
        } else {
            return Err(ToolError::new("invalid regex repetition"));
        };
        Ok(Some(Repetition::Counted { minimum, maximum }))
    }

    fn parse_decimal(&mut self) -> Result<usize> {
        let mut value = 0usize;
        let mut consumed = false;
        while let Some(character) = self.peek() {
            if !character.is_ascii_digit() {
                break;
            }
            consumed = true;
            self.cursor += 1;
            value = value
                .checked_mul(10)
                .and_then(|value| value.checked_add((character as u8 - b'0') as usize))
                .ok_or_else(|| ToolError::new("regex repetition is too large"))?;
        }
        if consumed {
            Ok(value)
        } else {
            Err(ToolError::new("expected a regex repetition count"))
        }
    }

    fn peek(&self) -> Option<char> {
        self.characters.get(self.cursor).copied()
    }

    fn next(&mut self) -> Option<char> {
        let character = self.peek()?;
        self.cursor += 1;
        Some(character)
    }

    fn consume_if(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
}

fn kind_class(kind: CharacterKind, inverted: bool) -> CharClass {
    CharClass {
        negated: false,
        items: vec![ClassItem::Kind(kind, inverted)],
    }
}

/// Escapes that stand for a character (class). Any other letter or digit is
/// rejected: silently matching `\p{L}` or `\x41` as a literal `p` or `x`
/// would report "no matches" for a pattern the model believed was valid.
fn parse_class_escape(escaped: char) -> Result<ClassItem> {
    match escaped {
        'd' => Ok(ClassItem::Kind(CharacterKind::Digit, false)),
        'D' => Ok(ClassItem::Kind(CharacterKind::Digit, true)),
        's' => Ok(ClassItem::Kind(CharacterKind::Whitespace, false)),
        'S' => Ok(ClassItem::Kind(CharacterKind::Whitespace, true)),
        'w' => Ok(ClassItem::Kind(CharacterKind::Word, false)),
        'W' => Ok(ClassItem::Kind(CharacterKind::Word, true)),
        'n' => Ok(ClassItem::Char('\n')),
        'r' => Ok(ClassItem::Char('\r')),
        't' => Ok(ClassItem::Char('\t')),
        'f' => Ok(ClassItem::Char('\u{000C}')),
        'v' => Ok(ClassItem::Char('\u{000B}')),
        'a' => Ok(ClassItem::Char('\u{0007}')),
        'e' => Ok(ClassItem::Char('\u{001B}')),
        unsupported if unsupported.is_ascii_alphanumeric() => Err(ToolError::new(format!(
            "unsupported regex escape \\{unsupported}"
        ))),
        literal => Ok(ClassItem::Char(literal)),
    }
}

#[derive(Default)]
struct RegexCompiler {
    states: Vec<BuildState>,
}

#[derive(Clone, Debug)]
enum BuildState {
    Consume {
        matcher: CharMatcher,
        next: Option<usize>,
    },
    Split {
        left: Option<usize>,
        right: Option<usize>,
    },
    Jump {
        next: Option<usize>,
    },
    Start {
        next: Option<usize>,
    },
    End {
        next: Option<usize>,
    },
    WordBoundary {
        negated: bool,
        next: Option<usize>,
    },
    Accept,
}

#[derive(Clone, Copy)]
enum PatchSlot {
    Next,
    Right,
}

#[derive(Clone, Copy)]
struct Patch {
    state: usize,
    slot: PatchSlot,
}

struct Fragment {
    start: usize,
    outputs: Vec<Patch>,
}

impl RegexCompiler {
    fn compile(mut self, expression: &RegexExpr) -> Result<(Vec<RegexState>, usize)> {
        let fragment = self.compile_expression(expression)?;
        let accept = self.add_state(BuildState::Accept)?;
        self.patch(fragment.outputs, accept);

        let states = self
            .states
            .into_iter()
            .map(|state| match state {
                BuildState::Consume { matcher, next } => next
                    .map(|next| RegexState::Consume { matcher, next })
                    .ok_or_else(|| ToolError::new("unpatched regex consume state")),
                BuildState::Split { left, right } => match (left, right) {
                    (Some(left), Some(right)) => Ok(RegexState::Split { left, right }),
                    _ => Err(ToolError::new("unpatched regex split state")),
                },
                BuildState::Jump { next } => next
                    .map(|next| RegexState::Jump { next })
                    .ok_or_else(|| ToolError::new("unpatched regex jump state")),
                BuildState::Start { next } => next
                    .map(|next| RegexState::Start { next })
                    .ok_or_else(|| ToolError::new("unpatched regex start anchor")),
                BuildState::End { next } => next
                    .map(|next| RegexState::End { next })
                    .ok_or_else(|| ToolError::new("unpatched regex end anchor")),
                BuildState::WordBoundary { negated, next } => next
                    .map(|next| RegexState::WordBoundary { negated, next })
                    .ok_or_else(|| ToolError::new("unpatched regex word boundary")),
                BuildState::Accept => Ok(RegexState::Accept),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((states, fragment.start))
    }

    fn compile_expression(&mut self, expression: &RegexExpr) -> Result<Fragment> {
        match expression {
            RegexExpr::Empty => self.empty_fragment(),
            RegexExpr::Consume(matcher) => {
                let state = self.add_state(BuildState::Consume {
                    matcher: matcher.clone(),
                    next: None,
                })?;
                Ok(Fragment {
                    start: state,
                    outputs: vec![Patch {
                        state,
                        slot: PatchSlot::Next,
                    }],
                })
            }
            RegexExpr::Start => self.single_out_fragment(BuildState::Start { next: None }),
            RegexExpr::End => self.single_out_fragment(BuildState::End { next: None }),
            RegexExpr::WordBoundary { negated } => {
                self.single_out_fragment(BuildState::WordBoundary {
                    negated: *negated,
                    next: None,
                })
            }
            RegexExpr::Concat(expressions) => {
                let mut fragment = self.empty_fragment()?;
                for expression in expressions {
                    let next = self.compile_expression(expression)?;
                    fragment = self.concat(fragment, next);
                }
                Ok(fragment)
            }
            RegexExpr::Alternation(expressions) => {
                let mut expressions = expressions.iter();
                let mut fragment = match expressions.next() {
                    Some(expression) => self.compile_expression(expression)?,
                    None => return self.empty_fragment(),
                };
                for expression in expressions {
                    let other = self.compile_expression(expression)?;
                    let split = self.add_state(BuildState::Split {
                        left: Some(fragment.start),
                        right: Some(other.start),
                    })?;
                    let mut outputs = fragment.outputs;
                    outputs.extend(other.outputs);
                    fragment = Fragment {
                        start: split,
                        outputs,
                    };
                }
                Ok(fragment)
            }
            RegexExpr::Repeat(expression, repetition) => {
                self.compile_repetition(expression, *repetition)
            }
        }
    }

    fn compile_repetition(
        &mut self,
        expression: &RegexExpr,
        repetition: Repetition,
    ) -> Result<Fragment> {
        match repetition {
            Repetition::ZeroOrMore => {
                let inner = self.compile_expression(expression)?;
                let split = self.add_state(BuildState::Split {
                    left: Some(inner.start),
                    right: None,
                })?;
                self.patch(inner.outputs, split);
                Ok(Fragment {
                    start: split,
                    outputs: vec![Patch {
                        state: split,
                        slot: PatchSlot::Right,
                    }],
                })
            }
            Repetition::OneOrMore => {
                let inner = self.compile_expression(expression)?;
                let split = self.add_state(BuildState::Split {
                    left: Some(inner.start),
                    right: None,
                })?;
                self.patch(inner.outputs, split);
                Ok(Fragment {
                    start: inner.start,
                    outputs: vec![Patch {
                        state: split,
                        slot: PatchSlot::Right,
                    }],
                })
            }
            Repetition::ZeroOrOne => {
                let inner = self.compile_expression(expression)?;
                let split = self.add_state(BuildState::Split {
                    left: Some(inner.start),
                    right: None,
                })?;
                let mut outputs = inner.outputs;
                outputs.push(Patch {
                    state: split,
                    slot: PatchSlot::Right,
                });
                Ok(Fragment {
                    start: split,
                    outputs,
                })
            }
            Repetition::Counted { minimum, maximum } => {
                let mut result = self.empty_fragment()?;
                for _ in 0..minimum {
                    let next = self.compile_expression(expression)?;
                    result = self.concat(result, next);
                }
                match maximum {
                    Some(maximum) => {
                        for _ in minimum..maximum {
                            let optional =
                                self.compile_repetition(expression, Repetition::ZeroOrOne)?;
                            result = self.concat(result, optional);
                        }
                    }
                    None => {
                        let tail = self.compile_repetition(expression, Repetition::ZeroOrMore)?;
                        result = self.concat(result, tail);
                    }
                }
                Ok(result)
            }
        }
    }

    fn empty_fragment(&mut self) -> Result<Fragment> {
        self.single_out_fragment(BuildState::Jump { next: None })
    }

    fn single_out_fragment(&mut self, state: BuildState) -> Result<Fragment> {
        let state = self.add_state(state)?;
        Ok(Fragment {
            start: state,
            outputs: vec![Patch {
                state,
                slot: PatchSlot::Next,
            }],
        })
    }

    fn concat(&mut self, left: Fragment, right: Fragment) -> Fragment {
        self.patch(left.outputs, right.start);
        Fragment {
            start: left.start,
            outputs: right.outputs,
        }
    }

    fn add_state(&mut self, state: BuildState) -> Result<usize> {
        if self.states.len() >= MAX_REGEX_STATES {
            return Err(ToolError::new(format!(
                "regex exceeds the {MAX_REGEX_STATES}-state safety limit"
            )));
        }
        let index = self.states.len();
        self.states.push(state);
        Ok(index)
    }

    fn patch(&mut self, patches: Vec<Patch>, destination: usize) {
        for patch in patches {
            match (&mut self.states[patch.state], patch.slot) {
                (BuildState::Consume { next, .. }, PatchSlot::Next)
                | (BuildState::Jump { next }, PatchSlot::Next)
                | (BuildState::Start { next }, PatchSlot::Next)
                | (BuildState::End { next }, PatchSlot::Next)
                | (BuildState::WordBoundary { next, .. }, PatchSlot::Next) => {
                    *next = Some(destination);
                }
                (BuildState::Split { right, .. }, PatchSlot::Right) => *right = Some(destination),
                _ => unreachable!("regex patch slot does not match state"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::BTreeMap, fs, path::PathBuf, sync::Arc, thread, time::Duration};

    struct TempDirectory {
        path: PathBuf,
    }

    impl TempDirectory {
        fn new() -> Self {
            let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "goshcoder-tools-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).expect("create temporary workspace");
            Self { path }
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn workspace() -> (TempDirectory, Workspace) {
        let directory = TempDirectory::new();
        let workspace = Workspace::new(&directory.path).expect("workspace");
        (directory, workspace)
    }

    fn parameters(
        values: impl IntoIterator<Item = (&'static str, Value)>,
    ) -> BTreeMap<String, Value> {
        values
            .into_iter()
            .map(|(name, value)| (name.to_owned(), value))
            .collect()
    }

    fn run(
        tool: agent::Tool,
        parameters: BTreeMap<String, Value>,
    ) -> std::result::Result<String, String> {
        let result = (tool.execute)(
            CancellationToken::default(),
            "test-call".to_owned(),
            parameters,
            Arc::new(|_| {}),
        )?;
        Ok(result
            .content
            .iter()
            .filter_map(|block| block.plain_text())
            .collect::<Vec<_>>()
            .join("\n"))
    }

    #[test]
    fn workspace_rejects_missing_paths_and_files() {
        let directory = TempDirectory::new();
        let file = directory.path.join("file");
        fs::write(&file, "x").expect("write file");
        assert!(Workspace::new(&file).is_err());
        assert!(Workspace::new(directory.path.join("missing")).is_err());
    }

    #[test]
    fn active_tool_set_has_the_seven_pi_tool_names_and_schemas() {
        let (_directory, workspace) = workspace();
        let planning = workspace.planning();
        let all = workspace.all();
        assert_eq!(planning.len(), 6);
        assert_eq!(all.len(), 7);
        assert_eq!(
            all.iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            ["read", "write", "edit", "ls", "grep", "find", "bash"]
        );
        assert!(planning.iter().all(|tool| tool.name != "bash"));
        assert!(all.iter().all(|tool| {
            !tool.description.is_empty()
                && tool.parameters.get("type") == Some(&Value::String("object".to_owned()))
        }));
        assert_eq!(workspace.list_tool().name, "list");
    }

    #[test]
    fn write_read_and_offset_round_trip() {
        let (_directory, workspace) = workspace();
        let write = run(
            workspace.write_tool(),
            parameters([
                ("path", json!("notes/todo.txt")),
                ("content", json!("one\ntwo\nthree\nfour\n")),
            ]),
        )
        .expect("write");
        assert!(write.contains("Wrote 19 bytes to notes/todo.txt"));

        let read = run(
            workspace.read_tool(),
            parameters([
                ("path", json!("notes/todo.txt")),
                ("offset", json!(2)),
                ("limit", json!(2)),
            ]),
        )
        .expect("read");
        assert!(read.starts_with("two\nthree"));
        assert!(read.contains("offset=4"));
    }

    #[test]
    fn paths_cannot_escape_or_follow_an_ancestor_symlink() {
        let (directory, workspace) = workspace();
        let outside = directory
            .path
            .parent()
            .expect("temporary parent")
            .join(format!("outside-{}", std::process::id()));
        fs::write(&outside, "secret").expect("write outside");
        for path in [
            "../outside",
            "../../outside",
            outside.to_str().expect("UTF-8 temporary path"),
        ] {
            assert!(
                run(
                    workspace.write_tool(),
                    parameters([("path", json!(path)), ("content", json!("pwned"))]),
                )
                .is_err(),
                "{path} escaped the workspace"
            );
        }
        assert_eq!(
            fs::read_to_string(&outside).expect("read outside"),
            "secret"
        );
        let _ = fs::remove_file(&outside);

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let target = TempDirectory::new();
            symlink(&target.path, directory.path.join("link")).expect("create symlink");
            fs::write(target.path.join("secret.txt"), "outside").expect("write symlink target");
            assert!(
                run(
                    workspace.write_tool(),
                    parameters([
                        ("path", json!("link/escaped.txt")),
                        ("content", json!("secret")),
                    ]),
                )
                .is_err()
            );
            assert!(!target.path.join("escaped.txt").exists());
            assert!(
                run(
                    workspace.read_tool(),
                    parameters([("path", json!("link/secret.txt"))]),
                )
                .is_err()
            );
        }
    }

    #[test]
    fn edit_requires_one_exact_match_and_keeps_failed_edits_unchanged() {
        let (directory, workspace) = workspace();
        let path = directory.path.join("code.txt");
        fs::write(&path, "repeat\nrepeat\n").expect("write fixture");

        let ambiguous = run(
            workspace.edit_tool(),
            parameters([
                ("path", json!("code.txt")),
                ("old_text", json!("repeat")),
                ("new_text", json!("changed")),
            ]),
        )
        .expect_err("ambiguous edit must fail");
        assert!(ambiguous.contains("appears 2 times"));
        assert_eq!(
            fs::read_to_string(&path).expect("read fixture"),
            "repeat\nrepeat\n"
        );

        run(
            workspace.edit_tool(),
            parameters([
                ("path", json!("code.txt")),
                ("old_text", json!("repeat\nrepeat")),
                ("new_text", json!("changed")),
            ]),
        )
        .expect("unique edit");
        assert_eq!(
            fs::read_to_string(&path).expect("read fixture"),
            "changed\n"
        );
    }

    #[test]
    fn atomic_writes_preserve_existing_permissions_and_leave_no_temp_file() {
        let (directory, workspace) = workspace();
        let path = directory.path.join("file.txt");
        fs::write(&path, "old").expect("write fixture");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).expect("set mode");
        }

        run(
            workspace.write_tool(),
            parameters([
                ("path", json!("file.txt")),
                ("content", json!("replacement")),
            ]),
        )
        .expect("atomic write");
        assert_eq!(
            fs::read_to_string(&path).expect("read replacement"),
            "replacement"
        );
        assert!(
            fs::read_dir(&directory.path)
                .expect("read workspace")
                .all(|entry| {
                    let name = entry
                        .expect("entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned();
                    !(name.contains(".goshcoder-") && name.ends_with(".tmp"))
                })
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn reads_and_output_clipping_do_not_split_unicode_scalars() {
        assert_eq!(clip_utf8("你", 1), "");
        assert_eq!(clip_utf8("你你", 4), "你");
        assert_eq!(
            decode_text(&"你".as_bytes()[..2], true, "partial").expect("partial UTF-8 prefix"),
            ""
        );
        let mut output = CappedBytes::new(4);
        output.push("你你".as_bytes());
        assert_eq!(output.render_text(), "你\n[output truncated]");

        let (directory, workspace) = workspace();
        let content = format!("{}{}", "你".repeat(MAX_READ_BYTES / 3 + 2), "tail");
        fs::write(directory.path.join("unicode.txt"), content).expect("write unicode fixture");
        let read = run(
            workspace.read_tool(),
            parameters([("path", json!("unicode.txt"))]),
        )
        .expect("read unicode");
        assert!(read.contains("[truncated:"));
        assert!(std::str::from_utf8(read.as_bytes()).is_ok());
    }

    #[test]
    fn read_scan_cap_never_claims_a_truncated_prefix_is_the_whole_file() {
        let (directory, workspace) = workspace();
        let line = format!("{}\n", "x".repeat(64));
        let mut content = String::new();
        while content.len() <= MAX_READ_SCAN_BYTES + line.len() {
            content.push_str(&line);
        }
        fs::write(directory.path.join("large-lines.txt"), content).expect("write large fixture");

        let output = run(
            workspace.read_tool(),
            parameters([("path", json!("large-lines.txt")), ("limit", json!(5))]),
        )
        .expect("read capped file");
        assert!(output.contains("at least"));

        let error = run(
            workspace.read_tool(),
            parameters([
                ("path", json!("large-lines.txt")),
                ("offset", json!(9_999_999)),
            ]),
        )
        .expect_err("offset past cap");
        assert!(error.contains("exceeds"));
        assert!(!error.contains("beyond end of file"));
    }

    #[test]
    fn glob_matching_handles_unicode_double_stars_and_basename_fallback() {
        let cases = [
            ("café*.rs", "café-utils.rs", true),
            ("日本*.txt", "日本語.txt", true),
            ("café*.rs", "cafe-utils.rs", false),
            ("*.rs", "src/main.rs", true),
            ("**/*.rs", "src/deep/main.rs", true),
            ("src/*.rs", "src/main.rs", true),
            ("src/*.rs", "other/main.rs", false),
            ("a?c.rs", "a你c.rs", true),
        ];
        for (pattern, name, expected) in cases {
            assert_eq!(
                compile_glob(pattern).expect("compile glob").is_match(name),
                expected,
                "{pattern:?} against {name:?}"
            );
        }
    }

    #[test]
    fn grep_find_and_list_use_safe_fallback_discovery() {
        let (directory, workspace) = workspace();
        fs::create_dir_all(directory.path.join("src")).expect("make source");
        fs::create_dir_all(directory.path.join("vendor/pkg")).expect("make vendor");
        fs::write(
            directory.path.join("src/main.rs"),
            "fn main() {\n // Needle\n}\n",
        )
        .expect("write source");
        fs::write(directory.path.join("src/café.rs"), "let café = 1;\n").expect("write unicode");
        fs::write(directory.path.join("vendor/pkg/lib.rs"), "needle").expect("write vendor");

        let found = run(
            workspace.find_tool(),
            parameters([("pattern", json!("**/*.rs"))]),
        )
        .expect("find");
        assert!(found.contains("src/main.rs"));
        assert!(found.contains("src/café.rs"));
        assert!(!found.contains("vendor/pkg/lib.rs"));

        let grep = run(
            workspace.grep_tool(),
            parameters([
                ("pattern", json!("needle")),
                ("ignoreCase", json!(true)),
                ("glob", json!("*.rs")),
            ]),
        )
        .expect("grep");
        assert!(grep.contains("src/main.rs:2:"));

        let listed = run(workspace.ls_tool(), parameters([("path", json!("src"))])).expect("list");
        assert!(listed.contains("main.rs"));
        assert!(listed.contains("café.rs"));
    }

    #[test]
    fn find_limit_follows_sorted_candidates() {
        let (directory, workspace) = workspace();
        fs::write(directory.path.join("b.rs"), "").expect("write b");
        fs::write(directory.path.join("a.rs"), "").expect("write a");
        let output = run(
            workspace.find_tool(),
            parameters([("pattern", json!("*.rs")), ("limit", json!(1))]),
        )
        .expect("limited find");
        assert!(output.starts_with("a.rs"));
        assert!(output.contains("results limit reached"));
    }

    #[test]
    fn grep_regex_supports_anchors_classes_alternation_and_literal_mode() {
        let (directory, workspace) = workspace();
        fs::write(
            directory.path.join("patterns.txt"),
            "alpha-42\nbeta\nliteral.*\n",
        )
        .expect("write patterns");

        let regex = run(
            workspace.grep_tool(),
            parameters([("pattern", json!("^(alpha|beta)-?\\d*$"))]),
        )
        .expect("regex grep");
        assert!(regex.contains("patterns.txt:1: alpha-42"));
        assert!(regex.contains("patterns.txt:2: beta"));

        let literal = run(
            workspace.grep_tool(),
            parameters([("pattern", json!("literal.*")), ("literal", json!(true))]),
        )
        .expect("literal grep");
        assert!(literal.contains("patterns.txt:3: literal.*"));
    }

    #[cfg(not(windows))]
    #[test]
    fn bash_runs_in_workspace_reports_failure_and_honors_timeout() {
        let (directory, mut workspace) = workspace();
        fs::write(directory.path.join("marker.txt"), "x").expect("write marker");
        let output = run(
            workspace.bash_tool(),
            parameters([("command", json!("printf marker.txt"))]),
        )
        .expect("bash");
        assert_eq!(output, "marker.txt");

        assert!(
            run(
                workspace.bash_tool(),
                parameters([("command", json!("exit 3"))]),
            )
            .is_err()
        );

        workspace.set_bash_timeout(Duration::from_millis(30));
        let timeout = run(
            workspace.bash_tool(),
            parameters([("command", json!("sleep 1"))]),
        )
        .expect_err("timeout");
        assert!(timeout.contains("timed out"));
    }

    #[cfg(not(windows))]
    #[test]
    fn bash_bounds_output_and_observes_mid_command_cancellation() {
        let (_directory, workspace) = workspace();
        let capped = run(
            workspace.bash_tool(),
            parameters([("command", json!("printf '%040000d' 0"))]),
        )
        .expect("capped output");
        assert!(capped.ends_with("[output truncated]"));
        assert!(capped.len() <= MAX_OUTPUT_BYTES + "[output truncated]\n".len());

        let cancellation = CancellationToken::default();
        let running_token = cancellation.clone();
        let tool = workspace.bash_tool();
        let task = thread::spawn(move || {
            (tool.execute)(
                running_token,
                "cancel-running-command".to_owned(),
                parameters([("command", json!("sleep 5"))]),
                Arc::new(|_| {}),
            )
        });
        thread::sleep(Duration::from_millis(30));
        cancellation.cancel();
        let error = task
            .join()
            .expect("bash tool thread must not panic")
            .expect_err("cancelled shell command must fail");
        assert!(error.contains("cancelled"));
    }

    #[test]
    fn cancelled_tools_stop_before_side_effects() {
        let (directory, workspace) = workspace();
        let token = CancellationToken::default();
        token.cancel();
        let result = (workspace.write_tool().execute)(
            token,
            "cancelled".to_owned(),
            parameters([
                ("path", json!("should-not-exist.txt")),
                ("content", json!("x")),
            ]),
            Arc::new(|_| {}),
        );
        assert!(result.is_err());
        assert!(!directory.path.join("should-not-exist.txt").exists());
    }

    #[test]
    fn git_candidate_paths_keep_unicode_names_when_git_is_available() {
        let (directory, workspace) = workspace();
        let initialized = Command::new("git")
            .args(["init", "-q"])
            .current_dir(&directory.path)
            .status();
        let Ok(status) = initialized else {
            return;
        };
        if !status.success() {
            return;
        }
        fs::write(directory.path.join(".gitignore"), "ignored.rs\n").expect("write ignore file");
        fs::write(directory.path.join("ignored.rs"), "NEEDLE\n").expect("write ignored path");
        fs::write(directory.path.join("café-日本.txt"), "NEEDLE\n").expect("write unicode path");
        let output = run(
            workspace.grep_tool(),
            parameters([("pattern", json!("NEEDLE"))]),
        )
        .expect("grep unicode git candidate");
        assert!(output.contains("café-日本.txt:1: NEEDLE"));
        assert!(!output.contains("ignored.rs"));
    }

    #[test]
    fn file_mutations_serialize_concurrent_writes_to_one_path() {
        let (directory, workspace) = workspace();
        let path = directory.path.join("shared.txt");
        fs::write(&path, "tail\n").expect("write fixture");
        let key = workspace.mutation_key(Path::new("shared.txt"));

        let held = FILE_MUTATIONS
            .acquire(key.clone(), &CancellationToken::default())
            .expect("hold the path");
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(FILE_MUTATIONS.acquire(key, &cancelled).is_err());

        let tool = workspace.write_tool();
        let writer = thread::spawn(move || {
            run(
                tool,
                parameters([
                    ("path", json!("shared.txt")),
                    ("content", json!("replaced")),
                ]),
            )
        });
        thread::sleep(Duration::from_millis(150));
        assert!(!writer.is_finished(), "write ran while the path was held");
        assert_eq!(fs::read_to_string(&path).expect("read held file"), "tail\n");
        drop(held);
        writer
            .join()
            .expect("writer thread")
            .expect("write after release");
        assert_eq!(
            fs::read_to_string(&path).expect("read released"),
            "replaced"
        );

        fs::write(&path, "tail\n").expect("reset fixture");
        let editors = (0..8)
            .map(|index| {
                let tool = workspace.edit_tool();
                thread::spawn(move || {
                    run(
                        tool,
                        parameters([
                            ("path", json!("shared.txt")),
                            ("old_text", json!("tail")),
                            ("new_text", json!(format!("line-{index}\ntail"))),
                        ]),
                    )
                })
            })
            .collect::<Vec<_>>();
        for editor in editors {
            editor.join().expect("editor thread").expect("racing edit");
        }
        let content = fs::read_to_string(&path).expect("read edited file");
        for index in 0..8 {
            assert!(content.contains(&format!("line-{index}\n")), "{content}");
        }
        assert_eq!(content.matches("tail").count(), 1);
    }

    #[cfg(not(windows))]
    #[test]
    fn bash_kills_the_whole_process_group_without_waiting_for_orphans() {
        let (directory, mut workspace) = workspace();
        workspace.set_bash_timeout(Duration::from_millis(50));
        let started = Instant::now();
        let error = run(
            workspace.bash_tool(),
            parameters([(
                "command",
                json!("(sleep 0.3; echo alive > marker.txt; sleep 5) & wait"),
            )]),
        )
        .expect_err("timeout");
        assert!(error.contains("timed out"));
        assert!(
            started.elapsed() < Duration::from_millis(1_500),
            "waited on a pipe held by a killed group: {:?}",
            started.elapsed()
        );
        thread::sleep(Duration::from_millis(600));
        assert!(
            !directory.path.join("marker.txt").exists(),
            "a backgrounded descendant survived the timeout"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn bash_does_not_inherit_the_agent_stdin() {
        let (_directory, mut workspace) = workspace();
        workspace.set_bash_timeout(Duration::from_secs(5));
        let output = run(
            workspace.bash_tool(),
            parameters([("command", json!("cat; echo eof"))]),
        )
        .expect("cat must see EOF immediately");
        assert_eq!(output.trim_end(), "eof");
    }

    #[test]
    fn edit_matches_lf_text_against_crlf_files_and_keeps_their_endings_and_bom() {
        let (directory, workspace) = workspace();
        let path = directory.path.join("dos.txt");
        fs::write(&path, "\u{FEFF}alpha\r\nbeta\r\ngamma\r\n").expect("write fixture");
        run(
            workspace.edit_tool(),
            parameters([
                ("path", json!("dos.txt")),
                ("old_text", json!("beta\ngamma")),
                ("new_text", json!("BETA\nGAMMA\nextra")),
            ]),
        )
        .expect("edit CRLF file");
        assert_eq!(
            fs::read_to_string(&path).expect("read edited"),
            "\u{FEFF}alpha\r\nBETA\r\nGAMMA\r\nextra\r\n"
        );

        fs::write(&path, "one\ntwo\n").expect("write LF fixture");
        run(
            workspace.edit_tool(),
            parameters([
                ("path", json!("dos.txt")),
                ("old_text", json!("one\r\ntwo")),
                ("new_text", json!("x")),
            ]),
        )
        .expect("CRLF old_text against LF file");
        assert_eq!(fs::read_to_string(&path).expect("read LF edit"), "x\n");
        assert_eq!(detect_line_ending("a\nb\r\n"), LineEnding::Lf);
        assert_eq!(detect_line_ending("a\r\nb\n"), LineEnding::CrLf);
    }

    #[test]
    fn regex_word_boundaries_lazy_quantifiers_and_unknown_escapes() {
        let token = CancellationToken::default();
        let matches = |pattern: &str, text: &str| {
            SearchRegex::compile(pattern, false)
                .unwrap_or_else(|error| panic!("{pattern}: {error}"))
                .is_match(text, &token)
                .expect("match")
        };
        assert!(matches(r"\bfoo\b", "a foo b"));
        assert!(matches(r"\bfoo\b", "foo"));
        assert!(!matches(r"\bfoo\b", "food"));
        assert!(!matches(r"\bfoo\b", "_foo"));
        assert!(matches(r"\Bfoo", "afoo"));
        assert!(!matches(r"\Bfoo", "foo bar"));
        assert!(matches(r"a+?b", "aab"));
        assert!(matches(r"colou??r", "color"));
        assert!(matches(r"x{1,2}?y", "xy"));
        assert!(matches(r"\Afoo\z", "foo"));
        assert!(matches(r"\.", "."));
        assert!(!matches(r"\.", "a"));
        assert!(matches(r"[\f\v]", "\u{000C}"));
        for unsupported in [r"\p{L}", r"\x41", r"[\p]", r"\Q", r"[\b]", r"\1"] {
            let error = SearchRegex::compile(unsupported, false).expect_err(unsupported);
            assert!(error.to_string().contains("unsupported regex escape"));
        }
        assert!(SearchRegex::compile("a**", false).is_err());
    }

    #[test]
    fn verbatim_prefixes_are_simplified_to_ordinary_paths() {
        let cases = [
            (r"\\?\C:\ws\file.txt", r"C:\ws\file.txt"),
            (r"\\?\UNC\server\share\dir", r"\\server\share\dir"),
            (r"\\?\Volume{guid}\x", r"\\?\Volume{guid}\x"),
            ("/plain/path", "/plain/path"),
        ];
        for (verbatim, expected) in cases {
            assert_eq!(
                simplify_verbatim(Path::new(verbatim)),
                PathBuf::from(expected),
                "{verbatim}"
            );
        }
    }

    #[test]
    fn legacy_wsl_bash_launcher_is_recognised() {
        assert!(is_legacy_wsl_bash(r"C:\Windows\System32\bash.exe"));
        assert!(is_legacy_wsl_bash("c:/windows/sysnative/bash.exe"));
        assert!(is_legacy_wsl_bash(r"D:\WINDOWS\System32\BASH.EXE"));
        assert!(!is_legacy_wsl_bash(r"C:\Program Files\Git\bin\bash.exe"));
        assert!(!is_legacy_wsl_bash(r"C:\Windows\System32\wsl\bash.exe"));
    }

    #[test]
    fn huge_directories_are_listed_partially_instead_of_failing() {
        let directory = TempDirectory::new();
        for index in 0..5 {
            fs::write(directory.path.join(format!("file-{index}")), "").expect("write entry");
        }
        let token = CancellationToken::default();
        let (entries, overflowed) =
            read_directory_entries(&directory.path, 3, &token).expect("capped listing");
        assert_eq!(entries.len(), 3);
        assert!(overflowed);
        let (entries, overflowed) =
            read_directory_entries(&directory.path, 10, &token).expect("full listing");
        assert_eq!(entries.len(), 5);
        assert!(!overflowed);
    }

    #[test]
    fn temporary_names_stay_short_for_long_destinations() {
        let (directory, workspace) = workspace();
        let long_name = "n".repeat(200);
        let temporary = temporary_path(&directory.path, &directory.path.join(&long_name), 0);
        let name = temporary.file_name().expect("name").to_string_lossy();
        assert!(name.len() < 128, "{name}");
        assert!(name.starts_with(&format!(".{}", "n".repeat(TEMP_NAME_PREFIX_BYTES))));

        run(
            workspace.write_tool(),
            parameters([("path", json!(long_name)), ("content", json!("ok"))]),
        )
        .expect("write to a long but valid file name");
        assert_eq!(
            fs::read_to_string(directory.path.join(&long_name)).expect("read long name"),
            "ok"
        );
    }

    #[cfg(unix)]
    #[test]
    fn atomic_writes_do_not_carry_setuid_bits_onto_the_replacement() {
        use std::os::unix::fs::PermissionsExt;

        let (directory, workspace) = workspace();
        let path = directory.path.join("tool");
        fs::write(&path, "old").expect("write fixture");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o4755)).expect("set setuid");
        run(
            workspace.write_tool(),
            parameters([("path", json!("tool")), ("content", json!("new"))]),
        )
        .expect("write");
        assert_eq!(
            fs::metadata(&path).expect("metadata").permissions().mode() & 0o7777,
            0o755
        );
    }

    #[test]
    fn git_candidates_skip_tracked_files_deleted_from_the_worktree() {
        let (directory, workspace) = workspace();
        let git = |arguments: &[&str]| {
            Command::new("git")
                .args(arguments)
                .current_dir(&directory.path)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|status| status.success())
        };
        fs::write(directory.path.join("kept.txt"), "").expect("write kept");
        fs::write(directory.path.join("gone.txt"), "").expect("write gone");
        if !(git(&["init", "-q"])
            && git(&["add", "."])
            && git(&[
                "-c",
                "user.name=test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-q",
                "-m",
                "init",
            ]))
        {
            return;
        }
        fs::remove_file(directory.path.join("gone.txt")).expect("delete tracked file");
        let output = run(
            workspace.find_tool(),
            parameters([("pattern", json!("*.txt"))]),
        )
        .expect("find");
        assert!(output.contains("kept.txt"));
        assert!(!output.contains("gone.txt"), "{output}");
    }

    #[cfg(unix)]
    #[test]
    fn walk_skips_unreadable_subdirectories_instead_of_aborting() {
        use std::os::unix::fs::PermissionsExt;

        struct RestoreMode(PathBuf);
        impl Drop for RestoreMode {
            fn drop(&mut self) {
                let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
            }
        }

        let (directory, workspace) = workspace();
        let locked = directory.path.join("locked");
        fs::create_dir(&locked).expect("make locked");
        fs::write(locked.join("hidden.txt"), "").expect("write hidden");
        fs::write(directory.path.join("open.txt"), "").expect("write open");
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).expect("lock directory");
        let _restore = RestoreMode(locked.clone());
        if fs::read_dir(&locked).is_ok() {
            // Root ignores directory modes, so the failure cannot be staged.
            return;
        }

        let candidates = workspace
            .walk_candidate_files(&CancellationToken::default(), Path::new("."), 100)
            .expect("walk past the unreadable directory");
        assert_eq!(candidates.files, [PathBuf::from("open.txt")]);
        assert!(
            workspace
                .walk_candidate_files(&CancellationToken::default(), Path::new("locked"), 100)
                .is_err(),
            "an unreadable search root is still an error"
        );
    }

    #[test]
    fn glob_braces_expand_to_alternatives() {
        let cases = [
            ("*.{rs,toml}", "src/main.rs", true),
            ("*.{rs,toml}", "Cargo.toml", true),
            ("*.{rs,toml}", "notes.md", false),
            ("src/{a,b}/*.rs", "src/b/lib.rs", true),
            ("src/{a,b}/*.rs", "src/c/lib.rs", false),
            ("{x,{y,z}}.txt", "z.txt", true),
            ("lit{eral.txt", "lit{eral.txt", true),
            ("{only}.rs", "only.rs", true),
        ];
        for (pattern, name, expected) in cases {
            assert_eq!(
                compile_glob(pattern).expect("compile glob").is_match(name),
                expected,
                "{pattern:?} against {name:?}"
            );
        }
        let explosive = "{a,b}".repeat(9);
        assert!(compile_glob(&explosive).is_err());
    }

    #[test]
    fn candidate_caps_truncate_with_a_notice_instead_of_failing() {
        let (directory, workspace) = workspace();
        for name in ["a.txt", "b.txt", "c.txt"] {
            fs::write(directory.path.join(name), "").expect("write candidate");
        }
        let candidates = workspace
            .collect_candidates(&CancellationToken::default(), Path::new("."), 2)
            .expect("capped candidates");
        assert_eq!(candidates.files.len(), 2);
        assert!(candidates.truncated);
        assert!(candidate_truncation_notice(true).contains("candidate files"));
        assert!(candidate_truncation_notice(false).is_empty());
    }
}
