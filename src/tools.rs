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
        Arc, Mutex,
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
pub const MAX_READ_SCAN_BYTES: usize = MAX_READ_BYTES * 40;
/// Maximum bytes held for an exact `edit`.
pub const MAX_EDIT_BYTES: usize = 10 * 1024 * 1024;
/// Maximum captured output from `bash`.
pub const MAX_OUTPUT_BYTES: usize = 30 * 1024;
/// Maximum bytes accepted from `git ls-files`.
pub const MAX_CANDIDATE_BYTES: usize = 8 * 1024 * 1024;
/// Maximum candidate paths a search can inspect.
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
const PROCESS_POLL_INTERVAL: Duration = Duration::from_millis(10);
const MAX_CONTEXT_LINES: usize = 100;
const MAX_GLOB_PATTERN_CHARS: usize = 4_096;
const MAX_REGEX_PATTERN_CHARS: usize = 4_096;
const MAX_REGEX_REPEAT: usize = 1_024;
const MAX_REGEX_STATES: usize = 8_192;
const MAX_REGEX_STEPS: usize = 20_000_000;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

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
        let canonical = fs::canonicalize(requested)
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
        self.write_file_atomic(&relative, content.as_bytes(), cancellation)?;
        Ok(format!(
            "Wrote {} bytes to {}",
            content.len(),
            self.display(&relative)
        ))
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
        let (bytes, truncated) = self.read_limited(&relative, MAX_EDIT_BYTES, cancellation)?;
        if truncated {
            return Err(ToolError::new(format!(
                "{} exceeds the {MAX_EDIT_BYTES}-byte edit limit",
                self.display(&relative)
            )));
        }
        let content = decode_text(&bytes, false, &self.display(&relative))?;

        let mut occurrences = 0usize;
        for _ in content.match_indices(old_text) {
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
            .find(old_text)
            .expect("the unique old_text occurrence was counted");
        let mut updated = String::with_capacity(
            content
                .len()
                .saturating_sub(old_text.len())
                .saturating_add(new_text.len()),
        );
        updated.push_str(&content[..index]);
        updated.push_str(new_text);
        updated.push_str(&content[index + old_text.len()..]);
        self.write_file_atomic(&relative, updated.as_bytes(), cancellation)?;
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

        let mut entries = Vec::new();
        let directory = fs::read_dir(&absolute)
            .map_err(|error| ToolError::io("read directory", &absolute, error))?;
        for entry in directory {
            check_cancelled(cancellation)?;
            if entries.len() >= MAX_DIRECTORY_ENTRIES {
                return Err(ToolError::new(format!(
                    "{} contains more than {MAX_DIRECTORY_ENTRIES} entries",
                    self.display(&relative)
                )));
            }
            let entry =
                entry.map_err(|error| ToolError::io("read directory entry", &absolute, error))?;
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
        for candidate in candidates {
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
                    return Ok(output.finish());
                }
            }
        }
        if matches == 0 {
            Ok("No matches found".to_owned())
        } else {
            Ok(output.finish())
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

        let mut matches = Vec::new();
        let mut limited = false;
        for candidate in candidates {
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
            return Ok("No files found matching pattern".to_owned());
        }
        matches.sort();
        let mut output = render_lines(&matches, MAX_LIST_OUTPUT_BYTES);
        if limited {
            output.push_str(&format!(
                "\n\n[{limit} results limit reached. Refine the pattern or increase limit.]"
            ));
        }
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
        let mut command = {
            let mut command = Command::new("cmd.exe");
            command.args(["/d", "/s", "/c", source]);
            command
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut command = Command::new("sh");
            command.args(["-c", source]);
            command
        };
        command.current_dir(&self.root);
        configure_process_tree(&mut command);

        let timeout = if self.bash_timeout.is_zero() {
            DEFAULT_BASH_TIMEOUT
        } else {
            self.bash_timeout
        };
        let result = run_process(
            &mut command,
            cancellation,
            timeout,
            MAX_OUTPUT_BYTES,
            true,
            true,
        )
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
            if let Ok(canonical) = fs::canonicalize(path) {
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
        let canonical = fs::canonicalize(&absolute)
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
            let canonical = fs::canonicalize(&current)
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
            Ok(metadata) => Some(metadata.permissions()),
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
    ) -> Result<Vec<PathBuf>> {
        check_cancelled(cancellation)?;
        let absolute = self.existing_path(search_root)?;
        let metadata = fs::symlink_metadata(&absolute)
            .map_err(|error| ToolError::io("inspect search path", &absolute, error))?;
        if metadata.is_file() {
            return Ok(vec![search_root.to_path_buf()]);
        }
        if !metadata.is_dir() {
            return Err(ToolError::new(format!(
                "not a regular file or directory: {}",
                self.display(search_root)
            )));
        }

        if let Some(files) = self.git_candidate_files(cancellation, search_root)? {
            return Ok(files);
        }
        self.walk_candidate_files(cancellation, search_root)
    }

    /// Uses git's ignored-file-aware index when it is available.
    fn git_candidate_files(
        &self,
        cancellation: &CancellationToken,
        search_root: &Path,
    ) -> Result<Option<Vec<PathBuf>>> {
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
        if result.output.truncated {
            return Err(ToolError::new(format!(
                "candidate file list exceeds {MAX_CANDIDATE_BYTES} bytes"
            )));
        }
        if let Some(error) = result.reader_error {
            return Err(ToolError::new(format!(
                "could not read git candidate list: {error}"
            )));
        }

        let mut files = Vec::new();
        for entry in result.output.bytes.split(|byte| *byte == 0) {
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
            files.push(relative);
            if files.len() > MAX_CANDIDATE_FILES {
                return Err(ToolError::new(format!(
                    "candidate file list exceeds {MAX_CANDIDATE_FILES} files"
                )));
            }
        }
        files.sort();
        Ok(Some(files))
    }

    /// Conservative, non-symlink-following candidate discovery.
    fn walk_candidate_files(
        &self,
        cancellation: &CancellationToken,
        search_root: &Path,
    ) -> Result<Vec<PathBuf>> {
        let mut directories = VecDeque::from([search_root.to_path_buf()]);
        let mut files = Vec::new();
        while let Some(directory_relative) = directories.pop_front() {
            check_cancelled(cancellation)?;
            let directory_absolute = self.root.join(&directory_relative);
            let entries = fs::read_dir(&directory_absolute).map_err(|error| {
                ToolError::io("read search directory", &directory_absolute, error)
            })?;
            for entry in entries {
                check_cancelled(cancellation)?;
                let entry = entry.map_err(|error| {
                    ToolError::io("read search directory entry", &directory_absolute, error)
                })?;
                let name = entry.file_name();
                let relative = join_relative(&directory_relative, &name);
                let file_type = entry.file_type().map_err(|error| {
                    ToolError::io("inspect search directory entry", &entry.path(), error)
                })?;
                if file_type.is_dir() {
                    if !skip_search_dir(&name) {
                        let canonical = fs::canonicalize(entry.path()).map_err(|error| {
                            ToolError::io("resolve search directory", &entry.path(), error)
                        })?;
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
                files.push(relative);
                if files.len() > MAX_CANDIDATE_FILES {
                    return Err(ToolError::new(format!(
                        "candidate file list exceeds {MAX_CANDIDATE_FILES} files"
                    )));
                }
            }
        }
        files.sort();
        Ok(files)
    }

    fn display(&self, relative: &Path) -> String {
        path_to_slash(relative).unwrap_or_else(|_| relative.to_string_lossy().into_owned())
    }
}

/// Convenience constructor mirroring the Go package's `NewWorkspace`.
pub fn new_workspace(root: impl AsRef<Path>) -> Result<Workspace> {
    Workspace::new(root)
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
/// Shell commands opt into isolated process-tree termination so an abort or
/// timeout reaches their descendants. Other short-lived helpers retain the
/// immediate-child behavior.
fn run_process(
    command: &mut Command,
    cancellation: &CancellationToken,
    timeout: Duration,
    output_limit: usize,
    combine_streams: bool,
    isolated_process_tree: bool,
) -> io::Result<ProcessResult> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
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
    let (status, timed_out, cancelled) = loop {
        if let Some(status) = child.try_wait()? {
            break (status, false, cancellation.is_cancelled());
        }
        if cancellation.is_cancelled() {
            terminate_process(&mut child, isolated_process_tree);
            break (child.wait()?, false, true);
        }
        if started.elapsed() >= timeout {
            terminate_process(&mut child, isolated_process_tree);
            break (child.wait()?, true, false);
        }
        thread::sleep(PROCESS_POLL_INTERVAL);
    };

    let drain_deadline = Instant::now() + OUTPUT_DRAIN_GRACE;
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

/// Places a shell command in a separate process group before it executes.
/// This makes a negative-PID signal target the shell and all descendants
/// without risking the agent's own terminal group.
fn configure_process_tree(command: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;

        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = command;
    }
}

fn terminate_process(child: &mut Child, isolated_process_tree: bool) {
    #[cfg(unix)]
    if isolated_process_tree && terminate_process_group(child.id()) {
        return;
    }

    #[cfg(windows)]
    if isolated_process_tree {
        let pid = child.id().to_string();
        if Command::new("taskkill")
            .args(["/T", "/F", "/PID", &pid])
            .status()
            .is_ok_and(|status| status.success())
        {
            return;
        }
    }

    let _ = child.kill();
}

#[cfg(unix)]
fn terminate_process_group(id: u32) -> bool {
    let Ok(id) = i32::try_from(id) else {
        return false;
    };
    // `configure_process_tree` makes the shell the group leader, so its
    // negative ID reaches the complete tool-owned process tree.
    unsafe { libc::kill(-id, libc::SIGKILL) == 0 }
}

/// A compiled glob with pi-style basename fallback and Unicode-safe matching.
#[derive(Clone, Debug)]
pub struct GlobPattern {
    tokens: Vec<GlobToken>,
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

/// Compiles `*`, `**`, `**/`, and `?` glob syntax.
pub fn compile_glob(pattern: &str) -> Result<GlobPattern> {
    if pattern.chars().count() > MAX_GLOB_PATTERN_CHARS {
        return Err(ToolError::new(format!(
            "glob pattern exceeds {MAX_GLOB_PATTERN_CHARS} characters"
        )));
    }
    let normalized = normalize_glob_separators(pattern);
    let characters = normalized.chars().collect::<Vec<_>>();
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
    Ok(GlobPattern {
        tokens,
        basename_too: !normalized.contains('/'),
    })
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
        let characters = name.chars().collect::<Vec<_>>();
        let mut current = vec![false; self.tokens.len() + 1];
        add_glob_closure(&self.tokens, &mut current, 0);

        for character in characters {
            let mut next = vec![false; self.tokens.len() + 1];
            for (index, active) in current.iter().enumerate().take(self.tokens.len()) {
                if !*active {
                    continue;
                }
                match self.tokens[index] {
                    GlobToken::Literal(expected) if expected == character => {
                        add_glob_closure(&self.tokens, &mut next, index + 1);
                    }
                    GlobToken::One if character != '/' => {
                        add_glob_closure(&self.tokens, &mut next, index + 1);
                    }
                    GlobToken::Star if character != '/' => {
                        add_glob_closure(&self.tokens, &mut next, index);
                    }
                    GlobToken::GlobStar => {
                        add_glob_closure(&self.tokens, &mut next, index);
                    }
                    GlobToken::GlobStarSlash => {
                        add_glob_closure(&self.tokens, &mut next, index);
                        if character == '/' {
                            add_glob_closure(&self.tokens, &mut next, index + 1);
                        }
                    }
                    _ => {}
                }
            }
            current = next;
        }
        current[self.tokens.len()]
    }
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
// anchors, character classes, escapes, and normal quantifiers. It is bounded
// by state and work limits so model-provided patterns cannot backtrack
// exponentially or consume unbounded CPU.

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
                characters.len(),
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
                        characters.len(),
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
        text_length: usize,
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
                RegexState::End { next } if position == text_length => pending.push(next),
                RegexState::Start { .. } | RegexState::End { .. } => {}
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
            '\\' => self.parse_escape().map(RegexExpr::Consume),
            '.' => Ok(RegexExpr::Consume(CharMatcher::Any)),
            '^' => Ok(RegexExpr::Start),
            '$' => Ok(RegexExpr::End),
            '*' | '+' | '?' | ')' | '|' => Err(ToolError::new(format!(
                "regex quantifier or delimiter {character:?} has no target"
            ))),
            literal => Ok(RegexExpr::Consume(CharMatcher::Literal(literal))),
        }
    }

    fn parse_escape(&mut self) -> Result<CharMatcher> {
        let character = self
            .next()
            .ok_or_else(|| ToolError::new("trailing regex escape"))?;
        match character {
            'd' => Ok(CharMatcher::Class(kind_class(CharacterKind::Digit, false))),
            'D' => Ok(CharMatcher::Class(kind_class(CharacterKind::Digit, true))),
            's' => Ok(CharMatcher::Class(kind_class(
                CharacterKind::Whitespace,
                false,
            ))),
            'S' => Ok(CharMatcher::Class(kind_class(
                CharacterKind::Whitespace,
                true,
            ))),
            'w' => Ok(CharMatcher::Class(kind_class(CharacterKind::Word, false))),
            'W' => Ok(CharMatcher::Class(kind_class(CharacterKind::Word, true))),
            'n' => Ok(CharMatcher::Literal('\n')),
            'r' => Ok(CharMatcher::Literal('\r')),
            't' => Ok(CharMatcher::Literal('\t')),
            literal => Ok(CharMatcher::Literal(literal)),
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
            literal => Ok(ClassItem::Char(literal)),
        }
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
                | (BuildState::End { next }, PatchSlot::Next) => *next = Some(destination),
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

    #[cfg(unix)]
    #[test]
    fn bash_cancellation_terminates_background_descendants() {
        let (directory, workspace) = workspace();
        let pid_path = directory.path.join("background.pid");
        let command = format!(
            "sleep 20 >/dev/null 2>&1 & echo $! > {}; wait",
            shell_quote(&pid_path)
        );
        let cancellation = CancellationToken::default();
        let running_token = cancellation.clone();
        let tool = workspace.bash_tool();
        let task = thread::spawn(move || {
            (tool.execute)(
                running_token,
                "cancel-process-tree".to_owned(),
                parameters([("command", json!(command))]),
                Arc::new(|_| {}),
            )
        });

        for _ in 0..100 {
            if pid_path.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let child_pid = fs::read_to_string(&pid_path)
            .expect("background child should start")
            .trim()
            .parse::<u32>()
            .expect("background child PID");
        cancellation.cancel();
        let error = task
            .join()
            .expect("bash tool thread must not panic")
            .expect_err("cancelled shell command must fail");
        assert!(error.contains("cancelled"));

        let deadline = Instant::now() + Duration::from_secs(1);
        while process_is_alive(child_pid) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        let alive = process_is_alive(child_pid);
        if alive {
            terminate_test_process(child_pid);
        }
        assert!(
            !alive,
            "background process {child_pid} survived cancellation"
        );
    }

    #[cfg(unix)]
    fn shell_quote(path: &Path) -> String {
        format!("'{}'", path.display().to_string().replace('\'', "'\\''"))
    }

    #[cfg(unix)]
    fn process_is_alive(pid: u32) -> bool {
        #[cfg(target_os = "linux")]
        if let Ok(stat) = fs::read_to_string(format!("/proc/{pid}/stat"))
            && let Some((_, state)) = stat.rsplit_once(") ")
        {
            return !state.starts_with('Z');
        }
        let Ok(pid) = i32::try_from(pid) else {
            return false;
        };
        let result = unsafe { libc::kill(pid, 0) };
        result == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }

    #[cfg(unix)]
    fn terminate_test_process(pid: u32) {
        if let Ok(pid) = i32::try_from(pid) {
            let _ = unsafe { libc::kill(pid, libc::SIGKILL) };
        }
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
}
