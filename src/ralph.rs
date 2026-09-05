//! Durable Ralph loop state and prompt helpers.
//!
//! This module deliberately has no terminal or agent-runtime lifecycle
//! integration. A caller can parse a CLI or slash command with
//! [`parse_command`], execute it against [`Store`], register [`Store::tools`]
//! with an agent, and use [`prepare_next_turn`] plus the prompt helpers to wire
//! the resulting state into its own runtime.
//!
//! Unlike the older workspace-local `.ralph` implementation, this store keeps
//! its files below the agent configuration directory:
//!
//! ```text
//! <agent-config>/ralph/<workspace-key>/<loop>.md
//! <agent-config>/ralph/<workspace-key>/<loop>.state.json
//! <agent-config>/ralph/<workspace-key>/archive/
//! ```
//!
//! The workspace key is a stable hash of the canonical workspace path. It
//! avoids leaking a full path into the configuration layout and keeps loops
//! with the same name in different workspaces independent.

use std::{
    collections::BTreeMap,
    error::Error as StdError,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{
    agent,
    llm::{self, AssistantMessage, ContentBlock},
};

/// Directory below the agent configuration directory that owns Ralph data.
pub const STORE_DIR: &str = "ralph";
pub const ARCHIVE_DIR: &str = "archive";
pub const COMPLETE_MARKER: &str = "<promise>COMPLETE</promise>";
pub const DEFAULT_MAX_ITERATIONS: i64 = 50;
pub const MAX_RALPH_FILE_BYTES: usize = 2 * 1024 * 1024;
pub const START_TOOL_NAME: &str = "ralph_start";
pub const DONE_TOOL_NAME: &str = "ralph_done";

/// The task document used when code starts a loop without task content.
pub const DEFAULT_TEMPLATE: &str = r#"# Task

Describe your task here.

## Goals
- Goal 1
- Goal 2

## Checklist
- [ ] Item 1
- [ ] Item 2

## Verification
- Commands run, working directories, relevant environment variables, outputs, and preserved artifacts

## Final Verification
- Exact monitor-rerunnable command: <command>
- Working directory: <path>
- Required preserved artifacts: <paths>
- Result: <output summary>

## Notes
(Update this as you work)
"#;

/// The completion bar carried in every iteration prompt.
pub const DEFAULT_COMPLETION_GATE: &str = r#"COMPLETION GATE

Do not output <promise>COMPLETE</promise> based only on checked checklist items.
Before completion:
1. Run a final verification command that an external monitor can rerun from the same worktree in a fresh shell.
2. Record the exact command, working directory, relevant environment variables, and output summary in the task file.
3. Preserve every artifact required by that command, including build directories, generated libraries, virtualenvs, caches, or copied dylibs.
4. If cleanup removes required artifacts, recreate them or update the final command before completing.
5. If the final command cannot be made externally rerunnable, mark the item blocked/deferred instead of complete."#;

/// The guard that prevents a queued prompt from reviving a completed loop.
pub const DEFAULT_STALE_PROMPT_GUARD: &str = r#"STALE PROMPT GUARD

Before doing any work from a Ralph prompt, reload the loop state file named in the prompt from the agent configuration directory.
If the state says "status": "completed", do not edit files, do not run task commands, and do not call ralph_done. Reply briefly that the stale prompt was ignored because the loop is already completed."#;

/// The reflection checkpoint inserted at configured iteration boundaries.
pub const DEFAULT_REFLECT_INSTRUCTIONS: &str = r#"REFLECTION CHECKPOINT

Pause and reflect on your progress:
1. What has been accomplished so far?
2. What's working well?
3. What's not working or blocking progress?
4. Should the approach be adjusted?
5. What are the next priorities?

Update the task file with your reflection, then continue working."#;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub type Result<T> = std::result::Result<T, RalphError>;

/// Errors surfaced by persistence, lifecycle operations, and command parsing.
#[derive(Debug)]
pub enum RalphError {
    Io(io::Error),
    Json(serde_json::Error),
    InvalidName(String),
    InvalidOptions(String),
    InvalidState(String),
    NotFound(String),
    AlreadyActive(String),
    PausedLoop { name: String, iteration: u64 },
    CompletedLoop(String),
    NoActiveLoop,
    InvalidCommand(String),
    FileTooLarge { path: PathBuf, max_bytes: usize },
    UnsafePath(PathBuf),
    IterationOverflow,
}

impl fmt::Display for RalphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
            Self::InvalidName(name) => {
                write!(formatter, "a loop name is required (got {name:?})")
            }
            Self::InvalidOptions(reason) => write!(formatter, "invalid loop options: {reason}"),
            Self::InvalidState(reason) => write!(formatter, "invalid Ralph loop state: {reason}"),
            Self::NotFound(name) => write!(formatter, "no loop named {name:?}"),
            Self::AlreadyActive(name) => write!(formatter, "loop {name:?} is already active"),
            Self::PausedLoop { name, iteration } => write!(
                formatter,
                "loop {name:?} is paused at iteration {iteration}; resume it, or delete or archive it first"
            ),
            Self::CompletedLoop(name) => write!(formatter, "loop {name:?} is already completed"),
            Self::NoActiveLoop => formatter.write_str("no active Ralph loop owned by this session"),
            Self::InvalidCommand(reason) => formatter.write_str(reason),
            Self::FileTooLarge { path, max_bytes } => write!(
                formatter,
                "Ralph file {} exceeds the {max_bytes}-byte limit",
                path.display()
            ),
            Self::UnsafePath(path) => write!(
                formatter,
                "refusing Ralph path {} because it is a symlink or escapes the agent configuration directory",
                path.display()
            ),
            Self::IterationOverflow => formatter.write_str("Ralph iteration counter overflowed"),
        }
    }
}

impl StdError for RalphError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for RalphError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for RalphError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// Persisted lifecycle status for a loop.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopStatus {
    Active,
    #[default]
    Paused,
    Completed,
}

impl LoopStatus {
    pub fn icon(self) -> &'static str {
        match self {
            Self::Active => "▶",
            Self::Paused => "⏸",
            Self::Completed => "✓",
        }
    }
}

/// A persisted loop record. `task_file` is relative to the agent config root.
///
/// Field names intentionally retain the existing camel-case JSON layout so
/// state files remain easy to inspect and migration tools can recognize them.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct LoopState {
    pub name: String,
    #[serde(rename = "taskFile")]
    pub task_file: String,
    pub iteration: u64,
    #[serde(rename = "maxIterations")]
    pub max_iterations: i64,
    #[serde(rename = "itemsPerIteration")]
    pub items_per_iteration: i64,
    #[serde(rename = "reflectEvery")]
    pub reflect_every: i64,
    #[serde(rename = "reflectInstructions")]
    pub reflect_instructions: String,
    pub active: bool,
    pub status: LoopStatus,
    #[serde(rename = "startedAt")]
    pub started_at: String,
    #[serde(rename = "completedAt", skip_serializing_if = "String::is_empty")]
    pub completed_at: String,
    #[serde(rename = "lastReflectionAt")]
    pub last_reflection_at: u64,
    #[serde(rename = "ownerSessionId", skip_serializing_if = "String::is_empty")]
    pub owner_session_id: String,
}

#[derive(Deserialize)]
struct StoredLoopState {
    #[serde(default)]
    name: String,
    #[serde(rename = "taskFile", default)]
    task_file: String,
    #[serde(default)]
    iteration: u64,
    #[serde(rename = "maxIterations", default)]
    max_iterations: i64,
    #[serde(rename = "itemsPerIteration", default)]
    items_per_iteration: i64,
    #[serde(rename = "reflectEvery", default)]
    reflect_every: i64,
    #[serde(rename = "reflectInstructions", default)]
    reflect_instructions: String,
    #[serde(default)]
    active: bool,
    #[serde(default)]
    status: Option<LoopStatus>,
    #[serde(rename = "startedAt", default)]
    started_at: String,
    #[serde(rename = "completedAt", default)]
    completed_at: String,
    #[serde(rename = "lastReflectionAt", default)]
    last_reflection_at: u64,
    #[serde(rename = "ownerSessionId", default)]
    owner_session_id: String,
}

impl<'de> Deserialize<'de> for LoopState {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let stored = StoredLoopState::deserialize(deserializer)?;
        let status = stored.status.unwrap_or(if stored.active {
            LoopStatus::Active
        } else {
            LoopStatus::Paused
        });
        let mut state = Self {
            name: stored.name,
            task_file: stored.task_file,
            iteration: stored.iteration,
            max_iterations: stored.max_iterations,
            items_per_iteration: stored.items_per_iteration,
            reflect_every: stored.reflect_every,
            reflect_instructions: stored.reflect_instructions,
            active: stored.active,
            status,
            started_at: stored.started_at,
            completed_at: stored.completed_at,
            last_reflection_at: stored.last_reflection_at,
            owner_session_id: stored.owner_session_id,
        };
        state.normalize();
        Ok(state)
    }
}

impl LoopState {
    /// Mirrors the old `active` field from the authoritative lifecycle status.
    pub fn normalize(&mut self) {
        self.active = self.status == LoopStatus::Active;
        if self.reflect_instructions.is_empty() {
            self.reflect_instructions = DEFAULT_REFLECT_INSTRUCTIONS.to_owned();
        }
    }

    /// Renders a concise line suitable for a command result or status rail.
    pub fn summary(&self) -> String {
        let iteration = if self.max_iterations > 0 {
            format!("{}/{}", self.iteration, self.max_iterations)
        } else {
            self.iteration.to_string()
        };
        format!(
            "{}: {} {} (iteration {iteration})",
            self.name,
            self.status.icon(),
            self.status
        )
    }

    fn validate(&self) -> Result<()> {
        if self.name.is_empty() || self.name != sanitize_name(&self.name) {
            return Err(RalphError::InvalidState(format!(
                "name {:?} is not a normalized loop name",
                self.name
            )));
        }
        if self.iteration == 0 {
            return Err(RalphError::InvalidState(
                "iteration must be at least one".to_owned(),
            ));
        }
        if self.items_per_iteration < 0 {
            return Err(RalphError::InvalidState(
                "itemsPerIteration cannot be negative".to_owned(),
            ));
        }
        if self.reflect_every < 0 {
            return Err(RalphError::InvalidState(
                "reflectEvery cannot be negative".to_owned(),
            ));
        }
        Ok(())
    }
}

impl fmt::Display for LoopStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => formatter.write_str("active"),
            Self::Paused => formatter.write_str("paused"),
            Self::Completed => formatter.write_str("completed"),
        }
    }
}

/// Settings that control a new loop.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LoopOptions {
    /// Zero uses [`DEFAULT_MAX_ITERATIONS`]; a negative value is unbounded.
    pub max_iterations: i64,
    /// A prompt-only pacing hint. Zero means no item-count hint.
    pub items_per_iteration: i64,
    /// Insert a reflection checkpoint every N completed transitions. Zero disables it.
    pub reflect_every: i64,
}

impl LoopOptions {
    fn validate(self) -> Result<Self> {
        if self.items_per_iteration < 0 {
            return Err(RalphError::InvalidOptions(
                "items_per_iteration cannot be negative".to_owned(),
            ));
        }
        if self.reflect_every < 0 {
            return Err(RalphError::InvalidOptions(
                "reflect_every cannot be negative".to_owned(),
            ));
        }
        Ok(self)
    }
}

/// Compatibility aliases for callers porting directly from the Go package.
pub type Options = LoopOptions;
pub type State = LoopState;
pub type Status = LoopStatus;

/// Outcome of moving an active loop forward by one iteration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Advance {
    pub done: bool,
    pub reflection: bool,
}

/// A prompt result suitable for an agent follow-up queue.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NextIteration {
    Prompt { prompt: String, reflection: bool },
    Completed,
}

/// One workspace-scoped loop store under the agent configuration directory.
#[derive(Clone, Debug)]
pub struct Store {
    agent_config_dir: PathBuf,
    workspace: PathBuf,
    workspace_key: String,
    session_id: String,
    // Tool closures own a cloned Store. Sharing this cache keeps those
    // closures and direct lifecycle calls scoped to the same current loop.
    current: Arc<Mutex<Option<String>>>,
}

impl Store {
    /// Creates a store beneath an explicit agent configuration directory.
    ///
    /// Supplying the config root explicitly makes this type easy to embed in a
    /// CLI, a slash-command handler, or tests without changing global
    /// environment variables.
    pub fn new(
        agent_config_dir: impl AsRef<Path>,
        workspace: impl AsRef<Path>,
        session_id: impl Into<String>,
    ) -> Self {
        let agent_config_dir = absolute_path(agent_config_dir.as_ref());
        let workspace = canonical_or_absolute(workspace.as_ref());
        let workspace_key = workspace_key(&workspace);
        Self {
            agent_config_dir,
            workspace,
            workspace_key,
            session_id: session_id.into(),
            current: Arc::new(Mutex::new(None)),
        }
    }

    /// Creates a store using GoshCoder's normal agent configuration directory.
    pub fn for_workspace(workspace: impl AsRef<Path>, session_id: impl Into<String>) -> Self {
        Self::new(crate::config::agent_dir(), workspace, session_id)
    }

    /// Creates a default-config store for the process's current working directory.
    pub fn for_current_workspace(session_id: impl Into<String>) -> Result<Self> {
        let workspace = std::env::current_dir()?;
        Ok(Self::for_workspace(workspace, session_id))
    }

    pub fn agent_config_dir(&self) -> &Path {
        &self.agent_config_dir
    }

    pub fn workspace(&self) -> &Path {
        &self.workspace
    }

    pub fn workspace_key(&self) -> &str {
        &self.workspace_key
    }

    /// Returns the lexical (not necessarily created) workspace storage root.
    pub fn storage_dir(&self) -> PathBuf {
        self.agent_config_dir
            .join(STORE_DIR)
            .join(&self.workspace_key)
    }

    pub fn archive_dir(&self) -> PathBuf {
        self.storage_dir().join(ARCHIVE_DIR)
    }

    /// Returns the agent-config-relative task path that will be stored for a loop.
    pub fn task_file(&self, name: &str, archived: bool) -> Result<String> {
        let name = checked_name(name)?;
        Ok(self.relative_task_file(&name, archived))
    }

    pub fn task_path(&self, name: &str, archived: bool) -> Result<PathBuf> {
        let name = checked_name(name)?;
        Ok(self.checked_store_dir(archived)?.join(format!("{name}.md")))
    }

    pub fn state_path(&self, name: &str, archived: bool) -> Result<PathBuf> {
        let name = checked_name(name)?;
        Ok(self
            .checked_store_dir(archived)?
            .join(format!("{name}.state.json")))
    }

    /// Reads a loop state. `Ok(None)` means the named loop does not exist.
    pub fn load(&self, name: &str, archived: bool) -> Result<Option<LoopState>> {
        let name = checked_name(name)?;
        let path = self
            .checked_store_dir(archived)?
            .join(format!("{name}.state.json"));
        let bytes = match read_bounded(&path) {
            Ok(bytes) => bytes,
            Err(RalphError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let state: LoopState = serde_json::from_slice(&bytes)?;
        self.validate_state_location(&state, archived)?;
        Ok(Some(state))
    }

    /// Saves a state atomically. `archived` selects the active or archive area.
    pub fn save(&self, state: &mut LoopState, archived: bool) -> Result<()> {
        state.normalize();
        self.validate_state_location(state, archived)?;
        let path = self
            .checked_store_dir(archived)?
            .join(format!("{}.state.json", state.name));
        let mut encoded = serde_json::to_vec_pretty(state)?;
        encoded.push(b'\n');
        atomic_write(&path, &encoded)
    }

    /// Lists valid loop records in lexical name order. Corrupt records are
    /// ignored so one damaged state file does not hide all other loops.
    pub fn list(&self, archived: bool) -> Result<Vec<LoopState>> {
        let directory = self.checked_store_dir(archived)?;
        let mut states = Vec::new();
        for entry in fs::read_dir(directory)? {
            let Ok(entry) = entry else {
                continue;
            };
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            let Some(name) = file_name.strip_suffix(".state.json") else {
                continue;
            };
            if let Ok(Some(state)) = self.load(name, archived) {
                states.push(state);
            }
        }
        states.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(states)
    }

    /// Reads the task associated with either an active or archived state.
    pub fn read_task(&self, state: &LoopState) -> Result<String> {
        let archived = self.state_is_archived(state)?;
        let path = self
            .checked_store_dir(archived)?
            .join(format!("{}.md", state.name));
        let bytes = read_bounded(&path)?;
        String::from_utf8(bytes)
            .map_err(|error| RalphError::InvalidState(format!("task is not UTF-8: {error}")))
    }

    /// Replaces a task atomically. The supplied state must belong to this store.
    pub fn write_task(&self, state: &LoopState, content: &str) -> Result<()> {
        let archived = self.state_is_archived(state)?;
        let path = self
            .checked_store_dir(archived)?
            .join(format!("{}.md", state.name));
        atomic_write(&path, content.as_bytes())
    }

    /// Returns the active loop owned by this session, adopting an unowned one
    /// after a process restart. It never adopts a loop owned by another session.
    pub fn current(&self) -> Result<Option<LoopState>> {
        let cached = lock(self.current.as_ref()).clone();
        if let Some(name) = cached {
            match self.load(&name, false)? {
                Some(state)
                    if state.status == LoopStatus::Active && self.owned_by_this_session(&state) =>
                {
                    return Ok(Some(state));
                }
                _ => self.set_current(None),
            }
        }

        for state in self.list(false)? {
            if state.status == LoopStatus::Active && self.owned_by_this_session(&state) {
                self.set_current(Some(state.name.clone()));
                return Ok(Some(state));
            }
        }
        Ok(None)
    }

    /// Alias for command handlers that call the lifecycle action `status`.
    pub fn status(&self) -> Result<Option<LoopState>> {
        self.current()
    }

    /// Starts a new active loop and writes its task and state files.
    pub fn start(&self, name: &str, task_content: &str, options: LoopOptions) -> Result<LoopState> {
        let name = checked_name(name)?;
        let options = options.validate()?;
        if let Some(existing) = self.load(&name, false)? {
            match existing.status {
                LoopStatus::Active => return Err(RalphError::AlreadyActive(name)),
                LoopStatus::Paused => {
                    return Err(RalphError::PausedLoop {
                        name,
                        iteration: existing.iteration,
                    });
                }
                LoopStatus::Completed => {}
            }
        }

        let mut state = LoopState {
            name: name.clone(),
            task_file: self.relative_task_file(&name, false),
            iteration: 1,
            max_iterations: if options.max_iterations == 0 {
                DEFAULT_MAX_ITERATIONS
            } else {
                options.max_iterations
            },
            items_per_iteration: options.items_per_iteration,
            reflect_every: options.reflect_every,
            reflect_instructions: DEFAULT_REFLECT_INSTRUCTIONS.to_owned(),
            active: true,
            status: LoopStatus::Active,
            started_at: now_timestamp(),
            completed_at: String::new(),
            last_reflection_at: 0,
            owner_session_id: self.session_id.clone(),
        };
        let task_content = if task_content.is_empty() {
            DEFAULT_TEMPLATE
        } else {
            task_content
        };
        self.write_task(&state, task_content)?;
        self.save(&mut state, false)?;
        self.set_current(Some(name));
        Ok(state)
    }

    /// Advances an active loop. A loop with a positive maximum finishes only
    /// after its last allowed iteration has completed.
    pub fn advance(&self, state: &mut LoopState) -> Result<Advance> {
        self.validate_state_location(state, false)?;
        if state.status != LoopStatus::Active {
            return Err(RalphError::InvalidState(format!(
                "cannot advance a {} loop",
                state.status
            )));
        }

        state.iteration = state
            .iteration
            .checked_add(1)
            .ok_or(RalphError::IterationOverflow)?;
        if state.max_iterations > 0 && state.iteration > state.max_iterations as u64 {
            self.complete(state)?;
            return Ok(Advance {
                done: true,
                reflection: false,
            });
        }

        let reflection = state.reflect_every > 0
            && (state.iteration - 1).is_multiple_of(state.reflect_every as u64);
        if reflection {
            state.last_reflection_at = state.iteration;
        }
        self.save(state, false)?;
        Ok(Advance {
            done: false,
            reflection,
        })
    }

    /// Advances and renders the following iteration prompt. If the task cannot
    /// be read, the loop is paused so callers do not spin without context.
    pub fn advance_to_next_prompt(&self, state: &mut LoopState) -> Result<NextIteration> {
        let advance = self.advance(state)?;
        if advance.done {
            return Ok(NextIteration::Completed);
        }
        let task = match self.read_task(state) {
            Ok(task) => task,
            Err(error) => {
                self.pause(state)?;
                return Err(error);
            }
        };
        Ok(NextIteration::Prompt {
            prompt: build_prompt(state, &task, advance.reflection),
            reflection: advance.reflection,
        })
    }

    /// Marks a loop complete without deleting its task record.
    pub fn complete(&self, state: &mut LoopState) -> Result<()> {
        self.validate_state_location(state, false)?;
        state.status = LoopStatus::Completed;
        state.completed_at = now_timestamp();
        self.save(state, false)?;
        self.clear_current_if(&state.name);
        Ok(())
    }

    /// Pauses a loop while preserving its task and iteration count.
    pub fn pause(&self, state: &mut LoopState) -> Result<()> {
        self.validate_state_location(state, false)?;
        state.status = LoopStatus::Paused;
        self.save(state, false)?;
        self.clear_current_if(&state.name);
        Ok(())
    }

    /// Reactivates a non-completed loop and gives ownership to this session.
    pub fn resume(&self, name: &str) -> Result<LoopState> {
        let name = checked_name(name)?;
        let mut state = self
            .load(&name, false)?
            .ok_or_else(|| RalphError::NotFound(name.clone()))?;
        if state.status == LoopStatus::Completed {
            return Err(RalphError::CompletedLoop(name));
        }
        state.status = LoopStatus::Active;
        state.owner_session_id = self.session_id.clone();
        self.save(&mut state, false)?;
        self.set_current(Some(state.name.clone()));
        Ok(state)
    }

    /// Implements the user-facing `stop` lifecycle action.
    ///
    /// With no name, it stops this session's active loop. A supplied name can
    /// stop any loop in this workspace scope.
    pub fn stop(&self, name: Option<&str>) -> Result<LoopState> {
        let mut state = match name {
            Some(name) => {
                let name = checked_name(name)?;
                self.load(&name, false)?.ok_or(RalphError::NotFound(name))?
            }
            None => self.current()?.ok_or(RalphError::NoActiveLoop)?,
        };
        self.complete(&mut state)?;
        Ok(state)
    }

    /// Moves a loop's task and state into the archive directory. Destination
    /// files are committed before sources are removed, favoring duplicates over
    /// data loss if an interruption occurs mid-operation.
    pub fn archive(&self, name: &str) -> Result<()> {
        let name = checked_name(name)?;
        let state = self
            .load(&name, false)?
            .ok_or_else(|| RalphError::NotFound(name.clone()))?;
        let task = self.read_task(&state)?;
        let mut archived = state.clone();
        archived.task_file = self.relative_task_file(&archived.name, true);

        self.write_task(&archived, &task)?;
        self.save(&mut archived, true)?;

        let state_path = self
            .checked_store_dir(false)?
            .join(format!("{}.state.json", state.name));
        let task_path = self
            .checked_store_dir(false)?
            .join(format!("{}.md", state.name));
        remove_file_if_exists(&state_path)?;
        remove_file_if_exists(&task_path)?;
        self.clear_current_if(&state.name);
        Ok(())
    }

    /// Deletes an active loop's state and task. Archived loops are retained.
    pub fn delete(&self, name: &str) -> Result<()> {
        let name = checked_name(name)?;
        if self.load(&name, false)?.is_none() {
            return Err(RalphError::NotFound(name));
        }
        let state_path = self
            .checked_store_dir(false)?
            .join(format!("{name}.state.json"));
        let task_path = self.checked_store_dir(false)?.join(format!("{name}.md"));
        remove_file_if_exists(&state_path)?;
        remove_file_if_exists(&task_path)?;
        self.clear_current_if(&name);
        Ok(())
    }

    /// Renders the current loop indicator, or an empty string when inactive.
    pub fn status_line(&self) -> Result<String> {
        let Some(state) = self.current()? else {
            return Ok(String::new());
        };
        let max = if state.max_iterations > 0 {
            format!("/{}", state.max_iterations)
        } else {
            String::new()
        };
        let reflection = if state.reflect_every > 0 {
            let every = state.reflect_every as u64;
            let remaining = every - ((state.iteration - 1) % every);
            format!(" · reflect in {remaining}")
        } else {
            String::new()
        };
        Ok(format!(
            "ralph · {} · {} {} · {}{} · {}{}",
            state.name,
            state.status.icon(),
            state.status,
            state.iteration,
            max,
            state.task_file,
            reflection
        ))
    }

    /// Executes a parsed lifecycle command without coupling it to stdout,
    /// stderr, a terminal UI, or an agent queue.
    pub fn execute(&self, command: RalphCommand) -> Result<CommandResult> {
        match command {
            RalphCommand::Start {
                name,
                task_content,
                options,
            } => self
                .start(&name, &task_content, options)
                .map(CommandResult::Started),
            RalphCommand::List { archived } => self.list(archived).map(CommandResult::Listed),
            RalphCommand::Status => self.status().map(CommandResult::Status),
            RalphCommand::Resume { name } => self.resume(&name).map(CommandResult::Resumed),
            RalphCommand::Stop { name } => self.stop(name.as_deref()).map(CommandResult::Stopped),
            RalphCommand::Archive { name } => {
                self.archive(&name)?;
                Ok(CommandResult::Archived(checked_name(&name)?))
            }
            RalphCommand::Delete { name } => {
                self.delete(&name)?;
                Ok(CommandResult::Deleted(checked_name(&name)?))
            }
        }
    }

    /// Builds a system-prompt update after an assistant turn. A completion
    /// marker persists the completed state before loop instructions are removed.
    pub fn prepare_next_turn(
        &self,
        base_system_prompt: &str,
        assistant: &AssistantMessage,
    ) -> Result<TurnPreparation> {
        let Some(mut state) = self.current()? else {
            return Ok(TurnPreparation {
                system_prompt: base_system_prompt.to_owned(),
                active_loop: None,
                completed: false,
            });
        };
        if has_complete_marker(assistant) {
            self.complete(&mut state)?;
            return Ok(TurnPreparation {
                system_prompt: base_system_prompt.to_owned(),
                active_loop: Some(state),
                completed: true,
            });
        }
        Ok(TurnPreparation {
            system_prompt: inject_system_prompt(base_system_prompt, &state),
            active_loop: Some(state),
            completed: false,
        })
    }

    /// Returns the model-facing Ralph tools bound to this store and queue.
    ///
    /// The queue is optional to support callers that only need the lifecycle
    /// result. A live agent should pass an [`agent::Agent`] behind
    /// [`SharedQueue`] so each iteration is delivered as a follow-up.
    #[must_use]
    pub fn tools(&self, queue: Option<SharedQueue>) -> Vec<agent::Tool> {
        vec![self.start_tool(queue.clone()), self.done_tool(queue)]
    }

    /// Returns the model-facing tool that starts a Ralph loop.
    #[must_use]
    pub fn start_tool(&self, queue: Option<SharedQueue>) -> agent::Tool {
        ralph_start(self.clone(), queue)
    }

    /// Returns the model-facing tool that advances a Ralph loop.
    #[must_use]
    pub fn done_tool(&self, queue: Option<SharedQueue>) -> agent::Tool {
        ralph_done(self.clone(), queue)
    }

    fn relative_task_file(&self, name: &str, archived: bool) -> String {
        let mut path = PathBuf::from(STORE_DIR).join(&self.workspace_key);
        if archived {
            path.push(ARCHIVE_DIR);
        }
        path.push(format!("{name}.md"));
        path.to_string_lossy().replace('\\', "/")
    }

    fn validate_state_location(&self, state: &LoopState, archived: bool) -> Result<()> {
        state.validate()?;
        let expected = self.relative_task_file(&state.name, archived);
        if state.task_file != expected {
            return Err(RalphError::InvalidState(format!(
                "taskFile {:?} does not match this store's expected path {:?}",
                state.task_file, expected
            )));
        }
        Ok(())
    }

    fn state_is_archived(&self, state: &LoopState) -> Result<bool> {
        if self.validate_state_location(state, false).is_ok() {
            return Ok(false);
        }
        self.validate_state_location(state, true)?;
        Ok(true)
    }

    fn owned_by_this_session(&self, state: &LoopState) -> bool {
        state.owner_session_id.is_empty() || state.owner_session_id == self.session_id
    }

    fn set_current(&self, name: Option<String>) {
        *lock(self.current.as_ref()) = name;
    }

    fn clear_current_if(&self, name: &str) {
        let mut current = lock(self.current.as_ref());
        if current.as_deref() == Some(name) {
            *current = None;
        }
    }

    /// Creates and verifies the storage directory a component at a time.
    /// Existing symlinks are refused before any loop file is opened.
    fn checked_store_dir(&self, archived: bool) -> Result<PathBuf> {
        fs::create_dir_all(&self.agent_config_dir)?;
        let root = fs::canonicalize(&self.agent_config_dir)?;
        let root_metadata = fs::symlink_metadata(&root)?;
        if !root_metadata.is_dir() {
            return Err(RalphError::UnsafePath(root));
        }

        let mut current = root.clone();
        for component in [
            Some(STORE_DIR),
            Some(self.workspace_key.as_str()),
            archived.then_some(ARCHIVE_DIR),
        ]
        .into_iter()
        .flatten()
        {
            let candidate = current.join(component);
            match fs::symlink_metadata(&candidate) {
                Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                    return Err(RalphError::UnsafePath(candidate));
                }
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    match fs::create_dir(&candidate) {
                        Ok(()) => {}
                        Err(create_error)
                            if create_error.kind() == io::ErrorKind::AlreadyExists => {}
                        Err(create_error) => return Err(create_error.into()),
                    }
                }
                Err(error) => return Err(error.into()),
            }
            let metadata = fs::symlink_metadata(&candidate)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(RalphError::UnsafePath(candidate));
            }
            let canonical = fs::canonicalize(&candidate)?;
            if !canonical.starts_with(&root) {
                return Err(RalphError::UnsafePath(canonical));
            }
            #[cfg(unix)]
            fs::set_permissions(&canonical, fs::Permissions::from_mode(0o700))?;
            current = canonical;
        }
        Ok(current)
    }
}

/// Compatibility alias for callers that prefer a descriptive store name.
pub type RalphStore = Store;

/// Parsed lifecycle operation. Both `ralph …` and `/ralph …` forms are accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RalphCommand {
    Start {
        name: String,
        task_content: String,
        options: LoopOptions,
    },
    List {
        archived: bool,
    },
    Status,
    Resume {
        name: String,
    },
    Stop {
        name: Option<String>,
    },
    Archive {
        name: String,
    },
    Delete {
        name: String,
    },
}

/// Typed result of [`Store::execute`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommandResult {
    Started(LoopState),
    Listed(Vec<LoopState>),
    Status(Option<LoopState>),
    Resumed(LoopState),
    Stopped(LoopState),
    Archived(String),
    Deleted(String),
}

/// A turn preparation result that a runtime can apply to its context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TurnPreparation {
    pub system_prompt: String,
    pub active_loop: Option<LoopState>,
    pub completed: bool,
}

/// Splits and parses a CLI or slash command.
///
/// It accepts `ralph list`, `/ralph list`, or just `list` so a CLI can pass
/// argv tail directly while a chat frontend can pass its full input line.
pub fn parse_command(input: &str) -> Result<RalphCommand> {
    parse_command_args(&split_command_line(input)?)
}

/// Parses already-tokenized command arguments.
pub fn parse_command_args(arguments: &[String]) -> Result<RalphCommand> {
    let arguments = match arguments.first().map(String::as_str) {
        Some("ralph" | "/ralph") => &arguments[1..],
        _ => arguments,
    };
    let Some((subcommand, rest)) = arguments.split_first() else {
        return Err(command_usage());
    };

    match subcommand.as_str() {
        "start" => parse_start(rest),
        "list" => {
            let archived = match rest {
                [] => false,
                [flag] if matches!(flag.as_str(), "--archived" | "-a") => true,
                _ => return Err(command_usage()),
            };
            Ok(RalphCommand::List { archived })
        }
        "status" => {
            if rest.is_empty() {
                Ok(RalphCommand::Status)
            } else {
                Err(command_usage())
            }
        }
        "resume" => parse_required_name(rest).map(|name| RalphCommand::Resume { name }),
        "stop" => match rest {
            [] => Ok(RalphCommand::Stop { name: None }),
            [name] => {
                let name = checked_name(name)?;
                Ok(RalphCommand::Stop { name: Some(name) })
            }
            _ => Err(command_usage()),
        },
        "archive" => parse_required_name(rest).map(|name| RalphCommand::Archive { name }),
        "delete" => parse_required_name(rest).map(|name| RalphCommand::Delete { name }),
        other => Err(RalphError::InvalidCommand(format!(
            "unknown Ralph subcommand {other:?}; use start, list, status, resume, stop, archive, or delete"
        ))),
    }
}

/// Reduces a loop name to a safe filename stem. Non-ASCII letters and
/// punctuation become underscores, and adjacent underscores collapse.
pub fn sanitize_name(name: &str) -> String {
    let mut output = String::with_capacity(name.len());
    let mut previous_underscore = false;
    for character in name.chars() {
        let character = if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
            character
        } else {
            '_'
        };
        if character == '_' && previous_underscore {
            continue;
        }
        previous_underscore = character == '_';
        output.push(character);
    }
    output
}

/// Renders a per-iteration user prompt.
pub fn build_prompt(state: &LoopState, task_content: &str, is_reflection: bool) -> String {
    let max = max_suffix(state);
    let reflection_tag = if is_reflection { " | REFLECTION" } else { "" };
    let rule = "─".repeat(71);
    let mut prompt = format!(
        "{rule}\nRALPH LOOP: {} | Iteration {}{max}{reflection_tag}\n{rule}\n\n",
        state.name, state.iteration
    );
    if is_reflection {
        let instructions = if state.reflect_instructions.is_empty() {
            DEFAULT_REFLECT_INSTRUCTIONS
        } else {
            &state.reflect_instructions
        };
        prompt.push_str(instructions);
        prompt.push_str("\n\n---\n\n");
    }
    prompt.push_str(&format!(
        "## Current Task (from {})\n\n{task_content}\n\n---\n",
        state.task_file
    ));
    prompt.push_str("\n## Stale Prompt Guard\n\n");
    prompt.push_str(DEFAULT_STALE_PROMPT_GUARD);
    prompt.push_str(&format!(
        "\n\nReload state file `{}` before acting on this prompt.\n",
        state_file_from_task_file(&state.task_file)
    ));
    prompt.push_str("\n## Completion Gate\n\n");
    prompt.push_str(DEFAULT_COMPLETION_GATE);
    prompt.push_str("\n\n## Instructions\n\n");
    prompt.push_str(&format!(
        "You are in a Ralph loop (iteration {}{max}).\n\n",
        state.iteration
    ));
    let mut step = 1;
    if state.items_per_iteration > 0 {
        prompt.push_str(&format!(
            "**THIS ITERATION: process approximately {} items, then call ralph_done.**\n\n",
            state.items_per_iteration
        ));
        prompt.push_str(&format!(
            "{step}. Work on the next ~{} items from your checklist\n",
            state.items_per_iteration
        ));
    } else {
        prompt.push_str(&format!("{step}. Continue working on the task\n"));
    }
    step += 1;
    prompt.push_str(&format!(
        "{step}. Update the task file ({}) with your progress\n",
        state.task_file
    ));
    step += 1;
    prompt.push_str(&format!(
        "{step}. When FULLY COMPLETE and the completion gate is satisfied, respond with: {COMPLETE_MARKER}\n"
    ));
    step += 1;
    prompt.push_str(&format!(
        "{step}. Otherwise, call the ralph_done tool to proceed to the next iteration\n"
    ));
    prompt
}

/// Appends the active loop's immutable instructions to a base system prompt.
pub fn system_prompt_suffix(state: &LoopState) -> String {
    let iteration = if state.max_iterations > 0 {
        format!("{}/{}", state.iteration, state.max_iterations)
    } else {
        state.iteration.to_string()
    };
    let mut suffix = format!(
        "\n[RALPH LOOP - {} - Iteration {iteration}]\n\nYou are in a Ralph loop working on: {}\n",
        state.name, state.task_file
    );
    suffix.push_str(&format!(
        "- Before doing work, reload {} from the agent configuration directory; if status is completed, ignore the stale prompt and do not call ralph_done\n",
        state_file_from_task_file(&state.task_file)
    ));
    if state.items_per_iteration > 0 {
        suffix.push_str(&format!(
            "- Work on ~{} items this iteration\n",
            state.items_per_iteration
        ));
    }
    suffix.push_str("- Update the task file as you progress\n");
    suffix.push_str("- Preserve artifacts needed by final verification\n");
    suffix.push_str("- Record an exact, externally rerunnable final command before completion\n");
    suffix.push_str(&format!(
        "- When FULLY COMPLETE and externally rerunnable: {COMPLETE_MARKER}\n"
    ));
    suffix.push_str("- Otherwise, call the ralph_done tool to proceed to the next iteration");
    suffix
}

/// Produces the system prompt that should be supplied for the next model turn.
pub fn inject_system_prompt(base_system_prompt: &str, state: &LoopState) -> String {
    format!("{base_system_prompt}{}", system_prompt_suffix(state))
}

/// Tests an assistant message's text blocks for the exact completion marker.
pub fn has_complete_marker(message: &AssistantMessage) -> bool {
    message.content.iter().any(|block| {
        matches!(
            block,
            ContentBlock::Text(text) if text.text.contains(COMPLETE_MARKER)
        )
    })
}

/// Tests arbitrary plain text for the exact completion marker.
pub fn contains_complete_marker(text: &str) -> bool {
    text.contains(COMPLETE_MARKER)
}

// ---------------------------------------------------------------------------
// Agent tools
// ---------------------------------------------------------------------------

/// A thread-safe destination for Ralph's follow-up prompts.
///
/// Tools hold this behind an [`Arc`], rather than borrowing a runtime-owned
/// queue, because [`agent::Tool`] executors must be `'static`, `Send`, and
/// `Sync`. [`agent::Agent`] implements this trait directly.
pub trait Queue: Send + Sync {
    /// Enqueues a message for the next agent turn.
    fn follow_up(&self, message: llm::Message);

    /// Reports whether steering or follow-up work is already queued.
    fn has_queued_messages(&self) -> bool;
}

/// An owned, shareable Ralph prompt queue.
pub type SharedQueue = Arc<dyn Queue>;

impl Queue for agent::Agent {
    fn follow_up(&self, message: llm::Message) {
        agent::Agent::follow_up(self, message);
    }

    fn has_queued_messages(&self) -> bool {
        agent::Agent::has_queued_messages(self)
    }
}

impl Queue for agent::WeakFollowUpQueue {
    fn follow_up(&self, message: llm::Message) {
        agent::WeakFollowUpQueue::follow_up(self, message);
    }

    fn has_queued_messages(&self) -> bool {
        agent::WeakFollowUpQueue::has_queued_messages(self)
    }
}

/// Returns the `ralph_start` model-facing agent tool.
///
/// The supplied store is cloned into the executor. Store clones share their
/// current-loop cache, so a `ralph_done` tool built from the same store sees a
/// loop started by this one.
#[must_use]
pub fn ralph_start(store: Store, queue: Option<SharedQueue>) -> agent::Tool {
    let mut tool = agent::Tool::new(
        START_TOOL_NAME,
        "Start Ralph Loop",
        "Start a long-running development loop with pacing and reflection controls. \
         Use when the task needs multiple iterations or paced multi-step execution; \
         avoid it for one-shot fixes. After starting, continue each finished iteration \
         with ralph_done unless the completion marker has been emitted.",
        start_tool_schema(),
        move |_cancellation, _call_id, parameters, _on_update| {
            execute_start_tool(&store, queue.as_ref(), &parameters)
        },
    );
    // Both tools mutate the same persisted loop state and inspect the queue.
    // Force a batch containing either tool to run in call order.
    tool.execution_mode = Some(agent::ToolExecutionMode::Sequential);
    tool
}

/// Returns the `ralph_done` model-facing agent tool.
#[must_use]
pub fn ralph_done(store: Store, queue: Option<SharedQueue>) -> agent::Tool {
    let mut tool = agent::Tool::new(
        DONE_TOOL_NAME,
        "Ralph Iteration Done",
        "Signal that the current Ralph loop iteration is complete and request the next one. \
         Call this after making real progress. Do not call it when there is no active loop, \
         when messages are already queued, or after emitting the completion marker.",
        done_tool_schema(),
        move |_cancellation, _call_id, _parameters, _on_update| {
            execute_done_tool(&store, queue.as_ref())
        },
    );
    tool.execution_mode = Some(agent::ToolExecutionMode::Sequential);
    tool
}

fn start_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Loop name, e.g. refactor-auth"
            },
            "taskContent": {
                "type": "string",
                "description": "Task in markdown with goals and a checklist"
            },
            "maxIterations": {
                "type": "integer",
                "description": "Maximum iterations (default 50)"
            },
            "itemsPerIteration": {
                "type": "integer",
                "description": "Suggest N checklist items per turn (0 = no limit)"
            },
            "reflectEvery": {
                "type": "integer",
                "description": "Reflect every N iterations (0 = never)"
            }
        },
        "required": ["name", "taskContent"]
    })
}

fn done_tool_schema() -> Value {
    json!({"type": "object", "properties": {}})
}

fn execute_start_tool(
    store: &Store,
    queue: Option<&SharedQueue>,
    parameters: &BTreeMap<String, Value>,
) -> std::result::Result<agent::ToolResult, String> {
    let name = required_tool_string(parameters, "name")?;
    let task_content = required_tool_string(parameters, "taskContent")?;
    let options = LoopOptions {
        max_iterations: optional_tool_integer(parameters, "maxIterations")?,
        items_per_iteration: optional_tool_integer(parameters, "itemsPerIteration")?,
        reflect_every: optional_tool_integer(parameters, "reflectEvery")?,
    };

    match store.start(name, task_content, options) {
        Ok(state) => {
            if let Some(queue) = queue {
                queue.follow_up(user_message(build_prompt(&state, task_content, false)));
            }
            Ok(agent::ToolResult::text(format!(
                "Started loop {:?} (max {} iterations). Task file: {}",
                state.name, state.max_iterations, state.task_file
            )))
        }
        // A duplicate start is a no-op the model can recover from, rather
        // than an execution failure that ends its tool batch.
        Err(error @ RalphError::AlreadyActive(_)) => Ok(agent::ToolResult::text(error.to_string())),
        Err(error) => Err(error.to_string()),
    }
}

fn execute_done_tool(
    store: &Store,
    queue: Option<&SharedQueue>,
) -> std::result::Result<agent::ToolResult, String> {
    let Some(mut state) = store.current().map_err(|error| error.to_string())? else {
        return Ok(agent::ToolResult::text(
            "No active Ralph loop owned by this session.",
        ));
    };
    if state.status != LoopStatus::Active {
        return Ok(agent::ToolResult::text(
            "No active Ralph loop owned by this session.",
        ));
    }

    // A queued prompt already drives the next turn. Advancing in this batch
    // would produce two prompts and skip an iteration's work.
    if queue.is_some_and(|queue| queue.has_queued_messages()) {
        return Ok(agent::ToolResult::text(
            "Messages are already queued; skipping ralph_done.",
        ));
    }

    let previous = state.iteration;
    let advance = store
        .advance(&mut state)
        .map_err(|error| error.to_string())?;
    if advance.done {
        return Ok(agent::ToolResult {
            content: vec![ContentBlock::text(format!(
                "Max iterations ({}) reached; the loop has stopped.",
                state.max_iterations
            ))],
            terminate: true,
            ..agent::ToolResult::default()
        });
    }

    let task_content = match store.read_task(&state) {
        Ok(task_content) => task_content,
        Err(error) => {
            store
                .pause(&mut state)
                .map_err(|pause_error| pause_error.to_string())?;
            return Err(format!(
                "cannot read task file {}: {error}",
                state.task_file
            ));
        }
    };
    if let Some(queue) = queue {
        queue.follow_up(user_message(build_prompt(
            &state,
            &task_content,
            advance.reflection,
        )));
    }
    let reflection = if advance.reflection {
        " (reflection checkpoint)"
    } else {
        ""
    };
    Ok(agent::ToolResult::text(format!(
        "Iteration {previous} complete; iteration {} queued{reflection}.",
        state.iteration
    )))
}

fn required_tool_string<'a>(
    parameters: &'a BTreeMap<String, Value>,
    name: &str,
) -> std::result::Result<&'a str, String> {
    match parameters.get(name) {
        Some(Value::String(value)) => Ok(value),
        Some(_) => Err(format!("{name:?} must be a string")),
        None => Err(format!("{name:?} is required")),
    }
}

/// Parses the integer representation accepted by the agent's JSON-schema
/// preparation so direct tool callers get the same validation behavior.
fn optional_tool_integer(
    parameters: &BTreeMap<String, Value>,
    name: &str,
) -> std::result::Result<i64, String> {
    let Some(value) = parameters.get(name) else {
        return Ok(0);
    };
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value).map_err(|_| format!("{name:?} must be an integer"));
    }
    if let Some(value) = value.as_f64()
        && value.is_finite()
        && value.fract() == 0.0
        && (i64::MIN as f64..=i64::MAX as f64).contains(&value)
    {
        return Ok(value as i64);
    }
    if let Some(value) = value.as_str() {
        return value
            .parse::<i64>()
            .map_err(|_| format!("{name:?} must be an integer"));
    }
    Err(format!("{name:?} must be an integer"))
}

fn user_message(text: impl Into<String>) -> llm::Message {
    llm::Message::User(llm::UserMessage::text(text, now_millis()))
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn parse_start(arguments: &[String]) -> Result<RalphCommand> {
    let Some((name, rest)) = arguments.split_first() else {
        return Err(command_usage());
    };
    let name = checked_name(name)?;
    let mut options = LoopOptions::default();
    let mut task_words = Vec::new();
    let mut index = 0;
    let mut positional_only = false;

    while index < rest.len() {
        let argument = &rest[index];
        if positional_only {
            task_words.push(argument.clone());
            index += 1;
            continue;
        }
        if argument == "--" {
            positional_only = true;
            index += 1;
            continue;
        }
        let value = |flag: &str, index: &mut usize| -> Result<i64> {
            *index += 1;
            let value = rest.get(*index).ok_or_else(|| {
                RalphError::InvalidCommand(format!("{flag} needs an integer value"))
            })?;
            value.parse::<i64>().map_err(|_| {
                RalphError::InvalidCommand(format!("{flag} needs an integer value, got {value:?}"))
            })
        };
        match argument.as_str() {
            "--max-iterations" | "-n" => {
                options.max_iterations = value(argument, &mut index)?;
            }
            "--items-per-iteration" => {
                options.items_per_iteration = value(argument, &mut index)?;
            }
            "--reflect-every" => {
                options.reflect_every = value(argument, &mut index)?;
            }
            flag if flag.starts_with('-') => {
                return Err(RalphError::InvalidCommand(format!(
                    "unknown Ralph start option {flag:?}"
                )));
            }
            _ => task_words.push(argument.clone()),
        }
        index += 1;
    }

    let task_content = task_words.join(" ");
    if task_content.trim().is_empty() {
        return Err(RalphError::InvalidCommand(
            "usage: ralph start <name> <task> [--max-iterations N] [--items-per-iteration N] [--reflect-every N]"
                .to_owned(),
        ));
    }
    options.validate()?;
    Ok(RalphCommand::Start {
        name,
        task_content,
        options,
    })
}

fn parse_required_name(arguments: &[String]) -> Result<String> {
    let [name] = arguments else {
        return Err(command_usage());
    };
    checked_name(name)
}

fn command_usage() -> RalphError {
    RalphError::InvalidCommand(
        "usage: ralph start|list|status|resume|stop|archive|delete".to_owned(),
    )
}

fn split_command_line(input: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut token = String::new();
    let mut quote = None;
    let mut escaped = false;
    let mut token_started = false;

    for character in input.trim().chars() {
        if escaped {
            token.push(character);
            token_started = true;
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            token_started = true;
            continue;
        }
        if let Some(quote_character) = quote {
            if character == quote_character {
                quote = None;
            } else {
                token.push(character);
            }
            token_started = true;
            continue;
        }
        match character {
            '\'' | '"' => {
                quote = Some(character);
                token_started = true;
            }
            character if character.is_whitespace() => {
                if token_started {
                    tokens.push(std::mem::take(&mut token));
                    token_started = false;
                }
            }
            _ => {
                token.push(character);
                token_started = true;
            }
        }
    }
    if escaped {
        return Err(RalphError::InvalidCommand(
            "Ralph command ends with an escape".to_owned(),
        ));
    }
    if quote.is_some() {
        return Err(RalphError::InvalidCommand(
            "Ralph command has an unterminated quote".to_owned(),
        ));
    }
    if token_started {
        tokens.push(token);
    }
    Ok(tokens)
}

fn checked_name(name: &str) -> Result<String> {
    let sanitized = sanitize_name(name);
    if sanitized.is_empty() {
        return Err(RalphError::InvalidName(name.to_owned()));
    }
    Ok(sanitized)
}

fn max_suffix(state: &LoopState) -> String {
    if state.max_iterations > 0 {
        format!("/{}", state.max_iterations)
    } else {
        String::new()
    }
}

fn state_file_from_task_file(task_file: &str) -> String {
    task_file
        .strip_suffix(".md")
        .map(|stem| format!("{stem}.state.json"))
        .unwrap_or_else(|| format!("{task_file}.state.json"))
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

fn canonical_or_absolute(path: &Path) -> PathBuf {
    let absolute = absolute_path(path);
    absolute.canonicalize().unwrap_or(absolute)
}

/// FNV-1a is small, deterministic, and sufficient for a directory namespace;
/// it is not used as a security boundary.
fn workspace_key(workspace: &Path) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in workspace.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("workspace-{hash:016x}")
}

fn now_timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_owned())
}

fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(RalphError::UnsafePath(path.to_path_buf()));
    }
    if metadata.len() > MAX_RALPH_FILE_BYTES as u64 {
        return Err(RalphError::FileTooLarge {
            path: path.to_path_buf(),
            max_bytes: MAX_RALPH_FILE_BYTES,
        });
    }
    let mut file = File::open(path)?;
    let mut content = Vec::with_capacity(metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(MAX_RALPH_FILE_BYTES as u64 + 1)
        .read_to_end(&mut content)?;
    if content.len() > MAX_RALPH_FILE_BYTES {
        return Err(RalphError::FileTooLarge {
            path: path.to_path_buf(),
            max_bytes: MAX_RALPH_FILE_BYTES,
        });
    }
    Ok(content)
}

fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    if content.len() > MAX_RALPH_FILE_BYTES {
        return Err(RalphError::FileTooLarge {
            path: path.to_path_buf(),
            max_bytes: MAX_RALPH_FILE_BYTES,
        });
    }
    let parent = path.parent().ok_or_else(|| {
        RalphError::InvalidState(format!("{} has no parent directory", path.display()))
    })?;
    let parent_metadata = fs::symlink_metadata(parent)?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(RalphError::UnsafePath(parent.to_path_buf()));
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() || metadata.is_dir() => {
            return Err(RalphError::UnsafePath(path.to_path_buf()));
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let file_name = path
        .file_name()
        .ok_or_else(|| RalphError::InvalidState(format!("{} has no file name", path.display())))?;
    for _ in 0..32 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or_default();
        let temporary = parent.join(format!(
            ".{}.{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            process::id(),
            nanos,
            sequence
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = match options.open(&temporary) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };
        let write_result = (|| -> Result<()> {
            file.write_all(content)?;
            file.sync_all()?;
            drop(file);
            fs::rename(&temporary, path)?;
            #[cfg(unix)]
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
            sync_directory(parent)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        return write_result;
    }
    Err(RalphError::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a temporary Ralph file",
    )))
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<()> {
    File::open(directory)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_: &Path) -> Result<()> {
    Ok(())
}

fn remove_file_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        fs,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    struct TestRoot(PathBuf);

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn test_store(label: &str) -> (Store, TestRoot, PathBuf, PathBuf) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock is after epoch")
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("goshcoder-ralph-{label}-{}-{nonce}", process::id()));
        let config = root.join("agent");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).expect("create test workspace");
        (
            Store::new(&config, &workspace, "session-1"),
            TestRoot(root),
            config,
            workspace,
        )
    }

    fn text_assistant(text: &str) -> AssistantMessage {
        AssistantMessage {
            content: vec![ContentBlock::text(text)],
            ..AssistantMessage::default()
        }
    }

    #[derive(Default)]
    struct TestQueue {
        messages: Mutex<Vec<llm::Message>>,
        pending: AtomicBool,
    }

    impl Queue for TestQueue {
        fn follow_up(&self, message: llm::Message) {
            lock(&self.messages).push(message);
        }

        fn has_queued_messages(&self) -> bool {
            self.pending.load(Ordering::Acquire)
        }
    }

    impl TestQueue {
        fn prompts(&self) -> Vec<String> {
            lock(&self.messages)
                .iter()
                .filter_map(|message| match message {
                    llm::Message::User(message) => message.content.text().map(ToOwned::to_owned),
                    _ => None,
                })
                .collect()
        }

        fn set_pending(&self, pending: bool) {
            self.pending.store(pending, Ordering::Release);
        }
    }

    fn execute_agent_tool(
        tool: agent::Tool,
        parameters: BTreeMap<String, Value>,
    ) -> std::result::Result<agent::ToolResult, String> {
        (tool.execute)(
            agent::CancellationToken::default(),
            "test-call".to_owned(),
            parameters,
            Arc::new(|_| {}),
        )
    }

    fn agent_tool_text(result: &agent::ToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(ContentBlock::plain_text)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn sanitizes_names_and_validates_options() {
        assert_eq!(sanitize_name("refactor-auth"), "refactor-auth");
        assert_eq!(sanitize_name("my loop"), "my_loop");
        assert_eq!(sanitize_name("a//b\\c"), "a_b_c");
        assert_eq!(sanitize_name("lots   of   spaces"), "lots_of_spaces");
        assert_eq!(sanitize_name("///"), "_");

        let (store, _root, _config, _workspace) = test_store("validation");
        assert!(matches!(
            store.start("", "task", LoopOptions::default()),
            Err(RalphError::InvalidName(_))
        ));
        assert!(matches!(
            store.start(
                "loop",
                "task",
                LoopOptions {
                    items_per_iteration: -1,
                    ..LoopOptions::default()
                }
            ),
            Err(RalphError::InvalidOptions(_))
        ));
    }

    #[test]
    fn start_persists_only_below_agent_config_with_pi_style_fields() {
        let (store, _root, config, workspace) = test_store("start");
        let state = store
            .start(
                "my loop",
                "# Task\n\n- [ ] one",
                LoopOptions {
                    max_iterations: 5,
                    ..LoopOptions::default()
                },
            )
            .expect("start");

        assert_eq!(state.name, "my_loop");
        assert_eq!(state.status, LoopStatus::Active);
        assert_eq!(state.iteration, 1);
        assert_eq!(state.summary(), "my_loop: ▶ active (iteration 1/5)");
        assert!(
            store
                .state_path("my_loop", false)
                .unwrap()
                .starts_with(&config)
        );
        assert!(
            !store
                .state_path("my_loop", false)
                .unwrap()
                .starts_with(&workspace)
        );
        assert_eq!(
            fs::read_to_string(store.task_path("my_loop", false).unwrap()).expect("task"),
            "# Task\n\n- [ ] one"
        );

        let raw = fs::read(store.state_path("my_loop", false).unwrap()).expect("state");
        let json: serde_json::Value = serde_json::from_slice(&raw).expect("JSON state");
        for field in [
            "name",
            "taskFile",
            "iteration",
            "maxIterations",
            "itemsPerIteration",
            "reflectEvery",
            "status",
            "startedAt",
            "active",
        ] {
            assert!(json.get(field).is_some(), "missing {field}: {json}");
        }
        assert_eq!(json["status"], "active");
        assert_eq!(json["active"], true);
        assert!(state.task_file.starts_with("ralph/workspace-"));
    }

    #[test]
    fn default_template_and_legacy_active_state_work() {
        let (store, _root, _config, _workspace) = test_store("legacy");
        let state = store
            .start("template", "", LoopOptions::default())
            .expect("start template");
        assert!(
            store
                .read_task(&state)
                .expect("read template")
                .contains("## Checklist")
        );

        let path = store.state_path("legacy", false).expect("legacy path");
        fs::write(
            &path,
            format!(
                r#"{{"name":"legacy","taskFile":"{}","iteration":3,"maxIterations":10,"active":true}}"#,
                store.task_file("legacy", false).expect("task file")
            ),
        )
        .expect("write legacy state");
        let legacy = store
            .load("legacy", false)
            .expect("load legacy")
            .expect("legacy exists");
        assert_eq!(legacy.status, LoopStatus::Active);
        assert!(legacy.active);
        assert_eq!(legacy.reflect_instructions, DEFAULT_REFLECT_INSTRUCTIONS);
    }

    #[test]
    fn start_protects_paused_work_and_reuses_completed_name() {
        let (store, _root, _config, _workspace) = test_store("duplicate");
        let mut state = store
            .start("refactor", "the original task", LoopOptions::default())
            .expect("start");
        state.iteration = 7;
        store.pause(&mut state).expect("pause");

        let error = store
            .start("refactor", "a different task", LoopOptions::default())
            .expect_err("paused loop must be retained");
        assert!(matches!(error, RalphError::PausedLoop { iteration: 7, .. }));
        assert!(
            store
                .read_task(&state)
                .expect("original task")
                .contains("original")
        );

        store.complete(&mut state).expect("complete");
        let restarted = store
            .start("refactor", "second run", LoopOptions::default())
            .expect("completed name is reusable");
        assert_eq!(restarted.iteration, 1);
        assert_eq!(store.read_task(&restarted).expect("task"), "second run");
    }

    #[test]
    fn advances_reflects_completes_and_allows_unbounded_loops() {
        let (store, _root, _config, _workspace) = test_store("advance");
        let mut state = store
            .start(
                "loop",
                "task",
                LoopOptions {
                    max_iterations: 3,
                    reflect_every: 2,
                    ..LoopOptions::default()
                },
            )
            .expect("start");

        assert_eq!(
            store.advance(&mut state).expect("iteration two"),
            Advance {
                done: false,
                reflection: false
            }
        );
        assert_eq!(
            store.advance(&mut state).expect("iteration three"),
            Advance {
                done: false,
                reflection: true
            }
        );
        assert_eq!(state.last_reflection_at, 3);
        assert!(store.advance(&mut state).expect("max exceeded").done);
        assert_eq!(state.status, LoopStatus::Completed);
        assert!(!state.completed_at.is_empty());

        let mut unbounded = store
            .start(
                "unbounded",
                "task",
                LoopOptions {
                    max_iterations: -1,
                    ..LoopOptions::default()
                },
            )
            .expect("unbounded start");
        for _ in 0..5 {
            assert!(!store.advance(&mut unbounded).expect("advance").done);
        }
    }

    #[test]
    fn session_ownership_pause_resume_and_status_are_scoped() {
        let (owner, _root, config, workspace) = test_store("ownership");
        let mut state = owner
            .start("loop", "task", LoopOptions::default())
            .expect("start");
        let other = Store::new(config, workspace, "session-2");
        assert!(other.current().expect("other current").is_none());
        assert!(owner.current().expect("owner current").is_some());

        owner.pause(&mut state).expect("pause");
        assert!(owner.current().expect("paused current").is_none());
        let resumed = owner.resume("loop").expect("resume");
        assert_eq!(resumed.status, LoopStatus::Active);
        assert_eq!(resumed.owner_session_id, "session-1");
        assert!(owner.status_line().expect("status line").contains("active"));
    }

    #[test]
    fn list_archive_and_delete_preserve_expected_files() {
        let (store, _root, _config, _workspace) = test_store("archive");
        for name in ["b-loop", "a-loop"] {
            let mut state = store
                .start(name, "task", LoopOptions::default())
                .expect("start");
            store.pause(&mut state).expect("pause");
        }
        let names = store
            .list(false)
            .expect("list")
            .into_iter()
            .map(|state| state.name)
            .collect::<Vec<_>>();
        assert_eq!(names, ["a-loop", "b-loop"]);

        store.archive("a-loop").expect("archive");
        assert!(store.load("a-loop", false).expect("load").is_none());
        let archived = store
            .load("a-loop", true)
            .expect("archived load")
            .expect("archive exists");
        assert_eq!(store.read_task(&archived).expect("archived task"), "task");
        assert!(store.task_path("a-loop", true).unwrap().exists());

        store.delete("b-loop").expect("delete");
        assert!(store.load("b-loop", false).expect("load").is_none());
        assert!(!store.task_path("b-loop", false).unwrap().exists());
    }

    #[test]
    fn prompts_system_injection_and_completion_marker_drive_turn_state() {
        let (store, _root, _config, _workspace) = test_store("prompts");
        let mut state = store
            .start(
                "loop",
                "# Task\n- [ ] one",
                LoopOptions {
                    max_iterations: 4,
                    items_per_iteration: 2,
                    ..LoopOptions::default()
                },
            )
            .expect("start");
        let prompt = build_prompt(&state, "# Task\n- [ ] one", false);
        for expected in [
            "RALPH LOOP: loop | Iteration 1/4",
            "## Current Task (from ",
            "COMPLETION GATE",
            "STALE PROMPT GUARD",
            "process approximately 2 items",
            COMPLETE_MARKER,
            "ralph_done",
        ] {
            assert!(prompt.contains(expected), "prompt omitted {expected:?}");
        }
        assert!(!prompt.contains("REFLECTION CHECKPOINT"));
        assert!(build_prompt(&state, "task", true).contains("REFLECTION CHECKPOINT"));

        let prepared = store
            .prepare_next_turn("base instructions", &text_assistant("still working"))
            .expect("prepare turn");
        assert!(!prepared.completed);
        assert!(prepared.system_prompt.starts_with("base instructions"));
        assert!(
            prepared
                .system_prompt
                .contains("[RALPH LOOP - loop - Iteration 1/4]")
        );
        assert!(prepared.system_prompt.contains("ralph_done"));

        let completed = store
            .prepare_next_turn(
                "base instructions",
                &text_assistant(&format!("done {COMPLETE_MARKER}")),
            )
            .expect("complete turn");
        assert!(completed.completed);
        assert_eq!(completed.system_prompt, "base instructions");
        assert_eq!(state.status, LoopStatus::Active);
        state = store.load("loop", false).expect("load").expect("loop");
        assert_eq!(state.status, LoopStatus::Completed);
        assert!(has_complete_marker(&text_assistant(COMPLETE_MARKER)));
        assert!(!has_complete_marker(&text_assistant("not yet")));
    }

    #[test]
    fn agent_queue_forwards_follow_ups_to_agent() {
        let agent = agent::Agent::new(agent::AgentOptions::default());
        let queue: SharedQueue = Arc::new(agent.clone());

        queue.follow_up(user_message("queued Ralph work"));

        assert!(queue.has_queued_messages());
        assert_eq!(agent.queued_message_count(), 1);
    }

    #[test]
    fn ralph_start_tool_validates_parameters_queues_prompt_and_is_sequential() {
        let (store, _root, _config, _workspace) = test_store("start-tool");
        let queue = Arc::new(TestQueue::default());
        let queue_handle: SharedQueue = queue.clone();
        let tools = store.tools(Some(queue_handle));

        assert_eq!(
            tools
                .iter()
                .map(|tool| tool.name.as_str())
                .collect::<Vec<_>>(),
            [START_TOOL_NAME, DONE_TOOL_NAME]
        );
        assert!(tools.iter().all(|tool| {
            tool.execution_mode == Some(agent::ToolExecutionMode::Sequential)
                && tool.parameters.get("type") == Some(&Value::String("object".to_owned()))
        }));

        let start = tools
            .iter()
            .find(|tool| tool.name == START_TOOL_NAME)
            .expect("start tool")
            .clone();
        assert!(
            execute_agent_tool(start.clone(), BTreeMap::new())
                .expect_err("name and task are required")
                .contains("name")
        );
        assert!(
            execute_agent_tool(
                start.clone(),
                BTreeMap::from([
                    ("name".to_owned(), Value::String("loop".to_owned())),
                    ("taskContent".to_owned(), Value::String("task".to_owned())),
                    ("maxIterations".to_owned(), json!(1.5)),
                ]),
            )
            .expect_err("fractional iteration limit must be rejected")
            .contains("maxIterations")
        );
        assert!(
            execute_agent_tool(
                start.clone(),
                BTreeMap::from([
                    ("name".to_owned(), Value::String("loop".to_owned())),
                    ("taskContent".to_owned(), Value::String("task".to_owned())),
                    ("reflectEvery".to_owned(), json!(-1)),
                ]),
            )
            .expect_err("negative reflection interval must be rejected")
            .contains("reflect_every")
        );

        let result = execute_agent_tool(
            start.clone(),
            BTreeMap::from([
                ("name".to_owned(), Value::String("refactor auth".to_owned())),
                (
                    "taskContent".to_owned(),
                    Value::String("# Task\n- [ ] split handler".to_owned()),
                ),
                // Direct execution accepts the same integer string that the
                // agent's schema preparation coerces before invoking a tool.
                ("maxIterations".to_owned(), Value::String("4".to_owned())),
            ]),
        )
        .expect("start tool");
        assert_eq!(
            agent_tool_text(&result),
            format!(
                "Started loop \"refactor_auth\" (max 4 iterations). Task file: {}",
                store.task_file("refactor_auth", false).expect("task file")
            )
        );
        assert_eq!(queue.prompts().len(), 1);
        assert!(queue.prompts()[0].contains("RALPH LOOP: refactor_auth | Iteration 1/4"));
        assert_eq!(
            store
                .load("refactor_auth", false)
                .expect("load")
                .expect("state")
                .status,
            LoopStatus::Active
        );

        let duplicate = execute_agent_tool(
            start,
            BTreeMap::from([
                ("name".to_owned(), Value::String("refactor auth".to_owned())),
                ("taskContent".to_owned(), Value::String("task".to_owned())),
            ]),
        )
        .expect("duplicate start is a user-visible no-op");
        assert!(agent_tool_text(&duplicate).contains("already active"));
        assert_eq!(queue.prompts().len(), 1);
    }

    #[test]
    fn ralph_done_tool_skips_queued_work_then_advances_and_queues() {
        let (store, _root, _config, _workspace) = test_store("done-tool");
        let queue = Arc::new(TestQueue::default());
        let queue_handle: SharedQueue = queue.clone();
        let start = ralph_start(store.clone(), Some(queue_handle.clone()));
        let done = ralph_done(store.clone(), Some(queue_handle));
        execute_agent_tool(
            start,
            BTreeMap::from([
                ("name".to_owned(), Value::String("loop".to_owned())),
                (
                    "taskContent".to_owned(),
                    Value::String("# Task\n- [ ] one".to_owned()),
                ),
                ("maxIterations".to_owned(), json!(5)),
                ("reflectEvery".to_owned(), json!(1)),
            ]),
        )
        .expect("start");

        queue.set_pending(true);
        let skipped = execute_agent_tool(done.clone(), BTreeMap::new()).expect("queued work");
        assert_eq!(
            agent_tool_text(&skipped),
            "Messages are already queued; skipping ralph_done."
        );
        assert_eq!(
            store
                .load("loop", false)
                .expect("load")
                .expect("state")
                .iteration,
            1
        );

        queue.set_pending(false);
        let advanced = execute_agent_tool(done, BTreeMap::new()).expect("advance");
        assert_eq!(
            agent_tool_text(&advanced),
            "Iteration 1 complete; iteration 2 queued (reflection checkpoint)."
        );
        let prompts = queue.prompts();
        assert_eq!(prompts.len(), 2);
        assert!(prompts[1].contains("RALPH LOOP: loop | Iteration 2/5 | REFLECTION"));
        assert!(prompts[1].contains("REFLECTION CHECKPOINT"));
    }

    #[test]
    fn ralph_done_tool_handles_absent_maxed_and_unreadable_loops() {
        let (store, _root, _config, _workspace) = test_store("done-terminal");
        let no_active = execute_agent_tool(ralph_done(store.clone(), None), BTreeMap::new())
            .expect("no active loop is a normal result");
        assert_eq!(
            agent_tool_text(&no_active),
            "No active Ralph loop owned by this session."
        );

        execute_agent_tool(
            ralph_start(store.clone(), None),
            BTreeMap::from([
                ("name".to_owned(), Value::String("maxed".to_owned())),
                ("taskContent".to_owned(), Value::String("task".to_owned())),
                ("maxIterations".to_owned(), json!(1)),
            ]),
        )
        .expect("start maxed loop");
        let terminal = execute_agent_tool(ralph_done(store.clone(), None), BTreeMap::new())
            .expect("max iteration result");
        assert!(terminal.terminate);
        assert_eq!(
            agent_tool_text(&terminal),
            "Max iterations (1) reached; the loop has stopped."
        );
        assert_eq!(
            store
                .load("maxed", false)
                .expect("load")
                .expect("state")
                .status,
            LoopStatus::Completed
        );

        let missing = store
            .start("missing-task", "task", LoopOptions::default())
            .expect("start missing-task loop");
        fs::remove_file(store.task_path(&missing.name, false).expect("task path"))
            .expect("remove task");
        let error = execute_agent_tool(ralph_done(store.clone(), None), BTreeMap::new())
            .expect_err("missing task must pause rather than spin");
        assert!(error.contains("cannot read task file"));
        assert_eq!(
            store
                .load(&missing.name, false)
                .expect("load")
                .expect("state")
                .status,
            LoopStatus::Paused
        );
    }

    #[test]
    fn advancing_to_a_missing_task_pauses_instead_of_spinning() {
        let (store, _root, _config, _workspace) = test_store("missing-task");
        let mut state = store
            .start("loop", "task", LoopOptions::default())
            .expect("start");
        fs::remove_file(store.task_path("loop", false).expect("task path")).expect("remove task");
        assert!(store.advance_to_next_prompt(&mut state).is_err());
        assert_eq!(state.status, LoopStatus::Paused);
        assert_eq!(
            store
                .load("loop", false)
                .expect("load")
                .expect("loop")
                .status,
            LoopStatus::Paused
        );
    }

    #[test]
    fn parser_accepts_cli_and_slash_forms_and_rejects_bad_input() {
        let command = parse_command(
            r##"/ralph start "refactor auth" "# Task split handler" --max-iterations 3 --items-per-iteration 2 --reflect-every 1"##,
        )
        .expect("parse slash command");
        assert_eq!(
            command,
            RalphCommand::Start {
                name: "refactor_auth".to_owned(),
                task_content: "# Task split handler".to_owned(),
                options: LoopOptions {
                    max_iterations: 3,
                    items_per_iteration: 2,
                    reflect_every: 1,
                },
            }
        );
        assert_eq!(
            parse_command_args(&["list".to_owned(), "--archived".to_owned()]).expect("list"),
            RalphCommand::List { archived: true }
        );
        assert_eq!(
            parse_command("ralph stop loop").expect("stop"),
            RalphCommand::Stop {
                name: Some("loop".to_owned())
            }
        );
        assert!(parse_command("/ralph start loop").is_err());
        assert!(parse_command(r#"/ralph start loop """#).is_err());
        assert!(parse_command("/ralph list --nope").is_err());
        assert!(parse_command("/ralph start loop task --reflect-every -1").is_err());
        assert!(parse_command("/ralph \"unterminated").is_err());
    }

    #[test]
    fn parsed_commands_execute_lifecycle_without_a_tui() {
        let (store, _root, _config, _workspace) = test_store("command-execution");
        let started = store
            .execute(parse_command("ralph start loop task --max-iterations 2").expect("parse"))
            .expect("execute");
        assert!(matches!(started, CommandResult::Started(_)));
        assert!(matches!(
            store
                .execute(parse_command("ralph status").expect("parse"))
                .expect("status"),
            CommandResult::Status(Some(_))
        ));
        assert!(matches!(
            store
                .execute(parse_command("ralph stop").expect("parse"))
                .expect("stop"),
            CommandResult::Stopped(state) if state.status == LoopStatus::Completed
        ));
        assert!(matches!(
            store
                .execute(parse_command("ralph archive loop").expect("parse"))
                .expect("archive"),
            CommandResult::Archived(name) if name == "loop"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn task_symlinks_are_refused_and_atomic_writes_leave_no_temp_files() {
        let (store, _root, _config, _workspace) = test_store("safe-write");
        let state = store
            .start("loop", "safe task", LoopOptions::default())
            .expect("start");
        let task_path = store.task_path("loop", false).expect("task path");
        let outside =
            std::env::temp_dir().join(format!("goshcoder-ralph-outside-{}", process::id()));
        fs::write(&outside, "outside").expect("outside");
        fs::remove_file(&task_path).expect("remove task");
        std::os::unix::fs::symlink(&outside, &task_path).expect("link task");
        assert!(matches!(
            store.write_task(&state, "must not escape"),
            Err(RalphError::UnsafePath(_))
        ));
        assert_eq!(
            fs::read_to_string(&outside).expect("outside remains"),
            "outside"
        );
        fs::remove_file(&outside).expect("cleanup outside");

        fs::remove_file(&task_path).expect("remove symlink");
        store
            .write_task(&state, "replaced atomically")
            .expect("write task");
        assert!(matches!(
            store.write_task(&state, &"x".repeat(MAX_RALPH_FILE_BYTES + 1)),
            Err(RalphError::FileTooLarge { .. })
        ));
        let temp_files = fs::read_dir(store.storage_dir())
            .expect("storage")
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(temp_files, 0);
    }
}
