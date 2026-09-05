//! Planner and review primitives for the Ratatui runtime.
//!
//! This module is intentionally independent of the executable's command
//! dispatch.  A future runtime integration can register [`Manager::tool`],
//! persist [`State`] through [`crate::session::SessionRuntime::record_custom`],
//! route [`Manager::before_tool_call`] before workspace writes, and invoke
//! [`Manager::prepare_next_turn`] after assistant messages.  Keeping those
//! joins outside this module makes the state machine testable and prevents a
//! planner state from being accidentally shared by separate sessions.
//!
//! The browser reviewer is similarly an adapter behind [`Reviewer`].  It
//! binds only a loopback listener, requires a one-time decision token, checks
//! the request Host header, and has no dependency on a particular terminal
//! frontend.  Ratatui callers can instead render [`ReviewDocument`] and pass a
//! decision to the same workflow.

use std::{
    collections::BTreeMap,
    fmt::{self, Write as FmtWrite},
    fs::{self, File},
    io::{self, Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream},
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, MutexGuard},
    thread,
    time::Duration,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use url::{Host, Url};
use uuid::Uuid;

use crate::{agent, llm};

/// The model-visible name of the plan approval tool.
pub const SUBMIT_TOOL_NAME: &str = "planner_submit_plan";
/// Largest Markdown plan that can be approved.
pub const MAX_PLAN_BYTES: usize = 2 * 1024 * 1024;
/// Largest document assembled for annotation.
pub const MAX_ANNOTATION_BYTES: usize = 2 * 1024 * 1024;
/// Largest diff accepted by a code-review request.
pub const MAX_REVIEW_DIFF_BYTES: usize = 4 * 1024 * 1024;
/// Maximum request headers accepted by the local review server.
pub const MAX_REVIEW_HEADER_BYTES: usize = 16 * 1024;
/// Maximum form or JSON decision body accepted by the local review server.
pub const MAX_REVIEW_DECISION_BYTES: usize = 2 * 1024 * 1024;

/// System-prompt suffix used while the planner is collecting a plan.
pub const PLANNING_PROMPT: &str = r#"

[PLANNER - PLANNING PHASE]
You are in plan mode. Do not modify the codebase, commit, install dependencies, or run destructive commands. You may only write or edit markdown plan files (.md or .mdx) inside the workspace.

Explore the codebase with read-only tools. Build a concise plan containing Context, Approach, Files to modify, Reuse, implementation checklist items using "- [ ]", and Verification. Ask the user only about ambiguities that cannot be answered from the code. When ready, call planner_submit_plan with the plan file path. If review denies it, make targeted edits to the same file and resubmit."#;

/// Planner phase persisted as a custom session payload.
///
/// [`Phase::Unknown`] is deliberate: old or newer session files must be
/// deserializable so [`Manager::new`] can recover to idle with a warning rather
/// than making a whole session impossible to open.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum Phase {
    #[default]
    Idle,
    Planning,
    Executing,
    Unknown(String),
}

impl Phase {
    /// Returns the stable serialized spelling of this phase.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Idle => "idle",
            Self::Planning => "planning",
            Self::Executing => "executing",
            Self::Unknown(value) => value,
        }
    }

    fn is_known(&self) -> bool {
        !matches!(self, Self::Unknown(_))
    }
}

impl Serialize for Phase {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Phase {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(match value.as_str() {
            "idle" => Self::Idle,
            "planning" => Self::Planning,
            "executing" => Self::Executing,
            _ => Self::Unknown(value),
        })
    }
}

/// One positional checklist task extracted from an approved plan.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ChecklistItem {
    pub step: usize,
    pub text: String,
    #[serde(default)]
    pub completed: bool,
}

/// Durable planner state.  Store it in a session's `custom` slot, not in a
/// workspace-global file, so multiple sessions in one repository stay
/// independent.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct State {
    #[serde(default)]
    pub phase: Phase,
    #[serde(rename = "planPath", default, skip_serializing_if = "String::is_empty")]
    pub plan_path: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<ChecklistItem>,
    /// SHA-256 of the source plan used to derive `items`.
    #[serde(rename = "planHash", default, skip_serializing_if = "String::is_empty")]
    pub plan_hash: String,
}

/// Callback invoked after a state transition that must be persisted.
pub type StateCallback = Arc<dyn Fn(State) + Send + Sync + 'static>;
/// Callback used for recoverable restored-state failures.
pub type WarningCallback = Arc<dyn Fn(String) + Send + Sync + 'static>;

/// Construction options for [`Manager`].
#[derive(Default)]
pub struct Options {
    /// State restored from the current session, if any.
    pub initial: Option<State>,
    /// Receives every normal phase or checklist mutation.
    pub on_change: Option<StateCallback>,
    /// Receives a corrupt phase or an unreadable changed plan warning.
    pub warn: Option<WarningCallback>,
}

/// Whether normal tools, planning-only tools, or implementation tools should
/// be registered for the current phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolAccess {
    Idle,
    Planning,
    Executing,
}

/// A decision returned by a human reviewer.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Decision {
    pub approved: bool,
    pub feedback: String,
}

/// Content passed to a human reviewer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewRequest {
    pub title: String,
    pub markdown: String,
}

impl ReviewRequest {
    #[must_use]
    pub fn new(title: impl Into<String>, markdown: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            markdown: markdown.into(),
        }
    }
}

/// A recoverable cancellation or a non-recoverable review failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReviewError {
    Cancelled,
    Failed(String),
}

impl fmt::Display for ReviewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("review cancelled"),
            Self::Failed(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for ReviewError {}

/// Frontend-independent human-review interface.
///
/// Implementations that support resubmission history should override
/// [`Self::review_version`].  The default preserves compatibility with simple
/// terminal dialogs and test reviewers.
pub trait Reviewer: Send + Sync {
    fn review(
        &self,
        cancellation: &agent::CancellationToken,
        request: &ReviewRequest,
    ) -> Result<Decision, ReviewError>;

    fn review_version(
        &self,
        cancellation: &agent::CancellationToken,
        request: &ReviewRequest,
        _previous: &str,
    ) -> Result<Decision, ReviewError> {
        self.review(cancellation, request)
    }
}

/// Errors from planner construction, safe plan reads, or an infrastructure
/// review failure.  Validation failures in [`Manager::submit`] intentionally
/// become model-facing [`SubmitResult`] text, as they did in the Go tool.
#[derive(Debug)]
pub enum PlannerError {
    InvalidWorkspace {
        path: PathBuf,
        reason: String,
    },
    InvalidPlanPath,
    NotRegularPlan {
        path: PathBuf,
    },
    PlanTooLarge {
        limit: usize,
    },
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Review(ReviewError),
}

impl fmt::Display for PlannerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorkspace { path, reason } => {
                write!(
                    formatter,
                    "workspace {} is invalid: {reason}",
                    path.display()
                )
            }
            Self::InvalidPlanPath => formatter
                .write_str("plan file must be a markdown file (.md or .mdx) inside the workspace"),
            Self::NotRegularPlan { .. } => formatter.write_str("not a regular file"),
            Self::PlanTooLarge { limit } => write!(formatter, "plan exceeds {limit} bytes"),
            Self::Io {
                action,
                path,
                source,
            } => write!(formatter, "{action} {}: {source}", path.display()),
            Self::Review(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PlannerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Review(error) => Some(error),
            _ => None,
        }
    }
}

pub type PlannerResult<T> = Result<T, PlannerError>;

/// Model-facing result of a submit attempt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubmitResult {
    pub text: String,
    pub approved: bool,
}

impl SubmitResult {
    fn message(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            approved: false,
        }
    }

    fn approved(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            approved: true,
        }
    }

    fn into_agent_result(self) -> agent::ToolResult {
        let mut result = agent::ToolResult::text(self.text);
        if self.approved {
            result.details = Some(json!({"approved": true}));
        }
        result
    }
}

/// Result of applying an assistant message before the next model turn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannerTurnUpdate {
    pub system_prompt: String,
    pub completed_steps: usize,
    pub phase: Phase,
    pub tool_access: ToolAccess,
}

/// The result of the planner's pre-tool write gate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolGateResult {
    pub block: bool,
    pub reason: String,
}

#[derive(Clone)]
pub struct Manager {
    inner: Arc<Mutex<ManagerInner>>,
}

struct ManagerInner {
    root: PathBuf,
    state: State,
    reviewer: Option<Arc<dyn Reviewer>>,
    on_change: Option<StateCallback>,
    warn: Option<WarningCallback>,
    last_reviewed_plan: String,
}

impl Manager {
    /// Creates one session-owned planner manager.
    ///
    /// The root must be an existing directory so the plan reader can use a
    /// canonical confinement root.  This matches the existing Rust workspace
    /// helpers and avoids making a lexical path check appear race-free.
    pub fn new(
        root: impl AsRef<Path>,
        reviewer: Option<Arc<dyn Reviewer>>,
        options: Options,
    ) -> PlannerResult<Self> {
        let requested = root.as_ref();
        let canonical =
            fs::canonicalize(requested).map_err(|source| PlannerError::InvalidWorkspace {
                path: requested.to_path_buf(),
                reason: source.to_string(),
            })?;
        let metadata =
            fs::metadata(&canonical).map_err(|source| PlannerError::InvalidWorkspace {
                path: canonical.clone(),
                reason: source.to_string(),
            })?;
        if !metadata.is_dir() {
            return Err(PlannerError::InvalidWorkspace {
                path: canonical,
                reason: "not a directory".to_owned(),
            });
        }

        let mut state = options.initial.unwrap_or_default();
        let invalid_phase = (!state.phase.is_known()).then(|| state.phase.as_str().to_owned());
        if invalid_phase.is_some() {
            state = State::default();
        }
        let manager = Self {
            inner: Arc::new(Mutex::new(ManagerInner {
                root: canonical,
                state,
                reviewer,
                on_change: options.on_change,
                warn: options.warn,
                last_reviewed_plan: String::new(),
            })),
        };

        if let Some(phase) = invalid_phase {
            manager.warn(format!(
                "ignoring an unrecognized saved Planner phase {phase:?}; starting idle"
            ));
        }
        let needs_rehydrate = {
            let inner = lock(&manager.inner);
            inner.state.phase == Phase::Executing && !inner.state.plan_path.is_empty()
        };
        if needs_rehydrate {
            manager.rehydrate_checklist();
        }
        Ok(manager)
    }

    /// Replaces the reviewer used by future submissions.
    pub fn set_reviewer(&self, reviewer: Option<Arc<dyn Reviewer>>) {
        lock(&self.inner).reviewer = reviewer;
    }

    /// Returns the canonical workspace root.
    #[must_use]
    pub fn root(&self) -> PathBuf {
        lock(&self.inner).root.clone()
    }

    /// Takes a deep, session-safe snapshot of the planner state.
    #[must_use]
    pub fn state(&self) -> State {
        lock(&self.inner).state.clone()
    }

    /// Starts planning without discarding a previous approved-plan record.
    pub fn enter(&self) {
        self.set_phase(Phase::Planning);
    }

    /// Leaves planner mode without discarding a previous approved-plan record.
    pub fn exit(&self) {
        self.set_phase(Phase::Idle);
    }

    /// Toggles idle/planning.  Executing also toggles to idle, matching the
    /// previous `/planner` command behavior.
    pub fn toggle(&self) -> Phase {
        let (state, callback) = {
            let mut inner = lock(&self.inner);
            inner.state.phase = if inner.state.phase == Phase::Idle {
                Phase::Planning
            } else {
                Phase::Idle
            };
            (inner.state.clone(), inner.on_change.clone())
        };
        publish(callback, state.clone());
        state.phase
    }

    /// Indicates which workspace tool set an integration should register.
    #[must_use]
    pub fn tool_access(&self) -> ToolAccess {
        match self.state().phase {
            Phase::Planning => ToolAccess::Planning,
            Phase::Executing => ToolAccess::Executing,
            Phase::Idle | Phase::Unknown(_) => ToolAccess::Idle,
        }
    }

    /// Appends the current planner instruction to a base system prompt.
    #[must_use]
    pub fn prompt(&self, base: impl AsRef<str>) -> String {
        let state = self.state();
        let suffix = prompt_suffix(&state);
        format!("{}{}", base.as_ref(), suffix)
    }

    /// Applies completion markers in `message`, then builds all values a host
    /// needs to apply before requesting the next model turn.
    #[must_use]
    pub fn prepare_next_turn(
        &self,
        base_prompt: impl AsRef<str>,
        message: &llm::AssistantMessage,
    ) -> PlannerTurnUpdate {
        let completed_steps = self.track_assistant(message);
        let state = self.state();
        PlannerTurnUpdate {
            system_prompt: format!("{}{}", base_prompt.as_ref(), prompt_suffix(&state)),
            completed_steps,
            phase: state.phase.clone(),
            tool_access: match state.phase {
                Phase::Planning => ToolAccess::Planning,
                Phase::Executing => ToolAccess::Executing,
                Phase::Idle | Phase::Unknown(_) => ToolAccess::Idle,
            },
        }
    }

    /// Returns a block only for a tool action forbidden during planning.
    #[must_use]
    pub fn before_tool_call(&self, call: &llm::ToolCall) -> Option<ToolGateResult> {
        if self.state().phase != Phase::Planning {
            return None;
        }
        if call.name == "bash" {
            return Some(ToolGateResult {
                block: true,
                reason:
                    "Planner: shell commands are disabled during planning because they can modify the workspace."
                        .to_owned(),
            });
        }
        if call.name != "write" && call.name != "edit" {
            return None;
        }
        let path = call
            .arguments
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default();
        (!self.is_plan_path_allowed(path)).then(|| ToolGateResult {
            block: true,
            reason: format!(
                "Planner: during planning, writes and edits are limited to markdown files inside the workspace. Blocked: {path}"
            ),
        })
    }

    /// Builds the sequential model-facing submit tool using the existing Rust
    /// agent runtime types.
    #[must_use]
    pub fn tool(&self) -> agent::Tool {
        let manager = self.clone();
        let mut tool = agent::Tool::new(
            SUBMIT_TOOL_NAME,
            "Submit Plan",
            "Submit a markdown plan for human review. Use only in Planner mode after writing the plan inside the workspace. If denied, revise the same file and resubmit.",
            submit_tool_schema(),
            move |cancellation, _, parameters, _| {
                let path = parameters
                    .get("filePath")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                manager
                    .submit(&cancellation, path)
                    .map(SubmitResult::into_agent_result)
                    .map_err(|error| error.to_string())
            },
        );
        tool.execution_mode = Some(agent::ToolExecutionMode::Sequential);
        tool
    }

    /// Validates, reads, and asks for approval of a plan.
    ///
    /// User-correctable problems are returned as a normal tool message so the
    /// model can repair the same plan and resubmit.  An infrastructure error
    /// from the reviewer is returned as [`PlannerError`].
    pub fn submit(
        &self,
        cancellation: &agent::CancellationToken,
        input_path: &str,
    ) -> PlannerResult<SubmitResult> {
        if self.state().phase != Phase::Planning {
            return Ok(SubmitResult::message("Error: Not in Planner mode."));
        }
        if !self.is_plan_path_allowed(input_path) {
            return Ok(SubmitResult::message(
                "Error: plan file must be a markdown file (.md or .mdx) inside the workspace.",
            ));
        }
        let content = match self.read_plan(input_path) {
            Ok(content) => content,
            Err(error) => {
                return Ok(SubmitResult::message(format!(
                    "Error: {input_path} cannot be read as a regular plan file: {error}"
                )));
            }
        };
        if String::from_utf8_lossy(&content).trim().is_empty() {
            return Ok(SubmitResult::message("Error: the plan file is empty."));
        }
        let markdown = String::from_utf8_lossy(&content).into_owned();
        let items = parse_checklist(&markdown);
        if items.is_empty() {
            return Ok(SubmitResult::message(
                "Error: the plan must contain at least one markdown checklist item using '- [ ]'.",
            ));
        }

        let (reviewer, previous) = {
            let inner = lock(&self.inner);
            (inner.reviewer.clone(), inner.last_reviewed_plan.clone())
        };
        let decision = if let Some(reviewer) = reviewer {
            let request = ReviewRequest::new(
                format!("Review plan: {}", input_basename(input_path)),
                markdown.clone(),
            );
            match reviewer.review_version(cancellation, &request, &previous) {
                Ok(decision) => {
                    lock(&self.inner).last_reviewed_plan = markdown.clone();
                    decision
                }
                Err(ReviewError::Cancelled) => {
                    return Ok(SubmitResult::message(
                        "Plan review was cancelled. The plan was not approved; resubmit to review again.",
                    ));
                }
                Err(error) => return Err(PlannerError::Review(error)),
            }
        } else {
            Decision {
                approved: true,
                feedback: String::new(),
            }
        };

        if !decision.approved {
            let feedback = if decision.feedback.trim().is_empty() {
                "Plan rejected. Please revise it.".to_owned()
            } else {
                decision.feedback.trim().to_owned()
            };
            return Ok(SubmitResult::message(format!(
                "The plan was denied. Edit the same plan file with targeted changes, then resubmit it.\n\nUser feedback:\n{feedback}"
            )));
        }

        let (state, callback) = {
            let mut inner = lock(&self.inner);
            inner.state = State {
                phase: Phase::Executing,
                plan_path: portable_input_path(input_path),
                items,
                plan_hash: hash_plan(&content),
            };
            (inner.state.clone(), inner.on_change.clone())
        };
        publish(callback, state);

        let mut message =
            "Plan approved. Begin implementation now using full tool access.".to_owned();
        if !decision.feedback.trim().is_empty() {
            message.push_str("\n\nImplementation notes from the reviewer:\n");
            message.push_str(decision.feedback.trim());
        }
        message
            .push_str("\nAfter completing each checklist step, include [DONE:n] in your response.");
        Ok(SubmitResult::approved(message))
    }

    /// Returns true only for a Markdown plan inside the canonical workspace.
    #[must_use]
    pub fn is_plan_path_allowed(&self, input_path: &str) -> bool {
        let root = self.root();
        relative_plan_path(&root, input_path).is_some()
    }

    /// Reads a plan through the same bounded, no-symlink gate used by submit
    /// and restored-state rehydration.
    pub fn read_plan(&self, input_path: &str) -> PlannerResult<Vec<u8>> {
        let root = self.root();
        let relative =
            relative_plan_path(&root, input_path).ok_or(PlannerError::InvalidPlanPath)?;
        read_plan_file(&root, &relative)
    }

    /// Extracts `[DONE:n]` markers from text and marks matching approved-plan
    /// tasks complete.  Finishing all tasks returns the phase to idle while
    /// retaining the plan record.
    pub fn track_text(&self, text: &str) -> usize {
        let markers = done_markers(text);
        if markers.is_empty() {
            return 0;
        }
        let (changed, state, callback) = {
            let mut inner = lock(&self.inner);
            let mut changed = 0;
            for step in markers {
                for item in &mut inner.state.items {
                    if item.step == step && !item.completed {
                        item.completed = true;
                        changed += 1;
                    }
                }
            }
            let mut finished = false;
            if inner.state.phase == Phase::Executing && !inner.state.items.is_empty() {
                let complete = inner.state.items.iter().all(|item| item.completed);
                if complete {
                    inner.state.phase = Phase::Idle;
                    finished = true;
                }
            }
            let state = (changed > 0 || finished).then(|| inner.state.clone());
            let callback = (changed > 0 || finished)
                .then(|| inner.on_change.clone())
                .flatten();
            (changed, state, callback)
        };
        if let Some(state) = state {
            publish(callback, state);
        }
        changed
    }

    /// Collects text blocks from an existing Rust LLM assistant message and
    /// delegates to [`Self::track_text`].
    pub fn track_assistant(&self, message: &llm::AssistantMessage) -> usize {
        let text = message
            .content
            .iter()
            .filter_map(|block| match block {
                llm::ContentBlock::Text(content) => Some(content.text.as_str()),
                _ => None,
            })
            .collect::<String>();
        self.track_text(&text)
    }

    /// Short status text suitable for a Ratatui sidebar or footer.
    #[must_use]
    pub fn status_line(&self) -> String {
        let state = self.state();
        match state.phase {
            Phase::Planning => "Planner: planning".to_owned(),
            Phase::Executing => {
                let completed = state.items.iter().filter(|item| item.completed).count();
                format!("Planner: executing {completed}/{}", state.items.len())
            }
            Phase::Idle | Phase::Unknown(_) => "Planner: idle".to_owned(),
        }
    }

    fn set_phase(&self, phase: Phase) {
        let (state, callback) = {
            let mut inner = lock(&self.inner);
            inner.state.phase = phase;
            (inner.state.clone(), inner.on_change.clone())
        };
        publish(callback, state);
    }

    fn warn(&self, message: String) {
        let callback = lock(&self.inner).warn.clone();
        if let Some(callback) = callback {
            callback(message);
        }
    }

    // Re-read a saved plan and only merge positional completion when its
    // content hash agrees.  Otherwise a reordered task could accidentally be
    // marked complete merely because it occupies an old step number.
    fn rehydrate_checklist(&self) {
        let (path, expected_hash, old_items) = {
            let inner = lock(&self.inner);
            (
                inner.state.plan_path.clone(),
                inner.state.plan_hash.clone(),
                inner.state.items.clone(),
            )
        };
        let content = match self.read_plan(&path) {
            Ok(content) => content,
            Err(error) => {
                lock(&self.inner).state = State::default();
                self.warn(format!(
                    "the plan {path} can no longer be read ({error}); Planner is starting idle"
                ));
                return;
            }
        };
        let actual_hash = hash_plan(&content);
        let mut fresh = parse_checklist(&String::from_utf8_lossy(&content));
        if !expected_hash.is_empty() && expected_hash != actual_hash {
            {
                let mut inner = lock(&self.inner);
                inner.state.items = fresh;
                inner.state.plan_hash = actual_hash;
            }
            self.warn(format!(
                "{path} changed since this plan was approved; its checklist progress was reset rather than guessed at"
            ));
            return;
        }
        let completed: BTreeMap<usize, bool> = old_items
            .into_iter()
            .map(|item| (item.step, item.completed))
            .collect();
        for item in &mut fresh {
            item.completed |= completed.get(&item.step).copied().unwrap_or(false);
        }
        let mut inner = lock(&self.inner);
        inner.state.items = fresh;
        inner.state.plan_hash = actual_hash;
    }
}

/// JSON schema used by [`Manager::tool`].
#[must_use]
pub fn submit_tool_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "filePath": {
                "type": "string",
                "description": "Markdown plan path relative to the workspace"
            }
        },
        "required": ["filePath"]
    })
}

/// Parses unchecked and checked Markdown checklist lines.  Steps are
/// intentionally positional, matching the original planner behavior.
#[must_use]
pub fn parse_checklist(content: &str) -> Vec<ChecklistItem> {
    let mut items = Vec::new();
    for line in content.replace("\r\n", "\n").split('\n') {
        let Some((completed, text)) = parse_checklist_line(line) else {
            continue;
        };
        items.push(ChecklistItem {
            step: items.len() + 1,
            text: text.to_owned(),
            completed,
        });
    }
    items
}

/// SHA-256 identity of the exact source bytes used to derive a checklist.
#[must_use]
pub fn hash_plan(content: &[u8]) -> String {
    let digest = Sha256::digest(content);
    let mut result = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(&mut result, "{byte:02x}");
    }
    result
}

fn parse_checklist_line(line: &str) -> Option<(bool, &str)> {
    let bytes = line.as_bytes();
    let mut cursor = match bytes.first().copied()? {
        b'-' | b'*' | b'+' => 1,
        byte if byte.is_ascii_digit() => {
            let digits = bytes
                .iter()
                .take_while(|byte| byte.is_ascii_digit())
                .count();
            if digits == bytes.len() || !matches!(bytes[digits], b'.' | b')') {
                return None;
            }
            digits + 1
        }
        _ => return None,
    };
    while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'[') || bytes.get(cursor + 2) != Some(&b']') {
        return None;
    }
    let completed = match bytes.get(cursor + 1).copied() {
        Some(b' ') => false,
        Some(b'x' | b'X') => true,
        _ => return None,
    };
    cursor += 3;
    if !bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
        return None;
    }
    let text = line[cursor..].trim();
    (!text.is_empty()).then_some((completed, text))
}

fn prompt_suffix(state: &State) -> String {
    match state.phase {
        Phase::Planning => PLANNING_PROMPT.to_owned(),
        Phase::Executing => {
            let remaining = state
                .items
                .iter()
                .filter(|item| !item.completed)
                .map(|item| format!("- [ ] {}. {}", item.step, item.text))
                .collect::<Vec<_>>();
            if remaining.is_empty() {
                String::new()
            } else {
                format!(
                    "\n\n[PLANNER - EXECUTING PLAN]\nFull tool access is enabled. Execute the approved plan from {}.\n\nRemaining steps:\n{}\n\nExecute each step in order. After completing a step, include [DONE:n] in your response where n is the step number.",
                    state.plan_path,
                    remaining.join("\n")
                )
            }
        }
        Phase::Idle | Phase::Unknown(_) => String::new(),
    }
}

fn done_markers(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut result = Vec::new();
    let mut cursor = 0;
    while cursor + 7 <= bytes.len() {
        if bytes[cursor] != b'[' || !bytes[cursor..cursor + 6].eq_ignore_ascii_case(b"[DONE:") {
            cursor += 1;
            continue;
        }
        let digits_start = cursor + 6;
        let mut end = digits_start;
        while bytes.get(end).is_some_and(u8::is_ascii_digit) {
            end += 1;
        }
        if end > digits_start && bytes.get(end) == Some(&b']') {
            if let Ok(step) = std::str::from_utf8(&bytes[digits_start..end])
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .ok_or(())
            {
                result.push(step);
            }
            cursor = end + 1;
        } else {
            cursor += 1;
        }
    }
    result
}

fn relative_plan_path(root: &Path, input_path: &str) -> Option<PathBuf> {
    if input_path.is_empty() {
        return None;
    }
    let source = Path::new(input_path);
    let target = if is_rooted_path(input_path) {
        // A Windows-rooted spelling on Unix is not an actual absolute path,
        // but accepting it as a literal filename would make the gate
        // platform-dependent.
        if !source.is_absolute() {
            return None;
        }
        normalize_absolute(source)?
    } else {
        normalize_absolute(&root.join(source))?
    };
    let relative = target.strip_prefix(root).ok()?.to_path_buf();
    if relative.as_os_str().is_empty() {
        return None;
    }
    relative
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            extension.eq_ignore_ascii_case("md") || extension.eq_ignore_ascii_case("mdx")
        })
        .then_some(relative)
}

/// Reports rooted path spellings on all platforms, including a Windows drive
/// spelling supplied to a Unix host.
#[must_use]
pub fn is_rooted_path(path: &str) -> bool {
    let bytes = path.as_bytes();
    Path::new(path).is_absolute()
        || matches!(bytes.first(), Some(b'/' | b'\\'))
        || (bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic())
}

fn normalize_absolute(path: &Path) -> Option<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() && !normalized.has_root() {
                    return None;
                }
            }
            Component::Normal(value) => normalized.push(value),
        }
    }
    normalized.is_absolute().then_some(normalized)
}

fn read_plan_file(root: &Path, relative: &Path) -> PlannerResult<Vec<u8>> {
    let path = checked_regular_plan_path(root, relative)?;
    let mut file = File::open(&path).map_err(|source| PlannerError::Io {
        action: "open plan",
        path: path.clone(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| PlannerError::Io {
        action: "inspect plan",
        path: path.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(PlannerError::NotRegularPlan { path });
    }
    let mut content = Vec::with_capacity(MAX_PLAN_BYTES.min(64 * 1024));
    Read::by_ref(&mut file)
        .take((MAX_PLAN_BYTES + 1) as u64)
        .read_to_end(&mut content)
        .map_err(|source| PlannerError::Io {
            action: "read plan",
            path: path.clone(),
            source,
        })?;
    if content.len() > MAX_PLAN_BYTES {
        return Err(PlannerError::PlanTooLarge {
            limit: MAX_PLAN_BYTES,
        });
    }
    Ok(content)
}

fn checked_regular_plan_path(root: &Path, relative: &Path) -> PlannerResult<PathBuf> {
    let mut current = root.to_path_buf();
    let mut components = relative.components().peekable();
    while let Some(component) = components.next() {
        let Component::Normal(name) = component else {
            return Err(PlannerError::InvalidPlanPath);
        };
        current.push(name);
        let metadata = fs::symlink_metadata(&current).map_err(|source| PlannerError::Io {
            action: "inspect plan",
            path: current.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink()
            || (components.peek().is_some() && !metadata.is_dir())
            || (components.peek().is_none() && !metadata.is_file())
        {
            return Err(PlannerError::NotRegularPlan { path: current });
        }
    }
    Ok(current)
}

fn input_basename(path: &str) -> String {
    path.rsplit(['/', '\\'])
        .next()
        .filter(|value| !value.is_empty())
        .unwrap_or(path)
        .to_owned()
}

fn portable_input_path(path: &str) -> String {
    #[cfg(windows)]
    {
        path.replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        path.to_owned()
    }
}

fn publish(callback: Option<StateCallback>, state: State) {
    if let Some(callback) = callback {
        callback(state);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

/// A line category shared by browser and future Ratatui review surfaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReviewLineKind {
    Paragraph,
    Fence,
    Code,
    DiffAdd,
    DiffRemove,
    DiffHunk,
    Heading(u8),
    Task { completed: bool },
    Bullet,
    Numbered,
    Quote,
    Blank,
}

impl ReviewLineKind {
    fn css_class(self) -> String {
        match self {
            Self::Paragraph => "paragraph".to_owned(),
            Self::Fence => "fence".to_owned(),
            Self::Code => "code".to_owned(),
            Self::DiffAdd => "diff-add".to_owned(),
            Self::DiffRemove => "diff-remove".to_owned(),
            Self::DiffHunk => "diff-hunk".to_owned(),
            Self::Heading(level) => format!("heading-{level}"),
            Self::Task { completed: false } => "task".to_owned(),
            Self::Task { completed: true } => "task-done".to_owned(),
            Self::Bullet => "bullet".to_owned(),
            Self::Numbered => "numbered".to_owned(),
            Self::Quote => "quote".to_owned(),
            Self::Blank => "blank".to_owned(),
        }
    }
}

/// One source line prepared for an interactive review surface.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewLine {
    pub number: usize,
    pub text: String,
    pub display: String,
    pub kind: ReviewLineKind,
    /// Indentation expressed in two-space review levels, capped at eight.
    pub indent: u8,
}

/// A heading used by a review table of contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReviewHeading {
    pub number: usize,
    pub level: u8,
    pub text: String,
}

/// Render-neutral representation of a Markdown document under review.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReviewDocument {
    pub lines: Vec<ReviewLine>,
    pub headings: Vec<ReviewHeading>,
}

/// Classifies Markdown in the same conservative way used by the former
/// browser review page.  In particular, `#include` remains ordinary code-like
/// text instead of becoming a heading.
#[must_use]
pub fn prepare_review_document(markdown: &str) -> ReviewDocument {
    let normalized = markdown.replace("\r\n", "\n");
    let mut document = ReviewDocument::default();
    let mut in_code = false;
    let mut diff_code = false;
    for (index, text) in normalized.split('\n').enumerate() {
        let trimmed = text.trim();
        let mut line = ReviewLine {
            number: index + 1,
            text: text.to_owned(),
            display: text.to_owned(),
            kind: ReviewLineKind::Paragraph,
            indent: review_indent(text),
        };
        if let Some(language) = trimmed.strip_prefix("```") {
            if !in_code {
                in_code = true;
                diff_code = language.trim().eq_ignore_ascii_case("diff");
            } else {
                in_code = false;
                diff_code = false;
            }
            line.kind = ReviewLineKind::Fence;
        } else if in_code {
            line.kind = if diff_code && text.starts_with('+') && !text.starts_with("+++") {
                ReviewLineKind::DiffAdd
            } else if diff_code && text.starts_with('-') && !text.starts_with("---") {
                ReviewLineKind::DiffRemove
            } else if diff_code && text.starts_with("@@") {
                ReviewLineKind::DiffHunk
            } else {
                ReviewLineKind::Code
            };
        } else if let Some((heading, level)) = parse_review_heading(trimmed) {
            line.kind = ReviewLineKind::Heading(level);
            line.display = heading.to_owned();
            document.headings.push(ReviewHeading {
                number: line.number,
                level,
                text: heading.to_owned(),
            });
        } else if let Some((display, completed)) = parse_review_task(trimmed) {
            line.kind = ReviewLineKind::Task { completed };
            line.display = display.to_owned();
        } else if let Some(display) = parse_bullet(trimmed) {
            line.kind = ReviewLineKind::Bullet;
            line.display = display.to_owned();
        } else if let Some((marker, body)) = parse_review_numbered(trimmed) {
            line.kind = ReviewLineKind::Numbered;
            line.display = format!("{marker} {body}");
        } else if let Some(quote) = trimmed.strip_prefix("> ") {
            line.kind = ReviewLineKind::Quote;
            line.display = quote.trim().to_owned();
        } else if trimmed.is_empty() {
            line.kind = ReviewLineKind::Blank;
        }
        document.lines.push(line);
    }
    document
}

fn parse_review_heading(trimmed: &str) -> Option<(&str, u8)> {
    let level = trimmed.bytes().take_while(|byte| *byte == b'#').count();
    if !(1..=6).contains(&level)
        || !trimmed
            .as_bytes()
            .get(level)
            .is_some_and(u8::is_ascii_whitespace)
    {
        return None;
    }
    Some((trimmed[level..].trim(), level as u8))
}

fn parse_review_task(trimmed: &str) -> Option<(&str, bool)> {
    for prefix in ["- [x] ", "- [X] ", "* [x] ", "* [X] ", "+ [x] ", "+ [X] "] {
        if let Some(display) = trimmed.strip_prefix(prefix) {
            return Some((display.trim(), true));
        }
    }
    for prefix in ["- [ ] ", "* [ ] ", "+ [ ] "] {
        if let Some(display) = trimmed.strip_prefix(prefix) {
            return Some((display.trim(), false));
        }
    }
    let (marker, body) = parse_review_numbered(trimmed)?;
    let body = body.strip_prefix('[')?;
    let bytes = body.as_bytes();
    if bytes.get(1).is_none()
        || bytes.get(2) != Some(&b']')
        || !bytes.get(3).is_some_and(u8::is_ascii_whitespace)
    {
        return None;
    }
    let completed = matches!(bytes[1], b'x' | b'X');
    matches!(bytes[1], b' ' | b'x' | b'X')
        .then_some((body[4..].trim(), completed))
        .filter(|(display, _)| !display.is_empty())
        .or_else(|| {
            let _ = marker;
            None
        })
}

fn parse_bullet(trimmed: &str) -> Option<&str> {
    ["- ", "* ", "+ "]
        .into_iter()
        .find_map(|prefix| trimmed.strip_prefix(prefix))
        .map(str::trim)
}

fn parse_review_numbered(trimmed: &str) -> Option<(&str, &str)> {
    let digits = trimmed
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 || digits >= trimmed.len() || !matches!(trimmed.as_bytes()[digits], b'.' | b')')
    {
        return None;
    }
    let after_marker = &trimmed[digits + 1..];
    if !after_marker.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }
    let body = after_marker.trim();
    (!body.is_empty()).then_some((&trimmed[..digits + 1], body))
}

fn review_indent(text: &str) -> u8 {
    let width: usize = text
        .chars()
        .take_while(|character| *character == ' ' || *character == '\t')
        .map(|character| if character == '\t' { 4 } else { 1 })
        .sum();
    (width / 2).min(8) as u8
}

/// A line-specific review comment.  Ratatui and browser frontends can build
/// these independently, then use [`build_annotation_feedback`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Annotation {
    pub line: usize,
    pub quote: String,
    pub comment: String,
}

/// Compiles annotations, overall notes, and a direct Markdown edit into the
/// feedback text sent back to the agent.
#[must_use]
pub fn build_annotation_feedback(
    annotations: &[Annotation],
    overall_notes: &str,
    edited_markdown: Option<&str>,
    original_markdown: &str,
) -> String {
    let mut chunks = Vec::new();
    for annotation in annotations {
        let comment = annotation.comment.trim();
        if comment.is_empty() {
            continue;
        }
        chunks.push(format!(
            "- Line {} (\"{}\"): {}",
            annotation.line,
            truncate_characters(annotation.quote.trim(), 120),
            comment
        ));
    }
    if !overall_notes.trim().is_empty() {
        chunks.push(format!("## Overall notes\n\n{}", overall_notes.trim()));
    }
    if let Some(edited) = edited_markdown.filter(|edited| *edited != original_markdown) {
        chunks.push(format!(
            "## Direct edits\n\nThe reviewer edited the plan directly. Apply this complete revised Markdown:\n\n~~~markdown\n{edited}\n~~~"
        ));
    }
    if chunks.is_empty() {
        String::new()
    } else {
        format!("## Planner review\n\n{}", chunks.join("\n\n"))
    }
}

/// Builds the next user-facing agent prompt from a human review decision.
/// Approval without feedback needs no additional agent turn.
#[must_use]
pub fn review_feedback_prompt(subject: &str, decision: &Decision) -> Option<String> {
    let feedback = decision.feedback.trim();
    if decision.approved && feedback.is_empty() {
        return None;
    }
    let feedback = if feedback.is_empty() {
        format!("{subject} was denied without notes.")
    } else {
        feedback.to_owned()
    };
    let status = if decision.approved {
        "approved with notes"
    } else {
        "denied"
    };
    Some(format!(
        "Planner review of {subject} was {status}. Address this feedback:\n\n{feedback}"
    ))
}

/// Builds a browser/Ratatui review request for a bounded git diff.
pub fn diff_review_request(diff: &str) -> CollectionResult<ReviewRequest> {
    if diff.trim().is_empty() {
        return Err(CollectionError::NoText);
    }
    if diff.len() > MAX_REVIEW_DIFF_BYTES {
        return Err(CollectionError::TooLarge {
            limit: MAX_REVIEW_DIFF_BYTES,
        });
    }
    Ok(ReviewRequest::new(
        "Review code changes",
        format!("```diff\n{diff}\n```"),
    ))
}

fn truncate_characters(value: &str, maximum: usize) -> String {
    value.chars().take(maximum).collect()
}

/// Callback used to open a browser.  A closure can implement this trait,
/// making a browser review deterministic in tests and in alternate hosts.
pub trait BrowserOpener: Send + Sync {
    fn open(&self, target: &str) -> io::Result<()>;
}

impl<F> BrowserOpener for F
where
    F: Fn(&str) -> io::Result<()> + Send + Sync,
{
    fn open(&self, target: &str) -> io::Result<()> {
        self(target)
    }
}

pub type BrowserOpenCallback = Arc<dyn BrowserOpener>;
pub type ReviewNoticeCallback = Arc<dyn Fn(String) + Send + Sync + 'static>;

/// Dependency-free local browser reviewer.
///
/// The browser process is only a convenience.  Even if it cannot launch, the
/// review URL is sent to `notify` and the caller can still open it manually.
#[derive(Clone)]
pub struct BrowserReviewer {
    /// A loopback IP literal or `localhost`; empty selects `127.0.0.1`.
    pub host: String,
    /// Optional injected browser launcher.
    pub open_browser: Option<BrowserOpenCallback>,
    /// Optional status delivery hook for a terminal or Ratatui notice area.
    pub notify: Option<ReviewNoticeCallback>,
    /// How often cancellation is checked while waiting for a connection.
    pub poll_interval: Duration,
}

impl Default for BrowserReviewer {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_owned(),
            open_browser: None,
            notify: None,
            poll_interval: Duration::from_millis(25),
        }
    }
}

impl BrowserReviewer {
    /// Opens a first-version review.
    pub fn review(
        &self,
        cancellation: &agent::CancellationToken,
        title: impl AsRef<str>,
        markdown: impl AsRef<str>,
    ) -> Result<Decision, ReviewError> {
        self.review_inner(cancellation, title.as_ref(), markdown.as_ref(), "")
    }

    /// Opens a review with a prior denied plan available as a change view.
    pub fn review_version(
        &self,
        cancellation: &agent::CancellationToken,
        title: impl AsRef<str>,
        markdown: impl AsRef<str>,
        previous: impl AsRef<str>,
    ) -> Result<Decision, ReviewError> {
        self.review_inner(
            cancellation,
            title.as_ref(),
            markdown.as_ref(),
            previous.as_ref(),
        )
    }

    fn review_inner(
        &self,
        cancellation: &agent::CancellationToken,
        title: &str,
        markdown: &str,
        previous: &str,
    ) -> Result<Decision, ReviewError> {
        if cancellation.is_cancelled() {
            return Err(ReviewError::Cancelled);
        }
        let host = if self.host.trim().is_empty() {
            "127.0.0.1"
        } else {
            self.host.trim()
        };
        validate_review_host(host)?;
        let listener = TcpListener::bind((host, 0))
            .map_err(|error| ReviewError::Failed(format!("bind review server: {error}")))?;
        let address = listener
            .local_addr()
            .map_err(|error| ReviewError::Failed(format!("inspect review server: {error}")))?;
        if !address.ip().is_loopback() {
            return Err(ReviewError::Failed(
                "review server did not bind to a loopback address".to_owned(),
            ));
        }
        listener
            .set_nonblocking(true)
            .map_err(|error| ReviewError::Failed(format!("configure review server: {error}")))?;

        let token = Uuid::now_v7().simple().to_string();
        let review_url = format!("http://{address}/");
        let opener_result = match &self.open_browser {
            Some(opener) => opener.open(&review_url),
            None => open_system_browser(&review_url),
        };
        if let Some(notify) = &self.notify {
            match opener_result {
                Ok(()) => notify(format!("Planner review: {review_url}")),
                Err(error) => notify(format!(
                    "Planner review: could not open a browser ({error}). Open this URL to continue:\n{review_url}"
                )),
            }
        }

        let interval = self.poll_interval.max(Duration::from_millis(1));
        loop {
            if cancellation.is_cancelled() {
                return Err(ReviewError::Cancelled);
            }
            match listener.accept() {
                Ok((mut stream, peer)) => {
                    if !peer.ip().is_loopback() {
                        let _ = write_plain_response(&mut stream, 421, "unexpected peer");
                        continue;
                    }
                    stream
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .map_err(|error| {
                            ReviewError::Failed(format!("configure review connection: {error}"))
                        })?;
                    stream
                        .set_write_timeout(Some(Duration::from_secs(5)))
                        .map_err(|error| {
                            ReviewError::Failed(format!("configure review connection: {error}"))
                        })?;
                    if let Some(decision) = serve_review_connection(
                        &mut stream,
                        address,
                        &token,
                        title,
                        markdown,
                        previous,
                    )? {
                        return Ok(decision);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => thread::sleep(interval),
                Err(error) => {
                    return Err(ReviewError::Failed(format!(
                        "accept review connection: {error}"
                    )));
                }
            }
        }
    }
}

impl Reviewer for BrowserReviewer {
    fn review(
        &self,
        cancellation: &agent::CancellationToken,
        request: &ReviewRequest,
    ) -> Result<Decision, ReviewError> {
        self.review(cancellation, &request.title, &request.markdown)
    }

    fn review_version(
        &self,
        cancellation: &agent::CancellationToken,
        request: &ReviewRequest,
        previous: &str,
    ) -> Result<Decision, ReviewError> {
        self.review_version(cancellation, &request.title, &request.markdown, previous)
    }
}

fn validate_review_host(host: &str) -> Result<(), ReviewError> {
    if host.eq_ignore_ascii_case("localhost") {
        return Ok(());
    }
    match host.parse::<IpAddr>() {
        Ok(address) if address.is_loopback() => Ok(()),
        _ => Err(ReviewError::Failed(format!(
            "review server host must be loopback, got {host:?}"
        ))),
    }
}

fn open_system_browser(target: &str) -> io::Result<()> {
    let parsed =
        Url::parse(target).map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "refusing to open non-http URL",
        ));
    }
    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("rundll32");
        command.arg("url.dll,FileProtocolHandler").arg(target);
        command
    };
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(target);
        command
    };
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(target);
        command
    };
    command.spawn().map(|_| ())
}

struct HttpRequest {
    method: String,
    target: String,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

fn serve_review_connection(
    stream: &mut TcpStream,
    address: SocketAddr,
    token: &str,
    title: &str,
    markdown: &str,
    previous: &str,
) -> Result<Option<Decision>, ReviewError> {
    let request = match read_http_request(stream) {
        Ok(request) => request,
        Err(error) => {
            let _ = write_plain_response(stream, 400, &format!("bad request: {error}"));
            return Ok(None);
        }
    };
    if !allowed_review_host(request.headers.get("host").map(String::as_str), address) {
        let _ = write_plain_response(stream, 421, "unexpected Host header");
        return Ok(None);
    }
    let path = request.target.split('?').next().unwrap_or_default();
    match (request.method.as_str(), path) {
        ("GET", "/") => {
            let page = render_review_page(title, markdown, previous, token);
            write_html_response(stream, 200, &page)
                .map_err(|error| ReviewError::Failed(format!("write review page: {error}")))?;
            Ok(None)
        }
        ("POST", "/api/decision") => {
            let Some(decision) = parse_form_decision(&request.body, token) else {
                let _ = write_plain_response(stream, 403, "invalid review decision");
                return Ok(None);
            };
            let page = render_review_complete_page(decision.approved);
            write_html_response(stream, 200, &page)
                .map_err(|error| ReviewError::Failed(format!("write review response: {error}")))?;
            Ok(Some(decision))
        }
        ("POST", "/api/decision.json") => {
            let Some(decision) = parse_json_decision(&request.body, token) else {
                let _ = write_plain_response(stream, 403, "invalid review decision");
                return Ok(None);
            };
            write_empty_response(stream, 204)
                .map_err(|error| ReviewError::Failed(format!("write review response: {error}")))?;
            Ok(Some(decision))
        }
        ("GET", _) => {
            let _ = write_plain_response(stream, 404, "not found");
            Ok(None)
        }
        _ => {
            let _ = write_plain_response(stream, 405, "method not allowed");
            Ok(None)
        }
    }
}

fn read_http_request(stream: &mut TcpStream) -> io::Result<HttpRequest> {
    let mut bytes = Vec::with_capacity(1024);
    let mut chunk = [0_u8; 1024];
    let header_end = loop {
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            let header_end = end + 4;
            if header_end > MAX_REVIEW_HEADER_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "request headers exceed the limit",
                ));
            }
            break header_end;
        }
        if bytes.len() >= MAX_REVIEW_HEADER_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request headers exceed the limit",
            ));
        }
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before headers",
            ));
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > MAX_REVIEW_HEADER_BYTES + MAX_REVIEW_DECISION_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "request exceeds the limit",
            ));
        }
    };

    let header_text = std::str::from_utf8(&bytes[..header_end - 4])
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "headers are not UTF-8"))?;
    let mut lines = header_text.split("\r\n");
    let first = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing request line"))?;
    let mut request_line = first.split_whitespace();
    let method = request_line
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing method"))?
        .to_owned();
    let target = request_line
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing target"))?
        .to_owned();
    if request_line.next().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing HTTP version",
        ));
    }
    let mut headers = BTreeMap::new();
    for line in lines {
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "invalid request header"))?;
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_owned());
    }
    if headers
        .get("transfer-encoding")
        .is_some_and(|value| !value.is_empty())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "chunked requests are not supported",
        ));
    }
    let content_length = headers
        .get("content-length")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid content length"))
        })
        .transpose()?
        .unwrap_or(0);
    if content_length > MAX_REVIEW_DECISION_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "decision body exceeds the limit",
        ));
    }
    let mut body = bytes[header_end..].to_vec();
    while body.len() < content_length {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "request ended before its body",
            ));
        }
        body.extend_from_slice(&chunk[..read]);
        if body.len() > content_length {
            break;
        }
    }
    body.truncate(content_length);
    Ok(HttpRequest {
        method,
        target,
        headers,
        body,
    })
}

fn allowed_review_host(host: Option<&str>, expected: SocketAddr) -> bool {
    let Some(host) = host.map(str::trim).filter(|host| !host.is_empty()) else {
        return false;
    };
    if host == expected.to_string() {
        return true;
    }
    let Some((name, port)) = split_host_port(host) else {
        return false;
    };
    if port != expected.port() {
        return false;
    }
    name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn split_host_port(value: &str) -> Option<(&str, u16)> {
    if let Some(rest) = value.strip_prefix('[') {
        let (name, port) = rest.split_once("]:")?;
        return port.parse().ok().map(|port| (name, port));
    }
    let (name, port) = value.rsplit_once(':')?;
    (!name.contains(':'))
        .then(|| port.parse().ok().map(|port| (name, port)))
        .flatten()
}

fn parse_form_decision(body: &[u8], token: &str) -> Option<Decision> {
    let values: BTreeMap<String, String> = url::form_urlencoded::parse(body)
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect();
    (values.get("token")? == token).then(|| Decision {
        approved: values
            .get("action")
            .is_some_and(|action| action == "approve"),
        feedback: values
            .get("feedback")
            .map_or_else(String::new, |feedback| feedback.trim().to_owned()),
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonDecision {
    token: String,
    approved: bool,
    #[serde(default)]
    feedback: String,
}

fn parse_json_decision(body: &[u8], token: &str) -> Option<Decision> {
    let payload: JsonDecision = serde_json::from_slice(body).ok()?;
    (payload.token == token).then(|| Decision {
        approved: payload.approved,
        feedback: payload.feedback.trim().to_owned(),
    })
}

fn write_review_headers(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    length: usize,
) -> io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {content_type}\r\nContent-Length: {length}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nReferrer-Policy: no-referrer\r\nContent-Security-Policy: default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'\r\nConnection: close\r\n\r\n",
        status,
        status_text(status)
    )
}

fn write_html_response(stream: &mut TcpStream, status: u16, page: &str) -> io::Result<()> {
    write_review_headers(stream, status, "text/html; charset=utf-8", page.len())?;
    stream.write_all(page.as_bytes())
}

fn write_empty_response(stream: &mut TcpStream, status: u16) -> io::Result<()> {
    write_review_headers(stream, status, "text/plain; charset=utf-8", 0)
}

fn write_plain_response(stream: &mut TcpStream, status: u16, body: &str) -> io::Result<()> {
    write_review_headers(stream, status, "text/plain; charset=utf-8", body.len())?;
    stream.write_all(body.as_bytes())
}

fn status_text(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        421 => "Misdirected Request",
        _ => "Error",
    }
}

/// Renders the browser review page.  It is public so a Ratatui host can
/// snapshot-test or serve the same review surface without duplicating prompt
/// and annotation semantics.
#[must_use]
pub fn render_review_page(title: &str, markdown: &str, previous: &str, token: &str) -> String {
    let document = prepare_review_document(markdown);
    let mut page = String::with_capacity(markdown.len().saturating_mul(2).saturating_add(8192));
    page.push_str(REVIEW_PAGE_START);
    let _ = write!(
        &mut page,
        "<title>{} · Planner</title></head><body><header><strong>Planner</strong><span>{}</span><button id=\"theme\" type=\"button\">Theme</button><button type=\"submit\" form=\"decision-form\" name=\"action\" value=\"deny\">Feedback</button><button type=\"submit\" form=\"decision-form\" name=\"action\" value=\"approve\">Approve</button></header><form id=\"decision-form\" method=\"post\" action=\"/api/decision\"><input type=\"hidden\" name=\"token\" value=\"{}\"><input id=\"compiled-feedback\" type=\"hidden\" name=\"feedback\"><main><aside><h2>Contents</h2>",
        escape_html(title),
        escape_html(title),
        escape_html(token)
    );
    for heading in &document.headings {
        let _ = write!(
            &mut page,
            "<a href=\"#line-{}\" class=\"toc level-{}\">{}</a>",
            heading.number,
            heading.level,
            escape_html(&heading.text)
        );
    }
    page.push_str("</aside><section><div class=\"tools\"><button id=\"select-mode\" type=\"button\">Select line</button><button id=\"edit-mode\" type=\"button\">Edit</button><button id=\"copy-plan\" type=\"button\">Copy plan</button></div><article id=\"document\">");
    for line in &document.lines {
        let _ = write!(
            &mut page,
            "<button id=\"line-{}\" class=\"line {}\" style=\"--indent:{}\" data-line=\"{}\" data-text=\"{}\" type=\"button\"><span>{}</span>{}</button>",
            line.number,
            line.kind.css_class(),
            line.indent,
            line.number,
            escape_html(&line.text),
            line.number,
            escape_html(&line.display)
        );
    }
    page.push_str("</article><section id=\"editor-wrap\"><label for=\"direct-edit\">Edit plan Markdown</label><textarea id=\"direct-edit\" spellcheck=\"false\">");
    page.push_str(&escape_html(markdown));
    page.push_str("</textarea></section>");
    if !previous.trim().is_empty() {
        page.push_str("<details><summary>± Changes from previous submission</summary>");
        page.push_str(&render_line_diff(previous, markdown));
        page.push_str("</details>");
    }
    page.push_str("</section><aside class=\"comments\"><h2>Annotations</h2><div id=\"annotations\">Click a line to add feedback.</div><label for=\"overall-notes\">Overall implementation notes</label><textarea id=\"overall-notes\" placeholder=\"Optional guidance, questions, or approval notes…\"></textarea></aside></main></form>");
    page.push_str(REVIEW_PAGE_SCRIPT);
    page.push_str("</body></html>");
    page
}

fn render_review_complete_page(approved: bool) -> String {
    let heading = if approved {
        "Plan approved"
    } else {
        "Feedback sent"
    };
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Planner</title></head><body><main><h1>{heading}</h1><p>GoshCoder received your review. You can safely close this tab.</p></main></body></html>"
    )
}

fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn render_line_diff(previous: &str, current: &str) -> String {
    let previous = previous.replace("\r\n", "\n");
    let current = current.replace("\r\n", "\n");
    let old = previous.split('\n').collect::<Vec<_>>();
    let new = current.split('\n').collect::<Vec<_>>();
    let mut output = String::from("<div class=\"diff-view\">");
    let maximum = old.len().max(new.len());
    for index in 0..maximum {
        if old.get(index) == new.get(index) {
            continue;
        }
        if let Some(line) = old.get(index) {
            let _ = write!(
                &mut output,
                "<div class=\"diff-row remove\"><span>-{}</span>{}</div>",
                index + 1,
                escape_html(line)
            );
        }
        if let Some(line) = new.get(index) {
            let _ = write!(
                &mut output,
                "<div class=\"diff-row add\"><span>+{}</span>{}</div>",
                index + 1,
                escape_html(line)
            );
        }
    }
    if maximum > 0 && old == new {
        output.push_str("<div class=\"diff-row\">No line changes.</div>");
    }
    output.push_str("</div>");
    output
}

const REVIEW_PAGE_START: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:dark;--bg:#090b10;--panel:#111721;--line:#2a3444;--text:#eef1f7;--muted:#9aa6bb;--accent:#a99cff;--ok:#65d38a}:root.light{color-scheme:light;--bg:#f5f6f8;--panel:#fff;--line:#d3dae5;--text:#1c2230;--muted:#687184;--accent:#6957d9;--ok:#27844b}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}header{position:sticky;top:0;z-index:2;display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid var(--line);background:var(--panel)}header span{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}header button:nth-last-of-type(2){margin-left:auto}button,textarea{font:inherit}button{cursor:pointer;color:inherit;background:#1b2432;border:1px solid var(--line);border-radius:6px;padding:7px 10px}:root.light button{background:#f4f6fa}main{display:grid;grid-template-columns:210px minmax(0,1fr) 290px;min-height:calc(100vh - 57px)}aside{padding:16px;background:var(--panel);border-right:1px solid var(--line)}aside.comments{border-left:1px solid var(--line);border-right:0}h2{margin:0 0 12px;font-size:12px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}.toc{display:block;padding:5px;color:var(--muted);text-decoration:none;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.level-2{padding-left:15px}.level-3,.level-4,.level-5,.level-6{padding-left:25px}section{min-width:0;padding:18px}.tools{display:flex;gap:8px;margin-bottom:14px}article{border:1px solid var(--line);border-radius:9px;background:var(--panel);overflow:hidden}.line{display:block;position:relative;width:100%;min-height:1.65em;padding:4px 12px 4px calc(42px + var(--indent) * 16px);border:0;border-radius:0;background:transparent;text-align:left;white-space:pre-wrap;overflow-wrap:anywhere}.line:hover,.line.selected{background:#a99cff1d}.line>span{position:absolute;left:8px;color:var(--muted);font:11px ui-monospace,monospace}.heading-1,.heading-2,.heading-3{font-weight:700;color:var(--accent)}.heading-1{font-size:23px}.heading-2{font-size:18px}.task:before{content:"☐";position:absolute;left:calc(21px + var(--indent) * 16px);color:var(--accent)}.task-done:before{content:"☑";position:absolute;left:calc(21px + var(--indent) * 16px);color:var(--ok)}.bullet:before{content:"•";position:absolute;left:calc(24px + var(--indent) * 16px);color:var(--accent)}.code,.diff-add,.diff-remove,.diff-hunk{background:#090d13;font-family:ui-monospace,monospace}.diff-add{color:#9be5b2}.diff-remove{color:#ffaaaa}.diff-hunk{color:#aaa0ef}.fence{display:none}.blank{height:12px}.comments textarea,#direct-edit{width:100%;min-height:100px;margin:7px 0 16px;padding:8px;border:1px solid var(--line);border-radius:6px;background:#090d13;color:var(--text)}:root.light .comments textarea,:root.light #direct-edit{background:#f8f9fc}#editor-wrap{display:none}#editor-wrap.open{display:block}#annotations{color:var(--muted);white-space:pre-wrap;line-height:1.5}details{margin-top:14px;border:1px solid var(--line);border-radius:7px;padding:9px}.diff-row{display:grid;grid-template-columns:42px 1fr;padding:2px 4px;white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,monospace}.diff-row.remove{background:#ef77771d;color:#ffaaaa}.diff-row.add{background:#65d38a1d;color:#9be5b2}@media(max-width:850px){main{grid-template-columns:1fr}main>aside:first-child{display:none}.comments{border-top:1px solid var(--line);border-left:0!important}}
</style>"#;

const REVIEW_PAGE_SCRIPT: &str = r#"<script>
(()=>{const form=document.querySelector('#decision-form'),editor=document.querySelector('#direct-edit'),original=editor.defaultValue,annotations=[],view=document.querySelector('#annotations'),hidden=document.querySelector('#compiled-feedback');function esc(v){return String(v||'').trim()}function render(){document.querySelectorAll('.line').forEach(x=>x.classList.remove('selected'));if(!annotations.length){view.textContent='Click a line to add feedback.';return}view.textContent=annotations.map(a=>{const line=document.querySelector('[data-line="'+a.line+'"]');if(line)line.classList.add('selected');return 'Line '+a.line+' ("'+a.text.trim().slice(0,120)+'"): '+a.comment}).join('\n\n')}document.querySelectorAll('.line').forEach(line=>line.addEventListener('click',()=>{const existing=annotations.find(a=>a.line===line.dataset.line),comment=prompt('Feedback for line '+line.dataset.line+':',existing?existing.comment:'');if(comment===null)return;if(existing)existing.comment=comment;else annotations.push({line:line.dataset.line,text:line.dataset.text||'',comment});render()}));function compile(){const chunks=annotations.filter(a=>esc(a.comment)).map(a=>'- Line '+a.line+' ("'+esc(a.text).slice(0,120)+'"): '+esc(a.comment)),notes=esc(document.querySelector('#overall-notes').value);if(notes)chunks.push('## Overall notes\n\n'+notes);if(editor.value!==original)chunks.push('## Direct edits\n\nThe reviewer edited the plan directly. Apply this complete revised Markdown:\n\n~~~markdown\n'+editor.value+'\n~~~');hidden.value=chunks.length?'## Planner review\n\n'+chunks.join('\n\n'):''}form.addEventListener('submit',event=>{compile();if(event.submitter&&event.submitter.value==='approve'&&editor.value!==original)event.submitter.value='deny'});document.querySelector('#edit-mode').addEventListener('click',()=>document.querySelector('#editor-wrap').classList.toggle('open'));document.querySelector('#copy-plan').addEventListener('click',()=>navigator.clipboard&&navigator.clipboard.writeText(editor.value));const theme=document.querySelector('#theme');try{if(localStorage.getItem('planner-theme')==='light')document.documentElement.classList.add('light');theme.addEventListener('click',()=>{document.documentElement.classList.toggle('light');localStorage.setItem('planner-theme',document.documentElement.classList.contains('light')?'light':'dark')})}catch(_){}render()})();
</script>"#;

/// Source from which a bounded annotation document was collected.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextOrigin {
    File(PathBuf),
    Directory(PathBuf),
    Url(Url),
    AssistantMessage,
}

/// A ready-to-review document plus display names needed to turn its decision
/// into an agent follow-up prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectedText {
    pub review_title: String,
    pub feedback_subject: String,
    pub text: String,
    pub origin: TextOrigin,
    pub files: usize,
}

impl CollectedText {
    /// Constructs the prompt a reviewer should receive.
    #[must_use]
    pub fn review_request(&self) -> ReviewRequest {
        ReviewRequest::new(&self.review_title, &self.text)
    }
}

/// Response abstraction used by [`UrlTextFetcher`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UrlTextResponse {
    pub status: u16,
    pub final_url: Url,
    pub content_type: Option<String>,
    pub body: Vec<u8>,
}

/// Injectable bounded HTTP implementation for URL annotations.
pub trait UrlTextFetcher: Send + Sync {
    fn fetch(&self, url: &Url, max_bytes: usize) -> Result<UrlTextResponse, CollectionError>;
}

/// `reqwest` implementation that follows only a small number of safe HTTP(S)
/// redirects and reads no more than the requested response budget.
#[derive(Clone)]
pub struct ReqwestUrlFetcher {
    client: reqwest::blocking::Client,
}

impl ReqwestUrlFetcher {
    pub fn new(timeout: Duration) -> CollectionResult<Self> {
        let client = reqwest::blocking::Client::builder()
            .connect_timeout(timeout.min(Duration::from_secs(10)))
            .timeout(timeout)
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() > 5 {
                    attempt.error("too many annotation redirects")
                } else if safe_redirect_url(attempt.url()) {
                    attempt.follow()
                } else {
                    attempt.error("unsafe annotation redirect")
                }
            }))
            .build()
            .map_err(|error| CollectionError::Fetch(error.to_string()))?;
        Ok(Self { client })
    }
}

impl UrlTextFetcher for ReqwestUrlFetcher {
    fn fetch(&self, url: &Url, max_bytes: usize) -> Result<UrlTextResponse, CollectionError> {
        let mut response = self
            .client
            .get(url.as_str())
            .send()
            .map_err(|error| CollectionError::Fetch(error.to_string()))?;
        let status = response.status().as_u16();
        let final_url = response.url().clone();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let mut body = Vec::with_capacity(max_bytes.min(64 * 1024));
        response
            .by_ref()
            .take(max_bytes.saturating_add(1) as u64)
            .read_to_end(&mut body)
            .map_err(|error| CollectionError::Fetch(error.to_string()))?;
        if body.len() > max_bytes {
            return Err(CollectionError::TooLarge { limit: max_bytes });
        }
        Ok(UrlTextResponse {
            status,
            final_url,
            content_type,
            body,
        })
    }
}

/// URL safety policy for annotation.  Direct private IPs and `localhost` are
/// refused by default.  Hostnames can still be subject to DNS rebinding, so a
/// production embedding should additionally enforce its normal egress policy
/// in the supplied [`UrlTextFetcher`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UrlFetchPolicy {
    pub allow_private_networks: bool,
    pub max_url_bytes: usize,
}

impl Default for UrlFetchPolicy {
    fn default() -> Self {
        Self {
            allow_private_networks: false,
            max_url_bytes: 8 * 1024,
        }
    }
}

/// Boundaries for local directory collection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectionOptions {
    pub max_bytes: usize,
    pub max_files: usize,
    pub max_entries: usize,
    pub url_policy: UrlFetchPolicy,
}

impl Default for CollectionOptions {
    fn default() -> Self {
        Self {
            max_bytes: MAX_ANNOTATION_BYTES,
            max_files: 10_000,
            max_entries: 100_000,
            url_policy: UrlFetchPolicy::default(),
        }
    }
}

/// Errors from bounded file, folder, URL, and assistant-response collection.
#[derive(Debug)]
pub enum CollectionError {
    EmptyTarget,
    InvalidUrl(String),
    UnsafePath(String),
    NotRegularFile(PathBuf),
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    TooLarge {
        limit: usize,
    },
    TooManyFiles {
        limit: usize,
    },
    TooManyEntries {
        limit: usize,
    },
    NoAnnotatableText,
    NoAssistantResponse,
    NoText,
    HttpStatus {
        url: String,
        status: u16,
    },
    Fetch(String),
}

impl fmt::Display for CollectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTarget => formatter.write_str("annotation target is empty"),
            Self::InvalidUrl(reason) => write!(formatter, "invalid annotation URL: {reason}"),
            Self::UnsafePath(path) => write!(formatter, "annotation path is unsafe: {path}"),
            Self::NotRegularFile(path) => {
                write!(formatter, "{} is not a regular file", path.display())
            }
            Self::Io {
                action,
                path,
                source,
            } => write!(formatter, "{action} {}: {source}", path.display()),
            Self::TooLarge { limit } => write!(
                formatter,
                "content exceeds the {limit}-byte annotation limit"
            ),
            Self::TooManyFiles { limit } => write!(
                formatter,
                "annotation folder exceeds the {limit}-file limit"
            ),
            Self::TooManyEntries { limit } => {
                write!(
                    formatter,
                    "annotation folder exceeds the {limit}-entry limit"
                )
            }
            Self::NoAnnotatableText => formatter.write_str("folder has no annotatable text files"),
            Self::NoAssistantResponse => formatter.write_str("no assistant response found"),
            Self::NoText => formatter.write_str("there are no changes to review"),
            Self::HttpStatus { url, status } => write!(formatter, "fetch {url}: status {status}"),
            Self::Fetch(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CollectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

pub type CollectionResult<T> = Result<T, CollectionError>;

/// Safe, bounded collector backing `/planner-annotate`-style integrations.
///
/// Local inputs are confined to the canonical workspace and symlinks are
/// refused.  As with the existing Rust workspace tools, standard-library
/// path checks cannot make this race-free against a malicious concurrent
/// process replacing a checked component.
#[derive(Clone)]
pub struct TextCollector {
    root: PathBuf,
    fetcher: Arc<dyn UrlTextFetcher>,
    options: CollectionOptions,
}

impl TextCollector {
    /// Constructs a collector with the crate's existing blocking HTTP client.
    pub fn new(root: impl AsRef<Path>) -> CollectionResult<Self> {
        let fetcher = Arc::new(ReqwestUrlFetcher::new(Duration::from_secs(30))?);
        Self::with_fetcher(root, fetcher)
    }

    /// Constructs a collector with an injectable URL transport.
    pub fn with_fetcher(
        root: impl AsRef<Path>,
        fetcher: Arc<dyn UrlTextFetcher>,
    ) -> CollectionResult<Self> {
        let requested = root.as_ref();
        let root = fs::canonicalize(requested).map_err(|source| CollectionError::Io {
            action: "resolve workspace",
            path: requested.to_path_buf(),
            source,
        })?;
        if !fs::metadata(&root)
            .map_err(|source| CollectionError::Io {
                action: "inspect workspace",
                path: root.clone(),
                source,
            })?
            .is_dir()
        {
            return Err(CollectionError::UnsafePath(format!(
                "{} is not a directory",
                root.display()
            )));
        }
        Ok(Self {
            root,
            fetcher,
            options: CollectionOptions::default(),
        })
    }

    /// Replaces collection limits and URL policy.
    #[must_use]
    pub fn with_options(mut self, options: CollectionOptions) -> Self {
        self.options = options;
        self
    }

    /// Returns the canonical workspace root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Collects a local file/folder or HTTP(S) URL into one bounded document.
    pub fn collect(&self, target: &str) -> CollectionResult<CollectedText> {
        let target = target.trim();
        if target.is_empty() {
            return Err(CollectionError::EmptyTarget);
        }
        if target.starts_with("http://") || target.starts_with("https://") {
            return self.collect_url(target);
        }
        if target.contains("://") {
            return Err(CollectionError::InvalidUrl(
                "only http:// and https:// URLs are supported".to_owned(),
            ));
        }
        self.collect_path(target)
    }

    fn collect_url(&self, target: &str) -> CollectionResult<CollectedText> {
        let url =
            Url::parse(target).map_err(|error| CollectionError::InvalidUrl(error.to_string()))?;
        validate_annotation_url(&url, &self.options.url_policy)?;
        let limit = self.options.max_bytes.max(1);
        let response = self.fetcher.fetch(&url, limit)?;
        validate_annotation_url(&response.final_url, &self.options.url_policy)?;
        if !(200..300).contains(&response.status) {
            return Err(CollectionError::HttpStatus {
                url: response.final_url.to_string(),
                status: response.status,
            });
        }
        if response.body.len() > limit {
            return Err(CollectionError::TooLarge { limit });
        }
        Ok(CollectedText {
            review_title: format!("Annotate {url}"),
            feedback_subject: url.to_string(),
            text: String::from_utf8_lossy(&response.body).into_owned(),
            origin: TextOrigin::Url(response.final_url),
            files: 1,
        })
    }

    fn collect_path(&self, input: &str) -> CollectionResult<CollectedText> {
        let (path, metadata) = self.confined_existing_path(input)?;
        if metadata.is_dir() {
            let mut budget = DirectoryBudget::default();
            self.collect_directory(&path, &path, &mut budget)?;
            if budget.files == 0 {
                return Err(CollectionError::NoAnnotatableText);
            }
            let title = format!("Annotate {}", display_basename(&path));
            return Ok(CollectedText {
                review_title: title,
                feedback_subject: display_path(&path),
                text: String::from_utf8_lossy(&budget.bytes).into_owned(),
                origin: TextOrigin::Directory(path),
                files: budget.files,
            });
        }
        if !metadata.is_file() {
            return Err(CollectionError::NotRegularFile(path));
        }
        let bytes = read_collection_file(&path, self.options.max_bytes.max(1))?;
        Ok(CollectedText {
            review_title: format!("Annotate {}", display_basename(&path)),
            feedback_subject: display_path(&path),
            text: String::from_utf8_lossy(&bytes).into_owned(),
            origin: TextOrigin::File(path),
            files: 1,
        })
    }

    fn confined_existing_path(&self, input: &str) -> CollectionResult<(PathBuf, fs::Metadata)> {
        let source = Path::new(input);
        let target = if is_rooted_path(input) {
            if !source.is_absolute() {
                return Err(CollectionError::UnsafePath(input.to_owned()));
            }
            normalize_absolute(source)
                .ok_or_else(|| CollectionError::UnsafePath(input.to_owned()))?
        } else {
            normalize_absolute(&self.root.join(source))
                .ok_or_else(|| CollectionError::UnsafePath(input.to_owned()))?
        };
        let relative = target
            .strip_prefix(&self.root)
            .map_err(|_| CollectionError::UnsafePath(input.to_owned()))?;
        let mut current = self.root.clone();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                return Err(CollectionError::UnsafePath(input.to_owned()));
            };
            current.push(name);
            let metadata =
                fs::symlink_metadata(&current).map_err(|source| CollectionError::Io {
                    action: "inspect annotation path",
                    path: current.clone(),
                    source,
                })?;
            if metadata.file_type().is_symlink() {
                return Err(CollectionError::UnsafePath(display_path(&current)));
            }
        }
        let metadata = fs::symlink_metadata(&target).map_err(|source| CollectionError::Io {
            action: "inspect annotation path",
            path: target.clone(),
            source,
        })?;
        if metadata.file_type().is_symlink() {
            return Err(CollectionError::UnsafePath(display_path(&target)));
        }
        Ok((target, metadata))
    }

    fn collect_directory(
        &self,
        base: &Path,
        directory: &Path,
        budget: &mut DirectoryBudget,
    ) -> CollectionResult<()> {
        let mut entries = fs::read_dir(directory)
            .map_err(|source| CollectionError::Io {
                action: "read annotation directory",
                path: directory.to_path_buf(),
                source,
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|source| CollectionError::Io {
                action: "read annotation directory",
                path: directory.to_path_buf(),
                source,
            })?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            budget.entries += 1;
            if budget.entries > self.options.max_entries.max(1) {
                return Err(CollectionError::TooManyEntries {
                    limit: self.options.max_entries.max(1),
                });
            }
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).map_err(|source| CollectionError::Io {
                action: "inspect annotation path",
                path: path.clone(),
                source,
            })?;
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                if ignored_annotation_directory(&entry.file_name().to_string_lossy()) {
                    continue;
                }
                self.collect_directory(base, &path, budget)?;
                continue;
            }
            if !metadata.is_file() || !annotatable_extension(&path) {
                continue;
            }
            if budget.files >= self.options.max_files.max(1) {
                return Err(CollectionError::TooManyFiles {
                    limit: self.options.max_files.max(1),
                });
            }
            let relative = path
                .strip_prefix(base)
                .map_err(|_| CollectionError::UnsafePath(display_path(&path)))?;
            let heading = format!("\n\n## {}\n\n", display_path(relative));
            let limit = self.options.max_bytes.max(1);
            if budget.bytes.len().saturating_add(heading.len()) > limit {
                return Err(CollectionError::TooLarge { limit });
            }
            let remaining = limit - budget.bytes.len() - heading.len();
            let bytes = read_collection_file(&path, remaining)?;
            budget.bytes.extend_from_slice(heading.as_bytes());
            budget.bytes.extend_from_slice(&bytes);
            budget.files += 1;
        }
        Ok(())
    }
}

#[derive(Default)]
struct DirectoryBudget {
    bytes: Vec<u8>,
    files: usize,
    entries: usize,
}

fn validate_annotation_url(url: &Url, policy: &UrlFetchPolicy) -> CollectionResult<()> {
    if url.as_str().len() > policy.max_url_bytes.max(1) {
        return Err(CollectionError::InvalidUrl(
            "URL exceeds the length limit".to_owned(),
        ));
    }
    if !matches!(url.scheme(), "http" | "https") {
        return Err(CollectionError::InvalidUrl(
            "only http:// and https:// URLs are supported".to_owned(),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(CollectionError::InvalidUrl(
            "URLs with embedded credentials are not allowed".to_owned(),
        ));
    }
    let Some(host) = url.host() else {
        return Err(CollectionError::InvalidUrl("URL has no host".to_owned()));
    };
    if !policy.allow_private_networks && host_is_private(host) {
        return Err(CollectionError::InvalidUrl(
            "private or loopback URL hosts are not allowed".to_owned(),
        ));
    }
    Ok(())
}

fn safe_redirect_url(url: &Url) -> bool {
    matches!(url.scheme(), "http" | "https")
        && url.username().is_empty()
        && url.password().is_none()
        && url.host().is_some_and(|host| !host_is_private(host))
}

fn host_is_private(host: Host<&str>) -> bool {
    match host {
        Host::Domain(domain) => {
            domain.eq_ignore_ascii_case("localhost")
                || domain.to_ascii_lowercase().ends_with(".localhost")
        }
        Host::Ipv4(address) => private_ipv4(address),
        Host::Ipv6(address) => private_ipv6(address),
    }
}

fn private_ipv4(address: Ipv4Addr) -> bool {
    let octets = address.octets();
    address.is_private()
        || address.is_loopback()
        || address.is_link_local()
        || address.is_broadcast()
        || address.is_unspecified()
        || address.is_multicast()
        || (octets[0] == 100 && (64..=127).contains(&octets[1]))
}

fn private_ipv6(address: Ipv6Addr) -> bool {
    address.is_loopback()
        || address.is_unspecified()
        || address.is_multicast()
        || address.is_unicast_link_local()
        || (address.segments()[0] & 0xfe00 == 0xfc00)
        || address.to_ipv4_mapped().is_some_and(private_ipv4)
}

fn read_collection_file(path: &Path, limit: usize) -> CollectionResult<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|source| CollectionError::Io {
        action: "inspect annotation file",
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(CollectionError::NotRegularFile(path.to_path_buf()));
    }
    let mut file = File::open(path).map_err(|source| CollectionError::Io {
        action: "open annotation file",
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::with_capacity(limit.min(64 * 1024));
    Read::by_ref(&mut file)
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|source| CollectionError::Io {
            action: "read annotation file",
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > limit {
        return Err(CollectionError::TooLarge { limit });
    }
    Ok(bytes)
}

fn ignored_annotation_directory(name: &str) -> bool {
    name.starts_with('.') || matches!(name, "node_modules" | "vendor" | "dist" | "build" | ".venv")
}

fn annotatable_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "md" | "mdx" | "txt" | "html" | "htm"
            )
        })
}

fn display_basename(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| display_path(path))
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Collects the most recent non-empty assistant response for
/// `/planner-last`-style integrations.
pub fn collect_last_assistant_response(
    messages: &[llm::Message],
) -> CollectionResult<CollectedText> {
    for message in messages.iter().rev() {
        let llm::Message::Assistant(message) = message else {
            continue;
        };
        let text = assistant_block_summary(&message.content);
        if text.trim().is_empty() {
            continue;
        }
        if text.len() > MAX_ANNOTATION_BYTES {
            return Err(CollectionError::TooLarge {
                limit: MAX_ANNOTATION_BYTES,
            });
        }
        return Ok(CollectedText {
            review_title: "Annotate last assistant response".to_owned(),
            feedback_subject: "the previous assistant response".to_owned(),
            text,
            origin: TextOrigin::AssistantMessage,
            files: 0,
        });
    }
    Err(CollectionError::NoAssistantResponse)
}

fn assistant_block_summary(blocks: &[llm::ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text(content) if !content.text.trim().is_empty() => {
                Some(content.text.clone())
            }
            llm::ContentBlock::Thinking(_) => Some("[thinking]".to_owned()),
            llm::ContentBlock::ToolCall(call) => Some(format!("[{}]", call.name)),
            llm::ContentBlock::Image(_) => Some("[image]".to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        fs,
        net::TcpStream,
        path::PathBuf,
        process,
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Mutex,
        },
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    static SCRATCH_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "goshcoder-plannotator-{}-{nonce}-{}",
                process::id(),
                SCRATCH_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create scratch");
            Self(path)
        }

        fn write(&self, relative: &str, content: impl AsRef<[u8]>) {
            let path = self.0.join(relative);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create parent");
            }
            fs::write(path, content).expect("write fixture");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Clone)]
    struct ReviewerStub {
        decision: Decision,
        error: Option<ReviewError>,
    }

    impl Reviewer for ReviewerStub {
        fn review(
            &self,
            _: &agent::CancellationToken,
            _: &ReviewRequest,
        ) -> Result<Decision, ReviewError> {
            self.error
                .clone()
                .map_or_else(|| Ok(self.decision.clone()), Err)
        }
    }

    struct VersionedReviewerStub {
        decisions: Mutex<Vec<Decision>>,
        previous: Mutex<Vec<String>>,
    }

    impl Reviewer for VersionedReviewerStub {
        fn review(
            &self,
            _: &agent::CancellationToken,
            _: &ReviewRequest,
        ) -> Result<Decision, ReviewError> {
            Err(ReviewError::Failed(
                "expected versioned review call".to_owned(),
            ))
        }

        fn review_version(
            &self,
            _: &agent::CancellationToken,
            _: &ReviewRequest,
            previous: &str,
        ) -> Result<Decision, ReviewError> {
            lock(&self.previous).push(previous.to_owned());
            Ok(lock(&self.decisions).remove(0))
        }
    }

    #[derive(Clone)]
    struct FakeFetcher {
        response: Result<UrlTextResponse, String>,
        calls: Arc<AtomicUsize>,
    }

    impl UrlTextFetcher for FakeFetcher {
        fn fetch(&self, _: &Url, _: usize) -> Result<UrlTextResponse, CollectionError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.response.clone().map_err(CollectionError::Fetch)
        }
    }

    fn manager(root: &Scratch, reviewer: Option<Arc<dyn Reviewer>>) -> Manager {
        Manager::new(&root.0, reviewer, Options::default()).expect("create manager")
    }

    fn approved_plan(manager: &Manager, path: &str) {
        manager.enter();
        let result = manager
            .submit(&agent::CancellationToken::default(), path)
            .expect("submit plan");
        assert!(result.approved, "{result:?}");
    }

    fn tool_call(name: &str, path: Option<&str>) -> llm::ToolCall {
        let mut arguments = BTreeMap::new();
        if let Some(path) = path {
            arguments.insert("path".to_owned(), Value::String(path.to_owned()));
        }
        llm::ToolCall {
            id: "call".to_owned(),
            name: name.to_owned(),
            arguments,
            thought_signature: String::new(),
            namespace: String::new(),
        }
    }

    fn assistant(text: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            content: vec![llm::ContentBlock::text(text)],
            ..llm::AssistantMessage::default()
        }
    }

    #[test]
    fn checklist_parser_accepts_numbered_and_plus_items() {
        let items = parse_checklist("1. [ ] First\n+ [x] Second\n- [ ] Third");
        assert_eq!(
            items,
            vec![
                ChecklistItem {
                    step: 1,
                    text: "First".to_owned(),
                    completed: false
                },
                ChecklistItem {
                    step: 2,
                    text: "Second".to_owned(),
                    completed: true
                },
                ChecklistItem {
                    step: 3,
                    text: "Third".to_owned(),
                    completed: false
                }
            ]
        );
    }

    #[test]
    fn checklist_parser_requires_a_real_line_item() {
        assert!(parse_checklist("  - [ ] indented\n- [q] invalid").is_empty());
        assert_eq!(parse_checklist("-[ ] allowed by the Go grammar").len(), 1);
    }

    #[test]
    fn review_document_keeps_headings_numbered_sections_and_c_includes() {
        let document = prepare_review_document(
            "# Plan\n#include <stdio.h>\n\n1. First section\n- [x] done\n- [ ] open\n",
        );
        assert_eq!(document.headings.len(), 1);
        assert_eq!(document.headings[0].text, "Plan");
        assert!(document
            .lines
            .iter()
            .any(|line| line.display == "#include <stdio.h>"
                && line.kind == ReviewLineKind::Paragraph));
        assert!(document
            .lines
            .iter()
            .any(|line| line.kind == ReviewLineKind::Numbered));
        assert!(document
            .lines
            .iter()
            .any(|line| line.kind == ReviewLineKind::Task { completed: true }));
    }

    #[test]
    fn review_document_indents_nested_sections_and_marks_diff_lines() {
        let document = prepare_review_document(
            "# Plan\n- Parent\n  - Nested child\n    1. Deep numbered\n```diff\n+added\n-removed\n@@ hunk\n```\n",
        );
        assert_eq!(document.lines[1].kind, ReviewLineKind::Bullet);
        assert_eq!(document.lines[1].indent, 0);
        assert_eq!(document.lines[2].kind, ReviewLineKind::Bullet);
        assert_eq!(document.lines[2].indent, 1);
        assert_eq!(document.lines[3].kind, ReviewLineKind::Numbered);
        assert_eq!(document.lines[3].indent, 2);
        assert!(document
            .lines
            .iter()
            .any(|line| line.kind == ReviewLineKind::DiffAdd));
        assert!(document
            .lines
            .iter()
            .any(|line| line.kind == ReviewLineKind::DiffRemove));
        assert!(document
            .lines
            .iter()
            .any(|line| line.kind == ReviewLineKind::DiffHunk));
    }

    #[test]
    fn approved_plan_tracks_done_markers_and_returns_idle() {
        let root = Scratch::new();
        root.write("PLAN.md", "- [ ] First\n* [x] Second\n- [ ] Third\n");
        let planner = manager(&root, None);
        approved_plan(&planner, "PLAN.md");
        assert_eq!(
            planner.track_assistant(&assistant("done [DONE:1] and [done:3]")),
            2
        );
        let state = planner.state();
        assert_eq!(state.phase, Phase::Idle);
        assert!(state.items.iter().all(|item| item.completed));
        assert_eq!(planner.track_text("[DONE:1] [DONE:999]"), 0);
    }

    #[test]
    fn planning_write_gate_matches_tool_contract() {
        let root = Scratch::new();
        let planner = manager(&root, None);
        planner.enter();

        assert_eq!(
            planner.before_tool_call(&tool_call("write", Some("plans/a.md"))),
            None
        );
        assert!(planner
            .before_tool_call(&tool_call("bash", None))
            .is_some_and(|result| result.block));
        assert!(planner
            .before_tool_call(&tool_call("edit", Some("main.rs")))
            .is_some_and(|result| result.block));
        assert!(planner
            .before_tool_call(&tool_call("write", None))
            .is_some_and(|result| result.block));
        assert!(!planner.is_plan_path_allowed("../escape.md"));
        assert!(!planner.is_plan_path_allowed("/tmp/escape.md"));
        assert!(!planner.is_plan_path_allowed("PLAN.txt"));
    }

    #[test]
    fn plan_path_gate_is_platform_independent() {
        let root = Scratch::new();
        let planner = manager(&root, None);
        for path in [
            "/plans/a.md",
            "\\plans\\a.md",
            "C:\\plans\\a.md",
            "c:/plans/a.md",
            "\\\\server\\share\\a.md",
            "../escape.md",
        ] {
            assert!(
                !planner.is_plan_path_allowed(path),
                "{path:?} unexpectedly allowed"
            );
        }
        for path in ["PLAN.md", "plans/a.md", "docs/design.mdx"] {
            assert!(planner.is_plan_path_allowed(path), "{path:?} was blocked");
        }
        let absolute_inside = root.0.join("PLAN.md");
        assert!(planner.is_plan_path_allowed(&absolute_inside.to_string_lossy()));
    }

    #[test]
    fn submit_denied_then_approved_preserves_planning_until_approval() {
        let root = Scratch::new();
        root.write("PLAN.md", "# Plan\n- [ ] Implement\n- [ ] Test\n");
        let planner = manager(
            &root,
            Some(Arc::new(ReviewerStub {
                decision: Decision {
                    approved: false,
                    feedback: "add rollback".to_owned(),
                },
                error: None,
            })),
        );
        planner.enter();
        let denied = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("deny result");
        assert!(!denied.approved);
        assert!(denied.text.contains("add rollback"));
        assert_eq!(planner.state().phase, Phase::Planning);

        planner.set_reviewer(Some(Arc::new(ReviewerStub {
            decision: Decision {
                approved: true,
                feedback: "keep commits small".to_owned(),
            },
            error: None,
        })));
        let approved = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("approved result");
        assert!(approved.approved);
        assert!(approved.text.contains("keep commits small"));
        let state = planner.state();
        assert_eq!(state.phase, Phase::Executing);
        assert_eq!(state.items.len(), 2);
    }

    #[test]
    fn submit_requires_a_non_empty_checklist_and_planning_phase() {
        let root = Scratch::new();
        root.write("PLAN.md", "# Plan\nJust do it.\n");
        let planner = manager(&root, None);
        let idle = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("idle tool response");
        assert!(idle.text.contains("Not in Planner"));
        planner.enter();
        let missing = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("checklist error");
        assert!(missing.text.contains("checklist"));
        assert_eq!(planner.state().phase, Phase::Planning);
        root.write("empty.md", " \n");
        assert!(planner
            .submit(&agent::CancellationToken::default(), "empty.md")
            .expect("empty response")
            .text
            .contains("empty"));
    }

    #[test]
    fn submit_rejects_plans_over_the_fixed_byte_limit() {
        let root = Scratch::new();
        let mut plan = vec![b' '; MAX_PLAN_BYTES + 1];
        let prefix = b"- [ ] work\n";
        plan[..prefix.len()].copy_from_slice(prefix);
        root.write("PLAN.md", plan);
        let planner = manager(&root, None);
        planner.enter();
        let response = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("model-facing size rejection");
        assert!(response
            .text
            .contains(&format!("plan exceeds {MAX_PLAN_BYTES} bytes")));
        assert_eq!(planner.state().phase, Phase::Planning);
    }

    #[cfg(unix)]
    #[test]
    fn submit_rejects_a_symlink_escape() {
        use std::os::unix::fs::symlink;

        let root = Scratch::new();
        let outside = Scratch::new();
        outside.write("outside.md", "- [ ] Exfiltrate");
        symlink(outside.0.join("outside.md"), root.0.join("PLAN.md")).expect("make symlink");
        let planner = manager(&root, None);
        planner.enter();
        let result = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("model-facing rejection");
        assert!(result.text.contains("cannot be read"));
        assert!(result.text.contains("not a regular file"));
    }

    #[test]
    fn unknown_saved_phase_recovers_to_idle_with_a_warning() {
        let root = Scratch::new();
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let capture = warnings.clone();
        let planner = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Unknown("nonsense".to_owned()),
                    ..State::default()
                }),
                on_change: None,
                warn: Some(Arc::new(move |message| lock(&capture).push(message))),
            },
        )
        .expect("recover manager");
        assert_eq!(planner.state().phase, Phase::Idle);
        let warnings = lock(&warnings);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("nonsense"));
    }

    #[test]
    fn state_serialization_keeps_pi_custom_field_names_and_unknown_phase() {
        let serialized = serde_json::to_value(State {
            phase: Phase::Executing,
            plan_path: "PLAN.md".to_owned(),
            items: vec![ChecklistItem {
                step: 1,
                text: "work".to_owned(),
                completed: false,
            }],
            plan_hash: "hash".to_owned(),
        })
        .expect("serialize");
        assert_eq!(serialized["planPath"], "PLAN.md");
        assert_eq!(serialized["planHash"], "hash");
        let restored: State =
            serde_json::from_str(r#"{"phase":"future-phase","planPath":"PLAN.md"}"#)
                .expect("deserialize");
        assert_eq!(restored.phase, Phase::Unknown("future-phase".to_owned()));
    }

    #[test]
    fn state_callback_receives_each_transition_and_restores_state() {
        let root = Scratch::new();
        let published = Arc::new(Mutex::new(Vec::new()));
        let capture = published.clone();
        let planner = Manager::new(
            &root.0,
            None,
            Options {
                initial: None,
                on_change: Some(Arc::new(move |state| lock(&capture).push(state))),
                warn: None,
            },
        )
        .expect("manager");
        planner.enter();
        planner.exit();
        assert_eq!(lock(&published).len(), 2);
        assert_eq!(lock(&published)[0].phase, Phase::Planning);
        assert_eq!(lock(&published)[1].phase, Phase::Idle);

        let resumed = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Planning,
                    ..State::default()
                }),
                ..Options::default()
            },
        )
        .expect("resumed manager");
        assert_eq!(resumed.state().phase, Phase::Planning);
    }

    #[test]
    fn completing_last_step_keeps_plan_record_and_publishes_idle() {
        let root = Scratch::new();
        let plan = "- [ ] first\n- [ ] second\n";
        root.write("PLAN.md", plan);
        let published = Arc::new(Mutex::new(Vec::new()));
        let capture = published.clone();
        let planner = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Executing,
                    plan_path: "PLAN.md".to_owned(),
                    plan_hash: hash_plan(plan.as_bytes()),
                    items: vec![
                        ChecklistItem {
                            step: 1,
                            text: "first".to_owned(),
                            completed: false,
                        },
                        ChecklistItem {
                            step: 2,
                            text: "second".to_owned(),
                            completed: false,
                        },
                    ],
                }),
                on_change: Some(Arc::new(move |state| lock(&capture).push(state))),
                warn: None,
            },
        )
        .expect("manager");
        assert_eq!(planner.track_text("[DONE:1] [DONE:2]"), 2);
        let state = planner.state();
        assert_eq!(state.phase, Phase::Idle);
        assert_eq!(state.plan_path, "PLAN.md");
        assert_eq!(state.items.len(), 2);
        assert_eq!(
            lock(&published).last().map(|state| &state.phase),
            Some(&Phase::Idle)
        );
    }

    #[test]
    fn edited_plan_drops_stale_completion_but_unedited_plan_keeps_it() {
        let root = Scratch::new();
        let original = "- [ ] set up database\n- [ ] delete staging data\n";
        let edited = "- [ ] delete staging data\n- [ ] set up database\n";
        root.write("PLAN.md", edited);
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let capture = warnings.clone();
        let planner = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Executing,
                    plan_path: "PLAN.md".to_owned(),
                    plan_hash: hash_plan(original.as_bytes()),
                    items: vec![ChecklistItem {
                        step: 1,
                        text: "set up database".to_owned(),
                        completed: true,
                    }],
                }),
                on_change: None,
                warn: Some(Arc::new(move |message| lock(&capture).push(message))),
            },
        )
        .expect("manager");
        assert!(!planner.state().items[0].completed);
        assert_eq!(lock(&warnings).len(), 1);

        root.write("PLAN.md", original);
        let unchanged = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Executing,
                    plan_path: "PLAN.md".to_owned(),
                    plan_hash: hash_plan(original.as_bytes()),
                    items: vec![ChecklistItem {
                        step: 1,
                        text: "set up database".to_owned(),
                        completed: true,
                    }],
                }),
                ..Options::default()
            },
        )
        .expect("unchanged manager");
        assert!(unchanged.state().items[0].completed);
    }

    #[test]
    fn unreadable_restored_plan_recovers_to_idle_with_warning() {
        let root = Scratch::new();
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let capture = warnings.clone();
        let planner = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Executing,
                    plan_path: "missing.md".to_owned(),
                    ..State::default()
                }),
                on_change: None,
                warn: Some(Arc::new(move |message| lock(&capture).push(message))),
            },
        )
        .expect("manager");
        assert_eq!(planner.state(), State::default());
        assert_eq!(lock(&warnings).len(), 1);
    }

    #[test]
    fn managers_in_one_workspace_do_not_share_state() {
        let root = Scratch::new();
        let first = manager(&root, None);
        let second = manager(&root, None);
        first.enter();
        assert_eq!(second.state().phase, Phase::Idle);
        second.toggle();
        second.toggle();
        assert_eq!(first.state().phase, Phase::Planning);
    }

    #[test]
    fn phase_toggles_keep_the_last_approved_plan_record() {
        let root = Scratch::new();
        let plan = "- [x] first\n- [ ] second\n";
        root.write("PLAN.md", plan);
        let planner = Manager::new(
            &root.0,
            None,
            Options {
                initial: Some(State {
                    phase: Phase::Executing,
                    plan_path: "PLAN.md".to_owned(),
                    items: vec![
                        ChecklistItem {
                            step: 1,
                            text: "first".to_owned(),
                            completed: true,
                        },
                        ChecklistItem {
                            step: 2,
                            text: "second".to_owned(),
                            completed: false,
                        },
                    ],
                    plan_hash: hash_plan(plan.as_bytes()),
                }),
                ..Options::default()
            },
        )
        .expect("manager");
        planner.enter();
        planner.toggle();
        planner.exit();
        let state = planner.state();
        assert_eq!(state.plan_path, "PLAN.md");
        assert_eq!(state.items.len(), 2);
        assert!(state.items[0].completed);
    }

    #[test]
    fn prompt_and_turn_update_switch_access_at_the_right_phase() {
        let root = Scratch::new();
        root.write("PLAN.md", "- [ ] work\n");
        let planner = manager(&root, None);
        planner.enter();
        assert!(planner
            .prompt("base")
            .contains("[PLANNER - PLANNING PHASE]"));
        approved_plan(&planner, "PLAN.md");
        let update = planner.prepare_next_turn("base", &assistant("still working"));
        assert_eq!(update.tool_access, ToolAccess::Executing);
        assert!(update.system_prompt.contains("[PLANNER - EXECUTING PLAN]"));
        assert!(update.system_prompt.contains("- [ ] 1. work"));
    }

    #[test]
    fn submit_tool_uses_existing_agent_types_and_sequential_execution() {
        let root = Scratch::new();
        root.write("PLAN.md", "- [ ] work\n");
        let planner = manager(&root, None);
        planner.enter();
        let tool = planner.tool();
        assert_eq!(tool.name, SUBMIT_TOOL_NAME);
        assert_eq!(
            tool.execution_mode,
            Some(agent::ToolExecutionMode::Sequential)
        );
        assert_eq!(tool.parameters["required"][0], "filePath");
        let result = (tool.execute)(
            agent::CancellationToken::default(),
            "tool-call".to_owned(),
            BTreeMap::from([("filePath".to_owned(), Value::String("PLAN.md".to_owned()))]),
            Arc::new(|_| {}),
        )
        .expect("tool execution");
        assert_eq!(result.details, Some(json!({"approved": true})));
        assert!(result.content[0]
            .plain_text()
            .is_some_and(|text| text.contains("Plan approved")));
    }

    #[test]
    fn submit_passes_previous_denied_plan_to_versioned_reviewer() {
        let root = Scratch::new();
        let initial = "# Plan\n- [ ] First\n";
        root.write("plan.md", initial);
        let reviewer = Arc::new(VersionedReviewerStub {
            decisions: Mutex::new(vec![
                Decision {
                    approved: false,
                    feedback: "revise".to_owned(),
                },
                Decision {
                    approved: true,
                    feedback: String::new(),
                },
            ]),
            previous: Mutex::new(Vec::new()),
        });
        let planner = manager(&root, Some(reviewer.clone()));
        planner.enter();
        planner
            .submit(&agent::CancellationToken::default(), "plan.md")
            .expect("denied submit");
        root.write("plan.md", "# Plan\n- [ ] First\n- [ ] Second\n");
        planner
            .submit(&agent::CancellationToken::default(), "plan.md")
            .expect("approved submit");
        assert_eq!(
            lock(&reviewer.previous).as_slice(),
            &["".to_owned(), initial.to_owned()]
        );
    }

    #[test]
    fn submit_cancellation_is_recoverable_but_other_review_failure_is_not() {
        let root = Scratch::new();
        root.write("PLAN.md", "- [ ] Work\n");
        let planner = manager(
            &root,
            Some(Arc::new(ReviewerStub {
                decision: Decision::default(),
                error: Some(ReviewError::Cancelled),
            })),
        );
        planner.enter();
        let cancelled = planner
            .submit(&agent::CancellationToken::default(), "PLAN.md")
            .expect("recoverable cancellation");
        assert!(cancelled.text.contains("cancelled"));
        assert_eq!(planner.state().phase, Phase::Planning);

        planner.set_reviewer(Some(Arc::new(ReviewerStub {
            decision: Decision::default(),
            error: Some(ReviewError::Failed("network failed".to_owned())),
        })));
        assert!(matches!(
            planner.submit(&agent::CancellationToken::default(), "PLAN.md"),
            Err(PlannerError::Review(ReviewError::Failed(message))) if message == "network failed"
        ));
    }

    #[test]
    fn annotation_feedback_and_review_follow_up_match_the_review_workflow() {
        let feedback = build_annotation_feedback(
            &[Annotation {
                line: 4,
                quote: "a useful line".to_owned(),
                comment: "Add tests".to_owned(),
            }],
            "Keep commits small.",
            Some("# Revised\n- [ ] work"),
            "# Original\n- [ ] work",
        );
        assert!(feedback.contains("Line 4"));
        assert!(feedback.contains("## Overall notes"));
        assert!(feedback.contains("## Direct edits"));
        assert_eq!(
            build_annotation_feedback(&[], " ", Some("same"), "same"),
            ""
        );
        assert_eq!(
            review_feedback_prompt(
                "the change",
                &Decision {
                    approved: true,
                    feedback: String::new()
                }
            ),
            None
        );
        assert_eq!(
            review_feedback_prompt("the change", &Decision::default())
                .expect("denial prompt"),
            "Planner review of the change was denied. Address this feedback:\n\nthe change was denied without notes."
        );
        assert!(diff_review_request("diff --git a/a b/a").is_ok());
        assert!(matches!(
            diff_review_request(""),
            Err(CollectionError::NoText)
        ));
    }

    #[test]
    fn browser_reviewer_serves_escaped_page_and_accepts_a_decision() {
        let (opened_sender, opened_receiver) = mpsc::channel();
        let reviewer = BrowserReviewer {
            open_browser: Some(Arc::new(move |target: &str| {
                opened_sender
                    .send(target.to_owned())
                    .map_err(|error| io::Error::other(error.to_string()))
            })),
            poll_interval: Duration::from_millis(2),
            ..BrowserReviewer::default()
        };
        let cancellation = agent::CancellationToken::default();
        let waiting = cancellation.clone();
        let thread = thread::spawn(move || {
            reviewer.review(
                &waiting,
                "Review <script>",
                "# Plan\n- [ ] <script>alert(1)</script>",
            )
        });
        let target = opened_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("browser URL");
        let page = http_request(&target, "GET", "/", "");
        let body = response_body(&page);
        assert!(body.contains("Planner"));
        for feature in [
            "Contents",
            "Annotations",
            "Feedback",
            "Edit",
            "Copy plan",
            "Overall implementation notes",
        ] {
            assert!(body.contains(feature), "missing {feature:?} in {body}");
        }
        assert!(!body.contains("<script>alert(1)</script>"));
        let token = review_token(&body);
        let response = http_request(
            &target,
            "POST",
            "/api/decision",
            &format!("action=deny&feedback=Needs+tests&token={token}"),
        );
        assert!(response.starts_with("HTTP/1.1 200"));
        let decision = thread.join().expect("review thread").expect("decision");
        assert_eq!(
            decision,
            Decision {
                approved: false,
                feedback: "Needs tests".to_owned()
            }
        );
    }

    #[test]
    fn browser_reviewer_supports_versions_and_json_decisions() {
        let (opened_sender, opened_receiver) = mpsc::channel();
        let reviewer = BrowserReviewer {
            open_browser: Some(Arc::new(move |target: &str| {
                opened_sender
                    .send(target.to_owned())
                    .map_err(|error| io::Error::other(error.to_string()))
            })),
            poll_interval: Duration::from_millis(2),
            ..BrowserReviewer::default()
        };
        let cancellation = agent::CancellationToken::default();
        let waiting = cancellation.clone();
        let thread = thread::spawn(move || {
            reviewer.review_version(
                &waiting,
                "Review",
                "# New\n- [ ] change",
                "# Old\n- [ ] old",
            )
        });
        let target = opened_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("browser URL");
        let page = response_body(&http_request(&target, "GET", "/", ""));
        assert!(page.contains("± Changes from previous submission"));
        assert!(page.contains("diff-row remove"));
        assert!(page.contains("diff-row add"));
        let token = review_token(&page);
        let body = format!(r#"{{"token":"{token}","approved":true,"feedback":"  note  "}}"#);
        let response = http_request(&target, "POST", "/api/decision.json", &body);
        assert!(response.starts_with("HTTP/1.1 204"));
        assert_eq!(
            thread.join().expect("review thread").expect("decision"),
            Decision {
                approved: true,
                feedback: "note".to_owned()
            }
        );
    }

    #[test]
    fn rendered_review_javascript_parses() {
        let node = Command::new("node").arg("--version").output();
        if node.is_err() {
            return;
        }
        let root = Scratch::new();
        let page = render_review_page(
            "Review",
            "# Plan `quoted`\n- [ ] don't break \\ paths\n<script>escaped</script>",
            "# Earlier",
            "token",
        );
        let script = page
            .split_once("<script>")
            .and_then(|(_, rest)| rest.split_once("</script>").map(|(script, _)| script))
            .expect("rendered script");
        let path = root.0.join("review.js");
        fs::write(&path, script).expect("write JavaScript");
        let output = Command::new("node")
            .arg("--check")
            .arg(&path)
            .output()
            .expect("run node");
        assert!(
            output.status.success(),
            "review JavaScript does not parse: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn browser_reviewer_rejects_non_loopback_hosts_and_honors_cancellation() {
        let cancellation = agent::CancellationToken::default();
        let invalid = BrowserReviewer {
            host: "0.0.0.0".to_owned(),
            ..BrowserReviewer::default()
        }
        .review(&cancellation, "x", "y");
        assert!(matches!(
            invalid,
            Err(ReviewError::Failed(message)) if message.contains("loopback")
        ));

        let cancelled = agent::CancellationToken::default();
        cancelled.cancel();
        assert_eq!(
            BrowserReviewer::default().review(&cancelled, "x", "y"),
            Err(ReviewError::Cancelled)
        );
    }

    #[test]
    fn local_annotation_collection_is_sorted_bounded_and_skips_noise() {
        let root = Scratch::new();
        root.write("docs/z.txt", "z");
        root.write("docs/a.md", "a");
        root.write("docs/code.rs", "not included");
        root.write("docs/node_modules/dependency.md", "not included");
        root.write("docs/.hidden/secret.md", "not included");
        let fetcher = fake_fetcher(UrlTextResponse {
            status: 200,
            final_url: Url::parse("https://example.test").expect("url"),
            content_type: None,
            body: Vec::new(),
        });
        let collector = TextCollector::with_fetcher(&root.0, fetcher)
            .expect("collector")
            .with_options(CollectionOptions {
                max_bytes: 200,
                max_files: 10,
                max_entries: 100,
                ..CollectionOptions::default()
            });
        let collected = collector.collect("docs").expect("collect folder");
        assert_eq!(collected.files, 2);
        assert!(collected.text.contains("## a.md"));
        assert!(collected.text.contains("## z.txt"));
        assert!(!collected.text.contains("not included"));
        assert!(
            collected.text.find("## a.md") < collected.text.find("## z.txt"),
            "directory output must be deterministic"
        );

        root.write("large.txt", "12345");
        let bounded = TextCollector::with_fetcher(
            &root.0,
            fake_fetcher(UrlTextResponse {
                status: 200,
                final_url: Url::parse("https://example.test").expect("url"),
                content_type: None,
                body: Vec::new(),
            }),
        )
        .expect("collector")
        .with_options(CollectionOptions {
            max_bytes: 4,
            ..CollectionOptions::default()
        });
        assert!(matches!(
            bounded.collect("large.txt"),
            Err(CollectionError::TooLarge { limit: 4 })
        ));
    }

    #[test]
    fn local_annotation_collection_confines_paths_and_limits_folder_files() {
        let root = Scratch::new();
        let outside = Scratch::new();
        outside.write("outside.txt", "secret");
        root.write("docs/a.md", "a");
        root.write("docs/b.md", "b");
        let collector = TextCollector::with_fetcher(
            &root.0,
            fake_fetcher(UrlTextResponse {
                status: 200,
                final_url: Url::parse("https://example.test").expect("url"),
                content_type: None,
                body: Vec::new(),
            }),
        )
        .expect("collector")
        .with_options(CollectionOptions {
            max_bytes: 100,
            max_files: 1,
            max_entries: 100,
            ..CollectionOptions::default()
        });
        assert!(matches!(
            collector.collect("../outside.txt"),
            Err(CollectionError::UnsafePath(_))
        ));
        assert!(matches!(
            collector.collect("docs"),
            Err(CollectionError::TooManyFiles { limit: 1 })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn local_annotation_collection_refuses_symlinked_targets() {
        use std::os::unix::fs::symlink;

        let root = Scratch::new();
        let outside = Scratch::new();
        outside.write("secret.md", "secret");
        symlink(outside.0.join("secret.md"), root.0.join("linked.md")).expect("symlink");
        let collector = TextCollector::with_fetcher(
            &root.0,
            fake_fetcher(UrlTextResponse {
                status: 200,
                final_url: Url::parse("https://example.test").expect("url"),
                content_type: None,
                body: Vec::new(),
            }),
        )
        .expect("collector");
        assert!(matches!(
            collector.collect("linked.md"),
            Err(CollectionError::UnsafePath(_))
        ));
    }

    #[test]
    fn url_annotation_collection_checks_scheme_host_status_and_byte_budget() {
        let calls = Arc::new(AtomicUsize::new(0));
        let root = Scratch::new();
        let collector = TextCollector::with_fetcher(
            &root.0,
            Arc::new(FakeFetcher {
                response: Ok(UrlTextResponse {
                    status: 200,
                    final_url: Url::parse("https://example.test/final").expect("url"),
                    content_type: Some("text/plain".to_owned()),
                    body: b"annotate me".to_vec(),
                }),
                calls: calls.clone(),
            }),
        )
        .expect("collector");
        let collected = collector
            .collect("https://example.test/start")
            .expect("url collection");
        assert_eq!(collected.text, "annotate me");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(matches!(
            collector.collect("file:///tmp/secret"),
            Err(CollectionError::InvalidUrl(_))
        ));
        assert!(matches!(
            collector.collect("http://127.0.0.1/private"),
            Err(CollectionError::InvalidUrl(_))
        ));
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let root = Scratch::new();
        let status = TextCollector::with_fetcher(
            &root.0,
            fake_fetcher(UrlTextResponse {
                status: 404,
                final_url: Url::parse("https://example.test/missing").expect("url"),
                content_type: None,
                body: Vec::new(),
            }),
        )
        .expect("collector");
        assert!(matches!(
            status.collect("https://example.test/missing"),
            Err(CollectionError::HttpStatus { status: 404, .. })
        ));
    }

    #[test]
    fn last_assistant_collection_uses_existing_llm_message_types() {
        let messages = vec![
            llm::Message::User(llm::UserMessage::text("user", 1)),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![
                    llm::ContentBlock::text("answer"),
                    llm::ContentBlock::ToolCall(tool_call("read", Some("a.md"))),
                ],
                ..llm::AssistantMessage::default()
            })),
        ];
        let collected = collect_last_assistant_response(&messages).expect("last assistant");
        assert_eq!(collected.text, "answer [read]");
        assert_eq!(collected.origin, TextOrigin::AssistantMessage);
        assert!(matches!(
            collect_last_assistant_response(&[]),
            Err(CollectionError::NoAssistantResponse)
        ));
    }

    fn fake_fetcher(response: UrlTextResponse) -> Arc<dyn UrlTextFetcher> {
        Arc::new(FakeFetcher {
            response: Ok(response),
            calls: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn http_request(target: &str, method: &str, path: &str, body: &str) -> String {
        let url = Url::parse(target).expect("review URL");
        let host = url.host_str().expect("review host");
        let port = url.port_or_known_default().expect("review port");
        let mut stream = TcpStream::connect((host, port)).expect("connect review server");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout");
        let host_header = if host.contains(':') {
            format!("[{host}]:{port}")
        } else {
            format!("{host}:{port}")
        };
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .expect("write request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        response
    }

    fn response_body(response: &str) -> String {
        response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body.to_owned())
            .expect("HTTP body")
    }

    fn review_token(page: &str) -> String {
        page.split_once(r#"name="token" value=""#)
            .and_then(|(_, rest)| rest.split_once('"').map(|(token, _)| token.to_owned()))
            .expect("review token")
    }
}
