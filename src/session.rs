//! Session lifecycle glue for the agent runtime.
//!
//! This module deliberately sits above [`crate::sessionlog`]: the log owns
//! the pi-compatible file format, while `SessionRuntime` owns selection,
//! restoration, and the subscription that turns agent events into durable
//! entries. It has no catalog, tool-discovery, or resource dependencies.
//! Callers provide models and callbacks through [`SessionOptions`].

use std::{
    collections::BTreeMap,
    error::Error as StdError,
    fmt, fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::{
    agent, config, llm,
    sessionlog::{self, Entry, Header, LoadReport, SessionError, SessionInfo, Store, Tree, Writer},
};

/// Resolves a model recorded in a session without coupling session handling to
/// a generated catalog. Returning `Ok(None)` asks the runtime to use its
/// configured fallback model instead.
pub type ModelResolver = Arc<
    dyn Fn(&str, &str) -> std::result::Result<Option<llm::Model>, String> + Send + Sync + 'static,
>;

/// Receives recoverable session notices, such as a repaired log tail or a
/// recorder that stopped after an I/O failure.
pub type SessionNoticeCallback = Arc<dyn Fn(SessionNotice) + Send + Sync + 'static>;

/// The way an initial runtime chooses its session file.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SessionSelection {
    /// Start a fresh, writable session.
    #[default]
    New,
    /// Reopen the most recent session in this workspace, or start a new one.
    Continue,
    /// Resolve an existing path, exact ID, or unambiguous ID prefix. A missing
    /// bare ID creates a new session with that ID.
    Session(String),
    /// Copy a source session's selected path into a new writable session.
    Fork {
        /// A path, exact ID, or unambiguous ID prefix for the source session.
        source: String,
        /// Optional entry ID at which to stop copying the source branch.
        at: Option<String>,
    },
    /// Run the agent without creating, opening, or recording a session.
    NoSession,
}

/// Construction inputs for [`SessionRuntime`].
///
/// `model`, `tools`, and `responder` form the agent's initial state. If a
/// resumed session recorded a model, `available_models` and `model_resolver`
/// are consulted unless `model_is_explicit` is set. This leaves model catalog
/// ownership with the later chat integration.
#[derive(Clone)]
pub struct SessionOptions {
    /// Workspace associated with the session shard and session header.
    pub cwd: PathBuf,
    /// Overrides [`config::sessions_dir`]. Supplying this makes embedding and
    /// temporary-directory tests independent from process environment state.
    pub sessions_dir: Option<PathBuf>,
    /// Selects a new, existing, continued, forked, or absent session.
    pub selection: SessionSelection,
    /// Open a selected existing session without an exclusive writer claim.
    pub read_only: bool,
    /// Optional display name to append after opening a writable session.
    pub name: Option<String>,

    /// Initial agent system prompt.
    pub system_prompt: String,
    /// Fallback model, used for new sessions and when a recorded model cannot
    /// be resolved by the caller.
    pub model: llm::Model,
    /// Preserves an intentional model override instead of applying a resumed
    /// model to a newly constructed agent.
    pub model_is_explicit: bool,
    /// Models already known to the caller. They are checked before
    /// `model_resolver`.
    pub available_models: Vec<llm::Model>,
    /// Optional catalog-independent lookup for a model recorded in a session.
    pub model_resolver: Option<ModelResolver>,
    /// Fallback thinking level for a new session.
    pub thinking_level: llm::ThinkingLevel,
    /// Initial tools for a newly constructed agent.
    pub tools: Vec<agent::Tool>,
    /// Provider callback for a newly constructed agent.
    pub responder: Option<agent::AssistantResponder>,
    pub steering_mode: agent::QueueMode,
    pub follow_up_mode: agent::QueueMode,
    pub tool_execution: agent::ToolExecutionMode,
    /// Optional non-UI notice delivery callback.
    pub on_notice: Option<SessionNoticeCallback>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            sessions_dir: None,
            selection: SessionSelection::New,
            read_only: false,
            name: None,
            system_prompt: String::new(),
            model: llm::Model::default(),
            model_is_explicit: false,
            available_models: Vec::new(),
            model_resolver: None,
            thinking_level: llm::THINKING_OFF.to_owned(),
            tools: Vec::new(),
            responder: None,
            steering_mode: agent::QueueMode::OneAtATime,
            follow_up_mode: agent::QueueMode::OneAtATime,
            tool_execution: agent::ToolExecutionMode::Parallel,
            on_notice: None,
        }
    }
}

/// A non-fatal condition that a frontend may render once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionNotice {
    pub kind: String,
    pub text: String,
}

/// Cloneable, thread-safe access to the active session's custom-entry writer.
///
/// Integrations use this instead of retaining a `SessionRuntime` reference in
/// callbacks that can outlive a single terminal draw. It silently represents
/// a no-session runtime through [`Self::recording`], while actual active-write
/// failures retain the normal recorder diagnostics.
#[derive(Clone)]
pub struct SessionCustomRecorder {
    recorder: Recorder,
}

impl SessionCustomRecorder {
    /// Returns whether a writable session log is currently available.
    pub fn recording(&self) -> bool {
        self.recorder.recording() && !self.recorder.read_only()
    }

    /// Appends one pi-compatible custom entry.
    pub fn record(&self, custom_type: impl Into<String>, data: Value) -> Result<String> {
        self.recorder.append(Entry {
            kind: sessionlog::TYPE_CUSTOM.to_owned(),
            custom_type: custom_type.into(),
            data: Some(data),
            ..Entry::default()
        })
    }
}

/// Cloneable notice sink for integration callbacks.
///
/// Background integrations must not write directly to stderr while Ratatui
/// owns the terminal. They enqueue notices here and the active frontend drains
/// them alongside ordinary session notices.
#[derive(Clone)]
pub struct SessionNoticeSender {
    notices: NoticeSink,
}

impl SessionNoticeSender {
    pub fn push(&self, kind: impl Into<String>, text: impl Into<String>) {
        self.notices.push(kind, text);
    }
}

/// Identifies the current session without leaking its writer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionHandle {
    pub id: String,
    pub path: PathBuf,
    pub read_only: bool,
}

/// A selectable user-message boundary on the current tree path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BranchPoint {
    /// One-based position used by chat commands and UI pickers.
    pub index: usize,
    pub id: String,
    pub text: String,
    pub label: Option<String>,
    pub children: usize,
    pub current: bool,
}

/// Metadata stored in a pi-compatible compaction entry.
///
/// Rust's current `llm::Message` model has no dedicated compaction marker, so
/// restored compactions are also represented as synthetic user messages in
/// [`RestoredSession::messages`]. The original fields remain available here
/// for future context accounting and UI work.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionSummary {
    pub summary: String,
    pub tokens_before: u64,
    pub cost_before: f64,
    pub retained_messages: usize,
    pub timestamp: i64,
}

impl Default for CompactionSummary {
    fn default() -> Self {
        Self {
            summary: String::new(),
            tokens_before: 0,
            cost_before: 0.0,
            retained_messages: 0,
            timestamp: 0,
        }
    }
}

/// The projection of one selected session branch back into agent state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RestoredSession {
    /// Agent-compatible context. Compaction markers become synthetic summary
    /// user messages before their retained messages.
    pub messages: Vec<llm::Message>,
    /// Provider recorded by the final explicit model change, or inferred from
    /// the latest assistant message when old logs have no model-change entry.
    pub provider: String,
    /// Model ID paired with [`Self::provider`].
    pub model_id: String,
    /// Last recorded thinking level on the complete selected path.
    pub thinking_level: llm::ThinkingLevel,
    /// Latest payload for every `customType` on the complete selected path.
    pub custom: BTreeMap<String, Value>,
    /// Compaction metadata on the projected context path.
    pub compactions: Vec<CompactionSummary>,
    /// Corrupt or unsupported message entries skipped during projection.
    pub warnings: Vec<String>,
}

impl RestoredSession {
    pub fn model_reference(&self) -> Option<(&str, &str)> {
        (!self.provider.is_empty() && !self.model_id.is_empty())
            .then_some((self.provider.as_str(), self.model_id.as_str()))
    }
}

/// Export representation produced by [`SessionRuntime::export`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExportFormat {
    Jsonl,
    Markdown,
}

/// Errors from runtime-level policy and the underlying session store.
#[derive(Debug)]
pub enum SessionRuntimeError {
    Session(SessionError),
    Io(std::io::Error),
    InvalidOptions(String),
    NotRecording,
    Busy(String),
    AlreadyCurrentSession,
}

impl fmt::Display for SessionRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Session(error) => error.fmt(formatter),
            Self::Io(error) => error.fmt(formatter),
            Self::InvalidOptions(message) => {
                write!(formatter, "invalid session options: {message}")
            }
            Self::NotRecording => formatter.write_str("this session is not being recorded"),
            Self::Busy(message) => formatter.write_str(message),
            Self::AlreadyCurrentSession => {
                formatter.write_str("that is already the current session")
            }
        }
    }
}

impl StdError for SessionRuntimeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Session(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::InvalidOptions(_)
            | Self::NotRecording
            | Self::Busy(_)
            | Self::AlreadyCurrentSession => None,
        }
    }
}

impl From<SessionError> for SessionRuntimeError {
    fn from(error: SessionError) -> Self {
        Self::Session(error)
    }
}

impl From<std::io::Error> for SessionRuntimeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, SessionRuntimeError>;

/// Owns an agent plus its session recorder subscription.
///
/// The recorder deliberately outlives writer swaps. The subscription captures
/// a stable `Recorder`, so `switch_to`, `clone_session`, and
/// `import_session` can replace only the writer without leaving future agent
/// events attached to a closed file.
pub struct SessionRuntime {
    store: Store,
    cwd: PathBuf,
    agent: agent::Agent,
    recorder: Recorder,
    subscription: Option<agent::Subscription>,
    notices: NoticeSink,
    model_selection: ModelSelection,
    resumed: bool,
}

impl SessionRuntime {
    /// Opens the selected log, restores it, constructs an agent, and subscribes
    /// recording before returning it to the caller.
    pub fn open(options: SessionOptions) -> Result<Self> {
        validate_options(&options)?;
        let store = Store::new(
            options
                .sessions_dir
                .clone()
                .unwrap_or_else(config::sessions_dir),
        );
        let opened = open_selected(&store, &options.cwd, &options)?;
        let mut model_notices = Vec::new();
        let model_selection = ModelSelection::from_options(&options);
        let model = model_selection.resolve(
            &opened.restored,
            options.model_is_explicit,
            &mut model_notices,
        );
        let thinking_level = restored_thinking(&options, &opened);
        let session_id = opened
            .writer
            .as_ref()
            .map(|writer| writer.id().to_owned())
            .unwrap_or_else(|| format!("goshcoder-{}", std::process::id()));
        let agent = agent::Agent::new(agent::AgentOptions {
            initial_state: agent::InitialState {
                system_prompt: options.system_prompt.clone(),
                model,
                thinking_level,
                tools: options.tools.clone(),
                messages: opened.restored.messages.clone(),
                compactions: opened
                    .restored
                    .compactions
                    .iter()
                    .map(compaction_info)
                    .collect(),
            },
            responder: options.responder.clone(),
            steering_mode: options.steering_mode,
            follow_up_mode: options.follow_up_mode,
            tool_execution: options.tool_execution,
            session_id,
        });
        Self::finish_open(
            agent,
            store,
            options,
            opened,
            model_notices,
            model_selection,
        )
    }

    /// Attaches session recording to an already constructed agent.
    ///
    /// Restoration happens before the event subscription, so loading a
    /// transcript never writes it into the log a second time. The supplied
    /// agent keeps its responder and tool configuration; the options' model
    /// lookup policy is still used to restore the recorded model.
    pub fn attach(agent: agent::Agent, options: SessionOptions) -> Result<Self> {
        validate_options(&options)?;
        let store = Store::new(
            options
                .sessions_dir
                .clone()
                .unwrap_or_else(config::sessions_dir),
        );
        let opened = open_selected(&store, &options.cwd, &options)?;
        let mut model_notices = Vec::new();
        let model_selection = ModelSelection::from_options(&options);
        let model = model_selection.resolve(
            &opened.restored,
            options.model_is_explicit,
            &mut model_notices,
        );
        agent.set_context(
            opened.restored.messages.clone(),
            opened
                .restored
                .compactions
                .iter()
                .map(compaction_info)
                .collect(),
        );
        agent.set_model(model);
        agent.set_thinking_level(restored_thinking(&options, &opened));
        Self::finish_open(
            agent,
            store,
            options,
            opened,
            model_notices,
            model_selection,
        )
    }

    fn finish_open(
        agent: agent::Agent,
        store: Store,
        options: SessionOptions,
        opened: OpenedSession,
        mut model_notices: Vec<String>,
        model_selection: ModelSelection,
    ) -> Result<Self> {
        let session_id = opened
            .writer
            .as_ref()
            .map(|writer| writer.id().to_owned())
            .unwrap_or_else(|| format!("goshcoder-{}", std::process::id()));
        agent.set_session_id(session_id);
        let notices = NoticeSink::new(options.on_notice.clone());
        for notice in opened.notices {
            notices.push("Session", notice);
        }
        for notice in model_notices.drain(..) {
            notices.push("Session", notice);
        }

        let recorder = Recorder::new(opened.writer, notices.clone());
        let subscription = recorder.has_writer().then(|| {
            let recorder = recorder.clone();
            agent.subscribe(move |event| bridge_agent_event(&recorder, event))
        });

        let runtime = Self {
            store,
            cwd: options.cwd,
            agent,
            recorder,
            subscription,
            notices,
            model_selection,
            resumed: opened.resumed,
        };
        if !runtime.resumed && runtime.recording() {
            runtime.record_initial_settings();
        }
        if let Some(name) = options.name {
            if runtime.recording() {
                runtime.set_name(name)?;
            } else if runtime.handle().is_some() {
                runtime.notices.push(
                    "Session",
                    "opened read-only; the requested session name was not saved",
                );
            }
        }
        Ok(runtime)
    }

    /// The runtime's agent. It is safe to clone [`agent::Agent`] for handles
    /// held by a UI; the recorder subscription remains owned by this runtime.
    pub fn agent(&self) -> &agent::Agent {
        &self.agent
    }

    /// Returns a callback-safe writer for integration-owned custom state.
    pub fn custom_recorder(&self) -> SessionCustomRecorder {
        SessionCustomRecorder {
            recorder: self.recorder.clone(),
        }
    }

    /// Returns a callback-safe notice sink for foreground and background
    /// integrations.
    pub fn notice_sender(&self) -> SessionNoticeSender {
        SessionNoticeSender {
            notices: self.notices.clone(),
        }
    }

    /// Returns whether opening selected an existing session branch.
    pub fn resumed(&self) -> bool {
        self.resumed
    }

    /// Returns the active session ID and path, if persistence is enabled.
    pub fn handle(&self) -> Option<SessionHandle> {
        self.recorder.handle()
    }

    pub fn id(&self) -> Option<String> {
        self.handle().map(|handle| handle.id)
    }

    pub fn path(&self) -> Option<PathBuf> {
        self.handle().map(|handle| handle.path)
    }

    pub fn read_only(&self) -> bool {
        self.recorder.read_only()
    }

    /// Returns true only while completed events can still append to disk.
    pub fn recording(&self) -> bool {
        self.recorder.recording()
    }

    pub fn header(&self) -> Option<Header> {
        self.recorder.header()
    }

    /// A point-in-time snapshot of the complete tree, including abandoned
    /// branches and metadata entries.
    pub fn tree(&self) -> Option<Tree> {
        self.recorder.snapshot()
    }

    /// Reprojects the current tree. It reflects resets, branches, labels, and
    /// direct compaction recording that happened after startup.
    pub fn restored(&self) -> RestoredSession {
        self.tree()
            .map(|tree| restore_from_tree(&tree, tree.leaf()))
            .unwrap_or_default()
    }

    pub fn name(&self) -> Option<String> {
        self.tree()
            .map(|tree| tree.name().to_owned())
            .filter(|name| !name.is_empty())
    }

    /// Uses the same name → first user message → ID fallback as session lists.
    pub fn title(&self) -> Option<String> {
        let handle = self.handle()?;
        let tree = self.tree()?;
        if !tree.name().is_empty() {
            return Some(tree.name().to_owned());
        }
        for entry in tree.path(tree.leaf()) {
            if entry.kind != sessionlog::TYPE_MESSAGE {
                continue;
            }
            let Some(message) = entry.message.as_ref() else {
                continue;
            };
            let Ok(llm::Message::User(message)) =
                serde_json::from_value::<llm::Message>(message.clone())
            else {
                continue;
            };
            let text = message_text(&message);
            if !text.is_empty() {
                return Some(first_line(&text, 120));
            }
        }
        Some(handle.id)
    }

    /// Takes startup, load, and fail-soft recording notices accumulated so far.
    pub fn drain_notices(&self) -> Vec<SessionNotice> {
        self.notices.drain()
    }

    /// Writes a `session_info` entry and flushes it so names survive an
    /// immediate process exit.
    pub fn set_name(&self, name: impl Into<String>) -> Result<()> {
        let name = name.into().trim().to_owned();
        self.recorder.append(Entry {
            kind: sessionlog::TYPE_SESSION_INFO.to_owned(),
            name,
            ..Entry::default()
        })?;
        self.sync()
    }

    /// Flushes a writable writer. Read-only and no-session runs have nothing
    /// to flush and return successfully.
    pub fn sync(&self) -> Result<()> {
        self.recorder.sync()
    }

    /// Records extension state in pi's `custom` entry slot.
    pub fn record_custom(&self, custom_type: impl Into<String>, data: Value) -> Result<String> {
        self.recorder.append(Entry {
            kind: sessionlog::TYPE_CUSTOM.to_owned(),
            custom_type: custom_type.into(),
            data: Some(data),
            ..Entry::default()
        })
    }

    /// Writes a compaction marker and its retained messages in pi-compatible
    /// order, then flushes the completed cut.
    ///
    /// Normal callers should prefer [`agent::Agent::compact`], whose lifecycle
    /// event reaches the same recorder automatically.
    pub fn record_compaction(
        &self,
        summary: CompactionSummary,
        retained: &[llm::Message],
    ) -> Result<()> {
        self.recorder.append_compaction(summary, retained)
    }

    /// Lists only user-message branch boundaries, because continuing from the
    /// middle of an assistant/tool turn can create invalid provider context.
    pub fn branch_points(&self) -> Vec<BranchPoint> {
        let Some(tree) = self.tree() else {
            return Vec::new();
        };
        branch_points(&tree, tree.leaf())
    }

    /// Rewinds the write head to a user-message branch point while preserving
    /// the abandoned path in the JSONL file.
    pub fn fork_to(&self, index: usize) -> Result<BranchPoint> {
        self.require_recording()?;
        self.require_idle("wait for the current response to finish before rewinding")?;
        let Some(tree) = self.tree() else {
            return Err(SessionRuntimeError::NotRecording);
        };
        let points = branch_points(&tree, tree.leaf());
        let target = points
            .get(index.checked_sub(1).ok_or_else(|| {
                SessionRuntimeError::InvalidOptions(
                    "choose a branch point starting at 1".to_owned(),
                )
            })?)
            .cloned()
            .ok_or_else(|| {
                SessionRuntimeError::InvalidOptions(format!(
                    "choose a branch point between 1 and {}",
                    points.len()
                ))
            })?;
        let old_path_len = tree.path(tree.leaf()).len();
        let target_path_len = tree.path(Some(&target.id)).len();
        let abandoned = old_path_len.saturating_sub(target_path_len);

        self.recorder.mutate(|writer| {
            writer.set_leaf(target.id.clone())?;
            if abandoned > 0 {
                writer.append(Entry {
                    kind: sessionlog::TYPE_BRANCH_SUMMARY.to_owned(),
                    from_id: target.id.clone(),
                    summary: format!(
                        "branched from {:?}, leaving {abandoned} entries on the previous path",
                        target.text
                    ),
                    ..Entry::default()
                })?;
            }
            writer.sync()
        })?;
        let restored = self.restored();
        let compactions = restored.compactions.iter().map(compaction_info).collect();
        self.agent.set_context(restored.messages, compactions);
        Ok(target)
    }

    /// Associates a label with a branch point. An empty label removes it.
    pub fn label(&self, index: usize, label: impl Into<String>) -> Result<()> {
        self.require_recording()?;
        let label = label.into().trim().to_owned();
        let points = self.branch_points();
        let target = points
            .get(index.checked_sub(1).ok_or_else(|| {
                SessionRuntimeError::InvalidOptions(
                    "choose a branch point starting at 1".to_owned(),
                )
            })?)
            .ok_or_else(|| {
                SessionRuntimeError::InvalidOptions(format!(
                    "choose a branch point between 1 and {}",
                    points.len()
                ))
            })?;
        self.recorder.append(Entry {
            kind: sessionlog::TYPE_LABEL.to_owned(),
            target_id: target.id.clone(),
            label: (!label.is_empty()).then_some(label),
            ..Entry::default()
        })?;
        self.sync()
    }

    pub fn clear_label(&self, index: usize) -> Result<()> {
        self.label(index, "")
    }

    /// Copies the current selected branch into a new session and makes that
    /// copy the target of the existing stable recorder subscription.
    pub fn clone_session(&self) -> Result<SessionHandle> {
        self.require_recording()?;
        self.require_idle("wait for the current response to finish before cloning")?;
        self.sync()?;
        let current = self.handle().ok_or(SessionRuntimeError::NotRecording)?;
        let source = self
            .store
            .resolve(&self.cwd, current.path.to_string_lossy().as_ref())?;
        let leaf = self.recorder.leaf();
        let writer = self.store.fork(&source, leaf.as_deref(), &self.cwd)?;
        self.adopt_writer(writer, "cloned session")
    }

    /// Attaches another writable v3 session without closing the current log
    /// until the target has opened successfully.
    pub fn switch_to(&self, reference: &str) -> Result<SessionHandle> {
        self.require_recording()?;
        self.require_idle("wait for the current response to finish before switching sessions")?;
        let current = self.handle().ok_or(SessionRuntimeError::NotRecording)?;
        let info = self.store.resolve(&self.cwd, reference)?;
        if info.path == current.path {
            return Err(SessionRuntimeError::AlreadyCurrentSession);
        }
        let (writer, report) = self.store.attach(&info.path)?;
        let handle = self.adopt_writer(writer, "switched session")?;
        for notice in report_notices(&report) {
            self.notices.push("Session", notice);
        }
        Ok(handle)
    }

    /// Copies an external or otherwise selected session into this runtime's
    /// workspace and adopts the writable copy.
    pub fn import_session(&self, source: &str) -> Result<SessionHandle> {
        self.require_recording()?;
        self.require_idle("wait for the current response to finish before importing")?;
        self.sync()?;
        if self.handle().is_none() {
            return Err(SessionRuntimeError::NotRecording);
        }
        let source = self.store.resolve(&self.cwd, source)?;
        let writer = self.store.fork(&source, None, &self.cwd)?;
        self.adopt_writer(writer, "imported session")
    }

    /// Copies a session into this runtime's workspace without changing the
    /// active recorder or the live conversation.
    ///
    /// This is the interactive `/import` behavior: importing makes a durable
    /// session available for a later `/resume`, rather than replacing the
    /// conversation that issued the command.
    pub fn import_copy(&self, source: &str) -> Result<SessionHandle> {
        let source = self.store.resolve(&self.cwd, source)?;
        let mut writer = self.store.fork(&source, None, &self.cwd)?;
        let handle = SessionHandle {
            id: writer.id().to_owned(),
            path: writer.path().to_path_buf(),
            read_only: writer.read_only(),
        };
        writer.close()?;
        Ok(handle)
    }

    /// Returns the raw JSONL or a readable Markdown rendering of the current
    /// projected branch.
    pub fn export(&self, format: ExportFormat) -> Result<Vec<u8>> {
        self.sync()?;
        let handle = self.handle().ok_or(SessionRuntimeError::NotRecording)?;
        match format {
            ExportFormat::Jsonl => Ok(fs::read(handle.path)?),
            ExportFormat::Markdown => {
                let (tree, header, _) = self.store.load(handle.path)?;
                Ok(export_markdown(&tree, &header).into_bytes())
            }
        }
    }

    /// Writes an export to a caller-selected destination.
    pub fn export_to(&self, format: ExportFormat, destination: impl AsRef<Path>) -> Result<()> {
        fs::write(destination, self.export(format)?)?;
        Ok(())
    }

    /// Stops recording, releases any exclusive claim, and removes the listener
    /// from the agent. The agent itself remains usable in memory.
    pub fn close(&mut self) -> Result<()> {
        self.subscription.take();
        self.recorder.close()
    }

    fn require_idle(&self, message: &str) -> Result<()> {
        if self.agent.state().is_streaming {
            return Err(SessionRuntimeError::Busy(message.to_owned()));
        }
        Ok(())
    }

    fn require_recording(&self) -> Result<()> {
        if self.recording() {
            Ok(())
        } else {
            Err(SessionRuntimeError::NotRecording)
        }
    }

    fn record_initial_settings(&self) {
        let state = self.agent.state();
        let _ = self.recorder.mutate(|writer| {
            if !state.model.provider.is_empty() && !state.model.id.is_empty() {
                writer.append(Entry {
                    kind: sessionlog::TYPE_MODEL_CHANGE.to_owned(),
                    provider: state.model.provider,
                    model_id: state.model.id,
                    ..Entry::default()
                })?;
            }
            if !state.thinking_level.is_empty() {
                writer.append(Entry {
                    kind: sessionlog::TYPE_THINKING_LEVEL_CHANGE.to_owned(),
                    thinking_level: state.thinking_level,
                    ..Entry::default()
                })?;
            }
            writer.sync()
        });
    }

    fn adopt_writer(&self, writer: Writer, action: &str) -> Result<SessionHandle> {
        let restored = restore_from_tree(&writer.snapshot(), writer.leaf());
        let mut model_notices = Vec::new();
        let model = self
            .model_selection
            .resolve(&restored, false, &mut model_notices);
        let thinking_level = if restored.thinking_level.is_empty() {
            self.agent.state().thinking_level
        } else {
            restored.thinking_level.clone()
        };

        // Set the transcript before the swap. `set_context` is intentionally
        // event-free, so the current writer cannot receive duplicate entries.
        let compactions = restored.compactions.iter().map(compaction_info).collect();
        self.agent.set_context(restored.messages, compactions);
        let handle = SessionHandle {
            id: writer.id().to_owned(),
            path: writer.path().to_path_buf(),
            read_only: writer.read_only(),
        };
        let previous = self.recorder.swap(writer);
        self.agent.set_session_id(handle.id.clone());
        if let Some(mut previous) = previous
            && let Err(error) = previous.close()
        {
            self.notices.push(
                "Session",
                format!("the previous session did not close cleanly: {error}"),
            );
        }

        // These events now belong to the new writer. If restoring required a
        // fallback model, recording that transition makes the next resume
        // reflect the model the agent actually runs.
        self.agent.set_model(model);
        self.agent.set_thinking_level(thinking_level);
        for warning in restored.warnings {
            self.notices.push("Session", warning);
        }
        for notice in model_notices {
            self.notices.push("Session", notice);
        }
        self.notices
            .push("Session", format!("{action} {}", handle.id));
        Ok(handle)
    }
}

impl Drop for SessionRuntime {
    fn drop(&mut self) {
        self.subscription.take();
        let _ = self.recorder.close();
    }
}

/// Folds a selected tree branch into agent-compatible context and retained
/// session settings. It is public so a frontend can inspect a session before
/// attaching it to an agent.
pub fn restore_from_tree(tree: &Tree, leaf: Option<&str>) -> RestoredSession {
    let mut restored = RestoredSession::default();
    let mut assistant_fallback = None;
    let mut explicit_model = None;

    for entry in tree.path(leaf) {
        match entry.kind.as_str() {
            sessionlog::TYPE_MODEL_CHANGE
                if !entry.provider.is_empty() && !entry.model_id.is_empty() =>
            {
                explicit_model = Some((entry.provider.clone(), entry.model_id.clone()));
            }
            sessionlog::TYPE_THINKING_LEVEL_CHANGE => {
                restored.thinking_level = entry.thinking_level.clone();
            }
            sessionlog::TYPE_CUSTOM if !entry.custom_type.is_empty() => {
                if let Some(data) = entry.data.clone() {
                    restored.custom.insert(entry.custom_type.clone(), data);
                }
            }
            sessionlog::TYPE_MESSAGE => {
                if let Some(message) = entry.message.as_ref()
                    && let Some(model) = assistant_model(message)
                {
                    assistant_fallback = Some(model);
                }
            }
            _ => {}
        }
    }
    if let Some((provider, model_id)) = explicit_model.or(assistant_fallback) {
        restored.provider = provider;
        restored.model_id = model_id;
    }

    for entry in tree.context_path(leaf) {
        match entry.kind.as_str() {
            sessionlog::TYPE_MESSAGE => {
                let Some(value) = entry.message.clone() else {
                    restored
                        .warnings
                        .push(format!("entry {} has no message and was skipped", entry.id));
                    continue;
                };
                match serde_json::from_value::<llm::Message>(value) {
                    Ok(message) => restored.messages.push(message),
                    Err(error) => restored.warnings.push(format!(
                        "entry {} could not be read back and was skipped: {error}",
                        entry.id
                    )),
                }
            }
            sessionlog::TYPE_COMPACTION => {
                let summary = compaction_summary_from_entry(entry);
                restored.messages.push(compaction_context_message(&summary));
                restored.compactions.push(summary);
            }
            _ => {}
        }
    }
    restored
}

#[derive(Clone)]
struct ModelSelection {
    fallback: llm::Model,
    available: Vec<llm::Model>,
    resolver: Option<ModelResolver>,
}

impl ModelSelection {
    fn from_options(options: &SessionOptions) -> Self {
        Self {
            fallback: options.model.clone(),
            available: options.available_models.clone(),
            resolver: options.model_resolver.clone(),
        }
    }

    fn resolve(
        &self,
        restored: &RestoredSession,
        explicit_override: bool,
        notices: &mut Vec<String>,
    ) -> llm::Model {
        if explicit_override {
            if let Some((provider, model_id)) = restored.model_reference()
                && (self.fallback.provider != provider || self.fallback.id != model_id)
            {
                notices.push(format!(
                    "this session was held on {provider}/{model_id}; continuing on {}/{} because the configured model was explicit",
                    self.fallback.provider, self.fallback.id
                ));
            }
            return self.fallback.clone();
        }
        let Some((provider, model_id)) = restored.model_reference() else {
            return self.fallback.clone();
        };
        if let Some(model) = self
            .available
            .iter()
            .find(|model| model.provider == provider && model.id == model_id)
        {
            return model.clone();
        }
        if let Some(resolve) = &self.resolver {
            match resolve(provider, model_id) {
                Ok(Some(model)) => return model,
                Ok(None) => {}
                Err(error) => notices.push(format!(
                    "the recorded model {provider}/{model_id} could not be resolved: {error}; using the configured fallback"
                )),
            }
        }
        if self.fallback.provider == provider && self.fallback.id == model_id {
            return self.fallback.clone();
        }
        notices.push(format!(
            "this session was held on {provider}/{model_id}, which is not available here; using {}{}",
            self.fallback.provider,
            if self.fallback.id.is_empty() {
                String::new()
            } else {
                format!("/{}", self.fallback.id)
            }
        ));
        self.fallback.clone()
    }
}

struct OpenedSession {
    writer: Option<Writer>,
    restored: RestoredSession,
    notices: Vec<String>,
    resumed: bool,
}

fn validate_options(options: &SessionOptions) -> Result<()> {
    match &options.selection {
        SessionSelection::NoSession => {
            if options.read_only {
                return Err(SessionRuntimeError::InvalidOptions(
                    "read_only has no effect with no-session".to_owned(),
                ));
            }
            if options
                .name
                .as_deref()
                .is_some_and(|name| !name.trim().is_empty())
            {
                return Err(SessionRuntimeError::InvalidOptions(
                    "a no-session run cannot be named".to_owned(),
                ));
            }
        }
        SessionSelection::New | SessionSelection::Fork { .. } if options.read_only => {
            return Err(SessionRuntimeError::InvalidOptions(
                "read_only requires selecting an existing session".to_owned(),
            ));
        }
        _ => {}
    }
    Ok(())
}

fn open_selected(store: &Store, cwd: &Path, options: &SessionOptions) -> Result<OpenedSession> {
    match &options.selection {
        SessionSelection::NoSession => Ok(OpenedSession {
            writer: None,
            restored: RestoredSession::default(),
            notices: Vec::new(),
            resumed: false,
        }),
        SessionSelection::New => Ok(opened_new(store.create(cwd)?)),
        SessionSelection::Continue => match store.most_recent(cwd)? {
            Some(info) => open_target(store, cwd, info, options.read_only),
            None => {
                let mut opened = opened_new(store.create(cwd)?);
                opened
                    .notices
                    .push("no previous session for this workspace; starting a new one".to_owned());
                Ok(opened)
            }
        },
        SessionSelection::Session(reference) => match store.resolve(cwd, reference) {
            Ok(info) => open_target(store, cwd, info, options.read_only),
            Err(SessionError::NotFound(_)) if !looks_like_path(reference) => {
                Ok(opened_new(store.create_with_id(cwd, None, reference)?))
            }
            Err(error) => Err(error.into()),
        },
        SessionSelection::Fork { source, at } => {
            let source = store.resolve(cwd, source)?;
            let writer = store.fork(&source, at.as_deref(), cwd)?;
            let mut opened = opened_existing(writer, LoadReport::default());
            opened
                .notices
                .push(format!("forked session {}", source.short_id()));
            Ok(opened)
        }
    }
}

fn opened_new(writer: Writer) -> OpenedSession {
    OpenedSession {
        writer: Some(writer),
        restored: RestoredSession::default(),
        notices: Vec::new(),
        resumed: false,
    }
}

fn opened_existing(writer: Writer, report: LoadReport) -> OpenedSession {
    let restored = restore_from_tree(&writer.snapshot(), writer.leaf());
    let mut notices = report_notices(&report);
    notices.extend(restored.warnings.iter().cloned());
    OpenedSession {
        writer: Some(writer),
        restored,
        notices,
        resumed: true,
    }
}

fn open_target(
    store: &Store,
    cwd: &Path,
    target: SessionInfo,
    read_only: bool,
) -> Result<OpenedSession> {
    if read_only {
        let (writer, report) = store.open(&target.path)?;
        let mut opened = opened_existing(writer, report);
        opened
            .notices
            .push("opened read-only; this session is not being saved".to_owned());
        return Ok(opened);
    }

    match store.attach(&target.path) {
        Ok((writer, report)) => Ok(opened_existing(writer, report)),
        Err(SessionError::LegacyFormat(version)) => {
            let (_, _, report) = store.load(&target.path)?;
            let writer = store.fork(&target, None, cwd)?;
            let mut opened = opened_existing(writer, report);
            opened.notices.push(format!(
                "{} is an older pi session (v{version}); it was copied into a new session so it can be continued",
                target.short_id()
            ));
            Ok(opened)
        }
        Err(SessionError::Busy(owner)) => {
            let (writer, report) = store.open(&target.path)?;
            let mut opened = opened_existing(writer, report);
            opened.notices.push(format!(
                "{} is open in another process ({owner}); continuing read-only, so this conversation is not being saved",
                target.short_id()
            ));
            Ok(opened)
        }
        Err(error) => Err(error.into()),
    }
}

fn restored_thinking(options: &SessionOptions, opened: &OpenedSession) -> llm::ThinkingLevel {
    if opened.resumed && !opened.restored.thinking_level.is_empty() {
        opened.restored.thinking_level.clone()
    } else {
        options.thinking_level.clone()
    }
}

fn looks_like_path(reference: &str) -> bool {
    reference.ends_with(".jsonl") || reference.contains(['/', '\\'])
}

fn report_notices(report: &LoadReport) -> Vec<String> {
    let mut notices = Vec::new();
    if report.repaired_tail {
        notices.push(
            "the session ended mid-write and its last incomplete entry was dropped".to_owned(),
        );
    }
    if report.skipped_lines > 0 {
        notices.push(format!(
            "{} unreadable line(s) were skipped: {}",
            report.skipped_lines,
            report.warnings.join("; ")
        ));
    } else {
        notices.extend(report.warnings.iter().cloned());
    }
    notices
}

#[derive(Clone)]
struct NoticeSink {
    notices: Arc<Mutex<Vec<SessionNotice>>>,
    callback: Option<SessionNoticeCallback>,
}

impl NoticeSink {
    fn new(callback: Option<SessionNoticeCallback>) -> Self {
        Self {
            notices: Arc::new(Mutex::new(Vec::new())),
            callback,
        }
    }

    fn push(&self, kind: impl Into<String>, text: impl Into<String>) {
        let notice = SessionNotice {
            kind: kind.into(),
            text: text.into(),
        };
        {
            let mut notices = lock(&self.notices);
            if notices.len() >= 64 {
                notices.remove(0);
            }
            notices.push(notice.clone());
        }
        if let Some(callback) = &self.callback {
            callback(notice);
        }
    }

    fn drain(&self) -> Vec<SessionNotice> {
        std::mem::take(&mut *lock(&self.notices))
    }
}

#[derive(Clone)]
struct Recorder {
    state: Arc<Mutex<RecorderState>>,
    notices: NoticeSink,
}

struct RecorderState {
    writer: Option<Writer>,
    reported: bool,
}

impl Recorder {
    fn new(writer: Option<Writer>, notices: NoticeSink) -> Self {
        Self {
            state: Arc::new(Mutex::new(RecorderState {
                writer,
                reported: false,
            })),
            notices,
        }
    }

    fn has_writer(&self) -> bool {
        lock(&self.state).writer.is_some()
    }

    fn handle(&self) -> Option<SessionHandle> {
        lock(&self.state)
            .writer
            .as_ref()
            .map(|writer| SessionHandle {
                id: writer.id().to_owned(),
                path: writer.path().to_path_buf(),
                read_only: writer.read_only(),
            })
    }

    fn header(&self) -> Option<Header> {
        lock(&self.state)
            .writer
            .as_ref()
            .map(|writer| writer.header().clone())
    }

    fn snapshot(&self) -> Option<Tree> {
        lock(&self.state).writer.as_ref().map(Writer::snapshot)
    }

    fn leaf(&self) -> Option<String> {
        lock(&self.state)
            .writer
            .as_ref()
            .and_then(|writer| writer.leaf().map(str::to_owned))
    }

    fn read_only(&self) -> bool {
        lock(&self.state)
            .writer
            .as_ref()
            .is_some_and(Writer::read_only)
    }

    fn recording(&self) -> bool {
        lock(&self.state)
            .writer
            .as_ref()
            .is_some_and(Writer::recording)
    }

    fn append(&self, entry: Entry) -> Result<String> {
        self.mutate(|writer| writer.append(entry))
    }

    fn append_compaction(
        &self,
        summary: CompactionSummary,
        retained: &[llm::Message],
    ) -> Result<()> {
        self.mutate(|writer| {
            // The marker refers to an entry that already exists, so retained
            // messages are appended immediately before it. `context_path`
            // then reconstructs exactly summary + retained messages.
            let mut first_kept_entry_id = String::new();
            for (index, message) in retained.iter().enumerate() {
                let id = writer.append(Entry::message(message)?)?;
                if index == 0 {
                    first_kept_entry_id = id;
                }
            }
            let details = compaction_details_value(&summary)?;
            writer.append(Entry {
                kind: sessionlog::TYPE_COMPACTION.to_owned(),
                summary: summary.summary,
                first_kept_entry_id,
                tokens_before: summary.tokens_before,
                details: Some(details),
                ..Entry::default()
            })?;
            writer.sync()
        })
    }

    fn append_compaction_best_effort(&self, summary: CompactionSummary, retained: &[llm::Message]) {
        let _ = self.append_compaction(summary, retained);
    }

    fn mutate<T>(&self, operation: impl FnOnce(&mut Writer) -> sessionlog::Result<T>) -> Result<T> {
        let (result, report) = {
            let mut state = lock(&self.state);
            let Some(writer) = state.writer.as_mut() else {
                return Err(SessionRuntimeError::NotRecording);
            };
            if writer.read_only() {
                return Err(SessionError::ReadOnly.into());
            }
            if !writer.recording() {
                return Err(writer_error(writer));
            }
            let result = operation(writer);
            let report = result.as_ref().err().is_some_and(recording_failure);
            (result, report)
        };
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                if report {
                    self.report(error.to_string());
                }
                Err(error.into())
            }
        }
    }

    fn append_best_effort(&self, entry: Entry) {
        let _ = self.append(entry);
    }

    fn sync(&self) -> Result<()> {
        let (result, report) = {
            let mut state = lock(&self.state);
            let Some(writer) = state.writer.as_mut() else {
                return Ok(());
            };
            let result = writer.sync();
            let report = result.as_ref().err().is_some_and(recording_failure);
            (result, report)
        };
        match result {
            Ok(()) => Ok(()),
            Err(error) => {
                if report {
                    self.report(error.to_string());
                }
                Err(error.into())
            }
        }
    }

    fn sync_best_effort(&self) {
        let _ = self.sync();
    }

    fn swap(&self, writer: Writer) -> Option<Writer> {
        let mut state = lock(&self.state);
        state.reported = false;
        state.writer.replace(writer)
    }

    fn close(&self) -> Result<()> {
        let writer = lock(&self.state).writer.take();
        let Some(mut writer) = writer else {
            return Ok(());
        };
        writer.close().map_err(Into::into)
    }

    fn report(&self, error: String) {
        let should_report = {
            let mut state = lock(&self.state);
            if state.reported {
                false
            } else {
                state.reported = true;
                true
            }
        };
        if should_report {
            self.notices.push(
                "Session",
                format!("recording stopped: {error} (the conversation is still in memory)"),
            );
        }
    }
}

fn writer_error(writer: &Writer) -> SessionRuntimeError {
    match writer.degraded() {
        Some(reason) => SessionError::Degraded(reason.to_owned()).into(),
        None => SessionError::Closed.into(),
    }
}

fn recording_failure(error: &SessionError) -> bool {
    matches!(
        error,
        SessionError::Io(_)
            | SessionError::SessionTooLarge(_)
            | SessionError::Degraded(_)
            | SessionError::Closed
    )
}

fn bridge_agent_event(recorder: &Recorder, event: agent::Event) {
    match event.kind {
        agent::EventKind::MessageEnd => {
            if let Some(message) = event.message {
                match Entry::message(&message) {
                    Ok(entry) => recorder.append_best_effort(entry),
                    Err(error) => recorder.report(format!(
                        "could not encode a completed {} message for the session log: {error}",
                        message.role()
                    )),
                }
            }
        }
        agent::EventKind::ModelChange => recorder.append_best_effort(Entry {
            kind: sessionlog::TYPE_MODEL_CHANGE.to_owned(),
            provider: event.provider,
            model_id: event.model_id,
            ..Entry::default()
        }),
        agent::EventKind::ThinkingLevelChange => recorder.append_best_effort(Entry {
            kind: sessionlog::TYPE_THINKING_LEVEL_CHANGE.to_owned(),
            thinking_level: event.thinking_level,
            ..Entry::default()
        }),
        agent::EventKind::ContextCompacted => {
            if let Some(info) = event.compaction.as_ref() {
                recorder.append_compaction_best_effort(
                    CompactionSummary {
                        summary: info.summary.clone(),
                        tokens_before: info.tokens_before,
                        cost_before: info.cost_before,
                        retained_messages: info.retained_messages,
                        timestamp: info.timestamp,
                    },
                    &event.kept,
                );
            }
        }
        agent::EventKind::TranscriptReset => recorder.append_best_effort(Entry {
            kind: sessionlog::TYPE_TRANSCRIPT_RESET.to_owned(),
            reason: if event.reason.is_empty() {
                "reset".to_owned()
            } else {
                event.reason
            },
            ..Entry::default()
        }),
        agent::EventKind::AgentEnd => recorder.sync_best_effort(),
        agent::EventKind::AgentStart
        | agent::EventKind::TurnStart
        | agent::EventKind::TurnEnd
        | agent::EventKind::MessageStart
        | agent::EventKind::MessageUpdate
        | agent::EventKind::ToolExecutionStart
        | agent::EventKind::ToolExecutionUpdate
        | agent::EventKind::ToolExecutionEnd => {}
    }
}

fn branch_points(tree: &Tree, leaf: Option<&str>) -> Vec<BranchPoint> {
    let mut points = tree
        .path(leaf)
        .into_iter()
        .filter(|entry| entry.kind == sessionlog::TYPE_MESSAGE)
        .filter_map(|entry| {
            let message = entry.message.as_ref()?;
            let decoded = serde_json::from_value::<llm::Message>(message.clone()).ok()?;
            let llm::Message::User(message) = decoded else {
                return None;
            };
            Some(BranchPoint {
                index: 0,
                id: entry.id.clone(),
                text: first_line(&message_text(&message), 120),
                label: tree.label(&entry.id).map(str::to_owned),
                children: tree.children(Some(&entry.id)).len(),
                current: false,
            })
        })
        .collect::<Vec<_>>();
    for (index, point) in points.iter_mut().enumerate() {
        point.index = index + 1;
    }
    if let Some(point) = points.last_mut() {
        point.current = true;
    }
    points
}

fn assistant_model(value: &Value) -> Option<(String, String)> {
    let object = value.as_object()?;
    (object.get("role")?.as_str()? == "assistant").then_some(())?;
    let provider = object.get("provider")?.as_str()?.to_owned();
    let model = object.get("model")?.as_str()?.to_owned();
    (!provider.is_empty() && !model.is_empty()).then_some((provider, model))
}

#[derive(Default, Deserialize, Serialize)]
struct CompactionDetails {
    #[serde(default)]
    goshcoder: GoshCoderCompactionDetails,
}

#[derive(Default, Deserialize, Serialize)]
struct GoshCoderCompactionDetails {
    #[serde(rename = "costBefore", default)]
    cost_before: f64,
    #[serde(rename = "retainedMessages", default)]
    retained_messages: usize,
    #[serde(default, skip_serializing_if = "is_zero_i64")]
    timestamp: i64,
}

fn compaction_details_value(summary: &CompactionSummary) -> sessionlog::Result<Value> {
    Ok(serde_json::to_value(CompactionDetails {
        goshcoder: GoshCoderCompactionDetails {
            cost_before: summary.cost_before,
            retained_messages: summary.retained_messages,
            timestamp: summary.timestamp,
        },
    })?)
}

fn compaction_summary_from_entry(entry: &Entry) -> CompactionSummary {
    let details = entry
        .details
        .as_ref()
        .and_then(|details| serde_json::from_value::<CompactionDetails>(details.clone()).ok())
        .unwrap_or_default()
        .goshcoder;
    CompactionSummary {
        summary: entry.summary.clone(),
        tokens_before: entry.tokens_before,
        cost_before: details.cost_before,
        retained_messages: details.retained_messages,
        timestamp: if details.timestamp != 0 {
            details.timestamp
        } else {
            timestamp_millis(&entry.timestamp)
        },
    }
}

fn compaction_info(summary: &CompactionSummary) -> agent::CompactionInfo {
    agent::CompactionInfo {
        summary: summary.summary.clone(),
        tokens_before: summary.tokens_before,
        cost_before: summary.cost_before,
        retained_messages: summary.retained_messages,
        timestamp: summary.timestamp,
    }
}

fn compaction_context_message(summary: &CompactionSummary) -> llm::Message {
    llm::Message::User(llm::UserMessage::text(
        format!(
            "<conversation-summary>\n{}\n</conversation-summary>\n\nContinue from this summary and the recent conversation below. Do not treat the summary as a new request.",
            summary.summary.trim()
        ),
        summary.timestamp,
    ))
}

fn timestamp_millis(value: &str) -> i64 {
    OffsetDateTime::parse(value, &Rfc3339)
        .map(|time| (time.unix_timestamp_nanos() / 1_000_000) as i64)
        .unwrap_or_default()
}

fn export_markdown(tree: &Tree, header: &Header) -> String {
    let title = if !tree.name().is_empty() {
        tree.name().to_owned()
    } else {
        tree.path(tree.leaf())
            .iter()
            .find_map(|entry| {
                (entry.kind == sessionlog::TYPE_MESSAGE)
                    .then_some(entry.message.as_ref())
                    .flatten()
                    .and_then(|value| serde_json::from_value::<llm::Message>(value.clone()).ok())
                    .and_then(|message| match message {
                        llm::Message::User(message) => {
                            let text = message_text(&message);
                            (!text.is_empty()).then_some(first_line(&text, 120))
                        }
                        _ => None,
                    })
            })
            .unwrap_or_else(|| header.id.clone())
    };
    let mut markdown = format!(
        "# {title}\n\n- Session: `{}`\n- Workspace: `{}`\n- Started: {}\n\n",
        header.id, header.cwd, header.timestamp
    );
    for entry in tree.context_path(tree.leaf()) {
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
                        markdown.push_str(message_text(&message).trim());
                        markdown.push_str("\n\n");
                    }
                    llm::Message::Assistant(message) => {
                        markdown.push_str("## Assistant\n\n");
                        for block in &message.content {
                            match block {
                                llm::ContentBlock::Text(text) => {
                                    markdown.push_str(text.text.trim());
                                    markdown.push_str("\n\n");
                                }
                                llm::ContentBlock::ToolCall(call) => {
                                    markdown.push_str(&format!("> called `{}`\n\n", call.name));
                                }
                                llm::ContentBlock::Thinking(_) | llm::ContentBlock::Image(_) => {}
                            }
                        }
                    }
                    llm::Message::ToolResult(message) => {
                        let error = if message.is_error { " (error)" } else { "" };
                        markdown
                            .push_str(&format!("> `{}` returned{error}\n\n", message.tool_name));
                    }
                }
            }
            _ => {}
        }
    }
    markdown
}

fn message_text(message: &llm::UserMessage) -> String {
    match &message.content {
        llm::UserContent::Text(text) => text.clone(),
        llm::UserContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(llm::ContentBlock::plain_text)
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn first_line(value: &str, limit: usize) -> String {
    let value = value.split(['\r', '\n']).next().unwrap_or_default().trim();
    let mut characters = value.chars();
    let short = characters.by_ref().take(limit).collect::<String>();
    if characters.next().is_some() {
        format!("{short}…")
    } else {
        short
    }
}

fn is_zero_i64(value: &i64) -> bool {
    *value == 0
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "goshcoder-session-runtime-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn model(provider: &str, id: &str) -> llm::Model {
        llm::Model {
            provider: provider.to_owned(),
            id: id.to_owned(),
            name: id.to_owned(),
            api: "test".to_owned(),
            ..llm::Model::default()
        }
    }

    fn assistant(text: &str, provider: &str, model_id: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            role: "assistant".to_owned(),
            content: vec![llm::ContentBlock::text(text)],
            api: "test".to_owned(),
            provider: provider.to_owned(),
            model: model_id.to_owned(),
            stop_reason: "stop".to_owned(),
            timestamp: 2,
            ..llm::AssistantMessage::default()
        }
    }

    fn responder(text: &'static str) -> agent::AssistantResponder {
        Arc::new(move |model, _, _| Ok(assistant(text, &model.provider, &model.id)))
    }

    fn options(root: &Path, cwd: &Path) -> SessionOptions {
        SessionOptions {
            cwd: cwd.to_path_buf(),
            sessions_dir: Some(root.join("sessions")),
            model: model("fallback", "fallback-model"),
            responder: Some(responder("answer")),
            ..SessionOptions::default()
        }
    }

    fn close(runtime: &mut SessionRuntime) {
        runtime.close().expect("close session");
    }

    #[test]
    fn no_session_never_creates_storage_and_still_runs_the_agent() {
        let root = temp_root("no-session");
        let cwd = root.join("workspace");
        let mut options = options(&root, &cwd);
        options.selection = SessionSelection::NoSession;
        let mut runtime = SessionRuntime::open(options).expect("open no-session runtime");

        runtime.agent().prompt("private question").expect("prompt");
        assert_eq!(runtime.agent().state().messages.len(), 2);
        assert!(!runtime.recording());
        assert!(runtime.handle().is_none());
        assert!(!root.join("sessions").exists());

        close(&mut runtime);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn event_bridge_records_full_turn_model_thinking_and_reset() {
        let root = temp_root("events");
        let cwd = root.join("workspace");
        let stored = model("stored", "saved-model");
        let mut first_options = options(&root, &cwd);
        first_options.available_models = vec![stored.clone()];
        let mut first = SessionRuntime::open(first_options).expect("open first");
        first.agent().set_model(stored.clone());
        first.agent().set_thinking_level(llm::THINKING_HIGH);
        first.agent().prompt("before reset").expect("first prompt");
        first
            .agent()
            .reset_with_reason("/clear")
            .expect("record reset");
        first.agent().prompt("after reset").expect("second prompt");
        let first_id = first.id().expect("id");
        close(&mut first);

        let mut second_options = options(&root, &cwd);
        second_options.selection = SessionSelection::Continue;
        second_options.available_models = vec![stored.clone()];
        let mut second = SessionRuntime::open(second_options).expect("resume");
        assert!(second.resumed());
        assert_eq!(second.id().as_deref(), Some(first_id.as_str()));
        assert_eq!(second.agent().state().model, stored);
        assert_eq!(second.agent().state().thinking_level, llm::THINKING_HIGH);
        let messages = second.agent().state().messages;
        assert_eq!(messages.len(), 2, "reset must cut restored context");
        assert_eq!(messages[0].text_preview(), "after reset");
        assert_eq!(messages[1].text_preview(), "answer");

        let tree = second.tree().expect("tree");
        let kinds = tree
            .path(None)
            .iter()
            .map(|entry| entry.kind.as_str())
            .collect::<Vec<_>>();
        assert!(kinds.contains(&sessionlog::TYPE_MODEL_CHANGE));
        assert!(kinds.contains(&sessionlog::TYPE_THINKING_LEVEL_CHANGE));
        assert!(kinds.contains(&sessionlog::TYPE_TRANSCRIPT_RESET));
        close(&mut second);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn assistant_message_model_is_a_fallback_when_old_logs_lack_model_change() {
        let root = temp_root("assistant-model");
        let cwd = root.join("workspace");
        let store = Store::new(root.join("sessions"));
        let mut writer = store
            .create_with_id(&cwd, None, "assistant-fallback")
            .expect("create fixture");
        writer
            .append(
                Entry::message(&llm::Message::Assistant(Box::new(assistant(
                    "old answer",
                    "legacy-provider",
                    "legacy-model",
                ))))
                .expect("encode assistant"),
            )
            .expect("append assistant");
        writer.close().expect("close fixture");

        let expected = model("legacy-provider", "legacy-model");
        let mut session_options = options(&root, &cwd);
        session_options.selection = SessionSelection::Session("assistant-fallback".to_owned());
        session_options.available_models = vec![expected.clone()];
        let mut runtime = SessionRuntime::open(session_options).expect("open fixture");
        assert_eq!(
            runtime.restored().model_reference(),
            Some(("legacy-provider", "legacy-model"))
        );
        assert_eq!(runtime.agent().state().model, expected);
        close(&mut runtime);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn continued_busy_session_opens_read_only_and_does_not_record() {
        let root = temp_root("read-only");
        let cwd = root.join("workspace");
        let mut first = SessionRuntime::open(options(&root, &cwd)).expect("open first");
        first.agent().prompt("held").expect("prompt");

        let mut continued = options(&root, &cwd);
        continued.selection = SessionSelection::Continue;
        let mut second = SessionRuntime::open(continued).expect("open second");
        assert!(second.read_only());
        assert!(!second.recording());
        assert!(
            second
                .drain_notices()
                .iter()
                .any(|notice| notice.text.contains("another process"))
        );
        let before = second.tree().expect("tree").len();
        second.agent().prompt("unsaved").expect("prompt second");
        assert_eq!(second.tree().expect("tree after").len(), before);

        close(&mut second);
        close(&mut first);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_bare_id_creates_while_missing_paths_fail() {
        let root = temp_root("selection");
        let cwd = root.join("workspace");
        let mut named = options(&root, &cwd);
        named.selection = SessionSelection::Session("chosen-id".to_owned());
        let mut runtime = SessionRuntime::open(named).expect("create named session");
        assert_eq!(runtime.id().as_deref(), Some("chosen-id"));
        runtime.set_name("chosen title").expect("name");
        assert_eq!(runtime.name().as_deref(), Some("chosen title"));
        assert_eq!(runtime.title().as_deref(), Some("chosen title"));
        close(&mut runtime);

        let mut missing_path = options(&root, &cwd);
        missing_path.selection = SessionSelection::Session(
            root.join("does-not-exist.jsonl")
                .to_string_lossy()
                .into_owned(),
        );
        assert!(matches!(
            SessionRuntime::open(missing_path),
            Err(SessionRuntimeError::Session(SessionError::NotFound(_)))
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn initial_model_and_thinking_settings_survive_resume_without_catalog_access() {
        let root = temp_root("initial-settings");
        let cwd = root.join("workspace");
        let selected = model("selected", "initial-model");
        let mut first_options = options(&root, &cwd);
        first_options.model = selected.clone();
        first_options.thinking_level = llm::THINKING_MEDIUM.to_owned();
        let mut first = SessionRuntime::open(first_options).expect("open");
        first.agent().prompt("persist settings").expect("prompt");
        let id = first.id().expect("id");
        close(&mut first);

        let mut resumed_options = options(&root, &cwd);
        resumed_options.selection = SessionSelection::Session(id);
        resumed_options.available_models = vec![selected.clone()];
        let mut resumed = SessionRuntime::open(resumed_options).expect("resume");
        assert_eq!(resumed.agent().state().model, selected);
        assert_eq!(resumed.agent().state().thinking_level, llm::THINKING_MEDIUM);
        close(&mut resumed);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn initial_fork_selection_copies_a_source_branch_into_a_new_session() {
        let root = temp_root("initial-fork");
        let cwd = root.join("workspace");
        let mut source = SessionRuntime::open(options(&root, &cwd)).expect("open source");
        source.agent().prompt("source question").expect("prompt");
        let source_id = source.id().expect("source id");
        close(&mut source);

        let mut fork_options = options(&root, &cwd);
        fork_options.selection = SessionSelection::Fork {
            source: source_id.clone(),
            at: None,
        };
        let mut fork = SessionRuntime::open(fork_options).expect("fork source");
        assert!(fork.resumed());
        assert_ne!(fork.id().as_deref(), Some(source_id.as_str()));
        assert_eq!(
            fork.header()
                .expect("fork header")
                .parent_session
                .as_deref(),
            Some(source_id.as_str())
        );
        assert_eq!(fork.agent().state().messages.len(), 2);
        close(&mut fork);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resuming_a_legacy_file_forks_it_into_a_continuable_v3_session() {
        let root = temp_root("legacy");
        let cwd = root.join("workspace");
        let legacy = root.join("legacy.jsonl");
        fs::create_dir_all(&root).expect("make root");
        fs::write(
            &legacy,
            concat!(
                "{\"type\":\"session\",\"id\":\"legacy-source\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/old\"}\n",
                "{\"type\":\"message\",\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{\"role\":\"user\",\"content\":\"old question\",\"timestamp\":1}}\n",
                "{\"type\":\"message\",\"timestamp\":\"2026-01-01T00:00:02.000Z\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"old answer\"}],\"api\":\"test\",\"provider\":\"legacy\",\"model\":\"legacy-model\",\"usage\":{},\"timestamp\":2}}\n"
            ),
        )
        .expect("write legacy fixture");

        let mut session_options = options(&root, &cwd);
        session_options.selection =
            SessionSelection::Session(legacy.to_string_lossy().into_owned());
        session_options.available_models = vec![model("legacy", "legacy-model")];
        let mut runtime = SessionRuntime::open(session_options).expect("fork legacy");
        assert!(runtime.resumed());
        assert_ne!(runtime.path().as_deref(), Some(legacy.as_path()));
        assert_eq!(
            runtime
                .header()
                .expect("fork header")
                .parent_session
                .as_deref(),
            Some("legacy-source")
        );
        assert_eq!(runtime.agent().state().messages.len(), 2);
        close(&mut runtime);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compaction_round_trips_details_and_projects_a_summary_context_message() {
        let root = temp_root("compaction");
        let cwd = root.join("workspace");
        let mut runtime = SessionRuntime::open(options(&root, &cwd)).expect("open");
        runtime.agent().prompt("old question").expect("prompt");
        let retained = llm::Message::Assistant(Box::new(assistant(
            "kept answer",
            "fallback",
            "fallback-model",
        )));
        runtime
            .record_compaction(
                CompactionSummary {
                    summary: "Earlier work.".to_owned(),
                    tokens_before: 148_002,
                    cost_before: 5.0,
                    retained_messages: 1,
                    timestamp: 1_756_044_151_512,
                },
                std::slice::from_ref(&retained),
            )
            .expect("record compaction");
        let id = runtime.id().expect("id");
        close(&mut runtime);

        let mut resumed = options(&root, &cwd);
        resumed.selection = SessionSelection::Session(id);
        let mut runtime = SessionRuntime::open(resumed).expect("reopen");
        let restored = runtime.restored();
        assert_eq!(restored.compactions.len(), 1);
        assert_eq!(restored.compactions[0].cost_before, 5.0);
        assert_eq!(restored.compactions[0].retained_messages, 1);
        assert_eq!(restored.compactions[0].tokens_before, 148_002);
        assert_eq!(restored.messages.len(), 2);
        assert!(
            restored.messages[0]
                .text_preview()
                .contains("<conversation-summary>")
        );
        assert_eq!(restored.messages[1].text_preview(), "kept answer");
        close(&mut runtime);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn compaction_resume_preserves_cost_and_pi_compatible_details() {
        let root = temp_root("compaction-cost");
        let cwd = root.join("workspace");
        let mut runtime = SessionRuntime::open(options(&root, &cwd)).expect("open");
        runtime.agent().prompt("old question").expect("prompt");
        let mut retained = assistant("kept answer", "fallback", "fallback-model");
        retained.usage.cost.total = 1.25;
        let retained = llm::Message::Assistant(Box::new(retained));
        runtime
            .record_compaction(
                CompactionSummary {
                    summary: "Earlier work.".to_owned(),
                    tokens_before: 148_002,
                    cost_before: 5.0,
                    retained_messages: 1,
                    timestamp: 1_756_044_151_512,
                },
                std::slice::from_ref(&retained),
            )
            .expect("record compaction");
        let id = runtime.id().expect("id");
        let path = runtime.path().expect("path");
        close(&mut runtime);

        let entries = fs::read_to_string(&path)
            .expect("read log")
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("valid JSONL entry"))
            .collect::<Vec<_>>();
        let compaction = entries
            .iter()
            .find(|entry| entry["type"] == sessionlog::TYPE_COMPACTION)
            .expect("compaction entry");
        assert!(
            compaction["firstKeptEntryId"]
                .as_str()
                .is_some_and(|id| !id.is_empty())
        );
        assert_eq!(compaction["tokensBefore"], 148_002);
        assert_eq!(compaction["details"]["goshcoder"]["costBefore"], 5.0);
        assert_eq!(compaction["details"]["goshcoder"]["retainedMessages"], 1);

        let mut resumed = options(&root, &cwd);
        resumed.selection = SessionSelection::Session(id);
        let mut runtime = SessionRuntime::open(resumed).expect("reopen");
        let restored = runtime.restored();
        assert!(
            entries.len() > restored.messages.len(),
            "the compacted prefix must remain on disk"
        );
        let state = runtime.agent().state();
        let mut messages = state.messages;
        let mut fresh = assistant("new answer", "fallback", "fallback-model");
        fresh.usage.cost.total = 0.75;
        messages.push(llm::Message::Assistant(Box::new(fresh)));
        assert_eq!(
            crate::compaction::conversation_cost(&messages, &state.compactions),
            5.75
        );

        close(&mut runtime);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn import_copy_keeps_the_current_session_active() {
        let root = temp_root("import-copy");
        let source_workspace = root.join("source-workspace");
        let target_workspace = root.join("target-workspace");
        let mut source = SessionRuntime::open(options(&root, &source_workspace)).expect("source");
        source
            .agent()
            .prompt("source prompt")
            .expect("source prompt");
        let source_path = source.path().expect("source path");
        close(&mut source);

        let mut target = SessionRuntime::open(options(&root, &target_workspace)).expect("target");
        target
            .agent()
            .prompt("target prompt")
            .expect("target prompt");
        let current = target.handle().expect("current handle");

        let imported = target
            .import_copy(source_path.to_string_lossy().as_ref())
            .expect("import copy");

        assert_ne!(imported.id, current.id);
        assert_eq!(target.handle().expect("still current").id, current.id);
        target
            .agent()
            .prompt("current session continues")
            .expect("continue current");
        assert!(
            String::from_utf8_lossy(&fs::read(&current.path).expect("read current"))
                .contains("current session continues")
        );
        assert!(
            String::from_utf8_lossy(&fs::read(&imported.path).expect("read imported"))
                .contains("source prompt")
        );

        close(&mut target);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn agent_compaction_event_persists_the_same_live_context_cut() {
        let root = temp_root("agent-compaction");
        let cwd = root.join("workspace");
        let mut runtime = SessionRuntime::open(options(&root, &cwd)).expect("open");
        runtime.agent().prompt("old question").expect("old prompt");
        runtime
            .agent()
            .prompt("latest question")
            .expect("latest prompt");
        let retained = runtime.agent().state().messages[2..].to_vec();
        runtime
            .agent()
            .compact(
                llm::Message::User(llm::UserMessage::text(
                    "<conversation-summary>\nOld work\n</conversation-summary>",
                    3,
                )),
                retained.clone(),
                agent::CompactionInfo {
                    summary: "Old work".to_owned(),
                    tokens_before: 42,
                    cost_before: 1.25,
                    retained_messages: retained.len(),
                    timestamp: 3,
                },
            )
            .expect("compact live agent");
        let id = runtime.id().expect("id");
        close(&mut runtime);

        let mut reopened_options = options(&root, &cwd);
        reopened_options.selection = SessionSelection::Session(id);
        let mut reopened = SessionRuntime::open(reopened_options).expect("reopen");
        let restored = reopened.restored();
        assert_eq!(restored.messages.len(), retained.len() + 1);
        assert!(
            restored.messages[0]
                .text_preview()
                .contains("<conversation-summary>")
        );
        assert_eq!(&restored.messages[1..], retained.as_slice());
        assert_eq!(restored.compactions[0].cost_before, 1.25);
        close(&mut reopened);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn switching_attaches_the_target_before_closing_the_current_recorder() {
        let root = temp_root("switch");
        let cwd = root.join("workspace");
        let mut current = SessionRuntime::open(options(&root, &cwd)).expect("open current");
        current.agent().prompt("current question").expect("prompt");
        let original = current.handle().expect("current handle");

        let mut target = SessionRuntime::open(options(&root, &cwd)).expect("open target");
        target.agent().prompt("target question").expect("prompt");
        let target_handle = target.handle().expect("target handle");
        close(&mut target);

        let switched = current
            .switch_to(&target_handle.id)
            .expect("attach target before closing current");
        assert_eq!(switched.id, target_handle.id);
        current
            .agent()
            .prompt("only in target")
            .expect("record after switch");
        assert!(
            String::from_utf8_lossy(&fs::read(&target_handle.path).expect("target log"))
                .contains("only in target")
        );
        assert!(
            !String::from_utf8_lossy(&fs::read(&original.path).expect("original log"))
                .contains("only in target")
        );
        close(&mut current);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn branches_labels_clone_exports_and_imports_delegate_to_the_session_store() {
        let root = temp_root("tree");
        let cwd = root.join("workspace");
        let mut runtime = SessionRuntime::open(options(&root, &cwd)).expect("open");
        runtime.agent().prompt("first question").expect("first");
        runtime.agent().prompt("second question").expect("second");
        runtime.agent().prompt("third question").expect("third");
        runtime.label(1, "before experiment").expect("label");
        assert_eq!(
            runtime.branch_points()[0].label.as_deref(),
            Some("before experiment")
        );

        let original = runtime.handle().expect("original");
        runtime.fork_to(2).expect("rewind");
        assert_eq!(runtime.agent().state().messages.len(), 3);
        assert_eq!(
            runtime.agent().state().messages[2].text_preview(),
            "second question"
        );
        runtime
            .agent()
            .prompt("different direction")
            .expect("branch prompt");
        let original_bytes = fs::read(&original.path).expect("read original");

        let clone = runtime.clone_session().expect("clone");
        assert_ne!(clone.id, original.id);
        assert_eq!(
            fs::read(&original.path).expect("re-read original"),
            original_bytes
        );
        runtime
            .agent()
            .prompt("only in clone")
            .expect("clone prompt");
        assert!(
            !String::from_utf8_lossy(&fs::read(&original.path).expect("original"))
                .contains("only in clone")
        );

        let markdown = String::from_utf8(runtime.export(ExportFormat::Markdown).expect("markdown"))
            .expect("utf8");
        assert!(markdown.contains("different direction"));
        assert!(markdown.contains("only in clone"));
        let raw = runtime.export(ExportFormat::Jsonl).expect("jsonl");
        assert!(String::from_utf8_lossy(&raw).contains("\"type\":\"session\""));

        let import_root = root.join("imported-workspace");
        let source = runtime.path().expect("clone path");
        let mut imported_options = options(&root, &import_root);
        imported_options.selection = SessionSelection::New;
        let mut imported = SessionRuntime::open(imported_options).expect("import target");
        let imported_handle = imported
            .import_session(source.to_string_lossy().as_ref())
            .expect("import");
        assert_ne!(imported_handle.id, clone.id);
        assert_eq!(
            imported
                .header()
                .expect("imported header")
                .parent_session
                .as_deref(),
            Some(clone.id.as_str())
        );

        close(&mut imported);
        close(&mut runtime);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn callbacks_and_custom_entries_are_delivered_without_a_catalog() {
        let root = temp_root("callbacks");
        let cwd = root.join("workspace");
        let notices = Arc::new(AtomicUsize::new(0));
        let notice_count = notices.clone();
        let model_requests = Arc::new(AtomicUsize::new(0));
        let request_count = model_requests.clone();

        let mut first_options = options(&root, &cwd);
        first_options.selection = SessionSelection::Continue;
        first_options.on_notice = Some(Arc::new(move |_| {
            notice_count.fetch_add(1, Ordering::Relaxed);
        }));
        let mut first = SessionRuntime::open(first_options).expect("open");
        first
            .record_custom("planner", serde_json::json!({"phase": "review"}))
            .expect("record custom");
        first.agent().set_model(model("dynamic", "resolved"));
        first.agent().prompt("save state").expect("prompt");
        let id = first.id().expect("id");
        close(&mut first);

        let mut second = options(&root, &cwd);
        second.selection = SessionSelection::Session(id);
        second.model_resolver =
            Some(Arc::new(move |provider, model_id| {
                request_count.fetch_add(1, Ordering::Relaxed);
                Ok((provider == "dynamic" && model_id == "resolved")
                    .then(|| model(provider, model_id)))
            }));
        let mut second = SessionRuntime::open(second).expect("resume with resolver");
        assert_eq!(model_requests.load(Ordering::Relaxed), 1);
        assert_eq!(second.agent().state().model.provider, "dynamic");
        assert_eq!(
            second.restored().custom.get("planner"),
            Some(&serde_json::json!({"phase": "review"}))
        );
        assert_eq!(notices.load(Ordering::Relaxed), 1);
        close(&mut second);
        let _ = fs::remove_dir_all(root);
    }
}
