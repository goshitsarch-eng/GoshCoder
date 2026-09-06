//! Session and agent integration for the native planner.
//!
//! The planner core deliberately has no terminal or session dependency. This
//! adapter owns the live joins: restoring durable state, recording state
//! changes, rebuilding the model-visible tool list, enforcing the planning
//! write gate, and forwarding browser-review notices to the active frontend.
//!
//! Planner state is scoped to a workspace. The authoritative copy lives in a
//! per-workspace file under the agent directory ([`WorkspaceStateStore`]),
//! which every window open on the same root re-checks before a model turn or
//! toggle and rewrites on every change, so `-no-session` runs keep their plan
//! mode and two windows share one. Each change is also recorded as a
//! pi-compatible session custom entry, which restores the state when no
//! workspace file exists.

use std::{
    collections::BTreeSet,
    fmt,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant, SystemTime},
};

use serde::{Deserialize, Serialize};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

use crate::{
    agent, config, llm, plannotator,
    session::{SessionCustomRecorder, SessionNoticeSender, SessionRuntime},
    tools::Workspace,
};

/// pi-compatible custom entry type used for one session's planner state.
pub const CUSTOM_TYPE: &str = "goshcoder.planner";
/// Largest workspace state file that is read.
pub const MAX_WORKSPACE_STATE_BYTES: u64 = 1024 * 1024;
/// Layout of the workspace state file written by this build.
const WORKSPACE_STATE_VERSION: u32 = 1;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Errors that prevent a planner from being attached to a prepared session.
#[derive(Debug)]
pub enum PlannerRuntimeError {
    Planner(plannotator::PlannerError),
    /// The workspace state file could not be read, trusted, or written.
    WorkspaceState {
        path: PathBuf,
        reason: String,
    },
}

impl fmt::Display for PlannerRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Planner(error) => error.fmt(formatter),
            Self::WorkspaceState { path, reason } => {
                write!(
                    formatter,
                    "workspace planner state {}: {reason}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for PlannerRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Planner(error) => Some(error),
            Self::WorkspaceState { .. } => None,
        }
    }
}

impl From<plannotator::PlannerError> for PlannerRuntimeError {
    fn from(error: plannotator::PlannerError) -> Self {
        Self::Planner(error)
    }
}

pub type Result<T> = std::result::Result<T, PlannerRuntimeError>;

/// A session-extension callback that applies Planner state to a raw system
/// prompt while rebuilding the corresponding model-visible tool set.
pub type SystemPromptSync = Arc<dyn Fn(String) + Send + Sync + 'static>;

/// Identity of one on-disk version of the workspace state file: its
/// modification time and length, the same signal other GoshCoder caches use
/// to notice an edit made by another process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fingerprint {
    modified: Option<SystemTime>,
    len: u64,
}

impl Fingerprint {
    fn of(metadata: &fs::Metadata) -> Self {
        Self {
            modified: metadata.modified().ok(),
            len: metadata.len(),
        }
    }
}

/// The planner state file shared by every window open on one workspace.
///
/// The file is `{"version": 1, "workspace": "<canonical root>", "state":
/// {...}}`. It is replaced atomically through a same-directory temporary
/// file and is readable by the user only. A file for a different workspace,
/// or one this build cannot parse, is an error the caller reports and then
/// ignores; it never prevents a session from opening.
#[derive(Clone, Debug)]
pub struct WorkspaceStateStore {
    path: PathBuf,
    workspace: String,
}

#[derive(Deserialize, Serialize)]
struct WorkspaceStateFile {
    version: u32,
    workspace: String,
    state: plannotator::State,
}

impl WorkspaceStateStore {
    /// Uses the per-user location from [`config::planner_state_path`].
    /// `workspace_root` must be canonical.
    #[must_use]
    pub fn for_workspace(workspace_root: &Path) -> Self {
        Self::at(config::planner_state_path(workspace_root), workspace_root)
    }

    /// Uses an explicit file, for embedding and tests. The file still names
    /// `workspace_root` so a copied or misplaced file is not trusted.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>, workspace_root: &Path) -> Self {
        Self {
            path: path.into(),
            workspace: workspace_root.to_string_lossy().into_owned(),
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the identity of the file currently on disk, if any.
    #[must_use]
    pub fn fingerprint(&self) -> Option<Fingerprint> {
        fs::metadata(&self.path)
            .ok()
            .map(|metadata| Fingerprint::of(&metadata))
    }

    /// Reads the file. A missing file is `Ok(None)`; an unreadable,
    /// oversized, malformed, or foreign-workspace file is an error.
    pub fn load(&self) -> Result<Option<(plannotator::State, Fingerprint)>> {
        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(self.error(error)),
        };
        let metadata = file.metadata().map_err(|error| self.error(error))?;
        if metadata.len() > MAX_WORKSPACE_STATE_BYTES {
            return Err(self.error(format!(
                "exceeds the {MAX_WORKSPACE_STATE_BYTES}-byte limit"
            )));
        }
        let mut contents = Vec::new();
        (&mut file)
            .take(MAX_WORKSPACE_STATE_BYTES)
            .read_to_end(&mut contents)
            .map_err(|error| self.error(error))?;
        let parsed: WorkspaceStateFile = serde_json::from_slice(&contents)
            .map_err(|error| self.error(format!("malformed: {error}")))?;
        if parsed.version != WORKSPACE_STATE_VERSION {
            return Err(self.error(format!("unsupported version {}", parsed.version)));
        }
        if parsed.workspace != self.workspace {
            return Err(self.error(format!(
                "belongs to workspace {}, not {}",
                parsed.workspace, self.workspace
            )));
        }
        Ok(Some((parsed.state, Fingerprint::of(&metadata))))
    }

    /// Replaces the file atomically and returns the identity of the version
    /// written, so the writer can tell its own file from a later one.
    pub fn save(&self, state: &plannotator::State) -> Result<Fingerprint> {
        let directory = self.path.parent().unwrap_or_else(|| Path::new("."));
        let mut contents = serde_json::to_vec_pretty(&WorkspaceStateFile {
            version: WORKSPACE_STATE_VERSION,
            workspace: self.workspace.clone(),
            state: state.clone(),
        })
        .map_err(|error| self.error(format!("encode: {error}")))?;
        contents.push(b'\n');

        create_private_directory(directory).map_err(|error| self.error(error))?;
        let (temporary_path, mut temporary) =
            create_temporary_file(directory).map_err(|error| self.error(error))?;
        let written = write_temporary(&mut temporary, &contents);
        drop(temporary);
        let metadata = match written {
            Ok(metadata) => metadata,
            Err(error) => {
                let _ = fs::remove_file(&temporary_path);
                return Err(self.error(error));
            }
        };
        if let Err(error) = fs::rename(&temporary_path, &self.path) {
            let _ = fs::remove_file(&temporary_path);
            return Err(self.error(error));
        }
        #[cfg(unix)]
        File::open(directory)
            .and_then(|handle| handle.sync_all())
            .map_err(|error| self.error(error))?;
        Ok(Fingerprint::of(&metadata))
    }

    fn error(&self, reason: impl ToString) -> PlannerRuntimeError {
        PlannerRuntimeError::WorkspaceState {
            path: self.path.clone(),
            reason: reason.to_string(),
        }
    }
}

/// Creates the state directory, and any missing parents, readable by the
/// user only. An existing directory keeps its permissions.
fn create_private_directory(directory: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder.create(directory)
}

fn create_temporary_file(directory: &Path) -> io::Result<(PathBuf, File)> {
    for _ in 0..256 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".planner-{}-{sequence}.tmp", process::id()));
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique planner state temporary file",
    ))
}

/// Writes and flushes the complete file, returning the metadata of the inode
/// that the rename will publish. It is taken before the rename so a file
/// another window replaces immediately afterwards is still noticed.
fn write_temporary(temporary: &mut File, contents: &[u8]) -> io::Result<fs::Metadata> {
    temporary.write_all(contents)?;
    temporary.sync_all()?;
    temporary.metadata()
}

/// Where a state change goes: the workspace file other windows watch, and
/// the session log when one is being recorded.
///
/// `seen` is the fingerprint of the file version this process last wrote or
/// read. A file with a different fingerprint was written by another window
/// and is adopted before this window's next model turn or toggle.
struct Persistence {
    store: WorkspaceStateStore,
    recorder: SessionCustomRecorder,
    notices: SessionNoticeSender,
    seen: Mutex<Option<Fingerprint>>,
}

impl Persistence {
    /// Reads the workspace file at attach time. A problem is reported once,
    /// and the file is then treated as absent.
    fn restore(&self) -> Option<plannotator::State> {
        let mut seen = lock(&self.seen);
        match self.store.load() {
            Ok(Some((state, fingerprint))) => {
                *seen = Some(fingerprint);
                Some(state)
            }
            Ok(None) => {
                *seen = None;
                None
            }
            Err(error) => {
                *seen = self.store.fingerprint();
                self.notices.push("Planner", format!("ignoring {error}"));
                None
            }
        }
    }

    /// Persists a change made in this process.
    fn save(&self, state: &plannotator::State) {
        {
            let mut seen = lock(&self.seen);
            match self.store.save(state) {
                Ok(fingerprint) => *seen = Some(fingerprint),
                Err(error) => self
                    .notices
                    .push("Planner", format!("could not save {error}")),
            }
        }
        self.record_session(state);
    }

    /// Adopts a file version written by another window since this process
    /// last read or wrote it. Returns whether the manager's state changed,
    /// in which case the caller re-syncs the agent.
    ///
    /// The fingerprint lock is held throughout so a toggle and a turn-end
    /// subscription racing on one file adopt it once. Nothing called here
    /// publishes a change, so the lock is never re-entered.
    fn adopt_external_change(&self, manager: &plannotator::Manager) -> bool {
        let mut seen = lock(&self.seen);
        let current = self.store.fingerprint();
        if current == *seen {
            return false;
        }
        match self.store.load() {
            Ok(Some((state, fingerprint))) => {
                *seen = Some(fingerprint);
                let previous = manager.state();
                if state == previous {
                    return false;
                }
                manager.adopt_state(state);
                let adopted = manager.state();
                if adopted == previous {
                    return false;
                }
                self.record_session(&adopted);
                self.notices.push(
                    "Planner",
                    format!(
                        "state updated by another window: now {}",
                        phase_summary(&adopted)
                    ),
                );
                true
            }
            Ok(None) => {
                // The file was removed. The in-memory state stays, and the
                // next change here recreates the file.
                *seen = None;
                false
            }
            Err(error) => {
                *seen = current;
                self.notices.push("Planner", format!("ignoring {error}"));
                false
            }
        }
    }

    /// Appends the pi-compatible custom entry when a session is recording.
    ///
    /// A no-session or read-only run keeps its state in the workspace file
    /// alone; that must not turn a harmless toggle into a persistence error.
    fn record_session(&self, state: &plannotator::State) {
        if !self.recorder.recording() {
            return;
        }
        let payload = match serde_json::to_value(state) {
            Ok(payload) => payload,
            Err(error) => {
                self.notices.push(
                    "Planner",
                    format!("could not encode planner state for the session: {error}"),
                );
                return;
            }
        };
        if let Err(error) = self.recorder.record(CUSTOM_TYPE, payload) {
            self.notices.push(
                "Planner",
                format!("could not save planner state to the session: {error}"),
            );
        }
    }
}

fn phase_summary(state: &plannotator::State) -> String {
    match state.phase {
        plannotator::Phase::Planning => "planning".to_owned(),
        plannotator::Phase::Executing => {
            let completed = state.items.iter().filter(|item| item.completed).count();
            format!("executing {completed}/{}", state.items.len())
        }
        plannotator::Phase::Idle | plannotator::Phase::Unknown(_) => "idle".to_owned(),
    }
}

/// Keeps a planner attached to one session and its live agent.
///
/// Its subscription must remain alive for the complete session: it applies
/// `[DONE:n]` markers and changes from `planner_submit_plan` before the agent
/// requests its next model turn.
pub struct PlannerRuntime {
    manager: plannotator::Manager,
    agent: agent::Agent,
    workspace: Workspace,
    /// The session's ordinary tools. Shared with the turn-end subscription so
    /// tools registered after startup (gateway connectors) survive every
    /// phase-specific rebuild instead of being unregistered by the next one.
    normal_tools: Arc<Mutex<Vec<agent::Tool>>>,
    base_system_prompt: Arc<Mutex<String>>,
    reviewer: Arc<dyn plannotator::Reviewer>,
    notices: SessionNoticeSender,
    persistence: Arc<Persistence>,
    review_cancellation: Arc<Mutex<Option<agent::CancellationToken>>>,
    _subscription: agent::Subscription,
}

/// Cloneable handle for a human planner review that runs off the terminal UI
/// thread. Only one review is expected per session at a time.
#[derive(Clone)]
pub struct PlannerReviewHandle {
    reviewer: Arc<dyn plannotator::Reviewer>,
    notices: SessionNoticeSender,
    cancellation: Arc<Mutex<Option<agent::CancellationToken>>>,
}

/// Upper bound on a slash-command review. A tool-driven review ends with the
/// agent turn that owns it; `/planner-review` and `/planner-annotate` have no
/// such owner, so an abandoned browser tab would otherwise hold the session
/// open indefinitely.
const SLASH_REVIEW_TIMEOUT: Duration = Duration::from_secs(30 * 60);

impl PlannerReviewHandle {
    /// Opens the configured review surface and waits for a decision.
    pub fn review(
        &self,
        request: &plannotator::ReviewRequest,
    ) -> std::result::Result<plannotator::Decision, plannotator::ReviewError> {
        self.review_within(request, SLASH_REVIEW_TIMEOUT)
    }

    fn review_within(
        &self,
        request: &plannotator::ReviewRequest,
        timeout: Duration,
    ) -> std::result::Result<plannotator::Decision, plannotator::ReviewError> {
        let cancellation = agent::CancellationToken::default();
        *lock(&self.cancellation) = Some(cancellation.clone());
        let (finished, watchdog) = mpsc::channel::<()>();
        let expiring = cancellation.clone();
        thread::spawn(move || {
            // Dropping `finished` wakes the watchdog early; only a real
            // timeout cancels the review.
            if watchdog.recv_timeout(timeout) == Err(mpsc::RecvTimeoutError::Timeout) {
                expiring.cancel();
            }
        });
        let result = self.reviewer.review(&cancellation, request);
        drop(finished);
        *lock(&self.cancellation) = None;
        result
    }

    /// Cancels an open browser review, if any.
    pub fn cancel(&self) -> bool {
        let cancellation = lock(&self.cancellation).clone();
        if let Some(cancellation) = cancellation {
            cancellation.cancel();
            true
        } else {
            false
        }
    }

    /// Sends an outcome to the current UI or line-mode renderer.
    pub fn notify(&self, message: impl Into<String>) {
        self.notices.push("Planner", message);
    }
}

impl PlannerRuntime {
    /// Attaches a planner to an already-opened session.
    ///
    /// `normal_tools` must be the session's ordinary unwrapped tool list. This
    /// adapter derives all phase-specific lists from it, so tools installed by
    /// other integrations cannot disappear when the planner switches phase.
    pub fn attach(
        runtime: &SessionRuntime,
        workspace: Workspace,
        normal_tools: Vec<agent::Tool>,
        base_system_prompt: String,
        start_in_planning: bool,
    ) -> Result<Self> {
        let store = WorkspaceStateStore::for_workspace(workspace.root());
        Self::attach_with_store(
            runtime,
            workspace,
            normal_tools,
            base_system_prompt,
            start_in_planning,
            store,
        )
    }

    /// [`Self::attach`] with an explicit workspace state file, for embedding
    /// and for tests that must not touch the user's agent directory.
    pub fn attach_with_store(
        runtime: &SessionRuntime,
        workspace: Workspace,
        normal_tools: Vec<agent::Tool>,
        base_system_prompt: String,
        start_in_planning: bool,
        store: WorkspaceStateStore,
    ) -> Result<Self> {
        let notices = runtime.notice_sender();
        let persistence = Arc::new(Persistence {
            store,
            recorder: runtime.custom_recorder(),
            notices: notices.clone(),
            seen: Mutex::new(None),
        });
        // The workspace file is authoritative whenever it exists. The
        // session's own entry only fills in for a workspace whose file was
        // never written or has since been removed.
        let initial = persistence
            .restore()
            .or_else(|| restored_state(runtime, &notices));
        let reviewer: Arc<dyn plannotator::Reviewer> = Arc::new(plannotator::BrowserReviewer {
            notify: Some(review_notice_callback(notices.clone())),
            ..plannotator::BrowserReviewer::default()
        });
        let manager = plannotator::Manager::new(
            workspace.root(),
            Some(Arc::clone(&reviewer)),
            plannotator::Options {
                initial,
                on_change: Some(persistence_callback(Arc::clone(&persistence))),
                warn: Some(warning_callback(notices.clone())),
            },
        )?;

        if start_in_planning && manager.state().phase == plannotator::Phase::Idle {
            manager.enter();
        }

        let agent = runtime.agent().clone();
        let base_system_prompt = Arc::new(Mutex::new(base_system_prompt));
        let normal_tools = Arc::new(Mutex::new(normal_tools));
        let subscription = planner_subscription(
            &agent,
            manager.clone(),
            workspace.clone(),
            Arc::clone(&normal_tools),
            Arc::clone(&base_system_prompt),
            Arc::clone(&persistence),
        );
        let integration = Self {
            manager,
            agent,
            workspace,
            normal_tools,
            base_system_prompt,
            reviewer,
            notices,
            persistence,
            review_cancellation: Arc::new(Mutex::new(None)),
            _subscription: subscription,
        };
        integration.sync_agent();
        Ok(integration)
    }

    /// Returns the session-owned planning state machine.
    pub fn manager(&self) -> &plannotator::Manager {
        &self.manager
    }

    /// Returns the current succinct state for a status bar.
    pub fn status_line(&self) -> String {
        self.manager.status_line()
    }

    /// Returns the workspace root used for planner files, diffs, and reviews.
    pub fn workspace_root(&self) -> &Path {
        self.workspace.root()
    }

    /// Returns a cloneable UI-thread-safe browser review handle.
    pub fn review_handle(&self) -> PlannerReviewHandle {
        PlannerReviewHandle {
            reviewer: Arc::clone(&self.reviewer),
            notices: self.notices.clone(),
            cancellation: Arc::clone(&self.review_cancellation),
        }
    }

    /// Cancels an outstanding browser review, if one is open.
    pub fn abort_review(&self) -> bool {
        self.review_handle().cancel()
    }

    /// Toggles idle/planning and immediately applies its prompt and tool set.
    ///
    /// A change another window saved since this one last looked is adopted
    /// first, so the toggle flips the phase the user can see in that window
    /// rather than a stale one.
    pub fn toggle(&self) -> plannotator::Phase {
        self.adopt_external_change();
        let phase = self.manager.toggle();
        self.sync_agent();
        phase
    }

    /// Picks up planner state saved by another window on this workspace
    /// since this process last read or wrote the shared file. Returns
    /// whether the state changed; the caller then re-syncs the agent.
    pub fn adopt_external_change(&self) -> bool {
        self.persistence.adopt_external_change(&self.manager)
    }

    /// Returns the workspace state file shared with other windows.
    pub fn workspace_state_path(&self) -> &Path {
        self.persistence.store.path()
    }

    /// Replaces the base prompt while retaining the current planner suffix.
    pub fn set_base_system_prompt(&self, prompt: impl Into<String>) {
        *lock(&self.base_system_prompt) = prompt.into();
        self.sync_agent();
    }

    /// Rebuilds prompt and tools after an external integration changes base
    /// state, adopting another window's planner change first. The method is
    /// idempotent and is safe to call from a UI loop.
    pub fn sync_agent(&self) {
        self.adopt_external_change();
        let normal_tools = lock(&self.normal_tools).clone();
        sync_agent(
            &self.agent,
            &self.manager,
            &self.workspace,
            &normal_tools,
            &self.base_system_prompt,
        );
    }

    /// Adds tools that arrived after startup to the ordinary tool set and
    /// re-applies the current phase. A tool whose name is already registered
    /// is ignored so a repeated registration cannot duplicate it.
    pub fn extend_normal_tools(&self, tools: Vec<agent::Tool>) {
        extend_tools(&self.normal_tools, tools);
        self.sync_agent();
    }

    /// A callback that performs [`PlannerRuntime::extend_normal_tools`] from
    /// a background thread that must not hold the session.
    #[must_use]
    pub fn tool_extender(&self) -> Arc<dyn Fn(Vec<agent::Tool>) + Send + Sync> {
        let agent = self.agent.clone();
        let manager = self.manager.clone();
        let workspace = self.workspace.clone();
        let normal_tools = Arc::clone(&self.normal_tools);
        let base_system_prompt = Arc::clone(&self.base_system_prompt);
        Arc::new(move |tools| {
            extend_tools(&normal_tools, tools);
            let snapshot = lock(&normal_tools).clone();
            sync_agent(&agent, &manager, &workspace, &snapshot, &base_system_prompt);
        })
    }

    /// Applies the current planner state to an extension-composed base prompt.
    ///
    /// Ralph invokes this after updating its own active-loop suffix so Planner
    /// remains the outermost prompt layer and phase-specific tool rebuilding
    /// cannot unregister Ralph's tools.
    pub fn sync_with_base(&self, base_system_prompt: impl AsRef<str>) {
        self.adopt_external_change();
        let normal_tools = lock(&self.normal_tools).clone();
        sync_agent_with_base(
            &self.agent,
            &self.manager,
            &self.workspace,
            &normal_tools,
            base_system_prompt.as_ref(),
        );
    }

    /// Returns a callback-safe prompt synchronizer for another session
    /// extension. It owns the same planner state and normal tool snapshot as
    /// this runtime, but does not alter the user's raw base prompt.
    #[must_use]
    pub fn system_prompt_sync(&self) -> SystemPromptSync {
        let agent = self.agent.clone();
        let manager = self.manager.clone();
        let workspace = self.workspace.clone();
        let normal_tools = Arc::clone(&self.normal_tools);
        let persistence = Arc::clone(&self.persistence);
        Arc::new(move |base_system_prompt| {
            persistence.adopt_external_change(&manager);
            let snapshot = lock(&normal_tools).clone();
            sync_agent_with_base(&agent, &manager, &workspace, &snapshot, &base_system_prompt);
        })
    }
}

fn restored_state(
    runtime: &SessionRuntime,
    notices: &SessionNoticeSender,
) -> Option<plannotator::State> {
    let restored = runtime.restored();
    let raw = restored.custom.get(CUSTOM_TYPE)?;
    match serde_json::from_value::<plannotator::State>(raw.clone()) {
        Ok(state) => Some(state),
        Err(error) => {
            notices.push(
                "Planner",
                format!("ignoring unreadable saved planner state: {error}"),
            );
            None
        }
    }
}

fn persistence_callback(persistence: Arc<Persistence>) -> plannotator::StateCallback {
    Arc::new(move |state| persistence.save(&state))
}

fn warning_callback(notices: SessionNoticeSender) -> plannotator::WarningCallback {
    Arc::new(move |message| notices.push("Planner", message))
}

fn review_notice_callback(notices: SessionNoticeSender) -> plannotator::ReviewNoticeCallback {
    Arc::new(move |message| notices.push("Planner", message))
}

fn planner_subscription(
    agent: &agent::Agent,
    manager: plannotator::Manager,
    workspace: Workspace,
    normal_tools: Arc<Mutex<Vec<agent::Tool>>>,
    base_system_prompt: Arc<Mutex<String>>,
    persistence: Arc<Persistence>,
) -> agent::Subscription {
    let agent = agent.clone();
    agent.clone().subscribe(move |event| {
        if event.kind != agent::EventKind::TurnEnd {
            return;
        }
        // Another window's change is adopted first so completion markers
        // from this turn land on the shared state instead of replacing it.
        persistence.adopt_external_change(&manager);
        if let Some(llm::Message::Assistant(message)) = event.message.as_ref() {
            manager.track_assistant(message);
        }
        let normal_tools = lock(&normal_tools).clone();
        sync_agent(
            &agent,
            &manager,
            &workspace,
            &normal_tools,
            &base_system_prompt,
        );
    })
}

fn sync_agent(
    agent: &agent::Agent,
    manager: &plannotator::Manager,
    workspace: &Workspace,
    normal_tools: &[agent::Tool],
    base_system_prompt: &Arc<Mutex<String>>,
) {
    let base = lock(base_system_prompt).clone();
    sync_agent_with_base(agent, manager, workspace, normal_tools, &base);
}

fn sync_agent_with_base(
    agent: &agent::Agent,
    manager: &plannotator::Manager,
    workspace: &Workspace,
    normal_tools: &[agent::Tool],
    base_system_prompt: &str,
) {
    agent.set_system_prompt(manager.prompt(base_system_prompt));
    agent.set_tools(planner_tools(manager, workspace, normal_tools));
}

fn planner_tools(
    manager: &plannotator::Manager,
    workspace: &Workspace,
    normal_tools: &[agent::Tool],
) -> Vec<agent::Tool> {
    let manager_tool = manager.tool();
    let tools = match manager.tool_access() {
        plannotator::ToolAccess::Idle => normal_tools.to_vec(),
        plannotator::ToolAccess::Planning => merge_tools([
            without_tool(normal_tools, "bash"),
            workspace.planning(),
            vec![manager_tool],
        ]),
        plannotator::ToolAccess::Executing => {
            merge_tools([normal_tools.to_vec(), workspace.all(), vec![manager_tool]])
        }
    };
    tools
        .into_iter()
        .map(|tool| guard_tool(tool, manager.clone()))
        .collect()
}

/// Appends tools whose names are not registered yet.
fn extend_tools(tools: &Mutex<Vec<agent::Tool>>, additions: Vec<agent::Tool>) {
    let mut tools = lock(tools);
    let mut names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<BTreeSet<_>>();
    tools.extend(
        additions
            .into_iter()
            .filter(|tool| names.insert(tool.name.clone())),
    );
}

fn without_tool(tools: &[agent::Tool], name: &str) -> Vec<agent::Tool> {
    tools
        .iter()
        .filter(|tool| tool.name != name)
        .cloned()
        .collect()
}

fn merge_tools<const N: usize>(groups: [Vec<agent::Tool>; N]) -> Vec<agent::Tool> {
    let mut names = BTreeSet::new();
    groups
        .into_iter()
        .flatten()
        .filter(|tool| names.insert(tool.name.clone()))
        .collect()
}

fn guard_tool(mut tool: agent::Tool, manager: plannotator::Manager) -> agent::Tool {
    let name = tool.name.clone();
    let execute = Arc::clone(&tool.execute);
    tool.execute = Arc::new(move |cancellation, call_id, arguments, update| {
        let call = llm::ToolCall {
            id: call_id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
            thought_signature: String::new(),
            namespace: String::new(),
        };
        if let Some(gate) = manager.before_tool_call(&call)
            && gate.block
        {
            return Err(gate.reason);
        }
        execute(cancellation, call_id, arguments, update)
    });
    tool
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

/// Loads the working-tree or a GitHub pull-request diff for `/planner-review`.
///
/// Command output is consumed concurrently and bounded while it is read, so a
/// malformed repository or remote PR cannot make the terminal process retain
/// an unbounded diff in memory.
pub fn load_diff_review(
    workspace: impl AsRef<Path>,
    pull_request_url: &str,
) -> std::result::Result<plannotator::ReviewRequest, String> {
    let workspace = workspace.as_ref();
    let target = pull_request_url.trim();
    let diff = if target.is_empty() {
        let unstaged = run_bounded_command(
            Command::new("git")
                .arg("diff")
                .arg("--no-ext-diff")
                .arg("--")
                .current_dir(workspace),
        )?;
        if unstaged.trim().is_empty() {
            run_bounded_command(
                Command::new("git")
                    .arg("diff")
                    .arg("--cached")
                    .arg("--no-ext-diff")
                    .arg("--")
                    .current_dir(workspace),
            )?
        } else {
            unstaged
        }
    } else if target.starts_with("https://") || target.starts_with("http://") {
        run_bounded_command(
            Command::new("gh")
                .arg("pr")
                .arg("diff")
                .arg(target)
                .current_dir(workspace),
        )?
    } else {
        return Err("usage: /planner-review [GitHub PR URL]".to_owned());
    };
    plannotator::diff_review_request(&diff).map_err(|error| error.to_string())
}

const REVIEW_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

fn run_bounded_command(command: &mut Command) -> std::result::Result<String, String> {
    // `git` and `gh` prompt on an inherited terminal (credentials, a pager)
    // and would then wait on the user behind the UI's back.
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("start review diff command: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "capture review diff stdout".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "capture review diff stderr".to_owned())?;
    let stdout = thread::spawn(move || read_bounded(stdout, plannotator::MAX_REVIEW_DIFF_BYTES));
    let stderr = thread::spawn(move || read_bounded(stderr, plannotator::MAX_REVIEW_DIFF_BYTES));

    let deadline = Instant::now() + REVIEW_COMMAND_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err("review diff command timed out".to_owned());
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(format!("wait for review diff command: {error}"));
            }
        }
    };
    let stdout = stdout
        .join()
        .map_err(|_| "read review diff stdout worker panicked".to_owned())?
        .map_err(|error| format!("read review diff stdout: {error}"))?;
    let stderr = stderr
        .join()
        .map_err(|_| "read review diff stderr worker panicked".to_owned())?
        .map_err(|error| format!("read review diff stderr: {error}"))?;
    let status = status?;
    if stdout.truncated || stderr.truncated {
        return Err(format!(
            "review diff exceeds the {}-byte limit",
            plannotator::MAX_REVIEW_DIFF_BYTES
        ));
    }
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr.bytes).trim().to_owned();
        let suffix = (!stderr.is_empty()).then(|| format!(": {stderr}"));
        return Err(format!(
            "load review diff failed with status {}{}",
            status
                .code()
                .map_or_else(|| "signal".to_owned(), |code| code.to_string()),
            suffix.unwrap_or_default()
        ));
    }
    String::from_utf8(stdout.bytes).map_err(|error| format!("review diff is not UTF-8: {error}"))
}

struct BoundedOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

fn read_bounded(mut reader: impl Read, limit: usize) -> std::io::Result<BoundedOutput> {
    let mut output = BoundedOutput {
        bytes: Vec::with_capacity(limit.min(64 * 1024)),
        truncated: false,
    };
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            return Ok(output);
        }
        let remaining = limit.saturating_sub(output.bytes.len());
        if remaining < count {
            output.bytes.extend_from_slice(&buffer[..remaining]);
            output.truncated = true;
        } else {
            output.bytes.extend_from_slice(&buffer[..count]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        fs,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use super::*;
    use crate::{
        llm,
        plannotator::{Phase, State},
        session::{SessionOptions, SessionSelection},
    };
    use serde_json::json;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_path(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "goshcoder-planner-runtime-{label}-{}-{sequence}",
            std::process::id()
        ))
    }

    fn responder(text: &'static str) -> agent::AssistantResponder {
        Arc::new(move |_, _, _| {
            Ok(llm::AssistantMessage {
                role: "assistant".to_owned(),
                content: vec![llm::ContentBlock::text(text)],
                api: "test".to_owned(),
                provider: "test".to_owned(),
                model: "test".to_owned(),
                stop_reason: "stop".to_owned(),
                timestamp: 1,
                ..llm::AssistantMessage::default()
            })
        })
    }

    fn options(root: &Path) -> SessionOptions {
        SessionOptions {
            cwd: root.to_path_buf(),
            sessions_dir: Some(root.join("sessions")),
            model: llm::Model {
                provider: "test".to_owned(),
                id: "test".to_owned(),
                api: "test".to_owned(),
                ..llm::Model::default()
            },
            responder: Some(responder("done")),
            ..SessionOptions::default()
        }
    }

    fn runtime(root: &Path) -> SessionRuntime {
        SessionRuntime::open(options(root)).expect("open session")
    }

    fn no_session_runtime(root: &Path) -> SessionRuntime {
        SessionRuntime::open(SessionOptions {
            selection: SessionSelection::NoSession,
            ..options(root)
        })
        .expect("open no-session runtime")
    }

    fn store(root: &Path, workspace: &Workspace) -> WorkspaceStateStore {
        WorkspaceStateStore::at(
            root.join("agent").join("planner").join("state.json"),
            workspace.root(),
        )
    }

    fn attach(
        runtime: &SessionRuntime,
        workspace: &Workspace,
        store: WorkspaceStateStore,
    ) -> PlannerRuntime {
        PlannerRuntime::attach_with_store(
            runtime,
            workspace.clone(),
            workspace.all(),
            "base prompt".to_owned(),
            false,
            store,
        )
        .expect("attach")
    }

    fn has_tool(runtime: &SessionRuntime, name: &str) -> bool {
        runtime
            .agent()
            .state()
            .tools
            .iter()
            .any(|tool| tool.name == name)
    }

    fn planner_notices(runtime: &SessionRuntime) -> Vec<String> {
        runtime
            .drain_notices()
            .into_iter()
            .filter(|notice| notice.kind == "Planner")
            .map(|notice| notice.text)
            .collect()
    }

    fn session_phase(runtime: &SessionRuntime) -> Option<Phase> {
        let raw = runtime.restored().custom.get(CUSTOM_TYPE)?.clone();
        Some(
            serde_json::from_value::<State>(raw)
                .expect("decode state")
                .phase,
        )
    }

    fn planning() -> State {
        State {
            phase: Phase::Planning,
            ..State::default()
        }
    }

    #[test]
    fn planner_toggle_rebuilds_tools_and_records_its_state() {
        let root = temporary_path("toggle");
        fs::create_dir_all(&root).expect("create root");
        let mut runtime = runtime(&root);
        let workspace = Workspace::new(&root).expect("workspace");
        let store = store(&root, &workspace);
        let integration = attach(&runtime, &workspace, store.clone());

        assert!(!has_tool(&runtime, plannotator::SUBMIT_TOOL_NAME));
        assert_eq!(integration.toggle(), Phase::Planning);
        let state = runtime.agent().state();
        assert!(
            state
                .tools
                .iter()
                .any(|tool| tool.name == plannotator::SUBMIT_TOOL_NAME)
        );
        assert!(!state.tools.iter().any(|tool| tool.name == "bash"));
        assert!(
            state
                .system_prompt
                .contains(plannotator::PLANNING_PROMPT.trim())
        );
        let write = state
            .tools
            .iter()
            .find(|tool| tool.name == "write")
            .expect("write tool");
        let error = (write.execute)(
            agent::CancellationToken::default(),
            "write-call".to_owned(),
            BTreeMap::from([("path".to_owned(), json!("src/main.rs"))]),
            Arc::new(|_| {}),
        )
        .expect_err("planning gate must block a source write");
        assert!(error.contains("writes and edits are limited"));

        // The change reaches both the session log and the workspace file.
        assert_eq!(session_phase(&runtime), Some(Phase::Planning));
        assert_eq!(integration.workspace_state_path(), store.path());
        let (saved, _) = store.load().expect("load").expect("workspace state");
        assert_eq!(saved, planning());

        drop(integration);
        runtime.close().expect("close");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn workspace_store_round_trips_state_with_fingerprints() {
        let root = temporary_path("store");
        fs::create_dir_all(&root).expect("create root");
        let store = WorkspaceStateStore::at(root.join("agent/planner/state.json"), &root);
        assert!(store.load().expect("load missing").is_none());
        assert_eq!(store.fingerprint(), None);

        let first = store.save(&planning()).expect("save");
        assert_eq!(store.fingerprint(), Some(first));
        let (state, fingerprint) = store.load().expect("load").expect("saved state");
        assert_eq!(state, planning());
        assert_eq!(fingerprint, first);
        let raw: serde_json::Value =
            serde_json::from_slice(&fs::read(store.path()).expect("read file")).expect("json");
        assert_eq!(raw["version"], 1);
        assert_eq!(raw["workspace"], root.to_string_lossy().as_ref());
        assert_eq!(raw["state"]["phase"], "planning");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let directory = store.path().parent().expect("parent");
            let mode = |path: &Path| fs::metadata(path).expect("metadata").permissions().mode();
            assert_eq!(mode(store.path()) & 0o777, 0o600);
            assert_eq!(mode(directory) & 0o777, 0o700);
        }

        let second = store.save(&State::default()).expect("replace");
        assert_ne!(second, first);
        assert_eq!(store.fingerprint(), Some(second));
        let (state, _) = store.load().expect("load").expect("replaced state");
        assert_eq!(state, State::default());
        let leftovers = fs::read_dir(store.path().parent().expect("parent"))
            .expect("list")
            .count();
        assert_eq!(leftovers, 1, "temporary files must not remain");

        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn unusable_workspace_files_are_ignored_with_a_notice() {
        let root = temporary_path("unusable");
        fs::create_dir_all(&root).expect("create root");
        let workspace = Workspace::new(&root).expect("workspace");
        let store = store(&root, &workspace);
        fs::create_dir_all(store.path().parent().expect("parent")).expect("create parent");

        fs::write(store.path(), b"{not json").expect("write malformed");
        let error = store.load().expect_err("malformed file");
        assert!(error.to_string().contains("malformed"), "{error}");

        fs::write(
            store.path(),
            vec![b' '; MAX_WORKSPACE_STATE_BYTES as usize + 1],
        )
        .expect("write oversized");
        let error = store.load().expect_err("oversized file");
        assert!(error.to_string().contains("limit"), "{error}");

        WorkspaceStateStore::at(store.path(), Path::new("/elsewhere"))
            .save(&planning())
            .expect("save foreign state");
        let error = store.load().expect_err("foreign file");
        assert!(
            error.to_string().contains("belongs to workspace"),
            "{error}"
        );

        let mut runtime = no_session_runtime(&root);
        let integration = attach(&runtime, &workspace, store.clone());
        assert_eq!(integration.manager().state().phase, Phase::Idle);
        let notices = planner_notices(&runtime);
        assert_eq!(notices.len(), 1, "{notices:?}");
        assert!(notices[0].starts_with("ignoring workspace planner state"));
        assert!(notices[0].contains("belongs to workspace /elsewhere"));
        // An unchanged unusable file is not reported again before every turn.
        integration.sync_agent();
        assert!(planner_notices(&runtime).is_empty());

        drop(integration);
        runtime.close().expect("close");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn no_session_run_restores_the_workspace_phase_and_writes_toggles() {
        let root = temporary_path("no-session");
        fs::create_dir_all(&root).expect("create root");
        let workspace = Workspace::new(&root).expect("workspace");
        let store = store(&root, &workspace);
        store.save(&planning()).expect("seed workspace state");

        let mut runtime = no_session_runtime(&root);
        assert!(!runtime.recording());
        let integration = attach(&runtime, &workspace, store.clone());
        assert_eq!(integration.manager().state().phase, Phase::Planning);
        assert!(has_tool(&runtime, plannotator::SUBMIT_TOOL_NAME));

        assert_eq!(integration.toggle(), Phase::Idle);
        assert!(!has_tool(&runtime, plannotator::SUBMIT_TOOL_NAME));
        let (saved, _) = store.load().expect("load").expect("workspace state");
        assert_eq!(saved, State::default());
        assert_eq!(session_phase(&runtime), None);

        drop(integration);
        runtime.close().expect("close");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn workspace_file_wins_over_the_session_entry_which_remains_the_fallback() {
        let root = temporary_path("precedence");
        fs::create_dir_all(&root).expect("create root");
        let workspace = Workspace::new(&root).expect("workspace");
        let store = store(&root, &workspace);

        let mut first = runtime(&root);
        let integration = attach(&first, &workspace, store.clone());
        assert_eq!(integration.toggle(), Phase::Planning);
        // Only a session with a message is offered to `-continue`.
        first.agent().prompt("plan this").expect("prompt");
        drop(integration);
        first.close().expect("close first");

        // Another window returned this workspace to idle after the session
        // recorded planning: the file wins on resume.
        store
            .save(&State::default())
            .expect("overwrite workspace state");
        let mut second = SessionRuntime::open(SessionOptions {
            selection: SessionSelection::Continue,
            ..options(&root)
        })
        .expect("continue session");
        assert!(second.resumed());
        assert_eq!(session_phase(&second), Some(Phase::Planning));
        let integration = attach(&second, &workspace, store.clone());
        assert_eq!(integration.manager().state().phase, Phase::Idle);
        drop(integration);
        second.close().expect("close second");

        // Without a workspace file the session entry still restores the phase.
        fs::remove_file(store.path()).expect("remove workspace state");
        let mut third = SessionRuntime::open(SessionOptions {
            selection: SessionSelection::Continue,
            ..options(&root)
        })
        .expect("continue session again");
        let integration = attach(&third, &workspace, store.clone());
        assert_eq!(integration.manager().state().phase, Phase::Planning);
        assert!(planner_notices(&third).is_empty());

        drop(integration);
        third.close().expect("close third");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn toggle_in_one_window_is_adopted_by_the_other_before_its_next_turn() {
        let root = temporary_path("windows");
        fs::create_dir_all(&root).expect("create root");
        let workspace = Workspace::new(&root).expect("workspace");
        let store = store(&root, &workspace);
        let mut first = no_session_runtime(&root);
        let mut second = runtime(&root);
        let one = attach(&first, &workspace, store.clone());
        let two = attach(&second, &workspace, store.clone());
        planner_notices(&second);

        assert_eq!(one.toggle(), Phase::Planning);
        assert_eq!(two.manager().state().phase, Phase::Idle);
        assert!(!has_tool(&second, plannotator::SUBMIT_TOOL_NAME));

        // Before a model request the runtime synchronizes its extensions;
        // that is where a change from another window is picked up.
        two.sync_agent();
        assert_eq!(two.manager().state().phase, Phase::Planning);
        assert!(has_tool(&second, plannotator::SUBMIT_TOOL_NAME));
        assert!(
            second
                .agent()
                .state()
                .system_prompt
                .contains(plannotator::PLANNING_PROMPT.trim())
        );
        assert_eq!(
            planner_notices(&second),
            vec!["state updated by another window: now planning".to_owned()]
        );
        assert_eq!(session_phase(&second), Some(Phase::Planning));
        two.sync_agent();
        assert!(planner_notices(&second).is_empty());

        // The turn-end subscription adopts too, so a change made while a
        // turn runs applies before the next model request.
        assert_eq!(one.toggle(), Phase::Idle);
        second.agent().prompt("continue").expect("prompt");
        assert_eq!(two.manager().state().phase, Phase::Idle);
        assert!(!has_tool(&second, plannotator::SUBMIT_TOOL_NAME));
        assert_eq!(
            planner_notices(&second),
            vec!["state updated by another window: now idle".to_owned()]
        );

        // A toggle flips the shared phase, not a stale one, and the first
        // window sees it in turn.
        assert_eq!(two.toggle(), Phase::Planning);
        assert!(one.adopt_external_change());
        assert_eq!(one.manager().state().phase, Phase::Planning);
        assert!(!one.adopt_external_change());

        drop(one);
        drop(two);
        first.close().expect("close first");
        second.close().expect("close second");
        fs::remove_dir_all(root).expect("remove root");
    }

    #[test]
    fn review_output_reader_keeps_a_bounded_prefix_and_drains_the_rest() {
        let output = read_bounded(b"abcdef".as_slice(), 3).expect("read");
        assert_eq!(output.bytes, b"abc");
        assert!(output.truncated);
    }

    #[test]
    fn planner_review_requires_an_omitted_or_github_url_target() {
        let error = load_diff_review(".", "not-a-url").expect_err("reject target");
        assert_eq!(error, "usage: /planner-review [GitHub PR URL]");
    }

    #[cfg(unix)]
    #[test]
    fn review_diff_commands_do_not_inherit_stdin() {
        let output = run_bounded_command(
            Command::new("sh")
                .arg("-c")
                .arg("if read -r line; then echo got; else echo eof; fi"),
        )
        .expect("run helper");
        assert_eq!(output.trim(), "eof");
    }

    /// Waits for cancellation the way an unattended browser review does.
    struct BlockingReviewer;

    impl plannotator::Reviewer for BlockingReviewer {
        fn review(
            &self,
            cancellation: &agent::CancellationToken,
            _: &plannotator::ReviewRequest,
        ) -> std::result::Result<plannotator::Decision, plannotator::ReviewError> {
            let give_up = Instant::now() + Duration::from_secs(10);
            while !cancellation.is_cancelled() {
                if Instant::now() > give_up {
                    return Err(plannotator::ReviewError::Failed(
                        "never cancelled".to_owned(),
                    ));
                }
                thread::sleep(Duration::from_millis(1));
            }
            Err(plannotator::ReviewError::Cancelled)
        }
    }

    #[test]
    fn slash_reviews_are_cancelled_at_their_deadline() {
        let root = temporary_path("deadline");
        fs::create_dir_all(&root).expect("create root");
        let mut runtime = runtime(&root);
        let handle = PlannerReviewHandle {
            reviewer: Arc::new(BlockingReviewer),
            notices: runtime.notice_sender(),
            cancellation: Arc::new(Mutex::new(None)),
        };
        let request = plannotator::ReviewRequest::new("Review", "# Plan");
        assert_eq!(
            handle.review_within(&request, Duration::from_millis(50)),
            Err(plannotator::ReviewError::Cancelled)
        );
        assert!(
            !handle.cancel(),
            "a finished review must not stay cancellable"
        );
        runtime.close().expect("close");
        fs::remove_dir_all(root).expect("remove root");
    }
}
