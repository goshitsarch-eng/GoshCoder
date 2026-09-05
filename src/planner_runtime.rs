//! Session and agent integration for the native planner.
//!
//! The planner core deliberately has no terminal or session dependency. This
//! adapter owns the live joins: restoring durable state, recording state
//! changes, rebuilding the model-visible tool list, enforcing the planning
//! write gate, and forwarding browser-review notices to the active frontend.

use std::{
    collections::BTreeSet,
    fmt,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::{Arc, Mutex, MutexGuard},
    thread,
    time::{Duration, Instant},
};

use crate::{
    agent, llm, plannotator,
    session::{SessionCustomRecorder, SessionNoticeSender, SessionRuntime},
    tools::Workspace,
};

/// pi-compatible custom entry type used for one session's planner state.
pub const CUSTOM_TYPE: &str = "goshcoder.planner";

/// Errors that prevent a planner from being attached to a prepared session.
#[derive(Debug)]
pub enum PlannerRuntimeError {
    Planner(plannotator::PlannerError),
}

impl fmt::Display for PlannerRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Planner(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for PlannerRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Planner(error) => Some(error),
        }
    }
}

impl From<plannotator::PlannerError> for PlannerRuntimeError {
    fn from(error: plannotator::PlannerError) -> Self {
        Self::Planner(error)
    }
}

pub type Result<T> = std::result::Result<T, PlannerRuntimeError>;

/// Keeps a planner attached to one session and its live agent.
///
/// Its subscription must remain alive for the complete session: it applies
/// `[DONE:n]` markers and changes from `planner_submit_plan` before the agent
/// requests its next model turn.
pub struct PlannerRuntime {
    manager: plannotator::Manager,
    agent: agent::Agent,
    workspace: Workspace,
    normal_tools: Vec<agent::Tool>,
    base_system_prompt: Arc<Mutex<String>>,
    reviewer: Arc<dyn plannotator::Reviewer>,
    notices: SessionNoticeSender,
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

impl PlannerReviewHandle {
    /// Opens the configured review surface and waits for a decision.
    pub fn review(
        &self,
        request: &plannotator::ReviewRequest,
    ) -> std::result::Result<plannotator::Decision, plannotator::ReviewError> {
        let cancellation = agent::CancellationToken::default();
        *lock(&self.cancellation) = Some(cancellation.clone());
        let result = self.reviewer.review(&cancellation, request);
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
        let notices = runtime.notice_sender();
        let initial = restored_state(runtime, &notices);
        let recorder = runtime.custom_recorder();
        let reviewer: Arc<dyn plannotator::Reviewer> = Arc::new(plannotator::BrowserReviewer {
            notify: Some(review_notice_callback(notices.clone())),
            ..plannotator::BrowserReviewer::default()
        });
        let manager = plannotator::Manager::new(
            workspace.root(),
            Some(Arc::clone(&reviewer)),
            plannotator::Options {
                initial,
                on_change: Some(persistence_callback(recorder, notices.clone())),
                warn: Some(warning_callback(notices.clone())),
            },
        )?;

        if start_in_planning && manager.state().phase == plannotator::Phase::Idle {
            manager.enter();
        }

        let agent = runtime.agent().clone();
        let base_system_prompt = Arc::new(Mutex::new(base_system_prompt));
        let subscription = planner_subscription(
            &agent,
            manager.clone(),
            workspace.clone(),
            normal_tools.clone(),
            Arc::clone(&base_system_prompt),
        );
        let integration = Self {
            manager,
            agent,
            workspace,
            normal_tools,
            base_system_prompt,
            reviewer,
            notices,
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
    pub fn toggle(&self) -> plannotator::Phase {
        let phase = self.manager.toggle();
        self.sync_agent();
        phase
    }

    /// Replaces the base prompt while retaining the current planner suffix.
    pub fn set_base_system_prompt(&self, prompt: impl Into<String>) {
        *lock(&self.base_system_prompt) = prompt.into();
        self.sync_agent();
    }

    /// Rebuilds prompt and tools after an external integration changes base
    /// state. The method is idempotent and is safe to call from a UI loop.
    pub fn sync_agent(&self) {
        sync_agent(
            &self.agent,
            &self.manager,
            &self.workspace,
            &self.normal_tools,
            &self.base_system_prompt,
        );
    }
}

fn restored_state(
    runtime: &SessionRuntime,
    notices: &SessionNoticeSender,
) -> Option<plannotator::State> {
    let restored = runtime.restored();
    let Some(raw) = restored.custom.get(CUSTOM_TYPE) else {
        return None;
    };
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

fn persistence_callback(
    recorder: SessionCustomRecorder,
    notices: SessionNoticeSender,
) -> plannotator::StateCallback {
    Arc::new(move |state| {
        // A no-session or read-only session intentionally keeps planner state
        // only in memory. It should not turn a harmless toggle into a noisy
        // persistence error.
        if !recorder.recording() {
            return;
        }
        let payload = match serde_json::to_value(state) {
            Ok(payload) => payload,
            Err(error) => {
                notices.push(
                    "Planner",
                    format!("could not encode planner state for the session: {error}"),
                );
                return;
            }
        };
        if let Err(error) = recorder.record(CUSTOM_TYPE, payload) {
            notices.push(
                "Planner",
                format!("could not save planner state to the session: {error}"),
            );
        }
    })
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
    normal_tools: Vec<agent::Tool>,
    base_system_prompt: Arc<Mutex<String>>,
) -> agent::Subscription {
    let agent = agent.clone();
    agent.clone().subscribe(move |event| {
        if event.kind != agent::EventKind::TurnEnd {
            return;
        }
        if let Some(llm::Message::Assistant(message)) = event.message {
            manager.track_assistant(&message);
        }
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
    agent.set_system_prompt(manager.prompt(base));
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
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
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
    use crate::{llm, session::SessionOptions};
    use serde_json::json;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_path(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "goshcoder-planner-runtime-{label}-{}-{sequence}",
            std::process::id()
        ))
    }

    fn runtime(root: &std::path::Path) -> SessionRuntime {
        SessionRuntime::open(SessionOptions {
            cwd: root.to_path_buf(),
            sessions_dir: Some(root.join("sessions")),
            model: llm::Model {
                provider: "test".to_owned(),
                id: "test".to_owned(),
                api: "test".to_owned(),
                ..llm::Model::default()
            },
            ..SessionOptions::default()
        })
        .expect("open session")
    }

    #[test]
    fn planner_toggle_rebuilds_tools_and_records_its_state() {
        let root = temporary_path("toggle");
        fs::create_dir_all(&root).expect("create root");
        let mut runtime = runtime(&root);
        let workspace = Workspace::new(&root).expect("workspace");
        let integration = PlannerRuntime::attach(
            &runtime,
            workspace.clone(),
            workspace.all(),
            "base prompt".to_owned(),
            false,
        )
        .expect("attach");

        assert!(
            !runtime
                .agent()
                .state()
                .tools
                .iter()
                .any(|tool| tool.name == plannotator::SUBMIT_TOOL_NAME)
        );
        assert_eq!(integration.toggle(), plannotator::Phase::Planning);
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

        let restored = runtime.restored();
        assert_eq!(
            serde_json::from_value::<plannotator::State>(
                restored
                    .custom
                    .get(CUSTOM_TYPE)
                    .cloned()
                    .expect("saved planner state"),
            )
            .expect("decode state")
            .phase,
            plannotator::Phase::Planning
        );

        drop(integration);
        runtime.close().expect("close");
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
}
