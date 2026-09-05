//! Testable runtime for independent `/btw` side threads.
//!
//! [`crate::btw`] owns the portable thread, context, settings, and export
//! primitives. This module owns the runtime concerns around those primitives:
//! model resolution, request dispatch, cancellation, and per-thread queues.
//! It deliberately accepts a snapshot of [`agent::State`] and a supplied
//! [`agent::AssistantResponder`], so dispatching a side question cannot append
//! to or otherwise mutate the main agent transcript.

use std::{
    collections::BTreeMap,
    error::Error as StdError,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
};

use crate::{agent, btw, catalog::Catalog, config, llm};

/// Result type returned by [`Runtime`] operations.
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Resolves a configured `provider/model` reference into model metadata.
///
/// Model-authentication failures should be returned as display-safe strings;
/// they become a warning and cause [`Runtime::resolve_selection`] to fall back
/// to the main agent's current model.
pub type ModelResolver =
    Arc<dyn Fn(&str) -> std::result::Result<llm::Model, String> + Send + Sync + 'static>;

/// The effective side-thread model, thinking level, and non-fatal warnings.
pub type Selection = btw::ResolvedSelection;

/// Construction settings for a [`Runtime`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Options {
    /// Location of the independently managed `pi-btw.json` file.
    pub settings_path: PathBuf,
    /// Prefix used to distinguish provider-side side-thread request IDs.
    ///
    /// A request for `btw-4` uses `<side_session_id_prefix>:btw-4`. An empty
    /// prefix falls back to `btw`.
    pub side_session_id_prefix: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            settings_path: config::btw_path(),
            side_session_id_prefix: "btw".to_owned(),
        }
    }
}

/// A new thread and the settings selection used to initialize it.
#[derive(Clone, Debug, PartialEq)]
pub struct CreateOutcome {
    pub thread: btw::Thread,
    pub selection: Selection,
}

/// Per-thread queue state for a frontend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueueStatus {
    pub thread_id: String,
    pub queued: usize,
    pub running: bool,
}

/// The terminal state of one dispatched side question.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DispatchStatus {
    /// A response was recorded as a successful side-thread turn.
    Answered { answer: String },
    /// A provider or responder failure was recorded as an error turn.
    Failed { message: String },
    /// The request was cancelled and deliberately did not create a turn.
    Cancelled,
}

/// Details produced after [`Runtime::run_next`] dispatches one queued question.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchOutcome {
    /// Snapshot after this request was recorded, or after cancellation.
    pub thread: btw::Thread,
    /// The normalized question sent to the responder.
    pub question: String,
    /// Whether the caller queued this as an initial prompt or a follow-up.
    pub question_kind: btw::QueuedQuestionKind,
    /// Provider result after cancellation and error handling.
    pub status: DispatchStatus,
    /// Model and thinking settings used for this request.
    pub selection: Selection,
    /// Questions still waiting after this request completed.
    pub queued: usize,
}

/// Deterministic, ready-to-insert output for bringing side context to main.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BringOutput {
    pub thread_id: String,
    pub segments: Vec<btw::Segment>,
    pub text: String,
    pub estimated_tokens: usize,
}

/// Result of changing a thread-local thinking level.
#[derive(Clone, Debug, PartialEq)]
pub struct ThinkingLevelChange {
    /// The level after clamping it to the independently selected side model.
    pub thinking_level: llm::ThinkingLevel,
    /// Whether the new level was saved to `pi-btw.json`.
    pub remembered: bool,
    /// A persistence failure that did not undo the in-memory thread change.
    pub persistence_error: Option<String>,
    /// Selection warnings encountered while choosing the side model.
    pub selection: Selection,
}

/// Runtime-level failures that are distinct from a side-model response failure.
#[derive(Debug)]
pub enum RuntimeError {
    EmptyQuestion,
    EmptyThinkingLevel,
    UnknownThread(String),
    ThreadBusy(String),
    NoSuccessfulAnswer(String),
    Settings(btw::SettingsError),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyQuestion => formatter.write_str("a BTW question cannot be empty"),
            Self::EmptyThinkingLevel => formatter.write_str("a BTW thinking level cannot be empty"),
            Self::UnknownThread(id) => write!(formatter, "unknown BTW thread {id:?}"),
            Self::ThreadBusy(id) => write!(formatter, "BTW thread {id:?} is already answering"),
            Self::NoSuccessfulAnswer(id) => {
                write!(
                    formatter,
                    "BTW thread {id:?} has no successful answer to bring"
                )
            }
            Self::Settings(error) => error.fmt(formatter),
        }
    }
}

impl StdError for RuntimeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Settings(error) => Some(error),
            Self::EmptyQuestion
            | Self::EmptyThinkingLevel
            | Self::UnknownThread(_)
            | Self::ThreadBusy(_)
            | Self::NoSuccessfulAnswer(_) => None,
        }
    }
}

impl From<btw::SettingsError> for RuntimeError {
    fn from(error: btw::SettingsError) -> Self {
        Self::Settings(error)
    }
}

/// Thread-safe side-thread runtime suitable for a terminal frontend worker.
///
/// Clones share the same in-memory threads and queues. A frontend can enqueue
/// work on its event loop and call [`run_next`](Self::run_next) on a worker
/// with a cloned runtime; [`cancel`](Self::cancel) remains available from the
/// event loop while the responder is running.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    manager: btw::Manager,
    responder: agent::AssistantResponder,
    resolve_model: ModelResolver,
    settings_path: PathBuf,
    side_session_id_prefix: String,
    thread_state: Mutex<BTreeMap<String, ThreadRuntimeState>>,
}

#[derive(Default)]
struct ThreadRuntimeState {
    queue: btw::QuestionQueue,
    activity: Option<Activity>,
}

enum Activity {
    Running(agent::CancellationToken),
    Completing,
}

enum ResponderOutcome {
    Answered(Box<llm::AssistantMessage>),
    Failed(String),
    Cancelled,
}

impl Runtime {
    /// Creates a runtime with an injectable side-model resolver.
    pub fn new<F>(responder: agent::AssistantResponder, resolve_model: F, options: Options) -> Self
    where
        F: Fn(&str) -> std::result::Result<llm::Model, String> + Send + Sync + 'static,
    {
        Self::with_model_resolver(responder, Arc::new(resolve_model), options)
    }

    /// Creates a runtime from a reusable resolver callback.
    pub fn with_model_resolver(
        responder: agent::AssistantResponder,
        resolve_model: ModelResolver,
        options: Options,
    ) -> Self {
        Self {
            inner: Arc::new(RuntimeInner {
                manager: btw::Manager::new(),
                responder,
                resolve_model,
                settings_path: options.settings_path,
                side_session_id_prefix: options.side_session_id_prefix,
                thread_state: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    /// Creates a runtime whose configured side model is resolved through a
    /// normal authenticated catalog.
    pub fn with_catalog(
        responder: agent::AssistantResponder,
        catalog: Catalog,
        options: Options,
    ) -> Self {
        Self::new(
            responder,
            move |reference| {
                catalog
                    .resolve_model(reference)
                    .map(|resolved| resolved.model)
                    .map_err(|error| error.to_string())
            },
            options,
        )
    }

    /// Returns the `pi-btw.json` path used by this runtime.
    #[must_use]
    pub fn settings_path(&self) -> &Path {
        &self.inner.settings_path
    }

    /// Reads the current `pi-btw.json` document without creating it.
    #[must_use]
    pub fn read_settings(&self) -> btw::SettingsResult {
        btw::read_settings(self.settings_path())
    }

    /// Applies a validated, non-destructive settings patch.
    pub fn update_settings(&self, patch: btw::SettingsPatch) -> Result<btw::Settings> {
        btw::update_settings(self.settings_path(), patch).map_err(Into::into)
    }

    /// Resolves the configured side model and thinking level against a main
    /// agent state snapshot.
    ///
    /// Invalid settings and unavailable configured models are warnings; both
    /// cases retain an independently owned copy of the current main model.
    #[must_use]
    pub fn resolve_selection(&self, main_state: &agent::State) -> Selection {
        let settings = self.read_settings();
        let mut selection = btw::resolve_selection(
            &main_state.model,
            &main_state.thinking_level,
            &settings.settings,
            |reference| (self.inner.resolve_model)(reference),
        );
        if settings.kind == btw::SettingsKind::Invalid {
            selection
                .warnings
                .insert(0, format!("pi-btw settings ignored: {}", settings.reason));
        }
        selection
    }

    /// Starts a new in-memory side thread with a bounded, read-only snapshot
    /// of the durable main transcript.
    #[must_use]
    pub fn create_thread(&self, main_state: &agent::State) -> CreateOutcome {
        let selection = self.resolve_selection(main_state);
        let conversation_context = btw::build_conversation_context(&main_state.messages);
        let thread = self
            .inner
            .manager
            .new_thread(conversation_context, selection.thinking_level.clone());
        lock(&self.inner.thread_state)
            .entry(thread.id.clone())
            .or_default();
        CreateOutcome { thread, selection }
    }

    /// Reopens an existing in-memory side thread for rendering or dispatch.
    ///
    /// Threads are intentionally session-local. This does not restore a
    /// thread from disk; it returns a fresh immutable snapshot of one created
    /// by this runtime.
    pub fn resume_thread(&self, thread: impl AsRef<str>) -> Result<btw::Thread> {
        let id = thread.as_ref().to_owned();
        let snapshot = self.require_thread(&id)?;
        lock(&self.inner.thread_state).entry(id).or_default();
        Ok(snapshot)
    }

    /// Returns one immutable thread snapshot.
    pub fn thread(&self, thread: impl AsRef<str>) -> Result<btw::Thread> {
        self.require_thread(thread.as_ref())
    }

    /// Lists non-empty side threads in deterministic most-recently-updated
    /// order.
    #[must_use]
    pub fn list_threads(&self) -> Vec<btw::Summary> {
        self.inner.manager.list()
    }

    /// Queues an initial side prompt for a thread.
    pub fn enqueue_prompt(
        &self,
        thread: impl AsRef<str>,
        question: impl Into<String>,
    ) -> Result<QueueStatus> {
        self.enqueue(
            thread.as_ref(),
            question.into(),
            btw::QueuedQuestionKind::Prompt,
        )
    }

    /// Queues a side follow-up without allowing it to overtake earlier work.
    pub fn enqueue_follow_up(
        &self,
        thread: impl AsRef<str>,
        question: impl Into<String>,
    ) -> Result<QueueStatus> {
        self.enqueue(
            thread.as_ref(),
            question.into(),
            btw::QueuedQuestionKind::FollowUp,
        )
    }

    /// Returns whether a thread has a running request and how many questions
    /// still await dispatch.
    pub fn queue_status(&self, thread: impl AsRef<str>) -> Result<QueueStatus> {
        let id = thread.as_ref().to_owned();
        self.require_thread(&id)?;
        let state = lock(&self.inner.thread_state);
        let thread_state = state.get(&id);
        Ok(QueueStatus {
            thread_id: id,
            queued: thread_state.map_or(0, |state| state.queue.len()),
            running: thread_state.is_some_and(|state| state.activity.is_some()),
        })
    }

    /// Removes pending questions without affecting a currently running one.
    pub fn clear_queue(&self, thread: impl AsRef<str>) -> Result<usize> {
        let id = thread.as_ref().to_owned();
        self.require_thread(&id)?;
        let mut states = lock(&self.inner.thread_state);
        let state = states.entry(id).or_default();
        let removed = state.queue.len();
        state.queue.clear();
        Ok(removed)
    }

    /// Cooperatively cancels a currently running request.
    ///
    /// This does not remove pending follow-ups. Use [`clear_queue`](Self::clear_queue)
    /// as well when closing a side-thread view should discard them.
    pub fn cancel(&self, thread: impl AsRef<str>) -> Result<bool> {
        let id = thread.as_ref().to_owned();
        self.require_thread(&id)?;
        let states = lock(&self.inner.thread_state);
        let Some(state) = states.get(&id) else {
            return Ok(false);
        };
        let Some(Activity::Running(cancellation)) = state.activity.as_ref() else {
            return Ok(false);
        };
        cancellation.cancel();
        Ok(true)
    }

    /// Runs exactly one queued question for a thread.
    ///
    /// The caller owns scheduling: a UI may invoke this on a worker, then
    /// react to `queued` in the returned outcome by scheduling another call.
    /// The supplied `main_state` is read only. Its messages are used only for
    /// model selection and the context captured when the thread was created;
    /// no main-agent API is called here.
    pub fn run_next(
        &self,
        main_state: &agent::State,
        thread: impl AsRef<str>,
    ) -> Result<Option<DispatchOutcome>> {
        let id = thread.as_ref().to_owned();
        self.require_thread(&id)?;
        let Some((queued_question, cancellation)) = self.start_next(&id)? else {
            return Ok(None);
        };

        let thread_before = match self.inner.manager.snapshot(&id) {
            Some(thread) => thread,
            None => {
                self.finish_activity(&id);
                return Err(RuntimeError::UnknownThread(id));
            }
        };
        let mut selection = self.resolve_selection(main_state);
        let thinking_level =
            btw::clamp_thinking_level(&selection.model, &thread_before.thinking_level);
        selection.thinking_level = thinking_level.clone();
        if let Err(error) = self.inner.manager.set_thinking_level(&id, thinking_level) {
            self.finish_activity(&id);
            return Err(runtime_thread_error(error));
        }
        let thread_for_request = match self.inner.manager.snapshot(&id) {
            Some(thread) => thread,
            None => {
                self.finish_activity(&id);
                return Err(RuntimeError::UnknownThread(id));
            }
        };
        let context = btw::build_request_context(&thread_for_request, &queued_question.question);
        let responder_outcome = self.respond(
            &selection.model,
            &context,
            selection.thinking_level.clone(),
            self.request_session_id(&id),
            cancellation,
        );

        // This lock is the linearization point between cancel and a completed
        // response. Once the request enters Completing, cancellation returns
        // false and the side turn is durably recorded exactly once.
        let cancelled = self.enter_completion(&id);
        let status_result = if cancelled {
            Ok(DispatchStatus::Cancelled)
        } else {
            match responder_outcome {
                ResponderOutcome::Answered(response) => self
                    .inner
                    .manager
                    .record_answered(&id, queued_question.question.clone(), *response)
                    .map(|answer| DispatchStatus::Answered { answer })
                    .map_err(runtime_thread_error),
                ResponderOutcome::Failed(message) => self
                    .inner
                    .manager
                    .record_error(&id, queued_question.question.clone(), message.clone())
                    .map(|()| DispatchStatus::Failed { message })
                    .map_err(runtime_thread_error),
                ResponderOutcome::Cancelled => Ok(DispatchStatus::Cancelled),
            }
        };
        let snapshot_result = self.require_thread(&id);
        let queued = self.finish_activity(&id);
        let status = status_result?;
        let thread = snapshot_result?;

        Ok(Some(DispatchOutcome {
            thread,
            question: queued_question.question,
            question_kind: queued_question.kind,
            status,
            selection,
            queued,
        }))
    }

    /// Changes a thread-local level and persists it only when the current
    /// `pi-btw.json` preference permits remembering such changes.
    ///
    /// A settings-write failure is returned in [`ThinkingLevelChange`] because
    /// the local thread change has already succeeded and should remain usable.
    pub fn set_thread_thinking_level(
        &self,
        main_state: &agent::State,
        thread: impl AsRef<str>,
        requested: impl AsRef<str>,
    ) -> Result<ThinkingLevelChange> {
        let id = thread.as_ref().to_owned();
        self.require_thread(&id)?;
        let requested = requested.as_ref().trim();
        if requested.is_empty() {
            return Err(RuntimeError::EmptyThinkingLevel);
        }

        let mut selection = self.resolve_selection(main_state);
        let thinking_level = btw::clamp_thinking_level(&selection.model, requested);
        self.inner
            .manager
            .set_thinking_level(&id, thinking_level.clone())
            .map_err(runtime_thread_error)?;
        selection.thinking_level = thinking_level.clone();

        let mut remembered = false;
        let mut persistence_error = None;
        if selection.remember_thinking_level_changes {
            let patch = btw::SettingsPatch {
                thinking_level: btw::SettingChange::Set(thinking_level.clone()),
                ..btw::SettingsPatch::default()
            };
            match btw::update_settings(self.settings_path(), patch) {
                Ok(_) => remembered = true,
                Err(error) => persistence_error = Some(error.to_string()),
            }
        }

        Ok(ThinkingLevelChange {
            thinking_level,
            remembered,
            persistence_error,
            selection,
        })
    }

    /// Selects successful turns using [`btw::BringSelection`] and produces the
    /// exact escaped main-context wrapper provided by [`crate::btw`].
    pub fn export(
        &self,
        thread: impl AsRef<str>,
        selection: btw::BringSelection,
    ) -> Result<BringOutput> {
        let thread = self.require_thread(thread.as_ref())?;
        let segments = btw::select_bring_segments(&thread, selection);
        if segments.is_empty() {
            return Err(RuntimeError::NoSuccessfulAnswer(thread.id));
        }
        let text = btw::format_bring_to_main(&segments);
        let estimated_tokens = btw::estimate_tokens(&segments);
        Ok(BringOutput {
            thread_id: thread.id,
            segments,
            text,
            estimated_tokens,
        })
    }

    /// Alias for [`export`](Self::export) used by a frontend's "bring to main"
    /// action.
    pub fn bring_to_main(
        &self,
        thread: impl AsRef<str>,
        selection: btw::BringSelection,
    ) -> Result<BringOutput> {
        self.export(thread, selection)
    }

    fn enqueue(
        &self,
        thread: &str,
        question: String,
        kind: btw::QueuedQuestionKind,
    ) -> Result<QueueStatus> {
        let id = thread.to_owned();
        self.require_thread(&id)?;
        let question = question.trim().to_owned();
        if question.is_empty() {
            return Err(RuntimeError::EmptyQuestion);
        }

        let mut states = lock(&self.inner.thread_state);
        let state = states.entry(id.clone()).or_default();
        match kind {
            btw::QueuedQuestionKind::Prompt => state.queue.enqueue_prompt(question),
            btw::QueuedQuestionKind::FollowUp => state.queue.enqueue_follow_up(question),
        }
        Ok(QueueStatus {
            thread_id: id,
            queued: state.queue.len(),
            running: state.activity.is_some(),
        })
    }

    fn start_next(
        &self,
        thread_id: &str,
    ) -> Result<Option<(btw::QueuedQuestion, agent::CancellationToken)>> {
        let mut states = lock(&self.inner.thread_state);
        let state = states.entry(thread_id.to_owned()).or_default();
        if state.activity.is_some() {
            return Err(RuntimeError::ThreadBusy(thread_id.to_owned()));
        }
        let Some(question) = state.queue.dequeue() else {
            return Ok(None);
        };
        let cancellation = agent::CancellationToken::default();
        state.activity = Some(Activity::Running(cancellation.clone()));
        Ok(Some((question, cancellation)))
    }

    fn respond(
        &self,
        model: &llm::Model,
        context: &llm::Context,
        thinking_level: llm::ThinkingLevel,
        session_id: String,
        cancellation: agent::CancellationToken,
    ) -> ResponderOutcome {
        if cancellation.is_cancelled() {
            return ResponderOutcome::Cancelled;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            (self.inner.responder)(
                model,
                context,
                agent::RequestOptions {
                    cancellation,
                    thinking_level,
                    session_id,
                },
            )
        }));
        match result {
            Ok(Ok(response)) if response_was_cancelled(&response) => ResponderOutcome::Cancelled,
            Ok(Ok(response)) if response_was_error(&response) => {
                ResponderOutcome::Failed(response_error_message(&response))
            }
            Ok(Ok(response)) => ResponderOutcome::Answered(Box::new(response)),
            Ok(Err(error)) if error.trim().is_empty() => {
                ResponderOutcome::Failed("side responder returned an error".to_owned())
            }
            Ok(Err(error)) => ResponderOutcome::Failed(error),
            Err(_) => ResponderOutcome::Failed("side responder panicked".to_owned()),
        }
    }

    fn enter_completion(&self, thread_id: &str) -> bool {
        let mut states = lock(&self.inner.thread_state);
        let Some(state) = states.get_mut(thread_id) else {
            return true;
        };
        let cancelled = match state.activity.take() {
            Some(Activity::Running(cancellation)) => cancellation.is_cancelled(),
            Some(Activity::Completing) | None => true,
        };
        state.activity = Some(Activity::Completing);
        cancelled
    }

    fn finish_activity(&self, thread_id: &str) -> usize {
        let mut states = lock(&self.inner.thread_state);
        let Some(state) = states.get_mut(thread_id) else {
            return 0;
        };
        state.activity = None;
        state.queue.len()
    }

    fn require_thread(&self, thread_id: &str) -> Result<btw::Thread> {
        self.inner
            .manager
            .snapshot(thread_id)
            .ok_or_else(|| RuntimeError::UnknownThread(thread_id.to_owned()))
    }

    fn request_session_id(&self, thread_id: &str) -> String {
        let prefix = self.inner.side_session_id_prefix.trim();
        let prefix = if prefix.is_empty() { "btw" } else { prefix };
        format!("{prefix}:{thread_id}")
    }
}

fn runtime_thread_error(error: btw::ThreadError) -> RuntimeError {
    match error {
        btw::ThreadError::UnknownThread(id) => RuntimeError::UnknownThread(id),
    }
}

fn response_was_cancelled(response: &llm::AssistantMessage) -> bool {
    response.stop_reason.eq_ignore_ascii_case("aborted")
        || response.stop_reason.eq_ignore_ascii_case("cancelled")
}

fn response_was_error(response: &llm::AssistantMessage) -> bool {
    response.stop_reason.eq_ignore_ascii_case("error") || !response.error_message.trim().is_empty()
}

fn response_error_message(response: &llm::AssistantMessage) -> String {
    let message = response.error_message.trim();
    if message.is_empty() {
        "side model returned an error".to_owned()
    } else {
        message.to_owned()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, AtomicUsize, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use super::*;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after the epoch")
            .as_nanos();
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("goshcoder-btw-runtime-{label}-{nonce}-{sequence}"));
        fs::create_dir_all(&path).expect("create test directory");
        path
    }

    fn model(provider: &str, id: &str, reasoning: bool) -> llm::Model {
        llm::Model {
            id: id.to_owned(),
            name: id.to_owned(),
            api: "test".to_owned(),
            provider: provider.to_owned(),
            reasoning,
            ..llm::Model::default()
        }
    }

    fn assistant(text: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            content: vec![llm::ContentBlock::text(text)],
            api: "test".to_owned(),
            provider: "test".to_owned(),
            model: "test-model".to_owned(),
            stop_reason: "stop".to_owned(),
            timestamp: 1,
            ..llm::AssistantMessage::default()
        }
    }

    fn user(text: impl Into<String>) -> llm::Message {
        llm::Message::User(llm::UserMessage::text(text, 1))
    }

    fn main_state(messages: Vec<llm::Message>) -> agent::State {
        agent::State {
            system_prompt: "main instructions".to_owned(),
            model: model("main", "current", true),
            thinking_level: llm::THINKING_MEDIUM.to_owned(),
            tools: Vec::new(),
            messages,
            compactions: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: Vec::new(),
            error_message: String::new(),
        }
    }

    fn options(path: PathBuf) -> Options {
        Options {
            settings_path: path,
            side_session_id_prefix: "test-side".to_owned(),
        }
    }

    fn runtime_without_configured_model(
        responder: agent::AssistantResponder,
        path: PathBuf,
    ) -> Runtime {
        Runtime::new(
            responder,
            |_| Err("no configured side model".to_owned()),
            options(path),
        )
    }

    #[test]
    fn creates_resumes_lists_and_bounds_the_main_context() {
        let directory = test_directory("create");
        let settings_path = btw::settings_path(&directory);
        let state = main_state(vec![user("x".repeat(btw::MAX_CONTEXT_CHARS + 20))]);
        let responder: agent::AssistantResponder = Arc::new(|_, _, _| Ok(assistant("answer")));
        let runtime = runtime_without_configured_model(responder, settings_path.clone());

        let created = runtime.create_thread(&state);
        assert_eq!(created.thread.id, "btw-1");
        assert_eq!(created.thread.thinking_level, llm::THINKING_MEDIUM);
        assert!(
            created
                .thread
                .conversation_context
                .starts_with("[Earlier context omitted; showing the last")
        );
        let marker = format!(
            "[Earlier context omitted; showing the last {} characters.]\n",
            btw::MAX_CONTEXT_CHARS
        );
        assert_eq!(
            created.thread.conversation_context[marker.len()..]
                .chars()
                .count(),
            btw::MAX_CONTEXT_CHARS,
            "only the newest source context is retained"
        );
        assert!(
            !settings_path.exists(),
            "a settings read must not create a file"
        );
        assert!(runtime.list_threads().is_empty());

        let resumed = runtime
            .resume_thread(&created.thread)
            .expect("resume in-memory thread");
        assert_eq!(resumed.id, created.thread.id);
        runtime
            .enqueue_prompt(&created.thread, "  first question  ")
            .expect("queue prompt");
        let outcome = runtime
            .run_next(&state, &created.thread)
            .expect("dispatch")
            .expect("queued prompt");
        assert_eq!(outcome.question, "first question");
        assert_eq!(
            outcome.status,
            DispatchStatus::Answered {
                answer: "answer".to_owned()
            }
        );
        assert_eq!(runtime.list_threads()[0].id, created.thread.id);

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn selection_uses_pi_btw_settings_and_remembers_only_when_enabled() {
        let directory = test_directory("selection");
        let settings_path = btw::settings_path(&directory);
        fs::write(
            &settings_path,
            br#"{
  "model": "side/fast",
  "thinkingLevel": "high",
  "rememberThinkingLevelChanges": false
}"#,
        )
        .expect("write settings");
        let state = main_state(vec![user("main task")]);
        let resolved_references = Arc::new(Mutex::new(Vec::new()));
        let resolver_references = Arc::clone(&resolved_references);
        let requests = Arc::new(Mutex::new(Vec::new()));
        let request_log = Arc::clone(&requests);
        let responder: agent::AssistantResponder = Arc::new(move |model, _, options| {
            lock(&request_log).push((
                model.provider.clone(),
                model.id.clone(),
                options.thinking_level,
                options.session_id,
            ));
            Ok(assistant("side answer"))
        });
        let runtime = Runtime::new(
            responder,
            move |reference| {
                lock(&resolver_references).push(reference.to_owned());
                Ok(model("side", "fast", true))
            },
            options(settings_path.clone()),
        );

        let created = runtime.create_thread(&state);
        assert_eq!(created.selection.model.provider, "side");
        assert_eq!(created.selection.model.id, "fast");
        assert_eq!(created.selection.thinking_level, llm::THINKING_HIGH);
        assert!(!created.selection.remember_thinking_level_changes);
        assert!(created.selection.warnings.is_empty());

        let changed = runtime
            .set_thread_thinking_level(&state, &created.thread, "low")
            .expect("change local side thinking");
        assert_eq!(changed.thinking_level, llm::THINKING_LOW);
        assert!(!changed.remembered);
        assert!(changed.persistence_error.is_none());
        assert_eq!(
            btw::read_settings(&settings_path).settings.thinking_level,
            llm::THINKING_HIGH,
            "remember=false must not rewrite the settings file"
        );

        runtime
            .enqueue_prompt(&created.thread, "use the configured model")
            .expect("queue question");
        let outcome = runtime
            .run_next(&state, &created.thread)
            .expect("dispatch")
            .expect("question");
        assert_eq!(outcome.selection.thinking_level, llm::THINKING_LOW);
        assert_eq!(
            lock(&requests).as_slice(),
            &[(
                "side".to_owned(),
                "fast".to_owned(),
                llm::THINKING_LOW.to_owned(),
                "test-side:btw-1".to_owned()
            )]
        );
        assert!(
            !lock(&resolved_references).is_empty(),
            "configured model resolution should be requested"
        );

        fs::write(&settings_path, "{").expect("write invalid settings");
        let fallback = runtime.resolve_selection(&state);
        assert_eq!(fallback.model, state.model);
        assert_eq!(fallback.thinking_level, state.thinking_level);
        assert_eq!(fallback.warnings.len(), 1);
        assert!(fallback.warnings[0].starts_with("pi-btw settings ignored:"));

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn dispatch_is_isolated_from_the_main_agent_transcript() {
        let main_agent = agent::Agent::new(agent::AgentOptions {
            initial_state: agent::InitialState {
                model: model("main", "current", true),
                thinking_level: llm::THINKING_HIGH.to_owned(),
                messages: vec![
                    user("main request"),
                    llm::Message::Assistant(Box::new(assistant("main reply"))),
                ],
                ..agent::InitialState::default()
            },
            ..agent::AgentOptions::default()
        });
        let before = main_agent.state();
        let seen_contexts = Arc::new(Mutex::new(Vec::new()));
        let context_log = Arc::clone(&seen_contexts);
        let responder: agent::AssistantResponder = Arc::new(move |_, context, _| {
            lock(&context_log).push(context.clone());
            Ok(assistant("independent answer"))
        });
        let directory = test_directory("isolation");
        let runtime = runtime_without_configured_model(responder, btw::settings_path(&directory));

        let created = runtime.create_thread(&before);
        runtime
            .enqueue_prompt(&created.thread, "what is the next step?")
            .expect("queue");
        let outcome = runtime
            .run_next(&main_agent.state(), &created.thread)
            .expect("dispatch")
            .expect("question");
        assert!(matches!(outcome.status, DispatchStatus::Answered { .. }));

        let contexts = lock(&seen_contexts);
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].system_prompt, btw::SYSTEM_PROMPT);
        assert!(contexts[0].tools.is_empty());
        assert_eq!(contexts[0].messages.len(), 1);
        let prompt = contexts[0].messages[0].text_preview();
        assert!(prompt.contains("User: main request"));
        assert!(prompt.contains("Assistant: main reply"));
        assert!(prompt.contains("what is the next step?"));
        drop(contexts);

        let after = main_agent.state();
        assert_eq!(after.messages, before.messages);
        assert_eq!(after.model, before.model);
        assert_eq!(after.thinking_level, before.thinking_level);
        assert!(after.error_message.is_empty());
        assert_eq!(
            runtime
                .thread(&created.thread)
                .expect("side thread")
                .turns
                .len(),
            1
        );

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn queued_follow_ups_are_fifo_and_replay_successful_turns() {
        let directory = test_directory("queue");
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let context_log = Arc::clone(&contexts);
        let calls = Arc::new(AtomicUsize::new(0));
        let call_count = Arc::clone(&calls);
        let responder: agent::AssistantResponder = Arc::new(move |_, context, _| {
            lock(&context_log).push(context.clone());
            let answer = if call_count.fetch_add(1, Ordering::Relaxed) == 0 {
                "first answer"
            } else {
                "second answer"
            };
            Ok(assistant(answer))
        });
        let runtime = runtime_without_configured_model(responder, btw::settings_path(&directory));
        let state = main_state(vec![user("main task")]);
        let created = runtime.create_thread(&state);
        runtime
            .enqueue_prompt(&created.thread, "first")
            .expect("queue first");
        runtime
            .enqueue_follow_up(&created.thread, "second")
            .expect("queue second");

        let first = runtime
            .run_next(&state, &created.thread)
            .expect("run first")
            .expect("first result");
        assert_eq!(first.question_kind, btw::QueuedQuestionKind::Prompt);
        assert_eq!(first.queued, 1);
        let second = runtime
            .run_next(&state, &created.thread)
            .expect("run second")
            .expect("second result");
        assert_eq!(second.question_kind, btw::QueuedQuestionKind::FollowUp);
        assert_eq!(second.queued, 0);
        assert!(
            runtime
                .run_next(&state, &created.thread)
                .expect("empty queue")
                .is_none()
        );

        let contexts = lock(&contexts);
        assert_eq!(contexts.len(), 2);
        assert_eq!(contexts[0].messages.len(), 1);
        assert_eq!(contexts[1].messages.len(), 3);
        assert!(
            contexts[1].messages[0]
                .text_preview()
                .contains("<conversation_context>")
        );
        assert_eq!(contexts[1].messages[1].text_preview(), "first answer");
        assert!(contexts[1].messages[2].text_preview().contains("second"));

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn responder_failures_are_recorded_and_excluded_from_follow_up_context() {
        let directory = test_directory("failure");
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let context_log = Arc::clone(&contexts);
        let calls = Arc::new(AtomicUsize::new(0));
        let call_count = Arc::clone(&calls);
        let responder: agent::AssistantResponder = Arc::new(move |_, context, _| {
            lock(&context_log).push(context.clone());
            if call_count.fetch_add(1, Ordering::Relaxed) == 0 {
                Err("provider unavailable".to_owned())
            } else {
                Ok(assistant("recovered"))
            }
        });
        let runtime = runtime_without_configured_model(responder, btw::settings_path(&directory));
        let state = main_state(vec![user("main task")]);
        let created = runtime.create_thread(&state);
        runtime
            .enqueue_prompt(&created.thread, "failed question")
            .expect("queue failing question");
        runtime
            .enqueue_follow_up(&created.thread, "recovery question")
            .expect("queue recovery question");

        let failed = runtime
            .run_next(&state, &created.thread)
            .expect("run failure")
            .expect("failure result");
        assert_eq!(
            failed.status,
            DispatchStatus::Failed {
                message: "provider unavailable".to_owned()
            }
        );
        let recovered = runtime
            .run_next(&state, &created.thread)
            .expect("run recovery")
            .expect("recovery result");
        assert_eq!(
            recovered.status,
            DispatchStatus::Answered {
                answer: "recovered".to_owned()
            }
        );

        let snapshot = runtime.thread(&created.thread).expect("thread");
        assert_eq!(snapshot.turns.len(), 2);
        assert_eq!(snapshot.turns[0].kind, btw::TurnKind::Error);
        assert_eq!(snapshot.turns[1].kind, btw::TurnKind::Answered);
        assert_eq!(runtime.list_threads()[0].questions, 2);
        let contexts = lock(&contexts);
        assert_eq!(contexts[1].messages.len(), 1);
        assert!(
            contexts[1].messages[0]
                .text_preview()
                .contains("recovery question")
        );
        assert!(
            !contexts[1].messages[0]
                .text_preview()
                .contains("failed question"),
            "error turns must never be replayed"
        );

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn cancellation_skips_the_turn_and_keeps_follow_ups_queued() {
        let directory = test_directory("cancel");
        let calls = Arc::new(AtomicUsize::new(0));
        let call_count = Arc::clone(&calls);
        let (started_sender, started_receiver) = mpsc::channel();
        let responder: agent::AssistantResponder = Arc::new(move |_, _, options| {
            if call_count.fetch_add(1, Ordering::Relaxed) == 0 {
                started_sender.send(()).expect("notify request start");
                while !options.cancellation.is_cancelled() {
                    thread::yield_now();
                }
                Err("responder observed cancellation".to_owned())
            } else {
                Ok(assistant("follow-up after cancellation"))
            }
        });
        let runtime = runtime_without_configured_model(responder, btw::settings_path(&directory));
        let state = main_state(vec![user("main task")]);
        let created = runtime.create_thread(&state);
        runtime
            .enqueue_prompt(&created.thread, "cancel me")
            .expect("queue first");
        runtime
            .enqueue_follow_up(&created.thread, "still queued")
            .expect("queue follow-up");

        let worker_runtime = runtime.clone();
        let worker_state = state.clone();
        let thread_id = created.thread.id.clone();
        let worker = thread::spawn(move || worker_runtime.run_next(&worker_state, thread_id));
        started_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("responder should start");
        assert!(runtime.cancel(&created.thread).expect("cancel request"));

        let cancelled = worker
            .join()
            .expect("worker should not panic")
            .expect("runtime result")
            .expect("queued question");
        assert_eq!(cancelled.status, DispatchStatus::Cancelled);
        assert_eq!(cancelled.queued, 1);
        assert!(
            runtime
                .thread(&created.thread)
                .expect("thread")
                .turns
                .is_empty(),
            "cancellation must not create an error turn"
        );
        assert_eq!(
            runtime
                .queue_status(&created.thread)
                .expect("queue status")
                .queued,
            1
        );

        let follow_up = runtime
            .run_next(&state, &created.thread)
            .expect("run follow-up")
            .expect("follow-up result");
        assert_eq!(
            follow_up.status,
            DispatchStatus::Answered {
                answer: "follow-up after cancellation".to_owned()
            }
        );
        assert_eq!(
            runtime.thread(&created.thread).expect("thread").turns.len(),
            1
        );

        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn export_delegates_to_deterministic_btw_primitives() {
        let directory = test_directory("export");
        let calls = Arc::new(AtomicUsize::new(0));
        let call_count = Arc::clone(&calls);
        let responder: agent::AssistantResponder = Arc::new(move |_, _, _| {
            let answer = if call_count.fetch_add(1, Ordering::Relaxed) == 0 {
                "first answer"
            } else {
                "second answer"
            };
            Ok(assistant(answer))
        });
        let runtime = runtime_without_configured_model(responder, btw::settings_path(&directory));
        let state = main_state(vec![user("main task")]);
        let empty = runtime.create_thread(&state);
        assert!(matches!(
            runtime.export(&empty.thread, btw::BringSelection::Latest),
            Err(RuntimeError::NoSuccessfulAnswer(_))
        ));

        let created = runtime.create_thread(&state);
        runtime
            .enqueue_prompt(&created.thread, "first")
            .expect("queue first");
        runtime
            .enqueue_follow_up(&created.thread, "second")
            .expect("queue second");
        runtime
            .run_next(&state, &created.thread)
            .expect("first dispatch");
        runtime
            .run_next(&state, &created.thread)
            .expect("second dispatch");

        let snapshot = runtime.thread(&created.thread).expect("thread");
        let all = runtime
            .export(&created.thread, btw::BringSelection::All)
            .expect("export all");
        let expected_segments = btw::select_bring_segments(&snapshot, btw::BringSelection::All);
        assert_eq!(all.segments, expected_segments);
        assert_eq!(all.text, btw::format_bring_to_main(&expected_segments));
        assert_eq!(
            all.estimated_tokens,
            btw::estimate_tokens(&expected_segments)
        );

        let latest = runtime
            .bring_to_main(&created.thread, btw::BringSelection::Latest)
            .expect("bring latest");
        assert_eq!(
            latest.segments,
            btw::select_bring_segments(&snapshot, btw::BringSelection::Latest)
        );

        fs::remove_dir_all(directory).expect("remove test directory");
    }
}
