//! Core coding-agent loop.
//!
//! The runtime owns the durable conversation state, queue semantics, tool
//! execution, cancellation, and lifecycle events independently of a provider
//! protocol or terminal frontend. Provider adapters can therefore stream into
//! the same state machine without coupling Ratatui to HTTP details.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Condvar, Mutex, Weak,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread::{self, ThreadId},
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Number, Value};

use crate::{llm, stream};

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentError {
    Busy,
    EmptyTranscript,
    CannotContinue(String),
    ResetWhileRunning,
    CompactWhileRunning,
    /// The transcript changed between the caller's snapshot and its compaction.
    StaleSnapshot {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy => formatter.write_str(
                "agent is already processing a prompt; use steering or follow-up messages, or wait for completion",
            ),
            Self::EmptyTranscript => formatter.write_str("no messages to continue from"),
            Self::CannotContinue(role) => {
                write!(formatter, "cannot continue from message role: {role}")
            }
            Self::ResetWhileRunning => {
                formatter.write_str("agent is already processing; wait for completion before resetting")
            }
            Self::CompactWhileRunning => {
                formatter.write_str("agent is already processing; wait for completion before compacting")
            }
            Self::StaleSnapshot { expected, actual } => write!(
                formatter,
                "the transcript changed while compaction was being prepared (expected {expected} messages, found {actual})"
            ),
        }
    }
}

impl std::error::Error for AgentError {}

/// Cooperative cancellation shared by a running provider request and its
/// active tools.
#[derive(Clone, Debug, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolExecutionMode {
    Sequential,
    #[default]
    Parallel,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueueMode {
    All,
    #[default]
    OneAtATime,
}

#[derive(Clone, Debug, Default)]
pub struct ToolResult {
    pub content: Vec<llm::ContentBlock>,
    pub details: Option<Value>,
    pub usage: Option<llm::Usage>,
    pub added_tool_names: Vec<String>,
    pub terminate: bool,
}

impl ToolResult {
    pub fn text(value: impl Into<String>) -> Self {
        Self {
            content: vec![llm::ContentBlock::text(value)],
            ..Self::default()
        }
    }
}

pub type ToolUpdate = Arc<dyn Fn(ToolResult) + Send + Sync + 'static>;
pub type ToolExecutor = Arc<
    dyn Fn(
            CancellationToken,
            String,
            BTreeMap<String, Value>,
            ToolUpdate,
        ) -> std::result::Result<ToolResult, String>
        + Send
        + Sync
        + 'static,
>;
pub type ToolArgumentPreparation =
    Arc<dyn Fn(BTreeMap<String, Value>) -> BTreeMap<String, Value> + Send + Sync + 'static>;

/// A model-facing tool definition with a schema and an executable handler.
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub label: String,
    pub description: String,
    pub parameters: Value,
    pub prepare_arguments: Option<ToolArgumentPreparation>,
    pub execute: ToolExecutor,
    pub execution_mode: Option<ToolExecutionMode>,
}

impl fmt::Debug for Tool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tool")
            .field("name", &self.name)
            .field("label", &self.label)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("execution_mode", &self.execution_mode)
            .finish_non_exhaustive()
    }
}

impl Tool {
    pub fn new(
        name: impl Into<String>,
        label: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        execute: impl Fn(
            CancellationToken,
            String,
            BTreeMap<String, Value>,
            ToolUpdate,
        ) -> std::result::Result<ToolResult, String>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            description: description.into(),
            parameters,
            prepare_arguments: None,
            execute: Arc::new(execute),
            execution_mode: None,
        }
    }

    pub fn llm_tool(&self) -> llm::Tool {
        llm::Tool {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
            constrained_sampling: None,
        }
    }
}

/// Receives normalized provider events as they arrive.
pub type AssistantEventListener =
    Arc<dyn Fn(stream::AssistantMessageEvent) + Send + Sync + 'static>;

/// Controls whether a provider may retain a prompt for follow-up turns.
///
/// `Short` is the common provider default. Providers that do not offer prompt
/// caching ignore this setting.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

/// Per-request provider inputs that are not encoded in the transcript.
#[derive(Clone)]
pub struct RequestOptions {
    pub cancellation: CancellationToken,
    pub thinking_level: llm::ThinkingLevel,
    pub thinking_budgets: Option<llm::ThinkingBudgets>,
    /// Optional runtime override for a provider's response temperature.
    pub temperature: Option<f64>,
    /// Optional runtime override for the model's default response limit.
    pub max_tokens: Option<u64>,
    /// Optional provider-native tool-choice value.
    ///
    /// Protocol adapters validate or ignore values that their wire format does
    /// not support.
    pub tool_choice: Option<Value>,
    pub cache_retention: CacheRetention,
    pub session_id: String,
    /// Streaming responders call this for each normalized provider event.
    /// Responders that only return a completed message can leave it unused.
    pub assistant_event_listener: Option<AssistantEventListener>,
}

impl fmt::Debug for RequestOptions {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RequestOptions")
            .field("cancellation", &self.cancellation)
            .field("thinking_level", &self.thinking_level)
            .field("thinking_budgets", &self.thinking_budgets)
            .field("temperature", &self.temperature)
            .field("max_tokens", &self.max_tokens)
            .field("tool_choice", &self.tool_choice)
            .field("cache_retention", &self.cache_retention)
            .field("session_id", &self.session_id)
            .field(
                "has_assistant_event_listener",
                &self.assistant_event_listener.is_some(),
            )
            .finish()
    }
}

pub type AssistantResponder = Arc<
    dyn Fn(
            &llm::Model,
            &llm::Context,
            RequestOptions,
        ) -> std::result::Result<llm::AssistantMessage, String>
        + Send
        + Sync
        + 'static,
>;

/// The turn a loop has just finished, offered to
/// [`AgentOptions::prepare_next_turn`] before the following turn starts.
#[derive(Clone, Copy, Debug)]
pub struct CompletedTurn<'a> {
    pub message: &'a llm::AssistantMessage,
    pub tool_results: &'a [llm::Message],
    /// Every message the current run has produced so far.
    pub new_messages: &'a [llm::Message],
}

/// Runs on the loop thread between turns, mirroring pi's `prepareNextTurn`.
/// Installed with [`Agent::set_prepare_next_turn`].
///
/// The loop reads the agent's state afresh for each request, so the hook
/// adjusts the next turn by calling back into the agent: it may compact the
/// transcript, switch the model, or change the thinking level. It is not
/// invoked after the final turn of a run.
pub type PrepareNextTurn = Arc<dyn Fn(&Agent, &CompletedTurn<'_>) + Send + Sync + 'static>;

#[derive(Clone)]
pub struct InitialState {
    pub system_prompt: String,
    pub model: llm::Model,
    pub thinking_level: llm::ThinkingLevel,
    pub tools: Vec<Tool>,
    pub messages: Vec<llm::Message>,
    /// Metadata for the compaction marker at the start of `messages`.
    pub compactions: Vec<CompactionInfo>,
}

impl Default for InitialState {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: llm::Model::default(),
            thinking_level: llm::THINKING_OFF.to_owned(),
            tools: Vec::new(),
            messages: Vec::new(),
            compactions: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct AgentOptions {
    pub initial_state: InitialState,
    pub responder: Option<AssistantResponder>,
    pub steering_mode: QueueMode,
    pub follow_up_mode: QueueMode,
    pub tool_execution: ToolExecutionMode,
    pub session_id: String,
}

impl Default for AgentOptions {
    fn default() -> Self {
        Self {
            initial_state: InitialState::default(),
            responder: None,
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            tool_execution: ToolExecutionMode::Parallel,
            session_id: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub system_prompt: String,
    pub model: llm::Model,
    pub thinking_level: llm::ThinkingLevel,
    pub tools: Vec<Tool>,
    pub messages: Vec<llm::Message>,
    /// Persisted context-compaction metadata for the visible marker.
    pub compactions: Vec<CompactionInfo>,
    pub is_streaming: bool,
    pub streaming_message: Option<llm::Message>,
    pub pending_tool_calls: Vec<String>,
    pub error_message: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind {
    AgentStart,
    AgentEnd,
    TurnStart,
    TurnEnd,
    MessageStart,
    MessageUpdate,
    MessageEnd,
    ToolExecutionStart,
    ToolExecutionUpdate,
    ToolExecutionEnd,
    ModelChange,
    ThinkingLevelChange,
    ContextCompacted,
    TranscriptReset,
}

/// Metadata carried with a durable transcript compaction event.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactionInfo {
    pub summary: String,
    pub tokens_before: u64,
    pub cost_before: f64,
    pub retained_messages: usize,
    pub timestamp: i64,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub kind: EventKind,
    /// The message a lifecycle event concerns.
    ///
    /// Streamed assistant notifications (a `MessageStart` raised by a stream
    /// and every `MessageUpdate`) leave this empty: their current snapshot is
    /// `assistant_event.partial`, shared with the provider rather than copied
    /// for every delta.
    pub message: Option<llm::Message>,
    /// The normalized provider event underlying a streamed assistant update.
    /// Present only for streamed message lifecycle notifications.
    pub assistant_event: Option<stream::AssistantMessageEvent>,
    /// Whether a completed assistant message was delivered incrementally.
    pub assistant_was_streamed: bool,
    pub messages: Vec<llm::Message>,
    pub kept: Vec<llm::Message>,
    pub compaction: Option<CompactionInfo>,
    pub tool_call_id: String,
    pub tool_name: String,
    pub arguments: BTreeMap<String, Value>,
    pub result: Option<ToolResult>,
    pub is_error: bool,
    pub provider: String,
    pub model_id: String,
    pub thinking_level: String,
    pub reason: String,
}

impl Event {
    fn kind(kind: EventKind) -> Self {
        Self {
            kind,
            message: None,
            assistant_event: None,
            assistant_was_streamed: false,
            messages: Vec::new(),
            kept: Vec::new(),
            compaction: None,
            tool_call_id: String::new(),
            tool_name: String::new(),
            arguments: BTreeMap::new(),
            result: None,
            is_error: false,
            provider: String::new(),
            model_id: String::new(),
            thinking_level: String::new(),
            reason: String::new(),
        }
    }
}

/// Observes lifecycle events. One event is built per emission and shared by
/// every listener, so a listener that needs to keep an event clones it.
pub type Listener = Arc<dyn Fn(&Event) + Send + Sync + 'static>;

/// Keeps a listener registered for its lifetime.
pub struct Subscription {
    agent: Weak<AgentInner>,
    id: usize,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        if let Some(agent) = self.agent.upgrade() {
            lock(&agent.listeners).remove(&self.id);
        }
    }
}

#[derive(Clone)]
pub struct Agent {
    inner: Arc<AgentInner>,
}

/// A non-owning queue handle for extensions that schedule a follow-up turn.
///
/// Tool executors are retained by an agent's tool list, so an extension must
/// not capture an [`Agent`] strongly merely to queue work: that would create
/// an `Agent → Tool → Queue → Agent` reference cycle.
#[derive(Clone)]
pub struct WeakFollowUpQueue {
    inner: Weak<AgentInner>,
}

impl WeakFollowUpQueue {
    /// Queues a message while the owning agent is still alive.
    ///
    /// Dropping an extension after its session has closed is intentionally a
    /// harmless no-op rather than a panic from a stale tool closure.
    pub fn follow_up(&self, message: llm::Message) {
        if let Some(agent) = self.inner.upgrade() {
            lock(&agent.state).follow_ups.push(message);
        }
    }

    /// Reports whether the live agent has pending steering or follow-up work.
    #[must_use]
    pub fn has_queued_messages(&self) -> bool {
        self.inner.upgrade().is_some_and(|agent| {
            let state = lock(&agent.state);
            !state.steering.is_empty() || !state.follow_ups.is_empty()
        })
    }
}

struct AgentInner {
    state: Mutex<InnerState>,
    idle: Condvar,
    listeners: Mutex<BTreeMap<usize, Listener>>,
    next_listener_id: AtomicUsize,
    /// A broken listener panics on every event; reporting it once keeps a
    /// terminal frontend readable.
    listener_panic_reported: AtomicBool,
    responder: AssistantResponder,
    prepare_next_turn: Mutex<Option<PrepareNextTurn>>,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    tool_execution: ToolExecutionMode,
    session_id: String,
}

/// A message that has started but not yet joined the transcript.
enum StreamingMessage {
    /// A complete message between its start and end events.
    Whole(llm::Message),
    /// The provider's shared streaming snapshot. It is materialized as a
    /// message only for a state snapshot, never once per delta.
    Partial(stream::SharedAssistantMessage),
}

impl StreamingMessage {
    fn to_message(&self) -> llm::Message {
        match self {
            Self::Whole(message) => message.clone(),
            Self::Partial(partial) => {
                llm::Message::Assistant(Box::new(llm::AssistantMessage::clone(partial)))
            }
        }
    }
}

struct InnerState {
    system_prompt: String,
    model: llm::Model,
    thinking_level: llm::ThinkingLevel,
    tools: Vec<Tool>,
    messages: Vec<llm::Message>,
    compactions: Vec<CompactionInfo>,
    streaming_message: Option<StreamingMessage>,
    pending_tool_calls: BTreeSet<String>,
    error_message: String,
    /// Present while the loop is streaming or executing tools; it is cleared
    /// before `AgentEnd` so that listeners observe an idle stream state.
    cancellation: Option<CancellationToken>,
    /// Held from the start of a run until its `AgentEnd` listeners settle:
    /// like pi's `activeRun`, the agent stays busy while they run.
    run_active: bool,
    /// The loop thread while `prepare_next_turn` runs. Only that thread may
    /// compact the transcript in the middle of a run.
    turn_preparation: Option<ThreadId>,
    /// A durable transcript rewrite whose event has not reached every listener
    /// yet. Runs are refused until it clears, so a recorder never sees a
    /// prompt land before the cut it must follow, and a listener that calls
    /// back into the agent gets `Busy` instead of a deadlock.
    lifecycle_transition: bool,
    steering: Vec<llm::Message>,
    follow_ups: Vec<llm::Message>,
}

impl InnerState {
    /// Compaction is safe when nothing is in flight: while idle, or between
    /// turns on the loop thread itself.
    fn accepts_compaction(&self) -> bool {
        !self.lifecycle_transition
            && (!self.run_active || self.turn_preparation == Some(thread::current().id()))
    }
}

/// How a run begins: pi's `runAgentLoop` for prompts, `runAgentLoopContinue`
/// when `messages` is empty.
struct RunStart {
    messages: Vec<llm::Message>,
    /// Set when the prompt itself came from the steering queue, so the loop
    /// must not immediately poll that queue again: in one-at-a-time mode the
    /// poll would deliver a second message in the same turn.
    skip_initial_steering_poll: bool,
}

/// Releases the run slot even if the loop unwinds twice, which would
/// otherwise leave the agent permanently busy.
struct ActiveRun<'a> {
    agent: &'a Agent,
}

impl Drop for ActiveRun<'_> {
    fn drop(&mut self) {
        let mut state = lock(&self.agent.inner.state);
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        state.cancellation = None;
        state.turn_preparation = None;
        state.run_active = false;
        self.agent.inner.idle.notify_all();
    }
}

impl Agent {
    pub fn new(options: AgentOptions) -> Self {
        let initial = options.initial_state;
        let responder = options.responder.unwrap_or_else(|| {
            Arc::new(|model, _, _| {
                Err(format!(
                    "no provider streamer is registered for api {:?}",
                    model.api
                ))
            })
        });
        Self {
            inner: Arc::new(AgentInner {
                state: Mutex::new(InnerState {
                    system_prompt: initial.system_prompt,
                    model: initial.model,
                    thinking_level: initial.thinking_level,
                    tools: initial.tools,
                    messages: initial.messages,
                    compactions: initial.compactions,
                    streaming_message: None,
                    pending_tool_calls: BTreeSet::new(),
                    error_message: String::new(),
                    cancellation: None,
                    run_active: false,
                    turn_preparation: None,
                    lifecycle_transition: false,
                    steering: Vec::new(),
                    follow_ups: Vec::new(),
                }),
                idle: Condvar::new(),
                listeners: Mutex::new(BTreeMap::new()),
                next_listener_id: AtomicUsize::new(1),
                listener_panic_reported: AtomicBool::new(false),
                responder,
                prepare_next_turn: Mutex::new(None),
                steering_mode: options.steering_mode,
                follow_up_mode: options.follow_up_mode,
                tool_execution: options.tool_execution,
                session_id: options.session_id,
            }),
        }
    }

    pub fn state(&self) -> State {
        let state = lock(&self.inner.state);
        State {
            system_prompt: state.system_prompt.clone(),
            model: state.model.clone(),
            thinking_level: state.thinking_level.clone(),
            tools: state.tools.clone(),
            messages: state.messages.clone(),
            compactions: state.compactions.clone(),
            is_streaming: state.cancellation.is_some(),
            streaming_message: state
                .streaming_message
                .as_ref()
                .map(StreamingMessage::to_message),
            pending_tool_calls: state.pending_tool_calls.iter().cloned().collect(),
            error_message: state.error_message.clone(),
        }
    }

    pub fn subscribe(&self, listener: impl Fn(&Event) + Send + Sync + 'static) -> Subscription {
        let id = self.inner.next_listener_id.fetch_add(1, Ordering::Relaxed);
        lock(&self.inner.listeners).insert(id, Arc::new(listener));
        Subscription {
            agent: Arc::downgrade(&self.inner),
            id,
        }
    }

    pub fn set_system_prompt(&self, system_prompt: impl Into<String>) {
        lock(&self.inner.state).system_prompt = system_prompt.into();
    }

    pub fn set_model(&self, model: llm::Model) {
        let changed = {
            let mut state = lock(&self.inner.state);
            let changed = state.model.provider != model.provider || state.model.id != model.id;
            state.model = model.clone();
            changed
        };
        if changed {
            let mut event = Event::kind(EventKind::ModelChange);
            event.provider = model.provider;
            event.model_id = model.id;
            self.emit(&event);
        }
    }

    pub fn set_thinking_level(&self, thinking_level: impl Into<String>) {
        let thinking_level = thinking_level.into();
        let changed = {
            let mut state = lock(&self.inner.state);
            let changed = state.thinking_level != thinking_level;
            state.thinking_level = thinking_level.clone();
            changed
        };
        if changed {
            let mut event = Event::kind(EventKind::ThinkingLevelChange);
            event.thinking_level = thinking_level;
            self.emit(&event);
        }
    }

    pub fn set_tools(&self, tools: Vec<Tool>) {
        lock(&self.inner.state).tools = tools;
    }

    /// Installs or removes the between-turn hook; see [`PrepareNextTurn`].
    ///
    /// A hook installed during a run applies from that run's next turn.
    pub fn set_prepare_next_turn(&self, hook: Option<PrepareNextTurn>) {
        *lock(&self.inner.prepare_next_turn) = hook;
    }

    /// Replaces both context messages and their retained compaction metadata.
    ///
    /// This is intentionally event-free because it is used while restoring a
    /// session before its recorder begins accepting lifecycle events.
    pub fn set_context(&self, messages: Vec<llm::Message>, compactions: Vec<CompactionInfo>) {
        let mut state = lock(&self.inner.state);
        state.messages = messages;
        state.compactions = compactions;
    }

    /// Returns the responder backing this agent.
    ///
    /// Internal services such as context compaction use this to create an
    /// isolated, unrecorded helper turn without changing the live transcript.
    pub fn responder(&self) -> AssistantResponder {
        Arc::clone(&self.inner.responder)
    }

    /// Returns a queue handle that does not keep this agent alive.
    #[must_use]
    pub fn weak_follow_up_queue(&self) -> WeakFollowUpQueue {
        WeakFollowUpQueue {
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub fn steer(&self, message: llm::Message) {
        lock(&self.inner.state).steering.push(message);
    }

    pub fn follow_up(&self, message: llm::Message) {
        lock(&self.inner.state).follow_ups.push(message);
    }

    pub fn clear_steering_queue(&self) {
        lock(&self.inner.state).steering.clear();
    }

    pub fn clear_follow_up_queue(&self) {
        lock(&self.inner.state).follow_ups.clear();
    }

    pub fn clear_all_queues(&self) {
        let mut state = lock(&self.inner.state);
        state.steering.clear();
        state.follow_ups.clear();
    }

    /// Empties both queues and returns what they held, steering first, so an
    /// interface can hand the text back to the user on abort (pi's
    /// `clearQueue`) instead of letting it run after the interrupted turn.
    pub fn take_queued_messages(&self) -> Vec<llm::Message> {
        let mut state = lock(&self.inner.state);
        let mut messages = std::mem::take(&mut state.steering);
        messages.append(&mut state.follow_ups);
        messages
    }

    pub fn has_queued_messages(&self) -> bool {
        let state = lock(&self.inner.state);
        !state.steering.is_empty() || !state.follow_ups.is_empty()
    }

    pub fn queued_message_count(&self) -> usize {
        let state = lock(&self.inner.state);
        state.steering.len() + state.follow_ups.len()
    }

    pub fn abort(&self) {
        if let Some(cancellation) = lock(&self.inner.state).cancellation.clone() {
            cancellation.cancel();
        }
    }

    /// Blocks until the current run, including its `AgentEnd` listeners, has
    /// finished. Returns immediately while idle.
    pub fn wait_for_idle(&self) {
        let mut state = lock(&self.inner.state);
        while state.run_active {
            state = self
                .inner
                .idle
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    /// Reports whether [`Agent::compact`] would be accepted right now.
    ///
    /// Compaction is allowed while idle and, for a [`PrepareNextTurn`] hook,
    /// on the loop thread between turns. Checking first spares the caller an
    /// expensive summary request that the agent would then refuse.
    #[must_use]
    pub fn can_compact(&self) -> bool {
        lock(&self.inner.state).accepts_compaction()
    }

    pub fn reset(&self) -> Result<()> {
        self.reset_with_reason("")
    }

    pub fn reset_with_reason(&self, reason: impl Into<String>) -> Result<()> {
        {
            let mut state = lock(&self.inner.state);
            if state.run_active {
                return Err(AgentError::ResetWhileRunning);
            }
            if state.lifecycle_transition {
                return Err(AgentError::Busy);
            }
            state.messages.clear();
            state.compactions.clear();
            state.steering.clear();
            state.follow_ups.clear();
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message.clear();
            state.lifecycle_transition = true;
        }
        let mut event = Event::kind(EventKind::TranscriptReset);
        event.reason = reason.into();
        self.emit(&event);
        lock(&self.inner.state).lifecycle_transition = false;
        Ok(())
    }

    /// Replaces the transcript with a summary marker and retained messages.
    ///
    /// The change is emitted as a single lifecycle event so session recorders
    /// can persist the exact cut rather than replaying the discarded prefix on
    /// the next resume. Like reset, compaction is refused during a live turn,
    /// except from a [`PrepareNextTurn`] hook between turns.
    ///
    /// Callers that computed `kept` from a state snapshot should prefer
    /// [`Agent::compact_if_unchanged`], which refuses to apply a cut that no
    /// longer matches the transcript.
    pub fn compact(
        &self,
        marker: llm::Message,
        kept: Vec<llm::Message>,
        info: CompactionInfo,
    ) -> Result<()> {
        self.apply_compaction(None, marker, kept, info)
    }

    /// [`Agent::compact`] guarded by the transcript length the cut was
    /// computed from. A run that completed in between changes the length, and
    /// applying the stale cut would silently drop its messages.
    pub fn compact_if_unchanged(
        &self,
        expected_messages: usize,
        marker: llm::Message,
        kept: Vec<llm::Message>,
        info: CompactionInfo,
    ) -> Result<()> {
        self.apply_compaction(Some(expected_messages), marker, kept, info)
    }

    fn apply_compaction(
        &self,
        expected_messages: Option<usize>,
        marker: llm::Message,
        kept: Vec<llm::Message>,
        info: CompactionInfo,
    ) -> Result<()> {
        {
            let mut state = lock(&self.inner.state);
            if state.lifecycle_transition {
                return Err(AgentError::Busy);
            }
            if !state.accepts_compaction() {
                return Err(AgentError::CompactWhileRunning);
            }
            if let Some(expected) = expected_messages
                && expected != state.messages.len()
            {
                return Err(AgentError::StaleSnapshot {
                    expected,
                    actual: state.messages.len(),
                });
            }
            let mut compacted = Vec::with_capacity(kept.len() + 1);
            compacted.push(marker.clone());
            compacted.extend(kept.iter().cloned());
            state.messages = compacted;
            state.compactions = vec![info.clone()];
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message.clear();
            state.lifecycle_transition = true;
        }
        let mut event = Event::kind(EventKind::ContextCompacted);
        event.message = Some(marker);
        event.kept = kept;
        event.compaction = Some(info);
        self.emit(&event);
        lock(&self.inner.state).lifecycle_transition = false;
        Ok(())
    }

    pub fn prompt(&self, prompt: impl Into<String>) -> Result<()> {
        self.prompt_messages(vec![llm::Message::User(llm::UserMessage::text(
            prompt,
            now_millis(),
        ))])
    }

    /// Runs a prompt, then any messages queued while it ran but left behind.
    ///
    /// This blocks until the whole sequence, including `AgentEnd` listeners,
    /// has settled, matching `await agent.prompt(...)` followed by pi's
    /// post-run `continue()` loop.
    pub fn prompt_messages(&self, messages: Vec<llm::Message>) -> Result<()> {
        self.run_with(|_| {
            Ok(Some(RunStart {
                messages,
                skip_initial_steering_poll: false,
            }))
        })?;
        self.run_queued_messages()
    }

    /// Continues from the current transcript, as pi's `Agent.continue()`.
    ///
    /// After an assistant message the queued steering messages, then the
    /// follow-ups, become the prompt; any other tail resumes as it stands.
    pub fn continue_run(&self) -> Result<()> {
        let steering_mode = self.inner.steering_mode;
        let follow_up_mode = self.inner.follow_up_mode;
        self.run_with(|state| continuation(state, steering_mode, follow_up_mode).map(Some))?;
        self.run_queued_messages()
    }

    /// pi's session loops on `agent.continue()` once `prompt()` resolves: the
    /// loop leaves queued messages behind when it stops on an error or abort,
    /// and `AgentEnd` listeners may queue more, so they would otherwise wait
    /// for the next prompt.
    fn run_queued_messages(&self) -> Result<()> {
        let steering_mode = self.inner.steering_mode;
        let follow_up_mode = self.inner.follow_up_mode;
        loop {
            let started = self.run_with(|state| {
                if state.steering.is_empty() && state.follow_ups.is_empty() {
                    return Ok(None);
                }
                continuation(state, steering_mode, follow_up_mode).map(Some)
            })?;
            if !started {
                return Ok(());
            }
        }
    }

    /// Claims the run slot and starts a run in one step.
    ///
    /// `prepare` decides what to run while the state is still locked, so a
    /// continuation that drains a queue cannot lose its messages to a busy
    /// agent. Returns `Ok(false)` when `prepare` found nothing to run.
    fn run_with(
        &self,
        prepare: impl FnOnce(&mut InnerState) -> Result<Option<RunStart>>,
    ) -> Result<bool> {
        let (start, cancellation) = {
            let mut state = lock(&self.inner.state);
            if state.run_active || state.lifecycle_transition {
                return Err(AgentError::Busy);
            }
            let Some(start) = prepare(&mut state)? else {
                return Ok(false);
            };
            let cancellation = CancellationToken::default();
            state.run_active = true;
            state.cancellation = Some(cancellation.clone());
            state.streaming_message = None;
            state.error_message.clear();
            (start, cancellation)
        };
        let active = ActiveRun { agent: self };

        let messages = match catch_unwind(AssertUnwindSafe(|| {
            self.run_loop(start, cancellation.clone())
        })) {
            Ok(messages) => messages,
            Err(_) => {
                // pi's handleRunFailure: the failure is reported as a complete
                // final turn so turn-oriented listeners settle their state.
                let mut error = self.error_message("the agent runtime panicked");
                if cancellation.is_cancelled() {
                    error.stop_reason = stream::STOP_ABORTED.to_owned();
                }
                let message = llm::Message::Assistant(Box::new(error));
                self.record_message(message.clone());
                let mut turn_end = Event::kind(EventKind::TurnEnd);
                turn_end.message = Some(message.clone());
                self.emit(&turn_end);
                vec![message]
            }
        };

        // Listeners observing the end of a run see an idle stream state, while
        // the run slot stays claimed until they settle (pi's finishRun).
        {
            let mut state = lock(&self.inner.state);
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.cancellation = None;
        }
        let mut event = Event::kind(EventKind::AgentEnd);
        event.messages = messages;
        self.emit(&event);
        drop(active);
        Ok(true)
    }

    fn run_loop(&self, start: RunStart, cancellation: CancellationToken) -> Vec<llm::Message> {
        let mut new_messages = Vec::new();
        self.emit(&Event::kind(EventKind::AgentStart));
        self.emit(&Event::kind(EventKind::TurnStart));
        for message in start.messages {
            self.record_message(message.clone());
            new_messages.push(message);
        }
        // A message steered while the prompt was being submitted rides along
        // with it rather than waiting for the first response.
        let mut pending = if start.skip_initial_steering_poll {
            Vec::new()
        } else {
            self.drain_steering()
        };
        let mut last_turn: Option<(llm::Message, Vec<llm::Message>)> = None;

        // Outer loop: resumes when follow-ups arrive after the agent would stop.
        loop {
            let mut has_more_tool_calls = true;

            // Inner loop: tool calls and steering messages.
            while has_more_tool_calls || !pending.is_empty() {
                if let Some((message, tool_results)) = last_turn.take() {
                    if let llm::Message::Assistant(message) = &message {
                        self.prepare_next_turn(&CompletedTurn {
                            message,
                            tool_results: &tool_results,
                            new_messages: &new_messages,
                        });
                    }
                    // Preparation can be long-running (compaction, say), so
                    // pick up steering queued meanwhile. Only poll again when
                    // the earlier poll came back empty: one-at-a-time mode
                    // would otherwise deliver two messages in this turn.
                    if pending.is_empty() {
                        pending = self.drain_steering();
                    }
                    self.emit(&Event::kind(EventKind::TurnStart));
                }
                for message in pending.drain(..) {
                    self.record_message(message.clone());
                    new_messages.push(message);
                }

                let assistant = if cancellation.is_cancelled() {
                    let mut aborted = self.error_message("Request was aborted");
                    aborted.stop_reason = stream::STOP_ABORTED.to_owned();
                    aborted
                } else {
                    self.request_assistant(cancellation.clone())
                };
                let is_terminal = matches!(
                    assistant.stop_reason.as_str(),
                    stream::STOP_ERROR | stream::STOP_ABORTED
                );
                self.record_message(llm::Message::Assistant(Box::new(assistant.clone())));
                new_messages.push(llm::Message::Assistant(Box::new(assistant.clone())));

                let mut tool_results = Vec::new();
                has_more_tool_calls = false;
                if !is_terminal {
                    let calls = tool_calls(&assistant);
                    if !calls.is_empty() {
                        let outcomes = if assistant.stop_reason == stream::STOP_LENGTH {
                            self.fail_truncated_tool_calls(calls)
                        } else {
                            self.execute_tool_calls(calls, cancellation.clone())
                        };
                        // A batch ends the run only when every result asked
                        // for it; an error result is the model's cue to retry.
                        has_more_tool_calls =
                            !outcomes.iter().all(|outcome| outcome.result.terminate);
                        for outcome in outcomes {
                            let message = outcome.as_message();
                            self.record_message(message.clone());
                            new_messages.push(message.clone());
                            tool_results.push(message);
                        }
                    }
                }

                let mut turn_end = Event::kind(EventKind::TurnEnd);
                turn_end.message = Some(llm::Message::Assistant(Box::new(assistant)));
                turn_end.messages = tool_results;
                self.emit(&turn_end);
                // Stopping here rather than at the next request spares an
                // already-cancelled run a wasted round trip and a spurious
                // empty assistant message.
                if is_terminal || cancellation.is_cancelled() {
                    return new_messages;
                }
                last_turn = turn_end
                    .message
                    .take()
                    .map(|message| (message, std::mem::take(&mut turn_end.messages)));
                pending = self.drain_steering();
            }

            // The agent would stop here; follow-ups reopen the inner loop.
            let follow_ups = self.drain_follow_ups();
            if follow_ups.is_empty() {
                break;
            }
            pending = follow_ups;
        }
        new_messages
    }

    fn prepare_next_turn(&self, turn: &CompletedTurn<'_>) {
        let Some(hook) = lock(&self.inner.prepare_next_turn).clone() else {
            return;
        };
        lock(&self.inner.state).turn_preparation = Some(thread::current().id());
        let outcome = catch_unwind(AssertUnwindSafe(|| hook(self, turn)));
        lock(&self.inner.state).turn_preparation = None;
        if let Err(panic) = outcome {
            std::panic::resume_unwind(panic);
        }
    }

    fn request_assistant(&self, cancellation: CancellationToken) -> llm::AssistantMessage {
        let (model, context, thinking_level) = {
            let state = lock(&self.inner.state);
            (
                state.model.clone(),
                llm::Context {
                    system_prompt: state.system_prompt.clone(),
                    messages: state.messages.clone(),
                    tools: state.tools.iter().map(Tool::llm_tool).collect(),
                },
                state.thinking_level.clone(),
            )
        };
        let event_agent = self.clone();
        let assistant_event_listener: AssistantEventListener = Arc::new(move |event| {
            event_agent.forward_assistant_event(event);
        });
        let response = catch_unwind(AssertUnwindSafe(|| {
            (self.inner.responder)(
                &model,
                &context,
                RequestOptions {
                    cancellation: cancellation.clone(),
                    thinking_level,
                    thinking_budgets: None,
                    temperature: None,
                    max_tokens: None,
                    tool_choice: None,
                    cache_retention: CacheRetention::Short,
                    session_id: self.inner.session_id.clone(),
                    assistant_event_listener: Some(assistant_event_listener),
                },
            )
        }));
        match response {
            Ok(Ok(message)) => message,
            Ok(Err(error)) => self.error_message(&error),
            Err(_) => self.error_message("provider streamer panicked"),
        }
    }

    /// pi's `failToolCallsFromTruncatedMessage`: streamed arguments are
    /// finalized by a lenient JSON salvage parser, so a `length` stop can
    /// yield calls that parse and validate yet are silently incomplete. None
    /// of them is safe to run; each gets an error result so the model
    /// re-issues them.
    fn fail_truncated_tool_calls(&self, calls: Vec<llm::ToolCall>) -> Vec<ToolOutcome> {
        calls
            .into_iter()
            .map(|call| {
                self.tool_started(&call);
                let message = format!(
                    "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
                    call.name
                );
                let outcome = ToolOutcome::error(call, message);
                self.tool_ended(&outcome);
                outcome
            })
            .collect()
    }

    fn execute_tool_calls(
        &self,
        calls: Vec<llm::ToolCall>,
        cancellation: CancellationToken,
    ) -> Vec<ToolOutcome> {
        if calls.is_empty() {
            return Vec::new();
        }
        let tools = lock(&self.inner.state).tools.clone();
        let mut prepared = Vec::new();
        let mut outcomes = Vec::new();
        let sequential = self.inner.tool_execution == ToolExecutionMode::Sequential
            || calls.iter().any(|call| {
                tools
                    .iter()
                    .find(|tool| tool.name == call.name)
                    .is_some_and(|tool| tool.execution_mode == Some(ToolExecutionMode::Sequential))
            });
        for (index, call) in calls.into_iter().enumerate() {
            self.tool_started(&call);
            match prepare_tool_call(&tools, call) {
                Ok(prepared_call) => prepared.push((index, prepared_call)),
                Err(outcome) => {
                    self.tool_ended(&outcome);
                    outcomes.push((index, *outcome));
                }
            }
        }

        if sequential {
            for (index, prepared_call) in prepared {
                let outcome = self.execute_prepared_tool(prepared_call, cancellation.clone());
                self.tool_ended(&outcome);
                outcomes.push((index, outcome));
            }
        } else {
            let mut handles = Vec::new();
            for (index, prepared_call) in prepared {
                let agent = self.clone();
                let cancellation = cancellation.clone();
                handles.push((
                    index,
                    thread::spawn(move || {
                        let outcome = agent.execute_prepared_tool(prepared_call, cancellation);
                        agent.tool_ended(&outcome);
                        outcome
                    }),
                ));
            }
            for (index, handle) in handles {
                let outcome = handle.join().unwrap_or_else(|_| {
                    ToolOutcome::error(
                        llm::ToolCall {
                            id: "unknown".to_owned(),
                            name: "unknown".to_owned(),
                            arguments: BTreeMap::new(),
                            thought_signature: String::new(),
                            namespace: String::new(),
                        },
                        "tool worker panicked",
                    )
                });
                outcomes.push((index, outcome));
            }
        }
        outcomes.sort_by_key(|(index, _)| *index);
        outcomes.into_iter().map(|(_, outcome)| outcome).collect()
    }

    fn execute_prepared_tool(
        &self,
        prepared: PreparedToolCall,
        cancellation: CancellationToken,
    ) -> ToolOutcome {
        if cancellation.is_cancelled() {
            return ToolOutcome::error(prepared.call, "Operation aborted");
        }
        let accepting_updates = Arc::new(AtomicBool::new(true));
        let update_agent = self.clone();
        let update_call = prepared.call.clone();
        let accepting = accepting_updates.clone();
        let update = Arc::new(move |result: ToolResult| {
            if accepting.load(Ordering::Acquire) {
                let mut event = Event::kind(EventKind::ToolExecutionUpdate);
                event.tool_call_id = update_call.id.clone();
                event.tool_name = update_call.name.clone();
                event.arguments = update_call.arguments.clone();
                event.result = Some(result);
                update_agent.emit(&event);
            }
        });
        let execution = catch_unwind(AssertUnwindSafe(|| {
            (prepared.tool.execute)(
                cancellation,
                prepared.call.id.clone(),
                prepared.arguments,
                update,
            )
        }));
        accepting_updates.store(false, Ordering::Release);
        match execution {
            Ok(Ok(result)) => ToolOutcome {
                call: prepared.call,
                result,
                is_error: false,
            },
            Ok(Err(error)) => ToolOutcome::error(prepared.call, error),
            Err(_) => ToolOutcome::error(prepared.call, "tool panicked"),
        }
    }

    fn tool_started(&self, call: &llm::ToolCall) {
        lock(&self.inner.state)
            .pending_tool_calls
            .insert(call.id.clone());
        let mut event = Event::kind(EventKind::ToolExecutionStart);
        event.tool_call_id = call.id.clone();
        event.tool_name = call.name.clone();
        event.arguments = call.arguments.clone();
        self.emit(&event);
    }

    fn tool_ended(&self, outcome: &ToolOutcome) {
        lock(&self.inner.state)
            .pending_tool_calls
            .remove(&outcome.call.id);
        let mut event = Event::kind(EventKind::ToolExecutionEnd);
        event.tool_call_id = outcome.call.id.clone();
        event.tool_name = outcome.call.name.clone();
        event.arguments = outcome.call.arguments.clone();
        event.result = Some(outcome.result.clone());
        event.is_error = outcome.is_error;
        self.emit(&event);
    }

    fn record_message(&self, message: llm::Message) {
        let assistant_was_streamed = {
            let mut state = lock(&self.inner.state);
            let assistant_was_streamed = matches!(&message, llm::Message::Assistant(_))
                && matches!(state.streaming_message, Some(StreamingMessage::Partial(_)));
            state.streaming_message = Some(StreamingMessage::Whole(message.clone()));
            assistant_was_streamed
        };
        let mut event = Event::kind(EventKind::MessageStart);
        event.message = Some(message);
        if !assistant_was_streamed {
            self.emit(&event);
        }
        {
            let mut state = lock(&self.inner.state);
            let message = event
                .message
                .as_ref()
                .expect("the recorded message was just attached");
            state.messages.push(message.clone());
            state.streaming_message = None;
            if let llm::Message::Assistant(assistant) = message
                && !assistant.error_message.is_empty()
            {
                state.error_message = assistant.error_message.clone();
            }
        }
        event.kind = EventKind::MessageEnd;
        event.assistant_was_streamed = assistant_was_streamed;
        self.emit(&event);
    }

    fn forward_assistant_event(&self, assistant_event: stream::AssistantMessageEvent) {
        if let Some(partial) = assistant_event
            .partial
            .clone()
            .or_else(|| assistant_event.terminal_message())
        {
            lock(&self.inner.state).streaming_message = Some(StreamingMessage::Partial(partial));
        }
        let mut event = Event::kind(if assistant_event.event_type == stream::EVENT_START {
            EventKind::MessageStart
        } else {
            EventKind::MessageUpdate
        });
        // The snapshot stays behind the event's shared `partial`: building a
        // `Message` here would deep-copy the whole reply on every delta.
        event.assistant_event = Some(assistant_event);
        self.emit(&event);
    }

    fn drain_steering(&self) -> Vec<llm::Message> {
        drain(
            &mut lock(&self.inner.state).steering,
            self.inner.steering_mode,
        )
        .unwrap_or_default()
    }

    fn drain_follow_ups(&self) -> Vec<llm::Message> {
        drain(
            &mut lock(&self.inner.state).follow_ups,
            self.inner.follow_up_mode,
        )
        .unwrap_or_default()
    }

    fn error_message(&self, message: impl AsRef<str>) -> llm::AssistantMessage {
        let model = lock(&self.inner.state).model.clone();
        llm::AssistantMessage::error(
            model.api,
            model.provider,
            model.id,
            message.as_ref(),
            now_millis(),
        )
    }

    fn emit(&self, event: &Event) {
        let listeners = lock(&self.inner.listeners)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for listener in listeners {
            if catch_unwind(AssertUnwindSafe(|| listener(event))).is_err()
                && !self
                    .inner
                    .listener_panic_reported
                    .swap(true, Ordering::Relaxed)
            {
                eprintln!(
                    "goshcoder: an agent event listener panicked while handling {:?}; later listener panics are dropped silently",
                    event.kind
                );
            }
        }
    }
}

/// pi's `Agent.continue()`: after an assistant message the queued steering
/// messages, then the follow-ups, become the prompt; any other tail resumes
/// as it stands.
fn continuation(
    state: &mut InnerState,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
) -> Result<RunStart> {
    let Some(last) = state.messages.last() else {
        return Err(AgentError::EmptyTranscript);
    };
    if last.role() != "assistant" {
        return Ok(RunStart {
            messages: Vec::new(),
            skip_initial_steering_poll: false,
        });
    }
    if let Some(messages) = drain(&mut state.steering, steering_mode) {
        return Ok(RunStart {
            messages,
            skip_initial_steering_poll: true,
        });
    }
    if let Some(messages) = drain(&mut state.follow_ups, follow_up_mode) {
        return Ok(RunStart {
            messages,
            skip_initial_steering_poll: false,
        });
    }
    Err(AgentError::CannotContinue("assistant".to_owned()))
}

fn tool_calls(assistant: &llm::AssistantMessage) -> Vec<llm::ToolCall> {
    assistant
        .content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

struct PreparedToolCall {
    call: llm::ToolCall,
    tool: Tool,
    arguments: BTreeMap<String, Value>,
}

#[derive(Clone)]
struct ToolOutcome {
    call: llm::ToolCall,
    result: ToolResult,
    is_error: bool,
}

impl ToolOutcome {
    fn error(call: llm::ToolCall, message: impl Into<String>) -> Self {
        Self {
            call,
            result: ToolResult::text(message),
            is_error: true,
        }
    }

    fn as_message(&self) -> llm::Message {
        llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
            role: "toolResult".to_owned(),
            tool_call_id: self.call.id.clone(),
            tool_name: self.call.name.clone(),
            content: self.result.content.clone(),
            details: self.result.details.clone(),
            usage: self.result.usage.clone(),
            added_tool_names: self.result.added_tool_names.clone(),
            is_error: self.is_error,
            timestamp: now_millis(),
        }))
    }
}

fn prepare_tool_call(
    tools: &[Tool],
    call: llm::ToolCall,
) -> std::result::Result<PreparedToolCall, Box<ToolOutcome>> {
    let Some(tool) = tools.iter().find(|tool| tool.name == call.name).cloned() else {
        let message = format!("Tool {} not found", call.name);
        return Err(Box::new(ToolOutcome::error(call, message)));
    };
    let arguments = tool.prepare_arguments.as_ref().map_or_else(
        || call.arguments.clone(),
        |prepare| prepare(call.arguments.clone()),
    );
    match validate_arguments(&tool.parameters, arguments) {
        Ok(arguments) => Ok(PreparedToolCall {
            call,
            tool,
            arguments,
        }),
        Err(error) => Err(Box::new(ToolOutcome::error(
            call,
            format!("validation failed: {error}"),
        ))),
    }
}

/// Supports the object/property subset used by GoshCoder's built-in tools,
/// including numeric coercion from model-generated JSON strings.
fn validate_arguments(
    schema: &Value,
    mut arguments: BTreeMap<String, Value>,
) -> std::result::Result<BTreeMap<String, Value>, String> {
    let Value::Object(schema) = schema else {
        return Ok(arguments);
    };
    if schema
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind != "object")
    {
        return Err("tool schema must describe an object".to_owned());
    }
    for required in schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if !arguments.contains_key(required) {
            return Err(format!("{required:?} is required"));
        }
    }
    let Some(Value::Object(properties)) = schema.get("properties") else {
        return Ok(arguments);
    };
    for (name, property) in properties {
        let Some(value) = arguments.get(name).cloned() else {
            continue;
        };
        let kind = property.get("type").and_then(Value::as_str);
        let coerced = match kind {
            Some("number") => coerce_number(value, false)?,
            Some("integer") => coerce_number(value, true)?,
            Some("string") if !value.is_string() => {
                return Err(format!("{name:?} must be a string"));
            }
            Some("boolean") if !value.is_boolean() => {
                return Err(format!("{name:?} must be a boolean"));
            }
            Some("array") if !value.is_array() => {
                return Err(format!("{name:?} must be an array"));
            }
            Some("object") if !value.is_object() => {
                return Err(format!("{name:?} must be an object"));
            }
            _ => value,
        };
        arguments.insert(name.clone(), coerced);
    }
    Ok(arguments)
}

fn coerce_number(value: Value, integer: bool) -> std::result::Result<Value, String> {
    if let Some(number) = value.as_f64() {
        if integer && number.fract() != 0.0 {
            return Err("value must be an integer".to_owned());
        }
        return Ok(value);
    }
    let Some(text) = value.as_str() else {
        return Err(if integer {
            "value must be an integer".to_owned()
        } else {
            "value must be a number".to_owned()
        });
    };
    if integer {
        let number = text
            .parse::<i64>()
            .map_err(|_| "value must be an integer".to_owned())?;
        return Ok(Value::Number(Number::from(number)));
    }
    let number = text
        .parse::<f64>()
        .ok()
        .and_then(Number::from_f64)
        .ok_or_else(|| "value must be a number".to_owned())?;
    Ok(Value::Number(number))
}

fn drain(messages: &mut Vec<llm::Message>, mode: QueueMode) -> Option<Vec<llm::Message>> {
    if messages.is_empty() {
        return None;
    }
    Some(match mode {
        QueueMode::All => std::mem::take(messages),
        QueueMode::OneAtATime => vec![messages.remove(0)],
    })
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };

    fn model() -> llm::Model {
        llm::Model {
            id: "test-model".to_owned(),
            name: "Test model".to_owned(),
            api: "test".to_owned(),
            provider: "test".to_owned(),
            ..llm::Model::default()
        }
    }

    fn assistant_text(text: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            role: "assistant".to_owned(),
            content: vec![llm::ContentBlock::text(text)],
            api: "test".to_owned(),
            provider: "test".to_owned(),
            model: "test-model".to_owned(),
            stop_reason: "stop".to_owned(),
            timestamp: now_millis(),
            ..llm::AssistantMessage::default()
        }
    }

    fn tool_call(id: &str, name: &str, arguments: Value) -> llm::ContentBlock {
        let arguments = match arguments {
            Value::Object(map) => map.into_iter().collect(),
            _ => BTreeMap::new(),
        };
        llm::ContentBlock::ToolCall(llm::ToolCall {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments,
            thought_signature: String::new(),
            namespace: String::new(),
        })
    }

    fn assistant_calls(calls: Vec<llm::ContentBlock>, stop_reason: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            content: calls,
            stop_reason: stop_reason.to_owned(),
            ..assistant_text("")
        }
    }

    fn user(text: &str) -> llm::Message {
        llm::Message::User(llm::UserMessage::text(text, now_millis()))
    }

    fn user_texts(context: &llm::Context) -> Vec<String> {
        context
            .messages
            .iter()
            .filter(|message| message.role() == "user")
            .map(llm::Message::text_preview)
            .collect()
    }

    fn echo_tool(executions: Arc<AtomicUsize>) -> Tool {
        Tool::new(
            "echo",
            "Echo",
            "Returns its value",
            json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
            move |_, id, arguments, _| {
                executions.fetch_add(1, Ordering::Relaxed);
                Ok(ToolResult::text(format!(
                    "{id}:{}",
                    arguments["value"].as_str().unwrap_or_default()
                )))
            },
        )
    }

    /// Runs `work` on a helper thread and fails, rather than hangs, when it
    /// does not finish in time.
    fn bounded<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            let _ = sender.send(work());
        });
        receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("the run must settle instead of hanging")
    }

    fn tool_results(messages: &[llm::Message]) -> Vec<&llm::ToolResultMessage> {
        messages
            .iter()
            .filter_map(|message| match message {
                llm::Message::ToolResult(result) => Some(result.as_ref()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn prompt_records_messages_and_lifecycle_events() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_log = events.clone();
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(|_, _, _| Ok(assistant_text("hello")))),
            ..AgentOptions::default()
        });
        let _subscription = agent.subscribe(move |event| lock(&event_log).push(event.kind));

        agent.prompt("hi").expect("prompt");
        let state = agent.state();
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.messages[0].role(), "user");
        assert_eq!(state.messages[1].role(), "assistant");
        assert!(!state.is_streaming);
        assert_eq!(
            lock(&events).as_slice(),
            &[
                EventKind::AgentStart,
                EventKind::TurnStart,
                EventKind::MessageStart,
                EventKind::MessageEnd,
                EventKind::MessageStart,
                EventKind::MessageEnd,
                EventKind::TurnEnd,
                EventKind::AgentEnd,
            ]
        );
    }

    #[test]
    fn streamed_responder_forwards_incremental_events_without_a_duplicate_start() {
        let events = Arc::new(Mutex::new(Vec::<Event>::new()));
        let event_log = Arc::clone(&events);
        let response = assistant_text("streamed answer");
        let partial = llm::AssistantMessage {
            stop_reason: stream::STOP_PENDING.to_owned(),
            ..response.clone()
        };
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, options| {
                let listener = options
                    .assistant_event_listener
                    .expect("agent supplies a stream listener");
                listener(stream::AssistantMessageEvent::start(Arc::new(
                    partial.clone(),
                )));
                listener(stream::AssistantMessageEvent {
                    event_type: stream::EVENT_TEXT_DELTA.to_owned(),
                    delta: "streamed answer".to_owned(),
                    partial: Some(Arc::new(response.clone())),
                    ..stream::AssistantMessageEvent::default()
                });
                listener(stream::AssistantMessageEvent::done(
                    stream::STOP_STOP,
                    Arc::new(response.clone()),
                ));
                Ok(response.clone())
            })),
            ..AgentOptions::default()
        });
        let _subscription = agent.subscribe(move |event| lock(&event_log).push(event.clone()));

        agent.prompt("hello").expect("prompt");

        let events = lock(&events);
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == EventKind::MessageStart)
                .count(),
            2,
            "one user start and one streamed assistant start"
        );
        let updates = events
            .iter()
            .filter(|event| event.kind == EventKind::MessageUpdate)
            .collect::<Vec<_>>();
        assert_eq!(updates.len(), 2);
        assert_eq!(
            updates[0]
                .assistant_event
                .as_ref()
                .map(|event| event.event_type.as_str()),
            Some(stream::EVENT_TEXT_DELTA)
        );
        assert_eq!(
            updates[0]
                .assistant_event
                .as_ref()
                .map(|event| event.delta.as_str()),
            Some("streamed answer")
        );
        // The streamed snapshot is shared with the provider, not copied into
        // the event for every delta.
        assert!(updates[0].message.is_none());
        assert_eq!(
            updates[0]
                .assistant_event
                .as_ref()
                .and_then(|event| event.partial.as_ref())
                .map(|partial| partial.stop_reason.as_str()),
            Some(stream::STOP_STOP)
        );
        assert!(events.iter().any(|event| {
            event.kind == EventKind::MessageEnd
                && event.assistant_was_streamed
                && matches!(event.message.as_ref(), Some(llm::Message::Assistant(_)))
        }));
        assert!(agent.state().streaming_message.is_none());
    }

    #[test]
    fn tool_calls_continue_the_turn_and_preserve_source_order() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let tool = echo_tool(Arc::new(AtomicUsize::new(0)));
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                tools: vec![tool],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(assistant_calls(
                        vec![
                            tool_call("one", "echo", json!({"value": "a"})),
                            tool_call("two", "echo", json!({"value": "b"})),
                        ],
                        stream::STOP_TOOL_USE,
                    ))
                } else {
                    Ok(assistant_text("done"))
                }
            })),
            ..AgentOptions::default()
        });

        agent.prompt("run tools").expect("prompt");
        assert_eq!(requests.load(Ordering::Relaxed), 2);
        let state = agent.state();
        let result_ids = tool_results(&state.messages)
            .iter()
            .map(|result| result.tool_call_id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(result_ids, ["one", "two"]);
    }

    #[test]
    fn truncated_tool_calls_are_failed_without_execution() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let executions = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(Mutex::new(Vec::<Event>::new()));
        let event_log = Arc::clone(&events);
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                tools: vec![echo_tool(Arc::clone(&executions))],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(assistant_calls(
                        vec![tool_call("cut", "echo", json!({"value": "a"}))],
                        stream::STOP_LENGTH,
                    ))
                } else {
                    Ok(assistant_text("re-issued"))
                }
            })),
            ..AgentOptions::default()
        });
        let _subscription = agent.subscribe(move |event| lock(&event_log).push(event.clone()));

        agent.prompt("go").expect("prompt");

        assert_eq!(
            executions.load(Ordering::Relaxed),
            0,
            "truncated call must not run"
        );
        assert_eq!(
            requests.load(Ordering::Relaxed),
            2,
            "the model gets to re-issue the call"
        );
        let state = agent.state();
        let results = tool_results(&state.messages);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error);
        let text = results[0].content[0].plain_text().unwrap_or_default();
        assert!(
            text.starts_with("Tool call \"echo\" was not executed"),
            "{text}"
        );
        assert!(text.contains("output token limit"), "{text}");
        assert_eq!(
            state
                .messages
                .last()
                .map(llm::Message::text_preview)
                .as_deref(),
            Some("re-issued")
        );
        assert!(state.pending_tool_calls.is_empty());
        let events = lock(&events);
        assert!(events.iter().any(|event| {
            event.kind == EventKind::ToolExecutionStart && event.tool_call_id == "cut"
        }));
        assert!(events.iter().any(|event| {
            event.kind == EventKind::ToolExecutionEnd
                && event.tool_call_id == "cut"
                && event.is_error
        }));
    }

    #[test]
    fn error_tool_results_do_not_end_the_run() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(assistant_calls(
                        vec![tool_call("bad", "missing", json!({}))],
                        stream::STOP_TOOL_USE,
                    ))
                } else {
                    Ok(assistant_text("recovered"))
                }
            })),
            ..AgentOptions::default()
        });

        agent.prompt("go").expect("prompt");

        assert_eq!(requests.load(Ordering::Relaxed), 2);
        let state = agent.state();
        let results = tool_results(&state.messages);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_error);
        assert_eq!(
            results[0].content[0].plain_text(),
            Some("Tool missing not found")
        );
        assert_eq!(
            state
                .messages
                .last()
                .map(llm::Message::text_preview)
                .as_deref(),
            Some("recovered")
        );
    }

    #[test]
    fn a_tool_batch_ends_the_run_only_when_every_result_terminates() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let stop = Tool::new(
            "stop",
            "Stop",
            "Optionally ends the run",
            json!({
                "type": "object",
                "properties": {"terminate": {"type": "boolean"}},
                "required": ["terminate"]
            }),
            |_, _, arguments, _| {
                Ok(ToolResult {
                    terminate: arguments["terminate"].as_bool().unwrap_or_default(),
                    ..ToolResult::text("ok")
                })
            },
        );
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                tools: vec![stop],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                let terminate_all = request_count.fetch_add(1, Ordering::Relaxed) > 0;
                Ok(assistant_calls(
                    vec![
                        tool_call("a", "stop", json!({"terminate": true})),
                        tool_call("b", "stop", json!({"terminate": terminate_all})),
                    ],
                    stream::STOP_TOOL_USE,
                ))
            })),
            ..AgentOptions::default()
        });

        agent.prompt("go").expect("prompt");

        assert_eq!(
            requests.load(Ordering::Relaxed),
            2,
            "a mixed batch continues; a unanimous batch stops"
        );
        let state = agent.state();
        assert_eq!(
            state.messages.last().map(llm::Message::role),
            Some("toolResult")
        );
        assert!(!state.is_streaming);
    }

    #[test]
    fn tool_schema_coerces_numbers_and_rejects_missing_values() {
        let schema = json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        });
        let coerced = validate_arguments(
            &schema,
            BTreeMap::from([("count".to_owned(), Value::String("5".to_owned()))]),
        )
        .expect("coerce integer");
        assert_eq!(coerced["count"], Value::Number(Number::from(5)));
        assert!(validate_arguments(&schema, BTreeMap::new()).is_err());
    }

    #[test]
    fn continue_uses_queued_follow_up_after_an_assistant_message() {
        let calls = Arc::new(AtomicUsize::new(0));
        let call_count = calls.clone();
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                messages: vec![llm::Message::Assistant(Box::new(assistant_text(
                    "previous",
                )))],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                call_count.fetch_add(1, Ordering::Relaxed);
                Ok(assistant_text("follow-up reply"))
            })),
            ..AgentOptions::default()
        });
        agent.follow_up(user("next"));
        agent.continue_run().expect("continue");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(
            agent
                .state()
                .messages
                .iter()
                .any(|message| message.text_preview() == "next")
        );
    }

    #[test]
    fn steering_queued_before_the_prompt_is_delivered_with_it() {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&contexts);
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, context, _| {
                lock(&seen).push(user_texts(context));
                Ok(assistant_text("reply"))
            })),
            ..AgentOptions::default()
        });
        agent.steer(user("aside"));

        agent.prompt("main").expect("prompt");

        let contexts = lock(&contexts);
        assert_eq!(contexts.len(), 1, "the aside joins the first request");
        assert_eq!(contexts[0], ["main", "aside"]);
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn continuing_from_steering_skips_the_initial_poll() {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&contexts);
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                messages: vec![llm::Message::Assistant(Box::new(assistant_text(
                    "previous",
                )))],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, context, _| {
                lock(&seen).push(user_texts(context));
                Ok(assistant_text("reply"))
            })),
            ..AgentOptions::default()
        });
        agent.steer(user("first"));
        agent.steer(user("second"));

        agent.continue_run().expect("continue");

        let contexts = lock(&contexts);
        assert_eq!(contexts.len(), 2);
        assert_eq!(
            contexts[0],
            ["first"],
            "one-at-a-time steering delivers one message per turn"
        );
        assert_eq!(contexts[1], ["first", "second"]);
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn agent_end_listeners_observe_an_idle_stream_and_late_follow_ups_still_run() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                request_count.fetch_add(1, Ordering::Relaxed);
                Ok(assistant_text("reply"))
            })),
            ..AgentOptions::default()
        });
        let observed = Arc::new(Mutex::new(Vec::new()));
        let _subscription = agent.subscribe({
            let agent = agent.clone();
            let observed = Arc::clone(&observed);
            move |event| {
                if event.kind != EventKind::AgentEnd {
                    return;
                }
                let first_end = lock(&observed).is_empty();
                lock(&observed).push((
                    agent.state().is_streaming,
                    agent.prompt("nested") == Err(AgentError::Busy),
                ));
                if first_end {
                    agent.follow_up(user("late"));
                }
            }
        });

        agent.prompt("go").expect("prompt");

        assert_eq!(
            lock(&observed).as_slice(),
            &[(false, true), (false, true)],
            "not streaming, yet still busy, inside each AgentEnd"
        );
        assert_eq!(requests.load(Ordering::Relaxed), 2);
        let state = agent.state();
        assert!(
            state
                .messages
                .iter()
                .any(|message| message.text_preview() == "late")
        );
        assert!(!state.is_streaming);
        assert!(!agent.has_queued_messages());
    }

    #[test]
    fn taking_the_queues_returns_steering_before_follow_ups_and_empties_both() {
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            ..AgentOptions::default()
        });
        agent.follow_up(user("later"));
        agent.steer(user("now"));
        assert_eq!(agent.queued_message_count(), 2);
        let taken = agent.take_queued_messages();
        assert_eq!(
            taken
                .iter()
                .filter_map(|message| match message {
                    llm::Message::User(user) => user.content.text().map(str::to_owned),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            ["now", "later"]
        );
        assert!(!agent.has_queued_messages());
        assert!(agent.take_queued_messages().is_empty());
    }

    #[test]
    fn queued_messages_survive_a_busy_continue_and_run_after_the_turn() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let (entered_sender, entered) = mpsc::channel();
        let (release_sender, release) = mpsc::channel();
        let release = Mutex::new(release);
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    entered_sender.send(()).expect("test observes the request");
                    lock(&release).recv().expect("test releases the request");
                }
                Ok(assistant_text("reply"))
            })),
            ..AgentOptions::default()
        });

        let run = thread::spawn({
            let agent = agent.clone();
            move || agent.prompt("go")
        });
        entered.recv().expect("request started");
        agent.follow_up(user("queued"));
        assert_eq!(agent.continue_run(), Err(AgentError::Busy));
        assert_eq!(
            agent.queued_message_count(),
            1,
            "a refused continuation keeps its messages"
        );
        release_sender.send(()).expect("release");
        run.join().expect("prompt thread").expect("prompt");

        assert_eq!(requests.load(Ordering::Relaxed), 2);
        assert!(
            agent
                .state()
                .messages
                .iter()
                .any(|message| message.text_preview() == "queued")
        );
    }

    #[test]
    fn between_turn_hook_sees_the_completed_turn_and_may_compact() {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&contexts);
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let hook_calls = Arc::new(Mutex::new(Vec::new()));
        let hook_log = Arc::clone(&hook_calls);
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                tools: vec![echo_tool(Arc::new(AtomicUsize::new(0)))],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, context, _| {
                lock(&seen).push(
                    context
                        .messages
                        .iter()
                        .map(llm::Message::text_preview)
                        .collect::<Vec<_>>(),
                );
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(assistant_calls(
                        vec![tool_call("one", "echo", json!({"value": "a"}))],
                        stream::STOP_TOOL_USE,
                    ))
                } else {
                    Ok(assistant_text("done"))
                }
            })),
            ..AgentOptions::default()
        });
        agent.set_prepare_next_turn(Some(Arc::new(move |agent, turn| {
            lock(&hook_log).push((
                turn.message.stop_reason.clone(),
                turn.tool_results.len(),
                turn.new_messages.len(),
            ));
            assert!(
                agent.can_compact(),
                "the loop thread may compact between turns"
            );
            let elsewhere = thread::spawn({
                let agent = agent.clone();
                move || agent.can_compact()
            });
            assert!(!elsewhere.join().expect("probe"), "other threads may not");
            let messages = agent.state().messages;
            let kept = messages[1..].to_vec();
            agent
                .compact_if_unchanged(
                    messages.len(),
                    llm::Message::User(llm::UserMessage::text(
                        "<conversation-summary>\nsummary\n</conversation-summary>",
                        now_millis(),
                    )),
                    kept.clone(),
                    CompactionInfo {
                        summary: "summary".to_owned(),
                        tokens_before: 10,
                        cost_before: 0.0,
                        retained_messages: kept.len(),
                        timestamp: now_millis(),
                    },
                )
                .expect("compact between turns");
        })));
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&events);
        let _subscription = agent.subscribe(move |event| lock(&event_log).push(event.kind));

        agent.prompt("go").expect("prompt");

        assert_eq!(
            lock(&hook_calls).as_slice(),
            &[(stream::STOP_TOOL_USE.to_owned(), 1, 3)],
            "invoked once, between the tool turn and the reply turn"
        );
        let contexts = lock(&contexts);
        assert_eq!(contexts.len(), 2);
        assert!(contexts[1][0].contains("<conversation-summary>"));
        assert_eq!(contexts[1].len(), 3, "summary, tool call, tool result");
        let events = lock(&events);
        let compacted = events
            .iter()
            .position(|kind| *kind == EventKind::ContextCompacted)
            .expect("compaction event");
        let first_turn_end = events
            .iter()
            .position(|kind| *kind == EventKind::TurnEnd)
            .expect("turn end");
        let second_turn_start = events
            .iter()
            .rposition(|kind| *kind == EventKind::TurnStart)
            .expect("second turn start");
        assert!(first_turn_end < compacted && compacted < second_turn_start);
        assert_eq!(agent.state().messages.len(), 4);
        assert!(!agent.state().is_streaming);
    }

    #[test]
    fn a_panicking_between_turn_hook_ends_the_run_with_a_full_final_turn() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                tools: vec![echo_tool(Arc::new(AtomicUsize::new(0)))],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(assistant_calls(
                        vec![tool_call("one", "echo", json!({"value": "a"}))],
                        stream::STOP_TOOL_USE,
                    ))
                } else {
                    Ok(assistant_text("fine"))
                }
            })),
            ..AgentOptions::default()
        });
        agent.set_prepare_next_turn(Some(Arc::new(|_, _| panic!("hook failure"))));
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&events);
        let _subscription = agent.subscribe(move |event| lock(&event_log).push(event.kind));

        bounded({
            let agent = agent.clone();
            move || agent.prompt("go")
        })
        .expect("a failed run is reported through events");

        let state = agent.state();
        assert_eq!(state.error_message, "the agent runtime panicked");
        assert!(!state.is_streaming);
        assert!(state.pending_tool_calls.is_empty());
        // The guard must not outlive this block: the listener locks the same
        // mutex on every event of the next prompt.
        {
            let events = lock(&events);
            assert_eq!(
                &events[events.len() - 4..],
                &[
                    EventKind::MessageStart,
                    EventKind::MessageEnd,
                    EventKind::TurnEnd,
                    EventKind::AgentEnd,
                ],
                "the failure is a complete final turn"
            );
        }
        // The agent is not left busy: the next prompt runs normally.
        bounded({
            let agent = agent.clone();
            move || agent.prompt("again")
        })
        .expect("prompt after failure");
        assert_eq!(requests.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_listener_re_entering_the_agent_during_a_lifecycle_event_gets_busy() {
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                messages: vec![
                    user("old request"),
                    llm::Message::Assistant(Box::new(assistant_text("old reply"))),
                    user("latest request"),
                ],
                ..InitialState::default()
            },
            responder: Some(Arc::new(|_, _, _| Ok(assistant_text("reply")))),
            ..AgentOptions::default()
        });
        let re_entries = Arc::new(Mutex::new(Vec::new()));
        let _subscription = agent.subscribe({
            let agent = agent.clone();
            let re_entries = Arc::clone(&re_entries);
            move |event| {
                if event.kind == EventKind::ContextCompacted {
                    lock(&re_entries).push(agent.prompt("inside"));
                    lock(&re_entries).push(agent.reset());
                }
            }
        });
        let marker = llm::Message::User(llm::UserMessage::text(
            "<conversation-summary>\nolder work\n</conversation-summary>",
            4,
        ));
        let info = CompactionInfo {
            summary: "older work".to_owned(),
            tokens_before: 1_000,
            cost_before: 1.5,
            retained_messages: 1,
            timestamp: 4,
        };

        assert_eq!(
            agent.compact_if_unchanged(
                99,
                marker.clone(),
                vec![user("latest request")],
                info.clone()
            ),
            Err(AgentError::StaleSnapshot {
                expected: 99,
                actual: 3
            })
        );
        agent
            .compact_if_unchanged(3, marker, vec![user("latest request")], info)
            .expect("compact");

        assert_eq!(
            lock(&re_entries).as_slice(),
            &[Err(AgentError::Busy), Err(AgentError::Busy)]
        );
        agent
            .prompt("after")
            .expect("the agent is usable once the event has been delivered");
        assert_eq!(agent.state().messages.len(), 4);
    }

    #[test]
    fn compaction_replaces_transcript_and_emits_its_durable_cut() {
        let kept = vec![llm::Message::User(llm::UserMessage::text(
            "latest request",
            3,
        ))];
        let marker = llm::Message::User(llm::UserMessage::text(
            "<conversation-summary>\nolder work\n</conversation-summary>",
            4,
        ));
        let info = CompactionInfo {
            summary: "older work".to_owned(),
            tokens_before: 1_000,
            cost_before: 1.5,
            retained_messages: kept.len(),
            timestamp: 4,
        };
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                messages: vec![
                    llm::Message::User(llm::UserMessage::text("old request", 1)),
                    llm::Message::Assistant(Box::new(assistant_text("old reply"))),
                    kept[0].clone(),
                ],
                ..InitialState::default()
            },
            ..AgentOptions::default()
        });
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&events);
        let _subscription = agent.subscribe(move |event| lock(&event_log).push(event.clone()));

        agent
            .compact(marker.clone(), kept.clone(), info.clone())
            .expect("compact");

        let mut expected = vec![marker];
        expected.extend(kept.clone());
        assert_eq!(agent.state().messages, expected);
        let events = lock(&events);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::ContextCompacted);
        assert_eq!(events[0].kept, kept);
        assert_eq!(events[0].compaction.as_ref(), Some(&info));
    }

    #[test]
    fn a_panicking_listener_is_reported_once_without_blocking_the_others() {
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(|_, _, _| Ok(assistant_text("reply")))),
            ..AgentOptions::default()
        });
        let _broken = agent.subscribe(|_| panic!("listener failure"));
        let events = Arc::new(Mutex::new(Vec::new()));
        let event_log = Arc::clone(&events);
        let _healthy = agent.subscribe(move |event| lock(&event_log).push(event.kind));

        agent.prompt("go").expect("prompt");

        assert_eq!(
            lock(&events).len(),
            8,
            "every event still reaches the healthy listener"
        );
        assert!(agent.inner.listener_panic_reported.load(Ordering::Relaxed));
        assert!(!agent.state().is_streaming);
    }

    #[test]
    fn weak_follow_up_queue_forwards_without_retaining_the_agent() {
        let queue = {
            let agent = Agent::new(AgentOptions {
                initial_state: InitialState {
                    model: model(),
                    ..InitialState::default()
                },
                ..AgentOptions::default()
            });
            let queue = agent.weak_follow_up_queue();
            queue.follow_up(user("queued"));
            assert!(agent.has_queued_messages());
            queue
        };

        assert!(!queue.has_queued_messages());
        queue.follow_up(user("ignored after drop"));
        assert!(!queue.has_queued_messages());
    }
}
