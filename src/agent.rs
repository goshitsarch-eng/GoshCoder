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
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Number, Value};

use crate::llm;

pub type Result<T> = std::result::Result<T, AgentError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AgentError {
    Busy,
    EmptyTranscript,
    CannotContinue(String),
    ResetWhileRunning,
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

/// Per-request provider inputs that are not encoded in the transcript.
#[derive(Clone, Debug)]
pub struct RequestOptions {
    pub cancellation: CancellationToken,
    pub thinking_level: llm::ThinkingLevel,
    pub session_id: String,
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

#[derive(Clone)]
pub struct InitialState {
    pub system_prompt: String,
    pub model: llm::Model,
    pub thinking_level: llm::ThinkingLevel,
    pub tools: Vec<Tool>,
    pub messages: Vec<llm::Message>,
}

impl Default for InitialState {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: llm::Model::default(),
            thinking_level: llm::THINKING_OFF.to_owned(),
            tools: Vec::new(),
            messages: Vec::new(),
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
    TranscriptReset,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub kind: EventKind,
    pub message: Option<llm::Message>,
    pub messages: Vec<llm::Message>,
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
            messages: Vec::new(),
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

pub type Listener = Arc<dyn Fn(Event) + Send + Sync + 'static>;

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

struct AgentInner {
    state: Mutex<InnerState>,
    idle: Condvar,
    listeners: Mutex<BTreeMap<usize, Listener>>,
    next_listener_id: AtomicUsize,
    responder: AssistantResponder,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    tool_execution: ToolExecutionMode,
    session_id: String,
}

struct InnerState {
    system_prompt: String,
    model: llm::Model,
    thinking_level: llm::ThinkingLevel,
    tools: Vec<Tool>,
    messages: Vec<llm::Message>,
    streaming_message: Option<llm::Message>,
    pending_tool_calls: BTreeSet<String>,
    error_message: String,
    cancellation: Option<CancellationToken>,
    steering: Vec<llm::Message>,
    follow_ups: Vec<llm::Message>,
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
                    streaming_message: None,
                    pending_tool_calls: BTreeSet::new(),
                    error_message: String::new(),
                    cancellation: None,
                    steering: Vec::new(),
                    follow_ups: Vec::new(),
                }),
                idle: Condvar::new(),
                listeners: Mutex::new(BTreeMap::new()),
                next_listener_id: AtomicUsize::new(1),
                responder,
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
            is_streaming: state.cancellation.is_some(),
            streaming_message: state.streaming_message.clone(),
            pending_tool_calls: state.pending_tool_calls.iter().cloned().collect(),
            error_message: state.error_message.clone(),
        }
    }

    pub fn subscribe(&self, listener: impl Fn(Event) + Send + Sync + 'static) -> Subscription {
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
            self.emit(event);
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
            self.emit(event);
        }
    }

    pub fn set_tools(&self, tools: Vec<Tool>) {
        lock(&self.inner.state).tools = tools;
    }

    pub fn set_messages(&self, messages: Vec<llm::Message>) {
        lock(&self.inner.state).messages = messages;
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

    pub fn wait_for_idle(&self) {
        let mut state = lock(&self.inner.state);
        while state.cancellation.is_some() {
            state = self
                .inner
                .idle
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    }

    pub fn reset(&self) -> Result<()> {
        self.reset_with_reason("")
    }

    pub fn reset_with_reason(&self, reason: impl Into<String>) -> Result<()> {
        {
            let mut state = lock(&self.inner.state);
            if state.cancellation.is_some() {
                return Err(AgentError::ResetWhileRunning);
            }
            state.messages.clear();
            state.steering.clear();
            state.follow_ups.clear();
            state.streaming_message = None;
            state.pending_tool_calls.clear();
            state.error_message.clear();
        }
        let mut event = Event::kind(EventKind::TranscriptReset);
        event.reason = reason.into();
        self.emit(event);
        Ok(())
    }

    pub fn prompt(&self, prompt: impl Into<String>) -> Result<()> {
        self.prompt_messages(vec![llm::Message::User(llm::UserMessage::text(
            prompt,
            now_millis(),
        ))])
    }

    pub fn prompt_messages(&self, messages: Vec<llm::Message>) -> Result<()> {
        self.run(messages)
    }

    pub fn continue_run(&self) -> Result<()> {
        let continuation = {
            let mut state = lock(&self.inner.state);
            if state.cancellation.is_some() {
                return Err(AgentError::Busy);
            }
            let Some(last) = state.messages.last() else {
                return Err(AgentError::EmptyTranscript);
            };
            if last.role() != "assistant" {
                Vec::new()
            } else if let Some(messages) = drain(&mut state.steering, self.inner.steering_mode) {
                messages
            } else if let Some(messages) = drain(&mut state.follow_ups, self.inner.follow_up_mode) {
                messages
            } else {
                return Err(AgentError::CannotContinue("assistant".to_owned()));
            }
        };
        self.run(continuation)
    }

    fn run(&self, initial_messages: Vec<llm::Message>) -> Result<()> {
        let cancellation = {
            let mut state = lock(&self.inner.state);
            if state.cancellation.is_some() {
                return Err(AgentError::Busy);
            }
            state.cancellation = Some(CancellationToken::default());
            state.streaming_message = None;
            state.error_message.clear();
            state
                .cancellation
                .clone()
                .expect("cancellation was just assigned")
        };

        let messages = match catch_unwind(AssertUnwindSafe(|| {
            self.run_loop(initial_messages, cancellation.clone())
        })) {
            Ok(messages) => messages,
            Err(_) => {
                let error = self.error_message("the agent runtime panicked");
                let message = llm::Message::Assistant(Box::new(error));
                self.record_message(message.clone());
                vec![message]
            }
        };
        let mut event = Event::kind(EventKind::AgentEnd);
        event.messages = messages;
        self.emit(event);

        let mut state = lock(&self.inner.state);
        state.streaming_message = None;
        state.pending_tool_calls.clear();
        state.cancellation = None;
        self.inner.idle.notify_all();
        Ok(())
    }

    fn run_loop(
        &self,
        mut pending_messages: Vec<llm::Message>,
        cancellation: CancellationToken,
    ) -> Vec<llm::Message> {
        let mut new_messages = Vec::new();
        self.emit(Event::kind(EventKind::AgentStart));

        loop {
            self.emit(Event::kind(EventKind::TurnStart));
            for message in pending_messages.drain(..) {
                self.record_message(message.clone());
                new_messages.push(message);
            }

            let assistant = if cancellation.is_cancelled() {
                self.error_message("Request was aborted")
            } else {
                self.request_assistant(cancellation.clone())
            };
            let is_terminal_error = matches!(assistant.stop_reason.as_str(), "error" | "aborted");
            self.record_message(llm::Message::Assistant(Box::new(assistant.clone())));
            new_messages.push(llm::Message::Assistant(Box::new(assistant.clone())));

            let outcomes = if is_terminal_error {
                Vec::new()
            } else {
                self.execute_tool_calls(&assistant, cancellation.clone())
            };
            let mut tool_results = Vec::new();
            for outcome in outcomes {
                let message = outcome.as_message();
                self.record_message(message.clone());
                new_messages.push(message.clone());
                tool_results.push(message);
            }

            let mut turn_end = Event::kind(EventKind::TurnEnd);
            turn_end.message = Some(llm::Message::Assistant(Box::new(assistant)));
            turn_end.messages = tool_results;
            self.emit(turn_end);
            if is_terminal_error || cancellation.is_cancelled() {
                break;
            }

            pending_messages = self.drain_steering();
            if !pending_messages.is_empty() {
                continue;
            }
            if !new_messages
                .last()
                .is_some_and(|message| matches!(message, llm::Message::ToolResult(_)))
            {
                pending_messages = self.drain_follow_ups();
                if pending_messages.is_empty() {
                    break;
                }
                continue;
            }
            if all_terminate(&new_messages) {
                break;
            }
        }
        new_messages
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
        let response = catch_unwind(AssertUnwindSafe(|| {
            (self.inner.responder)(
                &model,
                &context,
                RequestOptions {
                    cancellation: cancellation.clone(),
                    thinking_level,
                    session_id: self.inner.session_id.clone(),
                },
            )
        }));
        match response {
            Ok(Ok(message)) => message,
            Ok(Err(error)) => self.error_message(&error),
            Err(_) => self.error_message("provider streamer panicked"),
        }
    }

    fn execute_tool_calls(
        &self,
        assistant: &llm::AssistantMessage,
        cancellation: CancellationToken,
    ) -> Vec<ToolOutcome> {
        let calls = assistant
            .content
            .iter()
            .filter_map(|block| match block {
                llm::ContentBlock::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
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
                    outcomes.push((index, outcome));
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
                update_agent.emit(event);
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
        self.emit(event);
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
        self.emit(event);
    }

    fn record_message(&self, message: llm::Message) {
        {
            let mut state = lock(&self.inner.state);
            state.streaming_message = Some(message.clone());
        }
        let mut start = Event::kind(EventKind::MessageStart);
        start.message = Some(message.clone());
        self.emit(start);
        {
            let mut state = lock(&self.inner.state);
            state.messages.push(message.clone());
            state.streaming_message = None;
            if let llm::Message::Assistant(assistant) = &message
                && !assistant.error_message.is_empty()
            {
                state.error_message = assistant.error_message.clone();
            }
        }
        let mut end = Event::kind(EventKind::MessageEnd);
        end.message = Some(message);
        self.emit(end);
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

    fn emit(&self, event: Event) {
        let listeners = lock(&self.inner.listeners)
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for listener in listeners {
            let _ = catch_unwind(AssertUnwindSafe(|| listener(event.clone())));
        }
    }
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
) -> std::result::Result<PreparedToolCall, ToolOutcome> {
    let Some(tool) = tools.iter().find(|tool| tool.name == call.name).cloned() else {
        let message = format!("Tool {} not found", call.name);
        return Err(ToolOutcome::error(call, message));
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
        Err(error) => Err(ToolOutcome::error(
            call,
            format!("validation failed: {error}"),
        )),
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

fn all_terminate(messages: &[llm::Message]) -> bool {
    let results = messages
        .iter()
        .rev()
        .take_while(|message| matches!(message, llm::Message::ToolResult(_)))
        .collect::<Vec<_>>();
    !results.is_empty()
        && results.iter().all(|message| {
            matches!(
                message,
                llm::Message::ToolResult(result) if result.is_error || result.content.is_empty()
            )
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
    use std::sync::atomic::{AtomicUsize, Ordering};

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
    fn tool_calls_continue_the_turn_and_preserve_source_order() {
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = requests.clone();
        let tool = Tool::new(
            "echo",
            "Echo",
            "Returns its value",
            json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
            |_, id, arguments, _| {
                Ok(ToolResult::text(format!(
                    "{id}:{}",
                    arguments["value"].as_str().unwrap_or_default()
                )))
            },
        );
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                tools: vec![tool],
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                if request_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Ok(llm::AssistantMessage {
                        role: "assistant".to_owned(),
                        content: vec![
                            llm::ContentBlock::ToolCall(llm::ToolCall {
                                id: "one".to_owned(),
                                name: "echo".to_owned(),
                                arguments: BTreeMap::from([(
                                    "value".to_owned(),
                                    Value::String("a".to_owned()),
                                )]),
                                thought_signature: String::new(),
                                namespace: String::new(),
                            }),
                            llm::ContentBlock::ToolCall(llm::ToolCall {
                                id: "two".to_owned(),
                                name: "echo".to_owned(),
                                arguments: BTreeMap::from([(
                                    "value".to_owned(),
                                    Value::String("b".to_owned()),
                                )]),
                                thought_signature: String::new(),
                                namespace: String::new(),
                            }),
                        ],
                        api: "test".to_owned(),
                        provider: "test".to_owned(),
                        model: "test-model".to_owned(),
                        stop_reason: "toolUse".to_owned(),
                        timestamp: now_millis(),
                        ..llm::AssistantMessage::default()
                    })
                } else {
                    Ok(assistant_text("done"))
                }
            })),
            ..AgentOptions::default()
        });

        agent.prompt("run tools").expect("prompt");
        assert_eq!(requests.load(Ordering::Relaxed), 2);
        let state = agent.state();
        let result_ids = state
            .messages
            .iter()
            .filter_map(|message| match message {
                llm::Message::ToolResult(result) => Some(result.tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(result_ids, ["one", "two"]);
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
        agent.follow_up(llm::Message::User(llm::UserMessage::text(
            "next",
            now_millis(),
        )));
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
}
