//! Protocol-independent infrastructure for streaming LLM responses.
//!
//! Provider adapters can use this module without depending on an HTTP client:
//! it contains bounded event delivery, SSE framing, tolerant tool-argument
//! parsing, retry decisions, and model/context helpers. Wire transports remain
//! responsible for fetching bytes and turning provider payloads into these
//! public types.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    error::Error,
    fmt,
    io::{self, BufRead, BufReader, Read},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::llm;

/// A normalized reason why an assistant response stopped.
pub type StopReason = String;

pub const STOP_PENDING: &str = "pending";
pub const STOP_STOP: &str = "stop";
pub const STOP_LENGTH: &str = "length";
pub const STOP_TOOL_USE: &str = "toolUse";
pub const STOP_ERROR: &str = "error";
pub const STOP_ABORTED: &str = "aborted";
pub const STOP_DEFERRED: &str = "deferred";

/// Returns whether `reason` represents a terminal assistant outcome.
pub fn is_terminal_stop_reason(reason: &str) -> bool {
    matches!(
        reason,
        STOP_STOP | STOP_LENGTH | STOP_TOOL_USE | STOP_ERROR | STOP_ABORTED | STOP_DEFERRED
    )
}

/// Returns whether `reason` is a terminal outcome that did not fail or abort.
pub fn is_successful_stop_reason(reason: &str) -> bool {
    matches!(
        reason,
        STOP_STOP | STOP_LENGTH | STOP_TOOL_USE | STOP_DEFERRED
    )
}

/// Returns whether `reason` represents an error or explicit cancellation.
pub fn is_failure_stop_reason(reason: &str) -> bool {
    matches!(reason, STOP_ERROR | STOP_ABORTED)
}

pub const EVENT_START: &str = "start";
pub const EVENT_TEXT_START: &str = "text_start";
pub const EVENT_TEXT_DELTA: &str = "text_delta";
pub const EVENT_TEXT_END: &str = "text_end";
pub const EVENT_THINKING_START: &str = "thinking_start";
pub const EVENT_THINKING_DELTA: &str = "thinking_delta";
pub const EVENT_THINKING_END: &str = "thinking_end";
pub const EVENT_TOOLCALL_START: &str = "toolcall_start";
pub const EVENT_TOOLCALL_DELTA: &str = "toolcall_delta";
pub const EVENT_TOOLCALL_END: &str = "toolcall_end";
pub const EVENT_DONE: &str = "done";
pub const EVENT_ERROR: &str = "error";

/// The shared, progressively assembled assistant message carried by events.
///
/// `Arc` lets a producer publish snapshots without exposing mutable shared
/// state. A producer that needs to continue changing a message should publish
/// a fresh `Arc` (or use [`Arc::make_mut`], which clones when readers exist).
pub type SharedAssistantMessage = Arc<llm::AssistantMessage>;

/// One normalized event emitted while an assistant message is assembled.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssistantMessageEvent {
    /// One of the `EVENT_*` constants. A string deliberately keeps the
    /// protocol boundary forward-compatible with new event kinds.
    pub event_type: String,
    /// The content-block index for text, thinking, and tool-call events.
    pub content_index: Option<usize>,
    /// The incremental text, thinking, or tool-argument fragment.
    pub delta: String,
    /// The final text or thinking content for an end event.
    pub content: String,
    /// The completed tool call for a `toolcall_end` event.
    pub tool_call: Option<llm::ToolCall>,
    /// The normalized stop reason for terminal events.
    pub reason: StopReason,
    /// The current assistant-message snapshot while streaming.
    pub partial: Option<SharedAssistantMessage>,
    /// The final assistant message carried by a `done` event.
    pub message: Option<SharedAssistantMessage>,
    /// The final assistant error message carried by an `error` event.
    pub error: Option<SharedAssistantMessage>,
}

impl AssistantMessageEvent {
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            ..Self::default()
        }
    }

    pub fn start(partial: SharedAssistantMessage) -> Self {
        Self {
            event_type: EVENT_START.to_owned(),
            partial: Some(partial),
            ..Self::default()
        }
    }

    pub fn done(reason: impl Into<StopReason>, message: SharedAssistantMessage) -> Self {
        Self {
            event_type: EVENT_DONE.to_owned(),
            reason: reason.into(),
            message: Some(message),
            ..Self::default()
        }
    }

    pub fn error(reason: impl Into<StopReason>, error: SharedAssistantMessage) -> Self {
        Self {
            event_type: EVENT_ERROR.to_owned(),
            reason: reason.into(),
            error: Some(error),
            ..Self::default()
        }
    }

    pub fn is_terminal(&self) -> bool {
        is_terminal_event_type(&self.event_type)
    }

    /// Returns the message that resolves the event stream, if this is a valid
    /// terminal event.
    pub fn terminal_message(&self) -> Option<SharedAssistantMessage> {
        match self.event_type.as_str() {
            EVENT_DONE => self.message.clone(),
            EVENT_ERROR => self.error.clone(),
            _ => None,
        }
    }
}

pub fn is_terminal_event_type(event_type: &str) -> bool {
    matches!(event_type, EVENT_DONE | EVENT_ERROR)
}

/// The default maximum number of events retained for a stalled consumer.
pub const DEFAULT_EVENT_STREAM_CAPACITY: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventStreamConfigurationError {
    ZeroCapacity,
}

impl fmt::Display for EventStreamConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => {
                formatter.write_str("event stream capacity must be greater than zero")
            }
        }
    }
}

impl Error for EventStreamConfigurationError {}

/// A failed bounded-stream enqueue. The original event is retained so callers
/// can retry it, drop it intentionally, or turn it into a provider failure.
#[derive(Clone, Debug, PartialEq)]
pub enum EventStreamPushError {
    Closed(Box<AssistantMessageEvent>),
    Full(Box<AssistantMessageEvent>),
    TimedOut(Box<AssistantMessageEvent>),
    InvalidTerminalEvent(Box<AssistantMessageEvent>),
}

impl fmt::Display for EventStreamPushError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed(event) => write!(
                formatter,
                "assistant event stream is closed; cannot push {:?}",
                event.event_type
            ),
            Self::Full(event) => write!(
                formatter,
                "assistant event stream is full; cannot push {:?}",
                event.event_type
            ),
            Self::TimedOut(event) => write!(
                formatter,
                "timed out waiting to push {:?} to assistant event stream",
                event.event_type
            ),
            Self::InvalidTerminalEvent(event) => write!(
                formatter,
                "terminal event {:?} does not carry its final assistant message",
                event.event_type
            ),
        }
    }
}

impl Error for EventStreamPushError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventStreamWaitError {
    TimedOut,
}

impl fmt::Display for EventStreamWaitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("timed out waiting for an assistant stream event")
    }
}

impl Error for EventStreamWaitError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventStreamResultError {
    EndedWithoutTerminalEvent,
    TimedOut,
}

impl fmt::Display for EventStreamResultError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndedWithoutTerminalEvent => {
                formatter.write_str("assistant stream ended without a done or error event")
            }
            Self::TimedOut => formatter.write_str("timed out waiting for assistant stream result"),
        }
    }
}

impl Error for EventStreamResultError {}

#[derive(Debug)]
struct EventStreamState {
    queue: VecDeque<AssistantMessageEvent>,
    ended: bool,
    terminal_result: Option<SharedAssistantMessage>,
}

struct EventStreamInner {
    capacity: usize,
    state: Mutex<EventStreamState>,
    changed: Condvar,
    space_available: Condvar,
}

/// A thread-safe, bounded, ordered assistant-event stream.
///
/// Unlike an unbounded queue, this stream applies backpressure when a consumer
/// stalls. Providers can choose [`Self::try_push`] to avoid waiting or
/// [`Self::push_timeout`] to bound the wait. A terminal `done`/`error` event
/// closes the producer side, but queued events remain available to consumers.
#[derive(Clone)]
pub struct AssistantMessageEventStream {
    inner: Arc<EventStreamInner>,
}

impl fmt::Debug for AssistantMessageEventStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock_state();
        formatter
            .debug_struct("AssistantMessageEventStream")
            .field("capacity", &self.inner.capacity)
            .field("queued_events", &state.queue.len())
            .field("ended", &state.ended)
            .field("has_terminal_result", &state.terminal_result.is_some())
            .finish()
    }
}

impl Default for AssistantMessageEventStream {
    fn default() -> Self {
        Self::new()
    }
}

impl AssistantMessageEventStream {
    pub fn new() -> Self {
        // The constant is non-zero and is kept separate from callers' dynamic
        // input, so this cannot fail.
        Self::with_capacity(DEFAULT_EVENT_STREAM_CAPACITY)
            .expect("default event stream capacity must be non-zero")
    }

    pub fn with_capacity(capacity: usize) -> Result<Self, EventStreamConfigurationError> {
        if capacity == 0 {
            return Err(EventStreamConfigurationError::ZeroCapacity);
        }
        Ok(Self {
            inner: Arc::new(EventStreamInner {
                capacity,
                state: Mutex::new(EventStreamState {
                    queue: VecDeque::with_capacity(capacity),
                    ended: false,
                    terminal_result: None,
                }),
                changed: Condvar::new(),
                space_available: Condvar::new(),
            }),
        })
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn len(&self) -> usize {
        self.lock_state().queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true once no more events may be pushed.
    pub fn is_closed(&self) -> bool {
        let state = self.lock_state();
        state.ended || state.terminal_result.is_some()
    }

    /// Attempts an enqueue without blocking if the bounded queue is full.
    pub fn try_push(&self, event: AssistantMessageEvent) -> Result<(), EventStreamPushError> {
        self.push_inner(event, Some(Duration::ZERO))
    }

    /// Enqueues an event, waiting until there is room or the stream closes.
    pub fn push(&self, event: AssistantMessageEvent) -> Result<(), EventStreamPushError> {
        self.push_inner(event, None)
    }

    /// Enqueues an event while waiting for no longer than `timeout`.
    pub fn push_timeout(
        &self,
        event: AssistantMessageEvent,
        timeout: Duration,
    ) -> Result<(), EventStreamPushError> {
        self.push_inner(event, Some(timeout))
    }

    fn push_inner(
        &self,
        event: AssistantMessageEvent,
        timeout: Option<Duration>,
    ) -> Result<(), EventStreamPushError> {
        let terminal_result = if event.is_terminal() {
            match event.terminal_message() {
                Some(message) => Some(message),
                None => {
                    return Err(EventStreamPushError::InvalidTerminalEvent(Box::new(event)));
                }
            }
        } else {
            None
        };
        let deadline = timeout.and_then(|duration| Instant::now().checked_add(duration));
        let mut state = self.lock_state();

        loop {
            if state.ended || state.terminal_result.is_some() {
                return Err(EventStreamPushError::Closed(Box::new(event)));
            }
            if state.queue.len() < self.inner.capacity {
                if let Some(result) = terminal_result {
                    state.terminal_result = Some(result);
                }
                state.queue.push_back(event);
                self.inner.changed.notify_all();
                return Ok(());
            }

            match deadline {
                None if timeout.is_none() => {
                    state = self
                        .inner
                        .space_available
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return if timeout == Some(Duration::ZERO) {
                            Err(EventStreamPushError::Full(Box::new(event)))
                        } else {
                            Err(EventStreamPushError::TimedOut(Box::new(event)))
                        };
                    }
                    let (next_state, wait) = self
                        .inner
                        .space_available
                        .wait_timeout(state, remaining)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    state = next_state;
                    if wait.timed_out() && state.queue.len() >= self.inner.capacity {
                        if state.ended || state.terminal_result.is_some() {
                            return Err(EventStreamPushError::Closed(Box::new(event)));
                        }
                        return Err(EventStreamPushError::TimedOut(Box::new(event)));
                    }
                }
                // An extremely large timeout could not be represented as an
                // `Instant`; waiting indefinitely is safer than spinning.
                None => {
                    state = self
                        .inner
                        .space_available
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
            }
        }
    }

    /// Marks the producer side as closed. Consumers can still drain queued
    /// events. Calling this without a terminal event makes [`Self::result`]
    /// return [`EventStreamResultError::EndedWithoutTerminalEvent`].
    pub fn end(&self) {
        let mut state = self.lock_state();
        if !state.ended {
            state.ended = true;
            self.inner.changed.notify_all();
            self.inner.space_available.notify_all();
        }
    }

    /// Removes an already available event without waiting.
    pub fn try_next(&self) -> Option<AssistantMessageEvent> {
        let mut state = self.lock_state();
        let event = state.queue.pop_front();
        if event.is_some() {
            self.inner.space_available.notify_one();
        }
        event
    }

    /// Blocks until the next event is available or the stream is closed and
    /// drained. `None` means the stream can produce no more events.
    pub fn next(&self) -> Option<AssistantMessageEvent> {
        let mut state = self.lock_state();
        loop {
            if let Some(event) = state.queue.pop_front() {
                self.inner.space_available.notify_one();
                return Some(event);
            }
            if state.ended || state.terminal_result.is_some() {
                return None;
            }
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Waits for one event for at most `timeout`.
    ///
    /// `Ok(None)` means the stream closed and drained; `TimedOut` means it is
    /// still active but no event arrived in time.
    pub fn next_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<AssistantMessageEvent>, EventStreamWaitError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(Instant::now);
        let mut state = self.lock_state();
        loop {
            if let Some(event) = state.queue.pop_front() {
                self.inner.space_available.notify_one();
                return Ok(Some(event));
            }
            if state.ended || state.terminal_result.is_some() {
                return Ok(None);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(EventStreamWaitError::TimedOut);
            }
            let (next_state, wait) = self
                .inner
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next_state;
            if wait.timed_out() && state.queue.is_empty() {
                if state.ended || state.terminal_result.is_some() {
                    return Ok(None);
                }
                return Err(EventStreamWaitError::TimedOut);
            }
        }
    }

    /// Waits for the final message from a `done` or `error` event.
    pub fn result(&self) -> Result<SharedAssistantMessage, EventStreamResultError> {
        let mut state = self.lock_state();
        loop {
            if let Some(result) = &state.terminal_result {
                return Ok(Arc::clone(result));
            }
            if state.ended {
                return Err(EventStreamResultError::EndedWithoutTerminalEvent);
            }
            state = self
                .inner
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    /// Waits for a final `done`/`error` message for at most `timeout`.
    pub fn result_timeout(
        &self,
        timeout: Duration,
    ) -> Result<SharedAssistantMessage, EventStreamResultError> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(Instant::now);
        let mut state = self.lock_state();
        loop {
            if let Some(result) = &state.terminal_result {
                return Ok(Arc::clone(result));
            }
            if state.ended {
                return Err(EventStreamResultError::EndedWithoutTerminalEvent);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(EventStreamResultError::TimedOut);
            }
            let (next_state, wait) = self
                .inner
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next_state;
            if wait.timed_out() && state.terminal_result.is_none() {
                if state.ended {
                    return Err(EventStreamResultError::EndedWithoutTerminalEvent);
                }
                return Err(EventStreamResultError::TimedOut);
            }
        }
    }

    /// Creates a blocking iterator over the remaining events.
    pub fn iter(&self) -> AssistantMessageEventIter {
        AssistantMessageEventIter {
            stream: self.clone(),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, EventStreamState> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A blocking iterator returned by [`AssistantMessageEventStream::iter`].
pub struct AssistantMessageEventIter {
    stream: AssistantMessageEventStream,
}

impl Iterator for AssistantMessageEventIter {
    type Item = AssistantMessageEvent;

    fn next(&mut self) -> Option<Self::Item> {
        self.stream.next()
    }
}

/// One dispatched Server-Sent Event record.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
    pub id: String,
}

pub const MAX_PROVIDER_STREAM_BYTES: usize = 256 << 20;
pub const MAX_SSE_LINE_BYTES: usize = 16 << 20;
pub const MAX_SSE_EVENT_BYTES: usize = 32 << 20;

/// Size limits enforced before provider-controlled SSE input is buffered.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SseLimits {
    pub max_stream_bytes: usize,
    pub max_line_bytes: usize,
    pub max_event_bytes: usize,
}

impl Default for SseLimits {
    fn default() -> Self {
        Self {
            max_stream_bytes: MAX_PROVIDER_STREAM_BYTES,
            max_line_bytes: MAX_SSE_LINE_BYTES,
            max_event_bytes: MAX_SSE_EVENT_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SseConfigurationError {
    ZeroStreamLimit,
    ZeroLineLimit,
    ZeroEventLimit,
}

impl fmt::Display for SseConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStreamLimit => {
                formatter.write_str("SSE stream byte limit must be greater than zero")
            }
            Self::ZeroLineLimit => {
                formatter.write_str("SSE line byte limit must be greater than zero")
            }
            Self::ZeroEventLimit => {
                formatter.write_str("SSE event byte limit must be greater than zero")
            }
        }
    }
}

impl Error for SseConfigurationError {}

#[derive(Debug)]
pub enum SseError {
    Io(io::Error),
    InvalidUtf8(std::string::FromUtf8Error),
    LineTooLarge { limit: usize },
    EventTooLarge { limit: usize },
}

impl fmt::Display for SseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "failed to read SSE stream: {error}"),
            Self::InvalidUtf8(error) => write!(formatter, "SSE stream is not valid UTF-8: {error}"),
            Self::LineTooLarge { limit } => {
                write!(formatter, "SSE line exceeds the {limit}-byte size limit")
            }
            Self::EventTooLarge { limit } => {
                write!(formatter, "SSE event exceeds the {limit}-byte size limit")
            }
        }
    }
}

impl Error for SseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::InvalidUtf8(error) => Some(error),
            Self::LineTooLarge { .. } | Self::EventTooLarge { .. } => None,
        }
    }
}

struct LimitedReader<R> {
    reader: R,
    remaining: usize,
}

impl<R: Read> Read for LimitedReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || buffer.is_empty() {
            return Ok(0);
        }
        let allowed = buffer.len().min(self.remaining);
        let read = self.reader.read(&mut buffer[..allowed])?;
        self.remaining -= read;
        Ok(read)
    }
}

/// An SSE parser that performs framing only; it does not open connections or
/// interpret provider payloads.
pub struct SseReader<R> {
    reader: BufReader<LimitedReader<R>>,
    limits: SseLimits,
}

/// Alias for callers that prefer parser terminology.
pub type SseParser<R> = SseReader<R>;

impl<R: Read> SseReader<R> {
    pub fn new(reader: R) -> Self {
        Self::with_limits(reader, SseLimits::default())
            .expect("the default SSE limits must be valid")
    }

    pub fn with_limits(reader: R, limits: SseLimits) -> Result<Self, SseConfigurationError> {
        if limits.max_stream_bytes == 0 {
            return Err(SseConfigurationError::ZeroStreamLimit);
        }
        if limits.max_line_bytes == 0 {
            return Err(SseConfigurationError::ZeroLineLimit);
        }
        if limits.max_event_bytes == 0 {
            return Err(SseConfigurationError::ZeroEventLimit);
        }
        Ok(Self {
            reader: BufReader::new(LimitedReader {
                reader,
                remaining: limits.max_stream_bytes,
            }),
            limits,
        })
    }

    pub fn limits(&self) -> SseLimits {
        self.limits
    }

    /// Reads one event dispatched by a blank line. `Ok(None)` denotes EOF; a
    /// partial record at EOF is deliberately discarded per the SSE framing
    /// rules.
    pub fn next_event(&mut self) -> Result<Option<SseEvent>, SseError> {
        let mut event = SseEvent::default();
        let mut data = String::new();
        let mut have_data = false;

        loop {
            let Some((line, terminated)) = self.read_line()? else {
                return Ok(None);
            };
            // A final unterminated line cannot dispatch an SSE record. Any
            // preceding fields belong to that same incomplete record.
            if !terminated {
                return Ok(None);
            }
            if let Some(dispatched) =
                self.process_line(&line, &mut event, &mut data, &mut have_data)?
            {
                return Ok(Some(dispatched));
            }
        }
    }

    pub fn events(&mut self) -> SseEvents<'_, R> {
        SseEvents {
            reader: self,
            finished: false,
        }
    }

    fn read_line(&mut self) -> Result<Option<(String, bool)>, SseError> {
        let mut line = Vec::new();

        loop {
            let (fragment_len, terminated) = {
                let buffer = self.reader.fill_buf().map_err(SseError::Io)?;
                if buffer.is_empty() {
                    if line.is_empty() {
                        return Ok(None);
                    }
                    let text = String::from_utf8(line).map_err(SseError::InvalidUtf8)?;
                    return Ok(Some((text, false)));
                }
                let fragment_len = buffer
                    .iter()
                    .position(|byte| *byte == b'\n')
                    .map_or(buffer.len(), |index| index + 1);
                if line.len().saturating_add(fragment_len) > self.limits.max_line_bytes {
                    return Err(SseError::LineTooLarge {
                        limit: self.limits.max_line_bytes,
                    });
                }
                line.extend_from_slice(&buffer[..fragment_len]);
                (fragment_len, buffer[fragment_len - 1] == b'\n')
            };
            self.reader.consume(fragment_len);

            if terminated {
                debug_assert_eq!(line.last(), Some(&b'\n'));
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                let text = String::from_utf8(line).map_err(SseError::InvalidUtf8)?;
                return Ok(Some((text, true)));
            }
        }
    }

    fn process_line(
        &self,
        line: &str,
        event: &mut SseEvent,
        data: &mut String,
        have_data: &mut bool,
    ) -> Result<Option<SseEvent>, SseError> {
        if line.is_empty() {
            if !*have_data {
                *event = SseEvent::default();
                return Ok(None);
            }
            event.data = std::mem::take(data);
            *have_data = false;
            return Ok(Some(std::mem::take(event)));
        }
        if line.starts_with(':') {
            return Ok(None);
        }

        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => {
                let separator_bytes = usize::from(*have_data);
                if data
                    .len()
                    .saturating_add(separator_bytes)
                    .saturating_add(value.len())
                    > self.limits.max_event_bytes
                {
                    return Err(SseError::EventTooLarge {
                        limit: self.limits.max_event_bytes,
                    });
                }
                if *have_data {
                    data.push('\n');
                }
                data.push_str(value);
                *have_data = true;
            }
            "event" => event.event = value.to_owned(),
            "id" => event.id = value.to_owned(),
            "retry" => {
                // Reconnection delays are intentionally left to an owning
                // transport. LLM streams normally do not reconnect in place.
            }
            _ => {}
        }
        Ok(None)
    }
}

/// Fallible iterator over SSE records.
pub struct SseEvents<'a, R> {
    reader: &'a mut SseReader<R>,
    finished: bool,
}

impl<R: Read> Iterator for SseEvents<'_, R> {
    type Item = Result<SseEvent, SseError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.reader.next_event() {
            Ok(Some(event)) => Some(Ok(event)),
            Ok(None) => {
                self.finished = true;
                None
            }
            Err(error) => {
                self.finished = true;
                Some(Err(error))
            }
        }
    }
}

/// A tolerant-parser failure that means more stream data is required.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartialJsonError {
    Incomplete,
}

impl fmt::Display for PartialJsonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("incomplete JSON")
    }
}

impl Error for PartialJsonError {}

/// Escapes raw controls and invalid string escapes so an otherwise complete
/// provider JSON payload can be fed to a strict parser.
pub fn repair_json(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut repaired = Vec::with_capacity(bytes.len());
    let mut in_string = false;
    let mut index = 0;

    while index < bytes.len() {
        let byte = bytes[index];
        if !in_string {
            repaired.push(byte);
            if byte == b'"' {
                in_string = true;
            }
            index += 1;
            continue;
        }

        match byte {
            b'"' => {
                repaired.push(byte);
                in_string = false;
                index += 1;
            }
            b'\\' => {
                if index + 1 >= bytes.len() {
                    repaired.extend_from_slice(br"\\");
                    index += 1;
                    continue;
                }
                let next = bytes[index + 1];
                if next == b'u'
                    && index + 6 <= bytes.len()
                    && bytes[index + 2..index + 6]
                        .iter()
                        .all(|byte| byte.is_ascii_hexdigit())
                {
                    repaired.extend_from_slice(&bytes[index..index + 6]);
                    index += 6;
                } else if is_valid_json_escape(next) {
                    repaired.push(b'\\');
                    repaired.push(next);
                    index += 2;
                } else {
                    // Keep the following byte for the next iteration so the
                    // repaired result becomes `\\x`, not merely `\\`.
                    repaired.extend_from_slice(br"\\");
                    index += 1;
                }
            }
            0x00..=0x1f => {
                push_control_escape(&mut repaired, byte);
                index += 1;
            }
            _ => {
                repaired.push(byte);
                index += 1;
            }
        }
    }

    // `input` was UTF-8 and repairs only replace bytes with ASCII sequences.
    String::from_utf8(repaired).expect("JSON repair must preserve UTF-8")
}

fn is_valid_json_escape(byte: u8) -> bool {
    matches!(
        byte,
        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
    )
}

fn push_control_escape(output: &mut Vec<u8>, byte: u8) {
    match byte {
        b'\x08' => output.extend_from_slice(br"\b"),
        b'\x0c' => output.extend_from_slice(br"\f"),
        b'\n' => output.extend_from_slice(br"\n"),
        b'\r' => output.extend_from_slice(br"\r"),
        b'\t' => output.extend_from_slice(br"\t"),
        _ => {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            output.extend_from_slice(br"\u00");
            output.push(HEX[(byte >> 4) as usize]);
            output.push(HEX[(byte & 0x0f) as usize]);
        }
    }
}

/// Parses strict JSON first, retrying after [`repair_json`] when useful.
pub fn parse_json_with_repair(input: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str(input) {
        Ok(value) => Ok(value),
        Err(original_error) => {
            let repaired = repair_json(input);
            if repaired != input
                && let Ok(value) = serde_json::from_str(&repaired)
            {
                return Ok(value);
            }
            Err(original_error)
        }
    }
}

/// Parses an incomplete JSON value, completing strings and containers where
/// possible. It is intended for a live preview, not validation of final input.
pub fn parse_partial_json(input: &str) -> Result<Value, PartialJsonError> {
    let mut parser = PartialJsonParser { input, position: 0 };
    parser.skip_whitespace();
    if parser.at_end() {
        return Err(PartialJsonError::Incomplete);
    }
    parser.parse_value()
}

struct PartialJsonParser<'a> {
    input: &'a str,
    position: usize,
}

impl PartialJsonParser<'_> {
    fn at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    fn byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.position).copied()
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.byte(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.position += 1;
        }
    }

    fn parse_value(&mut self) -> Result<Value, PartialJsonError> {
        self.skip_whitespace();
        match self.byte() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(Value::String),
            Some(b't') => self.parse_literal("true", Value::Bool(true)),
            Some(b'f') => self.parse_literal("false", Value::Bool(false)),
            Some(b'n') => self.parse_literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            _ => Err(PartialJsonError::Incomplete),
        }
    }

    fn parse_literal(&mut self, literal: &str, value: Value) -> Result<Value, PartialJsonError> {
        let end = self.position.saturating_add(literal.len());
        if self.input.get(self.position..end) == Some(literal) {
            self.position = end;
            Ok(value)
        } else {
            Err(PartialJsonError::Incomplete)
        }
    }

    fn parse_string(&mut self) -> Result<String, PartialJsonError> {
        debug_assert_eq!(self.byte(), Some(b'"'));
        self.position += 1;
        let mut value = String::new();

        while let Some(byte) = self.byte() {
            match byte {
                b'"' => {
                    self.position += 1;
                    return Ok(value);
                }
                b'\\' => {
                    if self.position + 1 >= self.input.len() {
                        return Ok(value);
                    }
                    let next = self.input.as_bytes()[self.position + 1];
                    if next == b'u' {
                        match decode_unicode_escape(self.input, self.position) {
                            UnicodeEscape::Character(character, consumed) => {
                                value.push(character);
                                self.position += consumed;
                            }
                            UnicodeEscape::Incomplete => return Ok(value),
                            UnicodeEscape::Invalid => {
                                // Preserve malformed escapes verbatim. Moving
                                // one byte is safe because it is ASCII.
                                value.push('\\');
                                self.position += 1;
                            }
                        }
                    } else if let Some(decoded) = decode_simple_escape(next) {
                        value.push(decoded);
                        self.position += 2;
                    } else {
                        value.push('\\');
                        self.position += 1;
                    }
                }
                _ => {
                    let character = self.input[self.position..]
                        .chars()
                        .next()
                        .expect("position is at a valid UTF-8 boundary");
                    value.push(character);
                    self.position += character.len_utf8();
                }
            }
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Value, PartialJsonError> {
        let start = self.position;
        while matches!(
            self.byte(),
            Some(b'-' | b'+' | b'.' | b'0'..=b'9' | b'e' | b'E')
        ) {
            self.position += 1;
        }
        let mut end = self.position;
        while end > start {
            let token = &self.input[start..end];
            if let Ok(Value::Number(number)) = serde_json::from_str::<Value>(token) {
                return Ok(Value::Number(number));
            }
            end -= 1;
        }
        Err(PartialJsonError::Incomplete)
    }

    fn parse_object(&mut self) -> Result<Value, PartialJsonError> {
        debug_assert_eq!(self.byte(), Some(b'{'));
        self.position += 1;
        let mut object = Map::new();

        loop {
            self.skip_whitespace();
            if self.at_end() {
                return Ok(Value::Object(object));
            }
            if self.byte() == Some(b'}') {
                self.position += 1;
                return Ok(Value::Object(object));
            }
            if self.byte() == Some(b',') {
                self.position += 1;
                continue;
            }
            if self.byte() != Some(b'"') {
                return Ok(Value::Object(object));
            }
            let key = match self.parse_string() {
                Ok(key) => key,
                Err(_) => return Ok(Value::Object(object)),
            };
            // A key reaching EOF cannot have a value and is dropped.
            if self.at_end() {
                return Ok(Value::Object(object));
            }
            self.skip_whitespace();
            if self.at_end() || self.byte() != Some(b':') {
                return Ok(Value::Object(object));
            }
            self.position += 1;
            let value = match self.parse_value() {
                Ok(value) => value,
                Err(_) => return Ok(Value::Object(object)),
            };
            object.insert(key, value);

            self.skip_whitespace();
            if self.at_end() {
                return Ok(Value::Object(object));
            }
            match self.byte() {
                Some(b',') => self.position += 1,
                Some(b'}') => {
                    self.position += 1;
                    return Ok(Value::Object(object));
                }
                _ => return Ok(Value::Object(object)),
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, PartialJsonError> {
        debug_assert_eq!(self.byte(), Some(b'['));
        self.position += 1;
        let mut values = Vec::new();

        loop {
            self.skip_whitespace();
            if self.at_end() {
                return Ok(Value::Array(values));
            }
            if self.byte() == Some(b']') {
                self.position += 1;
                return Ok(Value::Array(values));
            }
            if self.byte() == Some(b',') {
                self.position += 1;
                continue;
            }
            let value = match self.parse_value() {
                Ok(value) => value,
                Err(_) => return Ok(Value::Array(values)),
            };
            values.push(value);

            self.skip_whitespace();
            if self.at_end() {
                return Ok(Value::Array(values));
            }
            match self.byte() {
                Some(b',') => self.position += 1,
                Some(b']') => {
                    self.position += 1;
                    return Ok(Value::Array(values));
                }
                _ => return Ok(Value::Array(values)),
            }
        }
    }
}

enum UnicodeEscape {
    Character(char, usize),
    Incomplete,
    Invalid,
}

fn decode_simple_escape(byte: u8) -> Option<char> {
    match byte {
        b'"' => Some('"'),
        b'\\' => Some('\\'),
        b'/' => Some('/'),
        b'b' => Some('\u{0008}'),
        b'f' => Some('\u{000c}'),
        b'n' => Some('\n'),
        b'r' => Some('\r'),
        b't' => Some('\t'),
        _ => None,
    }
}

fn decode_unicode_escape(input: &str, position: usize) -> UnicodeEscape {
    let bytes = input.as_bytes();
    if position.saturating_add(6) > bytes.len() {
        return UnicodeEscape::Incomplete;
    }
    let Some(first) = parse_hex4(bytes, position) else {
        return UnicodeEscape::Invalid;
    };
    if !(0xd800..=0xdfff).contains(&first) {
        return UnicodeEscape::Character(
            char::from_u32(u32::from(first)).expect("non-surrogate u16 is a valid scalar"),
            6,
        );
    }
    // A surrogate must wait for its complete second escape so the live preview
    // never emits a replacement character that a later delta must retract.
    if position.saturating_add(12) > bytes.len() {
        return UnicodeEscape::Incomplete;
    }
    let Some(second) = parse_hex4(bytes, position + 6) else {
        return UnicodeEscape::Character('\u{fffd}', 6);
    };
    if (0xd800..=0xdbff).contains(&first) && (0xdc00..=0xdfff).contains(&second) {
        let high = u32::from(first) - 0xd800;
        let low = u32::from(second) - 0xdc00;
        let scalar = 0x1_0000 + ((high << 10) | low);
        return UnicodeEscape::Character(
            char::from_u32(scalar).expect("valid surrogate pair is a scalar"),
            12,
        );
    }
    UnicodeEscape::Character('\u{fffd}', 6)
}

fn parse_hex4(bytes: &[u8], position: usize) -> Option<u16> {
    let end = position.checked_add(6)?;
    let escape = bytes.get(position..end)?;
    if escape.first() != Some(&b'\\') || escape.get(1) != Some(&b'u') {
        return None;
    }
    let mut value = 0u16;
    for byte in &escape[2..] {
        let digit = match byte {
            b'0'..=b'9' => u16::from(byte - b'0'),
            b'a'..=b'f' => u16::from(byte - b'a') + 10,
            b'A'..=b'F' => u16::from(byte - b'A') + 10,
            _ => return None,
        };
        value = (value << 4) | digit;
    }
    Some(value)
}

/// Parses tool-call arguments while they are still arriving. It always returns
/// an object because tool arguments are object-shaped; malformed and
/// non-object input produces an empty map.
pub fn parse_streaming_json(input: &str) -> Map<String, Value> {
    if input.trim().is_empty() {
        return Map::new();
    }
    if let Ok(value) = parse_json_with_repair(input) {
        return value.as_object().cloned().unwrap_or_default();
    }
    if let Ok(value) = parse_partial_json(input) {
        return value.as_object().cloned().unwrap_or_default();
    }
    if let Ok(value) = parse_partial_json(&repair_json(input)) {
        return value.as_object().cloned().unwrap_or_default();
    }
    Map::new()
}

/// Tool arguments in the shape used by [`llm::ToolCall`].
pub type ToolArguments = BTreeMap<String, Value>;

/// Parses streaming tool arguments directly into the map type used by the
/// existing LLM message model.
pub fn parse_streaming_tool_arguments(input: &str) -> ToolArguments {
    parse_streaming_json(input).into_iter().collect()
}

/// The minimum growth that warrants reparsing an incremental tool-argument
/// buffer. The geometric term in [`should_reparse_streaming_json`] prevents
/// quadratic work for large write-tool arguments.
pub const STREAMING_JSON_PARSE_FLOOR: usize = 8;

pub fn should_reparse_streaming_json(parsed_len: usize, current_len: usize) -> bool {
    let fixed_next = parsed_len.saturating_add(STREAMING_JSON_PARSE_FLOOR);
    let geometric_next = parsed_len.saturating_add(parsed_len / 4);
    current_len >= fixed_next.max(geometric_next)
}

/// Incrementally collects JSON tool arguments with bounded reparse frequency.
///
/// [`Self::finish`] must be called before executing the tool so the final
/// argument map is parsed authoritatively, even when the last delta did not
/// reach the preview throttle.
#[derive(Clone, Debug, Default)]
pub struct IncrementalJsonObjectParser {
    raw: String,
    preview: Map<String, Value>,
    parsed_bytes: usize,
}

impl IncrementalJsonObjectParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one provider delta. Returns true when the preview was reparsed.
    pub fn push(&mut self, delta: &str) -> bool {
        self.raw.push_str(delta);
        if should_reparse_streaming_json(self.parsed_bytes, self.raw.len()) {
            self.reparse();
            true
        } else {
            false
        }
    }

    pub fn raw(&self) -> &str {
        &self.raw
    }

    pub fn preview(&self) -> &Map<String, Value> {
        &self.preview
    }

    /// Returns the current preview in the map type expected by
    /// [`llm::ToolCall::arguments`].
    pub fn tool_arguments(&self) -> ToolArguments {
        self.preview.clone().into_iter().collect()
    }

    pub fn parsed_bytes(&self) -> usize {
        self.parsed_bytes
    }

    /// Forces a live-preview refresh.
    pub fn reparse(&mut self) -> &Map<String, Value> {
        self.preview = parse_streaming_json(&self.raw);
        self.parsed_bytes = self.raw.len();
        &self.preview
    }

    /// Forces one final parse and returns the exact final object.
    pub fn finish(&mut self) -> &Map<String, Value> {
        self.reparse()
    }

    /// Forces a final parse and returns arguments ready to assign to a
    /// [`llm::ToolCall`].
    pub fn finish_tool_arguments(&mut self) -> ToolArguments {
        self.finish();
        self.tool_arguments()
    }

    pub fn into_object(mut self) -> Map<String, Value> {
        self.reparse();
        self.preview
    }
}

/// Integration-oriented name for [`IncrementalJsonObjectParser`].
pub type IncrementalToolArguments = IncrementalJsonObjectParser;

/// A provider-level failure detached from any concrete HTTP client.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProviderError {
    /// HTTP status; `0` denotes a network failure before a response arrived.
    pub status: u16,
    /// Response headers. Header lookup is ASCII case-insensitive.
    pub headers: BTreeMap<String, String>,
    /// A bounded raw response body, if an adapter retained one.
    pub body: String,
    pub message: String,
}

impl ProviderError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            ..Self::default()
        }
    }

    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.message.is_empty() {
            formatter.write_str(&self.message)
        } else {
            write!(formatter, "provider error (status {})", self.status)
        }
    }
}

impl Error for ProviderError {}

/// Returns whether a provider failure follows the common retry policy used by
/// OpenAI- and Anthropic-compatible APIs.
pub fn is_retryable_provider_error(error: &ProviderError) -> bool {
    if let Some(value) = error.header("x-should-retry") {
        if value.eq_ignore_ascii_case("true") {
            return true;
        }
        if value.eq_ignore_ascii_case("false") {
            return false;
        }
    }
    error.status == 0 || matches!(error.status, 408 | 409 | 429) || error.status >= 500
}

/// Marker error for a request that was explicitly cancelled by its owner.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RequestAborted;

impl fmt::Display for RequestAborted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("request aborted")
    }
}

impl Error for RequestAborted {}

/// Traverses an error source chain looking for [`RequestAborted`].
pub fn is_abort_error(error: &(dyn Error + 'static)) -> bool {
    let mut current = error;
    loop {
        if current.downcast_ref::<RequestAborted>().is_some() {
            return true;
        }
        match current.source() {
            Some(source) => current = source,
            None => return false,
        }
    }
}

/// Maximum server-directed retry delay used by [`RetryDelayLimit::Default`].
pub const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

/// Controls the allowed delay from Retry-After headers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RetryDelayLimit {
    /// Reject server delays longer than [`DEFAULT_MAX_RETRY_DELAY`].
    #[default]
    Default,
    /// Do not impose a local cap.
    Unlimited,
    /// Reject server delays longer than this duration.
    Maximum(Duration),
}

impl RetryDelayLimit {
    fn maximum(self) -> Option<Duration> {
        match self {
            Self::Default => Some(DEFAULT_MAX_RETRY_DELAY),
            Self::Unlimited => None,
            Self::Maximum(duration) => Some(duration),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RetryDelayError {
    ServerDelayExceedsLimit {
        requested: Duration,
        maximum: Duration,
        provider_message: String,
    },
}

impl fmt::Display for RetryDelayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ServerDelayExceedsLimit {
                requested,
                maximum,
                provider_message,
            } => write!(
                formatter,
                "server requested a {}s retry delay (maximum: {}s). {provider_message}",
                rounded_seconds(*requested),
                rounded_seconds(*maximum),
            ),
        }
    }
}

impl Error for RetryDelayError {}

fn rounded_seconds(duration: Duration) -> u64 {
    duration
        .as_secs()
        .saturating_add(u64::from(duration.subsec_nanos() != 0))
}

/// Parses the `Retry-After-Ms` extension. Negative values are treated as an
/// immediate retry, matching common provider SDK behavior.
pub fn parse_retry_after_ms(value: &str) -> Option<Duration> {
    duration_from_milliseconds(value.trim().parse::<f64>().ok()?)
}

/// Parses a standard `Retry-After` value: decimal seconds or an HTTP date.
pub fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    if let Ok(seconds) = value.trim().parse::<f64>()
        && let Some(delay) = duration_from_milliseconds(seconds * 1_000.0)
    {
        return Some(delay);
    }
    parse_http_date(value.trim())
        .map(|retry_at| retry_at.duration_since(now).unwrap_or(Duration::ZERO))
}

/// Validates a server-requested delay against the configured local cap.
pub fn validate_server_retry_delay(
    delay: Duration,
    limit: RetryDelayLimit,
    provider_message: impl Into<String>,
) -> Result<Duration, RetryDelayError> {
    if let Some(maximum) = limit.maximum()
        && delay > maximum
    {
        return Err(RetryDelayError::ServerDelayExceedsLimit {
            requested: delay,
            maximum,
            provider_message: provider_message.into(),
        });
    }
    Ok(delay)
}

/// Calculates a capped exponential retry delay: 0.5, 1, 2, 4, then 8 seconds.
///
/// `jitter_fraction` is clamped to `[0, 1]` and removes up to 25% of the
/// delay. Supplying `0.0` is deterministic, which is useful for tests.
pub fn exponential_retry_delay(retry_index: u32, jitter_fraction: f64) -> Duration {
    let base_millis = match retry_index {
        0 => 500,
        1 => 1_000,
        2 => 2_000,
        3 => 4_000,
        _ => 8_000,
    };
    let jitter = if jitter_fraction.is_finite() {
        jitter_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let millis = (base_millis as f64 * (1.0 - 0.25 * jitter)).round() as u64;
    Duration::from_millis(millis)
}

/// Returns the retry delay for a provider error. Server-provided headers take
/// precedence over exponential backoff.
pub fn retry_delay(
    error: &ProviderError,
    retry_index: u32,
    now: SystemTime,
    limit: RetryDelayLimit,
) -> Result<Duration, RetryDelayError> {
    retry_delay_with_jitter(error, retry_index, now, limit, 0.0)
}

/// Like [`retry_delay`], with caller-provided jitter so a transport can inject
/// its own entropy without this module depending on an RNG crate.
pub fn retry_delay_with_jitter(
    error: &ProviderError,
    retry_index: u32,
    now: SystemTime,
    limit: RetryDelayLimit,
    jitter_fraction: f64,
) -> Result<Duration, RetryDelayError> {
    if let Some(value) = error.header("retry-after-ms")
        && let Some(delay) = parse_retry_after_ms(value)
    {
        return validate_server_retry_delay(delay, limit, error.to_string());
    }
    if let Some(value) = error.header("retry-after")
        && let Some(delay) = parse_retry_after(value, now)
    {
        return validate_server_retry_delay(delay, limit, error.to_string());
    }
    Ok(exponential_retry_delay(retry_index, jitter_fraction))
}

fn duration_from_milliseconds(milliseconds: f64) -> Option<Duration> {
    if !milliseconds.is_finite() {
        return None;
    }
    if milliseconds <= 0.0 {
        return Some(Duration::ZERO);
    }
    let seconds = milliseconds / 1_000.0;
    // Be conservative around the f64 representation of u64::MAX.
    if seconds >= u64::MAX as f64 {
        return None;
    }
    let whole_seconds = seconds.trunc() as u64;
    let nanos = ((seconds.fract() * 1_000_000_000.0).round()) as u64;
    if nanos >= 1_000_000_000 {
        return whole_seconds.checked_add(1).map(Duration::from_secs);
    }
    Some(Duration::new(whole_seconds, nanos as u32))
}

fn parse_http_date(value: &str) -> Option<SystemTime> {
    let parts: Vec<_> = value.split_ascii_whitespace().collect();
    match parts.as_slice() {
        [weekday, day, month, year, time, zone]
            if weekday.ends_with(',') && zone.eq_ignore_ascii_case("gmt") =>
        {
            system_time_from_date(
                year.parse().ok()?,
                month_number(month)?,
                day.parse().ok()?,
                parse_hms(time)?,
            )
        }
        [weekday, date, time, zone]
            if weekday.ends_with(',') && zone.eq_ignore_ascii_case("gmt") =>
        {
            let date_parts: Vec<_> = date.split('-').collect();
            let [day, month, short_year] = date_parts.as_slice() else {
                return None;
            };
            let short_year: i32 = short_year.parse().ok()?;
            if !(0..=99).contains(&short_year) {
                return None;
            }
            let year = if short_year <= 69 {
                2000 + short_year
            } else {
                1900 + short_year
            };
            system_time_from_date(
                year,
                month_number(month)?,
                day.parse().ok()?,
                parse_hms(time)?,
            )
        }
        [_weekday, month, day, time, year] => system_time_from_date(
            year.parse().ok()?,
            month_number(month)?,
            day.parse().ok()?,
            parse_hms(time)?,
        ),
        _ => None,
    }
}

fn month_number(value: &str) -> Option<u32> {
    if value.eq_ignore_ascii_case("jan") {
        Some(1)
    } else if value.eq_ignore_ascii_case("feb") {
        Some(2)
    } else if value.eq_ignore_ascii_case("mar") {
        Some(3)
    } else if value.eq_ignore_ascii_case("apr") {
        Some(4)
    } else if value.eq_ignore_ascii_case("may") {
        Some(5)
    } else if value.eq_ignore_ascii_case("jun") {
        Some(6)
    } else if value.eq_ignore_ascii_case("jul") {
        Some(7)
    } else if value.eq_ignore_ascii_case("aug") {
        Some(8)
    } else if value.eq_ignore_ascii_case("sep") {
        Some(9)
    } else if value.eq_ignore_ascii_case("oct") {
        Some(10)
    } else if value.eq_ignore_ascii_case("nov") {
        Some(11)
    } else if value.eq_ignore_ascii_case("dec") {
        Some(12)
    } else {
        None
    }
}

fn parse_hms(value: &str) -> Option<(u32, u32, u32)> {
    let parts: Vec<_> = value.split(':').collect();
    let [hour, minute, second] = parts.as_slice() else {
        return None;
    };
    let hour = hour.parse().ok()?;
    let minute = minute.parse().ok()?;
    let second = second.parse().ok()?;
    (hour < 24 && minute < 60 && second < 60).then_some((hour, minute, second))
}

fn system_time_from_date(
    year: i32,
    month: u32,
    day: u32,
    (hour, minute, second): (u32, u32, u32),
) -> Option<SystemTime> {
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    let days = days_from_civil(year, month, day);
    let seconds = days
        .checked_mul(86_400)?
        .checked_add(i64::from(hour) * 3_600 + i64::from(minute) * 60 + i64::from(second))?;
    if seconds >= 0 {
        UNIX_EPOCH.checked_add(Duration::from_secs(seconds as u64))
    } else {
        UNIX_EPOCH.checked_sub(Duration::from_secs(seconds.unsigned_abs()))
    }
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

// Howard Hinnant's civil-date conversion, returning days from 1970-01-01.
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let year = year - i32::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let march_month = month as i32 + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * march_month + 2) / 5 + day as i32 - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    i64::from(era) * 146_097 + i64::from(day_of_era) - 719_468
}

const NON_RETRYABLE_PROVIDER_LIMIT_PATTERNS: &[&str] = &[
    "gousagelimiterror",
    "freeusagelimiterror",
    "monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
];

const RETRYABLE_ASSISTANT_PATTERNS: &[&str] = &[
    "overloaded",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "524",
    "service unavailable",
    "server error",
    "internal error",
    "provider returned error",
    "exceeded request buffer limit while retrying upstream",
    "network error",
    "connection error",
    "connection refused",
    "connection lost",
    "other side closed",
    "fetch failed",
    "getaddrinfo",
    "enotfound",
    "eai_again",
    "upstream connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed out",
    "timeout",
    "terminated",
    "websocket closed",
    "websocket error",
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    "retry delay",
    "you can retry your request",
    "try your request again",
    "please retry your request",
    "resourceexhausted",
];

/// Returns whether an assistant error looks transient enough for a new attempt.
pub fn is_retryable_assistant_error(message: &llm::AssistantMessage) -> bool {
    if message.stop_reason != STOP_ERROR || message.error_message.is_empty() {
        return false;
    }
    let lower = message.error_message.to_ascii_lowercase();
    if NON_RETRYABLE_PROVIDER_LIMIT_PATTERNS
        .iter()
        .any(|pattern| lower.contains(pattern))
    {
        return false;
    }
    has_rate_limit_marker(&lower)
        || RETRYABLE_ASSISTANT_PATTERNS
            .iter()
            .any(|pattern| lower.contains(pattern))
}

fn has_rate_limit_marker(message: &str) -> bool {
    message.contains("rate limit")
        || message.contains("rate-limit")
        || message.contains("rate_limit")
        || message.contains("ratelimit")
}

/// Configuration for retrying a protocol-independent assistant operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
    pub enabled: bool,
    /// Number of retries after the initial attempt.
    pub max_retries: u32,
    pub base_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            max_retries: 0,
            base_delay: Duration::from_millis(500),
        }
    }
}

impl RetryPolicy {
    pub fn delay_for_retry(&self, retry_index: u32) -> Duration {
        let mut delay = self.base_delay;
        // A duration saturates long before this cap; limiting iterations keeps
        // hostile configuration values from creating a long CPU loop.
        for _ in 0..retry_index.min(64) {
            delay = delay.checked_mul(2).unwrap_or(Duration::MAX);
        }
        delay
    }
}

/// Runs an operation once plus at most `max_retries` times. The caller injects
/// the wait operation, keeping this helper usable with async runtimes, tests,
/// or a cancellation-aware scheduler.
pub fn retry_with_backoff<T, F, C, W>(
    mut operation: F,
    mut should_retry: C,
    policy: RetryPolicy,
    mut wait: W,
) -> T
where
    F: FnMut() -> T,
    C: FnMut(&T) -> bool,
    W: FnMut(Duration),
{
    let mut result = operation();
    if !policy.enabled {
        return result;
    }
    for retry_index in 0..policy.max_retries {
        if !should_retry(&result) {
            break;
        }
        wait(policy.delay_for_retry(retry_index));
        result = operation();
    }
    result
}

/// The coarse text-to-token ratio used when a provider does not report usage.
pub const CHARS_PER_TOKEN: u64 = 4;
pub const ESTIMATED_IMAGE_CHARS: u64 = 4_800;
pub const CONTEXT_SAFETY_TOKENS: u64 = 4_096;
pub const MIN_MAX_TOKENS: u64 = 1;
/// Tokens reserved for an answer when thinking shares a response budget.
pub const MIN_ANSWER_TOKENS: u64 = 1_024;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContextUsageEstimate {
    pub tokens: u64,
    pub usage_tokens: u64,
    pub trailing_tokens: u64,
    pub last_usage_index: Option<usize>,
}

/// Uses a provider's total context count when available, otherwise sums the
/// token count fields present in pi-compatible usage metadata.
pub fn calculate_context_tokens(usage: &llm::Usage) -> u64 {
    if usage.total_tokens != 0 {
        usage.total_tokens
    } else {
        usage
            .input
            .saturating_add(usage.output)
            .saturating_add(usage.cache_read)
            .saturating_add(usage.cache_write)
    }
}

/// Estimates plain-text tokens from UTF-8 bytes, matching the existing Go
/// approximation and avoiding locale-dependent behavior.
pub fn estimate_text_tokens(text: &str) -> u64 {
    let bytes = text.len() as u64;
    bytes.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

/// Estimates one pi-compatible transcript message.
pub fn estimate_message_tokens(message: &llm::Message) -> u64 {
    match message {
        llm::Message::User(user) => match &user.content {
            llm::UserContent::Text(text) => estimate_text_tokens(text),
            llm::UserContent::Blocks(blocks) => estimate_text_and_image_tokens(blocks),
        },
        llm::Message::ToolResult(result) => estimate_text_and_image_tokens(&result.content),
        llm::Message::Assistant(assistant) => estimate_assistant_tokens(assistant),
    }
}

fn estimate_text_and_image_tokens(blocks: &[llm::ContentBlock]) -> u64 {
    let bytes = blocks.iter().fold(0u64, |total, block| match block {
        llm::ContentBlock::Text(text) => total.saturating_add(text.text.len() as u64),
        llm::ContentBlock::Image(_) => total.saturating_add(ESTIMATED_IMAGE_CHARS),
        llm::ContentBlock::Thinking(_) | llm::ContentBlock::ToolCall(_) => total,
    });
    bytes.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

fn estimate_assistant_tokens(message: &llm::AssistantMessage) -> u64 {
    let bytes = message
        .content
        .iter()
        .fold(0u64, |total, block| match block {
            llm::ContentBlock::Text(text) => total.saturating_add(text.text.len() as u64),
            llm::ContentBlock::Thinking(thinking) => {
                total.saturating_add(thinking.thinking.len() as u64)
            }
            llm::ContentBlock::ToolCall(call) => total
                .saturating_add(call.name.len() as u64)
                .saturating_add(serialized_token_bytes(&call.arguments)),
            llm::ContentBlock::Image(_) => total,
        });
    bytes.saturating_add(CHARS_PER_TOKEN - 1) / CHARS_PER_TOKEN
}

fn serialized_token_bytes<T: Serialize>(value: &T) -> u64 {
    serde_json::to_string(value)
        .unwrap_or_else(|_| "[unserializable]".to_owned())
        .len() as u64
}

fn last_assistant_usage(messages: &[llm::Message]) -> Option<(&llm::Usage, usize)> {
    let mut latest_prefix_timestamp = i64::MIN;
    let mut found = None;
    for (index, message) in messages.iter().enumerate() {
        if let llm::Message::Assistant(assistant) = message
            && assistant.timestamp >= latest_prefix_timestamp
            && assistant.stop_reason != STOP_ABORTED
            && assistant.stop_reason != STOP_ERROR
            && calculate_context_tokens(&assistant.usage) > 0
        {
            found = Some((&assistant.usage, index));
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(message_timestamp(message));
    }
    found
}

fn message_timestamp(message: &llm::Message) -> i64 {
    match message {
        llm::Message::User(message) => message.timestamp,
        llm::Message::Assistant(message) => message.timestamp,
        llm::Message::ToolResult(message) => message.timestamp,
    }
}

fn estimate_serialized_tokens<T: Serialize + ?Sized>(tools: &T) -> u64 {
    let encoded = serde_json::to_string(tools).unwrap_or_else(|_| "[unserializable]".to_owned());
    estimate_text_tokens(&encoded)
}

fn estimate_tools_tokens(tools: &[llm::Tool]) -> u64 {
    if tools.is_empty() {
        0
    } else {
        estimate_serialized_tokens(tools)
    }
}

/// Estimates the context size, preferring the most recent valid provider usage
/// report and estimating only messages/tools added after it.
pub fn estimate_context_tokens(context: &llm::Context) -> ContextUsageEstimate {
    if let Some((usage, usage_index)) = last_assistant_usage(&context.messages) {
        let usage_tokens = calculate_context_tokens(usage);
        let mut trailing_tokens = context.messages[usage_index + 1..]
            .iter()
            .fold(0u64, |total, message| {
                total.saturating_add(estimate_message_tokens(message))
            });

        let mut added_tool_names = BTreeSet::new();
        for message in &context.messages[usage_index + 1..] {
            if let llm::Message::ToolResult(result) = message {
                added_tool_names.extend(result.added_tool_names.iter().map(String::as_str));
            }
        }
        let added_tools: Vec<_> = context
            .tools
            .iter()
            .filter(|tool| added_tool_names.contains(tool.name.as_str()))
            .collect();
        if !added_tools.is_empty() {
            trailing_tokens =
                trailing_tokens.saturating_add(estimate_serialized_tokens(&added_tools));
        }

        return ContextUsageEstimate {
            tokens: usage_tokens.saturating_add(trailing_tokens),
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(usage_index),
        };
    }

    let message_tokens = context.messages.iter().fold(0u64, |total, message| {
        total.saturating_add(estimate_message_tokens(message))
    });
    let mut prefix_tokens = estimate_tools_tokens(&context.tools);
    if !context.system_prompt.is_empty() {
        prefix_tokens = prefix_tokens.saturating_add(estimate_text_tokens(&context.system_prompt));
    }
    ContextUsageEstimate {
        tokens: message_tokens.saturating_add(prefix_tokens),
        usage_tokens: 0,
        trailing_tokens: message_tokens.saturating_add(prefix_tokens),
        last_usage_index: None,
    }
}

/// Clamps a requested response limit to the model context window after leaving
/// a fixed safety margin. A zero request remains zero when a context window is
/// known, allowing callers to distinguish "unspecified" from an explicit
/// minimum; callers that need defaults should resolve `model.max_tokens` first.
pub fn clamp_max_tokens_to_context(
    model: &llm::Model,
    context: &llm::Context,
    requested_max_tokens: u64,
) -> u64 {
    if model.context_window == 0 {
        return requested_max_tokens.max(MIN_MAX_TOKENS);
    }
    let available = model
        .context_window
        .saturating_sub(estimate_context_tokens(context).tokens)
        .saturating_sub(CONTEXT_SAFETY_TOKENS)
        .max(MIN_MAX_TOKENS);
    requested_max_tokens.min(available)
}

/// Selects and writes the model cost corresponding to `usage`.
pub fn calculate_usage_cost(model: &llm::Model, usage: &mut llm::Usage) -> llm::UsageCost {
    let cost = usage_cost(model, usage);
    usage.cost = cost.clone();
    cost
}

/// Calculates usage cost without mutating the supplied usage record.
pub fn usage_cost(model: &llm::Model, usage: &llm::Usage) -> llm::UsageCost {
    let billable_input = usage
        .input
        .saturating_add(usage.cache_read)
        .saturating_add(usage.cache_write);
    let mut rates = &model.cost.rates;
    let mut matched_threshold = None;
    for tier in &model.cost.tiers {
        if billable_input > tier.input_tokens_above
            && matched_threshold.is_none_or(|threshold| tier.input_tokens_above > threshold)
        {
            rates = &tier.rates;
            matched_threshold = Some(tier.input_tokens_above);
        }
    }

    let long_cache_write = usage.cache_write_1h.unwrap_or(0).min(usage.cache_write);
    let short_cache_write = usage.cache_write - long_cache_write;
    let input = rates.input / 1_000_000.0 * usage.input as f64;
    let output = rates.output / 1_000_000.0 * usage.output as f64;
    let cache_read = rates.cache_read / 1_000_000.0 * usage.cache_read as f64;
    let cache_write = (rates.cache_write * short_cache_write as f64
        + rates.input * 2.0 * long_cache_write as f64)
        / 1_000_000.0;
    llm::UsageCost {
        input,
        output,
        cache_read,
        cache_write,
        total: input + output + cache_read + cache_write,
    }
}

pub const EXTENDED_THINKING_LEVELS: [&str; 7] = [
    llm::THINKING_OFF,
    llm::THINKING_MINIMAL,
    llm::THINKING_LOW,
    llm::THINKING_MEDIUM,
    llm::THINKING_HIGH,
    llm::THINKING_XHIGH,
    llm::THINKING_MAX,
];

/// Returns all accepted thinking levels from weakest to strongest.
pub fn thinking_levels() -> Vec<String> {
    EXTENDED_THINKING_LEVELS
        .iter()
        .map(|level| (*level).to_owned())
        .collect()
}

/// Returns the thinking levels actually accepted by a model.
pub fn supported_thinking_levels(model: &llm::Model) -> Vec<String> {
    if !model.reasoning {
        return vec![llm::THINKING_OFF.to_owned()];
    }
    EXTENDED_THINKING_LEVELS
        .iter()
        .filter(|entry| {
            let level = **entry;
            let mapping = model.thinking_level_map.get(level);
            if mapping.is_some_and(Option::is_none) {
                return false;
            }
            !matches!(level, llm::THINKING_XHIGH | llm::THINKING_MAX) || mapping.is_some()
        })
        .map(|level| (*level).to_owned())
        .collect()
}

pub fn supports_thinking_level(model: &llm::Model, level: &str) -> bool {
    supported_thinking_levels(model)
        .iter()
        .any(|available| available == level)
}

/// Returns a supported level nearest to `requested`, preferring stronger
/// levels before weaker ones when the exact level is unavailable.
pub fn clamp_thinking_level(model: &llm::Model, requested: &str) -> String {
    let available = supported_thinking_levels(model);
    if available.iter().any(|level| level == requested) {
        return requested.to_owned();
    }
    let fallback = available
        .first()
        .cloned()
        .unwrap_or_else(|| llm::THINKING_OFF.to_owned());
    let Some(requested_index) = EXTENDED_THINKING_LEVELS
        .iter()
        .position(|level| *level == requested)
    else {
        return fallback;
    };

    for level in EXTENDED_THINKING_LEVELS.iter().skip(requested_index) {
        if available.iter().any(|available| available == level) {
            return (*level).to_owned();
        }
    }
    for level in EXTENDED_THINKING_LEVELS[..requested_index].iter().rev() {
        if available.iter().any(|available| available == level) {
            return (*level).to_owned();
        }
    }
    fallback
}

/// Maps levels unsupported by token-budget APIs down to `high`.
pub fn clamp_reasoning_level(level: &str) -> String {
    match level {
        llm::THINKING_XHIGH | llm::THINKING_MAX => llm::THINKING_HIGH.to_owned(),
        _ => level.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        io::Cursor,
        sync::Arc,
        time::{Duration, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::*;

    fn collect_sse(input: &str) -> Vec<SseEvent> {
        let mut reader = SseReader::new(Cursor::new(input.as_bytes()));
        reader
            .events()
            .collect::<Result<Vec<_>, _>>()
            .expect("SSE input should parse")
    }

    fn test_model() -> llm::Model {
        llm::Model {
            id: "test".to_owned(),
            name: "Test".to_owned(),
            api: "test".to_owned(),
            provider: "test".to_owned(),
            base_url: "https://example.test".to_owned(),
            reasoning: true,
            context_window: 8_000,
            max_tokens: 4_096,
            ..llm::Model::default()
        }
    }

    #[test]
    fn stop_reason_helpers_distinguish_terminal_outcomes() {
        assert!(!is_terminal_stop_reason(STOP_PENDING));
        assert!(is_terminal_stop_reason(STOP_TOOL_USE));
        assert!(is_successful_stop_reason(STOP_DEFERRED));
        assert!(!is_successful_stop_reason(STOP_ERROR));
        assert!(is_failure_stop_reason(STOP_ABORTED));
    }

    #[test]
    fn bounded_event_stream_preserves_order_and_resolves_result() {
        let stream = AssistantMessageEventStream::with_capacity(8).expect("valid capacity");
        let partial = Arc::new(llm::AssistantMessage::default());
        let final_message = Arc::new(llm::AssistantMessage {
            stop_reason: STOP_STOP.to_owned(),
            ..llm::AssistantMessage::default()
        });

        stream
            .try_push(AssistantMessageEvent::start(Arc::clone(&partial)))
            .expect("start fits");
        stream
            .try_push(AssistantMessageEvent {
                event_type: EVENT_TEXT_DELTA.to_owned(),
                content_index: Some(0),
                delta: "hello".to_owned(),
                partial: Some(partial),
                ..AssistantMessageEvent::default()
            })
            .expect("delta fits");
        stream
            .try_push(AssistantMessageEvent::done(
                STOP_STOP,
                Arc::clone(&final_message),
            ))
            .expect("done fits");

        let event_types: Vec<_> = stream.iter().map(|event| event.event_type).collect();
        assert_eq!(event_types, vec![EVENT_START, EVENT_TEXT_DELTA, EVENT_DONE]);
        assert_eq!(stream.result().expect("terminal result"), final_message);
        assert!(stream.next().is_none());
    }

    #[test]
    fn bounded_event_stream_applies_backpressure_and_rejects_late_events() {
        let stream = AssistantMessageEventStream::with_capacity(1).expect("valid capacity");
        stream
            .try_push(AssistantMessageEvent::new(EVENT_START))
            .expect("first event fits");
        let full = stream
            .try_push(AssistantMessageEvent::new(EVENT_TEXT_DELTA))
            .expect_err("second event must not grow the queue");
        assert!(matches!(full, EventStreamPushError::Full(_)));
        assert_eq!(stream.next().expect("queued event").event_type, EVENT_START);

        let final_message = Arc::new(llm::AssistantMessage {
            stop_reason: STOP_STOP.to_owned(),
            ..llm::AssistantMessage::default()
        });
        stream
            .try_push(AssistantMessageEvent::done(STOP_STOP, final_message))
            .expect("terminal event fits");
        let late = stream
            .try_push(AssistantMessageEvent::new(EVENT_TEXT_DELTA))
            .expect_err("terminal event closes producers");
        assert!(matches!(late, EventStreamPushError::Closed(_)));
    }

    #[test]
    fn event_stream_reports_end_without_a_terminal_message() {
        let stream = AssistantMessageEventStream::new();
        assert_eq!(
            stream.result_timeout(Duration::ZERO),
            Err(EventStreamResultError::TimedOut)
        );
        stream.end();
        assert_eq!(
            stream.result(),
            Err(EventStreamResultError::EndedWithoutTerminalEvent)
        );
        assert_eq!(
            stream.next_timeout(Duration::ZERO),
            Ok(None),
            "a closed stream is not a timeout"
        );
    }

    #[test]
    fn error_event_resolves_the_stream_with_its_error_message() {
        let stream = AssistantMessageEventStream::new();
        let failure = Arc::new(llm::AssistantMessage {
            stop_reason: STOP_ERROR.to_owned(),
            error_message: "provider disconnected".to_owned(),
            ..llm::AssistantMessage::default()
        });
        stream
            .try_push(AssistantMessageEvent::error(
                STOP_ERROR,
                Arc::clone(&failure),
            ))
            .expect("error event fits");

        assert_eq!(stream.next().expect("error event").event_type, EVENT_ERROR);
        assert_eq!(stream.result().expect("terminal result"), failure);
    }

    #[test]
    fn sse_parser_handles_multiline_metadata_comments_and_crlf() {
        let events = collect_sse(
            ": keepalive\r\n\
             event: message\r\n\
             id: 42\r\n\
             data: line1\r\n\
             data: line2\r\n\
             \r\n\
             data:nospace\r\n\
             \r\n\
             data: [DONE]\r\n\
             \r\n",
        );
        assert_eq!(
            events,
            vec![
                SseEvent {
                    event: "message".to_owned(),
                    id: "42".to_owned(),
                    data: "line1\nline2".to_owned(),
                },
                SseEvent {
                    data: "nospace".to_owned(),
                    ..SseEvent::default()
                },
                SseEvent {
                    data: "[DONE]".to_owned(),
                    ..SseEvent::default()
                },
            ]
        );
    }

    #[test]
    fn sse_parser_discards_unterminated_records_and_enforces_caps() {
        assert_eq!(
            collect_sse("data: complete\n\ndata: dangling"),
            vec![SseEvent {
                data: "complete".to_owned(),
                ..SseEvent::default()
            }]
        );

        let limits = SseLimits {
            max_stream_bytes: 128,
            max_line_bytes: 8,
            max_event_bytes: 8,
        };
        let mut line_reader =
            SseReader::with_limits(Cursor::new(b"data: much-too-long\n"), limits).expect("limits");
        assert!(matches!(
            line_reader.next_event(),
            Err(SseError::LineTooLarge { .. })
        ));

        let mut event_reader = SseReader::with_limits(
            Cursor::new(b"data: 12345\ndata: 67890\n\n"),
            SseLimits {
                max_line_bytes: 64,
                ..limits
            },
        )
        .expect("limits");
        assert!(matches!(
            event_reader.next_event(),
            Err(SseError::EventTooLarge { .. })
        ));

        // A colon-less `data` field is valid SSE and still contributes the
        // joining newline, so it must not bypass the event-size cap.
        let mut bare_data_reader =
            SseReader::with_limits(Cursor::new("data\n".repeat(10).into_bytes()), limits)
                .expect("limits");
        assert!(matches!(
            bare_data_reader.next_event(),
            Err(SseError::EventTooLarge { .. })
        ));
    }

    #[test]
    fn streaming_json_repairs_and_completes_critical_partial_shapes() {
        let cases = [
            (r#"{"a": "hel"#, json!({"a": "hel"})),
            (r#"{"a": 1, "b"#, json!({"a": 1})),
            (r#"{"a": [1, 2,"#, json!({"a": [1, 2]})),
            (r#"{"n": 12."#, json!({"n": 12})),
            (r#"{"ok": tr"#, json!({})),
            (r#"{"a": {"b": {"c": tr"#, json!({"a": {"b": {}}})),
        ];
        for (input, expected) in cases {
            assert_eq!(
                Value::Object(parse_streaming_json(input)),
                expected,
                "{input}"
            );
        }
        assert_eq!(
            Value::Object(parse_streaming_json("{\"command\":\"echo one\ntwo\"}")),
            json!({"command": "echo one\ntwo"})
        );
        assert_eq!(repair_json(r#"{"a": "x\y"}"#), r#"{"a": "x\\y"}"#);
    }

    #[test]
    fn partial_json_decodes_escapes_and_waits_for_surrogate_pairs() {
        let decoded =
            parse_partial_json(r#"{"command":"grep -n \"foo\" a\nb","path":"café/😀.txt"}"#)
                .expect("partial JSON");
        assert_eq!(
            decoded,
            json!({"command": "grep -n \"foo\" a\nb", "path": "café/😀.txt"})
        );

        let waiting = parse_partial_json(r#"{"text":"hi \ud83d"#).expect("partial JSON");
        assert_eq!(waiting, json!({"text": "hi "}));
        let pair = parse_partial_json(r#"{"text":"\ud83d\ude00"#).expect("partial JSON");
        assert_eq!(pair, json!({"text": "😀"}));
    }

    #[test]
    fn incremental_json_parser_is_live_for_small_inputs_and_linear_for_large_inputs() {
        let mut parser = IncrementalJsonObjectParser::new();
        for chunk in [r#"{"path":"#, r#""main.rs"#, r#""}"#] {
            parser.push(chunk);
        }
        assert_eq!(
            parser.finish_tool_arguments(),
            BTreeMap::from([("path".to_owned(), json!("main.rs"))])
        );
        assert_eq!(
            parse_streaming_tool_arguments(r#"{"line": 42"#),
            BTreeMap::from([("line".to_owned(), json!(42))])
        );

        let mut parses = 0usize;
        let mut parsed = 0usize;
        let mut work = 0usize;
        for length in 1..=54_000 {
            if should_reparse_streaming_json(parsed, length) {
                parses += 1;
                parsed = length;
                work += length;
            }
        }
        assert!((20..=80).contains(&parses), "parse count was {parses}");
        assert!(work <= 10 * 54_000, "rescanned {work} bytes");
    }

    #[test]
    fn retry_helpers_parse_headers_classify_errors_and_bound_delays() {
        let now = UNIX_EPOCH + Duration::from_secs(1_600_000_000);
        let malformed = ProviderError::new(429, "rate limited").with_header("Retry-After", "wat");
        assert_eq!(
            retry_delay(&malformed, 0, now, RetryDelayLimit::Default).expect("fallback"),
            Duration::from_millis(500)
        );
        assert_eq!(parse_retry_after_ms("-5"), Some(Duration::ZERO));
        assert_eq!(
            parse_retry_after("1.5", now),
            Some(Duration::from_millis(1_500))
        );
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT", now),
            Some(Duration::ZERO)
        );
        assert_eq!(
            parse_retry_after("Sun, 13 Sep 2020 12:26:50 GMT", now),
            Some(Duration::from_secs(10))
        );

        let delayed = ProviderError::new(429, "slow down").with_header("retry-after-ms", "61000");
        assert!(matches!(
            retry_delay(&delayed, 0, now, RetryDelayLimit::Default),
            Err(RetryDelayError::ServerDelayExceedsLimit { .. })
        ));
        assert!(is_retryable_provider_error(
            &ProviderError::new(400, "ignored").with_header("X-Should-Retry", "true")
        ));

        let retryable = llm::AssistantMessage {
            stop_reason: STOP_ERROR.to_owned(),
            error_message: "503 service unavailable".to_owned(),
            ..llm::AssistantMessage::default()
        };
        let quota = llm::AssistantMessage {
            stop_reason: STOP_ERROR.to_owned(),
            error_message: "429 insufficient_quota".to_owned(),
            ..llm::AssistantMessage::default()
        };
        assert!(is_retryable_assistant_error(&retryable));
        assert!(!is_retryable_assistant_error(&quota));
    }

    #[test]
    fn retry_with_backoff_retries_only_the_configured_number_of_times() {
        let mut attempts = 0;
        let mut delays = Vec::new();
        let result = retry_with_backoff(
            || {
                attempts += 1;
                attempts
            },
            |result| *result < 3,
            RetryPolicy {
                enabled: true,
                max_retries: 2,
                base_delay: Duration::from_millis(10),
            },
            |delay| delays.push(delay),
        );
        assert_eq!(result, 3);
        assert_eq!(
            delays,
            vec![Duration::from_millis(10), Duration::from_millis(20)]
        );
    }

    #[test]
    fn context_estimation_clamping_cost_and_thinking_helpers_match_model_rules() {
        let mut model = test_model();
        model.context_window = 5_000;
        model.thinking_level_map = BTreeMap::from([
            (llm::THINKING_LOW.to_owned(), None),
            (llm::THINKING_HIGH.to_owned(), Some("high".to_owned())),
            (llm::THINKING_XHIGH.to_owned(), Some("xhigh".to_owned())),
        ]);
        assert_eq!(
            supported_thinking_levels(&model),
            vec![
                llm::THINKING_OFF,
                llm::THINKING_MINIMAL,
                llm::THINKING_MEDIUM,
                llm::THINKING_HIGH,
                llm::THINKING_XHIGH,
            ]
        );
        assert_eq!(
            clamp_thinking_level(&model, llm::THINKING_LOW),
            llm::THINKING_MEDIUM
        );
        assert_eq!(clamp_reasoning_level(llm::THINKING_MAX), llm::THINKING_HIGH);
        assert_eq!(
            clamp_max_tokens_to_context(&model, &llm::Context::default(), 3_000),
            904
        );

        model.cost = llm::ModelCost {
            rates: llm::ModelCostRates {
                input: 1.0,
                output: 2.0,
                cache_read: 0.5,
                cache_write: 1.5,
            },
            tiers: vec![llm::ModelCostTier {
                rates: llm::ModelCostRates {
                    input: 10.0,
                    output: 20.0,
                    cache_read: 5.0,
                    cache_write: 15.0,
                },
                input_tokens_above: 50,
            }],
        };
        let mut usage = llm::Usage {
            input: 50,
            output: 10,
            cache_read: 10,
            cache_write: 10,
            cache_write_1h: Some(3),
            ..llm::Usage::default()
        };
        let cost = calculate_usage_cost(&model, &mut usage);
        assert!((cost.input - 0.0005).abs() < 1e-12);
        assert!((cost.output - 0.0002).abs() < 1e-12);
        assert!((cost.cache_read - 0.00005).abs() < 1e-12);
        assert!((cost.cache_write - 0.000165).abs() < 1e-12);
        assert_eq!(usage.cost, cost);
    }

    #[test]
    fn context_estimate_prefers_latest_valid_usage_and_counts_trailing_messages() {
        let context = llm::Context {
            messages: vec![
                llm::Message::User(llm::UserMessage::text("prior", 1)),
                llm::Message::Assistant(Box::new(llm::AssistantMessage {
                    timestamp: 2,
                    usage: llm::Usage {
                        total_tokens: 100,
                        ..llm::Usage::default()
                    },
                    ..llm::AssistantMessage::default()
                })),
                llm::Message::User(llm::UserMessage::text("more", 3)),
            ],
            ..llm::Context::default()
        };
        let estimate = estimate_context_tokens(&context);
        assert_eq!(estimate.last_usage_index, Some(1));
        assert_eq!(estimate.usage_tokens, 100);
        assert_eq!(estimate.trailing_tokens, 1);
        assert_eq!(estimate.tokens, 101);
    }
}
