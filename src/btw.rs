//! Session-independent `/btw` side-thread primitives.
//!
//! A side thread receives a read-only summary of the main conversation, keeps
//! its own turns in memory, and never mutates the main transcript.  This
//! module intentionally does not depend on the UI or [`crate::agent`], so a
//! future runtime can choose how and when to dispatch its requests.

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llm;

/// The newest portion of the main transcript made available to a side thread.
pub const MAX_CONTEXT_CHARS: usize = 40_000;
/// Upper bound for the small, user-controlled `pi-btw.json` settings file.
pub const MAX_SETTINGS_BYTES: usize = 64 << 10;
/// The filename used beneath the caller's agent configuration directory.
pub const SETTINGS_FILE_NAME: &str = "pi-btw.json";

/// System instruction used for every independent side-thread request.
pub const SYSTEM_PROMPT: &str = "You answer quick side questions for a coding-agent user.\n\
\n\
Use the provided conversation context only as background. Answer the user's side question directly and concisely. Do not claim to have changed files, run tools, or affected the main task. If the context is insufficient, say what is unknown and give the best next step.";

const THINKING_LEVELS: [&str; 7] = [
    llm::THINKING_OFF,
    llm::THINKING_MINIMAL,
    llm::THINKING_LOW,
    llm::THINKING_MEDIUM,
    llm::THINKING_HIGH,
    llm::THINKING_XHIGH,
    llm::THINKING_MAX,
];

static SETTINGS_LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A successful or failed request in an in-memory side discussion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TurnKind {
    Answered,
    Error,
}

/// One attempted side question.
///
/// `response` is populated for [`TurnKind::Answered`] turns.  Error turns
/// retain the default response only to preserve the same shape as the
/// original in-memory Go model; callers should use `answer` for display.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub kind: TurnKind,
    pub question: String,
    pub answer: String,
    pub response: llm::AssistantMessage,
}

impl Turn {
    pub fn answered(
        question: impl Into<String>,
        answer: impl Into<String>,
        response: llm::AssistantMessage,
    ) -> Self {
        Self {
            kind: TurnKind::Answered,
            question: question.into(),
            answer: answer.into(),
            response,
        }
    }

    pub fn error(question: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind: TurnKind::Error,
            question: question.into(),
            answer: message.into(),
            response: llm::AssistantMessage::default(),
        }
    }
}

/// An in-memory, session-independent side discussion.
#[derive(Clone, Debug, PartialEq)]
pub struct Thread {
    pub id: String,
    pub title: String,
    pub conversation_context: String,
    pub turns: Vec<Turn>,
    pub thinking_level: llm::ThinkingLevel,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    update_sequence: u64,
}

impl AsRef<str> for Thread {
    fn as_ref(&self) -> &str {
        &self.id
    }
}

/// Render-friendly metadata for a non-empty side thread.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Summary {
    pub id: String,
    pub title: String,
    pub questions: usize,
    pub updated_at: SystemTime,
}

/// Failure returned when a manager operation refers to a removed or unknown
/// thread.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThreadError {
    UnknownThread(String),
}

impl fmt::Display for ThreadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownThread(id) => write!(formatter, "unknown BTW thread {id:?}"),
        }
    }
}

impl Error for ThreadError {}

/// Thread-safe owner of all in-memory side threads.
///
/// Methods return snapshots instead of mutable references.  That permits a UI
/// to render a thread while a provider request records a completion on another
/// thread without exposing aliased mutable state.
pub struct Manager {
    state: Mutex<ManagerState>,
}

struct ManagerState {
    next_id: u64,
    next_update_sequence: u64,
    threads: BTreeMap<String, Thread>,
}

impl Default for Manager {
    fn default() -> Self {
        Self {
            state: Mutex::new(ManagerState {
                next_id: 1,
                next_update_sequence: 1,
                threads: BTreeMap::new(),
            }),
        }
    }
}

impl Manager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty side thread.  The caller may dispatch its first
    /// question with [`build_request_context`] and later record the result.
    pub fn new_thread(
        &self,
        conversation_context: impl Into<String>,
        thinking_level: impl Into<llm::ThinkingLevel>,
    ) -> Thread {
        let mut state = lock(&self.state);
        let id = format!("btw-{}", state.next_id);
        state.next_id += 1;
        let update_sequence = next_update_sequence(&mut state);
        let now = SystemTime::now();
        let thread = Thread {
            id: id.clone(),
            title: String::new(),
            conversation_context: conversation_context.into(),
            turns: Vec::new(),
            thinking_level: thinking_level.into(),
            created_at: now,
            updated_at: now,
            update_sequence,
        };
        state.threads.insert(id, thread.clone());
        thread
    }

    /// Alias for callers that prefer a shorter construction name.
    pub fn create(
        &self,
        conversation_context: impl Into<String>,
        thinking_level: impl Into<llm::ThinkingLevel>,
    ) -> Thread {
        self.new_thread(conversation_context, thinking_level)
    }

    /// Returns a snapshot of one thread, if it still exists.
    pub fn get(&self, thread: impl AsRef<str>) -> Option<Thread> {
        lock(&self.state).threads.get(thread.as_ref()).cloned()
    }

    /// Returns an immutable snapshot suitable for rendering.
    pub fn snapshot(&self, thread: impl AsRef<str>) -> Option<Thread> {
        self.get(thread)
    }

    /// Updates a thread-local thinking level without changing the main
    /// conversation's configuration.
    pub fn set_thinking_level(
        &self,
        thread: impl AsRef<str>,
        thinking_level: impl Into<llm::ThinkingLevel>,
    ) -> Result<(), ThreadError> {
        let id = thread.as_ref().to_owned();
        let mut state = lock(&self.state);
        let Some(thread) = state.threads.get_mut(&id) else {
            return Err(ThreadError::UnknownThread(id));
        };
        thread.thinking_level = thinking_level.into();
        Ok(())
    }

    /// Records a completed request and returns its displayable text.
    pub fn record_answered(
        &self,
        thread: impl AsRef<str>,
        question: impl Into<String>,
        response: llm::AssistantMessage,
    ) -> Result<String, ThreadError> {
        let id = thread.as_ref().to_owned();
        let question = question.into();
        let mut answer = assistant_text(&response);
        if answer.is_empty() {
            answer = "No response received.".to_owned();
        }

        let mut state = lock(&self.state);
        let update_sequence = next_update_sequence(&mut state);
        let Some(thread) = state.threads.get_mut(&id) else {
            return Err(ThreadError::UnknownThread(id));
        };
        thread
            .turns
            .push(Turn::answered(question.clone(), answer.clone(), response));
        set_title_if_empty(thread, &question);
        thread.updated_at = SystemTime::now();
        thread.update_sequence = update_sequence;
        Ok(answer)
    }

    /// Records a non-cancellation error from an independent side request.
    pub fn record_error(
        &self,
        thread: impl AsRef<str>,
        question: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<(), ThreadError> {
        let id = thread.as_ref().to_owned();
        let question = question.into();
        let mut state = lock(&self.state);
        let update_sequence = next_update_sequence(&mut state);
        let Some(thread) = state.threads.get_mut(&id) else {
            return Err(ThreadError::UnknownThread(id));
        };
        thread.turns.push(Turn::error(question.clone(), message));
        set_title_if_empty(thread, &question);
        thread.updated_at = SystemTime::now();
        thread.update_sequence = update_sequence;
        Ok(())
    }

    /// Lists started threads from most recently updated to least recently
    /// updated.  The monotonic sequence makes equal wall-clock timestamps
    /// deterministic.
    pub fn list(&self) -> Vec<Summary> {
        let state = lock(&self.state);
        let mut summaries = state
            .threads
            .values()
            .filter(|thread| !thread.title.is_empty() && !thread.turns.is_empty())
            .map(|thread| {
                (
                    thread.update_sequence,
                    Summary {
                        id: thread.id.clone(),
                        title: thread.title.clone(),
                        questions: thread.turns.len(),
                        updated_at: thread.updated_at,
                    },
                )
            })
            .collect::<Vec<_>>();
        summaries.sort_by(|(left_sequence, left), (right_sequence, right)| {
            right_sequence
                .cmp(left_sequence)
                .then_with(|| left.id.cmp(&right.id))
        });
        summaries.into_iter().map(|(_, summary)| summary).collect()
    }
}

/// Backwards-descriptive name for the side-thread manager.
pub type ThreadManager = Manager;

/// Whether a queued side question begins a discussion or follows an earlier
/// answered turn.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueuedQuestionKind {
    Prompt,
    FollowUp,
}

/// A question awaiting dispatch by a future provider/agent runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueuedQuestion {
    pub kind: QueuedQuestionKind,
    pub question: String,
}

/// How many questions an owner wants to drain from a [`QuestionQueue`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueueDrainMode {
    All,
    #[default]
    OneAtATime,
}

/// FIFO queue for a single side thread.
///
/// Prompt and follow-up questions share one queue: a follow-up can never
/// overtake the initial prompt that establishes its discussion history.  The
/// queue does not perform provider work; this keeps cancellation, streaming,
/// and scheduling owned by the future agent runtime.
#[derive(Clone, Debug, Default)]
pub struct QuestionQueue {
    questions: VecDeque<QueuedQuestion>,
}

impl QuestionQueue {
    pub fn enqueue_prompt(&mut self, question: impl Into<String>) {
        self.questions.push_back(QueuedQuestion {
            kind: QueuedQuestionKind::Prompt,
            question: question.into(),
        });
    }

    pub fn enqueue_follow_up(&mut self, question: impl Into<String>) {
        self.questions.push_back(QueuedQuestion {
            kind: QueuedQuestionKind::FollowUp,
            question: question.into(),
        });
    }

    pub fn dequeue(&mut self) -> Option<QueuedQuestion> {
        self.questions.pop_front()
    }

    pub fn drain(&mut self, mode: QueueDrainMode) -> Vec<QueuedQuestion> {
        match mode {
            QueueDrainMode::All => self.questions.drain(..).collect(),
            QueueDrainMode::OneAtATime => self.dequeue().into_iter().collect(),
        }
    }

    pub fn clear(&mut self) {
        self.questions.clear();
    }

    pub fn len(&self) -> usize {
        self.questions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }
}

/// Builds the side-thread transcript sent to the model for `question`.
///
/// The main-conversation context is present only in the first user message.
/// Every later question is a compact follow-up, and errored side turns are
/// intentionally excluded from replay.
pub fn build_messages(thread: &Thread, question: &str) -> Vec<llm::Message> {
    let answered = thread
        .turns
        .iter()
        .filter(|turn| turn.kind == TurnKind::Answered)
        .collect::<Vec<_>>();
    let Some(first) = answered.first() else {
        return vec![side_user_message(build_user_prompt(
            question,
            &thread.conversation_context,
        ))];
    };

    let mut messages = vec![
        side_user_message(build_user_prompt(
            &first.question,
            &thread.conversation_context,
        )),
        llm::Message::Assistant(Box::new(first.response.clone())),
    ];
    for turn in answered.iter().skip(1) {
        messages.push(side_user_message(build_follow_up_prompt(&turn.question)));
        messages.push(llm::Message::Assistant(Box::new(turn.response.clone())));
    }
    messages.push(side_user_message(build_follow_up_prompt(question)));
    messages
}

/// Builds a provider-ready LLM context without importing any tools or main
/// transcript state.
pub fn build_request_context(thread: &Thread, question: &str) -> llm::Context {
    llm::Context {
        system_prompt: SYSTEM_PROMPT.to_owned(),
        messages: build_messages(thread, question),
        tools: Vec::new(),
    }
}

/// Formats the initial side question and its read-only main context.
pub fn build_user_prompt(question: &str, conversation_context: &str) -> String {
    let conversation_context = if conversation_context.is_empty() {
        "No prior conversation context was available."
    } else {
        conversation_context
    };
    format!(
        "Answer this side question without modifying the main conversation.\n\
\n\
<side_question>\n\
{question}\n\
</side_question>\n\
\n\
<conversation_context>\n\
{conversation_context}\n\
</conversation_context>"
    )
}

/// Formats every question after the first successful side turn.
pub fn build_follow_up_prompt(question: &str) -> String {
    format!(
        "Continue the same side conversation.\n\
\n\
<side_question>\n\
{question}\n\
</side_question>"
    )
}

/// Extracts visible text blocks from a model response.
pub fn assistant_text(message: &llm::AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(llm::ContentBlock::plain_text)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_owned()
}

/// Creates a read-only text context from the main LLM transcript.
///
/// User and assistant text are retained.  Assistant tool calls are summarized
/// while tool-result messages are omitted.  When the context is too large,
/// keeping the newest Unicode scalar values matches pi-btw's behavior.
pub fn build_conversation_context(messages: &[llm::Message]) -> String {
    let mut sections = Vec::new();
    for message in messages {
        match message {
            llm::Message::User(message) => {
                append_context_section(&mut sections, "User", user_content_lines(&message.content));
            }
            llm::Message::Assistant(message) => {
                let role = if message.stop_reason.is_empty() || message.stop_reason == "stop" {
                    "Assistant".to_owned()
                } else {
                    format!("Assistant ({})", message.stop_reason)
                };
                append_context_section(&mut sections, &role, content_lines(&message.content));
            }
            llm::Message::ToolResult(_) => {}
        }
    }

    let context = sections.join("\n\n");
    if context.chars().count() <= MAX_CONTEXT_CHARS {
        return context;
    }
    let start = context
        .char_indices()
        .nth(context.chars().count() - MAX_CONTEXT_CHARS)
        .map_or(0, |(index, _)| index);
    format!(
        "[Earlier context omitted; showing the last {MAX_CONTEXT_CHARS} characters.]\n{}",
        &context[start..]
    )
}

/// A source role for a segment brought from a side thread into the main
/// conversation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentRole {
    User,
    Assistant,
}

/// One readable side-thread segment selected for export or `/btw bring`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    pub role: SegmentRole,
    pub text: String,
}

impl Segment {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: SegmentRole::User,
            text: text.into(),
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: SegmentRole::Assistant,
            text: text.into(),
        }
    }
}

/// Scope accepted by the `/btw bring` command layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BringSelection {
    /// The latest successful question and answer.
    Latest,
    /// Every successful question and answer.
    All,
    /// Successful turns starting at a one-based turn number.
    From(usize),
}

/// Invalid user input for a bring scope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BringSelectionError {
    value: String,
}

impl fmt::Display for BringSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid bring scope {:?}; use latest, all, or from:N with N >= 1",
            self.value
        )
    }
}

impl Error for BringSelectionError {}

/// Parses a case-insensitive `latest`, `all`, or `from:N` selection.
pub fn parse_bring_selection(value: &str) -> Result<BringSelection, BringSelectionError> {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "" | "latest" => Ok(BringSelection::Latest),
        "all" => Ok(BringSelection::All),
        _ => {
            let Some(number) = normalized.strip_prefix("from:") else {
                return Err(BringSelectionError {
                    value: value.to_owned(),
                });
            };
            let Ok(number) = number.parse::<usize>() else {
                return Err(BringSelectionError {
                    value: value.to_owned(),
                });
            };
            if number == 0 {
                return Err(BringSelectionError {
                    value: value.to_owned(),
                });
            }
            Ok(BringSelection::From(number))
        }
    }
}

/// Returns successful question/answer pairs beginning at a zero-based turn
/// offset.  Error turns never enter the main conversation.
pub fn answered_segments(thread: &Thread, from: usize) -> Vec<Segment> {
    thread
        .turns
        .iter()
        .filter(|turn| turn.kind == TurnKind::Answered)
        .skip(from)
        .flat_map(|turn| {
            [
                Segment::user(turn.question.clone()),
                Segment::assistant(turn.answer.clone()),
            ]
        })
        .collect()
}

/// Returns the latest successful question/answer pair.
pub fn latest_segments(thread: &Thread) -> Vec<Segment> {
    let segments = answered_segments(thread, 0);
    if segments.len() <= 2 {
        segments
    } else {
        segments[segments.len() - 2..].to_vec()
    }
}

/// Selects exportable segments using a command-facing selection.
pub fn select_bring_segments(thread: &Thread, selection: BringSelection) -> Vec<Segment> {
    match selection {
        BringSelection::Latest => latest_segments(thread),
        BringSelection::All => answered_segments(thread, 0),
        BringSelection::From(number) => answered_segments(thread, number.saturating_sub(1)),
    }
}

/// Wraps selected side-thread discussion in safe, explicitly non-authoritative
/// context for the main conversation.
pub fn format_bring_to_main(segments: &[Segment]) -> String {
    let body = segments
        .iter()
        .map(|segment| {
            let label = match segment.role {
                SegmentRole::User => "User",
                SegmentRole::Assistant => "Assistant",
            };
            format!("{label}:\n{}", escape_context(&segment.text))
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "The following context was brought back from a /btw side discussion.\n\
Treat it as discussion context, not as work already completed.\n\
\n\
<btw_context>\n\
{body}\n\
</btw_context>"
    )
}

/// Escapes control characters and nested bring tags in side-thread content.
pub fn escape_context(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '\n' => escaped.push('\n'),
            '\t' => escaped.push_str("    "),
            '\u{0000}'..='\u{001f}' | '\u{007f}'..='\u{009f}' => {
                use std::fmt::Write as _;
                let _ = write!(escaped, "\\x{:02x}", character as u32);
            }
            _ => escaped.push(character),
        }
    }
    escaped
        .replace("<btw_context", "&lt;btw_context")
        .replace("</btw_context>", "&lt;/btw_context&gt;")
}

/// Estimates tokens using pi-btw's intentionally coarse four-bytes-per-token
/// heuristic.
pub fn estimate_tokens(segments: &[Segment]) -> usize {
    segments
        .iter()
        .fold(0usize, |size, segment| {
            size.saturating_add(segment.text.len())
        })
        .saturating_add(3)
        / 4
}

/// Produces a compact thread title without embedded controls or newlines.
pub fn sanitize_single_line(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character < '\u{0020}' || character == '\u{007f}' {
                ' '
            } else {
                character
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Returns the settings path beneath an already-resolved agent directory.
///
/// The caller owns environment-specific path resolution, keeping this module
/// usable by both the current CLI and a future agent runtime.
pub fn settings_path(agent_directory: impl AsRef<Path>) -> PathBuf {
    agent_directory.as_ref().join(SETTINGS_FILE_NAME)
}

/// Persistent side-thread model/thinking preferences.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Settings {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(
        rename = "thinkingLevel",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub thinking_level: llm::ThinkingLevel,
    #[serde(
        rename = "rememberThinkingLevelChanges",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub remember_thinking_level_changes: Option<bool>,
}

impl Settings {
    /// An omitted setting preserves pi-btw's historical default of `true`.
    pub fn effective_remember(&self) -> bool {
        self.remember_thinking_level_changes.unwrap_or(true)
    }
}

/// Returns whether pi-btw should remember side-thread thinking-level changes.
pub fn effective_remember(settings: &Settings) -> bool {
    settings.effective_remember()
}

/// Result shape for a side-effect-free settings read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SettingsKind {
    Missing,
    Loaded,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SettingsResult {
    pub kind: SettingsKind,
    pub settings: Settings,
    pub reason: String,
}

impl SettingsResult {
    fn missing() -> Self {
        Self {
            kind: SettingsKind::Missing,
            settings: Settings::default(),
            reason: String::new(),
        }
    }

    fn loaded(settings: Settings) -> Self {
        Self {
            kind: SettingsKind::Loaded,
            settings,
            reason: String::new(),
        }
    }

    fn invalid(reason: impl Into<String>) -> Self {
        Self {
            kind: SettingsKind::Invalid,
            settings: Settings::default(),
            reason: reason.into(),
        }
    }
}

/// Describes how an existing settings field should be changed.
///
/// `Clear` removes only the known field, preserving all unrelated fields in
/// the JSON object.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum SettingChange<T> {
    #[default]
    Unchanged,
    Set(T),
    Clear,
}

/// A non-destructive update to `pi-btw.json`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SettingsPatch {
    pub model: SettingChange<String>,
    pub thinking_level: SettingChange<llm::ThinkingLevel>,
    pub remember_thinking_level_changes: SettingChange<bool>,
}

impl SettingsPatch {
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = SettingChange::Set(model.into());
        self
    }

    pub fn with_thinking_level(mut self, thinking_level: impl Into<llm::ThinkingLevel>) -> Self {
        self.thinking_level = SettingChange::Set(thinking_level.into());
        self
    }

    pub fn with_remember_thinking_level_changes(mut self, remember: bool) -> Self {
        self.remember_thinking_level_changes = SettingChange::Set(remember);
        self
    }
}

/// Error from validating or atomically updating `pi-btw.json`.
#[derive(Debug)]
pub enum SettingsError {
    InvalidModelReference(String),
    InvalidThinkingLevel(String),
    InvalidExisting { path: PathBuf, reason: String },
    InvalidPath(PathBuf),
    TooLarge { max_bytes: usize },
    Serialization(String),
    Io { path: PathBuf, source: io::Error },
}

impl fmt::Display for SettingsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelReference(model) => {
                write!(formatter, "invalid model reference {model:?}")
            }
            Self::InvalidThinkingLevel(level) => {
                write!(formatter, "invalid thinking level {level:?}")
            }
            Self::InvalidExisting { path, reason } => {
                write!(
                    formatter,
                    "pi-btw settings at {} are invalid: {reason}",
                    path.display()
                )
            }
            Self::InvalidPath(path) => write!(formatter, "{} has no file name", path.display()),
            Self::TooLarge { max_bytes } => {
                write!(formatter, "settings document exceeds {max_bytes} bytes")
            }
            Self::Serialization(reason) => write!(formatter, "could not encode settings: {reason}"),
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
        }
    }
}

impl Error for SettingsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Reads `pi-btw.json` without creating a missing file.
pub fn read_settings(path: impl AsRef<Path>) -> SettingsResult {
    let path = path.as_ref();
    let path_lock = path_lock(path);
    let _guard = lock(&path_lock);
    match read_settings_document(path) {
        Ok((settings, _)) => SettingsResult::loaded(settings),
        Err(SettingsDocumentError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            SettingsResult::missing()
        }
        Err(error) => SettingsResult::invalid(error.to_string()),
    }
}

/// Applies a validated patch while preserving unknown JSON fields.
///
/// The replacement is written to a same-directory private temporary file,
/// synced, atomically renamed, then the directory is synced on Unix.  Invalid
/// existing documents are deliberately never overwritten.
pub fn update_settings(
    path: impl AsRef<Path>,
    patch: SettingsPatch,
) -> Result<Settings, SettingsError> {
    validate_patch(&patch)?;
    let path = path.as_ref();
    let path_lock = path_lock(path);
    let _guard = lock(&path_lock);

    let (mut settings, mut document) = match read_settings_document(path) {
        Ok(document) => document,
        Err(SettingsDocumentError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            (Settings::default(), Map::new())
        }
        Err(SettingsDocumentError::Invalid(reason)) => {
            return Err(SettingsError::InvalidExisting {
                path: path.to_path_buf(),
                reason,
            });
        }
        Err(SettingsDocumentError::Io(source)) => {
            return Err(SettingsError::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };

    apply_patch(&mut settings, &mut document, patch);
    publish_settings(path, &document)?;
    Ok(settings)
}

/// Convenience port of the original settings operation, which changes only
/// thinking and whether such changes should be remembered.
pub fn update_thinking_preferences(
    path: impl AsRef<Path>,
    thinking_level: Option<&str>,
    remember_thinking_level_changes: Option<bool>,
) -> Result<Settings, SettingsError> {
    let mut patch = SettingsPatch::default();
    if let Some(thinking_level) = thinking_level {
        patch.thinking_level = SettingChange::Set(thinking_level.to_owned());
    }
    if let Some(remember) = remember_thinking_level_changes {
        patch.remember_thinking_level_changes = SettingChange::Set(remember);
    }
    update_settings(path, patch)
}

/// Splits a validated `provider/model` reference.  The model portion may itself
/// contain `/`, matching pi-btw's provider/reference convention.
pub fn parse_model_reference(reference: &str) -> Option<(&str, &str)> {
    if reference.trim() != reference || reference.contains(['\t', '\r', '\n']) {
        return None;
    }
    let (provider, model) = reference.split_once('/')?;
    (!provider.is_empty() && !model.is_empty()).then_some((provider, model))
}

/// Returns whether a value can be stored as a side-thread thinking level.
pub fn is_valid_thinking_level(value: &str) -> bool {
    THINKING_LEVELS.contains(&value)
}

/// Returns the levels accepted by `model`, honoring explicit `null` entries in
/// its pi-compatible `thinking_level_map`.
pub fn supported_thinking_levels(model: &llm::Model) -> Vec<&'static str> {
    if !model.reasoning {
        return vec![llm::THINKING_OFF];
    }

    THINKING_LEVELS
        .iter()
        .copied()
        .filter(|level| {
            let configured = model.thinking_level_map.get(*level);
            if matches!(configured, Some(None)) {
                return false;
            }
            !matches!(*level, llm::THINKING_XHIGH | llm::THINKING_MAX) || configured.is_some()
        })
        .collect()
}

/// Maps unsupported or unknown requested levels to the nearest supported one,
/// preferring a stronger level before a weaker level.
pub fn clamp_thinking_level(model: &llm::Model, requested: &str) -> llm::ThinkingLevel {
    let available = supported_thinking_levels(model);
    if available.contains(&requested) {
        return requested.to_owned();
    }
    let Some(requested_index) = THINKING_LEVELS.iter().position(|level| *level == requested) else {
        return available
            .first()
            .copied()
            .unwrap_or(llm::THINKING_OFF)
            .to_owned();
    };
    for level in THINKING_LEVELS.iter().skip(requested_index) {
        if available.contains(level) {
            return (*level).to_owned();
        }
    }
    for level in THINKING_LEVELS[..requested_index].iter().rev() {
        if available.contains(level) {
            return (*level).to_owned();
        }
    }
    available
        .first()
        .copied()
        .unwrap_or(llm::THINKING_OFF)
        .to_owned()
}

/// Model and thinking values selected independently for a side thread.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSelection {
    pub model: llm::Model,
    pub thinking_level: llm::ThinkingLevel,
    pub remember_thinking_level_changes: bool,
    pub warnings: Vec<String>,
}

/// Resolves settings against a caller-provided model catalog without importing
/// the main session.  An unavailable configured model safely falls back to the
/// current model.
pub fn resolve_selection<F>(
    current_model: &llm::Model,
    current_thinking_level: &str,
    settings: &Settings,
    mut resolve_model: F,
) -> ResolvedSelection
where
    F: FnMut(&str) -> Result<llm::Model, String>,
{
    let mut model = current_model.clone();
    let mut warnings = Vec::new();
    if !settings.model.is_empty() {
        if parse_model_reference(&settings.model).is_none() {
            warnings.push(format!(
                "pi-btw model {} is invalid; falling back to {}/{}",
                settings.model, current_model.provider, current_model.id
            ));
        } else {
            match resolve_model(&settings.model) {
                Ok(resolved) => model = resolved,
                Err(error) => warnings.push(format!(
                    "pi-btw model {} is unavailable ({}); falling back to {}/{}",
                    settings.model, error, current_model.provider, current_model.id
                )),
            }
        }
    }
    let requested_thinking = if settings.thinking_level.is_empty() {
        current_thinking_level
    } else {
        &settings.thinking_level
    };
    ResolvedSelection {
        thinking_level: clamp_thinking_level(&model, requested_thinking),
        model,
        remember_thinking_level_changes: effective_remember(settings),
        warnings,
    }
}

fn side_user_message(text: String) -> llm::Message {
    llm::Message::User(llm::UserMessage::text(text, now_millis()))
}

fn append_context_section(sections: &mut Vec<String>, role: &str, lines: Vec<String>) {
    if !lines.is_empty() {
        sections.push(format!("{role}: {}", lines.join("\n")));
    }
}

fn user_content_lines(content: &llm::UserContent) -> Vec<String> {
    match content {
        llm::UserContent::Text(text) => nonempty_trimmed_line(text).into_iter().collect(),
        llm::UserContent::Blocks(blocks) => content_lines(blocks),
    }
}

fn content_lines(content: &[llm::ContentBlock]) -> Vec<String> {
    content
        .iter()
        .filter_map(|block| match block {
            llm::ContentBlock::Text(text) => nonempty_trimmed_line(&text.text),
            llm::ContentBlock::ToolCall(call) => Some(format!(
                "Tool call: {}({})",
                call.name,
                serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned())
            )),
            llm::ContentBlock::Thinking(_) | llm::ContentBlock::Image(_) => None,
        })
        .collect()
}

fn nonempty_trimmed_line(text: &str) -> Option<String> {
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

fn set_title_if_empty(thread: &mut Thread, question: &str) {
    if thread.title.is_empty() {
        thread.title = sanitize_single_line(question);
        if thread.title.is_empty() {
            thread.title = "Untitled side thread".to_owned();
        }
    }
}

fn next_update_sequence(state: &mut ManagerState) -> u64 {
    let sequence = state.next_update_sequence;
    state.next_update_sequence = state.next_update_sequence.saturating_add(1);
    sequence
}

enum SettingsDocumentError {
    Io(io::Error),
    Invalid(String),
}

impl fmt::Display for SettingsDocumentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Invalid(reason) => formatter.write_str(reason),
        }
    }
}

fn read_settings_document(
    path: &Path,
) -> Result<(Settings, Map<String, Value>), SettingsDocumentError> {
    let mut file = File::open(path).map_err(SettingsDocumentError::Io)?;
    let metadata = file.metadata().map_err(SettingsDocumentError::Io)?;
    if !metadata.file_type().is_file() {
        return Err(SettingsDocumentError::Invalid(
            "settings path is not a regular file".to_owned(),
        ));
    }
    if metadata.len() > MAX_SETTINGS_BYTES as u64 {
        return Err(SettingsDocumentError::Invalid(format!(
            "settings file exceeds {MAX_SETTINGS_BYTES} bytes"
        )));
    }

    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(MAX_SETTINGS_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(SettingsDocumentError::Io)?;
    if bytes.len() > MAX_SETTINGS_BYTES {
        return Err(SettingsDocumentError::Invalid(format!(
            "settings file exceeds {MAX_SETTINGS_BYTES} bytes"
        )));
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| {
        SettingsDocumentError::Invalid("settings file is not valid UTF-8".to_owned())
    })?;
    let value: Value = serde_json::from_str(text)
        .map_err(|_| SettingsDocumentError::Invalid("invalid JSON".to_owned()))?;
    let Value::Object(document) = value else {
        return Err(SettingsDocumentError::Invalid(
            "invalid settings shape".to_owned(),
        ));
    };
    let settings = decode_settings(&document).map_err(SettingsDocumentError::Invalid)?;
    Ok((settings, document))
}

fn decode_settings(document: &Map<String, Value>) -> Result<Settings, String> {
    let mut settings = Settings::default();
    if let Some(value) = document.get("model") {
        let Some(model) = value.as_str() else {
            return Err("invalid model".to_owned());
        };
        if parse_model_reference(model).is_none() {
            return Err("invalid model reference".to_owned());
        }
        settings.model = model.to_owned();
    }
    if let Some(value) = document.get("thinkingLevel") {
        let Some(thinking_level) = value.as_str() else {
            return Err("invalid thinkingLevel".to_owned());
        };
        if thinking_level.is_empty() || !is_valid_thinking_level(thinking_level) {
            return Err("invalid thinkingLevel".to_owned());
        }
        settings.thinking_level = thinking_level.to_owned();
    }
    if let Some(value) = document.get("rememberThinkingLevelChanges") {
        let Some(remember) = value.as_bool() else {
            return Err("invalid rememberThinkingLevelChanges".to_owned());
        };
        settings.remember_thinking_level_changes = Some(remember);
    }
    Ok(settings)
}

fn validate_patch(patch: &SettingsPatch) -> Result<(), SettingsError> {
    match &patch.model {
        SettingChange::Set(model) if parse_model_reference(model).is_none() => {
            return Err(SettingsError::InvalidModelReference(model.clone()));
        }
        _ => {}
    }
    match &patch.thinking_level {
        SettingChange::Set(thinking_level)
            if thinking_level.is_empty() || !is_valid_thinking_level(thinking_level) =>
        {
            return Err(SettingsError::InvalidThinkingLevel(thinking_level.clone()));
        }
        _ => {}
    }
    Ok(())
}

fn apply_patch(settings: &mut Settings, document: &mut Map<String, Value>, patch: SettingsPatch) {
    match patch.model {
        SettingChange::Unchanged => {}
        SettingChange::Set(model) => {
            document.insert("model".to_owned(), Value::String(model.clone()));
            settings.model = model;
        }
        SettingChange::Clear => {
            document.remove("model");
            settings.model.clear();
        }
    }
    match patch.thinking_level {
        SettingChange::Unchanged => {}
        SettingChange::Set(thinking_level) => {
            document.insert(
                "thinkingLevel".to_owned(),
                Value::String(thinking_level.clone()),
            );
            settings.thinking_level = thinking_level;
        }
        SettingChange::Clear => {
            document.remove("thinkingLevel");
            settings.thinking_level.clear();
        }
    }
    match patch.remember_thinking_level_changes {
        SettingChange::Unchanged => {}
        SettingChange::Set(remember) => {
            document.insert(
                "rememberThinkingLevelChanges".to_owned(),
                Value::Bool(remember),
            );
            settings.remember_thinking_level_changes = Some(remember);
        }
        SettingChange::Clear => {
            document.remove("rememberThinkingLevelChanges");
            settings.remember_thinking_level_changes = None;
        }
    }
}

fn publish_settings(path: &Path, document: &Map<String, Value>) -> Result<(), SettingsError> {
    let mut data = serde_json::to_vec_pretty(document)
        .map_err(|error| SettingsError::Serialization(error.to_string()))?;
    data.push(b'\n');
    if data.len() > MAX_SETTINGS_BYTES {
        return Err(SettingsError::TooLarge {
            max_bytes: MAX_SETTINGS_BYTES,
        });
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or_else(|| SettingsError::InvalidPath(path.to_path_buf()))?;
    ensure_private_directory(parent).map_err(|source| SettingsError::Io {
        path: parent.to_path_buf(),
        source,
    })?;
    let (temporary, mut file) =
        create_temporary_file(parent, file_name).map_err(|source| SettingsError::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let write_result = (|| -> io::Result<()> {
        file.write_all(&data)?;
        file.sync_all()
    })();
    drop(file);
    if let Err(source) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(SettingsError::Io {
            path: path.to_path_buf(),
            source,
        });
    }
    if let Err(source) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(SettingsError::Io {
            path: path.to_path_buf(),
            source,
        });
    }
    sync_parent_directory(parent).map_err(|source| SettingsError::Io {
        path: parent.to_path_buf(),
        source,
    })
}

fn ensure_private_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700).create(path)
    }
    #[cfg(not(unix))]
    {
        fs::create_dir_all(path)
    }
}

fn create_temporary_file(
    parent: &Path,
    file_name: &std::ffi::OsStr,
) -> io::Result<(PathBuf, File)> {
    for _ in 0..128 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            process::id(),
            sequence
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&temporary) {
            Ok(file) => return Ok((temporary, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not create a unique settings temporary file",
    ))
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_: &Path) -> io::Result<()> {
    Ok(())
}

fn path_lock(path: &Path) -> Arc<Mutex<()>> {
    let key = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    let locks = SETTINGS_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    lock(locks)
        .entry(key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        env, fs,
        path::PathBuf,
        process,
        sync::{Arc, Barrier},
        thread,
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::*;

    fn test_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        env::temp_dir().join(format!("goshcoder-btw-{label}-{}-{nonce}", process::id()))
    }

    fn response(text: &str) -> llm::AssistantMessage {
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

    fn thread_with_turns(turns: Vec<Turn>) -> Thread {
        let now = SystemTime::now();
        Thread {
            id: "btw-test".to_owned(),
            title: "test".to_owned(),
            conversation_context: "User: main task".to_owned(),
            turns,
            thinking_level: llm::THINKING_LOW.to_owned(),
            created_at: now,
            updated_at: now,
            update_sequence: 1,
        }
    }

    fn model(reasoning: bool) -> llm::Model {
        llm::Model {
            id: "current".to_owned(),
            provider: "provider".to_owned(),
            reasoning,
            ..llm::Model::default()
        }
    }

    #[test]
    fn builds_an_isolated_initial_prompt_and_compact_follow_up_history() {
        let thread = thread_with_turns(vec![
            Turn::answered("first?", "first answer", response("first answer")),
            Turn::error("bad?", "provider failed"),
            Turn::answered("second?", "second answer", response("second answer")),
        ]);

        let messages = build_messages(&thread, "third?");
        assert_eq!(messages.len(), 5);
        let llm::Message::User(first) = &messages[0] else {
            panic!("first message should be a user prompt");
        };
        let first = first.content.text().expect("plain user content");
        assert!(first.contains("<conversation_context>\nUser: main task"));
        assert!(first.contains("<side_question>\nfirst?\n</side_question>"));
        assert!(!first.contains("bad?"));
        assert!(
            matches!(&messages[1], llm::Message::Assistant(message) if message.content == response("first answer").content)
        );
        let llm::Message::User(last) = &messages[4] else {
            panic!("last message should be a user prompt");
        };
        let last = last.content.text().expect("plain follow-up content");
        assert!(last.contains("<side_question>\nthird?\n</side_question>"));
        assert!(!last.contains("conversation_context"));

        let request = build_request_context(&thread, "third?");
        assert_eq!(request.system_prompt, SYSTEM_PROMPT);
        assert!(request.tools.is_empty());
        assert_eq!(request.messages, messages);
    }

    #[test]
    fn conversation_context_summarizes_calls_omits_results_and_keeps_newest_unicode() {
        let oversized = "🦀".repeat(MAX_CONTEXT_CHARS);
        let context = build_conversation_context(&[
            llm::Message::User(llm::UserMessage::text(oversized, 1)),
            llm::Message::Assistant(Box::new(llm::AssistantMessage {
                content: vec![
                    llm::ContentBlock::ToolCall(llm::ToolCall {
                        id: "call-1".to_owned(),
                        name: "read".to_owned(),
                        arguments: BTreeMap::from([("path".to_owned(), json!("a.rs"))]),
                        ..llm::ToolCall::default()
                    }),
                    llm::ContentBlock::text("newest"),
                ],
                stop_reason: "toolUse".to_owned(),
                ..llm::AssistantMessage::default()
            })),
            llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
                content: vec![llm::ContentBlock::text("secret tool result")],
                ..llm::ToolResultMessage::default()
            })),
        ]);
        let marker = format!(
            "[Earlier context omitted; showing the last {MAX_CONTEXT_CHARS} characters.]\n"
        );
        assert!(context.starts_with(&marker));
        let retained = &context[marker.len()..];
        assert_eq!(retained.chars().count(), MAX_CONTEXT_CHARS);
        assert!(context.contains("Assistant (toolUse):"));
        assert!(context.contains(r#"Tool call: read({"path":"a.rs"})"#));
        assert!(context.ends_with("newest"));
        assert!(!context.contains("secret tool result"));
    }

    #[test]
    fn manager_snapshots_are_isolated_and_listing_order_is_deterministic() {
        let manager = Manager::new();
        let first = manager.new_thread("context", "low");
        let second = manager.new_thread("context", "medium");
        assert!(manager.list().is_empty(), "empty threads are not listed");

        manager
            .record_answered(&first, " first\n question ", response("a1"))
            .expect("record first answer");
        let snapshot = manager.snapshot(&first).expect("first snapshot");
        manager
            .record_error(&second, "\t", "oops")
            .expect("record second error");
        manager
            .record_answered(&first, "another", response("a2"))
            .expect("record second first-thread answer");

        assert_eq!(snapshot.turns.len(), 1, "snapshot must not alias manager");
        assert_eq!(
            manager
                .snapshot(&first)
                .expect("updated snapshot")
                .turns
                .len(),
            2
        );
        let summaries = manager.list();
        assert_eq!(
            summaries
                .iter()
                .map(|summary| &summary.id)
                .collect::<Vec<_>>(),
            vec![&first.id, &second.id]
        );
        assert_eq!(summaries[0].title, "first question");
        assert_eq!(summaries[1].title, "Untitled side thread");
        assert_eq!(summaries[0].questions, 2);
    }

    #[test]
    fn manager_serializes_concurrent_answer_recording() {
        let manager = Arc::new(Manager::new());
        let side_thread = manager.new_thread("context", "off");
        let start = Arc::new(Barrier::new(9));
        let workers = (0..8)
            .map(|index| {
                let manager = Arc::clone(&manager);
                let start = Arc::clone(&start);
                let id = side_thread.id.clone();
                thread::spawn(move || {
                    start.wait();
                    manager
                        .record_answered(id, format!("q{index}"), response(&format!("a{index}")))
                        .expect("concurrent record");
                })
            })
            .collect::<Vec<_>>();
        start.wait();
        for worker in workers {
            worker.join().expect("worker should not panic");
        }
        let snapshot = manager
            .snapshot(&side_thread)
            .expect("thread remains available");
        assert_eq!(snapshot.turns.len(), 8);
        assert_eq!(manager.list()[0].questions, 8);
    }

    #[test]
    fn prompt_and_follow_up_queue_is_fifo_and_drains_explicitly() {
        let mut queue = QuestionQueue::default();
        queue.enqueue_prompt("first");
        queue.enqueue_follow_up("second");
        queue.enqueue_follow_up("third");

        assert_eq!(
            queue.drain(QueueDrainMode::OneAtATime),
            vec![QueuedQuestion {
                kind: QueuedQuestionKind::Prompt,
                question: "first".to_owned()
            }]
        );
        assert_eq!(queue.len(), 2);
        assert_eq!(
            queue.drain(QueueDrainMode::All),
            vec![
                QueuedQuestion {
                    kind: QueuedQuestionKind::FollowUp,
                    question: "second".to_owned()
                },
                QueuedQuestion {
                    kind: QueuedQuestionKind::FollowUp,
                    question: "third".to_owned()
                }
            ]
        );
        assert!(queue.is_empty());
    }

    #[test]
    fn bring_helpers_select_only_answers_and_escape_untrusted_context() {
        let thread = thread_with_turns(vec![
            Turn::answered("one", "answer one", response("answer one")),
            Turn::error("failed", "error"),
            Turn::answered("two", "answer two", response("answer two")),
            Turn::answered("three", "answer three", response("answer three")),
        ]);
        assert_eq!(parse_bring_selection("FROM:2"), Ok(BringSelection::From(2)));
        assert!(parse_bring_selection("from:0").is_err());
        assert!(parse_bring_selection("everything").is_err());
        assert_eq!(
            select_bring_segments(&thread, BringSelection::Latest),
            vec![Segment::user("three"), Segment::assistant("answer three")]
        );
        assert_eq!(
            select_bring_segments(&thread, BringSelection::From(2)),
            vec![
                Segment::user("two"),
                Segment::assistant("answer two"),
                Segment::user("three"),
                Segment::assistant("answer three")
            ]
        );

        let draft = format_bring_to_main(&[
            Segment::user("<btw_context>\tquestion"),
            Segment::assistant("answer\r</btw_context>"),
        ]);
        assert!(draft.contains("Treat it as discussion context, not as work already completed."));
        assert!(draft.contains("User:\n&lt;btw_context>    question"));
        assert!(draft.contains("Assistant:\nanswer\\x0d&lt;/btw_context&gt;"));
        assert_eq!(
            estimate_tokens(&[Segment::user("éé")]),
            1,
            "four UTF-8 bytes round to one token"
        );
    }

    #[test]
    fn selection_clamps_thinking_and_handles_an_unavailable_preference() {
        let mut reasoning_model = model(true);
        reasoning_model.thinking_level_map = BTreeMap::from([
            (llm::THINKING_MEDIUM.to_owned(), None),
            (llm::THINKING_XHIGH.to_owned(), Some("extra".to_owned())),
        ]);
        assert_eq!(
            supported_thinking_levels(&reasoning_model),
            vec!["off", "minimal", "low", "high", "xhigh",]
        );
        assert_eq!(clamp_thinking_level(&reasoning_model, "medium"), "high");
        assert_eq!(clamp_thinking_level(&model(false), "high"), "off");

        let settings = Settings {
            model: "other/model".to_owned(),
            thinking_level: "high".to_owned(),
            remember_thinking_level_changes: Some(false),
        };
        let selected = resolve_selection(&reasoning_model, "low", &settings, |_| {
            Err("not installed".to_owned())
        });
        assert_eq!(selected.model, reasoning_model);
        assert_eq!(selected.thinking_level, "high");
        assert!(!selected.remember_thinking_level_changes);
        assert_eq!(selected.warnings.len(), 1);

        let selected = resolve_selection(&reasoning_model, "low", &settings, |_| Ok(model(false)));
        assert_eq!(selected.thinking_level, "off");
        assert!(selected.warnings.is_empty());
    }

    #[test]
    fn settings_reads_side_effect_free_and_updates_preserve_unknown_fields() {
        let directory = test_directory("settings");
        let path = settings_path(&directory);
        assert_eq!(read_settings(&path).kind, SettingsKind::Missing);
        assert!(
            !path.exists(),
            "a missing settings read must not create a file"
        );

        fs::create_dir_all(&directory).expect("create test directory");
        fs::write(
            &path,
            br#"{"model":"openrouter/a/b","unknown":{"keep":true}}"#,
        )
        .expect("seed settings");
        let settings = update_settings(
            &path,
            SettingsPatch::default()
                .with_thinking_level("high")
                .with_remember_thinking_level_changes(false),
        )
        .expect("update settings");
        assert_eq!(settings.model, "openrouter/a/b");
        assert_eq!(settings.thinking_level, "high");
        assert!(!effective_remember(&settings));
        let document: Value =
            serde_json::from_slice(&fs::read(&path).expect("read settings")).expect("valid JSON");
        assert_eq!(document["unknown"], json!({"keep": true}));

        let settings = update_settings(
            &path,
            SettingsPatch::default().with_model("anthropic/claude"),
        )
        .expect("update model independently");
        assert_eq!(settings.model, "anthropic/claude");
        assert_eq!(read_settings(&path).kind, SettingsKind::Loaded);

        let temporary_files = fs::read_dir(&directory)
            .expect("read test directory")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
            .count();
        assert_eq!(
            temporary_files, 0,
            "temporary file must be published or removed"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            assert_eq!(
                fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(directory).expect("clean test directory");
    }

    #[test]
    fn settings_validation_never_overwrites_bad_documents_or_bad_patches() {
        let directory = test_directory("invalid-settings");
        fs::create_dir_all(&directory).expect("create test directory");
        let path = settings_path(&directory);
        let original = br#"{"thinkingLevel":"invalid"}"#;
        fs::write(&path, original).expect("write invalid settings");
        let error = update_settings(&path, SettingsPatch::default().with_thinking_level("low"))
            .expect_err("invalid existing document must block saving");
        assert!(matches!(error, SettingsError::InvalidExisting { .. }));
        assert_eq!(fs::read(&path).expect("read original"), original);

        let invalid_model = update_settings(
            directory.join("new.json"),
            SettingsPatch::default().with_model(" no-provider/model"),
        )
        .expect_err("invalid patch");
        assert!(matches!(
            invalid_model,
            SettingsError::InvalidModelReference(_)
        ));
        assert!(!directory.join("new.json").exists());

        for data in [
            vec![b'x'; MAX_SETTINGS_BYTES + 1],
            vec![0xff],
            b"[]".to_vec(),
            br#"{"rememberThinkingLevelChanges":"yes"}"#.to_vec(),
        ] {
            fs::write(&path, data).expect("write malformed fixture");
            assert_eq!(read_settings(&path).kind, SettingsKind::Invalid);
        }
        fs::remove_dir_all(directory).expect("clean test directory");
    }

    #[test]
    fn settings_lock_prevents_in_process_lost_updates() {
        let directory = test_directory("concurrent-settings");
        let path = settings_path(&directory);
        let start = Arc::new(Barrier::new(3));
        let model_path = path.clone();
        let model_start = Arc::clone(&start);
        let model_writer = thread::spawn(move || {
            model_start.wait();
            update_settings(
                model_path,
                SettingsPatch::default().with_model("provider/model"),
            )
        });
        let thinking_path = path.clone();
        let thinking_start = Arc::clone(&start);
        let thinking_writer = thread::spawn(move || {
            thinking_start.wait();
            update_settings(
                thinking_path,
                SettingsPatch::default()
                    .with_thinking_level("high")
                    .with_remember_thinking_level_changes(false),
            )
        });
        start.wait();
        model_writer
            .join()
            .expect("model writer should not panic")
            .expect("model update");
        thinking_writer
            .join()
            .expect("thinking writer should not panic")
            .expect("thinking update");

        let settings = read_settings(&path);
        assert_eq!(settings.kind, SettingsKind::Loaded);
        assert_eq!(settings.settings.model, "provider/model");
        assert_eq!(settings.settings.thinking_level, "high");
        assert_eq!(
            settings.settings.remember_thinking_level_changes,
            Some(false)
        );
        fs::remove_dir_all(directory).expect("clean test directory");
    }
}
