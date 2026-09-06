//! Pi v3-compatible append-only session files.
//!
//! Session data is deliberately kept as JSON values at this layer. It lets the
//! persistence format retain provider fields added by a newer client while the
//! Rust runtime incrementally grows strongly typed protocol support.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    error::Error as StdError,
    fmt,
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, BufRead, BufReader, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime},
};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

// std links libc, so kill(2) is reachable without a crate. Signal zero only
// checks whether the pid exists, which is what stale-lock detection needs.
#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

#[cfg(unix)]
const ESRCH: i32 = 3;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::{OffsetDateTime, format_description::FormatItem, macros::format_description};
use uuid::Uuid;

use crate::llm;

pub const FORMAT_VERSION: u32 = 3;
pub const MAX_ENTRY_BYTES: usize = 16 << 20;
pub const MAX_SESSION_BYTES: u64 = 256 << 20;
pub const WARN_SESSION_BYTES: u64 = 32 << 20;

const HEADER_TYPE: &str = "session";
const TIMESTAMP_FORMAT: &[FormatItem<'static>] =
    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");

pub const TYPE_MESSAGE: &str = "message";
pub const TYPE_MODEL_CHANGE: &str = "model_change";
pub const TYPE_THINKING_LEVEL_CHANGE: &str = "thinking_level_change";
pub const TYPE_COMPACTION: &str = "compaction";
pub const TYPE_BRANCH_SUMMARY: &str = "branch_summary";
pub const TYPE_CUSTOM: &str = "custom";
pub const TYPE_CUSTOM_MESSAGE: &str = "custom_message";
pub const TYPE_LABEL: &str = "label";
pub const TYPE_SESSION_INFO: &str = "session_info";
pub const TYPE_TRANSCRIPT_RESET: &str = "transcript_reset";

const LOCK_HEARTBEAT: Duration = Duration::from_secs(2);
const LOCK_STALE: Duration = Duration::from_secs(20);
const MAX_SEARCH_TEXT_BYTES: usize = 64 << 10;
const SHORT_ID_LENGTH: usize = 8;

/// Best-effort information about the process currently holding a session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LockOwner {
    pub pid: Option<u32>,
    pub since: Option<SystemTime>,
}

impl fmt::Display for LockOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.pid {
            Some(pid) => write!(formatter, "pid {pid}"),
            None => formatter.write_str("another process"),
        }
    }
}

#[derive(Debug)]
pub enum SessionError {
    Io(io::Error),
    Json(serde_json::Error),
    EmptySession,
    InvalidHeader(String),
    InvalidEntry(String),
    InvalidSessionId(String),
    DuplicateEntryId(String),
    VersionTooNew(u32),
    LegacyFormat(u32),
    EntryTooLarge(usize),
    SessionTooLarge(u64),
    MissingEntry(String),
    Busy(LockOwner),
    NotFound(String),
    Ambiguous(String),
    ReadOnly,
    Degraded(String),
    Closed,
}

impl fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Json(error) => error.fmt(formatter),
            Self::EmptySession => write!(formatter, "sessionlog: session file is empty"),
            Self::InvalidHeader(reason) => {
                write!(formatter, "sessionlog: invalid session header: {reason}")
            }
            Self::InvalidEntry(reason) => {
                write!(formatter, "sessionlog: invalid session entry: {reason}")
            }
            Self::InvalidSessionId(id) => write!(
                formatter,
                "sessionlog: invalid session id {id:?}; it must use letters, digits, '-', '_' or '.', and start and end with a letter or digit"
            ),
            Self::DuplicateEntryId(id) => write!(formatter, "sessionlog: duplicate entry id {id}"),
            Self::VersionTooNew(version) => {
                write!(
                    formatter,
                    "sessionlog: session format {version} is newer than this build"
                )
            }
            Self::LegacyFormat(version) => {
                write!(
                    formatter,
                    "sessionlog: legacy session format v{version} must be forked before it can be continued"
                )
            }
            Self::EntryTooLarge(bytes) => write!(
                formatter,
                "sessionlog: entry of {bytes} bytes exceeds the {MAX_ENTRY_BYTES}-byte limit"
            ),
            Self::SessionTooLarge(bytes) => write!(
                formatter,
                "sessionlog: session of {bytes} bytes exceeds the {MAX_SESSION_BYTES}-byte limit"
            ),
            Self::MissingEntry(id) => write!(formatter, "sessionlog: no entry {id} to branch from"),
            Self::Busy(owner) => {
                write!(
                    formatter,
                    "sessionlog: session is open in another process (held by {owner})"
                )
            }
            Self::NotFound(reference) => {
                write!(
                    formatter,
                    "sessionlog: no matching session for {reference:?}"
                )
            }
            Self::Ambiguous(reference) => {
                write!(
                    formatter,
                    "sessionlog: session id prefix {reference:?} matches more than one session"
                )
            }
            Self::ReadOnly => write!(formatter, "sessionlog: session is open read-only"),
            Self::Degraded(reason) => write!(formatter, "sessionlog: recording stopped: {reason}"),
            Self::Closed => write!(formatter, "sessionlog: writer is closed"),
        }
    }
}

impl StdError for SessionError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for SessionError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for SessionError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

pub type Result<T> = std::result::Result<T, SessionError>;

/// The first JSONL line of a pi v3 session.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Header {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub version: u32,
    pub id: String,
    // pi validates only `type` and `id` on a header; sessions written by other
    // clients or hand-edited files may omit the rest and must still be found.
    #[serde(default)]
    pub timestamp: String,
    #[serde(default)]
    pub cwd: String,
    #[serde(
        rename = "parentSession",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub parent_session: Option<String>,
}

impl Header {
    pub fn new(
        id: impl Into<String>,
        cwd: impl Into<String>,
        parent_session: Option<String>,
    ) -> Self {
        Self {
            kind: HEADER_TYPE.to_owned(),
            version: FORMAT_VERSION,
            id: id.into(),
            timestamp: now(),
            cwd: cwd.into(),
            parent_session,
        }
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut header = self.clone();
        header.kind = HEADER_TYPE.to_owned();
        if header.version == 0 {
            header.version = FORMAT_VERSION;
        }
        let mut encoded = serde_json::to_vec(&header)?;
        encoded.push(b'\n');
        Ok(encoded)
    }

    pub fn decode(line: &[u8]) -> Result<Self> {
        let mut header: Self = serde_json::from_slice(line)?;
        if header.kind != HEADER_TYPE {
            return Err(SessionError::InvalidHeader(
                "type is not \"session\"".to_owned(),
            ));
        }
        if header.id.is_empty() {
            return Err(SessionError::InvalidHeader("id is missing".to_owned()));
        }
        if header.version == 0 {
            header.version = 1;
        }
        if header.version > FORMAT_VERSION {
            return Err(SessionError::VersionTooNew(header.version));
        }
        Ok(header)
    }
}

/// One append-only JSONL session entry. `parentId` remains present as `null`
/// for roots, because pi's parser requires the field rather than treating an
/// omitted field as a root.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Entry {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub id: String,
    #[serde(rename = "parentId", default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub timestamp: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<Value>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub summary: String,
    #[serde(
        rename = "firstKeptEntryId",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub first_kept_entry_id: String,
    #[serde(rename = "tokensBefore", default, skip_serializing_if = "is_zero")]
    pub tokens_before: u64,
    #[serde(rename = "fromId", default, skip_serializing_if = "String::is_empty")]
    pub from_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,

    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub provider: String,
    #[serde(rename = "modelId", default, skip_serializing_if = "String::is_empty")]
    pub model_id: String,
    #[serde(
        rename = "thinkingLevel",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub thinking_level: String,

    #[serde(
        rename = "customType",
        default,
        skip_serializing_if = "String::is_empty"
    )]
    pub custom_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display: Option<bool>,

    #[serde(rename = "targetId", default, skip_serializing_if = "String::is_empty")]
    pub target_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
}

impl Entry {
    pub fn message(message: &llm::Message) -> Result<Self> {
        Ok(Self {
            kind: TYPE_MESSAGE.to_owned(),
            message: Some(serde_json::to_value(message)?),
            ..Self::default()
        })
    }

    pub fn decode(line: &[u8]) -> Result<Self> {
        Self::decode_with_id(line, true)
    }

    fn decode_with_id(line: &[u8], require_id: bool) -> Result<Self> {
        let entry: Self = serde_json::from_slice(line)?;
        if entry.kind.is_empty() {
            return Err(SessionError::InvalidEntry("type is missing".to_owned()));
        }
        if entry.kind == HEADER_TYPE {
            return Err(SessionError::InvalidEntry(
                "is a second session header".to_owned(),
            ));
        }
        if require_id && entry.id.is_empty() {
            return Err(SessionError::InvalidEntry("id is missing".to_owned()));
        }
        Ok(entry)
    }

    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut encoded = serde_json::to_vec(self)?;
        encoded.push(b'\n');
        Ok(encoded)
    }
}

/// In-memory index and branch projector for one session file.
#[derive(Clone, Debug, Default)]
pub struct Tree {
    entries: HashMap<String, Entry>,
    /// Verbatim lines, retained only for a tree built by [`Tree::retaining_raw`]:
    /// forking is the one consumer, and holding a second copy of every entry
    /// for the lifetime of a live writer doubled its memory for nothing.
    raw: Option<HashMap<String, Vec<u8>>>,
    order: Vec<String>,
    labels: HashMap<String, String>,
    name: String,
    leaf_id: Option<String>,
}

impl Tree {
    pub fn new() -> Self {
        Self::default()
    }

    /// A tree that also keeps every entry's original bytes.
    pub fn retaining_raw() -> Self {
        Self {
            raw: Some(HashMap::new()),
            ..Self::default()
        }
    }

    pub fn add(&mut self, entry: Entry) -> Result<()> {
        self.add_raw(entry, &[])
    }

    /// Adds an entry along with its source line. The line is kept only when
    /// this tree retains raw bytes.
    pub fn add_raw(&mut self, entry: Entry, raw: &[u8]) -> Result<()> {
        if self.entries.contains_key(&entry.id) {
            return Err(SessionError::DuplicateEntryId(entry.id));
        }
        let id = entry.id.clone();
        if entry.kind == TYPE_LABEL {
            match entry.label.as_deref().filter(|label| !label.is_empty()) {
                Some(label) => {
                    self.labels
                        .insert(entry.target_id.clone(), label.to_owned());
                }
                None => {
                    self.labels.remove(&entry.target_id);
                }
            }
        } else if entry.kind == TYPE_SESSION_INFO {
            self.name = entry.name.clone();
        }
        if let Some(lines) = self.raw.as_mut() {
            lines.insert(id.clone(), raw.to_vec());
        }
        self.order.push(id.clone());
        self.entries.insert(id.clone(), entry);
        self.leaf_id = Some(id);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn has(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    pub fn entry(&self, id: &str) -> Option<&Entry> {
        self.entries.get(id)
    }

    fn entry_mut(&mut self, id: &str) -> Option<&mut Entry> {
        self.entries.get_mut(id)
    }

    /// The original line of an entry, when this tree retains raw bytes.
    pub fn raw(&self, id: &str) -> Option<&[u8]> {
        self.raw.as_ref()?.get(id).map(Vec::as_slice)
    }

    /// Counts direct children of every entry in one pass, for callers that
    /// would otherwise ask [`Tree::children`] once per entry on a path.
    pub fn child_counts(&self) -> HashMap<&str, usize> {
        let mut counts = HashMap::new();
        for entry in self.entries.values() {
            if let Some(parent) = entry.parent_id.as_deref() {
                *counts.entry(parent).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn all(&self) -> Vec<&Entry> {
        self.order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .collect()
    }

    pub fn children(&self, parent: Option<&str>) -> Vec<&Entry> {
        self.order
            .iter()
            .filter_map(|id| self.entries.get(id))
            .filter(|entry| entry.parent_id.as_deref() == parent)
            .collect()
    }

    pub fn leaf(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn set_leaf(&mut self, id: impl Into<String>) -> Result<()> {
        let id = id.into();
        if !self.has(&id) {
            return Err(SessionError::MissingEntry(id));
        }
        self.leaf_id = Some(id);
        Ok(())
    }

    pub fn path(&self, leaf: Option<&str>) -> Vec<&Entry> {
        let Some(mut current) = leaf.or(self.leaf()) else {
            return Vec::new();
        };
        let mut reversed = Vec::new();
        let mut visited = HashMap::new();
        while visited.insert(current.to_owned(), ()).is_none() {
            let Some(entry) = self.entry(current) else {
                break;
            };
            reversed.push(entry);
            let Some(parent) = entry.parent_id.as_deref() else {
                break;
            };
            current = parent;
        }
        reversed.reverse();
        reversed
    }

    /// Projects the model context by honoring the newest reset or compaction
    /// marker on the selected branch.
    pub fn context_path(&self, leaf: Option<&str>) -> Vec<&Entry> {
        let path = self.path(leaf);
        let mut reset_index = None;
        let mut compaction_index = None;
        for (index, entry) in path.iter().enumerate() {
            match entry.kind.as_str() {
                TYPE_TRANSCRIPT_RESET => reset_index = Some(index),
                TYPE_COMPACTION => compaction_index = Some(index),
                _ => {}
            }
        }
        if reset_index > compaction_index {
            return path[reset_index.expect("compares as greater") + 1..].to_vec();
        }
        let Some(compaction_index) = compaction_index else {
            return path;
        };

        let compaction = path[compaction_index];
        let mut projected = vec![compaction];
        if let Some(first_kept) = path[..compaction_index]
            .iter()
            .position(|entry| entry.id == compaction.first_kept_entry_id)
        {
            projected.extend_from_slice(&path[first_kept..compaction_index]);
        }
        projected.extend_from_slice(&path[compaction_index + 1..]);
        projected
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn label(&self, id: &str) -> Option<&str> {
        self.labels.get(id).map(String::as_str)
    }

    pub fn labels(&self) -> &HashMap<String, String> {
        &self.labels
    }

    pub fn has_assistant_message(&self) -> bool {
        self.entries.values().any(|entry| {
            entry.kind == TYPE_MESSAGE
                && entry
                    .message
                    .as_ref()
                    .and_then(|message| message.get("role"))
                    .and_then(Value::as_str)
                    == Some("assistant")
        })
    }
}

/// Report of a load's recoverable problems.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LoadReport {
    pub skipped_lines: usize,
    pub warnings: Vec<String>,
    pub repaired_tail: bool,
    pub unterminated_tail: bool,
    /// This session was read from an older pi format and upgraded in memory.
    /// The source file is never rewritten.
    pub migrated: bool,
    pub source_version: u32,
}

/// In-memory migration for pre-v3 pi session files. Older files are never
/// rewritten: callers can inspect or fork them, but must not append in place.
struct Migrator {
    version: u32,
    previous: Option<String>,
    taken: HashSet<String>,
    pending_compaction_index: HashMap<String, usize>,
    entry_index: Vec<String>,
}

impl Migrator {
    fn new(version: u32) -> Self {
        Self {
            version,
            previous: None,
            taken: HashSet::new(),
            pending_compaction_index: HashMap::new(),
            entry_index: Vec::new(),
        }
    }

    fn needed(&self) -> bool {
        self.version < FORMAT_VERSION
    }

    fn apply(&mut self, entry: &mut Entry, raw: &[u8]) {
        if self.version < 2 {
            if entry.id.is_empty() {
                let id = new_unique_entry_id(|candidate| self.taken.contains(candidate));
                self.taken.insert(id.clone());
                entry.id = id.clone();
                entry.parent_id = self.previous.clone();
                self.previous = Some(id.clone());

                if entry.kind == TYPE_COMPACTION
                    && let Ok(Value::Object(fields)) = serde_json::from_slice::<Value>(raw)
                    && let Some(index) = fields
                        .get("firstKeptEntryIndex")
                        .and_then(Value::as_u64)
                        .and_then(|index| usize::try_from(index).ok())
                {
                    self.pending_compaction_index.insert(id.clone(), index);
                }
                self.entry_index.push(id);
            } else {
                self.taken.insert(entry.id.clone());
                self.previous = Some(entry.id.clone());
                self.entry_index.push(entry.id.clone());
            }
        }

        if self.version < 3 && entry.kind == TYPE_MESSAGE {
            rename_legacy_message_role(&mut entry.message);
        }
    }

    fn finish(self, tree: &mut Tree) {
        for (id, index) in self.pending_compaction_index {
            // v1 counts the header as index zero.
            let Some(target) = index.checked_sub(1) else {
                continue;
            };
            let Some(first_kept_entry_id) = self.entry_index.get(target) else {
                continue;
            };
            if let Some(entry) = tree.entry_mut(&id) {
                entry.first_kept_entry_id = first_kept_entry_id.clone();
            }
        }
    }
}

fn rename_legacy_message_role(message: &mut Option<Value>) {
    let Some(Value::Object(fields)) = message else {
        return;
    };
    if fields.get("role").and_then(Value::as_str) == Some("hookMessage") {
        fields.insert("role".to_owned(), Value::String("custom".to_owned()));
    }
}

struct ReadLine {
    bytes: Vec<u8>,
    consumed: u64,
    complete: bool,
    too_large: bool,
}

fn read_line_bounded(reader: &mut impl BufRead, limit: usize) -> io::Result<ReadLine> {
    let mut bytes = Vec::new();
    let mut consumed = 0_u64;
    let mut complete = false;

    loop {
        let (take, ends_line, retained) = {
            let buffer = reader.fill_buf()?;
            if buffer.is_empty() {
                break;
            }
            let newline = buffer.iter().position(|byte| *byte == b'\n');
            let take = newline.map_or(buffer.len(), |index| index + 1);
            let available = limit.saturating_add(1).saturating_sub(bytes.len());
            (
                take,
                newline.is_some(),
                buffer[..take.min(available)].to_vec(),
            )
        };
        bytes.extend_from_slice(&retained);
        consumed += take as u64;
        reader.consume(take);
        if ends_line {
            complete = true;
            break;
        }
    }

    Ok(ReadLine {
        too_large: bytes.len() > limit,
        bytes,
        consumed,
        complete,
    })
}

/// A root collection of pi-compatible sessions.
#[derive(Clone, Debug)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dir_name(cwd: impl AsRef<Path>) -> String {
        let resolved = absolute_path(cwd.as_ref());
        let trimmed = resolved
            .to_string_lossy()
            .trim_start_matches(['/', '\\'])
            .replace(['/', '\\', ':'], "-");
        format!("--{trimmed}--")
    }

    pub fn directory(&self, cwd: impl AsRef<Path>) -> PathBuf {
        self.root.join(Self::dir_name(cwd))
    }

    pub fn create(&self, cwd: impl AsRef<Path>) -> Result<Writer> {
        self.create_with_id(cwd, None, new_session_id())
    }

    pub fn create_with_id(
        &self,
        cwd: impl AsRef<Path>,
        parent_session: Option<String>,
        id: impl Into<String>,
    ) -> Result<Writer> {
        let id = id.into();
        validate_session_id(&id)?;
        let cwd = absolute_path(cwd.as_ref());
        let directory = self.directory(&cwd);
        create_private_dir(&directory)?;

        let header = Header::new(id.clone(), cwd.to_string_lossy(), parent_session);
        let path = directory.join(format!(
            "{}_{}.jsonl",
            filename_stamp(&header.timestamp),
            id
        ));
        let claim = claim(&path)?;
        let mut options = OpenOptions::new();
        options.write(true).append(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(&path)?;
        let header_line = match header.encode() {
            Ok(header_line) => header_line,
            Err(error) => {
                drop(file);
                let _ = fs::remove_file(&path);
                return Err(error);
            }
        };
        if let Err(error) = file.write_all(&header_line) {
            drop(file);
            let _ = fs::remove_file(&path);
            return Err(error.into());
        }
        Ok(Writer {
            file: Some(file),
            path,
            header,
            tree: Tree::new(),
            size: header_line.len() as u64,
            keep: false,
            claim: Some(claim),
            read_only: false,
            degraded: None,
            closed: false,
        })
    }

    pub fn load(&self, path: impl AsRef<Path>) -> Result<(Tree, Header, LoadReport)> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let (tree, header, report, _) = read_into(&mut file, false)?;
        Ok((tree, header, report))
    }

    /// Like [`Store::load`], but the tree keeps every entry's original bytes
    /// so a fork can copy lines verbatim.
    fn load_with_raw(&self, path: &Path) -> Result<(Tree, Header, LoadReport)> {
        let mut file = File::open(path)?;
        let (tree, header, report, _) = read_into(&mut file, true)?;
        Ok((tree, header, report))
    }

    /// Opens an existing v3 session for append while holding its on-disk
    /// claim. A pre-v3 session can be read or forked, but cannot safely be
    /// appended in place because its generated identities are not durable.
    pub fn attach(&self, path: impl AsRef<Path>) -> Result<(Writer, LoadReport)> {
        let path = path.as_ref();
        let claim = claim(path)?;
        // O_APPEND makes every write land at the current end of file even if
        // another writer got in, so a takeover can never produce torn lines
        // in the middle of the log.
        let mut file = OpenOptions::new().read(true).append(true).open(path)?;
        let (tree, header, mut report, offset) = read_into(&mut file, false)?;
        if report.migrated {
            return Err(SessionError::LegacyFormat(report.source_version));
        }

        let current_size = file.metadata()?.len();
        if current_size > offset {
            // An append-only handle cannot truncate on Windows (it carries no
            // write-data access), so the cut goes through a plain write
            // handle; the append handle keeps writing at the new end.
            OpenOptions::new().write(true).open(path)?.set_len(offset)?;
            report.repaired_tail = true;
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut size = offset;
        if report.unterminated_tail {
            file.write_all(b"\n")?;
            size += 1;
        }

        Ok((
            Writer {
                file: Some(file),
                path: path.to_path_buf(),
                header,
                tree,
                size,
                // The file already existed before this process touched it, so
                // it is never this writer's to discard: a resumed session that
                // is closed again without a new reply must survive intact.
                keep: true,
                claim: Some(claim),
                read_only: false,
                degraded: None,
                closed: false,
            },
            report,
        ))
    }

    /// Opens a session for read-only inspection without claiming it.
    pub fn open(&self, path: impl AsRef<Path>) -> Result<(Writer, LoadReport)> {
        let path = path.as_ref();
        let (tree, header, report) = self.load(path)?;
        let size = fs::metadata(path)?.len();
        Ok((
            Writer {
                file: None,
                path: path.to_path_buf(),
                header,
                tree,
                size,
                keep: true,
                claim: None,
                read_only: true,
                degraded: None,
                closed: false,
            },
            report,
        ))
    }

    /// Lists sessions belonging to one workspace by default, or every known
    /// workspace when `all_workspaces` is selected.
    pub fn list(&self, cwd: impl AsRef<Path>, options: ListOptions) -> Result<Vec<SessionInfo>> {
        let target_cwd = absolute_path(cwd.as_ref());
        let mut sessions = Vec::new();
        for directory in self.shard_directories(&target_cwd, options.all_workspaces)? {
            for path in session_files(&directory)? {
                let Ok(info) = self.describe(&path, options.with_text) else {
                    // A malformed or inaccessible file must not hide usable
                    // sessions in the same directory.
                    continue;
                };
                if options.all_workspaces || cwd_matches(&info.cwd, &target_cwd) {
                    sessions.push(info);
                }
            }
        }
        sessions.sort_by(|left, right| {
            right
                .modified
                .cmp(&left.modified)
                .then_with(|| right.id.cmp(&left.id))
        });
        if options.limit > 0 {
            sessions.truncate(options.limit);
        }
        Ok(sessions)
    }

    /// Finds the newest session of a workspace from headers and mtimes alone,
    /// the way pi's `findMostRecentSession` does, so a `-continue` start does
    /// not parse every transcript in the shard.
    pub fn most_recent(&self, cwd: impl AsRef<Path>) -> Result<Option<SessionInfo>> {
        let target_cwd = absolute_path(cwd.as_ref());
        let mut candidates = Vec::new();
        for path in session_files(&self.directory(&target_cwd))? {
            let Ok(header) = read_header(&path) else {
                continue;
            };
            if !cwd_matches(&header.cwd, &target_cwd) {
                continue;
            }
            let modified = fs::metadata(&path)
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            candidates.push((modified, header.id, path));
        }
        candidates.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
        // Only the winner is parsed in full. A newest file whose body is
        // unreadable falls through to the next one, as `list` would skip it.
        Ok(candidates
            .into_iter()
            .find_map(|(_, _, path)| self.describe(&path, false).ok()))
    }

    /// Resolves an explicit path, exact ID, or unambiguous ID prefix.
    pub fn resolve(&self, cwd: impl AsRef<Path>, reference: &str) -> Result<SessionInfo> {
        if reference.is_empty() {
            return Err(SessionError::NotFound(reference.to_owned()));
        }
        if reference.ends_with(".jsonl") || reference.contains(['/', '\\']) {
            return self
                .describe(&absolute_path(Path::new(reference)), false)
                .map_err(|_| SessionError::NotFound(reference.to_owned()));
        }

        let cwd = absolute_path(cwd.as_ref());
        for all_workspaces in [false, true] {
            let mut files = Vec::new();
            for directory in self.shard_directories(&cwd, all_workspaces)? {
                files.extend(session_files(&directory)?);
            }
            // Files are named `<stamp>_<id>.jsonl`, so an exact id is usually
            // settled by one header read instead of a scan of the shard.
            if let Some(path) = files
                .iter()
                .find(|path| filename_session_id(path) == Some(reference))
                && read_header(path).is_ok_and(|header| header.id == reference)
            {
                return self.describe(path, false);
            }
            // The header is the authority on ids, and it is all a prefix
            // match needs; transcripts are parsed only for the chosen file.
            let mut matches = Vec::new();
            for path in &files {
                let Ok(header) = read_header(path) else {
                    continue;
                };
                if header.id == reference {
                    return self.describe(path, false);
                }
                if header.id.starts_with(reference) {
                    matches.push(path);
                }
            }
            match matches.as_slice() {
                [] => {}
                [path] => return self.describe(path, false),
                _ => return Err(SessionError::Ambiguous(reference.to_owned())),
            }
        }
        Err(SessionError::NotFound(reference.to_owned()))
    }

    fn shard_directories(&self, cwd: &Path, all_workspaces: bool) -> Result<Vec<PathBuf>> {
        if !all_workspaces {
            return Ok(vec![self.directory(cwd)]);
        }
        match fs::read_dir(&self.root) {
            Ok(entries) => Ok(entries
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                .map(|entry| entry.path())
                .collect()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(error) => Err(error.into()),
        }
    }

    /// Removes a session only after it has been claimed, so a live writer
    /// cannot continue appending to an unlinked file.
    pub fn remove(&self, info: &SessionInfo) -> Result<()> {
        let _claim = claim(&info.path)?;
        fs::remove_file(&info.path)?;
        Ok(())
    }

    /// Copies one selected branch into a new v3 session. For an older source,
    /// migrated entries are encoded with their durable generated identities.
    pub fn fork(
        &self,
        source: &SessionInfo,
        at: Option<&str>,
        target_cwd: impl AsRef<Path>,
    ) -> Result<Writer> {
        let (tree, header, report) = self.load_with_raw(&source.path)?;
        let path = tree.path(at);
        if path.is_empty()
            && let Some(at) = at
        {
            return Err(SessionError::MissingEntry(at.to_owned()));
        }
        let target_cwd = target_cwd.as_ref();
        let target_cwd = if target_cwd.as_os_str().is_empty() {
            Path::new(&header.cwd)
        } else {
            target_cwd
        };
        // pi records the source file rather than its id, which stays
        // traceable even when ids repeat across workspaces or imports.
        let parent_session = absolute_path(&source.path).to_string_lossy().into_owned();
        let mut writer = self.create_with_id(target_cwd, Some(parent_session), new_session_id())?;
        let on_path = path
            .iter()
            .map(|entry| entry.id.as_str())
            .collect::<HashSet<_>>();

        for entry in tree.all() {
            let copy = on_path.contains(entry.id.as_str())
                || (entry.kind == TYPE_LABEL && on_path.contains(entry.target_id.as_str()));
            if !copy
                || (entry.kind == TYPE_LABEL
                    && entry
                        .parent_id
                        .as_deref()
                        .is_some_and(|parent| !on_path.contains(parent)))
            {
                continue;
            }
            let line = if report.migrated {
                entry.encode()?
            } else {
                tree.raw(&entry.id)
                    .ok_or_else(|| SessionError::MissingEntry(entry.id.clone()))?
                    .to_vec()
            };
            if let Err(error) = writer.append_raw(entry.clone(), &line) {
                let _ = writer.close();
                return Err(error);
            }
        }
        if let Some(id) = last_conversation_entry(&writer.tree) {
            writer.set_leaf(id)?;
        }
        writer.keep();
        writer.sync()?;
        Ok(writer)
    }

    fn describe(&self, path: &Path, with_text: bool) -> Result<SessionInfo> {
        let metadata = fs::metadata(path)?;
        let (tree, header, _) = self.load(path)?;
        let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let mut info = SessionInfo {
            id: header.id,
            path: path.to_path_buf(),
            cwd: header.cwd,
            name: tree.name().to_owned(),
            first_message: String::new(),
            created: parse_timestamp(&header.timestamp)
                .or_else(|| system_time_to_offset_datetime(modified)),
            modified,
            messages: 0,
            cleared: 0,
            size: metadata.len(),
            search_text: String::new(),
            locked: false,
            owner: LockOwner::default(),
        };
        let mut search_text = String::new();
        for entry in tree.all() {
            if entry.kind == TYPE_TRANSCRIPT_RESET {
                info.cleared += info.messages;
                info.messages = 0;
                info.first_message.clear();
                continue;
            }
            if entry.kind != TYPE_MESSAGE {
                continue;
            }
            info.messages += 1;
            let text = entry
                .message
                .as_ref()
                .map_or_else(String::new, message_text);
            let role = entry
                .message
                .as_ref()
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if info.first_message.is_empty() && role == "user" && !text.is_empty() {
                info.first_message = first_line(&text, 120);
            }
            if with_text && search_text.len() < MAX_SEARCH_TEXT_BYTES && !text.is_empty() {
                let remaining = MAX_SEARCH_TEXT_BYTES - search_text.len();
                search_text.push_str(truncate_utf8(&text, remaining));
                search_text.push('\n');
            }
        }
        info.search_text = search_text;
        (info.locked, info.owner) = probe_claim(path);
        Ok(info)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ListOptions {
    /// Scan all workspace shards instead of only the current workspace.
    pub all_workspaces: bool,
    /// Maximum number of newest sessions to return. Zero means unlimited.
    pub limit: usize,
    /// Include bounded transcript text for resume-picker search.
    pub with_text: bool,
}

/// One session's metadata as shown by the picker and consumed by resume.
#[derive(Clone, Debug)]
pub struct SessionInfo {
    pub id: String,
    pub path: PathBuf,
    pub cwd: String,
    pub name: String,
    pub first_message: String,
    pub created: Option<OffsetDateTime>,
    pub modified: SystemTime,
    pub messages: usize,
    pub cleared: usize,
    pub size: u64,
    pub search_text: String,
    pub locked: bool,
    pub owner: LockOwner,
}

impl SessionInfo {
    pub fn title(&self) -> &str {
        if !self.name.is_empty() {
            &self.name
        } else if !self.first_message.is_empty() {
            &self.first_message
        } else {
            &self.id
        }
    }

    pub fn short_id(&self) -> &str {
        truncate_utf8(&self.id, SHORT_ID_LENGTH)
    }
}

/// Returns unique, readable ID prefixes for a list of sessions. Ids written
/// by other clients are not restricted to ASCII, so prefixes are cut at
/// character boundaries.
pub fn short_ids(sessions: &[SessionInfo]) -> Vec<String> {
    let mut length = SHORT_ID_LENGTH;
    loop {
        let mut prefixes = HashSet::with_capacity(sessions.len());
        let mut collision = false;
        let mut longest = 0;
        for session in sessions {
            longest = longest.max(session.id.len());
            if !prefixes.insert(truncate_utf8(&session.id, length)) {
                collision = true;
                break;
            }
        }
        if !collision || length >= longest {
            return sessions
                .iter()
                .map(|session| truncate_utf8(&session.id, length).to_owned())
                .collect();
        }
        length += 4;
    }
}

/// An append-only writer. It owns the process claim so every successful append
/// has exclusive access to its session file.
pub struct Writer {
    file: Option<File>,
    path: PathBuf,
    header: Header,
    tree: Tree,
    size: u64,
    keep: bool,
    claim: Option<LockClaim>,
    read_only: bool,
    degraded: Option<String>,
    closed: bool,
}

impl fmt::Debug for Writer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Writer")
            .field("path", &self.path)
            .field("header", &self.header)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

impl Writer {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn id(&self) -> &str {
        &self.header.id
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn read_only(&self) -> bool {
        self.read_only
    }

    pub fn recording(&self) -> bool {
        !self.read_only
            && !self.closed
            && self.degraded.is_none()
            && self.file.is_some()
            && !self.claim_lost()
    }

    pub fn degraded(&self) -> Option<&str> {
        self.degraded.as_deref()
    }

    /// The live tree. Prefer this over [`Writer::snapshot`] for lookups that
    /// do not need to outlive the borrow.
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    pub fn snapshot(&self) -> Tree {
        self.tree.clone()
    }

    fn claim_lost(&self) -> bool {
        self.claim.as_ref().is_some_and(LockClaim::lost)
    }

    pub fn leaf(&self) -> Option<&str> {
        self.tree.leaf()
    }

    pub fn keep(&mut self) {
        self.keep = true;
    }

    /// Appends an entry at the current write head.
    pub fn append(&mut self, entry: Entry) -> Result<String> {
        let parent = self.tree.leaf().map(str::to_owned);
        self.append_at(parent.as_deref(), entry)
    }

    /// Appends an entry to a specific parent, which is used when recreating a
    /// branch without first moving the write head.
    pub fn append_at(&mut self, parent: Option<&str>, mut entry: Entry) -> Result<String> {
        self.ensure_writable()?;
        entry.id = new_entry_id(&self.tree);
        entry.parent_id = parent.map(str::to_owned);
        entry.timestamp = now();
        let line = entry.encode()?;
        if line.len() > MAX_ENTRY_BYTES {
            return Err(SessionError::EntryTooLarge(line.len()));
        }
        let new_size = self.size + line.len() as u64;
        if new_size > MAX_SESSION_BYTES {
            self.stop("the session file reached its maximum size");
            return Err(SessionError::SessionTooLarge(new_size));
        }
        self.write_line(&line)?;
        self.size = new_size;
        let id = entry.id.clone();
        self.tree.add(entry)?;
        Ok(id)
    }

    pub fn set_leaf(&mut self, id: impl Into<String>) -> Result<()> {
        self.tree.set_leaf(id)
    }

    pub fn sync(&mut self) -> Result<()> {
        if self.read_only || self.closed {
            return Ok(());
        }
        if let Some(reason) = &self.degraded {
            return Err(SessionError::Degraded(reason.clone()));
        }
        let Some(file) = self.file.as_mut() else {
            return Err(SessionError::Closed);
        };
        if let Err(error) = file.sync_all() {
            self.stop(&error.to_string());
            return Err(error.into());
        }
        Ok(())
    }

    /// Finalizes a session. Sessions without an assistant reply are discarded
    /// unless the caller explicitly retained one (for example, a user-created
    /// fork that intentionally ends at a prompt).
    pub fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        let mut first_error = None;
        if let Some(mut file) = self.file.take()
            && !self.read_only
            && self.degraded.is_none()
            && let Err(error) = file.flush().and_then(|()| file.sync_all())
        {
            first_error = Some(SessionError::Io(error));
        }
        // A file whose claim was taken over now belongs to another writer;
        // unlinking it would leave that process appending to a dead inode.
        if !self.read_only
            && !self.keep
            && !self.claim_lost()
            && !self.tree.has_assistant_message()
            && let Err(error) = fs::remove_file(&self.path)
            && error.kind() != io::ErrorKind::NotFound
            && first_error.is_none()
        {
            first_error = Some(SessionError::Io(error));
        }
        self.claim.take();
        first_error.map_or(Ok(()), Err)
    }

    fn ensure_writable(&mut self) -> Result<()> {
        if self.closed {
            return Err(SessionError::Closed);
        }
        if self.read_only {
            return Err(SessionError::ReadOnly);
        }
        // Re-check the claim before every append rather than trusting the
        // last heartbeat: a takeover during a long stall must stop this
        // writer at its next line, not two seconds later.
        if self.claim.as_ref().is_some_and(|claim| !claim.verify()) {
            self.stop("another process took over this session");
        }
        if let Some(reason) = &self.degraded {
            return Err(SessionError::Degraded(reason.clone()));
        }
        if self.file.is_none() {
            return Err(SessionError::Closed);
        }
        Ok(())
    }

    fn append_raw(&mut self, entry: Entry, raw: &[u8]) -> Result<()> {
        self.ensure_writable()?;
        let line = if raw.ends_with(b"\n") {
            raw.to_vec()
        } else {
            [raw, b"\n"].concat()
        };
        if line.len() > MAX_ENTRY_BYTES {
            return Err(SessionError::EntryTooLarge(line.len()));
        }
        let new_size = self.size + line.len() as u64;
        if new_size > MAX_SESSION_BYTES {
            self.stop("the session file reached its maximum size");
            return Err(SessionError::SessionTooLarge(new_size));
        }
        self.write_line(&line)?;
        self.size = new_size;
        self.tree.add(entry)
    }

    fn write_line(&mut self, line: &[u8]) -> Result<()> {
        let original_size = self.size;
        let write_result = self
            .file
            .as_mut()
            .ok_or(SessionError::Closed)
            .and_then(|file| file.write_all(line).map_err(SessionError::Io));
        if let Err(error) = write_result {
            if let Some(file) = self.file.as_mut() {
                let _ = file.set_len(original_size);
                let _ = file.seek(SeekFrom::Start(original_size));
            }
            self.stop(&error.to_string());
            return Err(error);
        }
        Ok(())
    }

    fn stop(&mut self, reason: &str) {
        if self.degraded.is_none() {
            self.degraded = Some(reason.to_owned());
        }
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// A sidecar `.lock` claim compatible with the Go implementation. It is an
/// advisory claim: ownership is a token, so a stale writer cannot remove a
/// lock reclaimed by another process.
struct LockClaim {
    path: PathBuf,
    token: String,
    /// Set once the lock file stops carrying this claim's token. A takeover is
    /// permanent: the successor owns the file from then on.
    lost: Arc<AtomicBool>,
    stop: Option<mpsc::Sender<()>>,
    heartbeat: Option<JoinHandle<()>>,
}

impl LockClaim {
    fn new(path: PathBuf, token: String) -> Self {
        let (stop, receiver) = mpsc::channel();
        let lost = Arc::new(AtomicBool::new(false));
        let heartbeat_path = path.clone();
        let heartbeat_token = token.clone();
        let heartbeat_lost = lost.clone();
        let heartbeat = thread::spawn(move || {
            loop {
                match receiver.recv_timeout(LOCK_HEARTBEAT) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        // A lost lock is never touched again: refreshing it
                        // would keep a successor's claim alive on its behalf.
                        if !still_held(&heartbeat_path, &heartbeat_token, &heartbeat_lost) {
                            break;
                        }
                        let _ = OpenOptions::new()
                            .write(true)
                            .open(&heartbeat_path)
                            .and_then(|file| {
                                file.set_times(fs::FileTimes::new().set_modified(SystemTime::now()))
                            });
                    }
                }
            }
        });
        Self {
            path,
            token,
            lost,
            stop: Some(stop),
            heartbeat: Some(heartbeat),
        }
    }

    /// Re-reads the lock file and reports whether this claim still owns it.
    fn verify(&self) -> bool {
        still_held(&self.path, &self.token, &self.lost)
    }

    fn lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }
}

fn still_held(path: &Path, token: &str, lost: &AtomicBool) -> bool {
    if lost.load(Ordering::Relaxed) {
        return false;
    }
    match fs::read(path) {
        Ok(contents) if contents == token.as_bytes() => true,
        // Rewritten or gone: a waiter declared this claim stale and took it.
        // Any other read error is transient and proves nothing.
        Ok(_) => {
            lost.store(true, Ordering::Relaxed);
            false
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            lost.store(true, Ordering::Relaxed);
            false
        }
        Err(_) => true,
    }
}

impl Drop for LockClaim {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(heartbeat) = self.heartbeat.take() {
            let _ = heartbeat.join();
        }
        if fs::read(&self.path).is_ok_and(|contents| contents.as_slice() == self.token.as_bytes()) {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn lock_path(path: &Path) -> PathBuf {
    let mut lock_path = path.as_os_str().to_os_string();
    lock_path.push(".lock");
    lock_path.into()
}

/// Claims a session that already exists on disk (or whose directory was just
/// created by this store), so no directory is created or re-moded here.
fn claim(path: &Path) -> Result<LockClaim> {
    let path = lock_path(path);
    let token = format!("{} {}\n", std::process::id(), Uuid::now_v7());
    loop {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&path) {
            Ok(mut file) => {
                if let Err(error) = file.write_all(token.as_bytes()) {
                    drop(file);
                    let _ = fs::remove_file(&path);
                    return Err(error.into());
                }
                return Ok(LockClaim::new(path, token));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                if lock_is_stale(&path) {
                    match fs::remove_file(&path) {
                        Ok(()) => continue,
                        Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                        Err(error) => return Err(error.into()),
                    }
                }
                return Err(SessionError::Busy(read_lock_owner(&path)));
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// A lock is stale once its heartbeat has stopped, or sooner when the process
/// that wrote it is known to be gone on this host: a crash followed by an
/// immediate relaunch should not leave the user silently read-only for the
/// rest of the heartbeat window.
fn lock_is_stale(path: &Path) -> bool {
    let heartbeat_stopped = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|elapsed| elapsed > LOCK_STALE);
    heartbeat_stopped || read_lock_owner(path).pid.is_some_and(pid_is_dead)
}

#[cfg(unix)]
fn pid_is_dead(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    // Signal zero delivers nothing. ESRCH is the one answer that proves the
    // pid is unused; EPERM means it is alive under another user, and a pid
    // from another host cannot be judged at all, so only ESRCH counts.
    let result = unsafe { kill(pid, 0) };
    result == -1 && io::Error::last_os_error().raw_os_error() == Some(ESRCH)
}

#[cfg(not(unix))]
fn pid_is_dead(_pid: u32) -> bool {
    false
}

fn read_lock_owner(path: &Path) -> LockOwner {
    let since = fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok();
    let pid = fs::read_to_string(path).ok().and_then(|contents| {
        contents
            .split_ascii_whitespace()
            .next()
            .and_then(|pid| pid.parse().ok())
    });
    LockOwner { pid, since }
}

fn probe_claim(path: &Path) -> (bool, LockOwner) {
    let path = lock_path(path);
    if !path.exists() || lock_is_stale(&path) {
        return (false, LockOwner::default());
    }
    (true, read_lock_owner(&path))
}

/// Reads the header line only. Discovery paths use it so that finding or
/// naming a session never pays for parsing its transcript.
fn read_header(path: &Path) -> Result<Header> {
    let file = File::open(path)?;
    let mut reader = BufReader::with_capacity(4 << 10, file);
    let first = read_line_bounded(&mut reader, MAX_ENTRY_BYTES)?;
    if first.consumed == 0 {
        return Err(SessionError::EmptySession);
    }
    if first.too_large {
        return Err(SessionError::InvalidHeader(format!(
            "line exceeds the {MAX_ENTRY_BYTES}-byte entry limit"
        )));
    }
    Header::decode(trim_line(&first.bytes))
}

fn read_into(file: &mut File, keep_raw: bool) -> Result<(Tree, Header, LoadReport, u64)> {
    let metadata = file.metadata()?;
    if metadata.len() > MAX_SESSION_BYTES {
        return Err(SessionError::SessionTooLarge(metadata.len()));
    }
    file.seek(SeekFrom::Start(0))?;
    let mut reader = BufReader::with_capacity(64 << 10, file);
    let first = read_line_bounded(&mut reader, MAX_ENTRY_BYTES)?;
    if first.consumed == 0 {
        return Err(SessionError::EmptySession);
    }
    if first.too_large {
        return Err(SessionError::InvalidHeader(format!(
            "line exceeds the {MAX_ENTRY_BYTES}-byte entry limit"
        )));
    }
    let mut header = Header::decode(trim_line(&first.bytes))?;
    let mut migration = Migrator::new(header.version);
    let mut report = LoadReport {
        source_version: header.version,
        migrated: migration.needed(),
        unterminated_tail: !first.complete,
        ..LoadReport::default()
    };
    if report.migrated {
        report.warnings.push(format!(
            "session is format v{}; it was read as v{FORMAT_VERSION} without being rewritten",
            header.version
        ));
    }

    let mut tree = if keep_raw {
        Tree::retaining_raw()
    } else {
        Tree::new()
    };
    let mut last_added = None::<String>;
    let mut line_number = 1_usize;
    let mut total = first.consumed;
    let mut keep_bytes = total;
    loop {
        let line = read_line_bounded(&mut reader, MAX_ENTRY_BYTES)?;
        if line.consumed == 0 {
            break;
        }
        let line_start = total;
        total += line.consumed;
        line_number += 1;
        if line.too_large {
            report.skipped_lines += 1;
            report.warnings.push(format!(
                "line {line_number} exceeds the {MAX_ENTRY_BYTES}-byte entry limit and was skipped"
            ));
            keep_bytes = if line.complete { total } else { line_start };
            if !line.complete {
                report.repaired_tail = true;
            }
        } else {
            let body = trim_line(&line.bytes);
            if body.is_empty() {
                keep_bytes = total;
            } else {
                let mut entry = match Entry::decode_with_id(body, !migration.needed()) {
                    Ok(entry) => entry,
                    Err(error) if !line.complete => {
                        report.repaired_tail = true;
                        report.warnings.push(format!(
                            "line {line_number} is an unterminated malformed tail and was ignored: {error}"
                        ));
                        break;
                    }
                    Err(error) => {
                        report.skipped_lines += 1;
                        report
                            .warnings
                            .push(format!("line {line_number} was skipped: {error}"));
                        keep_bytes = total;
                        if !line.complete {
                            report.repaired_tail = true;
                            keep_bytes = line_start;
                        }
                        if !line.complete {
                            break;
                        }
                        continue;
                    }
                };
                if migration.needed() {
                    migration.apply(&mut entry, body);
                }
                let mut raw = Cow::Borrowed(line.bytes.as_slice());
                if let Some(parent) = entry.parent_id.as_deref()
                    && !tree.has(parent)
                {
                    entry.parent_id = last_added.clone();
                    // The retained bytes must agree with the repaired parent,
                    // or a fork copies the dangling reference and the copy
                    // reattaches differently on its next load.
                    if keep_raw {
                        raw = Cow::Owned(with_parent_id(body, entry.parent_id.as_deref()));
                    }
                    report.warnings.push(format!(
                        "line {line_number} referenced an entry that is not in the file; it was reattached so earlier conversation stays reachable"
                    ));
                }
                match tree.add_raw(entry.clone(), &raw) {
                    Ok(()) => {
                        last_added = Some(entry.id);
                        if !line.complete {
                            report.unterminated_tail = true;
                        }
                        keep_bytes = total;
                    }
                    Err(error) => {
                        report.skipped_lines += 1;
                        report
                            .warnings
                            .push(format!("line {line_number} was skipped: {error}"));
                        // A rejected line that never got its newline has to be
                        // cut, or the next append is glued onto it and both
                        // lines are lost on the following load.
                        if line.complete {
                            keep_bytes = total;
                        } else {
                            keep_bytes = line_start;
                            report.repaired_tail = true;
                        }
                    }
                }
            }
        }
        if !line.complete {
            break;
        }
    }
    if migration.needed() {
        migration.finish(&mut tree);
        header.version = FORMAT_VERSION;
    }
    Ok((tree, header, report, keep_bytes))
}

fn last_conversation_entry(tree: &Tree) -> Option<String> {
    tree.all()
        .into_iter()
        .rev()
        .find(|entry| !matches!(entry.kind.as_str(), TYPE_LABEL | TYPE_SESSION_INFO))
        .map(|entry| entry.id.clone())
}

fn message_text(message: &Value) -> String {
    let Some(content) = message.get("content") else {
        return String::new();
    };
    if let Some(content) = content.as_str() {
        return content.to_owned();
    }
    content
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| {
            (block.get("type").and_then(Value::as_str) == Some("text"))
                .then(|| block.get("text").and_then(Value::as_str))
                .flatten()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn first_line(text: &str, limit: usize) -> String {
    let text = text.trim();
    let first = text.lines().next().unwrap_or_default().trim();
    let mut characters = first.chars();
    let visible = characters.by_ref().take(limit).collect::<String>();
    if characters.next().is_some() {
        format!("{visible}…")
    } else {
        visible
    }
}

fn truncate_utf8(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

fn parse_timestamp(value: &str) -> Option<OffsetDateTime> {
    OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339).ok()
}

fn system_time_to_offset_datetime(value: SystemTime) -> Option<OffsetDateTime> {
    let duration = value.duration_since(SystemTime::UNIX_EPOCH).ok()?;
    OffsetDateTime::from_unix_timestamp_nanos(duration.as_nanos() as i128).ok()
}

pub fn now() -> String {
    OffsetDateTime::now_utc()
        .format(TIMESTAMP_FORMAT)
        .expect("the static UTC timestamp format is valid")
}

pub fn filename_stamp(timestamp: &str) -> String {
    timestamp.replace([':', '.'], "-")
}

pub fn validate_session_id(id: &str) -> Result<()> {
    let mut characters = id.chars();
    let Some(first) = characters.next() else {
        return Err(SessionError::InvalidSessionId(id.to_owned()));
    };
    if !first.is_ascii_alphanumeric()
        || id
            .chars()
            .last()
            .is_some_and(|character| !character.is_ascii_alphanumeric())
        || !characters.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        })
    {
        return Err(SessionError::InvalidSessionId(id.to_owned()));
    }
    Ok(())
}

fn new_session_id() -> String {
    Uuid::now_v7().to_string()
}

fn new_entry_id(tree: &Tree) -> String {
    new_unique_entry_id(|candidate| tree.has(candidate))
}

fn new_unique_entry_id(taken: impl Fn(&str) -> bool) -> String {
    for _ in 0..100 {
        let encoded = Uuid::now_v7().simple().to_string();
        let id = encoded[encoded.len() - 8..].to_owned();
        if !taken(&id) {
            return id;
        }
    }
    Uuid::now_v7().simple().to_string()
}

/// Rewrites an entry line with a different `parentId`, keeping every other
/// field byte-for-byte so provider data from newer clients survives.
fn with_parent_id(body: &[u8], parent: Option<&str>) -> Vec<u8> {
    let Ok(Value::Object(mut fields)) = serde_json::from_slice::<Value>(body) else {
        return [body, b"\n"].concat();
    };
    fields.insert(
        "parentId".to_owned(),
        parent.map_or(Value::Null, |parent| Value::String(parent.to_owned())),
    );
    let mut line = serde_json::to_vec(&Value::Object(fields)).unwrap_or_else(|_| body.to_vec());
    line.push(b'\n');
    line
}

/// Folds `.` and `..` segments and trailing separators out of a path without
/// touching the filesystem, as pi's `resolvePath` does. Session shards are
/// keyed by the spelled path, so `/w/` and `/w/./x/..` must name one shard.
pub fn clean_path(path: &Path) -> PathBuf {
    let mut cleaned = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => cleaned.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => match cleaned.components().next_back() {
                Some(Component::Normal(_)) => {
                    cleaned.pop();
                }
                // Above the root there is nothing to climb to.
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => cleaned.push(".."),
            },
            Component::Normal(segment) => cleaned.push(segment),
        }
    }
    cleaned
}

/// Resolves a path against the current directory and cleans it, failing only
/// when the current directory itself cannot be read.
pub fn try_absolute_path(path: &Path) -> io::Result<PathBuf> {
    if path.is_absolute() {
        return Ok(clean_path(path));
    }
    Ok(clean_path(&std::env::current_dir()?.join(path)))
}

/// [`try_absolute_path`] with a lexical fallback for callers that cannot
/// report an unreadable current directory.
pub fn absolute_path(path: &Path) -> PathBuf {
    try_absolute_path(path).unwrap_or_else(|_| clean_path(path))
}

/// Compares a header's recorded workspace against a resolved one. A header
/// without a workspace stays visible in the shard it was found in.
fn cwd_matches(header_cwd: &str, target: &Path) -> bool {
    header_cwd.is_empty() || absolute_path(Path::new(header_cwd)) == target
}

/// Creates a directory and its missing parents as owner-only, leaving the
/// mode of anything that already exists alone: a shared or system-owned
/// parent is not this code's to lock down, and chmod there fails with EPERM.
fn create_private_dir(directory: &Path) -> io::Result<()> {
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    builder.mode(0o700);
    builder.create(directory)
}

/// Writes a file that only its owner can read. Transcripts carry the same
/// content as the 0600 session log, so an export must not widen that.
pub fn write_private(path: impl AsRef<Path>, contents: &[u8]) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    options.open(path)?.write_all(contents)
}

/// The `.jsonl` files directly inside one shard directory.
fn session_files(directory: &Path) -> Result<Vec<PathBuf>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error.into()),
    };
    Ok(entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|kind| !kind.is_dir()))
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("jsonl"))
        .collect())
}

/// The id encoded in a `<stamp>_<id>.jsonl` name, when the name follows the
/// convention. The stamp never contains an underscore.
fn filename_session_id(path: &Path) -> Option<&str> {
    path.file_stem()?
        .to_str()?
        .split_once('_')
        .map(|(_, id)| id)
}

fn trim_line(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 && matches!(line[end - 1], b'\r' | b'\n') {
        end -= 1;
    }
    &line[..end]
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "goshcoder-sessionlog-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    fn user(text: &str) -> Entry {
        Entry {
            kind: TYPE_MESSAGE.to_owned(),
            message: Some(json!({"role": "user", "content": text, "timestamp": 1})),
            ..Entry::default()
        }
    }

    fn assistant(text: &str) -> Entry {
        Entry {
            kind: TYPE_MESSAGE.to_owned(),
            message: Some(json!({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "api": "test",
                "provider": "test",
                "model": "test",
                "usage": {},
                "stopReason": "stop",
                "timestamp": 2
            })),
            ..Entry::default()
        }
    }

    #[test]
    fn root_entries_keep_required_null_parent_id() {
        let root = temp_root("parent");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        let first = writer.append(user("question")).expect("append user");
        writer
            .append(assistant("answer"))
            .expect("append assistant");
        writer.sync().expect("sync");
        let path = writer.path().to_path_buf();
        writer.close().expect("close");

        let contents = fs::read_to_string(path).expect("read session");
        let raw = contents
            .lines()
            .find(|line| line.contains(&format!("\"id\":\"{first}\"")))
            .expect("raw entry");
        assert!(raw.contains("\"parentId\":null"));
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn load_recovers_from_a_malformed_tail_without_losing_prefix() {
        let root = temp_root("tail");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        writer.append(user("question")).expect("append user");
        writer
            .append(assistant("answer"))
            .expect("append assistant");
        let path = writer.path().to_path_buf();
        writer.close().expect("close");

        OpenOptions::new()
            .append(true)
            .open(&path)
            .and_then(|mut file| file.write_all(br#"{"type":"mess"#))
            .expect("tear session");
        let (tree, _, report) = store.load(&path).expect("load valid prefix");

        assert_eq!(tree.len(), 2);
        assert!(report.repaired_tail);
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn compaction_and_reset_project_the_active_context() {
        let mut tree = Tree::new();
        for (id, kind, parent) in [
            ("one", TYPE_MESSAGE, None),
            ("two", TYPE_MESSAGE, Some("one")),
            ("three", TYPE_COMPACTION, Some("two")),
            ("four", TYPE_MESSAGE, Some("three")),
            ("five", TYPE_TRANSCRIPT_RESET, Some("four")),
            ("six", TYPE_MESSAGE, Some("five")),
        ] {
            tree.add(Entry {
                kind: kind.to_owned(),
                id: id.to_owned(),
                parent_id: parent.map(str::to_owned),
                timestamp: now(),
                first_kept_entry_id: "two".to_owned(),
                ..Entry::default()
            })
            .expect("add");
        }

        let projected = tree.context_path(None);
        assert_eq!(
            projected
                .iter()
                .map(|entry| entry.id.as_str())
                .collect::<Vec<_>>(),
            ["six"]
        );
    }

    #[test]
    fn invalid_session_ids_cannot_become_paths() {
        for id in ["", ".hidden", "../escape", "has space", "trailing-"] {
            assert!(validate_session_id(id).is_err(), "{id:?} was accepted");
        }
        validate_session_id("session_1.2-3").expect("safe identifier");
    }

    #[test]
    fn attach_repairs_torn_tails_and_refuses_a_second_writer() {
        let root = temp_root("attach");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        writer.append(user("question")).expect("append user");
        writer
            .append(assistant("answer"))
            .expect("append assistant");
        let path = writer.path().to_path_buf();

        let error = store.attach(&path).expect_err("second writer is refused");
        assert!(matches!(error, SessionError::Busy(_)));
        writer.close().expect("close first writer");

        OpenOptions::new()
            .append(true)
            .open(&path)
            .and_then(|mut file| file.write_all(br#"{"type":"mess"#))
            .expect("append torn tail");

        let (mut recovered, report) = store.attach(&path).expect("attach repaired session");
        assert!(report.repaired_tail);
        recovered.append(user("after crash")).expect("append again");
        recovered.close().expect("close recovered writer");

        let (tree, _, report) = store.load(&path).expect("reload");
        assert_eq!(tree.len(), 3);
        assert_eq!(report.skipped_lines, 0);
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn legacy_sessions_migrate_in_memory_and_fork_to_v3() {
        let root = temp_root("legacy");
        let store = Store::new(root.join("sessions"));
        let legacy = root.join("legacy.jsonl");
        fs::create_dir_all(&root).expect("make root");
        fs::write(
            &legacy,
            concat!(
                "{\"type\":\"session\",\"id\":\"legacy\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/legacy\"}\n",
                "{\"type\":\"message\",\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{\"role\":\"user\",\"content\":\"before\",\"timestamp\":1}}\n",
                "{\"type\":\"compaction\",\"timestamp\":\"2026-01-01T00:00:02.000Z\",\"summary\":\"summary\",\"firstKeptEntryIndex\":1}\n",
                "{\"type\":\"message\",\"timestamp\":\"2026-01-01T00:00:03.000Z\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"after\"}],\"timestamp\":2}}\n"
            ),
        )
        .expect("write legacy fixture");

        let (tree, header, report) = store.load(&legacy).expect("load legacy");
        assert!(report.migrated);
        assert_eq!(report.source_version, 1);
        assert_eq!(header.version, FORMAT_VERSION);
        assert!(tree.all().iter().all(|entry| !entry.id.is_empty()));
        assert!(
            tree.all()
                .iter()
                .find(|entry| entry.kind == TYPE_COMPACTION)
                .is_some_and(|entry| !entry.first_kept_entry_id.is_empty())
        );
        assert!(matches!(
            store.attach(&legacy),
            Err(SessionError::LegacyFormat(1))
        ));

        let source = store.describe(&legacy, false).expect("describe legacy");
        let mut fork = store
            .fork(&source, None, root.join("workspace"))
            .expect("fork migrated session");
        let fork_path = fork.path().to_path_buf();
        fork.append(user("continuing")).expect("append fork");
        fork.close().expect("close fork");

        let (forked, header, report) = store.load(&fork_path).expect("load fork");
        assert!(!report.migrated);
        assert_eq!(header.version, FORMAT_VERSION);
        assert_eq!(
            header.parent_session.as_deref(),
            Some(legacy.to_string_lossy().as_ref()),
            "pi records the source file, not its id"
        );
        assert_eq!(forked.len(), 4);
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn v2_hook_messages_are_renamed_for_current_message_decoders() {
        let root = temp_root("v2");
        let store = Store::new(root.join("sessions"));
        let path = root.join("v2.jsonl");
        fs::create_dir_all(&root).expect("make root");
        fs::write(
            &path,
            concat!(
                "{\"type\":\"session\",\"version\":2,\"id\":\"v2\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/legacy\"}\n",
                "{\"type\":\"message\",\"id\":\"message\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{\"role\":\"hookMessage\",\"content\":\"injected\"}}\n"
            ),
        )
        .expect("write v2 fixture");

        let (tree, _, report) = store.load(&path).expect("load v2");
        assert!(report.migrated);
        assert_eq!(
            tree.all()[0]
                .message
                .as_ref()
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str),
            Some("custom")
        );
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn listing_resolution_and_read_only_open_preserve_session_access() {
        let root = temp_root("listing");
        let workspace = root.join("workspace");
        let store = Store::new(root.join("sessions"));

        let mut first = store
            .create_with_id(&workspace, None, "a1234567-first")
            .expect("create first");
        first
            .append(assistant("first answer"))
            .expect("append first");
        first.close().expect("close first");

        let mut second = store
            .create_with_id(&workspace, None, "a1234567-second")
            .expect("create second");
        second
            .append(assistant("second answer"))
            .expect("append second");
        second.close().expect("close second");

        let sessions = store
            .list(
                &workspace,
                ListOptions {
                    with_text: true,
                    ..ListOptions::default()
                },
            )
            .expect("list");
        assert_eq!(sessions.len(), 2);
        assert!(
            sessions
                .iter()
                .all(|session| session.search_text.contains("answer"))
        );
        let ids = short_ids(&sessions);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a1234567-fir".to_owned()));
        assert!(ids.contains(&"a1234567-sec".to_owned()));
        assert!(matches!(
            store.resolve(&workspace, "a1234567"),
            Err(SessionError::Ambiguous(_))
        ));

        let resolved = store
            .resolve(&workspace, "a1234567-first")
            .expect("resolve exact");
        let (mut reader, _) = store.open(&resolved.path).expect("open read only");
        assert!(reader.read_only());
        assert!(matches!(
            reader.append(user("nope")),
            Err(SessionError::ReadOnly)
        ));
        reader.close().expect("close reader");
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn fork_keeps_reachable_labels_and_moves_the_write_head_to_conversation() {
        let root = temp_root("fork");
        let workspace = root.join("workspace");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(&workspace).expect("create");
        let first = writer.append(user("question")).expect("append question");
        writer.append(assistant("answer")).expect("append answer");
        let label = "remember this".to_owned();
        writer
            .append(Entry {
                kind: TYPE_LABEL.to_owned(),
                target_id: first.clone(),
                label: Some(label.clone()),
                ..Entry::default()
            })
            .expect("append label");
        let source_path = writer.path().to_path_buf();
        writer.close().expect("close source");

        let source = store
            .describe(&source_path, false)
            .expect("describe source");
        let fork = store.fork(&source, None, &workspace).expect("fork");
        let fork_path = fork.path().to_path_buf();
        let tree = fork.snapshot();
        assert_eq!(tree.label(&first), Some(label.as_str()));
        assert_ne!(
            tree.leaf()
                .and_then(|leaf| tree.entry(leaf))
                .map(|entry| entry.kind.as_str()),
            Some(TYPE_LABEL)
        );
        drop(fork);

        let (tree, _, _) = store.load(&fork_path).expect("reload fork");
        assert_eq!(tree.label(&first), Some(label.as_str()));
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn attach_keeps_a_resumed_session_that_is_closed_without_a_new_reply() {
        let root = temp_root("attach-keep");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        writer.append(user("question")).expect("append user");
        writer.keep();
        let path = writer.path().to_path_buf();
        writer.close().expect("close");

        let (mut resumed, _) = store.attach(&path).expect("attach");
        resumed.append(user("still no answer")).expect("append");
        resumed.close().expect("close resumed");

        assert!(
            path.exists(),
            "a session that existed before attach is never discarded by it"
        );
        let (tree, _, _) = store.load(&path).expect("reload");
        assert_eq!(tree.len(), 2);
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[cfg(unix)]
    #[test]
    fn creating_a_session_never_changes_the_mode_of_an_existing_directory() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_root("dir-mode");
        let sessions = root.join("sessions");
        fs::create_dir_all(&sessions).expect("make sessions dir");
        fs::set_permissions(&sessions, fs::Permissions::from_mode(0o755)).expect("set mode");
        let store = Store::new(&sessions);

        let mut writer = store.create(root.join("workspace")).expect("create");
        writer.append(assistant("answer")).expect("append");
        let shard = writer.path().parent().expect("shard").to_path_buf();
        writer.close().expect("close");

        let mode = |path: &Path| fs::metadata(path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(
            mode(&sessions),
            0o755,
            "an existing directory keeps its mode"
        );
        assert_eq!(
            mode(&shard),
            0o700,
            "a directory this store creates is private"
        );
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn a_claim_taken_over_by_another_process_stops_the_writer_and_keeps_the_file() {
        let root = temp_root("takeover");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        writer
            .append(user("question"))
            .expect("append before takeover");
        let path = writer.path().to_path_buf();
        let lock = lock_path(&path);
        // A waiter that declared this claim stale rewrites the lock with its
        // own token; from then on the file is its to append to and remove.
        fs::write(&lock, "999999 successor\n").expect("rewrite lock");

        assert!(matches!(
            writer.append(user("after takeover")),
            Err(SessionError::Degraded(_))
        ));
        assert!(!writer.recording());
        writer.close().expect("close");
        assert!(path.exists(), "the successor's file must not be unlinked");
        assert_eq!(
            fs::read_to_string(&lock).expect("lock"),
            "999999 successor\n",
            "the successor's lock must not be removed"
        );

        let (tree, _, report) = store.load(&path).expect("reload");
        assert_eq!(tree.len(), 1);
        assert_eq!(report.skipped_lines, 0);
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[cfg(unix)]
    #[test]
    fn a_fresh_lock_from_a_dead_process_is_reclaimed() {
        let root = temp_root("dead-pid");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        writer.append(assistant("answer")).expect("append");
        let path = writer.path().to_path_buf();
        writer.close().expect("close");

        // A process that has already exited leaves a pid nothing answers for.
        let dead = std::process::Command::new("true")
            .spawn()
            .and_then(|mut child| {
                let id = child.id();
                child.wait()?;
                Ok(id)
            })
            .expect("run a short-lived process");
        fs::write(lock_path(&path), format!("{dead} crashed\n")).expect("write stale lock");

        assert!(
            !store.describe(&path, false).expect("describe").locked,
            "a lock whose holder is gone does not show as open"
        );
        let (mut resumed, _) = store
            .attach(&path)
            .expect("attach reclaims the dead holder's lock inside the heartbeat window");
        resumed.append(user("continued")).expect("append");
        resumed.close().expect("close");
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn headers_without_cwd_or_timestamp_are_still_discoverable() {
        let root = temp_root("bare-header");
        let store = Store::new(root.join("sessions"));
        let workspace = root.join("workspace");
        let shard = store.directory(&workspace);
        fs::create_dir_all(&shard).expect("make shard");
        let path = shard.join("bare.jsonl");
        fs::write(
            &path,
            concat!(
                "{\"type\":\"session\",\"version\":3,\"id\":\"bare-header\"}\n",
                "{\"type\":\"message\",\"id\":\"m1\",\"parentId\":null,\"message\":{\"role\":\"user\",\"content\":\"hello\",\"timestamp\":1}}\n"
            ),
        )
        .expect("write fixture");

        let (tree, header, _) = store.load(&path).expect("load without cwd or timestamps");
        assert_eq!(header.id, "bare-header");
        assert!(header.cwd.is_empty());
        assert_eq!(tree.len(), 1);
        let listed = store
            .list(&workspace, ListOptions::default())
            .expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].first_message, "hello");
        assert_eq!(
            store
                .most_recent(&workspace)
                .expect("most recent")
                .map(|info| info.id)
                .as_deref(),
            Some("bare-header")
        );
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn workspace_paths_are_cleaned_before_they_name_a_shard() {
        assert_eq!(
            clean_path(Path::new("/a/b/../c/./d/")),
            PathBuf::from("/a/c/d")
        );
        assert_eq!(clean_path(Path::new("/../x")), PathBuf::from("/x"));
        assert_eq!(clean_path(Path::new("../x/./y")), PathBuf::from("../x/y"));
        assert_eq!(
            Store::dir_name("/work/project/"),
            Store::dir_name("/work/project")
        );
        assert_eq!(
            Store::dir_name("/work/other/../project"),
            Store::dir_name("/work/project")
        );
        assert_eq!(Store::dir_name("./relative"), Store::dir_name("relative"));
    }

    #[test]
    fn discovery_settles_on_headers_and_file_names_before_parsing_a_transcript() {
        let root = temp_root("discovery");
        let workspace = root.join("workspace");
        let store = Store::new(root.join("sessions"));
        let mut older = store
            .create_with_id(&workspace, None, "older-session")
            .expect("create older");
        older.append(assistant("first")).expect("append");
        let older_path = older.path().to_path_buf();
        older.close().expect("close older");
        let mut newer = store
            .create_with_id(&workspace, None, "newer-session")
            .expect("create newer");
        newer.append(assistant("second")).expect("append");
        let newer_path = newer.path().to_path_buf();
        newer.close().expect("close newer");

        let shard = older_path.parent().expect("shard").to_path_buf();
        // A file that does not follow the naming convention is still found
        // through its header, which is the authority on ids.
        let imported = shard.join("imported.jsonl");
        fs::write(
            &imported,
            format!(
                "{}\n",
                json!({
                    "type": "session",
                    "version": 3,
                    "id": "imported-session",
                    "timestamp": "2026-01-01T00:00:00.000Z",
                    "cwd": workspace.to_string_lossy()
                })
            ),
        )
        .expect("write imported");
        // A name that promises an id its header does not carry is not a match.
        fs::write(
            shard.join("2026-01-01T00-00-00-000Z_decoy.jsonl"),
            "{\"type\":\"message\",\"id\":\"x\"}\n",
        )
        .expect("write decoy");
        let now = SystemTime::now();
        for (path, age) in [(&imported, 120), (&older_path, 60), (&newer_path, 0)] {
            File::options()
                .write(true)
                .open(path)
                .and_then(|file| {
                    file.set_times(
                        fs::FileTimes::new().set_modified(now - Duration::from_secs(age)),
                    )
                })
                .expect("set mtime");
        }

        assert_eq!(
            store
                .most_recent(&workspace)
                .expect("most recent")
                .map(|info| info.id)
                .as_deref(),
            Some("newer-session")
        );
        assert_eq!(
            store
                .resolve(&workspace, "older-session")
                .expect("exact id via file name")
                .id,
            "older-session"
        );
        assert_eq!(
            store
                .resolve(&workspace, "imported-session")
                .expect("exact id via header")
                .id,
            "imported-session"
        );
        assert_eq!(
            store.resolve(&workspace, "newer").expect("prefix").id,
            "newer-session"
        );
        assert!(matches!(
            store.resolve(&workspace, "decoy"),
            Err(SessionError::NotFound(_))
        ));
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn short_ids_cut_at_character_boundaries() {
        let info = |id: &str| SessionInfo {
            id: id.to_owned(),
            path: PathBuf::new(),
            cwd: String::new(),
            name: String::new(),
            first_message: String::new(),
            created: None,
            modified: SystemTime::UNIX_EPOCH,
            messages: 0,
            cleared: 0,
            size: 0,
            search_text: String::new(),
            locked: false,
            owner: LockOwner::default(),
        };
        let sessions = [info("会話セッション一"), info("会話セッション二")];
        assert_eq!(sessions[0].short_id(), "会話");
        assert_eq!(
            short_ids(&sessions),
            ["会話セッション一", "会話セッション二"]
        );
    }

    #[test]
    fn attach_cuts_a_rejected_unterminated_tail_instead_of_appending_onto_it() {
        let root = temp_root("rejected-tail");
        let store = Store::new(root.join("sessions"));
        let mut writer = store.create(root.join("workspace")).expect("create");
        let first = writer.append(user("question")).expect("append user");
        writer
            .append(assistant("answer"))
            .expect("append assistant");
        let path = writer.path().to_path_buf();
        writer.close().expect("close");

        // A duplicate id parses but is rejected by the tree. Without its
        // newline it has to be cut, or the next append is glued onto it.
        let duplicate = format!(
            "{{\"type\":\"message\",\"id\":\"{first}\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"message\":{{\"role\":\"user\",\"content\":\"dup\"}}}}"
        );
        OpenOptions::new()
            .append(true)
            .open(&path)
            .and_then(|mut file| file.write_all(duplicate.as_bytes()))
            .expect("append rejected tail");

        let (mut recovered, report) = store.attach(&path).expect("attach");
        assert!(report.repaired_tail);
        assert_eq!(report.skipped_lines, 1);
        recovered.append(user("after")).expect("append");
        recovered.close().expect("close");

        let (tree, _, report) = store.load(&path).expect("reload");
        assert_eq!(tree.len(), 3);
        assert_eq!(report.skipped_lines, 0);
        assert!(!report.repaired_tail);
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[cfg(unix)]
    #[test]
    fn private_writes_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = temp_root("private-write");
        fs::create_dir_all(&root).expect("make root");
        let path = root.join("export.md");
        write_private(&path, b"secret").expect("write");
        assert_eq!(
            fs::metadata(&path).expect("metadata").permissions().mode() & 0o777,
            0o600
        );
        write_private(&path, b"replaced").expect("overwrite");
        assert_eq!(fs::read(&path).expect("read"), b"replaced");
        fs::remove_dir_all(root).expect("clean test root");
    }

    #[test]
    fn fork_copies_a_reattached_entry_with_its_repaired_parent() {
        let root = temp_root("fork-reattach");
        let workspace = root.join("workspace");
        let store = Store::new(root.join("sessions"));
        let source = root.join("source.jsonl");
        fs::create_dir_all(&root).expect("make root");
        fs::write(
            &source,
            concat!(
                "{\"type\":\"session\",\"version\":3,\"id\":\"src\",\"timestamp\":\"2026-01-01T00:00:00.000Z\",\"cwd\":\"/w\"}\n",
                "{\"type\":\"message\",\"id\":\"a\",\"parentId\":null,\"timestamp\":\"2026-01-01T00:00:01.000Z\",\"message\":{\"role\":\"user\",\"content\":\"q\",\"timestamp\":1}}\n",
                "{\"type\":\"message\",\"id\":\"b\",\"parentId\":\"missing\",\"timestamp\":\"2026-01-01T00:00:02.000Z\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"a\"}],\"timestamp\":2},\"usage\":{\"extra\":true}}\n"
            ),
        )
        .expect("write source");
        let info = store.describe(&source, false).expect("describe");
        let fork = store.fork(&info, None, &workspace).expect("fork");
        let fork_path = fork.path().to_path_buf();
        drop(fork);

        let (tree, _, report) = store.load(&fork_path).expect("reload fork");
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(
            tree.entry("b").and_then(|entry| entry.parent_id.as_deref()),
            Some("a")
        );
        // Fields this build does not model survive the repair byte-for-byte.
        assert!(
            fs::read_to_string(&fork_path)
                .expect("read fork")
                .contains("\"usage\":{\"extra\":true}")
        );
        fs::remove_dir_all(root).expect("clean test root");
    }
}
