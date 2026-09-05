//! Pi v3-compatible append-only session files.
//!
//! Session data is deliberately kept as JSON values at this layer. It lets the
//! persistence format retain provider fields added by a newer client while the
//! Rust runtime incrementally grows strongly typed protocol support.

use std::{
    collections::{HashMap, HashSet},
    error::Error as StdError,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    thread::{self, JoinHandle},
    time::{Duration, SystemTime},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

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
    pub timestamp: String,
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
    raw: HashMap<String, Vec<u8>>,
    order: Vec<String>,
    labels: HashMap<String, String>,
    name: String,
    leaf_id: Option<String>,
}

impl Tree {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, entry: Entry, raw: impl AsRef<[u8]>) -> Result<()> {
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
        self.raw.insert(id.clone(), raw.as_ref().to_vec());
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

    pub fn raw(&self, id: &str) -> Option<&[u8]> {
        self.raw.get(id).map(Vec::as_slice)
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
        fs::create_dir_all(&directory)?;
        #[cfg(unix)]
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;

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
        let (tree, header, report, _) = read_into(&mut file, path)?;
        Ok((tree, header, report))
    }

    /// Opens an existing v3 session for append while holding its on-disk
    /// claim. A pre-v3 session can be read or forked, but cannot safely be
    /// appended in place because its generated identities are not durable.
    pub fn attach(&self, path: impl AsRef<Path>) -> Result<(Writer, LoadReport)> {
        let path = path.as_ref();
        let claim = claim(path)?;
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let (tree, header, mut report, offset) = read_into(&mut file, path)?;
        if report.migrated {
            return Err(SessionError::LegacyFormat(report.source_version));
        }

        let current_size = file.metadata()?.len();
        if current_size > offset {
            file.set_len(offset)?;
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
                keep: false,
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
        let cwd = cwd.as_ref();
        let target_cwd = absolute_path(cwd).to_string_lossy().into_owned();
        let directories = if options.all_workspaces {
            match fs::read_dir(&self.root) {
                Ok(entries) => entries
                    .filter_map(std::result::Result::ok)
                    .filter_map(|entry| {
                        entry
                            .file_type()
                            .ok()
                            .filter(|kind| kind.is_dir())
                            .map(|_| entry.path())
                    })
                    .collect::<Vec<_>>(),
                Err(error) if error.kind() == io::ErrorKind::NotFound => Vec::new(),
                Err(error) => return Err(error.into()),
            }
        } else {
            vec![self.directory(cwd)]
        };

        let mut sessions = Vec::new();
        for directory in directories {
            let entries = match fs::read_dir(directory) {
                Ok(entries) => entries,
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error.into()),
            };
            for entry in entries.filter_map(std::result::Result::ok) {
                let path = entry.path();
                if entry.file_type().map_or(true, |kind| kind.is_dir())
                    || path.extension().and_then(|extension| extension.to_str()) != Some("jsonl")
                {
                    continue;
                }
                let Ok(info) = self.describe(&path, options.with_text) else {
                    // A malformed or inaccessible file must not hide usable
                    // sessions in the same directory.
                    continue;
                };
                if options.all_workspaces || info.cwd.is_empty() || info.cwd == target_cwd {
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

    pub fn most_recent(&self, cwd: impl AsRef<Path>) -> Result<Option<SessionInfo>> {
        Ok(self
            .list(
                cwd,
                ListOptions {
                    limit: 1,
                    ..ListOptions::default()
                },
            )?
            .into_iter()
            .next())
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

        for all_workspaces in [false, true] {
            let mut matches = Vec::new();
            for candidate in self.list(
                cwd.as_ref(),
                ListOptions {
                    all_workspaces,
                    ..ListOptions::default()
                },
            )? {
                if candidate.id == reference {
                    return Ok(candidate);
                }
                if candidate.id.starts_with(reference) {
                    matches.push(candidate);
                }
            }
            match matches.len() {
                0 => {}
                1 => return Ok(matches.remove(0)),
                _ => return Err(SessionError::Ambiguous(reference.to_owned())),
            }
        }
        Err(SessionError::NotFound(reference.to_owned()))
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
        let (tree, header, report) = self.load(&source.path)?;
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
        let mut writer =
            self.create_with_id(target_cwd, Some(header.id.clone()), new_session_id())?;
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
        &self.id[..self.id.len().min(SHORT_ID_LENGTH)]
    }
}

/// Returns unique, readable ID prefixes for a list of sessions.
pub fn short_ids(sessions: &[SessionInfo]) -> Vec<String> {
    let mut length = SHORT_ID_LENGTH;
    loop {
        let mut prefixes = HashSet::with_capacity(sessions.len());
        let mut collision = false;
        let mut longest = 0;
        for session in sessions {
            longest = longest.max(session.id.len());
            let end = session.id.len().min(length);
            if !prefixes.insert(&session.id[..end]) {
                collision = true;
                break;
            }
        }
        if !collision || length >= longest {
            return sessions
                .iter()
                .map(|session| session.id[..session.id.len().min(length)].to_owned())
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
        !self.read_only && !self.closed && self.degraded.is_none() && self.file.is_some()
    }

    pub fn degraded(&self) -> Option<&str> {
        self.degraded.as_deref()
    }

    pub fn snapshot(&self) -> Tree {
        self.tree.clone()
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
        self.tree.add(entry, &line)?;
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
        if !self.read_only
            && !self.keep
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

    fn ensure_writable(&self) -> Result<()> {
        if self.closed {
            return Err(SessionError::Closed);
        }
        if self.read_only {
            return Err(SessionError::ReadOnly);
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
        self.tree.add(entry, &line)
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
    stop: Option<mpsc::Sender<()>>,
    heartbeat: Option<JoinHandle<()>>,
}

impl LockClaim {
    fn new(path: PathBuf, token: String) -> Self {
        let (stop, receiver) = mpsc::channel();
        let heartbeat_path = path.clone();
        let heartbeat_token = token.clone();
        let heartbeat = thread::spawn(move || {
            loop {
                match receiver.recv_timeout(LOCK_HEARTBEAT) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if fs::read(&heartbeat_path)
                            .is_ok_and(|contents| contents.as_slice() == heartbeat_token.as_bytes())
                        {
                            let _ = OpenOptions::new()
                                .write(true)
                                .open(&heartbeat_path)
                                .and_then(|file| {
                                    file.set_times(
                                        fs::FileTimes::new().set_modified(SystemTime::now()),
                                    )
                                });
                        }
                    }
                }
            }
        });
        Self {
            path,
            token,
            stop: Some(stop),
            heartbeat: Some(heartbeat),
        }
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

fn claim(path: &Path) -> Result<LockClaim> {
    let path = lock_path(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
        #[cfg(unix)]
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    }
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

fn lock_is_stale(path: &Path) -> bool {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
        .is_some_and(|elapsed| elapsed > LOCK_STALE)
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

fn read_into(file: &mut File, _path: &Path) -> Result<(Tree, Header, LoadReport, u64)> {
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

    let mut tree = Tree::new();
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
                if let Some(parent) = entry.parent_id.as_deref()
                    && !tree.has(parent)
                {
                    entry.parent_id = last_added.clone();
                    report.warnings.push(format!(
                        "line {line_number} referenced an entry that is not in the file; it was reattached so earlier conversation stays reachable"
                    ));
                }
                match tree.add(entry.clone(), &line.bytes) {
                    Ok(()) => {
                        last_added = Some(entry.id);
                        if !line.complete {
                            report.unterminated_tail = true;
                        }
                    }
                    Err(error) => {
                        report.skipped_lines += 1;
                        report
                            .warnings
                            .push(format!("line {line_number} was skipped: {error}"));
                    }
                }
                keep_bytes = total;
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

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|current_dir| current_dir.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
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
        let raw = writer.snapshot().raw(&first).expect("raw entry").to_vec();
        writer.close().expect("close");

        assert!(
            String::from_utf8(raw)
                .expect("utf8")
                .contains("\"parentId\":null")
        );
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
            tree.add(
                Entry {
                    kind: kind.to_owned(),
                    id: id.to_owned(),
                    parent_id: parent.map(str::to_owned),
                    timestamp: now(),
                    first_kept_entry_id: "two".to_owned(),
                    ..Entry::default()
                },
                [],
            )
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
    fn real_pi_v1_compaction_fixture_projects_the_kept_context() {
        let root = temp_root("pi-v1-compaction");
        fs::create_dir_all(&root).expect("make root");
        let fixture = root.join("pi-v1-compaction.jsonl");
        fs::write(
            &fixture,
            include_str!("../internal/sessionlog/testdata/pi-v1-compaction.jsonl"),
        )
        .expect("write fixture");
        let store = Store::new(root.join("sessions"));

        let (tree, _, report) = store.load(&fixture).expect("load fixture");

        assert!(report.migrated, "the fixture must exercise v1 migration");
        let compaction = tree
            .all()
            .iter()
            .find(|entry| entry.kind == TYPE_COMPACTION)
            .expect("compaction entry");
        assert!(!compaction.first_kept_entry_id.is_empty());
        assert!(tree.has(&compaction.first_kept_entry_id));

        let full = tree.path(None);
        let context = tree.context_path(None);
        let projected = context
            .iter()
            .map(|entry| match entry.kind.as_str() {
                TYPE_COMPACTION => "[compaction]".to_owned(),
                TYPE_MESSAGE => entry.message.as_ref().map(message_text).unwrap_or_default(),
                _ => String::new(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            projected,
            [
                "[compaction]",
                "the first kept entry",
                "after the compaction"
            ]
        );
        assert!(context.len() < full.len());

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
        assert_eq!(header.parent_session.as_deref(), Some("legacy"));
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
}
