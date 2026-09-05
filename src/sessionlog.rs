//! Pi v3-compatible append-only session files.
//!
//! Session data is deliberately kept as JSON values at this layer. It lets the
//! persistence format retain provider fields added by a newer client while the
//! Rust runtime incrementally grows strongly typed protocol support.

use std::{
    collections::HashMap,
    error::Error as StdError,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
    path::{Path, PathBuf},
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
                    "sessionlog: legacy session format v{version} has not been migrated yet"
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
        let entry: Self = serde_json::from_slice(line)?;
        if entry.kind.is_empty() {
            return Err(SessionError::InvalidEntry("type is missing".to_owned()));
        }
        if entry.kind == HEADER_TYPE {
            return Err(SessionError::InvalidEntry(
                "is a second session header".to_owned(),
            ));
        }
        if entry.id.is_empty() {
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
    pub source_version: u32,
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
            .replace('/', "-")
            .replace('\\', "-")
            .replace(':', "-");
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
        let mut options = OpenOptions::new();
        options.write(true).append(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(&path)?;
        let header_line = header.encode()?;
        file.write_all(&header_line)?;
        Ok(Writer {
            file: Some(file),
            path,
            header,
            tree: Tree::new(),
            size: header_line.len() as u64,
            keep: false,
        })
    }

    pub fn load(&self, path: impl AsRef<Path>) -> Result<(Tree, Header, LoadReport)> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        if metadata.len() > MAX_SESSION_BYTES {
            return Err(SessionError::SessionTooLarge(metadata.len()));
        }
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Err(SessionError::EmptySession);
        }
        let header = Header::decode(trim_line(&line))?;
        if header.version < FORMAT_VERSION {
            return Err(SessionError::LegacyFormat(header.version));
        }
        let mut report = LoadReport {
            source_version: header.version,
            ..LoadReport::default()
        };
        let mut tree = Tree::new();
        let mut last_added = None::<String>;
        let mut line_number = 1usize;

        loop {
            line.clear();
            let bytes = reader.read_until(b'\n', &mut line)?;
            if bytes == 0 {
                break;
            }
            line_number += 1;
            if line.len() > MAX_ENTRY_BYTES {
                report.skipped_lines += 1;
                report.warnings.push(format!(
                    "line {line_number} exceeds the {MAX_ENTRY_BYTES}-byte entry limit and was skipped"
                ));
                continue;
            }
            let complete = line.ends_with(b"\n");
            let raw = trim_line(&line);
            if raw.is_empty() {
                continue;
            }
            let mut entry = match Entry::decode(raw) {
                Ok(entry) => entry,
                Err(error) if !complete => {
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
                    continue;
                }
            };

            if let Some(parent) = entry.parent_id.as_deref() {
                if !tree.has(parent) {
                    entry.parent_id = last_added.clone();
                    report.warnings.push(format!(
                        "line {line_number} referenced an entry that is not in the file; it was reattached so earlier conversation stays reachable"
                    ));
                }
            }
            match tree.add(entry.clone(), &line) {
                Ok(()) => {
                    last_added = Some(entry.id);
                    if !complete {
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
        }

        Ok((tree, header, report))
    }
}

/// An append-only writer. The higher-level session runtime owns synchronization
/// and process claims; this value keeps write ordering and JSONL invariants
/// local to persistence.
pub struct Writer {
    file: Option<File>,
    path: PathBuf,
    header: Header,
    tree: Tree,
    size: u64,
    keep: bool,
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

    pub fn snapshot(&self) -> Tree {
        self.tree.clone()
    }

    pub fn keep(&mut self) {
        self.keep = true;
    }

    pub fn append(&mut self, mut entry: Entry) -> Result<String> {
        if self.file.is_none() {
            return Err(SessionError::Closed);
        }
        entry.id = new_entry_id(&self.tree);
        entry.parent_id = self.tree.leaf().map(str::to_owned);
        entry.timestamp = now();
        let line = entry.encode()?;
        if line.len() > MAX_ENTRY_BYTES {
            return Err(SessionError::EntryTooLarge(line.len()));
        }
        let new_size = self.size + line.len() as u64;
        if new_size > MAX_SESSION_BYTES {
            return Err(SessionError::SessionTooLarge(new_size));
        }
        let file = self.file.as_mut().expect("checked above");
        file.write_all(&line)?;
        self.size = new_size;
        let id = entry.id.clone();
        self.tree.add(entry, &line)?;
        Ok(id)
    }

    pub fn set_leaf(&mut self, id: impl Into<String>) -> Result<()> {
        self.tree.set_leaf(id)
    }

    pub fn sync(&mut self) -> Result<()> {
        let Some(file) = self.file.as_mut() else {
            return Err(SessionError::Closed);
        };
        file.sync_all()?;
        Ok(())
    }

    /// Finalizes a session. Sessions without an assistant reply are discarded
    /// unless the caller explicitly retained one (for example, a user-created
    /// fork that intentionally ends at a prompt).
    pub fn close(&mut self) -> Result<()> {
        let Some(mut file) = self.file.take() else {
            return Ok(());
        };
        file.flush()?;
        file.sync_all()?;
        drop(file);
        if !self.keep && !self.tree.has_assistant_message() {
            fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        let _ = self.close();
    }
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
    for _ in 0..100 {
        let encoded = Uuid::now_v7().simple().to_string();
        let id = encoded[encoded.len() - 8..].to_owned();
        if !tree.has(&id) {
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
}
