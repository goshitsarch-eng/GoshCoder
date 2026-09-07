//! Pi-compatible configuration paths and durable small-file writes.
//!
//! The application keeps its state under `~/.goshcoder/agent` by default.
//! These helpers preserve the previous path layout so existing credentials,
//! sessions, prompt templates, and integration settings remain discoverable
//! while their Rust readers are migrated.

use std::{
    env,
    ffi::OsStr,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    process,
    sync::atomic::{AtomicU64, Ordering},
};

use sha2::{Digest, Sha256};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

pub const DIR_NAME: &str = ".goshcoder";
pub const ENV_AGENT_DIR: &str = "GOSHCODER_AGENT_DIR";
/// Directory under the agent root holding per-workspace planner state.
pub const PLANNER_DIR_NAME: &str = "planner";
const MAX_DEFAULT_MODEL_BYTES: usize = 4096;
const MAX_PLANNER_LABEL_CHARS: usize = 40;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Expands only a leading `~`, `~/`, or `~\`. User-name expansion is
/// intentionally unsupported, matching the previous implementation.
pub fn expand_tilde(path: impl AsRef<Path>) -> PathBuf {
    expand_tilde_with_home(path.as_ref(), user_home().as_deref())
}

/// Returns the configuration root, respecting `GOSHCODER_AGENT_DIR`.
pub fn agent_dir() -> PathBuf {
    let override_dir = env::var_os(ENV_AGENT_DIR);
    agent_dir_from(override_dir.as_deref(), user_home().as_deref())
}

pub fn auth_path() -> PathBuf {
    agent_dir().join("auth.json")
}

pub fn web_search_path() -> PathBuf {
    agent_dir().join("web-search.json")
}

pub fn omni_route_path() -> PathBuf {
    omni_route_path_in(&agent_dir())
}

/// The OmniRoute configuration inside an explicit agent directory.
pub fn omni_route_path_in(agent_dir: &Path) -> PathBuf {
    agent_dir.join("omniroute.json")
}

pub fn btw_path() -> PathBuf {
    agent_dir().join("pi-btw.json")
}

pub fn aperture_path() -> PathBuf {
    aperture_path_in(&agent_dir())
}

/// pi's `extensions/aperture.json` inside an explicit agent directory.
pub fn aperture_path_in(agent_dir: &Path) -> PathBuf {
    agent_dir.join("extensions").join("aperture.json")
}

pub fn aperture_cache_path() -> PathBuf {
    aperture_cache_path_in(&agent_dir())
}

/// The synchronized Aperture snapshot inside an explicit agent directory.
pub fn aperture_cache_path_in(agent_dir: &Path) -> PathBuf {
    agent_dir.join("extensions").join("aperture-cache.json")
}

pub fn mcp_config_path() -> PathBuf {
    agent_dir().join("mcp.json")
}

pub fn sessions_dir() -> PathBuf {
    agent_dir().join("sessions")
}

pub fn prompts_dir() -> PathBuf {
    agent_dir().join("prompts")
}

pub fn default_model_path() -> PathBuf {
    agent_dir().join("default-model")
}

/// Returns the per-user planner state file shared by every window open on
/// `workspace_root`. See [`planner_state_path_in`] for the key format.
pub fn planner_state_path(workspace_root: &Path) -> PathBuf {
    planner_state_path_in(&agent_dir(), workspace_root)
}

/// Returns `<agent_dir>/planner/<key>.json` for a workspace.
///
/// `workspace_root` must already be canonical: the key hashes the path text
/// as given, so two spellings of one directory would otherwise map to two
/// files. The key is the first 16 bytes of the SHA-256 of that text in
/// lower-case hex, followed by a sanitized basename so a directory listing
/// stays readable. Only the hash distinguishes workspaces.
pub fn planner_state_path_in(agent_dir: &Path, workspace_root: &Path) -> PathBuf {
    let digest = Sha256::digest(workspace_root.to_string_lossy().as_bytes());
    let mut key = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let label = planner_label(workspace_root);
    if !label.is_empty() {
        key.push('-');
        key.push_str(&label);
    }
    key.push_str(".json");
    agent_dir.join(PLANNER_DIR_NAME).join(key)
}

/// Reduces a workspace basename to at most 40 ASCII letters, digits, `-`,
/// and `_`. Runs of other characters collapse to one underscore.
fn planner_label(workspace_root: &Path) -> String {
    let basename = workspace_root
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let mut label = String::new();
    for character in basename.chars() {
        let character = if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
            character
        } else {
            '_'
        };
        if character == '_' && label.ends_with('_') {
            continue;
        }
        if label.len() == MAX_PLANNER_LABEL_CHARS {
            break;
        }
        label.push(character);
    }
    label.trim_matches('_').to_owned()
}

/// Reads the remembered model, treating unavailable, malformed, or oversized
/// data as absent. The bounded read keeps a corrupted config file from being
/// loaded into every interactive startup.
pub fn read_default_model() -> String {
    read_default_model_from(&default_model_path())
}

/// Writes the remembered model via a same-directory temporary file and atomic
/// rename. The file is user-readable only on Unix, where permission bits carry
/// that guarantee.
pub fn write_default_model(model: &str) -> io::Result<()> {
    let path = default_model_path();
    atomic_write(&path, format!("{}\n", model.trim()).as_bytes(), 0o600)
}

/// Creates the agent configuration directory with user-only permissions where
/// the platform supports Unix permission bits.
pub fn ensure_agent_dir() -> io::Result<PathBuf> {
    let dir = agent_dir();
    fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    Ok(dir)
}

fn user_home() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

fn expand_tilde_with_home(path: &Path, home: Option<&Path>) -> PathBuf {
    let text = path.as_os_str().to_string_lossy();
    if text == "~" {
        return home.map_or_else(|| path.to_path_buf(), Path::to_path_buf);
    }
    let remainder = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\"));
    match (remainder, home) {
        (Some(remainder), Some(home)) => home.join(remainder),
        _ => path.to_path_buf(),
    }
}

pub(crate) fn agent_dir_from(override_dir: Option<&OsStr>, home: Option<&Path>) -> PathBuf {
    if let Some(override_dir) = override_dir.filter(|path| !path.is_empty()) {
        return expand_tilde_with_home(Path::new(override_dir), home);
    }
    home.map_or_else(
        || PathBuf::from(DIR_NAME).join("agent"),
        |home| home.join(DIR_NAME).join("agent"),
    )
}

fn read_default_model_from(path: &Path) -> String {
    let Ok(contents) = fs::read(path) else {
        return String::new();
    };
    if contents.len() > MAX_DEFAULT_MODEL_BYTES {
        return String::new();
    }
    String::from_utf8(contents)
        .map(|contents| contents.trim().to_owned())
        .unwrap_or_default()
}

fn atomic_write(path: &Path, contents: &[u8], mode: u32) -> io::Result<()> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no parent directory", path.display()),
        )
    })?;
    fs::create_dir_all(parent)?;

    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no file name", path.display()),
        )
    })?;
    let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        process::id(),
        sequence
    ));

    let write_result: io::Result<()> = (|| {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(mode);
        let mut file = options.open(&temporary)?;
        // Force the mode on the descriptor rather than on the destination
        // path once the rename has happened. `open` applies the umask, so the
        // bits still have to be set explicitly, but a path-based chmod
        // afterwards would follow whatever sits at that path by then --
        // a symlink included.
        #[cfg(unix)]
        file.set_permissions(fs::Permissions::from_mode(mode))?;
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_dir(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        env::temp_dir().join(format!("goshcoder-rust-{label}-{}-{nonce}", process::id()))
    }

    /// `atomic_write` forces the mode on the descriptor before the rename.
    /// The race it avoids -- a symlink swapped in between the rename and a
    /// path-based chmod -- cannot be reproduced deterministically, so this
    /// pins the observable half: the file really does end up owner-only, and
    /// not merely because the ambient umask happened to say so.
    #[cfg(unix)]
    #[test]
    fn an_atomically_written_file_is_owner_only() {
        let directory = test_dir("atomic-mode");
        fs::create_dir_all(&directory).expect("create temp directory");
        let path = directory.join("default-model");

        atomic_write(&path, b"anthropic/claude-sonnet-5\n", 0o600).expect("write");
        assert_eq!(
            fs::metadata(&path)
                .expect("stat written file")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );

        // Replacing an existing file keeps the guarantee.
        atomic_write(&path, b"openai/gpt-5.6-terra\n", 0o600).expect("rewrite");
        assert_eq!(
            fs::metadata(&path)
                .expect("stat rewritten file")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        assert_eq!(
            fs::read_to_string(&path).expect("read back"),
            "openai/gpt-5.6-terra\n"
        );

        fs::remove_dir_all(&directory).expect("remove temp directory");
    }

    #[test]
    fn expands_only_supported_tilde_forms() {
        let home = Path::new("/home/example");

        assert_eq!(expand_tilde_with_home(Path::new("~"), Some(home)), home);
        assert_eq!(
            expand_tilde_with_home(Path::new("~/agent"), Some(home)),
            home.join("agent")
        );
        assert_eq!(
            expand_tilde_with_home(Path::new("~\\agent"), Some(home)),
            home.join("agent")
        );
        assert_eq!(
            expand_tilde_with_home(Path::new("~another/agent"), Some(home)),
            PathBuf::from("~another/agent")
        );
    }

    #[test]
    fn agent_directory_keeps_existing_layout() {
        let home = Path::new("/home/example");
        assert_eq!(
            agent_dir_from(None, Some(home)),
            PathBuf::from("/home/example/.goshcoder/agent")
        );
        assert_eq!(
            agent_dir_from(Some(OsStr::new("~/custom")), Some(home)),
            PathBuf::from("/home/example/custom")
        );
    }

    #[test]
    fn planner_state_path_is_stable_and_sanitized() {
        let agent = Path::new("/home/example/.goshcoder/agent");
        let root = Path::new("/srv/repos/My Project!");
        let path = planner_state_path_in(agent, root);
        assert_eq!(
            path,
            agent
                .join("planner")
                .join("c4f1efc5c4ff38c46cc7820baba3c2da-My_Project.json")
        );
        assert_ne!(
            path,
            planner_state_path_in(agent, Path::new("/srv/repos/My Project"))
        );

        let long = planner_state_path_in(agent, &Path::new("/srv").join("a".repeat(100)));
        let name = long.file_name().and_then(OsStr::to_str).expect("name");
        assert_eq!(name.len(), 32 + 1 + MAX_PLANNER_LABEL_CHARS + ".json".len());

        let bare = planner_state_path_in(agent, Path::new("/"));
        let name = bare.file_name().and_then(OsStr::to_str).expect("name");
        assert_eq!(name.len(), 32 + ".json".len());
        assert!(!name.contains('-'));
    }

    #[test]
    fn model_file_is_bounded_and_atomically_replaced() {
        let dir = test_dir("default-model");
        let path = dir.join("default-model");
        atomic_write(&path, b"vendor/model\n", 0o600).expect("write first model");
        assert_eq!(read_default_model_from(&path), "vendor/model");

        atomic_write(&path, b"vendor/replaced\n", 0o600).expect("replace model");
        assert_eq!(read_default_model_from(&path), "vendor/replaced");

        File::create(&path)
            .and_then(|mut file| file.write_all(&vec![b'x'; MAX_DEFAULT_MODEL_BYTES + 1]))
            .expect("write oversized model");
        assert!(read_default_model_from(&path).is_empty());
        fs::remove_dir_all(dir).expect("remove test directory");
    }
}
