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

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

pub const DIR_NAME: &str = ".goshcoder";
pub const ENV_AGENT_DIR: &str = "GOSHCODER_AGENT_DIR";
const MAX_DEFAULT_MODEL_BYTES: usize = 4096;
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
    agent_dir().join("omniroute.json")
}

pub fn btw_path() -> PathBuf {
    agent_dir().join("pi-btw.json")
}

pub fn aperture_path() -> PathBuf {
    agent_dir().join("extensions").join("aperture.json")
}

pub fn aperture_cache_path() -> PathBuf {
    agent_dir().join("extensions").join("aperture-cache.json")
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

fn agent_dir_from(override_dir: Option<&OsStr>, home: Option<&Path>) -> PathBuf {
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
        file.write_all(contents)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(mode))?;
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
