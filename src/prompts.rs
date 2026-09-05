//! Command-line management for portable prompt templates.
//!
//! This keeps `goshcoder prompts` usable without entering the Ratatui session.
//! The on-disk format and archive semantics are implemented by `resources`.

use std::{
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use time::OffsetDateTime;

use crate::{
    config,
    resources::{
        self, ArchiveRead, PromptCollection, PromptScope, ResourcePaths, RestoreOptions,
        RestoreOutcome,
    },
};

const RESERVED_COMMAND_NAMES: &[&str] = &[
    "exit",
    "quit",
    "help",
    "?",
    "system",
    "steer",
    "followup",
    "queue",
    "clear",
    "new",
    "compact",
    "reload",
    "resources",
    "messages",
    "tools",
    "status",
    "sidebar",
    "session",
    "sessions",
    "resume",
    "name",
    "model",
    "thinking",
    "login",
    "logout",
    "btw",
    "omni",
    "ralph",
    "planner",
    "planner-review",
    "planner-annotate",
    "planner-last",
    "prompt",
    "prompts",
    "tree",
    "fork",
    "label",
    "clone",
    "export",
    "import",
    "hotkeys",
];

/// Executes the `goshcoder prompts` command.
pub fn command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let workspace = env::current_dir()?;
    let paths = ResourcePaths::new(workspace, config::agent_dir())?;
    match arguments.first().map(String::as_str) {
        None | Some("list") => {
            let collection = collect(&paths)?;
            print_warnings(&collection.warnings);
            if collection.prompts.is_empty() {
                eprintln!("no saved prompts");
                return Ok(());
            }
            for prompt in collection.prompts {
                println!("/{:<24} {}", prompt.name, scope_label(prompt.scope));
            }
            Ok(())
        }
        Some("backup") => {
            let output = arguments.get(1).map(PathBuf::from);
            let (path, warnings) = backup_at(&paths, output.as_deref())?;
            print_warnings(&warnings);
            eprintln!("backed up prompts to {}", path.display());
            println!("{}", path.display());
            Ok(())
        }
        Some("restore") => {
            let Some(archive) = arguments.get(1) else {
                return Err(command_error("prompts restore needs an archive path"));
            };
            let options = RestoreOptions {
                overwrite: has_flag(&arguments[2..], &["--overwrite", "-overwrite"]),
                dry_run: has_flag(&arguments[2..], &["--dry-run", "-dry-run"]),
                reserved_names: RESERVED_COMMAND_NAMES
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect(),
                ..RestoreOptions::default()
            };
            let (archive, outcomes) = restore_at(&paths, Path::new(archive), &options)?;
            print_warnings(&archive.warnings);
            if !archive.manifest.tool.is_empty() && archive.manifest.tool != "goshcoder" {
                eprintln!(
                    "note: this archive was written by {}",
                    archive.manifest.tool
                );
            }
            for line in describe_restore(&outcomes) {
                eprintln!("{line}");
            }
            Ok(())
        }
        Some(subcommand) => Err(command_error(format!(
            "unknown prompts subcommand {subcommand:?}; use list, backup or restore"
        ))),
    }
}

fn collect(paths: &ResourcePaths) -> Result<PromptCollection, Box<dyn Error>> {
    Ok(resources::collect_prompts(
        paths,
        &[PromptScope::User, PromptScope::Workspace],
    )?)
}

/// Writes a prompt archive and returns discovery warnings alongside its path.
///
/// The output is deliberately returned rather than printed so an interactive
/// frontend can show it inside its transcript instead of corrupting an
/// alternate-screen terminal.
pub fn backup_at(
    paths: &ResourcePaths,
    requested: Option<&Path>,
) -> Result<(PathBuf, Vec<String>), Box<dyn Error>> {
    let collection = collect(paths)?;
    if collection.prompts.is_empty() {
        return Err(command_error("there are no saved prompts to back up"));
    }

    let path = backup_destination(requested);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = match options.open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            return Err(command_error(format!(
                "{} already exists; choose another name",
                path.display()
            )));
        }
        Err(error) => return Err(Box::new(error)),
    };
    if let Err(error) = resources::write_archive(&mut file, &collection.prompts) {
        drop(file);
        let _ = fs::remove_file(&path);
        return Err(Box::new(error));
    }
    if let Err(error) = file.sync_all() {
        drop(file);
        let _ = fs::remove_file(&path);
        return Err(Box::new(error));
    }
    drop(file);
    Ok((path, collection.warnings))
}

/// Reads and safely restores a prompt archive without prescribing terminal
/// output. Both the standalone CLI and the Ratatui frontend use this path.
pub fn restore_at(
    paths: &ResourcePaths,
    archive_path: &Path,
    options: &RestoreOptions,
) -> Result<(ArchiveRead, Vec<RestoreOutcome>), Box<dyn Error>> {
    let mut file = File::open(archive_path)?;
    let archive = resources::read_archive(&mut file)?;
    if archive.prompts.is_empty() {
        return Err(command_error("this archive holds no prompts"));
    }
    let outcomes = resources::restore_prompts(paths, &archive.prompts, options)?;
    Ok((archive, outcomes))
}

fn backup_destination(requested: Option<&Path>) -> PathBuf {
    let name = format!(
        "goshcoder-prompts-{}.tar.gz",
        OffsetDateTime::now_utc().date()
    );
    match requested {
        Some(path) if path.is_dir() => path.join(name),
        Some(path) => path.to_path_buf(),
        None => PathBuf::from(name),
    }
}

fn scope_label(scope: PromptScope) -> &'static str {
    match scope {
        PromptScope::User => "user",
        PromptScope::Workspace => "project",
    }
}

fn has_flag(arguments: &[String], names: &[&str]) -> bool {
    arguments
        .iter()
        .any(|argument| names.iter().any(|name| argument == name))
}

fn print_warnings(warnings: &[String]) {
    for warning in warnings {
        eprintln!("warning: {warning}");
    }
}

/// Produces the user-facing summary shared by the CLI and interactive prompt
/// command.
pub fn describe_restore(outcomes: &[RestoreOutcome]) -> Vec<String> {
    let mut restored = 0;
    let mut skipped = 0;
    let mut lines = Vec::with_capacity(outcomes.len() + 1);
    for outcome in outcomes {
        if outcome.skipped {
            skipped += 1;
            lines.push(format!(
                "skipped /{}: {}",
                outcome.name,
                outcome.reason.as_deref().unwrap_or("unknown reason")
            ));
        } else {
            restored += 1;
        }
    }
    lines.push(format!("restored {restored} prompt(s), skipped {skipped}"));
    lines
}

fn command_error(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{SaveTemplateOptions, save_template};
    use std::{
        process,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_nanos();
            let path = env::temp_dir().join(format!(
                "goshcoder-prompts-{}-{nonce}-{}",
                process::id(),
                TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create scratch directory");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn paths(root: &Scratch, name: &str) -> ResourcePaths {
        ResourcePaths::new(root.0.join(format!("{name}-workspace")), root.0.join(name))
            .expect("resource paths")
            .without_user_home()
    }

    #[test]
    fn backup_and_restore_cover_both_prompt_scopes() {
        let root = Scratch::new();
        let source = paths(&root, "source");
        save_template(
            &source,
            "user-prompt",
            "User prompt",
            SaveTemplateOptions::default(),
        )
        .expect("save user prompt");
        save_template(
            &source,
            "project-prompt",
            "Project prompt",
            SaveTemplateOptions {
                scope: PromptScope::Workspace,
                ..SaveTemplateOptions::default()
            },
        )
        .expect("save project prompt");

        let archive_path = root.0.join("prompts.tar.gz");
        let (archive_path, warnings) =
            backup_at(&source, Some(&archive_path)).expect("write backup");
        assert!(warnings.is_empty());
        let destination = paths(&root, "destination");
        let (decoded, outcomes) =
            restore_at(&destination, &archive_path, &RestoreOptions::default()).expect("restore");

        assert_eq!(decoded.manifest.prompts, 2);
        assert_eq!(
            outcomes.iter().filter(|outcome| !outcome.skipped).count(),
            2
        );
        let restored = collect(&destination).expect("collect restored prompts");
        assert_eq!(
            restored
                .prompts
                .iter()
                .map(|prompt| (prompt.name.as_str(), prompt.scope))
                .collect::<Vec<_>>(),
            vec![
                ("user-prompt", PromptScope::User),
                ("project-prompt", PromptScope::Workspace),
            ]
        );
    }

    #[test]
    fn restore_summary_and_option_flags_match_the_cli_contract() {
        let outcomes = vec![
            RestoreOutcome {
                name: "saved".to_owned(),
                scope: PromptScope::User,
                path: None,
                skipped: false,
                reason: None,
            },
            RestoreOutcome {
                name: "existing".to_owned(),
                scope: PromptScope::Workspace,
                path: None,
                skipped: true,
                reason: Some("a prompt with that name already exists".to_owned()),
            },
        ];
        assert_eq!(
            describe_restore(&outcomes),
            vec![
                "skipped /existing: a prompt with that name already exists",
                "restored 1 prompt(s), skipped 1",
            ]
        );
        assert!(has_flag(
            &["--dry-run".to_owned()],
            &["--dry-run", "-dry-run"]
        ));
        assert!(!has_flag(
            &["--other".to_owned()],
            &["--dry-run", "-dry-run"]
        ));
    }
}
