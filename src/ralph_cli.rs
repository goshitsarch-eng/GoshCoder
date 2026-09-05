//! Command-line adapter for durable Ralph loop lifecycle operations.

use std::{env, error::Error, process};

use crate::ralph::{self, CommandResult, Store};

/// Executes `goshcoder ralph …` against the current workspace's loop store.
pub fn command(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let workspace = env::current_dir()?;
    let store = Store::for_workspace(workspace, format!("cli-{}", process::id()));
    let command = ralph::parse_command_args(arguments)?;
    let archived_list = matches!(&command, ralph::RalphCommand::List { archived: true });
    let result = store.execute(command)?;
    print_result(&store, result, archived_list)?;
    Ok(())
}

fn print_result(store: &Store, result: CommandResult, archived_list: bool) -> ralph::Result<()> {
    match result {
        CommandResult::Started(state) => {
            println!("Started {} at iteration {}.", state.name, state.iteration);
        }
        CommandResult::Listed(states) => {
            if states.is_empty() {
                if archived_list {
                    println!("No archived loops.");
                } else {
                    println!("No loops. Start one with 'goshcoder run -ralph ...'.");
                }
            } else {
                for state in states {
                    println!("{}", state.summary());
                }
            }
        }
        CommandResult::Status(Some(state)) => {
            println!("{}", state.summary());
            let status_line = store.status_line()?;
            if !status_line.is_empty() {
                println!("{status_line}");
            }
        }
        CommandResult::Status(None) => {
            println!("No active loop in this workspace.");
        }
        CommandResult::Resumed(state) => {
            println!("Resumed {} at iteration {}.", state.name, state.iteration);
        }
        CommandResult::Stopped(state) => {
            println!("Stopped {} at iteration {}.", state.name, state.iteration);
        }
        CommandResult::Archived(name) => {
            println!("Archived {name}.");
        }
        CommandResult::Deleted(name) => {
            println!("Deleted {name}.");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ralph::{LoopOptions, RalphCommand};
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!(
            "goshcoder-ralph-cli-{}-{nonce}-{}",
            process::id(),
            TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    #[test]
    fn command_results_cover_the_external_lifecycle_messages() {
        let directory = temporary_directory();
        let store = Store::new(&directory, "test-session");
        let state = store
            .execute(RalphCommand::Start {
                name: "migration".to_owned(),
                task_content: "Task".to_owned(),
                options: LoopOptions::default(),
            })
            .expect("start loop");
        let CommandResult::Started(state) = state else {
            panic!("start result");
        };
        assert_eq!(state.name, "migration");
        let listed = store
            .execute(RalphCommand::List { archived: false })
            .expect("list loops");
        let CommandResult::Listed(states) = listed else {
            panic!("list result");
        };
        assert_eq!(states.len(), 1);
        let stopped = store
            .execute(RalphCommand::Stop {
                name: Some("migration".to_owned()),
            })
            .expect("stop loop");
        assert!(matches!(stopped, CommandResult::Stopped(_)));
        fs::remove_dir_all(directory).expect("remove temporary directory");
    }
}
