//! Session and agent integration for durable Ralph loops.
//!
//! [`crate::ralph`] owns the portable loop state and model-facing tools. This
//! adapter binds that state to one live agent: it installs an iteration-aware
//! system prompt after each assistant turn and routes loop completion notices
//! back through the session frontend.

use std::{
    error::Error as StdError,
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::{
    agent, llm, ralph,
    session::{SessionNoticeSender, SessionRuntime},
};

/// Callback that applies an extension-composed system prompt and any coupled
/// tool-set changes to the live agent.
pub type SystemPromptSync = Arc<dyn Fn(String) + Send + Sync + 'static>;

/// Errors while attaching or synchronizing a Ralph loop with a live session.
#[derive(Debug)]
pub enum RalphRuntimeError {
    Ralph(ralph::RalphError),
}

impl fmt::Display for RalphRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ralph(error) => error.fmt(formatter),
        }
    }
}

impl StdError for RalphRuntimeError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Ralph(error) => Some(error),
        }
    }
}

impl From<ralph::RalphError> for RalphRuntimeError {
    fn from(error: ralph::RalphError) -> Self {
        Self::Ralph(error)
    }
}

pub type Result<T> = std::result::Result<T, RalphRuntimeError>;

/// Keeps a Ralph store synchronized with one live agent session.
///
/// The subscription is retained for the session lifetime. It is registered
/// after Planner, so a completed assistant turn first updates planner state,
/// then applies the Ralph suffix beneath the current Planner prompt layer.
pub struct RalphRuntime {
    store: ralph::Store,
    base_system_prompt: Arc<Mutex<String>>,
    system_prompt_sync: SystemPromptSync,
    _subscription: agent::Subscription,
}

impl RalphRuntime {
    /// Attaches a store to a session that already has its Ralph tools
    /// installed. `system_prompt_sync` may apply Planner state on top of the
    /// supplied Ralph-expanded base prompt.
    pub fn attach(
        runtime: &SessionRuntime,
        store: ralph::Store,
        base_system_prompt: String,
        system_prompt_sync: SystemPromptSync,
    ) -> Result<Self> {
        let notices = runtime.notice_sender();
        let base_system_prompt = Arc::new(Mutex::new(base_system_prompt));
        let subscription = ralph_subscription(
            runtime.agent(),
            store.clone(),
            Arc::clone(&base_system_prompt),
            Arc::clone(&system_prompt_sync),
            notices.clone(),
        );
        let integration = Self {
            store,
            base_system_prompt,
            system_prompt_sync,
            _subscription: subscription,
        };
        integration.sync_agent()?;
        Ok(integration)
    }

    /// Returns the loop store for task loading and status rendering.
    #[must_use]
    pub fn store(&self) -> &ralph::Store {
        &self.store
    }

    /// Returns the active loop owned by this live session, if any.
    pub fn current(&self) -> Result<Option<ralph::LoopState>> {
        self.store.current().map_err(Into::into)
    }

    /// Rebuilds the active agent system prompt from the raw base prompt.
    ///
    /// Calling this before a user prompt protects against an external Ralph
    /// lifecycle update between agent turns.
    pub fn sync_agent(&self) -> Result<()> {
        let base_system_prompt = lock(&self.base_system_prompt).clone();
        let system_prompt = match self.store.current()? {
            Some(state) => ralph::inject_system_prompt(&base_system_prompt, &state),
            None => base_system_prompt,
        };
        (self.system_prompt_sync)(system_prompt);
        Ok(())
    }

    /// Replaces the prompt beneath the current Ralph and Planner layers.
    pub fn set_base_system_prompt(&self, prompt: impl Into<String>) -> Result<()> {
        *lock(&self.base_system_prompt) = prompt.into();
        self.sync_agent()
    }

    /// Executes one Ralph lifecycle command and synchronizes state-changing
    /// commands immediately, before a following model turn can observe stale
    /// instructions.
    pub fn execute(&self, command: ralph::RalphCommand) -> Result<ralph::CommandResult> {
        let mutates_agent_context = !matches!(
            &command,
            ralph::RalphCommand::List { .. } | ralph::RalphCommand::Status
        );
        let result = self.store.execute(command)?;
        if mutates_agent_context {
            self.sync_agent()?;
        }
        Ok(result)
    }
}

fn ralph_subscription(
    agent: &agent::Agent,
    store: ralph::Store,
    base_system_prompt: Arc<Mutex<String>>,
    system_prompt_sync: SystemPromptSync,
    notices: SessionNoticeSender,
) -> agent::Subscription {
    agent.subscribe(move |event| {
        if event.kind != agent::EventKind::TurnEnd {
            return;
        }
        let Some(llm::Message::Assistant(assistant)) = event.message else {
            return;
        };
        // A terminal provider failure must leave the active loop at the same
        // iteration. The normal before-prompt synchronization will restore
        // its instructions when the user retries.
        if !assistant.error_message.is_empty() {
            return;
        }

        let base_system_prompt = lock(&base_system_prompt).clone();
        match store.prepare_next_turn(&base_system_prompt, &assistant) {
            Ok(preparation) => {
                system_prompt_sync(preparation.system_prompt);
                if preparation.completed
                    && let Some(state) = preparation.active_loop
                {
                    notices.push(
                        "Ralph",
                        format!(
                            "loop {:?} completed at iteration {}",
                            state.name, state.iteration
                        ),
                    );
                }
            }
            Err(error) => notices.push(
                "Ralph",
                format!("could not update the loop after this turn: {error}"),
            ),
        }
    })
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::{
            Arc, Mutex,
            atomic::{AtomicU64, Ordering},
        },
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;
    use crate::{
        llm,
        session::{SessionOptions, SessionSelection},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "goshcoder-ralph-runtime-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    fn model() -> llm::Model {
        llm::Model {
            provider: "test".to_owned(),
            id: "test-model".to_owned(),
            api: "test".to_owned(),
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
            timestamp: 1,
            ..llm::AssistantMessage::default()
        }
    }

    #[test]
    fn completion_marker_closes_loop_before_the_next_request() {
        let directory = temporary_directory();
        let seen_contexts = Arc::new(Mutex::new(Vec::new()));
        let responder: agent::AssistantResponder = {
            let seen_contexts = Arc::clone(&seen_contexts);
            Arc::new(move |_, context, _| {
                seen_contexts
                    .lock()
                    .expect("contexts lock")
                    .push(context.clone());
                Ok(assistant_text(ralph::COMPLETE_MARKER))
            })
        };
        let mut runtime = SessionRuntime::open(SessionOptions {
            cwd: directory.clone(),
            sessions_dir: Some(directory.join("sessions")),
            selection: SessionSelection::NoSession,
            system_prompt: "base prompt".to_owned(),
            model: model(),
            responder: Some(responder),
            ..SessionOptions::default()
        })
        .expect("open runtime");
        let agent = runtime.agent().clone();
        let store = ralph::Store::new(directory.join("config"), &directory, "test-session");
        let queue: ralph::SharedQueue = Arc::new(agent.weak_follow_up_queue());
        agent.set_tools(store.tools(Some(queue)));
        let prompt_sync: SystemPromptSync = {
            let agent = agent.clone();
            Arc::new(move |prompt| agent.set_system_prompt(prompt))
        };
        let integration = RalphRuntime::attach(
            &runtime,
            store.clone(),
            "base prompt".to_owned(),
            prompt_sync,
        )
        .expect("attach Ralph");

        let started = integration
            .execute(ralph::RalphCommand::Start {
                name: "migration".to_owned(),
                task_content: "# Task\n\nFinish the migration".to_owned(),
                options: ralph::LoopOptions {
                    max_iterations: 2,
                    ..ralph::LoopOptions::default()
                },
            })
            .expect("start Ralph loop");
        assert!(matches!(started, ralph::CommandResult::Started(_)));
        assert!(
            agent
                .state()
                .system_prompt
                .contains("[RALPH LOOP - migration - Iteration 1/2]"),
            "active loop suffix was not installed"
        );

        agent.prompt("begin work").expect("run turn");

        let state = store
            .load("migration", false)
            .expect("load loop")
            .expect("loop exists");
        assert_eq!(state.status, ralph::LoopStatus::Completed);
        assert!(!state.completed_at.is_empty());
        assert!(!agent.state().system_prompt.contains("[RALPH LOOP"));
        assert!(
            seen_contexts
                .lock()
                .expect("contexts lock")
                .first()
                .expect("provider context")
                .system_prompt
                .contains("[RALPH LOOP - migration - Iteration 1/2]")
        );
        assert!(runtime.drain_notices().iter().any(|notice| {
            notice.kind == "Ralph" && notice.text.contains("completed at iteration 1")
        }));

        drop(integration);
        runtime.close().expect("close runtime");
        fs::remove_dir_all(directory).expect("remove temporary directory");
    }
}
