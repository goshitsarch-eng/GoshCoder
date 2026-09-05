//! Post-run turn policy: what happens between an agent run ending and the
//! next prompt.
//!
//! Ported from pi's `packages/coding-agent/src/core/agent-session.ts`
//! (`_runAgentPrompt`, `_handlePostAgentRun`, `_prepareRetry`,
//! `_checkCompaction`) and `packages/ai/src/utils/overflow.ts`. A transient
//! provider failure is retried with exponential backoff; a context overflow
//! is answered with one compact-and-retry attempt; a threshold crossing
//! compacts without retrying; and queued steering or follow-up messages keep
//! the run going. The agent loop itself knows nothing of this, exactly as
//! pi's `Agent` does not: the policy lives one layer up.

use std::{thread, time::Duration};

use crate::{agent, compaction, llm, session::SessionNoticeSender, stream};

/// pi's `settings.retry` defaults: three attempts, 2 s doubling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryPolicy {
    pub enabled: bool,
    pub max_retries: u32,
    pub base_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    /// Backoff before `attempt` (1-based): 2 s, 4 s, 8 s with the defaults.
    fn delay(&self, attempt: u32) -> Duration {
        let doubling = 1_u32 << attempt.saturating_sub(1).min(16);
        self.base_delay.saturating_mul(doubling)
    }
}

/// What the policy did after the prompt, for callers that report it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TurnReport {
    /// Transient errors retried on the way to the final message.
    pub retries: u32,
    /// Whether a compact-and-retry recovered from a context overflow.
    pub overflow_recovery: bool,
    /// Automatic compactions performed after the run.
    pub compactions: u32,
}

/// Runs one user prompt to completion under the policy.
///
/// The pre-prompt threshold compaction the callers already relied on stays in
/// place; the post-run handling is what pi adds on top of the raw agent.
pub fn run_prompt(
    agent: &agent::Agent,
    prompt: impl Into<String>,
    policy: &RetryPolicy,
    notices: Option<&SessionNoticeSender>,
) -> Result<TurnReport, String> {
    run_prompt_with_sleep(agent, prompt, policy, notices, thread::sleep)
}

fn run_prompt_with_sleep(
    agent: &agent::Agent,
    prompt: impl Into<String>,
    policy: &RetryPolicy,
    notices: Option<&SessionNoticeSender>,
    sleep: impl Fn(Duration),
) -> Result<TurnReport, String> {
    let outcome = compaction::maybe_auto_compact(agent).map_err(|error| error.to_string())?;
    report_dropped_queue(notices, outcome.as_ref());
    agent.prompt(prompt).map_err(|error| error.to_string())?;
    finish_run(agent, policy, notices, sleep)
}

/// The post-run loop shared by every prompt path.
fn finish_run(
    agent: &agent::Agent,
    policy: &RetryPolicy,
    notices: Option<&SessionNoticeSender>,
    sleep: impl Fn(Duration),
) -> Result<TurnReport, String> {
    let mut report = TurnReport::default();
    let mut attempt = 0_u32;
    let mut overflow_attempted = false;
    loop {
        let state = agent.state();
        let Some(last) = last_assistant(&state.messages) else {
            break;
        };
        if last.stop_reason != stream::STOP_ERROR {
            // pi resets the retry counter on any successful response.
            attempt = 0;
        }
        // An overflow error from a model with a smaller window must not
        // trigger compaction for the larger one the user just switched to.
        let same_model = last.provider == state.model.provider && last.model == state.model.id;
        let overflow = same_model && is_context_overflow(last, state.model.context_window);
        let recoverable_length = same_model && is_recoverable_length(last, state.model.max_tokens);

        // Case 1: a transient provider failure. Context overflow is handled
        // by compaction, never by retrying the same request.
        if !overflow && policy.enabled && stream::is_retryable_assistant_error(last) {
            if attempt < policy.max_retries {
                attempt += 1;
                let delay = policy.delay(attempt);
                notify(
                    notices,
                    "retry",
                    format!(
                        "attempt {attempt} of {} in {:.0}s: {}",
                        policy.max_retries,
                        delay.as_secs_f64(),
                        error_summary(last)
                    ),
                );
                // The failed message stays in the session file but leaves the
                // context the retry sees.
                drop_last_assistant(agent, &state);
                sleep(delay);
                agent.continue_run().map_err(|error| error.to_string())?;
                report.retries = report.retries.max(attempt);
                continue;
            }
            notify(
                notices,
                "retry",
                format!(
                    "giving up after {attempt} attempt(s): {}",
                    error_summary(last)
                ),
            );
        }

        // Cases 2 and 3: the context no longer fits. A failed or truncated
        // turn is compacted and retried once; a completed one is compacted
        // but kept, because a completed assistant turn cannot be continued.
        if overflow || recoverable_length {
            let will_retry = last.stop_reason != stream::STOP_STOP;
            if will_retry && overflow_attempted {
                notify(
                    notices,
                    "compaction",
                    if overflow {
                        "context overflow recovery failed after one compact-and-retry attempt; reduce context or switch to a larger-context model"
                    } else {
                        "truncated response recovery failed after one compact-and-retry attempt"
                    },
                );
            } else if will_retry {
                overflow_attempted = true;
                drop_last_assistant(agent, &state);
                match compaction::compact(agent, "") {
                    Ok(outcome) => {
                        report.compactions += 1;
                        report_dropped_queue(notices, Some(&outcome));
                        notify(
                            notices,
                            "compaction",
                            "context overflow: compacted and retrying the turn once",
                        );
                        agent.continue_run().map_err(|error| error.to_string())?;
                        report.overflow_recovery = true;
                        continue;
                    }
                    Err(error) => {
                        notify(
                            notices,
                            "compaction",
                            format!("context overflow, and compaction failed: {error}"),
                        );
                    }
                }
            } else if !overflow_attempted {
                overflow_attempted = true;
                match compaction::compact(agent, "") {
                    Ok(outcome) => {
                        report.compactions += 1;
                        report_dropped_queue(notices, Some(&outcome));
                    }
                    Err(error) => notify(
                        notices,
                        "compaction",
                        format!("context exceeded the window, and compaction failed: {error}"),
                    ),
                }
            }
        } else if let Some(outcome) =
            compaction::maybe_auto_compact(agent).map_err(|error| error.to_string())?
        {
            // Case 4: threshold crossed by a completed turn; compact between
            // turns so a long tool-heavy run cannot outgrow the window before
            // the next prompt.
            report.compactions += 1;
            report_dropped_queue(notices, Some(&outcome));
        }

        // Anything queued by a listener after the loop drained its queues
        // needs a continuation, as pi's post-run loop provides.
        if agent.has_queued_messages() {
            agent.continue_run().map_err(|error| error.to_string())?;
            continue;
        }
        break;
    }
    Ok(report)
}

fn last_assistant(messages: &[llm::Message]) -> Option<&llm::AssistantMessage> {
    match messages.last() {
        Some(llm::Message::Assistant(message)) => Some(message),
        _ => None,
    }
}

fn drop_last_assistant(agent: &agent::Agent, state: &agent::State) {
    if !matches!(state.messages.last(), Some(llm::Message::Assistant(_))) {
        return;
    }
    let mut messages = state.messages.clone();
    messages.pop();
    agent.set_context(messages, state.compactions.clone());
}

fn error_summary(message: &llm::AssistantMessage) -> String {
    let summary = message.error_message.trim();
    if summary.is_empty() {
        return "unknown error".to_owned();
    }
    let mut clipped = summary.chars().take(200).collect::<String>();
    if clipped.len() < summary.len() {
        clipped.push('…');
    }
    clipped
}

fn notify(notices: Option<&SessionNoticeSender>, kind: &str, text: impl Into<String>) {
    if let Some(notices) = notices {
        notices.push(kind, text);
    }
}

/// Compaction clears the steering and follow-up queues; anything typed while
/// the summary was being written is gone and the user should hear so.
pub fn report_dropped_queue(
    notices: Option<&SessionNoticeSender>,
    outcome: Option<&compaction::Outcome>,
) {
    if let Some(outcome) = outcome
        && outcome.dropped_queued_messages > 0
    {
        notify(
            notices,
            "compaction",
            format!(
                "{} queued message(s) were discarded by compaction",
                outcome.dropped_queued_messages
            ),
        );
    }
}

/// Whether an assistant message means the input no longer fits the model.
///
/// Three shapes, from pi's `isContextOverflow`: a provider error whose text
/// matches a known overflow message; a successful reply whose input usage
/// exceeds the window (z.ai accepts overflow silently); and a `length` stop
/// with no output at all while the input fills the window (Xiaomi MiMo
/// truncates the prompt to fit and leaves no room to answer).
pub fn is_context_overflow(message: &llm::AssistantMessage, context_window: u64) -> bool {
    if message.stop_reason == stream::STOP_ERROR && !message.error_message.is_empty() {
        let text = message.error_message.to_lowercase();
        if !is_non_overflow_error(&text) && matches_overflow_pattern(&text) {
            return true;
        }
    }
    let input_tokens = message.usage.input.saturating_add(message.usage.cache_read);
    if context_window > 0
        && message.stop_reason == stream::STOP_STOP
        && input_tokens > context_window
    {
        return true;
    }
    if context_window > 0
        && message.stop_reason == stream::STOP_LENGTH
        && message.usage.output == 0
        && input_tokens as f64 >= context_window as f64 * 0.99
    {
        return true;
    }
    false
}

/// A `length` stop that ended below the model's own output limit was cut by a
/// context-clamped request limit, not by the model, and is worth one
/// compact-and-retry.
pub fn is_recoverable_length(message: &llm::AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == stream::STOP_LENGTH
        && desired_max_output > 0
        && message.usage.output < desired_max_output
}

/// Throttling and rate-limit errors can mention tokens too ("Too many tokens,
/// please wait"); they are never overflow.
fn is_non_overflow_error(text: &str) -> bool {
    text.starts_with("throttling error:")
        || text.starts_with("service unavailable:")
        || text.contains("rate limit")
        || text.contains("too many requests")
}

/// The provider messages pi recognizes, matched on lowercase text. Patterns
/// that carried a number in pi are matched on their fixed words.
fn matches_overflow_pattern(text: &str) -> bool {
    const SUBSTRINGS: &[&str] = &[
        "prompt is too long",                    // Anthropic
        "request_too_large",                     // Anthropic 413
        "input is too long for requested model", // Bedrock
        "exceeds the context window",            // OpenAI
        "input token count",                     // Google (with "exceeds the maximum" below)
        "maximum prompt length is",              // xAI
        "reduce the length of the messages",     // Groq
        "maximum context length is",             // OpenRouter
        "maximum allowed input length of",       // OpenRouter/Poolside
        "exceeds the limit of",                  // GitHub Copilot
        "exceeds the available context size",    // llama.cpp
        "greater than the context length",       // LM Studio
        "context window exceeds limit",          // MiniMax
        "exceeded model token limit",            // Kimi
        "maximum context length", // Mistral ("too large for model with N maximum context length")
        "but the configured context size is", // DS4
        "model_context_window_exceeded", // z.ai
        "exceeded max context length", // Ollama
        "exceeded context length", // Ollama
        "range of input length should be", // DashScope
        "context_length_exceeded", // generic
        "context length exceeded", // generic
        "context length_exceeded",
        "context_length exceeded",
        "too many tokens",      // generic
        "token limit exceeded", // generic
    ];
    if text.contains("input token count") {
        return text.contains("exceeds the maximum");
    }
    if text.contains("is longer than the model") && text.contains("context length") {
        return true; // Together AI
    }
    if (text.starts_with("400") || text.starts_with("413")) && text.contains("(no body)") {
        return true; // Cerebras
    }
    SUBSTRINGS
        .iter()
        .any(|pattern| *pattern != "input token count" && text.contains(pattern))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{Agent, AgentOptions, InitialState};
    use std::sync::{Arc, Mutex};

    fn model() -> llm::Model {
        llm::Model {
            id: "test-model".to_owned(),
            name: "Test".to_owned(),
            api: "openai-completions".to_owned(),
            provider: "test".to_owned(),
            context_window: 200_000,
            max_tokens: 100,
            ..llm::Model::default()
        }
    }

    fn assistant(stop_reason: &str, text: &str, error: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            content: vec![llm::ContentBlock::text(text)],
            api: "openai-completions".to_owned(),
            provider: "test".to_owned(),
            model: "test-model".to_owned(),
            stop_reason: stop_reason.to_owned(),
            error_message: error.to_owned(),
            ..llm::AssistantMessage::default()
        }
    }

    /// Enough transcript that compaction has something older than the
    /// retained tail to summarize.
    fn long_history() -> Vec<llm::Message> {
        let mut history = Vec::new();
        for index in 0..40 {
            history.push(llm::Message::User(llm::UserMessage::text(
                format!("earlier question {index} {}", "x".repeat(4000)),
                index,
            )));
            history.push(llm::Message::Assistant(Box::new(assistant(
                "stop",
                &format!("earlier answer {index} {}", "y".repeat(4000)),
                "",
            ))));
        }
        history
    }

    /// A responder that answers from a script and records how many requests
    /// it saw.
    fn scripted_agent(script: Vec<llm::AssistantMessage>) -> (Agent, Arc<Mutex<usize>>) {
        let calls = Arc::new(Mutex::new(0_usize));
        let script = Arc::new(Mutex::new(script));
        let counter = Arc::clone(&calls);
        let agent = Agent::new(AgentOptions {
            initial_state: InitialState {
                model: model(),
                ..InitialState::default()
            },
            responder: Some(Arc::new(move |_, _, _| {
                *counter.lock().expect("calls") += 1;
                let mut script = script.lock().expect("script");
                if script.is_empty() {
                    return Ok(assistant("stop", "fallback", ""));
                }
                Ok(script.remove(0))
            })),
            ..AgentOptions::default()
        });
        (agent, calls)
    }

    #[test]
    fn transient_errors_are_retried_with_backoff_and_dropped_from_context() {
        let (agent, calls) = scripted_agent(vec![
            assistant("error", "", "503 service unavailable"),
            assistant("error", "", "overloaded_error: try again"),
            assistant("stop", "done", ""),
        ]);
        let slept = Arc::new(Mutex::new(Vec::new()));
        let sleeps = Arc::clone(&slept);
        let report = run_prompt_with_sleep(
            &agent,
            "hello",
            &RetryPolicy::default(),
            None,
            move |delay| sleeps.lock().expect("sleeps").push(delay),
        )
        .expect("prompt succeeds");
        assert_eq!(*calls.lock().expect("calls"), 3);
        assert_eq!(report.retries, 2);
        assert_eq!(
            *slept.lock().expect("sleeps"),
            [Duration::from_secs(2), Duration::from_secs(4)]
        );
        let messages = agent.state().messages;
        assert_eq!(
            messages.len(),
            2,
            "failed attempts leave the context: {messages:?}"
        );
        assert_eq!(
            last_assistant(&messages).map(|message| message.stop_reason.as_str()),
            Some("stop")
        );
    }

    #[test]
    fn retries_stop_at_the_policy_limit() {
        let (agent, calls) = scripted_agent(vec![
            assistant("error", "", "server error"),
            assistant("error", "", "server error"),
            assistant("error", "", "server error"),
            assistant("error", "", "server error"),
            assistant("stop", "never reached", ""),
        ]);
        let report = run_prompt_with_sleep(
            &agent,
            "hello",
            &RetryPolicy {
                max_retries: 2,
                ..RetryPolicy::default()
            },
            None,
            |_| {},
        )
        .expect("prompt returns the final error as a message");
        assert_eq!(*calls.lock().expect("calls"), 3);
        assert_eq!(report.retries, 2);
        assert_eq!(
            last_assistant(&agent.state().messages).map(|message| message.stop_reason.as_str()),
            Some("error")
        );
    }

    #[test]
    fn disabled_policy_never_retries() {
        let (agent, calls) = scripted_agent(vec![assistant("error", "", "server error")]);
        run_prompt_with_sleep(
            &agent,
            "hello",
            &RetryPolicy {
                enabled: false,
                ..RetryPolicy::default()
            },
            None,
            |_| panic!("no sleep expected"),
        )
        .expect("prompt");
        assert_eq!(*calls.lock().expect("calls"), 1);
    }

    #[test]
    fn overflow_detection_covers_error_text_silent_and_length_shapes() {
        let error = assistant(
            "error",
            "",
            "prompt is too long: 213462 tokens > 200000 maximum",
        );
        assert!(is_context_overflow(&error, 200_000));
        let throttled = assistant(
            "error",
            "",
            "Throttling error: Too many tokens, please wait",
        );
        assert!(!is_context_overflow(&throttled, 200_000));
        let google = assistant(
            "error",
            "",
            "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        );
        assert!(is_context_overflow(&google, 0));
        let cerebras = assistant("error", "", "413 status code (no body)");
        assert!(is_context_overflow(&cerebras, 0));
        let unrelated = assistant("error", "", "invalid api key");
        assert!(!is_context_overflow(&unrelated, 1000));

        let mut silent = assistant("stop", "ok", "");
        silent.usage.input = 900;
        silent.usage.cache_read = 200;
        assert!(is_context_overflow(&silent, 1000));
        assert!(
            !is_context_overflow(&silent, 0),
            "no window, no silent check"
        );

        let mut truncated = assistant("length", "", "");
        truncated.usage.input = 995;
        assert!(is_context_overflow(&truncated, 1000));
        truncated.usage.output = 5;
        assert!(!is_context_overflow(&truncated, 1000));
        assert!(is_recoverable_length(&truncated, 100));
        assert!(!is_recoverable_length(&truncated, 5));
    }

    #[test]
    fn overflow_errors_compact_and_retry_exactly_once() {
        let (agent, calls) = scripted_agent(vec![
            assistant(
                "error",
                "",
                "prompt is too long: 2000 tokens > 1000 maximum",
            ),
            assistant("stop", "summary of the earlier work", ""),
            assistant("stop", "recovered", ""),
        ]);
        agent.set_context(long_history(), Vec::new());
        let report = run_prompt_with_sleep(&agent, "hello", &RetryPolicy::default(), None, |_| {
            panic!("overflow is not a retry")
        })
        .expect("prompt");
        assert!(report.overflow_recovery);
        assert_eq!(report.compactions, 1);
        // The failed turn, the summarizer, and the retried turn.
        assert_eq!(*calls.lock().expect("calls"), 3);
        let messages = agent.state().messages;
        assert_eq!(
            last_assistant(&messages).map(|message| message.stop_reason.as_str()),
            Some("stop")
        );
        assert!(
            messages.iter().all(|message| !matches!(
                message,
                llm::Message::Assistant(inner) if inner.stop_reason == "error"
            )),
            "the overflowing turn leaves the context"
        );
    }

    #[test]
    fn a_second_overflow_is_reported_not_retried_forever() {
        let (agent, calls) = scripted_agent(vec![
            assistant("error", "", "prompt is too long"),
            assistant("stop", "summary", ""),
            assistant("error", "", "prompt is too long"),
            assistant("stop", "never requested", ""),
        ]);
        agent.set_context(long_history(), Vec::new());
        let report = run_prompt_with_sleep(&agent, "hello", &RetryPolicy::default(), None, |_| {})
            .expect("prompt");
        // The failed turn, the summarizer, the retried turn; the second
        // overflow is reported and nothing more is requested.
        assert_eq!(*calls.lock().expect("calls"), 3);
        assert!(report.overflow_recovery);
        assert_eq!(
            last_assistant(&agent.state().messages).map(|message| message.stop_reason.as_str()),
            Some("error")
        );
    }

    #[test]
    fn queued_messages_keep_the_run_going() {
        let (agent, calls) = scripted_agent(vec![
            assistant("stop", "first", ""),
            assistant("stop", "second", ""),
        ]);
        // A listener queues a follow-up as the run ends, the way an extension
        // hook would; the post-run loop must pick it up.
        let queued = Arc::new(Mutex::new(false));
        let flag = Arc::clone(&queued);
        let hook_agent = agent.clone();
        let _subscription = agent.subscribe(move |event| {
            if event.kind == agent::EventKind::AgentEnd
                && !std::mem::replace(&mut *flag.lock().expect("flag"), true)
            {
                hook_agent.follow_up(llm::Message::User(llm::UserMessage::text("more", 1)));
            }
        });
        run_prompt_with_sleep(&agent, "hello", &RetryPolicy::default(), None, |_| {})
            .expect("prompt");
        let preview = agent
            .state()
            .messages
            .iter()
            .map(|message| format!("{}:{}", message.role(), message.text_preview()))
            .collect::<Vec<_>>();
        assert_eq!(*calls.lock().expect("calls"), 2, "{preview:?}");
        assert_eq!(
            last_assistant(&agent.state().messages)
                .and_then(|message| message.content[0].plain_text()),
            Some("second")
        );
    }
}
