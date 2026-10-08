//! `/login` inside the fullscreen interface.
//!
//! pi runs its OAuth and API-key logins in its own interface
//! (`packages/coding-agent/src/modes/interactive/components/login-dialog.ts`)
//! instead of handing the terminal to a child process. This module is the
//! equivalent: the provider-neutral flow in `oauth` runs on a worker thread,
//! talking to the interface through [`TuiInteraction`]; its notifications
//! become transcript notices and its prompts become composer questions. An
//! API key is typed into the composer, masked, and stored the way
//! `goshcoder auth set` stores it.

use std::{
    sync::{
        Arc,
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    catalog::{Credential, CredentialStore},
    config, oauth,
};

/// What the login worker tells the interface.
pub enum LoginEvent {
    Notify(oauth::OAuthEvent),
    Prompt {
        prompt: PromptRequest,
        reply: Sender<Option<String>>,
    },
    Finished(Result<(), String>),
}

/// A question from the flow, without its cancellation token.
pub struct PromptRequest {
    pub message: String,
    pub options: Vec<oauth::OAuthPromptOption>,
    pub select: bool,
    /// An example of what to paste (a redirect URL), not a label.
    pub placeholder: String,
}

/// Bridges `oauth`'s blocking interaction trait to a channel the event loop
/// drains between frames.
pub struct TuiInteraction {
    events: Sender<LoginEvent>,
}

impl TuiInteraction {
    pub fn new(events: Sender<LoginEvent>) -> Self {
        Self { events }
    }
}

impl oauth::OAuthInteraction for TuiInteraction {
    fn prompt(&self, prompt: oauth::OAuthPrompt) -> oauth::Result<String> {
        prompt.cancellation.check()?;
        let (reply, answers) = mpsc::channel();
        let select = prompt.kind == oauth::OAuthPromptKind::Select;
        let options = prompt.options.clone();
        self.events
            .send(LoginEvent::Prompt {
                prompt: PromptRequest {
                    message: prompt.message.clone(),
                    options: prompt.options.clone(),
                    select,
                    placeholder: prompt.placeholder.clone(),
                },
                reply,
            })
            .map_err(|_| oauth::OAuthError::Cancelled)?;
        // `prompt` must return soon after cancellation, so a loopback
        // callback that wins the race is not held up by an unanswered paste.
        loop {
            prompt.cancellation.check()?;
            match answers.recv_timeout(Duration::from_millis(100)) {
                Ok(Some(answer)) if select => return Ok(select_option(&answer, &options)),
                Ok(Some(answer)) => return Ok(answer),
                Ok(None) | Err(RecvTimeoutError::Disconnected) => {
                    return Err(oauth::OAuthError::Cancelled);
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }

    fn notify(&self, event: oauth::OAuthEvent) {
        let _ = self.events.send(LoginEvent::Notify(event));
    }
}

/// A numbered answer picks that option; anything else is passed through.
fn select_option(input: &str, options: &[oauth::OAuthPromptOption]) -> String {
    if let Ok(index) = input.trim().parse::<usize>()
        && let Some(option) = index.checked_sub(1).and_then(|index| options.get(index))
    {
        return option.id.clone();
    }
    input.trim().to_owned()
}

/// A login in progress.
pub enum LoginFlow {
    OAuth {
        provider: String,
        events: Receiver<LoginEvent>,
        cancellation: oauth::CancellationToken,
        reply: Option<Sender<Option<String>>>,
        started: Instant,
    },
    ApiKey {
        provider: String,
        /// Extra values the provider needs beside the key (Cloudflare's
        /// account and gateway ids), still to be asked for.
        fields: Vec<(&'static str, &'static str)>,
        key: Option<String>,
        environment: Vec<(&'static str, String)>,
        store: CredentialStore,
    },
}

impl LoginFlow {
    pub fn provider(&self) -> &str {
        match self {
            Self::OAuth { provider, .. } | Self::ApiKey { provider, .. } => provider,
        }
    }

    /// Starts an OAuth login on a worker thread.
    pub fn start_oauth(provider_id: &str, store: CredentialStore) -> Result<Self, String> {
        let provider = oauth::OAuthProviderId::parse(provider_id)
            .ok_or_else(|| format!("{provider_id} has no browser or device login"))?;
        config::ensure_agent_dir().map_err(|error| error.to_string())?;
        let client = oauth::OAuthClient::system().map_err(|error| error.to_string())?;
        let cancellation = oauth::CancellationToken::new();
        let (sender, events) = mpsc::channel();
        let worker_cancellation = cancellation.clone();
        let interaction = Arc::new(TuiInteraction::new(sender.clone()));
        thread::spawn(move || {
            let result = client
                .login_and_persist(
                    provider,
                    &store,
                    interaction,
                    &oauth::ProcessEnvironment,
                    &worker_cancellation,
                )
                .map(|_| ())
                .map_err(|error| error.to_string());
            let _ = sender.send(LoginEvent::Finished(result));
        });
        Ok(Self::OAuth {
            provider: provider_id.to_owned(),
            events,
            cancellation,
            reply: None,
            started: Instant::now(),
        })
    }

    /// Starts an API-key login; the composer asks for the key next.
    pub fn start_api_key(
        provider_id: &str,
        fields: &[(&'static str, &'static str)],
        store: CredentialStore,
    ) -> Self {
        Self::ApiKey {
            provider: provider_id.to_owned(),
            fields: fields.to_vec(),
            key: None,
            environment: Vec::new(),
            store,
        }
    }

    /// The composer question an API-key login is waiting on, if any.
    pub fn api_key_question(&self) -> Option<(String, bool)> {
        let Self::ApiKey {
            provider,
            fields,
            key,
            ..
        } = self
        else {
            return None;
        };
        if key.is_none() {
            return Some((format!("API key for {provider}"), true));
        }
        fields.first().map(|(_, prompt)| {
            (
                prompt
                    .trim()
                    .trim_start_matches("Enter the ")
                    .trim_end_matches(':')
                    .to_owned(),
                false,
            )
        })
    }

    /// Takes one composer answer. Returns `Some(result)` once the login has
    /// finished (stored or failed) and `None` while it needs more input.
    pub fn answer(&mut self, answer: String) -> Option<Result<(), String>> {
        match self {
            Self::OAuth { reply, .. } => {
                if let Some(reply) = reply.take() {
                    let _ = reply.send(Some(answer));
                }
                None
            }
            Self::ApiKey {
                provider,
                fields,
                key,
                environment,
                store,
            } => {
                if answer.is_empty() {
                    return Some(Err(if key.is_none() {
                        "no key provided".to_owned()
                    } else {
                        "no value provided".to_owned()
                    }));
                }
                if key.is_none() {
                    *key = Some(answer);
                } else if !fields.is_empty() {
                    let (name, _) = fields.remove(0);
                    environment.push((name, answer));
                }
                if !fields.is_empty() {
                    return None;
                }
                let mut credential = Credential::api_key(key.take().unwrap_or_default());
                for (name, value) in environment.drain(..) {
                    credential.set_environment(name, value);
                }
                Some(
                    config::ensure_agent_dir()
                        .map_err(|error| error.to_string())
                        .and_then(|_| {
                            store
                                .put(provider, credential)
                                .map(|_| ())
                                .map_err(|error| error.to_string())
                        }),
                )
            }
        }
    }

    /// Abandons the login: the worker sees the cancellation and stops.
    pub fn cancel(&mut self) {
        if let Self::OAuth {
            cancellation,
            reply,
            ..
        } = self
        {
            cancellation.cancel();
            if let Some(reply) = reply.take() {
                let _ = reply.send(None);
            }
        }
    }

    pub fn started(&self) -> Option<Instant> {
        match self {
            Self::OAuth { started, .. } => Some(*started),
            Self::ApiKey { .. } => None,
        }
    }

    /// The next event from an OAuth worker, without blocking.
    pub fn poll(&mut self) -> Option<LoginEvent> {
        match self {
            Self::OAuth { events, reply, .. } => match events.try_recv() {
                Ok(LoginEvent::Prompt {
                    prompt,
                    reply: answer,
                }) => {
                    *reply = Some(answer.clone());
                    Some(LoginEvent::Prompt {
                        prompt,
                        reply: answer,
                    })
                }
                Ok(event) => Some(event),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => Some(LoginEvent::Finished(Err(
                    "the login worker stopped unexpectedly".to_owned(),
                ))),
            },
            Self::ApiKey { .. } => None,
        }
    }
}

/// What of an OAuth event belongs on the clipboard: the device code to type,
/// or the address to open when the browser did not. The fullscreen interface
/// wraps a long address over several rows, which a terminal cannot select as
/// one piece or recognise as one link.
pub fn clipboard_text(event: &oauth::OAuthEvent) -> Option<&str> {
    match event.kind {
        oauth::OAuthEventKind::DeviceCode => {
            Some(event.user_code.as_str()).filter(|code| !code.is_empty())
        }
        oauth::OAuthEventKind::AuthorizationUrl => event.authorization_url.as_deref(),
        _ => None,
    }
}

/// The transcript notice for an OAuth event. Indented lines are what the
/// interface renders as an address (blue) or a code (bold accent).
pub fn event_notice(provider: &str, event: &oauth::OAuthEvent) -> Option<String> {
    match event.kind {
        oauth::OAuthEventKind::DeviceCode => {
            let minutes = event.expires_in_seconds.div_ceil(60);
            let mut text = format!(
                "Signing in to {provider} with a device code. Open this address on any device:\n    {}\nand enter the code\n    {}",
                event.verification_uri, event.user_code
            );
            if minutes > 0 {
                text.push_str(&format!(
                    "\nThe code expires in {minutes} minute{}. Esc cancels.",
                    if minutes == 1 { "" } else { "s" }
                ));
            }
            Some(text)
        }
        oauth::OAuthEventKind::AuthorizationUrl => {
            let url = event.authorization_url.as_deref()?;
            let mut text = format!(
                "Signing in to {provider} in the browser. If it did not open, visit:\n    {url}"
            );
            if !event.instructions.is_empty() {
                text.push('\n');
                text.push_str(&event.instructions);
            }
            Some(text)
        }
        oauth::OAuthEventKind::Info => (!event.message.is_empty()).then(|| event.message.clone()),
        // Progress updates the status line instead of piling up notices.
        oauth::OAuthEventKind::Progress => None,
    }
}

/// The transcript notice for a flow question, with numbered options.
pub fn prompt_notice(prompt: &PromptRequest) -> String {
    let mut text = prompt.message.clone();
    for (index, option) in prompt.options.iter().enumerate() {
        text.push('\n');
        if option.description.is_empty() {
            text.push_str(&format!("{}. {}", index + 1, option.label));
        } else {
            text.push_str(&format!(
                "{}. {} · {}",
                index + 1,
                option.label,
                option.description
            ));
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oauth::OAuthInteraction;

    fn temp_store(name: &str) -> (std::path::PathBuf, CredentialStore) {
        let root = std::env::temp_dir().join(format!(
            "goshcoder-tui-login-{name}-{}",
            uuid::Uuid::now_v7()
        ));
        std::fs::create_dir_all(&root).expect("temp dir");
        let store = CredentialStore::file(root.join("auth.json"));
        (root, store)
    }

    #[test]
    fn prompts_round_trip_through_the_channel_and_select_by_number() {
        let (sender, events) = mpsc::channel();
        let interaction = TuiInteraction::new(sender);
        let cancellation = oauth::CancellationToken::new();
        let worker = thread::spawn({
            let cancellation = cancellation.clone();
            move || {
                interaction.prompt(oauth::OAuthPrompt {
                    kind: oauth::OAuthPromptKind::Select,
                    message: "Pick a method".to_owned(),
                    placeholder: String::new(),
                    options: vec![
                        oauth::OAuthPromptOption {
                            id: "device".to_owned(),
                            label: "Device code".to_owned(),
                            description: String::new(),
                        },
                        oauth::OAuthPromptOption {
                            id: "browser".to_owned(),
                            label: "Browser".to_owned(),
                            description: "opens a tab".to_owned(),
                        },
                    ],
                    cancellation,
                })
            }
        });
        let LoginEvent::Prompt { prompt, reply } = events.recv().expect("prompt event") else {
            panic!("expected a prompt");
        };
        assert_eq!(
            prompt_notice(&prompt),
            "Pick a method\n1. Device code\n2. Browser · opens a tab"
        );
        reply.send(Some("2".to_owned())).expect("reply");
        assert_eq!(worker.join().expect("worker").expect("answer"), "browser");
    }

    #[test]
    fn a_cancelled_prompt_returns_promptly_instead_of_waiting_for_an_answer() {
        let (sender, events) = mpsc::channel();
        let interaction = TuiInteraction::new(sender);
        let cancellation = oauth::CancellationToken::new();
        let worker = thread::spawn({
            let cancellation = cancellation.clone();
            move || {
                interaction.prompt(oauth::OAuthPrompt {
                    kind: oauth::OAuthPromptKind::ManualCode,
                    message: "Paste the code".to_owned(),
                    placeholder: String::new(),
                    options: Vec::new(),
                    cancellation,
                })
            }
        });
        let _held = events.recv().expect("prompt event");
        cancellation.cancel();
        let started = Instant::now();
        assert!(worker.join().expect("worker").is_err());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn device_code_notice_sets_the_address_and_code_apart() {
        let event = oauth::OAuthEvent {
            kind: oauth::OAuthEventKind::DeviceCode,
            message: String::new(),
            authorization_url: None,
            instructions: String::new(),
            user_code: "QWRT-8KDP".to_owned(),
            verification_uri: "https://auth.example/device".to_owned(),
            interval_seconds: 5,
            expires_in_seconds: 900,
        };
        let notice = event_notice("xai", &event).expect("notice");
        assert!(notice.starts_with("Signing in to xai with a device code."));
        assert!(notice.contains("\n    https://auth.example/device\n"));
        assert!(notice.contains("\n    QWRT-8KDP"));
        assert!(notice.contains("expires in 15 minutes"));
        let progress = oauth::OAuthEvent {
            kind: oauth::OAuthEventKind::Progress,
            message: "polling".to_owned(),
            ..event
        };
        assert!(event_notice("xai", &progress).is_none());
        assert_eq!(clipboard_text(&progress), None);
    }

    #[test]
    fn the_code_or_the_address_goes_on_the_clipboard() {
        let mut event = oauth::OAuthEvent {
            kind: oauth::OAuthEventKind::DeviceCode,
            message: String::new(),
            authorization_url: None,
            instructions: String::new(),
            user_code: "QWRT-8KDP".to_owned(),
            verification_uri: "https://auth.example/device".to_owned(),
            interval_seconds: 5,
            expires_in_seconds: 900,
        };
        assert_eq!(clipboard_text(&event), Some("QWRT-8KDP"));
        event.kind = oauth::OAuthEventKind::AuthorizationUrl;
        event.authorization_url = Some("https://auth.example/authorize?x=1".to_owned());
        assert_eq!(
            clipboard_text(&event),
            Some("https://auth.example/authorize?x=1")
        );
    }

    #[test]
    fn api_key_login_asks_for_each_value_then_stores_the_credential() {
        let (root, store) = temp_store("key");
        let mut flow = LoginFlow::start_api_key(
            "cloudflare-workers-ai",
            &[("CLOUDFLARE_ACCOUNT_ID", "Enter the Cloudflare account ID: ")],
            store,
        );
        assert_eq!(
            flow.api_key_question(),
            Some(("API key for cloudflare-workers-ai".to_owned(), true))
        );
        assert!(flow.answer("sk-secret".to_owned()).is_none());
        assert_eq!(
            flow.api_key_question(),
            Some(("Cloudflare account ID".to_owned(), false))
        );
        flow.answer("acct-1".to_owned())
            .expect("finished")
            .expect("stored");
        let written = std::fs::read_to_string(root.join("auth.json")).expect("auth.json");
        assert!(written.contains("sk-secret"));
        assert!(written.contains("acct-1"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_empty_api_key_is_refused_and_nothing_is_written() {
        let (root, store) = temp_store("empty");
        let mut flow = LoginFlow::start_api_key("openai", &[], store);
        let result = flow.answer(String::new()).expect("finished");
        assert!(result.is_err());
        assert!(!root.join("auth.json").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
