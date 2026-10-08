//! Several Grok CLI accounts, and switching to the next one when a
//! subscription's Grok Build balance runs out.
//!
//! Native adaptation of pi-grok-cli v0.9.3 (J Liew / kenryu42, MIT),
//! written against its `src/provider/accountVault.ts`,
//! `accountRouting.ts`, `accounts.ts` (the terminal half),
//! `sessionAccountSelection.ts`, `requestOwnership.ts` and `rotation.ts`.
//!
//! Where it differs, and why:
//! - Account 1 is the ordinary `grok-cli` login in `auth.json`, so
//!   `/login grok-cli` and `auth login grok-cli` keep working unchanged.
//!   Upstream moves every credential into its vault and leaves a marker in
//!   `auth.json`; GoshCoder's vault (`<agent_dir>/grok-cli/accounts.json`,
//!   0600) holds the labels, the default account and the credentials of the
//!   accounts added after the first.
//! - Accounts are managed with subcommands (`/grok-cli-accounts`,
//!   `goshcoder grok-cli accounts`) rather than a select menu or the browser
//!   dashboard, which is not ported.
//! - Upstream's migrations from its earlier `grok-cli-N` providers have
//!   nothing to migrate from here.

use std::{
    collections::{BTreeSet, HashMap},
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant, SystemTime},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use time::OffsetDateTime;

use crate::{
    agent,
    catalog::{Catalog, Credential, CredentialKind},
    grok_cli, llm, oauth,
    session::SessionNoticeSender,
    stream,
};

pub const ACCOUNT_1_ID: &str = grok_cli::ACCOUNT_ID;
/// Session custom entry naming the account a session uses; the newest valid
/// one on the current branch wins.
pub const SESSION_ACCOUNT_ENTRY: &str = "grok-cli-active-account-v1";
/// The follow-up upstream sends after switching accounts.
pub const ROTATION_CONTINUATION: &str = "Continue the previous request using the newly selected Grok account. Do not repeat completed work.";
/// The proxy's answer when a subscription's Grok Build balance is spent.
const EXHAUSTED_BODY: &str = "Grok Build usage balance exhausted";
const RECENT_EXHAUSTION_COOLDOWN: Duration = Duration::from_secs(5 * 60);
const LOCK_STALE: Duration = Duration::from_secs(30);
const LOCK_RETRY: Duration = Duration::from_millis(25);
const MAX_LABEL_CHARS: usize = 40;
const REFRESH_BUDGET: Duration = Duration::from_secs(15);
const TRANSIENT_REFRESH_FAILURE: Duration = Duration::from_secs(30);

/// A remembered refresh failure for one account at one revision.
struct RefreshFailure {
    message: String,
    /// `None` for a lost login, which only a new login (a new revision)
    /// clears.
    until: Option<Instant>,
}

fn refresh_failures() -> &'static Mutex<HashMap<(String, u64), RefreshFailure>> {
    static FAILURES: OnceLock<Mutex<HashMap<(String, u64), RefreshFailure>>> = OnceLock::new();
    FAILURES.get_or_init(Default::default)
}

// ---------------------------------------------------------------------------
// Vault (upstream accountVault.ts)

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VaultAccount {
    pub id: String,
    pub slot: u32,
    pub label: String,
    /// Never present for Account 1, whose login lives in `auth.json`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<Credential>,
    pub revision: u64,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VaultFile {
    pub version: u32,
    pub next_slot: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_account_id: Option<String>,
    pub accounts: Vec<VaultAccount>,
}

impl Default for VaultFile {
    fn default() -> Self {
        Self {
            version: 1,
            next_slot: 2,
            active_account_id: None,
            accounts: vec![VaultAccount {
                id: ACCOUNT_1_ID.to_owned(),
                slot: 1,
                label: default_label(1),
                credential: None,
                revision: 0,
            }],
        }
    }
}

fn default_label(slot: u32) -> String {
    format!("Account {slot}")
}

fn has_control_characters(value: &str) -> bool {
    value
        .chars()
        .any(|character| (character as u32) <= 31 || (127..=159).contains(&(character as u32)))
}

/// upstream `parseVault`, minus what cannot hold here: Account 1's login is
/// outside the file, so the default account only has to exist.
fn validate(file: &VaultFile, path: &Path) -> Result<(), String> {
    let invalid = |reason: &str| {
        Err(format!(
            "Invalid Grok CLI account vault at {}: {reason}",
            path.display()
        ))
    };
    if file.version != 1 {
        return invalid("unsupported version");
    }
    let Some(first) = file.accounts.first() else {
        return invalid("accounts are missing");
    };
    if first.id != ACCOUNT_1_ID || first.slot != 1 || first.credential.is_some() {
        return invalid("Account 1 must be first and permanent");
    }
    let mut ids = BTreeSet::new();
    let mut slots = BTreeSet::new();
    let mut labels = BTreeSet::new();
    for (index, account) in file.accounts.iter().enumerate() {
        let label = account.label.trim();
        if account.id.is_empty()
            || account.slot < 1
            || label.is_empty()
            || label.chars().count() > MAX_LABEL_CHARS
            || has_control_characters(label)
        {
            return invalid(&format!("invalid account at index {index}"));
        }
        if account
            .credential
            .as_ref()
            .is_some_and(|credential| credential.kind() != &CredentialKind::OAuth)
        {
            return invalid(&format!("invalid credential at index {index}"));
        }
        if !ids.insert(account.id.clone())
            || !slots.insert(account.slot)
            || !labels.insert(label.to_lowercase())
        {
            return invalid(&format!("duplicate account at index {index}"));
        }
    }
    if file.next_slot <= slots.iter().copied().max().unwrap_or(0) {
        return invalid("next slot must be greater than every account slot");
    }
    if let Some(active) = file.active_account_id.as_deref()
        && !ids.contains(active)
    {
        return invalid("the default account does not exist");
    }
    Ok(())
}

/// The account file and its lock. Without an agent directory (a catalog
/// built from an injected environment that names none) there is no vault:
/// only Account 1 exists, and nothing is read from the developer's home.
#[derive(Clone)]
pub struct Vault {
    path: Option<PathBuf>,
}

/// A lock file held for one read-modify-write; a lock older than
/// [`LOCK_STALE`] belongs to a process that died and is taken over.
struct VaultLock {
    path: PathBuf,
}

impl Drop for VaultLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

impl Vault {
    pub fn for_agent_dir(agent_dir: &Path) -> Self {
        Self {
            path: Some(grok_cli::state_dir(agent_dir).join("accounts.json")),
        }
    }

    pub fn disabled() -> Self {
        Self { path: None }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn required_path(&self) -> Result<&Path, String> {
        self.path()
            .ok_or_else(|| "Grok CLI accounts need an agent directory.".to_owned())
    }

    /// The quota cache beside the vault, where `/grok-cli-usage` writes.
    fn quota_cache(&self) -> Option<PathBuf> {
        Some(self.path()?.with_file_name("quota-cache.json"))
    }

    pub fn load(&self) -> Result<VaultFile, String> {
        let Some(path) = self.path() else {
            return Ok(VaultFile::default());
        };
        let raw = match fs::read(path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(VaultFile::default());
            }
            Err(error) => return Err(format!("{}: {error}", path.display())),
        };
        let file = serde_json::from_slice::<VaultFile>(&raw).map_err(|error| {
            format!(
                "Invalid Grok CLI account vault at {}: {error}",
                path.display()
            )
        })?;
        validate(&file, path)?;
        Ok(file)
    }

    fn lock(&self) -> Result<VaultLock, String> {
        let path = self.required_path()?;
        let directory = path.parent().ok_or("the account vault has no directory")?;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        let lock_path = path.with_extension("json.lock");
        let started = Instant::now();
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(mut file) => {
                    let _ = write!(file, "{}", std::process::id());
                    return Ok(VaultLock { path: lock_path });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    let abandoned = fs::metadata(&lock_path)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| SystemTime::now().duration_since(modified).ok())
                        .is_some_and(|age| age > LOCK_STALE);
                    if abandoned {
                        let _ = fs::remove_file(&lock_path);
                        continue;
                    }
                    if started.elapsed() > LOCK_STALE {
                        return Err(format!(
                            "Timed out waiting for file lock: {}",
                            lock_path.display()
                        ));
                    }
                    thread::sleep(LOCK_RETRY);
                }
                Err(error) => return Err(format!("{}: {error}", lock_path.display())),
            }
        }
    }

    /// Applies `change` under the lock and writes the result back (0600,
    /// atomically) when it validates.
    pub fn mutate<T>(
        &self,
        change: impl FnOnce(&mut VaultFile) -> Result<T, String>,
    ) -> Result<T, String> {
        let _lock = self.lock()?;
        let path = self.required_path()?;
        let mut file = self.load()?;
        let result = change(&mut file)?;
        validate(&file, path)?;
        let mut contents = serde_json::to_vec_pretty(&file).map_err(|error| error.to_string())?;
        contents.push(b'\n');
        crate::config::write_atomic(path, &contents, 0o600)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Accounts and routing (upstream accountRouting.ts and accounts.ts)

/// One row of the account list.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AccountStatus {
    pub id: String,
    /// 1-based position, the number commands accept.
    pub number: usize,
    pub label: String,
    pub authenticated: bool,
    pub active: bool,
    pub environment: bool,
}

impl AccountStatus {
    pub fn status(&self) -> &'static str {
        match (self.active, self.environment, self.authenticated) {
            (true, true, _) => "Active (environment)",
            (true, false, _) => "Active",
            (false, _, true) => "Authenticated",
            (false, _, false) => "Login required",
        }
    }
}

/// The accounts behind one catalog.
#[derive(Clone)]
pub struct Accounts {
    catalog: Catalog,
    vault: Vault,
}

impl Accounts {
    pub fn new(catalog: &Catalog) -> Self {
        Self {
            catalog: catalog.clone(),
            vault: catalog
                .dynamic_paths()
                .agent_dir
                .as_deref()
                .map_or_else(Vault::disabled, Vault::for_agent_dir),
        }
    }

    pub fn vault(&self) -> &Vault {
        &self.vault
    }

    fn environment_token(&self) -> Option<String> {
        self.catalog.environment_value(grok_cli::TOKEN_ENV)
    }

    /// Account 1 is logged in when `auth.json` holds a Grok CLI login.
    fn account_one_logged_in(&self) -> bool {
        self.catalog.credentials().is_some_and(|store| {
            matches!(
                store.read(grok_cli::PROVIDER_ID),
                Ok(Some(credential)) if credential.kind() == &CredentialKind::OAuth
            )
        })
    }

    fn logged_in(&self, file: &VaultFile, id: &str) -> bool {
        if id == ACCOUNT_1_ID {
            return self.account_one_logged_in();
        }
        file.accounts
            .iter()
            .any(|account| account.id == id && account.credential.is_some())
    }

    /// upstream `accountId`: the session's choice when it is still logged
    /// in, then the default account, then the first logged-in one.
    pub fn selected(&self, file: &VaultFile, session_choice: Option<&str>) -> Option<String> {
        session_choice
            .filter(|id| self.logged_in(file, id))
            .map(str::to_owned)
            .or_else(|| {
                file.active_account_id
                    .as_deref()
                    .filter(|id| self.logged_in(file, id))
                    .map(str::to_owned)
            })
            .or_else(|| {
                file.accounts
                    .iter()
                    .find(|account| self.logged_in(file, &account.id))
                    .map(|account| account.id.clone())
            })
    }

    /// The bearer token for an account added after the first, refreshed
    /// under the vault lock when it is about to expire. A revision that
    /// moved while waiting means another process refreshed it already.
    pub fn vault_token(&self, account_id: &str) -> Result<String, String> {
        let file = self.vault.load()?;
        let account = file
            .accounts
            .iter()
            .find(|account| account.id == account_id)
            .ok_or_else(|| format!("Unknown Grok CLI account: {account_id}"))?;
        let credential = account
            .credential
            .as_ref()
            .ok_or_else(|| format!("Log in to “{}” before using it.", account.label))?;
        let client = self.catalog.oauth_client();
        if !client.credential_needs_refresh(credential) {
            return Ok(credential.access().to_owned());
        }
        // A refresh that just failed for this exact credential is not tried
        // again on every lookup (the model picker resolves often): a lost
        // login stays failed until the account changes, a transient failure
        // for half a minute, as the catalog's own failure cache does.
        let failure_key = (account_id.to_owned(), account.revision);
        if let Some(failure) = lock(refresh_failures()).get(&failure_key)
            && failure.until.is_none_or(|until| Instant::now() < until)
        {
            return Err(failure.message.clone());
        }
        let outcome = self.vault.mutate(|file| {
            let account = file
                .accounts
                .iter_mut()
                .find(|account| account.id == account_id)
                .ok_or_else(|| format!("Unknown Grok CLI account: {account_id}"))?;
            let current = account
                .credential
                .clone()
                .ok_or_else(|| format!("Log in to “{}” before using it.", account.label))?;
            if !client.credential_needs_refresh(&current) {
                return Ok(current.access().to_owned());
            }
            let environment = oauth::CatalogEnvironment::new(Arc::new({
                let catalog = self.catalog.clone();
                move |name: &str| catalog.environment_value(name)
            }));
            let refreshed = client
                .refresh(
                    oauth::OAuthProviderId::GrokCli,
                    &current,
                    &environment,
                    &oauth::CancellationToken::with_timeout(REFRESH_BUDGET),
                )
                .map_err(|error| RefreshFailure {
                    message: format!("“{}”: {error}", account.label),
                    until: (!error.is_unauthorized())
                        .then(|| Instant::now() + TRANSIENT_REFRESH_FAILURE),
                })
                .map_err(|failure| {
                    let message = failure.message.clone();
                    lock(refresh_failures()).insert(failure_key.clone(), failure);
                    message
                })?;
            let token = refreshed.access().to_owned();
            account.credential = Some(refreshed);
            account.revision += 1;
            Ok(token)
        });
        if outcome.is_ok() {
            lock(refresh_failures()).remove(&failure_key);
        }
        outcome
    }

    /// A token from the first logged-in vault account, for when `auth.json`
    /// holds no Grok CLI login: upstream keeps the provider usable as long
    /// as any account is signed in.
    pub fn fallback_token(&self) -> Option<String> {
        let file = self.vault.load().ok()?;
        let account = file
            .active_account_id
            .as_deref()
            .and_then(|id| {
                file.accounts
                    .iter()
                    .find(|account| account.id == id && account.credential.is_some())
            })
            .or_else(|| {
                file.accounts
                    .iter()
                    .find(|account| account.credential.is_some())
            })?;
        self.vault_token(&account.id).ok()
    }

    /// The token a request for `request_session` should carry, when it is
    /// not the one the catalog resolved: the environment token and Account 1
    /// go through the catalog, every other account through the vault.
    pub fn request_token(&self, request_session: &str) -> Result<Option<String>, String> {
        if self.environment_token().is_some() {
            remember_request_account(request_session, ACCOUNT_1_ID);
            return Ok(None);
        }
        let file = self.vault.load()?;
        let choice = session_choice(request_session);
        let Some(selected) = self.selected(&file, choice.as_deref()) else {
            return Ok(None);
        };
        remember_request_account(request_session, &selected);
        if selected == ACCOUNT_1_ID {
            return Ok(None);
        }
        self.vault_token(&selected).map(Some)
    }

    /// The account and token `/grok-cli-usage` reports on.
    pub fn usage_route(&self, request_session: &str) -> Result<(String, String), String> {
        if let Some(token) = self.environment_token() {
            return Ok((ACCOUNT_1_ID.to_owned(), token));
        }
        let file = self.vault.load()?;
        let choice = session_choice(request_session);
        let selected = self
            .selected(&file, choice.as_deref())
            .ok_or("Grok CLI login is required. Run /login grok-cli.")?;
        if selected == ACCOUNT_1_ID {
            let token = self
                .catalog
                .resolve_auth(grok_cli::PROVIDER_ID)
                .map_err(|error| error.to_string())?
                .and_then(|auth| auth.api_key().map(str::to_owned))
                .ok_or("Grok CLI login is required. Run /login grok-cli.")?;
            return Ok((selected, token));
        }
        Ok((selected.clone(), self.vault_token(&selected)?))
    }

    pub fn list(&self, request_session: &str) -> Result<Vec<AccountStatus>, String> {
        let file = self.vault.load()?;
        let environment = self.environment_token().is_some();
        let choice = session_choice(request_session);
        let selected = self.selected(&file, choice.as_deref());
        Ok(file
            .accounts
            .iter()
            .enumerate()
            .map(|(index, account)| {
                let number = index + 1;
                let account_environment = environment && account.id == ACCOUNT_1_ID;
                AccountStatus {
                    id: account.id.clone(),
                    number,
                    // A default label follows the account's place in the list.
                    label: if account.label == default_label(account.slot) {
                        default_label(number as u32)
                    } else {
                        account.label.clone()
                    },
                    authenticated: account_environment || self.logged_in(&file, &account.id),
                    active: account_environment
                        || (!environment && selected.as_deref() == Some(account.id.as_str())),
                    environment: account_environment,
                }
            })
            .collect())
    }

    /// Finds an account by list number, id, or label (ignoring case).
    pub fn find(&self, reference: &str) -> Result<AccountStatus, String> {
        let reference = reference.trim();
        self.list("")?
            .into_iter()
            .find(|account| {
                reference.parse::<usize>().ok() == Some(account.number)
                    || account.id == reference
                    || account.label.to_lowercase() == reference.to_lowercase()
            })
            .ok_or_else(|| format!("Unknown Grok CLI account: {reference}"))
    }

    fn normalize_label(
        file: &VaultFile,
        id: &str,
        slot: u32,
        value: &str,
    ) -> Result<String, String> {
        let label = match value.trim() {
            "" => default_label(slot),
            label => label.to_owned(),
        };
        if label.chars().count() > MAX_LABEL_CHARS {
            return Err("Account labels must be 40 characters or fewer.".to_owned());
        }
        if has_control_characters(&label) {
            return Err("Account labels cannot contain control characters.".to_owned());
        }
        if file
            .accounts
            .iter()
            .any(|account| account.id != id && account.label.to_lowercase() == label.to_lowercase())
        {
            return Err(format!("An account named “{label}” already exists."));
        }
        Ok(label)
    }

    /// Adds an account that still needs a login; returns its id.
    pub fn add(&self, label: &str) -> Result<String, String> {
        self.vault.mutate(|file| {
            let slot = file.next_slot;
            let id = uuid::Uuid::now_v7().to_string();
            let label = Self::normalize_label(file, &id, slot, label)?;
            file.next_slot += 1;
            file.accounts.push(VaultAccount {
                id: id.clone(),
                slot,
                label,
                credential: None,
                revision: 0,
            });
            Ok(id)
        })
    }

    pub fn rename(&self, id: &str, label: &str) -> Result<String, String> {
        self.vault.mutate(|file| {
            let slot = file
                .accounts
                .iter()
                .find(|account| account.id == id)
                .map(|account| account.slot)
                .ok_or_else(|| format!("Unknown Grok CLI account: {id}"))?;
            let label = Self::normalize_label(file, id, slot, label)?;
            if let Some(account) = file.accounts.iter_mut().find(|account| account.id == id) {
                account.label = label.clone();
            }
            Ok(label)
        })
    }

    /// Stores a fresh login for `id`: Account 1's in `auth.json`, any other
    /// in the vault, refusing when the account changed meanwhile.
    pub fn store_login(
        &self,
        id: &str,
        revision: u64,
        credential: Credential,
    ) -> Result<(), String> {
        if id == ACCOUNT_1_ID {
            let store = self
                .catalog
                .credentials()
                .ok_or("no credential store is configured")?;
            store
                .put(grok_cli::PROVIDER_ID, credential)
                .map_err(|error| error.to_string())?;
            self.catalog
                .clear_oauth_refresh_failure(grok_cli::PROVIDER_ID);
            return self.vault.mutate(|file| {
                if file.active_account_id.is_none() {
                    file.active_account_id = Some(ACCOUNT_1_ID.to_owned());
                }
                Ok(())
            });
        }
        // Upstream's Account 1 became the default when it logged in; here
        // that login lives in auth.json, so it is honoured explicitly.
        let account_one = self.account_one_logged_in();
        self.vault.mutate(|file| {
            if file.active_account_id.is_none() && account_one {
                file.active_account_id = Some(ACCOUNT_1_ID.to_owned());
            }
            let account = file
                .accounts
                .iter_mut()
                .find(|account| account.id == id)
                .ok_or("The account was removed while login was in progress.")?;
            if account.revision != revision {
                return Err(
                    "The account changed while login was in progress. Try again.".to_owned(),
                );
            }
            account.credential = Some(credential);
            account.revision += 1;
            if file.active_account_id.is_none() {
                file.active_account_id = Some(id.to_owned());
            }
            Ok(())
        })?;
        forget_exhaustion(id);
        Ok(())
    }

    pub fn revision(&self, id: &str) -> Result<u64, String> {
        self.vault
            .load()?
            .accounts
            .iter()
            .find(|account| account.id == id)
            .map(|account| account.revision)
            .ok_or_else(|| format!("Unknown Grok CLI account: {id}"))
    }

    /// Makes `id` the default for sessions that have not chosen one.
    pub fn activate(&self, id: &str) -> Result<String, String> {
        if self.environment_token().is_some() {
            return Err(
                "Saved accounts cannot be selected while the environment token is active."
                    .to_owned(),
            );
        }
        let logged_in = self.logged_in(&self.vault.load()?, id);
        self.vault.mutate(|file| {
            let account = file
                .accounts
                .iter()
                .find(|account| account.id == id)
                .ok_or_else(|| format!("Unknown Grok CLI account: {id}"))?;
            if !logged_in {
                return Err(format!(
                    "Log in to “{}” before making it active.",
                    account.label
                ));
            }
            let label = account.label.clone();
            file.active_account_id = Some(id.to_owned());
            Ok(label)
        })
    }

    fn replace_default(&self, file: &mut VaultFile, removed: &str) {
        if file.active_account_id.as_deref() != Some(removed) {
            return;
        }
        let fallback = file
            .accounts
            .iter()
            .find(|account| account.id != removed && self.logged_in(file, &account.id))
            .map(|account| account.id.clone());
        file.active_account_id = fallback;
    }

    /// Signs an account out; Account 1's login leaves `auth.json`.
    pub fn logout(&self, id: &str) -> Result<Option<String>, String> {
        if self.environment_token().is_some() {
            return Err(
                "Unset GROK_CLI_OAUTH_TOKEN and restart GoshCoder to remove the environment token."
                    .to_owned(),
            );
        }
        if id == ACCOUNT_1_ID
            && let Some(store) = self.catalog.credentials()
        {
            store
                .delete(grok_cli::PROVIDER_ID)
                .map_err(|error| error.to_string())?;
        }
        let warning = self.vault.mutate(|file| {
            let account = file
                .accounts
                .iter_mut()
                .find(|account| account.id == id)
                .ok_or_else(|| format!("Unknown Grok CLI account: {id}"))?;
            account.credential = None;
            account.revision += 1;
            self.replace_default(file, id);
            Ok(file.active_account_id.is_none().then(|| {
                "No logged-in account remains; run /login grok-cli to sign in again.".to_owned()
            }))
        })?;
        remove_cached_quota(&self.vault, id);
        Ok(warning)
    }

    pub fn remove(&self, id: &str) -> Result<Option<String>, String> {
        if id == ACCOUNT_1_ID {
            return Err("The permanent Account 1 cannot be removed.".to_owned());
        }
        let warning = self.vault.mutate(|file| {
            if !file.accounts.iter().any(|account| account.id == id) {
                return Err(format!("Unknown Grok CLI account: {id}"));
            }
            self.replace_default(file, id);
            file.accounts.retain(|account| account.id != id);
            Ok(file.active_account_id.is_none().then(|| {
                "No logged-in account remains; run /login grok-cli to sign in again.".to_owned()
            }))
        })?;
        remove_cached_quota(&self.vault, id);
        Ok(warning)
    }
}

/// A login invalidates whatever usage was cached for the account before.
fn remove_cached_quota(vault: &Vault, id: &str) {
    let Some(path) = vault.quota_cache() else {
        return;
    };
    let mut accounts = grok_cli::load_quota_cache(&path);
    if accounts.remove(id).is_some() {
        let document = json!({ "version": 1, "accounts": accounts });
        if let Ok(mut contents) = serde_json::to_vec_pretty(&document) {
            contents.push(b'\n');
            let _ = crate::config::write_atomic(&path, &contents, 0o600);
        }
    }
}

// ---------------------------------------------------------------------------
// Session selection and request ownership
// (upstream sessionAccountSelection.ts and requestOwnership.ts)

#[derive(Default)]
struct SessionAccounts {
    /// Choices that could not be recorded (no session file).
    chosen: HashMap<String, String>,
    /// The account each session's latest request went to.
    last_request: HashMap<String, String>,
}

fn session_accounts() -> &'static Mutex<SessionAccounts> {
    static STATE: OnceLock<Mutex<SessionAccounts>> = OnceLock::new();
    STATE.get_or_init(Default::default)
}

fn stored_account_id(data: &Value) -> Option<String> {
    data.as_object()?
        .get("accountId")?
        .as_str()
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
}

/// The account a session chose: an unrecorded choice, else the newest
/// valid `grok-cli-active-account-v1` entry on its branch.
pub fn session_choice(request_session: &str) -> Option<String> {
    if request_session.is_empty() {
        return None;
    }
    if let Some(chosen) = lock(session_accounts()).chosen.get(request_session) {
        return Some(chosen.clone());
    }
    grok_cli::session_store(request_session).and_then(|store| {
        store
            .custom_values(SESSION_ACCOUNT_ENTRY)
            .iter()
            .rev()
            .find_map(stored_account_id)
    })
}

/// Records the session's choice as a custom entry, or in memory when the
/// session is not recorded.
pub fn choose_for_session(request_session: &str, account_id: &str) -> Result<(), String> {
    if request_session.is_empty() {
        return Ok(());
    }
    if session_choice(request_session).as_deref() == Some(account_id) {
        return Ok(());
    }
    let recorded = match grok_cli::session_store(request_session) {
        Some(store) => store.append(SESSION_ACCOUNT_ENTRY, json!({ "accountId": account_id }))?,
        None => false,
    };
    let mut state = lock(session_accounts());
    if recorded {
        state.chosen.remove(request_session);
    } else {
        state
            .chosen
            .insert(request_session.to_owned(), account_id.to_owned());
    }
    Ok(())
}

fn remember_request_account(request_session: &str, account_id: &str) {
    if request_session.is_empty() {
        return;
    }
    lock(session_accounts())
        .last_request
        .insert(request_session.to_owned(), account_id.to_owned());
}

fn request_account(request_session: &str) -> Option<String> {
    lock(session_accounts())
        .last_request
        .get(request_session)
        .cloned()
}

// ---------------------------------------------------------------------------
// Exhaustion rotation (upstream rotation.ts)

fn recent_exhaustion() -> &'static Mutex<HashMap<String, Instant>> {
    static RECENT: OnceLock<Mutex<HashMap<String, Instant>>> = OnceLock::new();
    RECENT.get_or_init(Default::default)
}

fn recently_exhausted(account_id: &str, now: Instant) -> bool {
    let mut recent = lock(recent_exhaustion());
    match recent.get(account_id) {
        Some(at) if now.duration_since(*at) < RECENT_EXHAUSTION_COOLDOWN => true,
        Some(_) => {
            recent.remove(account_id);
            false
        }
        None => false,
    }
}

/// A fresh login or a removed account starts over.
fn forget_exhaustion(account_id: &str) {
    lock(recent_exhaustion()).remove(account_id);
}

/// Whether an assistant message is the proxy refusing a spent balance.
pub fn is_exhaustion(message: &llm::AssistantMessage) -> bool {
    message.provider == grok_cli::PROVIDER_ID
        && message.stop_reason == stream::STOP_ERROR
        && message.error_message.contains("status 402")
        && message.error_message.contains(EXHAUSTED_BODY)
}

/// upstream `orderAccountsByQuota`: among accounts with fresh cached weekly
/// figures, the most remaining credit goes first; the others keep their
/// circular place.
fn order_by_quota(
    ids: Vec<String>,
    quotas: &std::collections::BTreeMap<String, grok_cli::CachedQuota>,
    now: OffsetDateTime,
) -> Vec<String> {
    let score = |id: &String| {
        let entry = quotas.get(id)?;
        let weekly = entry.usage.weekly.as_ref()?;
        grok_cli::is_cached_quota_fresh(entry, now)
            .then(|| (1.0 - weekly.credit_usage_percent / 100.0).clamp(0.0, 1.0))
    };
    let scored = ids
        .iter()
        .enumerate()
        .filter_map(|(index, id)| Some((index, score(id)?)))
        .collect::<Vec<_>>();
    let mut ranked = scored.clone();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    let mut result = ids.clone();
    for ((slot, _), (source, _)) in scored.iter().zip(ranked.iter()) {
        result[*slot] = ids[*source].clone();
    }
    result
}

#[derive(Default)]
struct Chain {
    exhausted: BTreeSet<String>,
    awaiting_continuation: bool,
}

/// Watches a session's turns: when a Grok CLI request fails because the
/// account's balance is spent, the session moves to the next logged-in
/// account (by fresh quota, then list order) and the turn continues with
/// upstream's continuation message. One chain never returns to an account
/// it exhausted, and an account that ran dry in the last five minutes is
/// skipped by every session in the process.
pub fn rotation_subscription(
    agent: &agent::Agent,
    accounts: Accounts,
    request_session: String,
    notices: SessionNoticeSender,
) -> agent::Subscription {
    let queue = agent.weak_follow_up_queue();
    let chain = Mutex::new(Chain::default());
    let current_model = Mutex::new(None::<(String, String)>);
    agent.subscribe(move |event| {
        if event.kind == agent::EventKind::ModelChange {
            *lock(&current_model) = Some((event.provider.clone(), event.model_id.clone()));
            let mut chain = lock(&chain);
            *chain = Chain::default();
            return;
        }
        if event.kind != agent::EventKind::TurnEnd {
            return;
        }
        let Some(llm::Message::Assistant(message)) = event.message.as_ref() else {
            return;
        };
        let mut chain = lock(&chain);
        if !is_exhaustion(message) || accounts.environment_token().is_some() {
            // The continuation settled; the next exhaustion starts afresh.
            if chain.awaiting_continuation {
                *chain = Chain::default();
            }
            return;
        }
        if let Some((provider, model)) = lock(&current_model).as_ref()
            && (provider != grok_cli::PROVIDER_ID || *model != message.model)
        {
            *chain = Chain::default();
            return;
        }
        let Ok(file) = accounts.vault.load() else {
            return;
        };
        let failed = request_account(&request_session)
            .or_else(|| accounts.selected(&file, session_choice(&request_session).as_deref()));
        let Some(failed) = failed else {
            return;
        };
        let now = Instant::now();
        chain.exhausted.insert(failed.clone());
        lock(recent_exhaustion()).insert(failed.clone(), now);
        chain.awaiting_continuation = false;

        let logged_in = file
            .accounts
            .iter()
            .filter(|account| accounts.logged_in(&file, &account.id))
            .map(|account| account.id.clone())
            .collect::<Vec<_>>();
        if logged_in.len() < 2 {
            *chain = Chain::default();
            return;
        }
        let ids = file
            .accounts
            .iter()
            .map(|account| account.id.clone())
            .collect::<Vec<_>>();
        let start = ids
            .iter()
            .position(|id| *id == failed)
            .map_or(0, |index| index + 1);
        let circular = ids[start..]
            .iter()
            .chain(ids[..start].iter())
            .filter(|id| {
                **id != failed
                    && !chain.exhausted.contains(*id)
                    && !recently_exhausted(id, now)
                    && logged_in.contains(id)
            })
            .cloned()
            .collect::<Vec<_>>();
        let quotas = accounts
            .vault
            .quota_cache()
            .map(|path| grok_cli::load_quota_cache(&path))
            .unwrap_or_default();
        let label = |id: &str| {
            accounts
                .list("")
                .ok()
                .and_then(|list| list.into_iter().find(|account| account.id == id))
                .map_or_else(|| id.to_owned(), |account| account.label)
        };
        match order_by_quota(circular, &quotas, OffsetDateTime::now_utc())
            .into_iter()
            .next()
        {
            Some(next) => {
                if let Err(error) = choose_for_session(&request_session, &next) {
                    notices.push("Grok CLI", format!("could not switch accounts: {error}"));
                    *chain = Chain::default();
                    return;
                }
                notices.push(
                    "Grok CLI",
                    format!(
                        "“{}” exhausted; switched to “{}” and continuing.",
                        label(&failed),
                        label(&next)
                    ),
                );
                chain.awaiting_continuation = true;
                queue.follow_up(llm::Message::User(llm::UserMessage::text(
                    ROTATION_CONTINUATION,
                    OffsetDateTime::now_utc().unix_timestamp() * 1_000,
                )));
            }
            None => {
                let all_spent = logged_in
                    .iter()
                    .all(|id| chain.exhausted.contains(id) || recently_exhausted(id, now));
                notices.push(
                    "Grok CLI",
                    if all_spent {
                        "all logged-in accounts are exhausted."
                    } else {
                        "no other logged-in account is available for automatic rotation."
                    },
                );
                *chain = Chain::default();
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Commands

pub const ACCOUNTS_USAGE: &str = "Usage: /grok-cli-accounts [list | use <n> | add [label] | login <n> | logout <n> | rename <n> <label> | remove <n>]";

/// What an accounts command needs from the frontend.
pub enum AccountsCommand {
    /// Answered in place.
    Done(String),
    /// Needs the terminal for an OAuth login: run
    /// `goshcoder grok-cli accounts <arguments>` and report its outcome.
    Terminal(Vec<String>),
}

fn format_list(accounts: &Accounts, request_session: &str, usage: &str) -> Result<String, String> {
    let quotas = accounts
        .vault
        .quota_cache()
        .map(|path| grok_cli::load_quota_cache(&path))
        .unwrap_or_default();
    let now = OffsetDateTime::now_utc();
    let mut lines = vec!["Grok CLI accounts:".to_owned()];
    for account in accounts.list(request_session)? {
        let mut line = format!(
            "  {}. {} — {}",
            account.number,
            account.label,
            account.status()
        );
        if let Some(quota) = quotas.get(&account.id) {
            line.push_str(&format!(" · {}", grok_cli::format_cached_quota(quota, now)));
        }
        lines.push(line);
    }
    lines.push(usage.to_owned());
    Ok(lines.join("\n"))
}

/// `/grok-cli-accounts` in chat, where `use` picks the account for this
/// session (and the default for new ones).
pub fn chat_command(
    accounts: &Accounts,
    request_session: &str,
    arguments: &str,
) -> Result<AccountsCommand, String> {
    let words = arguments.split_whitespace().collect::<Vec<_>>();
    match words.as_slice() {
        [] | ["list"] => {
            format_list(accounts, request_session, ACCOUNTS_USAGE).map(AccountsCommand::Done)
        }
        ["use", reference] => {
            let account = accounts.find(reference)?;
            let label = accounts.activate(&account.id)?;
            choose_for_session(request_session, &account.id)?;
            Ok(AccountsCommand::Done(format!(
                "Grok CLI: using “{label}” for this session and new ones."
            )))
        }
        ["add", ..] | ["login", _] => Ok(AccountsCommand::Terminal(
            std::iter::once("accounts".to_owned())
                .chain(words.iter().map(|word| (*word).to_owned()))
                .collect(),
        )),
        _ => shared_command(accounts, &words).map(AccountsCommand::Done),
    }
}

/// The subcommands chat and the CLI answer the same way.
fn shared_command(accounts: &Accounts, words: &[&str]) -> Result<String, String> {
    match words {
        ["rename", reference, label @ ..] if !label.is_empty() => {
            let account = accounts.find(reference)?;
            let label = accounts.rename(&account.id, &label.join(" "))?;
            Ok(format!("Renamed account {} to “{label}”.", account.number))
        }
        ["logout", reference] => {
            let account = accounts.find(reference)?;
            let warning = accounts.logout(&account.id)?;
            Ok(
                std::iter::once(format!("Logged out of “{}”.", account.label))
                    .chain(warning)
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        }
        ["remove", reference] => {
            let account = accounts.find(reference)?;
            let warning = accounts.remove(&account.id)?;
            forget_exhaustion(&account.id);
            Ok(std::iter::once(format!("Removed “{}”.", account.label))
                .chain(warning)
                .collect::<Vec<_>>()
                .join("\n"))
        }
        _ => Err(ACCOUNTS_USAGE.to_owned()),
    }
}

/// `goshcoder grok-cli accounts …`: the terminal half, which can run the
/// OAuth login. `use` here sets the default for new sessions.
pub fn cli_command(
    accounts: &Accounts,
    words: &[&str],
    login: &dyn Fn(&oauth::OAuthClient) -> Result<Credential, String>,
) -> Result<String, String> {
    match words {
        [] | ["list"] => format_list(
            accounts,
            "",
            "Run goshcoder grok-cli help for the account subcommands.",
        ),
        ["use", reference] => {
            let account = accounts.find(reference)?;
            let label = accounts.activate(&account.id)?;
            Ok(format!("“{label}” is now the default Grok CLI account."))
        }
        ["add", label @ ..] => {
            let id = accounts.add(&label.join(" "))?;
            let revision = accounts.revision(&id)?;
            let outcome = login(accounts.catalog.oauth_client())
                .and_then(|credential| accounts.store_login(&id, revision, credential));
            if let Err(error) = outcome {
                // A login that did not finish leaves no half-made account.
                let _ = accounts.remove(&id);
                return Err(error);
            }
            let account = accounts.find(&id)?;
            Ok(format!(
                "Added and logged in to “{}” (account {}).",
                account.label, account.number
            ))
        }
        ["login", reference] => {
            let account = accounts.find(reference)?;
            let revision = accounts.revision(&account.id)?;
            let credential = login(accounts.catalog.oauth_client())?;
            accounts.store_login(&account.id, revision, credential)?;
            Ok(format!("Logged in to “{}”.", account.label))
        }
        _ => shared_command(accounts, words),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, VecDeque},
        sync::atomic::{AtomicU64, Ordering},
    };

    use super::*;
    use crate::{
        catalog::CredentialStore,
        session::{SessionOptions, SessionRuntime},
        turns,
    };

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "goshcoder-grok-accounts-{label}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("temp dir");
        directory
    }

    fn login(access: &str, expires_at_ms: i64) -> Credential {
        let mut credential = Credential::oauth(access, format!("{access}-refresh"), expires_at_ms);
        credential
            .set_extra(
                "tokenEndpoint",
                Value::String("https://auth.x.ai/oauth2/token".to_owned()),
            )
            .expect("extra");
        credential
    }

    fn catalog(
        agent_dir: &Path,
        store: &Arc<CredentialStore>,
        environment: &[(&str, &str)],
    ) -> Catalog {
        let environment = environment
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect::<BTreeMap<_, _>>();
        Catalog::with_environment(
            Some(Arc::clone(store)),
            Arc::new(move |name| environment.get(name).cloned()),
        )
        .expect("catalog")
        .with_dynamic_paths(crate::catalog::DynamicPaths::for_agent_dir(agent_dir))
    }

    /// Adds a logged-in account the way `accounts add` does.
    fn add_logged_in(accounts: &Accounts, label: &str, access: &str) -> String {
        let id = accounts.add(label).expect("add");
        let revision = accounts.revision(&id).expect("revision");
        accounts
            .store_login(&id, revision, login(access, i64::MAX))
            .expect("store login");
        id
    }

    #[test]
    fn the_vault_keeps_account_one_first_and_validates_labels() {
        let agent_dir = temp_dir("vault");
        let store = Arc::new(CredentialStore::in_memory());
        let accounts = Accounts::new(&catalog(&agent_dir, &store, &[]));
        let path = agent_dir.join("grok-cli").join("accounts.json");
        assert_eq!(accounts.vault().path(), Some(path.as_path()));
        assert_eq!(accounts.vault().load().expect("default").accounts.len(), 1);

        let second = accounts.add("").expect("add default label");
        assert_eq!(accounts.find("2").expect("find").label, "Account 2");
        assert_eq!(
            accounts.add("account 2").expect_err("duplicate label"),
            "An account named “account 2” already exists."
        );
        assert_eq!(
            accounts.add(&"x".repeat(41)).expect_err("long label"),
            "Account labels must be 40 characters or fewer."
        );
        assert_eq!(
            accounts.add("tab\there").expect_err("control character"),
            "Account labels cannot contain control characters."
        );
        assert_eq!(accounts.rename(&second, "Work").expect("rename"), "Work");
        assert_eq!(accounts.find("work").expect("by label").id, second);
        assert_eq!(
            accounts.remove(ACCOUNT_1_ID).expect_err("permanent"),
            "The permanent Account 1 cannot be removed."
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).expect("metadata").permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        let document: Value =
            serde_json::from_slice(&fs::read(&path).expect("read")).expect("json");
        assert_eq!(document["version"], 1);
        assert_eq!(document["nextSlot"], 3);
        assert_eq!(document["accounts"][0]["id"], ACCOUNT_1_ID);
        assert!(document["accounts"][0].get("credential").is_none());

        // A vault that breaks the invariants is refused, not repaired.
        fs::write(
            &path,
            json!({"version": 1, "nextSlot": 2, "accounts": [{"id": "other", "slot": 1, "label": "A", "revision": 0}]})
                .to_string(),
        )
        .expect("write");
        assert!(
            accounts
                .vault()
                .load()
                .err()
                .expect("invalid")
                .contains("Account 1 must be first and permanent")
        );

        // Without an agent directory there is no vault to read or write.
        let bare = Catalog::with_environment(Some(store), Arc::new(|_| None)).expect("catalog");
        let none = Accounts::new(&bare);
        assert_eq!(none.vault().path(), None);
        assert_eq!(
            none.add("x").expect_err("disabled"),
            "Grok CLI accounts need an agent directory."
        );
        let _ = fs::remove_dir_all(agent_dir);
    }

    #[test]
    fn a_session_choice_routes_requests_and_survives_logouts_by_falling_back() {
        let agent_dir = temp_dir("routing");
        let store = Arc::new(CredentialStore::in_memory());
        store
            .put(grok_cli::PROVIDER_ID, login("account-one-access", i64::MAX))
            .expect("account 1 login");
        let catalog = catalog(&agent_dir, &store, &[]);
        let accounts = Accounts::new(&catalog);
        let work = add_logged_in(&accounts, "Work", "work-access");

        // Account 1 is the default: the catalog's own token is used.
        assert_eq!(
            accounts.request_token("routing-session").expect("route"),
            None
        );
        choose_for_session("routing-session", &work).expect("choose");
        assert_eq!(
            accounts.request_token("routing-session").expect("route"),
            Some("work-access".to_owned())
        );
        assert_eq!(
            request_account("routing-session").as_deref(),
            Some(work.as_str())
        );
        // Other sessions are unaffected by this one's choice.
        assert_eq!(
            accounts.request_token("other-session").expect("route"),
            None
        );

        let listed = accounts.list("routing-session").expect("list");
        assert_eq!(
            listed.iter().map(AccountStatus::status).collect::<Vec<_>>(),
            ["Authenticated", "Active"]
        );

        // A signed-out choice falls back to the default account.
        accounts.logout(&work).expect("logout");
        assert_eq!(
            accounts.request_token("routing-session").expect("route"),
            None
        );
        assert_eq!(accounts.find("2").expect("find").status(), "Login required");

        // The environment token overrides every saved account.
        let environment =
            super::tests::catalog(&agent_dir, &store, &[(grok_cli::TOKEN_ENV, "env")]);
        let with_token = Accounts::new(&environment);
        assert_eq!(
            with_token.request_token("routing-session").expect("route"),
            None
        );
        assert_eq!(
            with_token.activate(ACCOUNT_1_ID).expect_err("refused"),
            "Saved accounts cannot be selected while the environment token is active."
        );
        assert_eq!(
            with_token.list("").expect("list")[0].status(),
            "Active (environment)"
        );
        let _ = fs::remove_dir_all(agent_dir);
    }

    #[test]
    fn a_saved_account_keeps_the_provider_usable_without_an_auth_json_login() {
        let agent_dir = temp_dir("fallback");
        let store = Arc::new(CredentialStore::in_memory());
        let catalog = catalog(&agent_dir, &store, &[]);
        assert!(
            !catalog
                .is_configured(grok_cli::PROVIDER_ID)
                .expect("configured")
        );
        let accounts = Accounts::new(&catalog);
        add_logged_in(&accounts, "Side", "side-access");
        let auth = catalog
            .resolve_auth(grok_cli::PROVIDER_ID)
            .expect("resolve")
            .expect("vault fallback");
        assert_eq!(auth.api_key(), Some("side-access"));
        assert_eq!(auth.source(), "Grok CLI account");
        assert!(grok_cli::credential_present(&catalog) || auth.api_key().is_some());
        let _ = fs::remove_dir_all(agent_dir);
    }

    struct RefreshTransport {
        requests: Mutex<Vec<oauth::OAuthRequest>>,
        responses: Mutex<VecDeque<oauth::OAuthResponse>>,
    }

    impl oauth::OAuthTransport for RefreshTransport {
        fn execute(
            &self,
            request: oauth::OAuthRequest,
            _: &oauth::CancellationToken,
        ) -> oauth::Result<oauth::OAuthResponse> {
            lock(&self.requests).push(request);
            lock(&self.responses)
                .pop_front()
                .ok_or_else(|| oauth::OAuthError::Transport("unexpected request".to_owned()))
        }
    }

    #[test]
    fn an_expired_saved_account_is_refreshed_once_under_the_vault_lock() {
        let agent_dir = temp_dir("refresh");
        let store = Arc::new(CredentialStore::in_memory());
        let transport = Arc::new(RefreshTransport {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(VecDeque::from([oauth::OAuthResponse {
                status: 200,
                body: br#"{"access_token":"fresh-access","refresh_token":"fresh-refresh","expires_in":3600}"#.to_vec(),
            }])),
        });
        let catalog =
            catalog(&agent_dir, &store, &[]).with_oauth_client(Arc::new(oauth::OAuthClient::new(
                transport.clone(),
                Arc::new(oauth::SystemClock),
                Arc::new(oauth::NoopBrowser),
                oauth::OAuthEndpoints::default(),
            )));
        let accounts = Accounts::new(&catalog);
        let id = accounts.add("Stale").expect("add");
        accounts
            .store_login(&id, 0, login("stale-access", 0))
            .expect("store");
        let revision = accounts.revision(&id).expect("revision");
        assert_eq!(accounts.vault_token(&id).expect("refresh"), "fresh-access");
        // The stored copy is now valid, so a second use sends nothing.
        assert_eq!(accounts.vault_token(&id).expect("cached"), "fresh-access");
        let requests = lock(&transport.requests);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].url().as_str(), "https://auth.x.ai/oauth2/token");
        let file = accounts.vault().load().expect("load");
        let account = file
            .accounts
            .iter()
            .find(|account| account.id == id)
            .expect("account");
        assert_eq!(account.revision, revision + 1);
        assert_eq!(
            account.credential.as_ref().map(Credential::refresh),
            Some("fresh-refresh")
        );
        let _ = fs::remove_dir_all(agent_dir);
    }

    #[test]
    fn quota_ordering_prefers_fresh_remaining_credit_and_keeps_the_rest_in_place() {
        let now = OffsetDateTime::now_utc();
        let stamp = |minutes: i64| {
            (now - time::Duration::minutes(minutes))
                .format(&time::format_description::well_known::Rfc3339)
                .expect("format")
        };
        let entry = |percent: f64, minutes: i64| grok_cli::CachedQuota {
            updated_at: stamp(minutes),
            usage: grok_cli::BillingUsage {
                tier: None,
                monthly: grok_cli::MonthlyUsage {
                    monthly_limit: 1.0,
                    used: 0.0,
                    billing_period_end: stamp(0),
                },
                weekly: Some(grok_cli::WeeklyUsage {
                    credit_usage_percent: percent,
                    billing_period_end: stamp(0),
                }),
            },
        };
        let quotas = std::collections::BTreeMap::from([
            ("b".to_owned(), entry(90.0, 1)),
            ("c".to_owned(), entry(10.0, 1)),
            // Stale figures do not count.
            ("d".to_owned(), entry(0.0, 45)),
        ]);
        let ids = ["a", "b", "c", "d"].map(str::to_owned).to_vec();
        assert_eq!(order_by_quota(ids, &quotas, now), ["a", "c", "b", "d"]);
    }

    fn exhausted(model: &llm::Model) -> llm::AssistantMessage {
        llm::AssistantMessage {
            role: "assistant".to_owned(),
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            stop_reason: stream::STOP_ERROR.to_owned(),
            error_message:
                "provider request failed with status 402: \"Grok Build usage balance exhausted\""
                    .to_owned(),
            ..llm::AssistantMessage::default()
        }
    }

    fn answered(model: &llm::Model, text: &str) -> llm::AssistantMessage {
        llm::AssistantMessage {
            role: "assistant".to_owned(),
            content: vec![llm::ContentBlock::text(text)],
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            stop_reason: stream::STOP_STOP.to_owned(),
            ..llm::AssistantMessage::default()
        }
    }

    struct Rotation {
        runtime: SessionRuntime,
        seen: Arc<Mutex<Vec<String>>>,
        _registration: grok_cli::SessionRegistration,
        _subscription: agent::Subscription,
    }

    /// A recorded session whose responder routes like the real one and
    /// answers with an exhaustion error for every token in `spent`.
    fn rotation_session(
        root: &Path,
        catalog: &Catalog,
        spent: &'static [&'static str],
    ) -> Rotation {
        let accounts = Accounts::new(catalog);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        let routing = accounts.clone();
        let responder: agent::AssistantResponder = Arc::new(move |model, _context, options| {
            let token = routing
                .request_token(&options.session_id)?
                .unwrap_or_else(|| "account-one-access".to_owned());
            lock(&log).push(token.clone());
            Ok(if spent.contains(&token.as_str()) {
                exhausted(model)
            } else {
                answered(model, "done")
            })
        });
        let model = catalog
            .model(grok_cli::PROVIDER_ID, "grok-build")
            .expect("grok-build");
        let runtime = SessionRuntime::open(SessionOptions {
            cwd: root.join("workspace"),
            sessions_dir: Some(root.join("sessions")),
            model,
            responder: Some(responder),
            ..SessionOptions::default()
        })
        .expect("session");
        let id = runtime.id().expect("recorded session");
        let registration =
            grok_cli::register_session_store(&id, Arc::new(runtime.custom_recorder()));
        let subscription =
            rotation_subscription(runtime.agent(), accounts, id, runtime.notice_sender());
        Rotation {
            runtime,
            seen,
            _registration: registration,
            _subscription: subscription,
        }
    }

    fn no_retry() -> turns::RetryPolicy {
        turns::RetryPolicy {
            enabled: false,
            ..turns::RetryPolicy::default()
        }
    }

    #[test]
    fn an_exhausted_account_hands_the_turn_to_the_next_one_and_records_the_switch() {
        let root = temp_dir("rotation");
        let store = Arc::new(CredentialStore::in_memory());
        store
            .put(grok_cli::PROVIDER_ID, login("account-one-access", i64::MAX))
            .expect("account 1 login");
        let catalog = catalog(&root.join("agent"), &store, &[]);
        let backup = add_logged_in(&Accounts::new(&catalog), "Backup", "backup-access");
        let rotation = rotation_session(&root, &catalog, &["account-one-access"]);
        let notices = rotation.runtime.notice_sender();
        turns::run_prompt(
            rotation.runtime.agent(),
            "build it",
            &no_retry(),
            Some(&notices),
        )
        .expect("prompt");

        assert_eq!(
            *lock(&rotation.seen),
            ["account-one-access", "backup-access"]
        );
        let messages = rotation.runtime.agent().state().messages;
        let texts = messages
            .iter()
            .map(llm::Message::text_preview)
            .collect::<Vec<_>>();
        assert!(
            texts.iter().any(|text| text == ROTATION_CONTINUATION),
            "{texts:?}"
        );
        assert!(matches!(
            messages.last(),
            Some(llm::Message::Assistant(message)) if message.stop_reason == stream::STOP_STOP
        ));
        let notices = rotation.runtime.drain_notices();
        assert!(
            notices.iter().any(|notice| notice.kind == "Grok CLI"
                && notice.text == "“Account 1” exhausted; switched to “Backup” and continuing."),
            "{notices:?}"
        );
        assert_eq!(
            rotation
                .runtime
                .custom_recorder()
                .custom_values(SESSION_ACCOUNT_ENTRY),
            [json!({ "accountId": backup })]
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rotation_stops_when_no_other_account_can_take_over() {
        let root = temp_dir("rotation-stops");
        let store = Arc::new(CredentialStore::in_memory());
        store
            .put(grok_cli::PROVIDER_ID, login("account-one-access", i64::MAX))
            .expect("account 1 login");
        let catalog = catalog(&root.join("agent"), &store, &[]);

        // A single account: the error stands and nothing is said or sent.
        let alone = rotation_session(&root, &catalog, &["account-one-access"]);
        let notices = alone.runtime.notice_sender();
        turns::run_prompt(
            alone.runtime.agent(),
            "build it",
            &no_retry(),
            Some(&notices),
        )
        .expect("prompt");
        assert_eq!(lock(&alone.seen).len(), 1);
        assert!(
            !alone
                .runtime
                .drain_notices()
                .iter()
                .any(|notice| notice.kind == "Grok CLI")
        );

        // Two accounts, both spent: one switch, then the chain gives up.
        add_logged_in(&Accounts::new(&catalog), "Second", "second-access");
        let both = rotation_session(&root, &catalog, &["account-one-access", "second-access"]);
        let notices = both.runtime.notice_sender();
        turns::run_prompt(
            both.runtime.agent(),
            "build it",
            &no_retry(),
            Some(&notices),
        )
        .expect("prompt");
        let seen = lock(&both.seen).clone();
        assert_eq!(seen.last().map(String::as_str), Some("second-access"));
        let notices = both.runtime.drain_notices();
        assert!(
            notices
                .iter()
                .any(|notice| notice.text == "all logged-in accounts are exhausted."),
            "{notices:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn the_chat_command_lists_switches_and_hands_logins_to_the_terminal() {
        let agent_dir = temp_dir("chat-command");
        let store = Arc::new(CredentialStore::in_memory());
        store
            .put(grok_cli::PROVIDER_ID, login("account-one-access", i64::MAX))
            .expect("account 1 login");
        let accounts = Accounts::new(&catalog(&agent_dir, &store, &[]));
        add_logged_in(&accounts, "Work", "work-access");
        let Ok(AccountsCommand::Done(listing)) = chat_command(&accounts, "chat-session", "") else {
            panic!("listing");
        };
        assert!(listing.contains("  1. Account 1 — Active"), "{listing}");
        assert!(listing.contains("  2. Work — Authenticated"), "{listing}");
        let Ok(AccountsCommand::Done(switched)) = chat_command(&accounts, "chat-session", "use 2")
        else {
            panic!("use");
        };
        assert_eq!(
            switched,
            "Grok CLI: using “Work” for this session and new ones."
        );
        assert_eq!(
            accounts.vault().load().expect("load").active_account_id,
            Some(accounts.find("work").expect("find").id)
        );
        assert!(matches!(
            chat_command(&accounts, "chat-session", "add Personal"),
            Ok(AccountsCommand::Terminal(arguments)) if arguments == ["accounts", "add", "Personal"]
        ));
        assert_eq!(
            chat_command(&accounts, "chat-session", "bogus")
                .err()
                .as_deref(),
            Some(ACCOUNTS_USAGE)
        );
        let _ = fs::remove_dir_all(agent_dir);
    }

    #[test]
    fn a_failed_cli_login_leaves_no_half_made_account() {
        let agent_dir = temp_dir("cli-add");
        let store = Arc::new(CredentialStore::in_memory());
        let accounts = Accounts::new(&catalog(&agent_dir, &store, &[]));
        let error = cli_command(&accounts, &["add", "Broken"], &|_| {
            Err("login cancelled".to_owned())
        })
        .expect_err("login fails");
        assert_eq!(error, "login cancelled");
        assert_eq!(accounts.vault().load().expect("load").accounts.len(), 1);
        let added = cli_command(&accounts, &["add", "Good"], &|_| {
            Ok(login("good-access", i64::MAX))
        })
        .expect("login succeeds");
        assert_eq!(added, "Added and logged in to “Good” (account 2).");
        assert_eq!(
            accounts.vault().load().expect("load").active_account_id,
            Some(accounts.find("good").expect("find").id)
        );
        let _ = fs::remove_dir_all(agent_dir);
    }

    #[test]
    fn a_lost_saved_login_is_not_refreshed_again_until_the_account_changes() {
        let agent_dir = temp_dir("refresh-failure");
        let store = Arc::new(CredentialStore::in_memory());
        let transport = Arc::new(RefreshTransport {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(VecDeque::from([
                oauth::OAuthResponse {
                    status: 401,
                    body: br#"{"error":"invalid_grant"}"#.to_vec(),
                },
                oauth::OAuthResponse {
                    status: 200,
                    body: br#"{"access_token":"relogged-access","expires_in":3600}"#.to_vec(),
                },
            ])),
        });
        let catalog =
            catalog(&agent_dir, &store, &[]).with_oauth_client(Arc::new(oauth::OAuthClient::new(
                transport.clone(),
                Arc::new(oauth::SystemClock),
                Arc::new(oauth::NoopBrowser),
                oauth::OAuthEndpoints::default(),
            )));
        let accounts = Accounts::new(&catalog);
        let id = accounts.add("Revoked").expect("add");
        accounts
            .store_login(&id, 0, login("dead-access", 0))
            .expect("store");
        let first = accounts.vault_token(&id).expect_err("revoked");
        assert!(first.starts_with("“Revoked”: "), "{first}");
        assert_eq!(accounts.vault_token(&id).expect_err("remembered"), first);
        assert_eq!(lock(&transport.requests).len(), 1, "no second refresh");
        // A new login is a new revision, and refreshes again.
        let revision = accounts.revision(&id).expect("revision");
        accounts
            .store_login(&id, revision, login("expired-again", 0))
            .expect("relogin");
        assert_eq!(
            accounts.vault_token(&id).expect("refresh"),
            "relogged-access"
        );
        assert_eq!(lock(&transport.requests).len(), 2);
        let _ = fs::remove_dir_all(agent_dir);
    }
}
