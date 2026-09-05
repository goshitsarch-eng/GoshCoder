//! Built-in model catalog, credential storage, and provider auth resolution.
//!
//! This is deliberately a standalone module while the provider runtime is
//! migrated. It uses the existing [`crate::llm`] model types and
//! [`crate::config`] path layout, but does not expose credentials through
//! `Debug` or error messages.
//!
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::{
    fd::AsRawFd,
    raw::c_int,
    unix::fs::{OpenOptionsExt, PermissionsExt},
};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as DeError};
use serde_json::{Map, Value};

use crate::{aperture, config, llm, oauth};

const CATALOG_JSON: &str = include_str!("../internal/llm/catalog/catalog.json");
const CATALOG_EXTRA_JSON: &str = include_str!("../internal/llm/catalog/catalog_extra.json");
const CATALOG_OVERRIDES_JSON: &str = include_str!("../internal/llm/catalog/catalog_overrides.json");

const MAX_AUTH_FILE_BYTES: usize = 10 * 1024 * 1024;
const COMMAND_OUTPUT_LIMIT: usize = 1024 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Marker returned for credentials that the wire implementation must discover
/// itself (AWS profiles, IRSA, Google Application Default Credentials, etc.).
pub const AUTHENTICATED_SENTINEL: &str = "<authenticated>";

/// Process environment lookup used by config values and provider resolution.
///
/// Returning `None` and returning an empty value both mean that the variable
/// is unavailable. A cloneable closure keeps tests and embedders independent
/// from the host process environment.
pub type EnvironmentLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// File-existence lookup used for Vertex Application Default Credentials.
pub type FileExists = Arc<dyn Fn(&Path) -> bool + Send + Sync>;

/// Returns an environment lookup backed by the current process.
pub fn process_environment() -> EnvironmentLookup {
    Arc::new(|name| env::var(name).ok())
}

fn default_file_exists(path: &Path) -> bool {
    path.is_file()
}

fn environment_value(lookup: &EnvironmentLookup, name: &str) -> Option<String> {
    lookup(name).filter(|value| !value.is_empty())
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Errors emitted by catalog loading, credential persistence, and resolution.
///
/// Variants intentionally identify configuration names and model references,
/// never credential values, command text, or resolved header values.
#[derive(Debug)]
pub enum CatalogError {
    EmbeddedCatalog(String),
    InvalidCredentialFile,
    InvalidCredentialField(&'static str),
    KnownCredentialFieldCannotBeExtra(String),
    CredentialFileTooLarge,
    CredentialSerialization,
    Io {
        operation: &'static str,
        source: io::Error,
    },
    MissingConfigValue {
        description: String,
        variables: Vec<String>,
    },
    CommandConfigValueFailed {
        description: String,
    },
    UnknownModel {
        reference: String,
    },
    ProviderNotConfigured {
        provider_id: String,
    },
    NoConfiguredProvider {
        model_id: String,
    },
    AmbiguousModel {
        model_id: String,
        provider_ids: Vec<String>,
    },
    OAuthClientUnavailable,
    OAuthRefreshFailed {
        provider_id: String,
    },
}

impl fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbeddedCatalog(message) => {
                write!(formatter, "catalog data is invalid: {message}")
            }
            Self::InvalidCredentialFile => write!(formatter, "failed to parse auth.json"),
            Self::InvalidCredentialField(field) => {
                write!(formatter, "credential field {field:?} has an invalid type")
            }
            Self::KnownCredentialFieldCannotBeExtra(field) => {
                write!(
                    formatter,
                    "{field:?} is a known credential field, not an extra"
                )
            }
            Self::CredentialFileTooLarge => write!(formatter, "auth.json exceeds 10 MiB"),
            Self::CredentialSerialization => write!(formatter, "failed to serialize auth.json"),
            Self::Io { operation, source } => write!(formatter, "failed to {operation}: {source}"),
            Self::MissingConfigValue {
                description,
                variables,
            } => match variables.as_slice() {
                [] => write!(formatter, "failed to resolve {description}"),
                [variable] => write!(
                    formatter,
                    "failed to resolve {description} from environment variable {variable}"
                ),
                _ => write!(
                    formatter,
                    "failed to resolve {description} from environment variables {}",
                    variables.join(", ")
                ),
            },
            Self::CommandConfigValueFailed { description } => {
                write!(
                    formatter,
                    "failed to resolve {description} from its shell command"
                )
            }
            Self::UnknownModel { reference } => write!(formatter, "unknown model {reference:?}"),
            Self::ProviderNotConfigured { provider_id } => {
                write!(
                    formatter,
                    "provider {provider_id:?} has no credentials configured"
                )
            }
            Self::NoConfiguredProvider { model_id } => {
                write!(
                    formatter,
                    "no configured provider offers model {model_id:?}"
                )
            }
            Self::AmbiguousModel {
                model_id,
                provider_ids,
            } => write!(
                formatter,
                "model {model_id:?} is ambiguous; qualify it as one of {}",
                provider_ids.join(", ")
            ),
            Self::OAuthClientUnavailable => {
                formatter.write_str("could not initialize the OAuth client")
            }
            Self::OAuthRefreshFailed { provider_id } => write!(
                formatter,
                "OAuth credential for {provider_id:?} could not be refreshed; run `goshcoder auth login {provider_id}`"
            ),
        }
    }
}

impl Error for CatalogError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Selects a provider's non-OAuth authentication strategy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthKind {
    /// A stored API key, then the first configured environment key.
    EnvKey,
    /// Anthropic's bearer-token and API-key environment conventions.
    Anthropic,
    /// Cloudflare Workers AI's API key plus account ID.
    CloudflareWorkersAi,
    /// Cloudflare AI Gateway's API key, account ID, and gateway ID.
    CloudflareAiGateway,
    /// Ambient AWS credentials consumed by the Bedrock wire protocol.
    AmbientBedrock,
    /// Google Cloud API key or Application Default Credentials.
    Vertex,
    /// OAuth credentials are the only supported credential type.
    OAuthOnly,
    /// Meta Model API key sent in an Authorization bearer header.
    MetaBearer,
}

#[derive(Clone, Copy)]
struct ProviderDefinition {
    id: &'static str,
    name: &'static str,
    base_url: &'static str,
    key_name: &'static str,
    env_keys: &'static [&'static str],
    auth_kind: AuthKind,
    supports_oauth: bool,
}

/// Static provider metadata ported from the prior provider catalog.
///
/// The list is intentionally separate from the model JSON: a provider can be
/// known even when the generated snapshot has no built-in models for it.
const PROVIDER_DEFINITIONS: &[ProviderDefinition] = &[
    ProviderDefinition {
        id: "amazon-bedrock",
        name: "Amazon Bedrock",
        base_url: "",
        key_name: "AWS credentials or bearer token",
        env_keys: &[],
        auth_kind: AuthKind::AmbientBedrock,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "ant-ling",
        name: "Ant Ling",
        base_url: "https://api.ant-ling.com/v1",
        key_name: "Ant Ling API key",
        env_keys: &["ANT_LING_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "anthropic",
        name: "Anthropic",
        base_url: "https://api.anthropic.com",
        key_name: "Anthropic API key",
        env_keys: &[],
        auth_kind: AuthKind::Anthropic,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "aperture",
        name: "Aperture",
        base_url: "",
        key_name: "Aperture gateway",
        env_keys: &[],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "azure-openai-responses",
        name: "Azure OpenAI",
        base_url: "",
        key_name: "Azure OpenAI API key",
        env_keys: &["AZURE_OPENAI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "baseten",
        name: "Baseten",
        base_url: "https://inference.baseten.co/v1",
        key_name: "Baseten API key",
        env_keys: &["BASETEN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "cerebras",
        name: "Cerebras",
        base_url: "https://api.cerebras.ai/v1",
        key_name: "Cerebras API key",
        env_keys: &["CEREBRAS_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "cloudflare-ai-gateway",
        name: "Cloudflare AI Gateway",
        base_url: "",
        key_name: "Cloudflare API key",
        env_keys: &[],
        auth_kind: AuthKind::CloudflareAiGateway,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "cloudflare-workers-ai",
        name: "Cloudflare Workers AI",
        base_url: "",
        key_name: "Cloudflare API key",
        env_keys: &[],
        auth_kind: AuthKind::CloudflareWorkersAi,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "deepseek",
        name: "DeepSeek",
        base_url: "https://api.deepseek.com",
        key_name: "DeepSeek API key",
        env_keys: &["DEEPSEEK_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "fireworks",
        name: "Fireworks",
        base_url: "https://api.fireworks.ai/inference",
        key_name: "Fireworks API key",
        env_keys: &["FIREWORKS_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "github-copilot",
        name: "GitHub Copilot",
        base_url: "https://api.individual.githubcopilot.com",
        key_name: "GitHub Copilot token",
        env_keys: &["COPILOT_GITHUB_TOKEN"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "google",
        name: "Google",
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        key_name: "Gemini API key",
        env_keys: &["GEMINI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "google-vertex",
        name: "Google Vertex AI",
        base_url: "",
        key_name: "Google Cloud credentials",
        env_keys: &[],
        auth_kind: AuthKind::Vertex,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "groq",
        name: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        key_name: "Groq API key",
        env_keys: &["GROQ_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "huggingface",
        name: "Hugging Face",
        base_url: "https://router.huggingface.co/v1",
        key_name: "Hugging Face token",
        env_keys: &["HF_TOKEN"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "kimi-coding",
        name: "Kimi For Coding",
        base_url: "https://api.kimi.com/coding",
        key_name: "Kimi API key",
        env_keys: &["KIMI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "meta",
        name: "Meta",
        base_url: "https://api.meta.ai",
        key_name: "Meta Model API key",
        env_keys: &["META_API_KEY"],
        auth_kind: AuthKind::MetaBearer,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "minimax",
        name: "MiniMax",
        base_url: "https://api.minimax.io/anthropic",
        key_name: "MiniMax API key",
        env_keys: &["MINIMAX_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "minimax-cn",
        name: "MiniMax CN",
        base_url: "https://api.minimaxi.com/anthropic",
        key_name: "MiniMax CN API key",
        env_keys: &["MINIMAX_CN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "mistral",
        name: "Mistral",
        base_url: "https://api.mistral.ai",
        key_name: "Mistral API key",
        env_keys: &["MISTRAL_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "moonshotai",
        name: "Moonshot AI",
        base_url: "https://api.moonshot.ai/v1",
        key_name: "Moonshot AI API key",
        env_keys: &["MOONSHOT_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "moonshotai-cn",
        name: "Moonshot AI CN",
        base_url: "https://api.moonshot.cn/v1",
        key_name: "Moonshot AI API key",
        env_keys: &["MOONSHOT_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "nvidia",
        name: "NVIDIA",
        base_url: "https://integrate.api.nvidia.com/v1",
        key_name: "NVIDIA API key",
        env_keys: &["NVIDIA_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "openai",
        name: "OpenAI",
        base_url: "https://api.openai.com/v1",
        key_name: "OpenAI API key",
        env_keys: &["OPENAI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "openai-codex",
        name: "OpenAI Codex",
        base_url: "https://chatgpt.com/backend-api",
        key_name: "",
        env_keys: &[],
        auth_kind: AuthKind::OAuthOnly,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "opencode",
        name: "OpenCode Zen",
        base_url: "",
        key_name: "OpenCode API key",
        env_keys: &["OPENCODE_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "opencode-go",
        name: "OpenCode Go",
        base_url: "",
        key_name: "OpenCode API key",
        env_keys: &["OPENCODE_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "openrouter",
        name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        key_name: "OpenRouter API key",
        env_keys: &["OPENROUTER_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "omni",
        name: "OmniRoute",
        base_url: "http://127.0.0.1:20128/v1",
        key_name: "OmniRoute API key",
        env_keys: &["OMNIROUTE_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "qwen-token-plan",
        name: "Qwen Token Plan",
        base_url: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        key_name: "Qwen Token Plan API key",
        env_keys: &["QWEN_TOKEN_PLAN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "qwen-token-plan-cn",
        name: "Qwen Token Plan CN",
        base_url: "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        key_name: "Qwen Token Plan CN API key",
        env_keys: &["QWEN_TOKEN_PLAN_CN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "qwen-token-plan-individual",
        name: "Qwen Token Plan Individual",
        base_url: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        key_name: "Qwen Token Plan Individual API key",
        env_keys: &["QWEN_TOKEN_PLAN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "radius",
        name: "Radius",
        base_url: "",
        key_name: "Radius API key",
        env_keys: &["RADIUS_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "together",
        name: "Together",
        base_url: "https://api.together.ai/v1",
        key_name: "Together API key",
        env_keys: &["TOGETHER_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "vercel-ai-gateway",
        name: "Vercel AI Gateway",
        base_url: "https://ai-gateway.vercel.sh",
        key_name: "Vercel AI Gateway API key",
        env_keys: &["AI_GATEWAY_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "xai",
        name: "xAI",
        base_url: "https://api.x.ai/v1",
        key_name: "xAI API key",
        env_keys: &["XAI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: true,
    },
    ProviderDefinition {
        id: "xiaomi",
        name: "Xiaomi",
        base_url: "https://api.xiaomimimo.com/v1",
        key_name: "Xiaomi API key",
        env_keys: &["XIAOMI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "xiaomi-token-plan-ams",
        name: "Xiaomi Token Plan AMS",
        base_url: "https://token-plan-ams.xiaomimimo.com/v1",
        key_name: "Xiaomi Token Plan AMS API key",
        env_keys: &["XIAOMI_TOKEN_PLAN_AMS_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "xiaomi-token-plan-cn",
        name: "Xiaomi Token Plan CN",
        base_url: "https://token-plan-cn.xiaomimimo.com/v1",
        key_name: "Xiaomi Token Plan CN API key",
        env_keys: &["XIAOMI_TOKEN_PLAN_CN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "xiaomi-token-plan-sgp",
        name: "Xiaomi Token Plan SGP",
        base_url: "https://token-plan-sgp.xiaomimimo.com/v1",
        key_name: "Xiaomi Token Plan SGP API key",
        env_keys: &["XIAOMI_TOKEN_PLAN_SGP_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "zai",
        name: "Z.AI",
        base_url: "https://api.z.ai/api/coding/paas/v4",
        key_name: "Z.AI API key",
        env_keys: &["ZAI_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
    ProviderDefinition {
        id: "zai-coding-cn",
        name: "Z.AI Coding CN",
        base_url: "https://open.bigmodel.cn/api/coding/paas/v4",
        key_name: "Z.AI Coding CN API key",
        env_keys: &["ZAI_CODING_CN_API_KEY"],
        auth_kind: AuthKind::EnvKey,
        supports_oauth: false,
    },
];

fn provider_definition(id: &str) -> Option<&'static ProviderDefinition> {
    PROVIDER_DEFINITIONS
        .iter()
        .find(|definition| definition.id == id)
}

/// A provider and its catalog models. Model accessors always return copies.
#[derive(Clone, Debug)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub key_name: String,
    pub env_keys: Vec<String>,
    pub auth_kind: AuthKind,
    pub supports_oauth: bool,
    models: Vec<llm::Model>,
    raw_compat: BTreeMap<String, Value>,
}

impl Provider {
    /// Returns independent copies of the provider's models, ordered by ID.
    pub fn models(&self) -> Vec<llm::Model> {
        self.models.clone()
    }

    /// Returns an independent copy of one model.
    pub fn model(&self, id: &str) -> Option<llm::Model> {
        self.models.iter().find(|model| model.id == id).cloned()
    }

    /// Returns raw `compat` metadata, including protocol-specific fields that
    /// are not otherwise interpreted by `llm::Model`.
    pub fn raw_compat(&self, model_id: &str) -> Option<Value> {
        self.raw_compat.get(model_id).cloned()
    }
}

type RawModel = Map<String, Value>;
type RawCatalog = BTreeMap<String, BTreeMap<String, RawModel>>;

/// The configurable files which establish one immutable Aperture catalog
/// snapshot. Kept alongside the catalog so cloned catalogs observe the same
/// routing state throughout a session.
#[derive(Clone)]
struct AperturePaths {
    config: PathBuf,
    cache: PathBuf,
}

impl Default for AperturePaths {
    fn default() -> Self {
        Self {
            config: config::aperture_path(),
            cache: config::aperture_cache_path(),
        }
    }
}

#[derive(Clone)]
struct BuiltinData {
    models: BTreeMap<String, BTreeMap<String, llm::Model>>,
    raw_compat: BTreeMap<String, BTreeMap<String, Value>>,
}

static BUILTIN_DATA: OnceLock<Result<BuiltinData, String>> = OnceLock::new();

fn builtin_data() -> Result<&'static BuiltinData, CatalogError> {
    let cached = BUILTIN_DATA.get_or_init(|| {
        merge_catalog_documents(CATALOG_JSON, CATALOG_EXTRA_JSON, CATALOG_OVERRIDES_JSON)
            .map_err(|error| error.to_string())
    });
    cached
        .as_ref()
        .map_err(|message| CatalogError::EmbeddedCatalog(message.clone()))
}

fn parse_raw_catalog(document: &str, description: &str) -> Result<RawCatalog, CatalogError> {
    serde_json::from_str(document).map_err(|_| {
        CatalogError::EmbeddedCatalog(format!("could not parse {description} as a model catalog"))
    })
}

/// Merges generated catalog data, hand-maintained extras, and field overrides.
///
/// All maps are ordered maps, and override fields are sorted explicitly, so
/// results do not depend on JSON object iteration order. Generated models win
/// over duplicate extra models; an override may only target a generated model
/// and may not rewrite its `id` or `provider` identity.
fn merge_catalog_documents(
    generated_json: &str,
    extras_json: &str,
    overrides_json: &str,
) -> Result<BuiltinData, CatalogError> {
    let mut generated = parse_raw_catalog(generated_json, "catalog.json")?;
    let overrides = parse_raw_catalog(overrides_json, "catalog_overrides.json")?;

    for (provider_id, provider_overrides) in overrides {
        let Some(provider_models) = generated.get_mut(&provider_id) else {
            return Err(CatalogError::EmbeddedCatalog(format!(
                "catalog_overrides.json targets unknown provider {provider_id:?}"
            )));
        };

        for (model_id, patch) in provider_overrides {
            let Some(model) = provider_models.get_mut(&model_id) else {
                return Err(CatalogError::EmbeddedCatalog(format!(
                    "catalog_overrides.json targets unknown model {provider_id}/{model_id}"
                )));
            };

            let mut fields: Vec<_> = patch.into_iter().collect();
            fields.sort_by(|left, right| left.0.cmp(&right.0));
            for (field, value) in fields {
                if field == "id" || field == "provider" {
                    return Err(CatalogError::EmbeddedCatalog(format!(
                        "catalog_overrides.json cannot override {field:?}"
                    )));
                }
                model.insert(field, value);
            }
        }
    }

    let extras = parse_raw_catalog(extras_json, "catalog_extra.json")?;
    for (provider_id, provider_extras) in extras {
        let target = generated.entry(provider_id).or_default();
        for (model_id, model) in provider_extras {
            // Extras exist only for models absent from the generated source.
            // `or_insert` deliberately makes generated data authoritative.
            target.entry(model_id).or_insert(model);
        }
    }

    let mut models = BTreeMap::new();
    let mut raw_compat = BTreeMap::new();
    for (provider_id, provider_models) in generated {
        let mut decoded_models = BTreeMap::new();
        let mut provider_compat = BTreeMap::new();
        for (model_id, object) in provider_models {
            let compat = object
                .get("compat")
                .filter(|value| !value.is_null())
                .cloned();
            let model: llm::Model =
                serde_json::from_value(Value::Object(object)).map_err(|_| {
                    CatalogError::EmbeddedCatalog(format!(
                        "model {provider_id}/{model_id} has an invalid llm::Model shape"
                    ))
                })?;
            if model.id != model_id || model.provider != provider_id {
                return Err(CatalogError::EmbeddedCatalog(format!(
                    "model {provider_id}/{model_id} does not match its catalog identity"
                )));
            }
            if let Some(compat) = compat {
                provider_compat.insert(model_id.clone(), compat);
            }
            decoded_models.insert(model_id, model);
        }
        if !provider_compat.is_empty() {
            raw_compat.insert(provider_id.clone(), provider_compat);
        }
        models.insert(provider_id, decoded_models);
    }

    Ok(BuiltinData { models, raw_compat })
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum CredentialKind {
    ApiKey,
    OAuth,
    /// Preserved so a newer producer's auth.json does not lose its type.
    Other(String),
}

impl CredentialKind {
    fn from_json_type(value: &str) -> Self {
        match value {
            "api_key" => Self::ApiKey,
            "oauth" => Self::OAuth,
            other => Self::Other(other.to_owned()),
        }
    }

    /// The interoperable `auth.json` value.
    pub fn as_str(&self) -> &str {
        match self {
            Self::ApiKey => "api_key",
            Self::OAuth => "oauth",
            Self::Other(value) => value,
        }
    }
}

/// One `auth.json` credential.
///
/// This type intentionally does not implement `Debug` or `Display`: its key,
/// OAuth tokens, environment values, and extension fields are sensitive.
#[derive(Clone, Eq, PartialEq)]
pub struct Credential {
    kind: CredentialKind,
    key: String,
    environment: BTreeMap<String, String>,
    refresh: String,
    access: String,
    expires_at_ms: i64,
    extra: BTreeMap<String, Value>,
}

impl Credential {
    /// Creates an API-key credential. The key may be a literal, `$ENV`
    /// template, or `!command` config value.
    pub fn api_key(key: impl Into<String>) -> Self {
        Self {
            kind: CredentialKind::ApiKey,
            key: key.into(),
            environment: BTreeMap::new(),
            refresh: String::new(),
            access: String::new(),
            expires_at_ms: 0,
            extra: BTreeMap::new(),
        }
    }

    /// Creates an OAuth credential for persistence only.
    ///
    /// TODO: OAuth login, expiry handling, and refresh are intentionally not
    /// implemented in this non-OAuth catalog port.
    pub fn oauth(
        access: impl Into<String>,
        refresh: impl Into<String>,
        expires_at_ms: i64,
    ) -> Self {
        Self {
            kind: CredentialKind::OAuth,
            key: String::new(),
            environment: BTreeMap::new(),
            refresh: refresh.into(),
            access: access.into(),
            expires_at_ms,
            extra: BTreeMap::new(),
        }
    }

    /// Creates a forward-compatible credential of an unknown type.
    pub fn other(kind: impl Into<String>) -> Self {
        Self {
            kind: CredentialKind::Other(kind.into()),
            key: String::new(),
            environment: BTreeMap::new(),
            refresh: String::new(),
            access: String::new(),
            expires_at_ms: 0,
            extra: BTreeMap::new(),
        }
    }

    pub fn kind(&self) -> &CredentialKind {
        &self.kind
    }

    /// Returns the stored key or resolved API key. Callers must treat it as a
    /// secret and avoid logging it.
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn set_key(&mut self, key: impl Into<String>) {
        self.key = key.into();
    }

    /// Provider-scoped values used by config-value templates and selected
    /// auth strategies. Values are sensitive when used as credentials.
    pub fn environment(&self) -> &BTreeMap<String, String> {
        &self.environment
    }

    pub fn set_environment(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.environment.insert(name.into(), value.into());
    }

    pub fn remove_environment(&mut self, name: &str) -> Option<String> {
        self.environment.remove(name)
    }

    /// OAuth refresh token, retained for interoperability but never emitted by
    /// this module's auth resolver.
    pub fn refresh(&self) -> &str {
        &self.refresh
    }

    pub fn set_refresh(&mut self, refresh: impl Into<String>) {
        self.refresh = refresh.into();
    }

    /// OAuth access token, retained for interoperability but never emitted by
    /// this module's auth resolver.
    pub fn access(&self) -> &str {
        &self.access
    }

    pub fn set_access(&mut self, access: impl Into<String>) {
        self.access = access.into();
    }

    pub fn expires_at_ms(&self) -> i64 {
        self.expires_at_ms
    }

    pub fn set_expires_at_ms(&mut self, expires_at_ms: i64) {
        self.expires_at_ms = expires_at_ms;
    }

    /// Returns an unknown provider-specific field without coercing its JSON
    /// type. Such fields round-trip through auth.json unchanged.
    pub fn extra(&self, name: &str) -> Option<&Value> {
        self.extra.get(name)
    }

    pub fn extra_string(&self, name: &str) -> Option<&str> {
        self.extra(name).and_then(Value::as_str)
    }

    /// Sets a provider-specific field. Known auth.json fields are rejected so
    /// they cannot be shadowed or serialized ambiguously.
    pub fn set_extra(&mut self, name: impl Into<String>, value: Value) -> Result<(), CatalogError> {
        let name = name.into();
        if is_known_credential_field(&name) {
            return Err(CatalogError::KnownCredentialFieldCannotBeExtra(name));
        }
        self.extra.insert(name, value);
        Ok(())
    }

    pub fn remove_extra(&mut self, name: &str) -> Option<Value> {
        self.extra.remove(name)
    }
}

fn is_known_credential_field(name: &str) -> bool {
    matches!(
        name,
        "type" | "key" | "env" | "refresh" | "access" | "expires"
    )
}

fn credential_string(
    object: &Map<String, Value>,
    name: &'static str,
) -> Result<String, CatalogError> {
    match object.get(name) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(CatalogError::InvalidCredentialField(name)),
    }
}

fn credential_environment(
    object: &Map<String, Value>,
) -> Result<BTreeMap<String, String>, CatalogError> {
    let Some(value) = object.get("env") else {
        return Ok(BTreeMap::new());
    };
    let Value::Object(values) = value else {
        if value.is_null() {
            return Ok(BTreeMap::new());
        }
        return Err(CatalogError::InvalidCredentialField("env"));
    };

    let mut environment = BTreeMap::new();
    for (name, value) in values {
        let Some(value) = value.as_str() else {
            return Err(CatalogError::InvalidCredentialField("env"));
        };
        environment.insert(name.clone(), value.to_owned());
    }
    Ok(environment)
}

fn credential_expiry(object: &Map<String, Value>) -> Result<i64, CatalogError> {
    match object.get("expires") {
        None | Some(Value::Null) => Ok(0),
        Some(Value::Number(value)) => value
            .as_i64()
            .ok_or(CatalogError::InvalidCredentialField("expires")),
        Some(_) => Err(CatalogError::InvalidCredentialField("expires")),
    }
}

impl<'de> Deserialize<'de> for Credential {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let object = Map::<String, Value>::deserialize(deserializer)?;
        let kind = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("credential field \"type\" has an invalid type"))?
            .to_owned();
        let key = credential_string(&object, "key").map_err(D::Error::custom)?;
        let environment = credential_environment(&object).map_err(D::Error::custom)?;
        let refresh = credential_string(&object, "refresh").map_err(D::Error::custom)?;
        let access = credential_string(&object, "access").map_err(D::Error::custom)?;
        let expires_at_ms = credential_expiry(&object).map_err(D::Error::custom)?;
        let extra = object
            .into_iter()
            .filter(|(name, _)| !is_known_credential_field(name))
            .collect();

        Ok(Self {
            kind: CredentialKind::from_json_type(&kind),
            key,
            environment,
            refresh,
            access,
            expires_at_ms,
            extra,
        })
    }
}

impl Serialize for Credential {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut object = Map::new();
        for (name, value) in &self.extra {
            object.insert(name.clone(), value.clone());
        }
        object.insert(
            "type".to_owned(),
            Value::String(self.kind.as_str().to_owned()),
        );
        if !self.key.is_empty() {
            object.insert("key".to_owned(), Value::String(self.key.clone()));
        }
        if !self.environment.is_empty() {
            object.insert(
                "env".to_owned(),
                serde_json::to_value(&self.environment).map_err(serde::ser::Error::custom)?,
            );
        }
        if self.kind == CredentialKind::OAuth {
            object.insert("refresh".to_owned(), Value::String(self.refresh.clone()));
            object.insert("access".to_owned(), Value::String(self.access.clone()));
            object.insert(
                "expires".to_owned(),
                Value::Number(self.expires_at_ms.into()),
            );
        }
        Value::Object(object).serialize(serializer)
    }
}

/// Non-secret credential metadata suitable for provider/status lists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub kind: CredentialKind,
}

enum CredentialBacking {
    Memory,
    File(PathBuf),
}

/// In-memory or auth.json-backed credential storage.
///
/// File writes use a sibling advisory lock on Unix, a same-directory private
/// temporary file, `fsync`, and atomic rename. The final auth.json mode is
/// forced to `0600` on Unix.
pub struct CredentialStore {
    backing: CredentialBacking,
    memory: Mutex<BTreeMap<String, Credential>>,
    environment: EnvironmentLookup,
}

impl CredentialStore {
    pub fn in_memory() -> Self {
        Self {
            backing: CredentialBacking::Memory,
            memory: Mutex::new(BTreeMap::new()),
            environment: process_environment(),
        }
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self {
            backing: CredentialBacking::File(path.into()),
            memory: Mutex::new(BTreeMap::new()),
            environment: process_environment(),
        }
    }

    /// Uses the durable auth.json location defined by `src/config.rs`.
    pub fn default_file() -> Self {
        Self::file(config::auth_path())
    }

    /// Replaces the lookup used by [`Self::read`]. This is chiefly useful for
    /// embedding and tests; catalogs pass their own lookup to
    /// [`Self::read_with_environment`].
    pub fn with_environment(mut self, environment: EnvironmentLookup) -> Self {
        self.environment = environment;
        self
    }

    pub fn path(&self) -> Option<&Path> {
        match &self.backing {
            CredentialBacking::Memory => None,
            CredentialBacking::File(path) => Some(path),
        }
    }

    /// Reads the persisted credential without resolving `$ENV` or `!command`
    /// API-key values.
    pub fn read_raw(&self, provider_id: &str) -> Result<Option<Credential>, CatalogError> {
        let memory = lock_unpoisoned(&self.memory);
        let credentials = match &self.backing {
            CredentialBacking::Memory => memory.clone(),
            CredentialBacking::File(path) => read_auth_file(path)?,
        };
        Ok(credentials.get(provider_id).cloned())
    }

    /// Reads a credential and resolves its API key with the store's configured
    /// environment lookup.
    pub fn read(&self, provider_id: &str) -> Result<Option<Credential>, CatalogError> {
        self.read_with_environment(provider_id, &self.environment)
    }

    /// Reads a credential and resolves an API-key config value with `lookup`.
    /// The persisted source remains unchanged.
    pub fn read_with_environment(
        &self,
        provider_id: &str,
        lookup: &EnvironmentLookup,
    ) -> Result<Option<Credential>, CatalogError> {
        let Some(mut credential) = self.read_raw(provider_id)? else {
            return Ok(None);
        };
        if credential.kind == CredentialKind::ApiKey && !credential.key.is_empty() {
            credential.key = resolve_config_value(&credential.key, &credential.environment, lookup)
                .unwrap_or_default();
        }
        Ok(Some(credential))
    }

    /// Lists persisted credentials in provider-ID order without reading any
    /// secret field.
    pub fn list(&self) -> Result<Vec<CredentialInfo>, CatalogError> {
        let memory = lock_unpoisoned(&self.memory);
        let credentials = match &self.backing {
            CredentialBacking::Memory => memory.clone(),
            CredentialBacking::File(path) => read_auth_file(path)?,
        };
        Ok(credentials
            .into_iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id,
                kind: credential.kind,
            })
            .collect())
    }

    /// Atomically replaces one credential. `None` from `update` preserves the
    /// current entry, matching the prior storage API; use [`Self::delete`] to
    /// remove it deliberately.
    pub fn modify<F>(
        &self,
        provider_id: &str,
        update: F,
    ) -> Result<Option<Credential>, CatalogError>
    where
        F: FnOnce(Option<Credential>) -> Result<Option<Credential>, CatalogError>,
    {
        let mut memory = lock_unpoisoned(&self.memory);
        let _file_lock = match &self.backing {
            CredentialBacking::Memory => None,
            CredentialBacking::File(path) => Some(acquire_auth_file_lock(path)?),
        };
        let mut credentials = match &self.backing {
            CredentialBacking::Memory => memory.clone(),
            CredentialBacking::File(path) => read_auth_file(path)?,
        };
        let current = credentials.get(provider_id).cloned();
        let Some(next) = update(current.clone())? else {
            return Ok(current);
        };

        credentials.insert(provider_id.to_owned(), next.clone());
        match &self.backing {
            CredentialBacking::Memory => *memory = credentials,
            CredentialBacking::File(path) => write_auth_file(path, &credentials)?,
        }
        Ok(Some(next))
    }

    /// Replaces one credential.
    pub fn put(
        &self,
        provider_id: &str,
        credential: Credential,
    ) -> Result<Credential, CatalogError> {
        let stored = credential.clone();
        self.modify(provider_id, move |_| Ok(Some(stored)))?;
        Ok(credential)
    }

    /// Removes one credential and returns whether an entry was present.
    pub fn delete(&self, provider_id: &str) -> Result<bool, CatalogError> {
        let mut memory = lock_unpoisoned(&self.memory);
        let _file_lock = match &self.backing {
            CredentialBacking::Memory => None,
            CredentialBacking::File(path) => Some(acquire_auth_file_lock(path)?),
        };
        let mut credentials = match &self.backing {
            CredentialBacking::Memory => memory.clone(),
            CredentialBacking::File(path) => read_auth_file(path)?,
        };
        let removed = credentials.remove(provider_id).is_some();
        match &self.backing {
            CredentialBacking::Memory => *memory = credentials,
            CredentialBacking::File(path) => write_auth_file(path, &credentials)?,
        }
        Ok(removed)
    }
}

impl Default for CredentialStore {
    fn default() -> Self {
        Self::in_memory()
    }
}

fn read_auth_file(path: &Path) -> Result<BTreeMap<String, Credential>, CatalogError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(source) => {
            return Err(CatalogError::Io {
                operation: "read auth.json",
                source,
            });
        }
    };

    let mut bytes = Vec::new();
    let mut limited = file.take((MAX_AUTH_FILE_BYTES + 1) as u64);
    limited
        .read_to_end(&mut bytes)
        .map_err(|source| CatalogError::Io {
            operation: "read auth.json",
            source,
        })?;
    if bytes.len() > MAX_AUTH_FILE_BYTES {
        return Err(CatalogError::CredentialFileTooLarge);
    }
    if bytes.is_empty() {
        return Ok(BTreeMap::new());
    }
    serde_json::from_slice(&bytes).map_err(|_| CatalogError::InvalidCredentialFile)
}

fn auth_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn ensure_auth_parent(path: &Path) -> Result<&Path, CatalogError> {
    let parent = auth_parent(path);
    let needs_creation = !parent.exists();
    fs::create_dir_all(parent).map_err(|source| CatalogError::Io {
        operation: "create the auth.json directory",
        source,
    })?;
    #[cfg(unix)]
    if needs_creation {
        fs::set_permissions(parent, fs::Permissions::from_mode(0o700)).map_err(|source| {
            CatalogError::Io {
                operation: "secure the auth.json directory",
                source,
            }
        })?;
    }
    Ok(parent)
}

fn write_auth_file(
    path: &Path,
    credentials: &BTreeMap<String, Credential>,
) -> Result<(), CatalogError> {
    let mut bytes = serde_json::to_vec_pretty(credentials)
        .map_err(|_| CatalogError::CredentialSerialization)?;
    bytes.push(b'\n');
    atomic_private_write(path, &bytes)
}

fn atomic_private_write(path: &Path, bytes: &[u8]) -> Result<(), CatalogError> {
    let parent = ensure_auth_parent(path)?;
    let file_name = path.file_name().ok_or_else(|| {
        CatalogError::EmbeddedCatalog("auth.json path has no file name".to_owned())
    })?;

    let mut temporary = None;
    let mut file = None;
    for _ in 0..32 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            process::id(),
            sequence
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&candidate) {
            Ok(opened) => {
                temporary = Some(candidate);
                file = Some(opened);
                break;
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(CatalogError::Io {
                    operation: "create a private auth.json temporary file",
                    source,
                });
            }
        }
    }

    let temporary = temporary.ok_or_else(|| CatalogError::Io {
        operation: "create a unique auth.json temporary file",
        source: io::Error::new(
            io::ErrorKind::AlreadyExists,
            "temporary file name collision",
        ),
    })?;
    let mut file = file.ok_or_else(|| CatalogError::Io {
        operation: "create an auth.json temporary file",
        source: io::Error::other("temporary file was not retained"),
    })?;
    let write_result = (|| {
        file.write_all(bytes).map_err(|source| CatalogError::Io {
            operation: "write auth.json",
            source,
        })?;
        file.sync_all().map_err(|source| CatalogError::Io {
            operation: "sync auth.json",
            source,
        })?;
        drop(file);
        fs::rename(&temporary, path).map_err(|source| CatalogError::Io {
            operation: "atomically replace auth.json",
            source,
        })?;
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|source| {
            CatalogError::Io {
                operation: "secure auth.json",
                source,
            }
        })?;
        // Directory fsync makes the rename durable where the filesystem
        // supports it. The data file was already synced; unsupported directory
        // syncing is not a reason to reject an otherwise successful write.
        #[cfg(unix)]
        {
            let _ = File::open(parent).and_then(|directory| directory.sync_all());
        }
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

#[cfg(unix)]
const LOCK_EX: c_int = 2;
#[cfg(unix)]
const LOCK_UN: c_int = 8;

#[cfg(unix)]
unsafe extern "C" {
    fn flock(fd: c_int, operation: c_int) -> c_int;
}

#[cfg(unix)]
struct AuthFileLock {
    file: File,
}

#[cfg(unix)]
impl Drop for AuthFileLock {
    fn drop(&mut self) {
        // Closing also releases flock, but unlock explicitly so a held file
        // descriptor never keeps a lock after this guard's logical lifetime.
        unsafe {
            let _ = flock(self.file.as_raw_fd(), LOCK_UN);
        }
    }
}

#[cfg(not(unix))]
struct AuthFileLock;

fn acquire_auth_file_lock(path: &Path) -> Result<AuthFileLock, CatalogError> {
    let parent = ensure_auth_parent(path)?;
    let file_name = path.file_name().ok_or_else(|| {
        CatalogError::EmbeddedCatalog("auth.json path has no file name".to_owned())
    })?;
    let lock_path = parent.join(format!("{}.lock", file_name.to_string_lossy()));
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    options.mode(0o600);
    let file = options
        .open(&lock_path)
        .map_err(|source| CatalogError::Io {
            operation: "open the auth.json lock",
            source,
        })?;

    #[cfg(unix)]
    {
        fs::set_permissions(&lock_path, fs::Permissions::from_mode(0o600)).map_err(|source| {
            CatalogError::Io {
                operation: "secure the auth.json lock",
                source,
            }
        })?;
        if unsafe { flock(file.as_raw_fd(), LOCK_EX) } != 0 {
            return Err(CatalogError::Io {
                operation: "lock auth.json",
                source: io::Error::last_os_error(),
            });
        }
        Ok(AuthFileLock { file })
    }

    #[cfg(not(unix))]
    {
        let _ = file;
        Ok(AuthFileLock)
    }
}

enum TemplatePart {
    Literal(String),
    Environment(String),
}

enum ConfigValueReference {
    Command(String),
    Template(Vec<TemplatePart>),
}

fn append_literal(parts: &mut Vec<TemplatePart>, value: &str) {
    if value.is_empty() {
        return;
    }
    match parts.last_mut() {
        Some(TemplatePart::Literal(previous)) => previous.push_str(value),
        _ => parts.push(TemplatePart::Literal(value.to_owned())),
    }
}

fn is_environment_name(name: &str) -> bool {
    let mut characters = name.bytes();
    let Some(first) = characters.next() else {
        return false;
    };
    if !(first == b'_' || first.is_ascii_alphabetic()) {
        return false;
    }
    characters.all(|character| character == b'_' || character.is_ascii_alphanumeric())
}

fn parse_config_value(config: &str) -> ConfigValueReference {
    if config.starts_with('!') {
        return ConfigValueReference::Command(config.to_owned());
    }

    let mut parts = Vec::new();
    let bytes = config.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let Some(relative_dollar) = config[index..].find('$') else {
            append_literal(&mut parts, &config[index..]);
            break;
        };
        let dollar = index + relative_dollar;
        append_literal(&mut parts, &config[index..dollar]);
        let next = bytes.get(dollar + 1).copied();

        if matches!(next, Some(b'$' | b'!')) {
            append_literal(&mut parts, if next == Some(b'$') { "$" } else { "!" });
            index = dollar + 2;
            continue;
        }

        if next == Some(b'{') {
            let content_start = dollar + 2;
            if let Some(relative_end) = config[content_start..].find('}') {
                let end = content_start + relative_end;
                let name = &config[content_start..end];
                if is_environment_name(name) {
                    parts.push(TemplatePart::Environment(name.to_owned()));
                } else {
                    append_literal(&mut parts, &config[dollar..=end]);
                }
                index = end + 1;
                continue;
            }
            append_literal(&mut parts, "$");
            index = dollar + 1;
            continue;
        }

        let name_start = dollar + 1;
        let mut name_end = name_start;
        while name_end < bytes.len()
            && (bytes[name_end] == b'_' || bytes[name_end].is_ascii_alphanumeric())
        {
            name_end += 1;
        }
        if name_end > name_start && is_environment_name(&config[name_start..name_end]) {
            parts.push(TemplatePart::Environment(
                config[name_start..name_end].to_owned(),
            ));
            index = name_end;
        } else {
            append_literal(&mut parts, "$");
            index = dollar + 1;
        }
    }
    ConfigValueReference::Template(parts)
}

fn resolve_environment_config_value(
    name: &str,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Option<String> {
    scoped_environment
        .get(name)
        .filter(|value| !value.is_empty())
        .cloned()
        .or_else(|| environment_value(lookup, name))
}

fn resolve_template(
    parts: &[TemplatePart],
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Option<String> {
    let mut resolved = String::new();
    for part in parts {
        match part {
            TemplatePart::Literal(value) => resolved.push_str(value),
            TemplatePart::Environment(name) => {
                resolved.push_str(&resolve_environment_config_value(
                    name,
                    scoped_environment,
                    lookup,
                )?);
            }
        }
    }
    Some(resolved)
}

/// Returns the environment name when `config` is exactly one environment
/// reference, such as `$TOKEN` or `${TOKEN}`.
pub fn config_value_environment_name(config: &str) -> Option<String> {
    match parse_config_value(config) {
        ConfigValueReference::Template(parts) if parts.len() == 1 => match &parts[0] {
            TemplatePart::Environment(name) => Some(name.clone()),
            TemplatePart::Literal(_) => None,
        },
        ConfigValueReference::Template(_) | ConfigValueReference::Command(_) => None,
    }
}

/// Returns each referenced environment name once, in first-use order.
pub fn config_value_environment_names(config: &str) -> Vec<String> {
    let ConfigValueReference::Template(parts) = parse_config_value(config) else {
        return Vec::new();
    };
    let mut names = Vec::new();
    for part in parts {
        let TemplatePart::Environment(name) = part else {
            continue;
        };
        if !names.iter().any(|existing| existing == &name) {
            names.push(name);
        }
    }
    names
}

/// Returns referenced variables unavailable in either scoped or process env.
pub fn missing_config_value_environment_names(
    config: &str,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Vec<String> {
    config_value_environment_names(config)
        .into_iter()
        .filter(|name| resolve_environment_config_value(name, scoped_environment, lookup).is_none())
        .collect()
}

/// Reports whether `config` starts with an executable `!command`.
pub fn is_command_config_value(config: &str) -> bool {
    matches!(parse_config_value(config), ConfigValueReference::Command(_))
}

/// Reports whether all template variables resolve. Commands are considered
/// configured without executing them, matching the prior behavior.
pub fn is_config_value_configured(
    config: &str,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> bool {
    missing_config_value_environment_names(config, scoped_environment, lookup).is_empty()
}

/// Resolves a literal, environment template, or cached `!command`.
///
/// `None` indicates a missing template variable, empty command output,
/// non-zero command exit, non-UTF-8 command output, timeout, or output over
/// 1 MiB. Command failures deliberately do not include command text in errors.
pub fn resolve_config_value(
    config: &str,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Option<String> {
    match parse_config_value(config) {
        ConfigValueReference::Command(command) => execute_cached_command(&command),
        ConfigValueReference::Template(parts) => {
            resolve_template(&parts, scoped_environment, lookup)
        }
    }
}

/// Resolves a config value without using the process-lifetime command cache.
pub fn resolve_config_value_uncached(
    config: &str,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Option<String> {
    match parse_config_value(config) {
        ConfigValueReference::Command(command) => execute_command(&command),
        ConfigValueReference::Template(parts) => {
            resolve_template(&parts, scoped_environment, lookup)
        }
    }
}

/// Resolves a config value or returns an error that identifies only the
/// requested description and missing variable names.
pub fn resolve_config_value_or_error(
    config: &str,
    description: impl Into<String>,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Result<String, CatalogError> {
    let description = description.into();
    if let Some(value) = resolve_config_value_uncached(config, scoped_environment, lookup) {
        return Ok(value);
    }
    if is_command_config_value(config) {
        return Err(CatalogError::CommandConfigValueFailed { description });
    }
    Err(CatalogError::MissingConfigValue {
        description,
        variables: missing_config_value_environment_names(config, scoped_environment, lookup),
    })
}

/// Resolves configured header values, dropping unresolvable or empty entries.
pub fn resolve_headers(
    headers: &BTreeMap<String, String>,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            resolve_config_value(value, scoped_environment, lookup)
                .filter(|value| !value.is_empty())
                .map(|value| (name.clone(), value))
        })
        .collect()
}

/// Resolves configured header values, failing if any one cannot resolve.
pub fn resolve_headers_or_error(
    headers: &BTreeMap<String, String>,
    description: &str,
    scoped_environment: &BTreeMap<String, String>,
    lookup: &EnvironmentLookup,
) -> Result<BTreeMap<String, String>, CatalogError> {
    let mut resolved = BTreeMap::new();
    for (name, value) in headers {
        let value = resolve_config_value_or_error(
            value,
            format!("{description} header {name:?}"),
            scoped_environment,
            lookup,
        )?;
        resolved.insert(name.clone(), value);
    }
    Ok(resolved)
}

static COMMAND_CACHE: OnceLock<Mutex<BTreeMap<String, Option<String>>>> = OnceLock::new();

fn command_cache() -> &'static Mutex<BTreeMap<String, Option<String>>> {
    COMMAND_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Clears cached command results. Environment templates are never cached.
pub fn clear_config_value_cache() {
    lock_unpoisoned(command_cache()).clear();
}

fn execute_cached_command(command: &str) -> Option<String> {
    if let Some(cached) = lock_unpoisoned(command_cache()).get(command).cloned() {
        return cached;
    }

    let result = execute_command(command);
    lock_unpoisoned(command_cache()).insert(command.to_owned(), result.clone());
    result
}

fn execute_command(command_config: &str) -> Option<String> {
    let command = command_config.strip_prefix('!')?;
    let mut process = if cfg!(windows) {
        let mut process = Command::new("cmd.exe");
        process.args(["/d", "/s", "/c", command]);
        process
    } else {
        let mut process = Command::new("sh");
        process.args(["-c", command]);
        process
    };
    process
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = process.spawn().ok()?;
    let stdout = child.stdout.take()?;
    let reader = thread::spawn(move || drain_command_output(stdout));
    let deadline = Instant::now() + COMMAND_TIMEOUT;

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return None;
            }
        }
    };
    let (output, truncated) = reader.join().ok()?.ok()?;
    if !status.success() || truncated {
        return None;
    }
    let output = String::from_utf8(output).ok()?;
    let output = output.trim().to_owned();
    (!output.is_empty()).then_some(output)
}

fn drain_command_output(mut stdout: impl Read) -> io::Result<(Vec<u8>, bool)> {
    let mut output = Vec::new();
    let mut truncated = false;
    let mut buffer = [0_u8; 8192];
    loop {
        let read = stdout.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = COMMAND_OUTPUT_LIMIT.saturating_sub(output.len());
        let kept = remaining.min(read);
        output.extend_from_slice(&buffer[..kept]);
        if kept < read {
            truncated = true;
        }
    }
    Ok((output, truncated))
}

/// Header overrides supplied by provider authentication. `None` explicitly
/// suppresses a protocol default header.
pub type AuthHeaders = BTreeMap<String, Option<String>>;

/// Resolved provider authentication.
///
/// This type deliberately does not implement `Debug`, `Display`, or serde
/// traits because it can hold an API key or resolved bearer header.
#[derive(Clone, Eq, PartialEq)]
pub struct Auth {
    api_key: Option<String>,
    environment: BTreeMap<String, String>,
    headers: AuthHeaders,
    source: String,
}

impl Auth {
    fn with_api_key(
        api_key: String,
        environment: BTreeMap<String, String>,
        headers: AuthHeaders,
        source: impl Into<String>,
    ) -> Self {
        Self {
            api_key: Some(api_key),
            environment,
            headers,
            source: source.into(),
        }
    }

    fn without_api_key(
        environment: BTreeMap<String, String>,
        headers: AuthHeaders,
        source: impl Into<String>,
    ) -> Self {
        Self {
            api_key: None,
            environment,
            headers,
            source: source.into(),
        }
    }

    /// Returns the key for a request builder. Callers must not log it.
    pub fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }

    pub fn environment(&self) -> &BTreeMap<String, String> {
        &self.environment
    }

    /// Returns auth header overrides. Values are request secrets and must not
    /// be logged.
    pub fn headers(&self) -> &AuthHeaders {
        &self.headers
    }

    pub fn header(&self, name: &str) -> Option<Option<&str>> {
        self.headers.get(name).map(|value| value.as_deref())
    }

    /// Non-secret explanation of where the credential was discovered.
    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn is_ambient(&self) -> bool {
        self.api_key.as_deref() == Some(AUTHENTICATED_SENTINEL)
    }
}

/// The result of model resolution, carrying a model copy and secret-bearing
/// request auth separately.
///
/// It intentionally does not implement `Debug`, because effective headers can
/// contain values resolved from credential-scoped templates.
pub struct ResolvedModel {
    pub model: llm::Model,
    auth: Auth,
    effective_headers: AuthHeaders,
}

impl ResolvedModel {
    pub fn auth(&self) -> &Auth {
        &self.auth
    }

    /// Returns static model headers after config-value expansion, combined
    /// with auth headers. `None` suppresses a default protocol header.
    pub fn effective_headers(&self) -> &AuthHeaders {
        &self.effective_headers
    }

    pub fn into_parts(self) -> (llm::Model, Auth) {
        (self.model, self.auth)
    }
}

/// Provider collection backed by the embedded catalog and optional auth store.
///
/// Clones share the credential store and read-only catalog data. This allows
/// request-scoped tools to resolve current credentials without retaining a
/// borrow of a session constructor.
#[derive(Clone)]
pub struct Catalog {
    data: &'static BuiltinData,
    credentials: Option<Arc<CredentialStore>>,
    environment: EnvironmentLookup,
    file_exists: FileExists,
    oauth_client: Arc<oauth::OAuthClient>,
    oauth_refresh_failures: Arc<Mutex<BTreeSet<String>>>,
    aperture_paths: AperturePaths,
    aperture_state: Arc<Mutex<Option<aperture::ApertureState>>>,
}

impl Catalog {
    /// Creates a catalog using the current process environment.
    pub fn new(credentials: Option<Arc<CredentialStore>>) -> Result<Self, CatalogError> {
        Self::with_environment_and_file_exists(
            credentials,
            process_environment(),
            Arc::new(default_file_exists),
        )
    }

    /// Creates a catalog that persists credentials at `config::auth_path()`.
    pub fn with_default_credentials() -> Result<Self, CatalogError> {
        Self::new(Some(Arc::new(CredentialStore::default_file())))
    }

    pub fn with_environment(
        credentials: Option<Arc<CredentialStore>>,
        environment: EnvironmentLookup,
    ) -> Result<Self, CatalogError> {
        Self::with_environment_and_file_exists(
            credentials,
            environment,
            Arc::new(default_file_exists),
        )
    }

    /// Overrides the Aperture configuration and cache locations for this
    /// catalog. Embedders use this to keep a session's gateway routing
    /// separate from the process-wide agent directory.
    pub fn with_aperture_paths(
        mut self,
        configuration: impl Into<PathBuf>,
        cache: impl Into<PathBuf>,
    ) -> Self {
        self.aperture_paths = AperturePaths {
            config: configuration.into(),
            cache: cache.into(),
        };
        self.aperture_state = Arc::new(Mutex::new(None));
        self
    }

    /// Creates a catalog with injectable environment and ADC file checks.
    pub fn with_environment_and_file_exists(
        credentials: Option<Arc<CredentialStore>>,
        environment: EnvironmentLookup,
        file_exists: FileExists,
    ) -> Result<Self, CatalogError> {
        let oauth_client =
            oauth::OAuthClient::system().map_err(|_| CatalogError::OAuthClientUnavailable)?;
        Ok(Self {
            data: builtin_data()?,
            credentials,
            environment,
            file_exists,
            oauth_client: Arc::new(oauth_client),
            oauth_refresh_failures: Arc::new(Mutex::new(BTreeSet::new())),
            aperture_paths: AperturePaths::default(),
            aperture_state: Arc::new(Mutex::new(None)),
        })
    }

    pub fn credentials(&self) -> Option<&CredentialStore> {
        self.credentials.as_deref()
    }

    /// Clears a cached OAuth-refresh failure after an interactive login
    /// updates the credential store. Clones of this catalog share the cache.
    pub fn clear_oauth_refresh_failure(&self, provider_id: &str) {
        lock_unpoisoned(&self.oauth_refresh_failures).remove(provider_id);
    }

    /// Returns every statically known provider ID in lexical order.
    pub fn provider_ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = PROVIDER_DEFINITIONS
            .iter()
            .map(|definition| definition.id.to_owned())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Returns provider IDs present in embedded model data, in lexical order.
    pub fn builtin_provider_ids(&self) -> Vec<String> {
        self.data.models.keys().cloned().collect()
    }

    /// Returns Aperture's immutable dedicated/proxy routing state for this
    /// catalog. Configuration errors leave the static catalog usable; the
    /// Aperture management command reports those errors explicitly.
    pub fn aperture_state(&self) -> aperture::ApertureState {
        let mut state = lock_unpoisoned(&self.aperture_state);
        if state.is_none() {
            *state = Some(self.build_aperture_state());
        }
        state.clone().expect("Aperture state was initialized")
    }

    /// Re-reads Aperture configuration and cache after a successful gateway
    /// sync, so an already-open interactive session can select fresh models.
    pub fn reload_aperture_state(&self) -> aperture::ApertureState {
        let state = self.build_aperture_state();
        *lock_unpoisoned(&self.aperture_state) = Some(state.clone());
        state
    }

    /// Returns the configuration location used for this catalog's Aperture
    /// state. Runtime integrations use this rather than process-global paths
    /// so embedded and test catalogs keep their gateway state isolated.
    pub fn aperture_config_path(&self) -> PathBuf {
        self.aperture_paths.config.clone()
    }

    /// Returns the cache location used for this catalog's Aperture state.
    pub fn aperture_cache_path(&self) -> PathBuf {
        self.aperture_paths.cache.clone()
    }

    fn build_aperture_state(&self) -> aperture::ApertureState {
        let Ok(configuration) = aperture::load_config(&self.aperture_paths.config) else {
            return aperture::ApertureState::default();
        };
        let cache = aperture::load_cache(&self.aperture_paths.cache).ok();
        aperture::build_aperture_state(&configuration, cache.as_ref(), |provider_id| {
            self.native_provider_info(provider_id)
        })
    }

    fn native_provider_info(&self, provider_id: &str) -> Option<aperture::NativeProviderInfo> {
        let models = self.data.models.get(provider_id)?;
        let first_model = models.values().next()?;
        let mut base_url = first_model.base_url.clone();
        if let Some(definition) = provider_definition(provider_id)
            && !definition.base_url.is_empty()
        {
            base_url = definition.base_url.to_owned();
        }
        Some(aperture::NativeProviderInfo {
            api: first_model.api.clone(),
            base_url,
            model_ids: models.keys().cloned().collect(),
        })
    }

    fn aperture_proxy_auth(&self, provider_id: &str) -> Option<Auth> {
        let state = self.aperture_state();
        let route = state.routes.get(provider_id)?;
        (!route.passthrough).then(|| {
            Auth::with_api_key(
                "-".to_owned(),
                BTreeMap::new(),
                BTreeMap::new(),
                "aperture proxy",
            )
        })
    }

    /// Returns an independent provider/model view.
    ///
    /// TODO: Load configured OmniRoute base URLs and models here. This method
    /// intentionally exposes only embedded static Omni metadata.
    pub fn provider(&self, id: &str) -> Option<Provider> {
        let definition = provider_definition(id)?;
        let models = self
            .data
            .models
            .get(id)
            .map(|models| models.values().cloned().collect())
            .unwrap_or_default();
        let raw_compat = self.data.raw_compat.get(id).cloned().unwrap_or_default();
        let mut provider = Provider {
            id: definition.id.to_owned(),
            name: definition.name.to_owned(),
            base_url: definition.base_url.to_owned(),
            key_name: definition.key_name.to_owned(),
            env_keys: definition
                .env_keys
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            auth_kind: definition.auth_kind,
            supports_oauth: definition.supports_oauth,
            models,
            raw_compat,
        };
        let aperture_state = self.aperture_state();
        if id == aperture::DEDICATED_PROVIDER_ID {
            if aperture_state.configured && aperture_state.resolved.dedicated_enabled {
                provider.base_url = aperture::provider_base_url(&aperture_state.resolved.base_url);
                provider.models = aperture_state.dedicated_models.clone();
            }
        } else if let Some(route) = aperture_state.routes.get(id) {
            provider.models = provider
                .models
                .iter()
                .filter_map(|model| aperture::apply_proxy_route(model, route))
                .collect();
            provider.base_url = route.base_url.clone();
        }
        Some(provider)
    }

    /// Returns every statically known provider in lexical ID order.
    pub fn providers(&self) -> Vec<Provider> {
        self.provider_ids()
            .into_iter()
            .filter_map(|id| self.provider(&id))
            .collect()
    }

    /// Returns an independent model copy.
    pub fn model(&self, provider_id: &str, model_id: &str) -> Option<llm::Model> {
        self.provider(provider_id)
            .and_then(|provider| provider.model(model_id))
    }

    /// Returns raw protocol compatibility metadata for one model.
    pub fn raw_compat(&self, provider_id: &str, model_id: &str) -> Option<Value> {
        self.provider(provider_id)
            .and_then(|provider| provider.raw_compat(model_id))
    }

    /// Resolves stored or ambient authentication for a known provider.
    pub fn resolve_auth(&self, provider_id: &str) -> Result<Option<Auth>, CatalogError> {
        self.resolve_auth_with_optional_key(provider_id, None)
    }

    /// Resolves auth, giving a direct API-key override precedence over stored
    /// and ambient API-key credentials.
    pub fn resolve_auth_with_key(
        &self,
        provider_id: &str,
        api_key: &str,
    ) -> Result<Option<Auth>, CatalogError> {
        self.resolve_auth_with_optional_key(provider_id, (!api_key.is_empty()).then_some(api_key))
    }

    fn resolve_auth_with_optional_key(
        &self,
        provider_id: &str,
        override_key: Option<&str>,
    ) -> Result<Option<Auth>, CatalogError> {
        let Some(definition) = provider_definition(provider_id) else {
            return Ok(None);
        };

        if let Some(override_key) = override_key {
            if definition.auth_kind == AuthKind::OAuthOnly {
                return Ok(None);
            }
            let override_credential = Credential::api_key(override_key);
            return Ok(self.build_api_key_auth(definition, Some(&override_credential), "override"));
        }

        if provider_id == aperture::DEDICATED_PROVIDER_ID {
            let state = self.aperture_state();
            return Ok(
                (state.configured && state.resolved.dedicated_enabled).then(|| {
                    Auth::with_api_key(
                        "-".to_owned(),
                        BTreeMap::new(),
                        BTreeMap::new(),
                        "aperture gateway",
                    )
                }),
            );
        }

        if let Some(proxy_auth) = self.aperture_proxy_auth(provider_id) {
            if definition.supports_oauth
                && let Some(store) = &self.credentials
                && let Some(stored) = store.read_with_environment(provider_id, &self.environment)?
                && *stored.kind() == CredentialKind::OAuth
                && let Ok(Some(auth)) = self.resolve_stored_oauth(provider_id)
            {
                return Ok(Some(auth));
            }
            return Ok(Some(proxy_auth));
        }

        if let Some(store) = &self.credentials
            && let Some(stored) = store.read_with_environment(provider_id, &self.environment)?
        {
            match stored.kind() {
                CredentialKind::ApiKey => {
                    if definition.auth_kind == AuthKind::OAuthOnly {
                        return Ok(None);
                    }
                    return Ok(self.build_api_key_auth(
                        definition,
                        Some(&stored),
                        "stored credential",
                    ));
                }
                CredentialKind::OAuth => {
                    // An OAuth credential owns its provider. Do not silently
                    // fall back to an ambient API key if refresh fails.
                    return self.resolve_stored_oauth(provider_id);
                }
                CredentialKind::Other(_) => return Ok(None),
            }
        }

        if definition.auth_kind == AuthKind::OAuthOnly {
            return Ok(None);
        }
        Ok(self.build_api_key_auth(definition, None, ""))
    }

    fn resolve_stored_oauth(&self, provider_id: &str) -> Result<Option<Auth>, CatalogError> {
        let Some(provider) = oauth::OAuthProviderId::parse(provider_id) else {
            return Ok(None);
        };
        let Some(store) = self.credentials.as_deref() else {
            return Ok(None);
        };
        if lock_unpoisoned(&self.oauth_refresh_failures).contains(provider_id) {
            return Err(CatalogError::OAuthRefreshFailed {
                provider_id: provider_id.to_owned(),
            });
        }

        let environment = oauth::CatalogEnvironment::new(Arc::clone(&self.environment));
        let cancellation = oauth::CancellationToken::new();
        match self
            .oauth_client
            .resolve_stored_oauth(provider, store, &environment, &cancellation)
        {
            Ok(Some(auth)) => {
                self.clear_oauth_refresh_failure(provider_id);
                let (api_key, headers, source) = auth.into_parts();
                let auth = match api_key {
                    Some(api_key) => Auth::with_api_key(api_key, BTreeMap::new(), headers, source),
                    None => Auth::without_api_key(BTreeMap::new(), headers, source),
                };
                Ok(Some(auth))
            }
            Ok(None) => {
                self.clear_oauth_refresh_failure(provider_id);
                Ok(None)
            }
            Err(_) => {
                lock_unpoisoned(&self.oauth_refresh_failures).insert(provider_id.to_owned());
                Err(CatalogError::OAuthRefreshFailed {
                    provider_id: provider_id.to_owned(),
                })
            }
        }
    }

    fn build_api_key_auth(
        &self,
        definition: &ProviderDefinition,
        credential: Option<&Credential>,
        source: &str,
    ) -> Option<Auth> {
        match definition.auth_kind {
            AuthKind::EnvKey => self.resolve_environment_key_auth(definition, credential, source),
            AuthKind::Anthropic => self.resolve_anthropic_auth(credential, source),
            AuthKind::CloudflareWorkersAi => {
                self.resolve_cloudflare_auth(credential, source, false)
            }
            AuthKind::CloudflareAiGateway => self.resolve_cloudflare_auth(credential, source, true),
            AuthKind::AmbientBedrock => self.resolve_bedrock_auth(credential, source),
            AuthKind::Vertex => self.resolve_vertex_auth(credential, source),
            AuthKind::MetaBearer => self.resolve_meta_auth(definition, credential, source),
            AuthKind::OAuthOnly => None,
        }
    }

    fn resolve_environment_key_auth(
        &self,
        definition: &ProviderDefinition,
        credential: Option<&Credential>,
        source: &str,
    ) -> Option<Auth> {
        if let Some(credential) = credential.filter(|credential| !credential.key().is_empty()) {
            return Some(Auth::with_api_key(
                credential.key().to_owned(),
                credential.environment().clone(),
                BTreeMap::new(),
                source,
            ));
        }
        for environment_name in definition.env_keys {
            if let Some(value) = environment_value(&self.environment, environment_name) {
                return Some(Auth::with_api_key(
                    value,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    *environment_name,
                ));
            }
        }
        None
    }

    fn resolve_anthropic_auth(
        &self,
        credential: Option<&Credential>,
        source: &str,
    ) -> Option<Auth> {
        if let Some(credential) = credential.filter(|credential| !credential.key().is_empty()) {
            return Some(Auth::with_api_key(
                credential.key().to_owned(),
                credential.environment().clone(),
                BTreeMap::new(),
                source,
            ));
        }
        if let Some(token) = environment_value(&self.environment, "ANTHROPIC_AUTH_TOKEN") {
            let mut headers = BTreeMap::new();
            headers.insert("Authorization".to_owned(), Some(format!("Bearer {token}")));
            headers.insert("x-api-key".to_owned(), None);
            return Some(Auth::with_api_key(
                token,
                BTreeMap::new(),
                headers,
                "ANTHROPIC_AUTH_TOKEN",
            ));
        }
        for environment_name in ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"] {
            if let Some(value) = environment_value(&self.environment, environment_name) {
                return Some(Auth::with_api_key(
                    value,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    environment_name,
                ));
            }
        }
        None
    }

    fn resolve_cloudflare_auth(
        &self,
        credential: Option<&Credential>,
        source: &str,
        gateway: bool,
    ) -> Option<Auth> {
        let resolve = |name: &str| {
            credential
                .and_then(|credential| {
                    if name == "CLOUDFLARE_API_KEY" {
                        (!credential.key().is_empty()).then(|| credential.key().to_owned())
                    } else {
                        credential
                            .environment()
                            .get(name)
                            .filter(|value| !value.is_empty())
                            .cloned()
                    }
                })
                .or_else(|| environment_value(&self.environment, name))
        };
        let api_key = resolve("CLOUDFLARE_API_KEY")?;
        let account_id = resolve("CLOUDFLARE_ACCOUNT_ID")?;
        let gateway_id = gateway.then(|| resolve("CLOUDFLARE_GATEWAY_ID")).flatten();
        if gateway && gateway_id.is_none() {
            return None;
        }

        let mut environment = BTreeMap::from([("CLOUDFLARE_ACCOUNT_ID".to_owned(), account_id)]);
        let mut headers = BTreeMap::new();
        if let Some(gateway_id) = gateway_id {
            environment.insert("CLOUDFLARE_GATEWAY_ID".to_owned(), gateway_id);
            headers.insert(
                "cf-aig-authorization".to_owned(),
                Some(format!("Bearer {api_key}")),
            );
            headers.insert("Authorization".to_owned(), None);
            headers.insert("x-api-key".to_owned(), None);
        }
        Some(Auth::with_api_key(
            api_key,
            environment,
            headers,
            if source.is_empty() {
                "CLOUDFLARE_API_KEY"
            } else {
                source
            },
        ))
    }

    fn resolve_bedrock_auth(&self, credential: Option<&Credential>, source: &str) -> Option<Auth> {
        if let Some(credential) = credential.filter(|credential| !credential.key().is_empty()) {
            return Some(Auth::with_api_key(
                credential.key().to_owned(),
                credential.environment().clone(),
                BTreeMap::new(),
                source,
            ));
        }
        if environment_value(&self.environment, "AWS_ACCESS_KEY_ID").is_some()
            && environment_value(&self.environment, "AWS_SECRET_ACCESS_KEY").is_some()
        {
            return Some(Auth::with_api_key(
                AUTHENTICATED_SENTINEL.to_owned(),
                BTreeMap::new(),
                BTreeMap::new(),
                "AWS_ACCESS_KEY_ID",
            ));
        }
        for environment_name in [
            "AWS_PROFILE",
            "AWS_BEARER_TOKEN_BEDROCK",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        ] {
            if environment_value(&self.environment, environment_name).is_some() {
                return Some(Auth::with_api_key(
                    AUTHENTICATED_SENTINEL.to_owned(),
                    BTreeMap::new(),
                    BTreeMap::new(),
                    environment_name,
                ));
            }
        }
        None
    }

    fn resolve_vertex_auth(&self, credential: Option<&Credential>, source: &str) -> Option<Auth> {
        let environment = self.vertex_environment(credential);
        if let Some(credential) = credential.filter(|credential| !credential.key().is_empty()) {
            return Some(Auth::with_api_key(
                credential.key().to_owned(),
                environment,
                BTreeMap::new(),
                source,
            ));
        }
        if let Some(key) = environment
            .get("GOOGLE_CLOUD_API_KEY")
            .filter(|key| !key.is_empty())
        {
            return Some(Auth::with_api_key(
                key.clone(),
                environment,
                BTreeMap::new(),
                "GOOGLE_CLOUD_API_KEY",
            ));
        }

        let has_project = environment
            .get("GOOGLE_CLOUD_PROJECT")
            .is_some_and(|project| !project.is_empty())
            || environment
                .get("GCLOUD_PROJECT")
                .is_some_and(|project| !project.is_empty());
        let has_location = environment
            .get("GOOGLE_CLOUD_LOCATION")
            .is_some_and(|location| !location.is_empty());
        if has_project
            && has_location
            && environment
                .get("GOOGLE_OAUTH_ACCESS_TOKEN")
                .is_some_and(|token| !token.is_empty())
        {
            return Some(Auth::with_api_key(
                AUTHENTICATED_SENTINEL.to_owned(),
                environment,
                BTreeMap::new(),
                "GOOGLE_OAUTH_ACCESS_TOKEN",
            ));
        }
        if has_project && has_location && self.has_vertex_adc_credentials(&environment) {
            return Some(Auth::with_api_key(
                AUTHENTICATED_SENTINEL.to_owned(),
                environment,
                BTreeMap::new(),
                "Application Default Credentials",
            ));
        }
        None
    }

    fn vertex_environment(&self, credential: Option<&Credential>) -> BTreeMap<String, String> {
        let mut environment = credential
            .map(|credential| credential.environment().clone())
            .unwrap_or_default();
        for name in [
            "GOOGLE_CLOUD_API_KEY",
            "GOOGLE_OAUTH_ACCESS_TOKEN",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GCLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
        ] {
            environment
                .entry(name.to_owned())
                .or_insert_with(|| environment_value(&self.environment, name).unwrap_or_default());
        }
        environment.retain(|_, value| !value.is_empty());
        environment
    }

    fn has_vertex_adc_credentials(&self, environment: &BTreeMap<String, String>) -> bool {
        if let Some(path) = environment
            .get("GOOGLE_APPLICATION_CREDENTIALS")
            .filter(|path| !path.is_empty())
        {
            return (self.file_exists)(Path::new(&path));
        }
        let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) else {
            return false;
        };
        (self.file_exists)(
            &PathBuf::from(home)
                .join(".config")
                .join("gcloud")
                .join("application_default_credentials.json"),
        )
    }

    fn resolve_meta_auth(
        &self,
        definition: &ProviderDefinition,
        credential: Option<&Credential>,
        source: &str,
    ) -> Option<Auth> {
        let (key, environment, source) = if let Some(credential) =
            credential.filter(|credential| !credential.key().is_empty())
        {
            (
                credential.key().to_owned(),
                credential.environment().clone(),
                source.to_owned(),
            )
        } else {
            let environment_name = definition.env_keys.iter().find_map(|name| {
                environment_value(&self.environment, name).map(|value| (*name, value))
            })?;
            (
                environment_name.1,
                BTreeMap::new(),
                environment_name.0.to_owned(),
            )
        };
        let mut headers = BTreeMap::new();
        headers.insert("Authorization".to_owned(), Some(format!("Bearer {key}")));
        // Keep the Meta key out of APIKey: Anthropic-shaped request builders
        // otherwise inject x-api-key, which Meta does not accept.
        Some(Auth::without_api_key(environment, headers, source))
    }

    /// Returns API-key environment variables currently present for a provider.
    /// Ambient AWS and Google ADC sources are intentionally omitted.
    pub fn find_environment_keys(&self, provider_id: &str) -> Vec<String> {
        let Some(definition) = provider_definition(provider_id) else {
            return Vec::new();
        };
        let names: &[&str] = if definition.auth_kind == AuthKind::Anthropic {
            &[
                "ANTHROPIC_AUTH_TOKEN",
                "ANTHROPIC_OAUTH_TOKEN",
                "ANTHROPIC_API_KEY",
            ]
        } else {
            definition.env_keys
        };
        names
            .iter()
            .filter(|name| environment_value(&self.environment, name).is_some())
            .map(|name| (*name).to_owned())
            .collect()
    }

    pub fn is_configured(&self, provider_id: &str) -> Result<bool, CatalogError> {
        Ok(self.resolve_auth(provider_id)?.is_some())
    }

    /// Returns configured providers in lexical ID order.
    pub fn configured_provider_ids(&self) -> Result<Vec<String>, CatalogError> {
        let mut configured = Vec::new();
        for provider_id in self.provider_ids() {
            if self.is_configured(&provider_id)? {
                configured.push(provider_id);
            }
        }
        Ok(configured)
    }

    /// Resolves `provider/model` directly, or resolves a bare model only when
    /// exactly one configured provider offers it.
    pub fn resolve_model(&self, reference: &str) -> Result<ResolvedModel, CatalogError> {
        if let Some((provider_id, model_id)) = reference.split_once('/') {
            let model =
                self.model(provider_id, model_id)
                    .ok_or_else(|| CatalogError::UnknownModel {
                        reference: reference.to_owned(),
                    })?;
            let auth = self.resolve_auth(provider_id)?.ok_or_else(|| {
                CatalogError::ProviderNotConfigured {
                    provider_id: provider_id.to_owned(),
                }
            })?;
            return Ok(self.resolved_model(model, auth));
        }

        let mut matches = Vec::new();
        for provider_id in self.configured_provider_ids()? {
            if self.model(&provider_id, reference).is_some() {
                matches.push(provider_id);
            }
        }
        match matches.as_slice() {
            [] => Err(CatalogError::NoConfiguredProvider {
                model_id: reference.to_owned(),
            }),
            [provider_id] => {
                let model = self.model(provider_id, reference).ok_or_else(|| {
                    CatalogError::NoConfiguredProvider {
                        model_id: reference.to_owned(),
                    }
                })?;
                let auth = self.resolve_auth(provider_id)?.ok_or_else(|| {
                    CatalogError::ProviderNotConfigured {
                        provider_id: provider_id.to_owned(),
                    }
                })?;
                Ok(self.resolved_model(model, auth))
            }
            _ => Err(CatalogError::AmbiguousModel {
                model_id: reference.to_owned(),
                provider_ids: matches,
            }),
        }
    }

    fn resolved_model(&self, model: llm::Model, auth: Auth) -> ResolvedModel {
        let mut effective_headers: AuthHeaders =
            resolve_headers(&model.headers, auth.environment(), &self.environment)
                .into_iter()
                .map(|(name, value)| (name, Some(value)))
                .collect();
        for (name, value) in auth.headers() {
            effective_headers.insert(name.clone(), value.clone());
        }
        ResolvedModel {
            model,
            auth,
            effective_headers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        fs, process,
        sync::{Arc, Mutex, atomic::AtomicU64},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    #[cfg(unix)]
    static COMMAND_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn test_environment(values: &[(&str, &str)]) -> EnvironmentLookup {
        let values: BTreeMap<String, String> = values
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        Arc::new(move |name| values.get(name).cloned())
    }

    fn test_catalog(values: &[(&str, &str)]) -> Catalog {
        let mut catalog = Catalog::with_environment_and_file_exists(
            None,
            test_environment(values),
            Arc::new(|_| false),
        )
        .expect("embedded catalog loads");
        let root = test_directory("unconfigured-aperture");
        catalog.aperture_paths = AperturePaths {
            config: root.join("aperture.json"),
            cache: root.join("aperture-cache.json"),
        };
        catalog
    }

    fn test_directory(label: &str) -> PathBuf {
        let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        env::temp_dir().join(format!(
            "goshcoder-catalog-{label}-{}-{sequence}",
            process::id()
        ))
    }

    fn aperture_catalog(
        values: &[(&str, &str)],
        configuration: &aperture::Config,
        cache: &aperture::Cache,
    ) -> (Catalog, PathBuf) {
        let root = test_directory("aperture");
        let paths = AperturePaths {
            config: root.join("extensions").join("aperture.json"),
            cache: root.join("extensions").join("aperture-cache.json"),
        };
        aperture::save_config(&paths.config, configuration).expect("save aperture config");
        aperture::save_cache(&paths.cache, cache).expect("save aperture cache");
        let mut catalog = test_catalog(values);
        catalog.aperture_paths = paths;
        (catalog, root)
    }

    #[test]
    fn aperture_dedicated_provider_uses_matching_cached_models_and_gateway_auth() {
        let configuration = aperture::Config {
            base_url: "http://aperture.test".to_owned(),
            onboarding_done: Some(true),
            dedicated: Some(aperture::DedicatedConfig {
                enabled: Some(true),
                ..aperture::DedicatedConfig::default()
            }),
            ..aperture::Config::default()
        };
        let model = llm::Model {
            id: "openai/test-model".to_owned(),
            name: "Test model".to_owned(),
            api: "openai-completions".to_owned(),
            provider: aperture::DEDICATED_PROVIDER_ID.to_owned(),
            base_url: "http://aperture.test/v1".to_owned(),
            ..llm::Model::default()
        };
        let cache = aperture::Cache {
            catalog_key: aperture::build_catalog_key(
                &aperture::gateway_url(&configuration.base_url),
                &configuration.resolve(),
            ),
            models: vec![aperture::CachedModel {
                model: model.clone(),
                raw_compat: None,
            }],
            ..aperture::Cache::default()
        };
        let (catalog, root) = aperture_catalog(&[], &configuration, &cache);

        let provider = catalog
            .provider(aperture::DEDICATED_PROVIDER_ID)
            .expect("dedicated provider");
        assert_eq!(provider.base_url, "http://aperture.test/v1");
        assert_eq!(provider.models(), vec![model]);
        let resolved = catalog
            .resolve_model("aperture/openai/test-model")
            .expect("resolve dedicated model");
        assert_eq!(resolved.auth().api_key(), Some("-"));
        assert_eq!(resolved.auth().source(), "aperture gateway");
        assert!(
            catalog
                .configured_provider_ids()
                .expect("configured providers")
                .contains(&aperture::DEDICATED_PROVIDER_ID.to_owned())
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aperture_catalog_state_reloads_after_a_gateway_sync() {
        let configuration = aperture::Config {
            base_url: "http://aperture.test".to_owned(),
            onboarding_done: Some(true),
            dedicated: Some(aperture::DedicatedConfig {
                enabled: Some(true),
                ..aperture::DedicatedConfig::default()
            }),
            ..aperture::Config::default()
        };
        let catalog_key = aperture::build_catalog_key(
            &aperture::gateway_url(&configuration.base_url),
            &configuration.resolve(),
        );
        let (catalog, root) = aperture_catalog(
            &[],
            &configuration,
            &aperture::Cache {
                catalog_key: catalog_key.clone(),
                ..aperture::Cache::default()
            },
        );
        assert!(
            catalog
                .provider(aperture::DEDICATED_PROVIDER_ID)
                .expect("dedicated provider")
                .models()
                .is_empty()
        );

        let cache = aperture::Cache {
            catalog_key,
            models: vec![aperture::CachedModel {
                model: llm::Model {
                    id: "openai/fresh-model".to_owned(),
                    name: "Fresh model".to_owned(),
                    api: "openai-completions".to_owned(),
                    provider: aperture::DEDICATED_PROVIDER_ID.to_owned(),
                    base_url: "http://aperture.test/v1".to_owned(),
                    ..llm::Model::default()
                },
                raw_compat: None,
            }],
            ..aperture::Cache::default()
        };
        aperture::save_cache(root.join("extensions").join("aperture-cache.json"), &cache)
            .expect("refresh Aperture cache");

        catalog.reload_aperture_state();
        assert_eq!(
            catalog
                .provider(aperture::DEDICATED_PROVIDER_ID)
                .expect("reloaded dedicated provider")
                .models()
                .into_iter()
                .map(|model| model.id)
                .collect::<Vec<_>>(),
            vec!["openai/fresh-model"]
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aperture_proxy_routes_filter_models_and_supply_gateway_auth() {
        let template = test_catalog(&[])
            .provider("openai")
            .expect("OpenAI provider")
            .models()
            .into_iter()
            .next()
            .expect("OpenAI model");
        let configuration = aperture::Config {
            base_url: "http://aperture.test".to_owned(),
            onboarding_done: Some(true),
            proxy: Some(aperture::ProxyConfig {
                enabled: Some(true),
                upstream_providers: Some(vec![aperture::ProxiedProviderConfig {
                    id: "openai".to_owned(),
                    keep_gateway_models_only: true,
                    ..aperture::ProxiedProviderConfig::default()
                }]),
            }),
            ..aperture::Config::default()
        };
        let cache = aperture::Cache {
            gateway: vec![aperture::GatewaySnapshot {
                id: "openai".to_owned(),
                models: vec![template.id.clone()],
                ..aperture::GatewaySnapshot::default()
            }],
            ..aperture::Cache::default()
        };
        let (catalog, root) = aperture_catalog(&[], &configuration, &cache);

        let route = catalog
            .aperture_state()
            .routes
            .get("openai")
            .cloned()
            .expect("OpenAI proxy route");
        let provider = catalog.provider("openai").expect("proxied OpenAI provider");
        assert_eq!(provider.base_url, route.base_url);
        assert_eq!(provider.models().len(), 1);
        assert_eq!(provider.models()[0].id, template.id);
        let resolved = catalog
            .resolve_model(&format!("openai/{}", template.id))
            .expect("resolve proxied model");
        assert_eq!(resolved.model.base_url, route.base_url);
        assert_eq!(resolved.auth().api_key(), Some("-"));
        assert_eq!(resolved.auth().source(), "aperture proxy");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn aperture_proxy_passthrough_retains_native_authentication() {
        let template = test_catalog(&[])
            .provider("openai")
            .expect("OpenAI provider")
            .models()
            .into_iter()
            .next()
            .expect("OpenAI model");
        let configuration = aperture::Config {
            base_url: "http://aperture.test".to_owned(),
            onboarding_done: Some(true),
            proxy: Some(aperture::ProxyConfig {
                enabled: Some(true),
                upstream_providers: Some(vec![aperture::ProxiedProviderConfig {
                    id: "openai".to_owned(),
                    ..aperture::ProxiedProviderConfig::default()
                }]),
            }),
            ..aperture::Config::default()
        };
        let cache = aperture::Cache {
            gateway: vec![aperture::GatewaySnapshot {
                id: "openai".to_owned(),
                models: vec![template.id],
                requires_client_auth: true,
                ..aperture::GatewaySnapshot::default()
            }],
            ..aperture::Cache::default()
        };
        let (catalog, root) =
            aperture_catalog(&[("OPENAI_API_KEY", "native-key")], &configuration, &cache);

        let auth = catalog
            .resolve_auth("openai")
            .expect("resolve OpenAI auth")
            .expect("native OpenAI auth");
        assert_eq!(auth.api_key(), Some("native-key"));
        assert_eq!(auth.source(), "OPENAI_API_KEY");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn merges_extra_models_and_overrides_deterministically() {
        let generated = json!({
            "example": {
                "model": {
                    "id": "model",
                    "name": "generated",
                    "api": "openai-completions",
                    "provider": "example",
                    "baseUrl": "https://example.test/v1",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": {"input": 1},
                    "contextWindow": 1,
                    "maxTokens": 1,
                    "compat": {"protocolSpecific": true}
                }
            }
        })
        .to_string();
        let extras = json!({
            "example": {
                "model": {
                    "id": "model",
                    "name": "must-not-shadow-generated",
                    "api": "openai-completions",
                    "provider": "example",
                    "baseUrl": "https://example.test/v1",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": {},
                    "contextWindow": 1,
                    "maxTokens": 1
                },
                "extra": {
                    "id": "extra",
                    "name": "extra model",
                    "api": "openai-completions",
                    "provider": "example",
                    "baseUrl": "https://example.test/v1",
                    "reasoning": false,
                    "input": ["text"],
                    "cost": {},
                    "contextWindow": 2,
                    "maxTokens": 2
                }
            }
        })
        .to_string();
        let overrides = json!({
            "example": {
                "model": {
                    "name": "overridden",
                    "cost": {"input": 2, "output": 3}
                }
            }
        })
        .to_string();

        let merged =
            merge_catalog_documents(&generated, &extras, &overrides).expect("merge catalog");
        let models = merged.models.get("example").expect("provider");
        assert_eq!(
            models.keys().cloned().collect::<Vec<_>>(),
            vec!["extra", "model"]
        );
        let model = models.get("model").expect("generated model");
        assert_eq!(model.name, "overridden");
        assert_eq!(model.cost.rates.input, 2.0);
        assert_eq!(model.cost.rates.output, 3.0);
        assert_eq!(
            merged
                .raw_compat
                .get("example")
                .and_then(|models| models.get("model")),
            Some(&json!({"protocolSpecific": true}))
        );

        let forbidden = json!({"example": {"model": {"id": "replacement"}}}).to_string();
        assert!(merge_catalog_documents(&generated, &extras, &forbidden).is_err());
    }

    #[test]
    fn embedded_catalog_includes_overrides_and_defensive_model_copies() {
        let catalog = test_catalog(&[]);
        let sol = catalog
            .model("openai", "gpt-5.6-sol")
            .expect("overridden OpenAI model");
        assert_eq!(sol.context_window, 1_050_000);
        assert_eq!(sol.cost.rates.input, 4.0);

        let provider = catalog.provider("openai").expect("OpenAI provider");
        let mut models = provider.models();
        let original_name = models.first().expect("at least one model").name.clone();
        models.first_mut().expect("at least one model").name = "tampered".to_owned();
        let fresh = catalog
            .provider("openai")
            .expect("OpenAI provider")
            .models()
            .into_iter()
            .next()
            .expect("at least one model");
        assert_eq!(fresh.name, original_name);

        assert!(
            catalog.model("anthropic", "claude-mythos-5").is_some(),
            "catalog_extra.json model should be available"
        );
    }

    #[test]
    fn mistral_api_key_configures_native_conversations_models() {
        let catalog = test_catalog(&[("MISTRAL_API_KEY", "test-mistral-key")]);
        assert!(
            catalog
                .configured_provider_ids()
                .expect("configured providers")
                .contains(&"mistral".to_owned())
        );

        let model = catalog
            .provider("mistral")
            .expect("Mistral provider")
            .models()
            .into_iter()
            .find(|model| model.api == "mistral-conversations")
            .expect("Mistral Conversations model in embedded catalog");
        let resolved = catalog
            .resolve_model(&format!("mistral/{}", model.id))
            .expect("resolve configured Mistral model");
        assert_eq!(resolved.model.api, "mistral-conversations");
        assert_eq!(resolved.auth().api_key(), Some("test-mistral-key"));
    }

    #[test]
    fn config_values_expand_templates_without_caching_environment_reads() {
        let lookup =
            test_environment(&[("LEFT", "left"), ("RIGHT", "right"), ("SCOPED", "process")]);
        let scoped = BTreeMap::from([("SCOPED".to_owned(), "credential".to_owned())]);

        assert_eq!(
            resolve_config_value("${LEFT}_$RIGHT", &BTreeMap::new(), &lookup).as_deref(),
            Some("left_right")
        );
        assert_eq!(
            resolve_config_value("$$LEFT-$!literal", &BTreeMap::new(), &lookup).as_deref(),
            Some("$LEFT-!literal")
        );
        assert_eq!(
            resolve_config_value("$SCOPED", &scoped, &lookup).as_deref(),
            Some("credential")
        );
        assert_eq!(
            resolve_config_value("$MISSING", &BTreeMap::new(), &lookup),
            None
        );
        assert_eq!(
            config_value_environment_name("$LEFT").as_deref(),
            Some("LEFT")
        );
        assert_eq!(
            config_value_environment_names("${LEFT}-$RIGHT-$LEFT"),
            vec!["LEFT", "RIGHT"]
        );
        assert_eq!(
            missing_config_value_environment_names("$LEFT-$MISSING", &BTreeMap::new(), &lookup),
            vec!["MISSING"]
        );
        assert!(is_config_value_configured(
            "!this-command-is-not-run-during-configuration-check",
            &BTreeMap::new(),
            &lookup
        ));
    }

    #[cfg(unix)]
    #[test]
    fn command_config_values_are_trimmed_and_cached() {
        let _test_lock = lock_unpoisoned(&COMMAND_TEST_LOCK);
        clear_config_value_cache();
        let directory = test_directory("command-cache");
        fs::create_dir_all(&directory).expect("create temp directory");
        let counter = directory.join("counter");
        fs::write(&counter, "0").expect("create counter");
        let quoted = format!("'{}'", counter.to_string_lossy().replace('\'', "'\"'\"'"));
        let command = format!(
            "!count=$(cat {quoted}); echo $((count + 1)) > {quoted}; printf '  cached value  '"
        );
        let empty = BTreeMap::new();
        let lookup = test_environment(&[]);

        assert_eq!(
            resolve_config_value(&command, &empty, &lookup).as_deref(),
            Some("cached value")
        );
        assert_eq!(
            resolve_config_value(&command, &empty, &lookup).as_deref(),
            Some("cached value")
        );
        assert_eq!(
            fs::read_to_string(&counter)
                .expect("read counter")
                .trim()
                .to_owned(),
            "1"
        );

        clear_config_value_cache();
        fs::remove_dir_all(directory).expect("remove temp directory");
    }

    #[test]
    fn credential_store_preserves_unknown_fields_and_secures_auth_file() {
        let directory = test_directory("auth-store");
        fs::create_dir_all(&directory).expect("create temp directory");
        let path = directory.join("auth.json");
        let store = CredentialStore::file(&path);
        let mut credential = Credential::api_key("$TEST_STORED_KEY");
        credential.set_environment("REGION", "test-region");
        credential
            .set_extra("accountId", json!("test-account"))
            .expect("unknown field is accepted");
        store.put("example", credential).expect("write credential");

        let raw = store
            .read_raw("example")
            .expect("read raw credential")
            .expect("credential is present");
        assert_eq!(raw.extra_string("accountId"), Some("test-account"));
        assert_eq!(raw.key(), "$TEST_STORED_KEY");

        let resolved = store
            .read_with_environment(
                "example",
                &test_environment(&[("TEST_STORED_KEY", "resolved")]),
            )
            .expect("read resolved credential")
            .expect("credential is present");
        assert_eq!(resolved.key(), "resolved");
        assert_eq!(resolved.extra_string("accountId"), Some("test-account"));

        let reread = CredentialStore::file(&path)
            .read_raw("example")
            .expect("read persisted credential")
            .expect("credential is present");
        assert_eq!(reread.extra_string("accountId"), Some("test-account"));

        // Unknown fields from another auth.json producer must survive a
        // deserialize/re-serialize cycle, not only fields set through this API.
        fs::write(
            &path,
            r#"{"future-provider":{"type":"api_key","key":"$TEST_STORED_KEY","future":{"nested":true}}}"#,
        )
        .expect("write externally produced auth file");
        let future_store = CredentialStore::file(&path);
        let future = future_store
            .read_raw("future-provider")
            .expect("read future credential")
            .expect("future credential is present");
        assert_eq!(future.extra("future"), Some(&json!({"nested": true})));
        future_store
            .modify("future-provider", Ok)
            .expect("re-serialize future credential");
        let future = future_store
            .read_raw("future-provider")
            .expect("re-read future credential")
            .expect("future credential is present");
        assert_eq!(future.extra("future"), Some(&json!({"nested": true})));

        #[cfg(unix)]
        {
            assert_eq!(
                fs::metadata(&path)
                    .expect("stat auth.json")
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(directory).expect("remove temp directory");
    }

    #[test]
    fn auth_resolution_handles_provider_specific_non_oauth_strategies() {
        let anthropic = test_catalog(&[("ANTHROPIC_AUTH_TOKEN", "anthropic-token")]);
        let auth = anthropic
            .resolve_auth("anthropic")
            .expect("resolve auth")
            .expect("Anthropic auth");
        assert_eq!(auth.source(), "ANTHROPIC_AUTH_TOKEN");
        assert_eq!(
            auth.header("Authorization"),
            Some(Some("Bearer anthropic-token"))
        );
        assert_eq!(auth.header("x-api-key"), Some(None));

        let cloudflare = test_catalog(&[
            ("CLOUDFLARE_API_KEY", "cloudflare-key"),
            ("CLOUDFLARE_ACCOUNT_ID", "account"),
            ("CLOUDFLARE_GATEWAY_ID", "gateway"),
        ]);
        let auth = cloudflare
            .resolve_auth("cloudflare-ai-gateway")
            .expect("resolve auth")
            .expect("Cloudflare gateway auth");
        assert_eq!(
            auth.environment()
                .get("CLOUDFLARE_ACCOUNT_ID")
                .map(String::as_str),
            Some("account")
        );
        assert_eq!(
            auth.environment()
                .get("CLOUDFLARE_GATEWAY_ID")
                .map(String::as_str),
            Some("gateway")
        );
        assert_eq!(
            auth.header("cf-aig-authorization"),
            Some(Some("Bearer cloudflare-key"))
        );
        assert_eq!(auth.header("Authorization"), Some(None));

        let bedrock = test_catalog(&[("AWS_PROFILE", "default")]);
        assert!(
            bedrock
                .resolve_auth("amazon-bedrock")
                .expect("resolve auth")
                .expect("ambient Bedrock auth")
                .is_ambient()
        );

        let aperture_root = test_directory("vertex-aperture");
        let vertex = Catalog::with_environment_and_file_exists(
            None,
            test_environment(&[
                ("GOOGLE_APPLICATION_CREDENTIALS", "/test/adc.json"),
                ("GOOGLE_CLOUD_PROJECT", "project"),
                ("GOOGLE_CLOUD_LOCATION", "us-central1"),
            ]),
            Arc::new(|path| path == Path::new("/test/adc.json")),
        )
        .expect("catalog")
        .with_aperture_paths(
            aperture_root.join("aperture.json"),
            aperture_root.join("aperture-cache.json"),
        );
        assert!(
            vertex
                .resolve_auth("google-vertex")
                .expect("resolve auth")
                .expect("ambient Vertex auth")
                .is_ambient()
        );

        let meta = test_catalog(&[("META_API_KEY", "meta-key")]);
        let auth = meta
            .resolve_auth("meta")
            .expect("resolve auth")
            .expect("Meta auth");
        assert_eq!(auth.api_key(), None);
        assert_eq!(auth.header("Authorization"), Some(Some("Bearer meta-key")));
    }

    #[test]
    fn vertex_prefetched_token_configures_and_resolves_models() {
        let catalog = test_catalog(&[
            ("GOOGLE_OAUTH_ACCESS_TOKEN", "ya29.prefetched"),
            ("GOOGLE_CLOUD_PROJECT", "test-project"),
            ("GOOGLE_CLOUD_LOCATION", "us-central1"),
        ]);
        let auth = catalog
            .resolve_auth("google-vertex")
            .expect("resolve Vertex auth")
            .expect("prefetched token configures Vertex");
        assert!(auth.is_ambient());
        assert_eq!(auth.source(), "GOOGLE_OAUTH_ACCESS_TOKEN");
        assert_eq!(
            auth.environment()
                .get("GOOGLE_OAUTH_ACCESS_TOKEN")
                .map(String::as_str),
            Some("ya29.prefetched")
        );
        assert_eq!(
            catalog
                .configured_provider_ids()
                .expect("configured providers"),
            vec!["google-vertex"]
        );

        let model = catalog
            .provider("google-vertex")
            .expect("Vertex provider")
            .models()
            .into_iter()
            .next()
            .expect("Vertex model");
        let resolved = catalog
            .resolve_model(&format!("google-vertex/{}", model.id))
            .expect("resolve configured Vertex model");
        assert_eq!(
            resolved
                .auth()
                .environment()
                .get("GOOGLE_CLOUD_PROJECT")
                .map(String::as_str),
            Some("test-project")
        );
    }

    #[test]
    fn stored_oauth_is_resolved_without_falling_back_to_ambient_auth() {
        let store = Arc::new(CredentialStore::in_memory());
        store
            .put(
                "anthropic",
                Credential::oauth("access", "refresh", i64::MAX),
            )
            .expect("store OAuth credential");
        let aperture_root = test_directory("stored-oauth-aperture");
        let catalog = Catalog::with_environment_and_file_exists(
            Some(store),
            test_environment(&[("ANTHROPIC_API_KEY", "ambient-key")]),
            Arc::new(|_| false),
        )
        .expect("catalog")
        .with_aperture_paths(
            aperture_root.join("aperture.json"),
            aperture_root.join("aperture-cache.json"),
        );
        let auth = catalog
            .resolve_auth("anthropic")
            .expect("resolve auth")
            .expect("stored OAuth auth");
        assert_eq!(auth.source(), "OAuth");
        assert_eq!(auth.api_key(), Some("access"));
    }

    #[test]
    fn configured_providers_and_model_references_resolve_deterministically() {
        let catalog = test_catalog(&[
            ("OPENAI_API_KEY", "openai-key"),
            ("GROQ_API_KEY", "groq-key"),
        ]);
        assert_eq!(
            catalog
                .configured_provider_ids()
                .expect("configured providers"),
            vec!["groq", "openai"]
        );

        let model_id = catalog
            .provider("openai")
            .expect("OpenAI provider")
            .models()
            .into_iter()
            .next()
            .expect("OpenAI model")
            .id;
        let qualified = catalog
            .resolve_model(&format!("openai/{model_id}"))
            .expect("qualified model");
        assert_eq!(qualified.model.id, model_id);
        assert_eq!(qualified.auth().source(), "OPENAI_API_KEY");

        let bare = catalog.resolve_model(&model_id).expect("bare model");
        assert_eq!(bare.model.provider, "openai");

        match catalog.resolve_model("openai/not-a-model") {
            Err(CatalogError::UnknownModel { .. }) => {}
            _ => panic!("unknown qualified model should fail"),
        }
    }
}
