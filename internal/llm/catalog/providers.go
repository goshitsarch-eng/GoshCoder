package catalog

import (
	"encoding/json"

	"goshcoder/internal/llm"
)

// AuthKind selects the credential resolution strategy of a provider
// (pi's per-provider ApiKeyAuth implementations).
type AuthKind int

const (
	// AuthEnvKey is the standard api-key auth (pi's envApiKeyAuth): a stored
	// credential key wins, otherwise the first set env var in EnvKeys resolves.
	AuthEnvKey AuthKind = iota
	// AuthAnthropic resolves ANTHROPIC_AUTH_TOKEN as an Authorization: Bearer
	// header before falling back to ANTHROPIC_OAUTH_TOKEN / ANTHROPIC_API_KEY.
	AuthAnthropic
	// AuthCloudflareWorkersAI requires CLOUDFLARE_API_KEY plus
	// CLOUDFLARE_ACCOUNT_ID (carried as provider env).
	AuthCloudflareWorkersAI
	// AuthCloudflareAIGateway additionally requires CLOUDFLARE_GATEWAY_ID and
	// authenticates via the cf-aig-authorization header, suppressing the
	// default Authorization/x-api-key headers.
	AuthCloudflareAIGateway
	// AuthAmbientBedrock reports configured when any AWS credential source is
	// present (profile, IAM keys, bearer token, ECS/IRSA); the wire protocol
	// resolves the actual credentials.
	AuthAmbientBedrock
	// AuthVertex accepts GOOGLE_CLOUD_API_KEY, or Application Default
	// Credentials plus project and location env vars.
	AuthVertex
	// AuthOAuthOnly has no api-key auth; only a stored OAuth credential
	// configures the provider (pi: openai-codex).
	AuthOAuthOnly
	// AuthMetaBearer sends the Meta Model API key as an Authorization bearer
	// token. Meta serves the Anthropic Messages shape but authenticates with
	// Authorization rather than x-api-key, so the key must travel as a header
	// instead of as the protocol's api key.
	AuthMetaBearer
)

// Provider is the runtime unit of the catalog: static metadata, auth
// semantics, and its current model list (port of pi's Provider interface,
// reduced to the static/env-credential slice).
type Provider struct {
	ID      string
	Name    string
	BaseURL string
	// KeyName is the display name of the api-key credential, e.g.
	// "Anthropic API key".
	KeyName string
	// EnvKeys are the environment variables that can yield an API key, in
	// resolution order (first set wins). Mirrors env-api-keys.ts.
	EnvKeys []string
	// Kind selects the resolution strategy.
	Kind AuthKind
	// OAuth reports whether the provider supports OAuth. Login and refresh are
	// implemented for Anthropic, OpenAI Codex, Kimi Coding, xAI, and Meta.
	OAuth bool

	// models is the provider's built-in model list, keyed by model id in
	// rawCompat. Custom models.json loading is not ported.
	models    []llm.Model
	rawCompat map[string]json.RawMessage
}

// Models returns the provider's models as defensive copies.
func (p *Provider) Models() []llm.Model {
	out := make([]llm.Model, len(p.models))
	for i := range p.models {
		out[i] = *cloneModel(&p.models[i])
	}
	return out
}

// Model returns a copy of the named model, or nil.
func (p *Provider) Model(id string) *llm.Model {
	for i := range p.models {
		if p.models[i].ID == id {
			return cloneModel(&p.models[i])
		}
	}
	return nil
}

// RawCompat returns the raw compat JSON of one of the provider's models,
// including keys not represented in llm.OpenAICompletionsCompat.
func (p *Provider) RawCompat(modelID string) (json.RawMessage, bool) {
	c, ok := p.rawCompat[modelID]
	return c, ok
}

// builtinProviderConfig describes one provider entry from pi's providers/*.ts
// files (id, name, baseUrl, env key names, auth kind). Model data comes from
// the embedded catalog, not from this table.
type builtinProviderConfig struct {
	name    string
	baseURL string
	keyName string
	envKeys []string
	kind    AuthKind
	oauth   bool
}

// builtinProviderConfigs mirrors reference/pi/packages/ai/src/providers/*.ts
// and env-api-keys.ts. Providers whose base URL is model- or credential-specific
// (azure, bedrock, cloudflare, vertex, opencode) leave baseURL empty; the
// catalog models carry their own baseUrl.
var builtinProviderConfigs = map[string]builtinProviderConfig{
	"amazon-bedrock": {
		name: "Amazon Bedrock", keyName: "AWS credentials or bearer token",
		kind: AuthAmbientBedrock,
	},
	"ant-ling": {
		name: "Ant Ling", baseURL: "https://api.ant-ling.com/v1",
		keyName: "Ant Ling API key", envKeys: []string{"ANT_LING_API_KEY"},
	},
	"anthropic": {
		name: "Anthropic", baseURL: "https://api.anthropic.com",
		keyName: "Anthropic API key", kind: AuthAnthropic, oauth: true,
	},
	"azure-openai-responses": {
		name:    "Azure OpenAI",
		keyName: "Azure OpenAI API key", envKeys: []string{"AZURE_OPENAI_API_KEY"},
	},
	"baseten": {
		name: "Baseten", baseURL: "https://inference.baseten.co/v1",
		keyName: "Baseten API key", envKeys: []string{"BASETEN_API_KEY"},
	},
	"cerebras": {
		name: "Cerebras", baseURL: "https://api.cerebras.ai/v1",
		keyName: "Cerebras API key", envKeys: []string{"CEREBRAS_API_KEY"},
	},
	"cloudflare-ai-gateway": {
		name:    "Cloudflare AI Gateway",
		keyName: "Cloudflare API key", kind: AuthCloudflareAIGateway,
	},
	"cloudflare-workers-ai": {
		name:    "Cloudflare Workers AI",
		keyName: "Cloudflare API key", kind: AuthCloudflareWorkersAI,
	},
	"deepseek": {
		name: "DeepSeek", baseURL: "https://api.deepseek.com",
		keyName: "DeepSeek API key", envKeys: []string{"DEEPSEEK_API_KEY"},
	},
	"fireworks": {
		name: "Fireworks", baseURL: "https://api.fireworks.ai/inference",
		keyName: "Fireworks API key", envKeys: []string{"FIREWORKS_API_KEY"},
	},
	"github-copilot": {
		name: "GitHub Copilot", baseURL: "https://api.individual.githubcopilot.com",
		keyName: "GitHub Copilot token", envKeys: []string{"COPILOT_GITHUB_TOKEN"}, oauth: true,
	},
	"google": {
		name: "Google", baseURL: "https://generativelanguage.googleapis.com/v1beta",
		keyName: "Gemini API key", envKeys: []string{"GEMINI_API_KEY"},
	},
	"google-vertex": {
		name:    "Google Vertex AI",
		keyName: "Google Cloud credentials", kind: AuthVertex,
	},
	"groq": {
		name: "Groq", baseURL: "https://api.groq.com/openai/v1",
		keyName: "Groq API key", envKeys: []string{"GROQ_API_KEY"},
	},
	"huggingface": {
		name: "Hugging Face", baseURL: "https://router.huggingface.co/v1",
		keyName: "Hugging Face token", envKeys: []string{"HF_TOKEN"},
	},
	"kimi-coding": {
		name: "Kimi For Coding", baseURL: "https://api.kimi.com/coding",
		keyName: "Kimi API key", envKeys: []string{"KIMI_API_KEY"}, oauth: true,
	},
	"meta": {
		name: "Meta", baseURL: "https://api.meta.ai",
		keyName: "Meta Model API key", envKeys: []string{"META_API_KEY"},
		kind: AuthMetaBearer, oauth: true,
	},
	"minimax": {
		name: "MiniMax", baseURL: "https://api.minimax.io/anthropic",
		keyName: "MiniMax API key", envKeys: []string{"MINIMAX_API_KEY"},
	},
	"minimax-cn": {
		name: "MiniMax CN", baseURL: "https://api.minimaxi.com/anthropic",
		keyName: "MiniMax CN API key", envKeys: []string{"MINIMAX_CN_API_KEY"},
	},
	"mistral": {
		name: "Mistral", baseURL: "https://api.mistral.ai",
		keyName: "Mistral API key", envKeys: []string{"MISTRAL_API_KEY"},
	},
	"moonshotai": {
		name: "Moonshot AI", baseURL: "https://api.moonshot.ai/v1",
		keyName: "Moonshot AI API key", envKeys: []string{"MOONSHOT_API_KEY"},
	},
	"moonshotai-cn": {
		name: "Moonshot AI CN", baseURL: "https://api.moonshot.cn/v1",
		keyName: "Moonshot AI API key", envKeys: []string{"MOONSHOT_API_KEY"},
	},
	"nvidia": {
		name: "NVIDIA", baseURL: "https://integrate.api.nvidia.com/v1",
		keyName: "NVIDIA API key", envKeys: []string{"NVIDIA_API_KEY"},
	},
	"openai": {
		name: "OpenAI", baseURL: "https://api.openai.com/v1",
		keyName: "OpenAI API key", envKeys: []string{"OPENAI_API_KEY"},
	},
	"openai-codex": {
		name: "OpenAI Codex", baseURL: "https://chatgpt.com/backend-api",
		kind: AuthOAuthOnly, oauth: true,
	},
	"opencode": {
		name:    "OpenCode Zen",
		keyName: "OpenCode API key", envKeys: []string{"OPENCODE_API_KEY"},
	},
	"opencode-go": {
		name:    "OpenCode Go",
		keyName: "OpenCode API key", envKeys: []string{"OPENCODE_API_KEY"},
	},
	"openrouter": {
		name: "OpenRouter", baseURL: "https://openrouter.ai/api/v1",
		keyName: "OpenRouter API key", envKeys: []string{"OPENROUTER_API_KEY"}, oauth: true,
	},
	"omni": {
		name: "OmniRoute", baseURL: "http://127.0.0.1:20128/v1",
		keyName: "OmniRoute API key", envKeys: []string{"OMNIROUTE_API_KEY"},
	},
	"qwen-token-plan": {
		name: "Qwen Token Plan", baseURL: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
		keyName: "Qwen Token Plan API key", envKeys: []string{"QWEN_TOKEN_PLAN_API_KEY"},
	},
	"qwen-token-plan-cn": {
		name: "Qwen Token Plan CN", baseURL: "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
		keyName: "Qwen Token Plan CN API key", envKeys: []string{"QWEN_TOKEN_PLAN_CN_API_KEY"},
	},
	"qwen-token-plan-individual": {
		name: "Qwen Token Plan Individual", baseURL: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
		keyName: "Qwen Token Plan Individual API key", envKeys: []string{"QWEN_TOKEN_PLAN_API_KEY"},
	},
	// radius is a purely dynamic provider in pi: no static catalog entry.
	"radius": {
		name:    "Radius",
		keyName: "Radius API key", envKeys: []string{"RADIUS_API_KEY"}, oauth: true,
	},
	"together": {
		name: "Together", baseURL: "https://api.together.ai/v1",
		keyName: "Together API key", envKeys: []string{"TOGETHER_API_KEY"},
	},
	"vercel-ai-gateway": {
		name: "Vercel AI Gateway", baseURL: "https://ai-gateway.vercel.sh",
		keyName: "Vercel AI Gateway API key", envKeys: []string{"AI_GATEWAY_API_KEY"},
	},
	"xai": {
		name: "xAI", baseURL: "https://api.x.ai/v1",
		keyName: "xAI API key", envKeys: []string{"XAI_API_KEY"}, oauth: true,
	},
	"xiaomi": {
		name: "Xiaomi", baseURL: "https://api.xiaomimimo.com/v1",
		keyName: "Xiaomi API key", envKeys: []string{"XIAOMI_API_KEY"},
	},
	"xiaomi-token-plan-ams": {
		name: "Xiaomi Token Plan AMS", baseURL: "https://token-plan-ams.xiaomimimo.com/v1",
		keyName: "Xiaomi Token Plan AMS API key", envKeys: []string{"XIAOMI_TOKEN_PLAN_AMS_API_KEY"},
	},
	"xiaomi-token-plan-cn": {
		name: "Xiaomi Token Plan CN", baseURL: "https://token-plan-cn.xiaomimimo.com/v1",
		keyName: "Xiaomi Token Plan CN API key", envKeys: []string{"XIAOMI_TOKEN_PLAN_CN_API_KEY"},
	},
	"xiaomi-token-plan-sgp": {
		name: "Xiaomi Token Plan SGP", baseURL: "https://token-plan-sgp.xiaomimimo.com/v1",
		keyName: "Xiaomi Token Plan SGP API key", envKeys: []string{"XIAOMI_TOKEN_PLAN_SGP_API_KEY"},
	},
	"zai": {
		name: "Z.AI", baseURL: "https://api.z.ai/api/coding/paas/v4",
		keyName: "Z.AI API key", envKeys: []string{"ZAI_API_KEY"},
	},
	"zai-coding-cn": {
		name: "Z.AI Coding CN", baseURL: "https://open.bigmodel.cn/api/coding/paas/v4",
		keyName: "Z.AI Coding CN API key", envKeys: []string{"ZAI_CODING_CN_API_KEY"},
	},
}

// Anthropic env vars, in resolution order after a stored credential
// (anthropic.ts / env-api-keys.ts).
const (
	anthropicAuthTokenEnv  = "ANTHROPIC_AUTH_TOKEN"
	anthropicOAuthTokenEnv = "ANTHROPIC_OAUTH_TOKEN"
	anthropicAPIKeyEnv     = "ANTHROPIC_API_KEY"
)

// Cloudflare provider env names (cloudflare-auth.ts).
const (
	cloudflareAPIKeyEnv    = "CLOUDFLARE_API_KEY"
	cloudflareAccountIDEnv = "CLOUDFLARE_ACCOUNT_ID"
	cloudflareGatewayIDEnv = "CLOUDFLARE_GATEWAY_ID"
)
