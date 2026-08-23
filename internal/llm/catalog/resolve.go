package catalog

// Provider lookup and auth resolution (port of the static/env-credential slice
// of reference/pi/packages/ai/src/auth/resolve.ts, env-api-keys.ts, and the
// per-provider ApiKeyAuth implementations in providers/*.ts).
//
// Resolution order matches pi: an explicit override key wins, then a stored
// credential owns the provider, and ambient environment/file credentials are
// consulted only when nothing is stored.
//
// OAuth login and refresh are supported for Anthropic, OpenAI Codex, Kimi
// Coding, xAI, and Meta. Other providers use their static or ambient
// credential strategies.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	goshconfig "goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/omniroute"
)

// authenticatedSentinel is pi's marker for ambient credentials that are
// resolved later by the wire protocol (AWS profiles, Google ADC) rather than
// being a literal key.
const authenticatedSentinel = "<authenticated>"

// oauthMinimumValidity is how much remaining validity a stored OAuth token
// needs to be considered usable (pi's DEFAULT_OAUTH_MINIMUM_VALIDITY_MS).
const oauthMinimumValidity = 5 * time.Minute

// Catalog is a provider collection backed by the embedded model data plus an
// optional credential store (pi's Models collection, reduced).
type Catalog struct {
	credentials *CredentialStore
	getenv      Getenv
	// fileExists is injectable so ADC detection can be tested.
	fileExists func(string) bool
	// oauthCtx bounds implicit OAuth refreshes triggered by auth resolution.
	oauthCtx context.Context
	// oauthTimeout caps one implicit refresh; see SetOAuthTimeout.
	oauthTimeout time.Duration
	// oauthErrors records the last refresh failure per provider, so callers can
	// distinguish "no credential" from "your login expired".
	oauthMu     sync.Mutex
	oauthErrors map[string]error
}

// NewCatalog returns a catalog over the builtin model data. credentials may be
// nil, in which case only ambient environment credentials resolve.
func NewCatalog(credentials *CredentialStore) *Catalog {
	return &Catalog{
		credentials: credentials,
		getenv:      os.Getenv,
		fileExists:  fileExists,
	}
}

// SetOAuthContext bounds implicit OAuth refreshes, so a cancelled run does not
// leave a token exchange in flight.
func (c *Catalog) SetOAuthContext(ctx context.Context) { c.oauthCtx = ctx }

// SetOAuthTimeout caps a single implicit refresh.
//
// Resolution can trigger a token exchange, which is network I/O, and several
// paths that merely enumerate providers -- the startup model default, the model
// and login pickers, `goshcoder providers` -- resolve every one of them. With
// no deadline that runs on context.Background(), and a provider that retries
// (Kimi retries three times behind a 30 s timeout, plus backoff) can block the
// caller for over two minutes with nothing on screen. A caller that has a real
// context of its own -- a model request, an interactive login -- installs it
// with SetOAuthContext and is bounded by that instead.
func (c *Catalog) SetOAuthTimeout(d time.Duration) { c.oauthTimeout = d }

// Credentials returns the backing credential store, which login flows write to.
func (c *Catalog) Credentials() *CredentialStore { return c.credentials }

// SetGetenv overrides the environment lookup (tests).
func (c *Catalog) SetGetenv(getenv Getenv) { c.getenv = getenv }

// SetFileExists overrides the file existence check (tests).
func (c *Catalog) SetFileExists(fn func(string) bool) { c.fileExists = fn }

func (c *Catalog) env(name string) string {
	if c.getenv == nil {
		return os.Getenv(name)
	}
	return c.getenv(name)
}

// ProviderIDs returns the sorted ids of every known provider.
func (c *Catalog) ProviderIDs() []string {
	ids := make([]string, 0, len(builtinProviderConfigs))
	for id := range builtinProviderConfigs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Provider returns the provider with the given id, or nil when unknown. The
// returned value carries its own copy of the model list.
func (c *Catalog) Provider(id string) *Provider {
	config, ok := builtinProviderConfigs[id]
	if !ok {
		return nil
	}
	provider := &Provider{
		ID:      id,
		Name:    config.name,
		BaseURL: config.baseURL,
		KeyName: config.keyName,
		EnvKeys: append([]string(nil), config.envKeys...),
		Kind:    config.kind,
		OAuth:   config.oauth,
	}

	models := builtin.models[id]
	ids := make([]string, 0, len(models))
	for modelID := range models {
		ids = append(ids, modelID)
	}
	sort.Strings(ids)
	provider.models = make([]llm.Model, 0, len(ids))
	for _, modelID := range ids {
		provider.models = append(provider.models, *cloneModel(models[modelID]))
	}
	provider.rawCompat = builtin.rawCompat[id]
	if id == "omni" {
		if configured, err := omniroute.Load(goshconfig.OmniRoutePath()); err == nil {
			provider.BaseURL = configured.APIBaseURL()
			provider.models = make([]llm.Model, 0, len(configured.Models))
			for _, dynamic := range configured.Models {
				provider.models = append(provider.models, dynamic.LLMModel(configured))
			}
		}
	}
	return provider
}

// Providers returns every provider, ordered by id.
func (c *Catalog) Providers() []*Provider {
	ids := c.ProviderIDs()
	out := make([]*Provider, 0, len(ids))
	for _, id := range ids {
		if provider := c.Provider(id); provider != nil {
			out = append(out, provider)
		}
	}
	return out
}

// Model looks up a single model by provider and model id. It returns nil when
// either is unknown.
func (c *Catalog) Model(providerID, modelID string) *llm.Model {
	provider := c.Provider(providerID)
	if provider == nil {
		return nil
	}
	return provider.Model(modelID)
}

// Auth is the resolved credential for a provider.
type Auth struct {
	// APIKey is the key to send. For ambient AWS/Google credentials this is
	// the "<authenticated>" sentinel, meaning the wire protocol resolves the
	// real credential itself.
	APIKey string
	// Env carries provider-scoped values the wire protocol needs (Cloudflare
	// account and gateway ids).
	Env map[string]string
	// Headers are auth headers that must be sent instead of the protocol
	// default. A nil value suppresses a default header.
	Headers llm.ProviderHeaders
	// Source describes where the credential came from, for diagnostics:
	// "stored credential", "OAuth", or an environment variable name.
	Source string
}

// IsAmbient reports whether the credential is resolved by the wire protocol
// rather than being a usable key.
func (a *Auth) IsAmbient() bool { return a != nil && a.APIKey == authenticatedSentinel }

// ResolveAuth resolves credentials for a provider. ok is false when the
// provider is unknown or has no usable credential.
func (c *Catalog) ResolveAuth(providerID string) (auth *Auth, ok bool) {
	return c.resolveAuth(providerID, "")
}

// ResolveAuthWithKey resolves credentials, preferring an explicit override key
// over stored and ambient credentials (pi's overrides.apiKey).
func (c *Catalog) ResolveAuthWithKey(providerID, apiKey string) (auth *Auth, ok bool) {
	return c.resolveAuth(providerID, apiKey)
}

func (c *Catalog) resolveAuth(providerID, overrideKey string) (*Auth, bool) {
	config, known := builtinProviderConfigs[providerID]
	if !known {
		return nil, false
	}

	// An override key short-circuits everything except providers with no
	// api-key auth at all.
	if overrideKey != "" {
		if config.kind == AuthOAuthOnly {
			return nil, false
		}
		return c.buildAPIKeyAuth(providerID, config, &Credential{Type: "api_key", Key: overrideKey}, "override")
	}

	// A stored credential owns the provider: no ambient fallback when one
	// exists but cannot be used.
	if c.credentials != nil {
		stored, err := c.credentials.Read(providerID)
		if err == nil && stored != nil {
			switch stored.Type {
			case "oauth":
				if !config.oauth {
					return nil, false
				}
				return c.resolveOAuth(providerID, stored)
			case "api_key":
				if config.kind == AuthOAuthOnly {
					return nil, false
				}
				return c.buildAPIKeyAuth(providerID, config, stored, "stored credential")
			default:
				return nil, false
			}
		}
	}

	if config.kind == AuthOAuthOnly {
		return nil, false
	}
	return c.buildAPIKeyAuth(providerID, config, nil, "")
}

// buildAPIKeyAuth applies the provider's api-key strategy. credential is nil
// for ambient resolution.
func (c *Catalog) buildAPIKeyAuth(providerID string, config builtinProviderConfig, credential *Credential, source string) (*Auth, bool) {
	switch config.kind {
	case AuthAnthropic:
		return c.resolveAnthropicAuth(credential, source)
	case AuthCloudflareWorkersAI:
		return c.resolveCloudflareAuth(credential, source, false)
	case AuthCloudflareAIGateway:
		return c.resolveCloudflareAuth(credential, source, true)
	case AuthAmbientBedrock:
		return c.resolveBedrockAuth(credential, source)
	case AuthVertex:
		return c.resolveVertexAuth(credential, source)
	case AuthMetaBearer:
		return c.resolveMetaAuth(config, credential, source)
	default:
		return c.resolveEnvKeyAuth(config, credential, source)
	}
}

// resolveEnvKeyAuth is the standard strategy: a stored key wins, else the
// first environment variable that is set (pi's envApiKeyAuth).
func (c *Catalog) resolveEnvKeyAuth(config builtinProviderConfig, credential *Credential, source string) (*Auth, bool) {
	if credential != nil && credential.Key != "" {
		return &Auth{APIKey: credential.Key, Env: copyEnv(credential.Env), Source: source}, true
	}
	for _, name := range config.envKeys {
		if value := c.env(name); value != "" {
			return &Auth{APIKey: value, Source: name}, true
		}
	}
	return nil, false
}

// resolveMetaAuth sends the Model API key as a bearer token.
//
// The key is deliberately left out of Auth.APIKey: the anthropic-messages
// streamer sets x-api-key whenever it has one, and Meta's front door reads the
// Authorization header instead. Carrying it only in the header is what keeps
// the request to exactly the credential Meta expects.
func (c *Catalog) resolveMetaAuth(config builtinProviderConfig, credential *Credential, source string) (*Auth, bool) {
	var (
		key string
		env map[string]string
	)
	if credential != nil && credential.Key != "" {
		key, env = credential.Key, copyEnv(credential.Env)
	} else {
		for _, name := range config.envKeys {
			if value := c.env(name); value != "" {
				key, source = value, name
				break
			}
		}
	}
	if key == "" {
		return nil, false
	}
	return &Auth{
		Env:     env,
		Headers: metaBearerHeaders("Bearer " + key),
		Source:  source,
	}, true
}

// resolveAnthropicAuth handles ANTHROPIC_AUTH_TOKEN, which must travel as an
// Authorization: Bearer header rather than x-api-key (anthropic.ts).
func (c *Catalog) resolveAnthropicAuth(credential *Credential, source string) (*Auth, bool) {
	if credential != nil && credential.Key != "" {
		return &Auth{APIKey: credential.Key, Env: copyEnv(credential.Env), Source: source}, true
	}
	if token := c.env(anthropicAuthTokenEnv); token != "" {
		bearer := "Bearer " + token
		// The key is unused for the request itself; the header carries auth.
		return &Auth{
			APIKey:  token,
			Headers: llm.ProviderHeaders{"Authorization": &bearer, "x-api-key": nil},
			Source:  anthropicAuthTokenEnv,
		}, true
	}
	for _, name := range []string{anthropicOAuthTokenEnv, anthropicAPIKeyEnv} {
		if value := c.env(name); value != "" {
			return &Auth{APIKey: value, Source: name}, true
		}
	}
	return nil, false
}

// resolveCloudflareAuth merges credential and ambient values per field: a
// credential carrying only the key still picks up the account and gateway ids
// from the environment (cloudflare-auth.ts).
func (c *Catalog) resolveCloudflareAuth(credential *Credential, source string, gateway bool) (*Auth, bool) {
	resolve := func(name string) string {
		if credential != nil {
			if name == cloudflareAPIKeyEnv {
				if credential.Key != "" {
					return credential.Key
				}
			} else if value := credential.Env[name]; value != "" {
				return value
			}
		}
		return c.env(name)
	}

	apiKey := resolve(cloudflareAPIKeyEnv)
	accountID := resolve(cloudflareAccountIDEnv)
	gatewayID := ""
	if gateway {
		gatewayID = resolve(cloudflareGatewayIDEnv)
	}
	if apiKey == "" || accountID == "" || (gateway && gatewayID == "") {
		return nil, false
	}

	if source == "" {
		source = cloudflareAPIKeyEnv
	}
	auth := &Auth{
		APIKey: apiKey,
		Env:    map[string]string{cloudflareAccountIDEnv: accountID},
		Source: source,
	}
	if gateway {
		auth.Env[cloudflareGatewayIDEnv] = gatewayID
		// The gateway authenticates with its own header and suppresses the
		// protocol defaults.
		bearer := "Bearer " + apiKey
		auth.Headers = llm.ProviderHeaders{
			"cf-aig-authorization": &bearer,
			"Authorization":        nil,
			"x-api-key":            nil,
		}
	}
	return auth, true
}

// bedrockCredentialEnvVars are the AWS credential sources pi accepts for
// Bedrock (env-api-keys.ts).
var bedrockCredentialEnvVars = []string{
	"AWS_PROFILE",
	"AWS_BEARER_TOKEN_BEDROCK",
	"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
	"AWS_CONTAINER_CREDENTIALS_FULL_URI",
	"AWS_WEB_IDENTITY_TOKEN_FILE",
}

// resolveBedrockAuth reports configured when any AWS credential source is
// present; the wire protocol resolves the actual credentials.
func (c *Catalog) resolveBedrockAuth(credential *Credential, source string) (*Auth, bool) {
	if credential != nil && credential.Key != "" {
		return &Auth{APIKey: credential.Key, Env: copyEnv(credential.Env), Source: source}, true
	}
	if c.env("AWS_ACCESS_KEY_ID") != "" && c.env("AWS_SECRET_ACCESS_KEY") != "" {
		return &Auth{APIKey: authenticatedSentinel, Source: "AWS_ACCESS_KEY_ID"}, true
	}
	for _, name := range bedrockCredentialEnvVars {
		if c.env(name) != "" {
			return &Auth{APIKey: authenticatedSentinel, Source: name}, true
		}
	}
	return nil, false
}

// resolveVertexAuth accepts an explicit key, or Application Default
// Credentials together with project and location.
func (c *Catalog) resolveVertexAuth(credential *Credential, source string) (*Auth, bool) {
	if credential != nil && credential.Key != "" {
		return &Auth{APIKey: credential.Key, Env: copyEnv(credential.Env), Source: source}, true
	}
	if key := c.env("GOOGLE_CLOUD_API_KEY"); key != "" {
		return &Auth{APIKey: key, Source: "GOOGLE_CLOUD_API_KEY"}, true
	}

	hasProject := c.env("GOOGLE_CLOUD_PROJECT") != "" || c.env("GCLOUD_PROJECT") != ""
	hasLocation := c.env("GOOGLE_CLOUD_LOCATION") != ""
	if c.hasVertexADCCredentials() && hasProject && hasLocation {
		return &Auth{APIKey: authenticatedSentinel, Source: "Application Default Credentials"}, true
	}
	return nil, false
}

// hasVertexADCCredentials reports whether a Google ADC file is present.
func (c *Catalog) hasVertexADCCredentials() bool {
	exists := c.fileExists
	if exists == nil {
		exists = fileExists
	}
	if path := c.env("GOOGLE_APPLICATION_CREDENTIALS"); path != "" {
		return exists(path)
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	return exists(filepath.Join(home, ".config", "gcloud", "application_default_credentials.json"))
}

// resolveOAuth derives request auth from a stored OAuth credential,
// refreshing it first when it is expired or close to expiring.
//
// A stored credential owns the provider, so a failed refresh reports "not
// configured" rather than falling back to ambient environment keys. The
// refresh itself runs inside CredentialStore.Modify, so concurrent requests
// cannot double-refresh a rotated token.
func (c *Catalog) resolveOAuth(providerID string, stored *Credential) (*Auth, bool) {
	provider, hasProvider := OAuthProviderFor(providerID)

	if stored.Access != "" && !expiresSoon(stored) {
		if hasProvider {
			return provider.ToAuth(stored), true
		}
		// No ported flow: the access token is still usable as a bearer key.
		return &Auth{APIKey: stored.Access, Source: "OAuth"}, true
	}

	// Expired and no refresh implementation for this provider.
	if !hasProvider {
		return nil, false
	}

	ctx := c.oauthContext()
	// Do not re-spend the full retry budget on a provider that just failed:
	// these paths run on every picker rebuild, and a dead auth host would
	// otherwise stall each one all over again.
	if previous := c.OAuthError(providerID); previous != nil {
		return nil, false
	}
	if c.oauthTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.oauthTimeout)
		defer cancel()
	}
	refreshed, err := c.refreshOAuthCredential(ctx, providerID, provider)
	if err != nil {
		c.recordOAuthError(providerID, err)
		return nil, false
	}
	return provider.ToAuth(refreshed), true
}

// oauthContext bounds an implicit refresh triggered by auth resolution.
func (c *Catalog) oauthContext() context.Context {
	if c.oauthCtx != nil {
		return c.oauthCtx
	}
	return context.Background()
}

// recordOAuthError stores the most recent refresh failure so callers can
// explain why a provider stopped resolving.
func (c *Catalog) recordOAuthError(providerID string, err error) {
	c.oauthMu.Lock()
	defer c.oauthMu.Unlock()
	if c.oauthErrors == nil {
		c.oauthErrors = map[string]error{}
	}
	c.oauthErrors[providerID] = err
}

// OAuthError returns the last OAuth refresh failure for a provider, if any.
// ResolveAuth reports only configured/not-configured, so this is how the CLI
// tells "no credential" apart from "your login expired".
func (c *Catalog) OAuthError(providerID string) error {
	c.oauthMu.Lock()
	defer c.oauthMu.Unlock()
	return c.oauthErrors[providerID]
}

// FindEnvKeys returns the environment variables that currently provide a key
// for the provider (pi's findEnvKeys). Ambient sources such as AWS profiles
// and Google ADC are intentionally excluded.
func (c *Catalog) FindEnvKeys(providerID string) []string {
	config, ok := builtinProviderConfigs[providerID]
	if !ok {
		return nil
	}
	names := config.envKeys
	if config.kind == AuthAnthropic {
		names = []string{anthropicAuthTokenEnv, anthropicOAuthTokenEnv, anthropicAPIKeyEnv}
	}
	var found []string
	for _, name := range names {
		if c.env(name) != "" {
			found = append(found, name)
		}
	}
	return found
}

// IsConfigured reports whether the provider has a usable credential.
func (c *Catalog) IsConfigured(providerID string) bool {
	_, ok := c.ResolveAuth(providerID)
	return ok
}

// ConfiguredProviderIDs returns the ids of providers with usable credentials.
func (c *Catalog) ConfiguredProviderIDs() []string {
	var out []string
	for _, id := range c.ProviderIDs() {
		if c.IsConfigured(id) {
			out = append(out, id)
		}
	}
	return out
}

// ResolveModel finds a model by "provider/model" or by a bare model id, and
// resolves the provider's credentials in one step. The model returned carries
// the provider's auth headers merged into Model.Headers.
//
// A bare model id must be unambiguous among configured providers.
func (c *Catalog) ResolveModel(ref string) (model *llm.Model, auth *Auth, err error) {
	providerID, modelID, qualified := strings.Cut(ref, "/")
	if qualified {
		model = c.Model(providerID, modelID)
		if model == nil {
			return nil, nil, fmt.Errorf("catalog: unknown model %q", ref)
		}
		auth, ok := c.ResolveAuth(providerID)
		if !ok {
			return nil, nil, fmt.Errorf("catalog: provider %q has no credentials configured", providerID)
		}
		return model, auth, nil
	}

	// Bare id: search configured providers only, so an unconfigured provider
	// cannot shadow a usable one.
	var matches []string
	for _, id := range c.ConfiguredProviderIDs() {
		if c.Model(id, ref) != nil {
			matches = append(matches, id)
		}
	}
	switch len(matches) {
	case 0:
		return nil, nil, fmt.Errorf("catalog: no configured provider offers model %q", ref)
	case 1:
		providerID = matches[0]
		auth, _ := c.ResolveAuth(providerID)
		return c.Model(providerID, ref), auth, nil
	default:
		return nil, nil, fmt.Errorf("catalog: model %q is ambiguous; qualify it as one of %s",
			ref, strings.Join(matches, ", "))
	}
}

func copyEnv(env map[string]string) map[string]string {
	if len(env) == 0 {
		return nil
	}
	out := make(map[string]string, len(env))
	for k, v := range env {
		out[k] = v
	}
	return out
}
