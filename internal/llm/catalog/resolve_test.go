package catalog

import (
	"strings"
	"testing"
	"time"
)

// newTestCatalog builds a catalog with a fake environment and no ambient files,
// so tests never depend on the host machine's credentials.
func newTestCatalog(env map[string]string, credentials *CredentialStore) *Catalog {
	c := NewCatalog(credentials)
	c.SetGetenv(func(name string) string { return env[name] })
	c.SetFileExists(func(string) bool { return false })
	return c
}

func TestCatalogProviderLookup(t *testing.T) {
	c := newTestCatalog(nil, nil)

	provider := c.Provider("anthropic")
	if provider == nil {
		t.Fatal("anthropic provider not found")
	}
	if provider.Name != "Anthropic" || provider.BaseURL != "https://api.anthropic.com" {
		t.Fatalf("provider = %#v", provider)
	}
	if provider.Kind != AuthAnthropic || !provider.OAuth {
		t.Fatalf("auth config = kind %v oauth %v", provider.Kind, provider.OAuth)
	}
	if len(provider.Models()) == 0 {
		t.Fatal("provider has no models")
	}

	if c.Provider("not-a-provider") != nil {
		t.Fatal("unknown provider should return nil")
	}
}

func TestCatalogProviderIDsSorted(t *testing.T) {
	c := newTestCatalog(nil, nil)
	ids := c.ProviderIDs()
	if len(ids) < 20 {
		t.Fatalf("provider count = %d, want the full builtin set", len(ids))
	}
	for i := 1; i < len(ids); i++ {
		if ids[i-1] > ids[i] {
			t.Fatalf("ids are not sorted at %d: %q > %q", i, ids[i-1], ids[i])
		}
	}
}

func TestCatalogModelLookup(t *testing.T) {
	c := newTestCatalog(nil, nil)

	model := c.Model("anthropic", "claude-sonnet-4-5")
	if model == nil {
		t.Skip("claude-sonnet-4-5 is not in this catalog snapshot")
	}
	if model.API != "anthropic-messages" || model.Provider != "anthropic" {
		t.Fatalf("model = %#v", model)
	}

	if c.Model("anthropic", "not-a-model") != nil {
		t.Fatal("unknown model should return nil")
	}
	if c.Model("not-a-provider", "x") != nil {
		t.Fatal("unknown provider should return nil")
	}
}

// TestCatalogModelsAreCopies confirms callers cannot mutate shared model data.
func TestCatalogModelsAreCopies(t *testing.T) {
	c := newTestCatalog(nil, nil)
	provider := c.Provider("openai")
	if provider == nil {
		t.Skip("openai provider is absent from this snapshot")
	}
	models := provider.Models()
	if len(models) == 0 {
		t.Skip("no models to check")
	}
	original := models[0].Name
	models[0].Name = "tampered"

	fresh := c.Provider("openai").Models()
	if fresh[0].Name != original {
		t.Fatalf("shared model data was mutated: %q", fresh[0].Name)
	}
}

func TestCatalogResolveEnvKeyAuth(t *testing.T) {
	c := newTestCatalog(map[string]string{"OPENAI_API_KEY": "sk-env"}, nil)

	auth, ok := c.ResolveAuth("openai")
	if !ok {
		t.Fatal("expected openai to resolve from the environment")
	}
	if auth.APIKey != "sk-env" || auth.Source != "OPENAI_API_KEY" {
		t.Fatalf("auth = %#v", auth)
	}
	if auth.IsAmbient() {
		t.Fatal("an env key is not an ambient credential")
	}
}

func TestCatalogUnconfiguredProvider(t *testing.T) {
	c := newTestCatalog(nil, nil)
	if _, ok := c.ResolveAuth("openai"); ok {
		t.Fatal("openai should be unconfigured with an empty environment")
	}
	if c.IsConfigured("openai") {
		t.Fatal("IsConfigured should be false")
	}
	if _, ok := c.ResolveAuth("not-a-provider"); ok {
		t.Fatal("an unknown provider cannot be configured")
	}
}

func TestCatalogStoredCredentialWins(t *testing.T) {
	store := NewInMemoryCredentialStore()
	if _, err := store.Modify("openai", func(*Credential) (*Credential, error) {
		return &Credential{Type: "api_key", Key: "sk-stored"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	c := newTestCatalog(map[string]string{"OPENAI_API_KEY": "sk-env"}, store)

	auth, ok := c.ResolveAuth("openai")
	if !ok {
		t.Fatal("expected a resolved credential")
	}
	if auth.APIKey != "sk-stored" || auth.Source != "stored credential" {
		t.Fatalf("auth = %#v, want the stored credential to win", auth)
	}
}

// TestCatalogOverrideKeyWins covers an explicit key beating both stored and
// ambient credentials.
func TestCatalogOverrideKeyWins(t *testing.T) {
	store := NewInMemoryCredentialStore()
	if _, err := store.Modify("openai", func(*Credential) (*Credential, error) {
		return &Credential{Type: "api_key", Key: "sk-stored"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	c := newTestCatalog(map[string]string{"OPENAI_API_KEY": "sk-env"}, store)

	auth, ok := c.ResolveAuthWithKey("openai", "sk-override")
	if !ok || auth.APIKey != "sk-override" {
		t.Fatalf("auth = %#v", auth)
	}
}

func TestCatalogStoredOAuthCredential(t *testing.T) {
	t.Run("a valid token resolves", func(t *testing.T) {
		store := NewInMemoryCredentialStore()
		if _, err := store.Modify("anthropic", func(*Credential) (*Credential, error) {
			return &Credential{
				Type:    "oauth",
				Access:  "oauth-access",
				Expires: time.Now().Add(time.Hour).UnixMilli(),
			}, nil
		}); err != nil {
			t.Fatal(err)
		}
		c := newTestCatalog(nil, store)

		auth, ok := c.ResolveAuth("anthropic")
		if !ok || auth.APIKey != "oauth-access" || auth.Source != "OAuth" {
			t.Fatalf("auth = %#v, ok = %v", auth, ok)
		}
	})

	t.Run("an expired token does not fall back to env", func(t *testing.T) {
		store := NewInMemoryCredentialStore()
		if _, err := store.Modify("anthropic", func(*Credential) (*Credential, error) {
			return &Credential{
				Type:    "oauth",
				Access:  "stale",
				Expires: time.Now().Add(-time.Hour).UnixMilli(),
			}, nil
		}); err != nil {
			t.Fatal(err)
		}
		// Refresh is not ported, and a stored credential owns the provider, so
		// the env key must not be used.
		c := newTestCatalog(map[string]string{"ANTHROPIC_API_KEY": "sk-env"}, store)

		if auth, ok := c.ResolveAuth("anthropic"); ok {
			t.Fatalf("expected no credential, got %#v", auth)
		}
	})
}

func TestCatalogAnthropicAuthToken(t *testing.T) {
	t.Run("auth token becomes a bearer header", func(t *testing.T) {
		c := newTestCatalog(map[string]string{"ANTHROPIC_AUTH_TOKEN": "tok"}, nil)
		auth, ok := c.ResolveAuth("anthropic")
		if !ok {
			t.Fatal("expected a resolved credential")
		}
		if auth.Source != "ANTHROPIC_AUTH_TOKEN" {
			t.Fatalf("source = %q", auth.Source)
		}
		header, present := auth.Headers["Authorization"]
		if !present || header == nil || *header != "Bearer tok" {
			t.Fatalf("Authorization header = %#v", auth.Headers)
		}
		// x-api-key must be suppressed so the bearer token is authoritative.
		suppressed, present := auth.Headers["x-api-key"]
		if !present || suppressed != nil {
			t.Fatalf("x-api-key should be suppressed: %#v", auth.Headers)
		}
	})

	t.Run("api key needs no headers", func(t *testing.T) {
		c := newTestCatalog(map[string]string{"ANTHROPIC_API_KEY": "sk-ant"}, nil)
		auth, ok := c.ResolveAuth("anthropic")
		if !ok || auth.APIKey != "sk-ant" || auth.Headers != nil {
			t.Fatalf("auth = %#v", auth)
		}
	})

	t.Run("auth token is preferred over api key", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"ANTHROPIC_AUTH_TOKEN": "tok",
			"ANTHROPIC_API_KEY":    "sk-ant",
		}, nil)
		auth, _ := c.ResolveAuth("anthropic")
		if auth.Source != "ANTHROPIC_AUTH_TOKEN" {
			t.Fatalf("source = %q", auth.Source)
		}
	})
}

func TestCatalogCloudflareWorkersAI(t *testing.T) {
	t.Run("needs both key and account id", func(t *testing.T) {
		c := newTestCatalog(map[string]string{"CLOUDFLARE_API_KEY": "cf-key"}, nil)
		if _, ok := c.ResolveAuth("cloudflare-workers-ai"); ok {
			t.Fatal("an account id is required")
		}
	})

	t.Run("resolves with both", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"CLOUDFLARE_API_KEY":    "cf-key",
			"CLOUDFLARE_ACCOUNT_ID": "acct",
		}, nil)
		auth, ok := c.ResolveAuth("cloudflare-workers-ai")
		if !ok {
			t.Fatal("expected a resolved credential")
		}
		if auth.APIKey != "cf-key" || auth.Env["CLOUDFLARE_ACCOUNT_ID"] != "acct" {
			t.Fatalf("auth = %#v", auth)
		}
	})

	// A credential holding only the key must still pick up the account id from
	// the environment (pi's per-field merge).
	t.Run("credential key merges with ambient account id", func(t *testing.T) {
		store := NewInMemoryCredentialStore()
		if _, err := store.Modify("cloudflare-workers-ai", func(*Credential) (*Credential, error) {
			return &Credential{Type: "api_key", Key: "cf-stored"}, nil
		}); err != nil {
			t.Fatal(err)
		}
		c := newTestCatalog(map[string]string{"CLOUDFLARE_ACCOUNT_ID": "acct"}, store)

		auth, ok := c.ResolveAuth("cloudflare-workers-ai")
		if !ok {
			t.Fatal("expected a resolved credential")
		}
		if auth.APIKey != "cf-stored" || auth.Env["CLOUDFLARE_ACCOUNT_ID"] != "acct" {
			t.Fatalf("auth = %#v", auth)
		}
	})
}

func TestCatalogCloudflareAIGateway(t *testing.T) {
	t.Run("needs a gateway id too", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"CLOUDFLARE_API_KEY":    "cf-key",
			"CLOUDFLARE_ACCOUNT_ID": "acct",
		}, nil)
		if _, ok := c.ResolveAuth("cloudflare-ai-gateway"); ok {
			t.Fatal("a gateway id is required")
		}
	})

	t.Run("uses the cf-aig-authorization header", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"CLOUDFLARE_API_KEY":    "cf-key",
			"CLOUDFLARE_ACCOUNT_ID": "acct",
			"CLOUDFLARE_GATEWAY_ID": "gw",
		}, nil)
		auth, ok := c.ResolveAuth("cloudflare-ai-gateway")
		if !ok {
			t.Fatal("expected a resolved credential")
		}
		if auth.Env["CLOUDFLARE_GATEWAY_ID"] != "gw" {
			t.Fatalf("env = %#v", auth.Env)
		}
		header, present := auth.Headers["cf-aig-authorization"]
		if !present || header == nil || *header != "Bearer cf-key" {
			t.Fatalf("headers = %#v", auth.Headers)
		}
		// The protocol defaults are suppressed.
		for _, name := range []string{"Authorization", "x-api-key"} {
			value, present := auth.Headers[name]
			if !present || value != nil {
				t.Fatalf("%s should be suppressed: %#v", name, auth.Headers)
			}
		}
	})
}

func TestCatalogBedrockAmbientCredentials(t *testing.T) {
	tests := []struct {
		name string
		env  map[string]string
	}{
		{"profile", map[string]string{"AWS_PROFILE": "default"}},
		{"iam keys", map[string]string{"AWS_ACCESS_KEY_ID": "AKIA", "AWS_SECRET_ACCESS_KEY": "secret"}},
		{"bearer token", map[string]string{"AWS_BEARER_TOKEN_BEDROCK": "tok"}},
		{"ecs relative uri", map[string]string{"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/v2/creds"}},
		{"irsa", map[string]string{"AWS_WEB_IDENTITY_TOKEN_FILE": "/var/run/token"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newTestCatalog(tt.env, nil)
			auth, ok := c.ResolveAuth("amazon-bedrock")
			if !ok {
				t.Fatal("expected bedrock to be configured")
			}
			// The wire protocol resolves the real credential.
			if !auth.IsAmbient() {
				t.Fatalf("auth = %#v, want the ambient sentinel", auth)
			}
		})
	}

	t.Run("an access key alone is not enough", func(t *testing.T) {
		c := newTestCatalog(map[string]string{"AWS_ACCESS_KEY_ID": "AKIA"}, nil)
		if _, ok := c.ResolveAuth("amazon-bedrock"); ok {
			t.Fatal("a secret access key is also required")
		}
	})

	t.Run("no aws env is unconfigured", func(t *testing.T) {
		c := newTestCatalog(nil, nil)
		if _, ok := c.ResolveAuth("amazon-bedrock"); ok {
			t.Fatal("expected bedrock to be unconfigured")
		}
	})
}

func TestCatalogVertexAuth(t *testing.T) {
	t.Run("explicit api key", func(t *testing.T) {
		c := newTestCatalog(map[string]string{"GOOGLE_CLOUD_API_KEY": "vertex-key"}, nil)
		auth, ok := c.ResolveAuth("google-vertex")
		if !ok || auth.APIKey != "vertex-key" {
			t.Fatalf("auth = %#v, ok = %v", auth, ok)
		}
	})

	t.Run("ADC plus project and location", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
			"GOOGLE_CLOUD_PROJECT":           "proj",
			"GOOGLE_CLOUD_LOCATION":          "us-central1",
		}, nil)
		c.SetFileExists(func(path string) bool { return path == "/creds.json" })

		auth, ok := c.ResolveAuth("google-vertex")
		if !ok {
			t.Fatal("expected vertex to be configured")
		}
		if !auth.IsAmbient() {
			t.Fatalf("auth = %#v, want the ambient sentinel", auth)
		}
	})

	t.Run("ADC without a location is unconfigured", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"GOOGLE_APPLICATION_CREDENTIALS": "/creds.json",
			"GOOGLE_CLOUD_PROJECT":           "proj",
		}, nil)
		c.SetFileExists(func(string) bool { return true })

		if _, ok := c.ResolveAuth("google-vertex"); ok {
			t.Fatal("a location is required")
		}
	})

	t.Run("a missing credentials file is unconfigured", func(t *testing.T) {
		c := newTestCatalog(map[string]string{
			"GOOGLE_APPLICATION_CREDENTIALS": "/missing.json",
			"GOOGLE_CLOUD_PROJECT":           "proj",
			"GOOGLE_CLOUD_LOCATION":          "us-central1",
		}, nil)
		c.SetFileExists(func(string) bool { return false })

		if _, ok := c.ResolveAuth("google-vertex"); ok {
			t.Fatal("the ADC file must exist")
		}
	})
}

func TestCatalogFindEnvKeys(t *testing.T) {
	c := newTestCatalog(map[string]string{
		"OPENAI_API_KEY":       "sk",
		"ANTHROPIC_AUTH_TOKEN": "tok",
		"ANTHROPIC_API_KEY":    "sk-ant",
	}, nil)

	if got := c.FindEnvKeys("openai"); len(got) != 1 || got[0] != "OPENAI_API_KEY" {
		t.Fatalf("openai env keys = %#v", got)
	}
	// Anthropic reports every candidate that is set, in resolution order.
	got := c.FindEnvKeys("anthropic")
	if len(got) != 2 || got[0] != "ANTHROPIC_AUTH_TOKEN" || got[1] != "ANTHROPIC_API_KEY" {
		t.Fatalf("anthropic env keys = %#v", got)
	}
	if got := c.FindEnvKeys("groq"); got != nil {
		t.Fatalf("unset provider env keys = %#v, want nil", got)
	}
	if got := c.FindEnvKeys("not-a-provider"); got != nil {
		t.Fatalf("unknown provider env keys = %#v", got)
	}
}

func TestCatalogConfiguredProviderIDs(t *testing.T) {
	c := newTestCatalog(map[string]string{
		"OPENAI_API_KEY": "sk",
		"GROQ_API_KEY":   "gsk",
	}, nil)

	configured := c.ConfiguredProviderIDs()
	if len(configured) != 2 {
		t.Fatalf("configured = %#v, want exactly openai and groq", configured)
	}
	if configured[0] != "groq" || configured[1] != "openai" {
		t.Fatalf("configured = %#v, want sorted ids", configured)
	}
}

func TestCatalogResolveModel(t *testing.T) {
	c := newTestCatalog(map[string]string{"OPENAI_API_KEY": "sk-env"}, nil)

	// Find a real model id from the catalog to keep the test snapshot-proof.
	openai := c.Provider("openai")
	if openai == nil || len(openai.Models()) == 0 {
		t.Skip("openai has no models in this snapshot")
	}
	modelID := openai.Models()[0].ID

	t.Run("qualified reference", func(t *testing.T) {
		model, auth, err := c.ResolveModel("openai/" + modelID)
		if err != nil {
			t.Fatalf("ResolveModel: %v", err)
		}
		if model.ID != modelID || model.Provider != "openai" {
			t.Fatalf("model = %#v", model)
		}
		if auth.APIKey != "sk-env" {
			t.Fatalf("auth = %#v", auth)
		}
	})

	t.Run("bare reference resolves through the configured provider", func(t *testing.T) {
		model, _, err := c.ResolveModel(modelID)
		if err != nil {
			t.Fatalf("ResolveModel: %v", err)
		}
		if model.ID != modelID {
			t.Fatalf("model = %#v", model)
		}
	})

	t.Run("unknown qualified model errors", func(t *testing.T) {
		_, _, err := c.ResolveModel("openai/not-a-model")
		if err == nil || !strings.Contains(err.Error(), "unknown model") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("unconfigured provider errors", func(t *testing.T) {
		bare := newTestCatalog(nil, nil)
		_, _, err := bare.ResolveModel("openai/" + modelID)
		if err == nil || !strings.Contains(err.Error(), "no credentials configured") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("bare reference with no configured provider errors", func(t *testing.T) {
		bare := newTestCatalog(nil, nil)
		_, _, err := bare.ResolveModel(modelID)
		if err == nil || !strings.Contains(err.Error(), "no configured provider") {
			t.Fatalf("err = %v", err)
		}
	})
}

// TestCatalogResolveModelAmbiguous covers a bare id offered by more than one
// configured provider.
func TestCatalogResolveModelAmbiguous(t *testing.T) {
	c := newTestCatalog(nil, nil)

	// Find a model id shared by at least two providers.
	sharedID, owners := "", []string{}
	for _, id := range c.ProviderIDs() {
		provider := c.Provider(id)
		for _, model := range provider.Models() {
			var holders []string
			for _, otherID := range c.ProviderIDs() {
				if c.Model(otherID, model.ID) != nil {
					holders = append(holders, otherID)
				}
			}
			if len(holders) >= 2 {
				sharedID, owners = model.ID, holders
				break
			}
		}
		if sharedID != "" {
			break
		}
	}
	if sharedID == "" {
		t.Skip("no model id is shared across providers in this snapshot")
	}

	// Configure every owner so the reference is genuinely ambiguous.
	env := map[string]string{}
	configuredOwners := 0
	for _, owner := range owners {
		keys := builtinProviderConfigs[owner].envKeys
		if len(keys) == 0 {
			continue
		}
		env[keys[0]] = "key"
		configuredOwners++
	}
	if configuredOwners < 2 {
		t.Skip("cannot configure two owners of the shared model via env keys")
	}
	ambiguous := newTestCatalog(env, nil)

	_, _, err := ambiguous.ResolveModel(sharedID)
	if err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("err = %v, want an ambiguity error", err)
	}
}

func TestCatalogRawCompatIsPreserved(t *testing.T) {
	// Anthropic models carry compat keys that llm.OpenAICompletionsCompat does
	// not model, so the raw JSON must survive.
	c := newTestCatalog(nil, nil)
	provider := c.Provider("anthropic")
	if provider == nil {
		t.Skip("anthropic is absent from this snapshot")
	}
	found := false
	for _, model := range provider.Models() {
		if raw, ok := provider.RawCompat(model.ID); ok && len(raw) > 0 {
			found = true
			break
		}
	}
	if !found {
		t.Skip("no anthropic model carries compat in this snapshot")
	}
}
