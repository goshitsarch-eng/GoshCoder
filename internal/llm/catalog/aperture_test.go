package catalog

import (
	"os"
	"path/filepath"
	"testing"

	"goshcoder/internal/aperture"
	goshconfig "goshcoder/internal/config"
	"goshcoder/internal/llm"
)

// setupApertureDir writes an aperture.json plus a matching cache snapshot
// into a temp agent dir and points the process at it.
func setupApertureDir(t *testing.T, configJSON string, cache *aperture.Cache) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(goshconfig.EnvAgentDir, dir)
	if err := os.MkdirAll(filepath.Join(dir, "extensions"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(goshconfig.AperturePath(), []byte(configJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	if cache != nil {
		if err := aperture.SaveCache(goshconfig.ApertureCachePath(), *cache); err != nil {
			t.Fatal(err)
		}
	}
}

const apertureDedicatedConfig = `{
	"baseUrl": "http://gw.example",
	"onboardingDone": true,
	"dedicated": {"enabled": true, "providers": []},
	"proxy": {"enabled": false, "upstreamProviders": []}
}`

func dedicatedCache() *aperture.Cache {
	models := []llm.Model{{
		ID: "anthropic/claude-sonnet-5", Name: "Claude Sonnet 5",
		Provider: "aperture", API: "anthropic-messages",
		BaseURL: "http://gw.example", Input: []string{"text"},
		ContextWindow: 200_000, MaxTokens: 64_000,
	}}
	key := aperture.BuildCatalogKey("http://gw.example", mustResolve(apertureDedicatedConfig))
	cache := aperture.NewCache(key, models, []aperture.GatewayProvider{{
		ID: "anthropic", Name: "Anthropic", Models: []string{"claude-sonnet-5"},
		Compatibility: map[string]bool{"anthropic_messages": true},
	}})
	return &cache
}

func mustResolve(configJSON string) aperture.Resolved {
	dir, err := os.MkdirTemp("", "aperture-resolve")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "aperture.json")
	if err := os.WriteFile(path, []byte(configJSON), 0o600); err != nil {
		panic(err)
	}
	config, err := aperture.Load(path)
	if err != nil {
		panic(err)
	}
	return config.Resolve()
}

func TestApertureDedicatedProviderFromCache(t *testing.T) {
	setupApertureDir(t, apertureDedicatedConfig, dedicatedCache())
	c := NewCatalog(nil)
	provider := c.Provider("aperture")
	if provider == nil {
		t.Fatal("aperture provider missing")
	}
	if provider.BaseURL != "http://gw.example/v1" {
		t.Errorf("base URL = %q", provider.BaseURL)
	}
	models := provider.Models()
	if len(models) != 1 || models[0].ID != "anthropic/claude-sonnet-5" {
		t.Fatalf("models = %+v", models)
	}
	if models[0].API != "anthropic-messages" || models[0].BaseURL != "http://gw.example" {
		t.Errorf("model routing = %s %s", models[0].API, models[0].BaseURL)
	}

	auth, ok := c.ResolveAuth("aperture")
	if !ok || auth.APIKey != "-" || auth.Source != "aperture gateway" {
		t.Errorf("auth = %+v ok=%v", auth, ok)
	}

	model, _, err := c.ResolveModel("aperture/anthropic/claude-sonnet-5")
	if err != nil {
		t.Fatalf("ResolveModel: %v", err)
	}
	if model.ID != "anthropic/claude-sonnet-5" {
		t.Errorf("resolved id = %q", model.ID)
	}
}

func TestApertureUnconfigured(t *testing.T) {
	t.Setenv(goshconfig.EnvAgentDir, t.TempDir())
	c := NewCatalog(nil)
	provider := c.Provider("aperture")
	if provider == nil {
		t.Fatal("aperture provider must exist even unconfigured")
	}
	if len(provider.Models()) != 0 {
		t.Error("no models when unconfigured")
	}
	if _, ok := c.ResolveAuth("aperture"); ok {
		t.Error("no auth when unconfigured")
	}
}

func TestApertureDedicatedDisabledResolvesNoAuth(t *testing.T) {
	setupApertureDir(t, `{
		"baseUrl": "http://gw.example",
		"onboardingDone": true,
		"dedicated": {"enabled": false, "providers": []},
		"proxy": {"enabled": false, "upstreamProviders": []}
	}`, nil)
	c := NewCatalog(nil)
	if _, ok := c.ResolveAuth("aperture"); ok {
		t.Error("disabled dedicated capability must not resolve auth")
	}
	if models := c.Provider("aperture").Models(); len(models) != 0 {
		t.Errorf("disabled dedicated capability must expose no models: %d", len(models))
	}
}

const apertureProxyConfig = `{
	"baseUrl": "http://gw.example",
	"onboardingDone": true,
	"dedicated": {"enabled": false, "providers": []},
	"proxy": {"enabled": true, "upstreamProviders": [
		{"id": "anthropic", "shouldCheckGatewayModels": true}
	]}
}`

func proxyCache() *aperture.Cache {
	key := aperture.BuildCatalogKey("http://gw.example", mustResolve(apertureProxyConfig))
	cache := aperture.NewCache(key, nil, []aperture.GatewayProvider{{
		ID: "anthropic", Name: "Anthropic", Models: []string{"claude-sonnet-5"},
		Compatibility: map[string]bool{"anthropic_messages": true},
	}})
	return &cache
}

func TestApertureProxyRewritesProvider(t *testing.T) {
	setupApertureDir(t, apertureProxyConfig, proxyCache())
	c := NewCatalog(nil)
	provider := c.Provider("anthropic")
	if provider == nil {
		t.Fatal("anthropic provider missing")
	}
	models := provider.Models()
	if len(models) == 0 {
		t.Fatal("proxied provider keeps its own model definitions")
	}
	for _, model := range models {
		if model.BaseURL != "http://gw.example" {
			t.Fatalf("model %s base URL = %q, want the gateway root", model.ID, model.BaseURL)
		}
		if model.ID == "" || model.API != "anthropic-messages" {
			t.Fatalf("model identity must be untouched: %+v", model)
		}
	}
	// Ids stay bare in the picker; qualification happens per request.
	for _, model := range models {
		if len(model.ID) > 10 && model.ID[:10] == "anthropic/" {
			t.Fatalf("picker ids must stay bare: %q", model.ID)
		}
	}

	// Gateway-injected placeholder auth when no local credential exists.
	auth, ok := c.ResolveAuth("anthropic")
	if !ok || auth.APIKey != "-" || auth.Source != "aperture proxy" {
		t.Errorf("proxy auth = %+v ok=%v", auth, ok)
	}

	// Unproxied providers are untouched.
	openai := c.Provider("openai")
	for _, model := range openai.Models() {
		if model.BaseURL == "http://gw.example" {
			t.Fatalf("unproxied provider rewritten: %+v", model)
		}
	}
	if _, ok := c.ResolveAuth("openai"); ok {
		t.Error("unproxied provider without credentials must stay unconfigured")
	}
}

func TestAperturePassthroughKeepsNativeAuth(t *testing.T) {
	cache := proxyCache()
	cache.Gateway[0].RequiresClientAuth = true
	setupApertureDir(t, apertureProxyConfig, cache)
	c := NewCatalog(nil)
	// Passthrough providers keep native auth; with no local credential the
	// provider stays unconfigured so a real key is required.
	if _, ok := c.ResolveAuth("anthropic"); ok {
		t.Error("passthrough provider without credentials must not resolve")
	}
	// The routing rewrite still applies.
	models := c.Provider("anthropic").Models()
	if len(models) == 0 || models[0].BaseURL != "http://gw.example" {
		t.Error("passthrough providers still route through the gateway")
	}
}

func TestApertureProxyKeepGatewayModelsOnly(t *testing.T) {
	configJSON := `{
		"baseUrl": "http://gw.example",
		"onboardingDone": true,
		"dedicated": {"enabled": false, "providers": []},
		"proxy": {"enabled": true, "upstreamProviders": [
			{"id": "anthropic", "keepGatewayModelsOnly": true}
		]}
	}`
	// Discover one real anthropic model id to keep.
	t.Setenv(goshconfig.EnvAgentDir, t.TempDir())
	native := NewCatalog(nil).Provider("anthropic").Models()
	if len(native) < 2 {
		t.Skip("needs at least two builtin anthropic models")
	}
	kept := native[0].ID

	key := aperture.BuildCatalogKey("http://gw.example", mustResolve(configJSON))
	cache := aperture.NewCache(key, nil, []aperture.GatewayProvider{{
		ID: "anthropic", Models: []string{kept},
		Compatibility: map[string]bool{"anthropic_messages": true},
	}})
	setupApertureDir(t, configJSON, &cache)

	models := NewCatalog(nil).Provider("anthropic").Models()
	if len(models) != 1 || models[0].ID != kept {
		t.Fatalf("keepGatewayModelsOnly filter: %+v", models)
	}
}
