package aperture

import (
	"encoding/json"
	"math"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/llm"
)

func testGatewayProviders() []GatewayProvider {
	return []GatewayProvider{
		{
			ID: "anthropic", Name: "Anthropic",
			Models:        []string{"claude-sonnet-5"},
			Compatibility: map[string]bool{"anthropic_messages": true},
			ModelInfoByID: map[string]ModelInfo{
				"claude-sonnet-5": {ID: "claude-sonnet-5", Pricing: &ModelPricing{Input: "0.00000300", Output: "0.00001500"}},
			},
		},
		{
			ID: "google", Name: "Google",
			Models:        []string{"gemini-2.5-pro"},
			Compatibility: map[string]bool{"gemini_generate_content": true},
		},
	}
}

func TestBuildModelsQualificationAndRouting(t *testing.T) {
	catalogModels := []llm.Model{
		{ID: "claude-sonnet-5", Provider: "anthropic", API: "anthropic-messages",
			BaseURL: "https://api.anthropic.com", Name: "Claude Sonnet 5",
			Reasoning: true, Input: []string{"text", "image"},
			Cost:          llm.ModelCost{ModelCostRates: llm.ModelCostRates{Input: 3, Output: 15}},
			ContextWindow: 200_000, MaxTokens: 64_000,
			RawCompat: json.RawMessage(`{"supportsDeveloperRole": false}`)},
	}
	models := BuildModels(testGatewayProviders(), "http://gw.example", "http://gw.example/v1", catalogModels, nil, nil, nil)
	if len(models) != 2 {
		t.Fatalf("models = %d", len(models))
	}

	claude := models[0]
	if claude.ID != "anthropic/claude-sonnet-5" {
		t.Errorf("body-API ids are provider-qualified: %q", claude.ID)
	}
	if claude.Provider != "aperture" || claude.API != "anthropic-messages" {
		t.Errorf("provider/api = %s/%s", claude.Provider, claude.API)
	}
	if claude.BaseURL != "http://gw.example" {
		t.Errorf("anthropic-messages routes to the gateway root: %q", claude.BaseURL)
	}
	if !claude.Reasoning || claude.ContextWindow != 200_000 || claude.MaxTokens != 64_000 {
		t.Errorf("catalog metadata not applied: %+v", claude)
	}
	// Gateway pricing wins over catalog cost, per-token strings scaled to
	// per-million rates.
	if claude.Cost.Input != 3 || claude.Cost.Output != 15 {
		t.Errorf("cost = %+v", claude.Cost)
	}
	if len(claude.RawCompat) == 0 {
		t.Error("provider-exact match carries raw compat")
	}

	gemini := models[1]
	if gemini.ID != "gemini-2.5-pro" {
		t.Errorf("path-embedding APIs keep the bare id: %q", gemini.ID)
	}
	if gemini.API != "google-generative-ai" || gemini.BaseURL != "http://gw.example/v1beta" {
		t.Errorf("gemini routing = %s %s", gemini.API, gemini.BaseURL)
	}
	// Safe defaults where no source knows the model.
	if gemini.ContextWindow != 128_000 || gemini.MaxTokens != 8_192 || gemini.Reasoning {
		t.Errorf("safe defaults not applied: %+v", gemini)
	}
}

func TestBuildModelsAPIOverride(t *testing.T) {
	var warnings []string
	notify := func(warning string) { warnings = append(warnings, warning) }

	providers := testGatewayProviders()
	providers[0].Compatibility = map[string]bool{"anthropic_messages": true, "openai_chat": true}
	// Auto-pick prefers chat completions; a valid override selects the
	// provider's Anthropic surface instead.
	models := BuildModels(providers, "http://gw.example", "http://gw.example/v1", nil, nil,
		map[string]llm.API{"anthropic": "anthropic-messages"}, notify)
	if models[0].API != "anthropic-messages" || models[0].BaseURL != "http://gw.example" {
		t.Errorf("valid override not applied: %s %s", models[0].API, models[0].BaseURL)
	}
	if len(warnings) != 0 {
		t.Errorf("unexpected warnings: %v", warnings)
	}

	models = BuildModels(providers, "http://gw.example", "http://gw.example/v1", nil, nil,
		map[string]llm.API{"google": "anthropic-messages"}, notify)
	if models[1].API != "google-generative-ai" {
		t.Errorf("unserved override must fall back to auto: %s", models[1].API)
	}
	if len(warnings) != 1 || !strings.Contains(warnings[0], "not served by the gateway") {
		t.Errorf("fallback warning missing: %v", warnings)
	}
}

func TestBuildModelsModelsDevPrecedence(t *testing.T) {
	reasoning := true
	modelsDev := ModelsDevCatalog{
		"google": {Models: map[string]ModelsDevModel{
			"gemini-2.5-pro": func() ModelsDevModel {
				m := ModelsDevModel{Name: "Gemini 2.5 Pro", Reasoning: &reasoning}
				m.Modalities.Input = []string{"text", "image", "audio"}
				m.Limit.Context = 1_048_576
				m.Limit.Output = 65_536
				return m
			}(),
		}},
	}
	models := BuildModels(testGatewayProviders(), "http://gw.example", "http://gw.example/v1", nil, modelsDev, nil, nil)
	gemini := models[1]
	if gemini.Name != "Gemini 2.5 Pro" || !gemini.Reasoning || gemini.ContextWindow != 1_048_576 || gemini.MaxTokens != 65_536 {
		t.Errorf("models.dev metadata not applied: %+v", gemini)
	}
	if strings.Join(gemini.Input, ",") != "text,image" {
		t.Errorf("unsupported modalities must be dropped: %v", gemini.Input)
	}
}

func TestCatalogKeyIdentity(t *testing.T) {
	base := Resolved{DedicatedProviders: []DedicatedProviderConfig{
		{ID: "anthropic", Enabled: true},
		{ID: "google", Enabled: true, API: "google-generative-ai"},
	}}
	key := BuildCatalogKey("http://gw.example", base)
	if !strings.HasPrefix(key, "http://gw.example ") {
		t.Errorf("key uses the origin: %q", key)
	}
	// Origin equality, not a string prefix.
	other := BuildCatalogKey("http://gw.example.evil", base)
	if key == other {
		t.Error("different origins must produce different keys")
	}
	// A changed api override changes the identity.
	withoutOverride := base
	withoutOverride.DedicatedProviders = []DedicatedProviderConfig{
		{ID: "anthropic", Enabled: true},
		{ID: "google", Enabled: true},
	}
	if BuildCatalogKey("http://gw.example", withoutOverride) == key {
		t.Error("api override must be part of the identity")
	}
	// No filter is "*".
	if !strings.Contains(BuildCatalogKey("http://gw.example", Resolved{}), " * ") {
		t.Error("empty provider list keys as *")
	}
}

func TestCacheRoundTripAndKeyGuard(t *testing.T) {
	path := filepath.Join(t.TempDir(), "aperture-cache.json")
	models := []llm.Model{{
		ID: "anthropic/claude-sonnet-5", Provider: "aperture", API: "anthropic-messages",
		BaseURL: "http://gw.example", Input: []string{"text"},
		ContextWindow: 200_000, MaxTokens: 64_000,
		RawCompat: json.RawMessage(`{"forceAdaptiveThinking": true}`),
	}}
	gateway := testGatewayProviders()
	if err := SaveCache(path, NewCache("key-1", models, gateway)); err != nil {
		t.Fatal(err)
	}
	cache, err := LoadCache(path)
	if err != nil {
		t.Fatal(err)
	}
	restored := cache.CatalogModels("key-1")
	if len(restored) != 1 {
		t.Fatalf("restored = %d", len(restored))
	}
	var compat map[string]any
	if err := json.Unmarshal(restored[0].RawCompat, &compat); err != nil || compat["forceAdaptiveThinking"] != true {
		t.Errorf("raw compat lost across the cache: %s (%v)", restored[0].RawCompat, err)
	}
	if cache.CatalogModels("other-key") != nil {
		t.Error("a snapshot for a different catalog identity must not replay")
	}
	if len(cache.Gateway) != 2 || cache.Gateway[0].ID != "anthropic" {
		t.Errorf("gateway snapshot = %+v", cache.Gateway)
	}
}

func TestFilterProviders(t *testing.T) {
	providers := testGatewayProviders()
	all := FilterProviders(providers, Resolved{})
	if len(all) != 2 {
		t.Error("an empty selection includes all providers")
	}
	filtered := FilterProviders(providers, Resolved{DedicatedProviders: []DedicatedProviderConfig{
		{ID: "google", Enabled: true},
		{ID: "anthropic", Enabled: false},
	}})
	if len(filtered) != 1 || filtered[0].ID != "google" {
		t.Errorf("filtered = %+v", filtered)
	}
}

func TestMergeCostPartialPricing(t *testing.T) {
	base := &llm.ModelCost{ModelCostRates: llm.ModelCostRates{Input: 3, Output: 15, CacheRead: 0.3, CacheWrite: 3.75}}
	merged := MergeCost(&ModelPricing{InputCacheRead: "0.00000010"}, base)
	if merged.Input != 3 || merged.Output != 15 || merged.CacheWrite != 3.75 {
		t.Errorf("partial pricing must not zero other fields: %+v", merged)
	}
	if math.Abs(merged.CacheRead-0.1) > 1e-9 {
		t.Errorf("cacheRead = %v, want 0.1", merged.CacheRead)
	}
	if got := MergeCost(nil, nil); got.Input != 0 || got.Output != 0 {
		t.Errorf("no pricing, no base: %+v", got)
	}
}
