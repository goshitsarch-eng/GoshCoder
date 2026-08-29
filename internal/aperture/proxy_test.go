package aperture

import (
	"strings"
	"testing"

	"goshcoder/internal/llm"
)

func testResolvedProxy(providers ...ProxiedProviderConfig) Resolved {
	return Resolved{
		BaseURL:           "http://gw.example",
		ProxyEnabled:      true,
		UpstreamProviders: providers,
	}
}

func testNativeInfo(t *testing.T) func(string) (llm.API, string, []string, bool) {
	t.Helper()
	return func(providerID string) (llm.API, string, []string, bool) {
		switch providerID {
		case "anthropic":
			return "anthropic-messages", "https://api.anthropic.com", []string{"claude-sonnet-5", "claude-haiku-4"}, true
		case "zai":
			return "openai-completions", "https://api.z.ai/api/coding/paas/v4", []string{"glm-5"}, true
		case "openai":
			return "openai-responses", "https://api.openai.com/v1", []string{"gpt-5"}, true
		default:
			return "", "", nil, false
		}
	}
}

func testSnapshots() []GatewaySnapshot {
	return []GatewaySnapshot{
		{ID: "anthropic", Models: []string{"claude-sonnet-5"},
			Compatibility: map[string]bool{"anthropic_messages": true, "openai_chat": true}},
		{ID: "zai", Models: []string{"glm-5"}, Compatibility: map[string]bool{"openai_chat": true}},
		{ID: "openai", Models: []string{"gpt-5"},
			Compatibility: map[string]bool{"openai_responses": true}, RequiresClientAuth: true},
	}
}

func TestPlanRoutes(t *testing.T) {
	resolved := testResolvedProxy(
		ProxiedProviderConfig{ID: "anthropic", KeepGatewayModelsOnly: true},
		ProxiedProviderConfig{ID: "zai"},
		ProxiedProviderConfig{ID: "openai"},
		ProxiedProviderConfig{ID: "aperture"},
		ProxiedProviderConfig{ID: "unknown-local"},
	)
	routes := Plan(resolved, testSnapshots(), testNativeInfo(t))
	if len(routes) != 3 {
		t.Fatalf("routes = %d: %v", len(routes), routes)
	}

	anthropic := routes["anthropic"]
	if anthropic.BaseURL != "http://gw.example" {
		t.Errorf("anthropic-messages routes to the gateway root: %q", anthropic.BaseURL)
	}
	if anthropic.ServedModelIDs == nil || !anthropic.ServedModelIDs["claude-sonnet-5"] || anthropic.ServedModelIDs["claude-haiku-4"] {
		t.Errorf("keepGatewayModelsOnly filter = %v", anthropic.ServedModelIDs)
	}
	if anthropic.Passthrough {
		t.Error("anthropic is not passthrough")
	}

	// Z.ai's upstream ends in /v4, so the OpenAI-SDK path needs the root.
	if routes["zai"].BaseURL != "http://gw.example" {
		t.Errorf("zai base URL = %q", routes["zai"].BaseURL)
	}
	if !routes["openai"].Passthrough {
		t.Error("requires_client_auth marks passthrough")
	}
	if routes["openai"].BaseURL != "http://gw.example/v1" {
		t.Errorf("openai base URL = %q", routes["openai"].BaseURL)
	}
}

func TestPlanSkipsDisabledAndUnconfigured(t *testing.T) {
	disabled := false
	resolved := testResolvedProxy(ProxiedProviderConfig{ID: "anthropic", Enabled: &disabled})
	if routes := Plan(resolved, testSnapshots(), testNativeInfo(t)); len(routes) != 0 {
		t.Errorf("disabled provider must not route: %v", routes)
	}
	resolved = testResolvedProxy(ProxiedProviderConfig{ID: "anthropic"})
	resolved.ProxyEnabled = false
	if routes := Plan(resolved, testSnapshots(), testNativeInfo(t)); routes != nil {
		t.Errorf("proxy off must plan nothing: %v", routes)
	}
	resolved = testResolvedProxy(ProxiedProviderConfig{ID: "anthropic"})
	resolved.BaseURL = ""
	if routes := Plan(resolved, testSnapshots(), testNativeInfo(t)); routes != nil {
		t.Errorf("no base URL must plan nothing: %v", routes)
	}
}

func TestPlanAPIOverride(t *testing.T) {
	resolved := testResolvedProxy(ProxiedProviderConfig{ID: "anthropic", API: "openai-completions"})
	routes := Plan(resolved, testSnapshots(), testNativeInfo(t))
	route := routes["anthropic"]
	if !route.APIOverridden || route.API != "openai-completions" {
		t.Errorf("valid override not applied: %+v", route)
	}
	if route.BaseURL != "http://gw.example/v1" {
		t.Errorf("override changes routing: %q", route.BaseURL)
	}

	// An override the gateway does not serve falls back to the native api.
	resolved = testResolvedProxy(ProxiedProviderConfig{ID: "zai", API: "anthropic-messages"})
	route = Plan(resolved, testSnapshots(), testNativeInfo(t))["zai"]
	if route.APIOverridden || route.API != "openai-completions" {
		t.Errorf("unserved override must fall back: %+v", route)
	}
}

func TestPlanKeepGatewayModelsOnlySkipsFullyMissing(t *testing.T) {
	snapshots := []GatewaySnapshot{{ID: "anthropic", Models: []string{"other-model"}}}
	resolved := testResolvedProxy(ProxiedProviderConfig{ID: "anthropic", KeepGatewayModelsOnly: true})
	if routes := Plan(resolved, snapshots, testNativeInfo(t)); len(routes) != 0 {
		t.Errorf("a provider sharing no model with the gateway is skipped: %v", routes)
	}
}

func TestMissingModelsSummary(t *testing.T) {
	resolved := testResolvedProxy(
		ProxiedProviderConfig{ID: "anthropic", ShouldCheckGatewayModels: true},
		ProxiedProviderConfig{ID: "zai"},
	)
	gateway := []GatewayProvider{
		{ID: "anthropic", Models: []string{"claude-sonnet-5"}},
		{ID: "zai", Models: []string{}},
	}
	local := []llm.Model{
		{ID: "claude-sonnet-5", Provider: "anthropic"},
		{ID: "claude-haiku-4", Provider: "anthropic"},
		{ID: "claude-1", Provider: "anthropic"},
		{ID: "claude-2", Provider: "anthropic"},
		{ID: "claude-3", Provider: "anthropic"},
		{ID: "claude-4", Provider: "anthropic"},
		{ID: "claude-5", Provider: "anthropic"},
		// zai has no check flag, so its missing model must not appear.
		{ID: "glm-5", Provider: "zai"},
	}
	summary := MissingModelsSummary(resolved, gateway, local)
	if summary == "" {
		t.Fatal("expected a summary")
	}
	if strings.Contains(summary, "glm-5") {
		t.Error("unchecked provider leaked into the summary")
	}
	if !strings.Contains(summary, "1 more") {
		t.Errorf("per-provider cap missing: %s", summary)
	}
	if strings.Contains(summary, "claude-sonnet-5") {
		t.Error("served model reported missing")
	}

	if MissingModelsSummary(resolved, nil, local) != "" {
		t.Error("no gateway data yields no summary")
	}
	allServed := []GatewayProvider{{ID: "anthropic", Models: []string{
		"claude-sonnet-5", "claude-haiku-4", "claude-1", "claude-2", "claude-3", "claude-4", "claude-5"}}}
	if got := MissingModelsSummary(resolved, allServed, local); got != "" {
		t.Errorf("all served yields no summary, got %q", got)
	}
}

func TestMapProxyProviders(t *testing.T) {
	local := []llm.Model{
		{ID: "claude-sonnet-5", Provider: "anthropic"},
		{ID: "gpt-5", Provider: "openai"},
		{ID: "anthropic/claude-sonnet-5", Provider: "aperture"},
		{ID: "local-only", Provider: "ollama"},
	}
	gateway := []GatewayProvider{
		{ID: "openai", Name: "OpenAI"},
		{ID: "anthropic", Name: "Anthropic"},
	}
	existing := []ProxiedProviderConfig{{ID: "anthropic", ShouldCheckGatewayModels: false, KeepGatewayModelsOnly: true, API: "openai-completions"}}
	mapped := MapProxyProviders(local, gateway, existing)
	if len(mapped) != 2 || mapped[0].ID != "anthropic" || mapped[1].ID != "openai" {
		t.Fatalf("mapped = %+v", mapped)
	}
	if !mapped[0].Enabled || mapped[0].ShouldCheckGatewayModels || !mapped[0].KeepGatewayModelsOnly || mapped[0].API != "openai-completions" {
		t.Errorf("existing settings not preserved: %+v", mapped[0])
	}
	if mapped[1].Enabled || !mapped[1].ShouldCheckGatewayModels {
		t.Errorf("unconfigured provider defaults: %+v", mapped[1])
	}
}

func TestMapDedicatedProviders(t *testing.T) {
	gateway := []GatewayProvider{{ID: "anthropic", Name: "Anthropic"}, {ID: "google", Name: "Google"}}
	existing := []DedicatedProviderConfig{{ID: "google", Enabled: false, API: "google-generative-ai"}}
	mapped := MapDedicatedProviders(gateway, existing)
	if len(mapped) != 2 {
		t.Fatalf("mapped = %+v", mapped)
	}
	if !mapped[0].Enabled {
		t.Error("new providers default to enabled")
	}
	if mapped[1].Enabled || mapped[1].API != "google-generative-ai" {
		t.Errorf("existing state not preserved: %+v", mapped[1])
	}
}
