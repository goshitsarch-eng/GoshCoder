package aperture

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeConfigFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "aperture.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestResolveDefaults(t *testing.T) {
	resolved := Config{}.Resolve()
	if !resolved.DedicatedEnabled {
		t.Error("dedicated defaults to enabled")
	}
	if resolved.ProxyEnabled {
		t.Error("proxy defaults to disabled")
	}
	if resolved.ConnectorsEnabled {
		t.Error("connectors default to disabled")
	}
	if !resolved.DiscoveryTools {
		t.Error("discovery tools default to enabled")
	}
	if !resolved.OnboardingEnabled {
		t.Error("onboarding is active until done")
	}
	done := true
	if (Config{OnboardingDone: &done}).Resolve().OnboardingEnabled {
		t.Error("onboardingDone disables onboarding by default")
	}
	enabled := true
	config := Config{OnboardingDone: &done, Onboarding: &struct {
		Enabled *bool `json:"enabled,omitempty"`
	}{Enabled: &enabled}}
	if !config.Resolve().OnboardingEnabled {
		t.Error("an explicit onboarding.enabled wins")
	}
}

func TestMigrationLegacyToV06(t *testing.T) {
	path := writeConfigFile(t, `{
		"baseUrl": "http://gw.example",
		"providers": ["anthropic", "openai"],
		"checkGatewayModels": ["anthropic"]
	}`)
	config, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	resolved := config.Resolve()
	if !resolved.ProxyEnabled {
		t.Error("001 enables proxy for legacy provider lists")
	}
	if len(resolved.UpstreamProviders) != 2 {
		t.Fatalf("upstream providers = %d", len(resolved.UpstreamProviders))
	}
	if !resolved.UpstreamProviders[0].ShouldCheckGatewayModels || resolved.UpstreamProviders[1].ShouldCheckGatewayModels {
		t.Error("checkGatewayModels selection must carry over per provider")
	}
	if !resolved.OnboardingDone {
		t.Error("a configured baseUrl from before onboarding counts as onboarded")
	}
	if config.LegacyProviders != nil || config.LegacyCheckGatewayModels != nil {
		t.Error("legacy fields must be migrated away")
	}
}

func TestMigrationModeToCapabilities(t *testing.T) {
	path := writeConfigFile(t, `{"baseUrl": "http://gw.example", "mode": "proxy"}`)
	config, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	resolved := config.Resolve()
	if !resolved.ProxyEnabled || resolved.DedicatedEnabled {
		t.Errorf("mode=proxy: proxy=%v dedicated=%v", resolved.ProxyEnabled, resolved.DedicatedEnabled)
	}

	path = writeConfigFile(t, `{"baseUrl": "http://gw.example", "mode": "dedicated", "apertureProvider": false}`)
	config, err = Load(path)
	if err != nil {
		t.Fatal(err)
	}
	resolved = config.Resolve()
	// 001 maps apertureProvider first; 002's explicit mode then wins.
	if resolved.ProxyEnabled || !resolved.DedicatedEnabled {
		t.Errorf("mode=dedicated: proxy=%v dedicated=%v", resolved.ProxyEnabled, resolved.DedicatedEnabled)
	}
}

func TestMigrationNormalizeCapabilities(t *testing.T) {
	path := writeConfigFile(t, `{
		"baseUrl": "http://gw.example",
		"onboardingDone": true,
		"dedicated": {"enabled": true, "cachedModels": [{"id": "legacy"}]}
	}`)
	config, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if config.Dedicated.CachedModels != nil {
		t.Error("003 drops legacy dedicated.cachedModels")
	}
	if config.Proxy == nil || config.Proxy.Enabled == nil || *config.Proxy.Enabled {
		t.Error("003 makes proxy.enabled explicit (false)")
	}
	if config.Version != "0.8.0" {
		t.Errorf("version stamp = %q", config.Version)
	}
}

func TestSaveRoundTripDropsLegacyFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "aperture.json")
	config := Config{BaseURL: "http://gw.example", LegacyMode: "proxy", LegacyProviders: []string{"anthropic"}}
	if err := Save(path, config); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"mode", "providers", "checkGatewayModels", "apertureProvider"} {
		if _, present := decoded[forbidden]; present {
			t.Errorf("saved config still carries legacy top-level field %q:\n%s", forbidden, raw)
		}
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	resolved := loaded.Resolve()
	if !resolved.ProxyEnabled || len(resolved.UpstreamProviders) != 1 {
		t.Errorf("migrated round trip lost proxy state: %+v", resolved)
	}
}

func TestLoadRejectsMalformed(t *testing.T) {
	path := writeConfigFile(t, `{"baseUrl": [1]}`)
	if _, err := Load(path); err == nil {
		t.Fatal("malformed config must fail Load")
	}
	if _, err := Load(filepath.Join(t.TempDir(), "missing.json")); !os.IsNotExist(err) {
		t.Fatalf("missing file must report os.ErrNotExist, got %v", err)
	}
}

func TestSchemaFieldSurvivesRoundTrip(t *testing.T) {
	path := writeConfigFile(t, `{"$schema": "https://pi.dev/schema.json", "baseUrl": "http://gw.example", "onboardingDone": true}`)
	config, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := Save(path, config); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(path)
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded["$schema"] != "https://pi.dev/schema.json" {
		t.Errorf("$schema lost on round trip: %v", decoded["$schema"])
	}
}
