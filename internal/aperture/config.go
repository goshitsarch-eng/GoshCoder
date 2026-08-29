// Package aperture is GoshCoder's native Go adaptation of
// @aliou/pi-ts-aperture (github.com/aliou/pi-ts-aperture, version 0.14.1),
// which routes LLM providers and connector tools through Tailscale Aperture,
// a managed AI gateway on a tailnet.
//
// The package owns the persisted configuration (pi-compatible
// extensions/aperture.json plus its content-gated migrations), the gateway
// API client, dedicated-catalog construction with layered model metadata,
// proxy routing for existing providers, and the MCP connector tools. The
// catalog integration lives in internal/llm/catalog and the commands in
// cmd/goshcoder.
package aperture

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const maxConfigBytes = 4 << 20

// RoutableAPI is a Pi API derivable from a gateway provider's compatibility
// map (extensions/shared/config/types.ts RoutableApi).
type RoutableAPI = string

// ProxiedProviderConfig selects one existing provider for proxying
// (extensions/shared/config/types.ts).
type ProxiedProviderConfig struct {
	// ID is the Aperture provider id (matches the /api/providers response).
	ID string `json:"id"`
	// Name is an optional display name, persisted by the settings flow.
	Name string `json:"name,omitempty"`
	// Enabled defaults to true; false keeps per-provider settings without
	// proxying the provider.
	Enabled *bool `json:"enabled,omitempty"`
	// ShouldCheckGatewayModels warns when configured local models are missing
	// from the Aperture gateway.
	ShouldCheckGatewayModels bool `json:"shouldCheckGatewayModels,omitempty"`
	// KeepGatewayModelsOnly registers only models the gateway actually serves.
	KeepGatewayModelsOnly bool `json:"keepGatewayModelsOnly,omitempty"`
	// API routes this provider's models through a specific API instead of the
	// auto-detected one. Validated against the gateway compatibility map on
	// every sync; falls back to auto with a warning when not served.
	API RoutableAPI `json:"api,omitempty"`
}

// IsEnabled applies the enabled-unless-disabled default.
func (p ProxiedProviderConfig) IsEnabled() bool { return p.Enabled == nil || *p.Enabled }

// DedicatedProviderConfig filters one gateway provider in dedicated mode.
type DedicatedProviderConfig struct {
	ID   string `json:"id"`
	Name string `json:"name,omitempty"`
	// Enabled includes this provider's models in the dedicated provider.
	Enabled bool `json:"enabled"`
	// API overrides the auto-picked routing API; part of the catalog key so a
	// cached catalog under a different api is never replayed.
	API RoutableAPI `json:"api,omitempty"`
}

// PinnedConnectorTool is one gateway MCP tool registered as a first-class
// tool instead of being reached through the discovery meta-tools. Matching is
// by ToolName; ConnectorID (the tool name prefix before the first "_") is
// stored for traceability. Stale entries are silently skipped on
// registration.
type PinnedConnectorTool struct {
	ConnectorID string `json:"connectorId"`
	ToolName    string `json:"toolName"`
}

// ConnectorsConfig gates the connector tools feature.
type ConnectorsConfig struct {
	// Enabled is the master switch; when false nothing registers.
	Enabled bool `json:"enabled,omitempty"`
	// PinnedTools are registered as first-class tools (each adds its full
	// schema to the system prompt).
	PinnedTools []PinnedConnectorTool `json:"pinnedTools,omitempty"`
	// DiscoveryTools registers the list/search/describe/call meta-tools.
	// Defaults to true and is decorrelated from pinning.
	DiscoveryTools *bool `json:"discoveryTools,omitempty"`
}

// Config is the persisted aperture.json shape (extensions/shared/config).
// The legacy pre-v0.6 fields exist only so migrations can transform them out
// of older files, mirroring migration/legacy.ts.
type Config struct {
	// Schema is the JSON Schema URL pi's config loader stamps; preserved on
	// round-trips so pi tooling keeps validating the file.
	Schema string `json:"$schema,omitempty"`
	// Version is the config schema version stamped by content-gated migrations.
	Version        string `json:"version,omitempty"`
	BaseURL        string `json:"baseUrl,omitempty"`
	OnboardingDone *bool  `json:"onboardingDone,omitempty"`
	Onboarding     *struct {
		Enabled *bool `json:"enabled,omitempty"`
	} `json:"onboarding,omitempty"`
	Proxy *struct {
		Enabled           *bool                   `json:"enabled,omitempty"`
		UpstreamProviders []ProxiedProviderConfig `json:"upstreamProviders"`
	} `json:"proxy,omitempty"`
	Dedicated *struct {
		Enabled   *bool                     `json:"enabled,omitempty"`
		Providers []DedicatedProviderConfig `json:"providers"`
		// CachedModels is legacy (pre-v0.8) state migrated away by 003.
		CachedModels []json.RawMessage `json:"cachedModels,omitempty"`
	} `json:"dedicated,omitempty"`
	Connectors *ConnectorsConfig `json:"connectors,omitempty"`

	// Legacy pre-v0.6 fields (migration/legacy.ts). Never written back.
	LegacyMode               string   `json:"mode,omitempty"`
	LegacyProviders          []string `json:"providers,omitempty"`
	LegacyCheckGatewayModels []string `json:"checkGatewayModels,omitempty"`
	LegacyApertureProvider   *bool    `json:"apertureProvider,omitempty"`
}

// Resolved is the fully-defaulted view of a Config
// (extensions/shared/config/defaults.ts DEFAULT_CONFIG).
type Resolved struct {
	BaseURL            string
	OnboardingDone     bool
	OnboardingEnabled  bool
	ProxyEnabled       bool
	UpstreamProviders  []ProxiedProviderConfig
	DedicatedEnabled   bool
	DedicatedProviders []DedicatedProviderConfig
	ConnectorsEnabled  bool
	PinnedTools        []PinnedConnectorTool
	DiscoveryTools     bool
}

// Resolve applies the defaults: dedicated on, proxy off, connectors off,
// discovery tools on, onboarding enabled until done.
func (c Config) Resolve() Resolved {
	resolved := Resolved{
		BaseURL:          c.BaseURL,
		DedicatedEnabled: true,
		DiscoveryTools:   true,
	}
	if c.OnboardingDone != nil {
		resolved.OnboardingDone = *c.OnboardingDone
	}
	// isOnboardingExtensionEnabled: an explicit onboarding.enabled wins;
	// otherwise onboarding stays active until onboardingDone is true.
	resolved.OnboardingEnabled = !resolved.OnboardingDone
	if c.Onboarding != nil && c.Onboarding.Enabled != nil {
		resolved.OnboardingEnabled = *c.Onboarding.Enabled
	}
	if c.Proxy != nil {
		if c.Proxy.Enabled != nil {
			resolved.ProxyEnabled = *c.Proxy.Enabled
		}
		resolved.UpstreamProviders = append([]ProxiedProviderConfig(nil), c.Proxy.UpstreamProviders...)
	}
	if c.Dedicated != nil {
		if c.Dedicated.Enabled != nil {
			resolved.DedicatedEnabled = *c.Dedicated.Enabled
		}
		resolved.DedicatedProviders = append([]DedicatedProviderConfig(nil), c.Dedicated.Providers...)
	}
	if c.Connectors != nil {
		resolved.ConnectorsEnabled = c.Connectors.Enabled
		resolved.PinnedTools = append([]PinnedConnectorTool(nil), c.Connectors.PinnedTools...)
		if c.Connectors.DiscoveryTools != nil {
			resolved.DiscoveryTools = *c.Connectors.DiscoveryTools
		}
	}
	return resolved
}

// EnabledUpstreamProviders returns the proxy providers not explicitly
// disabled.
func (r Resolved) EnabledUpstreamProviders() []ProxiedProviderConfig {
	var out []ProxiedProviderConfig
	for _, provider := range r.UpstreamProviders {
		if provider.IsEnabled() {
			out = append(out, provider)
		}
	}
	return out
}

func boolPtr(v bool) *bool { return &v }

// migrate runs the three content-gated migrations in order
// (migration/001..003). Each stamps Version with the release that shipped it.
// Returns whether anything changed, so Load can report a state worth saving.
func migrate(c *Config) bool {
	changed := false
	// 001-legacy-to-v0-6: providers/checkGatewayModels -> proxy, and
	// apertureProvider -> dedicated.enabled; a configured baseUrl from before
	// onboarding existed counts as onboarded.
	if c.LegacyProviders != nil || c.LegacyCheckGatewayModels != nil || c.LegacyApertureProvider != nil ||
		(c.OnboardingDone == nil && c.BaseURL != "") {
		if c.LegacyProviders != nil || c.LegacyCheckGatewayModels != nil {
			checked := map[string]bool{}
			for _, id := range c.LegacyCheckGatewayModels {
				checked[id] = true
			}
			upstream := make([]ProxiedProviderConfig, 0, len(c.LegacyProviders))
			for _, id := range c.LegacyProviders {
				upstream = append(upstream, ProxiedProviderConfig{ID: id, ShouldCheckGatewayModels: checked[id]})
			}
			ensureProxy(c)
			c.Proxy.Enabled = boolPtr(true)
			c.Proxy.UpstreamProviders = upstream
			c.LegacyProviders, c.LegacyCheckGatewayModels = nil, nil
		}
		if c.LegacyApertureProvider != nil {
			ensureDedicated(c)
			c.Dedicated.Enabled = c.LegacyApertureProvider
			c.LegacyApertureProvider = nil
		}
		if c.OnboardingDone == nil && c.BaseURL != "" {
			c.OnboardingDone = boolPtr(true)
		}
		c.Version = "0.6.0"
		changed = true
	}
	// 002-mode-to-capabilities: mode "proxy"/"dedicated" -> the two toggles.
	if c.LegacyMode != "" {
		switch c.LegacyMode {
		case "proxy":
			ensureProxy(c)
			ensureDedicated(c)
			c.Proxy.Enabled = boolPtr(true)
			c.Dedicated.Enabled = boolPtr(false)
		case "dedicated":
			ensureProxy(c)
			ensureDedicated(c)
			c.Dedicated.Enabled = boolPtr(true)
			c.Proxy.Enabled = boolPtr(false)
		}
		c.LegacyMode = ""
		c.Version = "0.7.0"
		changed = true
	}
	// 003-normalize-capabilities: make both capability blocks explicit and
	// drop the legacy dedicated.cachedModels state.
	if c.Proxy == nil || c.Proxy.Enabled == nil || c.Proxy.UpstreamProviders == nil ||
		c.Dedicated == nil || c.Dedicated.Enabled == nil || c.Dedicated.Providers == nil ||
		(c.Dedicated != nil && c.Dedicated.CachedModels != nil) {
		ensureProxy(c)
		ensureDedicated(c)
		if c.Proxy.Enabled == nil {
			c.Proxy.Enabled = boolPtr(false)
		}
		if c.Proxy.UpstreamProviders == nil {
			c.Proxy.UpstreamProviders = []ProxiedProviderConfig{}
		}
		if c.Dedicated.Enabled == nil {
			c.Dedicated.Enabled = boolPtr(true)
		}
		if c.Dedicated.Providers == nil {
			c.Dedicated.Providers = []DedicatedProviderConfig{}
		}
		c.Dedicated.CachedModels = nil
		c.Version = "0.8.0"
		changed = true
	}
	return changed
}

func ensureProxy(c *Config) {
	if c.Proxy == nil {
		c.Proxy = &struct {
			Enabled           *bool                   `json:"enabled,omitempty"`
			UpstreamProviders []ProxiedProviderConfig `json:"upstreamProviders"`
		}{}
	}
}

func ensureDedicated(c *Config) {
	if c.Dedicated == nil {
		c.Dedicated = &struct {
			Enabled      *bool                     `json:"enabled,omitempty"`
			Providers    []DedicatedProviderConfig `json:"providers"`
			CachedModels []json.RawMessage         `json:"cachedModels,omitempty"`
		}{}
	}
}

// Load reads and migrates the config. Missing files report os.ErrNotExist so
// callers can distinguish unconfigured from malformed. Migration results are
// applied in memory only; Save persists them.
func Load(path string) (Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return Config{}, err
	}
	if !info.Mode().IsRegular() {
		return Config{}, errors.New("aperture config is not a regular file")
	}
	if info.Size() > maxConfigBytes {
		return Config{}, fmt.Errorf("aperture config exceeds %d bytes", maxConfigBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxConfigBytes+1))
	if err != nil {
		return Config{}, err
	}
	if len(data) > maxConfigBytes {
		return Config{}, fmt.Errorf("aperture config exceeds %d bytes", maxConfigBytes)
	}
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, fmt.Errorf("invalid aperture config: %w", err)
	}
	migrate(&config)
	return config, nil
}

// Save atomically publishes the migrated config with user-only permissions.
func Save(path string, config Config) error {
	migrate(&config)
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if len(data) > maxConfigBytes {
		return fmt.Errorf("aperture config exceeds %d bytes", maxConfigBytes)
	}
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".aperture-*.tmp")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}
