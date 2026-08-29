package aperture

// Proxy-mode routing (extensions/aperture/proxy/runtime.ts) and the provider
// mapping helpers shared with onboarding and settings
// (extensions/shared/provider-mapping.ts).
//
// pi re-registers each proxied provider with a wrapped model list; GoshCoder
// rebuilds provider model lists on every catalog read instead, so the proxy
// is expressed as a Plan the catalog applies: per-provider base URL, api
// override, gateway model filter, and passthrough auth. The upstream base URL
// used for the gateway-root-vs-/v1 inference always comes from the immutable
// builtin catalog, which is why the original's first-seen-provider cache has
// no equivalent here.

import (
	"fmt"
	"sort"
	"strings"

	"goshcoder/internal/llm"
)

const maxMissingModelsPerProvider = 5

// ProxyRoute is the rewrite the catalog applies to one proxied provider.
type ProxyRoute struct {
	ProviderID string
	// API is the routing api: the provider's own, or a validated override.
	API llm.API
	// APIOverridden reports whether API replaces the provider's native api.
	APIOverridden bool
	// BaseURL is the per-API gateway URL models are rewritten to.
	BaseURL string
	// ServedModelIDs filters the model list when keepGatewayModelsOnly is on;
	// nil keeps every local model.
	ServedModelIDs map[string]bool
	// Passthrough providers keep native auth so the client sends a real
	// credential the gateway forwards; others get gateway-injected auth.
	Passthrough bool
}

// Plan resolves the proxy routes for the current config against a gateway
// snapshot. nativeAPI and nativeBaseURL describe the provider before any
// rewrite (the builtin catalog view). Providers absent from the local catalog
// or, under keepGatewayModelsOnly, sharing no model with the gateway are
// skipped, matching the original's sync loop.
func Plan(resolved Resolved, gateway []GatewaySnapshot, nativeInfo func(providerID string) (api llm.API, baseURL string, modelIDs []string, ok bool)) map[string]ProxyRoute {
	if !resolved.ProxyEnabled || resolved.BaseURL == "" {
		return nil
	}
	gatewayURL := GatewayURL(resolved.BaseURL)
	baseURL := ProviderBaseURL(resolved.BaseURL)
	if gatewayURL == "" || baseURL == "" {
		return nil
	}
	snapshotByID := map[string]GatewaySnapshot{}
	for _, provider := range gateway {
		snapshotByID[provider.ID] = provider
	}

	routes := map[string]ProxyRoute{}
	for _, configured := range resolved.EnabledUpstreamProviders() {
		if configured.ID == DedicatedProviderID {
			continue
		}
		nativeAPI, upstreamBaseURL, modelIDs, ok := nativeInfo(configured.ID)
		if !ok || len(modelIDs) == 0 {
			continue
		}
		snapshot, hasSnapshot := snapshotByID[configured.ID]

		api := nativeAPI
		overridden := false
		if configured.API != "" && hasSnapshot && IsSelectableAPI(configured.API, snapshot.Compatibility) {
			api, overridden = configured.API, true
		}

		var served map[string]bool
		if configured.KeepGatewayModelsOnly && hasSnapshot {
			served = map[string]bool{}
			for _, id := range snapshot.Models {
				served[id] = true
			}
			any := false
			for _, id := range modelIDs {
				if served[id] {
					any = true
					break
				}
			}
			if !any {
				continue
			}
		}

		routes[configured.ID] = ProxyRoute{
			ProviderID:     configured.ID,
			API:            api,
			APIOverridden:  overridden,
			BaseURL:        BaseURLForAPI(api, gatewayURL, baseURL, upstreamBaseURL),
			ServedModelIDs: served,
			Passthrough:    hasSnapshot && snapshot.RequiresClientAuth,
		}
	}
	return routes
}

// proxyOverrideWarnings reports api overrides the gateway no longer serves,
// which fall back to the provider's own api.
func proxyOverrideWarnings(resolved Resolved, gateway []GatewayProvider) []string {
	compatByID := map[string]map[string]bool{}
	for _, provider := range gateway {
		compatByID[provider.ID] = provider.Compatibility
	}
	var warnings []string
	for _, configured := range resolved.EnabledUpstreamProviders() {
		if configured.API == "" {
			continue
		}
		compatibility, ok := compatByID[configured.ID]
		if !ok {
			continue
		}
		if !IsSelectableAPI(configured.API, compatibility) {
			warnings = append(warnings, fmt.Sprintf("[aperture] api override %q for proxied provider %s is not served by the gateway; falling back to the provider's own api.", configured.API, configured.ID))
		}
	}
	return warnings
}

// MissingModelsSummary warns when configured local models of checked proxied
// providers are missing from the gateway catalog, at most five model ids per
// provider (checkMissingModels).
func MissingModelsSummary(resolved Resolved, gateway []GatewayProvider, localModels []llm.Model) string {
	checked := map[string]bool{}
	for _, provider := range resolved.EnabledUpstreamProviders() {
		if provider.ShouldCheckGatewayModels {
			checked[provider.ID] = true
		}
	}
	if len(checked) == 0 || len(gateway) == 0 {
		return ""
	}
	servedByProvider := map[string]map[string]bool{}
	for _, provider := range gateway {
		served := map[string]bool{}
		for _, id := range provider.Models {
			served[id] = true
		}
		servedByProvider[provider.ID] = served
	}

	missingByProvider := map[string][]string{}
	for _, model := range localModels {
		if !checked[model.Provider] {
			continue
		}
		if servedByProvider[model.Provider][model.ID] {
			continue
		}
		missingByProvider[model.Provider] = append(missingByProvider[model.Provider], model.ID)
	}
	if len(missingByProvider) == 0 {
		return ""
	}

	providers := make([]string, 0, len(missingByProvider))
	for provider := range missingByProvider {
		providers = append(providers, provider)
	}
	sort.Strings(providers)
	parts := make([]string, 0, len(providers))
	for _, provider := range providers {
		models := missingByProvider[provider]
		shown := models
		more := ""
		if len(models) > maxMissingModelsPerProvider {
			shown = models[:maxMissingModelsPerProvider]
			more = fmt.Sprintf(", %d more", len(models)-maxMissingModelsPerProvider)
		}
		parts = append(parts, provider+": "+strings.Join(shown, ", ")+more)
	}
	return "[aperture] models not available on gateway: " + strings.Join(parts, "; ") + ". Add them to the gateway configuration."
}

// MappedProxyProvider is one row of the onboarding/settings proxy selection.
type MappedProxyProvider struct {
	ID                       string
	Name                     string
	Enabled                  bool
	ShouldCheckGatewayModels bool
	KeepGatewayModelsOnly    bool
	API                      RoutableAPI
}

// MapProxyProviders matches local providers against gateway providers by id
// (the /api/providers endpoint already reflects grant scope), preserving any
// existing per-provider settings (provider-mapping.ts mapProxyProviders).
func MapProxyProviders(localModels []llm.Model, gateway []GatewayProvider, existing []ProxiedProviderConfig) []MappedProxyProvider {
	names := map[string]string{}
	gatewayIDs := map[string]bool{}
	for _, provider := range gateway {
		names[provider.ID] = provider.Name
		gatewayIDs[provider.ID] = true
	}
	existingByID := map[string]ProxiedProviderConfig{}
	for _, provider := range existing {
		existingByID[provider.ID] = provider
	}

	seen := map[string]bool{}
	var ids []string
	for _, model := range localModels {
		if model.Provider == DedicatedProviderID || seen[model.Provider] || !gatewayIDs[model.Provider] {
			continue
		}
		seen[model.Provider] = true
		ids = append(ids, model.Provider)
	}
	sort.Strings(ids)

	out := make([]MappedProxyProvider, 0, len(ids))
	for _, id := range ids {
		entry, configured := existingByID[id]
		mapped := MappedProxyProvider{
			ID:                       id,
			Name:                     names[id],
			Enabled:                  configured && entry.IsEnabled(),
			ShouldCheckGatewayModels: true,
			KeepGatewayModelsOnly:    false,
			API:                      entry.API,
		}
		if configured {
			mapped.ShouldCheckGatewayModels = entry.ShouldCheckGatewayModels
			mapped.KeepGatewayModelsOnly = entry.KeepGatewayModelsOnly
		}
		out = append(out, mapped)
	}
	return out
}

// MapDedicatedProviders maps gateway providers onto the dedicated selection,
// preserving existing enabled state and api overrides. Providers absent from
// the existing config default to enabled.
func MapDedicatedProviders(gateway []GatewayProvider, existing []DedicatedProviderConfig) []DedicatedProviderConfig {
	existingByID := map[string]DedicatedProviderConfig{}
	for _, provider := range existing {
		existingByID[provider.ID] = provider
	}
	out := make([]DedicatedProviderConfig, 0, len(gateway))
	for _, provider := range gateway {
		enabled := true
		api := RoutableAPI("")
		if entry, ok := existingByID[provider.ID]; ok {
			enabled = entry.Enabled
			api = entry.API
		}
		out = append(out, DedicatedProviderConfig{ID: provider.ID, Name: provider.Name, Enabled: enabled, API: api})
	}
	return out
}
