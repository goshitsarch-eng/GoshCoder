package catalog

// Native @aliou/pi-ts-aperture integration: the dedicated "aperture"
// provider's model list and the proxy rewrites applied to existing
// providers.
//
// pi's extension registers and re-registers wrapped Provider objects at
// runtime; GoshCoder's catalog is rebuilt on every read, so the equivalent is
// computed here from extensions/aperture.json and the synchronized
// aperture-cache.json snapshot. Reading files per catalog construction
// matches the omniroute integration; the state is computed once per Catalog
// instance.

import (
	"os"

	"goshcoder/internal/aperture"
	goshconfig "goshcoder/internal/config"
	"goshcoder/internal/llm"
)

// ApertureState is the resolved gateway routing for one catalog view.
type ApertureState struct {
	// Configured is true when aperture.json exists and carries a gateway URL.
	Configured bool
	Resolved   aperture.Resolved
	// DedicatedModels is the cached dedicated catalog, empty until the first
	// networked sync or when the cache identity no longer matches the config.
	DedicatedModels []llm.Model
	// Routes are the proxy rewrites keyed by provider id.
	Routes map[string]aperture.ProxyRoute
}

// loadApertureState reads and resolves the gateway routing state. It is
// cached on the Catalog so one Providers() enumeration reads the config
// files once.
func (c *Catalog) loadApertureState() *ApertureState {
	c.apertureOnce.Do(func() {
		c.apertureState = buildApertureState()
	})
	return c.apertureState
}

func buildApertureState() *ApertureState {
	state := &ApertureState{}

	config, err := aperture.Load(goshconfig.AperturePath())
	if err != nil {
		// Unconfigured and malformed both leave the catalog untouched; the
		// /aperture command surfaces malformed files explicitly.
		return state
	}
	resolved := config.Resolve()
	if resolved.BaseURL == "" {
		return state
	}
	state.Configured = true
	state.Resolved = resolved

	cache, cacheErr := aperture.LoadCache(goshconfig.ApertureCachePath())
	if cacheErr != nil && !os.IsNotExist(cacheErr) {
		cache = aperture.Cache{}
	}

	if resolved.DedicatedEnabled {
		gatewayURL := aperture.GatewayURL(resolved.BaseURL)
		state.DedicatedModels = cache.CatalogModels(aperture.BuildCatalogKey(gatewayURL, resolved))
	}

	if resolved.ProxyEnabled {
		state.Routes = aperture.Plan(resolved, cache.Gateway, func(providerID string) (llm.API, string, []string, bool) {
			models := builtin.models[providerID]
			if len(models) == 0 {
				return "", "", nil, false
			}
			ids := make([]string, 0, len(models))
			api := llm.API("")
			baseURL := ""
			for id, model := range models {
				ids = append(ids, id)
				if api == "" {
					api = model.API
					baseURL = model.BaseURL
				}
			}
			if config, ok := builtinProviderConfigs[providerID]; ok && config.baseURL != "" {
				baseURL = config.baseURL
			}
			return api, baseURL, ids, true
		})
	}
	return state
}

// ApertureState exposes the resolved gateway routing, for the request path
// (model-id qualification, provenance headers) and diagnostics.
func (c *Catalog) ApertureState() *ApertureState {
	return c.loadApertureState()
}

// apertureDedicatedModels returns the dedicated provider's models, or nil
// when the gateway or the dedicated capability is unconfigured.
func (c *Catalog) apertureDedicatedModels() ([]llm.Model, string) {
	state := c.loadApertureState()
	if !state.Configured || !state.Resolved.DedicatedEnabled {
		return nil, ""
	}
	return state.DedicatedModels, aperture.ProviderBaseURL(state.Resolved.BaseURL)
}

// applyApertureProxy rewrites one provider's models per the proxy plan:
// gateway base URL, optional api override, and the keepGatewayModelsOnly
// filter. Model ids stay bare here; the request path qualifies them
// (streamAuthenticated), matching the original where the picker shows bare
// ids.
func (c *Catalog) applyApertureProxy(provider *Provider) {
	state := c.loadApertureState()
	route, ok := state.Routes[provider.ID]
	if !ok {
		return
	}
	rewritten := make([]llm.Model, 0, len(provider.models))
	for _, model := range provider.models {
		if route.ServedModelIDs != nil && !route.ServedModelIDs[model.ID] {
			continue
		}
		model.BaseURL = route.BaseURL
		if route.APIOverridden {
			model.API = route.API
		}
		rewritten = append(rewritten, model)
	}
	provider.models = rewritten
	provider.BaseURL = route.BaseURL
}

// apertureProxyAuth resolves the gateway-injected placeholder credential for
// a proxied provider. Passthrough providers keep native auth so the client
// sends a real credential the gateway forwards; stored OAuth credentials
// take precedence for the rest (the original replaces only the api-key
// resolver on the wrapped provider).
func (c *Catalog) apertureProxyAuth(providerID string) (*Auth, bool) {
	state := c.loadApertureState()
	route, ok := state.Routes[providerID]
	if !ok || route.Passthrough {
		return nil, false
	}
	return &Auth{APIKey: "-", Source: "aperture proxy"}, true
}
