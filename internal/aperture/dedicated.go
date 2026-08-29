package aperture

// Dedicated-catalog construction and the persisted snapshot
// (extensions/aperture/dedicated/runtime.ts, model-defaults.ts).
//
// pi persists the dedicated catalog in its models store so scoped models
// validate during startup even offline; GoshCoder persists the same data in
// extensions/aperture-cache.json, together with the gateway provider
// snapshot proxy mode needs for offline filtering and passthrough detection.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"goshcoder/internal/llm"
)

// DedicatedProviderID is the id of the standalone gateway provider.
const DedicatedProviderID = "aperture"

// BuildCatalogKey is the identity of the catalog a snapshot was built from:
// gateway origin plus the normalized dedicated provider filter, with api
// overrides recorded as id@api so a catalog stamped for a different routing
// api never replays. Origin equality, not a string prefix, so
// gateway.example.evil never matches gateway.example.
func BuildCatalogKey(gatewayURL string, resolved Resolved) string {
	origin := gatewayURL
	if parsed, err := url.Parse(gatewayURL); err == nil && parsed.Host != "" {
		origin = parsed.Scheme + "://" + parsed.Host
	}
	var enabled []string
	for _, provider := range resolved.DedicatedProviders {
		if !provider.Enabled {
			continue
		}
		if provider.API != "" {
			enabled = append(enabled, provider.ID+"@"+provider.API)
		} else {
			enabled = append(enabled, provider.ID)
		}
	}
	sort.Strings(enabled)
	filter := strings.Join(enabled, ",")
	if len(resolved.DedicatedProviders) == 0 {
		filter = "*"
	}
	return origin + " " + filter + " v2"
}

// FilterProviders applies the dedicated provider selection. An empty
// configured list means all gateway providers are included.
func FilterProviders(providers []GatewayProvider, resolved Resolved) []GatewayProvider {
	if len(resolved.DedicatedProviders) == 0 {
		return providers
	}
	selected := map[string]bool{}
	for _, provider := range resolved.DedicatedProviders {
		if provider.Enabled {
			selected[provider.ID] = true
		}
	}
	var out []GatewayProvider
	for _, provider := range providers {
		if selected[provider.ID] {
			out = append(out, provider)
		}
	}
	return out
}

// BuildModels assembles the dedicated catalog from the (already filtered)
// gateway providers. catalogModels supplies upstream base URLs for the
// gateway-root-vs-/v1 inference and capability metadata; models already
// rewritten to the gateway are skipped for URL inference, and the dedicated
// provider's own entries are excluded from metadata matching (they carry the
// safe defaults from a previous sync and would shadow real metadata).
func BuildModels(
	providers []GatewayProvider,
	gatewayURL, baseURL string,
	catalogModels []llm.Model,
	modelsDev ModelsDevCatalog,
	apiOverrides map[string]llm.API,
	notify func(warning string),
) []llm.Model {
	upstreamByProvider := map[string]string{}
	upstreamByModel := map[string]string{}
	for _, model := range catalogModels {
		if model.BaseURL == "" || model.BaseURL == gatewayURL || model.BaseURL == baseURL {
			continue
		}
		if model.Provider != "" {
			if _, ok := upstreamByProvider[model.Provider]; !ok {
				upstreamByProvider[model.Provider] = model.BaseURL
			}
		}
		if _, ok := upstreamByModel[model.ID]; !ok {
			upstreamByModel[model.ID] = model.BaseURL
		}
	}

	var metadataCatalog []llm.Model
	for _, model := range catalogModels {
		if model.Provider != DedicatedProviderID {
			metadataCatalog = append(metadataCatalog, model)
		}
	}

	var models []llm.Model
	for _, provider := range providers {
		api := APIForCompatibility(provider.Compatibility)
		if override, ok := apiOverrides[provider.ID]; ok {
			if IsSelectableAPI(override, provider.Compatibility) {
				api = override
			} else if notify != nil {
				notify(fmt.Sprintf("[aperture] api override %q for dedicated provider %s is not served by the gateway; using the auto-picked api.", override, provider.ID))
			}
		}
		providerUpstream := upstreamByProvider[provider.ID]
		for _, modelID := range provider.Models {
			upstreamBaseURL := providerUpstream
			if upstreamBaseURL == "" {
				// A model-id lookup survives custom gateway provider names:
				// model ids are upstream-standardized.
				upstreamBaseURL = upstreamByModel[modelID]
			}
			metadata := ResolveModelMetadata(provider.ID, modelID, MetadataSources{
				CatalogModels: metadataCatalog,
				ModelsDev:     modelsDev,
			})
			var pricing *ModelPricing
			if info, ok := provider.ModelInfoByID[modelID]; ok {
				pricing = info.Pricing
			}
			models = append(models, buildDefaultModel(provider, modelID, api, gatewayURL, baseURL, upstreamBaseURL, pricing, metadata))
		}
	}
	return models
}

// buildDefaultModel merges safe defaults, resolved metadata, and gateway
// pricing (precedence: defaults < models.dev < native catalog < gateway
// pricing, cost only) into one dedicated model
// (dedicated/model-defaults.ts buildDefaultModelConfig).
func buildDefaultModel(
	provider GatewayProvider,
	modelID string,
	api llm.API,
	gatewayURL, baseURL, upstreamBaseURL string,
	pricing *ModelPricing,
	metadata ModelMetadata,
) llm.Model {
	name := metadata.Name
	if name == "" {
		name = modelID
	}
	input := metadata.Input
	if len(input) == 0 {
		input = []string{"text"}
	}
	contextWindow := metadata.ContextWindow
	if contextWindow == 0 {
		contextWindow = 128_000
	}
	maxTokens := metadata.MaxTokens
	if maxTokens == 0 {
		maxTokens = 8_192
	}
	reasoning := metadata.Reasoning != nil && *metadata.Reasoning

	// The catalog id is provider-qualified so the picker disambiguates
	// duplicate gateway ids; path-embedding APIs keep the bare id because the
	// gateway only accepts bare ids in URL paths.
	id := QualifyModelID(provider.ID, api, modelID)

	model := llm.Model{
		ID:               id,
		Name:             name,
		API:              api,
		Provider:         DedicatedProviderID,
		BaseURL:          BaseURLForAPI(api, gatewayURL, baseURL, upstreamBaseURL),
		Reasoning:        reasoning,
		ThinkingLevelMap: metadata.ThinkingLevelMap,
		Input:            input,
		Cost:             MergeCost(pricing, metadata.Cost),
		ContextWindow:    contextWindow,
		MaxTokens:        maxTokens,
		RawCompat:        metadata.Compat,
	}
	if len(metadata.Compat) > 0 {
		var compat llm.OpenAICompletionsCompat
		if json.Unmarshal(metadata.Compat, &compat) == nil {
			model.Compat = &compat
		}
	}
	return model
}

// GatewaySnapshot is the per-provider slice of the gateway catalog proxy
// mode needs without a live gateway: served model ids, compatibility, and
// passthrough auth.
type GatewaySnapshot struct {
	ID                 string          `json:"id"`
	Name               string          `json:"name,omitempty"`
	Models             []string        `json:"models,omitempty"`
	Compatibility      map[string]bool `json:"compatibility,omitempty"`
	RequiresClientAuth bool            `json:"requires_client_auth,omitempty"`
}

// cachedModel round-trips llm.Model including RawCompat, which llm.Model
// itself deliberately does not serialize.
type cachedModel struct {
	llm.Model
	RawCompatJSON json.RawMessage `json:"rawCompat,omitempty"`
}

// Cache is the persisted aperture-cache.json shape.
type Cache struct {
	// CatalogKey guards restores: a snapshot for a different gateway origin,
	// provider selection, or api override set must not be replayed.
	CatalogKey string            `json:"catalogKey"`
	CheckedAt  int64             `json:"checkedAt"`
	Models     []cachedModel     `json:"models,omitempty"`
	Gateway    []GatewaySnapshot `json:"gateway,omitempty"`
}

// CatalogModels returns the cached dedicated models when the cache matches
// catalogKey, or nil.
func (c Cache) CatalogModels(catalogKey string) []llm.Model {
	if c.CatalogKey != catalogKey {
		return nil
	}
	models := make([]llm.Model, 0, len(c.Models))
	for _, cached := range c.Models {
		model := cached.Model
		model.RawCompat = cached.RawCompatJSON
		if model.Provider == "" {
			model.Provider = DedicatedProviderID
		}
		models = append(models, model)
	}
	return models
}

// NewCache builds a snapshot for saving.
func NewCache(catalogKey string, models []llm.Model, gateway []GatewayProvider) Cache {
	cache := Cache{CatalogKey: catalogKey, CheckedAt: time.Now().UnixMilli()}
	for _, model := range models {
		cache.Models = append(cache.Models, cachedModel{Model: model, RawCompatJSON: model.RawCompat})
	}
	for _, provider := range gateway {
		cache.Gateway = append(cache.Gateway, GatewaySnapshot{
			ID:                 provider.ID,
			Name:               provider.Name,
			Models:             provider.Models,
			Compatibility:      provider.Compatibility,
			RequiresClientAuth: provider.RequiresClientAuth,
		})
	}
	return cache
}

// LoadCache reads the persisted snapshot. Missing files report
// os.ErrNotExist.
func LoadCache(path string) (Cache, error) {
	file, err := os.Open(path)
	if err != nil {
		return Cache{}, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return Cache{}, err
	}
	if !info.Mode().IsRegular() {
		return Cache{}, errors.New("aperture cache is not a regular file")
	}
	if info.Size() > maxResponseBytes {
		return Cache{}, fmt.Errorf("aperture cache exceeds %d bytes", maxResponseBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxResponseBytes+1))
	if err != nil {
		return Cache{}, err
	}
	var cache Cache
	if err := json.Unmarshal(data, &cache); err != nil {
		return Cache{}, fmt.Errorf("invalid aperture cache: %w", err)
	}
	return cache, nil
}

// SaveCache atomically publishes the snapshot with user-only permissions.
func SaveCache(path string, cache Cache) error {
	data, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".aperture-cache-*.tmp")
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

// SyncResult reports one networked catalog refresh.
type SyncResult struct {
	// Models is the rebuilt dedicated catalog (empty when dedicated is off).
	Models []llm.Model
	// Gateway is the fresh provider snapshot.
	Gateway []GatewayProvider
	// Warnings are user-facing (api override fallbacks, missing models).
	Warnings []string
}

// Sync performs the networked refresh both the dedicated and proxy
// capabilities share: fetch the gateway catalog (and models.dev, best
// effort), rebuild the dedicated model list, and persist the snapshot to
// cachePath. catalogModels supplies the metadata/URL-inference inputs.
func Sync(ctx context.Context, config Config, cachePath string, catalogModels []llm.Model) (SyncResult, error) {
	resolved := config.Resolve()
	gatewayURL := GatewayURL(resolved.BaseURL)
	baseURL := ProviderBaseURL(resolved.BaseURL)
	if gatewayURL == "" || baseURL == "" {
		return SyncResult{}, errors.New("aperture gateway URL is not configured; run /aperture onboarding")
	}

	client := NewClient(gatewayURL)
	gatewayProviders, err := client.Providers(ctx)
	if err != nil {
		return SyncResult{}, fmt.Errorf("fetch Aperture providers: %w", err)
	}

	result := SyncResult{Gateway: gatewayProviders}
	notify := func(warning string) { result.Warnings = append(result.Warnings, warning) }

	if resolved.DedicatedEnabled {
		modelsDev := FetchModelsDevCatalog(ctx, nil)
		providers := FilterProviders(gatewayProviders, resolved)
		apiOverrides := map[string]llm.API{}
		for _, provider := range resolved.DedicatedProviders {
			if provider.Enabled && provider.API != "" {
				apiOverrides[provider.ID] = provider.API
			}
		}
		result.Models = BuildModels(providers, gatewayURL, baseURL, catalogModels, modelsDev, apiOverrides, notify)
	}

	if resolved.ProxyEnabled {
		for _, warning := range proxyOverrideWarnings(resolved, gatewayProviders) {
			notify(warning)
		}
		if summary := MissingModelsSummary(resolved, gatewayProviders, catalogModels); summary != "" {
			notify(summary)
		}
	}

	catalogKey := BuildCatalogKey(gatewayURL, resolved)
	if err := SaveCache(cachePath, NewCache(catalogKey, result.Models, gatewayProviders)); err != nil {
		return result, fmt.Errorf("save Aperture cache: %w", err)
	}
	return result, nil
}
