package aperture

// Model metadata resolution for dedicated mode (src/model-metadata/*.ts).
//
// Aperture's /v1/models only reports model ids and pricing, so capability
// metadata (vision input, reasoning, context window, output limit) is layered
// from two sources over safe defaults:
//
//  1. the models.dev catalog (broad coverage, fetched best-effort), then
//  2. GoshCoder's native model catalog (authoritative when it knows the
//     model; it carries thinkingLevelMap and compat, which models.dev lacks).
//
// The catalog wins over models.dev; gateway pricing is applied last by the
// caller and wins over both. Matching prefers an exact provider-id +
// model-id match; a model-id-only fallback match copies capabilities but
// never cost (the same id can be priced differently by another serving
// provider) and only the model-intrinsic compat fields.

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"time"

	"goshcoder/internal/llm"
)

// ModelsDevURL is the public models.dev catalog endpoint.
const ModelsDevURL = "https://models.dev/api.json"

const modelsDevTimeout = 10 * time.Second

// ModelsDevModel is the subset of a models.dev entry used for enrichment.
type ModelsDevModel struct {
	Name       string `json:"name,omitempty"`
	Reasoning  *bool  `json:"reasoning,omitempty"`
	Modalities struct {
		Input []string `json:"input,omitempty"`
	} `json:"modalities,omitempty"`
	Limit struct {
		Context int `json:"context,omitempty"`
		Output  int `json:"output,omitempty"`
	} `json:"limit,omitempty"`
	// Cost carries per-million-token USD rates.
	Cost *struct {
		Input      *float64 `json:"input,omitempty"`
		Output     *float64 `json:"output,omitempty"`
		CacheRead  *float64 `json:"cache_read,omitempty"`
		CacheWrite *float64 `json:"cache_write,omitempty"`
	} `json:"cost,omitempty"`
}

// ModelsDevCatalog is keyed by provider id, each with models keyed by id.
type ModelsDevCatalog map[string]struct {
	Models map[string]ModelsDevModel `json:"models,omitempty"`
}

// FetchModelsDevCatalog fetches the models.dev catalog. Best-effort: any
// failure (network, timeout, malformed body) returns nil so enrichment
// silently degrades to the native catalog and safe defaults.
func FetchModelsDevCatalog(ctx context.Context, client *http.Client) ModelsDevCatalog {
	if client == nil {
		client = &http.Client{Timeout: modelsDevTimeout}
	}
	requestCtx, cancel := context.WithTimeout(ctx, modelsDevTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, ModelsDevURL, nil)
	if err != nil {
		return nil
	}
	response, err := client.Do(req)
	if err != nil {
		return nil
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil || len(payload) > maxResponseBytes {
		return nil
	}
	var catalog ModelsDevCatalog
	if json.Unmarshal(payload, &catalog) != nil {
		return nil
	}
	return catalog
}

// ModelMetadata is the resolved metadata overrides for one gateway model.
type ModelMetadata struct {
	Name             string
	Reasoning        *bool
	ThinkingLevelMap llm.ThinkingLevelMap
	Input            []string
	ContextWindow    int
	MaxTokens        int
	// Cost is only set from provider-exact matches (per-million-token USD).
	Cost *llm.ModelCost
	// Compat is the raw compat JSON, only from a provider-exact catalog
	// match, or the intrinsic subset from a model-id fallback match.
	Compat json.RawMessage
}

// MetadataSources are the inputs to ResolveModelMetadata. The caller must
// pre-filter catalog models that would self-reference (the dedicated
// aperture provider's own previously registered models, which carry the safe
// defaults).
type MetadataSources struct {
	CatalogModels []llm.Model
	ModelsDev     ModelsDevCatalog
}

// ResolveModelMetadata resolves capability metadata for one gateway model,
// applying models.dev first and the native catalog on top (catalog wins where
// both know the model). Only fields a source actually provided are set.
func ResolveModelMetadata(providerID, modelID string, sources MetadataSources) ModelMetadata {
	metadata := ModelMetadata{}
	if sources.ModelsDev != nil {
		applyModelsDevMetadata(&metadata, sources.ModelsDev, providerID, modelID)
	}
	if len(sources.CatalogModels) > 0 {
		applyCatalogMetadata(&metadata, sources.CatalogModels, providerID, modelID)
	}
	return metadata
}

// findModelsDevMatch prefers a provider-exact match, then a unique model-id
// fallback. Context/output limits can genuinely differ for the same model id
// across serving providers, so an ambiguous match is worse than defaults.
func findModelsDevMatch(catalog ModelsDevCatalog, providerID, modelID string) (ModelsDevModel, bool, bool) {
	if provider, ok := catalog[providerID]; ok {
		if model, ok := provider.Models[modelID]; ok {
			return model, true, true
		}
	}
	var found *ModelsDevModel
	for _, provider := range catalog {
		if model, ok := provider.Models[modelID]; ok {
			if found != nil {
				return ModelsDevModel{}, false, false
			}
			copied := model
			found = &copied
		}
	}
	if found == nil {
		return ModelsDevModel{}, false, false
	}
	return *found, false, true
}

func applyModelsDevMetadata(metadata *ModelMetadata, catalog ModelsDevCatalog, providerID, modelID string) {
	model, providerExact, ok := findModelsDevMatch(catalog, providerID, modelID)
	if !ok {
		return
	}
	if model.Name != "" {
		metadata.Name = model.Name
	}
	if model.Reasoning != nil {
		metadata.Reasoning = model.Reasoning
	}
	if input := normalizeInputModalities(model.Modalities.Input); len(input) > 0 {
		metadata.Input = input
	}
	if model.Limit.Context > 0 {
		metadata.ContextWindow = model.Limit.Context
	}
	if model.Limit.Output > 0 {
		metadata.MaxTokens = model.Limit.Output
	}
	if providerExact && model.Cost != nil && (model.Cost.Input != nil || model.Cost.Output != nil) {
		cost := llm.ModelCost{}
		if model.Cost.Input != nil {
			cost.Input = *model.Cost.Input
		}
		if model.Cost.Output != nil {
			cost.Output = *model.Cost.Output
		}
		if model.Cost.CacheRead != nil {
			cost.CacheRead = *model.Cost.CacheRead
		}
		if model.Cost.CacheWrite != nil {
			cost.CacheWrite = *model.Cost.CacheWrite
		}
		metadata.Cost = &cost
	}
}

// intrinsicCompatKeys are compat fields intrinsic to a model rather than the
// provider serving it, so they survive a cross-provider model-id match.
// Everything else (supportsStore, deferredToolsMode, provider-named
// thinkingFormat, ...) is endpoint-specific and stays out of a fallback.
var intrinsicCompatKeys = []string{
	"supportsDeveloperRole",
	"maxTokensField",
	"requiresReasoningContentOnAssistantMessages",
}

func applyCatalogMetadata(metadata *ModelMetadata, catalogModels []llm.Model, providerID, modelID string) {
	var match *llm.Model
	providerExact := false
	for index := range catalogModels {
		model := &catalogModels[index]
		if model.Provider == providerID && model.ID == modelID {
			match, providerExact = model, true
			break
		}
	}
	if match == nil {
		for index := range catalogModels {
			if catalogModels[index].ID == modelID {
				match = &catalogModels[index]
				break
			}
		}
	}
	if match == nil {
		return
	}
	if match.Name != "" {
		metadata.Name = match.Name
	}
	reasoning := match.Reasoning
	metadata.Reasoning = &reasoning
	if match.ThinkingLevelMap != nil {
		metadata.ThinkingLevelMap = match.ThinkingLevelMap
	}
	if len(match.Input) > 0 {
		metadata.Input = append([]string(nil), match.Input...)
	}
	if match.ContextWindow > 0 {
		metadata.ContextWindow = match.ContextWindow
	}
	if match.MaxTokens > 0 {
		metadata.MaxTokens = match.MaxTokens
	}
	if providerExact {
		if hasCost(match.Cost) {
			cost := match.Cost
			metadata.Cost = &cost
		}
		if len(match.RawCompat) > 0 && string(match.RawCompat) != "null" {
			metadata.Compat = match.RawCompat
		}
		return
	}
	// Model-id fallback: copy only the model-intrinsic compat fields.
	if len(match.RawCompat) > 0 {
		var raw map[string]json.RawMessage
		if json.Unmarshal(match.RawCompat, &raw) == nil {
			intrinsic := map[string]json.RawMessage{}
			for _, key := range intrinsicCompatKeys {
				if value, ok := raw[key]; ok {
					intrinsic[key] = value
				}
			}
			if len(intrinsic) > 0 {
				if encoded, err := json.Marshal(intrinsic); err == nil {
					metadata.Compat = encoded
				}
			}
		}
	}
}

func hasCost(cost llm.ModelCost) bool {
	return cost.Input != 0 || cost.Output != 0
}

func normalizeInputModalities(values []string) []string {
	var out []string
	for _, value := range values {
		if value == "text" || value == "image" {
			out = append(out, value)
		}
	}
	return out
}

const tokensPerMillion = 1_000_000

func parsePrice(value string) float64 {
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0
	}
	return parsed * tokensPerMillion
}

// MergeCost merges gateway pricing field-by-field over the metadata cost.
// Every rate the gateway reports wins; rates it omits keep the resolved
// value, so a partial pricing response (e.g. only cache rates) neither zeroes
// the other fields nor is discarded (dedicated/model-defaults.ts mergeCost).
func MergeCost(pricing *ModelPricing, base *llm.ModelCost) llm.ModelCost {
	cost := llm.ModelCost{}
	if base != nil {
		cost = *base
	}
	if pricing == nil {
		return cost
	}
	if pricing.Input != "" {
		cost.Input = parsePrice(pricing.Input)
	}
	if pricing.Output != "" {
		cost.Output = parsePrice(pricing.Output)
	}
	if pricing.InputCacheRead != "" {
		cost.CacheRead = parsePrice(pricing.InputCacheRead)
	}
	if pricing.InputCacheWrite != "" {
		cost.CacheWrite = parsePrice(pricing.InputCacheWrite)
	}
	return cost
}
