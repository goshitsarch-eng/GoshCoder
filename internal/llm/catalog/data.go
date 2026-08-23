// Package catalog is a Go port of pi's built-in provider catalog and
// credential storage (reference/pi/packages/ai/src/providers/*, models.ts,
// and auth/*).
//
// The builtin model data lives in catalog.json, generated from the pi
// reference by .tmp/regen-catalog.sh (pi's scripts/generate-models.ts in
// --json-only mode), plus catalog_extra.json for models pi does not carry.
// Custom coding-agent models.json loading is not ported.
package catalog

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"

	"goshcoder/internal/llm"
)

//go:embed catalog.json
var catalogJSON []byte

// catalogExtraJSON holds model data that is not in the pi reference, in the
// same shape as catalog.json. It is a separate file because catalog.json is
// regenerated wholesale from pi and a hand-edit there would be lost on the
// next regeneration.
//
//go:embed catalog_extra.json
var catalogExtraJSON []byte

// modelJSON mirrors llm.Model but keeps compat as raw JSON. llm.Model types
// compat as *llm.OpenAICompletionsCompat only, while pi model data also
// carries anthropic-messages / openai-responses compat keys
// (forceAdaptiveThinking, supportsStrictTools, ...). Those keys are dropped
// from the typed view; the raw compat is preserved in rawCompat.
type modelJSON struct {
	ID               string               `json:"id"`
	Name             string               `json:"name"`
	API              llm.API              `json:"api"`
	Provider         llm.ProviderID       `json:"provider"`
	BaseURL          string               `json:"baseUrl"`
	Reasoning        bool                 `json:"reasoning"`
	ThinkingLevelMap llm.ThinkingLevelMap `json:"thinkingLevelMap,omitempty"`
	Input            []string             `json:"input"`
	Cost             llm.ModelCost        `json:"cost"`
	ContextWindow    int                  `json:"contextWindow"`
	MaxTokens        int                  `json:"maxTokens"`
	SamplingParams   map[string]any       `json:"samplingParams,omitempty"`
	Headers          map[string]string    `json:"headers,omitempty"`
	Compat           json.RawMessage      `json:"compat,omitempty"`
}

func (m *modelJSON) toModel() (*llm.Model, error) {
	model := &llm.Model{
		ID: m.ID, Name: m.Name, API: m.API, Provider: m.Provider,
		BaseURL: m.BaseURL, Reasoning: m.Reasoning,
		ThinkingLevelMap: m.ThinkingLevelMap, Input: m.Input, Cost: m.Cost,
		ContextWindow: m.ContextWindow, MaxTokens: m.MaxTokens,
		SamplingParams: m.SamplingParams, Headers: m.Headers,
	}
	if len(m.Compat) > 0 && string(m.Compat) != "null" {
		compat := &llm.OpenAICompletionsCompat{}
		if err := json.Unmarshal(m.Compat, compat); err != nil {
			return nil, fmt.Errorf("catalog: model %s/%s: invalid compat: %w", m.Provider, m.ID, err)
		}
		model.Compat = compat
		// The raw form carries keys the typed view drops, which the other wire
		// protocols decode into their own compat structs.
		model.RawCompat = append(json.RawMessage(nil), m.Compat...)
	}
	return model, nil
}

// builtinData is the decoded form of catalog.json: provider id -> model id ->
// model, plus the raw compat JSON of each model.
type builtinData struct {
	models    map[string]map[string]*llm.Model
	rawCompat map[string]map[string]json.RawMessage
	order     []string
}

var builtin builtinData

func init() {
	var raw map[string]map[string]*modelJSON
	if err := json.Unmarshal(catalogJSON, &raw); err != nil {
		panic(fmt.Errorf("catalog: embedded catalog.json is invalid: %w", err))
	}
	mergeExtraModels(raw)
	builtin.models = make(map[string]map[string]*llm.Model, len(raw))
	builtin.rawCompat = make(map[string]map[string]json.RawMessage, len(raw))
	for providerID, models := range raw {
		builtin.order = append(builtin.order, providerID)
		builtin.models[providerID] = make(map[string]*llm.Model, len(models))
		for modelID, mj := range models {
			model, err := mj.toModel()
			if err != nil {
				panic(err)
			}
			builtin.models[providerID][modelID] = model
			if len(mj.Compat) > 0 && string(mj.Compat) != "null" {
				if builtin.rawCompat[providerID] == nil {
					builtin.rawCompat[providerID] = map[string]json.RawMessage{}
				}
				builtin.rawCompat[providerID][modelID] = mj.Compat
			}
		}
	}
	sort.Strings(builtin.order)
}

// cloneModel deep-copies a model so callers never mutate shared builtin data.
func cloneModel(m *llm.Model) *llm.Model {
	data, err := json.Marshal(m)
	if err != nil {
		panic(fmt.Errorf("catalog: cannot clone model %s/%s: %v", m.Provider, m.ID, err))
	}
	// Round-trip through the llm.Model type directly; ThinkingLevelMap nil
	// values and compat pointers survive JSON faithfully.
	var out llm.Model
	if err := json.Unmarshal(data, &out); err != nil {
		panic(fmt.Errorf("catalog: cannot clone model %s/%s: %v", m.Provider, m.ID, err))
	}
	// RawCompat is not serialized, so it is copied explicitly.
	if len(m.RawCompat) > 0 {
		out.RawCompat = append(json.RawMessage(nil), m.RawCompat...)
	}
	return &out
}

// BuiltinProviderIDs returns the sorted ids of providers present in the
// embedded catalog.
func BuiltinProviderIDs() []string {
	return append([]string(nil), builtin.order...)
}

// RawCompat returns the raw compat JSON of a builtin model, including keys
// that do not map onto llm.OpenAICompletionsCompat (e.g. anthropic-messages
// compat). ok is false when the model is unknown or has no compat.
func RawCompat(providerID, modelID string) (compat json.RawMessage, ok bool) {
	compat, ok = builtin.rawCompat[providerID][modelID]
	return compat, ok
}

// shadowedExtras names the models catalog_extra.json declares that the
// generated catalog already carries. It is recorded rather than acted on:
// TestExtraCatalogIsNotShadowed is what turns it into a failure, so the day pi
// ships one of these models the duplicate gets deleted instead of quietly
// diverging from pi's own definition.
var shadowedExtras []string

// mergeExtraModels folds catalog_extra.json into the generated catalog. The
// generated data wins on a collision: the extra file exists to add models pi
// does not carry, so where pi does carry one its definition is the authority.
func mergeExtraModels(into map[string]map[string]*modelJSON) {
	var extra map[string]map[string]*modelJSON
	if err := json.Unmarshal(catalogExtraJSON, &extra); err != nil {
		panic(fmt.Errorf("catalog: embedded catalog_extra.json is invalid: %w", err))
	}
	for providerID, models := range extra {
		if into[providerID] == nil {
			into[providerID] = map[string]*modelJSON{}
		}
		for modelID, model := range models {
			if _, exists := into[providerID][modelID]; exists {
				shadowedExtras = append(shadowedExtras, providerID+"/"+modelID)
				continue
			}
			into[providerID][modelID] = model
		}
	}
	sort.Strings(shadowedExtras)
}
