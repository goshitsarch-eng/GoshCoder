package catalog

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"goshcoder/internal/llm"
)

// TestOverridesTargetGeneratedModels guards the rule that catalog_overrides.json
// patches pi's data and nothing else. An override for a model the generated
// catalog does not carry silently does nothing, which reads like a correction
// that is in force when it is not; the fix is a full entry in
// catalog_extra.json instead.
func TestOverridesTargetGeneratedModels(t *testing.T) {
	if len(orphanOverrides) != 0 {
		t.Fatalf("catalog_overrides.json patches models the generated catalog does not carry: %v; "+
			"add them to catalog_extra.json instead", orphanOverrides)
	}
	if len(appliedOverrides) == 0 {
		t.Fatal("no override applied: catalog_overrides.json never reached the catalog")
	}
}

// TestOverridesStillCorrectPi is the counterpart of TestExtraCatalogIsNotShadowed.
// An override that now restates what pi already says is dead weight that still
// reads as a live correction, so the regeneration that made it redundant should
// also delete it.
func TestOverridesStillCorrectPi(t *testing.T) {
	for _, record := range appliedOverrides {
		if bytes.Equal(canonicalJSON(t, record.was), canonicalJSON(t, record.now)) {
			t.Errorf("catalog_overrides.json %s/%s: %s now matches the generated catalog (%s); delete it",
				record.provider, record.model, record.field, record.now)
		}
	}
}

// TestOverriddenFieldsReachTheCatalog pins the corrections themselves: the
// values a request is actually built from, after the override pass, the extras
// merge and the typed decode.
func TestOverriddenFieldsReachTheCatalog(t *testing.T) {
	// developers.openai.com/api/docs/models/gpt-5.6-sol: 1,050,000 tokens, and
	// $4/$0.40/$20 since the 2026-08-21 cut. pi's snapshot predates both.
	sol := builtinModel(t, "openai", "gpt-5.6-sol")
	if sol.ContextWindow != 1_050_000 {
		t.Errorf("gpt-5.6-sol contextWindow = %d, want 1050000", sol.ContextWindow)
	}
	if sol.Cost.Input != 4 || sol.Cost.Output != 20 || sol.Cost.CacheRead != 0.4 {
		t.Errorf("gpt-5.6-sol cost = %+v, want 4/20/0.4 in/out/cacheRead", sol.Cost.ModelCostRates)
	}
	if len(sol.Cost.Tiers) != 1 || sol.Cost.Tiers[0].InputTokensAbove != 272_000 {
		t.Fatalf("gpt-5.6-sol tiers = %+v, want one tier above 272000", sol.Cost.Tiers)
	}
	if sol.Cost.Tiers[0].Input != 8 || sol.Cost.Tiers[0].Output != 30 {
		t.Errorf("gpt-5.6-sol long-context tier = %+v, want 8/30 in/out", sol.Cost.Tiers[0].ModelCostRates)
	}
	for _, id := range []string{"gpt-5.6-terra", "gpt-5.6-luna"} {
		if model := builtinModel(t, "openai", id); model.ContextWindow != 1_050_000 {
			t.Errorf("%s contextWindow = %d, want 1050000", id, model.ContextWindow)
		}
	}

	// ai.google.dev/gemini-api/docs/pricing: Gemini 3.6 Flash moved onto the
	// 3.7 Flash introductory rate on 2026-08-13, and OpenRouter followed.
	for provider, model := range map[string]string{
		"google":        "gemini-3.6-flash",
		"google-vertex": "gemini-3.6-flash",
		"openrouter":    "google/gemini-3.6-flash",
	} {
		flash := builtinModel(t, provider, model)
		if flash.Cost.Input != 0.75 || flash.Cost.Output != 3.75 || flash.Cost.CacheRead != 0.075 {
			t.Errorf("%s/%s cost = %+v, want 0.75/3.75/0.075", provider, model, flash.Cost.ModelCostRates)
		}
	}
}

// TestEffortLevelsMatchProviderDocs pins the levels the picker offers against
// what each provider documents. The interesting half is the top: xhigh and max
// need an explicit mapping to be offered at all, so a model that gained one
// stays silently capped until its entry says so.
func TestEffortLevelsMatchProviderDocs(t *testing.T) {
	for _, tc := range []struct {
		provider string
		model    string
		want     []llm.ModelThinkingLevel
		why      string
	}{
		// platform.claude.com/docs/en/build-with-claude/effort: max on Fable 5,
		// Mythos 5, Opus 5/4.8/4.7/4.6 and Sonnet 5/4.6; xhigh on all of those
		// but Opus 4.6 and Sonnet 4.6.
		{"anthropic", "claude-opus-5", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "Opus 5 supports both"},
		{"anthropic", "claude-sonnet-5", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "Sonnet 5 supports both"},
		{"anthropic", "claude-fable-5", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "Fable 5 supports both"},
		{"anthropic", "claude-mythos-5", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "Mythos 5 shares Fable 5's specs"},
		{"anthropic", "claude-opus-4-8", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "Opus 4.8 supports both"},
		{"anthropic", "claude-sonnet-4-6", []llm.ModelThinkingLevel{llm.ThinkingMax}, "Sonnet 4.6 has max but not xhigh"},
		{"anthropic", "claude-opus-4-6", []llm.ModelThinkingLevel{llm.ThinkingMax}, "Opus 4.6 has max but not xhigh"},
		// developers.openai.com: every GPT-5.6 model takes none..max.
		{"openai", "gpt-5.6-sol", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "GPT-5.6 Sol supports both"},
		{"openai", "gpt-5.6-terra", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "GPT-5.6 Terra supports both"},
		{"openai", "gpt-5.6-luna", []llm.ModelThinkingLevel{llm.ThinkingXHigh, llm.ThinkingMax}, "GPT-5.6 Luna supports both"},
		// docs.x.ai: xhigh is available on grok-4.6 and later; grok-4.5 treats
		// a request for it as high, so it is not offered there. Bedrock's own
		// model card lists the same four levels.
		{"xai", "grok-4.6", []llm.ModelThinkingLevel{llm.ThinkingXHigh}, "Grok 4.6 gained xhigh"},
		{"openrouter", "x-ai/grok-4.6", []llm.ModelThinkingLevel{llm.ThinkingXHigh}, "Grok 4.6 gained xhigh"},
		{"amazon-bedrock", "us.xai.grok-4.6", []llm.ModelThinkingLevel{llm.ThinkingXHigh}, "Grok 4.6 gained xhigh"},
		{"amazon-bedrock", "global.xai.grok-4.6", []llm.ModelThinkingLevel{llm.ThinkingXHigh}, "Grok 4.6 gained xhigh"},
		// docs.z.ai/guides/llm/glm-5.3: low, high and max, no way to disable.
		{"zai", "glm-5.3", []llm.ModelThinkingLevel{llm.ThinkingLow, llm.ThinkingHigh, llm.ThinkingMax}, "GLM-5.3 effort levels"},
		{"zai-coding-cn", "glm-5.3", []llm.ModelThinkingLevel{llm.ThinkingLow, llm.ThinkingHigh, llm.ThinkingMax}, "GLM-5.3 effort levels"},
		{"openrouter", "z-ai/glm-5.3", []llm.ModelThinkingLevel{llm.ThinkingLow, llm.ThinkingHigh, llm.ThinkingMax}, "GLM-5.3 effort levels"},
		// api-docs.deepseek.com/guides/thinking_mode: low, high and max are the
		// direct values; pi's snapshot dropped low.
		{"deepseek", "deepseek-v4-pro", []llm.ModelThinkingLevel{llm.ThinkingLow, llm.ThinkingHigh, llm.ThinkingMax}, "DeepSeek V4 Pro effort levels"},
		{"deepseek", "deepseek-v4-flash", []llm.ModelThinkingLevel{llm.ThinkingLow, llm.ThinkingHigh, llm.ThinkingMax}, "DeepSeek V4 Flash effort levels"},
	} {
		levels := llm.GetSupportedThinkingLevels(builtinModel(t, tc.provider, tc.model))
		for _, want := range tc.want {
			if !slices.Contains(levels, want) {
				t.Errorf("%s/%s: %q missing from %v (%s)", tc.provider, tc.model, want, levels, tc.why)
			}
		}
	}

	// The negative half: a level the model does not have must stay off the
	// picker, or clamping sends a value the API rejects or silently demotes.
	for _, tc := range []struct {
		provider string
		model    string
		notWant  llm.ModelThinkingLevel
	}{
		{"anthropic", "claude-sonnet-4-6", llm.ThinkingXHigh},
		{"anthropic", "claude-opus-4-6", llm.ThinkingXHigh},
		{"anthropic", "claude-haiku-4-5", llm.ThinkingMax},
		{"xai", "grok-4.5", llm.ThinkingXHigh},
		{"xai", "grok-4.6", llm.ThinkingMax},
		{"amazon-bedrock", "us.xai.grok-4.6", llm.ThinkingMax},
		{"openrouter", "x-ai/grok-4.6", llm.ThinkingMax},
		{"zai", "glm-5.3", llm.ThinkingOff},
		{"openrouter", "z-ai/glm-5.3", llm.ThinkingOff},
		// ai.google.dev/gemini-api/docs/thinking: 3.7 Flash takes low, medium
		// and high; MINIMAL, which 3.6 Flash still accepts, is gone.
		{"google", "gemini-3.7-flash", llm.ThinkingMinimal},
		{"google-vertex", "gemini-3.7-flash", llm.ThinkingMinimal},
		{"openrouter", "google/gemini-3.7-flash", llm.ThinkingMinimal},
	} {
		levels := llm.GetSupportedThinkingLevels(builtinModel(t, tc.provider, tc.model))
		if slices.Contains(levels, tc.notWant) {
			t.Errorf("%s/%s: %q offered but not supported (levels %v)", tc.provider, tc.model, tc.notWant, levels)
		}
	}
}

// TestCurrentFlagshipModelsArePresent names the models a user reaches for today.
// A provider that ships one after pi's snapshot leaves the picker a generation
// behind, which looks like the model is unavailable rather than unlisted.
func TestCurrentFlagshipModelsArePresent(t *testing.T) {
	for _, want := range []struct{ provider, model string }{
		{"anthropic", "claude-opus-5"},
		{"anthropic", "claude-sonnet-5"},
		{"anthropic", "claude-fable-5"},
		{"anthropic", "claude-mythos-5"},
		{"amazon-bedrock", "anthropic.claude-opus-5"},
		{"amazon-bedrock", "us.xai.grok-4.6"},
		{"amazon-bedrock", "global.xai.grok-4.6"},
		{"openrouter", "x-ai/grok-4.6"},
		{"openrouter", "z-ai/glm-5.3"},
		{"openrouter", "google/gemini-3.7-flash"},
		{"openai", "gpt-5.6-sol"},
		{"google", "gemini-3.7-flash"},
		{"google-vertex", "gemini-3.7-flash"},
		{"xai", "grok-4.6"},
		{"zai", "glm-5.3"},
		{"zai-coding-cn", "glm-5.3"},
		{"meta", "muse-spark-1.2"},
		{"meta", "muse-spark-1.2-contributor"},
		{"moonshotai", "kimi-k3"},
		{"deepseek", "deepseek-v4-pro"},
	} {
		if builtin.models[want.provider][want.model] == nil {
			t.Errorf("catalog is missing %s/%s", want.provider, want.model)
		}
	}
}

func builtinModel(t *testing.T, providerID, modelID string) *llm.Model {
	t.Helper()
	model := builtin.models[providerID][modelID]
	if model == nil {
		t.Fatalf("catalog has no %s/%s", providerID, modelID)
	}
	return model
}

// canonicalJSON re-encodes a value so two spellings of the same data compare
// equal: pi's generator and a hand-written override do not have to agree on
// whitespace or key order for the override to be redundant.
func canonicalJSON(t *testing.T, raw json.RawMessage) []byte {
	t.Helper()
	if len(raw) == 0 {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("invalid JSON %s: %v", raw, err)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("cannot re-encode %s: %v", raw, err)
	}
	return encoded
}
