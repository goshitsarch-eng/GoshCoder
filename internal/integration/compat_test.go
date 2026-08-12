package integration

// Catalog compat has to reach the wire protocols. pi keys model.compat by API;
// Go's Model.Compat is the openai-completions shape only, so the other
// protocols read Model.RawCompat instead. That indirection was silently
// dropping compat for 257 catalog models, so these tests assert the resolved
// request rather than the plumbing.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
)

// catalogModel fetches a real catalog model, skipping when the snapshot no
// longer carries it.
func catalogModel(t *testing.T, providerID, modelID string) *llm.Model {
	t.Helper()
	c := catalog.NewCatalog(nil)
	c.SetGetenv(func(string) string { return "" })
	model := c.Model(providerID, modelID)
	if model == nil {
		t.Skipf("%s/%s is absent from this catalog snapshot", providerID, modelID)
	}
	return model
}

// captureRequest points a model at a recording server and returns the request
// body the protocol produced. body is the canned response to stream back.
func captureRequest(t *testing.T, model *llm.Model, response string, run func(*llm.Model)) map[string]any {
	t.Helper()
	var captured map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, response)
	}))
	defer srv.Close()

	model.BaseURL = srv.URL
	run(model)
	return captured
}

// anthropicResponse is a minimal complete anthropic-messages stream.
const anthropicResponse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n" +
	"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n" +
	"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"

func drain(stream *llm.AssistantMessageEventStream) {
	for range stream.Events() {
	}
}

// TestKimiCodingCompatReachesWire is the case that motivated the fix: Kimi's
// k3 carries forceAdaptiveThinking, so a reasoning request must send an
// adaptive thinking block rather than a token budget.
func TestKimiCodingCompatReachesWire(t *testing.T) {
	model := catalogModel(t, "kimi-coding", "k3")
	if len(model.RawCompat) == 0 {
		t.Fatal("kimi-coding/k3 lost its raw compat")
	}

	conv := &llm.Context{Messages: []llm.Message{llm.UserMessage{Role: "user", Content: "hi"}}}
	captured := captureRequest(t, model, anthropicResponse, func(m *llm.Model) {
		drain(llm.StreamAnthropicMessagesSimple(m, conv, &llm.SimpleStreamOptions{
			StreamOptions: llm.StreamOptions{APIKey: "k"},
			Reasoning:     llm.ThinkingHigh,
		}))
	})

	thinking, ok := captured["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v", captured["thinking"])
	}
	// forceAdaptiveThinking selects an effort level, not budget_tokens.
	if thinking["type"] != "adaptive" {
		t.Fatalf("thinking type = %#v, want adaptive (forceAdaptiveThinking was dropped)", thinking["type"])
	}
	if _, hasBudget := thinking["budget_tokens"]; hasBudget {
		t.Fatalf("adaptive thinking must not carry budget_tokens: %#v", thinking)
	}
	outputConfig, ok := captured["output_config"].(map[string]any)
	if !ok || outputConfig["effort"] == nil {
		t.Fatalf("output_config = %#v", captured["output_config"])
	}
}

// TestAnthropicCatalogCompatReachesWire covers the same path for Anthropic's
// own adaptive models.
func TestAnthropicCatalogCompatReachesWire(t *testing.T) {
	model := catalogModel(t, "anthropic", "claude-fable-5")

	conv := &llm.Context{Messages: []llm.Message{llm.UserMessage{Role: "user", Content: "hi"}}}
	captured := captureRequest(t, model, anthropicResponse, func(m *llm.Model) {
		drain(llm.StreamAnthropicMessagesSimple(m, conv, &llm.SimpleStreamOptions{
			StreamOptions: llm.StreamOptions{APIKey: "k"},
			Reasoning:     llm.ThinkingHigh,
		}))
	})

	thinking, ok := captured["thinking"].(map[string]any)
	if !ok || thinking["type"] != "adaptive" {
		t.Fatalf("thinking = %#v, want adaptive", captured["thinking"])
	}
}

// TestBudgetBasedModelStillUsesBudget is the negative control: a model without
// forceAdaptiveThinking must keep the token-budget path, so the fix cannot be
// satisfied by always sending adaptive.
func TestBudgetBasedModelStillUsesBudget(t *testing.T) {
	model := catalogModel(t, "anthropic", "claude-haiku-4-5")

	conv := &llm.Context{Messages: []llm.Message{llm.UserMessage{Role: "user", Content: "hi"}}}
	captured := captureRequest(t, model, anthropicResponse, func(m *llm.Model) {
		drain(llm.StreamAnthropicMessagesSimple(m, conv, &llm.SimpleStreamOptions{
			StreamOptions: llm.StreamOptions{APIKey: "k"},
			Reasoning:     llm.ThinkingMedium,
		}))
	})

	thinking, ok := captured["thinking"].(map[string]any)
	if !ok {
		t.Fatalf("thinking = %#v", captured["thinking"])
	}
	if thinking["type"] != "enabled" {
		t.Fatalf("thinking type = %#v, want enabled", thinking["type"])
	}
	if _, hasBudget := thinking["budget_tokens"]; !hasBudget {
		t.Fatalf("budget-based thinking needs budget_tokens: %#v", thinking)
	}
}

// TestOptionsCompatStillWins confirms an explicit options compat overrides the
// catalog, which is how callers pin behavior.
func TestOptionsCompatStillWins(t *testing.T) {
	model := catalogModel(t, "kimi-coding", "k3")

	// Force the budget path on a model whose catalog compat asks for adaptive.
	forceAdaptiveOff := false
	conv := &llm.Context{Messages: []llm.Message{llm.UserMessage{Role: "user", Content: "hi"}}}
	enabled := true
	captured := captureRequest(t, model, anthropicResponse, func(m *llm.Model) {
		drain(llm.StreamAnthropicMessages(m, conv, &llm.AnthropicOptions{
			StreamOptions:        llm.StreamOptions{APIKey: "k", MaxTokens: 4096},
			ThinkingEnabled:      &enabled,
			ThinkingBudgetTokens: 2048,
			Compat:               &llm.AnthropicMessagesCompat{ForceAdaptiveThinking: &forceAdaptiveOff},
		}))
	})

	thinking, ok := captured["thinking"].(map[string]any)
	if !ok || thinking["type"] != "enabled" {
		t.Fatalf("thinking = %#v, want the options compat to win", captured["thinking"])
	}
}

// TestRawCompatSurvivesCatalogCopies guards the clone path: Model.RawCompat is
// not serialized, so a JSON round-trip would silently drop it.
func TestRawCompatSurvivesCatalogCopies(t *testing.T) {
	c := catalog.NewCatalog(nil)
	c.SetGetenv(func(string) string { return "" })

	provider := c.Provider("kimi-coding")
	if provider == nil {
		t.Skip("kimi-coding is absent from this catalog snapshot")
	}
	for _, model := range provider.Models() {
		if model.ID != "k3" {
			continue
		}
		if len(model.RawCompat) == 0 {
			t.Fatal("Provider.Models() dropped RawCompat")
		}
	}
	// Model() is a separate copy path.
	if model := c.Model("kimi-coding", "k3"); model != nil && len(model.RawCompat) == 0 {
		t.Fatal("Catalog.Model() dropped RawCompat")
	}
}

// TestEveryProtocolReceivesItsCatalogCompat is the broad guard: no catalog
// model carrying compat may reach a protocol that cannot see it.
func TestEveryProtocolReceivesItsCatalogCompat(t *testing.T) {
	c := catalog.NewCatalog(nil)
	c.SetGetenv(func(string) string { return "" })

	missing := map[string]int{}
	for _, providerID := range c.ProviderIDs() {
		provider := c.Provider(providerID)
		for _, model := range provider.Models() {
			// The catalog's raw compat is the authority.
			raw, hasCompat := provider.RawCompat(model.ID)
			if !hasCompat || len(raw) == 0 {
				continue
			}
			if len(model.RawCompat) == 0 {
				missing[model.API]++
			}
		}
	}
	if len(missing) > 0 {
		t.Fatalf("models whose compat cannot reach their protocol: %#v", missing)
	}
}
