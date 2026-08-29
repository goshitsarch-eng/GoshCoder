package main

import (
	"strings"
	"testing"

	"goshcoder/internal/aperture"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
)

func TestApplyApertureSetting(t *testing.T) {
	var config aperture.Config
	steps := [][2]string{
		{"baseUrl", "ai.host.ts.net/v1"},
		{"proxy.enabled", "enabled"},
		{"dedicated.enabled", "disabled"},
		{"connectors.enabled", "enabled"},
		{"connectors.discoveryTools", "disabled"},
		{"onboardingDone", "completed"},
		{"proxy.provider.anthropic.enabled", "enabled"},
		{"proxy.provider.anthropic.gatewayModelsOnly", "on"},
		{"proxy.provider.anthropic.api", "anthropic-messages"},
		{"dedicated.provider.google.enabled", "disabled"},
		{"dedicated.provider.google.api", "google-generative-ai"},
	}
	for _, step := range steps {
		if err := applyApertureSetting(&config, step[0], step[1]); err != nil {
			t.Fatalf("set %s: %v", step[0], err)
		}
	}
	resolved := config.Resolve()
	if resolved.BaseURL != "http://ai.host.ts.net" {
		t.Errorf("baseUrl normalized = %q", resolved.BaseURL)
	}
	if !resolved.ProxyEnabled || resolved.DedicatedEnabled || !resolved.ConnectorsEnabled || resolved.DiscoveryTools {
		t.Errorf("toggles: %+v", resolved)
	}
	if !resolved.OnboardingDone || resolved.OnboardingEnabled {
		t.Error("onboardingDone also disables the onboarding affordances")
	}
	if len(resolved.UpstreamProviders) != 1 {
		t.Fatalf("upstream = %+v", resolved.UpstreamProviders)
	}
	provider := resolved.UpstreamProviders[0]
	if !provider.IsEnabled() || !provider.KeepGatewayModelsOnly || provider.API != "anthropic-messages" {
		t.Errorf("proxy provider = %+v", provider)
	}
	if len(resolved.DedicatedProviders) != 1 || resolved.DedicatedProviders[0].Enabled || resolved.DedicatedProviders[0].API != "google-generative-ai" {
		t.Errorf("dedicated provider = %+v", resolved.DedicatedProviders)
	}

	if err := applyApertureSetting(&config, "proxy.provider.anthropic.api", "not-an-api"); err == nil {
		t.Error("invalid api must be rejected")
	}
	if err := applyApertureSetting(&config, "nope", "x"); err == nil {
		t.Error("unknown keys must be rejected")
	}
	if err := applyApertureSetting(&config, "proxy.provider.anthropic.api", "auto"); err != nil {
		t.Fatal(err)
	}
	if config.Resolve().UpstreamProviders[0].API != "" {
		t.Error("auto clears the override")
	}
}

func TestApertureRequestModel(t *testing.T) {
	state := &catalog.ApertureState{
		Configured: true,
		Routes: map[string]aperture.ProxyRoute{
			"anthropic": {ProviderID: "anthropic", API: "anthropic-messages", BaseURL: "http://gw.example"},
		},
	}

	// A proxied provider's request carries the qualified id and headers.
	model := &llm.Model{ID: "claude-sonnet-5", Provider: "anthropic", API: "anthropic-messages",
		Headers: map[string]string{"anthropic-beta": "x"}}
	routed := apertureRequestModel(state, model, "session-1")
	if routed == model {
		t.Fatal("proxied model must be rewritten")
	}
	if routed.ID != "anthropic/claude-sonnet-5" {
		t.Errorf("qualified id = %q", routed.ID)
	}
	if routed.Headers["x-session-id"] != "session-1" || routed.Headers["Referer"] != apertureReferer {
		t.Errorf("headers = %v", routed.Headers)
	}
	if routed.Headers["anthropic-beta"] != "x" {
		t.Error("existing model headers must survive")
	}
	if model.ID != "claude-sonnet-5" || model.Headers["x-session-id"] != "" {
		t.Error("the input model must not be mutated")
	}

	// Dedicated models on path-embedding APIs strip the catalog prefix.
	dedicated := &llm.Model{ID: "google/gemini-2.5-pro", Provider: "aperture", API: "google-generative-ai"}
	routed = apertureRequestModel(state, dedicated, "session-1")
	if routed.ID != "gemini-2.5-pro" {
		t.Errorf("dedicated path-API id = %q", routed.ID)
	}
	// ... and keep the qualified id on body APIs.
	dedicated = &llm.Model{ID: "anthropic/claude-sonnet-5", Provider: "aperture", API: "anthropic-messages"}
	if routed = apertureRequestModel(state, dedicated, ""); routed.ID != "anthropic/claude-sonnet-5" {
		t.Errorf("dedicated body-API id = %q", routed.ID)
	}

	// Unrouted providers pass through untouched.
	plain := &llm.Model{ID: "gpt-5", Provider: "openai", API: "openai-responses"}
	if got := apertureRequestModel(state, plain, "session-1"); got != plain {
		t.Error("unrouted model must be returned unchanged")
	}
	if got := apertureRequestModel(&catalog.ApertureState{}, model, "session-1"); got != model {
		t.Error("unconfigured state must be a no-op")
	}
}

func TestMarkApertureRetryableStream(t *testing.T) {
	source := llm.NewAssistantMessageEventStream()
	source.Push(llm.AssistantMessageEvent{Type: llm.EventStart})
	source.Push(llm.AssistantMessageEvent{Type: llm.EventError, Reason: llm.StopError, Error: &llm.AssistantMessage{
		Role: "assistant", StopReason: llm.StopError, ErrorMessage: "Aperture is restarting",
	}})
	source.End()

	wrapped := markApertureRetryable(source)
	var sawError bool
	for event := range wrapped.Events() {
		if event.Type == llm.EventError {
			sawError = true
			if !strings.Contains(event.Error.ErrorMessage, "(service unavailable)") {
				t.Errorf("error not tagged: %q", event.Error.ErrorMessage)
			}
			if !llm.IsRetryableAssistantError(event.Error) {
				t.Error("tagged error must classify as retryable")
			}
		}
	}
	if !sawError {
		t.Fatal("error event lost in the wrapper")
	}
}

func TestProviderSettingKey(t *testing.T) {
	id, field, ok := providerSettingKey("proxy.provider.qwen-token-plan.api", "proxy.provider.")
	if !ok || id != "qwen-token-plan" || field != "api" {
		t.Errorf("parsed = %q %q %v", id, field, ok)
	}
	if _, _, ok := providerSettingKey("proxy.provider.x", "proxy.provider."); ok {
		t.Error("missing field must not parse")
	}
	if _, _, ok := providerSettingKey("other.key", "proxy.provider."); ok {
		t.Error("wrong prefix must not parse")
	}
}
