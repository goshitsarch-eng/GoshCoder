package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"goshcoder/internal/config"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/omniroute"
)

func TestOmniSetupSyncAndDynamicCatalog(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.Header.Get("Authorization") != "Bearer omni-secret" {
			t.Fatalf("authorization = %q", r.Header.Get("Authorization"))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []any{
			map[string]any{"id": "combo/code", "context_length": 222000, "max_output_tokens": 12000, "capabilities": map[string]any{"reasoning": true}},
		}})
	}))
	defer server.Close()

	output, err := runOmniSetup(t.Context(), server.URL+"/v1", "omni-secret", false)
	if err != nil {
		t.Fatal(err)
	}
	if output == "" {
		t.Fatal("setup returned no feedback")
	}
	output, err = runOmniCommand(t.Context(), []string{"sync"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if output == "" {
		t.Fatal("sync returned no feedback")
	}
	if requests < 2 {
		t.Fatalf("requests = %d", requests)
	}
	provider := newCatalog().Provider("omni")
	if provider == nil || len(provider.Models()) != 1 {
		t.Fatalf("provider = %#v", provider)
	}
	model := provider.Model("combo/code")
	if model == nil || model.BaseURL != server.URL+"/v1" || model.ContextWindow != 222000 {
		t.Fatalf("model = %#v", model)
	}
	auth, ok := newCatalog().ResolveAuth("omni")
	if !ok || auth.APIKey != "omni-secret" {
		t.Fatalf("auth = %#v ok=%v", auth, ok)
	}
	if _, err := os.Stat(config.OmniRoutePath()); err != nil {
		t.Fatal(err)
	}
}

func TestOmniSetupPreservesOtherProviderCredentials(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(`{"data":[]}`)) }))
	defer server.Close()
	store := catalog.NewFileCredentialStore(config.AuthPath())
	if _, err := store.Modify("anthropic", func(*catalog.Credential) (*catalog.Credential, error) {
		return &catalog.Credential{Type: "api_key", Key: "claude-key"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := runOmniSetup(t.Context(), server.URL, "", false); err != nil {
		t.Fatal(err)
	}
	anthropic, err := store.Read("anthropic")
	if err != nil {
		t.Fatal(err)
	}
	if anthropic == nil || anthropic.Key != "claude-key" {
		t.Fatalf("anthropic credential = %#v", anthropic)
	}
	configured, err := omniroute.Load(config.OmniRoutePath())
	if err != nil {
		t.Fatal(err)
	}
	if configured.ServerURL != server.URL {
		t.Fatalf("server = %q", configured.ServerURL)
	}
}
