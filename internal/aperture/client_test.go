package aperture

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func gatewayStub(t *testing.T, providersBody, modelsBody, connectorsBody string, modelsStatus int) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/providers":
			w.Write([]byte(providersBody))
		case "/v1/models":
			if modelsStatus != 0 {
				w.WriteHeader(modelsStatus)
				return
			}
			w.Write([]byte(modelsBody))
		case "/api/connectors":
			w.Write([]byte(connectorsBody))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

const providersArrayBody = `[
	{"id": "anthropic", "name": "Anthropic", "models": ["claude-sonnet-5", "claude-haiku-4"],
	 "compatibility": {"anthropic_messages": true, "openai_chat": true}},
	{"id": "openai", "name": "OpenAI", "models": ["gpt-5"],
	 "compatibility": {"openai_chat": true}, "requires_client_auth": true},
	{"id": "disabled", "name": "Disabled", "models": ["gone-model"], "compatibility": {}}
]`

const modelsBody = `{"data": [
	{"id": "claude-sonnet-5", "pricing": {"input": "0.00000300", "output": "0.00001500"}},
	{"id": "gpt-5"}
]}`

func TestProvidersFilteredByEnabledModels(t *testing.T) {
	server := gatewayStub(t, providersArrayBody, modelsBody, `{"connectors": []}`, 0)
	providers, err := NewClient(server.URL).Providers(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(providers) != 2 {
		t.Fatalf("providers = %d, want 2 (disabled provider dropped)", len(providers))
	}
	anthropic := providers[0]
	if anthropic.ID != "anthropic" || len(anthropic.Models) != 1 || anthropic.Models[0] != "claude-sonnet-5" {
		t.Fatalf("anthropic models not filtered to /v1/models: %+v", anthropic)
	}
	pricing := anthropic.ModelInfoByID["claude-sonnet-5"].Pricing
	if pricing == nil || pricing.Input != "0.00000300" {
		t.Errorf("pricing not attached: %+v", pricing)
	}
	if !providers[1].RequiresClientAuth {
		t.Error("requires_client_auth lost")
	}
}

func TestProvidersModelsFetchFailsOpen(t *testing.T) {
	server := gatewayStub(t, providersArrayBody, "", `{}`, http.StatusNotFound)
	providers, err := NewClient(server.URL).Providers(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// The /api/providers result stays unfiltered as a safe fallback.
	if len(providers) != 3 {
		t.Fatalf("providers = %d, want 3 unfiltered", len(providers))
	}
	if len(providers[0].Models) != 2 {
		t.Errorf("model list should be unfiltered: %v", providers[0].Models)
	}
}

func TestProvidersMapBody(t *testing.T) {
	body := `{"providers": {"zeta": {"name": "Zeta", "models": ["m1"]}, "alpha": {"models": ["m2"]}}}`
	server := gatewayStub(t, body, "", `{}`, http.StatusNotFound)
	providers, err := NewClient(server.URL).Providers(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(providers) != 2 || providers[0].ID != "alpha" || providers[1].ID != "zeta" {
		t.Fatalf("map body parsing: %+v", providers)
	}
	if providers[0].Name != "alpha" {
		t.Errorf("missing name defaults to id, got %q", providers[0].Name)
	}
}

func TestConnectors(t *testing.T) {
	connectorsBody := `{"connectors": [
		{"id": "github", "provider": "GitHub", "status": "connected", "description": "GitHub tools"},
		{"description": "no id, dropped"}
	]}`
	server := gatewayStub(t, providersArrayBody, modelsBody, connectorsBody, 0)
	connectors, err := NewClient(server.URL).Connectors(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(connectors) != 1 || connectors[0].ID != "github" {
		t.Fatalf("connectors = %+v", connectors)
	}
}

func TestHealthReportsHTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusBadGateway)
	}))
	t.Cleanup(server.Close)
	err := NewClient(server.URL).Health(context.Background())
	if err == nil {
		t.Fatal("expected a health error")
	}
	httpErr, ok := err.(*HTTPError)
	if !ok || httpErr.Status != http.StatusBadGateway {
		t.Fatalf("error = %v", err)
	}
}
