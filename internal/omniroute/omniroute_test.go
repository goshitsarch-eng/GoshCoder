package omniroute

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestNormalizeServerURLAndAtomicConfig(t *testing.T) {
	got, err := NormalizeServerURL(" https://example.com/gateway/v1/ ")
	if err != nil {
		t.Fatal(err)
	}
	if got != "https://example.com/gateway" {
		t.Fatalf("normalized = %q", got)
	}
	path := filepath.Join(t.TempDir(), "omniroute.json")
	config := Config{ServerURL: got, Models: []Model{{ID: "combo", Input: []string{"text"}}}}
	if err := Save(path, config); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	// Windows has no Unix permission bits: Chmod there can only toggle the
	// read-only flag, so a file written 0600 reports 0666. The guarantee is
	// real on Unix and unavailable on Windows, so it is asserted where it
	// means something rather than dropped everywhere.
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("mode = %o", info.Mode().Perm())
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.ServerURL != got || len(loaded.Models) != 1 {
		t.Fatalf("loaded = %#v", loaded)
	}
}

func TestFetchModelsFiltersMergesMetadataAndUsesAuth(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		json.NewEncoder(w).Encode(map[string]any{"data": []any{
			map[string]any{"id": "cgpt-web/code", "name": "Code Web", "owned_by": "web", "input_modalities": []string{"TEXT", "IMAGE"}, "context_length": 400000, "capabilities": map[string]any{"reasoning": true}},
			map[string]any{"id": "cgpt-web/code", "owned_by": "web", "max_output_tokens": 32000},
			map[string]any{"id": "image-only", "type": "image", "output_modalities": []string{"image"}},
			"plain-model",
		}})
	}))
	defer server.Close()
	client := Client{Config: Config{ServerURL: server.URL}, APIKey: "secret", HTTPClient: server.Client()}
	models, err := client.FetchModels(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if authorization != "Bearer secret" {
		t.Fatalf("authorization = %q", authorization)
	}
	if len(models) != 2 {
		t.Fatalf("models = %#v", models)
	}
	var web Model
	for _, model := range models {
		if model.ID == "cgpt-web/code" {
			web = model
		}
	}
	if web.ContextWindow != 400000 || web.MaxTokens != 32000 || !web.Reasoning || web.ToolCalling == nil || *web.ToolCalling {
		t.Fatalf("merged web = %#v", web)
	}
	llmModel := web.LLMModel(Config{ServerURL: server.URL})
	if llmModel.API != PromptToolsAPI || llmModel.BaseURL != server.URL+"/v1" {
		t.Fatalf("llm model = %#v", llmModel)
	}
}

func TestDecodeExplicitToolCallingFalse(t *testing.T) {
	raw := json.RawMessage(`{"id":"chat","tool_calling":false,"output_modalities":["text"]}`)
	model, ok := decodeModel(raw)
	if !ok || model.ToolCalling == nil || *model.ToolCalling {
		t.Fatalf("model = %#v ok=%v", model, ok)
	}
}
