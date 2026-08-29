package aperture

import (
	"strings"
	"testing"
)

func TestNormalizeInputURL(t *testing.T) {
	cases := []struct{ in, want string }{
		{"ai.pango-lin.ts.net", "http://ai.pango-lin.ts.net"},
		{"  ai.host.ts.net  ", "http://ai.host.ts.net"},
		{"http://ai.host.ts.net/v1/models", "http://ai.host.ts.net"},
		{"https://gateway.example.com:8443/some/path?x=1#f", "https://gateway.example.com:8443"},
		{"", ""},
	}
	for _, c := range cases {
		if got := NormalizeInputURL(c.in); got != c.want {
			t.Errorf("NormalizeInputURL(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestGatewayAndProviderURL(t *testing.T) {
	if got := GatewayURL("http://gw.example/v1/"); got != "http://gw.example" {
		t.Errorf("GatewayURL trims /v1: got %q", got)
	}
	if got := ProviderBaseURL("http://gw.example"); got != "http://gw.example/v1" {
		t.Errorf("ProviderBaseURL: got %q", got)
	}
	if got := ProviderBaseURL(""); got != "" {
		t.Errorf("ProviderBaseURL empty: got %q", got)
	}
}

func TestAPISelection(t *testing.T) {
	compatibility := map[string]bool{
		"anthropic_messages": true,
		"openai_responses":   true,
		"openai_chat":        true,
		// Flags with no dispatch are never mapped.
		"google_raw_predict": true,
	}
	apis := SelectableAPIs(compatibility)
	want := []string{"openai-completions", "anthropic-messages", "openai-responses"}
	if strings.Join(apis, ",") != strings.Join(want, ",") {
		t.Fatalf("SelectableAPIs = %v, want %v", apis, want)
	}
	if got := APIForCompatibility(compatibility); got != "openai-completions" {
		t.Errorf("auto-pick = %q", got)
	}
	if got := APIForCompatibility(nil); got != "openai-completions" {
		t.Errorf("empty compatibility auto-pick = %q", got)
	}
	if !IsSelectableAPI("anthropic-messages", compatibility) {
		t.Error("anthropic-messages should be selectable")
	}
	if IsSelectableAPI("google-vertex", compatibility) {
		t.Error("google-vertex should not be selectable")
	}
}

func TestShouldUseGatewayRoot(t *testing.T) {
	cases := []struct {
		api      string
		upstream string
		want     bool
	}{
		// SDK-appends-path APIs always use the root.
		{"anthropic-messages", "", true},
		{"openai-codex-responses", "", true},
		// OpenAI-SDK APIs infer from the upstream base URL.
		{"openai-completions", "https://api.openai.com/v1", false},
		{"openai-completions", "https://api.z.ai/api/coding/paas/v4", true},
		{"openai-responses", "https://api.example.com/v4beta", true},
		{"openai-completions", "", false},
		{"openai-completions", "://broken", false},
		// Other APIs keep the conservative /v1.
		{"google-generative-ai", "https://api.z.ai/api/coding/paas/v4", false},
	}
	for _, c := range cases {
		if got := ShouldUseGatewayRoot(c.api, c.upstream); got != c.want {
			t.Errorf("ShouldUseGatewayRoot(%q, %q) = %v, want %v", c.api, c.upstream, got, c.want)
		}
	}
}

func TestBaseURLForAPI(t *testing.T) {
	gateway, base := "http://gw.example", "http://gw.example/v1"
	cases := []struct {
		api      string
		upstream string
		want     string
	}{
		{"anthropic-messages", "", gateway},
		{"google-generative-ai", "", gateway + "/v1beta"},
		{"google-vertex", "", gateway + "/v1"},
		{"bedrock-converse-stream", "", gateway + "/bedrock"},
		{"openai-completions", "https://api.openai.com/v1", base},
		{"openai-completions", "https://api.z.ai/api/coding/paas/v4", gateway},
		{"openai-codex-responses", "", gateway},
	}
	for _, c := range cases {
		if got := BaseURLForAPI(c.api, gateway, base, c.upstream); got != c.want {
			t.Errorf("BaseURLForAPI(%q, upstream %q) = %q, want %q", c.api, c.upstream, got, c.want)
		}
	}
}

func TestModelIDQualification(t *testing.T) {
	if got := QualifyModelID("anthropic", "anthropic-messages", "claude-sonnet-5"); got != "anthropic/claude-sonnet-5" {
		t.Errorf("body API qualification = %q", got)
	}
	if got := QualifyModelID("google", "google-generative-ai", "gemini-2.5-pro"); got != "gemini-2.5-pro" {
		t.Errorf("path API stays bare: got %q", got)
	}
	if got := StripCatalogPrefix("google-generative-ai", "google/gemini-2.5-pro"); got != "gemini-2.5-pro" {
		t.Errorf("path API strip = %q", got)
	}
	// Only the first segment is stripped: upstream ids can carry slashes.
	if got := StripCatalogPrefix("google-vertex", "acme/hf:org/some-model"); got != "hf:org/some-model" {
		t.Errorf("multi-slash strip = %q", got)
	}
	if got := StripCatalogPrefix("anthropic-messages", "anthropic/claude-sonnet-5"); got != "anthropic/claude-sonnet-5" {
		t.Errorf("body API keeps qualified id: got %q", got)
	}
}

func TestMarkRetryableError(t *testing.T) {
	if got := MarkRetryableError("Aperture is restarting, please hold"); got != "Aperture is restarting, please hold (service unavailable)" {
		t.Errorf("tagged = %q", got)
	}
	if got := MarkRetryableError("Aperture is restarting: service unavailable"); got != "" {
		t.Errorf("already-marked message must be left alone, got %q", got)
	}
	if got := MarkRetryableError("upstream rejected the key"); got != "" {
		t.Errorf("non-transient message must be left alone, got %q", got)
	}
}
