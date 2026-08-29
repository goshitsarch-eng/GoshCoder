package aperture

// URL normalization (src/url.ts), compatibility-to-API selection
// (extensions/shared/api-selection.ts), base-URL routing shared by proxy and
// dedicated modes (src/base-url-routing.ts), model-id qualification
// (extensions/aperture/proxy/runtime.ts, dedicated/api-routing.ts), and the
// transient-error tagging from src/retryable-errors.ts.

import (
	"net/url"
	"regexp"
	"strings"

	"goshcoder/internal/llm"
)

// NormalizeInputURL trims a user-input URL, defaults the scheme to http://,
// and reduces it to its origin (scheme + host + port), so pasting a full
// endpoint like "http://ai.host.ts.net/v1/models" still yields the gateway
// root. Unparseable input falls back to stripping /v1 and trailing slashes.
func NormalizeInputURL(raw string) string {
	result := strings.TrimSpace(raw)
	if result == "" {
		return result
	}
	if !strings.HasPrefix(result, "http://") && !strings.HasPrefix(result, "https://") {
		result = "http://" + result
	}
	parsed, err := url.Parse(result)
	if err != nil || parsed.Host == "" {
		return strings.TrimRight(trailingV1.ReplaceAllString(result, ""), "/")
	}
	origin := parsed.Scheme + "://" + parsed.Host
	return origin
}

var trailingV1 = regexp.MustCompile(`/v1/?$`)

// GatewayURL returns the configured gateway URL without a trailing slash or
// /v1 suffix, or "" when baseUrl is unset (resolveGatewayUrl).
func GatewayURL(baseURL string) string {
	if baseURL == "" {
		return ""
	}
	return strings.TrimRight(trailingV1.ReplaceAllString(baseURL, ""), "/")
}

// ProviderBaseURL returns the gateway /v1 endpoint used for OpenAI-shaped
// registration, or "" when the gateway URL cannot be resolved
// (resolveProviderBaseUrl).
func ProviderBaseURL(baseURL string) string {
	gateway := GatewayURL(baseURL)
	if gateway == "" {
		return ""
	}
	return gateway + "/v1"
}

// compatibilityToAPI orders chat completions first: Aperture's default and
// the broadest mode for the tool-calling path. Flags with no dispatch
// (google_raw_predict, bedrock_model_invoke, ...) are not mappable.
var compatibilityToAPI = []struct {
	flag string
	api  llm.API
}{
	{"openai_chat", "openai-completions"},
	{"anthropic_messages", "anthropic-messages"},
	{"openai_responses", "openai-responses"},
	{"gemini_generate_content", "google-generative-ai"},
	{"google_generate_content", "google-vertex"},
	{"bedrock_converse", "bedrock-converse-stream"},
}

// SelectableAPIs returns the APIs a gateway provider can serve, in auto-pick
// precedence order.
func SelectableAPIs(compatibility map[string]bool) []llm.API {
	var apis []llm.API
	for _, entry := range compatibilityToAPI {
		if compatibility[entry.flag] {
			apis = append(apis, entry.api)
		}
	}
	return apis
}

// APIForCompatibility auto-picks the first selectable API, defaulting to
// openai-completions.
func APIForCompatibility(compatibility map[string]bool) llm.API {
	if apis := SelectableAPIs(compatibility); len(apis) > 0 {
		return apis[0]
	}
	return "openai-completions"
}

// IsSelectableAPI validates a per-provider api override against a
// compatibility map.
func IsSelectableAPI(api llm.API, compatibility map[string]bool) bool {
	for _, candidate := range SelectableAPIs(compatibility) {
		if candidate == api {
			return true
		}
	}
	return false
}

// rootBaseURLAPIs are APIs whose client appends the full API path itself, so
// registering gateway/v1 would double the version segment.
var rootBaseURLAPIs = map[llm.API]bool{
	// The Anthropic client appends /v1/messages itself; /v1 would produce
	// /v1/v1/messages, which Aperture does not expose.
	"anthropic-messages": true,
	// The Codex client appends /codex/responses itself.
	"openai-codex-responses": true,
}

// openAISDKAPIs receive model.BaseURL directly and append
// /chat/completions or /responses.
var openAISDKAPIs = map[llm.API]bool{
	"openai-completions": true,
	"openai-responses":   true,
}

var versionSegment = regexp.MustCompile(`/(v\d+\w*)$`)

// hasNonV1VersionPath reports whether baseURL's path ends in a version
// segment that is not /v1 (e.g. Z.ai /api/coding/paas/v4). Such providers
// need a versionless client path because Aperture would otherwise double the
// version (/v4/v1/...). Missing or unparseable URLs report false to stay
// safe.
func hasNonV1VersionPath(baseURL string) bool {
	if baseURL == "" {
		return false
	}
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" {
		return false
	}
	path := strings.TrimRight(parsed.Path, "/")
	match := versionSegment.FindStringSubmatch(path)
	return match != nil && match[1] != "v1"
}

// ShouldUseGatewayRoot reports whether a model should register against the
// gateway root instead of gateway/v1 (src/base-url-routing.ts).
func ShouldUseGatewayRoot(api llm.API, upstreamBaseURL string) bool {
	if rootBaseURLAPIs[api] {
		return true
	}
	if !openAISDKAPIs[api] {
		return false
	}
	return hasNonV1VersionPath(upstreamBaseURL)
}

// EmbedsModelIDInPath reports whether the API embeds the model id in the
// request URL path (Gemini, Vertex, Bedrock) instead of the JSON body.
// Aperture strips the provider/ routing prefix from body-carried model fields
// but only accepts bare ids in URL paths, so both modes send the bare id on
// path-embedding APIs and keep the qualified form everywhere else.
func EmbedsModelIDInPath(api llm.API) bool {
	switch api {
	case "google-generative-ai", "google-vertex", "bedrock-converse-stream":
		return true
	default:
		return false
	}
}

// BaseURLForAPI is the per-API gateway base URL shared by proxy and dedicated
// modes. gatewayURL is the bare gateway origin; baseURL is the conservative
// gateway/v1 OpenAI-SDK fallback. Bedrock lives at /bedrock, not the
// OpenAI-shaped /v1.
func BaseURLForAPI(api llm.API, gatewayURL, baseURL, upstreamBaseURL string) string {
	switch api {
	case "anthropic-messages":
		return gatewayURL
	case "google-generative-ai":
		return gatewayURL + "/v1beta"
	case "google-vertex":
		return gatewayURL + "/v1"
	case "bedrock-converse-stream":
		return gatewayURL + "/bedrock"
	default:
		if ShouldUseGatewayRoot(api, upstreamBaseURL) {
			return gatewayURL
		}
		return baseURL
	}
}

// QualifyModelID prefixes the model id with its gateway provider for
// body-carried APIs so the gateway can disambiguate duplicate ids.
// Path-embedding APIs keep the bare id (the gateway forwards URL paths
// verbatim upstream, where a qualified id 404s).
func QualifyModelID(providerID string, api llm.API, modelID string) string {
	if EmbedsModelIDInPath(api) {
		return modelID
	}
	return providerID + "/" + modelID
}

// StripCatalogPrefix removes the provider/ catalog prefix from a dedicated
// model id for path-embedding APIs. Only the first path segment is stripped:
// upstream ids may themselves contain slashes (e.g. acme/hf:org/some-model).
func StripCatalogPrefix(api llm.API, modelID string) string {
	if !EmbedsModelIDInPath(api) {
		return modelID
	}
	if slash := strings.Index(modelID, "/"); slash != -1 {
		return modelID[slash+1:]
	}
	return modelID
}

// transientErrorPatterns are gateway errors GoshCoder's retry classifier does
// not already treat as retryable (src/retryable-errors.ts).
var transientErrorPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)aperture is restarting`),
}

// retryableMarker matches the classifier's service.?unavailable pattern.
const retryableMarker = "service unavailable"

// MarkRetryableError tags a transient gateway error message so the retry
// classifier picks it up. Returns "" to leave the message alone.
func MarkRetryableError(message string) string {
	transient := false
	for _, pattern := range transientErrorPatterns {
		if pattern.MatchString(message) {
			transient = true
			break
		}
	}
	if !transient {
		return ""
	}
	if strings.Contains(strings.ToLower(message), retryableMarker) {
		return ""
	}
	return message + " (" + retryableMarker + ")"
}
