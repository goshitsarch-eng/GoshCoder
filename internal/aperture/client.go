package aperture

// Gateway API client (src/api/client.ts, src/api/types.ts): /api/providers
// enriched and filtered by /v1/models, /api/connectors, and the health check.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	requestTimeout   = 5 * time.Second
	maxResponseBytes = 16 << 20
)

// ModelPricing carries per-token USD rates as strings, exactly as
// /v1/models reports them. web_search and input_cache_write_1h have no
// mapping in the model cost shape and are ignored when building defaults.
type ModelPricing struct {
	Input             string `json:"input,omitempty"`
	InputCacheRead    string `json:"input_cache_read,omitempty"`
	InputCacheWrite   string `json:"input_cache_write,omitempty"`
	InputCacheWrite1h string `json:"input_cache_write_1h,omitempty"`
	Output            string `json:"output,omitempty"`
	WebSearch         string `json:"web_search,omitempty"`
}

// ModelInfo is metadata retained from /v1/models so dedicated mode can
// attach pricing without re-fetching the gateway.
type ModelInfo struct {
	ID      string        `json:"id"`
	Pricing *ModelPricing `json:"pricing,omitempty"`
}

// GatewayProvider is one /api/providers entry, its model list filtered to
// what /v1/models reports enabled.
type GatewayProvider struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description,omitempty"`
	Models      []string `json:"models"`
	// Compatibility keys the gateway modes a provider can serve
	// (openai_chat, anthropic_messages, ...). Kept open-ended like the
	// original's additionalProperties schema.
	Compatibility map[string]bool `json:"compatibility,omitempty"`
	// RequiresClientAuth is set by auth_mode "passthrough" providers: the
	// gateway forwards the client's own credential, so the client must send a
	// real one.
	RequiresClientAuth bool `json:"requires_client_auth,omitempty"`
	// ModelInfoByID is populated from /v1/models; not present on the raw
	// /api/providers response.
	ModelInfoByID map[string]ModelInfo `json:"modelInfoById,omitempty"`
}

// ConnectorInfo is one /api/connectors entry.
type ConnectorInfo struct {
	ID          string `json:"id"`
	Description string `json:"description,omitempty"`
	Protocol    string `json:"protocol,omitempty"`
	Provider    string `json:"provider,omitempty"`
	Category    string `json:"category,omitempty"`
	Status      string `json:"status,omitempty"`
	AuthType    string `json:"auth_type,omitempty"`
}

// HTTPError is a non-OK gateway response (ApertureHttpError).
type HTTPError struct {
	Method string
	Path   string
	Status int
}

func (e *HTTPError) Error() string {
	return "[Aperture] " + e.Method + " " + e.Path + ": -> " + strconv.Itoa(e.Status) + " " + http.StatusText(e.Status)
}

// Client talks to one Aperture gateway.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewClient returns a client for the gateway at baseURL (trailing slashes
// are trimmed).
func NewClient(baseURL string) *Client {
	return &Client{BaseURL: strings.TrimRight(baseURL, "/")}
}

func (c *Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: requestTimeout}
}

func (c *Client) fetch(ctx context.Context, path string) ([]byte, error) {
	requestCtx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, c.BaseURL+path, nil)
	if err != nil {
		return nil, err
	}
	response, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, &HTTPError{Method: http.MethodGet, Path: path, Status: response.StatusCode}
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > maxResponseBytes {
		return nil, fmt.Errorf("aperture response exceeds %d bytes", maxResponseBytes)
	}
	return payload, nil
}

// enabledModelsByID fetches /v1/models. Disabled providers' models do not
// appear there, so it is the source of truth for which gateway providers are
// usable. Failures resolve to an empty map, which leaves the /api/providers
// result unfiltered as a safe fallback.
func (c *Client) enabledModelsByID(ctx context.Context) map[string]ModelInfo {
	payload, err := c.fetch(ctx, "/v1/models")
	if err != nil {
		return map[string]ModelInfo{}
	}
	var body struct {
		Data []struct {
			ID      string        `json:"id"`
			Pricing *ModelPricing `json:"pricing"`
		} `json:"data"`
	}
	if json.Unmarshal(payload, &body) != nil {
		return map[string]ModelInfo{}
	}
	byID := make(map[string]ModelInfo, len(body.Data))
	for _, entry := range body.Data {
		if entry.ID == "" {
			continue
		}
		byID[entry.ID] = ModelInfo{ID: entry.ID, Pricing: entry.Pricing}
	}
	return byID
}

// parseProvidersBody accepts an array, {"providers": [...]}, or
// {"providers": {id: {...}}} (parseProvidersBody in client.ts).
func parseProvidersBody(payload []byte) []GatewayProvider {
	decode := func(raw json.RawMessage, fallbackID string) (GatewayProvider, bool) {
		var provider GatewayProvider
		if json.Unmarshal(raw, &provider) != nil {
			return GatewayProvider{}, false
		}
		if provider.ID == "" {
			provider.ID = fallbackID
		}
		if provider.ID == "" {
			return GatewayProvider{}, false
		}
		if provider.Name == "" {
			provider.Name = provider.ID
		}
		if provider.Models == nil {
			provider.Models = []string{}
		}
		return provider, true
	}

	var asArray []json.RawMessage
	if json.Unmarshal(payload, &asArray) == nil {
		out := make([]GatewayProvider, 0, len(asArray))
		for _, raw := range asArray {
			if provider, ok := decode(raw, ""); ok {
				out = append(out, provider)
			}
		}
		return out
	}

	var envelope struct {
		Providers json.RawMessage `json:"providers"`
	}
	if json.Unmarshal(payload, &envelope) != nil || len(envelope.Providers) == 0 {
		return nil
	}
	if json.Unmarshal(envelope.Providers, &asArray) == nil {
		out := make([]GatewayProvider, 0, len(asArray))
		for _, raw := range asArray {
			if provider, ok := decode(raw, ""); ok {
				out = append(out, provider)
			}
		}
		return out
	}
	var asMap map[string]json.RawMessage
	if json.Unmarshal(envelope.Providers, &asMap) == nil {
		ids := make([]string, 0, len(asMap))
		for id := range asMap {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		out := make([]GatewayProvider, 0, len(ids))
		for _, id := range ids {
			if provider, ok := decode(asMap[id], id); ok {
				out = append(out, provider)
			}
		}
		return out
	}
	return nil
}

// Providers returns the grant-scoped gateway providers with their model
// lists filtered to what /v1/models serves, pricing attached. Providers left
// with no served models are dropped.
func (c *Client) Providers(ctx context.Context) ([]GatewayProvider, error) {
	payload, err := c.fetch(ctx, "/api/providers")
	if err != nil {
		return nil, err
	}
	parsed := parseProvidersBody(payload)
	enabled := c.enabledModelsByID(ctx)
	if len(enabled) == 0 {
		return parsed, nil
	}
	out := make([]GatewayProvider, 0, len(parsed))
	for _, provider := range parsed {
		var models []string
		infoByID := map[string]ModelInfo{}
		for _, id := range provider.Models {
			if info, ok := enabled[id]; ok {
				models = append(models, id)
				infoByID[id] = info
			}
		}
		if len(models) == 0 {
			continue
		}
		provider.Models = models
		provider.ModelInfoByID = infoByID
		out = append(out, provider)
	}
	return out, nil
}

// Connectors returns the /api/connectors entries.
func (c *Client) Connectors(ctx context.Context) ([]ConnectorInfo, error) {
	payload, err := c.fetch(ctx, "/api/connectors")
	if err != nil {
		return nil, err
	}
	var body struct {
		Connectors []ConnectorInfo `json:"connectors"`
	}
	if err := json.Unmarshal(payload, &body); err != nil {
		return nil, fmt.Errorf("decode Aperture connectors: %w", err)
	}
	var out []ConnectorInfo
	for _, connector := range body.Connectors {
		if connector.ID != "" {
			out = append(out, connector)
		}
	}
	return out, nil
}

// Health probes the gateway by listing providers.
func (c *Client) Health(ctx context.Context) error {
	_, err := c.Providers(ctx)
	return err
}
