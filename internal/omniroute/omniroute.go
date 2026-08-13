// Package omniroute is a native Go adaptation of
// omniroute-pi-ext-integration. It owns the gateway configuration, public
// model discovery, metadata normalization, and health checks. Authentication
// is deliberately stored by catalog.CredentialStore, not in this config file.
package omniroute

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"goshcoder/internal/llm"
)

const (
	DefaultServerURL = "http://127.0.0.1:20128"
	PromptToolsAPI   = "omni-prompt-tools"
	maxConfigBytes   = 4 << 20
	maxResponseBytes = 16 << 20
)

// Config is the persisted, non-secret part of an OmniRoute setup.
type Config struct {
	ServerURL    string  `json:"serverUrl"`
	DashboardURL string  `json:"dashboardUrl,omitempty"`
	Models       []Model `json:"models,omitempty"`
	SyncedAt     int64   `json:"syncedAt,omitempty"`
}

// Model preserves metadata returned by /v1/models. ToolCalling=false selects
// the prompt-emulated fallback used by web/chat-only model adapters.
type Model struct {
	ID            string               `json:"id"`
	Name          string               `json:"name,omitempty"`
	OwnedBy       string               `json:"ownedBy,omitempty"`
	ContextWindow int                  `json:"contextWindow,omitempty"`
	MaxTokens     int                  `json:"maxTokens,omitempty"`
	Reasoning     bool                 `json:"reasoning,omitempty"`
	Input         []string             `json:"input,omitempty"`
	ToolCalling   *bool                `json:"toolCalling,omitempty"`
	Cost          llm.ModelCost        `json:"cost,omitempty"`
	ThinkingMap   llm.ThinkingLevelMap `json:"thinkingLevelMap,omitempty"`
}

func DefaultConfig() Config { return Config{ServerURL: DefaultServerURL} }

// NormalizeServerURL validates a root server URL. Users may paste the OpenAI
// /v1 endpoint; the persisted form is normalized back to the dashboard root.
func NormalizeServerURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", errors.New("OmniRoute URL is required")
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return "", fmt.Errorf("invalid OmniRoute URL %q (expected http:// or https://)", raw)
	}
	parsed.RawQuery, parsed.Fragment = "", ""
	path := strings.TrimSuffix(strings.TrimRight(parsed.Path, "/"), "/v1")
	parsed.Path = path
	return strings.TrimRight(parsed.String(), "/"), nil
}

func (c Config) APIBaseURL() string { return strings.TrimRight(c.ServerURL, "/") + "/v1" }

func (c Config) Dashboard() string {
	if strings.TrimSpace(c.DashboardURL) != "" {
		return strings.TrimRight(c.DashboardURL, "/")
	}
	return strings.TrimRight(c.ServerURL, "/")
}

// Load reads a bounded strict config. Missing files are reported with
// os.ErrNotExist so callers can distinguish unconfigured from malformed.
func Load(path string) (Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return Config{}, err
	}
	if !info.Mode().IsRegular() {
		return Config{}, errors.New("OmniRoute config is not a regular file")
	}
	if info.Size() > maxConfigBytes {
		return Config{}, fmt.Errorf("OmniRoute config exceeds %d bytes", maxConfigBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxConfigBytes+1))
	if err != nil {
		return Config{}, err
	}
	if len(data) > maxConfigBytes {
		return Config{}, fmt.Errorf("OmniRoute config exceeds %d bytes", maxConfigBytes)
	}
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return Config{}, fmt.Errorf("invalid OmniRoute config: %w", err)
	}
	normalized, err := NormalizeServerURL(config.ServerURL)
	if err != nil {
		return Config{}, err
	}
	config.ServerURL = normalized
	return config, nil
}

// Save atomically publishes config with user-only permissions.
func Save(path string, config Config) error {
	normalized, err := NormalizeServerURL(config.ServerURL)
	if err != nil {
		return err
	}
	config.ServerURL = normalized
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if len(data) > maxConfigBytes {
		return fmt.Errorf("OmniRoute config exceeds %d bytes", maxConfigBytes)
	}
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".omniroute-*.tmp")
	if err != nil {
		return err
	}
	name := temporary.Name()
	defer os.Remove(name)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}

// Client calls the public OpenAI-compatible OmniRoute API.
type Client struct {
	Config     Config
	APIKey     string
	HTTPClient *http.Client
}

func (c Client) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 10 * time.Second}
}

func (c Client) do(ctx context.Context, method, endpoint string, body io.Reader) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.Config.APIBaseURL()+endpoint, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key := strings.TrimSpace(c.APIKey); key != "" && key != "omniroute-public" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	response, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > maxResponseBytes {
		return nil, fmt.Errorf("OmniRoute response exceeds %d bytes", maxResponseBytes)
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("OmniRoute returned %d: %s", response.StatusCode, truncate(strings.TrimSpace(string(payload)), 500))
	}
	return payload, nil
}

func (c Client) Health(ctx context.Context) error {
	_, err := c.do(ctx, http.MethodGet, "/models", nil)
	return err
}

// FetchModels normalizes /v1/models, filters non-chat output, merges duplicate
// IDs, and sorts by owned_by then id like the original extension.
func (c Client) FetchModels(ctx context.Context) ([]Model, error) {
	payload, err := c.do(ctx, http.MethodGet, "/models", nil)
	if err != nil {
		return nil, err
	}
	var envelope struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return nil, fmt.Errorf("decode OmniRoute models: %w", err)
	}
	byID := make(map[string]Model)
	for _, raw := range envelope.Data {
		model, ok := decodeModel(raw)
		if !ok {
			continue
		}
		if previous, exists := byID[model.ID]; exists {
			model = mergeModel(previous, model)
		}
		byID[model.ID] = model
	}
	models := make([]Model, 0, len(byID))
	for _, model := range byID {
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool {
		left, right := models[i].OwnedBy, models[j].OwnedBy
		if left == "" {
			left = "zz"
		}
		if right == "" {
			right = "zz"
		}
		if left != right {
			return left < right
		}
		return models[i].ID < models[j].ID
	})
	return models, nil
}

func decodeModel(raw json.RawMessage) (Model, bool) {
	var stringID string
	if json.Unmarshal(raw, &stringID) == nil {
		if strings.TrimSpace(stringID) == "" {
			return Model{}, false
		}
		return Model{ID: stringID, Name: HumanName(stringID), Input: []string{"text"}}, true
	}
	var source struct {
		ID               string   `json:"id"`
		Name             string   `json:"name"`
		OwnedBy          string   `json:"owned_by"`
		Provider         string   `json:"provider"`
		Type             string   `json:"type"`
		InputModalities  []string `json:"input_modalities"`
		Input            []string `json:"input"`
		OutputModalities []string `json:"output_modalities"`
		Output           []string `json:"output"`
		ContextLength    int      `json:"context_length"`
		MaxInputTokens   int      `json:"max_input_tokens"`
		MaxOutputTokens  int      `json:"max_output_tokens"`
		MaxTokens        int      `json:"max_tokens"`
		ToolCalling      *bool    `json:"tool_calling"`
		Capabilities     struct {
			Reasoning   bool  `json:"reasoning"`
			Thinking    bool  `json:"thinking"`
			ToolCalling *bool `json:"tool_calling"`
		} `json:"capabilities"`
	}
	if json.Unmarshal(raw, &source) != nil || strings.TrimSpace(source.ID) == "" {
		return Model{}, false
	}
	output := normalizeModalities(firstNonEmpty(source.OutputModalities, source.Output))
	if strings.EqualFold(source.Type, "image") || (len(output) > 0 && !contains(output, "text")) {
		return Model{}, false
	}
	input := normalizeModalities(firstNonEmpty(source.InputModalities, source.Input))
	if len(input) == 0 {
		input = []string{"text"}
	}
	contextWindow := source.ContextLength
	if contextWindow == 0 {
		contextWindow = source.MaxInputTokens
	}
	maxTokens := source.MaxOutputTokens
	if maxTokens == 0 {
		maxTokens = source.MaxTokens
	}
	toolCalling := source.ToolCalling
	if toolCalling == nil {
		toolCalling = source.Capabilities.ToolCalling
	}
	if isWebSynced(source.ID, source.Name, source.OwnedBy, source.Provider) {
		value := false
		toolCalling = &value
	}
	name := source.Name
	if name == "" {
		name = HumanName(source.ID)
	}
	return Model{ID: source.ID, Name: name, OwnedBy: source.OwnedBy, ContextWindow: contextWindow, MaxTokens: maxTokens,
		Reasoning: source.Capabilities.Reasoning || source.Capabilities.Thinking, Input: input, ToolCalling: toolCalling}, true
}

func (m Model) LLMModel(config Config) llm.Model {
	api := llm.APIOpenAICompletions
	if m.ToolCalling != nil && !*m.ToolCalling {
		api = PromptToolsAPI
	}
	contextWindow := m.ContextWindow
	if contextWindow <= 0 {
		contextWindow = 128000
	}
	maxTokens := m.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 16384
	}
	input := append([]string(nil), m.Input...)
	if len(input) == 0 {
		input = []string{"text"}
	}
	return llm.Model{ID: m.ID, Name: first(m.Name, HumanName(m.ID)), API: api, Provider: "omni",
		BaseURL: config.APIBaseURL(), Reasoning: m.Reasoning, ThinkingLevelMap: m.ThinkingMap,
		Input: input, Cost: m.Cost, ContextWindow: contextWindow, MaxTokens: maxTokens}
}

func HumanName(id string) string {
	parts := strings.Split(id, "/")
	name := parts[len(parts)-1]
	words := strings.Fields(strings.NewReplacer("-", " ", "_", " ").Replace(name))
	for index := range words {
		if len(words[index]) > 0 {
			words[index] = strings.ToUpper(words[index][:1]) + words[index][1:]
		}
	}
	return strings.Join(words, " ")
}

func mergeModel(old, next Model) Model {
	merged := old
	if next.Name != "" {
		merged.Name = next.Name
	}
	if next.OwnedBy != "" {
		merged.OwnedBy = next.OwnedBy
	}
	if next.ContextWindow != 0 {
		merged.ContextWindow = next.ContextWindow
	}
	if next.MaxTokens != 0 {
		merged.MaxTokens = next.MaxTokens
	}
	merged.Reasoning = old.Reasoning || next.Reasoning
	merged.Input = union(old.Input, next.Input)
	if next.ToolCalling != nil {
		merged.ToolCalling = next.ToolCalling
	}
	if next.Cost.Input != 0 || next.Cost.Output != 0 || next.Cost.CacheRead != 0 || next.Cost.CacheWrite != 0 || len(next.Cost.Tiers) > 0 {
		merged.Cost = next.Cost
	}
	if next.ThinkingMap != nil {
		merged.ThinkingMap = next.ThinkingMap
	}
	return merged
}

func normalizeModalities(values []string) []string {
	var output []string
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if (value == "text" || value == "image") && !contains(output, value) {
			output = append(output, value)
		}
	}
	return output
}
func union(a, b []string) []string {
	return normalizeModalities(append(append([]string(nil), a...), b...))
}
func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
func firstNonEmpty(a, b []string) []string {
	if len(a) > 0 {
		return a
	}
	return b
}
func first(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
func isWebSynced(values ...string) bool {
	for _, value := range values {
		if strings.Contains(strings.ToLower(value), "-web") {
			return true
		}
	}
	return false
}
func truncate(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	return value[:limit] + "..."
}
