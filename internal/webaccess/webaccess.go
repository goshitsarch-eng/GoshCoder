// Package webaccess provides GoshCoder's native Go adaptation of the
// web_search tool from github.com/nicobailon/pi-web-access/index.ts.
//
// The native port keeps the tool available without a TypeScript plugin host:
// keyless Exa MCP search works out of the box, OpenAI/Codex search reuses
// GoshCoder authentication, and Kagi can be selected with its API key.
package webaccess

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

const (
	defaultOpenAIResponsesURL = "https://api.openai.com/v1/responses"
	defaultCodexResponsesURL  = "https://chatgpt.com/backend-api/codex/responses"
	defaultExaMCPURL          = "https://mcp.exa.ai/mcp"
	defaultExaSearchURL       = "https://api.exa.ai/search"
	defaultKagiSearchURL      = "https://kagi.com/api/v1/search"
	defaultSearchModel        = "gpt-5.6-terra"

	maxConfigBytes  = 1 << 20
	maxResponseBody = 8 << 20
	maxToolOutput   = 50 << 10
	searchTimeout   = 60 * time.Second
)

// OpenAIAuth is the provider authentication web search needs. Provider is
// either openai or openai-codex; Codex credentials use the ChatGPT endpoint.
type OpenAIAuth struct {
	Provider string
	APIKey   string
	Model    string
	Headers  llm.ProviderHeaders
}

// ResolveOpenAIAuth finds a usable OpenAI or Codex credential. The session
// supplies this callback so OAuth refresh remains owned by the model catalog.
type ResolveOpenAIAuth func(context.Context) (*OpenAIAuth, error)

// Result is one cited web result.
type Result struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet,omitempty"`
}

// Response is a normalized provider response.
type Response struct {
	Provider string   `json:"provider"`
	Query    string   `json:"query"`
	Answer   string   `json:"answer"`
	Results  []Result `json:"results"`
}

// Options controls one search.
type Options struct {
	NumResults     int
	RecencyFilter  string
	DomainFilter   []string
	IncludeContent bool
	Provider       string
}

type configFile struct {
	Provider           string `json:"provider"`
	SearchProvider     string `json:"searchProvider"`
	OpenAIAPIKey       string `json:"openaiApiKey"`
	OpenAIResponsesURL string `json:"openaiResponsesUrl"`
	OpenAISearchModel  string `json:"openaiSearchModel"`
	ExaAPIKey          string `json:"exaApiKey"`
	KagiAPIKey         string `json:"kagiApiKey"`
}

// Service routes web searches and owns no background resources.
type Service struct {
	ConfigPath    string
	Client        *http.Client
	ResolveOpenAI ResolveOpenAIAuth
	Getenv        func(string) string
	openAIURL     string
	codexURL      string
	exaMCPURL     string
	exaSearchURL  string
	kagiSearchURL string
}

// New returns a search service. Config is read for each call so edits to
// web-search.json take effect without restarting GoshCoder.
func New(configPath string, resolveOpenAI ResolveOpenAIAuth) *Service {
	return &Service{
		ConfigPath:    configPath,
		Client:        &http.Client{Timeout: searchTimeout},
		ResolveOpenAI: resolveOpenAI,
		Getenv:        os.Getenv,
		openAIURL:     defaultOpenAIResponsesURL,
		codexURL:      defaultCodexResponsesURL,
		exaMCPURL:     defaultExaMCPURL,
		exaSearchURL:  defaultExaSearchURL,
		kagiSearchURL: defaultKagiSearchURL,
	}
}

// Tool returns the native web_search agent tool.
func (s *Service) Tool() agent.Tool {
	return agent.Tool{
		Name:  "web_search",
		Label: "Web Search",
		Description: "Search the web with cited sources. Supports OpenAI/Codex authentication, zero-config Exa, and Kagi. " +
			"For broad research prefer 2-4 varied queries. The configured provider is used when provider is omitted or auto.",
		Parameters: json.RawMessage(`{
			"type":"object",
			"properties":{
				"query":{"type":"string","description":"Single search query. Prefer queries for broad research."},
				"queries":{"type":"array","items":{"type":"string"},"description":"Multiple varied search queries."},
				"numResults":{"type":"number","description":"Results per query (default 5, max 20)"},
				"includeContent":{"type":"boolean","description":"Ask providers that support it to include fuller result text"},
				"recencyFilter":{"type":"string","enum":["day","week","month","year"],"description":"Filter by recency"},
				"domainFilter":{"type":"array","items":{"type":"string"},"description":"Limit to domains; prefix a domain with - to exclude it"},
				"provider":{"type":"string","enum":["auto","openai","exa","kagi"],"description":"Override the configured search provider"},
				"workflow":{"type":"string","enum":["none","summary-review","auto-summary"],"description":"Accepted for pi-web-access compatibility; native GoshCoder returns results directly"}
			}
		}`),
		Execute: func(ctx context.Context, _ string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			queries := queryParams(params)
			if len(queries) == 0 {
				return agent.ToolResult{}, errors.New("no query provided; use query or queries")
			}
			if len(queries) > 8 {
				return agent.ToolResult{}, errors.New("at most 8 queries may be searched at once")
			}
			options, err := optionsFromParams(params)
			if err != nil {
				return agent.ToolResult{}, err
			}

			responses := make([]Response, 0, len(queries))
			for index, query := range queries {
				if len(query) > 2_000 {
					return agent.ToolResult{}, fmt.Errorf("query %d exceeds 2000 characters", index+1)
				}
				if onUpdate != nil {
					onUpdate(textResult(fmt.Sprintf("Searching %d/%d: %q...", index+1, len(queries), query)))
				}
				response, err := s.Search(ctx, query, options)
				if err != nil {
					return agent.ToolResult{}, err
				}
				responses = append(responses, response)
			}
			output := formatResponses(responses)
			if len(output) > maxToolOutput {
				output = output[:maxToolOutput] + "\n\n[web search output truncated at 50 KiB]"
			}
			return agent.ToolResult{
				Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: output}},
				Details: responses,
			}, nil
		},
	}
}

// Search executes one normalized query.
func (s *Service) Search(ctx context.Context, query string, options Options) (Response, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return Response{}, errors.New("query is required")
	}
	options.NumResults = normalizeCount(options.NumResults)
	cfg, err := s.loadConfig()
	if err != nil {
		return Response{}, err
	}
	provider := strings.ToLower(strings.TrimSpace(options.Provider))
	if provider == "" || provider == "auto" {
		provider = strings.ToLower(strings.TrimSpace(firstNonempty(cfg.SearchProvider, cfg.Provider)))
	}
	if provider == "" || provider == "auto" {
		return s.searchAuto(ctx, cfg, query, options)
	}
	return s.searchProvider(ctx, cfg, provider, query, options)
}

func (s *Service) searchAuto(ctx context.Context, cfg configFile, query string, options Options) (Response, error) {
	var diagnostics []string
	// pi-web-access only prefers OpenAI for its native defaults; Exa handles
	// result-count and recency filtering more predictably.
	if options.RecencyFilter == "" && options.NumResults == 5 {
		response, available, err := s.searchOpenAI(ctx, cfg, query, options)
		if err == nil && available {
			return response, nil
		}
		if err != nil {
			if ctx.Err() != nil {
				return Response{}, ctx.Err()
			}
			diagnostics = append(diagnostics, "OpenAI: "+err.Error())
		}
	}
	response, err := s.searchExa(ctx, cfg, query, options)
	if err == nil {
		return response, nil
	}
	if ctx.Err() != nil {
		return Response{}, ctx.Err()
	}
	diagnostics = append(diagnostics, "Exa: "+err.Error())
	return Response{}, fmt.Errorf("automatic web search failed:\n  - %s", strings.Join(diagnostics, "\n  - "))
}

func (s *Service) searchProvider(ctx context.Context, cfg configFile, provider, query string, options Options) (Response, error) {
	switch provider {
	case "openai":
		response, available, err := s.searchOpenAI(ctx, cfg, query, options)
		if err != nil {
			return Response{}, err
		}
		if !available {
			return Response{}, fmt.Errorf("OpenAI web search is unavailable; log in to openai-codex, set OPENAI_API_KEY, or configure openaiApiKey in %s", s.ConfigPath)
		}
		return response, nil
	case "exa":
		return s.searchExa(ctx, cfg, query, options)
	case "kagi":
		return s.searchKagi(ctx, cfg, query, options)
	default:
		return Response{}, fmt.Errorf("unsupported web search provider %q (use auto, openai, exa, or kagi)", provider)
	}
}

func (s *Service) loadConfig() (configFile, error) {
	var cfg configFile
	if strings.TrimSpace(s.ConfigPath) == "" {
		return cfg, nil
	}
	file, err := os.Open(s.ConfigPath)
	if errors.Is(err, os.ErrNotExist) {
		return cfg, nil
	}
	if err != nil {
		return cfg, fmt.Errorf("read web search config: %w", err)
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, maxConfigBytes+1))
	if err != nil {
		return cfg, fmt.Errorf("read web search config: %w", err)
	}
	if len(content) > maxConfigBytes {
		return cfg, fmt.Errorf("web search config exceeds 1 MiB")
	}
	if len(strings.TrimSpace(string(content))) == 0 {
		return cfg, nil
	}
	if err := json.Unmarshal(content, &cfg); err != nil {
		return cfg, fmt.Errorf("parse %s: %w", s.ConfigPath, err)
	}
	return cfg, nil
}

func (s *Service) credential(configured, environment string) string {
	value := strings.TrimSpace(configured)
	if strings.HasPrefix(value, "${") && strings.HasSuffix(value, "}") {
		value = s.getenv(strings.TrimSuffix(strings.TrimPrefix(value, "${"), "}"))
	} else if strings.HasPrefix(value, "$") {
		value = s.getenv(strings.TrimPrefix(value, "$"))
	}
	if value == "" {
		value = s.getenv(environment)
	}
	return strings.TrimSpace(value)
}

func (s *Service) getenv(name string) string {
	if s.Getenv == nil {
		return os.Getenv(name)
	}
	return s.Getenv(name)
}

func (s *Service) client() *http.Client {
	if s.Client != nil {
		return s.Client
	}
	return &http.Client{Timeout: searchTimeout}
}

func queryParams(params map[string]any) []string {
	var queries []string
	if raw, ok := params["queries"].([]any); ok {
		for _, value := range raw {
			if query, ok := value.(string); ok && strings.TrimSpace(query) != "" {
				queries = append(queries, strings.TrimSpace(query))
			}
		}
	} else if raw, ok := params["queries"].([]string); ok {
		for _, query := range raw {
			if strings.TrimSpace(query) != "" {
				queries = append(queries, strings.TrimSpace(query))
			}
		}
	}
	if len(queries) == 0 {
		if query, ok := params["query"].(string); ok && strings.TrimSpace(query) != "" {
			queries = append(queries, strings.TrimSpace(query))
		}
	}
	return queries
}

func optionsFromParams(params map[string]any) (Options, error) {
	options := Options{NumResults: 5}
	switch value := params["numResults"].(type) {
	case float64:
		options.NumResults = int(value)
	case int:
		options.NumResults = value
	case json.Number:
		parsed, err := value.Int64()
		if err != nil {
			return options, errors.New("numResults must be a number")
		}
		options.NumResults = int(parsed)
	}
	options.RecencyFilter, _ = params["recencyFilter"].(string)
	if options.RecencyFilter != "" && options.RecencyFilter != "day" && options.RecencyFilter != "week" && options.RecencyFilter != "month" && options.RecencyFilter != "year" {
		return options, fmt.Errorf("invalid recencyFilter %q", options.RecencyFilter)
	}
	options.Provider, _ = params["provider"].(string)
	options.IncludeContent, _ = params["includeContent"].(bool)
	if values, ok := params["domainFilter"].([]any); ok {
		for _, value := range values {
			if domain, ok := value.(string); ok && strings.TrimSpace(domain) != "" {
				options.DomainFilter = append(options.DomainFilter, strings.TrimSpace(domain))
			}
		}
	} else if values, ok := params["domainFilter"].([]string); ok {
		options.DomainFilter = append(options.DomainFilter, values...)
	}
	return options, nil
}

func normalizeCount(value int) int {
	if value <= 0 {
		return 5
	}
	return min(value, 20)
}

func formatResponses(responses []Response) string {
	var sections []string
	for _, response := range responses {
		var section strings.Builder
		if len(responses) > 1 {
			fmt.Fprintf(&section, "## %s\n\n", response.Query)
		}
		fmt.Fprintf(&section, "Provider: %s\n", response.Provider)
		if strings.TrimSpace(response.Answer) != "" {
			section.WriteString("\n")
			section.WriteString(strings.TrimSpace(response.Answer))
			section.WriteString("\n")
		}
		if len(response.Results) > 0 {
			section.WriteString("\nSources:\n")
			for index, result := range response.Results {
				fmt.Fprintf(&section, "%d. [%s](%s)", index+1, fallbackTitle(result), result.URL)
				if snippet := compactText(result.Snippet, 500); snippet != "" {
					fmt.Fprintf(&section, " — %s", snippet)
				}
				section.WriteByte('\n')
			}
		}
		sections = append(sections, strings.TrimSpace(section.String()))
	}
	return strings.Join(sections, "\n\n---\n\n")
}

func compactText(value string, limit int) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) > limit {
		return value[:limit-3] + "..."
	}
	return value
}

func fallbackTitle(result Result) string {
	if title := strings.TrimSpace(result.Title); title != "" {
		return strings.ReplaceAll(title, "]", "\\]")
	}
	return result.URL
}

func textResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}

func firstNonempty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
