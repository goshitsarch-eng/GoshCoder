package llm

// Go port of reference/pi/packages/ai/src/api/azure-openai-responses.ts.
//
// Azure speaks the same Responses wire protocol as OpenAI, so message/tool
// conversion and stream assembly are reused from openai_responses_shared.go.
// What differs is endpoint resolution (resource name / base URL normalization),
// auth (an "api-key" header instead of a bearer token), the "api-version"
// query parameter, and the deployment-name indirection: the request body's
// "model" carries the Azure deployment, not the pi model id.
//
// DEVIATION (compat plumbing): as in the other API ports, compat travels on
// the options struct rather than model.compat, because Go's Model.Compat is
// *OpenAICompletionsCompat only.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const APIAzureOpenAIResponses = "azure-openai-responses"

func init() {
	RegisterStreamer(APIAzureOpenAIResponses, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*AzureOpenAIResponsesOptions)
			return StreamAzureOpenAIResponses(model, ctx, o)
		},
		StreamSimple: StreamAzureOpenAIResponsesSimple,
	})
}

// defaultAzureAPIVersion is the api-version used when none is configured.
const defaultAzureAPIVersion = "v1"

// azureToolCallProviders are the providers whose tool call ids carry a real
// Responses item id (AZURE_TOOL_CALL_PROVIDERS in azure-openai-responses.ts).
var azureToolCallProviders = map[string]bool{
	"openai":                 true,
	"openai-codex":           true,
	"opencode":               true,
	"azure-openai-responses": true,
}

// AzureOpenAIResponsesCompat holds compat settings for the Azure Responses
// API. Pointer fields are tri-state: nil means "use the default".
type AzureOpenAIResponsesCompat struct {
	// SupportsStrictMode defaults to true (unlike plain openai-responses,
	// which defaults to false).
	SupportsStrictMode         *bool `json:"supportsStrictMode,omitempty"`
	SupportsOpenAIGrammarTools *bool `json:"supportsOpenAIGrammarTools,omitempty"`
}

// AzureOpenAIResponsesOptions are the stream options for the
// azure-openai-responses API.
type AzureOpenAIResponsesOptions struct {
	StreamOptions
	// ReasoningEffort is "minimal" | "low" | "medium" | "high" | "xhigh" | "max".
	ReasoningEffort string
	// ReasoningSummary is "auto" | "detailed" | "concise".
	ReasoningSummary string
	// AzureAPIVersion overrides AZURE_OPENAI_API_VERSION.
	AzureAPIVersion string
	// AzureResourceName derives the endpoint as
	// https://<name>.openai.azure.com/openai/v1.
	AzureResourceName string
	// AzureBaseURL overrides AZURE_OPENAI_BASE_URL and model.BaseURL.
	AzureBaseURL string
	// AzureDeploymentName overrides the deployment map and model id.
	AzureDeploymentName string
	// Compat carries AzureOpenAIResponsesCompat (see DEVIATION note).
	Compat *AzureOpenAIResponsesCompat
}

// parseAzureDeploymentNameMap parses "modelId=deployment,other=deployment2".
func parseAzureDeploymentNameMap(value string) map[string]string {
	out := map[string]string{}
	if value == "" {
		return out
	}
	for _, entry := range strings.Split(value, ",") {
		trimmed := strings.TrimSpace(entry)
		if trimmed == "" {
			continue
		}
		modelID, deploymentName, found := strings.Cut(trimmed, "=")
		modelID, deploymentName = strings.TrimSpace(modelID), strings.TrimSpace(deploymentName)
		if !found || modelID == "" || deploymentName == "" {
			continue
		}
		out[modelID] = deploymentName
	}
	return out
}

// resolveAzureDeploymentName picks the Azure deployment for this model.
func resolveAzureDeploymentName(model *Model, options *AzureOpenAIResponsesOptions) string {
	if options != nil && options.AzureDeploymentName != "" {
		return options.AzureDeploymentName
	}
	var env map[string]string
	if options != nil {
		env = options.Env
	}
	if mapped := parseAzureDeploymentNameMap(getProviderEnvValue("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", env))[model.ID]; mapped != "" {
		return mapped
	}
	return model.ID
}

// azureHostSuffixes are the Azure-managed endpoint domains whose base paths
// get normalized to /openai/v1.
var azureHostSuffixes = []string{
	".openai.azure.com",
	".cognitiveservices.azure.com",
	".ai.azure.com",
}

// normalizeAzureBaseURL rewrites Azure-managed endpoints to the /openai/v1
// base path the Responses API expects, and leaves third-party proxy URLs
// (including their query strings) untouched.
func normalizeAzureBaseURL(baseURL string) (string, error) {
	trimmed := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	parsed, err := url.Parse(trimmed)
	// url.Parse accepts bare paths; require an absolute URL like JS `new URL`.
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("invalid Azure OpenAI base URL: %s", baseURL)
	}

	isAzureHost := false
	for _, suffix := range azureHostSuffixes {
		if strings.HasSuffix(parsed.Hostname(), suffix) {
			isAzureHost = true
			break
		}
	}

	normalizedPath := strings.TrimRight(parsed.Path, "/")
	switch normalizedPath {
	case "", "/openai", "/openai/v1/responses":
		if isAzureHost {
			parsed.Path = "/openai/v1"
			parsed.RawQuery = ""
			parsed.ForceQuery = false
		}
	}
	return strings.TrimRight(parsed.String(), "/"), nil
}

// resolveAzureConfig resolves the endpoint and api-version, preferring
// explicit options, then environment, then the resource name, then the model.
func resolveAzureConfig(model *Model, options *AzureOpenAIResponsesOptions) (baseURL string, apiVersion string, err error) {
	var env map[string]string
	if options != nil {
		env = options.Env
	}

	apiVersion = defaultAzureAPIVersion
	switch {
	case options != nil && options.AzureAPIVersion != "":
		apiVersion = options.AzureAPIVersion
	default:
		if fromEnv := getProviderEnvValue("AZURE_OPENAI_API_VERSION", env); fromEnv != "" {
			apiVersion = fromEnv
		}
	}

	resolved := ""
	if options != nil {
		resolved = strings.TrimSpace(options.AzureBaseURL)
	}
	if resolved == "" {
		resolved = strings.TrimSpace(getProviderEnvValue("AZURE_OPENAI_BASE_URL", env))
	}
	if resolved == "" {
		resourceName := ""
		if options != nil {
			resourceName = options.AzureResourceName
		}
		if resourceName == "" {
			resourceName = getProviderEnvValue("AZURE_OPENAI_RESOURCE_NAME", env)
		}
		if resourceName != "" {
			resolved = fmt.Sprintf("https://%s.openai.azure.com/openai/v1", resourceName)
		}
	}
	if resolved == "" {
		resolved = model.BaseURL
	}
	if resolved == "" {
		return "", "", fmt.Errorf("azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl")
	}

	normalized, err := normalizeAzureBaseURL(resolved)
	if err != nil {
		return "", "", err
	}
	return normalized, apiVersion, nil
}

func getAzureResponsesCompat(model *Model, options *AzureOpenAIResponsesOptions) (supportsStrictMode, supportsOpenAIGrammarTools bool) {
	var c AzureOpenAIResponsesCompat
	if options != nil && options.Compat != nil {
		c = *options.Compat
	} else {
		// Fall back to the model's raw catalog compat (see DEVIATION note).
		model.DecodeRawCompat(&c)
	}
	return boolOr(c.SupportsStrictMode, true), boolOr(c.SupportsOpenAIGrammarTools, false)
}

// StreamAzureOpenAIResponses streams a turn over the azure-openai-responses API.
func StreamAzureOpenAIResponses(model *Model, conv *Context, options *AzureOpenAIResponsesOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &AzureOpenAIResponsesOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &azureResponsesStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamAzureOpenAIResponsesSimple maps SimpleStreamOptions onto the Azure
// stream (streamSimple in azure-openai-responses.ts). Unlike the TS original,
// which throws when no API key is available, the Go port surfaces that failure
// as a regular error event on the returned stream.
func StreamAzureOpenAIResponsesSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)

	reasoningEffort := ""
	if options.Reasoning != "" {
		if clamped := ClampThinkingLevel(model, options.Reasoning); clamped != ThinkingOff {
			reasoningEffort = clamped
		}
	}
	return StreamAzureOpenAIResponses(model, conv, &AzureOpenAIResponsesOptions{
		StreamOptions:   base,
		ReasoningEffort: reasoningEffort,
	})
}

// azureResponsesStreamer owns one request/response cycle.
type azureResponsesStreamer struct {
	model   *Model
	conv    *Context
	options *AzureOpenAIResponsesOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *azureResponsesStreamer) run() {
	p.output = &AssistantMessage{
		Role:       "assistant",
		Content:    []ContentBlock{},
		API:        APIAzureOpenAIResponses,
		Provider:   p.model.Provider,
		Model:      p.model.ID,
		Usage:      zeroUsage(),
		StopReason: StopPending,
		Timestamp:  time.Now().UnixMilli(),
	}
	if err := p.streamOnce(); err != nil {
		if p.options.Ctx != nil && p.options.Ctx.Err() != nil {
			p.output.StopReason = StopAborted
		} else {
			p.output.StopReason = StopError
		}
		p.output.ErrorMessage = formatStreamError(err)
		p.es.Push(AssistantMessageEvent{Type: EventError, Reason: p.output.StopReason, Error: p.snapshot()})
		p.es.End()
	}
}

func (p *azureResponsesStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *azureResponsesStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *azureResponsesStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	// Azure has no header-only auth fallback: the key is required.
	if opts.APIKey == "" {
		return fmt.Errorf("no API key for provider: %s", p.model.Provider)
	}
	baseURL, apiVersion, err := resolveAzureConfig(p.model, opts)
	if err != nil {
		return err
	}
	_, supportsOpenAIGrammarTools := getAzureResponsesCompat(p.model, opts)
	grammarToolInputProperties, err := CreateGrammarToolInputProperties(p.conv.Tools, supportsOpenAIGrammarTools)
	if err != nil {
		return err
	}

	headers := map[string]string{}
	for k, v := range p.model.Headers {
		headers[k] = v
	}
	for k, v := range opts.Headers {
		if v == nil {
			deleteHeaderCaseInsensitive(headers, k)
			continue
		}
		headers[k] = *v
	}

	deploymentName := resolveAzureDeploymentName(p.model, opts)
	params, err := buildAzureResponsesParams(p.model, p.conv, opts, deploymentName, grammarToolInputProperties)
	if err != nil {
		return err
	}
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doAzureResponsesRequest(ctx, baseURL, apiVersion, opts.APIKey, headers, params, opts.TimeoutMs)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})

	processor := newResponsesStreamProcessor(p.model, p.output, &ResponsesStreamOptions{
		GrammarToolInputProperties: grammarToolInputProperties,
	}, p.push)
	if err := consumeResponsesSSE(ctx, resp.Body, processor); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return AbortError{}
	}
	if p.output.StopReason == StopPending {
		return fmt.Errorf("azure OpenAI Responses stream ended without a stop reason")
	}
	if p.output.StopReason == StopAborted || p.output.StopReason == StopError {
		if p.output.ErrorMessage != "" {
			return fmt.Errorf("%s", p.output.ErrorMessage)
		}
		return fmt.Errorf("an unknown error occurred")
	}

	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: p.output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

// buildAzureResponsesParams builds the Azure /responses request body
// (buildParams in azure-openai-responses.ts). Unlike plain openai-responses
// there is no deferred-tool handling, no service tier, and prompt caching is
// not gated on a retention preference.
func buildAzureResponsesParams(model *Model, conv *Context, options *AzureOpenAIResponsesOptions, deploymentName string, grammarToolInputProperties map[string]string) (map[string]any, error) {
	supportsStrictMode, supportsOpenAIGrammarTools := getAzureResponsesCompat(model, options)

	messages, err := ConvertResponsesMessages(model, conv, azureToolCallProviders, &ConvertResponsesMessagesOptions{
		GrammarToolInputProperties: grammarToolInputProperties,
	})
	if err != nil {
		return nil, err
	}

	params := map[string]any{
		// Azure routes by deployment, not by pi model id.
		"model":  deploymentName,
		"input":  messages,
		"stream": true,
		"store":  false,
	}
	if key := clampOpenAIPromptCacheKey(options.SessionID); key != "" {
		params["prompt_cache_key"] = key
	}
	if options.MaxTokens > 0 {
		params["max_output_tokens"] = max(options.MaxTokens, openAIResponsesMinOutputTokens)
	}
	if options.Temperature != nil {
		params["temperature"] = *options.Temperature
	}

	if len(conv.Tools) > 0 {
		tools, err := ConvertResponsesTools(conv.Tools, &ConvertResponsesToolsOptions{
			SupportsStrictMode:         &supportsStrictMode,
			SupportsOpenAIGrammarTools: supportsOpenAIGrammarTools,
		})
		if err != nil {
			return nil, err
		}
		params["tools"] = tools
	}

	if model.Reasoning {
		switch {
		case options.ReasoningEffort != "" || options.ReasoningSummary != "":
			effort := "medium"
			if options.ReasoningEffort != "" {
				effort = mapOrEffort(model, options.ReasoningEffort)
			}
			summary := options.ReasoningSummary
			if summary == "" {
				summary = "auto"
			}
			params["reasoning"] = map[string]any{"effort": effort, "summary": summary}
			params["include"] = []any{"reasoning.encrypted_content"}
		case offIsNotNull(model):
			params["reasoning"] = map[string]any{"effort": mapIfDefined(model, ThinkingOff, "none")}
		}
	}

	// Last so custom keys override the named request fields.
	for k, v := range options.SamplingParams {
		params[k] = v
	}
	return params, nil
}

// doAzureResponsesRequest performs one POST {baseURL}/responses?api-version=...
// Azure authenticates with an "api-key" header rather than a bearer token.
func doAzureResponsesRequest(ctx context.Context, baseURL, apiVersion, apiKey string, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	endpoint, err := url.Parse(strings.TrimSuffix(baseURL, "/") + "/responses")
	if err != nil {
		return nil, fmt.Errorf("invalid Azure OpenAI base URL: %s", baseURL)
	}
	query := endpoint.Query()
	query.Set("api-version", apiVersion)
	endpoint.RawQuery = query.Encode()
	requestURL := endpoint.String()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", apiKey)
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	client := &http.Client{}
	if timeoutMs > 0 {
		client.Timeout = time.Duration(timeoutMs) * time.Millisecond
	}
	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, AbortError{}
		}
		return nil, &ProviderError{Message: err.Error(), Cause: err}
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, MaxProviderErrorBodyChars+1))
		return nil, &ProviderError{
			Status:  resp.StatusCode,
			Headers: resp.Header,
			Body:    strings.TrimSpace(TruncateErrorText(string(respBody), MaxProviderErrorBodyChars)),
			Message: fmt.Sprintf("POST %s failed with status %d", requestURL, resp.StatusCode),
		}
	}
	return resp, nil
}
