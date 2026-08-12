package llm

// Go port of reference/pi/packages/ai/src/api/openai-responses.ts.
//
// Where the TS implementation drives the OpenAI SDK, this port uses a
// hand-rolled net/http POST to {baseUrl}/responses plus the SSE reader in
// sse.go. Message/tool conversion and stream assembly live in
// openai_responses_shared.go.
//
// DEVIATION (compat plumbing): pi keys model.compat by API. Go's Model.Compat
// is *OpenAICompletionsCompat only, so OpenAIResponsesCompat travels on
// OpenAIResponsesOptions.Compat instead, exactly as the anthropic-messages
// port does. A consequence: StreamSimple callers cannot inject compat
// overrides, since SimpleStreamOptions is shared.
//
// Deferred from the TS original: the github-copilot dynamic headers.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const APIOpenAIResponses = "openai-responses"

func init() {
	RegisterStreamer(APIOpenAIResponses, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*OpenAIResponsesOptions)
			return StreamOpenAIResponses(model, ctx, o)
		},
		StreamSimple: StreamOpenAIResponsesSimple,
	})
}

// openAIToolCallProviders are the providers whose tool call ids carry a real
// Responses item id (OPENAI_TOOL_CALL_PROVIDERS in openai-responses.ts).
var openAIToolCallProviders = map[string]bool{
	"openai":       true,
	"openai-codex": true,
	"opencode":     true,
}

// openAIResponsesMinOutputTokens is the floor the API enforces on
// max_output_tokens.
const openAIResponsesMinOutputTokens = 16

// OpenAIResponsesCompat holds compatibility settings for the Responses API.
// Pointer fields are tri-state: nil means "use the default" (pi's undefined).
type OpenAIResponsesCompat struct {
	SupportsDeveloperRole *bool `json:"supportsDeveloperRole,omitempty"`
	// SessionAffinityFormat is "openai" or "openrouter".
	SessionAffinityFormat           string `json:"sessionAffinityFormat,omitempty"`
	SupportsLongCacheRetention      *bool  `json:"supportsLongCacheRetention,omitempty"`
	SupportsStrictMode              *bool  `json:"supportsStrictMode,omitempty"`
	SupportsOpenAIGrammarTools      *bool  `json:"supportsOpenAIGrammarTools,omitempty"`
	SupportsAdditionalTools         *bool  `json:"supportsAdditionalTools,omitempty"`
	SupportsToolSearch              *bool  `json:"supportsToolSearch,omitempty"`
	SupportsExplicitPromptCacheMode *bool  `json:"supportsExplicitPromptCacheMode,omitempty"`
}

// OpenAIResponsesOptions are the stream options for the openai-responses API.
type OpenAIResponsesOptions struct {
	StreamOptions
	// ReasoningEffort is "minimal" | "low" | "medium" | "high" | "xhigh" | "max".
	ReasoningEffort string
	// ReasoningSummary is "auto" | "detailed" | "concise"; defaults to "auto"
	// when reasoning is requested.
	ReasoningSummary string
	// ServiceTier is "auto" | "default" | "flex" | "priority".
	ServiceTier string
	// ToolChoice is "auto" | "none" | "required" or a tool-selecting map.
	ToolChoice any
	// Compat carries OpenAIResponsesCompat on the options (see DEVIATION note).
	Compat *OpenAIResponsesCompat
}

// responsesResolvedCompat is OpenAIResponsesCompat after default resolution.
type responsesResolvedCompat struct {
	SupportsDeveloperRole           bool
	SessionAffinityFormat           string
	SupportsLongCacheRetention      bool
	SupportsStrictMode              bool
	SupportsOpenAIGrammarTools      bool
	SupportsAdditionalTools         bool
	SupportsToolSearch              bool
	SupportsExplicitPromptCacheMode bool
}

// detectResponsesSessionAffinityFormat picks the session header convention
// from the provider/base URL.
func detectResponsesSessionAffinityFormat(model *Model) string {
	if model.Provider == "openrouter" || strings.Contains(model.BaseURL, "openrouter.ai") {
		return "openrouter"
	}
	return "openai"
}

func getResponsesCompat(model *Model, options *OpenAIResponsesOptions) responsesResolvedCompat {
	var c OpenAIResponsesCompat
	if options != nil && options.Compat != nil {
		c = *options.Compat
	} else {
		// Fall back to the model's raw catalog compat (see DEVIATION note).
		model.DecodeRawCompat(&c)
	}
	sessionAffinityFormat := c.SessionAffinityFormat
	if sessionAffinityFormat == "" {
		sessionAffinityFormat = detectResponsesSessionAffinityFormat(model)
	}
	return responsesResolvedCompat{
		SupportsDeveloperRole:           boolOr(c.SupportsDeveloperRole, true),
		SessionAffinityFormat:           sessionAffinityFormat,
		SupportsLongCacheRetention:      boolOr(c.SupportsLongCacheRetention, true),
		SupportsStrictMode:              boolOr(c.SupportsStrictMode, false),
		SupportsOpenAIGrammarTools:      boolOr(c.SupportsOpenAIGrammarTools, false),
		SupportsAdditionalTools:         boolOr(c.SupportsAdditionalTools, false),
		SupportsToolSearch:              boolOr(c.SupportsToolSearch, false),
		SupportsExplicitPromptCacheMode: boolOr(c.SupportsExplicitPromptCacheMode, false),
	}
}

// getResponsesPromptCacheRetention returns the prompt_cache_retention value,
// or "" to omit it.
func getResponsesPromptCacheRetention(compat responsesResolvedCompat, cacheRetention CacheRetention) string {
	if cacheRetention == CacheRetentionLong && compat.SupportsLongCacheRetention {
		return "24h"
	}
	return ""
}

// StreamOpenAIResponses streams a turn over the openai-responses API.
func StreamOpenAIResponses(model *Model, conv *Context, options *OpenAIResponsesOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &OpenAIResponsesOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &responsesStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamOpenAIResponsesSimple maps SimpleStreamOptions onto the
// openai-responses stream (streamSimple in openai-responses.ts). Unlike the TS
// original, which throws when no API key is available, the Go port surfaces
// that failure as a regular error event on the returned stream.
func StreamOpenAIResponsesSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
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
	return StreamOpenAIResponses(model, conv, &OpenAIResponsesOptions{
		StreamOptions:   base,
		ReasoningEffort: reasoningEffort,
	})
}

// responsesStreamer owns one request/response cycle.
type responsesStreamer struct {
	model   *Model
	conv    *Context
	options *OpenAIResponsesOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *responsesStreamer) run() {
	p.output = &AssistantMessage{
		Role:       "assistant",
		Content:    []ContentBlock{},
		API:        p.model.API,
		Provider:   p.model.Provider,
		Model:      p.model.ID,
		Usage:      zeroUsage(),
		StopReason: StopPending,
		Timestamp:  time.Now().UnixMilli(),
	}
	if err := p.streamOnce(); err != nil {
		// Catch block of stream() in openai-responses.ts.
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

func (p *responsesStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *responsesStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *responsesStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	apiKey, err := getClientAPIKey(p.model.Provider, opts.APIKey, opts.Headers)
	if err != nil {
		return err
	}
	compat := getResponsesCompat(p.model, opts)
	cacheRetention := resolveCacheRetention(opts.CacheRetention, opts.Env)
	cacheSessionID := opts.SessionID
	if cacheRetention == CacheRetentionNone {
		cacheSessionID = ""
	}
	grammarToolInputProperties, err := CreateGrammarToolInputProperties(p.conv.Tools, compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return err
	}

	headers := buildResponsesRequestHeaders(p.model, opts.Headers, cacheSessionID, compat)
	params, err := buildResponsesParams(p.model, p.conv, opts, compat, cacheRetention, grammarToolInputProperties)
	if err != nil {
		return err
	}
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doResponsesRequest(ctx, p.model, apiKey, headers, params, opts.TimeoutMs)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})
	if err := p.consumeStream(ctx, resp.Body, grammarToolInputProperties); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return AbortError{}
	}
	if p.output.StopReason == StopPending {
		return fmt.Errorf("OpenAI Responses stream ended without a stop reason")
	}
	if p.output.StopReason == StopAborted || p.output.StopReason == StopError {
		if p.output.ErrorMessage != "" {
			return fmt.Errorf("%s", p.output.ErrorMessage)
		}
		return fmt.Errorf("An unknown error occurred")
	}

	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: p.output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

// consumeStream folds the SSE body into the output message.
func (p *responsesStreamer) consumeStream(ctx context.Context, body io.Reader, grammarToolInputProperties map[string]string) error {
	processor := newResponsesStreamProcessor(p.model, p.output, &ResponsesStreamOptions{
		ServiceTier:                p.options.ServiceTier,
		GrammarToolInputProperties: grammarToolInputProperties,
		ApplyServiceTierPricing: func(usage *Usage, serviceTier string) {
			applyResponsesServiceTierPricing(usage, serviceTier, p.model)
		},
	}, p.push)
	return consumeResponsesSSE(ctx, body, processor)
}

// consumeResponsesSSE folds a Responses SSE body into processor. Shared by the
// openai-responses and azure-openai-responses protocols.
func consumeResponsesSSE(ctx context.Context, body io.Reader, processor *responsesStreamProcessor) error {
	reader := NewSSEReader(body)
	for {
		sse, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			if ctx.Err() != nil {
				return AbortError{}
			}
			return fmt.Errorf("reading SSE stream: %w", err)
		}
		if sse.Data == "" || sse.Data == "[DONE]" {
			continue
		}
		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(sse.Data), &event); err != nil {
			return fmt.Errorf("parsing SSE data: %w", err)
		}
		// The "event:" field mirrors the payload's own type; the payload wins.
		if event.Type == "" {
			event.Type = sse.Event
		}
		if err := processor.ProcessEvent(&event); err != nil {
			return err
		}
	}
	if !processor.sawTerminalResponseEvent {
		return fmt.Errorf("OpenAI Responses stream ended before a terminal response event")
	}
	return nil
}

// buildResponsesRequestHeaders merges model headers, session-affinity headers,
// and option headers (createClient in openai-responses.ts). A nil
// option-header value suppresses a default header with the same name.
func buildResponsesRequestHeaders(model *Model, optionsHeaders ProviderHeaders, sessionID string, compat responsesResolvedCompat) map[string]string {
	headers := map[string]string{}
	for k, v := range model.Headers {
		headers[k] = v
	}

	if sessionID != "" {
		if compat.SessionAffinityFormat == "openrouter" {
			headers["x-session-id"] = sessionID
		} else {
			if compat.SessionAffinityFormat == "openai" {
				headers["session_id"] = sessionID
			}
			headers["x-client-request-id"] = sessionID
		}
	}

	for k, v := range optionsHeaders {
		if v == nil {
			deleteHeaderCaseInsensitive(headers, k)
			continue
		}
		headers[k] = *v
	}
	return headers
}

// buildResponsesParams builds the /responses request body (buildParams in
// openai-responses.ts).
func buildResponsesParams(model *Model, conv *Context, options *OpenAIResponsesOptions, compat responsesResolvedCompat, cacheRetention CacheRetention, grammarToolInputProperties map[string]string) (map[string]any, error) {
	deferredToolsMode := ""
	switch {
	case compat.SupportsAdditionalTools:
		deferredToolsMode = "additional-tools"
	case compat.SupportsToolSearch:
		deferredToolsMode = "tool-search"
	}
	immediate, deferred := splitResponsesDeferredTools(conv, deferredToolsMode != "")

	toolOptions := ConvertResponsesToolsOptions{
		SupportsStrictMode:         &compat.SupportsStrictMode,
		SupportsOpenAIGrammarTools: compat.SupportsOpenAIGrammarTools,
	}
	messages, err := ConvertResponsesMessages(model, conv, openAIToolCallProviders, &ConvertResponsesMessagesOptions{
		SupportsDeveloperRole:      &compat.SupportsDeveloperRole,
		GrammarToolInputProperties: grammarToolInputProperties,
		DeferredTools:              deferred,
		DeferredToolsMode:          deferredToolsMode,
		ToolOptions:                toolOptions,
	})
	if err != nil {
		return nil, err
	}

	params := map[string]any{
		"model":  model.ID,
		"input":  messages,
		"stream": true,
		"store":  false,
	}
	if cacheRetention != CacheRetentionNone {
		if key := clampOpenAIPromptCacheKey(options.SessionID); key != "" {
			params["prompt_cache_key"] = key
		}
	}
	if retention := getResponsesPromptCacheRetention(compat, cacheRetention); retention != "" {
		params["prompt_cache_retention"] = retention
	}
	if cacheRetention == CacheRetentionNone && compat.SupportsExplicitPromptCacheMode {
		params["prompt_cache_options"] = map[string]any{"mode": "explicit"}
	}

	if options.MaxTokens > 0 {
		params["max_output_tokens"] = max(options.MaxTokens, openAIResponsesMinOutputTokens)
	}
	if options.Temperature != nil {
		params["temperature"] = *options.Temperature
	}
	if options.ServiceTier != "" {
		params["service_tier"] = options.ServiceTier
	}

	if len(immediate) > 0 {
		tools, err := ConvertResponsesTools(immediate, &toolOptions)
		if err != nil {
			return nil, err
		}
		params["tools"] = tools
	}
	if options.ToolChoice != nil {
		params["tool_choice"] = options.ToolChoice
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
		case model.Provider != "github-copilot" && offIsNotNull(model):
			params["reasoning"] = map[string]any{"effort": mapIfDefined(model, ThinkingOff, "none")}
		}
		if model.Provider == "xai" {
			params["include"] = []any{"reasoning.encrypted_content"}
		}
	}

	// Last so custom keys override the named request fields.
	for k, v := range options.SamplingParams {
		params[k] = v
	}
	return params, nil
}

// splitResponsesDeferredTools splits tools into those sent upfront and those
// loaded mid-transcript (splitDeferredTools in utils/deferred-tools.ts).
func splitResponsesDeferredTools(conv *Context, enabled bool) (immediate []Tool, deferred map[string]Tool) {
	var order []string
	unique := map[string]Tool{}
	for _, tool := range conv.Tools {
		if _, seen := unique[tool.Name]; !seen {
			order = append(order, tool.Name)
		}
		unique[tool.Name] = tool
	}
	if !enabled {
		for _, name := range order {
			immediate = append(immediate, unique[name])
		}
		return immediate, map[string]Tool{}
	}

	// A tool is deferred when a tool result announced it and no earlier
	// assistant turn had already called it.
	deferredNames := map[string]bool{}
	usedNames := map[string]bool{}
	for _, msg := range conv.Messages {
		switch m := derefResponsesMessage(msg).(type) {
		case AssistantMessage:
			for _, block := range m.Content {
				if tc, ok := block.(ToolCall); ok {
					usedNames[tc.Name] = true
				}
			}
		case ToolResultMessage:
			for _, name := range m.AddedToolNames {
				if !usedNames[name] {
					deferredNames[name] = true
				}
			}
		}
	}

	deferred = map[string]Tool{}
	for _, name := range order {
		if deferredNames[name] {
			deferred[name] = unique[name]
		} else {
			immediate = append(immediate, unique[name])
		}
	}
	return immediate, deferred
}

// doResponsesRequest performs one POST {baseUrl}/responses and returns the
// streaming response. Non-2xx responses become *ProviderError so
// RetryProviderRequest can classify them.
func doResponsesRequest(ctx context.Context, model *Model, apiKey string, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := strings.TrimSuffix(model.BaseURL, "/") + "/responses"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
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
			Message: fmt.Sprintf("POST %s failed with status %d", url, resp.StatusCode),
		}
	}
	return resp, nil
}

// ---------------------------------------------------------------------------
// Service tier pricing
// ---------------------------------------------------------------------------

// responsesServiceTierCostMultiplier is the price multiplier for a tier.
func responsesServiceTierCostMultiplier(model *Model, serviceTier string) float64 {
	switch serviceTier {
	case "flex":
		return 0.5
	case "priority":
		if model.ID == "gpt-5.5" {
			return 2.5
		}
		return 2
	default:
		return 1
	}
}

// applyResponsesServiceTierPricing scales usage cost by the tier multiplier.
func applyResponsesServiceTierPricing(usage *Usage, serviceTier string, model *Model) {
	multiplier := responsesServiceTierCostMultiplier(model, serviceTier)
	if multiplier == 1 {
		return
	}
	usage.Cost.Input *= multiplier
	usage.Cost.Output *= multiplier
	usage.Cost.CacheRead *= multiplier
	usage.Cost.CacheWrite *= multiplier
	usage.Cost.Total = usage.Cost.Input + usage.Cost.Output + usage.Cost.CacheRead + usage.Cost.CacheWrite
}
