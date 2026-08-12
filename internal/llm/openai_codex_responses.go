package llm

// Go port of reference/pi/packages/ai/src/api/openai-codex-responses.ts.
//
// DEVIATION: this implementation intentionally uses the complete SSE path and
// omits WebSocket session reuse and optional zstd request compression. The
// Codex backend supports uncompressed JSON over SSE, which keeps this stdlib-
// only port functional without weakening the protocol surface visible to the
// agent.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
	"time"
)

const (
	APIOpenAICodexResponses = "openai-codex-responses"
	defaultCodexBaseURL     = "https://chatgpt.com/backend-api"
	codexJWTClaimPath       = "https://api.openai.com/auth"
)

func init() {
	RegisterStreamer(APIOpenAICodexResponses, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*OpenAICodexResponsesOptions)
			return StreamOpenAICodexResponses(model, ctx, o)
		},
		StreamSimple: StreamOpenAICodexResponsesSimple,
	})
}

var codexToolCallProviders = map[string]bool{
	"openai": true, "openai-codex": true, "opencode": true,
}

// OpenAICodexResponsesOptions configures the ChatGPT Codex Responses API.
type OpenAICodexResponsesOptions struct {
	StreamOptions
	ReasoningEffort  string
	ReasoningSummary string
	ServiceTier      string
	TextVerbosity    string
	ToolChoice       string
}

func StreamOpenAICodexResponses(model *Model, conv *Context, options *OpenAICodexResponsesOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &OpenAICodexResponsesOptions{}
	}
	es := NewAssistantMessageEventStream()
	streamer := &codexResponsesStreamer{model: model, conv: conv, options: options, es: es}
	go streamer.run()
	return es
}

func StreamOpenAICodexResponsesSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
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
	return StreamOpenAICodexResponses(model, conv, &OpenAICodexResponsesOptions{
		StreamOptions:   base,
		ReasoningEffort: reasoningEffort,
	})
}

type codexResponsesStreamer struct {
	model   *Model
	conv    *Context
	options *OpenAICodexResponsesOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *codexResponsesStreamer) run() {
	p.output = &AssistantMessage{
		Role: "assistant", Content: []ContentBlock{}, API: APIOpenAICodexResponses,
		Provider: p.model.Provider, Model: p.model.ID, Usage: zeroUsage(),
		StopReason: StopPending, Timestamp: time.Now().UnixMilli(),
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

func (p *codexResponsesStreamer) snapshot() *AssistantMessage {
	copy := *p.output
	copy.Content = append([]ContentBlock{}, p.output.Content...)
	return &copy
}

func (p *codexResponsesStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *codexResponsesStreamer) streamOnce() error {
	ctx := p.options.context()
	apiKey, err := getClientAPIKey(p.model.Provider, p.options.APIKey, p.options.Headers)
	if err != nil {
		return err
	}
	accountID, err := extractCodexAccountID(apiKey)
	if err != nil {
		return err
	}
	compat := getCodexResponsesCompat(p.model)
	grammarProperties, err := CreateGrammarToolInputProperties(p.conv.Tools, compat.SupportsOpenAIGrammarTools)
	if err != nil {
		return err
	}
	cacheSessionID := p.options.SessionID
	if resolveCacheRetention(p.options.CacheRetention, p.options.Env) == CacheRetentionNone {
		cacheSessionID = ""
	}
	cacheSessionID = clampOpenAIPromptCacheKey(cacheSessionID)
	params, err := buildCodexResponsesParams(p.model, p.conv, p.options, compat, cacheSessionID, grammarProperties)
	if err != nil {
		return err
	}
	if p.options.OnPayload != nil {
		if next := p.options.OnPayload(params, p.model); next != nil {
			params = next
		}
	}
	headers := buildCodexSSEHeaders(p.model, p.options.Headers, accountID, apiKey, cacheSessionID)
	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doCodexResponsesRequest(ctx, p.model.BaseURL, headers, params, p.options.TimeoutMs)
	}, p.options.MaxRetries, p.options.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if p.options.OnResponse != nil {
		p.options.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})
	processor := newResponsesStreamProcessor(p.model, p.output, &ResponsesStreamOptions{
		ServiceTier:                p.options.ServiceTier,
		GrammarToolInputProperties: grammarProperties,
		ResolveServiceTier:         resolveCodexServiceTier,
		ApplyServiceTierPricing: func(usage *Usage, tier string) {
			applyResponsesServiceTierPricing(usage, tier, p.model)
		},
	}, p.push)
	if err := consumeCodexResponsesSSE(ctx, resp.Body, processor, p.output); err != nil {
		return err
	}
	if ctx.Err() != nil {
		return AbortError{}
	}
	if p.output.StopReason == StopPending {
		return errors.New("Codex stream ended without a stop reason")
	}
	if p.output.StopReason == StopError || p.output.StopReason == StopAborted {
		if p.output.ErrorMessage != "" {
			return errors.New(p.output.ErrorMessage)
		}
		return errors.New("an unknown error occurred")
	}
	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: p.output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

func getCodexResponsesCompat(model *Model) responsesResolvedCompat {
	var compat OpenAIResponsesCompat
	model.DecodeRawCompat(&compat)
	return responsesResolvedCompat{
		SupportsDeveloperRole:      boolOr(compat.SupportsDeveloperRole, true),
		SupportsStrictMode:         boolOr(compat.SupportsStrictMode, true),
		SupportsOpenAIGrammarTools: boolOr(compat.SupportsOpenAIGrammarTools, false),
		SupportsAdditionalTools:    boolOr(compat.SupportsAdditionalTools, false),
		SupportsToolSearch:         boolOr(compat.SupportsToolSearch, false),
	}
}

func buildCodexResponsesParams(model *Model, conv *Context, options *OpenAICodexResponsesOptions, compat responsesResolvedCompat, cacheSessionID string, grammarProperties map[string]string) (map[string]any, error) {
	deferredMode := ""
	if compat.SupportsAdditionalTools {
		deferredMode = "additional-tools"
	} else if compat.SupportsToolSearch {
		deferredMode = "tool-search"
	}
	immediate, deferred := splitResponsesDeferredTools(conv, deferredMode != "")
	toolOptions := ConvertResponsesToolsOptions{
		StrictNull: true, SupportsStrictMode: &compat.SupportsStrictMode,
		SupportsOpenAIGrammarTools: compat.SupportsOpenAIGrammarTools,
	}
	includeSystem := false
	messages, err := ConvertResponsesMessages(model, conv, codexToolCallProviders, &ConvertResponsesMessagesOptions{
		IncludeSystemPrompt: &includeSystem, GrammarToolInputProperties: grammarProperties,
		DeferredTools: deferred, DeferredToolsMode: deferredMode, ToolOptions: toolOptions,
	})
	if err != nil {
		return nil, err
	}
	instructions := conv.SystemPrompt
	if instructions == "" {
		instructions = "You are a helpful assistant."
	}
	verbosity := options.TextVerbosity
	if verbosity == "" {
		verbosity = "low"
	}
	toolChoice := options.ToolChoice
	if toolChoice == "" {
		toolChoice = "auto"
	}
	params := map[string]any{
		"model": model.ID, "store": false, "stream": true,
		"instructions": instructions, "input": messages,
		"text":        map[string]any{"verbosity": verbosity},
		"include":     []any{"reasoning.encrypted_content"},
		"tool_choice": toolChoice, "parallel_tool_calls": true,
	}
	if cacheSessionID != "" {
		params["prompt_cache_key"] = cacheSessionID
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
	if options.ReasoningEffort != "" {
		effort := mapOrEffort(model, options.ReasoningEffort)
		summary := options.ReasoningSummary
		if summary == "" {
			summary = "auto"
		}
		params["reasoning"] = map[string]any{"effort": effort, "summary": summary}
	}
	for key, value := range options.SamplingParams {
		params[key] = value
	}
	return params, nil
}

func resolveCodexURL(baseURL string) string {
	normalized := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if normalized == "" {
		normalized = defaultCodexBaseURL
	}
	if strings.HasSuffix(normalized, "/codex/responses") {
		return normalized
	}
	if strings.HasSuffix(normalized, "/codex") {
		return normalized + "/responses"
	}
	return normalized + "/codex/responses"
}

func extractCodexAccountID(token string) (string, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return "", errors.New("failed to extract accountId from token")
	}
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return "", errors.New("failed to extract accountId from token")
	}
	var payload map[string]any
	if json.Unmarshal(decoded, &payload) != nil {
		return "", errors.New("failed to extract accountId from token")
	}
	claim, ok := payload[codexJWTClaimPath].(map[string]any)
	if !ok {
		return "", errors.New("failed to extract accountId from token")
	}
	accountID, ok := claim["chatgpt_account_id"].(string)
	if !ok || accountID == "" {
		return "", errors.New("failed to extract accountId from token")
	}
	return accountID, nil
}

func buildCodexSSEHeaders(model *Model, additional ProviderHeaders, accountID, token, sessionID string) map[string]string {
	headers := map[string]string{}
	for key, value := range model.Headers {
		headers[key] = value
	}
	for key, value := range additional {
		if value == nil {
			deleteHeaderCaseInsensitive(headers, key)
		} else {
			headers[key] = *value
		}
	}
	headers["Authorization"] = "Bearer " + token
	headers["chatgpt-account-id"] = accountID
	headers["originator"] = "goshcoder"
	headers["User-Agent"] = fmt.Sprintf("goshcoder (%s; %s)", runtime.GOOS, runtime.GOARCH)
	headers["OpenAI-Beta"] = "responses=experimental"
	headers["Accept"] = "text/event-stream"
	headers["Content-Type"] = "application/json"
	if sessionID != "" {
		headers["session-id"] = sessionID
		headers["x-client-request-id"] = sessionID
	}
	return headers
}

func doCodexResponsesRequest(ctx context.Context, baseURL string, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	endpoint := resolveCodexURL(baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for key, value := range headers {
		req.Header.Set(key, value)
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
		responseBody, _ := io.ReadAll(io.LimitReader(resp.Body, MaxProviderErrorBodyChars+1))
		return nil, &ProviderError{Status: resp.StatusCode, Headers: resp.Header,
			Body:    strings.TrimSpace(TruncateErrorText(string(responseBody), MaxProviderErrorBodyChars)),
			Message: fmt.Sprintf("POST %s failed with status %d", endpoint, resp.StatusCode)}
	}
	return resp, nil
}

func consumeCodexResponsesSSE(ctx context.Context, body io.Reader, processor *responsesStreamProcessor, output *AssistantMessage) error {
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
			return fmt.Errorf("reading Codex SSE stream: %w", err)
		}
		if sse.Data == "" || sse.Data == "[DONE]" {
			continue
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal([]byte(sse.Data), &raw); err != nil {
			return fmt.Errorf("invalid Codex SSE JSON: %w", err)
		}
		var event responsesStreamEvent
		if err := json.Unmarshal([]byte(sse.Data), &event); err != nil {
			return fmt.Errorf("invalid Codex SSE event: %w", err)
		}
		if event.Type == "" {
			event.Type = sse.Event
		}
		if event.Type == "error" && event.Message == "" {
			var nested struct {
				Error struct {
					Code, Message string
				} `json:"error"`
			}
			_ = json.Unmarshal([]byte(sse.Data), &nested)
			event.Code, event.Message = nested.Error.Code, nested.Error.Message
		}
		if event.Type == "response.done" || event.Type == "response.completed" || event.Type == "response.incomplete" {
			event.Type = "response.completed"
			if event.Response != nil {
				event.Response.Status = normalizeCodexResponseStatus(event.Response.Status)
				if event.Response.EndTurn != nil {
					output.EndTurn = event.Response.EndTurn
				}
			}
		}
		if err := processor.ProcessEvent(&event); err != nil {
			return err
		}
		if event.Type == "response.completed" || event.Type == "response.failed" {
			break
		}
	}
	if !processor.sawTerminalResponseEvent {
		return errors.New("Codex stream ended before a terminal response event")
	}
	return nil
}

func normalizeCodexResponseStatus(status string) string {
	switch status {
	case "completed", "incomplete", "failed", "cancelled", "queued", "in_progress":
		return status
	default:
		return ""
	}
}

func resolveCodexServiceTier(responseTier, requestTier string) string {
	if responseTier == "default" && (requestTier == "flex" || requestTier == "priority") {
		return requestTier
	}
	if responseTier != "" {
		return responseTier
	}
	return requestTier
}
