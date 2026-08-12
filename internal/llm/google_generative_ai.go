package llm

// Go port of reference/pi/packages/ai/src/api/google-generative-ai.ts.
//
// Where the TS implementation drives the @google/genai SDK, this port uses a
// hand-rolled net/http POST to
// {baseUrl}/models/{model}:streamGenerateContent?alt=sse plus the SSE reader in
// sse.go. Message/tool conversion and finish-reason mapping live in
// google_shared.go.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

const APIGoogleGenerativeAI = "google-generative-ai"

func init() {
	RegisterStreamer(APIGoogleGenerativeAI, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*GoogleOptions)
			return StreamGoogleGenerativeAI(model, ctx, o)
		},
		StreamSimple: StreamGoogleGenerativeAISimple,
	})
}

// GoogleThinkingConfig requests thinking output. Level takes precedence over
// BudgetTokens when both are set.
type GoogleThinkingConfig struct {
	Enabled bool
	// BudgetTokens is -1 for dynamic, 0 to disable. Only consulted when Level
	// is empty. nil means unset.
	BudgetTokens *int
	// Level is a Gemini 3+ thinking level.
	Level GoogleThinkingLevel
}

// GoogleOptions are the stream options for the google-generative-ai API.
type GoogleOptions struct {
	StreamOptions
	// ToolChoice is "auto" | "none" | "any".
	ToolChoice string
	// Thinking configures thought output. nil omits thinkingConfig entirely.
	Thinking *GoogleThinkingConfig
}

// StreamGoogleGenerativeAI streams a turn over the google-generative-ai API.
func StreamGoogleGenerativeAI(model *Model, conv *Context, options *GoogleOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &GoogleOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &googleStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamGoogleGenerativeAISimple maps SimpleStreamOptions onto the Google
// stream (streamSimple in google-generative-ai.ts). Unlike the TS original,
// which throws when no API key is available, the Go port surfaces that failure
// as a regular error event on the returned stream.
func StreamGoogleGenerativeAISimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)

	if options.Reasoning == "" {
		return StreamGoogleGenerativeAI(model, conv, &GoogleOptions{
			StreamOptions: base,
			Thinking:      &GoogleThinkingConfig{Enabled: false},
		})
	}

	// "off" maps to high here: pi treats an explicit reasoning request on a
	// model without an "off" level as "think as much as allowed".
	effort := clampReasoning(ClampThinkingLevel(model, options.Reasoning))
	if effort == ThinkingOff {
		effort = ThinkingHigh
	}

	if isGemini3ProModel(model) || isGemini3FlashModel(model) || isGemma4Model(model) {
		return StreamGoogleGenerativeAI(model, conv, &GoogleOptions{
			StreamOptions: base,
			Thinking:      &GoogleThinkingConfig{Enabled: true, Level: getGoogleThinkingLevel(effort, model)},
		})
	}

	budget := getGoogleBudget(model, effort, options.ThinkingBudgets)
	return StreamGoogleGenerativeAI(model, conv, &GoogleOptions{
		StreamOptions: base,
		Thinking:      &GoogleThinkingConfig{Enabled: true, BudgetTokens: &budget},
	})
}

// googleStreamer owns one request/response cycle.
type googleStreamer struct {
	model   *Model
	conv    *Context
	options *GoogleOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *googleStreamer) run() {
	p.output = &AssistantMessage{
		Role:       "assistant",
		Content:    []ContentBlock{},
		API:        APIGoogleGenerativeAI,
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

func (p *googleStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *googleStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *googleStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	// Google has no header-only auth fallback: the key is required.
	if opts.APIKey == "" {
		return fmt.Errorf("No API key for provider: %s", p.model.Provider)
	}
	if ctx.Err() != nil {
		return AbortError{}
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

	params := buildGoogleParams(p.model, p.conv, opts)
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doGoogleGenerateContentRequest(ctx, p.model, opts.APIKey, headers, params, opts.TimeoutMs)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})
	if err := consumeGoogleStream(ctx, resp.Body, p.model, p.output, p.push); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return AbortError{}
	}
	if p.output.StopReason == StopPending {
		return fmt.Errorf("Google stream ended without a finish reason")
	}
	if p.output.StopReason == StopAborted || p.output.StopReason == StopError {
		// Google reports failures as a finishReason rather than an error body.
		if p.output.RawStopReason != "" {
			return fmt.Errorf("Provider stopped with: %s", p.output.RawStopReason)
		}
		return fmt.Errorf("An unknown error occurred")
	}

	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: p.output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

// buildGoogleParams builds the generateContent request body (buildParams in
// google-generative-ai.ts).
func buildGoogleParams(model *Model, conv *Context, options *GoogleOptions) map[string]any {
	contents := convertGoogleMessages(model, conv)

	generationConfig := map[string]any{}
	if options.Temperature != nil {
		generationConfig["temperature"] = *options.Temperature
	}
	if options.MaxTokens > 0 {
		generationConfig["maxOutputTokens"] = options.MaxTokens
	}

	config := map[string]any{}
	for k, v := range generationConfig {
		config[k] = v
	}
	if conv.SystemPrompt != "" {
		config["systemInstruction"] = sanitizeSurrogates(conv.SystemPrompt)
	}
	if len(conv.Tools) > 0 {
		config["tools"] = convertGoogleTools(conv.Tools, false)
		if mode, ok := resolveGoogleFunctionCallingMode(conv.Tools, options.ToolChoice, SupportsGoogleStrictToolSampling(model.ID)); ok {
			config["toolConfig"] = map[string]any{"functionCallingConfig": map[string]any{"mode": mode}}
		}
	}

	if thinking := options.Thinking; thinking != nil && model.Reasoning {
		switch {
		case thinking.Enabled:
			thinkingConfig := map[string]any{"includeThoughts": true}
			if thinking.Level != "" {
				thinkingConfig["thinkingLevel"] = thinking.Level
			} else if thinking.BudgetTokens != nil {
				thinkingConfig["thinkingBudget"] = *thinking.BudgetTokens
			}
			config["thinkingConfig"] = thinkingConfig
		default:
			config["thinkingConfig"] = getDisabledGoogleThinkingConfig(model)
		}
	}

	params := map[string]any{
		"model":    model.ID,
		"contents": contents,
	}
	if len(config) > 0 {
		params["config"] = config
	}
	return params
}

// googleRequestBody reshapes the pi params into the REST request body. The SDK
// nests generation settings under "config"; the REST API expects
// generationConfig / systemInstruction / tools / toolConfig at the top level.
func googleRequestBody(params map[string]any) map[string]any {
	body := map[string]any{}
	for k, v := range params {
		if k == "model" || k == "config" {
			continue
		}
		body[k] = v
	}

	config, _ := params["config"].(map[string]any)
	generationConfig := map[string]any{}
	for key, value := range config {
		switch key {
		case "systemInstruction":
			body["systemInstruction"] = map[string]any{"parts": []any{map[string]any{"text": value}}}
		case "tools", "toolConfig":
			body[key] = value
		case "abortSignal":
			// Transport-level concern; carried by the request context.
		default:
			// temperature, maxOutputTokens, thinkingConfig, ...
			generationConfig[key] = value
		}
	}
	if len(generationConfig) > 0 {
		body["generationConfig"] = generationConfig
	}
	return body
}

// doGoogleGenerateContentRequest performs one POST
// {baseUrl}/models/{model}:streamGenerateContent?alt=sse. Non-2xx responses
// become *ProviderError so RetryProviderRequest can classify them.
func doGoogleGenerateContentRequest(ctx context.Context, model *Model, apiKey string, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	modelID := model.ID
	if v, ok := params["model"].(string); ok && v != "" {
		modelID = v
	}
	body, err := json.Marshal(googleRequestBody(params))
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	baseURL := strings.TrimSuffix(model.BaseURL, "/")
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com/v1beta"
	}
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", baseURL, modelID)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", apiKey)
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
// Thinking configuration
// ---------------------------------------------------------------------------

var (
	gemma4ModelRe        = regexp.MustCompile(`gemma-?4`)
	gemini3ProModelRe    = regexp.MustCompile(`gemini-3(?:\.\d+)?-pro`)
	gemini3FlashModelRe  = regexp.MustCompile(`gemini-3(?:\.\d+)?-flash`)
	gemini25ProSubstring = "2.5-pro"
)

func isGemma4Model(model *Model) bool {
	return gemma4ModelRe.MatchString(strings.ToLower(model.ID))
}

func isGemini3ProModel(model *Model) bool {
	return gemini3ProModelRe.MatchString(strings.ToLower(model.ID))
}

func isGemini3FlashModel(model *Model) bool {
	id := strings.ToLower(model.ID)
	return gemini3FlashModelRe.MatchString(id) || id == "gemini-flash-latest" || id == "gemini-flash-lite-latest"
}

// getDisabledGoogleThinkingConfig builds the "thinking off" config.
//
// Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash / Flash-Lite do
// not support full thinking-off either, so those use the lowest supported
// thinkingLevel without includeThoughts: thinking still happens but stays
// invisible. Gemini 2.x disables via thinkingBudget 0.
func getDisabledGoogleThinkingConfig(model *Model) map[string]any {
	switch {
	case isGemini3ProModel(model):
		return map[string]any{"thinkingLevel": GoogleThinkingLevelLow}
	case isGemini3FlashModel(model), isGemma4Model(model):
		return map[string]any{"thinkingLevel": GoogleThinkingLevelMinimal}
	default:
		return map[string]any{"thinkingBudget": 0}
	}
}

// getGoogleThinkingLevel maps a pi thinking level onto a Gemini level. effort
// is already clamped to minimal/low/medium/high.
func getGoogleThinkingLevel(effort ThinkingLevel, model *Model) GoogleThinkingLevel {
	if isGemini3ProModel(model) {
		// Gemini 3 Pro only exposes LOW and HIGH.
		switch effort {
		case ThinkingMinimal, ThinkingLow:
			return GoogleThinkingLevelLow
		case ThinkingMedium, ThinkingHigh:
			return GoogleThinkingLevelHigh
		}
	}
	if isGemma4Model(model) {
		switch effort {
		case ThinkingMinimal, ThinkingLow:
			return GoogleThinkingLevelMinimal
		case ThinkingMedium, ThinkingHigh:
			return GoogleThinkingLevelHigh
		}
	}
	switch effort {
	case ThinkingMinimal:
		return GoogleThinkingLevelMinimal
	case ThinkingLow:
		return GoogleThinkingLevelLow
	case ThinkingMedium:
		return GoogleThinkingLevelMedium
	default:
		return GoogleThinkingLevelHigh
	}
}

// getGoogleBudget resolves the thinking token budget for budget-based models.
// -1 means dynamic (the model decides).
func getGoogleBudget(model *Model, effort ThinkingLevel, custom *ThinkingBudgets) int {
	if custom != nil {
		if budget, ok := lookupThinkingBudget(custom, effort); ok {
			return budget
		}
	}

	id := model.ID
	switch {
	case strings.Contains(id, gemini25ProSubstring):
		return pickGoogleBudget(effort, 128, 2048, 8192, 32768)
	case strings.Contains(id, "2.5-flash-lite"):
		return pickGoogleBudget(effort, 512, 2048, 8192, 24576)
	case strings.Contains(id, "2.5-flash"):
		return pickGoogleBudget(effort, 128, 2048, 8192, 24576)
	default:
		return -1
	}
}

func pickGoogleBudget(effort ThinkingLevel, minimal, low, medium, high int) int {
	switch effort {
	case ThinkingMinimal:
		return minimal
	case ThinkingLow:
		return low
	case ThinkingMedium:
		return medium
	default:
		return high
	}
}

// lookupThinkingBudget reads a per-level override. ok is false when the level
// has no explicit entry.
func lookupThinkingBudget(budgets *ThinkingBudgets, effort ThinkingLevel) (int, bool) {
	switch effort {
	case ThinkingMinimal:
		if budgets.Minimal != 0 {
			return budgets.Minimal, true
		}
	case ThinkingLow:
		if budgets.Low != 0 {
			return budgets.Low, true
		}
	case ThinkingMedium:
		if budgets.Medium != 0 {
			return budgets.Medium, true
		}
	case ThinkingHigh:
		if budgets.High != 0 {
			return budgets.High, true
		}
	}
	return 0, false
}
