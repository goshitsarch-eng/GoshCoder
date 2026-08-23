package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
	"unicode/utf8"
)

// Go port of reference/pi/packages/ai/src/api/openai-completions.ts.
//
// Where the TS implementation drives the OpenAI SDK, this port uses a
// hand-rolled net/http POST plus the SSE reader in sse.go. Request-body
// construction, streaming assembly, compat detection, and error semantics
// mirror the TS line for line where behavior is observable.

// OpenAICompletionsCompat holds compatibility settings for OpenAI-compatible
// completions APIs. Pointer fields are tri-state: nil means "auto-detect from
// provider/baseUrl" (pi's undefined). Port of OpenAICompletionsCompat in
// types.ts; see that file for the per-field semantics.
type OpenAICompletionsCompat struct {
	SupportsStore                               *bool  `json:"supportsStore,omitempty"`
	SupportsDeveloperRole                       *bool  `json:"supportsDeveloperRole,omitempty"`
	SupportsReasoningEffort                     *bool  `json:"supportsReasoningEffort,omitempty"`
	SupportsUsageInStreaming                    *bool  `json:"supportsUsageInStreaming,omitempty"`
	SupportsFinishReason                        *bool  `json:"supportsFinishReason,omitempty"`
	MaxTokensField                              string `json:"maxTokensField,omitempty"` // "max_completion_tokens" | "max_tokens"
	RequiresToolResultName                      *bool  `json:"requiresToolResultName,omitempty"`
	RequiresAssistantAfterToolResult            *bool  `json:"requiresAssistantAfterToolResult,omitempty"`
	RequiresThinkingAsText                      *bool  `json:"requiresThinkingAsText,omitempty"`
	RequiresReasoningContentOnAssistantMessages *bool  `json:"requiresReasoningContentOnAssistantMessages,omitempty"`
	// "openai" | "openrouter" | "deepseek" | "together" | "baseten" | "zai" |
	// "qwen" | "chat-template" | "qwen-chat-template" | "string-thinking" | "ant-ling"
	ThinkingFormat              string                `json:"thinkingFormat,omitempty"`
	ChatTemplateKwargs          map[string]any        `json:"chatTemplateKwargs,omitempty"`
	ChatTemplateArgs            map[string]any        `json:"chatTemplateArgs,omitempty"`
	OpenRouterRouting           *OpenRouterRouting    `json:"openRouterRouting,omitempty"`
	VercelGatewayRouting        *VercelGatewayRouting `json:"vercelGatewayRouting,omitempty"`
	ZaiToolStream               *bool                 `json:"zaiToolStream,omitempty"`
	SupportsThinkingTokenBudget *bool                 `json:"supportsThinkingTokenBudget,omitempty"`
	SupportsOpenAIGrammarTools  *bool                 `json:"supportsOpenAIGrammarTools,omitempty"`
	SupportsStrictMode          *bool                 `json:"supportsStrictMode,omitempty"`
	// Cache control convention for prompt caching: "" or "anthropic".
	CacheControlFormat         string `json:"cacheControlFormat,omitempty"`
	SendSessionAffinityHeaders *bool  `json:"sendSessionAffinityHeaders,omitempty"`
	// Provider-specific deferred tool serialization mode: "" or "kimi".
	DeferredToolsMode string `json:"deferredToolsMode,omitempty"`
	// "openai" | "openai-nosession" | "openrouter"
	SessionAffinityFormat      string `json:"sessionAffinityFormat,omitempty"`
	SupportsLongCacheRetention *bool  `json:"supportsLongCacheRetention,omitempty"`
}

// OpenRouterRouting controls which upstream providers OpenRouter routes
// requests to (types.ts; sent as the "provider" request field).
type OpenRouterRouting struct {
	AllowFallbacks         *bool    `json:"allow_fallbacks,omitempty"`
	RequireParameters      *bool    `json:"require_parameters,omitempty"`
	DataCollection         string   `json:"data_collection,omitempty"` // "deny" | "allow"
	ZDR                    *bool    `json:"zdr,omitempty"`
	EnforceDistillableText *bool    `json:"enforce_distillable_text,omitempty"`
	Order                  []string `json:"order,omitempty"`
	Only                   []string `json:"only,omitempty"`
	Ignore                 []string `json:"ignore,omitempty"`
	Quantizations          []string `json:"quantizations,omitempty"`
	// Sort is a string ("price", "throughput", "latency") or an object with
	// by/partition; kept as any like the TS union.
	Sort any `json:"sort,omitempty"`
	// MaxPrice holds per-million-token USD prices (prompt, completion, image,
	// audio, request); values are numbers or strings.
	MaxPrice map[string]any `json:"max_price,omitempty"`
	// PreferredMinThroughput / PreferredMaxLatency are numbers or
	// percentile objects; kept as any like the TS union.
	PreferredMinThroughput any `json:"preferred_min_throughput,omitempty"`
	PreferredMaxLatency    any `json:"preferred_max_latency,omitempty"`
}

// VercelGatewayRouting controls Vercel AI Gateway routing (types.ts).
type VercelGatewayRouting struct {
	Only  []string `json:"only,omitempty"`
	Order []string `json:"order,omitempty"`
}

// resolvedCompat is the concrete form of OpenAICompletionsCompat after
// auto-detection and override resolution (ResolvedOpenAICompletionsCompat).
type resolvedCompat struct {
	SupportsStore                               bool
	SupportsDeveloperRole                       bool
	SupportsReasoningEffort                     bool
	SupportsUsageInStreaming                    bool
	SupportsFinishReason                        bool
	MaxTokensField                              string
	RequiresToolResultName                      bool
	RequiresAssistantAfterToolResult            bool
	RequiresThinkingAsText                      bool
	RequiresReasoningContentOnAssistantMessages bool
	ThinkingFormat                              string
	ChatTemplateKwargs                          map[string]any
	ChatTemplateArgs                            map[string]any
	OpenRouterRouting                           *OpenRouterRouting
	VercelGatewayRouting                        *VercelGatewayRouting
	ZaiToolStream                               bool
	SupportsThinkingTokenBudget                 bool
	SupportsStrictMode                          bool
	SupportsOpenAIGrammarTools                  bool
	CacheControlFormat                          string
	SendSessionAffinityHeaders                  bool
	DeferredToolsMode                           string
	SessionAffinityFormat                       string
	SupportsLongCacheRetention                  bool
}

// OpenAICompletionsOptions are the stream options for the openai-completions
// API (OpenAICompletionsOptions in openai-completions.ts).
type OpenAICompletionsOptions struct {
	StreamOptions
	// ToolChoice is passed through as the OpenAI tool_choice parameter
	// ("auto", "none", "required", or a {"type":"function",...} object).
	ToolChoice any
	// ReasoningEffort is a ThinkingLevel; empty disables reasoning.
	ReasoningEffort ThinkingLevel
	// ThinkingBudgets are per-level token budgets, only used when
	// compat.SupportsThinkingTokenBudget is set.
	ThinkingBudgets *ThinkingBudgets
}

// ---------------------------------------------------------------------------
// stream / streamSimple
// ---------------------------------------------------------------------------

// Stream starts an openai-completions streaming request and returns the event
// stream. Failures are encoded in the stream as an error event, never thrown
// or panicked (pi's StreamFunction contract).
func Stream(model *Model, conv *Context, options *OpenAICompletionsOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &OpenAICompletionsOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &completionsStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamSimple maps SimpleStreamOptions onto the completions stream
// (streamSimple in openai-completions.ts). Unlike the TS original, which
// throws when no API key is available, the Go port surfaces that failure as a
// regular error event on the returned stream.
func StreamSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)
	var reasoningEffort ThinkingLevel
	if options.Reasoning != "" {
		clamped := ClampThinkingLevel(model, options.Reasoning)
		if clamped != ThinkingOff {
			reasoningEffort = clamped
		}
	}
	return Stream(model, conv, &OpenAICompletionsOptions{
		StreamOptions:   base,
		ReasoningEffort: reasoningEffort,
		ThinkingBudgets: options.ThinkingBudgets,
	})
}

type completionsStreamer struct {
	model   *Model
	conv    *Context
	options *OpenAICompletionsOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func zeroUsage() Usage {
	return Usage{Cost: UsageCost{}}
}

func (p *completionsStreamer) run() {
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
		// Catch block of stream() in openai-completions.ts.
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

// snapshot returns a copy of the in-flight message for an event. The TS
// original shares one live object across all events; Go events carry a copy
// so consumers never race the producer.
func (p *completionsStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *completionsStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *completionsStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	apiKey, err := getClientAPIKey(p.model.Provider, opts.APIKey, opts.Headers)
	if err != nil {
		return err
	}
	compat := getCompat(p.model)
	cacheRetention := resolveCacheRetention(opts.CacheRetention, opts.Env)
	cacheSessionID := opts.SessionID
	if cacheRetention == CacheRetentionNone {
		cacheSessionID = ""
	}
	headers := buildRequestHeaders(p.model, opts.Headers, cacheSessionID, compat)
	params := buildParams(p.model, p.conv, opts, compat, cacheRetention)
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doChatCompletionRequest(ctx, p.model, apiKey, headers, params, opts.TimeoutMs)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})
	if err := p.consumeStream(ctx, resp.Body, compat); err != nil {
		return err
	}
	return nil
}

// consumeStream reads the SSE body and assembles the partial message. Mirrors
// the `for await (const chunk of openaiStream)` loop in openai-completions.ts.
func (p *completionsStreamer) consumeStream(ctx context.Context, body io.Reader, compat resolvedCompat) error {
	output := p.output
	reader := NewSSEReader(body)

	var textBlock *TextContent
	var thinkingBlock *ThinkingContent
	hasFinishReason := false
	toolCallBlocksByIndex := map[int]*streamingToolCall{}
	toolCallBlocksByID := map[string]*streamingToolCall{}
	pendingReasoningDetailsByToolCallID := map[string]string{}
	// blocks holds the in-flight content blocks; output.Content is rebuilt
	// from it on every event push.
	var blocks []streamingBlock

	contentIndex := func(block streamingBlock) int {
		for i, b := range blocks {
			if b == block {
				return i
			}
		}
		return -1
	}
	syncContent := func() {
		output.Content = make([]ContentBlock, len(blocks))
		for i, b := range blocks {
			output.Content[i] = b.content()
		}
	}
	finishBlock := func(block streamingBlock) {
		idx := contentIndex(block)
		if idx == -1 {
			return
		}
		syncContent()
		switch b := block.(type) {
		case *TextContent:
			p.push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: idx, Content: b.Text})
		case *ThinkingContent:
			p.push(AssistantMessageEvent{Type: EventThinkingEnd, ContentIndex: idx, Content: b.Thinking})
		case *streamingToolCall:
			b.ToolCall.Arguments = ParseStreamingJSON(b.partialArgs)
			syncContent()
			tc := b.ToolCall
			p.push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: idx, ToolCall: &tc})
		}
	}
	ensureTextBlock := func() *TextContent {
		if textBlock == nil {
			textBlock = &TextContent{Type: "text"}
			blocks = append(blocks, textBlock)
			syncContent()
			p.push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: contentIndex(textBlock)})
		}
		return textBlock
	}
	ensureThinkingBlock := func(thinkingSignature string) *ThinkingContent {
		if thinkingBlock == nil {
			thinkingBlock = &ThinkingContent{Type: "thinking", ThinkingSignature: thinkingSignature}
			blocks = append(blocks, thinkingBlock)
			syncContent()
			p.push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: contentIndex(thinkingBlock)})
		}
		return thinkingBlock
	}
	applyPendingReasoningDetail := func(block *streamingToolCall) {
		if block.ToolCall.ID == "" {
			return
		}
		if detail, ok := pendingReasoningDetailsByToolCallID[block.ToolCall.ID]; ok {
			block.ToolCall.ThoughtSignature = detail
			delete(pendingReasoningDetailsByToolCallID, block.ToolCall.ID)
		}
	}
	ensureToolCallBlock := func(tc chunkToolCall) *streamingToolCall {
		name := ""
		if tc.Function != nil {
			name = tc.Function.Name
		} else if tc.Custom != nil {
			name = tc.Custom.Name
		}
		var block *streamingToolCall
		if tc.Index != nil {
			block = toolCallBlocksByIndex[*tc.Index]
		}
		if block == nil && tc.ID != "" {
			block = toolCallBlocksByID[tc.ID]
		}
		if block == nil {
			block = &streamingToolCall{
				ToolCall: ToolCall{Type: "toolCall", ID: tc.ID, Name: name, Arguments: map[string]any{}},
			}
			if tc.Index != nil {
				toolCallBlocksByIndex[*tc.Index] = block
			}
			if tc.ID != "" {
				toolCallBlocksByID[tc.ID] = block
			}
			blocks = append(blocks, block)
			syncContent()
			p.push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: contentIndex(block)})
		}
		if tc.ID != "" {
			toolCallBlocksByID[tc.ID] = block
		}
		if block.ToolCall.Name == "" && name != "" {
			block.ToolCall.Name = name
		}
		applyPendingReasoningDetail(block)
		return block
	}

	for {
		event, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			if ctx.Err() != nil {
				return AbortError{}
			}
			return fmt.Errorf("reading SSE stream: %w", err)
		}
		data := strings.TrimSpace(event.Data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// The OpenAI SDK skips non-JSON keepalive lines; mirror that leniency.
			continue
		}

		// Each chunk in a streamed completion carries the same id.
		if output.ResponseID == "" {
			output.ResponseID = chunk.ID
		}
		if chunk.Model != "" && chunk.Model != p.model.ID && output.ResponseModel == "" {
			output.ResponseModel = chunk.Model
		}
		if chunk.Usage != nil {
			output.Usage = parseChunkUsage(chunk.Usage, p.model)
		}

		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		// Fallback: some providers (e.g., Moonshot) return usage in
		// choice.usage instead of the standard chunk.usage.
		if chunk.Usage == nil && choice.Usage != nil {
			output.Usage = parseChunkUsage(choice.Usage, p.model)
		}

		if choice.FinishReason != nil && *choice.FinishReason != "" {
			output.RawStopReason = *choice.FinishReason
			stopReason, errorMessage := mapStopReason(*choice.FinishReason)
			output.StopReason = stopReason
			if errorMessage != "" {
				output.ErrorMessage = errorMessage
			}
			hasFinishReason = true
		}

		delta := choice.Delta
		if delta.Content != nil && *delta.Content != "" {
			block := ensureTextBlock()
			block.Text += *delta.Content
			syncContent()
			p.push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: contentIndex(block), Delta: *delta.Content})
		}

		// Some endpoints return reasoning in reasoning_content (llama.cpp),
		// or reasoning (other openai compatible endpoints). Use the first
		// non-empty reasoning field to avoid duplication (e.g., chutes.ai
		// returns both reasoning_content and reasoning with same content).
		foundReasoningField, reasoningDelta := "", ""
		for _, field := range []struct {
			name  string
			value *string
		}{
			{"reasoning_content", delta.ReasoningContent},
			{"reasoning", delta.Reasoning},
			{"reasoning_text", delta.ReasoningText},
		} {
			if field.value != nil && *field.value != "" {
				foundReasoningField, reasoningDelta = field.name, *field.value
				break
			}
		}
		if foundReasoningField != "" {
			thinkingSignature := foundReasoningField
			if p.model.Provider == "opencode-go" && foundReasoningField == "reasoning" {
				thinkingSignature = "reasoning_content"
			}
			block := ensureThinkingBlock(thinkingSignature)
			block.Thinking += reasoningDelta
			syncContent()
			p.push(AssistantMessageEvent{Type: EventThinkingDelta, ContentIndex: contentIndex(block), Delta: reasoningDelta})
		}

		for _, tc := range delta.ToolCalls {
			block := ensureToolCallBlock(tc)
			if block.ToolCall.ID == "" && tc.ID != "" {
				block.ToolCall.ID = tc.ID
				toolCallBlocksByID[tc.ID] = block
			}
			name := ""
			if tc.Function != nil {
				name = tc.Function.Name
			} else if tc.Custom != nil {
				name = tc.Custom.Name
			}
			if block.ToolCall.Name == "" && name != "" {
				block.ToolCall.Name = name
			}

			deltaStr := ""
			if tc.Function != nil && tc.Function.Arguments != "" {
				deltaStr = tc.Function.Arguments
				block.partialArgs += tc.Function.Arguments
				if shouldReparseStreamingJSON(block.parsedArgsLen, len(block.partialArgs)) {
					block.ToolCall.Arguments = ParseStreamingJSON(block.partialArgs)
					block.parsedArgsLen = len(block.partialArgs)
				}
			} else if tc.Custom != nil && tc.Custom.Input != "" {
				// Custom (grammar) tool input: constrained sampling is out of
				// scope for this port slice, so the raw input accumulates into
				// partialArgs like function arguments (the TS original wraps it
				// via appendGrammarToolInputJsonDelta).
				deltaStr = tc.Custom.Input
				block.partialArgs += tc.Custom.Input
				if shouldReparseStreamingJSON(block.parsedArgsLen, len(block.partialArgs)) {
					block.ToolCall.Arguments = ParseStreamingJSON(block.partialArgs)
					block.parsedArgsLen = len(block.partialArgs)
				}
			}
			syncContent()
			p.push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: contentIndex(block), Delta: deltaStr})
		}

		for _, rawDetail := range delta.ReasoningDetails {
			var detail struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Data string `json:"data"`
			}
			if err := json.Unmarshal(rawDetail, &detail); err != nil {
				continue
			}
			if detail.Type != "reasoning.encrypted" || detail.ID == "" || detail.Data == "" {
				continue
			}
			serialized := compactJSON(rawDetail)
			if block, ok := toolCallBlocksByID[detail.ID]; ok {
				block.ToolCall.ThoughtSignature = serialized
			} else {
				pendingReasoningDetailsByToolCallID[detail.ID] = serialized
			}
		}
	}

	for _, block := range blocks {
		finishBlock(block)
	}
	if ctx.Err() != nil {
		return errors.New("request was aborted")
	}
	if output.StopReason == StopAborted {
		return errors.New("request was aborted")
	}
	if !hasFinishReason && !compat.SupportsFinishReason {
		output.StopReason = StopStop
		for _, block := range blocks {
			if _, isToolCall := block.(*streamingToolCall); isToolCall {
				output.StopReason = StopToolUse
				break
			}
		}
	}
	if output.StopReason == StopError {
		if output.ErrorMessage == "" {
			return errors.New("provider returned an error stop reason")
		}
		return errors.New(output.ErrorMessage)
	}
	if (compat.SupportsFinishReason && !hasFinishReason) || output.StopReason == StopPending {
		return errors.New("Stream ended without finish_reason")
	}

	syncContent()
	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

// streamingBlock is one in-flight content block during stream assembly.
type streamingBlock interface {
	content() ContentBlock
}

type streamingToolCall struct {
	ToolCall
	partialArgs string
	// parsedArgsLen is how much of partialArgs the Arguments map reflects.
	// See shouldReparseStreamingJSON.
	parsedArgsLen int
}

func (t *TextContent) content() ContentBlock       { return *t }
func (t *ThinkingContent) content() ContentBlock   { return *t }
func (t *streamingToolCall) content() ContentBlock { return t.ToolCall }

// Wire-format chunk types (OpenAI chat.completion.chunk plus common
// vendor extensions: reasoning_content / reasoning / reasoning_text /
// reasoning_details, prompt_cache_hit_tokens, cache_write_tokens).

type chatCompletionChunk struct {
	ID      string        `json:"id"`
	Model   string        `json:"model"`
	Usage   *chunkUsage   `json:"usage"`
	Choices []chunkChoice `json:"choices"`
}

type chunkChoice struct {
	Delta        chunkDelta  `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
	Usage        *chunkUsage `json:"usage"`
}

type chunkDelta struct {
	Content          *string           `json:"content"`
	ReasoningContent *string           `json:"reasoning_content"`
	Reasoning        *string           `json:"reasoning"`
	ReasoningText    *string           `json:"reasoning_text"`
	ToolCalls        []chunkToolCall   `json:"tool_calls"`
	ReasoningDetails []json.RawMessage `json:"reasoning_details"`
}

type chunkToolCall struct {
	Index    *int   `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function *struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	Custom *struct {
		Name  string `json:"name"`
		Input string `json:"input"`
	} `json:"custom"`
}

type chunkUsage struct {
	PromptTokens         int `json:"prompt_tokens"`
	CompletionTokens     int `json:"completion_tokens"`
	PromptCacheHitTokens int `json:"prompt_cache_hit_tokens"`
	PromptTokensDetails  *struct {
		// Pointer to mirror TS `?.cached_tokens ??`: an explicit 0 wins over
		// prompt_cache_hit_tokens, an absent field falls back to it.
		CachedTokens     *int `json:"cached_tokens"`
		CacheWriteTokens int  `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func compactJSON(raw json.RawMessage) string {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return string(raw)
	}
	return buf.String()
}

// parseChunkUsage maps OpenAI usage fields onto pi's Usage. Port of
// parseChunkUsage in openai-completions.ts: cached_tokens is cache-read
// (hits); cache_write_tokens is a separate write count (OpenRouter contract).
func parseChunkUsage(raw *chunkUsage, model *Model) Usage {
	promptTokens := raw.PromptTokens
	cacheReadTokens := raw.PromptCacheHitTokens
	cacheWriteTokens := 0
	if raw.PromptTokensDetails != nil {
		if raw.PromptTokensDetails.CachedTokens != nil {
			cacheReadTokens = *raw.PromptTokensDetails.CachedTokens
		}
		cacheWriteTokens = raw.PromptTokensDetails.CacheWriteTokens
	}
	input := max(0, promptTokens-cacheReadTokens-cacheWriteTokens)
	// OpenAI completion_tokens already includes reasoning_tokens.
	outputTokens := raw.CompletionTokens
	reasoning := 0
	if raw.CompletionTokensDetails != nil {
		reasoning = raw.CompletionTokensDetails.ReasoningTokens
	}
	usage := Usage{
		Input:       input,
		Output:      outputTokens,
		CacheRead:   cacheReadTokens,
		CacheWrite:  cacheWriteTokens,
		Reasoning:   &reasoning,
		TotalTokens: input + outputTokens + cacheReadTokens + cacheWriteTokens,
	}
	CalculateCost(model, &usage)
	return usage
}

// mapStopReason maps an OpenAI finish_reason to pi's StopReason. Port of
// mapStopReason in openai-completions.ts.
func mapStopReason(reason string) (StopReason, string) {
	switch reason {
	case "stop", "end":
		return StopStop, ""
	case "length":
		return StopLength, ""
	case "function_call", "tool_calls":
		return StopToolUse, ""
	case "content_filter":
		return StopError, "Provider finish_reason: content_filter"
	case "network_error":
		return StopError, "Provider finish_reason: network_error"
	default:
		return StopError, "Provider finish_reason: " + reason
	}
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

// getClientAPIKey resolves the API key, or "unused" when the caller supplies
// an authorization header directly (getClientApiKey in openai-completions.ts).
func getClientAPIKey(provider string, apiKey string, headers ProviderHeaders) (string, error) {
	if apiKey != "" {
		return apiKey, nil
	}
	if hasHeader(headers, "authorization") || hasHeader(headers, "cf-aig-authorization") {
		return "unused", nil
	}
	return "", fmt.Errorf("no API key for provider: %s", provider)
}

func hasHeader(headers ProviderHeaders, name string) bool {
	for key, value := range headers {
		if strings.EqualFold(key, name) && value != nil && strings.TrimSpace(*value) != "" {
			return true
		}
	}
	return false
}

// buildRequestHeaders merges model headers, session-affinity headers, and
// option headers (createClient in openai-completions.ts). A nil option-header
// value suppresses a default header with the same name.
//
// The github-copilot dynamic headers of the TS original are provider-specific
// and deferred to a later slice.
func buildRequestHeaders(model *Model, optionsHeaders ProviderHeaders, sessionID string, compat resolvedCompat) map[string]string {
	headers := map[string]string{}
	for k, v := range model.Headers {
		headers[k] = v
	}

	if sessionID != "" && compat.SendSessionAffinityHeaders {
		switch compat.SessionAffinityFormat {
		case "openrouter":
			headers["x-session-id"] = sessionID
		default:
			if compat.SessionAffinityFormat == "openai" {
				headers["session_id"] = sessionID
			}
			headers["x-client-request-id"] = sessionID
			headers["x-session-affinity"] = sessionID
		}
	}

	// Merge options headers last so they can override defaults.
	for k, v := range optionsHeaders {
		if v == nil {
			deleteHeaderCaseInsensitive(headers, k)
			continue
		}
		headers[k] = *v
	}
	return headers
}

func deleteHeaderCaseInsensitive(headers map[string]string, name string) {
	for k := range headers {
		if strings.EqualFold(k, name) {
			delete(headers, k)
		}
	}
}

func headersToRecord(h http.Header) map[string]string {
	out := map[string]string{}
	for k := range h {
		out[strings.ToLower(k)] = h.Get(k)
	}
	return out
}

// doChatCompletionRequest performs one POST {baseUrl}/chat/completions and
// returns the streaming response. Non-2xx responses become *ProviderError so
// RetryProviderRequest can classify them.
func doChatCompletionRequest(ctx context.Context, model *Model, apiKey string, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := strings.TrimSuffix(model.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Authorization", "Bearer "+apiKey)
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

// resolveCacheRetention mirrors resolveCacheRetention in
// openai-completions.ts: explicit option > PI_CACHE_RETENTION env > "short".
func resolveCacheRetention(cacheRetention CacheRetention, env map[string]string) CacheRetention {
	if cacheRetention != "" {
		return cacheRetention
	}
	if getProviderEnvValue("PI_CACHE_RETENTION", env) == "long" {
		return CacheRetentionLong
	}
	return CacheRetentionShort
}

// getProviderEnvValue resolves a provider env value from scoped overrides,
// then os.Getenv (utils/provider-env.ts).
func getProviderEnvValue(name string, env map[string]string) string {
	if env != nil {
		if v := env[name]; v != "" {
			return v
		}
	}
	return os.Getenv(name)
}

// sanitizeSurrogates is the TS sanitizeSurrogates; Go strings are UTF-8 and
// encoding/json already replaces invalid bytes, so unpaired surrogates cannot
// reach the wire. Kept as an identity to mark the call sites.
func sanitizeSurrogates(text string) string {
	if utf8.ValidString(text) {
		return text
	}
	return strings.ToValidUTF8(text, "")
}

// ---------------------------------------------------------------------------
// buildParams
// ---------------------------------------------------------------------------

const openAIPromptCacheKeyMaxLength = 64

// clampOpenAIPromptCacheKey caps the prompt cache key at 64 characters
// (openai-prompt-cache.ts).
func clampOpenAIPromptCacheKey(key string) string {
	runes := []rune(key)
	if len(runes) <= openAIPromptCacheKeyMaxLength {
		return key
	}
	return string(runes[:openAIPromptCacheKeyMaxLength])
}

// buildParams builds the chat.completions request body. Port of buildParams
// in openai-completions.ts.
func buildParams(model *Model, conv *Context, options *OpenAICompletionsOptions, compat resolvedCompat, cacheRetention CacheRetention) map[string]any {
	messages := convertMessages(model, conv, compat)
	cacheControl := getCompatCacheControl(compat, cacheRetention)

	params := map[string]any{
		"model":    model.ID,
		"messages": messages,
		"stream":   true,
	}
	if (strings.Contains(model.BaseURL, "api.openai.com") && cacheRetention != CacheRetentionNone) ||
		(cacheRetention == CacheRetentionLong && compat.SupportsLongCacheRetention) {
		if key := clampOpenAIPromptCacheKey(options.SessionID); key != "" {
			params["prompt_cache_key"] = key
		}
	}
	if cacheRetention == CacheRetentionLong && compat.SupportsLongCacheRetention {
		params["prompt_cache_retention"] = "24h"
	}

	if compat.SupportsUsageInStreaming {
		params["stream_options"] = map[string]any{"include_usage": true}
	}

	if compat.SupportsStore {
		params["store"] = false
	}

	if options.MaxTokens > 0 {
		if compat.MaxTokensField == "max_tokens" {
			params["max_tokens"] = options.MaxTokens
		} else {
			params["max_completion_tokens"] = options.MaxTokens
		}
	}

	if options.Temperature != nil {
		params["temperature"] = *options.Temperature
	}

	var deferredToolNames map[string]bool
	if compat.DeferredToolsMode == "kimi" {
		deferredToolNames = getDeferredToolNames(conv.Messages)
	} else {
		deferredToolNames = map[string]bool{}
	}
	var activeTools []Tool
	for _, tool := range conv.Tools {
		if !deferredToolNames[tool.Name] {
			activeTools = append(activeTools, tool)
		}
	}
	var tools []any
	if len(activeTools) > 0 {
		tools = convertTools(activeTools, compat)
		params["tools"] = tools
		if compat.ZaiToolStream {
			params["tool_stream"] = true
		}
	} else if hasToolHistory(conv.Messages) {
		// Anthropic (via LiteLLM/proxy) requires tools param when conversation
		// has tool_calls/tool_results.
		tools = []any{}
		params["tools"] = tools
	}

	if cacheControl != nil {
		applyAnthropicCacheControl(messages, tools, cacheControl)
	}

	if options.ToolChoice != nil {
		params["tool_choice"] = options.ToolChoice
	}

	applyThinkingParams(params, model, options, compat)

	// vLLM caps reasoning with a top-level thinking_token_budget. Independent
	// of thinkingFormat: the same server can serve zai, qwen or chat-template
	// models. Reasoning and the answer share max_tokens here, so an uncapped
	// reasoning phase can consume the whole response and leave no answer.
	if compat.SupportsThinkingTokenBudget && options.ReasoningEffort != "" && model.Reasoning {
		level := clampReasoning(options.ReasoningEffort)
		budgets := ThinkingBudgets{Minimal: 1024, Low: 2048, Medium: 8192, High: 16384}
		if options.ThinkingBudgets != nil {
			mergeThinkingBudgets(&budgets, options.ThinkingBudgets)
		}
		ceiling := model.MaxTokens
		if v, ok := params["max_tokens"].(int); ok {
			ceiling = v
		} else if v, ok := params["max_completion_tokens"].(int); ok {
			ceiling = v
		}
		// Always leave room for the answer.
		budget := min(thinkingBudgetForLevel(budgets, level), max(0, ceiling-MinAnswerTokens))
		if budget > 0 {
			params["thinking_token_budget"] = budget
		}
	}

	// OpenRouter provider routing preferences.
	if model.Compat != nil && model.Compat.OpenRouterRouting != nil {
		params["provider"] = model.Compat.OpenRouterRouting
	}

	// Vercel AI Gateway provider routing preferences.
	if model.Compat != nil && model.Compat.VercelGatewayRouting != nil {
		routing := model.Compat.VercelGatewayRouting
		if len(routing.Only) > 0 || len(routing.Order) > 0 {
			gatewayOptions := map[string]any{}
			if len(routing.Only) > 0 {
				gatewayOptions["only"] = routing.Only
			}
			if len(routing.Order) > 0 {
				gatewayOptions["order"] = routing.Order
			}
			params["providerOptions"] = map[string]any{"gateway": gatewayOptions}
		}
	}

	// Last so custom keys override the named request fields.
	for k, v := range options.SamplingParams {
		params[k] = v
	}

	return params
}

func mergeThinkingBudgets(dst *ThinkingBudgets, src *ThinkingBudgets) {
	if src.Minimal != 0 {
		dst.Minimal = src.Minimal
	}
	if src.Low != 0 {
		dst.Low = src.Low
	}
	if src.Medium != 0 {
		dst.Medium = src.Medium
	}
	if src.High != 0 {
		dst.High = src.High
	}
}

func thinkingBudgetForLevel(budgets ThinkingBudgets, level ThinkingLevel) int {
	switch level {
	case ThinkingMinimal:
		return budgets.Minimal
	case ThinkingLow:
		return budgets.Low
	case ThinkingMedium:
		return budgets.Medium
	default:
		return budgets.High
	}
}

// mapOrEffort mirrors TS `model.thinkingLevelMap?.[effort] ?? effort`: a
// mapped string wins; missing keys AND null entries fall back to effort.
func mapOrEffort(model *Model, effort string) string {
	if v, ok := model.ThinkingLevelMap[effort]; ok && v != nil {
		return *v
	}
	return effort
}

// mapIfDefined mirrors TS `mapped === undefined ? fallback : mapped`: only a
// missing key falls back; a null entry yields "" (suppresses the param).
func mapIfDefined(model *Model, key ModelThinkingLevel, fallback string) string {
	if v, ok := model.ThinkingLevelMap[key]; ok {
		if v == nil {
			return ""
		}
		return *v
	}
	return fallback
}

// offIsNotNull mirrors TS `model.thinkingLevelMap?.off !== null` (true when
// the key is missing or holds a string; false only for an explicit null).
func offIsNotNull(model *Model) bool {
	v, ok := model.ThinkingLevelMap[ThinkingOff]
	return !ok || v != nil
}

// applyThinkingParams ports the thinking/reasoning_format branches of
// buildParams in openai-completions.ts.
func applyThinkingParams(params map[string]any, model *Model, options *OpenAICompletionsOptions, compat resolvedCompat) {
	effort := options.ReasoningEffort

	switch {
	case compat.ThinkingFormat == "zai" && model.Reasoning:
		if effort != "" {
			params["thinking"] = map[string]any{"type": "enabled", "clear_thinking": false}
		} else {
			params["thinking"] = map[string]any{"type": "disabled"}
		}
		if effort != "" && compat.SupportsReasoningEffort {
			if e := mapIfDefined(model, effort, effort); e != "" {
				params["reasoning_effort"] = e
			}
		}
	case compat.ThinkingFormat == "qwen" && model.Reasoning:
		params["enable_thinking"] = effort != ""
		if effort != "" && compat.SupportsReasoningEffort {
			if e := mapOrEffort(model, effort); e != "" {
				params["reasoning_effort"] = e
			}
		}
	case compat.ThinkingFormat == "qwen-chat-template" && model.Reasoning:
		params["chat_template_kwargs"] = map[string]any{
			"enable_thinking":   effort != "",
			"preserve_thinking": true,
		}
	case compat.ThinkingFormat == "chat-template" && model.Reasoning:
		if kwargs := buildChatTemplateValues(model, options, compat.ChatTemplateKwargs); len(kwargs) > 0 {
			params["chat_template_kwargs"] = kwargs
		}
	case compat.ThinkingFormat == "baseten" && model.Reasoning:
		if args := buildChatTemplateValues(model, options, compat.ChatTemplateArgs); len(args) > 0 {
			params["chat_template_args"] = args
		}
		if compat.SupportsReasoningEffort {
			requestedEffort := effort
			mapped := requestedEffort
			if requestedEffort == "" {
				mapped = mapIfDefined(model, ThinkingOff, "")
			} else {
				mapped = mapIfDefined(model, requestedEffort, requestedEffort)
			}
			if mapped != "" {
				params["reasoning_effort"] = mapped
			}
		}
	case compat.ThinkingFormat == "deepseek" && model.Reasoning:
		if effort != "" {
			params["thinking"] = map[string]any{"type": "enabled"}
		} else if offIsNotNull(model) {
			params["thinking"] = map[string]any{"type": "disabled"}
		}
		if effort != "" && compat.SupportsReasoningEffort {
			params["reasoning_effort"] = mapOrEffort(model, effort)
		}
	case compat.ThinkingFormat == "openrouter" && model.Reasoning:
		// OpenRouter normalizes reasoning across providers via a nested
		// reasoning object.
		if effort != "" {
			params["reasoning"] = map[string]any{"effort": mapOrEffort(model, effort)}
		} else if offIsNotNull(model) {
			params["reasoning"] = map[string]any{"effort": mapOrEffort(model, ThinkingOff)}
		}
	case compat.ThinkingFormat == "ant-ling" && model.Reasoning && effort != "":
		if v, ok := model.ThinkingLevelMap[effort]; ok && v != nil {
			params["reasoning"] = map[string]any{"effort": *v}
		}
	case compat.ThinkingFormat == "together" && model.Reasoning:
		params["reasoning"] = map[string]any{"enabled": effort != ""}
		if effort != "" && compat.SupportsReasoningEffort {
			params["reasoning_effort"] = mapOrEffort(model, effort)
		}
	case compat.ThinkingFormat == "string-thinking" && model.Reasoning:
		if effort != "" {
			params["thinking"] = mapOrEffort(model, effort)
		} else if offIsNotNull(model) {
			params["thinking"] = mapOrEffort(model, ThinkingOff)
		}
	case effort != "" && model.Reasoning && compat.SupportsReasoningEffort:
		// OpenAI-style reasoning_effort.
		params["reasoning_effort"] = mapOrEffort(model, effort)
	case effort == "" && model.Reasoning && compat.SupportsReasoningEffort:
		if v, ok := model.ThinkingLevelMap[ThinkingOff]; ok && v != nil {
			params["reasoning_effort"] = *v
		}
	}
}

// buildChatTemplateValues resolves chat_template_kwargs/args values,
// expanding {$var: "thinking.enabled"|"thinking.effort"} entries
// (buildChatTemplateValues / resolveChatTemplateKwargValue in
// openai-completions.ts).
func buildChatTemplateValues(model *Model, options *OpenAICompletionsOptions, values map[string]any) map[string]any {
	resolved := map[string]any{}
	for key, value := range values {
		if v, ok := resolveChatTemplateKwargValue(model, options, value); ok {
			resolved[key] = v
		}
	}
	if len(resolved) == 0 {
		return nil
	}
	return resolved
}

func resolveChatTemplateKwargValue(model *Model, options *OpenAICompletionsOptions, value any) (any, bool) {
	kwarg, ok := value.(map[string]any)
	if !ok || kwarg == nil {
		return value, true // plain string/number/bool/null passes through
	}
	varName, isKwarg := kwarg["$var"].(string)
	if !isKwarg {
		return value, true
	}
	effort := options.ReasoningEffort
	omitWhenOff, _ := kwarg["omitWhenOff"].(bool)
	if effort == "" && omitWhenOff {
		return nil, false
	}
	if varName == "thinking.enabled" {
		return effort != "", true
	}
	// "thinking.effort"
	if effort != "" {
		return mapIfDefined(model, effort, effort), true
	}
	mapped := mapIfDefined(model, ThinkingOff, "")
	if mapped == "" {
		return nil, false
	}
	return mapped, true
}

// hasToolHistory reports whether conversation messages contain tool calls or
// tool results (openai-completions.ts).
func hasToolHistory(messages []Message) bool {
	for _, msg := range messages {
		switch m := msg.(type) {
		case ToolResultMessage:
			return true
		case AssistantMessage:
			for _, block := range m.Content {
				if _, ok := block.(ToolCall); ok {
					return true
				}
			}
		}
	}
	return false
}

func getDeferredToolNames(messages []Message) map[string]bool {
	names := map[string]bool{}
	for _, msg := range messages {
		if tr, ok := msg.(ToolResultMessage); ok {
			for _, name := range tr.AddedToolNames {
				names[name] = true
			}
		}
	}
	return names
}

// convertTools serializes tools for the request (convertTools in
// openai-completions.ts). Constrained sampling (grammar custom tools, strict
// resolution) is deferred: tools always serialize as function tools, with
// strict: false when the provider supports strict mode.
func convertTools(tools []Tool, compat resolvedCompat) []any {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		fn := map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
		}
		if len(tool.Parameters) > 0 {
			fn["parameters"] = json.RawMessage(tool.Parameters)
		} else {
			fn["parameters"] = map[string]any{}
		}
		if compat.SupportsStrictMode {
			fn["strict"] = false
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

// ---------------------------------------------------------------------------
// Prompt-cache (Anthropic-style cache_control) helpers
// ---------------------------------------------------------------------------

type openAICompatCacheControl map[string]any

func getCompatCacheControl(compat resolvedCompat, cacheRetention CacheRetention) openAICompatCacheControl {
	if compat.CacheControlFormat != "anthropic" || cacheRetention == CacheRetentionNone {
		return nil
	}
	cc := openAICompatCacheControl{"type": "ephemeral"}
	if cacheRetention == CacheRetentionLong && compat.SupportsLongCacheRetention {
		cc["ttl"] = "1h"
	}
	return cc
}

// applyAnthropicCacheControl marks the system prompt, the last tool, and the
// last conversation text part with cache_control (openai-completions.ts).
func applyAnthropicCacheControl(messages []any, tools []any, cacheControl openAICompatCacheControl) {
	// System prompt.
	for _, m := range messages {
		if msg, ok := m.(map[string]any); ok {
			if role, _ := msg["role"].(string); role == "system" || role == "developer" {
				addCacheControlToTextContent(msg, cacheControl)
				break
			}
		}
	}
	// Last tool.
	if len(tools) > 0 {
		if tool, ok := tools[len(tools)-1].(map[string]any); ok {
			tool["cache_control"] = cacheControl
		}
	}
	// Last conversation message.
	for i := len(messages) - 1; i >= 0; i-- {
		msg, ok := messages[i].(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "user" || role == "assistant" || role == "tool" {
			if addCacheControlToTextContent(msg, cacheControl) {
				return
			}
		}
	}
}

func addCacheControlToTextContent(message map[string]any, cacheControl openAICompatCacheControl) bool {
	content, ok := message["content"]
	if !ok {
		return false
	}
	if s, ok := content.(string); ok {
		if len(s) == 0 {
			return false
		}
		message["content"] = []any{map[string]any{
			"type":          "text",
			"text":          s,
			"cache_control": cacheControl,
		}}
		return true
	}
	parts, ok := content.([]any)
	if !ok {
		return false
	}
	for i := len(parts) - 1; i >= 0; i-- {
		part, ok := parts[i].(map[string]any)
		if !ok {
			continue
		}
		if partType, _ := part["type"].(string); partType == "text" {
			part["cache_control"] = cacheControl
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Compat detection
// ---------------------------------------------------------------------------

// detectCompat auto-detects compatibility settings from provider name and
// baseUrl (detectCompat in openai-completions.ts).
func detectCompat(model *Model) resolvedCompat {
	provider := model.Provider
	baseURL := model.BaseURL

	isZai := provider == "zai" || provider == "zai-coding-cn" ||
		strings.Contains(baseURL, "api.z.ai") || strings.Contains(baseURL, "open.bigmodel.cn")
	isTogether := provider == "together" || strings.Contains(baseURL, "api.together.ai") ||
		strings.Contains(baseURL, "api.together.xyz")
	isMoonshot := provider == "moonshotai" || provider == "moonshotai-cn" ||
		strings.Contains(baseURL, "api.moonshot.")
	isOpenRouter := provider == "openrouter" || strings.Contains(baseURL, "openrouter.ai")
	isCloudflareWorkersAI := provider == "cloudflare-workers-ai" || strings.Contains(baseURL, "api.cloudflare.com")
	isCloudflareAIGateway := provider == "cloudflare-ai-gateway" || strings.Contains(baseURL, "gateway.ai.cloudflare.com")
	isNvidia := provider == "nvidia" || strings.Contains(baseURL, "integrate.api.nvidia.com")
	isAntLing := provider == "ant-ling" || strings.Contains(baseURL, "api.ant-ling.com")

	isNonStandard := isNvidia ||
		provider == "cerebras" || strings.Contains(baseURL, "cerebras.ai") ||
		provider == "xai" || strings.Contains(baseURL, "api.x.ai") ||
		isTogether ||
		strings.Contains(baseURL, "chutes.ai") ||
		strings.Contains(baseURL, "deepseek.com") ||
		isZai || isMoonshot ||
		provider == "opencode" || strings.Contains(baseURL, "opencode.ai") ||
		isCloudflareWorkersAI || isCloudflareAIGateway ||
		isAntLing

	isDeepSeek := provider == "deepseek" || strings.Contains(baseURL, "deepseek.com")
	useMaxTokens := strings.Contains(baseURL, "chutes.ai") ||
		isDeepSeek || isMoonshot || isCloudflareAIGateway || isTogether || isNvidia || isAntLing || isZai

	isGrok := provider == "xai" || strings.Contains(baseURL, "api.x.ai")
	isOpenRouterDeveloperRoleModel := isOpenRouter &&
		(strings.HasPrefix(model.ID, "anthropic/") || strings.HasPrefix(model.ID, "openai/"))
	cacheControlFormat := ""
	if provider == "openrouter" && strings.HasPrefix(model.ID, "anthropic/") {
		cacheControlFormat = "anthropic"
	}

	thinkingFormat := "openai"
	switch {
	case isDeepSeek:
		thinkingFormat = "deepseek"
	case isZai:
		thinkingFormat = "zai"
	case isTogether:
		thinkingFormat = "together"
	case isAntLing:
		thinkingFormat = "ant-ling"
	case isOpenRouter:
		thinkingFormat = "openrouter"
	}

	maxTokensField := "max_completion_tokens"
	if useMaxTokens {
		maxTokensField = "max_tokens"
	}

	sessionAffinityFormat := "openai"
	if isOpenRouter {
		sessionAffinityFormat = "openrouter"
	}

	return resolvedCompat{
		SupportsStore:         !isNonStandard,
		SupportsDeveloperRole: isOpenRouterDeveloperRoleModel || (!isNonStandard && !isOpenRouter),
		SupportsReasoningEffort: !isGrok && !isZai && !isMoonshot && !isTogether &&
			!isCloudflareAIGateway && !isNvidia && !isAntLing,
		SupportsUsageInStreaming:                    true,
		SupportsFinishReason:                        true,
		MaxTokensField:                              maxTokensField,
		RequiresToolResultName:                      false,
		RequiresAssistantAfterToolResult:            false,
		RequiresThinkingAsText:                      false,
		RequiresReasoningContentOnAssistantMessages: isDeepSeek,
		ThinkingFormat:                              thinkingFormat,
		ChatTemplateKwargs:                          map[string]any{},
		ChatTemplateArgs:                            map[string]any{},
		OpenRouterRouting:                           &OpenRouterRouting{},
		VercelGatewayRouting:                        &VercelGatewayRouting{},
		ZaiToolStream:                               false,
		SupportsThinkingTokenBudget:                 false,
		SupportsStrictMode:                          !isMoonshot && !isTogether && !isCloudflareAIGateway && !isNvidia,
		SupportsOpenAIGrammarTools:                  false,
		CacheControlFormat:                          cacheControlFormat,
		SendSessionAffinityHeaders:                  false,
		DeferredToolsMode:                           "",
		SessionAffinityFormat:                       sessionAffinityFormat,
		SupportsLongCacheRetention: !(isTogether || isCloudflareWorkersAI || isCloudflareAIGateway ||
			isNvidia || isAntLing),
	}
}

// getCompat resolves compatibility settings: auto-detect, then override with
// explicit model.Compat entries (getCompat in openai-completions.ts).
func getCompat(model *Model) resolvedCompat {
	detected := detectCompat(model)
	c := model.Compat
	if c == nil {
		return detected
	}
	pick := func(override *bool, detected bool) bool {
		if override != nil {
			return *override
		}
		return detected
	}
	out := resolvedCompat{
		SupportsStore:                               pick(c.SupportsStore, detected.SupportsStore),
		SupportsDeveloperRole:                       pick(c.SupportsDeveloperRole, detected.SupportsDeveloperRole),
		SupportsReasoningEffort:                     pick(c.SupportsReasoningEffort, detected.SupportsReasoningEffort),
		SupportsUsageInStreaming:                    pick(c.SupportsUsageInStreaming, detected.SupportsUsageInStreaming),
		SupportsFinishReason:                        pick(c.SupportsFinishReason, detected.SupportsFinishReason),
		MaxTokensField:                              detected.MaxTokensField,
		RequiresToolResultName:                      pick(c.RequiresToolResultName, detected.RequiresToolResultName),
		RequiresAssistantAfterToolResult:            pick(c.RequiresAssistantAfterToolResult, detected.RequiresAssistantAfterToolResult),
		RequiresThinkingAsText:                      pick(c.RequiresThinkingAsText, detected.RequiresThinkingAsText),
		RequiresReasoningContentOnAssistantMessages: pick(c.RequiresReasoningContentOnAssistantMessages, detected.RequiresReasoningContentOnAssistantMessages),
		ThinkingFormat:                              detected.ThinkingFormat,
		ChatTemplateKwargs:                          map[string]any{},
		ChatTemplateArgs:                            map[string]any{},
		OpenRouterRouting:                           &OpenRouterRouting{},
		VercelGatewayRouting:                        detected.VercelGatewayRouting,
		ZaiToolStream:                               pick(c.ZaiToolStream, detected.ZaiToolStream),
		SupportsThinkingTokenBudget:                 pick(c.SupportsThinkingTokenBudget, detected.SupportsThinkingTokenBudget),
		SupportsStrictMode:                          pick(c.SupportsStrictMode, detected.SupportsStrictMode),
		SupportsOpenAIGrammarTools:                  pick(c.SupportsOpenAIGrammarTools, detected.SupportsOpenAIGrammarTools),
		CacheControlFormat:                          detected.CacheControlFormat,
		SendSessionAffinityHeaders:                  pick(c.SendSessionAffinityHeaders, detected.SendSessionAffinityHeaders),
		DeferredToolsMode:                           c.DeferredToolsMode,
		SessionAffinityFormat:                       detected.SessionAffinityFormat,
		SupportsLongCacheRetention:                  pick(c.SupportsLongCacheRetention, detected.SupportsLongCacheRetention),
	}
	if c.MaxTokensField != "" {
		out.MaxTokensField = c.MaxTokensField
	}
	if c.ThinkingFormat != "" {
		out.ThinkingFormat = c.ThinkingFormat
	}
	if c.ChatTemplateKwargs != nil {
		out.ChatTemplateKwargs = c.ChatTemplateKwargs
	}
	if c.ChatTemplateArgs != nil {
		out.ChatTemplateArgs = c.ChatTemplateArgs
	}
	if c.OpenRouterRouting != nil {
		out.OpenRouterRouting = c.OpenRouterRouting
	}
	if c.VercelGatewayRouting != nil {
		out.VercelGatewayRouting = c.VercelGatewayRouting
	}
	if c.CacheControlFormat != "" {
		out.CacheControlFormat = c.CacheControlFormat
	}
	if c.SessionAffinityFormat != "" {
		out.SessionAffinityFormat = c.SessionAffinityFormat
	}
	return out
}
