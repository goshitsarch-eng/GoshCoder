package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Go port of reference/pi/packages/ai/src/api/anthropic-messages.ts.
//
// Where the TS implementation drives the Anthropic SDK, this port uses a
// hand-rolled net/http POST to {baseUrl}/v1/messages plus the SSE reader in
// sse.go. Request-body construction, streaming assembly, and error semantics
// mirror the TS where behavior is observable.
//
// DEVIATION (compat plumbing): pi stores compat on model.compat, a
// discriminated union keyed by API. Go's Model.Compat is
// *OpenAICompletionsCompat only and types.go is frozen, so the Anthropic
// compat struct travels on AnthropicOptions.Compat instead. Everywhere the TS
// reads model.compat (getAnthropicCompat, forceAdaptiveThinking checks), this
// port reads the options-carried compat. A consequence: StreamSimple callers
// cannot inject compat overrides, since SimpleStreamOptions is shared.
//
// Deferred from the TS original (out of scope for this slice): the
// options.client injection point (alternative SDK clients such as Vertex),
// the github-copilot dynamic headers, and the websocket transport.

const APIAnthropicMessages = "anthropic-messages"

func init() {
	RegisterStreamer(APIAnthropicMessages, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*AnthropicOptions)
			return StreamAnthropicMessages(model, ctx, o)
		},
		StreamSimple: StreamAnthropicMessagesSimple,
	})
}

// AnthropicMessagesCompat holds compatibility settings for Anthropic
// Messages-compatible APIs. Pointer fields are tri-state: nil means "use the
// default" (pi's undefined). Port of AnthropicMessagesCompat in types.ts; see
// that file for the per-field semantics.
type AnthropicMessagesCompat struct {
	SupportsEagerToolInputStreaming *bool `json:"supportsEagerToolInputStreaming,omitempty"`
	SupportsLongCacheRetention      *bool `json:"supportsLongCacheRetention,omitempty"`
	SendSessionAffinityHeaders      *bool `json:"sendSessionAffinityHeaders,omitempty"`
	SupportsCacheControlOnTools     *bool `json:"supportsCacheControlOnTools,omitempty"`
	SupportsTemperature             *bool `json:"supportsTemperature,omitempty"`
	ForceAdaptiveThinking           *bool `json:"forceAdaptiveThinking,omitempty"`
	AllowEmptySignature             *bool `json:"allowEmptySignature,omitempty"`
	SupportsStrictTools             *bool `json:"supportsStrictTools,omitempty"`
	SupportsToolReferences          *bool `json:"supportsToolReferences,omitempty"`
}

// AnthropicOptions are the stream options for the anthropic-messages API
// (AnthropicOptions in anthropic-messages.ts).
type AnthropicOptions struct {
	StreamOptions
	// ThinkingEnabled enables extended thinking (tri-state: nil means omitted).
	ThinkingEnabled *bool
	// ThinkingBudgetTokens is the budget for older (non-adaptive) models.
	// Default: 1024 when ThinkingEnabled is true.
	ThinkingBudgetTokens int
	// Effort is the adaptive-thinking effort: "low" | "medium" | "high" |
	// "xhigh" | "max".
	Effort string
	// ThinkingDisplay is "summarized" (default) or "omitted".
	ThinkingDisplay string
	// InterleavedThinking requests the interleaved-thinking beta header for
	// non-adaptive models. Default: true.
	InterleavedThinking *bool
	// ToolChoice is "auto" | "any" | "none" or a {"type":"tool","name":...} map.
	ToolChoice any
	// Reasoning and ThinkingBudgets are carried for parity with
	// SimpleStreamOptions; the wire protocol itself is driven by
	// ThinkingEnabled/ThinkingBudgetTokens/Effort.
	Reasoning       ThinkingLevel
	ThinkingBudgets *ThinkingBudgets
	// Compat carries AnthropicMessagesCompat on the options (see the
	// DEVIATION note at the top of this file).
	Compat *AnthropicMessagesCompat
}

// anthropicResolvedCompat is the concrete form of AnthropicMessagesCompat
// after default resolution (getAnthropicCompat in anthropic-messages.ts).
type anthropicResolvedCompat struct {
	SupportsEagerToolInputStreaming bool
	SupportsLongCacheRetention      bool
	SendSessionAffinityHeaders      bool
	SupportsCacheControlOnTools     bool
	SupportsTemperature             bool
	ForceAdaptiveThinking           bool
	AllowEmptySignature             bool
	SupportsStrictTools             bool
	SupportsToolReferences          bool
}

// getAnthropicCompat mirrors getAnthropicCompat in anthropic-messages.ts.
// Options-carried compat wins; otherwise the model's raw catalog compat is
// decoded (see DEVIATION note).
func getAnthropicCompat(model *Model, options *AnthropicOptions) anthropicResolvedCompat {
	var c AnthropicMessagesCompat
	if options != nil && options.Compat != nil {
		c = *options.Compat
	} else {
		model.DecodeRawCompat(&c)
	}
	pick := func(override *bool, def bool) bool {
		if override != nil {
			return *override
		}
		return def
	}
	return anthropicResolvedCompat{
		SupportsEagerToolInputStreaming: pick(c.SupportsEagerToolInputStreaming, true),
		SupportsLongCacheRetention:      pick(c.SupportsLongCacheRetention, true),
		SendSessionAffinityHeaders:      pick(c.SendSessionAffinityHeaders, false),
		SupportsCacheControlOnTools:     pick(c.SupportsCacheControlOnTools, true),
		SupportsTemperature:             pick(c.SupportsTemperature, true),
		ForceAdaptiveThinking:           pick(c.ForceAdaptiveThinking, false),
		AllowEmptySignature:             pick(c.AllowEmptySignature, false),
		SupportsStrictTools:             pick(c.SupportsStrictTools, false),
		SupportsToolReferences:          pick(c.SupportsToolReferences, defaultAnthropicToolReferences(model)),
	}
}

var anthropicToolRefModelRe = regexp.MustCompile(`^claude-(?:opus|sonnet|fable)-(\d+)(?:-(\d+))?(?:-|$)`)

// defaultAnthropicToolReferences mirrors defaultSupportsToolReferences in
// anthropic-messages.ts: first-party Anthropic models except Haiku and models
// older than Claude 4.5.
func defaultAnthropicToolReferences(model *Model) bool {
	if model.Provider != "anthropic" || strings.Contains(model.ID, "haiku") {
		return false
	}
	m := anthropicToolRefModelRe.FindStringSubmatch(model.ID)
	if m == nil {
		return false
	}
	major, _ := strconv.Atoi(m[1])
	minor := 0
	if len(m[2]) > 0 && len(m[2]) < 8 {
		minor, _ = strconv.Atoi(m[2])
	}
	return major > 4 || (major == 4 && minor >= 5)
}

// ---------------------------------------------------------------------------
// Cache control
// ---------------------------------------------------------------------------

type anthropicCacheControl map[string]any

// getAnthropicCacheControl mirrors getCacheControl in anthropic-messages.ts:
// nil when retention resolves to "none"; ttl "1h" for long retention when the
// provider supports it.
func getAnthropicCacheControl(options *AnthropicOptions, compat anthropicResolvedCompat) anthropicCacheControl {
	var retentionOpt CacheRetention
	var env map[string]string
	if options != nil {
		retentionOpt = options.CacheRetention
		env = options.Env
	}
	retention := resolveCacheRetention(retentionOpt, env)
	if retention == CacheRetentionNone {
		return nil
	}
	cc := anthropicCacheControl{"type": "ephemeral"}
	if retention == CacheRetentionLong && compat.SupportsLongCacheRetention {
		cc["ttl"] = "1h"
	}
	return cc
}

// ---------------------------------------------------------------------------
// Claude Code (OAuth) identity
// ---------------------------------------------------------------------------

// Stealth mode: mimic Claude Code's tool naming exactly (anthropic-messages.ts).
const claudeCodeVersion = "2.1.75"

var claudeCodeTools = []string{
	"Read", "Write", "Edit", "Bash", "Grep", "Glob", "AskUserQuestion",
	"EnterPlanMode", "ExitPlanMode", "KillShell", "NotebookEdit", "Skill",
	"Task", "TaskOutput", "TodoWrite", "WebFetch", "WebSearch",
}

// toClaudeCodeName converts a tool name to CC canonical casing when it
// matches case-insensitively (toClaudeCodeName in anthropic-messages.ts).
func toClaudeCodeName(name string) string {
	for _, t := range claudeCodeTools {
		if strings.EqualFold(t, name) {
			return t
		}
	}
	return name
}

// fromClaudeCodeName maps a streamed tool name back to the caller's casing
// (fromClaudeCodeName in anthropic-messages.ts).
func fromClaudeCodeName(name string, tools []Tool) string {
	for _, tool := range tools {
		if strings.EqualFold(tool.Name, name) {
			return tool.Name
		}
	}
	return name
}

func isAnthropicOAuthToken(apiKey string) bool {
	return strings.Contains(apiKey, "sk-ant-oat")
}

// anthropicAssertRequestAuth mirrors assertRequestAuth in
// anthropic-messages.ts: an explicit API key wins (OAuth detected by prefix);
// otherwise a caller-supplied authorization/x-api-key/cf-aig-authorization
// header satisfies auth.
func anthropicAssertRequestAuth(provider, apiKey string, headers ProviderHeaders) (string, bool, error) {
	if apiKey != "" {
		return apiKey, isAnthropicOAuthToken(apiKey), nil
	}
	if hasHeader(headers, "authorization") || hasHeader(headers, "x-api-key") || hasHeader(headers, "cf-aig-authorization") {
		return "", false, nil
	}
	return "", false, fmt.Errorf("no API key for provider: %s", provider)
}

// ---------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------

const (
	anthropicFineGrainedToolStreamingBeta = "fine-grained-tool-streaming-2025-05-14"
	anthropicInterleavedThinkingBeta      = "interleaved-thinking-2025-05-14"
	// anthropicAPIVersion is the anthropic-version header the pinned SDK sends.
	anthropicAPIVersion = "2023-06-01"
)

// buildAnthropicRequestHeaders mirrors createClient in anthropic-messages.ts
// (the github-copilot provider branch is deferred). A nil option-header value
// suppresses a default header with the same name.
func buildAnthropicRequestHeaders(model *Model, options *AnthropicOptions, compat anthropicResolvedCompat, isOAuth bool, useFineGrainedToolStreamingBeta bool, cacheSessionID string) map[string]string {
	interleaved := true
	if options != nil && options.InterleavedThinking != nil {
		interleaved = *options.InterleavedThinking
	}
	// Adaptive thinking models have interleaved thinking built in.
	needsInterleavedBeta := interleaved && !compat.ForceAdaptiveThinking
	var betaFeatures []string
	if useFineGrainedToolStreamingBeta {
		betaFeatures = append(betaFeatures, anthropicFineGrainedToolStreamingBeta)
	}
	if needsInterleavedBeta {
		betaFeatures = append(betaFeatures, anthropicInterleavedThinkingBeta)
	}

	headers := map[string]string{
		"accept": "application/json",
		"anthropic-dangerous-direct-browser-access": "true",
	}
	if isOAuth {
		// OAuth: Bearer auth, Claude Code identity headers.
		betas := append([]string{"claude-code-20250219", "oauth-2025-04-20"}, betaFeatures...)
		headers["anthropic-beta"] = strings.Join(betas, ",")
		headers["user-agent"] = "claude-cli/" + claudeCodeVersion
		headers["x-app"] = "cli"
	} else if len(betaFeatures) > 0 {
		headers["anthropic-beta"] = strings.Join(betaFeatures, ",")
	}
	if cacheSessionID != "" && compat.SendSessionAffinityHeaders {
		headers["x-session-affinity"] = cacheSessionID
	}
	for k, v := range model.Headers {
		headers[k] = v
	}
	// Merge options headers last so they can override defaults.
	if options != nil {
		for k, v := range options.Headers {
			if v == nil {
				deleteHeaderCaseInsensitive(headers, k)
				continue
			}
			headers[k] = *v
		}
	}
	return headers
}

// ---------------------------------------------------------------------------
// buildParams
// ---------------------------------------------------------------------------

var anthropicToolCallIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// anthropicNormalizeToolCallID adapts normalizeToolCallId in
// anthropic-messages.ts to the TransformMessages signature.
func anthropicNormalizeToolCallID(id string, _ *Model, _ *AssistantMessage) string {
	out := anthropicToolCallIDChars.ReplaceAllString(id, "_")
	if len(out) > 64 {
		out = out[:64]
	}
	return out
}

// splitAnthropicDeferredTools ports splitDeferredTools in
// utils/deferred-tools.ts: tools named by toolResult.addedToolNames that were
// not yet used in the transcript become deferred (loaded via tool_reference).
func splitAnthropicDeferredTools(tools []Tool, messages []Message, enabled bool, normalizeName func(string) string) (immediate []Tool, deferred []Tool) {
	var order []string
	byName := map[string]Tool{}
	for _, tool := range tools {
		name := normalizeName(tool.Name)
		if _, ok := byName[name]; !ok {
			order = append(order, name)
		}
		byName[name] = tool
	}
	if !enabled {
		for _, name := range order {
			immediate = append(immediate, byName[name])
		}
		return immediate, nil
	}
	deferredNames := map[string]bool{}
	usedNames := map[string]bool{}
	for _, msg := range messages {
		switch m := anthropicDerefMessage(msg).(type) {
		case AssistantMessage:
			for _, block := range m.Content {
				if tc, ok := block.(ToolCall); ok {
					usedNames[normalizeName(tc.Name)] = true
				}
			}
		case ToolResultMessage:
			for _, name := range m.AddedToolNames {
				n := normalizeName(name)
				if !usedNames[n] {
					deferredNames[n] = true
				}
			}
		}
	}
	for _, name := range order {
		if deferredNames[name] {
			deferred = append(deferred, byName[name])
		} else {
			immediate = append(immediate, byName[name])
		}
	}
	return immediate, deferred
}

// anthropicDerefMessage normalizes pointer message values (untyped callers
// may hand us either form; TS dispatches on msg.role).
func anthropicDerefMessage(msg Message) Message {
	switch m := msg.(type) {
	case *UserMessage:
		if m != nil {
			return *m
		}
	case *AssistantMessage:
		if m != nil {
			return *m
		}
	case *ToolResultMessage:
		if m != nil {
			return *m
		}
	}
	return msg
}

// buildAnthropicParams builds the messages.create request body. Port of
// buildParams in anthropic-messages.ts.
func buildAnthropicParams(model *Model, conv *Context, options *AnthropicOptions, compat anthropicResolvedCompat, isOAuth bool, cacheControl anthropicCacheControl) (map[string]any, error) {
	transformed := TransformMessages(conv.Messages, model, anthropicNormalizeToolCallID)
	normalizeToolName := func(name string) string { return name }
	if isOAuth {
		normalizeToolName = toClaudeCodeName
	}
	immediate, deferred := splitAnthropicDeferredTools(conv.Tools, transformed, compat.SupportsToolReferences, normalizeToolName)
	if len(immediate) == 0 && len(deferred) > 0 {
		immediate, deferred = deferred, nil
	}
	deferredNames := map[string]bool{}
	for _, tool := range deferred {
		deferredNames[normalizeToolName(tool.Name)] = true
	}

	maxTokens := model.MaxTokens
	if options != nil && options.MaxTokens > 0 {
		maxTokens = options.MaxTokens
	}
	params := map[string]any{
		"model":      model.ID,
		"messages":   anthropicConvertMessages(transformed, isOAuth, cacheControl, compat.AllowEmptySignature, deferredNames, normalizeToolName),
		"max_tokens": maxTokens,
		"stream":     true,
	}

	// For OAuth tokens, we MUST include the Claude Code identity
	// (anthropic-messages.ts).
	if isOAuth {
		system := []any{}
		identity := map[string]any{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}
		if cacheControl != nil {
			identity["cache_control"] = cacheControl
		}
		system = append(system, identity)
		if conv.SystemPrompt != "" {
			block := map[string]any{"type": "text", "text": sanitizeSurrogates(conv.SystemPrompt)}
			if cacheControl != nil {
				block["cache_control"] = cacheControl
			}
			system = append(system, block)
		}
		params["system"] = system
	} else if conv.SystemPrompt != "" {
		block := map[string]any{"type": "text", "text": sanitizeSurrogates(conv.SystemPrompt)}
		if cacheControl != nil {
			block["cache_control"] = cacheControl
		}
		params["system"] = []any{block}
	}

	thinkingEnabled := options != nil && options.ThinkingEnabled != nil && *options.ThinkingEnabled

	// Temperature is incompatible with extended thinking and unsupported on
	// some models (anthropic-messages.ts).
	if options != nil && options.Temperature != nil && !thinkingEnabled && compat.SupportsTemperature {
		params["temperature"] = *options.Temperature
	}

	if len(immediate) > 0 || len(deferred) > 0 {
		var toolCacheControl anthropicCacheControl
		if compat.SupportsCacheControlOnTools {
			toolCacheControl = cacheControl
		}
		immediateTools, err := anthropicConvertTools(immediate, isOAuth, compat, toolCacheControl, false)
		if err != nil {
			return nil, err
		}
		deferredTools, err := anthropicConvertTools(deferred, isOAuth, compat, nil, true)
		if err != nil {
			return nil, err
		}
		params["tools"] = append(immediateTools, deferredTools...)
	}

	// Configure thinking mode: adaptive, budget-based, or explicitly disabled
	// (anthropic-messages.ts).
	if model.Reasoning {
		if thinkingEnabled {
			// Default to "summarized" so newer models behave like older
			// Claude 4 models.
			display := options.ThinkingDisplay
			if display == "" {
				display = "summarized"
			}
			if compat.ForceAdaptiveThinking {
				params["thinking"] = map[string]any{"type": "adaptive", "display": display}
				if options.Effort != "" {
					params["output_config"] = map[string]any{"effort": options.Effort}
				}
			} else {
				budget := options.ThinkingBudgetTokens
				if budget == 0 {
					budget = 1024
				}
				params["thinking"] = map[string]any{"type": "enabled", "budget_tokens": budget, "display": display}
			}
		} else if options.ThinkingEnabled != nil && !*options.ThinkingEnabled && offIsNotNull(model) {
			params["thinking"] = map[string]any{"type": "disabled"}
		}
	}

	if options != nil && options.Metadata != nil {
		if userID, ok := options.Metadata["user_id"].(string); ok {
			params["metadata"] = map[string]any{"user_id": userID}
		}
	}

	if options != nil && options.ToolChoice != nil {
		if s, ok := options.ToolChoice.(string); ok {
			params["tool_choice"] = map[string]any{"type": s}
		} else {
			params["tool_choice"] = options.ToolChoice
		}
	}

	return params, nil
}

// ---------------------------------------------------------------------------
// Tool serialization
// ---------------------------------------------------------------------------

// resolveAnthropicStrictSampling ports resolveJsonSchemaStrictSampling in
// constrained-sampling.ts.
func resolveAnthropicStrictSampling(tool Tool, supportsStrict bool) (bool, error) {
	if len(tool.ConstrainedSampling) == 0 {
		return false, nil
	}
	var config struct {
		Type   string `json:"type"`
		Strict string `json:"strict"`
	}
	if err := json.Unmarshal(tool.ConstrainedSampling, &config); err != nil || config.Type != "json_schema" {
		return false, nil
	}
	if supportsStrict {
		return true, nil
	}
	if config.Strict == "require" {
		return false, fmt.Errorf("Tool %q requires JSON-schema constrained sampling, but strict tools are unsupported", tool.Name)
	}
	return false, nil
}

// anthropicConvertTools serializes tools for the request (convertTools in
// anthropic-messages.ts).
func anthropicConvertTools(tools []Tool, isOAuth bool, compat anthropicResolvedCompat, cacheControl anthropicCacheControl, deferLoading bool) ([]any, error) {
	out := make([]any, 0, len(tools))
	for i, tool := range tools {
		strict, err := resolveAnthropicStrictSampling(tool, compat.SupportsStrictTools)
		if err != nil {
			return nil, err
		}
		var schema struct {
			Properties json.RawMessage `json:"properties"`
			Required   []string        `json:"required"`
		}
		if len(tool.Parameters) > 0 {
			_ = json.Unmarshal(tool.Parameters, &schema)
		}
		var properties any = map[string]any{}
		if len(schema.Properties) > 0 {
			properties = schema.Properties
		}
		var required any = []any{}
		if schema.Required != nil {
			required = schema.Required
		}
		legacyInputSchema := map[string]any{
			"type":       "object",
			"properties": properties,
			"required":   required,
		}
		inputSchema := legacyInputSchema
		if strict {
			// strict: {...tool.parameters, ...legacyInputSchema}
			merged := map[string]any{}
			if len(tool.Parameters) > 0 {
				_ = json.Unmarshal(tool.Parameters, &merged)
			}
			for k, v := range legacyInputSchema {
				merged[k] = v
			}
			inputSchema = merged
		}

		name := tool.Name
		if isOAuth {
			name = toClaudeCodeName(name)
		}
		entry := map[string]any{
			"name":         name,
			"description":  tool.Description,
			"input_schema": inputSchema,
		}
		if compat.SupportsEagerToolInputStreaming {
			entry["eager_input_streaming"] = true
		}
		if strict {
			entry["strict"] = true
		}
		if deferLoading {
			entry["defer_loading"] = true
		}
		if cacheControl != nil && i == len(tools)-1 {
			entry["cache_control"] = cacheControl
		}
		out = append(out, entry)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

// anthropicConvertContentBlocks ports convertContentBlocks in
// anthropic-messages.ts: text-only content collapses to a joined string;
// content with images becomes a block array (with a placeholder text block
// when there is no text).
func anthropicConvertContentBlocks(content []ContentBlock) any {
	hasImages := false
	for _, c := range content {
		if _, ok := c.(ImageContent); ok {
			hasImages = true
			break
		}
	}
	if !hasImages {
		var texts []string
		for _, c := range content {
			if t, ok := c.(TextContent); ok {
				texts = append(texts, t.Text)
			}
		}
		return sanitizeSurrogates(strings.Join(texts, "\n"))
	}

	blocks := make([]any, 0, len(content)+1)
	hasText := false
	for _, c := range content {
		switch b := c.(type) {
		case TextContent:
			hasText = true
			blocks = append(blocks, map[string]any{"type": "text", "text": sanitizeSurrogates(b.Text)})
		case ImageContent:
			blocks = append(blocks, map[string]any{
				"type":   "image",
				"source": map[string]any{"type": "base64", "media_type": b.MimeType, "data": b.Data},
			})
		}
	}
	if !hasText {
		blocks = append([]any{map[string]any{"type": "text", "text": "(see attached image)"}}, blocks...)
	}
	return blocks
}

// anthropicConvertToolResult ports convertToolResult in anthropic-messages.ts.
// Deferred tools are loaded via tool_reference blocks; Anthropic rejects
// references mixed with ordinary content, so the ordinary content is
// displaced into sibling blocks.
func anthropicConvertToolResult(msg ToolResultMessage, isOAuth bool, deferredNames map[string]bool, loadedNames map[string]bool, normalizeName func(string) string) (toolResult map[string]any, siblings []any) {
	var references []any
	for _, name := range msg.AddedToolNames {
		normalized := normalizeName(name)
		if !deferredNames[normalized] || loadedNames[normalized] {
			continue
		}
		loadedNames[normalized] = true
		refName := name
		if isOAuth {
			refName = toClaudeCodeName(name)
		}
		references = append(references, map[string]any{"type": "tool_reference", "tool_name": refName})
	}
	converted := anthropicConvertContentBlocks(msg.Content)
	var content = any(converted)
	if len(references) > 0 {
		content = references
	}
	toolResult = map[string]any{
		"type":        "tool_result",
		"tool_use_id": msg.ToolCallID,
		"content":     content,
		"is_error":    msg.IsError,
	}
	if len(references) > 0 {
		if s, ok := converted.(string); ok {
			siblings = []any{map[string]any{"type": "text", "text": s}}
		} else {
			siblings = converted.([]any)
		}
	}
	return toolResult, siblings
}

// anthropicConvertMessages ports convertMessages in anthropic-messages.ts.
func anthropicConvertMessages(messages []Message, isOAuth bool, cacheControl anthropicCacheControl, allowEmptySignature bool, deferredNames map[string]bool, normalizeName func(string) string) []any {
	params := []any{}
	loadedNames := map[string]bool{}

	for i := 0; i < len(messages); i++ {
		switch msg := anthropicDerefMessage(messages[i]).(type) {
		case UserMessage:
			if s, ok := msg.StringContent(); ok {
				if strings.TrimSpace(s) != "" {
					params = append(params, map[string]any{"role": "user", "content": sanitizeSurrogates(s)})
				}
				continue
			}
			var blocks []any
			for _, item := range msg.BlockContent() {
				switch b := item.(type) {
				case TextContent:
					if strings.TrimSpace(b.Text) == "" {
						continue
					}
					blocks = append(blocks, map[string]any{"type": "text", "text": sanitizeSurrogates(b.Text)})
				case ImageContent:
					blocks = append(blocks, map[string]any{
						"type":   "image",
						"source": map[string]any{"type": "base64", "media_type": b.MimeType, "data": b.Data},
					})
				}
			}
			if len(blocks) == 0 {
				continue
			}
			params = append(params, map[string]any{"role": "user", "content": blocks})
		case AssistantMessage:
			var blocks []any
			for _, block := range msg.Content {
				switch b := block.(type) {
				case TextContent:
					if strings.TrimSpace(b.Text) == "" {
						continue
					}
					blocks = append(blocks, map[string]any{"type": "text", "text": sanitizeSurrogates(b.Text)})
				case ThinkingContent:
					// Redacted thinking: pass the opaque payload back.
					if b.Redacted {
						blocks = append(blocks, map[string]any{"type": "redacted_thinking", "data": b.ThinkingSignature})
						continue
					}
					hasSignature := strings.TrimSpace(b.ThinkingSignature) != ""
					if strings.TrimSpace(b.Thinking) == "" && !hasSignature {
						continue
					}
					if !hasSignature {
						// Missing signature (e.g. aborted stream) → plain text,
						// unless the provider accepts empty signatures
						// (anthropic-messages.ts).
						if allowEmptySignature {
							blocks = append(blocks, map[string]any{"type": "thinking", "thinking": sanitizeSurrogates(b.Thinking), "signature": ""})
						} else {
							blocks = append(blocks, map[string]any{"type": "text", "text": sanitizeSurrogates(b.Thinking)})
						}
					} else {
						blocks = append(blocks, map[string]any{"type": "thinking", "thinking": sanitizeSurrogates(b.Thinking), "signature": b.ThinkingSignature})
					}
				case ToolCall:
					name := b.Name
					if isOAuth {
						name = toClaudeCodeName(name)
					}
					input := b.Arguments
					if input == nil {
						input = map[string]any{}
					}
					blocks = append(blocks, map[string]any{"type": "tool_use", "id": b.ID, "name": name, "input": input})
				}
			}
			if len(blocks) == 0 {
				continue
			}
			params = append(params, map[string]any{"role": "assistant", "content": blocks})
		case ToolResultMessage:
			// Collect all consecutive toolResult messages into one user
			// message (anthropic-messages.ts).
			var toolResults []any
			var siblings []any
			j := i
			for j < len(messages) {
				tr, ok := anthropicDerefMessage(messages[j]).(ToolResultMessage)
				if !ok {
					break
				}
				result, sib := anthropicConvertToolResult(tr, isOAuth, deferredNames, loadedNames, normalizeName)
				toolResults = append(toolResults, result)
				siblings = append(siblings, sib...)
				j++
			}
			i = j - 1
			// Displaced reference-bearing content follows every tool_result.
			params = append(params, map[string]any{"role": "user", "content": append(toolResults, siblings...)})
		}
	}

	// Add cache_control to the last user message to cache conversation
	// history (anthropic-messages.ts).
	if cacheControl != nil && len(params) > 0 {
		if last, ok := params[len(params)-1].(map[string]any); ok && last["role"] == "user" {
			switch content := last["content"].(type) {
			case []any:
				if len(content) > 0 {
					if block, ok := content[len(content)-1].(map[string]any); ok {
						switch block["type"] {
						case "text", "image", "tool_result":
							block["cache_control"] = cacheControl
						}
					}
				}
			case string:
				last["content"] = []any{map[string]any{"type": "text", "text": content, "cache_control": cacheControl}}
			}
		}
	}

	return params
}

// ---------------------------------------------------------------------------
// Stop reasons
// ---------------------------------------------------------------------------

// mapAnthropicStopReason ports mapStopReason in anthropic-messages.ts.
// Unknown stop reasons are an error, matching the TS throw.
func mapAnthropicStopReason(reason, refusalExplanation string) (StopReason, string, error) {
	switch reason {
	case "end_turn":
		return StopStop, "", nil
	case "max_tokens":
		return StopLength, "", nil
	case "tool_use":
		return StopToolUse, "", nil
	case "refusal":
		if refusalExplanation != "" {
			return StopError, refusalExplanation, nil
		}
		return StopError, "The model refused to complete the request", nil
	case "pause_turn": // Stop is good enough -> resubmit
		return StopStop, "", nil
	case "stop_sequence": // We don't supply stop sequences, so this should never happen
		return StopStop, "", nil
	case "sensitive": // Content flagged by safety filters
		return StopError, "Provider stopped with: sensitive", nil
	default:
		return "", "", fmt.Errorf("unhandled stop reason: %s", reason)
	}
}

// ---------------------------------------------------------------------------
// stream / streamSimple
// ---------------------------------------------------------------------------

// StreamAnthropicMessages starts an anthropic-messages streaming request and
// returns the event stream. Failures are encoded in the stream as an error
// event, never thrown or panicked (pi's StreamFunction contract).
func StreamAnthropicMessages(model *Model, conv *Context, options *AnthropicOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &AnthropicOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &anthropicMessagesStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamAnthropicMessagesSimple maps SimpleStreamOptions onto the
// anthropic-messages stream (streamSimple in anthropic-messages.ts). Unlike
// the TS original, which throws when no API key is available, the Go port
// surfaces that failure as a regular error event on the returned stream.
func StreamAnthropicMessagesSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)
	thinkingOff := false
	disabled := &AnthropicOptions{StreamOptions: base, ThinkingEnabled: &thinkingOff}
	if options.Reasoning == "" {
		return StreamAnthropicMessages(model, conv, disabled)
	}
	reasoning := ClampThinkingLevel(model, options.Reasoning)
	if reasoning == ThinkingOff {
		return StreamAnthropicMessages(model, conv, disabled)
	}

	// For models with adaptive thinking: use an effort level. For older
	// models: use budget-based thinking (streamSimple in anthropic-messages.ts).
	if getAnthropicCompat(model, nil).ForceAdaptiveThinking {
		enabled := true
		return StreamAnthropicMessages(model, conv, &AnthropicOptions{
			StreamOptions:   base,
			ThinkingEnabled: &enabled,
			Effort:          anthropicMapThinkingLevelToEffort(model, reasoning),
		})
	}

	maxTokens, thinkingBudget := adjustAnthropicMaxTokensForThinking(base.MaxTokens, model.MaxTokens, reasoning, options.ThinkingBudgets)
	maxTokens = ClampMaxTokensToContext(model, conv, maxTokens)
	enabled := true
	return StreamAnthropicMessages(model, conv, &AnthropicOptions{
		StreamOptions:        base,
		ThinkingEnabled:      &enabled,
		ThinkingBudgetTokens: min(thinkingBudget, max(0, maxTokens-MinAnswerTokens)),
	})
}

// anthropicMapThinkingLevelToEffort ports mapThinkingLevelToEffort in
// anthropic-messages.ts.
func anthropicMapThinkingLevelToEffort(model *Model, level ThinkingLevel) string {
	if level != "" {
		if mapped, ok := model.ThinkingLevelMap[level]; ok && mapped != nil {
			return *mapped
		}
	}
	switch level {
	case ThinkingMinimal, ThinkingLow:
		return "low"
	case ThinkingMedium:
		return "medium"
	default:
		return "high"
	}
}

// adjustAnthropicMaxTokensForThinking ports adjustMaxTokensForThinking in
// simple-options.ts for the budget-based path (base maxTokens is always set
// by buildBaseOptions).
func adjustAnthropicMaxTokensForThinking(baseMaxTokens, modelMaxTokens int, reasoning ThinkingLevel, custom *ThinkingBudgets) (maxTokens int, thinkingBudget int) {
	budgets := ThinkingBudgets{Minimal: 1024, Low: 2048, Medium: 8192, High: 16384}
	if custom != nil {
		mergeThinkingBudgets(&budgets, custom)
	}
	thinkingBudget = thinkingBudgetForLevel(budgets, clampReasoning(reasoning))
	maxTokens = min(baseMaxTokens+thinkingBudget, modelMaxTokens)
	if maxTokens <= thinkingBudget {
		// Always leave room for the answer.
		thinkingBudget = max(0, maxTokens-MinAnswerTokens)
	}
	return maxTokens, thinkingBudget
}

type anthropicMessagesStreamer struct {
	model   *Model
	conv    *Context
	options *AnthropicOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *anthropicMessagesStreamer) run() {
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
		// Catch block of stream() in anthropic-messages.ts.
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

// snapshot returns a copy of the in-flight message for an event (see
// completionsStreamer.snapshot).
func (p *anthropicMessagesStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *anthropicMessagesStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *anthropicMessagesStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	apiKey, isOAuth, err := anthropicAssertRequestAuth(p.model.Provider, opts.APIKey, opts.Headers)
	if err != nil {
		return err
	}
	compat := getAnthropicCompat(p.model, opts)
	cacheControl := getAnthropicCacheControl(opts, compat)
	cacheSessionID := opts.SessionID
	if resolveCacheRetention(opts.CacheRetention, opts.Env) == CacheRetentionNone {
		cacheSessionID = ""
	}
	// shouldUseFineGrainedToolStreamingBeta in anthropic-messages.ts.
	useFineGrainedToolStreamingBeta := len(p.conv.Tools) > 0 && !compat.SupportsEagerToolInputStreaming
	headers := buildAnthropicRequestHeaders(p.model, opts, compat, isOAuth, useFineGrainedToolStreamingBeta, cacheSessionID)
	params, err := buildAnthropicParams(p.model, p.conv, opts, compat, isOAuth, cacheControl)
	if err != nil {
		return err
	}
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doAnthropicMessagesRequest(ctx, p.model, apiKey, isOAuth, headers, params, opts.TimeoutMs)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})
	return p.consumeStream(ctx, resp.Body, isOAuth)
}

// ---------------------------------------------------------------------------
// SSE event handling
// ---------------------------------------------------------------------------

// Wire-format event types for the Anthropic message stream
// (RawMessageStreamEvent in the SDK).

type anthropicUsageWire struct {
	InputTokens              *int `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
	CacheCreation            *struct {
		Ephemeral1hInputTokens int `json:"ephemeral_1h_input_tokens"`
	} `json:"cache_creation"`
	OutputTokensDetails *struct {
		ThinkingTokens *int `json:"thinking_tokens"`
	} `json:"output_tokens_details"`
}

type anthropicStreamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message *struct {
		ID    string             `json:"id"`
		Model string             `json:"model"`
		Usage anthropicUsageWire `json:"usage"`
	} `json:"message"`
	ContentBlock *struct {
		Type      string         `json:"type"`
		Text      string         `json:"text"`
		Thinking  string         `json:"thinking"`
		Signature string         `json:"signature"`
		Data      string         `json:"data"`
		ID        string         `json:"id"`
		Name      string         `json:"name"`
		Input     map[string]any `json:"input"`
	} `json:"content_block"`
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
		StopDetails *struct {
			Explanation string `json:"explanation"`
		} `json:"stop_details"`
	} `json:"delta"`
	Usage *anthropicUsageWire `json:"usage"`
}

func anthropicIntOrZero(v *int) int {
	if v != nil {
		return *v
	}
	return 0
}

// parseAnthropicStreamEvent decodes one event's data. The TS parses with
// parseJsonWithRepair; mirror that leniency before failing.
func parseAnthropicStreamEvent(eventName, data string) (*anthropicStreamEvent, error) {
	var event anthropicStreamEvent
	if err := json.Unmarshal([]byte(data), &event); err != nil {
		repaired, rerr := ParseJSONWithRepair(data)
		if rerr != nil {
			return nil, fmt.Errorf("could not parse Anthropic SSE event %s: %s; data=%s", eventName, err, data)
		}
		raw, _ := json.Marshal(repaired)
		if err := json.Unmarshal(raw, &event); err != nil {
			return nil, fmt.Errorf("could not parse Anthropic SSE event %s: %s; data=%s", eventName, err, data)
		}
	}
	return &event, nil
}

// consumeStream reads the SSE body and assembles the partial message. Mirrors
// iterateAnthropicEvents plus the event loop of stream() in
// anthropic-messages.ts. Unlike the completions port, block-end events are
// pushed as each content_block_stop arrives (the TS does not defer them).
func (p *anthropicMessagesStreamer) consumeStream(ctx context.Context, body io.Reader, isOAuth bool) error {
	output := p.output
	reader := NewSSEReader(body)

	type indexedBlock struct {
		index int
		block streamingBlock
		// accum accumulates this block's text or thinking deltas; see
		// textAccumulator for why the field is not appended to directly.
		//
		// A pointer, not a value: indexedBlock lives in a slice that grows as
		// blocks arrive, and appending relocates its elements. strings.Builder
		// panics if it is used after being copied, so a by-value accumulator
		// here would crash the stream as soon as a message carried enough
		// content blocks to force a reallocation.
		accum *textAccumulator
	}
	var blocks []indexedBlock
	contentIndex := func(apiIndex int) int {
		for i, b := range blocks {
			if b.index == apiIndex {
				return i
			}
		}
		return -1
	}
	syncContent := func() {
		output.Content = make([]ContentBlock, len(blocks))
		for i, b := range blocks {
			output.Content[i] = b.block.content()
		}
	}

	sawMessageStart := false
	sawMessageStop := false

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
		if sse.Event == "error" {
			// TS: throw new Error(sse.data).
			return errors.New(sse.Data)
		}
		switch sse.Event {
		case "message_start", "message_delta", "message_stop",
			"content_block_start", "content_block_delta", "content_block_stop":
		default:
			continue // ping and unknown events are skipped (anthropic-messages.ts)
		}
		data := strings.TrimSpace(sse.Data)
		if data == "" {
			continue
		}
		event, err := parseAnthropicStreamEvent(sse.Event, data)
		if err != nil {
			return err
		}

		switch event.Type {
		case "message_start":
			sawMessageStart = true
			if event.Message != nil {
				output.ResponseID = event.Message.ID
				// Capture initial token usage so input counts survive an
				// early abort (anthropic-messages.ts).
				u := event.Message.Usage
				output.Usage.Input = anthropicIntOrZero(u.InputTokens)
				output.Usage.Output = anthropicIntOrZero(u.OutputTokens)
				output.Usage.CacheRead = anthropicIntOrZero(u.CacheReadInputTokens)
				output.Usage.CacheWrite = anthropicIntOrZero(u.CacheCreationInputTokens)
				cacheWrite1h := 0
				if u.CacheCreation != nil {
					cacheWrite1h = u.CacheCreation.Ephemeral1hInputTokens
				}
				output.Usage.CacheWrite1h = &cacheWrite1h
				// Anthropic doesn't provide total_tokens; compute from components.
				output.Usage.TotalTokens = output.Usage.Input + output.Usage.Output + output.Usage.CacheRead + output.Usage.CacheWrite
				CalculateCost(p.model, &output.Usage)
			}
		case "content_block_start":
			if event.ContentBlock == nil {
				continue
			}
			cb := event.ContentBlock
			switch cb.Type {
			case "text":
				blocks = append(blocks, indexedBlock{index: event.Index, block: &TextContent{Type: "text", Text: cb.Text}})
				blocks[len(blocks)-1].accum = &textAccumulator{}
				blocks[len(blocks)-1].accum.add(cb.Text)
				syncContent()
				p.push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: len(blocks) - 1})
			case "thinking":
				blocks = append(blocks, indexedBlock{index: event.Index, block: &ThinkingContent{
					Type: "thinking", Thinking: cb.Thinking, ThinkingSignature: cb.Signature,
				}})
				blocks[len(blocks)-1].accum = &textAccumulator{}
				blocks[len(blocks)-1].accum.add(cb.Thinking)
				syncContent()
				p.push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: len(blocks) - 1})
			case "redacted_thinking":
				blocks = append(blocks, indexedBlock{index: event.Index, block: &ThinkingContent{
					Type: "thinking", Thinking: "[Reasoning redacted]", ThinkingSignature: cb.Data, Redacted: true,
				}})
				syncContent()
				p.push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: len(blocks) - 1})
			case "tool_use":
				name := cb.Name
				if isOAuth {
					name = fromClaudeCodeName(name, p.conv.Tools)
				}
				args := cb.Input
				if args == nil {
					args = map[string]any{}
				}
				blocks = append(blocks, indexedBlock{index: event.Index, block: &streamingToolCall{
					ToolCall: ToolCall{Type: "toolCall", ID: cb.ID, Name: name, Arguments: args},
				}})
				syncContent()
				p.push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: len(blocks) - 1})
			}
		case "content_block_delta":
			if event.Delta == nil {
				continue
			}
			d := event.Delta
			idx := contentIndex(event.Index)
			if idx == -1 {
				continue
			}
			switch d.Type {
			case "text_delta":
				if block, ok := blocks[idx].block.(*TextContent); ok {
					if blocks[idx].accum == nil {
						blocks[idx].accum = &textAccumulator{}
					}
					block.Text = blocks[idx].accum.add(d.Text)
					syncContent()
					p.push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: idx, Delta: d.Text})
				}
			case "thinking_delta":
				if block, ok := blocks[idx].block.(*ThinkingContent); ok {
					if blocks[idx].accum == nil {
						blocks[idx].accum = &textAccumulator{}
					}
					block.Thinking = blocks[idx].accum.add(d.Thinking)
					syncContent()
					p.push(AssistantMessageEvent{Type: EventThinkingDelta, ContentIndex: idx, Delta: d.Thinking})
				}
			case "input_json_delta":
				if block, ok := blocks[idx].block.(*streamingToolCall); ok {
					block.partialArgs.WriteString(d.PartialJSON)
					if shouldReparseStreamingJSON(block.parsedArgsLen, block.partialArgs.Len()) {
						block.ToolCall.Arguments = ParseStreamingJSON(block.partialArgs.String())
						block.parsedArgsLen = block.partialArgs.Len()
					}
					syncContent()
					p.push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: idx, Delta: d.PartialJSON})
				}
			case "signature_delta":
				if block, ok := blocks[idx].block.(*ThinkingContent); ok {
					block.ThinkingSignature += d.Signature
					// No event is pushed for signatures (anthropic-messages.ts).
					syncContent()
				}
			}
		case "content_block_stop":
			idx := contentIndex(event.Index)
			if idx == -1 {
				continue
			}
			syncContent()
			switch block := blocks[idx].block.(type) {
			case *TextContent:
				p.push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: idx, Content: block.Text})
			case *ThinkingContent:
				p.push(AssistantMessageEvent{Type: EventThinkingEnd, ContentIndex: idx, Content: block.Thinking})
			case *streamingToolCall:
				block.ToolCall.Arguments = ParseStreamingJSON(block.partialArgs.String())
				syncContent()
				tc := block.ToolCall
				p.push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: idx, ToolCall: &tc})
			}
		case "message_delta":
			if event.Delta != nil && event.Delta.StopReason != "" {
				output.RawStopReason = event.Delta.StopReason
				explanation := ""
				if event.Delta.StopDetails != nil {
					explanation = event.Delta.StopDetails.Explanation
				}
				stopReason, errorMessage, err := mapAnthropicStopReason(event.Delta.StopReason, explanation)
				if err != nil {
					return err
				}
				output.StopReason = stopReason
				if errorMessage != "" {
					output.ErrorMessage = errorMessage
				}
			}
			// Only update usage fields when present; preserves input_tokens
			// from message_start when proxies omit it here
			// (anthropic-messages.ts).
			if u := event.Usage; u != nil {
				if u.InputTokens != nil {
					output.Usage.Input = *u.InputTokens
				}
				if u.OutputTokens != nil {
					output.Usage.Output = *u.OutputTokens
				}
				if u.CacheReadInputTokens != nil {
					output.Usage.CacheRead = *u.CacheReadInputTokens
				}
				if u.CacheCreationInputTokens != nil {
					output.Usage.CacheWrite = *u.CacheCreationInputTokens
				}
				if u.OutputTokensDetails != nil && u.OutputTokensDetails.ThinkingTokens != nil {
					reasoning := *u.OutputTokensDetails.ThinkingTokens
					output.Usage.Reasoning = &reasoning
				}
			}
			output.Usage.TotalTokens = output.Usage.Input + output.Usage.Output + output.Usage.CacheRead + output.Usage.CacheWrite
			CalculateCost(p.model, &output.Usage)
		case "message_stop":
			sawMessageStop = true
		}
	}

	if sawMessageStart && !sawMessageStop {
		return errors.New("anthropic stream ended before message_stop")
	}
	if ctx.Err() != nil {
		return errors.New("request was aborted")
	}
	if output.StopReason == StopPending {
		return errors.New("anthropic stream ended without a stop reason")
	}
	if output.StopReason == StopAborted || output.StopReason == StopError {
		if output.ErrorMessage != "" {
			return errors.New(output.ErrorMessage)
		}
		return errors.New("an unknown error occurred")
	}

	syncContent()
	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

// doAnthropicMessagesRequest performs one POST {baseUrl}/v1/messages and
// returns the streaming response. Non-2xx responses become *ProviderError so
// RetryProviderRequest can classify them (mirrors the SDK request path used
// by stream() in anthropic-messages.ts).
func doAnthropicMessagesRequest(ctx context.Context, model *Model, apiKey string, isOAuth bool, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := strings.TrimSuffix(model.BaseURL, "/") + "/v1/messages"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", anthropicAPIVersion)
	if isOAuth {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	} else if apiKey != "" {
		req.Header.Set("x-api-key", apiKey)
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
