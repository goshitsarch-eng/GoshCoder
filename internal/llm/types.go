// Package llm is a Go port of the core of pi's packages/ai toolkit
// (reference/pi/packages/ai/src). JSON field names are kept identical to pi
// so session files and provider payloads stay compatible.
package llm

import (
	"context"
	"encoding/json"
	"fmt"
)

// Known APIs. Pi keeps this open-ended (Api = KnownApi | string); Go mirrors
// that with a plain string type.
type API = string

const (
	APIOpenAICompletions = "openai-completions"
)

// ProviderID identifies a provider ("openai", "openrouter", ...). Open-ended
// like pi's ProviderId.
type ProviderID = string

type ThinkingLevel = string

const (
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
	ThinkingMax     ThinkingLevel = "max"
)

// ModelThinkingLevel is ThinkingLevel plus "off".
type ModelThinkingLevel = string

const ThinkingOff ModelThinkingLevel = "off"

// ThinkingLevelMap maps pi thinking levels to provider/model-specific values.
// A nil *string value marks a level as unsupported (TS null); a missing key
// uses provider defaults (TS undefined).
type ThinkingLevelMap map[ModelThinkingLevel]*string

// ThinkingBudgets are token budgets for each thinking level (token-based
// providers only).
type ThinkingBudgets struct {
	Minimal int `json:"minimal,omitempty"`
	Low     int `json:"low,omitempty"`
	Medium  int `json:"medium,omitempty"`
	High    int `json:"high,omitempty"`
}

type CacheRetention = string

const (
	CacheRetentionNone  CacheRetention = "none"
	CacheRetentionShort CacheRetention = "short"
	CacheRetentionLong  CacheRetention = "long"
)

type Transport = string

const TransportSSE Transport = "sse"

// ProviderHeaders maps header names to values; a nil value suppresses a
// provider/API default header with the same name (TS: Record<string, string|null>).
type ProviderHeaders map[string]*string

// ProviderResponse is passed to OnResponse after an HTTP response is received.
type ProviderResponse struct {
	Status  int
	Headers map[string]string
}

// Content blocks. The concrete types implement ContentBlock; the Type field
// is the TS discriminant and is serialized as "type".

type ContentBlock interface {
	contentBlockType() string
}

type TextContent struct {
	Type string `json:"type"` // always "text"
	Text string `json:"text"`
	// e.g. for OpenAI responses, message metadata (legacy id string or
	// TextSignatureV1 JSON)
	TextSignature string `json:"textSignature,omitempty"`
}

func (TextContent) contentBlockType() string { return "text" }

type ThinkingContent struct {
	Type     string `json:"type"` // always "thinking"
	Thinking string `json:"thinking"`
	// e.g. for OpenAI responses, the reasoning item ID
	ThinkingSignature string `json:"thinkingSignature,omitempty"`
	// Redacted marks thinking redacted by safety filters; the opaque payload is
	// in ThinkingSignature.
	Redacted bool `json:"redacted,omitempty"`
}

func (ThinkingContent) contentBlockType() string { return "thinking" }

type ImageContent struct {
	Type     string `json:"type"` // always "image"
	Data     string `json:"data"` // base64 encoded image data
	MimeType string `json:"mimeType"`
}

func (ImageContent) contentBlockType() string { return "image" }

type ToolCall struct {
	Type      string         `json:"type"` // always "toolCall"
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
	// Google-specific: opaque signature for reusing thought context.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
	// OpenAI Responses namespace for calls to dynamically loaded or namespaced tools.
	Namespace string `json:"namespace,omitempty"`
}

func (ToolCall) contentBlockType() string { return "toolCall" }

// Usage mirrors pi's Usage. Token counts are ints; cost is USD.
type Usage struct {
	Input      int `json:"input"`
	Output     int `json:"output"`
	CacheRead  int `json:"cacheRead"`
	CacheWrite int `json:"cacheWrite"`
	// Subset of CacheWrite written with 1h retention. Only Anthropic reports this split.
	CacheWrite1h *int `json:"cacheWrite1h,omitempty"`
	// Reasoning tokens, a subset of Output. Set (possibly 0) by providers that
	// expose a reasoning breakdown; nil otherwise.
	Reasoning   *int      `json:"reasoning,omitempty"`
	TotalTokens int       `json:"totalTokens"`
	Cost        UsageCost `json:"cost"`
}

type UsageCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

type StopReason = string

const (
	StopPending  StopReason = "pending"
	StopStop     StopReason = "stop"
	StopLength   StopReason = "length"
	StopToolUse  StopReason = "toolUse"
	StopError    StopReason = "error"
	StopAborted  StopReason = "aborted"
	StopDeferred StopReason = "deferred"
)

type DeferredHandle struct {
	Provider string `json:"provider"`
	ModelID  string `json:"modelId"`
	API      string `json:"api"`
	// Provider token, such as a response id or batch id plus row id.
	ID        string          `json:"id"`
	ExpiresAt *int64          `json:"expiresAt,omitempty"`
	PollAfter *int64          `json:"pollAfterMs,omitempty"`
	Data      json.RawMessage `json:"data,omitempty"`
}

// Message is the sum of UserMessage, AssistantMessage, ToolResultMessage.
type Message interface {
	messageRole() string
}

type UserMessage struct {
	Role string `json:"role"` // always "user"
	// string or []ContentBlock of TextContent/ImageContent. Use StringContent /
	// BlockContent to access; custom (un)marshaling keeps pi's JSON shape.
	Content   any   `json:"content"`
	Timestamp int64 `json:"timestamp"`
}

func (UserMessage) messageRole() string { return "user" }

// StringContent reports whether Content is a plain string and returns it.
func (m *UserMessage) StringContent() (string, bool) {
	s, ok := m.Content.(string)
	return s, ok
}

// BlockContent returns Content as blocks, converting a plain string to a
// single text block for convenience. Returns nil when empty.
func (m *UserMessage) BlockContent() []ContentBlock {
	switch c := m.Content.(type) {
	case string:
		if c == "" {
			return nil
		}
		return []ContentBlock{TextContent{Type: "text", Text: c}}
	case []ContentBlock:
		return c
	default:
		return nil
	}
}

func (m UserMessage) MarshalJSON() ([]byte, error) {
	type alias UserMessage
	if m.Role == "" {
		m.Role = "user"
	}
	return json.Marshal(alias(m))
}

func (m *UserMessage) UnmarshalJSON(data []byte) error {
	type alias struct {
		Role      string          `json:"role"`
		Content   json.RawMessage `json:"content"`
		Timestamp int64           `json:"timestamp"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	m.Role, m.Timestamp = a.Role, a.Timestamp
	if len(a.Content) == 0 || string(a.Content) == "null" {
		m.Content = nil
		return nil
	}
	var s string
	if err := json.Unmarshal(a.Content, &s); err == nil {
		m.Content = s
		return nil
	}
	blocks, err := unmarshalContentBlocks(a.Content)
	if err != nil {
		return err
	}
	m.Content = blocks
	return nil
}

type AssistantMessage struct {
	Role     string         `json:"role"` // always "assistant"
	Content  []ContentBlock `json:"content"`
	API      API            `json:"api"`
	Provider ProviderID     `json:"provider"`
	Model    string         `json:"model"`
	// Concrete chunk.model when different from the requested model.
	ResponseModel string `json:"responseModel,omitempty"`
	// Provider-specific response/message identifier.
	ResponseID string `json:"responseId,omitempty"`
	// Redacted provider/runtime diagnostics, preserved verbatim for round-trip.
	Diagnostics   []json.RawMessage `json:"diagnostics,omitempty"`
	Usage         Usage             `json:"usage"`
	StopReason    StopReason        `json:"stopReason"`
	Deferred      *DeferredHandle   `json:"deferred,omitempty"`
	ErrorMessage  string            `json:"errorMessage,omitempty"`
	RawStopReason string            `json:"rawStopReason,omitempty"`
	// Provider indication of whether the model explicitly ended its turn.
	EndTurn   *bool `json:"endTurn,omitempty"`
	Timestamp int64 `json:"timestamp"`
}

func (AssistantMessage) messageRole() string { return "assistant" }

func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	type alias AssistantMessage
	if m.Role == "" {
		m.Role = "assistant"
	}
	if m.Content == nil {
		m.Content = []ContentBlock{}
	}
	return json.Marshal(alias(m))
}

func (m *AssistantMessage) UnmarshalJSON(data []byte) error {
	type alias struct {
		Role          string            `json:"role"`
		Content       json.RawMessage   `json:"content"`
		API           API               `json:"api"`
		Provider      ProviderID        `json:"provider"`
		Model         string            `json:"model"`
		ResponseModel string            `json:"responseModel,omitempty"`
		ResponseID    string            `json:"responseId,omitempty"`
		Diagnostics   []json.RawMessage `json:"diagnostics,omitempty"`
		Usage         Usage             `json:"usage"`
		StopReason    StopReason        `json:"stopReason"`
		Deferred      *DeferredHandle   `json:"deferred,omitempty"`
		ErrorMessage  string            `json:"errorMessage,omitempty"`
		RawStopReason string            `json:"rawStopReason,omitempty"`
		EndTurn       *bool             `json:"endTurn,omitempty"`
		Timestamp     int64             `json:"timestamp"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*m = AssistantMessage{
		Role: a.Role, API: a.API, Provider: a.Provider, Model: a.Model,
		ResponseModel: a.ResponseModel, ResponseID: a.ResponseID,
		Diagnostics: a.Diagnostics, Usage: a.Usage, StopReason: a.StopReason,
		Deferred: a.Deferred, ErrorMessage: a.ErrorMessage,
		RawStopReason: a.RawStopReason, EndTurn: a.EndTurn, Timestamp: a.Timestamp,
	}
	if len(a.Content) > 0 && string(a.Content) != "null" {
		blocks, err := unmarshalContentBlocks(a.Content)
		if err != nil {
			return err
		}
		m.Content = blocks
	}
	return nil
}

type ToolResultMessage struct {
	Role       string         `json:"role"` // always "toolResult"
	ToolCallID string         `json:"toolCallId"`
	ToolName   string         `json:"toolName"`
	Content    []ContentBlock `json:"content"` // text and image blocks
	Details    any            `json:"details,omitempty"`
	// Usage from the tool execution itself. Not part of main LLM context accounting.
	Usage *Usage `json:"usage,omitempty"`
	// Names from Context.Tools that became available after this result.
	AddedToolNames []string `json:"addedToolNames,omitempty"`
	IsError        bool     `json:"isError"`
	Timestamp      int64    `json:"timestamp"`
}

func (ToolResultMessage) messageRole() string { return "toolResult" }

func (m ToolResultMessage) MarshalJSON() ([]byte, error) {
	type alias ToolResultMessage
	if m.Role == "" {
		m.Role = "toolResult"
	}
	if m.Content == nil {
		m.Content = []ContentBlock{}
	}
	return json.Marshal(alias(m))
}

func (m *ToolResultMessage) UnmarshalJSON(data []byte) error {
	type alias struct {
		Role           string          `json:"role"`
		ToolCallID     string          `json:"toolCallId"`
		ToolName       string          `json:"toolName"`
		Content        json.RawMessage `json:"content"`
		Details        any             `json:"details,omitempty"`
		Usage          *Usage          `json:"usage,omitempty"`
		AddedToolNames []string        `json:"addedToolNames,omitempty"`
		IsError        bool            `json:"isError"`
		Timestamp      int64           `json:"timestamp"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*m = ToolResultMessage{
		Role: a.Role, ToolCallID: a.ToolCallID, ToolName: a.ToolName,
		Details: a.Details, Usage: a.Usage, AddedToolNames: a.AddedToolNames,
		IsError: a.IsError, Timestamp: a.Timestamp,
	}
	if len(a.Content) > 0 && string(a.Content) != "null" {
		blocks, err := unmarshalContentBlocks(a.Content)
		if err != nil {
			return err
		}
		m.Content = blocks
	}
	return nil
}

func unmarshalContentBlocks(data json.RawMessage) ([]ContentBlock, error) {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	blocks := make([]ContentBlock, 0, len(raw))
	for _, r := range raw {
		var head struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(r, &head); err != nil {
			return nil, err
		}
		var block ContentBlock
		switch head.Type {
		case "text":
			var b TextContent
			if err := json.Unmarshal(r, &b); err != nil {
				return nil, err
			}
			block = b
		case "thinking":
			var b ThinkingContent
			if err := json.Unmarshal(r, &b); err != nil {
				return nil, err
			}
			block = b
		case "image":
			var b ImageContent
			if err := json.Unmarshal(r, &b); err != nil {
				return nil, err
			}
			block = b
		case "toolCall":
			var b ToolCall
			if err := json.Unmarshal(r, &b); err != nil {
				return nil, err
			}
			block = b
		default:
			return nil, fmt.Errorf("llm: unknown content block type %q", head.Type)
		}
		blocks = append(blocks, block)
	}
	return blocks, nil
}

// Tool describes a callable tool. Parameters is the raw JSON Schema.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	// ConstrainedSampling is carried for serialization fidelity; the
	// constrained-sampling behavior is out of scope for this port slice.
	ConstrainedSampling json.RawMessage `json:"constrainedSampling,omitempty"`
}

type Context struct {
	SystemPrompt string    `json:"systemPrompt,omitempty"`
	Messages     []Message `json:"messages"`
	Tools        []Tool    `json:"tools,omitempty"`
}

func (c Context) MarshalJSON() ([]byte, error) {
	type alias Context
	if c.Messages == nil {
		c.Messages = []Message{}
	}
	return json.Marshal(alias(c))
}

func (c *Context) UnmarshalJSON(data []byte) error {
	type alias struct {
		SystemPrompt string          `json:"systemPrompt,omitempty"`
		Messages     json.RawMessage `json:"messages"`
		Tools        []Tool          `json:"tools,omitempty"`
	}
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	c.SystemPrompt, c.Tools = a.SystemPrompt, a.Tools
	if len(a.Messages) > 0 && string(a.Messages) != "null" {
		msgs, err := UnmarshalMessages(a.Messages)
		if err != nil {
			return err
		}
		c.Messages = msgs
	}
	return nil
}

// UnmarshalMessages decodes a JSON array of pi messages, dispatching on "role".
func UnmarshalMessages(data json.RawMessage) ([]Message, error) {
	var raw []json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	msgs := make([]Message, 0, len(raw))
	for _, r := range raw {
		var head struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(r, &head); err != nil {
			return nil, err
		}
		var msg Message
		switch head.Role {
		case "user":
			var m UserMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			msg = m
		case "assistant":
			var m AssistantMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			msg = m
		case "toolResult":
			var m ToolResultMessage
			if err := json.Unmarshal(r, &m); err != nil {
				return nil, err
			}
			msg = m
		default:
			return nil, fmt.Errorf("llm: unknown message role %q", head.Role)
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}

// Model cost rates in $/million tokens.
type ModelCostRates struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

type ModelCostTier struct {
	ModelCostRates
	// Use this tier for requests whose total input usage exceeds this token count.
	InputTokensAbove int `json:"inputTokensAbove"`
}

type ModelCost struct {
	ModelCostRates
	Tiers []ModelCostTier `json:"tiers,omitempty"`
}

// Model mirrors pi's Model. Compat holds OpenAI-completions compatibility
// overrides; RawCompat carries the original compat JSON so the other wire
// protocols can decode their own compat shapes (pi keys model.compat by API,
// which Go's single typed field cannot express).
type Model struct {
	ID        string     `json:"id"`
	Name      string     `json:"name"`
	API       API        `json:"api"`
	Provider  ProviderID `json:"provider"`
	BaseURL   string     `json:"baseUrl"`
	Reasoning bool       `json:"reasoning"`
	// Maps pi thinking levels to provider/model-specific values. Missing keys
	// use provider defaults; a nil *string marks a level unsupported.
	ThinkingLevelMap ThinkingLevelMap `json:"thinkingLevelMap,omitempty"`
	// Input modalities: "text" and/or "image".
	Input          []string                 `json:"input"`
	Cost           ModelCost                `json:"cost"`
	ContextWindow  int                      `json:"contextWindow"`
	MaxTokens      int                      `json:"maxTokens"`
	SamplingParams map[string]any           `json:"samplingParams,omitempty"`
	Headers        map[string]string        `json:"headers,omitempty"`
	Compat         *OpenAICompletionsCompat `json:"compat,omitempty"`
	// RawCompat is the model's compat object exactly as the catalog carries it,
	// including keys absent from OpenAICompletionsCompat (forceAdaptiveThinking,
	// supportsOpenAIGrammarTools, ...). Protocols read it through their own
	// compat resolver. Not serialized: Compat already round-trips the typed view.
	RawCompat json.RawMessage `json:"-"`
}

// DecodeRawCompat decodes the model's raw compat JSON into target. It is a
// no-op returning false when the model carries no raw compat.
func (m *Model) DecodeRawCompat(target any) bool {
	if m == nil || len(m.RawCompat) == 0 || string(m.RawCompat) == "null" {
		return false
	}
	// A malformed compat object is ignored rather than failing the request: the
	// defaults are always a usable configuration.
	return json.Unmarshal(m.RawCompat, target) == nil
}

// SupportsImages reports whether the model accepts image input.
func (m *Model) SupportsImages() bool {
	for _, in := range m.Input {
		if in == "image" {
			return true
		}
	}
	return false
}

// StreamOptions are the base options all providers share
// (pi's StreamOptions / ProviderRequestOptions).
type StreamOptions struct {
	// Ctx maps to pi's AbortSignal. nil means context.Background().
	Ctx    context.Context `json:"-"`
	APIKey string          `json:"-"`
	// OnPayload can inspect or replace the request payload before sending.
	// Return nil to keep the payload unchanged.
	OnPayload func(payload map[string]any, model *Model) map[string]any `json:"-"`
	// OnResponse is invoked after an HTTP response is received and before its
	// body stream is consumed.
	OnResponse func(response ProviderResponse, model *Model) `json:"-"`
	// Headers are merged with provider defaults; caller values override. A nil
	// value suppresses a default header with the same name.
	Headers ProviderHeaders `json:"-"`
	// HTTP request timeout in milliseconds.
	TimeoutMs int64 `json:"-"`
	// Maximum retry attempts (0 = no retries, matching the TS default).
	MaxRetries int `json:"-"`
	// Maximum delay in milliseconds to wait for a server-requested retry.
	// Default: 60000. Set to 0 to disable the cap.
	MaxRetryDelayMs *int64 `json:"-"`

	Temperature *float64 `json:"-"`
	// Arbitrary sampling parameters merged into the request body after the
	// named request fields, so keys here override them. Merged over
	// Model.SamplingParams per key by StreamSimple.
	SamplingParams map[string]any `json:"-"`
	MaxTokens      int            `json:"-"`
	// Prompt cache retention preference. Default: "short".
	CacheRetention CacheRetention `json:"-"`
	// Session identifier for providers that support session-based caching.
	SessionID string `json:"-"`
	// Provider-scoped environment overrides. Values take precedence over os.Getenv.
	Env map[string]string `json:"-"`
	// Optional metadata; providers extract the fields they understand.
	Metadata map[string]any `json:"-"`
}

func (o *StreamOptions) context() context.Context {
	if o != nil && o.Ctx != nil {
		return o.Ctx
	}
	return context.Background()
}

// SimpleStreamOptions are the unified options passed to StreamSimple.
type SimpleStreamOptions struct {
	StreamOptions
	Reasoning       ThinkingLevel
	ThinkingBudgets *ThinkingBudgets
}
