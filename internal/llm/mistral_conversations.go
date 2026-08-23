package llm

// Go port of reference/pi/packages/ai/src/api/mistral-conversations.ts.
//
// Mistral's native chat endpoint is close to openai-completions but differs in
// several ways that make it its own protocol: tool call ids must be exactly 9
// alphanumeric characters, thinking arrives as structured content chunks rather
// than a reasoning field, tool results are content arrays, and the request body
// uses camelCase field names that are remapped to snake_case on the wire.

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

const APIMistralConversations = "mistral-conversations"

func init() {
	RegisterStreamer(APIMistralConversations, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*MistralOptions)
			return StreamMistral(model, ctx, o)
		},
		StreamSimple: StreamMistralSimple,
	})
}

// mistralToolCallIDLength is the exact id length Mistral accepts.
const mistralToolCallIDLength = 9

// mistralDefaultTimeout matches the TS AbortSignal.timeout default.
const mistralDefaultTimeout = 60 * time.Second

// MistralOptions are the stream options for the mistral-conversations API.
type MistralOptions struct {
	StreamOptions
	// ToolChoice is "auto" | "none" | "any" | "required", or a map selecting a
	// specific function.
	ToolChoice any
	// PromptMode is "reasoning" for models that gate thinking behind it.
	PromptMode string
	// ReasoningEffort is "none" or "high" for models that use an effort field.
	ReasoningEffort string
}

// StreamMistral streams a turn over the mistral-conversations API.
func StreamMistral(model *Model, conv *Context, options *MistralOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &MistralOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &mistralStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamMistralSimple maps SimpleStreamOptions onto the Mistral stream
// (streamSimple in mistral-conversations.ts). Unlike the TS original, which
// throws when no API key is available, the Go port surfaces that failure as a
// regular error event on the returned stream.
func StreamMistralSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)

	reasoning := ""
	if options.Reasoning != "" {
		if clamped := ClampThinkingLevel(model, options.Reasoning); clamped != ThinkingOff {
			reasoning = clamped
		}
	}
	useReasoning := model.Reasoning && reasoning != ""

	opts := &MistralOptions{StreamOptions: base}
	switch {
	case useReasoning && mistralUsesReasoningEffort(model):
		opts.ReasoningEffort = mistralMapReasoningEffort(model, reasoning)
	case useReasoning && mistralUsesPromptModeReasoning(model):
		opts.PromptMode = "reasoning"
	}
	return StreamMistral(model, conv, opts)
}

// mistralUsesReasoningEffort reports whether the model takes a reasoning_effort
// field rather than prompt_mode.
func mistralUsesReasoningEffort(model *Model) bool {
	switch model.ID {
	case "mistral-small-2603", "mistral-small-latest", "mistral-medium-3.5":
		return true
	default:
		return false
	}
}

func mistralUsesPromptModeReasoning(model *Model) bool {
	return model.Reasoning && !mistralUsesReasoningEffort(model)
}

// mistralMapReasoningEffort maps a pi thinking level onto Mistral's effort
// values, defaulting to "high".
func mistralMapReasoningEffort(model *Model, level ThinkingLevel) string {
	if mapped, ok := model.ThinkingLevelMap[level]; ok && mapped != nil {
		return *mapped
	}
	return "high"
}

// ---------------------------------------------------------------------------
// Tool call id normalization
// ---------------------------------------------------------------------------

var mistralNonAlphanumeric = regexp.MustCompile(`[^a-zA-Z0-9]`)

// deriveMistralToolCallID produces a 9-character alphanumeric id. attempt
// disambiguates collisions.
func deriveMistralToolCallID(id string, attempt int) string {
	normalized := mistralNonAlphanumeric.ReplaceAllString(id, "")
	if attempt == 0 && len(normalized) == mistralToolCallIDLength {
		return normalized
	}
	seed := normalized
	if seed == "" {
		seed = id
	}
	if attempt > 0 {
		seed = fmt.Sprintf("%s:%d", seed, attempt)
	}
	hashed := mistralNonAlphanumeric.ReplaceAllString(ShortHash(seed), "")
	if len(hashed) > mistralToolCallIDLength {
		hashed = hashed[:mistralToolCallIDLength]
	}
	return hashed
}

// mistralToolCallIDNormalizer returns a stateful normalizer that keeps ids
// stable and collision-free within one request.
func mistralToolCallIDNormalizer() func(string, *Model, *AssistantMessage) string {
	forward := map[string]string{}
	reverse := map[string]string{}

	return func(id string, _ *Model, _ *AssistantMessage) string {
		if existing, ok := forward[id]; ok {
			return existing
		}
		for attempt := 0; ; attempt++ {
			candidate := deriveMistralToolCallID(id, attempt)
			owner, taken := reverse[candidate]
			if !taken || owner == id {
				forward[id] = candidate
				reverse[candidate] = id
				return candidate
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

// buildMistralParams builds the request body. Field names are the wire
// (snake_case) forms directly, which is where the TS remapping step lands.
func buildMistralParams(model *Model, conv *Context, options *MistralOptions) (map[string]any, error) {
	normalize := mistralToolCallIDNormalizer()
	transformed := TransformMessages(conv.Messages, model, normalize)

	messages := mistralConvertMessages(transformed, model.SupportsImages())
	if conv.SystemPrompt != "" {
		system := map[string]any{"role": "system", "content": sanitizeSurrogates(conv.SystemPrompt)}
		messages = append([]any{system}, messages...)
	}

	params := map[string]any{
		"model":    model.ID,
		"stream":   true,
		"messages": messages,
	}
	if len(conv.Tools) > 0 {
		tools, err := mistralConvertTools(conv.Tools)
		if err != nil {
			return nil, err
		}
		params["tools"] = tools
	}
	if options.Temperature != nil {
		params["temperature"] = *options.Temperature
	}
	if options.MaxTokens > 0 {
		params["max_tokens"] = options.MaxTokens
	}
	if choice := mistralMapToolChoice(options.ToolChoice); choice != nil {
		params["tool_choice"] = choice
	}
	if options.PromptMode != "" {
		params["prompt_mode"] = options.PromptMode
	}
	if options.ReasoningEffort != "" {
		params["reasoning_effort"] = options.ReasoningEffort
	}
	if mistralShouldCachePrompt(options) {
		params["prompt_cache_key"] = options.SessionID
	}

	// Last so custom keys override the named request fields.
	for k, v := range options.SamplingParams {
		params[k] = v
	}
	return params, nil
}

// mistralShouldCachePrompt reports whether prompt caching applies.
func mistralShouldCachePrompt(options *MistralOptions) bool {
	return options.CacheRetention != CacheRetentionNone && options.SessionID != ""
}

// mistralMapToolChoice normalizes the tool choice, returning nil to omit it.
func mistralMapToolChoice(choice any) any {
	switch value := choice.(type) {
	case nil:
		return nil
	case string:
		switch value {
		case "auto", "none", "any", "required":
			return value
		default:
			return nil
		}
	default:
		// A function-selecting map passes through as-is.
		return value
	}
}

// mistralConvertTools serializes tools as Mistral function tools.
func mistralConvertTools(tools []Tool) ([]any, error) {
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		// Mistral always supports strict mode, so a "require" config cannot fail.
		strict, hasStrict, err := ResolveJSONSchemaStrictSampling(tool, true)
		if err != nil {
			return nil, err
		}
		var parameters any = map[string]any{}
		if len(tool.Parameters) > 0 {
			parameters = json.RawMessage(tool.Parameters)
		}
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  parameters,
				"strict":      hasStrict && strict,
			},
		})
	}
	return out, nil
}

// mistralConvertMessages converts a transformed transcript to Mistral messages.
func mistralConvertMessages(messages []Message, supportsImages bool) []any {
	out := []any{}
	for _, msg := range messages {
		switch m := derefResponsesMessage(msg).(type) {
		case UserMessage:
			if text, isString := m.StringContent(); isString {
				out = append(out, map[string]any{"role": "user", "content": sanitizeSurrogates(text)})
				continue
			}
			blocks := m.BlockContent()
			hadImages := false
			content := []any{}
			for _, block := range blocks {
				switch b := block.(type) {
				case TextContent:
					content = append(content, map[string]any{"type": "text", "text": sanitizeSurrogates(b.Text)})
				case ImageContent:
					hadImages = true
					if supportsImages {
						content = append(content, map[string]any{
							"type":      "image_url",
							"image_url": fmt.Sprintf("data:%s;base64,%s", b.MimeType, b.Data),
						})
					}
				}
			}
			if len(content) > 0 {
				out = append(out, map[string]any{"role": "user", "content": content})
				continue
			}
			// An image-only message for a text model still needs a placeholder
			// so the turn is not dropped entirely.
			if hadImages && !supportsImages {
				out = append(out, map[string]any{
					"role":    "user",
					"content": "(image omitted: model does not support images)",
				})
			}

		case AssistantMessage:
			content := []any{}
			toolCalls := []any{}
			for _, block := range m.Content {
				switch b := block.(type) {
				case TextContent:
					if strings.TrimSpace(b.Text) != "" {
						content = append(content, map[string]any{"type": "text", "text": sanitizeSurrogates(b.Text)})
					}
				case ThinkingContent:
					if strings.TrimSpace(b.Thinking) != "" {
						content = append(content, map[string]any{
							"type": "thinking",
							"thinking": []any{
								map[string]any{"type": "text", "text": sanitizeSurrogates(b.Thinking)},
							},
						})
					}
				case ToolCall:
					arguments := b.Arguments
					if arguments == nil {
						arguments = map[string]any{}
					}
					encoded, err := json.Marshal(arguments)
					if err != nil {
						encoded = []byte("{}")
					}
					toolCalls = append(toolCalls, map[string]any{
						"id":       b.ID,
						"type":     "function",
						"function": map[string]any{"name": b.Name, "arguments": string(encoded)},
						"index":    0,
					})
				}
			}
			if len(content) == 0 && len(toolCalls) == 0 {
				continue
			}
			assistant := map[string]any{"role": "assistant", "prefix": false}
			if len(content) > 0 {
				assistant["content"] = content
			}
			if len(toolCalls) > 0 {
				assistant["tool_calls"] = toolCalls
			}
			out = append(out, assistant)

		case ToolResultMessage:
			var texts []string
			hasImages := false
			for _, block := range m.Content {
				switch b := block.(type) {
				case TextContent:
					texts = append(texts, sanitizeSurrogates(b.Text))
				case ImageContent:
					hasImages = true
				}
			}
			content := []any{map[string]any{
				"type": "text",
				"text": mistralToolResultText(strings.Join(texts, "\n"), hasImages, supportsImages, m.IsError),
			}}
			if supportsImages {
				for _, block := range m.Content {
					if image, ok := block.(ImageContent); ok {
						content = append(content, map[string]any{
							"type":      "image_url",
							"image_url": fmt.Sprintf("data:%s;base64,%s", image.MimeType, image.Data),
						})
					}
				}
			}
			out = append(out, map[string]any{
				"role":         "tool",
				"tool_call_id": m.ToolCallID,
				"name":         m.ToolName,
				"content":      content,
			})
		}
	}
	return out
}

// mistralToolResultText renders tool result text, marking errors and noting
// dropped images.
func mistralToolResultText(text string, hasImages, supportsImages, isError bool) string {
	trimmed := strings.TrimSpace(text)
	prefix := ""
	if isError {
		prefix = "[tool error] "
	}

	if trimmed != "" {
		suffix := ""
		if hasImages && !supportsImages {
			suffix = "\n[tool image omitted: model does not support images]"
		}
		return prefix + trimmed + suffix
	}
	if hasImages {
		if supportsImages {
			return prefix + "(see attached image)"
		}
		return prefix + "(image omitted: model does not support images)"
	}
	return prefix + "(no tool output)"
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

type mistralStreamer struct {
	model   *Model
	conv    *Context
	options *MistralOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *mistralStreamer) run() {
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

func (p *mistralStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *mistralStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *mistralStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	// Mistral has no header-only auth fallback: the key is required.
	if opts.APIKey == "" {
		return fmt.Errorf("no API key for provider: %s", p.model.Provider)
	}

	params, err := buildMistralParams(p.model, p.conv, opts)
	if err != nil {
		return err
	}
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doMistralRequest(ctx, p.model, opts, params)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	p.push(AssistantMessageEvent{Type: EventStart})
	if err := p.consumeStream(ctx, resp.Body); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return AbortError{}
	}
	if p.output.StopReason == StopPending {
		return fmt.Errorf("mistral stream ended without a finish reason")
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

// mistralStreamChunk is one streamed completion chunk.
type mistralStreamChunk struct {
	ID    string `json:"id"`
	Usage *struct {
		PromptTokens        int `json:"prompt_tokens"`
		CompletionTokens    int `json:"completion_tokens"`
		TotalTokens         int `json:"total_tokens"`
		PromptTokensDetails *struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		NumCachedTokens *int `json:"num_cached_tokens"`
	} `json:"usage"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Delta        struct {
			// Content is a string or an array of content chunks.
			Content   json.RawMessage `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Index    *int   `json:"index"`
				Function struct {
					Name string `json:"name"`
					// Arguments is a string or an object.
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
	} `json:"choices"`
}

// mistralContentChunk is one structured delta content item.
type mistralContentChunk struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	Thinking []struct {
		Text string `json:"text"`
	} `json:"thinking"`
}

// mistralStreamingToolCall tracks a tool call being assembled.
type mistralStreamingToolCall struct {
	ToolCall
	// partialArgs accumulates argument deltas; see streamingToolCall for why
	// this is a Builder rather than a string.
	partialArgs strings.Builder
	// parsedArgsLen is how much of partialArgs the Arguments map reflects.
	// See shouldReparseStreamingJSON.
	parsedArgsLen int
	contentIndex  int
}

// consumeStream folds the SSE body into the output message.
func (p *mistralStreamer) consumeStream(ctx context.Context, body io.Reader) error {
	output := p.output
	reader := NewSSEReader(body)

	// Only one of these is set at a time: the open text or thinking block.
	var currentText *TextContent
	var currentThinking *ThinkingContent
	// Accumulators for the blocks above; see textAccumulator. Reset whenever a
	// new block starts.
	var textAccum, thinkingAccum textAccumulator
	toolCallsByKey := map[string]*mistralStreamingToolCall{}
	// Wire index -> key, so a continuation delta that omits the id still finds
	// the block its first chunk created.
	toolCallKeyByIndex := map[int]string{}
	var toolCallOrder []string

	blockIndex := func() int { return len(output.Content) - 1 }
	syncCurrent := func() {
		if n := len(output.Content); n > 0 {
			switch {
			case currentText != nil:
				output.Content[n-1] = *currentText
			case currentThinking != nil:
				output.Content[n-1] = *currentThinking
			}
		}
	}
	closeCurrent := func() {
		switch {
		case currentText != nil:
			syncCurrent()
			p.push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: blockIndex(), Content: currentText.Text})
			currentText = nil
		case currentThinking != nil:
			syncCurrent()
			p.push(AssistantMessageEvent{Type: EventThinkingEnd, ContentIndex: blockIndex(), Content: currentThinking.Thinking})
			currentThinking = nil
		}
	}
	appendText := func(delta string) {
		if currentText == nil {
			closeCurrent()
			textAccum.reset()
			currentText = &TextContent{Type: "text"}
			output.Content = append(output.Content, *currentText)
			p.push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: blockIndex()})
		}
		currentText.Text = textAccum.add(delta)
		syncCurrent()
		p.push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: blockIndex(), Delta: delta})
	}
	appendThinking := func(delta string) {
		if currentThinking == nil {
			closeCurrent()
			thinkingAccum.reset()
			currentThinking = &ThinkingContent{Type: "thinking"}
			output.Content = append(output.Content, *currentThinking)
			p.push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: blockIndex()})
		}
		currentThinking.Thinking = thinkingAccum.add(delta)
		syncCurrent()
		p.push(AssistantMessageEvent{Type: EventThinkingDelta, ContentIndex: blockIndex(), Delta: delta})
	}

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
		data := strings.TrimSpace(sse.Data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}

		var chunk mistralStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("parsing SSE data: %w", err)
		}

		// Keep the first non-empty id as the response identifier.
		if output.ResponseID == "" {
			output.ResponseID = chunk.ID
		}

		if usage := chunk.Usage; usage != nil {
			cached := 0
			switch {
			case usage.PromptTokensDetails != nil:
				cached = usage.PromptTokensDetails.CachedTokens
			case usage.NumCachedTokens != nil:
				cached = *usage.NumCachedTokens
			}
			cached = min(max(cached, 0), usage.PromptTokens)

			output.Usage.Input = max(0, usage.PromptTokens-cached)
			output.Usage.Output = usage.CompletionTokens
			output.Usage.CacheRead = cached
			output.Usage.CacheWrite = 0
			output.Usage.TotalTokens = usage.TotalTokens
			if output.Usage.TotalTokens == 0 {
				output.Usage.TotalTokens = output.Usage.Input + output.Usage.Output + output.Usage.CacheRead
			}
			CalculateCost(p.model, &output.Usage)
		}

		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		if choice.FinishReason != "" {
			output.RawStopReason = choice.FinishReason
			stopReason, errorMessage := mapMistralStopReason(choice.FinishReason)
			output.StopReason = stopReason
			if errorMessage != "" {
				output.ErrorMessage = errorMessage
			}
		}

		// Content is either a bare string or an array of typed chunks.
		if len(choice.Delta.Content) > 0 && string(choice.Delta.Content) != "null" {
			var text string
			if err := json.Unmarshal(choice.Delta.Content, &text); err == nil {
				if text != "" {
					appendText(sanitizeSurrogates(text))
				}
			} else {
				var items []mistralContentChunk
				if err := json.Unmarshal(choice.Delta.Content, &items); err != nil {
					return fmt.Errorf("parsing delta content: %w", err)
				}
				for _, item := range items {
					switch item.Type {
					case "thinking":
						var parts []string
						for _, part := range item.Thinking {
							if part.Text != "" {
								parts = append(parts, part.Text)
							}
						}
						if delta := sanitizeSurrogates(strings.Join(parts, "")); delta != "" {
							appendThinking(delta)
						}
					case "text":
						appendText(sanitizeSurrogates(item.Text))
					}
				}
			}
		}

		for _, raw := range choice.Delta.ToolCalls {
			closeCurrent()

			index := 0
			if raw.Index != nil {
				index = *raw.Index
			}
			// Continuation deltas carry only the index; the id appears once, on
			// the first chunk. Deriving a synthetic id from the index produced
			// a different key from the id-bearing block, so every subsequent
			// chunk forked a second tool call that carried the arguments while
			// the original kept the name -- and the model's single request
			// reached the agent as two broken ones.
			key, known := toolCallKeyByIndex[index]
			callID := raw.ID
			if !known {
				if callID == "" || callID == "null" {
					callID = deriveMistralToolCallID(fmt.Sprintf("toolcall:%d", index), 0)
				}
				key = fmt.Sprintf("%s:%d", callID, index)
				toolCallKeyByIndex[index] = key
			}

			block, exists := toolCallsByKey[key]
			if !exists {
				block = &mistralStreamingToolCall{
					ToolCall: ToolCall{
						Type:      "toolCall",
						ID:        callID,
						Name:      raw.Function.Name,
						Arguments: map[string]any{},
					},
				}
				output.Content = append(output.Content, block.ToolCall)
				block.contentIndex = len(output.Content) - 1
				toolCallsByKey[key] = block
				toolCallOrder = append(toolCallOrder, key)
				p.push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: block.contentIndex})
			}

			// Arguments arrive as a JSON string, or occasionally as an object.
			argsDelta := ""
			if len(raw.Function.Arguments) > 0 && string(raw.Function.Arguments) != "null" {
				var asString string
				if err := json.Unmarshal(raw.Function.Arguments, &asString); err == nil {
					argsDelta = asString
				} else {
					argsDelta = string(raw.Function.Arguments)
				}
			}
			block.partialArgs.WriteString(argsDelta)
			if shouldReparseStreamingJSON(block.parsedArgsLen, block.partialArgs.Len()) {
				block.ToolCall.Arguments = ParseStreamingJSON(block.partialArgs.String())
				block.parsedArgsLen = block.partialArgs.Len()
			}
			output.Content[block.contentIndex] = block.ToolCall
			p.push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: block.contentIndex, Delta: argsDelta})
		}
	}

	closeCurrent()

	// Finalize tool calls in the order they first appeared.
	for _, key := range toolCallOrder {
		block := toolCallsByKey[key]
		block.ToolCall.Arguments = ParseStreamingJSON(block.partialArgs.String())
		output.Content[block.contentIndex] = block.ToolCall
		tc := block.ToolCall
		p.push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: block.contentIndex, ToolCall: &tc})
	}
	return nil
}

// mapMistralStopReason maps a Mistral finish reason onto a pi stop reason.
func mapMistralStopReason(reason string) (StopReason, string) {
	switch reason {
	case "", "stop":
		return StopStop, ""
	case "length", "model_length":
		return StopLength, ""
	case "tool_calls":
		return StopToolUse, ""
	default:
		return StopError, "Provider stopped with: " + reason
	}
}

// doMistralRequest performs one POST {baseUrl}/v1/chat/completions.
func doMistralRequest(ctx context.Context, model *Model, options *MistralOptions, params map[string]any) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := strings.TrimSuffix(model.BaseURL, "/") + "/v1/chat/completions"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+options.APIKey)

	headers := map[string]string{}
	for k, v := range model.Headers {
		headers[k] = v
	}
	hasAffinity := hasMistralHeader(model.Headers, options.Headers, "x-affinity")
	for k, v := range options.Headers {
		if v == nil {
			deleteHeaderCaseInsensitive(headers, k)
			req.Header.Del(k)
			continue
		}
		headers[k] = *v
	}
	// Session affinity routes follow-up requests to the same cache.
	if mistralShouldCachePrompt(options) && !hasAffinity {
		headers["x-affinity"] = options.SessionID
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	timeout := mistralDefaultTimeout
	if options.TimeoutMs > 0 {
		timeout = time.Duration(options.TimeoutMs) * time.Millisecond
	}
	// The bound goes on the transport, not on the client.
	//
	// http.Client.Timeout covers the whole exchange including reading the
	// response body, and this is a streaming request: a model that took longer
	// than the timeout to finish answering had its stream cut off mid-response
	// with "context deadline exceeded (Client.Timeout ... while reading body)",
	// the partial answer discarded, and the identical request retried -- each
	// attempt paying for the input again and dying at the same point.
	// ResponseHeaderTimeout bounds time-to-first-byte instead, so a slow or
	// unreachable endpoint still fails fast while a long answer streams to
	// completion. Cancellation remains the request context's job.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.ResponseHeaderTimeout = timeout
	client := &http.Client{Transport: transport}

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
			Message: fmt.Sprintf("Mistral API error (%d)", resp.StatusCode),
		}
	}
	return resp, nil
}

// hasMistralHeader reports whether either header map sets name (case-insensitive).
func hasMistralHeader(modelHeaders map[string]string, optionHeaders ProviderHeaders, name string) bool {
	for key := range modelHeaders {
		if strings.EqualFold(key, name) {
			return true
		}
	}
	for key := range optionHeaders {
		if strings.EqualFold(key, name) {
			return true
		}
	}
	return false
}
