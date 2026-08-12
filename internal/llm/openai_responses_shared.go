package llm

// Go port of reference/pi/packages/ai/src/api/openai-responses-shared.ts.
//
// The Responses API models a turn as an ordered list of output *items*
// (reasoning, message, function_call, custom_tool_call) rather than a single
// message with deltas. Streaming events carry an output_index identifying the
// item they belong to, so this port keeps a slot table keyed by that index and
// maps each item onto a pi content block.
//
// Deferred from the TS original (not needed by the wire protocols ported so
// far): web-search / file-search / computer-use output items.

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// ---------------------------------------------------------------------------
// Text signatures
// ---------------------------------------------------------------------------

// textSignatureV1 persists the Responses item id (and optional phase) on a
// text block so a later turn can replay the item verbatim.
type textSignatureV1 struct {
	V     int    `json:"v"`
	ID    string `json:"id"`
	Phase string `json:"phase,omitempty"`
}

func encodeTextSignatureV1(id, phase string) string {
	payload := textSignatureV1{V: 1, ID: id, Phase: phase}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return ""
	}
	return string(encoded)
}

// parseTextSignature decodes a text signature, falling back to treating the
// whole value as a bare item id (the legacy plain-string format).
func parseTextSignature(signature string) (id string, phase string, ok bool) {
	if signature == "" {
		return "", "", false
	}
	if strings.HasPrefix(signature, "{") {
		var parsed textSignatureV1
		if err := json.Unmarshal([]byte(signature), &parsed); err == nil && parsed.V == 1 && parsed.ID != "" {
			if parsed.Phase == "commentary" || parsed.Phase == "final_answer" {
				return parsed.ID, parsed.Phase, true
			}
			return parsed.ID, "", true
		}
	}
	return signature, "", true
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

// ResponsesStreamOptions configures processResponsesStream.
type ResponsesStreamOptions struct {
	ServiceTier string
	// GrammarToolInputProperties maps tool name -> grammar input property.
	GrammarToolInputProperties map[string]string
	// ResolveServiceTier reconciles the tier echoed by the response with the
	// requested one. nil means "response tier, else requested".
	ResolveServiceTier func(responseServiceTier, requestServiceTier string) string
	// ApplyServiceTierPricing adjusts usage cost for the effective tier.
	ApplyServiceTierPricing func(usage *Usage, serviceTier string)
}

// ConvertResponsesMessagesOptions configures ConvertResponsesMessages.
type ConvertResponsesMessagesOptions struct {
	// IncludeSystemPrompt defaults to true when nil.
	IncludeSystemPrompt *bool
	// SupportsDeveloperRole selects "developer" over "system" for reasoning
	// models. Defaults to true when nil.
	SupportsDeveloperRole      *bool
	GrammarToolInputProperties map[string]string
	// DeferredTools are tools loaded mid-transcript rather than sent upfront.
	DeferredTools map[string]Tool
	// DeferredToolsMode is "additional-tools" or "tool-search".
	DeferredToolsMode string
	ToolOptions       ConvertResponsesToolsOptions
}

// ConvertResponsesToolsOptions configures ConvertResponsesTools.
type ConvertResponsesToolsOptions struct {
	// Strict is the default `strict` value; nil means false.
	Strict *bool
	// SupportsStrictMode defaults to true when nil.
	SupportsStrictMode         *bool
	SupportsOpenAIGrammarTools bool
	DeferLoading               bool
}

func boolOr(v *bool, def bool) bool {
	if v != nil {
		return *v
	}
	return def
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

var responsesIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)
var responsesTrailingUnderscores = regexp.MustCompile(`_+$`)

// normalizeResponsesIDPart sanitizes one id segment to the character set and
// length the Responses API accepts.
func normalizeResponsesIDPart(part string) string {
	sanitized := responsesIDChars.ReplaceAllString(part, "_")
	if len(sanitized) > 64 {
		sanitized = sanitized[:64]
	}
	return responsesTrailingUnderscores.ReplaceAllString(sanitized, "")
}

// buildForeignResponsesItemID derives a stable fc_* item id for a tool call
// that came from another provider, whose item ids are meaningless here.
func buildForeignResponsesItemID(itemID string) string {
	normalized := "fc_" + ShortHash(itemID)
	if len(normalized) > 64 {
		normalized = normalized[:64]
	}
	return normalized
}

// responsesNormalizeToolCallID builds the TransformMessages normalizer. pi
// encodes Responses tool calls as "callID|itemID"; only providers that speak
// the real Responses protocol keep the item id, and it must look like fc_*.
func responsesNormalizeToolCallID(allowedToolCallProviders map[string]bool) func(string, *Model, *AssistantMessage) string {
	return func(id string, model *Model, source *AssistantMessage) string {
		if !allowedToolCallProviders[model.Provider] || !strings.Contains(id, "|") {
			return normalizeResponsesIDPart(id)
		}
		callID, itemID, _ := strings.Cut(id, "|")
		normalizedCallID := normalizeResponsesIDPart(callID)

		isForeignToolCall := source == nil || source.Provider != model.Provider || source.API != model.API
		var normalizedItemID string
		if isForeignToolCall {
			normalizedItemID = buildForeignResponsesItemID(itemID)
		} else {
			normalizedItemID = normalizeResponsesIDPart(itemID)
		}
		if !strings.HasPrefix(normalizedItemID, "fc_") {
			normalizedItemID = normalizeResponsesIDPart("fc_" + normalizedItemID)
		}
		return normalizedCallID + "|" + normalizedItemID
	}
}

// convertResponsesToolResultOutput renders tool result content as either a
// plain string or a content array (only when the model accepts images).
func convertResponsesToolResultOutput(model *Model, content []ContentBlock) any {
	var texts []string
	var images []ImageContent
	for _, block := range content {
		switch b := block.(type) {
		case TextContent:
			texts = append(texts, b.Text)
		case ImageContent:
			images = append(images, b)
		}
	}
	textResult := strings.Join(texts, "\n")
	hasText := textResult != ""

	if len(images) == 0 || !model.SupportsImages() {
		switch {
		case hasText:
			return sanitizeSurrogates(textResult)
		case len(images) > 0:
			return "(see attached image)"
		default:
			return "(no tool output)"
		}
	}

	output := []any{}
	if hasText {
		output = append(output, map[string]any{"type": "input_text", "text": sanitizeSurrogates(textResult)})
	}
	for _, image := range images {
		output = append(output, map[string]any{
			"type":      "input_image",
			"detail":    "auto",
			"image_url": fmt.Sprintf("data:%s;base64,%s", image.MimeType, image.Data),
		})
	}
	return output
}

// ConvertResponsesMessages converts a pi context into the Responses `input`
// array. allowedToolCallProviders lists providers whose tool call ids carry a
// meaningful Responses item id.
func ConvertResponsesMessages(model *Model, conv *Context, allowedToolCallProviders map[string]bool, options *ConvertResponsesMessagesOptions) ([]any, error) {
	if options == nil {
		options = &ConvertResponsesMessagesOptions{}
	}
	messages := []any{}
	loadedToolNames := map[string]bool{}

	transformed := TransformMessages(conv.Messages, model, responsesNormalizeToolCallID(allowedToolCallProviders))

	if boolOr(options.IncludeSystemPrompt, true) && conv.SystemPrompt != "" {
		role := "system"
		if model.Reasoning && boolOr(options.SupportsDeveloperRole, true) {
			role = "developer"
		}
		messages = append(messages, map[string]any{
			"role":    role,
			"content": sanitizeSurrogates(conv.SystemPrompt),
		})
	}

	msgIndex := 0
	for _, msg := range transformed {
		switch m := derefResponsesMessage(msg).(type) {
		case UserMessage:
			if text, isString := m.StringContent(); isString {
				messages = append(messages, map[string]any{
					"role":    "user",
					"content": []any{map[string]any{"type": "input_text", "text": sanitizeSurrogates(text)}},
				})
				break
			}
			content := []any{}
			for _, block := range m.BlockContent() {
				switch b := block.(type) {
				case TextContent:
					content = append(content, map[string]any{"type": "input_text", "text": sanitizeSurrogates(b.Text)})
				case ImageContent:
					content = append(content, map[string]any{
						"type":      "input_image",
						"detail":    "auto",
						"image_url": fmt.Sprintf("data:%s;base64,%s", b.MimeType, b.Data),
					})
				}
			}
			if len(content) == 0 {
				continue
			}
			messages = append(messages, map[string]any{"role": "user", "content": content})

		case AssistantMessage:
			output, err := convertResponsesAssistantMessage(model, &m, msgIndex, options)
			if err != nil {
				return nil, err
			}
			if len(output) == 0 {
				continue
			}
			messages = append(messages, output...)

		case ToolResultMessage:
			callID, _, _ := strings.Cut(m.ToolCallID, "|")
			output := convertResponsesToolResultOutput(model, m.Content)

			itemType := "function_call_output"
			if _, isGrammarTool := options.GrammarToolInputProperties[m.ToolName]; isGrammarTool {
				itemType = "custom_tool_call_output"
			}
			messages = append(messages, map[string]any{
				"type":    itemType,
				"call_id": callID,
				"output":  output,
			})

			// Tools that became available because of this result are injected
			// here, so the model sees them from this point onward only.
			var deferredTools []Tool
			for _, name := range m.AddedToolNames {
				tool, isDeferred := options.DeferredTools[name]
				if !isDeferred || loadedToolNames[name] {
					continue
				}
				loadedToolNames[name] = true
				deferredTools = append(deferredTools, tool)
			}
			if len(deferredTools) == 0 {
				break
			}
			switch options.DeferredToolsMode {
			case "additional-tools":
				tools, err := ConvertResponsesTools(deferredTools, &options.ToolOptions)
				if err != nil {
					return nil, err
				}
				messages = append(messages, map[string]any{
					"type":  "additional_tools",
					"role":  "developer",
					"tools": tools,
				})
			case "tool-search":
				names := make([]string, 0, len(deferredTools))
				for _, tool := range deferredTools {
					names = append(names, tool.Name)
				}
				searchCallID := "pi_tool_load_" + ShortHash(m.ToolCallID+":"+strings.Join(names, ","))
				messages = append(messages, map[string]any{
					"type":      "tool_search_call",
					"call_id":   searchCallID,
					"execution": "client",
					"status":    "completed",
					"arguments": map[string]any{"query": strings.Join(names, " "), "limit": len(names)},
				})
				deferLoadingOptions := options.ToolOptions
				deferLoadingOptions.DeferLoading = true
				tools, err := ConvertResponsesTools(deferredTools, &deferLoadingOptions)
				if err != nil {
					return nil, err
				}
				messages = append(messages, map[string]any{
					"type":      "tool_search_output",
					"call_id":   searchCallID,
					"execution": "client",
					"status":    "completed",
					"tools":     tools,
				})
			}
		}
		msgIndex++
	}

	return messages, nil
}

// derefResponsesMessage normalizes pointer messages to values so the type
// switches above only need value cases.
func derefResponsesMessage(msg Message) Message {
	switch m := msg.(type) {
	case *UserMessage:
		return *m
	case *AssistantMessage:
		return *m
	case *ToolResultMessage:
		return *m
	}
	return msg
}

// convertResponsesAssistantMessage replays one assistant message as Responses
// output items.
func convertResponsesAssistantMessage(model *Model, assistant *AssistantMessage, msgIndex int, options *ConvertResponsesMessagesOptions) ([]any, error) {
	output := []any{}
	isSameProviderAndAPI := assistant.Provider == model.Provider && assistant.API == model.API
	isSameModel := isSameProviderAndAPI && assistant.Model == model.ID
	isDifferentModel := isSameProviderAndAPI && assistant.Model != model.ID
	textBlockIndex := 0

	for _, block := range assistant.Content {
		switch b := block.(type) {
		case ThinkingContent:
			// The signature is the serialized reasoning item; without it there
			// is nothing the API will accept.
			if b.ThinkingSignature == "" {
				continue
			}
			var reasoningItem map[string]any
			if err := json.Unmarshal([]byte(b.ThinkingSignature), &reasoningItem); err != nil {
				return nil, fmt.Errorf("invalid reasoning item signature: %w", err)
			}
			output = append(output, reasoningItem)

		case TextContent:
			id, phase, _ := parseTextSignature(b.TextSignature)
			fallbackMessageID := fmt.Sprintf("msg_pi_%d", msgIndex)
			if textBlockIndex > 0 {
				fallbackMessageID = fmt.Sprintf("msg_pi_%d_%d", msgIndex, textBlockIndex)
			}
			textBlockIndex++
			switch {
			case id == "":
				id = fallbackMessageID
			case len(id) > 64:
				id = "msg_" + ShortHash(id)
			}
			item := map[string]any{
				"type":    "message",
				"role":    "assistant",
				"content": []any{map[string]any{"type": "output_text", "text": sanitizeSurrogates(b.Text), "annotations": []any{}}},
				"status":  "completed",
				"id":      id,
			}
			if phase != "" {
				item["phase"] = phase
			}
			output = append(output, item)

		case ToolCall:
			callID, itemIDRaw, _ := strings.Cut(b.ID, "|")
			customInputProperty, isGrammarTool := options.GrammarToolInputProperties[b.Name]
			itemID := itemIDRaw
			hasFCPrefix := strings.HasPrefix(itemID, "fc_")

			// Dropping the item id avoids OpenAI's reasoning/function-call
			// pairing validation for messages from a different model, and is
			// required when replaying a custom-tool call (ctc_*) as a
			// function_call, whose item ids must be fc_*.
			if (isDifferentModel && hasFCPrefix) || (!isGrammarTool && !hasFCPrefix) {
				itemID = ""
			}

			// A namespace only round-trips when the target model is the one
			// that produced it, or the tool is being (re)loaded now.
			_, isDeferred := options.DeferredTools[b.Name]
			canReplayNamespace := isSameModel || isDeferred

			item := map[string]any{"call_id": callID, "name": b.Name}
			if itemID != "" {
				item["id"] = itemID
			} else {
				item["id"] = nil
			}
			if canReplayNamespace && b.Namespace != "" {
				item["namespace"] = b.Namespace
			}

			if isGrammarTool {
				input, err := GetGrammarToolInput(b.Name, b.Arguments, customInputProperty)
				if err != nil {
					return nil, err
				}
				item["type"] = "custom_tool_call"
				item["input"] = sanitizeSurrogates(input)
			} else {
				arguments := b.Arguments
				if arguments == nil {
					arguments = map[string]any{}
				}
				encoded, err := json.Marshal(arguments)
				if err != nil {
					return nil, fmt.Errorf("marshal tool call arguments: %w", err)
				}
				item["type"] = "function_call"
				item["arguments"] = string(encoded)
			}
			output = append(output, item)
		}
	}
	return output, nil
}

// ---------------------------------------------------------------------------
// Tool conversion
// ---------------------------------------------------------------------------

// ConvertResponsesTools serializes tools for the Responses `tools` field.
func ConvertResponsesTools(tools []Tool, options *ConvertResponsesToolsOptions) ([]any, error) {
	if options == nil {
		options = &ConvertResponsesToolsOptions{}
	}
	defaultStrict := boolOr(options.Strict, false)
	supportsStrictMode := boolOr(options.SupportsStrictMode, true)

	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		grammar, isGrammar, err := ResolveGrammarConstrainedSampling(tool, options.SupportsOpenAIGrammarTools)
		if err != nil {
			return nil, err
		}
		if isGrammar {
			item := map[string]any{
				"type":        "custom",
				"name":        tool.Name,
				"description": tool.Description,
				"format": map[string]any{
					"type":       "grammar",
					"syntax":     grammar.Format,
					"definition": grammar.Definition,
				},
			}
			if options.DeferLoading {
				item["defer_loading"] = true
			}
			out = append(out, item)
			continue
		}

		constrainedStrict, hasConstrainedStrict, err := ResolveJSONSchemaStrictSampling(tool, supportsStrictMode)
		if err != nil {
			return nil, err
		}
		var parameters any
		if len(tool.Parameters) > 0 {
			parameters = json.RawMessage(tool.Parameters)
		}
		item := map[string]any{
			"type":        "function",
			"name":        tool.Name,
			"description": tool.Description,
			"parameters":  parameters,
		}
		if options.DeferLoading {
			item["defer_loading"] = true
		}
		if supportsStrictMode {
			if hasConstrainedStrict {
				item["strict"] = constrainedStrict
			} else {
				item["strict"] = defaultStrict
			}
		}
		out = append(out, item)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Stream processing
// ---------------------------------------------------------------------------

// responsesStreamingToolCall is a tool call being assembled from deltas.
type responsesStreamingToolCall struct {
	ToolCall
	// partialJSON accumulates function_call argument deltas. Nil for custom
	// tool calls, which carry a customInput buffer instead.
	partialJSON    *string
	customProperty string
	customBuffer   *GrammarToolInputJSONBuffer
}

func (t *responsesStreamingToolCall) content() ContentBlock { return t.ToolCall }

// responsesSlot links one Responses output item to a pi content block.
type responsesSlot struct {
	kind         string // "thinking" | "text" | "toolCall"
	thinking     *ThinkingContent
	text         *TextContent
	toolCall     *responsesStreamingToolCall
	contentIndex int
}

// responsesOutputItem is a Responses output item as it appears on the wire.
type responsesOutputItem struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Status    string `json:"status,omitempty"`
	Phase     string `json:"phase,omitempty"`
	Namespace string `json:"namespace,omitempty"`

	// reasoning
	Summary []struct {
		Text string `json:"text"`
	} `json:"summary,omitempty"`
	Content          json.RawMessage `json:"content,omitempty"`
	EncryptedContent string          `json:"encrypted_content,omitempty"`

	// function_call / custom_tool_call
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Input     string `json:"input,omitempty"`

	// raw preserves every field so reasoning items can be replayed verbatim.
	raw json.RawMessage
}

func (i *responsesOutputItem) UnmarshalJSON(data []byte) error {
	type plain responsesOutputItem
	var p plain
	if err := json.Unmarshal(data, &p); err != nil {
		return err
	}
	*i = responsesOutputItem(p)
	i.raw = append(json.RawMessage(nil), data...)
	return nil
}

// hasNamespace reports whether the item explicitly carried a namespace.
func (i *responsesOutputItem) hasNamespace() bool {
	if len(i.raw) == 0 {
		return false
	}
	var probe struct {
		Namespace *string `json:"namespace"`
	}
	if err := json.Unmarshal(i.raw, &probe); err != nil {
		return false
	}
	return probe.Namespace != nil
}

// messageContentText renders a message item's content, using the refusal text
// when a part is a refusal rather than output text.
func (i *responsesOutputItem) messageContentText() string {
	if len(i.Content) == 0 {
		return ""
	}
	var parts []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Refusal string `json:"refusal"`
	}
	if err := json.Unmarshal(i.Content, &parts); err != nil {
		return ""
	}
	var sb strings.Builder
	for _, part := range parts {
		if part.Type == "output_text" {
			sb.WriteString(part.Text)
		} else {
			sb.WriteString(part.Refusal)
		}
	}
	return sb.String()
}

// reasoningContentText joins the text parts of a reasoning item's content.
func (i *responsesOutputItem) reasoningContentText() string {
	if len(i.Content) == 0 {
		return ""
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(i.Content, &parts); err != nil {
		return ""
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, "\n\n")
}

func (i *responsesOutputItem) summaryText() string {
	texts := make([]string, 0, len(i.Summary))
	for _, part := range i.Summary {
		texts = append(texts, part.Text)
	}
	return strings.Join(texts, "\n\n")
}

// responsesStreamEvent is one Responses SSE event.
type responsesStreamEvent struct {
	Type        string               `json:"type"`
	OutputIndex int                  `json:"output_index"`
	Delta       string               `json:"delta,omitempty"`
	Arguments   string               `json:"arguments,omitempty"`
	Input       string               `json:"input,omitempty"`
	Item        *responsesOutputItem `json:"item,omitempty"`
	Response    *responsesResponse   `json:"response,omitempty"`

	// error events
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// responsesResponse is the response envelope on terminal events.
type responsesResponse struct {
	ID          string                `json:"id"`
	Status      string                `json:"status"`
	ServiceTier string                `json:"service_tier,omitempty"`
	Output      []responsesOutputItem `json:"output,omitempty"`
	Usage       *responsesUsage       `json:"usage,omitempty"`

	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type responsesUsage struct {
	InputTokens       int `json:"input_tokens"`
	OutputTokens      int `json:"output_tokens"`
	TotalTokens       int `json:"total_tokens"`
	InputTokenDetails *struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokenDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details,omitempty"`
}

// responsesStreamProcessor assembles an AssistantMessage from a Responses
// event stream. pusher is the protocol streamer that owns event emission.
type responsesStreamProcessor struct {
	model   *Model
	output  *AssistantMessage
	options *ResponsesStreamOptions
	push    func(AssistantMessageEvent)

	slots  map[int]*responsesSlot
	blocks []streamingBlock
	// reasoningBlocksByID lets response.completed backfill encrypted content.
	reasoningBlocksByID map[string]*ThinkingContent

	sawTerminalResponseEvent bool
}

func newResponsesStreamProcessor(model *Model, output *AssistantMessage, options *ResponsesStreamOptions, push func(AssistantMessageEvent)) *responsesStreamProcessor {
	if options == nil {
		options = &ResponsesStreamOptions{}
	}
	return &responsesStreamProcessor{
		model:               model,
		output:              output,
		options:             options,
		push:                push,
		slots:               map[int]*responsesSlot{},
		reasoningBlocksByID: map[string]*ThinkingContent{},
	}
}

// syncContent rebuilds output.Content from the in-flight blocks.
func (p *responsesStreamProcessor) syncContent() {
	p.output.Content = make([]ContentBlock, len(p.blocks))
	for i, b := range p.blocks {
		p.output.Content[i] = b.content()
	}
}

func (p *responsesStreamProcessor) getSlot(outputIndex int, kind string) *responsesSlot {
	slot := p.slots[outputIndex]
	if slot == nil || slot.kind != kind {
		return nil
	}
	return slot
}

// createSlot maps a newly started output item onto a content block.
func (p *responsesStreamProcessor) createSlot(outputIndex int, item *responsesOutputItem) *responsesSlot {
	switch item.Type {
	case "reasoning":
		block := &ThinkingContent{Type: "thinking"}
		p.blocks = append(p.blocks, block)
		slot := &responsesSlot{kind: "thinking", thinking: block, contentIndex: len(p.blocks) - 1}
		p.slots[outputIndex] = slot
		p.syncContent()
		p.push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: slot.contentIndex})
		return slot

	case "message":
		p.applyMessagePhaseStopReason(item)
		block := &TextContent{Type: "text"}
		p.blocks = append(p.blocks, block)
		slot := &responsesSlot{kind: "text", text: block, contentIndex: len(p.blocks) - 1}
		p.slots[outputIndex] = slot
		p.syncContent()
		p.push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: slot.contentIndex})
		return slot

	case "function_call":
		partialJSON := item.Arguments
		block := &responsesStreamingToolCall{
			ToolCall: ToolCall{
				Type:      "toolCall",
				ID:        item.CallID + "|" + item.ID,
				Name:      item.Name,
				Arguments: map[string]any{},
				Namespace: item.Namespace,
			},
			partialJSON: &partialJSON,
		}
		p.blocks = append(p.blocks, block)
		slot := &responsesSlot{kind: "toolCall", toolCall: block, contentIndex: len(p.blocks) - 1}
		p.slots[outputIndex] = slot
		p.syncContent()
		p.push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: slot.contentIndex})
		return slot

	case "custom_tool_call":
		inputProperty := "input"
		if property, ok := p.options.GrammarToolInputProperties[item.Name]; ok {
			inputProperty = property
		}
		block := &responsesStreamingToolCall{
			ToolCall: ToolCall{
				Type:      "toolCall",
				ID:        item.CallID + "|" + item.ID,
				Name:      item.Name,
				Arguments: map[string]any{inputProperty: item.Input},
				Namespace: item.Namespace,
			},
			customProperty: inputProperty,
			customBuffer:   &GrammarToolInputJSONBuffer{},
		}
		p.blocks = append(p.blocks, block)
		slot := &responsesSlot{kind: "toolCall", toolCall: block, contentIndex: len(p.blocks) - 1}
		p.slots[outputIndex] = slot
		p.syncContent()
		p.push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: slot.contentIndex})
		return slot
	}
	return nil
}

func (p *responsesStreamProcessor) getOrCreateSlot(outputIndex int, item *responsesOutputItem) *responsesSlot {
	if slot := p.slots[outputIndex]; slot != nil {
		return slot
	}
	return p.createSlot(outputIndex, item)
}

// applyMessagePhaseStopReason treats a final_answer message as a stop signal.
func (p *responsesStreamProcessor) applyMessagePhaseStopReason(item *responsesOutputItem) {
	if item.Type == "message" && item.Phase == "final_answer" {
		p.output.StopReason = StopStop
	}
}

func (p *responsesStreamProcessor) pushThinkingDelta(slot *responsesSlot, delta string) {
	slot.thinking.Thinking += delta
	p.syncContent()
	p.push(AssistantMessageEvent{Type: EventThinkingDelta, ContentIndex: slot.contentIndex, Delta: delta})
}

func (p *responsesStreamProcessor) pushTextDelta(slot *responsesSlot, delta string) {
	slot.text.Text += delta
	p.syncContent()
	p.push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: slot.contentIndex, Delta: delta})
}

func (p *responsesStreamProcessor) pushToolCallDelta(slot *responsesSlot, delta string) {
	if delta == "" {
		return
	}
	p.syncContent()
	p.push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: slot.contentIndex, Delta: delta})
}

// customToolCallInput reads the current input string off a custom tool call.
func customToolCallInput(block *responsesStreamingToolCall) string {
	if block.customProperty == "" {
		return ""
	}
	value, _ := block.Arguments[block.customProperty].(string)
	return value
}

// appendCustomToolCallInput advances the grammar input buffer and returns the
// JSON delta to emit.
func appendCustomToolCallInput(block *responsesStreamingToolCall, nextInput string, close bool) (string, error) {
	if block.customBuffer == nil {
		return "", nil
	}
	delta, err := AppendGrammarToolInputJSONDelta(block.customBuffer, block.customProperty, nextInput, close)
	if err != nil {
		return "", err
	}
	block.Arguments = map[string]any{block.customProperty: nextInput}
	return delta, nil
}

// ProcessEvent folds one Responses stream event into the output message.
func (p *responsesStreamProcessor) ProcessEvent(event *responsesStreamEvent) error {
	switch event.Type {
	case "response.created":
		if event.Response != nil {
			p.output.ResponseID = event.Response.ID
		}

	case "response.output_item.added":
		if event.Item != nil {
			p.createSlot(event.OutputIndex, event.Item)
		}

	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		if slot := p.getSlot(event.OutputIndex, "thinking"); slot != nil {
			p.pushThinkingDelta(slot, event.Delta)
		}

	case "response.reasoning_summary_part.done":
		// Parts are separated by a blank line in the assembled thinking text.
		if slot := p.getSlot(event.OutputIndex, "thinking"); slot != nil {
			p.pushThinkingDelta(slot, "\n\n")
		}

	case "response.output_text.delta", "response.refusal.delta":
		if slot := p.getSlot(event.OutputIndex, "text"); slot != nil {
			p.pushTextDelta(slot, event.Delta)
		}

	case "response.function_call_arguments.delta":
		slot := p.getSlot(event.OutputIndex, "toolCall")
		if slot == nil || slot.toolCall.partialJSON == nil {
			return nil
		}
		*slot.toolCall.partialJSON += event.Delta
		slot.toolCall.Arguments = ParseStreamingJSON(*slot.toolCall.partialJSON)
		p.pushToolCallDelta(slot, event.Delta)

	case "response.function_call_arguments.done":
		slot := p.getSlot(event.OutputIndex, "toolCall")
		if slot == nil || slot.toolCall.partialJSON == nil {
			return nil
		}
		previous := *slot.toolCall.partialJSON
		*slot.toolCall.partialJSON = event.Arguments
		slot.toolCall.Arguments = ParseStreamingJSON(event.Arguments)
		// Emit only the unseen tail, and only when the final value extends
		// what was streamed.
		if strings.HasPrefix(event.Arguments, previous) {
			p.pushToolCallDelta(slot, event.Arguments[len(previous):])
		} else {
			p.syncContent()
		}

	case "response.custom_tool_call_input.delta":
		slot := p.getSlot(event.OutputIndex, "toolCall")
		if slot == nil || slot.toolCall.customBuffer == nil {
			return nil
		}
		delta, err := appendCustomToolCallInput(slot.toolCall, customToolCallInput(slot.toolCall)+event.Delta, false)
		if err != nil {
			return err
		}
		p.pushToolCallDelta(slot, delta)

	case "response.custom_tool_call_input.done":
		slot := p.getSlot(event.OutputIndex, "toolCall")
		if slot == nil || slot.toolCall.customBuffer == nil {
			return nil
		}
		delta, err := appendCustomToolCallInput(slot.toolCall, event.Input, true)
		if err != nil {
			return err
		}
		p.pushToolCallDelta(slot, delta)

	case "response.output_item.done":
		return p.finishItem(event)

	case "response.completed", "response.incomplete":
		if event.Response != nil {
			p.finalizeResponse(event.Response)
		}

	case "error":
		return fmt.Errorf("Error Code %s: %s", event.Code, event.Message)

	case "response.failed":
		p.sawTerminalResponseEvent = true
		if event.Response == nil {
			return fmt.Errorf("Unknown error (no error details in response)")
		}
		p.output.RawStopReason = event.Response.Status
		if err := event.Response.Error; err != nil {
			code, message := err.Code, err.Message
			if code == "" {
				code = "unknown"
			}
			if message == "" {
				message = "no message"
			}
			return fmt.Errorf("%s: %s", code, message)
		}
		if details := event.Response.IncompleteDetails; details != nil && details.Reason != "" {
			return fmt.Errorf("incomplete: %s", details.Reason)
		}
		return fmt.Errorf("Unknown error (no error details in response)")
	}
	return nil
}

// finishItem finalizes the block behind a completed output item.
func (p *responsesStreamProcessor) finishItem(event *responsesStreamEvent) error {
	item := event.Item
	if item == nil {
		return nil
	}
	p.applyMessagePhaseStopReason(item)
	slot := p.getOrCreateSlot(event.OutputIndex, item)
	if slot == nil {
		return nil
	}

	switch {
	case item.Type == "reasoning" && slot.kind == "thinking":
		// The terminal item carries the authoritative text; the signature is
		// the whole item, replayed verbatim on the next turn.
		if text := item.summaryText(); text != "" {
			slot.thinking.Thinking = text
		} else if text := item.reasoningContentText(); text != "" {
			slot.thinking.Thinking = text
		}
		slot.thinking.ThinkingSignature = string(item.raw)
		p.reasoningBlocksByID[item.ID] = slot.thinking
		p.syncContent()
		p.push(AssistantMessageEvent{Type: EventThinkingEnd, ContentIndex: slot.contentIndex, Content: slot.thinking.Thinking})
		delete(p.slots, event.OutputIndex)

	case item.Type == "message" && slot.kind == "text":
		slot.text.Text = item.messageContentText()
		slot.text.TextSignature = encodeTextSignatureV1(item.ID, item.Phase)
		p.syncContent()
		p.push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: slot.contentIndex, Content: slot.text.Text})
		delete(p.slots, event.OutputIndex)

	case item.Type == "function_call" && slot.kind == "toolCall" && slot.toolCall.partialJSON != nil:
		arguments := item.Arguments
		if arguments == "" {
			arguments = *slot.toolCall.partialJSON
		}
		if arguments == "" {
			arguments = "{}"
		}
		slot.toolCall.Arguments = ParseStreamingJSON(arguments)
		if item.hasNamespace() {
			slot.toolCall.Namespace = item.Namespace
		}
		// Drop the scratch buffer so replay carries parsed arguments only.
		slot.toolCall.partialJSON = nil
		p.syncContent()
		tc := slot.toolCall.ToolCall
		p.push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: slot.contentIndex, ToolCall: &tc})
		delete(p.slots, event.OutputIndex)

	case item.Type == "custom_tool_call" && slot.kind == "toolCall" && slot.toolCall.customBuffer != nil:
		input := item.Input
		if input == "" {
			input = customToolCallInput(slot.toolCall)
		}
		delta, err := appendCustomToolCallInput(slot.toolCall, input, true)
		if err != nil {
			return err
		}
		p.pushToolCallDelta(slot, delta)
		if item.hasNamespace() {
			slot.toolCall.Namespace = item.Namespace
		}
		slot.toolCall.customBuffer = nil
		p.syncContent()
		tc := slot.toolCall.ToolCall
		p.push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: slot.contentIndex, ToolCall: &tc})
		delete(p.slots, event.OutputIndex)
	}
	return nil
}

// backfillReasoningSignatures copies encrypted_content from the terminal
// response into stored signatures. Azure omits it from output_item.done, and
// without it stateless (store:false) replay loses the reasoning payload.
func (p *responsesStreamProcessor) backfillReasoningSignatures(output []responsesOutputItem) {
	for i := range output {
		item := &output[i]
		if item.Type != "reasoning" || item.EncryptedContent == "" {
			continue
		}
		block := p.reasoningBlocksByID[item.ID]
		if block == nil || block.ThinkingSignature == "" {
			continue
		}
		var stored map[string]any
		if err := json.Unmarshal([]byte(block.ThinkingSignature), &stored); err != nil {
			continue
		}
		if existing, ok := stored["encrypted_content"].(string); ok && existing != "" {
			continue
		}
		stored["encrypted_content"] = item.EncryptedContent
		if encoded, err := json.Marshal(stored); err == nil {
			block.ThinkingSignature = string(encoded)
		}
	}
	p.syncContent()
}

// finalizeResponse applies usage, cost, and stop reason from a terminal event.
func (p *responsesStreamProcessor) finalizeResponse(response *responsesResponse) {
	p.sawTerminalResponseEvent = true
	p.backfillReasoningSignatures(response.Output)
	if response.ID != "" {
		p.output.ResponseID = response.ID
	}

	if usage := response.Usage; usage != nil {
		cachedTokens, cacheWriteTokens := 0, 0
		if details := usage.InputTokenDetails; details != nil {
			cachedTokens, cacheWriteTokens = details.CachedTokens, details.CacheWriteTokens
		}
		reasoning := 0
		if details := usage.OutputTokenDetails; details != nil {
			reasoning = details.ReasoningTokens
		}
		p.output.Usage = Usage{
			// OpenAI counts cached and cache-write tokens inside input_tokens.
			Input:       max(0, usage.InputTokens-cachedTokens-cacheWriteTokens),
			Output:      usage.OutputTokens,
			CacheRead:   cachedTokens,
			CacheWrite:  cacheWriteTokens,
			Reasoning:   &reasoning,
			TotalTokens: usage.TotalTokens,
		}
	}
	CalculateCost(p.model, &p.output.Usage)

	if p.options.ApplyServiceTierPricing != nil {
		serviceTier := response.ServiceTier
		if p.options.ResolveServiceTier != nil {
			serviceTier = p.options.ResolveServiceTier(response.ServiceTier, p.options.ServiceTier)
		} else if serviceTier == "" {
			serviceTier = p.options.ServiceTier
		}
		p.options.ApplyServiceTierPricing(&p.output.Usage, serviceTier)
	}

	incompleteReason := ""
	if details := response.IncompleteDetails; details != nil {
		incompleteReason = details.Reason
	}
	if incompleteReason != "" {
		p.output.RawStopReason = response.Status + "." + incompleteReason
	} else {
		p.output.RawStopReason = response.Status
	}

	stopReason, errorMessage := mapResponsesStopReason(response.Status, incompleteReason)
	p.output.StopReason = stopReason
	p.output.ErrorMessage = errorMessage

	if p.output.StopReason == StopStop && hasToolCallBlock(p.output.Content) {
		p.output.StopReason = StopToolUse
	}
	p.syncContent()
}

func hasToolCallBlock(content []ContentBlock) bool {
	for _, block := range content {
		if _, ok := block.(ToolCall); ok {
			return true
		}
	}
	return false
}

// mapResponsesStopReason maps a Responses status onto a pi stop reason,
// keeping the provider's incomplete reason so truncation stays distinct from
// content filtering.
func mapResponsesStopReason(status, incompleteReason string) (StopReason, string) {
	switch status {
	case "":
		return StopStop, ""
	case "completed":
		return StopStop, ""
	case "incomplete":
		if incompleteReason == "max_output_tokens" {
			return StopLength, ""
		}
		if incompleteReason != "" {
			return StopError, "Response incomplete: " + incompleteReason
		}
		return StopError, "Response incomplete without a provider reason"
	case "failed", "cancelled":
		return StopError, ""
	case "in_progress", "queued":
		return StopStop, ""
	default:
		return StopError, "Unhandled stop reason: " + status
	}
}
