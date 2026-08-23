package llm

// Go port of reference/pi/packages/ai/src/api/google-shared.ts.
//
// Shared conversion and mapping helpers for the google-generative-ai and
// google-vertex protocols. Where the TS implementation types these against the
// @google/genai SDK, this port builds the same JSON shapes as plain maps.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// GoogleThinkingLevel mirrors Google's ThinkingLevel enum values.
type GoogleThinkingLevel = string

const (
	GoogleThinkingLevelUnspecified GoogleThinkingLevel = "THINKING_LEVEL_UNSPECIFIED"
	GoogleThinkingLevelMinimal     GoogleThinkingLevel = "MINIMAL"
	GoogleThinkingLevelLow         GoogleThinkingLevel = "LOW"
	GoogleThinkingLevelMedium      GoogleThinkingLevel = "MEDIUM"
	GoogleThinkingLevelHigh        GoogleThinkingLevel = "HIGH"
)

// Google function-calling modes (FunctionCallingConfigMode in the SDK).
const (
	googleFunctionCallingAuto      = "AUTO"
	googleFunctionCallingNone      = "NONE"
	googleFunctionCallingAny       = "ANY"
	googleFunctionCallingValidated = "VALIDATED"
)

// googlePart is one Gemini content part on the wire.
//
// Protocol note (thought signatures): `thought: true` is the definitive marker
// for thinking content. `thoughtSignature` is an encrypted representation of
// the model's reasoning used to preserve context across turns, and can appear
// on ANY part type, so it does not itself mark a part as thinking. Parts
// bearing signatures must be replayed as-is; signatures are never merged or
// moved between parts. See https://ai.google.dev/gemini-api/docs/thought-signatures
type googlePart struct {
	// Text is a pointer so an absent field stays distinguishable from an
	// explicit empty string. Gemini attaches signatures to parts whose visible
	// text is empty and requires them echoed back verbatim, and the stream
	// loop keys off "text present" rather than "text non-empty".
	Text             *string `json:"text,omitempty"`
	Thought          bool    `json:"thought,omitempty"`
	ThoughtSignature string  `json:"thoughtSignature,omitempty"`

	InlineData *googleInlineData `json:"inlineData,omitempty"`

	FunctionCall     *googleFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *googleFunctionResponse `json:"functionResponse,omitempty"`
}

type googleInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type googleFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
	ID   string         `json:"id,omitempty"`
}

type googleFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
	Parts    []googlePart   `json:"parts,omitempty"`
	ID       string         `json:"id,omitempty"`
}

// googleContent is one Gemini turn ("user" or "model").
type googleContent struct {
	Role  string       `json:"role"`
	Parts []googlePart `json:"parts"`
}

// googleText returns a pointer suitable for googlePart.Text, so that an
// explicit empty string still serializes as "text": "".
func googleText(text string) *string {
	return &text
}

// isThinkingPart reports whether a streamed part is thinking content.
func isThinkingPart(part *googlePart) bool {
	return part.Thought
}

// retainThoughtSignature preserves the last non-empty signature for the
// current block. Some backends only send thoughtSignature on the first delta
// of a part; later deltas omit it. This never merges signatures across
// distinct parts, it only stops one from being cleared within a block.
func retainThoughtSignature(existing, incoming string) string {
	if incoming != "" {
		return incoming
	}
	return existing
}

// base64SignaturePattern validates that a signature is base64; Google's APIs
// type thought signatures as TYPE_BYTES and reject anything else.
var base64SignaturePattern = regexp.MustCompile(`^[A-Za-z0-9+/]+={0,2}$`)

func isValidThoughtSignature(signature string) bool {
	if signature == "" || len(signature)%4 != 0 {
		return false
	}
	return base64SignaturePattern.MatchString(signature)
}

// resolveThoughtSignature keeps a signature only when it came from the same
// provider and model and is valid base64.
func resolveThoughtSignature(isSameProviderAndModel bool, signature string) string {
	if isSameProviderAndModel && isValidThoughtSignature(signature) {
		return signature
	}
	return ""
}

var geminiMajorVersionRe = regexp.MustCompile(`^gemini(?:-live)?-(\d+)`)

// getGeminiMajorVersion extracts the major version from a Gemini model id.
func getGeminiMajorVersion(modelID string) (int, bool) {
	match := geminiMajorVersionRe.FindStringSubmatch(strings.ToLower(modelID))
	if match == nil {
		return 0, false
	}
	version, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, false
	}
	return version, true
}

// RequiresToolCallID reports whether a model behind a Google API needs explicit
// tool call ids on function calls and responses.
func RequiresToolCallID(modelID string) bool {
	if strings.HasPrefix(modelID, "claude-") || strings.HasPrefix(modelID, "gpt-oss-") {
		return true
	}
	version, ok := getGeminiMajorVersion(modelID)
	return ok && version >= 3
}

// supportsMultimodalFunctionResponse reports whether images may be nested
// inside functionResponse.parts. Gemini 3+ supports it; older Gemini and
// non-Gemini models behind Cloud Code Assist need a separate user image turn.
func supportsMultimodalFunctionResponse(modelID string) bool {
	if version, ok := getGeminiMajorVersion(modelID); ok {
		return version >= 3
	}
	return true
}

var googleToolCallIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// googleNormalizeToolCallID adapts normalizeToolCallId in google-shared.ts to
// the TransformMessages signature. Ids are only rewritten for models that
// require them.
func googleNormalizeToolCallID(id string, model *Model, _ *AssistantMessage) string {
	if !RequiresToolCallID(model.ID) {
		return id
	}
	out := googleToolCallIDChars.ReplaceAllString(id, "_")
	if len(out) > 64 {
		out = out[:64]
	}
	return out
}

// convertGoogleMessages converts a pi context into Gemini contents.
func convertGoogleMessages(model *Model, conv *Context) []googleContent {
	contents := []googleContent{}
	transformed := TransformMessages(conv.Messages, model, googleNormalizeToolCallID)

	for _, msg := range transformed {
		switch m := derefResponsesMessage(msg).(type) {
		case UserMessage:
			if text, isString := m.StringContent(); isString {
				contents = append(contents, googleContent{
					Role:  "user",
					Parts: []googlePart{{Text: googleText(sanitizeSurrogates(text))}},
				})
				continue
			}
			var parts []googlePart
			for _, block := range m.BlockContent() {
				switch b := block.(type) {
				case TextContent:
					parts = append(parts, googlePart{Text: googleText(sanitizeSurrogates(b.Text))})
				case ImageContent:
					parts = append(parts, googlePart{InlineData: &googleInlineData{MimeType: b.MimeType, Data: b.Data}})
				}
			}
			if len(parts) == 0 {
				continue
			}
			contents = append(contents, googleContent{Role: "user", Parts: parts})

		case AssistantMessage:
			parts := convertGoogleAssistantParts(model, &m)
			if len(parts) == 0 {
				continue
			}
			contents = append(contents, googleContent{Role: "model", Parts: parts})

		case ToolResultMessage:
			contents = appendGoogleToolResult(model, contents, &m)
		}
	}
	return contents
}

// convertGoogleAssistantParts replays one assistant message as Gemini parts.
func convertGoogleAssistantParts(model *Model, assistant *AssistantMessage) []googlePart {
	var parts []googlePart
	// Signatures are only meaningful for the exact model that produced them.
	isSameProviderAndModel := assistant.Provider == model.Provider && assistant.Model == model.ID

	for _, block := range assistant.Content {
		switch b := block.(type) {
		case TextContent:
			signature := resolveThoughtSignature(isSameProviderAndModel, b.TextSignature)
			// Gemini can attach a signature to a part with no visible text and
			// requires it echoed back; dropping it breaks the reasoning chain
			// and the model intermittently ends turns with a thought-only STOP.
			if strings.TrimSpace(b.Text) == "" && signature == "" {
				continue
			}
			parts = append(parts, googlePart{Text: googleText(sanitizeSurrogates(b.Text)), ThoughtSignature: signature})

		case ThinkingContent:
			if isSameProviderAndModel {
				signature := resolveThoughtSignature(isSameProviderAndModel, b.ThinkingSignature)
				if strings.TrimSpace(b.Thinking) == "" && signature == "" {
					continue
				}
				parts = append(parts, googlePart{
					Thought:          true,
					Text:             googleText(sanitizeSurrogates(b.Thinking)),
					ThoughtSignature: signature,
				})
				continue
			}
			// Cross-provider/model: the signature is unusable, so the thinking
			// is replayed as plain text (untagged, to avoid the model mimicking
			// tags). Empty blocks stay dropped.
			if strings.TrimSpace(b.Thinking) == "" {
				continue
			}
			parts = append(parts, googlePart{Text: googleText(sanitizeSurrogates(b.Thinking))})

		case ToolCall:
			args := b.Arguments
			if args == nil {
				args = map[string]any{}
			}
			call := &googleFunctionCall{Name: b.Name, Args: args}
			if RequiresToolCallID(model.ID) {
				call.ID = b.ID
			}
			parts = append(parts, googlePart{
				FunctionCall:     call,
				ThoughtSignature: resolveThoughtSignature(isSameProviderAndModel, b.ThoughtSignature),
			})
		}
	}
	return parts
}

// appendGoogleToolResult appends a tool result, merging into a preceding
// function-response user turn when possible.
func appendGoogleToolResult(model *Model, contents []googleContent, msg *ToolResultMessage) []googleContent {
	var texts []string
	var images []ImageContent
	for _, block := range msg.Content {
		switch b := block.(type) {
		case TextContent:
			texts = append(texts, b.Text)
		case ImageContent:
			if model.SupportsImages() {
				images = append(images, b)
			}
		}
	}
	textResult := strings.Join(texts, "\n")

	responseValue := ""
	switch {
	case textResult != "":
		responseValue = sanitizeSurrogates(textResult)
	case len(images) > 0:
		responseValue = "(see attached image)"
	}

	imageParts := make([]googlePart, 0, len(images))
	for _, image := range images {
		imageParts = append(imageParts, googlePart{InlineData: &googleInlineData{MimeType: image.MimeType, Data: image.Data}})
	}

	// "output" for success, "error" for failures, per the SDK docs.
	response := map[string]any{"output": responseValue}
	if msg.IsError {
		response = map[string]any{"error": responseValue}
	}

	multimodal := supportsMultimodalFunctionResponse(model.ID)
	functionResponse := &googleFunctionResponse{Name: msg.ToolName, Response: response}
	if len(imageParts) > 0 && multimodal {
		functionResponse.Parts = imageParts
	}
	if RequiresToolCallID(model.ID) {
		functionResponse.ID = msg.ToolCallID
	}
	part := googlePart{FunctionResponse: functionResponse}

	// Cloud Code Assist requires all function responses in a single user turn.
	if n := len(contents); n > 0 && contents[n-1].Role == "user" && hasGoogleFunctionResponse(contents[n-1].Parts) {
		contents[n-1].Parts = append(contents[n-1].Parts, part)
	} else {
		contents = append(contents, googleContent{Role: "user", Parts: []googlePart{part}})
	}

	// Gemini < 3 cannot nest images, so they follow as their own user turn.
	if len(imageParts) > 0 && !multimodal {
		parts := append([]googlePart{{Text: googleText("Tool result image:")}}, imageParts...)
		contents = append(contents, googleContent{Role: "user", Parts: parts})
	}
	return contents
}

func hasGoogleFunctionResponse(parts []googlePart) bool {
	for _, part := range parts {
		if part.FunctionResponse != nil {
			return true
		}
	}
	return false
}

// jsonSchemaMetaDeclarations are the JSON Schema keywords OpenAPI 3.03 (the
// legacy `parameters` field) rejects.
var jsonSchemaMetaDeclarations = map[string]bool{
	"$schema": true, "$id": true, "$anchor": true, "$dynamicAnchor": true,
	"$vocabulary": true, "$comment": true, "$defs": true,
	// pre-draft-2019-09 equivalent of $defs
	"definitions": true,
}

// sanitizeForOpenAPI recursively strips JSON Schema meta-declarations.
func sanitizeForOpenAPI(schema any) any {
	switch s := schema.(type) {
	case map[string]any:
		result := map[string]any{}
		for key, value := range s {
			if jsonSchemaMetaDeclarations[key] {
				continue
			}
			result[key] = sanitizeForOpenAPI(value)
		}
		return result
	default:
		// Arrays and scalars pass through, matching the TS early return.
		return schema
	}
}

// convertGoogleTools converts tools to Gemini function declarations.
//
// By default this uses `parametersJsonSchema`, which accepts full JSON Schema
// (anyOf, oneOf, const, ...). useParameters selects the legacy `parameters`
// field (OpenAPI 3.03), needed for Cloud Code Assist with Claude models, where
// the API translates `parameters` into Anthropic's `input_schema`.
func convertGoogleTools(tools []Tool, useParameters bool) []any {
	if len(tools) == 0 {
		return nil
	}
	declarations := make([]any, 0, len(tools))
	for _, tool := range tools {
		declaration := map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
		}
		if useParameters {
			var schema any
			if len(tool.Parameters) > 0 {
				if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
					schema = map[string]any{}
				}
			}
			declaration["parameters"] = sanitizeForOpenAPI(schema)
		} else if len(tool.Parameters) > 0 {
			declaration["parametersJsonSchema"] = json.RawMessage(tool.Parameters)
		}
		declarations = append(declarations, declaration)
	}
	return []any{map[string]any{"functionDeclarations": declarations}}
}

// SupportsGoogleStrictToolSampling reports whether the model enforces required
// function parameters in validated tool-calling modes (Gemini 3+).
func SupportsGoogleStrictToolSampling(modelID string) bool {
	version, ok := getGeminiMajorVersion(modelID)
	return ok && version >= 3
}

// mapGoogleToolChoice maps a pi tool choice onto a Gemini function-calling mode.
func mapGoogleToolChoice(choice string) string {
	switch choice {
	case "none":
		return googleFunctionCallingNone
	case "any":
		return googleFunctionCallingAny
	default:
		return googleFunctionCallingAuto
	}
}

// resolveGoogleFunctionCallingMode picks the function-calling mode. ok is false
// when the field should be omitted entirely.
func resolveGoogleFunctionCallingMode(tools []Tool, toolChoice string, supportsStrictMode bool) (mode string, ok bool) {
	useStrictMode := false
	for _, tool := range tools {
		if strict, hasStrict, err := ResolveJSONSchemaStrictSampling(tool, supportsStrictMode); err == nil && hasStrict && strict {
			useStrictMode = true
			break
		}
	}
	if toolChoice == "none" || toolChoice == "any" {
		return mapGoogleToolChoice(toolChoice), true
	}
	if useStrictMode {
		return googleFunctionCallingValidated, true
	}
	if toolChoice != "" {
		return mapGoogleToolChoice(toolChoice), true
	}
	return "", false
}

// mapGoogleStopReason maps a Gemini finishReason onto a pi stop reason. Every
// reason other than STOP and MAX_TOKENS (safety blocks, recitation, malformed
// calls, ...) is an error.
func mapGoogleStopReason(reason string) StopReason {
	switch reason {
	case "STOP":
		return StopStop
	case "MAX_TOKENS":
		return StopLength
	default:
		return StopError
	}
}

// ---------------------------------------------------------------------------
// Stream assembly
// ---------------------------------------------------------------------------

// googleStreamChunk is one GenerateContentResponse from the stream. Vertex uses
// the same response shape as the Generative Language API.
type googleStreamChunk struct {
	ResponseID string `json:"responseId"`
	Candidates []struct {
		Content struct {
			Parts []googlePart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata *struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
		TotalTokenCount         int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// googleToolCallCounter disambiguates generated tool call ids within a process.
var googleToolCallCounter atomic.Int64

// consumeGoogleStream folds an SSE body into output, pushing events through
// push. Shared by the google-generative-ai and google-vertex protocols.
//
// Google streams whole GenerateContentResponse objects rather than typed
// deltas, so text/thinking parts are coalesced into blocks here: consecutive
// parts of the same kind extend the current block, and a kind change (or a
// function call) closes it.
func consumeGoogleStream(ctx context.Context, body io.Reader, model *Model, output *AssistantMessage, push func(AssistantMessageEvent)) error {
	reader := NewSSEReader(body)

	// Only one of these is set at a time: the open text or thinking block.
	var currentText *TextContent
	var currentThinking *ThinkingContent
	// Accumulators for the blocks above; see textAccumulator. Reset whenever a
	// new block starts.
	var textAccum, thinkingAccum textAccumulator

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
			push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: blockIndex(), Content: currentText.Text})
			currentText = nil
		case currentThinking != nil:
			syncCurrent()
			push(AssistantMessageEvent{Type: EventThinkingEnd, ContentIndex: blockIndex(), Content: currentThinking.Thinking})
			currentThinking = nil
		}
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
		if sse.Data == "" || sse.Data == "[DONE]" {
			continue
		}
		var chunk googleStreamChunk
		if err := json.Unmarshal([]byte(sse.Data), &chunk); err != nil {
			return fmt.Errorf("parsing SSE data: %w", err)
		}

		// responseId is output-only; keep the first non-empty one.
		if output.ResponseID == "" {
			output.ResponseID = chunk.ResponseID
		}

		if len(chunk.Candidates) > 0 {
			candidate := chunk.Candidates[0]
			for i := range candidate.Content.Parts {
				part := &candidate.Content.Parts[i]

				// The TS original keys off `part.text !== undefined`, so a part
				// carrying an explicit empty string still opens/extends a block
				// (that is how a bare thought signature arrives).
				if part.Text != nil {
					text := *part.Text
					isThinking := isThinkingPart(part)
					// A kind change closes the open block and opens a new one.
					if (isThinking && currentThinking == nil) || (!isThinking && currentText == nil) {
						closeCurrent()
						if isThinking {
							thinkingAccum.reset()
							currentThinking = &ThinkingContent{Type: "thinking"}
							output.Content = append(output.Content, *currentThinking)
							push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: blockIndex()})
						} else {
							textAccum.reset()
							currentText = &TextContent{Type: "text"}
							output.Content = append(output.Content, *currentText)
							push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: blockIndex()})
						}
					}
					if isThinking {
						currentThinking.Thinking = thinkingAccum.add(text)
						currentThinking.ThinkingSignature = retainThoughtSignature(currentThinking.ThinkingSignature, part.ThoughtSignature)
						syncCurrent()
						push(AssistantMessageEvent{Type: EventThinkingDelta, ContentIndex: blockIndex(), Delta: text})
					} else {
						currentText.Text = textAccum.add(text)
						currentText.TextSignature = retainThoughtSignature(currentText.TextSignature, part.ThoughtSignature)
						syncCurrent()
						push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: blockIndex(), Delta: text})
					}
				}

				if part.FunctionCall != nil {
					closeCurrent()

					// Google may omit the id, or repeat one; both need a fresh
					// unique id so tool results can be matched.
					toolCallID := part.FunctionCall.ID
					if toolCallID == "" || hasGoogleToolCallID(output.Content, toolCallID) {
						toolCallID = fmt.Sprintf("%s_%d_%d", part.FunctionCall.Name, time.Now().UnixMilli(), googleToolCallCounter.Add(1))
					}
					args := part.FunctionCall.Args
					if args == nil {
						args = map[string]any{}
					}
					toolCall := ToolCall{
						Type:             "toolCall",
						ID:               toolCallID,
						Name:             part.FunctionCall.Name,
						Arguments:        args,
						ThoughtSignature: part.ThoughtSignature,
					}
					output.Content = append(output.Content, toolCall)

					// Google delivers complete calls, so start/delta/end are
					// emitted back to back to keep the event contract uniform.
					encoded, err := json.Marshal(toolCall.Arguments)
					if err != nil {
						return fmt.Errorf("marshal tool call arguments: %w", err)
					}
					push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: blockIndex()})
					push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: blockIndex(), Delta: string(encoded)})
					tc := toolCall
					push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: blockIndex(), ToolCall: &tc})
				}
			}

			if candidate.FinishReason != "" {
				output.RawStopReason = candidate.FinishReason
				output.StopReason = mapGoogleStopReason(candidate.FinishReason)
				if hasToolCallBlock(output.Content) {
					output.StopReason = StopToolUse
				}
			}
		}

		if usage := chunk.UsageMetadata; usage != nil {
			reasoning := usage.ThoughtsTokenCount
			output.Usage = Usage{
				// promptTokenCount includes cached tokens; thoughts are billed
				// as output.
				Input:       usage.PromptTokenCount - usage.CachedContentTokenCount,
				Output:      usage.CandidatesTokenCount + usage.ThoughtsTokenCount,
				CacheRead:   usage.CachedContentTokenCount,
				CacheWrite:  0,
				Reasoning:   &reasoning,
				TotalTokens: usage.TotalTokenCount,
			}
			CalculateCost(model, &output.Usage)
		}
	}

	closeCurrent()
	return nil
}

func hasGoogleToolCallID(content []ContentBlock, id string) bool {
	for _, block := range content {
		if tc, ok := block.(ToolCall); ok && tc.ID == id {
			return true
		}
	}
	return false
}
