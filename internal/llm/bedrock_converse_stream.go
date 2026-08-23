package llm

// Go port of reference/pi/packages/ai/src/api/bedrock-converse-stream.ts.
//
// Where the TS implementation drives @aws-sdk/client-bedrock-runtime, this port
// posts to the ConverseStream REST endpoint directly, signing with SigV4
// (aws_sigv4.go) and decoding the binary event stream (aws_eventstream.go).
//
// DEVIATION (compat plumbing): as in the other API ports, compat travels on the
// options struct rather than model.compat.
//
// Deferred from the TS original: HTTP proxy support, HTTP/1.1 forcing, and the
// Smithy middleware hook. Custom headers are applied before signing, so they are
// covered by the signature exactly as the TS middleware arranges.

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

const APIBedrockConverseStream = "bedrock-converse-stream"

func init() {
	RegisterStreamer(APIBedrockConverseStream, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*BedrockOptions)
			return StreamBedrock(model, ctx, o)
		},
		StreamSimple: StreamBedrockSimple,
	})
}

// bedrockService is the SigV4 service name for the Bedrock runtime.
const bedrockService = "bedrock"

// bedrockEmptyTextPlaceholder stands in for text Bedrock would reject as blank.
const bedrockEmptyTextPlaceholder = "<empty>"

// bedrockDefaultRegion is the fallback when nothing else configures one.
const bedrockDefaultRegion = "us-east-1"

// BedrockCompat holds compatibility settings for the Bedrock Converse API.
type BedrockCompat struct {
	SupportsStrictMode *bool `json:"supportsStrictMode,omitempty"`
}

// BedrockOptions are the stream options for the bedrock-converse-stream API.
type BedrockOptions struct {
	StreamOptions
	// Region overrides AWS_REGION / AWS_DEFAULT_REGION.
	Region string
	// Profile selects a shared-credentials profile, winning over ambient keys.
	Profile string
	// ToolChoice is "auto" | "any" | "none", or a map selecting one tool.
	ToolChoice any
	// Reasoning requests extended thinking.
	Reasoning ThinkingLevel
	// ThinkingBudgets are per-level token budgets for budget-based models.
	ThinkingBudgets *ThinkingBudgets
	// InterleavedThinking requests the interleaved-thinking beta for
	// non-adaptive Claude models. Defaults to true.
	InterleavedThinking *bool
	// ThinkingDisplay is "summarized" (default) or "omitted".
	ThinkingDisplay string
	// RequestMetadata attaches cost-allocation tags to the inference request.
	RequestMetadata map[string]string
	// BearerToken authenticates with a Bedrock API key, bypassing SigV4.
	BearerToken string
	// Compat carries BedrockCompat (see DEVIATION note).
	Compat *BedrockCompat
}

// StreamBedrock streams a turn over the bedrock-converse-stream API.
func StreamBedrock(model *Model, conv *Context, options *BedrockOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &BedrockOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &bedrockStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamBedrockSimple maps SimpleStreamOptions onto the Bedrock stream
// (streamSimple in bedrock-converse-stream.ts).
func StreamBedrockSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)

	if options.Reasoning == "" {
		return StreamBedrock(model, conv, &BedrockOptions{StreamOptions: base})
	}

	if isAnthropicClaudeBedrockModel(model) && !bedrockSupportsAdaptiveThinking(model) {
		// Budget-based Claude: carve the thinking budget out of maxTokens and
		// always leave room for an answer.
		maxTokens, thinkingBudget := adjustAnthropicMaxTokensForThinking(
			base.MaxTokens, model.MaxTokens, options.Reasoning, options.ThinkingBudgets)
		maxTokens = ClampMaxTokensToContext(model, conv, maxTokens)
		base.MaxTokens = maxTokens

		budgets := ThinkingBudgets{}
		if options.ThinkingBudgets != nil {
			budgets = *options.ThinkingBudgets
		}
		clamped := min(thinkingBudget, max(0, maxTokens-MinAnswerTokens))
		setThinkingBudget(&budgets, clampReasoning(options.Reasoning), clamped)

		return StreamBedrock(model, conv, &BedrockOptions{
			StreamOptions:   base,
			Reasoning:       options.Reasoning,
			ThinkingBudgets: &budgets,
		})
	}

	return StreamBedrock(model, conv, &BedrockOptions{
		StreamOptions:   base,
		Reasoning:       options.Reasoning,
		ThinkingBudgets: options.ThinkingBudgets,
	})
}

// setThinkingBudget writes a per-level budget override.
func setThinkingBudget(budgets *ThinkingBudgets, level ThinkingLevel, value int) {
	switch level {
	case ThinkingMinimal:
		budgets.Minimal = value
	case ThinkingLow:
		budgets.Low = value
	case ThinkingMedium:
		budgets.Medium = value
	default:
		budgets.High = value
	}
}

// ---------------------------------------------------------------------------
// Model capability probes
// ---------------------------------------------------------------------------

// bedrockMatchCandidates returns lower-cased id/name variants, with separators
// normalized to "-". Application inference profile ARNs do not carry the model
// name, so the user-controlled Name is checked too.
func bedrockMatchCandidates(model *Model) []string {
	values := []string{model.ID}
	if model.Name != "" {
		values = append(values, model.Name)
	}
	var candidates []string
	for _, value := range values {
		lower := strings.ToLower(value)
		candidates = append(candidates, lower, bedrockSeparators.ReplaceAllString(lower, "-"))
	}
	return candidates
}

var bedrockSeparators = regexp.MustCompile(`[\s_.:]+`)

func bedrockCandidatesContain(model *Model, needles ...string) bool {
	for _, candidate := range bedrockMatchCandidates(model) {
		for _, needle := range needles {
			if strings.Contains(candidate, needle) {
				return true
			}
		}
	}
	return false
}

// isAnthropicClaudeBedrockModel reports whether the model is a Claude model.
func isAnthropicClaudeBedrockModel(model *Model) bool {
	id := strings.ToLower(model.ID)
	name := strings.ToLower(model.Name)
	return strings.Contains(id, "anthropic.claude") ||
		strings.Contains(id, "anthropic/claude") ||
		strings.Contains(name, "anthropic.claude") ||
		strings.Contains(name, "anthropic/claude") ||
		strings.Contains(name, "claude")
}

// bedrockSupportsAdaptiveThinking reports whether the model takes adaptive
// thinking with an effort level rather than a token budget.
func bedrockSupportsAdaptiveThinking(model *Model) bool {
	return bedrockCandidatesContain(model,
		"opus-4-6", "opus-4-7", "opus-4-8", "opus-5",
		"sonnet-4-6", "sonnet-5", "fable-5")
}

// bedrockSupportsNativeXHigh reports whether "xhigh" is a real effort level.
func bedrockSupportsNativeXHigh(model *Model) bool {
	return bedrockCandidatesContain(model, "opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5")
}

// bedrockSupportsPromptCaching reports whether explicit cache points apply.
// Nova models cache automatically and need none.
func bedrockSupportsPromptCaching(model *Model, env map[string]string) bool {
	if !bedrockCandidatesContain(model, "claude") {
		// Application inference profile ARNs hide the model name, so allow an
		// explicit override.
		return getProviderEnvValue("AWS_BEDROCK_FORCE_CACHE", env) == "1"
	}
	switch {
	case bedrockCandidatesContain(model, "fable-5", "opus-5", "sonnet-5"):
		return true
	case bedrockCandidatesContain(model, "-4-"):
		return true
	case bedrockCandidatesContain(model, "claude-3-7-sonnet"):
		return true
	case bedrockCandidatesContain(model, "claude-3-5-haiku"):
		return true
	default:
		return false
	}
}

// bedrockSupportsThinkingSignature reports whether reasoningContent may carry a
// signature. Non-Claude models reject the field outright.
func bedrockSupportsThinkingSignature(model *Model) bool {
	return isAnthropicClaudeBedrockModel(model)
}

// bedrockMapThinkingLevelToEffort maps a pi level onto an adaptive effort.
func bedrockMapThinkingLevelToEffort(model *Model, level ThinkingLevel) string {
	if level == ThinkingXHigh && bedrockSupportsNativeXHigh(model) {
		return "xhigh"
	}
	if mapped, ok := model.ThinkingLevelMap[level]; ok && mapped != nil {
		return *mapped
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

// ---------------------------------------------------------------------------
// Endpoint and region resolution
// ---------------------------------------------------------------------------

var bedrockEndpointRegionRe = regexp.MustCompile(`^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$`)
var bedrockARNRegionRe = regexp.MustCompile(`^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):`)

// standardBedrockEndpointRegion extracts the region from a standard Bedrock
// runtime host, or "" for a custom endpoint.
func standardBedrockEndpointRegion(baseURL string) string {
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" {
		return ""
	}
	match := bedrockEndpointRegionRe.FindStringSubmatch(strings.ToLower(parsed.Hostname()))
	if match == nil {
		return ""
	}
	return match[1]
}

// configuredBedrockRegion resolves an explicitly configured region.
func configuredBedrockRegion(options *BedrockOptions) string {
	if options.Region != "" {
		return options.Region
	}
	if region := getProviderEnvValue("AWS_REGION", options.Env); region != "" {
		return region
	}
	return getProviderEnvValue("AWS_DEFAULT_REGION", options.Env)
}

// resolveBedrockProfile picks the AWS profile to sign with.
//
// An ambient AWS_PROFILE must not override credentials the user configured for
// this provider. resolveAWSCredentials short-circuits to the profile whenever
// one is named, so consulting the process environment unconditionally means a
// user who happens to export AWS_PROFILE for an unrelated project has their
// configured Bedrock keys silently ignored -- and either fails with "no
// credentials for AWS profile" or signs against the wrong AWS account.
//
// Precedence therefore is: an explicit option, then an explicitly configured
// AWS_PROFILE, then the ambient one but only when no static key pair was
// configured to be overridden.
func resolveBedrockProfile(options *BedrockOptions) string {
	if options == nil {
		return ""
	}
	if options.Profile != "" {
		return options.Profile
	}
	if options.Env != nil {
		if profile := options.Env["AWS_PROFILE"]; profile != "" {
			return profile
		}
		if options.Env["AWS_ACCESS_KEY_ID"] != "" && options.Env["AWS_SECRET_ACCESS_KEY"] != "" {
			return ""
		}
	}
	return os.Getenv("AWS_PROFILE")
}

// resolveBedrockTarget resolves the region and endpoint for a request.
//
// Region precedence matches the TS: an ARN-embedded region, then an explicit
// option or env var, then a standard endpoint's region, then the profile's
// configured region, then us-east-1.
func resolveBedrockTarget(model *Model, options *BedrockOptions) (region string, endpoint string) {
	profile := resolveBedrockProfile(options)
	configured := configuredBedrockRegion(options)
	endpointRegion := standardBedrockEndpointRegion(model.BaseURL)

	// A standard endpoint is only pinned when nothing else selects a region, so
	// a catalog default cannot override AWS_REGION or a profile.
	useExplicitEndpoint := endpointRegion == "" || (configured == "" && profile == "")

	switch {
	case bedrockARNRegionRe.MatchString(model.ID):
		region = bedrockARNRegionRe.FindStringSubmatch(model.ID)[1]
	case configured != "":
		region = configured
	case endpointRegion != "" && useExplicitEndpoint:
		region = endpointRegion
	default:
		if fromProfile := regionFromProfile(profile, options.Env); fromProfile != "" {
			region = fromProfile
		} else {
			region = bedrockDefaultRegion
		}
	}

	if useExplicitEndpoint && model.BaseURL != "" {
		endpoint = strings.TrimSuffix(model.BaseURL, "/")
	} else {
		endpoint = fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com", region)
	}
	return region, endpoint
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

// buildBedrockParams builds the ConverseStream request body.
func buildBedrockParams(model *Model, conv *Context, options *BedrockOptions) (map[string]any, error) {
	cacheRetention := resolveCacheRetention(options.CacheRetention, options.Env)

	messages, err := bedrockConvertMessages(model, conv, cacheRetention, options.Env)
	if err != nil {
		return nil, err
	}

	params := map[string]any{"messages": messages}

	if system := bedrockBuildSystem(conv.SystemPrompt, model, cacheRetention, options.Env); system != nil {
		params["system"] = system
	}

	inference := map[string]any{}
	// Claude on Bedrock requires an explicit output cap.
	maxTokens := options.MaxTokens
	if maxTokens == 0 && isAnthropicClaudeBedrockModel(model) {
		maxTokens = model.MaxTokens
	}
	if maxTokens > 0 {
		inference["maxTokens"] = maxTokens
	}
	if options.Temperature != nil {
		inference["temperature"] = *options.Temperature
	}
	if len(inference) > 0 {
		params["inferenceConfig"] = inference
	}

	// Options-carried compat wins; otherwise the model's raw catalog compat is
	// decoded (see DEVIATION note).
	var compat BedrockCompat
	if options.Compat != nil {
		compat = *options.Compat
	} else {
		model.DecodeRawCompat(&compat)
	}
	supportsStrict := compat.SupportsStrictMode != nil && *compat.SupportsStrictMode
	toolConfig, err := bedrockConvertToolConfig(conv.Tools, options.ToolChoice, supportsStrict)
	if err != nil {
		return nil, err
	}
	if toolConfig != nil {
		params["toolConfig"] = toolConfig
	}

	if fields := bedrockAdditionalModelRequestFields(model, options); fields != nil {
		params["additionalModelRequestFields"] = fields
	}
	if len(options.RequestMetadata) > 0 {
		params["requestMetadata"] = options.RequestMetadata
	}

	// Last so custom keys override the named request fields.
	for k, v := range options.SamplingParams {
		params[k] = v
	}
	return params, nil
}

// bedrockCachePoint builds a cache point block for the configured retention.
func bedrockCachePoint(cacheRetention CacheRetention) map[string]any {
	point := map[string]any{"type": "default"}
	if cacheRetention == CacheRetentionLong {
		point["ttl"] = "1h"
	}
	return map[string]any{"cachePoint": point}
}

// bedrockBuildSystem builds the system content blocks.
func bedrockBuildSystem(systemPrompt string, model *Model, cacheRetention CacheRetention, env map[string]string) []any {
	if systemPrompt == "" {
		return nil
	}
	blocks := []any{map[string]any{"text": sanitizeSurrogates(systemPrompt)}}
	if cacheRetention != CacheRetentionNone && bedrockSupportsPromptCaching(model, env) {
		blocks = append(blocks, bedrockCachePoint(cacheRetention))
	}
	return blocks
}

var bedrockToolCallIDChars = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// bedrockNormalizeToolCallID adapts normalizeToolCallId to the
// TransformMessages signature.
func bedrockNormalizeToolCallID(id string, _ *Model, _ *AssistantMessage) string {
	sanitized := bedrockToolCallIDChars.ReplaceAllString(id, "_")
	if len(sanitized) > 64 {
		sanitized = sanitized[:64]
	}
	return sanitized
}

// bedrockTextBlock returns a text block, or nil when the text is blank.
func bedrockTextBlock(text string) map[string]any {
	sanitized := sanitizeSurrogates(text)
	if strings.TrimSpace(sanitized) == "" {
		return nil
	}
	return map[string]any{"text": sanitized}
}

// bedrockRequiredTextBlock returns a text block, substituting a placeholder for
// blank text so Bedrock does not reject the message.
func bedrockRequiredTextBlock(text string) map[string]any {
	if block := bedrockTextBlock(text); block != nil {
		return block
	}
	return map[string]any{"text": bedrockEmptyTextPlaceholder}
}

// bedrockImageFormats maps mime types onto Bedrock image formats.
var bedrockImageFormats = map[string]string{
	"image/jpeg": "jpeg",
	"image/jpg":  "jpeg",
	"image/png":  "png",
	"image/gif":  "gif",
	"image/webp": "webp",
}

// bedrockImageBlock builds an image block. Bedrock takes raw bytes, which the
// JSON encoding re-base64s, so the incoming base64 is validated here.
func bedrockImageBlock(mimeType, data string) (map[string]any, error) {
	format, ok := bedrockImageFormats[mimeType]
	if !ok {
		return nil, fmt.Errorf("unknown image type: %s", mimeType)
	}
	if _, err := base64.StdEncoding.DecodeString(data); err != nil {
		return nil, fmt.Errorf("invalid base64 image data: %w", err)
	}
	return map[string]any{
		"image": map[string]any{
			"format": format,
			"source": map[string]any{"bytes": data},
		},
	}, nil
}

// bedrockToolResultContent converts tool result content blocks.
func bedrockToolResultContent(content []ContentBlock) ([]any, error) {
	result := []any{}
	for _, block := range content {
		switch b := block.(type) {
		case ImageContent:
			image, err := bedrockImageBlock(b.MimeType, b.Data)
			if err != nil {
				return nil, err
			}
			result = append(result, image)
		case TextContent:
			if text := bedrockTextBlock(b.Text); text != nil {
				result = append(result, text)
			}
		}
	}
	if len(result) == 0 {
		result = append(result, map[string]any{"text": bedrockEmptyTextPlaceholder})
	}
	return result, nil
}

// bedrockConvertMessages converts a transcript to Converse messages.
func bedrockConvertMessages(model *Model, conv *Context, cacheRetention CacheRetention, env map[string]string) ([]any, error) {
	transformed := TransformMessages(conv.Messages, model, bedrockNormalizeToolCallID)
	result := []any{}

	for i := 0; i < len(transformed); i++ {
		switch m := derefResponsesMessage(transformed[i]).(type) {
		case UserMessage:
			content := []any{}
			if text, isString := m.StringContent(); isString {
				content = append(content, bedrockRequiredTextBlock(text))
			} else {
				for _, block := range m.BlockContent() {
					switch b := block.(type) {
					case TextContent:
						if text := bedrockTextBlock(b.Text); text != nil {
							content = append(content, text)
						}
					case ImageContent:
						image, err := bedrockImageBlock(b.MimeType, b.Data)
						if err != nil {
							return nil, err
						}
						content = append(content, image)
					}
				}
				if len(content) == 0 {
					content = append(content, map[string]any{"text": bedrockEmptyTextPlaceholder})
				}
			}
			result = append(result, map[string]any{"role": "user", "content": content})

		case AssistantMessage:
			// Bedrock rejects messages with empty content, which is what an
			// aborted turn leaves behind.
			if len(m.Content) == 0 {
				continue
			}
			content := []any{}
			for _, block := range m.Content {
				switch b := block.(type) {
				case TextContent:
					if text := bedrockTextBlock(b.Text); text != nil {
						content = append(content, text)
					}
				case ToolCall:
					arguments := b.Arguments
					if arguments == nil {
						arguments = map[string]any{}
					}
					content = append(content, map[string]any{
						"toolUse": map[string]any{
							"toolUseId": b.ID,
							"name":      b.Name,
							"input":     sanitizeBedrockDocument(arguments),
						},
					})
				case ThinkingContent:
					thinking := sanitizeSurrogates(b.Thinking)
					if strings.TrimSpace(thinking) == "" {
						continue
					}
					if !bedrockSupportsThinkingSignature(model) {
						content = append(content, map[string]any{
							"reasoningContent": map[string]any{
								"reasoningText": map[string]any{"text": thinking},
							},
						})
						continue
					}
					// Signatures arrive after thinking deltas, so a partial or
					// externally persisted message may lack one. Bedrock rejects
					// an unsigned reasoning block, so fall back to plain text.
					if strings.TrimSpace(b.ThinkingSignature) == "" {
						content = append(content, map[string]any{"text": thinking})
						continue
					}
					content = append(content, map[string]any{
						"reasoningContent": map[string]any{
							"reasoningText": map[string]any{
								"text":      thinking,
								"signature": b.ThinkingSignature,
							},
						},
					})
				}
			}
			if len(content) == 0 {
				continue
			}
			result = append(result, map[string]any{"role": "assistant", "content": content})

		case ToolResultMessage:
			// Bedrock requires every consecutive tool result in one user turn.
			toolResults := []any{}
			appendResult := func(msg ToolResultMessage) error {
				content, err := bedrockToolResultContent(msg.Content)
				if err != nil {
					return err
				}
				status := "success"
				if msg.IsError {
					status = "error"
				}
				toolResults = append(toolResults, map[string]any{
					"toolResult": map[string]any{
						"toolUseId": msg.ToolCallID,
						"content":   content,
						"status":    status,
					},
				})
				return nil
			}
			if err := appendResult(m); err != nil {
				return nil, err
			}

			j := i + 1
			for j < len(transformed) {
				next, ok := derefResponsesMessage(transformed[j]).(ToolResultMessage)
				if !ok {
					break
				}
				if err := appendResult(next); err != nil {
					return nil, err
				}
				j++
			}
			i = j - 1

			result = append(result, map[string]any{"role": "user", "content": toolResults})
		}
	}

	// A trailing cache point on the last user turn caches the whole prefix.
	if cacheRetention != CacheRetentionNone && bedrockSupportsPromptCaching(model, env) && len(result) > 0 {
		last, _ := result[len(result)-1].(map[string]any)
		if last != nil && last["role"] == "user" {
			if content, ok := last["content"].([]any); ok {
				last["content"] = append(content, bedrockCachePoint(cacheRetention))
			}
		}
	}
	return result, nil
}

// sanitizeBedrockDocument drops empty object keys, which Bedrock's document
// type rejects.
func sanitizeBedrockDocument(value any) any {
	switch v := value.(type) {
	case map[string]any:
		out := make(map[string]any, len(v))
		for key, nested := range v {
			if key == "" {
				continue
			}
			out[key] = sanitizeBedrockDocument(nested)
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = sanitizeBedrockDocument(item)
		}
		return out
	default:
		return value
	}
}

// bedrockConvertToolConfig builds the toolConfig field, or nil to omit it.
func bedrockConvertToolConfig(tools []Tool, toolChoice any, supportsStrictMode bool) (map[string]any, error) {
	if len(tools) == 0 {
		return nil, nil
	}
	if choice, ok := toolChoice.(string); ok && choice == "none" {
		return nil, nil
	}

	specs := make([]any, 0, len(tools))
	for _, tool := range tools {
		strict, hasStrict, err := ResolveJSONSchemaStrictSampling(tool, supportsStrictMode)
		if err != nil {
			return nil, err
		}
		var parameters any = map[string]any{}
		if len(tool.Parameters) > 0 {
			parameters = json.RawMessage(tool.Parameters)
		}
		spec := map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
			"inputSchema": map[string]any{"json": parameters},
		}
		if hasStrict && strict {
			spec["strict"] = true
		}
		specs = append(specs, map[string]any{"toolSpec": spec})
	}

	config := map[string]any{"tools": specs}
	switch choice := toolChoice.(type) {
	case string:
		switch choice {
		case "auto":
			config["toolChoice"] = map[string]any{"auto": map[string]any{}}
		case "any":
			config["toolChoice"] = map[string]any{"any": map[string]any{}}
		}
	case map[string]any:
		if choice["type"] == "tool" {
			if name, ok := choice["name"].(string); ok {
				config["toolChoice"] = map[string]any{"tool": map[string]any{"name": name}}
			}
		}
	}
	return config, nil
}

// isGovCloudBedrockTarget reports whether the request targets GovCloud, which
// rejects the Claude thinking.display field.
func isGovCloudBedrockTarget(model *Model, options *BedrockOptions) bool {
	if region := configuredBedrockRegion(options); strings.HasPrefix(strings.ToLower(region), "us-gov-") {
		return true
	}
	id := strings.ToLower(model.ID)
	return strings.HasPrefix(id, "us-gov.") || strings.HasPrefix(id, "arn:aws-us-gov:")
}

// bedrockAdditionalModelRequestFields builds the passthrough field carrying
// Claude thinking configuration.
func bedrockAdditionalModelRequestFields(model *Model, options *BedrockOptions) map[string]any {
	if options.Reasoning == "" || !model.Reasoning {
		return nil
	}
	if !isAnthropicClaudeBedrockModel(model) {
		return nil
	}

	display := options.ThinkingDisplay
	if display == "" {
		display = "summarized"
	}
	if isGovCloudBedrockTarget(model, options) {
		display = ""
	}

	result := map[string]any{}
	if bedrockSupportsAdaptiveThinking(model) {
		thinking := map[string]any{"type": "adaptive"}
		if display != "" {
			thinking["display"] = display
		}
		result["thinking"] = thinking
		result["output_config"] = map[string]any{
			"effort": bedrockMapThinkingLevelToEffort(model, options.Reasoning),
		}
		return result
	}

	// Budget-based Claude. Extended levels clamp to the high budget.
	budgets := ThinkingBudgets{Minimal: 1024, Low: 2048, Medium: 8192, High: 16384}
	level := clampReasoning(options.Reasoning)
	budget := thinkingBudgetForLevel(budgets, level)
	if options.ThinkingBudgets != nil {
		if override, ok := lookupThinkingBudget(options.ThinkingBudgets, level); ok {
			budget = override
		}
	}
	thinking := map[string]any{"type": "enabled", "budget_tokens": budget}
	if display != "" {
		thinking["display"] = display
	}
	result["thinking"] = thinking

	if options.InterleavedThinking == nil || *options.InterleavedThinking {
		result["anthropic_beta"] = []any{anthropicInterleavedThinkingBeta}
	}
	return result
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

type bedrockStreamer struct {
	model   *Model
	conv    *Context
	options *BedrockOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *bedrockStreamer) run() {
	p.output = &AssistantMessage{
		Role:       "assistant",
		Content:    []ContentBlock{},
		API:        APIBedrockConverseStream,
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

func (p *bedrockStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *bedrockStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *bedrockStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	params, err := buildBedrockParams(p.model, p.conv, opts)
	if err != nil {
		return err
	}
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doBedrockRequest(ctx, p.model, opts, params)
	}, opts.MaxRetries, opts.MaxRetryDelayMs)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if opts.OnResponse != nil {
		opts.OnResponse(ProviderResponse{Status: resp.StatusCode, Headers: headersToRecord(resp.Header)}, p.model)
	}

	if err := p.consumeStream(ctx, resp.Body); err != nil {
		return err
	}

	if ctx.Err() != nil {
		return AbortError{}
	}
	if p.output.StopReason == StopPending {
		return fmt.Errorf("bedrock stream ended without a stop reason")
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

// bedrockStreamEvent is the union of ConverseStream event payloads. Which
// fields are set depends on the frame's :event-type header.
type bedrockStreamEvent struct {
	Role  string `json:"role"`
	Start *struct {
		ToolUse *struct {
			ToolUseID string `json:"toolUseId"`
			Name      string `json:"name"`
		} `json:"toolUse"`
	} `json:"start"`
	Delta *struct {
		Text    *string `json:"text"`
		ToolUse *struct {
			Input string `json:"input"`
		} `json:"toolUse"`
		ReasoningContent *struct {
			Text      string `json:"text"`
			Signature string `json:"signature"`
		} `json:"reasoningContent"`
	} `json:"delta"`
	ContentBlockIndex *int   `json:"contentBlockIndex"`
	StopReason        string `json:"stopReason"`
	Usage             *struct {
		InputTokens           int `json:"inputTokens"`
		OutputTokens          int `json:"outputTokens"`
		TotalTokens           int `json:"totalTokens"`
		CacheReadInputTokens  int `json:"cacheReadInputTokens"`
		CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
	} `json:"usage"`
	Message string `json:"message"`
}

// bedrockBlock tracks one in-flight content block, keyed by the wire's
// contentBlockIndex.
type bedrockBlock struct {
	kind     string // "text" | "thinking" | "toolCall"
	text     *TextContent
	thinking *ThinkingContent
	toolCall *ToolCall
	// partialJSON accumulates tool-use input deltas; see streamingToolCall for
	// why this is a Builder rather than a string.
	partialJSON strings.Builder
	// parsedJSONLen is how much of partialJSON the Arguments map reflects.
	// See shouldReparseStreamingJSON.
	parsedJSONLen int
	// textAccum accumulates text or thinking deltas for this block; see
	// textAccumulator. Safe by value here because bedrockBlock is only ever
	// held through a pointer.
	textAccum    textAccumulator
	wireIndex    int
	contentIndex int
}

// consumeStream folds the binary event stream into the output message.
func (p *bedrockStreamer) consumeStream(ctx context.Context, body io.Reader) error {
	output := p.output
	reader := newEventStreamReader(body)

	var blocks []*bedrockBlock
	findBlock := func(wireIndex int) *bedrockBlock {
		for _, block := range blocks {
			if block.wireIndex == wireIndex {
				return block
			}
		}
		return nil
	}
	sync := func(block *bedrockBlock) {
		switch block.kind {
		case "text":
			output.Content[block.contentIndex] = *block.text
		case "thinking":
			output.Content[block.contentIndex] = *block.thinking
		case "toolCall":
			output.Content[block.contentIndex] = *block.toolCall
		}
	}

	for {
		message, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			if ctx.Err() != nil {
				return AbortError{}
			}
			return err
		}

		// Exception frames carry the modeled error type in a header.
		if exceptionType := message.ExceptionType(); exceptionType != "" {
			return bedrockExceptionError(exceptionType, message.Payload)
		}
		if message.MessageType() == "exception" || message.MessageType() == "error" {
			return bedrockExceptionError(message.Headers[":error-code"], message.Payload)
		}

		var event bedrockStreamEvent
		if len(message.Payload) > 0 {
			if err := json.Unmarshal(message.Payload, &event); err != nil {
				return fmt.Errorf("parsing bedrock event: %w", err)
			}
		}

		switch message.EventType() {
		case "messageStart":
			if event.Role != "assistant" {
				return fmt.Errorf("unexpected assistant message start but got %s message start instead", event.Role)
			}
			p.push(AssistantMessageEvent{Type: EventStart})

		case "contentBlockStart":
			if event.Start == nil || event.Start.ToolUse == nil || event.ContentBlockIndex == nil {
				continue
			}
			toolCall := &ToolCall{
				Type:      "toolCall",
				ID:        event.Start.ToolUse.ToolUseID,
				Name:      event.Start.ToolUse.Name,
				Arguments: map[string]any{},
			}
			output.Content = append(output.Content, *toolCall)
			block := &bedrockBlock{
				kind:         "toolCall",
				toolCall:     toolCall,
				wireIndex:    *event.ContentBlockIndex,
				contentIndex: len(output.Content) - 1,
			}
			blocks = append(blocks, block)
			p.push(AssistantMessageEvent{Type: EventToolCallStart, ContentIndex: block.contentIndex})

		case "contentBlockDelta":
			if event.Delta == nil || event.ContentBlockIndex == nil {
				continue
			}
			wireIndex := *event.ContentBlockIndex
			block := findBlock(wireIndex)

			switch {
			case event.Delta.Text != nil:
				// Text blocks get no contentBlockStart, so open one lazily.
				if block == nil {
					text := &TextContent{Type: "text"}
					output.Content = append(output.Content, *text)
					block = &bedrockBlock{
						kind: "text", text: text,
						wireIndex: wireIndex, contentIndex: len(output.Content) - 1,
					}
					blocks = append(blocks, block)
					p.push(AssistantMessageEvent{Type: EventTextStart, ContentIndex: block.contentIndex})
				}
				if block.kind != "text" {
					continue
				}
				block.text.Text = block.textAccum.add(*event.Delta.Text)
				sync(block)
				p.push(AssistantMessageEvent{Type: EventTextDelta, ContentIndex: block.contentIndex, Delta: *event.Delta.Text})

			case event.Delta.ToolUse != nil:
				if block == nil || block.kind != "toolCall" {
					continue
				}
				block.partialJSON.WriteString(event.Delta.ToolUse.Input)
				if shouldReparseStreamingJSON(block.parsedJSONLen, block.partialJSON.Len()) {
					block.toolCall.Arguments = ParseStreamingJSON(block.partialJSON.String())
					block.parsedJSONLen = block.partialJSON.Len()
				}
				sync(block)
				p.push(AssistantMessageEvent{Type: EventToolCallDelta, ContentIndex: block.contentIndex, Delta: event.Delta.ToolUse.Input})

			case event.Delta.ReasoningContent != nil:
				if block == nil {
					thinking := &ThinkingContent{Type: "thinking"}
					output.Content = append(output.Content, *thinking)
					block = &bedrockBlock{
						kind: "thinking", thinking: thinking,
						wireIndex: wireIndex, contentIndex: len(output.Content) - 1,
					}
					blocks = append(blocks, block)
					p.push(AssistantMessageEvent{Type: EventThinkingStart, ContentIndex: block.contentIndex})
				}
				if block.kind != "thinking" {
					continue
				}
				if text := event.Delta.ReasoningContent.Text; text != "" {
					block.thinking.Thinking = block.textAccum.add(text)
					sync(block)
					p.push(AssistantMessageEvent{Type: EventThinkingDelta, ContentIndex: block.contentIndex, Delta: text})
				}
				// The signature arrives in its own delta and is not streamed to
				// consumers; it only matters for replay.
				if signature := event.Delta.ReasoningContent.Signature; signature != "" {
					block.thinking.ThinkingSignature += signature
					sync(block)
				}
			}

		case "contentBlockStop":
			if event.ContentBlockIndex == nil {
				continue
			}
			block := findBlock(*event.ContentBlockIndex)
			if block == nil {
				continue
			}
			switch block.kind {
			case "text":
				p.push(AssistantMessageEvent{Type: EventTextEnd, ContentIndex: block.contentIndex, Content: block.text.Text})
			case "thinking":
				p.push(AssistantMessageEvent{Type: EventThinkingEnd, ContentIndex: block.contentIndex, Content: block.thinking.Thinking})
			case "toolCall":
				block.toolCall.Arguments = ParseStreamingJSON(block.partialJSON.String())
				sync(block)
				tc := *block.toolCall
				p.push(AssistantMessageEvent{Type: EventToolCallEnd, ContentIndex: block.contentIndex, ToolCall: &tc})
			}

		case "messageStop":
			output.RawStopReason = event.StopReason
			stopReason, errorMessage := mapBedrockStopReason(event.StopReason)
			output.StopReason = stopReason
			if errorMessage != "" {
				output.ErrorMessage = errorMessage
			}

		case "metadata":
			if event.Usage == nil {
				continue
			}
			output.Usage.Input = event.Usage.InputTokens
			output.Usage.Output = event.Usage.OutputTokens
			output.Usage.CacheRead = event.Usage.CacheReadInputTokens
			output.Usage.CacheWrite = event.Usage.CacheWriteInputTokens
			output.Usage.TotalTokens = event.Usage.TotalTokens
			if output.Usage.TotalTokens == 0 {
				output.Usage.TotalTokens = output.Usage.Input + output.Usage.Output
			}
			CalculateCost(p.model, &output.Usage)
		}
	}
	return nil
}

// bedrockErrorPrefixes are human-readable prefixes for modeled exceptions. The
// retry logic matches patterns like "server.?error", so the legacy prefix
// wording is preserved rather than the raw exception name.
var bedrockErrorPrefixes = map[string]string{
	"InternalServerException":     "Internal server error",
	"ModelStreamErrorException":   "Model stream error",
	"ValidationException":         "Validation error",
	"ThrottlingException":         "Throttling error",
	"ServiceUnavailableException": "Service unavailable",
}

// bedrockDataRetentionDocsURL is appended when a model rejects the account's
// configured data retention mode.
const bedrockDataRetentionDocsURL = "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html"

// bedrockExceptionError builds an error from a mid-stream exception frame.
func bedrockExceptionError(exceptionType string, payload []byte) error {
	var body struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(payload, &body)

	message := body.Message
	if message == "" {
		message = strings.TrimSpace(string(payload))
	}
	if message == "" {
		message = "Unknown error"
	}
	if strings.Contains(strings.ToLower(message), "data retention mode") {
		message += fmt.Sprintf(" See %s for supported data retention modes.", bedrockDataRetentionDocsURL)
	}

	prefix := exceptionType
	if human, ok := bedrockErrorPrefixes[exceptionType]; ok {
		prefix = human
	}
	if prefix == "" {
		return fmt.Errorf("%s", message)
	}
	return fmt.Errorf("%s: %s", prefix, message)
}

// mapBedrockStopReason maps a Converse stop reason onto a pi stop reason.
func mapBedrockStopReason(reason string) (StopReason, string) {
	switch reason {
	case "end_turn", "stop_sequence":
		return StopStop, ""
	case "max_tokens", "model_context_window_exceeded":
		return StopLength, ""
	case "tool_use":
		return StopToolUse, ""
	case "":
		return StopError, ""
	default:
		return StopError, "Provider stopped with: " + reason
	}
}

// bedrockReservedHeaders must never be overwritten by caller headers: they
// participate in the SigV4 canonical request or carry auth.
func isBedrockReservedHeader(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasPrefix(lower, "x-amz-") || lower == "authorization" || lower == "host"
}

// doBedrockRequest performs one signed POST to the ConverseStream endpoint.
func doBedrockRequest(ctx context.Context, model *Model, options *BedrockOptions, params map[string]any) (*http.Response, error) {
	body, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	region, endpoint := resolveBedrockTarget(model, options)
	requestURL := fmt.Sprintf("%s/model/%s/converse-stream", endpoint, url.PathEscape(model.ID))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/vnd.amazon.eventstream")

	// Caller headers are applied before signing so the signature covers them,
	// matching the TS build-step middleware.
	for name, value := range options.Headers {
		if value == nil || isBedrockReservedHeader(name) {
			continue
		}
		req.Header.Set(name, *value)
	}

	skipAuth := getProviderEnvValue("AWS_BEDROCK_SKIP_AUTH", options.Env) == "1"
	bearerToken := options.BearerToken
	if bearerToken == "" {
		bearerToken = options.APIKey
	}
	if bearerToken == "" {
		bearerToken = getProviderEnvValue("AWS_BEARER_TOKEN_BEDROCK", options.Env)
	}
	// The sentinel from catalog auth resolution is not a real token.
	if bearerToken == "<authenticated>" {
		bearerToken = ""
	}

	switch {
	case skipAuth:
		// Unauthenticated proxies still expect a well-formed signature.
		creds := &awsCredentials{AccessKeyID: "dummy-access-key", SecretAccessKey: "dummy-secret-key"}
		if err := signAWSRequest(req, body, creds, region, bedrockService, time.Now()); err != nil {
			return nil, err
		}
	case bearerToken != "":
		req.Header.Set("Authorization", "Bearer "+bearerToken)
	default:
		creds, err := resolveAWSCredentials(resolveBedrockProfile(options), options.Env)
		if err != nil {
			return nil, err
		}
		if err := signAWSRequest(req, body, creds, region, bedrockService, time.Now()); err != nil {
			return nil, err
		}
	}

	client := &http.Client{}
	if options.TimeoutMs > 0 {
		client.Timeout = time.Duration(options.TimeoutMs) * time.Millisecond
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
