package llm

// Go port of reference/pi/packages/ai/src/api/google-vertex.ts.
//
// Vertex speaks the same generateContent wire protocol as the Generative
// Language API, so message/tool conversion and stream assembly are reused from
// google_shared.go. What differs is endpoint construction (project/location
// resource paths) and auth: either a Vertex API key (express mode) or a Google
// Cloud bearer token (see google_auth.go).

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

const APIGoogleVertex = "google-vertex"

func init() {
	RegisterStreamer(APIGoogleVertex, StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*GoogleVertexOptions)
			return StreamGoogleVertex(model, ctx, o)
		},
		StreamSimple: StreamGoogleVertexSimple,
	})
}

// vertexAPIVersion is the Vertex AI API version pi pins.
const vertexAPIVersion = "v1"

// gcpVertexCredentialsMarker is the sentinel pi stores in auth.json when Vertex
// should use ambient Google Cloud credentials rather than an API key.
const gcpVertexCredentialsMarker = "gcp-vertex-credentials"

// GoogleVertexOptions are the stream options for the google-vertex API.
type GoogleVertexOptions struct {
	StreamOptions
	// ToolChoice is "auto" | "none" | "any".
	ToolChoice string
	// Thinking configures thought output. nil omits thinkingConfig entirely.
	Thinking *GoogleThinkingConfig
	// Project is the Google Cloud project id (else GOOGLE_CLOUD_PROJECT /
	// GCLOUD_PROJECT).
	Project string
	// Location is the Vertex region (else GOOGLE_CLOUD_LOCATION).
	Location string
}

// StreamGoogleVertex streams a turn over the google-vertex API.
func StreamGoogleVertex(model *Model, conv *Context, options *GoogleVertexOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &GoogleVertexOptions{}
	}
	es := NewAssistantMessageEventStream()
	p := &vertexStreamer{model: model, conv: conv, options: options, es: es}
	go p.run()
	return es
}

// StreamGoogleVertexSimple maps SimpleStreamOptions onto the Vertex stream
// (streamSimple in google-vertex.ts).
func StreamGoogleVertexSimple(model *Model, conv *Context, options *SimpleStreamOptions) *AssistantMessageEventStream {
	if options == nil {
		options = &SimpleStreamOptions{}
	}
	base := buildBaseOptions(model, conv, options)

	if options.Reasoning == "" {
		return StreamGoogleVertex(model, conv, &GoogleVertexOptions{
			StreamOptions: base,
			Thinking:      &GoogleThinkingConfig{Enabled: false},
		})
	}

	effort := clampReasoning(ClampThinkingLevel(model, options.Reasoning))
	if effort == ThinkingOff {
		effort = ThinkingHigh
	}

	// Vertex checks only the Gemini 3 families here; Gemma is not offered.
	if isGemini3ProModel(model) || isGemini3FlashModel(model) {
		return StreamGoogleVertex(model, conv, &GoogleVertexOptions{
			StreamOptions: base,
			Thinking:      &GoogleThinkingConfig{Enabled: true, Level: getVertexThinkingLevel(effort, model)},
		})
	}

	budget := getVertexBudget(model, effort, options.ThinkingBudgets)
	return StreamGoogleVertex(model, conv, &GoogleVertexOptions{
		StreamOptions: base,
		Thinking:      &GoogleThinkingConfig{Enabled: true, BudgetTokens: &budget},
	})
}

// vertexStreamer owns one request/response cycle.
type vertexStreamer struct {
	model   *Model
	conv    *Context
	options *GoogleVertexOptions
	es      *AssistantMessageEventStream
	output  *AssistantMessage
}

func (p *vertexStreamer) run() {
	p.output = &AssistantMessage{
		Role:       "assistant",
		Content:    []ContentBlock{},
		API:        APIGoogleVertex,
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

func (p *vertexStreamer) snapshot() *AssistantMessage {
	cp := *p.output
	cp.Content = append([]ContentBlock{}, p.output.Content...)
	return &cp
}

func (p *vertexStreamer) push(event AssistantMessageEvent) {
	if event.Type != EventDone && event.Type != EventError {
		event.Partial = p.snapshot()
	}
	p.es.Push(event)
}

func (p *vertexStreamer) streamOnce() error {
	ctx := p.options.context()
	opts := p.options

	if ctx.Err() != nil {
		return AbortError{}
	}

	// An express-mode API key skips project/location entirely; otherwise the
	// request is signed with an ambient Google Cloud token.
	apiKey := resolveVertexAPIKey(opts)
	var (
		requestURL  string
		accessToken string
		err         error
	)
	if apiKey != "" {
		requestURL, err = buildVertexExpressURL(p.model)
	} else {
		requestURL, err = buildVertexResourceURL(p.model, opts)
		if err == nil {
			accessToken, err = resolveGoogleAccessToken(ctx, opts.Env)
		}
	}
	if err != nil {
		return err
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

	params := buildVertexParams(p.model, p.conv, opts)
	if opts.OnPayload != nil {
		if next := opts.OnPayload(params, p.model); next != nil {
			params = next
		}
	}

	resp, err := RetryProviderRequest(ctx, func(ctx context.Context) (*http.Response, error) {
		return doVertexRequest(ctx, requestURL, apiKey, accessToken, headers, params, opts.TimeoutMs)
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
		return fmt.Errorf("Google Vertex stream ended without a finish reason")
	}
	if p.output.StopReason == StopAborted || p.output.StopReason == StopError {
		if p.output.RawStopReason != "" {
			return fmt.Errorf("Provider stopped with: %s", p.output.RawStopReason)
		}
		return fmt.Errorf("An unknown error occurred")
	}

	p.es.Push(AssistantMessageEvent{Type: EventDone, Reason: p.output.StopReason, Message: p.snapshot()})
	p.es.End()
	return nil
}

// buildVertexParams builds the generateContent request body. Identical to the
// Generative Language build except for the options type.
func buildVertexParams(model *Model, conv *Context, options *GoogleVertexOptions) map[string]any {
	return buildGoogleParams(model, conv, &GoogleOptions{
		StreamOptions: options.StreamOptions,
		ToolChoice:    options.ToolChoice,
		Thinking:      options.Thinking,
	})
}

// vertexPlaceholderAPIKeyRe matches "<PLACEHOLDER>" style values.
var vertexPlaceholderAPIKeyRe = regexp.MustCompile(`^<[^>]+>$`)

// resolveVertexAPIKey returns an express-mode API key, or "" when the request
// should use ambient Google Cloud credentials.
func resolveVertexAPIKey(options *GoogleVertexOptions) string {
	apiKey := strings.TrimSpace(options.APIKey)
	if apiKey == "" || apiKey == gcpVertexCredentialsMarker || vertexPlaceholderAPIKeyRe.MatchString(apiKey) {
		return ""
	}
	return apiKey
}

// resolveVertexProject resolves the Google Cloud project id.
func resolveVertexProject(options *GoogleVertexOptions) (string, error) {
	if options.Project != "" {
		return options.Project, nil
	}
	if project := getProviderEnvValue("GOOGLE_CLOUD_PROJECT", options.Env); project != "" {
		return project, nil
	}
	if project := getProviderEnvValue("GCLOUD_PROJECT", options.Env); project != "" {
		return project, nil
	}
	return "", fmt.Errorf("Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.")
}

// resolveVertexLocation resolves the Vertex region.
func resolveVertexLocation(options *GoogleVertexOptions) (string, error) {
	if options.Location != "" {
		return options.Location, nil
	}
	if location := getProviderEnvValue("GOOGLE_CLOUD_LOCATION", options.Env); location != "" {
		return location, nil
	}
	return "", fmt.Errorf("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
}

// resolveVertexCustomBaseURL returns model.BaseURL when it is a usable
// override. The catalog stores a "{location}" template, which is a placeholder
// rather than a real endpoint.
func resolveVertexCustomBaseURL(baseURL string) string {
	trimmed := strings.TrimSpace(baseURL)
	if trimmed == "" || strings.Contains(trimmed, "{location}") {
		return ""
	}
	return trimmed
}

var vertexAPIVersionSegmentRe = regexp.MustCompile(`^v\d+(?:beta\d*)?$`)

// vertexBaseURLIncludesAPIVersion reports whether a custom base URL already
// carries a version segment, so one is not appended twice.
func vertexBaseURLIncludesAPIVersion(baseURL string) bool {
	for _, part := range strings.Split(strings.TrimSuffix(baseURL, "/"), "/") {
		if vertexAPIVersionSegmentRe.MatchString(part) {
			return true
		}
	}
	return false
}

// vertexHost builds the regional Vertex host. The "global" location uses the
// unprefixed host.
func vertexHost(location string) string {
	if location == "global" {
		return "https://aiplatform.googleapis.com"
	}
	return fmt.Sprintf("https://%s-aiplatform.googleapis.com", location)
}

// buildVertexResourceURL builds the full streaming URL for project/location
// credentials.
func buildVertexResourceURL(model *Model, options *GoogleVertexOptions) (string, error) {
	project, err := resolveVertexProject(options)
	if err != nil {
		return "", err
	}
	location, err := resolveVertexLocation(options)
	if err != nil {
		return "", err
	}

	base := resolveVertexCustomBaseURL(model.BaseURL)
	if base == "" {
		base = vertexHost(location) + "/" + vertexAPIVersion
	} else {
		base = strings.TrimSuffix(base, "/")
		if !vertexBaseURLIncludesAPIVersion(base) {
			base += "/" + vertexAPIVersion
		}
	}

	return fmt.Sprintf("%s/projects/%s/locations/%s/publishers/google/models/%s:streamGenerateContent?alt=sse",
		base, project, location, model.ID), nil
}

// buildVertexExpressURL builds the streaming URL for express-mode API keys,
// which address publisher models directly with no project or location.
func buildVertexExpressURL(model *Model) (string, error) {
	base := resolveVertexCustomBaseURL(model.BaseURL)
	if base == "" {
		base = "https://aiplatform.googleapis.com/" + vertexAPIVersion
	} else {
		base = strings.TrimSuffix(base, "/")
		if !vertexBaseURLIncludesAPIVersion(base) {
			base += "/" + vertexAPIVersion
		}
	}
	return fmt.Sprintf("%s/publishers/google/models/%s:streamGenerateContent?alt=sse", base, model.ID), nil
}

// doVertexRequest performs one streaming POST. Express mode authenticates with
// an API key; otherwise a Google Cloud bearer token is used.
func doVertexRequest(ctx context.Context, requestURL, apiKey, accessToken string, headers map[string]string, params map[string]any, timeoutMs int64) (*http.Response, error) {
	body, err := json.Marshal(googleRequestBody(params))
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	switch {
	case apiKey != "":
		req.Header.Set("x-goog-api-key", apiKey)
	case accessToken != "":
		req.Header.Set("Authorization", "Bearer "+accessToken)
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
			Message: fmt.Sprintf("POST %s failed with status %d", requestURL, resp.StatusCode),
		}
	}
	return resp, nil
}

// ---------------------------------------------------------------------------
// Thinking configuration
// ---------------------------------------------------------------------------

// getVertexThinkingLevel maps a pi thinking level onto a Gemini level. Unlike
// the Generative Language variant there is no Gemma branch.
func getVertexThinkingLevel(effort ThinkingLevel, model *Model) GoogleThinkingLevel {
	if isGemini3ProModel(model) {
		// Gemini 3 Pro only exposes LOW and HIGH.
		switch effort {
		case ThinkingMinimal, ThinkingLow:
			return GoogleThinkingLevelLow
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

// getVertexBudget resolves the thinking token budget. Vertex collapses
// flash-lite into the flash table, so 2.5-flash-lite gets the flash budgets.
func getVertexBudget(model *Model, effort ThinkingLevel, custom *ThinkingBudgets) int {
	if custom != nil {
		if budget, ok := lookupThinkingBudget(custom, effort); ok {
			return budget
		}
	}
	switch {
	case strings.Contains(model.ID, "2.5-pro"):
		return pickGoogleBudget(effort, 128, 2048, 8192, 32768)
	case strings.Contains(model.ID, "2.5-flash"):
		return pickGoogleBudget(effort, 128, 2048, 8192, 24576)
	default:
		return -1
	}
}
