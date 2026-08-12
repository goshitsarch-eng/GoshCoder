// Native provider transports adapted from github.com/nicobailon/pi-web-access
// openai-search.ts, exa.ts, and kagi.ts. Requests are bounded, cancellable, and
// redact keys from provider error bodies before returning them to the model.
package webaccess

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

var exaBlockPattern = regexp.MustCompile(`(?m)(?:^|\n)Title:\s*`)

func (s *Service) searchExa(ctx context.Context, cfg configFile, query string, options Options) (Response, error) {
	apiKey := s.credential(cfg.ExaAPIKey, "EXA_API_KEY")
	if apiKey == "" {
		return s.searchExaMCP(ctx, query, options)
	}
	return s.searchExaAPI(ctx, apiKey, query, options)
}

func (s *Service) searchExaMCP(ctx context.Context, query string, options Options) (Response, error) {
	searchQuery := constrainedQuery(query, options)
	endpoint, err := url.Parse(s.exaMCPURL)
	if err != nil {
		return Response{}, fmt.Errorf("invalid Exa MCP URL: %w", err)
	}
	values := endpoint.Query()
	values.Set("tools", "web_search_exa")
	endpoint.RawQuery = values.Encode()
	body := map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params": map[string]any{
			"name": "web_search_exa",
			"arguments": map[string]any{
				"query": searchQuery, "numResults": options.NumResults,
			},
		},
	}
	raw, err := s.doJSON(ctx, http.MethodPost, endpoint.String(), body, map[string]string{
		"Accept":       "application/json, text/event-stream",
		"x-exa-source": "pi-web-access",
	}, "")
	if err != nil {
		return Response{}, fmt.Errorf("Exa MCP search: %w", err)
	}
	text, err := parseExaMCPEnvelope(raw)
	if err != nil {
		return Response{}, err
	}
	results := parseExaTextResults(text, options.NumResults)
	if len(results) == 0 {
		return Response{}, errors.New("Exa MCP returned no parseable results")
	}
	return Response{Provider: "exa", Query: query, Answer: answerFromResults(results), Results: results}, nil
}

func parseExaMCPEnvelope(raw []byte) (string, error) {
	var candidates [][]byte
	for _, line := range bytes.Split(raw, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if bytes.HasPrefix(line, []byte("data:")) {
			if value := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:"))); len(value) > 0 {
				candidates = append(candidates, value)
			}
		}
	}
	candidates = append(candidates, bytes.TrimSpace(raw))
	for _, candidate := range candidates {
		var envelope struct {
			Result struct {
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
				IsError bool `json:"isError"`
			} `json:"result"`
			Error *struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(candidate, &envelope) != nil {
			continue
		}
		if envelope.Error != nil {
			return "", fmt.Errorf("Exa MCP error %d: %s", envelope.Error.Code, envelope.Error.Message)
		}
		for _, content := range envelope.Result.Content {
			if content.Type == "text" && strings.TrimSpace(content.Text) != "" {
				if envelope.Result.IsError {
					return "", errors.New(compactText(content.Text, 500))
				}
				return content.Text, nil
			}
		}
	}
	return "", errors.New("Exa MCP returned an empty response")
}

func parseExaTextResults(text string, limit int) []Result {
	starts := exaBlockPattern.FindAllStringIndex(text, -1)
	results := make([]Result, 0, min(len(starts), limit))
	for index, match := range starts {
		start := match[1]
		end := len(text)
		if index+1 < len(starts) {
			end = starts[index+1][0]
		}
		block := strings.TrimSpace(text[start:end])
		lines := strings.Split(block, "\n")
		if len(lines) == 0 {
			continue
		}
		result := Result{Title: strings.TrimSpace(lines[0])}
		contentAt := -1
		for lineIndex, line := range lines[1:] {
			line = strings.TrimSpace(line)
			switch {
			case strings.HasPrefix(line, "URL:"):
				result.URL = strings.TrimSpace(strings.TrimPrefix(line, "URL:"))
			case strings.HasPrefix(line, "Text:"):
				contentAt = lineIndex + 2
			case strings.HasPrefix(line, "Highlights:"):
				contentAt = lineIndex + 2
			}
		}
		if !validResultURL(result.URL) {
			continue
		}
		if contentAt >= 0 && contentAt < len(lines) {
			result.Snippet = compactText(strings.Join(lines[contentAt:], " "), 1_000)
		}
		results = append(results, result)
		if len(results) >= limit {
			break
		}
	}
	return results
}

func (s *Service) searchExaAPI(ctx context.Context, apiKey, query string, options Options) (Response, error) {
	body := map[string]any{
		"query": query, "type": "auto", "numResults": options.NumResults,
		"contents": map[string]any{"highlights": true},
	}
	applyExaFilters(body, options)
	if options.IncludeContent {
		body["contents"] = map[string]any{"highlights": true, "text": true}
	}
	raw, err := s.doJSON(ctx, http.MethodPost, s.exaSearchURL, body, map[string]string{
		"x-api-key": apiKey, "x-exa-integration": "pi-web-access",
	}, apiKey)
	if err != nil {
		return Response{}, fmt.Errorf("Exa API search: %w", err)
	}
	var payload struct {
		Results []struct {
			Title      string   `json:"title"`
			URL        string   `json:"url"`
			Text       string   `json:"text"`
			Highlights []string `json:"highlights"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return Response{}, fmt.Errorf("Exa API returned invalid JSON: %w", err)
	}
	results := make([]Result, 0, min(len(payload.Results), options.NumResults))
	for _, item := range payload.Results {
		if !validResultURL(item.URL) {
			continue
		}
		snippet := strings.Join(item.Highlights, " ")
		if snippet == "" {
			snippet = item.Text
		}
		results = append(results, Result{Title: item.Title, URL: item.URL, Snippet: compactText(snippet, 1_000)})
		if len(results) >= options.NumResults {
			break
		}
	}
	if len(results) == 0 {
		return Response{}, errors.New("Exa API returned no results")
	}
	return Response{Provider: "exa", Query: query, Answer: answerFromResults(results), Results: results}, nil
}

func applyExaFilters(body map[string]any, options Options) {
	var included, excluded []string
	for _, raw := range options.DomainFilter {
		domain, blocked := normalizeDomain(raw)
		if domain == "" {
			continue
		}
		if blocked {
			excluded = append(excluded, domain)
		} else {
			included = append(included, domain)
		}
	}
	if len(included) > 0 {
		body["includeDomains"] = included
	}
	if len(excluded) > 0 {
		body["excludeDomains"] = excluded
	}
	if options.RecencyFilter != "" {
		body["startPublishedDate"] = recencyStart(options.RecencyFilter)
	}
}

func (s *Service) searchKagi(ctx context.Context, cfg configFile, query string, options Options) (Response, error) {
	apiKey := s.credential(cfg.KagiAPIKey, "KAGI_API_KEY")
	if apiKey == "" {
		return Response{}, fmt.Errorf("Kagi API key not found; set KAGI_API_KEY or kagiApiKey in %s", s.ConfigPath)
	}
	raw, err := s.doJSON(ctx, http.MethodPost, s.kagiSearchURL, map[string]any{
		"query": query, "limit": options.NumResults,
	}, map[string]string{"Authorization": "Bearer " + apiKey, "Accept": "application/json"}, apiKey)
	if err != nil {
		return Response{}, fmt.Errorf("Kagi API search: %w", err)
	}
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return Response{}, fmt.Errorf("Kagi API returned invalid JSON: %w", err)
	}
	if message := kagiError(payload); message != "" {
		return Response{}, fmt.Errorf("Kagi API returned an error: %s", message)
	}
	var results []Result
	appendKagiResults(kagiData(payload), &results, options.NumResults)
	if len(results) == 0 {
		return Response{}, errors.New("Kagi API returned no results")
	}
	return Response{Provider: "kagi", Query: query, Answer: answerFromResults(results), Results: results}, nil
}

func kagiData(payload any) any {
	record, ok := payload.(map[string]any)
	if !ok {
		return nil
	}
	data := record["data"]
	if nested, ok := data.(map[string]any); ok {
		if search, exists := nested["search"]; exists {
			return search
		}
	}
	return data
}

func kagiError(payload any) string {
	record, ok := payload.(map[string]any)
	if !ok {
		return ""
	}
	value := record["errors"]
	if value == nil {
		value = record["error"]
	}
	var messages []string
	appendErrorMessages(value, &messages)
	return strings.Join(messages, "; ")
}

func appendErrorMessages(value any, messages *[]string) {
	switch item := value.(type) {
	case []any:
		for _, child := range item {
			appendErrorMessages(child, messages)
		}
	case map[string]any:
		for _, field := range []string{"message", "msg", "code"} {
			if text := stringValue(item[field]); text != "" {
				*messages = append(*messages, text)
				return
			}
		}
	case string:
		if strings.TrimSpace(item) != "" {
			*messages = append(*messages, strings.TrimSpace(item))
		}
	}
}

func appendKagiResults(value any, results *[]Result, limit int) {
	if len(*results) >= limit {
		return
	}
	switch item := value.(type) {
	case []any:
		for _, child := range item {
			appendKagiResults(child, results, limit)
			if len(*results) >= limit {
				return
			}
		}
	case map[string]any:
		resultURL := firstStringValue(item, "url", "href", "link")
		if validResultURL(resultURL) {
			title := firstStringValue(item, "title", "name")
			if title == "" {
				title = resultURL
			}
			snippet := firstStringValue(item, "snippet", "description", "summary", "content", "markdown", "text")
			*results = append(*results, Result{Title: title, URL: resultURL, Snippet: compactText(snippet, 1_000)})
		}
	}
}

func (s *Service) searchOpenAI(ctx context.Context, cfg configFile, query string, options Options) (Response, bool, error) {
	auth, err := s.resolveOpenAI(ctx, cfg)
	if err != nil {
		return Response{}, false, err
	}
	if auth == nil || auth.APIKey == "" {
		return Response{}, false, nil
	}
	endpoint := s.openAIURL
	if configured := strings.TrimSpace(cfg.OpenAIResponsesURL); configured != "" {
		parsed, parseErr := url.Parse(configured)
		if parseErr != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return Response{}, true, fmt.Errorf("openaiResponsesUrl in %s must be an absolute HTTP(S) URL", s.ConfigPath)
		}
		endpoint = parsed.String()
	}
	if auth.Provider == "openai-codex" || isCodexJWT(auth.APIKey) {
		endpoint = s.codexURL
	}
	model := firstNonempty(cfg.OpenAISearchModel, auth.Model, defaultSearchModel)
	headers := map[string]string{
		"Authorization": "Bearer " + auth.APIKey,
		"OpenAI-Beta":   "responses=experimental",
	}
	for name, value := range auth.Headers {
		if value == nil {
			deleteHeaderFold(headers, name)
		} else {
			headers[name] = *value
		}
	}
	if auth.Provider == "openai-codex" || isCodexJWT(auth.APIKey) {
		if accountID := codexAccountID(auth.APIKey); accountID != "" {
			headers["chatgpt-account-id"] = accountID
		}
		headers["originator"] = "goshcoder"
	}
	body := map[string]any{
		"model":        model,
		"instructions": buildOpenAIInstructions(options),
		"input": []any{map[string]any{
			"role": "user", "content": []any{map[string]any{"type": "input_text", "text": query}},
		}},
		"tools":   []any{buildOpenAIWebTool(options)},
		"include": []string{"web_search_call.action.sources"},
		"store":   false, "stream": true, "tool_choice": "required", "parallel_tool_calls": true,
	}
	raw, err := s.doJSON(ctx, http.MethodPost, endpoint, body, headers, auth.APIKey)
	if err != nil {
		return Response{}, true, fmt.Errorf("OpenAI web search: %w", err)
	}
	output, err := parseOpenAIOutput(raw)
	if err != nil {
		return Response{}, true, err
	}
	answer := openAIAnswer(output)
	results := openAIResults(output, options.NumResults)
	if answer == "" && len(results) == 0 {
		return Response{}, true, errors.New("OpenAI web search returned no answer or sources")
	}
	return Response{Provider: "openai", Query: query, Answer: answer, Results: results}, true, nil
}

func (s *Service) resolveOpenAI(ctx context.Context, cfg configFile) (*OpenAIAuth, error) {
	var resolverErr error
	if s.ResolveOpenAI != nil {
		auth, err := s.ResolveOpenAI(ctx)
		resolverErr = err
		if auth != nil && auth.APIKey != "" {
			copy := *auth
			return &copy, nil
		}
	}
	apiKey := s.credential(cfg.OpenAIAPIKey, "OPENAI_API_KEY")
	if apiKey == "" {
		return nil, resolverErr
	}
	return &OpenAIAuth{Provider: "openai", APIKey: apiKey, Model: firstNonempty(cfg.OpenAISearchModel, defaultSearchModel)}, nil
}

func buildOpenAIInstructions(options Options) string {
	lines := []string{
		"Search the web and return a concise answer grounded only in the web results.",
		"Include clickable source citations in the response text when possible.",
	}
	if options.RecencyFilter != "" {
		lines = append(lines, "Prefer sources from the past "+options.RecencyFilter+".")
	}
	lines = append(lines, fmt.Sprintf("Prefer around %d distinct sources.", options.NumResults))
	var allowed, blocked []string
	for _, raw := range options.DomainFilter {
		domain, excluded := normalizeDomain(raw)
		if domain == "" {
			continue
		}
		if excluded {
			blocked = append(blocked, domain)
		} else {
			allowed = append(allowed, domain)
		}
	}
	if len(allowed) > 0 {
		lines = append(lines, "Only use sources from: "+strings.Join(allowed, ", ")+".")
	}
	if len(blocked) > 0 {
		lines = append(lines, "Do not use sources from: "+strings.Join(blocked, ", ")+".")
	}
	return strings.Join(lines, " ")
}

func buildOpenAIWebTool(options Options) map[string]any {
	tool := map[string]any{"type": "web_search"}
	var allowed, blocked []string
	for _, raw := range options.DomainFilter {
		domain, excluded := normalizeDomain(raw)
		if domain == "" {
			continue
		}
		if excluded {
			blocked = append(blocked, domain)
		} else {
			allowed = append(allowed, domain)
		}
	}
	if len(allowed) > 0 || len(blocked) > 0 {
		filters := map[string]any{}
		if len(allowed) > 0 {
			filters["allowed_domains"] = allowed
		}
		if len(blocked) > 0 {
			filters["blocked_domains"] = blocked
		}
		tool["filters"] = filters
	}
	return tool
}

func parseOpenAIOutput(raw []byte) ([]any, error) {
	trimmed := bytes.TrimSpace(raw)
	var direct map[string]any
	if len(trimmed) > 0 && trimmed[0] == '{' && json.Unmarshal(trimmed, &direct) == nil {
		if output, ok := direct["output"].([]any); ok {
			return output, nil
		}
	}
	var output []any
	var completed []any
	for _, line := range bytes.Split(raw, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		var event map[string]any
		if json.Unmarshal(data, &event) != nil {
			continue
		}
		switch event["type"] {
		case "response.output_item.done":
			if item := event["item"]; item != nil {
				output = append(output, item)
			}
		case "response.done", "response.completed":
			if response, ok := event["response"].(map[string]any); ok {
				if items, ok := response["output"].([]any); ok && len(items) > 0 {
					completed = items
				}
			}
		}
	}
	if len(completed) > 0 {
		return completed, nil
	}
	if len(output) > 0 {
		return output, nil
	}
	return nil, errors.New("OpenAI API returned no parseable response output")
}

func openAIAnswer(output []any) string {
	var parts []string
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok || item["type"] != "message" {
			continue
		}
		for _, rawContent := range anySlice(item["content"]) {
			content, ok := rawContent.(map[string]any)
			if !ok {
				continue
			}
			if text := stringValue(content["text"]); text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func openAIResults(output []any, limit int) []Result {
	var results []Result
	seen := map[string]bool{}
	add := func(rawURL, title, snippet any) {
		resultURL := cleanOpenAIURL(stringValue(rawURL))
		if !validResultURL(resultURL) || seen[resultURL] || len(results) >= limit {
			return
		}
		seen[resultURL] = true
		results = append(results, Result{Title: stringValue(title), URL: resultURL, Snippet: compactText(stringValue(snippet), 500)})
	}
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if item["type"] == "message" {
			for _, rawContent := range anySlice(item["content"]) {
				content, ok := rawContent.(map[string]any)
				if !ok {
					continue
				}
				for _, rawAnnotation := range anySlice(content["annotations"]) {
					annotation, ok := rawAnnotation.(map[string]any)
					if ok && annotation["type"] == "url_citation" {
						add(annotation["url"], annotation["title"], "")
					}
				}
			}
		}
		if item["type"] == "web_search_call" {
			groups := []any{item["sources"], item["results"]}
			if action, ok := item["action"].(map[string]any); ok {
				groups = append(groups, action["sources"])
			}
			for _, group := range groups {
				for _, rawSource := range anySlice(group) {
					source, ok := rawSource.(map[string]any)
					if ok {
						add(firstAny(source["url"], source["source_website_url"]), firstAny(source["title"], source["caption"]), "")
					}
				}
			}
		}
	}
	return results
}

func (s *Service) doJSON(ctx context.Context, method, endpoint string, body any, headers map[string]string, secret string) ([]byte, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	requestCtx, cancel := context.WithTimeout(ctx, searchTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(requestCtx, method, endpoint, bytes.NewReader(encoded))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "goshcoder-web-access/1.0 (+https://github.com/goshitsarch-eng/GoshCoder)")
	for name, value := range headers {
		req.Header.Set(name, value)
	}
	resp, err := s.client().Do(req)
	if err != nil {
		return nil, redactError(err, secret)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBody+1))
	if err != nil {
		return nil, redactError(err, secret)
	}
	if len(raw) > maxResponseBody {
		return nil, errors.New("search provider response exceeds 8 MiB")
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message := compactText(redact(string(raw), secret), 300)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, message)
	}
	return raw, nil
}

func constrainedQuery(query string, options Options) string {
	parts := []string{query}
	for _, raw := range options.DomainFilter {
		domain, excluded := normalizeDomain(raw)
		if domain == "" {
			continue
		}
		if excluded {
			parts = append(parts, "-site:"+domain)
		} else {
			parts = append(parts, "site:"+domain)
		}
	}
	if options.RecencyFilter != "" {
		parts = append(parts, "past "+options.RecencyFilter)
	}
	return strings.Join(parts, " ")
}

func normalizeDomain(raw string) (domain string, excluded bool) {
	value := strings.ToLower(strings.TrimSpace(raw))
	excluded = strings.HasPrefix(value, "-")
	value = strings.TrimSpace(strings.TrimPrefix(value, "-"))
	if value == "" {
		return "", excluded
	}
	if !strings.Contains(value, "://") {
		value = "https://" + value
	}
	parsed, err := url.Parse(value)
	if err != nil || parsed.Hostname() == "" {
		return "", excluded
	}
	return strings.Trim(parsed.Hostname(), "."), excluded
}

func recencyStart(filter string) string {
	days := map[string]int{"day": 1, "week": 7, "month": 30, "year": 365}[filter]
	return time.Now().AddDate(0, 0, -days).UTC().Format("2006-01-02T15:04:05Z")
}

func answerFromResults(results []Result) string {
	var parts []string
	for _, result := range results {
		if result.Snippet != "" {
			parts = append(parts, result.Snippet+"\nSource: "+fallbackTitle(result)+" ("+result.URL+")")
		} else {
			parts = append(parts, "Source: "+fallbackTitle(result)+" ("+result.URL+")")
		}
	}
	return strings.Join(parts, "\n\n")
}

func validResultURL(value string) bool {
	parsed, err := url.Parse(strings.TrimSpace(value))
	return err == nil && parsed.Host != "" && (parsed.Scheme == "http" || parsed.Scheme == "https")
}

func stringValue(value any) string {
	text, _ := value.(string)
	return strings.TrimSpace(text)
}

func firstStringValue(record map[string]any, names ...string) string {
	for _, name := range names {
		if value := stringValue(record[name]); value != "" {
			return value
		}
	}
	return ""
}

func anySlice(value any) []any {
	values, _ := value.([]any)
	return values
}

func firstAny(values ...any) any {
	for _, value := range values {
		if value != nil && stringValue(value) != "" {
			return value
		}
	}
	return nil
}

func cleanOpenAIURL(value string) string {
	parsed, err := url.Parse(value)
	if err != nil {
		return value
	}
	if parsed.Query().Get("utm_source") == "openai" {
		query := parsed.Query()
		query.Del("utm_source")
		parsed.RawQuery = query.Encode()
	}
	return parsed.String()
}

func deleteHeaderFold(headers map[string]string, target string) {
	for name := range headers {
		if strings.EqualFold(name, target) {
			delete(headers, name)
		}
	}
}

func isCodexJWT(token string) bool { return codexAccountID(token) != "" }

func codexAccountID(token string) string {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return ""
	}
	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return ""
	}
	var claims map[string]any
	if json.Unmarshal(payload, &claims) != nil {
		return ""
	}
	auth, _ := claims["https://api.openai.com/auth"].(map[string]any)
	return stringValue(auth["chatgpt_account_id"])
}

func redact(value, secret string) string {
	if secret == "" {
		return value
	}
	return strings.ReplaceAll(value, secret, "[REDACTED]")
}

func redactError(err error, secret string) error {
	if err == nil || secret == "" {
		return err
	}
	return errors.New(redact(err.Error(), secret))
}
