package aperture

// MCP Streamable HTTP client for Aperture connectors (src/mcp-client.ts):
// session initialization, capability discovery, tool calls, and resource
// reads through the gateway's /v1/mcp endpoint using the 2024-11-05
// protocol.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync/atomic"
	"time"
)

const (
	mcpVersion     = "2024-11-05"
	mcpInitTimeout = 10 * time.Second
	mcpCallTimeout = 60 * time.Second
)

// McpTool is one connector tool as reported by tools/list.
type McpTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// McpResource is one resources/list entry.
type McpResource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

// McpResourceTemplate is one resources/templates/list entry.
type McpResourceTemplate struct {
	URITemplate string `json:"uriTemplate"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

// McpResourceContent is one resources/read content item.
type McpResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType,omitempty"`
	Text     string `json:"text,omitempty"`
	Blob     string `json:"blob,omitempty"`
}

// McpContentItem is one tools/call result content item.
type McpContentItem struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// McpCallResult is a tools/call result.
type McpCallResult struct {
	Content []McpContentItem `json:"content"`
	IsError bool             `json:"isError,omitempty"`
}

type jsonRPCRequest struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
	ID      *int64 `json:"id,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// McpSession is one initialized connector session. There is no explicit
// close in Streamable HTTP; the session expires server-side.
type McpSession struct {
	url       string
	sessionID string
	client    *http.Client
	nextID    atomic.Int64
}

var sseDataLine = regexp.MustCompile(`(?s)data: (.+)`)

func postJSONRPC(ctx context.Context, client *http.Client, url string, body jsonRPCRequest, sessionID string, timeout time.Duration) (jsonRPCResponse, string, error) {
	requestCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	encoded, err := json.Marshal(body)
	if err != nil {
		return jsonRPCResponse{}, "", err
	}
	req, err := http.NewRequestWithContext(requestCtx, http.MethodPost, url, bytes.NewReader(encoded))
	if err != nil {
		return jsonRPCResponse{}, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if sessionID != "" {
		req.Header.Set("Mcp-Session-Id", sessionID)
	}
	response, err := client.Do(req)
	if err != nil {
		return jsonRPCResponse{}, "", err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return jsonRPCResponse{}, "", fmt.Errorf("MCP request failed: HTTP %d %s", response.StatusCode, http.StatusText(response.StatusCode))
	}
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return jsonRPCResponse{}, "", err
	}
	if len(payload) > maxResponseBytes {
		return jsonRPCResponse{}, "", fmt.Errorf("MCP response exceeds %d bytes", maxResponseBytes)
	}
	responseSessionID := response.Header.Get("Mcp-Session-Id")

	// Responses may be SSE-framed ("event: message\ndata: {...}").
	text := string(payload)
	dataText := strings.TrimSpace(text)
	if match := sseDataLine.FindStringSubmatch(text); match != nil {
		dataText = strings.TrimSpace(match[1])
	}
	var parsed jsonRPCResponse
	if err := json.Unmarshal([]byte(dataText), &parsed); err != nil {
		preview := dataText
		if len(preview) > 200 {
			preview = preview[:200]
		}
		return jsonRPCResponse{}, "", fmt.Errorf("MCP response is not valid JSON: %s", preview)
	}
	if parsed.Error != nil {
		return jsonRPCResponse{}, "", fmt.Errorf("MCP error: %s (code %d)", parsed.Error.Message, parsed.Error.Code)
	}
	return parsed, responseSessionID, nil
}

// NewMcpSession initializes a connector session against the gateway's
// /v1/mcp endpoint and sends the best-effort initialized notification.
func NewMcpSession(ctx context.Context, baseURL string) (*McpSession, error) {
	url := strings.TrimRight(baseURL, "/") + "/v1/mcp"
	client := &http.Client{Timeout: mcpCallTimeout}

	one := int64(1)
	initResponse, sessionID, err := postJSONRPC(ctx, client, url, jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  "initialize",
		Params: map[string]any{
			"protocolVersion": mcpVersion,
			"capabilities":    map[string]any{},
			// The upstream extension identifies as pi-ts-aperture; the native
			// adaptation names itself so gateway logs stay truthful.
			"clientInfo": map[string]any{"name": "goshcoder-aperture", "version": "1"},
		},
		ID: &one,
	}, "", mcpInitTimeout)
	if err != nil {
		return nil, err
	}
	if sessionID == "" {
		return nil, errors.New("MCP initialize response missing Mcp-Session-Id header")
	}
	var initResult struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if json.Unmarshal(initResponse.Result, &initResult) != nil || initResult.ProtocolVersion != mcpVersion {
		return nil, fmt.Errorf("MCP initialize returned unexpected result: %s", string(initResponse.Result))
	}

	session := &McpSession{url: url, sessionID: sessionID, client: client}
	session.nextID.Store(1)

	// notifications/initialized is fire-and-forget.
	notifyCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	encoded, _ := json.Marshal(jsonRPCRequest{JSONRPC: "2.0", Method: "notifications/initialized"})
	if req, err := http.NewRequestWithContext(notifyCtx, http.MethodPost, url, bytes.NewReader(encoded)); err == nil {
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json, text/event-stream")
		req.Header.Set("Mcp-Session-Id", sessionID)
		if response, err := client.Do(req); err == nil {
			io.Copy(io.Discard, io.LimitReader(response.Body, 1<<20))
			response.Body.Close()
		}
	}
	return session, nil
}

func (s *McpSession) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := s.nextID.Add(1)
	response, _, err := postJSONRPC(ctx, s.client, s.url, jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
		ID:      &id,
	}, s.sessionID, mcpCallTimeout)
	if err != nil {
		return nil, err
	}
	return response.Result, nil
}

// ListTools returns the gateway's connector tools.
func (s *McpSession) ListTools(ctx context.Context) ([]McpTool, error) {
	result, err := s.call(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}
	var decoded struct {
		Tools []McpTool `json:"tools"`
	}
	if err := json.Unmarshal(result, &decoded); err != nil {
		return nil, fmt.Errorf("decode MCP tools/list: %w", err)
	}
	return decoded.Tools, nil
}

// CallTool executes one connector tool.
func (s *McpSession) CallTool(ctx context.Context, name string, args map[string]any) (McpCallResult, error) {
	if args == nil {
		args = map[string]any{}
	}
	result, err := s.call(ctx, "tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return McpCallResult{}, err
	}
	if len(result) == 0 || string(result) == "null" {
		return McpCallResult{}, fmt.Errorf("MCP tools/call returned empty result for %s", name)
	}
	var decoded McpCallResult
	if err := json.Unmarshal(result, &decoded); err != nil {
		return McpCallResult{}, fmt.Errorf("decode MCP tools/call: %w", err)
	}
	return decoded, nil
}

// ListResources returns the connector resources.
func (s *McpSession) ListResources(ctx context.Context) ([]McpResource, error) {
	result, err := s.call(ctx, "resources/list", nil)
	if err != nil {
		return nil, err
	}
	var decoded struct {
		Resources []McpResource `json:"resources"`
	}
	if err := json.Unmarshal(result, &decoded); err != nil {
		return nil, fmt.Errorf("decode MCP resources/list: %w", err)
	}
	return decoded.Resources, nil
}

// ListResourceTemplates returns the connector resource templates.
func (s *McpSession) ListResourceTemplates(ctx context.Context) ([]McpResourceTemplate, error) {
	result, err := s.call(ctx, "resources/templates/list", nil)
	if err != nil {
		return nil, err
	}
	var decoded struct {
		ResourceTemplates []McpResourceTemplate `json:"resourceTemplates"`
	}
	if err := json.Unmarshal(result, &decoded); err != nil {
		return nil, fmt.Errorf("decode MCP resources/templates/list: %w", err)
	}
	return decoded.ResourceTemplates, nil
}

// ReadResource reads one connector resource by URI.
func (s *McpSession) ReadResource(ctx context.Context, uri string) ([]McpResourceContent, error) {
	result, err := s.call(ctx, "resources/read", map[string]any{"uri": uri})
	if err != nil {
		return nil, err
	}
	if len(result) == 0 || string(result) == "null" {
		return nil, fmt.Errorf("MCP resources/read returned empty result for %s", uri)
	}
	var decoded struct {
		Contents []McpResourceContent `json:"contents"`
	}
	if err := json.Unmarshal(result, &decoded); err != nil {
		return nil, fmt.Errorf("decode MCP resources/read: %w", err)
	}
	return decoded.Contents, nil
}
