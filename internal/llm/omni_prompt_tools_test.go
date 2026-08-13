package llm

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOmniPromptToolsConvertsTextBlocksToNativeCalls(t *testing.T) {
	var requestBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data := make([]byte, r.ContentLength)
		_, _ = r.Body.Read(data)
		requestBody = string(data)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"id":"omni-1","choices":[{"index":0,"delta":{"role":"assistant","content":"I will inspect it.\n<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"main.go\"}}\n</tool_call>"},"finish_reason":null}]}

data: {"id":"omni-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}

data: [DONE]

`)
	}))
	defer server.Close()
	model := &Model{ID: "chat-web/model", Name: "Chat Web", API: APIOmniPromptTools, Provider: "omni", BaseURL: server.URL, Input: []string{"text"}, ContextWindow: 128000, MaxTokens: 4096}
	request := &Context{SystemPrompt: "base", Messages: []Message{UserMessage{Role: "user", Content: "inspect"}}, Tools: []Tool{{Name: "read", Description: "Read a file", Parameters: []byte(`{"type":"object","properties":{"path":{"type":"string"}}}`)}}}
	options := &SimpleStreamOptions{}
	options.APIKey = "key"
	result, err := streamOmniPromptTools(model, request, options).Result(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopToolUse || len(result.Content) != 2 {
		t.Fatalf("result = %#v", result)
	}
	call, ok := result.Content[1].(ToolCall)
	if !ok || call.Name != "read" || call.Arguments["path"] != "main.go" {
		t.Fatalf("call = %#v", result.Content[1])
	}
	if !strings.Contains(requestBody, "Tool calling protocol") || strings.Contains(requestBody, `\"tools\":[{`) {
		t.Fatalf("request did not use prompt protocol: %s", requestBody)
	}
}

func TestParseOmniToolCallsToleratesFenceAndStringArguments(t *testing.T) {
	text := "before <tool_call>```json\n{\"name\":\"bash\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}\n```</tool_call> after"
	prose, calls, problems := parseOmniToolCalls(text)
	if prose != "before  after" || len(problems) != 0 || len(calls) != 1 || calls[0].Arguments["command"] != "pwd" {
		t.Fatalf("prose=%q calls=%#v problems=%v", prose, calls, problems)
	}
}
