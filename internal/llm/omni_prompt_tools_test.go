package llm

import (
	"encoding/json"
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

// omniServer serves one completions SSE response for the prompt-tool adapter.
func omniServer(t *testing.T, content string, finishReason string) (*Model, *Context, *SimpleStreamOptions) {
	t.Helper()
	encoded, err := json.Marshal(content)
	if err != nil {
		t.Fatal(err)
	}
	body := `data: {"id":"omni-1","choices":[{"index":0,"delta":{"role":"assistant","content":` + string(encoded) +
		`},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"omni-1","choices":[{"index":0,"delta":{},"finish_reason":"` + finishReason + `"}]}` + "\n\n" +
		"data: [DONE]\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, body)
	}))
	t.Cleanup(server.Close)

	model := &Model{ID: "chat-web/model", Name: "Chat Web", API: APIOmniPromptTools, Provider: "omni",
		BaseURL: server.URL, Input: []string{"text"}, ContextWindow: 128000, MaxTokens: 4096}
	request := &Context{
		SystemPrompt: "base",
		Messages:     []Message{UserMessage{Role: "user", Content: "inspect"}},
		Tools:        []Tool{{Name: "read", Description: "Read a file", Parameters: []byte(`{"type":"object","properties":{"path":{"type":"string"}}}`)}},
	}
	options := &SimpleStreamOptions{}
	options.APIKey = "key"
	return model, request, options
}

// TestOmniPromptToolsPreservesTruncationStopReason covers a stop reason that
// was being overwritten. The adapter hard-set StopStop/StopToolUse, discarding
// StopLength from the inner stream, so the agent loop's guard against tool
// calls parsed out of a truncated response never fired for this API and the
// model's half-written call was executed as if it were complete.
func TestOmniPromptToolsPreservesTruncationStopReason(t *testing.T) {
	model, request, options := omniServer(t,
		`<tool_call>{"name":"read","arguments":{"path":"a.go"}}</tool_call>`, "length")

	result, err := streamOmniPromptTools(model, request, options).Result(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopLength {
		t.Fatalf("stop reason = %q, want StopLength to survive so the truncation guard fires", result.StopReason)
	}
}

// TestOmniPromptToolsStillReportsToolUse is the negative control: an untruncated
// response carrying tool calls must still say so.
func TestOmniPromptToolsStillReportsToolUse(t *testing.T) {
	model, request, options := omniServer(t,
		`<tool_call>{"name":"read","arguments":{"path":"a.go"}}</tool_call>`, "stop")

	result, err := streamOmniPromptTools(model, request, options).Result(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if result.StopReason != StopToolUse {
		t.Fatalf("stop reason = %q, want StopToolUse", result.StopReason)
	}
}

// TestOmniPromptToolsPublishesSnapshots keeps the adapter from handing the
// consumer the message it is still writing to. Without a copy, a reader on
// another goroutine -- which is exactly how the agent loop consumes -- sees
// Content grow and StopReason flip underneath it.
func TestOmniPromptToolsPublishesSnapshots(t *testing.T) {
	model, request, options := omniServer(t,
		"prose here\n"+`<tool_call>{"name":"read","arguments":{"path":"a.go"}}</tool_call>`, "stop")

	events := drainEvents(streamOmniPromptTools(model, request, options))

	var first *AssistantMessage
	for _, event := range events {
		if event.Type == EventTextStart && event.Partial != nil {
			first = event.Partial
			break
		}
	}
	if first == nil {
		t.Fatal("no text_start event carried a partial message")
	}
	// The tool call is appended after text_start is pushed, so a partial that
	// already contains it is the live message rather than a snapshot of it.
	for _, block := range first.Content {
		if _, ok := block.(ToolCall); ok {
			t.Fatal("text_start's partial mutated after publication: it is the live message, not a copy")
		}
	}
	if first.StopReason == StopToolUse {
		t.Fatal("text_start's partial saw a stop reason written after it was published")
	}
}
