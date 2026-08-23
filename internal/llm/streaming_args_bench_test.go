package llm

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

// largeToolCallStream builds an SSE body streaming one write-tool call whose
// arguments carry a ~54 KB file body, arriving in token-sized chunks the way a
// provider actually sends them.
func largeToolCallStream(tb testing.TB) (body string, chunks int) {
	tb.Helper()
	fileBody := strings.Repeat("some source code line here\n", 2000)
	arguments, err := json.Marshal(map[string]any{"path": "main.go", "content": fileBody})
	if err != nil {
		tb.Fatal(err)
	}

	var sse strings.Builder
	sse.WriteString(`data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"write","arguments":""}}]},"finish_reason":null}]}` + "\n\n")

	const chunkSize = 4 // roughly one token
	payload := string(arguments)
	for i := 0; i < len(payload); i += chunkSize {
		end := min(i+chunkSize, len(payload))
		fragment, err := json.Marshal(payload[i:end])
		if err != nil {
			tb.Fatal(err)
		}
		sse.WriteString(`data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":` +
			string(fragment) + `}}]},"finish_reason":null}]}` + "\n\n")
		chunks++
	}
	sse.WriteString(`data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}` + "\n\n")
	sse.WriteString("data: [DONE]\n\n")
	return sse.String(), chunks
}

// BenchmarkLargeToolCallStream measures assembling one large tool call end to
// end. Re-parsing the whole accumulated argument buffer on every delta made
// this quadratic: 5.1 s and 2.5 GB for a single 54 KB `write`, during which the
// UI could not redraw.
func BenchmarkLargeToolCallStream(b *testing.B) {
	body, _ := largeToolCallStream(b)
	server := sseServerB(b, body)
	defer server.Close()

	model := testModel(server.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "write the file"}}}

	for b.Loop() {
		drainEvents(Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}}))
	}
}

// TestLargeToolCallStreamStillParsesExactArguments is the correctness guard on
// the throttle: intermediate previews may lag, but the completed tool call must
// carry the exact arguments the provider sent.
func TestLargeToolCallStreamStillParsesExactArguments(t *testing.T) {
	body, chunks := largeToolCallStream(t)
	if chunks < 1000 {
		t.Fatalf("fixture only produced %d chunks", chunks)
	}
	server := sseServer(t, body)
	defer server.Close()

	events := drainEvents(Stream(testModel(server.URL),
		&Context{Messages: []Message{UserMessage{Role: "user", Content: "write the file"}}},
		&OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}}))

	final := events[len(events)-1]
	if final.Message == nil {
		t.Fatalf("final = %#v", final)
	}
	var call *ToolCall
	for _, block := range final.Message.Content {
		if tc, ok := block.(ToolCall); ok {
			call = &tc
		}
	}
	if call == nil {
		t.Fatal("no tool call in the final message")
	}
	if call.Name != "write" {
		t.Fatalf("name = %q", call.Name)
	}
	if got, _ := call.Arguments["path"].(string); got != "main.go" {
		t.Fatalf("path = %q", got)
	}
	content, _ := call.Arguments["content"].(string)
	if want := strings.Repeat("some source code line here\n", 2000); content != want {
		t.Fatalf("content length = %d, want %d", len(content), len(want))
	}
}

// TestStreamingParseThrottleStaysLiveAndBounded pins both halves of the
// trade-off: a small tool call re-parses on essentially every delta so its
// preview stays live, while a large one re-parses a bounded number of times so
// assembling it stays linear rather than quadratic.
func TestStreamingParseThrottleStaysLiveAndBounded(t *testing.T) {
	// Small call: a shell command of a few dozen bytes.
	parses, parsed := 0, 0
	for length := 1; length <= 64; length++ {
		if shouldReparseStreamingJSON(parsed, length) {
			parses++
			parsed = length
		}
	}
	if parses < 5 {
		t.Fatalf("a 64-byte tool call re-parsed only %d times; the preview would look frozen", parses)
	}

	// Large call: a 54 KB file body.
	const large = 54_000
	parses, parsed = 0, 0
	work := 0
	for length := 1; length <= large; length++ {
		if shouldReparseStreamingJSON(parsed, length) {
			parses++
			parsed = length
			work += length
		}
	}
	if parses > 80 {
		t.Fatalf("a %d-byte tool call re-parsed %d times; that is the quadratic behaviour again", large, parses)
	}
	if parses < 20 {
		t.Fatalf("a %d-byte tool call re-parsed only %d times; the preview would barely move", large, parses)
	}
	// Total bytes re-scanned must stay a small multiple of the payload rather
	// than growing with its square.
	if work > 10*large {
		t.Fatalf("re-scanned %d bytes for a %d-byte payload", work, large)
	}
}

// largeTextStream builds an SSE body streaming a long prose answer in
// token-sized chunks.
func largeTextStream(tb testing.TB) string {
	tb.Helper()
	answer := strings.Repeat("This is a sentence of a fairly long explanation. ", 1200) // ~58 KB

	var sse strings.Builder
	sse.WriteString(`data: {"id":"c1","model":"gpt-test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}` + "\n\n")
	const chunkSize = 4
	for i := 0; i < len(answer); i += chunkSize {
		fragment, err := json.Marshal(answer[i:min(i+chunkSize, len(answer))])
		if err != nil {
			tb.Fatal(err)
		}
		sse.WriteString(`data: {"id":"c1","choices":[{"index":0,"delta":{"content":` + string(fragment) + `},"finish_reason":null}]}` + "\n\n")
	}
	sse.WriteString(`data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n")
	sse.WriteString("data: [DONE]\n\n")
	return sse.String()
}

// BenchmarkLargeTextStream measures assembling one long prose answer.
func BenchmarkLargeTextStream(b *testing.B) {
	server := sseServerB(b, largeTextStream(b))
	defer server.Close()

	model := testModel(server.URL)
	conv := &Context{Messages: []Message{UserMessage{Role: "user", Content: "explain"}}}

	for b.Loop() {
		drainEvents(Stream(model, conv, &OpenAICompletionsOptions{StreamOptions: StreamOptions{APIKey: "k"}}))
	}
}

// TestAnthropicManyContentBlocksDoesNotPanic covers the accumulator's storage.
// The per-block accumulators live in a slice that grows as blocks arrive, and
// strings.Builder panics if it is used after being copied by value -- which is
// exactly what appending to that slice does when it reallocates.
func TestAnthropicManyContentBlocksDoesNotPanic(t *testing.T) {
	var sse strings.Builder
	sse.WriteString(`event: message_start
data: {"type":"message_start","message":{"id":"m1","model":"claude-test","role":"assistant","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}

`)
	// Far more blocks than any initial slice capacity, each written to after
	// later blocks have forced the slice to grow.
	const blockCount = 64
	for index := range blockCount {
		fmt.Fprintf(&sse, `event: content_block_start
data: {"type":"content_block_start","index":%d,"content_block":{"type":"text","text":""}}

`, index)
	}
	for index := range blockCount {
		fmt.Fprintf(&sse, `event: content_block_delta
data: {"type":"content_block_delta","index":%d,"delta":{"type":"text_delta","text":"chunk-%d "}}

`, index, index)
	}
	for index := range blockCount {
		fmt.Fprintf(&sse, `event: content_block_stop
data: {"type":"content_block_stop","index":%d}

`, index)
	}
	sse.WriteString(`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}

event: message_stop
data: {"type":"message_stop"}

`)

	server := sseServerB(t, sse.String())
	defer server.Close()

	model := &Model{ID: "claude-test", Name: "Claude Test", API: APIAnthropicMessages,
		Provider: "anthropic", BaseURL: server.URL, Input: []string{"text"},
		ContextWindow: 200000, MaxTokens: 8192}
	options := &AnthropicOptions{}
	options.APIKey = "k"

	result, err := StreamAnthropicMessages(model, &Context{Messages: []Message{UserMessage{Role: "user", Content: "hi"}}}, options).Result(t.Context())
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if result.StopReason == StopError {
		t.Fatalf("stream failed: %s", result.ErrorMessage)
	}
	var seen int
	for _, block := range result.Content {
		if text, ok := block.(TextContent); ok && strings.HasPrefix(text.Text, "chunk-") {
			seen++
		}
	}
	if seen != blockCount {
		t.Fatalf("recovered %d of %d text blocks", seen, blockCount)
	}
}
