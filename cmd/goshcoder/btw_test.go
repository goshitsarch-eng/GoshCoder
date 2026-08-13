package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/btw"
	"goshcoder/internal/llm"
)

func TestBTWQuestionDoesNotPolluteMainTranscript(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	t.Setenv("OPENAI_API_KEY", "side-key")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"id":"side-1","choices":[{"index":0,"delta":{"role":"assistant","content":"side answer"},"finish_reason":null}]}

data: {"id":"side-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}

data: [DONE]

`)
	}))
	defer server.Close()

	model := llm.Model{ID: "side-model", API: llm.APIOpenAICompletions, Provider: "openai", BaseURL: server.URL, ContextWindow: 32000, MaxTokens: 1024}
	mainMessages := []agent.Message{llm.UserMessage{Role: "user", Content: "main task"}}
	session := &session{
		agent: agent.NewAgent(agent.AgentOptions{InitialState: &agent.InitialState{Model: model, Messages: mainMessages}}),
		btw:   btw.NewManager(),
	}
	thread := session.btw.New(btw.BuildConversationContext(mainMessages), llm.ThinkingOff)
	answer, err := session.askBTW(t.Context(), thread, "what does this mean?", llm.ThinkingOff)
	if err != nil {
		t.Fatal(err)
	}
	if answer != "side answer" {
		t.Fatalf("answer = %q", answer)
	}
	state := session.agent.State()
	if len(state.Messages) != 1 || userText(state.Messages[0].(llm.UserMessage)) != "main task" {
		t.Fatalf("main transcript was changed: %#v", state.Messages)
	}
	if len(thread.Turns) != 1 || thread.Turns[0].Question != "what does this mean?" {
		t.Fatalf("side thread = %#v", thread)
	}
}
