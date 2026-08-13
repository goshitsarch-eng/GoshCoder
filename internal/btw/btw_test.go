package btw

import (
	"strings"
	"testing"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

func TestBuildMessagesKeepsSideThreadSeparateAndReusesAnswers(t *testing.T) {
	thread := &Thread{
		ConversationContext: "User: main task",
		Turns: []Turn{{
			Kind: "answered", Question: "first?", Answer: "first answer",
			Response: llm.AssistantMessage{
				Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "first answer"}},
				StopReason: llm.StopStop,
			},
		}},
	}
	messages := BuildMessages(thread, "follow up?")
	if len(messages) != 3 {
		t.Fatalf("messages = %#v", messages)
	}
	first := messages[0].(llm.UserMessage)
	text, _ := first.StringContent()
	if !strings.Contains(text, "<conversation_context>\nUser: main task") || !strings.Contains(text, "first?") {
		t.Fatalf("first = %q", text)
	}
	last := messages[2].(llm.UserMessage)
	follow, _ := last.StringContent()
	if strings.Contains(follow, "conversation_context") || !strings.Contains(follow, "follow up?") {
		t.Fatalf("follow = %q", follow)
	}
}

func TestBuildConversationContextTruncatesNewestAndSummarizesCalls(t *testing.T) {
	messages := []agent.Message{
		llm.UserMessage{Role: "user", Content: strings.Repeat("x", MaxContextChars)},
		llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{
			llm.ToolCall{Type: "toolCall", Name: "read", Arguments: map[string]any{"path": "a.go"}},
			llm.TextContent{Type: "text", Text: "newest"},
		}, StopReason: llm.StopToolUse},
	}
	context := BuildConversationContext(messages)
	if !strings.HasPrefix(context, "[Earlier context omitted") || !strings.Contains(context, "Tool call: read") || !strings.HasSuffix(context, "newest") {
		t.Fatalf("context = %q", context[:min(len(context), 300)])
	}
}

func TestManagerResumeOrderingAndBringFormatting(t *testing.T) {
	manager := NewManager()
	first := manager.New("context", "low")
	second := manager.New("context", "medium")
	manager.RecordAnswered(first, "q1", llm.AssistantMessage{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "a1"}}})
	time.Sleep(time.Millisecond)
	manager.RecordError(second, "q2", "oops")
	list := manager.List()
	if len(list) != 2 || list[0].ID != second.ID {
		t.Fatalf("list = %#v", list)
	}
	draft := FormatBringToMain(LatestSegments(first))
	if !strings.Contains(draft, "<btw_context>") || !strings.Contains(draft, "User:\nq1") || !strings.Contains(draft, "Assistant:\na1") {
		t.Fatalf("draft = %q", draft)
	}
}
