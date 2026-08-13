// Package btw is a native Go adaptation of @narumitw/pi-btw. Side threads
// use the current conversation as read-only background, retain follow-ups in
// memory, and never append their questions or answers to the main transcript.
package btw

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

const MaxContextChars = 40_000

const SystemPrompt = `You answer quick side questions for a coding-agent user.

Use the provided conversation context only as background. Answer the user's side question directly and concisely. Do not claim to have changed files, run tools, or affected the main task. If the context is insufficient, say what is unknown and give the best next step.`

type Turn struct {
	Kind     string               `json:"kind"`
	Question string               `json:"question"`
	Answer   string               `json:"answer"`
	Response llm.AssistantMessage `json:"-"`
}

type Thread struct {
	ID                  string
	Title               string
	ConversationContext string
	Turns               []Turn
	ThinkingLevel       llm.ModelThinkingLevel
	CreatedAt           time.Time
	UpdatedAt           time.Time
}

type Summary struct {
	ID, Title string
	Questions int
	UpdatedAt time.Time
}

type Manager struct {
	mu      sync.Mutex
	next    int
	threads map[string]*Thread
}

func NewManager() *Manager { return &Manager{next: 1, threads: map[string]*Thread{}} }

func (m *Manager) New(context string, thinking llm.ModelThinkingLevel) *Thread {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	thread := &Thread{ID: fmt.Sprintf("btw-%d", m.next), ConversationContext: context,
		ThinkingLevel: thinking, CreatedAt: now, UpdatedAt: now}
	m.next++
	m.threads[thread.ID] = thread
	return thread
}

func (m *Manager) Get(id string) *Thread { m.mu.Lock(); defer m.mu.Unlock(); return m.threads[id] }

// Snapshot returns an immutable copy suitable for rendering while a side
// model request may finish on another goroutine.
func (m *Manager) Snapshot(thread *Thread) Thread {
	m.mu.Lock()
	defer m.mu.Unlock()
	copy := *thread
	copy.Turns = append([]Turn(nil), thread.Turns...)
	return copy
}

func (m *Manager) SetThinkingLevel(thread *Thread, level llm.ModelThinkingLevel) {
	m.mu.Lock()
	defer m.mu.Unlock()
	thread.ThinkingLevel = level
}

func (m *Manager) List() []Summary {
	m.mu.Lock()
	defer m.mu.Unlock()
	output := make([]Summary, 0, len(m.threads))
	for _, thread := range m.threads {
		if thread.Title == "" || len(thread.Turns) == 0 {
			continue
		}
		output = append(output, Summary{ID: thread.ID, Title: thread.Title, Questions: len(thread.Turns), UpdatedAt: thread.UpdatedAt})
	}
	sort.Slice(output, func(i, j int) bool { return output[i].UpdatedAt.After(output[j].UpdatedAt) })
	return output
}

func (m *Manager) RecordAnswered(thread *Thread, question string, response llm.AssistantMessage) string {
	answer := AssistantText(response)
	if answer == "" {
		answer = "No response received."
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	thread.Turns = append(thread.Turns, Turn{Kind: "answered", Question: question, Answer: answer, Response: response})
	if thread.Title == "" {
		thread.Title = SanitizeSingleLine(question)
		if thread.Title == "" {
			thread.Title = "Untitled side thread"
		}
	}
	thread.UpdatedAt = time.Now()
	return answer
}

func (m *Manager) RecordError(thread *Thread, question, message string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	thread.Turns = append(thread.Turns, Turn{Kind: "error", Question: question, Answer: message})
	if thread.Title == "" {
		thread.Title = SanitizeSingleLine(question)
		if thread.Title == "" {
			thread.Title = "Untitled side thread"
		}
	}
	thread.UpdatedAt = time.Now()
}

func BuildMessages(thread *Thread, question string) []llm.Message {
	var answered []Turn
	for _, turn := range thread.Turns {
		if turn.Kind == "answered" {
			answered = append(answered, turn)
		}
	}
	if len(answered) == 0 {
		return []llm.Message{userMessage(BuildUserPrompt(question, thread.ConversationContext))}
	}
	messages := []llm.Message{userMessage(BuildUserPrompt(answered[0].Question, thread.ConversationContext)), answered[0].Response}
	for _, turn := range answered[1:] {
		messages = append(messages, userMessage(BuildFollowUpPrompt(turn.Question)), turn.Response)
	}
	return append(messages, userMessage(BuildFollowUpPrompt(question)))
}

func BuildUserPrompt(question, conversation string) string {
	if conversation == "" {
		conversation = "No prior conversation context was available."
	}
	return strings.Join([]string{"Answer this side question without modifying the main conversation.", "", "<side_question>", question,
		"</side_question>", "", "<conversation_context>", conversation, "</conversation_context>"}, "\n")
}

func BuildFollowUpPrompt(question string) string {
	return "Continue the same side conversation.\n\n<side_question>\n" + question + "\n</side_question>"
}

func userMessage(text string) llm.UserMessage {
	return llm.UserMessage{Role: "user", Content: text, Timestamp: time.Now().UnixMilli()}
}

func AssistantText(message llm.AssistantMessage) string {
	var output []string
	for _, block := range message.Content {
		if text, ok := block.(llm.TextContent); ok {
			output = append(output, text.Text)
		}
	}
	return strings.TrimSpace(strings.Join(output, "\n"))
}

// BuildConversationContext mirrors pi-btw: user/assistant content and tool-call
// summaries are included, tool-result transcript entries are excluded, and the
// newest 40,000 characters win.
func BuildConversationContext(messages []agent.Message) string {
	var sections []string
	for _, raw := range messages {
		var role, stop string
		var content any
		switch message := raw.(type) {
		case llm.UserMessage:
			role, content = "User", message.Content
		case *llm.UserMessage:
			if message != nil {
				role, content = "User", message.Content
			}
		case llm.AssistantMessage:
			role, content, stop = "Assistant", message.Content, message.StopReason
		case *llm.AssistantMessage:
			if message != nil {
				role, content, stop = "Assistant", message.Content, message.StopReason
			}
		}
		if role == "" {
			continue
		}
		lines := contentLines(content)
		if len(lines) == 0 {
			continue
		}
		if stop != "" && stop != llm.StopStop {
			role += " (" + stop + ")"
		}
		sections = append(sections, role+": "+strings.Join(lines, "\n"))
	}
	text := strings.Join(sections, "\n\n")
	if len([]rune(text)) <= MaxContextChars {
		return text
	}
	runes := []rune(text)
	return fmt.Sprintf("[Earlier context omitted; showing the last %d characters.]\n%s", MaxContextChars, string(runes[len(runes)-MaxContextChars:]))
}

func contentLines(content any) []string {
	if text, ok := content.(string); ok {
		if text = strings.TrimSpace(text); text != "" {
			return []string{text}
		}
		return nil
	}
	blocks, ok := content.([]llm.ContentBlock)
	if !ok {
		return nil
	}
	var lines []string
	for _, block := range blocks {
		switch value := block.(type) {
		case llm.TextContent:
			if text := strings.TrimSpace(value.Text); text != "" {
				lines = append(lines, text)
			}
		case llm.ToolCall:
			encoded, _ := json.Marshal(value.Arguments)
			lines = append(lines, fmt.Sprintf("Tool call: %s(%s)", value.Name, encoded))
		}
	}
	return lines
}

type Segment struct{ Role, Text string }

func AnsweredSegments(thread *Thread, from int) []Segment {
	var answered []Turn
	for _, turn := range thread.Turns {
		if turn.Kind == "answered" {
			answered = append(answered, turn)
		}
	}
	if from < 0 {
		from = 0
	}
	if from > len(answered) {
		from = len(answered)
	}
	var segments []Segment
	for _, turn := range answered[from:] {
		segments = append(segments, Segment{Role: "user", Text: turn.Question}, Segment{Role: "assistant", Text: turn.Answer})
	}
	return segments
}

func LatestSegments(thread *Thread) []Segment {
	segments := AnsweredSegments(thread, 0)
	if len(segments) <= 2 {
		return segments
	}
	return segments[len(segments)-2:]
}

func FormatBringToMain(segments []Segment) string {
	var body []string
	for _, segment := range segments {
		label := "Assistant"
		if segment.Role == "user" {
			label = "User"
		}
		body = append(body, label+":\n"+escapeContext(segment.Text))
	}
	return strings.Join([]string{"The following context was brought back from a /btw side discussion.",
		"Treat it as discussion context, not as work already completed.", "", "<btw_context>", strings.Join(body, "\n\n"), "</btw_context>"}, "\n")
}

func escapeContext(text string) string {
	var output strings.Builder
	for _, r := range text {
		if r == '\n' {
			output.WriteRune(r)
		} else if r == '\t' {
			output.WriteString("    ")
		} else if r < 32 || (r >= 127 && r <= 159) {
			output.WriteString(fmt.Sprintf("\\x%02x", r))
		} else {
			output.WriteRune(r)
		}
	}
	text = output.String()
	text = strings.ReplaceAll(text, "<btw_context", "&lt;btw_context")
	text = strings.ReplaceAll(text, "</btw_context>", "&lt;/btw_context&gt;")
	return text
}

func EstimateTokens(segments []Segment) int {
	size := 0
	for _, segment := range segments {
		size += len([]byte(segment.Text))
	}
	return (size + 3) / 4
}

func SanitizeSingleLine(text string) string {
	return strings.Join(strings.Fields(strings.Map(func(r rune) rune {
		if r < 32 || r == 127 {
			return ' '
		}
		return r
	}, text)), " ")
}

// Complete performs one side-thread model call without using the main Agent.
type Complete func(model *llm.Model, request *llm.Context, options *llm.SimpleStreamOptions) (*llm.AssistantMessage, error)

func Ask(ctx context.Context, thread *Thread, model *llm.Model, question string, thinking llm.ModelThinkingLevel, complete Complete) (*llm.AssistantMessage, error) {
	options := &llm.SimpleStreamOptions{}
	options.Ctx = ctx
	if thinking != llm.ThinkingOff {
		options.Reasoning = thinking
	}
	return complete(model, &llm.Context{SystemPrompt: SystemPrompt, Messages: BuildMessages(thread, question)}, options)
}
