package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"io"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
)

func TestTerminalLoginInteractionSelectsByNumber(t *testing.T) {
	var output bytes.Buffer
	interaction := newTerminalLoginInteraction(strings.NewReader("2\n"), &output, false)
	choice, err := interaction.Prompt(catalog.LoginPrompt{
		Kind:    catalog.PromptSelect,
		Message: "Choose:",
		Options: []catalog.LoginPromptOption{{ID: "browser", Label: "Browser"}, {ID: "device", Label: "Device code"}},
		Ctx:     t.Context(),
	})
	if err != nil || choice != "device" {
		t.Fatalf("choice = %q, err = %v", choice, err)
	}
	if !strings.Contains(output.String(), "2) Device code") {
		t.Fatalf("output = %q", output.String())
	}
}

func TestTerminalLoginInteractionDefaultsToFirstOption(t *testing.T) {
	interaction := newTerminalLoginInteraction(strings.NewReader("\n"), io.Discard, false)
	choice, err := interaction.Prompt(catalog.LoginPrompt{
		Kind:    catalog.PromptSelect,
		Options: []catalog.LoginPromptOption{{ID: "browser", Label: "Browser"}, {ID: "device", Label: "Device"}},
		Ctx:     t.Context(),
	})
	if err != nil || choice != "browser" {
		t.Fatalf("choice = %q, err = %v", choice, err)
	}
}

func TestTerminalLoginInteractionHonorsPromptCancellation(t *testing.T) {
	reader, writer := io.Pipe()
	defer reader.Close()
	defer writer.Close()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	interaction := newTerminalLoginInteraction(reader, io.Discard, false)
	_, err := interaction.Prompt(catalog.LoginPrompt{Kind: catalog.PromptManualCode, Ctx: ctx})
	if !errors.Is(err, catalog.ErrLoginCancelled) {
		t.Fatalf("err = %v, want ErrLoginCancelled", err)
	}
}

func TestTerminalLoginInteractionRendersOAuthEvents(t *testing.T) {
	var output bytes.Buffer
	interaction := newTerminalLoginInteraction(strings.NewReader(""), &output, false)
	interaction.Notify(catalog.LoginEvent{Kind: "auth_url", URL: "https://example.com/auth", Instructions: "Sign in."})
	interaction.Notify(catalog.LoginEvent{Kind: "device_code", VerificationURI: "https://example.com/device", UserCode: "ABCD"})
	interaction.Notify(catalog.LoginEvent{Kind: "progress", Message: "Exchanging..."})
	got := output.String()
	for _, want := range []string{"https://example.com/auth", "Sign in.", "https://example.com/device", "ABCD", "Exchanging..."} {
		if !strings.Contains(got, want) {
			t.Fatalf("output %q does not contain %q", got, want)
		}
	}
}

func TestFirstLine(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"single line", "hello", "hello"},
		{"trims surrounding space", "  hello  ", "hello"},
		{"multiline is elided", "first\nsecond", "first ..."},
		{"empty", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := firstLine(tt.in); got != tt.want {
				t.Fatalf("firstLine(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}

	// Long single lines are truncated with an ellipsis marker.
	long := strings.Repeat("x", 200)
	got := firstLine(long)
	if len(got) != 124 || !strings.HasSuffix(got, " ...") {
		t.Fatalf("long line = %d chars, suffix %q", len(got), got[len(got)-4:])
	}
}

func TestSummarizeArgs(t *testing.T) {
	// Keys are sorted so the activity line is stable across runs.
	got := summarizeArgs(map[string]any{"path": "a.txt", "content": "hi"})
	if got != "content=hi path=a.txt" {
		t.Fatalf("summarizeArgs = %q", got)
	}

	if got := summarizeArgs(nil); got != "" {
		t.Fatalf("empty args = %q", got)
	}

	// Newlines are flattened so one tool call stays on one line.
	if got := summarizeArgs(map[string]any{"c": "a\nb"}); got != "c=a b" {
		t.Fatalf("multiline arg = %q", got)
	}

	// Long values are truncated.
	long := summarizeArgs(map[string]any{"c": strings.Repeat("y", 100)})
	if !strings.HasSuffix(long, "...") || len(long) > 70 {
		t.Fatalf("long arg = %q (%d chars)", long, len(long))
	}
}

func TestResultText(t *testing.T) {
	if got := resultText(nil); got != "" {
		t.Fatalf("nil result = %q", got)
	}

	result := &agent.ToolResult{Content: []llm.ContentBlock{
		llm.TextContent{Type: "text", Text: "one"},
		// Non-text blocks are skipped.
		llm.ImageContent{Type: "image", Data: "AAAA", MimeType: "image/png"},
		llm.TextContent{Type: "text", Text: "two"},
	}}
	if got := resultText(result); got != "one\ntwo" {
		t.Fatalf("resultText = %q", got)
	}
}

func TestLastAssistant(t *testing.T) {
	messages := []agent.Message{
		llm.UserMessage{Role: "user", Content: "hi"},
		llm.AssistantMessage{Role: "assistant", Model: "first"},
		llm.ToolResultMessage{Role: "toolResult"},
		llm.AssistantMessage{Role: "assistant", Model: "last"},
	}
	message, ok := lastAssistant(messages)
	if !ok || message.Model != "last" {
		t.Fatalf("lastAssistant = %#v, ok = %v", message, ok)
	}

	if _, ok := lastAssistant([]agent.Message{llm.UserMessage{Role: "user"}}); ok {
		t.Fatal("a transcript without an assistant message should report false")
	}
	if _, ok := lastAssistant(nil); ok {
		t.Fatal("an empty transcript should report false")
	}
}

// TestDimAndBoldAreInert confirms styling is skipped when stderr is not a
// terminal, so piped output stays clean.
func TestDimAndBoldAreInert(t *testing.T) {
	// Under `go test` stderr is a pipe, so colorEnabled is false.
	if colorEnabled() {
		t.Skip("stderr is a terminal in this environment")
	}
	if got := dim("text"); got != "text" {
		t.Fatalf("dim = %q, want unstyled", got)
	}
	if got := bold("text"); got != "text" {
		t.Fatalf("bold = %q, want unstyled", got)
	}
	if got := dim(""); got != "" {
		t.Fatalf("dim(\"\") = %q", got)
	}
}

func TestBindSessionFlags(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	cfg := bindSessionFlags(flags)

	if err := flags.Parse([]string{
		"-m", "openai/gpt-4",
		"-s", "be brief",
		"-thinking", "high",
		"-tools",
		"-ralph",
		"-planner",
		"-claude-tui",
		"-C", "/tmp/work",
		"trailing", "prompt",
	}); err != nil {
		t.Fatal(err)
	}

	if cfg.ModelRef != "openai/gpt-4" || cfg.SystemPrompt != "be brief" {
		t.Fatalf("cfg = %#v", cfg)
	}
	if cfg.Thinking != "high" || !cfg.EnableTools || !cfg.EnableRalph || !cfg.EnablePlannotator || !cfg.ClaudeTUI {
		t.Fatalf("cfg = %#v", cfg)
	}
	if cfg.Workdir != "/tmp/work" {
		t.Fatalf("workdir = %q", cfg.Workdir)
	}
	// Positional arguments remain for the caller to join into a prompt.
	if strings.Join(flags.Args(), " ") != "trailing prompt" {
		t.Fatalf("args = %v", flags.Args())
	}
}

// TestBindSessionFlagsLongForms confirms the long flag aliases work too.
func TestBindSessionFlagsLongForms(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	cfg := bindSessionFlags(flags)
	if err := flags.Parse([]string{"-model", "anthropic/claude", "-system", "sys"}); err != nil {
		t.Fatal(err)
	}
	if cfg.ModelRef != "anthropic/claude" || cfg.SystemPrompt != "sys" {
		t.Fatalf("cfg = %#v", cfg)
	}
}

func TestChatDefaultsEnableToolsAndRalphWithOptOut(t *testing.T) {
	flags := flag.NewFlagSet("chat", flag.ContinueOnError)
	cfg := bindSessionFlags(flags)
	applyChatDefaults(flags, cfg)
	if err := flags.Parse([]string{"-tools=false", "-ralph=false"}); err != nil {
		t.Fatal(err)
	}
	if cfg.EnableTools || cfg.EnableRalph {
		t.Fatalf("explicit chat opt-out ignored: %#v", cfg)
	}

	flags = flag.NewFlagSet("chat", flag.ContinueOnError)
	cfg = bindSessionFlags(flags)
	applyChatDefaults(flags, cfg)
	if err := flags.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if !cfg.EnableTools || !cfg.EnableRalph {
		t.Fatalf("chat defaults = %#v", cfg)
	}
}

func TestBindSessionFlagsDefaults(t *testing.T) {
	flags := flag.NewFlagSet("test", flag.ContinueOnError)
	cfg := bindSessionFlags(flags)
	if err := flags.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if cfg.Thinking != "off" || cfg.Workdir != "." {
		t.Fatalf("defaults = %#v", cfg)
	}
	if cfg.EnableTools || cfg.EnableRalph || cfg.EnablePlannotator || !cfg.ClaudeTUI {
		t.Fatalf("unexpected native feature defaults: %#v", cfg)
	}
}

func TestIsThinkingLevel(t *testing.T) {
	for _, level := range []string{"off", "minimal", "low", "medium", "high", "xhigh", "max"} {
		if !isThinkingLevel(level) {
			t.Errorf("%q should be a valid thinking level", level)
		}
	}
	for _, level := range []string{"", "none", "HIGH", "extreme"} {
		if isThinkingLevel(level) {
			t.Errorf("%q should not be a valid thinking level", level)
		}
	}
}

func TestSummarizeMessage(t *testing.T) {
	tests := []struct {
		name    string
		message agent.Message
		want    string
	}{
		{
			name:    "string user message",
			message: llm.UserMessage{Role: "user", Content: "hello"},
			want:    "hello",
		},
		{
			name: "assistant text",
			message: llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{
				llm.TextContent{Type: "text", Text: "answer"},
			}},
			want: "answer",
		},
		{
			name: "assistant thinking and tool call are named",
			message: llm.AssistantMessage{Role: "assistant", Content: []llm.ContentBlock{
				llm.ThinkingContent{Type: "thinking", Thinking: "hmm"},
				llm.ToolCall{Type: "toolCall", Name: "read"},
			}},
			want: "[thinking] [read]",
		},
		{
			name: "tool result is prefixed with the tool name",
			message: llm.ToolResultMessage{Role: "toolResult", ToolName: "read", Content: []llm.ContentBlock{
				llm.TextContent{Type: "text", Text: "contents"},
			}},
			want: "read: contents",
		},
		{
			name:    "unknown message type",
			message: "not a message",
			want:    "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := summarizeMessage(tt.message); got != tt.want {
				t.Fatalf("summarizeMessage = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBlockSummary(t *testing.T) {
	got := blockSummary([]llm.ContentBlock{
		llm.TextContent{Type: "text", Text: "text"},
		// Blank text is skipped so it does not pad the summary.
		llm.TextContent{Type: "text", Text: "   "},
		llm.ImageContent{Type: "image"},
	})
	if got != "text [image]" {
		t.Fatalf("blockSummary = %q", got)
	}
	if got := blockSummary(nil); got != "" {
		t.Fatalf("empty blockSummary = %q", got)
	}
}

func TestUserMessage(t *testing.T) {
	message := userMessage("hi")
	if message.Role != "user" {
		t.Fatalf("role = %q", message.Role)
	}
	text, ok := message.StringContent()
	if !ok || text != "hi" {
		t.Fatalf("content = %#v", message.Content)
	}
	if message.Timestamp == 0 {
		t.Fatal("timestamp should be set")
	}
}

// TestErrorStreamReportsFailure confirms an unsupported protocol surfaces
// through the stream contract rather than panicking.
func TestErrorStreamReportsFailure(t *testing.T) {
	model := &llm.Model{ID: "m", API: "unknown-api", Provider: "p"}
	stream := errorStream(model, "no streamer registered")

	final, err := stream.Result(t.Context())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if final.StopReason != llm.StopError {
		t.Fatalf("stopReason = %q", final.StopReason)
	}
	if !strings.Contains(final.ErrorMessage, "no streamer registered") {
		t.Fatalf("errorMessage = %q", final.ErrorMessage)
	}
	// The failing model is identified so callers can report it.
	if final.API != "unknown-api" || final.Model != "m" {
		t.Fatalf("message = %#v", final)
	}
}

// TestAuthStreamFnAppliesCredentials confirms resolved headers and provider env
// reach the wire protocol, which the agent's default path does not carry.
func TestAuthStreamFnAppliesCredentials(t *testing.T) {
	headerValue := "Bearer token"
	auth := &catalog.Auth{
		APIKey:  "sk-test",
		Headers: llm.ProviderHeaders{"Authorization": &headerValue},
		Env:     map[string]string{"CLOUDFLARE_ACCOUNT_ID": "acct"},
	}

	// A model with an unregistered API short-circuits before any HTTP call,
	// which is enough to observe the options that were prepared.
	model := &llm.Model{ID: "m", API: "unknown-api", Provider: "p"}
	opts := &llm.SimpleStreamOptions{}

	stream := authStreamFn(auth)(model, &llm.Context{}, opts)
	if _, err := stream.Result(t.Context()); err != nil {
		t.Fatalf("Result: %v", err)
	}

	if opts.Headers == nil || opts.Headers["Authorization"] == nil {
		t.Fatalf("headers = %#v", opts.Headers)
	}
	if *opts.Headers["Authorization"] != "Bearer token" {
		t.Fatalf("authorization = %q", *opts.Headers["Authorization"])
	}
	if opts.Env["CLOUDFLARE_ACCOUNT_ID"] != "acct" {
		t.Fatalf("env = %#v", opts.Env)
	}
}

// TestAuthStreamFnToleratesNilAuth confirms a session without credentials still
// produces a usable stream rather than panicking.
func TestAuthStreamFnToleratesNilAuth(t *testing.T) {
	model := &llm.Model{ID: "m", API: "unknown-api", Provider: "p"}
	stream := authStreamFn(nil)(model, &llm.Context{}, &llm.SimpleStreamOptions{})

	final, err := stream.Result(t.Context())
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if final.StopReason != llm.StopError {
		t.Fatalf("stopReason = %q", final.StopReason)
	}
}

func TestLimitedCommandOutputCapsMemory(t *testing.T) {
	output := &limitedCommandOutput{limit: 4}
	if n, err := output.Write([]byte("abcdef")); err != nil || n != 6 {
		t.Fatalf("Write = %d, %v", n, err)
	}
	if output.builder.String() != "abcd" || !output.truncated {
		t.Fatalf("output = %#v", output)
	}
}
