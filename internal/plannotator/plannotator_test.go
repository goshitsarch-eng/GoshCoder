package plannotator

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

type reviewerStub struct {
	decision Decision
	err      error
}

func (r reviewerStub) Review(context.Context, string, string) (Decision, error) {
	return r.decision, r.err
}

func newTestManager(t *testing.T, reviewer Reviewer) *Manager {
	t.Helper()
	manager, err := New(t.TempDir(), filepath.Join(t.TempDir(), "state.json"), reviewer)
	if err != nil {
		t.Fatal(err)
	}
	return manager
}

func TestParseChecklistAndDoneMarkers(t *testing.T) {
	items := ParseChecklist("- [ ] First\n* [x] Second\nnot a task\n- [ ] Third")
	if len(items) != 3 || items[0].Text != "First" || !items[1].Completed || items[2].Step != 3 {
		t.Fatalf("items = %#v", items)
	}
	manager := newTestManager(t, nil)
	manager.state = State{Phase: PhaseExecuting, Items: items}
	message := llm.AssistantMessage{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "done [DONE:1] and [done:3]"}}}
	if changed := manager.TrackAssistant(message); changed != 2 {
		t.Fatalf("changed = %d", changed)
	}
	state := manager.State()
	if state.Phase != PhaseIdle {
		t.Fatalf("phase = %s", state.Phase)
	}
}

func TestPlanningWriteGate(t *testing.T) {
	manager := newTestManager(t, nil)
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	allowed := manager.BeforeToolCall(t.Context(), agent.BeforeToolCallContext{ToolCall: llm.ToolCall{Name: "write"}, Args: map[string]any{"path": "plans/a.md"}})
	if allowed != nil {
		t.Fatalf("markdown write blocked: %#v", allowed)
	}
	blocked := manager.BeforeToolCall(t.Context(), agent.BeforeToolCallContext{ToolCall: llm.ToolCall{Name: "edit"}, Args: map[string]any{"path": "main.go"}})
	if blocked == nil || !blocked.Block {
		t.Fatalf("code edit was not blocked: %#v", blocked)
	}
	if manager.IsPlanPathAllowed("../escape.md") || manager.IsPlanPathAllowed("/tmp/escape.md") || manager.IsPlanPathAllowed("PLAN.txt") {
		t.Fatal("unsafe plan path accepted")
	}
}

func TestSubmitDeniedThenApproved(t *testing.T) {
	root := t.TempDir()
	plan := "# Plan\n- [ ] Implement\n- [ ] Test\n"
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	manager, _ := New(root, "", reviewerStub{decision: Decision{Approved: false, Feedback: "add rollback"}})
	_ = manager.Enter()
	result, err := manager.Submit(t.Context(), "PLAN.md")
	if err != nil {
		t.Fatal(err)
	}
	text := result.Content[0].(llm.TextContent).Text
	if !strings.Contains(text, "add rollback") || manager.State().Phase != PhasePlanning {
		t.Fatalf("result/state = %q %#v", text, manager.State())
	}

	manager.reviewer = reviewerStub{decision: Decision{Approved: true, Feedback: "keep commits small"}}
	result, err = manager.Submit(t.Context(), "PLAN.md")
	if err != nil {
		t.Fatal(err)
	}
	if manager.State().Phase != PhaseExecuting || len(manager.State().Items) != 2 {
		t.Fatalf("state = %#v", manager.State())
	}
	if !strings.Contains(result.Content[0].(llm.TextContent).Text, "keep commits small") {
		t.Fatalf("result = %#v", result)
	}
}

func TestSubmitRequiresChecklist(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte("# Plan\nJust do it."), 0o644); err != nil {
		t.Fatal(err)
	}
	manager, err := New(root, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	result, err := manager.Submit(t.Context(), "PLAN.md")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content[0].(llm.TextContent).Text, "checklist") {
		t.Fatalf("result = %#v", result)
	}
	if manager.State().Phase != PhasePlanning {
		t.Fatalf("phase = %s", manager.State().Phase)
	}
}

func TestSubmitRejectsSymlinkEscape(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.md")
	if err := os.WriteFile(outside, []byte("- [ ] Exfiltrate"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "PLAN.md")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	manager, err := New(root, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	_ = manager.Enter()
	result, err := manager.Submit(t.Context(), "PLAN.md")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content[0].(llm.TextContent).Text, "cannot be read") {
		t.Fatalf("result = %#v", result)
	}
}

func TestCorruptStateIsReported(t *testing.T) {
	stateFile := filepath.Join(t.TempDir(), "state.json")
	if err := os.WriteFile(stateFile, []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := New(t.TempDir(), stateFile, nil); err == nil {
		t.Fatal("corrupt state was silently ignored")
	}
}

func TestStatePersists(t *testing.T) {
	root, config := t.TempDir(), filepath.Join(t.TempDir(), "state.json")
	manager, _ := New(root, config, nil)
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	reloaded, err := New(root, config, nil)
	if err != nil {
		t.Fatal(err)
	}
	if reloaded.State().Phase != PhasePlanning {
		t.Fatalf("state = %#v", reloaded.State())
	}
}

func TestBrowserReviewerDecision(t *testing.T) {
	opened := make(chan string, 1)
	reviewer := BrowserReviewer{
		OpenBrowser: func(target string) error { opened <- target; return nil },
	}
	result := make(chan Decision, 1)
	errs := make(chan error, 1)
	go func() {
		decision, err := reviewer.Review(t.Context(), "Review", "# Plan")
		result <- decision
		errs <- err
	}()
	target := <-opened
	page, err := http.Get(target)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(page.Body)
	page.Body.Close()
	match := regexp.MustCompile(`name="token" value="([^"]+)"`).FindStringSubmatch(string(body))
	if len(match) != 2 {
		t.Fatalf("review page has no token: %s", body)
	}
	form := url.Values{"action": {"deny"}, "feedback": {"Needs tests"}, "token": {match[1]}}
	response, err := http.PostForm(target+"api/decision", form)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
	decision := <-result
	if decision.Approved || decision.Feedback != "Needs tests" {
		t.Fatalf("decision = %#v", decision)
	}
}

func TestSubmitCancellationIsRecoverable(t *testing.T) {
	root := t.TempDir()
	_ = os.WriteFile(filepath.Join(root, "PLAN.md"), []byte("- [ ] Work"), 0o644)
	manager, _ := New(root, "", reviewerStub{err: context.Canceled})
	_ = manager.Enter()
	result, err := manager.Submit(t.Context(), "PLAN.md")
	if err != nil || !strings.Contains(result.Content[0].(llm.TextContent).Text, "cancelled") {
		t.Fatalf("result = %#v, err = %v", result, err)
	}
}

func TestBrowserReviewerRejectsNonLoopbackHost(t *testing.T) {
	_, err := (BrowserReviewer{Host: "0.0.0.0"}).Review(t.Context(), "x", "y")
	if err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("err = %v", err)
	}
}

func TestBrowserReviewerHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	_, err := (BrowserReviewer{OpenBrowser: func(string) error { return nil }}).Review(ctx, "x", "y")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}
