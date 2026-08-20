package plannotator

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
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

type versionedReviewerStub struct {
	decisions []Decision
	previous  []string
}

func (r *versionedReviewerStub) Review(context.Context, string, string) (Decision, error) {
	return Decision{}, errors.New("expected versioned review")
}

func (r *versionedReviewerStub) ReviewVersion(_ context.Context, _, _ string, previous string) (Decision, error) {
	r.previous = append(r.previous, previous)
	decision := r.decisions[0]
	r.decisions = r.decisions[1:]
	return decision, nil
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

func TestParseChecklistAcceptsNumberedAndPlusItems(t *testing.T) {
	items := ParseChecklist("1. [ ] First\n+ [x] Second\n- [ ] Third")
	if len(items) != 3 || items[0].Text != "First" || !items[1].Completed || items[2].Step != 3 {
		t.Fatalf("items = %#v", items)
	}
}

func TestPrepareReviewDocumentKeepsHeadingsAndNumberedSections(t *testing.T) {
	lines, headings := prepareReviewDocument("# Plan\n#include <stdio.h>\n\n1. First section\n- [x] done\n- [ ] open\n")
	if len(headings) != 1 || headings[0].Text != "Plan" {
		t.Fatalf("headings = %#v", headings)
	}
	kinds := make([]string, 0, len(lines))
	displays := make([]string, 0, len(lines))
	for _, line := range lines {
		kinds = append(kinds, line.Kind)
		displays = append(displays, line.Display)
	}
	joined := strings.Join(displays, "\n")
	if !strings.Contains(joined, "#include <stdio.h>") {
		t.Fatalf("C include was rewritten: %q", joined)
	}
	if !strings.Contains(strings.Join(kinds, ","), "numbered") || !strings.Contains(strings.Join(kinds, ","), "task-done") {
		t.Fatalf("kinds = %s", strings.Join(kinds, ","))
	}
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
	shell := manager.BeforeToolCall(t.Context(), agent.BeforeToolCallContext{ToolCall: llm.ToolCall{Name: "bash"}, Args: map[string]any{"command": "rm -rf ."}})
	if shell == nil || !shell.Block {
		t.Fatalf("shell was not blocked: %#v", shell)
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
	pageText := string(body)
	match := regexp.MustCompile(`name="token" value="([^"]+)"`).FindStringSubmatch(pageText)
	if len(match) != 2 {
		t.Fatalf("review page has no token: %s", body)
	}
	for _, feature := range []string{"Planner", "Contents", "Annotations", "Feedback", "Edit", "Copy plan", "Overall implementation notes"} {
		if !strings.Contains(pageText, feature) {
			t.Fatalf("review page missing %q: %s", feature, pageText)
		}
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

func TestReviewPageJavaScriptParses(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		t.Skip("node is unavailable")
	}
	markdown := "# Plan `quoted`\n- [ ] don't break \\ paths\n<script>escaped</script>"
	lines, headings := prepareReviewDocument(markdown)
	var page bytes.Buffer
	if err := reviewPage.Execute(&page, reviewPageData{
		Title: "Review", Markdown: markdown, Previous: "# Earlier", Lines: lines,
		Headings: headings, Token: "token", HasPrevious: true,
	}); err != nil {
		t.Fatal(err)
	}
	match := regexp.MustCompile(`(?s)<script>(.*)</script>`).FindSubmatch(page.Bytes())
	if len(match) != 2 {
		t.Fatal("rendered page has no script")
	}
	path := filepath.Join(t.TempDir(), "review.js")
	if err := os.WriteFile(path, match[1], 0o600); err != nil {
		t.Fatal(err)
	}
	if output, err := exec.Command(node, "--check", path).CombinedOutput(); err != nil {
		t.Fatalf("review JavaScript does not parse: %v\n%s", err, output)
	}
}

func TestBrowserReviewerVersionShowsChangesAndEscapesContent(t *testing.T) {
	opened := make(chan string, 1)
	reviewer := BrowserReviewer{OpenBrowser: func(target string) error { opened <- target; return nil }}
	result := make(chan error, 1)
	go func() {
		_, err := reviewer.ReviewVersion(t.Context(), "Review <script>", "# New\n- [ ] <script>alert(1)</script>", "# Old")
		result <- err
	}()
	target := <-opened
	response, err := http.Get(target)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	text := string(body)
	if !strings.Contains(text, "± Changes") || strings.Contains(text, "<script>alert(1)</script>") {
		t.Fatalf("versioned/escaped page = %s", text)
	}
	match := regexp.MustCompile(`name="token" value="([^"]+)"`).FindStringSubmatch(text)
	if len(match) != 2 {
		t.Fatal("missing token")
	}
	form := url.Values{"action": {"approve"}, "token": {match[1]}}
	posted, err := http.PostForm(target+"api/decision", form)
	if err != nil {
		t.Fatal(err)
	}
	posted.Body.Close()
	if err := <-result; err != nil {
		t.Fatal(err)
	}
}

func TestSubmitPassesDeniedPlanToNextVersion(t *testing.T) {
	root := t.TempDir()
	plan := "# Plan\n- [ ] First\n"
	if err := os.WriteFile(filepath.Join(root, "plan.md"), []byte(plan), 0o600); err != nil {
		t.Fatal(err)
	}
	reviewer := &versionedReviewerStub{decisions: []Decision{{Feedback: "revise"}, {Approved: true}}}
	manager, err := New(root, filepath.Join(t.TempDir(), "state.json"), reviewer)
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Submit(t.Context(), "plan.md"); err != nil {
		t.Fatal(err)
	}
	updated := "# Plan\n- [ ] First\n- [ ] Second\n"
	if err := os.WriteFile(filepath.Join(root, "plan.md"), []byte(updated), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := manager.Submit(t.Context(), "plan.md"); err != nil {
		t.Fatal(err)
	}
	if len(reviewer.previous) != 2 || reviewer.previous[0] != "" || reviewer.previous[1] != plan {
		t.Fatalf("previous versions = %#v", reviewer.previous)
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
