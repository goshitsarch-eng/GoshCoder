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
	manager, err := New(t.TempDir(), reviewer, Options{})
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

func TestPrepareReviewDocumentIndentsNestedSections(t *testing.T) {
	lines, _ := prepareReviewDocument("# Plan\n- Parent\n  - Nested child\n    1. Deep numbered\n")
	if len(lines) < 4 {
		t.Fatalf("lines = %#v", lines)
	}
	if lines[1].Kind != "bullet" || lines[1].Indent != 0 {
		t.Fatalf("parent = %#v", lines[1])
	}
	if lines[2].Kind != "bullet" || lines[2].Indent != 1 {
		t.Fatalf("nested bullet = %#v", lines[2])
	}
	if lines[3].Kind != "numbered" || lines[3].Indent != 2 {
		t.Fatalf("nested numbered = %#v", lines[3])
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

// TestPlanPathGateIsPlatformIndependent covers the spellings of "rooted" that
// only one platform recognises. filepath.IsAbs says no to "/plans/a.md" on
// Windows -- there it is relative to the current drive -- so the gate used to
// join it onto the workspace and accept a path Unix rejected outright. A write
// gate that means different things on different platforms is the bug; the
// cases below therefore run everywhere rather than behind a GOOS check.
func TestPlanPathGateIsPlatformIndependent(t *testing.T) {
	manager := newTestManager(t, nil)
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}

	for _, path := range []string{
		"/plans/a.md",             // rooted on Unix, drive-relative on Windows
		"\\plans\\a.md",           // the same thing spelled for Windows
		"C:\\plans\\a.md",         // an absolute path on another drive
		"c:/plans/a.md",           // lowercase drive, forward slashes
		"\\\\server\\share\\a.md", // UNC
		"../escape.md",            // ordinary traversal, rejected before and now
	} {
		if manager.IsPlanPathAllowed(path) {
			t.Errorf("IsPlanPathAllowed(%q) = true, want it refused on every platform", path)
		}
	}

	// The negative control: the gate still has to accept the thing it exists
	// to allow, or it would pass by refusing everything.
	for _, path := range []string{"PLAN.md", "plans/a.md", "docs/design.mdx"} {
		if !manager.IsPlanPathAllowed(path) {
			t.Errorf("IsPlanPathAllowed(%q) = false, want a workspace-relative plan accepted", path)
		}
	}
}

func TestSubmitDeniedThenApproved(t *testing.T) {
	root := t.TempDir()
	plan := "# Plan\n- [ ] Implement\n- [ ] Test\n"
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	manager, _ := New(root, reviewerStub{decision: Decision{Approved: false, Feedback: "add rollback"}}, Options{})
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
	manager, err := New(root, nil, Options{})
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
	manager, err := New(root, nil, Options{})
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

// TestUnrecognizedPhaseStartsIdleWithAWarning covers what used to be a hard
// error. Plan state now travels in the session log, so refusing to parse it
// would make a corrupt Planner entry a reason the whole session will not open --
// a much worse failure than starting idle and saying so.
func TestUnrecognizedPhaseStartsIdleWithAWarning(t *testing.T) {
	var warnings []string
	manager, err := New(t.TempDir(), nil, Options{
		Initial: &State{Phase: Phase("nonsense")},
		Warn:    func(message string) { warnings = append(warnings, message) },
	})
	if err != nil {
		t.Fatalf("an unrecognized phase failed the whole construction: %v", err)
	}
	if manager.State().Phase != PhaseIdle {
		t.Fatalf("phase = %q, want idle", manager.State().Phase)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one explaining the state was ignored", warnings)
	}
}

// TestStateIsHandedToTheHostOnEveryTransition replaces the old
// TestStatePersists. The promise moved: it used to be "a fresh process in the
// same workspace resumes the plan", and is now "the host is told, so -continue
// resumes the plan". The Manager owns no file.
func TestStateIsHandedToTheHostOnEveryTransition(t *testing.T) {
	var published []State
	manager, err := New(t.TempDir(), nil, Options{
		OnChange: func(state State) { published = append(published, state) },
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	if err := manager.Exit(); err != nil {
		t.Fatal(err)
	}
	if len(published) != 2 {
		t.Fatalf("published %d states, want one per transition", len(published))
	}
	if published[0].Phase != PhasePlanning || published[1].Phase != PhaseIdle {
		t.Fatalf("published = %v", published)
	}

	// And a restored state seeds a fresh Manager, which is how -continue works.
	resumed, err := New(t.TempDir(), nil, Options{Initial: &State{Phase: PhasePlanning}})
	if err != nil {
		t.Fatal(err)
	}
	if resumed.State().Phase != PhasePlanning {
		t.Fatalf("restored phase = %q, want planning", resumed.State().Phase)
	}
}

// TestCompletingTheLastStepKeepsThePlanRecord covers the fourth site that
// assigned a bare State{} literal, and the one easiest to miss: fixing Enter,
// Exit and Toggle still leaves finishing a plan wiping the record of it.
func TestCompletingTheLastStepKeepsThePlanRecord(t *testing.T) {
	root := t.TempDir()
	plan := "- [ ] first\n- [ ] second\n"
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte(plan), 0o600); err != nil {
		t.Fatal(err)
	}
	var published []State
	manager, err := New(root, nil, Options{
		Initial: &State{
			Phase: PhaseExecuting, PlanPath: "PLAN.md", PlanHash: hashPlan([]byte(plan)),
			Items: []ChecklistItem{{Step: 1, Text: "first"}, {Step: 2, Text: "second"}},
		},
		OnChange: func(state State) { published = append(published, state) },
	})
	if err != nil {
		t.Fatal(err)
	}

	manager.TrackAssistant(llm.AssistantMessage{Content: []llm.ContentBlock{
		llm.TextContent{Type: "text", Text: "done with [DONE:1] and [DONE:2]"}}})

	state := manager.State()
	if state.Phase != PhaseIdle {
		t.Fatalf("phase = %q, want idle once every step is done", state.Phase)
	}
	if state.PlanPath != "PLAN.md" || len(state.Items) != 2 {
		t.Fatalf("finishing the plan wiped its record: %#v", state)
	}
	// The transition itself has to be persisted. It used to be gated on the
	// marker count, so a plan could finish without the change being saved.
	if len(published) == 0 {
		t.Fatal("completing the plan was not published to the host")
	}
	if published[len(published)-1].Phase != PhaseIdle {
		t.Fatalf("last published phase = %q, want idle", published[len(published)-1].Phase)
	}
}

// TestEditedPlanDropsStaleCompletionInsteadOfMisattributingIt covers the
// positional-step hazard: checklist steps are numbered by position, so merging
// saved completion into an edited plan silently marks whatever task now sits at
// that position as done.
func TestEditedPlanDropsStaleCompletionInsteadOfMisattributingIt(t *testing.T) {
	root := t.TempDir()
	original := "- [ ] set up the database\n- [ ] delete the staging data\n"
	edited := "- [ ] delete the staging data\n- [ ] set up the database\n"
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte(edited), 0o600); err != nil {
		t.Fatal(err)
	}

	var warnings []string
	manager, err := New(root, nil, Options{
		Initial: &State{
			Phase: PhaseExecuting, PlanPath: "PLAN.md", PlanHash: hashPlan([]byte(original)),
			Items: []ChecklistItem{{Step: 1, Text: "set up the database", Completed: true}, {Step: 2, Text: "delete the staging data"}},
		},
		Warn: func(message string) { warnings = append(warnings, message) },
	})
	if err != nil {
		t.Fatal(err)
	}

	items := manager.State().Items
	if len(items) != 2 {
		t.Fatalf("items = %#v", items)
	}
	// Step 1 is now "delete the staging data". Carrying the old completion
	// across would mark it done without it ever having run.
	if items[0].Completed {
		t.Fatalf("stale completion was carried onto %q after the plan was edited", items[0].Text)
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one saying progress was reset", warnings)
	}
}

// TestUneditedPlanKeepsItsCompletion is the negative control: the guard above
// must not throw away progress every time a plan is merely reopened.
func TestUneditedPlanKeepsItsCompletion(t *testing.T) {
	root := t.TempDir()
	plan := "- [ ] first\n- [ ] second\n"
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte(plan), 0o600); err != nil {
		t.Fatal(err)
	}
	manager, err := New(root, nil, Options{Initial: &State{
		Phase: PhaseExecuting, PlanPath: "PLAN.md", PlanHash: hashPlan([]byte(plan)),
		Items: []ChecklistItem{{Step: 1, Text: "first", Completed: true}, {Step: 2, Text: "second"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	items := manager.State().Items
	if len(items) != 2 || !items[0].Completed {
		t.Fatalf("completion was dropped for an unedited plan: %#v", items)
	}
}

// TestTwoManagersDoNotShareAPhase is the regression the README documented and
// that no test covered: two sessions in one workspace shared one state file, so
// window B pressing /planner turned window A's mode off. Nothing in this
// package ever built two Managers before.
func TestTwoManagersDoNotShareAPhase(t *testing.T) {
	root := t.TempDir()

	first, err := New(root, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}
	second, err := New(root, nil, Options{})
	if err != nil {
		t.Fatal(err)
	}

	if err := first.Enter(); err != nil {
		t.Fatal(err)
	}
	if got := second.State().Phase; got != PhaseIdle {
		t.Fatalf("the second session's phase became %q when the first entered planning", got)
	}
	if _, err := second.Toggle(); err != nil {
		t.Fatal(err)
	}
	if _, err := second.Toggle(); err != nil {
		t.Fatal(err)
	}
	if got := first.State().Phase; got != PhasePlanning {
		t.Fatalf("the first session's phase became %q when the second toggled its own", got)
	}
}

// TestTogglingKeepsPlanPathAndItems covers a live data-loss bug that has
// nothing to do with scoping: Enter, Exit and Toggle assigned a fresh State{}
// literal, so one /planner keypress wiped the checklist of an approved plan
// with no way to get it back.
func TestTogglingKeepsPlanPathAndItems(t *testing.T) {
	root := t.TempDir()
	plan := "- [x] first\n- [ ] second\n"
	if err := os.WriteFile(filepath.Join(root, "PLAN.md"), []byte(plan), 0o600); err != nil {
		t.Fatal(err)
	}
	manager, err := New(root, nil, Options{Initial: &State{
		Phase:    PhaseExecuting,
		PlanPath: "PLAN.md",
		Items:    []ChecklistItem{{Step: 1, Text: "first", Completed: true}, {Step: 2, Text: "second"}},
		PlanHash: hashPlan([]byte(plan)),
	}})
	if err != nil {
		t.Fatal(err)
	}

	for _, transition := range []func() error{
		manager.Enter,
		func() error { _, err := manager.Toggle(); return err },
		manager.Exit,
	} {
		if err := transition(); err != nil {
			t.Fatal(err)
		}
		state := manager.State()
		if state.PlanPath != "PLAN.md" {
			t.Fatalf("PlanPath was lost by a phase change: %#v", state)
		}
		if len(state.Items) != 2 || !state.Items[0].Completed {
			t.Fatalf("checklist was lost by a phase change: %#v", state.Items)
		}
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
	manager, err := New(root, reviewer, Options{})
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
	manager, _ := New(root, reviewerStub{err: context.Canceled}, Options{})
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
