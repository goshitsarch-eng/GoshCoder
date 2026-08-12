package ralph

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

// fakeQueue captures the follow-up prompts a loop generates.
type fakeQueue struct {
	mu       sync.Mutex
	messages []agent.Message
	pending  bool
}

func (q *fakeQueue) FollowUp(message agent.Message) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.messages = append(q.messages, message)
}

func (q *fakeQueue) HasQueuedMessages() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.pending
}

func (q *fakeQueue) texts() []string {
	q.mu.Lock()
	defer q.mu.Unlock()
	var out []string
	for _, message := range q.messages {
		if user, ok := message.(llm.UserMessage); ok {
			if text, isString := user.StringContent(); isString {
				out = append(out, text)
			}
		}
	}
	return out
}

func newTestStore(t *testing.T) *Store {
	t.Helper()
	return NewStore(t.TempDir(), "session-1")
}

// runTool executes a tool and returns its text output.
func runTool(t *testing.T, tool agent.Tool, params map[string]any) (string, error) {
	t.Helper()
	result, err := tool.Execute(context.Background(), "call-1", params, func(agent.ToolResult) {})
	if err != nil {
		return "", err
	}
	var parts []string
	for _, block := range result.Content {
		if text, ok := block.(llm.TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n"), nil
}

func TestSanitizeName(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"refactor-auth", "refactor-auth"},
		{"my loop", "my_loop"},
		{"a//b\\c", "a_b_c"},
		{"lots   of   spaces", "lots_of_spaces"},
		{"keep_under_scores", "keep_under_scores"},
		// Punctuation collapses to a single underscore rather than vanishing.
		{"///", "_"},
	}
	for _, tt := range tests {
		if got := SanitizeName(tt.in); got != tt.want {
			t.Errorf("SanitizeName(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestStartCreatesTaskFileAndState(t *testing.T) {
	store := newTestStore(t)

	state, err := store.Start("my loop", "# Task\n\n- [ ] one", Options{MaxIterations: 5})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if state.Name != "my_loop" {
		t.Fatalf("name = %q, want the sanitized form", state.Name)
	}
	if state.Status != StatusActive || state.Iteration != 1 {
		t.Fatalf("state = %#v", state)
	}
	if state.OwnerSessionID != "session-1" {
		t.Fatalf("owner = %q", state.OwnerSessionID)
	}

	// The task file and state file both land under .ralph.
	taskPath := filepath.Join(store.Root, Dir, "my_loop.md")
	content, err := os.ReadFile(taskPath)
	if err != nil {
		t.Fatalf("task file: %v", err)
	}
	if string(content) != "# Task\n\n- [ ] one" {
		t.Fatalf("task content = %q", content)
	}
	if _, err := os.Stat(filepath.Join(store.Root, Dir, "my_loop.state.json")); err != nil {
		t.Fatalf("state file: %v", err)
	}
}

func TestStartDefaultsMaxIterations(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if state.MaxIterations != DefaultMaxIterations {
		t.Fatalf("maxIterations = %d, want %d", state.MaxIterations, DefaultMaxIterations)
	}
}

func TestStartUsesTemplateWhenEmpty(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "", Options{})
	if err != nil {
		t.Fatal(err)
	}
	content, err := store.ReadTask(state)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(content, "## Checklist") {
		t.Fatalf("task content = %q, want the default template", content)
	}
}

func TestStartRejectsDuplicateActiveLoop(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.Start("loop", "task", Options{}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Start("loop", "task", Options{}); err == nil {
		t.Fatal("a second active loop with the same name must be rejected")
	}
}

func TestStartRejectsEmptyName(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.Start("", "task", Options{}); err == nil {
		t.Fatal("an empty name must be rejected")
	}
	// Punctuation-only names are not empty: they sanitize to underscores, which
	// is a usable filename stem.
	if _, err := store.Start("///", "task", Options{}); err != nil {
		t.Fatalf("a punctuation-only name should sanitize to a usable stem: %v", err)
	}
}

// TestStateRoundTripsOnDisk confirms the persisted JSON uses pi's field names,
// which is what keeps a .ralph directory interoperable.
func TestStateRoundTripsOnDisk(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.Start("loop", "task", Options{MaxIterations: 7, ItemsPerIteration: 3, ReflectEvery: 2}); err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(filepath.Join(store.Root, Dir, "loop.state.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{
		"name", "taskFile", "iteration", "maxIterations",
		"itemsPerIteration", "reflectEvery", "status", "startedAt", "active",
	} {
		if _, present := fields[key]; !present {
			t.Errorf("state JSON is missing %q: %v", key, fields)
		}
	}
	// active mirrors status for older readers.
	if fields["active"] != true || fields["status"] != StatusActive {
		t.Fatalf("active/status = %v/%v", fields["active"], fields["status"])
	}
}

// TestLoadMigratesLegacyState covers state written before the status field
// existed, and the renamed reflection fields.
func TestLoadMigratesLegacyState(t *testing.T) {
	store := newTestStore(t)
	dir := filepath.Join(store.Root, Dir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	legacy := `{"name":"old","taskFile":".ralph/old.md","iteration":3,"maxIterations":10,"active":true}`
	if err := os.WriteFile(filepath.Join(dir, "old.state.json"), []byte(legacy), 0o644); err != nil {
		t.Fatal(err)
	}

	state, ok := store.Load("old", false)
	if !ok {
		t.Fatal("legacy state did not load")
	}
	// active: true becomes status: active.
	if state.Status != StatusActive {
		t.Fatalf("status = %q, want active", state.Status)
	}
	if state.ReflectInstructions == "" {
		t.Fatal("reflect instructions should default")
	}
}

func TestAdvanceIncrementsAndReflects(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{MaxIterations: 10, ReflectEvery: 2})
	if err != nil {
		t.Fatal(err)
	}

	// Iteration 1 -> 2. (2-1) % 2 == 1, so no reflection.
	done, reflect, err := store.Advance(state)
	if err != nil || done {
		t.Fatalf("advance: done = %v, err = %v", done, err)
	}
	if reflect {
		t.Fatal("iteration 2 should not reflect with reflectEvery=2")
	}
	if state.Iteration != 2 {
		t.Fatalf("iteration = %d", state.Iteration)
	}

	// Iteration 2 -> 3. (3-1) % 2 == 0, so reflect.
	_, reflect, err = store.Advance(state)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect {
		t.Fatal("iteration 3 should reflect with reflectEvery=2")
	}
	if state.LastReflectionAt != 3 {
		t.Fatalf("lastReflectionAt = %d", state.LastReflectionAt)
	}
}

func TestAdvanceCompletesAtMaxIterations(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{MaxIterations: 2})
	if err != nil {
		t.Fatal(err)
	}

	// 1 -> 2 is still within the limit.
	done, _, err := store.Advance(state)
	if err != nil || done {
		t.Fatalf("done = %v, err = %v", done, err)
	}
	// 2 -> 3 exceeds it.
	done, _, err = store.Advance(state)
	if err != nil {
		t.Fatal(err)
	}
	if !done {
		t.Fatal("exceeding maxIterations should finish the loop")
	}
	if state.Status != StatusCompleted || state.CompletedAt == "" {
		t.Fatalf("state = %#v", state)
	}

	// The completion is persisted.
	reloaded, ok := store.Load("loop", false)
	if !ok || reloaded.Status != StatusCompleted {
		t.Fatalf("reloaded = %#v", reloaded)
	}
}

// TestUnboundedLoopNeverCompletesOnCount covers a negative max, which means
// "no iteration limit".
func TestUnboundedLoopNeverCompletesOnCount(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{MaxIterations: -1})
	if err != nil {
		t.Fatal(err)
	}
	for range 5 {
		done, _, err := store.Advance(state)
		if err != nil {
			t.Fatal(err)
		}
		if done {
			t.Fatal("an unbounded loop should not finish on iteration count")
		}
	}
}

func TestPauseResumeAndCurrent(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Current(); !ok {
		t.Fatal("the started loop should be current")
	}

	if err := store.Pause(state); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Current(); ok {
		t.Fatal("a paused loop must not be current")
	}

	resumed, err := store.Resume("loop")
	if err != nil {
		t.Fatalf("Resume: %v", err)
	}
	if resumed.Status != StatusActive {
		t.Fatalf("resumed status = %q", resumed.Status)
	}
	if _, ok := store.Current(); !ok {
		t.Fatal("a resumed loop should be current")
	}
}

func TestResumeRejectsCompletedAndUnknown(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Complete(state); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Resume("loop"); err == nil {
		t.Fatal("a completed loop must not resume")
	}
	if _, err := store.Resume("nope"); err == nil {
		t.Fatal("an unknown loop must not resume")
	}
}

// TestCurrentIgnoresOtherSessions confirms loop ownership is per session, so
// two CLIs in one workspace do not fight over the same loop.
func TestCurrentIgnoresOtherSessions(t *testing.T) {
	root := t.TempDir()
	owner := NewStore(root, "session-a")
	if _, err := owner.Start("loop", "task", Options{}); err != nil {
		t.Fatal(err)
	}

	other := NewStore(root, "session-b")
	if state, ok := other.Current(); ok {
		t.Fatalf("another session adopted the loop: %#v", state)
	}
	// The owner still sees it.
	if _, ok := owner.Current(); !ok {
		t.Fatal("the owning session lost its loop")
	}
}

// TestCurrentAdoptsUnownedLoop covers state written without an owner, which a
// fresh session may pick up.
func TestCurrentAdoptsUnownedLoop(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, Dir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	unowned := `{"name":"orphan","taskFile":".ralph/orphan.md","iteration":1,"maxIterations":5,"status":"active"}`
	if err := os.WriteFile(filepath.Join(dir, "orphan.state.json"), []byte(unowned), 0o644); err != nil {
		t.Fatal(err)
	}

	store := NewStore(root, "session-new")
	state, ok := store.Current()
	if !ok || state.Name != "orphan" {
		t.Fatalf("current = %#v, ok = %v", state, ok)
	}
}

func TestListAndArchive(t *testing.T) {
	store := newTestStore(t)
	for _, name := range []string{"b-loop", "a-loop"} {
		state, err := store.Start(name, "task", Options{})
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Pause(state); err != nil {
			t.Fatal(err)
		}
	}

	loops := store.List(false)
	if len(loops) != 2 {
		t.Fatalf("loops = %#v", loops)
	}
	// Sorted by name.
	if loops[0].Name != "a-loop" || loops[1].Name != "b-loop" {
		t.Fatalf("order = %q, %q", loops[0].Name, loops[1].Name)
	}

	if err := store.Archive("a-loop"); err != nil {
		t.Fatalf("Archive: %v", err)
	}
	if len(store.List(false)) != 1 {
		t.Fatal("the archived loop should leave the active list")
	}
	archived := store.List(true)
	if len(archived) != 1 || archived[0].Name != "a-loop" {
		t.Fatalf("archived = %#v", archived)
	}
}

func TestDelete(t *testing.T) {
	store := newTestStore(t)
	if _, err := store.Start("loop", "task", Options{}); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete("loop"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, ok := store.Load("loop", false); ok {
		t.Fatal("the state file should be gone")
	}
	if _, err := os.Stat(filepath.Join(store.Root, Dir, "loop.md")); !os.IsNotExist(err) {
		t.Fatal("the task file should be gone")
	}
	if err := store.Delete("loop"); err == nil {
		t.Fatal("deleting a missing loop should error")
	}
}

func TestBuildPrompt(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{MaxIterations: 5, ItemsPerIteration: 3})
	if err != nil {
		t.Fatal(err)
	}

	prompt := BuildPrompt(state, "# Task\n- [ ] one", false)
	for _, want := range []string{
		"RALPH LOOP: loop | Iteration 1/5",
		"## Current Task (from .ralph/loop.md)",
		"- [ ] one",
		"COMPLETION GATE",
		"STALE PROMPT GUARD",
		"process approximately 3 items",
		CompleteMarker,
		"ralph_done",
	} {
		if !strings.Contains(prompt, want) {
			t.Errorf("prompt is missing %q", want)
		}
	}
	// Reflection text only appears on reflection turns.
	if strings.Contains(prompt, "REFLECTION CHECKPOINT") {
		t.Error("a non-reflection prompt must not carry the checkpoint text")
	}

	reflection := BuildPrompt(state, "task", true)
	if !strings.Contains(reflection, "REFLECTION CHECKPOINT") {
		t.Error("a reflection prompt should carry the checkpoint text")
	}
}

func TestSystemPromptSuffix(t *testing.T) {
	store := newTestStore(t)
	state, err := store.Start("loop", "task", Options{MaxIterations: 4, ItemsPerIteration: 2})
	if err != nil {
		t.Fatal(err)
	}

	suffix := SystemPromptSuffix(state)
	for _, want := range []string{
		"[RALPH LOOP - loop - Iteration 1/4]",
		".ralph/loop.md",
		"~2 items",
		CompleteMarker,
		"ralph_done",
	} {
		if !strings.Contains(suffix, want) {
			t.Errorf("suffix is missing %q", want)
		}
	}
}

func TestHasCompleteMarker(t *testing.T) {
	with := llm.AssistantMessage{Content: []llm.ContentBlock{
		llm.TextContent{Type: "text", Text: "all done " + CompleteMarker},
	}}
	if !HasCompleteMarker(with) {
		t.Fatal("the marker was not detected")
	}

	without := llm.AssistantMessage{Content: []llm.ContentBlock{
		llm.TextContent{Type: "text", Text: "still working"},
	}}
	if HasCompleteMarker(without) {
		t.Fatal("a false positive on the marker")
	}
}

func TestStatusLine(t *testing.T) {
	store := newTestStore(t)
	if got := store.StatusLine(); got != "" {
		t.Fatalf("status line with no loop = %q", got)
	}

	if _, err := store.Start("loop", "task", Options{MaxIterations: 3, ReflectEvery: 2}); err != nil {
		t.Fatal(err)
	}
	line := store.StatusLine()
	for _, want := range []string{"ralph", "loop", "active", "1/3", "reflect in"} {
		if !strings.Contains(line, want) {
			t.Errorf("status line %q is missing %q", line, want)
		}
	}
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

func TestStartToolQueuesFirstIteration(t *testing.T) {
	store := newTestStore(t)
	queue := &fakeQueue{}

	out, err := runTool(t, store.StartTool(queue), map[string]any{
		"name":          "refactor auth",
		"taskContent":   "# Task\n- [ ] split the handler",
		"maxIterations": 4.0,
	})
	if err != nil {
		t.Fatalf("ralph_start: %v", err)
	}
	if !strings.Contains(out, "refactor_auth") {
		t.Fatalf("tool output = %q", out)
	}

	// The first iteration prompt is queued as a follow-up.
	prompts := queue.texts()
	if len(prompts) != 1 {
		t.Fatalf("queued prompts = %#v", prompts)
	}
	if !strings.Contains(prompts[0], "RALPH LOOP: refactor_auth | Iteration 1/4") {
		t.Fatalf("queued prompt = %q", prompts[0])
	}

	// State and task file exist on disk.
	state, ok := store.Load("refactor auth", false)
	if !ok || state.Status != StatusActive {
		t.Fatalf("state = %#v", state)
	}
	content, err := store.ReadTask(state)
	if err != nil || !strings.Contains(content, "split the handler") {
		t.Fatalf("task content = %q, err = %v", content, err)
	}
}

func TestStartToolReportsAlreadyActive(t *testing.T) {
	store := newTestStore(t)
	queue := &fakeQueue{}
	if _, err := runTool(t, store.StartTool(queue), map[string]any{"name": "loop", "taskContent": "task"}); err != nil {
		t.Fatal(err)
	}

	// A duplicate is a reported no-op rather than a tool error.
	out, err := runTool(t, store.StartTool(queue), map[string]any{"name": "loop", "taskContent": "task"})
	if err != nil {
		t.Fatalf("a duplicate start should not error: %v", err)
	}
	if !strings.Contains(out, "already active") {
		t.Fatalf("tool output = %q", out)
	}
}

func TestDoneToolAdvancesAndQueues(t *testing.T) {
	store := newTestStore(t)
	queue := &fakeQueue{}
	if _, err := runTool(t, store.StartTool(queue), map[string]any{
		"name": "loop", "taskContent": "# Task\n- [ ] one", "maxIterations": 5.0,
	}); err != nil {
		t.Fatal(err)
	}

	out, err := runTool(t, store.DoneTool(queue), nil)
	if err != nil {
		t.Fatalf("ralph_done: %v", err)
	}
	if !strings.Contains(out, "Iteration 1 complete") {
		t.Fatalf("tool output = %q", out)
	}

	prompts := queue.texts()
	if len(prompts) != 2 {
		t.Fatalf("queued prompts = %d, want 2", len(prompts))
	}
	if !strings.Contains(prompts[1], "Iteration 2/5") {
		t.Fatalf("second prompt = %q", prompts[1])
	}
}

func TestDoneToolWithoutActiveLoop(t *testing.T) {
	store := newTestStore(t)
	out, err := runTool(t, store.DoneTool(&fakeQueue{}), nil)
	if err != nil {
		t.Fatalf("ralph_done: %v", err)
	}
	if !strings.Contains(out, "No active Ralph loop") {
		t.Fatalf("tool output = %q", out)
	}
}

// TestDoneToolSkipsWhenMessagesQueued covers the guard against double-injecting
// a prompt when the user has already queued work.
func TestDoneToolSkipsWhenMessagesQueued(t *testing.T) {
	store := newTestStore(t)
	queue := &fakeQueue{}
	if _, err := runTool(t, store.StartTool(queue), map[string]any{"name": "loop", "taskContent": "task"}); err != nil {
		t.Fatal(err)
	}
	queue.pending = true

	out, err := runTool(t, store.DoneTool(queue), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "already queued") {
		t.Fatalf("tool output = %q", out)
	}
	// The iteration did not advance.
	state, _ := store.Load("loop", false)
	if state.Iteration != 1 {
		t.Fatalf("iteration = %d, want 1", state.Iteration)
	}
}

func TestDoneToolTerminatesAtMaxIterations(t *testing.T) {
	store := newTestStore(t)
	queue := &fakeQueue{}
	if _, err := runTool(t, store.StartTool(queue), map[string]any{
		"name": "loop", "taskContent": "task", "maxIterations": 1.0,
	}); err != nil {
		t.Fatal(err)
	}

	result, err := store.DoneTool(queue).Execute(context.Background(), "c1", nil, func(agent.ToolResult) {})
	if err != nil {
		t.Fatal(err)
	}
	// Terminate stops the agent loop rather than queueing more work.
	if !result.Terminate {
		t.Fatal("reaching maxIterations should terminate the batch")
	}
	state, _ := store.Load("loop", false)
	if state.Status != StatusCompleted {
		t.Fatalf("status = %q", state.Status)
	}
}

// TestDoneToolPausesOnMissingTaskFile covers a deleted task file: the loop
// pauses instead of spinning without context.
func TestDoneToolPausesOnMissingTaskFile(t *testing.T) {
	store := newTestStore(t)
	queue := &fakeQueue{}
	if _, err := runTool(t, store.StartTool(queue), map[string]any{
		"name": "loop", "taskContent": "task", "maxIterations": 5.0,
	}); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(store.Root, Dir, "loop.md")); err != nil {
		t.Fatal(err)
	}

	if _, err := runTool(t, store.DoneTool(queue), nil); err == nil {
		t.Fatal("expected an error for a missing task file")
	}
	state, _ := store.Load("loop", false)
	if state.Status != StatusPaused {
		t.Fatalf("status = %q, want paused", state.Status)
	}
}

func TestToolsExposeBothTools(t *testing.T) {
	store := newTestStore(t)
	tools := store.Tools(&fakeQueue{})
	if len(tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(tools))
	}
	names := map[string]bool{}
	for _, tool := range tools {
		if len(tool.Parameters) == 0 || tool.Execute == nil {
			t.Fatalf("tool %s is incomplete", tool.Name)
		}
		names[tool.Name] = true
	}
	if !names["ralph_start"] || !names["ralph_done"] {
		t.Fatalf("tool names = %#v", names)
	}
}
