package main

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/sessionlog"
)

func infoFor(id, name, first, text string) sessionlog.Info {
	return sessionlog.Info{
		ID: id, Name: name, FirstMessage: first, SearchText: text,
		Cwd: "/w", Modified: time.Now(), Messages: 4,
	}
}

// TestPickerSearchesTranscriptsNotJustMetadata is the reason SearchText exists.
// Nobody remembers a session by its first message; they remember what was
// discussed in the middle of it, and a metadata-only match cannot find that.
func TestPickerSearchesTranscriptsNotJustMetadata(t *testing.T) {
	sessions := []sessionlog.Info{
		infoFor("aaaa1111", "", "hello there", "we fixed the SSE reader together"),
		infoFor("bbbb2222", "", "unrelated", "something about tax returns"),
	}

	got := fullscreenResumeSuggestions(sessions, "sse reader", "")
	if len(got) != 1 {
		t.Fatalf("matched %d sessions, want the one whose transcript mentions it", len(got))
	}
	if got[0].label != "aaaa1111" {
		t.Fatalf("matched %q", got[0].label)
	}
}

// TestPickerMatchesEveryTermSeparately copies the /model picker's behaviour: a
// search box that breaks on a space is worse than no search box.
func TestPickerMatchesEveryTermSeparately(t *testing.T) {
	sessions := []sessionlog.Info{
		infoFor("aaaa1111", "refactor", "", "the streaming parser and the retry loop"),
		infoFor("bbbb2222", "streaming", "", "nothing about retries here"),
	}

	got := fullscreenResumeSuggestions(sessions, "streaming retry", "")
	if len(got) != 1 || got[0].label != "aaaa1111" {
		t.Fatalf("multi-term search matched %v, want only the session containing both", got)
	}

	// Negative control: a term present in neither must match nothing.
	if got := fullscreenResumeSuggestions(sessions, "kubernetes", ""); len(got) != 0 {
		t.Fatalf("matched %d sessions for a term nothing contains", len(got))
	}
}

// TestPickerExcludesTheCurrentSession keeps the list from offering a resume
// that would be a no-op.
func TestPickerExcludesTheCurrentSession(t *testing.T) {
	current := infoFor("aaaa1111", "current", "", "")
	current.Path = "/sessions/current.jsonl"
	other := infoFor("bbbb2222", "other", "", "")
	other.Path = "/sessions/other.jsonl"

	got := fullscreenResumeSuggestions([]sessionlog.Info{current, other}, "", current.Path)
	if len(got) != 1 || got[0].label != "bbbb2222" {
		t.Fatalf("picker offered %v, want only the session that is not current", got)
	}
}

// TestPickerRowNamesTheSessionAndItsState covers what a user needs to choose:
// when it was, how big it is, and whether another window already has it.
func TestPickerRowNamesTheSessionAndItsState(t *testing.T) {
	info := infoFor("aaaa1111", "the SSE fix", "", "")
	info.Locked = true

	label, description := describeSession(info, false)
	if label != "aaaa1111" {
		t.Fatalf("label = %q", label)
	}
	for _, want := range []string{"the SSE fix", "4 msg", "open elsewhere"} {
		if !strings.Contains(description, want) {
			t.Fatalf("description %q does not mention %q", description, want)
		}
	}
}

// TestSwitchingSessionsReplacesTheTranscriptAndReleasesTheClaim covers
// /resume: the old session's claim must be released, or switching back to it
// later fails against a lock this same process still holds.
func TestSwitchingSessionsReplacesTheTranscriptAndReleasesTheClaim(t *testing.T) {
	agentDir := t.TempDir()
	t.Setenv("GOSHCODER_AGENT_DIR", agentDir)
	cwd := t.TempDir()
	store := defaultStore()

	// A session to switch away from, and one to switch to.
	first := newRecordingFixtureIn(t, store, cwd)
	first.session.recordUserMessage("the first conversation")
	first.session.log.appendMessage(assistant("first answer", 0))
	firstPath := first.session.log.path()

	other, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	otherRecorder := newSessionRecorder(other, nil)
	otherRecorder.appendMessage(llm.UserMessage{Role: "user", Content: "the second conversation"})
	otherRecorder.appendMessage(assistant("second answer", 0))
	otherID := other.ID()
	if err := otherRecorder.close(); err != nil {
		t.Fatalf("close other: %v", err)
	}

	// The full id, not an 8-character prefix: uuidv7 leads with the
	// millisecond, so two sessions created in the same second share one.
	if err := first.session.switchSession(otherID); err != nil {
		t.Fatalf("switchSession: %v", err)
	}

	messages := first.session.agent.State().Messages
	if len(messages) != 2 {
		t.Fatalf("transcript has %d messages after switching, want the target session's 2", len(messages))
	}
	user, ok := messages[0].(llm.UserMessage)
	if !ok {
		t.Fatalf("first message is %T", messages[0])
	}
	if text, _ := user.StringContent(); text != "the second conversation" {
		t.Fatalf("transcript = %q, want the session switched to", text)
	}

	// The session we left must be free for anyone -- including us -- to reopen.
	reattached, _, err := store.Attach(firstPath)
	if err != nil {
		t.Fatalf("the previous session's claim was not released: %v", err)
	}
	reattached.Close()
}

// TestAmbiguousSessionPrefixIsRefusedRatherThanGuessed covers the consequence
// of uuidv7 leading with a timestamp: two sessions created in the same second
// share their first eight characters, so a short prefix can name two files.
// Picking one would silently resume the wrong conversation.
func TestAmbiguousSessionPrefixIsRefusedRatherThanGuessed(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	cwd := t.TempDir()
	store := defaultStore()

	var ids []string
	for i := 0; i < 2; i++ {
		writer, err := store.Create(cwd, "")
		if err != nil {
			t.Fatalf("create: %v", err)
		}
		recorder := newSessionRecorder(writer, nil)
		recorder.appendMessage(assistant("answer", 0))
		ids = append(ids, writer.ID())
		if err := recorder.close(); err != nil {
			t.Fatalf("close: %v", err)
		}
	}
	if ids[0][:8] != ids[1][:8] {
		t.Skip("the two sessions did not land in the same millisecond window")
	}

	if _, err := store.Resolve(cwd, ids[0][:8]); !errors.Is(err, sessionlog.ErrAmbiguous) {
		t.Fatalf("Resolve error = %v, want ErrAmbiguous rather than a guess", err)
	}
	// Each full id must still select exactly one.
	for _, id := range ids {
		info, err := store.Resolve(cwd, id)
		if err != nil {
			t.Fatalf("Resolve(%s): %v", id, err)
		}
		if info.ID != id {
			t.Fatalf("Resolve(%s) = %s", id, info.ID)
		}
	}
}

// TestListingWidensCollidingShortIDs is the display half of the same problem:
// showing a prefix the user cannot then type back is worse than showing a
// longer one.
func TestListingWidensCollidingShortIDs(t *testing.T) {
	sessions := []sessionlog.Info{
		{ID: "01a02d09-4d21-7f0a-9b55-000000000001"},
		{ID: "01a02d09-4d21-7f0a-9b55-000000000002"},
	}
	labels := sessionlog.ShortIDs(sessions)
	if labels[0] == labels[1] {
		t.Fatalf("both sessions render as %q; the label cannot select either", labels[0])
	}
	for index, label := range labels {
		if !strings.HasPrefix(sessions[index].ID, label) {
			t.Fatalf("label %q is not a prefix of %q, so it cannot be typed back", label, sessions[index].ID)
		}
	}

	// Negative control: distinct sessions must not be widened needlessly.
	distinct := []sessionlog.Info{
		{ID: "aaaaaaaa-0000-7000-8000-000000000001"},
		{ID: "bbbbbbbb-0000-7000-8000-000000000002"},
	}
	for _, label := range sessionlog.ShortIDs(distinct) {
		if len(label) != 8 {
			t.Fatalf("label %q was widened even though eight characters suffice", label)
		}
	}
}

// TestSwitchingToTheCurrentSessionIsRefused keeps a no-op from closing and
// reopening the file for nothing.
func TestSwitchingToTheCurrentSessionIsRefused(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	cwd := t.TempDir()
	f := newRecordingFixtureIn(t, defaultStore(), cwd)
	f.session.log.appendMessage(assistant("answer", 0))
	f.session.log.sync()

	if err := f.session.switchSession(f.session.log.id()); err == nil {
		t.Fatal("switching to the current session was allowed")
	}
	if !f.session.recordingActive() {
		t.Fatal("a refused switch left the session not recording")
	}
}

// TestMissingWorkspaceIsReported covers resuming a session whose directory
// moved or was deleted. Without it the first tool call fails inside os.OpenRoot
// with an error naming neither the session nor the directory it wanted.
func TestMissingWorkspaceIsReported(t *testing.T) {
	gone := filepath.Join(t.TempDir(), "deleted-checkout")
	notice := missingWorkspaceNotice(sessionlog.Info{Cwd: gone}, t.TempDir())
	if !strings.Contains(notice, "no longer exists") {
		t.Fatalf("notice = %q, want it to say the original workspace is gone", notice)
	}

	// A workspace that merely differs is a weaker warning, not the same one.
	existing := t.TempDir()
	notice = missingWorkspaceNotice(sessionlog.Info{Cwd: existing}, t.TempDir())
	if notice == "" || strings.Contains(notice, "no longer exists") {
		t.Fatalf("notice = %q, want it to distinguish a moved workspace from a deleted one", notice)
	}

	// Negative control: the same directory needs no warning at all.
	same := t.TempDir()
	if notice := missingWorkspaceNotice(sessionlog.Info{Cwd: same}, same); notice != "" {
		t.Fatalf("notice = %q for an unchanged workspace", notice)
	}
}

// TestSessionListLinesStayShort covers a real constraint of the fullscreen
// interface: runFullscreenCommand folds a command's whole output into one
// capped card, so a fifty-row listing arrives as a single unscrollable blob.
func TestSessionListLinesStayShort(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	cwd := t.TempDir()
	store := defaultStore()

	for i := 0; i < 25; i++ {
		writer, err := store.Create(cwd, "")
		if err != nil {
			t.Fatalf("create: %v", err)
		}
		recorder := newSessionRecorder(writer, nil)
		recorder.appendMessage(assistant("answer", 0))
		if err := recorder.close(); err != nil {
			t.Fatalf("close: %v", err)
		}
	}

	lines := sessionListLines(cwd)
	if len(lines) > 12 {
		t.Fatalf("/sessions produced %d lines; it must stay short enough to read in one card", len(lines))
	}
	if !strings.Contains(strings.Join(lines, "\n"), "more") {
		t.Fatal("the listing did not say how many sessions it left out")
	}
}

// TestTwoSessionsInOneWorkspaceHaveIndependentPlannerPhases is the regression
// the README documented: plan state was keyed by a hash of the workspace path,
// so two windows in one repository shared one phase and the last writer won.
func TestTwoSessionsInOneWorkspaceHaveIndependentPlannerPhases(t *testing.T) {
	root := t.TempDir()

	var firstSaved, secondSaved []plannotator.State
	first, err := plannotator.New(root, nil, plannotator.Options{
		OnChange: func(state plannotator.State) { firstSaved = append(firstSaved, state) },
	})
	if err != nil {
		t.Fatal(err)
	}
	second, err := plannotator.New(root, nil, plannotator.Options{
		OnChange: func(state plannotator.State) { secondSaved = append(secondSaved, state) },
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := first.Enter(); err != nil {
		t.Fatal(err)
	}
	if got := second.State().Phase; got != plannotator.PhaseIdle {
		t.Fatalf("the second window's phase became %q when the first entered planning", got)
	}
	if len(secondSaved) != 0 {
		t.Fatalf("the second window persisted %d states it never asked for", len(secondSaved))
	}
	if len(firstSaved) != 1 || firstSaved[0].Phase != plannotator.PhasePlanning {
		t.Fatalf("the first window persisted %v", firstSaved)
	}
}

// TestPlannerStateRoundTripsThroughTheSessionLog is what makes -continue
// restore a plan now that no per-workspace file exists.
func TestPlannerStateRoundTripsThroughTheSessionLog(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.recordCustom(plannerCustomType, plannotator.State{
		Phase: plannotator.PhasePlanning, PlanPath: "PLAN.md",
	})
	f.session.log.appendMessage(assistant("planning", 0))

	got := f.reopen(t)
	raw, ok := got.Custom[plannerCustomType]
	if !ok {
		t.Fatal("the Planner state was not recorded in the session")
	}
	var notices []string
	opened := &openedSession{Resumed: true, restored: got}
	state := restoredPlannerState(opened, f.cwd, &notices)
	if state == nil || state.Phase != plannotator.PhasePlanning || state.PlanPath != "PLAN.md" {
		t.Fatalf("restored planner state = %#v (raw %s)", state, raw)
	}

	// A plan entry never reaches the model: it is state, not conversation.
	for _, message := range got.Messages {
		if text, ok := message.(llm.UserMessage); ok {
			if content, _ := text.StringContent(); strings.Contains(content, "PLAN.md") {
				t.Fatal("the Planner state leaked into the model's context")
			}
		}
	}
}

// TestCorruptPlannerStateYieldsIdleRatherThanFailingTheSession covers the risk
// the rescoping creates: plan state now shares a file with the transcript, so a
// payload that will not parse must not be a reason the session refuses to open.
func TestCorruptPlannerStateYieldsIdleRatherThanFailingTheSession(t *testing.T) {
	opened := &openedSession{Resumed: true}
	opened.restored.Custom = map[string]json.RawMessage{plannerCustomType: json.RawMessage("{not json")}

	var notices []string
	state := restoredPlannerState(opened, t.TempDir(), &notices)
	if state != nil {
		t.Fatalf("a corrupt payload produced state %#v", state)
	}
	if len(notices) != 1 || !strings.Contains(notices[0], "could not be read") {
		t.Fatalf("notices = %v, want one explaining the state was ignored", notices)
	}
}

// newRecordingFixtureIn is newRecordingFixture against a caller-supplied store,
// for tests that need two sessions in one workspace.
func newRecordingFixtureIn(t *testing.T, store *sessionlog.Store, cwd string) *recordingFixture {
	t.Helper()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	model := &llm.Model{ID: "test-model", Name: "Test", Provider: "test", API: "test-api", ContextWindow: 128000}
	s := newTestSessionWithModel(model)
	s.log = newSessionRecorder(writer, s.pushNotice)
	s.agent.Subscribe(s.log.listener())
	t.Cleanup(func() { _ = s.log.close() })
	return &recordingFixture{session: s, store: store, cwd: cwd, path: writer.Path()}
}
