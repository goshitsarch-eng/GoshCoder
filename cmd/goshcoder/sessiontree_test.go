package main

import (
	"os"
	"strings"
	"testing"

	"goshcoder/internal/llm"
	"goshcoder/internal/sessionlog"
)

// seedConversation records n question/answer pairs.
func seedConversation(f *recordingFixture, pairs int) {
	for i := 1; i <= pairs; i++ {
		f.session.recordUserMessage(question(i))
		f.session.log.appendMessage(assistant(answer(i), 0.01))
	}
	f.session.log.sync()
}

func question(n int) string { return "question " + string(rune('0'+n)) }
func answer(n int) string   { return "answer " + string(rune('0'+n)) }

// TestBranchPointsListOnlyUserMessages covers the constraint that makes a
// rewind safe: stopping in the middle of an assistant turn would leave a tool
// call with no result, which every provider rejects on the next request.
func TestBranchPointsListOnlyUserMessages(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 3)

	points := branchPoints(f.session.log.snapshot(), f.session.log.leaf())
	if len(points) != 3 {
		t.Fatalf("found %d branch points, want one per user message", len(points))
	}
	for _, point := range points {
		if point.Role != "user" {
			t.Fatalf("branch point %d has role %q", point.Index, point.Role)
		}
	}
	if points[0].Index != 1 || points[2].Index != 3 {
		t.Fatalf("points are not numbered from one: %+v", points)
	}
}

// TestForkRewindsTheContextWithoutDeletingAnything is the whole promise of
// branching: take a different path, and the one you left is still there.
func TestForkRewindsTheContextWithoutDeletingAnything(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 3)

	entriesBefore := f.session.log.snapshot().Len()
	if err := f.session.forkTo("2"); err != nil {
		t.Fatalf("forkTo: %v", err)
	}

	// Context is now everything up to and including the second question.
	messages := f.session.agent.State().Messages
	if len(messages) != 3 {
		t.Fatalf("context has %d messages after rewinding to point 2, want 3", len(messages))
	}
	last, ok := messages[len(messages)-1].(llm.UserMessage)
	if !ok {
		t.Fatalf("last context message is %T, want the question rewound to", messages[len(messages)-1])
	}
	if text, _ := last.StringContent(); text != question(2) {
		t.Fatalf("context ends at %q, want %q", text, question(2))
	}

	// Nothing was removed: the abandoned path is still in the file.
	tree := f.session.log.snapshot()
	if tree.Len() < entriesBefore {
		t.Fatalf("the file lost entries: %d -> %d", entriesBefore, tree.Len())
	}
	var foundAbandoned bool
	for _, entry := range tree.All() {
		if entry.Type == sessionlog.TypeMessage && strings.Contains(entryText(entry), answer(3)) {
			foundAbandoned = true
		}
	}
	if !foundAbandoned {
		t.Fatal("the abandoned branch was deleted rather than left in place")
	}
}

// TestNewTurnAfterAForkAttachesToTheRewindPoint is what makes the branch a
// branch rather than a truncation.
func TestNewTurnAfterAForkAttachesToTheRewindPoint(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 3)
	if err := f.session.forkTo("2"); err != nil {
		t.Fatalf("forkTo: %v", err)
	}

	f.session.recordUserMessage("a different direction")
	f.session.log.appendMessage(assistant("a different answer", 0))

	got := f.reopen(t)
	var texts []string
	for _, message := range got.Messages {
		if user, ok := message.(llm.UserMessage); ok {
			text, _ := user.StringContent()
			texts = append(texts, text)
		}
	}
	joined := strings.Join(texts, "|")
	if !strings.Contains(joined, "a different direction") {
		t.Fatalf("the new turn is not on the resumed path: %q", joined)
	}
	if strings.Contains(joined, question(3)) {
		t.Fatalf("the abandoned branch is back in context: %q", joined)
	}
}

// TestForkRejectsAnOutOfRangePoint keeps a typo from silently doing something
// destructive-looking.
func TestForkRejectsAnOutOfRangePoint(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 2)
	before := len(f.session.agent.State().Messages)

	for _, argument := range []string{"0", "99", "", "two"} {
		if err := f.session.forkTo(argument); err == nil {
			t.Fatalf("forkTo(%q) was accepted", argument)
		}
	}
	if got := len(f.session.agent.State().Messages); got != before {
		t.Fatalf("a refused fork changed the transcript: %d -> %d", before, got)
	}
}

// TestLabelNamesAPointAndSurvivesAResume covers the bookmark: a branch is only
// useful if you can tell the branches apart later.
func TestLabelNamesAPointAndSurvivesAResume(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 2)

	if err := f.session.labelPoint("1 before the refactor"); err != nil {
		t.Fatalf("labelPoint: %v", err)
	}
	points := branchPoints(f.session.log.snapshot(), f.session.log.leaf())
	if points[0].Label != "before the refactor" {
		t.Fatalf("label = %q", points[0].Label)
	}

	// It must still be there in a fresh process.
	if err := f.session.log.close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	tree, _, _, err := f.store.Load(f.path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	reloaded := branchPoints(tree, tree.Leaf())
	if len(reloaded) == 0 || reloaded[0].Label != "before the refactor" {
		t.Fatalf("the label did not survive a resume: %+v", reloaded)
	}
}

// TestEmptyLabelRemovesIt is the other half; without it a mistyped label is
// permanent.
func TestEmptyLabelRemovesIt(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 1)

	if err := f.session.labelPoint("1 a name"); err != nil {
		t.Fatalf("labelPoint: %v", err)
	}
	if err := f.session.labelPoint("1"); err != nil {
		t.Fatalf("labelPoint clear: %v", err)
	}
	points := branchPoints(f.session.log.snapshot(), f.session.log.leaf())
	if points[0].Label != "" {
		t.Fatalf("label = %q, want it removed", points[0].Label)
	}
}

// TestCloneLeavesTheOriginalUntouched covers the safety of /clone: the point of
// duplicating a session is to experiment without risking the original.
func TestCloneLeavesTheOriginalUntouched(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	cwd := t.TempDir()
	f := newRecordingFixtureIn(t, defaultStore(), cwd)
	seedConversation(f, 2)

	originalID := f.session.log.id()
	originalPath := f.session.log.path()
	originalBytes, err := os.ReadFile(originalPath)
	if err != nil {
		t.Fatalf("read original: %v", err)
	}

	if err := f.session.cloneSession(); err != nil {
		t.Fatalf("cloneSession: %v", err)
	}

	if f.session.log.id() == originalID {
		t.Fatal("the session did not switch to the clone")
	}
	after, err := os.ReadFile(originalPath)
	if err != nil {
		t.Fatalf("read original after clone: %v", err)
	}
	if string(after) != string(originalBytes) {
		t.Fatal("cloning modified the original session file")
	}

	// The clone must carry the conversation and be independently appendable.
	if got := len(f.session.agent.State().Messages); got != 4 {
		t.Fatalf("the clone has %d messages, want the original's 4", got)
	}
	f.session.recordUserMessage("only in the clone")
	f.session.log.sync()
	stillOriginal, _ := os.ReadFile(originalPath)
	if strings.Contains(string(stillOriginal), "only in the clone") {
		t.Fatal("a write to the clone reached the original")
	}
}

// TestBranchingIsRefusedWhenNothingIsBeingSaved keeps the commands from
// pretending to work in a -no-session run.
func TestBranchingIsRefusedWhenNothingIsBeingSaved(t *testing.T) {
	model := &llm.Model{ID: "test-model", Provider: "test", API: "test-api"}
	s := newTestSessionWithModel(model)

	for name, run := range map[string]func() error{
		"fork":  func() error { return s.forkTo("1") },
		"label": func() error { return s.labelPoint("1 x") },
		"clone": s.cloneSession,
	} {
		if err := run(); err == nil {
			t.Fatalf("/%s succeeded with no session being saved", name)
		}
	}
}

// TestTreeLinesMarkTheCurrentPointAndBranchCounts covers what the listing has
// to convey for the numbers to be usable.
func TestTreeLinesMarkTheCurrentPointAndBranchCounts(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 3)
	if err := f.session.forkTo("1"); err != nil {
		t.Fatalf("forkTo: %v", err)
	}
	f.session.recordUserMessage("second branch")
	f.session.log.appendMessage(assistant("second answer", 0))

	lines := treeLines(f.session.log.snapshot(), f.session.log.leaf())
	joined := strings.Join(lines, "\n")
	if !strings.Contains(joined, "*") {
		t.Fatalf("no current-point marker in:\n%s", joined)
	}
	if !strings.Contains(joined, "branches") {
		t.Fatalf("the point with two branches is not marked as such:\n%s", joined)
	}
}

// TestExportMarkdownRendersTheConversation covers the readable export. HTML
// export is deliberately not implemented: pi's inlines ~165 KB of vendored
// JavaScript into every file, and Markdown is what people paste into an issue.
func TestExportMarkdownRendersTheConversation(t *testing.T) {
	f := newRecordingFixture(t)
	seedConversation(f, 2)
	f.session.log.setName("the export test")
	f.session.log.sync()

	tree, header, _, err := f.store.Load(f.path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	info, err := f.store.Resolve(f.cwd, f.session.log.id())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	markdown := exportMarkdown(info, header, tree)
	for _, want := range []string{"# the export test", "## User", "## Assistant", question(1), answer(2)} {
		if !strings.Contains(markdown, want) {
			t.Fatalf("the export does not contain %q:\n%s", want, markdown)
		}
	}
}

// TestImportAdoptsASessionIntoThisWorkspace covers handing a session to someone
// else: the copy has to belong to the importer's workspace, not the exporter's.
func TestImportAdoptsASessionIntoThisWorkspace(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	store := defaultStore()
	origin := t.TempDir()

	source := newRecordingFixtureIn(t, store, origin)
	seedConversation(source, 2)
	sourcePath := source.session.log.path()
	if err := source.session.log.close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	destination := t.TempDir()
	info, err := store.Resolve(destination, sourcePath)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	writer, err := store.Fork(info, nil, destination)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	importedPath := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, header, _, err := store.Load(importedPath)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if header.Cwd != destination {
		t.Fatalf("the imported session belongs to %s, want %s", header.Cwd, destination)
	}
	if header.ParentSession == "" {
		t.Fatal("the import does not record where it came from")
	}
	// It must be listed in the importing workspace, not the original one.
	listed, err := store.List(destination, sessionlog.ListOptions{})
	if err != nil || len(listed) != 1 {
		t.Fatalf("List: %v (%d sessions)", err, len(listed))
	}
}
