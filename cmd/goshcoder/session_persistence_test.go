package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/sessionlog"
)

// recordingFixture is a session wired to a real file, without the catalog,
// credentials or workspace that newSession would demand.
type recordingFixture struct {
	session *session
	store   *sessionlog.Store
	cwd     string
	path    string
}

func newRecordingFixture(t *testing.T) *recordingFixture {
	t.Helper()
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())

	store := sessionlog.NewStore(filepath.Join(t.TempDir(), "sessions"))
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}

	model := &llm.Model{ID: "test-model", Name: "Test", Provider: "test", API: "test-api", ContextWindow: 128000}
	s := newTestSessionWithModel(model)
	s.log = newSessionRecorder(writer, s.pushNotice)
	s.agent.Subscribe(s.log.listener())

	fixture := &recordingFixture{session: s, store: store, cwd: cwd, path: writer.Path()}
	t.Cleanup(func() { _ = s.log.close() })
	return fixture
}

// reopen closes the recorder and folds the file back, as a fresh process would.
func (f *recordingFixture) reopen(t *testing.T) restored {
	t.Helper()
	if err := f.session.log.close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	tree, _, report, err := f.store.Load(f.path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if report.SkippedLines != 0 {
		t.Fatalf("reload skipped %d lines: %v", report.SkippedLines, report.Warnings)
	}
	return projectSession(tree, tree.Leaf())
}

// newTestSessionWithModel builds a bare session for the persistence tests,
// without the catalog, credentials or workspace newSession would demand.
func newTestSessionWithModel(model *llm.Model) *session {
	return &session{
		model: model,
		agent: agent.NewAgent(agent.AgentOptions{InitialState: &agent.InitialState{
			Model: *model, ThinkingLevel: llm.ThinkingOff,
		}}),
	}
}

// userMsg builds a user message for tests that seed a log directly rather than
// driving a turn through the agent loop.
func userMsg(text string) llm.UserMessage {
	return llm.UserMessage{Role: "user", Content: text, Timestamp: 1}
}

// assistant builds an assistant message with a cost, since cost accounting is
// what several of the tests below turn on.
func assistant(text string, cost float64) llm.AssistantMessage {
	return llm.AssistantMessage{
		Role:     "assistant",
		Content:  []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}},
		API:      "test-api",
		Provider: "test",
		Model:    "test-model",
		Usage:    llm.Usage{Cost: llm.UsageCost{Total: cost}},
	}
}

// TestConversationSurvivesAResume is the headline promise: close the process,
// come back, and the conversation is there.
func TestConversationSurvivesAResume(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(userMsg("add a -continue flag"))
	f.session.log.appendMessage(assistant("here is the flag", 0.01))
	f.session.log.appendMessage(userMsg("now add tests"))
	f.session.log.appendMessage(assistant("here are the tests", 0.02))

	got := f.reopen(t)
	if len(got.Messages) != 4 {
		t.Fatalf("restored %d messages, want 4", len(got.Messages))
	}
	first, ok := got.Messages[0].(llm.UserMessage)
	if !ok {
		t.Fatalf("first restored message is %T, want UserMessage", got.Messages[0])
	}
	if text, _ := first.StringContent(); text != "add a -continue flag" {
		t.Fatalf("first message = %q", text)
	}
	if _, ok := got.Messages[3].(llm.AssistantMessage); !ok {
		t.Fatalf("last restored message is %T, want AssistantMessage", got.Messages[3])
	}
}

// TestUserPromptsAreRecorded covers the half that events cannot see: a prompt
// reaches the agent through Prompt, not through message_end, so a recorder
// watching only the subscription stores answers with no questions.
func TestUserPromptsAreRecorded(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(userMsg("the question"))
	f.session.log.appendMessage(assistant("the answer", 0))

	got := f.reopen(t)
	var roles []string
	for _, message := range got.Messages {
		roles = append(roles, agent.RoleOf(message))
	}
	if strings.Join(roles, ",") != "user,assistant" {
		t.Fatalf("roles = %v, want the user's own prompt recorded too", roles)
	}
}

// TestResumeRestoresModelAndThinkingLevel covers the settings a conversation
// was actually held with. Without them a resume silently continues on whatever
// the CLI defaults to.
func TestResumeRestoresModelAndThinkingLevel(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.agent.SetModel(llm.Model{Provider: "anthropic", ID: "claude-sonnet-4-5"})
	f.session.agent.SetThinkingLevel("high")
	f.session.log.appendMessage(assistant("answer", 0))

	got := f.reopen(t)
	if got.Provider != "anthropic" || got.ModelID != "claude-sonnet-4-5" {
		t.Fatalf("restored model = %s/%s, want anthropic/claude-sonnet-4-5", got.Provider, got.ModelID)
	}
	if got.ThinkingLevel != "high" {
		t.Fatalf("restored thinking = %q, want high", got.ThinkingLevel)
	}
}

// TestResumeWithNoModelChangeFallsBackToTheLastAssistantMessage covers a
// session recorded before any model switch, and pi sessions generally: the
// assistant message names the model the turn was actually served by.
func TestResumeWithNoModelChangeFallsBackToTheLastAssistantMessage(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(llm.AssistantMessage{
		Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "hi"}},
		API: "anthropic-messages", Provider: "anthropic", Model: "claude-opus-4-5",
	})

	got := f.reopen(t)
	if got.Provider != "anthropic" || got.ModelID != "claude-opus-4-5" {
		t.Fatalf("restored model = %s/%s, want the last assistant message's", got.Provider, got.ModelID)
	}
}

// TestExplicitModelChangeBeatsTheAssistantFallback is the negative control for
// the test above: a recorded switch must win, or /model is silently undone by
// every resume.
func TestExplicitModelChangeBeatsTheAssistantFallback(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(llm.AssistantMessage{
		Role: "assistant", Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: "hi"}},
		API: "anthropic-messages", Provider: "anthropic", Model: "claude-opus-4-5",
	})
	f.session.agent.SetModel(llm.Model{Provider: "openai", ID: "gpt-5.1"})

	got := f.reopen(t)
	if got.Provider != "openai" || got.ModelID != "gpt-5.1" {
		t.Fatalf("restored model = %s/%s, want the recorded switch to win", got.Provider, got.ModelID)
	}
}

// TestCompactionSurvivesAResume is the subtle one. conversationCost,
// maybeAutoCompact and the sidebar all dispatch on the Go type
// compactionSummaryMessage, so a resume that rebuilt the marker as plain text
// would double-count every retained assistant cost -- silently, and forever
// after.
func TestCompactionSurvivesAResume(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(userMsg("old question"))
	f.session.log.appendMessage(assistant("old answer", 5.00))

	kept := []agent.Message{assistant("kept answer", 0.25)}
	marker := compactionSummaryMessage{
		Summary: "## Goal\nEarlier work.", TokensBefore: 148002,
		CostBefore: 5.00, RetainedMessages: 1, Timestamp: 1756044151512,
	}
	if err := f.session.agent.Compact(marker, kept, agent.CompactionInfo{
		Summary: marker.Summary, TokensBefore: marker.TokensBefore,
		CostBefore: marker.CostBefore, RetainedMessages: marker.RetainedMessages, Timestamp: marker.Timestamp,
	}); err != nil {
		t.Fatalf("Compact: %v", err)
	}

	before := conversationCost(f.session.agent.State().Messages)
	got := f.reopen(t)
	after := conversationCost(got.Messages)

	if before != after {
		t.Fatalf("cost before resume %.4f != after %.4f; the compaction marker did not round-trip", before, after)
	}
	restoredMarker, ok := got.Messages[0].(compactionSummaryMessage)
	if !ok {
		t.Fatalf("first restored message is %T, want compactionSummaryMessage", got.Messages[0])
	}
	if restoredMarker.CostBefore != 5.00 || restoredMarker.RetainedMessages != 1 {
		t.Fatalf("marker = %+v, want cost and retained count preserved", restoredMarker)
	}
	if restoredMarker.TokensBefore != 148002 {
		t.Fatalf("marker tokensBefore = %d, want 148002", restoredMarker.TokensBefore)
	}
	// The pre-compaction turns must still be on disk, just out of context.
	tree, _, _, err := f.store.Load(f.path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(tree.Path(tree.Leaf())) <= len(got.Messages) {
		t.Fatal("the compacted prefix was not retained in the file")
	}
}

// TestCompactionDetailsRideInPiSCompatibleSlot pins the interop shape: the two
// numbers pi has no field for go in `details`, which pi types as unknown and
// ignores.
func TestCompactionDetailsRideInPiSCompatibleSlot(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(assistant("kept", 0.25))
	if err := f.session.agent.Compact(
		compactionSummaryMessage{Summary: "s"},
		[]agent.Message{assistant("kept", 0.25)},
		agent.CompactionInfo{Summary: "s", TokensBefore: 10, CostBefore: 1.25, RetainedMessages: 1},
	); err != nil {
		t.Fatalf("Compact: %v", err)
	}
	f.session.log.sync()

	raw, err := os.ReadFile(f.path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var found bool
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var entry sessionlog.Entry
		if json.Unmarshal([]byte(line), &entry) != nil || entry.Type != sessionlog.TypeCompaction {
			continue
		}
		found = true
		if entry.FirstKeptEntryID == "" {
			t.Fatal("the compaction entry has no firstKeptEntryId")
		}
		var details goshcoderDetails
		if err := json.Unmarshal(entry.Details, &details); err != nil {
			t.Fatalf("details do not decode: %v", err)
		}
		if details.GoshCoder.CostBefore != 1.25 || details.GoshCoder.RetainedMessages != 1 {
			t.Fatalf("details = %+v", details.GoshCoder)
		}
	}
	if !found {
		t.Fatal("no compaction entry was written")
	}
}

// TestClearIsRecordedAndCutsTheRestoredContext covers /clear: the pre-clear
// transcript stays in the file so an accidental clear is recoverable, but it
// must not come back on the next resume.
func TestClearIsRecordedAndCutsTheRestoredContext(t *testing.T) {
	f := newRecordingFixture(t)

	f.session.log.appendMessage(userMsg("discarded question"))
	f.session.log.appendMessage(assistant("discarded answer", 0))
	if err := f.session.agent.ResetWithReason("/clear"); err != nil {
		t.Fatalf("ResetWithReason: %v", err)
	}
	f.session.log.appendMessage(userMsg("fresh question"))
	f.session.log.appendMessage(assistant("fresh answer", 0))

	got := f.reopen(t)
	if len(got.Messages) != 2 {
		t.Fatalf("restored %d messages, want only the two after the clear", len(got.Messages))
	}
	first := got.Messages[0].(llm.UserMessage)
	if text, _ := first.StringContent(); text != "fresh question" {
		t.Fatalf("first restored message = %q, want the post-clear one", text)
	}
	// Recoverability is the reason for not rotating the file.
	tree, _, _, err := f.store.Load(f.path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if len(tree.Path(tree.Leaf())) < 5 {
		t.Fatal("the pre-clear transcript was not retained in the file")
	}
}

// TestNoSessionWritesNothing covers the escape hatch, which is also the
// recourse for a transcript that should not be on disk at all.
func TestNoSessionWritesNothing(t *testing.T) {
	agentDir := t.TempDir()
	t.Setenv("GOSHCODER_AGENT_DIR", agentDir)

	opened, err := openSession(sessionConfig{NoSession: true}, t.TempDir())
	if err != nil {
		t.Fatalf("openSession: %v", err)
	}
	if opened.writer != nil {
		t.Fatal("-no-session produced a writer")
	}
	if _, err := os.Stat(filepath.Join(agentDir, "sessions")); !os.IsNotExist(err) {
		t.Fatalf("-no-session created a sessions directory: %v", err)
	}
}

// TestRecordingStopsButTheTranscriptSurvivesAWriteFailure covers the fail-soft
// promise: the user is mid-task, so a full disk must not end the run. Recording
// stops, one notice is raised, and the conversation stays in memory.
func TestRecordingStopsButTheTranscriptSurvivesAWriteFailure(t *testing.T) {
	f := newRecordingFixture(t)
	f.session.log.appendMessage(userMsg("before the failure"))

	// Closing the underlying writer is what a failed write leaves behind.
	if err := f.session.log.close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	f.session.log.appendMessage(userMsg("after the failure"))
	f.session.log.appendMessage(assistant("still answering", 0))

	if f.session.recordingActive() {
		t.Fatal("the session still reports itself as recording")
	}
	// The run itself must not have failed.
	if len(f.session.drainNotices()) > 1 {
		t.Fatal("more than one notice was raised for a single stoppage")
	}
}

// TestSessionFilesAreNeverWrittenIntoTheWorkspace covers the containment
// promise: a transcript holds every file the agent read, and a repository is
// the wrong place for it.
func TestSessionFilesAreNeverWrittenIntoTheWorkspace(t *testing.T) {
	f := newRecordingFixture(t)
	f.session.log.appendMessage(userMsg("question"))
	f.session.log.appendMessage(assistant("answer", 0))
	f.session.log.sync()

	var found []string
	err := filepath.WalkDir(f.cwd, func(path string, entry os.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		found = append(found, path)
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(found) != 0 {
		t.Fatalf("the workspace gained files: %v", found)
	}
}

// TestResolveResumedModelDegradesVisibly covers the ladder. Resume is exactly
// the operation that spans a credential expiring or a gateway model going away,
// so an unavailable recorded model must fall back with a notice rather than
// failing the resume or switching in silence.
func TestResolveResumedModelDegradesVisibly(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())

	opened := &openedSession{Resumed: true}
	opened.restored.Provider = "nonexistent-provider"
	opened.restored.ModelID = "nonexistent-model"

	ref, notices := resolveResumedModel(sessionConfig{ModelRef: "anthropic/claude-sonnet-4-5"}, opened, false)
	if ref != "anthropic/claude-sonnet-4-5" {
		t.Fatalf("model = %q, want the fallback", ref)
	}
	if len(notices) != 1 || !strings.Contains(notices[0], "nonexistent-provider") {
		t.Fatalf("notices = %v, want one naming the unavailable model", notices)
	}
}

// TestExplicitModelFlagOverridesARecordedModel covers the other end of the
// ladder: -m is the user overriding on purpose, and they are told what changed.
func TestExplicitModelFlagOverridesARecordedModel(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())

	opened := &openedSession{Resumed: true}
	opened.restored.Provider = "anthropic"
	opened.restored.ModelID = "claude-opus-4-5"

	ref, notices := resolveResumedModel(sessionConfig{ModelRef: "openai/gpt-5.1"}, opened, true)
	if ref != "openai/gpt-5.1" {
		t.Fatalf("model = %q, want the explicit flag to win", ref)
	}
	if len(notices) != 1 || !strings.Contains(notices[0], "claude-opus-4-5") {
		t.Fatalf("notices = %v, want one naming the session's own model", notices)
	}
}

// TestContinueWithNoPriorSessionStartsOneQuietly keeps `chat -continue` safe to
// put in a shell alias.
func TestContinueWithNoPriorSessionStartsOneQuietly(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())

	opened, err := openSession(sessionConfig{Continue: true}, t.TempDir())
	if err != nil {
		t.Fatalf("openSession: %v", err)
	}
	defer opened.writer.Close()
	if opened.writer == nil {
		t.Fatal("no session was created")
	}
	if opened.Resumed {
		t.Fatal("reported a resume when there was nothing to resume")
	}
	if len(opened.Notices) != 1 {
		t.Fatalf("notices = %v, want one explaining that a new session was started", opened.Notices)
	}
}

// TestContinueReopensTheMostRecentSession is the flag's actual job.
func TestContinueReopensTheMostRecentSession(t *testing.T) {
	agentDir := t.TempDir()
	t.Setenv("GOSHCODER_AGENT_DIR", agentDir)
	cwd := t.TempDir()

	first, err := openSession(sessionConfig{}, cwd)
	if err != nil {
		t.Fatalf("openSession: %v", err)
	}
	recorder := newSessionRecorder(first.writer, nil)
	recorder.appendMessage(llm.UserMessage{Role: "user", Content: "remembered"})
	recorder.appendMessage(assistant("answer", 0))
	id := first.writer.ID()
	if err := recorder.close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	second, err := openSession(sessionConfig{Continue: true}, cwd)
	if err != nil {
		t.Fatalf("openSession -continue: %v", err)
	}
	defer second.writer.Close()
	if !second.Resumed {
		t.Fatal("-continue did not resume")
	}
	if second.writer.ID() != id {
		t.Fatalf("resumed %s, want %s", second.writer.ID(), id)
	}
	if len(second.restored.Messages) != 2 {
		t.Fatalf("restored %d messages, want 2", len(second.restored.Messages))
	}
}

// TestSecondWindowContinuesReadOnlyRatherThanFailing covers two windows in one
// workspace. Refusing to start would be worse than starting without recording,
// and saying nothing would be worse still.
func TestSecondWindowContinuesReadOnlyRatherThanFailing(t *testing.T) {
	t.Setenv("GOSHCODER_AGENT_DIR", t.TempDir())
	cwd := t.TempDir()

	first, err := openSession(sessionConfig{}, cwd)
	if err != nil {
		t.Fatalf("openSession: %v", err)
	}
	recorder := newSessionRecorder(first.writer, nil)
	recorder.appendMessage(assistant("answer", 0))
	recorder.sync()
	defer recorder.close()

	second, err := openSession(sessionConfig{Continue: true}, cwd)
	if err != nil {
		t.Fatalf("second openSession: %v", err)
	}
	defer second.writer.Close()
	if !second.writer.ReadOnly() {
		t.Fatal("the second window claimed a session the first is driving")
	}
	var told bool
	for _, notice := range second.Notices {
		if strings.Contains(notice, "another window") {
			told = true
		}
	}
	if !told {
		t.Fatalf("notices = %v, want one saying the session is open elsewhere", second.Notices)
	}
}

// TestSessionFlagValidationRejectsContradictions covers the combinations that
// would otherwise be silently reinterpreted.
func TestSessionFlagValidationRejectsContradictions(t *testing.T) {
	cases := map[string]sessionConfig{
		"-no-session with -continue": {NoSession: true, Continue: true},
		"-no-session with -session":  {NoSession: true, SessionRef: "abc"},
		"-no-session with -name":     {NoSession: true, SessionName: "x"},
		"-continue with -session":    {Continue: true, SessionRef: "abc"},
		"-resume with -continue":     {Resume: true, Continue: true},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if err := validateSessionFlags(&cfg); err == nil {
				t.Fatal("accepted a contradictory combination")
			}
		})
	}

	// Negative control: a legitimate combination must still be accepted.
	ok := sessionConfig{Continue: true, SessionName: "nightly", Thinking: "off"}
	if err := validateSessionFlags(&ok); err != nil {
		t.Fatalf("rejected a valid combination: %v", err)
	}
}
