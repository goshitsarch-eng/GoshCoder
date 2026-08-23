package sessionlog

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"goshcoder/internal/llm"
)

// ---------------------------------------------------------------------------
// helpers

func userEntry(text string) Entry {
	raw, _ := json.Marshal(llm.UserMessage{Role: "user", Content: text, Timestamp: 1})
	return Entry{Type: TypeMessage, Message: raw}
}

func assistantEntry(text string) Entry {
	raw, _ := json.Marshal(llm.AssistantMessage{
		Role:      "assistant",
		Content:   []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}},
		API:       "anthropic-messages",
		Provider:  "anthropic",
		Model:     "claude-sonnet-4-5",
		Timestamp: 2,
	})
	return Entry{Type: TypeMessage, Message: raw}
}

func newStore(t *testing.T) *Store {
	t.Helper()
	return NewStore(filepath.Join(t.TempDir(), "sessions"))
}

// texts renders the user/assistant text of a projected path, for comparisons
// that do not care about ids or timestamps.
func texts(entries []Entry) []string {
	var out []string
	for _, e := range entries {
		if e.Type != TypeMessage {
			out = append(out, "["+e.Type+"]")
			continue
		}
		out = append(out, messageText(e.Message))
	}
	return out
}

// ---------------------------------------------------------------------------
// round trip

// TestTranscriptSurvivesARoundTrip is the load-bearing promise of the whole
// package: what the agent held in memory is what a resume hands back.
//
// The comparison goes through llm.UnmarshalMessages rather than reflect.DeepEqual
// on the structs, because AssistantMessage.MarshalJSON substitutes an empty
// slice for nil content and the decoder always returns a non-nil slice -- so a
// nil-content message never deep-equals itself after a round trip, and a test
// written that way fails for a reason that has nothing to do with the format.
func TestTranscriptSurvivesARoundTrip(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()

	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	signature := "ErUBCkYIBBgCIkDLONG"
	assistant := llm.AssistantMessage{
		Role: "assistant",
		Content: []llm.ContentBlock{
			llm.ThinkingContent{Type: "thinking", Thinking: "weighing it up", ThinkingSignature: signature},
			llm.ToolCall{Type: "toolCall", ID: "toolu_01A", Name: "read", Arguments: map[string]any{"path": "main.go"}},
		},
		API: "anthropic-messages", Provider: "anthropic", Model: "claude-sonnet-4-5",
		StopReason: "toolUse", Timestamp: 2,
	}
	assistantRaw, err := json.Marshal(assistant)
	if err != nil {
		t.Fatalf("marshal assistant: %v", err)
	}

	if _, err := writer.Append(userEntry("add a -continue flag")); err != nil {
		t.Fatalf("append user: %v", err)
	}
	if _, err := writer.Append(Entry{Type: TypeMessage, Message: assistantRaw}); err != nil {
		t.Fatalf("append assistant: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tree, header, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if report.SkippedLines != 0 {
		t.Fatalf("skipped %d lines on a clean file: %v", report.SkippedLines, report.Warnings)
	}
	if header.Version != FormatVersion {
		t.Fatalf("version = %d, want %d", header.Version, FormatVersion)
	}

	entries := tree.ContextPath(nil)
	if len(entries) != 2 {
		t.Fatalf("context path has %d entries, want 2", len(entries))
	}
	decoded, err := llm.UnmarshalMessages(json.RawMessage("[" + string(entries[1].Message) + "]"))
	if err != nil {
		t.Fatalf("decode assistant: %v", err)
	}
	restored, ok := decoded[0].(llm.AssistantMessage)
	if !ok {
		t.Fatalf("decoded message is %T, want AssistantMessage", decoded[0])
	}
	if restored.Model != assistant.Model || restored.Provider != assistant.Provider {
		t.Fatalf("model/provider = %s/%s, want %s/%s",
			restored.Provider, restored.Model, assistant.Provider, assistant.Model)
	}
	thinking, ok := restored.Content[0].(llm.ThinkingContent)
	if !ok {
		t.Fatalf("first block is %T, want ThinkingContent", restored.Content[0])
	}
	// The signature is what lets a provider accept replayed thinking. Losing it
	// turns a resumed session into an API error on the first turn.
	if thinking.ThinkingSignature != signature {
		t.Fatalf("thinkingSignature = %q, want it preserved", thinking.ThinkingSignature)
	}
	call, ok := restored.Content[1].(llm.ToolCall)
	if !ok {
		t.Fatalf("second block is %T, want ToolCallContent", restored.Content[1])
	}
	if call.ID != "toolu_01A" || call.Name != "read" {
		t.Fatalf("tool call = %s/%s, want toolu_01A/read", call.ID, call.Name)
	}
}

// TestEntriesFormATreeThroughParentID covers the property that makes branching
// possible later: each append attaches to the previous leaf, and the root's
// parentId is null rather than absent.
func TestEntriesFormATreeThroughParentID(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()

	first, _ := writer.Append(userEntry("one"))
	second, _ := writer.Append(assistantEntry("two"))

	tree := writer.Snapshot()
	root, _ := tree.Entry(first)
	if root.ParentID != nil {
		t.Fatalf("root parentId = %v, want nil", *root.ParentID)
	}
	child, _ := tree.Entry(second)
	if child.ParentID == nil || *child.ParentID != first {
		t.Fatalf("child parentId = %v, want %s", child.ParentID, first)
	}

	// pi requires the key to be present and null at the root; omitting it
	// makes the file unreadable to pi even though Go round-trips it fine.
	line, _ := tree.Raw(first)
	if !strings.Contains(string(line), `"parentId":null`) {
		t.Fatalf("root line lacks an explicit null parentId: %s", line)
	}
}

// ---------------------------------------------------------------------------
// durability

// TestEveryTruncationOffsetLoadsAValidPrefix covers the crash-mid-append case
// at every possible point. A torn tail must cost the partial entry and nothing
// else -- never a partially applied entry, and never a load failure.
func TestEveryTruncationOffsetLoadsAValidPrefix(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("first question")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("first answer")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(userEntry("second question")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	full, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	for cut := 1; cut <= len(full); cut++ {
		torn := filepath.Join(t.TempDir(), "torn.jsonl")
		if err := os.WriteFile(torn, full[:cut], 0o600); err != nil {
			t.Fatalf("write torn file: %v", err)
		}
		tree, _, _, err := store.Load(torn)
		if err != nil {
			// Only a truncation that lost the header may fail to load.
			headerEnd := strings.Index(string(full), "\n") + 1
			if cut >= headerEnd {
				t.Fatalf("cut at %d/%d failed to load: %v", cut, len(full), err)
			}
			continue
		}
		// Whatever survived must be internally consistent: every entry's
		// parent is either nil or an entry that is also present.
		for _, entry := range tree.All() {
			if entry.ParentID != nil && !tree.Has(*entry.ParentID) {
				t.Fatalf("cut at %d produced entry %s with dangling parent %s", cut, entry.ID, *entry.ParentID)
			}
		}
	}
}

// TestAttachRepairsATornTailSoTheNextAppendIsClean is the half the test above
// cannot cover: loading a torn file is not enough, because the next append
// would otherwise concatenate onto the partial line and corrupt both entries.
func TestAttachRepairsATornTailSoTheNextAppendIsClean(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("question")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("answer")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Simulate a process killed mid-write.
	content, _ := os.ReadFile(path)
	if err := os.WriteFile(path, append(content, []byte(`{"type":"mess`)...), 0o600); err != nil {
		t.Fatalf("tear file: %v", err)
	}

	reattached, report, err := store.Attach(path)
	if err != nil {
		t.Fatalf("Attach: %v", err)
	}
	if !report.RepairedTail {
		t.Fatal("Attach did not report repairing the torn tail")
	}
	if _, err := reattached.Append(userEntry("after the crash")); err != nil {
		t.Fatalf("append after repair: %v", err)
	}
	if err := reattached.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tree, _, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if report.SkippedLines != 0 {
		t.Fatalf("repaired file still has %d bad lines: %v", report.SkippedLines, report.Warnings)
	}
	if got := texts(tree.ContextPath(nil)); len(got) != 3 || got[2] != "after the crash" {
		t.Fatalf("context = %q, want the repaired tail replaced by the new entry", got)
	}
}

// TestMalformedMiddleLineIsSkippedAndReported covers pi's leniency: one bad
// line must never cost a whole session.
func TestMalformedMiddleLineIsSkippedAndReported(t *testing.T) {
	store := newStore(t)
	path := filepath.Join(t.TempDir(), "s.jsonl")
	content := `{"type":"session","version":3,"id":"abc","timestamp":"2026-08-23T14:02:11.431Z","cwd":"/w"}
{"type":"message","id":"a1","parentId":null,"timestamp":"2026-08-23T14:02:12.000Z","message":{"role":"user","content":"kept","timestamp":1}}
{ this line is not json
{"type":"message","id":"a2","parentId":"a1","timestamp":"2026-08-23T14:02:13.000Z","message":{"role":"assistant","content":[{"type":"text","text":"also kept"}],"api":"anthropic-messages","provider":"anthropic","model":"m","usage":{},"stopReason":"stop","timestamp":2}}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	tree, _, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if report.SkippedLines != 1 {
		t.Fatalf("SkippedLines = %d, want 1", report.SkippedLines)
	}
	if len(report.Warnings) != 1 || !strings.Contains(report.Warnings[0], "line 3") {
		t.Fatalf("warnings = %v, want one naming line 3", report.Warnings)
	}
	if got := texts(tree.ContextPath(nil)); len(got) != 2 {
		t.Fatalf("context = %q, want both good entries", got)
	}
}

// TestFileWhoseFirstLineIsNotASessionHeaderIsRefused is the negative control
// for the leniency above: tolerating bad entry lines must not extend to
// treating an arbitrary file as a session.
func TestFileWhoseFirstLineIsNotASessionHeaderIsRefused(t *testing.T) {
	store := newStore(t)
	path := filepath.Join(t.TempDir(), "notasession.jsonl")
	if err := os.WriteFile(path, []byte("{\"hello\":\"world\"}\n"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, _, _, err := store.Load(path); err == nil {
		t.Fatal("Load accepted a file with no session header")
	}
}

// TestVersionNewerThanThisBuildIsRefusedWithAnActionableError covers the
// forward direction: entries a newer writer considered essential must not be
// silently dropped.
func TestVersionNewerThanThisBuildIsRefusedWithAnActionableError(t *testing.T) {
	store := newStore(t)
	path := filepath.Join(t.TempDir(), "future.jsonl")
	if err := os.WriteFile(path, []byte(`{"type":"session","version":99,"id":"abc","timestamp":"2026-08-23T14:02:11.431Z","cwd":"/w"}`+"\n"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	_, _, _, err := store.Load(path)
	if !errors.Is(err, ErrVersionTooNew) {
		t.Fatalf("error = %v, want ErrVersionTooNew", err)
	}
}

// TestUnknownEntryTypeJoinsTheTreeAndProjectsToNoMessages is the executable
// statement that every later format addition is additive. A build that has
// never heard of an entry type must still index it, keep entries chained
// through it reachable, and send nothing for it.
func TestUnknownEntryTypeJoinsTheTreeAndProjectsToNoMessages(t *testing.T) {
	store := newStore(t)
	path := filepath.Join(t.TempDir(), "future-entry.jsonl")
	content := `{"type":"session","version":3,"id":"abc","timestamp":"2026-08-23T14:02:11.431Z","cwd":"/w"}
{"type":"message","id":"a1","parentId":null,"timestamp":"2026-08-23T14:02:12.000Z","message":{"role":"user","content":"before","timestamp":1}}
{"type":"telepathy_change","id":"a2","parentId":"a1","timestamp":"2026-08-23T14:02:13.000Z","vibe":"ambitious"}
{"type":"message","id":"a3","parentId":"a2","timestamp":"2026-08-23T14:02:14.000Z","message":{"role":"user","content":"after","timestamp":3}}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	tree, _, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if report.SkippedLines != 0 {
		t.Fatalf("an unknown entry type was skipped: %v", report.Warnings)
	}
	if !tree.Has("a2") {
		t.Fatal("the unknown entry did not join the tree")
	}
	// The child chained through it must still be reachable, which is the whole
	// point: otherwise an old build silently truncates a newer session.
	got := texts(tree.ContextPath(nil))
	want := []string{"before", "[telepathy_change]", "after"}
	if len(got) != len(want) {
		t.Fatalf("context = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("context = %q, want %q", got, want)
		}
	}
}

// ---------------------------------------------------------------------------
// bounds

// TestOversizedEntryIsRefusedAtAppendWithoutWritingAPartialLine covers why the
// cap is enforced on the way in and not only on the way out: a load-only limit
// lets a session grow past what this build can reopen, turning a working
// session into a permanently unopenable file.
func TestOversizedEntryIsRefusedAtAppendWithoutWritingAPartialLine(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("small and fine")); err != nil {
		t.Fatalf("append: %v", err)
	}
	before, err := os.ReadFile(writer.Path())
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	if _, err := writer.Append(userEntry(strings.Repeat("x", MaxEntryBytes+1))); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("append error = %v, want ErrTooLarge", err)
	}
	after, err := os.ReadFile(writer.Path())
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(after) != len(before) {
		t.Fatalf("a refused entry wrote %d bytes", len(after)-len(before))
	}
	// The writer must still be usable: one oversized tool result should not
	// end recording for the rest of the session.
	if _, err := writer.Append(assistantEntry("still working")); err != nil {
		t.Fatalf("append after a refused entry: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// ---------------------------------------------------------------------------
// claims

// TestSecondWriterIsRefusedAndWritesNothing covers `chat -continue` run twice
// in one workspace. Interleaved appends produce a tree whose parent links cross
// between two conversations, which is unrecoverable.
func TestSecondWriterIsRefusedAndWritesNothing(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()
	if _, err := writer.Append(assistantEntry("held")); err != nil {
		t.Fatalf("append: %v", err)
	}
	before, _ := os.ReadFile(writer.Path())

	_, _, err = store.Attach(writer.Path())
	if !errors.Is(err, ErrBusy) {
		t.Fatalf("second Attach error = %v, want ErrBusy", err)
	}
	if !strings.Contains(err.Error(), fmt.Sprintf("pid %d", os.Getpid())) {
		t.Fatalf("error %q does not name the owning process", err)
	}
	after, _ := os.ReadFile(writer.Path())
	if len(after) != len(before) {
		t.Fatal("a refused writer changed the file")
	}
}

// TestReadOnlyOpenDoesNotTakeTheClaim covers -read-only: inspecting a session
// another window is driving must not lock that window out.
func TestReadOnlyOpenDoesNotTakeTheClaim(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(assistantEntry("content")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	viewer, _, err := store.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if !viewer.ReadOnly() {
		t.Fatal("Open returned a writable session")
	}
	if _, err := viewer.Append(userEntry("nope")); err == nil {
		t.Fatal("a read-only session accepted an append")
	}
	// The claim must still be free.
	attached, _, err := store.Attach(path)
	if err != nil {
		t.Fatalf("Attach while a reader is open: %v", err)
	}
	attached.Close()
}

// TestListingDoesNotLeaveLockFilesBehind covers the claim probe in describe():
// listing sessions must not leave a lock file next to every one of them, which
// would make each of them look busy to the next process that looked.
func TestListingDoesNotLeaveLockFilesBehind(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(assistantEntry("content")); err != nil {
		t.Fatalf("append: %v", err)
	}
	dir := filepath.Dir(writer.Path())
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := store.List(cwd, ListOptions{}); err != nil {
		t.Fatalf("List: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".lock") {
			t.Fatalf("listing left a lock file behind: %s", entry.Name())
		}
	}
}

// ---------------------------------------------------------------------------
// lifecycle

// TestSessionWithNoAssistantMessageIsDeletedOnClose keeps the sessions
// directory from filling with a stub for every accidental launch.
func TestSessionWithNoAssistantMessageIsDeletedOnClose(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("started but never answered")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("an unanswered session was kept: %v", err)
	}
}

// TestSessionWithAnAssistantMessageIsKeptOnClose is the negative control: the
// cleanup above must not be able to eat a real conversation.
func TestSessionWithAnAssistantMessageIsKeptOnClose(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("question")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("answer")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("a real session was deleted: %v", err)
	}
}

// TestSessionFileIsCreated0600 covers the permission promise: a transcript
// holds every file the agent read.
func TestSessionFileIsCreated0600(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows does not honor Unix permission bits")
	}
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()

	info, err := os.Stat(writer.Path())
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if got := info.Mode().Perm(); got != 0o600 {
		t.Fatalf("session mode = %o, want 600", got)
	}
	dir, err := os.Stat(filepath.Dir(writer.Path()))
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if got := dir.Mode().Perm(); got != 0o700 {
		t.Fatalf("shard mode = %o, want 700", got)
	}
}

// ---------------------------------------------------------------------------
// projection

// TestContextPathStopsAtTheLatestCompaction covers the projection compaction
// depends on: summarized entries stay in the file and browsable, but are never
// sent to the model again.
func TestContextPathStopsAtTheLatestCompaction(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()

	if _, err := writer.Append(userEntry("old one")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("old two")); err != nil {
		t.Fatalf("append: %v", err)
	}
	keep, err := writer.Append(userEntry("kept"))
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(Entry{
		Type: TypeCompaction, Summary: "a summary", FirstKeptEntryID: keep, TokensBefore: 1000,
	}); err != nil {
		t.Fatalf("append compaction: %v", err)
	}
	if _, err := writer.Append(assistantEntry("after")); err != nil {
		t.Fatalf("append: %v", err)
	}

	tree := writer.Snapshot()
	got := texts(tree.ContextPath(nil))
	want := []string{"[compaction]", "kept", "after"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("context = %q, want %q", got, want)
	}
	// The whole history is still on disk.
	if full := texts(tree.Path(nil)); len(full) != 5 {
		t.Fatalf("full path = %q, want all five entries retained", full)
	}
}

// TestContextPathStopsAtTheLatestResetEvenAfterACompaction covers /clear: it
// cuts harder than a compaction, so the summary a compaction left behind must
// not survive it.
func TestContextPathStopsAtTheLatestResetEvenAfterACompaction(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()

	keep, _ := writer.Append(userEntry("kept by compaction"))
	if _, err := writer.Append(Entry{Type: TypeCompaction, Summary: "s", FirstKeptEntryID: keep}); err != nil {
		t.Fatalf("append compaction: %v", err)
	}
	if _, err := writer.Append(Entry{Type: TypeTranscriptReset, Reason: "/clear"}); err != nil {
		t.Fatalf("append reset: %v", err)
	}
	if _, err := writer.Append(userEntry("fresh start")); err != nil {
		t.Fatalf("append: %v", err)
	}

	got := texts(writer.Snapshot().ContextPath(nil))
	if len(got) != 1 || got[0] != "fresh start" {
		t.Fatalf("context = %q, want only what followed the reset", got)
	}
}

// TestPathTerminatesOnACycle covers a hand-edited or corrupt file: returning
// the prefix walked so far is strictly better than hanging the process.
func TestPathTerminatesOnACycle(t *testing.T) {
	tree := NewTree()
	a, b := "aaaa1111", "bbbb2222"
	if err := tree.Add(Entry{Type: TypeMessage, ID: a, ParentID: &b}, []byte("{}")); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := tree.Add(Entry{Type: TypeMessage, ID: b, ParentID: &a}, []byte("{}")); err != nil {
		t.Fatalf("add: %v", err)
	}
	if got := tree.Path(nil); len(got) != 2 {
		t.Fatalf("path over a cycle returned %d entries, want the 2 it could walk", len(got))
	}
}

// TestDuplicateEntryIDIsRejected covers tree ambiguity: two entries with one id
// would make parent resolution depend on read order.
func TestDuplicateEntryIDIsRejected(t *testing.T) {
	tree := NewTree()
	if err := tree.Add(Entry{Type: TypeMessage, ID: "dup"}, []byte("{}")); err != nil {
		t.Fatalf("first add: %v", err)
	}
	if err := tree.Add(Entry{Type: TypeMessage, ID: "dup"}, []byte("{}")); !errors.Is(err, ErrDuplicateEntryID) {
		t.Fatalf("second add error = %v, want ErrDuplicateEntryID", err)
	}
}

// ---------------------------------------------------------------------------
// listing and resolution

// TestCwdCollisionIsResolvedByTheHeader covers the lossy shard encoding: /a/b
// and /a-b produce the same directory name, so the header is the authority on
// which workspace a session belongs to.
func TestCwdCollisionIsResolvedByTheHeader(t *testing.T) {
	root := t.TempDir()
	store := NewStore(filepath.Join(root, "sessions"))

	base := t.TempDir()
	first := filepath.Join(base, "a", "b")
	second := filepath.Join(base, "a-b")
	for _, dir := range []string{first, second} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}
	if DirName(first) != DirName(second) {
		t.Skipf("this platform does not produce the collision under test")
	}

	for _, dir := range []string{first, second} {
		writer, err := store.Create(dir, "")
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		if _, err := writer.Append(assistantEntry("in " + dir)); err != nil {
			t.Fatalf("append: %v", err)
		}
		if err := writer.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
	}

	listed, err := store.List(first, ListOptions{})
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(listed) != 1 {
		t.Fatalf("listed %d sessions for %s, want only its own", len(listed), first)
	}
	if listed[0].Cwd != first {
		t.Fatalf("listed session cwd = %s, want %s", listed[0].Cwd, first)
	}
}

// TestResolveAcceptsAPathAnExactIDAndAPrefix covers the three shapes a user
// can type, and the ambiguity that must not be guessed at.
func TestResolveAcceptsAPathAnExactIDAndAPrefix(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()

	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(assistantEntry("content")); err != nil {
		t.Fatalf("append: %v", err)
	}
	id, path := writer.ID(), writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	for name, ref := range map[string]string{"path": path, "exact id": id, "prefix": id[:8]} {
		got, err := store.Resolve(cwd, ref)
		if err != nil {
			t.Fatalf("Resolve by %s: %v", name, err)
		}
		if got.ID != id {
			t.Fatalf("Resolve by %s = %s, want %s", name, got.ID, id)
		}
	}

	if _, err := store.Resolve(cwd, "definitely-not-a-session"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("error = %v, want ErrNotFound", err)
	}
}

// TestMostRecentPrefersTheNewestSession is what -continue relies on.
func TestMostRecentPrefersTheNewestSession(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()

	var newest string
	for i := 0; i < 3; i++ {
		writer, err := store.Create(cwd, "")
		if err != nil {
			t.Fatalf("Create: %v", err)
		}
		if _, err := writer.Append(assistantEntry(fmt.Sprintf("session %d", i))); err != nil {
			t.Fatalf("append: %v", err)
		}
		newest = writer.ID()
		if err := writer.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		// Modification times can share a coarse clock tick; the id tiebreak
		// keeps ordering deterministic, and uuidv7 ids are time-ordered.
		now := ParseTimestamp(Now()).Add(time.Duration(i) * time.Second)
		_ = os.Chtimes(store.Dir(cwd), now, now)
	}

	info, ok, err := store.MostRecent(cwd)
	if err != nil || !ok {
		t.Fatalf("MostRecent: %v ok=%v", err, ok)
	}
	if info.ID != newest {
		t.Fatalf("MostRecent = %s, want the newest %s", info.ID, newest)
	}
}

// TestListingCountsMessagesAfterTheLatestResetSeparately covers the reporting
// consequence of keeping /clear in the same file: a listing that ignored the
// reset would call a session holding hundreds of messages a two-message one.
func TestListingCountsMessagesAfterTheLatestResetSeparately(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	for i := 0; i < 4; i++ {
		if _, err := writer.Append(userEntry(fmt.Sprintf("old %d", i))); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	if _, err := writer.Append(Entry{Type: TypeTranscriptReset, Reason: "/clear"}); err != nil {
		t.Fatalf("append reset: %v", err)
	}
	if _, err := writer.Append(userEntry("new question")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("new answer")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	listed, err := store.List(cwd, ListOptions{})
	if err != nil || len(listed) != 1 {
		t.Fatalf("List: %v (%d sessions)", err, len(listed))
	}
	if listed[0].Messages != 2 {
		t.Fatalf("Messages = %d, want 2 since the reset", listed[0].Messages)
	}
	if listed[0].Cleared != 4 {
		t.Fatalf("Cleared = %d, want the 4 from before the reset", listed[0].Cleared)
	}
	if listed[0].FirstMessage != "new question" {
		t.Fatalf("FirstMessage = %q, want the first since the reset", listed[0].FirstMessage)
	}
}

// TestRemoveRefusesAClaimedSession covers deleting a session another window is
// driving: the running process would keep appending to an unlinked inode and
// silently lose everything after.
func TestRemoveRefusesAClaimedSession(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()
	if _, err := writer.Append(assistantEntry("live")); err != nil {
		t.Fatalf("append: %v", err)
	}

	info, err := store.Resolve(cwd, writer.ID())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !info.Locked {
		t.Fatal("a live session was not reported as claimed")
	}
	if err := store.Remove(info); !errors.Is(err, ErrBusy) {
		t.Fatalf("Remove error = %v, want ErrBusy", err)
	}
	if _, statErr := os.Stat(info.Path); statErr != nil {
		t.Fatalf("a refused Remove deleted the file anyway: %v", statErr)
	}
}

// ---------------------------------------------------------------------------
// ids

// TestSessionIDsAreTimeOrdered is what makes "the most recent session" a
// filename comparison rather than a read of every header.
func TestSessionIDsAreTimeOrdered(t *testing.T) {
	var previous string
	for i := 0; i < 200; i++ {
		id, err := NewSessionID()
		if err != nil {
			t.Fatalf("NewSessionID: %v", err)
		}
		if err := ValidateSessionID(id); err != nil {
			t.Fatalf("generated id is invalid: %v", err)
		}
		if id <= previous {
			t.Fatalf("id %q did not sort after %q", id, previous)
		}
		previous = id
	}
}

// TestValidateSessionIDRejectsPathComponents covers -session with a hostile
// value: the id becomes part of a filename.
func TestValidateSessionIDRejectsPathComponents(t *testing.T) {
	for _, bad := range []string{"", "../escape", "a/b", `a\b`, ".hidden", "trailing-", "with space", "nul\x00byte"} {
		if err := ValidateSessionID(bad); err == nil {
			t.Fatalf("ValidateSessionID(%q) accepted an unsafe id", bad)
		}
	}
	for _, good := range []string{"a", "A1", "01998b3c-4d21-7f0a-9b55-3f0a12c4e6b8", "my.session_1"} {
		if err := ValidateSessionID(good); err != nil {
			t.Fatalf("ValidateSessionID(%q) rejected a valid id: %v", good, err)
		}
	}
}

// TestDirNameMatchesPiEncoding pins the interop-critical encoding.
func TestDirNameMatchesPiEncoding(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("path shapes differ on Windows; the encoding is asserted on POSIX")
	}
	if got, want := DirName("/home/user/GoshCoder"), "--home-user-GoshCoder--"; got != want {
		t.Fatalf("DirName = %q, want %q", got, want)
	}
}

// TestFilenameStampIsWindowsSafe covers why the timestamp is rewritten at all.
func TestFilenameStampIsWindowsSafe(t *testing.T) {
	got := FilenameStamp("2026-08-23T14:02:11.431Z")
	if got != "2026-08-23T14-02-11-431Z" {
		t.Fatalf("FilenameStamp = %q", got)
	}
	if strings.ContainsAny(got, `:.`) {
		t.Fatalf("stamp %q still contains a character Windows refuses in a filename", got)
	}
}
