package sessionlog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The repair path in Attach decides which trailing bytes of a session file are
// a torn append and which are real conversation. Getting that wrong destroys
// work, so each case below is the exact shape that broke it.

// TestAttachKeepsAValidUnterminatedLastEntry covers a process killed after its
// write reached the file but before the newline did. The entry is complete JSON
// and was folded into the tree, so deleting it from the file while keeping it in
// memory leaves the next append pointing at a parent that no longer exists.
func TestAttachKeepsAValidUnterminatedLastEntry(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("first")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("second")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Strip the final newline: the entry itself is intact.
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if err := os.WriteFile(path, content[:len(content)-1], 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	attached, _, err := store.Attach(path)
	if err != nil {
		t.Fatalf("Attach: %v", err)
	}
	if got := attached.Snapshot().Len(); got != 2 {
		t.Fatalf("tree holds %d entries, want both", got)
	}
	if _, err := attached.Append(userEntry("third")); err != nil {
		t.Fatalf("append after repair: %v", err)
	}
	if err := attached.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tree, _, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if report.SkippedLines != 0 {
		t.Fatalf("reload skipped %d lines: %v", report.SkippedLines, report.Warnings)
	}
	if tree.Len() != 3 {
		t.Fatalf("file holds %d entries, want all three", tree.Len())
	}
	// Every entry must still reach the root, which is what a deleted parent
	// would break.
	for _, entry := range tree.All() {
		if entry.ParentID != nil && !tree.Has(*entry.ParentID) {
			t.Fatalf("entry %s lost its parent %s", entry.ID, *entry.ParentID)
		}
	}
	if got := len(tree.Path(tree.Leaf())); got != 3 {
		t.Fatalf("the path holds %d entries, want 3", got)
	}
}

// TestAttachDoesNotTruncatePastAnOversizedLine covers a session containing one
// line above MaxEntryBytes. That line is skipped, but the bytes it occupies are
// still in the file, and a repair that forgets them cuts the file in the middle
// of a later entry -- destroying every entry after the oversized one.
func TestAttachDoesNotTruncatePastAnOversizedLine(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("before the big one")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("a reply, so the session is kept")); err != nil {
		t.Fatalf("append: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Splice in an oversized line, then more real entries after it.
	oversized := `{"type":"message","id":"big00000","parentId":null,"timestamp":"2026-08-23T00:00:00.000Z","message":{"role":"user","content":"` +
		strings.Repeat("x", MaxEntryBytes) + `","timestamp":1}}` + "\n"
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := file.WriteString(oversized); err != nil {
		t.Fatalf("write oversized: %v", err)
	}
	file.Close()

	reattached, report, err := store.Attach(path)
	if err != nil {
		t.Fatalf("Attach: %v", err)
	}
	if report.SkippedLines != 1 {
		t.Fatalf("SkippedLines = %d, want the oversized line reported", report.SkippedLines)
	}
	if _, err := reattached.Append(assistantEntry("after the big one")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := reattached.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The file must still parse, and must still hold the entry that preceded
	// the oversized line.
	tree, _, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if report.SkippedLines != 1 {
		t.Fatalf("reload skipped %d lines, want only the oversized one: %v", report.SkippedLines, report.Warnings)
	}
	var sawBefore, sawAfter bool
	for _, entry := range tree.All() {
		text := messageText(entry.Message)
		if strings.Contains(text, "before the big one") {
			sawBefore = true
		}
		if strings.Contains(text, "after the big one") {
			sawAfter = true
		}
	}
	if !sawBefore {
		t.Fatal("the entry before the oversized line was destroyed")
	}
	if !sawAfter {
		t.Fatal("the entry appended after the repair was lost")
	}
}

// TestSkippedLineDoesNotOrphanTheEntriesBeforeIt covers what a corrupt middle
// line costs. Walking the parent chain stops dead at a missing parent, so
// without re-parenting, one bad line silently drops the whole conversation
// before it while the warning claims only that line was skipped.
func TestSkippedLineDoesNotOrphanTheEntriesBeforeIt(t *testing.T) {
	store := newStore(t)
	path := filepath.Join(t.TempDir(), "s.jsonl")
	content := `{"type":"session","version":3,"id":"abc","timestamp":"2026-08-23T14:02:11.431Z","cwd":"/w"}
{"type":"message","id":"a1","parentId":null,"timestamp":"2026-08-23T14:02:12.000Z","message":{"role":"user","content":"the first question","timestamp":1}}
{"type":"message","id":"a2","parentId":"a1","timestamp":"2026-08-23T14:02:13.000Z","message":{"role":"assistant","content":[{"type":"text","text":"the first answer"}],"api":"anthropic-messages","provider":"anthropic","model":"m","usage":{},"stopReason":"stop","timestamp":2}}
{ corrupt line that cannot be parsed
{"type":"message","id":"a4","parentId":"a3","timestamp":"2026-08-23T14:02:15.000Z","message":{"role":"user","content":"the later question","timestamp":3}}
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

	// Everything that survived must be reachable from the leaf. Losing the two
	// earlier turns would be silent context loss on the next request.
	path4 := texts(tree.ContextPath(nil))
	joined := strings.Join(path4, "|")
	for _, want := range []string{"the first question", "the first answer", "the later question"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("context = %q, want it to still contain %q", joined, want)
		}
	}
}

// TestForkOfALabelledSessionProducesAUsableClone covers a fork whose last
// copied entry is a label. A label is not a conversation entry, so leaving the
// write head on one -- or copying a label whose target was not copied -- yields
// a clone that projects to nothing.
func TestForkOfALabelledSessionProducesAUsableClone(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	first, err := writer.Append(userEntry("a question"))
	if err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("an answer")); err != nil {
		t.Fatalf("append: %v", err)
	}
	label := "worth remembering"
	if _, err := writer.Append(Entry{Type: TypeLabel, TargetID: first, Label: &label}); err != nil {
		t.Fatalf("append label: %v", err)
	}
	if err := writer.Sync(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	id := writer.ID()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	info, err := store.Resolve(cwd, id)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	forked, err := store.Fork(info, nil, cwd)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	forkedPath := forked.Path()
	if err := forked.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The clone must still exist -- Close must not mistake a copied
	// conversation for an empty session.
	if _, err := os.Stat(forkedPath); err != nil {
		t.Fatalf("the fork was deleted on close: %v", err)
	}
	tree, _, _, err := store.Load(forkedPath)
	if err != nil {
		t.Fatalf("Load the fork: %v", err)
	}
	// Count message entries: a label lives on the path but is not conversation.
	var messages int
	for _, entry := range tree.ContextPath(tree.Leaf()) {
		if entry.Type == TypeMessage {
			messages++
		}
	}
	if messages != 2 {
		t.Fatalf("the fork projects to %d messages, want the copied conversation", messages)
	}
	if got, ok := tree.Label(first); !ok || got != label {
		t.Fatalf("the label did not survive the fork: %q %v", got, ok)
	}
}

// TestForkOfASessionWithoutAnAssistantReplyIsKept is the narrower half of the
// same hazard: the delete-if-empty rule exists for a session nobody used, and
// must not fire on a copy the caller explicitly asked for.
func TestForkOfASessionWithoutAnAssistantReplyIsKept(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()
	writer, err := store.Create(cwd, "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("asked but never answered")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("answered")); err != nil {
		t.Fatalf("append: %v", err)
	}
	id := writer.ID()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	info, err := store.Resolve(cwd, id)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	// Fork only the first entry: the copy legitimately contains no reply.
	tree, _, _, err := store.Load(info.Path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	firstID := tree.All()[0].ID
	forked, err := store.Fork(info, &firstID, cwd)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	forkedPath := forked.Path()
	if err := forked.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := os.Stat(forkedPath); err != nil {
		t.Fatalf("a deliberately created fork was deleted on close: %v", err)
	}
}
