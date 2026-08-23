package sessionlog

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/llm"
)

// testdata/pi-v1-session.jsonl is a contiguous prefix of a real pi session,
// taken from the reference tree's own fixtures. Long text payloads were
// shortened to keep it committable; no structural field was touched and no
// entry was reordered or dropped from the middle, because v1 derives parentage
// from file order and any edit to that ordering would test a file pi could
// never have written.
//
// testdata/pi-v1-compaction.jsonl is synthetic, and says so. The real
// fixture's compaction entries point at file offsets 293 and 551, so covering
// the firstKeptEntryIndex conversion with real data would mean committing the
// whole 2.3 MB file. This one has the same shape at a size worth keeping.
const (
	piFixture           = "testdata/pi-v1-session.jsonl"
	piCompactionFixture = "testdata/pi-v1-compaction.jsonl"
)

// TestRealPiSessionLoads is the interop claim made executable. A hand-written
// fixture only proves this package agrees with itself; this one proves it
// agrees with pi.
func TestRealPiSessionLoads(t *testing.T) {
	store := newStore(t)
	tree, header, report, err := store.Load(piFixture)
	if err != nil {
		t.Fatalf("Load a real pi session: %v", err)
	}
	if report.SkippedLines != 0 {
		t.Fatalf("skipped %d lines of a real pi session: %v", report.SkippedLines, report.Warnings)
	}
	if !report.Migrated || report.SourceVersion != 1 {
		t.Fatalf("report = migrated:%v version:%d, want a v1 migration", report.Migrated, report.SourceVersion)
	}
	if header.Cwd == "" {
		t.Fatal("header lost its cwd")
	}

	// Every entry must have gained an identity, and the chain the file implied
	// by ordering must be intact.
	var previous string
	for i, entry := range tree.All() {
		if entry.ID == "" {
			t.Fatalf("entry %d has no id after migration", i)
		}
		if i == 0 {
			if entry.ParentID != nil {
				t.Fatalf("first entry has parent %s, want nil", *entry.ParentID)
			}
		} else if entry.ParentID == nil || *entry.ParentID != previous {
			t.Fatalf("entry %d parent = %v, want %s", i, entry.ParentID, previous)
		}
		previous = entry.ID
	}
}

// TestRealPiMessagesDecodeIntoThisBuildsTypes is the half that matters for
// resume: the tree can be intact while every payload is unreadable.
func TestRealPiMessagesDecodeIntoThisBuildsTypes(t *testing.T) {
	store := newStore(t)
	tree, _, _, err := store.Load(piFixture)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	var messages int
	for _, entry := range tree.All() {
		if entry.Type != TypeMessage {
			continue
		}
		decoded, err := llm.UnmarshalMessages(json.RawMessage("[" + string(entry.Message) + "]"))
		if err != nil {
			t.Fatalf("entry %s does not decode into this build's message types: %v", entry.ID, err)
		}
		if len(decoded) != 1 {
			t.Fatalf("entry %s decoded to %d messages", entry.ID, len(decoded))
		}
		messages++
	}
	if messages == 0 {
		t.Fatal("the fixture contained no messages; it is not exercising anything")
	}
}

// TestRealPiCompactionProjectsCorrectly covers the v1 compaction field that
// has no v3 equivalent: firstKeptEntryIndex is a position in the file, and it
// has to become the id of whatever entry sat at that position.
func TestRealPiCompactionProjectsCorrectly(t *testing.T) {
	store := newStore(t)
	tree, _, report, err := store.Load(piCompactionFixture)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !report.Migrated {
		t.Fatal("the fixture is not v1; it is not exercising the conversion")
	}

	var compactions int
	for _, entry := range tree.All() {
		if entry.Type != TypeCompaction {
			continue
		}
		compactions++
		if entry.FirstKeptEntryID == "" {
			t.Fatalf("compaction %s has no firstKeptEntryId after migration", entry.ID)
		}
		if !tree.Has(entry.FirstKeptEntryID) {
			t.Fatalf("compaction %s points at %s, which is not in the tree", entry.ID, entry.FirstKeptEntryID)
		}
	}
	if compactions == 0 {
		t.Fatal("the fixture has no compaction entry; it is not exercising the index conversion")
	}

	// v1 counts the header as index 0, so firstKeptEntryIndex 3 is the third
	// entry -- "the first kept entry". Getting the off-by-one wrong here
	// silently keeps or drops one turn on every migrated session.
	full := tree.Path(nil)
	context := tree.ContextPath(nil)
	got := texts(context)
	want := []string{"[compaction]", "the first kept entry", "after the compaction"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("context = %q, want %q", got, want)
	}
	if len(context) >= len(full) {
		t.Fatalf("context path (%d) is not shorter than the full path (%d)", len(context), len(full))
	}
}

// TestLegacySessionCannotBeAppendedTo covers the trap the migration creates: a
// v1 file's ids exist only in the fold, so appending a v3 entry would leave a
// file that migrates differently next time -- the new entry's id reassigned and
// its children orphaned.
func TestLegacySessionCannotBeAppendedTo(t *testing.T) {
	store := newStore(t)
	copied := filepath.Join(t.TempDir(), "legacy.jsonl")
	content, err := os.ReadFile(piFixture)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	if err := os.WriteFile(copied, content, 0o600); err != nil {
		t.Fatalf("copy fixture: %v", err)
	}

	_, report, err := store.Attach(copied)
	if !errors.Is(err, ErrLegacyFormat) {
		t.Fatalf("Attach error = %v, want ErrLegacyFormat", err)
	}
	if !report.Migrated {
		t.Fatal("the report did not say the file was legacy")
	}
	// The refusal must not have taken the claim or altered the file.
	after, err := os.ReadFile(copied)
	if err != nil || len(after) != len(content) {
		t.Fatalf("a refused Attach modified the file: %v", err)
	}
	if _, statErr := os.Stat(copied + ".lock"); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatal("a refused Attach left its claim behind")
	}
}

// TestForkingALegacySessionProducesAValidV3File is the recovery path for the
// refusal above. Fork copies raw lines for a v3 source, which would produce
// id-less entries here, so it must re-encode instead.
func TestForkingALegacySessionProducesAValidV3File(t *testing.T) {
	store := newStore(t)
	cwd := t.TempDir()

	source, err := store.describe(piFixture, false)
	if err != nil {
		t.Fatalf("describe fixture: %v", err)
	}
	forked, err := store.Fork(source, nil, cwd)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	path := forked.Path()
	if err := forked.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tree, header, report, err := store.Load(path)
	if err != nil {
		t.Fatalf("Load the fork: %v", err)
	}
	if report.Migrated {
		t.Fatal("the fork is still in the legacy format")
	}
	if header.Version != FormatVersion {
		t.Fatalf("fork version = %d, want %d", header.Version, FormatVersion)
	}
	if header.ParentSession == "" {
		t.Fatal("the fork does not record which session it came from")
	}
	if tree.Len() == 0 {
		t.Fatal("the fork is empty")
	}
	// Every line must now carry a real id, which is what makes the fork
	// appendable where its source was not.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fork: %v", err)
	}
	for i, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if i == 0 {
			continue
		}
		var probe struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal([]byte(line), &probe); err != nil || probe.ID == "" {
			t.Fatalf("forked line %d has no id: %s", i+1, line[:min(len(line), 120)])
		}
	}

	// And the fork must be appendable, which is the whole point.
	attached, _, err := store.Attach(path)
	if err != nil {
		t.Fatalf("Attach the fork: %v", err)
	}
	if _, err := attached.Append(assistantEntry("continuing a pi session")); err != nil {
		t.Fatalf("append to the fork: %v", err)
	}
	if err := attached.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// TestGoshCoderSessionSatisfiesPiHeaderAndEntryChecks is the other direction:
// a file this build wrote must pass the validation pi applies when reading.
// Transliterated from loadEntriesFromFile and _buildIndex.
func TestGoshCoderSessionSatisfiesPiHeaderAndEntryChecks(t *testing.T) {
	store := newStore(t)
	writer, err := store.Create(t.TempDir(), "")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := writer.Append(userEntry("a question")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(assistantEntry("an answer")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := writer.Append(Entry{Type: TypeModelChange, Provider: "anthropic", ModelID: "claude-sonnet-4-5"}); err != nil {
		t.Fatalf("append model change: %v", err)
	}
	path := writer.Path()
	if err := writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	lines := strings.Split(strings.TrimSuffix(string(raw), "\n"), "\n")

	var header struct {
		Type      string `json:"type"`
		Version   int    `json:"version"`
		ID        string `json:"id"`
		Timestamp string `json:"timestamp"`
		Cwd       string `json:"cwd"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &header); err != nil {
		t.Fatalf("pi could not parse the header: %v", err)
	}
	if header.Type != "session" || header.Version != 3 || header.ID == "" || header.Timestamp == "" || header.Cwd == "" {
		t.Fatalf("header does not satisfy pi's SessionHeader: %+v", header)
	}

	seen := map[string]bool{}
	for i, line := range lines[1:] {
		// pi decodes into SessionEntryBase: type, id, parentId, timestamp,
		// with parentId required to be present and either a string or null.
		var fields map[string]json.RawMessage
		if err := json.Unmarshal([]byte(line), &fields); err != nil {
			t.Fatalf("pi could not parse entry line %d: %v", i+2, err)
		}
		for _, required := range []string{"type", "id", "parentId", "timestamp"} {
			if _, ok := fields[required]; !ok {
				t.Fatalf("entry line %d is missing %q, which pi's SessionEntryBase requires", i+2, required)
			}
		}
		var id string
		if err := json.Unmarshal(fields["id"], &id); err != nil || id == "" {
			t.Fatalf("entry line %d has an unusable id", i+2)
		}
		if seen[id] {
			t.Fatalf("entry line %d reuses id %s; pi's index would collapse them", i+2, id)
		}
		seen[id] = true
		if string(fields["parentId"]) != "null" {
			var parent string
			if err := json.Unmarshal(fields["parentId"], &parent); err != nil {
				t.Fatalf("entry line %d has a parentId that is neither a string nor null", i+2)
			}
			if !seen[parent] {
				t.Fatalf("entry line %d references parent %s before it appears", i+2, parent)
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
