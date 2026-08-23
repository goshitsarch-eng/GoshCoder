package sessionlog

import (
	"errors"
	"fmt"
)

// Tree is the in-memory fold of a session file: every entry, indexed by id,
// with the parent links that make it a tree. It performs no I/O, which is what
// lets every invariant live in one place and be tested without a filesystem.
//
// Port of _buildIndex / buildSessionPath / buildContextEntries.
type Tree struct {
	byID   map[string]Entry
	raw    map[string][]byte
	order  []string
	labels map[string]string
	name   string
	leafID *string
}

// NewTree returns an empty tree.
func NewTree() *Tree {
	return &Tree{
		byID:   map[string]Entry{},
		raw:    map[string][]byte{},
		labels: map[string]string{},
	}
}

// ErrDuplicateEntryID is returned when a file names the same entry twice,
// which would make the tree ambiguous.
var ErrDuplicateEntryID = errors.New("sessionlog: duplicate entry id")

// Add folds one entry into the tree. raw is the exact line it was read from,
// retained so that Fork can copy bytes rather than re-encode -- which is what
// preserves fields a newer writer added.
//
// An entry whose type this build does not know is indexed like any other. It
// joins the tree so that entries chained through it stay reachable, and
// projection simply produces nothing for it. This is the property that makes
// every later format addition backward-compatible.
func (t *Tree) Add(e Entry, raw []byte) error {
	if _, exists := t.byID[e.ID]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateEntryID, e.ID)
	}
	stored := make([]byte, len(raw))
	copy(stored, raw)

	t.byID[e.ID] = e
	t.raw[e.ID] = stored
	t.order = append(t.order, e.ID)

	// The leaf follows file order, not the parent chain: the last entry
	// written is where the next one attaches. pi does the same.
	id := e.ID
	t.leafID = &id

	switch e.Type {
	case TypeLabel:
		if e.Label != nil && *e.Label != "" {
			t.labels[e.TargetID] = *e.Label
		} else {
			delete(t.labels, e.TargetID)
		}
	case TypeSessionInfo:
		t.name = e.Name
	}
	return nil
}

// Leaf returns the id the next entry will attach to, or nil for an empty tree.
func (t *Tree) Leaf() *string {
	if t.leafID == nil {
		return nil
	}
	id := *t.leafID
	return &id
}

// SetLeaf moves the write head, which is how a branch is started: the next
// entry attaches to id rather than to the last entry in the file.
func (t *Tree) SetLeaf(id string) error {
	if _, ok := t.byID[id]; !ok {
		return fmt.Errorf("sessionlog: no entry %s to branch from", id)
	}
	t.leafID = &id
	return nil
}

// Has reports whether id is in use.
func (t *Tree) Has(id string) bool {
	_, ok := t.byID[id]
	return ok
}

// Entry returns one entry by id.
func (t *Tree) Entry(id string) (Entry, bool) {
	e, ok := t.byID[id]
	return e, ok
}

// Raw returns the exact line an entry was read from, for byte-preserving copy.
func (t *Tree) Raw(id string) ([]byte, bool) {
	raw, ok := t.raw[id]
	return raw, ok
}

// All returns every entry in file order.
func (t *Tree) All() []Entry {
	out := make([]Entry, 0, len(t.order))
	for _, id := range t.order {
		out = append(out, t.byID[id])
	}
	return out
}

// Children returns the entries whose parent is parent, in file order. A nil
// parent selects the roots.
func (t *Tree) Children(parent *string) []Entry {
	var out []Entry
	for _, id := range t.order {
		e := t.byID[id]
		switch {
		case parent == nil && e.ParentID == nil:
			out = append(out, e)
		case parent != nil && e.ParentID != nil && *e.ParentID == *parent:
			out = append(out, e)
		}
	}
	return out
}

// Path returns the entries from the root down to leaf, in order. A nil leaf
// selects the tree's current leaf.
//
// A cycle -- which a hand-edited or corrupt file can contain -- terminates the
// walk instead of hanging. Returning the prefix walked so far is strictly
// better than refusing to open the session.
func (t *Tree) Path(leaf *string) []Entry {
	id := leaf
	if id == nil {
		id = t.leafID
	}
	if id == nil {
		return nil
	}
	var reversed []Entry
	seen := map[string]bool{}
	current := *id
	for {
		if seen[current] {
			break
		}
		seen[current] = true
		e, ok := t.byID[current]
		if !ok {
			break
		}
		reversed = append(reversed, e)
		if e.ParentID == nil {
			break
		}
		current = *e.ParentID
	}
	path := make([]Entry, len(reversed))
	for i, e := range reversed {
		path[len(reversed)-1-i] = e
	}
	return path
}

// ContextPath returns the entries on the path to leaf that still participate
// in the model's context, honoring the latest compaction on that path.
//
// The compaction entry itself comes first, standing in for everything it
// summarized, followed by the entries kept from firstKeptEntryId onward and
// then everything after the compaction. Older summarized entries stay in the
// file and remain browsable; they are simply never sent again.
func (t *Tree) ContextPath(leaf *string) []Entry {
	path := t.Path(leaf)

	// A /clear cuts harder than a compaction: it keeps nothing. Whichever
	// marker is later on the path wins, so clearing after compacting discards
	// the summary too, and compacting after clearing summarizes only what came
	// after the clear.
	resetIdx := -1
	compactionIdx := -1
	for i, e := range path {
		switch e.Type {
		case TypeCompaction:
			compactionIdx = i
		case TypeTranscriptReset:
			resetIdx = i
		}
	}
	if resetIdx > compactionIdx {
		return path[resetIdx+1:]
	}
	if compactionIdx < 0 {
		return path
	}

	compaction := path[compactionIdx]
	out := []Entry{compaction}
	kept := false
	for i := 0; i < compactionIdx; i++ {
		if path[i].ID == compaction.FirstKeptEntryID {
			kept = true
		}
		if kept {
			out = append(out, path[i])
		}
	}
	out = append(out, path[compactionIdx+1:]...)
	return out
}

// Name returns the session's display name, or "" when none was set.
func (t *Tree) Name() string { return t.name }

// Label returns a user-defined bookmark on an entry.
func (t *Tree) Label(id string) (string, bool) {
	label, ok := t.labels[id]
	return label, ok
}

// Labels returns every live label, keyed by the entry it marks.
func (t *Tree) Labels() map[string]string {
	out := make(map[string]string, len(t.labels))
	for id, label := range t.labels {
		out[id] = label
	}
	return out
}

// Len returns the number of entries of every type.
func (t *Tree) Len() int { return len(t.order) }

// HasAssistantMessage reports whether the session ever got a reply. A session
// that never did is deleted on close rather than left as a stub; see
// Writer.Close.
func (t *Tree) HasAssistantMessage() bool {
	for _, id := range t.order {
		e := t.byID[id]
		if e.Type != TypeMessage {
			continue
		}
		if messageRole(e.Message) == "assistant" {
			return true
		}
	}
	return false
}
