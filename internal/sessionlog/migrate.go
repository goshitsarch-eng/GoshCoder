package sessionlog

import "encoding/json"

// Migration of pi's older session formats, ported from migrateV1ToV2 /
// migrateV2ToV3 in session-manager.ts.
//
// This is not in the original plan for this work, which assumed v3 files only.
// It is here because the real pi fixtures in the reference tree are v1, which
// means pi users have v1 and v2 sessions on disk today. Claiming interop while
// refusing to open most of a user's existing history would be a claim that
// does not survive first contact.
//
// Migration is read-only: the file on disk is never rewritten. An older
// session opens, resumes, and continues appending as v3 in a *new* file only
// if the user forks it. That is deliberate -- rewriting someone's pi history in
// place is not this tool's business, and pi migrates on its own next write.

// migrateEntries brings a decoded v1 or v2 entry up to v3 in place. It returns
// false when the entry cannot be salvaged.
//
// v1 has no entry ids at all: identity and parentage came from file order.
// Assigning ids in that order reconstructs exactly the chain the file implied.
type migrator struct {
	version  int
	previous *string
	taken    map[string]bool
	// v1 compaction referenced its kept entry by index into the file. Indices
	// are resolved after the whole file is read, since a compaction can point
	// forward as well as back.
	pendingCompactionIndex map[string]int
	entryIndex             []string
}

func newMigrator(version int) *migrator {
	return &migrator{
		version:                version,
		taken:                  map[string]bool{},
		pendingCompactionIndex: map[string]int{},
	}
}

// needed reports whether this file predates v3 and must be migrated.
func (m *migrator) needed() bool { return m.version < FormatVersion }

// apply fills in what an older format left implicit. raw is the original line,
// consulted for fields that only exist in the older shape.
func (m *migrator) apply(e *Entry, raw []byte) error {
	if m.version < 2 && e.ID == "" {
		id, err := newEntryID(func(candidate string) bool { return m.taken[candidate] })
		if err != nil {
			return err
		}
		m.taken[id] = true
		e.ID = id
		e.ParentID = m.previous
		next := id
		m.previous = &next

		if e.Type == TypeCompaction {
			var legacy struct {
				FirstKeptEntryIndex *int `json:"firstKeptEntryIndex"`
			}
			if err := json.Unmarshal(raw, &legacy); err == nil && legacy.FirstKeptEntryIndex != nil {
				m.pendingCompactionIndex[id] = *legacy.FirstKeptEntryIndex
			}
		}
		m.entryIndex = append(m.entryIndex, id)
	} else if m.version < 2 {
		// An entry that already carries an id in a v1 file is not something
		// this build wrote; keep its identity and let it anchor the chain.
		m.taken[e.ID] = true
		next := e.ID
		m.previous = &next
		m.entryIndex = append(m.entryIndex, e.ID)
	}
	if m.version < 3 && e.Type == TypeMessage {
		// v2 called hook-injected messages "hookMessage"; v3 calls them
		// "custom". Left as-is the role reaches llm.UnmarshalMessages, which
		// rejects an unknown role and fails the whole load.
		e.Message = renameLegacyRole(e.Message)
	}
	return nil
}

// finish resolves the v1 compaction indices now that every entry has an id.
// The index is into the file including its header line, which is why the
// lookup subtracts one.
func (m *migrator) finish(tree *Tree) {
	for id, index := range m.pendingCompactionIndex {
		target := index - 1
		if target < 0 || target >= len(m.entryIndex) {
			continue
		}
		entry, ok := tree.byID[id]
		if !ok {
			continue
		}
		entry.FirstKeptEntryID = m.entryIndex[target]
		tree.byID[id] = entry
	}
}

// renameLegacyRole rewrites a v2 "hookMessage" role to v3's "custom". Any
// other payload is returned untouched, including one that does not parse:
// this runs over every message of every migrated session and must not be able
// to damage one it does not understand.
func renameLegacyRole(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	var shaped map[string]json.RawMessage
	if err := json.Unmarshal(raw, &shaped); err != nil {
		return raw
	}
	roleRaw, ok := shaped["role"]
	if !ok {
		return raw
	}
	var role string
	if err := json.Unmarshal(roleRaw, &role); err != nil || role != "hookMessage" {
		return raw
	}
	shaped["role"] = json.RawMessage(`"custom"`)
	rewritten, err := json.Marshal(shaped)
	if err != nil {
		return raw
	}
	return rewritten
}
