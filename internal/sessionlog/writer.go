package sessionlog

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"

	"goshcoder/internal/lockfile"
)

// Size bounds.
//
// MaxEntryBytes sits above maxEditBytes (10 MiB, internal/tools), the largest
// payload the tool surface can produce, so it is a safety net rather than a
// routine limit. It is enforced at append as well as at load: a cap applied
// only on the way in would let a session grow past what this build can reopen,
// turning a working session into a permanently unopenable file.
const (
	MaxEntryBytes    = 16 << 20
	MaxSessionBytes  = 256 << 20
	WarnSessionBytes = 32 << 20
)

// ErrBusy is returned when a session file is claimed by another process.
var ErrBusy = errors.New("sessionlog: session is open in another process")

// ErrTooLarge is returned when an entry or a session exceeds its bound.
var ErrTooLarge = errors.New("sessionlog: entry exceeds the maximum size")

// Writer owns one session file: one claim, one append stream, one tree.
//
// Deviation from pi: pi buffers every write until the first assistant message
// arrives and then rewrites the file whole. That loses everything if the
// process dies between the user pressing enter and the model replying, and the
// rewrite is incompatible with O_APPEND. This writer appends from the first
// entry and instead deletes the file on Close when no assistant message ever
// landed -- the same observable outcome, with no window in which a real turn
// is only in memory.
type Writer struct {
	mu       sync.Mutex
	file     *os.File
	path     string
	header   Header
	tree     *Tree
	release  func()
	size     int64
	degraded error
	closed   bool
	readOnly bool
	// keep suppresses the delete-if-empty rule. A fork or an import is a file
	// the caller explicitly asked for, and its copied path may legitimately
	// contain no assistant reply -- rewinding to a question is exactly that.
	// Deleting it would destroy the copy the moment it was closed.
	keep bool
}

// Keep marks this session as one the caller created deliberately, so Close
// will not apply the delete-if-empty rule to it.
func (w *Writer) Keep() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.keep = true
}

// Path returns the session file's location.
func (w *Writer) Path() string { return w.path }

// Header returns the session's header.
func (w *Writer) Header() Header { return w.header }

// ID returns the session id.
func (w *Writer) ID() string { return w.header.ID }

// ReadOnly reports whether this writer refuses appends because it was opened
// without taking the claim.
func (w *Writer) ReadOnly() bool { return w.readOnly }

// Degraded returns the failure that stopped recording, or nil.
//
// A write failure does not abort the run: the transcript is intact in memory
// and the user is mid-task. It stops further writes -- retrying every turn
// against a full disk just produces noise -- and the interface reports it once.
func (w *Writer) Degraded() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.degraded
}

// Recording reports whether appends are still reaching disk. The interactive
// interface asks before promising the user their session is saved.
func (w *Writer) Recording() bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return !w.readOnly && !w.closed && w.degraded == nil
}

// Snapshot returns a copy of the tree that is safe to read from another
// goroutine while the writer keeps appending.
func (w *Writer) Snapshot() *Tree {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.snapshotLocked()
}

func (w *Writer) snapshotLocked() *Tree {
	clone := NewTree()
	for _, id := range w.tree.order {
		_ = clone.Add(w.tree.byID[id], w.tree.raw[id])
	}
	if w.tree.leafID != nil {
		_ = clone.SetLeaf(*w.tree.leafID)
	}
	return clone
}

// Leaf returns the id the next append will attach to.
func (w *Writer) Leaf() *string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.tree.Leaf()
}

// SetLeaf moves the write head so the next append starts a branch.
func (w *Writer) SetLeaf(id string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.tree.SetLeaf(id)
}

// Append writes one entry, attaching it to the current leaf. The entry's ID,
// ParentID and Timestamp are stamped here; anything the caller set in those
// fields is replaced.
func (w *Writer) Append(e Entry) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.appendLocked(e, w.tree.Leaf())
}

// AppendAt writes one entry attached to a specific parent, without moving the
// write head first. Used when reconstructing a branch.
func (w *Writer) AppendAt(parent *string, e Entry) (string, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.appendLocked(e, parent)
}

func (w *Writer) appendLocked(e Entry, parent *string) (string, error) {
	if w.closed {
		return "", errors.New("sessionlog: session is closed")
	}
	if w.readOnly {
		return "", errors.New("sessionlog: session is open read-only")
	}
	if w.degraded != nil {
		return "", w.degraded
	}

	id, err := newEntryID(w.tree.Has)
	if err != nil {
		return "", err
	}
	e.ID = id
	e.ParentID = parent
	e.Timestamp = Now()

	line, err := EncodeEntry(e)
	if err != nil {
		return "", err
	}
	// Refuse an oversized entry without writing a partial line: a half-written
	// entry would be indistinguishable from a torn tail on the next load.
	if len(line) > MaxEntryBytes {
		return "", fmt.Errorf("%w: %d bytes exceeds the %d-byte limit", ErrTooLarge, len(line), MaxEntryBytes)
	}
	if w.size+int64(len(line)) > MaxSessionBytes {
		w.stopLocked(fmt.Errorf("sessionlog: session file reached the %d-byte limit", MaxSessionBytes))
		return "", w.degraded
	}

	written, err := w.file.Write(line)
	w.size += int64(written)
	if err != nil {
		// A short write leaves a partial line. Truncating back to the last
		// complete entry is what keeps the file loadable.
		if written > 0 {
			_ = w.file.Truncate(w.size - int64(written))
			w.size -= int64(written)
		}
		w.stopLocked(fmt.Errorf("sessionlog: append to %s: %w", w.path, err))
		return "", w.degraded
	}
	if err := w.tree.Add(e, line); err != nil {
		return "", err
	}
	return id, nil
}

// Sync flushes the file to disk. Callers do this at a turn boundary rather
// than per entry: a completed turn is the unit a user would be upset to lose.
func (w *Writer) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.readOnly || w.degraded != nil || w.file == nil {
		return nil
	}
	if err := w.file.Sync(); err != nil {
		w.stopLocked(fmt.Errorf("sessionlog: sync %s: %w", w.path, err))
		return w.degraded
	}
	return nil
}

// stopLocked records the failure that ends recording. Only the first is kept:
// it is the one that explains what happened.
func (w *Writer) stopLocked(err error) {
	if w.degraded == nil {
		w.degraded = err
	}
}

// Close flushes, releases the claim, and removes the file when the session
// never got an assistant reply.
//
// Deleting an empty session is what keeps the sessions directory from filling
// with stubs from every accidental launch. It is deliberately keyed on an
// assistant message rather than on entry count: a session that recorded only a
// model choice and a prompt the user then abandoned is still nothing anyone
// would want to resume.
func (w *Writer) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true

	var firstErr error
	if w.file != nil {
		if !w.readOnly && w.degraded == nil {
			if err := w.file.Sync(); err != nil {
				firstErr = fmt.Errorf("sessionlog: sync %s: %w", w.path, err)
			}
		}
		if err := w.file.Close(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("sessionlog: close %s: %w", w.path, err)
		}
		w.file = nil
	}

	discard := !w.readOnly && !w.keep && !w.tree.HasAssistantMessage()
	if discard {
		if err := os.Remove(w.path); err != nil && !errors.Is(err, os.ErrNotExist) && firstErr == nil {
			firstErr = fmt.Errorf("sessionlog: remove empty session %s: %w", w.path, err)
		}
	}
	if w.release != nil {
		w.release()
		w.release = nil
	}
	return firstErr
}

// Discarded reports whether Close removed the file. The interface uses it to
// avoid telling the user where a session was saved when it was not.
func (w *Writer) Discarded() bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.closed && !w.readOnly && !w.keep && !w.tree.HasAssistantMessage()
}

// messageRole reads the role out of a raw message payload without decoding the
// whole thing. Used by listing and by the empty-session check, both of which
// run over every entry of every session and must not pay for full decoding.
func messageRole(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var head struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return ""
	}
	return head.Role
}

// claim takes the session file's exclusive claim, or reports its owner.
//
// Deviation from pi, which takes no lock in either format: two processes
// appending to one file is the realistic corruption mode for `chat -continue`
// run twice in one workspace, and interleaved appends produce a tree whose
// parent links cross between two conversations.
func claim(path string) (func(), lockfile.Owner, error) {
	release, owner, ok, err := lockfile.TryAcquire(path+".lock", lockfile.SessionTiming())
	if err != nil {
		return nil, lockfile.Owner{}, err
	}
	if !ok {
		return nil, owner, ErrBusy
	}
	return release, lockfile.Owner{}, nil
}
