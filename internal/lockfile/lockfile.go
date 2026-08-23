// Package lockfile provides the cross-process advisory lock used for state
// files in the agent directory.
//
// Extracted from internal/llm/catalog/auth.go, which is where the protocol was
// designed and hardened; see the timing commentary on Timing for why the
// heartbeat and the stale threshold are separate knobs. Nothing about the
// on-disk protocol changed in the move, so an auth.json lock written by an
// older build is still understood by this one.
//
// The protocol: a holder creates `<path>` with O_EXCL and writes a token that
// is unique to the acquisition, then touches the file on a heartbeat for as
// long as it holds the lock. A waiter that finds the file may reclaim it only
// once its mtime has gone stale, and the winner of a reclaim race is decided
// by the next O_EXCL create rather than by who removed the file. Release
// removes the file only while the token still matches, so a holder that was
// declared stale cannot delete its successor's lock.
package lockfile

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ErrTimeout is returned by Acquire when Wait elapses with the lock still held
// by a live owner.
var ErrTimeout = errors.New("lockfile: timed out acquiring lock")

// Timing groups a lock's timings. It is passed per call rather than kept in
// package variables so that tests can shrink it without a data race against a
// running heartbeat.
//
// A holder proves it is alive by touching the lock file on a heartbeat, and a
// waiter only reclaims a lock whose heartbeat has stopped. Separating the two
// matters because a lock is held across the caller's work, and for OAuth that
// work is a token exchange: Kimi alone retries three times behind a 30 s HTTP
// timeout with 1+2+4 s of backoff, so a legitimate hold can approach two
// minutes. When the stale threshold is not comfortably longer than the longest
// hold -- as it was when both were 30 s and nothing refreshed the mtime -- a
// waiter declares a live holder dead, both processes write from snapshots
// taken before the other's write, and one rotated refresh token is silently
// destroyed.
type Timing struct {
	Heartbeat time.Duration
	// Stale is how long a lock may go untouched before a waiter treats its
	// owner as dead. Only reached if the owner crashed or was SIGKILLed,
	// since a live owner refreshes the file every heartbeat.
	Stale time.Duration
	// Wait bounds how long Acquire waits for a sibling process. It must
	// exceed the worst-case legitimate hold or a concurrent process fails to
	// save work it has already done.
	Wait  time.Duration
	Retry time.Duration
}

// DefaultTiming is the credential store's timing: a long Wait, because the
// work under the lock is a network token exchange worth waiting for.
func DefaultTiming() Timing {
	return Timing{
		Heartbeat: 2 * time.Second,
		Stale:     20 * time.Second,
		Wait:      5 * time.Minute,
		Retry:     20 * time.Millisecond,
	}
}

// SessionTiming is the session log's timing. Wait is short because a session
// claim is held for the lifetime of the process: a second window will not get
// it by waiting, so failing fast with the owner's pid beats blocking for
// minutes on a lock that is not going to be released.
func SessionTiming() Timing {
	return Timing{
		Heartbeat: 2 * time.Second,
		Stale:     20 * time.Second,
		Wait:      2 * time.Second,
		Retry:     20 * time.Millisecond,
	}
}

// Owner identifies the process currently holding a lock, for error messages.
// Both fields are best-effort: a lock file can be read between its create and
// its write, and a pid is not unique across containers sharing a directory.
type Owner struct {
	PID int
	// Since is the lock file's last heartbeat, not its creation time.
	Since time.Time
}

// String renders an owner for a user-facing message.
func (o Owner) String() string {
	if o.PID == 0 {
		return "another process"
	}
	return "pid " + strconv.Itoa(o.PID)
}

// token returns a value unique to this acquisition. The pid alone is not
// enough: pids are recycled, and two containers sharing a mounted agent
// directory can hold the same one simultaneously.
func token() (string, error) {
	var random [16]byte
	if _, err := rand.Read(random[:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("%d %s\n", os.Getpid(), hex.EncodeToString(random[:])), nil
}

// readOwner parses a lock file's token. A malformed or unreadable file yields
// the zero Owner rather than an error: the caller is reporting who holds a
// lock it failed to take, and that report must not itself fail.
func readOwner(path string) Owner {
	var owner Owner
	if info, err := os.Stat(path); err == nil {
		owner.Since = info.ModTime()
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return owner
	}
	fields := strings.Fields(string(content))
	if len(fields) == 0 {
		return owner
	}
	if pid, err := strconv.Atoi(fields[0]); err == nil {
		owner.PID = pid
	}
	return owner
}

// Acquire takes the lock at path, waiting up to t.Wait for a live holder to
// release it. The returned release function is idempotent.
//
// It returns ErrTimeout (wrapped) when the wait elapses; use errors.Is to
// distinguish that from an I/O failure.
func Acquire(path string, t Timing) (func(), error) {
	release, _, ok, err := acquire(path, t)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, ErrTimeout
	}
	return release, nil
}

// TryAcquire takes the lock at path without waiting. When ok is false the lock
// is held elsewhere and owner names the holder; err is non-nil only for an I/O
// failure, which is a different condition from a busy lock.
func TryAcquire(path string, t Timing) (release func(), owner Owner, ok bool, err error) {
	t.Wait = 0
	return acquire(path, t)
}

func acquire(path string, t Timing) (func(), Owner, bool, error) {
	tok, err := token()
	if err != nil {
		return nil, Owner{}, false, fmt.Errorf("lockfile: failed to generate a lock token: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, Owner{}, false, fmt.Errorf("lockfile: failed to create the lock directory: %w", err)
	}
	deadline := time.Now().Add(t.Wait)
	for {
		f, openErr := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if openErr == nil {
			if _, writeErr := f.WriteString(tok); writeErr != nil {
				f.Close()
				os.Remove(path)
				return nil, Owner{}, false, fmt.Errorf("lockfile: failed to initialize lock: %w", writeErr)
			}
			if closeErr := f.Close(); closeErr != nil {
				os.Remove(path)
				return nil, Owner{}, false, fmt.Errorf("lockfile: failed to initialize lock: %w", closeErr)
			}
			return hold(path, tok, t.Heartbeat), Owner{}, true, nil
		}
		if !errors.Is(openErr, os.ErrExist) {
			return nil, Owner{}, false, fmt.Errorf("lockfile: failed to acquire lock: %w", openErr)
		}
		// Reclaim a lock whose owner stopped heartbeating. Removing it can
		// race another waiter doing the same, which is why the winner is
		// decided by the O_EXCL create on the next iteration rather than by
		// who removed the file.
		if info, statErr := os.Stat(path); statErr == nil && time.Since(info.ModTime()) > t.Stale {
			_ = os.Remove(path)
			continue
		}
		if !time.Now().Before(deadline) {
			return nil, readOwner(path), false, nil
		}
		time.Sleep(t.Retry)
	}
}

// hold keeps path's mtime fresh for as long as the lock is held and returns
// the release function.
func hold(path, tok string, heartbeat time.Duration) func() {
	done := make(chan struct{})
	go func() {
		ticker := time.NewTicker(heartbeat)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case now := <-ticker.C:
				// Best effort: a failure here only risks the lock being
				// reclaimed, which the ownership check below still contains.
				_ = os.Chtimes(path, now, now)
			}
		}
	}()

	var once sync.Once
	return func() {
		once.Do(func() {
			close(done)
			// Remove the lock only while it is still ours. If a waiter
			// declared us stale and took it, deleting its file would admit a
			// third writer.
			if content, err := os.ReadFile(path); err != nil || string(content) != tok {
				return
			}
			_ = os.Remove(path)
		})
	}
}
