package lockfile

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fast is the production timing compressed so the hazards below are reachable
// in a test, and production constants are never slept on. The ratio is what
// matters -- a heartbeat an order of magnitude shorter than the stale
// threshold, and a wait well beyond it -- not the absolute values.
//
// The stale window is a whole second rather than the 200ms it started as. A
// live holder is declared dead if its heartbeat goroutine is starved for
// longer than that window, and on a CI runner executing every package in
// parallel -- more so under -race, which multiplies every scheduling delay --
// a 200ms stall is ordinary: these tests passed by luck. One second still
// exercises the same behaviour and needs a stall two orders of magnitude past
// the heartbeat to produce a false failure. Production is 2s/20s, where the
// equivalent stall would have to last twenty seconds.
func fast() Timing {
	return Timing{
		Heartbeat: 20 * time.Millisecond,
		Stale:     time.Second,
		Wait:      30 * time.Second,
		Retry:     5 * time.Millisecond,
	}
}

// TestAcquireCreatesAndReleaseRemoves is the baseline both hazard tests below
// are contrasted against: without it, a test asserting "the lock file is still
// there" cannot distinguish correct retention from a lock that was never
// removed under any circumstances.
func TestAcquireCreatesAndReleaseRemoves(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")

	release, err := Acquire(path, fast())
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("lock file missing while held: %v", err)
	}

	release()
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lock file survived release: %v", err)
	}
}

// TestReleaseIsIdempotent covers the deferred-release path: callers defer the
// release and may also call it explicitly on a success path, and a second call
// must not remove a lock a later acquisition has since taken.
func TestReleaseIsIdempotent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")

	release, err := Acquire(path, fast())
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	release()

	second, err := Acquire(path, fast())
	if err != nil {
		t.Fatalf("re-acquire: %v", err)
	}
	defer second()

	release() // the stale first release
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("a repeated release removed the second holder's lock: %v", err)
	}
}

// TestTryAcquireReportsTheLiveOwnerInsteadOfWaiting covers the session claim:
// a second window cannot win by waiting, because the holder keeps the lock for
// the lifetime of its process, so the caller needs the owner's identity to put
// in an error message rather than a timeout minutes later.
func TestTryAcquireReportsTheLiveOwnerInsteadOfWaiting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.lock")

	release, err := Acquire(path, fast())
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer release()

	start := time.Now()
	_, owner, ok, err := TryAcquire(path, fast())
	if err != nil {
		t.Fatalf("TryAcquire returned an I/O error for a busy lock: %v", err)
	}
	if ok {
		t.Fatal("TryAcquire took a lock that was already held")
	}
	if owner.PID != os.Getpid() {
		t.Fatalf("owner.PID = %d, want this process %d", owner.PID, os.Getpid())
	}
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("TryAcquire waited %v; it must not wait at all", elapsed)
	}
}

// TestStaleLockIsReclaimed covers recovery from a crashed holder: nothing
// refreshes the mtime after a SIGKILL, so a waiter must eventually take over
// or the state file is locked out forever.
func TestStaleLockIsReclaimed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")
	timing := fast()

	// A lock file with no live holder behind it, as a crash would leave.
	if err := os.WriteFile(path, []byte("999999 deadbeef\n"), 0o600); err != nil {
		t.Fatalf("seed lock: %v", err)
	}
	old := time.Now().Add(-2 * timing.Stale)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatalf("age lock: %v", err)
	}

	release, err := Acquire(path, timing)
	if err != nil {
		t.Fatalf("Acquire did not reclaim a stale lock: %v", err)
	}
	release()
}

// TestLiveHolderIsNotReclaimed is the negative control for the test above: the
// heartbeat must keep a legitimately long hold alive. Without it, a waiter
// declares a working holder dead, both processes write from stale snapshots,
// and one of the two writes is silently destroyed.
func TestLiveHolderIsNotReclaimed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")
	timing := fast()

	release, err := Acquire(path, timing)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer release()

	// Hold past the stale threshold; the heartbeat should keep it fresh.
	time.Sleep(2 * timing.Stale)

	timing.Wait = 0
	_, owner, ok, err := TryAcquire(path, timing)
	if err != nil {
		t.Fatalf("TryAcquire: %v", err)
	}
	if ok {
		t.Fatal("a live holder's lock was reclaimed as stale")
	}
	if owner.PID != os.Getpid() {
		t.Fatalf("owner.PID = %d, want %d", owner.PID, os.Getpid())
	}
}

// TestReleaseStopsTheHeartbeatBeforeReturning covers the window between asking
// the heartbeat to stop and it actually stopping. A beat that lands after
// release has returned touches a lock file this process no longer owns -- on
// Windows it also holds the handle that turns the unlink into a delete-pending
// state, where every waiter's open answers "Access is denied" on a lock nobody
// holds.
func TestReleaseStopsTheHeartbeatBeforeReturning(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")
	timing := fast()
	timing.Heartbeat = time.Millisecond

	release, err := Acquire(path, timing)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	release()

	// Stand in for the successor: a lock file this process must not touch.
	if err := os.WriteFile(path, []byte("4242 a-different-holder\n"), 0o600); err != nil {
		t.Fatalf("seed successor lock: %v", err)
	}
	before, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat successor lock: %v", err)
	}

	// Long enough that a heartbeat still running would have beaten many times.
	time.Sleep(100 * timing.Heartbeat)

	after, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat successor lock: %v", err)
	}
	if !after.ModTime().Equal(before.ModTime()) {
		t.Fatalf("a released holder's heartbeat touched the successor's lock: %v -> %v",
			before.ModTime(), after.ModTime())
	}
}

// TestDeletePendingIsWindowsOnly pins the classification acquire relies on. A
// permission error on a lock path means one thing on Windows, where it is how
// an unlink in progress reports itself, and another everywhere else, where it
// is a path this process genuinely cannot write and retrying to the deadline
// would only delay saying so.
func TestDeletePendingIsWindowsOnly(t *testing.T) {
	wrapped := &os.PathError{Op: "open", Path: "state.lock", Err: os.ErrPermission}
	if got, want := deletePending(wrapped), runtime.GOOS == "windows"; got != want {
		t.Errorf("deletePending(permission) = %v on %s, want %v", got, runtime.GOOS, want)
	}
	if deletePending(&os.PathError{Op: "open", Path: "state.lock", Err: os.ErrExist}) {
		t.Error("deletePending(exists) = true; only a permission error is the pending-unlink case")
	}
	if deletePending(nil) {
		t.Error("deletePending(nil) = true")
	}
}

// TestReleaseDoesNotRemoveAnotherHoldersLock covers the release path when this
// holder was declared stale and the lock re-taken: deleting the successor's
// file would admit a third writer.
func TestReleaseDoesNotRemoveAnotherHoldersLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")

	release, err := Acquire(path, fast())
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	if err := os.WriteFile(path, []byte("4242 a-different-holder\n"), 0o600); err != nil {
		t.Fatalf("overwrite lock: %v", err)
	}
	release()

	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("release removed another holder's lock file: %v", err)
	}
	if !strings.Contains(string(content), "a-different-holder") {
		t.Fatalf("lock content = %q, want the other holder's token", content)
	}
}

// TestAcquireTimesOutWithErrTimeout covers the waiting path's failure mode.
// Callers distinguish a busy lock from an I/O failure with errors.Is, and
// report them very differently, so the sentinel has to survive wrapping.
func TestAcquireTimesOutWithErrTimeout(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")
	timing := fast()

	release, err := Acquire(path, timing)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	defer release()

	timing.Wait = 30 * time.Millisecond
	if _, err := Acquire(path, timing); !errors.Is(err, ErrTimeout) {
		t.Fatalf("second Acquire error = %v, want ErrTimeout", err)
	}
}

// TestOwnerOfAMalformedLockDoesNotFail covers a lock file read between another
// process's O_EXCL create and its token write. Reporting who holds a lock must
// never itself fail, or a busy-lock message becomes an I/O error.
func TestOwnerOfAMalformedLockDoesNotFail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.lock")
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}

	timing := fast()
	timing.Stale = time.Hour // keep it from being reclaimed
	_, owner, ok, err := TryAcquire(path, timing)
	if err != nil {
		t.Fatalf("TryAcquire: %v", err)
	}
	if ok {
		t.Fatal("took a lock that exists")
	}
	if owner.PID != 0 {
		t.Fatalf("owner.PID = %d, want 0 for an unreadable token", owner.PID)
	}
	if got := owner.String(); got != "another process" {
		t.Fatalf("owner.String() = %q, want a usable fallback", got)
	}
}

// TestUnremovableStaleLockFailsInsteadOfSpinning covers a lock path that looks
// stale but cannot be unlinked: another user's file in a sticky directory, a
// read-only mount, or a directory left where the lock belongs.
//
// The reclaim branch used to `continue` on a failed removal, skipping both the
// deadline check and the retry sleep -- an unkillable 100% CPU spin. It is
// reachable from a session listing, and a listing runs inside the render loop,
// so one such path would hang the whole interface.
func TestUnremovableStaleLockFailsInsteadOfSpinning(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "state.lock")

	// A non-empty directory where the lock file should be: os.Remove refuses it.
	if err := os.MkdirAll(filepath.Join(path, "occupied"), 0o700); err != nil {
		t.Fatalf("seed: %v", err)
	}
	old := time.Now().Add(-time.Hour)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatalf("age: %v", err)
	}

	timing := fast()
	done := make(chan error, 1)
	go func() {
		_, _, _, err := TryAcquire(path, timing)
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("TryAcquire reported success against an unusable lock path")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("TryAcquire never returned; it is spinning on an unremovable stale lock")
	}
}
