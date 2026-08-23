package catalog

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"goshcoder/internal/lockfile"
)

// readAuthFile returns the raw provider->credential map on disk.
func readAuthFile(t *testing.T, path string) map[string]*Credential {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read auth.json: %v", err)
	}
	out := map[string]*Credential{}
	if err := json.Unmarshal(content, &out); err != nil {
		t.Fatalf("parse auth.json: %v", err)
	}
	return out
}

// fastLockTiming scales the lock's timings down so the tests below finish
// quickly while keeping their ratios -- a heartbeat well inside the stale
// threshold, and a wait well beyond it.
func fastLockTiming() lockfile.Timing {
	return lockfile.Timing{
		Heartbeat: 20 * time.Millisecond,
		// A second, not the 200ms this started as: a heartbeat goroutine
		// starved for longer than the stale window makes a live holder look
		// dead, and a 200ms stall is ordinary on a loaded CI runner. See
		// lockfile.fast in internal/lockfile.
		Stale: time.Second,
		Wait:  30 * time.Second,
		Retry: 5 * time.Millisecond,
	}
}

// newFastLockStore is NewFileCredentialStore with the shrunken timings.
func newFastLockStore(path string) *CredentialStore {
	store := NewFileCredentialStore(path)
	store.lockTiming = fastLockTiming()
	return store
}

// TestSlowModifyKeepsItsLock covers the case that matters for OAuth: Modify
// runs the caller's function -- a token refresh, which is network I/O and can
// legitimately take minutes when the provider is retrying -- while holding the
// cross-process lock. A waiter must not decide the live holder is stale and
// take the lock from underneath it, because both processes then write the file
// from snapshots taken before the other's write and one rotated refresh token
// is silently destroyed.
func TestSlowModifyKeepsItsLock(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "auth.json")

	slow := newFastLockStore(path)
	if _, err := slow.Modify("kimi-coding", func(*Credential) (*Credential, error) {
		return &Credential{Type: "oauth", Refresh: "kimi-v1", Access: "K1"}, nil
	}); err != nil {
		t.Fatalf("seed kimi: %v", err)
	}
	if _, err := slow.Modify("anthropic", func(*Credential) (*Credential, error) {
		return &Credential{Type: "oauth", Refresh: "anthropic-v1", Access: "A1"}, nil
	}); err != nil {
		t.Fatalf("seed anthropic: %v", err)
	}

	// A separate store stands in for a second process: it shares the file and
	// the lock file but not the in-process mutex.
	fast := newFastLockStore(path)

	inFn := make(chan struct{})
	releaseFn := make(chan struct{})

	var wg sync.WaitGroup
	wg.Add(1)
	var slowErr error
	go func() {
		defer wg.Done()
		_, slowErr = slow.Modify("kimi-coding", func(current *Credential) (*Credential, error) {
			close(inFn)
			<-releaseFn // stands in for a long token exchange
			next := current.clone()
			next.Refresh = "kimi-v2"
			next.Access = "K2"
			return next, nil
		})
	}()

	<-inFn

	// The holder is mid-refresh. The waiter must block rather than reclaim.
	fastDone := make(chan error, 1)
	go func() {
		_, err := fast.Modify("anthropic", func(current *Credential) (*Credential, error) {
			next := current.clone()
			next.Refresh = "anthropic-v2-rotated"
			next.Access = "A2"
			return next, nil
		})
		fastDone <- err
	}()

	select {
	case err := <-fastDone:
		t.Fatalf("the waiter took the lock from a live holder (err=%v)", err)
	case <-time.After(2 * fastLockTiming().Stale):
		// Still blocked, which is correct: the holder heartbeats its lock.
	}

	close(releaseFn)
	wg.Wait()
	if slowErr != nil {
		t.Fatalf("slow Modify: %v", slowErr)
	}
	if err := <-fastDone; err != nil {
		t.Fatalf("waiting Modify: %v", err)
	}

	stored := readAuthFile(t, path)
	if got := stored["anthropic"]; got == nil || got.Refresh != "anthropic-v2-rotated" {
		t.Fatalf("anthropic refresh token = %#v, want the rotated value to survive", got)
	}
	if got := stored["kimi-coding"]; got == nil || got.Refresh != "kimi-v2" {
		t.Fatalf("kimi refresh token = %#v, want the refreshed value", got)
	}
}

// TestUnlockDoesNotRemoveAnotherHoldersLock covers the release path: a store
// whose lock was reclaimed as stale must not delete the reclaiming process's
// lock file, which would admit a third writer.
func TestUnlockDoesNotRemoveAnotherHoldersLock(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "auth.json")
	lockPath := path + ".lock"

	store := newFastLockStore(path)
	unlock, err := store.lock()
	if err != nil {
		t.Fatalf("lock: %v", err)
	}

	// Simulate the lock having been reclaimed and re-taken by someone else.
	if err := os.WriteFile(lockPath, []byte("a-different-holder\n"), 0o600); err != nil {
		t.Fatalf("overwrite lock: %v", err)
	}

	unlock()

	if _, err := os.Stat(lockPath); err != nil {
		t.Fatalf("unlock removed another holder's lock file: %v", err)
	}
	if content, err := os.ReadFile(lockPath); err != nil || !strings.Contains(string(content), "a-different-holder") {
		t.Fatalf("lock file content = %q, %v", content, err)
	}
}

// TestModifyPreservesConcurrentWritesToOtherProviders covers the whole-file
// write: Modify marshals the entire in-memory map, so an entry another process
// wrote while fn was running must not be rolled back.
func TestModifyPreservesConcurrentWritesToOtherProviders(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "auth.json")

	store := newFastLockStore(path)
	if _, err := store.Modify("anthropic", func(*Credential) (*Credential, error) {
		return &Credential{Type: "oauth", Refresh: "anthropic-v1"}, nil
	}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	_, err := store.Modify("kimi-coding", func(*Credential) (*Credential, error) {
		// Another process writes a rotated Anthropic token while this
		// callback runs; it cannot take the lock here, but the equivalent
		// happens whenever a lock is legitimately reclaimed after a crash.
		other := map[string]*Credential{
			"anthropic": {Type: "oauth", Refresh: "anthropic-v2-rotated"},
		}
		content, _ := json.MarshalIndent(other, "", "  ")
		if writeErr := os.WriteFile(path, content, 0o600); writeErr != nil {
			t.Fatalf("simulate other process: %v", writeErr)
		}
		return &Credential{Type: "oauth", Refresh: "kimi-v1"}, nil
	})
	if err != nil {
		t.Fatalf("Modify: %v", err)
	}

	stored := readAuthFile(t, path)
	if got := stored["anthropic"]; got == nil || got.Refresh != "anthropic-v2-rotated" {
		t.Fatalf("anthropic = %#v, want the concurrently written token preserved", got)
	}
	if got := stored["kimi-coding"]; got == nil || got.Refresh != "kimi-v1" {
		t.Fatalf("kimi-coding = %#v, want this Modify's own write", got)
	}
}
