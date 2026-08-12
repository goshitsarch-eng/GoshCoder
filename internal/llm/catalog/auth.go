package catalog

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Credential is one stored credential for a provider — the shape of a
// auth.json entry (auth/types.ts: ApiKeyCredential | OAuthCredential).
// Unknown OAuth fields (provider-specific extras) are preserved verbatim.
type Credential struct {
	Type string `json:"type"` // "api_key" | "oauth"

	// api_key fields. Key may itself be a config value ("$ENV_VAR",
	// "!command"); Env holds provider-scoped values (Cloudflare account /
	// gateway ids) and feeds config-value interpolation.
	Key string            `json:"key,omitempty"`
	Env map[string]string `json:"env,omitempty"`

	// oauth fields.
	Refresh string `json:"refresh,omitempty"`
	Access  string `json:"access,omitempty"`
	// Expires is a Unix milliseconds timestamp.
	Expires int64 `json:"expires,omitempty"`

	extra map[string]json.RawMessage
}

var credentialKnownFields = map[string]bool{
	"type": true, "key": true, "env": true,
	"refresh": true, "access": true, "expires": true,
}

func (c *Credential) UnmarshalJSON(data []byte) error {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	type plain struct {
		Type    string            `json:"type"`
		Key     string            `json:"key,omitempty"`
		Env     map[string]string `json:"env,omitempty"`
		Refresh string            `json:"refresh,omitempty"`
		Access  string            `json:"access,omitempty"`
		Expires int64             `json:"expires,omitempty"`
	}
	var p plain
	if err := json.Unmarshal(data, &p); err != nil {
		return err
	}
	c.Type, c.Key, c.Env, c.Refresh, c.Access, c.Expires = p.Type, p.Key, p.Env, p.Refresh, p.Access, p.Expires
	c.extra = nil
	for k, v := range raw {
		if !credentialKnownFields[k] {
			if c.extra == nil {
				c.extra = map[string]json.RawMessage{}
			}
			c.extra[k] = v
		}
	}
	return nil
}

func (c Credential) MarshalJSON() ([]byte, error) {
	out := map[string]json.RawMessage{}
	for k, v := range c.extra {
		out[k] = v
	}
	put := func(name string, value any) error {
		data, err := json.Marshal(value)
		if err != nil {
			return err
		}
		out[name] = data
		return nil
	}
	if err := put("type", c.Type); err != nil {
		return nil, err
	}
	if c.Key != "" {
		if err := put("key", c.Key); err != nil {
			return nil, err
		}
	}
	if len(c.Env) > 0 {
		if err := put("env", c.Env); err != nil {
			return nil, err
		}
	}
	if c.Type == "oauth" {
		if err := put("refresh", c.Refresh); err != nil {
			return nil, err
		}
		if err := put("access", c.Access); err != nil {
			return nil, err
		}
		if err := put("expires", c.Expires); err != nil {
			return nil, err
		}
	}
	return json.Marshal(out)
}

// SetExtra stores a provider-specific field on the credential, e.g. Codex's
// accountId. Extras round-trip through auth.json verbatim, so they stay
// interoperable with pi.
func (c *Credential) SetExtra(name string, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if credentialKnownFields[name] {
		return fmt.Errorf("catalog: %q is a known credential field, not an extra", name)
	}
	if c.extra == nil {
		c.extra = map[string]json.RawMessage{}
	}
	c.extra[name] = encoded
	return nil
}

// Extra reads a provider-specific string field. ok is false when absent or not
// a string.
func (c *Credential) Extra(name string) (value string, ok bool) {
	if c == nil || c.extra == nil {
		return "", false
	}
	raw, present := c.extra[name]
	if !present {
		return "", false
	}
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", false
	}
	return value, value != ""
}

// clone returns a deep copy.
func (c *Credential) clone() *Credential {
	if c == nil {
		return nil
	}
	data, err := json.Marshal(c)
	if err != nil {
		panic(fmt.Errorf("catalog: cannot clone credential: %v", err))
	}
	var out Credential
	if err := json.Unmarshal(data, &out); err != nil {
		panic(fmt.Errorf("catalog: cannot clone credential: %v", err))
	}
	return &out
}

// CredentialInfo is non-secret credential metadata for account/status
// enumeration (auth/types.ts).
type CredentialInfo struct {
	ProviderID string
	Type       string
}

// CredentialStore stores one credential per provider id (auth/types.ts
// CredentialStore). When constructed with NewFileCredentialStore it is backed
// by an auth.json file: a top-level object mapping provider id to credential,
// written with mode 0600. All writes go through Modify/Delete, which are
// serialized read-modify-write cycles guarded by an in-process mutex and a
// sibling lock file (`<path>.lock`), mirroring pi's AuthStorage
// (proper-lockfile) so concurrent OAuth refreshes cannot double-refresh.
type CredentialStore struct {
	path string // empty: in-memory only
	mu   sync.Mutex
	data map[string]*Credential
	// getenv feeds config-value resolution on Read; defaults to os.Getenv.
	getenv func(string) string
}

// NewInMemoryCredentialStore returns a store that keeps credentials in
// memory only (pi's InMemoryCredentialStore).
func NewInMemoryCredentialStore() *CredentialStore {
	return &CredentialStore{data: map[string]*Credential{}, getenv: os.Getenv}
}

// NewFileCredentialStore returns a store backed by the auth.json file at
// path. A missing file reads as empty.
func NewFileCredentialStore(path string) *CredentialStore {
	return &CredentialStore{path: path, getenv: os.Getenv}
}

// SetGetenv overrides the environment lookup used for config-value
// resolution (tests).
func (s *CredentialStore) SetGetenv(getenv func(string) string) {
	s.getenv = getenv
}

func (s *CredentialStore) loadLocked() error {
	if s.path == "" {
		return nil
	}
	content, err := os.ReadFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		s.data = map[string]*Credential{}
		return nil
	}
	if err != nil {
		return fmt.Errorf("catalog: failed to read auth.json: %w", err)
	}
	data := map[string]*Credential{}
	if len(content) > 0 {
		if err := json.Unmarshal(content, &data); err != nil {
			return fmt.Errorf("catalog: failed to parse auth.json: %w", err)
		}
	}
	s.data = data
	return nil
}

func (s *CredentialStore) writeLocked() error {
	if s.path == "" {
		return nil
	}
	content, err := json.MarshalIndent(s.data, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, content, 0o600); err != nil {
		return fmt.Errorf("catalog: failed to write auth.json: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("catalog: failed to write auth.json: %w", err)
	}
	// Best-effort: tighten permissions on platforms that honor chmod.
	_ = os.Chmod(s.path, 0o600)
	return nil
}

// lockFileTimeout bounds how long lock waits for a sibling process's lock.
const lockFileTimeout = 30 * time.Second
const lockFileRetry = 20 * time.Millisecond

// lock acquires the cross-process lock file. In-memory stores skip it.
func (s *CredentialStore) lock() (func(), error) {
	if s.path == "" {
		return func() {}, nil
	}
	lockPath := s.path + ".lock"
	deadline := time.Now().Add(lockFileTimeout)
	for {
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			fmt.Fprintf(f, "%d\n", os.Getpid())
			f.Close()
			return func() { os.Remove(lockPath) }, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, fmt.Errorf("catalog: failed to acquire auth.json lock: %w", err)
		}
		if time.Now().After(deadline) {
			return nil, errors.New("catalog: timed out acquiring auth.json lock")
		}
		time.Sleep(lockFileRetry)
	}
}

// Read returns the stored credential for providerID, or nil when absent.
// api_key credentials have their Key resolved as a config value ("$ENV_VAR"
// interpolation / "!command" execution) against the credential's env and the
// process environment, mirroring pi's AuthStorage.read.
func (s *CredentialStore) Read(providerID string) (*Credential, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return nil, err
	}
	cred := s.data[providerID].clone()
	if cred == nil || cred.Type != "api_key" || cred.Key == "" {
		return cred, nil
	}
	resolved := *cred
	resolved.Key = ResolveConfigValue(cred.Key, cred.Env, s.getenv)
	return &resolved, nil
}

// List returns metadata of all stored credentials, sorted by provider id.
func (s *CredentialStore) List() ([]CredentialInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return nil, err
	}
	infos := make([]CredentialInfo, 0, len(s.data))
	for providerID, cred := range s.data {
		infos = append(infos, CredentialInfo{ProviderID: providerID, Type: cred.Type})
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].ProviderID < infos[j].ProviderID })
	return infos, nil
}

// Modify is the only write path: fn sees the current credential and returns
// the new one (nil to leave the entry unchanged). Cycles are serialized per
// store and across processes via the lock file. Returns the post-write
// credential.
func (s *CredentialStore) Modify(providerID string, fn func(current *Credential) (*Credential, error)) (*Credential, error) {
	unlock, err := s.lock()
	if err != nil {
		return nil, err
	}
	defer unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return nil, err
	}
	current := s.data[providerID].clone()
	next, err := fn(current)
	if err != nil {
		return nil, err
	}
	if next == nil {
		return current, nil
	}
	s.data[providerID] = next.clone()
	if err := s.writeLocked(); err != nil {
		return nil, err
	}
	return next.clone(), nil
}

// Delete removes the credential for providerID, serialized against Modify.
func (s *CredentialStore) Delete(providerID string) error {
	unlock, err := s.lock()
	if err != nil {
		return err
	}
	defer unlock()
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.loadLocked(); err != nil {
		return err
	}
	delete(s.data, providerID)
	return s.writeLocked()
}
