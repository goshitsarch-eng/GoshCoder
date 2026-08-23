package catalog

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// tokenServer is a fake OAuth token endpoint.
type tokenServer struct {
	server *httptest.Server

	mu       sync.Mutex
	requests []map[string]string
	// responses are replayed in order; the last one repeats.
	responses []tokenServerResponse
}

type tokenServerResponse struct {
	status int
	body   string
}

func newTokenServer(t *testing.T, responses ...tokenServerResponse) *tokenServer {
	t.Helper()
	ts := &tokenServer{responses: responses}
	ts.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Both form and JSON bodies are recorded as flat maps so assertions
		// stay uniform across providers.
		fields := map[string]string{}
		if strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
			var payload map[string]any
			_ = json.NewDecoder(r.Body).Decode(&payload)
			for key, value := range payload {
				fields[key] = fmt.Sprint(value)
			}
		} else {
			_ = r.ParseForm()
			for key := range r.PostForm {
				fields[key] = r.PostForm.Get(key)
			}
		}

		ts.mu.Lock()
		index := len(ts.requests)
		ts.requests = append(ts.requests, fields)
		if index >= len(ts.responses) {
			index = len(ts.responses) - 1
		}
		response := ts.responses[index]
		ts.mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.status)
		fmt.Fprint(w, response.body)
	}))
	t.Cleanup(ts.server.Close)
	return ts
}

func (ts *tokenServer) requestCount() int {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	return len(ts.requests)
}

func (ts *tokenServer) request(i int) map[string]string {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	return ts.requests[i]
}

// okToken is a successful token response body.
func okToken(access, refresh string, expiresIn int) tokenServerResponse {
	return tokenServerResponse{
		status: http.StatusOK,
		body: fmt.Sprintf(`{"access_token":%q,"refresh_token":%q,"expires_in":%d}`,
			access, refresh, expiresIn),
	}
}

// makeJWT builds an unsigned JWT carrying the Codex account-id claim. Only the
// payload segment is read, so the header and signature are placeholders.
func makeJWT(accountID string) string {
	payload := map[string]any{
		codexJWTClaimPath: map[string]any{"chatgpt_account_id": accountID},
	}
	encoded, _ := json.Marshal(payload)
	return "header." + base64.RawURLEncoding.EncodeToString(encoded) + ".signature"
}

// oauthCatalog builds a catalog with a file-backed store and a fake env.
func oauthCatalog(t *testing.T, env map[string]string) (*Catalog, *CredentialStore) {
	t.Helper()
	store := NewFileCredentialStore(filepath.Join(t.TempDir(), "auth.json"))
	c := NewCatalog(store)
	c.SetGetenv(func(name string) string { return env[name] })
	c.SetFileExists(func(string) bool { return false })
	return c, store
}

// storeOAuth writes an OAuth credential expiring in the given duration.
func storeOAuth(t *testing.T, store *CredentialStore, providerID, access, refresh string, expiresIn time.Duration) {
	t.Helper()
	if _, err := store.Modify(providerID, func(*Credential) (*Credential, error) {
		return &Credential{
			Type:    "oauth",
			Access:  access,
			Refresh: refresh,
			Expires: time.Now().Add(expiresIn).UnixMilli(),
		}, nil
	}); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

func TestExpiresSoon(t *testing.T) {
	tests := []struct {
		name      string
		expiresIn time.Duration
		want      bool
	}{
		{"comfortably valid", time.Hour, false},
		// The refresh window is five minutes, so anything inside it is stale.
		{"inside the refresh window", 2 * time.Minute, true},
		{"already expired", -time.Hour, true},
		{"just outside the window", oauthMinimumValidity + time.Minute, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cred := &Credential{Type: "oauth", Expires: time.Now().Add(tt.expiresIn).UnixMilli()}
			if got := expiresSoon(cred); got != tt.want {
				t.Fatalf("expiresSoon = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCredentialFrom(t *testing.T) {
	parsed := &tokenResponse{AccessToken: "a", RefreshToken: "r", ExpiresIn: 3600}
	cred, err := credentialFrom(parsed, 0)
	if err != nil {
		t.Fatal(err)
	}
	if cred.Type != "oauth" || cred.Access != "a" || cred.Refresh != "r" {
		t.Fatalf("credential = %#v", cred)
	}
	// The expiry is roughly an hour out.
	remaining := time.Until(time.UnixMilli(cred.Expires))
	if remaining < 59*time.Minute || remaining > time.Hour+time.Minute {
		t.Fatalf("expiry = %v out", remaining)
	}

	// Skew shortens the lifetime so the token is refreshed early.
	skewed, err := credentialFrom(parsed, 5*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if skewed.Expires >= cred.Expires {
		t.Fatal("skew was not applied")
	}
}

func TestCredentialFromRejectsIncompleteResponses(t *testing.T) {
	tests := []struct {
		name   string
		parsed *tokenResponse
	}{
		{"no access token", &tokenResponse{RefreshToken: "r", ExpiresIn: 60}},
		{"no refresh token", &tokenResponse{AccessToken: "a", ExpiresIn: 60}},
		{"no expiry", &tokenResponse{AccessToken: "a", RefreshToken: "r"}},
		{"negative expiry", &tokenResponse{AccessToken: "a", RefreshToken: "r", ExpiresIn: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := credentialFrom(tt.parsed, 0); err == nil {
				t.Fatal("an incomplete token response must be rejected")
			}
		})
	}
}

func TestIsUnauthorized(t *testing.T) {
	if !isUnauthorized(http.StatusUnauthorized, nil) {
		t.Error("401 is unauthorized")
	}
	if !isUnauthorized(http.StatusForbidden, nil) {
		t.Error("403 is unauthorized")
	}
	// invalid_grant is terminal regardless of status.
	if !isUnauthorized(http.StatusBadRequest, &tokenResponse{Error: "invalid_grant"}) {
		t.Error("invalid_grant is unauthorized")
	}
	if isUnauthorized(http.StatusInternalServerError, nil) {
		t.Error("a server error is not a dead credential")
	}
}

func TestIsRetryableStatus(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, 500, 503} {
		if !isRetryableStatus(status) {
			t.Errorf("%d should be retryable", status)
		}
	}
	for _, status := range []int{400, 401, 403, 404} {
		if isRetryableStatus(status) {
			t.Errorf("%d should not be retryable", status)
		}
	}
}

func TestDecodeJWTPayload(t *testing.T) {
	payload, err := decodeJWTPayload(makeJWT("acct-123"))
	if err != nil {
		t.Fatal(err)
	}
	claim := payload[codexJWTClaimPath].(map[string]any)
	if claim["chatgpt_account_id"] != "acct-123" {
		t.Fatalf("claim = %#v", claim)
	}

	for _, invalid := range []string{"", "not-a-jwt", "only.two", "a.!!!not-base64!!!.c"} {
		if _, err := decodeJWTPayload(invalid); err == nil {
			t.Errorf("decodeJWTPayload(%q) should fail", invalid)
		}
	}
}

func TestCodexAccountID(t *testing.T) {
	got, err := codexAccountID(makeJWT("acct-abc"))
	if err != nil || got != "acct-abc" {
		t.Fatalf("codexAccountID = %q, err = %v", got, err)
	}

	// A well-formed JWT without the claim is still an error: the wire protocol
	// needs the account id.
	empty, _ := json.Marshal(map[string]any{"sub": "x"})
	noClaim := "h." + base64.RawURLEncoding.EncodeToString(empty) + ".s"
	if _, err := codexAccountID(noClaim); err == nil {
		t.Fatal("a token without the auth claim must fail")
	}
}

func TestOAuthProviderRegistry(t *testing.T) {
	ids := OAuthProviderIDs()
	// The three providers ported so far, sorted.
	want := []string{"anthropic", "kimi-coding", "openai-codex"}
	if strings.Join(ids, ",") != strings.Join(want, ",") {
		t.Fatalf("OAuthProviderIDs = %v, want %v", ids, want)
	}
	for _, id := range want {
		provider, ok := OAuthProviderFor(id)
		if !ok || provider.Name() == "" {
			t.Fatalf("OAuthProviderFor(%q) = %#v, ok = %v", id, provider, ok)
		}
	}
	if _, ok := OAuthProviderFor("github-copilot"); ok {
		t.Fatal("github-copilot has no ported OAuth flow")
	}
}

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

func TestAnthropicRefresh(t *testing.T) {
	ts := newTokenServer(t, okToken("new-access", "new-refresh", 3600))
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	cred, err := anthropicOAuth{}.Refresh(t.Context(),
		&Credential{Type: "oauth", Refresh: "old-refresh"}, nil)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if cred.Access != "new-access" || cred.Refresh != "new-refresh" {
		t.Fatalf("credential = %#v", cred)
	}
	// Anthropic subtracts a five-minute skew, so the stored expiry is early.
	remaining := time.Until(time.UnixMilli(cred.Expires))
	if remaining > 56*time.Minute {
		t.Fatalf("expiry = %v out, want the refresh skew applied", remaining)
	}

	request := ts.request(0)
	if request["grant_type"] != "refresh_token" || request["refresh_token"] != "old-refresh" {
		t.Fatalf("request = %#v", request)
	}
	if request["client_id"] != anthropicOAuthClientID {
		t.Fatalf("client_id = %q", request["client_id"])
	}
}

func TestAnthropicRefreshUnauthorizedIsTerminal(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{
		status: http.StatusUnauthorized,
		body:   `{"error":"invalid_grant","error_description":"refresh token revoked"}`,
	})
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	_, err := anthropicOAuth{}.Refresh(t.Context(), &Credential{Type: "oauth", Refresh: "dead"}, nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	// Callers key off this to tell the user to log in again.
	if !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("err = %v, want ErrOAuthUnauthorized", err)
	}
	if !strings.Contains(err.Error(), "refresh token revoked") {
		t.Fatalf("err = %v, want the provider detail preserved", err)
	}
}

func TestAnthropicToAuth(t *testing.T) {
	auth := anthropicOAuth{}.ToAuth(&Credential{Type: "oauth", Access: "tok"})
	if auth.APIKey != "tok" || auth.Source != "OAuth" {
		t.Fatalf("auth = %#v", auth)
	}
	// Anthropic sends the token as the API key, so no headers are needed.
	if auth.Headers != nil {
		t.Fatalf("headers = %#v, want none", auth.Headers)
	}
}

// ---------------------------------------------------------------------------
// Kimi Code
// ---------------------------------------------------------------------------

func TestKimiOAuthHostOverrides(t *testing.T) {
	if got := kimiOAuthHost(nil); got != kimiDefaultOAuthHost {
		t.Fatalf("default host = %q", got)
	}
	if got := kimiOAuthHost(map[string]string{"KIMI_OAUTH_HOST": "https://alt.example.com/"}); got != "https://alt.example.com" {
		t.Fatalf("override host = %q, want the trailing slash trimmed", got)
	}
	// The code-specific variable wins.
	env := map[string]string{
		"KIMI_CODE_OAUTH_HOST": "https://code.example.com",
		"KIMI_OAUTH_HOST":      "https://generic.example.com",
	}
	if got := kimiOAuthHost(env); got != "https://code.example.com" {
		t.Fatalf("host = %q", got)
	}
}

func TestKimiRefresh(t *testing.T) {
	ts := newTokenServer(t, okToken("kimi-access", "kimi-refresh", 7200))
	env := map[string]string{"KIMI_CODE_OAUTH_HOST": ts.server.URL}

	cred, err := kimiCodingOAuth{}.Refresh(t.Context(),
		&Credential{Type: "oauth", Refresh: "old"}, env)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if cred.Access != "kimi-access" || cred.Refresh != "kimi-refresh" {
		t.Fatalf("credential = %#v", cred)
	}
	request := ts.request(0)
	if request["client_id"] != kimiOAuthClientID || request["grant_type"] != "refresh_token" {
		t.Fatalf("request = %#v", request)
	}
}

// TestKimiRefreshRetriesTransientFailures covers the backoff: a 500 is retried,
// unlike a 401.
func TestKimiRefreshRetriesTransientFailures(t *testing.T) {
	ts := newTokenServer(t,
		tokenServerResponse{status: http.StatusInternalServerError, body: `{}`},
		okToken("after-retry", "r", 3600),
	)
	env := map[string]string{"KIMI_CODE_OAUTH_HOST": ts.server.URL}

	cred, err := kimiCodingOAuth{}.Refresh(t.Context(), &Credential{Type: "oauth", Refresh: "old"}, env)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if cred.Access != "after-retry" {
		t.Fatalf("access = %q", cred.Access)
	}
	if ts.requestCount() != 2 {
		t.Fatalf("requests = %d, want a retry", ts.requestCount())
	}
}

func TestKimiRefreshUnauthorizedIsTerminal(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{
		status: http.StatusUnauthorized,
		body:   `{"error":"invalid_grant"}`,
	})
	env := map[string]string{"KIMI_CODE_OAUTH_HOST": ts.server.URL}

	_, err := kimiCodingOAuth{}.Refresh(t.Context(), &Credential{Type: "oauth", Refresh: "dead"}, env)
	if !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("err = %v, want ErrOAuthUnauthorized", err)
	}
	// A dead credential is not retried.
	if ts.requestCount() != 1 {
		t.Fatalf("requests = %d, want no retry on 401", ts.requestCount())
	}
}

// TestKimiRefreshHonorsCancellation confirms the backoff sleep is cancellable.
func TestKimiRefreshHonorsCancellation(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{status: http.StatusInternalServerError, body: `{}`})
	env := map[string]string{"KIMI_CODE_OAUTH_HOST": ts.server.URL}

	ctx, cancel := context.WithCancel(t.Context())
	// Cancel while the first backoff is pending.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	refresher := kimiCodingOAuth{}
	if _, err := refresher.Refresh(ctx, &Credential{Type: "oauth", Refresh: "x"}, env); err == nil {
		t.Fatal("expected an error")
	}
	// Without cancellation the retries would take seconds.
	if elapsed := time.Since(start); elapsed > 2*time.Second {
		t.Fatalf("refresh took %v; cancellation was not honored", elapsed)
	}
}

// TestKimiToAuthUsesAuthorizationHeader is the shape that differs from
// Anthropic: the Kimi coding endpoint wants a bearer header.
func TestKimiToAuthUsesAuthorizationHeader(t *testing.T) {
	auth := kimiCodingOAuth{}.ToAuth(&Credential{Type: "oauth", Access: "tok"})
	header, present := auth.Headers["Authorization"]
	if !present || header == nil || *header != "Bearer tok" {
		t.Fatalf("headers = %#v", auth.Headers)
	}
}

// ---------------------------------------------------------------------------
// OpenAI Codex
// ---------------------------------------------------------------------------

func TestCodexRefreshStoresAccountID(t *testing.T) {
	access := makeJWT("acct-codex")
	ts := newTokenServer(t, okToken(access, "codex-refresh", 3600))
	codexTokenURL = ts.server.URL
	t.Cleanup(func() { codexTokenURL = codexAuthBaseURL + "/oauth/token" })

	cred, err := openAICodexOAuth{}.Refresh(t.Context(),
		&Credential{Type: "oauth", Refresh: "old"}, nil)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	// The account id travels with the credential; the wire protocol sends it
	// as a header.
	accountID, ok := cred.Extra("accountId")
	if !ok || accountID != "acct-codex" {
		t.Fatalf("accountId = %q, ok = %v", accountID, ok)
	}
}

// TestCodexRefreshRejectsTokenWithoutAccountID confirms a token that cannot
// yield an account id fails rather than being stored unusable.
func TestCodexRefreshRejectsTokenWithoutAccountID(t *testing.T) {
	ts := newTokenServer(t, okToken("not-a-jwt", "r", 3600))
	codexTokenURL = ts.server.URL
	t.Cleanup(func() { codexTokenURL = codexAuthBaseURL + "/oauth/token" })

	refresher := openAICodexOAuth{}
	if _, err := refresher.Refresh(t.Context(), &Credential{Type: "oauth", Refresh: "old"}, nil); err == nil {
		t.Fatal("a token without an account id must be rejected")
	}
}

func TestCodexRefreshUnauthorizedIsTerminal(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{status: http.StatusForbidden, body: `{"error":"invalid_grant"}`})
	codexTokenURL = ts.server.URL
	t.Cleanup(func() { codexTokenURL = codexAuthBaseURL + "/oauth/token" })

	_, err := openAICodexOAuth{}.Refresh(t.Context(), &Credential{Type: "oauth", Refresh: "dead"}, nil)
	if !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("err = %v, want ErrOAuthUnauthorized", err)
	}
}

// ---------------------------------------------------------------------------
// Credential extras
// ---------------------------------------------------------------------------

func TestCredentialStoreRecoversStaleLock(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	lockPath := path + ".lock"
	if err := os.WriteFile(lockPath, []byte("stale\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := time.Now().Add(-2 * defaultLockTiming().stale)
	if err := os.Chtimes(lockPath, old, old); err != nil {
		t.Fatal(err)
	}
	store := NewFileCredentialStore(path)
	if _, err := store.Modify("test", func(*Credential) (*Credential, error) {
		return &Credential{Type: "api_key", Key: "secret"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Fatalf("lock remains: %v", err)
	}
}

func TestCredentialExtraRoundTripsThroughAuthJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "auth.json")
	store := NewFileCredentialStore(path)

	if _, err := store.Modify("openai-codex", func(*Credential) (*Credential, error) {
		cred := &Credential{Type: "oauth", Access: "a", Refresh: "r", Expires: time.Now().Add(time.Hour).UnixMilli()}
		if err := cred.SetExtra("accountId", "acct-1"); err != nil {
			return nil, err
		}
		return cred, nil
	}); err != nil {
		t.Fatal(err)
	}

	// The extra survives a full write/read cycle, which is what keeps the file
	// interoperable with pi.
	reread := NewFileCredentialStore(path)
	cred, err := reread.Read("openai-codex")
	if err != nil {
		t.Fatal(err)
	}
	accountID, ok := cred.Extra("accountId")
	if !ok || accountID != "acct-1" {
		t.Fatalf("accountId = %q, ok = %v", accountID, ok)
	}

	// It is written under its own key, not folded into a known field.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"accountId"`) {
		t.Fatalf("auth.json = %s", raw)
	}
}

func TestCredentialSetExtraRejectsKnownFields(t *testing.T) {
	cred := &Credential{Type: "oauth"}
	if err := cred.SetExtra("access", "x"); err == nil {
		t.Fatal("a known field must not be settable as an extra")
	}
}

func TestCredentialExtraMissing(t *testing.T) {
	cred := &Credential{Type: "oauth"}
	if _, ok := cred.Extra("accountId"); ok {
		t.Fatal("an absent extra should report false")
	}
	var nilCred *Credential
	if _, ok := nilCred.Extra("accountId"); ok {
		t.Fatal("a nil credential should report false")
	}
}

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

// TestResolveOAuthValidTokenSkipsNetwork confirms a healthy credential resolves
// without a token request.
func TestResolveOAuthValidTokenSkipsNetwork(t *testing.T) {
	ts := newTokenServer(t, okToken("should-not-be-used", "r", 3600))
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	c, store := oauthCatalog(t, nil)
	storeOAuth(t, store, "anthropic", "live-token", "refresh", time.Hour)

	auth, ok := c.ResolveAuth("anthropic")
	if !ok || auth.APIKey != "live-token" {
		t.Fatalf("auth = %#v, ok = %v", auth, ok)
	}
	if ts.requestCount() != 0 {
		t.Fatalf("a valid token must not trigger a refresh (%d requests)", ts.requestCount())
	}
}

// TestResolveOAuthRefreshesExpiredToken is the behavior this step exists for:
// an expired credential is refreshed and persisted, and the request proceeds.
func TestResolveOAuthRefreshesExpiredToken(t *testing.T) {
	ts := newTokenServer(t, okToken("refreshed-token", "rotated-refresh", 3600))
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	c, store := oauthCatalog(t, nil)
	storeOAuth(t, store, "anthropic", "stale-token", "old-refresh", -time.Hour)

	auth, ok := c.ResolveAuth("anthropic")
	if !ok {
		t.Fatalf("an expired credential should refresh; error: %v", c.OAuthError("anthropic"))
	}
	if auth.APIKey != "refreshed-token" {
		t.Fatalf("apiKey = %q", auth.APIKey)
	}

	// The rotated credential is persisted, so the next process reuses it.
	stored, err := store.Read("anthropic")
	if err != nil {
		t.Fatal(err)
	}
	if stored.Access != "refreshed-token" || stored.Refresh != "rotated-refresh" {
		t.Fatalf("stored credential = %#v", stored)
	}
	if expiresSoon(stored) {
		t.Fatal("the refreshed credential should not be expiring")
	}
}

// TestResolveOAuthFailedRefreshDoesNotFallBackToEnv is the security-relevant
// case: a stored credential owns the provider, so a dead token must not
// silently fall through to an ambient API key.
func TestResolveOAuthFailedRefreshDoesNotFallBackToEnv(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{
		status: http.StatusUnauthorized,
		body:   `{"error":"invalid_grant"}`,
	})
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	c, store := oauthCatalog(t, map[string]string{"ANTHROPIC_API_KEY": "sk-ambient"})
	storeOAuth(t, store, "anthropic", "stale", "dead-refresh", -time.Hour)

	if auth, ok := c.ResolveAuth("anthropic"); ok {
		t.Fatalf("auth = %#v, want no credential rather than the ambient key", auth)
	}
	// The reason is recorded so the CLI can tell the user to log in again.
	err := c.OAuthError("anthropic")
	if err == nil || !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("OAuthError = %v, want ErrOAuthUnauthorized", err)
	}
}

// TestResolveOAuthTransientFailureKeepsCredential confirms a network blip does
// not destroy a recoverable token.
func TestResolveOAuthTransientFailureKeepsCredential(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{status: http.StatusInternalServerError, body: `{}`})
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	c, store := oauthCatalog(t, nil)
	storeOAuth(t, store, "anthropic", "stale", "still-good-refresh", -time.Hour)

	if _, ok := c.ResolveAuth("anthropic"); ok {
		t.Fatal("a failed refresh should not resolve")
	}
	// The refresh token is still on disk for the next attempt.
	stored, err := store.Read("anthropic")
	if err != nil {
		t.Fatal(err)
	}
	if stored == nil || stored.Refresh != "still-good-refresh" {
		t.Fatalf("stored credential = %#v, want it preserved", stored)
	}
}

// TestResolveOAuthKimiAppliesHeaders confirms the provider-specific ToAuth
// shape survives resolution.
func TestResolveOAuthKimiAppliesHeaders(t *testing.T) {
	c, store := oauthCatalog(t, nil)
	storeOAuth(t, store, "kimi-coding", "kimi-token", "r", time.Hour)

	auth, ok := c.ResolveAuth("kimi-coding")
	if !ok {
		t.Fatal("expected the stored credential to resolve")
	}
	header, present := auth.Headers["Authorization"]
	if !present || header == nil || *header != "Bearer kimi-token" {
		t.Fatalf("headers = %#v", auth.Headers)
	}
}

// TestResolveOAuthUnportedProviderStillUsesValidToken covers a provider with no
// refresh implementation: an unexpired token is still usable.
func TestResolveOAuthUnportedProviderStillUsesValidToken(t *testing.T) {
	c, store := oauthCatalog(t, nil)
	storeOAuth(t, store, "github-copilot", "copilot-token", "r", time.Hour)

	auth, ok := c.ResolveAuth("github-copilot")
	if !ok || auth.APIKey != "copilot-token" {
		t.Fatalf("auth = %#v, ok = %v", auth, ok)
	}

	// But an expired one cannot be refreshed.
	storeOAuth(t, store, "github-copilot", "expired", "r", -time.Hour)
	if _, ok := c.ResolveAuth("github-copilot"); ok {
		t.Fatal("an expired credential with no refresher must not resolve")
	}
}

// TestRefreshSkippedWhenAnotherProcessAlreadyRotated covers the double-checked
// locking: if the stored credential is healthy by the time the lock is held,
// the refresh is skipped and the fresh token is used.
func TestRefreshSkippedWhenAnotherProcessAlreadyRotated(t *testing.T) {
	ts := newTokenServer(t, okToken("unexpected", "r", 3600))
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = "https://platform.claude.com/v1/oauth/token" })

	c, store := oauthCatalog(t, nil)
	// Healthy on disk, but the caller passes a stale snapshot, which is exactly
	// what a concurrent rotation looks like.
	storeOAuth(t, store, "anthropic", "already-rotated", "r", time.Hour)
	stale := &Credential{Type: "oauth", Access: "stale", Refresh: "r", Expires: time.Now().Add(-time.Hour).UnixMilli()}

	auth, ok := c.resolveOAuth("anthropic", stale)
	if !ok {
		t.Fatalf("expected resolution; error: %v", c.OAuthError("anthropic"))
	}
	if auth.APIKey != "already-rotated" {
		t.Fatalf("apiKey = %q, want the credential another process stored", auth.APIKey)
	}
	if ts.requestCount() != 0 {
		t.Fatalf("no token request should be made (%d made)", ts.requestCount())
	}
}

// TestRefreshWithoutStoreFails confirms a catalog with no credential store
// cannot refresh.
func TestRefreshWithoutStoreFails(t *testing.T) {
	c := NewCatalog(nil)
	c.SetGetenv(func(string) string { return "" })

	provider, _ := OAuthProviderFor("anthropic")
	if _, err := c.refreshOAuthCredential(t.Context(), "anthropic", provider); err == nil {
		t.Fatal("refreshing without a store must fail")
	}
}

func TestSetOAuthContextIsHonored(t *testing.T) {
	// A cancelled context stops the refresh before it reaches the network.
	c, store := oauthCatalog(t, nil)
	storeOAuth(t, store, "anthropic", "stale", "r", -time.Hour)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	c.SetOAuthContext(ctx)

	if _, ok := c.ResolveAuth("anthropic"); ok {
		t.Fatal("a cancelled context should not resolve")
	}
	if err := c.OAuthError("anthropic"); err == nil {
		t.Fatal("the cancellation should be recorded")
	}
}

func TestCredentialsAccessor(t *testing.T) {
	store := NewInMemoryCredentialStore()
	c := NewCatalog(store)
	if c.Credentials() != store {
		t.Fatal("Credentials() should return the backing store")
	}
}

// TestImplicitRefreshIsBoundedAndNotRetriedForever covers the paths that merely
// enumerate providers -- the startup model default, the model and login
// pickers, `goshcoder providers`. Each resolves every provider, and resolution
// can trigger a token refresh; with no deadline that ran on
// context.Background() against a hanging auth host and blocked the caller for
// minutes with nothing on screen, then did it again on the next rebuild.
func TestImplicitRefreshIsBoundedAndNotRetriedForever(t *testing.T) {
	// An auth host that accepts the connection and never answers. The handler
	// waits on a channel the test controls rather than on the request context,
	// so shutting the server down cannot deadlock on an in-flight handler.
	var attempts atomic.Int32
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseHandlers := func() { releaseOnce.Do(func() { close(release) }) }
	defer releaseHandlers()

	hang := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		<-release
	}))
	defer hang.Close()

	restore := anthropicTokenURL
	anthropicTokenURL = hang.URL
	t.Cleanup(func() { anthropicTokenURL = restore })

	path := filepath.Join(t.TempDir(), "auth.json")
	store := NewFileCredentialStore(path)
	if _, err := store.Modify("anthropic", func(*Credential) (*Credential, error) {
		return &Credential{
			Type:    "oauth",
			Access:  "expired",
			Refresh: "refresh-token",
			Expires: time.Now().Add(-time.Hour).UnixMilli(),
		}, nil
	}); err != nil {
		t.Fatal(err)
	}

	c := NewCatalog(store)
	c.SetOAuthTimeout(300 * time.Millisecond)

	start := time.Now()
	if _, ok := c.ResolveAuth("anthropic"); ok {
		t.Fatal("a hanging refresh resolved successfully")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("an implicit refresh blocked for %v; the cap did not apply", elapsed)
	}
	if c.OAuthError("anthropic") == nil {
		t.Fatal("the failure was not recorded, so the CLI cannot say the login expired")
	}

	// A picker rebuild must not re-spend the whole budget against a host
	// already known to be failing.
	before := attempts.Load()
	second := time.Now()
	if _, ok := c.ResolveAuth("anthropic"); ok {
		t.Fatal("resolved after a recorded failure")
	}
	if took := time.Since(second); took > 100*time.Millisecond {
		t.Fatalf("the second resolution took %v; the failure was not remembered", took)
	}
	if attempts.Load() != before {
		t.Fatal("the auth host was contacted again after a recorded failure")
	}

	releaseHandlers()
}
