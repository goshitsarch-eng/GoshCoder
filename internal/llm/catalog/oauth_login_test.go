package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

type loginInteractionStub struct {
	prompt func(LoginPrompt) (string, error)

	mu     sync.Mutex
	events []LoginEvent
}

func (i *loginInteractionStub) Prompt(prompt LoginPrompt) (string, error) {
	if i.prompt == nil {
		return "", errors.New("unexpected prompt")
	}
	return i.prompt(prompt)
}

func (i *loginInteractionStub) Notify(event LoginEvent) {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.events = append(i.events, event)
}

func (i *loginInteractionStub) event(kind string) (LoginEvent, bool) {
	i.mu.Lock()
	defer i.mu.Unlock()
	for _, event := range i.events {
		if event.Kind == kind {
			return event, true
		}
	}
	return LoginEvent{}, false
}

func disableBrowser(t *testing.T) {
	t.Helper()
	previous := browserOpener
	browserOpener = func(string) error { return nil }
	t.Cleanup(func() { browserOpener = previous })
}

func unusedLoopbackPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	return port
}

func TestGeneratePKCEChallengeMatchesVerifier(t *testing.T) {
	pair, err := generatePKCE()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := base64.RawURLEncoding.DecodeString(pair.Verifier)
	if err != nil {
		t.Fatalf("verifier is not base64url: %v", err)
	}
	if len(raw) != 32 {
		t.Fatalf("verifier decodes to %d bytes, want 32", len(raw))
	}
	sum := sha256.Sum256([]byte(pair.Verifier))
	want := base64.RawURLEncoding.EncodeToString(sum[:])
	if pair.Challenge != want {
		t.Fatalf("challenge = %q, want %q", pair.Challenge, want)
	}
}

func TestParseAuthorizationInput(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantCode  string
		wantState string
	}{
		{"redirect URL", "http://localhost/callback?code=url-code&state=url-state", "url-code", "url-state"},
		{"code and hash state", "hash-code#hash-state", "hash-code", "hash-state"},
		{"query fragment", "code=query-code&state=query-state", "query-code", "query-state"},
		{"bare code", "  bare-code  ", "bare-code", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, state := parseAuthorizationInput(tt.input)
			if code != tt.wantCode || state != tt.wantState {
				t.Fatalf("parseAuthorizationInput = (%q, %q), want (%q, %q)", code, state, tt.wantCode, tt.wantState)
			}
		})
	}
}

func TestCallbackServerRejectsStateMismatch(t *testing.T) {
	server, err := startCallbackServer("127.0.0.1", 0, "/callback", "expected")
	if err != nil {
		t.Fatal(err)
	}
	defer server.close()

	response, err := http.Get("http://" + server.addr + "/callback?code=stolen&state=wrong")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(response.Body)
	response.Body.Close()
	if response.StatusCode != http.StatusBadRequest || !strings.Contains(string(body), "State mismatch") {
		t.Fatalf("response = %d %q", response.StatusCode, body)
	}
	select {
	case result := <-server.results:
		t.Fatalf("mismatched callback was accepted: %#v", result)
	default:
	}
}

func TestRunLoopbackLoginManualPromptWins(t *testing.T) {
	disableBrowser(t)
	port := unusedLoopbackPort(t)
	interaction := &loginInteractionStub{prompt: func(prompt LoginPrompt) (string, error) {
		if prompt.Kind != PromptManualCode || prompt.Ctx == nil {
			t.Fatalf("prompt = %#v", prompt)
		}
		return "manual-code#state-1", nil
	}}

	code, err := runLoopbackLogin(t.Context(), interaction, "https://example.com/authorize",
		"http://localhost/callback", "state-1", "127.0.0.1", port, "/callback")
	if err != nil || code != "manual-code" {
		t.Fatalf("runLoopbackLogin = %q, %v", code, err)
	}
	if _, ok := interaction.event("auth_url"); !ok {
		t.Fatal("authorization URL was not announced")
	}
}

func TestRunLoopbackLoginCallbackWins(t *testing.T) {
	disableBrowser(t)
	port := unusedLoopbackPort(t)
	requestErr := make(chan error, 1)
	interaction := &loginInteractionStub{}
	interaction.prompt = func(prompt LoginPrompt) (string, error) {
		<-prompt.Ctx.Done()
		return "", ErrLoginCancelled
	}
	// Notify is synchronous and runs after the callback listener is ready.
	callbackInteraction := &callbackLoginInteraction{
		loginInteractionStub: interaction,
		onNotify: func(event LoginEvent) {
			if event.Kind != "auth_url" {
				return
			}
			go func() {
				response, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/callback?code=callback-code&state=state-2", port))
				if err == nil {
					response.Body.Close()
					if response.StatusCode != http.StatusOK {
						err = fmt.Errorf("callback status %d", response.StatusCode)
					}
				}
				requestErr <- err
			}()
		},
	}

	code, err := runLoopbackLogin(t.Context(), callbackInteraction, "https://example.com/authorize",
		"http://localhost/callback", "state-2", "127.0.0.1", port, "/callback")
	if err != nil || code != "callback-code" {
		t.Fatalf("runLoopbackLogin = %q, %v", code, err)
	}
	if err := <-requestErr; err != nil {
		t.Fatal(err)
	}
}

// callbackLoginInteraction adds an observable side effect to Notify while
// retaining the thread-safe event recording used by the other tests.
type callbackLoginInteraction struct {
	*loginInteractionStub
	onNotify func(LoginEvent)
}

func (i *callbackLoginInteraction) Notify(event LoginEvent) {
	i.loginInteractionStub.Notify(event)
	if i.onNotify != nil {
		i.onNotify(event)
	}
}

func TestRunLoopbackLoginRejectsPastedMismatchedState(t *testing.T) {
	disableBrowser(t)
	interaction := &loginInteractionStub{prompt: func(LoginPrompt) (string, error) {
		return "code#wrong-state", nil
	}}
	_, err := runLoopbackLogin(t.Context(), interaction, "https://example.com/authorize",
		"http://localhost/callback", "right-state", "127.0.0.1", unusedLoopbackPort(t), "/callback")
	if err == nil || !strings.Contains(err.Error(), "state mismatch") {
		t.Fatalf("err = %v, want state mismatch", err)
	}
}

func TestPollDeviceCodePendingThenSucceeds(t *testing.T) {
	polls := 0
	err := pollDeviceCode(t.Context(), 1, 2*time.Second, func() (bool, error) {
		polls++
		return polls == 2, nil
	})
	if err != nil || polls != 2 {
		t.Fatalf("pollDeviceCode: polls = %d, err = %v", polls, err)
	}
}

func TestPollDeviceCodeTimeoutAndCancellation(t *testing.T) {
	t.Run("timeout", func(t *testing.T) {
		err := pollDeviceCode(t.Context(), 1, 20*time.Millisecond, func() (bool, error) { return false, nil })
		if err == nil || !strings.Contains(err.Error(), "timed out") {
			t.Fatalf("err = %v, want timeout", err)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		called := false
		err := pollDeviceCode(ctx, 1, time.Second, func() (bool, error) {
			called = true
			return false, nil
		})
		if !errors.Is(err, ErrLoginCancelled) || called {
			t.Fatalf("err = %v, called = %v", err, called)
		}
	})
}

func TestAnthropicLoginExchangesPKCEAndPersistsCredential(t *testing.T) {
	disableBrowser(t)
	ts := newTokenServer(t, okToken("anthropic-access", "anthropic-refresh", 3600))
	previousTokenURL := anthropicTokenURL
	anthropicTokenURL = ts.server.URL
	t.Cleanup(func() { anthropicTokenURL = previousTokenURL })

	interaction := &loginInteractionStub{prompt: func(prompt LoginPrompt) (string, error) {
		if prompt.Kind != PromptManualCode {
			return "", fmt.Errorf("unexpected prompt kind %q", prompt.Kind)
		}
		return "authorization-code", nil
	}}
	c, store := oauthCatalog(t, nil)
	cred, err := c.Login(t.Context(), "anthropic", interaction)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if cred.Access != "anthropic-access" || cred.Refresh != "anthropic-refresh" {
		t.Fatalf("credential = %#v", cred)
	}

	authEvent, ok := interaction.event("auth_url")
	if !ok {
		t.Fatal("missing auth_url event")
	}
	parsedURL, err := url.Parse(authEvent.URL)
	if err != nil {
		t.Fatal(err)
	}
	query := parsedURL.Query()
	verifier := query.Get("state")
	sum := sha256.Sum256([]byte(verifier))
	if query.Get("code_challenge") != base64.RawURLEncoding.EncodeToString(sum[:]) {
		t.Fatal("authorization URL does not carry the verifier's S256 challenge")
	}
	request := ts.request(0)
	if request["code"] != "authorization-code" || request["state"] != verifier || request["code_verifier"] != verifier {
		t.Fatalf("token request = %#v", request)
	}
	stored, err := store.Read("anthropic")
	if err != nil || stored == nil || stored.Access != "anthropic-access" || stored.Type != "oauth" {
		t.Fatalf("stored credential = %#v, err = %v", stored, err)
	}
}

func TestCodexExchangeCodeStoresAccountID(t *testing.T) {
	access := makeJWT("acct-login")
	ts := newTokenServer(t, okToken(access, "refresh", 3600))
	previousTokenURL := codexTokenURL
	codexTokenURL = ts.server.URL
	t.Cleanup(func() { codexTokenURL = previousTokenURL })

	cred, err := (openAICodexOAuth{}).exchangeCode(t.Context(), "auth-code", "verifier", "http://redirect")
	if err != nil {
		t.Fatal(err)
	}
	accountID, ok := cred.Extra("accountId")
	if !ok || accountID != "acct-login" {
		t.Fatalf("accountId = %q, ok = %v", accountID, ok)
	}
	request := ts.request(0)
	if request["grant_type"] != "authorization_code" || request["code"] != "auth-code" || request["code_verifier"] != "verifier" {
		t.Fatalf("token request = %#v", request)
	}
}

func TestKimiLoginUsesDeviceFlow(t *testing.T) {
	ts := newTokenServer(t,
		tokenServerResponse{status: http.StatusOK, body: `{"device_code":"device-1","user_code":"ABCD","verification_uri":"https://auth.example/device","verification_uri_complete":"https://auth.example/device?code=ABCD","interval":0.01,"expires_in":10}`},
		okToken("kimi-access", "kimi-refresh", 3600),
	)
	interaction := &loginInteractionStub{prompt: func(LoginPrompt) (string, error) {
		return "", errors.New("Kimi device login must not prompt on stdin")
	}}
	cred, err := (kimiCodingOAuth{}).Login(t.Context(), interaction, map[string]string{"KIMI_CODE_OAUTH_HOST": ts.server.URL})
	if err != nil {
		t.Fatal(err)
	}
	if cred.Access != "kimi-access" || cred.Refresh != "kimi-refresh" {
		t.Fatalf("credential = %#v", cred)
	}
	event, ok := interaction.event("device_code")
	if !ok || event.UserCode != "ABCD" || !strings.Contains(event.VerificationURI, "ABCD") {
		t.Fatalf("device event = %#v, ok = %v", event, ok)
	}
	if ts.requestCount() != 2 {
		t.Fatalf("requests = %d, want authorization and token", ts.requestCount())
	}
	poll := ts.request(1)
	if poll["device_code"] != "device-1" || poll["grant_type"] != kimiDeviceGrantType {
		t.Fatalf("poll request = %#v", poll)
	}
}

func TestKimiLoginRejectsUntrustedVerificationURL(t *testing.T) {
	ts := newTokenServer(t, tokenServerResponse{status: http.StatusOK, body: `{"device_code":"d","user_code":"u","verification_uri":"javascript:alert(1)","verification_uri_complete":"https://example.com"}`})
	_, err := (kimiCodingOAuth{}).Login(t.Context(), &loginInteractionStub{}, map[string]string{"KIMI_CODE_OAUTH_HOST": ts.server.URL})
	if err == nil || !strings.Contains(err.Error(), "invalid Kimi") {
		t.Fatalf("err = %v", err)
	}
}

func TestLoginProviderRegistry(t *testing.T) {
	for _, id := range []string{"anthropic", "kimi-coding", "openai-codex"} {
		if _, ok := LoginProviderFor(id); !ok {
			t.Errorf("%s should have a login flow", id)
		}
	}
}
