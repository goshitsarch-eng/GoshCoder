package catalog

// OAuth login flows.
//
// Port of the login half of the OAuthAuth contract in
// reference/pi/packages/ai/src/auth/types.ts, plus the PKCE, loopback-callback,
// and device-code machinery in auth/oauth/*.ts.
//
// Two mechanisms cover the supported providers:
//
//   - PKCE + loopback redirect (Anthropic, OpenAI Codex): open a browser at the
//     authorize URL, serve 127.0.0.1 for the callback, exchange the code and
//     verifier for tokens. A manual-code prompt races the callback so a headless
//     or remote session can paste the redirect URL by hand.
//   - Device code (OpenAI Codex headless): request a user code, display it, poll
//     until the user authorizes.
//
// Refresh and request-auth derivation live in oauth.go.

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// ErrLoginCancelled reports that the user or caller abandoned a login.
var ErrLoginCancelled = errors.New("login cancelled")

// LoginPromptKind identifies what a flow is asking for.
type LoginPromptKind = string

const (
	// PromptText asks for a visible line of input.
	PromptText LoginPromptKind = "text"
	// PromptSecret asks for input that should not be echoed.
	PromptSecret LoginPromptKind = "secret"
	// PromptSelect asks the user to choose one of Options.
	PromptSelect LoginPromptKind = "select"
	// PromptManualCode asks for a pasted code or redirect URL. It races an
	// out-of-band event (the callback server), so the implementation must
	// return ErrLoginCancelled when Ctx is done.
	PromptManualCode LoginPromptKind = "manual_code"
)

// LoginPromptOption is one choice in a select prompt.
type LoginPromptOption struct {
	ID          string
	Label       string
	Description string
}

// LoginPrompt is a request for user input.
type LoginPrompt struct {
	Kind        LoginPromptKind
	Message     string
	Placeholder string
	Options     []LoginPromptOption
	// Ctx cancels this prompt alone, which is how a manual-code prompt is
	// abandoned when the callback server wins the race.
	Ctx context.Context
}

// LoginEvent is progress information for the user.
type LoginEvent struct {
	// Kind is "info", "auth_url", "device_code", or "progress".
	Kind string
	// Message carries info and progress text.
	Message string
	// URL is the authorization URL for "auth_url".
	URL string
	// Instructions accompany an auth URL.
	Instructions string
	// UserCode and VerificationURI are set for "device_code".
	UserCode         string
	VerificationURI  string
	IntervalSeconds  int
	ExpiresInSeconds int
}

// LoginInteraction is how a flow talks to the user. The CLI implements it over
// stdin and stderr.
type LoginInteraction interface {
	// Prompt asks for input, returning ErrLoginCancelled when abandoned. A
	// select prompt returns the chosen option id.
	Prompt(prompt LoginPrompt) (string, error)
	// Notify reports progress. It must not block.
	Notify(event LoginEvent)
}

// OAuthLoginProvider is implemented by providers with a ported login flow.
type OAuthLoginProvider interface {
	OAuthProvider
	// Login runs the interactive flow and returns a credential to store.
	Login(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error)
}

// LoginProviderFor returns the login flow for a provider id.
func LoginProviderFor(providerID string) (OAuthLoginProvider, bool) {
	provider, ok := oauthProviders[providerID]
	if !ok {
		return nil, false
	}
	login, ok := provider.(OAuthLoginProvider)
	return login, ok
}

// Login runs a provider's OAuth login and persists the credential.
func (c *Catalog) Login(ctx context.Context, providerID string, interaction LoginInteraction) (*Credential, error) {
	provider, ok := LoginProviderFor(providerID)
	if !ok {
		return nil, fmt.Errorf("catalog: no OAuth login flow for provider %q", providerID)
	}
	if c.credentials == nil {
		return nil, fmt.Errorf("catalog: no credential store to save the login into")
	}

	cred, err := provider.Login(ctx, interaction, c.envMap())
	if err != nil {
		return nil, err
	}
	cred.Type = "oauth"

	// Overwrite whatever was stored: a fresh login supersedes an expired token
	// or a stale API key.
	stored, err := c.credentials.Modify(providerID, func(*Credential) (*Credential, error) {
		return cred, nil
	})
	if err != nil {
		return nil, fmt.Errorf("catalog: saving the credential failed: %w", err)
	}
	// A successful login clears any recorded refresh failure.
	c.recordOAuthError(providerID, nil)
	return stored, nil
}

// ---------------------------------------------------------------------------
// PKCE
// ---------------------------------------------------------------------------

// pkcePair is a PKCE verifier and its S256 challenge.
type pkcePair struct {
	Verifier  string
	Challenge string
}

// generatePKCE produces a 32-byte random verifier and its SHA-256 challenge,
// both base64url without padding (RFC 7636).
func generatePKCE() (pkcePair, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return pkcePair{}, fmt.Errorf("generating PKCE verifier: %w", err)
	}
	verifier := base64.RawURLEncoding.EncodeToString(raw)
	sum := sha256.Sum256([]byte(verifier))
	return pkcePair{
		Verifier:  verifier,
		Challenge: base64.RawURLEncoding.EncodeToString(sum[:]),
	}, nil
}

// randomState produces an opaque state value for CSRF protection.
func randomState() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("generating OAuth state: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}

// ---------------------------------------------------------------------------
// Loopback callback server
// ---------------------------------------------------------------------------

// callbackResult is what the redirect delivered.
type callbackResult struct {
	Code  string
	State string
}

// callbackServer receives the OAuth redirect on the loopback interface.
type callbackServer struct {
	server *http.Server
	// results carries the first successful callback.
	results chan callbackResult
	// addr is the bound address, which the test suite needs.
	addr string
}

// callbackHost resolves the loopback host, honoring pi's override so an
// environment with an unusual loopback setup can still work.
func callbackHost(env map[string]string) string {
	if host := env["GOSHCODER_OAUTH_CALLBACK_HOST"]; host != "" {
		return host
	}
	if host := env["PI_OAUTH_CALLBACK_HOST"]; host != "" {
		return host
	}
	return "127.0.0.1"
}

// startCallbackServer listens on host:port and serves path, accepting a
// callback whose state matches expectedState.
func startCallbackServer(host string, port int, path, expectedState string) (*callbackServer, error) {
	if host != "localhost" {
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			return nil, fmt.Errorf("oauth callback host must be loopback, got %q", host)
		}
	}
	listener, err := net.Listen("tcp", net.JoinHostPort(host, fmt.Sprint(port)))
	if err != nil {
		return nil, fmt.Errorf("cannot listen on %s:%d for the OAuth callback: %w", host, port, err)
	}
	if address, ok := listener.Addr().(*net.TCPAddr); !ok || !address.IP.IsLoopback() {
		listener.Close()
		return nil, errors.New("oauth callback server did not bind to a loopback address")
	}

	cs := &callbackServer{
		results: make(chan callbackResult, 1),
		addr:    listener.Addr().String(),
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != path {
			writeCallbackPage(w, http.StatusNotFound, "Callback route not found.")
			return
		}
		query := r.URL.Query()
		if errorCode := query.Get("error"); errorCode != "" {
			writeCallbackPage(w, http.StatusBadRequest, "Authentication did not complete: "+errorCode)
			return
		}
		code, state := query.Get("code"), query.Get("state")
		if code == "" || state == "" {
			writeCallbackPage(w, http.StatusBadRequest, "Missing code or state parameter.")
			return
		}
		// A mismatched state means the callback did not originate from this
		// login attempt.
		if state != expectedState {
			writeCallbackPage(w, http.StatusBadRequest, "State mismatch.")
			return
		}
		writeCallbackPage(w, http.StatusOK, "Authentication completed. You can close this window.")
		select {
		case cs.results <- callbackResult{Code: code, State: state}:
		default:
		}
	})

	cs.server = &http.Server{
		Handler: mux, ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Second,
		IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10,
	}
	go func() { _ = cs.server.Serve(listener) }()
	return cs, nil
}

func (cs *callbackServer) close() {
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = cs.server.Shutdown(shutdownCtx)
}

// writeCallbackPage renders a minimal page for the browser.
func writeCallbackPage(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'")
	w.WriteHeader(status)
	_, _ = fmt.Fprintf(w, "<!doctype html><meta charset=utf-8><title>GoshCoder</title>"+
		"<body style=\"font:16px system-ui;padding:3rem\"><p>%s</p>", html.EscapeString(message))
}

// browserOpener is replaceable so login tests never launch a real browser.
var browserOpener = openBrowser

// openBrowser tries to open a URL in the user's browser. A failure is not fatal:
// the flow prints the URL and accepts a pasted code instead.
func openBrowser(target string) error {
	// Only http(s) URLs are handed to the shell.
	parsed, err := url.Parse(target)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("refusing to open a non-http URL")
	}

	switch runtime.GOOS {
	case "windows":
		// rundll32 avoids cmd.exe's special handling of & in URLs.
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", target).Start()
	case "darwin":
		return exec.Command("open", target).Start()
	default:
		return exec.Command("xdg-open", target).Start()
	}
}

// parseAuthorizationInput accepts a full redirect URL, a "code#state" pair, a
// query fragment, or a bare code (parseAuthorizationInput in anthropic.ts).
func parseAuthorizationInput(input string) (code, state string) {
	value := strings.TrimSpace(input)
	if value == "" {
		return "", ""
	}
	if parsed, err := url.Parse(value); err == nil && parsed.Scheme != "" && parsed.Host != "" {
		query := parsed.Query()
		return query.Get("code"), query.Get("state")
	}
	if before, after, found := strings.Cut(value, "#"); found {
		return before, after
	}
	if strings.Contains(value, "code=") {
		if query, err := url.ParseQuery(value); err == nil {
			return query.Get("code"), query.Get("state")
		}
	}
	return value, ""
}

// runLoopbackLogin drives the shared PKCE + loopback flow: announce the URL,
// race the callback against a manual-code prompt, and return the code.
//
// expectedState is what the callback must carry. The manual path accepts input
// with no state, since a user pasting a bare code cannot supply one.
func runLoopbackLogin(
	ctx context.Context,
	interaction LoginInteraction,
	authURL, redirectURI, expectedState string,
	host string, port int, path string,
) (code string, err error) {
	// The callback ports are fixed and shared with the vendors' own CLIs
	// (Codex binds 1455 too), so a busy port is ordinary rather than fatal.
	// Aborting here made the manual-paste path this function exists to offer
	// unreachable: the auth URL was never shown and there was no way to log in
	// until the user found and killed the other process.
	var callbacks <-chan callbackResult
	server, err := startCallbackServer(host, port, path, expectedState)
	if err != nil {
		interaction.Notify(LoginEvent{
			Kind: "info",
			Message: fmt.Sprintf("Could not listen on %s:%d for the browser callback (%v). "+
				"Complete login in the browser and paste the redirect URL below.", host, port, err),
		})
	} else {
		defer server.close()
		callbacks = server.results
	}

	interaction.Notify(LoginEvent{
		Kind: "auth_url",
		URL:  authURL,
		Instructions: "Complete login in your browser. If the browser is on another machine, " +
			"paste the final redirect URL here.",
	})
	// A failure to launch the browser is expected on headless machines.
	_ = browserOpener(authURL)

	// The prompt and the callback race; whichever finishes first cancels the
	// other.
	raceCtx, cancelRace := context.WithCancel(ctx)
	defer cancelRace()

	type manualResult struct {
		input string
		err   error
	}
	manual := make(chan manualResult, 1)
	go func() {
		input, err := interaction.Prompt(LoginPrompt{
			Kind:        PromptManualCode,
			Message:     "Complete login in your browser, or paste the authorization code / redirect URL here:",
			Placeholder: redirectURI,
			Ctx:         raceCtx,
		})
		manual <- manualResult{input: input, err: err}
	}()

	select {
	case result, ok := <-callbacks:
		if !ok {
			return "", ErrLoginCancelled
		}
		cancelRace()
		return result.Code, nil

	case result := <-manual:
		cancelRace()
		if result.err != nil {
			return "", result.err
		}
		parsedCode, parsedState := parseAuthorizationInput(result.input)
		if parsedCode == "" {
			return "", errors.New("no authorization code was provided")
		}
		// A pasted redirect URL carries state, which must still match.
		if parsedState != "" && parsedState != expectedState {
			return "", errors.New("OAuth state mismatch")
		}
		return parsedCode, nil

	case <-ctx.Done():
		cancelRace()
		return "", ErrLoginCancelled
	}
}

// ---------------------------------------------------------------------------
// Anthropic login
// ---------------------------------------------------------------------------

// Login runs Anthropic's PKCE loopback flow. Anthropic uses the PKCE verifier
// as the state value, which is what its token endpoint expects back.
func (a anthropicOAuth) Login(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	pkce, err := generatePKCE()
	if err != nil {
		return nil, err
	}

	host := callbackHost(env)
	redirectURI := fmt.Sprintf("http://localhost:%d%s", anthropicCallbackPort, anthropicCallbackPath)
	authURL := anthropicAuthorizeURL + "?" + url.Values{
		"code":                  {"true"},
		"client_id":             {anthropicOAuthClientID},
		"response_type":         {"code"},
		"redirect_uri":          {redirectURI},
		"scope":                 {anthropicOAuthScopes},
		"code_challenge":        {pkce.Challenge},
		"code_challenge_method": {"S256"},
		"state":                 {pkce.Verifier},
	}.Encode()

	code, err := runLoopbackLogin(ctx, interaction, authURL, redirectURI,
		pkce.Verifier, host, anthropicCallbackPort, anthropicCallbackPath)
	if err != nil {
		return nil, err
	}

	interaction.Notify(LoginEvent{Kind: "progress", Message: "Exchanging the authorization code for tokens..."})
	status, body, err := postJSON(ctx, anthropicTokenURL, map[string]string{
		"grant_type":    "authorization_code",
		"client_id":     anthropicOAuthClientID,
		"code":          code,
		"state":         pkce.Verifier,
		"redirect_uri":  redirectURI,
		"code_verifier": pkce.Verifier,
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic token exchange request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		return nil, tokenError(a.Name(), "exchange", status, body, &parsed)
	}
	cred, err := credentialFrom(&parsed, anthropicRefreshSkew)
	if err != nil {
		return nil, fmt.Errorf("anthropic token exchange: %w", err)
	}
	return cred, nil
}

// ---------------------------------------------------------------------------
// Kimi Code login
// ---------------------------------------------------------------------------

const (
	kimiDeviceCodeTimeout         = 15 * time.Minute
	kimiDeviceCodeDefaultInterval = 5 * time.Second
	kimiDeviceGrantType           = "urn:ietf:params:oauth:grant-type:device_code"
)

type kimiDeviceAuthorization struct {
	DeviceCode              string  `json:"device_code"`
	UserCode                string  `json:"user_code"`
	VerificationURI         string  `json:"verification_uri"`
	VerificationURIComplete string  `json:"verification_uri_complete"`
	Interval                float64 `json:"interval"`
	ExpiresIn               float64 `json:"expires_in"`
}

// Login runs Kimi's RFC 8628 device authorization flow.
func (k kimiCodingOAuth) Login(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	host := kimiOAuthHost(env)
	status, body, err := postForm(ctx, host+"/api/oauth/device_authorization", url.Values{
		"client_id": {kimiOAuthClientID},
	})
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrLoginCancelled
		}
		return nil, fmt.Errorf("kimi Code device authorization request failed: %w", err)
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("kimi Code device authorization failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	}

	var device kimiDeviceAuthorization
	if err := json.Unmarshal(body, &device); err != nil {
		return nil, fmt.Errorf("invalid Kimi Code device authorization response: %w", err)
	}
	if device.DeviceCode == "" || device.UserCode == "" ||
		!trustedHTTPURL(device.VerificationURI) || !trustedHTTPURL(device.VerificationURIComplete) {
		return nil, fmt.Errorf("invalid Kimi Code device authorization response: %s", truncate(string(body), 300))
	}
	interval := kimiDeviceCodeDefaultInterval
	if device.Interval > 0 {
		interval = max(time.Second, time.Duration(device.Interval*float64(time.Second)))
	}
	timeout := kimiDeviceCodeTimeout
	if device.ExpiresIn > 0 {
		timeout = time.Duration(device.ExpiresIn * float64(time.Second))
	}

	interaction.Notify(LoginEvent{
		Kind:             "device_code",
		UserCode:         device.UserCode,
		VerificationURI:  device.VerificationURIComplete,
		IntervalSeconds:  int(interval.Seconds()),
		ExpiresInSeconds: int(timeout.Seconds()),
	})
	return k.pollDeviceToken(ctx, host, &device, interval, timeout)
}

func trustedHTTPURL(value string) bool {
	parsed, err := url.Parse(value)
	return err == nil && parsed.Host != "" && (parsed.Scheme == "http" || parsed.Scheme == "https")
}

func (k kimiCodingOAuth) pollDeviceToken(ctx context.Context, host string, device *kimiDeviceAuthorization, interval, timeout time.Duration) (*Credential, error) {
	deadline := time.Now().Add(timeout)
	wait := func(delay time.Duration) error {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return errors.New("device flow timed out")
		}
		if delay > remaining {
			delay = remaining
		}
		select {
		case <-ctx.Done():
			return ErrLoginCancelled
		case <-time.After(delay):
			return nil
		}
	}

	// Kimi explicitly requires waiting before the first poll.
	if err := wait(interval); err != nil {
		return nil, err
	}
	for time.Now().Before(deadline) {
		status, body, err := postForm(ctx, host+"/api/oauth/token", url.Values{
			"client_id":   {kimiOAuthClientID},
			"device_code": {device.DeviceCode},
			"grant_type":  {kimiDeviceGrantType},
		})
		if err != nil {
			if ctx.Err() != nil {
				return nil, ErrLoginCancelled
			}
			return nil, fmt.Errorf("kimi Code device token request failed: %w", err)
		}
		if status >= 500 {
			return nil, fmt.Errorf("kimi Code device token request failed (status %d): %s",
				status, strings.TrimSpace(truncate(string(body), 300)))
		}

		var parsed tokenResponse
		_ = json.Unmarshal(body, &parsed)
		if status >= 200 && status < 300 {
			cred, err := credentialFrom(&parsed, 0)
			if err != nil {
				return nil, fmt.Errorf("kimi Code token poll: %w", err)
			}
			return cred, nil
		}
		switch parsed.Error {
		case "authorization_pending":
		case "slow_down":
			var payload struct {
				Interval float64 `json:"interval"`
			}
			_ = json.Unmarshal(body, &payload)
			if payload.Interval > 0 {
				interval = max(time.Second, time.Duration(payload.Interval*float64(time.Second)))
			} else {
				interval += 5 * time.Second
			}
		case "expired_token":
			return nil, errors.New("kimi Code device authorization expired; restart login")
		case "access_denied":
			return nil, errors.New("kimi Code login was denied")
		default:
			return nil, fmt.Errorf("kimi Code device token request failed (status %d): %s",
				status, strings.TrimSpace(truncate(string(body), 300)))
		}
		if err := wait(interval); err != nil {
			return nil, err
		}
	}
	return nil, errors.New("device flow timed out")
}

// ---------------------------------------------------------------------------
// OpenAI Codex login
// ---------------------------------------------------------------------------

const (
	codexBrowserLoginMethod    = "browser"
	codexDeviceCodeLoginMethod = "device_code"
	// codexCallbackPort is fixed: the redirect URI is registered with the
	// client id and cannot be varied.
	codexCallbackPort = 1455
	codexCallbackPath = "/auth/callback"
	// codexDeviceCodeTimeout bounds the device flow.
	codexDeviceCodeTimeout = 15 * time.Minute
)

// Login offers browser and device-code methods, matching pi's selector.
func (o openAICodexOAuth) Login(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	method, err := interaction.Prompt(LoginPrompt{
		Kind:    PromptSelect,
		Message: "Select the OpenAI Codex login method:",
		Options: []LoginPromptOption{
			{ID: codexBrowserLoginMethod, Label: "Browser login (default)"},
			{ID: codexDeviceCodeLoginMethod, Label: "Device code login (headless)"},
		},
		Ctx: ctx,
	})
	if err != nil {
		return nil, err
	}

	switch method {
	case codexDeviceCodeLoginMethod:
		return o.loginDeviceCode(ctx, interaction)
	case codexBrowserLoginMethod, "":
		return o.loginBrowser(ctx, interaction, env)
	default:
		return nil, fmt.Errorf("unknown OpenAI Codex login method %q", method)
	}
}

// loginBrowser runs the PKCE loopback flow.
func (o openAICodexOAuth) loginBrowser(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	pkce, err := generatePKCE()
	if err != nil {
		return nil, err
	}
	state, err := randomState()
	if err != nil {
		return nil, err
	}

	authURL := codexAuthorizeURL + "?" + url.Values{
		"response_type":              {"code"},
		"client_id":                  {codexOAuthClientID},
		"redirect_uri":               {codexRedirectURI},
		"scope":                      {codexOAuthScope},
		"code_challenge":             {pkce.Challenge},
		"code_challenge_method":      {"S256"},
		"state":                      {state},
		"id_token_add_organizations": {"true"},
		"codex_cli_simplified_flow":  {"true"},
		"originator":                 {"goshcoder"},
	}.Encode()

	code, err := runLoopbackLogin(ctx, interaction, authURL, codexRedirectURI,
		state, callbackHost(env), codexCallbackPort, codexCallbackPath)
	if err != nil {
		return nil, err
	}

	interaction.Notify(LoginEvent{Kind: "progress", Message: "Exchanging the authorization code for tokens..."})
	return o.exchangeCode(ctx, code, pkce.Verifier, codexRedirectURI)
}

// exchangeCode swaps an authorization code for a credential and attaches the
// account id the wire protocol needs.
func (o openAICodexOAuth) exchangeCode(ctx context.Context, code, verifier, redirectURI string) (*Credential, error) {
	status, body, err := postForm(ctx, codexTokenURL, url.Values{
		"grant_type":    {"authorization_code"},
		"client_id":     {codexOAuthClientID},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"code_verifier": {verifier},
	})
	if err != nil {
		return nil, fmt.Errorf("OpenAI Codex token exchange request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		return nil, tokenError(o.Name(), "exchange", status, body, &parsed)
	}
	cred, err := credentialFrom(&parsed, 0)
	if err != nil {
		return nil, fmt.Errorf("OpenAI Codex token exchange: %w", err)
	}
	accountID, err := codexAccountID(cred.Access)
	if err != nil {
		return nil, err
	}
	if err := cred.SetExtra("accountId", accountID); err != nil {
		return nil, err
	}
	return cred, nil
}

// codexDeviceAuth is the device authorization Codex issues.
type codexDeviceAuth struct {
	DeviceAuthID    string
	UserCode        string
	IntervalSeconds int
}

// loginDeviceCode runs the headless device flow: show a code, poll until the
// user authorizes, then exchange the resulting authorization code.
func (o openAICodexOAuth) loginDeviceCode(ctx context.Context, interaction LoginInteraction) (*Credential, error) {
	device, err := o.startDeviceAuth(ctx)
	if err != nil {
		return nil, err
	}

	interaction.Notify(LoginEvent{
		Kind:             "device_code",
		UserCode:         device.UserCode,
		VerificationURI:  codexDeviceVerifyURI,
		IntervalSeconds:  device.IntervalSeconds,
		ExpiresInSeconds: int(codexDeviceCodeTimeout.Seconds()),
	})

	code, verifier, err := o.pollDeviceAuth(ctx, device)
	if err != nil {
		return nil, err
	}
	interaction.Notify(LoginEvent{Kind: "progress", Message: "Exchanging the authorization code for tokens..."})
	return o.exchangeCode(ctx, code, verifier, codexDeviceRedirectURI)
}

// startDeviceAuth requests a device and user code.
func (o openAICodexOAuth) startDeviceAuth(ctx context.Context) (*codexDeviceAuth, error) {
	status, body, err := postJSON(ctx, codexDeviceUserCodeURL, map[string]string{
		"client_id": codexOAuthClientID,
	})
	if err != nil {
		return nil, fmt.Errorf("OpenAI Codex device code request failed: %w", err)
	}
	if status == http.StatusNotFound {
		return nil, errors.New("OpenAI Codex device code login is not enabled for this server; use browser login")
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("OpenAI Codex device code request failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	}

	var parsed struct {
		DeviceAuthID string `json:"device_auth_id"`
		UserCode     string `json:"user_code"`
		// interval is sometimes a string.
		Interval json.Number `json:"interval"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("invalid OpenAI Codex device code response: %w", err)
	}
	if parsed.DeviceAuthID == "" || parsed.UserCode == "" {
		return nil, fmt.Errorf("invalid OpenAI Codex device code response: %s", truncate(string(body), 300))
	}
	interval, err := parsed.Interval.Int64()
	if err != nil || interval < 0 {
		return nil, fmt.Errorf("invalid OpenAI Codex device code interval: %q", parsed.Interval.String())
	}
	return &codexDeviceAuth{
		DeviceAuthID:    parsed.DeviceAuthID,
		UserCode:        parsed.UserCode,
		IntervalSeconds: int(interval),
	}, nil
}

// pollDeviceAuth polls until the user authorizes, returning the authorization
// code and its verifier.
func (o openAICodexOAuth) pollDeviceAuth(ctx context.Context, device *codexDeviceAuth) (code, verifier string, err error) {
	err = pollDeviceCode(ctx, device.IntervalSeconds, codexDeviceCodeTimeout, func() (done bool, pollErr error) {
		status, body, requestErr := postJSON(ctx, codexDeviceTokenURL, map[string]string{
			"device_auth_id": device.DeviceAuthID,
			"user_code":      device.UserCode,
		})
		if requestErr != nil {
			return false, fmt.Errorf("OpenAI Codex device token request failed: %w", requestErr)
		}

		if status >= 200 && status < 300 {
			var parsed struct {
				AuthorizationCode string `json:"authorization_code"`
				CodeVerifier      string `json:"code_verifier"`
			}
			if err := json.Unmarshal(body, &parsed); err != nil {
				return false, fmt.Errorf("invalid OpenAI Codex device token response: %w", err)
			}
			if parsed.AuthorizationCode == "" || parsed.CodeVerifier == "" {
				return false, fmt.Errorf("invalid OpenAI Codex device token response: %s", truncate(string(body), 300))
			}
			code, verifier = parsed.AuthorizationCode, parsed.CodeVerifier
			return true, nil
		}

		// Codex reports "still waiting" as 403/404 as well as the RFC's
		// authorization_pending.
		if status == http.StatusForbidden || status == http.StatusNotFound {
			return false, nil
		}
		if codexDevicePending(body) {
			return false, nil
		}
		return false, fmt.Errorf("OpenAI Codex device token request failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	})
	if err != nil {
		return "", "", err
	}
	return code, verifier, nil
}

// codexDevicePending reports whether a device token error means "keep waiting".
func codexDevicePending(body []byte) bool {
	var parsed struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return false
	}
	var asString string
	if err := json.Unmarshal(parsed.Error, &asString); err == nil {
		return asString == "deviceauth_authorization_pending" || asString == "authorization_pending"
	}
	var asObject struct {
		Code string `json:"code"`
	}
	if err := json.Unmarshal(parsed.Error, &asObject); err == nil {
		return asObject.Code == "deviceauth_authorization_pending" || asObject.Code == "authorization_pending"
	}
	return false
}

// ---------------------------------------------------------------------------
// Device code polling
// ---------------------------------------------------------------------------

// deviceCodeMinInterval is the floor on polling frequency.
const deviceCodeMinInterval = time.Second

// deviceCodeDefaultInterval is RFC 8628's default when the server omits one.
const deviceCodeDefaultInterval = 5 * time.Second

// pollDeviceCode polls until poll reports done, the deadline passes, or ctx is
// cancelled (pollOAuthDeviceCodeFlow in device-code.ts).
func pollDeviceCode(ctx context.Context, intervalSeconds int, timeout time.Duration, poll func() (bool, error)) error {
	interval := time.Duration(intervalSeconds) * time.Second
	if interval < deviceCodeMinInterval {
		interval = deviceCodeDefaultInterval
	}
	deadline := time.Now().Add(timeout)

	for time.Now().Before(deadline) {
		if ctx.Err() != nil {
			return ErrLoginCancelled
		}
		done, err := poll()
		if err != nil {
			return err
		}
		if done {
			return nil
		}

		remaining := time.Until(deadline)
		if remaining <= 0 {
			break
		}
		wait := interval
		if remaining < wait {
			wait = remaining
		}
		select {
		case <-ctx.Done():
			return ErrLoginCancelled
		case <-time.After(wait):
		}
	}
	return errors.New("device flow timed out")
}
