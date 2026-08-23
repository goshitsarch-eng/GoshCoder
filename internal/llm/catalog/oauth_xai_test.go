package catalog

import (
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"
)

// xaiServer is a fake xAI issuer: discovery, device authorization, and token,
// each on the path auth.x.ai actually serves it from.
type xaiServer struct {
	server *httptest.Server

	mu       sync.Mutex
	requests map[string][]map[string]string
	// discovery overrides the endpoints the document advertises.
	discovery map[string]string
	// tokenResponses are replayed in order; the last one repeats.
	tokenResponses []tokenServerResponse
	deviceResponse tokenServerResponse
}

func newXAIServer(t *testing.T, tokenResponses ...tokenServerResponse) *xaiServer {
	t.Helper()
	x := &xaiServer{
		requests:       map[string][]map[string]string{},
		tokenResponses: tokenResponses,
		deviceResponse: tokenServerResponse{
			status: http.StatusOK,
			body: `{"device_code":"device-code","user_code":"AAAA-BBBB",` +
				`"verification_uri":"https://accounts.x.ai/oauth2/device",` +
				`"verification_uri_complete":"https://accounts.x.ai/oauth2/device?user_code=AAAA-BBBB",` +
				`"expires_in":600,"interval":1}`,
		},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		x.mu.Lock()
		overrides := x.discovery
		x.mu.Unlock()
		base := x.server.URL
		document := map[string]string{
			"issuer":                        base,
			"authorization_endpoint":        base + "/oauth2/authorize",
			"token_endpoint":                base + "/oauth2/token",
			"device_authorization_endpoint": base + "/oauth2/device/code",
		}
		for key, value := range overrides {
			document[key] = value
		}
		w.Header().Set("Content-Type", "application/json")
		var pairs []string
		for key, value := range document {
			pairs = append(pairs, fmt.Sprintf("%q:%q", key, value))
		}
		fmt.Fprintf(w, "{%s}", strings.Join(pairs, ","))
	})
	mux.HandleFunc("/oauth2/device/code", func(w http.ResponseWriter, r *http.Request) {
		x.record("device", r)
		x.mu.Lock()
		response := x.deviceResponse
		x.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.status)
		fmt.Fprint(w, response.body)
	})
	mux.HandleFunc("/oauth2/token", func(w http.ResponseWriter, r *http.Request) {
		index := x.record("token", r)
		x.mu.Lock()
		if index >= len(x.tokenResponses) {
			index = len(x.tokenResponses) - 1
		}
		response := x.tokenResponses[index]
		x.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.status)
		fmt.Fprint(w, response.body)
	})

	x.server = httptest.NewServer(mux)
	t.Cleanup(x.server.Close)

	previous := xaiIssuer
	xaiIssuer = x.server.URL
	t.Cleanup(func() { xaiIssuer = previous })
	return x
}

// record stores a request's form fields and returns how many came before it.
func (x *xaiServer) record(route string, r *http.Request) int {
	_ = r.ParseForm()
	fields := map[string]string{}
	for key := range r.PostForm {
		fields[key] = r.PostForm.Get(key)
	}
	x.mu.Lock()
	defer x.mu.Unlock()
	index := len(x.requests[route])
	x.requests[route] = append(x.requests[route], fields)
	return index
}

func (x *xaiServer) request(route string, index int) map[string]string {
	x.mu.Lock()
	defer x.mu.Unlock()
	if index >= len(x.requests[route]) {
		return nil
	}
	return x.requests[route][index]
}

func pendingDevice() tokenServerResponse {
	return tokenServerResponse{
		status: http.StatusBadRequest,
		body:   `{"error":"authorization_pending","error_description":"User has not yet authorized"}`,
	}
}

// TestXAIDeviceLoginPollsUntilAuthorized covers the flow xAI's own device
// endpoint serves: a code, a stretch of authorization_pending, then tokens.
func TestXAIDeviceLoginPollsUntilAuthorized(t *testing.T) {
	server := newXAIServer(t, pendingDevice(), okToken("xai-access", "xai-refresh", 3600))

	interaction := &loginInteractionStub{prompt: func(prompt LoginPrompt) (string, error) {
		if prompt.Kind != PromptSelect {
			return "", fmt.Errorf("unexpected prompt kind %q", prompt.Kind)
		}
		return xaiDeviceCodeLoginMethod, nil
	}}
	c, store := oauthCatalog(t, nil)
	cred, err := c.Login(t.Context(), "xai", interaction)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if cred.Access != "xai-access" || cred.Refresh != "xai-refresh" {
		t.Fatalf("credential = %#v", cred)
	}

	event, ok := interaction.event("device_code")
	if !ok {
		t.Fatal("the user was never shown a device code")
	}
	if event.UserCode != "AAAA-BBBB" || !strings.Contains(event.VerificationURI, "accounts.x.ai") {
		t.Fatalf("device_code event = %#v", event)
	}

	device := server.request("device", 0)
	if device["client_id"] != xaiOAuthDefaultClientID || !strings.Contains(device["scope"], "grok-cli:access") {
		t.Fatalf("device authorization request = %#v", device)
	}
	poll := server.request("token", 0)
	if poll["grant_type"] != xaiDeviceGrantType || poll["device_code"] != "device-code" ||
		poll["client_id"] != xaiOAuthDefaultClientID {
		t.Fatalf("device token request = %#v", poll)
	}
	stored, err := store.Read("xai")
	if err != nil || stored == nil || stored.Type != "oauth" || stored.Access != "xai-access" {
		t.Fatalf("stored credential = %#v, err = %v", stored, err)
	}
}

// TestXAIBrowserLoginExchangesPKCE covers the loopback half: the authorize URL
// must carry the S256 challenge for the verifier the exchange then sends, or
// the grant is refused and no amount of retrying helps.
func TestXAIBrowserLoginExchangesPKCE(t *testing.T) {
	disableBrowser(t)
	server := newXAIServer(t, okToken("browser-access", "browser-refresh", 3600))

	interaction := &loginInteractionStub{prompt: func(prompt LoginPrompt) (string, error) {
		switch prompt.Kind {
		case PromptSelect:
			return xaiBrowserLoginMethod, nil
		case PromptManualCode:
			return "authorization-code", nil
		}
		return "", fmt.Errorf("unexpected prompt kind %q", prompt.Kind)
	}}
	c, _ := oauthCatalog(t, nil)
	cred, err := c.Login(t.Context(), "xai", interaction)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if cred.Access != "browser-access" {
		t.Fatalf("credential = %#v", cred)
	}

	event, ok := interaction.event("auth_url")
	if !ok {
		t.Fatal("missing auth_url event")
	}
	parsed, err := url.Parse(event.URL)
	if err != nil {
		t.Fatal(err)
	}
	// Discovery moved the authorize endpoint onto the test server; the flow
	// must have used it rather than the compiled-in default.
	if !strings.HasPrefix(event.URL, server.server.URL+"/oauth2/authorize") {
		t.Fatalf("authorize URL = %q, want the discovered endpoint", event.URL)
	}
	query := parsed.Query()
	if query.Get("client_id") != xaiOAuthDefaultClientID ||
		query.Get("redirect_uri") != xaiRedirectURI ||
		query.Get("code_challenge_method") != "S256" ||
		query.Get("state") == "" || query.Get("nonce") == "" {
		t.Fatalf("authorize query = %v", query)
	}
	// The consent screen must be told who is actually asking.
	if query.Get("referrer") != "goshcoder" {
		t.Fatalf("referrer = %q, want this program's own name", query.Get("referrer"))
	}

	exchange := server.request("token", 0)
	sum := sha256.Sum256([]byte(exchange["code_verifier"]))
	if query.Get("code_challenge") != base64.RawURLEncoding.EncodeToString(sum[:]) {
		t.Fatal("the authorize URL's challenge does not match the verifier the exchange sent")
	}
	if exchange["grant_type"] != "authorization_code" || exchange["code"] != "authorization-code" ||
		exchange["redirect_uri"] != xaiRedirectURI {
		t.Fatalf("token exchange request = %#v", exchange)
	}
}

// TestXAIDiscoveryRefusesAnEndpointOffTheIssuer is the reason discovery is
// consulted at all rather than trusted: a document that relocated the token
// endpoint would be handed the authorization code and the refresh token.
func TestXAIDiscoveryRefusesAnEndpointOffTheIssuer(t *testing.T) {
	server := newXAIServer(t, okToken("a", "b", 3600))
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("the token exchange reached a host the issuer does not own")
	}))
	t.Cleanup(elsewhere.Close)

	server.mu.Lock()
	server.discovery = map[string]string{"token_endpoint": elsewhere.URL + "/oauth2/token"}
	server.mu.Unlock()

	endpoints := discoverXAIEndpoints(t.Context())
	if endpoints.Token != strings.TrimRight(xaiIssuer, "/")+"/oauth2/token" {
		t.Fatalf("token endpoint = %q, want the issuer's own", endpoints.Token)
	}
	// The endpoints it does own are still honored.
	if endpoints.Device != server.server.URL+"/oauth2/device/code" {
		t.Fatalf("device endpoint = %q", endpoints.Device)
	}
}

// TestXAIDiscoveryFallsBackWhenUnavailable keeps a login possible when the
// discovery document is down: the paths are documented, so an outage there is
// not a reason to refuse to log in.
func TestXAIDiscoveryFallsBackWhenUnavailable(t *testing.T) {
	down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(down.Close)
	previous := xaiIssuer
	xaiIssuer = down.URL
	t.Cleanup(func() { xaiIssuer = previous })

	endpoints := discoverXAIEndpoints(t.Context())
	if endpoints != xaiDefaultEndpoints() {
		t.Fatalf("endpoints = %#v, want the documented defaults", endpoints)
	}
}

// TestXAIRefreshCarriesANonRotatingTokenForward covers a token endpoint that
// omits refresh_token because the old one still works. Storing the empty value
// would log the user out at the next expiry.
func TestXAIRefreshCarriesANonRotatingTokenForward(t *testing.T) {
	newXAIServer(t, tokenServerResponse{
		status: http.StatusOK,
		body:   `{"access_token":"fresh-access","expires_in":3600}`,
	})

	cred, err := xaiOAuth{}.Refresh(t.Context(),
		&Credential{Type: "oauth", Access: "stale", Refresh: "long-lived-refresh"}, nil)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if cred.Access != "fresh-access" || cred.Refresh != "long-lived-refresh" {
		t.Fatalf("credential = %#v", cred)
	}
	if cred.Expires <= time.Now().UnixMilli() {
		t.Fatalf("expiry %d is not in the future", cred.Expires)
	}
}

// TestXAIRefreshReportsADeadCredential distinguishes "log in again" from a
// transient failure; only the former should stop the caller retrying.
func TestXAIRefreshReportsADeadCredential(t *testing.T) {
	newXAIServer(t, tokenServerResponse{
		status: http.StatusBadRequest,
		body:   `{"error":"invalid_grant","error_description":"refresh token revoked"}`,
	})

	_, err := xaiOAuth{}.Refresh(t.Context(),
		&Credential{Type: "oauth", Refresh: "revoked"}, nil)
	if !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("Refresh error = %v, want ErrOAuthUnauthorized", err)
	}
}

// TestXAIAuthSendsTheTokenWhereTheAPIKeyGoes: the Grok models speak
// openai-completions, which authenticates with Authorization: Bearer built
// from the api key, so an OAuth token needs no separate header.
func TestXAIAuthSendsTheTokenWhereTheAPIKeyGoes(t *testing.T) {
	auth := xaiOAuth{}.ToAuth(&Credential{Type: "oauth", Access: "token"})
	if auth.APIKey != "token" || len(auth.Headers) != 0 || auth.Source != "OAuth" {
		t.Fatalf("auth = %#v", auth)
	}
}

// TestXAIClientIDOverride covers the escape hatch for a user with their own
// registered xAI application.
func TestXAIClientIDOverride(t *testing.T) {
	if got := xaiOAuthClientID(nil); got != xaiOAuthDefaultClientID {
		t.Fatalf("default client id = %q", got)
	}
	env := map[string]string{"GOSHCODER_XAI_OAUTH_CLIENT_ID": "own-client"}
	if got := xaiOAuthClientID(env); got != "own-client" {
		t.Fatalf("overridden client id = %q", got)
	}
}

// TestXAIAPIKeyPathIsUnaffectedByTheOAuthFlow: adding a subscription login must
// not disturb the developer-account route, which is the one that reaches the
// paid API.
func TestXAIAPIKeyPathIsUnaffectedByTheOAuthFlow(t *testing.T) {
	c, _ := oauthCatalog(t, map[string]string{"XAI_API_KEY": "xai-key"})
	auth, ok := c.ResolveAuth("xai")
	if !ok || auth.APIKey != "xai-key" || auth.Source != "XAI_API_KEY" {
		t.Fatalf("auth = %#v, ok = %v", auth, ok)
	}
}
