package catalog

// xAI (Grok) OAuth.
//
// xAI publishes a first-party OIDC authorization server at https://auth.x.ai:
// a discovery document, PKCE S256, RFC 8628 device authorization, and public
// clients (its token_endpoint_auth_methods_supported includes "none"). This
// flow uses that published surface. The credential it stores is an access
// token issued by xAI's own authorization server, not a scraped browser
// session.
//
// What the flow authenticates is a consumer Grok subscription rather than a
// developer API account, and xAI applies its own entitlement checks to the
// inference API afterwards: a login can succeed and inference still answer 403
// for an account without the plan the endpoint requires. The API-key path
// (XAI_API_KEY, or `goshcoder auth set xai`) is unaffected and remains the
// route for a developer account.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

const (
	// xaiOAuthDefaultClientID is xAI's public desktop client, the one its own
	// CLI login uses. A public client has no secret by construction -- PKCE
	// replaces it -- so this is configuration rather than a credential.
	xaiOAuthDefaultClientID = "b1a00492-073a-47ea-816f-4c329264a828"

	// xaiCallbackPort is fixed: the redirect URI is registered against the
	// client id and cannot be varied.
	xaiCallbackPort = 56121
	xaiCallbackPath = "/callback"
	xaiRedirectURI  = "http://127.0.0.1:56121/callback"

	// xaiOAuthScope is what the CLI surface needs: identity, a refresh token,
	// and the two access scopes the coding endpoints check.
	xaiOAuthScope = "openid profile email offline_access grok-cli:access api:access"

	xaiDeviceCodeTimeout     = 15 * time.Minute
	xaiDeviceGrantType       = "urn:ietf:params:oauth:grant-type:device_code"
	xaiBrowserLoginMethod    = "browser"
	xaiDeviceCodeLoginMethod = "device_code"
)

// xaiIssuer is a var so tests can point the whole flow at a fake OIDC server.
// Every endpoint is derived from it and pinned back to its host, so a
// discovery document cannot move the token exchange to another origin.
var xaiIssuer = "https://auth.x.ai"

// xaiOAuthClientID resolves the client id, honoring an override for a user who
// has registered their own xAI OAuth application.
func xaiOAuthClientID(env map[string]string) string {
	if id := env["GOSHCODER_XAI_OAUTH_CLIENT_ID"]; id != "" {
		return id
	}
	return xaiOAuthDefaultClientID
}

// xaiEndpoints is the subset of the discovery document this flow uses.
type xaiEndpoints struct {
	Authorize string
	Token     string
	Device    string
}

// xaiDefaultEndpoints is what auth.x.ai's discovery document currently
// publishes, used when discovery is unavailable.
func xaiDefaultEndpoints() xaiEndpoints {
	base := strings.TrimRight(xaiIssuer, "/")
	return xaiEndpoints{
		Authorize: base + "/oauth2/authorize",
		Token:     base + "/oauth2/token",
		Device:    base + "/oauth2/device/code",
	}
}

// xaiDiscoveryTimeout bounds the discovery fetch, separately from the 30 s a
// token request gets. Refresh runs inside the credential store's cross-process
// lock, and an advisory fetch must not hold that for half a minute when the
// documented paths are right there.
const xaiDiscoveryTimeout = 5 * time.Second

// xaiDiscovered caches a successful discovery for the process, keyed by issuer
// so a test that redirects the issuer never reads another test's answer. Only
// successes are cached: a transient failure should not pin the fallback for the
// life of the process.
var xaiDiscovered struct {
	sync.Mutex
	issuer    string
	endpoints xaiEndpoints
	valid     bool
}

// discoverXAIEndpoints reads the issuer's OIDC discovery document.
//
// Discovery is advisory: an unreachable or unparsable document falls back to
// the documented paths rather than blocking a login. An endpoint that names
// another host is not advisory -- it is refused, because honoring it would
// send the authorization code and refresh token somewhere other than xAI.
func discoverXAIEndpoints(ctx context.Context) xaiEndpoints {
	issuer := strings.TrimRight(xaiIssuer, "/")
	fallback := xaiDefaultEndpoints()

	xaiDiscovered.Lock()
	if xaiDiscovered.valid && xaiDiscovered.issuer == issuer {
		cached := xaiDiscovered.endpoints
		xaiDiscovered.Unlock()
		return cached
	}
	xaiDiscovered.Unlock()

	fetchCtx, cancel := context.WithTimeout(ctx, xaiDiscoveryTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(fetchCtx, http.MethodGet,
		issuer+"/.well-known/openid-configuration", nil)
	if err != nil {
		return fallback
	}
	req.Header.Set("Accept", "application/json")
	status, body, err := doTokenRequest(req)
	if err != nil || status < 200 || status >= 300 {
		return fallback
	}

	var document struct {
		AuthorizationEndpoint       string `json:"authorization_endpoint"`
		TokenEndpoint               string `json:"token_endpoint"`
		DeviceAuthorizationEndpoint string `json:"device_authorization_endpoint"`
	}
	if err := json.Unmarshal(body, &document); err != nil {
		return fallback
	}

	resolved := fallback
	for _, field := range []struct {
		value string
		into  *string
		name  string
	}{
		{document.AuthorizationEndpoint, &resolved.Authorize, "authorization_endpoint"},
		{document.TokenEndpoint, &resolved.Token, "token_endpoint"},
		{document.DeviceAuthorizationEndpoint, &resolved.Device, "device_authorization_endpoint"},
	} {
		if field.value == "" {
			continue
		}
		if pinned, err := pinToXAIIssuer(field.value); err == nil {
			*field.into = pinned
		}
	}

	xaiDiscovered.Lock()
	xaiDiscovered.issuer, xaiDiscovered.endpoints, xaiDiscovered.valid = issuer, resolved, true
	xaiDiscovered.Unlock()
	return resolved
}

// pinToXAIIssuer rejects an endpoint that is not on the issuer's own host.
// HTTPS is required unless the issuer itself is loopback, which only a test
// server is.
func pinToXAIIssuer(endpoint string) (string, error) {
	issuer, err := url.Parse(xaiIssuer)
	if err != nil || issuer.Host == "" {
		return "", fmt.Errorf("xAI OAuth issuer %q is not a URL", xaiIssuer)
	}
	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" {
		return "", fmt.Errorf("xAI OAuth discovery returned an unusable endpoint %q", endpoint)
	}
	if !strings.EqualFold(parsed.Host, issuer.Host) {
		return "", fmt.Errorf("xAI OAuth discovery endpoint %q is not on %s", endpoint, issuer.Host)
	}
	if parsed.Scheme != "https" && !isLoopbackHostname(parsed.Hostname()) {
		return "", fmt.Errorf("xAI OAuth discovery returned a non-HTTPS endpoint %q", endpoint)
	}
	return endpoint, nil
}

func isLoopbackHostname(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

type xaiOAuth struct{}

func (xaiOAuth) Name() string { return "xAI (Grok subscription)" }

// ToAuth sends the access token where an xAI API key would go. The Grok models
// in the catalog speak openai-completions, which authenticates with
// Authorization: Bearer, so an OAuth token needs no separate header.
func (xaiOAuth) ToAuth(cred *Credential) *Auth {
	return &Auth{APIKey: cred.Access, Source: "OAuth"}
}

func (x xaiOAuth) Refresh(ctx context.Context, current *Credential, env map[string]string) (*Credential, error) {
	endpoints := discoverXAIEndpoints(ctx)
	status, body, err := postForm(ctx, endpoints.Token, url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {xaiOAuthClientID(env)},
		"refresh_token": {current.Refresh},
	})
	if err != nil {
		return nil, fmt.Errorf("xAI token refresh request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		return nil, tokenError(x.Name(), "refresh", status, body, &parsed)
	}
	// A rotating-refresh-token server returns a new one; a non-rotating one
	// omits it and expects the old one to keep working. Dropping it in the
	// second case would log the user out on the next expiry.
	return xaiCredential(&parsed, current.Refresh)
}

// xaiCredential builds a stored credential, carrying the previous refresh
// token forward when the response omits one.
func xaiCredential(parsed *tokenResponse, previousRefresh string) (*Credential, error) {
	if parsed.AccessToken == "" {
		return nil, errors.New("xAI token response carried no access token")
	}
	refresh := parsed.RefreshToken
	if refresh == "" {
		refresh = previousRefresh
	}
	if refresh == "" {
		return nil, errors.New("xAI token response carried no refresh token")
	}
	expiresIn := parsed.ExpiresIn
	if expiresIn <= 0 {
		// xAI's access tokens are short-lived; assuming an hour keeps the
		// refresh cadence sane when the server omits the field.
		expiresIn = 3600
	}
	return &Credential{
		Type:    "oauth",
		Access:  parsed.AccessToken,
		Refresh: refresh,
		Expires: time.Now().Add(time.Duration(expiresIn) * time.Second).UnixMilli(),
	}, nil
}

// ---------------------------------------------------------------------------
// Login
// ---------------------------------------------------------------------------

// Login offers the device flow first. Both work, but the device flow needs no
// listening socket and no browser on this machine, so it is the one that
// behaves the same over SSH as it does locally.
func (x xaiOAuth) Login(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	method, err := interaction.Prompt(LoginPrompt{
		Kind:    PromptSelect,
		Message: "Select the xAI (Grok) login method:",
		Options: []LoginPromptOption{
			{ID: xaiDeviceCodeLoginMethod, Label: "Device code login (default)", Description: "Shows a code to enter at accounts.x.ai; works headless."},
			{ID: xaiBrowserLoginMethod, Label: "Browser login", Description: "Opens a browser and waits on a loopback callback."},
		},
		Ctx: ctx,
	})
	if err != nil {
		return nil, err
	}

	switch method {
	case xaiBrowserLoginMethod:
		return x.loginBrowser(ctx, interaction, env)
	case xaiDeviceCodeLoginMethod, "":
		return x.loginDeviceCode(ctx, interaction, env)
	default:
		return nil, fmt.Errorf("unknown xAI login method %q", method)
	}
}

// loginBrowser runs the PKCE loopback flow.
func (x xaiOAuth) loginBrowser(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	pkce, err := generatePKCE()
	if err != nil {
		return nil, err
	}
	state, err := randomState()
	if err != nil {
		return nil, err
	}
	nonce, err := randomState()
	if err != nil {
		return nil, err
	}

	endpoints := discoverXAIEndpoints(ctx)
	clientID := xaiOAuthClientID(env)
	authURL := endpoints.Authorize + "?" + url.Values{
		"response_type":         {"code"},
		"client_id":             {clientID},
		"redirect_uri":          {xaiRedirectURI},
		"scope":                 {xaiOAuthScope},
		"code_challenge":        {pkce.Challenge},
		"code_challenge_method": {"S256"},
		"state":                 {state},
		"nonce":                 {nonce},
		// referrer identifies the client to the consent page. It names this
		// program, not another vendor's: misreporting who is asking for the
		// grant is exactly what a consent screen exists to prevent.
		"referrer": {"goshcoder"},
	}.Encode()

	code, err := runLoopbackLogin(ctx, interaction, authURL, xaiRedirectURI,
		state, callbackHost(env), xaiCallbackPort, xaiCallbackPath)
	if err != nil {
		return nil, err
	}

	interaction.Notify(LoginEvent{Kind: "progress", Message: "Exchanging the authorization code for tokens..."})
	status, body, err := postForm(ctx, endpoints.Token, url.Values{
		"grant_type":            {"authorization_code"},
		"client_id":             {clientID},
		"code":                  {code},
		"redirect_uri":          {xaiRedirectURI},
		"code_verifier":         {pkce.Verifier},
		"code_challenge":        {pkce.Challenge},
		"code_challenge_method": {"S256"},
	})
	if err != nil {
		return nil, fmt.Errorf("xAI token exchange request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		return nil, tokenError(x.Name(), "exchange", status, body, &parsed)
	}
	return xaiCredential(&parsed, "")
}

// xaiDeviceAuthorization is xAI's RFC 8628 device authorization response.
type xaiDeviceAuthorization struct {
	DeviceCode              string  `json:"device_code"`
	UserCode                string  `json:"user_code"`
	VerificationURI         string  `json:"verification_uri"`
	VerificationURIComplete string  `json:"verification_uri_complete"`
	Interval                float64 `json:"interval"`
	ExpiresIn               float64 `json:"expires_in"`
}

// loginDeviceCode runs the RFC 8628 flow: request a code, show it, poll.
func (x xaiOAuth) loginDeviceCode(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	endpoints := discoverXAIEndpoints(ctx)
	clientID := xaiOAuthClientID(env)

	status, body, err := postForm(ctx, endpoints.Device, url.Values{
		"client_id": {clientID},
		"scope":     {xaiOAuthScope},
	})
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrLoginCancelled
		}
		return nil, fmt.Errorf("xAI device authorization request failed: %w", err)
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("xAI device authorization failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	}

	var device xaiDeviceAuthorization
	if err := json.Unmarshal(body, &device); err != nil {
		return nil, fmt.Errorf("invalid xAI device authorization response: %w", err)
	}
	// The verification URI is opened in a browser rather than sent a
	// credential, and xAI serves it from accounts.x.ai rather than the issuer
	// host, so it is checked for being a usable http(s) URL rather than pinned.
	if device.DeviceCode == "" || device.UserCode == "" || !trustedHTTPURL(device.VerificationURI) {
		return nil, fmt.Errorf("invalid xAI device authorization response: %s", truncate(string(body), 300))
	}
	verification := device.VerificationURI
	if device.VerificationURIComplete != "" && trustedHTTPURL(device.VerificationURIComplete) {
		verification = device.VerificationURIComplete
	}

	interval := int(device.Interval)
	timeout := xaiDeviceCodeTimeout
	if device.ExpiresIn > 0 {
		timeout = time.Duration(device.ExpiresIn * float64(time.Second))
	}
	interaction.Notify(LoginEvent{
		Kind:             "device_code",
		UserCode:         device.UserCode,
		VerificationURI:  verification,
		IntervalSeconds:  interval,
		ExpiresInSeconds: int(timeout.Seconds()),
	})

	var credential *Credential
	err = pollDeviceCode(ctx, interval, timeout, func() (bool, error) {
		pollStatus, pollBody, pollErr := postForm(ctx, endpoints.Token, url.Values{
			"grant_type":  {xaiDeviceGrantType},
			"client_id":   {clientID},
			"device_code": {device.DeviceCode},
		})
		if pollErr != nil {
			if ctx.Err() != nil {
				return false, ErrLoginCancelled
			}
			return false, fmt.Errorf("xAI device token request failed: %w", pollErr)
		}

		var parsed tokenResponse
		_ = json.Unmarshal(pollBody, &parsed)
		if pollStatus >= 200 && pollStatus < 300 {
			built, buildErr := xaiCredential(&parsed, "")
			if buildErr != nil {
				return false, buildErr
			}
			credential = built
			return true, nil
		}

		switch parsed.Error {
		case "authorization_pending":
			return false, nil
		case "slow_down":
			// pollDeviceCode owns the cadence; the extra wait here is the
			// backoff RFC 8628 asks for.
			select {
			case <-ctx.Done():
				return false, ErrLoginCancelled
			case <-time.After(5 * time.Second):
			}
			return false, nil
		case "expired_token":
			return false, errors.New("xAI device authorization expired; restart login")
		case "access_denied":
			return false, errors.New("xAI login was denied")
		}
		return false, fmt.Errorf("xAI device token request failed (status %d): %s",
			pollStatus, strings.TrimSpace(truncate(string(pollBody), 300)))
	})
	if err != nil {
		return nil, err
	}
	return credential, nil
}
