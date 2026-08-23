package catalog

// Meta Model API OAuth.
//
// Meta's coding CLI signs in with an RFC 8628 device authorization against
// https://auth.meta.com and then exchanges the resulting identity token for a
// Model API key at https://api.meta.ai/muse-code/key. The key, not the
// identity token, is what api.meta.ai accepts on inference requests, and Meta
// re-mints it about once a day.
//
// That shape maps onto the stored credential the same way pi's does: the
// identity token lives in the refresh slot, the minted key in the access slot,
// and an expiry short enough that a re-mint happens before the key goes stale.
// Refresh therefore means "mint a new key", not "exchange a refresh token" --
// though a refresh token is kept when the server issues one, so an expired
// identity can be renewed once without a new login.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"goshcoder/internal/llm"
)

const (
	// metaOAuthDefaultClientID is the public client Meta's own installer uses.
	metaOAuthDefaultClientID = "1031625952748946"

	metaDeviceGrantType = "urn:ietf:params:oauth:grant-type:device_code"
	// metaAPIVersion is the version header Meta's CLI sends to the muse-code
	// endpoints.
	metaAPIVersion = "1.0.0"

	// metaKeyValidity is how long a minted key is treated as good for. Meta
	// re-mints daily, so this sits just inside a day: the credential refreshes
	// itself before the key it holds can be retired underneath a running turn.
	metaKeyValidity = 20 * time.Hour

	metaDeviceCodeTimeout = 15 * time.Minute

	// metaRefreshTokenExtra carries the OAuth refresh token, which the stored
	// credential's own refresh slot cannot hold: that slot holds the identity
	// token the mint endpoint needs.
	metaRefreshTokenExtra = "metaRefreshToken"
	// metaIdentityExpiresExtra records when the identity token dies, so a
	// re-mint that fails can tell "expired login" from "Meta said no".
	metaIdentityExpiresExtra = "metaIdentityExpires"
)

// Meta's hosts are vars so tests can point the flow at a fake server.
var (
	metaAuthBaseURL = "https://auth.meta.com"
	metaAPIBaseURL  = "https://api.meta.ai"
)

func metaDeviceAuthorizationURL() string {
	return strings.TrimRight(metaAuthBaseURL, "/") + "/oidc/device/authorization/"
}

func metaTokenURL() string {
	return strings.TrimRight(metaAuthBaseURL, "/") + "/oidc/device/token/"
}

func metaMintURL() string {
	return strings.TrimRight(metaAPIBaseURL, "/") + "/muse-code/key"
}

// metaOAuthClientID resolves the client id, honoring an override for a user
// whose account is enrolled against a different Meta application.
func metaOAuthClientID(env map[string]string) string {
	if id := env["GOSHCODER_META_OAUTH_CLIENT_ID"]; id != "" {
		return id
	}
	return metaOAuthDefaultClientID
}

type metaOAuth struct{}

func (metaOAuth) Name() string { return "Meta (Model API)" }

// ToAuth sends the minted key as a bearer token. api.meta.ai speaks the
// Anthropic Messages shape but authenticates with Authorization rather than
// x-api-key, so the key travels as a header and not as the protocol's api key
// -- which is also what stops the streamer sending an x-api-key Meta would
// have to ignore.
func (metaOAuth) ToAuth(cred *Credential) *Auth {
	bearer := "Bearer " + cred.Access
	return &Auth{
		Headers: metaBearerHeaders(bearer),
		Source:  "OAuth",
	}
}

// metaBearerHeaders is the header set both credential paths produce.
func metaBearerHeaders(bearer string) llm.ProviderHeaders {
	value := bearer
	return llm.ProviderHeaders{"Authorization": &value}
}

// Refresh mints a new Model API key.
//
// The cheap path is a re-mint with the identity token already stored, which is
// what the daily key rotation needs. Only when Meta rejects that identity is
// the refresh token spent, and only then can the result be a terminal
// "log in again".
func (m metaOAuth) Refresh(ctx context.Context, current *Credential, env map[string]string) (*Credential, error) {
	identity := current.Refresh
	refreshToken, _ := current.Extra(metaRefreshTokenExtra)
	identityExpires := metaIdentityExpiry(current)

	// Skip straight to renewal when the identity is known to be dead; a mint
	// call with it could only fail.
	identityUsable := identity != "" &&
		(identityExpires == 0 || time.Now().Add(oauthMinimumValidity).UnixMilli() < identityExpires)

	if identityUsable {
		minted, err := m.mint(ctx, identity)
		if err == nil {
			return metaCredential(identity, refreshToken, identityExpires, minted.APIKey), nil
		}
		if !errors.Is(err, ErrOAuthUnauthorized) {
			return nil, err
		}
	}

	if refreshToken == "" {
		return nil, fmt.Errorf("%s login is no longer valid; log in again: %w", m.Name(), ErrOAuthUnauthorized)
	}
	grant, err := m.exchange(ctx, env, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
	})
	if err != nil {
		return nil, err
	}
	minted, err := m.mint(ctx, grant.AccessToken)
	if err != nil {
		return nil, err
	}
	return metaCredential(grant.AccessToken, grant.refreshOr(refreshToken), grant.expiryMillis(), minted.APIKey), nil
}

// metaIdentityExpiry reads the recorded identity expiry, or 0 when unknown.
func metaIdentityExpiry(cred *Credential) int64 {
	raw, ok := cred.Extra(metaIdentityExpiresExtra)
	if !ok {
		return 0
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0
	}
	return value
}

// metaCredential assembles the stored credential. The expiry is the sooner of
// the key's own re-mint deadline and the identity token's death, because
// either one ending is a reason to run Refresh.
func metaCredential(identity, refreshToken string, identityExpires int64, apiKey string) *Credential {
	expires := time.Now().Add(metaKeyValidity).UnixMilli()
	if identityExpires > 0 && identityExpires < expires {
		expires = identityExpires
	}
	cred := &Credential{
		Type:    "oauth",
		Access:  apiKey,
		Refresh: identity,
		Expires: expires,
	}
	if refreshToken != "" {
		_ = cred.SetExtra(metaRefreshTokenExtra, refreshToken)
	}
	if identityExpires > 0 {
		_ = cred.SetExtra(metaIdentityExpiresExtra, strconv.FormatInt(identityExpires, 10))
	}
	return cred
}

// ---------------------------------------------------------------------------
// Token exchange
// ---------------------------------------------------------------------------

// metaTokenGrant is Meta's token response.
type metaTokenGrant struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

func (g metaTokenGrant) refreshOr(previous string) string {
	if g.RefreshToken != "" {
		return g.RefreshToken
	}
	return previous
}

// expiryMillis converts the grant's lifetime to an absolute deadline, or 0
// when the server did not say.
func (g metaTokenGrant) expiryMillis() int64 {
	if g.ExpiresIn <= 0 {
		return 0
	}
	return time.Now().Add(time.Duration(g.ExpiresIn) * time.Second).UnixMilli()
}

// exchange posts to Meta's token endpoint with the client id attached.
func (m metaOAuth) exchange(ctx context.Context, env map[string]string, form url.Values) (*metaTokenGrant, error) {
	form.Set("client_id", metaOAuthClientID(env))

	status, body, err := postForm(ctx, metaTokenURL(), form)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrLoginCancelled
		}
		return nil, fmt.Errorf("meta token request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		// Meta answers an unusable refresh token with a bare 404 rather than
		// invalid_grant, so a missing token endpoint and a dead credential
		// look alike. Both mean the same thing to the user: log in again.
		if status == http.StatusNotFound {
			return nil, fmt.Errorf("%s login is no longer valid; log in again: %w", m.Name(), ErrOAuthUnauthorized)
		}
		return nil, tokenError(m.Name(), "exchange", status, body, &parsed)
	}

	var grant metaTokenGrant
	if err := json.Unmarshal(body, &grant); err != nil {
		return nil, fmt.Errorf("invalid Meta token response: %w", err)
	}
	if grant.AccessToken == "" {
		return nil, errors.New("meta token response carried no access token")
	}
	return &grant, nil
}

// ---------------------------------------------------------------------------
// Key minting
// ---------------------------------------------------------------------------

// metaMintedKey is what the mint endpoint returns.
type metaMintedKey struct {
	APIKey         string `json:"api_key"`
	RequirePayment bool   `json:"require_payment"`
	ActionURL      string `json:"action_url"`
	UserEmail      string `json:"user_email"`
}

// metaProblem is Meta's RFC 7807-shaped error body.
type metaProblem struct {
	Title  string `json:"title"`
	Detail string `json:"detail"`
	Status int    `json:"status"`
}

// mint exchanges an identity token for a Model API key.
//
// The identity travels both as a bearer token and in the request body, which
// is what Meta's own client sends; the endpoint answers every unauthenticated
// shape with the same 401, so there is no way to narrow it from outside and no
// cost to satisfying both.
func (m metaOAuth) mint(ctx context.Context, identity string) (*metaMintedKey, error) {
	if identity == "" {
		return nil, fmt.Errorf("%s has no identity token to mint from: %w", m.Name(), ErrOAuthUnauthorized)
	}
	payload, err := json.Marshal(map[string]string{"dca_token": identity})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, metaMintURL(), strings.NewReader(string(payload)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+identity)
	req.Header.Set("x-api-version", metaAPIVersion)

	status, body, err := doTokenRequest(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrLoginCancelled
		}
		return nil, fmt.Errorf("meta Model API key request failed: %w", err)
	}

	if status == http.StatusUnauthorized || status == http.StatusForbidden {
		return nil, fmt.Errorf("%s rejected the saved login; log in again: %w", m.Name(), ErrOAuthUnauthorized)
	}
	if status < 200 || status >= 300 {
		var problem metaProblem
		_ = json.Unmarshal(body, &problem)
		if problem.Detail != "" {
			return nil, fmt.Errorf("meta Model API key request failed (status %d): %s", status, problem.Detail)
		}
		return nil, fmt.Errorf("meta Model API key request failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	}

	var minted metaMintedKey
	if err := json.Unmarshal(body, &minted); err != nil {
		return nil, fmt.Errorf("invalid Meta Model API key response: %w", err)
	}
	// An account that has not finished signup mints nothing, and says where to
	// finish. Reporting that is the difference between a fixable message and
	// "no credentials configured".
	if minted.RequirePayment {
		if minted.ActionURL != "" {
			return nil, fmt.Errorf("this Meta account is not set up for the Model API yet; finish setup at %s", minted.ActionURL)
		}
		return nil, errors.New("this Meta account is not set up for the Model API yet")
	}
	if minted.APIKey == "" {
		return nil, errors.New("meta returned an empty Model API key")
	}
	return &minted, nil
}

// ---------------------------------------------------------------------------
// Login
// ---------------------------------------------------------------------------

// metaDeviceAuthorization is Meta's RFC 8628 device authorization response.
type metaDeviceAuthorization struct {
	DeviceCode              string  `json:"device_code"`
	UserCode                string  `json:"user_code"`
	VerificationURI         string  `json:"verification_uri"`
	VerificationURIComplete string  `json:"verification_uri_complete"`
	Interval                float64 `json:"interval"`
	ExpiresIn               float64 `json:"expires_in"`
}

// Login runs the device flow and mints the first Model API key.
func (m metaOAuth) Login(ctx context.Context, interaction LoginInteraction, env map[string]string) (*Credential, error) {
	status, body, err := postForm(ctx, metaDeviceAuthorizationURL(), url.Values{
		"client_id": {metaOAuthClientID(env)},
	})
	if err != nil {
		if ctx.Err() != nil {
			return nil, ErrLoginCancelled
		}
		return nil, fmt.Errorf("meta device authorization request failed: %w", err)
	}
	if status < 200 || status >= 300 {
		return nil, fmt.Errorf("meta device authorization failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	}

	var device metaDeviceAuthorization
	if err := json.Unmarshal(body, &device); err != nil {
		return nil, fmt.Errorf("invalid Meta device authorization response: %w", err)
	}
	if device.DeviceCode == "" || device.UserCode == "" || !trustedHTTPURL(device.VerificationURI) {
		return nil, fmt.Errorf("invalid Meta device authorization response: %s", truncate(string(body), 300))
	}
	verification := device.VerificationURI
	if device.VerificationURIComplete != "" && trustedHTTPURL(device.VerificationURIComplete) {
		verification = device.VerificationURIComplete
	}

	timeout := metaDeviceCodeTimeout
	if device.ExpiresIn > 0 {
		timeout = time.Duration(device.ExpiresIn * float64(time.Second))
	}
	interaction.Notify(LoginEvent{
		Kind:             "device_code",
		UserCode:         device.UserCode,
		VerificationURI:  verification,
		IntervalSeconds:  int(device.Interval),
		ExpiresInSeconds: int(timeout.Seconds()),
	})

	grant, err := m.pollDeviceToken(ctx, env, device.DeviceCode, int(device.Interval), timeout)
	if err != nil {
		return nil, err
	}

	interaction.Notify(LoginEvent{Kind: "progress", Message: "Requesting a Meta Model API key..."})
	minted, err := m.mint(ctx, grant.AccessToken)
	if err != nil {
		return nil, err
	}
	return metaCredential(grant.AccessToken, grant.RefreshToken, grant.expiryMillis(), minted.APIKey), nil
}

func (m metaOAuth) pollDeviceToken(ctx context.Context, env map[string]string, deviceCode string, interval int, timeout time.Duration) (*metaTokenGrant, error) {
	clientID := metaOAuthClientID(env)
	var grant *metaTokenGrant

	err := pollDeviceCode(ctx, interval, timeout, func() (bool, error) {
		status, body, err := postForm(ctx, metaTokenURL(), url.Values{
			"grant_type":  {metaDeviceGrantType},
			"device_code": {deviceCode},
			"client_id":   {clientID},
		})
		if err != nil {
			if ctx.Err() != nil {
				return false, ErrLoginCancelled
			}
			return false, fmt.Errorf("meta device token request failed: %w", err)
		}

		var parsed tokenResponse
		_ = json.Unmarshal(body, &parsed)
		if status >= 200 && status < 300 {
			var decoded metaTokenGrant
			if err := json.Unmarshal(body, &decoded); err != nil {
				return false, fmt.Errorf("invalid Meta token response: %w", err)
			}
			if decoded.AccessToken == "" {
				return false, errors.New("meta token response carried no access token")
			}
			grant = &decoded
			return true, nil
		}

		switch parsed.Error {
		case "authorization_pending":
			return false, nil
		case "slow_down":
			select {
			case <-ctx.Done():
				return false, ErrLoginCancelled
			case <-time.After(5 * time.Second):
			}
			return false, nil
		case "expired_token":
			return false, errors.New("meta device authorization expired; restart login")
		case "access_denied":
			return false, errors.New("meta login was denied")
		}
		return false, fmt.Errorf("meta device token request failed (status %d): %s",
			status, strings.TrimSpace(truncate(string(body), 300)))
	})
	if err != nil {
		return nil, err
	}
	return grant, nil
}
