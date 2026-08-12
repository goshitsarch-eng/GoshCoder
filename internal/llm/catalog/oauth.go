package catalog

// OAuth credential handling: token refresh and request-auth derivation.
//
// Port of the refresh/toAuth halves of the OAuthAuth contract in
// reference/pi/packages/ai/src/auth/types.ts, plus the per-provider
// implementations in auth/oauth/*.ts.
//
// The split mirrors pi's: Refresh produces a new credential, ToAuth derives
// request auth from whatever credential ends up stored. That is what lets the
// refresh run inside CredentialStore.Modify, so concurrent requests cannot
// double-refresh a rotated token.
//
// Login flows live in oauth_login.go; this file is what keeps an already-issued
// credential alive.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"goshcoder/internal/llm"
)

// oauthRequestTimeout bounds a single token request.
const oauthRequestTimeout = 30 * time.Second

// ErrOAuthUnauthorized reports a terminal refresh failure: the stored
// credential is dead and the user has to log in again. Callers should surface
// this rather than retrying or silently falling back to ambient credentials.
var ErrOAuthUnauthorized = errors.New("oauth credential is no longer valid; log in again")

// OAuthProvider is the refresh half of pi's OAuthAuth contract.
type OAuthProvider interface {
	// Name is the display name, e.g. "Anthropic (Claude Pro/Max)".
	Name() string
	// Refresh exchanges the refresh token for a new credential. It returns
	// ErrOAuthUnauthorized when the credential is terminally dead.
	Refresh(ctx context.Context, current *Credential, env map[string]string) (*Credential, error)
	// ToAuth derives request auth from a valid credential. Providers differ:
	// most pass the access token as the API key, Kimi sends an Authorization
	// header instead.
	ToAuth(cred *Credential) *Auth
}

// oauthProviders maps provider id to its OAuth implementation.
var oauthProviders = map[string]OAuthProvider{
	"anthropic":    anthropicOAuth{},
	"kimi-coding":  kimiCodingOAuth{},
	"openai-codex": openAICodexOAuth{},
}

// OAuthProviderFor returns the OAuth implementation for a provider id.
func OAuthProviderFor(providerID string) (OAuthProvider, bool) {
	provider, ok := oauthProviders[providerID]
	return provider, ok
}

// OAuthProviderIDs returns the provider ids with a ported OAuth flow, sorted.
func OAuthProviderIDs() []string {
	ids := make([]string, 0, len(oauthProviders))
	for id := range oauthProviders {
		ids = append(ids, id)
	}
	sortStrings(ids)
	return ids
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j-1] > values[j]; j-- {
			values[j-1], values[j] = values[j], values[j-1]
		}
	}
}

// expiresSoon reports whether a credential needs refreshing.
func expiresSoon(cred *Credential) bool {
	return time.Now().Add(oauthMinimumValidity).UnixMilli() >= cred.Expires
}

// refreshOAuthCredential refreshes a stored credential under the store lock.
//
// The expiry is re-checked inside Modify because another process may have
// rotated the token while this one waited for the lock, exactly as pi's
// double-checked locking does.
func (c *Catalog) refreshOAuthCredential(ctx context.Context, providerID string, provider OAuthProvider) (*Credential, error) {
	if c.credentials == nil {
		return nil, fmt.Errorf("catalog: no credential store to refresh %s", providerID)
	}

	var refreshErr error
	post, err := c.credentials.Modify(providerID, func(current *Credential) (*Credential, error) {
		// Logged out while waiting for the lock.
		if current == nil || current.Type != "oauth" {
			return nil, nil
		}
		// Another process refreshed it; keep theirs.
		if !expiresSoon(current) {
			return nil, nil
		}
		refreshed, err := provider.Refresh(ctx, current, c.envMap())
		if err != nil {
			refreshErr = err
			// Returning nil leaves the stored credential untouched, so a
			// transient failure does not destroy a recoverable token.
			return nil, nil
		}
		refreshed.Type = "oauth"
		return refreshed, nil
	})
	if refreshErr != nil {
		return nil, refreshErr
	}
	if err != nil {
		return nil, err
	}
	if post == nil || post.Type != "oauth" {
		return nil, ErrOAuthUnauthorized
	}
	return post, nil
}

// envMap exposes the catalog's environment as a map for OAuth hosts that honor
// overrides. Only the keys the flows consult are included.
func (c *Catalog) envMap() map[string]string {
	env := map[string]string{}
	for _, name := range []string{
		"KIMI_CODE_OAUTH_HOST",
		"KIMI_OAUTH_HOST",
		"PI_OAUTH_CALLBACK_HOST",
		"GOSHCODER_OAUTH_CALLBACK_HOST",
	} {
		if value := c.env(name); value != "" {
			env[name] = value
		}
	}
	return env
}

// ---------------------------------------------------------------------------
// Token request helpers
// ---------------------------------------------------------------------------

// tokenResponse is the standard OAuth token payload.
type tokenResponse struct {
	AccessToken      string `json:"access_token"`
	RefreshToken     string `json:"refresh_token"`
	ExpiresIn        int64  `json:"expires_in"`
	Error            string `json:"error"`
	ErrorDescription string `json:"error_description"`
}

// postForm posts application/x-www-form-urlencoded and returns the status and
// body. Transport failures are errors; HTTP status is left to the caller, since
// a 401 is meaningful rather than exceptional.
func postForm(ctx context.Context, endpoint string, form url.Values) (status int, body []byte, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	return doTokenRequest(req)
}

// postJSON posts a JSON body and returns the status and body.
func postJSON(ctx context.Context, endpoint string, payload any) (status int, body []byte, err error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return 0, nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(encoded)))
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	return doTokenRequest(req)
}

func doTokenRequest(req *http.Request) (int, []byte, error) {
	resp, err := (&http.Client{Timeout: oauthRequestTimeout}).Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	// Token responses are small; the cap only guards against a hostile server.
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return resp.StatusCode, nil, err
	}
	return resp.StatusCode, body, nil
}

// isUnauthorized reports whether a token response means the credential is dead
// rather than the request being transiently unlucky.
func isUnauthorized(status int, parsed *tokenResponse) bool {
	return status == http.StatusUnauthorized ||
		status == http.StatusForbidden ||
		(parsed != nil && parsed.Error == "invalid_grant")
}

// isRetryableStatus reports whether a failed token request is worth retrying.
func isRetryableStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

// credentialFrom builds a stored credential from a token response. skew is
// subtracted from the expiry so a token is refreshed slightly before the
// provider considers it expired.
func credentialFrom(parsed *tokenResponse, skew time.Duration) (*Credential, error) {
	if parsed.AccessToken == "" || parsed.RefreshToken == "" || parsed.ExpiresIn <= 0 {
		return nil, fmt.Errorf("token response is missing fields")
	}
	expires := time.Now().Add(time.Duration(parsed.ExpiresIn)*time.Second - skew)
	return &Credential{
		Type:    "oauth",
		Access:  parsed.AccessToken,
		Refresh: parsed.RefreshToken,
		Expires: expires.UnixMilli(),
	}, nil
}

// tokenError builds a descriptive error for a failed token request.
func tokenError(providerName, operation string, status int, body []byte, parsed *tokenResponse) error {
	if isUnauthorized(status, parsed) {
		detail := ""
		if parsed != nil && parsed.ErrorDescription != "" {
			detail = ": " + parsed.ErrorDescription
		}
		return fmt.Errorf("%s token %s unauthorized (status %d)%s: %w",
			providerName, operation, status, detail, ErrOAuthUnauthorized)
	}
	return fmt.Errorf("%s token %s failed (status %d): %s",
		providerName, operation, status, strings.TrimSpace(truncate(string(body), 500)))
}

func truncate(text string, limit int) string {
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}

// ---------------------------------------------------------------------------
// Anthropic (Claude Pro/Max)
// ---------------------------------------------------------------------------

const (
	anthropicOAuthClientID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
	anthropicAuthorizeURL  = "https://claude.ai/oauth/authorize"
	anthropicCallbackPort  = 53692
	anthropicCallbackPath  = "/callback"
	anthropicOAuthScopes   = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
	anthropicRefreshSkew   = 5 * time.Minute
)

// anthropicTokenURL is a var so tests can point it at a fake token endpoint.
var anthropicTokenURL = "https://platform.claude.com/v1/oauth/token"

type anthropicOAuth struct{}

func (anthropicOAuth) Name() string { return "Anthropic (Claude Pro/Max)" }

func (a anthropicOAuth) Refresh(ctx context.Context, current *Credential, _ map[string]string) (*Credential, error) {
	status, body, err := postJSON(ctx, anthropicTokenURL, map[string]string{
		"grant_type":    "refresh_token",
		"client_id":     anthropicOAuthClientID,
		"refresh_token": current.Refresh,
	})
	if err != nil {
		return nil, fmt.Errorf("anthropic token refresh request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		return nil, tokenError(a.Name(), "refresh", status, body, &parsed)
	}
	cred, err := credentialFrom(&parsed, anthropicRefreshSkew)
	if err != nil {
		return nil, fmt.Errorf("anthropic token refresh: %w", err)
	}
	return cred, nil
}

func (anthropicOAuth) ToAuth(cred *Credential) *Auth {
	return &Auth{APIKey: cred.Access, Source: "OAuth"}
}

// ---------------------------------------------------------------------------
// Kimi Code (subscription)
// ---------------------------------------------------------------------------

const (
	kimiOAuthClientID     = "17e5f671-d194-4dfb-9706-5516cb48c098"
	kimiDefaultOAuthHost  = "https://auth.kimi.com"
	kimiRefreshMaxRetries = 3
)

type kimiCodingOAuth struct{}

func (kimiCodingOAuth) Name() string { return "Kimi Code (subscription)" }

// kimiOAuthHost resolves the OAuth host, honoring pi's override variables.
func kimiOAuthHost(env map[string]string) string {
	host := env["KIMI_CODE_OAUTH_HOST"]
	if host == "" {
		host = env["KIMI_OAUTH_HOST"]
	}
	if host == "" {
		host = kimiDefaultOAuthHost
	}
	return strings.TrimRight(host, "/")
}

func (k kimiCodingOAuth) Refresh(ctx context.Context, current *Credential, env map[string]string) (*Credential, error) {
	endpoint := kimiOAuthHost(env) + "/api/oauth/token"
	form := url.Values{
		"client_id":     {kimiOAuthClientID},
		"grant_type":    {"refresh_token"},
		"refresh_token": {current.Refresh},
	}

	var lastErr error
	for attempt := 0; attempt <= kimiRefreshMaxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff, cancellable.
			delay := time.Duration(1<<(attempt-1)) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
			}
		}

		status, body, err := postForm(ctx, endpoint, form)
		if err != nil {
			lastErr = fmt.Errorf("kimi Code token refresh request failed: %w", err)
			continue
		}

		var parsed tokenResponse
		_ = json.Unmarshal(body, &parsed)
		if status >= 200 && status < 300 {
			cred, err := credentialFrom(&parsed, 0)
			if err != nil {
				return nil, fmt.Errorf("kimi Code token refresh: %w", err)
			}
			return cred, nil
		}
		// A dead credential is terminal: no point retrying.
		if isUnauthorized(status, &parsed) {
			return nil, tokenError(k.Name(), "refresh", status, body, &parsed)
		}
		if isRetryableStatus(status) && attempt < kimiRefreshMaxRetries {
			lastErr = tokenError(k.Name(), "refresh", status, body, &parsed)
			continue
		}
		return nil, tokenError(k.Name(), "refresh", status, body, &parsed)
	}
	if lastErr == nil {
		lastErr = errors.New("kimi Code token refresh failed")
	}
	return nil, lastErr
}

// ToAuth sends the access token as an Authorization header rather than an API
// key, which is what the Kimi coding endpoint expects.
func (kimiCodingOAuth) ToAuth(cred *Credential) *Auth {
	bearer := "Bearer " + cred.Access
	return &Auth{
		APIKey:  cred.Access,
		Headers: llm.ProviderHeaders{"Authorization": &bearer},
		Source:  "OAuth",
	}
}

// ---------------------------------------------------------------------------
// OpenAI Codex (ChatGPT Plus/Pro)
// ---------------------------------------------------------------------------

const (
	codexOAuthClientID     = "app_EMoamEEZ73f0CkXaXp7hrann"
	codexAuthBaseURL       = "https://auth.openai.com"
	codexAuthorizeURL      = codexAuthBaseURL + "/oauth/authorize"
	codexRedirectURI       = "http://localhost:1455/auth/callback"
	codexDeviceVerifyURI   = codexAuthBaseURL + "/codex/device"
	codexDeviceRedirectURI = codexAuthBaseURL + "/deviceauth/callback"
	codexOAuthScope        = "openid profile email offline_access"
	// codexJWTClaimPath is the namespaced claim carrying the account id.
	codexJWTClaimPath = "https://api.openai.com/auth"
)

// Codex endpoints are vars so tests can point them at a fake server.
var (
	codexTokenURL          = codexAuthBaseURL + "/oauth/token"
	codexDeviceUserCodeURL = codexAuthBaseURL + "/api/accounts/deviceauth/usercode"
	codexDeviceTokenURL    = codexAuthBaseURL + "/api/accounts/deviceauth/token"
)

type openAICodexOAuth struct{}

func (openAICodexOAuth) Name() string { return "OpenAI (ChatGPT Plus/Pro)" }

func (o openAICodexOAuth) Refresh(ctx context.Context, current *Credential, _ map[string]string) (*Credential, error) {
	status, body, err := postForm(ctx, codexTokenURL, url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {current.Refresh},
		"client_id":     {codexOAuthClientID},
	})
	if err != nil {
		return nil, fmt.Errorf("OpenAI Codex token refresh request failed: %w", err)
	}

	var parsed tokenResponse
	_ = json.Unmarshal(body, &parsed)
	if status < 200 || status >= 300 {
		return nil, tokenError(o.Name(), "refresh", status, body, &parsed)
	}
	cred, err := credentialFrom(&parsed, 0)
	if err != nil {
		return nil, fmt.Errorf("OpenAI Codex token refresh: %w", err)
	}
	// The account id is derived from the access token and must travel with the
	// credential: the wire protocol sends it as a header.
	accountID, err := codexAccountID(cred.Access)
	if err != nil {
		return nil, err
	}
	if err := cred.SetExtra("accountId", accountID); err != nil {
		return nil, err
	}
	return cred, nil
}

func (openAICodexOAuth) ToAuth(cred *Credential) *Auth {
	return &Auth{APIKey: cred.Access, Source: "OAuth"}
}

// codexAccountID extracts chatgpt_account_id from the access token's JWT
// claims.
func codexAccountID(accessToken string) (string, error) {
	payload, err := decodeJWTPayload(accessToken)
	if err != nil {
		return "", fmt.Errorf("failed to extract accountId from token: %w", err)
	}
	claim, ok := payload[codexJWTClaimPath].(map[string]any)
	if !ok {
		return "", errors.New("failed to extract accountId from token: no auth claim")
	}
	accountID, ok := claim["chatgpt_account_id"].(string)
	if !ok || accountID == "" {
		return "", errors.New("failed to extract accountId from token: no account id")
	}
	return accountID, nil
}

// decodeJWTPayload decodes the claim set of a JWT without verifying it. The
// token is issued to this client over TLS and only read for the account id, so
// signature verification is the issuer's concern.
func decodeJWTPayload(token string) (map[string]any, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errors.New("not a JWT")
	}
	// JWT payloads are base64url without padding.
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(decoded, &payload); err != nil {
		return nil, err
	}
	return payload, nil
}
