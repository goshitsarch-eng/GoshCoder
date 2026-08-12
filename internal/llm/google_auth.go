package llm

// Google Cloud credential handling for the google-vertex protocol.
//
// DEVIATION: pi delegates this to google-auth-library (via @google/genai).
// GoshCoder has no third-party dependencies, so the flows pi actually relies on
// are implemented here against stdlib crypto:
//
//   - a service account key file (GOOGLE_APPLICATION_CREDENTIALS): a signed JWT
//     assertion is exchanged for an access token
//   - an authorized user / gcloud file with a refresh token: the refresh token
//     is exchanged for an access token
//   - a pre-fetched token (GOOGLE_OAUTH_ACCESS_TOKEN), for callers that already
//     ran `gcloud auth print-access-token`
//
// Not covered: GCE/Cloud Run metadata-server credentials and workload identity
// federation. Those environments should pass GOOGLE_OAUTH_ACCESS_TOKEN.

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// googleCloudPlatformScope is the OAuth scope Vertex AI requires.
const googleCloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

// googleTokenEndpoint is the OAuth2 token exchange endpoint.
const googleTokenEndpoint = "https://oauth2.googleapis.com/token"

// googleTokenSkew renews a token slightly before it actually expires.
const googleTokenSkew = 60 * time.Second

// googleAccessToken is a cached bearer token.
type googleAccessToken struct {
	Token     string
	ExpiresAt time.Time
}

func (t *googleAccessToken) valid() bool {
	return t != nil && t.Token != "" && time.Now().Add(googleTokenSkew).Before(t.ExpiresAt)
}

// googleTokenCache caches access tokens per credential source, so repeated
// requests do not re-run the token exchange.
var (
	googleTokenMu    sync.Mutex
	googleTokenCache = map[string]*googleAccessToken{}
)

// googleCredentialFile is the subset of a credentials JSON file that matters
// for the supported flows.
type googleCredentialFile struct {
	Type string `json:"type"`

	// service_account
	ClientEmail  string `json:"client_email"`
	PrivateKey   string `json:"private_key"`
	PrivateKeyID string `json:"private_key_id"`
	TokenURI     string `json:"token_uri"`
	ProjectID    string `json:"project_id"`

	// authorized_user
	ClientID       string `json:"client_id"`
	ClientSecret   string `json:"client_secret"`
	RefreshToken   string `json:"refresh_token"`
	QuotaProjectID string `json:"quota_project_id"`
}

// resolveGoogleAccessToken obtains a bearer token for Vertex AI. Sources are
// tried in the order pi's auth library uses: an explicit token, then the
// credentials file, then the well-known gcloud location.
func resolveGoogleAccessToken(ctx context.Context, env map[string]string) (string, error) {
	if token := getProviderEnvValue("GOOGLE_OAUTH_ACCESS_TOKEN", env); token != "" {
		return token, nil
	}

	path := getProviderEnvValue("GOOGLE_APPLICATION_CREDENTIALS", env)
	if path == "" {
		path = wellKnownGoogleCredentialsPath()
	}
	if path == "" {
		return "", fmt.Errorf("no Google credentials found. Set GOOGLE_APPLICATION_CREDENTIALS to a service account or authorized user key file, or set GOOGLE_OAUTH_ACCESS_TOKEN")
	}

	googleTokenMu.Lock()
	cached := googleTokenCache[path]
	googleTokenMu.Unlock()
	if cached.valid() {
		return cached.Token, nil
	}

	creds, err := loadGoogleCredentialFile(path)
	if err != nil {
		return "", err
	}
	token, err := exchangeGoogleCredentials(ctx, creds)
	if err != nil {
		return "", err
	}

	googleTokenMu.Lock()
	googleTokenCache[path] = token
	googleTokenMu.Unlock()
	return token.Token, nil
}

// wellKnownGoogleCredentialsPath returns the default gcloud ADC file location,
// or "" when it does not exist.
func wellKnownGoogleCredentialsPath() string {
	var dir string
	if appData := os.Getenv("APPDATA"); appData != "" {
		dir = appData
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		dir = home + "/.config"
	}
	path := dir + "/gcloud/application_default_credentials.json"
	if fileExistsAt(path) {
		return path
	}
	return ""
}

func fileExistsAt(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func loadGoogleCredentialFile(path string) (*googleCredentialFile, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading Google credentials %s: %w", path, err)
	}
	var creds googleCredentialFile
	if err := json.Unmarshal(content, &creds); err != nil {
		return nil, fmt.Errorf("parsing Google credentials %s: %w", path, err)
	}
	return &creds, nil
}

// exchangeGoogleCredentials turns a credentials file into an access token.
func exchangeGoogleCredentials(ctx context.Context, creds *googleCredentialFile) (*googleAccessToken, error) {
	switch {
	case creds.Type == "service_account" || (creds.PrivateKey != "" && creds.ClientEmail != ""):
		return exchangeGoogleServiceAccount(ctx, creds)
	case creds.Type == "authorized_user" || creds.RefreshToken != "":
		return exchangeGoogleRefreshToken(ctx, creds)
	default:
		return nil, fmt.Errorf("unsupported Google credential type %q; set GOOGLE_OAUTH_ACCESS_TOKEN instead", creds.Type)
	}
}

// exchangeGoogleServiceAccount signs a JWT assertion and exchanges it for an
// access token (the two-legged OAuth flow).
func exchangeGoogleServiceAccount(ctx context.Context, creds *googleCredentialFile) (*googleAccessToken, error) {
	key, err := parseGoogleRSAPrivateKey(creds.PrivateKey)
	if err != nil {
		return nil, err
	}
	tokenURI := creds.TokenURI
	if tokenURI == "" {
		tokenURI = googleTokenEndpoint
	}

	now := time.Now()
	header := map[string]any{"alg": "RS256", "typ": "JWT"}
	if creds.PrivateKeyID != "" {
		header["kid"] = creds.PrivateKeyID
	}
	claims := map[string]any{
		"iss":   creds.ClientEmail,
		"scope": googleCloudPlatformScope,
		"aud":   tokenURI,
		"iat":   now.Unix(),
		"exp":   now.Add(time.Hour).Unix(),
	}

	assertion, err := signGoogleJWT(header, claims, key)
	if err != nil {
		return nil, err
	}

	form := url.Values{
		"grant_type": {"urn:ietf:params:oauth:grant-type:jwt-bearer"},
		"assertion":  {assertion},
	}
	return postGoogleTokenRequest(ctx, tokenURI, form)
}

// exchangeGoogleRefreshToken exchanges an authorized-user refresh token.
func exchangeGoogleRefreshToken(ctx context.Context, creds *googleCredentialFile) (*googleAccessToken, error) {
	if creds.ClientID == "" || creds.ClientSecret == "" || creds.RefreshToken == "" {
		return nil, fmt.Errorf("incomplete authorized_user credentials: client_id, client_secret and refresh_token are required")
	}
	tokenURI := creds.TokenURI
	if tokenURI == "" {
		tokenURI = googleTokenEndpoint
	}
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {creds.ClientID},
		"client_secret": {creds.ClientSecret},
		"refresh_token": {creds.RefreshToken},
	}
	return postGoogleTokenRequest(ctx, tokenURI, form)
}

// postGoogleTokenRequest performs the token exchange and parses the response.
func postGoogleTokenRequest(ctx context.Context, tokenURI string, form url.Values) (*googleAccessToken, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURI, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("Google token exchange failed: %w", err)
	}
	defer resp.Body.Close()

	var payload struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int64  `json:"expires_in"`
		Error       string `json:"error"`
		ErrorDesc   string `json:"error_description"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("parsing Google token response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 || payload.AccessToken == "" {
		detail := payload.ErrorDesc
		if detail == "" {
			detail = payload.Error
		}
		if detail == "" {
			detail = fmt.Sprintf("status %d", resp.StatusCode)
		}
		return nil, fmt.Errorf("Google token exchange failed: %s", detail)
	}

	expiresIn := payload.ExpiresIn
	if expiresIn <= 0 {
		expiresIn = 3600
	}
	return &googleAccessToken{
		Token:     payload.AccessToken,
		ExpiresAt: time.Now().Add(time.Duration(expiresIn) * time.Second),
	}, nil
}

// parseGoogleRSAPrivateKey decodes a PEM private key, accepting both PKCS#1 and
// PKCS#8 encodings (Google issues PKCS#8).
func parseGoogleRSAPrivateKey(pemKey string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(pemKey))
	if block == nil {
		return nil, fmt.Errorf("service account private_key is not valid PEM")
	}
	if key, err := x509.ParsePKCS1PrivateKey(block.Bytes); err == nil {
		return key, nil
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing service account private key: %w", err)
	}
	key, ok := parsed.(*rsa.PrivateKey)
	if !ok {
		return nil, fmt.Errorf("service account private key is not an RSA key")
	}
	return key, nil
}

// signGoogleJWT builds a compact RS256 JWT.
func signGoogleJWT(header, claims map[string]any, key *rsa.PrivateKey) (string, error) {
	encodeSegment := func(value map[string]any) (string, error) {
		encoded, err := json.Marshal(value)
		if err != nil {
			return "", err
		}
		return base64.RawURLEncoding.EncodeToString(encoded), nil
	}

	headerSegment, err := encodeSegment(header)
	if err != nil {
		return "", err
	}
	claimsSegment, err := encodeSegment(claims)
	if err != nil {
		return "", err
	}

	signingInput := headerSegment + "." + claimsSegment
	digest := crypto.SHA256.New()
	digest.Write([]byte(signingInput))
	signature, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, digest.Sum(nil))
	if err != nil {
		return "", fmt.Errorf("signing service account assertion: %w", err)
	}
	return signingInput + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

// ClearGoogleTokenCache drops cached access tokens. Exported for tests.
func ClearGoogleTokenCache() {
	googleTokenMu.Lock()
	defer googleTokenMu.Unlock()
	googleTokenCache = map[string]*googleAccessToken{}
}
