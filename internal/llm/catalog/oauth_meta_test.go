package catalog

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"goshcoder/internal/llm"
)

// metaServers is a fake Meta: the identity provider and the Model API are
// separate hosts there, and separate here, so a test can tell which one a
// request reached.
type metaServers struct {
	auth *httptest.Server
	api  *httptest.Server

	mu sync.Mutex
	// tokenResponses are replayed in order; the last one repeats.
	tokenResponses []tokenServerResponse
	// mintResponses are replayed in order; the last one repeats.
	mintResponses []tokenServerResponse
	tokenRequests []map[string]string
	mintRequests  []metaMintRequest
}

type metaMintRequest struct {
	Authorization string
	APIVersion    string
	Body          map[string]string
}

func newMetaServers(t *testing.T) *metaServers {
	t.Helper()
	m := &metaServers{}

	authMux := http.NewServeMux()
	authMux.HandleFunc("/oidc/device/authorization/", func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		m.mu.Lock()
		m.tokenRequests = append(m.tokenRequests, map[string]string{
			"__route":   "authorization",
			"client_id": r.PostForm.Get("client_id"),
		})
		m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"device_code":"device-code","user_code":"VGGF-VLQT",`+
			`"verification_uri":"https://auth.meta.com/oauth/device/",`+
			`"verification_uri_complete":"https://auth.meta.com/oauth/device/?code=VGGF-VLQT",`+
			`"expires_in":600,"interval":1}`)
	})
	authMux.HandleFunc("/oidc/device/token/", func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		fields := map[string]string{"__route": "token"}
		for key := range r.PostForm {
			fields[key] = r.PostForm.Get(key)
		}
		m.mu.Lock()
		index := 0
		for _, recorded := range m.tokenRequests {
			if recorded["__route"] == "token" {
				index++
			}
		}
		m.tokenRequests = append(m.tokenRequests, fields)
		if index >= len(m.tokenResponses) {
			index = len(m.tokenResponses) - 1
		}
		response := m.tokenResponses[index]
		m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.status)
		fmt.Fprint(w, response.body)
	})
	m.auth = httptest.NewServer(authMux)
	t.Cleanup(m.auth.Close)

	apiMux := http.NewServeMux()
	apiMux.HandleFunc("/muse-code/key", func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		body := map[string]string{}
		_ = json.Unmarshal(raw, &body)
		m.mu.Lock()
		index := len(m.mintRequests)
		m.mintRequests = append(m.mintRequests, metaMintRequest{
			Authorization: r.Header.Get("Authorization"),
			APIVersion:    r.Header.Get("x-api-version"),
			Body:          body,
		})
		if index >= len(m.mintResponses) {
			index = len(m.mintResponses) - 1
		}
		response := m.mintResponses[index]
		m.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(response.status)
		fmt.Fprint(w, response.body)
	})
	m.api = httptest.NewServer(apiMux)
	t.Cleanup(m.api.Close)

	previousAuth, previousAPI := metaAuthBaseURL, metaAPIBaseURL
	metaAuthBaseURL, metaAPIBaseURL = m.auth.URL, m.api.URL
	t.Cleanup(func() { metaAuthBaseURL, metaAPIBaseURL = previousAuth, previousAPI })
	return m
}

func (m *metaServers) withTokens(responses ...tokenServerResponse) *metaServers {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenResponses = responses
	return m
}

func (m *metaServers) withMints(responses ...tokenServerResponse) *metaServers {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.mintResponses = responses
	return m
}

func (m *metaServers) mint(index int) (metaMintRequest, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if index >= len(m.mintRequests) {
		return metaMintRequest{}, false
	}
	return m.mintRequests[index], true
}

func (m *metaServers) tokenRoutes() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var routes []string
	for _, request := range m.tokenRequests {
		routes = append(routes, request["__route"])
	}
	return routes
}

func mintedKey(apiKey string) tokenServerResponse {
	return tokenServerResponse{
		status: http.StatusOK,
		body:   fmt.Sprintf(`{"api_key":%q,"require_payment":false,"user_email":"a@example.com"}`, apiKey),
	}
}

// TestMetaLoginMintsAModelAPIKey covers the whole flow, which is two exchanges
// rather than one: the device grant yields an identity, and only the mint that
// follows yields the credential api.meta.ai will actually accept.
func TestMetaLoginMintsAModelAPIKey(t *testing.T) {
	servers := newMetaServers(t).
		withTokens(pendingDevice(), okToken("identity-token", "meta-refresh", 3600)).
		withMints(mintedKey("model-api-key"))

	interaction := &loginInteractionStub{}
	c, store := oauthCatalog(t, nil)
	cred, err := c.Login(t.Context(), "meta", interaction)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	// The key is what requests carry; the identity is only good for minting
	// the next one.
	if cred.Access != "model-api-key" || cred.Refresh != "identity-token" {
		t.Fatalf("credential = %#v", cred)
	}
	if refresh, ok := cred.Extra(metaRefreshTokenExtra); !ok || refresh != "meta-refresh" {
		t.Fatalf("stored refresh token = %q, ok = %v", refresh, ok)
	}

	event, ok := interaction.event("device_code")
	if !ok || event.UserCode != "VGGF-VLQT" {
		t.Fatalf("device_code event = %#v, ok = %v", event, ok)
	}

	mint, ok := servers.mint(0)
	if !ok {
		t.Fatal("no key was ever minted")
	}
	if mint.Authorization != "Bearer identity-token" || mint.Body["dca_token"] != "identity-token" {
		t.Fatalf("mint request = %#v", mint)
	}
	if mint.APIVersion != metaAPIVersion {
		t.Fatalf("x-api-version = %q", mint.APIVersion)
	}

	stored, err := store.Read("meta")
	if err != nil || stored == nil || stored.Type != "oauth" || stored.Access != "model-api-key" {
		t.Fatalf("stored credential = %#v, err = %v", stored, err)
	}
	// The refresh token rides in an extra, so it only survives if extras
	// round-trip through auth.json. Without it every re-mint that Meta rejects
	// would become a fresh login.
	if refresh, ok := stored.Extra(metaRefreshTokenExtra); !ok || refresh != "meta-refresh" {
		t.Fatalf("refresh token did not survive the write: %q, ok = %v", refresh, ok)
	}
}

// TestMetaRefreshRemintsWithoutSpendingTheRefreshToken covers the daily key
// rotation, which is the common case: the identity is still good, so a new key
// costs one request and the refresh token stays untouched.
func TestMetaRefreshRemintsWithoutSpendingTheRefreshToken(t *testing.T) {
	servers := newMetaServers(t).
		withTokens(okToken("should-not-be-used", "unused", 3600)).
		withMints(mintedKey("second-key"))

	current := metaCredential("identity-token", "meta-refresh",
		time.Now().Add(24*time.Hour).UnixMilli(), "first-key")
	cred, err := metaOAuth{}.Refresh(t.Context(), current, nil)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if cred.Access != "second-key" || cred.Refresh != "identity-token" {
		t.Fatalf("credential = %#v", cred)
	}
	if routes := servers.tokenRoutes(); len(routes) != 0 {
		t.Fatalf("a re-mint reached the token endpoint: %v", routes)
	}
}

// TestMetaRefreshRenewsARejectedIdentity is the other half: once Meta stops
// accepting the identity, the refresh token buys a new one and the key is
// minted from that, with no new login.
func TestMetaRefreshRenewsARejectedIdentity(t *testing.T) {
	servers := newMetaServers(t).
		withTokens(okToken("second-identity", "second-refresh", 3600)).
		withMints(
			tokenServerResponse{status: http.StatusUnauthorized, body: `{"title":"Authentication Error","status":401}`},
			mintedKey("renewed-key"),
		)

	current := metaCredential("dead-identity", "meta-refresh",
		time.Now().Add(24*time.Hour).UnixMilli(), "old-key")
	cred, err := metaOAuth{}.Refresh(t.Context(), current, nil)
	if err != nil {
		t.Fatalf("Refresh: %v", err)
	}
	if cred.Access != "renewed-key" || cred.Refresh != "second-identity" {
		t.Fatalf("credential = %#v", cred)
	}
	if refresh, _ := cred.Extra(metaRefreshTokenExtra); refresh != "second-refresh" {
		t.Fatalf("refresh token = %q, want the rotated one", refresh)
	}
	second, ok := servers.mint(1)
	if !ok || second.Authorization != "Bearer second-identity" {
		t.Fatalf("second mint = %#v, ok = %v", second, ok)
	}
}

// TestMetaRefreshIsTerminalWithoutARefreshToken: a credential that cannot be
// renewed must say so, because the caller's alternative -- retrying every turn
// against a login that is gone -- is worse than telling the user to log in.
func TestMetaRefreshIsTerminalWithoutARefreshToken(t *testing.T) {
	newMetaServers(t).
		withTokens(okToken("unused", "unused", 3600)).
		withMints(tokenServerResponse{status: http.StatusUnauthorized, body: `{"status":401}`})

	current := &Credential{Type: "oauth", Refresh: "dead-identity", Access: "old-key"}
	_, err := metaOAuth{}.Refresh(t.Context(), current, nil)
	if !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("Refresh error = %v, want ErrOAuthUnauthorized", err)
	}
}

// TestMetaRefreshTreatsAMissingTokenEndpointAsADeadLogin covers Meta answering
// an unusable refresh token with a bare 404 rather than invalid_grant. Read as
// a transport failure it would be retried forever; it means log in again.
func TestMetaRefreshTreatsAMissingTokenEndpointAsADeadLogin(t *testing.T) {
	newMetaServers(t).
		withTokens(tokenServerResponse{status: http.StatusNotFound, body: ``}).
		withMints(tokenServerResponse{status: http.StatusUnauthorized, body: `{"status":401}`})

	current := metaCredential("dead-identity", "meta-refresh", 0, "old-key")
	_, err := metaOAuth{}.Refresh(t.Context(), current, nil)
	if !errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatalf("Refresh error = %v, want ErrOAuthUnauthorized", err)
	}
}

// TestMetaMintReportsAnAccountThatNeedsSetup: Meta mints nothing for an
// account that has not finished signup and says where to finish. Passing that
// through is the difference between a fixable message and "not configured".
func TestMetaMintReportsAnAccountThatNeedsSetup(t *testing.T) {
	newMetaServers(t).withMints(tokenServerResponse{
		status: http.StatusOK,
		body:   `{"api_key":"","require_payment":true,"action_url":"https://dev.meta.ai/billing"}`,
	})

	_, err := metaOAuth{}.mint(t.Context(), "identity-token")
	if err == nil || !strings.Contains(err.Error(), "https://dev.meta.ai/billing") {
		t.Fatalf("mint error = %v, want it to name where to finish setup", err)
	}
	if errors.Is(err, ErrOAuthUnauthorized) {
		t.Fatal("an unfinished signup is not a dead credential; logging in again fixes nothing")
	}
}

// TestMetaAuthTravelsAsABearerHeaderOnly is the load-bearing detail of the
// whole provider: api.meta.ai speaks the Anthropic Messages shape but
// authenticates with Authorization. Leaving the key in APIKey would make the
// streamer send x-api-key as well.
func TestMetaAuthTravelsAsABearerHeaderOnly(t *testing.T) {
	auth := metaOAuth{}.ToAuth(&Credential{Type: "oauth", Access: "model-api-key"})
	if auth.APIKey != "" {
		t.Fatalf("APIKey = %q, want it empty so no x-api-key is sent", auth.APIKey)
	}
	header, ok := auth.Headers["Authorization"]
	if !ok || header == nil || *header != "Bearer model-api-key" {
		t.Fatalf("headers = %#v", auth.Headers)
	}
}

// TestMetaAPIKeyResolvesTheSameWayAsTheOAuthKey: a key pasted into
// `goshcoder auth set meta`, or exported as META_API_KEY, is the same kind of
// credential the mint returns and has to travel the same way.
func TestMetaAPIKeyResolvesTheSameWayAsTheOAuthKey(t *testing.T) {
	c, store := oauthCatalog(t, map[string]string{"META_API_KEY": "env-key"})

	auth, ok := c.ResolveAuth("meta")
	if !ok || auth.APIKey != "" || auth.Source != "META_API_KEY" {
		t.Fatalf("ambient auth = %#v, ok = %v", auth, ok)
	}
	if header := auth.Headers["Authorization"]; header == nil || *header != "Bearer env-key" {
		t.Fatalf("ambient headers = %#v", auth.Headers)
	}

	if _, err := store.Modify("meta", func(*Credential) (*Credential, error) {
		return &Credential{Type: "api_key", Key: "stored-key"}, nil
	}); err != nil {
		t.Fatal(err)
	}
	auth, ok = c.ResolveAuth("meta")
	if !ok {
		t.Fatal("a stored key did not configure the provider")
	}
	if header := auth.Headers["Authorization"]; header == nil || *header != "Bearer stored-key" {
		t.Fatalf("stored headers = %#v", auth.Headers)
	}
}

// TestMetaCredentialExpiresAtTheSoonerDeadline: the key is re-minted daily and
// the identity dies on its own schedule. Either ending is a reason to refresh,
// so the stored expiry has to be whichever comes first.
func TestMetaCredentialExpiresAtTheSoonerDeadline(t *testing.T) {
	identityExpires := time.Now().Add(time.Hour).UnixMilli()
	cred := metaCredential("identity", "refresh", identityExpires, "key")
	if cred.Expires != identityExpires {
		t.Fatalf("expires = %d, want the identity's earlier deadline %d", cred.Expires, identityExpires)
	}
	if recorded, _ := cred.Extra(metaIdentityExpiresExtra); recorded != strconv.FormatInt(identityExpires, 10) {
		t.Fatalf("recorded identity expiry = %q", recorded)
	}

	distant := time.Now().Add(30 * 24 * time.Hour).UnixMilli()
	cred = metaCredential("identity", "refresh", distant, "key")
	if cred.Expires >= distant {
		t.Fatalf("expires = %d, want the daily re-mint deadline instead of %d", cred.Expires, distant)
	}
}

// TestMetaModelsAreUsable checks the catalog entry the provider is useless
// without: the wrong api or base URL here fails at the first request.
func TestMetaModelsAreUsable(t *testing.T) {
	c, _ := oauthCatalog(t, nil)
	provider := c.Provider("meta")
	if provider == nil {
		t.Fatal("meta is not a known provider")
	}
	model := provider.Model("muse-spark-1.2")
	if model == nil {
		t.Fatal("meta has no muse-spark-1.2")
	}
	if model.API != llm.API("anthropic-messages") {
		t.Fatalf("api = %q", model.API)
	}
	if model.BaseURL != "https://api.meta.ai" {
		t.Fatalf("baseUrl = %q", model.BaseURL)
	}
	if _, ok := llm.GetStreamer(model.API); !ok {
		t.Fatalf("no streamer for %q; the model could not be used", model.API)
	}
	if model.ContextWindow <= 0 || model.MaxTokens <= 0 {
		t.Fatalf("model = %#v", model)
	}
}

// TestExtraCatalogIsNotShadowed guards the merge rule in data.go. The
// generated catalog wins on a collision, so the day pi ships one of these
// models its definition takes over and this build's own entry becomes dead
// data that still reads as authoritative. Failing here is the prompt to
// delete it.
func TestExtraCatalogIsNotShadowed(t *testing.T) {
	if len(shadowedExtras) != 0 {
		t.Fatalf("catalog_extra.json duplicates generated models: %v; delete them from catalog_extra.json",
			shadowedExtras)
	}
	// And what it declares must actually be reachable.
	if builtin.models["meta"]["muse-spark-1.2"] == nil {
		t.Fatal("catalog_extra.json did not reach the merged catalog")
	}
}
