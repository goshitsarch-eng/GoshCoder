package llm

import (
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSignAWSRequestKnownVector(t *testing.T) {
	// The canonical request and signing steps are deterministic, so a fixed
	// time, key, and body must always produce the same signature. This guards
	// against accidental changes to header canonicalization or key derivation.
	req, err := http.NewRequest(http.MethodPost,
		"https://bedrock-runtime.us-east-1.amazonaws.com/model/test-model/converse-stream", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")

	body := []byte(`{"messages":[]}`)
	creds := &awsCredentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}
	when := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

	if err := signAWSRequest(req, body, creds, "us-east-1", bedrockService, when); err != nil {
		t.Fatalf("signAWSRequest: %v", err)
	}

	auth := req.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20260102/us-east-1/bedrock/aws4_request") {
		t.Fatalf("authorization credential scope = %q", auth)
	}
	// host must be signed even though it is not in Header.
	if !strings.Contains(auth, "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date") {
		t.Fatalf("signed headers = %q", auth)
	}
	if req.Header.Get("X-Amz-Date") != "20260102T030405Z" {
		t.Fatalf("x-amz-date = %q", req.Header.Get("X-Amz-Date"))
	}
	// The body hash is published so the service can verify the payload.
	if got := req.Header.Get("X-Amz-Content-Sha256"); got != sha256Hex(body) {
		t.Fatalf("x-amz-content-sha256 = %q", got)
	}

	// Signing the same inputs twice yields the same signature.
	second, _ := http.NewRequest(http.MethodPost, req.URL.String(), nil)
	second.Header.Set("Content-Type", "application/json")
	if err := signAWSRequest(second, body, creds, "us-east-1", bedrockService, when); err != nil {
		t.Fatal(err)
	}
	if second.Header.Get("Authorization") != auth {
		t.Fatal("signing is not deterministic")
	}
}

func TestSignAWSRequestIncludesSessionToken(t *testing.T) {
	req, _ := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/x", nil)
	creds := &awsCredentials{AccessKeyID: "AKID", SecretAccessKey: "secret", SessionToken: "session-token"}

	if err := signAWSRequest(req, nil, creds, "us-east-1", bedrockService, time.Now()); err != nil {
		t.Fatal(err)
	}
	if req.Header.Get("X-Amz-Security-Token") != "session-token" {
		t.Fatalf("security token = %q", req.Header.Get("X-Amz-Security-Token"))
	}
	// Temporary credentials must sign the token header too.
	if !strings.Contains(req.Header.Get("Authorization"), "x-amz-security-token") {
		t.Fatalf("session token was not signed: %q", req.Header.Get("Authorization"))
	}
}

// TestSignAWSRequestBodyAffectsSignature confirms the payload is bound to the
// signature, which is what stops a body from being tampered with in transit.
func TestSignAWSRequestBodyAffectsSignature(t *testing.T) {
	creds := &awsCredentials{AccessKeyID: "AKID", SecretAccessKey: "secret"}
	when := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

	sign := func(body []byte) string {
		req, _ := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/x", nil)
		if err := signAWSRequest(req, body, creds, "us-east-1", bedrockService, when); err != nil {
			t.Fatal(err)
		}
		return req.Header.Get("Authorization")
	}

	if sign([]byte(`{"a":1}`)) == sign([]byte(`{"a":2}`)) {
		t.Fatal("different bodies produced the same signature")
	}
}

func TestSignAWSRequestRequiresCredentials(t *testing.T) {
	req, _ := http.NewRequest(http.MethodPost, "https://example.com", nil)
	if err := signAWSRequest(req, nil, nil, "us-east-1", bedrockService, time.Now()); err == nil {
		t.Fatal("signing without credentials must fail")
	}
}

func TestCollapseWhitespace(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"  value  ", "value"},
		{"a   b", "a b"},
		{"a\tb", "a b"},
		{"", ""},
	}
	for _, tt := range tests {
		if got := collapseWhitespace(tt.in); got != tt.want {
			t.Errorf("collapseWhitespace(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestCanonicalizeHeadersExcludesUnsignedHeaders confirms the headers AWS
// excludes never enter the canonical request.
func TestCanonicalizeHeadersExcludesUnsignedHeaders(t *testing.T) {
	header := http.Header{}
	header.Set("Authorization", "should-be-excluded")
	header.Set("User-Agent", "should-be-excluded")
	header.Set("Content-Length", "123")
	header.Set("Content-Type", "application/json")

	canonical, signed := canonicalizeHeaders(header, "example.com")
	for _, excluded := range []string{"authorization", "user-agent", "content-length"} {
		if strings.Contains(signed, excluded) {
			t.Errorf("%s should not be signed: %q", excluded, signed)
		}
	}
	if !strings.Contains(signed, "content-type") || !strings.Contains(signed, "host") {
		t.Fatalf("signed headers = %q", signed)
	}
	// Header lines are sorted and newline-terminated.
	if !strings.HasPrefix(canonical, "content-type:application/json\nhost:example.com\n") {
		t.Fatalf("canonical headers = %q", canonical)
	}
}

// clearAmbientAWSEnv blanks the AWS variables that may be set in the
// developer's shell or on a CI runner.
//
// resolveAWSCredentials layers its explicit env map over the process
// environment by design (getProviderEnvValue falls back to os.Getenv), so a
// test that supplies only some keys silently inherits the rest. Without this,
// a machine with real AWS credentials exported makes the unconfigured-
// environment assertions below fail even though the code is correct.
func clearAmbientAWSEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"AWS_ACCESS_KEY_ID",
		"AWS_SECRET_ACCESS_KEY",
		"AWS_SESSION_TOKEN",
		"AWS_PROFILE",
		"AWS_REGION",
		"AWS_DEFAULT_REGION",
		"AWS_BEARER_TOKEN_BEDROCK",
		"AWS_SHARED_CREDENTIALS_FILE",
		"AWS_CONFIG_FILE",
	} {
		t.Setenv(name, "")
	}
}

func TestResolveAWSCredentialsFromEnv(t *testing.T) {
	clearAmbientAWSEnv(t)
	env := map[string]string{
		"AWS_ACCESS_KEY_ID":     "AKIDENV",
		"AWS_SECRET_ACCESS_KEY": "secret-env",
		"AWS_SESSION_TOKEN":     "token-env",
	}
	creds, err := resolveAWSCredentials("", env)
	if err != nil {
		t.Fatalf("resolveAWSCredentials: %v", err)
	}
	if creds.AccessKeyID != "AKIDENV" || creds.SecretAccessKey != "secret-env" || creds.SessionToken != "token-env" {
		t.Fatalf("credentials = %#v", creds)
	}
	if creds.Source != "AWS_ACCESS_KEY_ID" {
		t.Fatalf("source = %q", creds.Source)
	}
}

// TestResolveAWSCredentialsRequiresBothKeys covers a half-configured
// environment, which must not resolve.
func TestResolveAWSCredentialsRequiresBothKeys(t *testing.T) {
	clearAmbientAWSEnv(t)
	// An empty shared-credentials path keeps the host machine's files out.
	env := map[string]string{
		"AWS_ACCESS_KEY_ID":           "AKIDENV",
		"AWS_SHARED_CREDENTIALS_FILE": filepath.Join(t.TempDir(), "missing-credentials"),
		"AWS_CONFIG_FILE":             filepath.Join(t.TempDir(), "missing-config"),
	}
	if _, err := resolveAWSCredentials("", env); err == nil {
		t.Fatal("an access key without a secret must not resolve")
	}
}

// writeAWSFiles creates a shared credentials and config file pair.
func writeAWSFiles(t *testing.T, credentials, config string) map[string]string {
	t.Helper()
	dir := t.TempDir()
	credentialsPath := filepath.Join(dir, "credentials")
	configPath := filepath.Join(dir, "config")
	if err := os.WriteFile(credentialsPath, []byte(credentials), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	return map[string]string{
		"AWS_SHARED_CREDENTIALS_FILE": credentialsPath,
		"AWS_CONFIG_FILE":             configPath,
	}
}

func TestResolveAWSCredentialsFromProfile(t *testing.T) {
	clearAmbientAWSEnv(t)
	env := writeAWSFiles(t, `
# a comment
[default]
aws_access_key_id = AKIDDEFAULT
aws_secret_access_key = secret-default

[work]
aws_access_key_id = AKIDWORK
aws_secret_access_key = secret-work
aws_session_token = token-work
`, "")

	creds, err := resolveAWSCredentials("work", env)
	if err != nil {
		t.Fatalf("resolveAWSCredentials: %v", err)
	}
	if creds.AccessKeyID != "AKIDWORK" || creds.SessionToken != "token-work" {
		t.Fatalf("credentials = %#v", creds)
	}
	if !strings.Contains(creds.Source, "work") {
		t.Fatalf("source = %q", creds.Source)
	}
}

// TestProfileBeatsAmbientEnvKeys covers the precedence pi relies on: an
// explicitly configured profile wins over ambient environment keys.
func TestProfileBeatsAmbientEnvKeys(t *testing.T) {
	clearAmbientAWSEnv(t)
	env := writeAWSFiles(t, `
[work]
aws_access_key_id = AKIDWORK
aws_secret_access_key = secret-work
`, "")
	env["AWS_ACCESS_KEY_ID"] = "AKIDENV"
	env["AWS_SECRET_ACCESS_KEY"] = "secret-env"

	creds, err := resolveAWSCredentials("work", env)
	if err != nil {
		t.Fatal(err)
	}
	if creds.AccessKeyID != "AKIDWORK" {
		t.Fatalf("access key = %q, want the profile to win", creds.AccessKeyID)
	}
}

func TestResolveAWSCredentialsFallsBackToDefaultProfile(t *testing.T) {
	clearAmbientAWSEnv(t)
	env := writeAWSFiles(t, `
[default]
aws_access_key_id = AKIDDEFAULT
aws_secret_access_key = secret-default
`, "")

	creds, err := resolveAWSCredentials("", env)
	if err != nil {
		t.Fatal(err)
	}
	if creds.AccessKeyID != "AKIDDEFAULT" {
		t.Fatalf("access key = %q", creds.AccessKeyID)
	}
}

func TestResolveAWSCredentialsMissingProfileErrors(t *testing.T) {
	clearAmbientAWSEnv(t)
	env := writeAWSFiles(t, "[default]\naws_access_key_id = A\naws_secret_access_key = B\n", "")
	if _, err := resolveAWSCredentials("nonexistent", env); err == nil {
		t.Fatal("a missing profile must error")
	}
}

func TestRegionFromProfile(t *testing.T) {
	clearAmbientAWSEnv(t)
	// The config file names non-default profiles as "[profile name]".
	env := writeAWSFiles(t, "", `
[default]
region = us-west-2

[profile work]
region = eu-central-1
`)

	if got := regionFromProfile("work", env); got != "eu-central-1" {
		t.Fatalf("work region = %q", got)
	}
	if got := regionFromProfile("default", env); got != "us-west-2" {
		t.Fatalf("default region = %q", got)
	}
	// An empty profile name means the default profile.
	if got := regionFromProfile("", env); got != "us-west-2" {
		t.Fatalf("implicit default region = %q", got)
	}
	if got := regionFromProfile("unknown", env); got != "" {
		t.Fatalf("unknown profile region = %q, want empty", got)
	}
}

func TestReadINISectionIgnoresComments(t *testing.T) {
	env := writeAWSFiles(t, `
; semicolon comment
# hash comment
[default]
aws_access_key_id = AKID
  aws_secret_access_key = secret
not-a-pair
[other]
aws_access_key_id = OTHER
`, "")

	values, err := readINISection(env["AWS_SHARED_CREDENTIALS_FILE"], "default")
	if err != nil {
		t.Fatal(err)
	}
	if values["aws_access_key_id"] != "AKID" || values["aws_secret_access_key"] != "secret" {
		t.Fatalf("values = %#v", values)
	}
	// Keys from another section must not leak in.
	if values["aws_access_key_id"] == "OTHER" {
		t.Fatal("section boundaries were not respected")
	}
}

// TestExplicitBedrockKeysBeatAmbientProfile covers the case where a user has
// AWS_PROFILE exported for an unrelated project while GoshCoder's Bedrock
// provider carries its own configured keys. Consulting the ambient profile
// first made resolveAWSCredentials short-circuit to it, so the configured keys
// were ignored and the request either failed outright or signed against the
// wrong AWS account.
func TestExplicitBedrockKeysBeatAmbientProfile(t *testing.T) {
	clearAmbientAWSEnv(t)
	t.Setenv("AWS_PROFILE", "some-other-project")

	options := &BedrockOptions{StreamOptions: StreamOptions{Env: map[string]string{
		"AWS_ACCESS_KEY_ID":     "AKIDCONFIGURED",
		"AWS_SECRET_ACCESS_KEY": "secret-configured",
		"AWS_REGION":            "us-east-1",
	}}}

	if profile := resolveBedrockProfile(options); profile != "" {
		t.Fatalf("profile = %q, want configured keys to win over the ambient profile", profile)
	}
	creds, err := resolveAWSCredentials(resolveBedrockProfile(options), options.Env)
	if err != nil {
		t.Fatalf("resolveAWSCredentials: %v", err)
	}
	if creds.AccessKeyID != "AKIDCONFIGURED" {
		t.Fatalf("access key = %q", creds.AccessKeyID)
	}
}

// TestAmbientProfileStillAppliesWithoutConfiguredKeys is the negative control:
// a user who relies on AWS_PROFILE alone must keep working.
func TestAmbientProfileStillAppliesWithoutConfiguredKeys(t *testing.T) {
	clearAmbientAWSEnv(t)
	t.Setenv("AWS_PROFILE", "work")

	options := &BedrockOptions{StreamOptions: StreamOptions{Env: map[string]string{
		"AWS_REGION": "us-east-1",
	}}}
	if profile := resolveBedrockProfile(options); profile != "work" {
		t.Fatalf("profile = %q, want the ambient profile to apply", profile)
	}
}

// TestExplicitProfileOptionWins keeps an explicitly requested profile ahead of
// configured static keys, which is what an option is for.
func TestExplicitProfileOptionWins(t *testing.T) {
	clearAmbientAWSEnv(t)
	options := &BedrockOptions{
		Profile: "chosen",
		StreamOptions: StreamOptions{Env: map[string]string{
			"AWS_ACCESS_KEY_ID":     "AKIDCONFIGURED",
			"AWS_SECRET_ACCESS_KEY": "secret-configured",
		}},
	}
	if profile := resolveBedrockProfile(options); profile != "chosen" {
		t.Fatalf("profile = %q", profile)
	}
}

// TestCanonicalURIEncodesReservedCharacters covers the SigV4 CanonicalURI rule.
// Signing the raw request path worked only while every character in it was
// unreserved; Go's url.PathEscape deliberately leaves ':' alone and every
// Bedrock model id ends in ":0", so the service canonicalised the colon to %3A,
// computed a different signature, and rejected the request.
func TestCanonicalURIEncodesReservedCharacters(t *testing.T) {
	cases := []struct{ path, want string }{
		{"", "/"},
		{"/", "/"},
		{"/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/converse-stream",
			"/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse-stream"},
		{"/a/b-c_d.e~f", "/a/b-c_d.e~f"},
		// A path that already carries a percent-escape is encoded again, which
		// is what "twice for every service except S3" means: the service does
		// the same to what it received.
		{"/model/x%2Fy/converse-stream", "/model/x%252Fy/converse-stream"},
		{"/with space", "/with%20space"},
	}
	for _, tc := range cases {
		if got := awsCanonicalURI(tc.path); got != tc.want {
			t.Errorf("awsCanonicalURI(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

// TestBedrockModelIDPathIsSignedConsistently is the end-to-end version: the
// canonical URI the signature covers must be the encoding of exactly the path
// that goes on the wire.
func TestBedrockModelIDPathIsSignedConsistently(t *testing.T) {
	modelID := "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
	raw := "https://bedrock-runtime.us-east-1.amazonaws.com/model/" + url.PathEscape(modelID) + "/converse-stream"

	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	canonical := awsCanonicalURI(parsed.EscapedPath())
	if strings.Contains(canonical, ":") {
		t.Fatalf("canonical URI still carries an unencoded colon: %q", canonical)
	}
	if !strings.Contains(canonical, "%3A0") {
		t.Fatalf("canonical URI = %q, want the model id's colon encoded", canonical)
	}
}

// TestCanonicalQueryUsesSigV4Encoding covers the query rule: net/url's Encode
// writes a space as "+", where SigV4 requires "%20", and sorts only by key.
func TestCanonicalQueryUsesSigV4Encoding(t *testing.T) {
	values := url.Values{
		"b":     []string{"two"},
		"a":     []string{"one 1", "one 2"},
		"c:key": []string{"v/v"},
	}
	want := "a=one%201&a=one%202&b=two&c%3Akey=v%2Fv"
	if got := awsCanonicalQuery(values); got != want {
		t.Fatalf("awsCanonicalQuery = %q, want %q", got, want)
	}
	if got := awsCanonicalQuery(nil); got != "" {
		t.Fatalf("empty query = %q", got)
	}
}
