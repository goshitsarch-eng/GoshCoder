package llm

// AWS Signature Version 4 request signing and credential resolution.
//
// DEVIATION: pi delegates this to @aws-sdk/client-bedrock-runtime. GoshCoder has
// no third-party dependencies, so the pieces the Bedrock protocol needs are
// implemented here against stdlib crypto:
//
//   - static credentials from the environment
//   - shared credential/config files (~/.aws/credentials, ~/.aws/config),
//     including a profile's region and credential_source-free role-less entries
//   - a Bedrock bearer token, which bypasses signing entirely
//
// Not covered: IMDS/ECS container credentials, SSO, web identity federation,
// and assume-role chaining. Those environments should export static credentials
// or AWS_BEARER_TOKEN_BEDROCK.

import (
	"bufio"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// awsCredentials are resolved static credentials.
type awsCredentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
	// Source describes where the credentials came from, for diagnostics.
	Source string
}

// resolveAWSCredentials resolves static credentials, preferring an explicitly
// configured profile over ambient environment keys (matching the SDK's default
// chain when `credentials` is not set on the client config).
func resolveAWSCredentials(profile string, env map[string]string) (*awsCredentials, error) {
	if profile != "" {
		if creds, err := credentialsFromProfile(profile, env); err == nil {
			return creds, nil
		} else {
			return nil, err
		}
	}

	accessKeyID := getProviderEnvValue("AWS_ACCESS_KEY_ID", env)
	secretAccessKey := getProviderEnvValue("AWS_SECRET_ACCESS_KEY", env)
	if accessKeyID != "" && secretAccessKey != "" {
		return &awsCredentials{
			AccessKeyID:     accessKeyID,
			SecretAccessKey: secretAccessKey,
			SessionToken:    getProviderEnvValue("AWS_SESSION_TOKEN", env),
			Source:          "AWS_ACCESS_KEY_ID",
		}, nil
	}

	// Fall back to the default profile in the shared credentials file.
	if creds, err := credentialsFromProfile("default", env); err == nil {
		return creds, nil
	}
	return nil, fmt.Errorf("no AWS credentials found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, configure a profile, or set AWS_BEARER_TOKEN_BEDROCK")
}

// sharedCredentialsPath returns the credentials file location.
func sharedCredentialsPath(env map[string]string) string {
	if path := getProviderEnvValue("AWS_SHARED_CREDENTIALS_FILE", env); path != "" {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".aws", "credentials")
}

// sharedConfigPath returns the config file location, which holds a profile's
// region (and credentials for profiles written by some tools).
func sharedConfigPath(env map[string]string) string {
	if path := getProviderEnvValue("AWS_CONFIG_FILE", env); path != "" {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".aws", "config")
}

// credentialsFromProfile reads static credentials for a named profile.
func credentialsFromProfile(profile string, env map[string]string) (*awsCredentials, error) {
	// The config file names non-default profiles as "[profile name]".
	sections := []struct {
		path string
		name string
	}{
		{sharedCredentialsPath(env), profile},
		{sharedConfigPath(env), profile},
		{sharedConfigPath(env), "profile " + profile},
	}

	for _, section := range sections {
		if section.path == "" {
			continue
		}
		values, err := readINISection(section.path, section.name)
		if err != nil {
			continue
		}
		accessKeyID := values["aws_access_key_id"]
		secretAccessKey := values["aws_secret_access_key"]
		if accessKeyID == "" || secretAccessKey == "" {
			continue
		}
		return &awsCredentials{
			AccessKeyID:     accessKeyID,
			SecretAccessKey: secretAccessKey,
			SessionToken:    values["aws_session_token"],
			Source:          fmt.Sprintf("profile %s", profile),
		}, nil
	}
	return nil, fmt.Errorf("no credentials for AWS profile %q", profile)
}

// regionFromProfile reads a profile's configured region.
func regionFromProfile(profile string, env map[string]string) string {
	if profile == "" {
		profile = "default"
	}
	for _, section := range []struct {
		path string
		name string
	}{
		{sharedConfigPath(env), "profile " + profile},
		{sharedConfigPath(env), profile},
		{sharedCredentialsPath(env), profile},
	} {
		if section.path == "" {
			continue
		}
		if values, err := readINISection(section.path, section.name); err == nil {
			if region := values["region"]; region != "" {
				return region
			}
		}
	}
	return ""
}

// readINISection parses one section of an AWS ini-style file. Keys are
// lower-cased; nested sub-sections are ignored.
func readINISection(path, section string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	values := map[string]string{}
	inSection := false
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			name := strings.TrimSpace(line[1 : len(line)-1])
			inSection = name == section
			continue
		}
		if !inSection {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		values[strings.ToLower(strings.TrimSpace(key))] = strings.TrimSpace(value)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if !inSection && len(values) == 0 {
		return nil, fmt.Errorf("section %q not found in %s", section, path)
	}
	return values, nil
}

// ---------------------------------------------------------------------------
// Signing
// ---------------------------------------------------------------------------

const (
	sigV4Algorithm    = "AWS4-HMAC-SHA256"
	sigV4RequestSufix = "aws4_request"
)

// signAWSRequest signs req in place with SigV4. payload is the exact request
// body; an empty body still hashes to the SHA-256 of the empty string.
//
// The signature covers whichever headers are present when this is called, so
// caller-supplied headers must be attached first.
func signAWSRequest(req *http.Request, payload []byte, creds *awsCredentials, region, service string, now time.Time) error {
	if creds == nil {
		return fmt.Errorf("no AWS credentials to sign with")
	}
	now = now.UTC()
	amzDate := now.Format("20060102T150405Z")
	dateStamp := now.Format("20060102")

	payloadHash := sha256Hex(payload)
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	if creds.SessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", creds.SessionToken)
	}
	// Host participates in the canonical request and is not in Header by default.
	host := req.Host
	if host == "" {
		host = req.URL.Host
	}

	canonicalHeaders, signedHeaders := canonicalizeHeaders(req.Header, host)
	canonicalURI := req.URL.EscapedPath()
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	canonicalQuery := req.URL.Query().Encode()

	canonicalRequest := strings.Join([]string{
		req.Method,
		canonicalURI,
		canonicalQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	credentialScope := strings.Join([]string{dateStamp, region, service, sigV4RequestSufix}, "/")
	stringToSign := strings.Join([]string{
		sigV4Algorithm,
		amzDate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	}, "\n")

	signingKey := deriveSigningKey(creds.SecretAccessKey, dateStamp, region, service)
	signature := hex.EncodeToString(hmacSHA256(signingKey, []byte(stringToSign)))

	req.Header.Set("Authorization", fmt.Sprintf("%s Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		sigV4Algorithm, creds.AccessKeyID, credentialScope, signedHeaders, signature))
	return nil
}

// canonicalizeHeaders builds the canonical header block and signed header list.
// Header names are lower-cased and sorted; values are trimmed and internal
// whitespace runs collapsed.
func canonicalizeHeaders(header http.Header, host string) (canonical string, signed string) {
	values := map[string]string{"host": host}
	for name, list := range header {
		lower := strings.ToLower(name)
		// Skip headers AWS excludes from signing.
		if lower == "authorization" || lower == "content-length" || lower == "user-agent" {
			continue
		}
		parts := make([]string, 0, len(list))
		for _, value := range list {
			parts = append(parts, collapseWhitespace(value))
		}
		values[lower] = strings.Join(parts, ",")
	}

	names := make([]string, 0, len(values))
	for name := range values {
		names = append(names, name)
	}
	sort.Strings(names)

	var b strings.Builder
	for _, name := range names {
		b.WriteString(name)
		b.WriteByte(':')
		b.WriteString(values[name])
		b.WriteByte('\n')
	}
	return b.String(), strings.Join(names, ";")
}

// collapseWhitespace trims a header value and collapses internal runs of
// spaces, as the canonical request requires.
func collapseWhitespace(value string) string {
	return strings.Join(strings.Fields(value), " ")
}

func deriveSigningKey(secret, dateStamp, region, service string) []byte {
	kDate := hmacSHA256([]byte("AWS4"+secret), []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	return hmacSHA256(kService, []byte(sigV4RequestSufix))
}

func hmacSHA256(key, data []byte) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write(data)
	return mac.Sum(nil)
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
