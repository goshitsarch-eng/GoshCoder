package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// MaxProviderErrorBodyChars caps how much of a provider error body is
// surfaced (utils/error-body.ts).
const MaxProviderErrorBodyChars = 4000

// TruncateErrorText truncates long provider error text (utils/error-body.ts).
func TruncateErrorText(text string, maxChars int) string {
	if len(text) <= maxChars {
		return text
	}
	return fmt.Sprintf("%s... [truncated %d chars]", text[:maxChars], len(text)-maxChars)
}

// FormatProviderError composes the display string for a provider failure.
// This is the Go counterpart of formatProviderError(normalizeProviderError(e))
// in utils/error-body.ts: the TS original probes SDK-specific error fields;
// here the fields come straight from the HTTP response. When the message
// already carries the body (or either side is missing), the message is
// returned unchanged; otherwise "<status>: <body>".
func FormatProviderError(status int, body, message string) string {
	if strings.Contains(message, body) || status == 0 || body == "" {
		if status != 0 {
			return fmt.Sprintf("%d: %s", status, message)
		}
		return message
	}
	return fmt.Sprintf("%d: %s", status, body)
}

// formatStreamError builds the errorMessage for a failed assistant message,
// mirroring the catch block in api/openai-completions.ts: format the provider
// error, then append OpenRouter-style error.metadata.raw when it adds
// information.
func formatStreamError(err error) string {
	var perr *ProviderError
	if asProviderError(err, &perr) {
		body := strings.TrimSpace(perr.Body)
		if body != "" {
			body = TruncateErrorText(body, MaxProviderErrorBodyChars)
		}
		msg := FormatProviderError(perr.Status, body, perr.Error())
		// Some providers via OpenRouter give additional information in
		// error.metadata.raw. Only append when not already present.
		if raw := extractOpenRouterRawMetadata(perr.Body); raw != "" && !strings.Contains(msg, raw) {
			msg += "\n" + raw
		}
		return msg
	}
	return err.Error()
}

// extractOpenRouterRawMetadata pulls error.metadata.raw out of a provider
// error body (api/openai-completions.ts reads (error as any)?.error?.metadata?.raw).
func extractOpenRouterRawMetadata(body string) string {
	var parsed struct {
		Error struct {
			Metadata struct {
				Raw string `json:"raw"`
			} `json:"metadata"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(body), &parsed); err != nil {
		return ""
	}
	return parsed.Error.Metadata.Raw
}
