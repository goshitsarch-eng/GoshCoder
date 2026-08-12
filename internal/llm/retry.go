package llm

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"regexp"
	"strconv"
	"time"
)

// AbortError mirrors the TS "Request aborted" AbortError created when an
// AbortSignal fires during a request or retry sleep. In Go it is raised when
// the request context is canceled.
type AbortError struct{}

func (AbortError) Error() string { return "Request aborted" }

// IsAbort reports whether err is an AbortError or a context cancellation.
func IsAbort(err error) bool {
	if err == nil {
		return false
	}
	if _, ok := err.(AbortError); ok {
		return true
	}
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// ProviderError is an HTTP-level provider failure, carrying what pi extracts
// from SDK error objects (status + headers + body) for retry decisions and
// error formatting. See utils/provider-retry.ts and utils/error-body.ts.
type ProviderError struct {
	Status  int // 0 when no HTTP response was received (network failure)
	Headers http.Header
	// Body is the raw response body, trimmed and truncated to
	// MaxProviderErrorBodyChars.
	Body    string
	Message string
	Cause   error
}

func (e *ProviderError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	if e.Cause != nil {
		return e.Cause.Error()
	}
	return fmt.Sprintf("provider error (status %d)", e.Status)
}

func (e *ProviderError) Unwrap() error { return e.Cause }

const DefaultMaxRetryDelayMs int64 = 60_000

// isRetryableProviderError mirrors the pinned OpenAI/Anthropic SDK retry
// policy in utils/provider-retry.ts.
func isRetryableProviderError(err *ProviderError) bool {
	if err.Headers != nil {
		switch err.Headers.Get("x-should-retry") {
		case "true":
			return true
		case "false":
			return false
		}
	}
	if err.Status == 0 {
		return true
	}
	return err.Status == 408 || err.Status == 409 || err.Status == 429 || err.Status >= 500
}

// validateServerRetryDelayMs enforces the maxRetryDelayMs cap on
// server-requested delays, failing immediately when the cap is exceeded
// (utils/provider-retry.ts).
func validateServerRetryDelayMs(delayMs float64, maxRetryDelayMs *int64, providerErrorMessage string) (time.Duration, error) {
	if math.IsNaN(delayMs) || math.IsInf(delayMs, 0) {
		return 0, fmt.Errorf("invalid server retry delay")
	}
	if delayMs < 0 {
		delayMs = 0
	}
	maxDelayMs := DefaultMaxRetryDelayMs
	if maxRetryDelayMs != nil {
		maxDelayMs = *maxRetryDelayMs
	}
	if maxDelayMs > 0 && delayMs > float64(maxDelayMs) {
		return 0, fmt.Errorf("server requested %ds retry delay (max: %ds). %s",
			int64(math.Ceil(delayMs/1000)), int64(math.Ceil(float64(maxDelayMs)/1000)), providerErrorMessage)
	}
	return time.Duration(delayMs * float64(time.Millisecond)), nil
}

// retryDelay computes the backoff for a failed attempt: Retry-After-Ms and
// Retry-After headers take precedence; otherwise exponential backoff
// min(0.5 * 2^retryIndex, 8) seconds with up to 25% jitter removed.
func retryDelay(err *ProviderError, retryIndex int, maxRetryDelayMs *int64) (time.Duration, error) {
	if err.Headers != nil {
		if v := err.Headers.Get("retry-after-ms"); v != "" {
			if ms, parseErr := strconv.ParseFloat(v, 64); parseErr == nil && !math.IsNaN(ms) && !math.IsInf(ms, 0) {
				return validateServerRetryDelayMs(ms, maxRetryDelayMs, err.Error())
			}
		}
		if v := err.Headers.Get("retry-after"); v != "" {
			if seconds, parseErr := strconv.ParseFloat(v, 64); parseErr == nil && !math.IsNaN(seconds) && !math.IsInf(seconds, 0) {
				return validateServerRetryDelayMs(seconds*1000, maxRetryDelayMs, err.Error())
			}
			if retryAt, parseErr := http.ParseTime(v); parseErr == nil {
				return validateServerRetryDelayMs(float64(time.Until(retryAt).Milliseconds()), maxRetryDelayMs, err.Error())
			}
		}
	}
	exponentialMs := math.Min(0.5*math.Pow(2, float64(retryIndex)), 8) * 1000
	jittered := exponentialMs * (1 - rand.Float64()*0.25)
	return time.Duration(jittered * float64(time.Millisecond)), nil
}

// sleepCtx sleeps for d, returning AbortError when ctx is canceled first.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if ctx.Err() != nil {
		return AbortError{}
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return AbortError{}
	}
}

// RetryProviderRequest reproduces the retry behavior used by the OpenAI and
// Anthropic SDKs with an interruptible backoff sleep (utils/provider-retry.ts).
// maxRetries 0 means a single attempt. Server-requested delays above
// maxRetryDelayMs fail immediately (60s default; a nil *int64 uses the
// default, a pointed-to 0 disables the cap).
func RetryProviderRequest[T any](
	ctx context.Context,
	request func(ctx context.Context) (T, error),
	maxRetries int,
	maxRetryDelayMs *int64,
) (T, error) {
	var zero T
	retriesRemaining := maxRetries
	for {
		result, err := request(ctx)
		if err == nil {
			return result, nil
		}
		if ctx.Err() != nil {
			return zero, AbortError{}
		}
		var perr *ProviderError
		if !asProviderError(err, &perr) || retriesRemaining <= 0 || !isRetryableProviderError(perr) {
			return zero, err
		}
		retryIndex := maxRetries - retriesRemaining
		retriesRemaining--
		delay, derr := retryDelay(perr, retryIndex, maxRetryDelayMs)
		if derr != nil {
			return zero, derr
		}
		if serr := sleepCtx(ctx, delay); serr != nil {
			return zero, serr
		}
	}
}

func asProviderError(err error, out **ProviderError) bool {
	for err != nil {
		if pe, ok := err.(*ProviderError); ok {
			*out = pe
			return true
		}
		u, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = u.Unwrap()
	}
	return false
}

// --- retry.ts: assistant-call retry policy ---

var nonRetryableProviderLimitErrorPattern = regexp.MustCompile(`(?i)` + joinPatterns([]string{
	// OpenCode Go/free-tier subscription/account limits, not transient throttles.
	"GoUsageLimitError",
	"FreeUsageLimitError",
	"Monthly usage limit reached",
	"available balance",
	// Generic quota/budget/billing exhaustion.
	"insufficient_quota",
	"out of budget",
	"quota exceeded",
	"billing",
}))

var retryableProviderErrorPattern = regexp.MustCompile(`(?i)` + joinPatterns([]string{
	"overloaded",
	"rate.?limit",
	"too many requests",
	"429", "500", "502", "503", "504", "524",
	"service.?unavailable",
	"server.?error",
	"internal.?error",
	"provider.?returned.?error",
	"exceeded request buffer limit while retrying upstream",
	"network.?error",
	"connection.?error",
	"connection.?refused",
	"connection.?lost",
	"other side closed",
	"fetch failed",
	"getaddrinfo",
	"ENOTFOUND",
	"EAI_AGAIN",
	"upstream.?connect",
	"reset before headers",
	"socket hang up",
	"socket connection was closed",
	"timed? out",
	"timeout",
	"terminated",
	"websocket.?closed",
	"websocket.?error",
	"ended without",
	"stream ended before message_stop",
	"stream ended before a terminal response event",
	"http2 request did not get a response",
	"retry delay",
	"you can retry your request",
	"try your request again",
	"please retry your request",
	"ResourceExhausted",
}))

func joinPatterns(patterns []string) string {
	out := ""
	for i, p := range patterns {
		if i > 0 {
			out += "|"
		}
		out += "(?:" + p + ")"
	}
	return out
}

// IsRetryableAssistantError classifies whether a failed assistant message
// looks like a transient provider or transport error (utils/retry.ts).
func IsRetryableAssistantError(message *AssistantMessage) bool {
	if message.StopReason != StopError || message.ErrorMessage == "" {
		return false
	}
	if nonRetryableProviderLimitErrorPattern.MatchString(message.ErrorMessage) {
		return false
	}
	return retryableProviderErrorPattern.MatchString(message.ErrorMessage)
}

// RetryPolicy: bounded attempts with exponential backoff
// (baseDelayMs * 2^(attempt-1)). Mirrors utils/retry.ts.
type RetryPolicy struct {
	Enabled     bool
	MaxRetries  int
	BaseDelayMs int64
}

// RetryCallbacks are emitted by RetryAssistantCall around each retry.
type RetryCallbacks struct {
	OnRetryScheduled    func(attempt, maxAttempts int, delayMs int64, errorMessage string)
	OnRetryAttemptStart func()
	OnRetryFinished     func(success bool, attempt int, finalError string)
}

// RetryAssistantCall runs a single assistant-producing call with bounded
// retry on transient errors. Port of retryAssistantCall in utils/retry.ts.
// Aborts are terminal and never retried; non-retryable errors fail fast.
func RetryAssistantCall(
	ctx context.Context,
	produce func() *AssistantMessage,
	policy *RetryPolicy,
	callbacks *RetryCallbacks,
) *AssistantMessage {
	maxAttempts := 0
	if policy != nil && policy.Enabled {
		maxAttempts = policy.MaxRetries
	}

	attempt := 0
	lastRetryAttempt := 0
	var lastRetryError string
	hasRetried := false

	finish := func(success bool, finalError string) {
		if hasRetried && callbacks != nil && callbacks.OnRetryFinished != nil {
			callbacks.OnRetryFinished(success, lastRetryAttempt, finalError)
		}
	}

	for {
		response := produce()

		// Abort: terminal but not successful. Never retry an aborted message.
		if response.StopReason == StopAborted {
			finish(false, "")
			return response
		}

		// Success: non-error, non-abort responses return as-is.
		if response.StopReason != StopError {
			finish(true, "")
			return response
		}

		// Non-retryable, or budget exhausted: return the final error message.
		if attempt >= maxAttempts || !IsRetryableAssistantError(response) {
			finish(false, response.ErrorMessage)
			return response
		}

		attempt++
		hasRetried = true
		lastRetryAttempt = attempt
		lastRetryError = response.ErrorMessage
		if lastRetryError == "" {
			lastRetryError = "Unknown error"
		}
		delayMs := policy.BaseDelayMs * int64(math.Pow(2, float64(attempt-1)))
		if callbacks != nil && callbacks.OnRetryScheduled != nil {
			callbacks.OnRetryScheduled(attempt, maxAttempts, delayMs, lastRetryError)
		}

		// Normalize aborts during retry backoff to the same AssistantMessage
		// shape as provider stream aborts.
		if err := sleepCtx(ctx, time.Duration(delayMs)*time.Millisecond); err != nil {
			finish(false, lastRetryError)
			aborted := *response
			aborted.StopReason = StopAborted
			aborted.ErrorMessage = ""
			return &aborted
		}
		if callbacks != nil && callbacks.OnRetryAttemptStart != nil {
			callbacks.OnRetryAttemptStart()
		}
	}
}
