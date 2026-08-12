package llm

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"testing"
	"time"
)

func TestIsAbortRecognizesWrappedContextErrors(t *testing.T) {
	if !IsAbort(fmt.Errorf("wrapped: %w", context.Canceled)) {
		t.Fatal("wrapped cancellation was not recognized")
	}
	if !IsAbort(fmt.Errorf("wrapped: %w", context.DeadlineExceeded)) {
		t.Fatal("deadline was not recognized")
	}
}

func TestRetryDelayIgnoresMalformedHeader(t *testing.T) {
	delay, err := retryDelay(&ProviderError{Headers: http.Header{"Retry-After": {"not-a-delay"}}}, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if delay < 300*time.Millisecond || delay > 500*time.Millisecond {
		t.Fatalf("fallback delay = %s", delay)
	}
}

func TestRetryDelayClampsPastDate(t *testing.T) {
	header := http.Header{"Retry-After": {time.Now().Add(-time.Hour).UTC().Format(http.TimeFormat)}}
	delay, err := retryDelay(&ProviderError{Headers: header}, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if delay != 0 {
		t.Fatalf("delay = %s, want zero", delay)
	}
}

func TestRetryDelayRejectsNonFiniteValue(t *testing.T) {
	if _, err := validateServerRetryDelayMs(math.Inf(1), nil, "bad"); err == nil {
		t.Fatal("infinite delay was accepted")
	}
}
