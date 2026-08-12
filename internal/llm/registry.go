package llm

import "sync"

// StreamFuncs is the uniform contract of a wire-protocol implementation
// (pi's ProviderStreams). Each api/* module in pi maps to one StreamFuncs
// registration here.
type StreamFuncs struct {
	// Stream runs a request with full (protocol-specific) options. The caller
	// passes the protocol's options struct as `any`; the implementation
	// type-asserts it (e.g. *OpenAICompletionsOptions).
	Stream func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream
	// StreamSimple runs a request with simplified reasoning options.
	StreamSimple func(model *Model, ctx *Context, opts *SimpleStreamOptions) *AssistantMessageEventStream
}

var (
	streamersMu sync.RWMutex
	streamers   = map[string]StreamFuncs{}
)

// RegisterStreamer registers a wire protocol under its api name
// (e.g. "openai-completions", "anthropic-messages"). Panics on duplicate
// registration, which indicates a programming error.
func RegisterStreamer(api string, f StreamFuncs) {
	streamersMu.Lock()
	defer streamersMu.Unlock()
	if _, exists := streamers[api]; exists {
		panic("llm: duplicate streamer registration for api " + api)
	}
	streamers[api] = f
}

// GetStreamer returns the StreamFuncs registered for api, and whether it was
// found.
func GetStreamer(api string) (StreamFuncs, bool) {
	streamersMu.RLock()
	defer streamersMu.RUnlock()
	f, ok := streamers[api]
	return f, ok
}

func init() {
	RegisterStreamer("openai-completions", StreamFuncs{
		Stream: func(model *Model, ctx *Context, opts any) *AssistantMessageEventStream {
			o, _ := opts.(*OpenAICompletionsOptions)
			return Stream(model, ctx, o)
		},
		StreamSimple: StreamSimple,
	})
}
