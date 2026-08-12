package llm

// Port of reference/pi/packages/ai/src/api/simple-options.ts.

const contextSafetyTokens = 4096
const minMaxTokens = 1

// MinAnswerTokens are the tokens always left for the answer when a thinking
// budget shares the response ceiling.
const MinAnswerTokens = 1024

// ClampMaxTokensToContext clamps maxTokens to what fits in the context window.
func ClampMaxTokensToContext(model *Model, context *Context, maxTokens int) int {
	if model.ContextWindow <= 0 {
		return max(minMaxTokens, maxTokens)
	}
	available := model.ContextWindow - EstimateContextTokens(context).Tokens - contextSafetyTokens
	return min(maxTokens, max(minMaxTokens, available))
}

// buildBaseOptions builds the shared StreamOptions for StreamSimple.
func buildBaseOptions(model *Model, context *Context, options *SimpleStreamOptions) StreamOptions {
	var opts StreamOptions
	if options != nil {
		opts = options.StreamOptions
	}
	var samplingParams map[string]any
	if len(model.SamplingParams) > 0 || len(opts.SamplingParams) > 0 {
		samplingParams = map[string]any{}
		for k, v := range model.SamplingParams {
			samplingParams[k] = v
		}
		for k, v := range opts.SamplingParams {
			samplingParams[k] = v
		}
	}
	maxTokens := opts.MaxTokens
	if maxTokens == 0 {
		maxTokens = model.MaxTokens
	}
	opts.SamplingParams = samplingParams
	opts.MaxTokens = ClampMaxTokensToContext(model, context, maxTokens)
	return opts
}

// clampReasoning maps xhigh/max down to high for token-budget purposes
// (clampReasoning in simple-options.ts).
func clampReasoning(effort ThinkingLevel) ThinkingLevel {
	if effort == ThinkingXHigh || effort == ThinkingMax {
		return ThinkingHigh
	}
	return effort
}
