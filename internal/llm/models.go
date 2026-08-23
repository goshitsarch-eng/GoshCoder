package llm

// This file ports the model helpers pi's providers use from
// reference/pi/packages/ai/src/models.ts: cost calculation and thinking-level
// clamping.

// CalculateCost computes the cost fields of usage in place from the model's
// cost rates, honoring pricing tiers and the 2x 1h-cache-write rate. Port of
// calculateCost in models.ts.
func CalculateCost(model *Model, usage *Usage) UsageCost {
	inputTokens := usage.Input + usage.CacheRead + usage.CacheWrite
	rates := model.Cost.ModelCostRates
	matchedThreshold := -1
	for _, tier := range model.Cost.Tiers {
		if inputTokens > tier.InputTokensAbove && tier.InputTokensAbove > matchedThreshold {
			rates = tier.ModelCostRates
			matchedThreshold = tier.InputTokensAbove
		}
	}

	// Anthropic charges 2x base input for 1h cache writes.
	var longWrite int
	if usage.CacheWrite1h != nil {
		longWrite = *usage.CacheWrite1h
	}
	shortWrite := usage.CacheWrite - longWrite
	usage.Cost.Input = rates.Input / 1_000_000 * float64(usage.Input)
	usage.Cost.Output = rates.Output / 1_000_000 * float64(usage.Output)
	usage.Cost.CacheRead = rates.CacheRead / 1_000_000 * float64(usage.CacheRead)
	usage.Cost.CacheWrite = (rates.CacheWrite*float64(shortWrite) + rates.Input*2*float64(longWrite)) / 1_000_000
	usage.Cost.Total = usage.Cost.Input + usage.Cost.Output + usage.Cost.CacheRead + usage.Cost.CacheWrite
	return usage.Cost
}

var extendedThinkingLevels = []ModelThinkingLevel{
	ThinkingOff, ThinkingMinimal, ThinkingLow, ThinkingMedium, ThinkingHigh, ThinkingXHigh, ThinkingMax,
}

// ThinkingLevels returns every thinking level the CLI accepts, weakest first.
func ThinkingLevels() []string {
	levels := make([]string, 0, len(extendedThinkingLevels))
	for _, level := range extendedThinkingLevels {
		levels = append(levels, string(level))
	}
	return levels
}

// GetSupportedThinkingLevels returns the thinking levels a model supports,
// honoring ThinkingLevelMap nil (unsupported) entries. Port of
// getSupportedThinkingLevels in models.ts.
func GetSupportedThinkingLevels(model *Model) []ModelThinkingLevel {
	if !model.Reasoning {
		return []ModelThinkingLevel{ThinkingOff}
	}
	levels := make([]ModelThinkingLevel, 0, len(extendedThinkingLevels))
	for _, level := range extendedThinkingLevels {
		mapped, ok := model.ThinkingLevelMap[level]
		if ok && mapped == nil {
			continue // null: level unsupported
		}
		if level == ThinkingXHigh || level == ThinkingMax {
			if !ok {
				continue // xhigh/max require an explicit mapping
			}
		}
		levels = append(levels, level)
	}
	return levels
}

// ClampThinkingLevel returns level if supported, else the nearest supported
// level (up first, then down). Port of clampThinkingLevel in models.ts.
func ClampThinkingLevel(model *Model, level ModelThinkingLevel) ModelThinkingLevel {
	available := GetSupportedThinkingLevels(model)
	for _, l := range available {
		if l == level {
			return level
		}
	}
	requestedIndex := -1
	for i, l := range extendedThinkingLevels {
		if l == level {
			requestedIndex = i
			break
		}
	}
	fallback := ThinkingOff
	if len(available) > 0 {
		fallback = available[0]
	}
	if requestedIndex == -1 {
		return fallback
	}
	for i := requestedIndex; i < len(extendedThinkingLevels); i++ {
		for _, l := range available {
			if l == extendedThinkingLevels[i] {
				return l
			}
		}
	}
	for i := requestedIndex - 1; i >= 0; i-- {
		for _, l := range available {
			if l == extendedThinkingLevels[i] {
				return l
			}
		}
	}
	return fallback
}
