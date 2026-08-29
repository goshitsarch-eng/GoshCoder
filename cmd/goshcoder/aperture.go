package main

// The /aperture command surface and session integration for the native
// @aliou/pi-ts-aperture adaptation (see internal/aperture).
//
// pi registers /aperture:onboarding and /aperture:settings as TUI panels;
// GoshCoder exposes the same capabilities as line-oriented subcommands
// (onboarding, settings, sync, status, providers, connectors, pin, unpin)
// reachable from chat, the fullscreen palette, and the CLI. The colon-form
// command names survive as aliases.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"goshcoder/internal/aperture"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
)

// apertureReferer identifies GoshCoder traffic in the Aperture dashboard.
// The original sends https://pi.dev; the native adaptation names itself so
// the dashboard's provenance grouping stays truthful.
const apertureReferer = "https://github.com/goshitsarch-eng/goshcoder"

func apertureCommand(args []string) error {
	output, err := runApertureCommand(context.Background(), args, true)
	if output != "" {
		fmt.Fprintln(os.Stdout, output)
	}
	return err
}

func runApertureCommand(ctx context.Context, args []string, interactive bool) (string, error) {
	subcommand := "status"
	rest := []string(nil)
	if len(args) > 0 && strings.TrimSpace(args[0]) != "" {
		subcommand = strings.ToLower(args[0])
		rest = args[1:]
	}
	switch subcommand {
	case "status":
		return apertureStatus(ctx)
	case "onboarding", "setup":
		if !interactive {
			return "", errors.New("aperture onboarding requires an interactive terminal")
		}
		return apertureOnboarding(ctx)
	case "settings":
		return apertureSettings(ctx, rest)
	case "sync":
		return apertureSync(ctx)
	case "providers":
		return apertureProviders(ctx)
	case "connectors":
		return apertureConnectors(ctx)
	case "pin":
		if len(rest) != 1 {
			return "", errors.New("usage: /aperture pin <toolName>")
		}
		return aperturePin(ctx, rest[0])
	case "unpin":
		if len(rest) != 1 {
			return "", errors.New("usage: /aperture unpin <toolName>")
		}
		return apertureUnpin(rest[0])
	default:
		return "", fmt.Errorf("unknown Aperture command %q; use status, onboarding, settings, sync, providers, connectors, pin, or unpin", subcommand)
	}
}

// loadApertureConfig distinguishes unconfigured from malformed.
func loadApertureConfig() (aperture.Config, bool, error) {
	configured, err := aperture.Load(config.AperturePath())
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return aperture.Config{}, false, nil
		}
		return aperture.Config{}, false, err
	}
	return configured, true, nil
}

func apertureStatus(ctx context.Context) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	if !exists || configured.Resolve().BaseURL == "" {
		return "Aperture is unconfigured. Run /aperture onboarding.", nil
	}
	resolved := configured.Resolve()
	gatewayURL := aperture.GatewayURL(resolved.BaseURL)

	probeCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	health := "healthy"
	if err := aperture.NewClient(gatewayURL).Health(probeCtx); err != nil {
		health = "DOWN (" + err.Error() + ")"
	}

	var lines []string
	lines = append(lines, "Aperture: "+health, "Gateway: "+gatewayURL)
	lines = append(lines, "Dedicated provider: "+enabledWord(resolved.DedicatedEnabled))
	if resolved.DedicatedEnabled {
		cache, cacheErr := aperture.LoadCache(config.ApertureCachePath())
		count := 0
		if cacheErr == nil {
			count = len(cache.CatalogModels(aperture.BuildCatalogKey(gatewayURL, resolved)))
		}
		lines = append(lines, fmt.Sprintf("  Synchronized models: %d (refresh with /aperture sync)", count))
	}
	lines = append(lines, "Proxy: "+enabledWord(resolved.ProxyEnabled))
	if resolved.ProxyEnabled {
		enabled := resolved.EnabledUpstreamProviders()
		ids := make([]string, 0, len(enabled))
		for _, provider := range enabled {
			ids = append(ids, provider.ID)
		}
		summary := "none"
		if len(ids) > 0 {
			summary = strings.Join(ids, ", ")
		}
		lines = append(lines, "  Upstream providers: "+summary)
	}
	lines = append(lines, "Connectors: "+enabledWord(resolved.ConnectorsEnabled))
	if resolved.ConnectorsEnabled {
		lines = append(lines, fmt.Sprintf("  Discovery tools: %s · pinned: %d", enabledWord(resolved.DiscoveryTools), len(resolved.PinnedTools)))
	}
	return strings.Join(lines, "\n"), nil
}

func enabledWord(enabled bool) string {
	if enabled {
		return "enabled"
	}
	return "disabled"
}

// apertureOnboarding is the first-run wizard
// (extensions/aperture/onboarding): URL with inline health check,
// capability choice, per-capability provider selection, recap, save, sync.
func apertureOnboarding(ctx context.Context) (string, error) {
	existing, _, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	resolved := existing.Resolve()

	fmt.Fprintln(os.Stderr, "Aperture lets you route LLM traffic through your Tailscale tailnet.")
	fmt.Fprintln(os.Stderr, "You can use it two ways:")
	fmt.Fprintln(os.Stderr, "  - Dedicated provider: a standalone \"aperture\" provider with all models from your gateway")
	fmt.Fprintln(os.Stderr, "  - Proxy: reroute existing providers (e.g. anthropic, openai) through Aperture")
	fmt.Fprintln(os.Stderr, "You can change these settings later with /aperture settings.")
	fmt.Fprintln(os.Stderr)

	// URL step with an inline health check; a failed check loops back.
	gatewayURL := ""
	for {
		prompt := "Aperture base URL (e.g. ai.pango-lin.ts.net)"
		if resolved.BaseURL != "" {
			prompt += " [" + resolved.BaseURL + "]"
		}
		fmt.Fprintf(os.Stderr, "%s: ", prompt)
		entered, readErr := readTerminalLine(os.Stdin)
		if readErr != nil {
			return "", readErr
		}
		entered = strings.TrimSpace(entered)
		if entered == "" {
			entered = resolved.BaseURL
		}
		if entered == "" {
			fmt.Fprintln(os.Stderr, "A gateway URL is required.")
			continue
		}
		candidate := aperture.NormalizeInputURL(entered)
		fmt.Fprintln(os.Stderr, "Checking connection...")
		probeCtx, cancel := context.WithTimeout(ctx, 8*time.Second)
		healthErr := aperture.NewClient(candidate).Health(probeCtx)
		cancel()
		if healthErr != nil {
			fmt.Fprintf(os.Stderr, "Could not connect: %v\nFix the URL and press Enter to retry.\n", healthErr)
			continue
		}
		fmt.Fprintln(os.Stderr, "Connected.")
		gatewayURL = candidate
		break
	}

	// Capabilities step.
	fmt.Fprintln(os.Stderr, "\nHow do you want to use Aperture?")
	fmt.Fprintln(os.Stderr, "  1. Dedicated only — all gateway models under one aperture provider")
	fmt.Fprintln(os.Stderr, "  2. Proxy only — reroute existing providers, keeping their model definitions")
	fmt.Fprintln(os.Stderr, "  3. Both")
	fmt.Fprint(os.Stderr, "Choice [1]: ")
	choice, err := readTerminalLine(os.Stdin)
	if err != nil {
		return "", err
	}
	dedicatedEnabled, proxyEnabled := true, false
	switch strings.TrimSpace(choice) {
	case "", "1":
	case "2":
		dedicatedEnabled, proxyEnabled = false, true
	case "3":
		dedicatedEnabled, proxyEnabled = true, true
	default:
		return "", fmt.Errorf("unknown choice %q", strings.TrimSpace(choice))
	}

	client := aperture.NewClient(aperture.GatewayURL(gatewayURL))
	fetchCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	gatewayProviders, err := client.Providers(fetchCtx)
	cancel()
	if err != nil {
		return "", fmt.Errorf("fetch providers from gateway: %w", err)
	}

	var dedicatedProviders []aperture.DedicatedProviderConfig
	if dedicatedEnabled {
		dedicatedProviders = aperture.MapDedicatedProviders(gatewayProviders, resolved.DedicatedProviders)
		if len(dedicatedProviders) == 0 {
			fmt.Fprintln(os.Stderr, "\nNo providers found on the Aperture gateway.")
		} else {
			fmt.Fprintln(os.Stderr, "\nSelect Aperture providers to include:")
			selected, selectErr := selectByIndex(dedicatedNames(dedicatedProviders), dedicatedEnabledSet(dedicatedProviders))
			if selectErr != nil {
				return "", selectErr
			}
			for index := range dedicatedProviders {
				dedicatedProviders[index].Enabled = selected[dedicatedProviders[index].ID]
			}
		}
	}

	var upstreamProviders []aperture.ProxiedProviderConfig
	if proxyEnabled {
		mapped := aperture.MapProxyProviders(allCatalogModels(), gatewayProviders, resolved.UpstreamProviders)
		if len(mapped) == 0 {
			fmt.Fprintln(os.Stderr, "\nNo local providers match the Aperture gateway providers.")
			fmt.Fprintln(os.Stderr, "You can add proxy providers later with /aperture settings.")
		} else {
			fmt.Fprintln(os.Stderr, "\nSelect providers to route through Aperture:")
			names := make([]indexedChoice, 0, len(mapped))
			checked := map[string]bool{}
			for _, provider := range mapped {
				label := provider.Name
				if label == "" {
					label = provider.ID
				}
				names = append(names, indexedChoice{ID: provider.ID, Label: label})
				checked[provider.ID] = provider.Enabled
			}
			selected, selectErr := selectByIndex(names, checked)
			if selectErr != nil {
				return "", selectErr
			}
			fmt.Fprint(os.Stderr, "Warn when local models are missing from the gateway? [Y/n]: ")
			answer, readErr := readTerminalLine(os.Stdin)
			if readErr != nil {
				return "", readErr
			}
			checkGateway := !strings.EqualFold(strings.TrimSpace(answer), "n")
			for _, provider := range mapped {
				if !selected[provider.ID] {
					continue
				}
				upstreamProviders = append(upstreamProviders, aperture.ProxiedProviderConfig{
					ID:                       provider.ID,
					ShouldCheckGatewayModels: checkGateway,
				})
			}
		}
	}

	// Recap step.
	fmt.Fprintln(os.Stderr, "\nRecap:")
	fmt.Fprintln(os.Stderr, "  URL: "+gatewayURL)
	capability := "Proxy existing providers"
	if dedicatedEnabled && proxyEnabled {
		capability = "Dedicated provider and proxy"
	} else if dedicatedEnabled {
		capability = "Dedicated provider"
	}
	fmt.Fprintln(os.Stderr, "  Capabilities: "+capability)
	if proxyEnabled {
		ids := make([]string, 0, len(upstreamProviders))
		for _, provider := range upstreamProviders {
			ids = append(ids, provider.ID)
		}
		summary := "none selected"
		if len(ids) > 0 {
			summary = strings.Join(ids, ", ")
		}
		fmt.Fprintln(os.Stderr, "  Upstream providers: "+summary)
	}
	if dedicatedEnabled {
		enabledCount := 0
		for _, provider := range dedicatedProviders {
			if provider.Enabled {
				enabledCount++
			}
		}
		if len(dedicatedProviders) == 0 {
			fmt.Fprintln(os.Stderr, "  Aperture providers: all (no filter)")
		} else {
			fmt.Fprintf(os.Stderr, "  Aperture providers: %d/%d enabled\n", enabledCount, len(dedicatedProviders))
		}
	}
	fmt.Fprint(os.Stderr, "Save? [Y/n]: ")
	confirm, err := readTerminalLine(os.Stdin)
	if err != nil {
		return "", err
	}
	if strings.EqualFold(strings.TrimSpace(confirm), "n") {
		return "Aperture onboarding cancelled.", nil
	}

	updated := existing
	updated.BaseURL = gatewayURL
	done := true
	onboardingOff := false
	updated.OnboardingDone = &done
	updated.Onboarding = &struct {
		Enabled *bool `json:"enabled,omitempty"`
	}{Enabled: &onboardingOff}
	setProxyConfig(&updated, proxyEnabled, upstreamProviders)
	setDedicatedConfig(&updated, dedicatedEnabled, dedicatedProviders)
	if err := aperture.Save(config.AperturePath(), updated); err != nil {
		return "", err
	}
	syncMessage, syncErr := apertureSync(ctx)
	if syncErr != nil {
		return "Aperture onboarding completed, but the first sync failed: " + syncErr.Error(), nil
	}
	return "Aperture onboarding completed. " + syncMessage, nil
}

type indexedChoice struct {
	ID    string
	Label string
}

func dedicatedNames(providers []aperture.DedicatedProviderConfig) []indexedChoice {
	out := make([]indexedChoice, 0, len(providers))
	for _, provider := range providers {
		label := provider.Name
		if label == "" {
			label = provider.ID
		}
		out = append(out, indexedChoice{ID: provider.ID, Label: label})
	}
	return out
}

func dedicatedEnabledSet(providers []aperture.DedicatedProviderConfig) map[string]bool {
	out := map[string]bool{}
	for _, provider := range providers {
		out[provider.ID] = provider.Enabled
	}
	return out
}

// selectByIndex prints a numbered checklist and reads a selection: "all",
// "none", or space/comma-separated indices to toggle from the given state.
func selectByIndex(choices []indexedChoice, checked map[string]bool) (map[string]bool, error) {
	for index, choice := range choices {
		mark := " "
		if checked[choice.ID] {
			mark = "x"
		}
		fmt.Fprintf(os.Stderr, "  %2d. [%s] %s\n", index+1, mark, choice.Label)
	}
	fmt.Fprint(os.Stderr, "Toggle by number (e.g. \"1 3\"), or \"all\"/\"none\"; Enter keeps the selection: ")
	line, err := readTerminalLine(os.Stdin)
	if err != nil {
		return nil, err
	}
	selection := strings.TrimSpace(strings.ToLower(line))
	switch selection {
	case "":
	case "all":
		for _, choice := range choices {
			checked[choice.ID] = true
		}
	case "none":
		for _, choice := range choices {
			checked[choice.ID] = false
		}
	default:
		for _, field := range strings.FieldsFunc(selection, func(r rune) bool { return r == ' ' || r == ',' }) {
			index, parseErr := strconv.Atoi(field)
			if parseErr != nil || index < 1 || index > len(choices) {
				return nil, fmt.Errorf("invalid selection %q", field)
			}
			id := choices[index-1].ID
			checked[id] = !checked[id]
		}
	}
	return checked, nil
}

func setProxyConfig(configured *aperture.Config, enabled bool, providers []aperture.ProxiedProviderConfig) {
	if providers == nil {
		providers = []aperture.ProxiedProviderConfig{}
	}
	configured.Proxy = &struct {
		Enabled           *bool                            `json:"enabled,omitempty"`
		UpstreamProviders []aperture.ProxiedProviderConfig `json:"upstreamProviders"`
	}{Enabled: &enabled, UpstreamProviders: providers}
}

func setDedicatedConfig(configured *aperture.Config, enabled bool, providers []aperture.DedicatedProviderConfig) {
	if providers == nil {
		providers = []aperture.DedicatedProviderConfig{}
	}
	configured.Dedicated = &struct {
		Enabled      *bool                              `json:"enabled,omitempty"`
		Providers    []aperture.DedicatedProviderConfig `json:"providers"`
		CachedModels []json.RawMessage                  `json:"cachedModels,omitempty"`
	}{Enabled: &enabled, Providers: providers}
}

// allCatalogModels flattens every provider's models for metadata resolution,
// upstream URL inference, and proxy provider mapping.
func allCatalogModels() []llm.Model {
	var models []llm.Model
	for _, provider := range newCatalog().Providers() {
		models = append(models, provider.Models()...)
	}
	return models
}

func apertureSync(ctx context.Context) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	if !exists {
		return "", errors.New("aperture is unconfigured; run /aperture onboarding")
	}
	syncCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	result, err := aperture.Sync(syncCtx, configured, config.ApertureCachePath(), allCatalogModels())
	if err != nil {
		return "", err
	}
	lines := []string{fmt.Sprintf("Synchronized %d Aperture models from %d gateway provider(s). They are available under /model immediately.", len(result.Models), len(result.Gateway))}
	lines = append(lines, result.Warnings...)
	return strings.Join(lines, "\n"), nil
}

func apertureProviders(ctx context.Context) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	if !exists || configured.Resolve().BaseURL == "" {
		return "", errors.New("aperture is unconfigured; run /aperture onboarding")
	}
	fetchCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	providers, err := aperture.NewClient(aperture.GatewayURL(configured.Resolve().BaseURL)).Providers(fetchCtx)
	if err != nil {
		return "", err
	}
	if len(providers) == 0 {
		return "No providers found on the Aperture gateway.", nil
	}
	lines := make([]string, 0, len(providers)+1)
	lines = append(lines, fmt.Sprintf("%d gateway provider(s):", len(providers)))
	for _, provider := range providers {
		apis := aperture.SelectableAPIs(provider.Compatibility)
		summary := "no routable api"
		if len(apis) > 0 {
			summary = "auto: " + apis[0]
			if len(apis) > 1 {
				summary += " (also " + strings.Join(apis[1:], ", ") + ")"
			}
		}
		auth := ""
		if provider.RequiresClientAuth {
			auth = " · client auth required"
		}
		lines = append(lines, fmt.Sprintf("  %s (%s) — %d model(s) — %s%s", provider.Name, provider.ID, len(provider.Models), summary, auth))
	}
	return strings.Join(lines, "\n"), nil
}

func apertureConnectors(ctx context.Context) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	resolved := configured.Resolve()
	if !exists || resolved.BaseURL == "" {
		return "", errors.New("aperture is unconfigured; run /aperture onboarding")
	}
	gatewayURL := aperture.GatewayURL(resolved.BaseURL)
	fetchCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	session, err := aperture.NewMcpSession(fetchCtx, gatewayURL)
	if err != nil {
		return "", fmt.Errorf("connector session failed: %w", err)
	}
	tools, err := session.ListTools(fetchCtx)
	if err != nil {
		return "", fmt.Errorf("connector tools/list failed: %w", err)
	}
	connectors, err := aperture.NewClient(gatewayURL).Connectors(fetchCtx)
	if err != nil {
		connectors = nil
	}

	counts := map[string]int{}
	for _, tool := range tools {
		counts[aperture.ConnectorIDFromToolName(tool.Name)]++
	}
	pinned := map[string]bool{}
	for _, pin := range resolved.PinnedTools {
		pinned[pin.ToolName] = true
	}

	var lines []string
	lines = append(lines, fmt.Sprintf("Connectors feature: %s · discovery tools: %s · %d pinned", enabledWord(resolved.ConnectorsEnabled), enabledWord(resolved.DiscoveryTools), len(resolved.PinnedTools)))
	if len(connectors) > 0 {
		for _, connector := range connectors {
			if counts[connector.ID] == 0 {
				continue
			}
			lines = append(lines, fmt.Sprintf("  %s (%s): %d tool(s) — %s", connector.Provider, connector.ID, counts[connector.ID], connector.Status))
		}
	}
	lines = append(lines, fmt.Sprintf("Gateway tools (%d):", len(tools)))
	for _, tool := range tools {
		marker := ""
		if pinned[tool.Name] {
			marker = " [pinned]"
		}
		lines = append(lines, "  "+tool.Name+marker)
	}
	lines = append(lines, "Pin a tool first-class with /aperture pin <toolName>; enable the feature with /aperture settings connectors.enabled enabled.")
	return strings.Join(lines, "\n"), nil
}

func aperturePin(ctx context.Context, toolName string) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	resolved := configured.Resolve()
	if !exists || resolved.BaseURL == "" {
		return "", errors.New("aperture is unconfigured; run /aperture onboarding")
	}
	for _, pin := range resolved.PinnedTools {
		if pin.ToolName == toolName {
			return fmt.Sprintf("%s is already pinned.", toolName), nil
		}
	}
	// Validate against the live gateway so a typo is caught here rather than
	// silently skipped at registration.
	fetchCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	session, err := aperture.NewMcpSession(fetchCtx, aperture.GatewayURL(resolved.BaseURL))
	if err != nil {
		return "", fmt.Errorf("connector session failed: %w", err)
	}
	tools, err := session.ListTools(fetchCtx)
	if err != nil {
		return "", fmt.Errorf("connector tools/list failed: %w", err)
	}
	found := false
	for _, tool := range tools {
		if tool.Name == toolName {
			found = true
			break
		}
	}
	if !found {
		return "", fmt.Errorf("tool %q not found on the gateway; /aperture connectors lists the available tools", toolName)
	}

	pins := append(append([]aperture.PinnedConnectorTool(nil), resolved.PinnedTools...), aperture.PinnedConnectorTool{
		ConnectorID: aperture.ConnectorIDFromToolName(toolName),
		ToolName:    toolName,
	})
	if configured.Connectors == nil {
		configured.Connectors = &aperture.ConnectorsConfig{}
	}
	configured.Connectors.PinnedTools = pins
	if err := aperture.Save(config.AperturePath(), configured); err != nil {
		return "", err
	}
	message := fmt.Sprintf("Pinned %s (%d pinned). Pin changes take effect on the next session.", toolName, len(pins))
	if len(pins) > aperture.ContextCostWarningThreshold {
		message += " Each pinned tool adds its full schema to the system prompt; prefer pinning only the few tools you use every session."
	}
	return message, nil
}

func apertureUnpin(toolName string) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	if !exists {
		return "", errors.New("aperture is unconfigured; run /aperture onboarding")
	}
	resolved := configured.Resolve()
	var pins []aperture.PinnedConnectorTool
	removed := false
	for _, pin := range resolved.PinnedTools {
		if pin.ToolName == toolName {
			removed = true
			continue
		}
		pins = append(pins, pin)
	}
	if !removed {
		return "", fmt.Errorf("%s is not pinned", toolName)
	}
	if configured.Connectors == nil {
		configured.Connectors = &aperture.ConnectorsConfig{}
	}
	if pins == nil {
		pins = []aperture.PinnedConnectorTool{}
	}
	configured.Connectors.PinnedTools = pins
	if err := aperture.Save(config.AperturePath(), configured); err != nil {
		return "", err
	}
	return fmt.Sprintf("Unpinned %s (%d pinned). Pin changes take effect on the next session.", toolName, len(pins)), nil
}

// apertureSettings prints all settings, or applies one change:
// /aperture settings <key> <value>. Keys mirror the original settings panel
// (global, proxy, dedicated, connectors tabs) plus per-provider entries.
func apertureSettings(ctx context.Context, args []string) (string, error) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		return "", err
	}
	if len(args) == 0 {
		if !exists {
			return "Aperture is unconfigured. Run /aperture onboarding, or set a URL with /aperture settings baseUrl <url>.", nil
		}
		return apertureSettingsSummary(configured), nil
	}
	if len(args) != 2 {
		return "", errors.New("usage: /aperture settings [<key> <value>]; run /aperture settings to list keys")
	}
	key, value := args[0], args[1]
	if err := applyApertureSetting(&configured, key, value); err != nil {
		return "", err
	}
	if err := aperture.Save(config.AperturePath(), configured); err != nil {
		return "", err
	}
	// Routing settings change the catalog identity; refresh eagerly so the
	// model picker reflects the change without waiting for the next session.
	if message, syncErr := apertureSync(ctx); syncErr == nil {
		return fmt.Sprintf("Set %s = %s.\n%s", key, value, message), nil
	}
	return fmt.Sprintf("Set %s = %s. Run /aperture sync to refresh the catalog.", key, value), nil
}

func apertureSettingsSummary(configured aperture.Config) string {
	resolved := configured.Resolve()
	var lines []string
	lines = append(lines,
		"Aperture settings (change with /aperture settings <key> <value>):",
		"  baseUrl                     = "+orUnset(resolved.BaseURL),
		"  onboardingDone              = "+strconv.FormatBool(resolved.OnboardingDone),
		"  onboarding.enabled          = "+strconv.FormatBool(resolved.OnboardingEnabled),
		"  proxy.enabled               = "+strconv.FormatBool(resolved.ProxyEnabled),
		"  dedicated.enabled           = "+strconv.FormatBool(resolved.DedicatedEnabled),
		"  connectors.enabled          = "+strconv.FormatBool(resolved.ConnectorsEnabled),
		"  connectors.discoveryTools   = "+strconv.FormatBool(resolved.DiscoveryTools),
	)
	if len(resolved.UpstreamProviders) > 0 {
		lines = append(lines, "Proxy providers (proxy.provider.<id>.enabled|check|gatewayModelsOnly|api):")
		for _, provider := range resolved.UpstreamProviders {
			api := provider.API
			if api == "" {
				api = "auto"
			}
			lines = append(lines, fmt.Sprintf("  %-16s enabled=%v check=%v gatewayModelsOnly=%v api=%s",
				provider.ID, provider.IsEnabled(), provider.ShouldCheckGatewayModels, provider.KeepGatewayModelsOnly, api))
		}
	}
	if len(resolved.DedicatedProviders) > 0 {
		lines = append(lines, "Dedicated providers (dedicated.provider.<id>.enabled|api):")
		for _, provider := range resolved.DedicatedProviders {
			api := provider.API
			if api == "" {
				api = "auto"
			}
			lines = append(lines, fmt.Sprintf("  %-16s enabled=%v api=%s", provider.ID, provider.Enabled, api))
		}
	} else if resolved.DedicatedEnabled {
		lines = append(lines, "Dedicated providers: all (no filter)")
	}
	if len(resolved.PinnedTools) > 0 {
		names := make([]string, 0, len(resolved.PinnedTools))
		for _, pin := range resolved.PinnedTools {
			names = append(names, pin.ToolName)
		}
		sort.Strings(names)
		lines = append(lines, "Pinned connector tools: "+strings.Join(names, ", "))
	}
	return strings.Join(lines, "\n")
}

func orUnset(value string) string {
	if value == "" {
		return "(not set)"
	}
	return value
}

func parseSettingBool(value string) (bool, error) {
	switch strings.ToLower(value) {
	case "true", "enabled", "on", "yes", "completed":
		return true, nil
	case "false", "disabled", "off", "no", "pending":
		return false, nil
	default:
		return false, fmt.Errorf("expected enabled/disabled, got %q", value)
	}
}

var routableAPIs = map[string]bool{
	"openai-completions":      true,
	"anthropic-messages":      true,
	"openai-responses":        true,
	"google-generative-ai":    true,
	"google-vertex":           true,
	"bedrock-converse-stream": true,
}

func parseSettingAPI(value string) (aperture.RoutableAPI, error) {
	if value == "auto" || value == "" {
		return "", nil
	}
	if !routableAPIs[value] {
		return "", fmt.Errorf("unknown api %q (auto, openai-completions, anthropic-messages, openai-responses, google-generative-ai, google-vertex, bedrock-converse-stream)", value)
	}
	return value, nil
}

func applyApertureSetting(configured *aperture.Config, key, value string) error {
	resolved := configured.Resolve()
	switch key {
	case "baseUrl":
		configured.BaseURL = aperture.NormalizeInputURL(value)
		return nil
	case "onboardingDone":
		done, err := parseSettingBool(value)
		if err != nil {
			return err
		}
		enabled := !done
		configured.OnboardingDone = &done
		configured.Onboarding = &struct {
			Enabled *bool `json:"enabled,omitempty"`
		}{Enabled: &enabled}
		return nil
	case "onboarding.enabled":
		enabled, err := parseSettingBool(value)
		if err != nil {
			return err
		}
		configured.Onboarding = &struct {
			Enabled *bool `json:"enabled,omitempty"`
		}{Enabled: &enabled}
		return nil
	case "proxy.enabled":
		enabled, err := parseSettingBool(value)
		if err != nil {
			return err
		}
		setProxyConfig(configured, enabled, resolved.UpstreamProviders)
		return nil
	case "dedicated.enabled":
		enabled, err := parseSettingBool(value)
		if err != nil {
			return err
		}
		setDedicatedConfig(configured, enabled, resolved.DedicatedProviders)
		return nil
	case "connectors.enabled":
		enabled, err := parseSettingBool(value)
		if err != nil {
			return err
		}
		if configured.Connectors == nil {
			configured.Connectors = &aperture.ConnectorsConfig{}
		}
		configured.Connectors.Enabled = enabled
		return nil
	case "connectors.discoveryTools":
		enabled, err := parseSettingBool(value)
		if err != nil {
			return err
		}
		if configured.Connectors == nil {
			configured.Connectors = &aperture.ConnectorsConfig{}
		}
		configured.Connectors.DiscoveryTools = &enabled
		return nil
	}

	if id, field, ok := providerSettingKey(key, "proxy.provider."); ok {
		providers := resolved.UpstreamProviders
		index := -1
		for i := range providers {
			if providers[i].ID == id {
				index = i
				break
			}
		}
		if index == -1 {
			providers = append(providers, aperture.ProxiedProviderConfig{ID: id, ShouldCheckGatewayModels: true})
			index = len(providers) - 1
		}
		switch field {
		case "enabled":
			enabled, err := parseSettingBool(value)
			if err != nil {
				return err
			}
			providers[index].Enabled = &enabled
		case "check", "shouldCheckGatewayModels":
			check, err := parseSettingBool(value)
			if err != nil {
				return err
			}
			providers[index].ShouldCheckGatewayModels = check
		case "gatewayModelsOnly", "keepGatewayModelsOnly":
			keep, err := parseSettingBool(value)
			if err != nil {
				return err
			}
			providers[index].KeepGatewayModelsOnly = keep
		case "api":
			api, err := parseSettingAPI(value)
			if err != nil {
				return err
			}
			providers[index].API = api
		default:
			return fmt.Errorf("unknown proxy provider setting %q (enabled, check, gatewayModelsOnly, api)", field)
		}
		setProxyConfig(configured, resolved.ProxyEnabled, providers)
		return nil
	}

	if id, field, ok := providerSettingKey(key, "dedicated.provider."); ok {
		providers := resolved.DedicatedProviders
		index := -1
		for i := range providers {
			if providers[i].ID == id {
				index = i
				break
			}
		}
		if index == -1 {
			providers = append(providers, aperture.DedicatedProviderConfig{ID: id, Enabled: true})
			index = len(providers) - 1
		}
		switch field {
		case "enabled":
			enabled, err := parseSettingBool(value)
			if err != nil {
				return err
			}
			providers[index].Enabled = enabled
		case "api":
			api, err := parseSettingAPI(value)
			if err != nil {
				return err
			}
			providers[index].API = api
		default:
			return fmt.Errorf("unknown dedicated provider setting %q (enabled, api)", field)
		}
		setDedicatedConfig(configured, resolved.DedicatedEnabled, providers)
		return nil
	}

	return fmt.Errorf("unknown Aperture setting %q; run /aperture settings to list keys", key)
}

func providerSettingKey(key, prefix string) (id, field string, ok bool) {
	if !strings.HasPrefix(key, prefix) {
		return "", "", false
	}
	rest := strings.TrimPrefix(key, prefix)
	index := strings.LastIndex(rest, ".")
	if index <= 0 || index == len(rest)-1 {
		return "", "", false
	}
	return rest[:index], rest[index+1:], true
}
