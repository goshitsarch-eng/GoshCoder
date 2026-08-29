package main

// Session-lifecycle integration for the native Aperture and
// computer-use-linux adaptations: the per-request routing the pi extension
// installs via hooks, the session_start refresh, connector tool
// registration, and the desktop MCP tool.

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/aperture"
	"goshcoder/internal/computeruse"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
)

// apertureRequestModel applies the per-request rewrites pi's extension does
// in its stream wrappers and before_provider_headers hook: model-id
// qualification (bare-id stripping for path-embedding APIs on the dedicated
// provider), and the Referer plus live x-session-id provenance headers.
// Returns the model unchanged when the request is not gateway-routed.
func apertureRequestModel(state *catalog.ApertureState, model *llm.Model, sessionID string) *llm.Model {
	if state == nil || !state.Configured {
		return model
	}
	routed := false
	rewritten := *model
	if model.Provider == aperture.DedicatedProviderID {
		// Dedicated catalog ids are provider-qualified; path-embedding APIs
		// need the bare id because the gateway forwards URL paths verbatim.
		rewritten.ID = aperture.StripCatalogPrefix(model.API, model.ID)
		routed = true
	} else if _, ok := state.Routes[model.Provider]; ok {
		// Proxied models keep bare ids in the picker; the request carries the
		// provider-qualified id so the gateway can disambiguate duplicates.
		rewritten.ID = aperture.QualifyModelID(model.Provider, model.API, model.ID)
		routed = true
	}
	if !routed {
		return model
	}
	headers := make(map[string]string, len(model.Headers)+2)
	for name, value := range model.Headers {
		headers[name] = value
	}
	headers["Referer"] = apertureReferer
	if sessionID != "" {
		headers["x-session-id"] = sessionID
	}
	rewritten.Headers = headers
	return &rewritten
}

// markApertureRetryable re-emits the stream, tagging transient gateway
// errors ("Aperture is restarting") with the marker the retry classifier
// already recognizes, so brief gateway restarts recover in place
// (src/retryable-errors.ts + the message_end hook).
func markApertureRetryable(stream *llm.AssistantMessageEventStream) *llm.AssistantMessageEventStream {
	out := llm.NewAssistantMessageEventStream()
	go func() {
		for {
			event, ok := stream.Next(context.Background())
			if !ok {
				out.End()
				return
			}
			if event.Type == llm.EventError && event.Error != nil && event.Error.ErrorMessage != "" {
				if tagged := aperture.MarkRetryableError(event.Error.ErrorMessage); tagged != "" {
					copied := *event.Error
					copied.ErrorMessage = tagged
					event.Error = &copied
				}
			}
			out.Push(event)
		}
	}()
	return out
}

// apertureExtras is the per-session state of the two adaptations.
type apertureExtras struct {
	mu sync.Mutex
	// connectorTools are the Aperture connector tools fetched at session
	// start; merged into every tool-list rebuild.
	connectorTools []agent.Tool
	// connectorSession is the live gateway MCP session the connector tools
	// call through.
	connectorSession *aperture.McpSession
	// desktop is the lazily-spawned computer-use-linux server; nil off Linux
	// or when tools are disabled.
	desktop *computeruse.Session
}

func (e *apertureExtras) tools() []agent.Tool {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]agent.Tool(nil), e.connectorTools...)
}

func (e *apertureExtras) session() *aperture.McpSession {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.connectorSession
}

func (e *apertureExtras) close() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.desktop != nil {
		e.desktop.Close()
		e.desktop = nil
	}
}

// desktopMCPTool registers the native computer-use-linux mcp proxy tool.
// The package targets Linux desktops only (its npm manifest declares
// os: linux); elsewhere nothing registers, like the extension never
// installing.
func (s *session) desktopMCPTool(quiet bool) []agent.Tool {
	if runtime.GOOS != "linux" {
		return nil
	}
	binary := computeruse.FindBinary(nil)
	if binary == "" {
		// The extension warns on session_start when the binary is missing.
		s.startupNotices = append(s.startupNotices, computeruse.PackageName+": computer-use-linux binary not found. "+computeruse.InstallHint)
		return nil
	}
	// Keep the pi-mcp-adapter-compatible mcp.json entry registered, exactly
	// as the extension does, so other MCP hosts sharing the agent directory
	// can spawn the server too. A malformed existing file is reported, never
	// overwritten.
	switch result, err := computeruse.EnsureServerEntry(config.MCPConfigPath(), binary); result {
	case computeruse.EnsureUpdated:
		if !quiet {
			s.startupNotices = append(s.startupNotices, computeruse.PackageName+": MCP server configured at "+config.MCPConfigPath())
		}
	case computeruse.EnsureFailed:
		s.startupNotices = append(s.startupNotices, fmt.Sprintf("%s: failed to configure MCP server at %s: %v", computeruse.PackageName, config.MCPConfigPath(), err))
	}
	s.extras.desktop = computeruse.NewSession(binary)
	return []agent.Tool{computeruse.Tool(s.extras.desktop)}
}

// apertureSessionStart is the session_start work of both aperture
// extensions: notify pending onboarding, refresh the dedicated catalog and
// gateway snapshot in the background, surface sync warnings, and register
// the connector tools once the gateway answers.
func (s *session) apertureSessionStart(toolsEnabled bool) {
	configured, exists, err := loadApertureConfig()
	if err != nil {
		s.startupNotices = append(s.startupNotices, "[aperture] "+err.Error())
		return
	}
	if !exists {
		return
	}
	resolved := configured.Resolve()
	if resolved.OnboardingEnabled && !resolved.OnboardingDone {
		s.startupNotices = append(s.startupNotices, "[aperture] extension installed. Run /aperture onboarding to configure.")
	}
	if resolved.BaseURL == "" {
		return
	}

	// The networked refresh must not block startup: the cached catalog keeps
	// models loading instantly (even offline), and this revalidates it.
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		result, syncErr := aperture.Sync(ctx, configured, config.ApertureCachePath(), allCatalogModels())
		if syncErr != nil {
			s.pushNotice("aperture", "model refresh failed: "+syncErr.Error())
		}
		for _, warning := range result.Warnings {
			s.pushNotice("aperture", warning)
		}

		if !resolved.ConnectorsEnabled || !toolsEnabled {
			return
		}
		gatewayURL := aperture.GatewayURL(resolved.BaseURL)
		mcpSession, mcpErr := aperture.NewMcpSession(ctx, gatewayURL)
		if mcpErr != nil {
			s.pushNotice("aperture", "[connectors] connector session failed: "+mcpErr.Error())
			return
		}
		gatewayTools, listErr := mcpSession.ListTools(ctx)
		if listErr != nil {
			s.pushNotice("aperture", "[connectors] connector tools/list failed: "+listErr.Error())
			return
		}
		connectors, connectorsErr := aperture.NewClient(gatewayURL).Connectors(ctx)
		if connectorsErr != nil {
			// The list tool degrades to grouping everything under "other".
			connectors = nil
		}
		set := aperture.BuildConnectorTools(resolved, connectors, gatewayTools, s.extras.session)
		if len(set.MissingPins) > 0 {
			s.pushNotice("aperture", "[connectors] pinned tool(s) not found on gateway: "+strings.Join(set.MissingPins, ", "))
		}
		s.extras.mu.Lock()
		s.extras.connectorSession = mcpSession
		s.extras.connectorTools = set.Tools
		s.extras.mu.Unlock()
		if len(set.Tools) > 0 && s.agent != nil {
			s.agent.SetTools(s.planRuntimeTools())
		}
	}()
}

// apertureSessionID is the live session id sent as x-session-id, which must
// track /fork, /new, and /resume rather than being baked into registration.
func (s *session) apertureSessionID() string {
	if s.log != nil {
		return s.log.id()
	}
	return fmt.Sprintf("goshcoder-%d", os.Getpid())
}
