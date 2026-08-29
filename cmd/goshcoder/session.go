package main

// Session construction shared by the one-shot `run` command and the
// interactive `chat` command.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"

	"goshcoder/internal/agent"
	"goshcoder/internal/btw"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/ralph"
	coderresources "goshcoder/internal/resources"
	"goshcoder/internal/tools"
	"goshcoder/internal/webaccess"
)

// sessionConfig is the resolved command-line configuration for a session.
type sessionConfig struct {
	ModelRef          string
	SystemPrompt      string
	Thinking          string
	Workdir           string
	EnableTools       bool
	EnableRalph       bool
	EnablePlannotator bool
	LoadPlannotator   bool
	ClaudeTUI         bool
	Fullscreen        bool
	// Quiet suppresses the startup banner.
	Quiet bool

	// Session persistence. SessionRef is a path, an exact id, or an id prefix.
	SessionRef  string
	Continue    bool
	Resume      bool
	NoSession   bool
	ReadOnly    bool
	SessionName string
	SessionsDir string
	// ModelFromFlag records whether ModelRef came from -m or from the
	// remembered default. Without it a resumed session cannot tell the user
	// overriding its model from the CLI merely filling in a default, and would
	// silently ignore the model the conversation was actually held on.
	ModelFromFlag bool
}

// session bundles an agent with the pieces its tools and hooks need.
type session struct {
	agent *agent.Agent
	// model is retained so /model can swap it at runtime.
	model     *llm.Model
	loops     *ralph.Store
	btw       *btw.Manager
	workspace *tools.Workspace
	plan      *plannotator.Manager
	// promptMu guards baseSystemPrompt, explicitSystemPrompt and resources.
	//
	// /reload and /system rewrite all three from whichever goroutine runs the
	// slash command -- the Bubble Tea update loop in the fullscreen UI -- while
	// the agent's PrepareNextTurn hook reads baseSystemPrompt between tool
	// turns on the run goroutine. A Go string is a two-word value with no
	// atomicity, so an unsynchronized overlap can hand the agent one string's
	// pointer with another's length.
	promptMu sync.RWMutex
	// baseSystemPrompt is the prompt without extension suffixes.
	baseSystemPrompt     string
	explicitSystemPrompt string
	resources            *coderresources.Set
	normalTools          []agent.Tool
	// contextEstimate caches the transcript's token estimate; see its type.
	contextEstimate contextEstimate
	// noticeMu guards pendingNotices.
	noticeMu sync.Mutex
	// pendingNotices carries messages raised from agent-goroutine work -- the
	// Planner's review URL, for instance -- to whichever interface is running.
	// The fullscreen UI owns the screen, so such work cannot write to stderr;
	// dropping the message instead left the Planner blocking on a review the
	// user had no way to reach.
	pendingNotices []sessionNotice
	// loopTools are the Ralph tools. They are kept separately from
	// normalTools because planRuntimeTools rebuilds the agent's tool list from
	// normalTools before every turn, and anything not merged back in there is
	// silently dropped from the running agent.
	loopTools  []agent.Tool
	claudeTUI  bool
	fullscreen bool
	// extras carries the native Aperture connector tools and the
	// computer-use-linux desktop session (see aperture_session.go).
	extras apertureExtras

	// log records the conversation to disk. It is nil only when -no-session
	// was given, and every call site tolerates that rather than branching.
	log *sessionRecorder
	// startupNotices are messages produced while opening the session, shown
	// once by whichever interface starts.
	startupNotices []string
	// resumeBanner describes a reopened session, or is empty for a new one.
	resumeBanner string
	// restoredMessages is the transcript a resume folded back, retained so the
	// line-oriented interfaces can print it after construction.
	restoredMessages []agent.Message
}

// newSession resolves credentials, builds the tool set, and constructs the
// agent. It returns a usage error when the model is missing or unsupported.
func newSession(cfg sessionConfig) (*session, error) {
	resourceRoot, err := filepath.Abs(cfg.Workdir)
	if err != nil {
		return nil, err
	}

	// The session is opened before the model is resolved so that a resumed
	// conversation's own model can supply the default. Doing this the other way
	// round -- as this function used to -- means chatCommand's remembered
	// default is already in ModelRef, and the recorded model is silently
	// ignored on every resume.
	opened, err := openSession(cfg, resourceRoot)
	if err != nil {
		return nil, err
	}
	modelRef, modelNotices := resolveResumedModel(cfg, opened, cfg.ModelFromFlag)
	if modelRef == "" {
		if opened.writer != nil {
			_ = opened.writer.Close()
		}
		return nil, errors.New("a model is required (-m provider/model); run 'goshcoder models' to see options")
	}

	model, _, err := newCatalog().ResolveModel(modelRef)
	if err != nil {
		if opened.writer != nil {
			_ = opened.writer.Close()
		}
		return nil, err
	}
	if _, ok := llm.GetStreamer(model.API); !ok {
		if opened.writer != nil {
			_ = opened.writer.Close()
		}
		return nil, fmt.Errorf("model %s uses the %q protocol, which is not implemented yet", model.ID, model.API)
	}
	loadedResources, err := coderresources.Discover(resourceRoot, config.AgentDir())
	if err != nil {
		closeOpened(opened)
		return nil, fmt.Errorf("discover local resources: %w", err)
	}
	s := &session{
		model: model, btw: btw.NewManager(), explicitSystemPrompt: cfg.SystemPrompt,
		resources: loadedResources, claudeTUI: cfg.ClaudeTUI, fullscreen: cfg.Fullscreen,
		startupNotices:   append(append([]string(nil), opened.Notices...), modelNotices...),
		restoredMessages: opened.restored.Messages,
	}
	if opened.writer != nil {
		s.log = newSessionRecorder(opened.writer, s.pushNotice)
	}
	s.resumeBanner = resumeBanner(opened, s.log)

	var agentTools []agent.Tool
	if cfg.EnableTools || cfg.LoadPlannotator || cfg.EnablePlannotator {
		workspace, err := tools.NewWorkspace(cfg.Workdir)
		if err != nil {
			closeOpened(opened)
			return nil, err
		}
		s.workspace = workspace
		if cfg.EnableTools {
			agentTools = append(workspace.All(), webaccess.New(
				config.WebSearchPath(), resolveOpenAIWebSearchAuth,
			).Tool())
			// Native computer-use-linux adaptation: the desktop mcp proxy
			// tool, registered when the server binary is discoverable.
			agentTools = append(agentTools, s.desktopMCPTool(cfg.Quiet)...)
		}
		if !cfg.Quiet && cfg.EnableTools {
			// The bash tool runs arbitrary commands with the user's privileges.
			fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
				"tools enabled in %s (includes shell execution)", workspace.Root)))
		}
	}

	s.normalTools = append([]agent.Tool(nil), agentTools...)
	promptTools := s.normalTools
	if len(promptTools) == 0 && (cfg.LoadPlannotator || cfg.EnablePlannotator) {
		promptTools = s.workspace.Planning()
	}
	toolNames := make([]string, 0, len(promptTools))
	for _, tool := range promptTools {
		toolNames = append(toolNames, tool.Name)
	}
	s.setSystemPromptBase(s.resources.BuildSystemPrompt(cfg.SystemPrompt, resourceRoot, toolNames))
	if !cfg.Quiet {
		for _, warning := range s.resources.Warnings {
			fmt.Fprintln(os.Stderr, dim("resource warning: "+warning))
		}
	}
	if cfg.LoadPlannotator || cfg.EnablePlannotator {
		root := s.workspace.Root
		reviewer := plannotator.BrowserReviewer{Notify: func(message string) {
			if cfg.Fullscreen {
				// The fullscreen UI owns the terminal, so this cannot go to
				// stderr; it is queued for the interface to render instead.
				s.pushNotice("Planner", message)
				return
			}
			fmt.Fprintln(os.Stderr, message)
		}}
		// Plan state now travels in the session log as a custom entry, which
		// is what pi documents that entry type for: extension state that
		// survives a reload and never participates in context.
		initial := restoredPlannerState(opened, root, &s.startupNotices)
		manager, err := plannotator.New(root, reviewer, plannotator.Options{
			Initial: initial,
			OnChange: func(state plannotator.State) {
				s.log.recordCustom(plannerCustomType, state)
				// The adopted per-workspace file is only retired once its
				// contents have actually been written somewhere durable.
				retireLegacyPlannerState(root)
			},
			Warn: func(message string) { s.pushNotice("Planner", message) },
		})
		if err != nil {
			_ = s.workspace.Close()
			closeOpened(opened)
			return nil, err
		}
		s.plan = manager
		if cfg.EnablePlannotator && manager.State().Phase == plannotator.PhaseIdle {
			if err := manager.Enter(); err != nil {
				_ = s.workspace.Close()
				closeOpened(opened)
				return nil, err
			}
		}
		switch manager.State().Phase {
		case plannotator.PhasePlanning:
			agentTools = mergeTools(withoutTool(s.normalTools, "bash"), s.workspace.Planning(), []agent.Tool{manager.Tool()})
		case plannotator.PhaseExecuting:
			agentTools = mergeTools(s.normalTools, s.workspace.All(), []agent.Tool{manager.Tool()})
		}
		if !cfg.Quiet && manager.State().Phase != plannotator.PhaseIdle {
			fmt.Fprintln(os.Stderr, dim(manager.StatusLine()))
		}
	}

	// Ralph loops need somewhere to keep state, and a session id so two CLIs in
	// one workspace do not drive the same loop.
	if cfg.EnableRalph {
		root, err := filepath.Abs(cfg.Workdir)
		if err != nil {
			if s.workspace != nil {
				_ = s.workspace.Close()
			}
			closeOpened(opened)
			return nil, err
		}
		s.loops = ralph.NewStore(root, fmt.Sprintf("cli-%d", os.Getpid()))
		// The ralph tools queue follow-ups on the agent, so they are registered
		// against a pointer the agent is assigned to below.
		s.loopTools = s.loops.Tools(agentQueue{&s.agent})
		agentTools = append(agentTools, s.loopTools...)
	}

	systemPrompt := s.runtimeSystemPrompt()
	if s.loops != nil {
		if _, ok := s.loops.Current(); ok && !cfg.Quiet {
			fmt.Fprintf(os.Stderr, "%s\n", dim(s.loops.StatusLine()))
		}
	}

	s.agent = agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{
			SystemPrompt: systemPrompt,
			Model:        *model,
			// A resumed session's own thinking level wins over the flag's
			// default, and is re-clamped because the model it was recorded
			// against may not be the model this process ended up on.
			ThinkingLevel: llm.ClampThinkingLevel(model, resumedThinking(cfg.Thinking, opened)),
			Tools:         agentTools,
			// Seeded here rather than through SetMessages after construction:
			// InitialState emits nothing, so the restored transcript is not
			// re-recorded into the log it just came out of.
			Messages: opened.restored.Messages,
		},
		// Resolve credentials for every provider call. Besides following /model,
		// this refreshes OAuth during long-running sessions and observes a
		// credential added or replaced through /login without restarting.
		StreamFn:     s.streamAuthenticated,
		ConvertToLLM: convertSessionMessages,
		// Brief provider outages should recover in-place instead of forcing the
		// user to restart the app and reconstruct the task with "continue".
		MaxRetries: 2,
		BeforeToolCall: func(ctx context.Context, call agent.BeforeToolCallContext) *agent.BeforeToolCallResult {
			if s.plan == nil {
				return nil
			}
			return s.plan.BeforeToolCall(ctx, call)
		},
		// Compose native extension hooks in load order.
		PrepareNextTurn: composePrepareNextTurn(
			ralphPrepareNextTurn(s.loops, s.systemPromptBase),
			s.planPrepareNextTurn(),
		),
	})
	// Subscribed unconditionally and before any renderer: one registration
	// covers line mode, fullscreen, chat and run, and going first means an
	// entry is durable before anything draws it.
	if s.log != nil {
		s.agent.Subscribe(s.log.listener())
	}
	if !cfg.Fullscreen {
		s.agent.Subscribe(renderEvent)
	}
	if cfg.SessionName != "" && s.log != nil {
		s.log.setName(cfg.SessionName)
	}
	// Native Aperture adaptation: onboarding notice, background catalog
	// refresh, and connector tool registration.
	s.apertureSessionStart(cfg.EnableTools)
	return s, nil
}

// closeOpened releases a session opened before construction failed.
//
// Every error path after openSession has to do this. Returning without it left
// the claim held for the stale threshold and a stub session file on disk, so a
// mistyped -C directory made the next launch report the workspace as busy.
func closeOpened(opened *openedSession) {
	if opened != nil && opened.writer != nil {
		_ = opened.writer.Close()
	}
}

// resumedThinking prefers the level a resumed session recorded over the flag
// default, since -thinking carries a default value on every invocation and
// would otherwise silently reset a session that was running at "high".
func resumedThinking(flagLevel string, opened *openedSession) string {
	if opened != nil && opened.Resumed && opened.restored.ThinkingLevel != "" {
		return opened.restored.ThinkingLevel
	}
	return flagLevel
}

// resolveOpenAIWebSearchAuth gives the native web search port the same
// credential reuse as pi-web-access: a stored Codex subscription is preferred,
// then a regular OpenAI API credential. Resolving per call also refreshes OAuth
// tokens without coupling the search service to the active chat model.
func resolveOpenAIWebSearchAuth(ctx context.Context) (*webaccess.OpenAIAuth, error) {
	var firstError error
	for _, providerID := range []string{"openai-codex", "openai"} {
		models := newCatalog()
		models.SetOAuthContext(ctx)
		auth, ok := models.ResolveAuth(providerID)
		if !ok {
			if err := models.OAuthError(providerID); err != nil && firstError == nil {
				firstError = err
			}
			continue
		}
		modelID := "gpt-5.6-terra"
		if models.Model(providerID, modelID) == nil {
			provider := models.Provider(providerID)
			if provider == nil {
				continue
			}
			providerModels := provider.Models()
			if len(providerModels) == 0 {
				continue
			}
			modelID = providerModels[len(providerModels)-1].ID
		}
		return &webaccess.OpenAIAuth{
			Provider: providerID,
			APIKey:   auth.APIKey,
			Model:    modelID,
			Headers:  auth.Headers,
		}, nil
	}
	return nil, firstError
}

func (s *session) streamAuthenticated(model *llm.Model, request *llm.Context, options *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
	ctx := context.Background()
	if options != nil && options.Ctx != nil {
		ctx = options.Ctx
	}
	models := newCatalog()
	// A real request carries its own cancellation, so the incidental cap must
	// not cut a legitimate refresh short.
	models.SetOAuthTimeout(0)
	models.SetOAuthContext(ctx)
	auth, ok := models.ResolveAuth(model.Provider)
	if !ok {
		if err := models.OAuthError(model.Provider); err != nil {
			return errorStream(model, fmt.Sprintf("refresh credentials for %s: %v", model.Provider, err))
		}
		return errorStream(model, fmt.Sprintf("provider %q has no credentials configured", model.Provider))
	}
	// Native Aperture adaptation: gateway-routed requests carry the
	// provider-qualified model id and the provenance headers, and transient
	// gateway errors are tagged retryable (aperture_session.go).
	apertureState := models.ApertureState()
	routedModel := apertureRequestModel(apertureState, model, s.apertureSessionID())
	stream := authStreamFn(auth)(routedModel, request, options)
	if routedModel != model {
		stream = markApertureRetryable(stream)
	}
	return stream
}

// sessionNotice is one message raised for the user from background work.
type sessionNotice struct {
	Kind string
	Text string
}

// pushNotice queues a message for the interface to display.
func (s *session) pushNotice(kind, text string) {
	s.noticeMu.Lock()
	defer s.noticeMu.Unlock()
	// Bounded: a runaway producer must not grow this without limit.
	if len(s.pendingNotices) >= 64 {
		s.pendingNotices = s.pendingNotices[1:]
	}
	s.pendingNotices = append(s.pendingNotices, sessionNotice{Kind: kind, Text: text})
}

// drainNotices removes and returns the queued messages.
func (s *session) drainNotices() []sessionNotice {
	s.noticeMu.Lock()
	defer s.noticeMu.Unlock()
	if len(s.pendingNotices) == 0 {
		return nil
	}
	drained := s.pendingNotices
	s.pendingNotices = nil
	return drained
}

// systemPromptBase returns the prompt without extension suffixes.
func (s *session) systemPromptBase() string {
	s.promptMu.RLock()
	defer s.promptMu.RUnlock()
	return s.baseSystemPrompt
}

// setSystemPromptBase replaces the prompt without extension suffixes.
func (s *session) setSystemPromptBase(prompt string) {
	s.promptMu.Lock()
	defer s.promptMu.Unlock()
	s.baseSystemPrompt = prompt
}

// setExplicitSystemPrompt records a user-supplied /system override.
func (s *session) setExplicitSystemPrompt(prompt string) {
	s.promptMu.Lock()
	defer s.promptMu.Unlock()
	s.explicitSystemPrompt = prompt
}

// currentResources returns the loaded resource set, which /reload replaces.
func (s *session) currentResources() *coderresources.Set {
	if s == nil {
		return nil
	}
	s.promptMu.RLock()
	defer s.promptMu.RUnlock()
	return s.resources
}

func (s *session) expandResourceInput(input string) (string, bool, error) {
	loaded := s.currentResources()
	if loaded == nil {
		return "", false, nil
	}
	return loaded.Expand(input)
}

func (s *session) reloadResources() error {
	loaded, err := coderresources.Discover(s.workspaceRoot(), config.AgentDir())
	if err != nil {
		return err
	}
	promptTools := s.normalTools
	if len(promptTools) == 0 && s.plan != nil && s.workspace != nil {
		promptTools = s.workspace.Planning()
	}
	toolNames := make([]string, 0, len(promptTools))
	for _, tool := range promptTools {
		toolNames = append(toolNames, tool.Name)
	}

	s.promptMu.Lock()
	s.resources = loaded
	s.baseSystemPrompt = loaded.BuildSystemPrompt(s.explicitSystemPrompt, s.workspaceRoot(), toolNames)
	s.promptMu.Unlock()

	s.syncPlanRuntime()
	return nil
}

func (s *session) close() error {
	if s == nil {
		return nil
	}
	// The log closes first: it syncs, releases the claim, and deletes a session
	// that never got a reply. Doing it after the workspace would leave the
	// claim held for the stale threshold if closing the workspace failed.
	var firstErr error
	if err := s.log.close(); err != nil {
		firstErr = err
	}
	if s.workspace != nil {
		if err := s.workspace.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	s.extras.close()
	return firstErr
}

// recordingActive reports whether the conversation is reaching disk. The
// Ctrl+C confirmation and the sidebar both ask.
func (s *session) recordingActive() bool {
	return s != nil && s.log != nil && s.log.active()
}

// drainStartupNotices returns the messages produced while opening the session.
func (s *session) drainStartupNotices() []string {
	if s == nil {
		return nil
	}
	notices := s.startupNotices
	s.startupNotices = nil
	return notices
}

// handleInterrupts routes Ctrl-C to the agent so a run aborts and settles
// instead of killing the process. The returned function stops the handler.
//
// In interactive mode onIdleInterrupt is called when Ctrl-C arrives with no run
// in flight, which is how the REPL exits.
func (s *session) handleInterrupts(onIdleInterrupt func()) (stop func()) {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, os.Interrupt, syscall.SIGTERM)
	done := make(chan struct{})

	go func() {
		for {
			select {
			case <-done:
				return
			case _, ok := <-signals:
				if !ok {
					return
				}
				if s.agent.State().IsStreaming {
					fmt.Fprintln(os.Stderr, "\n"+dim("aborting..."))
					s.agent.Abort()
					continue
				}
				if onIdleInterrupt != nil {
					onIdleInterrupt()
					return
				}
			}
		}
	}()

	return func() {
		signal.Stop(signals)
		close(done)
	}
}

// runTurn prompts the agent and reports a run failure as an error.
func (s *session) runTurn(prompt string) error {
	s.syncPlanRuntime()
	if err := s.maybeAutoCompact(); err != nil {
		return fmt.Errorf("automatic context compaction: %w", err)
	}
	if err := s.agent.Prompt(prompt); err != nil {
		return err
	}
	if message := s.agent.State().ErrorMessage; message != "" {
		return errors.New(message)
	}
	s.syncPlanRuntime()
	if !s.fullscreen && s.plan != nil && s.plan.State().Phase != plannotator.PhaseIdle {
		fmt.Fprintln(os.Stderr, dim(s.plan.StatusLine()))
	}
	return nil
}

func (s *session) planPrepareNextTurn() agent.PrepareNextTurnFunc {
	if s.plan == nil {
		return nil
	}
	return func(_ context.Context, turn agent.PrepareNextTurnContext) *agent.TurnUpdate {
		s.plan.TrackAssistant(turn.Message)
		updated := turn.Context
		updated.SystemPrompt = s.runtimeSystemPrompt()
		// planner_submit_plan can change planning → executing inside an active
		// agent run. Update the low-level loop context now so the very next model
		// turn receives full tools rather than waiting for Prompt to return.
		if s.workspace != nil {
			updated.Tools = s.planRuntimeTools()
		}
		return &agent.TurnUpdate{Context: &updated}
	}
}

func composePrepareNextTurn(hooks ...agent.PrepareNextTurnFunc) agent.PrepareNextTurnFunc {
	return func(ctx context.Context, turn agent.PrepareNextTurnContext) *agent.TurnUpdate {
		updated := turn
		changed := false
		for _, hook := range hooks {
			if hook == nil {
				continue
			}
			if next := hook(ctx, updated); next != nil && next.Context != nil {
				updated.Context = *next.Context
				changed = true
			}
		}
		if !changed {
			return nil
		}
		return &agent.TurnUpdate{Context: &updated.Context}
	}
}

func (s *session) runtimeSystemPrompt() string {
	prompt := s.systemPromptBase()
	if s.loops != nil {
		if state, ok := s.loops.Current(); ok && state.Status == ralph.StatusActive {
			prompt += ralph.SystemPromptSuffix(state)
		}
	}
	if s.plan != nil {
		prompt = s.plan.Prompt(prompt)
	}
	return prompt
}

func (s *session) syncPlanRuntime() {
	if s.agent == nil {
		return
	}
	s.agent.SetSystemPrompt(s.runtimeSystemPrompt())
	if s.plan != nil && s.workspace != nil {
		s.agent.SetTools(s.planRuntimeTools())
	}
}

func (s *session) planRuntimeTools() []agent.Tool {
	// loopTools is merged into every branch: this list replaces the agent's
	// tools wholesale before each turn, so anything omitted here is
	// unregistered for the rest of the session even though the system prompt
	// still instructs the model to call it.
	if s.plan == nil || s.workspace == nil {
		return mergeTools(s.normalTools, s.loopTools, s.extras.tools())
	}
	switch s.plan.State().Phase {
	case plannotator.PhasePlanning:
		return mergeTools(withoutTool(s.normalTools, "bash"), s.workspace.Planning(), []agent.Tool{s.plan.Tool()}, s.loopTools, s.extras.tools())
	case plannotator.PhaseExecuting:
		return mergeTools(s.normalTools, s.workspace.All(), []agent.Tool{s.plan.Tool()}, s.loopTools, s.extras.tools())
	default:
		return mergeTools(s.normalTools, s.loopTools, s.extras.tools())
	}
}

func withoutTool(group []agent.Tool, name string) []agent.Tool {
	filtered := make([]agent.Tool, 0, len(group))
	for _, tool := range group {
		if tool.Name != name {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

func mergeTools(groups ...[]agent.Tool) []agent.Tool {
	seen := map[string]bool{}
	var merged []agent.Tool
	for _, group := range groups {
		for _, tool := range group {
			if seen[tool.Name] {
				continue
			}
			seen[tool.Name] = true
			merged = append(merged, tool)
		}
	}
	return merged
}

// setModel swaps the active model, re-resolving credentials for its provider.
func (s *session) setModel(ref string) error {
	model, _, err := newCatalog().ResolveModel(ref)
	if err != nil {
		return err
	}
	if _, ok := llm.GetStreamer(model.API); !ok {
		return fmt.Errorf("model %s uses the %q protocol, which is not implemented yet", model.ID, model.API)
	}
	s.model = model
	s.agent.SetModel(*model)
	// Persist every switch path, including Ctrl+P model cycling in fullscreen.
	_ = config.WriteDefaultModel(model.Provider + "/" + model.ID)
	// Model switches also update the UI-visible level. Provider streamers clamp
	// defensively, but keeping agent state valid avoids advertising an option
	// the newly selected model cannot use.
	current := s.agent.State().ThinkingLevel
	s.agent.SetThinkingLevel(llm.ClampThinkingLevel(model, current))
	return nil
}
