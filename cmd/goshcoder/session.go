package main

// Session construction shared by the one-shot `run` command and the
// interactive `chat` command.

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
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
	// baseSystemPrompt is the prompt without extension suffixes.
	baseSystemPrompt     string
	explicitSystemPrompt string
	resources            *coderresources.Set
	normalTools          []agent.Tool
	// loopTools are the Ralph tools. They are kept separately from
	// normalTools because planRuntimeTools rebuilds the agent's tool list from
	// normalTools before every turn, and anything not merged back in there is
	// silently dropped from the running agent.
	loopTools  []agent.Tool
	claudeTUI  bool
	fullscreen bool
}

// newSession resolves credentials, builds the tool set, and constructs the
// agent. It returns a usage error when the model is missing or unsupported.
func newSession(cfg sessionConfig) (*session, error) {
	if cfg.ModelRef == "" {
		return nil, errors.New("a model is required (-m provider/model); run 'goshcoder models' to see options")
	}

	model, _, err := newCatalog().ResolveModel(cfg.ModelRef)
	if err != nil {
		return nil, err
	}
	if _, ok := llm.GetStreamer(model.API); !ok {
		return nil, fmt.Errorf("model %s uses the %q protocol, which is not implemented yet", model.ID, model.API)
	}

	resourceRoot, err := filepath.Abs(cfg.Workdir)
	if err != nil {
		return nil, err
	}
	loadedResources, err := coderresources.Discover(resourceRoot, config.AgentDir())
	if err != nil {
		return nil, fmt.Errorf("discover local resources: %w", err)
	}
	s := &session{
		model: model, btw: btw.NewManager(), explicitSystemPrompt: cfg.SystemPrompt,
		resources: loadedResources, claudeTUI: cfg.ClaudeTUI, fullscreen: cfg.Fullscreen,
	}

	var agentTools []agent.Tool
	if cfg.EnableTools || cfg.LoadPlannotator || cfg.EnablePlannotator {
		workspace, err := tools.NewWorkspace(cfg.Workdir)
		if err != nil {
			return nil, err
		}
		s.workspace = workspace
		if cfg.EnableTools {
			agentTools = append(workspace.All(), webaccess.New(
				config.WebSearchPath(), resolveOpenAIWebSearchAuth,
			).Tool())
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
	s.baseSystemPrompt = s.resources.BuildSystemPrompt(cfg.SystemPrompt, resourceRoot, toolNames)
	if !cfg.Quiet {
		for _, warning := range s.resources.Warnings {
			fmt.Fprintln(os.Stderr, dim("resource warning: "+warning))
		}
	}
	if cfg.LoadPlannotator || cfg.EnablePlannotator {
		root := s.workspace.Root
		stateID := fmt.Sprintf("%x", sha256.Sum256([]byte(root)))[:16]
		reviewer := plannotator.BrowserReviewer{Notify: func(message string) {
			if !cfg.Fullscreen {
				fmt.Fprintln(os.Stderr, message)
			}
		}}
		manager, err := plannotator.New(root, filepath.Join(config.AgentDir(), "plannotator", stateID+".json"), reviewer)
		if err != nil {
			_ = s.workspace.Close()
			return nil, err
		}
		s.plan = manager
		if cfg.EnablePlannotator && manager.State().Phase == plannotator.PhaseIdle {
			if err := manager.Enter(); err != nil {
				_ = s.workspace.Close()
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
			SystemPrompt:  systemPrompt,
			Model:         *model,
			ThinkingLevel: llm.ClampThinkingLevel(model, cfg.Thinking),
			Tools:         agentTools,
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
			ralphPrepareNextTurn(s.loops, func() string { return s.baseSystemPrompt }),
			s.planPrepareNextTurn(),
		),
	})
	if !cfg.Fullscreen {
		s.agent.Subscribe(renderEvent)
	}
	return s, nil
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
	models.SetOAuthContext(ctx)
	auth, ok := models.ResolveAuth(model.Provider)
	if !ok {
		if err := models.OAuthError(model.Provider); err != nil {
			return errorStream(model, fmt.Sprintf("refresh credentials for %s: %v", model.Provider, err))
		}
		return errorStream(model, fmt.Sprintf("provider %q has no credentials configured", model.Provider))
	}
	return authStreamFn(auth)(model, request, options)
}

func (s *session) expandResourceInput(input string) (string, bool, error) {
	if s == nil || s.resources == nil {
		return "", false, nil
	}
	return s.resources.Expand(input)
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
	s.resources = loaded
	s.baseSystemPrompt = loaded.BuildSystemPrompt(s.explicitSystemPrompt, s.workspaceRoot(), toolNames)
	s.syncPlanRuntime()
	return nil
}

func (s *session) close() error {
	if s == nil || s.workspace == nil {
		return nil
	}
	return s.workspace.Close()
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
	prompt := s.baseSystemPrompt
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
		return mergeTools(s.normalTools, s.loopTools)
	}
	switch s.plan.State().Phase {
	case plannotator.PhasePlanning:
		return mergeTools(withoutTool(s.normalTools, "bash"), s.workspace.Planning(), []agent.Tool{s.plan.Tool()}, s.loopTools)
	case plannotator.PhaseExecuting:
		return mergeTools(s.normalTools, s.workspace.All(), []agent.Tool{s.plan.Tool()}, s.loopTools)
	default:
		return mergeTools(s.normalTools, s.loopTools)
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
