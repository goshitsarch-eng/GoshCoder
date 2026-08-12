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
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/ralph"
	"goshcoder/internal/tools"
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
	// model and auth are retained so /model can swap them at runtime.
	model     *llm.Model
	auth      *catalog.Auth
	loops     *ralph.Store
	workspace *tools.Workspace
	plan      *plannotator.Manager
	// baseSystemPrompt is the prompt without extension suffixes.
	baseSystemPrompt string
	normalTools      []agent.Tool
	claudeTUI        bool
	fullscreen       bool
}

// newSession resolves credentials, builds the tool set, and constructs the
// agent. It returns a usage error when the model is missing or unsupported.
func newSession(cfg sessionConfig) (*session, error) {
	if cfg.ModelRef == "" {
		return nil, errors.New("a model is required (-m provider/model); run 'goshcoder models' to see options")
	}

	model, auth, err := newCatalog().ResolveModel(cfg.ModelRef)
	if err != nil {
		return nil, err
	}
	if _, ok := llm.GetStreamer(model.API); !ok {
		return nil, fmt.Errorf("model %s uses the %q protocol, which is not implemented yet", model.ID, model.API)
	}

	s := &session{model: model, auth: auth, baseSystemPrompt: cfg.SystemPrompt, claudeTUI: cfg.ClaudeTUI, fullscreen: cfg.Fullscreen}

	var agentTools []agent.Tool
	if cfg.EnableTools || cfg.LoadPlannotator || cfg.EnablePlannotator {
		workspace, err := tools.NewWorkspace(cfg.Workdir)
		if err != nil {
			return nil, err
		}
		s.workspace = workspace
		if cfg.EnableTools {
			agentTools = workspace.All()
		}
		if !cfg.Quiet && cfg.EnableTools {
			// The bash tool runs arbitrary commands with the user's privileges.
			fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
				"tools enabled in %s (includes shell execution)", workspace.Root)))
		}
	}

	s.normalTools = append([]agent.Tool(nil), agentTools...)
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
		agentTools = append(agentTools, s.loops.Tools(agentQueue{&s.agent})...)
	}

	systemPrompt := cfg.SystemPrompt
	if s.plan != nil {
		systemPrompt = s.plan.Prompt(systemPrompt)
	}
	if s.loops != nil {
		if state, ok := s.loops.Current(); ok {
			systemPrompt = cfg.SystemPrompt + ralph.SystemPromptSuffix(state)
			if !cfg.Quiet {
				fmt.Fprintf(os.Stderr, "%s\n", dim(s.loops.StatusLine()))
			}
		}
	}

	s.agent = agent.NewAgent(agent.AgentOptions{
		InitialState: &agent.InitialState{
			SystemPrompt:  systemPrompt,
			Model:         *model,
			ThinkingLevel: llm.ClampThinkingLevel(model, cfg.Thinking),
			Tools:         agentTools,
		},
		// A custom StreamFn injects the resolved auth headers and provider env,
		// which the agent's default path does not carry.
		StreamFn: func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
			// Resolve auth at call time so /model can switch providers without
			// leaving the stream function bound to the session's first provider.
			return authStreamFn(s.auth)(model, ctx, opts)
		},
		GetAPIKey: func(string) string {
			if s.auth == nil {
				return ""
			}
			return s.auth.APIKey
		},
		BeforeToolCall: func(ctx context.Context, call agent.BeforeToolCallContext) *agent.BeforeToolCallResult {
			if s.plan == nil {
				return nil
			}
			return s.plan.BeforeToolCall(ctx, call)
		},
		// Compose native extension hooks in load order.
		PrepareNextTurn: composePrepareNextTurn(
			ralphPrepareNextTurn(s.loops, cfg.SystemPrompt),
			s.planPrepareNextTurn(),
		),
	})
	if !cfg.Fullscreen {
		s.agent.Subscribe(renderEvent)
	}
	return s, nil
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
		updated.SystemPrompt = s.plan.Prompt(updated.SystemPrompt)
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

func (s *session) syncPlanRuntime() {
	if s.plan == nil || s.agent == nil {
		return
	}
	s.agent.SetSystemPrompt(s.plan.Prompt(s.baseSystemPrompt))
	switch s.plan.State().Phase {
	case plannotator.PhaseIdle:
		s.agent.SetTools(s.normalTools)
	case plannotator.PhasePlanning:
		s.agent.SetTools(mergeTools(withoutTool(s.normalTools, "bash"), s.workspace.Planning(), []agent.Tool{s.plan.Tool()}))
	case plannotator.PhaseExecuting:
		s.agent.SetTools(mergeTools(s.normalTools, s.workspace.All(), []agent.Tool{s.plan.Tool()}))
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
	model, auth, err := newCatalog().ResolveModel(ref)
	if err != nil {
		return err
	}
	if _, ok := llm.GetStreamer(model.API); !ok {
		return fmt.Errorf("model %s uses the %q protocol, which is not implemented yet", model.ID, model.API)
	}
	s.model, s.auth = model, auth
	s.agent.SetModel(*model)
	// Model switches also update the UI-visible level. Provider streamers clamp
	// defensively, but keeping agent state valid avoids advertising an option
	// the newly selected model cannot use.
	current := s.agent.State().ThinkingLevel
	s.agent.SetThinkingLevel(llm.ClampThinkingLevel(model, current))
	return nil
}
