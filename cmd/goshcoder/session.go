package main

// Session construction shared by the one-shot `run` command and the
// interactive `chat` command.

import (
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/ralph"
	"goshcoder/internal/tools"
)

// sessionConfig is the resolved command-line configuration for a session.
type sessionConfig struct {
	ModelRef     string
	SystemPrompt string
	Thinking     string
	Workdir      string
	EnableTools  bool
	EnableRalph  bool
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
	// baseSystemPrompt is the prompt without any ralph loop suffix.
	baseSystemPrompt string
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

	s := &session{model: model, auth: auth, baseSystemPrompt: cfg.SystemPrompt}

	var agentTools []agent.Tool
	if cfg.EnableTools {
		workspace, err := tools.NewWorkspace(cfg.Workdir)
		if err != nil {
			return nil, err
		}
		s.workspace = workspace
		agentTools = workspace.All()
		if !cfg.Quiet {
			// The bash tool runs arbitrary commands with the user's privileges.
			fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
				"tools enabled in %s (includes shell execution)", workspace.Root)))
		}
	}

	// Ralph loops need somewhere to keep state, and a session id so two CLIs in
	// one workspace do not drive the same loop.
	if cfg.EnableRalph {
		root, err := filepath.Abs(cfg.Workdir)
		if err != nil {
			return nil, err
		}
		s.loops = ralph.NewStore(root, fmt.Sprintf("cli-%d", os.Getpid()))
		// The ralph tools queue follow-ups on the agent, so they are registered
		// against a pointer the agent is assigned to below.
		agentTools = append(agentTools, s.loops.Tools(agentQueue{&s.agent})...)
	}

	systemPrompt := cfg.SystemPrompt
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
			ThinkingLevel: cfg.Thinking,
			Tools:         agentTools,
		},
		// A custom StreamFn injects the resolved auth headers and provider env,
		// which the agent's default path does not carry.
		StreamFn:  authStreamFn(auth),
		GetAPIKey: func(string) string { return auth.APIKey },
		// Refresh the loop banner between turns as the iteration advances, and
		// close the loop out when the model emits the completion marker.
		PrepareNextTurn: ralphPrepareNextTurn(s.loops, cfg.SystemPrompt),
	})
	s.agent.Subscribe(renderEvent)
	return s, nil
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
	if err := s.agent.Prompt(prompt); err != nil {
		return err
	}
	if message := s.agent.State().ErrorMessage; message != "" {
		return errors.New(message)
	}
	return nil
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
	return nil
}
