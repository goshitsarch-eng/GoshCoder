package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/ralph"
	"goshcoder/internal/tools"
)

func TestPlanPrepareNextTurnEnablesFullToolsImmediatelyAfterApproval(t *testing.T) {
	root := t.TempDir()
	workspace, err := tools.NewWorkspace(root)
	if err != nil {
		t.Fatal(err)
	}
	defer workspace.Close()
	manager, err := plannotator.New(root, filepath.Join(t.TempDir(), "state.json"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "plan.md"), []byte("- [ ] implement it\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	session := &session{baseSystemPrompt: "base", plan: manager, workspace: workspace, normalTools: workspace.All()}
	planningTools := session.planRuntimeTools()
	if hasTool(planningTools, "bash") {
		t.Fatal("planning tools unexpectedly contain bash")
	}
	if _, err := manager.Submit(t.Context(), "plan.md"); err != nil {
		t.Fatal(err)
	}
	update := session.planPrepareNextTurn()(t.Context(), agent.PrepareNextTurnContext{
		Context: agent.Context{Tools: planningTools},
	})
	if update == nil || update.Context == nil || !hasTool(update.Context.Tools, "bash") {
		t.Fatalf("executing tools do not contain bash: %#v", update)
	}
}

func hasTool(available []agent.Tool, name string) bool {
	for _, tool := range available {
		if tool.Name == name {
			return true
		}
	}
	return false
}

func TestPlanPrepareNextTurnDoesNotDuplicateModePrompt(t *testing.T) {
	manager, err := plannotator.New(t.TempDir(), filepath.Join(t.TempDir(), "state.json"), nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := manager.Enter(); err != nil {
		t.Fatal(err)
	}
	session := &session{baseSystemPrompt: "base prompt", plan: manager}
	hook := session.planPrepareNextTurn()
	initial := session.runtimeSystemPrompt()
	update := hook(context.Background(), agent.PrepareNextTurnContext{Context: agent.Context{SystemPrompt: initial}})
	if update == nil || update.Context == nil {
		t.Fatal("plan hook returned no context")
	}
	if count := strings.Count(update.Context.SystemPrompt, "[PLANNER - PLANNING PHASE]"); count != 1 {
		t.Fatalf("planning prompt count = %d\n%s", count, update.Context.SystemPrompt)
	}
}

// TestPlanRuntimeToolsKeepRalphTools covers a regression that silently disabled
// Ralph in the default chat mode. The Ralph tools were appended to the agent's
// initial tool list, but every rebuild in planRuntimeTools started from
// normalTools -- a snapshot taken before that append -- and runTurn applies a
// rebuild before every prompt. The model kept being told to call ralph_done by
// the system prompt suffix while the tool itself was no longer registered.
func TestPlanRuntimeToolsKeepRalphTools(t *testing.T) {
	root := t.TempDir()
	workspace, err := tools.NewWorkspace(root)
	if err != nil {
		t.Fatal(err)
	}
	defer workspace.Close()

	manager, err := plannotator.New(root, filepath.Join(t.TempDir(), "state.json"), nil)
	if err != nil {
		t.Fatal(err)
	}

	store := ralph.NewStore(root, "test-session")
	session := &session{
		baseSystemPrompt: "base",
		plan:             manager,
		workspace:        workspace,
		normalTools:      workspace.All(),
		loops:            store,
	}
	session.loopTools = store.Tools(agentQueue{&session.agent})

	for _, phase := range []string{"idle", "planning", "executing"} {
		switch phase {
		case "planning":
			if err := manager.Enter(); err != nil {
				t.Fatal(err)
			}
		case "executing":
			if err := os.WriteFile(filepath.Join(root, "plan.md"), []byte("- [ ] do it\n"), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := manager.Submit(t.Context(), "plan.md"); err != nil {
				t.Fatal(err)
			}
		}
		runtime := session.planRuntimeTools()
		for _, name := range []string{"ralph_start", "ralph_done"} {
			if !hasTool(runtime, name) {
				t.Fatalf("%s phase: %s is missing from the runtime tools", phase, name)
			}
		}
	}
}
