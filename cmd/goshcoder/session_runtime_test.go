package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"fmt"
	"goshcoder/internal/agent"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/ralph"
	"goshcoder/internal/tools"
	"sync"
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

// TestSystemPromptAccessIsSynchronized covers a data race between /reload or
// /system -- which run on whichever goroutine handles the slash command, the
// Bubble Tea update loop in the fullscreen UI -- and the agent's PrepareNextTurn
// hook, which reads the base prompt between tool turns on the run goroutine.
// A Go string is a two-word value with no atomicity, so an unsynchronized
// overlap can pair one string's pointer with another's length.
//
// Both sides run the same fixed number of iterations behind a start barrier so
// they genuinely overlap; without that the reader can finish before the writer
// is scheduled and the race detector never sees the conflicting pair.
func TestSystemPromptAccessIsSynchronized(t *testing.T) {
	const iterations = 5000

	session := &session{}
	session.setSystemPromptBase("initial prompt")

	start := make(chan struct{})
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for i := range iterations {
			session.setSystemPromptBase(strings.Repeat("x", i%64+1))
			session.setExplicitSystemPrompt(strings.Repeat("y", i%32+1))
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for range iterations {
			_ = session.systemPromptBase()
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		for range iterations {
			_ = session.currentResources()
		}
	}()

	close(start)
	wg.Wait()
}

// TestPlannerNoticesReachTheInterface covers how a message raised from the
// agent goroutine gets to the user. The Planner's browser review runs inside a
// tool call and its notice was written to stderr only in line mode, so in the
// fullscreen UI -- the default on any terminal -- the review URL was dropped.
// The port is ephemeral, so with no browser to open (a headless box with no
// xdg-open) the tool then blocked on a review the user had no way to reach.
func TestPlannerNoticesReachTheInterface(t *testing.T) {
	s := &session{}

	if got := s.drainNotices(); got != nil {
		t.Fatalf("a fresh session had notices: %#v", got)
	}

	s.pushNotice("Planner", "Planner review: http://127.0.0.1:54321/")
	drained := s.drainNotices()
	if len(drained) != 1 || !strings.Contains(drained[0].Text, "127.0.0.1:54321") {
		t.Fatalf("drained = %#v", drained)
	}
	if got := s.drainNotices(); got != nil {
		t.Fatalf("notices were delivered twice: %#v", got)
	}
}

// TestSessionNoticesAreBounded keeps a runaway producer from growing the queue
// without limit.
func TestSessionNoticesAreBounded(t *testing.T) {
	s := &session{}
	for i := range 500 {
		s.pushNotice("Planner", fmt.Sprintf("notice %d", i))
	}
	drained := s.drainNotices()
	if len(drained) > 64 {
		t.Fatalf("queue grew to %d entries", len(drained))
	}
	// The most recent must survive; it is the one that matters.
	if !strings.Contains(drained[len(drained)-1].Text, "notice 499") {
		t.Fatalf("newest notice lost: %q", drained[len(drained)-1].Text)
	}
}

// TestSessionNoticesAreConcurrencySafe: notices are pushed from the agent
// goroutine and drained from the UI goroutine.
func TestSessionNoticesAreConcurrencySafe(t *testing.T) {
	s := &session{}
	var wg sync.WaitGroup
	start := make(chan struct{})

	wg.Add(2)
	go func() {
		defer wg.Done()
		<-start
		for i := range 1000 {
			s.pushNotice("Planner", fmt.Sprintf("n%d", i))
		}
	}()
	go func() {
		defer wg.Done()
		<-start
		for range 1000 {
			_ = s.drainNotices()
		}
	}()
	close(start)
	wg.Wait()
}
