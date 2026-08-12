package main

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/agent"
	"goshcoder/internal/plannotator"
)

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
	if count := strings.Count(update.Context.SystemPrompt, "[PLANNOTATOR - PLANNING PHASE]"); count != 1 {
		t.Fatalf("planning prompt count = %d\n%s", count, update.Context.SystemPrompt)
	}
}
