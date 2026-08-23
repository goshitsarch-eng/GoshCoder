// Package plannotator provides a native Go adaptation of
// @plannotator/pi-extension. It preserves the extension's planning state
// machine, write gate, checklist tracking, and human review workflow without a
// JavaScript extension host.
package plannotator

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

const SubmitToolName = "planner_submit_plan"
const maxPlanBytes = 2 * 1024 * 1024

type Phase string

const (
	PhaseIdle      Phase = "idle"
	PhasePlanning  Phase = "planning"
	PhaseExecuting Phase = "executing"
)

type ChecklistItem struct {
	Step      int    `json:"step"`
	Text      string `json:"text"`
	Completed bool   `json:"completed"`
}

type Decision struct {
	Approved bool
	Feedback string
}

type Reviewer interface {
	Review(ctx context.Context, title, markdown string) (Decision, error)
}

type versionedReviewer interface {
	ReviewVersion(ctx context.Context, title, markdown, previous string) (Decision, error)
}

type State struct {
	Phase    Phase           `json:"phase"`
	PlanPath string          `json:"planPath,omitempty"`
	Items    []ChecklistItem `json:"items,omitempty"`
	// PlanHash is the plan file's content hash when Items was derived from it.
	//
	// Checklist steps are positional -- ParseChecklist numbers them by their
	// order in the file -- and completion is merged back by that integer. Edit
	// PLAN.md between two runs and the merge silently moves completion flags
	// onto different tasks. On a mismatch, completion is dropped and the user
	// is told, which is recoverable; misattributed completion is not.
	PlanHash string `json:"planHash,omitempty"`
}

// Options configures a Manager. Initial seeds the phase from wherever the host
// keeps it; OnChange is called whenever the phase or checklist moves, so the
// host can persist it.
type Options struct {
	Initial  *State
	OnChange func(State)
	// Warn reports a recoverable problem with the restored state.
	Warn func(string)
}

type Manager struct {
	mu               sync.Mutex
	root             string
	state            State
	reviewer         Reviewer
	onChange         func(State)
	warn             func(string)
	lastReviewedPlan string
}

// New builds a Manager.
//
// The Manager no longer owns a file. Plan state used to live in
// ~/.goshcoder/agent/plannotator/<sha256-of-workspace>.json, which meant two
// sessions in one workspace shared one phase and the last writer won: window B
// pressing /planner turned window A's mode off. A capability restriction on an
// agent is a property of that agent, not of a directory -- one window planning
// a refactor while another fixes an unrelated test is a normal workflow the old
// scoping forbade. Session persistence is what made per-session state possible;
// before it there was nowhere durable to put it.
//
// A restored state that cannot be parsed or no longer matches its plan file
// yields idle plus a warning. It must never be an error: once phase lives in
// the session log, a corrupt payload would otherwise be a reason the whole
// session refuses to open.
func New(root string, reviewer Reviewer, options Options) (*Manager, error) {
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	m := &Manager{
		root: absolute, reviewer: reviewer, state: State{Phase: PhaseIdle},
		onChange: options.OnChange, warn: options.Warn,
	}
	if options.Initial != nil {
		m.state = *options.Initial
	}
	switch m.state.Phase {
	case PhaseIdle, PhasePlanning, PhaseExecuting:
	default:
		m.warnf("ignoring an unrecognized saved Planner phase %q; starting idle", m.state.Phase)
		m.state = State{Phase: PhaseIdle}
	}
	if m.state.Phase == PhaseExecuting && m.state.PlanPath != "" {
		m.rehydrateChecklistLocked()
	}
	return m, nil
}

func (m *Manager) warnf(format string, args ...any) {
	if m.warn != nil {
		m.warn(fmt.Sprintf(format, args...))
	}
}

// rehydrateChecklistLocked re-reads the plan file and merges saved completion
// into it, refusing the merge when the file changed underneath.
func (m *Manager) rehydrateChecklistLocked() {
	content, err := m.readPlan(m.state.PlanPath)
	if err != nil {
		m.warnf("the plan %s can no longer be read (%v); Planner is starting idle", m.state.PlanPath, err)
		m.state = State{Phase: PhaseIdle}
		return
	}
	hash := hashPlan(content)
	fresh := ParseChecklist(string(content))
	if m.state.PlanHash != "" && m.state.PlanHash != hash {
		// Steps are positional, so merging across an edit moves completion onto
		// whatever task now occupies that position.
		m.warnf("%s changed since this plan was approved; its checklist progress was reset rather than guessed at", m.state.PlanPath)
		m.state.Items = fresh
		m.state.PlanHash = hash
		return
	}
	completed := map[int]bool{}
	for _, item := range m.state.Items {
		completed[item.Step] = item.Completed
	}
	for index := range fresh {
		fresh[index].Completed = fresh[index].Completed || completed[fresh[index].Step]
	}
	m.state.Items = fresh
	m.state.PlanHash = hash
}

// hashPlan identifies a plan file's content.
func hashPlan(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// publishLocked hands the current state to the host for persistence.
func (m *Manager) publishLocked() {
	if m.onChange != nil {
		state := m.state
		state.Items = append([]ChecklistItem(nil), m.state.Items...)
		m.onChange(state)
	}
}

func (m *Manager) State() State {
	m.mu.Lock()
	defer m.mu.Unlock()
	state := m.state
	state.Items = append([]ChecklistItem(nil), m.state.Items...)
	return state
}

// Enter, Exit and Toggle change the phase and nothing else.
//
// They used to assign a fresh State{} literal, which threw away PlanPath and
// Items: one /planner keypress wiped the checklist of an approved plan, and
// there was no way to get it back. Only Submit replaces the plan itself.

func (m *Manager) Enter() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state.Phase = PhasePlanning
	m.publishLocked()
	return nil
}

func (m *Manager) Exit() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state.Phase = PhaseIdle
	m.publishLocked()
	return nil
}

func (m *Manager) Toggle() (Phase, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.state.Phase == PhaseIdle {
		m.state.Phase = PhasePlanning
	} else {
		m.state.Phase = PhaseIdle
	}
	m.publishLocked()
	return m.state.Phase, nil
}

func (m *Manager) Prompt(base string) string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return base + m.promptSuffixLocked()
}

func (m *Manager) promptSuffixLocked() string {
	switch m.state.Phase {
	case PhasePlanning:
		return planningPrompt
	case PhaseExecuting:
		remaining := make([]string, 0)
		for _, item := range m.state.Items {
			if !item.Completed {
				remaining = append(remaining, fmt.Sprintf("- [ ] %d. %s", item.Step, item.Text))
			}
		}
		if len(remaining) == 0 {
			return ""
		}
		return fmt.Sprintf("\n\n[PLANNER - EXECUTING PLAN]\nFull tool access is enabled. Execute the approved plan from %s.\n\nRemaining steps:\n%s\n\nExecute each step in order. After completing a step, include [DONE:n] in your response where n is the step number.", m.state.PlanPath, strings.Join(remaining, "\n"))
	default:
		return ""
	}
}

func (m *Manager) BeforeToolCall(_ context.Context, call agent.BeforeToolCallContext) *agent.BeforeToolCallResult {
	m.mu.Lock()
	planning := m.state.Phase == PhasePlanning
	m.mu.Unlock()
	if !planning {
		return nil
	}
	if call.ToolCall.Name == "bash" {
		return &agent.BeforeToolCallResult{Block: true, Reason: "Planner: shell commands are disabled during planning because they can modify the workspace."}
	}
	if call.ToolCall.Name != "write" && call.ToolCall.Name != "edit" {
		return nil
	}
	path, _ := call.Args["path"].(string)
	if !m.IsPlanPathAllowed(path) {
		return &agent.BeforeToolCallResult{Block: true, Reason: fmt.Sprintf("Planner: during planning, writes and edits are limited to markdown files inside the workspace. Blocked: %s", path)}
	}
	return nil
}

func (m *Manager) PrepareNextTurn(base string) agent.PrepareNextTurnFunc {
	return func(_ context.Context, turn agent.PrepareNextTurnContext) *agent.TurnUpdate {
		m.TrackAssistant(turn.Message)
		updated := turn.Context
		updated.SystemPrompt = m.Prompt(base)
		return &agent.TurnUpdate{Context: &updated}
	}
}

func (m *Manager) Tool() agent.Tool {
	return agent.Tool{
		Name: SubmitToolName, Label: "Submit Plan",
		Description:   "Submit a markdown plan for human review. Use only in Planner mode after writing the plan inside the workspace. If denied, revise the same file and resubmit.",
		Parameters:    json.RawMessage(`{"type":"object","properties":{"filePath":{"type":"string","description":"Markdown plan path relative to the workspace"}},"required":["filePath"]}`),
		ExecutionMode: agent.ToolExecutionSequential,
		Execute: func(ctx context.Context, _ string, params map[string]any, _ func(agent.ToolResult)) (agent.ToolResult, error) {
			path, _ := params["filePath"].(string)
			return m.Submit(ctx, path)
		},
	}
}

func (m *Manager) Submit(ctx context.Context, inputPath string) (agent.ToolResult, error) {
	m.mu.Lock()
	if m.state.Phase != PhasePlanning {
		m.mu.Unlock()
		return textResult("Error: Not in Planner mode."), nil
	}
	m.mu.Unlock()

	if !m.IsPlanPathAllowed(inputPath) {
		return textResult("Error: plan file must be a markdown file (.md or .mdx) inside the workspace."), nil
	}
	content, err := m.readPlan(inputPath)
	if err != nil {
		return textResult(fmt.Sprintf("Error: %s cannot be read as a regular plan file: %v", inputPath, err)), nil
	}
	if strings.TrimSpace(string(content)) == "" {
		return textResult("Error: the plan file is empty."), nil
	}
	items := ParseChecklist(string(content))
	if len(items) == 0 {
		return textResult("Error: the plan must contain at least one markdown checklist item using '- [ ]'."), nil
	}

	decision := Decision{Approved: true}
	if m.reviewer != nil {
		m.mu.Lock()
		previous := m.lastReviewedPlan
		m.mu.Unlock()
		if reviewer, ok := m.reviewer.(versionedReviewer); ok {
			decision, err = reviewer.ReviewVersion(ctx, "Review plan: "+filepath.Base(inputPath), string(content), previous)
		} else {
			decision, err = m.reviewer.Review(ctx, "Review plan: "+filepath.Base(inputPath), string(content))
		}
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return textResult("Plan review was cancelled. The plan was not approved; resubmit to review again."), nil
			}
			return agent.ToolResult{}, err
		}
		m.mu.Lock()
		m.lastReviewedPlan = string(content)
		m.mu.Unlock()
	}
	if !decision.Approved {
		feedback := strings.TrimSpace(decision.Feedback)
		if feedback == "" {
			feedback = "Plan rejected. Please revise it."
		}
		return textResult("The plan was denied. Edit the same plan file with targeted changes, then resubmit it.\n\nUser feedback:\n" + feedback), nil
	}

	m.mu.Lock()
	m.state = State{
		Phase: PhaseExecuting, PlanPath: filepath.ToSlash(inputPath),
		Items: items, PlanHash: hashPlan(content),
	}
	m.publishLocked()
	m.mu.Unlock()
	message := "Plan approved. Begin implementation now using full tool access."
	if feedback := strings.TrimSpace(decision.Feedback); feedback != "" {
		message += "\n\nImplementation notes from the reviewer:\n" + feedback
	}
	message += "\nAfter completing each checklist step, include [DONE:n] in your response."
	result := textResult(message)
	result.Details = map[string]any{"approved": true}
	return result, nil
}

func (m *Manager) IsPlanPathAllowed(inputPath string) bool {
	if inputPath == "" {
		return false
	}
	target := inputPath
	if !filepath.IsAbs(target) {
		target = filepath.Join(m.root, target)
	}
	target = filepath.Clean(target)
	relative, err := filepath.Rel(m.root, target)
	if err != nil || relative == "." || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return false
	}
	ext := strings.ToLower(filepath.Ext(target))
	return ext == ".md" || ext == ".mdx"
}

var checklistPattern = regexp.MustCompile(`(?m)^(?:[-*+]|\d+[.)])\s*\[([ xX])\]\s+(.+)$`)
var donePattern = regexp.MustCompile(`(?i)\[DONE:(\d+)\]`)

func ParseChecklist(content string) []ChecklistItem {
	matches := checklistPattern.FindAllStringSubmatch(content, -1)
	items := make([]ChecklistItem, 0, len(matches))
	for _, match := range matches {
		text := strings.TrimSpace(match[2])
		if text != "" {
			items = append(items, ChecklistItem{Step: len(items) + 1, Text: text, Completed: match[1] != " "})
		}
	}
	return items
}

func (m *Manager) TrackAssistant(message llm.AssistantMessage) int {
	var text strings.Builder
	for _, block := range message.Content {
		if content, ok := block.(llm.TextContent); ok {
			text.WriteString(content.Text)
		}
	}
	matches := donePattern.FindAllStringSubmatch(text.String(), -1)
	if len(matches) == 0 {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	changed := 0
	for _, match := range matches {
		step, _ := strconv.Atoi(match[1])
		for index := range m.state.Items {
			if m.state.Items[index].Step == step && !m.state.Items[index].Completed {
				m.state.Items[index].Completed = true
				changed++
			}
		}
	}
	finished := false
	if m.state.Phase == PhaseExecuting && len(m.state.Items) > 0 {
		complete := true
		for _, item := range m.state.Items {
			complete = complete && item.Completed
		}
		if complete {
			// Completing the last step returns to idle but keeps the plan and
			// its checklist. This was the fourth site assigning a bare
			// State{} literal, and the easiest to miss: fixing the other three
			// leaves finishing a plan still wiping the record of it.
			m.state.Phase = PhaseIdle
			finished = true
		}
	}
	// Publishing is not gated on `changed`: the phase transition above happens
	// on the same call that marks the final step done, and gating on the
	// marker count meant a plan could finish without the change being saved.
	if changed > 0 || finished {
		m.publishLocked()
	}
	return changed
}

func (m *Manager) StatusLine() string {
	state := m.State()
	switch state.Phase {
	case PhasePlanning:
		return "Planner: planning"
	case PhaseExecuting:
		completed := 0
		for _, item := range state.Items {
			if item.Completed {
				completed++
			}
		}
		return fmt.Sprintf("Planner: executing %d/%d", completed, len(state.Items))
	default:
		return "Planner: idle"
	}
}

func (m *Manager) readPlan(inputPath string) ([]byte, error) {
	name := filepath.Clean(filepath.FromSlash(inputPath))
	if filepath.IsAbs(name) {
		var err error
		name, err = filepath.Rel(m.root, name)
		if err != nil {
			return nil, err
		}
	}
	root, err := os.OpenRoot(m.root)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	info, err := root.Stat(name)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, errors.New("not a regular file")
	}
	file, err := root.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, maxPlanBytes+1))
	if err != nil {
		return nil, err
	}
	if len(content) > maxPlanBytes {
		return nil, fmt.Errorf("plan exceeds %d bytes", maxPlanBytes)
	}
	return content, nil
}

func textResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}

const planningPrompt = `

[PLANNER - PLANNING PHASE]
You are in plan mode. Do not modify the codebase, commit, install dependencies, or run destructive commands. You may only write or edit markdown plan files (.md or .mdx) inside the workspace.

Explore the codebase with read-only tools. Build a concise plan containing Context, Approach, Files to modify, Reuse, implementation checklist items using "- [ ]", and Verification. Ask the user only about ambiguities that cannot be answered from the code. When ready, call planner_submit_plan with the plan file path. If review denies it, make targeted edits to the same file and resubmit.`
