// Package plannotator provides a native Go adaptation of
// @plannotator/pi-extension. It preserves the extension's planning state
// machine, write gate, checklist tracking, and human review workflow without a
// JavaScript extension host.
package plannotator

import (
	"context"
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
}

type Manager struct {
	mu               sync.Mutex
	root             string
	state            State
	reviewer         Reviewer
	stateFile        string
	lastReviewedPlan string
}

func New(root, stateFile string, reviewer Reviewer) (*Manager, error) {
	absolute, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	m := &Manager{root: absolute, reviewer: reviewer, stateFile: stateFile, state: State{Phase: PhaseIdle}}
	if err := m.load(); err != nil {
		return nil, err
	}
	switch m.state.Phase {
	case PhaseIdle, PhasePlanning, PhaseExecuting:
	default:
		return nil, fmt.Errorf("invalid persisted Planner phase %q", m.state.Phase)
	}
	if m.state.Phase == PhaseExecuting && m.state.PlanPath != "" {
		if content, err := m.readPlan(m.state.PlanPath); err == nil {
			fresh := ParseChecklist(string(content))
			completed := map[int]bool{}
			for _, item := range m.state.Items {
				completed[item.Step] = item.Completed
			}
			for index := range fresh {
				fresh[index].Completed = fresh[index].Completed || completed[fresh[index].Step]
			}
			m.state.Items = fresh
		} else {
			m.state = State{Phase: PhaseIdle}
		}
	}
	return m, nil
}

func (m *Manager) State() State {
	m.mu.Lock()
	defer m.mu.Unlock()
	state := m.state
	state.Items = append([]ChecklistItem(nil), m.state.Items...)
	return state
}

func (m *Manager) Enter() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = State{Phase: PhasePlanning}
	return m.saveLocked()
}

func (m *Manager) Exit() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = State{Phase: PhaseIdle}
	return m.saveLocked()
}

func (m *Manager) Toggle() (Phase, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.state.Phase == PhaseIdle {
		m.state = State{Phase: PhasePlanning}
	} else {
		m.state = State{Phase: PhaseIdle}
	}
	return m.state.Phase, m.saveLocked()
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
	previous := m.state
	m.state = State{Phase: PhaseExecuting, PlanPath: filepath.ToSlash(inputPath), Items: items}
	if err := m.saveLocked(); err != nil {
		m.state = previous
		m.mu.Unlock()
		return agent.ToolResult{}, err
	}
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
	if m.state.Phase == PhaseExecuting && len(m.state.Items) > 0 {
		complete := true
		for _, item := range m.state.Items {
			complete = complete && item.Completed
		}
		if complete {
			m.state = State{Phase: PhaseIdle}
		}
	}
	if changed > 0 {
		_ = m.saveLocked()
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

func (m *Manager) saveLocked() error {
	if m.stateFile == "" {
		return nil
	}
	directory := filepath.Dir(m.stateFile)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(m.state, "", "  ")
	if err != nil {
		return err
	}
	temporary, err := os.CreateTemp(directory, ".plannotator-*.tmp")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	defer os.Remove(temporaryName)
	if err := temporary.Chmod(0o600); err != nil {
		temporary.Close()
		return err
	}
	if _, err := temporary.Write(append(encoded, '\n')); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryName, m.stateFile)
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

func (m *Manager) load() error {
	if m.stateFile == "" {
		return nil
	}
	file, err := os.Open(m.stateFile)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	defer file.Close()
	encoded, err := io.ReadAll(io.LimitReader(file, 1<<20))
	if err != nil {
		return err
	}
	if len(encoded) == 1<<20 {
		var extra [1]byte
		if count, _ := file.Read(extra[:]); count > 0 {
			return errors.New("plannotator state exceeds 1 MiB")
		}
	}
	return json.Unmarshal(encoded, &m.state)
}

func textResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}

const planningPrompt = `

[PLANNER - PLANNING PHASE]
You are in plan mode. Do not modify the codebase, commit, install dependencies, or run destructive commands. You may only write or edit markdown plan files (.md or .mdx) inside the workspace.

Explore the codebase with read-only tools. Build a concise plan containing Context, Approach, Files to modify, Reuse, implementation checklist items using "- [ ]", and Verification. Ask the user only about ambiguities that cannot be answered from the code. When ready, call planner_submit_plan with the plan file path. If review denies it, make targeted edits to the same file and resubmit.`
