// Package ralph provides long-running iterative development loops.
//
// This is a native Go port of the @tmustier/pi-ralph-wiggum pi extension
// (Geoffrey Huntley's "ralph" approach), found installed in the local pi
// configuration. Where pi implements it as a TypeScript extension against its
// ExtensionAPI, GoshCoder ships it built in: the loop state machine lives here
// and is exposed as two agent tools plus a system-prompt contribution.
//
// The on-disk format is deliberately identical to the extension's, so a
// .ralph directory can be driven by either tool:
//
//	.ralph/<name>.md            the task file the agent maintains
//	.ralph/<name>.state.json    loop state (iteration, limits, status)
//	.ralph/archive/             archived loops
package ralph

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
)

// Dir is the per-workspace loop directory.
const Dir = ".ralph"

// CompleteMarker is what the agent emits when the task is genuinely finished.
// It is deliberately verbatim from the extension so prompts and task files
// carry over.
const CompleteMarker = "<promise>COMPLETE</promise>"

// DefaultMaxIterations bounds a loop that does not specify its own limit.
const DefaultMaxIterations = 50

// Status is the loop lifecycle state.
type Status = string

const (
	StatusActive    Status = "active"
	StatusPaused    Status = "paused"
	StatusCompleted Status = "completed"
)

// statusIcons mirror the extension's status rail glyphs.
var statusIcons = map[Status]string{
	StatusActive:    "▶",
	StatusPaused:    "⏸",
	StatusCompleted: "✓",
}

// State is one loop's persisted state. JSON field names match the pi
// extension's LoopState exactly.
type State struct {
	Name          string `json:"name"`
	TaskFile      string `json:"taskFile"`
	Iteration     int    `json:"iteration"`
	MaxIterations int    `json:"maxIterations"`
	// ItemsPerIteration is a prompt hint only ("process ~N items per turn").
	ItemsPerIteration int `json:"itemsPerIteration"`
	// ReflectEvery triggers a reflection checkpoint every N iterations.
	ReflectEvery        int    `json:"reflectEvery"`
	ReflectInstructions string `json:"reflectInstructions"`
	// Active is retained for backwards compatibility with older state files;
	// Status is authoritative.
	Active      bool   `json:"active"`
	Status      Status `json:"status"`
	StartedAt   string `json:"startedAt"`
	CompletedAt string `json:"completedAt,omitempty"`
	// LastReflectionAt is the iteration a reflection last happened at.
	LastReflectionAt int `json:"lastReflectionAt"`
	// OwnerSessionID scopes automatic prompt injection to one session.
	OwnerSessionID string `json:"ownerSessionId,omitempty"`
}

// migrate normalizes a state file read from disk, applying the same field
// migrations the extension does.
func (s *State) migrate() {
	if s.Status == "" {
		s.Status = StatusPaused
		if s.Active {
			s.Status = StatusActive
		}
	}
	s.Active = s.Status == StatusActive
	if s.ReflectInstructions == "" {
		s.ReflectInstructions = DefaultReflectInstructions
	}
}

// Summary renders a one-line status description.
func (s *State) Summary() string {
	iteration := fmt.Sprint(s.Iteration)
	if s.MaxIterations > 0 {
		iteration = fmt.Sprintf("%d/%d", s.Iteration, s.MaxIterations)
	}
	return fmt.Sprintf("%s: %s %s (iteration %s)", s.Name, statusIcons[s.Status], s.Status, iteration)
}

var sanitizeRe = regexp.MustCompile(`[^a-zA-Z0-9_-]`)
var collapseRe = regexp.MustCompile(`_+`)

// SanitizeName reduces a loop name to a safe filename stem.
func SanitizeName(name string) string {
	return collapseRe.ReplaceAllString(sanitizeRe.ReplaceAllString(name, "_"), "_")
}

// Store persists loop state under a workspace root.
type Store struct {
	// Root is the workspace directory containing .ralph.
	Root string
	// SessionID scopes loop ownership. Loops started by another session are
	// visible but not driven automatically.
	SessionID string

	mu sync.Mutex
	// current is the loop this session is driving.
	current string
}

// NewStore returns a store rooted at the given workspace directory.
func NewStore(root, sessionID string) *Store {
	return &Store{Root: root, SessionID: sessionID}
}

func (s *Store) dir(archived bool) string {
	if archived {
		return filepath.Join(s.Root, Dir, "archive")
	}
	return filepath.Join(s.Root, Dir)
}

func (s *Store) path(name, ext string, archived bool) string {
	return filepath.Join(s.dir(archived), SanitizeName(name)+ext)
}

// TaskFile returns the workspace-relative task file path for a loop.
func (s *Store) TaskFile(name string) string {
	return filepath.ToSlash(filepath.Join(Dir, SanitizeName(name)+".md"))
}

// Load reads a loop's state. ok is false when the loop does not exist.
func (s *Store) Load(name string, archived bool) (*State, bool) {
	content, err := os.ReadFile(s.path(name, ".state.json", archived))
	if err != nil {
		return nil, false
	}
	var state State
	if err := json.Unmarshal(content, &state); err != nil {
		return nil, false
	}
	state.migrate()
	return &state, true
}

// Save writes a loop's state.
func (s *Store) Save(state *State, archived bool) error {
	state.Active = state.Status == StatusActive
	path := s.path(state.Name, ".state.json", archived)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	encoded, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(encoded, '\n'), 0o644)
}

// List returns every loop, sorted by name.
func (s *Store) List(archived bool) []*State {
	entries, err := os.ReadDir(s.dir(archived))
	if err != nil {
		return nil
	}
	var states []*State
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".state.json") {
			continue
		}
		name := strings.TrimSuffix(entry.Name(), ".state.json")
		if state, ok := s.Load(name, archived); ok {
			states = append(states, state)
		}
	}
	sort.Slice(states, func(i, j int) bool { return states[i].Name < states[j].Name })
	return states
}

// ReadTask reads a loop's task file.
func (s *Store) ReadTask(state *State) (string, error) {
	content, err := os.ReadFile(filepath.Join(s.Root, filepath.FromSlash(state.TaskFile)))
	if err != nil {
		return "", err
	}
	return string(content), nil
}

// WriteTask writes a loop's task file, creating .ralph as needed.
func (s *Store) WriteTask(state *State, content string) error {
	path := filepath.Join(s.Root, filepath.FromSlash(state.TaskFile))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

// ownedByThisSession reports whether this session drives the loop. A loop with
// no owner is adoptable, matching the extension's behavior for state written
// before sessions were tracked.
func (s *Store) ownedByThisSession(state *State) bool {
	return state.OwnerSessionID == "" || state.OwnerSessionID == s.SessionID
}

// Current returns the active loop this session drives, if any.
func (s *Store) Current() (*State, bool) {
	s.mu.Lock()
	name := s.current
	s.mu.Unlock()

	if name != "" {
		if state, ok := s.Load(name, false); ok && s.ownedByThisSession(state) {
			return state, true
		}
		s.mu.Lock()
		s.current = ""
		s.mu.Unlock()
	}

	// Fall back to any active loop this session may adopt, so a restarted CLI
	// can resume where it left off.
	for _, state := range s.List(false) {
		if state.Status == StatusActive && s.ownedByThisSession(state) {
			s.mu.Lock()
			s.current = state.Name
			s.mu.Unlock()
			return state, true
		}
	}
	return nil, false
}

func (s *Store) setCurrent(name string) {
	s.mu.Lock()
	s.current = name
	s.mu.Unlock()
}

// Start creates and activates a loop, writing its task file.
func (s *Store) Start(name, taskContent string, options Options) (*State, error) {
	loopName := SanitizeName(name)
	if loopName == "" {
		return nil, fmt.Errorf("a loop name is required")
	}
	if existing, ok := s.Load(loopName, false); ok && existing.Status == StatusActive {
		return nil, fmt.Errorf("loop %q is already active", loopName)
	}

	maxIterations := options.MaxIterations
	if maxIterations == 0 {
		maxIterations = DefaultMaxIterations
	}
	state := &State{
		Name:                loopName,
		TaskFile:            s.TaskFile(loopName),
		Iteration:           1,
		MaxIterations:       maxIterations,
		ItemsPerIteration:   options.ItemsPerIteration,
		ReflectEvery:        options.ReflectEvery,
		ReflectInstructions: DefaultReflectInstructions,
		Status:              StatusActive,
		StartedAt:           time.Now().UTC().Format(time.RFC3339),
		OwnerSessionID:      s.SessionID,
	}
	if taskContent == "" {
		taskContent = DefaultTemplate
	}
	if err := s.WriteTask(state, taskContent); err != nil {
		return nil, err
	}
	if err := s.Save(state, false); err != nil {
		return nil, err
	}
	s.setCurrent(loopName)
	return state, nil
}

// Options configure a new loop.
type Options struct {
	// MaxIterations bounds the loop; 0 uses DefaultMaxIterations. A negative
	// value means unbounded.
	MaxIterations int
	// ItemsPerIteration suggests how many checklist items to process per turn.
	ItemsPerIteration int
	// ReflectEvery triggers a reflection checkpoint every N iterations.
	ReflectEvery int
}

// Advance moves the loop to its next iteration. done reports that the loop
// finished (max iterations reached); reflect reports a reflection checkpoint.
func (s *Store) Advance(state *State) (done bool, reflect bool, err error) {
	state.Iteration++

	if state.MaxIterations > 0 && state.Iteration > state.MaxIterations {
		if err := s.Complete(state); err != nil {
			return true, false, err
		}
		return true, false, nil
	}

	reflect = state.ReflectEvery > 0 && (state.Iteration-1)%state.ReflectEvery == 0
	if reflect {
		state.LastReflectionAt = state.Iteration
	}
	if err := s.Save(state, false); err != nil {
		return false, reflect, err
	}
	return false, reflect, nil
}

// Complete marks a loop finished.
func (s *Store) Complete(state *State) error {
	state.Status = StatusCompleted
	state.CompletedAt = time.Now().UTC().Format(time.RFC3339)
	s.setCurrent("")
	return s.Save(state, false)
}

// Pause suspends a loop, leaving it resumable.
func (s *Store) Pause(state *State) error {
	state.Status = StatusPaused
	s.setCurrent("")
	return s.Save(state, false)
}

// Resume reactivates a paused loop under this session.
func (s *Store) Resume(name string) (*State, error) {
	state, ok := s.Load(name, false)
	if !ok {
		return nil, fmt.Errorf("no loop named %q", name)
	}
	if state.Status == StatusCompleted {
		return nil, fmt.Errorf("loop %q is already completed", name)
	}
	state.Status = StatusActive
	state.OwnerSessionID = s.SessionID
	if err := s.Save(state, false); err != nil {
		return nil, err
	}
	s.setCurrent(state.Name)
	return state, nil
}

// Archive moves a loop's state and task file into .ralph/archive.
func (s *Store) Archive(name string) error {
	state, ok := s.Load(name, false)
	if !ok {
		return fmt.Errorf("no loop named %q", name)
	}
	if err := os.MkdirAll(s.dir(true), 0o755); err != nil {
		return err
	}
	if err := s.Save(state, true); err != nil {
		return err
	}
	if err := os.Remove(s.path(name, ".state.json", false)); err != nil && !os.IsNotExist(err) {
		return err
	}
	if state.Name == s.currentName() {
		s.setCurrent("")
	}
	return nil
}

func (s *Store) currentName() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

// Delete removes a loop's state and task file.
func (s *Store) Delete(name string) error {
	if _, ok := s.Load(name, false); !ok {
		return fmt.Errorf("no loop named %q", name)
	}
	for _, path := range []string{
		s.path(name, ".state.json", false),
		filepath.Join(s.Root, Dir, SanitizeName(name)+".md"),
	} {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if SanitizeName(name) == s.currentName() {
		s.setCurrent("")
	}
	return nil
}

// StatusLine renders the loop indicator for the terminal status rail.
func (s *Store) StatusLine() string {
	state, ok := s.Current()
	if !ok {
		return ""
	}
	maxPart := ""
	if state.MaxIterations > 0 {
		maxPart = fmt.Sprintf("/%d", state.MaxIterations)
	}
	reflection := ""
	if state.ReflectEvery > 0 {
		remaining := state.ReflectEvery - ((state.Iteration - 1) % state.ReflectEvery)
		reflection = fmt.Sprintf(" · reflect in %d", remaining)
	}
	return fmt.Sprintf("ralph · %s · %s %s · %d%s · %s%s",
		state.Name, statusIcons[state.Status], state.Status,
		state.Iteration, maxPart, state.TaskFile, reflection)
}

// ---------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------

// BuildPrompt renders the per-iteration prompt handed back to the agent.
func BuildPrompt(state *State, taskContent string, isReflection bool) string {
	maxPart := ""
	if state.MaxIterations > 0 {
		maxPart = fmt.Sprintf("/%d", state.MaxIterations)
	}
	reflectionTag := ""
	if isReflection {
		reflectionTag = " | REFLECTION"
	}

	var b strings.Builder
	rule := strings.Repeat("─", 71)
	fmt.Fprintf(&b, "%s\nRALPH LOOP: %s | Iteration %d%s%s\n%s\n\n",
		rule, state.Name, state.Iteration, maxPart, reflectionTag, rule)

	if isReflection {
		instructions := state.ReflectInstructions
		if instructions == "" {
			instructions = DefaultReflectInstructions
		}
		b.WriteString(instructions + "\n\n---\n\n")
	}

	fmt.Fprintf(&b, "## Current Task (from %s)\n\n%s\n\n---\n", state.TaskFile, taskContent)
	fmt.Fprintf(&b, "\n## Stale Prompt Guard\n\n%s\n", DefaultStalePromptGuard)
	fmt.Fprintf(&b, "\n## Completion Gate\n\n%s\n", DefaultCompletionGate)
	b.WriteString("\n## Instructions\n\n")
	fmt.Fprintf(&b, "You are in a Ralph loop (iteration %d%s).\n\n", state.Iteration, maxPart)

	step := 1
	if state.ItemsPerIteration > 0 {
		fmt.Fprintf(&b, "**THIS ITERATION: process approximately %d items, then call ralph_done.**\n\n", state.ItemsPerIteration)
		fmt.Fprintf(&b, "%d. Work on the next ~%d items from your checklist\n", step, state.ItemsPerIteration)
	} else {
		fmt.Fprintf(&b, "%d. Continue working on the task\n", step)
	}
	step++
	fmt.Fprintf(&b, "%d. Update the task file (%s) with your progress\n", step, state.TaskFile)
	step++
	fmt.Fprintf(&b, "%d. When FULLY COMPLETE and the completion gate is satisfied, respond with: %s\n", step, CompleteMarker)
	step++
	fmt.Fprintf(&b, "%d. Otherwise, call the ralph_done tool to proceed to the next iteration\n", step)
	return b.String()
}

// SystemPromptSuffix is appended to the system prompt while a loop is active,
// mirroring the extension's before_agent_start hook.
func SystemPromptSuffix(state *State) string {
	iteration := fmt.Sprint(state.Iteration)
	if state.MaxIterations > 0 {
		iteration = fmt.Sprintf("%d/%d", state.Iteration, state.MaxIterations)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "\n[RALPH LOOP - %s - Iteration %s]\n\n", state.Name, iteration)
	fmt.Fprintf(&b, "You are in a Ralph loop working on: %s\n", state.TaskFile)
	fmt.Fprintf(&b, "- Before doing work, reload %s/%s.state.json; if status is completed, ignore the stale prompt and do not call ralph_done\n", Dir, state.Name)
	if state.ItemsPerIteration > 0 {
		fmt.Fprintf(&b, "- Work on ~%d items this iteration\n", state.ItemsPerIteration)
	}
	b.WriteString("- Update the task file as you progress\n")
	b.WriteString("- Preserve artifacts needed by final verification\n")
	b.WriteString("- Record an exact, externally rerunnable final command before completion\n")
	fmt.Fprintf(&b, "- When FULLY COMPLETE and externally rerunnable: %s\n", CompleteMarker)
	b.WriteString("- Otherwise, call the ralph_done tool to proceed to the next iteration")
	return b.String()
}

// HasCompleteMarker reports whether an assistant message claims completion.
func HasCompleteMarker(message llm.AssistantMessage) bool {
	for _, block := range message.Content {
		if text, ok := block.(llm.TextContent); ok && strings.Contains(text.Text, CompleteMarker) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

// Queue receives the prompts the loop generates. The CLI wires this to the
// agent's follow-up queue, which is how pi's `deliverAs: "followUp"` behaves:
// the message is processed after the current turn and stays interruptible.
type Queue interface {
	FollowUp(message agent.Message)
	HasQueuedMessages() bool
}

func textResult(text string) agent.ToolResult {
	return agent.ToolResult{Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: text}}}
}

func userMessage(text string) llm.UserMessage {
	return llm.UserMessage{Role: "user", Content: text, Timestamp: time.Now().UnixMilli()}
}

// Tools returns the ralph_start and ralph_done agent tools.
func (s *Store) Tools(queue Queue) []agent.Tool {
	return []agent.Tool{s.StartTool(queue), s.DoneTool(queue)}
}

// StartTool exposes loop creation to the model.
func (s *Store) StartTool(queue Queue) agent.Tool {
	return agent.Tool{
		Name:  "ralph_start",
		Label: "Start Ralph Loop",
		Description: "Start a long-running development loop with pacing and reflection controls. " +
			"Use when the task needs multiple iterations or paced multi-step execution; " +
			"avoid it for one-shot fixes. After starting, continue each finished iteration " +
			"with ralph_done unless the completion marker has been emitted.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"name": {"type": "string", "description": "Loop name, e.g. refactor-auth"},
				"taskContent": {"type": "string", "description": "Task in markdown with goals and a checklist"},
				"maxIterations": {"type": "integer", "description": "Maximum iterations (default 50)"},
				"itemsPerIteration": {"type": "integer", "description": "Suggest N checklist items per turn (0 = no limit)"},
				"reflectEvery": {"type": "integer", "description": "Reflect every N iterations (0 = never)"}
			},
			"required": ["name", "taskContent"]
		}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			name, _ := params["name"].(string)
			taskContent, _ := params["taskContent"].(string)

			state, err := s.Start(name, taskContent, Options{
				MaxIterations:     intParam(params, "maxIterations"),
				ItemsPerIteration: intParam(params, "itemsPerIteration"),
				ReflectEvery:      intParam(params, "reflectEvery"),
			})
			if err != nil {
				// An already-active loop is reported as a result, not an error:
				// it is a no-op the model can react to.
				if strings.Contains(err.Error(), "already active") {
					return textResult(err.Error()), nil
				}
				return agent.ToolResult{}, err
			}

			if queue != nil {
				queue.FollowUp(userMessage(BuildPrompt(state, taskContent, false)))
			}
			return textResult(fmt.Sprintf("Started loop %q (max %d iterations). Task file: %s",
				state.Name, state.MaxIterations, state.TaskFile)), nil
		},
	}
}

// DoneTool advances the active loop to its next iteration.
func (s *Store) DoneTool(queue Queue) agent.Tool {
	return agent.Tool{
		Name:  "ralph_done",
		Label: "Ralph Iteration Done",
		Description: "Signal that the current Ralph loop iteration is complete and request the next one. " +
			"Call this after making real progress. Do not call it when there is no active loop, " +
			"when messages are already queued, or after emitting the completion marker.",
		Parameters: json.RawMessage(`{"type": "object", "properties": {}}`),
		Execute: func(ctx context.Context, id string, params map[string]any, onUpdate func(agent.ToolResult)) (agent.ToolResult, error) {
			state, ok := s.Current()
			if !ok || state.Status != StatusActive {
				return textResult("No active Ralph loop owned by this session."), nil
			}
			// Queued work already drives the next turn, so advancing here would
			// double-inject a prompt.
			if queue != nil && queue.HasQueuedMessages() {
				return textResult("Messages are already queued; skipping ralph_done."), nil
			}

			previous := state.Iteration
			done, reflect, err := s.Advance(state)
			if err != nil {
				return agent.ToolResult{}, err
			}
			if done {
				return agent.ToolResult{
					Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: fmt.Sprintf(
						"Max iterations (%d) reached; the loop has stopped.", state.MaxIterations)}},
					// Nothing further should run in this batch.
					Terminate: true,
				}, nil
			}

			taskContent, err := s.ReadTask(state)
			if err != nil {
				// An unreadable task file cannot drive another iteration.
				if pauseErr := s.Pause(state); pauseErr != nil {
					return agent.ToolResult{}, pauseErr
				}
				return agent.ToolResult{}, fmt.Errorf("cannot read task file %s: %w", state.TaskFile, err)
			}

			if queue != nil {
				queue.FollowUp(userMessage(BuildPrompt(state, taskContent, reflect)))
			}
			suffix := ""
			if reflect {
				suffix = " (reflection checkpoint)"
			}
			return textResult(fmt.Sprintf("Iteration %d complete; iteration %d queued%s.",
				previous, state.Iteration, suffix)), nil
		},
	}
}

// intParam reads an integer tool parameter, tolerating the float64 that JSON
// decoding produces.
func intParam(params map[string]any, key string) int {
	switch value := params[key].(type) {
	case float64:
		return int(value)
	case int:
		return value
	default:
		return 0
	}
}
