package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"goshcoder/internal/llm"
)

// AgentOptions configure an Agent (TS AgentOptions in
// reference/pi/packages/agent/src/agent.ts).
type AgentOptions struct {
	// InitialState seeds system prompt, model, thinking level, tools, and
	// messages. Slices are copied.
	InitialState *InitialState
	// StreamFn runs LLM requests. When nil, the loop resolves a streamer per
	// turn via llm.GetStreamer(model.API).StreamSimple.
	StreamFn StreamFn
	// ConvertToLLM filters/transforms transcript messages for the LLM.
	// Default: DefaultConvertToLLM.
	ConvertToLLM ConvertToLLMFunc
	// TransformContext optionally rewrites the transcript before ConvertToLLM.
	TransformContext TransformContextFunc
	// GetAPIKey resolves an API key per LLM call.
	GetAPIKey GetAPIKeyFunc
	// OnPayload can inspect or replace the request payload before sending.
	OnPayload func(payload map[string]any, model *llm.Model) map[string]any
	// OnResponse is invoked after an HTTP response is received.
	OnResponse          func(response llm.ProviderResponse, model *llm.Model)
	BeforeToolCall      BeforeToolCallFunc
	AfterToolCall       AfterToolCallFunc
	ShouldStopAfterTurn ShouldStopAfterTurnFunc
	// PrepareNextTurn replaces context/model/thinking state between turns.
	PrepareNextTurn PrepareNextTurnFunc
	// SteeringMode controls how queued steering messages drain.
	// Default: QueueOneAtATime.
	SteeringMode QueueMode
	// FollowUpMode controls how queued follow-up messages drain.
	// Default: QueueOneAtATime.
	FollowUpMode QueueMode
	// SessionID is forwarded to providers for cache-aware backends.
	SessionID string
	// ThinkingBudgets are per-level thinking token budgets.
	ThinkingBudgets *llm.ThinkingBudgets
	// MaxRetries is the number of retries for transient provider failures such
	// as 429 and 5xx responses. Zero preserves the provider default of no retry.
	MaxRetries int
	// MaxRetryDelayMs caps provider-requested retry delays. Nil uses the
	// provider default.
	MaxRetryDelayMs *int64
	// ToolExecution is the tool execution strategy for assistant messages
	// with multiple tool calls. Default: ToolExecutionParallel.
	ToolExecution ToolExecutionMode
}

// InitialState seeds the agent transcript and settings (subset of TS
// AgentState that callers may initialize).
type InitialState struct {
	SystemPrompt  string
	Model         llm.Model
	ThinkingLevel llm.ModelThinkingLevel
	Tools         []Tool
	Messages      []Message
}

// State is a snapshot of the public agent state (TS AgentState). Slices are
// copies; mutating them does not affect the agent.
type State struct {
	// SystemPrompt is sent with each model request.
	SystemPrompt string
	// Model is used for future turns.
	Model llm.Model
	// ThinkingLevel is the requested reasoning level for future turns.
	ThinkingLevel llm.ModelThinkingLevel
	// Tools are the available tools.
	Tools []Tool
	// Messages are the conversation transcript.
	Messages []Message
	// IsStreaming is true while the agent is processing a prompt or
	// continuation. It remains true until agent_end listeners have run.
	IsStreaming bool
	// StreamingMessage is the partial assistant message for the current
	// streamed response, if any.
	StreamingMessage Message
	// PendingToolCalls lists tool call IDs currently executing.
	PendingToolCalls []string
	// ErrorMessage is the error from the most recent failed or aborted
	// assistant turn, if any.
	ErrorMessage string
}

var defaultModel = llm.Model{
	ID:       "unknown",
	Name:     "unknown",
	API:      "unknown",
	Provider: "unknown",
}

// pendingMessageQueue mirrors PendingMessageQueue in agent.ts.
type pendingMessageQueue struct {
	messages []Message
	mode     QueueMode
}

func (q *pendingMessageQueue) enqueue(m Message) { q.messages = append(q.messages, m) }

func (q *pendingMessageQueue) hasItems() bool { return len(q.messages) > 0 }

func (q *pendingMessageQueue) drain() []Message {
	if q.mode == QueueAll {
		drained := q.messages
		q.messages = nil
		return drained
	}
	if len(q.messages) == 0 {
		return nil
	}
	first := q.messages[0]
	q.messages = q.messages[1:]
	return []Message{first}
}

func (q *pendingMessageQueue) clear() { q.messages = nil }

// activeRun tracks the in-flight run (TS ActiveRun).
type activeRun struct {
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{} // closed when the run and all listeners have settled
}

// Listener receives agent lifecycle events. Listeners are invoked
// synchronously in subscription order and are part of the run's settlement:
// Prompt/Continue return only after every listener for agent_end has run.
// The ctx is the active run's context (canceled by Abort).
type Listener func(ctx context.Context, event Event)

type listenerEntry struct {
	fn Listener
}

// Agent is a stateful wrapper around the low-level agent loop (TS Agent in
// reference/pi/packages/agent/src/agent.ts). It owns the transcript, emits
// lifecycle events, executes tools, and exposes queueing APIs for steering
// and follow-up messages.
//
// An Agent is safe for concurrent use: state access, queue operations,
// Abort, and event delivery are mutex-protected. The TS original relies on
// the single-threaded event loop; the Go port serializes with mutexes.
type Agent struct {
	mu        sync.Mutex // guards state, listeners, queues, activeRun
	emitMu    sync.Mutex // serializes event emission (state mutation + listeners)
	listeners []*listenerEntry

	state struct {
		systemPrompt     string
		model            llm.Model
		thinkingLevel    llm.ModelThinkingLevel
		tools            []Tool
		messages         []Message
		isStreaming      bool
		streamingMessage Message
		pendingToolCalls map[string]bool
		errorMessage     string
	}

	steeringQueue pendingMessageQueue
	followUpQueue pendingMessageQueue

	run *activeRun

	// Options (public in TS; here read-only after construction except where
	// setter methods exist).
	streamFn            StreamFn
	convertToLLM        ConvertToLLMFunc
	transformContext    TransformContextFunc
	getAPIKey           GetAPIKeyFunc
	onPayload           func(map[string]any, *llm.Model) map[string]any
	onResponse          func(llm.ProviderResponse, *llm.Model)
	beforeToolCall      BeforeToolCallFunc
	afterToolCall       AfterToolCallFunc
	shouldStopAfterTurn ShouldStopAfterTurnFunc
	prepareNextTurn     PrepareNextTurnFunc
	sessionID           string
	thinkingBudgets     *llm.ThinkingBudgets
	maxRetries          int
	maxRetryDelayMs     *int64
	toolExecution       ToolExecutionMode
}

// NewAgent creates an Agent from options.
func NewAgent(opts AgentOptions) *Agent {
	a := &Agent{
		streamFn:            opts.StreamFn,
		convertToLLM:        opts.ConvertToLLM,
		transformContext:    opts.TransformContext,
		getAPIKey:           opts.GetAPIKey,
		onPayload:           opts.OnPayload,
		onResponse:          opts.OnResponse,
		beforeToolCall:      opts.BeforeToolCall,
		afterToolCall:       opts.AfterToolCall,
		shouldStopAfterTurn: opts.ShouldStopAfterTurn,
		prepareNextTurn:     opts.PrepareNextTurn,
		sessionID:           opts.SessionID,
		thinkingBudgets:     opts.ThinkingBudgets,
		maxRetries:          opts.MaxRetries,
		maxRetryDelayMs:     opts.MaxRetryDelayMs,
		toolExecution:       opts.ToolExecution,
	}
	if a.convertToLLM == nil {
		a.convertToLLM = DefaultConvertToLLM
	}
	if a.toolExecution == "" {
		a.toolExecution = ToolExecutionParallel
	}
	a.steeringQueue.mode = opts.SteeringMode
	if a.steeringQueue.mode == "" {
		a.steeringQueue.mode = QueueOneAtATime
	}
	a.followUpQueue.mode = opts.FollowUpMode
	if a.followUpQueue.mode == "" {
		a.followUpQueue.mode = QueueOneAtATime
	}
	a.state.model = defaultModel
	a.state.thinkingLevel = llm.ThinkingOff
	a.state.pendingToolCalls = map[string]bool{}
	if s := opts.InitialState; s != nil {
		a.state.systemPrompt = s.SystemPrompt
		if s.Model.ID != "" {
			a.state.model = s.Model
		}
		if s.ThinkingLevel != "" {
			a.state.thinkingLevel = s.ThinkingLevel
		}
		a.state.tools = append([]Tool(nil), s.Tools...)
		a.state.messages = append([]Message(nil), s.Messages...)
	}
	return a
}

// Subscribe registers a listener for agent lifecycle events and returns an
// unsubscribe function (TS Agent.subscribe). Listeners run synchronously in
// subscription order; agent_end is the final event of a run, but the run is
// not idle until all agent_end listeners have returned.
func (a *Agent) Subscribe(l Listener) (unsubscribe func()) {
	entry := &listenerEntry{fn: l}
	a.mu.Lock()
	a.listeners = append(a.listeners, entry)
	a.mu.Unlock()
	return func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		for i, registered := range a.listeners {
			if registered == entry {
				a.listeners = append(a.listeners[:i], a.listeners[i+1:]...)
				return
			}
		}
	}
}

// State returns a snapshot of the current agent state.
func (a *Agent) State() State {
	a.mu.Lock()
	defer a.mu.Unlock()
	s := State{
		SystemPrompt:     a.state.systemPrompt,
		Model:            a.state.model,
		ThinkingLevel:    a.state.thinkingLevel,
		Tools:            append([]Tool(nil), a.state.tools...),
		Messages:         append([]Message(nil), a.state.messages...),
		IsStreaming:      a.state.isStreaming,
		StreamingMessage: a.state.streamingMessage,
		ErrorMessage:     a.state.errorMessage,
	}
	for id := range a.state.pendingToolCalls {
		s.PendingToolCalls = append(s.PendingToolCalls, id)
	}
	return s
}

// SetSystemPrompt sets the system prompt for future turns.
func (a *Agent) SetSystemPrompt(s string) {
	a.mu.Lock()
	a.state.systemPrompt = s
	a.mu.Unlock()
}

// SetModel sets the model for future turns, emitting model_change when the
// model actually differs.
//
// Emitting only on a real change is not an optimization: syncPlanRuntime calls
// the setters twice per turn, so emitting on every call would write two no-op
// entries into the session log for every turn, forever.
func (a *Agent) SetModel(m llm.Model) {
	a.emitMu.Lock()
	defer a.emitMu.Unlock()
	a.mu.Lock()
	changed := a.state.model.Provider != m.Provider || a.state.model.ID != m.ID
	a.state.model = m
	a.mu.Unlock()
	if changed {
		a.emitLocked(nil, Event{Type: EventModelChange, Provider: m.Provider, ModelID: m.ID})
	}
}

// SetThinkingLevel sets the reasoning level for future turns, emitting
// thinking_level_change when it actually differs.
func (a *Agent) SetThinkingLevel(level llm.ModelThinkingLevel) {
	a.emitMu.Lock()
	defer a.emitMu.Unlock()
	a.mu.Lock()
	changed := a.state.thinkingLevel != level
	a.state.thinkingLevel = level
	a.mu.Unlock()
	if changed {
		a.emitLocked(nil, Event{Type: EventThinkingLevelChange, ThinkingLevel: level})
	}
}

// SetTools replaces the tool list. The slice is copied (TS tools setter).
func (a *Agent) SetTools(tools []Tool) {
	a.mu.Lock()
	a.state.tools = append([]Tool(nil), tools...)
	a.mu.Unlock()
}

// SetMessages replaces the transcript. The slice is copied (TS messages
// setter).
//
// This is unrecorded: it emits nothing, so a session recorder cannot see it.
// It exists for seeding an agent at construction and for tests. Anything that
// rewrites a live transcript must go through Compact or Reset instead, or the
// log and the transcript diverge and the next resume replays a conversation
// that never happened.
func (a *Agent) SetMessages(messages []Message) {
	a.mu.Lock()
	a.state.messages = append([]Message(nil), messages...)
	a.mu.Unlock()
}

// SteeringMode returns the steering queue drain mode.
func (a *Agent) SteeringMode() QueueMode {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.steeringQueue.mode
}

// SetSteeringMode controls how queued steering messages are drained.
func (a *Agent) SetSteeringMode(mode QueueMode) {
	a.mu.Lock()
	a.steeringQueue.mode = mode
	a.mu.Unlock()
}

// FollowUpMode returns the follow-up queue drain mode.
func (a *Agent) FollowUpMode() QueueMode {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.followUpQueue.mode
}

// SetFollowUpMode controls how queued follow-up messages are drained.
func (a *Agent) SetFollowUpMode(mode QueueMode) {
	a.mu.Lock()
	a.followUpQueue.mode = mode
	a.mu.Unlock()
}

// Steer queues a message to be injected after the current assistant turn
// finishes.
func (a *Agent) Steer(m Message) {
	a.mu.Lock()
	a.steeringQueue.enqueue(m)
	a.mu.Unlock()
}

// FollowUp queues a message to run only after the agent would otherwise stop.
func (a *Agent) FollowUp(m Message) {
	a.mu.Lock()
	a.followUpQueue.enqueue(m)
	a.mu.Unlock()
}

// ClearSteeringQueue removes all queued steering messages.
func (a *Agent) ClearSteeringQueue() {
	a.mu.Lock()
	a.steeringQueue.clear()
	a.mu.Unlock()
}

// ClearFollowUpQueue removes all queued follow-up messages.
func (a *Agent) ClearFollowUpQueue() {
	a.mu.Lock()
	a.followUpQueue.clear()
	a.mu.Unlock()
}

// ClearAllQueues removes all queued steering and follow-up messages.
func (a *Agent) ClearAllQueues() {
	a.ClearSteeringQueue()
	a.ClearFollowUpQueue()
}

// HasQueuedMessages reports whether either queue still contains pending
// messages.
func (a *Agent) HasQueuedMessages() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.steeringQueue.hasItems() || a.followUpQueue.hasItems()
}

// RunContext returns the active run's context, or nil when idle (TS
// Agent.signal).
func (a *Agent) RunContext() context.Context {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.run == nil {
		return nil
	}
	return a.run.ctx
}

// Abort cancels the current run, if one is active (TS Agent.abort). The
// stream function and tools receive the run context and are expected to
// honor cancellation; blocked tool calls are reported as aborted error
// results.
func (a *Agent) Abort() {
	a.mu.Lock()
	run := a.run
	a.mu.Unlock()
	if run != nil {
		run.cancel()
	}
}

// WaitForIdle blocks until the current run and all event listeners have
// finished (TS Agent.waitForIdle). It returns after agent_end listeners
// settle. It returns immediately when the agent is idle.
func (a *Agent) WaitForIdle() {
	a.mu.Lock()
	run := a.run
	a.mu.Unlock()
	if run != nil {
		<-run.done
	}
}

var errBusy = errors.New("agent is already processing a prompt; use Steer or FollowUp to queue messages, or wait for completion")

// Reset clears transcript state, runtime state, and queued messages. It
// returns an error while a run is active (TS Agent.reset).
func (a *Agent) Reset() error { return a.ResetWithReason("") }

// ResetWithReason is Reset, naming what cleared the transcript so the session
// log can record it.
//
// The emit happens after a.mu is released: emit takes a.mu itself, and a
// listener -- the session recorder is one -- writes to disk while it runs.
func (a *Agent) ResetWithReason(reason string) error {
	a.emitMu.Lock()
	defer a.emitMu.Unlock()
	a.mu.Lock()
	if a.run != nil {
		a.mu.Unlock()
		return errors.New("agent is already processing; wait for completion before resetting")
	}
	a.state.messages = nil
	a.state.isStreaming = false
	a.state.streamingMessage = nil
	a.state.pendingToolCalls = map[string]bool{}
	a.state.errorMessage = ""
	a.steeringQueue.clear()
	a.followUpQueue.clear()
	a.mu.Unlock()

	a.emitLocked(nil, Event{Type: EventTranscriptReset, Reason: reason})
	return nil
}

// Compact replaces the transcript with a summary marker followed by the
// messages worth keeping, and emits context_compacted so the session log
// records the same cut the agent just made.
//
// It exists because compaction previously called SetMessages, which emits
// nothing: a recorder watching only Subscribe would keep appending onto the
// pre-compaction prefix, and the next resume would replay the full uncompacted
// transcript and exceed the context window on its first turn.
//
// Like Reset, it refuses while a run is active: rewriting the transcript out
// from under a streaming turn would strand the run's own message list.
func (a *Agent) Compact(marker Message, kept []Message, info CompactionInfo) error {
	a.emitMu.Lock()
	defer a.emitMu.Unlock()
	a.mu.Lock()
	if a.run != nil {
		a.mu.Unlock()
		return errors.New("agent is already processing; wait for completion before compacting")
	}
	compacted := make([]Message, 0, len(kept)+1)
	compacted = append(compacted, marker)
	compacted = append(compacted, kept...)
	a.state.messages = compacted
	a.mu.Unlock()

	a.emitLocked(nil, Event{
		Type:       EventContextCompacted,
		Message:    marker,
		Kept:       append([]Message(nil), kept...),
		Compaction: &info,
	})
	return nil
}

// Prompt starts a new run from text, a single message, or a batch of
// messages. It blocks until the run settles (including agent_end listeners),
// matching `await agent.prompt(...)` in TS. Run failures are reported through
// events and state, not returned; only usage errors (busy agent, invalid
// input) are returned.
func (a *Agent) Prompt(input any, images ...llm.ImageContent) error {
	return a.runPromptMessages(normalizePromptInput(input, images), false)
}

// Continue continues from the current transcript (TS Agent.continue). The
// last message must be a user or tool-result message; when the transcript
// ends with an assistant message, queued steering messages (then follow-ups)
// are used instead.
func (a *Agent) Continue() error {
	a.mu.Lock()
	if a.run != nil {
		a.mu.Unlock()
		return errBusy
	}
	messages := append([]Message(nil), a.state.messages...)
	a.mu.Unlock()

	if len(messages) == 0 {
		return errors.New("no messages to continue from")
	}

	last := messages[len(messages)-1]
	if RoleOf(last) == "assistant" {
		// Drain one queue at a time. Draining both up front discarded the
		// follow-ups whenever steering also had messages, because only one
		// drain result is ever run.
		a.mu.Lock()
		queuedSteering := a.steeringQueue.drain()
		a.mu.Unlock()
		if len(queuedSteering) > 0 {
			// skipInitialSteeringPoll: the drained messages are the prompt, so
			// the loop must not immediately re-poll the (now empty) queue.
			return a.runPromptMessages(queuedSteering, true)
		}

		a.mu.Lock()
		queuedFollowUps := a.followUpQueue.drain()
		a.mu.Unlock()
		if len(queuedFollowUps) > 0 {
			return a.runPromptMessages(queuedFollowUps, false)
		}
		return errors.New("cannot continue from message role: assistant")
	}

	return a.runContinuation()
}

func normalizePromptInput(input any, images []llm.ImageContent) []Message {
	if s, ok := input.(string); ok {
		content := []llm.ContentBlock{llm.TextContent{Type: "text", Text: s}}
		for _, img := range images {
			content = append(content, img)
		}
		return []Message{llm.UserMessage{Role: "user", Content: content, Timestamp: nowMillis()}}
	}
	if batch, ok := input.([]Message); ok {
		return batch
	}
	// Single message (llm.UserMessage, llm.AssistantMessage,
	// llm.ToolResultMessage, or a custom message).
	return []Message{input}
}

func (a *Agent) runPromptMessages(messages []Message, skipInitialSteeringPoll bool) error {
	return a.runWithLifecycle(func(ctx context.Context, emit func(Event)) {
		runAgentLoop(ctx, messages, a.contextSnapshot(), a.loopConfig(skipInitialSteeringPoll), emit, a.streamFn)
	})
}

func (a *Agent) runContinuation() error {
	return a.runWithLifecycle(func(ctx context.Context, emit func(Event)) {
		if _, err := runAgentLoopContinue(ctx, a.contextSnapshot(), a.loopConfig(false), emit, a.streamFn); err != nil {
			panic(err)
		}
	})
}

func (a *Agent) contextSnapshot() Context {
	a.mu.Lock()
	defer a.mu.Unlock()
	return Context{
		SystemPrompt: a.state.systemPrompt,
		Messages:     append([]Message(nil), a.state.messages...),
		Tools:        append([]Tool(nil), a.state.tools...),
	}
}

// loopConfig snapshots the per-run loop configuration (TS createLoopConfig).
func (a *Agent) loopConfig(skipInitialSteeringPoll bool) loopConfig {
	a.mu.Lock()
	cfg := loopConfig{
		model:               a.state.model,
		sessionID:           a.sessionID,
		onPayload:           a.onPayload,
		onResponse:          a.onResponse,
		thinkingBudgets:     a.thinkingBudgets,
		maxRetries:          a.maxRetries,
		maxRetryDelayMs:     a.maxRetryDelayMs,
		toolExecution:       a.toolExecution,
		beforeToolCall:      a.beforeToolCall,
		afterToolCall:       a.afterToolCall,
		shouldStopAfterTurn: a.shouldStopAfterTurn,
		prepareNextTurn:     a.prepareNextTurn,
		convertToLLM:        a.convertToLLM,
		transformContext:    a.transformContext,
		getAPIKey:           a.getAPIKey,
	}
	// reasoning: "off" maps to no reasoning option (TS: undefined).
	if a.state.thinkingLevel != "" && a.state.thinkingLevel != llm.ThinkingOff {
		cfg.reasoning = llm.ThinkingLevel(a.state.thinkingLevel)
	}
	a.mu.Unlock()

	skipped := skipInitialSteeringPoll
	cfg.getSteeringMessages = func() []Message {
		a.mu.Lock()
		defer a.mu.Unlock()
		if skipped {
			skipped = false
			return nil
		}
		return a.steeringQueue.drain()
	}
	cfg.getFollowUpMessages = func() []Message {
		a.mu.Lock()
		defer a.mu.Unlock()
		return a.followUpQueue.drain()
	}
	return cfg
}

// runWithLifecycle establishes the active run, executes, and settles
// (TS runWithLifecycle / handleRunFailure / finishRun).
func (a *Agent) runWithLifecycle(executor func(ctx context.Context, emit func(Event))) error {
	a.mu.Lock()
	if a.run != nil {
		a.mu.Unlock()
		return errBusy
	}
	ctx, cancel := context.WithCancel(context.Background())
	run := &activeRun{ctx: ctx, cancel: cancel, done: make(chan struct{})}
	a.run = run
	a.state.isStreaming = true
	a.state.streamingMessage = nil
	a.state.errorMessage = ""
	a.mu.Unlock()

	emit := func(ev Event) { a.emit(run, ev) }

	var failure any
	func() {
		defer func() { failure = recover() }()
		executor(ctx, emit)
	}()
	if failure != nil {
		err, ok := failure.(error)
		if !ok {
			err = fmt.Errorf("%v", failure)
		}
		a.handleRunFailure(emit, err, ctx.Err() != nil)
	}

	// finishRun
	a.mu.Lock()
	a.state.isStreaming = false
	a.state.streamingMessage = nil
	a.state.pendingToolCalls = map[string]bool{}
	a.run = nil
	a.mu.Unlock()
	cancel()
	close(run.done)
	return nil
}

// handleRunFailure emits the full lifecycle for a run that failed by
// panicking/returning an error rather than through normal events
// (TS handleRunFailure).
func (a *Agent) handleRunFailure(emit func(Event), err error, aborted bool) {
	a.mu.Lock()
	model := a.state.model
	a.mu.Unlock()
	stopReason := llm.StopError
	if aborted {
		stopReason = llm.StopAborted
	}
	failureMessage := llm.AssistantMessage{
		Role:         "assistant",
		Content:      []llm.ContentBlock{llm.TextContent{Type: "text", Text: ""}},
		API:          model.API,
		Provider:     model.Provider,
		Model:        model.ID,
		StopReason:   stopReason,
		ErrorMessage: err.Error(),
		Timestamp:    nowMillis(),
	}
	emit(Event{Type: EventMessageStart, Message: failureMessage})
	emit(Event{Type: EventMessageEnd, Message: failureMessage})
	emit(Event{Type: EventTurnEnd, Message: failureMessage, ToolResults: []llm.ToolResultMessage{}})
	emit(Event{Type: EventAgentEnd, Messages: []Message{failureMessage}})
}

// emit reduces internal state for a loop event, then invokes listeners in
// subscription order (TS processEvents). agent_end means no further loop
// events will be emitted; the run becomes idle after its listeners return.
func (a *Agent) emit(run *activeRun, ev Event) {
	a.emitMu.Lock()
	defer a.emitMu.Unlock()
	a.emitLocked(run, ev)
}

// emitLocked is emit with emitMu already held. The state setters below take
// emitMu themselves so that their mutation and their event are one atomic step
// with respect to every other emit -- otherwise two concurrent SetModel calls
// can mutate in one order and emit in the other, and the session log ends up
// claiming a turn was held on a model it was not.
//
// No listener may call back into a setter: emitMu is not reentrant.
func (a *Agent) emitLocked(run *activeRun, ev Event) {
	a.mu.Lock()
	switch ev.Type {
	case EventMessageStart, EventMessageUpdate:
		a.state.streamingMessage = ev.Message
	case EventMessageEnd:
		a.state.streamingMessage = nil
		a.state.messages = append(a.state.messages, ev.Message)
	case EventToolExecutionStart:
		a.state.pendingToolCalls[ev.ToolCallID] = true
	case EventToolExecutionEnd:
		delete(a.state.pendingToolCalls, ev.ToolCallID)
	case EventTurnEnd:
		if msg, ok := ev.Message.(llm.AssistantMessage); ok && msg.ErrorMessage != "" {
			a.state.errorMessage = msg.ErrorMessage
		}
	case EventAgentEnd:
		a.state.streamingMessage = nil
	}
	listeners := append([]*listenerEntry(nil), a.listeners...)
	a.mu.Unlock()

	// A setter emits while idle, so there is no run to borrow a context from.
	ctx := context.Background()
	if run != nil {
		ctx = run.ctx
	}
	for _, l := range listeners {
		l.fn(ctx, ev)
	}
}
