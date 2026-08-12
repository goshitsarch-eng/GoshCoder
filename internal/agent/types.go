// Package agent is a Go port of pi's packages/agent agent loop
// (reference/pi/packages/agent/src). Event names, hook call order, and
// queue/abort semantics mirror the TypeScript original.
package agent

import (
	"context"
	"encoding/json"

	"goshcoder/internal/llm"
)

// Message is a transcript entry: one of llm.UserMessage, llm.AssistantMessage,
// llm.ToolResultMessage (TS Message), or a custom app message implementing
// CustomMessage (TS CustomAgentMessages via declaration merging).
//
// Pointers to the llm message types are also accepted for convenience.
type Message = any

// CustomMessage is an app-defined message type carried in the agent
// transcript. ConvertToLLM decides how (or whether) it reaches the model;
// the default converter filters it out.
type CustomMessage interface {
	// MessageRole returns the message role, e.g. "notification".
	MessageRole() string
}

// RoleOf returns the role of a transcript message ("user", "assistant",
// "toolResult", or a custom role). Unknown types return "".
func RoleOf(m Message) string {
	switch v := m.(type) {
	case llm.UserMessage:
		return "user"
	case *llm.UserMessage:
		return "user"
	case llm.AssistantMessage:
		return "assistant"
	case *llm.AssistantMessage:
		return "assistant"
	case llm.ToolResultMessage:
		return "toolResult"
	case *llm.ToolResultMessage:
		return "toolResult"
	case CustomMessage:
		return v.MessageRole()
	default:
		return ""
	}
}

// ToolExecutionMode configures how tool calls from a single assistant message
// are executed (reference/pi/packages/agent/src/types.ts).
type ToolExecutionMode = string

const (
	// ToolExecutionSequential prepares, executes, and finalizes each tool call
	// before the next one starts.
	ToolExecutionSequential ToolExecutionMode = "sequential"
	// ToolExecutionParallel preflights tool calls sequentially, then executes
	// allowed tools concurrently. tool_execution_end is emitted in tool
	// completion order; tool-result messages are emitted afterwards in
	// assistant source order.
	ToolExecutionParallel ToolExecutionMode = "parallel"
)

// QueueMode controls how many queued messages are injected when the agent
// loop reaches a queue drain point.
type QueueMode = string

const (
	// QueueAll drains and injects every queued message at a drain point.
	QueueAll QueueMode = "all"
	// QueueOneAtATime drains and injects only the oldest queued message,
	// leaving the rest queued for later drain points.
	QueueOneAtATime QueueMode = "one-at-a-time"
)

// StreamFn runs one LLM request, returning an event stream (TS StreamFn in
// reference/pi/packages/agent/src/types.ts).
//
// Contract: must not panic or fail for request/model/runtime errors. Failures
// are encoded in the returned stream via protocol events and a final
// AssistantMessage with stopReason "error" or "aborted" and ErrorMessage set.
type StreamFn func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream

// ToolResult is the final or partial result produced by a tool
// (TS AgentToolResult<TDetails>).
type ToolResult struct {
	// Content is the text/image content returned to the model.
	Content []llm.ContentBlock
	// Details holds arbitrary structured details for logs or UI rendering.
	Details any
	// Usage from the tool execution itself, if available. Not used for main
	// LLM context accounting.
	Usage *llm.Usage
	// AddedToolNames lists tools introduced by this result, available from
	// this transcript point onward.
	AddedToolNames []string
	// Terminate hints that the agent should stop after the current tool
	// batch. Early termination only happens when every finalized tool result
	// in the batch sets this to true.
	Terminate bool
}

// Tool is the tool definition used by the agent runtime (TS AgentTool).
type Tool struct {
	// Name, Description, and Parameters (raw JSON Schema) mirror llm.Tool.
	Name        string
	Description string
	Parameters  json.RawMessage
	// Label is a human-readable label for UI display.
	Label string
	// PrepareArguments is an optional compatibility shim applied to raw
	// tool-call arguments before schema validation. Must return arguments
	// that match Parameters.
	PrepareArguments func(args map[string]any) map[string]any
	// Execute runs the tool call. Return an error on failure instead of
	// encoding errors in Content; the loop converts errors into isError tool
	// results. onUpdate may be called to stream partial results; calls made
	// after Execute returns are ignored.
	Execute func(ctx context.Context, toolCallID string, params map[string]any, onUpdate func(partial ToolResult)) (ToolResult, error)
	// ExecutionMode overrides the global tool execution mode for this tool.
	// Empty means the default applies. Setting ToolExecutionSequential forces
	// the whole batch to run sequentially (matching the TS preflight).
	ExecutionMode ToolExecutionMode
}

// LLMTool converts the tool to its wire-protocol form.
func (t Tool) LLMTool() llm.Tool {
	return llm.Tool{Name: t.Name, Description: t.Description, Parameters: t.Parameters}
}

// EventType is the discriminant of an agent lifecycle event. Names are kept
// identical to the TS originals.
type EventType = string

const (
	EventAgentStart          EventType = "agent_start"
	EventAgentEnd            EventType = "agent_end"
	EventTurnStart           EventType = "turn_start"
	EventTurnEnd             EventType = "turn_end"
	EventMessageStart        EventType = "message_start"
	EventMessageUpdate       EventType = "message_update"
	EventMessageEnd          EventType = "message_end"
	EventToolExecutionStart  EventType = "tool_execution_start"
	EventToolExecutionUpdate EventType = "tool_execution_update"
	EventToolExecutionEnd    EventType = "tool_execution_end"
)

// Event is an agent lifecycle event (TS AgentEvent union flattened into one
// struct). Which fields are populated depends on Type:
//
//	agent_start:           (none)
//	agent_end:             Messages
//	turn_start:            (none)
//	turn_end:              Message, ToolResults
//	message_start/end:     Message
//	message_update:        Message, AssistantEvent (assistant messages only)
//	tool_execution_start:  ToolCallID, ToolName, Args
//	tool_execution_update: ToolCallID, ToolName, Args, PartialResult
//	tool_execution_end:    ToolCallID, ToolName, Result, IsError
type Event struct {
	Type EventType
	// Message is the message for message_* and turn_end events.
	Message Message
	// Messages is set on agent_end: the messages produced by this run.
	Messages []Message
	// ToolResults is set on turn_end.
	ToolResults []llm.ToolResultMessage
	// AssistantEvent is the underlying stream event for message_update.
	AssistantEvent *llm.AssistantMessageEvent
	// ToolCallID and ToolName identify the tool call for tool_execution_*.
	ToolCallID string
	ToolName   string
	// Args are the raw tool call arguments for tool_execution_start/update.
	Args map[string]any
	// PartialResult is set on tool_execution_update.
	PartialResult *ToolResult
	// Result is the final tool result on tool_execution_end.
	Result  *ToolResult
	IsError bool
}

// BeforeToolCallResult is returned from BeforeToolCall. Block prevents the
// tool from executing; the loop emits an error tool result with Reason (or a
// default blocked message) instead. Terminate participates in the batch
// early-termination rule.
type BeforeToolCallResult struct {
	Block     bool
	Reason    string
	Terminate bool
}

// AfterToolCallResult is a partial override returned from AfterToolCall.
// Merge semantics are field-by-field (TS AfterToolCallResult): nil/omitted
// fields keep the executed tool result values. There is no deep merge for
// Content, Details, or Usage.
type AfterToolCallResult struct {
	Content   []llm.ContentBlock // nil = keep original
	Details   any                // nil = keep original
	IsError   *bool              // nil = keep original
	Usage     *llm.Usage         // nil = keep original
	Terminate *bool              // nil = keep original
}

// BeforeToolCallContext is passed to BeforeToolCall.
type BeforeToolCallContext struct {
	// AssistantMessage requested the tool call.
	AssistantMessage llm.AssistantMessage
	// ToolCall is the raw tool call block from AssistantMessage.Content.
	ToolCall llm.ToolCall
	// Args are the validated tool arguments for the target tool schema.
	Args map[string]any
	// Context is the agent context at the time the tool call is prepared.
	Context Context
}

// AfterToolCallContext is passed to AfterToolCall.
type AfterToolCallContext struct {
	AssistantMessage llm.AssistantMessage
	ToolCall         llm.ToolCall
	Args             map[string]any
	// Result is the executed tool result before overrides are applied.
	Result ToolResult
	// IsError reports whether Result is currently treated as an error.
	IsError bool
	Context Context
}

// ShouldStopAfterTurnContext is passed to ShouldStopAfterTurn and
// PrepareNextTurn.
type ShouldStopAfterTurnContext struct {
	// Message is the assistant message that completed the turn.
	Message llm.AssistantMessage
	// ToolResults were passed to the preceding turn_end event.
	ToolResults []llm.ToolResultMessage
	// Context is the agent context after the turn's assistant message and
	// tool results have been appended.
	Context Context
	// NewMessages are the messages this loop invocation returns if it exits
	// at this point.
	NewMessages []Message
}

// PrepareNextTurnContext is passed to PrepareNextTurn.
type PrepareNextTurnContext = ShouldStopAfterTurnContext

// TurnUpdate is replacement runtime state used by the agent loop before
// starting another provider request (TS AgentLoopTurnUpdate). Nil fields
// keep the current values.
type TurnUpdate struct {
	Context *Context
	Model   *llm.Model
	// ThinkingLevel: "" keeps the current level; llm.ThinkingOff clears
	// reasoning; any other level sets it.
	ThinkingLevel llm.ModelThinkingLevel
}

// Context is a snapshot passed into the low-level agent loop (TS AgentContext).
type Context struct {
	SystemPrompt string
	Messages     []Message
	Tools        []Tool
}

// Hook function types. In the TS these return promises; in Go they are
// synchronous and receive the run's context (the abort signal).

// ConvertToLLMFunc converts transcript messages to LLM-compatible messages
// before each LLM call. Custom messages that cannot be converted should be
// filtered out. Contract: must not panic; return a safe fallback instead.
type ConvertToLLMFunc func(messages []Message) []llm.Message

// TransformContextFunc is an optional transform applied to the context before
// ConvertToLLM (context window management, external context injection).
// Contract: must not panic; return the original messages on failure.
type TransformContextFunc func(ctx context.Context, messages []Message) []Message

// GetAPIKeyFunc resolves an API key dynamically for each LLM call (useful for
// short-lived OAuth tokens). Return "" when no key is available.
type GetAPIKeyFunc func(provider string) string

// ShouldStopAfterTurnFunc is called after each turn fully completes and
// turn_end has been emitted. Returning true makes the loop emit agent_end and
// exit before polling steering or follow-up queues.
type ShouldStopAfterTurnFunc func(ctx context.Context, c ShouldStopAfterTurnContext) bool

// PrepareNextTurnFunc is called after turn_end and before the loop decides
// whether another provider request should start. Return a TurnUpdate to
// replace context/model/thinking state for the next turn, or nil to keep the
// current state.
type PrepareNextTurnFunc func(ctx context.Context, c PrepareNextTurnContext) *TurnUpdate

// BeforeToolCallFunc is called before a tool is executed, after arguments
// have been validated. The hook receives the run context and is responsible
// for honoring cancellation.
type BeforeToolCallFunc func(ctx context.Context, c BeforeToolCallContext) *BeforeToolCallResult

// AfterToolCallFunc is called after a tool finishes executing, before
// tool_execution_end and tool-result message events are emitted.
type AfterToolCallFunc func(ctx context.Context, c AfterToolCallContext) *AfterToolCallResult

// DefaultConvertToLLM is the default ConvertToLLM: it keeps user, assistant,
// and toolResult messages and filters out custom message types
// (defaultConvertToLlm in reference/pi/packages/agent/src/agent.ts).
func DefaultConvertToLLM(messages []Message) []llm.Message {
	out := make([]llm.Message, 0, len(messages))
	for _, m := range messages {
		switch v := m.(type) {
		case llm.UserMessage:
			out = append(out, v)
		case *llm.UserMessage:
			out = append(out, *v)
		case llm.AssistantMessage:
			out = append(out, v)
		case *llm.AssistantMessage:
			out = append(out, *v)
		case llm.ToolResultMessage:
			out = append(out, v)
		case *llm.ToolResultMessage:
			out = append(out, *v)
		}
	}
	return out
}
