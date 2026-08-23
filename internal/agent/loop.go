package agent

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"goshcoder/internal/llm"
)

func nowMillis() int64 { return time.Now().UnixMilli() }

// loopConfig is the per-run loop configuration (TS AgentLoopConfig). The
// queue poll functions are bound to the agent's queues at run start.
type loopConfig struct {
	model               llm.Model
	reasoning           llm.ThinkingLevel // "" means no reasoning option (TS undefined)
	sessionID           string
	onPayload           func(map[string]any, *llm.Model) map[string]any
	onResponse          func(llm.ProviderResponse, *llm.Model)
	thinkingBudgets     *llm.ThinkingBudgets
	maxRetries          int
	maxRetryDelayMs     *int64
	toolExecution       ToolExecutionMode
	beforeToolCall      BeforeToolCallFunc
	afterToolCall       AfterToolCallFunc
	shouldStopAfterTurn ShouldStopAfterTurnFunc
	prepareNextTurn     PrepareNextTurnFunc
	convertToLLM        ConvertToLLMFunc
	transformContext    TransformContextFunc
	getAPIKey           GetAPIKeyFunc
	getSteeringMessages func() []Message
	getFollowUpMessages func() []Message
}

// runAgentLoop starts an agent loop with new prompt messages (TS runAgentLoop
// in reference/pi/packages/agent/src/agent-loop.ts). The prompts are added to
// the context and events are emitted for them. Returns the messages produced
// by the run.
func runAgentLoop(ctx context.Context, prompts []Message, agentCtx Context, config loopConfig, emit func(Event), streamFn StreamFn) []Message {
	newMessages := append([]Message(nil), prompts...)
	currentContext := Context{
		SystemPrompt: agentCtx.SystemPrompt,
		Messages:     append(append([]Message(nil), agentCtx.Messages...), prompts...),
		Tools:        agentCtx.Tools,
	}

	emit(Event{Type: EventAgentStart})
	emit(Event{Type: EventTurnStart})
	for _, prompt := range prompts {
		emit(Event{Type: EventMessageStart, Message: prompt})
		emit(Event{Type: EventMessageEnd, Message: prompt})
	}

	runLoop(ctx, &currentContext, &newMessages, &config, emit, streamFn)
	return newMessages
}

// runAgentLoopContinue continues an agent loop from the current context
// without adding a new message (TS runAgentLoopContinue). Used for retries:
// the context already ends with a user message or tool results.
func runAgentLoopContinue(ctx context.Context, agentCtx Context, config loopConfig, emit func(Event), streamFn StreamFn) ([]Message, error) {
	if len(agentCtx.Messages) == 0 {
		return nil, fmt.Errorf("cannot continue: no messages in context")
	}
	if RoleOf(agentCtx.Messages[len(agentCtx.Messages)-1]) == "assistant" {
		return nil, fmt.Errorf("cannot continue from message role: assistant")
	}

	newMessages := []Message{}
	currentContext := Context{
		SystemPrompt: agentCtx.SystemPrompt,
		Messages:     append([]Message(nil), agentCtx.Messages...),
		Tools:        agentCtx.Tools,
	}

	emit(Event{Type: EventAgentStart})
	emit(Event{Type: EventTurnStart})

	runLoop(ctx, &currentContext, &newMessages, &config, emit, streamFn)
	return newMessages, nil
}

// runLoop is the main loop shared by runAgentLoop and runAgentLoopContinue
// (TS runLoop). The inner loop processes tool calls and steering messages;
// the outer loop continues when queued follow-up messages arrive after the
// agent would stop.
func runLoop(ctx context.Context, currentContext *Context, newMessages *[]Message, config *loopConfig, emit func(Event), streamFn StreamFn) {
	firstTurn := true
	// Check for steering messages at start (user may have typed while waiting).
	var pendingMessages []Message
	if config.getSteeringMessages != nil {
		pendingMessages = config.getSteeringMessages()
	}

	// Outer loop: continues when queued follow-up messages arrive after the
	// agent would stop.
	for {
		hasMoreToolCalls := true

		// Inner loop: process tool calls and steering messages.
		for hasMoreToolCalls || len(pendingMessages) > 0 {
			// A batch cut short by an abort still leaves hasMoreToolCalls set,
			// so without this the loop opened another turn and issued another
			// provider request with an already-cancelled context -- a wasted
			// round trip on every abort, plus a spurious empty assistant
			// message appended to the transcript.
			if ctx.Err() != nil {
				emit(Event{Type: EventAgentEnd, Messages: append([]Message(nil), (*newMessages)...)})
				return
			}
			if !firstTurn {
				emit(Event{Type: EventTurnStart})
			} else {
				firstTurn = false
			}

			// Process pending messages (inject before next assistant response).
			if len(pendingMessages) > 0 {
				for _, message := range pendingMessages {
					emit(Event{Type: EventMessageStart, Message: message})
					emit(Event{Type: EventMessageEnd, Message: message})
					currentContext.Messages = append(currentContext.Messages, message)
					*newMessages = append(*newMessages, message)
				}
				pendingMessages = nil
			}

			// Stream assistant response.
			message := streamAssistantResponse(ctx, currentContext, config, emit, streamFn)
			*newMessages = append(*newMessages, message)

			if message.StopReason == llm.StopError || message.StopReason == llm.StopAborted {
				emit(Event{Type: EventTurnEnd, Message: message, ToolResults: []llm.ToolResultMessage{}})
				emit(Event{Type: EventAgentEnd, Messages: append([]Message(nil), (*newMessages)...)})
				return
			}

			// Check for tool calls.
			var toolCalls []llm.ToolCall
			for _, block := range message.Content {
				if tc, ok := block.(llm.ToolCall); ok {
					toolCalls = append(toolCalls, tc)
				}
			}

			var toolResults []llm.ToolResultMessage
			hasMoreToolCalls = false
			if len(toolCalls) > 0 {
				// A "length" stop means the output was cut off by the token
				// limit, so every tool call in the message may carry truncated
				// arguments. Fail them all instead of executing potentially
				// borked calls.
				var batch executedToolCallBatch
				if message.StopReason == llm.StopLength {
					batch = failToolCallsFromTruncatedMessage(toolCalls, emit)
				} else {
					batch = executeToolCalls(ctx, currentContext, message, config, emit)
				}
				toolResults = batch.messages
				hasMoreToolCalls = !batch.terminate

				for _, result := range toolResults {
					currentContext.Messages = append(currentContext.Messages, result)
					*newMessages = append(*newMessages, result)
				}
			}

			emit(Event{Type: EventTurnEnd, Message: message, ToolResults: toolResults})

			turnCtx := ShouldStopAfterTurnContext{
				Message:     message,
				ToolResults: toolResults,
				Context:     *currentContext,
				NewMessages: append([]Message(nil), (*newMessages)...),
			}
			if config.prepareNextTurn != nil {
				if update := config.prepareNextTurn(ctx, turnCtx); update != nil {
					if update.Context != nil {
						*currentContext = *update.Context
					}
					if update.Model != nil {
						config.model = *update.Model
					}
					// "" keeps the current reasoning; "off" clears it.
					switch update.ThinkingLevel {
					case "":
					case llm.ThinkingOff:
						config.reasoning = ""
					default:
						config.reasoning = llm.ThinkingLevel(update.ThinkingLevel)
					}
				}
			}

			if config.shouldStopAfterTurn != nil && config.shouldStopAfterTurn(ctx, turnCtx) {
				emit(Event{Type: EventAgentEnd, Messages: append([]Message(nil), (*newMessages)...)})
				return
			}

			if config.getSteeringMessages != nil {
				pendingMessages = config.getSteeringMessages()
			} else {
				pendingMessages = nil
			}
		}

		// Agent would stop here. Check for follow-up messages.
		var followUpMessages []Message
		if config.getFollowUpMessages != nil {
			followUpMessages = config.getFollowUpMessages()
		}
		if len(followUpMessages) > 0 {
			// Set as pending so the inner loop processes them.
			pendingMessages = followUpMessages
			continue
		}

		// No more messages, exit.
		break
	}

	emit(Event{Type: EventAgentEnd, Messages: append([]Message(nil), (*newMessages)...)})
}

// streamAssistantResponse streams one assistant response from the LLM
// (TS streamAssistantResponse). This is where transcript messages get
// transformed to llm.Message for the LLM call.
//
// The event loop ends only when the stream ends (matching the TS for-await
// loop): cancellation is delivered to the stream function via opts.Ctx, and
// providers encode aborts as a final message with stopReason "aborted".
func streamAssistantResponse(ctx context.Context, agentCtx *Context, config *loopConfig, emit func(Event), streamFn StreamFn) llm.AssistantMessage {
	// Apply context transform if configured (Message[] -> Message[]).
	messages := agentCtx.Messages
	if config.transformContext != nil {
		messages = config.transformContext(ctx, messages)
	}

	// Convert to LLM-compatible messages (Message[] -> llm.Message[]).
	convertToLLM := config.convertToLLM
	if convertToLLM == nil {
		convertToLLM = DefaultConvertToLLM
	}
	llmMessages := convertToLLM(messages)

	// Build LLM context.
	llmTools := make([]llm.Tool, 0, len(agentCtx.Tools))
	for _, t := range agentCtx.Tools {
		llmTools = append(llmTools, t.LLMTool())
	}
	llmContext := &llm.Context{
		SystemPrompt: agentCtx.SystemPrompt,
		Messages:     llmMessages,
		Tools:        llmTools,
	}

	// Resolve API key (important for expiring tokens).
	apiKey := ""
	if config.getAPIKey != nil {
		apiKey = config.getAPIKey(config.model.Provider)
	}

	opts := &llm.SimpleStreamOptions{
		Reasoning:       config.reasoning,
		ThinkingBudgets: config.thinkingBudgets,
	}
	opts.Ctx = ctx
	opts.APIKey = apiKey
	opts.SessionID = config.sessionID
	opts.OnPayload = config.onPayload
	opts.OnResponse = config.onResponse
	opts.MaxRetries = config.maxRetries
	opts.MaxRetryDelayMs = config.maxRetryDelayMs

	fn := streamFn
	if fn == nil {
		fn = defaultStreamFn
	}
	response := fn(&config.model, llmContext, opts)

	var partialMessage *llm.AssistantMessage
	addedPartial := false

	// replaceLast swaps the streamed partial into the context tail.
	replaceLast := func(m llm.AssistantMessage) {
		agentCtx.Messages[len(agentCtx.Messages)-1] = m
	}

	for {
		event, ok := response.Next(ctx)
		if !ok {
			break
		}
		switch event.Type {
		case llm.EventStart:
			partialMessage = event.Partial
			agentCtx.Messages = append(agentCtx.Messages, *partialMessage)
			addedPartial = true
			emit(Event{Type: EventMessageStart, Message: *partialMessage})

		case llm.EventTextStart, llm.EventTextDelta, llm.EventTextEnd,
			llm.EventThinkingStart, llm.EventThinkingDelta, llm.EventThinkingEnd,
			llm.EventToolCallStart, llm.EventToolCallDelta, llm.EventToolCallEnd:
			if partialMessage != nil {
				partialMessage = event.Partial
				replaceLast(*partialMessage)
				ev := event
				emit(Event{Type: EventMessageUpdate, AssistantEvent: &ev, Message: *partialMessage})
			}

		case llm.EventDone, llm.EventError:
			finalMessage := streamResult(ctx, response, partialMessage)
			if addedPartial {
				replaceLast(finalMessage)
			} else {
				agentCtx.Messages = append(agentCtx.Messages, finalMessage)
				emit(Event{Type: EventMessageStart, Message: finalMessage})
			}
			emit(Event{Type: EventMessageEnd, Message: finalMessage})
			return finalMessage
		}
	}

	// Stream ended without a terminal done/error event, or the run context was
	// canceled while waiting. Abort must not wait on a hung provider.
	finalMessage := streamResult(ctx, response, partialMessage)
	if addedPartial {
		replaceLast(finalMessage)
	} else {
		agentCtx.Messages = append(agentCtx.Messages, finalMessage)
		emit(Event{Type: EventMessageStart, Message: finalMessage})
	}
	emit(Event{Type: EventMessageEnd, Message: finalMessage})
	return finalMessage
}

// streamResult resolves the stream's final message. A stream that closes
// without a terminal event yields an error stop message so the loop can exit
// through the normal error path. Cancellation becomes stopReason "aborted".
func streamResult(ctx context.Context, response *llm.AssistantMessageEventStream, partial *llm.AssistantMessage) llm.AssistantMessage {
	final, err := response.Result(ctx)
	if err == nil && final != nil {
		return *final
	}
	stop := llm.StopError
	errText := llm.ErrStreamClosedWithoutResult.Error()
	if err != nil {
		errText = err.Error()
	}
	if ctx.Err() != nil {
		stop = llm.StopAborted
		errText = "Request was aborted"
	}
	if partial != nil {
		message := *partial
		message.StopReason = stop
		message.ErrorMessage = errText
		return message
	}
	return llm.AssistantMessage{
		Role:         "assistant",
		Content:      []llm.ContentBlock{llm.TextContent{Type: "text", Text: ""}},
		StopReason:   stop,
		ErrorMessage: errText,
		Timestamp:    nowMillis(),
	}
}

// defaultStreamFn resolves the registered streamer for the model's API,
// matching pi's streamSimple default. An unregistered API yields a stream
// carrying an error stop message (StreamFn contract: never throw).
func defaultStreamFn(model *llm.Model, llmCtx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
	funcs, ok := llm.GetStreamer(model.API)
	if !ok || funcs.StreamSimple == nil {
		stream := llm.NewAssistantMessageEventStream()
		stream.Push(llm.AssistantMessageEvent{
			Type:   llm.EventError,
			Reason: llm.StopError,
			Error: &llm.AssistantMessage{
				Role:         "assistant",
				Content:      []llm.ContentBlock{llm.TextContent{Type: "text", Text: ""}},
				API:          model.API,
				Provider:     model.Provider,
				Model:        model.ID,
				StopReason:   llm.StopError,
				ErrorMessage: fmt.Sprintf("no streamer registered for api %q", model.API),
				Timestamp:    nowMillis(),
			},
		})
		stream.End()
		return stream
	}
	return funcs.StreamSimple(model, llmCtx, opts)
}

// executedToolCallBatch is the outcome of one assistant message's tool calls
// (TS ExecutedToolCallBatch).
type executedToolCallBatch struct {
	messages  []llm.ToolResultMessage
	terminate bool
}

// preparedToolCall is a tool call that passed preflight and can execute
// (TS PreparedToolCall).
type preparedToolCall struct {
	toolCall llm.ToolCall
	tool     Tool
	args     map[string]any
}

// immediateOutcome is a tool call outcome produced during preflight without
// executing the tool (TS ImmediateToolCallOutcome): unknown tool, validation
// failure, blocked call, or abort.
type immediateOutcome struct {
	result  ToolResult
	isError bool
}

// finalizedOutcome pairs a tool call with its final result (TS
// FinalizedToolCallOutcome).
type finalizedOutcome struct {
	toolCall llm.ToolCall
	result   ToolResult
	isError  bool
}

// failToolCallsFromTruncatedMessage fails all tool calls from an assistant
// message truncated by the output token limit (TS
// failToolCallsFromTruncatedMessage). Streamed tool-call arguments are
// finalized with a best-effort JSON salvage parser, so a truncated message
// can yield tool calls whose arguments parse and validate but are silently
// incomplete. None of them are safe to execute; report each as an error so
// the model can re-issue them.
func failToolCallsFromTruncatedMessage(toolCalls []llm.ToolCall, emit func(Event)) executedToolCallBatch {
	var messages []llm.ToolResultMessage
	for _, toolCall := range toolCalls {
		emit(Event{Type: EventToolExecutionStart, ToolCallID: toolCall.ID, ToolName: toolCall.Name, Args: toolCall.Arguments})
		finalized := finalizedOutcome{
			toolCall: toolCall,
			result: createErrorToolResult(fmt.Sprintf(
				"Tool call %q was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
				toolCall.Name,
			)),
			isError: true,
		}
		emitToolExecutionEnd(finalized, emit)
		toolResultMessage := createToolResultMessage(finalized)
		emitToolResultMessage(toolResultMessage, emit)
		messages = append(messages, toolResultMessage)
	}
	return executedToolCallBatch{messages: messages, terminate: false}
}

// executeToolCalls executes the tool calls from an assistant message
// (TS executeToolCalls). A single sequential-mode tool forces the whole batch
// to run sequentially.
func executeToolCalls(ctx context.Context, currentContext *Context, assistantMessage llm.AssistantMessage, config *loopConfig, emit func(Event)) executedToolCallBatch {
	var toolCalls []llm.ToolCall
	for _, block := range assistantMessage.Content {
		if tc, ok := block.(llm.ToolCall); ok {
			toolCalls = append(toolCalls, tc)
		}
	}
	hasSequentialToolCall := false
	for _, tc := range toolCalls {
		if tool := findTool(currentContext.Tools, tc.Name); tool != nil && tool.ExecutionMode == ToolExecutionSequential {
			hasSequentialToolCall = true
			break
		}
	}
	if config.toolExecution == ToolExecutionSequential || hasSequentialToolCall {
		return executeToolCallsSequential(ctx, currentContext, assistantMessage, toolCalls, config, emit)
	}
	return executeToolCallsParallel(ctx, currentContext, assistantMessage, toolCalls, config, emit)
}

func findTool(tools []Tool, name string) *Tool {
	for i := range tools {
		if tools[i].Name == name {
			return &tools[i]
		}
	}
	return nil
}

// executeToolCallsSequential prepares, executes, and finalizes each tool call
// before the next one starts (TS executeToolCallsSequential).
func executeToolCallsSequential(ctx context.Context, currentContext *Context, assistantMessage llm.AssistantMessage, toolCalls []llm.ToolCall, config *loopConfig, emit func(Event)) executedToolCallBatch {
	var finalizedCalls []finalizedOutcome
	var messages []llm.ToolResultMessage

	for _, toolCall := range toolCalls {
		emit(Event{Type: EventToolExecutionStart, ToolCallID: toolCall.ID, ToolName: toolCall.Name, Args: toolCall.Arguments})

		prepared, immediate := prepareToolCall(ctx, currentContext, assistantMessage, toolCall, config)
		var finalized finalizedOutcome
		if immediate != nil {
			finalized = finalizedOutcome{toolCall: toolCall, result: immediate.result, isError: immediate.isError}
		} else {
			executed := executePreparedToolCall(ctx, prepared, emit)
			finalized = finalizeExecutedToolCall(ctx, currentContext, assistantMessage, prepared, executed, config)
		}

		emitToolExecutionEnd(finalized, emit)
		toolResultMessage := createToolResultMessage(finalized)
		emitToolResultMessage(toolResultMessage, emit)
		finalizedCalls = append(finalizedCalls, finalized)
		messages = append(messages, toolResultMessage)
	}

	return executedToolCallBatch{
		messages:  messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	}
}

// executeToolCallsParallel preflights tool calls sequentially, then executes
// allowed tools concurrently (TS executeToolCallsParallel).
// tool_execution_end is emitted in tool completion order after each tool is
// finalized; tool-result messages are emitted later in assistant source order.
func executeToolCallsParallel(ctx context.Context, currentContext *Context, assistantMessage llm.AssistantMessage, toolCalls []llm.ToolCall, config *loopConfig, emit func(Event)) executedToolCallBatch {
	// entries preserves assistant source order: each entry is either an
	// already-finalized outcome or an index into thunk results.
	type entry struct {
		finalized *finalizedOutcome
		thunkIdx  int
	}
	var entries []entry
	var thunks []preparedToolCall

	for _, toolCall := range toolCalls {
		emit(Event{Type: EventToolExecutionStart, ToolCallID: toolCall.ID, ToolName: toolCall.Name, Args: toolCall.Arguments})

		prepared, immediate := prepareToolCall(ctx, currentContext, assistantMessage, toolCall, config)
		if immediate != nil {
			finalized := finalizedOutcome{toolCall: toolCall, result: immediate.result, isError: immediate.isError}
			emitToolExecutionEnd(finalized, emit)
			entries = append(entries, entry{finalized: &finalized, thunkIdx: -1})
			continue
		}

		thunks = append(thunks, *prepared)
		entries = append(entries, entry{thunkIdx: len(thunks) - 1})
	}

	// Execute prepared calls concurrently; each worker finalizes and emits
	// tool_execution_end when it completes (completion order).
	results := make([]finalizedOutcome, len(thunks))
	var wg sync.WaitGroup
	for i, prepared := range thunks {
		wg.Add(1)
		go func(i int, prepared preparedToolCall) {
			defer wg.Done()
			// Every other tool path recovers, and the run as a whole is
			// wrapped, but neither covers a goroutine: a panic raised here --
			// including one from a subscriber reached through emit -- unwinds
			// past all of them and takes the process down with the transcript.
			defer func() {
				if r := recover(); r != nil {
					results[i] = finalizedOutcome{
						toolCall: prepared.toolCall,
						result:   createErrorToolResult(fmt.Sprint(r)),
						isError:  true,
					}
				}
			}()
			executed := executePreparedToolCall(ctx, &prepared, emit)
			results[i] = finalizeExecutedToolCall(ctx, currentContext, assistantMessage, &prepared, executed, config)
			emitToolExecutionEnd(results[i], emit)
		}(i, prepared)
	}
	wg.Wait()

	var messages []llm.ToolResultMessage
	var finalizedCalls []finalizedOutcome
	for _, e := range entries {
		var finalized finalizedOutcome
		if e.finalized != nil {
			finalized = *e.finalized
		} else {
			finalized = results[e.thunkIdx]
		}
		finalizedCalls = append(finalizedCalls, finalized)
		toolResultMessage := createToolResultMessage(finalized)
		emitToolResultMessage(toolResultMessage, emit)
		messages = append(messages, toolResultMessage)
	}

	return executedToolCallBatch{
		messages:  messages,
		terminate: shouldTerminateToolBatch(finalizedCalls),
	}
}

// shouldTerminateToolBatch reports early termination: every finalized tool
// result in the batch must set terminate (TS shouldTerminateToolBatch).
func shouldTerminateToolBatch(finalizedCalls []finalizedOutcome) bool {
	if len(finalizedCalls) == 0 {
		return false
	}
	for _, finalized := range finalizedCalls {
		if !finalized.result.Terminate {
			return false
		}
	}
	return true
}

// prepareToolCall resolves the tool, prepares and validates arguments, and
// runs the beforeToolCall hook (TS prepareToolCall). An immediate outcome is
// returned for unknown tools, validation failures, blocked calls, and aborts;
// hook panics become immediate error results (TS try/catch).
func prepareToolCall(ctx context.Context, currentContext *Context, assistantMessage llm.AssistantMessage, toolCall llm.ToolCall, config *loopConfig) (prepared *preparedToolCall, immediate *immediateOutcome) {
	defer func() {
		if r := recover(); r != nil {
			prepared = nil
			immediate = &immediateOutcome{result: createErrorToolResult(fmt.Sprint(r)), isError: true}
		}
	}()

	tool := findTool(currentContext.Tools, toolCall.Name)
	if tool == nil {
		return nil, &immediateOutcome{
			result:  createErrorToolResult(fmt.Sprintf("Tool %s not found", toolCall.Name)),
			isError: true,
		}
	}

	args := toolCall.Arguments
	if tool.PrepareArguments != nil {
		args = tool.PrepareArguments(args)
	}
	validatedArgs, err := validateToolArguments(*tool, toolCall.Name, args)
	if err != nil {
		return nil, &immediateOutcome{result: createErrorToolResult(err.Error()), isError: true}
	}

	if config.beforeToolCall != nil {
		beforeResult := config.beforeToolCall(ctx, BeforeToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         toolCall,
			Args:             validatedArgs,
			Context:          *currentContext,
		})
		if ctx.Err() != nil {
			return nil, &immediateOutcome{result: createErrorToolResult("Operation aborted"), isError: true}
		}
		if beforeResult != nil && beforeResult.Block {
			result := createErrorToolResult("Tool execution was blocked")
			if beforeResult.Reason != "" {
				result = createErrorToolResult(beforeResult.Reason)
			}
			result.Terminate = beforeResult.Terminate
			return nil, &immediateOutcome{result: result, isError: true}
		}
	}
	if ctx.Err() != nil {
		return nil, &immediateOutcome{result: createErrorToolResult("Operation aborted"), isError: true}
	}
	return &preparedToolCall{toolCall: toolCall, tool: *tool, args: validatedArgs}, nil
}

// executedOutcome is the raw result of running a tool (TS
// ExecutedToolCallOutcome).
type executedOutcome struct {
	result  ToolResult
	isError bool
}

// executePreparedToolCall runs the tool (TS executePreparedToolCall). Tool
// errors and panics become isError results. Update callbacks are ignored once
// the tool has returned.
func executePreparedToolCall(ctx context.Context, prepared *preparedToolCall, emit func(Event)) (outcome executedOutcome) {
	var acceptingUpdates atomic.Bool

	onUpdate := func(partialResult ToolResult) {
		if !acceptingUpdates.Load() {
			return
		}
		partial := partialResult
		emit(Event{
			Type:          EventToolExecutionUpdate,
			ToolCallID:    prepared.toolCall.ID,
			ToolName:      prepared.toolCall.Name,
			Args:          prepared.toolCall.Arguments,
			PartialResult: &partial,
		})
	}

	acceptingUpdates.Store(true)
	defer func() {
		acceptingUpdates.Store(false)
		if r := recover(); r != nil {
			outcome = executedOutcome{result: createErrorToolResult(fmt.Sprint(r)), isError: true}
		}
	}()

	if prepared.tool.Execute == nil {
		return executedOutcome{result: createErrorToolResult(fmt.Sprintf("Tool %s has no Execute function", prepared.toolCall.Name)), isError: true}
	}
	result, err := prepared.tool.Execute(ctx, prepared.toolCall.ID, prepared.args, onUpdate)
	acceptingUpdates.Store(false)
	if err != nil {
		return executedOutcome{result: createErrorToolResult(err.Error()), isError: true}
	}
	return executedOutcome{result: result, isError: false}
}

// finalizeExecutedToolCall applies the afterToolCall hook (TS
// finalizeExecutedToolCall). Hook overrides merge field-by-field; hook panics
// become error results.
func finalizeExecutedToolCall(ctx context.Context, currentContext *Context, assistantMessage llm.AssistantMessage, prepared *preparedToolCall, executed executedOutcome, config *loopConfig) (finalized finalizedOutcome) {
	defer func() {
		if r := recover(); r != nil {
			finalized = finalizedOutcome{
				toolCall: prepared.toolCall,
				result:   createErrorToolResult(fmt.Sprint(r)),
				isError:  true,
			}
		}
	}()

	result := executed.result
	isError := executed.isError

	if config.afterToolCall != nil {
		afterResult := config.afterToolCall(ctx, AfterToolCallContext{
			AssistantMessage: assistantMessage,
			ToolCall:         prepared.toolCall,
			Args:             prepared.args,
			Result:           result,
			IsError:          isError,
			Context:          *currentContext,
		})
		if afterResult != nil {
			if afterResult.Content != nil {
				result.Content = afterResult.Content
			}
			if afterResult.Details != nil {
				result.Details = afterResult.Details
			}
			if afterResult.Usage != nil {
				result.Usage = afterResult.Usage
			}
			if afterResult.Terminate != nil {
				result.Terminate = *afterResult.Terminate
			}
			if afterResult.IsError != nil {
				isError = *afterResult.IsError
			}
		}
	}

	return finalizedOutcome{toolCall: prepared.toolCall, result: result, isError: isError}
}

// createErrorToolResult builds an error tool result (TS createErrorToolResult).
func createErrorToolResult(message string) ToolResult {
	return ToolResult{
		Content: []llm.ContentBlock{llm.TextContent{Type: "text", Text: message}},
		Details: map[string]any{},
	}
}

func emitToolExecutionEnd(finalized finalizedOutcome, emit func(Event)) {
	result := finalized.result
	emit(Event{
		Type:       EventToolExecutionEnd,
		ToolCallID: finalized.toolCall.ID,
		ToolName:   finalized.toolCall.Name,
		Result:     &result,
		IsError:    finalized.isError,
	})
}

// createToolResultMessage builds the tool-result transcript message (TS
// createToolResultMessage). Results without content are normalized to an
// empty slice so nil never enters session history or provider payloads.
func createToolResultMessage(finalized finalizedOutcome) llm.ToolResultMessage {
	content := finalized.result.Content
	if content == nil {
		content = []llm.ContentBlock{}
	}
	return llm.ToolResultMessage{
		Role:           "toolResult",
		ToolCallID:     finalized.toolCall.ID,
		ToolName:       finalized.toolCall.Name,
		Content:        content,
		Details:        finalized.result.Details,
		Usage:          finalized.result.Usage,
		AddedToolNames: finalized.result.AddedToolNames,
		IsError:        finalized.isError,
		Timestamp:      nowMillis(),
	}
}

func emitToolResultMessage(toolResultMessage llm.ToolResultMessage, emit func(Event)) {
	emit(Event{Type: EventMessageStart, Message: toolResultMessage})
	emit(Event{Type: EventMessageEnd, Message: toolResultMessage})
}
