package main

// The session recorder: an agent.Listener that appends the conversation to a
// sessionlog file, and the projection that turns a loaded session back into
// the agent's message list.
//
// It lives in package main rather than in internal/sessionlog because
// compactionSummaryMessage is unexported here, and the whole point of carrying
// cost and retained-message counts through the log is that this exact type
// comes back on resume -- conversationCost, maybeAutoCompact and the sidebar
// all dispatch on the Go type, not on a field.

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/sessionlog"
)

// goshcoderDetails is what rides in a compaction entry's `details` slot, which
// pi types as unknown and ignores. It carries the two numbers pi's own
// CompactionEntry has no field for.
type goshcoderDetails struct {
	GoshCoder struct {
		CostBefore       float64 `json:"costBefore"`
		RetainedMessages int     `json:"retainedMessages"`
		Timestamp        int64   `json:"timestamp,omitempty"`
	} `json:"goshcoder"`
}

// sessionRecorder appends agent events to a session file.
//
// A write failure does not abort the run: the user is mid-task and the
// transcript is intact in memory. Recording stops -- retrying every turn
// against a full disk is just noise -- and the interface reports it once,
// which is also what re-arms the Ctrl+C confirmation.
type sessionRecorder struct {
	mu       sync.Mutex
	writer   *sessionlog.Writer
	notify   func(kind, text string)
	reported bool
}

func newSessionRecorder(writer *sessionlog.Writer, notify func(kind, text string)) *sessionRecorder {
	return &sessionRecorder{writer: writer, notify: notify}
}

// swap replaces the file this recorder writes to and returns the previous
// writer for the caller to close.
//
// The recorder object itself must outlive every swap. Agent.Subscribe binds a
// listener to the receiver it was created from, so replacing session.log with a
// freshly constructed recorder left every event going to the old, closed one --
// /resume and /clone silently recorded nothing from the moment they ran.
func (r *sessionRecorder) swap(writer *sessionlog.Writer) *sessionlog.Writer {
	r.mu.Lock()
	defer r.mu.Unlock()
	previous := r.writer
	r.writer = writer
	// A new file gets a fresh chance to report a failure of its own.
	r.reported = false
	return previous
}

// readOnly reports whether the current file was opened without a claim.
func (r *sessionRecorder) readOnly() bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.writer != nil && r.writer.ReadOnly()
}

// active reports whether appends are reaching disk.
func (r *sessionRecorder) active() bool {
	if r == nil {
		return false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.writer != nil && r.writer.Recording()
}

func (r *sessionRecorder) path() string {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.writer == nil {
		return ""
	}
	return r.writer.Path()
}

func (r *sessionRecorder) id() string {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.writer == nil {
		return ""
	}
	return r.writer.ID()
}

func (r *sessionRecorder) snapshot() *sessionlog.Tree {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.writer == nil {
		return nil
	}
	return r.writer.Snapshot()
}

// append writes one entry, reporting the first failure and then going quiet.
func (r *sessionRecorder) append(entry sessionlog.Entry) {
	if r == nil {
		return
	}
	r.mu.Lock()
	writer := r.writer
	r.mu.Unlock()
	if writer == nil || !writer.Recording() {
		return
	}
	if _, err := writer.Append(entry); err != nil {
		r.report(err)
	}
}

func (r *sessionRecorder) report(err error) {
	r.mu.Lock()
	if r.reported {
		r.mu.Unlock()
		return
	}
	r.reported = true
	notify := r.notify
	r.mu.Unlock()
	if notify != nil {
		notify("Session", "recording stopped: "+err.Error()+" (the conversation is still in memory)")
	}
}

// sync flushes at a turn boundary. A completed turn is the unit a user would
// be upset to lose; fsyncing per entry would pay the cost per tool call.
func (r *sessionRecorder) sync() {
	if r == nil {
		return
	}
	r.mu.Lock()
	writer := r.writer
	r.mu.Unlock()
	if writer == nil {
		return
	}
	if err := writer.Sync(); err != nil {
		r.report(err)
	}
}

func (r *sessionRecorder) close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	writer := r.writer
	r.writer = nil
	r.mu.Unlock()
	if writer == nil {
		return nil
	}
	return writer.Close()
}

// setName records a display name for the session.
func (r *sessionRecorder) setName(name string) {
	r.append(sessionlog.Entry{Type: sessionlog.TypeSessionInfo, Name: name})
}

// listener returns the agent subscription. It is registered unconditionally
// and before any renderer, so an entry is durable before anything draws it.
func (r *sessionRecorder) listener() agent.Listener {
	return func(_ context.Context, ev agent.Event) {
		if r == nil {
			return
		}
		switch ev.Type {
		case agent.EventMessageEnd:
			r.appendMessage(ev.Message)
		case agent.EventModelChange:
			r.append(sessionlog.Entry{
				Type:     sessionlog.TypeModelChange,
				Provider: string(ev.Provider),
				ModelID:  ev.ModelID,
			})
		case agent.EventThinkingLevelChange:
			r.append(sessionlog.Entry{
				Type:          sessionlog.TypeThinkingLevel,
				ThinkingLevel: string(ev.ThinkingLevel),
			})
		case agent.EventContextCompacted:
			r.appendCompaction(ev)
		case agent.EventTranscriptReset:
			reason := ev.Reason
			if reason == "" {
				reason = "reset"
			}
			r.append(sessionlog.Entry{Type: sessionlog.TypeTranscriptReset, Reason: reason})
		case agent.EventAgentEnd:
			r.sync()
		}
	}
}

// appendMessage records one conversation message.
//
// The user's own prompt reaches the agent through Prompt rather than through
// an event, so message_end alone would record only the model's half. The
// session wrapper calls recordUserMessage for the other half.
func (r *sessionRecorder) appendMessage(message agent.Message) {
	if message == nil {
		return
	}
	if _, ok := message.(compactionSummaryMessage); ok {
		// The marker is written by appendCompaction with its full detail; it
		// must not also arrive here as a plain message.
		return
	}
	raw, err := json.Marshal(message)
	if err != nil {
		r.report(fmt.Errorf("encode %T for the session log: %w", message, err))
		return
	}
	r.append(sessionlog.Entry{Type: sessionlog.TypeMessage, Message: raw})
}

// appendCompaction writes the compaction marker followed by the retained
// messages, which is exactly the cut the agent just made in memory.
func (r *sessionRecorder) appendCompaction(ev agent.Event) {
	info := ev.Compaction
	if info == nil {
		return
	}
	// firstKeptEntryId has to name an entry that already exists, so the kept
	// messages are appended first and the marker points back at the first of
	// them. pi reads the same shape.
	var firstKept string
	for index, message := range ev.Kept {
		raw, err := json.Marshal(message)
		if err != nil {
			r.report(fmt.Errorf("encode a retained message for the session log: %w", err))
			return
		}
		id := r.appendReturningID(sessionlog.Entry{Type: sessionlog.TypeMessage, Message: raw})
		if index == 0 {
			firstKept = id
		}
	}

	var details goshcoderDetails
	details.GoshCoder.CostBefore = info.CostBefore
	details.GoshCoder.RetainedMessages = info.RetainedMessages
	details.GoshCoder.Timestamp = info.Timestamp
	encoded, err := json.Marshal(details)
	if err != nil {
		r.report(fmt.Errorf("encode compaction details: %w", err))
		return
	}
	r.append(sessionlog.Entry{
		Type:             sessionlog.TypeCompaction,
		Summary:          info.Summary,
		FirstKeptEntryID: firstKept,
		TokensBefore:     info.TokensBefore,
		Details:          encoded,
	})
	r.sync()
}

func (r *sessionRecorder) appendReturningID(entry sessionlog.Entry) string {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	writer := r.writer
	r.mu.Unlock()
	if writer == nil || !writer.Recording() {
		return ""
	}
	id, err := writer.Append(entry)
	if err != nil {
		r.report(err)
		return ""
	}
	return id
}

// recordCustom persists extension state -- the Planner's phase, for instance --
// as a custom entry. pi documents this entry type as exactly that: state that
// survives a reload and never participates in context.
func (r *sessionRecorder) recordCustom(customType string, payload any) {
	raw, err := json.Marshal(payload)
	if err != nil {
		r.report(fmt.Errorf("encode %s state for the session log: %w", customType, err))
		return
	}
	r.append(sessionlog.Entry{Type: sessionlog.TypeCustom, CustomType: customType, Data: raw})
}

// ---------------------------------------------------------------------------
// projection

// restored is what a session file folds back into.
type restored struct {
	Messages      []agent.Message
	Provider      string
	ModelID       string
	ThinkingLevel string
	// Custom holds the latest payload for each customType, so an extension can
	// reconstruct its state without walking the tree itself.
	Custom map[string]json.RawMessage
	// Warnings are load problems worth telling the user about.
	Warnings []string
}

// projectSession folds a session tree into the agent's message list.
//
// It walks ContextPath, which already honors compaction and /clear, so entries
// before the latest cut stay in the file and remain browsable but are never
// sent to the model again. That reproduces exactly what compactContext
// installs in memory.
func projectSession(tree *sessionlog.Tree, leaf *string) restored {
	out := restored{Custom: map[string]json.RawMessage{}}
	if tree == nil {
		return out
	}

	// Model and thinking come from the whole path, not the context path: a
	// model chosen before a compaction is still the model in use.
	for _, entry := range tree.Path(leaf) {
		switch entry.Type {
		case sessionlog.TypeModelChange:
			out.Provider, out.ModelID = entry.Provider, entry.ModelID
		case sessionlog.TypeThinkingLevel:
			out.ThinkingLevel = entry.ThinkingLevel
		case sessionlog.TypeCustom:
			if entry.CustomType != "" {
				out.Custom[entry.CustomType] = entry.Data
			}
		case sessionlog.TypeMessage:
			// pi's fallback: an older session may carry no model_change entry
			// at all, in which case the last assistant message names the model
			// the conversation was actually held with.
			if provider, model, ok := assistantModel(entry.Message); ok {
				if out.Provider == "" || out.ModelID == "" {
					out.Provider, out.ModelID = provider, model
				}
			}
		}
	}
	// A model_change entry always wins over the fallback above.
	for _, entry := range tree.Path(leaf) {
		if entry.Type == sessionlog.TypeModelChange {
			out.Provider, out.ModelID = entry.Provider, entry.ModelID
		}
	}

	for _, entry := range tree.ContextPath(leaf) {
		switch entry.Type {
		case sessionlog.TypeMessage:
			decoded, err := llm.UnmarshalMessages(json.RawMessage("[" + string(entry.Message) + "]"))
			if err != nil {
				out.Warnings = append(out.Warnings,
					fmt.Sprintf("entry %s could not be read back and was skipped: %v", entry.ID, err))
				continue
			}
			for _, message := range decoded {
				out.Messages = append(out.Messages, message)
			}
		case sessionlog.TypeCompaction:
			out.Messages = append(out.Messages, compactionMarkerFrom(entry))
		}
	}
	return out
}

// compactionMarkerFrom rebuilds the in-memory marker from a compaction entry.
//
// Reconstructing the Go type matters more than it looks: conversationCost,
// maybeAutoCompact and the sidebar's context reporting all switch on
// compactionSummaryMessage. A resume that produced a plain text message
// instead would double-count every retained assistant cost and misreport
// context usage, silently, forever after.
func compactionMarkerFrom(entry sessionlog.Entry) compactionSummaryMessage {
	marker := compactionSummaryMessage{
		Summary:      entry.Summary,
		TokensBefore: entry.TokensBefore,
	}
	if len(entry.Details) > 0 {
		var details goshcoderDetails
		if err := json.Unmarshal(entry.Details, &details); err == nil {
			marker.CostBefore = details.GoshCoder.CostBefore
			marker.RetainedMessages = details.GoshCoder.RetainedMessages
			marker.Timestamp = details.GoshCoder.Timestamp
		}
	}
	if marker.Timestamp == 0 {
		marker.Timestamp = sessionlog.ParseTimestamp(entry.Timestamp).UnixMilli()
	}
	return marker
}

// assistantModel reads the provider and model off an assistant message payload
// without decoding the whole thing.
func assistantModel(raw json.RawMessage) (string, string, bool) {
	if len(raw) == 0 {
		return "", "", false
	}
	var shaped struct {
		Role     string `json:"role"`
		Provider string `json:"provider"`
		Model    string `json:"model"`
	}
	if err := json.Unmarshal(raw, &shaped); err != nil || shaped.Role != "assistant" {
		return "", "", false
	}
	if shaped.Provider == "" || shaped.Model == "" {
		return "", "", false
	}
	return shaped.Provider, shaped.Model, true
}
