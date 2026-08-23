// Package sessionlog is a Go port of pi's v3 session file format
// (reference/pi/packages/coding-agent/src/core/session-manager.ts). Files
// written here are interchangeable with pi's: same directory sharding, same
// filename scheme, same header, same entry types.
//
// Why v3 and not the v4 mutation log under packages/agent/src/harness/session:
// pi's shipping CLI never imports v4 -- CURRENT_SESSION_VERSION is 3, and
// nothing outside packages/agent references JsonlSessionRepo. v3's reader is
// also lenient exactly where forward compatibility needs it. It skips a
// malformed line and indexes an entry type it has never heard of, so a file
// written by a newer build still loads in an older one; v4's codec treats both
// as fatal, which would make every added entry type a breaking change.
//
// A session file is one header line followed by one entry per line. Entries
// form a tree through parentId, which is what makes branching a UI project
// rather than a format change.
//
// Deviations from pi, each with its reason at the site that implements it:
// appends happen from the first entry rather than being buffered until the
// first assistant message (writer.go); a writer takes an exclusive claim
// (writer.go); writes are fsynced (writer.go); skipped lines are reported
// rather than silently dropped (this file, and store.go); and Fork copies raw
// lines rather than re-encoding them (store.go).
package sessionlog

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// FormatVersion is the session format this package reads and writes. A file
// declaring a higher version is refused rather than partially understood.
const FormatVersion = 3

// Entry types. Unknown types are preserved and indexed rather than rejected;
// see Tree.Add.
const (
	TypeMessage       = "message"
	TypeModelChange   = "model_change"
	TypeThinkingLevel = "thinking_level_change"
	TypeCompaction    = "compaction"
	TypeBranchSummary = "branch_summary"
	TypeCustom        = "custom"
	TypeCustomMessage = "custom_message"
	TypeLabel         = "label"
	TypeSessionInfo   = "session_info"
	// TypeTranscriptReset marks a /clear. It is a GoshCoder addition rather
	// than a pi entry type, which v3 tolerates: pi indexes an unknown type and
	// projects nothing for it. The interop consequence is stated plainly --
	// pi reading a GoshCoder session that was cleared will still replay the
	// pre-clear messages, because only this build knows the marker cuts the
	// context. Nothing is lost or corrupted either way.
	TypeTranscriptReset   = "transcript_reset"
	headerType            = "session"
	timestampLayout       = "2006-01-02T15:04:05.000Z"
	filenameStampReplacer = "-"
)

// Header is a session file's first line.
type Header struct {
	Type    string `json:"type"`
	Version int    `json:"version"`
	ID      string `json:"id"`
	// Timestamp is ISO-8601 with milliseconds in UTC. Note the asymmetry with
	// the timestamp inside a message payload, which is Unix milliseconds --
	// that is pi's, and changing either breaks interchange.
	Timestamp     string `json:"timestamp"`
	Cwd           string `json:"cwd"`
	ParentSession string `json:"parentSession,omitempty"`
}

// Entry is the union of pi's v3 entry types, flattened into one struct. Fields
// that are unset are omitted, so an encoded entry is byte-comparable with pi's
// for the same input.
//
// ParentID deliberately carries no omitempty: pi requires the key to be
// present and null at the root of the tree.
type Entry struct {
	Type      string  `json:"type"`
	ID        string  `json:"id"`
	ParentID  *string `json:"parentId"`
	Timestamp string  `json:"timestamp"`

	// message
	Message json.RawMessage `json:"message,omitempty"`

	// compaction, branch_summary
	Summary          string          `json:"summary,omitempty"`
	FirstKeptEntryID string          `json:"firstKeptEntryId,omitempty"`
	TokensBefore     int             `json:"tokensBefore,omitempty"`
	FromID           string          `json:"fromId,omitempty"`
	Details          json.RawMessage `json:"details,omitempty"`

	// model_change
	Provider string `json:"provider,omitempty"`
	ModelID  string `json:"modelId,omitempty"`

	// thinking_level_change
	ThinkingLevel string `json:"thinkingLevel,omitempty"`

	// custom, custom_message
	CustomType string          `json:"customType,omitempty"`
	Data       json.RawMessage `json:"data,omitempty"`
	Content    json.RawMessage `json:"content,omitempty"`
	Display    *bool           `json:"display,omitempty"`

	// label
	TargetID string  `json:"targetId,omitempty"`
	Label    *string `json:"label,omitempty"`

	// session_info
	Name string `json:"name,omitempty"`

	// transcript_reset
	Reason string `json:"reason,omitempty"`
}

// DecodeError describes a line this build could not use. It is reported rather
// than fatal, matching pi, so one corrupt line never costs a whole session.
type DecodeError struct {
	Line   int
	Reason string
}

func (e DecodeError) Error() string {
	return fmt.Sprintf("line %d %s", e.Line, e.Reason)
}

// ErrVersionTooNew is returned when a file declares a format this build does
// not understand. Loading it anyway would silently drop entries the newer
// writer considered essential.
var ErrVersionTooNew = errors.New("sessionlog: session format is newer than this build")

// Now returns the timestamp format used by entry and header lines.
func Now() string { return time.Now().UTC().Format(timestampLayout) }

// ParseTimestamp reads an entry timestamp. A malformed one yields the zero
// time rather than an error: timestamps drive display and sorting, and a bad
// one must not make a session unopenable.
func ParseTimestamp(value string) time.Time {
	for _, layout := range []string{timestampLayout, time.RFC3339Nano, time.RFC3339} {
		if parsed, err := time.Parse(layout, value); err == nil {
			return parsed
		}
	}
	return time.Time{}
}

// FilenameStamp converts an entry timestamp into the filename form pi uses:
// colons and dots become dashes, so the name is valid on Windows.
func FilenameStamp(timestamp string) string {
	return strings.NewReplacer(":", filenameStampReplacer, ".", filenameStampReplacer).Replace(timestamp)
}

// EncodeHeader returns the header line including its newline.
func EncodeHeader(h Header) ([]byte, error) {
	h.Type = headerType
	if h.Version == 0 {
		h.Version = FormatVersion
	}
	encoded, err := json.Marshal(h)
	if err != nil {
		return nil, fmt.Errorf("sessionlog: encode header: %w", err)
	}
	return append(encoded, '\n'), nil
}

// DecodeHeader parses a session file's first line.
func DecodeHeader(line []byte) (Header, error) {
	var h Header
	if err := json.Unmarshal(line, &h); err != nil {
		return Header{}, DecodeError{Line: 1, Reason: "is not valid JSON"}
	}
	if h.Type != headerType {
		return Header{}, DecodeError{Line: 1, Reason: "is not a session header"}
	}
	if h.ID == "" {
		return Header{}, DecodeError{Line: 1, Reason: "has no session id"}
	}
	// A missing version is v1: the format predates the field. A version above
	// ours is refused outright rather than partially understood, since entries
	// the newer writer considered essential would be silently dropped.
	if h.Version == 0 {
		h.Version = 1
	}
	if h.Version > FormatVersion {
		return Header{}, ErrVersionTooNew
	}
	return h, nil
}

// EncodeEntry returns the entry line including its newline.
func EncodeEntry(e Entry) ([]byte, error) {
	encoded, err := json.Marshal(e)
	if err != nil {
		return nil, fmt.Errorf("sessionlog: encode entry: %w", err)
	}
	return append(encoded, '\n'), nil
}

// DecodeEntry parses one entry line. An entry whose type this build does not
// know is returned intact: it still joins the tree, and projection ignores it.
//
// requireID is false only while migrating a pre-v3 file, where identity came
// from file order rather than from a field; see migrate.go.
func DecodeEntry(line []byte) (Entry, error) { return decodeEntry(line, true) }

func decodeEntry(line []byte, requireID bool) (Entry, error) {
	var e Entry
	if err := json.Unmarshal(line, &e); err != nil {
		return Entry{}, errors.New("is not valid JSON")
	}
	if e.Type == "" {
		return Entry{}, errors.New("has no entry type")
	}
	if e.Type == headerType {
		return Entry{}, errors.New("is a second session header")
	}
	if requireID && e.ID == "" {
		return Entry{}, errors.New("has no entry id")
	}
	return e, nil
}
