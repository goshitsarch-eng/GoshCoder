package main

// Branching: rewinding to an earlier point in a session and taking a different
// path, plus forking and labelling.
//
// The tree has been in the file since the first entry -- every entry carries a
// parentId, and appends attach to the current leaf. This is the interface to
// it: moving the write head backwards is all a branch is.

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"goshcoder/internal/agent"
	"goshcoder/internal/llm"
	"goshcoder/internal/sessionlog"
)

// branchPoint is one place a user can rewind to.
type branchPoint struct {
	Index int
	ID    string
	Role  string
	Text  string
	Label string
	// Children counts the branches that already leave this point, so an entry
	// that has been rewound to before is visibly different from a fresh one.
	Children int
	Current  bool
}

// branchPoints lists the user messages on the current path.
//
// Only user messages: rewinding to the middle of an assistant turn would leave
// a tool call with no result, which every provider rejects on the next request.
func branchPoints(tree *sessionlog.Tree, leaf *string) []branchPoint {
	if tree == nil {
		return nil
	}
	path := tree.Path(leaf)

	var points []branchPoint
	for _, entry := range path {
		if entry.Type != sessionlog.TypeMessage {
			continue
		}
		role := entryRole(entry)
		if role != "user" {
			continue
		}
		point := branchPoint{
			ID:       entry.ID,
			Role:     role,
			Text:     firstLine(entryText(entry)),
			Children: len(tree.Children(&entry.ID)),
		}
		if label, ok := tree.Label(entry.ID); ok {
			point.Label = label
		}
		points = append(points, point)
	}
	for index := range points {
		points[index].Index = index + 1
	}
	// The marker sits on the newest point on the current path, which is where
	// the conversation actually stands. It cannot be the write head itself:
	// the leaf is almost always an assistant message, and those are not
	// rewind targets.
	if len(points) > 0 {
		points[len(points)-1].Current = true
	}
	return points
}

// entryRole reads a message entry's role without decoding the whole payload.
func entryRole(entry sessionlog.Entry) string {
	var shaped struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(entry.Message, &shaped); err != nil {
		return ""
	}
	return shaped.Role
}

// entryText renders a message entry's text for a listing.
func entryText(entry sessionlog.Entry) string {
	messages, err := decodeEntryMessages(entry)
	if err != nil || len(messages) == 0 {
		return ""
	}
	return summarizeMessage(messages[0])
}

// treeLines renders /tree.
func treeLines(tree *sessionlog.Tree, leaf *string) []string {
	points := branchPoints(tree, leaf)
	if len(points) == 0 {
		return []string{"nothing to rewind to yet"}
	}
	lines := make([]string, 0, len(points)+1)
	for _, point := range points {
		marker := " "
		if point.Current {
			marker = "*"
		}
		line := fmt.Sprintf("%s %2d  %s", marker, point.Index, point.Text)
		if point.Label != "" {
			line += "  [" + point.Label + "]"
		}
		if point.Children > 1 {
			line += fmt.Sprintf("  (%d branches)", point.Children)
		}
		lines = append(lines, line)
	}
	lines = append(lines, "/fork <n> rewinds to a point · /label <n> <text> names one")
	return lines
}

// forkTo rewinds the session to a branch point and continues from there.
//
// The entries after that point stay in the file; they simply stop being on the
// path. That is what makes a branch cheap and reversible: nothing is deleted,
// and the abandoned path is still reachable through /tree.
func (s *session) forkTo(argument string) error {
	if s == nil || s.log == nil || !s.log.active() {
		return errors.New("this session is not being saved, so it cannot branch")
	}
	// Reset and Compact refuse while a run is active; SetMessages does not, so
	// rewinding mid-turn would strand the run's own message list against a
	// transcript it no longer describes.
	if s.agent.State().IsStreaming {
		return errors.New("wait for the current response to finish before rewinding")
	}
	tree := s.log.snapshot()
	points := branchPoints(tree, s.log.leaf())
	if len(points) == 0 {
		return errors.New("there is nothing to rewind to yet")
	}

	index, err := strconv.Atoi(strings.TrimSpace(argument))
	if err != nil || index < 1 || index > len(points) {
		return fmt.Errorf("choose a point between 1 and %d; /tree lists them", len(points))
	}
	target := points[index-1]

	abandoned := len(tree.Path(s.log.leaf())) - len(tree.Path(&target.ID))

	// The write head moves FIRST. Appending the summary before the rewind
	// attached it to the abandoned tip, which then became the leaf again --
	// silently undoing the rewind on the next resume, since the leaf is what a
	// reopened session continues from.
	if err := s.log.setLeaf(target.ID); err != nil {
		return err
	}
	// A branch summary records what was left behind, so an old branch is
	// identifiable in /tree rather than being an unlabelled fork.
	if abandoned > 0 {
		s.log.append(sessionlog.Entry{
			Type:    sessionlog.TypeBranchSummary,
			FromID:  target.ID,
			Summary: fmt.Sprintf("branched from %q, leaving %d entries on the previous path", target.Text, abandoned),
		})
	}

	// The transcript has to follow the write head, or the next turn is sent
	// with a context the log no longer describes.
	projected := projectSession(s.log.snapshot(), s.log.leaf())
	s.agent.SetMessages(projected.Messages)
	s.contextEstimate.reset()
	s.restoredMessages = projected.Messages
	s.log.sync()

	fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
		"rewound to %q; %d message(s) in context, %d left on the other branch",
		target.Text, len(projected.Messages), abandoned)))
	return nil
}

// labelPoint names a branch point so it can be recognized later.
func (s *session) labelPoint(argument string) error {
	if s == nil || s.log == nil || !s.log.active() {
		return errors.New("this session is not being saved, so it cannot be labelled")
	}
	field, text, _ := strings.Cut(strings.TrimSpace(argument), " ")
	text = strings.TrimSpace(text)
	if field == "" {
		return errors.New("/label needs a point number, e.g. /label 3 before the refactor")
	}
	points := branchPoints(s.log.snapshot(), s.log.leaf())
	index, err := strconv.Atoi(field)
	if err != nil || index < 1 || index > len(points) {
		return fmt.Errorf("choose a point between 1 and %d; /tree lists them", len(points))
	}

	target := points[index-1]
	entry := sessionlog.Entry{Type: sessionlog.TypeLabel, TargetID: target.ID}
	if text != "" {
		entry.Label = &text
	}
	s.log.append(entry)
	s.log.sync()
	if text == "" {
		fmt.Fprintf(os.Stderr, "%s\n", dim("removed the label on point "+field))
		return nil
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("labelled point "+field+" as "+text))
	return nil
}

// cloneSession duplicates the current session into a new one and switches to
// it, leaving the original untouched.
func (s *session) cloneSession() error {
	if s == nil || s.log == nil || !s.log.active() {
		return errors.New("this session is not being saved, so it cannot be cloned")
	}
	if s.agent.State().IsStreaming {
		return errors.New("wait for the current response to finish before cloning")
	}
	store := defaultStore()
	info, err := store.Resolve(s.workspaceRoot(), s.log.id())
	if err != nil {
		return err
	}
	leaf := s.log.leaf()
	// Sync first: Fork reads the file, so anything still buffered would be
	// missing from the copy.
	s.log.sync()

	writer, err := store.Fork(info, leaf, s.workspaceRoot())
	if err != nil {
		return err
	}
	clonedID := writer.ID()
	if err := writer.Close(); err != nil {
		return err
	}
	if err := s.switchSession(clonedID); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("cloned into "+shortSessionID(clonedID)+"; the original is unchanged"))
	return nil
}

// leaf exposes the writer's current write head.
func (r *sessionRecorder) leaf() *string {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.writer == nil {
		return nil
	}
	return r.writer.Leaf()
}

// setLeaf moves the write head so the next append starts a branch.
func (r *sessionRecorder) setLeaf(id string) error {
	if r == nil {
		return errors.New("this session is not being saved")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.writer == nil {
		return errors.New("this session is not being saved")
	}
	return r.writer.SetLeaf(id)
}

// decodeEntryMessages decodes one message entry into agent messages.
func decodeEntryMessages(entry sessionlog.Entry) ([]agent.Message, error) {
	decoded, err := llm.UnmarshalMessages(json.RawMessage("[" + string(entry.Message) + "]"))
	if err != nil {
		return nil, err
	}
	messages := make([]agent.Message, 0, len(decoded))
	for _, message := range decoded {
		messages = append(messages, message)
	}
	return messages, nil
}
