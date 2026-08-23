package main

// The resume picker: choosing a session at startup with -resume, and switching
// to one mid-session with /resume.

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"goshcoder/internal/sessionlog"
)

// pickerLimit caps how many sessions a picker offers. Beyond this the search
// box is the tool, not the list.
const pickerLimit = 50

// listSessionsForPicker gathers the candidates for a picker.
//
// Transcript text is loaded, not just metadata. Nobody remembers a session by
// its first message; they remember "the one where we fixed the SSE reader". A
// metadata-only search is useful for "yesterday's" and useless for "that one
// from October". sessionlog bounds the text it keeps per session so this stays
// affordable.
func listSessionsForPicker(workdir string, allWorkspaces bool) ([]sessionlog.Info, error) {
	store := activeStore()
	sessions, err := store.List(workdir, sessionlog.ListOptions{
		AllWorkspaces: allWorkspaces,
		Limit:         pickerLimit,
		WithText:      true,
	})
	if err != nil {
		return nil, err
	}
	if len(sessions) > 0 || allWorkspaces {
		return sessions, nil
	}
	// A session started at the repository root is not in this subdirectory's
	// shard. Looking there is what makes `chat -resume` work from anywhere in a
	// checkout rather than only from where the session began.
	if root := repositoryRoot(workdir); root != "" && root != workdir {
		return store.List(root, sessionlog.ListOptions{Limit: pickerLimit, WithText: true})
	}
	return nil, nil
}

// repositoryRoot returns the enclosing git worktree root, or "".
func repositoryRoot(dir string) string {
	// One bounded subprocess, matching the sidebar's git inspection: an
	// unbounded one on a network filesystem would hang startup.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	command := exec.CommandContext(ctx, "git", "-C", dir, "rev-parse", "--show-toplevel")
	output, err := command.Output()
	if err != nil {
		return ""
	}
	root := strings.TrimSpace(string(output))
	if root == "" {
		return ""
	}
	// Session shards are keyed by the resolved path, so a symlinked checkout
	// must be resolved the same way here or the shard is never found.
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		return resolved
	}
	return root
}

// matchesSession reports whether every search term appears somewhere in the
// session's identity or its transcript.
//
// Multi-term matching, copied from the /model picker: it is the only existing
// matcher that tolerates whitespace, and a search box that breaks on a space is
// worse than no search box.
func matchesSession(info sessionlog.Info, terms []string) bool {
	haystack := strings.ToLower(strings.Join([]string{
		info.ID, info.Name, info.FirstMessage, info.Cwd, info.SearchText,
	}, " "))
	for _, term := range terms {
		if !strings.Contains(haystack, term) {
			return false
		}
	}
	return true
}

// describeSession renders one picker row. label is the id prefix to show,
// which the caller widens with sessionlog.ShortIDs so the rows stay
// distinguishable.
func describeSession(info sessionlog.Info, showCwd bool) (label, description string) {
	return describeSessionAs(info, info.ShortID(), showCwd)
}

func describeSessionAs(info sessionlog.Info, label string, showCwd bool) (string, string) {
	parts := []string{info.Modified.Local().Format("Jan 2 15:04")}
	if info.Messages > 0 {
		parts = append(parts, fmt.Sprintf("%d msg", info.Messages))
	}
	if info.Locked {
		parts = append(parts, "open elsewhere")
	}
	if showCwd {
		parts = append(parts, filepath.Base(info.Cwd))
	}
	title := info.Title()
	if title != "" && title != info.ID {
		parts = append(parts, title)
	}
	return label, strings.Join(parts, " · ")
}

// chooseSessionInteractively runs the startup picker for -resume.
//
// It runs before the fullscreen program starts, so it uses the same
// line-oriented selection the model onboarding uses rather than a Bubble Tea
// view of its own.
func chooseSessionInteractively(workdir string) (sessionlog.Info, bool, error) {
	sessions, err := listSessionsForPicker(workdir, false)
	if err != nil {
		return sessionlog.Info{}, false, err
	}
	if len(sessions) == 0 {
		fmt.Fprintln(os.Stderr, dim("no saved sessions for this workspace; starting a new one"))
		return sessionlog.Info{}, false, nil
	}

	fmt.Fprintln(os.Stderr, dim("saved sessions for this workspace:"))
	labels := sessionlog.ShortIDs(sessions)
	for index, info := range sessions {
		label, description := describeSessionAs(info, labels[index], false)
		fmt.Fprintf(os.Stderr, "%3d  %s  %s\n", index+1, label, dim(description))
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("choose a number, or press Enter to start a new session:"))

	choice, err := readLineFromStdin()
	if err != nil {
		return sessionlog.Info{}, false, err
	}
	choice = strings.TrimSpace(choice)
	if choice == "" {
		return sessionlog.Info{}, false, nil
	}
	index, err := parsePositiveIndex(choice, len(sessions))
	if err != nil {
		return sessionlog.Info{}, false, err
	}
	return sessions[index-1], true, nil
}

// switchSession swaps the live session for another one mid-conversation.
//
// The target is attached BEFORE the current file is closed. Closing first left
// every failure path -- a claim held elsewhere, a pre-v3 file, an I/O error --
// with no recorder at all: the window kept running and silently recorded
// nothing for the rest of its life.
//
// The recorder object is reused rather than replaced. Agent.Subscribe binds its
// listener to the recorder it was given, so installing a new one would leave
// every event going to the old, closed recorder.
func (s *session) switchSession(ref string) error {
	if s == nil {
		return errors.New("no session")
	}
	if s.log == nil {
		return errors.New("this session is not being saved, so it cannot be switched")
	}
	if s.agent.State().IsStreaming {
		return errors.New("wait for the current response to finish before switching sessions")
	}
	store := s.sessionStore()
	workdir := s.workspaceRoot()
	info, err := store.Resolve(workdir, ref)
	if err != nil {
		return err
	}
	if info.Path == s.log.path() {
		return errors.New("that is already the current session")
	}

	writer, report, err := store.Attach(info.Path)
	if err != nil {
		if errors.Is(err, sessionlog.ErrBusy) {
			return fmt.Errorf("%s is open in another window (%s)", info.ShortID(), info.Owner)
		}
		if errors.Is(err, sessionlog.ErrLegacyFormat) {
			return fmt.Errorf("%s is an older pi session; start it with -session %s to copy it into a continuable one",
				info.ShortID(), info.ShortID())
		}
		return err
	}

	projected := projectSession(writer.Snapshot(), writer.Leaf())
	// Seeding the transcript before the swap keeps the agent and the log
	// describing the same conversation at every instant.
	s.agent.SetMessages(projected.Messages)
	s.contextEstimate.reset()

	if previous := s.log.swap(writer); previous != nil {
		if closeErr := previous.Close(); closeErr != nil {
			s.pushNotice("Session", "the previous session did not close cleanly: "+closeErr.Error())
		}
	}

	for _, notice := range reportNotices(report) {
		s.pushNotice("Session", notice)
	}
	for _, warning := range projected.Warnings {
		s.pushNotice("Session", warning)
	}
	if projected.Provider != "" && projected.ModelID != "" {
		if err := s.setModel(projected.Provider + "/" + projected.ModelID); err != nil {
			s.pushNotice("Session", fmt.Sprintf(
				"this session was held on %s/%s, which is not available now: %v",
				projected.Provider, projected.ModelID, err))
		}
	}
	s.pushNotice("Session", fmt.Sprintf("switched to %s (%d messages)", info.ShortID(), len(projected.Messages)))
	return nil
}

// sessionListLines renders /sessions.
//
// It is deliberately short. runFullscreenCommand folds a command's whole output
// into one capped message card, so a fifty-row listing arrives as a single
// unscrollable blob; the picker is the real interface and the CLI is where a
// long listing belongs.
func sessionListLines(workdir string) []string {
	sessions, err := listSessionsForPicker(workdir, false)
	if err != nil {
		return []string{"could not read sessions: " + err.Error()}
	}
	if len(sessions) == 0 {
		return []string{"no saved sessions for this workspace"}
	}
	const shown = 10
	labels := sessionlog.ShortIDs(sessions)
	lines := make([]string, 0, shown+1)
	for index, info := range sessions {
		if index >= shown {
			lines = append(lines, fmt.Sprintf("… %d more · goshcoder sessions list", len(sessions)-shown))
			break
		}
		label, description := describeSessionAs(info, labels[index], false)
		lines = append(lines, label+"  "+description)
	}
	lines = append(lines, "/resume <search> to switch")
	return lines
}

// fullscreenResumeSuggestions builds the /resume drill-down.
func fullscreenResumeSuggestions(sessions []sessionlog.Info, query string, currentPath string) []fullscreenSuggestion {
	terms := strings.Fields(strings.ToLower(strings.TrimSpace(query)))
	suggestions := make([]fullscreenSuggestion, 0, len(sessions))
	labels := sessionlog.ShortIDs(sessions)
	for index, info := range sessions {
		// Guarded on a non-empty currentPath: an unsaved session has none, and
		// an empty match would hide every candidate.
		if currentPath != "" && info.Path == currentPath {
			continue
		}
		if len(terms) > 0 && !matchesSession(info, terms) {
			continue
		}
		label, description := describeSessionAs(info, labels[index], false)
		suggestions = append(suggestions, fullscreenSuggestion{
			label:       label,
			description: description,
			value:       "/resume " + label,
			execute:     true,
		})
	}
	return suggestions
}

// cachedSessionSuggestions keeps the picker off the render path.
//
// suggestions() runs inside View(), on every keystroke and on a 120 ms tick.
// Caching alone was not enough: the very first "/resume " keystroke still
// populated the cache inline, and populating it parses every session file in
// the shard and can fork git for the repository root. On a large history that
// is seconds of frozen interface, and on a network filesystem it can hang.
//
// The scan therefore runs on its own goroutine. Until it lands the picker
// simply reports that it is still loading, and invalidation clears the flag so
// the next miss starts a fresh scan.
func (model *fullscreenModel) cachedSessionSuggestions(query string) []fullscreenSuggestion {
	if !model.sessionChoicesValid {
		if !model.sessionScanRunning {
			model.sessionScanRunning = true
			root := model.session.workspaceRoot()
			target := model.sessionScanResult
			go func() {
				sessions, err := listSessionsForPicker(root, false)
				if err != nil {
					sessions = nil
				}
				select {
				case target <- sessions:
				default:
				}
			}()
		}
		select {
		case sessions := <-model.sessionScanResult:
			model.sessionChoices = sessions
			model.sessionChoicesValid = true
			model.sessionScanRunning = false
		default:
			return []fullscreenSuggestion{{
				label: "…", description: "reading saved sessions", value: "", execute: false,
			}}
		}
	}
	current := ""
	if model.session != nil && model.session.log != nil {
		current = model.session.log.path()
	}
	return fullscreenResumeSuggestions(model.sessionChoices, query, current)
}

// invalidateSessionSuggestions forces the picker to be rebuilt.
func (model *fullscreenModel) invalidateSessionSuggestions() {
	model.sessionChoicesValid = false
}

// missingWorkspaceNotice warns when a resumed session's workspace is gone.
//
// Without it the first tool call fails inside os.OpenRoot with an error that
// names neither the session nor the directory it expected.
func missingWorkspaceNotice(info sessionlog.Info, workdir string) string {
	if info.Cwd == "" || info.Cwd == workdir {
		return ""
	}
	if _, err := os.Stat(info.Cwd); err == nil {
		return fmt.Sprintf("this session was started in %s; tools will operate on %s instead", info.Cwd, workdir)
	}
	return fmt.Sprintf("this session's original workspace %s no longer exists; tools will operate on %s instead",
		info.Cwd, workdir)
}

// stdinReader is shared by every prompt that reads a line before the
// interactive interface starts.
//
// A fresh bufio.Reader per prompt discards whatever it buffered past the line
// it returned, so a second prompt -- or the chat loop itself -- silently lost
// input the user had already typed.
var (
	stdinOnce   sync.Once
	stdinShared *bufio.Reader
)

func sharedStdin() *bufio.Reader {
	stdinOnce.Do(func() { stdinShared = bufio.NewReader(os.Stdin) })
	return stdinShared
}

// readLineFromStdin reads one line for the startup picker.
func readLineFromStdin() (string, error) {
	line, err := sharedStdin().ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	return line, nil
}

// parsePositiveIndex reads a 1-based menu choice.
func parsePositiveIndex(input string, limit int) (int, error) {
	index, err := strconv.Atoi(input)
	if err != nil || index < 1 || index > limit {
		return 0, fmt.Errorf("choose a number between 1 and %d", limit)
	}
	return index, nil
}
