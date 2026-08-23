package main

// Opening, resuming and reporting sessions: how the flags in sessionConfig
// become a live sessionRecorder, and how a loaded transcript is handed back to
// the user in every interface.

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/plannotator"
	"goshcoder/internal/sessionlog"
)

// sessionStore returns the store the flags select.
func sessionStore(cfg sessionConfig) *sessionlog.Store {
	root := cfg.SessionsDir
	if root == "" {
		root = config.SessionsDir()
	}
	return sessionlog.NewStore(config.ExpandTilde(root))
}

// openedSession is the result of resolving the session flags.
type openedSession struct {
	writer   *sessionlog.Writer
	restored restored
	// Notices are messages the interface should show once at startup.
	Notices []string
	// Resumed is true when an existing session was reopened.
	Resumed bool
}

// openSession resolves -continue / -session / -resume into a writer, folding
// any existing transcript.
//
// The ordering matters: this runs before the model is resolved, so a resumed
// session's recorded model can supply the default when the user passed no -m.
func openSession(cfg sessionConfig, workdir string) (*openedSession, error) {
	if cfg.NoSession {
		return &openedSession{}, nil
	}
	store := sessionStore(cfg)
	result := &openedSession{}

	var target sessionlog.Info
	var haveTarget bool
	switch {
	case cfg.Resume:
		info, chosen, err := chooseSessionInteractively(workdir)
		if err != nil {
			return nil, err
		}
		if !chosen {
			break
		}
		target, haveTarget = info, true
	case cfg.SessionRef != "":
		info, err := store.Resolve(workdir, cfg.SessionRef)
		if err != nil {
			if !errors.Is(err, sessionlog.ErrNotFound) {
				return nil, err
			}
			// A reference that names nothing is a request to create it, as long
			// as it is a bare id rather than a path.
			if strings.ContainsAny(cfg.SessionRef, `/\`) || strings.HasSuffix(cfg.SessionRef, ".jsonl") {
				return nil, err
			}
			writer, createErr := store.CreateWithID(workdir, "", cfg.SessionRef)
			if createErr != nil {
				return nil, createErr
			}
			result.writer = writer
			return result, nil
		}
		target, haveTarget = info, true
	case cfg.Continue:
		info, ok, err := store.MostRecent(workdir)
		if err != nil {
			return nil, err
		}
		if !ok {
			// Nothing to continue is not a failure: start one, so that
			// `chat -continue` is a safe thing to put in a shell alias.
			result.Notices = append(result.Notices, "no previous session for this workspace; starting a new one")
			break
		}
		target, haveTarget = info, true
	}

	if !haveTarget {
		writer, err := store.Create(workdir, "")
		if err != nil {
			return nil, err
		}
		result.writer = writer
		return result, nil
	}

	writer, report, err := openTarget(store, target, cfg, workdir, result)
	if err != nil {
		return nil, err
	}
	result.writer = writer
	result.Resumed = true
	result.restored = projectSession(writer.Snapshot(), writer.Leaf())
	result.Notices = append(result.Notices, reportNotices(report)...)
	result.Notices = append(result.Notices, result.restored.Warnings...)
	if notice := missingWorkspaceNotice(target, workdir); notice != "" {
		result.Notices = append(result.Notices, notice)
	}
	return result, nil
}

// openTarget attaches to an existing session, handling the two cases where
// attaching is not possible: a claim held elsewhere, and a pre-v3 pi file.
func openTarget(
	store *sessionlog.Store, target sessionlog.Info, cfg sessionConfig, workdir string, result *openedSession,
) (*sessionlog.Writer, sessionlog.LoadReport, error) {
	if cfg.ReadOnly {
		writer, report, err := store.Open(target.Path)
		if err == nil {
			result.Notices = append(result.Notices, "opened read-only; this session is not being saved")
		}
		return writer, report, err
	}

	writer, report, err := store.Attach(target.Path)
	switch {
	case err == nil:
		return writer, report, nil

	case errors.Is(err, sessionlog.ErrLegacyFormat):
		// A pi session older than v3 has no entry ids on disk, so appending to
		// it would make it migrate differently next time. Forking preserves
		// the transcript and gives it a file this build can actually extend.
		forked, forkErr := store.Fork(target, nil, workdir)
		if forkErr != nil {
			return nil, report, fmt.Errorf("continue a pre-v3 session: %w", forkErr)
		}
		result.Notices = append(result.Notices, fmt.Sprintf(
			"%s is an older pi session (v%d); it was copied into a new one so it can be continued",
			target.ShortID(), report.SourceVersion))
		return forked, report, nil

	case errors.Is(err, sessionlog.ErrBusy):
		// Another window is driving this session. Reading it is still useful,
		// and is strictly better than refusing to start.
		reader, readReport, readErr := store.Open(target.Path)
		if readErr != nil {
			return nil, report, err
		}
		result.Notices = append(result.Notices, fmt.Sprintf(
			"%s is open in another window (%s); continuing read-only, so this conversation is not being saved",
			target.ShortID(), target.Owner))
		return reader, readReport, nil

	default:
		return nil, report, err
	}
}

func reportNotices(report sessionlog.LoadReport) []string {
	var out []string
	if report.RepairedTail {
		out = append(out, "the session ended mid-write and its last incomplete entry was dropped")
	}
	if report.SkippedLines > 0 {
		out = append(out, fmt.Sprintf("%d unreadable line(s) were skipped: %s",
			report.SkippedLines, strings.Join(report.Warnings, "; ")))
	} else {
		for _, warning := range report.Warnings {
			out = append(out, warning)
		}
	}
	return out
}

// resolveResumedModel picks the model a resumed session should run on.
//
// The ladder is: the model recorded in the session, then an explicit -m, then
// the remembered default. A recorded model that no longer resolves -- a
// credential that expired, a gateway model that went away, a provider the user
// logged out of -- degrades to the next rung with a visible notice rather than
// failing the resume outright, because resume is exactly the operation that
// spans those changes.
func resolveResumedModel(cfg sessionConfig, opened *openedSession, modelFromFlag bool) (string, []string) {
	if opened == nil || !opened.Resumed {
		return cfg.ModelRef, nil
	}
	recorded := ""
	if opened.restored.Provider != "" && opened.restored.ModelID != "" {
		recorded = opened.restored.Provider + "/" + opened.restored.ModelID
	}
	// An explicit -m is the user overriding the session on purpose.
	if modelFromFlag {
		if recorded != "" && recorded != cfg.ModelRef {
			return cfg.ModelRef, []string{fmt.Sprintf(
				"this session was held on %s; continuing on %s because -m was given", recorded, cfg.ModelRef)}
		}
		return cfg.ModelRef, nil
	}
	if recorded == "" {
		return cfg.ModelRef, nil
	}
	if _, _, err := newCatalog().ResolveModel(recorded); err != nil {
		if cfg.ModelRef == "" {
			return recorded, nil
		}
		return cfg.ModelRef, []string{fmt.Sprintf(
			"this session was held on %s, which is not available now (%v); continuing on %s",
			recorded, err, cfg.ModelRef)}
	}
	return recorded, nil
}

// renderRestoredTranscript prints a resumed conversation for the interfaces
// that have no transcript of their own.
//
// renderEvent is purely event-driven, so line mode, piped mode and `run` would
// otherwise show nothing at all after a resume -- leaving the user unable to
// tell a successful resume from a silent failure. The fullscreen interface
// renders restored state for free, since its view reads the agent's messages.
func renderRestoredTranscript(messages []agent.Message, header string) {
	if len(messages) == 0 {
		return
	}
	fmt.Fprintln(os.Stderr, dim(header))
	for _, message := range messages {
		role := agent.RoleOf(message)
		if marker, ok := message.(compactionSummaryMessage); ok {
			fmt.Fprintf(os.Stderr, "%s %s\n", dim("· compacted"), dim(firstLine(marker.Summary)))
			continue
		}
		summary := summarizeMessage(message)
		if strings.TrimSpace(summary) == "" {
			continue
		}
		fmt.Fprintf(os.Stderr, "%s %s\n", dim("· "+role), dim(summary))
	}
	fmt.Fprintln(os.Stderr, dim(strings.Repeat("─", 40)))
}

// resumeBanner describes what was reopened.
func resumeBanner(opened *openedSession, recorder *sessionRecorder) string {
	if opened == nil || !opened.Resumed {
		return ""
	}
	count := len(opened.restored.Messages)
	id := recorder.id()
	if id == "" {
		id = "session"
	}
	return fmt.Sprintf("resumed %d message(s) from %s", count, shortSessionID(id))
}

func shortSessionID(id string) string {
	if len(id) >= 8 {
		return id[:8]
	}
	return id
}

// ---------------------------------------------------------------------------
// sessions subcommand

func sessionsCommand(args []string) error {
	if len(args) == 0 {
		return sessionsList(nil)
	}
	switch args[0] {
	case "list":
		return sessionsList(args[1:])
	case "show":
		return sessionsShow(args[1:])
	case "rm", "remove", "delete":
		return sessionsRemove(args[1:])
	case "gc":
		return sessionsGC(args[1:])
	case "export":
		return sessionsExport(args[1:])
	case "import":
		return sessionsImport(args[1:])
	default:
		return fmt.Errorf("unknown sessions subcommand %q; use list, show, rm, gc, export or import", args[0])
	}
}

func currentWorkdir() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	return dir
}

func defaultStore() *sessionlog.Store {
	return sessionlog.NewStore(config.SessionsDir())
}

func sessionsList(args []string) error {
	all := false
	for _, arg := range args {
		switch arg {
		case "--all", "-all", "-a":
			all = true
		default:
			return fmt.Errorf("unknown flag %q for sessions list", arg)
		}
	}
	workdir := currentWorkdir()
	sessions, err := defaultStore().List(workdir, sessionlog.ListOptions{AllWorkspaces: all})
	if err != nil {
		return err
	}
	if len(sessions) == 0 {
		if all {
			fmt.Fprintln(os.Stderr, dim("no sessions recorded yet"))
		} else {
			fmt.Fprintf(os.Stderr, "%s\n", dim("no sessions for "+workdir+" (use --all to see every workspace)"))
		}
		return nil
	}
	labels := sessionlog.ShortIDs(sessions)
	for index, info := range sessions {
		marker := " "
		if info.Locked {
			marker = "*"
		}
		line := fmt.Sprintf("%s %s  %s  %3d msg", marker, labels[index],
			info.Modified.Local().Format("2006-01-02 15:04"), info.Messages)
		if info.Cleared > 0 {
			line += fmt.Sprintf(" (+%d cleared)", info.Cleared)
		}
		fmt.Printf("%s  %s\n", line, info.Title())
		if all {
			fmt.Printf("     %s\n", dim(info.Cwd))
		}
	}
	if anyLocked(sessions) {
		fmt.Fprintln(os.Stderr, dim("* open in another window"))
	}
	return nil
}

func anyLocked(sessions []sessionlog.Info) bool {
	for _, info := range sessions {
		if info.Locked {
			return true
		}
	}
	return false
}

func sessionsShow(args []string) error {
	full := false
	var ref string
	for _, arg := range args {
		switch arg {
		case "--full", "-full":
			full = true
		default:
			if ref != "" {
				return errors.New("sessions show takes one session")
			}
			ref = arg
		}
	}
	if ref == "" {
		return errors.New("sessions show needs a session id, prefix, or path")
	}

	store := defaultStore()
	info, err := store.Resolve(currentWorkdir(), ref)
	if err != nil {
		return err
	}
	tree, header, report, err := store.Load(info.Path)
	if err != nil {
		return err
	}

	fmt.Printf("id        %s\n", info.ID)
	fmt.Printf("path      %s\n", info.Path)
	fmt.Printf("workspace %s\n", header.Cwd)
	if info.Name != "" {
		fmt.Printf("name      %s\n", info.Name)
	}
	if header.ParentSession != "" {
		fmt.Printf("forked    %s\n", header.ParentSession)
	}
	fmt.Printf("created   %s\n", info.Created.Local().Format("2006-01-02 15:04:05"))
	fmt.Printf("modified  %s\n", info.Modified.Local().Format("2006-01-02 15:04:05"))
	fmt.Printf("messages  %d", info.Messages)
	if info.Cleared > 0 {
		fmt.Printf(" (+%d before the last clear)", info.Cleared)
	}
	fmt.Println()
	fmt.Printf("size      %s\n", humanBytes(info.Size))
	if info.Locked {
		fmt.Printf("state     open in %s\n", info.Owner)
	}
	if report.Migrated {
		fmt.Printf("format    v%d, read as v%d (fork it to continue)\n", report.SourceVersion, sessionlog.FormatVersion)
	}
	for _, warning := range report.Warnings {
		fmt.Fprintln(os.Stderr, dim("warning: "+warning))
	}

	leaf := tree.Leaf()
	entries := tree.ContextPath(leaf)
	if full {
		entries = tree.Path(leaf)
	}
	fmt.Println()
	projected := projectSession(tree, leaf)
	if full {
		fmt.Fprintln(os.Stderr, dim(fmt.Sprintf("%d entries on the full path", len(entries))))
	}
	renderRestoredTranscript(projected.Messages, "transcript")
	return nil
}

func humanBytes(size int64) string {
	const unit = 1024
	if size < unit {
		return fmt.Sprintf("%d B", size)
	}
	div, exp := int64(unit), 0
	for n := size / unit; n >= unit && exp < 3; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(size)/float64(div), "KMGT"[exp])
}

func sessionsRemove(args []string) error {
	if len(args) == 0 {
		return errors.New("sessions rm needs at least one session id, prefix, or path")
	}
	store := defaultStore()
	workdir := currentWorkdir()
	var failures []string
	for _, ref := range args {
		info, err := store.Resolve(workdir, ref)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", ref, err))
			continue
		}
		if err := store.Remove(info); err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", ref, err))
			continue
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim("removed "+info.ShortID()+"  "+info.Title()))
	}
	if len(failures) > 0 {
		return errors.New(strings.Join(failures, "; "))
	}
	return nil
}

// sessionsExport writes a session out as v3 JSONL or as Markdown.
func sessionsExport(args []string) error {
	format := "jsonl"
	var ref, out string
	for _, arg := range args {
		switch arg {
		case "--jsonl", "-jsonl":
			format = "jsonl"
		case "--md", "-md", "--markdown":
			format = "md"
		default:
			if ref == "" {
				ref = arg
			} else if out == "" {
				out = arg
			} else {
				return errors.New("sessions export takes one session and one output path")
			}
		}
	}
	if ref == "" {
		return errors.New("sessions export needs a session id, prefix, or path")
	}

	store := defaultStore()
	info, err := store.Resolve(currentWorkdir(), ref)
	if err != nil {
		return err
	}

	var content []byte
	switch format {
	case "jsonl":
		content, err = os.ReadFile(info.Path)
		if err != nil {
			return fmt.Errorf("read %s: %w", info.Path, err)
		}
	case "md":
		tree, header, _, loadErr := store.Load(info.Path)
		if loadErr != nil {
			return loadErr
		}
		content = []byte(exportMarkdown(info, header, tree))
	}

	if out == "" || out == "-" {
		_, err = os.Stdout.Write(content)
		return err
	}
	if err := os.WriteFile(out, content, 0o600); err != nil {
		return fmt.Errorf("write %s: %w", out, err)
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("exported "+info.ShortID()+" to "+out))
	return nil
}

// exportMarkdown renders a readable transcript. It deliberately produces plain
// Markdown rather than pi's HTML export: that export inlines ~165 KB of
// vendored JavaScript into every file, and a Markdown transcript is what people
// actually paste into an issue or a review.
func exportMarkdown(info sessionlog.Info, header sessionlog.Header, tree *sessionlog.Tree) string {
	var out strings.Builder
	title := info.Title()
	fmt.Fprintf(&out, "# %s\n\n", title)
	fmt.Fprintf(&out, "- Session: `%s`\n- Workspace: `%s`\n- Started: %s\n\n",
		info.ID, header.Cwd, info.Created.Local().Format("2006-01-02 15:04:05"))

	projected := projectSession(tree, tree.Leaf())
	for _, message := range projected.Messages {
		switch value := message.(type) {
		case compactionSummaryMessage:
			fmt.Fprintf(&out, "---\n\n**Context compacted** (%d tokens before)\n\n%s\n\n",
				value.TokensBefore, value.Summary)
		case llm.UserMessage:
			text, ok := value.StringContent()
			if !ok {
				text = blockSummary(value.BlockContent())
			}
			fmt.Fprintf(&out, "## User\n\n%s\n\n", strings.TrimSpace(text))
		case llm.AssistantMessage:
			fmt.Fprintf(&out, "## Assistant\n\n")
			for _, block := range value.Content {
				switch content := block.(type) {
				case llm.TextContent:
					fmt.Fprintf(&out, "%s\n\n", strings.TrimSpace(content.Text))
				case llm.ToolCall:
					fmt.Fprintf(&out, "> called `%s`\n\n", content.Name)
				}
			}
		case llm.ToolResultMessage:
			status := ""
			if value.IsError {
				status = " (error)"
			}
			fmt.Fprintf(&out, "> `%s` returned%s\n\n", value.ToolName, status)
		}
	}
	return out.String()
}

// sessionsImport adopts an external session file into this workspace.
func sessionsImport(args []string) error {
	if len(args) != 1 {
		return errors.New("sessions import needs exactly one .jsonl path")
	}
	source := args[0]
	store := defaultStore()
	info, err := store.Resolve(currentWorkdir(), source)
	if err != nil {
		return fmt.Errorf("read %s: %w", source, err)
	}
	writer, err := store.Fork(info, nil, currentWorkdir())
	if err != nil {
		return err
	}
	path := writer.Path()
	id := writer.ID()
	if err := writer.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("imported as "+shortSessionID(id)))
	fmt.Println(path)
	return nil
}

// sessionsGC removes sessions older than a cutoff.
//
// It is explicit and prints what it will do. Nothing else in this tree deletes
// user data on its own, and a history that silently pruned itself would be a
// worse surprise than a directory that grows.
func sessionsGC(args []string) error {
	olderThan := ""
	keepNamed := false
	apply := false
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--older-than", "-older-than":
			if i+1 >= len(args) {
				return errors.New("--older-than needs a duration like 30d")
			}
			i++
			olderThan = args[i]
		case "--keep-named", "-keep-named":
			keepNamed = true
		case "--yes", "-yes", "-y":
			apply = true
		default:
			return fmt.Errorf("unknown flag %q for sessions gc", args[i])
		}
	}
	if olderThan == "" {
		return errors.New("sessions gc needs --older-than, e.g. --older-than 30d")
	}
	cutoff, err := parseAgeCutoff(olderThan)
	if err != nil {
		return err
	}

	store := defaultStore()
	sessions, err := store.List(currentWorkdir(), sessionlog.ListOptions{AllWorkspaces: true})
	if err != nil {
		return err
	}
	var doomed []sessionlog.Info
	for _, info := range sessions {
		if info.Modified.After(cutoff) || info.Locked {
			continue
		}
		if keepNamed && info.Name != "" {
			continue
		}
		doomed = append(doomed, info)
	}
	if len(doomed) == 0 {
		fmt.Fprintln(os.Stderr, dim("nothing to remove"))
		return nil
	}

	var freed int64
	for _, info := range doomed {
		freed += info.Size
		fmt.Printf("%s  %s  %s\n", info.ShortID(), info.Modified.Local().Format("2006-01-02"), info.Title())
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf("%d session(s), %s", len(doomed), humanBytes(freed))))
	if !apply {
		fmt.Fprintln(os.Stderr, dim("nothing was deleted; re-run with --yes to apply"))
		return nil
	}
	for _, info := range doomed {
		if err := store.Remove(info); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", dim("skipped "+info.ShortID()+": "+err.Error()))
		}
	}
	return nil
}

// parseAgeCutoff accepts a day or week count. A session age in hours is not
// something anyone reasons about, and accepting a bare number would leave the
// unit to guesswork on a command that deletes things.
func parseAgeCutoff(value string) (time.Time, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return time.Time{}, errors.New("empty duration")
	}
	unit := trimmed[len(trimmed)-1]
	number, err := strconv.Atoi(trimmed[:len(trimmed)-1])
	if err != nil || number < 0 {
		return time.Time{}, fmt.Errorf("cannot read %q as an age; use a form like 30d or 6w", value)
	}
	switch unit {
	case 'd':
		return time.Now().AddDate(0, 0, -number), nil
	case 'w':
		return time.Now().AddDate(0, 0, -7*number), nil
	default:
		return time.Time{}, fmt.Errorf("cannot read %q as an age; use a form like 30d or 6w", value)
	}
}

// exitHint tells the user where the conversation is being saved and how to
// come back to it. Suppressed when nothing is being recorded, so it never
// promises something that is not happening.
func (s *session) exitHint() string {
	if s == nil || !s.recordingActive() {
		return ""
	}
	return "session " + shortSessionID(s.log.id()) + " · resume with: goshcoder chat -continue"
}

// plannerCustomType names the Planner's custom entry in the session log.
const plannerCustomType = "goshcoder.planner"

// restoredPlannerState reads the Planner's phase out of a resumed session, and
// otherwise adopts the old per-workspace file once before removing it.
//
// A payload that does not decode yields nil plus a warning rather than an
// error: once phase lives in the session log, refusing to parse it would make
// a corrupt Planner entry a reason the whole session will not open.
func restoredPlannerState(opened *openedSession, workspaceRoot string, notices *[]string) *plannotator.State {
	if opened != nil && opened.Resumed {
		if raw, ok := opened.restored.Custom[plannerCustomType]; ok && len(raw) > 0 {
			var state plannotator.State
			if err := json.Unmarshal(raw, &state); err != nil {
				*notices = append(*notices,
					"the saved Planner state could not be read and was ignored: "+err.Error())
				return nil
			}
			return &state
		}
		return nil
	}
	return adoptLegacyPlannerState(workspaceRoot, notices)
}

// adoptLegacyPlannerState migrates the per-workspace file this build no longer
// writes. It runs once, for a new session in a workspace that was mid-plan, so
// the change of scoping does not silently drop an in-flight plan.
func adoptLegacyPlannerState(workspaceRoot string, notices *[]string) *plannotator.State {
	stateID := fmt.Sprintf("%x", sha256.Sum256([]byte(workspaceRoot)))[:16]
	path := filepath.Join(config.AgentDir(), "plannotator", stateID+".json")
	content, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var state plannotator.State
	if err := json.Unmarshal(content, &state); err != nil || state.Phase == "" || state.Phase == plannotator.PhaseIdle {
		_ = os.Remove(path)
		return nil
	}
	// Two sessions racing this removal is benign: whoever loses simply finds
	// nothing to adopt, and the state is already in the winner's session log.
	_ = os.Remove(path)
	*notices = append(*notices, fmt.Sprintf(
		"adopted the workspace's saved Planner state (%s); plan state now belongs to a session, so use -continue to come back to it",
		state.Phase))
	return &state
}

// sessionStorageLines describe where the conversation is being kept, for
// /session and for the startup card.
func (s *session) sessionStorageLines() []string {
	if s == nil {
		return nil
	}
	if s.log == nil || s.log.path() == "" {
		return []string{"session   not being saved (-no-session)"}
	}
	lines := []string{
		"session   " + shortSessionID(s.log.id()) + " · " + sessionRuntimeNotice(s.log, s.sessionReadOnly),
		"file      " + sessionRelativePath(s.log.path()),
	}
	if name := s.sessionName(); name != "" {
		lines = append(lines, "name      "+name)
	}
	return lines
}

// sidebarStorageLine is the one-line recording state for the fullscreen panel.
func (s *session) sidebarStorageLine() string {
	if s == nil || s.log == nil || s.log.path() == "" {
		return "not recording"
	}
	return "session " + shortSessionID(s.log.id()) + " · " + sessionRuntimeNotice(s.log, s.sessionReadOnly)
}

// sessionName returns the display name recorded for this session.
func (s *session) sessionName() string {
	if s == nil || s.log == nil {
		return ""
	}
	tree := s.log.snapshot()
	if tree == nil {
		return ""
	}
	return tree.Name()
}

// setSessionName records a display name.
func (s *session) setSessionName(name string) error {
	if s == nil || s.log == nil || !s.log.active() {
		return errors.New("this session is not being saved, so it cannot be named")
	}
	s.log.setName(strings.TrimSpace(name))
	s.log.sync()
	return nil
}

// sessionRuntimeNotice renders the recording state for the sidebar and
// /session.
func sessionRuntimeNotice(recorder *sessionRecorder, readOnly bool) string {
	switch {
	case recorder == nil || recorder.path() == "":
		return "not recording"
	case readOnly:
		return "read-only"
	case recorder.active():
		return "recording"
	default:
		return "recording stopped"
	}
}

// sessionRelativePath shortens a session path for display when it sits under
// the agent directory, which is where it almost always is.
func sessionRelativePath(path string) string {
	root := config.SessionsDir()
	if rel, err := filepath.Rel(root, path); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return path
}
