package main

// /prompt and the `prompts` subcommand: saving a reusable prompt from inside
// the app, and backing the collection up so it survives a machine.

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"goshcoder/internal/agent"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/resources"
)

// reservedCommandNames is every name a template must not take.
//
// A template is expanded *before* the slash dispatcher runs in both interfaces,
// so a template called "model" would shadow /model outright. The set is the
// union of two dispatchers and it is not uniform: /btw and /omni setup are
// handled before expansion and are therefore immune, while /exit, /quit and
// /login are handled after and are not. It is built from the live tables rather
// than a copied list so it cannot drift as commands are added.
func reservedCommandNames(set *resources.Set) []string {
	seen := map[string]bool{}
	var names []string
	add := func(name string) {
		name = strings.TrimPrefix(strings.TrimSpace(name), "/")
		if name == "" || seen[name] {
			return
		}
		seen[name] = true
		names = append(names, name)
	}
	for _, command := range fullscreenSlashCommands {
		add(command.name)
	}
	// The line-oriented dispatcher handles names the fullscreen palette does
	// not list, and a template shadows a command in BOTH interfaces because
	// expansion runs before either dispatcher.
	for _, name := range lineModeSlashCommands {
		add(name)
	}
	for alias, canonical := range chatCommandAliases {
		add(alias)
		add(canonical)
	}
	add("?")
	add("help")
	// A live skill is reachable as /skill:<name>; the prefix itself is
	// reserved in ValidTemplateName, but a bare collision is worth refusing too.
	if set != nil {
		for _, skill := range set.Skills {
			add("skill:" + skill.Name)
		}
	}
	return names
}

// lineModeSlashCommands are the commands handleSlashCommand accepts that the
// fullscreen palette does not advertise.
var lineModeSlashCommands = []string{
	"exit", "quit", "help", "?", "system", "steer", "followup", "queue",
	"clear", "new", "compact", "reload", "resources", "messages", "tools",
	"status", "sidebar", "session", "sessions", "resume", "name", "model",
	"thinking", "login", "logout", "btw", "omni", "ralph", "planner",
	"planner-review", "planner-annotate", "planner-last", "prompt", "prompts",
	"tree", "fork", "label", "clone", "export", "import", "hotkeys",
}

// promptTargets returns the scopes a backup covers.
func promptTargets() []resources.SaveTarget {
	return []resources.SaveTarget{resources.SaveUser, resources.SaveProject}
}

// handlePromptCommand runs /prompt.
func (s *session) handlePromptCommand(rest string) error {
	action, argument, _ := strings.Cut(strings.TrimSpace(rest), " ")
	argument = strings.TrimSpace(argument)

	switch action {
	case "", "list":
		return s.promptList()
	case "save":
		return s.promptSave(argument)
	case "rm", "remove", "delete":
		return s.promptRemove(argument)
	case "edit":
		return s.promptEdit(argument)
	case "backup":
		return s.promptBackup(argument)
	case "restore":
		return s.promptRestore(argument)
	default:
		return fmt.Errorf("unknown /prompt action %q; use list, save, edit, rm, backup or restore", action)
	}
}

func (s *session) promptList() error {
	loaded := s.currentResources()
	if loaded == nil || len(loaded.Templates) == 0 {
		fmt.Fprintln(os.Stderr, dim("no saved prompts; /prompt save <name> stores the last thing you asked"))
		return nil
	}
	for _, template := range loaded.Templates {
		line := "/" + template.Name
		if template.Description != "" {
			line += "  " + template.Description
		}
		fmt.Fprintln(os.Stderr, line)
		fmt.Fprintln(os.Stderr, dim("    "+template.Path))
	}
	return nil
}

// promptSave captures the previous user message, or explicit inline text.
func (s *session) promptSave(argument string) error {
	name, inline, _ := strings.Cut(argument, " ")
	project := false
	if name == "--project" || name == "-project" {
		project = true
		name, inline, _ = strings.Cut(strings.TrimSpace(inline), " ")
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("/prompt save needs a name, e.g. /prompt save review")
	}

	body := strings.TrimSpace(inline)
	captured := false
	if body == "" {
		last, ok := lastUserPrompt(s.agent.State().Messages)
		if !ok {
			return errors.New("there is no previous message to save; pass the text after the name")
		}
		body, captured = last, true
	}

	target := resources.SaveUser
	if project {
		target = resources.SaveProject
	}
	options := resources.SaveOptions{
		Target:   target,
		Reserved: reservedCommandNames(s.currentResources()),
		// Captured text is escaped; text typed on the command line is taken as
		// written, since someone spelling out a template inline is likely to
		// want its placeholders live.
		Literal: captured,
	}

	path, shadowedBy, err := resources.SaveTemplate(s.workspaceRoot(), config.AgentDir(), name, body, options)
	if errors.Is(err, resources.ErrTemplateExists) {
		return fmt.Errorf("%w; /prompt rm %s first if you meant to replace it", err, name)
	}
	if err != nil {
		return err
	}
	if err := s.reloadTemplates(); err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "%s\n", dim("saved /"+name+" to "+path))
	if captured && resources.HasPlaceholders(body) {
		fmt.Fprintln(os.Stderr, dim(
			"the captured text contains $ placeholders; they were escaped so it expands exactly as written"))
	}
	if shadowedBy != "" {
		fmt.Fprintf(os.Stderr, "%s\n", dim("note: /"+name+" already resolves to "+shadowedBy+", which takes precedence"))
	}
	return nil
}

func (s *session) promptRemove(argument string) error {
	name, target := parsePromptScope(argument)
	if name == "" {
		return errors.New("/prompt rm needs a name")
	}
	path, err := resources.RemoveTemplate(s.workspaceRoot(), config.AgentDir(), name, target)
	if path != "" && err != nil {
		// A symlink was removed and the caller is being told what happened.
		fmt.Fprintf(os.Stderr, "%s\n", dim(err.Error()))
	} else if err != nil {
		return err
	}
	if err := s.reloadTemplates(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("removed "+path))
	return nil
}

// promptEdit opens a template in $EDITOR.
//
// Handing off rather than building an in-app editor: the composer strips
// pasted newlines and caps at three visible lines, so a multi-line editing
// surface would mean a fourth full-screen mode with its own copy of the roughly
// twenty-case key table.
//
// The editor is only launched from line mode. In the fullscreen interface a
// slash command runs with the alternate screen held and its output captured
// into a message card, so handing the terminal to vim there would leave both
// programs drawing over each other; the path is printed instead.
func (s *session) promptEdit(argument string) error {
	name, _ := parsePromptScope(argument)
	if name == "" {
		return errors.New("/prompt edit needs a name")
	}
	loaded := s.currentResources()
	if loaded == nil {
		return errors.New("no resources are loaded")
	}
	template, ok := loaded.FindTemplate(name)
	if !ok {
		return fmt.Errorf("no prompt named %q; /prompt list shows what is saved", name)
	}

	editor := firstNonEmptyEnv("VISUAL", "EDITOR")
	if editor == "" {
		fmt.Fprintf(os.Stderr, "%s\n", dim("set $EDITOR to edit in place; the file is "+template.Path))
		return nil
	}
	if s.fullscreen {
		fmt.Fprintf(os.Stderr, "%s\n", dim("edit it outside the app: "+editor+" "+template.Path))
		return nil
	}

	fields := strings.Fields(editor)
	command := exec.Command(fields[0], append(fields[1:], template.Path)...)
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := command.Run(); err != nil {
		return fmt.Errorf("run %s: %w", editor, err)
	}
	if err := s.reloadTemplates(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("reloaded /"+name))
	return nil
}

func firstNonEmptyEnv(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}

// parsePromptScope splits "<name> [--project|--user]".
func parsePromptScope(argument string) (string, resources.SaveTarget) {
	target := resources.SaveUser
	var name string
	for _, field := range strings.Fields(argument) {
		switch field {
		case "--project", "-project":
			target = resources.SaveProject
		case "--user", "-user":
			target = resources.SaveUser
		default:
			if name == "" {
				name = field
			}
		}
	}
	return name, target
}

// lastUserPrompt returns the most recent thing the user asked.
//
// This is the whole point of capture-from-transcript: the prompt worth keeping
// is usually the one just spent several turns refining, and asking the user to
// retype it defeats the feature.
func lastUserPrompt(messages []agent.Message) (string, bool) {
	for index := len(messages) - 1; index >= 0; index-- {
		user, ok := messages[index].(llm.UserMessage)
		if !ok {
			continue
		}
		if text, ok := user.StringContent(); ok && strings.TrimSpace(text) != "" {
			return strings.TrimSpace(text), true
		}
	}
	return "", false
}

// reloadTemplates re-reads only the prompt directories.
//
// Deliberately not reloadResources, which re-reads every AGENTS.md to the
// filesystem root, re-walks four skill trees, and then pushes a new system
// prompt and a rebuilt tool list into the running agent. Saving a prompt must
// not reset the tool set of a session that is mid-task.
func (s *session) reloadTemplates() error {
	if s == nil {
		return nil
	}
	s.promptMu.Lock()
	defer s.promptMu.Unlock()
	if s.resources == nil {
		return nil
	}
	s.resources = resources.ReloadTemplates(s.resources, s.workspaceRoot(), config.AgentDir())
	return nil
}

// ---------------------------------------------------------------------------
// backup and restore

func (s *session) promptBackup(argument string) error {
	path, err := backupPrompts(s.workspaceRoot(), strings.TrimSpace(argument))
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "%s\n", dim("backed up prompts to "+path))
	return nil
}

func (s *session) promptRestore(argument string) error {
	fields := strings.Fields(argument)
	if len(fields) == 0 {
		return errors.New("/prompt restore needs an archive path")
	}
	outcomes, err := restorePrompts(s.workspaceRoot(), fields[0], resources.RestoreOptions{
		Overwrite: hasFlag(fields[1:], "--overwrite", "-overwrite"),
		DryRun:    hasFlag(fields[1:], "--dry-run", "-dry-run"),
		Reserved:  reservedCommandNames(s.currentResources()),
	})
	if err != nil {
		return err
	}
	if err := s.reloadTemplates(); err != nil {
		return err
	}
	for _, line := range describeRestore(outcomes) {
		fmt.Fprintln(os.Stderr, dim(line))
	}
	return nil
}

func hasFlag(args []string, names ...string) bool {
	for _, arg := range args {
		for _, name := range names {
			if arg == name {
				return true
			}
		}
	}
	return false
}

// backupPrompts writes an archive, choosing a dated name when none is given.
func backupPrompts(cwd, out string) (string, error) {
	prompts, warnings, err := resources.CollectPrompts(cwd, config.AgentDir(), promptTargets())
	if err != nil {
		return "", err
	}
	for _, warning := range warnings {
		fmt.Fprintln(os.Stderr, dim("warning: "+warning))
	}
	if len(prompts) == 0 {
		return "", errors.New("there are no saved prompts to back up")
	}

	if out == "" {
		out = fmt.Sprintf("goshcoder-prompts-%s.tar.gz", time.Now().Format("2006-01-02"))
	}
	if info, err := os.Stat(out); err == nil && info.IsDir() {
		out = filepath.Join(out, fmt.Sprintf("goshcoder-prompts-%s.tar.gz", time.Now().Format("2006-01-02")))
	}
	// O_EXCL: a backup that silently replaced yesterday's would be the one
	// thing a backup must never do.
	file, err := os.OpenFile(out, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return "", fmt.Errorf("%s already exists; choose another name", out)
		}
		return "", err
	}
	if err := resources.WriteArchive(file, prompts, time.Now()); err != nil {
		file.Close()
		_ = os.Remove(out)
		return "", err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	return out, nil
}

func restorePrompts(cwd, archivePath string, options resources.RestoreOptions) ([]resources.RestoreOutcome, error) {
	file, err := os.Open(archivePath)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", archivePath, err)
	}
	defer file.Close()

	prompts, manifest, warnings, err := resources.ReadArchive(file)
	if err != nil {
		return nil, err
	}
	for _, warning := range warnings {
		fmt.Fprintln(os.Stderr, dim("warning: "+warning))
	}
	if len(prompts) == 0 {
		return nil, errors.New("this archive holds no prompts")
	}
	if manifest.Tool != "" && manifest.Tool != "goshcoder" {
		fmt.Fprintf(os.Stderr, "%s\n", dim("note: this archive was written by "+manifest.Tool))
	}
	return resources.RestorePrompts(cwd, config.AgentDir(), prompts, options)
}

func describeRestore(outcomes []resources.RestoreOutcome) []string {
	var restored, skipped int
	lines := make([]string, 0, len(outcomes)+1)
	for _, outcome := range outcomes {
		if outcome.Skipped {
			skipped++
			lines = append(lines, fmt.Sprintf("skipped /%s: %s", outcome.Name, outcome.Reason))
			continue
		}
		restored++
	}
	lines = append(lines, fmt.Sprintf("restored %d prompt(s), skipped %d", restored, skipped))
	return lines
}

// ---------------------------------------------------------------------------
// prompts subcommand

func promptsCommand(args []string) error {
	if len(args) == 0 {
		return promptsList()
	}
	switch args[0] {
	case "list":
		return promptsList()
	case "backup":
		out := ""
		if len(args) > 1 {
			out = args[1]
		}
		path, err := backupPrompts(currentWorkdir(), out)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s\n", dim("backed up prompts to "+path))
		fmt.Println(path)
		return nil
	case "restore":
		if len(args) < 2 {
			return errors.New("prompts restore needs an archive path")
		}
		outcomes, err := restorePrompts(currentWorkdir(), args[1], resources.RestoreOptions{
			Overwrite: hasFlag(args[2:], "--overwrite", "-overwrite"),
			DryRun:    hasFlag(args[2:], "--dry-run", "-dry-run"),
			Reserved:  reservedCommandNames(nil),
		})
		if err != nil {
			return err
		}
		for _, line := range describeRestore(outcomes) {
			fmt.Fprintln(os.Stderr, dim(line))
		}
		return nil
	default:
		return fmt.Errorf("unknown prompts subcommand %q; use list, backup or restore", args[0])
	}
}

func promptsList() error {
	prompts, warnings, err := resources.CollectPrompts(currentWorkdir(), config.AgentDir(), promptTargets())
	if err != nil {
		return err
	}
	for _, warning := range warnings {
		fmt.Fprintln(os.Stderr, dim("warning: "+warning))
	}
	if len(prompts) == 0 {
		fmt.Fprintln(os.Stderr, dim("no saved prompts"))
		return nil
	}
	for _, prompt := range prompts {
		scope := "user"
		if prompt.Target == resources.SaveProject {
			scope = "project"
		}
		fmt.Printf("/%-24s %s\n", prompt.Name, scope)
	}
	return nil
}
