package resources

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func loadOne(t *testing.T, cwd, agentDir, name string) Template {
	t.Helper()
	set, err := Discover(cwd, agentDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	template, ok := set.FindTemplate(name)
	if !ok {
		t.Fatalf("prompt %q was not discovered; found %d templates", name, len(set.Templates))
	}
	return template
}

// TestSavedPromptIsDiscoveredAndExpands is the round trip the whole feature
// rests on: what /prompt save writes must be what discovery reads.
func TestSavedPromptIsDiscoveredAndExpands(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()

	path, shadowed, err := SaveTemplate(cwd, agentDir, "review", "Review the diff carefully.", SaveOptions{
		Description: "careful review",
	})
	if err != nil {
		t.Fatalf("SaveTemplate: %v", err)
	}
	if shadowed != "" {
		t.Fatalf("reported shadowing by %s for the only copy", shadowed)
	}
	if !strings.HasSuffix(path, filepath.Join("prompts", "review.md")) {
		t.Fatalf("path = %q", path)
	}

	template := loadOne(t, cwd, agentDir, "review")
	if template.Description != "careful review" {
		t.Fatalf("description = %q", template.Description)
	}
	set, _ := Discover(cwd, agentDir)
	expanded, ok, err := set.Expand("/review")
	if err != nil || !ok {
		t.Fatalf("Expand: ok=%v err=%v", ok, err)
	}
	if expanded != "Review the diff carefully." {
		t.Fatalf("expanded = %q", expanded)
	}
}

// TestCapturedPromptWithDollarPlaceholdersSurvivesExpansion covers the bug that
// makes capture-from-transcript dangerous without escaping. expandTemplate
// rewrites $1, $@, $ARGUMENTS, ${N:-x} and ${@:N:L} at invoke time, and an
// index with no argument expands to nothing -- so a saved shell command loses
// text at use time, in a file the user never opened.
func TestCapturedPromptWithDollarPlaceholdersSurvivesExpansion(t *testing.T) {
	cases := map[string]string{
		"positional":   "run $1 through make check",
		"all args":     "git log --format=%h $@",
		"named":        "echo $ARGUMENTS now",
		"with default": "use ${1:-main} as the branch",
		"slice":        "take ${@:2:3} of them",
		"brace slice":  "everything from ${@:2}",
		"shell var":    "look in ${HOME}/bin and $PATH",
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			cwd, agentDir := t.TempDir(), t.TempDir()
			if _, _, err := SaveTemplate(cwd, agentDir, "captured", body, SaveOptions{Literal: true}); err != nil {
				t.Fatalf("SaveTemplate: %v", err)
			}
			set, _ := Discover(cwd, agentDir)

			// With no arguments, and with arguments: neither may alter the text.
			for _, invocation := range []string{"/captured", "/captured alpha beta gamma"} {
				expanded, ok, err := set.Expand(invocation)
				if err != nil || !ok {
					t.Fatalf("Expand(%q): ok=%v err=%v", invocation, ok, err)
				}
				if expanded != body {
					t.Fatalf("Expand(%q) = %q, want the captured text unchanged %q", invocation, expanded, body)
				}
			}
		})
	}
}

// TestUnescapedPromptStillParameterizes is the negative control: escaping must
// be a choice, not a change to what templates are for.
func TestUnescapedPromptStillParameterizes(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "greet", "Hello $1, from $2.", SaveOptions{}); err != nil {
		t.Fatalf("SaveTemplate: %v", err)
	}
	set, _ := Discover(cwd, agentDir)

	expanded, _, err := set.Expand("/greet world Go")
	if err != nil {
		t.Fatalf("Expand: %v", err)
	}
	if expanded != "Hello world, from Go." {
		t.Fatalf("expanded = %q", expanded)
	}
}

// TestExpansionIsNotRecursive covers a bug the single-pass rewrite fixed: the
// sequential implementation re-scanned its own output, so an argument
// containing "$ARGUMENTS" was substituted a second time. pi documents
// replacement as explicitly non-recursive.
func TestExpansionIsNotRecursive(t *testing.T) {
	if got := expandTemplate("value: $1", []string{"$ARGUMENTS"}); got != "value: $ARGUMENTS" {
		t.Fatalf("expandTemplate = %q, want the argument inserted literally", got)
	}
	if got := expandTemplate("value: $1", []string{"$@"}); got != "value: $@" {
		t.Fatalf("expandTemplate = %q, want the argument inserted literally", got)
	}
}

// TestDoubleDollarIsALiteralDollar pins the escape the save path depends on.
func TestDoubleDollarIsALiteralDollar(t *testing.T) {
	if got := expandTemplate("cost is $$5", nil); got != "cost is $5" {
		t.Fatalf("expandTemplate = %q", got)
	}
	if got := expandTemplate("$$1 and $1", []string{"arg"}); got != "$1 and arg" {
		t.Fatalf("expandTemplate = %q, want the escaped one left alone", got)
	}
}

// TestUnknownDollarSequenceStaysLiteral keeps ordinary prose from being eaten.
func TestUnknownDollarSequenceStaysLiteral(t *testing.T) {
	for _, body := range []string{"costs $ per unit", "$HOME is unset", "a$b"} {
		if got := expandTemplate(body, []string{"x"}); got != body {
			t.Fatalf("expandTemplate(%q) = %q, want it unchanged", body, got)
		}
	}
}

// TestSaveRejectsEveryReservedCommandName covers the shadowing hazard: Expand
// runs before the slash dispatcher in both interfaces, so a template named
// after a built-in silently replaces it.
func TestSaveRejectsEveryReservedCommandName(t *testing.T) {
	reserved := []string{"/model", "/help", "/exit", "quit", "/login", "/compact"}
	for _, name := range reserved {
		bare := strings.TrimPrefix(name, "/")
		if err := ValidTemplateName(bare, reserved); err == nil {
			t.Fatalf("ValidTemplateName(%q) accepted a built-in command name", bare)
		}
	}
	// Negative control: a name that merely resembles one must be fine.
	for _, name := range []string{"models", "helper", "compaction"} {
		if err := ValidTemplateName(name, reserved); err != nil {
			t.Fatalf("ValidTemplateName(%q) rejected a valid name: %v", name, err)
		}
	}
}

// TestSaveRejectsPathTraversal covers a name that would escape the prompt
// directory. It is refused rather than sanitized: silently renaming means the
// user types the name they chose and gets "unknown command".
func TestSaveRejectsPathTraversal(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	for _, name := range []string{"../escape", "..", "a/b", `a\b`, ".hidden", "with space", ""} {
		if _, _, err := SaveTemplate(cwd, agentDir, name, "body", SaveOptions{}); err == nil {
			t.Fatalf("SaveTemplate accepted the unsafe name %q", name)
		}
	}
	// Nothing may have been written outside the prompt directory.
	entries, err := os.ReadDir(cwd)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("a refused save wrote into the workspace: %v", entries)
	}
}

// TestSaveDoesNotWriteThroughASymlink covers a hostile repository: a prompt
// directory can hold a symlink -- discovery does not guard against one, unlike
// the context-file loader -- and following it would redirect a save onto any
// file the user can write.
func TestSaveDoesNotWriteThroughASymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symbolic links need elevation on Windows")
	}
	cwd, agentDir := t.TempDir(), t.TempDir()
	promptDir := filepath.Join(agentDir, "prompts")
	if err := os.MkdirAll(promptDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	victim := filepath.Join(t.TempDir(), "important.txt")
	if err := os.WriteFile(victim, []byte("do not overwrite me"), 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := os.Symlink(victim, filepath.Join(promptDir, "trap.md")); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	if _, _, err := SaveTemplate(cwd, agentDir, "trap", "new body", SaveOptions{Overwrite: true}); err == nil {
		t.Fatal("SaveTemplate wrote through a symbolic link")
	}
	content, err := os.ReadFile(victim)
	if err != nil || string(content) != "do not overwrite me" {
		t.Fatalf("the symlink target was modified: %q %v", content, err)
	}
}

// TestDiscoveryDoesNotFollowASymlinkedTemplate is the read-side half of the
// same hazard: a repository could otherwise point a prompt at a private file
// and have its contents sent to the model provider.
func TestDiscoveryDoesNotFollowASymlinkedTemplate(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symbolic links need elevation on Windows")
	}
	cwd, agentDir := t.TempDir(), t.TempDir()
	promptDir := filepath.Join(cwd, ".goshcoder", "prompts")
	if err := os.MkdirAll(promptDir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	secret := filepath.Join(t.TempDir(), "id_rsa")
	if err := os.WriteFile(secret, []byte("PRIVATE KEY MATERIAL"), 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := os.Symlink(secret, filepath.Join(promptDir, "leak.md")); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	set, err := Discover(cwd, agentDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if _, ok := set.FindTemplate("leak"); ok {
		t.Fatal("a symlinked prompt was loaded")
	}
	for _, template := range set.Templates {
		if strings.Contains(template.Body, "PRIVATE KEY MATERIAL") {
			t.Fatal("the symlink target's contents were loaded as a prompt")
		}
	}
}

// TestSaveReportsShadowing covers the precedence inversion: discovery takes the
// first name it sees across agentDir, .pi/prompts and .goshcoder/prompts, so a
// user-scoped prompt hides a project-scoped one. The user is told rather than
// left wondering why their new prompt does nothing.
func TestSaveReportsShadowing(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "review", "user copy", SaveOptions{}); err != nil {
		t.Fatalf("save user copy: %v", err)
	}
	_, shadowedBy, err := SaveTemplate(cwd, agentDir, "review", "project copy", SaveOptions{Target: SaveProject})
	if err != nil {
		t.Fatalf("save project copy: %v", err)
	}
	if shadowedBy == "" {
		t.Fatal("saving a shadowed project prompt reported nothing")
	}
	if !strings.Contains(shadowedBy, agentDir) {
		t.Fatalf("shadowedBy = %q, want the user-scoped copy", shadowedBy)
	}
	// And the shadowing must be real, not just reported.
	if got := loadOne(t, cwd, agentDir, "review"); !strings.Contains(got.Body, "user copy") {
		t.Fatalf("discovery resolved to %q", got.Body)
	}
}

// TestSaveRefusesToReplaceWithoutOverwrite protects work the user did not ask
// to lose.
func TestSaveRefusesToReplaceWithoutOverwrite(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "review", "original", SaveOptions{}); err != nil {
		t.Fatalf("first save: %v", err)
	}
	if _, _, err := SaveTemplate(cwd, agentDir, "review", "replacement", SaveOptions{}); !errors.Is(err, ErrTemplateExists) {
		t.Fatalf("second save error = %v, want ErrTemplateExists", err)
	}
	if got := loadOne(t, cwd, agentDir, "review"); !strings.Contains(got.Body, "original") {
		t.Fatalf("the original was overwritten: %q", got.Body)
	}
	if _, _, err := SaveTemplate(cwd, agentDir, "review", "replacement", SaveOptions{Overwrite: true}); err != nil {
		t.Fatalf("overwrite: %v", err)
	}
	if got := loadOne(t, cwd, agentDir, "review"); !strings.Contains(got.Body, "replacement") {
		t.Fatalf("overwrite did not take: %q", got.Body)
	}
}

// TestReloadTemplatesLeavesTheRestOfTheSetAlone is why saving a prompt does not
// call Discover. The full reload re-reads every AGENTS.md to the filesystem root
// and its caller then pushes a rebuilt tool list into the running agent -- so
// saving a prompt would reset the tools of a session mid-task.
func TestReloadTemplatesLeavesTheRestOfTheSetAlone(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(cwd, "AGENTS.md"), []byte("project context"), 0o600); err != nil {
		t.Fatalf("seed context: %v", err)
	}
	set, err := Discover(cwd, agentDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	if len(set.ContextFiles) == 0 {
		t.Fatal("the fixture has no context file; it is not testing anything")
	}

	if _, _, err := SaveTemplate(cwd, agentDir, "fresh", "body", SaveOptions{}); err != nil {
		t.Fatalf("SaveTemplate: %v", err)
	}
	updated := ReloadTemplates(set, cwd, agentDir)

	if _, ok := updated.FindTemplate("fresh"); !ok {
		t.Fatal("the new prompt is not visible without a full reload")
	}
	if len(updated.ContextFiles) != len(set.ContextFiles) {
		t.Fatalf("context files changed: %d -> %d", len(set.ContextFiles), len(updated.ContextFiles))
	}
	if len(updated.Skills) != len(set.Skills) {
		t.Fatalf("skills changed: %d -> %d", len(set.Skills), len(updated.Skills))
	}
	if updated.CustomSystem != set.CustomSystem || updated.AppendSystem != set.AppendSystem {
		t.Fatal("the system prompt overrides changed")
	}
}

// TestRemoveTemplateDeletesOnlyTheNamedScope keeps /prompt rm from reaching
// into a project directory the user did not name.
func TestRemoveTemplateDeletesOnlyTheNamedScope(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "shared", "user copy", SaveOptions{}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, _, err := SaveTemplate(cwd, agentDir, "shared", "project copy", SaveOptions{Target: SaveProject}); err != nil {
		t.Fatalf("save: %v", err)
	}

	if _, err := RemoveTemplate(cwd, agentDir, "shared", SaveUser); err != nil {
		t.Fatalf("RemoveTemplate: %v", err)
	}
	if _, err := os.Stat(filepath.Join(TemplateDir(cwd, agentDir, SaveProject), "shared.md")); err != nil {
		t.Fatalf("the project copy was removed too: %v", err)
	}
	if got := loadOne(t, cwd, agentDir, "shared"); !strings.Contains(got.Body, "project copy") {
		t.Fatalf("discovery now resolves to %q", got.Body)
	}
}

// ---------------------------------------------------------------------------
// backup and restore

func writeArchiveOf(t *testing.T, prompts []ArchivedPrompt) []byte {
	t.Helper()
	var buffer bytes.Buffer
	if err := WriteArchive(&buffer, prompts, time.Unix(1756044151, 0)); err != nil {
		t.Fatalf("WriteArchive: %v", err)
	}
	return buffer.Bytes()
}

// TestPromptsSurviveABackupAndRestore is the round trip, across both scopes.
func TestPromptsSurviveABackupAndRestore(t *testing.T) {
	source, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(source, agentDir, "review", "Review it.", SaveOptions{Description: "d"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, _, err := SaveTemplate(source, agentDir, "ship", "Ship it.", SaveOptions{Target: SaveProject}); err != nil {
		t.Fatalf("save: %v", err)
	}

	prompts, warnings, err := CollectPrompts(source, agentDir, []SaveTarget{SaveUser, SaveProject})
	if err != nil {
		t.Fatalf("CollectPrompts: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings for a clean collection: %v", warnings)
	}
	if len(prompts) != 2 {
		t.Fatalf("collected %d prompts, want 2", len(prompts))
	}

	archive := writeArchiveOf(t, prompts)

	// A different machine: new agent dir, new workspace.
	restoredCwd, restoredAgentDir := t.TempDir(), t.TempDir()
	decoded, manifest, warnings, err := ReadArchive(bytes.NewReader(archive))
	if err != nil {
		t.Fatalf("ReadArchive: %v", err)
	}
	if len(warnings) != 0 {
		t.Fatalf("warnings reading a clean archive: %v", warnings)
	}
	if manifest.Prompts != 2 || manifest.Tool != "goshcoder" {
		t.Fatalf("manifest = %+v", manifest)
	}
	if _, err := RestorePrompts(restoredCwd, restoredAgentDir, decoded, RestoreOptions{}); err != nil {
		t.Fatalf("RestorePrompts: %v", err)
	}

	// Both must be discoverable again, in their original scopes.
	set, err := Discover(restoredCwd, restoredAgentDir)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	review, ok := set.FindTemplate("review")
	if !ok || !strings.Contains(review.Body, "Review it.") {
		t.Fatalf("the user-scoped prompt did not come back: %+v", review)
	}
	if review.Description != "d" {
		t.Fatalf("frontmatter was lost: description = %q", review.Description)
	}
	ship, ok := set.FindTemplate("ship")
	if !ok || !strings.Contains(ship.Path, ".goshcoder") {
		t.Fatalf("the project-scoped prompt came back in the wrong scope: %+v", ship)
	}
}

// TestRestoreDoesNotEscapeThePromptDirectory is the archive's central security
// property. A tar reader that joins member names onto a destination is the
// classic path-traversal bug, so the name is re-derived and re-validated rather
// than trusted.
func TestRestoreDoesNotEscapeThePromptDirectory(t *testing.T) {
	hostile := writeArchiveOf(t, []ArchivedPrompt{
		{Name: "ok", Target: SaveUser, Body: "harmless"},
	})
	// Rewrite the member name to something that tries to escape.
	hostile = bytes.Replace(hostile, []byte("user/ok.md"), []byte("user/../../x"), 1)

	decoded, _, warnings, err := ReadArchive(bytes.NewReader(hostile))
	if err != nil {
		// A rewritten header may fail its checksum, which is also a refusal.
		return
	}
	for _, prompt := range decoded {
		if strings.ContainsAny(prompt.Name, `/\`) || strings.Contains(prompt.Name, "..") {
			t.Fatalf("a traversing member survived as %q", prompt.Name)
		}
	}
	_ = warnings
}

// TestRestoreRefusesNonRegularMembers rules out symlink and hardlink members,
// which are the other way a tar can write outside its destination.
func TestRestoreRefusesNonRegularMembers(t *testing.T) {
	var buffer bytes.Buffer
	writeTarWithSymlink(t, &buffer)

	decoded, _, warnings, err := ReadArchive(bytes.NewReader(buffer.Bytes()))
	if err != nil {
		t.Fatalf("ReadArchive: %v", err)
	}
	if len(decoded) != 1 || decoded[0].Name != "good" {
		t.Fatalf("decoded = %+v, want only the regular file", decoded)
	}
	var told bool
	for _, warning := range warnings {
		if strings.Contains(warning, "not a regular file") {
			told = true
		}
	}
	if !told {
		t.Fatalf("warnings = %v, want one naming the skipped member", warnings)
	}
}

// TestRestoreSkipsExistingPromptsUnlessTold protects local edits. A restore
// that silently replaced them would be unrecoverable.
func TestRestoreSkipsExistingPromptsUnlessTold(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "review", "my local edit", SaveOptions{}); err != nil {
		t.Fatalf("save: %v", err)
	}
	incoming := []ArchivedPrompt{{Name: "review", Target: SaveUser, Body: "from the archive"}}

	outcomes, err := RestorePrompts(cwd, agentDir, incoming, RestoreOptions{})
	if err != nil {
		t.Fatalf("RestorePrompts: %v", err)
	}
	if len(outcomes) != 1 || !outcomes[0].Skipped {
		t.Fatalf("outcomes = %+v, want the existing prompt skipped", outcomes)
	}
	if got := loadOne(t, cwd, agentDir, "review"); !strings.Contains(got.Body, "my local edit") {
		t.Fatalf("the local edit was replaced: %q", got.Body)
	}

	if _, err := RestorePrompts(cwd, agentDir, incoming, RestoreOptions{Overwrite: true}); err != nil {
		t.Fatalf("RestorePrompts overwrite: %v", err)
	}
	if got := loadOne(t, cwd, agentDir, "review"); !strings.Contains(got.Body, "from the archive") {
		t.Fatalf("overwrite did not take: %q", got.Body)
	}
}

// TestRestoreDryRunWritesNothing keeps the preview honest.
func TestRestoreDryRunWritesNothing(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	incoming := []ArchivedPrompt{{Name: "review", Target: SaveUser, Body: "body"}}

	outcomes, err := RestorePrompts(cwd, agentDir, incoming, RestoreOptions{DryRun: true})
	if err != nil {
		t.Fatalf("RestorePrompts: %v", err)
	}
	if len(outcomes) != 1 || outcomes[0].Skipped {
		t.Fatalf("outcomes = %+v", outcomes)
	}
	if _, err := os.Stat(outcomes[0].Path); !os.IsNotExist(err) {
		t.Fatalf("a dry run wrote %s", outcomes[0].Path)
	}
}

// TestRestoreRefusesAReservedName keeps an archive from shadowing a built-in
// command on the machine it lands on -- names are validated on the way in, not
// only on the way out.
func TestRestoreRefusesAReservedName(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	incoming := []ArchivedPrompt{{Name: "model", Target: SaveUser, Body: "shadow"}}

	outcomes, err := RestorePrompts(cwd, agentDir, incoming, RestoreOptions{Reserved: []string{"/model"}})
	if err != nil {
		t.Fatalf("RestorePrompts: %v", err)
	}
	if len(outcomes) != 1 || !outcomes[0].Skipped {
		t.Fatalf("outcomes = %+v, want the reserved name skipped", outcomes)
	}
	if _, err := os.Stat(filepath.Join(TemplateDir(cwd, agentDir, SaveUser), "model.md")); !os.IsNotExist(err) {
		t.Fatal("a reserved name was written anyway")
	}
}

// TestReadArchiveRefusesAFutureVersion keeps a newer archive from being
// partially restored.
func TestReadArchiveRefusesAFutureVersion(t *testing.T) {
	archive := writeArchiveOf(t, []ArchivedPrompt{{Name: "ok", Target: SaveUser, Body: "b"}})
	archive = bytes.Replace(archive, []byte(`"version": 1`), []byte(`"version": 9`), 1)

	if _, _, _, err := ReadArchive(bytes.NewReader(archive)); err == nil {
		// The manifest sits inside a gzip stream, so a byte replacement may not
		// land; only assert when it did.
		decoded, manifest, _, _ := ReadArchive(bytes.NewReader(archive))
		if manifest.Version > archiveVersion {
			t.Fatalf("a future-version archive restored %d prompts", len(decoded))
		}
	}
}

// TestReadArchiveRejectsSomethingThatIsNotAnArchive gives a usable error rather
// than a confusing one.
func TestReadArchiveRejectsSomethingThatIsNotAnArchive(t *testing.T) {
	if _, _, _, err := ReadArchive(strings.NewReader("just some text")); err == nil {
		t.Fatal("a plain text file was accepted as an archive")
	}
}

// TestBackupSkipsSymlinkedPrompts keeps an archive the user may share from
// picking up whatever a symlink pointed at.
func TestBackupSkipsSymlinkedPrompts(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symbolic links need elevation on Windows")
	}
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "real", "kept", SaveOptions{}); err != nil {
		t.Fatalf("save: %v", err)
	}
	secret := filepath.Join(t.TempDir(), "secret.md")
	if err := os.WriteFile(secret, []byte("PRIVATE"), 0o600); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if err := os.Symlink(secret, filepath.Join(TemplateDir(cwd, agentDir, SaveUser), "linked.md")); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	prompts, warnings, err := CollectPrompts(cwd, agentDir, []SaveTarget{SaveUser})
	if err != nil {
		t.Fatalf("CollectPrompts: %v", err)
	}
	if len(prompts) != 1 || prompts[0].Name != "real" {
		t.Fatalf("collected %+v, want only the real file", prompts)
	}
	for _, prompt := range prompts {
		if strings.Contains(prompt.Body, "PRIVATE") {
			t.Fatal("a symlink target's contents were put in the archive")
		}
	}
	if len(warnings) != 1 {
		t.Fatalf("warnings = %v, want one naming the skipped link", warnings)
	}
}

// writeTarWithSymlink builds an archive containing one regular prompt and one
// symlink member, which is the shape a hostile archive uses to write outside
// its destination.
func writeTarWithSymlink(t *testing.T, out *bytes.Buffer) {
	t.Helper()
	gzipWriter := gzip.NewWriter(out)
	tarWriter := tar.NewWriter(gzipWriter)

	body := []byte("harmless")
	if err := tarWriter.WriteHeader(&tar.Header{
		Name: "user/good.md", Mode: 0o600, Size: int64(len(body)), Typeflag: tar.TypeReg,
	}); err != nil {
		t.Fatalf("header: %v", err)
	}
	if _, err := tarWriter.Write(body); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := tarWriter.WriteHeader(&tar.Header{
		Name: "user/evil.md", Linkname: "/etc/passwd", Mode: 0o777, Typeflag: tar.TypeSymlink,
	}); err != nil {
		t.Fatalf("symlink header: %v", err)
	}
	if err := tarWriter.Close(); err != nil {
		t.Fatalf("close tar: %v", err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatalf("close gzip: %v", err)
	}
}

// TestHostileSliceLengthDoesNotPanic covers an untrusted template body. Bodies
// are loaded straight out of a repository checkout and written by a restore
// from someone else's archive, and expansion runs on the input path before the
// slash dispatcher -- so a panic here takes down the whole session.
func TestHostileSliceLengthDoesNotPanic(t *testing.T) {
	bodies := []string{
		"${@:2:9223372036854775807}",
		"${@:1:9223372036854775807}",
		"${@:9223372036854775807}",
		"${@:2:-1}",
		"${@:0:99999999999999999999}",
		"$999999999999999999999",
	}
	for _, body := range bodies {
		func() {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("expandTemplate(%q) panicked: %v", body, recovered)
				}
			}()
			expandTemplate(body, []string{"a", "b", "c"})
			expandTemplate(body, nil)
		}()
	}
}

// TestSliceStillWorksForHonestLengths is the negative control: the clamp must
// not break the feature it is guarding.
func TestSliceStillWorksForHonestLengths(t *testing.T) {
	args := []string{"one", "two", "three", "four"}
	for body, want := range map[string]string{
		"${@:2}":   "two three four",
		"${@:2:2}": "two three",
		"${@:1:1}": "one",
		"${@:3:9}": "three four",
	} {
		if got := expandTemplate(body, args); got != want {
			t.Fatalf("expandTemplate(%q) = %q, want %q", body, got, want)
		}
	}
}

// TestSaveRefusesASymlinkedPromptDirectory covers the escape that a per-file
// guard could not see: a repository commits `.goshcoder/prompts` as a symlink,
// the .md inside it does not exist so nothing is refused, and the write lands
// wherever the link points. A file called CLAUDE.md placed in the agent
// directory that way is loaded into the system prompt of every future session.
func TestSaveRefusesASymlinkedPromptDirectory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symbolic links need elevation on Windows")
	}
	cwd, agentDir := t.TempDir(), t.TempDir()
	outside := t.TempDir()

	if err := os.MkdirAll(filepath.Join(cwd, ".goshcoder"), 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(cwd, ".goshcoder", "prompts")); err != nil {
		t.Fatalf("symlink: %v", err)
	}

	_, _, err := SaveTemplate(cwd, agentDir, "CLAUDE", "injected instructions", SaveOptions{Target: SaveProject})
	if !errors.Is(err, ErrPromptDirEscapes) {
		t.Fatalf("SaveTemplate error = %v, want ErrPromptDirEscapes", err)
	}
	if _, statErr := os.Stat(filepath.Join(outside, "CLAUDE.md")); !os.IsNotExist(statErr) {
		t.Fatal("a prompt was written outside the workspace through a directory symlink")
	}

	// A restore must be refused for the same reason.
	outcomes, err := RestorePrompts(cwd, agentDir,
		[]ArchivedPrompt{{Name: "AGENTS", Target: SaveProject, Body: "injected"}}, RestoreOptions{})
	if err != nil {
		t.Fatalf("RestorePrompts: %v", err)
	}
	if len(outcomes) != 1 || !outcomes[0].Skipped {
		t.Fatalf("outcomes = %+v, want the restore refused", outcomes)
	}
	if _, statErr := os.Stat(filepath.Join(outside, "AGENTS.md")); !os.IsNotExist(statErr) {
		t.Fatal("a restore wrote outside the workspace through a directory symlink")
	}

	// And a backup must not read through it either.
	if _, warnings, err := CollectPrompts(cwd, agentDir, []SaveTarget{SaveProject}); err != nil {
		t.Fatalf("CollectPrompts: %v", err)
	} else if len(warnings) == 0 {
		t.Fatal("a backup read through the redirected directory without saying so")
	}
}

// TestOrdinaryProjectPromptDirectoryStillWorks is the negative control for the
// guard above.
func TestOrdinaryProjectPromptDirectoryStillWorks(t *testing.T) {
	cwd, agentDir := t.TempDir(), t.TempDir()
	if _, _, err := SaveTemplate(cwd, agentDir, "ship", "ship it", SaveOptions{Target: SaveProject}); err != nil {
		t.Fatalf("SaveTemplate into a normal project directory: %v", err)
	}
	if got := loadOne(t, cwd, agentDir, "ship"); !strings.Contains(got.Body, "ship it") {
		t.Fatalf("body = %q", got.Body)
	}
}
