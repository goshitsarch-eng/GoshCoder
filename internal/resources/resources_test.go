package resources

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDiscoverBuildsContextTemplatesAndSkills(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	home := t.TempDir()
	root := filepath.Join(t.TempDir(), "repo")
	child := filepath.Join(root, "src")
	if err := os.MkdirAll(filepath.Join(child, ".pi", "prompts"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(child, ".pi", "skills", "review"), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(path, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(filepath.Join(home, "AGENTS.md"), "global rules")
	write(filepath.Join(root, "AGENTS.md"), "root rules")
	write(filepath.Join(child, "AGENTS.override.md"), "child override")
	write(filepath.Join(child, ".pi", "prompts", "review.md"), "---\ndescription: Review code\nargument-hint: <focus>\n---\nReview $1 and ${2:-tests}: $@")
	write(filepath.Join(child, ".pi", "skills", "review", "SKILL.md"), "---\nname: review\ndescription: Reviews code safely\n---\n# Review skill\nRead all files.")

	set, err := Discover(child, home)
	if err != nil {
		t.Fatal(err)
	}
	if len(set.ContextFiles) != 3 || len(set.Templates) != 1 || len(set.Skills) != 1 {
		t.Fatalf("resources = %#v", set)
	}
	prompt := set.BuildSystemPrompt("", child, []string{"read", "grep"})
	for _, expected := range []string{"root rules", "child override", "<available_skills>", "grep: Search file contents"} {
		if !strings.Contains(prompt, expected) {
			t.Fatalf("prompt missing %q:\n%s", expected, prompt)
		}
	}
	expanded, ok, err := set.Expand(`/review auth "edge cases"`)
	if err != nil || !ok {
		t.Fatalf("expand = %q, %v, %v", expanded, ok, err)
	}
	if expanded != "Review auth and edge cases: auth edge cases" {
		t.Fatalf("expanded = %q", expanded)
	}
	skill, ok, err := set.Expand("/skill:review focus on auth")
	if err != nil || !ok || !strings.Contains(skill, "# Review skill") || !strings.Contains(skill, "User: focus on auth") {
		t.Fatalf("skill = %q, %v, %v", skill, ok, err)
	}
}

func TestOverrideReplacesOnlySameDirectoryContext(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	child := filepath.Join(root, "child")
	if err := os.MkdirAll(child, 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(root, "AGENTS.md"), []byte("parent"), 0o600)
	_ = os.WriteFile(filepath.Join(child, "AGENTS.md"), []byte("ignored"), 0o600)
	_ = os.WriteFile(filepath.Join(child, "AGENTS.override.md"), []byte("override"), 0o600)
	set, err := Discover(child, filepath.Join(t.TempDir(), "agent"))
	if err != nil {
		t.Fatal(err)
	}
	prompt := set.BuildSystemPrompt("custom", child, nil)
	if !strings.Contains(prompt, "parent") || !strings.Contains(prompt, "override") || strings.Contains(prompt, "ignored") {
		t.Fatalf("prompt = %s", prompt)
	}
}
