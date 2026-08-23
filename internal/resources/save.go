package resources

// Creating, editing and deleting prompt templates from inside the app.
//
// This has no pi equivalent: pi's slash-command table contains neither /prompt
// nor /save, and its templates are read-only Markdown a user authors by hand.
// There is therefore a pi *file format* to stay compatible with -- a template
// written here is discovered by pi identically -- but no pi *behaviour* to be
// faithful to.

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"goshcoder/internal/atomicfile"
)

// SaveTarget selects which prompt directory a template is written to.
type SaveTarget uint8

const (
	// SaveUser writes to the agent directory, where it is available in every
	// workspace.
	SaveUser SaveTarget = iota
	// SaveProject writes to <cwd>/.goshcoder/prompts, where it can be
	// committed and shared with a repository.
	SaveProject
)

// SaveOptions configures SaveTemplate.
type SaveOptions struct {
	Description  string
	ArgumentHint string
	Target       SaveTarget
	// Reserved names a template may not take, because a template is expanded
	// before the slash dispatcher runs and would otherwise shadow a built-in.
	Reserved []string
	// Literal escapes $ placeholders so the saved text expands back to exactly
	// what was captured. On by default for capture; see EscapePlaceholders.
	Literal bool
	// Overwrite permits replacing an existing template of the same name.
	Overwrite bool
}

// validTemplateName is deliberately stricter than a filesystem would require.
// A template name becomes both a filename and a slash command, so anything
// that is awkward in either is refused up front rather than sanitized into
// something the user did not ask for.
var validTemplateName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

// ErrTemplateExists is returned when a template of that name is already there
// and Overwrite was not set.
var ErrTemplateExists = errors.New("resources: a prompt with that name already exists")

// ValidTemplateName rejects a name that cannot safely be a file or a command.
func ValidTemplateName(name string, reserved []string) error {
	if name == "" {
		return errors.New("a prompt name is required")
	}
	if !validTemplateName.MatchString(name) {
		return fmt.Errorf(
			"%q cannot be a prompt name: use letters, digits, '.', '-' and '_', starting with a letter or digit", name)
	}
	// Refuse rather than rename. Silently sanitizing a name means the user
	// types the name they chose and gets "unknown command".
	if strings.Contains(name, "..") {
		return fmt.Errorf("%q cannot be a prompt name", name)
	}
	if strings.HasPrefix(name, "skill:") {
		return errors.New("names beginning with \"skill:\" are reserved for Agent Skills")
	}
	for _, taken := range reserved {
		if strings.EqualFold(name, strings.TrimPrefix(taken, "/")) {
			return fmt.Errorf("/%s is a built-in command; choose another name", name)
		}
	}
	return nil
}

// TemplateDir returns the directory a target writes to.
func TemplateDir(cwd, agentDir string, target SaveTarget) string {
	if target == SaveProject {
		return filepath.Join(cwd, ".goshcoder", "prompts")
	}
	return filepath.Join(agentDir, "prompts")
}

// placeholderPattern matches every form expandTemplate rewrites.
var placeholderPattern = regexp.MustCompile(`\$\{(?:\d+|@|ARGUMENTS):-[^}]*\}|\$\{@:\d+(?::\d+)?\}|\$(?:ARGUMENTS|@|\d+)`)

// EscapePlaceholders makes captured text expand back to itself.
//
// This is a correctness fix, not a nicety. expandTemplate unconditionally
// rewrites $1, $@, $ARGUMENTS, ${N:-default} and ${@:N:L} at invoke time, and
// an index with no matching argument expands to the empty string. The prompts
// worth saving are full of shell and code -- "run $1 through make check",
// "git log --format=%h $@" -- so without escaping, a saved prompt silently
// loses text at use time, in a file the user never opened.
//
// $$ is expandTemplate's escape for a literal $, so doubling is the inverse.
func EscapePlaceholders(body string) string {
	return placeholderPattern.ReplaceAllStringFunc(body, func(match string) string {
		return "$" + match
	})
}

// HasPlaceholders reports whether body contains anything expansion would
// rewrite, so a caller can warn before saving it raw.
func HasPlaceholders(body string) bool { return placeholderPattern.MatchString(body) }

// SaveTemplate writes a prompt template.
//
// It returns the path written and, when the new template is shadowed by one of
// the same name in a higher-precedence directory, the path doing the shadowing.
// Discovery takes the first name it sees across agentDir, .pi/prompts and
// .goshcoder/prompts in that order, so a user-scoped template hides a
// project-scoped one -- the reverse of what most people expect, which is why
// the caller is told rather than left to wonder why nothing changed.
func SaveTemplate(cwd, agentDir, name, body string, options SaveOptions) (path string, shadowedBy string, err error) {
	if err := ValidTemplateName(name, options.Reserved); err != nil {
		return "", "", err
	}
	body = strings.TrimSpace(body)
	if body == "" {
		return "", "", errors.New("a prompt needs some text")
	}
	if options.Literal {
		body = EscapePlaceholders(body)
	}

	content := renderTemplateFile(options.Description, options.ArgumentHint, body)
	if len(content) > maxResourceBytes {
		// Enforced here as well as at read time: a template that discovery
		// would refuse is not worth writing.
		return "", "", fmt.Errorf("this prompt is %d bytes, over the %d-byte limit", len(content), maxResourceBytes)
	}

	directory := TemplateDir(cwd, agentDir, options.Target)
	path = filepath.Join(directory, name+".md")

	// Never write through a discovered path. A prompt directory can contain a
	// symlink -- discoverTemplates does not guard against one, unlike the
	// context-file loader -- and following it would let a repository redirect a
	// save onto any file the user can write.
	if info, statErr := os.Lstat(path); statErr == nil {
		if info.Mode()&os.ModeSymlink != 0 {
			return "", "", fmt.Errorf("%s is a symbolic link; refusing to write through it", path)
		}
		if !options.Overwrite {
			return "", "", fmt.Errorf("%w: %s", ErrTemplateExists, path)
		}
	}

	if err := atomicfile.Write(path, []byte(content), 0o600); err != nil {
		return "", "", err
	}
	return path, shadowingTemplate(cwd, agentDir, name, path), nil
}

// renderTemplateFile produces the frontmatter shape discoverTemplates reads,
// which is byte-identical to what pi's own loader expects.
func renderTemplateFile(description, argumentHint, body string) string {
	var out strings.Builder
	if description != "" || argumentHint != "" {
		out.WriteString("---\n")
		if description != "" {
			out.WriteString("description: " + singleLine(description) + "\n")
		}
		if argumentHint != "" {
			out.WriteString("argument-hint: " + singleLine(argumentHint) + "\n")
		}
		out.WriteString("---\n")
	}
	out.WriteString(body)
	out.WriteString("\n")
	return out.String()
}

// singleLine keeps a frontmatter value on one line; the parser is line-based.
func singleLine(value string) string {
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	return strings.TrimSpace(value)
}

// shadowingTemplate returns the path of a same-named template that discovery
// would find first, or "".
func shadowingTemplate(cwd, agentDir, name, written string) string {
	for _, directory := range templateLocations(cwd, agentDir) {
		candidate := filepath.Join(directory, name+".md")
		if candidate == written {
			return ""
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return ""
}

// templateLocations is discovery's search order, in precedence order.
func templateLocations(cwd, agentDir string) []string {
	return []string{
		filepath.Join(agentDir, "prompts"),
		filepath.Join(cwd, ".pi", "prompts"),
		filepath.Join(cwd, ".goshcoder", "prompts"),
	}
}

// RemoveTemplate deletes a prompt template.
func RemoveTemplate(cwd, agentDir, name string, target SaveTarget) (string, error) {
	if err := ValidTemplateName(name, nil); err != nil {
		return "", err
	}
	path := filepath.Join(TemplateDir(cwd, agentDir, target), name+".md")
	info, err := os.Lstat(path)
	if err != nil {
		return "", fmt.Errorf("no prompt named %q in %s", name, TemplateDir(cwd, agentDir, target))
	}
	if info.Mode()&os.ModeSymlink != 0 {
		// Removing the link is what the user asked for, but say so: they may
		// have meant the file it pointed at.
		if err := os.Remove(path); err != nil {
			return "", err
		}
		return path, fmt.Errorf("%s was a symbolic link; the link was removed, not its target", path)
	}
	if err := os.Remove(path); err != nil {
		return "", err
	}
	return path, nil
}

// FindTemplate returns a loaded template by name.
func (set *Set) FindTemplate(name string) (Template, bool) {
	for _, template := range set.Templates {
		if template.Name == name {
			return template, true
		}
	}
	return Template{}, false
}

// ReloadTemplates re-reads only the prompt directories, returning a copy of set
// with fresh Templates.
//
// Narrow on purpose. The full Discover re-reads every AGENTS.md up to the
// filesystem root and re-walks four skill trees, and its caller then pushes a
// new system prompt and a rebuilt tool list into the running agent. Saving a
// prompt must not reset the tool set of a session mid-task. Templates are
// verifiably not part of the system prompt: BuildSystemPrompt embeds context
// files and skills only.
func ReloadTemplates(set *Set, cwd, agentDir string) *Set {
	if set == nil {
		return nil
	}
	updated := *set
	var warnings []string
	updated.Templates = discoverTemplates(cwd, agentDir, &warnings)
	// Warnings from the previous discovery that concerned prompts are replaced
	// wholesale; anything else is kept.
	kept := make([]string, 0, len(set.Warnings))
	for _, warning := range set.Warnings {
		if !strings.HasPrefix(warning, "prompt") {
			kept = append(kept, warning)
		}
	}
	updated.Warnings = append(kept, warnings...)
	return &updated
}
