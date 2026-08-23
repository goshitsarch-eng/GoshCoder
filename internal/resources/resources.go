// Package resources discovers and renders pi-compatible local context files,
// prompt templates, skills, and system-prompt overrides.
package resources

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const maxResourceBytes = 2 * 1024 * 1024

// ContextFile is one AGENTS.md, AGENTS.override.md, or CLAUDE.md file.
type ContextFile struct {
	Path    string
	Content string
}

// Template is a slash-expandable Markdown prompt.
type Template struct {
	Name         string
	Description  string
	ArgumentHint string
	Path         string
	Body         string
}

// Skill is an Agent Skills-compatible SKILL.md resource.
type Skill struct {
	Name                   string
	Description            string
	Path                   string
	Body                   string
	DisableModelInvocation bool
}

// Set is the complete local resource snapshot for a workspace.
type Set struct {
	ContextFiles []ContextFile
	Templates    []Template
	Skills       []Skill
	CustomSystem string
	AppendSystem string
	Warnings     []string
}

// Discover scans the same conventional Markdown locations as pi. GoshCoder
// does not execute TypeScript resources, so project discovery is limited to
// inert text files.
func Discover(cwd, agentDir string) (*Set, error) {
	root, err := filepath.Abs(cwd)
	if err != nil {
		return nil, err
	}
	result := &Set{}
	result.ContextFiles = discoverContextFiles(root, agentDir, &result.Warnings)
	// A SYSTEM.md replaces the entire system prompt rather than adding to it,
	// so whoever supplies it controls the agent's instructions. The user's own
	// agent directory is therefore consulted first: cloning a repository must
	// not silently override the prompt its owner configured. A workspace copy
	// still works -- that is the point of the feature -- but it is announced
	// rather than applied invisibly.
	result.CustomSystem = readText(filepath.Join(agentDir, "SYSTEM.md"))
	if result.CustomSystem == "" {
		for _, candidate := range []string{
			filepath.Join(root, ".pi", "SYSTEM.md"),
			filepath.Join(root, ".goshcoder", "SYSTEM.md"),
		} {
			if text := readText(candidate); text != "" {
				result.CustomSystem = text
				result.Warnings = append(result.Warnings, fmt.Sprintf(
					"%s replaces the whole system prompt; review it if this repository is not yours", candidate))
				break
			}
		}
	}
	var appendParts []string
	for _, name := range []string{
		filepath.Join(agentDir, "APPEND_SYSTEM.md"),
		filepath.Join(root, ".pi", "APPEND_SYSTEM.md"),
		filepath.Join(root, ".goshcoder", "APPEND_SYSTEM.md"),
	} {
		if text := readText(name); text != "" {
			appendParts = append(appendParts, text)
		}
	}
	result.AppendSystem = strings.Join(appendParts, "\n\n")
	result.Templates = discoverTemplates(root, agentDir, &result.Warnings)
	result.Skills = discoverSkills(root, agentDir, &result.Warnings)
	return result, nil
}

func discoverContextFiles(cwd, agentDir string, warnings *[]string) []ContextFile {
	var files []ContextFile
	seen := map[string]bool{}
	add := func(path string) {
		absolute, _ := filepath.Abs(path)
		if seen[absolute] {
			return
		}
		// Refuse a symlink: a repository can ship AGENTS.md pointing at the
		// user's SSH key or auth.json, and the content of every context file
		// is injected into the system prompt and sent to the model provider.
		if info, lstatErr := os.Lstat(path); lstatErr == nil && info.Mode()&os.ModeSymlink != 0 {
			*warnings = append(*warnings, fmt.Sprintf("context %s: ignored because it is a symbolic link", path))
			return
		}
		content, err := readLimited(path)
		if errors.Is(err, os.ErrNotExist) {
			return
		}
		if err != nil {
			*warnings = append(*warnings, fmt.Sprintf("context %s: %v", path, err))
			return
		}
		seen[absolute] = true
		files = append(files, ContextFile{Path: absolute, Content: content})
	}
	add(filepath.Join(agentDir, "AGENTS.md"))
	for _, dir := range ancestorDirectories(cwd) {
		override := filepath.Join(dir, "AGENTS.override.md")
		if fileExists(override) {
			add(override)
			continue
		}
		agents := filepath.Join(dir, "AGENTS.md")
		if fileExists(agents) {
			add(agents)
			continue
		}
		add(filepath.Join(dir, "CLAUDE.md"))
	}
	return files
}

func discoverTemplates(cwd, agentDir string, warnings *[]string) []Template {
	locations := []string{
		filepath.Join(agentDir, "prompts"),
		filepath.Join(cwd, ".pi", "prompts"),
		filepath.Join(cwd, ".goshcoder", "prompts"),
	}
	seen := map[string]bool{}
	var templates []Template
	for _, directory := range locations {
		entries, err := os.ReadDir(directory)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			*warnings = append(*warnings, fmt.Sprintf("prompts %s: %v", directory, err))
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() || strings.ToLower(filepath.Ext(entry.Name())) != ".md" {
				continue
			}
			path := filepath.Join(directory, entry.Name())
			body, err := readLimited(path)
			if err != nil {
				*warnings = append(*warnings, fmt.Sprintf("prompt %s: %v", path, err))
				continue
			}
			meta, content := parseFrontmatter(body)
			name := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
			if seen[name] {
				continue
			}
			seen[name] = true
			description := meta["description"]
			if description == "" {
				description = firstNonEmptyLine(content)
			}
			templates = append(templates, Template{
				Name: name, Description: description, ArgumentHint: meta["argument-hint"],
				Path: path, Body: content,
			})
		}
	}
	sort.SliceStable(templates, func(i, j int) bool { return templates[i].Name < templates[j].Name })
	return templates
}

func discoverSkills(cwd, agentDir string, warnings *[]string) []Skill {
	locations := []string{filepath.Join(agentDir, "skills")}
	if home, err := os.UserHomeDir(); err == nil {
		locations = append(locations, filepath.Join(home, ".agents", "skills"))
	}
	locations = append(locations, filepath.Join(cwd, ".pi", "skills"), filepath.Join(cwd, ".goshcoder", "skills"))
	for _, dir := range ancestorDirectoriesUntilGit(cwd) {
		locations = append(locations, filepath.Join(dir, ".agents", "skills"))
	}
	seenPaths := map[string]bool{}
	seenNames := map[string]bool{}
	var skills []Skill
	for _, location := range locations {
		_ = filepath.WalkDir(location, func(path string, entry os.DirEntry, walkErr error) error {
			if walkErr != nil {
				if !errors.Is(walkErr, os.ErrNotExist) {
					*warnings = append(*warnings, fmt.Sprintf("skills %s: %v", location, walkErr))
				}
				return filepath.SkipDir
			}
			if entry.IsDir() {
				if path != location && strings.HasPrefix(entry.Name(), ".") {
					return filepath.SkipDir
				}
				return nil
			}
			if entry.Name() != "SKILL.md" && !(filepath.Dir(path) == location && strings.ToLower(filepath.Ext(path)) == ".md") {
				return nil
			}
			absolute, _ := filepath.Abs(path)
			if seenPaths[absolute] {
				return nil
			}
			seenPaths[absolute] = true
			body, err := readLimited(path)
			if err != nil {
				*warnings = append(*warnings, fmt.Sprintf("skill %s: %v", path, err))
				return nil
			}
			meta, content := parseFrontmatter(body)
			name := strings.TrimSpace(meta["name"])
			if name == "" {
				name = strings.TrimSuffix(filepath.Base(filepath.Dir(path)), filepath.Ext(path))
				if entry.Name() != "SKILL.md" {
					name = strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
				}
			}
			description := strings.TrimSpace(meta["description"])
			if description == "" {
				*warnings = append(*warnings, fmt.Sprintf("skill %s has no description and was skipped", path))
				return nil
			}
			if seenNames[name] {
				*warnings = append(*warnings, fmt.Sprintf("duplicate skill %q at %s was skipped", name, path))
				return nil
			}
			seenNames[name] = true
			disabled, _ := strconv.ParseBool(meta["disable-model-invocation"])
			skills = append(skills, Skill{
				Name: name, Description: description, Path: absolute, Body: content,
				DisableModelInvocation: disabled,
			})
			return nil
		})
	}
	sort.SliceStable(skills, func(i, j int) bool { return skills[i].Name < skills[j].Name })
	return skills
}

// BuildSystemPrompt creates pi's default coding-assistant framing and appends
// discovered context and skill metadata. explicit replaces the default prompt.
func (set *Set) BuildSystemPrompt(explicit, cwd string, toolNames []string) string {
	selected := make(map[string]bool, len(toolNames))
	for _, name := range toolNames {
		selected[name] = true
	}
	base := strings.TrimSpace(explicit)
	if base == "" {
		base = strings.TrimSpace(set.CustomSystem)
	}
	if base == "" {
		var tools []string
		for _, name := range []string{"read", "write", "edit", "bash", "grep", "find", "ls", "web_search"} {
			if selected[name] {
				tools = append(tools, "- "+name+": "+toolSnippet(name))
			}
		}
		if len(tools) == 0 {
			tools = []string{"(none)"}
		}
		guidelines := []string{
			"- Be concise in your responses",
			"- Show file paths clearly when working with files",
		}
		if selected["read"] {
			guidelines = append([]string{"- Use read to examine files instead of cat or sed"}, guidelines...)
		}
		base = "You are an expert coding assistant operating inside GoshCoder, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\nAvailable tools:\n" + strings.Join(tools, "\n") + "\n\nGuidelines:\n" + strings.Join(guidelines, "\n")
	}
	if strings.TrimSpace(set.AppendSystem) != "" {
		base += "\n\n" + strings.TrimSpace(set.AppendSystem)
	}
	if len(set.ContextFiles) > 0 {
		base += "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
		for _, file := range set.ContextFiles {
			base += fmt.Sprintf("<project_instructions path=%q>\n%s\n</project_instructions>\n\n",
				filepath.ToSlash(file.Path), sealDelimiters(file.Content))
		}
		base += "</project_context>"
	}
	if selected["read"] {
		var visible []Skill
		for _, skill := range set.Skills {
			if !skill.DisableModelInvocation {
				visible = append(visible, skill)
			}
		}
		if len(visible) > 0 {
			base += "\n\n<available_skills>\n"
			for _, skill := range visible {
				base += fmt.Sprintf("<skill><name>%s</name><description>%s</description><location>%s</location></skill>\n", xmlEscape(skill.Name), xmlEscape(skill.Description), xmlEscape(filepath.ToSlash(skill.Path)))
			}
			base += "</available_skills>"
		}
	}
	absolute, _ := filepath.Abs(cwd)
	return base + "\nCurrent working directory: " + filepath.ToSlash(absolute)
}

// sealDelimiters neutralises the framing tags inside untrusted context content.
//
// Project context is interpolated between <project_instructions> tags, and its
// content comes from files in whatever repository the user happens to have
// open. Without this an AGENTS.md that contains the closing tag can break out
// of its container and address the model as if it were the harness itself.
//
// Only the delimiters are altered, not the whole text: escaping every angle
// bracket would mangle ordinary Markdown and code samples that the file is
// there to communicate.
func sealDelimiters(content string) string {
	replacer := strings.NewReplacer(
		"</project_instructions>", "<\u200b/project_instructions>",
		"<project_instructions", "<\u200bproject_instructions",
		"</project_context>", "<\u200b/project_context>",
		"<project_context>", "<\u200bproject_context>",
	)
	return replacer.Replace(content)
}

func toolSnippet(name string) string {
	switch name {
	case "read":
		return "Read file contents"
	case "write":
		return "Create or overwrite files"
	case "edit":
		return "Apply exact text replacements"
	case "bash":
		return "Run shell commands"
	case "grep":
		return "Search file contents for patterns (respects .gitignore)"
	case "find":
		return "Find files by glob pattern (respects .gitignore)"
	case "ls":
		return "List directory contents"
	case "web_search":
		return "Search the web and return cited sources"
	default:
		return name
	}
}

// Expand resolves /template and /skill:name input. ok is false when the input
// is not a resource command.
func (set *Set) Expand(input string) (expanded string, ok bool, err error) {
	command, rest, _ := strings.Cut(strings.TrimSpace(input), " ")
	for _, template := range set.Templates {
		if command != "/"+template.Name {
			continue
		}
		args, parseErr := splitArguments(rest)
		if parseErr != nil {
			return "", true, parseErr
		}
		return expandTemplate(template.Body, args), true, nil
	}
	if strings.HasPrefix(command, "/skill:") {
		name := strings.TrimPrefix(command, "/skill:")
		for _, skill := range set.Skills {
			if skill.Name != name {
				continue
			}
			text := fmt.Sprintf("Follow the %q skill from %s:\n\n%s", skill.Name, filepath.ToSlash(skill.Path), skill.Body)
			if strings.TrimSpace(rest) != "" {
				text += "\n\nUser: " + strings.TrimSpace(rest)
			}
			return text, true, nil
		}
	}
	return "", false, nil
}

func expandTemplate(body string, args []string) string {
	all := strings.Join(args, " ")
	defaultPattern := regexp.MustCompile(`\$\{(\d+|@|ARGUMENTS):-([^}]*)\}`)
	body = defaultPattern.ReplaceAllStringFunc(body, func(match string) string {
		parts := defaultPattern.FindStringSubmatch(match)
		value := argumentValue(parts[1], args, all)
		if value == "" {
			return parts[2]
		}
		return value
	})
	slicePattern := regexp.MustCompile(`\$\{@:(\d+)(?::(\d+))?\}`)
	body = slicePattern.ReplaceAllStringFunc(body, func(match string) string {
		parts := slicePattern.FindStringSubmatch(match)
		start, _ := strconv.Atoi(parts[1])
		start--
		if start < 0 || start >= len(args) {
			return ""
		}
		end := len(args)
		if parts[2] != "" {
			length, _ := strconv.Atoi(parts[2])
			end = min(len(args), start+length)
		}
		return strings.Join(args[start:end], " ")
	})
	positional := regexp.MustCompile(`\$(\d+)`)
	body = positional.ReplaceAllStringFunc(body, func(match string) string {
		index, _ := strconv.Atoi(strings.TrimPrefix(match, "$"))
		if index < 1 || index > len(args) {
			return ""
		}
		return args[index-1]
	})
	body = strings.ReplaceAll(body, "$ARGUMENTS", all)
	body = strings.ReplaceAll(body, "$@", all)
	return strings.TrimSpace(body)
}

func argumentValue(key string, args []string, all string) string {
	if key == "@" || key == "ARGUMENTS" {
		return all
	}
	index, _ := strconv.Atoi(key)
	if index < 1 || index > len(args) {
		return ""
	}
	return args[index-1]
}

func splitArguments(input string) ([]string, error) {
	var args []string
	var current strings.Builder
	var quote rune
	escaped := false
	flush := func() {
		if current.Len() > 0 {
			args = append(args, current.String())
			current.Reset()
		}
	}
	for _, r := range strings.TrimSpace(input) {
		if escaped {
			current.WriteRune(r)
			escaped = false
			continue
		}
		if r == '\\' && quote != '\'' {
			escaped = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			} else {
				current.WriteRune(r)
			}
			continue
		}
		switch r {
		case '\'', '"':
			quote = r
		case ' ', '\t', '\n':
			flush()
		default:
			current.WriteRune(r)
		}
	}
	if escaped {
		current.WriteRune('\\')
	}
	if quote != 0 {
		return nil, errors.New("unterminated quote in prompt-template arguments")
	}
	flush()
	return args, nil
}

func parseFrontmatter(input string) (map[string]string, string) {
	meta := map[string]string{}
	normalized := strings.ReplaceAll(input, "\r\n", "\n")
	if !strings.HasPrefix(normalized, "---\n") {
		return meta, input
	}
	end := strings.Index(normalized[4:], "\n---\n")
	if end < 0 {
		return meta, input
	}
	head := normalized[4 : 4+end]
	body := normalized[4+end+5:]
	for _, line := range strings.Split(head, "\n") {
		key, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		meta[strings.TrimSpace(key)] = strings.Trim(strings.TrimSpace(value), `"'`)
	}
	return meta, body
}

func ancestorDirectories(path string) []string {
	var reverse []string
	for current := filepath.Clean(path); ; current = filepath.Dir(current) {
		reverse = append(reverse, current)
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
	}
	result := make([]string, len(reverse))
	for index := range reverse {
		result[len(reverse)-1-index] = reverse[index]
	}
	return result
}

// ancestorDirectoriesUntilGit lists path and its ancestors, stopping at the
// repository root.
//
// The stop condition used fileExists, which requires a *regular* file, but .git
// is a directory in an ordinary clone -- it is only a file in a worktree or a
// submodule. So the walk never stopped and ran all the way to the filesystem
// root, reaching sibling checkouts and the user's home directory and offering
// their skills to the model as if they belonged to this project.
func ancestorDirectoriesUntilGit(path string) []string {
	var result []string
	for current := filepath.Clean(path); ; current = filepath.Dir(current) {
		result = append(result, current)
		if isRepositoryRoot(current) {
			break
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
	}
	return result
}

// isRepositoryRoot reports whether dir holds a .git entry, of either shape.
func isRepositoryRoot(dir string) bool {
	_, err := os.Lstat(filepath.Join(dir, ".git"))
	return err == nil
}

func readLimited(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	content, err := io.ReadAll(io.LimitReader(file, maxResourceBytes+1))
	if err != nil {
		return "", err
	}
	if len(content) > maxResourceBytes {
		return "", fmt.Errorf("resource exceeds %d bytes", maxResourceBytes)
	}
	return strings.TrimSpace(string(content)), nil
}

func readText(path string) string {
	text, err := readLimited(path)
	if err != nil {
		return ""
	}
	return text
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func firstNonEmptyLine(text string) string {
	for _, line := range strings.Split(text, "\n") {
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return ""
}

func xmlEscape(text string) string {
	replacer := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;")
	return replacer.Replace(text)
}
