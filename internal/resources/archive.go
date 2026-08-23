package resources

// Backing up and restoring saved prompts.
//
// A prompt archive is a gzipped tar holding a manifest and one .md file per
// prompt, using only archive/tar and compress/gzip from the standard library --
// no new module dependency, and a format anyone can open with `tar tzf`.
//
// Restoring reads a file the user may have been handed by someone else, so the
// reader treats every archive as hostile: names are re-derived rather than
// trusted, entry types other than a regular file are refused outright, and both
// the entry count and the decompressed size are bounded. A tar reader that
// simply joins its member names onto a destination directory is the classic
// path-traversal bug, and a gzip stream can decompress to far more than it
// costs to send.

import (
	"archive/tar"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"goshcoder/internal/atomicfile"
)

const (
	// archiveVersion is the manifest's format version. A newer archive is
	// refused rather than partially restored.
	archiveVersion = 1
	// maxArchiveEntries bounds a hostile archive's member count.
	maxArchiveEntries = 10_000
	// maxArchiveBytes bounds total decompressed size, independent of the
	// compressed size on disk.
	maxArchiveBytes  = 64 << 20
	manifestName     = "manifest.json"
	userArchiveDir   = "user"
	projectArchiveDi = "project"
)

// ArchiveManifest describes a prompt archive.
type ArchiveManifest struct {
	Version int    `json:"version"`
	Tool    string `json:"tool"`
	// Created is RFC3339. Informational only; restore never trusts it.
	Created string `json:"created"`
	Prompts int    `json:"prompts"`
}

// ArchivedPrompt is one prompt in an archive.
type ArchivedPrompt struct {
	Name   string
	Target SaveTarget
	Body   string
}

// scopeDir maps a target onto its directory inside the archive.
func scopeDir(target SaveTarget) string {
	if target == SaveProject {
		return projectArchiveDi
	}
	return userArchiveDir
}

// CollectPrompts gathers the prompt files for a backup.
//
// Files are read verbatim, frontmatter included, so a restore reproduces
// exactly what was saved rather than a re-rendered approximation.
func CollectPrompts(cwd, agentDir string, targets []SaveTarget) ([]ArchivedPrompt, []string, error) {
	var prompts []ArchivedPrompt
	var warnings []string
	for _, target := range targets {
		directory := TemplateDir(cwd, agentDir, target)
		entries, err := os.ReadDir(directory)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return nil, nil, fmt.Errorf("read %s: %w", directory, err)
		}
		for _, entry := range entries {
			if entry.IsDir() || strings.ToLower(filepath.Ext(entry.Name())) != ".md" {
				continue
			}
			full := filepath.Join(directory, entry.Name())
			// Skipped rather than followed: a prompt directory can hold a
			// symlink, and backing up whatever it points at would copy a file
			// the user never meant to put in an archive they may share.
			info, err := os.Lstat(full)
			if err != nil {
				warnings = append(warnings, fmt.Sprintf("%s: %v", full, err))
				continue
			}
			if info.Mode()&os.ModeSymlink != 0 {
				warnings = append(warnings, fmt.Sprintf("%s is a symbolic link and was not backed up", full))
				continue
			}
			body, err := readLimited(full)
			if err != nil {
				warnings = append(warnings, fmt.Sprintf("%s: %v", full, err))
				continue
			}
			name := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
			if err := ValidTemplateName(name, nil); err != nil {
				warnings = append(warnings, fmt.Sprintf("%s: %v", full, err))
				continue
			}
			prompts = append(prompts, ArchivedPrompt{Name: name, Target: target, Body: body})
		}
	}
	sort.SliceStable(prompts, func(i, j int) bool {
		if prompts[i].Target != prompts[j].Target {
			return prompts[i].Target < prompts[j].Target
		}
		return prompts[i].Name < prompts[j].Name
	})
	return prompts, warnings, nil
}

// WriteArchive writes prompts as a gzipped tar.
func WriteArchive(out io.Writer, prompts []ArchivedPrompt, now time.Time) error {
	gzipWriter := gzip.NewWriter(out)
	tarWriter := tar.NewWriter(gzipWriter)

	manifest := ArchiveManifest{
		Version: archiveVersion,
		Tool:    "goshcoder",
		Created: now.UTC().Format(time.RFC3339),
		Prompts: len(prompts),
	}
	encoded, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	if err := writeArchiveFile(tarWriter, manifestName, append(encoded, '\n'), now); err != nil {
		return err
	}
	for _, prompt := range prompts {
		name := path.Join(scopeDir(prompt.Target), prompt.Name+".md")
		if err := writeArchiveFile(tarWriter, name, []byte(prompt.Body), now); err != nil {
			return err
		}
	}
	if err := tarWriter.Close(); err != nil {
		return err
	}
	return gzipWriter.Close()
}

func writeArchiveFile(writer *tar.Writer, name string, content []byte, now time.Time) error {
	header := &tar.Header{
		Name:     name,
		Mode:     0o600,
		Size:     int64(len(content)),
		ModTime:  now,
		Typeflag: tar.TypeReg,
	}
	if err := writer.WriteHeader(header); err != nil {
		return err
	}
	_, err := writer.Write(content)
	return err
}

// ReadArchive parses a prompt archive.
//
// Every member name is reduced to its base name and re-validated as a template
// name, so a member called "../../.ssh/authorized_keys" becomes an invalid
// name and is refused rather than escaping the prompt directory. Nothing that
// is not a regular file is accepted, which rules out symlink and hardlink
// members pointing outside the destination.
func ReadArchive(in io.Reader) ([]ArchivedPrompt, ArchiveManifest, []string, error) {
	var manifest ArchiveManifest
	var prompts []ArchivedPrompt
	var warnings []string

	gzipReader, err := gzip.NewReader(in)
	if err != nil {
		return nil, manifest, nil, fmt.Errorf("this is not a prompt archive: %w", err)
	}
	defer gzipReader.Close()

	// Bound the decompressed stream: a small archive can expand to gigabytes.
	limited := &countingReader{reader: io.LimitReader(gzipReader, maxArchiveBytes+1), limit: maxArchiveBytes}
	tarReader := tar.NewReader(limited)

	for entries := 0; ; entries++ {
		if entries > maxArchiveEntries {
			return nil, manifest, warnings, fmt.Errorf("this archive holds more than %d entries", maxArchiveEntries)
		}
		header, err := tarReader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, manifest, warnings, fmt.Errorf("read the archive: %w", err)
		}
		if header.Typeflag == tar.TypeDir {
			continue
		}
		if header.Typeflag != tar.TypeReg {
			warnings = append(warnings, fmt.Sprintf(
				"%s is not a regular file and was skipped", path.Clean(header.Name)))
			continue
		}
		content, err := io.ReadAll(tarReader)
		if err != nil {
			return nil, manifest, warnings, fmt.Errorf("read %s from the archive: %w", header.Name, err)
		}
		if limited.exceeded {
			return nil, manifest, warnings, fmt.Errorf("this archive expands beyond %d bytes", maxArchiveBytes)
		}

		clean := path.Clean(strings.ReplaceAll(header.Name, `\`, "/"))
		if clean == manifestName {
			if err := json.Unmarshal(content, &manifest); err != nil {
				warnings = append(warnings, "the archive manifest could not be read")
			}
			continue
		}

		// The member's directory selects the scope; anything else is ignored,
		// since the name is not trusted to describe a location on disk.
		target := SaveUser
		switch {
		case strings.HasPrefix(clean, projectArchiveDi+"/"):
			target = SaveProject
		case strings.HasPrefix(clean, userArchiveDir+"/"):
			target = SaveUser
		default:
			warnings = append(warnings, fmt.Sprintf("%s is not in a recognized archive directory and was skipped", clean))
			continue
		}

		base := path.Base(clean)
		if strings.ToLower(path.Ext(base)) != ".md" {
			warnings = append(warnings, fmt.Sprintf("%s is not a Markdown prompt and was skipped", clean))
			continue
		}
		name := strings.TrimSuffix(base, path.Ext(base))
		if err := ValidTemplateName(name, nil); err != nil {
			warnings = append(warnings, fmt.Sprintf("%s was skipped: %v", clean, err))
			continue
		}
		if len(content) > maxResourceBytes {
			warnings = append(warnings, fmt.Sprintf("%s is larger than the %d-byte limit and was skipped", clean, maxResourceBytes))
			continue
		}
		prompts = append(prompts, ArchivedPrompt{Name: name, Target: target, Body: string(content)})
	}

	if manifest.Version > archiveVersion {
		return nil, manifest, warnings, fmt.Errorf(
			"this archive is version %d; this build understands version %d", manifest.Version, archiveVersion)
	}
	return prompts, manifest, warnings, nil
}

// countingReader tracks how much has been read so an over-limit stream can be
// reported as such rather than as a truncated archive.
type countingReader struct {
	reader   io.Reader
	limit    int64
	read     int64
	exceeded bool
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.reader.Read(p)
	c.read += int64(n)
	if c.read > c.limit {
		c.exceeded = true
	}
	return n, err
}

// RestoreOutcome records what happened to one prompt during a restore.
type RestoreOutcome struct {
	Name    string
	Target  SaveTarget
	Path    string
	Skipped bool
	Reason  string
}

// RestoreOptions configures RestorePrompts.
type RestoreOptions struct {
	// Overwrite replaces a prompt that already exists. Off by default: a
	// restore that silently replaced local edits would be unrecoverable.
	Overwrite bool
	// DryRun reports what would happen without writing anything.
	DryRun bool
	// Reserved names that may not be taken, as for SaveTemplate.
	Reserved []string
	// Only restores just these names, when non-empty.
	Only []string
}

// RestorePrompts writes archived prompts back to disk.
func RestorePrompts(cwd, agentDir string, prompts []ArchivedPrompt, options RestoreOptions) ([]RestoreOutcome, error) {
	wanted := map[string]bool{}
	for _, name := range options.Only {
		wanted[name] = true
	}

	outcomes := make([]RestoreOutcome, 0, len(prompts))
	for _, prompt := range prompts {
		outcome := RestoreOutcome{Name: prompt.Name, Target: prompt.Target}
		if len(wanted) > 0 && !wanted[prompt.Name] {
			continue
		}
		if err := ValidTemplateName(prompt.Name, options.Reserved); err != nil {
			outcome.Skipped, outcome.Reason = true, err.Error()
			outcomes = append(outcomes, outcome)
			continue
		}

		path := filepath.Join(TemplateDir(cwd, agentDir, prompt.Target), prompt.Name+".md")
		outcome.Path = path
		if info, err := os.Lstat(path); err == nil {
			if info.Mode()&os.ModeSymlink != 0 {
				outcome.Skipped, outcome.Reason = true, "the destination is a symbolic link"
				outcomes = append(outcomes, outcome)
				continue
			}
			if !options.Overwrite {
				outcome.Skipped, outcome.Reason = true, "a prompt with that name already exists"
				outcomes = append(outcomes, outcome)
				continue
			}
		}
		if options.DryRun {
			outcomes = append(outcomes, outcome)
			continue
		}
		if err := atomicfile.Write(path, []byte(prompt.Body), 0o600); err != nil {
			outcome.Skipped, outcome.Reason = true, err.Error()
			outcomes = append(outcomes, outcome)
			continue
		}
		outcomes = append(outcomes, outcome)
	}
	return outcomes, nil
}
