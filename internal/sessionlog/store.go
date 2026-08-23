package sessionlog

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"goshcoder/internal/lockfile"
)

// Sentinel errors for session resolution.
var (
	ErrNotFound  = errors.New("sessionlog: no matching session")
	ErrAmbiguous = errors.New("sessionlog: session id prefix matches more than one session")
	// ErrLegacyFormat is returned when a pre-v3 pi session is opened for
	// appending. It can be read and forked, but not continued in place.
	ErrLegacyFormat = errors.New("sessionlog: session predates this format and must be forked to continue")
)

// maxSearchTextBytes bounds the transcript text retained per session for the
// resume picker. Sessions can reach hundreds of megabytes once images are in
// them; holding all of that to answer a search box is not a trade worth making.
const maxSearchTextBytes = 64 << 10

// Store is a rooted collection of session files, sharded by the working
// directory a session was started in.
type Store struct{ Root string }

// NewStore returns a store rooted at dir.
func NewStore(dir string) *Store { return &Store{Root: dir} }

// DirName encodes a working directory into the shard name pi uses:
// `--path-with-separators-replaced--`. Byte-identical to pi's, so the two
// tools address the same directories.
//
// The encoding is lossy -- /a/b and /a-b collide -- which is why every listing
// filters on the cwd recorded in each header rather than trusting the name.
func DirName(cwd string) string {
	resolved := cwd
	if abs, err := filepath.Abs(cwd); err == nil {
		resolved = abs
	}
	trimmed := strings.TrimLeft(resolved, `/\`)
	replaced := strings.NewReplacer("/", "-", `\`, "-", ":", "-").Replace(trimmed)
	return "--" + replaced + "--"
}

// Dir returns the shard directory for cwd.
func (s *Store) Dir(cwd string) string { return filepath.Join(s.Root, DirName(cwd)) }

// LoadReport carries what a load had to work around. Deviation from pi, which
// drops malformed lines silently: surfacing them costs nothing and turns
// invisible data loss into something a user can act on.
type LoadReport struct {
	SkippedLines int
	Warnings     []string
	// RepairedTail is set when an unterminated final line was removed so the
	// next append would not concatenate onto it.
	RepairedTail bool
	// Migrated is set when the file was read as a pre-v3 pi session. Such a
	// file cannot be appended to -- see ErrLegacyFormat.
	Migrated bool
	// SourceVersion is the format the file declared on disk.
	SourceVersion int
}

// Info summarizes one session for listing and for the resume picker.
type Info struct {
	ID           string
	Path         string
	Cwd          string
	Name         string
	FirstMessage string
	Created      time.Time
	Modified     time.Time
	Messages     int
	// Cleared counts messages before the most recent transcript reset. They
	// remain in the file; a listing that ignored them would report "2 messages"
	// for a session holding hundreds.
	Cleared int
	Size    int64
	// SearchText is a bounded concatenation of the transcript's text, so the
	// picker can match on what was discussed rather than only on metadata.
	SearchText string
	Locked     bool
	Owner      lockfile.Owner
}

// Title returns the best available human label for a session.
func (i Info) Title() string {
	if i.Name != "" {
		return i.Name
	}
	if i.FirstMessage != "" {
		return i.FirstMessage
	}
	return i.ID
}

// shortIDLength is the shortest id prefix ever displayed.
const shortIDLength = 8

// ShortID returns the id prefix used in listings and messages.
//
// Eight characters is only a floor. uuidv7 puts the millisecond timestamp
// first, so two sessions created in the same second share their first eight
// characters exactly -- displaying that prefix would offer the user an
// identifier that cannot select what it names. Use ShortIDs when rendering a
// list, which widens the prefix until it distinguishes the sessions on screen.
func (i Info) ShortID() string {
	if len(i.ID) >= shortIDLength {
		return i.ID[:shortIDLength]
	}
	return i.ID
}

// ShortIDs returns a display prefix for each session, widened as needed so no
// two collide. Same idea as an abbreviated git hash: short enough to read,
// long enough to be an identifier.
func ShortIDs(sessions []Info) []string {
	length := shortIDLength
	for {
		seen := make(map[string]bool, len(sessions))
		collision := false
		longest := 0
		for _, info := range sessions {
			if len(info.ID) > longest {
				longest = len(info.ID)
			}
			prefix := info.ID
			if len(prefix) > length {
				prefix = prefix[:length]
			}
			if seen[prefix] {
				collision = true
				break
			}
			seen[prefix] = true
		}
		if !collision || length >= longest {
			break
		}
		length += 4
	}
	out := make([]string, len(sessions))
	for index, info := range sessions {
		if len(info.ID) > length {
			out[index] = info.ID[:length]
			continue
		}
		out[index] = info.ID
	}
	return out
}

// Create starts a new session for cwd. parent names the session this one was
// forked from, and is empty otherwise.
func (s *Store) Create(cwd, parent string) (*Writer, error) {
	id, err := NewSessionID()
	if err != nil {
		return nil, err
	}
	return s.createWithID(cwd, parent, id)
}

// CreateWithID starts a new session with a caller-supplied id, which is how
// -session names a session that does not exist yet.
func (s *Store) CreateWithID(cwd, parent, id string) (*Writer, error) {
	if err := ValidateSessionID(id); err != nil {
		return nil, err
	}
	return s.createWithID(cwd, parent, id)
}

func (s *Store) createWithID(cwd, parent, id string) (*Writer, error) {
	absCwd := cwd
	if abs, err := filepath.Abs(cwd); err == nil {
		absCwd = abs
	}
	dir := s.Dir(absCwd)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("sessionlog: create %s: %w", dir, err)
	}

	header := Header{
		Type:          headerType,
		Version:       FormatVersion,
		ID:            id,
		Timestamp:     Now(),
		Cwd:           absCwd,
		ParentSession: parent,
	}
	path := filepath.Join(dir, FilenameStamp(header.Timestamp)+"_"+id+".jsonl")

	release, owner, err := claim(path)
	if err != nil {
		if errors.Is(err, ErrBusy) {
			return nil, fmt.Errorf("%w (held by %s)", ErrBusy, owner)
		}
		return nil, err
	}

	// O_EXCL: a session id is supposed to be new here, and silently appending
	// to someone else's file would be worse than failing.
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		release()
		return nil, fmt.Errorf("sessionlog: create %s: %w", path, err)
	}
	line, err := EncodeHeader(header)
	if err != nil {
		file.Close()
		release()
		return nil, err
	}
	written, err := file.Write(line)
	if err != nil {
		file.Close()
		_ = os.Remove(path)
		release()
		return nil, fmt.Errorf("sessionlog: write header to %s: %w", path, err)
	}
	return &Writer{
		file: file, path: path, header: header,
		tree: NewTree(), release: release, size: int64(written),
	}, nil
}

// Load reads a session without claiming it. Used for listing, export, and
// read-only inspection.
func (s *Store) Load(path string) (*Tree, Header, LoadReport, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, Header{}, LoadReport{}, fmt.Errorf("sessionlog: open %s: %w", path, err)
	}
	defer file.Close()
	tree, header, report, _, err := readInto(file, path)
	return tree, header, report, err
}

// readInto folds a session file. It returns the byte offset of the end of the
// last complete line, so a caller with write access can cut off an
// unterminated tail before appending to it.
func readInto(file *os.File, path string) (*Tree, Header, LoadReport, int64, error) {
	var report LoadReport
	reader := bufio.NewReaderSize(file, 64<<10)

	// readLine returns a whole line including its terminator, or the trailing
	// bytes with io.EOF when the file does not end in a newline. Reading
	// through ErrBufferFull rather than using bufio.Scanner is what lets one
	// oversized line be reported and skipped instead of ending the scan.
	readLine := func() ([]byte, error) {
		var line []byte
		for {
			chunk, err := reader.ReadSlice('\n')
			line = append(line, chunk...)
			if len(line) > MaxEntryBytes {
				// Drain the rest of this line so the next call starts on the
				// next one.
				for err == bufio.ErrBufferFull {
					_, err = reader.ReadSlice('\n')
				}
				return line, ErrTooLarge
			}
			if err == nil || !errors.Is(err, bufio.ErrBufferFull) {
				return line, err
			}
		}
	}

	first, err := readLine()
	if len(first) == 0 {
		return nil, Header{}, report, 0, fmt.Errorf("sessionlog: %s is empty", path)
	}
	if err != nil && !errors.Is(err, io.EOF) && !errors.Is(err, ErrTooLarge) {
		return nil, Header{}, report, 0, fmt.Errorf("sessionlog: read %s: %w", path, err)
	}
	header, headerErr := DecodeHeader(trimLine(first))
	if headerErr != nil {
		if errors.Is(headerErr, ErrVersionTooNew) {
			return nil, Header{}, report, 0, fmt.Errorf("%w: %s", ErrVersionTooNew, path)
		}
		return nil, Header{}, report, 0, fmt.Errorf("sessionlog: %s is not a session file: %s", path, headerErr)
	}

	// A pi session older than v3 is brought up to date in memory; the file on
	// disk is never rewritten. See migrate.go for why.
	migration := newMigrator(header.Version)
	report.SourceVersion = header.Version
	report.Migrated = migration.needed()
	if migration.needed() {
		report.Warnings = append(report.Warnings,
			fmt.Sprintf("session is format v%d; it was read as v%d without being rewritten", header.Version, FormatVersion))
	}

	// offset tracks complete lines only. A skipped-but-complete line still
	// counts: it is pi-compatible content that a repair must not delete.
	var offset int64
	if complete(first) {
		offset = int64(len(first))
	}

	tree := NewTree()
	lineNumber := 1
	for {
		raw, readErr := readLine()
		if len(raw) > 0 {
			lineNumber++
			whole := complete(raw)
			if whole {
				offset += int64(len(raw))
			}
			body := trimLine(raw)
			switch {
			case errors.Is(readErr, ErrTooLarge):
				report.SkippedLines++
				report.Warnings = append(report.Warnings,
					fmt.Sprintf("line %d exceeds the %d-byte entry limit and was skipped", lineNumber, MaxEntryBytes))
			case len(body) == 0:
				// Blank lines are not entries; pi ignores them too.
			default:
				entry, decodeErr := decodeEntry(body, !migration.needed())
				if decodeErr == nil && migration.needed() {
					decodeErr = migration.apply(&entry, body)
				}
				switch {
				case decodeErr != nil && !whole:
					// An unterminated final line is a torn append: the process
					// died between issuing the write and its completion.
					report.RepairedTail = true
				case decodeErr != nil:
					report.SkippedLines++
					report.Warnings = append(report.Warnings,
						fmt.Sprintf("line %d %s and was skipped", lineNumber, decodeErr))
				default:
					if addErr := tree.Add(entry, raw); addErr != nil {
						report.SkippedLines++
						report.Warnings = append(report.Warnings,
							fmt.Sprintf("line %d %s and was skipped", lineNumber, addErr))
					}
				}
			}
		}
		if readErr != nil {
			if errors.Is(readErr, ErrTooLarge) {
				continue
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, Header{}, report, 0, fmt.Errorf("sessionlog: read %s: %w", path, readErr)
		}
	}
	if migration.needed() {
		migration.finish(tree)
		header.Version = FormatVersion
	}
	return tree, header, report, offset, nil
}

func complete(raw []byte) bool { return len(raw) > 0 && raw[len(raw)-1] == '\n' }

func trimLine(raw []byte) []byte {
	body := raw
	for len(body) > 0 && (body[len(body)-1] == '\n' || body[len(body)-1] == '\r') {
		body = body[:len(body)-1]
	}
	return body
}

// Attach opens a session for appending: it claims the file, folds it, and
// repairs an unterminated tail so the next append starts on a fresh line.
func (s *Store) Attach(path string) (*Writer, LoadReport, error) {
	release, owner, err := claim(path)
	if err != nil {
		if errors.Is(err, ErrBusy) {
			return nil, LoadReport{}, fmt.Errorf("%w (held by %s)", ErrBusy, owner)
		}
		return nil, LoadReport{}, err
	}
	file, err := os.OpenFile(path, os.O_RDWR, 0o600)
	if err != nil {
		release()
		return nil, LoadReport{}, fmt.Errorf("sessionlog: open %s: %w", path, err)
	}
	tree, header, report, offset, err := readInto(file, path)
	if err != nil {
		file.Close()
		release()
		return nil, report, err
	}
	// A pre-v3 file carries no entry ids, so identity is reconstructed from
	// file order on every read. Appending a v3 entry to it would leave a file
	// that migrates differently next time -- the new entry's id would be
	// reassigned and its children orphaned. Fork it instead.
	if report.Migrated {
		file.Close()
		release()
		return nil, report, fmt.Errorf("%w: %s is format v%d", ErrLegacyFormat, path, report.SourceVersion)
	}
	// Cut the file back to the last complete line. Without this, the next
	// append concatenates onto a partial entry and corrupts both.
	if info, statErr := file.Stat(); statErr == nil && info.Size() != offset {
		if truncErr := file.Truncate(offset); truncErr != nil {
			file.Close()
			release()
			return nil, report, fmt.Errorf("sessionlog: repair %s: %w", path, truncErr)
		}
		report.RepairedTail = true
	}
	if _, err := file.Seek(offset, io.SeekStart); err != nil {
		file.Close()
		release()
		return nil, report, fmt.Errorf("sessionlog: seek %s: %w", path, err)
	}
	return &Writer{file: file, path: path, header: header, tree: tree, release: release, size: offset}, report, nil
}

// Open reads a session without claiming it, for -read-only. Appends are
// refused rather than silently dropped.
func (s *Store) Open(path string) (*Writer, LoadReport, error) {
	tree, header, report, _, err := func() (*Tree, Header, LoadReport, int64, error) {
		file, err := os.Open(path)
		if err != nil {
			return nil, Header{}, LoadReport{}, 0, fmt.Errorf("sessionlog: open %s: %w", path, err)
		}
		defer file.Close()
		return readInto(file, path)
	}()
	if err != nil {
		return nil, report, err
	}
	return &Writer{path: path, header: header, tree: tree, readOnly: true}, report, nil
}

// ListOptions selects which sessions a listing returns.
type ListOptions struct {
	// AllWorkspaces ignores cwd and scans every shard.
	AllWorkspaces bool
	// Limit caps the result, newest first. Zero means no cap.
	Limit int
	// WithText populates Info.SearchText, which costs a full read per session.
	WithText bool
}

// List returns the sessions for cwd, newest first.
func (s *Store) List(cwd string, options ListOptions) ([]Info, error) {
	var dirs []string
	if options.AllWorkspaces {
		entries, err := os.ReadDir(s.Root)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil, nil
			}
			return nil, fmt.Errorf("sessionlog: read %s: %w", s.Root, err)
		}
		for _, entry := range entries {
			if entry.IsDir() {
				dirs = append(dirs, filepath.Join(s.Root, entry.Name()))
			}
		}
	} else {
		dirs = append(dirs, s.Dir(cwd))
	}

	targetCwd := cwd
	if abs, err := filepath.Abs(cwd); err == nil {
		targetCwd = abs
	}

	var out []Info
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, fmt.Errorf("sessionlog: read %s: %w", dir, err)
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
				continue
			}
			info, err := s.describe(filepath.Join(dir, entry.Name()), options.WithText)
			if err != nil {
				// One unreadable file must not hide every other session.
				continue
			}
			// The shard name is lossy, so the header is the authority on which
			// workspace a session belongs to.
			if !options.AllWorkspaces && info.Cwd != "" && info.Cwd != targetCwd {
				continue
			}
			out = append(out, info)
		}
	}

	sort.SliceStable(out, func(i, j int) bool {
		if !out[i].Modified.Equal(out[j].Modified) {
			return out[i].Modified.After(out[j].Modified)
		}
		return out[i].ID > out[j].ID
	})
	if options.Limit > 0 && len(out) > options.Limit {
		out = out[:options.Limit]
	}
	return out, nil
}

// describe folds one session into an Info.
func (s *Store) describe(path string, withText bool) (Info, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return Info{}, err
	}
	tree, header, _, err := s.Load(path)
	if err != nil {
		return Info{}, err
	}

	info := Info{
		ID:       header.ID,
		Path:     path,
		Cwd:      header.Cwd,
		Name:     tree.Name(),
		Created:  ParseTimestamp(header.Timestamp),
		Modified: stat.ModTime(),
		Size:     stat.Size(),
	}
	if info.Created.IsZero() {
		info.Created = stat.ModTime()
	}

	var text strings.Builder
	for _, entry := range tree.All() {
		switch entry.Type {
		case TypeTranscriptReset:
			// Everything before a reset is a previous conversation in the same
			// file. Counting it as current would misreport the session.
			info.Cleared += info.Messages
			info.Messages = 0
			info.FirstMessage = ""
			continue
		case TypeMessage:
		default:
			continue
		}
		info.Messages++
		role := messageRole(entry.Message)
		snippet := messageText(entry.Message)
		if info.FirstMessage == "" && role == "user" && snippet != "" {
			info.FirstMessage = firstLine(snippet, 120)
		}
		if withText && text.Len() < maxSearchTextBytes && snippet != "" {
			remaining := maxSearchTextBytes - text.Len()
			if len(snippet) > remaining {
				snippet = snippet[:remaining]
			}
			text.WriteString(snippet)
			text.WriteByte('\n')
		}
	}
	info.SearchText = text.String()

	// Probing the claim is advisory display only, so a probe that wins must
	// release immediately -- otherwise merely listing sessions would leave a
	// lock file behind for every one of them.
	if release, owner, ok, lockErr := lockfile.TryAcquire(path+".lock", lockfile.SessionTiming()); lockErr == nil {
		if ok {
			release()
		} else {
			info.Locked = true
			info.Owner = owner
		}
	}
	return info, nil
}

// messageText extracts a plain-text preview from a message payload. It reads
// only what it needs: listing runs over every entry of every session.
func messageText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var shaped struct {
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &shaped); err != nil {
		return ""
	}
	if len(shaped.Content) == 0 {
		return ""
	}
	var asString string
	if err := json.Unmarshal(shaped.Content, &asString); err == nil {
		return asString
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(shaped.Content, &blocks); err != nil {
		return ""
	}
	var out strings.Builder
	for _, block := range blocks {
		if block.Type == "text" && block.Text != "" {
			if out.Len() > 0 {
				out.WriteByte(' ')
			}
			out.WriteString(block.Text)
		}
	}
	return out.String()
}

func firstLine(text string, limit int) string {
	trimmed := strings.TrimSpace(text)
	if index := strings.IndexAny(trimmed, "\r\n"); index >= 0 {
		trimmed = trimmed[:index]
	}
	runes := []rune(trimmed)
	if len(runes) > limit {
		return strings.TrimSpace(string(runes[:limit])) + "…"
	}
	return trimmed
}

// MostRecent returns the newest session for cwd.
func (s *Store) MostRecent(cwd string) (Info, bool, error) {
	sessions, err := s.List(cwd, ListOptions{Limit: 1})
	if err != nil || len(sessions) == 0 {
		return Info{}, false, err
	}
	return sessions[0], true, nil
}

// Resolve turns a user-supplied reference into a session. A reference is a
// path, an exact id, or an unambiguous id prefix; ids are looked for in this
// workspace first and then everywhere, because a user who typed an id knows
// which session they mean even if they changed directory.
func (s *Store) Resolve(cwd, ref string) (Info, error) {
	if ref == "" {
		return Info{}, ErrNotFound
	}
	if strings.HasSuffix(ref, ".jsonl") || strings.ContainsAny(ref, `/\`) {
		path := ref
		if abs, err := filepath.Abs(ref); err == nil {
			path = abs
		}
		info, err := s.describe(path, false)
		if err != nil {
			return Info{}, fmt.Errorf("%w: %s", ErrNotFound, ref)
		}
		return info, nil
	}

	for _, all := range []bool{false, true} {
		sessions, err := s.List(cwd, ListOptions{AllWorkspaces: all})
		if err != nil {
			return Info{}, err
		}
		var matches []Info
		for _, candidate := range sessions {
			if candidate.ID == ref {
				return candidate, nil
			}
			if strings.HasPrefix(candidate.ID, ref) {
				matches = append(matches, candidate)
			}
		}
		if len(matches) == 1 {
			return matches[0], nil
		}
		if len(matches) > 1 {
			return Info{}, fmt.Errorf("%w: %s", ErrAmbiguous, ref)
		}
	}
	return Info{}, fmt.Errorf("%w: %s", ErrNotFound, ref)
}

// Remove deletes a session file. A claimed session is refused: deleting the
// file out from under a running process would leave it appending to an unlinked
// inode and silently losing everything after.
func (s *Store) Remove(info Info) error {
	release, owner, err := claim(info.Path)
	if err != nil {
		if errors.Is(err, ErrBusy) {
			return fmt.Errorf("%w (held by %s)", ErrBusy, owner)
		}
		return err
	}
	defer release()
	if err := os.Remove(info.Path); err != nil {
		return fmt.Errorf("sessionlog: remove %s: %w", info.Path, err)
	}
	return nil
}

// Fork copies a session up to and including at into a new one.
//
// Deviation from pi's createBranchedSession, which strips label entries and
// re-chains parentId around the entries it removed: copying the raw lines
// preserves any field a newer writer added, and keeping the labels means
// nothing needs re-chaining. Both files remain valid v3.
func (s *Store) Fork(src Info, at *string, targetCwd string) (*Writer, error) {
	tree, header, report, err := s.Load(src.Path)
	if err != nil {
		return nil, err
	}
	path := tree.Path(at)
	if len(path) == 0 && at != nil {
		return nil, fmt.Errorf("sessionlog: no entry %s in session %s", *at, src.ShortID())
	}
	if targetCwd == "" {
		targetCwd = header.Cwd
	}

	writer, err := s.Create(targetCwd, header.ID)
	if err != nil {
		return nil, err
	}
	// Labels reference entries by id, so they only survive if they are copied
	// alongside the path they annotate.
	onPath := make(map[string]bool, len(path))
	for _, entry := range path {
		onPath[entry.ID] = true
	}
	for _, entry := range tree.All() {
		if !onPath[entry.ID] && !(entry.Type == TypeLabel && onPath[entry.TargetID]) {
			continue
		}
		raw, ok := tree.Raw(entry.ID)
		if !ok {
			continue
		}
		// A migrated source's raw lines carry no ids: the ids exist only in
		// the fold. Copying those bytes would produce a v3 file whose entries
		// cannot be referenced, so the fork re-encodes them instead.
		if report.Migrated {
			if raw, err = EncodeEntry(entry); err != nil {
				writer.Close()
				return nil, err
			}
		}
		if err := writer.appendRaw(entry, raw); err != nil {
			writer.Close()
			return nil, err
		}
	}
	if err := writer.Sync(); err != nil {
		writer.Close()
		return nil, err
	}
	return writer, nil
}

// appendRaw writes a line verbatim, preserving its ids and timestamps. Used
// only by Fork; every other path stamps a fresh identity.
func (w *Writer) appendRaw(e Entry, raw []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed || w.readOnly {
		return errors.New("sessionlog: session is not writable")
	}
	if w.degraded != nil {
		return w.degraded
	}
	line := raw
	if len(line) == 0 || line[len(line)-1] != '\n' {
		line = append(append([]byte(nil), line...), '\n')
	}
	written, err := w.file.Write(line)
	w.size += int64(written)
	if err != nil {
		w.stopLocked(fmt.Errorf("sessionlog: append to %s: %w", w.path, err))
		return w.degraded
	}
	return w.tree.Add(e, line)
}
