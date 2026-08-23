// Package atomicfile writes a file by staging a sibling temporary and
// renaming it into place, so a reader never observes a partial write and a
// crash mid-write leaves the previous content intact.
//
// The idiom was repeated in five packages before this one existed
// (internal/llm/catalog, internal/plannotator, internal/omniroute,
// internal/btw, internal/ralph); each copy differed in whether it chmodded the
// temporary before writing and whether it fsynced before the rename. Both
// matter: the chmod closes the window in which a 0600-destined file exists at
// the process umask, and the fsync is what makes the rename a durability
// barrier rather than just an atomicity one.
//
// The containing directory is deliberately not fsynced, matching every other
// durable write in this tree. A completed write therefore survives a process
// crash but not necessarily a power loss.
package atomicfile

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
)

// Write replaces path with data. The parent directory is created 0700 if it
// does not exist, since every caller here writes into the agent directory.
func Write(path string, data []byte, perm fs.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("atomicfile: create %s: %w", dir, err)
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+"-*.tmp")
	if err != nil {
		return fmt.Errorf("atomicfile: stage %s: %w", path, err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(perm); err != nil {
		tmp.Close()
		return fmt.Errorf("atomicfile: secure %s: %w", path, err)
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("atomicfile: write %s: %w", path, err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("atomicfile: sync %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("atomicfile: close %s: %w", path, err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("atomicfile: publish %s: %w", path, err)
	}
	// Windows and some filesystems ignore the temporary's mode on rename.
	_ = os.Chmod(path, perm)
	return nil
}
