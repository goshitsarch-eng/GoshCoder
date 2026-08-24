package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestDevVersionIsConsistent keeps the four spellings of the development
// version in step. Three of them are outside Go: the Makefile, install.sh and
// install.ps1 each carry their own fallback for a build with no reachable tag,
// so bumping only version.go leaves `make build` and both installers stamping
// the previous release's number onto the same source.
func TestDevVersionIsConsistent(t *testing.T) {
	if !strings.HasSuffix(Version, "-dev") {
		t.Fatalf("Version = %q, want a -dev value: the release build stamps the real one with -ldflags", Version)
	}
	for _, file := range []string{"Makefile", "install.sh", "install.ps1"} {
		path := filepath.Join("..", "..", file)
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		if !strings.Contains(string(content), Version) {
			t.Errorf("%s does not carry the in-tree version %q; its untagged builds would report an older release", file, Version)
		}
	}
}
