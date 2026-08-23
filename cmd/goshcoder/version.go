// Build metadata for the goshcoder binary.
//
// These are variables rather than constants so that release builds can stamp
// them with -ldflags "-X main.Version=... -X main.Commit=... -X main.BuildDate=...".
// The linker cannot rewrite a Go constant, so a const would silently ignore the
// stamp and ship every release claiming to be the development version.
package main

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"strings"
)

var (
	// Version is the CLI version, stamped at release time.
	Version = "0.2.0-dev"
	// Commit is the git revision the binary was built from.
	Commit = ""
	// BuildDate is the RFC 3339 build timestamp.
	BuildDate = ""
)

// versionInfo renders the full version banner printed by `goshcoder version`.
//
// When the binary was built with `go install` rather than the release
// pipeline, Commit and BuildDate are empty; Go's embedded VCS stamps are used
// as a fallback so that even ad-hoc builds identify their source revision.
func versionInfo() string {
	commit, date, dirty := Commit, BuildDate, false
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			switch setting.Key {
			case "vcs.revision":
				if commit == "" {
					commit = setting.Value
				}
			case "vcs.time":
				if date == "" {
					date = setting.Value
				}
			case "vcs.modified":
				dirty = setting.Value == "true"
			}
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "goshcoder %s", Version)
	if commit != "" {
		short := commit
		if len(short) > 12 {
			short = short[:12]
		}
		fmt.Fprintf(&b, " (%s", short)
		if dirty {
			b.WriteString("-dirty")
		}
		b.WriteString(")")
	}
	b.WriteString("\n")
	if date != "" {
		fmt.Fprintf(&b, "built:    %s\n", date)
	}
	fmt.Fprintf(&b, "go:       %s\n", runtime.Version())
	fmt.Fprintf(&b, "platform: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	return b.String()
}
