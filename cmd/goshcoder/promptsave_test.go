package main

import (
	"strings"
	"testing"

	"goshcoder/internal/resources"
)

// TestReservedNamesCoverBothDispatchers covers the shadowing hazard end to end.
// A template is expanded before the slash dispatcher runs in BOTH interfaces, so
// a set built only from the fullscreen palette let a saved prompt shadow any
// command the palette does not advertise.
func TestReservedNamesCoverBothDispatchers(t *testing.T) {
	reserved := reservedCommandNames(nil)

	// Names the fullscreen palette does list.
	for _, name := range []string{"model", "compact", "clear", "resume"} {
		if err := resources.ValidTemplateName(name, reserved); err == nil {
			t.Fatalf("a prompt named %q was accepted; it would shadow the built-in", name)
		}
	}
	// Names only the line-oriented dispatcher handles, plus the aliases.
	for _, name := range []string{"exit", "quit", "help", "logout", "plannotator", "planner"} {
		if err := resources.ValidTemplateName(name, reserved); err == nil {
			t.Fatalf("a prompt named %q was accepted; it would shadow the built-in", name)
		}
	}
	// Negative control: an ordinary name must still be allowed, or the guard is
	// passing by refusing everything.
	for _, name := range []string{"review", "ship-it", "my.notes", "models2"} {
		if err := resources.ValidTemplateName(name, reserved); err != nil {
			t.Fatalf("ValidTemplateName(%q) rejected an ordinary name: %v", name, err)
		}
	}
}

// TestReservedNamesAreBuiltFromTheLiveTables keeps the set from drifting as
// commands are added: it must be derived, not copied.
func TestReservedNamesAreBuiltFromTheLiveTables(t *testing.T) {
	reserved := strings.Join(reservedCommandNames(nil), " ")
	for _, command := range fullscreenSlashCommands {
		bare := strings.TrimPrefix(command.name, "/")
		if !strings.Contains(reserved+" ", bare+" ") {
			t.Fatalf("the palette lists /%s but it is not reserved", bare)
		}
	}
	for alias := range chatCommandAliases {
		bare := strings.TrimPrefix(alias, "/")
		if !strings.Contains(reserved+" ", bare+" ") {
			t.Fatalf("the alias table lists %s but it is not reserved", alias)
		}
	}
}
