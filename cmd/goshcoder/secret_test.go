package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"context"
	"errors"
	"goshcoder/internal/llm/catalog"
	"io"
	"runtime"
)

// TestReadSecretFromPipedInputReadsALine covers scripted use
// (`goshcoder auth set anthropic < key.txt`): there is no terminal, so the
// prompt is suppressed and the line is read verbatim.
func TestReadSecretFromPipedInputReadsALine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "key")
	if err := os.WriteFile(path, []byte("sk-ant-test-key\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	var out bytes.Buffer
	secret, err := readSecretFrom(file, &out, "Enter the API key: ")
	if err != nil {
		t.Fatalf("readSecretFrom: %v", err)
	}
	if secret != "sk-ant-test-key" {
		t.Fatalf("secret = %q", secret)
	}
	if out.Len() != 0 {
		t.Fatalf("prompted on non-terminal input: %q", out.String())
	}
}

// TestReadSecretFromNeverEchoesTheSecret is the property that matters: whatever
// path is taken, the secret must not appear in what is written to the terminal.
func TestReadSecretFromNeverEchoesTheSecret(t *testing.T) {
	path := filepath.Join(t.TempDir(), "key")
	if err := os.WriteFile(path, []byte("sk-ant-super-secret\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	var out bytes.Buffer
	if _, err := readSecretFrom(file, &out, "Enter the API key: "); err != nil {
		t.Fatalf("readSecretFrom: %v", err)
	}
	if strings.Contains(out.String(), "super-secret") {
		t.Fatalf("the secret was written to the terminal: %q", out.String())
	}
}

// TestReadSecretFromRejectsNilInput guards the error path rather than panicking
// on a nil stdin.
func TestReadSecretFromRejectsNilInput(t *testing.T) {
	if _, err := readSecretFrom(nil, &bytes.Buffer{}, "key: "); err == nil {
		t.Fatal("a nil input stream must error")
	}
}

// loginSecretPrompt builds the prompt shape used for a pasted OAuth code.
func loginSecretPrompt() catalog.LoginPrompt {
	return catalog.LoginPrompt{Kind: catalog.PromptSecret, Message: "Paste the authorization code"}
}

// TestLoginInteractionDoesNotAnnounceVisibleInputForPipedSecrets keeps the
// "(input is visible)" warning tied to an actual terminal.
func TestLoginInteractionDoesNotAnnounceVisibleInputForPipedSecrets(t *testing.T) {
	var out bytes.Buffer
	interaction := newTerminalLoginInteraction(strings.NewReader("pasted-code\n"), &out, false)
	got, err := interaction.Prompt(loginSecretPrompt())
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got != "pasted-code" {
		t.Fatalf("prompt result = %q", got)
	}
	if strings.Contains(out.String(), "input is visible") {
		t.Fatalf("warned about visible input on a non-terminal: %q", out.String())
	}
}

// TestLoginPromptDoesNotLeakAReaderPerCancellation covers a goroutine leak that
// stole the user's keystrokes. Each cancellable prompt spawned its own
// goroutine blocked on stdin; when the context won the race the goroutine
// stayed blocked forever, and because /login runs in-process from chat, every
// orphan went on to swallow one line of the user's later input.
func TestLoginPromptDoesNotLeakAReaderPerCancellation(t *testing.T) {
	// A reader that never returns, standing in for a terminal awaiting input.
	blocked, writer := io.Pipe()
	t.Cleanup(func() { _ = writer.Close() })

	interaction := newTerminalLoginInteraction(blocked, io.Discard, false)
	t.Cleanup(interaction.Close)

	before := runtime.NumGoroutine()
	for range 20 {
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // the callback server won the race
		_, err := interaction.Prompt(catalog.LoginPrompt{
			Kind: catalog.PromptText, Message: "paste the code", Ctx: ctx,
		})
		if !errors.Is(err, catalog.ErrLoginCancelled) {
			t.Fatalf("Prompt error = %v, want ErrLoginCancelled", err)
		}
	}

	// One shared reader is expected; twenty are the bug.
	if grew := runtime.NumGoroutine() - before; grew > 3 {
		t.Fatalf("twenty cancelled prompts left %d extra goroutines blocked on stdin", grew)
	}
}

// TestLoginPromptWithoutCancellationUsesNoGoroutine keeps the common prompts --
// which cannot be cancelled -- off the goroutine path entirely.
func TestLoginPromptWithoutCancellationUsesNoGoroutine(t *testing.T) {
	interaction := newTerminalLoginInteraction(strings.NewReader("typed answer\n"), io.Discard, false)
	t.Cleanup(interaction.Close)

	before := runtime.NumGoroutine()
	got, err := interaction.Prompt(catalog.LoginPrompt{Kind: catalog.PromptText, Message: "name?"})
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	if got != "typed answer" {
		t.Fatalf("answer = %q", got)
	}
	if grew := runtime.NumGoroutine() - before; grew > 0 {
		t.Fatalf("a non-cancellable prompt started %d goroutines", grew)
	}
}
