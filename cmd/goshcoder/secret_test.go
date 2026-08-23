package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/llm/catalog"
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
