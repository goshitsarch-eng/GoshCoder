// Terminal input for secrets: API keys and OAuth codes.
package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	charmterm "github.com/charmbracelet/x/term"
)

// readSecretFrom reads one secret from in without echoing it.
//
// A key pasted at a visible prompt survives in the terminal's scrollback and in
// anything recording it -- tmux capture-pane, iTerm2 session logs, script(1), a
// screen share -- so a provider key is recoverable long after entry by anyone
// who scrolls up. Echo suppression needs a real terminal, so a piped or
// redirected stdin (the scripted `goshcoder auth set x < key.txt` case) falls
// back to a plain line read, where there is no terminal to echo to anyway.
//
// prompt is written to out; it is suppressed when stdin is not a terminal so
// that scripted use produces clean output.
func readSecretFrom(in *os.File, out io.Writer, prompt string) (string, error) {
	if in == nil {
		return "", errors.New("no input stream")
	}
	if !charmterm.IsTerminal(in.Fd()) {
		line, err := bufio.NewReader(in).ReadString('\n')
		if err != nil && line == "" {
			return "", err
		}
		return strings.TrimSpace(line), nil
	}

	fmt.Fprint(out, prompt)
	secret, err := charmterm.ReadPassword(in.Fd())
	if err != nil {
		// Some terminals (notably older Windows consoles) cannot disable echo.
		// Fall back to a visible read rather than blocking the user out of
		// authenticating, but say plainly that the key will be on screen.
		fmt.Fprintln(out, "\n(this terminal cannot hide input; the key will be visible)")
		fmt.Fprint(out, prompt)
		line, readErr := bufio.NewReader(in).ReadString('\n')
		if readErr != nil && line == "" {
			return "", readErr
		}
		return strings.TrimSpace(line), nil
	}
	// ReadPassword consumes the Enter key without echoing it, so the cursor is
	// still on the prompt line.
	fmt.Fprintln(out)
	return strings.TrimSpace(string(secret)), nil
}

// readSecret prompts on stderr and reads one secret from stdin.
func readSecret(prompt string) (string, error) {
	return readSecretFrom(os.Stdin, os.Stderr, prompt)
}
