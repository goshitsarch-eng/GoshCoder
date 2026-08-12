//go:build !linux

package tui

import "os"

type terminalState struct{}

func isTerminal(*os.File) bool                 { return false }
func makeRaw(*os.File) (*terminalState, error) { return &terminalState{}, nil }
func (*terminalState) restore() error          { return nil }
func terminalSize(*os.File) (int, int)         { return 80, 24 }
