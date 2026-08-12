//go:build linux

package tui

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

type terminalState struct {
	fd  uintptr
	old syscall.Termios
}

type windowSize struct {
	Rows, Cols uint16
	X, Y       uint16
}

func isTerminal(file *os.File) bool {
	var termios syscall.Termios
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, file.Fd(), syscall.TCGETS, uintptr(unsafe.Pointer(&termios)), 0, 0, 0)
	return errno == 0
}

func makeRaw(file *os.File) (*terminalState, error) {
	state := &terminalState{fd: file.Fd()}
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, state.fd, syscall.TCGETS, uintptr(unsafe.Pointer(&state.old)), 0, 0, 0); errno != 0 {
		return nil, fmt.Errorf("read terminal settings: %w", errno)
	}
	raw := state.old
	raw.Iflag &^= syscall.BRKINT | syscall.ICRNL | syscall.INPCK | syscall.ISTRIP | syscall.IXON
	raw.Oflag |= syscall.OPOST
	raw.Cflag |= syscall.CS8
	raw.Lflag &^= syscall.ECHO | syscall.ICANON | syscall.IEXTEN | syscall.ISIG
	raw.Cc[syscall.VMIN] = 1
	raw.Cc[syscall.VTIME] = 0
	if _, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, state.fd, syscall.TCSETS, uintptr(unsafe.Pointer(&raw)), 0, 0, 0); errno != 0 {
		return nil, fmt.Errorf("enable raw terminal mode: %w", errno)
	}
	return state, nil
}

func (state *terminalState) restore() error {
	if state == nil {
		return nil
	}
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, state.fd, syscall.TCSETS, uintptr(unsafe.Pointer(&state.old)), 0, 0, 0)
	if errno != 0 {
		return fmt.Errorf("restore terminal settings: %w", errno)
	}
	return nil
}

func terminalSize(file *os.File) (int, int) {
	var size windowSize
	_, _, errno := syscall.Syscall6(syscall.SYS_IOCTL, file.Fd(), syscall.TIOCGWINSZ, uintptr(unsafe.Pointer(&size)), 0, 0, 0)
	if errno != 0 || size.Cols == 0 || size.Rows == 0 {
		return 80, 24
	}
	return int(size.Cols), int(size.Rows)
}
