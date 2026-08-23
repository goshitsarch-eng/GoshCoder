//go:build !windows

// Process-group isolation for the bash tool.
package tools

import (
	"os/exec"
	"syscall"
)

// isolateProcessGroup makes cancelling cmd tear down its whole descendant tree.
//
// exec.CommandContext's default cancellation kills only the shell's own pid, so
// a build, test run, or dev server the shell spawned survives the timeout and
// keeps consuming CPU after the tool has already reported "command timed out".
// Putting the shell in a new process group lets a single negative-pid signal
// reach every descendant.
func isolateProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// With Setpgid the child leads a group whose id equals its pid, so
		// signalling -pid reaches the whole tree. Fall back to the single
		// process if the group is already gone.
		if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
			return cmd.Process.Kill()
		}
		return nil
	}
}
