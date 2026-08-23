//go:build windows

// Process-group isolation for the bash tool.
package tools

import (
	"os/exec"
	"strconv"
	"syscall"
)

// isolateProcessGroup makes cancelling cmd tear down its whole descendant tree.
//
// Windows has no process-group signal, so the tree is walked by taskkill /T.
// CREATE_NEW_PROCESS_GROUP still matters: it detaches the child from this
// console's Ctrl+C group, so a Ctrl+C aimed at goshcoder does not also
// interrupt a command the model is running.
func isolateProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		kill := exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(cmd.Process.Pid))
		if err := kill.Run(); err != nil {
			return cmd.Process.Kill()
		}
		return nil
	}
}
