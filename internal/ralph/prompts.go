package ralph

// The prompt text below is ported from the @tmustier/pi-ralph-wiggum
// extension. It is kept close to verbatim on purpose: task files and loop
// prompts are meant to read the same whether a loop is driven by pi or by
// GoshCoder.

// DefaultTemplate seeds a task file when the caller supplies no content.
const DefaultTemplate = `# Task

Describe your task here.

## Goals
- Goal 1
- Goal 2

## Checklist
- [ ] Item 1
- [ ] Item 2

## Verification
- Commands run, working directories, relevant environment variables, outputs, and preserved artifacts

## Final Verification
- Exact monitor-rerunnable command: <command>
- Working directory: <path>
- Required preserved artifacts: <paths>
- Result: <output summary>

## Notes
(Update this as you work)
`

// DefaultCompletionGate is the bar a loop must clear before claiming
// completion. Checked boxes alone are not enough: the work has to be verifiable
// by something other than the agent that did it.
const DefaultCompletionGate = `COMPLETION GATE

Do not output ` + CompleteMarker + ` based only on checked checklist items.
Before completion:
1. Run a final verification command that an external monitor can rerun from the same worktree in a fresh shell.
2. Record the exact command, working directory, relevant environment variables, and output summary in the task file.
3. Preserve every artifact required by that command, including build directories, generated libraries, virtualenvs, caches, or copied dylibs.
4. If cleanup removes required artifacts, recreate them or update the final command before completing.
5. If the final command cannot be made externally rerunnable, mark the item blocked/deferred instead of complete.`

// DefaultStalePromptGuard prevents a queued prompt from resurrecting a loop
// that already finished.
const DefaultStalePromptGuard = `STALE PROMPT GUARD

Before doing any work from a Ralph prompt, reload the loop state file named in the prompt (usually .ralph/<name>.state.json).
If the state says "status": "completed", do not edit files, do not run task commands, and do not call ralph_done. Reply briefly that the stale prompt was ignored because the loop is already completed.`

// DefaultReflectInstructions is the reflection checkpoint prompt.
const DefaultReflectInstructions = `REFLECTION CHECKPOINT

Pause and reflect on your progress:
1. What has been accomplished so far?
2. What's working well?
3. What's not working or blocking progress?
4. Should the approach be adjusted?
5. What are the next priorities?

Update the task file with your reflection, then continue working.`
