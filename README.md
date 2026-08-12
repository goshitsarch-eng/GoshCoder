# GoshCoder

A Go coding agent, ported from [pi](https://github.com/earendil-works/pi).
Its interactive interface is built with
[Bubble Tea](https://github.com/charmbracelet/bubbletea) and Lip Gloss.

## Install

Requires Go 1.26.5 or newer. The patch-level minimum avoids known standard-library security vulnerabilities in earlier Go 1.26 releases.

```sh
go build -o bin/goshcoder ./cmd/goshcoder
```

## Use

```sh
# List providers and whether credentials are configured
goshcoder providers

# List models for configured providers
goshcoder models [provider]

# One-shot prompt
goshcoder run -m anthropic/claude-sonnet-4-5 "explain this repo"

# Fullscreen interactive session; the last selected model is remembered
goshcoder -m openai/gpt-5.1 -tools

# After the first launch, simply run
goshcoder

# Store a key (also reads the usual env vars, e.g. ANTHROPIC_API_KEY)
goshcoder auth set anthropic

# Or log in with an Anthropic, OpenAI Codex, or Kimi subscription
goshcoder auth login anthropic
goshcoder auth login openai-codex
goshcoder auth login kimi-coding
```

`run` and `chat` flags:

| Flag | Meaning |
| --- | --- |
| `-m`, `-model` | Model as `provider/model`, or a bare id when unambiguous |
| `-s`, `-system` | System prompt |
| `-thinking` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `-tools` | Enable built-in file and shell tools (default true in chat; use `-tools=false` for read-only chat) |
| `-ralph` | Enable long-running ralph loops |
| `-plan` | Start in native Plannotator planning/review mode |
| `-claude-tui` | Use the native `pi-claude-code-tui` appearance in line mode |
| `-fullscreen` | Use the full-screen alternate-screen TUI (default true on an interactive terminal) |
| `-C` | Workspace directory for tools |

Credentials and config live in `~/.goshcoder/agent` (override with
`GOSHCODER_AGENT_DIR`). The `auth.json` format matches pi's, so a directory can
be pointed at either tool.

## Layout

| Package | Contents |
| --- | --- |
| `internal/llm` | Wire protocols, message types, event stream, retry, partial-JSON |
| `internal/llm/catalog` | Provider catalog, model data, credential store, auth resolution |
| `internal/agent` | Agent loop: turns, tool execution, hooks, steering and follow-up queues |
| `internal/tools` | Built-in tools (read, write, edit, list, bash) |
| `internal/ralph` | Long-running iterative development loops |
| `internal/plannotator` | Plan mode, browser approval/annotations, and checklist progress |
| `internal/claudetui` | Claude-style cards and session information rendering |
| `internal/tui` | Bubble Tea view renderer, responsive transcript, palette, composer, and sidebar |
| `internal/config` | On-disk paths |
| `cmd/goshcoder` | CLI |

## Provider coverage

All 1220 catalog models (100%) across nine wire protocols:
`openai-completions`, `anthropic-messages`, `openai-responses`,
`openai-codex-responses`, `azure-openai-responses`, `google-generative-ai`,
`google-vertex`, `mistral-conversations`, `bedrock-converse-stream`.

Bedrock is reached without the AWS SDK: SigV4 signing and the binary
event-stream framing are implemented against stdlib crypto.

OpenAI Codex uses its supported uncompressed SSE transport. WebSocket session
reuse and optional zstd request compression are intentionally omitted.

## Extensions

pi extensions are TypeScript modules loaded at runtime. GoshCoder has no plugin
host, so selected extension features are native built-ins:

- `@tmustier/pi-ralph-wiggum`: persistent iterative loops and completion tools.
- `@plannotator/pi-extension`: `/plannotator`, browser plan approval/denial with
  line annotations, planning write gates, persisted phase state, checklist
  progress, `/plannotator-review`, `/plannotator-annotate`, and
  `/plannotator-last`. Use `-plan` to begin in planning mode. PR URL review
  uses the optional GitHub CLI (`gh`); local git review needs only `git`.
- `pi-claude-code-tui`: startup card, half-open rounded chat prompt, and an
  OpenCode-inspired right sidebar with model, context usage, cost, messages,
  tools, changed files, branch, and active mode. The panel refreshes
  automatically after turns and state changes; `/status` can also print it on
  demand. It is enabled by default in chat. Use `/use-default-tui` or
  `-claude-tui=false` for the plain interface, and `/use-claude-code-tui` to
  switch back.

The visual extensions are implemented by GoshCoder's native Bubble Tea TUI
rather than depending on pi's TypeScript TUI or an npm runtime. Type `/` to open the
command palette; arrows select an item, Tab completes it, and Enter accepts it.
`/thinking` opens a second palette containing only levels supported by the
active provider/model.

## Deviations from pi

Documented at the top of each ported file. The notable ones:

- **Compat plumbing.** pi keys `model.compat` by API; here per-API compat
  structs travel on the options struct.
- **No SDKs.** Provider requests are hand-rolled `net/http` plus an SSE reader
  rather than the OpenAI, Anthropic, Google, and AWS SDKs.
- **OAuth.** Login and refresh are ported for Anthropic, OpenAI Codex, and Kimi
  Code. Kimi also supports API-key authentication.
- **Interface.** Interactive chat uses a Bubble Tea alternate-screen TUI with a
  command palette, model-aware thinking picker, live activity, fixed transcript,
  editor, and responsive information sidebar. Redirected input/output
  automatically falls back to the pipeable line-oriented interface.

## Development

```sh
go build ./... && go vet -all ./... && go test ./...
staticcheck ./...   # optional external audit tool
govulncheck ./...   # optional external vulnerability scanner
```

Release validation also runs the race detector. Filesystem tools use Go's
`os.Root` confinement, planning mode disables shell execution, local callback
servers bind only to loopback, and network, disk, and subprocess inputs have
explicit resource limits.

The pi reference clone lives in `reference/pi` (gitignored) and is what the
ports are written against.

## License

pi is MIT-licensed by Mario Zechner; see `NOTICE`.
