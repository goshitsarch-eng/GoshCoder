# GoshCoder

A Go coding agent, ported from [pi](https://github.com/earendil-works/pi).
No third-party dependencies: everything is Go's standard library.

## Install

Requires Go 1.26+.

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

# Interactive session (/help lists slash commands)
goshcoder chat -m openai/gpt-5.1 -tools

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
| `-tools` | Enable the built-in file and shell tools |
| `-ralph` | Enable long-running ralph loops |
| `-plan` | Start in native Plannotator planning/review mode |
| `-claude-tui` | Use the native `pi-claude-code-tui` chat appearance (default true; disable with `-claude-tui=false`) |
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
| `internal/claudetui` | Native startup card and half-open chat input styling |
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

The visual extensions are adapted to GoshCoder's line-oriented terminal rather
than depending on pi's TypeScript TUI or an npm runtime.

## Deviations from pi

Documented at the top of each ported file. The notable ones:

- **Compat plumbing.** pi keys `model.compat` by API; here per-API compat
  structs travel on the options struct.
- **No SDKs.** Provider requests are hand-rolled `net/http` plus an SSE reader
  rather than the OpenAI, Anthropic, Google, and AWS SDKs.
- **OAuth.** Login and refresh are ported for Anthropic, OpenAI Codex, and Kimi
  Code. Kimi also supports API-key authentication.
- **Interface.** pi renders a full-screen TUI; `goshcoder chat` is
  line-oriented, keeping session semantics behind slash commands so stdout
  stays pipeable.

## Development

```sh
go build ./... && go vet ./... && go test ./...
```

The pi reference clone lives in `reference/pi` (gitignored) and is what the
ports are written against.

## License

pi is MIT-licensed by Mario Zechner; see `NOTICE`.
