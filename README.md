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
| `internal/tools` | Pi-compatible built-in tools (`read`, `write`, `edit`, `bash`, `grep`, `find`, `ls`) |
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
  line annotations, overall notes, direct Markdown edits, resubmission change
  views, responsive navigation, light/dark themes, planning write gates,
  persisted phase state, checklist progress, `/plannotator-review`,
  `/plannotator-annotate`, and `/plannotator-last`. Use `-plan` to begin in
  planning mode. PR URL review uses the optional GitHub CLI (`gh`); local git
  review needs only `git`.
- `pi-claude-code-tui`: startup card, half-open rounded chat prompt, and an
  OpenCode-inspired right sidebar with model, context usage, cost, messages,
  tools, changed files, branch, and active mode. The panel refreshes
  automatically after turns and state changes; `/status` can also print it on
  demand. It is enabled by default in chat. Use `/use-default-tui` or
  `-claude-tui=false` for the plain interface, and `/use-claude-code-tui` to
  switch back.

The visual extensions are implemented by GoshCoder's lightweight, Crush-inspired
Bubble Tea TUI rather than pi's TypeScript runtime. Type `/` to open the command
palette; arrows select an item, Tab completes it, and Enter accepts it. `/model`
opens a searchable picker containing models from authenticated providers, while
`/thinking` contains only levels supported by the active model. The composer
supports up to three visible lines (`Shift+Enter`/`Ctrl+J` inserts a newline),
word navigation, and line-aware Home/End. Tool calls render as compact cards;
`Ctrl+O` expands their output and `Ctrl+T` toggles thinking. `/compact [focus]`
creates a Pi-style structured summary while retaining recent turns; the same
compaction runs automatically near the active model's context limit. Transient
429/5xx provider failures are retried automatically.

## Local resources

GoshCoder discovers Pi-compatible inert resources at startup and with `/reload`:

- ancestor `AGENTS.md`, `AGENTS.override.md`, and `CLAUDE.md` instructions;
- `SYSTEM.md` and `APPEND_SYSTEM.md` under the agent directory, `.pi`, or
  `.goshcoder`;
- Markdown prompt templates under `prompts/`, including frontmatter and
  positional arguments;
- Agent Skills (`SKILL.md`) under agent, project, and `.agents/skills`
  locations, available as `/skill:<name>`.

Use `/resources` to inspect what was loaded. GoshCoder generates Pi's coding
system prompt when no explicit or local `SYSTEM.md` override exists.

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
  multiline editor, compact tool cards, and responsive OpenCode-style sidebar.
  Redirected input/output automatically falls back to the pipeable line-oriented
  interface.
- **Session scope.** Runtime transcript clearing and context compaction are
  implemented. Pi's persisted JSONL session tree, resume picker, branching,
  HTML export/share, TypeScript plugin host, packages, LSP, and MCP management
  are not yet implemented.

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
