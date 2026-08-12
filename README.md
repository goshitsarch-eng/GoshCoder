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
```

`run` and `chat` flags:

| Flag | Meaning |
| --- | --- |
| `-m`, `-model` | Model as `provider/model`, or a bare id when unambiguous |
| `-s`, `-system` | System prompt |
| `-thinking` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `-tools` | Enable the built-in file and shell tools |
| `-ralph` | Enable long-running ralph loops |
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
| `internal/config` | On-disk paths |
| `cmd/goshcoder` | CLI |

## Provider coverage

1213 of 1220 catalog models (99.4%) across eight wire protocols:
`openai-completions`, `anthropic-messages`, `openai-responses`,
`azure-openai-responses`, `google-generative-ai`, `google-vertex`,
`mistral-conversations`, `bedrock-converse-stream`.

Bedrock is reached without the AWS SDK: SigV4 signing and the binary
event-stream framing are implemented against stdlib crypto.

Not ported: `openai-codex-responses` (7 models). It needs a WebSocket
transport, zstd compression, and JWT parsing, and is OAuth-only, so those models
cannot authenticate without the OAuth login flows that are also out of scope.

## Extensions

pi extensions are TypeScript modules loaded at runtime. GoshCoder has no plugin
host, so extension features are ported as built-ins instead. `internal/ralph`
is a native port of `@tmustier/pi-ralph-wiggum`, keeping the same `.ralph`
file format so a loop directory works with either tool.

## Deviations from pi

Documented at the top of each ported file. The notable ones:

- **Compat plumbing.** pi keys `model.compat` by API; here per-API compat
  structs travel on the options struct.
- **No SDKs.** Provider requests are hand-rolled `net/http` plus an SSE reader
  rather than the OpenAI, Anthropic, Google, and AWS SDKs.
- **OAuth.** Stored, unexpired OAuth credentials resolve, but the login and
  refresh flows are not ported.
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
