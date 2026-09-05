# GoshCoder

**GoshCoder is a derivative work, not an original one.** It is a Rust
reimplementation of [pi](https://github.com/earendil-works/pi) — Mario
Zechner's TypeScript coding agent — and the design it is good at is pi's.
Its interactive interface is built with
[Ratatui](https://ratatui.rs/) and Crossterm.

[Relationship to pi](#relationship-to-pi) explains what that means in
practice. [`NOTICE`](NOTICE) credits every project adapted here, and
[`LICENSE`](LICENSE) covers GoshCoder itself.

## Relationship to pi

GoshCoder is a **fork in substance**. It is not a git fork — the two share no
source language, and every file here was written against pi's source rather
than branched from it — but the lineage is not incidental. The wire protocols,
the session format, the tool surface, the resource conventions, the credential
layout and the model catalog are all pi's, and the project would not exist
without it.

What that means concretely:

- **The formats are pi's on purpose.** `auth.json` uses pi's encoding, and
  sessions are written in pi's v3 JSONL, so `-sessions-dir
  ~/.pi/agent/sessions` reads and writes the same files pi does.
  Interoperability is a goal here, not a side effect.
- **The model catalog is generated from the pi reference**, not maintained by
  hand. Models pi does not carry yet live in a separate
  `catalog_extra.json` so a regeneration cannot silently drop them.
- **Ported files name their source.** Each opens with a comment giving the pi
  path it was written against, and documents inline wherever GoshCoder's
  behaviour diverges.
- **Where it differs, it says so.** See [Deviations from pi](#deviations-from-pi)
  for the list, and [Known gaps](#known-gaps) for what is not implemented.

Several pi extensions are reimplemented here as native Rust built-ins rather
than loaded as npm packages. Each is an adaptation with no upstream code
bundled, and each is credited with its author, repository and licence in
[`NOTICE`](NOTICE) — see [Extensions](#extensions).

GoshCoder is not affiliated with, sponsored by, or endorsed by the pi project,
its author, or any other project named in `NOTICE`. **Bugs you find here are
GoshCoder's own — please report them on this repository, not upstream.**

## Install

**Linux and macOS**

```sh
curl -fsSL https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.sh | sh
```

**Windows** (PowerShell)

```powershell
irm https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.ps1 | iex
```

Both installers download the release for your platform, verify its SHA-256
against the published checksums file, and refuse to install anything that does
not match. When no release is published they fall back to building from source.
Re-running upgrades in place. Useful flags: `--dir <path>` to choose the install
location, `--version <tag>` to pin a release, `--from-source` to always compile,
`--no-modify-path` to leave your shell profile alone.

**From a checkout**

Requires the current stable Rust toolchain. `rust-toolchain.toml` selects it
for Cargo and rustup, while the installers build source releases with the same
toolchain.

```sh
make build      # stamped binary in bin/
make install    # and onto your PATH
make check      # the Rust gate: rustfmt, cargo check, Clippy, and tests
```

`make help` lists every target. `make dist` cross-compiles release archives for
Linux, macOS, and Windows on amd64 and arm64, with checksums.

**First run**

```sh
goshcoder auth login anthropic   # or: auth set <provider>, or an env var
goshcoder providers              # shows what is configured and how to fix what is not
goshcoder                        # start coding
```

## Use

```sh
# List providers and whether credentials are configured
goshcoder providers

# List models for configured providers
goshcoder models [provider]

# One-shot prompt
goshcoder run -m anthropic/claude-sonnet-5 "explain this repo"

# Fullscreen interactive session; the last selected model is remembered
goshcoder -m openai/gpt-5.6-terra -tools

# After the first launch, simply run
goshcoder

# Store a key (also reads the usual env vars, e.g. ANTHROPIC_API_KEY)
goshcoder auth set anthropic

# Come back to the last conversation in this workspace
goshcoder chat -continue

# Or pick one from a searchable list
goshcoder chat -resume

# Manage what has been saved
goshcoder sessions list
goshcoder sessions show <id>
goshcoder sessions export <id> --md notes.md

# Keep a prompt you refined, and carry your collection between machines
goshcoder prompts list
goshcoder prompts backup
goshcoder prompts restore goshcoder-prompts-2026-08-23.tar.gz

# Or log in with a subscription or account you already have
goshcoder auth login anthropic
goshcoder auth login openai-codex
goshcoder auth login kimi-coding
goshcoder auth login xai      # Grok subscription; device code or browser
goshcoder auth login meta     # Meta account; mints a Model API key

# The same providers by API key, for a developer account
goshcoder auth set xai        # then select xai/grok-4.6 or grok-build-0.1
goshcoder auth set meta       # then select meta/muse-spark-1.2
```

Inside chat, `/login` opens a provider picker. OAuth subscriptions and API-key
providers are added to `auth.json` independently, so signing in to one does not
remove existing logins. Use `/model` immediately afterward to search models
across every authenticated provider.

Session flags (`-claude-tui` and `-fullscreen` affect interactive chat only):

| Flag | Meaning |
| --- | --- |
| `-m`, `-model` | Model as `provider/model`, or a bare id when unambiguous |
| `-s`, `-system` | System prompt |
| `-thinking` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `-tools` | Enable built-in file and shell tools (default true in chat; use `-tools=false` for read-only chat) |
| `-ralph` | Enable long-running Ralph loops (default in chat; use `-ralph=false` to disable) |
| `-planner` | Start in native Planner review mode (`-plan` remains an alias) |
| `-claude-tui` | Use the native `pi-claude-code-tui` appearance in line mode |
| `-fullscreen` | Use the full-screen alternate-screen TUI (default true on an interactive terminal) |
| `-C` | Workspace directory for tools |
| `-continue` | Reopen the most recent session for this workspace |
| `-resume` | Choose a session to resume (chat only) |
| `-session` | Session id, id prefix, or path |
| `-name` | Display name for the session |
| `-no-session` | Do not record this session |
| `-read-only` | Open a session without claiming it |
| `-sessions-dir` | Session storage root (default `~/.goshcoder/agent/sessions`) |

`chat` records by default. `run` records only with `-continue`, `-session` or
`-name`: it is the scripting entry point, and recording by default would turn
every cron invocation into a permanent file.

Credentials and config live in `~/.goshcoder/agent` (override with
`GOSHCODER_AGENT_DIR`). The `auth.json` format matches pi's, so a directory can
be pointed at either tool. Sessions live under `sessions/`, sharded by working
directory with pi's exact encoding and written in pi's v3 JSONL format, so
`-sessions-dir ~/.pi/agent/sessions` reads and writes the same files pi does.

## Layout

| Package | Contents |
| --- | --- |
| `src/{llm,stream,providers,bedrock}` | Wire protocols, normalized messages, stream parsing, retries, and request adapters |
| `src/catalog.rs` | Provider catalog, model data, credential store, and auth resolution |
| `src/agent.rs` | Agent turns, tool execution, hooks, steering, follow-up queues, and compaction events |
| `src/tools.rs` | Pi-compatible built-in tools (`read`, `write`, `edit`, `bash`, `grep`, `find`, `ls`) |
| `src/webaccess.rs` | Native cited web search (OpenAI/Codex, Exa, and Kagi) |
| `src/{omniroute,omni_cli}.rs` | OmniRoute setup, catalog synchronization, and command adapter |
| `src/{aperture,aperture_mcp}.rs` | Tailscale Aperture core, routing, cache, and connector MCP client |
| `src/computeruse.rs` | computer-use-linux discovery, mcp.json upkeep, stdio MCP client, and `mcp` tool |
| `src/btw.rs` | Ephemeral context-aware side threads and settings |
| `src/{ralph,ralph_runtime,ralph_cli}.rs` | Durable iterative development loops |
| `src/{plannotator,planner_runtime}.rs` | Native Planner mode, browser approval, annotations, and checklist progress |
| `src/{session,sessionlog,sessions}.rs` | pi-compatible v3 sessions, lifecycle, tree, and command adapter |
| `src/{resources,prompts}.rs` | Local context, skills, templates, and portable prompt archives |
| `src/{state,ui,markdown}.rs` | Ratatui state, responsive rendering, composer, sidebar, and Markdown |
| `src/main.rs` | CLI dispatch and interactive Ratatui runtime |

## Provider coverage

The built-in catalog covers all bundled models across nine wire protocols:
`openai-completions`, `anthropic-messages`, `openai-responses`,
`openai-codex-responses`, `azure-openai-responses`, `google-generative-ai`,
`google-vertex`, `mistral-conversations`, `bedrock-converse-stream`.

The live Rust responder currently supports `openai-completions`,
`anthropic-messages`, `openai-responses`, `azure-openai-responses`,
`openai-codex-responses`, and `bedrock-converse-stream`. Google Generative AI,
Google Vertex, and Mistral Conversations are still being migrated before the Go
implementation is retired.

Model data is generated from the pi reference into `catalog.json`, which is
replaced wholesale on every regeneration. Two hand-maintained files sit beside
it: `catalog_extra.json` adds models the reference does not carry yet, and
`catalog_overrides.json` corrects fields it carries wrongly -- a context window,
a price, or the reasoning-effort levels a model accepts. Tests fail once the
regenerated data catches up with either, so neither file quietly outlives the
gap it was written for.

Bedrock is reached without the AWS SDK: SigV4 signing and the binary
event-stream framing are implemented against stdlib crypto.

OpenAI Codex uses its supported uncompressed SSE transport. WebSocket session
reuse and optional zstd request compression are intentionally omitted.

## Extensions

pi extensions are TypeScript modules loaded at runtime. GoshCoder has no plugin
host, so selected extension features are reimplemented as native built-ins.
Each is an adaptation written against the original, with no upstream code,
generated asset or npm dependency bundled; each author, repository and licence
is recorded in [`NOTICE`](NOTICE).

- [`pi-web-access`](https://github.com/nicobailon/pi-web-access) by Nico Bailon
  — **Web Search** (native adaptation): the `web_search` tool is
  enabled with the coding tools and supports single or batched queries, result
  counts, recency/domain filters, cited output, OpenAI/Codex subscription auth,
  zero-config Exa MCP fallback, direct Exa API keys, and Kagi Search. It reads
  `web-search.json` from the agent directory. For Kagi, set `KAGI_API_KEY` or:

  ```json
  {
    "provider": "kagi",
    "kagiApiKey": "$KAGI_API_KEY"
  }
  ```

  Set `provider` to `auto`, `openai`, `exa`, or `kagi`; `auto` reuses an
  OpenAI/Codex login when suitable and otherwise works without a key through
  Exa. This is a native Rust adaptation of the search tool, not an npm plugin
  runtime.
  The package's `fetch_content`, `get_search_content`, `source_check`, browser
  curator, video/PDF handling, and providers not listed above are not ported.
- [`omniroute-agent-extension`](https://github.com/md-riaz/omniroute-agent-extension)
  by Oscar Andrea / md-riaz — **OmniRoute** (native adaptation, written against
  the package when it was named `omniroute-pi-ext-integration`): `/omni
  setup` validates and stores a local or remote gateway, `/omni sync` imports
  `/v1/models` into GoshCoder's live `/model`/Ctrl+P picker, `/omni status`
  checks health, and `/omni dashboard` reports the management URL. Synchronized
  context, output, reasoning, vision, and native-tool metadata are retained.
  Web/chat-only models use the package version 2.0.1 buffered `<tool_call>`
  prompt adapter and are converted back into normal agent tool events. Config
  is in `omniroute.json`; its API key stays independently in `auth.json` or
  `OMNIROUTE_API_KEY`.
- [`@narumitw/pi-btw`](https://github.com/narumiruna/pi-extensions/tree/main/packages/pi-btw)
  by narumiruna — **BTW** (native adaptation of version 0.50.0): `/btw <question>` opens a
  context-aware side thread without adding the question or answer to the main
  transcript. The fullscreen side UI supports follow-ups, queued Steering,
  in-memory resume, independent model/thinking settings in `pi-btw.json`,
  Shift+Tab thinking changes, scrolling, cancellation, and Ctrl+R to bring the
  latest Q&A into the editable main composer. GoshCoder requires the main agent
  to be idle before opening BTW rather than rendering both agents concurrently.
  `/btw` lists retained threads;
  `/btw resume <id> <question>` resumes one. The original's exact character/
  line range selector and nested bring-preview menus are not ported; native
  line mode offers deterministic `latest`, `all`, and `from:N` export instead.
- [`@tmustier/pi-ralph-wiggum`](https://github.com/tmustier/pi-extensions) by Thomas
  Mustier — **Ralph** (native adaptation): persistent iterative loops and
  completion tools.
  Ralph is enabled by default in chat; use `/ralph start <name> <task>` or ask
  the model to start a loop. Loop state is kept in the workspace-local
  `.ralph/` directory so existing Go and pi-ralph-wiggum loops remain
  discoverable. `/ralph` opens loop controls in the command palette.
- **Planner** (native adaptation of
  [`@plannotator/pi-extension`](https://github.com/backnotprop/plannotator) by
  the Plannotator contributors): `/planner`,
  browser plan approval/denial with line annotations, overall notes, direct
  Markdown edits, resubmission change views, responsive navigation, light/dark
  themes, planning write gates, persisted phase state, checklist progress,
  `/planner-review`, `/planner-annotate`, and `/planner-last`. Use `-planner` to
  begin in planning mode. PR URL review uses the optional GitHub CLI (`gh`);
  local git review needs only `git`.
- [`@aliou/pi-ts-aperture`](https://github.com/aliou/pi-ts-aperture) by Aliou
  Diallo — **Aperture (Tailscale)** (native adaptation of version 0.14.1):
  route LLM providers and connector tools through
  [Tailscale Aperture](https://tailscale.com/docs/features/aperture), a managed
  AI gateway on your tailnet, so GoshCoder never needs upstream provider
  credentials. All three capabilities are adapted:

  - **Dedicated** (default): a standalone `aperture` provider whose models come
    from the gateway, filtered per provider, each routed through the API
    auto-picked from the gateway's compatibility map or a per-provider `api`
    override (falling back to auto with a warning when the gateway stops
    serving it). Model ids are provider-qualified except on APIs that embed the
    id in the URL (Gemini, Vertex, Bedrock), which get bare ids; Anthropic and
    Codex register the gateway root, Bedrock `/bedrock`, Gemini `/v1beta`, and
    OpenAI-shaped providers `/v1` unless their upstream ends in a non-`/v1`
    version segment. Capabilities come from the first source that knows the
    model — GoshCoder's catalog, then [models.dev](https://models.dev), then
    safe defaults — and costs come from the gateway. The catalog is cached
    (`extensions/aperture-cache.json`) under a gateway/selection identity key,
    so models load instantly on startup, even offline.
  - **Proxy**: reroute existing providers through Aperture, keeping their own
    model definitions; only the base URL, credentials, and headers are
    overridden, requests carry the provider-qualified model id, passthrough
    (`auth_mode`) providers keep sending real credentials, optional gateway
    model checks warn about locally configured models the gateway is missing,
    and `keepGatewayModelsOnly` hides them from the picker entirely.
  - **Connectors**: gateway MCP tools surface either as the four discovery
    meta-tools (`aperture_connector_list`, `..._tool_search`,
    `..._tool_describe`, `..._tool_call`) that keep tool schemas out of the
    system prompt, or as pinned first-class tools
    (`/aperture pin <tool>`), with large responses overflowing to a temp file.

  `/aperture onboarding` (also `/aperture:onboarding`) is the first-run wizard
  — URL with inline health check, capability choice, provider selection, recap
  — and `/aperture settings` (also `/aperture:settings`) shows or changes
  every setting, including per-provider toggles and API overrides. `/aperture
  sync`, `status`, `providers`, and `connectors` cover refresh and
  diagnostics; the catalog also revalidates in the background at session
  start. Config lives in pi's `extensions/aperture.json`, including the
  original's content-gated migrations from its pre-0.6 shapes. Requests
  carry the live session id in `x-session-id` for the Aperture dashboard; the
  `Referer` names this repository rather than the original's `https://pi.dev`.
  Transient "Aperture is restarting" failures are tagged so the standard
  auto-retry recovers them. Not ported: the panel-based settings/onboarding
  TUI (GoshCoder uses prompts and subcommands) and pi's models-store cache
  file (a sibling cache file with the same identity-key semantics is used
  instead).
- [`@agent-sh/computer-use-linux`](https://github.com/agent-sh/computer-use-linux)
  by Avi Fenesh — **Desktop control / Screenshots** (native adaptation of
  version 0.4.10): Linux desktop observation and control through the
  `computer-use-linux` MCP server — accessibility trees, window targeting,
  screenshots, and input synthesis. The upstream pi extension registers the
  server into `mcp.json` for the separate `pi-mcp-adapter`; GoshCoder has no
  adapter package, so the whole chain is native: the same binary discovery
  order (`COMPUTER_USE_LINUX_BIN`, then `PATH`, then `~/.local/bin` — the
  npm-bundle-relative probe has no native equivalent), the same
  pi-compatible `mcp.json` maintenance (other entries preserved; malformed
  files reported, never overwritten), a stdio MCP client speaking the rmcp
  2024-11-05 protocol, and an `mcp` tool covering the documented call shapes:
  `mcp({server: "computer-use-linux"})`, `mcp({search: "windows"})`, and
  `mcp({tool: "computer_use_linux_screenshot", args: {...}})`. Screenshots
  return as inline images the model can see; upstream tool annotations are
  surfaced as read-only/mutating/destructive markers; desktop calls run
  sequentially because input is stateful. The bundled skill's operating
  procedure (doctor first, semantic targeting, explicit input targets,
  re-check after mutations) is folded into the tool description. Linux only,
  like the upstream package; the server binary itself is not bundled —
  install it with `npm install -g @agent-sh/computer-use-linux` or `cargo
  install computer-use-linux` and check readiness with
  `computer-use-linux doctor`.
- [`pi-claude-code-tui`](https://pi.dev/packages/pi-claude-code-tui) by Phoobobo
  — startup card, half-open rounded chat prompt, and an
  OpenCode-inspired right sidebar with model, context usage, cost, messages,
  tools, changed files, branch, and active mode. The panel refreshes
  automatically after turns and state changes; `/status` can also print it on
  demand. It is enabled by default in chat. In line mode, use
  `/use-default-tui` or `-claude-tui=false` for the plain interface and
  `/use-claude-code-tui` to switch back. The fullscreen layout is fixed for the
  lifetime of the process.

The visual extensions are implemented by GoshCoder's Ratatui terminal UI rather
than pi's TypeScript runtime. Type `/` to open the command
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
  locations, available as `/skill:<name>`. Ancestor discovery stops at the
  repository root, so a sibling checkout's skills are not offered as this
  project's.

Use `/resources` to inspect what was loaded; it also lists any warning raised
during discovery, including a workspace `SYSTEM.md` that replaced the prompt
and any context file skipped for being a symbolic link. GoshCoder generates Pi's coding
system prompt when no explicit or local `SYSTEM.md` override exists.

## Security notes

GoshCoder reads instructions out of whatever repository you open, so the trust
boundary matters:

- **`AGENTS.md` / `CLAUDE.md`** from the workspace are added as project context.
  Their content is framed so it cannot break out of its container and address
  the model as if it were the harness. Symlinked context files are skipped: a
  repository could otherwise point `AGENTS.md` at your SSH key and have it sent
  to the model provider.
- **`.pi/SYSTEM.md` / `.goshcoder/SYSTEM.md`** in a workspace *replace* the whole
  system prompt. Your own `SYSTEM.md` in the agent directory takes precedence,
  and a workspace one is reported in `/resources` rather than applied silently.
  Treat it like any other executable content in a repository you did not write.
- **Filesystem tools** are confined to the workspace by Rust path and symlink
  validation, so symlink and rename races cannot escape it. The `bash` tool
  runs with your privileges;
  planning mode removes it and independently blocks shell execution.
- **Credentials** live in `auth.json` (mode 0600) and are never echoed to the
  terminal. Concurrent sessions coordinate through a heartbeat lock file so a
  refreshed token cannot be lost to a racing writer. The 0600 here and below is
  a Unix guarantee: Windows has no permission bits, so those files are
  protected by whatever ACLs they inherit from your user profile instead.
- **Local servers** (OAuth callback, Planner review) bind loopback only, validate
  state/CSRF tokens and the `Host` header, and set no-store and CSP headers.
- **Session transcripts** contain every file the agent read and every command's
  output. `read` returns up to 50 KiB and `bash` output up to 30 KiB, and there
  is no content filter, so `cat .env` lands in the session file verbatim. Files
  are mode 0600 under `~/.goshcoder/agent/sessions`, never inside the workspace,
  and are never sent anywhere. **No redaction is performed**: the only redaction
  primitive in this tree replaces a *known* secret string, which cannot find a
  key the agent read out of a file, and a partial filter would buy false
  confidence. Use `-no-session` for work that should not be written down, and
  `goshcoder sessions rm` to delete what already was.
- **Prompt archives** written by `prompts backup` are read back as untrusted
  input: member names are re-derived and re-validated rather than trusted, any
  member that is not a regular file is refused, and both the entry count and the
  decompressed size are bounded. Symlinked prompts are skipped on backup rather
  than followed, so an archive you share cannot carry a file you did not choose.

**Durability.** A completed turn survives a machine crash: entries are appended
as they happen and flushed at each turn boundary. A turn still in flight
survives a process crash -- a kill, a panic, a closed terminal -- but not a power
loss, because the containing directory is not fsynced. That matches every other
durable write in this repository. A session torn mid-append loses only the
partial entry; the rest of the file loads normally.

CI and release builds run `cargo audit`; keep the Rust toolchain patched and
rerun the gate before releases.

## Known gaps

- Planner state belongs to a session rather than to a workspace, so
  `-no-session` and a `run` without `-continue` do not persist it. Two windows
  in one repository now have independent plan modes; `-continue` restores the
  phase along with the transcript.
- HTML export and `/share` are not implemented, and are not planned. pi's HTML
  export inlines roughly 165 KB of vendored JavaScript into every output file;
  `sessions export --md` writes Markdown instead. `/share` would mean sending a
  file that by construction contains everything the agent read to a third-party
  host.
- BTW side threads are still memory-only. Closing a window discards them even
  though the main conversation is saved.

## Deviations from pi

Documented at the top of each ported file. The notable ones:

- **Compat plumbing.** pi keys `model.compat` by API; here per-API compat
  structs travel on the options struct.
- **No SDKs.** Provider requests are hand-rolled blocking HTTP plus an SSE reader
  rather than the OpenAI, Anthropic, Google, and AWS SDKs.
- **OAuth.** Login and refresh are ported for Anthropic, OpenAI Codex, Kimi
  Code, xAI, and Meta. Every one of them also accepts an API key, so a
  developer account never has to go through a subscription login.
  - **xAI (Grok)** uses xAI's own OIDC server at `auth.x.ai`: PKCE S256 over a
    loopback callback, or RFC 8628 device code for a headless session, against
    the public desktop client. Discovered endpoints are pinned to the issuer's
    host, so a discovery document cannot move the token exchange elsewhere.
    What this authenticates is a consumer Grok subscription, and xAI applies
    its own entitlement checks afterwards: a login can succeed and inference
    still answer 403 for an account without the plan the endpoint wants. That
    is xAI's decision, not a client bug, and `XAI_API_KEY` is unaffected.
  - **Meta** signs in by device code at `auth.meta.com` and then mints a Model
    API key at `api.meta.ai/muse-code/key`; the key is what requests carry, and
    Meta re-mints it about once a day, which the stored expiry accounts for.
    Meta's Model API speaks the Anthropic Messages shape but authenticates with
    `Authorization`, so the key travels as a header and never as `x-api-key`.
  - The client ids both flows use are public desktop clients with no secret --
    PKCE and the device flow replace one. `GOSHCODER_XAI_OAUTH_CLIENT_ID` and
    `GOSHCODER_META_OAUTH_CLIENT_ID` override them for an account registered
    against a different application.
- **Interface.** Interactive chat uses a Ratatui alternate-screen TUI with a
  command palette, model-aware thinking picker, live activity, fixed transcript,
  multiline editor, compact tool cards, and responsive OpenCode-style sidebar.
  Redirected input/output automatically falls back to the pipeable line-oriented
  interface.
- **Session scope.** Sessions are persisted in pi's v3 JSONL format, including
  the entry tree, resume, branching, fork/clone, labels, and JSONL/Markdown
  export and import. pi's older v1 and v2 files are read and migrated in memory;
  they are never rewritten in place, so continuing one forks it into a v3 file.
  Pi's HTML export/share, TypeScript plugin host, packages, custom `models.json`
  loading, LSP, and MCP management are not implemented. See **Known gaps**.

  Two deliberate differences inside the format. GoshCoder takes an exclusive
  claim on a session file, which pi does not: two processes appending to one
  file is the realistic corruption mode for `chat -continue` run twice, and
  interleaved appends cross the parent links of two conversations. And `/clear`
  appends a reset marker rather than starting a new file, so an accidental clear
  stays recoverable with `sessions show --full`; pi reading such a file will
  still replay the cleared prefix, since only this build knows the marker cuts
  the context.

## Development

```sh
make check          # rustfmt, cargo check, Clippy, and Rust tests
make tools          # install cargo-audit, cargo-llvm-cov, and cargo-zigbuild
```

Individual targets: `make test`, `make test-race`, `make cover`, `make lint`,
`make vuln`. `make test-hermetic` runs the suite with deliberately wrong
provider credentials exported, so a test that reads the developer's real
environment instead of its own fixtures fails loudly rather than passing on one
machine and failing on another.

GitHub Actions runs the same gate on Linux, macOS, and Windows, cross-compiles
every release target, and exercises both installer scripts -- including a
round-trip that serves real release archives over HTTP and drives the
installer's actual download path, so the Makefile and the installers cannot
drift apart into a release that 404s. Filesystem tools use Rust workspace
confinement, planning mode disables shell execution, local callback
servers bind only to loopback, and network, disk, and subprocess inputs have
explicit resource limits.

Ports are written against a local clone of the upstream source, which lives at
`reference/pi` and is gitignored — it is pi's code, not GoshCoder's, and is
never committed here:

```sh
git clone https://github.com/earendil-works/pi reference/pi
```

Read the TypeScript before porting anything, and name the file you read in a
comment at the top of the Rust file. That comment is what lets the next person
check a port against its original instead of guessing at intent.

## License

GoshCoder is MIT-licensed — see [`LICENSE`](LICENSE).

It is a derivative work of [pi](https://github.com/earendil-works/pi),
Copyright (c) 2025 Mario Zechner, used under the MIT License.
[`NOTICE`](NOTICE) reproduces that copyright and credits every other project
adapted here — `pi-web-access`, Plannotator, `pi-ralph-wiggum`,
`pi-claude-code-tui`, OmniRoute and `pi-btw` — with its author, repository and
licence. If you redistribute GoshCoder or a build of it, carry `NOTICE` with
it: that is the condition every one of those licences attaches.
