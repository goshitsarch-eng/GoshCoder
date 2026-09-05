# GoshCoder — continuation notes

Maintainer handoff notes. For what GoshCoder *is* and where it comes from, read
[`README.md`](README.md) first — in particular **Relationship to pi**, because
almost every decision recorded below only makes sense against pi's design.
Attribution for pi and for every natively adapted extension is in
[`NOTICE`](NOTICE); GoshCoder's own licence is in [`LICENSE`](LICENSE).

## Verify state first

```sh
make check
```

That is the gate CI runs: `fmt-check vet lint test test-hermetic vuln`, on
Linux, macOS and Windows, followed by a cross-compile of every release target
and an installer round-trip. Expected: rustfmt clean, `cargo check` clean,
Clippy clean under `-D warnings`, every test passing, and `cargo audit` quiet
when it is installed.

Ports are written against a clone of the upstream source at `reference/pi`,
which is gitignored because it is pi's code rather than GoshCoder's:

```sh
git clone https://github.com/earendil-works/pi reference/pi
```

Read the TypeScript before porting anything, and name the file you read at the
top of the Rust file.

## History, in one paragraph

GoshCoder began as a Go port of pi (0.1 through 0.4) and was rewritten in Rust
with a Ratatui interface in the "Migrate GoshCoder runtime and UI to
Rust/Ratatui" change (#13). The Go tree was kept alongside the Rust one until
the rewrite reached parity and was then deleted; its last revision is the
parent of #13, and comments that cite an `internal/...` or `cmd/goshcoder/...`
path refer to that history. The rewrite reached parity on this branch: the
gateway integrations were wired into the catalog and request path, the
prompt-emulated tool protocol was ported, the in-chat login and gateway
commands were added, and a module-by-module audit against pi and the Go
original fixed the defects listed under **Audit** below.

## Goal

User's priorities, in their words: **Codex, the Chinese models, and Anthropic.
GitHub Copilot explicitly not wanted.** Everything under that is done: Chinese
providers are plain API-key providers on ported protocols (Kimi additionally
has OAuth), Anthropic and Codex have PKCE/loopback and device-code logins with
refresh, and `openai-codex-responses` is ported end to end. The same is true of
xAI and Meta, added later.

## Layout

The README's **Layout** table is the map. The pieces that matter most when
something goes wrong:

| Concern | Where |
| --- | --- |
| Model data, credentials, auth resolution | `src/catalog.rs`, data in `data/*.json` |
| OAuth logins and refresh | `src/oauth.rs` |
| Wire protocols and streaming | `src/providers.rs`, `src/stream.rs`, `src/bedrock.rs`, `src/mistral.rs` |
| Chat-only OmniRoute models | `src/omni_prompt_tools.rs` (protocol), glue in `providers.rs` |
| Agent loop, queues, compaction | `src/agent.rs`, `src/compaction.rs` |
| Session files (pi v3 JSONL) | `src/sessionlog.rs`, `src/session.rs` |
| Tools and workspace confinement | `src/tools.rs` |
| Gateways | `src/omniroute.rs`, `src/aperture*.rs`, session start in `src/runtime.rs` |
| Fullscreen interface | `src/state.rs` (editor/palette), `src/ui.rs` (rendering), `src/main.rs` (event loop, slash commands) |

## Catalog data

`data/catalog.json` is generated from the pi reference (pi's
`scripts/generate-models.ts`, run against `reference/pi`) and replaced
wholesale on every regeneration; a hand edit there is lost. Two hand-maintained
files sit beside it:

- `data/catalog_extra.json` adds models pi does not carry yet. The generated
  data wins on a collision and a test fails if pi ever ships the same model, so
  the duplicate is deleted rather than quietly diverging.
- `data/catalog_overrides.json` patches fields pi carries wrongly (a context
  window, a price, the effort levels a model accepts). An override is a partial
  model object whose top-level keys replace the generated ones wholesale; `id`
  and `provider` are refused. Tests fail on an override whose target pi does
  not carry, and on one that restates what pi already says.

`xhigh` and `max` need an explicit `thinkingLevelMap` entry to be offered at
all, so a model that gains one stays silently capped until its entry says so.
Two corrected prices are promotional and need revisiting: GPT-5.6 Sol's
$4/$20 (at least through 2026-11-21) and the Gemini 3.6/3.7 Flash introductory
rate (through 2026-12-31).

## Gateway integrations

pi's extensions register wrapped provider objects at runtime. GoshCoder's
catalog is read on demand instead, so the equivalent is a **dynamic layer**
computed from the integrations' files and cached by their modification time
and size (`DynamicPaths`/`DynamicLayer` in `catalog.rs`):

- `omni` takes its base URL and models from `omniroute.json`; a model whose
  `toolCalling` is false gets the `omni-prompt-tools` API and is served by
  the prompt-emulated protocol.
- The dedicated `aperture` provider serves the synchronized
  `extensions/aperture-cache.json`, so models load instantly even offline; a
  cache built for another gateway or selection is ignored.
- A provider routed through an Aperture proxy carries the gateway URL, API
  override and gateway model filter on every model, keeps bare ids in the
  picker, and resolves the `-` placeholder credential unless the gateway
  passes client auth through. A stored OAuth login still wins.
- The request path (`catalog_assistant_responder` in `providers.rs`)
  qualifies the model id and adds the `Referer` and live `x-session-id`
  headers for gateway-routed models.

The paths derive from the catalog's injected environment, so a test lookup
without `HOME` reads nothing; `Catalog::with_dynamic_paths` points tests at a
temporary agent directory. `Catalog::refresh_dynamic` is called after an
in-process sync so it never depends on timestamp granularity.

Session start (`PreparedSession::aperture_session_start`) refreshes the cache
in the background and, once the gateway answers, registers the connector
tools: pinned tools first-class plus the four discovery meta-tools from
`src/aperture_tools.rs`. Tools that arrive after startup go through
`PreparedSession::register_tools`, which with a planner attached adds them to
the planner's ordinary set so a phase change cannot unregister them. Gateway
tool names outside `[A-Za-z0-9_-]{1,64}` or colliding with a session tool are
skipped and reported, because a single bad name breaks every later request.

On Linux the computer-use-linux `mcp` proxy tool is registered when the binary
is installed; the server is spawned lazily and closed with the session.

## Provider protocols and deviations

- Compat travels on the option structs, with `Model.compat` as the fallback.
- No vendor SDKs: hand-rolled blocking HTTP plus an SSE reader. Bedrock does
  SigV4 and binary event-stream framing against the sha2/hmac crates.
- Streaming requests have connect and idle-read timeouts, never a whole-request
  deadline: a long reasoning turn must not be cut off at an arbitrary limit.
- Same-model replay keeps original tool call ids and thinking signatures;
  cross-model replay normalizes. This tripped up test expectations repeatedly
  in the Go port; keep the rule in mind when a provider test surprises you.
- Anthropic: `cache_control` on the system block, the last tool and the last
  user block, the beta header set pi sends, adaptive thinking for models that
  ask for it, and the Claude Code identity shape for `sk-ant-oat` tokens.
- OAuth token endpoints and callback ports are the ones pi uses; xAI's
  discovered endpoints are pinned to the issuer host. `Auth` deliberately
  implements no `Debug` or `Display`.

## Sessions

Sessions are pi's **v3** JSONL: header, entries with `parentId` links,
`YYYY-MM-DDTHH:MM:SS.mmmZ` timestamps, `<stamp>_<id>.jsonl` names in a
per-workspace shard. Resume, branching, fork/clone, labels and JSONL/Markdown
export/import are implemented; v1 and v2 pi files are migrated in memory and
never rewritten, so continuing one forks it into a v3 file.

Two deliberate differences: GoshCoder takes an exclusive claim on a session
file (a lock file with a heartbeat; a claim that is taken over stops the
writer rather than letting two processes interleave), and `/clear` appends a
`transcript_reset` marker instead of rotating to a new file — pi tolerates the
entry but replays the cleared prefix, which is the one documented interop
divergence.

## Audit

Every module was reviewed against pi and the Go original for correctness,
security and performance, and the confirmed findings were fixed with
regression tests. The ones worth knowing about when reading the code:

- The agent loop never executes tool calls from a `length`-truncated message,
  continues after a tool batch unless every result set `terminate` (an error
  result is feedback, not a stop), polls steering before the first request,
  and keeps running queued messages after a prompt the way pi's
  `agent-session` loops on `continue`.
- Provider replay skips `error` *and* `aborted` turns and synthesizes
  "No result provided" tool results, so a cancelled tool call cannot poison
  every later request.
- Filesystem tools serialize read-modify-write per path (pi's file mutation
  queue), kill the whole process group on timeout or cancel, give the shell no
  stdin, and edit CRLF files the way `read` shows them.
- Session files: a pre-existing file is never deleted on close, existing
  directories are never re-permissioned, exports are user-only.
- The fullscreen interface pre-wraps every transcript line, so scrolling is in
  rows and the newest reply is always reachable; input events are coalesced
  and the view is rebuilt only when something changed.
- Loopback servers (OAuth callback, planner review) survive a malformed or
  reset connection, keep a per-connection deadline, and never lose a decision
  to a failed response write.

Anything an audit found that was not fixed is recorded in the README's
**Known gaps**.

## Cutting a release

`.github/workflows/release.yml` publishes the cross-compiled archives and the
`checksums.txt` both installers verify against. Three ways in, one code path:

- **Push a `v*` tag.** The ordinary route.
- **Push a `release/v*` branch.** `release/v0.5.0` cuts `v0.5.0` from that
  branch's head. The branch is only a trigger and is safe to delete once the
  release is published.
- **Run the workflow manually** (Actions → Release → Run workflow), naming the
  tag and optionally the branch or commit to cut it from.

The last two exist because pushing a tag needs credentials scoped to
`refs/tags/*`, and an automation session's token is often scoped to branches
only; dispatching a workflow needs `actions: write`, which the same integrations
tend not to have. A branch push is the write such a token does have. The
workflow's own `GITHUB_TOKEN` has `contents: write` and creates the tag itself,
and a push made with it does not trigger another run.

The tag is validated against `v1.2.3` / `v1.2.3-rc.1` before it reaches a
shell. The release build stamps the version through `GOSHCODER_VERSION`
(`make dist VERSION=<tag>`); an ordinary `cargo build` reports the manifest
version from `Cargo.toml`, and the Makefile, `install.sh` and `install.ps1`
each carry the same fallback for the case where no tag is reachable. Bump all
four after a release; skipping it is how two Go-era releases shipped from a
tree that still claimed the previous version.

## Conventions to keep

- Every ported file opens with a comment naming its pi source path, and
  documents deviations inline where behaviour differs.
- Comments explain *why*, not *what*.
- Tests assert observable behaviour (the request that reaches the wire, the
  events emitted, the bytes on disk), not internal plumbing, and include
  negative controls. Tests are hermetic: injected environments, temporary
  agent directories, loopback fake servers. CI runs the suite a second time
  with deliberately wrong `AWS_*` values to catch a test that reads the
  developer's real environment.
- When a test fails, check whether the test or the implementation is wrong.
  Several failures in this project's history were bad fixtures or wrong
  expectations about pi's replay rules.
- No new crate without a reason that survives `cargo audit` and the licence
  collection in `make dist`.
