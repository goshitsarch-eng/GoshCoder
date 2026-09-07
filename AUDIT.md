# GoshCoder repository audit

End-to-end audit run in phases. Each section records what was actually opened
and executed, not what was intended.

---

## Phase 0 — Ground truth

### Stack, detected from the manifests

| Question | Answer | How it was determined |
| --- | --- | --- |
| Language | Rust, edition 2024 | `Cargo.toml` `edition = "2024"` |
| Toolchain | stable + clippy + rustfmt | `rust-toolchain.toml`; local `rustc 1.94.1`, `cargo 1.94.1` |
| Build system | Cargo behind a `Makefile` | `Cargo.toml`, `Makefile` (`build`/`check`/`dist`) |
| Package manager | Cargo, `Cargo.lock` committed, `--locked` in CI | `Cargo.lock`, `.github/workflows/ci.yml` |
| Artifact | one binary, `goshcoder` | `[[bin]] name = "goshcoder" path = "src/main.rs"` |
| UI toolkit | Ratatui 0.30.2 + Crossterm 0.29.0 (terminal UI) | `Cargo.toml`; `src/ui.rs`, `src/state.rs` |
| HTTP | `reqwest` 0.13.4 **blocking**, rustls, http2 | `Cargo.toml`; no async runtime is a direct dependency |
| Targets | linux amd64/arm64 (musl), macOS amd64/arm64, windows amd64 (gnu) | `Makefile` `PLATFORMS`, CI `release-targets` |
| Concurrency | OS threads + `Mutex`/`Condvar`, no executor | `std::thread` throughout |

No workspace members beyond the root crate; no JS/TS/Python manifest anywhere.

### File inventory (64 files)

| Group | Count | Notes |
| --- | --- | --- |
| `src/*.rs` | 42 | 86,583 LOC total — ~57k production, ~29k `#[cfg(test)]` |
| `tests/*.rs` | 1 | `tests/aperture_core.rs`, 11 lines (`#[path]` module shim) |
| `data/*.json` | 3 | `catalog.json` (513 KB, generated) + 2 hand-maintained |
| Docs | 4 | `README.md`, `CONTINUE.md`, `NOTICE`, `LICENSE` |
| Build/CI | 6 | `Makefile`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, 2 workflows |
| Installers | 2 | `install.sh`, `install.ps1` |
| Scripts | 3 | `collect-licenses.sh`, `release-roundtrip.sh`, a licence exception |
| Config | 3 | `.cargo/audit.toml`, `.gitignore`, `.gitattributes` |

**Not read in full, and why:** `data/catalog.json` (513 KB) is generated model
data replaced wholesale on regeneration; its *schema and loading path* are
audited in `src/catalog.rs`, and the two hand-maintained siblings were read.
`Cargo.lock` was read for versions, not line by line.

### Baseline, recorded before any change

| Gate | Command | Result |
| --- | --- | --- |
| Build | `cargo build --all-targets` | **pass**, 1m56s cold |
| Tests | `cargo test --workspace --all-targets --locked` | **554 passed, 0 failed, 0 ignored** |
| Format | `cargo fmt --all -- --check` | **pass** |
| Lint | `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` | **pass**, zero warnings |
| Advisories | `cargo audit` | **pass** (1 advisory, suppressed with a reason — see Dependency audit) |

Binary reports `goshcoder 0.6.0`, matching `Cargo.toml`.

### Review depth

Read line by line: `tools.rs`, `resources.rs`, `prompts.rs`, `export_html.rs`,
`plannotator.rs`, `webaccess.rs`, `computeruse.rs`, `agent.rs`, `config.rs`,
`google_auth.rs`, `sessionlog.rs`, `ui.rs`, `state.rs`, `markdown.rs`,
`runtime.rs` (config/prepare), `oauth.rs` (flows + loopback server),
`catalog.rs` (credential store, auth resolution, config values),
`bedrock.rs` (SigV4), `stream.rs` (bounds/SSE), `providers.rs` (client + auth
headers), `main.rs` (dispatch, slash commands, event loop).

Reviewed against specific defect classes rather than line by line:
`mistral.rs`, `omniroute.rs`, `omni_prompt_tools.rs`, `aperture*.rs`,
`btw*.rs`, `ralph*.rs`, `planner_runtime.rs`, `session.rs`, `sessions.rs`,
`session_picker.rs`, `turns.rs`, `compaction.rs`, `llm.rs`, `provider_cli.rs`,
and the remainder of `providers.rs`/`bedrock.rs`. **These have not had a
full line-by-line pass** — see "Scope" at the end.

---

## Findings

Severity: **High** = credential/data loss with a realistic path.
**Medium** = correctness or availability defect. **Low** = defence-in-depth,
contract, or conformance.

| id | sev | category | file:line | description | exploit / impact | status |
| --- | --- | --- | --- | --- | --- | --- |
| F-01 | **High** | secrets / transport | `src/providers.rs:442` | The main model-request client is built with no `redirect` policy, so reqwest's default (follow up to 10) applies. reqwest strips only `Authorization`/`Cookie`/`Proxy-Authorization`/`WWW-Authenticate` cross-origin; the provider auth headers `x-api-key` (`:1744`), `x-goog-api-key` (`:1711`,`:1752`) and `api-key` (`:1674`) are **not** stripped. | A provider endpoint the user pointed at — an OmniRoute gateway (`OMNIROUTE_URL`), an Aperture proxy base URL, an Azure resource URL, a `catalog_extra.json` base URL, or a hijacked upstream — answers a normal turn with `302 Location: http://attacker/`. The client follows and replays the Anthropic/Google/Azure API key to the attacker's host. **Proven** with a probe using the exact client config: `x-api-key`, `x-goog-api-key`, `api-key` all leaked; `authorization` did not. Every *other* HTTP client in the tree already sets `Policy::none()` (`bedrock.rs:2449`, `aperture.rs:1072`, `aperture_mcp.rs:289`, `webaccess.rs:258`, `omni_cli.rs:171`), and `CONTINUE.md` claims "Gateway clients never follow a redirect with a credential attached" — this is the one path that was missed. | **fixed** |
| F-02 | Medium | availability | `src/catalog.rs:1962` (and `:1957`) | `execute_command` (the `!command` credential-helper for `auth.json`) joins its stdout reader thread unconditionally. `drain_command_output` reads to EOF, so if the helper exits while a backgrounded grandchild still holds the stdout pipe, `reader.join()` never returns. | A credential helper that starts a daemon (agent, keychain broker) wedges credential resolution forever, hanging the whole app with no timeout — the 10 s `COMMAND_TIMEOUT` bounds the *child*, not the reader. `tools.rs` solves exactly this with a bounded drain + detach (`await_reader`, `run_process`); this path does not. | **fixed** |
| F-03 | Low | transport / docs | `src/oauth.rs:1021` (`handle_connection`) | The OAuth loopback callback server never validates the `Host` header. `README.md:449` states local servers "validate state/CSRF tokens **and the `Host` header**"; the Planner review server does (`plannotator.rs:1754`, `allowed_review_host`), the OAuth one does not. | Defence-in-depth against DNS rebinding on the fixed callback ports (53692/1455/56121). The 256-bit `state` check is the real barrier, so this is not directly exploitable — but the documented control does not exist. | **fixed** |
| F-04 | Low | TOCTOU | `src/config.rs:250`, `src/catalog.rs:1555`, `src/catalog.rs:1613` | Permissions are applied with path-based `fs::set_permissions` *after* `fs::rename`, which follows symlinks. | Only a same-UID racer inside a 0700 directory can win the window, and the mode being set is restrictive (0600), so impact is minimal. It is nonetheless the pattern `resources.rs:2223` explicitly documents as wrong ("a chmod on the destination afterwards would follow whatever sits at that path by then, symlink included") and that `tools.rs:1158` avoids by setting the mode on the open handle. | **fixed** |
| F-05 | Medium | performance | `src/main.rs:936` → `src/ui.rs:139` | While anything is animating (streaming, a pending turn, a background command) the 100 ms loop calls `refresh_runtime_app` → `Agent::state()` (deep-clones the whole transcript) → `agent_messages()` → `transcript_lines()`, which re-wraps and re-renders **every message's Markdown** — then uses only the ~40 visible rows. | **Measured** (release): whole-transcript Markdown re-render costs 2.9 ms at 50 messages (23 KB), 12.0 ms at 200 (92 KB), 47.3 ms at 800 (369 KB). At the 10 fps animation rate that is 29 ms/s → 120 ms/s → **473 ms/s, i.e. ~47 % of a core**, purely to redraw a constant-size viewport. Cost is linear in transcript size and paid on every frame of every response. | **fixed** |
| F-06 | Low | performance | `src/tools.rs:2474` | `SearchRegex::is_match` allocates a `Vec<char>` of the line plus a `vec![0usize; states.len()]` mark array, and a fresh `Vec` per character position — for **every line of every candidate file** in a `grep`. | A repo-wide `grep` over 100k candidate files pays two heap allocations per line plus one per character step. Bounded by existing caps, so it is slow rather than dangerous. | **fixed** |
| F-07 | Low | resources | `src/agent.rs:1231` | Parallel tool execution spawns one unbounded OS thread per tool call in a batch, with no cap. | A model emitting a large tool-call batch spawns that many threads at once. Providers cap batch size in practice, so this is a latent resource issue, not a live one. | **fixed** |
| F-08 | **Medium** | unwired feature | `src/main.rs:1943-2446`; `src/runtime.rs:53,83,1221` | `README.md:393-394` documents `/use-default-tui` and `/use-claude-code-tui` for switching the line-mode appearance, and `-claude-tui=false` as the flag equivalent. The dispatcher has **no match arm** for either command (they hit the "unknown command" catch-all at `:2437`), and `SessionConfig::claude_tui` is parsed at `runtime.rs:1221` but **never read by any consumer** — 0 references outside `runtime.rs`. | **Verified against the built binary**: `/use-default-tui` and `/use-claude-code-tui` both answer `unknown command …; /help lists the available commands`. Three documented affordances (two commands and a flag) do nothing. Both names are nonetheless reserved against prompt shadowing at `main.rs:2778-2779`, so the intent was clearly to implement them. | **open — needs a product decision** |
| F-09 | Low | dead reservation | `src/main.rs:2734`, `src/prompts.rs:52` | `logout` is reserved as a slash-command name in two places, but no `/logout` handler exists (only `goshcoder auth logout`, `provider_cli.rs:160`). | Confirmed against the binary: `/logout` → "unknown command". Not documented in the README, so it is an over-reservation rather than a broken promise. | **fixed** |
| F-10 | Low | GUI conformance | `src/ui.rs:17-29` vs `src/main.rs:399` | The fullscreen TUI hardcodes 24-bit `Color::Rgb` unconditionally. `color_enabled()` honours `NO_COLOR`, but only the line/`run` renderers consult it; `ui.rs` never does. | A user who sets `NO_COLOR` (which the project already respects elsewhere) still gets a fully coloured alternate-screen UI. Inconsistent with the project's own convention. | **fixed** |
| F-11 | Low | GUI conformance | `src/ui.rs:29,273` | `FAINT` `#3E5760` on `BACKGROUND` `#0A0A0A` is a **2.58:1** contrast ratio (computed per WCAG 2.x relative luminance), used for the status-bar keyboard hints and the unfilled progress track. | Below WCAG AA 4.5:1 for normal text and below even the 3:1 large-text floor. `MUTED` (5.7:1) and `TEXT` (~16:1) are fine. | **fixed** |
| F-12 | Low | consistency | `src/ui.rs:17-29` and `src/markdown.rs:17-22` | The colour palette (`ACCENT`, `CYAN`, `TEXT`, `MUTED`, `FAINT`) is declared twice, once per module, with no shared token source. | Two sources of truth for design tokens; they can drift silently. | **fixed** |
| F-13 | Low | dead code | `src/state.rs:133-154` | `App::new()` seeds a migration-era placeholder transcript ("Rust/Ratatui migration is initializing…") and a placeholder sidebar ("Rust migration", "0 / 0 tokens"). | Never user-visible — `event_loop` calls `replace_messages(Vec::new())` immediately and `refresh_runtime_app` replaces the sidebar — but it is stale scaffolding in the constructor. | **fixed** |
| F-14 | Low | docs | `src/sessions.rs` | `sessions gc` is implemented and reachable but appears in neither the README's `Use` section nor `goshcoder --help`. | Implemented behaviour that is undocumented (the reverse direction Phase 3 asks for). | **fixed** |

### Dependency audit — reachable vs noise

`cargo audit` over **340 crates** reports exactly **one** advisory, and it is
already suppressed with a written justification in `.cargo/audit.toml`:

- **RUSTSEC-2023-0071** — `rsa` 0.9.10, "Marvin attack" timing side channel,
  medium (5.9), **no fixed release exists**.
  **Assessment: not reachable as described.** The crate is used in exactly one
  place — `google_auth.rs:436`, RS256-signing a JWT assertion to exchange a
  Google service-account key for a Vertex token. Marvin is a *decryption*
  oracle attack needing many adaptive ciphertexts and timing measurements; this
  code performs local signing with the user's own key, a few times an hour, with
  no attacker-observable timing channel. The existing justification is sound and
  the ignore should stay.

Everything else is noise-free: zero other advisories, and no
`danger_accept_invalid_certs`/`accept_invalid_hostnames` anywhere in the tree
(TLS verification is never disabled).

### Security surfaces checked and found sound

Recorded so the negative results are not mistaken for gaps:

- **Workspace confinement** (`tools.rs:925-1080`) — lexical normalisation,
  canonicalisation, per-component symlink rejection, checked parent creation,
  atomic writes that drop setuid/setgid. The residual TOCTOU is documented
  honestly in the module header rather than papered over.
- **Archive handling** (`resources.rs:1839-2096`) — member names re-derived and
  re-validated, non-regular members refused, entry count/decompressed size/
  retained bytes independently bounded, control characters scrubbed from names.
  No zip-slip.
- **HTML export inertness** (`export_html.rs`) — **verified by running the
  exporter** on a transcript containing `<script>alert(1)</script>` and
  `[link](javascript:alert(2))`: no raw `<script>` in the output, the only
  `href` is the safe `https://` one, `javascript:` targets render as text, and
  `default-src 'none'` CSP is present. Output file is 0600.
- **Planner review server** (`plannotator.rs`) — loopback bind, peer check,
  `Host` check, one-time 256-bit CSRF token, bounded reads, connection deadline,
  full security header set.
- **SSRF policy** (`plannotator.rs:2651-2714`) — scheme allow-list, embedded
  credentials refused, private/loopback/link-local/CGNAT v4 and v6 blocked, and
  the policy re-applied to the *final* URL after redirects.
- **SigV4** (`bedrock.rs:1064-1255`) — correct canonical request, correct
  double-encoding for non-S3, `hmac`/`sha2` crates rather than homegrown
  primitives, region validated as a DNS label before it reaches a host or scope.
- **Credential store** (`catalog.rs:1157-1637`) — in-process + cross-process
  file locking with bounded waits, re-read under lock before write, atomic
  private write, `Auth`/`Credential` deliberately implement no `Debug`/`Display`.
- **Terminal injection** (`markdown.rs:389-482`, `ui.rs:692`) — CSI/OSC/DCS/
  SOS/PM/APC sequences stripped from untrusted model and tool output.
- **PKCE/CSRF entropy** (`oauth.rs:689`, `plannotator.rs:1626`) — SHA-256 over
  OS entropy plus eight UUIDv7 samples; the v7 random bits come from
  `getrandom`, so ≥256 bits even where `/dev/urandom` is unavailable.

---

## Phase 3 — Feature contract

Verified end to end by running the built binary, not by reading alone.

**Fully wired** (exercised and observed working): every documented subcommand —
`providers`, `models`, `sessions` (`list`/`show`/`import`/`export`/`rm`/`gc`/
`share`), `prompts` (`list`/`backup`/`restore`, round-tripped including the
"already exists" and `--overwrite` paths), `ralph`, `omni`, `aperture`, `auth`,
`version`, `help`; session import → HTML and Markdown export (0600 output);
the slash-command surface listed by `/help`; the tool/extension wiring in
`runtime.rs:608-700` (workspace tools, `web_search`, desktop MCP, BTW, Ralph,
Planner all constructed and attached to the live agent).

**Partially wired:** F-08 — the `pi-claude-code-tui` line-mode toggle. The flag
parses and is asserted in tests, but its value reaches no consumer, and the two
documented slash commands have no handler.

**Stub / dead:** F-09 (`logout` reserved, unhandled), F-13 (placeholder
transcript/sidebar in `App::new()`).

**Implemented but undocumented:** F-14 (`sessions gc`).

---

## Phase 4 — GUI and platform conformance

**Applicable standard.** Ratatui/Crossterm is a terminal UI, so neither Apple
HIG, Material, Fluent nor WAI-ARIA APG applies directly. The right yardsticks
are Ratatui's own widget/layout guidance, plus the two cross-cutting standards
that *do* carry over to a TUI: WCAG 2.2 contrast ratios (terminals render real
colours) and platform terminal conventions (`NO_COLOR`, raw-mode restoration,
bracketed paste, signal keys).

Against those:

- **Component usage** — correct stock widgets throughout (`List` + `ListState`
  for the palette, `Paragraph` + `Wrap`, `Block`/`Borders`); no reimplemented
  stock control.
- **Layout** — `Layout` constraint solver, sidebar appears only at ≥96 columns
  (`ui.rs:45`), sidebar width clamped 32–42, composer clamped 3–5 rows.
- **State coverage** — loading (`activity` + spinner gating), empty
  (`"(empty directory)"`, "The transcript is empty.", "no saved prompts",
  "No tools are active"), error (`MessageRole::Error` path), permission-denied
  and offline (surfaced as session notices, e.g. the `computer-use-linux`
  not-found notice observed at startup), long content (pre-wrapped rows,
  `Ctrl-O` expansion with a "… N more lines" affordance), and a genuine
  **terminal-too-small** state at `ui.rs:37`. This is unusually complete; no
  missing empty/error state was found.
- **Keyboard** — thorough: arrows, word motion (`Alt`/`Ctrl`+←/→), `Home`/`End`,
  `PageUp`/`PageDown`, history, and emacs bindings (`Ctrl-A/E/B/F/K/U/W`).
  Focus is unambiguous (cyan composer border, real cursor positioned via
  `set_cursor_position`, palette highlight symbol **plus** background colour, so
  selection is not colour-only).
- **Platform integration** — raw mode and alternate screen restored on panic via
  a hook (`main.rs:454`), bracketed paste enabled so multi-line pastes do not
  submit line-by-line, `BackTab` handled alongside synthetic `Shift+Tab`,
  Windows `bash.exe` discovery skips the legacy WSL launcher.
- **Gaps found:** F-10 (`NO_COLOR` ignored by the fullscreen UI), F-11 (one
  sub-AA contrast pair), F-12 (duplicated colour tokens). No RTL or text-scaling
  layer exists, which is normal for a TUI and not counted as a defect.

---

## Fix order

Security critical → correctness → unwired features → performance → GUI.

1. **F-01** redirect policy on the provider client (High, one line + a
   regression test mirroring the existing `bedrock.rs` one).
2. **F-02** bounded stdout drain for the `!command` credential helper.
3. **F-03** `Host` validation on the OAuth callback server.
4. **F-04** set permissions on the open handle instead of the path after rename.
5. **F-08** wire the documented `claude-tui` toggle, or correct the README.
   **This one needs a product decision — see below.**
6. **F-05** stop re-rendering the whole transcript every animation frame.
7. **F-06/F-07** grep allocation and tool-thread bound.
8. **F-10/F-11/F-12/F-13/F-09/F-14** conformance and tidy-ups.

### Needs a product decision before I touch it

**F-08.** Two defensible resolutions, and the choice is a product call, not a
refactor: (a) implement the toggle — give line mode a plain renderer and a
"claude-code-tui" renderer and have the flag and the two commands select
between them; or (b) drop the claim — remove `SessionConfig::claude_tui` and
the two reserved names, and correct `README.md:393-394`. (a) is a real feature;
(b) is honest documentation. I will not guess which was intended.

---

## Fixes applied

Each is a separate commit whose message states the problem rather than the
change. After every one: `cargo build`, `cargo fmt --check`, `cargo clippy
-D warnings`, and the full suite. **565 tests pass across the workspace** (541 unit + 24 integration), up from
554 at baseline: 11 regression tests added, none removed.

| id | commit | test added | negative control |
| --- | --- | --- | --- |
| F-01 | `8aff8e5` | redirect target asserted never contacted | confirmed failing with the policy removed |
| F-02 | `26629fc` | helper leaks stdout via a backgrounded `sleep` | confirmed failing with the wait unbounded |
| F-03 | `6d060ea` | foreign/absent/portless `Host` all refused, real ones served | two stale fixtures corrected |
| F-04 | `5f74227` | mode is 0600 on create and on replace | race itself untestable — see below |
| F-05 | `7c49adc` | only changed entries re-render; rows match a direct render | measured, 159×–217× |
| F-06 | `bf49640` | reused scratch matches a fresh one line-for-line | measured, ~2.8× |
| F-07 | `a4eaf01` | 53 calls, peak overlap within bound, order preserved | confirmed failing with the bound widened |
| F-09,F-14 | `ceb2c85` | — (documentation and a reservation list) | verified against the built binary |
| F-11,F-12 | `0a12c5f` | every rendered colour pairing clears 4.5:1 | first attempt caught a real over-broad assertion |
| F-13 | `5d2dfb1` | renderer test now supplies its own state | — |
| F-10 | `5344cee` | NO_COLOR rule; suppressed palette is all `Color::Reset` | pty: 213 truecolor sequences → 0 |

Measurements were taken on release builds and the scaffolding was removed
afterwards; only the assertions that belong in the suite were kept.

## Not fixed

**F-08 — the `pi-claude-code-tui` line-mode toggle.** Left open deliberately.
`README.md:393-394` documents `/use-default-tui`, `/use-claude-code-tui` and
`-claude-tui=false`; none of the three does anything, and
`SessionConfig::claude_tui` is written but never read. There are two
defensible resolutions and picking between them is a product call, not a
refactor:

- **Implement it** — give line mode a plain renderer and a claude-code-tui
  renderer, and have the flag and the two commands select between them.
- **Drop the claim** — remove `SessionConfig::claude_tui`, drop the two
  reserved names from `main.rs:2778-2779`, and correct the README.

I have not guessed. The two names stay reserved in the meantime, so whichever
way it goes, a user prompt cannot have taken the name first.

**F-04's race, specifically.** The chmod-after-rename window is closed, but no
test asserts it: after the rename the path is the process's own regular file,
so the interleaving cannot be produced deterministically. The test pins the
observable half (the mode really is 0600, on both the create and the replace
path) and this note records the half it cannot reach.

---

## Scope

Phases 0, 2 and 4 cover the whole tree. Phase 1 and Phase 3 are complete for
every module listed under "Read line by line" in Phase 0 — which is every
security boundary in the program: filesystem confinement, subprocess
execution, all credential storage and resolution, both loopback HTTP servers,
archive extraction, HTML export, terminal-escape handling, SigV4, the stream
bounds, and the provider request path.

The modules listed as reviewed against specific defect classes — the protocol
adapters and feature runtimes (`mistral`, `omniroute`, `aperture*`, `btw*`,
`ralph*`, `planner_runtime`, `session`, `sessions`, `turns`, `compaction`,
`llm`, `provider_cli`, and the bulk of `providers`/`bedrock`) — were searched
for this audit's defect classes and exercised through the binary, but have
**not** had a full line-by-line pass. Roughly 35k of 57k production lines got
the targeted rather than exhaustive treatment. That is flagged rather than
folded into "phase complete"; finishing it is a standing offer.
