# GoshCoder — continuation notes

Handoff for a fresh session. Last verified 2026-08-12.

## Verify state first

```sh
export PATH="$PATH:/c/Program Files/Go/bin"   # Go is not on PATH in fresh shells
cd /c/Users/vaugh/OneDrive/Desktop/GoshCoder
go build ./... && go vet ./... && gofmt -l ./internal ./cmd && go test ./...
```

Expected: all clean across 8 packages. Zero third-party dependencies (no
`go.sum`) — keep it that way.

The pi reference clone is at `reference/pi` (gitignored). Every port is written
against it; read the TS before porting anything.

## Goal

User's priorities, in their words: **Codex, the Chinese models, and Anthropic.
GitHub Copilot explicitly not wanted.**

The five-step plan agreed for that:

| Step | What | State |
| --- | --- | --- |
| 1 | Verify Chinese providers work | **Done** (found + fixed a real bug) |
| 2 | OAuth refresh infrastructure | **Done**, tested |
| 3 | PKCE + loopback login, Anthropic | **Done**, tested and wired to CLI |
| 4 | Codex OAuth | **Done**, tested and wired to CLI |
| 5 | `openai-codex-responses` protocol (SSE-only) | **Done**, tested end to end |

## Step 1 — done, and it found a bug worth knowing about

Chinese providers need **no OAuth at all** except Kimi. They are plain API-key
providers on already-ported protocols, reachable today via
`goshcoder auth set <provider>` or the env var:

| Provider | Models | Protocol | Env var |
| --- | --- | --- | --- |
| moonshotai / -cn | 10 / 10 | openai-completions | `MOONSHOT_API_KEY` |
| qwen-token-plan (×3) | 16 / 16 / 7 | openai-completions | `QWEN_TOKEN_PLAN_API_KEY` |
| zai / zai-coding-cn | 4 / 4 | openai-completions | `ZAI_API_KEY` |
| xiaomi (×4) | 6 / 3 / 3 / 3 | openai-completions | `XIAOMI_API_KEY` |
| deepseek | 2 | openai-completions | `DEEPSEEK_API_KEY` |
| ant-ling | 3 | openai-completions | `ANT_LING_API_KEY` |
| minimax / -cn | 3 / 3 | anthropic-messages | `MINIMAX_API_KEY` |
| kimi-coding | 4 | anthropic-messages | OAuth **or** `KIMI_API_KEY` |

**The bug:** `llm.Model.Compat` is typed `*OpenAICompletionsCompat`, so the
other four protocols could only read compat from their options struct — which
nothing outside tests populated. 257 models were silently losing their compat,
including `kimi-coding`'s `forceAdaptiveThinking` and every Anthropic
adaptive-thinking flag.

Fix: added `Model.RawCompat json.RawMessage` (`json:"-"`) plus
`Model.DecodeRawCompat(target)`. The catalog populates it in `toModel`, and
`cloneModel` copies it explicitly because `json:"-"` would drop it on the JSON
round-trip. All four protocols now fall back to it when options carry no compat.
Covered by `internal/integration/compat_test.go`, including a broad guard
(`TestEveryProtocolReceivesItsCatalogCompat`) that fails if any catalog model's
compat can't reach its protocol.

**Still unverified:** none of these providers has been exercised against a real
endpoint. Worth one live smoke test each with a real key.

## Step 2 — done

`internal/llm/catalog/oauth.go`. The `OAuthProvider` interface splits
`Refresh` (network, produces a credential) from `ToAuth` (pure, derives request
auth), which is what lets the refresh run inside `CredentialStore.Modify` — so
the existing cross-process lock file prevents double-refresh of a rotated token.
Expiry is re-checked under the lock (double-checked locking, as pi does).

Registered refreshers: `anthropic`, `kimi-coding`, `openai-codex`.

Shape differences that matter:
- Anthropic `ToAuth` → `apiKey`, and subtracts a 5-minute skew on expiry.
- Kimi `ToAuth` → `Authorization: Bearer` **header**, and refresh retries
  429/5xx with cancellable exponential backoff.
- Codex `Refresh` extracts `chatgpt_account_id` from the access token's JWT
  claims and stores it as a credential extra (`accountId`).

`resolveAuth` now calls `resolveOAuth`, which refreshes when expired. Two
deliberate behaviours: a **failed refresh never falls back to ambient env keys**
(a stored credential owns the provider), and a **transient failure preserves the
stored refresh token** so it stays recoverable. Failures are recorded in
`Catalog.OAuthError(providerID)` so the CLI can say "your login expired" rather
than "not configured".

Also added `Credential.SetExtra`/`Extra` — the `extra` field existed for
round-tripping unknown auth.json fields but had no accessor.

Token URLs were changed from consts to **vars** so tests can point them at a
fake server (`anthropicTokenURL`, `codexTokenURL`, `codexDeviceUserCodeURL`,
`codexDeviceTokenURL`). Tests restore them via `t.Cleanup`.

Tested in `internal/llm/catalog/oauth_test.go`.

## Step 3 + 4 — done

`internal/llm/catalog/oauth_login.go` compiles, is gofmt/vet clean, and contains:

- `LoginInteraction` interface (`Prompt`/`Notify`) with prompt kinds `text`,
  `secret`, `select`, `manual_code`, and `LoginEvent` kinds `info`, `auth_url`,
  `device_code`, `progress`.
- `generatePKCE` (32 random bytes → base64url verifier, SHA-256 challenge),
  `randomState`.
- Loopback `callbackServer` with state validation, plus `openBrowser`
  (rundll32 / open / xdg-open) and `parseAuthorizationInput` accepting a full
  redirect URL, `code#state`, a query fragment, or a bare code.
- `runLoopbackLogin` — races the callback against a `manual_code` prompt so
  headless/remote sessions work; whichever wins cancels the other.
- `anthropicOAuth.Login` — PKCE loopback, port 53692, path `/callback`.
  Anthropic uses the PKCE **verifier as the state**, which its token endpoint
  expects echoed back.
- `openAICodexOAuth.Login` — select between browser (port 1455,
  `/auth/callback`) and device code; `exchangeCode` attaches `accountId`.
- `pollDeviceCode` — RFC 8628 polling with a deadline and cancellation.
- `kimiCodingOAuth.Login` — RFC 8628 device authorization with pending,
  slow-down, expiry, denial, timeout, and cancellation handling.
- `Catalog.Login(ctx, providerID, interaction)` — runs the flow and persists via
  `Modify`, clearing any recorded OAuth error.

`internal/llm/catalog/oauth_login_test.go` covers PKCE, all authorization-input
shapes, callback state validation, both sides of the manual/callback race,
device polling, Anthropic login persistence, and Codex code exchange. Browser
opening is injectable in tests. `goshcoder auth login <provider>` now provides a
line-oriented `LoginInteraction`; its prompt rendering and cancellation are
covered in `cmd/goshcoder/main_test.go`.

Kimi login is covered against a fake device authorization/token server,
including URL trust validation. It remains available through `KIMI_API_KEY` as
well.

## Step 5 — done

`internal/llm/openai_codex_responses.go` ports the complete uncompressed SSE
path for all seven Codex models. It reuses the shared Responses message/tool
conversion and stream processor, adds Codex request construction, JWT account
extraction, required headers, URL resolution, response event normalization,
service-tier pricing, and `end_turn` handling. WebSocket reuse and optional
zstd compression are deliberately omitted; the backend supports SSE with plain
JSON.

Wire-level tests cover request bodies, auth/session headers, stream events,
nested errors, endpoint normalization, invalid tokens, and registration.
`TestEndToEndCodexOAuthTurn` covers the full stored OAuth credential → catalog →
agent → Codex wire protocol path. Catalog protocol coverage is now **1220/1220
models (100%)**.

## Native extension ports

The two user-requested packages are now native:

- `internal/plannotator`: planning/executing/idle state machine, markdown-only
  planning write gate, `plannotator_submit_plan`, loopback browser review with
  approve/deny/notes and clickable line annotations, persisted per-workspace
  state, `[DONE:n]` checklist tracking, and chat commands for plan, git-diff,
  file, and last-message review. `-plan` starts planning immediately; chat
  always loads `/plannotator` so it can be toggled later.
- `internal/claudetui`: width-aware startup card, model/thinking/cwd display,
  a half-open rounded prompt, and an OpenCode-inspired session sidebar showing
  context, cost, transcript/tool counts, changed files, branch, and mode. It
  refreshes automatically after turns and state changes; `/status` also prints
  it on demand. It defaults on in chat; `-claude-tui=false` and the
  `/use-claude-code-tui`/`/use-default-tui` commands control it.

Both are dependency-free adaptations. They do not load npm or require pi's
full-screen TypeScript TUI. Plannotator intentionally serves a compact native
review page instead of embedding its roughly 39 MB generated SPA assets.

## Conventions to keep

- Every ported file opens with a comment naming its pi source path, and
  documents deviations inline where behaviour differs.
- Comments explain *why*, not *what*. No narration of obvious code.
- Tests assert observable behaviour (the request that reaches the wire, the
  events emitted), not internal plumbing. Include negative controls — several
  bugs here were only caught because a test checked the *other* branch too.
- When a test fails, check whether the test or the implementation is wrong. Four
  of the failures during this work were bad test fixtures, and one
  (`TestBedrockReplaysAssistantAndToolResults`) was a wrong expectation about
  cache points.
- `TransformMessages` normalization is **cross-model only**. Same-model replay
  keeps original tool call ids and thinking signatures. This tripped up test
  expectations three separate times.

## Known deviations from pi (unchanged)

- Compat travels on options structs, with `Model.RawCompat` as the fallback.
- No vendor SDKs: hand-rolled `net/http` + SSE reader. Bedrock does SigV4 and
  binary event-stream framing against stdlib crypto.
- `goshcoder chat` is line-oriented, not a full-screen TUI.
- Selected pi extensions are native built-ins: `internal/ralph`
  (`@tmustier/pi-ralph-wiggum`), `internal/plannotator`
  (`@plannotator/pi-extension`), and `internal/claudetui`
  (`pi-claude-code-tui`). Plannotator uses a compact stdlib browser reviewer,
  and the Claude-style UI is adapted to the line-oriented chat rather than
  importing pi's npm/TUI runtime.
