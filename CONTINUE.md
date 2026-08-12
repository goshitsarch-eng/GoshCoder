# GoshCoder — continuation notes

Handoff for a fresh session. Last verified 2026-08-12.

## Verify state first

```sh
export PATH="$PATH:/c/Program Files/Go/bin"   # Go is not on PATH in fresh shells
cd /c/Users/vaugh/OneDrive/Desktop/GoshCoder
go build ./... && go vet ./... && gofmt -l ./internal ./cmd && go test ./...
```

Expected: all clean, 659 tests passing across 8 packages. Zero third-party
dependencies (no `go.sum`) — keep it that way.

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
| 3 | PKCE + loopback login, Anthropic | **Code written, NOT tested, NOT wired** |
| 4 | Codex OAuth | Code written alongside step 3, same gaps |
| 5 | `openai-codex-responses` protocol (SSE-only) | **Not started** |

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

## Step 3 + 4 — code written, three gaps

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
- `Catalog.Login(ctx, providerID, interaction)` — runs the flow and persists via
  `Modify`, clearing any recorded OAuth error.

### The three gaps, in the order they should be closed

1. **No tests.** Nothing in `oauth_login.go` is covered. The endpoints are
   already vars, so a fake token server works the same way step 2's tests do.
   Worth covering: PKCE challenge correctness, `parseAuthorizationInput`'s four
   input shapes, callback state mismatch rejection, the manual-vs-callback race
   (both directions), and `pollDeviceCode`'s pending/timeout/cancel paths.
   `openBrowser` should not be invoked by tests — consider making it injectable.

2. **No CLI wiring.** `goshcoder auth login <provider>` does not exist and there
   is no `LoginInteraction` implementation. Needs a stdin/stderr one in
   `cmd/goshcoder`. Note `PromptSecret` currently cannot suppress echo —
   `readSecret` in `main.go` already warns "(input is visible)"; same limitation
   applies. Add `login` to the `authCommand` switch and the usage text.

3. **Kimi has no `Login`.** `kimiCodingOAuth` implements `OAuthProvider` but not
   `OAuthLoginProvider`, so `goshcoder auth login kimi-coding` would report no
   flow. It is RFC 8628 device code against `{host}/api/oauth/device_authorization`
   then `/api/oauth/token`, client id `17e5f671-d194-4dfb-9706-5516cb48c098`,
   host overridable via `KIMI_CODE_OAUTH_HOST`/`KIMI_OAUTH_HOST`. `pollDeviceCode`
   already exists. Reference: `reference/pi/packages/ai/src/auth/oauth/kimi-coding.ts`.
   Lower priority — Kimi accepts `KIMI_API_KEY` instead.

Minor: `callbackServer.wait` is dead code (`runLoopbackLogin` selects on
`server.results` directly). Delete it or use it.

## Step 5 — not started

`openai-codex-responses`, 7 models, the last unported protocol. Coverage is
1213/1220 (99.4%) without it.

**Correction to an earlier assessment:** I previously called this too expensive
because of WebSocket + zstd. Re-reading `openai-codex-responses.ts` shows that
was wrong:

- `transport` defaults to `"auto"` but there is a **full SSE path**, and pi
  already falls back to it. WebSocket is not required.
- zstd is **optional** — `compressRequestBodyZstd` returns null when
  unavailable and the `content-encoding` header is simply omitted.
- JWT parsing is already done (`decodeJWTPayload`, `codexAccountID` in
  `oauth.go`).

So an SSE-only port is viable and should reuse the existing shared Responses
layer in `openai_responses_shared.go`. Skip WebSocket, zstd, and the session
websocket cache. Endpoint is `resolveCodexUrl(model.baseUrl)`; headers come from
`buildBaseCodexHeaders` and need the `accountId` extra from the credential.

Note the provider is `AuthOAuthOnly`, so step 3/4 must work before these models
are usable at all.

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
- pi extensions are ported as built-ins; only `internal/ralph`
  (`@tmustier/pi-ralph-wiggum`) is done. Six others remain unported and need a
  priority call from the user — `pi-web-access` alone is ~4.5k lines wrapping a
  dozen commercial search APIs.
