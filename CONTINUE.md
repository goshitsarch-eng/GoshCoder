# GoshCoder — continuation notes

Handoff for a fresh session. Last verified 2026-08-12.

## Verify state first

```sh
export PATH="$PATH:/c/Program Files/Go/bin"   # Go is not on PATH in fresh shells
cd /c/Users/vaugh/OneDrive/Desktop/GoshCoder
go build ./... && go vet ./... && gofmt -l ./internal ./cmd && go test ./...
```

Expected: all packages build, vet, and test clean. The fullscreen UI uses
Bubble Tea/Lip Gloss dependencies recorded in `go.sum`.

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

Registered refreshers: `anthropic`, `kimi-coding`, `meta`, `openai-codex`,
`xai` (the last two of those were added later; see "xAI and Meta OAuth" below).

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

The requested extension features are now native:

- Provider auth is re-resolved for every model request, so long sessions refresh
  OAuth and in-app credential replacement takes effect immediately. The same
  authenticated stream path is used by the nested compaction agent.
- Planner phase changes update both prompt and tools inside the active agent
  loop; approval exposes full tools on the immediately following model turn.
- **Grok Code/xAI**: `xai` uses `XAI_API_KEY` or `auth set xai`; the current
  canonical coding model `grok-build-0.1` and documented aliases
  `grok-code-fast-1`, `grok-code-fast`, and `grok-code-fast-1-0825` are in the
  model picker.
- **OmniRoute** (`internal/omniroute`, `internal/llm/omni_prompt_tools.go`):
  `/omni setup|sync|status|dashboard`, dynamically discovered models in
  `/model` and Ctrl+P, normalized metadata, and prompt-emulated tools for
  web/chat-only models. Config is `omniroute.json`; auth stays in `auth.json`.
- **BTW** (`internal/btw`): native ephemeral side threads with main-transcript
  context, follow-ups, queued Steering, session-memory resume, dedicated
  fullscreen view, independent model/thinking settings, and latest-answer
  bring-to-main. Exact text-range selection is the documented parity exception.
- **Web Search** (`internal/webaccess`): native `pi-web-access` adaptation registered
  as `web_search`, with OpenAI/Codex auth reuse, zero-config Exa MCP fallback,
  direct Exa, and Kagi Search (`KAGI_API_KEY` or `web-search.json`). The in-app
  `/login` picker adds OAuth or API-key providers without replacing existing
  credentials and refreshes `/model` choices immediately.
- **Planner** (`internal/plannotator`): planning/executing/idle state machine,
  markdown-only planning write gate, `planner_submit_plan`, and a polished loopback review
  app with approve/deny, line annotations, notes, direct Markdown edits,
  resubmission changes, themes, responsive navigation, persisted per-workspace
  state, `[DONE:n]` checklist tracking, and chat commands for plan, git-diff,
  file, and last-message review. `-plan` starts planning immediately; chat
  always loads `/planner` so it can be toggled later.
- `internal/claudetui`: model/context/cost/git/mode information used by both
  the startup card and the responsive fullscreen sidebar.
- `internal/tui`: Bubble Tea/Lip Gloss alternate-screen interface with a fixed
  transcript viewport, OpenCode-style responsive sidebar, compact/expandable
  tool cards, a multiline editor, Unicode width handling, history, scrolling,
  live agent updates, and a nested command palette. `/model` searches models
  from authenticated providers and `/thinking` derives its choices from the
  active model's supported level map.
- `internal/resources`: Pi-compatible context files, system prompt overrides,
  prompt templates, and Agent Skills. The CLI supports `/reload`, `/resources`,
  and resource slash expansion.
- `cmd/goshcoder/compaction.go`: manual `/compact [focus]` and near-limit
  automatic structured context compaction while retaining recent turns.

These are native adaptations and do not load npm or pi's TypeScript TUI.
Planner intentionally serves a native review page instead of
embedding its roughly 39 MB generated SPA assets.

## 2026 Go quality and security audit

Audited the full Go tree against Effective Go and the Google Go Style Guide.
The tree is clean under `go vet -all`, Staticcheck, `govulncheck`, normal tests,
repeated tests, and the race detector. Important hardening completed:

- workspace and Ralph filesystem operations use `os.Root`, closing symlink and
  rename race escapes; workspace handles are closed with sessions;
- reads, edits, command output, plans, state, credentials, diffs, annotations,
  SSE records, and provider streams have explicit memory bounds;
- auth and extension state writes use secured temporary files and atomic
  renames; stale auth locks recover safely;
- OAuth and Planner web servers are loopback-only, size/time limited, use
  CSRF/state validation, no-store/CSP headers, and escaped HTML;
- planning mode removes and independently blocks shell execution;
- retry parsing handles malformed, negative, non-finite, wrapped-cancellation,
  and past-date cases correctly;
- subprocesses have contexts and `WaitDelay`; sidebar git inspection uses one
  bounded, two-second command instead of two unbounded subprocesses;
- Ralph archives now move both state and task content atomically enough to
  recover safely from an interrupted operation;
- Go error style, exported comments, initialisms, dead code, and simplification
  findings from Staticcheck were corrected.

Go 1.26.5 is the minimum because earlier 1.26 patch releases contain reachable
standard-library vulnerabilities. Keep it patched and rerun `govulncheck` for
release builds. The fullscreen interface now depends on Bubble Tea and Lip
Gloss; dependency checksums are committed in `go.sum`.

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

## Cutting a release

`.github/workflows/release.yml` publishes the cross-compiled archives and the
`checksums.txt` both installers verify against. Three ways in, one code path:

- **Push a `v*` tag.** The ordinary route.
- **Push a `release/v*` branch.** `release/v0.2.0` cuts `v0.2.0` from that
  branch's head. The branch is only a trigger; it is left in place afterwards
  and is safe to delete once the release is published.
- **Run the workflow manually** (Actions → Release → Run workflow), naming the
  tag and optionally the branch or commit to cut it from.

In the last two, a tag that does not exist yet is created and pushed before
building; a tag that already exists is built as it stands, and the source ref
is ignored.

The two extra routes exist because pushing a tag needs credentials scoped to
`refs/tags/*`, and an automation session's token is often scoped to branches
only. That alone left no way to cut a release -- and the manual route did not
close the gap either, because dispatching a workflow through the API needs
`actions: write`, which the same integrations tend not to have (it answers
`403 Resource not accessible by integration`). A branch push is the write such
a token does have, so `release/v*` is the route that works from inside a
session. The workflow's own `GITHUB_TOKEN` has `contents: write` and creates
the tag itself.

A push made with `GITHUB_TOKEN` deliberately does not trigger another workflow
run, so creating the tag cannot start a second release of the same version.
Cutting a release still requires push access to the repository, exactly as
pushing a tag did; none of this grants anybody anything new.

The tag is validated against `v1.2.3` / `v1.2.3-rc.1` before it reaches a
shell. It becomes a git ref, the archive filenames, and the version the binary
reports (`make dist VERSION=<tag>`, which strips the leading `v` because that
is the name `install.sh` derives from the release tag).

The in-tree version stays at the `-dev` value for the release being prepared:
the release build stamps the real one with `-ldflags`, so an untagged build
never claims to be a release it is not. It is written in four places, because
each build route falls back on its own copy when no tag is reachable --
`cmd/goshcoder/version.go`, the Makefile's `VERSION`, and the same fallback in
`install.sh` and `install.ps1`. `TestDevVersionIsConsistent` fails when they
disagree; bumping after a release is still a manual step, and skipping it is
how `v0.2.0` and `v0.2.1` both shipped from a tree that still said
`0.2.0-dev`. It now reads `0.3.0-dev`.

## xAI and Meta OAuth — done

Both providers accept an API key *and* a subscription/account login; adding the
login left the key path untouched.

`internal/llm/catalog/oauth_xai.go` — xAI has a real first-party OIDC server at
`auth.x.ai`, so this is its published surface rather than a scraped browser
session: discovery, PKCE S256 over a loopback callback on `127.0.0.1:56121`, and
RFC 8628 device code for headless sessions. Two decisions are load-bearing:

- **Discovered endpoints are pinned to the issuer's host.** A discovery document
  is fetched over the network; honoring an endpoint it moved to another host
  would hand that host the authorization code and the refresh token. Discovery
  failing falls back to the documented paths, because the paths are documented
  and an outage there is not a reason to refuse a login. A discovery document
  that *lies* is not advisory, and is refused.
- **The device flow is offered first.** It needs no listening socket and no
  browser on this machine, so it behaves the same over SSH as it does locally.

The authorize URL sends `referrer=goshcoder`. Other clients of this public
client send another product's name; misreporting who is asking for a grant is
exactly what a consent screen exists to prevent.

`internal/llm/catalog/oauth_meta.go` — Meta's login is two exchanges, not one.
The device grant at `auth.meta.com/oidc/device/*` yields an identity token, and
only the mint that follows at `api.meta.ai/muse-code/key` yields the credential
`api.meta.ai` accepts. So the stored credential is laid out as pi's is: identity
in `refresh`, the minted key in `access`, the OAuth refresh token in an extra.

- `Refresh` means "mint a new key", which is what Meta's daily key rotation
  needs and costs one request. Only when Meta rejects the identity is the
  refresh token spent, so a renewal that fails is genuinely terminal.
- Expiry is the sooner of the key's re-mint deadline and the identity's death:
  either ending is a reason to refresh.
- Meta answers an unusable refresh token with a bare `404`, not `invalid_grant`.
  Read as a transport failure that would retry forever; it is mapped to
  `ErrOAuthUnauthorized`, which is what it means.
- `require_payment` on a mint is an account that has not finished signup, and
  carries where to finish. It is reported as such rather than as a dead
  credential -- logging in again fixes nothing.

Meta's Model API speaks `anthropic-messages` but authenticates with
`Authorization`. `AuthMetaBearer` therefore leaves `Auth.APIKey` **empty** and
carries the key only in the header, because the streamer sets `x-api-key`
whenever it has an api key. `TestAnthropicHeaderOnlyAuthSendsNoAPIKeyHeader` in
`internal/llm` is what keeps that true.

Meta's models are not in the pi reference, so they live in
`internal/llm/catalog/catalog_extra.json`, merged in `data.go`.
`catalog.json` is regenerated wholesale from pi and a hand-edit there would be
lost. A model pi *does* carry but describes wrongly is corrected in
`catalog_overrides.json` instead; see [Model catalog currency and effort
levels](#model-catalog-currency-and-effort-levels--done). The generated data wins on a collision and
`TestExtraCatalogIsNotShadowed` fails if pi ever ships the same model, so the
duplicate is deleted rather than quietly diverging.

The same file carries the Grok models pi's snapshot predates: `grok-4.6` (the
current flagship) and the three `grok-4.20-*` variants. Prices, context windows
and the 200K long-context tier come from `docs.x.ai/docs/models`. **Max output
tokens is the one figure xAI does not publish**, so each entry mirrors its
closest sibling -- `grok-4.6` follows `grok-4.5`, the 4.20 family follows
`grok-4.3`, which they match on context window, pricing and wire protocol. A
value that is too low truncates a reply where one that is too high is refused
by the API, so mirroring downward is the safer error.

Note that the *generated* `xai` entries still carry no `tiers`, so a
long-context turn on `grok-4.3` or `grok-4.5` is costed at the sub-200K rate.
That is pi's data, and correcting it in `catalog_extra.json` would not help:
the generated definition wins on a collision by design.

The client ids are public desktop clients with no secret -- PKCE and the device
flow replace one -- and `GOSHCODER_XAI_OAUTH_CLIENT_ID` /
`GOSHCODER_META_OAUTH_CLIENT_ID` override them.

Covered by `oauth_xai_test.go` and `oauth_meta_test.go` against fake issuer and
mint servers: both login shapes, endpoint pinning, discovery fallback,
non-rotating refresh tokens, the re-mint and renewal paths, terminal failures,
and both providers' request-auth derivation.

**Known caveat, xAI.** The subscription login authenticates a consumer Grok
plan, and xAI enforces its own entitlement checks on the inference API
afterwards: a login can succeed and inference still answer 403 for an account
without the plan the endpoint wants. That is xAI's decision, not a client bug.
`XAI_API_KEY` and `goshcoder auth set xai` remain the route for a developer
account and are unaffected.

## Model catalog currency and effort levels — done

The catalog was audited against each provider's own documentation on
2026-08-23. Two kinds of drift showed up, and they need different files.

**Models pi's snapshot predates** go in `catalog_extra.json`, the file that
already carried Meta and the newer Grok entries:

| Added | Source |
| --- | --- |
| `anthropic/claude-mythos-5` | Shares Claude Fable 5's specs and pricing; invitation-only under Project Glasswing, so the entry only resolves for an approved key |
| `amazon-bedrock/anthropic.claude-opus-5` | The plain inference profile the Anthropic docs list as the Bedrock ID; pi carries only the regional `us.`/`eu.`/`au.`/`jp.`/`global.` ones |
| `amazon-bedrock/us.xai.grok-4.6`, `global.xai.grok-4.6` | AWS's Grok 4.6 model card. In-Region inference is *not* offered on `bedrock-runtime`, so only the two cross-Region profiles are listed, at their separate rates ($2.20/$6.60/$0.55 Geo, $2.00/$6.00/$0.50 Global) |
| `google/gemini-3.7-flash`, `google-vertex/gemini-3.7-flash` | GA 2026-08-13, 1M context, 64K output, introductory $0.75/$3.75 |
| `zai/glm-5.3`, `zai-coding-cn/glm-5.3` | Released 2026-08-18 and on both coding-plan endpoints, so priced at zero like its siblings there |
| `openrouter/x-ai/grok-4.6`, `z-ai/glm-5.3`, `google/gemini-3.7-flash` | OpenRouter's own model pages |
| `meta/muse-spark-1.2-contributor` | The contributor tier on the Meta Model API, $0.10/$0.20 |

**Fields pi's snapshot has since got wrong** cannot go there: the generated
definition wins on a collision by design, so an entry for a model pi already
ships is ignored. `catalog_overrides.json` patches those in place:

| Overridden | Was | Now |
| --- | --- | --- |
| `openai/gpt-5.6-{sol,terra,luna}` context window | 272000 | 1050000 |
| `openai/gpt-5.6-sol` pricing | $5/$30 | $4/$0.40/$20, long-context tier $8/$0.80/$30 |
| `google/gemini-3.6-flash`, `google-vertex`, `openrouter` pricing | $1.50/$7.50/$0.15 | $0.75/$3.75/$0.075 |
| `deepseek/deepseek-v4-{pro,flash}` thinking levels | `low` unsupported | `low` maps to `low`, which DeepSeek documents as a direct value |

An override is a partial model object whose top-level keys replace the
generated ones **wholesale**: a `thinkingLevelMap` override replaces the whole
map rather than one level of it, because a null in that map means "unsupported"
and a key-wise merge could not express removing one. `id` and `provider` are
refused: they are what the merged catalog is indexed by. The pass runs *before*
`mergeExtraModels`, so an override can only ever reach pi's data -- a model in
`catalog_extra.json` is hand-written here and gets edited there instead.

Two tests keep the file honest, the same way `TestExtraCatalogIsNotShadowed`
keeps the extras honest. `TestOverridesTargetGeneratedModels` fails on an
override whose target pi does not carry, which would silently do nothing.
`TestOverridesStillCorrectPi` fails once an override restates what pi already
says, which is the prompt to delete it after a regeneration.

**Effort levels.** `xhigh` and `max` need an explicit `thinkingLevelMap` entry
to be offered at all (`GetSupportedThinkingLevels`), so a model that gains one
stays silently capped until its entry says so. The audit found the Anthropic and
OpenAI maps already correct -- `max` on Fable 5, Mythos 5, Opus 5/4.8/4.7/4.6
and Sonnet 5/4.6, `xhigh` on all of those but Opus 4.6 and Sonnet 4.6, both on
every GPT-5.6 -- and three that were not: Grok 4.6 gained `xhigh` (xAI treats it
as `high` on 4.5 and earlier, so 4.5 correctly still lacks it), DeepSeek V4 lost
`low`, and Gemini 3.7 Flash dropped `MINIMAL`, which 3.6 Flash still takes.
`TestEffortLevelsMatchProviderDocs` pins both halves: the levels each model must
offer, and the ones it must not, since clamping onto an unsupported level sends
a value the API rejects or silently demotes.

**Deliberately not added.**

- `gpt-5.6-cyber` and `gemini-3.5-flash-cyber`: exploit-development models behind
  applicant-vetted programs (OpenAI's Daybreak Red, Google's equivalent), with no
  public API spec to encode.
- `claude-mythos-preview`: named in Anthropic's effort docs but with no published
  specs of its own; Mythos 5's are documented as Fable 5's, the preview's are not.
- `meta/muse-glimmer-30b`: open weights, not served by the Meta Model API. pi
  already lists it on the aggregators that do host it.
- `openai-codex` context windows: still 272000 for the GPT-5.6 family, which
  matches what the Codex backend reports even after the long-context rollout.
  Raising it here would only make requests fail further along.
- Aggregator catalogs beyond OpenRouter's three flagships (`vercel-ai-gateway`,
  `cloudflare-ai-gateway`, `huggingface`, `azure-openai-responses`): pi
  regenerates those lists wholesale and they run to hundreds of entries.

**The app's own defaults.** `defaultChatModel` in `cmd/goshcoder/chat.go` names
a model per provider for the case where nothing is configured and nothing is
remembered, and that list had aged into `gpt-5.4` / `claude-sonnet-4-5` /
`gpt-5.1`. It now reads `gpt-5.6-sol` on Codex, `claude-sonnet-5`, and
`gpt-5.6-terra` on the OpenAI API: the flagship where a subscription is paying
and the balanced model where tokens are. `kimi-for-coding` stays as it is --
the name is the plan's own pointer, not a pinned generation. The README's
examples moved with it.

**Pricing with an expiry.** Two of the corrected rates are promotional and will
need revisiting rather than keeping: GPT-5.6 Sol's $4/$20 runs at least through
2026-11-21, and the Gemini 3.6/3.7 Flash introductory rate through 2026-12-31.
`gpt-5.6-sol`'s `cacheWrite` is the one figure not published at the new price;
it keeps pi's 1.25x-of-input ratio ($5, $10 on the long-context tier).

## Windows CI

`build & test (windows-latest)` was red on `main` from the repository's first
CI run until this change -- four failures, none of them Windows-specific
flakiness, all of them real differences the other two platforms hide.

- **`internal/plannotator`.** `IsPlanPathAllowed` gated on `filepath.IsAbs`,
  which is false for `/plans/a.md` on Windows: that path is relative to the
  current drive, so the gate joined it onto the workspace and accepted a path
  Unix rejected outright. Confinement still held -- the write landed inside the
  root -- so this was a write gate that meant two different things rather than
  an escape. `isRootedPath` now recognises every rooted spelling on every
  platform: leading slash or backslash, drive letter, and UNC.
  `TestPlanPathGateIsPlatformIndependent` fails on Linux without the fix, which
  is the point: the rule is now testable everywhere it applies.
- **`internal/integration` (two tests).** `Workspace` holds an open `os.Root`
  for race-free path confinement and has always had a `Close`; the `cmd` tests
  called it and these did not. Unix lets `t.TempDir` unlink a directory out
  from under an open handle, so the leak was invisible there. Windows refuses,
  and the cleanup failure failed the test.
- **`internal/omniroute`.** The test asserted the config file is written
  `0600`. Windows has no Unix permission bits -- `Chmod` there toggles only the
  read-only flag -- so it reports `0666`. The assertion is now made where it
  means something rather than dropped everywhere. **The 0600 guarantee is a
  Unix guarantee**; on Windows the file is protected by directory ACLs
  inherited from the user profile, not by mode bits.

**The compressed lock timings, again.** `3f2cfb8` widened the test stale
window from 200ms to a second so a starved heartbeat goroutine could not make a
live holder look dead. It landed that value in
`internal/llm/catalog/auth_lock_test.go` but not in `internal/lockfile`, where
only the prose and the sleep multiplier changed -- so `fast()` still returned
200ms while its own comment said a second, and `TestLiveHolderIsNotReclaimed`
went on failing, this time on the Linux `-race` job, which multiplies every
scheduling delay. The value now matches what that commit intended and what the
sibling package already has. Like the original, it is a reasoned widening
rather than a locally reproduced repair: ten runs at `GOMAXPROCS=1` under a
saturated four-core box reproduce neither the old failure nor a new one, so CI
remains the real test.

## Known deviations from pi (unchanged)

- Compat travels on options structs, with `Model.RawCompat` as the fallback.
- No vendor SDKs: hand-rolled `net/http` + SSE reader. Bedrock does SigV4 and
  binary event-stream framing against stdlib crypto.
- Interactive terminals use the native Bubble Tea fullscreen TUI; pipes and
  unsupported platforms fall back to the line-oriented interface.
- Selected pi extensions are native built-ins: `internal/ralph`
  (`@tmustier/pi-ralph-wiggum`), `internal/plannotator`
  (`@plannotator/pi-extension`), and `internal/claudetui`
  (`pi-claude-code-tui`). Planner uses a dependency-free stdlib browser
  reviewer; the visual UI is native Go rather than pi's npm/TUI runtime.
- Sessions are persisted in pi's **v3** JSONL format (`internal/sessionlog`),
  not the v4 mutation log under `packages/agent/src/harness/session`. pi's
  shipping CLI never imports v4 -- `CURRENT_SESSION_VERSION` is 3 -- and v3's
  reader is lenient where forward compatibility needs it: an unknown entry type
  joins the tree and projects to nothing, where v4 treats it as fatal. That is
  what makes every later format addition backward-compatible.
  Resume, branching, fork/clone, labels and JSONL/Markdown export/import are
  implemented. v1 and v2 pi files are migrated in memory and never rewritten,
  so continuing one forks it into a v3 file.
- TypeScript packages and plugins, custom `models.json`, LSP/MCP management, and
  HTML export/share remain outside current parity.
- Plan state lives in the session log as a `custom` entry rather than in a
  per-workspace file. `plannotator.Manager` owns no file: the host supplies
  `Options.Initial` and receives `Options.OnChange`.
- `/clear` appends a `transcript_reset` marker instead of rotating to a new
  file. That entry type is a GoshCoder addition; pi tolerates it but will still
  replay the cleared prefix, which is the one documented interop divergence.
