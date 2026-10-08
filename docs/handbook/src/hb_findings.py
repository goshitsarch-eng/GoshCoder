from hb_lib import *

S = src  # shorthand

# id, severity, title, feature, topic, platform, status, evidence, impact, repro, recommendation
FINDINGS = [
("F-01", "High", "Repository prompt templates can replace built-in slash commands, including <code>/share confirm</code>",
 "Prompt templates, all slash commands", "security sessions", "all", "runtime",
 f"Templates found in <code>.pi/prompts</code> or <code>.goshcoder/prompts</code> are never checked against built-in names ({S('src/resources.rs',540)}), and template expansion runs before built-in dispatch; expanded text that starts with <code>/</code> is then run as a command ({S('src/main.rs',2083)}; line mode {S('src/main.rs',1145)}). Reserved names are enforced only on <code>/prompt save</code> and restore.",
 "A repository you open can make a built-in command do something else. A template named <code>clear.md</code> whose body is <code>/share confirm</code> turns <code>/clear</code> into an unconfirmed upload of the transcript as a gist (with <code>gh</code> logged in); another body can send repo-chosen text to the model.",
 "Reproduced with a harmless body: <code>mkdir -p .goshcoder/prompts; printf '/status\\n' &gt; .goshcoder/prompts/clear.md</code>, start chat, <code>/reload</code>, type <code>/clear</code> → the <code>/status</code> report appears and nothing is cleared. Screenshot: <a href=\"#shot-hijack\">evidence</a>.",
 "Dispatch built-in commands before template expansion, or drop discovered templates whose names are reserved (with a warning), and never re-dispatch expanded template text as a slash command."),

("F-02", "High", "<code>chat -tools=false</code> is not read-only: the Planner can grant <code>bash</code>, <code>write</code> and <code>edit</code>",
 "-tools=false, Planner", "security tools extensions", "all", "runtime",
 f"The Planner is attached to every chat regardless of <code>enable_tools</code> ({S('src/runtime.rs',398)}). In the Executing phase its tool set is <code>normal_tools + workspace.all()</code>, which includes <code>bash</code> ({S('src/planner_runtime.rs',829)}); Planning adds write/edit for Markdown. The phase is read from the per-workspace state file, which wins over the session ({S('src/planner_runtime.rs',543)}). Ralph tools also stay registered ({S('src/runtime.rs',760)}).",
 "A user who starts <code>goshcoder chat -tools=false</code> expecting a read-only conversation can get a model with shell access, with no action of their own, if another window on the same repository left an approved plan executing.",
 "Set the workspace planner state to <code>executing</code> (as an approved plan does), then run <code>goshcoder -tools=false</code> and <code>/tools</code>: <code>bash</code>, <code>write</code> and <code>edit</code> are listed, and a model request for <code>bash</code> ran <code>ls -la</code>. Screenshot: <a href=\"#shot-readonly\">evidence</a>.",
 "When tools are disabled, leave the workspace tools out of the Planner's phase tool sets (or force the phase to Idle). Until then the README now says that <code>-tools=false</code> does not guarantee a read-only session."),

("F-03", "High", "Cloudflare AI Gateway and Workers AI requests go to a URL with literal <code>{CLOUDFLARE_ACCOUNT_ID}</code> placeholders",
 "cloudflare-ai-gateway (43 models), cloudflare-workers-ai (13 models)", "providers", "all", "source",
 f"Every Cloudflare model's <code>baseUrl</code> in <code>data/catalog.json</code> contains <code>{{CLOUDFLARE_ACCOUNT_ID}}</code> (and <code>{{CLOUDFLARE_GATEWAY_ID}}</code>). Auth resolution only stores the ids in the auth environment ({S('src/catalog.rs',2829)}); the request path parses <code>model.base_url</code> verbatim and nothing in <code>src/</code> substitutes the placeholders. pi does substitute them.",
 "All 56 Cloudflare models fail, yet <code>goshcoder providers</code> shows them as configured and the curated default-model list includes them, so a Cloudflare-only user gets a default model that fails on the first prompt.",
 "Not run (it needs a Cloudflare account). The absence of substitution was confirmed by reading the code and data.",
 "Replace both placeholders from the resolved auth environment in <code>Catalog::resolved_model</code> (or before the request), and add an endpoint test for both providers."),

("F-04", "Medium", "<code>cargo audit</code> fails: rustls 0.23.43 is affected by RUSTSEC-2026-0285",
 "Dependencies, CI lint job, release verify step", "build security", "all", "runtime",
 f"Running <code>cargo audit</code> (cargo-audit 0.22.2, 1,295 advisories, {META['audit_date']}) on the lockfile reported <code>rustls 0.23.43 — TLS 1.3 handshake messages incorrectly accepted across encryption level boundaries — RUSTSEC-2026-0285, severity 5.3 (medium), solution: upgrade to &gt;=0.23.45</code> and exited 1. {S('.cargo/audit.toml')} ignores only RUSTSEC-2023-0071.",
 "rustls is the TLS stack for every provider request. CI's <code>cargo audit</code> step, <code>make vuln</code> (when cargo-audit is installed) and the release workflow's verify step fail until the lockfile moves.",
 "<code>cargo install cargo-audit --locked &amp;&amp; cargo audit</code> in the repository.",
 "Run <code>cargo update -p rustls</code> to 0.23.45 or later, re-run the gate, and cut a patch release."),

("F-05", "Medium", "The planning-phase gate blocks only <code>bash</code>, <code>write</code> and <code>edit</code>",
 "Planner", "security extensions", "all", "source",
 f"<code>before_tool_call</code> returns no block for any other tool ({S('src/plannotator.rs',503)}); the planning tool set keeps every normal tool except <code>bash</code> ({S('src/planner_runtime.rs',824)}), including desktop control (<code>mcp</code>), Aperture connector calls, <code>image_gen</code> and <code>ralph_start</code>.",
 "During planning, which promises not to modify the codebase, the model can still type into a terminal window through desktop control, call gateway tools that change external systems, or start Ralph loops.",
 "Inspection only: no desktop-control server or Aperture gateway was available.",
 "Use an allow-list while planning (read, ls, grep, find, web_search, and write/edit on Markdown), or have each tool declare whether it has side effects."),

("F-06", "Medium", "A workspace <code>SYSTEM.md</code> silently replaces the whole system prompt",
 "Local resources", "security resources", "all", "runtime",
 f"The warning is only added to the resource report ({S('src/resources.rs',476)}); nothing prints it at startup. It is visible only through <code>/resources</code> or <code>/reload</code>.",
 "A cloned repository's <code>.pi/SYSTEM.md</code> or <code>.goshcoder/SYSTEM.md</code> replaces GoshCoder's instructions without the user noticing.",
 "Created <code>.goshcoder/SYSTEM.md</code> containing a pirate persona, started chat: no notice appeared; the request captured by the test gateway began with the pirate text; <code>/resources</code> then showed the warning.",
 "Show resource warnings, at least the SYSTEM.md replacement, as a startup notice."),

("F-07", "Medium", "Bare <code>/export</code> writes the full transcript into the project directory",
 "/export", "sessions security", "all", "runtime",
 f"With no path the destination is <code>&lt;cwd&gt;/goshcoder-session-&lt;id&gt;.html</code> ({S('src/main.rs',3972)}). Existing files are overwritten without asking.",
 "Tool output, thinking and anything the agent read (including secrets) lands where <code>git add -A</code> will pick it up.",
 "<code>/export</code> in the test workspace created <code>todo-app/goshcoder-session-01a11caa-4166.html</code>.",
 "Default to a path outside the workspace, or warn when writing inside a git work tree."),

("F-08", "Medium", "<code>goshcoder sessions rm</code> deletes immediately and can match sessions in other workspaces",
 "sessions rm", "sessions", "all", "runtime",
 f"No confirmation is asked ({S('src/sessions.rs',190)}). When no session in the current workspace matches a prefix, resolution retries across all workspaces ({S('src/sessionlog.rs',968)}).",
 "A short or mistyped prefix can delete a session from a different project with no undo.",
 "<code>goshcoder sessions rm 01a11cb2-73b4</code> printed <code>removed …</code> and deleted the file with no prompt. The cross-workspace fallback was read in source.",
 "Require <code>--yes</code> or a confirmation, and do not fall back to other workspaces for prefix matches in <code>rm</code>."),

("F-09", "Medium", "The <code>bash</code> tool runs <code>sh -c</code>, not bash",
 "bash tool", "tools", "linux macos", "source",
 f"{S('src/tools.rs',882)}: <code>Command::new(\"sh\")</code> with <code>-c</code>. On Debian and Ubuntu <code>sh</code> is dash.",
 "Bash syntax a model naturally writes (<code>[[ ]]</code>, arrays, <code>source</code>, brace expansion, <code>set -o pipefail</code>) fails or behaves differently while the tool tells the model it is bash.",
 "Inspection; on dash, <code>echo {a,b}</code> prints <code>{a,b}</code>.",
 "Run <code>bash -c</code> when bash is on PATH and fall back to <code>sh</code>, or rename and describe the tool accurately."),

("F-10", "Medium", "<code>bash</code> output keeps the first 30 KiB and drops the rest; the 120 s timeout cannot be changed",
 "bash tool", "tools", "all", "source",
 f"<code>CappedBytes::push</code> keeps a prefix ({S('src/tools.rs',1867)}); <code>DEFAULT_BASH_TIMEOUT</code> is 120 s ({S('src/tools.rs',71)}) and the setters have no production caller.",
 "For long builds and test runs the failures and summary at the end are cut off; anything over two minutes is killed with its process group.",
 "Inspection.",
 "Keep the tail (or head and tail) and say which part was dropped; add an optional bounded <code>timeout</code> parameter."),

("F-11", "Medium", "A failed automatic compaction blocks the prompt, and compaction cannot be cancelled",
 "Context compaction", "agent", "all", "source",
 f"<code>maybe_auto_compact(agent)…?</code> runs before the prompt is sent ({S('src/turns.rs',74)}); the summary request has no retry. The summary runs on a separate agent whose cancellation is not connected to <kbd>Esc</kbd> ({S('src/compaction.rs',424)}).",
 "Near the context limit a transient 429/5xx during summarisation makes every prompt fail until the provider recovers; a slow summary cannot be interrupted.",
 "Inspection. <code>/compact</code> itself was run successfully at runtime.",
 "Treat an automatic-compaction failure as a notice and send the prompt anyway; connect the main abort to the summariser."),

("F-12", "Medium", "BTW side threads follow the process, not the session",
 "BTW", "extensions sessions", "all", "source",
 f"The BTW runtime is built once in <code>prepare_session</code> ({S('src/runtime.rs',755)}) and restores threads only there. In-chat <code>/resume</code>, <code>/new</code> and <code>/clear</code> do not reset it, and the persist hook writes the in-memory snapshot into whichever session is open.",
 "After <code>/resume B</code>, B's saved threads do not appear and A's threads (with up to 40,000 characters of A's transcript each) are written into B's file. Threads come back only with <code>-continue</code> or <code>-resume</code> at startup.",
 "Inspection.",
 "Rebuild the BTW runtime from the session's custom entries whenever the session switches."),

("F-13", "Medium", "Every BTW change appends a full snapshot of all threads to the session file",
 "BTW", "extensions sessions performance", "all", "source",
 f"<code>persist</code> writes <code>snapshot_threads()</code> after each thread creation, answer and failure ({S('src/btw_runtime.rs',792)}); each snapshot is a new entry of up to 4 MiB.",
 "Session files grow roughly quadratically with BTW use, slowing <code>-continue</code>, export and import.",
 "Inspection.",
 "Record per-thread deltas, or replace earlier <code>goshcoder.btw</code> entries."),

("F-14", "Medium", "<code>-claude-tui</code> does nothing; <code>/use-default-tui</code> and <code>/use-claude-code-tui</code> do not exist",
 "Line-mode appearance", "interface docs", "all", "runtime",
 f"The flag is parsed into <code>SessionConfig::claude_tui</code> ({S('src/runtime.rs',1361)}) and never read. The two commands appear only in the reserved-name list ({S('src/main.rs',3898)}).",
 "Following the previous README produced <code>unknown command /use-default-tui</code>.",
 "Typed both commands in chat: both answered <code>× Error unknown command …</code>.",
 "Remove the flag and names, or implement the switch. The README no longer documents them."),

("F-15", "Medium", "Ralph loops are owned by a process id, so they are orphaned after a restart",
 "Ralph", "extensions", "all", "runtime",
 f"The Ralph store's session id is <code>cli-&lt;pid&gt;</code> ({S('src/runtime.rs',762)}, {S('src/ralph_cli.rs',10)}); <code>current()</code> only accepts a loop owned by that id.",
 "After restarting chat (even with <code>-continue</code>) an active loop is no longer “current”: <code>/ralph</code> says there is no active loop and <code>ralph_done</code> refuses, while <code>/ralph list</code> still shows it active. <code>goshcoder ralph status</code> never shows a loop.",
 "<code>goshcoder ralph start orphan \"demo task\"</code>, then <code>goshcoder ralph list</code> → <code>orphan: ▶ active</code>, <code>goshcoder ralph status</code> → <code>No active loop in this workspace.</code> The state file recorded <code>\"ownerSessionId\": \"cli-9372\"</code>.",
 "Own loops by session id (pid only with <code>-no-session</code>); have the CLI report all active loops."),

("F-16", "Medium", "A <code>.ralph</code> that is a symlink or a file blocks every prompt",
 "Ralph", "extensions security", "all", "runtime",
 f"The per-turn sync propagates the store error ({S('src/ralph_runtime.rs',120)}, {S('src/runtime.rs',239)}) although attach treats it as non-fatal.",
 "A repository can ship a <code>.ralph</code> symlink that makes chat refuse every prompt.",
 "<code>ln -s /elsewhere .ralph</code>; <code>printf 'hello\\n' | goshcoder chat …</code> → <code>error: synchronize Ralph loop: refusing Ralph path … because it is a symlink or escapes the workspace directory</code>. Workaround: <code>-ralph=false</code>.",
 "Treat a store error during sync like attach does: post a notice and keep chatting without Ralph."),

("F-17", "Medium", "A hand-set <code>toolCalling: false</code> in <code>omniroute.json</code> is lost on every sync",
 "OmniRoute", "extensions providers", "all", "runtime",
 f"<code>sync_command</code> and <code>setup</code> replace <code>config.models</code> with the fetched list ({S('src/omniroute.rs',1430)}, {S('src/omniroute.rs',1544)}).",
 "The documented way to opt a chat-only model into the prompt-emulated tool adapter silently reverts.",
 "Added <code>\"toolCalling\": false</code> to a model, ran <code>goshcoder omni sync</code>: the key count in the file went from 1 to 0.",
 "Merge stored per-model overrides into the fetched list, or keep overrides in a separate map."),

("F-18", "Medium", "Several catalog compat flags are never read (OpenRouter Claude prompt caching, session-affinity headers, Z.AI tool streaming)",
 "openai-completions requests", "providers", "all", "source",
 "The catalog sets <code>cacheControlFormat</code> (32 OpenRouter <code>anthropic/*</code> models), <code>sendSessionAffinityHeaders</code> (54), <code>sessionAffinityFormat</code> (23), <code>zaiToolStream</code> (10), <code>supportsStrictTools</code> (14) and <code>supportsExplicitPromptCacheMode</code> (3); none of these names appears in <code>src/</code>. pi applies them.",
 "Claude models through OpenRouter get no prompt caching, so every turn is billed at the full input rate while the cost display assumes cache-read prices; gateways lose cache affinity.",
 "Inspection plus a scan of <code>data/*.json</code>.",
 "Port the flags into the completions builder, or list them as known gaps; add a test that every compat key in the data is consumed."),

("F-19", "Medium", "Bedrock is shown as configured for credentials it cannot use",
 "amazon-bedrock", "providers", "all", "source",
 f"Detection accepts <code>AWS_PROFILE</code>, container credentials and <code>AWS_WEB_IDENTITY_TOKEN_FILE</code> ({S('src/catalog.rs',2884)}), but the request code only reads static keys from the environment or INI files and the Bedrock bearer token ({S('src/bedrock.rs',966)}).",
 "On EKS/ECS or with SSO/assume-role profiles, <code>goshcoder providers</code> shows <code>✓ amazon-bedrock</code> and every request fails with “no AWS credentials found”.",
 "Inspection.",
 "Narrow detection to supported sources, or implement the missing credential providers."),

("F-20", "Medium", "<code>auth.json.lock</code> does not interoperate with pi's lock",
 "Credential store shared with pi", "providers security", "all", "inferred",
 f"GoshCoder takes an OS lock on a regular file <code>auth.json.lock</code> and never deletes it ({S('src/catalog.rs',1598)}). pi's lock library creates a <em>directory</em> with that name (behaviour of <code>proper-lockfile</code>, not vendored here).",
 "Pointing both tools at one agent directory, which the README encourages, can make writes from either tool fail until the stale lock is removed.",
 "Not run (pi not installed). Inferred from the code and the library's documented behaviour.",
 "Use a GoshCoder-specific lock name and also honour pi's directory lock, or document that concurrent use is unsafe."),

("F-21", "Medium", "Line-mode chat and <code>run</code> print model and tool text without removing terminal escape sequences",
 "Line mode, run", "security interface", "all", "source",
 f"<code>render_run_event</code> writes deltas, errors and the first line of tool results raw ({S('src/main.rs',592)}); the fullscreen interface sanitises everything.",
 "A prompt-injected model or a hostile file shown by a tool can write escape sequences to your terminal (clipboard writes via OSC 52, title changes, cursor movement).",
 "Inspection.",
 "Sanitise stderr always and stdout when it is a terminal."),

("F-22", "Medium", "A panic on any worker thread restores the terminal while the interface keeps drawing",
 "Fullscreen interface", "interface reliability", "all", "inferred",
 f"The panic hook installed by <code>run_interactive</code> is process-wide and calls <code>restore_terminal_modes()</code> ({S('src/main.rs',799)}), although worker panics are otherwise contained.",
 "A recoverable panic in a tool or background thread leaves the screen garbled.",
 "Inferred from code; no panic was triggered.",
 "Restore only when the panicking thread is the interface thread."),

("F-23", "Medium", "Installers fall back to building the default branch on any download failure, ignoring <code>--version</code>",
 "install.sh, install.ps1", "install security", "linux macos windows", "source",
 f"Any failure of the release path leads to a source build that clones the default branch with no use of the requested version ({S('install.sh',274)}, {S('install.ps1',250)}).",
 "A typo in <code>--version</code>, a missing asset or a network hiccup replaces a pinned, checksum-verified binary with an unpinned, unverified build of <code>main</code>.",
 "Inspection.",
 "Fail when a version was pinned, or clone that tag; distinguish “no release” from “download failed”."),

("F-24", "Medium", "The release round-trip test never runs the installers",
 "CI", "build", "all", "source",
 f"{S('scripts/release-roundtrip.sh')} computes the archive name with <code>make dist-name</code> (the Makefile's own formula) and never invokes <code>install.sh</code>; <code>install.ps1</code> is only parse-checked in CI.",
 "A drift between the installers' archive naming or checksum parsing and the Makefile would not be caught before a release.",
 "Inspection.",
 "Add a base-URL override for tests and run the real installers against the local server."),

("F-25", "Medium", "<code>install.sh</code> keeps going after Ctrl-C or SIGTERM",
 "install.sh", "install reliability", "linux macos", "source",
 f"<code>trap cleanup EXIT INT TERM</code> ({S('install.sh',27)}) deletes the temporary directory and returns, so the script continues into the source-build fallback.",
 "Interrupting an install starts a source build in a deleted directory and exits 0.",
 "A stand-alone script with the same trap, run under dash and bash, continued after TERM and exited 0 (reproduced during the audit). The installer itself was not interrupted.",
 "<code>trap 'cleanup; exit 130' INT</code> and <code>trap 'cleanup; exit 143' TERM</code>."),

("F-26", "Medium", "<code>install.ps1</code> closes the PowerShell window on any error when run through <code>irm | iex</code>",
 "install.ps1", "install", "windows", "inferred",
 f"<code>Stop-WithError</code> calls <code>exit 1</code> ({S('install.ps1',46)}), which ends the host session under <code>iex</code>.",
 "Errors such as a checksum mismatch close the window before the message can be read.",
 "Not run (no PowerShell in the audit environment).",
 "<code>throw</code> instead of <code>exit</code>, or wrap the body in a scriptblock."),

("F-27", "Medium", "Windows: AltGr characters are dropped and Ctrl-C during a line-mode reply kills chat",
 "Editors on Windows", "interface", "windows", "inferred",
 f"Characters are inserted only without Ctrl/Alt ({S('src/state.rs',603)}, {S('src/line_editor.rs',170)}); crossterm reports AltGr as Ctrl+Alt on Windows. The line-mode SIGINT handler is Unix-only ({S('src/line_editor.rs',333)}).",
 "On German, French and similar layouts <code>@ { } [ ] \\ | ~</code> cannot be typed (pasting works). Ctrl-C during a line-mode reply terminates the process without closing the session.",
 "Not run (no Windows machine).",
 "Treat Ctrl+Alt+character as text when it is not a letter; install a console control handler on Windows."),

("F-28", "Medium", "Documentation overstated the filesystem confinement of tools",
 "Security documentation", "security docs tools", "all", "docs",
 f"The README said symlink and rename races “cannot escape” the workspace. The code says the opposite: without <code>openat</code>/<code>O_NOFOLLOW</code> the checks cannot be race-free against a concurrent process ({S('src/tools.rs',14)}); hard links are not detected.",
 "Users could rely on a guarantee that does not exist when running in a workspace another process can write to.",
 "Inspection.",
 "Fixed in this change: the README and this handbook now state the limit. A code fix would use <code>openat2(RESOLVE_BENEATH)</code> or <code>cap-std</code>."),

("F-29", "Medium", "After an in-interface <code>/login grok-cli</code> the <code>image_gen</code> tool stays missing until restart",
 "Grok Imagine", "extensions providers", "all", "source",
 f"Only the line-mode subprocess path calls <code>sync_image_tool()</code> after login ({S('src/main.rs',2762)}); <code>finish_login</code> does not ({S('src/main.rs',1977)}).",
 "The default fullscreen login does not offer image generation to the model; <code>/grok-cli-imagine:tool status</code> shows <code>persisted: on; active: off</code>.",
 "Inspection (no Grok account). The <code>status</code> output format was observed.",
 "Call <code>sync_image_tool()</code> in <code>finish_login</code> on success."),

("F-30", "Low", "The retry classifier matches status codes anywhere in the error text",
 "Automatic retries", "agent providers", "all", "source",
 f"Error text is reduced to lowercase alphanumerics and checked with <code>contains</code> for patterns such as <code>500</code>, <code>429</code> and <code>ratelimit</code> ({S('src/stream.rs',1905)}).",
 "Non-transient errors like “max_tokens must be ≤ 1500” or a date such as 2025-02 are retried three times, adding about 14 seconds.",
 "Inspection.", "Match on the parsed HTTP status, or use word boundaries before collapsing."),

("F-31", "Low", "Retry notices appear below the eventual answer, and attempt numbers differ between modes",
 "Retries display", "interface", "all", "runtime",
 "In fullscreen the retry notices are appended at the end of the transcript, below the reply that finally succeeded. <code>run</code> prints “attempt 1 of 3 in 2s” while the interface says “attempt 2 of 4” for the same retry.",
 "Confusing history when reading back a session.",
 "Prompted the test gateway with a scenario that fails three times with HTTP 500, in both <code>run</code> and fullscreen.",
 "Anchor retry notices to the turn they belong to; use one numbering."),

("F-32", "Low", "Ralph CLI and labels disagree with behaviour",
 "Ralph", "extensions docs", "all", "runtime",
 f"Help says <code>status [name]</code> but the parser rejects any argument ({S('src/ralph.rs',1096)}). <code>stop</code> marks a loop <em>completed</em> ({S('src/ralph.rs',727)}), so <code>resume</code> (“Continue a stopped loop” in the palette) always fails; there is no pause. The empty-list hint points to <code>goshcoder run -ralph …</code>.",
 "Users cannot pause and resume a loop; the documented status form errors.",
 "<code>goshcoder ralph status duedates</code> → <code>error: usage: ralph start|list|status|resume|stop|archive|delete</code>; after <code>/ralph stop</code> the state file said <code>\"status\": \"completed\"</code>.",
 "Accept a name for <code>status</code>, add <code>pause</code>, fix the labels."),

("F-33", "Low", "<code>-resume</code> is not searchable and <code>sessions show --full</code> does not recover cleared text",
 "Sessions", "sessions docs", "all", "runtime",
 f"The startup picker accepts only a number or Enter ({S('src/session_picker.rs',145)}). <code>--full</code> prints one-line previews of up to 120 characters ({S('src/sessions.rs',610)}).",
 "The previous README promised both. A cleared conversation is fully recoverable only from a JSONL export, or partly with <code>/tree</code> and <code>/fork</code>.",
 "Ran <code>goshcoder chat -resume</code> (numbered list) and <code>goshcoder sessions show &lt;id&gt; --full</code> (previews).",
 "Documentation fixed in this change; implement filtering or drop the unused text loading."),

("F-34", "Low", "<code>-session &lt;typo&gt;</code> silently creates a session; <code>-read-only</code> can be ignored",
 "Session flags", "sessions", "all", "source",
 f"An unmatched, non-path reference creates a new session with that literal id ({S('src/session.rs',1227)}). When <code>-continue</code> or <code>-session</code> ends up creating a new session, <code>-read-only</code> is ignored.",
 "A mistyped id starts an empty, recorded session without saying so.",
 "Inspection.", "Print a notice, and refuse <code>-read-only</code> when nothing existing was selected."),

("F-35", "Low", "A branch label cannot be removed from chat",
 "/label", "sessions", "all", "source",
 f"The handler requires a name after the point number ({S('src/main.rs',3263)}); the runtime's <code>clear_label</code> has no caller.",
 "Labels can be renamed but never cleared.", "Inspection.", "Accept <code>/label N</code> with no name as removal."),

("F-36", "Low", "<code>sessions show</code> and <code>sessions list</code> print transcript text without sanitising",
 "sessions CLI", "security sessions", "all", "source",
 f"Previews and titles are printed raw ({S('src/sessions.rs',610)}, {S('src/sessions.rs',109)}).",
 "A tool result containing escape sequences is replayed to the terminal.", "Inspection.", "Pass previews through <code>markdown::sanitize_terminal_text</code>."),

("F-37", "Low", "<code>sessions gc</code> always covers every workspace",
 "sessions gc", "sessions", "all", "source",
 f"<code>all_workspaces: true</code> regardless of the current directory ({S('src/sessions.rs',315)}); <code>--older-than 0d</code> selects everything.",
 "Users expecting per-project cleanup delete sessions everywhere (mitigated by the default dry run, which was verified).",
 "Dry run verified at runtime; the scope was read in source.", "Say “all workspaces” in help and output, or scope to the current directory unless <code>--all</code>."),

("F-38", "Low", "<code>install.sh</code> installs without verification when no SHA-256 tool exists",
 "install.sh", "install security", "linux macos", "source",
 f"It warns and returns success ({S('install.sh',160)}).",
 "On minimal systems integrity checking silently degrades.", "Inspection.", "Fall back to <code>openssl dgst -sha256</code> or stop."),

("F-39", "Low", "<kbd>Ctrl</kbd>+<kbd>D</kbd> quits an unsaved session at once and aborts a streaming reply",
 "Keyboard", "interface", "all", "source",
 f"{S('src/state.rs',418)}: Ctrl-D on an empty editor returns Quit regardless of whether the session is recorded, and Abort while streaming; only Ctrl-C asks twice.",
 "A <code>-no-session</code> conversation can be lost with one key.", "Inspection.", "Apply the same confirmation as Ctrl-C."),

("F-40", "Low", "Fullscreen ignores <code>NO_COLOR</code>; status-bar hints have 2.6:1 contrast",
 "Accessibility", "interface accessibility", "all", "source",
 f"Colours are fixed RGB constants and a dark background is painted over the terminal theme ({S('src/ui.rs',27)}); <code>NO_COLOR</code> is read only by line mode. Hints use <code>FAINT</code> (62,87,96) on (10,10,10), about 2.58:1.",
 "Users who need monochrome or high contrast must use line mode.", "Inspection; contrast computed from the constants.", "Honour <code>NO_COLOR</code>; use the 5.8:1 muted colour for hints."),

("F-41", "Low", "Mouse capture reports every pointer move and each triggers a full redraw",
 "Fullscreen performance", "interface performance", "all", "inferred",
 f"<code>EnableMouseCapture</code> turns on any-motion tracking; every event marks the frame dirty ({S('src/main.rs',1455)}). Also, normal click-drag text selection needs the terminal's bypass modifier (usually Shift).",
 "CPU use on long transcripts while the pointer moves.", "Inferred.", "Enable only button and wheel reporting, or ignore motion events."),

("F-42", "Low", "Anthropic login sends the PKCE verifier as <code>state</code>",
 "Anthropic OAuth", "security providers", "all", "source",
 f"<code>(\"state\", pkce.verifier())</code> ({S('src/oauth.rs',2628)}); the authorization URL is shown, copied to the clipboard through OSC 52 and passed to the browser launcher. The redirect says <code>localhost</code> while the listener binds 127.0.0.1 only.",
 "Anyone who sees the URL and intercepts the code can redeem it. This matches pi and may be required by the provider.",
 "The login was started and cancelled at runtime; the screenshots in this handbook redact the parameters.",
 "Use an independent random state if Anthropic accepts it; do not auto-copy that URL."),

("F-43", "Low", "In-interface <code>/login meta-muse</code> does not refresh the model list; the login picker can block on token refresh",
 "Logins", "providers interface", "all", "source",
 f"Only the CLI login refreshes the Muse model cache ({S('src/provider_cli.rs',236)}). Building the <code>/login</code> list calls <code>is_configured</code> for every provider, which may refresh OAuth tokens on the interface thread (15 s budget each).",
 "Stale model list until restart; a possible freeze when opening <code>/login</code> with expiring tokens.", "Inspection.", "Refresh the cache after login; compute “configured” without refreshing."),

("F-44", "Low", "Generated images are written with default permissions",
 "Grok Imagine", "security extensions", "all", "source",
 f"No mode is set when saving images, and without a session they go to a shared temporary directory ({S('src/grok_imagine.rs',601)}).",
 "Other local users may read them.", "Inspection.", "Use 0600 files in a per-user directory."),

("F-45", "Low", "Release workflow hardening",
 "Release workflow", "build security", "all", "source",
 f"<code>contents: write</code> applies to every job, including the one that compiles and runs tests; third-party actions are pinned to tags; <code>-rc</code> tags are published as normal releases ({S('.github/workflows/release.yml')}).",
 "A compromised dependency or action could use the write token; <code>releases/latest</code>, which the installers follow, could point at a release candidate.",
 "Inspection.", "Per-job permissions, <code>persist-credentials: false</code>, SHA pinning, and mark pre-release tags."),

("F-46", "Low", "The version was not bumped after the v0.6.0 tag",
 "Versioning", "build", "all", "runtime",
 "HEAD is <code>v0.6.0-27-g3870a05</code>; <code>Cargo.toml</code>, <code>Cargo.lock</code>, the Makefile and both installers still say 0.6.0 / <code>0.6.0-dev+…</code>.",
 "A plain <code>cargo build</code> of unreleased code reports <code>goshcoder 0.6.0</code>.",
 "<code>target/debug/goshcoder version</code> → <code>goshcoder 0.6.0</code>; <code>git describe</code> → <code>v0.6.0-27-g3870a05</code>.",
 "Bump to the next development version as <code>CONTINUE.md</code> instructs."),

("F-47", "Low", "An invalid <code>GOSHCODER_MODEL</code> prevents chat from starting",
 "Model selection", "agent", "all", "runtime",
 f"The environment value is used without validation ({S('src/runtime.rs',513)}), unlike the remembered model.",
 "A stale value in your shell profile stops GoshCoder from opening.",
 "<code>GOSHCODER_MODEL=nope/missing goshcoder chat</code> → <code>error: unknown model \"nope/missing\"</code>.", "Validate it like the remembered model and warn when ignoring it."),

("F-48", "Low", "Without HOME or USERPROFILE the agent directory is relative to the current directory",
 "Configuration", "config security", "all", "source",
 f"{S('src/config.rs',205)} falls back to <code>.goshcoder/agent</code>; <code>APPDATA</code> is never used.",
 "In stripped environments credentials and sessions can be written into the project.", "Inspection.", "Fail loudly or use the platform config directory."),

("F-49", "Low", "<code>mcp.json</code> is rewritten at session start and its mode changed to 0600",
 "Desktop control", "extensions config", "linux", "source",
 f"When <code>computer-use-linux</code> is found, every session start rewrites the entry if the recorded path differs, reformats the file and follows symlinks ({S('src/computeruse.rs',216)}).",
 "A shared or dotfiles-managed <code>mcp.json</code> is reformatted and made owner-only.", "Inspection (binary not installed).", "Keep the existing mode; allow opting out."),

("F-50", "Low", "Web search: abort can take 60 s; non-default searches go to Exa without configuration",
 "web_search", "tools security", "all", "source",
 f"Requests are blocking with a 60 s timeout and cancellation is checked between reads ({S('src/webaccess.rs',257)}). In <code>auto</code> mode, any search with a recency filter or a result count other than 5, and any OpenAI failure, goes to <code>https://mcp.exa.ai/mcp</code> unauthenticated ({S('src/webaccess.rs',404)}).",
 "Slow aborts; queries reach a third party even when you have an OpenAI login.", "Inspection (no network search was run).", "Make the request cancellable; document or configure the fallback."),

("F-51", "Low", "<code>edit</code> normalises mixed line endings; <code>grep</code> silently skips large files",
 "edit, grep tools", "tools", "all", "source",
 f"Line ending is detected from the first line only and applied to the whole file ({S('src/tools.rs',1566)}); files over 4 MiB or containing NUL are skipped without notice ({S('src/tools.rs',739)}).",
 "Noisy diffs in mixed-ending files; “No matches found” can be wrong.", "Inspection.", "Preserve per-line endings; report skipped files."),

("F-52", "Low", "<code>/omni setup &lt;url&gt;</code> ignores the URL",
 "OmniRoute", "extensions", "all", "source",
 f"Extra arguments are dropped ({S('src/omniroute.rs',1840)}) and setup always prompts with the stored URL as default.",
 "Pressing Enter at the prompt keeps the old gateway.", "Inspection; the interactive prompt was used at runtime.", "Accept <code>setup [url]</code> or reject extra arguments."),

("F-53", "Low", "Aperture rough edges",
 "Tailscale Aperture", "extensions providers", "all", "source",
 f"Onboarding never offers connectors, which are off by default, yet <code>pin</code> succeeds silently; every session start downloads <code>https://models.dev/api.json</code> when dedicated mode is on; connector overflow files accumulate in the system temp directory where the confined <code>read</code> tool cannot reach them; the MCP session is never re-initialised after a gateway restart; in proxy mode a stored OAuth token is still sent to the gateway ({S('src/catalog.rs',2552)}).",
 "Surprising setup steps, an external request on every launch, and connector failures after gateway restarts.",
 "Not run (no Aperture gateway).", "See the recommendations in the source comments; at minimum document them."),

("F-54", "Low", "Planner review page details",
 "Planner browser review", "extensions security", "all", "source",
 f"The page's CSRF token is served to any local client on 127.0.0.1 ({S('src/plannotator.rs',1765)}), so another user on a shared host could approve a plan; the “Select line” button has no handler; the change view compares lines by position; <code>/planner</code> can toggle while a reply streams.",
 "Mostly cosmetic; the shared-host case lets another local user switch the agent to full tool access.",
 "The page, the Host-header check (421 for a foreign Host) and the feedback round-trip were verified at runtime; the rest is inspection.",
 "Put a one-time secret in the opened URL; remove or wire up “Select line”; use a real diff."),

("F-55", "Low", "BTW details",
 "BTW", "extensions", "all", "source",
 f"<code>/btw settings remember</code> changes a setting no code path reads; in line mode <code>/btw bring</code> says the text is in the editor but discards it ({S('src/main.rs',1155)}); questions starting with <code>list</code>, <code>resume</code>, <code>bring</code> or <code>settings</code> are parsed as subcommands.",
 "Small surprises.", "Inspection.", "Remove or wire the setting; print the text in line mode; add a quoting rule."),

("F-56", "Low", "Small output defects",
 "CLI and status", "interface", "all", "runtime",
 "<code>/status</code> with no model prints <code>Model: /</code> (source, <code>src/main.rs:5383</code>). <code>goshcoder sessions show … | head</code> ends with <code>error: Broken pipe (os error 32)</code>. Exporting to a missing directory reports the raw OS error. Collapsed tool cards say “… 1 more lines”.",
 "Cosmetic.", "Observed at runtime except the first, which was read in source.", "Use the model label helper; exit quietly on EPIPE; create or name the missing directory."),

("F-57", "Low", "Prompt-cache retention is fixed to short; <code>PI_CACHE_RETENTION</code> affects Bedrock only",
 "Prompt caching", "providers", "all", "source",
 f"The agent always sends <code>CacheRetention::Short</code> ({S('src/agent.rs',1231)}).",
 "Long cache TTLs are unreachable.", "Inspection.", "Expose one setting for all protocols."),

("F-58", "Low", "API keys are stored verbatim but later interpreted",
 "auth set", "providers security", "all", "source",
 f"Stored keys are parsed as config values: <code>$NAME</code>/<code>${{NAME}}</code> read the environment and a leading <code>!</code> runs a command ({S('src/catalog.rs',1692)}); a failed command is cached as failure for the life of the process.",
 "A key that happens to contain <code>$</code> or start with <code>!</code> is misread; this pi-compatible feature is otherwise undocumented.", "Inspection.", "Escape pasted keys on <code>auth set</code>; document the syntax."),

("F-59", "Low", "Documentation said CI runs the same gate on three operating systems",
 "CI", "build docs", "all", "docs",
 f"rustfmt, Clippy and <code>cargo audit</code> run on Linux only; macOS and Windows run build and tests ({S('.github/workflows/ci.yml')}). <code>make test-hermetic</code> sets only <code>AWS_*</code> values.",
 "Overstated assurance.", "Inspection.", "Fixed in this change."),

("F-60", "Info", "Dead code and stale comments",
 "Codebase", "maintenance", "all", "source",
 "Examples: <code>Workspace::list_tool</code>, the bash timeout setters, <code>QueueMode::All</code>, the never-installed <code>PrepareNextTurn</code> hook, <code>WARN_SESSION_BYTES</code>, <code>SessionSelection::Fork</code>, <code>matches_session</code>, <code>PROVIDER_METADATA.methods</code>, the BTW palette title, placeholder text “Rust/Ratatui migration is initializing…” in <code>App::new</code>, and module headers that still describe integration as future work (<code>oauth.rs</code>, <code>aperture.rs</code>, <code>plannotator.rs</code>, <code>btw.rs</code>). The error “No authenticated model with a migrated provider protocol is available” uses migration jargon.",
 "Maintenance cost and misleading comments.", "Inspection.", "Remove or update."),

("F-61", "Info", "<code>bash</code> inherits every environment variable, including provider keys",
 "bash tool", "security tools", "all", "source",
 f"No environment filtering for the shell ({S('src/tools.rs',847)}).",
 "A prompt-injected model can read keys with <code>env</code>; recorded sessions keep the output.", "Inspection.", "Document it; optionally strip known secret variables."),

("F-62", "Info", "Requests identify as other clients",
 "web_search, Grok CLI, Meta Muse", "providers", "all", "source",
 "Exa requests carry <code>x-exa-source: pi-web-access</code>; Grok CLI sends the official client's identifier and a <code>grok-shell/&lt;ver&gt; (macos; aarch64)</code> user agent; Meta Muse sends the Muse launcher's user agent. This is how the upstream extensions work.",
 "Terms-of-service and compatibility risk for users, not a code defect.", "Inspection.", "Keep documented; let users opt out where possible."),
]

DOC_FIXES = [
 ("README: <code>goshcoder run … \"explain this repo\"</code>", "<code>run</code> has no tools unless <code>-tools</code> is given; the example now passes <code>-tools</code>.", "runtime"),
 ("README: <code>-tools=false</code> gives “read-only chat”", "Not guaranteed while the Planner is planning or executing (F-02).", "runtime"),
 ("README: filesystem tools cannot be escaped by races", "The checks are not race-free (F-28).", "source"),
 ("README: credentials coordinate through a “heartbeat lock file”", "auth.json uses an OS advisory lock with a 60 s wait and no heartbeat; the heartbeat lock belongs to session files.", "source"),
 ("README: pi-claude-code-tui startup card, rounded half-open prompt, sidebar with messages and branch, <code>/use-default-tui</code>, <code>/use-claude-code-tui</code>, <code>-claude-tui</code>", "None of these exist (F-14); the sidebar shows title, model, thinking, storage, context, activity, plan and workspace.", "runtime"),
 ("README: <code>/hotkeys</code> shows the same list as the README", "It omits the mouse wheel and Ctrl+Shift+P; several working keys were undocumented.", "runtime"),
 ("README: <code>chat -resume</code> is a searchable list; <code>sessions show --full</code> recovers a clear", "Numbered list; previews only (F-33).", "runtime"),
 ("README: transcripts are “never inside the workspace”", "Bare <code>/export</code> writes into the workspace (F-07).", "runtime"),
 ("README: a workspace SYSTEM.md is “reported rather than applied silently”", "Applied with no startup notice (F-06).", "runtime"),
 ("README: installers accept <code>--dir --version --from-source --no-modify-path</code>; fall back only when no release exists; refuse anything that does not match", "PowerShell uses <code>-InstallDir -Version -FromSource -NoModifyPath</code>; fallback happens on any failure (F-23); install.sh skips verification without a SHA tool (F-38).", "source"),
 ("README: “flushed at each turn boundary”", "fsync happens at the end of each agent run (a prompt with all its tool turns) and on name/label/fork/compaction/close; a partially streamed reply is not on disk.", "source"),
 ("README: Bedrock signing uses “stdlib crypto”", "It uses the <code>hmac</code> and <code>sha2</code> crates.", "source"),
 ("README: “the same gate on Linux, macOS, and Windows”; round-trip “drives the installer's actual download path”; <code>make test-hermetic</code> exports wrong provider credentials", "Lint and audit run on Linux only; the round-trip never runs an installer; only <code>AWS_*</code> is set (F-24, F-59).", "source"),
 ("README: BTW threads come back with <code>/resume</code>; Meta Muse live list replaces the bundled one after login; Grok accounts login takes over the terminal “as /login does”", "Only at startup (F-12); only after the CLI login (F-43); fullscreen <code>/login</code> runs inside the interface.", "source"),
 ("README: layout table", "Omitted <code>mistral.rs</code>, <code>google_auth.rs</code>, <code>omni_prompt_tools.rs</code>, <code>provider_cli.rs</code>, <code>tui_login.rs</code>, <code>line_editor.rs</code>, <code>export_html.rs</code> and others.", "source"),
 ("README: session flags table", "Omitted <code>-quiet</code>; <code>--flag</code> and <code>-flag=value</code> forms are accepted.", "runtime"),
 ("CONTINUE.md: Anthropic has a device-code login", "Anthropic offers browser and copy-code; Codex has device code.", "source"),
 ("CONTINUE.md: the agent keeps its session id across <code>/resume</code>", "<code>/resume</code>, <code>/clone</code> and <code>/import</code> set the agent's session id to the new session.", "source"),
 ("CONTINUE.md: <code>make check</code> is the gate CI runs on three OSes", "CI never runs <code>make check</code>; the release workflow does, on Linux.", "source"),
]


def finding_html(f):
    fid, sev, title, feature, topic, platform, status, ev, impact, repro, rec = f
    return f"""<details class="unit finding" id="{fid}" data-audience="developer maintainer" data-topic="findings {topic}" data-platform="{'all' if platform=='all' else platform}" data-status="{status}" data-severity="{sev.lower()}">
<summary>{sev_badge(sev)}<span class="fid">{fid}</span><span class="ftitle">{title}</span>{badge(status)}</summary>
<div class="fbody">
<dl>
<dt>Affected</dt><dd>{feature}</dd>
<dt>Platform</dt><dd>{'All' if platform=='all' else platform.replace(' ', ', ').title().replace('Macos','macOS')}</dd>
<dt>Evidence</dt><dd>{ev}</dd>
<dt>Impact</dt><dd>{impact}</dd>
<dt>Reproduction</dt><dd>{repro}</dd>
<dt>Recommendation</dt><dd>{rec}</dd>
</dl>
<p><a href="#{fid}">Permalink</a></p>
</div>
</details>"""


def findings_chapter():
    counts = {}
    for f in FINDINGS:
        counts[f[1]] = counts.get(f[1], 0) + 1
    bar = "".join(f'<div class="card"><div class="n">{counts.get(s,0)}</div>{sev_badge(s)}</div>' for s in ["Critical", "High", "Medium", "Low", "Info"])
    vcounts = {}
    for f in FINDINGS:
        vcounts[f[6]] = vcounts.get(f[6], 0) + 1
    vline = ", ".join(f"{vcounts[k]} {STATUS_LABEL[k].lower()}" for k in ["runtime", "source", "inferred", "docs"] if k in vcounts)
    intro = f"""<p class="lead">Defects found by the audit, kept separate from the instructions. Nothing here was fixed in code; documentation findings were fixed in this change.</p>
<div class="sevbar">{bar}</div>
<p>{len(FINDINGS)} findings: {vline}. Severity reflects impact on users of a typical installation: <strong>High</strong> means a security boundary or a whole feature does not hold; <strong>Medium</strong> means data loss, a safety gap or a broken documented workflow; <strong>Low</strong> is a contained defect; <strong>Info</strong> is worth knowing. Use the Severity filter above to narrow the list; each finding has a permanent link.</p>"""
    items = "\n".join(finding_html(f) for f in FINDINGS)
    rows = [[a, b, badge(c)] for a, b, c in DOC_FIXES]
    docfix = unit("doc-corrections", "Documentation corrections made in this change", f"""
<p>Claims in <code>README.md</code> and <code>CONTINUE.md</code> that did not match the code. Each has been corrected.</p>
{table(["Previous claim", "What the code does", "Checked by"], rows, "wrapcode")}
""", audience="developer maintainer user", topic="findings docs", status="docs")
    hijack = f"""<section class="unit" id="finding-evidence" data-audience="developer maintainer" data-topic="findings security" data-platform="all" data-status="runtime">
<h3><a class="anchor" href="#finding-evidence">#</a>Screenshot evidence {badge('runtime')}</h3>
<div id="shot-hijack">{fig("finding-template-hijacks-clear", "F-01: with <code>.goshcoder/prompts/clear.md</code> containing <code>/status</code>, typing <code>/clear</code> printed the status report instead of clearing the conversation.", "Transcript showing the command /clear followed by a Session status notice instead of a cleared screen.")}</div>
<div id="shot-readonly">{fig("finding-readonly-chat-runs-bash", "F-02: a chat started with <code>-tools=false</code> in a workspace whose planner phase was executing lists bash, write and edit in <code>/tools</code>, and a requested <code>bash</code> call ran.", "Tool list including read, write, edit, ls, grep, find, bash and planner_submit_plan, followed by a successful bash tool card showing ls -la output.")}</div>
</section>"""
    return chapter("findings", "Audit findings", intro, items + "\n" + hijack + "\n" + docfix)
