from hb_lib import *


def arrow_defs(px):
    return f"""<defs>
<marker id="{px}ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker>
<marker id="{px}aha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head acc" d="M0,0 L10,5 L0,10 z"/></marker>
</defs>"""


def box(x, y, w, h, title, lines, cls="box"):
    t = [f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>',
         f'<text class="title" x="{x+10}" y="{y+20}">{esc(title)}</text>']
    for i, ln in enumerate(lines):
        t.append(f'<text class="small mono" x="{x+10}" y="{y+38+i*15}">{esc(ln)}</text>')
    return "\n".join(t)


PX = ["d1"]


def line(x1, y1, x2, y2, acc=False):
    return f'<line class="arrow{" acc" if acc else ""}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#{PX[0]}{"aha" if acc else "ah"})"/>'


def component_diagram():
    p = ['<svg class="diagram" viewBox="0 0 960 560" role="img" aria-labelledby="dg1t dg1d">',
         '<title id="dg1t">GoshCoder component diagram</title>',
         '<desc id="dg1d">Three frontends (fullscreen interface, line mode, run) use the runtime, which assembles the agent, tools, session recorder and extensions. The agent calls the responder, which resolves credentials through the catalog and talks to provider APIs over HTTPS. The session recorder writes JSONL files; the catalog reads auth.json and the bundled model data.</desc>',
         arrow_defs('d1')]
    PX[0] = 'd1'
    p.append('<rect class="lane" x="10" y="10" width="940" height="92" rx="10"/>')
    p.append('<text class="small" x="22" y="28">Frontends</text>')
    p.append(box(30, 36, 280, 58, "Fullscreen interface", ["main.rs event_loop · state.rs · ui.rs", "tui_login.rs · markdown.rs"], "box accent"))
    p.append(box(340, 36, 270, 58, "Line mode", ["run_line_interactive · line_editor.rs"]))
    p.append(box(640, 36, 290, 58, "CLI subcommands", ["run providers models auth sessions", "prompts ralph omni aperture grok-cli"]))
    p.append(box(260, 140, 440, 66, "Session runtime (runtime.rs PreparedSession)", ["parse flags · choose model · build system prompt", "register tools and extensions · background refreshes"], "box accent"))
    for x in (170, 475, 785):
        p.append(line(x, 94, 480 if x == 475 else (330 if x == 170 else 630), 140))
    p.append(box(30, 250, 270, 96, "Agent loop", ["agent.rs  run_loop · tool exec", "turns.rs  retries · overflow", "compaction.rs  summaries", "llm.rs / stream.rs  types"], "box warm"))
    p.append(box(330, 250, 300, 96, "Tools", ["tools.rs  read write edit ls grep", "          find bash (workspace-confined)", "webaccess.rs  web_search", "computeruse.rs  mcp · grok_imagine image_gen"]))
    p.append(box(660, 250, 270, 96, "Extensions", ["plannotator / planner_runtime", "ralph* · btw* · omniroute", "aperture* · grok_cli/accounts", "meta_muse"]))
    p.append(line(420, 206, 170, 250)); p.append(line(480, 206, 480, 250)); p.append(line(560, 206, 790, 250))
    p.append(line(300, 298, 330, 298, True))
    p.append(box(30, 392, 300, 82, "Responder (providers.rs)", ["9 protocols over blocking HTTPS + SSE", "bedrock.rs SigV4 · mistral.rs", "google_auth.rs · omni_prompt_tools.rs"], "box warm"))
    p.append(box(360, 392, 270, 82, "Catalog & credentials", ["catalog.rs  45 providers, auth order", "oauth.rs  8 login flows", "data/catalog*.json (embedded)"]))
    p.append(box(660, 392, 270, 82, "Session recorder", ["session.rs  lifecycle · tree", "sessionlog.rs  v3 JSONL · locks", "sessions.rs · export_html.rs"]))
    p.append(line(165, 346, 165, 392, True)); p.append(line(330, 432, 360, 432))
    p.append(line(250, 346, 700, 392))
    p.append('<rect class="lane" x="10" y="500" width="940" height="50" rx="10"/>')
    p.append('<text x="40" y="530">Provider APIs (HTTPS)</text>')
    p.append('<text x="370" y="530">~/.goshcoder/agent/auth.json</text>')
    p.append('<text x="670" y="530">~/.goshcoder/agent/sessions/*.jsonl</text>')
    p.append(line(165, 474, 165, 505, True)); p.append(line(495, 474, 495, 505)); p.append(line(795, 474, 795, 505))
    p.append("</svg>")
    return "\n".join(p)


def sequence_diagram():
    cols = [("You / UI thread", 70), ("Turn thread", 240), ("Agent", 400), ("Responder", 560), ("Provider", 710), ("Recorder", 870)]
    p = ['<svg class="diagram" viewBox="0 0 960 520" role="img" aria-labelledby="dg2t dg2d">',
         '<title id="dg2t">Data flow of one prompt</title>',
         '<desc id="dg2d">Pressing Enter starts a turn thread, which may compact context, then asks the agent to prompt. The agent sends the context to the responder, which resolves credentials and streams the reply from the provider. Events flow back to the interface through a channel; tool calls are executed and their results sent in another request; every finished message is appended to the session file; failed turns are retried.</desc>',
         arrow_defs('d2')]
    PX[0] = 'd2'
    for name, x in cols:
        p.append(f'<rect class="box" x="{x-62}" y="12" width="124" height="30" rx="6"/><text x="{x}" y="32" text-anchor="middle" class="title">{esc(name)}</text>')
        p.append(f'<line x1="{x}" y1="42" x2="{x}" y2="505" class="arrow" stroke-dasharray="3 4"/>')
    steps = [
        (70, 240, 70, "Enter → submit_interactive_input → begin_interactive_turn"),
        (240, 400, 100, "turns::run_prompt: maybe_auto_compact, then Agent::prompt"),
        (400, 560, 130, "request_assistant(context, tools)"),
        (560, 560, 158, "catalog: resolve model + auth (refresh OAuth)"),
        (560, 710, 186, "POST …/chat/completions · /responses · /v1/messages"),
        (710, 560, 214, "SSE events (text, thinking, tool calls)"),
        (560, 400, 242, "MessageEmitter → MessageUpdate events"),
        (400, 70, 270, "listener → mpsc channel → redraw"),
        (400, 870, 298, "MessageEnd → append JSONL entry"),
        (400, 400, 326, "tool calls → execute_tool_calls (threads)"),
        (400, 560, 354, "next request with tool results …"),
        (400, 240, 382, "AgentEnd (fsync session)"),
        (240, 240, 410, "finish_run: retry 2/4/8 s if retryable; compact"),
        (240, 70, 438, "TurnCompletion → status Ready"),
    ]
    for x1, x2, y, label in steps:
        if x1 == x2:
            p.append(f'<path class="arrow" d="M{x1},{y-8} h28 v14 h-26" marker-end="url(#d2ah)"/>')
            p.append(f'<text class="small halo" x="{x1+34}" y="{y+3}">{esc(label)}</text>')
        else:
            acc = x2 > x1
            p.append(line(x1, y, x2, y, acc))
            tx = min(x1, x2) + 6
            p.append(f'<text class="small halo" x="{tx}" y="{y-5}">{esc(label)}</text>')
    p.append("</svg>")
    return "\n".join(p)


def architecture():
    u = []
    u.append(unit("arch-overview", "Components", f"""
<p>GoshCoder is one binary crate (<code>goshcoder</code>, edition 2024) of about 104,000 lines in 49 modules under <code>src/</code>. It has no async runtime: work runs on OS threads with blocking <code>reqwest</code> HTTP, and the interface talks to them through channels. Model data is compiled in from <code>data/*.json</code>.</p>
{component_diagram()}
<p>Entry point: {src('src/main.rs',401,'main')} → {src('src/main.rs',416,'run')}, which dispatches subcommands; anything starting with <code>-</code>, or nothing, starts chat through <code>run_interactive</code>.</p>
""", audience="developer", topic="architecture", status="source"))

    mods = [
        ("main.rs", "CLI dispatch, fullscreen event loop, slash commands, line-mode chat, signal handling"),
        ("state.rs, ui.rs, markdown.rs", "Editor/palette state machine; Ratatui rendering; Markdown and terminal sanitising"),
        ("tui_login.rs, line_editor.rs", "<code>/login</code> inside the interface; raw-mode line editor"),
        ("runtime.rs, config.rs", "Flag parsing, model selection, session assembly; agent-directory paths"),
        ("agent.rs, turns.rs, compaction.rs", "Agent state machine and tool execution; retries and overflow; summaries"),
        ("llm.rs, stream.rs", "pi-compatible message types; streaming, SSE parsing, retry classification, thinking levels"),
        ("providers.rs, bedrock.rs, mistral.rs, google_auth.rs, omni_prompt_tools.rs", "Wire protocols and request builders"),
        ("catalog.rs, oauth.rs, provider_cli.rs", "Providers, models, credential store and resolution; OAuth flows; <code>providers/models/auth</code> commands"),
        ("tools.rs, webaccess.rs, computeruse.rs", "Workspace tools; web search; desktop-control MCP client"),
        ("session.rs, sessionlog.rs, sessions.rs, session_picker.rs, export_html.rs", "Session lifecycle, v3 JSONL store and locks, CLI, picker, HTML export"),
        ("resources.rs, prompts.rs", "Context files, SYSTEM.md, templates, skills; prompt archives"),
        ("plannotator.rs, planner_runtime.rs", "Planner state machine, browser review server, tool gating"),
        ("ralph.rs, ralph_runtime.rs, ralph_cli.rs", "Ralph loop store, prompts, tools, CLI"),
        ("btw.rs, btw_runtime.rs", "Side threads"),
        ("omniroute.rs, omni_cli.rs", "OmniRoute config, sync, commands"),
        ("aperture.rs, aperture_cli.rs, aperture_mcp.rs, aperture_tools.rs", "Aperture config and routing, wizard, MCP client, connector tools"),
        ("grok_cli.rs, grok_accounts.rs, grok_imagine.rs, meta_muse.rs", "Subscription-provider extensions"),
    ]
    u.append(unit("modules", "Module map", f"""
{table(["Files (src/)", "Responsibility"], [[", ".join(src('src/'+f.strip(), label=f.strip()) for f in a.split(',')), b] for a, b in mods], "wrapcode")}
<p>Ported files begin with a comment naming the pi source they were written against. The integration test {src('tests/aperture_core.rs')} recompiles <code>llm.rs</code> and <code>aperture.rs</code> as a separate crate.</p>
""", audience="developer", topic="architecture", status="source"))

    u.append(unit("dataflow", "What happens when you press Enter", f"""
{sequence_diagram()}
<ol class="steps">
<li>The key handler returns <code>Action::Submit</code>; {src('src/main.rs',2067,'submit_interactive_input')} decides between a new prompt, steering (a reply is running) and a follow-up (<kbd>Alt</kbd>+<kbd>Enter</kbd>). Templates and skills are expanded first.</li>
<li>{src('src/main.rs',2222,'begin_interactive_turn')} spawns a thread that calls {src('src/turns.rs',58,'turns::run_prompt')}: automatic compaction if near the limit, then {src('src/agent.rs',968,'Agent::prompt')}.</li>
<li>{src('src/agent.rs',1082,'run_loop')} records the message, polls steering, and calls the responder with the system prompt, messages and tools (cache retention short).</li>
<li>{src('src/providers.rs',1043,'catalog_assistant_responder')} resolves the model and credentials on every turn (refreshing OAuth under the auth lock), applies gateway rewrites, builds the protocol request and streams it. A connect timeout of 30 s and an idle-read timeout of 300 s apply; there is no whole-request deadline.</li>
<li>Stream events become <code>MessageUpdate</code> events. The interface's listener pushes them into a channel; the event loop redraws when something changed or every 100 ms while busy.</li>
<li>Tool calls are validated and run in parallel threads (sequential for Ralph, Planner, desktop and connector tools). Results go back in the next request until a reply has no tool calls. Tool calls in a reply cut off by the length limit are never run.</li>
<li>The session recorder appends each finished message to the JSONL file and syncs it at the end of the run.</li>
<li>{src('src/turns.rs',80,'finish_run')} retries retryable errors (2, 4, 8 s), compacts and retries once on context overflow, and runs queued follow-ups.</li>
</ol>
""", audience="developer", topic="architecture agent", status="runtime"))

    u.append(unit("state-concurrency", "State, concurrency and cancellation", f"""
<ul>
<li><strong>Agent state</strong> is one mutex-protected struct; listeners are never called with it held, so they can call back in. <code>Agent::state()</code> clones the whole transcript and is called on every redraw.</li>
<li><strong>Threads:</strong> one per prompt, one per tool call in a batch (unbounded), two pipe readers per process, one-shot background refreshes at session start (OmniRoute health, Aperture sync, Meta Muse models), a session-lock heartbeat, and a termination watchdog.</li>
<li><strong>Cancellation</strong> is a shared flag per run. <code>bash</code> polls it every 10 ms and kills its process group; HTTP reads poll every 25 ms; the retry wait wakes on abort. Compaction and in-flight web searches cannot be cancelled.</li>
<li><strong>Locks on disk:</strong> <code>auth.json.lock</code> (OS advisory lock, 60 s wait); session <code>.lock</code> files (2 s heartbeat, 20 s stale); Grok account vault lock (30 s stale).</li>
<li><strong>Signals:</strong> SIGTERM/SIGHUP abort the turn, close the session and exit with 128+n; a watchdog forces exit after 5 s (TERM) or 0.5 s (HUP). Verified: SIGTERM during a streaming reply exited 143 and saved the interrupted turn.</li>
</ul>
""", audience="developer", topic="architecture agent", status="runtime"))

    u.append(unit("data-formats", "Data formats", f"""
<h4>Session file (pi v3 JSONL)</h4>
{code('''{"type":"session","version":3,"id":"<uuidv7>","timestamp":"2026-10-08T17:58:02.342Z","cwd":"/path/to/project"}
{"type":"model_change","id":"c05aae73","parentId":null,"timestamp":"…","provider":"omni","modelId":"mock-coder"}
{"type":"message","id":"…","parentId":"c05aae73","timestamp":"…","message":{"role":"user","content":[…]}}''', "json")}
<p>Entry types: <code>message</code>, <code>model_change</code>, <code>thinking_level_change</code>, <code>compaction</code>, <code>branch_summary</code>, <code>label</code>, <code>session_info</code>, <code>transcript_reset</code> (GoshCoder's <code>/clear</code> marker), <code>custom</code> (<code>goshcoder.btw</code>, <code>goshcoder.planner</code>, <code>grok-cli-conv-id-v1</code>, <code>grok-cli-active-account-v1</code>, <code>grok-cli-imagine</code>) and <code>custom_message</code> (read only). Entries form a tree through <code>parentId</code>. pi's v1 and v2 files are migrated in memory and forked into a new v3 file when continued; the original is never rewritten. Limits: 16 MiB per entry, 256 MiB per file.</p>
<h4>auth.json</h4>
{code('''{
  "openai":    { "type": "api_key", "key": "…" },
  "anthropic": { "type": "oauth", "access": "…", "refresh": "…", "expires": 1791482125405 }
}''', "json")}
<p>pi-compatible. Unknown entry types are preserved. Written atomically (temporary file, fsync, rename) with mode 0600.</p>
<h4>Model catalog</h4>
<p><code>data/catalog.json</code> is generated from pi's model list and replaced wholesale; <code>data/catalog_extra.json</code> adds models pi lacks and <code>data/catalog_overrides.json</code> corrects fields. Tests fail when the generated data catches up with an extra or an override.</p>
""", audience="developer", topic="architecture sessions config", status="runtime"))

    u.append(unit("security-model", "Security and privacy boundaries", f"""
<ul>
<li><strong>Trust boundary:</strong> the workspace is untrusted input. Its <code>AGENTS.md</code> is quoted as context; its <code>SYSTEM.md</code> replaces the system prompt; its prompt templates become commands (<a href="#F-01">F-01</a>, <a href="#F-06">F-06</a>).</li>
<li><strong>File tools</strong> refuse paths and symlinks leading outside the workspace, best-effort (<a href="#F-28">F-28</a>). <code>bash</code> runs with your privileges and environment.</li>
<li><strong>Network:</strong> requests go to the configured provider, plus: OAuth endpoints during login; Exa/OpenAI/Kagi for web search; <code>x.ai/cli/stable</code> for the Grok CLI version; <code>models.dev</code> for Aperture; GitHub through <code>gh</code> for <code>/share</code>. Nothing is sent elsewhere; there is no telemetry.</li>
<li><strong>Local servers</strong> (OAuth callbacks, Planner review) bind to 127.0.0.1, check the Host header, use no-store and CSP headers, and validate state or a CSRF token (OpenRouter's callback uses a random path instead). Any local user can reach them (<a href="#F-54">F-54</a>).</li>
<li><strong>Secrets</strong> are stored 0600 and never echoed; credential types have no debug output. Session transcripts are not redacted.</li>
<li><strong>Exports</strong> are script-free HTML with <code>default-src 'none'</code>; verified: 0 <code>&lt;script&gt;</code> tags and that CSP in an exported file.</li>
</ul>
""", audience="developer maintainer", topic="architecture security", status="runtime"))

    return chapter("architecture", "Architecture", '<p class="lead">How the program is put together, for people reading or changing the code.</p>', "\n".join(u))


def development():
    u = []
    u.append(unit("dev-setup", "Working on GoshCoder", f"""
{code('''make check          # rustfmt check, cargo check, Clippy -D warnings, tests, hermetic tests, cargo audit
make test           # cargo test --workspace --all-targets --locked
make lint           # Clippy with -D warnings
make tools          # cargo-audit, cargo-llvm-cov, cargo-zigbuild
make cover          # coverage when cargo-llvm-cov is installed''')}
<p>Verified on {META['audit_date']}: <code>make check</code> passed (689 unit tests and 25 integration tests, twice), with <code>vuln</code> skipped because cargo-audit was not installed at the time; installing it and running <code>cargo audit</code> then failed on RUSTSEC-2026-0285 (<a href="#F-04">F-04</a>). <code>make test-hermetic</code> re-runs the suite with dummy <code>AWS_*</code> values so a test that reads the developer's real AWS environment fails.</p>
<p>Ports are written against a local clone of pi at <code>reference/pi</code> (gitignored):</p>
{code("git clone https://github.com/earendil-works/pi reference/pi")}
<p>Conventions (from <code>CONTINUE.md</code>): name the pi source file at the top of a ported file; comments explain why; tests assert observable behaviour with injected environments, temporary agent directories and loopback fake servers; no new crate without a reason that survives <code>cargo audit</code> and the licence collection.</p>
""", audience="developer", topic="development", status="runtime"))

    u.append(unit("dev-manual-testing", "Testing the interface without a provider account", f"""
<p>The screenshots in this handbook were made this way. Run a small OpenAI-compatible server on localhost that answers <code>GET /v1/models</code> and streams <code>POST /v1/chat/completions</code>, then point an isolated agent directory at it:</p>
{code('''export GOSHCODER_AGENT_DIR=$(mktemp -d)
goshcoder omni setup            # URL: http://127.0.0.1:18080, blank key
goshcoder -m omni/<model-id>''')}
<p>Driving the interface in <code>tmux</code> (<code>send-keys</code>, <code>capture-pane -e</code>) gives reproducible captures, including tool calls, retries (answer HTTP 500) and errors (HTTP 400).</p>
""", audience="developer", topic="development", status="runtime"))

    u.append(unit("dev-catalog", "Updating model data", f"""
<p><code>data/catalog.json</code> is produced by pi's <code>scripts/generate-models.ts</code> run against <code>reference/pi</code>; hand edits are lost on the next regeneration. Put models pi does not have in <code>catalog_extra.json</code> and corrections in <code>catalog_overrides.json</code> (a partial model object whose top-level keys replace the generated ones; <code>id</code> and <code>provider</code> are refused). <code>xhigh</code> and <code>max</code> reasoning levels are offered only when a model's <code>thinkingLevelMap</code> lists them.</p>
""", audience="developer", topic="development providers", status="source"))
    u.append(unit("dev-handbook", "Updating this handbook", f"""
<p>This page is generated. Its content lives in Python modules under <code>docs/handbook/src/</code> (<code>hb_part1.py</code> overview and getting started, <code>hb_part2.py</code> user manual, <code>hb_part3.py</code> architecture through coverage, <code>hb_findings.py</code> audit findings, <code>hb_lib.py</code> styles, scripts and helpers). Regenerate with the standard library only:</p>
{code("python3 docs/handbook/src/build.py")}
<p>The build reports missing anchors, missing screenshots and duplicate ids. Screenshots are PNG files in <code>docs/handbook/screenshots/</code>; replace them when the interface changes, and update the version, revision and audit date in <code>META</code> in <code>hb_lib.py</code>.</p>
""", audience="developer maintainer", topic="development docs", status="runtime"))
    return chapter("development", "Development", '<p class="lead">Building, testing and changing the code.</p>', "\n".join(u))


def build_release():
    u = []
    targets = [
        ("build", "Release binary in <code>bin/</code>, stamped with <code>git describe</code> through <code>GOSHCODER_VERSION</code>"),
        ("install / uninstall", "Copy to / remove from <code>$CARGO_INSTALL_ROOT/bin</code> or <code>~/.cargo/bin</code>"),
        ("run", "Build and start chat"),
        ("check", "<code>fmt-check vet lint test test-hermetic vuln</code>"),
        ("fmt, fmt-check, vet, test, test-race, test-hermetic, lint, vuln, cover, tools", "Individual steps (<code>test-race</code> is an alias of <code>test</code>; <code>vuln</code> and <code>cover</code> skip when the tool is missing)"),
        ("dist", "Release archives for this host: Linux builds linux/amd64, linux/arm64 (musl) and windows/amd64 with cargo-zigbuild; macOS builds both Apple targets"),
        ("dist-name, checksums, clean-dist, clean, help", "Archive name helper, <code>dist/checksums.txt</code>, cleanup, target list"),
    ]
    u.append(unit("makefile", "Makefile targets", f"""
{table(["Target", "What it does"], [[f"<code>{a}</code>", b] for a, b in targets], "wrapcode")}
<p><code>make dist</code> also needs <code>zig</code>, <code>zip</code>, <code>python3</code> and the Rust targets installed with <code>rustup target add</code>. Each archive contains the binary, <code>README.md</code>, <code>NOTICE</code>, <code>LICENSE</code> and a <code>licenses/</code> directory gathered by <code>scripts/collect-licenses.sh</code>, which fails the build if a dependency has no licence text.</p>
""", audience="developer maintainer", topic="build", status="runtime"))

    u.append(unit("ci", "Continuous integration", f"""
<p>{src('.github/workflows/ci.yml')} runs on every push and pull request:</p>
<ul>
<li><strong>build &amp; test</strong> on Ubuntu, macOS and Windows: <code>cargo build</code>, <code>cargo test</code>, and the tests again with dummy AWS credentials.</li>
<li><strong>rustfmt, clippy, cargo-audit</strong> on Ubuntu only.</li>
<li><strong>release targets</strong>: <code>make dist</code> on Ubuntu and macOS.</li>
<li><strong>installer scripts</strong>: ShellCheck on <code>install.sh</code>, a PowerShell parse check of <code>install.ps1</code>, a from-source install, and <code>scripts/release-roundtrip.sh</code> (which builds and serves an archive but does not run an installer, <a href="#F-24">F-24</a>).</li>
</ul>
""", audience="developer maintainer", topic="build", status="source"))

    u.append(unit("release", "Cutting a release", f"""
<p>{src('.github/workflows/release.yml')} has three triggers that share one path: pushing a <code>v*</code> tag; pushing a <code>release/v*</code> branch (the workflow creates the tag); or a manual run naming the tag and ref. The tag must match <code>v1.2.3</code> or <code>v1.2.3-rc.1</code>. The Linux job runs <code>make check</code> and <code>cargo audit</code>, both hosts run <code>make dist VERSION=&lt;tag&gt;</code>, and a final job merges the archives, writes one <code>checksums.txt</code> and publishes the GitHub release.</p>
<ol class="steps">
<li>Make sure <code>cargo audit</code> is clean (it is not at this revision, <a href="#F-04">F-04</a>).</li>
<li>Push the tag or <code>release/vX.Y.Z</code> branch.</li>
<li>After publishing, merge any release branch back and bump the version in <code>Cargo.toml</code>, <code>Cargo.lock</code>, the Makefile and both installers (not done after v0.6.0, <a href="#F-46">F-46</a>).</li>
</ol>
<p>Release candidates are published as normal releases, so <code>releases/latest</code> can point at one (<a href="#F-45">F-45</a>).</p>
""", audience="maintainer", topic="build", status="source"))
    return chapter("build-release", "Build and release", '<p class="lead">Packaging, CI and the release workflow.</p>', "\n".join(u))


def troubleshooting():
    items = [
        ("ts-no-model", "“no authenticated model is available”", "<code>run</code> or a prompt with no provider configured.", "Run <code>goshcoder providers</code>, then <code>goshcoder auth login &lt;provider&gt;</code> or <code>auth set &lt;provider&gt;</code>, or set the provider's API-key variable. In chat, <code>/login</code>.", "runtime"),
        ("ts-unknown-model", "error: unknown model \"…\"", "A <code>-m</code> value or <code>GOSHCODER_MODEL</code> that is not in the catalog, or whose provider is not configured.", "Check <code>goshcoder models</code>. Unset or fix <code>GOSHCODER_MODEL</code>; an invalid value stops chat from starting.", "runtime"),
        ("ts-tool-not-found", "✗ ls: Tool ls not found (in <code>run</code>)", "<code>run</code> has no tools by default.", "Add <code>-tools</code>.", "runtime"),
        ("ts-unknown-command", "unknown command /x; /help lists the available commands", "A typo, or a documented command that does not exist (<code>/use-default-tui</code>, <code>/logout</code>).", "Use <code>/help</code>. To remove a credential use <code>goshcoder auth logout &lt;provider&gt;</code>.", "runtime"),
        ("ts-ralph-sync", "error: synchronize Ralph loop: refusing Ralph path … because it is a symlink or escapes the workspace directory", "The workspace's <code>.ralph</code> is a symlink or a file.", "Start with <code>-ralph=false</code>, or remove/replace <code>.ralph</code> with a real directory.", "runtime"),
        ("ts-retrying", "provider request failed with status 500 … Retrying in 2s (attempt 2 of 4)", "A transient provider error.", "Wait, or press <kbd>Esc</kbd> to stop retrying. Persistent errors: check the provider's status page.", "runtime"),
        ("ts-400", "provider request failed with status 400: …", "The provider rejected the request (bad parameter, model unavailable for your account, context too long).", "Read the provider's message; try another model; <code>/compact</code> if the context is large.", "runtime"),
        ("ts-oauth-expired", "OAuth credential for \"…\" is no longer authorized; run `goshcoder auth login …`", "The refresh token was revoked or expired.", "Log in again. A transient refresh failure (“could not be refreshed right now”) is retried after 30 s.", "source"),
        ("ts-auth-lock", "timed out after 60s waiting for another process to release auth.json", "Another GoshCoder process holds the credential lock, or pi left a lock directory (F-20).", "Close other sessions; if pi is involved, remove a stale <code>auth.json.lock</code> directory.", "source"),
        ("ts-compact", "there is not enough conversation history to compact", "<code>/compact</code> with fewer than three messages in context.", "Nothing to do.", "runtime"),
        ("ts-planner-browser", "Planner: could not open a browser (…). Open this URL to continue the review", "No browser launcher (headless machine).", "Open the printed <code>http://127.0.0.1:…/</code> URL in a browser on the same machine (e.g. through an SSH port forward).", "runtime"),
        ("ts-planner-nochanges", "there are no changes to review", "<code>/planner-review</code> with a clean working tree and index.", "Make changes, stage them, or pass a pull-request URL.", "runtime"),
        ("ts-omni-tty", "OmniRoute setup requires an interactive terminal", "<code>goshcoder omni setup</code> with redirected input.", "Run it in a terminal (or <code>/omni setup</code> in chat), or set <code>OMNIROUTE_URL</code>.", "runtime"),
        ("ts-export-missing", "…: No such file or directory (os error 2) after /export", "The target directory does not exist.", "Create the directory first.", "runtime"),
        ("ts-busy", "Session opened read-only / “another process took over this session”", "The session is open in another window, or its lock was taken over.", "Close the other window; use <code>/clone</code> to continue separately.", "source"),
        ("ts-too-small", "Terminal is too small", "Window below 20×8 cells.", "Enlarge the terminal.", "runtime"),
        ("ts-garbled", "Terminal left in a strange state after a crash", "The process was killed with SIGKILL, or a worker panicked (F-22).", "Run <code>reset</code> (or <code>stty sane</code>). Sessions are intact up to the last completed message.", "inferred"),
        ("ts-select-text", "Cannot select text with the mouse", "The interface captures the mouse.", "Hold <kbd>Shift</kbd> (or your terminal's bypass modifier) while dragging. Login URLs are also copied to the clipboard automatically.", "source"),
        ("ts-broken-pipe", "error: Broken pipe (os error 32)", "Piping <code>sessions show</code> into <code>head</code> or similar.", "Harmless; the output you saw is complete up to that point.", "runtime"),
    ]
    units = []
    for uid, msg, cause, fix, st in items:
        units.append(unit(uid, f"<code>{msg}</code>" if not msg.startswith("“") and "<code>" not in msg else msg, f"<dl class=\"kv\"><dt>Cause</dt><dd>{cause}</dd><dt>Fix</dt><dd>{fix}</dd></dl>", audience="user", topic="troubleshooting", status=st, level=3))
    return chapter("troubleshooting", "Troubleshooting", '<p class="lead">Messages you may see, with their cause and fix. Messages marked “Verified at runtime” were reproduced while writing this handbook.</p>', "\n".join(units))


# ---------------- coverage ----------------

T, I, B, N = "runtime", "source", "blocked", "na"

COVERAGE = [
 # (area, item, status, notes)
 ("CLI", "goshcoder version, --version, help, &lt;cmd&gt; --help", T, "All subcommand help texts printed"),
 ("CLI", "unknown command suggestion", T, "<code>provider</code> → did you mean <code>providers</code>"),
 ("CLI", "run (prompt, -tools, -tools=false, errors, retries, exit status)", T, "Exit 1 on HTTP 400; retries 2/4/8 s then success"),
 ("CLI", "providers, models, models &lt;provider&gt;", T, "No-credential and configured states"),
 ("CLI", "auth list, logout, set (via TUI), login (started + cancelled)", T, "No real login completed"),
 ("CLI", "auth set from the shell, auth login completion", B, "Needs real accounts"),
 ("CLI", "omni status, setup, sync, models, test, dashboard, config, help", T, "Against a local test gateway"),
 ("CLI", "aperture status, help", T, "Unconfigured state only"),
 ("CLI", "aperture onboarding, settings, sync, providers, connectors, pin/unpin", B, "Needs a Tailscale Aperture gateway"),
 ("CLI", "grok-cli usage, accounts (logged out)", T, ""),
 ("CLI", "grok-cli accounts add/login/logout/rename/remove/use", B, "Needs Grok accounts"),
 ("CLI", "ralph start, list, status, delete", T, "status rejects a name (F-32)"),
 ("CLI", "ralph resume, archive", I, ""),
 ("CLI", "sessions list (--all), show (--full), export (html/md/jsonl/stdout), import, rm, gc (dry run)", T, ""),
 ("CLI", "sessions share --yes, gc --yes", I, "Not run: would publish / delete across workspaces"),
 ("CLI", "prompts list, backup, restore, restore --dry-run", T, ""),
 ("Flags", "-m, -tools, -tools=false, -no-session, -name, -continue, -resume, -fullscreen=false", T, ""),
 ("Flags", "-C, -session, -read-only, -sessions-dir, -quiet, -system, -thinking, -planner", I, ""),
 ("Flags", "-claude-tui", T, "Accepted; no effect (F-14)"),
 ("Screens", "First launch with no credentials (pre-filled /login)", T, "Screenshot"),
 ("Screens", "Main screen, sidebar, status bar, composer", T, "Screenshots"),
 ("Screens", "Narrow layout (&lt;96 cols) and too-small screen", T, "Screenshots"),
 ("Screens", "Command palette, model picker, thinking picker, login picker", T, "Screenshots"),
 ("Screens", "API-key prompt (masked), login method choice, OAuth wait", T, "Screenshots; URL redacted"),
 ("Screens", "Device-code login display", B, "Needs a device-code provider account"),
 ("Screens", "Streaming, abort, retry countdown, error card", T, "Screenshots"),
 ("Screens", "Tool cards collapsed/expanded, failed tool card", T, "Screenshots"),
 ("Screens", "Planner review page (annotate, feedback)", T, "Driven with Playwright"),
 ("Screens", "Planner approval → executing with checklist", B, "Needs a model that writes and submits a plan"),
 ("Screens", "HTML export page", T, "Screenshot"),
 ("Screens", "-resume picker, line mode", T, "Screenshots"),
 ("Slash", "/help, /hotkeys, /status, /tools, /messages, /queue, /resources, /reload, /sessions, /tree", T, ""),
 ("Slash", "/model, /thinking, /login, /resume (picker and id)", T, ""),
 ("Slash", "/clear, /new, /name, /fork, /label, /clone, /export, /import", T, ""),
 ("Slash", "/share (confirmation text)", T, "Upload not performed"),
 ("Slash", "/share confirm", I, "Would publish a gist"),
 ("Slash", "/steer, /followup, /compact, /system (show and set)", T, ""),
 ("Slash", "/btw, /btw list, /btw settings", T, ""),
 ("Slash", "/btw resume, /btw bring", I, ""),
 ("Slash", "/prompt list, save, edit (no editor), backup/restore via CLI", T, ""),
 ("Slash", "/ralph start, status, stop", T, ""),
 ("Slash", "/planner, /planner-annotate, /planner-review (no changes), /planner-last", T, ""),
 ("Slash", "/omni, /aperture (unconfigured)", T, ""),
 ("Slash", "/grok-cli-usage, /grok-cli-accounts, /grok-cli-conv, /grok-cli-imagine:tool status", T, "Logged-out states"),
 ("Slash", "/grok-cli-imagine (generation)", B, "Needs a Grok CLI login"),
 ("Slash", "/use-default-tui, /use-claude-code-tui, /logout", T, "Do not exist"),
 ("Keys", "Enter, Ctrl+J, Esc, Ctrl+C, Ctrl+D, Ctrl+L, Ctrl+P, Shift+Tab, Ctrl+O, Ctrl+T, PgUp, Ctrl+End, Ctrl+U", T, ""),
 ("Keys", "Alt+Enter, word movement, Ctrl+K/W, mouse wheel, Ctrl+Shift+P", I, ""),
 ("Workflows", "Tool calls: ls, read, write, bash", T, ""),
 ("Workflows", "Tool calls: edit, grep, find", I, ""),
 ("Workflows", "web_search", B, "No live search service used"),
 ("Workflows", "Desktop control (mcp tool)", B, "No computer-use-linux / desktop"),
 ("Workflows", "Steering, queue, abort restoring queued text", T, ""),
 ("Workflows", "Automatic compaction near the limit", I, "Manual /compact verified"),
 ("Workflows", "Planner write gate during planning", T, ""),
 ("Workflows", "Prompt-template hijack (F-01), SYSTEM.md replacement (F-06), -tools=false with executing planner (F-02)", T, ""),
 ("Workflows", ".ralph symlink (F-16), toolCalling override loss (F-17), invalid GOSHCODER_MODEL (F-47)", T, ""),
 ("Workflows", "SIGTERM during a reply (exit 143, session saved)", T, ""),
 ("Workflows", "Piped stdin chat (line mode fallback)", T, ""),
 ("Workflows", "Two windows on one session (read-only fallback)", I, ""),
 ("Providers", "openai-completions protocol (via OmniRoute)", T, "Streaming text and tool calls"),
 ("Providers", "Other 8 protocols, OAuth refresh, Bedrock SigV4, Vertex ADC", I, "Covered by the 714 unit tests, not live"),
 ("Build", "cargo fmt --check, cargo build, cargo test (689+25), Clippy -D warnings", T, ""),
 ("Build", "make check, make build", T, "make check skipped vuln (cargo-audit absent at that point)"),
 ("Build", "cargo audit", T, "Fails: RUSTSEC-2026-0285 (F-04)"),
 ("Build", "make dist, release workflow, CI workflows", I, "Needs zig/cross targets and GitHub"),
 ("Build", "install.sh, install.ps1", I, "Not run (would install system-wide / no PowerShell)"),
 ("Platforms", "Linux x86_64", T, ""),
 ("Platforms", "macOS, Windows", B, "No machines available"),
]

MODULE_COVERAGE = [
 ("main.rs, state.rs, ui.rs, markdown.rs, tui_login.rs, line_editor.rs, session_picker.rs", T),
 ("runtime.rs, config.rs, agent.rs, turns.rs, compaction.rs, llm.rs, stream.rs", T),
 ("providers.rs (completions path), catalog.rs, provider_cli.rs, oauth.rs (flow start)", T),
 ("bedrock.rs, mistral.rs, google_auth.rs, omni_prompt_tools.rs", I),
 ("tools.rs, session.rs, sessionlog.rs, sessions.rs, export_html.rs, resources.rs, prompts.rs", T),
 ("webaccess.rs, computeruse.rs", I),
 ("plannotator.rs, planner_runtime.rs, ralph.rs, ralph_runtime.rs, ralph_cli.rs, btw.rs, btw_runtime.rs", T),
 ("omniroute.rs, omni_cli.rs", T),
 ("aperture.rs, aperture_cli.rs, aperture_mcp.rs, aperture_tools.rs, tests/aperture_core.rs", I),
 ("grok_cli.rs, grok_accounts.rs, grok_imagine.rs, meta_muse.rs", I),
 ("data/catalog.json, catalog_extra.json, catalog_overrides.json", I),
 ("Makefile, install.sh, install.ps1, scripts/*.sh, .github/workflows/*.yml, .cargo/audit.toml, rust-toolchain.toml", I),
]


def coverage():
    total = len(COVERAGE)
    c = {}
    for _, _, s, _ in COVERAGE:
        c[s] = c.get(s, 0) + 1
    rows = []
    for i, (area, item, st, note) in enumerate(COVERAGE, 1):
        rows.append(f'<tr class="unit" id="cov-{i}" data-audience="developer maintainer" data-topic="coverage" data-platform="all" data-status="{st}"><td>{area}</td><td>{item}</td><td>{badge(st)}</td><td>{note}</td></tr>')
    mods = "".join(f"<tr><td>{', '.join('<code>'+esc(x.strip())+'</code>' for x in a.split(','))}</td><td>{badge(s)}</td></tr>" for a, s in MODULE_COVERAGE)
    body = f"""<p class="lead">What was exercised for this handbook. {c.get(T,0)} of {total} items were verified at runtime, {c.get(I,0)} by source inspection only, and {c.get(B,0)} were blocked.</p>
<section class="unit" id="coverage-method" data-audience="developer maintainer user" data-topic="coverage" data-platform="all" data-status="docs">
<h3><a class="anchor" href="#coverage-method">#</a>Method {badge('docs')}</h3>
<p>All 49 modules in <code>src/</code>, the integration test, the model data, the Makefile, both installers, the scripts and both workflows were read by six parallel source reviews (agent and tools; providers and catalog; authentication; sessions and resources; extensions and gateways; interface and packaging), whose top findings were then checked against the code. The release binary and a debug build were run on Linux in an isolated agent directory, driven through tmux against a scripted local gateway that logged every request. Screenshots are captures of the real terminal rendered to PNG, and of the real pages in Chromium. Generated files (<code>target/</code>), third-party crates, and the gitignored <code>reference/pi</code> clone are excluded.</p>
<p><strong>Not covered:</strong> live providers and logins, Grok CLI and Meta Muse beyond logged-out states, Aperture, web search against real services, desktop control, gist upload, macOS and Windows.</p>
</section>
<div class="tablewrap"><table><thead><tr><th scope="col">Area</th><th scope="col">Item</th><th scope="col">Status</th><th scope="col">Notes</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<section class="unit" id="coverage-modules" data-audience="developer maintainer" data-topic="coverage architecture" data-platform="all" data-status="source">
<h3><a class="anchor" href="#coverage-modules">#</a>Modules {badge('source')}</h3>
<p>Every module was read. “Verified at runtime” means code in that module ran during the session above.</p>
<div class="tablewrap"><table><thead><tr><th scope="col">Modules</th><th scope="col">Deepest verification</th></tr></thead><tbody>{mods}</tbody></table></div>
</section>"""
    return chapter("coverage", "Coverage", "", body)
