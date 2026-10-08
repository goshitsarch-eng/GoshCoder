from hb_lib import *


def overview():
    m = META
    units = []
    units.append(unit("what-it-is", "What GoshCoder is", f"""
<p>GoshCoder is a terminal coding agent written in Rust. You type a request; a large language model answers and, when tools are enabled, reads and edits files in your project and runs shell commands for you. It is a reimplementation of the TypeScript agent <a href="https://github.com/earendil-works/pi">pi</a>: it uses pi's session file format, credential file format, tool names and model catalog on purpose, so the two tools can share data.</p>
<div class="grid2">
<div class="card"><h4>Interfaces</h4><p>A fullscreen terminal interface (Ratatui), a plain line-mode chat for pipes and screen readers, and a one-shot <code>run</code> command for scripts.</p></div>
<div class="card"><h4>Providers</h4><p>45 provider definitions and 1,256 bundled models over nine wire protocols, plus two self-hosted gateways (OmniRoute, Tailscale Aperture). Logins for eight subscription providers.</p></div>
<div class="card"><h4>Tools</h4><p><code>read</code>, <code>write</code>, <code>edit</code>, <code>ls</code>, <code>grep</code>, <code>find</code>, <code>bash</code>, <code>web_search</code>, plus optional desktop control, image generation and gateway connector tools.</p></div>
<div class="card"><h4>Built-in extensions</h4><p>Planner (plan review in the browser), Ralph (long-running loops), BTW (side questions), Grok CLI accounts and image generation, Meta Muse Code.</p></div>
</div>
""", audience="user developer", topic="overview", status="runtime"))

    units.append(unit("audited-version", "Audited version and limits of this handbook", f"""
<dl class="kv">
<dt>Application</dt><dd>GoshCoder {m['version']} (<code>Cargo.toml</code>); <code>make build</code> stamps <code>{m['describe']}</code></dd>
<dt>Revision</dt><dd><code>{m['revision_full']}</code> on branch <code>{m['branch']}</code> (27 commits after tag <code>v0.6.0</code>)</dd>
<dt>Audit date</dt><dd>{m['audit_date']}</dd>
<dt>Test machine</dt><dd>Linux x86_64 container, {m['toolchain']}, tmux 3.4, Chromium (Playwright) for screenshots</dd>
<dt>Platforms</dt><dd>Release archives for Linux amd64/arm64, macOS amd64/arm64 and Windows amd64. Only Linux was run.</dd>
</dl>
<div class="callout warn">
<p><strong>What was and was not verified.</strong> Every screen and most commands were exercised on Linux against a local mock OpenAI-compatible gateway (configured through OmniRoute), so turns, tools, retries, errors, sessions, exports, Planner, Ralph and BTW ran for real. No real provider account was used: browser and device-code logins were started and cancelled but never completed, and Grok CLI, Meta Muse, Aperture, web search, desktop control and gist upload were not exercised against live services. macOS and Windows behaviour comes from source inspection only.</p>
</div>
<p>Each section and finding carries a badge: {badge('runtime')} means it was exercised on the running program; {badge('source')} means it was read in the code; {badge('inferred')} is a conclusion from code that was not observed; {badge('blocked')} could not be checked here. See <a href="#coverage">Coverage</a> for the item-by-item list.</p>
""", audience="user developer maintainer", topic="overview", status="docs"))

    units.append(unit("limitations", "Current limitations at a glance", f"""
<ul>
<li>No plugin host, package manager, LSP, MCP server management or custom <code>models.json</code> loading (pi has these).</li>
<li>Cloudflare AI Gateway and Workers AI models are listed but every request goes to an unusable URL (finding <a href="#F-03">F-03</a>).</li>
<li>Amazon Bedrock works with static keys and the Bedrock bearer token only; SSO, container and web-identity credentials are detected but not used (<a href="#F-19">F-19</a>).</li>
<li><code>-claude-tui</code>, <code>/use-default-tui</code> and <code>/use-claude-code-tui</code> do nothing or do not exist (<a href="#F-14">F-14</a>).</li>
<li>Fullscreen mode ignores <code>NO_COLOR</code> and always draws a dark theme.</li>
<li>Windows: no signal handling, and AltGr characters cannot be typed in the editors (inferred, <a href="#F-27">F-27</a>).</li>
</ul>
""", audience="user", topic="overview", status="source"))

    units.append(unit("orientation", "How this handbook is organised", """
<p><a href="#getting-started">Getting started</a> and the <a href="#manual">User manual</a> are for people using GoshCoder. <a href="#architecture">Architecture</a>, <a href="#development">Development</a> and <a href="#build-release">Build and release</a> are for people changing it. <a href="#troubleshooting">Troubleshooting</a> collects real error messages. <a href="#findings">Audit findings</a> lists defects with evidence, kept apart from the instructions; <a href="#coverage">Coverage</a> lists what was tested.</p>
<p>Use the search box (press <kbd>/</kbd>) and the filters to narrow everything down; matching sections open automatically. Every heading has a stable link you can bookmark.</p>
""", audience="user developer maintainer", topic="overview", status="docs"))

    return chapter("overview", "Overview", f'<p class="lead">A verified user manual, architecture guide and audit for GoshCoder {META["version"]}.</p>', "\n".join(units))


def getting_started():
    u = []
    u.append(unit("requirements", "System requirements", f"""
<ul>
<li><strong>Operating system:</strong> Linux (x86_64, arm64; static musl builds), macOS (Intel, Apple Silicon), Windows 10/11 (x64; ARM64 runs the x64 build under emulation).</li>
<li><strong>Terminal:</strong> any terminal with 24-bit colour for the fullscreen interface; at least 20×8 cells (below that it shows “Terminal is too small”). The sidebar appears at 96 columns and wider. A terminal with the kitty keyboard protocol makes <kbd>Shift</kbd>+<kbd>Enter</kbd> and <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd> work.</li>
<li><strong>An account with a model provider</strong> (subscription login or API key), or a local OpenAI-compatible gateway.</li>
<li><strong>Optional tools:</strong> <code>git</code> (faster <code>grep</code>/<code>find</code>, Planner code review), <code>gh</code> (<code>/share</code>, Planner PR review), <code>computer-use-linux</code> (desktop control, Linux only), <code>$VISUAL</code>/<code>$EDITOR</code> (<code>/prompt edit</code>).</li>
<li><strong>Building from source:</strong> current stable Rust (the code uses let-chains; tested with rustc 1.97.0). <code>rust-toolchain.toml</code> selects stable with clippy and rustfmt.</li>
</ul>
""", audience="user", topic="install", platform="all", status="source"))

    u.append(unit("install", "Install", f"""
<h4>Linux and macOS</h4>
{code("curl -fsSL https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.sh | sh")}
<p>Options (pass them with <code>sh -s -- …</code> when piping): <code>--dir &lt;path&gt;</code>, <code>--version &lt;tag&gt;</code>, <code>--from-source</code>, <code>--no-modify-path</code>. The same can be set with <code>GOSHCODER_INSTALL_DIR</code>, <code>GOSHCODER_VERSION</code> and <code>GOSHCODER_FROM_SOURCE=1</code>. The default directory is the first of <code>~/.cargo/bin</code> or <code>~/.local/bin</code> that is already on <code>PATH</code>, else <code>~/.local/bin</code>.</p>
{code('curl -fsSL https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.sh | sh -s -- --version v0.6.0 --dir "$HOME/bin"')}
<h4>Windows (PowerShell)</h4>
{code("irm https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.ps1 | iex", "powershell")}
<p>The PowerShell installer takes PowerShell-style parameters: <code>-InstallDir</code> (default <code>%LOCALAPPDATA%\\Programs\\goshcoder</code>), <code>-Version</code>, <code>-FromSource</code>, <code>-NoModifyPath</code>. Through <code>irm | iex</code> you cannot pass them; set <code>GOSHCODER_INSTALL_DIR</code> or <code>GOSHCODER_VERSION</code> first, or download the script and run it.</p>
<h4>What the installers do</h4>
<ol class="steps">
<li>Download <code>goshcoder_&lt;version&gt;_&lt;os&gt;_&lt;arch&gt;</code> (<code>.tar.gz</code>, or <code>.zip</code> on Windows) and <code>checksums.txt</code> from the GitHub release.</li>
<li>Check the SHA-256. The PowerShell installer refuses a missing or mismatched checksum. <code>install.sh</code> also refuses a mismatch, but installs without checking when neither <code>sha256sum</code> nor <code>shasum</code> exists (<a href="#F-38">F-38</a>).</li>
<li>Replace the binary in place, run <code>goshcoder version</code>, and offer to add the directory to your shell profile (only with a terminal on stdin; <code>curl | sh</code> just prints the line to add).</li>
</ol>
<div class="callout warn"><p><strong>Fallback to source.</strong> If <em>any</em> download step fails, both installers build the current default branch from source instead, even if you asked for a specific <code>--version</code> (<a href="#F-23">F-23</a>). Read the warnings it prints.</p></div>
<h4>From a checkout</h4>
{code('''git clone https://github.com/goshitsarch-eng/goshcoder
cd goshcoder
make build      # release binary in bin/goshcoder, stamped with git describe
make install    # copies it to ~/.cargo/bin (or $CARGO_INSTALL_ROOT/bin)''')}
<p>Verified here: <code>make build</code> produced <code>bin/goshcoder</code> reporting <code>goshcoder v0.6.0-27-g3870a05</code>. A plain <code>cargo build</code> reports <code>0.6.0</code>.</p>
""", audience="user", topic="install", platform="linux macos windows", status="source"))

    u.append(unit("first-launch", "First launch", f"""
<p>Run <code>goshcoder</code> in your project directory. With no credentials it opens fullscreen chat with the <strong>Add provider</strong> picker already open.</p>
{fig("first-launch-login-picker", "First launch with no credentials: a notice explains the next step and <code>/login</code> is pre-filled, so the provider list is open. Subscription logins come first, then API-key providers.", "GoshCoder fullscreen interface on first launch, showing a notice that no provider is authenticated and an Add Provider list with anthropic, openai-codex, grok-cli, xai, meta, meta-muse, kimi-coding, openai and google.")}
<ol class="steps">
<li>Type to filter the list (for example <code>deeps</code>), use <kbd>↑</kbd>/<kbd>↓</kbd>, and press <kbd>Enter</kbd>.</li>
<li>For an API-key provider, paste the key and press <kbd>Enter</kbd>. The key is masked and stored in <code>~/.goshcoder/agent/auth.json</code> with mode 0600.</li>
<li>For a subscription, choose the login method; the sign-in address (or device code) is shown and also copied to your clipboard through the terminal. See <a href="#login">Logging in</a>.</li>
<li>The first successful login selects that provider's default model. The next thing you type is a prompt.</li>
</ol>
{fig("login-api-key-prompt", "Entering an API key: the answer field shows bullets, not the key.", "API key prompt for deepseek with the typed key shown as a row of bullets.")}
{fig("login-success", "After the key is saved the model is set (here deepseek/deepseek-v4-pro) and the sidebar shows its context window.", "Notice confirming the key was stored and the model set to deepseek/deepseek-v4-pro; the sidebar shows 0 / 1,000,000 tokens.")}
<p>To leave, type <code>/exit</code>, or press <kbd>Ctrl</kbd>+<kbd>D</kbd> on an empty editor. If the session was saved, GoshCoder prints <code>Resume with: goshcoder chat -continue</code>.</p>
""", audience="user", topic="install interface providers", status="runtime"))

    u.append(unit("quick-start", "Quick-start tutorial", f"""
<p>This walk-through uses the real commands; the screenshots come from a session against a local test gateway, so the model replies are canned.</p>
<ol class="steps">
<li><strong>Start in your project.</strong> <code>cd ~/code/todo-app &amp;&amp; goshcoder</code>. The sidebar shows the model, context use and workspace.</li>
<li><strong>Ask something.</strong> Type <code>Please list files in the project</code> and press <kbd>Enter</kbd>. While the reply streams, the status bar shows a spinner and elapsed time; typing and pressing <kbd>Enter</kbd> now <em>steers</em> the reply instead of starting a new one.</li>
<li><strong>Watch tool calls.</strong> Each tool call becomes a card (<code>✓ ls .</code>, <code>✓ read README.md</code>). Press <kbd>Ctrl</kbd>+<kbd>O</kbd> to expand or collapse their output.</li>
<li><strong>Switch model.</strong> Press <kbd>Ctrl</kbd>+<kbd>L</kbd> (or type <code>/model</code>), type part of a name, press <kbd>Enter</kbd>. The choice is remembered for next time.</li>
<li><strong>Name the session.</strong> <code>/name Due dates spike</code>. Sessions are saved automatically under <code>~/.goshcoder/agent/sessions</code>.</li>
<li><strong>Come back later.</strong> <code>goshcoder -continue</code> reopens the latest conversation in this directory; <code>goshcoder -resume</code> lets you pick one.</li>
</ol>
{fig("tool-cards", "Two turns with tool calls. Tool cards show three lines of output; <kbd>Ctrl</kbd>+<kbd>O</kbd> expands them. The sidebar's Activity block counts turns and tools and shows the last tool.", "Transcript with user messages on a grey bar, assistant replies, and tool cards for ls and read README.md; the sidebar shows 3 turns and 2 tools.")}
<p>For one-off scripted use there is <code>goshcoder run</code>. Note that <code>run</code> has <strong>no tools unless you pass <code>-tools</code></strong>:</p>
{code('''goshcoder run -m anthropic/claude-sonnet-5 -tools "explain this repository"''')}
""", audience="user", topic="interface tools", status="runtime"))

    return chapter("getting-started", "Getting started", '<p class="lead">Install GoshCoder, connect a model provider and run your first session.</p>', "\n".join(u))
