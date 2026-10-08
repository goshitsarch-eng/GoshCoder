from hb_lib import *


def kb(*keys):
    return "+".join(f"<kbd>{esc(k)}</kbd>" for k in keys)


def manual():
    u = []

    # ---------------- interface ----------------
    u.append(unit("interface-tour", "A tour of the interface", f"""
<p>Fullscreen chat is the default when GoshCoder runs in a terminal. It draws on the terminal's alternate screen, so your scrollback is untouched when you leave.</p>
{fig("interface-tour-annotated", "The main screen at 140×42 columns. Numbers refer to the list below; the unannotated original is in the quick-start tutorial.", "Annotated GoshCoder main screen: 1 header, 2 transcript with tool cards, 3 composer, 4 status bar, 5 sidebar.")}
<ol class="callouts">
<li><strong>Header</strong> — the session name, or the first message, or “interactive session”.</li>
<li><strong>Transcript</strong> — your messages on a grey bar, replies rendered as Markdown, tool calls as cards (<code>●</code> running, <code>✓</code> done, <code>×</code> failed), notices (<code>i Notice</code>), errors (<code>× Error</code>) and echoed commands (<code>◇ Command</code>). It is anchored to the bottom and capped at 124 columns wide.</li>
<li><strong>Composer</strong> — where you type. It grows to eight lines. Type <kbd>/</kbd> for the command palette.</li>
<li><strong>Status bar</strong> — <code>● Ready</code>, or a spinner with the current activity and elapsed time, and the keys that apply right now.</li>
<li><strong>Sidebar</strong> (96 columns and wider) — session title, model, thinking level and mode, whether the session is being recorded, a context-usage bar with tokens and cost, an Activity block (turns, tools, changed files, last tool), the Planner checklist when there is one, and the workspace path.</li>
</ol>
{fig("narrow-90-cols", "Below 96 columns the sidebar is hidden and the context percentage moves into the status bar.", "Narrow GoshCoder screen without a sidebar; the status bar reads Ready · 1% context.", wide=False)}
{fig("too-small", "Below 20×8 cells only this message is drawn.", "Text reading GoshCoder, Terminal is too small.", wide=False)}
{fig("startup-empty", "A new session once a model is selected: an empty transcript, the composer placeholder and the sidebar.", "Empty GoshCoder session with the composer placeholder Tell GoshCoder what to build… and a sidebar showing omni/mock-coder and 0 / 128,000 tokens.")}
""", topic="interface", status="runtime"))

    u.append(unit("composer", "Typing, sending and steering", f"""
<ul>
<li><kbd>Enter</kbd> sends. {kb('Ctrl','J')} inserts a newline ({kb('Shift','Enter')} and {kb('Ctrl','Enter')} too, on terminals with the kitty keyboard protocol). Pasted text is never sent line by line; pastes over 20 lines are shown as <code>[pasted N lines]</code>.</li>
<li>While a reply is streaming, <kbd>Enter</kbd> <strong>steers</strong> it: the message is delivered to the model at the next opportunity. {kb('Alt','Enter')} instead <strong>queues a follow-up</strong> for after the reply. Queued messages appear above the composer as <code>↳ queued …</code>; <code>/queue</code> lists them.</li>
<li><kbd>Esc</kbd> while streaming aborts the reply. Queued messages are put back into the editor so nothing runs after an interrupted turn; the partial reply is kept and marked “Interrupted”.</li>
<li><kbd>Esc</kbd> on a draft clears it; <kbd>↑</kbd> brings it back. <kbd>↑</kbd>/<kbd>↓</kbd> also move between lines and through history.</li>
<li>Editing works by grapheme, so emoji and accented letters are never split.</li>
</ul>
{fig("multiline-composer", "A two-line message typed with Ctrl+J.", "Composer containing two lines: Write a haiku about / todo lists and deadlines.")}
{fig("streaming-reply", "A reply streaming: the status bar shows the spinner, “Responding” and the elapsed time, and that typing will steer.", "Transcript with a partially streamed reply ending in a block cursor; status bar reads Responding · 6s, esc abort · type to steer.")}
{fig("aborted", "Esc during a reply: the queued steer (“Also mention tests”) is restored to the composer and the partial reply is marked as interrupted.", "Transcript with a partial reply followed by Interrupted · stopped before the reply finished; the composer contains Also mention tests.")}
""", topic="interface", status="runtime"))

    keys = [
        ("Enter", "Send; accept the palette item; steer while a reply streams"),
        ("Alt+Enter", "Queue a follow-up for after the current reply"),
        ("Ctrl+J · Shift+Enter · Ctrl+Enter", "New line (Shift/Ctrl+Enter need the kitty keyboard protocol)"),
        ("↑ / ↓", "Move in the palette, between editor lines, or through history"),
        ("← / → · Ctrl+B / Ctrl+F", "Move one character"),
        ("Alt+← / Alt+→ · Ctrl+← / Ctrl+→", "Move one word"),
        ("Home / End · Ctrl+A / Ctrl+E", "Start or end of the current line"),
        ("Ctrl+U / Ctrl+K / Ctrl+W", "Delete to line start / to line end / the previous word (line-scoped)"),
        ("Tab", "Complete the palette selection"),
        ("Shift+Tab", "Cycle thinking levels supported by the model"),
        ("Ctrl+L", "Open the model picker"),
        ("Ctrl+P · Ctrl+Shift+P", "Next model / previous model (previous needs the kitty protocol)"),
        ("Ctrl+O", "Expand or collapse tool output"),
        ("Ctrl+T", "Collapse or show thinking"),
        ("PgUp / PgDn · mouse wheel", "Scroll the transcript"),
        ("Ctrl+Home / Ctrl+End · End (empty editor)", "Jump to the top / bottom"),
        ("Esc", "Close the palette; else abort a reply; else clear the draft"),
        ("Ctrl+C", "Clear the draft; abort a reply; quit (asks twice when the session is not being saved)"),
        ("Ctrl+D", "Quit when the editor is empty (no confirmation, see F-39); abort while streaming"),
    ]
    u.append(unit("keyboard", "Keyboard shortcuts", f"""
<p>This is the full map from the key handler ({src('src/state.rs',366)}). <code>/hotkeys</code> prints a shorter version inside chat.</p>
{table(["Keys", "Action"], [[esc(k), esc(v)] for k, v in keys])}
<p>Line mode (see <a href="#line-mode">Line mode</a>) has a smaller set: <kbd>Enter</kbd> sends, <kbd>Ctrl</kbd>+<kbd>C</kbd> clears or (twice) exits, <kbd>Ctrl</kbd>+<kbd>D</kbd> exits on an empty line, <kbd>Ctrl</kbd>+<kbd>A</kbd>/<kbd>E</kbd>/<kbd>U</kbd>/<kbd>K</kbd>/<kbd>W</kbd> edit, <kbd>↑</kbd>/<kbd>↓</kbd> browse history.</p>
{fig("hotkeys", "<code>/hotkeys</code> inside chat.", "Notice listing keyboard shortcuts as printed by /hotkeys.")}
<p>Mouse: only the wheel is used. Because the interface captures the mouse, select text with your terminal's bypass modifier (usually <kbd>Shift</kbd>) held down.</p>
""", topic="interface keyboard", status="runtime"))

    u.append(unit("palette", "The command palette and pickers", f"""
<p>Type <kbd>/</kbd> to open the palette. Arrows select, <kbd>Tab</kbd> completes, <kbd>Enter</kbd> runs. Typing filters the list; a long list shows its position (<code>1/45</code>) in the title. Your saved prompt templates and skills appear after the built-in commands.</p>
{fig("command-palette", "The command palette.", "Command palette listing /help, /model, /login, /omni, /aperture, /btw, /thinking, /tools and /status.")}
<p>Four commands open a searchable picker when run without an argument: <code>/model</code> (or {kb('Ctrl','L')}), <code>/thinking</code>, <code>/login</code> and <code>/resume</code>. Every word you type must appear in an entry.</p>
{fig("model-picker", "<code>/model</code> lists the models of every authenticated provider; the current one is marked.", "Select Model picker listing omni/auto variants and omni/mock-coder marked current.")}
{fig("thinking-picker", "<code>/thinking</code> lists only the levels the current model supports (here a model without reasoning, so only “off”).", "Thinking Level picker with the single entry off, No extended reasoning, current.")}
""", topic="interface", status="runtime"))

    cmds = [
        ("/help, /?", "List commands", ""),
        ("/hotkeys", "Show keyboard shortcuts", ""),
        ("/model [provider/id]", "Open the model picker, or switch directly; the choice is remembered", "#models"),
        ("/thinking [level]", "Pick a reasoning level supported by the model", "#models"),
        ("/login [provider] [key]", "Provider picker, or log in to one; <code>key</code> forces the API-key route", "#login"),
        ("/tools", "List the tools the model can use right now", "#tools"),
        ("/status, /session, /sidebar", "Session id, model, thinking, Planner and Ralph state, context, storage", ""),
        ("/messages", "One line per message in the current context", ""),
        ("/queue", "Show queued steering and follow-up messages", "#composer"),
        ("/steer &lt;text&gt;, /followup &lt;text&gt;", "Steer the running reply / queue the next turn (when idle, both just send)", "#composer"),
        ("/clear, /new", "Start a fresh conversation in the same session file (recoverable)", "#sessions"),
        ("/compact [focus]", "Summarise older turns and keep recent ones", "#compaction"),
        ("/name [text]", "Show or set the session name", "#sessions"),
        ("/sessions", "List saved sessions for this workspace", "#sessions"),
        ("/resume [id]", "Picker, or switch to a saved session", "#sessions"),
        ("/tree, /fork N, /label N name", "List rewind points, rewind, name a point", "#branching"),
        ("/clone", "Copy this session into a new one and switch to it", "#branching"),
        ("/export [path]", "Save as HTML, or Markdown/JSONL by extension", "#export"),
        ("/import &lt;path&gt;", "Copy a session file into this workspace and switch to it", "#export"),
        ("/share [confirm]", "Upload as a secret GitHub gist (asks first)", "#export"),
        ("/prompt, /prompts", "List, save, edit, remove, back up or restore prompt templates", "#prompts"),
        ("/reload, /resources", "Reload / show context files, templates and skills", "#resources"),
        ("/system [text]", "Show or replace the system prompt for this session", "#resources"),
        ("/btw …", "Side questions that do not enter the transcript", "#btw"),
        ("/ralph …", "Long-running loops", "#ralph"),
        ("/planner, /planner-review, /planner-annotate, /planner-last", "Planning mode and browser review", "#planner"),
        ("/omni …", "OmniRoute gateway", "#omniroute"),
        ("/aperture …, /aperture:onboarding, /aperture:settings", "Tailscale Aperture gateway", "#aperture"),
        ("/grok-cli-usage, -accounts, -conv, -imagine, -imagine:tool", "Grok CLI subscription tools", "#grok-cli"),
        ("/skill:&lt;name&gt; [text], /&lt;template&gt; [args]", "Run a skill or prompt template", "#resources"),
        ("/exit, /quit", "Leave chat", ""),
    ]
    rows = [[f"<code>{c}</code>", d + (f' <a href="{l}">More</a>' if l else "")] for c, d, l in cmds]
    u.append(unit("slash-commands", "Slash command reference", f"""
<p>Every command the dispatcher accepts ({src('src/main.rs',2945)}). An unknown command answers <code>unknown command /x; /help lists the available commands</code>.</p>
{table(["Command", "What it does"], rows, "wrapcode")}
{fig("status", "<code>/status</code> output.", "Notice listing Session, Model, Thinking, Planner, Ralph, Context, Activity and Storage.")}
<div class="callout warn"><p>A prompt template with the same name as a built-in command currently takes precedence over it. Check <code>/resources</code> in repositories you did not write (<a href="#F-01">F-01</a>).</p></div>
""", topic="interface reference", status="runtime"))

    # ---------------- models & login ----------------
    u.append(unit("models", "Choosing a model and reasoning level", f"""
<p>Models are named <code>provider/model</code>, for example <code>anthropic/claude-sonnet-5</code>. A bare id works when only one configured provider has it.</p>
<ul>
<li><strong>At startup</strong> the model is chosen from, in order: <code>-m</code>; the <code>GOSHCODER_MODEL</code> environment variable (an unknown value stops chat from starting, <a href="#F-47">F-47</a>); the model you last picked (stored in <code>~/.goshcoder/agent/default-model</code>); the first configured provider in a built-in preference list (OpenAI Codex, Anthropic, Kimi, OpenAI, Azure, DeepSeek, Grok CLI, xAI, Meta, Meta Muse, Google, Vertex, Z.AI, Moonshot, MiniMax, Mistral, Xiaomi, Copilot, OpenCode, Bedrock, Cloudflare); OmniRoute's <code>auto</code>; any model of the first configured provider.</li>
<li><strong>In chat</strong> use <code>/model</code>, {kb('Ctrl','L')}, or {kb('Ctrl','P')} to cycle. A choice made this way (or by a first login) is remembered; <code>-m</code> is not.</li>
<li><strong>Reasoning</strong>: levels are <code>off</code>, <code>minimal</code>, <code>low</code>, <code>medium</code> (default), <code>high</code>, <code>xhigh</code>, <code>max</code>. Each model supports a subset, and the requested level is clamped to it; <code>/thinking</code> and {kb('Shift','Tab')} offer only supported levels. <code>-thinking &lt;level&gt;</code> sets it at startup.</li>
</ul>
<p>List models from the command line:</p>
{code('''goshcoder models              # models of configured providers
goshcoder models anthropic    # every model of one provider''')}
""", topic="providers", status="runtime"))

    u.append(unit("login", "Logging in and adding API keys", f"""
<p>Credentials are kept in <code>~/.goshcoder/agent/auth.json</code> (mode 0600 on Unix), in pi's format. Adding one login never removes another.</p>
<h4>Inside chat</h4>
<p><code>/login</code> opens the picker. Providers with both a subscription login and an API key have two rows; <code>/login &lt;provider&gt; key</code> goes straight to the key prompt. The login runs inside the interface; <kbd>Esc</kbd> or {kb('Ctrl','C')} cancels.</p>
{fig("login-method-choice", "Choosing the Anthropic login method: browser, or copy-code for a machine whose browser cannot reach this one.", "Anthropic login method picker with Browser login (default) and Copy code login (headless).")}
{fig("login-oauth-waiting", "Waiting for the browser. The sign-in address is shown and copied to the clipboard through the terminal (OSC 52); you can also paste the final redirect URL. Parameters are redacted in this image.", "Notice with the claude.ai authorization URL (parameters redacted) and a prompt asking to paste the redirect URL or code.")}
<h4>From the shell</h4>
{code('''goshcoder providers                 # what is configured, and how to set up the rest
goshcoder auth login anthropic      # browser or copy-code
goshcoder auth set openai           # store an API key (prompted, not echoed)
goshcoder auth list                 # stored credentials (names only)
goshcoder auth logout anthropic     # remove a stored credential''')}
<p><code>auth logout</code> removes only the stored entry; a key in an environment variable keeps the provider configured.</p>
<h4>Subscription logins</h4>
{table(["Provider", "Account", "Methods"], [
 ["<code>anthropic</code>", "Claude Pro / Max", "Browser (port 53692, or any free port), copy-code"],
 ["<code>openai-codex</code>", "ChatGPT Plus / Pro", "Browser (port 1455), device code"],
 ["<code>kimi-coding</code>", "Kimi Code", "Device code"],
 ["<code>xai</code>", "Grok subscription against api.x.ai", "Device code (default), browser (port 56121)"],
 ["<code>grok-cli</code>", "X Premium / SuperGrok via the Grok CLI endpoint", "Browser (port 56122, or any free port), device code"],
 ["<code>meta</code>", "Meta account; mints a Model API key", "Device code"],
 ["<code>meta-muse</code>", "Muse Code subscription", "Device code; refused unless Meta confirms an active subscription"],
 ["<code>openrouter</code>", "OpenRouter; mints a permanent key", "Browser (random local port)"],
])}
<p><code>openai-codex</code>, <code>grok-cli</code> and <code>meta-muse</code> take no API key; their key-based counterparts are <code>openai</code>, <code>xai</code> and <code>meta</code>. Browser logins race the local callback against a pasted redirect URL, so a remote browser works too.</p>
{fig("first-launch-login-picker", "The same picker on first launch.", "Add Provider picker.")}
""", topic="providers install", status="runtime"))

    prov = [
        ("anthropic", "ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_OAUTH_TOKEN", "Login or key"),
        ("openai", "OPENAI_API_KEY", "Key"), ("openai-codex", "—", "Login only"),
        ("azure-openai-responses", "AZURE_OPENAI_API_KEY + AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME", "Key"),
        ("google", "GEMINI_API_KEY", "Key"),
        ("google-vertex", "GOOGLE_CLOUD_API_KEY, or project + location + ADC / GOOGLE_OAUTH_ACCESS_TOKEN", "Key or ambient"),
        ("amazon-bedrock", "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, a profile with static keys, or AWS_BEARER_TOKEN_BEDROCK", "Ambient (F-19)"),
        ("mistral", "MISTRAL_API_KEY", "Key"), ("deepseek", "DEEPSEEK_API_KEY", "Key"),
        ("xai", "XAI_API_KEY", "Login or key"), ("grok-cli", "GROK_CLI_OAUTH_TOKEN (bearer, no refresh)", "Login"),
        ("meta", "META_API_KEY", "Login or key"), ("meta-muse", "—", "Login only"),
        ("kimi-coding", "KIMI_API_KEY", "Login or key"), ("moonshotai, moonshotai-cn", "MOONSHOT_API_KEY", "Key"),
        ("zai, zai-coding-cn", "ZAI_API_KEY, ZAI_CODING_CN_API_KEY", "Key"),
        ("minimax, minimax-cn", "MINIMAX_API_KEY, MINIMAX_CN_API_KEY", "Key"),
        ("openrouter", "OPENROUTER_API_KEY", "Login or key"), ("vercel-ai-gateway", "AI_GATEWAY_API_KEY", "Key"),
        ("groq, cerebras, together, fireworks, baseten, nvidia, huggingface", "GROQ_API_KEY, CEREBRAS_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY, BASETEN_API_KEY, NVIDIA_API_KEY, HF_TOKEN", "Key"),
        ("opencode, opencode-go", "OPENCODE_API_KEY", "Key"),
        ("qwen-token-plan(-cn, -individual)", "QWEN_TOKEN_PLAN_API_KEY, QWEN_TOKEN_PLAN_CN_API_KEY", "Key"),
        ("xiaomi, xiaomi-token-plan-ams/-cn/-sgp", "XIAOMI_API_KEY, XIAOMI_TOKEN_PLAN_*_API_KEY", "Key"),
        ("ant-ling", "ANT_LING_API_KEY", "Key"), ("github-copilot", "COPILOT_GITHUB_TOKEN", "Key (bare token)"),
        ("cloudflare-ai-gateway, cloudflare-workers-ai", "CLOUDFLARE_API_KEY + CLOUDFLARE_ACCOUNT_ID (+ CLOUDFLARE_GATEWAY_ID)", "Key — currently broken (F-03)"),
        ("radius", "RADIUS_API_KEY", "Key; no bundled models"),
        ("omni", "OMNIROUTE_API_KEY (optional)", "Gateway, see OmniRoute"),
        ("aperture", "—", "Gateway, see Aperture"),
    ]
    u.append(unit("providers", "Providers and API-key environment variables", f"""
<p>GoshCoder knows 45 providers and 1,256 bundled models across nine wire protocols: <code>openai-completions</code>, <code>openai-responses</code>, <code>openai-codex-responses</code>, <code>azure-openai-responses</code>, <code>anthropic-messages</code>, <code>google-generative-ai</code>, <code>google-vertex</code>, <code>mistral-conversations</code> and <code>bedrock-converse-stream</code>. OmniRoute models that cannot call tools natively use a tenth, prompt-emulated adapter. Requests go straight to each provider over HTTPS; there are no vendor SDKs.</p>
<p>A key stored with <code>auth set</code> wins over an environment variable. A stored OAuth login is used as-is (if its refresh fails you must log in again; environment keys are not used as a fallback).</p>
{table(["Provider id", "Environment variables", "How to authenticate"], [[f"<code>{a}</code>", f"<code>{esc(b)}</code>" if b != "—" else "—", c] for a, b, c in prov], "wrapcode")}
<p>Stored keys and keys in <code>auth.json</code> may reference the environment (<code>$NAME</code>) or a command (<code>!command</code>), as in pi (<a href="#F-58">F-58</a>).</p>
""", topic="providers reference", status="runtime"))

    # ---------------- tools ----------------
    tools = [
        ("read", "path, offset, limit", "Up to 50 KiB per call (within the first ~2 MB of a file); UTF-8 text only"),
        ("write", "path, content", "Atomic; creates parent directories; keeps existing permissions"),
        ("edit", "path, old_text, new_text", "old_text must occur exactly once; files up to 10 MiB"),
        ("ls", "path, limit", "Default 500 entries; output up to 30 KiB"),
        ("grep", "pattern, path, glob, ignoreCase, literal, context, limit", "Built-in regex engine; uses git's file list when available; files over 4 MiB skipped"),
        ("find", "pattern, path, limit", "Glob with ** and {a,b}; default 1,000 results"),
        ("bash", "command", "Runs <code>sh -c</code> in the workspace (F-09); 120 s timeout; first 30 KiB of output kept (F-10); no stdin"),
        ("web_search", "query or queries, numResults, recencyFilter, domainFilter, provider", "See Web search"),
        ("mcp", "server, search, tool, args", "Linux desktop control, only when computer-use-linux is installed"),
        ("image_gen", "prompt, image, aspect_ratio", "Grok Imagine, only with a Grok CLI login"),
        ("ralph_start, ralph_done", "loop fields", "Ralph loops (chat only, on by default)"),
        ("planner_submit_plan", "filePath", "Only in Planner mode"),
        ("aperture_connector_*", "varies", "Only with Aperture connectors enabled"),
    ]
    u.append(unit("tools", "Tools the model can use", f"""
<p>Tools are on by default in chat and <strong>off</strong> in <code>goshcoder run</code> unless you pass <code>-tools</code>. <code>-tools=false</code> turns off the file, shell and web tools in chat, but see the warning below. <code>/tools</code> lists what is active.</p>
{fig("tool-cards-expanded", "Ctrl+O expands every tool card to its full output (up to 2,000 lines).", "Expanded ls tool card listing .git/, README.md, notes.txt and src/, with the hint ctrl+o to collapse.")}
{table(["Tool", "Parameters", "Limits and notes"], [[f"<code>{a}</code>", esc(b), c] for a, b, c in tools], "wrapcode")}
<p>File tools are confined to the workspace (the directory you started in, or <code>-C &lt;dir&gt;</code>): paths outside it and symbolic links pointing out are refused, and writes to the same file are serialised. The checks are made before each operation, so they are not proof against another process swapping directories at the same moment, and hard links are not detected (<a href="#F-28">F-28</a>). <code>bash</code> runs with your privileges and your environment, including API keys (<a href="#F-61">F-61</a>).</p>
<div class="callout danger"><p><strong><code>-tools=false</code> is not a guarantee of read-only chat.</strong> If the Planner is in its planning or executing phase for this repository (possibly set by another window), it adds write/edit or full tools (<a href="#F-02">F-02</a>). Ralph tools also stay available.</p></div>
""", topic="tools security", status="runtime"))

    u.append(unit("compaction", "Long conversations and compaction", f"""
<p>When a conversation approaches the model's context window (within <code>clamp(window/5, 2048, 16384)</code> tokens of it), GoshCoder summarises the older part before the next prompt and keeps roughly the last 20,000 tokens. <code>/compact [focus]</code> does the same on demand; the optional text tells the summariser what to keep.</p>
{fig("compaction", "<code>/compact</code> replaced older turns with a summary and kept the two most recent messages.", "Notice: Context compacted: 441 tokens → summary + 2 recent messages.")}
<p>Compaction cannot be interrupted, and queued messages are dropped when it finishes (a notice says so). If the provider fails while summarising before a prompt, the prompt is not sent (<a href="#F-11">F-11</a>); switch model or wait and retry.</p>
""", topic="agent", status="runtime"))

    u.append(unit("errors-retries", "Errors and automatic retries", f"""
<p>Transient provider failures (HTTP 429 and 5xx, rate limits, overloads, timeouts, dropped streams) are retried up to three times after 2, 4 and 8 seconds. The status bar counts down; <kbd>Esc</kbd> cancels the wait. Other errors appear as <code>× Error</code> with the provider's message; quota and billing errors are not retried.</p>
{fig("retry-countdown", "A provider returning HTTP 500 three times: each failure is shown and the status bar counts down to the next attempt.", "Error cards for status 500, notices Retrying in 2s (attempt 2 of 4) and Retrying in 4s (attempt 3 of 4), and a status bar reading Retrying in 3s · attempt 3/4.")}
{fig("error-400", "A non-retryable error (HTTP 400) ends the turn.", "Error card: provider request failed with status 400.")}
<p>Streams time out after 300 seconds without data, and connections after 30 seconds.</p>
""", topic="agent providers troubleshooting", status="runtime"))

    # ---------------- sessions ----------------
    u.append(unit("sessions", "Sessions: saving, resuming and naming", f"""
<p>Chat records every conversation automatically to <code>~/.goshcoder/agent/sessions/&lt;workspace&gt;/&lt;timestamp&gt;_&lt;id&gt;.jsonl</code> (pi's v3 format, mode 0600). A conversation with no reply is not kept. <code>run</code> records only with <code>-continue</code>, <code>-session</code> or <code>-name</code>.</p>
{table(["To…", "Use"], [
 ["Reopen the latest conversation here", "<code>goshcoder -continue</code>"],
 ["Pick one at startup", "<code>goshcoder -resume</code> (numbered list; type the number)"],
 ["Open a specific one", "<code>goshcoder -session &lt;id, prefix or path&gt;</code> (an unknown id creates a new session, F-34)"],
 ["Not record at all", "<code>goshcoder -no-session</code>"],
 ["Look without writing", "<code>goshcoder -session &lt;id&gt; -read-only</code>"],
 ["Store sessions elsewhere", "<code>-sessions-dir &lt;dir&gt;</code> (e.g. pi's <code>~/.pi/agent/sessions</code>)"],
 ["Switch in chat", "<code>/resume</code> (picker) or <code>/resume &lt;id&gt;</code>; <code>/sessions</code> lists"],
 ["Name it", "<code>/name &lt;text&gt;</code> or <code>-name &lt;text&gt;</code>"],
 ["Start over", "<code>/clear</code> or <code>/new</code> (same file; the old part stays recoverable)"],
], "wrapcode")}
{fig("resume-picker", "<code>goshcoder chat -resume</code>: a numbered list of this workspace's sessions; Enter starts a new one.", "Text list of five saved sessions with numbers, ids, dates, message counts and names, and a prompt to choose a number.", wide=False)}
{fig("resumed-session", "A resumed session: the transcript and sidebar are restored.", "Fullscreen interface showing a resumed session named Line mode demo.")}
<p>Only one process can write a session at a time: a lock file next to it is refreshed every 2 seconds and considered stale after 20. A second window opening the same session gets it read-only.</p>
""", topic="sessions", status="runtime"))

    u.append(unit("branching", "Rewinding, branching and cloning", f"""
<ol class="steps">
<li><code>/tree</code> lists your messages as numbered rewind points, with labels and other branches.</li>
<li><code>/fork N</code> rewinds to just before message N. In fullscreen, message N is put back in the editor so you can change it and press <kbd>Enter</kbd> to take the conversation another way. <code>/fork</code> on a point from another branch jumps back to that branch.</li>
<li><code>/label N name</code> names a point; labels show in <code>/tree</code>. Labels cannot be removed from chat (<a href="#F-35">F-35</a>).</li>
<li><code>/clone</code> copies the current branch into a new session file and switches to it.</li>
</ol>
<p>All of this needs a recorded session and an idle agent.</p>
""", topic="sessions", status="runtime"))

    u.append(unit("export", "Export, import and share", f"""
{code('''/export                       # HTML into the current directory (see warning)
/export ~/notes/spike.md      # Markdown
/export ~/backup/spike.jsonl  # the raw session file

goshcoder sessions export <id> transcript.html
goshcoder sessions export <id> --md notes.md
goshcoder sessions export <id> > copy.jsonl   # JSONL to stdout
goshcoder sessions import copy.jsonl          # adopt into this workspace
/import ~/backup/spike.jsonl                  # same, and switch to it''')}
<ul>
<li><strong>HTML</strong> is a single page with no scripts and a policy that blocks all network loads. It includes Markdown, thinking blocks and collapsible tool cards with full arguments and results, from the current context (after the last clear).</li>
<li><strong>Markdown</strong> includes user and assistant text and tool names, but no tool output or thinking.</li>
<li><strong>JSONL</strong> is the complete, lossless file (every branch and cleared message).</li>
<li>Exports are written with mode 0600 and overwrite existing files. The target directory must exist.</li>
</ul>
{fig("export-html-transcript", "An HTML export opened in a browser.", "Exported HTML transcript titled Due dates spike with session metadata and message cards.")}
<div class="callout warn"><p>Bare <code>/export</code> writes <code>goshcoder-session-&lt;id&gt;.html</code> into the project directory, where it can be committed by accident (<a href="#F-07">F-07</a>). Give it a path outside the repository.</p></div>
<h4>Sharing</h4>
<p><code>/share</code> explains what will be uploaded and waits for <code>/share confirm</code>; <code>goshcoder sessions share &lt;id&gt; --yes</code> does the same from the shell. It uploads the HTML page as a <strong>secret</strong> gist through the GitHub CLI (<code>gh</code> must be logged in). Secret gists are unlisted, not private. Set <code>GOSHCODER_SHARE_VIEWER_URL</code> to print a viewer link.</p>
""", topic="sessions", status="runtime"))

    u.append(unit("sessions-cli", "Managing sessions from the shell", f"""
{code('''goshcoder sessions                      # this workspace (same as: sessions list)
goshcoder sessions list --all            # every workspace
goshcoder sessions show <id> [--full]    # details and one-line previews
goshcoder sessions rm <id>               # delete — no confirmation
goshcoder sessions gc --older-than 30d   # dry run across ALL workspaces
goshcoder sessions gc --older-than 30d --keep-named --yes''')}
<div class="callout danger"><p><code>sessions rm</code> deletes at once, and a prefix that matches nothing here can match a session in another workspace (<a href="#F-08">F-08</a>). <code>gc --yes</code> deletes across all workspaces. There is no trash.</p></div>
<p>The <code>sessions</code> subcommands always use the default session directory; they have no <code>-sessions-dir</code> option (pass a full path instead).</p>
""", topic="sessions", status="runtime"))

    u.append(unit("backup", "Backup and recovery", f"""
<ul>
<li><strong>Everything:</strong> copy <code>~/.goshcoder/agent/</code>. It holds <code>auth.json</code> (secrets — protect the copy), sessions, prompts, skills and gateway settings.</li>
<li><strong>One session:</strong> <code>goshcoder sessions export &lt;id&gt; file.jsonl</code> is lossless; restore with <code>sessions import</code>.</li>
<li><strong>Prompt templates:</strong> <code>goshcoder prompts backup</code> writes <code>goshcoder-prompts-YYYY-MM-DD.tar.gz</code>; <code>goshcoder prompts restore &lt;archive&gt;</code> brings them back, skipping existing names unless <code>--overwrite</code>; <code>--dry-run</code> previews.</li>
<li><strong>After an accidental <code>/clear</code>:</strong> the cleared messages are still in the file. Export the JSONL to keep them, or use <code>/tree</code> and <code>/fork N</code> on the last message before the clear (this restores the conversation up to just before that message). <code>sessions show --full</code> shows only one-line previews.</li>
<li><strong>Damaged files:</strong> a half-written last line is ignored (and trimmed when the session is next opened for writing); other unreadable lines are skipped with a notice. A file with a broken header, or over 256 MiB, cannot be opened and disappears from listings.</li>
<li><strong>Durability:</strong> each entry is appended as it happens and synced to disk at the end of each prompt's run. A reply that was still streaming when the process died is lost; a power cut can lose the last entries.</li>
</ul>
<div class="callout warn"><p>Session files contain everything the agent read and every command's output, with no redaction. Use <code>-no-session</code> for work that should not be written down.</p></div>
""", topic="sessions config", status="source"))

    # ---------------- resources ----------------
    u.append(unit("resources", "Project instructions, system prompt, templates and skills", f"""
<p>GoshCoder reads these at startup and on <code>/reload</code>; <code>/resources</code> shows what was loaded and any warnings.</p>
{table(["Resource", "Where it is looked for", "Effect"], [
 ["<code>AGENTS.md</code> (or <code>AGENTS.override.md</code>, else <code>CLAUDE.md</code>)", "<code>~/.goshcoder/agent/AGENTS.md</code>, then each directory from the repository root (the first ancestor with <code>.git</code>; without one, up to <code>/</code>) down to the workspace — one file per directory", "Added to the system prompt as project context"],
 ["<code>SYSTEM.md</code>", "<code>~/.goshcoder/agent/</code>, then <code>&lt;workspace&gt;/.pi/</code>, then <code>&lt;workspace&gt;/.goshcoder/</code> — first found", "<strong>Replaces</strong> the built-in system prompt (<code>-s</code> overrides it)"],
 ["<code>APPEND_SYSTEM.md</code>", "Same three places — all found", "Appended to the system prompt"],
 ["Prompt templates <code>*.md</code>", "<code>~/.goshcoder/agent/prompts/</code>, <code>&lt;workspace&gt;/.pi/prompts/</code>, <code>&lt;workspace&gt;/.goshcoder/prompts/</code> (first name wins)", "Become <code>/&lt;name&gt;</code> commands"],
 ["Skills (<code>SKILL.md</code>)", "<code>~/.goshcoder/agent/skills</code>, <code>~/.agents/skills</code>, <code>&lt;workspace&gt;/.pi/skills</code>, <code>&lt;workspace&gt;/.goshcoder/skills</code>, and <code>.agents/skills</code> in each ancestor up to the repository root", "Listed to the model; run with <code>/skill:&lt;name&gt;</code>"],
], "wrapcode")}
<p>Templates and <code>SYSTEM.md</code> are looked up in the directory you started in, not the repository root. Symbolic links are skipped, files over 2 MiB are ignored, and project context is wrapped so it cannot impersonate the harness.</p>
{fig("resources-report", "<code>/resources</code> with a project <code>AGENTS.md</code> and a workspace <code>SYSTEM.md</code>; the warning appears only here.", "Resources report listing a custom system file, AGENTS.md, a /review template and a warning that SYSTEM.md replaces the whole system prompt.")}
<div class="callout danger"><p><strong>Repositories you did not write:</strong> a workspace <code>SYSTEM.md</code> replaces the system prompt with no notice at startup (<a href="#F-06">F-06</a>), and a prompt template can take over a built-in command such as <code>/clear</code> (<a href="#F-01">F-01</a>). Run <code>/resources</code> first.</p></div>
<h4>Template syntax</h4>
<p>Optional front matter (<code>---</code>, <code>description:</code>, <code>argument-hint:</code>, <code>---</code>). In the body, <code>$1</code>…<code>$N</code>, <code>$@</code> / <code>$ARGUMENTS</code>, <code>${{1:-default}}</code>, <code>${{@:2}}</code> and <code>$$</code> are replaced; arguments are split like a shell command line.</p>
<p><code>/system</code> shows the current system prompt; <code>/system &lt;text&gt;</code> replaces it for the rest of this session.</p>
""", topic="resources security config", status="runtime"))

    u.append(unit("prompts", "Saving and managing prompt templates", f"""
{code('''/prompt                                   # list templates
/prompt save review Review the diff in $1  # save text as /review (user scope)
/prompt save --project review              # save your last message, in .goshcoder/prompts
/prompt edit review                        # open in $VISUAL or $EDITOR
/prompt rm review [--project]
/prompt backup [path]  ·  /prompt restore <archive> [--overwrite] [--dry-run]

goshcoder prompts list | backup [path] | restore <archive> [--overwrite] [--dry-run]''')}
<p>Names must start with a letter or digit and may not be a built-in command. Archives are gzip tar files with a manifest; on restore, member names are re-validated, non-regular files are refused, and sizes are capped (10,000 entries, 64 MiB, 2 MiB per prompt).</p>
""", topic="resources", status="runtime"))

    # ---------------- extensions ----------------
    u.append(unit("btw", "BTW: side questions", f"""
<p><code>/btw &lt;question&gt;</code> asks the current model a question with a copy of the conversation as context; the answer appears as a card and <em>does not</em> enter the main transcript. The main agent must be idle; <kbd>Esc</kbd> cancels.</p>
{code('''/btw what does notes.txt contain?
/btw                                   # list threads
/btw resume btw-1 and what about src/?
/btw bring btw-1 [latest|all|from:N]   # put the exchange into the composer to send
/btw settings [level] | model provider/id | remember on|off''')}
{fig("btw-card", "A BTW answer card. The test gateway's canned reply is shown.", "Card titled BTW · btw-1 under the /btw command.")}
<p>Settings live in <code>~/.goshcoder/agent/pi-btw.json</code>. Threads are saved with the session (newest 50) and come back with <code>-continue</code>/<code>-resume</code>, but not after an in-chat <code>/resume</code> (<a href="#F-12">F-12</a>). In line mode <code>/btw bring</code> loses the text (<a href="#F-55">F-55</a>).</p>
""", topic="extensions", status="runtime"))

    u.append(unit("ralph", "Ralph: long-running loops", f"""
<p>A Ralph loop repeats a task over many iterations: each iteration the model works, updates <code>.ralph/&lt;name&gt;.md</code>, and calls <code>ralph_done</code> to continue, until it replies <code>&lt;promise&gt;COMPLETE&lt;/promise&gt;</code> or reaches the iteration limit (default 50).</p>
{code('''/ralph start duedates Add due dates to todo items [--max-iterations 20]
/ralph            # status of this session's loop
/ralph list [--archived]
/ralph stop [name]     # ends the loop (it is marked completed)
/ralph resume <name> · /ralph archive <name> · /ralph delete <name>
goshcoder ralph list | start | stop | resume | archive | delete''')}
{fig("ralph-loop", "Iteration 1 of a loop: the injected instructions and the first reply.", "Transcript showing the Ralph loop instructions (iteration 1/50) and a reply.")}
<p>Loop files live in the workspace's <code>.ralph/</code> directory. Ralph is on by default in chat; <code>-ralph=false</code> turns it off. Known problems: a stopped loop cannot be resumed, loops are orphaned when chat restarts, and <code>goshcoder ralph status</code> never shows a loop (<a href="#F-15">F-15</a>, <a href="#F-32">F-32</a>). If <code>.ralph</code> is a symlink every prompt fails; start with <code>-ralph=false</code> (<a href="#F-16">F-16</a>).</p>
""", topic="extensions", status="runtime"))

    u.append(unit("planner", "Planner: plan first, then review in the browser", f"""
<ol class="steps">
<li><strong>Enter planning mode:</strong> <code>/planner</code> (toggle) or start with <code>-planner</code>. The sidebar says <code>planner</code>; <code>bash</code> is removed and <code>write</code>/<code>edit</code> are limited to Markdown files in the workspace.</li>
<li><strong>The model writes a plan</strong> (a Markdown file with <code>- [ ]</code> items) and calls <code>planner_submit_plan</code>.</li>
<li><strong>Review in the browser.</strong> GoshCoder starts a page on <code>http://127.0.0.1:&lt;random port&gt;/</code> and opens it (or prints the URL). Click a line to annotate it, add overall notes, edit the Markdown directly, then press <strong>Approve</strong> or <strong>Feedback</strong>.</li>
<li><strong>Feedback</strong> goes back to the model, which revises and resubmits; the page then shows what changed. <strong>Approve</strong> switches to executing with full tools; the sidebar tracks the checklist as the model reports <code>[DONE:n]</code>.</li>
</ol>
{fig("planner-mode", "<code>/tools</code> in planning mode: no bash, and <code>planner_submit_plan</code> is offered.", "Tool list without bash and with planner_submit_plan.")}
{fig("planner-write-blocked", "A write to <code>notes.txt</code> during planning is refused.", "Failed write tool card: Planner: during planning, writes and edits are limited to markdown files inside the workspace.")}
{fig("planner-annotate-browser", "The review page served by GoshCoder.", "Browser page titled Planner, Annotate README.md, with line-numbered content, Feedback and Approve buttons and an annotations panel.")}
{fig("planner-annotate-with-note", "Clicking line 3 asked for a note; overall notes are typed on the right.", "Planner page with an annotation for line 3 listed in the Annotations panel and text in the overall notes box.")}
{fig("planner-feedback-sent", "After Feedback the page confirms that GoshCoder received the review.", "Page reading Feedback sent. GoshCoder received your review. You can safely close this tab.", wide=False)}
{fig("planner-feedback-in-chat", "After a line annotation and overall note are sent, the feedback arrives in chat as the next prompt.", "Transcript with the Planner review feedback listing the line 3 annotation and overall notes.")}
<h4>Reviewing without a plan</h4>
{code('''/planner-review                 # review the working-tree (or staged) diff
/planner-review https://github.com/o/r/pull/1   # needs gh
/planner-annotate docs/design.md  # a file, folder, or http(s) URL
/planner-last                     # the latest assistant reply''')}
<p>The phase and checklist are stored per repository in <code>~/.goshcoder/agent/planner/</code> and shared by every window on that repository. The page accepts requests only from this machine with a matching Host header. Caveats: the planning gate does not cover desktop control, gateway tools or Ralph (<a href="#F-05">F-05</a>); an executing phase can give tools to a <code>-tools=false</code> session (<a href="#F-02">F-02</a>); the “Select line” button does nothing (<a href="#F-54">F-54</a>).</p>
""", topic="extensions", status="runtime"))

    u.append(unit("omniroute", "OmniRoute gateway", f"""
<p><a href="https://github.com/md-riaz/omniroute-agent-extension">OmniRoute</a> is a self-hosted OpenAI-compatible router. Point GoshCoder at it once and its models appear as <code>omni/…</code>.</p>
{code('''goshcoder omni setup      # prompts for the URL (default http://127.0.0.1:20128) and an optional key
goshcoder omni            # status: health, server, model count
goshcoder omni sync       # re-import /v1/models
goshcoder omni models [search] · omni test <model> · omni dashboard · omni config''')}
<p>The same commands work in chat as <code>/omni …</code>. Seven routing aliases (<code>auto</code>, <code>auto/coding</code>, <code>auto/fast</code>, <code>auto/cheap</code>, <code>auto/offline</code>, <code>auto/smart</code>, <code>auto/lkgp</code>) always come first. <code>OMNIROUTE_URL</code> overrides the stored server; the key lives in <code>auth.json</code> or <code>OMNIROUTE_API_KEY</code>; without a key the public placeholder is used. Settings are in <code>~/.goshcoder/agent/omniroute.json</code>.</p>
<p>Setting <code>"toolCalling": false</code> on a model in that file sends it through the prompt-emulated tool adapter, but the next <code>sync</code> or <code>setup</code> removes the setting (<a href="#F-17">F-17</a>). <code>setup</code> ignores a URL given as an argument (<a href="#F-52">F-52</a>).</p>
<p>This handbook's screenshots were made with OmniRoute pointed at a local test gateway.</p>
""", topic="extensions providers", status="runtime"))

    u.append(unit("aperture", "Tailscale Aperture gateway", f"""
<p><a href="https://tailscale.com/docs/features/aperture">Aperture</a> is an AI gateway on your tailnet. GoshCoder can use it three ways: a dedicated <code>aperture</code> provider with the gateway's models; a proxy that routes existing providers through it; and connector tools exposed by the gateway (MCP).</p>
{code('''goshcoder aperture onboarding        # wizard: URL + health check, capability, providers, recap
goshcoder aperture status | sync | providers | connectors
goshcoder aperture settings [<key> <value>]   # e.g. connectors.enabled enabled
goshcoder aperture pin <tool> · unpin <tool>''')}
<p>In chat: <code>/aperture …</code>, <code>/aperture:onboarding</code>, <code>/aperture:settings</code>. Configuration is in <code>~/.goshcoder/agent/extensions/aperture.json</code>, with a model cache beside it so models load offline. Connectors are off by default and the wizard does not offer them; enable them with <code>settings connectors.enabled enabled</code> and start a new session (<a href="#F-53">F-53</a>).</p>
<p>Not tested here (no gateway available); the description comes from the code.</p>
""", topic="extensions providers", status="source"))

    u.append(unit("grok-cli", "Grok CLI subscription, accounts and Grok Imagine", f"""
<p>The <code>grok-cli</code> provider uses an X Premium or SuperGrok subscription through the endpoint the official Grok CLI uses. It is separate from <code>xai</code> (the API).</p>
{code('''goshcoder auth login grok-cli
goshcoder grok-cli usage                 # weekly usage, tier, reset (UTC)
goshcoder grok-cli accounts [add [label] | login <n> | logout <n> | rename <n> <label> | remove <n> | use <n>]
/grok-cli-usage · /grok-cli-accounts … · /grok-cli-conv [status|rotate]
/grok-cli-imagine <prompt> [--image <path>] [--aspect <ratio>] [--out <path>]
/grok-cli-imagine:tool [on|off|status]   # offer image_gen to the model''')}
<ul>
<li><strong>Several accounts:</strong> account 1 is the normal login; others are stored in <code>~/.goshcoder/agent/grok-cli/accounts.json</code>. When an account's balance is exhausted (HTTP 402) the session moves to the next logged-in account and continues.</li>
<li><strong>Grok Imagine</strong> generates or edits images (PNG, JPEG or WebP up to 400 KiB). Images are saved as <code>&lt;session dir&gt;/&lt;session id&gt;/images/N.jpg</code>; the interface shows the path, not the image.</li>
<li><code>GROK_CLI_OAUTH_TOKEN</code> overrides any login. <code>PI_GROK_CLI_*</code> variables override the endpoint, models and login parameters.</li>
</ul>
<p>Only the logged-out states were exercised (<code>usage</code>, <code>accounts</code>, <code>/grok-cli-conv</code>, <code>/grok-cli-imagine:tool status</code>).</p>
""", topic="extensions providers", status="blocked"))

    u.append(unit("meta", "Meta and Meta Muse Code", f"""
<p><code>meta</code> signs in by device code at auth.meta.com and mints a Model API key (or takes <code>META_API_KEY</code>). <code>meta-muse</code> uses a Muse Code subscription: the key is minted only if Meta confirms an active subscription, is re-minted every 12 hours, and the live model list is cached in <code>~/.goshcoder/agent/extensions/meta-muse-models.json</code> (refreshed after the shell login and at each session start).</p>
{code("goshcoder auth login meta-muse   # then pick meta-muse/muse-spark-1.3")}
<p>Not exercised (no Meta account).</p>
""", topic="extensions providers", status="blocked"))

    u.append(unit("web-search", "Web search", f"""
<p>The <code>web_search</code> tool is enabled with the other tools. It takes one query or several, a result count (default 5), a recency filter and domain filters, and returns cited results. Configure it in <code>~/.goshcoder/agent/web-search.json</code>:</p>
{code('''{
  "provider": "auto",            // auto | openai | exa | kagi
  "kagiApiKey": "$KAGI_API_KEY", // or set KAGI_API_KEY
  "exaApiKey": "$EXA_API_KEY"    // optional
}''', "json")}
<p>In <code>auto</code> mode an OpenAI or ChatGPT (Codex) login is used for plain searches; searches with a recency filter or a different result count, and failed OpenAI searches, go to Exa's public endpoint without a key (<a href="#F-50">F-50</a>).</p>
<p>Not exercised against live search services.</p>
""", topic="tools extensions", status="source"))

    u.append(unit("desktop-control", "Desktop control on Linux", f"""
<p>If the <a href="https://github.com/agent-sh/computer-use-linux">computer-use-linux</a> server is installed (<code>COMPUTER_USE_LINUX_BIN</code>, then <code>PATH</code>, then <code>~/.local/bin</code>), chat registers an <code>mcp</code> tool that lets the model read accessibility trees, take screenshots and send input. GoshCoder also adds the server to <code>~/.goshcoder/agent/mcp.json</code> (<a href="#F-49">F-49</a>).</p>
{code('''npm install -g @agent-sh/computer-use-linux    # or: cargo install computer-use-linux
computer-use-linux doctor''')}
<p>Linux only. Not exercised (no desktop session in the test container).</p>
""", topic="tools extensions", platform="linux", status="source"))

    # ---------------- other modes ----------------
    u.append(unit("line-mode", "Line mode", f"""
<p>When standard input or standard error is not a terminal, or with <code>-fullscreen=false</code>, chat uses a plain line interface: replies stream to standard output, tool activity to standard error, and slash commands work as in fullscreen. This mode is the better choice for screen readers and honours <code>NO_COLOR</code>.</p>
{code('''goshcoder -fullscreen=false
printf 'summarise README.md\\n/exit\\n' | goshcoder chat -no-session''')}
{fig("line-mode", "Line-mode chat with a tool call and <code>/status</code>.", "Plain terminal text: a prompt, a tool call line, the reply, token counts and the /status output.", wide=False)}
<p>In line mode, <kbd>Ctrl</kbd>+<kbd>C</kbd> during a reply aborts it (Unix only); twice at an empty prompt exits. Output is not sanitised for escape sequences (<a href="#F-21">F-21</a>).</p>
""", topic="interface accessibility", status="runtime"))

    u.append(unit("run-mode", "One-shot runs for scripts", f"""
{code('''goshcoder run -m openai/gpt-5.6-terra "summarise the changes in CHANGELOG.md"
goshcoder run -tools -C ~/code/app "run the tests and fix the first failure"
goshcoder run -continue -tools "keep going"     # continue (and record) the last session''')}
<p>The answer goes to standard output; reasoning, tool lines, notices and token usage go to standard error. The exit status is 1 when the turn fails or is aborted. <code>run</code> has no tools unless <code>-tools</code> is given, does not start Ralph or the Planner, and records a session only with <code>-continue</code>, <code>-session</code> or <code>-name</code>. It never asks questions: without an authenticated model it exits with an error.</p>
""", topic="interface tools", status="runtime"))

    flags = [
        ("-m, -model &lt;ref&gt;", "Model as provider/id, or a unique bare id"),
        ("-s, -system &lt;text&gt;", "System prompt (replaces SYSTEM.md and the default)"),
        ("-thinking &lt;level&gt;", "off, minimal, low, medium (default), high, xhigh, max"),
        ("-tools[=false]", "File, shell and web tools (on in chat, off in run)"),
        ("-ralph[=false]", "Ralph loops (on in chat)"),
        ("-planner, -plan", "Start in planning mode (only if the stored phase is idle)"),
        ("-C &lt;dir&gt;", "Workspace directory"),
        ("-continue", "Reopen the latest session with content"),
        ("-resume", "Pick a session at startup (chat only)"),
        ("-session &lt;ref&gt;", "Session id, prefix or path"),
        ("-name &lt;text&gt;", "Session name"),
        ("-no-session", "Do not record"),
        ("-read-only", "Open the selected session without writing"),
        ("-sessions-dir &lt;dir&gt;", "Session storage root"),
        ("-fullscreen[=false]", "Fullscreen interface (default on a terminal)"),
        ("-quiet", "Hide session notices"),
        ("-claude-tui[=false]", "Accepted but has no effect (F-14)"),
    ]
    u.append(unit("flags", "Command-line flags", f"""
<p>Flags work for both <code>goshcoder [chat]</code> and <code>goshcoder run</code>. One or two leading dashes both work, and values can be given as <code>-flag=value</code>. Boolean flags accept <code>true</code>/<code>false</code>, <code>1</code>/<code>0</code>, <code>t</code>/<code>f</code>.</p>
{table(["Flag", "Meaning"], [[f"<code>{a}</code>", b] for a, b in flags])}
<p>Conflicts are rejected: <code>-no-session</code> with <code>-continue</code>/<code>-resume</code>/<code>-session</code>/<code>-name</code>; <code>-continue</code> with <code>-session</code>; <code>-resume</code> with <code>-session</code> or <code>-continue</code>. <code>goshcoder &lt;command&gt; --help</code> prints usage for every subcommand.</p>
""", topic="reference config", status="runtime"))

    # ---------------- reference ----------------
    u.append(unit("files", "Files and directories", f"""
<p>Everything GoshCoder keeps lives in the agent directory: <code>$GOSHCODER_AGENT_DIR</code>, else <code>~/.goshcoder/agent</code> (on Windows <code>%USERPROFILE%\\.goshcoder\\agent</code>).</p>
<pre class="tree">~/.goshcoder/agent/
├── auth.json                 credentials (0600) · auth.json.lock
├── default-model             last model chosen with /model or a login
├── sessions/&lt;--workspace--&gt;/&lt;timestamp&gt;_&lt;id&gt;.jsonl   (+ .lock while open)
├── prompts/*.md              user prompt templates
├── skills/                   user skills
├── AGENTS.md, SYSTEM.md, APPEND_SYSTEM.md   optional global instructions
├── web-search.json           web search settings (hand-written)
├── omniroute.json            OmniRoute gateway and synced models
├── pi-btw.json               BTW settings
├── mcp.json                  desktop-control server entry (Linux)
├── planner/&lt;hash&gt;-&lt;repo&gt;.json      Planner phase per repository
├── extensions/aperture.json, aperture-cache.json, meta-muse-models.json
└── grok-cli/accounts.json, quota-cache.json, config.json</pre>
<p>In the workspace: <code>.ralph/</code> (Ralph loops), and optionally <code>.pi/</code> or <code>.goshcoder/</code> with <code>SYSTEM.md</code>, <code>APPEND_SYSTEM.md</code>, <code>prompts/</code> and <code>skills/</code>. Files GoshCoder creates are 0600 and directories 0700 on Unix; Windows relies on the profile's ACLs. If neither <code>HOME</code> nor <code>USERPROFILE</code> is set, the agent directory becomes relative to the current directory (<a href="#F-48">F-48</a>).</p>
""", topic="config reference", status="runtime"))

    envs = [
        ("GOSHCODER_AGENT_DIR", "Agent directory"),
        ("GOSHCODER_MODEL", "Default model (wins over the remembered one)"),
        ("GOSHCODER_SHARE_VIEWER_URL", "Viewer base URL printed after /share"),
        ("GOSHCODER_OAUTH_CALLBACK_HOST, PI_OAUTH_CALLBACK_HOST", "Loopback bind host for browser logins"),
        ("GOSHCODER_XAI_OAUTH_CLIENT_ID, GOSHCODER_META_OAUTH_CLIENT_ID", "Alternative OAuth client ids"),
        ("KIMI_CODE_OAUTH_HOST, KIMI_OAUTH_HOST", "Kimi login host"),
        ("OMNIROUTE_URL, OMNIROUTE_API_KEY", "OmniRoute server and key"),
        ("GROK_CLI_OAUTH_TOKEN", "Grok CLI bearer token (overrides logins)"),
        ("PI_GROK_CLI_BASE_URL, GROK_CLI_BASE_URL, GOSHCODER_GROK_CLI_BASE_URL", "Grok CLI endpoint"),
        ("PI_GROK_CLI_MODELS, PI_GROK_CLI_VERSION_URL", "Grok CLI model list filter, version pointer"),
        ("PI_GROK_CLI_OAUTH_CLIENT_ID, PI_GROK_CLI_OAUTH_SCOPE, PI_GROK_CLI_CALLBACK_HOST, PI_GROK_CLI_CALLBACK_PORT", "Grok CLI login parameters"),
        ("PI_GROK_CLI_IMAGINE_BASE_URL, PI_GROK_CLI_IMAGINE_MODEL", "Grok Imagine endpoint and model"),
        ("KAGI_API_KEY, EXA_API_KEY", "Web search keys"),
        ("COMPUTER_USE_LINUX_BIN", "Path to the desktop-control server"),
        ("AWS_REGION, AWS_DEFAULT_REGION, AWS_PROFILE, AWS_SHARED_CREDENTIALS_FILE, AWS_CONFIG_FILE, AWS_BEDROCK_SKIP_AUTH, AWS_BEDROCK_FORCE_CACHE, PI_CACHE_RETENTION", "Bedrock"),
        ("AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME_MAP", "Azure endpoint details"),
        ("GOOGLE_CLOUD_PROJECT, GCLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_OAUTH_ACCESS_TOKEN", "Vertex AI"),
        ("VISUAL, EDITOR", "Editor for /prompt edit"),
        ("NO_COLOR", "Plain output in line mode and run (fullscreen ignores it)"),
        ("GOSHCODER_INSTALL_DIR, GOSHCODER_VERSION, GOSHCODER_FROM_SOURCE", "Installers"),
    ]
    u.append(unit("environment", "Environment variables", f"""
<p>Provider API-key variables are listed under <a href="#providers">Providers</a>. Everything else:</p>
{table(["Variable", "Effect"], [[f"<code>{esc(a)}</code>", b] for a, b in envs], "wrapcode")}
""", topic="config reference", status="source"))

    u.append(unit("accessibility", "Accessibility", f"""
<ul>
<li>Fullscreen uses fixed colours on a near-black background and ignores <code>NO_COLOR</code>; most text has good contrast (body text about 15:1, muted text about 5.8:1), but the status-bar key hints are about 2.6:1 (<a href="#F-40">F-40</a>).</li>
<li>Fullscreen redraws about ten times a second while busy and uses the alternate screen, which many screen readers handle poorly. <strong>Line mode</strong> (<code>-fullscreen=false</code>) is plain text and honours <code>NO_COLOR</code>.</li>
<li>Everything is reachable from the keyboard; the mouse is only used for scrolling.</li>
<li>The Planner review page is a normal web page with buttons, a text area and a light/dark theme toggle.</li>
</ul>
""", topic="accessibility interface", status="source"))

    u.append(unit("platforms", "Platform differences", f"""
{table(["Area", "Linux / macOS", "Windows"], [
 ["Signals", "SIGTERM/SIGHUP end chat cleanly (exit 128+n); verified on Linux", "No handling; closing the console skips the session close"],
 ["File permissions", "0600 files, 0700 directories", "Inherited ACLs"],
 ["bash tool", "<code>sh -c</code>", "<code>bash.exe</code> from PATH (not the WSL launcher), else <code>cmd.exe</code>"],
 ["Typing", "All keys", "AltGr characters dropped (F-27)"],
 ["Line-mode Ctrl+C during a reply", "Aborts the reply", "Ends the process (F-27)"],
 ["Desktop control", "Linux only", "—"],
 ["Release build", "musl static (Linux), native (macOS)", "x64 (ARM64 via emulation)"],
])}
<p>Only Linux was run for this handbook; macOS and Windows rows come from the code.</p>
""", topic="install interface", platform="linux macos windows", status="source"))

    return chapter("manual", "User manual", '<p class="lead">Everything the interface and command line can do, with the exact labels and commands.</p>', "\n".join(u))
