"""Helpers, styles and scripts for the GoshCoder handbook generator.

The generated page is a single self-contained HTML file. Everything here is
inlined into it; nothing is fetched at runtime, so it opens from disk."""

import html as _html
import re

META = {
    "app": "GoshCoder",
    "version": "0.6.0",
    "describe": "v0.6.0-27-g3870a05",
    "revision": "3870a05",
    "revision_full": "3870a054274c08c113651b614006aa9ee175cc49",
    "branch": "claude/brave-babbage-rkr6ob",
    "audit_date": "2026-10-08",
    "toolchain": "rustc 1.97.0 / cargo 1.97.0",
}

REPO_ROOT = "../../"  # handbook lives in docs/handbook/


def esc(text):
    return _html.escape(text, quote=True)


def src(path, line=None, label=None):
    """A link to a repository-relative source path, optionally with a line."""
    shown = label or (f"{path}:{line}" if line else path)
    return f'<a class="src" href="{REPO_ROOT}{esc(path)}"><code>{esc(shown)}</code></a>'


def code(text, lang="sh"):
    return f'<pre class="code" data-lang="{lang}"><code>{esc(text.strip(chr(10)))}</code></pre>'


STATUS_LABEL = {
    "runtime": "Verified at runtime",
    "source": "Source inspection",
    "inferred": "Inferred",
    "blocked": "Blocked / unverified",
    "na": "Not applicable",
    "docs": "Documentation",
}


def badge(status):
    return f'<span class="vbadge v-{status}" title="{STATUS_LABEL[status]}">{STATUS_LABEL[status]}</span>'


def sev_badge(sev):
    return f'<span class="sev sev-{sev.lower()}">{sev}</span>'


def fig(name, caption, alt, wide=True):
    path = f"screenshots/{name}.png"
    cls = "shot wide" if wide else "shot"
    return f"""<figure class="{cls}">
<a class="thumb" href="{path}" data-full="{path}" data-caption="{esc(caption)}"><img src="{path}" alt="{esc(alt)}" loading="lazy"></a>
<figcaption>{caption}</figcaption>
</figure>"""


def unit(uid, title, body, audience="user", topic="general", platform="all", status="runtime", level=3, tag="section"):
    """A searchable, filterable block of content with a stable anchor."""
    h = f"h{level}"
    return f"""<{tag} class="unit" id="{uid}" data-audience="{audience}" data-topic="{topic}" data-platform="{platform}" data-status="{status}">
<{h}><a class="anchor" href="#{uid}" aria-label="Link to this section">#</a>{title} {badge(status)}</{h}>
{body}
</{tag}>"""


def chapter(cid, title, intro, units_html):
    return f"""<section class="chapter" id="{cid}" aria-labelledby="{cid}-h">
<h2 id="{cid}-h"><a class="anchor" href="#{cid}" aria-label="Link to this chapter">#</a>{title}</h2>
{intro}
{units_html}
</section>"""


def table(headers, rows, cls=""):
    head = "".join(f"<th scope=\"col\">{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


CSS = r"""
:root{
  --bg:#f7f6f2;--surface:#ffffff;--surface-2:#f0eee8;--border:#dcd8cf;--text:#1d2125;--muted:#565c63;
  --accent:#0b6e79;--accent-2:#9a5b13;--accent-bg:#e3f1f2;--link:#0a5f8a;--code-bg:#f1efe9;--code-text:#22272b;
  --mark:#ffe38a;--focus:#1a73e8;--shadow:0 1px 2px rgba(0,0,0,.06),0 4px 14px rgba(0,0,0,.05);
  --sev-critical:#8f1d1d;--sev-high:#b4321f;--sev-medium:#a1580c;--sev-low:#3c6e1f;--sev-info:#4a5866;
  --v-runtime:#1e6b3a;--v-source:#255e8f;--v-inferred:#7a5a0a;--v-blocked:#8a2b2b;--v-na:#5a6168;--v-docs:#5b3f8f;
  --term:#0a0a0a;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans",sans-serif;
  --mono:ui-monospace,"SFMono-Regular","SF Mono",Menlo,Consolas,"DejaVu Sans Mono","Liberation Mono",monospace;
  color-scheme:light;
}
:root[data-theme="dark"]{
  --bg:#121417;--surface:#1a1d21;--surface-2:#22262b;--border:#343a41;--text:#e6e8ea;--muted:#a3abb4;
  --accent:#5cc3cc;--accent-2:#e3a65a;--accent-bg:#17363a;--link:#7cc4ee;--code-bg:#0f1114;--code-text:#dfe3e6;
  --mark:#6b5a12;--focus:#8ab4f8;--shadow:0 1px 2px rgba(0,0,0,.4);
  --sev-critical:#ff8a8a;--sev-high:#ff9b85;--sev-medium:#f0b166;--sev-low:#9fd47c;--sev-info:#b4c0cc;
  --v-runtime:#79d29a;--v-source:#8cc2f0;--v-inferred:#e8c36a;--v-blocked:#f29a9a;--v-na:#b0b7be;--v-docs:#c7b0f2;
  color-scheme:dark;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#121417;--surface:#1a1d21;--surface-2:#22262b;--border:#343a41;--text:#e6e8ea;--muted:#a3abb4;
    --accent:#5cc3cc;--accent-2:#e3a65a;--accent-bg:#17363a;--link:#7cc4ee;--code-bg:#0f1114;--code-text:#dfe3e6;
    --mark:#6b5a12;--focus:#8ab4f8;--shadow:0 1px 2px rgba(0,0,0,.4);
    --sev-critical:#ff8a8a;--sev-high:#ff9b85;--sev-medium:#f0b166;--sev-low:#9fd47c;--sev-info:#b4c0cc;
    --v-runtime:#79d29a;--v-source:#8cc2f0;--v-inferred:#e8c36a;--v-blocked:#f29a9a;--v-na:#b0b7be;--v-docs:#c7b0f2;
    color-scheme:dark;
  }
}
*{box-sizing:border-box}
html{scroll-padding-top:72px}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 var(--sans);-webkit-text-size-adjust:100%}
a{color:var(--link)}
a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid var(--focus);outline-offset:2px;border-radius:4px}
code,pre,kbd{font-family:var(--mono);font-size:.9em}
:not(pre)>code{background:var(--code-bg);padding:.08em .35em;border-radius:4px;word-break:break-word}
kbd{border:1px solid var(--border);border-bottom-width:2px;border-radius:4px;padding:0 .35em;background:var(--surface);white-space:nowrap}
.skip{position:absolute;left:-999px;top:0;background:var(--surface);padding:.5rem 1rem;z-index:100}
.skip:focus{left:8px;top:8px}

/* header */
.topbar{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:.75rem;padding:.55rem 1rem;background:var(--surface);border-bottom:1px solid var(--border);box-shadow:var(--shadow)}
.brand{font-weight:700;letter-spacing:.02em;white-space:nowrap}
.brand .g{color:var(--accent-2)} .brand .c{color:var(--accent)}
.meta-pill{font-size:.78rem;color:var(--muted);border:1px solid var(--border);border-radius:999px;padding:.1rem .6rem;white-space:nowrap}
.topbar .spacer{flex:1}
.btn{font:inherit;font-size:.88rem;color:var(--text);background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:.35rem .7rem;cursor:pointer}
.btn:hover{border-color:var(--accent)}
.btn[aria-pressed="true"]{background:var(--accent-bg);border-color:var(--accent)}
#nav-toggle{display:none}

/* layout */
.layout{display:grid;grid-template-columns:270px minmax(0,1fr);max-width:1400px;margin:0 auto}
nav.side{position:sticky;top:52px;height:calc(100vh - 52px);overflow:auto;padding:1rem .75rem 2rem 1rem;border-right:1px solid var(--border)}
nav.side ol{list-style:none;margin:0;padding:0}
nav.side li{margin:0}
nav.side a{display:flex;justify-content:space-between;gap:.5rem;padding:.28rem .5rem;border-radius:6px;color:var(--text);text-decoration:none;font-size:.93rem}
nav.side a:hover{background:var(--surface-2)}
nav.side a[aria-current="true"]{background:var(--accent-bg);color:var(--accent);font-weight:600}
nav.side ol ol a{font-size:.84rem;padding-left:1.4rem;color:var(--muted)}
nav.side .count{font-size:.75rem;color:var(--muted);background:var(--surface-2);border-radius:999px;padding:0 .45rem;min-width:1.6rem;text-align:center}
nav.side .count[hidden]{display:none}
nav.side a.dim{opacity:.45}
main{padding:1.25rem clamp(1rem,3vw,2.5rem) 4rem;min-width:0}
.content{max-width:960px}

/* filters */
.filters{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:.85rem 1rem;margin:0 0 1.25rem;box-shadow:var(--shadow)}
.filters .row{display:flex;flex-wrap:wrap;gap:.6rem .8rem;align-items:flex-end}
.filters label{display:flex;flex-direction:column;font-size:.78rem;color:var(--muted);gap:.2rem}
.filters input[type=search]{min-width:min(100%,320px);flex:1}
.filters .grow{flex:1 1 260px}
.filters input,.filters select{font:inherit;font-size:.92rem;color:var(--text);background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:.38rem .55rem}
.filters input{width:100%}
.status-line{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin-top:.65rem;font-size:.9rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem}
.chip{font-size:.8rem;border:1px solid var(--accent);color:var(--accent);background:var(--accent-bg);border-radius:999px;padding:.05rem .55rem;cursor:pointer}
.chip::after{content:" ×"}
#results{margin-top:.75rem;border-top:1px solid var(--border);padding-top:.6rem;max-height:300px;overflow:auto}
#results[hidden]{display:none}
#results ol{margin:0;padding-left:1.2rem;font-size:.9rem}
#results li{margin:.15rem 0}
#results .where{color:var(--muted);font-size:.8rem}
#noresults{border:1px dashed var(--border);border-radius:10px;padding:1rem;margin:1rem 0;background:var(--surface)}
#noresults[hidden]{display:none}
.hidden-by-filter{display:none!important}
mark{background:var(--mark);color:inherit;border-radius:2px}

/* content */
h1{font-size:2rem;line-height:1.2;margin:.2rem 0 .6rem}
h2{font-size:1.55rem;margin:2.6rem 0 .6rem;padding-top:.6rem;border-top:2px solid var(--border)}
h3{font-size:1.18rem;margin:1.8rem 0 .5rem;display:flex;flex-wrap:wrap;align-items:center;gap:.5rem}
h4{font-size:1rem;margin:1.2rem 0 .3rem}
.anchor{opacity:0;text-decoration:none;color:var(--muted);margin-left:-1.1em;width:1.1em;display:inline-block}
h2:hover .anchor,h3:hover .anchor,.anchor:focus{opacity:1}
.lead{font-size:1.08rem;color:var(--muted)}
.unit{scroll-margin-top:72px}
.callout{border-left:4px solid var(--accent);background:var(--accent-bg);padding:.6rem .9rem;border-radius:0 8px 8px 0;margin:1rem 0}
.callout.warn{border-color:var(--sev-medium);background:color-mix(in srgb,var(--sev-medium) 12%,var(--surface))}
.callout.danger{border-color:var(--sev-high);background:color-mix(in srgb,var(--sev-high) 12%,var(--surface))}
.callout p{margin:.3rem 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.8rem}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.8rem 1rem}
.card h4{margin-top:0}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem;font-size:.93rem}
.kv dt{color:var(--muted)}
.kv dd{margin:0}
ol.steps{padding-left:1.3rem} ol.steps li{margin:.3rem 0}

.vbadge,.sev{font-size:.7rem;font-weight:600;letter-spacing:.02em;border-radius:999px;padding:.08rem .5rem;border:1px solid currentColor;white-space:nowrap;vertical-align:middle}
.v-runtime{color:var(--v-runtime)}.v-source{color:var(--v-source)}.v-inferred{color:var(--v-inferred)}.v-blocked{color:var(--v-blocked)}.v-na{color:var(--v-na)}.v-docs{color:var(--v-docs)}
.sev-critical{color:var(--sev-critical)}.sev-high{color:var(--sev-high)}.sev-medium{color:var(--sev-medium)}.sev-low{color:var(--sev-low)}.sev-info{color:var(--sev-info)}

pre.code{position:relative;background:var(--code-bg);color:var(--code-text);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;overflow:auto;line-height:1.45}
pre.code .copy{position:absolute;top:.35rem;right:.35rem;font:600 .72rem var(--sans);padding:.15rem .5rem;border-radius:6px;border:1px solid var(--border);background:var(--surface);color:var(--text);cursor:pointer;opacity:.85}
pre.code .copy:hover{opacity:1;border-color:var(--accent)}
pre.tree{background:var(--code-bg);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;overflow:auto;font-size:.86rem;line-height:1.45}

.tablewrap{overflow-x:auto;margin:.8rem 0;border:1px solid var(--border);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--surface)}
th,td{text-align:left;vertical-align:top;padding:.45rem .6rem;border-bottom:1px solid var(--border)}
th{background:var(--surface-2);font-weight:600;position:sticky;top:0}
tbody tr:last-child td{border-bottom:0}
td code{white-space:nowrap}
table.wrapcode td code{white-space:normal}

figure.shot{margin:1rem 0;background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.6rem}
figure.shot a.thumb{display:block;cursor:zoom-in;border-radius:6px;overflow:hidden;background:var(--term)}
figure.shot img{display:block;width:100%;height:auto}
figure.shot:not(.wide){max-width:520px}
figcaption{font-size:.86rem;color:var(--muted);margin-top:.45rem}
.callouts{counter-reset:co;padding-left:0;list-style:none}
.callouts li{counter-increment:co;position:relative;padding-left:2rem;margin:.3rem 0}
.callouts li::before{content:counter(co);position:absolute;left:0;top:.1rem;width:1.45rem;height:1.45rem;border-radius:50%;background:var(--accent);color:var(--surface);font:700 .78rem/1.45rem var(--sans);text-align:center}

details.finding{background:var(--surface);border:1px solid var(--border);border-radius:10px;margin:.6rem 0;padding:0}
details.finding>summary{list-style:none;cursor:pointer;padding:.65rem .9rem;display:flex;flex-wrap:wrap;align-items:center;gap:.45rem .6rem}
details.finding>summary::-webkit-details-marker{display:none}
details.finding>summary::before{content:"▸";color:var(--muted);transition:transform .15s}
details.finding[open]>summary::before{transform:rotate(90deg)}
details.finding .fid{font:600 .82rem var(--mono);color:var(--muted)}
details.finding .ftitle{font-weight:600;flex:1 1 320px}
details.finding .fbody{padding:0 1rem .9rem 1rem;border-top:1px solid var(--border)}
details.finding dl{display:grid;grid-template-columns:max-content 1fr;gap:.35rem .9rem;margin:.75rem 0 0}
details.finding dt{font-weight:600;color:var(--muted);font-size:.86rem}
details.finding dd{margin:0}
details.more{border:1px solid var(--border);border-radius:8px;padding:.4rem .8rem;margin:.8rem 0;background:var(--surface)}
details.more>summary{cursor:pointer;font-weight:600}
.sevbar{display:flex;flex-wrap:wrap;gap:.5rem;margin:.6rem 0}
.sevbar .card{padding:.4rem .8rem;min-width:110px}
.sevbar .n{font-size:1.4rem;font-weight:700}

svg.diagram{width:100%;height:auto;display:block;margin:1rem 0;color:var(--text)}
svg.diagram .box{fill:var(--surface);stroke:var(--border);stroke-width:1.5}
svg.diagram .box.accent{stroke:var(--accent);stroke-width:2}
svg.diagram .box.warm{stroke:var(--accent-2);stroke-width:2}
svg.diagram .lane{fill:var(--surface-2);stroke:none}
svg.diagram text{fill:var(--text);font:13px var(--sans)}
svg.diagram text.small{font-size:11px;fill:var(--muted)}
svg.diagram text.halo{paint-order:stroke;stroke:var(--bg);stroke-width:5px;stroke-linejoin:round}
svg.diagram text.mono{font-family:var(--mono);font-size:11.5px}
svg.diagram text.title{font-weight:700;font-size:13.5px}
svg.diagram .arrow{stroke:var(--muted);stroke-width:1.6;fill:none}
svg.diagram .arrow.acc{stroke:var(--accent)}
svg.diagram .head{fill:var(--muted)} svg.diagram .head.acc{fill:var(--accent)}

dialog#lightbox{border:0;padding:0;background:transparent;max-width:96vw;max-height:94vh}
dialog#lightbox::backdrop{background:rgba(0,0,0,.78)}
dialog#lightbox .lb{background:var(--surface);border-radius:10px;padding:.6rem;display:flex;flex-direction:column;gap:.4rem;max-height:94vh}
dialog#lightbox img{max-width:calc(96vw - 1.2rem);max-height:calc(94vh - 5rem);object-fit:contain;background:var(--term);border-radius:6px}
dialog#lightbox .lbbar{display:flex;gap:.6rem;align-items:center;justify-content:space-between}
dialog#lightbox p{margin:0;font-size:.9rem;color:var(--muted)}

footer.foot{border-top:1px solid var(--border);margin-top:3rem;padding-top:1rem;font-size:.85rem;color:var(--muted)}
.noscript{border:1px solid var(--border);background:var(--surface);padding:.6rem 1rem;border-radius:8px;margin-bottom:1rem;font-size:.9rem}

@media (max-width: 960px){
  .layout{grid-template-columns:1fr}
  #nav-toggle{display:inline-block}
  nav.side{position:fixed;top:52px;left:0;bottom:0;width:min(86vw,320px);height:auto;background:var(--surface);z-index:25;transform:translateX(-105%);transition:transform .18s;box-shadow:var(--shadow)}
  nav.side.open{transform:none}
  .meta-pill.opt{display:none}
  nav.side ol ol{display:none}
  nav.side a{padding:.5rem .6rem}
}
@media (max-width: 560px){
  body{font-size:15px}
  .topbar{gap:.45rem;padding:.5rem .6rem}
  .brand{font-size:.95rem}
  .meta-pill,#print-btn{display:none}
  .brand .hb{display:none}
  h1{font-size:1.55rem} h2{font-size:1.3rem}
  details.finding dl,.kv{grid-template-columns:1fr}
  .filters .row{display:grid;grid-template-columns:1fr 1fr;gap:.5rem .6rem}
  .filters .row label.grow{grid-column:1/-1}
  .filters select{width:100%;min-width:0}
  .filters{padding:.7rem .75rem}
}
@page{margin:16mm 14mm}
@media print{
  .topbar,nav.side,.filters,#results,#noresults,.copy,.anchor,dialog{display:none!important}
  .layout{display:block} main{padding:0} .content{max-width:none}
  body{background:#fff;color:#000;font-size:11pt}
  .hidden-by-filter{display:block!important}
  a{color:#000} a.src::after{content:""}
  h2{page-break-before:always;border-top:0}
  h2,h3{page-break-after:avoid}
  figure.shot,pre,table,details.finding{page-break-inside:avoid}
  figure.shot img{max-height:9cm;width:auto;max-width:100%}
  details.finding{border:1px solid #999}
}
@media (prefers-reduced-motion: reduce){*{transition:none!important;scroll-behavior:auto!important}}
"""

JS = r"""
(function(){
  'use strict';
  var doc = document, root = doc.documentElement;
  root.classList.add('js');

  /* ---------- theme ---------- */
  var THEME_KEY = 'goshcoder-handbook-theme';
  function storedTheme(){ try { return localStorage.getItem(THEME_KEY); } catch(e){ return null; } }
  function storeTheme(v){ try { if (v) localStorage.setItem(THEME_KEY, v); else localStorage.removeItem(THEME_KEY); } catch(e){} }
  var themeBtn = doc.getElementById('theme-toggle');
  function applyTheme(v){
    if (v === 'light' || v === 'dark') root.setAttribute('data-theme', v); else root.removeAttribute('data-theme');
    var shown = v || 'auto';
    themeBtn.textContent = 'Theme: ' + shown;
    themeBtn.setAttribute('aria-label', 'Colour theme: ' + shown + '. Activate to change.');
  }
  applyTheme(storedTheme());
  themeBtn.addEventListener('click', function(){
    var cur = storedTheme(); var next = cur === null ? 'light' : cur === 'light' ? 'dark' : null;
    storeTheme(next); applyTheme(next);
  });

  /* ---------- mobile nav ---------- */
  var nav = doc.getElementById('side'), navBtn = doc.getElementById('nav-toggle');
  navBtn.addEventListener('click', function(){
    var open = nav.classList.toggle('open'); navBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) { var a = nav.querySelector('a'); if (a) a.focus(); }
  });
  nav.addEventListener('click', function(e){ if (e.target.closest('a') && nav.classList.contains('open')) { nav.classList.remove('open'); navBtn.setAttribute('aria-expanded','false'); } });
  doc.addEventListener('keydown', function(e){ if (e.key === 'Escape' && nav.classList.contains('open')) { nav.classList.remove('open'); navBtn.setAttribute('aria-expanded','false'); navBtn.focus(); } });

  /* ---------- copy buttons ---------- */
  function copyText(text, done){
    function fallback(){
      var ta = doc.createElement('textarea'); ta.value = text; ta.setAttribute('readonly',''); ta.style.position='fixed'; ta.style.opacity='0';
      doc.body.appendChild(ta); ta.select(); var ok = false; try { ok = doc.execCommand('copy'); } catch(e){} doc.body.removeChild(ta); done(ok);
    }
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(function(){ done(true); }, fallback); else fallback();
  }
  Array.prototype.forEach.call(doc.querySelectorAll('pre.code'), function(pre){
    var b = doc.createElement('button'); b.type = 'button'; b.className = 'copy'; b.textContent = 'Copy';
    b.setAttribute('aria-label', 'Copy this code to the clipboard');
    b.addEventListener('click', function(){
      copyText(pre.querySelector('code').innerText, function(ok){ b.textContent = ok ? 'Copied' : 'Copy failed'; setTimeout(function(){ b.textContent = 'Copy'; }, 1600); });
    });
    pre.appendChild(b);
  });

  /* ---------- lightbox ---------- */
  var lb = doc.getElementById('lightbox'), lbImg = doc.getElementById('lb-img'), lbCap = doc.getElementById('lb-cap'), lbLink = doc.getElementById('lb-open'), lastThumb = null;
  doc.addEventListener('click', function(e){
    var a = e.target.closest('a.thumb'); if (!a || !lb.showModal) return;
    e.preventDefault(); lastThumb = a;
    lbImg.src = a.getAttribute('data-full'); lbImg.alt = a.querySelector('img').alt;
    lbCap.textContent = a.getAttribute('data-caption'); lbLink.href = a.getAttribute('data-full');
    lb.showModal(); doc.getElementById('lb-close').focus();
  });
  doc.getElementById('lb-close').addEventListener('click', function(){ lb.close(); });
  lb.addEventListener('click', function(e){ if (e.target === lb) lb.close(); });
  lb.addEventListener('close', function(){ if (lastThumb) lastThumb.focus(); });

  /* ---------- search and filters ---------- */
  var units = Array.prototype.slice.call(doc.querySelectorAll('.unit'));
  var chapters = Array.prototype.slice.call(doc.querySelectorAll('section.chapter'));
  units.forEach(function(u){ u._text = (u.textContent || '').toLowerCase().replace(/\s+/g,' '); });
  var q = doc.getElementById('q'), fAud = doc.getElementById('f-audience'), fTopic = doc.getElementById('f-topic'),
      fPlat = doc.getElementById('f-platform'), fStat = doc.getElementById('f-status'), fSev = doc.getElementById('f-severity');
  var countEl = doc.getElementById('count'), chipsEl = doc.getElementById('chips'), resetBtn = doc.getElementById('reset'),
      resultsEl = doc.getElementById('results'), noRes = doc.getElementById('noresults');
  var selects = [fAud, fTopic, fPlat, fStat, fSev];

  function clearMarks(){
    Array.prototype.forEach.call(doc.querySelectorAll('mark.hit'), function(m){ var p = m.parentNode; p.replaceChild(doc.createTextNode(m.textContent), m); p.normalize(); });
  }
  function markTerms(el, terms){
    if (!terms.length) return;
    var re = new RegExp('(' + terms.map(function(t){ return t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'); }).join('|') + ')', 'gi');
    var walker = doc.createTreeWalker(el, NodeFilter.SHOW_TEXT, { acceptNode: function(n){
      if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      var p = n.parentNode; if (p.closest('script,style,button,mark,.vbadge,.sev,svg')) return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT; } });
    var nodes = [], n, budget = 400; while ((n = walker.nextNode()) && budget--) nodes.push(n);
    nodes.forEach(function(node){
      var s = node.nodeValue; re.lastIndex = 0; if (!re.test(s)) return; re.lastIndex = 0;
      var frag = doc.createDocumentFragment(), last = 0, m;
      while ((m = re.exec(s))) { if (m.index > last) frag.appendChild(doc.createTextNode(s.slice(last, m.index)));
        var mk = doc.createElement('mark'); mk.className = 'hit'; mk.textContent = m[0]; frag.appendChild(mk); last = m.index + m[0].length; }
      if (last < s.length) frag.appendChild(doc.createTextNode(s.slice(last)));
      node.parentNode.replaceChild(frag, node);
    });
  }
  function matchesAttr(u, attr, val){
    if (!val) return true;
    var have = (u.getAttribute(attr) || '').split(/\s+/);
    if (attr === 'data-platform' && have.indexOf('all') !== -1) return true;
    return have.indexOf(val) !== -1;
  }
  function titleOf(u){ var h = u.querySelector('h3,h4,.ftitle,summary'); return h ? h.textContent.replace(/^#/,'').replace(/(Verified at runtime|Source inspection|Inferred|Blocked \/ unverified|Not applicable|Documentation)\s*$/,'').trim() : u.id; }
  function chapterTitle(u){ var c = u.closest('section.chapter'); return c ? c.querySelector('h2').textContent.replace(/^#/,'').trim() : ''; }

  var timer = null;
  function apply(){
    clearMarks();
    var terms = q.value.toLowerCase().split(/\s+/).filter(function(t){ return t.length > 1 || /\w/.test(t); });
    var sev = fSev.value, active = terms.length || selects.some(function(s){ return s.value; });
    var shown = 0, hits = [];
    units.forEach(function(u){
      var ok = terms.every(function(t){ return u._text.indexOf(t) !== -1; })
        && matchesAttr(u,'data-audience',fAud.value) && matchesAttr(u,'data-topic',fTopic.value)
        && matchesAttr(u,'data-platform',fPlat.value) && matchesAttr(u,'data-status',fStat.value)
        && (!sev || u.getAttribute('data-severity') === sev);
      // a container unit stays visible when one of its nested units matches
      u._ok = ok;
    });
    units.forEach(function(u){
      var keep = u._ok || (active && u.querySelector('.unit') && Array.prototype.some.call(u.querySelectorAll('.unit'), function(c){ return c._ok; }));
      u.classList.toggle('hidden-by-filter', active && !keep);
      if (active && u._ok) { shown++; hits.push(u); }
      if (active && keep && u.tagName === 'DETAILS' && terms.length) u.open = true;
    });
    chapters.forEach(function(c){
      var any = !active || c.querySelector('.unit:not(.hidden-by-filter)') !== null;
      c.classList.toggle('hidden-by-filter', !any);
      var link = nav.querySelector('a[href="#' + c.id + '"]'); if (!link) return;
      var cnt = link.querySelector('.count');
      if (active) { var k = hits.filter(function(h){ return c.contains(h); }).length; cnt.textContent = k; cnt.hidden = false; link.classList.toggle('dim', k === 0); }
      else { cnt.hidden = true; link.classList.remove('dim'); }
    });
    if (active && terms.length) hits.slice(0, 80).forEach(function(u){ markTerms(u, terms); });
    // nested <details> that hold a hit open up
    if (terms.length) Array.prototype.forEach.call(doc.querySelectorAll('mark.hit'), function(m){ var d = m.parentNode.closest('details'); while (d) { d.open = true; d = d.parentNode.closest('details'); } });

    countEl.textContent = active ? (shown + ' matching item' + (shown === 1 ? '' : 's') + ' of ' + units.length) : (units.length + ' items · type to search');
    noRes.hidden = !(active && shown === 0);
    resetBtn.disabled = !active;
    // chips
    chipsEl.innerHTML = '';
    if (q.value.trim()) addChip('Search: “' + q.value.trim() + '”', function(){ q.value=''; });
    selects.forEach(function(s){ if (s.value) addChip(s.getAttribute('data-label') + ': ' + s.options[s.selectedIndex].text, function(){ s.value=''; }); });
    // result list
    if (active && shown) {
      var ol = doc.createElement('ol');
      hits.slice(0, 60).forEach(function(u){
        var li = doc.createElement('li'), a = doc.createElement('a'); a.href = '#' + u.id; a.textContent = titleOf(u);
        var w = doc.createElement('span'); w.className = 'where'; w.textContent = ' — ' + chapterTitle(u);
        li.appendChild(a); li.appendChild(w); ol.appendChild(li);
      });
      resultsEl.innerHTML = '<strong>Results</strong>' + (hits.length > 60 ? ' (first 60 shown)' : '');
      resultsEl.appendChild(ol); resultsEl.hidden = false;
    } else { resultsEl.hidden = true; resultsEl.innerHTML = ''; }
  }
  function addChip(text, clear){
    var b = doc.createElement('button'); b.type = 'button'; b.className = 'chip'; b.textContent = text;
    b.setAttribute('aria-label', 'Remove filter ' + text);
    b.addEventListener('click', function(){ clear(); apply(); q.focus(); }); chipsEl.appendChild(b);
  }
  q.addEventListener('input', function(){ clearTimeout(timer); timer = setTimeout(apply, 120); });
  selects.forEach(function(s){ s.addEventListener('change', apply); });
  resetBtn.addEventListener('click', function(){ q.value=''; selects.forEach(function(s){ s.value=''; }); apply(); q.focus(); });
  doc.getElementById('noresults-reset').addEventListener('click', function(){ resetBtn.click(); });
  doc.addEventListener('keydown', function(e){
    if (e.key === '/' && doc.activeElement && !/INPUT|SELECT|TEXTAREA/.test(doc.activeElement.tagName)) { e.preventDefault(); q.focus(); q.select(); }
  });

  /* ---------- deep links ---------- */
  function reveal(id){
    if (!id) return; var el = doc.getElementById(decodeURIComponent(id)); if (!el) return;
    if (el.closest('.hidden-by-filter')) { q.value=''; selects.forEach(function(s){ s.value=''; }); apply(); }
    var d = el.tagName === 'DETAILS' ? el : el.closest('details');
    while (d) { d.open = true; d = d.parentNode.closest('details'); }
    el.scrollIntoView({block:'start'});
    var h = el.matches('h2,h3') ? el : el.querySelector('h2,h3,summary'); if (h) { h.setAttribute('tabindex','-1'); h.focus({preventScroll:true}); }
  }
  window.addEventListener('hashchange', function(){ reveal(location.hash.slice(1)); });
  doc.addEventListener('click', function(e){
    var a = e.target.closest('a[href^="#"]'); if (!a) return; var id = a.getAttribute('href').slice(1);
    if (id && id === location.hash.slice(1)) { e.preventDefault(); reveal(id); }
  });

  /* ---------- current section in nav ---------- */
  var navLinks = Array.prototype.slice.call(nav.querySelectorAll('a[href^="#"]'));
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function(entries){
      entries.forEach(function(en){ if (en.isIntersecting) {
        navLinks.forEach(function(a){ a.removeAttribute('aria-current'); });
        var a = nav.querySelector('a[href="#' + en.target.id + '"]'); if (a) a.setAttribute('aria-current','true');
      }});
    }, { rootMargin: '-70px 0px -70% 0px' });
    chapters.forEach(function(c){ io.observe(c); });
  }

  doc.getElementById('print-btn').addEventListener('click', function(){ window.print(); });
  /* ---------- print: open everything ---------- */
  var openedForPrint = [];
  window.addEventListener('beforeprint', function(){ Array.prototype.forEach.call(doc.querySelectorAll('details:not([open])'), function(d){ d.open = true; openedForPrint.push(d); }); });
  window.addEventListener('afterprint', function(){ openedForPrint.forEach(function(d){ d.open = false; }); openedForPrint = []; });

  apply();
  if (location.hash) reveal(location.hash.slice(1));
})();
"""


def page(title, description, nav_html, filters_html, body_html, intro_html=""):
    m = META
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="light dark">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="topbar">
  <button class="btn" id="nav-toggle" type="button" aria-controls="side" aria-expanded="false">Contents</button>
  <span class="brand"><span class="g">GOSH</span><span class="c">CODER</span><span class="hb"> Handbook</span></span>
  <span class="meta-pill">v{m['version']} · {m['revision']}</span>
  <span class="meta-pill opt">Audited {m['audit_date']}</span>
  <span class="spacer"></span>
  <button class="btn" id="theme-toggle" type="button">Theme: auto</button>
  <button class="btn" id="print-btn" type="button">Print</button>
</header>
<div class="layout">
<nav class="side" id="side" aria-label="Handbook contents">
{nav_html}
</nav>
<main id="content" tabindex="-1">
<div class="content">
<noscript><div class="noscript">JavaScript is off: every section below is shown in full. Search, filters, the theme switch and image enlargement need JavaScript; screenshot links still open the full image.</div></noscript>
{intro_html}
{filters_html}
{body_html}
<footer class="foot">
<p>GoshCoder handbook for version {m['version']} (<code>{m['describe']}</code>, commit <code>{m['revision_full']}</code> on <code>{m['branch']}</code>). Audit and screenshots taken {m['audit_date']} on Linux with {m['toolchain']}. GoshCoder is MIT-licensed and a derivative of pi; see <a href="{REPO_ROOT}NOTICE">NOTICE</a> and <a href="{REPO_ROOT}LICENSE">LICENSE</a>.</p>
</footer>
</div>
</main>
</div>
<dialog id="lightbox" aria-label="Enlarged screenshot">
  <div class="lb">
    <img id="lb-img" alt="">
    <div class="lbbar"><p id="lb-cap"></p><span><a id="lb-open" href="#" target="_blank" rel="noopener">Open image</a> <button class="btn" id="lb-close" type="button">Close</button></span></div>
  </div>
</dialog>
<script>{JS}</script>
</body>
</html>
"""
