"""Builds docs/handbook/index.html from the content modules beside this file.

Run from anywhere:  python3 docs/handbook/src/build.py
The page is self-contained; screenshots live in docs/handbook/screenshots/."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hb_lib import *  # noqa
from hb_part1 import overview, getting_started
from hb_part2 import manual
from hb_part3 import architecture, development, build_release, troubleshooting, coverage
from hb_findings import findings_chapter

CHAPTERS = [
    ("overview", "Overview", overview, True),
    ("getting-started", "Getting started", getting_started, True),
    ("manual", "User manual", manual, True),
    ("architecture", "Architecture", architecture, True),
    ("development", "Development", development, True),
    ("build-release", "Build and release", build_release, True),
    ("troubleshooting", "Troubleshooting", troubleshooting, False),
    ("findings", "Audit findings", findings_chapter, False),
    ("coverage", "Coverage", coverage, False),
]

TOPICS = [
    ("install", "Installation"), ("interface", "Interface"), ("keyboard", "Keyboard"), ("providers", "Providers and logins"),
    ("tools", "Tools"), ("agent", "Agent and retries"), ("sessions", "Sessions"), ("resources", "Context, templates, skills"),
    ("extensions", "Extensions and gateways"), ("security", "Security and privacy"), ("config", "Configuration and files"),
    ("reference", "Reference tables"), ("accessibility", "Accessibility"), ("architecture", "Architecture"),
    ("development", "Development"), ("build", "Build, CI, release"), ("troubleshooting", "Troubleshooting"),
    ("findings", "Audit findings"), ("coverage", "Coverage"), ("docs", "Documentation"), ("performance", "Performance"),
]


def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s)


def build_nav(chapter_html):
    out = ['<ol>']
    for cid, title, _, sub in CHAPTERS:
        html_c = chapter_html[cid]
        out.append(f'<li><a href="#{cid}"><span>{esc(title)}</span><span class="count" hidden>0</span></a>')
        if sub:
            items = re.findall(r'<section class="unit" id="([^"]+)"[^>]*>\s*<h3><a class="anchor"[^>]*>#</a>(.*?) <span class="vbadge', html_c, re.S)
            if items:
                out.append("<ol>")
                for uid, t in items:
                    out.append(f'<li><a href="#{uid}">{esc(strip_tags(t))}</a></li>')
                out.append("</ol>")
        out.append("</li>")
    out.append("</ol>")
    return "\n".join(out)


def select(sid, label, options):
    opts = "".join(f'<option value="{v}">{esc(t)}</option>' for v, t in options)
    return f'<label for="{sid}">{label}<select id="{sid}" data-label="{label}"><option value="">All</option>{opts}</select></label>'


def build_filters():
    return f"""<section class="filters" aria-label="Search and filters">
<div class="row">
<label class="grow" for="q">Search the handbook <input type="search" id="q" placeholder="e.g. resume, auth.json, F-02, Esc" autocomplete="off" aria-describedby="count"></label>
{select('f-audience','Audience',[('user','Users'),('developer','Developers'),('maintainer','Maintainers')])}
{select('f-topic','Topic',TOPICS)}
{select('f-platform','Platform',[('linux','Linux'),('macos','macOS'),('windows','Windows')])}
{select('f-status','Verification',[(k, STATUS_LABEL[k]) for k in ['runtime','source','inferred','blocked','docs']])}
{select('f-severity','Severity',[('high','High'),('medium','Medium'),('low','Low'),('info','Info')])}
</div>
<div class="status-line"><span id="count" role="status" aria-live="polite">All sections shown</span><span class="chips" id="chips"></span><button class="btn" type="button" id="reset" disabled>Reset filters</button></div>
<div id="results" hidden></div>
</section>
<div id="noresults" hidden><p><strong>Nothing matches.</strong> Try fewer or different words, or clear a filter: the Severity filter only matches audit findings, and Platform hides items specific to other systems.</p><button class="btn" type="button" id="noresults-reset">Reset search and filters</button></div>"""


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    chapter_html = {cid: fn() for cid, _, fn, _ in CHAPTERS}
    intro = f"""<h1>GoshCoder Handbook</h1>
<p class="lead">How GoshCoder {META['version']} is built, how it works and how to use it — with the results of a source and runtime audit of revision <code>{META['revision']}</code> ({META['audit_date']}).</p>"""
    body = "\n".join(chapter_html[c] for c, *_ in CHAPTERS)
    out = page("GoshCoder Handbook", f"User manual, architecture guide and audit for GoshCoder {META['version']} ({META['revision']}).",
               build_nav(chapter_html), build_filters(), body, intro)
    path = os.path.join(repo, "docs", "handbook", "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    # sanity: every referenced finding id exists, every screenshot exists
    ids = set(re.findall(r'id="([^"]+)"', out))
    missing = sorted(set(re.findall(r'href="#([A-Za-z0-9_-]+)"', out)) - ids)
    shots = set(re.findall(r'src="(screenshots/[^"]+)"', out))
    missing_shots = [s for s in shots if not os.path.exists(os.path.join(repo, "docs", "handbook", s))]
    dup = [i for i in ids if out.count(f'id="{i}"') > 1]
    print("wrote", path, len(out), "bytes; missing anchors:", missing, "missing shots:", missing_shots, "duplicate ids:", dup)


if __name__ == "__main__":
    main()
