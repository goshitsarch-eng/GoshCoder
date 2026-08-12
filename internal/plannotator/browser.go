package plannotator

// Native loopback browser reviewer. Its layout and review controls follow the
// original upstream plan surface while keeping the implementation
// dependency-free and local to GoshCoder.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"net"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"time"
)

type BrowserReviewer struct {
	Host        string
	OpenBrowser func(string) error
	Notify      func(string)
}

type reviewLine struct {
	Number  int
	Text    string
	Display string
	Kind    string
}

type reviewHeading struct {
	Number int
	Level  int
	Text   string
}

type reviewPageData struct {
	Title       string
	Markdown    string
	Previous    string
	Lines       []reviewLine
	Headings    []reviewHeading
	Token       string
	HasPrevious bool
}

// Review opens a first-version review.
func (b BrowserReviewer) Review(ctx context.Context, title, markdown string) (Decision, error) {
	return b.review(ctx, title, markdown, "")
}

// ReviewVersion opens a review with the previously denied plan available as a
// Changes view, matching the upstream resubmission workflow.
func (b BrowserReviewer) ReviewVersion(ctx context.Context, title, markdown, previous string) (Decision, error) {
	return b.review(ctx, title, markdown, previous)
}

func (b BrowserReviewer) review(ctx context.Context, title, markdown, previous string) (Decision, error) {
	host := b.Host
	if host == "" {
		host = "127.0.0.1"
	}
	if host != "localhost" {
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			return Decision{}, fmt.Errorf("review server host must be loopback, got %q", host)
		}
	}
	listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
	if err != nil {
		return Decision{}, err
	}
	defer listener.Close()
	if address, ok := listener.Addr().(*net.TCPAddr); !ok || !address.IP.IsLoopback() {
		return Decision{}, errors.New("review server did not bind to a loopback address")
	}

	tokenBytes := make([]byte, 24)
	if _, err := rand.Read(tokenBytes); err != nil {
		return Decision{}, err
	}
	token := hex.EncodeToString(tokenBytes)
	result := make(chan Decision, 1)
	var once sync.Once
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		setReviewHeaders(w)
		lines, headings := prepareReviewDocument(markdown)
		_ = reviewPage.Execute(w, reviewPageData{
			Title: title, Markdown: markdown, Previous: previous, Lines: lines,
			Headings: headings, Token: token, HasPrevious: strings.TrimSpace(previous) != "",
		})
	})
	mux.HandleFunc("/api/decision", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		if err := r.ParseForm(); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if r.FormValue("token") != token {
			http.Error(w, "invalid review token", http.StatusForbidden)
			return
		}
		decision := Decision{
			Approved: r.FormValue("action") == "approve",
			Feedback: strings.TrimSpace(r.FormValue("feedback")),
		}
		once.Do(func() { result <- decision })
		setReviewHeaders(w)
		_ = reviewCompletePage.Execute(w, map[string]any{"Approved": decision.Approved})
	})
	mux.HandleFunc("/api/decision.json", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		var payload struct {
			Decision
			Token string `json:"token"`
		}
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&payload); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if payload.Token != token {
			http.Error(w, "invalid review token", http.StatusForbidden)
			return
		}
		payload.Feedback = strings.TrimSpace(payload.Feedback)
		once.Do(func() { result <- payload.Decision })
		w.WriteHeader(http.StatusNoContent)
	})

	server := &http.Server{
		Handler: mux, ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Second,
		IdleTimeout: 30 * time.Second, MaxHeaderBytes: 16 << 10,
	}
	go func() { _ = server.Serve(listener) }()
	defer func() {
		shutdown, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdown)
	}()

	reviewURL := "http://" + listener.Addr().String() + "/"
	if b.Notify != nil {
		b.Notify("Planner review: " + reviewURL)
	}
	opener := b.OpenBrowser
	if opener == nil {
		opener = openBrowser
	}
	_ = opener(reviewURL)

	select {
	case decision := <-result:
		return decision, nil
	case <-ctx.Done():
		return Decision{}, ctx.Err()
	}
}

func setReviewHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
}

func prepareReviewDocument(markdown string) ([]reviewLine, []reviewHeading) {
	parts := strings.Split(strings.ReplaceAll(markdown, "\r\n", "\n"), "\n")
	lines := make([]reviewLine, 0, len(parts))
	var headings []reviewHeading
	inCode, diffCode := false, false
	for index, text := range parts {
		trimmed := strings.TrimSpace(text)
		line := reviewLine{Number: index + 1, Text: text, Display: text, Kind: "paragraph"}
		if strings.HasPrefix(trimmed, "```") {
			if !inCode {
				inCode = true
				diffCode = strings.EqualFold(strings.TrimSpace(strings.TrimPrefix(trimmed, "```")), "diff")
			} else {
				inCode, diffCode = false, false
			}
			line.Kind = "fence"
		} else if inCode {
			line.Kind = "code"
			if diffCode {
				switch {
				case strings.HasPrefix(text, "+") && !strings.HasPrefix(text, "+++"):
					line.Kind = "diff-add"
				case strings.HasPrefix(text, "-") && !strings.HasPrefix(text, "---"):
					line.Kind = "diff-remove"
				case strings.HasPrefix(text, "@@"):
					line.Kind = "diff-hunk"
				}
			}
		} else if strings.HasPrefix(trimmed, "#") {
			level := 0
			for level < len(trimmed) && trimmed[level] == '#' {
				level++
			}
			if level > 6 {
				level = 6
			}
			display := strings.TrimSpace(trimmed[level:])
			line.Kind = fmt.Sprintf("heading-%d", level)
			line.Display = display
			headings = append(headings, reviewHeading{Number: line.Number, Level: level, Text: display})
		} else if strings.HasPrefix(trimmed, "- [ ] ") || strings.HasPrefix(strings.ToLower(trimmed), "- [x] ") {
			line.Kind = "task"
			line.Display = strings.TrimSpace(trimmed[6:])
		} else if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			line.Kind = "bullet"
			line.Display = strings.TrimSpace(trimmed[2:])
		} else if strings.HasPrefix(trimmed, "> ") {
			line.Kind = "quote"
			line.Display = strings.TrimSpace(trimmed[2:])
		} else if trimmed == "" {
			line.Kind = "blank"
		}
		lines = append(lines, line)
	}
	return lines, headings
}

func openBrowser(target string) error {
	parsed, err := url.Parse(target)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("refusing to open non-http URL")
	}
	switch runtime.GOOS {
	case "windows":
		return exec.Command("rundll32", "url.dll,FileProtocolHandler", target).Start()
	case "darwin":
		return exec.Command("open", target).Start()
	default:
		return exec.Command("xdg-open", target).Start()
	}
}

var reviewCompletePage = template.Must(template.New("complete").Parse(`<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Planner</title><style>
:root{color-scheme:dark;--bg:#090b10;--panel:#10141c;--line:#242b38;--text:#eef1f7;--muted:#8b94a5;--accent:#a99cff;--ok:#65d38a}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 50% 20%,#171c2b,var(--bg) 58%);color:var(--text);font:15px Inter,ui-sans-serif,system-ui}.done{width:min(440px,calc(100% - 32px));padding:38px;text-align:center;background:var(--panel);border:1px solid var(--line);border-radius:18px;box-shadow:0 24px 80px #0008}.icon{display:grid;place-items:center;width:50px;height:50px;margin:auto;border-radius:50%;background:#65d38a18;color:var(--ok);font-size:25px}h1{font-size:21px;margin:18px 0 8px}p{color:var(--muted);margin:0;line-height:1.6}</style></head><body><section class="done"><div class="icon">✓</div><h1>{{if .Approved}}Plan approved{{else}}Feedback sent{{end}}</h1><p>GoshCoder received your review. You can safely close this tab.</p></section></body></html>`))

var reviewPage = template.Must(template.New("review").Parse(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{.Title}} · Planner</title><style>
:root{color-scheme:dark;--bg:#090b10;--panel:#0f131b;--panel2:#131924;--line:#242b38;--line2:#30394a;--text:#edf0f6;--muted:#8993a5;--faint:#596274;--accent:#a99cff;--accent2:#7869ea;--ok:#65d38a;--danger:#ef7777;--warning:#e7ad61;--code:#0a0e15;--shadow:0 22px 70px #0007}
:root.light{color-scheme:light;--bg:#f5f6f8;--panel:#fff;--panel2:#f8f9fb;--line:#dfe3e9;--line2:#cbd1db;--text:#1c2230;--muted:#687184;--faint:#9299a8;--accent:#6957d9;--accent2:#5945d1;--ok:#27844b;--danger:#c04444;--warning:#aa681f;--code:#f1f3f6;--shadow:0 22px 70px #2232}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;overflow:hidden;background:var(--bg);color:var(--text);font:14px Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,textarea{font:inherit}button{color:inherit}.header{height:52px;display:flex;align-items:center;gap:14px;padding:0 16px;border-bottom:1px solid var(--line);background:color-mix(in srgb,var(--panel) 92%,transparent);backdrop-filter:blur(16px);position:relative;z-index:10}.brand{font-weight:700;letter-spacing:-.02em}.crumb{min-width:0;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.actions{margin-left:auto;display:flex;align-items:center;gap:8px}.btn{height:32px;padding:0 13px;border:1px solid var(--line2);border-radius:7px;background:var(--panel2);cursor:pointer;font-size:12px;font-weight:650;transition:.15s}.btn:hover{border-color:var(--faint);transform:translateY(-1px)}.btn.primary{border-color:#0000;background:var(--ok);color:#07130b}.btn.feedback{border-color:var(--accent2);background:var(--accent2);color:white}.iconbtn{width:32px;padding:0;color:var(--muted)}.layout{height:calc(100% - 52px);display:grid;grid-template-columns:248px minmax(0,1fr) 304px}.left,.right{min-width:0;background:var(--panel);overflow:auto}.left{border-right:1px solid var(--line);padding:14px 10px}.right{border-left:1px solid var(--line);display:flex;flex-direction:column}.side-title{padding:4px 9px 10px;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase}.toc{display:block;padding:7px 9px;border-radius:6px;color:var(--muted);text-decoration:none;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.toc:hover{color:var(--text);background:var(--panel2)}.toc.l2{padding-left:18px}.toc.l3,.toc.l4,.toc.l5,.toc.l6{padding-left:28px;color:var(--faint)}.meta-card{margin-top:16px;padding:12px;border:1px solid var(--line);border-radius:9px;background:var(--panel2);color:var(--muted);font-size:12px;line-height:1.55}.main{min-width:0;overflow:auto;background:radial-gradient(circle at 50% -20%,color-mix(in srgb,var(--accent) 8%,transparent),transparent 38%),var(--bg)}.toolbar{position:sticky;top:0;z-index:4;display:flex;justify-content:center;padding:12px;pointer-events:none}.toolgroup{pointer-events:auto;display:flex;align-items:center;gap:3px;padding:4px;border:1px solid var(--line);border-radius:9px;background:color-mix(in srgb,var(--panel) 90%,transparent);box-shadow:0 8px 30px #0003;backdrop-filter:blur(12px)}.tool{border:0;border-radius:6px;padding:7px 10px;background:transparent;color:var(--muted);font-size:12px;cursor:pointer}.tool:hover,.tool.active{background:var(--panel2);color:var(--text)}.document{width:min(880px,calc(100% - 48px));margin:0 auto 80px;padding:44px 58px 70px;border:1px solid var(--line);border-radius:13px;background:var(--panel);box-shadow:var(--shadow)}.line{position:relative;min-height:1.55em;padding:1px 12px 1px 34px;border-radius:5px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65}.line:hover{background:color-mix(in srgb,var(--accent) 8%,transparent)}.line.selected{background:color-mix(in srgb,var(--accent) 15%,transparent);box-shadow:inset 2px 0 var(--accent)}.ln{position:absolute;left:4px;top:4px;width:22px;text-align:right;color:transparent;font:10px ui-monospace,monospace;user-select:none}.line:hover .ln,.line.selected .ln{color:var(--faint)}.heading-1{font-size:26px;font-weight:750;letter-spacing:-.025em;margin:12px 0 18px}.heading-2{font-size:19px;font-weight:700;margin:25px 0 9px}.heading-3{font-size:16px;font-weight:680;margin:18px 0 6px}.heading-4,.heading-5,.heading-6{font-weight:680;margin-top:12px}.blank{height:12px;min-height:12px}.bullet:before{content:"•";position:absolute;left:20px;color:var(--accent)}.task:before{content:"☐";position:absolute;left:17px;color:var(--accent)}.quote{margin:6px 0;border-left:2px solid var(--accent);border-radius:0;color:var(--muted);font-style:italic}.code,.diff-add,.diff-remove,.diff-hunk{border-radius:0;background:var(--code);font:12px/1.6 "Geist Mono",ui-monospace,SFMono-Regular,Consolas,monospace}.fence{display:none}.diff-add{color:#9be5b2;background:color-mix(in srgb,var(--ok) 12%,var(--code))}.diff-remove{color:#ffaaaa;background:color-mix(in srgb,var(--danger) 12%,var(--code))}.diff-hunk{color:#aaa0ef}.line code{padding:2px 5px;border:1px solid var(--line);border-radius:5px;background:var(--code);color:var(--accent);font:12px ui-monospace,monospace}.line a{color:var(--accent);text-decoration:none}.line a:hover{text-decoration:underline}.right-head{height:43px;display:flex;align-items:center;padding:0 14px;border-bottom:1px solid var(--line);font-size:12px;font-weight:700}.badge{margin-left:7px;min-width:18px;height:18px;padding:0 5px;display:inline-grid;place-items:center;border-radius:10px;background:color-mix(in srgb,var(--accent) 16%,transparent);color:var(--accent);font:10px ui-monospace,monospace}.annotations{flex:1;overflow:auto;padding:10px}.empty{height:100%;display:grid;place-content:center;text-align:center;color:var(--muted);line-height:1.6}.empty-icon{font-size:25px;color:var(--faint)}.annotation{padding:11px;margin-bottom:8px;border:1px solid var(--line);border-radius:8px;background:var(--panel2)}.annotation-top{display:flex;gap:8px;align-items:center;color:var(--warning);font-size:11px;font-weight:700}.remove{margin-left:auto;border:0;background:transparent;color:var(--faint);cursor:pointer}.quote-text{margin:8px 0;color:var(--muted);font:11px ui-monospace,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.comment{width:100%;min-height:72px;resize:vertical;padding:9px;border:1px solid var(--line);border-radius:6px;background:var(--code);color:var(--text);outline:none;font-size:12px}.comment:focus,.notes:focus{border-color:var(--accent)}.notes-wrap{padding:12px;border-top:1px solid var(--line)}.notes-label{display:block;margin-bottom:7px;color:var(--muted);font-size:11px;font-weight:700}.notes{width:100%;min-height:76px;resize:vertical;padding:10px;border:1px solid var(--line);border-radius:7px;background:var(--code);color:var(--text);outline:none;font-size:12px}.edit-pane{display:none;width:min(980px,calc(100% - 48px));height:calc(100% - 74px);margin:0 auto 28px}.edit-pane.open{display:flex}.editor{width:100%;height:100%;resize:none;padding:24px;border:1px solid var(--line);border-radius:12px;background:var(--code);color:var(--text);font:13px/1.65 "Geist Mono",ui-monospace,monospace;outline:none}.diff-view{display:none;width:min(980px,calc(100% - 48px));margin:0 auto 70px;padding:24px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}.diff-view.open{display:block}.diff-row{display:grid;grid-template-columns:40px 1fr;padding:2px 8px;font:12px/1.55 ui-monospace,monospace;white-space:pre-wrap}.diff-row.add{background:color-mix(in srgb,var(--ok) 12%,transparent);color:#9be5b2}.diff-row.remove{background:color-mix(in srgb,var(--danger) 12%,transparent);color:#ffaaaa}.mobile-panel{display:none}@media(max-width:1050px){.layout{grid-template-columns:0 minmax(0,1fr) 288px}.left{visibility:hidden}.document{width:calc(100% - 28px);padding:34px 34px 60px}}@media(max-width:760px){.layout{grid-template-columns:1fr}.right{display:flex;position:fixed;z-index:20;inset:52px 0 0 15%;box-shadow:-24px 0 70px #0008;transform:translateX(110%);transition:transform .2s}.right.open{transform:translateX(0)}.crumb{display:none}.header{padding:0 9px}.btn{padding:0 9px}.btn .wide{display:none}.document{padding:26px 18px 50px}.toolbar{justify-content:flex-start}.mobile-panel{display:inline-block}}
</style></head><body>
<header class="header"><div class="brand">Planner</div><div class="crumb">{{.Title}}</div><div class="actions"><button class="btn mobile-panel" type="button" id="toggle-panel" title="Toggle annotations">Comments</button><button class="btn iconbtn" type="button" id="theme" title="Toggle theme">◐</button><button class="btn feedback" type="submit" form="decision-form" name="action" value="deny"><span class="wide">Send </span>Feedback</button><button class="btn primary" type="submit" form="decision-form" name="action" value="approve">✓ Approve</button></div></header>
<form id="decision-form" method="post" action="/api/decision"><input type="hidden" name="token" value="{{.Token}}"><input type="hidden" id="feedback" name="feedback">
<div class="layout"><aside class="left"><div class="side-title">Contents</div>{{range .Headings}}<a class="toc l{{.Level}}" href="#line-{{.Number}}">{{.Text}}</a>{{end}}<div class="meta-card">Click any line to attach a comment.<br><br><b>⌘/Ctrl + Enter</b> sends the current review.</div></aside>
<main class="main"><div class="toolbar"><div class="toolgroup"><button class="tool active" type="button" id="annotate-mode">◫ Select</button><button class="tool" type="button" id="edit-mode">✎ Edit</button>{{if .HasPrevious}}<button class="tool" type="button" id="diff-mode">± Changes</button>{{end}}<button class="tool" type="button" id="copy">⧉ Copy plan</button></div></div>
<article class="document" id="document">{{range .Lines}}<div id="line-{{.Number}}" class="line {{.Kind}}" data-line="{{.Number}}" data-text="{{.Text}}" data-display="{{.Display}}"><span class="ln">{{.Number}}</span>{{.Display}}</div>{{end}}</article>
<div class="edit-pane" id="edit-pane"><textarea class="editor" id="direct-edit" spellcheck="false" aria-label="Edit plan markdown">{{.Markdown}}</textarea></div>
{{if .HasPrevious}}<div class="diff-view" id="diff-view" data-previous="{{.Previous}}" data-current="{{.Markdown}}"></div>{{end}}</main>
<aside class="right"><div class="right-head">Annotations <span class="badge" id="count">0</span></div><div class="annotations" id="annotations"><div class="empty" id="empty"><div><div class="empty-icon">▢</div>Click a line to add feedback</div></div></div><div class="notes-wrap"><label class="notes-label" for="notes">Overall implementation notes</label><textarea class="notes" id="notes" placeholder="Optional guidance, questions, or approval notes…"></textarea></div></aside></div></form>
<script>
(function(){
const annotations=[],$=s=>document.querySelector(s),all=s=>Array.from(document.querySelectorAll(s));
const documentView=$('#document'),editPane=$('#edit-pane'),diffView=$('#diff-view'),feedback=$('#feedback'),form=$('#decision-form'),rightPanel=$('.right');
function escapeText(s){return String(s||'').trim()}
function renderInline(el){if(el.matches('.code,.diff-add,.diff-remove,.diff-hunk,.blank'))return;const source=el.dataset.display||'',number=el.querySelector('.ln');while(el.lastChild&&el.lastChild!==number)el.removeChild(el.lastChild);const pattern=/(\x60[^\x60]+\x60|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;let at=0;for(const match of source.matchAll(pattern)){el.append(document.createTextNode(source.slice(at,match.index)));const token=match[0];if(token.startsWith('\x60')){const code=document.createElement('code');code.textContent=token.slice(1,-1);el.append(code)}else if(token.startsWith('**')){const strong=document.createElement('strong');strong.textContent=token.slice(2,-2);el.append(strong)}else{const split=token.indexOf(']('),link=document.createElement('a'),href=token.slice(split+2,-1);link.textContent=token.slice(1,split);if(/^(https?:|#)/.test(href)){link.href=href;if(!href.startsWith('#')){link.target='_blank';link.rel='noreferrer'}}el.append(link)}at=match.index+token.length}el.append(document.createTextNode(source.slice(at)))}
all('.line').forEach(renderInline);
function render(){const host=$('#annotations'),empty=$('#empty');all('.line').forEach(x=>x.classList.remove('selected'));host.querySelectorAll('.annotation').forEach(x=>x.remove());annotations.forEach((a,i)=>{const line=document.querySelector('[data-line="'+a.line+'"]');if(line)line.classList.add('selected');const card=document.createElement('div');card.className='annotation';const top=document.createElement('div');top.className='annotation-top';top.textContent='Comment · Line '+a.line;const remove=document.createElement('button');remove.type='button';remove.className='remove';remove.textContent='×';remove.title='Remove annotation';remove.onclick=()=>{annotations.splice(i,1);render()};top.appendChild(remove);const quote=document.createElement('div');quote.className='quote-text';quote.textContent='“'+a.text.trim()+'”';const input=document.createElement('textarea');input.className='comment';input.placeholder='What should change here?';input.value=a.comment;input.oninput=()=>{a.comment=input.value};card.append(top,quote,input);host.appendChild(card)});empty.style.display=annotations.length?'none':'grid';$('#count').textContent=annotations.length}
all('.line').forEach(line=>line.addEventListener('click',()=>{const n=line.dataset.line;if(annotations.some(a=>a.line===n)){const index=annotations.findIndex(a=>a.line===n);const boxes=all('.comment');boxes[index]&&boxes[index].focus();return}annotations.push({line:n,text:line.dataset.text||'',comment:''});render();if(innerWidth<=760)rightPanel.classList.add('open');setTimeout(()=>{const boxes=all('.comment');boxes[boxes.length-1]&&boxes[boxes.length-1].focus()},0)}));
function setMode(mode){documentView.style.display=mode==='view'?'block':'none';editPane.classList.toggle('open',mode==='edit');if(diffView)diffView.classList.toggle('open',mode==='diff');$('#annotate-mode').classList.toggle('active',mode==='view');$('#edit-mode').classList.toggle('active',mode==='edit');$('#diff-mode')&&$('#diff-mode').classList.toggle('active',mode==='diff')}
$('#annotate-mode').onclick=()=>setMode('view');$('#edit-mode').onclick=()=>setMode('edit');if($('#diff-mode'))$('#diff-mode').onclick=()=>{buildDiff();setMode('diff')};
function buildDiff(){if(!diffView||diffView.dataset.ready)return;diffView.dataset.ready='1';const old=(diffView.dataset.previous||'').split('\n'),now=(diffView.dataset.current||'').split('\n'),max=Math.max(old.length,now.length);for(let i=0;i<max;i++){if(old[i]===now[i])continue;if(old[i]!==undefined)addDiff('-',old[i],'remove',i+1);if(now[i]!==undefined)addDiff('+',now[i],'add',i+1)}}
function addDiff(mark,text,kind,n){const row=document.createElement('div');row.className='diff-row '+kind;const no=document.createElement('span');no.textContent=mark+n;const body=document.createElement('span');body.textContent=text;row.append(no,body);diffView.appendChild(row)}
function compile(){const chunks=[];annotations.forEach(a=>{const comment=escapeText(a.comment);if(comment)chunks.push('- Line '+a.line+' ("'+escapeText(a.text).slice(0,120)+'"): '+comment)});const notes=escapeText($('#notes').value);if(notes)chunks.push('## Overall notes\n\n'+notes);const edited=$('#direct-edit').value,current={{.Markdown}};if(edited!==current)chunks.push('## Direct edits\n\nThe reviewer edited the plan directly. Apply this complete revised Markdown:\n\n~~~markdown\n'+edited+'\n~~~');feedback.value=chunks.length?'## Planner review\n\n'+chunks.join('\n\n'):''}
form.addEventListener('submit',e=>{compile();if(e.submitter&&e.submitter.value==='approve'&&$('#direct-edit').value!=={{.Markdown}})e.submitter.value='deny'});document.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter'){e.preventDefault();compile();const action=(annotations.length||escapeText($('#notes').value)||$('#direct-edit').value!=={{.Markdown}})?'deny':'approve';const button=document.createElement('button');button.type='submit';button.name='action';button.value=action;form.appendChild(button);button.click()}});
$('#copy').onclick=async()=>{try{await navigator.clipboard.writeText({{.Markdown}});$('#copy').textContent='✓ Copied';setTimeout(()=>$('#copy').textContent='⧉ Copy plan',1200)}catch(_){}};
$('#toggle-panel').onclick=()=>rightPanel.classList.toggle('open');$('#theme').onclick=()=>{document.documentElement.classList.toggle('light');localStorage.setItem('planner-theme',document.documentElement.classList.contains('light')?'light':'dark')};if(localStorage.getItem('planner-theme')==='light')document.documentElement.classList.add('light');render();
})();
</script></body></html>`))
