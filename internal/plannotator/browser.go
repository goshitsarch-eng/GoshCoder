package plannotator

// Native loopback browser reviewer. The upstream extension ships a large SPA;
// GoshCoder keeps the same approve/deny-with-annotations contract in a compact,
// dependency-free page suitable for its line-oriented interface.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
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
	Number int
	Text   string
}
type reviewPageData struct {
	Title string
	Lines []reviewLine
	Token string
}

func (b BrowserReviewer) Review(ctx context.Context, title, markdown string) (Decision, error) {
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
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
		parts := strings.Split(markdown, "\n")
		lines := make([]reviewLine, len(parts))
		for index, text := range parts {
			lines[index] = reviewLine{Number: index + 1, Text: text}
		}
		_ = reviewPage.Execute(w, reviewPageData{Title: title, Lines: lines, Token: token})
	})
	mux.HandleFunc("/api/decision", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
		if err := r.ParseForm(); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if r.FormValue("token") != token {
			http.Error(w, "invalid review token", http.StatusForbidden)
			return
		}
		decision := Decision{Approved: r.FormValue("action") == "approve", Feedback: strings.TrimSpace(r.FormValue("feedback"))}
		once.Do(func() { result <- decision })
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, `<!doctype html><title>Plannotator</title><style>body{font:16px system-ui;padding:3rem;background:#111;color:#eee}</style><h2>Review submitted</h2><p>You can close this tab.</p>`)
	})
	mux.HandleFunc("/api/decision.json", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
		var payload struct {
			Decision
			Token string `json:"token"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if payload.Token != token {
			http.Error(w, "invalid review token", http.StatusForbidden)
			return
		}
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
		b.Notify("Plannotator review: " + reviewURL)
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

var reviewPage = template.Must(template.New("review").Parse(`<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{{.Title}}</title><style>
:root{color-scheme:dark} body{margin:0;background:#111318;color:#e8e8ea;font:15px system-ui}
main{max-width:1100px;margin:auto;padding:28px}.card{background:#1a1d24;border:1px solid #343944;border-radius:12px;padding:20px}
pre{overflow:auto;line-height:1.5;background:#101217;padding:18px;border-radius:8px}.line{display:block;white-space:pre-wrap;overflow-wrap:anywhere}.line:hover{background:#282d37;cursor:pointer}.ln{display:inline-block;width:3.5em;color:#737b8c;user-select:none}
textarea{box-sizing:border-box;width:100%;min-height:130px;background:#101217;color:#eee;border:1px solid #444b59;border-radius:8px;padding:12px}
.actions{display:flex;gap:10px;margin-top:12px}button{border:0;border-radius:8px;padding:10px 18px;font-weight:650;cursor:pointer}
.approve{background:#58a66b;color:#07140a}.deny{background:#d06464;color:#180707}h1{font-size:22px}
</style></head><body><main><h1>{{.Title}}</h1><div class="card"><p>Click a line to attach feedback to it.</p><pre>{{range .Lines}}<span class="line" data-line="{{.Number}}"><span class="ln">{{.Number}}</span>{{.Text}}</span>{{end}}</pre>
<form method="post" action="/api/decision"><input type="hidden" name="token" value="{{.Token}}"><label for="feedback"><strong>Annotations / implementation notes</strong></label>
<textarea id="feedback" name="feedback" placeholder="Optional feedback. Add required changes before denying, or guidance before approving."></textarea>
<div class="actions"><button class="approve" name="action" value="approve">Approve plan</button>
<button class="deny" name="action" value="deny">Deny and request changes</button></div></form></div></main>
<script>document.querySelectorAll('.line').forEach(function(line){line.addEventListener('click',function(){var box=document.getElementById('feedback');var prefix='Line '+line.dataset.line+': ';if(box.value && !box.value.endsWith('\n'))box.value+='\n';box.value+=prefix;box.focus();box.setSelectionRange(box.value.length,box.value.length);});});</script></body></html>`))
