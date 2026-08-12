// Command goshcoder is the GoshCoder CLI: a Go port of pi's coding agent.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"

	"goshcoder/internal/agent"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/ralph"
)

// Version is the CLI version.
const Version = "0.2.0-dev"

const usage = `GoshCoder - a Go coding agent

Usage:
  goshcoder                         Start fullscreen interactive chat
  goshcoder [chat flags]            Start chat without typing the subcommand
  goshcoder run [flags] <prompt>     Run a single prompt
  goshcoder chat [flags]             Interactive session (slash commands, /help)
  goshcoder providers                List providers and credential status
  goshcoder models [provider]        List available models
  goshcoder auth set <provider>      Store an API key (read from stdin)
  goshcoder auth login <provider>    Log in with OAuth
  goshcoder auth list                List stored credentials
  goshcoder auth logout <provider>   Remove a stored credential
  goshcoder ralph list [--archived]  List ralph loops
  goshcoder ralph status             Show the active ralph loop
  goshcoder ralph resume <name>      Reactivate a paused loop
  goshcoder ralph stop <name>        Mark a loop completed
  goshcoder ralph archive <name>     Move a loop to .ralph/archive
  goshcoder ralph delete <name>      Delete a loop and its task file
  goshcoder version                  Print the version

Chat/run flags:
  -m, -model <ref>    Model as "provider/model" or a bare model id (chat remembers the last model)
  -s, -system <text>  System prompt
  -thinking <level>   off|minimal|low|medium|high|xhigh|max (default off)
  -tools              Enable file and shell tools (chat default true; disable with -tools=false)
  -ralph              Enable long-running ralph loops (ralph_start/ralph_done)
  -planner            Start with native Planner review enabled
  -claude-tui         Use the native pi-claude-code-tui look in line mode (default true)
  -fullscreen         Use the native fullscreen TUI in chat (default true on terminals)
  -C <dir>            Workspace directory for tools (default: current directory)

Environment:
  GOSHCODER_AGENT_DIR   Override the config directory (default ~/.goshcoder/agent)
  Provider API keys are read from the usual variables, e.g. ANTHROPIC_API_KEY,
  OPENAI_API_KEY, GEMINI_API_KEY.
`

func main() {
	if len(os.Args) < 2 {
		if err := chatCommand(nil); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}

	command, args := os.Args[1], os.Args[2:]
	if strings.HasPrefix(command, "-") {
		if err := chatCommand(os.Args[1:]); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}
	var err error
	switch command {
	case "run":
		err = runCommand(args)
	case "chat":
		err = chatCommand(args)
	case "providers":
		err = providersCommand()
	case "models":
		err = modelsCommand(args)
	case "auth":
		err = authCommand(args)
	case "ralph":
		err = ralphCommand(args)
	case "version", "--version", "-v":
		fmt.Println("goshcoder", Version)
	case "help", "--help", "-h":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n%s", command, usage)
		os.Exit(2)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// newCatalog builds a catalog backed by the on-disk credential store.
func newCatalog() *catalog.Catalog {
	return catalog.NewCatalog(catalog.NewFileCredentialStore(config.AuthPath()))
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

func runCommand(args []string) error {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	cfg := bindSessionFlags(flags)
	if err := flags.Parse(args); err != nil {
		return err
	}

	prompt := strings.TrimSpace(strings.Join(flags.Args(), " "))
	if prompt == "" {
		return errors.New("a prompt is required")
	}

	s, err := newSession(*cfg)
	if err != nil {
		return err
	}
	defer s.close()
	stop := s.handleInterrupts(nil)
	defer stop()

	return s.runTurn(prompt)
}

// bindSessionFlags registers the flags shared by run and chat.
func bindSessionFlags(flags *flag.FlagSet) *sessionConfig {
	cfg := &sessionConfig{}
	flags.StringVar(&cfg.ModelRef, "m", "", "model reference")
	flags.StringVar(&cfg.ModelRef, "model", "", "model reference")
	flags.StringVar(&cfg.SystemPrompt, "s", "", "system prompt")
	flags.StringVar(&cfg.SystemPrompt, "system", "", "system prompt")
	flags.StringVar(&cfg.Thinking, "thinking", "off", "thinking level")
	flags.BoolVar(&cfg.EnableTools, "tools", false, "enable built-in tools")
	flags.BoolVar(&cfg.EnableRalph, "ralph", false, "enable long-running ralph loops")
	flags.BoolVar(&cfg.EnablePlannotator, "planner", false, "start in Planner mode")
	flags.BoolVar(&cfg.EnablePlannotator, "plan", false, "alias for -planner")
	flags.BoolVar(&cfg.ClaudeTUI, "claude-tui", true, "use the pi-claude-code-tui look")
	flags.StringVar(&cfg.Workdir, "C", ".", "workspace directory")
	return cfg
}

// agentQueue adapts the agent to ralph.Queue. It holds a pointer to the agent
// variable rather than the agent itself, because the ralph tools have to be
// registered before the agent that owns them exists.
type agentQueue struct {
	agent **agent.Agent
}

func (q agentQueue) FollowUp(message agent.Message) {
	if q.agent != nil && *q.agent != nil {
		(*q.agent).FollowUp(message)
	}
}

func (q agentQueue) HasQueuedMessages() bool {
	return q.agent != nil && *q.agent != nil && (*q.agent).HasQueuedMessages()
}

// ralphPrepareNextTurn keeps the loop and the agent in sync between turns: it
// closes the loop out when the model emits the completion marker, and otherwise
// refreshes the system prompt so the iteration counter stays current.
//
// This is the Go equivalent of the extension's agent_end and
// before_agent_start hooks, folded into the one hook the agent loop offers
// between turns.
func ralphPrepareNextTurn(loops *ralph.Store, baseSystemPrompt func() string) agent.PrepareNextTurnFunc {
	if loops == nil {
		return nil
	}
	return func(ctx context.Context, c agent.PrepareNextTurnContext) *agent.TurnUpdate {
		base := ""
		if baseSystemPrompt != nil {
			base = baseSystemPrompt()
		}
		state, ok := loops.Current()
		if !ok || state.Status != ralph.StatusActive {
			return nil
		}

		if ralph.HasCompleteMarker(c.Message) {
			if err := loops.Complete(state); err != nil {
				fmt.Fprintf(os.Stderr, "ralph: %v\n", err)
				return nil
			}
			fmt.Fprintf(os.Stderr, "\n%s\n", dim(fmt.Sprintf(
				"ralph: loop %q completed at iteration %d", state.Name, state.Iteration)))
			// Drop the loop instructions now that the loop is done.
			updated := c.Context
			updated.SystemPrompt = base
			return &agent.TurnUpdate{Context: &updated}
		}

		fmt.Fprintf(os.Stderr, "%s\n", dim(loops.StatusLine()))
		updated := c.Context
		updated.SystemPrompt = base + ralph.SystemPromptSuffix(state)
		return &agent.TurnUpdate{Context: &updated}
	}
}

// authStreamFn returns a StreamFn that applies resolved credentials and then
// delegates to the protocol registered for the model's API.
func authStreamFn(auth *catalog.Auth) agent.StreamFn {
	return func(model *llm.Model, ctx *llm.Context, opts *llm.SimpleStreamOptions) *llm.AssistantMessageEventStream {
		if auth != nil {
			opts.Headers = auth.Headers
			opts.Env = auth.Env
		}
		funcs, ok := llm.GetStreamer(model.API)
		if !ok || funcs.StreamSimple == nil {
			return errorStream(model, fmt.Sprintf("no streamer registered for api %q", model.API))
		}
		return funcs.StreamSimple(model, ctx, opts)
	}
}

// errorStream reports a failure through the stream contract, which forbids
// panicking or returning errors directly.
func errorStream(model *llm.Model, message string) *llm.AssistantMessageEventStream {
	stream := llm.NewAssistantMessageEventStream()
	stream.Push(llm.AssistantMessageEvent{
		Type:   llm.EventError,
		Reason: llm.StopError,
		Error: &llm.AssistantMessage{
			Role:         "assistant",
			Content:      []llm.ContentBlock{llm.TextContent{Type: "text", Text: ""}},
			API:          model.API,
			Provider:     model.Provider,
			Model:        model.ID,
			StopReason:   llm.StopError,
			ErrorMessage: message,
		},
	})
	stream.End()
	return stream
}

// renderEvent streams the run to the terminal: assistant text on stdout,
// thinking and tool activity on stderr so stdout stays pipeable.
func renderEvent(ctx context.Context, ev agent.Event) {
	switch ev.Type {
	case agent.EventMessageUpdate:
		if ev.AssistantEvent == nil {
			return
		}
		switch ev.AssistantEvent.Type {
		case llm.EventTextDelta:
			fmt.Print(ev.AssistantEvent.Delta)
		case llm.EventThinkingDelta:
			fmt.Fprint(os.Stderr, dim(ev.AssistantEvent.Delta))
		case llm.EventThinkingEnd:
			fmt.Fprintln(os.Stderr)
		}

	case agent.EventToolExecutionStart:
		fmt.Fprintf(os.Stderr, "\n%s %s\n", dim("→"), bold(ev.ToolName+" "+summarizeArgs(ev.Args)))

	case agent.EventToolExecutionEnd:
		status := dim("✓")
		if ev.IsError {
			status = "✗"
		}
		fmt.Fprintf(os.Stderr, "%s %s\n", status, dim(firstLine(resultText(ev.Result))))

	case agent.EventTurnEnd:
		// Separate consecutive assistant turns in the output.
		fmt.Println()

	case agent.EventAgentEnd:
		if message, ok := lastAssistant(ev.Messages); ok && message.Usage.TotalTokens > 0 {
			fmt.Fprintf(os.Stderr, "%s\n", dim(fmt.Sprintf(
				"tokens: %d in / %d out  cost: $%.4f",
				message.Usage.Input, message.Usage.Output, message.Usage.Cost.Total)))
		}
	}
}

func lastAssistant(messages []agent.Message) (llm.AssistantMessage, bool) {
	for i := len(messages) - 1; i >= 0; i-- {
		if message, ok := messages[i].(llm.AssistantMessage); ok {
			return message, true
		}
	}
	return llm.AssistantMessage{}, false
}

func resultText(result *agent.ToolResult) string {
	if result == nil {
		return ""
	}
	var parts []string
	for _, block := range result.Content {
		if text, ok := block.(llm.TextContent); ok {
			parts = append(parts, text.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func firstLine(text string) string {
	text = strings.TrimSpace(text)
	if index := strings.IndexByte(text, '\n'); index >= 0 {
		return text[:index] + " ..."
	}
	if len(text) > 120 {
		return text[:120] + " ..."
	}
	return text
}

// summarizeArgs renders tool arguments compactly for the activity line.
func summarizeArgs(args map[string]any) string {
	if len(args) == 0 {
		return ""
	}
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var parts []string
	for _, key := range keys {
		value := fmt.Sprint(args[key])
		if len(value) > 60 {
			value = value[:60] + "..."
		}
		parts = append(parts, key+"="+strings.ReplaceAll(value, "\n", " "))
	}
	return strings.Join(parts, " ")
}

// colorEnabled reports whether ANSI styling should be used.
func colorEnabled() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	info, err := os.Stderr.Stat()
	return err == nil && (info.Mode()&os.ModeCharDevice) != 0
}

func dim(text string) string {
	if text == "" || !colorEnabled() {
		return text
	}
	return "\x1b[2m" + text + "\x1b[0m"
}

func bold(text string) string {
	if text == "" || !colorEnabled() {
		return text
	}
	return "\x1b[1m" + text + "\x1b[0m"
}

// ---------------------------------------------------------------------------
// providers / models
// ---------------------------------------------------------------------------

func providersCommand() error {
	c := newCatalog()
	for _, provider := range c.Providers() {
		status := "-"
		detail := ""
		if auth, ok := c.ResolveAuth(provider.ID); ok {
			status = "✓"
			detail = auth.Source
			if auth.IsAmbient() {
				detail += " (ambient)"
			}
		} else if names := providerEnvNames(provider); names != "" {
			detail = "set " + names
		}
		fmt.Printf("%s %-24s %-22s %s\n", status, provider.ID, provider.Name, dim(detail))
	}
	return nil
}

func providerEnvNames(provider *catalog.Provider) string {
	if len(provider.EnvKeys) == 0 {
		return ""
	}
	return strings.Join(provider.EnvKeys, " or ")
}

func modelsCommand(args []string) error {
	c := newCatalog()

	if len(args) > 0 {
		provider := c.Provider(args[0])
		if provider == nil {
			return fmt.Errorf("unknown provider %q", args[0])
		}
		return printModels(provider)
	}

	// With no argument, list models for providers that have credentials.
	configured := c.ConfiguredProviderIDs()
	if len(configured) == 0 {
		fmt.Fprintln(os.Stderr, "No providers are configured. Run 'goshcoder providers' to see the options.")
		return nil
	}
	for _, id := range configured {
		if err := printModels(c.Provider(id)); err != nil {
			return err
		}
	}
	return nil
}

func printModels(provider *catalog.Provider) error {
	models := provider.Models()
	if len(models) == 0 {
		return nil
	}
	fmt.Printf("\n%s (%s)\n", bold(provider.Name), provider.ID)
	for _, model := range models {
		supported := ""
		if _, ok := llm.GetStreamer(model.API); !ok {
			supported = dim("  [protocol not implemented]")
		}
		fmt.Printf("  %s/%s%s\n", provider.ID, model.ID, supported)
	}
	return nil
}

// ---------------------------------------------------------------------------
// auth
// ---------------------------------------------------------------------------

func authCommand(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: goshcoder auth set|login|list|logout")
	}
	store := catalog.NewFileCredentialStore(config.AuthPath())

	switch args[0] {
	case "list":
		infos, err := store.List()
		if err != nil {
			return err
		}
		if len(infos) == 0 {
			fmt.Println("No stored credentials.")
			return nil
		}
		for _, info := range infos {
			fmt.Printf("%-24s %s\n", info.ProviderID, info.Type)
		}
		return nil

	case "set":
		if len(args) < 2 {
			return errors.New("usage: goshcoder auth set <provider>")
		}
		providerID := args[1]
		if newCatalog().Provider(providerID) == nil {
			return fmt.Errorf("unknown provider %q", providerID)
		}
		if _, err := config.EnsureAgentDir(); err != nil {
			return err
		}
		key, err := readSecret(fmt.Sprintf("Enter the API key for %s: ", providerID))
		if err != nil {
			return err
		}
		if key == "" {
			return errors.New("no key provided")
		}
		if _, err := store.Modify(providerID, func(*catalog.Credential) (*catalog.Credential, error) {
			return &catalog.Credential{Type: "api_key", Key: key}, nil
		}); err != nil {
			return err
		}
		fmt.Printf("Stored an API key for %s in %s\n", providerID, config.AuthPath())
		return nil

	case "login":
		if len(args) < 2 {
			return errors.New("usage: goshcoder auth login <provider>")
		}
		providerID := args[1]
		c := catalog.NewCatalog(store)
		if c.Provider(providerID) == nil {
			return fmt.Errorf("unknown provider %q", providerID)
		}
		if _, ok := catalog.LoginProviderFor(providerID); !ok {
			return fmt.Errorf("provider %q does not have an OAuth login flow; use 'goshcoder auth set %s' if it accepts an API key", providerID, providerID)
		}
		if _, err := config.EnsureAgentDir(); err != nil {
			return err
		}

		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		visibleInput := false
		if info, err := os.Stdin.Stat(); err == nil {
			visibleInput = (info.Mode() & os.ModeCharDevice) != 0
		}
		interaction := newTerminalLoginInteraction(os.Stdin, os.Stderr, visibleInput)
		if _, err := c.Login(ctx, providerID, interaction); err != nil {
			return err
		}
		fmt.Printf("Logged in to %s; credential stored in %s\n", providerID, config.AuthPath())
		return nil

	case "logout":
		if len(args) < 2 {
			return errors.New("usage: goshcoder auth logout <provider>")
		}
		if err := store.Delete(args[1]); err != nil {
			return err
		}
		fmt.Printf("Removed the stored credential for %s\n", args[1])
		return nil

	default:
		return fmt.Errorf("unknown auth subcommand %q", args[0])
	}
}

// terminalLoginInteraction adapts OAuth flows to the line-oriented CLI.
type terminalLoginInteraction struct {
	reader       *bufio.Reader
	out          io.Writer
	visibleInput bool
}

func newTerminalLoginInteraction(in io.Reader, out io.Writer, visibleInput bool) *terminalLoginInteraction {
	return &terminalLoginInteraction{reader: bufio.NewReader(in), out: out, visibleInput: visibleInput}
}

func (i *terminalLoginInteraction) Prompt(prompt catalog.LoginPrompt) (string, error) {
	fmt.Fprintln(i.out, prompt.Message)
	if prompt.Kind == catalog.PromptSelect {
		for index, option := range prompt.Options {
			detail := option.Label
			if option.Description != "" {
				detail += " — " + option.Description
			}
			fmt.Fprintf(i.out, "  %d) %s\n", index+1, detail)
		}
		fmt.Fprint(i.out, "Choice [1]: ")
	} else {
		if prompt.Placeholder != "" {
			fmt.Fprintf(i.out, "%s\n", dim("Example: "+prompt.Placeholder))
		}
		if prompt.Kind == catalog.PromptSecret && i.visibleInput {
			fmt.Fprint(i.out, "(input is visible) ")
		} else {
			fmt.Fprint(i.out, "> ")
		}
	}

	type lineResult struct {
		line string
		err  error
	}
	result := make(chan lineResult, 1)
	go func() {
		line, err := i.reader.ReadString('\n')
		if err == io.EOF && line != "" {
			err = nil
		}
		result <- lineResult{line: strings.TrimSpace(line), err: err}
	}()

	ctx := prompt.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return "", catalog.ErrLoginCancelled
	case read := <-result:
		if read.err != nil {
			return "", read.err
		}
		if prompt.Kind != catalog.PromptSelect {
			return read.line, nil
		}
		if read.line == "" && len(prompt.Options) > 0 {
			return prompt.Options[0].ID, nil
		}
		for index, option := range prompt.Options {
			if read.line == option.ID || read.line == fmt.Sprint(index+1) {
				return option.ID, nil
			}
		}
		return "", fmt.Errorf("invalid choice %q", read.line)
	}
}

func (i *terminalLoginInteraction) Notify(event catalog.LoginEvent) {
	switch event.Kind {
	case "auth_url":
		fmt.Fprintf(i.out, "\nOpen this URL to authenticate:\n%s\n", event.URL)
		if event.Instructions != "" {
			fmt.Fprintln(i.out, event.Instructions)
		}
	case "device_code":
		fmt.Fprintf(i.out, "\nOpen %s and enter code: %s\n", event.VerificationURI, event.UserCode)
	case "info", "progress":
		if event.Message != "" {
			fmt.Fprintln(i.out, event.Message)
		}
	}
}

// readSecret prompts on stderr and reads one line from stdin. Terminal echo is
// not suppressed, so the prompt says so when stdin is a terminal.
func readSecret(prompt string) (string, error) {
	if info, err := os.Stdin.Stat(); err == nil && (info.Mode()&os.ModeCharDevice) != 0 {
		fmt.Fprint(os.Stderr, prompt+"(input is visible) ")
	}
	line, err := bufio.NewReader(os.Stdin).ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	return strings.TrimSpace(line), nil
}

// ---------------------------------------------------------------------------
// ralph
// ---------------------------------------------------------------------------

// ralphCommand manages loop state from outside a run. It is the CLI equivalent
// of the extension's /ralph slash commands.
func ralphCommand(args []string) error {
	if len(args) == 0 {
		return errors.New("usage: goshcoder ralph list|status|resume|stop|archive|delete")
	}

	root, err := filepath.Abs(".")
	if err != nil {
		return err
	}
	// Loop ownership is per session; these commands inspect and edit state
	// rather than driving a loop, so a distinct id is fine.
	loops := ralph.NewStore(root, fmt.Sprintf("cli-%d", os.Getpid()))

	switch args[0] {
	case "list":
		archived := len(args) > 1 && (args[1] == "--archived" || args[1] == "-a")
		states := loops.List(archived)
		if len(states) == 0 {
			if archived {
				fmt.Println("No archived loops.")
			} else {
				fmt.Println("No loops. Start one with 'goshcoder run -ralph ...'.")
			}
			return nil
		}
		for _, state := range states {
			fmt.Println(state.Summary())
		}
		return nil

	case "status":
		state, ok := loops.Current()
		if !ok {
			fmt.Println("No active loop in this workspace.")
			return nil
		}
		fmt.Println(state.Summary())
		fmt.Println(loops.StatusLine())
		return nil

	case "resume":
		if len(args) < 2 {
			return errors.New("usage: goshcoder ralph resume <name>")
		}
		state, err := loops.Resume(args[1])
		if err != nil {
			return err
		}
		fmt.Printf("Resumed %s at iteration %d.\n", state.Name, state.Iteration)
		return nil

	case "stop":
		if len(args) < 2 {
			return errors.New("usage: goshcoder ralph stop <name>")
		}
		state, ok := loops.Load(args[1], false)
		if !ok {
			return fmt.Errorf("no loop named %q", args[1])
		}
		if err := loops.Complete(state); err != nil {
			return err
		}
		fmt.Printf("Stopped %s at iteration %d.\n", state.Name, state.Iteration)
		return nil

	case "archive":
		if len(args) < 2 {
			return errors.New("usage: goshcoder ralph archive <name>")
		}
		if err := loops.Archive(args[1]); err != nil {
			return err
		}
		fmt.Printf("Archived %s.\n", args[1])
		return nil

	case "delete":
		if len(args) < 2 {
			return errors.New("usage: goshcoder ralph delete <name>")
		}
		if err := loops.Delete(args[1]); err != nil {
			return err
		}
		fmt.Printf("Deleted %s.\n", args[1])
		return nil

	default:
		return fmt.Errorf("unknown ralph subcommand %q", args[0])
	}
}
