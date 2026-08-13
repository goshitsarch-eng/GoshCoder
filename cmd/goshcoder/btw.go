package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"goshcoder/internal/btw"
	"goshcoder/internal/config"
	"goshcoder/internal/llm"
)

type btwSelection struct {
	model    *llm.Model
	thinking llm.ModelThinkingLevel
	remember bool
}

func (s *session) resolveBTWSelection() (btwSelection, []string, error) {
	state := s.agent.State()
	current := state.Model
	thinking := state.ThinkingLevel
	settings := btw.ReadSettings(config.BTWPath())
	var warnings []string
	loaded := btw.Settings{}
	switch settings.Kind {
	case "loaded":
		loaded = settings.Settings
	case "invalid":
		warnings = append(warnings, "pi-btw settings ignored: "+settings.Reason)
	}
	selected := &current
	if loaded.Model != "" {
		if candidate, _, err := newCatalog().ResolveModel(loaded.Model); err == nil {
			selected = candidate
		} else {
			warnings = append(warnings, fmt.Sprintf("pi-btw model %s is unavailable (%v); falling back to %s/%s", loaded.Model, err, current.Provider, current.ID))
		}
	}
	if loaded.ThinkingLevel != "" {
		thinking = loaded.ThinkingLevel
	}
	thinking = llm.ClampThinkingLevel(selected, thinking)
	return btwSelection{model: selected, thinking: thinking, remember: btw.EffectiveRemember(loaded)}, warnings, nil
}

func (s *session) askBTW(ctx context.Context, thread *btw.Thread, question string, thinking llm.ModelThinkingLevel) (string, error) {
	selection, _, err := s.resolveBTWSelection()
	if err != nil {
		return "", err
	}
	if thinking == "" {
		thinking = selection.thinking
	}
	thinking = llm.ClampThinkingLevel(selection.model, thinking)
	s.btw.SetThinkingLevel(thread, thinking)
	response, err := btw.Ask(ctx, thread, selection.model, question, thinking, func(model *llm.Model, request *llm.Context, options *llm.SimpleStreamOptions) (*llm.AssistantMessage, error) {
		stream := s.streamAuthenticated(model, request, options)
		result, err := stream.Result(ctx)
		if err != nil {
			return nil, err
		}
		if result == nil {
			return nil, errors.New("side model returned no response")
		}
		if result.StopReason == llm.StopAborted {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			return nil, context.Canceled
		}
		if result.StopReason == llm.StopError {
			return nil, errors.New(result.ErrorMessage)
		}
		return result, nil
	})
	if err != nil {
		if !errors.Is(err, context.Canceled) {
			s.btw.RecordError(thread, question, err.Error())
		}
		return "", err
	}
	return s.btw.RecordAnswered(thread, question, *response), nil
}

func (s *session) handleBTWCommand(rest string) error {
	fields := strings.Fields(rest)
	if len(fields) == 0 {
		summaries := s.btw.List()
		fmt.Fprintln(os.Stderr, "BTW side threads (in memory only):")
		fmt.Fprintln(os.Stderr, "  /btw <question>                  start a fresh side thread")
		fmt.Fprintln(os.Stderr, "  /btw resume <id> <question>      continue one")
		fmt.Fprintln(os.Stderr, "  /btw settings [level|remember]   view/change preferences")
		for _, summary := range summaries {
			fmt.Fprintf(os.Stderr, "  %s  %d question(s)  %s\n", summary.ID, summary.Questions, summary.Title)
		}
		return nil
	}
	switch strings.ToLower(fields[0]) {
	case "list":
		return s.handleBTWCommand("")
	case "resume":
		if len(fields) < 3 {
			return errors.New("usage: /btw resume <thread-id> <question>")
		}
		thread := s.btw.Get(fields[1])
		if thread == nil {
			return fmt.Errorf("unknown BTW thread %q", fields[1])
		}
		question := strings.TrimSpace(strings.Join(fields[2:], " "))
		answer, err := s.askBTW(context.Background(), thread, question, thread.ThinkingLevel)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "\nBTW · %s\n%s\n", thread.ID, answer)
		return nil
	case "settings":
		return handleBTWSettings(fields[1:])
	case "bring":
		if len(fields) < 2 {
			return errors.New("usage: /btw bring <thread-id> [latest|all|from:N]")
		}
		thread := s.btw.Get(fields[1])
		if thread == nil {
			return fmt.Errorf("unknown BTW thread %q", fields[1])
		}
		segments := btw.LatestSegments(thread)
		if len(fields) > 2 {
			scope := strings.ToLower(fields[2])
			if scope == "all" {
				segments = btw.AnsweredSegments(thread, 0)
			} else if strings.HasPrefix(scope, "from:") {
				index, err := strconv.Atoi(strings.TrimPrefix(scope, "from:"))
				if err != nil || index < 1 {
					return errors.New("from scope must be from:N with N >= 1")
				}
				segments = btw.AnsweredSegments(thread, index-1)
			}
		}
		if len(segments) == 0 {
			return errors.New("the side thread has no successful answer")
		}
		fmt.Fprintln(os.Stderr, btw.FormatBringToMain(segments))
		return nil
	default:
		selection, _, _ := s.resolveBTWSelection()
		thread := s.btw.New(btw.BuildConversationContext(s.agent.State().Messages), selection.thinking)
		answer, err := s.askBTW(context.Background(), thread, rest, selection.thinking)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "\nBTW · %s\n%s\n", thread.ID, answer)
		return nil
	}
}

func handleBTWSettings(args []string) error {
	loaded := btw.ReadSettings(config.BTWPath())
	if len(args) == 0 {
		if loaded.Kind == "invalid" {
			return errors.New(loaded.Reason)
		}
		settings := loaded.Settings
		remember := btw.EffectiveRemember(settings)
		model := settings.Model
		if model == "" {
			model = "current session model"
		}
		level := settings.ThinkingLevel
		if level == "" {
			level = "current session level"
		}
		fmt.Fprintf(os.Stderr, "pi-btw settings (%s)\n  model: %s\n  thinking: %s\n  remember changes: %t\n", config.BTWPath(), model, level, remember)
		return nil
	}
	if strings.EqualFold(args[0], "remember") {
		if len(args) != 2 {
			return errors.New("usage: /btw settings remember <on|off>")
		}
		value := strings.EqualFold(args[1], "on")
		if !value && !strings.EqualFold(args[1], "off") {
			return errors.New("remember must be on or off")
		}
		_, err := btw.UpdateSettings(config.BTWPath(), nil, &value)
		return err
	}
	level := llm.ModelThinkingLevel(strings.ToLower(args[0]))
	_, err := btw.UpdateSettings(config.BTWPath(), &level, nil)
	return err
}
