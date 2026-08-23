package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"goshcoder/internal/config"
	"goshcoder/internal/llm/catalog"
	"goshcoder/internal/omniroute"
)

func omniCommand(args []string) error {
	output, err := runOmniCommand(context.Background(), args, true)
	if output != "" {
		fmt.Fprintln(os.Stdout, output)
	}
	return err
}

func runOmniCommand(ctx context.Context, args []string, interactive bool) (string, error) {
	// A URL argument is used by the fullscreen two-step wizard. The API key is
	// already saved independently through `auth set omni` by that point.
	subcommand := "status"
	if len(args) > 0 && strings.TrimSpace(args[0]) != "" {
		subcommand = strings.ToLower(args[0])
	}
	switch subcommand {
	case "status":
		configured, err := omniroute.Load(config.OmniRoutePath())
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return "OmniRoute is unconfigured. Run /omni setup.", nil
			}
			return "", err
		}
		client, err := omniClient(configured)
		if err != nil {
			return "", err
		}
		probeCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		health := "healthy"
		if err := client.Health(probeCtx); err != nil {
			health = "DOWN (" + err.Error() + ")"
		}
		return fmt.Sprintf("OmniRoute: %s\nServer: %s\nModels: %d\nDashboard: %s", health, configured.ServerURL, len(configured.Models), configured.Dashboard()), nil
	case "dashboard", "dash":
		configured, err := omniroute.Load(config.OmniRoutePath())
		if err != nil {
			return "", fmt.Errorf("OmniRoute is unconfigured; run /omni setup: %w", err)
		}
		return "OmniRoute Dashboard: " + configured.Dashboard() + "\nOpen it to manage combos, providers, usage, and request logs.", nil
	case "sync":
		configured, err := omniroute.Load(config.OmniRoutePath())
		if err != nil {
			return "", fmt.Errorf("OmniRoute is unconfigured; run /omni setup: %w", err)
		}
		client, err := omniClient(configured)
		if err != nil {
			return "", err
		}
		models, err := client.FetchModels(ctx)
		if err != nil {
			return "", fmt.Errorf("sync OmniRoute models: %w", err)
		}
		old := len(configured.Models)
		configured.Models, configured.SyncedAt = models, time.Now().UnixMilli()
		if err := omniroute.Save(config.OmniRoutePath(), configured); err != nil {
			return "", err
		}
		return fmt.Sprintf("Synced %d OmniRoute models (was %d). They are available under /model immediately.", len(models), old), nil
	case "setup":
		if !interactive {
			return "", errors.New("OmniRoute setup requires an interactive terminal")
		}
		fmt.Fprintf(os.Stderr, "OmniRoute URL [%s]: ", omniroute.DefaultServerURL)
		urlValue, err := readTerminalLine(os.Stdin)
		if err != nil {
			return "", err
		}
		key, err := readSecret("OmniRoute API key (blank for local/public): ")
		if err != nil {
			return "", err
		}
		return runOmniSetup(ctx, urlValue, key, true)
	default:
		return "", fmt.Errorf("unknown OmniRoute command %q; use status, sync, setup, or dashboard", subcommand)
	}
}

func runOmniSetup(ctx context.Context, urlValue, key string, allowDefault bool) (string, error) {
	urlValue = strings.TrimSpace(urlValue)
	if urlValue == "" && allowDefault {
		urlValue = omniroute.DefaultServerURL
	}
	normalized, err := omniroute.NormalizeServerURL(urlValue)
	if err != nil {
		return "", err
	}
	key = strings.TrimSpace(key)
	candidate := omniroute.Config{ServerURL: normalized}
	if current, loadErr := omniroute.Load(config.OmniRoutePath()); loadErr == nil && current.ServerURL == normalized {
		candidate.Models, candidate.SyncedAt = current.Models, current.SyncedAt
	}
	client := omniroute.Client{Config: candidate, APIKey: key}
	probeCtx, cancel := context.WithTimeout(ctx, 4*time.Second)
	defer cancel()
	if err := client.Health(probeCtx); err != nil {
		return "", fmt.Errorf("OmniRoute unreachable at %s: %w", normalized, err)
	}
	store := catalog.NewFileCredentialStore(config.AuthPath())
	stored := key
	if stored == "" {
		stored = "omniroute-public"
	}
	if _, err := store.Modify("omni", func(*catalog.Credential) (*catalog.Credential, error) {
		return &catalog.Credential{Type: "api_key", Key: stored}, nil
	}); err != nil {
		return "", err
	}
	if err := omniroute.Save(config.OmniRoutePath(), candidate); err != nil {
		return "", err
	}
	return "OmniRoute setup complete. Run /omni sync to load models.", nil
}

func omniClient(configured omniroute.Config) (omniroute.Client, error) {
	auth, ok := newCatalog().ResolveAuth("omni")
	if !ok {
		return omniroute.Client{}, errors.New("OmniRoute credentials are missing; run /omni setup")
	}
	return omniroute.Client{Config: configured, APIKey: auth.APIKey}, nil
}
