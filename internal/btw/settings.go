package btw

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"goshcoder/internal/llm"
)

const MaxSettingsBytes = 64 << 10

type Settings struct {
	Model                        string                 `json:"model,omitempty"`
	ThinkingLevel                llm.ModelThinkingLevel `json:"thinkingLevel,omitempty"`
	RememberThinkingLevelChanges *bool                  `json:"rememberThinkingLevelChanges,omitempty"`
}

type SettingsResult struct {
	Kind     string
	Settings Settings
	Reason   string
}

var settingsLocks sync.Map

func EffectiveRemember(settings Settings) bool {
	return settings.RememberThinkingLevelChanges == nil || *settings.RememberThinkingLevelChanges
}

func ParseModelReference(reference string) (provider, model string, ok bool) {
	if strings.TrimSpace(reference) != reference || strings.ContainsAny(reference, "\t\r\n") {
		return "", "", false
	}
	at := strings.IndexByte(reference, '/')
	if at <= 0 || at == len(reference)-1 {
		return "", "", false
	}
	return reference[:at], reference[at+1:], true
}

func validThinking(value llm.ModelThinkingLevel) bool {
	switch value {
	case "", llm.ThinkingOff, llm.ThinkingMinimal, llm.ThinkingLow, llm.ThinkingMedium, llm.ThinkingHigh, llm.ThinkingXHigh, llm.ThinkingMax:
		return true
	}
	return false
}

func ReadSettings(path string) SettingsResult {
	lock := pathLock(path)
	lock.Lock()
	defer lock.Unlock()
	settings, _, err := readSettingsDocument(path)
	if errors.Is(err, os.ErrNotExist) {
		return SettingsResult{Kind: "missing"}
	}
	if err != nil {
		return SettingsResult{Kind: "invalid", Reason: err.Error()}
	}
	return SettingsResult{Kind: "loaded", Settings: settings}
}

// UpdateSettings preserves model and unknown fields and atomically applies the
// supported preference patch. Invalid existing files are never overwritten.
func UpdateSettings(path string, thinking *llm.ModelThinkingLevel, remember *bool) (Settings, error) {
	lock := pathLock(path)
	lock.Lock()
	defer lock.Unlock()
	settings, document, err := readSettingsDocument(path)
	if errors.Is(err, os.ErrNotExist) {
		document = map[string]json.RawMessage{}
		settings = Settings{}
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return Settings{}, fmt.Errorf("pi-btw settings at %s are invalid: %w", path, err)
	}
	if thinking != nil {
		if !validThinking(*thinking) || *thinking == "" {
			return Settings{}, fmt.Errorf("invalid thinking level %q", *thinking)
		}
		settings.ThinkingLevel = *thinking
		encoded, _ := json.Marshal(*thinking)
		document["thinkingLevel"] = encoded
	}
	if remember != nil {
		settings.RememberThinkingLevelChanges = remember
		encoded, _ := json.Marshal(*remember)
		document["rememberThinkingLevelChanges"] = encoded
	}
	if err := publishSettings(path, document); err != nil {
		return Settings{}, err
	}
	return settings, nil
}

func readSettingsDocument(path string) (Settings, map[string]json.RawMessage, error) {
	file, err := os.Open(path)
	if err != nil {
		return Settings{}, nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return Settings{}, nil, err
	}
	if !info.Mode().IsRegular() {
		return Settings{}, nil, errors.New("settings path is not a regular file")
	}
	if info.Size() > MaxSettingsBytes {
		return Settings{}, nil, fmt.Errorf("settings file exceeds %d bytes", MaxSettingsBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, MaxSettingsBytes+1))
	if err != nil {
		return Settings{}, nil, err
	}
	if len(data) > MaxSettingsBytes {
		return Settings{}, nil, fmt.Errorf("settings file exceeds %d bytes", MaxSettingsBytes)
	}
	if !utf8.Valid(data) {
		return Settings{}, nil, errors.New("settings file is not valid UTF-8")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	var document map[string]json.RawMessage
	if err := decoder.Decode(&document); err != nil {
		return Settings{}, nil, errors.New("invalid JSON")
	}
	if document == nil {
		return Settings{}, nil, errors.New("invalid settings shape")
	}
	var settings Settings
	if raw, ok := document["model"]; ok {
		if json.Unmarshal(raw, &settings.Model) != nil {
			return Settings{}, nil, errors.New("invalid model")
		}
		if _, _, ok := ParseModelReference(settings.Model); !ok {
			return Settings{}, nil, errors.New("invalid model reference")
		}
	}
	if raw, ok := document["thinkingLevel"]; ok {
		if json.Unmarshal(raw, &settings.ThinkingLevel) != nil || !validThinking(settings.ThinkingLevel) || settings.ThinkingLevel == "" {
			return Settings{}, nil, errors.New("invalid thinkingLevel")
		}
	}
	if raw, ok := document["rememberThinkingLevelChanges"]; ok {
		var value bool
		if json.Unmarshal(raw, &value) != nil {
			return Settings{}, nil, errors.New("invalid rememberThinkingLevelChanges")
		}
		settings.RememberThinkingLevelChanges = &value
	}
	return settings, document, nil
}

func publishSettings(path string, document map[string]json.RawMessage) error {
	data, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if len(data) > MaxSettingsBytes {
		return fmt.Errorf("settings document exceeds %d bytes", MaxSettingsBytes)
	}
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0o700); err != nil {
		return err
	}
	file, err := os.CreateTemp(directory, ".pi-btw-*.tmp")
	if err != nil {
		return err
	}
	name := file.Name()
	defer os.Remove(name)
	if err := file.Chmod(0o600); err != nil {
		file.Close()
		return err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(name, path)
}

func pathLock(path string) *sync.Mutex {
	value, _ := settingsLocks.LoadOrStore(path, &sync.Mutex{})
	return value.(*sync.Mutex)
}
