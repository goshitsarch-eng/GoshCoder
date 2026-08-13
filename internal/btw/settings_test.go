package btw

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"goshcoder/internal/llm"
)

func TestSettingsMissingReadIsSideEffectFreeAndUpdatePreservesUnknown(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pi-btw.json")
	if got := ReadSettings(path); got.Kind != "missing" {
		t.Fatalf("result = %#v", got)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("missing read created file: %v", err)
	}
	if err := os.WriteFile(path, []byte(`{"model":"openrouter/a/b","unknown":{"keep":true}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	level := llm.ModelThinkingLevel("high")
	remember := false
	settings, err := UpdateSettings(path, &level, &remember)
	if err != nil {
		t.Fatal(err)
	}
	if settings.Model != "openrouter/a/b" || settings.ThinkingLevel != "high" || EffectiveRemember(settings) {
		t.Fatalf("settings = %#v", settings)
	}
	data, _ := os.ReadFile(path)
	if !strings.Contains(string(data), `"unknown"`) {
		t.Fatalf("unknown field lost: %s", data)
	}
}

func TestSettingsInvalidFileBlocksSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pi-btw.json")
	original := []byte(`{"thinkingLevel":"invalid"}`)
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	level := llm.ModelThinkingLevel("low")
	if _, err := UpdateSettings(path, &level, nil); err == nil {
		t.Fatal("invalid file was overwritten")
	}
	after, _ := os.ReadFile(path)
	if string(after) != string(original) {
		t.Fatalf("file changed to %s", after)
	}
}

func TestSettingsRejectsOversizedAndInvalidUTF8(t *testing.T) {
	for name, data := range map[string][]byte{"large": []byte(strings.Repeat("x", MaxSettingsBytes+1)), "utf8": {0xff}} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "pi-btw.json")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			if result := ReadSettings(path); result.Kind != "invalid" {
				t.Fatalf("result = %#v", result)
			}
		})
	}
}
