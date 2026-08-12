package main

import "testing"

func TestFullscreenEditorUnicodeAndNavigation(t *testing.T) {
	editor := fullscreenEditor{}
	if action := handleFullscreenInput(&editor, []byte("你好"), false); action != "" {
		t.Fatalf("action = %q", action)
	}
	if string(editor.input) != "你好" || editor.cursor != 2 {
		t.Fatalf("editor = %q, %d", editor.input, editor.cursor)
	}
	handleFullscreenInput(&editor, []byte("\x1b[D"), false)
	handleFullscreenInput(&editor, []byte{127}, false)
	if string(editor.input) != "好" || editor.cursor != 0 {
		t.Fatalf("editor = %q, %d", editor.input, editor.cursor)
	}
}

func TestFullscreenEditorHandlesSplitUTF8(t *testing.T) {
	editor := fullscreenEditor{}
	bytes := []byte("你")
	handleFullscreenInput(&editor, bytes[:1], false)
	handleFullscreenInput(&editor, bytes[1:], false)
	if string(editor.input) != "你" {
		t.Fatalf("input = %q", editor.input)
	}
}

func TestFullscreenEditorShiftTabAction(t *testing.T) {
	if action := handleFullscreenInput(&fullscreenEditor{}, []byte("\x1b[Z"), false); action != "thinking" {
		t.Fatalf("action = %q", action)
	}
}

func TestFullscreenEditorHistory(t *testing.T) {
	editor := fullscreenEditor{history: []string{"first", "second"}, historyIndex: -1, input: []rune("draft")}
	editor.cursor = len(editor.input)
	editorHistory(&editor, -1)
	if string(editor.input) != "second" {
		t.Fatalf("input = %q", editor.input)
	}
	editorHistory(&editor, 1)
	if string(editor.input) != "draft" {
		t.Fatalf("input = %q", editor.input)
	}
}
