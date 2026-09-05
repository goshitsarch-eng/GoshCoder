use std::{
    cell::Cell,
    time::{Duration, Instant},
};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// A transcript item rendered by the terminal interface.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub title: String,
    pub text: String,
    pub detail: String,
    pub is_error: bool,
}

// These variants are the stable view-model contract for runtime modules that
// are still being ported. Some are not populated by the initial shell yet.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MessageRole {
    User,
    #[default]
    Assistant,
    Thinking,
    Tool,
    Error,
    Notice,
    Command,
}

/// A completion item in the command palette.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Suggestion {
    pub label: String,
    pub description: String,
    pub value: String,
    pub execute: bool,
}

/// The action requested by a key press. Runtime code owns side effects; this
/// state machine only owns editor and presentation behavior.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action {
    None,
    Quit,
    Submit(String),
    FollowUp(String),
    Abort,
    /// Ask the runtime to choose the next or previous available model.
    CycleModel {
        direction: i8,
    },
    /// Ask the runtime to advance through levels supported by the active model.
    CycleThinking,
}

/// State shared by the Ratatui renderer and terminal event loop.
///
/// Keeping the editor state independent of provider and session code lets the
/// Rust runtime replace the previous Bubble Tea event loop without coupling UI
/// behavior to networking or persistence.
#[derive(Debug)]
pub struct App {
    pub title: String,
    pub messages: Vec<Message>,
    pub sidebar: Vec<SidebarLine>,
    pub input: String,
    /// Byte offset at a UTF-8 character boundary.
    pub cursor: usize,
    pub status: String,
    pub streaming: bool,
    /// True when the active transcript is safely durable. A recorded session
    /// can exit immediately on Ctrl-C; an in-memory transcript retains the
    /// two-press safeguard from the previous fullscreen interface.
    pub recording_active: bool,
    pub scroll: u16,
    pub selected_suggestion: usize,
    pub tools_expanded: bool,
    pub hide_thinking: bool,
    pub history: Vec<String>,
    /// Suggestions the runtime supplies for an argument palette (`/model `,
    /// `/thinking `, `/login `); the static command list covers the rest.
    pub dynamic_suggestions: Vec<Suggestion>,
    /// The largest useful scroll offset at the last draw. The renderer
    /// writes it so key handling can clamp instead of letting the offset
    /// run past the top of the transcript.
    pub last_max_scroll: Cell<u16>,
    history_index: Option<usize>,
    draft: String,
    quit_armed_at: Option<Instant>,
}

/// A typed line in the responsive sidebar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SidebarLine {
    pub kind: SidebarKind,
    pub value: String,
}

// The sidebar accepts every status rendered by the previous terminal UI; the
// early Ratatui shell only populates the fields it currently owns.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SidebarKind {
    Title,
    Section,
    Accent,
    Active,
    Meta,
    Path,
    Brand,
    Progress(u8),
    Todo { complete: bool },
    File { status: FileStatus },
    Blank,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Untracked,
}

impl App {
    pub fn new() -> Self {
        Self {
            title: "interactive session".to_owned(),
            messages: vec![Message {
                role: MessageRole::Notice,
                text: "Rust/Ratatui migration is initializing. The terminal UI is active while runtime features are ported."
                    .to_owned(),
                ..Message::default()
            }],
            sidebar: vec![
                SidebarLine::title("New Session"),
                SidebarLine::accent("goshcoder"),
                SidebarLine::meta("off thinking · normal"),
                SidebarLine::blank(),
                SidebarLine::section("Context"),
                SidebarLine::progress(0),
                SidebarLine::meta("0 / 0 tokens"),
                SidebarLine::meta("0% used · $0.0000 spent"),
                SidebarLine::blank(),
                SidebarLine::section("Workspace"),
                SidebarLine::meta("Rust migration"),
                SidebarLine::path("."),
                SidebarLine::blank(),
                SidebarLine::brand("● GoshCoder"),
            ],
            input: String::new(),
            cursor: 0,
            status: "Ready".to_owned(),
            streaming: false,
            recording_active: false,
            scroll: 0,
            selected_suggestion: 0,
            tools_expanded: false,
            hide_thinking: false,
            history: Vec::new(),
            dynamic_suggestions: Vec::new(),
            last_max_scroll: Cell::new(0),
            history_index: None,
            draft: String::new(),
            quit_armed_at: None,
        }
    }

    pub fn suggestions(&self) -> Vec<Suggestion> {
        if let Some(argument) = dynamic_palette_argument(&self.input) {
            let query = argument.to_lowercase();
            return self
                .dynamic_suggestions
                .iter()
                .filter(|suggestion| {
                    query.is_empty() || suggestion.label.to_lowercase().contains(&query)
                })
                .cloned()
                .collect();
        }
        suggestions_for(&self.input)
    }

    /// Scrolls the transcript towards older content, never past its top.
    pub fn scroll_up(&mut self, rows: u16) {
        self.scroll = self
            .scroll
            .saturating_add(rows)
            .min(self.last_max_scroll.get());
    }

    /// Scrolls the transcript towards the newest content.
    pub fn scroll_down(&mut self, rows: u16) {
        self.scroll = self.scroll.saturating_sub(rows);
    }

    /// Clears the composer and remembers a submitted value without adding
    /// placeholder transcript messages. A live runtime owns transcript rows
    /// through its agent state, so the same input cannot be rendered twice.
    pub fn record_submission(&mut self, prompt: &str) {
        self.history.push(prompt.to_owned());
        self.history_index = None;
        self.draft.clear();
        self.input.clear();
        self.cursor = 0;
        self.selected_suggestion = 0;
    }

    /// Retains an externally generated transcript while keeping editor history
    /// and selection state consistent with a completed submission.
    pub fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Sets whether Ctrl-C can safely exit an already-recorded session.
    pub fn set_recording_active(&mut self, recording_active: bool) {
        self.recording_active = recording_active;
        if recording_active {
            self.clear_quit_arm();
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Action {
        let modifiers = key.modifiers;
        if key.code != KeyCode::Char('c') || !modifiers.contains(KeyModifiers::CONTROL) {
            self.clear_quit_arm();
        }

        match (key.code, modifiers) {
            (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.handle_ctrl_c()
            }
            (KeyCode::Char('d'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                if self.input.is_empty() {
                    if self.streaming {
                        self.status = "Aborting".to_owned();
                        Action::Abort
                    } else {
                        Action::Quit
                    }
                } else {
                    self.delete_at_cursor();
                    Action::None
                }
            }
            (KeyCode::Esc, _) => {
                if self.streaming {
                    self.status = "Aborting".to_owned();
                    Action::Abort
                } else if !self.input.is_empty() {
                    self.clear_input();
                    Action::None
                } else {
                    Action::None
                }
            }
            // Real terminals report Shift+Tab as BackTab; a synthetic
            // Tab+SHIFT is accepted for callers that build events directly.
            (KeyCode::BackTab, _) => Action::CycleThinking,
            (KeyCode::Tab, modifiers) if modifiers.contains(KeyModifiers::SHIFT) => {
                Action::CycleThinking
            }
            // Shift/Ctrl+Enter need the kitty keyboard protocol; Ctrl-J is
            // the newline every legacy terminal can deliver.
            (KeyCode::Char('j'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.insert("\n");
                Action::None
            }
            (KeyCode::Char('l'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.set_input("/model ");
                Action::None
            }
            (KeyCode::Char('p'), modifiers)
                if modifiers.contains(KeyModifiers::CONTROL)
                    && modifiers.contains(KeyModifiers::SHIFT) =>
            {
                Action::CycleModel { direction: -1 }
            }
            (KeyCode::Char('p'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                Action::CycleModel { direction: 1 }
            }
            (KeyCode::Char('o'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.tools_expanded = !self.tools_expanded;
                self.status = if self.tools_expanded {
                    "Tool output expanded"
                } else {
                    "Tool output collapsed"
                }
                .to_owned();
                Action::None
            }
            (KeyCode::Char('t'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.hide_thinking = !self.hide_thinking;
                self.status = if self.hide_thinking {
                    "Thinking collapsed"
                } else {
                    "Thinking expanded"
                }
                .to_owned();
                Action::None
            }
            (KeyCode::Enter, modifiers) if modifiers.contains(KeyModifiers::ALT) => {
                self.request_submission(true)
            }
            (KeyCode::Up, _) => self.handle_up(),
            (KeyCode::Down, _) => self.handle_down(),
            (KeyCode::Left, modifiers)
                if modifiers.contains(KeyModifiers::CONTROL)
                    || modifiers.contains(KeyModifiers::ALT) =>
            {
                self.move_word(-1);
                Action::None
            }
            (KeyCode::Right, modifiers)
                if modifiers.contains(KeyModifiers::CONTROL)
                    || modifiers.contains(KeyModifiers::ALT) =>
            {
                self.move_word(1);
                Action::None
            }
            (KeyCode::Char('b'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_left();
                Action::None
            }
            (KeyCode::Char('f'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.move_right();
                Action::None
            }
            (KeyCode::Left, _) => {
                self.move_left();
                Action::None
            }
            (KeyCode::Right, _) => {
                self.move_right();
                Action::None
            }
            (KeyCode::Home, _) => {
                self.cursor = line_start(&self.input, self.cursor);
                Action::None
            }
            (KeyCode::Char('a'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = line_start(&self.input, self.cursor);
                Action::None
            }
            (KeyCode::End, _) => {
                self.cursor = line_end(&self.input, self.cursor);
                Action::None
            }
            (KeyCode::Char('e'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.cursor = line_end(&self.input, self.cursor);
                Action::None
            }
            (KeyCode::PageUp, _) => {
                self.scroll_up(10);
                Action::None
            }
            (KeyCode::PageDown, _) => {
                self.scroll_down(10);
                Action::None
            }
            (KeyCode::Enter, modifiers)
                if modifiers.contains(KeyModifiers::SHIFT)
                    || modifiers.contains(KeyModifiers::CONTROL) =>
            {
                self.insert("\n");
                Action::None
            }
            (KeyCode::Backspace, _) => {
                self.backspace();
                Action::None
            }
            (KeyCode::Delete, _) => {
                self.delete_at_cursor();
                Action::None
            }
            (KeyCode::Char('k'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.truncate(self.cursor);
                self.selected_suggestion = 0;
                Action::None
            }
            (KeyCode::Char('u'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.drain(..self.cursor);
                self.cursor = 0;
                self.selected_suggestion = 0;
                Action::None
            }
            (KeyCode::Char('w'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.delete_previous_word();
                Action::None
            }
            (KeyCode::Tab, _) => {
                let suggestions = self.suggestions();
                if let Some(item) = suggestions.get(self.clamped_suggestion(suggestions.len())) {
                    self.set_input(&item.value);
                }
                Action::None
            }
            (KeyCode::Enter, _) => self.handle_enter(),
            (KeyCode::Char(character), modifiers)
                if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.insert(&character.to_string());
                Action::None
            }
            _ => Action::None,
        }
    }

    pub fn paste(&mut self, text: &str) {
        // Terminals deliver pasted line breaks as CR; keep them as newlines
        // instead of filtering them out with the other control characters.
        let text = text
            .replace("\r\n", "\n")
            .replace('\r', "\n")
            .replace('\t', "    ");
        let sanitized: String = crate::markdown::sanitize_terminal_text(&text)
            .chars()
            .filter(|character| !character.is_control() || *character == '\n')
            .collect();
        self.insert(&sanitized);
    }

    pub fn set_input(&mut self, input: &str) {
        self.input = input.to_owned();
        self.cursor = self.input.len();
        self.selected_suggestion = 0;
    }

    fn handle_ctrl_c(&mut self) -> Action {
        if !self.input.is_empty() {
            self.clear_input();
            return Action::None;
        }
        if self.streaming {
            self.status = "Aborting".to_owned();
            return Action::Abort;
        }
        if self.recording_active {
            return Action::Quit;
        }
        if self
            .quit_armed_at
            .is_some_and(|armed_at| armed_at.elapsed() < Duration::from_secs(3))
        {
            return Action::Quit;
        }
        self.quit_armed_at = Some(Instant::now());
        self.status = "Press Ctrl+C again to exit (this session is not being saved)".to_owned();
        Action::None
    }

    fn handle_up(&mut self) -> Action {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected_suggestion = self.selected_suggestion.saturating_sub(1);
            return Action::None;
        }
        if !self.move_vertical(-1) {
            self.history(-1);
        }
        Action::None
    }

    fn handle_down(&mut self) -> Action {
        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            self.selected_suggestion = (self.selected_suggestion + 1).min(suggestions.len() - 1);
            return Action::None;
        }
        if !self.move_vertical(1) {
            self.history(1);
        }
        Action::None
    }

    fn handle_enter(&mut self) -> Action {
        let suggestions = self.suggestions();
        if let Some(item) = suggestions.get(self.clamped_suggestion(suggestions.len())) {
            self.set_input(&item.value);
            if !item.execute {
                return Action::None;
            }
        }
        self.request_submission(false)
    }

    fn request_submission(&mut self, follow_up: bool) -> Action {
        let prompt = self.input.trim().to_owned();
        if prompt.is_empty() {
            return Action::None;
        }
        if follow_up && self.streaming {
            Action::FollowUp(prompt)
        } else {
            Action::Submit(prompt)
        }
    }

    fn clamped_suggestion(&self, count: usize) -> usize {
        self.selected_suggestion.min(count.saturating_sub(1))
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.scroll = 0;
        self.selected_suggestion = 0;
    }

    fn clear_quit_arm(&mut self) {
        self.quit_armed_at = None;
        if self.status.starts_with("Press Ctrl+C again") {
            self.status = "Ready".to_owned();
        }
    }

    fn insert(&mut self, text: &str) {
        self.input.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.selected_suggestion = 0;
    }

    fn move_left(&mut self) {
        if let Some((index, _)) = self.input[..self.cursor].char_indices().next_back() {
            self.cursor = index;
        }
    }

    fn move_right(&mut self) {
        if self.cursor < self.input.len() {
            let width = self.input[self.cursor..]
                .chars()
                .next()
                .expect("cursor always stays at a character boundary")
                .len_utf8();
            self.cursor += width;
        }
    }

    fn move_word(&mut self, direction: i8) {
        if direction < 0 {
            while self.cursor > 0
                && self.input[..self.cursor]
                    .chars()
                    .next_back()
                    .is_some_and(char::is_whitespace)
            {
                self.move_left();
            }
            while self.cursor > 0
                && self.input[..self.cursor]
                    .chars()
                    .next_back()
                    .is_some_and(|character| !character.is_whitespace())
            {
                self.move_left();
            }
        } else {
            while self.cursor < self.input.len()
                && self.input[self.cursor..]
                    .chars()
                    .next()
                    .is_some_and(|character| !character.is_whitespace())
            {
                self.move_right();
            }
            while self.cursor < self.input.len()
                && self.input[self.cursor..]
                    .chars()
                    .next()
                    .is_some_and(char::is_whitespace)
            {
                self.move_right();
            }
        }
    }

    fn move_vertical(&mut self, direction: i8) -> bool {
        if !self.input.contains('\n') {
            return false;
        }
        let start = line_start(&self.input, self.cursor);
        let column = self.input[start..self.cursor].chars().count();
        if direction < 0 {
            if start == 0 {
                return false;
            }
            let previous_end = start - 1;
            let previous_start = line_start(&self.input, previous_end);
            self.cursor = byte_at_character(&self.input, previous_start, column).min(previous_end);
            true
        } else {
            let end = line_end(&self.input, self.cursor);
            if end == self.input.len() {
                return false;
            }
            let next_start = end + 1;
            let next_end = line_end(&self.input, next_start);
            self.cursor = byte_at_character(&self.input, next_start, column).min(next_end);
            true
        }
    }

    fn history(&mut self, direction: i8) {
        if self.history.is_empty() {
            return;
        }
        match (self.history_index, direction) {
            (None, -1) => {
                self.draft = self.input.clone();
                self.history_index = Some(self.history.len() - 1);
            }
            (None, _) => return,
            (Some(0), -1) => {}
            (Some(index), -1) => self.history_index = Some(index - 1),
            (Some(index), 1) if index + 1 >= self.history.len() => {
                self.history_index = None;
                self.set_input(&self.draft.clone());
                return;
            }
            (Some(index), 1) => self.history_index = Some(index + 1),
            _ => {}
        }
        if let Some(index) = self.history_index {
            self.set_input(&self.history[index].clone());
        }
    }

    fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let previous = self.input[..self.cursor]
            .char_indices()
            .next_back()
            .map(|(index, _)| index)
            .expect("cursor always stays at a character boundary");
        self.input.drain(previous..self.cursor);
        self.cursor = previous;
        self.selected_suggestion = 0;
    }

    fn delete_at_cursor(&mut self) {
        if self.cursor >= self.input.len() {
            return;
        }
        let next = self.cursor
            + self.input[self.cursor..]
                .chars()
                .next()
                .expect("cursor always stays at a character boundary")
                .len_utf8();
        self.input.drain(self.cursor..next);
        self.selected_suggestion = 0;
    }

    fn delete_previous_word(&mut self) {
        let end = self.cursor;
        self.move_word(-1);
        self.input.drain(self.cursor..end);
        self.selected_suggestion = 0;
    }
}

impl SidebarLine {
    pub fn title(value: impl Into<String>) -> Self {
        Self {
            kind: SidebarKind::Title,
            value: value.into(),
        }
    }

    pub fn section(value: impl Into<String>) -> Self {
        Self {
            kind: SidebarKind::Section,
            value: value.into(),
        }
    }

    pub fn accent(value: impl Into<String>) -> Self {
        Self {
            kind: SidebarKind::Accent,
            value: value.into(),
        }
    }

    pub fn meta(value: impl Into<String>) -> Self {
        Self {
            kind: SidebarKind::Meta,
            value: value.into(),
        }
    }

    pub fn path(value: impl Into<String>) -> Self {
        Self {
            kind: SidebarKind::Path,
            value: value.into(),
        }
    }

    pub fn brand(value: impl Into<String>) -> Self {
        Self {
            kind: SidebarKind::Brand,
            value: value.into(),
        }
    }

    pub fn progress(percent: u8) -> Self {
        Self {
            kind: SidebarKind::Progress(percent),
            value: String::new(),
        }
    }

    pub fn blank() -> Self {
        Self {
            kind: SidebarKind::Blank,
            value: String::new(),
        }
    }
}

fn line_start(input: &str, cursor: usize) -> usize {
    input[..cursor].rfind('\n').map_or(0, |index| index + 1)
}

fn line_end(input: &str, cursor: usize) -> usize {
    input[cursor..]
        .find('\n')
        .map_or(input.len(), |index| cursor + index)
}

fn byte_at_character(input: &str, start: usize, character_offset: usize) -> usize {
    input[start..]
        .char_indices()
        .nth(character_offset)
        .map_or(input.len(), |(index, _)| start + index)
}

/// The commands whose argument the runtime completes, and the argument typed
/// so far. `None` for everything else, including the bare command.
pub fn dynamic_palette_argument(input: &str) -> Option<&str> {
    ["/model ", "/thinking ", "/login "]
        .into_iter()
        .find_map(|prefix| input.strip_prefix(prefix))
        .map(str::trim)
}

fn suggestions_for(input: &str) -> Vec<Suggestion> {
    const COMMANDS: &[(&str, &str, bool)] = &[
        ("/help", "Show all commands", true),
        ("/model", "Choose from authenticated models", false),
        ("/login", "Add an OAuth or API-key provider", false),
        ("/omni", "Manage an OmniRoute gateway", false),
        (
            "/aperture",
            "Route providers through Tailscale Aperture",
            false,
        ),
        ("/btw", "Open an ephemeral side-question thread", false),
        ("/thinking", "Choose reasoning effort for this model", false),
        ("/tools", "List active tools", true),
        ("/status", "Show session information", true),
        ("/session", "Show session information", true),
        ("/sidebar", "Show session information", true),
        ("/hotkeys", "Show keyboard shortcuts", true),
        ("/messages", "Show transcript summary", true),
        ("/name", "Name this session", true),
        ("/resume", "Switch to a saved session", false),
        ("/sessions", "List saved sessions", true),
        ("/prompt", "Save, list, back up, or restore prompts", false),
        ("/tree", "Show rewind points in this session", true),
        ("/fork", "Rewind to an earlier point", false),
        ("/label", "Name a rewind point", false),
        ("/clone", "Duplicate this session", true),
        ("/export", "Export this session", false),
        ("/import", "Adopt a session file", false),
        ("/steer", "Guide the active response", false),
        ("/followup", "Queue the next message", false),
        ("/queue", "Show queued messages", true),
        ("/clear", "Clear the transcript", true),
        ("/new", "Start a fresh conversation", true),
        ("/compact", "Summarize older context", false),
        ("/reload", "Reload local resources", true),
        ("/resources", "Show loaded resources", true),
        ("/planner", "Toggle planning mode", true),
        ("/planner-review", "Review code changes", false),
        ("/planner-annotate", "Annotate a target", false),
        ("/planner-last", "Annotate last response", true),
        ("/ralph", "Manage Ralph loops", false),
        ("/system", "Show or replace system prompt", false),
        ("/exit", "Exit GoshCoder", true),
        ("/quit", "Exit GoshCoder", true),
    ];

    if !input.starts_with('/') || input.chars().any(char::is_whitespace) {
        return Vec::new();
    }
    let query = input.to_lowercase();
    COMMANDS
        .iter()
        .filter(|(name, _, _)| name.starts_with(&query))
        .map(|(name, description, execute)| Suggestion {
            label: (*name).to_owned(),
            description: (*description).to_owned(),
            value: if *execute {
                (*name).to_owned()
            } else {
                format!("{name} ")
            },
            execute: *execute,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn editor_navigation_respects_utf8_boundaries() {
        let mut app = App::new();
        app.set_input("你好");
        app.handle_key(key(KeyCode::Left));
        app.handle_key(key(KeyCode::Backspace));

        assert_eq!(app.input, "好");
        assert_eq!(app.cursor, 0);
    }

    #[test]
    fn backtab_cycles_thinking_and_ctrl_j_inserts_a_newline() {
        let mut app = App::new();
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)),
            Action::CycleThinking
        );
        app.set_input("a");
        app.handle_key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::CONTROL));
        assert_eq!(app.input, "a\n");
    }

    #[test]
    fn paste_keeps_carriage_return_line_breaks() {
        let mut app = App::new();
        app.paste("a\r\nb\rc\td\x1b[2Je");
        assert_eq!(app.input, "a\nb\nc    de");
    }

    #[test]
    fn scrolling_is_clamped_to_the_last_rendered_extent() {
        let mut app = App::new();
        app.last_max_scroll.set(4);
        app.handle_key(key(KeyCode::PageUp));
        assert_eq!(app.scroll, 4);
        app.scroll_down(10);
        assert_eq!(app.scroll, 0);
    }

    #[test]
    fn argument_palettes_filter_runtime_suggestions() {
        let mut app = App::new();
        app.dynamic_suggestions = vec![
            Suggestion {
                label: "openai/gpt-5.6-terra".to_owned(),
                description: "GPT".to_owned(),
                value: "/model openai/gpt-5.6-terra".to_owned(),
                execute: true,
            },
            Suggestion {
                label: "anthropic/claude-sonnet-5".to_owned(),
                description: "Claude".to_owned(),
                value: "/model anthropic/claude-sonnet-5".to_owned(),
                execute: true,
            },
        ];
        app.set_input("/model claude");
        let suggestions = app.suggestions();
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].label, "anthropic/claude-sonnet-5");
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(dynamic_palette_argument("/thinking "), Some(""));
        assert_eq!(dynamic_palette_argument("/help"), None);
    }

    #[test]
    fn command_completion_preserves_palette_behavior() {
        let mut app = App::new();
        app.set_input("/mo");
        app.handle_key(key(KeyCode::Tab));

        assert_eq!(app.input, "/model ");
        assert!(app.suggestions().is_empty());
    }

    #[test]
    fn multiline_navigation_matches_editor_lines() {
        let mut app = App::new();
        app.set_input("short\na longer line\nlast");
        app.cursor = "short\na long".len();

        app.handle_key(key(KeyCode::Up));
        assert_eq!(app.cursor, "short".len());

        app.handle_key(key(KeyCode::Down));
        assert_eq!(app.cursor, "short\na lon".len());
    }

    #[test]
    fn idle_ctrl_c_requires_confirmation() {
        let mut app = App::new();

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::None
        );
        assert!(app.status.contains("again"));
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Quit
        );
    }

    #[test]
    fn recorded_session_exits_on_the_first_idle_ctrl_c() {
        let mut app = App::new();
        app.set_recording_active(true);

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            Action::Quit
        );
    }

    #[test]
    fn model_and_thinking_hotkeys_delegate_to_the_runtime() {
        let mut app = App::new();

        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL)),
            Action::CycleModel { direction: 1 }
        );
        assert_eq!(
            app.handle_key(KeyEvent::new(
                KeyCode::Char('p'),
                KeyModifiers::CONTROL | KeyModifiers::SHIFT,
            )),
            Action::CycleModel { direction: -1 }
        );
        assert_eq!(
            app.handle_key(KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT)),
            Action::CycleThinking
        );
    }

    #[test]
    fn enter_submits_selected_command() {
        let mut app = App::new();
        app.set_input("/help");

        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            Action::Submit("/help".to_owned())
        );
    }
}
