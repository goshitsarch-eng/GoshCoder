use std::{
    cell::Cell,
    time::{Duration, Instant},
};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Width, tools expanded, thinking hidden: what the transcript rows depend on.
pub type TranscriptLayout = (u16, bool, bool);

/// A paste longer than this is summarized in the transcript.
pub const LARGE_PASTE_LINES: usize = 20;

/// A transcript item rendered by the terminal interface.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Message {
    pub role: MessageRole,
    pub title: String,
    pub text: String,
    pub detail: String,
    pub is_error: bool,
    /// The reply still arriving; the renderer ends it with a cursor block.
    pub streaming: bool,
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
    /// Context the session carries for the model but the user did not type,
    /// such as pi's branch summary; shown as one dim line, not a bubble.
    Summary,
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
    /// The composer's answer to an open [`ComposerPrompt`].
    Answer(String),
    /// Esc or Ctrl-C while a [`ComposerPrompt`] is open.
    CancelPrompt,
}

/// A question the runtime asked through the composer (an OAuth paste, an API
/// key). Enter answers it instead of sending a message, and a secret answer
/// is masked on screen and kept out of the editor history.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComposerPrompt {
    pub label: String,
    pub secret: bool,
    /// An example of what to enter, shown while the composer is empty.
    pub placeholder: String,
    /// Choices for a select question, offered in the palette; Enter answers
    /// with the highlighted choice's value.
    pub options: Vec<Suggestion>,
}

impl ComposerPrompt {
    /// A free-text question.
    pub fn text(label: impl Into<String>, secret: bool) -> Self {
        Self {
            label: label.into(),
            secret,
            placeholder: String::new(),
            options: Vec::new(),
        }
    }
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
    /// Rows the user scrolled above the newest content.
    pub scroll: u16,
    pub selected_suggestion: usize,
    pub tools_expanded: bool,
    pub hide_thinking: bool,
    pub history: Vec<String>,
    /// Suggestions the runtime supplies for an argument palette (`/model `,
    /// `/thinking `, `/login `); the static command list covers the rest.
    pub dynamic_suggestions: Vec<Suggestion>,
    /// Prompt templates and invocable skills, offered beside the built-in
    /// commands in the slash palette.
    pub command_suggestions: Vec<Suggestion>,
    /// Steering and follow-up texts waiting for the agent, shown above the
    /// composer so a queued message is not invisible until it runs.
    pub queued: Vec<String>,
    /// A question the composer is answering instead of composing a message.
    pub prompt: Option<ComposerPrompt>,
    /// Context use for the status line when the sidebar is hidden.
    pub context_hint: String,
    /// Large pastes, which the transcript shows as "[pasted N lines]"
    /// although the whole text is sent.
    pub pasted_blocks: Vec<String>,
    /// The largest useful scroll offset at the last draw. The renderer
    /// writes it so key handling can clamp instead of letting the offset
    /// run past the top of the transcript.
    pub last_max_scroll: Cell<u16>,
    /// Rows that arrived below the viewport while it was scrolled up. The
    /// offset is measured from the bottom, so without this the view would
    /// drift down with every streamed line; the renderer adds the growth so
    /// the same content stays on screen.
    pub scroll_growth: Cell<usize>,
    /// Whether new rows arrived below a scrolled-up viewport.
    pub unseen_output: Cell<bool>,
    /// Transcript row count at the last draw and the layout it was measured
    /// under (width, tools expanded, thinking hidden): growth only counts
    /// when the layout is unchanged, so expanding cards is not "new output".
    pub last_transcript_rows: Cell<Option<(usize, TranscriptLayout)>>,
    /// Transcript viewport height at the last draw, for page scrolling.
    pub last_transcript_height: Cell<u16>,
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
            command_suggestions: Vec::new(),
            queued: Vec::new(),
            prompt: None,
            context_hint: String::new(),
            pasted_blocks: Vec::new(),
            last_max_scroll: Cell::new(0),
            scroll_growth: Cell::new(0),
            unseen_output: Cell::new(false),
            last_transcript_rows: Cell::new(None),
            last_transcript_height: Cell::new(0),
            history_index: None,
            draft: String::new(),
            quit_armed_at: None,
        }
    }

    pub fn suggestions(&self) -> Vec<Suggestion> {
        if let Some(prompt) = self.prompt.as_ref() {
            let query = self.input.trim().to_lowercase();
            return prompt
                .options
                .iter()
                .filter(|option| {
                    query.is_empty()
                        || option.label.to_lowercase().contains(&query)
                        || option.value.to_lowercase().contains(&query)
                })
                .cloned()
                .collect();
        }
        if let Some(argument) = dynamic_palette_argument(&self.input) {
            let query = argument.to_lowercase();
            return self
                .dynamic_suggestions
                .iter()
                .filter(|suggestion| {
                    // Descriptions carry the display names ("Claude", "API
                    // key"), which are what people tend to type.
                    query.is_empty()
                        || query.split_whitespace().all(|word| {
                            suggestion.label.to_lowercase().contains(word)
                                || suggestion.description.to_lowercase().contains(word)
                        })
                })
                .cloned()
                .collect();
        }
        let mut suggestions = suggestions_for(&self.input);
        if self.input.starts_with('/') && !self.input.contains(char::is_whitespace) {
            let query = self.input.to_lowercase();
            suggestions.extend(
                self.command_suggestions
                    .iter()
                    .filter(|suggestion| suggestion.label.to_lowercase().starts_with(&query))
                    .cloned(),
            );
        }
        suggestions
    }

    /// The rows above the bottom the renderer shows: the user's own offset
    /// plus whatever arrived below it since.
    pub fn scroll_offset(&self) -> usize {
        if self.scroll == 0 {
            0
        } else {
            usize::from(self.scroll) + self.scroll_growth.get()
        }
    }

    /// Scrolls the transcript towards older content, never past its top.
    pub fn scroll_up(&mut self, rows: u16) {
        let offset = self.scroll_offset().saturating_add(usize::from(rows));
        self.set_scroll(offset.min(usize::from(self.last_max_scroll.get())));
    }

    /// Scrolls the transcript towards the newest content.
    pub fn scroll_down(&mut self, rows: u16) {
        self.set_scroll(self.scroll_offset().saturating_sub(usize::from(rows)));
    }

    /// Follows the newest content again.
    pub fn scroll_to_bottom(&mut self) {
        self.set_scroll(0);
    }

    fn set_scroll(&mut self, offset: usize) {
        self.scroll = offset.min(usize::from(u16::MAX)) as u16;
        self.scroll_growth.set(0);
        if self.scroll == 0 {
            self.unseen_output.set(false);
        }
    }

    /// One transcript page, less two rows of overlap so the reader keeps
    /// their place.
    fn page_rows(&self) -> u16 {
        self.last_transcript_height.get().saturating_sub(2).max(1)
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
        // Whoever sends something wants to see what it starts.
        self.scroll_to_bottom();
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

        if self.prompt.is_some() {
            match (key.code, modifiers) {
                (KeyCode::Esc, _) => {
                    self.clear_input();
                    return Action::CancelPrompt;
                }
                (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                    self.clear_input();
                    return Action::CancelPrompt;
                }
                (KeyCode::Enter, modifiers)
                    if !modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
                {
                    let choices = self.suggestions();
                    let chosen = choices
                        .get(self.clamped_suggestion(choices.len()))
                        .map(|choice| choice.value.clone());
                    let typed = std::mem::take(&mut self.input);
                    self.cursor = 0;
                    self.selected_suggestion = 0;
                    return Action::Answer(chosen.unwrap_or_else(|| typed.trim().to_owned()));
                }
                _ => {}
            }
        }

        match (key.code, modifiers) {
            // pi's fullscreen bindings: Ctrl-Home and Ctrl-End scroll to the
            // top and bottom without taking Home/End from the editor.
            (KeyCode::End, modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_to_bottom();
                Action::None
            }
            (KeyCode::Home, modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
                self.scroll_up(u16::MAX);
                Action::None
            }
            // End has nothing to do in an empty composer, so it jumps to the
            // newest output there as well.
            (KeyCode::End, _) if self.input.is_empty() && self.scroll > 0 => {
                self.scroll_to_bottom();
                Action::None
            }
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
                if !self.suggestions().is_empty() {
                    // The palette's "esc close" comes before aborting a turn.
                    self.clear_input();
                    Action::None
                } else if self.streaming {
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
                self.scroll_up(self.page_rows());
                Action::None
            }
            (KeyCode::PageDown, _) => {
                self.scroll_down(self.page_rows());
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
        // Tabs stay tabs: they are part of what the model should see (a
        // Makefile, a TSV), and only the editor's rendering expands them.
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let sanitized: String = crate::markdown::sanitize_terminal_text(&text)
            .chars()
            .filter(|character| !character.is_control() || matches!(*character, '\n' | '\t'))
            .collect();
        if sanitized.lines().count() > LARGE_PASTE_LINES {
            self.pasted_blocks.push(sanitized.clone());
            if self.pasted_blocks.len() > 16 {
                self.pasted_blocks.remove(0);
            }
        }
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
            // A command typed out in full runs on Enter when its argument
            // is optional (`/compact`, `/export`); only a required argument
            // makes Enter complete the name and wait. Tab still completes.
            let typed_in_full = self.input.trim() == item.label
                && item.value.trim_end() == item.label
                && !REQUIRES_ARGUMENT.contains(&item.label.as_str());
            if !typed_in_full {
                self.set_input(&item.value);
                if !item.execute {
                    return Action::None;
                }
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

/// Commands that do nothing useful without an argument, so Enter on the bare
/// name completes it instead of running it.
const REQUIRES_ARGUMENT: &[&str] = &[
    "/fork",
    "/label",
    "/import",
    "/steer",
    "/followup",
    "/planner-annotate",
    "/grok-cli-imagine",
];

fn suggestions_for(input: &str) -> Vec<Suggestion> {
    const COMMANDS: &[(&str, &str, bool)] = &[
        ("/help", "Show all commands", true),
        ("/model", "Open the model picker", false),
        ("/login", "Open the provider picker and log in", false),
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
        (
            "/export",
            "Save this session as HTML, Markdown, or JSONL",
            false,
        ),
        ("/import", "Adopt a session file", false),
        ("/share", "Upload this session as a secret gist", true),
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
        // Provider-specific commands come after the general ones, so the
        // palette opens on the commands everyone uses.
        (
            "/grok-cli-imagine",
            "Generate or edit an image with Grok Imagine",
            false,
        ),
        (
            "/grok-cli-imagine:tool",
            "Turn the image_gen tool on or off",
            false,
        ),
        (
            "/grok-cli-usage",
            "Show the Grok CLI subscription's usage",
            true,
        ),
        (
            "/grok-cli-accounts",
            "List, add, or switch Grok CLI accounts",
            false,
        ),
        (
            "/grok-cli-conv",
            "Show or rotate the Grok CLI conversation ID",
            false,
        ),
        ("/exit", "Exit GoshCoder", true),
        ("/quit", "Exit GoshCoder", true),
    ];

    if let Some(suggestions) = subcommand_suggestions(input) {
        return suggestions;
    }
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

/// Static subcommand completion for the gateway commands. The word is
/// completed in place; subcommands that take an argument keep the cursor in
/// the input, the rest run on selection.
fn subcommand_suggestions(input: &str) -> Option<Vec<Suggestion>> {
    /// A subcommand word, its description, and whether it runs on selection.
    type Word = (&'static str, &'static str, bool);
    const GATEWAYS: &[(&str, &[Word])] = &[
        (
            "/omni ",
            &[
                ("status", "Check the gateway and show the setup", true),
                ("setup", "Configure server URL and API key", true),
                ("sync", "Sync models into the /model picker", true),
                ("models", "Browse models, optionally filtered", false),
                ("test", "Smoke-test a model through the gateway", false),
                ("dashboard", "Show the OmniRoute dashboard URL", true),
                ("config", "Show config paths and settings", true),
                ("help", "Show the OmniRoute commands", true),
            ],
        ),
        (
            "/aperture ",
            &[
                ("status", "Show gateway and cached configuration", true),
                ("onboarding", "Configure an Aperture gateway", true),
                ("settings", "Show or change configuration settings", false),
                ("sync", "Refresh the gateway model snapshot", true),
                ("providers", "List gateway providers and routing APIs", true),
                ("connectors", "List gateway connector tools", true),
                ("pin", "Pin a connector tool for the next session", false),
                ("unpin", "Remove a pinned connector tool", false),
                ("help", "Show the Aperture commands", true),
            ],
        ),
        (
            "/ralph ",
            &[
                ("status", "Show loop progress and iteration count", true),
                ("resume", "Continue a stopped loop", false),
                ("list", "Loops in .ralph/", true),
                ("stop", "End the loop", true),
                ("start", "Start a loop: <name> <task>", false),
                ("archive", "Move a finished loop out of the list", false),
                ("delete", "Remove a loop and its state", false),
            ],
        ),
    ];
    let lowered = input.to_lowercase();
    let (prefix, words) = GATEWAYS
        .iter()
        .find(|(prefix, _)| lowered.starts_with(prefix))?;
    let typed = lowered[prefix.len()..].trim_start();
    if typed.contains(char::is_whitespace) {
        // The subcommand is complete; its argument is free text.
        return Some(Vec::new());
    }
    Some(
        words
            .iter()
            .filter(|(word, _, _)| word.starts_with(typed))
            .map(|(word, description, execute)| Suggestion {
                label: (*word).to_owned(),
                description: (*description).to_owned(),
                value: if *execute {
                    format!("{prefix}{word}")
                } else {
                    format!("{prefix}{word} ")
                },
                execute: *execute,
            })
            .collect(),
    )
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
    fn paste_keeps_carriage_return_line_breaks_and_tabs() {
        let mut app = App::new();
        app.paste("a\r\nb\rc\td\x1b[2Je");
        // The tab is what the model should receive; only the editor's
        // rendering expands it.
        assert_eq!(app.input, "a\nb\nc\tde");
        assert!(app.pasted_blocks.is_empty());
    }

    #[test]
    fn a_large_paste_is_remembered_for_the_transcript_marker() {
        let mut app = App::new();
        let text = (1..=30)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        app.paste(&text);
        assert_eq!(app.input, text);
        assert_eq!(app.pasted_blocks, [text]);
    }

    #[test]
    fn scrolling_is_clamped_to_the_last_rendered_extent() {
        let mut app = App::new();
        app.last_max_scroll.set(4);
        app.last_transcript_height.set(12);
        app.handle_key(key(KeyCode::PageUp));
        assert_eq!(app.scroll, 4);
        app.scroll_down(10);
        assert_eq!(app.scroll, 0);
    }

    #[test]
    fn page_keys_move_a_page_less_two_rows() {
        let mut app = App::new();
        app.last_max_scroll.set(100);
        app.last_transcript_height.set(20);
        app.handle_key(key(KeyCode::PageUp));
        assert_eq!(app.scroll, 18);
        app.handle_key(key(KeyCode::PageUp));
        assert_eq!(app.scroll, 36);
        app.handle_key(key(KeyCode::PageDown));
        assert_eq!(app.scroll, 18);
    }

    #[test]
    fn submitting_or_ctrl_end_returns_to_the_newest_output() {
        let mut app = App::new();
        app.last_max_scroll.set(50);
        app.scroll_up(20);
        app.scroll_growth.set(7);
        app.unseen_output.set(true);
        assert_eq!(app.scroll_offset(), 27);
        app.handle_key(KeyEvent::new(KeyCode::End, KeyModifiers::CONTROL));
        assert_eq!(app.scroll_offset(), 0);
        assert!(!app.unseen_output.get());

        // End in an empty composer jumps too; with text it is the editor's.
        app.scroll_up(5);
        app.set_input("abc");
        app.cursor = 0;
        app.handle_key(key(KeyCode::End));
        assert_eq!(app.cursor, 3);
        assert_eq!(app.scroll, 5);
        app.set_input("");
        app.handle_key(key(KeyCode::End));
        assert_eq!(app.scroll, 0);

        app.scroll_up(9);
        app.set_input("next prompt");
        app.record_submission("next prompt");
        assert_eq!(app.scroll_offset(), 0);
    }

    #[test]
    fn scrolling_further_keeps_the_rows_that_arrived_below() {
        let mut app = App::new();
        app.last_max_scroll.set(100);
        app.scroll_up(10);
        // The renderer measured 6 new rows below the viewport.
        app.scroll_growth.set(6);
        app.scroll_up(3);
        assert_eq!(app.scroll_offset(), 19);
        app.scroll_down(19);
        assert_eq!(app.scroll_offset(), 0);
    }

    #[test]
    fn enter_runs_a_command_whose_argument_is_optional() {
        for command in ["/compact", "/export", "/prompt", "/ralph"] {
            let mut app = App::new();
            app.set_input(command);
            assert_eq!(
                app.handle_key(key(KeyCode::Enter)),
                Action::Submit(command.to_owned()),
                "{command}"
            );
        }
        // Tab still completes, and a partial name still completes on Enter.
        let mut app = App::new();
        app.set_input("/compact");
        app.handle_key(key(KeyCode::Tab));
        assert_eq!(app.input, "/compact ");
        let mut app = App::new();
        app.set_input("/comp");
        assert_eq!(app.handle_key(key(KeyCode::Enter)), Action::None);
        assert_eq!(app.input, "/compact ");
        // A required argument keeps Enter as completion.
        let mut app = App::new();
        app.set_input("/fork");
        assert_eq!(app.handle_key(key(KeyCode::Enter)), Action::None);
        assert_eq!(app.input, "/fork ");
    }

    #[test]
    fn prompt_templates_and_skills_join_the_slash_palette() {
        let mut app = App::new();
        app.command_suggestions = vec![
            Suggestion {
                label: "/review".to_owned(),
                description: "Prompt template".to_owned(),
                value: "/review ".to_owned(),
                execute: false,
            },
            Suggestion {
                label: "/skill:deploy".to_owned(),
                description: "Skill".to_owned(),
                value: "/skill:deploy ".to_owned(),
                execute: false,
            },
        ];
        app.set_input("/rev");
        let labels = app
            .suggestions()
            .into_iter()
            .map(|suggestion| suggestion.label)
            .collect::<Vec<_>>();
        assert_eq!(labels, ["/review"]);
        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            Action::None,
            "a partial name completes first"
        );
        assert_eq!(app.input, "/review ");
        app.set_input("/review");
        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            Action::Submit("/review".to_owned())
        );
        app.set_input("/sk");
        assert_eq!(app.suggestions()[0].label, "/skill:deploy");
        app.set_input("/zzz");
        assert!(app.suggestions().is_empty());
    }

    #[test]
    fn a_composer_prompt_takes_enter_and_escape() {
        let mut app = App::new();
        app.prompt = Some(ComposerPrompt::text("API key for openai", true));
        app.set_input("/model");
        assert!(app.suggestions().is_empty(), "no palette while answering");
        app.set_input("sk-test");
        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            Action::Answer("sk-test".to_owned())
        );
        assert!(app.input.is_empty());
        assert!(app.history.is_empty(), "a secret never enters the history");
        app.set_input("half");
        assert_eq!(app.handle_key(key(KeyCode::Esc)), Action::CancelPrompt);
        assert!(app.input.is_empty());
    }

    #[test]
    fn a_select_question_is_answered_from_the_palette() {
        let mut app = App::new();
        let option = |label: &str, value: &str| Suggestion {
            label: label.to_owned(),
            description: String::new(),
            value: value.to_owned(),
            execute: true,
        };
        app.prompt = Some(ComposerPrompt {
            label: "anthropic login method".to_owned(),
            secret: false,
            placeholder: String::new(),
            options: vec![
                option("Browser login (default)", "browser"),
                option("Copy code login (headless)", "copy_code"),
            ],
        });
        assert_eq!(app.suggestions().len(), 2);
        app.handle_key(key(KeyCode::Down));
        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            Action::Answer("copy_code".to_owned())
        );
        // Typing filters the choices.
        app.set_input("brow");
        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            Action::Answer("browser".to_owned())
        );
    }

    #[test]
    fn escape_closes_the_palette_before_aborting_a_turn() {
        let mut app = App::new();
        app.streaming = true;
        app.set_input("/he");
        assert_eq!(app.handle_key(key(KeyCode::Esc)), Action::None);
        assert!(app.input.is_empty());
        assert_eq!(app.handle_key(key(KeyCode::Esc)), Action::Abort);
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

    #[test]
    fn gateway_subcommands_complete_in_place() {
        let all = suggestions_for("/omni ");
        assert_eq!(all.len(), 8);
        assert_eq!(all[0].value, "/omni status");
        assert!(all[0].execute);
        let models = suggestions_for("/omni mo");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].value, "/omni models ");
        assert!(!models[0].execute, "models takes an optional search");
        assert!(
            suggestions_for("/omni test auto").is_empty(),
            "an argument is free text"
        );
        let pin = suggestions_for("/aperture PI");
        assert_eq!(pin.len(), 1);
        assert_eq!(pin[0].value, "/aperture pin ");
        assert!(suggestions_for("/omni").iter().any(|s| s.value == "/omni "));
    }
}
