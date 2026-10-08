//! The prompt editor and Ctrl-C handling of line-mode chat
//! (`-fullscreen=false`).
//!
//! Reading the prompt with the terminal in cooked mode let the tty echo
//! arrow keys as `^[[A` and send them to the model, and left Ctrl-C to the
//! default SIGINT disposition, which killed the process in the middle of a
//! reply. The prompt is now read in raw mode with a small editor (cursor
//! keys, history, Home/End, word deletion, bracketed paste), and while a
//! reply streams a SIGINT handler counts presses so the chat loop can abort
//! the turn instead of dying. pi's equivalent behaviour lives in its
//! interactive mode, where Ctrl-C clears the editor, aborts a running turn,
//! and exits on a second press.

use std::{
    io::{self, Write},
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use crossterm::{
    event::{
        self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};
use unicode_width::UnicodeWidthStr;

/// What one prompt read produced.
#[derive(Debug, Eq, PartialEq)]
pub enum LineInput {
    Line(String),
    /// Ctrl-C twice at an empty prompt: leave chat.
    Exit,
    /// Ctrl-D at an empty prompt, or the input ended.
    Eof,
}

/// The editable line, independent of the terminal so it can be tested.
#[derive(Debug, Default)]
pub struct LineBuffer {
    text: String,
    /// Byte offset at a character boundary.
    cursor: usize,
    history_index: Option<usize>,
    draft: String,
    quit_armed: Option<Instant>,
}

/// What a key did to the line.
#[derive(Debug, Eq, PartialEq)]
pub enum KeyOutcome {
    Continue,
    Submit(String),
    /// First Ctrl-C at an empty prompt: say how to leave.
    ArmExit,
    Exit,
    Eof,
}

impl LineBuffer {
    #[cfg(test)]
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn handle_key(&mut self, key: KeyEvent, history: &[String]) -> KeyOutcome {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        if !(control && key.code == KeyCode::Char('c')) {
            self.quit_armed = None;
        }
        match key.code {
            KeyCode::Char('c') if control => {
                if !self.text.is_empty() {
                    self.text.clear();
                    self.cursor = 0;
                    return KeyOutcome::Continue;
                }
                if self
                    .quit_armed
                    .is_some_and(|armed| armed.elapsed() < Duration::from_secs(3))
                {
                    self.quit_armed = None;
                    return KeyOutcome::Exit;
                }
                self.quit_armed = Some(Instant::now());
                KeyOutcome::ArmExit
            }
            KeyCode::Char('d') if control => {
                if self.text.is_empty() {
                    KeyOutcome::Eof
                } else {
                    self.delete_forward();
                    KeyOutcome::Continue
                }
            }
            KeyCode::Enter => {
                self.history_index = None;
                self.cursor = 0;
                KeyOutcome::Submit(std::mem::take(&mut self.text))
            }
            KeyCode::Char('a') if control => {
                self.cursor = 0;
                KeyOutcome::Continue
            }
            KeyCode::Char('e') if control => {
                self.cursor = self.text.len();
                KeyOutcome::Continue
            }
            KeyCode::Char('u') if control => {
                self.text.drain(..self.cursor);
                self.cursor = 0;
                KeyOutcome::Continue
            }
            KeyCode::Char('k') if control => {
                self.text.truncate(self.cursor);
                KeyOutcome::Continue
            }
            KeyCode::Char('w') if control => {
                let end = self.cursor;
                self.move_word_left();
                self.text.drain(self.cursor..end);
                KeyOutcome::Continue
            }
            KeyCode::Home => {
                self.cursor = 0;
                KeyOutcome::Continue
            }
            KeyCode::End => {
                self.cursor = self.text.len();
                KeyOutcome::Continue
            }
            KeyCode::Left if control || alt => {
                self.move_word_left();
                KeyOutcome::Continue
            }
            KeyCode::Left => {
                self.cursor = previous_boundary(&self.text, self.cursor);
                KeyOutcome::Continue
            }
            KeyCode::Right => {
                self.cursor = next_boundary(&self.text, self.cursor);
                KeyOutcome::Continue
            }
            KeyCode::Backspace => {
                let previous = previous_boundary(&self.text, self.cursor);
                self.text.drain(previous..self.cursor);
                self.cursor = previous;
                KeyOutcome::Continue
            }
            KeyCode::Delete => {
                self.delete_forward();
                KeyOutcome::Continue
            }
            KeyCode::Up => {
                self.history(-1, history);
                KeyOutcome::Continue
            }
            KeyCode::Down => {
                self.history(1, history);
                KeyOutcome::Continue
            }
            KeyCode::Tab => {
                self.insert("\t");
                KeyOutcome::Continue
            }
            KeyCode::Char(character) if !control && !alt => {
                self.insert(&character.to_string());
                KeyOutcome::Continue
            }
            // Everything else (function keys, unbound chords) is ignored
            // rather than typed into the prompt as escape bytes.
            _ => KeyOutcome::Continue,
        }
    }

    pub fn insert(&mut self, text: &str) {
        self.text.insert_str(self.cursor, text);
        self.cursor += text.len();
    }

    fn delete_forward(&mut self) {
        let next = next_boundary(&self.text, self.cursor);
        self.text.drain(self.cursor..next);
    }

    fn move_word_left(&mut self) {
        while self.cursor > 0 && self.text[..self.cursor].ends_with(char::is_whitespace) {
            self.cursor = previous_boundary(&self.text, self.cursor);
        }
        while self.cursor > 0 && !self.text[..self.cursor].ends_with(char::is_whitespace) {
            self.cursor = previous_boundary(&self.text, self.cursor);
        }
    }

    fn history(&mut self, direction: i8, history: &[String]) {
        if history.is_empty() {
            return;
        }
        let next = match (self.history_index, direction) {
            (None, -1) => {
                self.draft = self.text.clone();
                Some(history.len() - 1)
            }
            (None, _) => return,
            (Some(index), -1) => Some(index.saturating_sub(1)),
            (Some(index), _) if index + 1 >= history.len() => None,
            (Some(index), _) => Some(index + 1),
        };
        self.history_index = next;
        self.text = match next {
            Some(index) => history[index].clone(),
            None => std::mem::take(&mut self.draft),
        };
        self.cursor = self.text.len();
    }

    /// The visible part of the line and the cursor column within it, for a
    /// row of `width` cells: the line scrolls sideways to keep the cursor in
    /// view instead of wrapping, which a `\r` redraw could not erase.
    fn view(&self, width: usize) -> (String, usize) {
        let display = |text: &str| text.replace(['\n', '\t'], " ");
        let before = display(&self.text[..self.cursor]);
        let after = display(&self.text[self.cursor..]);
        let width = width.max(2);
        let mut start = 0;
        while before[start..].width() >= width {
            start = next_boundary(&before, start);
        }
        let shown_before = &before[start..];
        let mut shown = shown_before.to_owned();
        for character in after.chars() {
            if shown.width() + character.to_string().width() >= width {
                break;
            }
            shown.push(character);
        }
        (shown, shown_before.width())
    }
}

fn previous_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .char_indices()
        .next_back()
        .map_or(0, |(index, _)| index)
}

fn next_boundary(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .chars()
        .next()
        .map_or(cursor, |character| cursor + character.len_utf8())
}

/// Reads one prompt line from the terminal in raw mode.
pub fn read_line(prompt: &str, history: &[String]) -> io::Result<LineInput> {
    enable_raw_mode()?;
    let _ = execute!(io::stderr(), EnableBracketedPaste);
    let result = edit(prompt, history);
    let _ = execute!(io::stderr(), DisableBracketedPaste);
    let _ = disable_raw_mode();
    eprintln!();
    result
}

fn edit(prompt: &str, history: &[String]) -> io::Result<LineInput> {
    let mut buffer = LineBuffer::default();
    redraw(prompt, &buffer)?;
    loop {
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => {
                match buffer.handle_key(key, history) {
                    KeyOutcome::Continue => {}
                    KeyOutcome::Submit(line) => return Ok(LineInput::Line(line)),
                    KeyOutcome::Exit => return Ok(LineInput::Exit),
                    KeyOutcome::Eof => return Ok(LineInput::Eof),
                    KeyOutcome::ArmExit => {
                        let mut stderr = io::stderr();
                        write!(stderr, "\r\x1b[K(press Ctrl-C again to exit)\r\n")?;
                    }
                }
            }
            Event::Paste(text) => {
                let text = text.replace("\r\n", "\n").replace('\r', "\n");
                let clean: String = crate::markdown::sanitize_terminal_text(&text)
                    .chars()
                    .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
                    .collect();
                buffer.insert(&clean);
            }
            Event::Resize(_, _) => {}
            _ => continue,
        }
        redraw(prompt, &buffer)?;
    }
}

fn redraw(prompt: &str, buffer: &LineBuffer) -> io::Result<()> {
    let width = crossterm::terminal::size()
        .map(|(columns, _)| usize::from(columns))
        .unwrap_or(80);
    let (shown, column) = buffer.view(width.saturating_sub(prompt.width()));
    let mut stderr = io::stderr();
    write!(stderr, "\r\x1b[K{prompt}{shown}\r")?;
    let target = prompt.width() + column;
    if target > 0 {
        write!(stderr, "\x1b[{target}C")?;
    }
    stderr.flush()
}

/// Ctrl-C presses seen by the SIGINT handler while a reply streams.
static INTERRUPTS: AtomicUsize = AtomicUsize::new(0);

/// Routes SIGINT to [`take_interrupts`] instead of killing the process.
/// Installed once for line-mode chat; the prompt reads in raw mode, where
/// Ctrl-C is a key and never becomes a signal.
pub fn install_interrupt_handler() {
    #[cfg(unix)]
    {
        use std::os::raw::c_int;
        const SIGINT: c_int = 2;
        unsafe extern "C" {
            // Declared as main.rs declares it (the handler as an address),
            // since one symbol must have one signature.
            fn signal(signum: c_int, handler: usize) -> usize;
        }
        extern "C" fn count(_: c_int) {
            // An atomic increment is async-signal-safe.
            INTERRUPTS.fetch_add(1, Ordering::SeqCst);
        }
        // SAFETY: installs a handler that only touches an atomic counter.
        unsafe {
            signal(SIGINT, count as extern "C" fn(c_int) as usize);
        }
    }
}

/// Presses since the last call.
pub fn take_interrupts() -> usize {
    INTERRUPTS.swap(0, Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(character: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(character), KeyModifiers::CONTROL)
    }

    fn type_text(buffer: &mut LineBuffer, text: &str) {
        for character in text.chars() {
            buffer.handle_key(key(KeyCode::Char(character)), &[]);
        }
    }

    #[test]
    fn arrow_keys_move_the_cursor_instead_of_typing_escape_bytes() {
        let mut buffer = LineBuffer::default();
        type_text(&mut buffer, "helo");
        buffer.handle_key(key(KeyCode::Left), &[]);
        buffer.handle_key(key(KeyCode::Char('l')), &[]);
        buffer.handle_key(key(KeyCode::F(5)), &[]);
        assert_eq!(
            buffer.handle_key(key(KeyCode::Enter), &[]),
            KeyOutcome::Submit("hello".to_owned())
        );
    }

    #[test]
    fn up_and_down_walk_the_history_and_restore_the_draft() {
        let history = vec!["first".to_owned(), "second".to_owned()];
        let mut buffer = LineBuffer::default();
        type_text(&mut buffer, "draft");
        buffer.handle_key(key(KeyCode::Up), &history);
        assert_eq!(buffer.text(), "second");
        buffer.handle_key(key(KeyCode::Up), &history);
        assert_eq!(buffer.text(), "first");
        buffer.handle_key(key(KeyCode::Down), &history);
        buffer.handle_key(key(KeyCode::Down), &history);
        assert_eq!(buffer.text(), "draft");
    }

    #[test]
    fn ctrl_c_clears_then_asks_before_exiting() {
        let mut buffer = LineBuffer::default();
        type_text(&mut buffer, "half typed");
        assert_eq!(buffer.handle_key(ctrl('c'), &[]), KeyOutcome::Continue);
        assert_eq!(buffer.text(), "");
        assert_eq!(buffer.handle_key(ctrl('c'), &[]), KeyOutcome::ArmExit);
        assert_eq!(buffer.handle_key(ctrl('c'), &[]), KeyOutcome::Exit);

        // Any other key in between disarms the exit.
        assert_eq!(buffer.handle_key(ctrl('c'), &[]), KeyOutcome::ArmExit);
        buffer.handle_key(key(KeyCode::Char('x')), &[]);
        buffer.handle_key(key(KeyCode::Backspace), &[]);
        assert_eq!(buffer.handle_key(ctrl('c'), &[]), KeyOutcome::ArmExit);
        assert_eq!(buffer.handle_key(ctrl('d'), &[]), KeyOutcome::Eof);
    }

    #[test]
    fn a_long_line_scrolls_to_keep_the_cursor_visible() {
        let mut buffer = LineBuffer::default();
        type_text(&mut buffer, "abcdefghijklmnopqrstuvwxyz");
        let (shown, column) = buffer.view(10);
        assert!(shown.width() < 10, "{shown}");
        assert!(shown.ends_with('z'));
        assert_eq!(column, shown.width());
        buffer.handle_key(key(KeyCode::Home), &[]);
        let (shown, column) = buffer.view(10);
        assert!(shown.starts_with('a'));
        assert_eq!(column, 0);
    }
}
