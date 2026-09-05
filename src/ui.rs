use std::{cmp::min, collections::HashMap};

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthStr;

use crate::{
    markdown::{MarkdownRenderer, MarkdownRole},
    state::{App, FileStatus, Message, MessageRole, SidebarKind, SidebarLine},
};

const BACKGROUND: Color = Color::Rgb(10, 10, 10);
const PANEL_BACKGROUND: Color = Color::Rgb(20, 20, 20);
const USER_BACKGROUND: Color = Color::Rgb(30, 30, 30);
const TOOL_BACKGROUND: Color = Color::Rgb(24, 24, 24);
const ACCENT: Color = Color::Rgb(255, 172, 92);
const VIOLET: Color = Color::Rgb(73, 166, 191);
const CYAN: Color = Color::Rgb(86, 182, 194);
const GREEN: Color = Color::Rgb(127, 216, 143);
const AMBER: Color = Color::Rgb(242, 201, 108);
const RED: Color = Color::Rgb(246, 116, 116);
const TEXT: Color = Color::Rgb(220, 230, 232);
const MUTED: Color = Color::Rgb(119, 143, 150);
const FAINT: Color = Color::Rgb(62, 87, 96);
const MESSAGE_CACHE_LIMIT: usize = 4_096;

/// Memoizes immutable transcript rows across fullscreen redraws.
///
/// Markdown parsing and wrapping dominate a long transcript's redraw cost.
/// The cache is owned by the interactive event loop rather than globally, so
/// it is naturally discarded when the terminal session ends.
#[derive(Debug, Default)]
pub(crate) struct MessageCache {
    width: u16,
    tools_expanded: bool,
    hide_thinking: bool,
    entries: HashMap<Message, Vec<Line<'static>>>,
}

impl MessageCache {
    fn render(
        &mut self,
        message: &Message,
        width: u16,
        tools_expanded: bool,
        hide_thinking: bool,
    ) -> Vec<Line<'static>> {
        if self.entries.is_empty()
            || self.width != width
            || self.tools_expanded != tools_expanded
            || self.hide_thinking != hide_thinking
        {
            self.entries.clear();
            self.width = width;
            self.tools_expanded = tools_expanded;
            self.hide_thinking = hide_thinking;
        }
        if let Some(lines) = self.entries.get(message) {
            return lines.clone();
        }
        if self.entries.len() >= MESSAGE_CACHE_LIMIT {
            self.entries.clear();
        }
        let lines = render_transcript_message(message, width, tools_expanded, hide_thinking);
        self.entries.insert(message.clone(), lines.clone());
        lines
    }
}

pub(crate) fn draw(frame: &mut Frame, app: &App, cache: &mut MessageCache) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND)),
        area,
    );
    if area.width < 20 || area.height < 8 {
        let too_small = Paragraph::new("GoshCoder\nTerminal is too small")
            .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center);
        frame.render_widget(too_small, area);
        return;
    }

    let sidebar_width = if area.width >= 96 {
        (area.width / 3).clamp(32, 42)
    } else {
        0
    };
    let regions = if sidebar_width > 0 {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(20),
                Constraint::Length(1),
                Constraint::Length(sidebar_width),
            ])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(0),
                Constraint::Length(0),
            ])
            .split(area)
    };
    let main = regions[0];

    render_main(frame, main, app, cache);
    if sidebar_width > 0 {
        render_sidebar(frame, regions[2], &app.sidebar);
    }
}

fn render_main(frame: &mut Frame, area: Rect, app: &App, cache: &mut MessageCache) {
    let editor = editor_window(
        &app.input,
        app.cursor,
        area.width.saturating_sub(8) as usize,
    );
    let composer_height = (editor.lines.len() as u16 + 2).clamp(3, 5);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(composer_height),
            Constraint::Length(1),
        ])
        .split(area);

    let title = Line::from(vec![
        Span::styled(
            "  GOSH",
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "CODER",
            Style::default().fg(VIOLET).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "  {}",
                truncate(&app.title, area.width.saturating_sub(17) as usize)
            ),
            Style::default().fg(MUTED),
        ),
    ]);
    let header = Paragraph::new(vec![
        title,
        Line::from(Span::styled("  · · · · · ·", Style::default().fg(ACCENT))),
    ]);
    frame.render_widget(header, chunks[0]);

    let suggestions = app.suggestions();
    let visible_suggestions = suggestions.len().min(9) as u16;
    let palette_height = if visible_suggestions > 0 && chunks[1].height > 3 {
        (visible_suggestions + 2).min(chunks[1].height.saturating_sub(1))
    } else {
        0
    };
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(palette_height)])
        .split(chunks[1]);
    render_transcript(frame, body[0], app, cache);
    if palette_height > 0 {
        render_suggestions(frame, body[1], app, &suggestions);
    }
    render_editor(frame, chunks[2], app, &editor);
    render_status(frame, chunks[3], app);
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &App, cache: &mut MessageCache) {
    let lines = transcript_lines_with_cache(
        &app.messages,
        area.width.saturating_sub(2),
        app.tools_expanded,
        app.hide_thinking,
        Some(cache),
    );
    let max_scroll = lines.len().saturating_sub(area.height as usize);
    let scroll = usize::from(app.scroll).min(max_scroll);
    let start = lines
        .len()
        .saturating_sub(area.height as usize)
        .saturating_sub(scroll)
        .min(u16::MAX as usize) as u16;
    let transcript = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT).bg(BACKGROUND))
        .scroll((start, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript, area);
}

fn render_suggestions(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    suggestions: &[crate::state::Suggestion],
) {
    if suggestions.is_empty() || area.width < 16 || area.height < 3 {
        return;
    }
    let item_capacity = area.height.saturating_sub(2) as usize;
    if item_capacity == 0 {
        return;
    }
    let selected = app
        .selected_suggestion
        .min(suggestions.len().saturating_sub(1));
    let first = selected
        .saturating_add(1)
        .saturating_sub(item_capacity)
        .min(suggestions.len().saturating_sub(item_capacity));
    let visible = &suggestions[first..suggestions.len().min(first + item_capacity)];
    let items: Vec<ListItem<'_>> = visible
        .iter()
        .map(|suggestion| {
            let text = format!(
                " {}  {}",
                suggestion.label,
                truncate(
                    &suggestion.description,
                    area.width.saturating_sub(6) as usize
                )
            );
            ListItem::new(Line::from(text))
        })
        .collect();
    let title = suggestion_title(&app.input);
    let list = List::new(items)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(VIOLET))
                .title(Line::from(title).style(Style::default().fg(MUTED))),
        )
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(27, 67, 76))
                .fg(Color::Rgb(239, 248, 246))
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▌");
    let mut state = ListState::default();
    state.select(Some(selected.saturating_sub(first)));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_editor(frame: &mut Frame, area: Rect, app: &App, editor: &EditorWindow) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(CYAN));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let placeholder = "Tell GoshCoder what to build…  / for commands";
    let lines: Vec<Line<'_>> = if app.input.is_empty() {
        vec![Line::from(Span::styled(
            truncate(placeholder, inner.width as usize),
            Style::default().fg(MUTED),
        ))]
    } else {
        editor
            .lines
            .iter()
            .map(|line| Line::from(Span::styled(line.as_str(), Style::default().fg(TEXT))))
            .collect()
    };
    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::default().bg(BACKGROUND))
            .wrap(Wrap { trim: false }),
        inner,
    );

    let cursor_x = inner.x.saturating_add(min(
        editor.cursor_column as u16,
        inner.width.saturating_sub(1),
    ));
    let cursor_y = inner.y.saturating_add(min(
        editor.cursor_row as u16,
        inner.height.saturating_sub(1),
    ));
    frame.set_cursor_position(Position::new(cursor_x, cursor_y));
}

fn render_status(frame: &mut Frame, area: Rect, app: &App) {
    let left = if app.streaming {
        Line::from(vec![
            Span::styled("  ● ", Style::default().fg(ACCENT)),
            Span::styled(sanitize(&app.status), Style::default().fg(TEXT)),
        ])
    } else {
        Line::from(vec![
            Span::styled("  ● ", Style::default().fg(CYAN)),
            Span::styled(sanitize(&app.status), Style::default().fg(MUTED)),
        ])
    };
    let hint = if app.streaming {
        "esc abort  ·  type to steer"
    } else {
        "enter send · shift+enter newline · ctrl+l model · / commands"
    };
    let available = area.width as usize;
    let mut spans = left.spans.to_vec();
    let left_width = spans.iter().map(|span| span.content.width()).sum::<usize>();
    let hint_width = hint.width();
    if available > left_width + hint_width + 2 {
        spans.push(Span::raw(" ".repeat(available - left_width - hint_width)));
        spans.push(Span::styled(hint, Style::default().fg(FAINT)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_sidebar(frame: &mut Frame, area: Rect, lines: &[SidebarLine]) {
    let rendered = fit_sidebar(lines, area.height as usize)
        .iter()
        .map(|line| sidebar_line(line))
        .collect::<Vec<_>>();
    let sidebar = Paragraph::new(rendered)
        .style(Style::default().bg(PANEL_BACKGROUND))
        .wrap(Wrap { trim: true });
    frame.render_widget(sidebar, area);
}

fn fit_sidebar(lines: &[SidebarLine], height: usize) -> Vec<&SidebarLine> {
    if height == 0 || lines.len() <= height {
        return lines.iter().collect();
    }
    let mut blank_indices = lines
        .iter()
        .enumerate()
        .rev()
        .filter_map(|(index, line)| (line.kind == SidebarKind::Blank).then_some(index));
    let footer_start = blank_indices
        .nth(1)
        .unwrap_or_else(|| lines.len().saturating_sub(6));
    let footer = &lines[footer_start..];
    if footer.len() >= height {
        return footer[footer.len() - height..].iter().collect();
    }
    let head_budget = height - footer.len();
    if head_budget <= 1 {
        return footer.iter().collect();
    }
    let mut result = lines[..footer_start.min(head_budget - 1)]
        .iter()
        .collect::<Vec<_>>();
    result.push(&SIDEBAR_ELLIPSIS);
    result.extend(footer);
    result
}

static SIDEBAR_ELLIPSIS: SidebarLine = SidebarLine {
    kind: SidebarKind::Meta,
    value: String::new(),
};

fn sidebar_line(line: &SidebarLine) -> Line<'static> {
    let value = sanitize(&line.value);
    if matches!(line.kind, SidebarKind::Meta) && value.is_empty() {
        return Line::from(Span::styled("  …", Style::default().fg(MUTED)));
    }
    match line.kind {
        SidebarKind::Title | SidebarKind::Section => Line::from(Span::styled(
            format!("  {}", value),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        SidebarKind::Accent => Line::from(Span::styled(
            format!("  {}", value),
            Style::default().fg(ACCENT),
        )),
        SidebarKind::Active => Line::from(vec![
            Span::styled("  ● ", Style::default().fg(GREEN)),
            Span::styled(value, Style::default().fg(TEXT)),
        ]),
        SidebarKind::Meta => Line::from(Span::styled(
            format!("  {}", value),
            Style::default().fg(MUTED),
        )),
        SidebarKind::Path => Line::from(Span::styled(
            format!("  {}", truncate_left(&value, 36)),
            Style::default().fg(MUTED),
        )),
        SidebarKind::Brand => Line::from(Span::styled(
            format!("  {}", value),
            Style::default().fg(GREEN),
        )),
        SidebarKind::Progress(percent) => {
            let cells = 24usize;
            let filled = min(cells, cells * percent as usize / 100);
            Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled("━".repeat(filled), Style::default().fg(ACCENT)),
                Span::styled("━".repeat(cells - filled), Style::default().fg(FAINT)),
            ])
        }
        SidebarKind::Todo { complete } => {
            let marker = if complete { "☑" } else { "☐" };
            let color = if complete { GREEN } else { MUTED };
            Line::from(vec![
                Span::styled(format!("  {marker} "), Style::default().fg(color)),
                Span::styled(
                    value,
                    Style::default().fg(if complete { MUTED } else { TEXT }),
                ),
            ])
        }
        SidebarKind::File { status } => {
            let (prefix, color) = match status {
                FileStatus::Added | FileStatus::Untracked => ("A ", GREEN),
                FileStatus::Modified => ("M ", AMBER),
                FileStatus::Deleted => ("D ", RED),
            };
            Line::from(vec![
                Span::styled(format!("  {prefix}"), Style::default().fg(color)),
                Span::styled(truncate_left(&value, 32), Style::default().fg(MUTED)),
            ])
        }
        SidebarKind::Blank => Line::from(""),
    }
}

fn transcript_lines(
    messages: &[Message],
    width: u16,
    tools_expanded: bool,
    hide_thinking: bool,
) -> Vec<Line<'static>> {
    transcript_lines_with_cache(messages, width, tools_expanded, hide_thinking, None)
}

fn transcript_lines_with_cache(
    messages: &[Message],
    width: u16,
    tools_expanded: bool,
    hide_thinking: bool,
    mut cache: Option<&mut MessageCache>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for message in messages {
        let mut rendered = match cache.as_deref_mut() {
            Some(cache) => cache.render(message, width, tools_expanded, hide_thinking),
            None => render_transcript_message(message, width, tools_expanded, hide_thinking),
        };
        lines.append(&mut rendered);
    }
    lines
}

fn render_transcript_message(
    message: &Message,
    width: u16,
    tools_expanded: bool,
    hide_thinking: bool,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    match message.role {
        MessageRole::User => {
            lines.push(styled_line(
                format!("  {}", sanitize(&message.text)),
                Style::default().fg(TEXT).bg(USER_BACKGROUND),
            ));
        }
        MessageRole::Assistant => {
            lines.extend(markdown_lines(
                &message.text,
                width,
                MarkdownRole::Assistant,
            ));
        }
        MessageRole::Thinking if hide_thinking => {
            lines.push(styled_line(
                "  Thinking…".to_owned(),
                Style::default().fg(MUTED).add_modifier(Modifier::DIM),
            ));
        }
        MessageRole::Thinking => {
            lines.extend(markdown_lines(&message.text, width, MarkdownRole::Thinking));
        }
        MessageRole::Tool => {
            let (icon, color) = if message.is_error {
                ("×", RED)
            } else {
                ("✓", CYAN)
            };
            let title = if message.title.is_empty() {
                "tool".to_owned()
            } else {
                sanitize(&message.title)
            };
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default().bg(TOOL_BACKGROUND)),
                Span::styled(icon, Style::default().fg(color).bg(TOOL_BACKGROUND)),
                Span::styled(
                    format!(" {title}"),
                    Style::default()
                        .fg(TEXT)
                        .bg(TOOL_BACKGROUND)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            let detail = if tools_expanded || message.is_error {
                if message.detail.is_empty() {
                    &message.text
                } else {
                    &message.detail
                }
            } else {
                &message.text
            };
            let max_lines = if tools_expanded || message.is_error {
                20
            } else {
                3
            };
            for (index, line) in sanitize(detail).lines().take(max_lines).enumerate() {
                lines.push(Line::from(vec![
                    Span::styled("    ", Style::default().bg(TOOL_BACKGROUND)),
                    Span::styled(
                        line.to_owned(),
                        Style::default().fg(MUTED).bg(TOOL_BACKGROUND),
                    ),
                ]));
                if index + 1 == max_lines && sanitize(detail).lines().count() > max_lines {
                    lines.push(styled_line(
                        "    … more lines (ctrl+o to expand)".to_owned(),
                        Style::default().fg(MUTED).bg(TOOL_BACKGROUND),
                    ));
                }
            }
        }
        MessageRole::Error => {
            lines.push(styled_line(
                "  Error".to_owned(),
                Style::default().fg(RED).add_modifier(Modifier::BOLD),
            ));
            lines.extend(message_lines(&message.text, Style::default().fg(RED), "  "));
        }
        MessageRole::Notice | MessageRole::Command => {
            let (symbol, color, label) = match message.role {
                MessageRole::Command => ("◇", VIOLET, "Command"),
                _ => ("i", CYAN, "Notice"),
            };
            lines.push(Line::from(vec![
                Span::styled(format!("  {symbol} "), Style::default().fg(color)),
                Span::styled(label, Style::default().fg(MUTED)),
            ]));
            lines.extend(message_lines(
                &message.text,
                Style::default().fg(MUTED),
                "  ",
            ));
        }
    }
    lines.push(Line::from(""));
    lines
}

fn markdown_lines(text: &str, width: u16, role: MarkdownRole) -> Vec<Line<'static>> {
    let mut lines = MarkdownRenderer::with_role(width.saturating_sub(2), role).render_lines(text);
    for line in &mut lines {
        line.spans.insert(0, Span::raw("  "));
    }
    lines
}

fn message_lines(text: &str, style: Style, prefix: &str) -> Vec<Line<'static>> {
    let sanitized = sanitize(text);
    let lines = sanitized.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return vec![styled_line(prefix.to_owned(), style)];
    }
    lines
        .into_iter()
        .map(|line| styled_line(format!("{prefix}{line}"), style))
        .collect()
}

fn styled_line(text: String, style: Style) -> Line<'static> {
    Line::from(Span::styled(text, style))
}

struct EditorWindow {
    lines: Vec<String>,
    cursor_row: usize,
    cursor_column: usize,
}

fn editor_window(input: &str, cursor: usize, width: usize) -> EditorWindow {
    if input.is_empty() {
        return EditorWindow {
            lines: vec![String::new()],
            cursor_row: 0,
            cursor_column: 0,
        };
    }
    let all_lines: Vec<&str> = input.split('\n').collect();
    let before_cursor = &input[..cursor];
    let cursor_line = before_cursor.bytes().filter(|byte| *byte == b'\n').count();
    let mut start = cursor_line.saturating_sub(2);
    if all_lines.len().saturating_sub(start) < 3 {
        start = all_lines.len().saturating_sub(3);
    }
    let end = min(all_lines.len(), start + 3);
    let cursor_text = before_cursor.rsplit('\n').next().unwrap_or_default();
    let (cursor_window, cursor_column) = horizontal_editor_window(cursor_text, width);
    EditorWindow {
        lines: all_lines[start..end]
            .iter()
            .enumerate()
            .map(|(index, line)| {
                if start + index == cursor_line {
                    let suffix = &line[cursor_text.len()..];
                    let mut value = cursor_window.clone();
                    append_to_width(&mut value, suffix, width);
                    value
                } else {
                    truncate(line, width)
                }
            })
            .collect(),
        cursor_row: cursor_line - start,
        cursor_column,
    }
}

fn horizontal_editor_window(before_cursor: &str, width: usize) -> (String, usize) {
    if width == 0 {
        return (String::new(), 0);
    }
    let mut start = before_cursor.len();
    while start > 0 {
        let previous = before_cursor[..start]
            .char_indices()
            .next_back()
            .map_or(0, |(index, _)| index);
        if before_cursor[previous..].width() > width {
            break;
        }
        start = previous;
    }
    let visible = before_cursor[start..].to_owned();
    let column = visible.width().min(width);
    (visible, column)
}

fn append_to_width(output: &mut String, suffix: &str, width: usize) {
    for character in suffix.chars() {
        let character_width = character.to_string().width();
        if output.width() + character_width > width {
            break;
        }
        output.push(character);
    }
}

fn suggestion_title(input: &str) -> &'static str {
    let input = input.to_lowercase();
    if input.starts_with("/model ") {
        " SELECT MODEL "
    } else if input.starts_with("/thinking ") {
        " THINKING LEVEL "
    } else if input.starts_with("/login ") {
        " ADD PROVIDER "
    } else if input.starts_with("/omni ") {
        " OMNIROUTE "
    } else if input.starts_with("/aperture ") {
        " APERTURE "
    } else if input.starts_with("/btw ") {
        " BTW "
    } else if input.starts_with("/ralph ") {
        " RALPH LOOP "
    } else {
        " COMMANDS "
    }
}

fn sanitize(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\u{1b}' && characters.peek() == Some(&'[') {
            characters.next();
            for parameter in characters.by_ref() {
                if ('@'..='~').contains(&parameter) {
                    break;
                }
            }
            continue;
        }
        if !character.is_control() || matches!(character, '\n' | '\t') {
            output.push(character);
        }
    }
    output
}

fn truncate(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_owned();
    }
    let mut output = String::new();
    for character in text.chars() {
        if output.width() + character.to_string().width() > width - 1 {
            break;
        }
        output.push(character);
    }
    output.push('…');
    output
}

fn truncate_left(text: &str, width: usize) -> String {
    if text.width() <= width {
        return text.to_owned();
    }
    if width <= 1 {
        return "…".to_owned();
    }
    let mut output = String::new();
    for character in text.chars().rev() {
        if output.width() + character.to_string().width() > width - 1 {
            break;
        }
        output.insert(0, character);
    }
    format!("…{output}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    #[test]
    fn renderer_draws_responsive_ratatui_layout() {
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        let mut app = App::new();
        app.set_input("/mo");
        let mut cache = MessageCache::default();

        terminal
            .draw(|frame| draw(frame, &app, &mut cache))
            .expect("draw");

        let output = terminal.backend().buffer().content();
        let text: String = output.iter().map(|cell| cell.symbol()).collect();
        assert!(text.contains("GOSHCODER"));
        assert!(text.contains("COMMANDS"));
        assert!(text.contains("New Session"));
    }

    #[test]
    fn renderer_removes_terminal_control_sequences() {
        let lines = transcript_lines(
            &[Message {
                role: MessageRole::Assistant,
                text: "hello\x1b[2Jworld".to_owned(),
                ..Message::default()
            }],
            80,
            false,
            false,
        );
        let output = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(output, "  helloworld");
    }

    #[test]
    fn message_cache_reuses_rendered_transcript_rows() {
        let messages = vec![
            Message {
                role: MessageRole::User,
                text: "first question".to_owned(),
                ..Message::default()
            },
            Message {
                role: MessageRole::Assistant,
                text: "# Heading\n\nSome **bold** text and `code`.".to_owned(),
                ..Message::default()
            },
            Message {
                role: MessageRole::Assistant,
                text: "second answer".to_owned(),
                ..Message::default()
            },
        ];
        let mut cache = MessageCache::default();

        let first = transcript_lines_with_cache(&messages, 80, false, false, Some(&mut cache));
        let second = transcript_lines_with_cache(&messages, 80, false, false, Some(&mut cache));

        assert_eq!(first, second);
        assert_eq!(cache.entries.len(), messages.len());
        assert_eq!(
            transcript_lines(&messages, 80, false, false),
            first,
            "cached rows must match an uncached render"
        );
    }

    #[test]
    fn message_cache_invalidates_when_transcript_layout_changes() {
        let messages = vec![Message {
            role: MessageRole::Assistant,
            text: "word ".repeat(40),
            ..Message::default()
        }];
        let mut cache = MessageCache::default();

        let wide = transcript_lines_with_cache(&messages, 100, false, false, Some(&mut cache));
        let narrow = transcript_lines_with_cache(&messages, 40, false, false, Some(&mut cache));

        assert_ne!(wide, narrow);
        assert_eq!(narrow, transcript_lines(&messages, 40, false, false));
    }

    #[test]
    fn editor_window_keeps_a_long_cursor_line_visible() {
        let input = "prefix-which-is-long";
        let editor = editor_window(input, input.len(), 8);

        assert_eq!(editor.lines, ["-is-long"]);
        assert_eq!(editor.cursor_column, 8);
    }

    #[test]
    fn fitted_sidebar_preserves_workspace_footer() {
        let mut lines = vec![
            SidebarLine::title("Session"),
            SidebarLine::section("Context"),
            SidebarLine::meta("120 tokens"),
        ];
        lines.extend((0..30).map(|_| SidebarLine {
            kind: SidebarKind::Todo { complete: false },
            value: "step".to_owned(),
        }));
        lines.extend([
            SidebarLine::blank(),
            SidebarLine::section("Workspace"),
            SidebarLine::meta("main"),
            SidebarLine::path("~/src"),
            SidebarLine::blank(),
            SidebarLine::brand("● GoshCoder"),
        ]);

        let fitted = fit_sidebar(&lines, 12);
        let values = fitted
            .iter()
            .map(|line| line.value.as_str())
            .collect::<Vec<_>>();
        assert!(values.contains(&"Workspace"));
        assert!(values.contains(&"● GoshCoder"));
        assert!(values.contains(&""));
    }
}
