use std::cmp::min;

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthStr;

use crate::state::{App, FileStatus, Message, MessageRole, SidebarKind, SidebarLine};

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

pub fn draw(frame: &mut Frame, app: &App) {
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
            .constraints([Constraint::Min(20), Constraint::Length(sidebar_width)])
            .split(area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(0)])
            .split(area)
    };
    let main = regions[0];

    render_main(frame, main, app);
    if sidebar_width > 0 {
        render_sidebar(frame, regions[1], &app.sidebar);
    }
}

fn render_main(frame: &mut Frame, area: Rect, app: &App) {
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

    render_transcript(frame, chunks[1], app);
    render_editor(frame, chunks[2], app, &editor);
    render_status(frame, chunks[3], app);
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let lines = transcript_lines(&app.messages, app.tools_expanded, app.hide_thinking);
    let transcript = Paragraph::new(Text::from(lines))
        .style(Style::default().fg(TEXT).bg(BACKGROUND))
        .scroll((app.scroll, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript, area);

    let suggestions = app.suggestions();
    if suggestions.is_empty() || area.width < 16 || area.height < 3 {
        return;
    }
    let popup_height = min(area.height.saturating_sub(1), suggestions.len() as u16 + 2);
    let popup_width = min(area.width.saturating_sub(4), 72);
    let popup = Rect {
        x: area.x.saturating_add(2),
        y: area
            .y
            .saturating_add(area.height.saturating_sub(popup_height)),
        width: popup_width,
        height: popup_height,
    };
    frame.render_widget(Clear, popup);
    let items: Vec<ListItem<'_>> = suggestions
        .iter()
        .map(|suggestion| {
            let text = format!(
                " {}  {}",
                suggestion.label,
                truncate(
                    &suggestion.description,
                    popup.width.saturating_sub(6) as usize
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
    state.select(Some(app.selected_suggestion.min(suggestions.len() - 1)));
    frame.render_stateful_widget(list, popup, &mut state);
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
    let rendered = lines.iter().map(sidebar_line).collect::<Vec<_>>();
    let sidebar = Paragraph::new(rendered)
        .style(Style::default().bg(PANEL_BACKGROUND))
        .wrap(Wrap { trim: true });
    frame.render_widget(sidebar, area);
}

fn sidebar_line(line: &SidebarLine) -> Line<'static> {
    let value = sanitize(&line.value);
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
    tools_expanded: bool,
    hide_thinking: bool,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for message in messages {
        match message.role {
            MessageRole::User => {
                lines.push(styled_line(
                    format!("  {}", sanitize(&message.text)),
                    Style::default().fg(TEXT).bg(USER_BACKGROUND),
                ));
            }
            MessageRole::Assistant => {
                lines.extend(message_lines(
                    &message.text,
                    Style::default().fg(TEXT),
                    "  ",
                ));
            }
            MessageRole::Thinking if hide_thinking => {
                lines.push(styled_line(
                    "  Thinking…".to_owned(),
                    Style::default().fg(MUTED).add_modifier(Modifier::DIM),
                ));
            }
            MessageRole::Thinking => {
                lines.extend(message_lines(
                    &message.text,
                    Style::default().fg(MUTED).add_modifier(Modifier::DIM),
                    "  ",
                ));
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
    let start = cursor_line.saturating_sub(2);
    let end = min(all_lines.len(), start + 3);
    let cursor_column = before_cursor
        .rsplit('\n')
        .next()
        .map_or(0, UnicodeWidthStr::width);
    EditorWindow {
        lines: all_lines[start..end]
            .iter()
            .map(|line| truncate(line, width))
            .collect(),
        cursor_row: cursor_line - start,
        cursor_column: min(cursor_column, width),
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
    text.chars()
        .filter(|character| !character.is_control() || matches!(character, '\n' | '\t'))
        .collect()
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

        terminal.draw(|frame| draw(frame, &app)).expect("draw");

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
            false,
            false,
        );
        let output = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(output, "  hello[2Jworld");
    }
}
