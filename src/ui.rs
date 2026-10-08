//! Fullscreen rendering. The layout follows the GoshCoder interface mockup:
//! a two-row wordmark header, a bottom-anchored transcript whose cards are
//! separated by one blank row, the palette directly above a bordered
//! composer, a one-line status bar with context-sensitive key hints, and a
//! sidebar of session, context, activity, plan and workspace sections.
//!
//! pi renders into terminal scrollback rather than an alternate screen; the
//! scrolling keys (PgUp/PgDn, Ctrl-Home/Ctrl-End) follow its fullscreen
//! bindings in `packages/tui/src/keybindings.ts`.

use std::cmp::min;

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Direction, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::{
    markdown::{MarkdownRenderer, MarkdownRole},
    state::{App, FileStatus, Message, MessageRole, SidebarKind, SidebarLine},
};

const BACKGROUND: Color = Color::Rgb(10, 10, 10);
const PANEL_BACKGROUND: Color = Color::Rgb(20, 20, 20);
const USER_BACKGROUND: Color = Color::Rgb(30, 30, 30);
const TOOL_BACKGROUND: Color = Color::Rgb(24, 24, 24);
const SELECTED_BACKGROUND: Color = Color::Rgb(27, 67, 76);
const SELECTED_TEXT: Color = Color::Rgb(239, 248, 246);
const ACCENT: Color = Color::Rgb(255, 172, 92);
const VIOLET: Color = Color::Rgb(73, 166, 191);
const CYAN: Color = Color::Rgb(86, 182, 194);
const BLUE: Color = Color::Rgb(112, 174, 221);
const GREEN: Color = Color::Rgb(127, 216, 143);
const AMBER: Color = Color::Rgb(242, 201, 108);
const RED: Color = Color::Rgb(246, 116, 116);
const TEXT: Color = Color::Rgb(220, 230, 232);
const MUTED: Color = Color::Rgb(119, 143, 150);
const FAINT: Color = Color::Rgb(62, 87, 96);

/// Collapsed tool cards show this many output lines.
const COLLAPSED_TOOL_LINES: usize = 3;
/// Expanded cards show everything up to this bound, which only exists so a
/// multi-megabyte log cannot make every frame re-wrap it.
const EXPANDED_TOOL_LINES: usize = 2000;
/// Prose wider than this is hard to read; very wide terminals leave the rest
/// of the row empty instead.
const MAX_READING_WIDTH: u16 = 124;
/// The composer grows with its content up to this many rows.
const MAX_COMPOSER_ROWS: usize = 8;

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(BACKGROUND)),
        area,
    );
    if area.width < 20 || area.height < 8 {
        // Wrapped, not truncated, so the message survives a phone-width pane.
        let too_small = Paragraph::new("GoshCoder\nTerminal is too small")
            .style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            .alignment(Alignment::Center)
            .wrap(Wrap { trim: true });
        frame.render_widget(too_small, area);
        return;
    }

    let sidebar_width = sidebar_width(area.width);
    let regions = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(20),
            Constraint::Length(u16::from(sidebar_width > 0)),
            Constraint::Length(sidebar_width),
        ])
        .split(area);

    render_main(frame, regions[0], app, sidebar_width == 0);
    if sidebar_width > 0 {
        render_sidebar(frame, regions[2], &app.sidebar);
    }
}

/// The sidebar appears from 96 columns, as in the previous interface.
pub fn sidebar_width(total: u16) -> u16 {
    if total >= 96 {
        (total / 3).clamp(32, 42)
    } else {
        0
    }
}

fn render_main(frame: &mut Frame, area: Rect, app: &App, compact: bool) {
    let editor = editor_window(
        &app.input,
        app.cursor,
        // Two border cells, one padding cell each side, one cell kept free
        // so the cursor can sit after the last character.
        usize::from(area.width.saturating_sub(5)),
        app.prompt.as_ref().is_some_and(|prompt| prompt.secret),
    );
    let composer_height = editor.lines.len() as u16 + 2;
    let queued = queued_lines(app, area.width);
    let status = status_lines(app, area.width, compact);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(1),
            Constraint::Length(queued.len() as u16),
            Constraint::Length(composer_height),
            Constraint::Length(status.len() as u16),
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
                truncate(
                    &sanitize(&app.title),
                    area.width.saturating_sub(13) as usize
                )
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
    render_transcript(frame, body[0], app);
    if palette_height > 0 {
        render_suggestions(frame, body[1], app, &suggestions);
    }
    if !queued.is_empty() {
        frame.render_widget(
            Paragraph::new(queued).style(Style::default().bg(BACKGROUND)),
            chunks[2],
        );
    }
    render_editor(frame, chunks[3], app, &editor);
    frame.render_widget(
        Paragraph::new(status).style(Style::default().bg(BACKGROUND)),
        chunks[4],
    );
}

fn render_transcript(frame: &mut Frame, area: Rect, app: &App) {
    let width = area.width.min(MAX_READING_WIDTH);
    // Every line is pre-wrapped to the area, so one line is one row and the
    // scroll arithmetic below cannot push the newest content out of view.
    let lines = transcript_lines(&app.messages, width, app.tools_expanded, app.hide_thinking);
    let height = usize::from(area.height);
    app.last_transcript_height.set(area.height);
    let max_scroll = lines.len().saturating_sub(height);
    app.last_max_scroll
        .set(max_scroll.min(usize::from(u16::MAX)) as u16);

    // The offset counts rows from the bottom; while the user reads older
    // output, rows arriving below are added to it so the view stays put.
    let layout = (width, app.tools_expanded, app.hide_thinking);
    let previous = app
        .last_transcript_rows
        .replace(Some((lines.len(), layout)));
    if app.scroll == 0 {
        app.scroll_growth.set(0);
        app.unseen_output.set(false);
    } else if let Some((rows, previous_layout)) = previous
        && previous_layout == layout
        && lines.len() > rows
    {
        app.scroll_growth
            .set(app.scroll_growth.get() + lines.len() - rows);
        app.unseen_output.set(true);
    }
    let scroll = app.scroll_offset().min(max_scroll);
    let start = max_scroll.saturating_sub(scroll);
    let end = (start + height).min(lines.len());
    let mut visible = lines[start..end].to_vec();
    // Short transcripts sit on the composer, like a chat, not under the
    // header.
    if visible.len() < height {
        let mut padded = vec![Line::from(""); height - visible.len()];
        padded.append(&mut visible);
        visible = padded;
    }
    if scroll > 0 && app.unseen_output.get() && height > 1 {
        let hint = "↓ new output below · ctrl+end to jump";
        if let Some(last) = visible.last_mut() {
            *last = Line::from(Span::styled(
                pad_to(&format!("  {hint}"), usize::from(area.width)),
                Style::default()
                    .fg(ACCENT)
                    .bg(PANEL_BACKGROUND)
                    .add_modifier(Modifier::BOLD),
            ));
        }
    }
    let transcript =
        Paragraph::new(Text::from(visible)).style(Style::default().fg(TEXT).bg(BACKGROUND));
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
    // Label, two spaces, description in one colour, as the mockup draws the
    // pickers; an aligned column would push short labels' descriptions
    // away from them.
    let items: Vec<ListItem<'_>> = visible
        .iter()
        .map(|suggestion| {
            let label = sanitize(&suggestion.label);
            let room = usize::from(area.width)
                .saturating_sub(6)
                .saturating_sub(label.width());
            let text = format!(
                " {label}  {}",
                truncate(&sanitize(&suggestion.description), room)
            );
            ListItem::new(Line::from(text))
        })
        .collect();
    let title = match app.prompt.as_ref() {
        Some(prompt) => format!(" {} ", prompt.label.to_uppercase()),
        None => suggestion_title(&app.input).to_owned(),
    };
    let list = List::new(items)
        .style(Style::default().fg(TEXT).bg(BACKGROUND))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(VIOLET))
                .title(Line::from(title).style(Style::default().fg(MUTED))),
        )
        .highlight_style(
            Style::default()
                .bg(SELECTED_BACKGROUND)
                .fg(SELECTED_TEXT)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("▌");
    let mut state = ListState::default();
    state.select(Some(selected.saturating_sub(first)));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_editor(frame: &mut Frame, area: Rect, app: &App, editor: &EditorWindow) {
    let mut block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(CYAN));
    if let Some(prompt) = app.prompt.as_ref() {
        block = block.title(
            Line::from(format!(" {} ", sanitize(&prompt.label))).style(Style::default().fg(ACCENT)),
        );
    }
    let inner = block.inner(area);
    frame.render_widget(block, area);
    // One cell of padding inside the border, as in the mockup.
    let inner = Rect {
        x: inner.x.saturating_add(1),
        width: inner.width.saturating_sub(2),
        ..inner
    };
    let lines: Vec<Line<'_>> = if app.input.is_empty() {
        let placeholder = match app.prompt.as_ref() {
            Some(prompt) if !prompt.options.is_empty() => {
                "↑↓ choose, Enter confirms · esc cancels".to_owned()
            }
            Some(prompt) if !prompt.placeholder.is_empty() => {
                format!("Paste here, e.g. {}", prompt.placeholder)
            }
            Some(_) => "Type the answer and press Enter · esc cancels".to_owned(),
            None => "Tell GoshCoder what to build…  / for commands".to_owned(),
        };
        vec![Line::from(Span::styled(
            truncate(&placeholder, inner.width as usize),
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
        Paragraph::new(lines).style(Style::default().bg(BACKGROUND)),
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

/// The key hints for what the keyboard does right now.
fn status_hint(app: &App) -> &'static str {
    if app
        .prompt
        .as_ref()
        .is_some_and(|prompt| !prompt.options.is_empty())
    {
        "↑↓ select · enter choose · esc cancel"
    } else if app.prompt.is_some() {
        "enter submit · esc cancel"
    } else if !app.suggestions().is_empty() {
        "↑↓ select · enter choose · esc close"
    } else if app.streaming {
        "esc abort  ·  type to steer"
    } else {
        "enter send · ctrl+j newline · ctrl+l model · / commands"
    }
}

/// The status bar: activity on the left, key hints on the right. A status
/// too long to share the row wraps onto a second one instead of losing the
/// hints.
fn status_lines(app: &App, width: u16, compact: bool) -> Vec<Line<'static>> {
    let (dot, text_style) = if app.streaming {
        (ACCENT, Style::default().fg(TEXT))
    } else {
        (CYAN, Style::default().fg(MUTED))
    };
    let mut status = sanitize(&app.status).replace('\n', " ");
    if compact && !app.context_hint.is_empty() {
        status = format!("{status}  ·  {}", app.context_hint);
    }
    let hint = status_hint(app);
    let width = usize::from(width);
    let prefix = 4; // "  ● "
    let hint_room = hint.width() + 2;
    // The hints keep their place while the status fits beside them, or can
    // wrap into a reasonable column there (as the mockup's two-row Ctrl-C
    // warning does); a narrow terminal gives the row to the status instead.
    let beside = width.saturating_sub(prefix + hint_room);
    let fits_hint = status.width() <= beside || beside >= 28;
    let status_room = if fits_hint {
        beside
    } else {
        width.saturating_sub(prefix)
    };
    let mut rows = wrap_plain(&status, status_room.max(1));
    rows.truncate(2);
    if rows.len() == 2 && status.width() > status_room * 2 {
        rows[1] = truncate(&rows[1], status_room.saturating_sub(1).max(1));
    }
    let mut lines = Vec::new();
    for (index, row) in rows.into_iter().enumerate() {
        let mut spans = vec![
            Span::styled(
                if index == 0 { "  ● " } else { "    " },
                Style::default().fg(dot),
            ),
            Span::styled(row.clone(), text_style),
        ];
        if index == 0 && fits_hint {
            let used = prefix + row.width();
            spans.push(Span::raw(
                " ".repeat(width.saturating_sub(used + hint.width())),
            ));
            spans.push(Span::styled(hint, Style::default().fg(FAINT)));
        }
        lines.push(Line::from(spans));
    }
    lines
}

/// Queued steering and follow-up messages, newest last, above the composer.
fn queued_lines(app: &App, width: u16) -> Vec<Line<'static>> {
    const SHOWN: usize = 3;
    let mut lines = Vec::new();
    let room = usize::from(width).saturating_sub(14);
    for text in app.queued.iter().take(SHOWN) {
        let first = text.lines().next().unwrap_or_default();
        lines.push(Line::from(vec![
            Span::styled("  ↳ queued  ", Style::default().fg(AMBER)),
            Span::styled(truncate(&sanitize(first), room), Style::default().fg(MUTED)),
        ]));
    }
    if app.queued.len() > SHOWN {
        lines.push(Line::from(Span::styled(
            format!("    … {} more queued", app.queued.len() - SHOWN),
            Style::default().fg(MUTED),
        )));
    }
    lines
}

fn render_sidebar(frame: &mut Frame, area: Rect, lines: &[SidebarLine]) {
    // The mockup leaves one blank row above the session title.
    let area = Rect {
        y: area.y.saturating_add(1),
        height: area.height.saturating_sub(1),
        ..area
    };
    frame.render_widget(
        Block::default().style(Style::default().bg(PANEL_BACKGROUND)),
        Rect {
            y: area.y.saturating_sub(1),
            height: area.height + 1,
            ..area
        },
    );
    let rendered = fit_sidebar(lines, area.height as usize)
        .iter()
        .map(|line| sidebar_line(line, usize::from(area.width)))
        .collect::<Vec<_>>();
    let sidebar = Paragraph::new(rendered).style(Style::default().bg(PANEL_BACKGROUND));
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

fn sidebar_line(line: &SidebarLine, width: usize) -> Line<'static> {
    let value = sanitize(&line.value).replace('\n', " ");
    let room = width.saturating_sub(3);
    if matches!(line.kind, SidebarKind::Meta) && value.is_empty() {
        return Line::from(Span::styled("  …", Style::default().fg(MUTED)));
    }
    match line.kind {
        SidebarKind::Title | SidebarKind::Section => Line::from(Span::styled(
            format!("  {}", truncate(&value, room)),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        SidebarKind::Accent => Line::from(Span::styled(
            format!("  {}", truncate(&value, room)),
            Style::default().fg(ACCENT),
        )),
        SidebarKind::Active => Line::from(vec![
            Span::styled("  ● ", Style::default().fg(GREEN)),
            Span::styled(
                truncate(&value, room.saturating_sub(2)),
                Style::default().fg(TEXT),
            ),
        ]),
        SidebarKind::Meta => Line::from(Span::styled(
            format!("  {}", truncate(&value, room)),
            Style::default().fg(MUTED),
        )),
        SidebarKind::Path => Line::from(Span::styled(
            format!("  {}", truncate_left(&value, room)),
            Style::default().fg(MUTED),
        )),
        SidebarKind::Brand => Line::from(Span::styled(
            format!("  {value}"),
            Style::default().fg(GREEN),
        )),
        SidebarKind::Progress(percent) => {
            let cells = 24usize;
            // Any use at all shows one cell, as the mockup's 1% bar does.
            let filled = if percent == 0 {
                0
            } else {
                (cells * percent as usize / 100).clamp(1, cells)
            };
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
                    truncate(&value, room.saturating_sub(2)),
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
                Span::styled(
                    truncate_left(&value, room.saturating_sub(2)),
                    Style::default().fg(MUTED),
                ),
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
    let full = usize::from(width);
    // Two cells of margin on each side of prose.
    let inner = full.saturating_sub(4).max(1);
    let mut lines = Vec::new();
    for message in messages {
        match message.role {
            MessageRole::User => {
                let style = Style::default().fg(TEXT).bg(USER_BACKGROUND);
                for row in wrap_plain(&sanitize(&message.text), inner) {
                    lines.push(styled_line(pad_to(&format!("  {row}"), full), style));
                }
            }
            MessageRole::Assistant => {
                let mut rendered = markdown_lines(&message.text, width, MarkdownRole::Assistant);
                if message.streaming
                    && let Some(last) = rendered.last_mut()
                {
                    last.spans
                        .push(Span::styled("█", Style::default().fg(ACCENT)));
                }
                lines.extend(rendered);
            }
            MessageRole::Thinking if hide_thinking => {
                lines.push(styled_line(
                    "  Thinking…".to_owned(),
                    Style::default().fg(MUTED).add_modifier(Modifier::ITALIC),
                ));
            }
            MessageRole::Thinking => {
                lines.extend(markdown_lines(&message.text, width, MarkdownRole::Thinking));
            }
            MessageRole::Tool => tool_card(&mut lines, message, full, tools_expanded),
            MessageRole::Error => {
                lines.push(styled_line(
                    "  Error".to_owned(),
                    Style::default().fg(RED).add_modifier(Modifier::BOLD),
                ));
                lines.extend(message_lines(
                    &message.text,
                    Style::default().fg(RED),
                    "  ",
                    width,
                ));
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
                notice_body(&mut lines, &message.text, inner);
            }
            MessageRole::Summary => {
                let style = Style::default().fg(MUTED).add_modifier(Modifier::ITALIC);
                let text = sanitize(&message.text);
                let mut body = text.trim().to_owned();
                if !tools_expanded && body.contains('\n') {
                    let first = body.lines().next().unwrap_or_default().to_owned();
                    body = format!("{first} (ctrl+o to expand)");
                }
                let title = if message.title.is_empty() {
                    "summary"
                } else {
                    message.title.as_str()
                };
                for (index, row) in
                    wrap_plain(&format!("{title} · {body}"), inner.saturating_sub(2))
                        .into_iter()
                        .enumerate()
                {
                    let lead = if index == 0 { "  ↳ " } else { "    " };
                    lines.push(styled_line(format!("{lead}{row}"), style));
                }
            }
        }
        lines.push(Line::from(""));
    }
    lines
}

/// A full-width tool card: a bold title with a status icon, then the output
/// indented under it. Failed calls open by themselves, so the error is never
/// one keypress away.
fn tool_card(lines: &mut Vec<Line<'static>>, message: &Message, full: usize, tools_expanded: bool) {
    let card = Style::default().bg(TOOL_BACKGROUND);
    let running = message.text == "running…" && message.detail.is_empty();
    let (icon, color) = if message.is_error {
        ("×", RED)
    } else if running {
        ("●", ACCENT)
    } else {
        ("✓", CYAN)
    };
    let title = if message.title.is_empty() {
        "tool".to_owned()
    } else {
        sanitize(&message.title).replace('\n', " ")
    };
    let title_style = Style::default()
        .fg(TEXT)
        .bg(TOOL_BACKGROUND)
        .add_modifier(Modifier::BOLD);
    let title_room = full.saturating_sub(6).max(1);
    let mut title_rows = wrap_plain(&title, title_room);
    let first_title = title_rows.first().cloned().unwrap_or_default();
    lines.push(Line::from(vec![
        Span::styled("  ", card),
        Span::styled(icon, Style::default().fg(color).bg(TOOL_BACKGROUND)),
        Span::styled(
            pad_to(&format!(" {first_title}"), full.saturating_sub(3)),
            title_style,
        ),
    ]));
    for row in title_rows.drain(1..) {
        lines.push(Line::from(Span::styled(
            pad_to(&format!("    {row}"), full),
            title_style,
        )));
    }

    let expanded = tools_expanded || message.is_error;
    let full_detail = if message.detail.is_empty() {
        &message.text
    } else {
        &message.detail
    };
    let shown = sanitize(if expanded { full_detail } else { &message.text });
    let shown_lines = shown.lines().collect::<Vec<_>>();
    // The collapsed text is a prefix of the full detail, so the hint counts
    // what the full detail still holds.
    let total_lines = sanitize(full_detail).lines().count().max(shown_lines.len());
    let limit = if expanded {
        EXPANDED_TOOL_LINES
    } else {
        COLLAPSED_TOOL_LINES
    };
    let visible = shown_lines.len().min(limit);
    let detail_style = Style::default().fg(MUTED).bg(TOOL_BACKGROUND);
    let room = full.saturating_sub(6).max(1);
    for line in &shown_lines[..visible] {
        for row in wrap_plain(line, room) {
            lines.push(styled_line(
                pad_to(&format!("    {row}"), full),
                detail_style,
            ));
        }
    }
    let toggle = if tools_expanded {
        "ctrl+o to collapse"
    } else {
        "ctrl+o to expand"
    };
    if total_lines > visible {
        lines.push(styled_line(
            pad_to(
                &format!("    … {} more lines ({toggle})", total_lines - visible),
                full,
            ),
            detail_style,
        ));
    } else if tools_expanded && !message.is_error && total_lines > COLLAPSED_TOOL_LINES {
        lines.push(styled_line(
            pad_to(&format!("    ({toggle})"), full),
            detail_style,
        ));
    }
}

/// Notice text in muted prose. A line indented by four spaces is something
/// to act on, as in a device-code login: an address in blue, a code in bold
/// accent, both set apart by blank rows.
fn notice_body(lines: &mut Vec<Line<'static>>, text: &str, inner: usize) {
    let text = sanitize(text);
    if text.is_empty() {
        lines.push(Line::from("  "));
        return;
    }
    for paragraph in text.split('\n') {
        if let Some(emphasis) = paragraph.strip_prefix("    ")
            && !emphasis.trim().is_empty()
        {
            let emphasis = emphasis.trim();
            let style = if emphasis.starts_with("http://") || emphasis.starts_with("https://") {
                Style::default().fg(BLUE)
            } else {
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
            };
            lines.push(Line::from(""));
            for row in wrap_plain(emphasis, inner.saturating_sub(4).max(1)) {
                lines.push(styled_line(format!("      {row}"), style));
            }
            lines.push(Line::from(""));
            continue;
        }
        for row in wrap_plain(paragraph, inner) {
            lines.push(styled_line(format!("  {row}"), Style::default().fg(MUTED)));
        }
    }
}

/// Wraps plain text to `width` cells: words first, then a hard break inside
/// a word that does not fit on its own. Tabs and control characters are
/// expected to be gone already. A blank line stays a blank line.
fn wrap_plain(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut rows = Vec::new();
    for line in text.split('\n') {
        if line.width() <= width {
            rows.push(line.to_owned());
            continue;
        }
        let mut current = String::new();
        let mut current_width = 0;
        for word in line.split(' ') {
            let word_width = word.width();
            let separator = usize::from(!current.is_empty());
            if current_width + separator + word_width <= width {
                if separator == 1 {
                    current.push(' ');
                }
                current.push_str(word);
                current_width += separator + word_width;
                continue;
            }
            if !current.is_empty() {
                rows.push(std::mem::take(&mut current));
                current_width = 0;
            }
            if word_width <= width {
                current.push_str(word);
                current_width = word_width;
                continue;
            }
            // A single token wider than the row is split by cell width.
            for character in word.chars() {
                let character_width = character.width().unwrap_or(0);
                if current_width + character_width > width && !current.is_empty() {
                    rows.push(std::mem::take(&mut current));
                    current_width = 0;
                }
                current.push(character);
                current_width += character_width;
            }
        }
        rows.push(current);
    }
    rows
}

fn markdown_lines(text: &str, width: u16, role: MarkdownRole) -> Vec<Line<'static>> {
    let mut lines = MarkdownRenderer::with_role(width.saturating_sub(4), role).render_lines(text);
    // A reply ending in a newline would otherwise add a second blank row
    // before the next card.
    while lines
        .last()
        .is_some_and(|line| line.spans.iter().all(|span| span.content.trim().is_empty()))
    {
        lines.pop();
    }
    for line in &mut lines {
        line.spans.insert(0, Span::raw("  "));
    }
    lines
}

fn message_lines(text: &str, style: Style, prefix: &str, width: u16) -> Vec<Line<'static>> {
    let sanitized = sanitize(text);
    if sanitized.is_empty() {
        return vec![styled_line(prefix.to_owned(), style)];
    }
    let inner = usize::from(width).saturating_sub(prefix.width() + 2);
    wrap_plain(&sanitized, inner)
        .into_iter()
        .map(|line| styled_line(format!("{prefix}{line}"), style))
        .collect()
}

fn styled_line(text: String, style: Style) -> Line<'static> {
    Line::from(Span::styled(text, style))
}

/// Pads `text` with spaces to `width` cells so a background fills the row.
fn pad_to(text: &str, width: usize) -> String {
    let current = text.width();
    if current >= width {
        return text.to_owned();
    }
    format!("{text}{}", " ".repeat(width - current))
}

struct EditorWindow {
    lines: Vec<String>,
    cursor_row: usize,
    cursor_column: usize,
}

/// One display cell group of the composer: the byte where its character
/// starts, what is drawn, and how wide that is.
struct EditorCell {
    byte: usize,
    glyph: String,
    width: usize,
}

/// Lays the composer out as wrapped rows, at most [`MAX_COMPOSER_ROWS`],
/// scrolled so the cursor row is visible. Tabs stay tabs in the buffer and
/// are drawn as four spaces here, which is also where the cursor column is
/// measured, so the two always agree. A secret answer is drawn as dots.
fn editor_window(input: &str, cursor: usize, width: usize, secret: bool) -> EditorWindow {
    let width = width.max(1);
    let mut rows: Vec<Vec<EditorCell>> = Vec::new();
    let mut cursor_position = (0, 0);
    let mut line_start = 0;
    for logical in input.split('\n') {
        let cells = logical
            .char_indices()
            .map(|(offset, character)| {
                let glyph = if secret {
                    "•".to_owned()
                } else if character == '\t' {
                    "    ".to_owned()
                } else {
                    character.to_string()
                };
                let width = glyph.width();
                EditorCell {
                    byte: line_start + offset,
                    glyph,
                    width,
                }
            })
            .collect::<Vec<_>>();
        let line_end = line_start + logical.len();
        let wrapped = wrap_cells(cells, width);
        let row_count = wrapped.len();
        for (index, row) in wrapped.into_iter().enumerate() {
            let is_last = index + 1 == row_count;
            let start = row.first().map_or(line_end, |cell| cell.byte);
            let end = if is_last {
                line_end
            } else {
                row.last().map_or(line_end, |cell| {
                    cell.byte + input[cell.byte..].chars().next().map_or(0, char::len_utf8)
                })
            };
            // A cursor at a wrap point starts the next row; only the line's
            // last row owns the position after its final character.
            if cursor >= start && (cursor < end || (is_last && cursor == line_end)) {
                let column = row
                    .iter()
                    .take_while(|cell| cell.byte < cursor)
                    .map(|cell| cell.width)
                    .sum();
                cursor_position = (rows.len(), column);
            }
            rows.push(row);
        }
        line_start = line_end + 1;
    }
    if rows.is_empty() {
        rows.push(Vec::new());
    }
    let (cursor_row, cursor_column) = cursor_position;
    let window = rows.len().min(MAX_COMPOSER_ROWS);
    let first = cursor_row
        .saturating_add(1)
        .saturating_sub(window)
        .min(rows.len() - window);
    EditorWindow {
        lines: rows[first..first + window]
            .iter()
            .map(|cells| cells.iter().map(|cell| cell.glyph.as_str()).collect())
            .collect(),
        cursor_row: cursor_row - first,
        cursor_column,
    }
}

/// Word-wraps composer cells: a row breaks after its last space when it has
/// one, otherwise at the width.
fn wrap_cells(cells: Vec<EditorCell>, width: usize) -> Vec<Vec<EditorCell>> {
    let mut rows = Vec::new();
    let mut current: Vec<EditorCell> = Vec::new();
    let mut current_width = 0;
    for cell in cells {
        if current_width + cell.width > width && !current.is_empty() {
            let split = current
                .iter()
                .rposition(|cell| cell.glyph == " ")
                .filter(|index| index + 1 < current.len())
                .map_or(current.len(), |index| index + 1);
            let carry = current.split_off(split);
            rows.push(std::mem::take(&mut current));
            current_width = carry.iter().map(|cell| cell.width).sum();
            current = carry;
        }
        current_width += cell.width;
        current.push(cell);
    }
    rows.push(current);
    rows
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
        // ratatui drops control characters when it writes cells, so a tab
        // has to become the spaces it stands for.
        if character == '\t' {
            output.push_str("    ");
        } else if !character.is_control() || character == '\n' {
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
    let mut used = 0;
    for character in text.chars() {
        let character_width = character.width().unwrap_or(0);
        if used + character_width > width - 1 {
            break;
        }
        used += character_width;
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
    use crate::state::{ComposerPrompt, Suggestion};
    use ratatui::{Terminal, backend::TestBackend, buffer::Buffer};

    fn render(app: &App, width: u16, height: u16) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("terminal");
        terminal.draw(|frame| draw(frame, app)).expect("draw");
        terminal.backend().buffer().clone()
    }

    fn rows(buffer: &Buffer) -> Vec<String> {
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect()
    }

    fn row_of(buffer: &Buffer, needle: &str) -> Option<u16> {
        rows(buffer)
            .iter()
            .position(|row| row.contains(needle))
            .map(|row| row as u16)
    }

    fn plain(lines: &[Line<'static>]) -> Vec<String> {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect()
    }

    fn message(role: MessageRole, text: &str) -> Message {
        Message {
            role,
            text: text.to_owned(),
            ..Message::default()
        }
    }

    fn quiet_app() -> App {
        let mut app = App::new();
        app.replace_messages(Vec::new());
        app.title = "Fix flaky session lock test".to_owned();
        app
    }

    #[test]
    fn layout_has_the_wordmark_title_composer_status_and_sidebar() {
        let mut app = quiet_app();
        app.set_input("/mo");
        let buffer = render(&app, 110, 30);
        let screen = rows(&buffer).join("\n");
        assert!(
            screen.contains("GOSHCODER  Fix flaky session lock test"),
            "{screen}"
        );
        assert!(screen.contains("· · · · · ·"));
        assert!(screen.contains("COMMANDS"));
        assert!(screen.contains("/model  Open the model picker"));
        assert!(
            screen.contains("New Session"),
            "sidebar is shown at 110 columns"
        );
        // The palette is open, so the hints say how to use it.
        assert!(screen.contains("↑↓ select · enter choose · esc close"));
        assert!(!screen.contains("enter send"));

        // Without a sidebar below 96 columns, and with idle hints.
        let mut app = quiet_app();
        app.status = "Ready".to_owned();
        let buffer = render(&app, 90, 20);
        let screen = rows(&buffer).join("\n");
        assert!(!screen.contains("New Session"));
        assert!(screen.contains("enter send · ctrl+j newline"));
        assert!(screen.contains("Tell GoshCoder what to build…"));
    }

    #[test]
    fn streaming_hints_and_a_composer_question_change_the_status_bar() {
        let mut app = quiet_app();
        app.streaming = true;
        app.status = "⠙ Thinking · 4s".to_owned();
        let screen = rows(&render(&app, 90, 20)).join("\n");
        assert!(screen.contains("● ⠙ Thinking · 4s"));
        assert!(screen.contains("esc abort  ·  type to steer"));

        let mut app = quiet_app();
        app.prompt = Some(ComposerPrompt::text("API key for openai", true));
        app.set_input("sk-secret");
        let screen = rows(&render(&app, 90, 20)).join("\n");
        assert!(screen.contains("API key for openai"));
        assert!(screen.contains("•••••••••"));
        assert!(!screen.contains("sk-secret"), "a secret is never drawn");
        assert!(screen.contains("enter submit · esc cancel"));
    }

    #[test]
    fn a_long_status_wraps_beside_the_hints_or_takes_the_row() {
        let mut app = quiet_app();
        app.status = "Press Ctrl+C again to exit (this session is not being saved)".to_owned();
        let screen = rows(&render(&app, 140, 20));
        let last = &screen[screen.len() - 2..];
        assert!(last[0].contains("Press Ctrl+C again"), "{last:#?}");
        assert!(last[0].contains("enter send"), "{last:#?}");
        assert!(last[1].contains("saved"), "{last:#?}");
        // Too narrow to share: the status gets the row, nothing collides.
        let screen = rows(&render(&app, 60, 20)).join("\n");
        assert!(screen.contains("Press Ctrl+C again to exit"), "{screen}");
    }

    #[test]
    fn a_short_transcript_sits_on_the_composer() {
        let mut app = quiet_app();
        app.replace_messages(vec![message(MessageRole::Assistant, "LAST ANSWER")]);
        let buffer = render(&app, 90, 30);
        let answer = row_of(&buffer, "LAST ANSWER").expect("answer drawn");
        let composer = row_of(&buffer, "Tell GoshCoder").expect("composer drawn");
        // One blank row between the last card and the composer border.
        assert_eq!(composer - answer, 3, "{:#?}", rows(&buffer));
    }

    #[test]
    fn scrolled_up_view_stays_on_its_content_while_output_arrives() {
        let mut app = quiet_app();
        let mut messages = (0..40)
            .map(|index| message(MessageRole::Assistant, &format!("line {index}")))
            .collect::<Vec<_>>();
        app.replace_messages(messages.clone());
        render(&app, 90, 24);
        app.scroll_up(12);
        let before = rows(&render(&app, 90, 24));
        let top_line = before
            .iter()
            .find(|row| row.contains("line "))
            .cloned()
            .expect("visible line");

        // A reply streams in below the viewport.
        messages.extend(
            (40..45).map(|index| message(MessageRole::Assistant, &format!("line {index}"))),
        );
        app.replace_messages(messages.clone());
        let after = rows(&render(&app, 90, 24));
        assert_eq!(
            after.iter().find(|row| row.contains("line ")),
            Some(&top_line),
            "the viewport must not drift"
        );
        let screen = after.join("\n");
        assert!(
            screen.contains("↓ new output below · ctrl+end to jump"),
            "{screen}"
        );

        // Back at the bottom the newest line shows and the hint is gone.
        app.scroll_to_bottom();
        let bottom = rows(&render(&app, 90, 24)).join("\n");
        assert!(bottom.contains("line 44"));
        assert!(!bottom.contains("new output below"));
    }

    #[test]
    fn output_at_the_bottom_shows_no_new_output_hint() {
        let mut app = quiet_app();
        app.replace_messages(vec![message(MessageRole::Assistant, "one")]);
        render(&app, 90, 20);
        app.replace_messages(vec![
            message(MessageRole::Assistant, "one"),
            message(MessageRole::Assistant, "two"),
        ]);
        let screen = rows(&render(&app, 90, 20)).join("\n");
        assert!(screen.contains("two"));
        assert!(!screen.contains("new output below"));
    }

    #[test]
    fn expanding_tool_cards_while_scrolled_is_not_new_output() {
        let mut app = quiet_app();
        let mut messages = (0..30)
            .map(|index| message(MessageRole::Assistant, &format!("line {index}")))
            .collect::<Vec<_>>();
        messages.push(Message {
            role: MessageRole::Tool,
            title: "bash seq".to_owned(),
            text: "1\n2\n3".to_owned(),
            detail: (1..=20)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
            ..Message::default()
        });
        app.replace_messages(messages);
        render(&app, 90, 24);
        app.scroll_up(5);
        render(&app, 90, 24);
        app.tools_expanded = true;
        let screen = rows(&render(&app, 90, 24)).join("\n");
        assert!(!screen.contains("new output below"), "{screen}");
    }

    #[test]
    fn tool_cards_collapse_to_three_lines_and_expand_to_the_whole_output() {
        let detail = (1..=40)
            .map(|index| format!("line {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let card = Message {
            role: MessageRole::Tool,
            title: "bash seq 1 40".to_owned(),
            text: detail.lines().take(3).collect::<Vec<_>>().join("\n"),
            detail: detail.clone(),
            ..Message::default()
        };
        let collapsed = plain(&transcript_lines(
            std::slice::from_ref(&card),
            80,
            false,
            false,
        ));
        assert!(collapsed[0].starts_with("  ✓ bash seq 1 40"));
        assert!(collapsed[1].starts_with("    line 1"));
        assert!(
            collapsed
                .iter()
                .any(|row| row.trim_end() == "    … 37 more lines (ctrl+o to expand)"),
            "{collapsed:#?}"
        );
        assert!(!collapsed.iter().any(|row| row.trim_end() == "    line 4"));

        let expanded = plain(&transcript_lines(
            std::slice::from_ref(&card),
            80,
            true,
            false,
        ));
        assert!(expanded.iter().any(|row| row.trim_end() == "    line 40"));
        assert!(!expanded.iter().any(|row| row.contains("more lines")));
        assert!(
            expanded
                .iter()
                .any(|row| row.trim_end() == "    (ctrl+o to collapse)")
        );
        assert!(!expanded.iter().any(|row| row.contains("ctrl+o to expand")));

        // Past the very high bound the rest is counted, not drawn.
        let huge = (1..=2100)
            .map(|index| format!("l{index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let big = Message {
            detail: huge,
            ..card.clone()
        };
        let rows = plain(&transcript_lines(
            std::slice::from_ref(&big),
            80,
            true,
            false,
        ));
        assert!(
            rows.iter()
                .any(|row| row.trim_end() == "    … 100 more lines (ctrl+o to collapse)")
        );

        // Every card row is padded so its background spans the transcript.
        let lines = transcript_lines(std::slice::from_ref(&card), 80, false, false);
        assert_eq!(lines[1].width(), 80);
        assert_eq!(lines[1].spans[0].style.bg, Some(TOOL_BACKGROUND));
    }

    #[test]
    fn failed_tools_open_by_themselves() {
        let card = Message {
            role: MessageRole::Tool,
            title: "bash false".to_owned(),
            text: "error: one".to_owned(),
            detail: "error: one\ntwo\nthree\nfour\nexit status 1".to_owned(),
            is_error: true,
            ..Message::default()
        };
        let rows = plain(&transcript_lines(&[card], 80, false, false));
        assert!(rows[0].starts_with("  × bash false"));
        assert!(rows.iter().any(|row| row.trim_end() == "    exit status 1"));
    }

    #[test]
    fn user_turns_are_full_width_bars_and_summaries_are_not() {
        let lines = transcript_lines(
            &[
                message(
                    MessageRole::User,
                    "first line\nsecond line that is long enough to wrap",
                ),
                Message {
                    role: MessageRole::Summary,
                    title: "branch summary".to_owned(),
                    text: "rewound to before \"x\"".to_owned(),
                    ..Message::default()
                },
            ],
            24,
            false,
            false,
        );
        let rows = plain(&lines);
        assert_eq!(rows[0].trim_end(), "  first line");
        assert_eq!(rows[1].trim_end(), "  second line that is");
        assert!(lines[..3].iter().all(|line| line.width() == 24));
        assert_eq!(lines[0].spans[0].style.bg, Some(USER_BACKGROUND));
        let summary = lines
            .iter()
            .find(|line| plain(std::slice::from_ref(line))[0].contains("branch summary"))
            .expect("summary row");
        assert!(plain(std::slice::from_ref(summary))[0].starts_with("  ↳ branch summary ·"));
        assert_ne!(summary.spans[0].style.bg, Some(USER_BACKGROUND));
    }

    #[test]
    fn notices_set_device_code_addresses_and_codes_apart() {
        let lines = transcript_lines(
            &[message(
                MessageRole::Notice,
                "Open this address:\n    https://auth.example/device\nand enter the code\n    QWRT-8KDP",
            )],
            60,
            false,
            false,
        );
        let rows = plain(&lines);
        assert_eq!(rows[0], "  i Notice");
        let find = |needle: &str| {
            lines
                .iter()
                .find(|line| plain(std::slice::from_ref(line))[0].contains(needle))
                .cloned()
                .expect("row")
        };
        assert_eq!(find("https://").spans[0].style.fg, Some(BLUE));
        let code = find("QWRT");
        assert_eq!(code.spans[0].style.fg, Some(ACCENT));
        assert!(code.spans[0].style.add_modifier.contains(Modifier::BOLD));
        // Ordinary notice prose stays muted.
        assert_eq!(find("Open this").spans[0].style.fg, Some(MUTED));
    }

    #[test]
    fn streaming_replies_end_with_a_cursor_block() {
        let mut reply = message(MessageRole::Assistant, "partial answer");
        reply.streaming = true;
        let rows = plain(&transcript_lines(
            std::slice::from_ref(&reply),
            60,
            false,
            false,
        ));
        assert_eq!(rows[0], "  partial answer█");
        reply.streaming = false;
        let rows = plain(&transcript_lines(&[reply], 60, false, false));
        assert_eq!(rows[0], "  partial answer");
    }

    #[test]
    fn editor_expands_tabs_for_display_and_keeps_the_cursor_on_them() {
        let editor = editor_window("a\tb", 2, 20, false);
        assert_eq!(editor.lines, ["a    b"]);
        assert_eq!(editor.cursor_column, 5, "after the tab's four cells");
        let editor = editor_window("a\tb", 3, 20, false);
        assert_eq!(editor.cursor_column, 6);
    }

    #[test]
    fn editor_wraps_long_lines_and_grows_to_eight_rows() {
        let text = "also run it under cargo-zigbuild for the windows target";
        let editor = editor_window(text, text.len(), 20, false);
        assert!(editor.lines.len() > 1);
        assert!(
            editor.lines.iter().all(|line| line.width() <= 20),
            "{:?}",
            editor.lines
        );
        assert_eq!(editor.lines.concat(), text);
        assert_eq!(editor.cursor_row, editor.lines.len() - 1);
        assert_eq!(
            editor.cursor_column,
            editor.lines.last().expect("row").width()
        );

        let tall = (0..12)
            .map(|index| format!("row {index}"))
            .collect::<Vec<_>>()
            .join("\n");
        let editor = editor_window(&tall, tall.len(), 20, false);
        assert_eq!(editor.lines.len(), MAX_COMPOSER_ROWS);
        assert_eq!(editor.lines.last().map(String::as_str), Some("row 11"));
        assert_eq!(editor.cursor_row, MAX_COMPOSER_ROWS - 1);
        let editor = editor_window(&tall, 0, 20, false);
        assert_eq!(editor.lines[0], "row 0");
        assert_eq!((editor.cursor_row, editor.cursor_column), (0, 0));
    }

    #[test]
    fn too_small_message_wraps_instead_of_being_cut() {
        let screen = rows(&render(&quiet_app(), 12, 6)).join("\n");
        for word in ["GoshCoder", "Terminal", "is", "too", "small"] {
            assert!(screen.contains(word), "{word} missing from {screen}");
        }
    }

    #[test]
    fn palette_rows_show_label_and_description() {
        let mut app = quiet_app();
        app.dynamic_suggestions = vec![Suggestion {
            label: "anthropic".to_owned(),
            description: "Claude Pro / Max subscription · OAuth".to_owned(),
            value: "/login anthropic".to_owned(),
            execute: true,
        }];
        app.set_input("/login ");
        let screen = rows(&render(&app, 100, 24)).join("\n");
        assert!(screen.contains("ADD PROVIDER"));
        assert!(screen.contains("▌ anthropic  Claude Pro / Max subscription · OAuth"));
    }

    #[test]
    fn queued_messages_show_above_the_composer() {
        let mut app = quiet_app();
        app.streaming = true;
        app.queued = vec!["also run it under zigbuild".to_owned()];
        let buffer = render(&app, 90, 20);
        let queued = row_of(&buffer, "↳ queued  also run it under zigbuild").expect("queued row");
        let composer = row_of(&buffer, "Tell GoshCoder").expect("composer");
        assert_eq!(composer, queued + 2, "directly above the composer border");
    }

    #[test]
    fn wrap_plain_breaks_on_words_then_inside_long_tokens() {
        assert_eq!(wrap_plain("short", 10), ["short"]);
        assert_eq!(
            wrap_plain("one two three four", 9),
            ["one two", "three", "four"]
        );
        assert_eq!(wrap_plain("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        assert_eq!(wrap_plain("a\n\nb", 5), ["a", "", "b"]);
        assert_eq!(wrap_plain("你好世界", 4), ["你好", "世界"]);
    }

    #[test]
    fn renderer_removes_terminal_control_sequences() {
        let rows = plain(&transcript_lines(
            &[message(MessageRole::Assistant, "hello\x1b[2Jworld")],
            80,
            false,
            false,
        ));
        assert_eq!(rows[0], "  helloworld");
        assert_eq!(sanitize("a\tb"), "a    b");
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
    }
}
