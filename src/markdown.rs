//! A small, streaming-friendly Markdown renderer for Ratatui transcript text.
//!
//! This intentionally recognizes the Markdown shapes that are useful in a
//! terminal transcript instead of trying to implement all of CommonMark.  It
//! produces owned [`Text`] so callers can cache or render the result without
//! retaining the input string.

use std::collections::BTreeMap;

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const ACCENT: Color = Color::Rgb(255, 172, 92);
const CYAN: Color = Color::Rgb(86, 182, 194);
const BLUE: Color = Color::Rgb(112, 174, 221);
const TEXT: Color = Color::Rgb(220, 230, 232);
const MUTED: Color = Color::Rgb(119, 143, 150);
const FAINT: Color = Color::Rgb(62, 87, 96);

/// The transcript role whose base style is applied to ordinary prose.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum MarkdownRole {
    /// Normal assistant output.
    #[default]
    Assistant,
    /// Deliberation output, rendered in a muted, dimmed style.
    Thinking,
}

/// Width-aware renderer for a single Markdown transcript entry.
///
/// The width is measured in terminal cells, not bytes. A zero width is treated
/// as one cell so rendering remains well-defined for collapsed layouts.
#[derive(Debug, Clone, Copy)]
pub struct MarkdownRenderer {
    width: usize,
    role: MarkdownRole,
}

impl MarkdownRenderer {
    /// Creates an assistant-output renderer for a Ratatui area width.
    #[must_use]
    pub const fn new(width: u16) -> Self {
        Self::with_role(width, MarkdownRole::Assistant)
    }

    /// Creates a renderer with an explicit transcript role.
    #[must_use]
    pub const fn with_role(width: u16, role: MarkdownRole) -> Self {
        Self {
            width: if width == 0 { 1 } else { width as usize },
            role,
        }
    }

    /// Renders Markdown into owned Ratatui text.
    #[must_use]
    pub fn render(self, source: &str) -> Text<'static> {
        Text::from(self.render_lines(source))
    }

    /// Renders Markdown into owned Ratatui lines.
    #[must_use]
    pub fn render_lines(self, source: &str) -> Vec<Line<'static>> {
        let source = sanitize_terminal_text(source).replace('\t', "    ");
        let raw_lines: Vec<&str> = source.split('\n').collect();
        let mut rendered = Vec::new();
        let mut fenced_by: Option<String> = None;
        let mut ordered_at: BTreeMap<usize, i64> = BTreeMap::new();
        let mut hang_at: BTreeMap<usize, usize> = BTreeMap::new();
        let mut in_quote = false;
        let mut index = 0;

        while index < raw_lines.len() {
            let raw = raw_lines[index];
            let trimmed = raw.trim();

            if let Some(marker) = parse_fence(trimmed) {
                match &fenced_by {
                    None => {
                        fenced_by = Some(marker);
                        reset_lists(&mut ordered_at, &mut hang_at);
                        in_quote = false;
                        index += 1;
                        continue;
                    }
                    Some(opening) if trimmed.starts_with(opening) => {
                        fenced_by = None;
                        index += 1;
                        continue;
                    }
                    Some(_) => {}
                }
            }
            if fenced_by.is_some() {
                self.push_code_line(raw, &mut rendered);
                index += 1;
                continue;
            }

            if is_table_row(raw) {
                let (rows, next_index) = collect_table(&raw_lines, index);
                self.push_table(&rows, &mut rendered);
                reset_lists(&mut ordered_at, &mut hang_at);
                in_quote = false;
                index = next_index;
                continue;
            }

            if trimmed.is_empty() {
                rendered.push(Line::from(""));
                in_quote = false;
                index += 1;
                continue;
            }

            if is_horizontal_rule(trimmed) {
                rendered.push(line_from_fragments(vec![Fragment::new(
                    "─".repeat(self.width.max(3)),
                    Style::default().fg(FAINT),
                )]));
                reset_lists(&mut ordered_at, &mut hang_at);
                in_quote = false;
                index += 1;
                continue;
            }

            if let Some(heading) = parse_atx_heading(trimmed) {
                append_wrapped(
                    &mut rendered,
                    inline_fragments(
                        heading,
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    self.width,
                    Vec::new(),
                    Vec::new(),
                );
                reset_lists(&mut ordered_at, &mut hang_at);
                in_quote = false;
                index += 1;
                continue;
            }

            if let Some(item) = parse_list_item(raw) {
                in_quote = false;
                let nest = item.indent / 2;
                let display = match item.marker {
                    ListMarker::Unordered => {
                        ordered_at.retain(|depth, _| *depth < nest);
                        hang_at.retain(|depth, _| *depth <= nest);
                        "-".to_owned()
                    }
                    ListMarker::Ordered { number, suffix } => {
                        let current = {
                            let current = ordered_at.entry(nest).or_insert(number - 1);
                            *current += 1;
                            *current
                        };
                        ordered_at.retain(|depth, _| *depth <= nest);
                        hang_at.retain(|depth, _| *depth <= nest);
                        format!("{current}{suffix}")
                    }
                };

                let (checked, content) = match parse_task(item.body) {
                    Some((checked, rest)) => (Some(checked), rest),
                    None => (None, item.body),
                };
                let prefix = list_prefix(nest, &display, checked);
                let hang = fragments_width(&prefix);
                hang_at.insert(nest, hang);

                if let Some((_, quote)) = parse_quote(content.trim()) {
                    append_wrapped(
                        &mut rendered,
                        inline_fragments(quote, Style::default().fg(MUTED)),
                        self.width.saturating_sub(hang + 2).max(1),
                        append_fragments_to(prefix, quote_prefix(1)),
                        append_fragments_to(spaces(hang), quote_prefix(1)),
                    );
                } else {
                    append_wrapped(
                        &mut rendered,
                        inline_fragments(content, self.body_style()),
                        self.width.saturating_sub(hang).max(1),
                        prefix,
                        spaces(hang),
                    );
                }

                index += 1;
                continue;
            }

            if !hang_at.is_empty() && leading_width(raw) >= 2 {
                in_quote = false;
                let hang = deepest_hang(&hang_at);
                let content = raw.trim();
                if let Some((_, quote)) = parse_quote(content) {
                    append_wrapped(
                        &mut rendered,
                        inline_fragments(quote, Style::default().fg(MUTED)),
                        self.width.saturating_sub(hang + 2).max(1),
                        append_fragments_to(spaces(hang), quote_prefix(1)),
                        append_fragments_to(spaces(hang), quote_prefix(1)),
                    );
                } else {
                    append_wrapped(
                        &mut rendered,
                        inline_fragments(content, self.body_style()),
                        self.width.saturating_sub(hang).max(1),
                        spaces(hang),
                        spaces(hang),
                    );
                }

                index += 1;
                continue;
            }

            if index + 1 < raw_lines.len() && is_setext_underline(raw_lines[index + 1].trim()) {
                let heading = raw.trim();
                if !heading.is_empty() {
                    append_wrapped(
                        &mut rendered,
                        inline_fragments(
                            heading,
                            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                        ),
                        self.width,
                        Vec::new(),
                        Vec::new(),
                    );
                    reset_lists(&mut ordered_at, &mut hang_at);
                    in_quote = false;
                    index += 2;
                    continue;
                }
            }

            reset_lists(&mut ordered_at, &mut hang_at);
            if let Some((depth, quote)) = parse_quote(trimmed) {
                self.push_quote(depth, quote, &mut rendered);
                in_quote = true;
                index += 1;
                continue;
            }
            if in_quote {
                self.push_quote(1, trimmed, &mut rendered);
                index += 1;
                continue;
            }

            in_quote = false;
            if leading_width(raw) >= 4 {
                self.push_code_line(strip_indent_prefix(raw, 4), &mut rendered);
            } else {
                append_wrapped(
                    &mut rendered,
                    inline_fragments(raw.trim_end_matches(' '), self.body_style()),
                    self.width,
                    Vec::new(),
                    Vec::new(),
                );
            }
            index += 1;
        }

        rendered
    }

    fn body_style(self) -> Style {
        match self.role {
            MarkdownRole::Assistant => Style::default().fg(TEXT),
            MarkdownRole::Thinking => Style::default().fg(MUTED).add_modifier(Modifier::DIM),
        }
    }

    fn push_code_line(self, raw: &str, rendered: &mut Vec<Line<'static>>) {
        let content = vec![Fragment::new(raw, Style::default().fg(BLUE))];
        for row in hard_wrap(&content, self.width.saturating_sub(2).max(1)) {
            append_line(rendered, quote_prefix(1), row);
        }
    }

    fn push_quote(self, depth: usize, quote: &str, rendered: &mut Vec<Line<'static>>) {
        let prefix = quote_prefix(depth);
        append_wrapped(
            rendered,
            inline_fragments(quote, Style::default().fg(MUTED)),
            self.width.saturating_sub(fragments_width(&prefix)).max(1),
            prefix.clone(),
            prefix,
        );
    }

    fn push_table(self, rows: &[Vec<String>], rendered: &mut Vec<Line<'static>>) {
        if rows.is_empty() {
            return;
        }

        let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
        if columns == 0 {
            return;
        }

        let cell_fragments: Vec<Vec<Vec<Fragment>>> = rows
            .iter()
            .map(|row| {
                (0..columns)
                    .map(|column| {
                        inline_fragments(
                            row.get(column).map_or("", String::as_str),
                            Style::default().fg(TEXT),
                        )
                    })
                    .collect()
            })
            .collect();
        let widths = table_widths(&cell_fragments, self.width);

        rendered.push(table_rule(&widths, "┌", "┬", "┐"));
        for (row_index, row) in cell_fragments.iter().enumerate() {
            let wrapped_cells: Vec<Vec<Vec<Fragment>>> = row
                .iter()
                .zip(&widths)
                .map(|(cell, width)| wrap_words(cell, *width))
                .collect();
            let height = wrapped_cells.iter().map(Vec::len).max().unwrap_or(1);

            for line_index in 0..height {
                let mut fragments = vec![Fragment::new("│", Style::default().fg(FAINT))];
                for (column, width) in widths.iter().enumerate() {
                    fragments.push(Fragment::new(" ", Style::default().fg(TEXT)));
                    let cell = wrapped_cells[column]
                        .get(line_index)
                        .cloned()
                        .unwrap_or_default();
                    let cell_width = fragments_width(&cell);
                    append_fragments(&mut fragments, cell);
                    if cell_width < *width {
                        fragments.push(Fragment::new(
                            " ".repeat(*width - cell_width),
                            Style::default().fg(TEXT),
                        ));
                    }
                    fragments.push(Fragment::new(" ", Style::default().fg(TEXT)));
                    fragments.push(Fragment::new("│", Style::default().fg(FAINT)));
                }
                rendered.push(line_from_fragments(fragments));
            }

            if row_index + 1 < cell_fragments.len() {
                rendered.push(table_rule(&widths, "├", "┼", "┤"));
            }
        }
        rendered.push(table_rule(&widths, "└", "┴", "┘"));
    }
}

/// Renders assistant Markdown into owned Ratatui text.
#[must_use]
pub fn render_markdown(source: &str, width: u16) -> Text<'static> {
    MarkdownRenderer::new(width).render(source)
}

/// Renders Markdown for an explicit transcript role into owned Ratatui text.
#[must_use]
pub fn render_markdown_with_role(source: &str, width: u16, role: MarkdownRole) -> Text<'static> {
    MarkdownRenderer::with_role(width, role).render(source)
}

/// Renders assistant Markdown into owned Ratatui lines.
#[must_use]
pub fn render_markdown_lines(source: &str, width: u16) -> Vec<Line<'static>> {
    MarkdownRenderer::new(width).render_lines(source)
}

/// Removes terminal control characters and escape sequences from untrusted text.
///
/// Newlines and tabs are retained because they carry Markdown structure. Tabs
/// are expanded to four spaces by [`MarkdownRenderer::render_lines`].
#[must_use]
pub fn sanitize_terminal_text(source: &str) -> String {
    let characters: Vec<char> = source.chars().collect();
    let mut sanitized = String::with_capacity(source.len());
    let mut index = 0;

    while index < characters.len() {
        match characters[index] {
            '\u{1b}' => {
                index = consume_escape_sequence(&characters, index + 1);
            }
            '\u{009b}' => {
                index = consume_csi(&characters, index + 1);
            }
            '\u{009d}' | '\u{0090}' | '\u{0098}' | '\u{009e}' | '\u{009f}' => {
                index = consume_string_control(&characters, index + 1);
            }
            character if character == '\n' || character == '\t' || !character.is_control() => {
                sanitized.push(character);
                index += 1;
            }
            _ => {
                index += 1;
            }
        }
    }

    sanitized
}

#[derive(Clone)]
struct Fragment {
    text: String,
    style: Style,
}

impl Fragment {
    fn new(text: impl Into<String>, style: Style) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }
}

#[derive(Clone, Copy)]
enum ListMarker {
    Unordered,
    Ordered { number: i64, suffix: char },
}

struct ParsedListItem<'a> {
    indent: usize,
    marker: ListMarker,
    body: &'a str,
}

struct Token {
    fragments: Vec<Fragment>,
    whitespace: bool,
    width: usize,
}

fn consume_escape_sequence(characters: &[char], index: usize) -> usize {
    let Some(&kind) = characters.get(index) else {
        return index;
    };
    match kind {
        '[' => consume_csi(characters, index + 1),
        ']' | 'P' | 'X' | '^' | '_' => consume_string_control(characters, index + 1),
        _ => index + 1,
    }
}

fn consume_csi(characters: &[char], mut index: usize) -> usize {
    while index < characters.len() {
        let code = characters[index] as u32;
        index += 1;
        if (0x40..=0x7e).contains(&code) {
            break;
        }
    }
    index
}

fn consume_string_control(characters: &[char], mut index: usize) -> usize {
    while index < characters.len() {
        match characters[index] {
            '\u{0007}' | '\u{009c}' => return index + 1,
            '\u{1b}' if characters.get(index + 1) == Some(&'\\') => return index + 2,
            _ => index += 1,
        }
    }
    index
}

fn reset_lists(ordered_at: &mut BTreeMap<usize, i64>, hang_at: &mut BTreeMap<usize, usize>) {
    ordered_at.clear();
    hang_at.clear();
}

fn parse_fence(trimmed: &str) -> Option<String> {
    let marker = trimmed.chars().next()?;
    if marker != '`' && marker != '~' {
        return None;
    }
    let count = trimmed
        .chars()
        .take_while(|character| *character == marker)
        .count();
    (count >= 3).then(|| marker.to_string().repeat(count))
}

fn parse_atx_heading(trimmed: &str) -> Option<&str> {
    let count = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if !(1..=6).contains(&count) {
        return None;
    }
    let after_marker = &trimmed[count..];
    if !starts_with_whitespace(after_marker) {
        return None;
    }
    let without_leading_space = after_marker.trim_start();
    if without_leading_space.is_empty() {
        return None;
    }

    let without_trailing_space = without_leading_space.trim_end();
    let trailing_hashes = without_trailing_space
        .chars()
        .rev()
        .take_while(|character| *character == '#')
        .count();
    if trailing_hashes == 0 {
        return Some(without_trailing_space);
    }

    let before_hashes = &without_trailing_space[..without_trailing_space.len() - trailing_hashes];
    if before_hashes
        .chars()
        .last()
        .is_some_and(char::is_whitespace)
    {
        Some(before_hashes.trim_end())
    } else {
        Some(without_trailing_space)
    }
}

fn is_setext_underline(trimmed: &str) -> bool {
    let Some(marker) = trimmed.chars().next() else {
        return false;
    };
    matches!(marker, '=' | '-') && trimmed.chars().all(|character| character == marker)
}

fn is_horizontal_rule(trimmed: &str) -> bool {
    let compact: String = trimmed
        .chars()
        .filter(|character| *character != ' ' && *character != '\t')
        .collect();
    let Some(marker) = compact.chars().next() else {
        return false;
    };
    compact.chars().count() >= 3
        && matches!(marker, '-' | '*' | '_')
        && compact.chars().all(|character| character == marker)
}

fn parse_list_item(raw: &str) -> Option<ParsedListItem<'_>> {
    let mut byte_index = 0;
    for (index, character) in raw.char_indices() {
        if character == ' ' || character == '\t' {
            byte_index = index + character.len_utf8();
        } else {
            break;
        }
    }
    let rest = &raw[byte_index..];
    let bytes = rest.as_bytes();
    let first = *bytes.first()?;

    if matches!(first, b'-' | b'*' | b'+') {
        let body = list_body(&rest[1..])?;
        return Some(ParsedListItem {
            indent: leading_width(&raw[..byte_index]),
            marker: ListMarker::Unordered,
            body,
        });
    }

    let digits = bytes
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if digits == 0 || digits == bytes.len() {
        return None;
    }
    let suffix = bytes[digits];
    if suffix != b'.' && suffix != b')' {
        return None;
    }
    let body = list_body(&rest[digits + 1..])?;
    let number = rest[..digits].parse::<i64>().ok()?;
    Some(ParsedListItem {
        indent: leading_width(&raw[..byte_index]),
        marker: ListMarker::Ordered {
            number,
            suffix: suffix as char,
        },
        body,
    })
}

fn list_body(rest: &str) -> Option<&str> {
    if rest.is_empty() {
        return Some("");
    }
    starts_with_whitespace(rest).then(|| rest.trim_start_matches(char::is_whitespace))
}

fn parse_task(body: &str) -> Option<(bool, &str)> {
    let bytes = body.as_bytes();
    if bytes.len() < 4 || bytes[0] != b'[' || bytes[2] != b']' {
        return None;
    }
    let checked = match bytes[1] {
        b' ' => false,
        b'x' | b'X' => true,
        _ => return None,
    };
    starts_with_whitespace(&body[3..])
        .then(|| (checked, body[3..].trim_start_matches(char::is_whitespace)))
}

fn parse_quote(trimmed: &str) -> Option<(usize, &str)> {
    let mut depth = 0;
    let mut rest = trimmed;
    while let Some(after_marker) = rest.strip_prefix('>') {
        depth += 1;
        rest = after_marker.strip_prefix(' ').unwrap_or(after_marker);
    }
    (depth > 0).then_some((depth, rest))
}

fn starts_with_whitespace(text: &str) -> bool {
    text.chars().next().is_some_and(char::is_whitespace)
}

fn leading_width(text: &str) -> usize {
    text.chars()
        .take_while(|character| *character == ' ' || *character == '\t')
        .map(|character| if character == '\t' { 4 } else { 1 })
        .sum()
}

fn strip_indent_prefix(raw: &str, mut width: usize) -> &str {
    let mut end = 0;
    for (index, character) in raw.char_indices() {
        if width == 0 || character != ' ' {
            break;
        }
        width -= 1;
        end = index + character.len_utf8();
    }
    &raw[end..]
}

fn deepest_hang(hang_at: &BTreeMap<usize, usize>) -> usize {
    hang_at.last_key_value().map_or(2, |(_, hang)| *hang)
}

fn list_prefix(nest: usize, display: &str, checked: Option<bool>) -> Vec<Fragment> {
    let mut prefix = spaces(nest * 4);
    let marker = match checked {
        Some(true) => format!("{display} [x]"),
        Some(false) => format!("{display} [ ]"),
        None => display.to_owned(),
    };
    prefix.push(Fragment::new(marker, Style::default().fg(CYAN)));
    prefix.push(Fragment::new(" ", Style::default()));
    prefix
}

fn quote_prefix(depth: usize) -> Vec<Fragment> {
    vec![Fragment::new(
        "│ ".repeat(depth.max(1)),
        Style::default().fg(FAINT),
    )]
}

fn spaces(width: usize) -> Vec<Fragment> {
    if width == 0 {
        Vec::new()
    } else {
        vec![Fragment::new(" ".repeat(width), Style::default())]
    }
}

fn inline_fragments(text: &str, style: Style) -> Vec<Fragment> {
    inline_fragments_at_depth(text, style, 0)
}

fn inline_fragments_at_depth(text: &str, style: Style, depth: usize) -> Vec<Fragment> {
    if text.is_empty() {
        return Vec::new();
    }
    if depth >= 16 {
        return vec![Fragment::new(text, style)];
    }

    let mut fragments = Vec::new();
    let mut plain_start = 0;
    let mut cursor = 0;
    while cursor < text.len() {
        if is_escaped(text, cursor) {
            cursor += next_character_len(text, cursor);
            continue;
        }

        if text[cursor..].starts_with('`')
            && let Some(end) = find_unescaped(text, cursor + 1, "`")
            && end > cursor + 1
        {
            push_fragment(&mut fragments, &text[plain_start..cursor], style);
            push_fragment(&mut fragments, &text[cursor + 1..end], style.fg(BLUE));
            cursor = end + 1;
            plain_start = cursor;
            continue;
        }

        if text[cursor..].starts_with('[')
            && let Some(label_end) = find_unescaped(text, cursor + 1, "]")
            && text[label_end..].starts_with("](")
            && let Some(url_end) = find_unescaped(text, label_end + 2, ")")
            && label_end > cursor + 1
            && url_end > label_end + 2
        {
            push_fragment(&mut fragments, &text[plain_start..cursor], style);
            append_fragments(
                &mut fragments,
                inline_fragments_at_depth(&text[cursor + 1..label_end], style.fg(CYAN), depth + 1),
            );
            push_fragment(&mut fragments, " (", style.fg(FAINT));
            push_fragment(
                &mut fragments,
                &text[label_end + 2..url_end],
                style.fg(FAINT),
            );
            push_fragment(&mut fragments, ")", style.fg(FAINT));
            cursor = url_end + 1;
            plain_start = cursor;
            continue;
        }

        let mut matched = false;
        for (delimiter, modifier) in [
            ("**", Modifier::BOLD),
            ("__", Modifier::BOLD),
            ("~~", Modifier::CROSSED_OUT),
        ] {
            if text[cursor..].starts_with(delimiter)
                && let Some(end) = find_unescaped(text, cursor + delimiter.len(), delimiter)
                && end > cursor + delimiter.len()
            {
                push_fragment(&mut fragments, &text[plain_start..cursor], style);
                append_fragments(
                    &mut fragments,
                    inline_fragments_at_depth(
                        &text[cursor + delimiter.len()..end],
                        style.add_modifier(modifier),
                        depth + 1,
                    ),
                );
                cursor = end + delimiter.len();
                plain_start = cursor;
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        let delimiter = text[cursor..]
            .starts_with('*')
            .then_some("*")
            .or_else(|| text[cursor..].starts_with('_').then_some("_"));
        if let Some(delimiter) = delimiter
            && let Some(end) = find_unescaped(text, cursor + 1, delimiter)
            && end > cursor + 1
        {
            push_fragment(&mut fragments, &text[plain_start..cursor], style);
            append_fragments(
                &mut fragments,
                inline_fragments_at_depth(
                    &text[cursor + 1..end],
                    style.add_modifier(Modifier::ITALIC),
                    depth + 1,
                ),
            );
            cursor = end + 1;
            plain_start = cursor;
            continue;
        }

        cursor += next_character_len(text, cursor);
    }
    push_fragment(&mut fragments, &text[plain_start..], style);
    fragments
}

fn next_character_len(text: &str, index: usize) -> usize {
    text[index..].chars().next().map_or(1, char::len_utf8)
}

fn is_escaped(text: &str, index: usize) -> bool {
    let mut slashes = 0;
    let mut cursor = index;
    while cursor > 0 && text.as_bytes()[cursor - 1] == b'\\' {
        slashes += 1;
        cursor -= 1;
    }
    slashes % 2 == 1
}

fn find_unescaped(text: &str, start: usize, delimiter: &str) -> Option<usize> {
    let mut search = start;
    while let Some(offset) = text[search..].find(delimiter) {
        let found = search + offset;
        if !is_escaped(text, found) {
            return Some(found);
        }
        search = found + delimiter.len();
    }
    None
}

fn is_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed
        .strip_prefix('|')
        .is_some_and(|rest| rest.contains('|'))
}

fn collect_table(lines: &[&str], start: usize) -> (Vec<Vec<String>>, usize) {
    let mut rows = Vec::new();
    let mut end = start;
    while end < lines.len() && is_table_row(lines[end]) {
        let cells = split_table_row(lines[end]);
        if !is_divider_row(&cells) {
            rows.push(cells);
        }
        end += 1;
    }
    (rows, end)
}

fn split_table_row(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|cell| cell.trim().to_owned())
        .collect()
}

fn is_divider_row(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells.iter().all(|cell| {
            let compact: String = cell
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect();
            let compact = compact.strip_prefix(':').unwrap_or(&compact);
            let compact = compact.strip_suffix(':').unwrap_or(compact);
            compact.len() >= 3 && compact.bytes().all(|byte| byte == b'-')
        })
}

fn table_widths(rows: &[Vec<Vec<Fragment>>], width: usize) -> Vec<usize> {
    let columns = rows.first().map_or(0, Vec::len);
    let mut desired = vec![1; columns];
    for row in rows {
        for (column, cell) in row.iter().enumerate() {
            desired[column] = desired[column].max(fragments_width(cell).max(1));
        }
    }

    let available = width.saturating_sub(columns.saturating_mul(3) + 1);
    let desired_total: usize = desired.iter().sum();
    if desired_total <= available {
        return desired;
    }
    if available < columns {
        return vec![1; columns];
    }

    let extra_budget = available - columns;
    let extra_needed: usize = desired.iter().map(|column| column - 1).sum();
    if extra_needed == 0 {
        return desired;
    }

    let mut widths = vec![1; columns];
    let mut assigned = 0;
    for (index, target) in desired.iter().enumerate() {
        let share = (target - 1) * extra_budget / extra_needed;
        widths[index] += share;
        assigned += share;
    }
    while assigned < extra_budget {
        let Some((index, _)) = desired
            .iter()
            .enumerate()
            .filter(|(index, target)| widths[*index] < **target)
            .max_by_key(|(index, target)| *target - widths[*index])
        else {
            break;
        };
        widths[index] += 1;
        assigned += 1;
    }
    widths
}

fn table_rule(widths: &[usize], left: &str, middle: &str, right: &str) -> Line<'static> {
    let mut rule = String::from(left);
    for (index, width) in widths.iter().enumerate() {
        rule.push_str(&"─".repeat(width + 2));
        if index + 1 < widths.len() {
            rule.push_str(middle);
        }
    }
    rule.push_str(right);
    line_from_fragments(vec![Fragment::new(rule, Style::default().fg(FAINT))])
}

fn append_wrapped(
    output: &mut Vec<Line<'static>>,
    fragments: Vec<Fragment>,
    width: usize,
    first_prefix: Vec<Fragment>,
    continuation_prefix: Vec<Fragment>,
) {
    for (index, row) in wrap_words(&fragments, width.max(1)).into_iter().enumerate() {
        let prefix = if index == 0 {
            first_prefix.clone()
        } else {
            continuation_prefix.clone()
        };
        append_line(output, prefix, row);
    }
}

fn append_line(output: &mut Vec<Line<'static>>, mut prefix: Vec<Fragment>, content: Vec<Fragment>) {
    append_fragments(&mut prefix, content);
    output.push(line_from_fragments(prefix));
}

fn line_from_fragments(fragments: Vec<Fragment>) -> Line<'static> {
    let spans: Vec<Span<'static>> = fragments
        .into_iter()
        .map(|fragment| Span::styled(fragment.text, fragment.style))
        .collect();
    Line::from(spans)
}

fn append_fragments_to(mut first: Vec<Fragment>, second: Vec<Fragment>) -> Vec<Fragment> {
    append_fragments(&mut first, second);
    first
}

fn append_fragments(destination: &mut Vec<Fragment>, source: Vec<Fragment>) {
    for fragment in source {
        push_fragment(destination, &fragment.text, fragment.style);
    }
}

fn push_fragment(destination: &mut Vec<Fragment>, text: &str, style: Style) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = destination.last_mut()
        && last.style == style
    {
        last.text.push_str(text);
        return;
    }
    destination.push(Fragment::new(text, style));
}

fn fragments_width(fragments: &[Fragment]) -> usize {
    fragments
        .iter()
        .map(|fragment| UnicodeWidthStr::width(fragment.text.as_str()))
        .sum()
}

fn tokenize(fragments: &[Fragment]) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    for fragment in fragments {
        for grapheme in fragment.text.graphemes(true) {
            let whitespace = grapheme.chars().all(char::is_whitespace);
            if let Some(token) = tokens.last_mut()
                && token.whitespace == whitespace
            {
                push_fragment(&mut token.fragments, grapheme, fragment.style);
                token.width += UnicodeWidthStr::width(grapheme);
                continue;
            }
            tokens.push(Token {
                fragments: vec![Fragment::new(grapheme, fragment.style)],
                whitespace,
                width: UnicodeWidthStr::width(grapheme),
            });
        }
    }
    tokens
}

fn wrap_words(fragments: &[Fragment], width: usize) -> Vec<Vec<Fragment>> {
    let width = width.max(1);
    if fragments.is_empty() {
        return vec![Vec::new()];
    }

    let mut lines = Vec::new();
    let mut current = Vec::new();
    let mut current_width = 0;
    let mut pending_space = Vec::new();
    let mut pending_width = 0;

    for token in tokenize(fragments) {
        if token.whitespace {
            append_fragments(&mut pending_space, token.fragments);
            pending_width += token.width;
            continue;
        }

        if current.is_empty() {
            if lines.is_empty() && !pending_space.is_empty() && pending_width + token.width <= width
            {
                append_fragments(&mut current, std::mem::take(&mut pending_space));
                current_width += pending_width;
            }
            pending_space.clear();
            pending_width = 0;
            append_hard(
                &mut lines,
                &mut current,
                &mut current_width,
                token.fragments,
                width,
            );
            continue;
        }

        if current_width + pending_width + token.width <= width {
            append_fragments(&mut current, std::mem::take(&mut pending_space));
            current_width += pending_width;
            pending_width = 0;
            append_fragments(&mut current, token.fragments);
            current_width += token.width;
        } else {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
            pending_space.clear();
            pending_width = 0;
            append_hard(
                &mut lines,
                &mut current,
                &mut current_width,
                token.fragments,
                width,
            );
        }
    }

    if !current.is_empty() {
        lines.push(current);
    } else if lines.is_empty() && !pending_space.is_empty() {
        lines.push(pending_space);
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

fn hard_wrap(fragments: &[Fragment], width: usize) -> Vec<Vec<Fragment>> {
    let mut lines = Vec::new();
    let mut current = Vec::new();
    let mut current_width = 0;
    append_hard(
        &mut lines,
        &mut current,
        &mut current_width,
        fragments.to_vec(),
        width.max(1),
    );
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(Vec::new());
    }
    lines
}

fn append_hard(
    lines: &mut Vec<Vec<Fragment>>,
    current: &mut Vec<Fragment>,
    current_width: &mut usize,
    fragments: Vec<Fragment>,
    width: usize,
) {
    for fragment in fragments {
        for grapheme in fragment.text.graphemes(true) {
            let grapheme_width = UnicodeWidthStr::width(grapheme);
            if !current.is_empty()
                && *current_width > 0
                && current_width.saturating_add(grapheme_width) > width
            {
                lines.push(std::mem::take(current));
                *current_width = 0;
            }
            push_fragment(current, grapheme, fragment.style);
            *current_width += grapheme_width;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(text: &Text<'_>) -> String {
        text.lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn keeps_markdown_sections_from_the_go_acceptance_test() {
        let source = r#"# Overview

#include <stdio.h>

- Item 1
  - Nested 1.1
  - Nested 1.2
- Item 2

1. alpha
1. beta
1. gamma

- [ ] beep
- [x] boop

| Name | Age |
|------|-----|
| Alice | 12 |
| Bob | 9 |

```go
func main() {}
```
"#;
        let rendered = plain(&render_markdown(source, 80));

        for expected in [
            "Overview",
            "#include <stdio.h>",
            "- Item 1",
            "    - Nested 1.1",
            "    - Nested 1.2",
            "- Item 2",
            "1. alpha",
            "2. beta",
            "3. gamma",
            "- [ ] beep",
            "- [x] boop",
            "Alice",
            "Bob",
            "func main() {}",
        ] {
            assert!(
                rendered.contains(expected),
                "missing {expected:?} in:\n{rendered}"
            );
        }
        assert!(
            rendered.contains("#include <stdio.h>"),
            "a C include must not become a heading"
        );
    }

    #[test]
    fn keeps_loose_ordered_list_continuations() {
        let source = r#"1. Lorem ipsum dolor sit amet.

   Ut enim ad minim veniam.

2. Duis aute irure dolor.

   Excepteur sint occaecat cupidatat.

3. Beep boop"#;

        assert_eq!(
            plain(&render_markdown(source, 80)),
            "1. Lorem ipsum dolor sit amet.\n\n   Ut enim ad minim veniam.\n\n2. Duis aute irure dolor.\n\n   Excepteur sint occaecat cupidatat.\n\n3. Beep boop"
        );
    }

    #[test]
    fn keeps_ordered_numbering_across_fenced_sections() {
        let source = "1. First item\n\n```typescript\n// code block\n```\n\n2. Second item\n\n```typescript\n// another code block\n```\n\n3. Third item";
        let rendered = plain(&render_markdown(source, 80));
        let numbered: Vec<&str> = rendered
            .lines()
            .map(str::trim)
            .filter(|line| {
                line.as_bytes().get(1) == Some(&b'.')
                    && line.as_bytes().first().is_some_and(u8::is_ascii_digit)
            })
            .collect();

        assert_eq!(
            numbered,
            ["1. First item", "2. Second item", "3. Third item"]
        );
    }

    #[test]
    fn keeps_setext_headings_and_leading_indented_code() {
        let rendered = plain(&render_markdown(
            "Overview\n========\n\n    func main() {}\n",
            80,
        ));
        assert!(rendered.contains("Overview"));
        assert!(!rendered.contains("===="));
        assert!(rendered.contains("func main() {}"));
        assert!(rendered.contains('│'));
    }

    #[test]
    fn wraps_list_items_with_a_hanging_indent() {
        assert_eq!(
            plain(&render_markdown("- alpha beta gamma delta epsilon", 20)),
            "- alpha beta gamma\n  delta epsilon"
        );
    }

    #[test]
    fn keeps_nested_quotes_and_list_quotes() {
        let rendered = plain(&render_markdown("> outer\n>> inner\n\n- > quoted item", 80));
        assert!(
            rendered.contains("│ │ inner") || rendered.contains("│ inner"),
            "nested quote missing: {rendered:?}"
        );
        assert!(
            rendered.contains("- │ quoted item"),
            "list quote missing: {rendered:?}"
        );
    }

    #[test]
    fn preserves_indented_code_after_regular_prose() {
        let rendered = plain(&render_markdown(
            "intro\n    aligned block\n    still aligned",
            40,
        ));
        assert!(rendered.contains("aligned block"));
        assert_eq!(rendered.matches("aligned").count(), 2);
        assert!(rendered.contains('│'));
    }

    #[test]
    fn renders_inline_markup_and_link_destinations() {
        assert_eq!(
            plain(&render_markdown(
                "**bold** and *italic* with `code`, [site](https://example.test), and ~~gone~~",
                120,
            )),
            "bold and italic with code, site (https://example.test), and gone"
        );
    }

    #[test]
    fn fits_wrapped_table_rows_to_the_requested_width() {
        let text = render_markdown(
            "| A long heading | State |\n| --- | --- |\n| text that wraps | active |",
            24,
        );
        assert!(plain(&text).contains("text that"));
        for line in &text.lines {
            let width = line
                .spans
                .iter()
                .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
                .sum::<usize>();
            assert!(width <= 24, "table line exceeds width: {line:?}");
        }
    }

    #[test]
    fn strips_terminal_control_sequences_without_losing_text() {
        let rendered = plain(&render_markdown(
            "\u{1b}[31mvisible\u{1b}[0m\n\u{1b}]8;;https://example.test\u{1b}\\link\u{1b}]8;;\u{1b}\\",
            80,
        ));
        assert_eq!(rendered, "visible\nlink");
        assert!(!rendered.contains('\u{1b}'));
    }
}
