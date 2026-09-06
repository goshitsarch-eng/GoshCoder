//! Self-contained HTML export of a session transcript.
//!
//! pi's exporter embeds roughly 165 KB of vendored JavaScript (marked and
//! highlight.js) plus a session-tree explorer in every file. This renderer
//! writes the context path as static HTML with inline CSS and no script at
//! all: the file opens anywhere, cannot load anything from the network, and
//! is small enough to attach to an issue. Markdown is converted by a compact
//! renderer covering what models write (headings, lists, fenced code, tables,
//! quotes, emphasis, links). Every string is HTML-escaped and only `http`,
//! `https`, and `mailto` links become clickable, so a transcript that quotes
//! hostile content stays inert when opened in a browser.

use std::collections::HashMap;

use serde_json::Value;

use crate::{
    llm::{self, ContentBlock},
    sessionlog::{self, Tree},
};

/// The facts shown in the exported page's header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    pub title: String,
    pub id: String,
    pub cwd: String,
    pub started: String,
    /// The exporter's version string, shown in the footer.
    pub generator: String,
}

/// Renders the session's current context path as a complete HTML document.
pub fn render_document(meta: &Meta, tree: &Tree) -> String {
    let entries = tree.context_path(None);
    let messages = decode_messages(&entries);
    let results = collect_tool_results(&messages);
    let mut body = String::new();
    let mut usage_input = 0u64;
    let mut usage_output = 0u64;
    let mut cost = 0.0f64;
    let mut message_count = 0usize;

    for item in &messages {
        match item {
            Item::Compaction {
                tokens_before,
                summary,
            } => {
                body.push_str("<div class=\"divider\">Context compacted");
                if *tokens_before > 0 {
                    body.push_str(&format!(" ({tokens_before} tokens before)"));
                }
                body.push_str("</div>\n");
                if !summary.trim().is_empty() {
                    body.push_str(
                        "<details class=\"summary\"><summary>Summary</summary><div class=\"body\">",
                    );
                    body.push_str(&markdown_to_html(summary));
                    body.push_str("</div></details>\n");
                }
            }
            Item::Message(llm::Message::User(message)) => {
                message_count += 1;
                body.push_str("<section class=\"msg user\"><div class=\"role\">You");
                body.push_str(&timestamp_html(message.timestamp));
                body.push_str("</div><div class=\"body\">");
                body.push_str(&render_user_content(&message.content));
                body.push_str("</div></section>\n");
            }
            Item::Message(llm::Message::Assistant(message)) => {
                message_count += 1;
                usage_input = usage_input.saturating_add(message.usage.input);
                usage_output = usage_output.saturating_add(message.usage.output);
                cost += message.usage.cost.total;
                body.push_str("<section class=\"msg assistant\"><div class=\"role\">Assistant");
                let model = model_label(message);
                if !model.is_empty() {
                    body.push_str(&format!(" <span class=\"model\">{}</span>", escape(&model)));
                }
                body.push_str(&timestamp_html(message.timestamp));
                body.push_str("</div><div class=\"body\">");
                for block in &message.content {
                    match block {
                        ContentBlock::Text(text) => body.push_str(&markdown_to_html(&text.text)),
                        ContentBlock::Thinking(thinking) => {
                            if !thinking.thinking.trim().is_empty() {
                                body.push_str(
                                    "<details class=\"thinking\"><summary>Thinking</summary><div class=\"pre\">",
                                );
                                body.push_str(&escape(thinking.thinking.trim()));
                                body.push_str("</div></details>\n");
                            }
                        }
                        ContentBlock::Image(image) => body.push_str(&image_html(image)),
                        ContentBlock::ToolCall(call) => {
                            body.push_str(&tool_call_html(call, results.get(&call.id).copied()));
                        }
                    }
                }
                if message.stop_reason == "error" || message.stop_reason == "aborted" {
                    let note = if message.error_message.is_empty() {
                        message.stop_reason.clone()
                    } else {
                        message.error_message.clone()
                    };
                    body.push_str(&format!("<p class=\"error\">{}</p>", escape(&note)));
                }
                body.push_str("</div></section>\n");
            }
            Item::Message(llm::Message::ToolResult(result)) => {
                if !results.contains_key(&result.tool_call_id)
                    || results_orphaned(&messages, result)
                {
                    body.push_str(&tool_result_html(result, None));
                }
            }
        }
    }

    let mut header = format!(
        "<h1>{}</h1>\n<dl class=\"facts\">",
        escape(if meta.title.is_empty() {
            &meta.id
        } else {
            &meta.title
        })
    );
    for (name, value) in [
        ("Session", meta.id.as_str()),
        ("Workspace", meta.cwd.as_str()),
        ("Started", meta.started.as_str()),
    ] {
        if !value.is_empty() {
            header.push_str(&format!(
                "<dt>{}</dt><dd>{}</dd>",
                escape(name),
                escape(value)
            ));
        }
    }
    header.push_str(&format!(
        "<dt>Messages</dt><dd>{message_count}</dd><dt>Tokens</dt><dd>{usage_input} in / {usage_output} out</dd>"
    ));
    if cost > 0.0 {
        header.push_str(&format!("<dt>Cost</dt><dd>${cost:.4}</dd>"));
    }
    header.push_str("</dl>\n");

    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<meta name=\"referrer\" content=\"no-referrer\">\n<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; img-src data:; style-src 'unsafe-inline'\">\n<title>{}</title>\n<style>\n{}\n</style>\n</head>\n<body>\n<main>\n{}{}<footer>Exported by {}</footer>\n</main>\n</body>\n</html>\n",
        escape(if meta.title.is_empty() {
            &meta.id
        } else {
            &meta.title
        }),
        STYLE,
        header,
        body,
        escape(if meta.generator.is_empty() {
            "GoshCoder"
        } else {
            &meta.generator
        }),
    )
}

enum Item {
    Message(llm::Message),
    Compaction { tokens_before: u64, summary: String },
}

fn decode_messages(entries: &[&sessionlog::Entry]) -> Vec<Item> {
    let mut items = Vec::new();
    for entry in entries {
        match entry.kind.as_str() {
            sessionlog::TYPE_COMPACTION => items.push(Item::Compaction {
                tokens_before: entry.tokens_before,
                summary: entry.summary.clone(),
            }),
            sessionlog::TYPE_MESSAGE => {
                if let Some(value) = entry.message.as_ref()
                    && let Ok(message) = serde_json::from_value::<llm::Message>(value.clone())
                {
                    items.push(Item::Message(message));
                }
            }
            _ => {}
        }
    }
    items
}

/// Tool results keyed by call id, so each result renders inside its call.
fn collect_tool_results(items: &[Item]) -> HashMap<String, &llm::ToolResultMessage> {
    let mut results = HashMap::new();
    for item in items {
        if let Item::Message(llm::Message::ToolResult(result)) = item {
            results
                .entry(result.tool_call_id.clone())
                .or_insert(result.as_ref());
        }
    }
    results
}

/// A result whose call is absent from the path (a truncated or hand-edited
/// file) still deserves a place in the transcript.
fn results_orphaned(items: &[Item], result: &llm::ToolResultMessage) -> bool {
    !items.iter().any(|item| {
        matches!(item, Item::Message(llm::Message::Assistant(message))
        if message.content.iter().any(|block| {
            matches!(block, ContentBlock::ToolCall(call) if call.id == result.tool_call_id)
        }))
    })
}

fn model_label(message: &llm::AssistantMessage) -> String {
    let model = if message.model.is_empty() {
        message.response_model.as_str()
    } else {
        message.model.as_str()
    };
    match (message.provider.is_empty(), model.is_empty()) {
        (true, true) => String::new(),
        (true, false) => model.to_owned(),
        (false, true) => message.provider.clone(),
        (false, false) => format!("{}/{model}", message.provider),
    }
}

fn timestamp_html(timestamp: i64) -> String {
    if timestamp <= 0 {
        return String::new();
    }
    match time::OffsetDateTime::from_unix_timestamp(timestamp / 1000) {
        Ok(time) => {
            let (year, month, day) = (time.year(), u8::from(time.month()), time.day());
            format!(
                " <time>{year:04}-{month:02}-{day:02} {:02}:{:02} UTC</time>",
                time.hour(),
                time.minute()
            )
        }
        Err(_) => String::new(),
    }
}

fn render_user_content(content: &llm::UserContent) -> String {
    match content {
        llm::UserContent::Text(text) => format!("<div class=\"pre\">{}</div>", escape(text.trim())),
        llm::UserContent::Blocks(blocks) => {
            let mut html = String::new();
            for block in blocks {
                match block {
                    ContentBlock::Text(text) => {
                        html.push_str(&format!(
                            "<div class=\"pre\">{}</div>",
                            escape(text.text.trim())
                        ));
                    }
                    ContentBlock::Image(image) => html.push_str(&image_html(image)),
                    ContentBlock::Thinking(_) | ContentBlock::ToolCall(_) => {}
                }
            }
            html
        }
    }
}

/// Inline images travel as data URIs, which the page's policy allows; a
/// non-image MIME type or non-base64 payload is described instead of embedded.
fn image_html(image: &llm::ImageContent) -> String {
    let mime = image.mime_type.trim().to_ascii_lowercase();
    let safe_mime = mime.starts_with("image/")
        && mime
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'.' | b'+' | b'-'));
    let safe_data = image
        .data
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='));
    if safe_mime && safe_data && !image.data.is_empty() {
        format!(
            "<img src=\"data:{mime};base64,{}\" alt=\"attached image\">",
            image.data
        )
    } else {
        format!(
            "<p class=\"attachment\">[attachment omitted: {}]</p>",
            escape(if mime.is_empty() {
                "unknown type"
            } else {
                &mime
            })
        )
    }
}

fn tool_call_html(call: &llm::ToolCall, result: Option<&llm::ToolResultMessage>) -> String {
    let mut html = String::from("<details class=\"tool");
    if result.is_some_and(|result| result.is_error) {
        html.push_str(" failed");
    }
    html.push_str("\"><summary><code>");
    html.push_str(&escape(&call.name));
    html.push_str("</code>");
    let brief = brief_arguments(&call.arguments);
    if !brief.is_empty() {
        html.push_str(&format!(" <span class=\"args\">{}</span>", escape(&brief)));
    }
    match result {
        Some(result) if result.is_error => html.push_str(" <span class=\"status\">error</span>"),
        Some(_) => {}
        None => html.push_str(" <span class=\"status\">no result</span>"),
    }
    html.push_str("</summary>");
    if !call.arguments.is_empty() {
        let pretty = serde_json::to_string_pretty(&call.arguments).unwrap_or_default();
        html.push_str(&format!(
            "<div class=\"label\">Arguments</div><pre>{}</pre>",
            escape(&pretty)
        ));
    }
    if let Some(result) = result {
        html.push_str(&tool_result_html(result, Some(call)));
    }
    html.push_str("</details>\n");
    html
}

fn tool_result_html(result: &llm::ToolResultMessage, call: Option<&llm::ToolCall>) -> String {
    let mut html = String::new();
    if call.is_none() {
        html.push_str("<details class=\"tool");
        if result.is_error {
            html.push_str(" failed");
        }
        html.push_str(&format!(
            "\"><summary><code>{}</code> result</summary>",
            escape(&result.tool_name)
        ));
    }
    html.push_str(&format!(
        "<div class=\"label\">{}</div>",
        if result.is_error { "Error" } else { "Result" }
    ));
    let mut text = String::new();
    for block in &result.content {
        match block {
            ContentBlock::Text(block) => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&block.text);
            }
            ContentBlock::Image(image) => html.push_str(&image_html(image)),
            ContentBlock::Thinking(_) | ContentBlock::ToolCall(_) => {}
        }
    }
    if !text.trim().is_empty() {
        html.push_str(&format!("<pre>{}</pre>", escape(&truncate_text(&text))));
    } else if result.content.is_empty() {
        html.push_str("<p class=\"attachment\">(empty)</p>");
    }
    if call.is_none() {
        html.push_str("</details>\n");
    }
    html
}

/// Tool output is kept whole up to this size; a page that embeds a multi
/// megabyte log is unreadable and slow, and the session file still has it.
const MAX_RESULT_CHARS: usize = 200_000;

fn truncate_text(text: &str) -> String {
    if text.chars().count() <= MAX_RESULT_CHARS {
        return text.to_owned();
    }
    let kept: String = text.chars().take(MAX_RESULT_CHARS).collect();
    format!("{kept}\n… [truncated in the export]")
}

fn brief_arguments(arguments: &std::collections::BTreeMap<String, Value>) -> String {
    let mut parts = Vec::new();
    for (name, value) in arguments {
        let rendered = match value {
            Value::String(text) => text.lines().next().unwrap_or_default().to_owned(),
            Value::Number(number) => number.to_string(),
            Value::Bool(flag) => flag.to_string(),
            Value::Null => "null".to_owned(),
            Value::Array(_) | Value::Object(_) => "…".to_owned(),
        };
        let rendered: String = rendered.chars().take(60).collect();
        parts.push(format!("{name}={rendered}"));
        if parts.len() == 3 {
            break;
        }
    }
    parts.join(" ")
}

/// Escapes text for an HTML text node or a double-quoted attribute.
pub fn escape(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

/// Whether a link target may be made clickable. Everything else, including
/// `javascript:` and `data:` targets, is shown as text.
fn safe_link(url: &str) -> bool {
    let lowered = url.trim().to_ascii_lowercase();
    lowered.starts_with("http://")
        || lowered.starts_with("https://")
        || lowered.starts_with("mailto:")
        || lowered.starts_with('#')
}

// ---------------------------------------------------------------------------
// Markdown
// ---------------------------------------------------------------------------

/// Converts Markdown to HTML. Unknown constructs degrade to escaped text, so
/// the output is always well-formed and never executes anything.
pub fn markdown_to_html(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut html = String::new();
    render_blocks(&lines, &mut html);
    html
}

fn render_blocks(lines: &[&str], html: &mut String) {
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index];
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            index += 1;
            continue;
        }

        // Fenced code.
        if let Some(fence) = fence_marker(trimmed) {
            let language = trimmed[fence.len()..].trim();
            let mut code = Vec::new();
            index += 1;
            while index < lines.len() && !is_closing_fence(lines[index], fence) {
                code.push(lines[index]);
                index += 1;
            }
            index += 1; // closing fence (or end of input)
            html.push_str("<pre><code");
            let language: String = language
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
                .collect();
            if !language.is_empty() {
                html.push_str(&format!(" class=\"language-{}\"", escape(&language)));
            }
            html.push('>');
            html.push_str(&escape(&code.join("\n")));
            html.push_str("</code></pre>\n");
            continue;
        }

        // Headings.
        if let Some((level, text)) = heading(trimmed) {
            html.push_str(&format!("<h{level}>{}</h{level}>\n", render_inline(text)));
            index += 1;
            continue;
        }

        // Horizontal rule.
        if is_rule(trimmed) {
            html.push_str("<hr>\n");
            index += 1;
            continue;
        }

        // Block quote.
        if trimmed.starts_with('>') {
            let mut quoted = Vec::new();
            while index < lines.len() {
                let candidate = lines[index].trim_start();
                if let Some(rest) = candidate.strip_prefix('>') {
                    quoted.push(rest.strip_prefix(' ').unwrap_or(rest));
                    index += 1;
                } else if !candidate.is_empty() && !quoted.is_empty() && !is_block_start(candidate)
                {
                    // Lazy continuation.
                    quoted.push(candidate);
                    index += 1;
                } else {
                    break;
                }
            }
            html.push_str("<blockquote>\n");
            render_blocks(&quoted, html);
            html.push_str("</blockquote>\n");
            continue;
        }

        // Lists.
        if list_item(line).is_some() {
            index = render_list(lines, index, html);
            continue;
        }

        // Tables.
        if index + 1 < lines.len() && is_table_row(trimmed) && is_table_separator(lines[index + 1])
        {
            index = render_table(lines, index, html);
            continue;
        }

        // Paragraph.
        let mut paragraph = Vec::new();
        while index < lines.len() {
            let candidate = lines[index];
            let candidate_trimmed = candidate.trim_start();
            if candidate_trimmed.is_empty()
                || (!paragraph.is_empty() && is_block_start(candidate_trimmed))
                || (!paragraph.is_empty() && list_item(candidate).is_some())
            {
                break;
            }
            paragraph.push(candidate);
            index += 1;
        }
        html.push_str("<p>");
        for (position, line) in paragraph.iter().enumerate() {
            if position > 0 {
                if paragraph[position - 1].ends_with("  ") {
                    html.push_str("<br>");
                } else {
                    html.push(' ');
                }
            }
            html.push_str(&render_inline(line.trim()));
        }
        html.push_str("</p>\n");
    }
}

fn fence_marker(trimmed: &str) -> Option<&str> {
    for marker in ["```", "~~~"] {
        if trimmed.starts_with(marker) {
            let count = trimmed
                .chars()
                .take_while(|character| *character == marker.chars().next().unwrap_or('`'))
                .count();
            return Some(&trimmed[..count]);
        }
    }
    None
}

fn is_closing_fence(line: &str, fence: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with(fence)
        && trimmed
            .chars()
            .all(|character| character == fence.chars().next().unwrap_or('`'))
}

fn heading(trimmed: &str) -> Option<(usize, &str)> {
    let level = trimmed
        .chars()
        .take_while(|character| *character == '#')
        .count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if rest.is_empty() {
        return Some((level, ""));
    }
    let text = rest.strip_prefix(' ')?;
    Some((level, text.trim().trim_end_matches('#').trim_end()))
}

fn is_rule(trimmed: &str) -> bool {
    let compact: String = trimmed
        .chars()
        .filter(|character| *character != ' ')
        .collect();
    compact.len() >= 3
        && ["-", "*", "_"]
            .iter()
            .any(|marker| compact.chars().all(|c| c.to_string() == *marker))
}

fn is_block_start(trimmed: &str) -> bool {
    fence_marker(trimmed).is_some()
        || heading(trimmed).is_some()
        || is_rule(trimmed)
        || trimmed.starts_with('>')
        || (is_table_row(trimmed) && trimmed.starts_with('|'))
}

/// `(indent, ordered, content)` for a list-item line.
fn list_item(line: &str) -> Option<(usize, bool, &str)> {
    let indent = line.len() - line.trim_start().len();
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        return Some((indent, false, rest));
    }
    if matches!(trimmed, "-" | "*" | "+") {
        return Some((indent, false, ""));
    }
    let digits = trimmed
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .count();
    if digits > 0 && digits <= 9 {
        let rest = &trimmed[digits..];
        if let Some(rest) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some((indent, true, rest));
        }
    }
    None
}

/// Renders one list starting at `start`; returns the index after it.
fn render_list(lines: &[&str], start: usize, html: &mut String) -> usize {
    let (base_indent, ordered, _) = list_item(lines[start]).expect("caller checked a list item");
    html.push_str(if ordered { "<ol>\n" } else { "<ul>\n" });
    let mut index = start;
    while index < lines.len() {
        let Some((indent, item_ordered, content)) = list_item(lines[index]) else {
            break;
        };
        if indent < base_indent || (indent == base_indent && item_ordered != ordered) {
            break;
        }
        if indent > base_indent {
            // A deeper item without its own parent: treat as nested in the last item.
            let inner_end = render_nested(lines, index, html);
            index = inner_end;
            continue;
        }
        // Collect the item's continuation lines (indented deeper than the marker).
        let mut item_lines = vec![content];
        index += 1;
        let mut nested_start = None;
        while index < lines.len() {
            let candidate = lines[index];
            let candidate_trimmed = candidate.trim_start();
            if candidate_trimmed.is_empty() {
                // A blank line ends the item unless a deeper-indented line follows.
                if index + 1 < lines.len() {
                    let next = lines[index + 1];
                    let next_indent = next.len() - next.trim_start().len();
                    if next_indent > base_indent && !next.trim().is_empty() {
                        index += 1;
                        continue;
                    }
                }
                break;
            }
            let candidate_indent = candidate.len() - candidate_trimmed.len();
            if let Some((child_indent, _, _)) = list_item(candidate)
                && child_indent > base_indent
            {
                nested_start = Some(index);
                break;
            }
            if list_item(candidate).is_some() || candidate_indent <= base_indent {
                break;
            }
            item_lines.push(candidate_trimmed);
            index += 1;
        }
        html.push_str("<li>");
        let checkbox = task_checkbox(item_lines[0]);
        if let Some((checked, rest)) = checkbox {
            item_lines[0] = rest;
            html.push_str(if checked {
                "<input type=\"checkbox\" checked disabled> "
            } else {
                "<input type=\"checkbox\" disabled> "
            });
        }
        html.push_str(&render_inline(&item_lines.join(" ")));
        if let Some(nested) = nested_start {
            html.push('\n');
            index = render_list(lines, nested, html);
        }
        html.push_str("</li>\n");
    }
    html.push_str(if ordered { "</ol>\n" } else { "</ul>\n" });
    index
}

fn render_nested(lines: &[&str], start: usize, html: &mut String) -> usize {
    html.push_str("<li>\n");
    let end = render_list(lines, start, html);
    html.push_str("</li>\n");
    end
}

fn task_checkbox(content: &str) -> Option<(bool, &str)> {
    if let Some(rest) = content.strip_prefix("[ ] ") {
        return Some((false, rest));
    }
    if let Some(rest) = content
        .strip_prefix("[x] ")
        .or_else(|| content.strip_prefix("[X] "))
    {
        return Some((true, rest));
    }
    None
}

fn is_table_row(trimmed: &str) -> bool {
    trimmed.contains('|')
}

fn is_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.contains('-') || !trimmed.contains('|') {
        return false;
    }
    trimmed
        .chars()
        .all(|character| matches!(character, '|' | '-' | ':' | ' '))
}

fn split_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let trimmed = trimmed.strip_prefix('|').unwrap_or(trimmed);
    let trimmed = trimmed.strip_suffix('|').unwrap_or(trimmed);
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut escaped = false;
    for character in trimmed.chars() {
        if escaped {
            current.push(character);
            escaped = false;
        } else if character == '\\' {
            escaped = true;
        } else if character == '|' {
            cells.push(current.trim().to_owned());
            current.clear();
        } else {
            current.push(character);
        }
    }
    cells.push(current.trim().to_owned());
    cells
}

fn render_table(lines: &[&str], start: usize, html: &mut String) -> usize {
    let header = split_table_row(lines[start]);
    let alignments: Vec<&str> = split_table_row(lines[start + 1])
        .iter()
        .map(|cell| match (cell.starts_with(':'), cell.ends_with(':')) {
            (true, true) => " style=\"text-align:center\"",
            (false, true) => " style=\"text-align:right\"",
            _ => "",
        })
        .collect();
    html.push_str("<table>\n<thead><tr>");
    for (position, cell) in header.iter().enumerate() {
        html.push_str(&format!(
            "<th{}>{}</th>",
            alignments.get(position).copied().unwrap_or(""),
            render_inline(cell)
        ));
    }
    html.push_str("</tr></thead>\n<tbody>\n");
    let mut index = start + 2;
    while index < lines.len() {
        let line = lines[index];
        if line.trim().is_empty() || !is_table_row(line) {
            break;
        }
        html.push_str("<tr>");
        for (position, cell) in split_table_row(line).iter().enumerate() {
            html.push_str(&format!(
                "<td{}>{}</td>",
                alignments.get(position).copied().unwrap_or(""),
                render_inline(cell)
            ));
        }
        html.push_str("</tr>\n");
        index += 1;
    }
    html.push_str("</tbody>\n</table>\n");
    index
}

/// Renders inline Markdown: code spans, emphasis, strikethrough, links, and
/// bare URLs. Text outside those constructs is escaped verbatim.
pub fn render_inline(text: &str) -> String {
    render_inline_with(text, true)
}

/// `links` disables every link construct, which is how a link label is
/// rendered: a bare URL inside its own label would otherwise recurse forever.
fn render_inline_with(text: &str, links: bool) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut html = String::new();
    let mut index = 0;
    while index < chars.len() {
        let character = chars[index];

        // Code span: a run of backticks closed by an equal run.
        if character == '`' {
            let run = chars[index..]
                .iter()
                .take_while(|character| **character == '`')
                .count();
            if let Some(close) = find_run(&chars, index + run, '`', run) {
                let code: String = chars[index + run..close].iter().collect();
                html.push_str(&format!("<code>{}</code>", escape(code.trim())));
                index = close + run;
                continue;
            }
            html.push_str(&"`".repeat(run));
            index += run;
            continue;
        }

        // Escaped punctuation.
        if character == '\\' && index + 1 < chars.len() && chars[index + 1].is_ascii_punctuation() {
            html.push_str(&escape(&chars[index + 1].to_string()));
            index += 2;
            continue;
        }

        // Image: rendered as a text link, never fetched.
        if links
            && character == '!'
            && chars.get(index + 1) == Some(&'[')
            && let Some((alt, url, end)) = parse_link(&chars, index + 1)
        {
            html.push_str(&link_html(&format!("[image: {alt}]"), &url));
            index = end;
            continue;
        }

        // Link.
        if links
            && character == '['
            && let Some((label, url, end)) = parse_link(&chars, index)
        {
            html.push_str(&link_html(&label, &url));
            index = end;
            continue;
        }

        // Autolink in angle brackets.
        if links
            && character == '<'
            && let Some(close) = chars[index + 1..].iter().position(|c| *c == '>')
        {
            let inner: String = chars[index + 1..index + 1 + close].iter().collect();
            if safe_link(&inner) && !inner.contains(char::is_whitespace) {
                html.push_str(&link_html(&inner, &inner));
                index += close + 2;
                continue;
            }
        }

        // Bare URL.
        if links
            && (character == 'h' || character == 'H')
            && let Some(url_end) = bare_url_end(&chars, index)
        {
            let url: String = chars[index..url_end].iter().collect();
            html.push_str(&link_html(&url, &url));
            index = url_end;
            continue;
        }

        // Emphasis and strikethrough.
        if let Some((tag, marker_len, close)) = emphasis(&chars, index) {
            let inner: String = chars[index + marker_len..close].iter().collect();
            html.push_str(&format!(
                "<{tag}>{}</{tag}>",
                render_inline_with(&inner, links)
            ));
            index = close + marker_len;
            continue;
        }

        html.push_str(&escape(&character.to_string()));
        index += 1;
    }
    html
}

fn find_run(chars: &[char], from: usize, marker: char, run: usize) -> Option<usize> {
    let mut index = from;
    while index < chars.len() {
        if chars[index] == marker {
            let length = chars[index..]
                .iter()
                .take_while(|character| **character == marker)
                .count();
            if length == run {
                return Some(index);
            }
            index += length;
        } else {
            index += 1;
        }
    }
    None
}

/// `[label](url "title")` starting at `start` (which must be `[`).
fn parse_link(chars: &[char], start: usize) -> Option<(String, String, usize)> {
    let mut depth = 0;
    let mut close = None;
    for (offset, character) in chars[start..].iter().enumerate() {
        match character {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    close = Some(start + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let mut end = None;
    let mut nesting = 0;
    for (offset, character) in chars[close + 2..].iter().enumerate() {
        match character {
            '(' => nesting += 1,
            ')' if nesting == 0 => {
                end = Some(close + 2 + offset);
                break;
            }
            ')' => nesting -= 1,
            '\n' => return None,
            _ => {}
        }
    }
    let end = end?;
    let label: String = chars[start + 1..close].iter().collect();
    let target: String = chars[close + 2..end].iter().collect();
    let url = target
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .trim_matches(|character| character == '<' || character == '>')
        .to_owned();
    Some((label, url, end + 1))
}

fn link_html(label: &str, url: &str) -> String {
    if safe_link(url) {
        format!(
            "<a href=\"{}\" rel=\"noopener noreferrer nofollow\">{}</a>",
            escape(url),
            render_inline_plain(label)
        )
    } else if label == url {
        escape(url)
    } else {
        format!("{} ({})", render_inline_plain(label), escape(url))
    }
}

/// Inline rendering for link labels: emphasis and code but no nested links.
fn render_inline_plain(label: &str) -> String {
    render_inline_with(label, false)
}

fn bare_url_end(chars: &[char], start: usize) -> Option<usize> {
    let rest: String = chars[start..].iter().take(8).collect();
    let lowered = rest.to_ascii_lowercase();
    if !lowered.starts_with("http://") && !lowered.starts_with("https://") {
        return None;
    }
    if start > 0 && (chars[start - 1].is_alphanumeric() || chars[start - 1] == '/') {
        return None;
    }
    let mut end = start;
    while end < chars.len()
        && !chars[end].is_whitespace()
        && !matches!(chars[end], '<' | '>' | '"' | '\'' | '`' | ')' | ']')
    {
        end += 1;
    }
    // Trailing punctuation is prose, not part of the URL.
    while end > start && matches!(chars[end - 1], '.' | ',' | ';' | ':' | '!' | '?') {
        end -= 1;
    }
    (end > start + 8).then_some(end)
}

/// `(tag, marker length, closing index)` for emphasis starting at `index`.
fn emphasis(chars: &[char], index: usize) -> Option<(&'static str, usize, usize)> {
    let character = chars[index];
    if !matches!(character, '*' | '_' | '~') {
        return None;
    }
    let run = chars[index..]
        .iter()
        .take_while(|c| **c == character)
        .count();
    let (tag, marker_len) = match (character, run) {
        ('~', 2) => ("del", 2),
        ('*' | '_', 3..) => ("strong", 2),
        ('*' | '_', 2) => ("strong", 2),
        ('*' | '_', 1) => ("em", 1),
        _ => return None,
    };
    // Underscore emphasis must not start inside a word (snake_case).
    if character == '_' && index > 0 && chars[index - 1].is_alphanumeric() {
        return None;
    }
    let content_start = index + marker_len;
    if content_start >= chars.len() || chars[content_start].is_whitespace() {
        return None;
    }
    let close = find_emphasis_close(chars, content_start, character, marker_len)?;
    if chars[close - 1].is_whitespace() {
        return None;
    }
    Some((tag, marker_len, close))
}

fn find_emphasis_close(chars: &[char], from: usize, marker: char, run: usize) -> Option<usize> {
    let mut index = from;
    let mut in_code = false;
    while index < chars.len() {
        let character = chars[index];
        if character == '`' {
            in_code = !in_code;
        } else if !in_code && character == marker {
            let length = chars[index..].iter().take_while(|c| **c == marker).count();
            if length >= run && index > from {
                if marker == '_'
                    && index + length < chars.len()
                    && chars[index + length].is_alphanumeric()
                {
                    index += length;
                    continue;
                }
                return Some(index);
            }
            index += length;
            continue;
        } else if character == '\n' && chars.get(index + 1) == Some(&'\n') {
            return None;
        }
        index += 1;
    }
    None
}

const STYLE: &str = r#":root { color-scheme: light dark; --bg: #f7f7f8; --card: #ffffff; --ink: #1f2328; --muted: #6a737d; --line: #d9dde3; --user: #e8f0fe; --tool: #f3f4f6; --error: #b42318; --code: #f0f1f3; }
@media (prefers-color-scheme: dark) { :root { --bg: #16171b; --card: #1f2126; --ink: #e6e8eb; --muted: #9aa3ad; --line: #33363c; --user: #263048; --tool: #24262b; --error: #ff8a80; --code: #2a2c32; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 60rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
h1 { font-size: 1.5rem; margin: 0 0 0.5rem; overflow-wrap: anywhere; }
.facts { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 1rem; margin: 0 0 2rem; font-size: 0.9rem; color: var(--muted); }
.facts dt { font-weight: 600; } .facts dd { margin: 0; overflow-wrap: anywhere; }
.msg { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 0.9rem 1.1rem; margin: 0 0 1rem; }
.msg.user { background: var(--user); }
.role { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 0.4rem; }
.role .model { text-transform: none; letter-spacing: 0; font-weight: 400; }
.role time { float: right; font-weight: 400; text-transform: none; letter-spacing: 0; }
.body > :first-child { margin-top: 0; } .body > :last-child { margin-bottom: 0; }
.pre { white-space: pre-wrap; overflow-wrap: anywhere; }
pre { background: var(--code); padding: 0.7rem 0.9rem; border-radius: 6px; overflow-x: auto; font: 0.86rem/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
code { background: var(--code); padding: 0.1em 0.35em; border-radius: 4px; font: 0.9em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre code { background: none; padding: 0; font-size: inherit; }
details { border: 1px solid var(--line); border-radius: 6px; padding: 0.35rem 0.7rem; margin: 0.6rem 0; background: var(--tool); }
details.failed { border-color: var(--error); }
details.thinking, details.summary { background: transparent; }
summary { cursor: pointer; color: var(--muted); }
summary .args { color: var(--muted); font-size: 0.9em; overflow-wrap: anywhere; }
summary .status { color: var(--error); font-size: 0.85em; }
.label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin: 0.5rem 0 0.2rem; }
.divider { text-align: center; color: var(--muted); font-size: 0.85rem; margin: 1.5rem 0 0.5rem; }
.error { color: var(--error); }
.attachment { color: var(--muted); font-style: italic; }
img { max-width: 100%; height: auto; border-radius: 6px; margin: 0.4rem 0; }
blockquote { border-left: 3px solid var(--line); margin: 0.6rem 0; padding: 0.1rem 0.9rem; color: var(--muted); }
table { border-collapse: collapse; margin: 0.6rem 0; max-width: 100%; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: 0.3rem 0.6rem; text-align: left; vertical-align: top; }
a { color: inherit; text-decoration: underline; overflow-wrap: anywhere; }
hr { border: 0; border-top: 1px solid var(--line); margin: 1rem 0; }
footer { margin-top: 3rem; color: var(--muted); font-size: 0.8rem; text-align: center; }"#;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn markdown_blocks_and_inline_constructs_render() {
        let html = markdown_to_html(
            "# Title\n\nA **bold** and *em* and `code` line with [a link](https://example.com/x?a=1&b=2).\n\n- one\n- two\n  - nested\n1. first\n2. second\n\n> quoted\n> text\n\n```rust\nfn main() { let x = 1 < 2; }\n```\n\n| a | b |\n|---|--:|\n| 1 | 2 |\n\n---\n\nSee https://example.org/path. Done.",
        );
        assert!(html.contains("<h1>Title</h1>"));
        assert!(html.contains("<strong>bold</strong>"));
        assert!(html.contains("<em>em</em>"));
        assert!(html.contains("<code>code</code>"));
        assert!(html.contains(
            "<a href=\"https://example.com/x?a=1&amp;b=2\" rel=\"noopener noreferrer nofollow\">a link</a>"
        ));
        assert!(
            html.contains(
                "<ul>\n<li>one</li>\n<li>two\n<ul>\n<li>nested</li>\n</ul>\n</li>\n</ul>"
            )
        );
        assert!(html.contains("<ol>\n<li>first</li>\n<li>second</li>\n</ol>"));
        assert!(html.contains("<blockquote>\n<p>quoted text</p>\n</blockquote>"));
        assert!(html.contains(
            "<pre><code class=\"language-rust\">fn main() { let x = 1 &lt; 2; }</code></pre>"
        ));
        assert!(html.contains("<th>a</th><th style=\"text-align:right\">b</th>"));
        assert!(html.contains("<td>1</td><td style=\"text-align:right\">2</td>"));
        assert!(html.contains("<hr>"));
        assert!(html.contains("<a href=\"https://example.org/path\" rel=\"noopener noreferrer nofollow\">https://example.org/path</a>. Done."));
    }

    #[test]
    fn hostile_content_is_escaped_and_unsafe_links_stay_text() {
        let html = markdown_to_html(
            "<script>alert(1)</script> [click](javascript:alert(1)) ![x](data:text/html,oops) <b>bold</b> snake_case_name",
        );
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;alert(1)&lt;/script&gt;"));
        assert!(!html.contains("href=\"javascript:"));
        assert!(html.contains("click (javascript:alert(1))"));
        assert!(html.contains("[image: x] (data:text/html,oops)"));
        assert!(html.contains("&lt;b&gt;bold&lt;/b&gt;"));
        assert!(html.contains("snake_case_name"), "{html}");
        assert!(!html.contains("<em>"));
        assert_eq!(escape("a\"b'c&d<e>"), "a&quot;b&#39;c&amp;d&lt;e&gt;");
    }

    #[test]
    fn task_lists_strikethrough_and_hard_breaks_render() {
        let html = markdown_to_html("- [x] done\n- [ ] todo\n\n~~gone~~ line one  \nline two");
        assert!(html.contains("<input type=\"checkbox\" checked disabled> done"));
        assert!(html.contains("<input type=\"checkbox\" disabled> todo"));
        assert!(html.contains("<del>gone</del>"));
        assert!(html.contains("line one<br>line two"));
    }

    fn tree_with(messages: &[llm::Message]) -> Tree {
        let mut tree = Tree::new();
        let mut parent: Option<String> = None;
        for (index, message) in messages.iter().enumerate() {
            let mut entry = sessionlog::Entry::message(message).expect("entry");
            entry.id = format!("e{index}");
            entry.parent_id = parent.clone();
            tree.add(entry).expect("add entry");
            parent = Some(format!("e{index}"));
        }
        tree
    }

    #[test]
    fn document_renders_messages_tools_and_images_without_script() {
        let mut assistant = llm::AssistantMessage {
            provider: "anthropic".to_owned(),
            model: "claude".to_owned(),
            ..llm::AssistantMessage::default()
        };
        assistant.usage.input = 10;
        assistant.usage.output = 5;
        assistant.usage.cost.total = 0.0123;
        assistant.content = vec![
            ContentBlock::Thinking(llm::ThinkingContent {
                thinking: "let me think".to_owned(),
                ..llm::ThinkingContent::default()
            }),
            ContentBlock::text("Running `ls` now."),
            ContentBlock::ToolCall(llm::ToolCall {
                id: "call-1".to_owned(),
                name: "bash".to_owned(),
                arguments: [("command".to_owned(), json!("ls <dir>"))]
                    .into_iter()
                    .collect(),
                ..llm::ToolCall::default()
            }),
        ];
        let result = llm::ToolResultMessage {
            tool_call_id: "call-1".to_owned(),
            tool_name: "bash".to_owned(),
            content: vec![
                ContentBlock::text("a.txt\nb.txt"),
                ContentBlock::Image(llm::ImageContent {
                    data: "aGVsbG8=".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ],
            is_error: true,
            ..llm::ToolResultMessage::default()
        };
        let tree = tree_with(&[
            llm::Message::User(llm::UserMessage::text("hello <world>", 1_700_000_000_000)),
            llm::Message::Assistant(Box::new(assistant)),
            llm::Message::ToolResult(Box::new(result)),
        ]);
        let html = render_document(
            &Meta {
                title: "My <session>".to_owned(),
                id: "abc".to_owned(),
                cwd: "/work".to_owned(),
                started: "2026-01-01".to_owned(),
                generator: "GoshCoder 0.1".to_owned(),
            },
            &tree,
        );
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(!html.contains("<script"));
        assert!(html.contains("<title>My &lt;session&gt;</title>"));
        assert!(html.contains("hello &lt;world&gt;"));
        assert!(html.contains("<span class=\"model\">anthropic/claude</span>"));
        assert!(html.contains("<time>2023-11-14 22:13 UTC</time>"));
        assert!(html.contains("<details class=\"thinking\"><summary>Thinking</summary><div class=\"pre\">let me think</div></details>"));
        assert!(html.contains("<details class=\"tool failed\"><summary><code>bash</code> <span class=\"args\">command=ls &lt;dir&gt;</span> <span class=\"status\">error</span></summary>"));
        assert!(html.contains("<pre>a.txt\nb.txt</pre>"));
        assert!(
            html.contains("<img src=\"data:image/png;base64,aGVsbG8=\" alt=\"attached image\">")
        );
        assert!(
            html.contains("<dt>Tokens</dt><dd>10 in / 5 out</dd><dt>Cost</dt><dd>$0.0123</dd>")
        );
        assert!(html.contains("<footer>Exported by GoshCoder 0.1</footer>"));
        assert_eq!(
            html.matches("<details class=\"tool").count(),
            1,
            "the result renders inside its call only"
        );
    }

    #[test]
    fn unsafe_images_and_orphan_results_are_described() {
        let tree = tree_with(&[llm::Message::ToolResult(Box::new(llm::ToolResultMessage {
            tool_call_id: "missing".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ContentBlock::Image(llm::ImageContent {
                data: "<svg onload=alert(1)>".to_owned(),
                mime_type: "text/html".to_owned(),
            })],
            ..llm::ToolResultMessage::default()
        }))]);
        let html = render_document(&Meta::default(), &tree);
        assert!(
            html.contains("<details class=\"tool\"><summary><code>read</code> result</summary>")
        );
        assert!(html.contains("[attachment omitted: text/html]"));
        assert!(!html.contains("onload"));
    }
}
