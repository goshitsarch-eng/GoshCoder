package tui

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/charmbracelet/x/ansi"
)

// Pi's markdown renderer (packages/tui/src/components/markdown.ts) is the
// visual source of truth for assistant output. GoshCoder keeps a streaming-
// friendly line parser instead of pulling in marked, but it follows the same
// CommonMark/GFM shapes so headings, nested lists, tables, and fenced sections
// actually display as sections instead of flattened paragraphs.

var (
	atxHeading     = regexp.MustCompile(`^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$`)
	listItem       = regexp.MustCompile(`^([ \t]*)([-*+]|\d+[.)])[ \t]+(.*)$`)
	taskPrefix     = regexp.MustCompile(`^\[([ xX])\][ \t]+(.*)$`)
	horizontalRule = regexp.MustCompile(`^((?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$`)
	tableDivider   = regexp.MustCompile(`^\s*:?-{3,}:?\s*$`)
	inlineToken    = regexp.MustCompile("(?s)`[^`]+`|\\*\\*[^*]+\\*\\*|__[^_]+__|~~[^~]+~~|\\[[^\\]]+\\]\\([^)]+\\)|\\*[^*]+\\*|_[^_]+_")
)

func renderRichText(source string, width int, role string) []string {
	source = safeTerminalText(strings.ReplaceAll(source, "\t", "    "))
	if width < 1 {
		width = 1
	}
	color := textColor
	if role == "thinking" {
		color = muted
	}
	rawLines := strings.Split(source, "\n")
	var lines []string
	inCode := false
	fenceMarker := ""
	orderedAt := map[int]int{}

	for index := 0; index < len(rawLines); index++ {
		raw := rawLines[index]
		trimmed := strings.TrimSpace(raw)
		if marker, ok := parseFence(trimmed); ok {
			if !inCode {
				inCode, fenceMarker = true, marker
				continue
			}
			if strings.HasPrefix(trimmed, fenceMarker) {
				inCode, fenceMarker = false, ""
				continue
			}
		}
		if inCode {
			lines = append(lines, wrapCode(raw, width)...)
			continue
		}
		if isTableRow(raw) {
			table, consumed := collectTable(rawLines, index)
			index = consumed
			lines = append(lines, renderTable(table, width)...)
			for indent := range orderedAt {
				delete(orderedAt, indent)
			}
			continue
		}
		if trimmed == "" {
			lines = append(lines, "")
			for indent := range orderedAt {
				delete(orderedAt, indent)
			}
			continue
		}
		if isHorizontalRule(trimmed) {
			lines = append(lines, faint+strings.Repeat("─", max(3, width))+reset)
			continue
		}
		if heading, ok := parseATXHeading(trimmed); ok {
			lines = append(lines, wrapStyled(accent+bold+heading+reset, width)...)
			for indent := range orderedAt {
				delete(orderedAt, indent)
			}
			continue
		}
		if indent, marker, body, ok := parseListItem(raw); ok {
			nest := indent / 2
			pad := strings.Repeat(" ", nest*4)
			display := marker
			if number, ordered := orderedMarker(marker); ordered {
				orderedAt[nest]++
				for deeper := range orderedAt {
					if deeper > nest {
						delete(orderedAt, deeper)
					}
				}
				display = fmt.Sprintf("%d%s", orderedAt[nest], numberSuffix(number))
			} else {
				display = "-"
				for deeper := range orderedAt {
					if deeper >= nest {
						delete(orderedAt, deeper)
					}
				}
			}
			task, rest, isTask := parseTask(body)
			content := rest
			if !isTask {
				content = body
			}
			prefix := pad + cyan + display + reset + " "
			if isTask {
				box := "[ ]"
				if task != " " {
					box = "[x]"
				}
				prefix = pad + cyan + display + " " + box + reset + " "
			}
			hang := ansi.StringWidth(stripANSI(prefix))
			wrapped := wrapStyled(styleInline(content, color), max(1, width-hang))
			for lineIndex, line := range wrapped {
				if lineIndex == 0 {
					lines = append(lines, prefix+line)
				} else {
					lines = append(lines, strings.Repeat(" ", hang)+line)
				}
			}
			continue
		}
		for indent := range orderedAt {
			delete(orderedAt, indent)
		}
		if strings.HasPrefix(trimmed, "> ") || trimmed == ">" {
			quote := strings.TrimSpace(strings.TrimPrefix(trimmed, ">"))
			wrapped := wrapStyled(styleInline(quote, muted), max(1, width-2))
			for _, line := range wrapped {
				lines = append(lines, faint+"│ "+reset+line)
			}
			continue
		}
		if leadingWidth(raw) >= 4 {
			lines = append(lines, wrapCode(strings.TrimRight(raw, " "), width)...)
			continue
		}
		lines = append(lines, wrapStyled(styleInline(strings.TrimRight(raw, " "), color), width)...)
	}
	return lines
}

func isHorizontalRule(trimmed string) bool {
	compact := strings.ReplaceAll(strings.ReplaceAll(trimmed, " ", ""), "\t", "")
	return len(compact) >= 3 && horizontalRule.MatchString(compact)
}

func parseFence(trimmed string) (string, bool) {
	if len(trimmed) < 3 {
		return "", false
	}
	marker := trimmed[0]
	if marker != '`' && marker != '~' {
		return "", false
	}
	count := 0
	for count < len(trimmed) && trimmed[count] == marker {
		count++
	}
	if count < 3 {
		return "", false
	}
	return strings.Repeat(string(marker), count), true
}

func parseATXHeading(trimmed string) (string, bool) {
	match := atxHeading.FindStringSubmatch(trimmed)
	if match == nil {
		return "", false
	}
	return strings.TrimSpace(match[2]), true
}

func parseListItem(raw string) (indent int, marker, body string, ok bool) {
	match := listItem.FindStringSubmatch(raw)
	if match == nil {
		return 0, "", "", false
	}
	return leadingWidth(match[1]), match[2], match[3], true
}

func parseTask(body string) (state, rest string, ok bool) {
	match := taskPrefix.FindStringSubmatch(body)
	if match == nil {
		return "", body, false
	}
	return match[1], match[2], true
}

func orderedMarker(marker string) (string, bool) {
	if marker == "" {
		return "", false
	}
	last := marker[len(marker)-1]
	if last != '.' && last != ')' {
		return "", false
	}
	for _, r := range marker[:len(marker)-1] {
		if r < '0' || r > '9' {
			return "", false
		}
	}
	return marker, true
}

func numberSuffix(marker string) string {
	if strings.HasSuffix(marker, ")") {
		return ")"
	}
	return "."
}

func leadingWidth(text string) int {
	count := 0
	for _, r := range text {
		if r == ' ' {
			count++
			continue
		}
		if r == '\t' {
			count += 4
			continue
		}
		break
	}
	return count
}

func wrapCode(raw string, width int) []string {
	inner := max(1, width-2)
	styled := blue + raw + reset
	wrapped := ansi.Hardwrap(styled, inner, true)
	if wrapped == "" {
		return []string{faint + "│ " + reset}
	}
	var lines []string
	for _, line := range strings.Split(wrapped, "\n") {
		lines = append(lines, faint+"│ "+reset+line)
	}
	return lines
}

func wrapStyled(text string, width int) []string {
	if width < 1 {
		width = 1
	}
	if strings.TrimSpace(stripANSI(text)) == "" {
		return []string{text}
	}
	wrapped := ansi.Wrap(text, width, "")
	if wrapped == "" {
		return []string{""}
	}
	return strings.Split(wrapped, "\n")
}

func styleInline(text, color string) string {
	if text == "" {
		return color + reset
	}
	var builder strings.Builder
	last := 0
	for _, loc := range inlineToken.FindAllStringIndex(text, -1) {
		builder.WriteString(color)
		builder.WriteString(text[last:loc[0]])
		builder.WriteString(reset)
		token := text[loc[0]:loc[1]]
		switch {
		case strings.HasPrefix(token, "`"):
			builder.WriteString(blue)
			builder.WriteString(token[1 : len(token)-1])
			builder.WriteString(reset)
		case strings.HasPrefix(token, "**") || strings.HasPrefix(token, "__"):
			builder.WriteString(color)
			builder.WriteString(bold)
			builder.WriteString(token[2 : len(token)-2])
			builder.WriteString(reset)
		case strings.HasPrefix(token, "~~"):
			builder.WriteString(color)
			builder.WriteString("\x1b[9m")
			builder.WriteString(token[2 : len(token)-2])
			builder.WriteString("\x1b[29m")
			builder.WriteString(reset)
		case strings.HasPrefix(token, "["):
			split := strings.Index(token, "](")
			label := token[1:split]
			href := token[split+2 : len(token)-1]
			builder.WriteString(cyan)
			builder.WriteString(label)
			builder.WriteString(reset)
			builder.WriteString(faint)
			builder.WriteString(" (")
			builder.WriteString(href)
			builder.WriteString(")")
			builder.WriteString(reset)
		default:
			inner := token[1 : len(token)-1]
			builder.WriteString(color)
			builder.WriteString("\x1b[3m")
			builder.WriteString(inner)
			builder.WriteString("\x1b[23m")
			builder.WriteString(reset)
		}
		last = loc[1]
	}
	builder.WriteString(color)
	builder.WriteString(text[last:])
	builder.WriteString(reset)
	return builder.String()
}

func isTableRow(line string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, "|") && strings.Contains(trimmed[1:], "|")
}

func collectTable(lines []string, start int) ([][]string, int) {
	var rows [][]string
	end := start
	for end < len(lines) && isTableRow(lines[end]) {
		cells := splitTableRow(lines[end])
		if isDividerRow(cells) {
			end++
			continue
		}
		rows = append(rows, cells)
		end++
	}
	if end == start {
		return nil, start
	}
	return rows, end - 1
}

func splitTableRow(line string) []string {
	trimmed := strings.TrimSpace(line)
	trimmed = strings.Trim(trimmed, "|")
	parts := strings.Split(trimmed, "|")
	cells := make([]string, len(parts))
	for index, part := range parts {
		cells[index] = strings.TrimSpace(part)
	}
	return cells
}

func isDividerRow(cells []string) bool {
	if len(cells) == 0 {
		return false
	}
	for _, cell := range cells {
		if !tableDivider.MatchString(strings.ReplaceAll(cell, " ", "")) && !tableDivider.MatchString(cell) {
			return false
		}
	}
	return true
}

func renderTable(rows [][]string, width int) []string {
	if len(rows) == 0 {
		return nil
	}
	columns := 0
	for _, row := range rows {
		columns = max(columns, len(row))
	}
	widths := make([]int, columns)
	for _, row := range rows {
		for index := 0; index < columns; index++ {
			value := ""
			if index < len(row) {
				value = row[index]
			}
			widths[index] = max(widths[index], ansi.StringWidth(value))
		}
	}
	available := max(8, width-columns*3-1)
	total := 0
	for _, column := range widths {
		total += max(1, column)
	}
	if total > available {
		for index := range widths {
			widths[index] = max(3, widths[index]*available/max(1, total))
		}
	}
	var lines []string
	lines = append(lines, tableRule(widths, "┌", "┬", "┐"))
	for rowIndex, row := range rows {
		lines = append(lines, tableRow(row, widths))
		if rowIndex == 0 && len(rows) > 1 {
			lines = append(lines, tableRule(widths, "├", "┼", "┤"))
		}
	}
	lines = append(lines, tableRule(widths, "└", "┴", "┘"))
	return lines
}

func tableRule(widths []int, left, mid, right string) string {
	var builder strings.Builder
	builder.WriteString(faint)
	builder.WriteString(left)
	for index, width := range widths {
		builder.WriteString(strings.Repeat("─", width+2))
		if index+1 < len(widths) {
			builder.WriteString(mid)
		}
	}
	builder.WriteString(right)
	builder.WriteString(reset)
	return builder.String()
}

func tableRow(row []string, widths []int) string {
	var builder strings.Builder
	builder.WriteString(faint)
	builder.WriteString("│")
	builder.WriteString(reset)
	for index, width := range widths {
		value := ""
		if index < len(row) {
			value = row[index]
		}
		cell := pad(styleInline(value, textColor), width)
		builder.WriteString(" ")
		builder.WriteString(cell)
		builder.WriteString(" ")
		builder.WriteString(faint)
		builder.WriteString("│")
		builder.WriteString(reset)
	}
	return builder.String()
}
