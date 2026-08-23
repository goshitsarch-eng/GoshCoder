package tui

import (
	"fmt"
	"regexp"
	"strconv"
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
	listItem       = regexp.MustCompile(`^([ \t]*)([-*+]|\d+[.)])(?:[ \t]+(.*))?$`)
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
	hangAt := map[int]int{}
	inQuote := false

	endList := func() {
		for indent := range orderedAt {
			delete(orderedAt, indent)
		}
		for indent := range hangAt {
			delete(hangAt, indent)
		}
	}
	inList := func() bool { return len(hangAt) > 0 }

	for index := 0; index < len(rawLines); index++ {
		raw := rawLines[index]
		trimmed := strings.TrimSpace(raw)
		if marker, ok := parseFence(trimmed); ok {
			if !inCode {
				inCode, fenceMarker = true, marker
				endList()
				inQuote = false
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
			endList()
			inQuote = false
			continue
		}
		if trimmed == "" {
			lines = append(lines, "")
			inQuote = false
			continue
		}
		if isHorizontalRule(trimmed) {
			lines = append(lines, faint+strings.Repeat("─", max(3, width))+reset)
			endList()
			inQuote = false
			continue
		}
		if heading, ok := parseATXHeading(trimmed); ok {
			lines = append(lines, wrapStyled(accent+bold+heading+reset, width)...)
			endList()
			inQuote = false
			continue
		}
		if indent, marker, body, ok := parseListItem(raw); ok {
			inQuote = false
			nest := indent / 2
			pad := strings.Repeat(" ", nest*4)
			display := "-"
			if number, suffix, ordered := orderedNumber(marker); ordered {
				if _, exists := orderedAt[nest]; !exists {
					orderedAt[nest] = number - 1
				}
				orderedAt[nest]++
				for deeper := range orderedAt {
					if deeper > nest {
						delete(orderedAt, deeper)
					}
				}
				for deeper := range hangAt {
					if deeper > nest {
						delete(hangAt, deeper)
					}
				}
				display = fmt.Sprintf("%d%s", orderedAt[nest], suffix)
			} else {
				for deeper := range orderedAt {
					if deeper >= nest {
						delete(orderedAt, deeper)
					}
				}
				for deeper := range hangAt {
					if deeper > nest {
						delete(hangAt, deeper)
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
			hangAt[nest] = hang
			if quote, isQuote := parseQuoteLine(content); isQuote {
				lines = append(lines, wrapQuote(quote, prefix, hang, width)...)
				continue
			}
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
		if inList() && leadingWidth(raw) >= 2 {
			inQuote = false
			hang := deepestHang(hangAt)
			content := strings.TrimSpace(raw)
			if quote, isQuote := parseQuoteLine(content); isQuote {
				lines = append(lines, wrapQuote(quote, strings.Repeat(" ", hang), hang, width)...)
				continue
			}
			wrapped := wrapStyled(styleInline(content, color), max(1, width-hang))
			for _, line := range wrapped {
				lines = append(lines, strings.Repeat(" ", hang)+line)
			}
			continue
		}
		if index+1 < len(rawLines) {
			if level := setextLevel(strings.TrimSpace(rawLines[index+1])); level > 0 {
				heading := strings.TrimSpace(raw)
				if heading != "" {
					lines = append(lines, wrapStyled(accent+bold+heading+reset, width)...)
					index++
					endList()
					inQuote = false
					continue
				}
			}
		}
		endList()
		if depth, quote, ok := parseQuote(trimmed); ok || inQuote {
			if !ok {
				quote = trimmed
				depth = 1
			}
			inQuote = true
			border := strings.Repeat(faint+"│ "+reset, max(1, depth))
			borderWidth := 2 * max(1, depth)
			wrapped := wrapStyled(styleInline(quote, muted), max(1, width-borderWidth))
			for _, line := range wrapped {
				lines = append(lines, border+line)
			}
			continue
		}
		inQuote = false
		if leadingWidth(raw) >= 4 {
			lines = append(lines, wrapCode(stripIndentPrefix(strings.TrimRight(raw, " "), 4), width)...)
			continue
		}
		lines = append(lines, wrapStyled(styleInline(strings.TrimRight(raw, " "), color), width)...)
	}
	return lines
}

func deepestHang(hangAt map[int]int) int {
	nest, hang := -1, 2
	for level, value := range hangAt {
		if level >= nest {
			nest, hang = level, value
		}
	}
	return hang
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

func setextLevel(trimmed string) int {
	if trimmed == "" {
		return 0
	}
	marker := trimmed[0]
	if marker != '=' && marker != '-' {
		return 0
	}
	for i := 0; i < len(trimmed); i++ {
		if trimmed[i] != marker {
			return 0
		}
	}
	if marker == '=' {
		return 1
	}
	if len(trimmed) < 3 {
		// A thematic break needs three dashes; a short underline is still a
		// setext heading so "Section\n--" renders as a heading, not a paragraph.
		return 2
	}
	return 2
}

func parseListItem(raw string) (indent int, marker, body string, ok bool) {
	match := listItem.FindStringSubmatch(raw)
	if match == nil {
		return 0, "", "", false
	}
	body = ""
	if len(match) > 3 {
		body = match[3]
	}
	return leadingWidth(match[1]), match[2], body, true
}

func parseTask(body string) (state, rest string, ok bool) {
	match := taskPrefix.FindStringSubmatch(body)
	if match == nil {
		return "", body, false
	}
	return match[1], match[2], true
}

func orderedNumber(marker string) (number int, suffix string, ok bool) {
	if marker == "" {
		return 0, "", false
	}
	last := marker[len(marker)-1]
	if last != '.' && last != ')' {
		return 0, "", false
	}
	value, err := strconv.Atoi(marker[:len(marker)-1])
	if err != nil {
		return 0, "", false
	}
	return value, string(last), true
}

func parseQuote(trimmed string) (depth int, body string, ok bool) {
	if !strings.HasPrefix(trimmed, ">") {
		return 0, "", false
	}
	rest := trimmed
	for strings.HasPrefix(rest, ">") {
		depth++
		rest = strings.TrimPrefix(rest, ">")
		rest = strings.TrimPrefix(rest, " ")
	}
	return depth, rest, depth > 0
}

func parseQuoteLine(text string) (string, bool) {
	_, body, ok := parseQuote(strings.TrimSpace(text))
	return body, ok
}

func wrapQuote(quote, prefix string, hang, width int) []string {
	border := faint + "│ " + reset
	wrapped := wrapStyled(styleInline(quote, muted), max(1, width-hang-2))
	lines := make([]string, 0, len(wrapped))
	for lineIndex, line := range wrapped {
		lead := prefix
		if lineIndex > 0 {
			lead = strings.Repeat(" ", hang)
		}
		lines = append(lines, lead+border+line)
	}
	if len(lines) == 0 {
		return []string{prefix + border}
	}
	return lines
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

func stripIndentPrefix(raw string, n int) string {
	i := 0
	for i < len(raw) && n > 0 && raw[i] == ' ' {
		i++
		n--
	}
	return raw[i:]
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
			widths[index] = max(widths[index], ansi.StringWidth(value), longestWord(value))
		}
	}
	available := max(8, width-columns*3-1)
	total := 0
	for _, column := range widths {
		total += max(1, column)
	}
	if total > available {
		for index := range widths {
			widths[index] = max(3, max(longestWord(cellAt(rows, index)), widths[index]*available/max(1, total)))
		}
	}
	var lines []string
	lines = append(lines, tableRule(widths, "┌", "┬", "┐"))
	for rowIndex, row := range rows {
		lines = append(lines, tableRow(row, widths))
		if rowIndex < len(rows)-1 {
			lines = append(lines, tableRule(widths, "├", "┼", "┤"))
		}
	}
	lines = append(lines, tableRule(widths, "└", "┴", "┘"))
	return lines
}

func cellAt(rows [][]string, column int) string {
	longest := ""
	for _, row := range rows {
		if column < len(row) && ansi.StringWidth(row[column]) > ansi.StringWidth(longest) {
			longest = row[column]
		}
	}
	return longest
}

func longestWord(text string) int {
	best := 0
	for _, word := range strings.Fields(text) {
		best = max(best, ansi.StringWidth(word))
	}
	return best
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
