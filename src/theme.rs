//! The terminal interface's colour tokens.
//!
//! These live in one place because the transcript renderer and the Markdown
//! renderer draw the same surface and previously each declared their own copy
//! of the shared half, which could drift apart without anything noticing.
//!
//! Contrast ratios in the comments are against [`BACKGROUND`], computed the
//! WCAG 2.2 way. Anything carrying text is kept at or above the 4.5:1 the
//! standard asks for normal text; a terminal is not a web page, but the
//! measure is the same and the colours are real.

use ratatui::style::Color;

/// The transcript ground.
pub const BACKGROUND: Color = Color::Rgb(10, 10, 10);
/// The sidebar ground.
pub const PANEL_BACKGROUND: Color = Color::Rgb(20, 20, 20);
/// Behind a user message.
pub const USER_BACKGROUND: Color = Color::Rgb(30, 30, 30);
/// Behind a tool card.
pub const TOOL_BACKGROUND: Color = Color::Rgb(24, 24, 24);

/// Headings and the brand mark. 10.66:1.
pub const ACCENT: Color = Color::Rgb(255, 172, 92);
/// Palette borders and the second half of the brand mark. 7.06:1.
pub const VIOLET: Color = Color::Rgb(73, 166, 191);
/// The composer border and tool-success marks. 8.36:1.
pub const CYAN: Color = Color::Rgb(86, 182, 194);
/// Completed work. 11.43:1.
pub const GREEN: Color = Color::Rgb(127, 216, 143);
/// Modified files. 12.58:1.
pub const AMBER: Color = Color::Rgb(242, 201, 108);
/// Errors and deletions. 7.23:1.
pub const RED: Color = Color::Rgb(246, 116, 116);
/// Code spans inside Markdown. 8.28:1.
pub const BLUE: Color = Color::Rgb(112, 174, 221);
/// Body text. 15.58:1.
pub const TEXT: Color = Color::Rgb(220, 230, 232);
/// Secondary text. 5.80:1.
pub const MUTED: Color = Color::Rgb(119, 143, 150);
/// The dimmest text tier: status-bar hints, rules, table borders, and the
/// unfilled part of the context meter. 5.23:1 on the transcript ground and
/// 4.87:1 on the sidebar's, so it clears 4.5:1 on both surfaces it is drawn
/// against. It was #3E5760 (2.58:1), which did not.
pub const FAINT: Color = Color::Rgb(106, 136, 146);

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG 2.2 relative luminance.
    fn luminance(color: Color) -> f64 {
        let Color::Rgb(red, green, blue) = color else {
            panic!("every palette entry is a literal RGB colour");
        };
        let channel = |value: u8| {
            let value = f64::from(value) / 255.0;
            if value <= 0.040_45 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)
    }

    fn contrast(foreground: Color, background: Color) -> f64 {
        let (first, second) = (luminance(foreground), luminance(background));
        (first.max(second) + 0.05) / (first.min(second) + 0.05)
    }

    /// Every foreground the renderer actually pairs with a ground, and the
    /// ground it is drawn on. Combinations that never render are deliberately
    /// absent: forcing FAINT to clear the brightest ground would push it up
    /// against MUTED and collapse the two tiers for no one's benefit.
    const RENDERED_PAIRINGS: &[(&str, Color, &str, Color)] = &[
        ("TEXT", TEXT, "the transcript", BACKGROUND),
        ("MUTED", MUTED, "the transcript", BACKGROUND),
        ("FAINT", FAINT, "the transcript", BACKGROUND),
        ("ACCENT", ACCENT, "the transcript", BACKGROUND),
        ("VIOLET", VIOLET, "the transcript", BACKGROUND),
        ("CYAN", CYAN, "the transcript", BACKGROUND),
        ("GREEN", GREEN, "the transcript", BACKGROUND),
        ("AMBER", AMBER, "the transcript", BACKGROUND),
        ("RED", RED, "the transcript", BACKGROUND),
        ("BLUE", BLUE, "the transcript", BACKGROUND),
        ("TEXT", TEXT, "the sidebar", PANEL_BACKGROUND),
        ("MUTED", MUTED, "the sidebar", PANEL_BACKGROUND),
        ("FAINT", FAINT, "the sidebar", PANEL_BACKGROUND),
        ("ACCENT", ACCENT, "the sidebar", PANEL_BACKGROUND),
        ("GREEN", GREEN, "the sidebar", PANEL_BACKGROUND),
        ("AMBER", AMBER, "the sidebar", PANEL_BACKGROUND),
        ("RED", RED, "the sidebar", PANEL_BACKGROUND),
        ("TEXT", TEXT, "a user message", USER_BACKGROUND),
        ("TEXT", TEXT, "a tool card", TOOL_BACKGROUND),
        ("MUTED", MUTED, "a tool card", TOOL_BACKGROUND),
        ("CYAN", CYAN, "a tool card", TOOL_BACKGROUND),
        ("RED", RED, "a tool card", TOOL_BACKGROUND),
    ];

    /// FAINT was #3E5760, a 2.58:1 ratio that failed even the 3:1 floor for
    /// large text, while carrying the status-bar keyboard hints. This is the
    /// guard that keeps a palette edit from quietly reintroducing that.
    #[test]
    fn every_rendered_colour_pairing_clears_the_contrast_floor() {
        const MINIMUM: f64 = 4.5;
        for (name, color, surface, background) in RENDERED_PAIRINGS {
            let ratio = contrast(*color, *background);
            assert!(
                ratio >= MINIMUM,
                "{name} on {surface} is {ratio:.2}:1, below the {MINIMUM}:1 floor"
            );
        }
    }

    /// The tiers have to stay distinguishable from each other, or the
    /// hierarchy the palette encodes stops being visible.
    #[test]
    fn the_text_tiers_stay_ordered_from_brightest_to_dimmest() {
        assert!(luminance(TEXT) > luminance(MUTED));
        assert!(luminance(MUTED) > luminance(FAINT));
        assert!(luminance(FAINT) > luminance(PANEL_BACKGROUND));
    }
}
