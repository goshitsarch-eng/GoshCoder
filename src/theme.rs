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

use std::sync::OnceLock;

use ratatui::style::Color;

/// One complete set of interface colours.
///
/// Every field is a foreground except the four grounds. Selecting between
/// [`COLOUR`] and [`MONOCHROME`] once, rather than testing a flag at each
/// call site, keeps the renderers free of colour policy.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    /// The transcript ground.
    pub background: Color,
    /// The sidebar ground.
    pub panel_background: Color,
    /// Behind a user message.
    pub user_background: Color,
    /// Behind a tool card.
    pub tool_background: Color,
    /// Headings and the brand mark. 10.66:1.
    pub accent: Color,
    /// Palette borders and the second half of the brand mark. 7.06:1.
    pub violet: Color,
    /// The composer border and tool-success marks. 8.36:1.
    pub cyan: Color,
    /// Completed work. 11.43:1.
    pub green: Color,
    /// Modified files. 12.58:1.
    pub amber: Color,
    /// Errors and deletions. 7.23:1.
    pub red: Color,
    /// Code spans inside Markdown. 8.28:1.
    pub blue: Color,
    /// Body text. 15.58:1.
    pub text: Color,
    /// Secondary text. 5.80:1.
    pub muted: Color,
    /// The dimmest text tier: status-bar hints, rules, table borders, and the
    /// unfilled part of the context meter. 5.23:1 on the transcript ground
    /// and 4.87:1 on the sidebar's, so it clears 4.5:1 on both surfaces it is
    /// drawn against. It was #3E5760 (2.58:1), which did not.
    pub faint: Color,
}

/// The palette used on a colour terminal.
pub const COLOUR: Palette = Palette {
    background: Color::Rgb(10, 10, 10),
    panel_background: Color::Rgb(20, 20, 20),
    user_background: Color::Rgb(30, 30, 30),
    tool_background: Color::Rgb(24, 24, 24),
    accent: Color::Rgb(255, 172, 92),
    violet: Color::Rgb(73, 166, 191),
    cyan: Color::Rgb(86, 182, 194),
    green: Color::Rgb(127, 216, 143),
    amber: Color::Rgb(242, 201, 108),
    red: Color::Rgb(246, 116, 116),
    blue: Color::Rgb(112, 174, 221),
    text: Color::Rgb(220, 230, 232),
    muted: Color::Rgb(119, 143, 150),
    faint: Color::Rgb(106, 136, 146),
};

/// The palette used when colour is suppressed.
///
/// Every entry defers to whatever the terminal is already set to, including
/// the grounds, so the interface adopts the user's own scheme instead of
/// painting over it. The hierarchy that survives is the structural kind the
/// renderers already draw regardless of colour: bold headings, dimmed
/// thinking, the tick and cross on tool cards, and the Error/Notice/Command
/// labels.
pub const MONOCHROME: Palette = Palette {
    background: Color::Reset,
    panel_background: Color::Reset,
    user_background: Color::Reset,
    tool_background: Color::Reset,
    accent: Color::Reset,
    violet: Color::Reset,
    cyan: Color::Reset,
    green: Color::Reset,
    amber: Color::Reset,
    red: Color::Reset,
    blue: Color::Reset,
    text: Color::Reset,
    muted: Color::Reset,
    faint: Color::Reset,
};

/// Whether the user asked for no colour, per <https://no-color.org>.
///
/// This is the single place the rule is decided, so the fullscreen interface
/// and the line-oriented renderers cannot disagree about it.
#[must_use]
pub fn monochrome_requested() -> bool {
    monochrome_from(std::env::var_os("NO_COLOR").as_deref())
}

/// The rule itself, separated from the process environment so it can be
/// checked without mutating global state.
fn monochrome_from(no_color: Option<&std::ffi::OsStr>) -> bool {
    no_color.is_some()
}

/// The palette for this process. Resolved once: an environment variable does
/// not change while the interface is running.
#[must_use]
pub fn theme() -> &'static Palette {
    static ACTIVE: OnceLock<Palette> = OnceLock::new();
    ACTIVE.get_or_init(|| {
        if monochrome_requested() {
            MONOCHROME
        } else {
            COLOUR
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WCAG 2.2 relative luminance.
    fn luminance(color: Color) -> f64 {
        let Color::Rgb(red, green, blue) = color else {
            panic!("every COLOUR entry is a literal RGB colour");
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
        ("text", COLOUR.text, "the transcript", COLOUR.background),
        ("muted", COLOUR.muted, "the transcript", COLOUR.background),
        ("faint", COLOUR.faint, "the transcript", COLOUR.background),
        ("accent", COLOUR.accent, "the transcript", COLOUR.background),
        ("violet", COLOUR.violet, "the transcript", COLOUR.background),
        ("cyan", COLOUR.cyan, "the transcript", COLOUR.background),
        ("green", COLOUR.green, "the transcript", COLOUR.background),
        ("amber", COLOUR.amber, "the transcript", COLOUR.background),
        ("red", COLOUR.red, "the transcript", COLOUR.background),
        ("blue", COLOUR.blue, "the transcript", COLOUR.background),
        ("text", COLOUR.text, "the sidebar", COLOUR.panel_background),
        (
            "muted",
            COLOUR.muted,
            "the sidebar",
            COLOUR.panel_background,
        ),
        (
            "faint",
            COLOUR.faint,
            "the sidebar",
            COLOUR.panel_background,
        ),
        (
            "accent",
            COLOUR.accent,
            "the sidebar",
            COLOUR.panel_background,
        ),
        (
            "green",
            COLOUR.green,
            "the sidebar",
            COLOUR.panel_background,
        ),
        (
            "amber",
            COLOUR.amber,
            "the sidebar",
            COLOUR.panel_background,
        ),
        ("red", COLOUR.red, "the sidebar", COLOUR.panel_background),
        (
            "text",
            COLOUR.text,
            "a user message",
            COLOUR.user_background,
        ),
        ("text", COLOUR.text, "a tool card", COLOUR.tool_background),
        ("muted", COLOUR.muted, "a tool card", COLOUR.tool_background),
        ("cyan", COLOUR.cyan, "a tool card", COLOUR.tool_background),
        ("red", COLOUR.red, "a tool card", COLOUR.tool_background),
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
        assert!(luminance(COLOUR.text) > luminance(COLOUR.muted));
        assert!(luminance(COLOUR.muted) > luminance(COLOUR.faint));
        assert!(luminance(COLOUR.faint) > luminance(COLOUR.panel_background));
    }

    /// Suppressed colour has to actually defer to the terminal -- an entry
    /// left as an RGB literal would paint over the user's scheme.
    #[test]
    fn the_monochrome_palette_defers_to_the_terminal_everywhere() {
        let Palette {
            background,
            panel_background,
            user_background,
            tool_background,
            accent,
            violet,
            cyan,
            green,
            amber,
            red,
            blue,
            text,
            muted,
            faint,
        } = MONOCHROME;
        for entry in [
            background,
            panel_background,
            user_background,
            tool_background,
            accent,
            violet,
            cyan,
            green,
            amber,
            red,
            blue,
            text,
            muted,
            faint,
        ] {
            assert_eq!(entry, Color::Reset);
        }
    }

    /// The line-oriented renderers and the fullscreen interface both route
    /// through this rule, so it is the one place the meaning of NO_COLOR is
    /// decided.
    #[test]
    fn any_setting_of_no_color_suppresses_colour() {
        use std::ffi::OsStr;
        assert!(!monochrome_from(None));
        assert!(monochrome_from(Some(OsStr::new("1"))));
        assert!(monochrome_from(Some(OsStr::new("anything"))));
        // An empty value still counts, matching what the line renderers did
        // before this rule was shared.
        assert!(monochrome_from(Some(OsStr::new(""))));
    }
}
