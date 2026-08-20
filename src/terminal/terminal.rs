use std::{
    fmt,
    io::{self, IsTerminal as _, Write as _},
};

use time::{OffsetDateTime, format_description::well_known::Iso8601};

/// The visual treatment for one diagnostic segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextStyle {
    /// Use dim diagnostic text.
    Neutral,
    /// Use bold, dim diagnostic text.
    BoldNeutral,
    /// Use dim green text.
    Green,
    /// Use dim red text.
    Red,
    /// Use orange text.
    Orange,
    /// Use red error text.
    Error,
}

/// A styled text fragment or a spinner frame within a diagnostic line.
pub enum Segment<'a> {
    Text {
        style: TextStyle,
        content: fmt::Arguments<'a>,
    },
    Spinner {
        style: TextStyle,
        frame: &'a str,
    },
}

impl<'a> Segment<'a> {
    #[must_use]
    pub const fn text(style: TextStyle, content: fmt::Arguments<'a>) -> Self {
        Self::Text { style, content }
    }

    #[must_use]
    pub const fn spinner(style: TextStyle, frame: &'a str) -> Self {
        Self::Spinner { style, frame }
    }
}

/// Styled terminal output reserved for diagnostics on stderr.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // These flags represent independent terminal capabilities.
pub struct Terminal {
    color: bool,
    interactive: bool,
    progress: bool,
    timestamp: bool,
}

impl Terminal {
    const CLEAR: &str = "\x1b[0m";
    // Include foreground 39 explicitly. While SGR 0 nominally restores it, spelling out the
    // default foreground makes neutral segments independent of an immediately adjacent color.
    const NEUTRAL: &str = "\x1b[2;39m";
    const BOLD_NEUTRAL: &str = "\x1b[1;2;39m";
    const GREEN: &str = "\x1b[2;32m";
    const RED: &str = "\x1b[2;31m";
    const ORANGE: &str = "\x1b[38;5;208m";
    const ERROR: &str = "\x1b[31m";

    /// Create terminal output, optionally decorated with ANSI colors and timestamps.
    #[must_use]
    pub fn new(color: bool, timestamp: bool) -> Self {
        Self {
            color,
            interactive: io::stderr().is_terminal(),
            progress: true,
            timestamp,
        }
    }

    /// Enable or disable diagnostic progress output while retaining error output.
    #[must_use]
    pub const fn with_progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    /// Write a dim diagnostic line to stderr.
    pub fn status(self, message: fmt::Arguments<'_>) {
        if !self.progress {
            return;
        }
        self.status_segments([Segment::text(TextStyle::Neutral, message)]);
    }

    /// Write a dim diagnostic line with individually styled fragments.
    pub fn status_segments<'a>(self, segments: impl IntoIterator<Item = Segment<'a>>) {
        if !self.progress {
            return;
        }
        self.write_line(false, false, segments);
    }

    /// Write an orange warning line to stderr.
    pub fn warning(self, message: fmt::Arguments<'_>) {
        if !self.progress {
            return;
        }
        self.write_line(false, false, [Segment::text(TextStyle::Orange, message)]);
    }

    /// Redraw a dim spinner line. A [`Segment::Spinner`] may appear anywhere in `segments`.
    pub fn spinner<'a>(self, first: bool, segments: impl IntoIterator<Item = Segment<'a>>) {
        if self.progress && self.should_write_spinner(first) {
            self.write_line(self.interactive, first, segments);
        }
    }

    /// Replace an active spinner with a styled completion line.
    pub fn complete<'a>(self, segments: impl IntoIterator<Item = Segment<'a>>) {
        if !self.progress {
            return;
        }
        self.write_line(self.interactive, false, segments);
    }

    /// Write an error and its context chain in red to stderr.
    pub fn error(self, error: &anyhow::Error) {
        self.write_line(
            false,
            false,
            [Segment::text(TextStyle::Error, format_args!("{error:?}"))],
        );
    }

    /// Render a fixed-width, proportional progress bar.
    #[must_use]
    pub fn progress_bar(value: usize, total: usize, width: usize) -> String {
        if width == 0 {
            return String::new();
        }

        let filled = value
            .saturating_mul(width)
            .checked_div(total)
            .unwrap_or(0)
            .min(width);
        format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
    }

    fn write_line<'a>(
        self,
        replace_previous: bool,
        blank_before: bool,
        segments: impl IntoIterator<Item = Segment<'a>>,
    ) {
        let mut stderr = io::stderr().lock();
        if blank_before {
            let _ = writeln!(stderr);
        }
        if replace_previous {
            let _ = write!(stderr, "\x1b[A\r\x1b[K");
        }
        self.write_timestamp(&mut stderr);
        self.write_segments(&mut stderr, segments);
        let _ = writeln!(stderr);
        let _ = stderr.flush();
    }

    fn write_segments<'a, W: io::Write>(
        self,
        output: &mut W,
        segments: impl IntoIterator<Item = Segment<'a>>,
    ) {
        for segment in segments {
            match segment {
                Segment::Text { style, content } => {
                    self.apply_style(output, style);
                    let _ = output.write_fmt(content);
                    self.reset_style(output);
                }
                Segment::Spinner { style, frame } => {
                    self.apply_style(output, style);
                    let _ = write!(output, "{frame}");
                    self.reset_style(output);
                }
            }
        }
    }

    const fn should_write_spinner(self, first: bool) -> bool {
        self.interactive || first
    }

    fn apply_style<W: io::Write>(self, output: &mut W, style: TextStyle) {
        if self.color {
            let _ = write!(output, "{}{}", Self::CLEAR, Self::style_code(style));
        }
    }

    fn reset_style<W: io::Write>(self, output: &mut W) {
        if self.color {
            let _ = write!(output, "{}", Self::CLEAR);
        }
    }

    fn style_code(style: TextStyle) -> &'static str {
        match style {
            TextStyle::Neutral => Self::NEUTRAL,
            TextStyle::BoldNeutral => Self::BOLD_NEUTRAL,
            TextStyle::Green => Self::GREEN,
            TextStyle::Red => Self::RED,
            TextStyle::Orange => Self::ORANGE,
            TextStyle::Error => Self::ERROR,
        }
    }

    fn write_timestamp<W: io::Write>(self, output: &mut W) {
        if self.timestamp {
            self.apply_style(output, TextStyle::Neutral);
            let _ = write!(output, "[{}] ", timestamp());
            self.reset_style(output);
        }
    }
}

fn timestamp() -> String {
    OffsetDateTime::now_local()
        .unwrap_or_else(|_| OffsetDateTime::now_utc())
        .format(&Iso8601::TIME)
        .map_or_else(|_| "??:??:??".to_owned(), |time| time[..8].to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_segments_each_reset_and_set_their_complete_style() {
        let terminal = Terminal::new(true, false);
        let mut output = Vec::new();

        terminal.write_segments(
            &mut output,
            [
                Segment::text(TextStyle::Green, format_args!("+1")),
                Segment::text(TextStyle::Neutral, format_args!("|")),
                Segment::text(TextStyle::Red, format_args!("-1")),
            ],
        );

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                "\x1b[0m\x1b[2;32m+1\x1b[0m",
                "\x1b[0m\x1b[2;39m|\x1b[0m",
                "\x1b[0m\x1b[2;31m-1\x1b[0m",
            )
        );
    }

    #[test]
    fn non_interactive_output_writes_each_spinner_phase_once() {
        let terminal = Terminal {
            color: false,
            interactive: false,
            progress: true,
            timestamp: false,
        };

        assert!(terminal.should_write_spinner(true));
        assert!(!terminal.should_write_spinner(false));
    }
}
