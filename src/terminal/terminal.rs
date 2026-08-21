use std::{
    fmt,
    io::{self, IsTerminal as _, Write as _},
};

use time::{OffsetDateTime, format_description::well_known::Iso8601};

/// The visual treatment for one diagnostic segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextStyle {
    /// Use faint dark-gray text for timestamps.
    Timestamp,
    /// Use dim diagnostic text.
    Neutral,
    /// Use bold, dim diagnostic text.
    BoldNeutral,
    /// Use green text.
    Green,
    /// Use red text.
    Red,
    /// Use dim red text.
    DimRed,
    /// Use dim slate-blue text for model reasoning.
    Thinking,
    /// Use orange text.
    Orange,
    /// Use red error text.
    Error,
}

/// A styled text fragment or a spinner frame within a diagnostic line.
pub enum Segment<'a> {
    /// Styled formatted text.
    Text {
        /// Text style.
        style: TextStyle,
        /// Deferred formatted content.
        content: fmt::Arguments<'a>,
    },
    /// Styled spinner frame.
    Spinner {
        /// Frame style.
        style: TextStyle,
        /// Frame text.
        frame: &'a str,
    },
}

impl<'a> Segment<'a> {
    /// Construct a styled text segment.
    #[must_use]
    pub const fn text(style: TextStyle, content: fmt::Arguments<'a>) -> Self {
        Self::Text { style, content }
    }

    /// Construct a styled spinner segment.
    #[must_use]
    pub const fn spinner(style: TextStyle, frame: &'a str) -> Self {
        Self::Spinner { style, frame }
    }
}

/// Styled terminal output reserved for diagnostics on stderr.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // These flags represent independent terminal capabilities.
pub struct Terminal {
    /// Whether ANSI colors are enabled.
    color: bool,
    /// Whether stderr is interactive.
    interactive: bool,
    /// Whether progress output is enabled.
    progress: bool,
    /// Whether timestamps are shown.
    timestamp: bool,
}

impl Terminal {
    /// ANSI reset sequence.
    const CLEAR: &str = "\x1b[0m";
    /// ANSI faint dark-gray sequence for timestamps.
    const TIMESTAMP: &str = "\x1b[2;38;5;240m";
    // Include foreground 39 explicitly. While SGR 0 nominally restores it, spelling out the
    // default foreground makes neutral segments independent of an immediately adjacent color.
    const NEUTRAL: &str = "\x1b[2;39m";
    /// ANSI bold-neutral sequence.
    const BOLD_NEUTRAL: &str = "\x1b[1;2;39m";
    /// ANSI green sequence.
    const GREEN: &str = "\x1b[32m";
    /// ANSI red sequence.
    const RED: &str = "\x1b[31m";
    /// ANSI dim-red sequence.
    const DIM_RED: &str = "\x1b[2;31m";
    /// ANSI thinking sequence.
    const THINKING: &str = "\x1b[2;38;5;103m";
    /// ANSI orange sequence.
    const ORANGE: &str = "\x1b[38;5;208m";
    /// ANSI error sequence.
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

    /// Redraw a spinner above the latest streamed reasoning lines.
    ///
    /// Returns the number of visible reasoning lines to pass to the next call.
    #[must_use]
    pub fn thinking<'a>(
        self,
        first: bool,
        previous_lines: usize,
        spinner: impl IntoIterator<Item = Segment<'a>>,
        lines: &[String],
    ) -> usize {
        if !self.progress {
            return 0;
        }

        if !self.interactive {
            if first {
                self.write_line(false, false, spinner);
            }
            return 0;
        }

        let mut stderr = io::stderr().lock();
        if !first {
            let _ = write!(stderr, "\x1b[{}A", previous_lines.saturating_add(1));
        }
        let _ = write!(stderr, "\r\x1b[K");
        self.write_rendered_line(&mut stderr, spinner);
        self.write_thinking_lines(&mut stderr, previous_lines, lines);
        let _ = stderr.flush();
        lines.len()
    }

    /// Remove a thinking spinner, preserve its latest reasoning lines, and append a completion.
    pub fn finish_thinking<'a>(
        self,
        previous_lines: usize,
        lines: &[String],
        completion: impl IntoIterator<Item = Segment<'a>>,
    ) {
        if !self.progress {
            return;
        }

        if !self.interactive {
            self.complete(completion);
            return;
        }

        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\x1b[{}A", previous_lines.saturating_add(1));
        self.write_thinking_lines(&mut stderr, previous_lines.saturating_add(1), lines);
        self.write_rendered_line(&mut stderr, completion);
        let _ = stderr.flush();
    }

    /// Render the current thinking preview and clear stale rows.
    fn write_thinking_lines<W: io::Write>(
        self,
        output: &mut W,
        previous_lines: usize,
        lines: &[String],
    ) {
        let rows = previous_lines.max(lines.len());
        for line_index in 0..rows {
            let _ = write!(output, "\r\x1b[K");
            if let Some(line) = lines.get(line_index) {
                self.write_timestamp(output);
                self.apply_style(output, TextStyle::Thinking);
                let _ = write!(output, "{line}");
                self.reset_style(output);
            }
            let _ = writeln!(output);
        }

        let stale_lines = rows.saturating_sub(lines.len());
        if stale_lines > 0 {
            let _ = write!(output, "\x1b[{stale_lines}A");
        }
    }

    /// Redraw a dim spinner line. A [`Segment::Spinner`] may appear anywhere in `segments`.
    pub fn spinner<'a>(self, first: bool, segments: impl IntoIterator<Item = Segment<'a>>) {
        if self.progress && self.should_write_spinner(first) {
            self.write_line(self.interactive && !first, false, segments);
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

    /// Render a line, optionally replacing the previous line.
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
        self.write_rendered_line(&mut stderr, segments);
        let _ = stderr.flush();
    }

    /// Render timestamp and segments to an output stream.
    fn write_rendered_line<'a, W: io::Write>(
        self,
        output: &mut W,
        segments: impl IntoIterator<Item = Segment<'a>>,
    ) {
        self.write_timestamp(output);
        self.write_segments(output, segments);
        let _ = writeln!(output);
    }

    /// Render styled segments to an output stream.
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

    /// Check whether a spinner should be emitted.
    const fn should_write_spinner(self, first: bool) -> bool {
        self.interactive || first
    }

    /// Apply an ANSI style when colors are enabled.
    fn apply_style<W: io::Write>(self, output: &mut W, style: TextStyle) {
        if self.color {
            let _ = write!(output, "{}{}", Self::CLEAR, Self::style_code(style));
        }
    }

    /// Reset ANSI styling when colors are enabled.
    fn reset_style<W: io::Write>(self, output: &mut W) {
        if self.color {
            let _ = write!(output, "{}", Self::CLEAR);
        }
    }

    /// Return the ANSI code for a text style.
    fn style_code(style: TextStyle) -> &'static str {
        match style {
            TextStyle::Timestamp => Self::TIMESTAMP,
            TextStyle::Neutral => Self::NEUTRAL,
            TextStyle::BoldNeutral => Self::BOLD_NEUTRAL,
            TextStyle::Green => Self::GREEN,
            TextStyle::Red => Self::RED,
            TextStyle::DimRed => Self::DIM_RED,
            TextStyle::Thinking => Self::THINKING,
            TextStyle::Orange => Self::ORANGE,
            TextStyle::Error => Self::ERROR,
        }
    }

    /// Write the optional local timestamp.
    fn write_timestamp<W: io::Write>(self, output: &mut W) {
        if self.timestamp {
            self.apply_style(output, TextStyle::Timestamp);
            let _ = write!(output, "[{}] ", timestamp());
            self.reset_style(output);
        }
    }
}

/// Return the current local time as `HH:MM:SS`.
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
                "\x1b[0m\x1b[32m+1\x1b[0m",
                "\x1b[0m\x1b[2;39m|\x1b[0m",
                "\x1b[0m\x1b[31m-1\x1b[0m",
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
