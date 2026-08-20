use std::{
    fmt,
    io::{self, Write as _},
};

/// Styled terminal output reserved for diagnostics on stderr.
#[derive(Debug, Clone, Copy)]
pub struct Terminal {
    color: bool,
}

impl Terminal {
    /// Create terminal output, optionally decorated with ANSI colors.
    #[must_use]
    pub const fn new(color: bool) -> Self {
        Self { color }
    }

    /// Write a dim diagnostic line to stderr.
    pub fn status(self, message: fmt::Arguments<'_>) {
        self.write("\x1b[2m", message, true);
    }

    /// Write an orange warning line to stderr.
    pub fn warning(self, message: fmt::Arguments<'_>) {
        self.write("\x1b[38;5;208m", message, true);
    }

    /// Redraw a dim spinner status on the current stderr line.
    pub fn spinner(self, frame: &str, message: fmt::Arguments<'_>) {
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\r");
        if self.color {
            let _ = write!(stderr, "\x1b[2m");
        }
        let _ = write!(stderr, "{frame} ");
        let _ = stderr.write_fmt(message);
        if self.color {
            let _ = write!(stderr, "\x1b[0m");
        }
        let _ = stderr.flush();
    }

    /// Replace an active spinner with a successful completion marker.
    pub fn complete(self, message: fmt::Arguments<'_>) {
        let mut stderr = io::stderr().lock();
        let _ = write!(stderr, "\r");
        if self.color {
            // let _ = write!(stderr, "\x1b[2;32m"); // dim green
            let _ = write!(stderr, "\x1b[2m");
        }
        let _ = write!(stderr, "✔ ");
        let _ = stderr.write_fmt(message);
        // Clear any trailing characters left by a longer spinner message without adding ANSI
        // control sequences when colors are disabled.
        let _ = write!(stderr, "        ");
        if self.color {
            let _ = write!(stderr, "\x1b[0m");
        }
        let _ = writeln!(stderr);
        let _ = stderr.flush();
    }

    /// Write an error and its context chain in red to stderr.
    pub fn error(self, error: &anyhow::Error) {
        self.write("\x1b[31m", format_args!("{error:?}"), true);
    }

    fn write(self, style: &str, message: fmt::Arguments<'_>, newline: bool) {
        let mut stderr = io::stderr().lock();
        if self.color {
            let _ = write!(stderr, "{style}");
        }
        let _ = stderr.write_fmt(message);
        if self.color {
            let _ = write!(stderr, "\x1b[0m");
        }
        if newline {
            let _ = writeln!(stderr);
        }
        let _ = stderr.flush();
    }
}
