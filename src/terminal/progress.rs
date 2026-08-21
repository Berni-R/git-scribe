use std::time::{Duration, Instant};

use crate::ollama::{ChatEvent, ChatResponse};
use crate::segments;
use terminal_size::{Width, terminal_size};

use super::{Segment, Spinner, Terminal, TextStyle};

/// Maximum reasoning lines shown in the live preview.
const THINKING_PREVIEW_LINES: usize = 5;
/// Fallback preview width when terminal width is unavailable.
const THINKING_PREVIEW_FALLBACK_COLUMNS: usize = 72;

/// Renders the lifecycle of a streamed model response on a [`Terminal`].
///
/// This owns display-specific streaming state, leaving callers to submit the
/// request and handle the completed response.
pub struct ChatProgress {
    /// Terminal renderer.
    terminal: Terminal,
    /// Whether reasoning text is visible.
    show_thinking: bool,
    /// Spinner animation state.
    spinner: Spinner,
    /// Time the prompt was sent.
    prompt_sent_at: Instant,
    /// Start of reasoning, if any.
    thinking_started_at: Option<Instant>,
    /// Start of generation, if any.
    generating_started_at: Option<Instant>,
    /// Bounded reasoning preview.
    thinking_preview: ThinkingPreview,
    /// Number of currently rendered reasoning lines.
    rendered_thinking_lines: usize,
}

impl ChatProgress {
    /// Start displaying progress for a prompt being sent to `model`.
    #[must_use]
    pub fn new(terminal: Terminal, model: &str, show_thinking: bool) -> Self {
        terminal.status_segments(segments![
            Neutral: "Sending prompt to ";
            BoldNeutral: "{model}";
        ]);
        terminal.status(format_args!("Waiting for Ollama..."));

        Self {
            terminal,
            show_thinking,
            spinner: Spinner::default(),
            prompt_sent_at: Instant::now(),
            thinking_started_at: None,
            generating_started_at: None,
            thinking_preview: ThinkingPreview::new(thinking_preview_columns()),
            rendered_thinking_lines: 0,
        }
    }

    /// Render one event from a streaming chat response.
    pub fn handle(&mut self, event: ChatEvent) {
        match event {
            ChatEvent::ResponseStarted => self.response_started(),
            ChatEvent::Thinking(thinking) => self.thinking(&thinking),
            ChatEvent::Generating(_) => self.generating(),
        }
    }

    /// Finish the active display state using completed response metrics.
    pub fn finish(&mut self, response: &ChatResponse) {
        if let Some(generating_started_at) = self.generating_started_at {
            let total_tokens = response
                .prompt_eval_count
                .unwrap_or_default()
                .saturating_add(response.eval_count.unwrap_or_default());
            self.terminal.complete([Segment::text(
                TextStyle::Neutral,
                format_args!(
                    "Generation done in {} · {total_tokens} total tokens",
                    format_elapsed(generating_started_at.elapsed()),
                ),
            )]);
        } else if let Some(thinking_started_at) = self.thinking_started_at {
            self.finish_thinking("Thinking done", thinking_started_at);
        }
    }

    /// Render the response-started transition.
    fn response_started(&self) {
        self.terminal.complete([Segment::text(
            TextStyle::Neutral,
            format_args!(
                "Ollama responded in {}",
                format_elapsed(self.prompt_sent_at.elapsed())
            ),
        )]);
    }

    /// Render a reasoning fragment.
    fn thinking(&mut self, thinking: &str) {
        let first = self.thinking_started_at.is_none();
        let started_at = *self.thinking_started_at.get_or_insert_with(Instant::now);
        let elapsed = format_elapsed(started_at.elapsed());
        if self.show_thinking {
            let lines = self.thinking_preview.push(thinking);
            self.rendered_thinking_lines = self.terminal.thinking(
                first,
                self.rendered_thinking_lines,
                [
                    Segment::spinner(TextStyle::Neutral, self.spinner.next_frame()),
                    Segment::text(TextStyle::Neutral, format_args!(" Thinking ({elapsed})")),
                ],
                &lines,
            );
        } else {
            self.terminal.spinner(
                first,
                [
                    Segment::spinner(TextStyle::Neutral, self.spinner.next_frame()),
                    Segment::text(TextStyle::Neutral, format_args!(" Thinking ({elapsed})")),
                ],
            );
        }
    }

    /// Render the generation transition and spinner.
    fn generating(&mut self) {
        if self.generating_started_at.is_none()
            && let Some(thinking_started_at) = self.thinking_started_at
        {
            self.finish_thinking("Thought for", thinking_started_at);
        }

        let first = self.generating_started_at.is_none();
        let started_at = *self.generating_started_at.get_or_insert_with(Instant::now);
        self.terminal.spinner(
            first,
            [
                Segment::spinner(TextStyle::Neutral, self.spinner.next_frame()),
                Segment::text(
                    TextStyle::Neutral,
                    format_args!(" Generating ({})", format_elapsed(started_at.elapsed())),
                ),
            ],
        );
    }

    /// Complete the reasoning phase.
    fn finish_thinking(&mut self, label: &str, started_at: Instant) {
        let elapsed = format_elapsed(started_at.elapsed());
        if self.show_thinking {
            let lines = self.thinking_preview.finish();
            self.terminal.finish_thinking(
                self.rendered_thinking_lines,
                &lines,
                [Segment::text(
                    TextStyle::Neutral,
                    format_args!("{label} {elapsed}"),
                )],
            );
        } else {
            self.terminal.complete([Segment::text(
                TextStyle::Neutral,
                format_args!("{label} {elapsed}"),
            )]);
        }
    }
}

/// Format elapsed time as seconds or minutes and seconds.
fn format_elapsed(duration: Duration) -> String {
    let seconds = duration
        .as_secs()
        .saturating_add(u64::from(duration.subsec_millis() >= 500));
    let minutes = seconds / 60;
    let seconds = seconds % 60;

    if minutes == 0 {
        format!("{seconds}s")
    } else {
        format!("{minutes}m {seconds}s")
    }
}

struct ThinkingPreview {
    /// Maximum display width.
    column_limit: usize,
    /// Completed preview lines.
    completed_lines: Vec<String>,
    /// Current unfinished line.
    current_line: String,
}

impl ThinkingPreview {
    /// Create an empty preview with a column limit.
    const fn new(column_limit: usize) -> Self {
        Self {
            column_limit,
            completed_lines: Vec::new(),
            current_line: String::new(),
        }
    }

    /// Add streamed text and return visible lines.
    fn push(&mut self, fragment: &str) -> Vec<String> {
        for character in fragment.chars() {
            match character {
                '\r' => {}
                '\n' => self.complete_current_line(),
                _ => {
                    self.current_line.push(character);
                    if self.current_line.chars().count() == self.column_limit {
                        self.complete_current_line();
                    }
                }
            }
        }
        self.visible_lines()
    }

    /// Flush the current line and return visible lines.
    fn finish(&mut self) -> Vec<String> {
        if !self.current_line.is_empty() {
            self.complete_current_line();
        }
        self.visible_lines()
    }

    /// Move the current line into the bounded history.
    fn complete_current_line(&mut self) {
        self.completed_lines
            .push(std::mem::take(&mut self.current_line));
        if self.completed_lines.len() > THINKING_PREVIEW_LINES {
            self.completed_lines.remove(0);
        }
    }

    /// Return only the most recent visible lines.
    fn visible_lines(&self) -> Vec<String> {
        let mut lines = self.completed_lines.clone();
        if !self.current_line.is_empty() {
            lines.push(self.current_line.clone());
        }
        let first = lines.len().saturating_sub(THINKING_PREVIEW_LINES);
        lines.drain(..first);
        lines
    }
}

/// Determine preview width from the connected terminal.
fn thinking_preview_columns() -> usize {
    const TIMESTAMP_COLUMNS: usize = 11; // `[HH:MM:SS] `
    const MINIMUM_COLUMNS: usize = 20;

    terminal_size()
        .map(|(Width(columns), _)| usize::from(columns))
        .filter(|&columns| columns >= MINIMUM_COLUMNS)
        .map_or(THINKING_PREVIEW_FALLBACK_COLUMNS, |columns| {
            columns.saturating_sub(TIMESTAMP_COLUMNS).max(1)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_times_round_to_whole_seconds_and_include_minutes() {
        assert_eq!(format_elapsed(Duration::from_millis(499)), "0s");
        assert_eq!(format_elapsed(Duration::from_millis(500)), "1s");
        assert_eq!(format_elapsed(Duration::from_secs(70)), "1m 10s");
    }

    #[test]
    fn thinking_preview_keeps_the_latest_five_lines_across_fragments() {
        let mut preview = ThinkingPreview::new(80);
        preview.push("one\ntwo\n");
        let lines = preview.push("three\nfour\nfive\nsix");

        assert_eq!(lines, ["two", "three", "four", "five", "six"]);
    }
}
