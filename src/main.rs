#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

use std::{
    env, fs,
    io::ErrorKind,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, bail};
use clap::Parser as _;
use git_scribe::{
    GitRepo,
    generation::{CommitMessage, Confidence, Prompt},
    ollama::{
        self, ChatEvent, ChatOptions, KeepAlive, Message, ModelOptions, Role, is_model_contained,
    },
    terminal::{Segment, Spinner, Terminal, TextStyle},
};

mod cli;

/// Maximum number of tokens Ollama may generate for the structured response.
///
/// A commit message should be much shorter than this, but the schema permits a 100-character subject,
/// a body of up to 700 characters, and a few small classification fields.
/// 384 leaves enough room for that worst case without reserving a significant fraction of a typical 16k model context.
const NUM_PREDICT: i32 = 384;

/// Minimum context reserved for model reasoning and the final response when generation itself is not capped.
const THINKING_CONTEXT_RESERVE: u32 = 4_096;

const THINKING_PREVIEW_LINES: usize = 5;
const THINKING_PREVIEW_FALLBACK_COLUMNS: usize = 72;

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn main() {
    let args = cli::Cli::parse();
    let terminal = Terminal::new(!args.no_color, true).with_progress(!args.quiet);

    if let Err(error) = run(&args, terminal) {
        terminal.error(&error);
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn run(args: &cli::Cli, terminal: Terminal) -> Result<()> {
    args.validate()?;
    ensure_output_paths_are_available(args)?;
    let mode = args.commit_mode();

    let predict_reserve = if args.think.is_on() {
        THINKING_CONTEXT_RESERVE
    } else {
        NUM_PREDICT as u32
    };
    let prompt_token_budget = Prompt::available_tokens(args.model_context, predict_reserve)?;

    let repo = GitRepo::discover(&args.path)?;
    let commit = repo.prospective_commit(mode)?;

    if commit.is_empty() && !args.amend {
        bail!("no staged changes");
    }

    let prompt = Prompt::new(
        &repo,
        &args.context,
        &commit,
        &args.exclude_diff,
        prompt_token_budget,
    )?;

    let stats = commit.stats();
    terminal.status_segments([
        Segment::text(
            TextStyle::Neutral,
            format_args!("{} file(s) in commit; ", stats.files_changed),
        ),
        Segment::text(TextStyle::Neutral, format_args!("line changes: ")),
        Segment::text(TextStyle::Green, format_args!("+{}", stats.insertions)),
        Segment::text(TextStyle::Neutral, format_args!(" ")),
        Segment::text(TextStyle::Red, format_args!("-{}", stats.deletions)),
    ]);
    let token_bar = Terminal::progress_bar(prompt.estimated_tokens, prompt_token_budget, 25);
    let prompt_token_percentage = prompt.estimated_tokens.saturating_mul(100) / prompt_token_budget;
    terminal.status(format_args!(
        "Estimated input budget: {token_bar} {} / {} tokens ({}%)",
        prompt.estimated_tokens, prompt_token_budget, prompt_token_percentage,
    ));

    if let Some(path) = &args.context_file {
        prompt.write_context(path)?;
        terminal.status(format_args!("Wrote model context to: {}", path.display()));
    }

    let client = ollama::Client::default();
    let chat_options = ChatOptions {
        options: Some(ModelOptions {
            temperature: Some(args.temperature),
            num_ctx: Some(args.model_context),
            num_predict: if args.think.is_on() {
                None
            } else {
                Some(NUM_PREDICT)
            },
            seed: args.seed,
        }),
        think: Some(args.think),
        format: Some(CommitMessage::schema()),
        keep_alive: args.keep_alive.map(KeepAlive::from),
        timeout: Some(args.timeout),
    };

    {
        let loaded = client.list_running_models()?;
        if !is_model_contained(&args.model, args.model_context, &loaded) {
            terminal.status_segments([
                Segment::text(TextStyle::Neutral, format_args!("Loading ")),
                Segment::text(TextStyle::BoldNeutral, format_args!("{}", args.model)),
                Segment::text(
                    TextStyle::Neutral,
                    format_args!(" with a {} token context...", args.model_context),
                ),
            ]);
            if let Some(done) = client.prepare_model(&args.model, &chat_options)?
                && done != "load"
            {
                terminal.warning(format_args!("Unexpected done reason: {done}"));
            }
        }
    }

    let mut spinner = Spinner::default();
    let prompt_sent_at = Instant::now();
    let mut thinking_started_at = None;
    let mut generating_started_at = None;
    let mut thinking_preview = ThinkingPreview::new(thinking_preview_columns());
    let mut rendered_thinking_lines = 0;
    terminal.status_segments([
        Segment::text(TextStyle::Neutral, format_args!("Sending prompt to ")),
        Segment::text(TextStyle::BoldNeutral, format_args!("{}", args.model)),
    ]);
    terminal.status(format_args!("Waiting for Ollama..."));
    let response = client.chat_stream(
        &args.model,
        vec![
            Message {
                role: Role::System,
                content: Prompt::SYSTEM.to_owned(),
            },
            Message {
                role: Role::User,
                content: prompt.text,
            },
        ],
        &chat_options,
        |event| match event {
            ChatEvent::ResponseStarted => {
                let response_time = prompt_sent_at.elapsed();
                terminal.complete([Segment::text(
                    TextStyle::Neutral,
                    format_args!("Ollama responded in {}", format_elapsed(response_time)),
                )]);
            }
            ChatEvent::Thinking(thinking) => {
                let first = thinking_started_at.is_none();
                let thinking_elapsed = thinking_started_at
                    .get_or_insert_with(Instant::now)
                    .elapsed();
                if args.show_thinking {
                    rendered_thinking_lines = terminal.thinking(
                        first,
                        rendered_thinking_lines,
                        [
                            Segment::spinner(TextStyle::Neutral, spinner.next_frame()),
                            Segment::text(
                                TextStyle::Neutral,
                                format_args!(" Thinking ({})", format_elapsed(thinking_elapsed)),
                            ),
                        ],
                        &thinking_preview.push(&thinking),
                    );
                } else {
                    terminal.spinner(
                        first,
                        [
                            Segment::spinner(TextStyle::Neutral, spinner.next_frame()),
                            Segment::text(
                                TextStyle::Neutral,
                                format_args!(" Thinking ({})", format_elapsed(thinking_elapsed)),
                            ),
                        ],
                    );
                }
            }
            ChatEvent::Generating(_) => {
                if generating_started_at.is_none()
                    && let Some(thinking_started_at) = thinking_started_at
                {
                    if args.show_thinking {
                        terminal.finish_thinking(
                            rendered_thinking_lines,
                            &thinking_preview.finish(),
                            [Segment::text(
                                TextStyle::Neutral,
                                format_args!(
                                    "Thought for {}",
                                    format_elapsed(thinking_started_at.elapsed()),
                                ),
                            )],
                        );
                    } else {
                        terminal.complete([Segment::text(
                            TextStyle::Neutral,
                            format_args!(
                                "Thought for {}",
                                format_elapsed(thinking_started_at.elapsed()),
                            ),
                        )]);
                    }
                }
                let first = generating_started_at.is_none();
                let generating_elapsed = generating_started_at
                    .get_or_insert_with(Instant::now)
                    .elapsed();
                terminal.spinner(
                    first,
                    [
                        Segment::spinner(TextStyle::Neutral, spinner.next_frame()),
                        Segment::text(
                            TextStyle::Neutral,
                            format_args!(" Generating ({})", format_elapsed(generating_elapsed)),
                        ),
                    ],
                );
            }
        },
    )?;
    if let Some(generating_started_at) = generating_started_at {
        let total_tokens = response
            .prompt_eval_count
            .unwrap_or_default()
            .saturating_add(response.eval_count.unwrap_or_default());
        terminal.complete([Segment::text(
            TextStyle::Neutral,
            format_args!(
                "Generation done in {} · {total_tokens} total tokens",
                format_elapsed(generating_started_at.elapsed()),
            ),
        )]);
    } else if let Some(thinking_started_at) = thinking_started_at {
        if args.show_thinking {
            terminal.finish_thinking(
                rendered_thinking_lines,
                &thinking_preview.finish(),
                [Segment::text(
                    TextStyle::Neutral,
                    format_args!(
                        "Thinking done in {}",
                        format_elapsed(thinking_started_at.elapsed()),
                    ),
                )],
            );
        } else {
            terminal.complete([Segment::text(
                TextStyle::Neutral,
                format_args!(
                    "Thinking done in {}",
                    format_elapsed(thinking_started_at.elapsed()),
                ),
            )]);
        }
    }

    if let Some(path) = &args.stream_file {
        write_stream_file(
            path,
            response.message.thinking.as_deref(),
            &response.message.content,
        )?;
        terminal.status(format_args!("Wrote model streams to: {}", path.display()));
    }

    match response.done_reason.as_deref() {
        Some("stop") => {}
        Some("length") => terminal.warning(format_args!("Model output hit the token limit")),
        Some(done) => terminal.warning(format_args!("Unexpected done reason: {done}")),
        None => terminal.warning(format_args!("No done reason reported")),
    }

    let content = response.message.content.trim();
    if content.is_empty() {
        bail!(
            "model returned an empty response (generated {} tokens, thinking: {} bytes)",
            response.eval_count.unwrap_or(0),
            response.message.thinking.as_deref().map_or(0, str::len),
        );
    }

    let answer: CommitMessage = match serde_json::from_str(content) {
        Ok(answer) => answer,
        Err(error) => {
            if !content.trim().contains('\n') {
                println!("{}", content.trim()); // might still be a valid suggestion
            }
            return Err(error).context("model returned invalid commit-message JSON");
        }
    };

    if matches!(answer.confidence, Confidence::Low) {
        terminal.warning(format_args!(
            "Model reports low confidence ({:?}): {}",
            answer.change_kind, answer.intent,
        ));
    }

    if args.show_analysis {
        terminal.status(format_args!("Model analysis:"));
        terminal.status(format_args!("  intent: {}", answer.intent));
        terminal.status(format_args!("  kind: {:?}", answer.change_kind));
        terminal.status(format_args!("  confidence: {:?}", answer.confidence));

        if !answer.key_changes.is_empty() {
            terminal.status(format_args!("  key changes:"));
            for change in &answer.key_changes {
                terminal.status(format_args!("    - {change}"));
            }
        }
    }

    let subject = answer.normalized_subject()?;
    let mut commit_message = subject;
    if !args.no_body
        && let Some(body) = answer.normalized_body()
    {
        commit_message.push_str("\n\n");
        commit_message.push_str(body);
    }

    if args.print {
        println!("{commit_message}");
    } else {
        terminal.status(format_args!("Opening the Git commit editor"));
        repo.commit_interactively(mode, &commit_message)?;
    }

    Ok(())
}

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
    column_limit: usize,
    completed_lines: Vec<String>,
    current_line: String,
}

impl ThinkingPreview {
    const fn new(column_limit: usize) -> Self {
        Self {
            column_limit,
            completed_lines: Vec::new(),
            current_line: String::new(),
        }
    }

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

    fn finish(&mut self) -> Vec<String> {
        if !self.current_line.is_empty() {
            self.complete_current_line();
        }
        self.visible_lines()
    }

    fn complete_current_line(&mut self) {
        self.completed_lines
            .push(std::mem::take(&mut self.current_line));
        if self.completed_lines.len() > THINKING_PREVIEW_LINES {
            self.completed_lines.remove(0);
        }
    }

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

fn thinking_preview_columns() -> usize {
    const TIMESTAMP_COLUMNS: usize = 11; // `[HH:MM:SS] `
    const MINIMUM_COLUMNS: usize = 20;

    env::var("COLUMNS")
        .ok()
        .and_then(|columns| columns.parse::<usize>().ok())
        .filter(|&columns| columns >= MINIMUM_COLUMNS)
        .map_or(THINKING_PREVIEW_FALLBACK_COLUMNS, |columns| {
            columns.saturating_sub(TIMESTAMP_COLUMNS).max(1)
        })
}

fn write_stream_file(path: &std::path::Path, thinking: Option<&str>, content: &str) -> Result<()> {
    let contents = format!(
        "# Thinking\n{}\n\n# Generation\n{content}",
        thinking.unwrap_or_default(),
    );
    write_new_file(path, &contents)
        .with_context(|| format!("failed to write model streams to {}", path.display()))
}

fn ensure_output_paths_are_available(args: &cli::Cli) -> Result<()> {
    if let (Some(context_file), Some(stream_file)) = (&args.context_file, &args.stream_file)
        && context_file == stream_file
    {
        bail!(
            "--context-file and --stream-file must name different files ({})",
            context_file.display(),
        );
    }

    for path in [args.context_file.as_deref(), args.stream_file.as_deref()]
        .into_iter()
        .flatten()
    {
        match fs::symlink_metadata(path) {
            Ok(_) => bail!("refusing to overwrite existing file: {}", path.display()),
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("failed to inspect output path {}", path.display()));
            }
        }
    }

    Ok(())
}

fn write_new_file(path: &Path, contents: &str) -> std::io::Result<()> {
    use std::io::Write as _;

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(contents.as_bytes())
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

    #[test]
    fn existing_output_file_is_rejected_before_generation() {
        let args = cli::Cli::try_parse_from(["git-scribe", "--stream-file", "Cargo.toml"]).unwrap();

        assert!(
            ensure_output_paths_are_available(&args)
                .unwrap_err()
                .to_string()
                .contains("refusing to overwrite")
        );
    }
}
