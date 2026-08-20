#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

use std::time::{Duration, Instant};

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

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn main() {
    let args = cli::Cli::parse();
    let terminal = Terminal::new(!args.no_color, true);

    if let Err(error) = run(&args, terminal) {
        terminal.error(&error);
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn run(args: &cli::Cli, terminal: Terminal) -> Result<()> {
    args.validate()?;
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
    terminal.status(format_args!(
        "Estimated prompt tokens: {token_bar} {}/{}",
        prompt.estimated_tokens, prompt_token_budget,
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
    terminal.status_segments([
        Segment::text(TextStyle::Neutral, format_args!("Sending prompt to ")),
        Segment::text(TextStyle::BoldNeutral, format_args!("{}", args.model)),
    ]);
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
                terminal.status(format_args!(
                    "Ollama responded in {}",
                    format_elapsed(response_time),
                ));
            }
            ChatEvent::Thinking(_) => {
                let first = thinking_started_at.is_none();
                thinking_started_at.get_or_insert_with(Instant::now);
                terminal.spinner(
                    first,
                    [
                        Segment::spinner(TextStyle::Neutral, spinner.next_frame()),
                        Segment::text(TextStyle::Neutral, format_args!(" Thinking")),
                    ],
                );
            }
            ChatEvent::Generating(_) => {
                if generating_started_at.is_none()
                    && let Some(thinking_started_at) = thinking_started_at
                {
                    terminal.complete([Segment::text(
                        TextStyle::Neutral,
                        format_args!(
                            "Thinking done in {}",
                            format_elapsed(thinking_started_at.elapsed()),
                        ),
                    )]);
                }
                let first = generating_started_at.is_none();
                generating_started_at.get_or_insert_with(Instant::now);
                terminal.spinner(
                    first,
                    [
                        Segment::spinner(TextStyle::Neutral, spinner.next_frame()),
                        Segment::text(TextStyle::Neutral, format_args!(" Generating")),
                    ],
                );
            }
        },
    )?;
    if let Some(generating_started_at) = generating_started_at {
        terminal.complete([Segment::text(
            TextStyle::Neutral,
            format_args!(
                "Generation done in {}",
                format_elapsed(generating_started_at.elapsed()),
            ),
        )]);
    } else if let Some(thinking_started_at) = thinking_started_at {
        terminal.complete([Segment::text(
            TextStyle::Neutral,
            format_args!(
                "Thinking done in {}",
                format_elapsed(thinking_started_at.elapsed()),
            ),
        )]);
    }

    // terminal.status(format_args!(
    //     "Tokens used: {} + {} = {}/{}",
    //     response.prompt_eval_count.unwrap_or(0),
    //     response.eval_count.unwrap_or(0),
    //     // TODO: double-check if that is the total budget
    //     response.prompt_eval_count.unwrap_or(0) + response.eval_count.unwrap_or(0),
    //     args.model_context,
    // ));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_times_round_to_whole_seconds_and_include_minutes() {
        assert_eq!(format_elapsed(Duration::from_millis(499)), "0s");
        assert_eq!(format_elapsed(Duration::from_millis(500)), "1s");
        assert_eq!(format_elapsed(Duration::from_secs(70)), "1m 10s");
    }
}
