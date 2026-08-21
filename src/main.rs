#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

use std::{fs, io::ErrorKind, path::Path};

use anyhow::{Context as _, Result, bail};
use clap::Parser as _;
use git_scribe::{
    GitRepo,
    generation::{CommitMessage, Confidence, Prompt, PromptEstimate},
    git::ProspectiveCommit,
    ollama::{self, ChatOptions, KeepAlive, Message, ModelOptions, Role, is_model_contained},
    segments,
    terminal::{ChatProgress, Terminal},
};

mod cli;

/// Maximum number of tokens Ollama may generate for the structured response.
///
/// A commit message should be much shorter than this, but the schema permits a 100-character subject,
/// a body of up to 700 characters, and a few small classification fields.
/// 384 leaves enough room for that worst case without reserving a significant fraction of a typical 16k model context.
const NUM_PREDICT: i32 = 384;

/// Minimum context reserved for model reasoning when thinking is enabled.
const THINKING_CONTEXT_RESERVE: u32 = 4_096;

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

    let repo = GitRepo::discover(&args.path)?;
    let commit = repo.prospective_commit(mode)?;

    if commit.is_empty() && !args.amend {
        bail!("no staged changes");
    }

    let stats = commit.stats();
    terminal.status_segments(segments![
        BoldNeutral: "{} file(s)", stats.files_changed;
        Neutral: " in commit; line changes: ";
        Green: "+{}", stats.insertions;
        Neutral: " ";
        Red: "-{}", stats.deletions;
    ]);

    let (thinking_tokens, generation_tokens, predict_reserve, num_predict) = if args.think.is_on() {
        (
            THINKING_CONTEXT_RESERVE as usize,
            0,
            THINKING_CONTEXT_RESERVE,
            None,
        )
    } else {
        (
            0,
            NUM_PREDICT as usize,
            NUM_PREDICT as u32,
            Some(NUM_PREDICT),
        )
    };
    let estimate = Prompt::estimate(
        &repo,
        &args.hint,
        &commit,
        &args.exclude_diff,
        thinking_tokens,
        generation_tokens,
    )?;
    let prompt_token_budget = match Prompt::available_tokens(args.context_window, predict_reserve) {
        Ok(budget) => budget,
        Err(error) => {
            return Err(error.context(report_budget_overflow(
                terminal,
                &estimate,
                args.context_window,
                &commit,
            )?));
        }
    };

    let prompt = match Prompt::new(
        &repo,
        &args.hint,
        &commit,
        &args.exclude_diff,
        prompt_token_budget,
    ) {
        Ok(prompt) => prompt,
        Err(error) => {
            return Err(error.context(report_budget_overflow(
                terminal,
                &estimate,
                args.context_window,
                &commit,
            )?));
        }
    };

    let token_bar = Terminal::progress_bar(prompt.estimated_tokens, prompt_token_budget, 25);
    terminal.status(format_args!(
        "Estimated input budget: {token_bar} {} / {prompt_token_budget} tokens ",
        prompt.estimated_tokens,
    ));

    if let Some(path) = &args.prompt_file {
        prompt.write_context(path)?;
        terminal.status(format_args!("Wrote model context to: {}", path.display()));
    }

    let client = ollama::Client::default();
    let chat_options = ChatOptions {
        options: Some(ModelOptions {
            temperature: Some(args.temperature),
            num_ctx: Some(args.context_window),
            num_predict,
            seed: args.seed,
        }),
        think: Some(args.think),
        format: Some(CommitMessage::schema()),
        keep_alive: args.keep_alive.map(KeepAlive::from),
        timeout: Some(args.timeout),
    };

    {
        let loaded = client.list_running_models()?;
        if !is_model_contained(&args.model, args.context_window, &loaded) {
            terminal.status_segments(segments![
                Neutral: "Loading ";
                BoldNeutral: "{}", args.model;
                Neutral: " with a {} token context...", args.context_window;
            ]);
            if let Some(done) = client.prepare_model(&args.model, &chat_options)?
                && done != "load"
            {
                terminal.warning(format_args!("Unexpected done reason: {done}"));
            }
        }
    }

    let mut progress = ChatProgress::new(terminal, &args.model, args.show_thinking);
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
        |event| progress.handle(event),
    )?;
    progress.finish(&response);

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

/// Render an over-budget estimate and return actionable context-sizing guidance.
fn report_budget_overflow(
    terminal: Terminal,
    estimate: &PromptEstimate,
    model_context: u32,
    commit: &ProspectiveCommit,
) -> Result<String> {
    let required_context = u32::try_from(estimate.total_tokens())
        .context("required model context exceeds the supported token range")?;
    let required_context = 1000 * ((required_context + 500) / 1000);
    let overflow = estimate
        .total_tokens()
        .saturating_sub(model_context as usize);
    let token_bar =
        Terminal::progress_bar(estimate.total_tokens(), (model_context as usize).max(1), 25);
    terminal.status_segments(segments![
        Neutral: "Estimated context budget: ";
        DimRed: "{token_bar} {} / {model_context} tokens ({overflow} over)", estimate.total_tokens();
    ]);
    let mut budget_parts = vec![
        format!("system {}", estimate.system_tokens),
        format!("user {}", estimate.user_tokens),
    ];
    if estimate.thinking_tokens > 0 {
        budget_parts.push(format!("thinking {}", estimate.thinking_tokens));
    }
    if estimate.generation_tokens > 0 {
        budget_parts.push(format!("generation {}", estimate.generation_tokens));
    }
    budget_parts.push(format!("safety {}", estimate.safety_margin_tokens));
    terminal.status(format_args!("  {}", budget_parts.join(" + ")));
    report_file_change_stats(terminal, commit);

    Ok(format!(
        "Split the commit, increase --context-window to >~{required_context}, \
         use (multiple) --exclude-diff, or turn thinking off."
    ))
}

/// Render changed-line counts for every file in the prospective commit.
fn report_file_change_stats(terminal: Terminal, commit: &ProspectiveCommit) {
    terminal.status(format_args!("Changed lines by file:"));
    let changes = commit
        .changes()
        .iter()
        .map(|change| {
            let insertions = change
                .hunks
                .iter()
                .filter_map(|hunk| hunk.after)
                .map(|range| range.count)
                .sum::<usize>();
            let deletions = change
                .hunks
                .iter()
                .filter_map(|hunk| hunk.before)
                .map(|range| range.count)
                .sum::<usize>();
            (
                change.summary_line().replace('\t', " "),
                format!("+{insertions}"),
                format!("-{deletions}"),
            )
        })
        .collect::<Vec<_>>();
    let insertion_width = changes
        .iter()
        .map(|(_, insertions, _)| insertions.len())
        .max()
        .unwrap_or(1);
    let deletion_width = changes
        .iter()
        .map(|(_, _, deletions)| deletions.len())
        .max()
        .unwrap_or(1);
    let summary_width = changes
        .iter()
        .map(|(summary, _, _)| summary.len())
        .max()
        .unwrap_or(0);

    for (summary, insertions, deletions) in changes {
        terminal.status_segments(segments![
            Neutral: "  {summary:<summary_width$} | ";
            Green: "{insertions:>insertion_width$}";
            Neutral: " ";
            Red: "{deletions:>deletion_width$}";
        ]);
    }
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
    if let (Some(prompt_file), Some(stream_file)) = (&args.prompt_file, &args.stream_file)
        && prompt_file == stream_file
    {
        bail!(
            "--prompt-file and --stream-file must name different files ({})",
            prompt_file.display(),
        );
    }

    for path in [args.prompt_file.as_deref(), args.stream_file.as_deref()]
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
