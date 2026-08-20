#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

use anyhow::{Context as _, Result, bail};
use clap::Parser as _;
use git_scribe::{
    GitRepo,
    generation::{CommitMessage, Confidence, Prompt},
    ollama::{self, ChatOptions, KeepAlive, Message, ModelOptions, Role, is_model_contained},
    terminal,
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
    let terminal = terminal::Terminal::new(!args.no_color);

    if let Err(error) = run(&args, terminal) {
        terminal.error(&error);
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn run(args: &cli::Cli, terminal: terminal::Terminal) -> Result<()> {
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

    terminal.status(format_args!("{} file(s) in commit", commit.len()));
    terminal.status(format_args!(
        "Estimated prompt token usage: {}/{} ({:.0}%)",
        prompt.estimated_tokens,
        prompt_token_budget,
        100.0 * prompt.estimated_tokens as f64 / prompt_token_budget as f64,
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
            terminal.status(format_args!(
                "Loading {} with a {}-token context...",
                args.model, args.model_context
            ));
            if let Some(done) = client.prepare_model(&args.model, &chat_options)?
                && done != "load"
            {
                terminal.warning(format_args!("Unexpected done reason: {done}"));
            }
        }
    }

    let mut spinner = terminal::Spinner::default();
    let mut thinking_started = false;
    let mut generating_started = false;
    terminal.status(format_args!("Sending prompt to {}", args.model));
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
            ollama::ChatEvent::Thinking(_) => {
                thinking_started = true;
                terminal.spinner(spinner.next_frame(), format_args!("Thinking"));
            }
            ollama::ChatEvent::Generating(_) => {
                if thinking_started && !generating_started {
                    terminal.complete(format_args!("Done thinking."));
                }
                generating_started = true;
                terminal.spinner(
                    spinner.next_frame(),
                    format_args!("Generating commit message"),
                );
            }
        },
    )?;
    if generating_started {
        terminal.complete(format_args!("Commit message generated."));
    }

    terminal.status(format_args!(
        "Tokens used: {} + {}",
        response.prompt_eval_count.unwrap_or(0),
        response.eval_count.unwrap_or(0),
    ));
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
