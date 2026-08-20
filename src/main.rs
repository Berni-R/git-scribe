#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

use anyhow::{Context as _, Result, bail};
use clap::Parser as _;
use git_scribe::{
    GitRepo,
    generation::{CommitMessage, Confidence, Prompt},
    ollama::{self, ChatOptions, KeepAlive, Message, ModelOptions, Role},
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
fn main() -> Result<()> {
    let args = cli::Cli::parse();
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

    eprintln!(
        "git-scribe: {} file(s) in commit, ~{}/{} (~{:.0}%) prompt tokens used, {}",
        commit.len(),
        prompt.estimated_tokens,
        prompt_token_budget,
        100.0 * prompt.estimated_tokens as f64 / prompt_token_budget as f64,
        args.model,
    );

    if let Some(path) = &args.context_file {
        prompt.write_context(path)?;
        eprintln!("git-scribe: wrote model context to {}", path.display());
    }

    let client = ollama::Client::default();

    let response = client.chat(
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
        &ChatOptions {
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
        },
    )?;

    if let Some(actual) = response.prompt_eval_count {
        eprintln!("git-scribe: Ollama actually used {actual} prompt tokens");
    }

    eprintln!(
        "git-scribe: Ollama generated {} tokens, done reason: {:?}",
        response.eval_count.unwrap_or(0),
        response.done_reason,
    );

    if let Some(thinking) = response.message.thinking.as_deref()
        && !thinking.is_empty()
    {
        eprintln!(
            "git-scribe: model produced {} bytes of thinking",
            thinking.len(),
        );
    }

    let content = response.message.content.trim();

    if content.is_empty() {
        bail!(
            "model returned an empty response \
         (generated {} tokens, done reason: {:?}, thinking: {} bytes)",
            response.eval_count.unwrap_or(0),
            response.done_reason,
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

    if response.done_reason.as_deref() == Some("length") {
        eprintln!("git-scribe: warning: model output hit the token limit");
    }
    if matches!(answer.confidence, Confidence::Low) {
        eprintln!(
            "git-scribe: warning: model reports low confidence ({:?}): {}",
            answer.change_kind, answer.intent,
        );
    }

    if args.show_analysis {
        eprintln!("git-scribe: model analysis:");
        eprintln!("  intent: {}", answer.intent);
        eprintln!("  kind: {:?}", answer.change_kind);
        eprintln!("  confidence: {:?}", answer.confidence);

        if !answer.key_changes.is_empty() {
            eprintln!("  key changes:");
            for change in &answer.key_changes {
                eprintln!("    - {change}");
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
        eprintln!("git-scribe: opening the Git commit editor");
        repo.commit_interactively(mode, &commit_message)?;
    }

    Ok(())
}
