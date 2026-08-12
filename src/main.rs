use std::{path::PathBuf, time::Duration};

use clap::Parser;
use git_sight::ollama;

#[derive(Debug, Parser)]
#[command(version, about = "Generate commit messages from staged changes")]
struct Cli {
    /// Repository to inspect.
    #[arg(value_name = "PATH", default_value = ".")]
    path: PathBuf,

    /// Ollama model to use.
    #[arg(long, default_value = "qwen3:4b-instruct")]
    model: String,

    /// Model context-window size, in tokens.
    #[arg(short, long, default_value_t = 16_384)]
    context_size: u32,

    /// Keep the Ollama model loaded in memory after the request.
    ///
    /// Accepts human-readable durations such as `30s`, `5m`, or `1h`.
    /// A duration of `0` unloads the model immediately after the request.
    /// If omitted, Ollama's configured default is used.
    #[arg(long, value_parser = humantime::parse_duration)]
    keep_alive: Option<Duration>,
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    dbg!(&args);

    let ollama = ollama::Client::default();
    let models = ollama.list_models()?;
    println!("available Ollama models:");
    println!("{:30}  {:>8}  {:>5}", "NAME", "SIZE", "LOCAL",);
    for model in models {
        println!(
            "{:30}  {:5.2} GB  {:^5}",
            model.model,
            (model.size as f64) / 1024_f64.powi(3),
            if model.is_local() { "✔" } else { " " },
        )
    }
    println!();

    let messages = vec![
        ollama::Message {
            role: ollama::Role::System,
            content: "You are a scientifically correct, concise knowledge assistant.".to_string(),
        },
        ollama::Message {
            role: ollama::Role::User,
            content: "Why is the sky blue?".to_string(),
        },
    ];
    let options = ollama::ChatOptions {
        options: Some(ollama::ModelOptions {
            num_ctx: Some(args.context_size),
            num_predict: Some(256),
            ..Default::default()
        }),
        keep_alive: args.keep_alive.map(Into::into),
        ..Default::default()
    };
    dbg!(&options);
    let answer = ollama.chat(args.model, messages, &options)?;

    println!("Model used: {:?}", answer.model);
    println!("Done: {:?}", answer.done);
    println!("Done reasoning: {:?}", answer.done_reason);
    println!("prompt tokens: {:?}", answer.prompt_eval_count);
    println!("tokens generated: {:?}", answer.eval_count);
    println!();
    println!("{}", answer.message.content);

    // let repo = GitRepo::discover(args.path)?;

    Ok(())
}
