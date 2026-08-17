use std::{path::PathBuf, time::Duration};

use clap::Parser;
use git_sight::{git::CommitMode, ollama::Think};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Generate a commit message and open it in Git's commit editor"
)]
pub struct Cli {
    /// Repository to inspect.
    #[arg(value_name = "PATH", default_value = ".")]
    pub path: PathBuf,

    /// Generate a message for amending the current HEAD commit.
    #[arg(long)]
    pub amend: bool,

    /// Optional context or hints about the commit.
    #[arg(long, value_name = "TEXT")]
    pub context: Vec<String>,

    /// Ollama model to use (e.g. "gemma4:e2b", "qwen3.5:4b", "qwen3:4b-instruct", "mistral:7b").
    ///
    /// Note:
    /// As of now (Aug 2026), models using Apple MLX framework do not respect the output format with Ollama.
    ///
    /// See <https://github.com/ollama/ollama/issues/16563>.
    #[arg(long, default_value = "qwen3.5:4b")]
    pub model: String,

    /// Prefill the commit editor with only the generated one-line subject.
    #[arg(short, long)]
    pub no_body: bool,

    /// Model context window, in tokens.
    #[arg(short = 'c', long, default_value_t = 16_384)]
    pub model_context: u32,

    /// Sampling temperature used by the model.
    #[arg(long, default_value_t = 0.15)]
    pub temperature: f32,

    /// Random seed used for reproducible outputs, if given.
    #[arg(long)]
    pub seed: Option<i64>,

    /// Whether and how strongly the model should use explicit thinking.
    #[arg(long, value_enum, default_value_t = Think::Off)]
    pub think: Think,

    /// Keep the Ollama model alive after execution.
    ///
    /// A value of `0` unloads the model immediately.
    /// If not specified, use the Ollama default.
    #[arg(long, value_parser = humantime::parse_duration)]
    pub keep_alive: Option<Duration>,

    /// Print the model's structured analysis to stderr.
    #[arg(long)]
    pub show_analysis: bool,

    /// Write the complete generated model context to this file.
    #[arg(long, value_name = "FILE")]
    pub context_file: Option<PathBuf>,
}

impl Cli {
    /// The [`CommitMode`] derived from the argument `--amend`.
    pub fn commit_mode(&self) -> CommitMode {
        if self.amend {
            CommitMode::Amend
        } else {
            CommitMode::Normal
        }
    }

    /// Validate all parameters.
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            anyhow::bail!(
                "temperature must be a finite, non-negative number, got {:?}",
                self.temperature
            );
        }

        Ok(())
    }
}
