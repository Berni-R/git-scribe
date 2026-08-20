use std::{path::PathBuf, time::Duration};

use clap::Parser;
use git_scribe::{git::CommitMode, ollama::Think};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Generate a commit message, then open it in Git's editor or print it"
)]
#[allow(clippy::struct_excessive_bools)] // Independent command-line switches are naturally boolean.
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

    /// Exclude a repository-relative file from the concrete and syntax diffs in the prompt.
    ///
    /// The file's added, modified, or deleted status remains available to the model.
    #[arg(long = "exclude-diff", value_name = "PATH")]
    pub exclude_diff: Vec<PathBuf>,

    /// Ollama model to use (e.g. "gemma4:e2b" or "qwen3:4b-instruct").
    ///
    /// Note:
    /// As of now (Aug 2026), models using Apple MLX framework do not respect the output format with Ollama.
    ///
    /// See <https://github.com/ollama/ollama/issues/16563>.
    #[arg(short, long, default_value = "qwen3.5:9b")]
    pub model: String,

    /// Use only the generated one-line subject, omitting the body.
    #[arg(short, long)]
    pub no_body: bool,

    /// Print the generated message instead of creating a commit.
    #[arg(short, long)]
    pub print: bool,

    /// Disable ANSI colors in diagnostic output.
    #[arg(long)]
    pub no_color: bool,

    /// Suppress progress output and print only the generated message.
    #[arg(short = 'q', long = "quite", alias = "quiet")]
    pub quiet: bool,

    /// Model context window, in tokens.
    #[arg(short = 'c', long, default_value_t = 16_384)]
    pub model_context: u32,

    /// Sampling temperature used by the model.
    #[arg(short, long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Random seed used for reproducible outputs, if given.
    #[arg(long)]
    pub seed: Option<i64>,

    /// Whether and how strongly the model should use explicit thinking.
    #[arg(long, value_enum, default_value_t = Think::Off)]
    pub think: Think,

    /// Show the latest five lines of the streamed model reasoning trace.
    #[arg(long)]
    pub show_thinking: bool,

    /// Write the complete thinking trace and generated response to this file.
    #[arg(long, value_name = "FILE")]
    pub stream_file: Option<PathBuf>,

    /// Keep the Ollama model alive after execution.
    ///
    /// A value of `0` unloads the model immediately.
    /// If not specified, use the Ollama default.
    #[arg(short, long, value_parser = humantime::parse_duration)]
    pub keep_alive: Option<Duration>,

    /// Maximum time to wait for Ollama to respond.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "2m")]
    pub timeout: Duration,

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_flag_selects_print_only_mode() {
        let cli = Cli::try_parse_from(["git-scribe", "--print"]).unwrap();

        assert!(cli.print);
        assert!(!cli.amend);
    }

    #[test]
    fn diff_exclusions_and_timeout_are_parsed() {
        let cli = Cli::try_parse_from([
            "git-scribe",
            "--exclude-diff",
            "generated.css",
            "--exclude-diff",
            "daisyui.mjs",
            "--timeout",
            "30s",
        ])
        .unwrap();

        assert_eq!(
            cli.exclude_diff,
            [PathBuf::from("generated.css"), PathBuf::from("daisyui.mjs")]
        );
        assert_eq!(cli.timeout, Duration::from_secs(30));
    }

    #[test]
    fn timeout_defaults_to_two_minutes() {
        let cli = Cli::try_parse_from(["git-scribe"]).unwrap();

        assert_eq!(cli.timeout, Duration::from_mins(2));
    }

    #[test]
    fn no_color_flag_disables_terminal_styling() {
        let cli = Cli::try_parse_from(["git-scribe", "--no-color"]).unwrap();

        assert!(cli.no_color);
    }

    #[test]
    fn quiet_flag_suppresses_progress_output() {
        let short = Cli::try_parse_from(["git-scribe", "-q"]).unwrap();
        let long = Cli::try_parse_from(["git-scribe", "--quite"]).unwrap();
        let conventional = Cli::try_parse_from(["git-scribe", "--quiet"]).unwrap();

        assert!(short.quiet);
        assert!(long.quiet);
        assert!(conventional.quiet);
    }

    #[test]
    fn thinking_display_and_stream_file_are_parsed() {
        let cli = Cli::try_parse_from([
            "git-scribe",
            "--show-thinking",
            "--stream-file",
            "response.txt",
        ])
        .unwrap();

        assert!(cli.show_thinking);
        assert_eq!(cli.stream_file, Some(PathBuf::from("response.txt")));
    }
}
