use std::{path::PathBuf, time::Duration};

use clap::Parser;
use git_scribe::{git::CommitMode, ollama::Think};

/// Parse a token count with optional `k` or `m` binary suffixes.
fn parse_token_count(value: &str) -> Result<u32, String> {
    let value = value.replace('_', "");
    let (number, multiplier) = value
        .strip_suffix('k')
        .or_else(|| value.strip_suffix('K'))
        .map_or_else(
            || {
                value
                    .strip_suffix('m')
                    .or_else(|| value.strip_suffix('M'))
                    .map_or((&value[..], 1), |number| (number, 1024 * 1024))
            },
            |number| (number, 1024),
        );
    let number = number
        .parse::<u32>()
        .map_err(|_| format!("expected a positive token count such as `64k`, got `{value}`"))?;
    let tokens = number
        .checked_mul(multiplier)
        .ok_or_else(|| format!("token count `{value}` exceeds the supported range"))?;

    if tokens == 0 {
        return Err("token count must be positive".to_owned());
    }
    Ok(tokens)
}

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

    /// Generate a message for amending HEAD.
    ///
    /// Uses staged changes and compares them with HEAD's first parent. Merge commits are unsupported.
    #[arg(long, help_heading = "Commit")]
    pub amend: bool,

    /// Print the generated message instead of opening Git's editor.
    #[arg(short, long, help_heading = "Commit")]
    pub print: bool,

    /// Use only the generated one-line subject.
    #[arg(short, long, help_heading = "Commit")]
    pub no_body: bool,

    /// Add an author-provided hint about the commit.
    ///
    /// Repeat to provide intent or motivation that is not evident from the staged diff.
    #[arg(long, value_name = "TEXT", help_heading = "Prompt")]
    pub hint: Vec<String>,

    /// Omit a repository-relative file's diff and syntax context from the prompt.
    ///
    /// The file's status remains visible. Repeat to exclude multiple files.
    #[arg(short = 'x', long, value_name = "PATH", help_heading = "Prompt")]
    pub exclude_diff: Vec<PathBuf>,

    /// Ollama model name; it must already be available in Ollama.
    #[arg(short, long, default_value = "qwen3.5:9b", help_heading = "Model")]
    pub model: String,

    /// Maximum model context window in tokens.
    ///
    /// Supports binary suffixes: `k` is 1024 and `m` is 1024²; for example, `-c 64k` or `-c 2m`.
    /// The window includes prompt, reasoning, generated output, and safety reserves.
    #[arg(
        short = 'c',
        long,
        value_name = "TOKENS",
        default_value = "16k",
        value_parser = parse_token_count,
        help_heading = "Model"
    )]
    pub context_window: u32,

    /// Whether and how strongly the model should use explicit thinking.
    ///
    /// Use `--think` or `-T` for `on`; use `--think=high` for a level.
    #[arg(
        short = 'T',
        long,
        value_enum,
        default_value_t = Think::Off,
        default_missing_value = "on",
        num_args = 0..=1,
        require_equals = true,
        help_heading = "Model"
    )]
    pub think: Think,

    /// Show the latest five lines of the streamed model reasoning trace.
    ///
    /// Requires thinking to be enabled with `--think`.
    #[arg(long, help_heading = "Model")]
    pub show_thinking: bool,

    /// Sampling temperature.
    ///
    /// Lower values are more deterministic.
    #[arg(short, long, default_value_t = 0.0, help_heading = "Model")]
    pub temperature: f32,

    /// Random seed for reproducible output on supported models.
    #[arg(long, help_heading = "Model")]
    pub seed: Option<i64>,

    /// Keep the Ollama model loaded after execution.
    ///
    /// Accepts durations such as `30s`, `2m`, and `1h`. Use `0s` to unload it immediately;
    /// omit to use Ollama's default.
    #[arg(short, long, value_parser = humantime::parse_duration, help_heading = "Runtime")]
    pub keep_alive: Option<Duration>,

    /// Maximum time to wait for Ollama to respond, such as `30s`, `2m`, or `1h`.
    #[arg(long, value_parser = humantime::parse_duration, default_value = "2m", help_heading = "Runtime")]
    pub timeout: Duration,

    /// Print the model's structured analysis to stderr.
    #[arg(long, help_heading = "Diagnostics")]
    pub show_analysis: bool,

    /// Write the system and user prompt sent to the model.
    #[arg(long, value_name = "FILE", help_heading = "Diagnostics")]
    pub prompt_file: Option<PathBuf>,

    /// Write the complete streamed thinking trace and generated response to this file.
    #[arg(long, value_name = "FILE", help_heading = "Diagnostics")]
    pub stream_file: Option<PathBuf>,

    /// Disable ANSI colors in diagnostic output.
    #[arg(long, help_heading = "Diagnostics")]
    pub no_color: bool,

    /// Suppress progress diagnostics; errors are still shown.
    #[arg(short = 'q', long, help_heading = "Diagnostics")]
    pub quiet: bool,
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
        let long = Cli::try_parse_from(["git-scribe", "--quiet"]).unwrap();

        assert!(short.quiet);
        assert!(long.quiet);
        assert!(Cli::try_parse_from(["git-scribe", "--quite"]).is_err());
    }

    #[test]
    fn hints_and_diagnostic_files_are_parsed() {
        let cli = Cli::try_parse_from([
            "git-scribe",
            "--hint",
            "explain the migration",
            "--show-thinking",
            "--prompt-file",
            "prompt.txt",
            "--stream-file",
            "response.txt",
        ])
        .unwrap();

        assert_eq!(cli.hint, ["explain the migration"]);
        assert!(cli.show_thinking);
        assert_eq!(cli.prompt_file, Some(PathBuf::from("prompt.txt")));
        assert_eq!(cli.stream_file, Some(PathBuf::from("response.txt")));
    }

    #[test]
    fn context_window_accepts_binary_suffixes() {
        let cli = Cli::try_parse_from(["git-scribe", "-c", "64k"]).unwrap();

        assert_eq!(cli.context_window, 65_536);
        assert_eq!(parse_token_count("2m"), Ok(2_097_152));
        assert!(parse_token_count("0").is_err());
    }

    #[test]
    fn thinking_flag_defaults_to_on_when_its_value_is_omitted() {
        let long = Cli::try_parse_from(["git-scribe", "--think"]).unwrap();
        let short = Cli::try_parse_from(["git-scribe", "-T"]).unwrap();
        let level = Cli::try_parse_from(["git-scribe", "--think=high"]).unwrap();

        assert_eq!(long.think, Think::On);
        assert_eq!(short.think, Think::On);
        assert_eq!(level.think, Think::High);
    }
}
