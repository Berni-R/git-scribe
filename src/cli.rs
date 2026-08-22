use std::{path::PathBuf, time::Duration};

use anyhow::{Context as _, Result};
use clap::{ArgMatches, Parser, ValueEnum as _, parser::ValueSource};
use git_scribe::{GitRepo, git::CommitMode, ollama::Think};

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
    about = "Generate a commit message, then open it in Git's editor or print it",
    after_help = "Repository defaults may be set in .git/config, for example:\n  git config --local git-scribe.think high\n\nSupported keys: git-scribe.think, git-scribe.showThinking, git-scribe.temperature, git-scribe.seed, git-scribe.keepAlive, git-scribe.timeout, and repeated git-scribe.excludeDiff values. Command-line options take precedence; --exclude-diff values are added to configured exclusions."
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

    /// Omit a repository-relative file or directory's diff and syntax context from the prompt.
    ///
    /// Statuses remain visible. Repeat to exclude multiple files or directories.
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
    #[arg(long, value_parser = humantime::parse_duration, default_value = "10m", help_heading = "Runtime")]
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
    /// Apply repository-local defaults from `.git/config`.
    ///
    /// Values passed on the command line always take precedence. The supported
    /// keys are `git-scribe.think`, `git-scribe.showThinking`,
    /// `git-scribe.temperature`, `git-scribe.seed`, `git-scribe.keepAlive`,
    /// `git-scribe.timeout`, and repeated `git-scribe.excludeDiff` values.
    pub fn apply_local_config(&mut self, matches: &ArgMatches, repo: &GitRepo) -> Result<()> {
        let config = repo.local_config()?;

        Self::apply_config_value(matches, &config, "think", |value| {
            Think::from_str(value, true)
                .map(|think| self.think = think)
                .map_err(|error| error.clone())
        })?;
        Self::apply_config_value(matches, &config, "show_thinking", |value| {
            value
                .parse::<bool>()
                .map(|show_thinking| self.show_thinking = show_thinking)
                .map_err(|error| error.to_string())
        })?;
        Self::apply_config_value(matches, &config, "temperature", |value| {
            value
                .parse::<f32>()
                .map(|temperature| self.temperature = temperature)
                .map_err(|error| error.to_string())
        })?;
        Self::apply_config_value(matches, &config, "seed", |value| {
            value
                .parse::<i64>()
                .map(|seed| self.seed = Some(seed))
                .map_err(|error| error.to_string())
        })?;
        Self::apply_config_value(matches, &config, "keep_alive", |value| {
            humantime::parse_duration(value)
                .map(|keep_alive| self.keep_alive = Some(keep_alive))
                .map_err(|error| error.to_string())
        })?;
        Self::apply_config_value(matches, &config, "timeout", |value| {
            humantime::parse_duration(value)
                .map(|timeout| self.timeout = timeout)
                .map_err(|error| error.to_string())
        })?;
        self.append_config_exclude_diffs(&config)?;

        Ok(())
    }

    /// Configured exclusions are prepended so command-line occurrences retain
    /// their order and add to, rather than replace, repository defaults.
    fn append_config_exclude_diffs(&mut self, config: &git2::Config) -> Result<()> {
        const KEY: &str = "git-scribe.excludeDiff";
        let mut entries = match config.multivar(KEY, None) {
            Ok(entries) => entries,
            Err(error) if error.code() == git2::ErrorCode::NotFound => return Ok(()),
            Err(error) => return Err(error).with_context(|| format!("failed to read {KEY}")),
        };

        let mut configured = Vec::new();
        while let Some(entry) = entries.next() {
            let value = entry
                .with_context(|| format!("failed to read {KEY}"))?
                .value()
                .with_context(|| format!("invalid non-UTF-8 {KEY} value"))?;
            if value.is_empty() {
                anyhow::bail!("invalid {KEY} value: path must not be empty");
            }
            configured.push(PathBuf::from(value));
        }
        self.exclude_diff.splice(0..0, configured);
        Ok(())
    }

    fn apply_config_value(
        matches: &ArgMatches,
        config: &git2::Config,
        argument: &str,
        apply: impl FnOnce(&str) -> std::result::Result<(), String>,
    ) -> Result<()> {
        if matches.value_source(argument) == Some(ValueSource::CommandLine) {
            return Ok(());
        }

        let key = format!("git-scribe.{}", argument.replace('_', ""));
        match config.get_string(&key) {
            Ok(value) => apply(&value)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("invalid {key} value `{value}`")),
            Err(error) if error.code() == git2::ErrorCode::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| format!("failed to read {key}")),
        }
    }

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
    use clap::{CommandFactory as _, FromArgMatches as _};

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
    fn timeout_defaults_to_ten_minutes() {
        let cli = Cli::try_parse_from(["git-scribe"]).unwrap();

        assert_eq!(cli.timeout, Duration::from_mins(10));
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

    #[test]
    fn local_git_config_supplies_defaults_but_the_command_line_wins() -> anyhow::Result<()> {
        let path = std::env::temp_dir().join(format!(
            "git-scribe-cli-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let repository = git2::Repository::init(&path)?;
        let mut config = repository.config()?;
        config.set_str("git-scribe.think", "high")?;
        config.set_bool("git-scribe.showThinking", true)?;
        config.set_str("git-scribe.temperature", "0.25")?;
        config.set_i64("git-scribe.seed", 42)?;
        config.set_str("git-scribe.keepAlive", "1h")?;
        config.set_str("git-scribe.timeout", "30s")?;
        config.set_multivar("git-scribe.excludeDiff", "^$", "generated.css")?;
        config.set_multivar("git-scribe.excludeDiff", "^$", "vendor/app.js")?;
        drop(config);
        drop(repository);

        let repo = GitRepo::discover(&path)?;
        let matches = Cli::command().try_get_matches_from(["git-scribe"])?;
        let mut cli = Cli::from_arg_matches(&matches)?;
        cli.apply_local_config(&matches, &repo)?;

        assert_eq!(cli.think, Think::High);
        assert!(cli.show_thinking);
        assert!((cli.temperature - 0.25).abs() < f32::EPSILON);
        assert_eq!(cli.seed, Some(42));
        assert_eq!(cli.keep_alive, Some(Duration::from_hours(1)));
        assert_eq!(cli.timeout, Duration::from_secs(30));
        assert_eq!(
            cli.exclude_diff,
            [
                PathBuf::from("generated.css"),
                PathBuf::from("vendor/app.js")
            ]
        );

        let matches = Cli::command().try_get_matches_from([
            "git-scribe",
            "--think=low",
            "--temperature",
            "0.5",
            "--timeout",
            "1m",
            "--exclude-diff",
            "manual.css",
            "--exclude-diff",
            "generated/types.ts",
        ])?;
        let mut cli = Cli::from_arg_matches(&matches)?;
        cli.apply_local_config(&matches, &repo)?;
        assert_eq!(cli.think, Think::Low);
        assert!((cli.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(cli.timeout, Duration::from_mins(1));
        assert_eq!(
            cli.exclude_diff,
            [
                PathBuf::from("generated.css"),
                PathBuf::from("vendor/app.js"),
                PathBuf::from("manual.css"),
                PathBuf::from("generated/types.ts"),
            ]
        );

        drop(repo);
        std::fs::remove_dir_all(path)?;
        Ok(())
    }
}
