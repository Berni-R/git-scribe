use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::{Context as _, Result, anyhow, bail};

/// A Git working-tree repository.
///
/// `GitRepo` stores the absolute path to the repository's top-level working directory.
/// It can be discovered from any directory inside that working tree.
///
/// This type currently represents repositories with a working tree;
/// discovery of bare repositories is not supported because `git rev-parse --show-toplevel` requires a working tree.
#[derive(Debug, Clone)]
pub struct GitRepo {
    root: PathBuf,
}

/// Selects how the prospective commit is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitMode {
    /// Create a new commit from the changes between `HEAD` and the current index.
    Normal,

    /// Amend `HEAD`, using the changes between `HEAD^` and the current index.
    Amend,
}

impl CommitMode {
    /// The base commit to use: `"HEAD"` for [`CommitMode::Normal`] and `"HEAD^` for [`CommitMode::Amend`].
    #[must_use]
    pub(super) fn base(self) -> &'static str {
        match self {
            Self::Normal => "HEAD",
            Self::Amend => "HEAD^", // TODO: `--amend` is valid for `git` if the tree is empty, but this then fails
        }
    }
}

impl GitRepo {
    /// Returns the absolute path to the repository's working-tree root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Discovers the Git repository containing `directory`.
    ///
    /// `directory` may be either the repository root or any directory below it.
    ///
    /// Returns an error if `directory` is not inside a Git working tree, Git cannot be executed,
    /// or the returned repository path is not valid UTF-8.
    pub fn discover(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();

        let stdout = run_git(directory, &["rev-parse", "--show-toplevel"], None)?;

        let root = String::from_utf8(stdout).context("Git repository path is not valid UTF-8")?;
        let root = root.trim_end_matches(['\r', '\n']);
        if root.is_empty() {
            bail!(
                "Git returned an empty repository root for {}",
                directory.display()
            );
        }

        Ok(Self {
            root: PathBuf::from(root),
        })
    }

    /// Executes Git in this repository without requiring a successful exit.
    ///
    /// Use this when a command's exit status itself carries information.
    pub(super) fn execute(&self, args: &[&str]) -> Result<Output> {
        execute_git(&self.root, args, None)
    }

    /// Executes Git in this repository and returns stdout.
    ///
    /// A non-zero Git exit status is treated as an error.
    pub(super) fn run(&self, args: &[&str], path: Option<&Path>) -> Result<Vec<u8>> {
        run_git(&self.root, args, path)
    }

    /// Executes Git and decodes stdout as UTF-8.
    pub(super) fn text(&self, args: &[&str], path: Option<&Path>) -> Result<String> {
        let stdout = self.run(args, path)?;

        String::from_utf8(stdout).context("Git output is not valid UTF-8")
    }
}

/// Executes Git and returns its complete process output.
///
/// Unlike [`run_git`], a non-zero Git exit status is not considered an error.
/// This is useful for commands where particular exit codes represent ordinary
/// states rather than failures.
fn execute_git<S>(directory: &Path, args: &[S], path: Option<&Path>) -> Result<Output>
where
    S: AsRef<OsStr>,
{
    let mut command = Command::new("git");

    command.arg("-C").arg(&directory).args(args);
    if let Some(path) = path {
        command.arg("--").arg(path);
    }

    command.output().with_context(|| {
        format!(
            "failed to execute Git in {:?} \
                 (is Git installed and available on PATH?)",
            directory
        )
    })
}

/// Executes Git and returns stdout if the command succeeds.
fn run_git<S>(directory: &Path, args: &[S], path: Option<&Path>) -> Result<Vec<u8>>
where
    S: AsRef<OsStr>,
{
    let output = execute_git(directory, args, path)?;

    if !output.status.success() {
        return Err(git_command_error(directory, args, path, &output));
    }

    Ok(output.stdout)
}

/// Constructs a diagnostic error for an unsuccessful Git command.
pub(super) fn git_command_error<S>(
    directory: &Path,
    args: &[S],
    path: Option<&Path>,
    output: &Output,
) -> anyhow::Error
where
    S: AsRef<OsStr>,
{
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim_end();

    let args = args.iter().map(|arg| format!("{:?}", arg.as_ref()));
    let args: Vec<String> = if let Some(path) = path {
        args.chain(["--".to_string(), path.to_string_lossy().into_owned()])
            .collect()
    } else {
        args.collect()
    };
    let args = args.join(" ");
    // TODO: earlier we had a debug print of a list – here individual argument should pontetially be put in quote (and be escaped?)

    if stderr.is_empty() {
        anyhow!(
            "git {args} failed in {} ({})",
            directory.display(),
            output.status,
        )
    } else {
        anyhow!(
            "git {args} failed in {} ({}): {stderr}",
            directory.display(),
            output.status,
        )
    }
}
