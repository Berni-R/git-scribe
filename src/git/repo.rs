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

impl GitRepo {
    /// Returns the absolute path to the repository's working-tree root.
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

        let stdout = run_git(directory, &["rev-parse", "--show-toplevel"])?;

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
        execute_git(&self.root, args)
    }

    /// Executes Git in this repository and returns stdout.
    ///
    /// A non-zero Git exit status is treated as an error.
    pub(super) fn run(&self, args: &[&str]) -> Result<Vec<u8>> {
        run_git(&self.root, args)
    }

    /// Executes Git and decodes stdout as UTF-8.
    pub(super) fn text(&self, args: &[&str]) -> Result<String> {
        let stdout = self.run(args)?;

        String::from_utf8(stdout).context("Git output is not valid UTF-8")
    }
}

/// Executes Git and returns its complete process output.
///
/// Unlike [`run_git`], a non-zero Git exit status is not considered an error.
/// This is useful for commands where particular exit codes represent ordinary
/// states rather than failures.
fn execute_git(directory: &Path, args: &[&str]) -> Result<Output> {
    Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(args)
        .output()
        .with_context(|| {
            format!(
                "failed to execute Git in {} \
                 (is Git installed and available on PATH?)",
                directory.display()
            )
        })
}

/// Executes Git and returns stdout if the command succeeds.
fn run_git(directory: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = execute_git(directory, args)?;

    if !output.status.success() {
        return Err(git_command_error(directory, args, &output));
    }

    Ok(output.stdout)
}

/// Constructs a diagnostic error for an unsuccessful Git command.
pub(super) fn git_command_error(directory: &Path, args: &[&str], output: &Output) -> anyhow::Error {
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr = stderr.trim_end();

    if stderr.is_empty() {
        anyhow!(
            "git {args:?} failed in {} ({})",
            directory.display(),
            output.status,
        )
    } else {
        anyhow!(
            "git {args:?} failed in {} ({}): {stderr}",
            directory.display(),
            output.status,
        )
    }
}
