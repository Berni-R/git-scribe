use anyhow::{Context as _, Result};

use crate::git::{CommitMode, GitRepo, repo::git_command_error, staged::StagedChange};

impl GitRepo {
    /// Returns the commit ID currently referenced by `HEAD`.
    ///
    /// Returns `None` when the repository has no commits yet,
    /// such as a freshly initialized repository with an unborn branch.
    ///
    /// Returns an error if Git cannot be executed, `HEAD` cannot be inspected, or Git returns unexpected output.
    pub fn head_sha(&self) -> Result<Option<String>> {
        let args = [
            "rev-parse",
            "--verify",
            "--quiet",
            "--path-format=absolute",
            "HEAD^{commit}",
        ];
        let output = self.execute(&args)?;
        if output.status.success() {
            let sha =
                String::from_utf8(output.stdout).context("Git HEAD SHA is not valid UTF-8")?;

            return Ok(Some(sha.trim_end_matches(['\r', '\n']).to_owned()));
        }

        // An unborn branch still has a symbolic HEAD, but that symbolic ref does not resolve to a commit yet.
        let head = self.execute(&["symbolic-ref", "--quiet", "HEAD"])?;
        if head.status.success() {
            return Ok(None);
        }
        Err(git_command_error(self.root(), &args, &output))
    }

    /// Returns the name of the currently checked-out branch.
    ///
    /// Returns `None` when the repository is in detached-HEAD state.
    ///
    /// An unborn repository may still return a branch name because `HEAD` already points to that branch
    /// even though the branch has no commits.
    pub fn current_branch(&self) -> Result<Option<String>> {
        let args = ["symbolic-ref", "--quiet", "--short", "HEAD"];
        let output = self.execute(&args)?;
        match output.status.code() {
            Some(0) => {
                let branch = String::from_utf8(output.stdout)
                    .context("Git branch name is not valid UTF-8")?;
                Ok(Some(branch.trim_end_matches(['\r', '\n']).to_owned()))
            }

            // `git symbolic-ref --quiet` returns exit status 1 when the requested ref is not symbolic,
            // which is the normal detached HEAD case.
            Some(1) => Ok(None),

            _ => Err(git_command_error(self.root(), &args, &output)),
        }
    }

    /// Returns whether the repository has staged, unstaged, or untracked changes.
    ///
    /// Ignored files are not considered dirty.
    ///
    /// This explicitly enables normal untracked-file reporting so that the result is not affected by the user's
    /// `status.showUntrackedFiles` configuration.
    pub fn is_dirty(&self) -> Result<bool> {
        let output = self.run(&["status", "--porcelain=v1", "--untracked-files=normal", "-z"])?;
        Ok(!output.is_empty())
    }

    /// Returns structured information about changes staged for the next commit.
    ///
    /// Rename detection is enabled explicitly so that related delete/add operations can be represented as renames
    /// when Git detects sufficient similarity.
    ///
    /// Copy detection is intentionally not enabled. Detecting copies from unchanged files requires Git's more
    /// expensive `--find-copies-harder` mode, while copy information is not currently important enough to justify that
    /// additional work.
    pub fn commit_changes(&self, mode: CommitMode) -> Result<Vec<StagedChange>> {
        let output = self.run(&[
            "diff",
            "--cached",
            "--name-status",
            "-z",
            "--find-renames",
            mode.base(),
        ])?;

        StagedChange::parse(&output)
    }

    /// Returns the complete patch that would be represented by the next commit.
    ///
    /// In amend mode, the patch includes both the current HEAD commit and any additionally staged changes.
    ///
    /// The result is returned as raw bytes because arbitrary file contents are not guaranteed to be valid UTF-8.
    ///
    /// External diff programs and text-conversion filters are disabled to keep the output deterministic
    /// and suitable for programmatic use.
    pub fn commit_diff(&self, mode: CommitMode) -> Result<Vec<u8>> {
        self.run(&[
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--default-prefix",
            "--unified=3",
            "--find-renames",
            mode.base(),
        ])
    }

    /// Returns a compact summary of the changes that would be represented by the next commit.
    ///
    /// In amend mode, the summary includes both the current HEAD commit and any
    /// additionally staged changes.
    ///
    /// The summary shows the relative size of each changed file and aggregate insertion/deletion counts.
    /// Output is bounded because the complete file list is provided separately as structured staged-change context.
    pub fn commit_diff_stat(&self, mode: CommitMode) -> Result<String> {
        self.text(&[
            "diff",
            "--cached",
            "--stat=100,70,30",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--find-renames",
            mode.base(),
        ])
    }

    /// Returns up to `limit` recent commit subjects, newest first.
    ///
    /// Only the subject line of each commit message is returned.
    ///
    /// Returns an empty vector when `limit` is zero or when the repository does not have any commits yet.
    pub fn recent_commit_subjects(&self, limit: usize) -> Result<Vec<String>> {
        if limit == 0 || self.head_sha()?.is_none() {
            return Ok(Vec::new());
        }
        let limit = limit.to_string();
        let output = self.text(&["log", "-n", &limit, "--format=%s", "HEAD"])?;
        Ok(output.lines().map(str::to_owned).collect())
    }

    /// Returns the contents of a file from the Git index.
    ///
    /// Returns `None` when `path` does not exist in the index.
    pub fn index_file(&self, path: &str) -> Result<Option<Vec<u8>>> {
        let listed = self.run(&["ls-files", "--cached", "-z", "--", path])?;
        if listed.is_empty() {
            return Ok(None);
        }

        let spec = format!(":{path}");
        Ok(Some(self.run(&["show", &spec])?))
    }
}
