use std::{fmt, path::Path};

use anyhow::{Context as _, Result, bail};
use git2::{Commit, Config, ErrorCode, Repository, Tree};

/// A non-bare Git repository backed by `libgit2`.
pub struct GitRepo(Repository);

impl fmt::Debug for GitRepo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GitRepo")
            .field("path", &self.0.path())
            .finish_non_exhaustive()
    }
}

/// Selects how the prospective commit is constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitMode {
    /// Compare the current index with the tree at HEAD.
    ///
    /// An unborn repository uses an empty base tree.
    Normal,

    /// Compare the current index with the first parent's tree of HEAD.
    ///
    /// A root commit uses an empty base tree. Merge commits are rejected.
    Amend,
}

impl GitRepo {
    /// Discover the Git working-tree repository containing directory.
    pub fn discover(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let repository = Repository::discover(directory).with_context(|| {
            format!(
                "failed to discover a Git repository from {}",
                directory.display()
            )
        })?;
        repository.workdir().with_context(|| {
            format!("Git repository at {} is bare", repository.path().display())
        })?;

        Ok(Self(repository))
    }

    /// Open this repository's local Git configuration.
    ///
    /// This deliberately excludes system and global Git configuration: git-scribe's
    /// settings are scoped to the repository in which they are configured.
    pub fn local_config(&self) -> Result<Config> {
        let path = self.0.path().join("config");
        Config::open(&path).with_context(|| {
            format!(
                "failed to open repository configuration at {}",
                path.display()
            )
        })
    }

    /// Return the underlying `libgit2` repository for use within the Git abstraction.
    #[must_use]
    pub(super) fn repository(&self) -> &Repository {
        &self.0
    }

    /// Return the repository working tree, rejecting bare repositories.
    pub(super) fn workdir(&self) -> Result<&Path> {
        self.0
            .workdir()
            .with_context(|| format!("Git repository at {} is bare", self.0.path().display()))
    }

    /// Resolve HEAD to a commit, or return None for an unborn branch.
    pub(super) fn head_commit(&self) -> Result<Option<Commit<'_>>> {
        let head = match self.0.head() {
            Ok(head) => head,
            Err(error) if error.code() == ErrorCode::UnbornBranch => return Ok(None),
            Err(error) => return Err(error).context("failed to read Git HEAD"),
        };

        head.peel_to_commit()
            .map(Some)
            .context("Git HEAD does not reference a commit")
    }

    /// Resolve the application commit mode to the tree that precedes the prospective commit.
    ///
    /// None deliberately represents the empty tree accepted by
    /// `Repository::diff_tree_to_index`.
    pub(super) fn base_tree(&self, mode: CommitMode) -> Result<Option<Tree<'_>>> {
        let Some(head) = self.head_commit()? else {
            return match mode {
                CommitMode::Normal => Ok(None),
                CommitMode::Amend => {
                    bail!("cannot amend because the repository has no existing commit")
                }
            };
        };

        match mode {
            CommitMode::Normal => head
                .tree()
                .map(Some)
                .context("failed to read the Git HEAD tree"),
            CommitMode::Amend => match head.parent_count() {
                0 => Ok(None),
                1 => head
                    .parent(0)
                    .context("failed to read the parent of Git HEAD")?
                    .tree()
                    .map(Some)
                    .context("failed to read the parent tree of Git HEAD"),
                count => bail!(
                    "cannot amend merge commit {} with {count} parents; merge amendments are not supported",
                    head.id()
                ),
            },
        }
    }
}
