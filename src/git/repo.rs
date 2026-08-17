use std::{
    fmt,
    path::{Path, PathBuf},
};

use anyhow::{Context as _, Result, bail};
use git2::{Commit, ErrorCode, Repository, Tree};

/// A Git working-tree repository.
///
/// The repository remains open through libgit2 and records its canonical, absolute working-tree root.
/// Bare repositories are not supported.
pub struct GitRepo {
    repository: Repository,
    root: PathBuf,
}

impl fmt::Debug for GitRepo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GitRepo")
            .field("root", &self.root)
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
        let workdir = repository.workdir().with_context(|| {
            format!("Git repository at {} is bare", repository.path().display())
        })?;
        let root = workdir.canonicalize().with_context(|| {
            format!(
                "failed to resolve Git working tree at {}",
                workdir.display()
            )
        })?;

        Ok(Self { repository, root })
    }

    /// Return the canonical, absolute path to the working-tree root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the underlying libgit2 repository for use within the Git abstraction.
    pub(super) fn repository(&self) -> &Repository {
        &self.repository
    }

    /// Resolve HEAD to a commit, or return None for an unborn branch.
    pub(super) fn head_commit(&self) -> Result<Option<Commit<'_>>> {
        let head = match self.repository.head() {
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
