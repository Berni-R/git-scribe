use std::{fmt, path::PathBuf};

use git2::{FileMode, Oid};

use super::DiffHunk;

/// Select a singular label or append `s` for its plural form.
macro_rules! plural {
    ($count:expr, $word:literal) => {
        if $count == 1 {
            $word
        } else {
            concat!($word, "s")
        }
    };
}

/// One side of a file change in a prospective commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileVersion {
    /// Path relative to the repository root.
    pub path: PathBuf,

    /// ID of the Git object stored at this path.
    pub oid: Oid,

    /// Git file mode for the object.
    pub mode: FileMode,
}

impl FileVersion {
    /// Whether this version contains ordinary source-file contents suitable for syntax analysis.
    #[must_use]
    pub fn is_blob(&self) -> bool {
        matches!(
            self.mode,
            FileMode::Blob | FileMode::BlobExecutable | FileMode::BlobGroupWritable
        )
    }
}

/// The path and object versions involved in a prospective commit change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitChangeKind {
    /// A newly added file.
    Added {
        /// The file in the prospective commit.
        after: FileVersion,
    },
    /// A modified file.
    Modified {
        /// The file before the change.
        before: FileVersion,
        /// The file after the change.
        after: FileVersion,
    },
    /// A deleted file.
    Deleted {
        /// The file before deletion.
        before: FileVersion,
    },
    /// A renamed file, with or without edits.
    Renamed {
        /// The original file.
        before: FileVersion,
        /// The renamed file.
        after: FileVersion,
    },
    /// A file whose Git type changed.
    TypeChanged {
        /// The file before the type change.
        before: FileVersion,
        /// The file after the type change.
        after: FileVersion,
    },
    /// An unresolved merge conflict.
    Unmerged {
        /// The conflicted path.
        path: PathBuf,
    },
}

/// An owned, internally consistent file change in a prospective commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitChange {
    /// Kind and file versions for the change.
    pub kind: CommitChangeKind,

    /// Changed line ranges in the file.
    pub hunks: Vec<DiffHunk>,
}

impl CommitChange {
    /// The version before the prospective commit, when one exists.
    #[must_use]
    pub fn before(&self) -> Option<&FileVersion> {
        match &self.kind {
            CommitChangeKind::Modified { before, .. }
            | CommitChangeKind::Deleted { before }
            | CommitChangeKind::Renamed { before, .. }
            | CommitChangeKind::TypeChanged { before, .. } => Some(before),
            CommitChangeKind::Added { .. } | CommitChangeKind::Unmerged { .. } => None,
        }
    }

    /// The version after the prospective commit, when one exists.
    #[must_use]
    pub fn after(&self) -> Option<&FileVersion> {
        match &self.kind {
            CommitChangeKind::Added { after }
            | CommitChangeKind::Modified { after, .. }
            | CommitChangeKind::Renamed { after, .. }
            | CommitChangeKind::TypeChanged { after, .. } => Some(after),
            CommitChangeKind::Deleted { .. } | CommitChangeKind::Unmerged { .. } => None,
        }
    }

    /// A concise status-style description of the change.
    #[must_use]
    pub fn summary_line(&self) -> String {
        match &self.kind {
            CommitChangeKind::Added { after } => format!("A\t{}", after.path.display()),
            CommitChangeKind::Modified { after, .. } => format!("M\t{}", after.path.display()),
            CommitChangeKind::Deleted { before } => format!("D\t{}", before.path.display()),
            CommitChangeKind::Renamed { before, after } => {
                format!("R\t{} -> {}", before.path.display(), after.path.display())
            }
            CommitChangeKind::TypeChanged { before, after } if before.path == after.path => {
                format!("T\t{}", after.path.display())
            }
            CommitChangeKind::TypeChanged { before, after } => {
                format!("T\t{} -> {}", before.path.display(), after.path.display())
            }
            CommitChangeKind::Unmerged { path } => format!("U\t{}", path.display()),
        }
    }
}

/// Aggregate line and file counts for a prospective commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitStats {
    /// Number of changed files.
    pub files_changed: usize,

    /// Number of inserted lines.
    pub insertions: usize,

    /// Number of deleted lines.
    pub deletions: usize,
}

impl fmt::Display for CommitStats {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = plural!(self.files_changed, "file");
        let insertion = plural!(self.insertions, "insertion");
        let deletion = plural!(self.deletions, "deletion");

        write!(
            formatter,
            "{} {file} changed, {} {insertion}(+), {} {deletion}(-)",
            self.files_changed, self.insertions, self.deletions
        )
    }
}

/// All application-facing views of the commit currently represented by the index.
#[derive(Debug, PartialEq, Eq)]
pub struct ProspectiveCommit {
    /// Mode used to construct the commit view.
    pub(super) mode: super::CommitMode,

    /// Structured file changes.
    pub(super) changes: Vec<CommitChange>,

    /// Rendered zero-context patch.
    pub(super) patch: Vec<u8>,

    /// Aggregate change statistics.
    pub(super) stats: CommitStats,
}

impl ProspectiveCommit {
    /// Return the mode used to construct this commit view.
    #[must_use]
    pub fn mode(&self) -> super::CommitMode {
        self.mode
    }

    /// Return the structured file changes.
    #[must_use]
    pub fn changes(&self) -> &[CommitChange] {
        &self.changes
    }

    /// Return the rendered patch.
    #[must_use]
    pub fn patch(&self) -> &[u8] {
        &self.patch
    }

    /// Return aggregate change statistics.
    #[must_use]
    pub fn stats(&self) -> CommitStats {
        self.stats
    }

    /// Whether the prospective commit contains no changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Return the number of changed files.
    #[must_use]
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    /// Consume the commit view and return its rendered patch.
    pub(super) fn into_patch(self) -> Vec<u8> {
        self.patch
    }
}
