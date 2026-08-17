use std::{fmt, path::PathBuf};

use git2::{FileMode, Oid};

use super::DiffHunk;

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
    Added {
        after: FileVersion,
    },
    Modified {
        before: FileVersion,
        after: FileVersion,
    },
    Deleted {
        before: FileVersion,
    },
    Renamed {
        before: FileVersion,
        after: FileVersion,
    },
    TypeChanged {
        before: FileVersion,
        after: FileVersion,
    },
    Unmerged {
        path: PathBuf,
    },
}

/// An owned, internally consistent file change in a prospective commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitChange {
    pub kind: CommitChangeKind,
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
    pub files_changed: usize,
    pub insertions: usize,
    pub deletions: usize,
}

impl fmt::Display for CommitStats {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = if self.files_changed == 1 {
            "file"
        } else {
            "files"
        };
        let insertion = if self.insertions == 1 {
            "insertion"
        } else {
            "insertions"
        };
        let deletion = if self.deletions == 1 {
            "deletion"
        } else {
            "deletions"
        };

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
    pub(super) mode: super::CommitMode,
    pub(super) changes: Vec<CommitChange>,
    pub(super) patch: Vec<u8>,
    pub(super) stats: CommitStats,
}

impl ProspectiveCommit {
    #[must_use]
    pub fn mode(&self) -> super::CommitMode {
        self.mode
    }

    #[must_use]
    pub fn changes(&self) -> &[CommitChange] {
        &self.changes
    }

    #[must_use]
    pub fn patch(&self) -> &[u8] {
        &self.patch
    }

    #[must_use]
    pub fn stats(&self) -> CommitStats {
        self.stats
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.changes.len()
    }

    pub(super) fn into_patch(self) -> Vec<u8> {
        // TODO: implement `Into` trait?!
        self.patch
    }
}
