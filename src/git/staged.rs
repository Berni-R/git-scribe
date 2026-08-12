use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};

/// A single path-level change staged for the next commit.
///
/// Paths are relative to the repository root.
///
/// For ordinary additions, modifications, deletions, and type changes,
/// [`path`](Self::path) identifies the affected path directly.
/// Renames and copies additionally store the source path in [`StagedChangeKind`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagedChange {
    /// The primary path associated with the staged change.
    ///
    /// For additions and modifications, this is the affected path; for deletions, this is the deleted path.
    ///
    /// For renames and copies, this is the destination path;
    /// the source path is stored in [`StagedChangeKind::Renamed`] or [`StagedChangeKind::Copied`].
    pub path: PathBuf,

    /// The kind of change staged for this path.
    pub kind: StagedChangeKind,
}

/// Describes how a staged path differs from the current `HEAD`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagedChangeKind {
    /// A new file has been added.
    Added,

    /// An existing file has been modified.
    Modified,

    /// An existing file has been deleted.
    Deleted,

    /// A file has been renamed.
    ///
    /// `from` is the original path and `similarity` is Git's similarity percentage in the range `0..=100`.
    Renamed { from: PathBuf, similarity: u8 },

    /// The type of an entry has changed.
    ///
    /// For example, a regular file may have become a symbolic link.
    TypeChanged,

    /// The path is in an unmerged state.
    ///
    /// This normally indicates unresolved merge conflicts in the index.
    Unmerged,
}

impl StagedChange {
    /// Parses NUL-delimited `git diff --name-status` output.
    ///
    /// This expects output produced by a command equivalent to:
    ///
    /// ```sh
    /// git diff --cached --name-status -z --find-renames --find-copies
    /// ```
    ///
    /// Rename and copy records contain two paths: the source path followed by the destination path.
    /// Other supported status records contain one path.
    ///
    /// This parser assumes paths are valid UTF-8.
    pub(super) fn parse(output: &[u8]) -> Result<Vec<StagedChange>> {
        if output.is_empty() {
            return Ok(Vec::new());
        }
        if !output.ends_with(b"\0") {
            bail!("Git change status output is not NUL-terminated");
        }

        let mut fields = output[..output.len() - 1].split(|byte| *byte == b'\0');
        let mut changes = Vec::new();
        while let Some(status) = fields.next() {
            let status =
                std::str::from_utf8(status).context("Git change status is not valid UTF-8")?;

            let kind = status
                .as_bytes()
                .first()
                .copied()
                .context("Git returned an empty change status")?;

            match kind {
                b'A' | b'M' | b'D' | b'T' | b'U' => {
                    let path = Self::next_path(&mut fields)?;
                    let kind = match kind {
                        b'A' => StagedChangeKind::Added,
                        b'M' => StagedChangeKind::Modified,
                        b'D' => StagedChangeKind::Deleted,
                        b'T' => StagedChangeKind::TypeChanged,
                        b'U' => StagedChangeKind::Unmerged,
                        _ => unreachable!(),
                    };
                    changes.push(StagedChange { path, kind });
                }

                b'R' => {
                    let similarity = status[1..].parse::<u8>().with_context(|| {
                        format!("Git returned invalid similarity score: {status:?}")
                    })?;
                    let from = Self::next_path(&mut fields)?;
                    let path = Self::next_path(&mut fields)?;
                    let kind = StagedChangeKind::Renamed { from, similarity };
                    changes.push(StagedChange { path, kind });
                }

                _ => {
                    bail!("Git returned unsupported change status: {status:?}");
                }
            }
        }

        Ok(changes)
    }

    /// Reads and decodes the next pathname from NUL-delimited Git output.
    fn next_path<'a>(fields: &mut impl Iterator<Item = &'a [u8]>) -> Result<PathBuf> {
        let path = fields
            .next()
            .context("Git change status is missing a path")?;

        let path = std::str::from_utf8(path).context("Git path is not valid UTF-8")?;

        Ok(PathBuf::from(path))
    }
}
