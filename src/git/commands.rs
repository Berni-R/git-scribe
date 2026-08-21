use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context as _, Result, bail};
use git2::{
    Delta, Diff, DiffDelta, DiffFile, DiffFindOptions, DiffFormat, DiffOptions, Oid, Patch, Sort,
    StatusOptions,
};

use crate::git::{
    CommitChange, CommitChangeKind, CommitMode, CommitStats, DiffHunk, FileVersion, GitRepo,
    LineRange, ProspectiveCommit,
};

impl GitRepo {
    /// Return the object ID referenced by HEAD, or None for an unborn repository.
    pub fn head_sha(&self) -> Result<Option<Oid>> {
        Ok(self.head_commit()?.map(|commit| commit.id()))
    }

    /// Return the checked-out local branch, or None for detached HEAD.
    ///
    /// Symbolic references outside refs/heads are not reported as local branches.
    pub fn current_branch(&self) -> Result<Option<String>> {
        let head = self
            .repository()
            .find_reference("HEAD")
            .context("failed to read Git HEAD")?;
        let Some(target) = head.symbolic_target_bytes() else {
            return Ok(None);
        };
        let Some(branch) = target.strip_prefix(b"refs/heads/") else {
            return Ok(None);
        };
        let branch = std::str::from_utf8(branch).context("Git branch name is not valid UTF-8")?;

        Ok(Some(branch.to_owned()))
    }

    /// Return whether the index or working tree contains tracked or untracked changes.
    ///
    /// Ignored files are excluded. Untracked directories use normal, non-recursive status behavior.
    pub fn is_dirty(&self) -> Result<bool> {
        let mut options = StatusOptions::new();
        options
            .include_untracked(true)
            .include_ignored(false)
            .recurse_untracked_dirs(false);

        let statuses = self
            .repository()
            .statuses(Some(&mut options))
            .context("failed to read Git status")?;
        Ok(!statuses.is_empty())
    }

    /// Build all views of the commit currently represented by the index from one tree-to-index diff.
    pub fn prospective_commit(&self, mode: CommitMode) -> Result<ProspectiveCommit> {
        let diff = self.prospective_diff(mode)?;
        let changes = commit_changes(&diff)?;
        let raw_stats = diff
            .stats()
            .context("failed to calculate prospective commit statistics")?;
        let stats = CommitStats {
            files_changed: raw_stats.files_changed(),
            insertions: raw_stats.insertions(),
            deletions: raw_stats.deletions(),
        };
        let patch = render_patch(&diff)?;

        Ok(ProspectiveCommit {
            mode,
            changes,
            patch,
            stats,
        })
    }

    /// Return the complete raw patch for the prospective commit.
    ///
    /// The patch is rendered from the same rename-aware, zero-context diff used for structured changes.
    pub fn commit_diff(&self, mode: CommitMode) -> Result<Vec<u8>> {
        Ok(self.prospective_commit(mode)?.into_patch())
    }

    /// Open Git's normal commit editor with `message` as its initial contents.
    ///
    /// The Git command remains responsible for editing, hooks, signing, and creating the commit.
    pub fn commit_interactively(&self, mode: CommitMode, message: &str) -> Result<()> {
        let mut command = Command::new("git");
        command.current_dir(self.workdir()?).arg("commit");
        if mode == CommitMode::Amend {
            command.arg("--amend");
        }
        command.args(["--edit", "--message", message]);

        let status = command.status().context("failed to run git commit")?;
        if !status.success() {
            bail!("git commit failed with {status}");
        }

        Ok(())
    }

    /// Read a blob by object ID.
    pub fn blob(&self, oid: Oid) -> Result<Vec<u8>> {
        let blob = self
            .repository()
            .find_blob(oid)
            .with_context(|| format!("failed to read Git blob {oid}"))?;
        Ok(blob.content().to_vec())
    }

    /// Return up to limit recent commit subjects, newest first.
    pub fn recent_commit_subjects(&self, limit: usize) -> Result<Vec<String>> {
        if limit == 0 || self.head_commit()?.is_none() {
            return Ok(Vec::new());
        }

        let mut walk = self
            .repository()
            .revwalk()
            .context("failed to create Git revision walk")?;
        walk.set_sorting(Sort::TOPOLOGICAL | Sort::TIME)
            .context("failed to configure Git revision walk")?;
        walk.push_head().context("failed to walk from Git HEAD")?;

        walk.take(limit)
            .map(|id| {
                let id = id.context("failed to walk Git commit history")?;
                let commit = self
                    .repository()
                    .find_commit(id)
                    .with_context(|| format!("failed to read Git commit {id}"))?;
                Ok(commit.summary()?.unwrap_or_default().to_owned())
            })
            .collect()
    }

    /// Return the contents of a file from the current index.
    ///
    /// None is returned when the path has no stage-zero index entry.
    pub fn index_file(&self, path: impl AsRef<Path>) -> Result<Option<Vec<u8>>> {
        let index = self
            .repository()
            .index()
            .context("failed to read Git index")?;
        let Some(entry) = index.get_path(path.as_ref(), 0) else {
            return Ok(None);
        };

        self.blob(entry.id)
            .map(Some)
            .with_context(|| format!("failed to read indexed file {}", path.as_ref().display()))
    }

    /// Return the files visible in the working tree after applying Git ignore rules.
    ///
    /// Git metadata is excluded and symbolic links are returned as files rather than followed.
    pub fn working_tree_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.collect_working_tree_files(Path::new(""), &mut files)?;
        files.sort();
        Ok(files)
    }

    /// Recursively collect non-ignored working-tree files.
    fn collect_working_tree_files(
        &self,
        relative_directory: &Path,
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        let directory = self.workdir()?.join(relative_directory);
        let mut entries = fs::read_dir(&directory)
            .with_context(|| {
                format!(
                    "failed to read working-tree directory {}",
                    directory.display()
                )
            })?
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(fs::DirEntry::file_name);

        for entry in entries {
            let name = entry.file_name();
            if name == OsStr::new(".git") {
                continue;
            }

            let relative_path = relative_directory.join(name);
            if self
                .repository()
                .status_should_ignore(&relative_path)
                .with_context(|| {
                    format!(
                        "failed to apply Git ignore rules to {}",
                        relative_path.display()
                    )
                })?
            {
                continue;
            }

            let file_type = entry.file_type().with_context(|| {
                format!(
                    "failed to inspect working-tree entry {}",
                    relative_path.display()
                )
            })?;
            if file_type.is_dir() {
                self.collect_working_tree_files(&relative_path, files)?;
            } else {
                files.push(relative_path);
            }
        }

        Ok(())
    }

    /// Build the rename-aware tree-to-index diff for a commit mode.
    fn prospective_diff(&self, mode: CommitMode) -> Result<Diff<'_>> {
        let base = self.base_tree(mode)?;
        let index = self
            .repository()
            .index()
            .context("failed to read Git index")?;
        let mut options = DiffOptions::new();
        options
            .context_lines(0)
            .include_typechange(true)
            .include_typechange_trees(true)
            .indent_heuristic(true)
            .old_prefix("a/")
            .new_prefix("b/");

        let mut diff = self
            .repository()
            .diff_tree_to_index(base.as_ref(), Some(&index), Some(&mut options))
            .context("failed to compare the prospective commit base with the Git index")?;
        let mut find = DiffFindOptions::new();
        find.renames(true);
        diff.find_similar(Some(&mut find))
            .context("failed to detect prospective commit renames")?;

        Ok(diff)
    }
}

/// Convert all diff deltas into owned commit changes.
fn commit_changes(diff: &Diff<'_>) -> Result<Vec<CommitChange>> {
    diff.deltas()
        .enumerate()
        .map(|(index, delta)| commit_change(diff, index, &delta))
        .collect()
}

/// Convert one diff delta into an owned commit change.
fn commit_change(diff: &Diff<'_>, index: usize, delta: &DiffDelta<'_>) -> Result<CommitChange> {
    let status = delta.status();
    let kind = match status {
        Delta::Added => CommitChangeKind::Added {
            after: file_version(&delta.new_file(), "added file")?,
        },
        Delta::Modified => CommitChangeKind::Modified {
            before: file_version(&delta.old_file(), "modified file before version")?,
            after: file_version(&delta.new_file(), "modified file after version")?,
        },
        Delta::Deleted => CommitChangeKind::Deleted {
            before: file_version(&delta.old_file(), "deleted file")?,
        },
        Delta::Renamed => CommitChangeKind::Renamed {
            before: file_version(&delta.old_file(), "renamed file before version")?,
            after: file_version(&delta.new_file(), "renamed file after version")?,
        },
        Delta::Typechange => CommitChangeKind::TypeChanged {
            before: file_version(&delta.old_file(), "type-changed file before version")?,
            after: file_version(&delta.new_file(), "type-changed file after version")?,
        },
        Delta::Conflicted => CommitChangeKind::Unmerged {
            path: delta
                .new_file()
                .path()
                .or_else(|| delta.old_file().path())
                .context("libgit2 omitted the path for an unmerged change")?
                .to_path_buf(),
        },
        status @ (Delta::Unmodified
        | Delta::Copied
        | Delta::Ignored
        | Delta::Untracked
        | Delta::Unreadable) => {
            bail!("libgit2 returned unsupported prospective commit status: {status:?}")
        }
    };

    let hunks = if status == Delta::Conflicted {
        Vec::new()
    } else {
        patch_hunks(diff, index)?
    };
    Ok(CommitChange { kind, hunks })
}

/// Convert a libgit2 file entry into a validated file version.
fn file_version(file: &DiffFile<'_>, description: &str) -> Result<FileVersion> {
    if !file.exists() {
        bail!("libgit2 reported a missing {description}");
    }
    let path = file
        .path()
        .with_context(|| format!("libgit2 omitted the path for {description}"))?
        .to_path_buf();
    let oid = file.id();
    if oid.is_zero() {
        bail!(
            "libgit2 omitted the object ID for {description} at {}",
            path.display()
        );
    }

    Ok(FileVersion {
        path,
        oid,
        mode: file.mode(),
    })
}

/// Extract line ranges from one diff patch.
fn patch_hunks(diff: &Diff<'_>, index: usize) -> Result<Vec<DiffHunk>> {
    let Some(patch) =
        Patch::from_diff(diff, index).context("failed to read a prospective commit patch")?
    else {
        return Ok(Vec::new());
    };

    (0..patch.num_hunks())
        .map(|index| {
            let (hunk, _) = patch
                .hunk(index)
                .context("failed to read a prospective commit hunk")?;
            Ok(DiffHunk {
                before: line_range(hunk.old_start(), hunk.old_lines()),
                after: line_range(hunk.new_start(), hunk.new_lines()),
            })
        })
        .collect()
}

/// Convert a non-empty libgit2 line range to the public representation.
fn line_range(start: u32, count: u32) -> Option<LineRange> {
    (count != 0).then(|| LineRange {
        start: usize::try_from(start).expect("u32 fits in usize"),
        count: usize::try_from(count).expect("u32 fits in usize"),
    })
}

/// Render a diff as a zero-context patch.
fn render_patch(diff: &Diff<'_>) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    diff.print(DiffFormat::Patch, |_, _, line| {
        match line.origin() {
            ' ' => output.push(b' '),
            '+' => output.push(b'+'),
            '-' => output.push(b'-'),
            _ => {}
        }
        output.extend_from_slice(line.content());
        true
    })
    .context("failed to render the prospective commit patch")?;
    Ok(output)
}
