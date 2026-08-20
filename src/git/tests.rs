use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context as _, Result};
use git2::{ErrorCode, FileMode, Oid, Repository, Signature};

use super::*;

static NEXT_REPOSITORY: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
    repository: Repository,
}

impl Fixture {
    fn new() -> Result<Self> {
        let sequence = NEXT_REPOSITORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "git-synopsis-test-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        let repository = Repository::init(&path)?;
        repository.set_head("refs/heads/main")?;
        Ok(Self { path, repository })
    }

    fn git_repo(&self) -> Result<GitRepo> {
        GitRepo::discover(&self.path)
    }

    fn write(&self, path: &str, contents: impl AsRef<[u8]>) -> Result<()> {
        let path = self.path.join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, contents)?;
        Ok(())
    }

    fn stage(&self, path: &str) -> Result<()> {
        let mut index = self.repository.index()?;
        index.add_path(Path::new(path))?;
        index.write()?;
        Ok(())
    }

    fn write_and_stage(&self, path: &str, contents: impl AsRef<[u8]>) -> Result<()> {
        self.write(path, contents)?;
        self.stage(path)
    }

    fn configure_cli_commit(&self) -> Result<()> {
        let mut config = self.repository.config()?;
        config.set_str("user.name", "Git Synopsis Tests")?;
        config.set_str("user.email", "tests@example.com")?;
        config.set_str("core.editor", "true")?;
        config.set_bool("commit.gpgsign", false)?;
        Ok(())
    }

    fn delete_and_stage(&self, path: &str) -> Result<()> {
        fs::remove_file(self.path.join(path))?;
        let mut index = self.repository.index()?;
        index.remove_path(Path::new(path))?;
        index.write()?;
        Ok(())
    }

    fn rename_and_stage(&self, from: &str, to: &str) -> Result<()> {
        fs::rename(self.path.join(from), self.path.join(to))?;
        let mut index = self.repository.index()?;
        index.remove_path(Path::new(from))?;
        index.add_path(Path::new(to))?;
        index.write()?;
        Ok(())
    }

    fn commit(&self, message: &str) -> Result<Oid> {
        let parents = match self.repository.head() {
            Ok(head) => vec![head.peel_to_commit()?.id()],
            Err(error) if error.code() == ErrorCode::UnbornBranch => Vec::new(),
            Err(error) => return Err(error).context("failed to read fixture HEAD"),
        };
        self.commit_with_parents(message, &parents, true)
    }

    fn commit_with_parents(
        &self,
        message: &str,
        parent_ids: &[Oid],
        update_head: bool,
    ) -> Result<Oid> {
        let tree_id = self.repository.index()?.write_tree()?;
        let tree = self.repository.find_tree(tree_id)?;
        let signature = Signature::now("Git Synopsis Tests", "tests@example.com")?;
        let parents = parent_ids
            .iter()
            .map(|id| self.repository.find_commit(*id))
            .collect::<Result<Vec<_>, _>>()?;
        let parent_refs = parents.iter().collect::<Vec<_>>();

        Ok(self.repository.commit(
            update_head.then_some("HEAD"),
            &signature,
            &signature,
            message,
            &tree,
            &parent_refs,
        )?)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!("failed to remove test repository: {error}");
        }
    }
}

fn only_change(commit: &ProspectiveCommit) -> &CommitChange {
    assert_eq!(commit.len(), 1);
    &commit.changes()[0]
}

fn numbered_lines(count: usize) -> String {
    (1..=count).fold(String::new(), |mut lines, line| {
        writeln!(lines, "line {line}").expect("writing to a String cannot fail");
        lines
    })
}

#[test]
fn unborn_repository_uses_empty_tree() -> Result<()> {
    let fixture = Fixture::new()?;
    let nested = fixture.path.join("nested");
    fs::create_dir(&nested)?;
    fixture.write_and_stage("new.rs", "fn main() {}\n")?;
    let repo = GitRepo::discover(&nested)?;

    assert_eq!(repo.root(), fixture.path.canonicalize()?);
    assert_eq!(repo.current_branch()?.as_deref(), Some("main"));
    assert_eq!(repo.head_sha()?, None);
    assert!(repo.is_dirty()?);
    assert_eq!(
        repo.index_file("new.rs")?.as_deref(),
        Some(b"fn main() {}\n".as_slice())
    );

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let change = only_change(&commit);
    let CommitChangeKind::Added { after } = &change.kind else {
        panic!("expected an addition, got {:?}", change.kind);
    };
    assert_eq!(after.path, Path::new("new.rs"));
    assert_eq!(repo.blob(after.oid)?, b"fn main() {}\n");
    assert_eq!(
        change.hunks,
        [DiffHunk {
            before: None,
            after: Some(LineRange { start: 1, count: 1 }),
        }]
    );
    assert_eq!(
        commit.stats(),
        CommitStats {
            files_changed: 1,
            insertions: 1,
            deletions: 0,
        }
    );
    assert_eq!(
        commit.stats().to_string(),
        "1 file changed, 1 insertion(+), 0 deletions(-)"
    );
    let patch = String::from_utf8(commit.patch().to_vec())?;
    assert!(patch.contains("diff --git a/new.rs b/new.rs"));
    assert!(patch.contains("@@ -0,0 +1 @@"));
    assert!(
        repo.prospective_commit(CommitMode::Amend)
            .unwrap_err()
            .to_string()
            .contains("no existing commit")
    );
    Ok(())
}

#[test]
fn normal_modification_has_versions_and_zero_context_hunk() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("file.txt", "one\ntwo\nthree\n")?;
    fixture.commit("initial")?;
    fixture.write_and_stage("file.txt", "one\nsecond\nextra\nthree\n")?;
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let change = only_change(&commit);
    let CommitChangeKind::Modified { before, after } = &change.kind else {
        panic!("expected a modification, got {:?}", change.kind);
    };
    assert_eq!(before.path, after.path);
    assert_ne!(before.oid, after.oid);
    assert_eq!(repo.blob(before.oid)?, b"one\ntwo\nthree\n");
    assert_eq!(repo.blob(after.oid)?, b"one\nsecond\nextra\nthree\n");
    assert_eq!(
        change.hunks,
        [DiffHunk {
            before: Some(LineRange { start: 2, count: 1 }),
            after: Some(LineRange { start: 2, count: 2 }),
        }]
    );
    assert!(String::from_utf8(commit.patch().to_vec())?.contains("-two\n+second\n+extra\n"));
    Ok(())
}

#[test]
fn interactive_commit_prefills_message_and_supports_amend() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.configure_cli_commit()?;
    fixture.write_and_stage("file.txt", "first\n")?;
    let repo = fixture.git_repo()?;

    repo.commit_interactively(CommitMode::Normal, "Describe first version\n\nInitial body")?;
    let first = fixture.repository.head()?.peel_to_commit()?;
    assert_eq!(
        first.message()?.trim_end(),
        "Describe first version\n\nInitial body"
    );
    assert_eq!(first.parent_count(), 0);
    let first_id = first.id();
    drop(first);

    fixture.write_and_stage("file.txt", "amended\n")?;
    repo.commit_interactively(CommitMode::Amend, "Describe amended version")?;
    let amended = fixture.repository.head()?.peel_to_commit()?;
    assert_ne!(amended.id(), first_id);
    assert_eq!(amended.message()?.trim_end(), "Describe amended version");
    assert_eq!(amended.parent_count(), 0);
    assert_eq!(
        amended.tree()?.get_path(Path::new("file.txt"))?.id(),
        fixture.repository.blob(b"amended\n")?
    );
    Ok(())
}

#[test]
fn prospective_commit_ignores_worktree_only_changes() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("tracked.txt", "indexed\n")?;
    fixture.commit("initial")?;

    fixture.write("tracked.txt", "unstaged tracked edit\n")?;
    fixture.write("untracked.txt", "unstaged untracked file\n")?;

    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    assert!(repo.is_dirty()?);
    assert!(commit.is_empty());
    assert!(commit.patch().is_empty());
    assert_eq!(
        commit.stats(),
        CommitStats {
            files_changed: 0,
            insertions: 0,
            deletions: 0,
        }
    );
    Ok(())
}

#[test]
fn prospective_commit_reads_index_not_worktree() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("file.txt", "base\n")?;
    fixture.commit("initial")?;

    fixture.write_and_stage("file.txt", "staged\n")?;
    fixture.write("file.txt", "unstaged later edit\n")?;

    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let change = only_change(&commit);
    let CommitChangeKind::Modified { before, after } = &change.kind else {
        panic!("expected a modification, got {:?}", change.kind);
    };

    assert_eq!(repo.blob(before.oid)?, b"base\n");
    assert_eq!(repo.blob(after.oid)?, b"staged\n");
    assert_eq!(
        repo.index_file("file.txt")?.as_deref(),
        Some(b"staged\n".as_slice())
    );
    assert_eq!(
        change.hunks,
        [DiffHunk {
            before: Some(LineRange { start: 1, count: 1 }),
            after: Some(LineRange { start: 1, count: 1 }),
        }]
    );

    let patch = String::from_utf8(commit.patch().to_vec())?;
    assert!(patch.contains("-base\n+staged\n"));
    assert!(!patch.contains("unstaged later edit"));
    Ok(())
}

#[test]
fn working_tree_context_includes_only_non_ignored_files() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write(".gitignore", "ignored.txt\nignored-directory/\n")?;
    fixture.write("visible.txt", "visible\n")?;
    fixture.write("nested/untracked.rs", "fn untracked() {}\n")?;
    fixture.write("ignored.txt", "ignored\n")?;
    fixture.write("ignored-directory/secret.txt", "ignored\n")?;
    let repo = fixture.git_repo()?;

    assert_eq!(
        repo.working_tree_files()?,
        [
            PathBuf::from(".gitignore"),
            PathBuf::from("nested/untracked.rs"),
            PathBuf::from("visible.txt"),
        ]
    );

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let prompt = crate::generation::Prompt::new(&repo, &[], &commit, &[], 10_000)?;
    assert!(prompt.text.contains("## Working-tree layout"));
    assert!(prompt.text.contains("nested/\n  untracked.rs"));
    assert!(!prompt.text.contains("secret.txt"));
    assert!(!prompt.text.contains("ignored-directory"));
    Ok(())
}

#[cfg(unix)]
#[test]
fn working_tree_context_does_not_follow_directory_symlinks() -> Result<()> {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new()?;
    fixture.write("target/file.txt", "contents\n")?;
    symlink("target", fixture.path.join("linked"))?;
    let repo = fixture.git_repo()?;

    assert_eq!(
        repo.working_tree_files()?,
        [PathBuf::from("linked"), PathBuf::from("target/file.txt")]
    );
    Ok(())
}

#[test]
fn deletion_has_only_before_version() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("deleted.txt", "one\ntwo\n")?;
    fixture.commit("initial")?;
    fixture.delete_and_stage("deleted.txt")?;
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let change = only_change(&commit);
    let CommitChangeKind::Deleted { before } = &change.kind else {
        panic!("expected a deletion, got {:?}", change.kind);
    };
    assert_eq!(before.path, Path::new("deleted.txt"));
    assert_eq!(repo.blob(before.oid)?, b"one\ntwo\n");
    assert_eq!(
        change.hunks,
        [DiffHunk {
            before: Some(LineRange { start: 1, count: 2 }),
            after: None,
        }]
    );
    Ok(())
}

#[test]
fn pure_rename_preserves_both_versions() -> Result<()> {
    let fixture = Fixture::new()?;
    let contents = numbered_lines(20);
    fixture.write_and_stage("before.txt", &contents)?;
    fixture.commit("initial")?;
    fixture.rename_and_stage("before.txt", "after.txt")?;
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let change = only_change(&commit);
    let CommitChangeKind::Renamed { before, after } = &change.kind else {
        panic!("expected a rename, got {:?}", change.kind);
    };
    assert_eq!(before.path, Path::new("before.txt"));
    assert_eq!(after.path, Path::new("after.txt"));
    assert_eq!(before.oid, after.oid);
    assert!(change.hunks.is_empty());
    assert_eq!(change.summary_line(), "R\tbefore.txt -> after.txt");
    assert!(crate::syntax::context_for_change(&repo, change)?.is_none());
    Ok(())
}

#[test]
fn rename_with_edit_is_one_change_with_only_edit_hunk() -> Result<()> {
    let fixture = Fixture::new()?;
    let before = numbered_lines(30);
    fixture.write_and_stage("before.rs", &before)?;
    fixture.commit("initial")?;
    fixture.rename_and_stage("before.rs", "after.rs")?;
    let after = before.replace("line 15\n", "changed 15\n");
    fixture.write_and_stage("after.rs", &after)?;
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let change = only_change(&commit);
    let CommitChangeKind::Renamed {
        before: old,
        after: new,
    } = &change.kind
    else {
        panic!("expected a rename with edits, got {:?}", change.kind);
    };
    assert_eq!(old.path, Path::new("before.rs"));
    assert_eq!(new.path, Path::new("after.rs"));
    assert_ne!(old.oid, new.oid);
    assert_eq!(
        change.hunks,
        [DiffHunk {
            before: Some(LineRange {
                start: 15,
                count: 1,
            }),
            after: Some(LineRange {
                start: 15,
                count: 1,
            }),
        }]
    );
    Ok(())
}

#[test]
fn root_commit_amend_uses_empty_tree() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("root.txt", "original\n")?;
    fixture.commit("root")?;
    fixture.write_and_stage("root.txt", "replacement\n")?;
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Amend)?;
    let change = only_change(&commit);
    let CommitChangeKind::Added { after } = &change.kind else {
        panic!(
            "expected root amend to add the index tree, got {:?}",
            change.kind
        );
    };
    assert_eq!(repo.blob(after.oid)?, b"replacement\n");
    assert_eq!(change.hunks[0].before, None);
    Ok(())
}

#[test]
fn ordinary_amend_uses_parent_tree() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("file.txt", "base\n")?;
    fixture.commit("base")?;
    fixture.write_and_stage("file.txt", "head\n")?;
    fixture.commit("head")?;
    fixture.write_and_stage("file.txt", "amended\n")?;
    let repo = fixture.git_repo()?;

    let amended = repo.prospective_commit(CommitMode::Amend)?;
    let CommitChangeKind::Modified { before, after } = &only_change(&amended).kind else {
        panic!("expected an amended modification");
    };
    assert_eq!(repo.blob(before.oid)?, b"base\n");
    assert_eq!(repo.blob(after.oid)?, b"amended\n");

    let normal = repo.prospective_commit(CommitMode::Normal)?;
    let CommitChangeKind::Modified { before, .. } = &only_change(&normal).kind else {
        panic!("expected a normal modification");
    };
    assert_eq!(repo.blob(before.oid)?, b"head\n");
    Ok(())
}

#[test]
fn merge_commit_amend_is_rejected() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("file.txt", "contents\n")?;
    let root = fixture.commit("root")?;
    let main = fixture.commit_with_parents("main", &[root], true)?;
    let side = fixture.commit_with_parents("side", &[root], false)?;
    fixture.commit_with_parents("merge", &[main, side], true)?;
    let repo = fixture.git_repo()?;

    let error = repo.prospective_commit(CommitMode::Amend).unwrap_err();
    assert!(error.to_string().contains("cannot amend merge commit"));
    assert!(error.to_string().contains("2 parents"));
    Ok(())
}

#[test]
fn detached_head_has_no_current_branch() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("file.txt", "contents\n")?;
    let head = fixture.commit("root")?;
    fixture.repository.set_head_detached(head)?;
    let repo = fixture.git_repo()?;

    assert_eq!(repo.current_branch()?, None);
    assert_eq!(repo.head_sha()?, Some(head));
    assert_eq!(repo.recent_commit_subjects(1)?, ["root"]);
    Ok(())
}

#[test]
fn pathspec_characters_are_literal_paths() -> Result<()> {
    let fixture = Fixture::new()?;
    for path in ["literal*.rs", "question?.rs", "[bracket].rs"] {
        fixture.write_and_stage(path, "fn changed() {}\n")?;
    }
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let mut paths = commit
        .changes()
        .iter()
        .map(|change| {
            change
                .after()
                .expect("addition has after version")
                .path
                .clone()
        })
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(
        paths,
        [
            PathBuf::from("[bracket].rs"),
            PathBuf::from("literal*.rs"),
            PathBuf::from("question?.rs"),
        ]
    );
    assert!(commit.changes().iter().all(|change| {
        change.hunks
            == [DiffHunk {
                before: None,
                after: Some(LineRange { start: 1, count: 1 }),
            }]
    }));
    Ok(())
}

#[cfg(unix)]
#[test]
fn symbolic_link_change_is_a_type_change() -> Result<()> {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new()?;
    fixture.write_and_stage("entry", "regular contents\n")?;
    fixture.commit("regular")?;
    fs::remove_file(fixture.path.join("entry"))?;
    symlink("target", fixture.path.join("entry"))?;
    fixture.stage("entry")?;
    let repo = fixture.git_repo()?;

    let commit = repo.prospective_commit(CommitMode::Normal)?;
    let CommitChangeKind::TypeChanged { before, after } = &only_change(&commit).kind else {
        panic!("expected a type change");
    };
    assert_eq!(before.mode, FileMode::Blob);
    assert_eq!(after.mode, FileMode::Link);
    assert_eq!(repo.blob(after.oid)?, b"target");
    Ok(())
}

#[test]
fn syntax_context_consumes_owned_change_data() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("source.rs", "fn changed() {\n    let value = 1;\n}\n")?;
    fixture.commit("initial")?;
    fixture.write_and_stage("source.rs", "fn changed() {\n    let value = 2;\n}\n")?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    let context = crate::syntax::context_for_change(&repo, only_change(&commit))?
        .expect("Rust modification should have syntax context");
    let before = context.before.as_ref().expect("before side");
    let after = context.after.as_ref().expect("after side");
    assert_eq!(before.entries.len(), 1);
    assert_eq!(after.entries.len(), 1);
    assert_eq!(after.entries[0].items[0].declaration, "fn changed()");

    let prompt = crate::generation::Prompt::new(&repo, &[], &commit, &[], 10_000)?;
    assert!(prompt.text.contains("## Syntax context"));
    assert!(prompt.text.contains("### source.rs"));
    assert!(prompt.text.contains("CONTEXT:\nfn changed()"));
    assert!(!prompt.text.contains("BEFORE:\nfn changed()"));
    assert!(!prompt.text.contains("AFTER:\nfn changed()"));
    Ok(())
}

#[test]
fn excluded_diff_file_keeps_status_but_omits_patch_and_syntax_context() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("kept.rs", "fn kept() { old(); }\n")?;
    fixture.write_and_stage("generated.rs", "fn generated() { old(); }\n")?;
    fixture.commit("initial")?;
    fixture.write_and_stage("kept.rs", "fn kept() { new(); }\n")?;
    fixture.write_and_stage("generated.rs", "fn generated() { secret(); }\n")?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    let prompt = crate::generation::Prompt::new(
        &repo,
        &[],
        &commit,
        &[PathBuf::from("generated.rs")],
        10_000,
    )?;

    assert!(prompt.text.contains("M\tgenerated.rs"));
    assert!(prompt.text.contains("fn kept() { new(); }"));
    assert!(prompt.text.contains("### kept.rs"));
    assert!(!prompt.text.contains("secret();"));
    assert!(!prompt.text.contains("### generated.rs"));
    Ok(())
}

#[test]
fn added_function_has_after_syntax_context_only() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("added.rs", "pub fn added() {\n    work();\n}\n")?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    let context = crate::syntax::context_for_change(&repo, only_change(&commit))?
        .expect("added Rust function should have syntax context");
    assert!(context.before.is_none());
    let after = context.after.as_ref().expect("after side");
    assert_eq!(after.path, Path::new("added.rs"));
    assert_eq!(after.entries.len(), 1);
    assert_eq!(after.entries[0].items[0].declaration, "pub fn added()");
    Ok(())
}

#[test]
fn deleted_function_has_before_syntax_context_only() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("deleted.rs", "fn deleted() {\n    work();\n}\n")?;
    fixture.commit("initial")?;
    fixture.delete_and_stage("deleted.rs")?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    let context = crate::syntax::context_for_change(&repo, only_change(&commit))?
        .expect("deleted Rust function should have syntax context");
    let before = context.before.as_ref().expect("before side");
    assert!(context.after.is_none());
    assert_eq!(before.path, Path::new("deleted.rs"));
    assert_eq!(before.entries.len(), 1);
    assert_eq!(before.entries[0].items[0].declaration, "fn deleted()");
    Ok(())
}

#[test]
fn rename_with_edit_uses_both_paths_and_blobs_for_syntax() -> Result<()> {
    let fixture = Fixture::new()?;
    let before = "fn process() {\n    step_1();\n    step_2();\n    step_3();\n    step_4();\n    step_5();\n    step_6();\n}\n";
    let after = before.replace("step_4();", "changed_step();");
    fixture.write_and_stage("old.rs", before)?;
    fixture.commit("initial")?;
    fixture.rename_and_stage("old.rs", "new.rs")?;
    fixture.write_and_stage("new.rs", &after)?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    let change = only_change(&commit);
    assert!(matches!(change.kind, CommitChangeKind::Renamed { .. }));
    let context = crate::syntax::context_for_change(&repo, change)?
        .expect("edited Rust rename should have syntax context");
    let before = context.before.as_ref().expect("before side");
    let after = context.after.as_ref().expect("after side");
    assert_eq!(before.path, Path::new("old.rs"));
    assert_eq!(after.path, Path::new("new.rs"));
    assert_eq!(before.entries[0].items[0].declaration, "fn process()");
    assert_eq!(after.entries[0].items[0].declaration, "fn process()");
    let prompt = crate::generation::Prompt::new(&repo, &[], &commit, &[], 10_000)?;
    assert!(prompt.text.contains("### old.rs -> new.rs"));
    Ok(())
}

#[test]
fn syntax_context_preserves_different_before_and_after_structure() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("scope.rs", "fn production() {\n    changed();\n}\n")?;
    fixture.commit("initial")?;
    fixture.write_and_stage(
        "scope.rs",
        "#[test]\nfn changed_test() {\n    changed();\n}\n",
    )?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    let context = crate::syntax::context_for_change(&repo, only_change(&commit))?
        .expect("scope change should have syntax context");
    let before = context.before.as_ref().expect("before side");
    let after = context.after.as_ref().expect("after side");
    assert_eq!(before.entries[0].items[0].declaration, "fn production()");
    assert_eq!(
        after.entries[0].items[0].declaration,
        "#[test]\nfn changed_test()"
    );
    let prompt = crate::generation::Prompt::new(&repo, &[], &commit, &[], 10_000)?;
    assert!(prompt.text.contains("BEFORE:\nfn production()"));
    assert!(prompt.text.contains("AFTER:\n#[test]\nfn changed_test()"));
    Ok(())
}

#[test]
fn unsupported_file_has_no_syntax_context_or_prompt_section() -> Result<()> {
    let fixture = Fixture::new()?;
    fixture.write_and_stage("notes.unknown", "before\n")?;
    fixture.commit("initial")?;
    fixture.write_and_stage("notes.unknown", "after\n")?;
    let repo = fixture.git_repo()?;
    let commit = repo.prospective_commit(CommitMode::Normal)?;

    assert!(crate::syntax::context_for_change(&repo, only_change(&commit))?.is_none());
    let prompt = crate::generation::Prompt::new(&repo, &[], &commit, &[], 10_000)?;
    assert!(!prompt.text.contains("## Syntax context"));
    Ok(())
}
