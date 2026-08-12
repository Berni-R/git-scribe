mod git;

use git::{GitRepo, StagedChangeKind};

fn main() -> anyhow::Result<()> {
    let repo = GitRepo::discover("../tmp")?;

    println!("Repository: {}", repo.root().display());

    match repo.current_branch()? {
        Some(branch) => {
            println!("Branch:     {branch}")
        }
        None => {
            println!("Branch:     <detached HEAD>")
        }
    }

    match repo.head_sha()? {
        Some(sha) => println!("HEAD:       {sha}"),
        None => println!("HEAD:       <no commits yet>"),
    }

    println!("Dirty:      {}", repo.is_dirty()?);

    let changes = repo.staged_changes()?;

    println!();
    println!("Staged changes: {}", changes.len());

    for change in &changes {
        match &change.kind {
            StagedChangeKind::Added => {
                println!("  added      {}", change.path.display());
            }

            StagedChangeKind::Modified => {
                println!("  modified   {}", change.path.display());
            }

            StagedChangeKind::Deleted => {
                println!("  deleted    {}", change.path.display());
            }

            StagedChangeKind::Renamed { from, similarity } => {
                println!(
                    "  renamed    {} -> {} ({similarity}%)",
                    from.display(),
                    change.path.display(),
                );
            }

            StagedChangeKind::TypeChanged => {
                println!("  type       {}", change.path.display());
            }

            StagedChangeKind::Unmerged => {
                println!("  unmerged   {}", change.path.display());
            }
        }
    }

    let diff = repo.staged_diff()?;
    println!("Staged diff: {} bytes", diff.len());

    let recent = repo.recent_commit_subjects(5)?;

    println!();
    println!("Recent commits:");

    if recent.is_empty() {
        println!("  <none>");
    } else {
        for subject in recent {
            println!("  {subject}");
        }
    }

    Ok(())
}
