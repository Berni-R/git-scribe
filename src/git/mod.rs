//! Small, read-only wrapper around the installed `git` executable.
//!
//! This module intentionally uses the Git CLI rather than a Git implementation library.
//! That gives us Git's own repository/configuration semantics while keeping the wrapper small.

mod commands;
mod repo;
mod staged;

pub use repo::{CommitMode, GitRepo};
pub use staged::{StagedChange, StagedChangeKind};
