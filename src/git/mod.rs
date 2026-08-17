//! Application-facing Git concepts backed by `git2`/libgit2.
//!
//! The boundary exposes owned prospective commits and file versions rather than command-shaped operations.

mod change;
mod commands;
mod hunk;
mod repo;

pub use change::*;
pub use hunk::*;
pub use repo::*;

#[cfg(test)]
mod tests;
