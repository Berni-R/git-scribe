#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]

pub mod generation;
pub mod git;
pub mod ollama;
pub mod syntax;
pub mod terminal;

pub use git::GitRepo;
