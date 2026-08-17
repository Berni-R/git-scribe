# git-sight

`git-sight` suggests commit messages from your **staged Git changes** using a local LLM.

Unlike tools that simply summarize `git diff`,
`git-sight` tries to infer the **intent** of a change in the context of the repository.
It combines the staged diff with a small amount of relevant context, including:

* the repository `README.md`
* the repository directory structure
* changed files
* branch and recent commit history
* Tree-sitter AST context around changed code

This helps distinguish, for example, a timeout change in production networking code from one made to stabilize a test.

`git-sight` is written in Rust;
runs locally through [Ollama](https://ollama.com/) and currently uses `qwen3:4b-instruct`.

## Usage

Stage the changes you want to commit:

```bash
git add ...
git sight
```

Git automatically maps the `git sight` command to an executable named `git-sight` on your `PATH`.

## Installation

Install `git-sight` from the repository with Cargo:

```sh
cargo install --path .
```

This installs the `git-sight` executable into Cargo's binary directory (usually `~/.cargo/bin`),
which must be on your `PATH`.

Git automatically treats executables named `git-<command>` on your `PATH` as external Git commands.
Therefore, both forms are equivalent:

```sh
git-sight [OPTIONS]
git sight [OPTIONS]
```
