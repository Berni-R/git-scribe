# git-scribe

`git-scribe` suggests commit messages from your **staged Git changes** using a local LLM,
then opens Git's commit editor with the suggestion prefilled.

Unlike tools that simply summarize `git diff`,
`git-scribe` tries to infer the **intent** of a change in the context of the repository.
It combines the staged diff with a small amount of relevant context, including:

* the repository `README.md`
* the working-tree directory structure, excluding Git-ignored entries
* changed files
* branch and recent commit history
* Tree-sitter AST context around changed code

This helps distinguish, for example, a timeout change in production networking code from one made to stabilize a test.

`git-scribe` is written in Rust;
runs locally through [Ollama](https://ollama.com/) and currently uses `qwen3.5:9b`.

## Usage

Stage the changes you want to commit:

```bash
git add ...
git scribe
```

Review or revise the generated message in your configured Git editor, then save and close it to
create the commit.
Pass `--amend` to generate and edit a message for the complete amended commit.
Use `--print` to print the suggestion without opening the editor or creating a commit.
Checkout `--help` for more options.

Git automatically maps the `git scribe` command to an executable named `git-scribe` on your `PATH`.

## Installation

Install `git-scribe` from the repository with Cargo:

```sh
cargo install --path .
```

This installs the `git-scribe` executable into Cargo's binary directory (usually `~/.cargo/bin`),
which must be on your `PATH`.

Git automatically treats executables named `git-<command>` on your `PATH` as external Git commands.
Therefore, both forms are equivalent:

```sh
git-scribe [OPTIONS]
git scribe [OPTIONS]
```
