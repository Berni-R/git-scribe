use std::path::Path;

/// Programming languages supoorted for parsing via [`tree_sitter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    C,
    Cpp,
    Python,
    Swift,
    Bash,
    JavaScript,
    TypeScript,
    Tsx,
    Css,
    Html,
    Toml,
    Json,
}

impl Language {
    /// Language identifier suitable for Markdown code fences.
    #[must_use]
    pub const fn fence_name(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Python => "python",
            Self::Swift => "swift",
            Self::Bash => "bash",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::Css => "css",
            Self::Html => "html",
            Self::Toml => "toml",
            Self::Json => "json",
        }
    }
}

impl Language {
    /// Infer the language of a file from its extension, falling back to its shebang.
    #[must_use]
    pub fn detect(path: &Path, source: &[u8]) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(Self::from_extension)
            .or_else(|| Self::from_shebang(source))
    }

    /// Infer the language from a file extension such as `rs` (without the dot).
    ///
    /// # Note
    /// The extension `"h"` is always mapped to C++, not C!
    #[must_use]
    pub fn from_extension(extension: &str) -> Option<Self> {
        Some(match extension {
            "rs" => Self::Rust,

            "c" => Self::C,
            "h" | "cc" | "cpp" | "cxx" | "hh" | "hpp" | "hxx" => Self::Cpp,

            "py" => Self::Python,
            "swift" => Self::Swift,

            "sh" | "bash" => Self::Bash,

            "js" | "mjs" | "cjs" | "jsx" => Self::JavaScript,
            "ts" | "mts" | "cts" => Self::TypeScript,
            "tsx" => Self::Tsx,

            "css" => Self::Css,
            "html" | "htm" => Self::Html,
            "toml" => Self::Toml,
            "json" => Self::Json,

            _ => return None,
        })
    }

    /// Infer the language from the source file's shebang line.
    #[must_use]
    pub fn from_shebang(source: &[u8]) -> Option<Self> {
        let first_line = source.split(|&b| b == b'\n').next()?;
        let shebang = std::str::from_utf8(first_line)
            .ok()?
            .strip_prefix("#!")?
            .trim();

        let command = if let Some(rest) = shebang.strip_prefix("/usr/bin/env") {
            let mut args = rest.split_whitespace();
            match args.next()? {
                "-S" => args.next()?,
                command => command,
            }
        } else {
            shebang.rsplit('/').next()?.split_whitespace().next()?
        };

        Some(match command {
            "sh" | "bash" => Self::Bash,
            "python" | "python2" | "python3" => Self::Python,
            "node" | "nodejs" | "bun" => Self::JavaScript,
            "deno" => Self::TypeScript,
            "swift" => Self::Swift,
            _ => return None,
        })
    }

    /// Get the corresponding [`tree_sitter::Language`].
    #[must_use]
    pub fn tree_sitter(self) -> tree_sitter::Language {
        match self {
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::Swift => tree_sitter_swift::LANGUAGE.into(),
            Self::Bash => tree_sitter_bash::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Css => tree_sitter_css::LANGUAGE.into(),
            Self::Html => tree_sitter_html::LANGUAGE.into(),
            Self::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
            Self::Json => tree_sitter_json::LANGUAGE.into(),
        }
    }
}
