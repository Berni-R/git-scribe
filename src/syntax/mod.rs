mod context;
mod language;

pub use context::*;
pub use language::*;

#[test]
fn parses_supported_languages() -> anyhow::Result<()> {
    let cases = [
        (Language::Rust, "fn f() {}"),
        (Language::C, "void f(void) {}"),
        (Language::Cpp, "void f() {}"),
        (Language::Python, "def f():\n    pass\n"),
        (Language::Swift, "func f() {}"),
        (Language::Bash, "f() { echo hi; }\n"),
        (Language::JavaScript, "function f() {}"),
        (Language::TypeScript, "function f(): void {}"),
        (Language::Tsx, "const f = <div />;"),
        (Language::Css, ".f { background: #1166ee; }"),
        (Language::Html, "<div id=\"f\"></div>"),
        (Language::Toml, "[f]\nfoo = 42"),
        (Language::Json, "{\"f\": 42}"),
    ];

    for (language, source) in cases {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&language.tree_sitter())
            .expect("failed to load Tree-sitter language");
        let tree = parser
            .parse(source, None)
            .expect("Tree-sitter failed to parse source");

        assert!(
            !tree.root_node().has_error(),
            "{language:?}: {}",
            tree.root_node().to_sexp()
        );
        println!("{:?}", tree.root_node().utf8_text(source.as_bytes()));
    }

    Ok(())
}
