use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context as _, Result};
use tree_sitter::{Node, Parser, Point};

use crate::{
    GitRepo,
    git::{CommitChange, DiffHunk, FileVersion, LineRange},
};

use super::Language;

const MAX_ENTRIES_PER_SIDE: usize = 12;
const MAX_ITEMS_PER_ENTRY: usize = 4;
const MAX_DECLARATION_BYTES: usize = 400;

/// Language-independent role of source-derived syntax evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntaxKind {
    Module,
    Type,
    Impl,
    Function,
    Method,
    Test,
    Field,
    Constant,
    Import,
    ControlFlow,
    Other,
}

/// One useful source-derived construct associated with changed lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxItem {
    pub kind: SyntaxKind,

    /// Faithful, compact source text for the selected construct without its body.
    pub declaration: String,

    /// One-based source line on which the declaration begins.
    pub start_line: usize,
}

/// A chain of useful constructs from outer context to the most local changed construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxEntry {
    pub items: Vec<SyntaxItem>,
}

/// Structural evidence for one available side of a changed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxSide {
    pub path: PathBuf,
    pub entries: Vec<SyntaxEntry>,
}

/// Structural evidence associated with the changed lines of one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxContext {
    pub language: Language,
    pub before: Option<SyntaxSide>,
    pub after: Option<SyntaxSide>,
}

/// Extract Tree-sitter context for one owned prospective-commit change.
///
/// Before and after blobs are parsed independently using their corresponding hunk ranges. Pure
/// renames, non-blob entries, and unsupported languages produce no context.
pub fn context_for_change(repo: &GitRepo, change: &CommitChange) -> Result<Option<SyntaxContext>> {
    if change.hunks.is_empty() {
        return Ok(None);
    }

    let before = load_side(repo, change.before(), &change.hunks, false)?;
    let after = load_side(repo, change.after(), &change.hunks, true)?;
    let language = match &after {
        Some(side) => detect_side(side),
        None => before.as_ref().and_then(detect_side),
    };
    let Some(language) = language else {
        return Ok(None);
    };

    let before = match before {
        Some(side) => Some(analyze_side(side, language)?),
        None => None,
    };
    let after = match after {
        Some(side) => Some(analyze_side(side, language)?),
        None => None,
    };

    if before
        .iter()
        .chain(&after)
        .all(|side| side.entries.is_empty())
    {
        return Ok(None);
    }

    Ok(Some(SyntaxContext {
        language,
        before,
        after,
    }))
}

struct SourceSide {
    version: FileVersion,
    source: Vec<u8>,
    ranges: Vec<LineRange>,
}

fn load_side(
    repo: &GitRepo,
    version: Option<&FileVersion>,
    hunks: &[DiffHunk],
    after: bool,
) -> Result<Option<SourceSide>> {
    let Some(version) = version.filter(|version| version.is_blob()) else {
        return Ok(None);
    };
    let ranges = hunks
        .iter()
        .filter_map(|hunk| if after { hunk.after } else { hunk.before })
        .collect::<Vec<_>>();
    if ranges.is_empty() {
        return Ok(None);
    }

    Ok(Some(SourceSide {
        version: version.clone(),
        source: repo.blob(version.oid)?,
        ranges,
    }))
}

fn detect_side(side: &SourceSide) -> Option<Language> {
    Language::detect(&side.version.path, &side.source)
}

fn analyze_side(side: SourceSide, language: Language) -> Result<SyntaxSide> {
    Ok(SyntaxSide {
        path: side.version.path,
        entries: context_for_ranges(&side.source, language, &side.ranges)?,
    })
}

/// Extract compact declaration chains around one-based changed line ranges.
pub fn context_for_ranges(
    source: &[u8],
    language: Language,
    ranges: &[LineRange],
) -> Result<Vec<SyntaxEntry>> {
    if ranges.is_empty() {
        return Ok(Vec::new());
    }

    let mut parser = Parser::new();
    parser
        .set_language(&language.tree_sitter())
        .context("failed to load Tree-sitter grammar")?;
    let tree = parser
        .parse(source, None)
        .context("Tree-sitter failed to parse source")?;
    let root = tree.root_node();
    let lines = source.split(|&byte| byte == b'\n').collect::<Vec<_>>();

    let mut entries = Vec::new();
    let mut seen = HashSet::new();
    'ranges: for range in ranges {
        for row in changed_rows(*range, lines.len()) {
            let Some(focus) = focus_on_row(root, &lines, row) else {
                continue;
            };
            let (key, entry) = syntax_entry(source, focus, language, row);
            if entry.items.is_empty() || !seen.insert(key) {
                continue;
            }
            entries.push(entry);
            if entries.len() == MAX_ENTRIES_PER_SIDE {
                break 'ranges;
            }
        }
    }

    Ok(entries)
}

fn changed_rows(range: LineRange, line_count: usize) -> impl Iterator<Item = usize> {
    let start = range.start.saturating_sub(1).min(line_count);
    let end = start.saturating_add(range.count).min(line_count);
    start..end
}

fn focus_on_row<'tree>(root: Node<'tree>, lines: &[&[u8]], row: usize) -> Option<Node<'tree>> {
    let line = *lines.get(row)?;
    let column = line
        .iter()
        .take_while(|&&byte| matches!(byte, b' ' | b'\t'))
        .count()
        .min(line.len().saturating_sub(1));
    let point = Point::new(row, column);
    root.named_descendant_for_point_range(point, point)
}

fn syntax_entry(
    source: &[u8],
    focus: Node<'_>,
    language: Language,
    changed_row: usize,
) -> (Vec<(usize, usize)>, SyntaxEntry) {
    let mut items = Vec::new();
    let mut current = Some(focus);
    while let Some(node) = current {
        if let Some(item) = syntax_item(source, node, language)
            && (item.kind != SyntaxKind::ControlFlow || control_header_contains(node, changed_row))
        {
            items.push((node.start_byte(), node.end_byte(), item));
        }
        current = node.parent();
    }
    if items.len() > MAX_ITEMS_PER_ENTRY {
        items.truncate(MAX_ITEMS_PER_ENTRY);
    }
    items.reverse();

    let key = items.iter().map(|(start, end, _)| (*start, *end)).collect();
    let items = items.into_iter().map(|(_, _, item)| item).collect();
    (key, SyntaxEntry { items })
}

fn control_header_contains(node: Node<'_>, changed_row: usize) -> bool {
    let header_end = ["body", "consequence"]
        .into_iter()
        .find_map(|field| node.child_by_field_name(field))
        .map_or(node.start_position().row, |body| body.start_position().row);

    (node.start_position().row..=header_end).contains(&changed_row)
}

fn syntax_item(source: &[u8], node: Node<'_>, language: Language) -> Option<SyntaxItem> {
    let (start, start_line) = declaration_start(node, language);
    let end = declaration_end(node, language);
    let declaration = compact_source(source.get(start..end)?)?;
    let kind = syntax_kind(node, language, &declaration)?;

    Some(SyntaxItem {
        kind,
        declaration,
        start_line,
    })
}

#[allow(clippy::too_many_lines)] // One declarative mapping keeps grammar knowledge in one place.
fn syntax_kind(node: Node<'_>, language: Language, declaration: &str) -> Option<SyntaxKind> {
    let kind = match (language, node.kind()) {
        (Language::Rust, "mod_item")
        | (Language::Cpp, "namespace_definition")
        | (Language::TypeScript | Language::Tsx, "internal_module")
        | (Language::Toml, "table" | "table_array_element") => SyntaxKind::Module,

        (
            Language::Rust,
            "struct_item" | "enum_item" | "union_item" | "trait_item" | "type_item",
        )
        | (
            Language::C | Language::Cpp,
            "struct_specifier" | "union_specifier" | "enum_specifier",
        )
        | (Language::Cpp, "class_specifier")
        | (Language::Python, "class_definition")
        | (
            Language::Swift,
            "class_declaration"
            | "struct_declaration"
            | "enum_declaration"
            | "protocol_declaration",
        )
        | (Language::JavaScript, "class_declaration")
        | (
            Language::TypeScript | Language::Tsx,
            "class_declaration"
            | "abstract_class_declaration"
            | "interface_declaration"
            | "type_alias_declaration"
            | "enum_declaration",
        ) => SyntaxKind::Type,

        (Language::Rust, "impl_item") | (Language::Swift, "extension_declaration") => {
            SyntaxKind::Impl
        }

        (Language::Rust, "field_declaration")
        | (Language::Swift, "property_declaration")
        | (Language::Toml | Language::Json, "pair") => SyntaxKind::Field,

        (Language::Rust, "const_item" | "static_item") => SyntaxKind::Constant,

        (Language::Rust, "use_declaration")
        | (Language::C | Language::Cpp, "preproc_include")
        | (Language::Python, "import_statement" | "import_from_statement")
        | (Language::Swift, "import_declaration")
        | (Language::JavaScript | Language::TypeScript | Language::Tsx, "import_statement") => {
            SyntaxKind::Import
        }

        (Language::Rust, "function_item" | "function_signature_item")
            if is_rust_test(declaration) =>
        {
            SyntaxKind::Test
        }

        (Language::Swift, "initializer_declaration" | "subscript_declaration")
        | (Language::JavaScript | Language::TypeScript | Language::Tsx, "method_definition")
        | (Language::TypeScript | Language::Tsx, "method_signature") => SyntaxKind::Method,

        (Language::Rust, "function_item" | "function_signature_item")
        | (
            Language::C | Language::Cpp | Language::Python | Language::Bash,
            "function_definition",
        )
        | (Language::Swift, "function_declaration")
        | (
            Language::JavaScript | Language::TypeScript | Language::Tsx,
            "function_declaration"
            | "generator_function_declaration"
            | "function_expression"
            | "generator_function"
            | "arrow_function",
        ) if has_method_parent(node) => SyntaxKind::Method,

        (Language::Rust, "function_item" | "function_signature_item")
        | (
            Language::C | Language::Cpp | Language::Python | Language::Bash,
            "function_definition",
        )
        | (Language::Swift, "function_declaration")
        | (
            Language::JavaScript | Language::TypeScript | Language::Tsx,
            "function_declaration"
            | "generator_function_declaration"
            | "function_expression"
            | "generator_function"
            | "arrow_function",
        ) => SyntaxKind::Function,

        (
            Language::Rust,
            "if_expression" | "match_expression" | "for_expression" | "while_expression"
            | "loop_expression",
        )
        | (
            Language::C
            | Language::Cpp
            | Language::Swift
            | Language::JavaScript
            | Language::TypeScript
            | Language::Tsx,
            "if_statement" | "for_statement" | "while_statement" | "switch_statement",
        )
        | (
            Language::Python,
            "if_statement" | "for_statement" | "while_statement" | "with_statement"
            | "match_statement",
        )
        | (
            Language::Bash,
            "if_statement" | "for_statement" | "while_statement" | "case_statement",
        ) => SyntaxKind::ControlFlow,

        (Language::Rust, "enum_variant")
        | (Language::C | Language::Cpp, "declaration")
        | (Language::Cpp, "template_declaration")
        | (Language::Python, "decorated_definition")
        | (
            Language::Css,
            "rule_set"
            | "media_statement"
            | "supports_statement"
            | "scope_statement"
            | "keyframes_statement",
        )
        | (Language::Html, "element" | "script_element" | "style_element") => SyntaxKind::Other,

        _ => return None,
    };

    Some(kind)
}

fn has_method_parent(node: Node<'_>) -> bool {
    let mut parent = node.parent();
    while let Some(node) = parent {
        if matches!(
            node.kind(),
            "impl_item"
                | "trait_item"
                | "class_specifier"
                | "class_declaration"
                | "abstract_class_declaration"
                | "interface_declaration"
                | "extension_declaration"
        ) {
            return true;
        }
        parent = node.parent();
    }
    false
}

fn is_rust_test(declaration: &str) -> bool {
    declaration.lines().any(|line| {
        let attribute = line.trim();
        attribute == "#[test]"
            || attribute
                .strip_prefix("#[")
                .and_then(|attribute| attribute.split(['(', ']']).next())
                .is_some_and(|attribute| attribute.ends_with("::test"))
    })
}

fn declaration_end(node: Node<'_>, language: Language) -> usize {
    if let Some(content) = ["body", "consequence", "value"]
        .into_iter()
        .find_map(|field| node.child_by_field_name(field))
    {
        return content.start_byte();
    }

    if language == Language::Css {
        let mut cursor = node.walk();
        if let Some(block) = node
            .named_children(&mut cursor)
            .find(|child| child.kind() == "block")
        {
            return block.start_byte();
        }
    }

    if language == Language::Html {
        let mut cursor = node.walk();
        if let Some(start_tag) = node.named_children(&mut cursor).find(|child| {
            matches!(
                child.kind(),
                "start_tag" | "script_start_tag" | "style_start_tag"
            )
        }) {
            return start_tag.end_byte();
        }
    }

    node.end_byte()
}

fn declaration_start(mut node: Node<'_>, language: Language) -> (usize, usize) {
    if language == Language::Rust {
        while let Some(attribute) = node
            .prev_named_sibling()
            .filter(|sibling| sibling.kind() == "attribute_item")
        {
            node = attribute;
        }
    }

    (node.start_byte(), node.start_position().row + 1)
}

fn compact_source(source: &[u8]) -> Option<String> {
    let source = String::from_utf8_lossy(source);
    let declaration = source.trim().trim_end_matches('{').trim();
    (!declaration.is_empty() && declaration.len() <= MAX_DECLARATION_BYTES)
        .then(|| declaration.to_owned())
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::*;

    fn rust_context(source: &str, ranges: &[(usize, usize)]) -> Vec<SyntaxEntry> {
        let ranges = ranges
            .iter()
            .map(|&(start, count)| LineRange { start, count })
            .collect::<Vec<_>>();
        context_for_ranges(source.as_bytes(), Language::Rust, &ranges).unwrap()
    }

    fn declarations(entry: &SyntaxEntry) -> Vec<&str> {
        entry
            .items
            .iter()
            .map(|item| item.declaration.as_str())
            .collect()
    }

    fn kinds(entry: &SyntaxEntry) -> Vec<SyntaxKind> {
        entry.items.iter().map(|item| item.kind).collect()
    }

    #[test]
    fn changed_statement_identifies_free_function() {
        let entries = rust_context("fn calculate() {\n    let value = 2;\n}\n", &[(2, 1)]);

        assert_eq!(entries.len(), 1);
        assert_eq!(declarations(&entries[0]), ["fn calculate()"]);
        assert_eq!(kinds(&entries[0]), [SyntaxKind::Function]);
    }

    #[test]
    fn changed_statement_identifies_method_and_impl() {
        let source = "impl ApiClient {\n    pub async fn send_request(&self) -> Result<()> {\n        timeout(120);\n        Ok(())\n    }\n}\n";
        let entries = rust_context(source, &[(3, 1)]);

        assert_eq!(entries.len(), 1);
        assert_eq!(
            declarations(&entries[0]),
            [
                "impl ApiClient",
                "pub async fn send_request(&self) -> Result<()>"
            ]
        );
        assert_eq!(kinds(&entries[0]), [SyntaxKind::Impl, SyntaxKind::Method]);
    }

    #[test]
    fn test_attribute_is_preserved() {
        let source = "#[test]\nfn request_times_out() {\n    assert!(true);\n}\n";
        let entries = rust_context(source, &[(3, 1)]);

        assert_eq!(entries.len(), 1);
        assert_eq!(
            declarations(&entries[0]),
            ["#[test]\nfn request_times_out()"]
        );
        assert_eq!(kinds(&entries[0]), [SyntaxKind::Test]);
    }

    #[test]
    fn test_module_attribute_is_preserved_as_enclosing_context() {
        let source = "#[cfg(test)]\nmod tests {\n    fn helper() {\n        changed();\n    }\n}\n";
        let entries = rust_context(source, &[(4, 1)]);

        assert_eq!(
            declarations(&entries[0]),
            ["#[cfg(test)]\nmod tests", "fn helper()"]
        );
    }

    #[test]
    fn changed_function_signature_is_rendered_without_body() {
        let source = "pub fn request(timeout: Duration) -> Result<Response> {\n    todo!()\n}\n";
        let entries = rust_context(source, &[(1, 1)]);

        assert_eq!(
            declarations(&entries[0]),
            ["pub fn request(timeout: Duration) -> Result<Response>"]
        );
    }

    #[test]
    fn changed_struct_field_identifies_field_and_struct() {
        let source = "struct Config {\n    retries: usize,\n    request_timeout: Duration,\n}\n";
        let entries = rust_context(source, &[(3, 1)]);

        assert_eq!(
            declarations(&entries[0]),
            ["struct Config", "request_timeout: Duration"]
        );
        assert_eq!(kinds(&entries[0]), [SyntaxKind::Type, SyntaxKind::Field]);
    }

    #[test]
    fn imports_and_constants_have_specific_semantic_kinds() {
        let cases = [
            ("use crate::client::Client;\n", SyntaxKind::Import),
            ("const TIMEOUT: u64 = 120;\n", SyntaxKind::Constant),
        ];

        for (source, expected) in cases {
            let entries = rust_context(source, &[(1, 1)]);
            assert_eq!(entries[0].items.last().unwrap().kind, expected);
        }
    }

    #[test]
    fn trait_method_signature_identifies_method_and_trait() {
        let source = "trait Client {\n    fn request(&self) -> Response;\n}\n";
        let entries = rust_context(source, &[(2, 1)]);

        assert_eq!(
            declarations(&entries[0]),
            ["trait Client", "fn request(&self) -> Response;"]
        );
    }

    #[test]
    fn enclosing_control_flow_does_not_obscure_method_and_impl() {
        let source = "impl Client {\n    fn request(&self) {\n        if ready() {\n            for item in items() {\n                while active() {\n                    changed();\n                }\n            }\n        }\n    }\n}\n";
        let entries = rust_context(source, &[(6, 1)]);

        assert_eq!(
            declarations(&entries[0]),
            ["impl Client", "fn request(&self)"]
        );
    }

    #[test]
    fn changed_control_flow_header_is_included() {
        let source = "fn request() {\n    if timeout > 30 {\n        retry();\n    }\n}\n";
        let entries = rust_context(source, &[(2, 1)]);

        assert_eq!(
            kinds(&entries[0]),
            [SyntaxKind::Function, SyntaxKind::ControlFlow]
        );
        assert_eq!(
            declarations(&entries[0]),
            ["fn request()", "if timeout > 30"]
        );
    }

    #[test]
    fn multiple_hunks_in_same_function_are_deduplicated() {
        let source = "fn update() {\n    first();\n    middle();\n    second();\n}\n";
        let entries = rust_context(source, &[(2, 1), (4, 1)]);

        assert_eq!(entries.len(), 1);
        assert_eq!(declarations(&entries[0]), ["fn update()"]);
    }

    #[test]
    fn hunks_in_different_functions_preserve_source_order() {
        let source = "fn first() {\n    changed();\n}\n\nfn second() {\n    changed();\n}\n";
        let entries = rust_context(source, &[(2, 1), (6, 1)]);

        assert_eq!(entries.len(), 2);
        assert_eq!(declarations(&entries[0]), ["fn first()"]);
        assert_eq!(declarations(&entries[1]), ["fn second()"]);
    }

    #[test]
    fn incomplete_source_still_provides_partial_context() {
        let source = "fn incomplete() {\n    let value = ;\n}\n";
        let entries = rust_context(source, &[(2, 1)]);

        assert_eq!(declarations(&entries[0]), ["fn incomplete()"]);
    }

    #[test]
    fn lines_without_useful_structure_are_omitted() {
        let entries = rust_context("let value = 1;\n", &[(1, 1)]);

        assert!(entries.is_empty());
    }

    #[test]
    fn large_function_body_is_not_rendered() {
        let mut source = String::from("fn large() {\n");
        for number in 1..=200 {
            writeln!(source, "    let value_{number} = {number};").unwrap();
        }
        source.push_str("}\n");

        let entries = rust_context(&source, &[(101, 1)]);
        assert_eq!(declarations(&entries[0]), ["fn large()"]);
        assert!(entries[0].items[0].declaration.len() < 100);
    }

    #[test]
    fn repeated_ranges_are_deduplicated() {
        let source = "fn update() {\n    changed();\n}\n";
        let entries = rust_context(source, &[(2, 1), (2, 1), (2, 1)]);

        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn entries_per_side_are_bounded() {
        let mut source = String::new();
        let mut ranges = Vec::new();
        for number in 0..20 {
            writeln!(source, "fn changed_{number}() {{ work(); }}").unwrap();
            ranges.push((number + 1, 1));
        }

        let entries = rust_context(&source, &ranges);
        assert_eq!(entries.len(), MAX_ENTRIES_PER_SIDE);
    }

    #[test]
    fn supported_languages_produce_structural_context() {
        let cases = [
            (Language::C, "void f(void) {\n    return;\n}\n", 2),
            (Language::Cpp, "void f() {\n    return;\n}\n", 2),
            (Language::Python, "def f():\n    value = 1\n", 2),
            (Language::Swift, "func f() {\n    let value = 1\n}\n", 2),
            (Language::Bash, "f() {\n    echo hi\n}\n", 2),
            (
                Language::JavaScript,
                "function f() {\n    return 1;\n}\n",
                2,
            ),
            (
                Language::TypeScript,
                "function f(): number {\n    return 1;\n}\n",
                2,
            ),
            (Language::Tsx, "function F() {\n    return <div />;\n}\n", 2),
            (Language::Css, ".item {\n    color: red;\n}\n", 2),
            (Language::Html, "<main>\n  <p>changed</p>\n</main>\n", 2),
            (Language::Toml, "[client]\ntimeout = 120\n", 2),
            (Language::Json, "{\n  \"timeout\": 120\n}\n", 2),
        ];

        for (language, source, line) in cases {
            let entries = context_for_ranges(
                source.as_bytes(),
                language,
                &[LineRange {
                    start: line,
                    count: 1,
                }],
            )
            .unwrap();
            assert!(!entries.is_empty(), "missing context for {language:?}");
        }
    }
}
