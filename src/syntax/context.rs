use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context as _, Result};
use tree_sitter::{Node, Parser, Point, Query, QueryCursor, StreamingIterator};

use crate::{
    GitRepo,
    git::{CommitChange, DiffHunk, FileVersion, LineRange},
};

use super::Language;

/// Maximum declarations retained for one file side.
const MAX_ENTRIES_PER_SIDE: usize = 12;
/// Maximum nested declarations retained for one entry.
const MAX_ITEMS_PER_ENTRY: usize = 4;
/// Maximum bytes retained for one declaration.
const MAX_DECLARATION_BYTES: usize = 400;
/// Maximum call sites retained for one changed declaration.
const MAX_CALL_SITES_PER_ENTRY: usize = 3;

/// One useful source-derived construct associated with changed lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxItem {
    /// Faithful, compact source text for the selected construct without its body.
    pub declaration: String,

    /// One-based source line on which the declaration begins.
    pub start_line: usize,
}

/// A chain of useful constructs from outer context to the most local changed construct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxEntry {
    /// Nested constructs from outermost to innermost.
    pub items: Vec<SyntaxItem>,
}

/// One direct call to a changed function or method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxCallSite {
    /// Enclosing declaration chain of the caller.
    pub caller: SyntaxEntry,
    /// Source line containing the call.
    pub call: String,
}

/// Structural evidence for one available side of a changed file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxSide {
    /// File path for this side.
    pub path: PathBuf,
    /// Extracted declaration chains.
    pub entries: Vec<SyntaxEntry>,
    /// Direct same-file callers of changed functions or methods.
    pub call_sites: Vec<SyntaxCallSite>,
}

/// Structural evidence associated with the changed lines of one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxContext {
    /// Detected source language.
    pub language: Language,
    /// Context from the base version.
    pub before: Option<SyntaxSide>,
    /// Context from the prospective version.
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
    /// File version and path.
    version: FileVersion,
    /// Blob contents.
    source: Vec<u8>,
    /// Changed line ranges.
    ranges: Vec<LineRange>,
}

/// Definition metadata indexed by source-byte range.
type DefinitionIndex = HashMap<(usize, usize), Definition>;

/// A definition identified by a Tree-sitter tag query.
struct Definition {
    /// Declared symbol name.
    name: String,
    /// Whether this definition can be called directly.
    callable: bool,
}

/// Extracted structure for one source side.
struct ExtractedContext {
    /// Declaration chains around changed lines.
    entries: Vec<SyntaxEntry>,
    /// Direct callers of changed functions and methods.
    call_sites: Vec<SyntaxCallSite>,
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

/// Detect the language for one source side.
fn detect_side(side: &SourceSide) -> Option<Language> {
    Language::detect(&side.version.path, &side.source)
}

/// Analyze one source side into declaration context.
fn analyze_side(side: SourceSide, language: Language) -> Result<SyntaxSide> {
    let extracted = extract_context(&side.source, language, &side.ranges)?;
    Ok(SyntaxSide {
        path: side.version.path,
        entries: extracted.entries,
        call_sites: extracted.call_sites,
    })
}

/// Extract compact declaration chains around one-based changed line ranges.
pub fn context_for_ranges(
    source: &[u8],
    language: Language,
    ranges: &[LineRange],
) -> Result<Vec<SyntaxEntry>> {
    Ok(extract_context(source, language, ranges)?.entries)
}

/// Extract declaration chains and caller context around changed line ranges.
fn extract_context(
    source: &[u8],
    language: Language,
    ranges: &[LineRange],
) -> Result<ExtractedContext> {
    if ranges.is_empty() {
        return Ok(ExtractedContext {
            entries: Vec::new(),
            call_sites: Vec::new(),
        });
    }

    let mut parser = Parser::new();
    parser
        .set_language(&language.tree_sitter())
        .context("failed to load Tree-sitter grammar")?;
    let tree = parser
        .parse(source, None)
        .context("Tree-sitter failed to parse source")?;
    let root = tree.root_node();
    let definitions = definition_index(root, source, language)?;
    let lines = source.split(|&byte| byte == b'\n').collect::<Vec<_>>();

    let mut entries = Vec::new();
    let mut seen = HashSet::new();
    let mut changed_symbols = HashSet::new();
    'ranges: for range in ranges {
        for row in changed_rows(*range, lines.len()) {
            let Some(focus) = focus_on_row(root, &lines, row) else {
                continue;
            };
            let (key, entry, symbols) = syntax_entry(source, focus, language, row, &definitions);
            if entry.items.is_empty() || !seen.insert(key) {
                continue;
            }
            changed_symbols.extend(symbols);
            entries.push(entry);
            if entries.len() == MAX_ENTRIES_PER_SIDE {
                break 'ranges;
            }
        }
    }

    Ok(ExtractedContext {
        call_sites: call_sites(source, root, language, &definitions, &changed_symbols),
        entries,
    })
}

/// Convert a line range to bounded zero-based rows.
fn changed_rows(range: LineRange, line_count: usize) -> impl Iterator<Item = usize> {
    let start = range.start.saturating_sub(1).min(line_count);
    let end = start.saturating_add(range.count).min(line_count);
    start..end
}

/// Find the smallest named syntax node at a changed row.
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

/// Build the enclosing declaration chain for one changed node.
fn syntax_entry(
    source: &[u8],
    focus: Node<'_>,
    language: Language,
    changed_row: usize,
    definitions: &DefinitionIndex,
) -> (Vec<(usize, usize)>, SyntaxEntry, Vec<String>) {
    let mut items = Vec::new();
    let mut symbols = Vec::new();
    let mut current = Some(focus);
    while let Some(node) = current {
        if let Some(item) = syntax_item(source, node, language, definitions)
            && (!is_control_flow(node) || control_header_contains(node, changed_row))
        {
            if let Some(definition) = definitions
                .get(&node_key(node))
                .filter(|definition| definition.callable)
            {
                symbols.push(definition.name.clone());
            }
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
    (key, SyntaxEntry { items }, symbols)
}

/// Check whether a changed row belongs to a control-flow header.
fn control_header_contains(node: Node<'_>, changed_row: usize) -> bool {
    let header_end = ["body", "consequence"]
        .into_iter()
        .find_map(|field| node.child_by_field_name(field))
        .map_or(node.start_position().row, |body| body.start_position().row);

    (node.start_position().row..=header_end).contains(&changed_row)
}

/// Convert a syntax node into a compact semantic item.
fn syntax_item(
    source: &[u8],
    node: Node<'_>,
    language: Language,
    definitions: &DefinitionIndex,
) -> Option<SyntaxItem> {
    if !is_context_node(node, language, definitions) {
        return None;
    }

    let (start, start_line) = declaration_start(node, language);
    let end = declaration_end(node, language);
    let declaration = compact_source(source.get(start..end)?)?;

    Some(SyntaxItem {
        declaration,
        start_line,
    })
}

/// Check whether a node provides useful enclosing source context.
fn is_context_node(node: Node<'_>, language: Language, definitions: &DefinitionIndex) -> bool {
    if language == Language::Rust {
        return definitions.contains_key(&node_key(node))
            || matches!(
                node.kind(),
                "impl_item"
                    | "function_signature_item"
                    | "field_declaration"
                    | "const_item"
                    | "static_item"
            )
            || is_control_flow(node);
    }

    ["body", "consequence", "name"]
        .into_iter()
        .any(|field| node.child_by_field_name(field).is_some())
        || matches!(
            (language, node.kind()),
            (
                Language::Css,
                "rule_set"
                    | "media_statement"
                    | "supports_statement"
                    | "scope_statement"
                    | "keyframes_statement"
            ) | (
                Language::Html,
                "element" | "script_element" | "style_element"
            )
        )
}

/// Check whether a context node is a control-flow construct.
fn is_control_flow(node: Node<'_>) -> bool {
    matches!(
        node.kind(),
        "if_expression"
            | "match_expression"
            | "for_expression"
            | "while_expression"
            | "loop_expression"
            | "if_statement"
            | "for_statement"
            | "while_statement"
            | "with_statement"
            | "case_statement"
            | "switch_statement"
            | "match_statement"
    )
}

/// Build a source-range index of Rust definitions from the grammar's tag query.
fn definition_index(root: Node<'_>, source: &[u8], language: Language) -> Result<DefinitionIndex> {
    if language != Language::Rust {
        return Ok(DefinitionIndex::new());
    }

    let query = Query::new(&language.tree_sitter(), tree_sitter_rust::TAGS_QUERY)
        .context("failed to load Rust Tree-sitter tag query")?;
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source);
    let mut definitions = DefinitionIndex::new();

    while let Some(query_match) = matches.next() {
        let definition = query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize].starts_with("definition."));
        let name = query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize] == "name");
        let (Some(definition), Some(name)) = (definition, name) else {
            continue;
        };
        let Ok(name) = name.node.utf8_text(source) else {
            continue;
        };
        let capture_name = capture_names[definition.index as usize];
        definitions.insert(
            node_key(definition.node),
            Definition {
                name: name.to_owned(),
                callable: matches!(capture_name, "definition.function" | "definition.method"),
            },
        );
    }

    Ok(definitions)
}

/// Find bounded direct same-file callers of changed Rust functions and methods.
fn call_sites(
    source: &[u8],
    root: Node<'_>,
    language: Language,
    definitions: &DefinitionIndex,
    changed_symbols: &HashSet<String>,
) -> Vec<SyntaxCallSite> {
    if language != Language::Rust || changed_symbols.is_empty() {
        return Vec::new();
    }

    let Ok(query) = Query::new(&language.tree_sitter(), tree_sitter_rust::TAGS_QUERY) else {
        return Vec::new();
    };
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source);
    let mut sites = Vec::new();
    let mut seen = HashSet::new();

    while let Some(query_match) = matches.next() {
        let reference = query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize] == "reference.call");
        let name = query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize] == "name");
        let (Some(reference), Some(name)) = (reference, name) else {
            continue;
        };
        if reference
            .node
            .child_by_field_name("function")
            .is_none_or(|function| function.kind() != "identifier")
        {
            continue;
        }
        let Ok(name) = name.node.utf8_text(source) else {
            continue;
        };
        if !changed_symbols.contains(name) {
            continue;
        }
        let Some((caller_node, caller_symbol)) = enclosing_caller(reference.node, definitions)
        else {
            continue;
        };
        if caller_symbol == name {
            continue;
        }

        let (_, caller, _) = syntax_entry(
            source,
            caller_node,
            language,
            caller_node.start_position().row,
            definitions,
        );
        let Some(call) = source_line(source, reference.node.start_position().row) else {
            continue;
        };
        let key = (node_key(caller_node), reference.node.start_byte());
        if !seen.insert(key) {
            continue;
        }
        sites.push(SyntaxCallSite { caller, call });
        if sites.len() == MAX_CALL_SITES_PER_ENTRY {
            break;
        }
    }

    sites
}

/// Find the callable definition enclosing a node.
fn enclosing_caller<'tree, 'definitions>(
    node: Node<'tree>,
    definitions: &'definitions DefinitionIndex,
) -> Option<(Node<'tree>, &'definitions str)> {
    let mut current = node.parent();
    while let Some(node) = current {
        if let Some(definition) = definitions
            .get(&node_key(node))
            .filter(|definition| definition.callable)
        {
            return Some((node, &definition.name));
        }
        current = node.parent();
    }
    None
}

/// Return a trimmed source line by zero-based row.
fn source_line(source: &[u8], row: usize) -> Option<String> {
    let line = source.split(|&byte| byte == b'\n').nth(row)?;
    let line = String::from_utf8_lossy(line);
    let line = line.trim();
    (!line.is_empty() && line.len() <= MAX_DECLARATION_BYTES).then(|| line.to_owned())
}

/// Return a stable key for one syntax node.
fn node_key(node: Node<'_>) -> (usize, usize) {
    (node.start_byte(), node.end_byte())
}

/// Find the end of a compact declaration.
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

/// Find the start byte and one-based line of a declaration.
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

/// Trim and bound source text for prompt context.
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

    #[test]
    fn changed_statement_identifies_free_function() {
        let entries = rust_context("fn calculate() {\n    let value = 2;\n}\n", &[(2, 1)]);

        assert_eq!(entries.len(), 1);
        assert_eq!(declarations(&entries[0]), ["fn calculate()"]);
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
    }

    #[test]
    fn imports_are_omitted_but_constants_are_retained() {
        assert!(rust_context("use crate::client::Client;\n", &[(1, 1)]).is_empty());

        let entries = rust_context("const TIMEOUT: u64 = 120;\n", &[(1, 1)]);
        assert_eq!(
            entries[0].items.last().unwrap().declaration,
            "const TIMEOUT: u64 ="
        );
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
    fn references_are_not_rendered_as_declarations() {
        let source = "fn terminal_width() -> usize {\n    env::var(\"COLUMNS\").unwrap().parse().unwrap()\n}\n";
        let entries = rust_context(source, &[(2, 1)]);

        assert_eq!(declarations(&entries[0]), ["fn terminal_width() -> usize"]);
    }

    #[test]
    fn changed_function_includes_direct_same_file_caller() {
        let source = "struct ChatProgress;\n\nimpl ChatProgress {\n    fn new() -> Self {\n        let width = thinking_preview_columns();\n        Self\n    }\n}\n\nfn thinking_preview_columns() -> usize {\n    72\n}\n";
        let ranges = [LineRange {
            start: 11,
            count: 1,
        }];
        let extracted = extract_context(source.as_bytes(), Language::Rust, &ranges).unwrap();

        assert_eq!(
            declarations(&extracted.entries[0]),
            ["fn thinking_preview_columns() -> usize"]
        );
        assert_eq!(extracted.call_sites.len(), 1);
        assert_eq!(
            declarations(&extracted.call_sites[0].caller),
            ["impl ChatProgress", "fn new() -> Self"]
        );
        assert_eq!(
            extracted.call_sites[0].call,
            "let width = thinking_preview_columns();"
        );
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

    #[test]
    fn configuration_pairs_are_omitted_as_redundant_diff_context() {
        for (language, source) in [
            (Language::Toml, "[client]\ntimeout = 120\n"),
            (Language::Json, "{\n  \"timeout\": 120\n}\n"),
        ] {
            let entries = context_for_ranges(
                source.as_bytes(),
                language,
                &[LineRange { start: 2, count: 1 }],
            )
            .unwrap();
            assert!(entries.is_empty(), "unexpected context for {language:?}");
        }
    }
}
