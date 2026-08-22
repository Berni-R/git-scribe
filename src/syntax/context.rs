use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use anyhow::{Context as _, Result};
use tree_sitter::{Node, Parser, Query, QueryCursor, StreamingIterator};

use crate::{
    GitRepo,
    git::{CommitChange, DiffHunk, FileVersion, LineRange},
};

use super::Language;

/// Maximum public entry points shown for one changed callable.
const MAX_ENTRY_POINTS_PER_CALLABLE: usize = 3;
/// Maximum cross-file call sites shown for one changed callable.
const MAX_EXTERNAL_CALLERS_PER_CALLABLE: usize = 3;
/// Maximum private-call depth followed to reach an entry point.
const MAX_CALL_DEPTH: usize = 4;
/// Maximum bytes retained for one declaration or call line.
const MAX_SOURCE_BYTES: usize = 400;

/// An unchanged public entry point affected by a changed callable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPoint {
    /// Qualified public function or method reached through the call path.
    pub name: String,
    /// Brief documentation for the entry point, when available.
    pub documentation: Option<String>,
    /// Private helpers between the entry point and changed callable.
    pub via: Vec<String>,
}

/// One direct caller in another Rust source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalCallSite {
    /// Repository-relative path of the caller.
    pub path: PathBuf,
    /// Enclosing callable.
    pub caller: String,
    /// Source line containing the call.
    pub call: String,
}

/// A changed callable and the unchanged code it affects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AffectedCode {
    /// Changed function or method declaration.
    pub changed: String,
    /// Whether the changed callable is itself a public entry point.
    pub public: bool,
    /// Brief documentation for the changed callable, when it is public.
    pub documentation: Option<String>,
    /// Unchanged public entry points affected by the change.
    pub entry_points: Vec<EntryPoint>,
    /// Unchanged direct callers in other Rust files.
    pub external_callers: Vec<ExternalCallSite>,
}

/// Same-file public code affected by a changed Rust file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxContext {
    /// File path.
    pub path: PathBuf,
    /// Changed callables with unchanged public entry points.
    pub affected: Vec<AffectedCode>,
}

/// Extract affected code for one prospective-commit change.
///
/// This currently supports Rust only. It identifies changed functions and methods, then follows
/// unchanged, direct same-file calls to public production entry points.
pub fn context_for_change(repo: &GitRepo, change: &CommitChange) -> Result<Option<SyntaxContext>> {
    if change.hunks.is_empty() {
        return Ok(None);
    }

    let Some(side) = load_side(repo, change.after(), &change.hunks, true)?.or(load_side(
        repo,
        change.before(),
        &change.hunks,
        false,
    )?) else {
        return Ok(None);
    };
    if Language::detect(&side.version.path, &side.source) != Some(Language::Rust) {
        return Ok(None);
    }

    let affected = affected_code(repo, &side.version.path, &side.source, &side.ranges)?;
    if affected.is_empty() {
        Ok(None)
    } else {
        Ok(Some(SyntaxContext {
            path: side.version.path,
            affected,
        }))
    }
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
type Definitions = HashMap<(usize, usize), Definition>;

/// A callable definition identified by the Rust tag query.
#[derive(Clone)]
struct Definition {
    /// Declared name.
    name: String,
    /// Qualified callable name.
    label: String,
    /// Declaration without its body.
    declaration: String,
    /// Zero-based start row.
    start_row: usize,
    /// Zero-based end row.
    end_row: usize,
    /// Whether the definition is a direct callable.
    callable: bool,
    /// Whether the callable is publicly visible.
    public: bool,
    /// Brief preceding doc comment, if any.
    documentation: Option<String>,
    /// Enclosing implementation type for methods.
    impl_type: Option<String>,
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

/// Find changed public APIs and unchanged callers of changed Rust functions and methods.
fn affected_code(
    repo: &GitRepo,
    path: &std::path::Path,
    source: &[u8],
    ranges: &[LineRange],
) -> Result<Vec<AffectedCode>> {
    affected_code_with_external_callers(source, ranges, |changed| {
        external_callers(repo, path, changed)
    })
}

/// Find affected code without cross-file callers.
#[cfg(test)]
fn local_affected_code(source: &[u8], ranges: &[LineRange]) -> Result<Vec<AffectedCode>> {
    affected_code_with_external_callers(source, ranges, |_| Ok(HashMap::new()))
}

/// Extract affected code, attaching cross-file callers supplied by `find_external_callers`.
fn affected_code_with_external_callers(
    source: &[u8],
    ranges: &[LineRange],
    find_external_callers: impl FnOnce(
        &[((usize, usize), Definition)],
    ) -> Result<HashMap<(usize, usize), Vec<ExternalCallSite>>>,
) -> Result<Vec<AffectedCode>> {
    let mut parser = Parser::new();
    parser
        .set_language(&Language::Rust.tree_sitter())
        .context("failed to load the Rust Tree-sitter grammar")?;
    let tree = parser
        .parse(source, None)
        .context("Tree-sitter failed to parse Rust source")?;
    let root = tree.root_node();
    let definitions = definitions(root, source)?;
    let mut changed = definitions
        .iter()
        .filter(|(_, definition)| definition.callable && overlaps_changed_lines(definition, ranges))
        .map(|(key, definition)| (*key, definition.clone()))
        .collect::<Vec<_>>();
    changed.sort_unstable_by_key(|(key, _)| key.0);

    let callable_index = callable_index(&definitions);
    let callers = reverse_call_graph(source, root, &definitions, &callable_index, ranges);
    let external_callers = find_external_callers(&changed)?;
    Ok(changed
        .into_iter()
        .filter_map(|(key, definition)| {
            let entry_points = entry_points(key, &callers, &definitions);
            let external_callers = external_callers.get(&key).cloned().unwrap_or_default();
            (definition.public || !entry_points.is_empty() || !external_callers.is_empty())
                .then_some(AffectedCode {
                    changed: definition.declaration,
                    public: definition.public,
                    documentation: definition.documentation,
                    entry_points,
                    external_callers,
                })
        })
        .collect())
}

/// Find bounded direct callers of changed functions and methods in other Rust files.
fn external_callers(
    repo: &GitRepo,
    changed_path: &std::path::Path,
    changed: &[((usize, usize), Definition)],
) -> Result<HashMap<(usize, usize), Vec<ExternalCallSite>>> {
    let mut callers = HashMap::new();
    for path in repo.working_tree_files()? {
        if path == changed_path {
            continue;
        }
        let Some(source) = repo.index_file(&path)? else {
            continue;
        };
        if Language::detect(&path, &source) != Some(Language::Rust) {
            continue;
        }
        collect_external_callers(&path, &source, changed, &mut callers)?;
    }
    Ok(callers)
}

/// Collect direct external calls from one Rust source file.
fn collect_external_callers(
    path: &std::path::Path,
    source: &[u8],
    changed: &[((usize, usize), Definition)],
    callers: &mut HashMap<(usize, usize), Vec<ExternalCallSite>>,
) -> Result<()> {
    let mut parser = Parser::new();
    parser
        .set_language(&Language::Rust.tree_sitter())
        .context("failed to load the Rust Tree-sitter grammar")?;
    let tree = parser
        .parse(source, None)
        .context("Tree-sitter failed to parse Rust source")?;
    let root = tree.root_node();
    let definitions = definitions(root, source)?;
    let query = Query::new(
        &Language::Rust.tree_sitter(),
        "(call_expression function: (identifier) @function) @call\n\
         (call_expression function: (scoped_identifier path: (identifier) @type name: (identifier) @method)) @call",
    )
    .context("failed to load the Rust call query")?;
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source);

    while let Some(query_match) = matches.next() {
        let call = query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize] == "call");
        let Some(call) = call else {
            continue;
        };
        if is_test_context(call.node, source) {
            continue;
        }
        let Some((_caller_node, caller)) = enclosing_caller(call.node, &definitions) else {
            continue;
        };
        let target = changed.iter().find(|(_, definition)| {
            external_call_matches(query_match, capture_names, source, definition)
        });
        let Some(((start, end), _)) = target else {
            continue;
        };
        let sites = callers.entry((*start, *end)).or_default();
        if sites.len() == MAX_EXTERNAL_CALLERS_PER_CALLABLE {
            continue;
        }
        let Some(call_line) = source_line(source, call.node.start_position().row) else {
            continue;
        };
        sites.push(ExternalCallSite {
            path: path.to_path_buf(),
            caller: caller.label.clone(),
            call: call_line,
        });
    }
    Ok(())
}

/// Check whether a query match calls a changed free function or associated method.
fn external_call_matches(
    query_match: &tree_sitter::QueryMatch<'_, '_>,
    capture_names: &[&str],
    source: &[u8],
    definition: &Definition,
) -> bool {
    let capture_text = |name| {
        query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize] == name)
            .and_then(|capture| capture.node.utf8_text(source).ok())
    };
    match &definition.impl_type {
        None => capture_text("function") == Some(definition.name.as_str()),
        Some(impl_type) => {
            capture_text("type") == Some(impl_type.as_str())
                && capture_text("method") == Some(definition.name.as_str())
        }
    }
}

/// Index Rust definitions from the grammar's maintained tag query.
fn definitions(root: Node<'_>, source: &[u8]) -> Result<Definitions> {
    let query = Query::new(&Language::Rust.tree_sitter(), tree_sitter_rust::TAGS_QUERY)
        .context("failed to load the Rust Tree-sitter tag query")?;
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source);
    let mut definitions = Definitions::new();

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
        let (Ok(name), Some(declaration)) = (
            name.node.utf8_text(source),
            declaration(source, definition.node),
        ) else {
            continue;
        };
        let capture_name = capture_names[definition.index as usize];
        let impl_type = enclosing_impl_type(source, definition.node);
        definitions.insert(
            node_key(definition.node),
            Definition {
                name: name.to_owned(),
                label: callable_label(impl_type.as_deref(), name),
                public: definition
                    .node
                    .utf8_text(source)
                    .is_ok_and(|text| text.trim_start().starts_with("pub ")),
                documentation: doc_summary(source, definition.node),
                declaration,
                start_row: definition.node.start_position().row,
                end_row: definition.node.end_position().row,
                callable: matches!(capture_name, "definition.function" | "definition.method"),
                impl_type,
            },
        );
    }

    Ok(definitions)
}

/// Index uniquely resolvable free functions and methods within an implementation.
struct CallableIndex {
    functions: HashMap<String, (usize, usize)>,
    methods: HashMap<(String, String), (usize, usize)>,
}

fn callable_index(definitions: &Definitions) -> CallableIndex {
    let mut functions = HashMap::new();
    let mut duplicate_functions = HashSet::new();
    let mut methods = HashMap::new();
    let mut duplicate_methods = HashSet::new();
    for (key, definition) in definitions {
        if !definition.callable {
            continue;
        }
        if let Some(impl_type) = &definition.impl_type {
            let method = (impl_type.clone(), definition.name.clone());
            if methods.insert(method.clone(), *key).is_some() {
                duplicate_methods.insert(method);
            }
        } else if functions.insert(definition.name.clone(), *key).is_some() {
            duplicate_functions.insert(definition.name.clone());
        }
    }
    functions.retain(|name, _| !duplicate_functions.contains(name));
    methods.retain(|method, _| !duplicate_methods.contains(method));
    CallableIndex { functions, methods }
}

/// Build a reverse graph of unchanged direct production calls.
fn reverse_call_graph(
    source: &[u8],
    root: Node<'_>,
    definitions: &Definitions,
    callable_index: &CallableIndex,
    ranges: &[LineRange],
) -> HashMap<(usize, usize), Vec<(usize, usize)>> {
    let Ok(query) = Query::new(&Language::Rust.tree_sitter(), tree_sitter_rust::TAGS_QUERY) else {
        return HashMap::new();
    };
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, source);
    let mut callers = HashMap::<(usize, usize), Vec<(usize, usize)>>::new();
    let mut seen = HashSet::new();

    while let Some(query_match) = matches.next() {
        let reference = query_match
            .captures
            .iter()
            .find(|capture| capture_names[capture.index as usize] == "reference.call");
        let Some(reference) = reference else {
            continue;
        };
        if changed_line(reference.node.start_position().row, ranges)
            || is_test_context(reference.node, source)
        {
            continue;
        }
        let Some((caller_node, caller_definition)) = enclosing_caller(reference.node, definitions)
        else {
            continue;
        };
        if overlaps_changed_lines(caller_definition, ranges) {
            continue;
        }
        let Some(target) = called_target(reference.node, source, caller_definition, callable_index)
        else {
            continue;
        };
        let caller = node_key(caller_node);
        if target == caller {
            continue;
        }
        let key = (target, caller, reference.node.start_byte());
        if !seen.insert(key) {
            continue;
        }
        callers.entry(target).or_default().push(caller);
    }

    callers
}

/// Resolve a direct free-function call or `self.method()` call within the same implementation.
fn called_target(
    call: Node<'_>,
    source: &[u8],
    caller: &Definition,
    callable_index: &CallableIndex,
) -> Option<(usize, usize)> {
    let function = call.child_by_field_name("function")?;
    match function.kind() {
        "identifier" => callable_index
            .functions
            .get(function.utf8_text(source).ok()?)
            .copied(),
        "field_expression" if is_self_receiver(function) => {
            let method = function
                .child_by_field_name("field")?
                .utf8_text(source)
                .ok()?;
            callable_index
                .methods
                .get(&(caller.impl_type.clone()?, method.to_owned()))
                .copied()
        }
        _ => None,
    }
}

/// Check whether a field expression has `self` as its receiver.
fn is_self_receiver(node: Node<'_>) -> bool {
    node.child_by_field_name("value")
        .is_some_and(|argument| argument.kind() == "self")
}

/// Follow callers until reaching unchanged public entry points.
fn entry_points(
    changed: (usize, usize),
    callers: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    definitions: &Definitions,
) -> Vec<EntryPoint> {
    let mut paths = Vec::new();
    let mut path = vec![changed];
    collect_entry_paths(changed, callers, definitions, &mut path, &mut paths);

    let mut seen = HashSet::new();
    paths
        .into_iter()
        .filter_map(|path| {
            let (&entry_point, helpers) = path.split_last()?;
            let definition = definitions.get(&entry_point)?;
            let via = helpers
                .iter()
                .skip(1)
                .rev()
                .filter_map(|key| definitions.get(key))
                .map(|definition| definition.name.clone())
                .collect::<Vec<_>>();
            let key = (entry_point, via.clone());
            seen.insert(key).then_some(EntryPoint {
                name: definition.label.clone(),
                documentation: definition.documentation.clone(),
                via,
            })
        })
        .take(MAX_ENTRY_POINTS_PER_CALLABLE)
        .collect()
}

/// Recursively collect bounded reverse call paths ending at public callables.
fn collect_entry_paths(
    current: (usize, usize),
    callers: &HashMap<(usize, usize), Vec<(usize, usize)>>,
    definitions: &Definitions,
    path: &mut Vec<(usize, usize)>,
    paths: &mut Vec<Vec<(usize, usize)>>,
) {
    if path.len() > 1
        && definitions
            .get(&current)
            .is_some_and(|definition| definition.public)
    {
        paths.push(path.clone());
        return;
    }
    if path.len() == MAX_CALL_DEPTH {
        return;
    }
    for &caller in callers.get(&current).into_iter().flatten() {
        if path.contains(&caller) {
            continue;
        }
        path.push(caller);
        collect_entry_paths(caller, callers, definitions, path, paths);
        path.pop();
    }
}

/// Check whether a definition overlaps any changed source line.
fn overlaps_changed_lines(definition: &Definition, ranges: &[LineRange]) -> bool {
    ranges.iter().any(|range| {
        let start = range.start.saturating_sub(1);
        let end = start.saturating_add(range.count);
        start <= definition.end_row && definition.start_row < end
    })
}

/// Check whether a zero-based source row belongs to a changed range.
fn changed_line(row: usize, ranges: &[LineRange]) -> bool {
    ranges.iter().any(|range| {
        let start = range.start.saturating_sub(1);
        (start..start.saturating_add(range.count)).contains(&row)
    })
}

/// Find the callable definition enclosing a node.
fn enclosing_caller<'tree, 'definitions>(
    node: Node<'tree>,
    definitions: &'definitions Definitions,
) -> Option<(Node<'tree>, &'definitions Definition)> {
    let mut current = node.parent();
    while let Some(node) = current {
        if let Some(definition) = definitions
            .get(&node_key(node))
            .filter(|definition| definition.callable)
        {
            return Some((node, definition));
        }
        current = node.parent();
    }
    None
}

/// Check whether a node is enclosed by a conventional Rust test module or test attribute.
fn is_test_context(node: Node<'_>, source: &[u8]) -> bool {
    let mut current = Some(node);
    while let Some(node) = current {
        if node.kind() == "mod_item"
            && node
                .child_by_field_name("name")
                .and_then(|name| name.utf8_text(source).ok())
                == Some("tests")
            || matches!(node.kind(), "mod_item" | "function_item")
                && has_test_attribute(node, source)
        {
            return true;
        }
        current = node.parent();
    }
    false
}

/// Check the attributes immediately preceding a declaration for test markers.
fn has_test_attribute(mut node: Node<'_>, source: &[u8]) -> bool {
    while let Some(attribute) = node
        .prev_named_sibling()
        .filter(|sibling| sibling.kind() == "attribute_item")
    {
        let text = String::from_utf8_lossy(&source[attribute.byte_range()]);
        let compact = text.replace(char::is_whitespace, "");
        if compact.contains("#[test]")
            || compact.contains("::test]")
            || compact.contains("cfg(test)")
        {
            return true;
        }
        node = attribute;
    }
    false
}

/// Return a declaration without its body.
fn declaration(source: &[u8], node: Node<'_>) -> Option<String> {
    let start = declaration_start(node);
    let end = node
        .child_by_field_name("body")
        .map_or(node.end_byte(), |body| body.start_byte());
    compact_source(source.get(start..end)?)
}

/// Return the first attribute or declaration node.
fn declaration_node(mut node: Node<'_>) -> Node<'_> {
    while let Some(attribute) = node
        .prev_named_sibling()
        .filter(|sibling| sibling.kind() == "attribute_item")
    {
        node = attribute;
    }
    node
}

/// Include Rust attributes immediately preceding a declaration.
fn declaration_start(node: Node<'_>) -> usize {
    declaration_node(node).start_byte()
}

/// Return the enclosing implementation type for a method.
fn enclosing_impl_type(source: &[u8], node: Node<'_>) -> Option<String> {
    let mut current = node.parent();
    while let Some(parent) = current {
        if parent.kind() == "impl_item" {
            if let Some(kind) = parent.child_by_field_name("type")
                && let Ok(kind) = kind.utf8_text(source)
            {
                return Some(kind.trim().to_owned());
            }
            break;
        }
        current = parent.parent();
    }
    None
}

/// Return a callable name qualified with its enclosing implementation type.
fn callable_label(impl_type: Option<&str>, name: &str) -> String {
    impl_type.map_or_else(
        || name.to_owned(),
        |impl_type| format!("{impl_type}::{name}"),
    )
}

/// Return the first sentence of a declaration's immediately preceding doc comment.
fn doc_summary(source: &[u8], node: Node<'_>) -> Option<String> {
    let lines = source.split(|&byte| byte == b'\n').collect::<Vec<_>>();
    let mut docs = Vec::new();
    let start_row = declaration_node(node).start_position().row;
    for line in lines.get(..start_row)?.iter().rev() {
        let line = String::from_utf8_lossy(line);
        let Some(doc) = line.trim_start().strip_prefix("///") else {
            break;
        };
        docs.push(doc.trim().to_owned());
    }
    docs.reverse();
    let documentation = docs.join(" ");
    let summary = documentation
        .split_once('.')
        .map_or(documentation.as_str(), |(sentence, _)| sentence)
        .trim();
    (!summary.is_empty()).then(|| summary.to_owned())
}

/// Trim and bound source text for prompt context.
fn compact_source(source: &[u8]) -> Option<String> {
    let source = String::from_utf8_lossy(source);
    let text = source.trim().trim_end_matches('{').trim();
    (!text.is_empty() && text.len() <= MAX_SOURCE_BYTES).then(|| text.to_owned())
}

/// Return a compact source line by zero-based row.
fn source_line(source: &[u8], row: usize) -> Option<String> {
    compact_source(source.split(|&byte| byte == b'\n').nth(row)?)
}

/// Return a stable key for one syntax node.
fn node_key(node: Node<'_>) -> (usize, usize) {
    (node.start_byte(), node.end_byte())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affected(source: &str, ranges: &[(usize, usize)]) -> Vec<AffectedCode> {
        let ranges = ranges
            .iter()
            .map(|&(start, count)| LineRange { start, count })
            .collect::<Vec<_>>();
        local_affected_code(source.as_bytes(), &ranges).unwrap()
    }

    #[test]
    fn finds_public_entry_point_reaching_changed_function() {
        let source = "/// Start preview rendering.\npub fn start() {\n    configure();\n}\n\nfn configure() {\n    let width = thinking_preview_columns();\n}\n\nfn thinking_preview_columns() -> usize {\n    72\n}\n";

        assert_eq!(
            affected(source, &[(12, 1)]),
            [AffectedCode {
                changed: "fn thinking_preview_columns() -> usize".to_owned(),
                public: false,
                documentation: None,
                entry_points: vec![EntryPoint {
                    name: "start".to_owned(),
                    documentation: Some("Start preview rendering".to_owned()),
                    via: vec!["configure".to_owned()],
                }],
                external_callers: Vec::new(),
            }]
        );
    }

    #[test]
    fn omits_changed_and_test_callers() {
        let source = "fn caller() {\n    changed();\n}\n\nfn changed() {}\n\n#[cfg(test)]\nmod tests {\n    fn checks() {\n        changed();\n    }\n}\n";

        assert!(affected(source, &[(2, 1), (5, 1)]).is_empty());
    }

    #[test]
    fn omits_changed_functions_without_callers() {
        assert!(affected("fn changed() {\n    work();\n}\n", &[(2, 1)]).is_empty());
    }

    #[test]
    fn reports_public_changed_callable_without_callers() {
        let source = "/// Parse the supplied input.\n#[must_use]\npub fn parse() -> String {\n    String::new()\n}\n";

        assert_eq!(
            affected(source, &[(4, 1)]),
            [AffectedCode {
                changed: "#[must_use]\npub fn parse() -> String".to_owned(),
                public: true,
                documentation: Some("Parse the supplied input".to_owned()),
                entry_points: Vec::new(),
                external_callers: Vec::new(),
            }]
        );
    }

    #[test]
    fn follows_self_method_calls_within_the_same_implementation() {
        let source = "impl First {\n    /// Run the first implementation.\n    #[must_use]\n    pub fn run(&self) -> usize {\n        self.update()\n    }\n\n    fn update(&self) -> usize {\n        1\n    }\n}\n\nimpl Second {\n    fn update(&self) -> usize {\n        2\n    }\n}\n";

        assert_eq!(
            affected(source, &[(9, 1)]),
            [AffectedCode {
                changed: "fn update(&self) -> usize".to_owned(),
                public: false,
                documentation: None,
                entry_points: vec![EntryPoint {
                    name: "First::run".to_owned(),
                    documentation: Some("Run the first implementation".to_owned()),
                    via: Vec::new(),
                }],
                external_callers: Vec::new(),
            }]
        );
    }
}
