use std::collections::HashSet;

use anyhow::{Context as _, Result};
use tree_sitter::{Node, Parser, Point};

use super::Language;

const MAX_ENCLOSING_LINES: usize = 100;
const MAX_AST_DEPTH: usize = 8;
const MAX_FULL_EXCERPT_LINES: usize = 36;
const MAX_EXCERPT_BYTES: usize = 5_000;

/// Extract Tree-sitter context around changed source ranges.
///
/// Each range is `(start_line, line_count)` with 1-based line numbers.
pub fn context_for_ranges(
    source: &[u8],
    language: Language,
    ranges: &[(usize, usize)],
) -> Result<Vec<String>> {
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
    let lines = source
        .split_inclusive(|&byte| byte == b'\n')
        .collect::<Vec<_>>();

    let mut result = Vec::new();
    let mut seen = HashSet::new();

    for &(line, count) in ranges {
        if count == 0 {
            continue;
        }

        let Some(point) = point_on_line(&lines, line) else {
            continue;
        };

        let Some(focus) = root.named_descendant_for_point_range(point, point) else {
            continue;
        };

        let enclosing = enclosing_node(root, focus, language);
        let key = enclosing.byte_range();

        if !seen.insert((key.start, key.end)) {
            continue;
        }

        let name = name_of(source, enclosing);
        let kind = match name {
            Some(name) => format!("{} {name}", enclosing.kind()),
            None => enclosing.kind().to_owned(),
        };

        result.push(format!(
            "hunk at line {line}; enclosing {kind} (lines {}-{})\n\
             AST: {}\n\
             ```{}\n{}\n```",
            enclosing.start_position().row + 1,
            enclosing.end_position().row + 1,
            ast_path(source, focus, enclosing),
            language.fence_name(),
            excerpt(source, enclosing, line - 1),
        ));
    }

    Ok(result)
}

fn point_on_line(lines: &[&[u8]], line: usize) -> Option<Point> {
    if lines.is_empty() {
        return None;
    }

    let row = line.saturating_sub(1).min(lines.len() - 1);
    let raw = lines[row].strip_suffix(b"\n").unwrap_or(lines[row]);
    let raw = raw.strip_suffix(b"\r").unwrap_or(raw);

    let indent = raw
        .iter()
        .take_while(|&&byte| matches!(byte, b' ' | b'\t'))
        .count();

    Some(Point::new(row, indent.min(raw.len().saturating_sub(1))))
}

fn enclosing_node<'tree>(root: Node<'tree>, focus: Node<'tree>, language: Language) -> Node<'tree> {
    let preferred = enclosing_kinds(language);

    let mut node = Some(focus);
    let mut fallback = focus;

    while let Some(current) = node {
        if current == root {
            break;
        }

        let line_count = current
            .end_position()
            .row
            .saturating_sub(current.start_position().row);

        if line_count < MAX_ENCLOSING_LINES {
            fallback = current;
        }

        if preferred.contains(&current.kind()) {
            return current;
        }

        node = current.parent();
    }

    fallback
}

fn name_of(source: &[u8], node: Node<'_>) -> Option<String> {
    let name = node.child_by_field_name("name")?;
    Some(String::from_utf8_lossy(&source[name.byte_range()]).into_owned())
}

fn ast_path(source: &[u8], focus: Node<'_>, enclosing: Node<'_>) -> String {
    let mut nodes = Vec::new();
    let mut node = Some(focus);

    while let Some(current) = node {
        nodes.push(current);

        if current == enclosing || nodes.len() >= MAX_AST_DEPTH {
            break;
        }

        node = current.parent();
    }

    nodes.reverse();

    nodes
        .into_iter()
        .map(|node| match name_of(source, node) {
            Some(name) => format!("{} {name}", node.kind()),
            None => node.kind().to_owned(),
        })
        .collect::<Vec<_>>()
        .join(" > ")
}

fn excerpt(source: &[u8], node: Node<'_>, focus_row: usize) -> String {
    let source = String::from_utf8_lossy(source);
    let lines = source.lines().collect::<Vec<_>>();

    let start = node.start_position().row.saturating_sub(3);
    let end = (node.end_position().row + 1).min(lines.len());

    if start >= end {
        return String::new();
    }

    if end - start <= MAX_FULL_EXCERPT_LINES {
        let text = lines[start..end].join("\n");

        if text.len() <= MAX_EXCERPT_BYTES {
            return text;
        }
    }

    let focus_row = focus_row.clamp(start, end - 1);
    let lo = focus_row.saturating_sub(14).max(start);
    let hi = (focus_row + 15).min(end);

    let mut result = Vec::new();

    let prefix_end = (start + 2).min(lo);
    result.extend_from_slice(&lines[start..prefix_end]);

    if lo > prefix_end {
        result.push("    ...");
    }

    result.extend_from_slice(&lines[lo..hi]);

    if hi < end {
        result.push("    ...");

        let suffix_start = end.saturating_sub(2).max(hi);
        result.extend_from_slice(&lines[suffix_start..end]);
    }

    let text = result.join("\n");

    if text.len() <= MAX_EXCERPT_BYTES {
        text
    } else {
        let end = text.floor_char_boundary(MAX_EXCERPT_BYTES);
        text[..end].to_owned()
    }
}

fn enclosing_kinds(language: Language) -> &'static [&'static str] {
    match language {
        Language::Rust => &[
            "function_item",
            "impl_item",
            "trait_item",
            "struct_item",
            "enum_item",
            "mod_item",
        ],

        Language::C => &[
            "function_definition",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
        ],

        Language::Cpp => &[
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "template_declaration",
            "enum_specifier",
        ],

        Language::Python => &[
            "function_definition",
            "class_definition",
            "decorated_definition",
        ],

        Language::Swift => &[
            "function_declaration",
            "class_declaration",
            "struct_declaration",
            "enum_declaration",
            "protocol_declaration",
            "extension_declaration",
            "initializer_declaration",
            "subscript_declaration",
        ],

        Language::Bash => &["function_definition"],

        Language::JavaScript => &[
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "class_declaration",
            "function_expression",
            "generator_function",
            "arrow_function",
        ],

        Language::TypeScript | Language::Tsx => &[
            "function_declaration",
            "generator_function_declaration",
            "method_definition",
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "internal_module",
            "function_expression",
            "generator_function",
            "arrow_function",
        ],

        Language::Css => &[
            "rule_set",
            "media_statement",
            "supports_statement",
            "scope_statement",
            "keyframes_statement",
        ],

        Language::Html => &["element", "script_element", "style_element"],

        Language::Toml => &["table", "table_array_element"],

        Language::Json => &["object", "array"],
    }
}
