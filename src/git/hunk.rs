/// A contiguous 1-based source-line range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    /// 1-based source line.
    pub start: usize,

    /// Number of changed lines.
    pub count: usize,
}

/// The changed ranges on each side of a diff hunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffHunk {
    /// Range in the base version, if present.
    pub before: Option<LineRange>,

    /// Range in the prospective version, if present.
    pub after: Option<LineRange>,
}
