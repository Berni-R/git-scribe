#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineRange {
    /// 1-based source line.
    pub start: usize,

    /// Number of changed lines.
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffHunk {
    pub before: Option<LineRange>,
    pub after: Option<LineRange>,
}
