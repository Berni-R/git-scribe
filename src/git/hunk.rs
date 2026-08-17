use anyhow::{Context as _, Result, bail};

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

pub(super) fn parse_hunks(diff: &str) -> Result<Vec<DiffHunk>> {
    diff.lines()
        .filter(|line| line.starts_with("@@ "))
        .map(parse_hunk)
        .collect()
}

fn parse_hunk(header: &str) -> Result<DiffHunk> {
    let mut fields = header.split_whitespace();

    if fields.next() != Some("@@") {
        bail!("invalid Git hunk header: {header}");
    }

    let before = fields
        .next()
        .context("Git hunk header is missing the old range")?;

    let after = fields
        .next()
        .context("Git hunk header is missing the new range")?;

    if fields.next() != Some("@@") {
        bail!("invalid Git hunk header: {header}");
    }

    Ok(DiffHunk {
        before: parse_range(before, '-')?,
        after: parse_range(after, '+')?,
    })
}

fn parse_range(text: &str, prefix: char) -> Result<Option<LineRange>> {
    let text = text
        .strip_prefix(prefix)
        .with_context(|| format!("invalid Git hunk range: {text}"))?;

    let (start, count) = match text.split_once(',') {
        Some((start, count)) => (start.parse::<usize>()?, count.parse::<usize>()?),
        None => (text.parse::<usize>()?, 1),
    };

    Ok((count != 0).then_some(LineRange { start, count }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_modified_hunk() -> Result<()> {
        let hunks = parse_hunks("@@ -10,2 +10,3 @@\n")?;

        assert_eq!(
            hunks,
            vec![DiffHunk {
                before: Some(LineRange {
                    start: 10,
                    count: 2,
                }),
                after: Some(LineRange {
                    start: 10,
                    count: 3,
                }),
            }]
        );

        Ok(())
    }

    #[test]
    fn parses_insertion_and_deletion() -> Result<()> {
        let hunks = parse_hunks(
            "@@ -10,0 +11,3 @@\n\
             @@ -20,2 +22,0 @@\n",
        )?;

        assert_eq!(hunks[0].before, None);
        assert_eq!(
            hunks[0].after,
            Some(LineRange {
                start: 11,
                count: 3,
            })
        );

        assert_eq!(
            hunks[1].before,
            Some(LineRange {
                start: 20,
                count: 2,
            })
        );
        assert_eq!(hunks[1].after, None);

        Ok(())
    }
}
