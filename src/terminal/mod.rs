mod progress;
mod spinner;
#[allow(clippy::module_inception)]
mod terminal;

/// Build a compact collection of styled terminal text segments.
///
/// Semicolons separate segments so format arguments can retain their usual
/// comma-separated form.
#[macro_export]
macro_rules! segments {
    ($( $style:ident : $format:literal $(, $argument:expr)* );+ $(;)?) => {
        [$(
            $crate::terminal::Segment::text(
                $crate::terminal::TextStyle::$style,
                format_args!($format $(, $argument)*),
            )
        ),+]
    };
}

pub use progress::*;
pub use spinner::*;
pub use terminal::*;
