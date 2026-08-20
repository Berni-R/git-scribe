/// An event-driven terminal spinner.
#[derive(Debug)]
pub struct Spinner {
    frame: usize,
    frames: Vec<&'static str>,
}

impl Spinner {
    /// The frames for a braille spinner: `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`.
    pub const BRAILLE_SPIN: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    /// The frames for an animated ellipsis.
    pub const ELLIPSIS: [&str; 4] = [".  ", ".. ", "...", "   "];
    /// The frames for center dots that get bigger.
    pub const CENTER_DOTS: [&str; 5] = ["∙∙∙", "●∙∙", "∙●∙", "∙∙●", "∙∙∙"];
    /// The frames for a ballon spinner animation: `.oO°Oo.`.
    pub const BALLON: [&str; 7] = [".", "o", "O", "°", "O", "o", "."];
    /// The frames for a spinning half circle.
    pub const HALVES: [&str; 4] = ["◐", "◓", "◑", "◒"];
    /// A small orbital animation.
    pub const ORBIT: [&str; 4] = ["◜", "◝", "◞", "◟"];

    #[must_use]
    pub fn new(frames: Vec<&'static str>) -> Option<Self> {
        if frames.is_empty() {
            None
        } else {
            Some(Self { frame: 0, frames })
        }
    }

    #[must_use]
    pub fn from_slice(frames: &[&'static str]) -> Option<Self> {
        Self::new(frames.to_vec())
    }

    #[must_use]
    pub fn next_frame(&mut self) -> &'static str {
        let frame = self.frames[self.frame];
        self.frame = (self.frame + 1) % self.frames.len();
        frame
    }
}

impl Default for Spinner {
    fn default() -> Self {
        Self::from_slice(&Self::BRAILLE_SPIN).expect("default spinner is non empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spinner_cycles_through_all_frames() {
        let mut spinner = Spinner::default();
        let frames = (0..spinner.frames.len())
            .map(|_| spinner.next_frame())
            .collect::<Vec<_>>();

        assert_eq!(frames, spinner.frames);
        assert_eq!(spinner.next_frame(), spinner.frames[0]);
    }
}
