use crate::{experience::Experience, timeline::TimelineFrame};

/// A Wit understands things over time.
///
/// Wits consume a [`TimelineFrame`] and produce [`Experience`]s—meaning
/// extracted from the ordered stream of sensations and impressions. A Wit
/// might, for example, notice that a sequence of face observations followed by
/// a speech observation implies a greeting.
///
/// No concrete implementations are provided here; this trait defines the
/// contract only. In particular, language models and inference engines are
/// explicitly out of scope for this crate.
pub trait Wit {
    /// Given a frame of recent cognitive events, derive zero or more
    /// experiences (interpretations).
    fn interpret(&mut self, frame: &TimelineFrame) -> Vec<Experience>;
}
