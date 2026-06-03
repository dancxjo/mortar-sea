use crate::{impression::Impression, sensation::Sensation};

/// A Faculty notices things.
///
/// Faculties operate at the boundary between the world and cognition. A Faculty
/// may consume raw [`Sensation`]s (e.g. a vision faculty consuming camera
/// frames) and emit new `Sensation`s or [`Impression`]s back into the pipeline.
///
/// No concrete implementations are provided here; this trait defines the
/// contract only.
pub trait Faculty {
    /// Called when the system presents a sensation to this faculty.
    ///
    /// Returns zero or more sensations and impressions derived from the input.
    fn process(&mut self, sensation: &Sensation) -> (Vec<Sensation>, Vec<Impression>);
}
