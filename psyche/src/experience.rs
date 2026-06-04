use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Meaning extracted from one or more impressions.
///
/// Where an [`Impression`](crate::impression::Impression) *observes*, an
/// Experience *explains*. It answers "what does this mean?" rather than
/// "what did I notice?". Experiences are the output of understanding and the
/// primary content stored in, and later retrieved from, memory.
///
/// ## `occurred_at` vs `observed_at`
///
/// - **`occurred_at`**: when the events that led to this meaning took place,
///   typically inherited from the impression(s) that were synthesized. For
///   retrospective analysis or memory recall the experience may be created long
///   after the underlying events, yet its `occurred_at` reflects those original
///   events.
/// - **`observed_at`**: when the wit produced this experience. This is usually
///   close to the time the wit ran, which may be after any deliberate delay.
///
/// When an experience is recalled from memory and re-enters the pipeline as a
/// `"memory.related_experience"` sensation, the recollection sensation occurs
/// at recall time (now). The original experience timestamps remain available in
/// payload metadata (`original_occurred_at`, `original_observed_at`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Unique identifier for this experience.
    pub id: Uuid,
    /// The impressions from which this meaning was extracted.
    pub impression_ids: Vec<Uuid>,
    /// When this meaning was understood, inherited from the source impressions.
    ///
    /// Used as the sort key in [`TimelineFrame`](crate::timeline::TimelineFrame)
    /// and preserved verbatim in memory-recall payload metadata.
    pub occurred_at: DateTime<Utc>,
    /// When the wit produced this experience.
    ///
    /// For synchronous wits this is close to `occurred_at`. For asynchronous
    /// or retrospective wits it may be significantly later.
    pub observed_at: DateTime<Utc>,
    /// A natural-language explanation, e.g. `"A visitor may have arrived."`.
    pub what: String,
}

impl Experience {
    /// Create a new Experience with a freshly generated UUID.
    pub fn new(
        impression_ids: Vec<Uuid>,
        occurred_at: DateTime<Utc>,
        observed_at: DateTime<Utc>,
        what: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            impression_ids,
            occurred_at,
            observed_at,
            what: what.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{impression::Impression, sensation::Sensation, time::now};
    use serde_json::json;

    #[test]
    fn experience_references_impressions() {
        let s = Sensation::new("audio.utterance", "mic", now(), now(), json!({}));
        let imp = Impression::new(vec![s.id], now(), now(), "The speaker said hello.");
        let exp = Experience::new(vec![imp.id], now(), now(), "Someone greeted the system.");
        assert!(exp.impression_ids.contains(&imp.id));
    }

    #[test]
    fn experience_roundtrips_through_json() {
        let t = now();
        let exp = Experience::new(
            vec![Uuid::new_v4()],
            t,
            t,
            "A package appears to have been delivered.",
        );
        let json = serde_json::to_string(&exp).expect("serialize");
        let back: Experience = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(exp.id, back.id);
        assert_eq!(exp.what, back.what);
    }
}
