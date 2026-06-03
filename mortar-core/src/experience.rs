use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Meaning extracted from one or more impressions.
///
/// Where an [`Impression`](crate::impression::Impression) *observes*, an
/// Experience *explains*. It answers "what does this mean?" rather than
/// "what did I notice?". Experiences are the output of understanding and the
/// primary content stored in, and later retrieved from, memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Unique identifier for this experience.
    pub id: Uuid,
    /// The impressions from which this meaning was extracted.
    pub impression_ids: Vec<Uuid>,
    /// When this meaning was understood.
    pub occurred_at: DateTime<Utc>,
    /// When the system recorded this experience.
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
