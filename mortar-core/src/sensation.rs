use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// A thing entering cognition.
///
/// Sensations are the raw inputs to the cognitive pipeline. They can represent
/// anything the system perceives: a video frame, a spoken utterance, a recalled
/// experience, or a matched face. The `kind` field is deliberately open-ended
/// so that new sensation types can be introduced without changing this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sensation {
    /// Unique identifier for this sensation.
    pub id: Uuid,
    /// Open-ended category string, e.g. `"vision.frame"`, `"audio.utterance"`,
    /// or `"memory.related_experience"`.
    pub kind: String,
    /// Origin of the sensation, e.g. `"camera_0"`, `"microphone"`, `"memory"`.
    pub source: String,
    /// When the underlying event actually happened.
    pub occurred_at: DateTime<Utc>,
    /// When the system first became aware of the event.
    pub observed_at: DateTime<Utc>,
    /// Arbitrary JSON payload carrying sensation-specific data.
    pub payload: Value,
}

impl Sensation {
    /// Create a new Sensation with a freshly generated UUID.
    pub fn new(
        kind: impl Into<String>,
        source: impl Into<String>,
        occurred_at: DateTime<Utc>,
        observed_at: DateTime<Utc>,
        payload: Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            kind: kind.into(),
            source: source.into(),
            occurred_at,
            observed_at,
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sensation_roundtrips_through_json() {
        let s = Sensation::new(
            "vision.frame",
            "camera_0",
            crate::time::now(),
            crate::time::now(),
            json!({"width": 1920, "height": 1080}),
        );
        let json = serde_json::to_string(&s).expect("serialize");
        let back: Sensation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(s.id, back.id);
        assert_eq!(s.kind, back.kind);
        assert_eq!(s.source, back.source);
    }
}
