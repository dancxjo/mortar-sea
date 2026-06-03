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
///
/// ## `occurred_at` vs `observed_at`
///
/// Every sensation carries two timestamps with distinct semantics:
///
/// - **`occurred_at`**: when the underlying real-world event happened. This is
///   the canonical time used for timeline ordering. It may lie in the past
///   relative to processing time—for example, a camera frame timestamped at the
///   moment of capture, a replayed sensor log, or a recalled memory whose
///   original event occurred long ago.
/// - **`observed_at`**: when the cognitive system first became aware of this
///   event. For real-time sensors this is typically very close to `occurred_at`.
///   For delayed delivery, batch replay, or memory recall it will be
///   meaningfully later. `observed_at` is never before `occurred_at`.
///
/// [`TimelineFrame`](crate::timeline::TimelineFrame) always orders entries by
/// `occurred_at`, preserving causal history regardless of when data arrived.
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
    ///
    /// Used as the sort key for [`TimelineFrame`](crate::timeline::TimelineFrame)
    /// ordering. May precede `observed_at` for delayed, replayed, or recalled
    /// data.
    pub occurred_at: DateTime<Utc>,
    /// When the cognitive system first became aware of this event.
    ///
    /// For live sensors this is usually equal to `occurred_at`. For delayed
    /// delivery (batch upload, replay, memory recall) it is later. Never before
    /// `occurred_at`.
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
    use chrono::Duration;
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

    /// A delayed observation has `observed_at` strictly after `occurred_at`.
    ///
    /// This models real-world situations such as a camera delivering footage
    /// after buffering, a batch log upload, or any sensor with latency between
    /// the event and its delivery to the cognitive pipeline.
    #[test]
    fn delayed_observation_has_observed_at_after_occurred_at() {
        let occurred = crate::time::now();
        let observed = occurred + Duration::seconds(10);
        let s = Sensation::new("vision.frame", "camera_0", occurred, observed, json!({}));
        assert!(
            s.observed_at > s.occurred_at,
            "delayed observation: observed_at must be after occurred_at"
        );
    }

    /// A live sensor delivers a sensation in real time.
    ///
    /// When there is no measurable delay, `occurred_at` and `observed_at` may
    /// be equal. This is the degenerate case of a delayed observation where the
    /// delay is zero.
    #[test]
    fn live_sensation_may_have_equal_timestamps() {
        let t = crate::time::now();
        let s = Sensation::new("audio.utterance", "mic", t, t, json!({}));
        assert_eq!(s.occurred_at, s.observed_at);
    }
}
