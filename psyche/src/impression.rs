use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

const DEFAULT_IMPRESSION_KIND: &str = "observation.unknown";
const DEFAULT_IMPRESSION_CONFIDENCE: f32 = 0.5;

/// An observation about one or more sensations.
///
/// Impressions describe *what was noticed*, not what it means. They answer
/// questions like "what did I see?" or "what was said?" rather than "what does
/// this imply?". That interpretive step belongs to [`Experience`](crate::experience::Experience).
///
/// ## `occurred_at` vs `observed_at`
///
/// - **`occurred_at`**: when the noticed event happened, typically inherited
///   from the source sensation(s). A faculty processing a delayed sensation
///   propagates the original `occurred_at` so the impression is placed
///   correctly in historical order.
/// - **`observed_at`**: when the faculty produced this impression. This matches
///   the source sensation's `observed_at` for faculties that run synchronously,
///   but may differ for asynchronous or batch-processing faculties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impression {
    /// Unique identifier for this impression.
    pub id: Uuid,
    /// Natural-language observation text (claim, not fact).
    #[serde(default, alias = "how")]
    pub text: String,
    /// Canonical classification kind, e.g. `"vision.face"` or
    /// `"recognition.person"`.
    #[serde(default = "default_impression_kind")]
    pub kind: String,
    /// When the observation occurred, inherited from the source sensation.
    ///
    /// Used as the sort key in [`TimelineFrame`](crate::timeline::TimelineFrame).
    pub occurred_at: DateTime<Utc>,
    /// When the faculty produced this impression.
    ///
    /// For synchronous processing this equals the source sensation's
    /// `observed_at`. May be later for asynchronous or deferred faculties.
    pub observed_at: DateTime<Utc>,
    /// Producing faculty name, e.g. `"Face Faculty"`.
    #[serde(default)]
    pub faculty: String,
    /// Referenced sensation IDs this claim is about.
    #[serde(default, alias = "sensation_ids")]
    pub about: Vec<Uuid>,
    /// Confidence score in [0.0, 1.0].
    #[serde(default = "default_impression_confidence")]
    pub confidence: f32,
    /// Structured auxiliary metadata for downstream reasoning/scheduling.
    #[serde(default)]
    pub payload: Value,
}

impl Impression {
    /// Create a new Impression with a freshly generated UUID.
    pub fn new(
        about: Vec<Uuid>,
        occurred_at: DateTime<Utc>,
        observed_at: DateTime<Utc>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            text: text.into(),
            kind: default_impression_kind(),
            occurred_at,
            observed_at,
            faculty: String::new(),
            about,
            confidence: default_impression_confidence(),
            payload: Value::Null,
        }
    }
}

fn default_impression_kind() -> String {
    DEFAULT_IMPRESSION_KIND.to_owned()
}

fn default_impression_confidence() -> f32 {
    DEFAULT_IMPRESSION_CONFIDENCE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sensation::Sensation, time::now};
    use chrono::Duration;
    use serde_json::json;

    #[test]
    fn impression_references_sensations() {
        let s = Sensation::new("vision.frame", "camera_0", now(), now(), json!({}));
        let imp = Impression::new(vec![s.id], now(), now(), "I see one face.");
        assert!(imp.about.contains(&s.id));
    }

    #[test]
    fn impression_roundtrips_through_json() {
        let t = now();
        let mut imp = Impression::new(vec![Uuid::new_v4()], t, t, "The speaker said hello.");
        imp.kind = "audio.utterance".to_owned();
        imp.faculty = "ASR Faculty".to_owned();
        imp.confidence = 0.92;
        imp.payload = json!({ "lang": "en" });
        let json = serde_json::to_string(&imp).expect("serialize");
        let back: Impression = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(imp.id, back.id);
        assert_eq!(imp.text, back.text);
        assert_eq!(imp.kind, back.kind);
        assert_eq!(imp.faculty, back.faculty);
        assert_eq!(imp.about, back.about);
        assert_eq!(imp.confidence, back.confidence);
        assert_eq!(imp.payload, back.payload);
    }

    #[test]
    fn impression_deserializes_legacy_how_and_sensation_ids() {
        let id = Uuid::new_v4();
        let sid = Uuid::new_v4();
        let t = now();
        let legacy = json!({
            "id": id,
            "how": "Legacy format text.",
            "occurred_at": t,
            "observed_at": t,
            "sensation_ids": [sid]
        });

        let back: Impression = serde_json::from_value(legacy).expect("deserialize legacy");
        assert_eq!(back.id, id);
        assert_eq!(back.text, "Legacy format text.");
        assert_eq!(back.about, vec![sid]);
        assert_eq!(back.kind, "observation.unknown");
    }

    /// A faculty processing a delayed sensation inherits the source
    /// `occurred_at` so the impression is positioned correctly in historical
    /// time, even when the faculty ran much later.
    #[test]
    fn impression_propagates_source_occurred_at_for_delayed_sensation() {
        let occurred = now();
        let observed = occurred + Duration::seconds(30);

        // Simulate a delayed sensation (e.g. a batched camera frame).
        let s = Sensation::new("vision.frame", "camera_0", occurred, observed, json!({}));

        // A faculty that processes the delayed sensation synchronously should
        // produce an impression whose occurred_at matches the source event time,
        // not the late delivery time.
        let imp = Impression::new(
            vec![s.id],
            s.occurred_at,
            s.observed_at,
            "One face detected.",
        );

        assert_eq!(
            imp.occurred_at, occurred,
            "impression occurred_at must match the source sensation's occurred_at"
        );
        assert_eq!(
            imp.observed_at, observed,
            "impression observed_at must match the source sensation's observed_at"
        );
    }
}
