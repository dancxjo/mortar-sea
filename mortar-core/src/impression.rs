use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    /// The sensations this impression was drawn from.
    pub sensation_ids: Vec<Uuid>,
    /// When the observation occurred, inherited from the source sensation.
    ///
    /// Used as the sort key in [`TimelineFrame`](crate::timeline::TimelineFrame).
    pub occurred_at: DateTime<Utc>,
    /// When the faculty produced this impression.
    ///
    /// For synchronous processing this equals the source sensation's
    /// `observed_at`. May be later for asynchronous or deferred faculties.
    pub observed_at: DateTime<Utc>,
    /// A single natural-language observation, e.g. `"I'm seeing three faces."`.
    pub how: String,
}

impl Impression {
    /// Create a new Impression with a freshly generated UUID.
    pub fn new(
        sensation_ids: Vec<Uuid>,
        occurred_at: DateTime<Utc>,
        observed_at: DateTime<Utc>,
        how: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            sensation_ids,
            occurred_at,
            observed_at,
            how: how.into(),
        }
    }
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
        assert!(imp.sensation_ids.contains(&s.id));
    }

    #[test]
    fn impression_roundtrips_through_json() {
        let t = now();
        let imp = Impression::new(vec![Uuid::new_v4()], t, t, "The speaker said hello.");
        let json = serde_json::to_string(&imp).expect("serialize");
        let back: Impression = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(imp.id, back.id);
        assert_eq!(imp.how, back.how);
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
