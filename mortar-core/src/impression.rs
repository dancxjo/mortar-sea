use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An observation about one or more sensations.
///
/// Impressions describe *what was noticed*, not what it means. They answer
/// questions like "what did I see?" or "what was said?" rather than "what does
/// this imply?". That interpretive step belongs to [`Experience`](crate::experience::Experience).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Impression {
    /// Unique identifier for this impression.
    pub id: Uuid,
    /// The sensations this impression was drawn from.
    pub sensation_ids: Vec<Uuid>,
    /// When the observation occurred.
    pub occurred_at: DateTime<Utc>,
    /// When the system recorded this observation.
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
}
