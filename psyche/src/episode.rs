use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A named grouping of [`Experience`](crate::experience::Experience)s that form
/// a coherent narrative or temporal unit.
///
/// Episodes are the coarse-grained memory structure above individual
/// experiences. They answer "what was this period of time about?" rather than
/// "what did a single observation mean?".
///
/// ## Relationship to experiences
///
/// An episode references experiences by id. The experiences themselves remain
/// in the flat memory store; the episode is just an index — a labelled window
/// over a subset of that store. This keeps the episode model backend-independent:
/// any implementation of [`LinkedMemory`](crate::memory::LinkedMemory) can form
/// episodes by recording ids without copying data.
///
/// ## Time bounds
///
/// `started_at` and `ended_at` are derived from the earliest and latest
/// `occurred_at` values among the member experiences. They are stored explicitly
/// so consumers can query or sort episodes without loading every member
/// experience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Unique identifier for this episode.
    pub id: Uuid,
    /// Human-readable label describing the episode.
    pub label: String,
    /// Ids of the experiences that belong to this episode, in formation order.
    pub experience_ids: Vec<Uuid>,
    /// Earliest `occurred_at` among member experiences.
    pub started_at: DateTime<Utc>,
    /// Latest `occurred_at` among member experiences.
    pub ended_at: DateTime<Utc>,
}

impl Episode {
    /// Create a new episode with a freshly generated UUID.
    pub fn new(
        experience_ids: Vec<Uuid>,
        started_at: DateTime<Utc>,
        ended_at: DateTime<Utc>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            label: label.into(),
            experience_ids,
            started_at,
            ended_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::now;
    use chrono::Duration;

    #[test]
    fn episode_holds_experience_ids_and_label() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let t0 = now();
        let t1 = t0 + Duration::seconds(5);
        let ep = Episode::new(vec![a, b], t0, t1, "Morning greeting");
        assert_eq!(ep.label, "Morning greeting");
        assert!(ep.experience_ids.contains(&a));
        assert!(ep.experience_ids.contains(&b));
        assert_eq!(ep.started_at, t0);
        assert_eq!(ep.ended_at, t1);
    }

    #[test]
    fn episode_roundtrips_through_json() {
        let t = now();
        let ep = Episode::new(vec![Uuid::new_v4()], t, t, "Single-event episode");
        let json = serde_json::to_string(&ep).expect("serialize");
        let back: Episode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ep.id, back.id);
        assert_eq!(ep.label, back.label);
    }
}
