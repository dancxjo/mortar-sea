use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::{experience::Experience, impression::Impression, sensation::Sensation};

/// A single entry in a [`TimelineFrame`].
///
/// The timeline is heterogeneous: sensations, impressions, and experiences all
/// live together, ordered only by `occurred_at`. No grouping by type or source
/// is performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimelineEntry {
    Sensation(Sensation),
    Impression(Impression),
    Experience(Experience),
}

impl TimelineEntry {
    /// Returns the `occurred_at` timestamp regardless of entry type.
    pub fn occurred_at(&self) -> DateTime<Utc> {
        match self {
            TimelineEntry::Sensation(s) => s.occurred_at,
            TimelineEntry::Impression(i) => i.occurred_at,
            TimelineEntry::Experience(e) => e.occurred_at,
        }
    }
}

/// A heterogeneous, time-ordered collection of cognitive events.
///
/// A `TimelineFrame` holds sensations, impressions, and experiences in a single
/// sequence sorted strictly by `occurred_at`. Reasoning systems should consume
/// a timeline rather than individual subsystem outputs; this ensures that
/// temporal ordering—not type or source—governs cognition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimelineFrame {
    entries: Vec<TimelineEntry>,
}

impl TimelineFrame {
    /// Create an empty frame.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a sensation, impression, or experience into the frame.
    ///
    /// The frame remains sorted by `occurred_at` after every insertion.
    /// Insertion uses binary search to locate the correct position, giving
    /// O(log n) search and O(n) shift cost rather than a full O(n log n) sort.
    pub fn push(&mut self, entry: TimelineEntry) {
        let t = entry.occurred_at();
        let pos = self.entries.partition_point(|e| e.occurred_at() <= t);
        self.entries.insert(pos, entry);
    }

    /// Returns a slice of all entries in chronological order.
    pub fn entries(&self) -> &[TimelineEntry] {
        &self.entries
    }

    /// Returns the number of entries in the frame.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the frame contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{experience::Experience, impression::Impression, sensation::Sensation, time::now};
    use chrono::Duration;
    use serde_json::json;

    #[test]
    fn timeline_orders_mixed_entries_by_occurred_at() {
        let t0 = now();
        let t1 = t0 + Duration::seconds(1);
        let t2 = t0 + Duration::seconds(2);
        let t3 = t0 + Duration::seconds(3);

        let obs = now();

        // Insert out of order to verify sorting.
        let exp = Experience::new(vec![], t3, obs, "A visitor may have arrived.");
        let imp = Impression::new(vec![], t1, obs, "I'm seeing three faces.");
        let s1 = Sensation::new("vision.face_crop", "camera_0", t2, obs, json!({}));
        let s0 = Sensation::new("vision.frame", "camera_0", t0, obs, json!({}));

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Experience(exp));
        frame.push(TimelineEntry::Impression(imp));
        frame.push(TimelineEntry::Sensation(s1));
        frame.push(TimelineEntry::Sensation(s0));

        assert_eq!(frame.len(), 4);

        let times: Vec<DateTime<Utc>> = frame.entries().iter().map(|e| e.occurred_at()).collect();
        assert!(
            times.windows(2).all(|w| w[0] <= w[1]),
            "entries must be sorted"
        );

        // Verify exact order by type
        assert!(matches!(frame.entries()[0], TimelineEntry::Sensation(_)));
        assert!(matches!(frame.entries()[1], TimelineEntry::Impression(_)));
        assert!(matches!(frame.entries()[2], TimelineEntry::Sensation(_)));
        assert!(matches!(frame.entries()[3], TimelineEntry::Experience(_)));
    }
}
