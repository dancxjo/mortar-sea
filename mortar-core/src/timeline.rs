use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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

/// A coarse type discriminator for [`TimelineEntry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimelineEntryKind {
    Sensation,
    Impression,
    Experience,
}

impl TimelineEntry {
    /// Returns this entry's stable identifier.
    pub fn id(&self) -> Uuid {
        match self {
            TimelineEntry::Sensation(s) => s.id,
            TimelineEntry::Impression(i) => i.id,
            TimelineEntry::Experience(e) => e.id,
        }
    }

    /// Returns the coarse entry type.
    pub fn kind(&self) -> TimelineEntryKind {
        match self {
            TimelineEntry::Sensation(_) => TimelineEntryKind::Sensation,
            TimelineEntry::Impression(_) => TimelineEntryKind::Impression,
            TimelineEntry::Experience(_) => TimelineEntryKind::Experience,
        }
    }

    /// Returns the `occurred_at` timestamp regardless of entry type.
    pub fn occurred_at(&self) -> DateTime<Utc> {
        match self {
            TimelineEntry::Sensation(s) => s.occurred_at,
            TimelineEntry::Impression(i) => i.occurred_at,
            TimelineEntry::Experience(e) => e.occurred_at,
        }
    }

    /// Returns true when this entry directly references the sensation.
    pub fn references_sensation(&self, sensation_id: Uuid) -> bool {
        matches!(
            self,
            TimelineEntry::Impression(impression) if impression.sensation_ids.contains(&sensation_id)
        )
    }

    /// Returns true when this entry directly references the impression.
    pub fn references_impression(&self, impression_id: Uuid) -> bool {
        matches!(
            self,
            TimelineEntry::Experience(experience) if experience.impression_ids.contains(&impression_id)
        )
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

    /// Returns the last `limit` entries in chronological order.
    pub fn recent_entries(&self, limit: usize) -> &[TimelineEntry] {
        let start = self.entries.len().saturating_sub(limit);
        &self.entries[start..]
    }

    /// Returns all entries matching a single type discriminator.
    pub fn entries_by_kind(
        &self,
        kind: TimelineEntryKind,
    ) -> impl Iterator<Item = &TimelineEntry> + '_ {
        self.entries
            .iter()
            .filter(move |entry| entry.kind() == kind)
    }

    /// Returns entries that directly reference a sensation id.
    pub fn entries_referencing_sensation(
        &self,
        sensation_id: Uuid,
    ) -> impl Iterator<Item = &TimelineEntry> + '_ {
        self.entries
            .iter()
            .filter(move |entry| entry.references_sensation(sensation_id))
    }

    /// Returns entries that directly reference an impression id.
    pub fn entries_referencing_impression(
        &self,
        impression_id: Uuid,
    ) -> impl Iterator<Item = &TimelineEntry> + '_ {
        self.entries
            .iter()
            .filter(move |entry| entry.references_impression(impression_id))
    }

    /// Returns entries with `occurred_at` in the inclusive window.
    pub fn entries_between(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> &[TimelineEntry] {
        if start > end {
            return &[];
        }

        let start_idx = self
            .entries
            .partition_point(|entry| entry.occurred_at() < start);
        let end_idx = self
            .entries
            .partition_point(|entry| entry.occurred_at() <= end);
        &self.entries[start_idx..end_idx]
    }

    /// Returns one-hop neighbors for future graph-style traversal.
    ///
    /// Edges are inferred from direct references:
    /// - impression -> sensation ids
    /// - experience -> impression ids
    /// - sensation <- impressions that reference it
    /// - impression <- experiences that reference it
    pub fn related_entries(&self, entry_id: Uuid) -> Vec<&TimelineEntry> {
        let mut related = Vec::new();

        let Some(target) = self.entries.iter().find(|entry| entry.id() == entry_id) else {
            return related;
        };

        match target {
            TimelineEntry::Sensation(sensation) => {
                related.extend(self.entries_referencing_sensation(sensation.id));
            }
            TimelineEntry::Impression(impression) => {
                related.extend(self.entries.iter().filter(|entry| {
                    matches!(
                        entry,
                        TimelineEntry::Sensation(sensation)
                        if impression.sensation_ids.contains(&sensation.id)
                    )
                }));
                related.extend(self.entries_referencing_impression(impression.id));
            }
            TimelineEntry::Experience(experience) => {
                related.extend(self.entries.iter().filter(|entry| {
                    matches!(
                        entry,
                        TimelineEntry::Impression(impression)
                        if experience.impression_ids.contains(&impression.id)
                    )
                }));
            }
        }

        related
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
