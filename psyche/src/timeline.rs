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

/// A temporally local group of nearby timeline entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventCluster {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub entries: Vec<TimelineEntry>,
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

    /// Returns the `observed_at` timestamp regardless of entry type.
    ///
    /// This is when the cognitive system first became aware of the event.
    /// It may equal `occurred_at` for live sensors or be strictly later for
    /// delayed delivery, replay, or memory recall.
    pub fn observed_at(&self) -> DateTime<Utc> {
        match self {
            TimelineEntry::Sensation(s) => s.observed_at,
            TimelineEntry::Impression(i) => i.observed_at,
            TimelineEntry::Experience(e) => e.observed_at,
        }
    }

    /// Returns true when this entry directly references the sensation.
    pub fn references_sensation(&self, sensation_id: Uuid) -> bool {
        match self {
            TimelineEntry::Sensation(sensation) => {
                sensation.provenance.references_sensation(sensation_id)
            }
            TimelineEntry::Impression(impression) => impression.about.contains(&sensation_id),
            TimelineEntry::Experience(_) => false,
        }
    }

    /// Returns true when this entry directly references the impression.
    pub fn references_impression(&self, impression_id: Uuid) -> bool {
        matches!(
            self,
            TimelineEntry::Experience(experience) if experience.impression_ids.contains(&impression_id)
        )
    }

    /// Returns true when this entry directly references the experience.
    pub fn references_experience(&self, experience_id: Uuid) -> bool {
        matches!(
            self,
            TimelineEntry::Sensation(sensation) if sensation.provenance.references_experience(experience_id)
        )
    }
}

/// Partition a chronologically ordered slice of timeline entries into nearby
/// clusters.
///
/// A new cluster begins whenever the gap between consecutive entries exceeds
/// `max_gap`. The original entry ordering is preserved inside each cluster.
pub fn event_clusters(entries: &[TimelineEntry], max_gap: chrono::Duration) -> Vec<EventCluster> {
    let Some(first) = entries.first() else {
        return Vec::new();
    };

    let mut clusters = vec![EventCluster {
        start: first.occurred_at(),
        end: first.occurred_at(),
        entries: vec![first.clone()],
    }];

    for entry in &entries[1..] {
        let cluster = clusters.last_mut().expect("clusters is never empty");
        let gap = entry.occurred_at().signed_duration_since(cluster.end);

        if gap <= max_gap {
            cluster.end = entry.occurred_at();
            cluster.entries.push(entry.clone());
        } else {
            clusters.push(EventCluster {
                start: entry.occurred_at(),
                end: entry.occurred_at(),
                entries: vec![entry.clone()],
            });
        }
    }

    clusters
}

/// A heterogeneous, time-ordered collection of cognitive events.
///
/// A `TimelineFrame` holds sensations, impressions, and experiences in a single
/// sequence sorted strictly by `occurred_at`. Reasoning systems should consume
/// a timeline rather than individual subsystem outputs; this ensures that
/// temporal ordering—not type or source—governs cognition.
///
/// ## Ordering rules
///
/// Entries are always ordered by `occurred_at`, the time the underlying event
/// took place. `observed_at` (when the system became aware) is **not** used for
/// ordering. Consequences:
///
/// - A delayed sensation whose `occurred_at` is in the past is inserted before
///   entries that were added to the timeline earlier but whose `occurred_at` is
///   more recent.
/// - Replayed entries sort according to when their events originally happened.
///   Memory recollection sensations sort at recollection time (now) while
///   retaining original-event timestamps in payload metadata.
/// - Two entries with identical `occurred_at` retain stable relative insertion
///   order (new entries are placed after existing ones with the same timestamp).
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

    /// Returns the last `limit` entries in chronological order
    /// (oldest-to-newest within that most-recent subset).
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

    /// Returns entries that directly reference an experience id.
    pub fn entries_referencing_experience(
        &self,
        experience_id: Uuid,
    ) -> impl Iterator<Item = &TimelineEntry> + '_ {
        self.entries
            .iter()
            .filter(move |entry| entry.references_experience(experience_id))
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
    /// Neighbor relationships are inferred from direct references:
    /// - impressions reference sensation ids
    /// - experiences reference impression ids
    /// - sensations connect to impressions that reference them
    /// - impressions connect to experiences that reference them
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
                        if impression.about.contains(&sensation.id)
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
                related.extend(self.entries_referencing_experience(experience.id));
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

    #[test]
    fn event_clusters_group_nearby_entries_and_preserve_order() {
        let t0 = now();
        let s0 = Sensation::new("vision.face_crop", "camera", t0, t0, json!({}));
        let i0 = Impression::new(
            vec![s0.id],
            t0 + Duration::milliseconds(180),
            t0 + Duration::milliseconds(180),
            "That face looks like Tim.",
        );
        let s1 = Sensation::new(
            "audio.utterance",
            "mic",
            t0 + Duration::milliseconds(420),
            t0 + Duration::milliseconds(420),
            json!({"text": "hello"}),
        );
        let i1 = Impression::new(
            vec![s0.id],
            t0 + Duration::seconds(2),
            t0 + Duration::seconds(2),
            "The speaker may be someone else.",
        );

        let entries = vec![
            TimelineEntry::Sensation(s0.clone()),
            TimelineEntry::Impression(i0.clone()),
            TimelineEntry::Sensation(s1.clone()),
            TimelineEntry::Impression(i1.clone()),
        ];

        let clusters = event_clusters(&entries, Duration::milliseconds(500));

        assert_eq!(clusters.len(), 2);
        assert_eq!(clusters[0].start, s0.occurred_at);
        assert_eq!(clusters[0].end, s1.occurred_at);
        assert_eq!(clusters[0].entries.len(), 3);
        assert_eq!(clusters[0].entries[0].id(), s0.id);
        assert_eq!(clusters[0].entries[1].id(), i0.id);
        assert_eq!(clusters[0].entries[2].id(), s1.id);
        assert_eq!(clusters[1].start, i1.occurred_at);
        assert_eq!(clusters[1].end, i1.occurred_at);
    }

    #[test]
    fn event_clusters_obey_configured_gap_threshold() {
        let t0 = now();
        let a =
            TimelineEntry::Sensation(Sensation::new("vision.frame", "camera", t0, t0, json!({})));
        let b = TimelineEntry::Sensation(Sensation::new(
            "vision.face_crop",
            "camera",
            t0 + Duration::milliseconds(600),
            t0 + Duration::milliseconds(600),
            json!({}),
        ));

        assert_eq!(
            event_clusters(&[a.clone(), b.clone()], Duration::milliseconds(300)).len(),
            2
        );
        assert_eq!(
            event_clusters(&[a, b], Duration::milliseconds(800)).len(),
            1
        );
    }

    /// A delayed sensation has `observed_at` after `occurred_at`.
    ///
    /// The timeline places it at its historical `occurred_at` position
    /// regardless of when it arrived in the pipeline.
    #[test]
    fn delayed_sensation_is_ordered_by_occurred_at_not_observed_at() {
        let t_past = now();
        let t_current = t_past + Duration::seconds(60);

        // A sensor delivers a past event late: occurred 60 s ago, observed now.
        let delayed = Sensation::new(
            "sensor.reading",
            "delayed_source",
            t_past,
            t_current,
            json!({}),
        );
        // A live event that entered the pipeline just before the delayed one.
        let live = Sensation::new(
            "sensor.reading",
            "live_source",
            t_current,
            t_current,
            json!({}),
        );

        let mut frame = TimelineFrame::new();
        // Live entry is pushed first, then the late-arriving delayed entry.
        frame.push(TimelineEntry::Sensation(live));
        frame.push(TimelineEntry::Sensation(delayed));

        // Despite being pushed second, the delayed entry must appear first.
        assert_eq!(frame.entries()[0].occurred_at(), t_past);
        assert_eq!(frame.entries()[1].occurred_at(), t_current);
    }

    /// Replayed data (old `occurred_at`) sorts before current-time entries.
    ///
    /// This verifies that batch replay or log ingestion preserves causal order
    /// even when replayed events enter the pipeline long after they happened.
    #[test]
    fn replayed_data_sorts_before_current_entries() {
        let now_t = now();
        let replay_time = now_t - Duration::hours(1);
        let replay_observed = now_t; // replayed data is observed now

        let current = Sensation::new("vision.frame", "camera_0", now_t, now_t, json!({}));
        let replayed = Sensation::new(
            "replay.event",
            "replay",
            replay_time,
            replay_observed,
            json!({}),
        );

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(current));
        frame.push(TimelineEntry::Sensation(replayed));

        assert_eq!(
            frame.entries()[0].occurred_at(),
            replay_time,
            "replayed entry with past occurred_at must sort first"
        );
        // observed_at of the replayed entry is the current time, not replay_time
        assert_eq!(frame.entries()[0].observed_at(), replay_observed);
    }

    /// Two entries with the same `occurred_at` retain stable relative order.
    #[test]
    fn same_occurred_at_preserves_insertion_order() {
        let t = now();
        let obs = t;

        let s0 = Sensation::new("vision.frame", "camera_0", t, obs, json!({"seq": 0}));
        let s1 = Sensation::new("vision.frame", "camera_0", t, obs, json!({"seq": 1}));

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(s0.clone()));
        frame.push(TimelineEntry::Sensation(s1.clone()));

        // Both have the same occurred_at; insertion order must be preserved.
        match (&frame.entries()[0], &frame.entries()[1]) {
            (TimelineEntry::Sensation(a), TimelineEntry::Sensation(b)) => {
                assert_eq!(a.payload["seq"], 0);
                assert_eq!(b.payload["seq"], 1);
            }
            _ => panic!("expected two sensations"),
        }
    }

    /// `observed_at` is accessible on every entry type via the helper method.
    #[test]
    fn observed_at_is_accessible_on_all_entry_kinds() {
        let occurred = now();
        let observed = occurred + Duration::seconds(5);

        let s = Sensation::new("vision.frame", "cam", occurred, observed, json!({}));
        let imp = Impression::new(vec![s.id], occurred, observed, "Something seen.");
        let exp = Experience::new(vec![imp.id], occurred, observed, "Someone was present.");

        let se = TimelineEntry::Sensation(s);
        let ie = TimelineEntry::Impression(imp);
        let ee = TimelineEntry::Experience(exp);

        for entry in [&se, &ie, &ee] {
            assert_eq!(
                entry.occurred_at(),
                occurred,
                "occurred_at must match for {:?}",
                entry.kind()
            );
            assert_eq!(
                entry.observed_at(),
                observed,
                "observed_at must match for {:?}",
                entry.kind()
            );
        }
    }

    /// Derived sensations produced by a faculty inherit `occurred_at` from the
    /// parent sensation so they are positioned correctly in historical time.
    ///
    /// For example: a face-crop derived from a delayed camera frame should
    /// appear at the same point in the timeline as that frame, not at the time
    /// the faculty ran.
    #[test]
    fn derived_sensation_inherits_parent_occurred_at() {
        let frame_occurred = now();
        let frame_observed = frame_occurred + Duration::seconds(5); // delivered late

        let parent = Sensation::new(
            "vision.frame",
            "camera_0",
            frame_occurred,
            frame_observed,
            json!({}),
        );
        // Faculty derives a face-crop from the frame, preserving timestamps.
        let derived = Sensation::new(
            "vision.face_crop",
            "face_detector",
            parent.occurred_at, // inherited from parent
            parent.observed_at, // inherited from parent
            json!({"face_id": 1}),
        )
        .with_provenance(
            crate::sensation::Provenance::derived_from_sensation(parent.id)
                .with_faculty("face_detector"),
        );

        assert_eq!(
            derived.occurred_at, parent.occurred_at,
            "derived sensation must inherit parent's occurred_at"
        );
        assert_eq!(
            derived.observed_at, parent.observed_at,
            "derived sensation must inherit parent's observed_at"
        );

        // Both sort to the same position in the timeline.
        let mut tl = TimelineFrame::new();
        tl.push(TimelineEntry::Sensation(parent.clone()));
        tl.push(TimelineEntry::Sensation(derived.clone()));

        let times: Vec<_> = tl.entries().iter().map(|e| e.occurred_at()).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(times[0], frame_occurred);
        assert_eq!(times[1], frame_occurred);
        let related = tl.related_entries(parent.id);
        assert_eq!(related.len(), 1);
        assert!(matches!(
            related[0],
            TimelineEntry::Sensation(sensation)
            if sensation.kind == "vision.face_crop" && sensation.provenance.references_sensation(parent.id)
        ));
    }

    #[test]
    fn memory_recall_provenance_links_back_to_experience() {
        let t0 = now();
        let experience = Experience::new(vec![], t0, t0, "A recalled memory.");
        let recalled = Sensation::new(
            "memory.related_experience",
            "memory",
            t0,
            t0 + Duration::seconds(1),
            json!({}),
        )
        .with_provenance(crate::sensation::Provenance::memory_recall(experience.id));

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Experience(experience.clone()));
        frame.push(TimelineEntry::Sensation(recalled.clone()));

        let related = frame.related_entries(experience.id);
        assert_eq!(related.len(), 1);
        assert!(matches!(
            related[0],
            TimelineEntry::Sensation(sensation)
            if sensation.id == recalled.id && sensation.provenance.references_experience(experience.id)
        ));
        assert_eq!(
            frame.entries_referencing_experience(experience.id).count(),
            1
        );
    }
}
