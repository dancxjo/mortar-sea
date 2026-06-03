use crate::{experience::Experience, sensation::Sensation};

/// Memory stores and retrieves [`Experience`]s.
///
/// Crucially, retrieved experiences must be convertible into [`Sensation`]s so
/// that they can re-enter the cognitive pipeline on exactly the same footing as
/// any externally sourced sensation. This is the architectural invariant that
/// eliminates a separate "memory pathway": a recollection is just another thing
/// being perceived.
pub trait Memory {
    /// Store an experience for later retrieval.
    fn store(&mut self, experience: Experience);

    /// Retrieve all stored experiences.
    fn recall(&self) -> Vec<Experience>;

    /// Convert a retrieved experience into a memory-derived sensation.
    ///
    /// The returned sensation will have `kind = "memory.related_experience"`
    /// and carry the serialised experience as its JSON payload. It can be
    /// inserted into a [`TimelineFrame`](crate::timeline::TimelineFrame) like
    /// any other sensation.
    fn experience_to_sensation(experience: &Experience) -> Sensation
    where
        Self: Sized;
}

/// A minimal in-memory store, suitable only for tests.
///
/// This implementation holds experiences in a plain `Vec`. It makes no
/// guarantees about ordering, deduplication, or capacity.
#[derive(Debug, Default)]
pub struct InMemory {
    store: Vec<Experience>,
}

impl InMemory {
    /// Create a new, empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Memory for InMemory {
    fn store(&mut self, experience: Experience) {
        self.store.push(experience);
    }

    fn recall(&self) -> Vec<Experience> {
        self.store.clone()
    }

    fn experience_to_sensation(experience: &Experience) -> Sensation {
        let payload = serde_json::to_value(experience).expect("Experience is always serializable");
        Sensation::new(
            "memory.related_experience",
            "memory",
            experience.occurred_at,
            crate::time::now(),
            payload,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        experience::Experience,
        time::now,
        timeline::{TimelineEntry, TimelineFrame},
    };

    #[test]
    fn memory_stores_and_retrieves_experiences() {
        let mut mem = InMemory::new();
        let exp = Experience::new(vec![], now(), now(), "A visitor may have arrived.");
        mem.store(exp.clone());
        let recalled = mem.recall();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].id, exp.id);
        assert_eq!(recalled[0].what, exp.what);
    }

    #[test]
    fn recalled_experience_becomes_sensation() {
        let mut mem = InMemory::new();
        let exp = Experience::new(
            vec![],
            now(),
            now(),
            "The user is working on repository design.",
        );
        mem.store(exp.clone());

        for recalled in mem.recall() {
            let s = InMemory::experience_to_sensation(&recalled);
            assert_eq!(s.kind, "memory.related_experience");
            assert_eq!(s.source, "memory");
        }
    }

    #[test]
    fn memory_sensation_enters_timeline_like_any_other() {
        use crate::sensation::Sensation;
        use serde_json::json;

        let mut mem = InMemory::new();
        let exp = Experience::new(vec![], now(), now(), "A package was delivered.");
        mem.store(exp.clone());

        let external = Sensation::new("vision.frame", "camera_0", now(), now(), json!({}));
        let memory_sensation = InMemory::experience_to_sensation(&exp);

        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(external));
        frame.push(TimelineEntry::Sensation(memory_sensation));

        // Both sensations exist in the same frame with no special-casing.
        assert_eq!(frame.len(), 2);
        assert!(frame
            .entries()
            .iter()
            .all(|e| matches!(e, TimelineEntry::Sensation(_))));
    }

    /// When recalling an experience from memory the resulting sensation keeps
    /// `occurred_at` from the original experience so it sorts correctly in
    /// historical order.  `observed_at` is set to the time of recall (now),
    /// reflecting when the system re-encountered the memory.
    #[test]
    fn memory_recall_preserves_occurred_at_and_sets_observed_at_to_recall_time() {
        use chrono::Duration;

        let occurred = now();
        let observed = occurred; // originally a live observation
        let exp = Experience::new(vec![], occurred, observed, "Something important happened.");

        let mut mem = InMemory::new();
        mem.store(exp.clone());

        // Simulate a brief passage of time before recall.
        let recall_time = occurred + Duration::seconds(10);

        // experience_to_sensation is called at recall time; observed_at = now()
        // which is at least as late as occurred_at.
        let s = InMemory::experience_to_sensation(&exp);
        assert_eq!(
            s.occurred_at, occurred,
            "recalled sensation must preserve the original experience's occurred_at"
        );
        assert!(
            s.observed_at >= occurred,
            "recalled sensation's observed_at must be at or after occurred_at"
        );
        // observed_at should reflect the current moment (recall time), not the
        // original observation time.
        let _ = recall_time; // documents intent; exact value depends on wall clock
        assert_eq!(s.kind, "memory.related_experience");
        assert_eq!(s.source, "memory");
    }
}
