use chrono::Duration;
use uuid::Uuid;

use crate::{
    episode::Episode,
    experience::Experience,
    link::ExperienceLink,
    sensation::{Provenance, Sensation},
};

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
        .with_provenance(Provenance::memory_recall(experience.id))
    }
}

/// Extension of [`Memory`] with experience-to-experience linking and episode
/// formation.
///
/// `LinkedMemory` is intentionally backend-independent. Implementations may
/// store links and episodes however they like — in-process vecs, graph
/// databases, vector databases, or any combination — as long as they honour the
/// trait contract.
///
/// ## Core operations
///
/// - **Linking**: record a directed [`ExperienceLink`] between two stored
///   experiences. Links encode causal, social, sequential, or custom
///   relationships.
/// - **Episode formation**: group a subset of experiences into a labelled
///   [`Episode`] with explicit time bounds.
/// - **Temporal clustering**: automatically partition all stored experiences
///   into [`Episode`]s by proximity in time. The default implementation is
///   provided by the trait itself and does not require mutation.
pub trait LinkedMemory: Memory {
    /// Record a directed link between two experiences.
    ///
    /// The link is stored but the referenced experiences are not validated;
    /// callers are responsible for ensuring both ids are present in the store.
    fn link_experiences(&mut self, link: ExperienceLink);

    /// Returns all outgoing links whose `from_id` matches `experience_id`.
    fn links_from(&self, experience_id: Uuid) -> Vec<ExperienceLink>;

    /// Returns all incoming links whose `to_id` matches `experience_id`.
    fn links_to(&self, experience_id: Uuid) -> Vec<ExperienceLink>;

    /// Group a set of experiences into a named [`Episode`] and persist it.
    ///
    /// `started_at` and `ended_at` are derived from the `occurred_at` values of
    /// the named experiences. If `experience_ids` is empty an episode spanning
    /// `now()` is created with no members.
    fn form_episode(&mut self, experience_ids: Vec<Uuid>, label: impl Into<String>) -> Episode;

    /// Retrieve a specific episode by id, or `None` if not found.
    fn recall_episode(&self, episode_id: Uuid) -> Option<Episode>;

    /// Returns all episodes that have been explicitly formed via
    /// [`form_episode`](Self::form_episode).
    fn episodes(&self) -> Vec<Episode>;

    /// Partition all stored experiences into temporal clusters.
    ///
    /// Experiences are sorted by `occurred_at`. A new cluster begins whenever
    /// the gap between consecutive experiences exceeds `window`. The resulting
    /// [`Episode`]s are returned as a pure computation — they are **not**
    /// persisted to the episode store.
    ///
    /// This default implementation is O(n log n) in the number of stored
    /// experiences.
    fn temporal_clusters(&self, window: Duration) -> Vec<Episode> {
        let mut experiences = self.recall();
        experiences.sort_by_key(|e| e.occurred_at);

        if experiences.is_empty() {
            return vec![];
        }

        let mut groups: Vec<Vec<Experience>> = vec![vec![experiences[0].clone()]];

        for exp in &experiences[1..] {
            let last_group = groups.last_mut().expect("at least one group");
            let last_time = last_group
                .last()
                .expect("groups are never empty")
                .occurred_at;
            let gap = exp.occurred_at.signed_duration_since(last_time);
            if gap <= window {
                last_group.push(exp.clone());
            } else {
                groups.push(vec![exp.clone()]);
            }
        }

        groups
            .into_iter()
            .map(|group| {
                let started_at = group.first().expect("non-empty").occurred_at;
                let ended_at = group.last().expect("non-empty").occurred_at;
                let ids = group.iter().map(|e| e.id).collect();
                Episode::new(ids, started_at, ended_at, "temporal cluster")
            })
            .collect()
    }
}

/// An in-process [`LinkedMemory`] implementation suitable for tests and
/// development.
///
/// `InMemoryLinked` stores experiences, links, and episodes in plain `Vec`s.
/// It makes no guarantees about ordering, deduplication, or capacity beyond
/// what the trait requires.
#[derive(Debug, Default)]
pub struct InMemoryLinked {
    experiences: Vec<Experience>,
    links: Vec<ExperienceLink>,
    episodes: Vec<Episode>,
}

impl InMemoryLinked {
    /// Create a new, empty linked memory store.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Memory for InMemoryLinked {
    fn store(&mut self, experience: Experience) {
        self.experiences.push(experience);
    }

    fn recall(&self) -> Vec<Experience> {
        self.experiences.clone()
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
        .with_provenance(Provenance::memory_recall(experience.id))
    }
}

impl LinkedMemory for InMemoryLinked {
    fn link_experiences(&mut self, link: ExperienceLink) {
        self.links.push(link);
    }

    fn links_from(&self, experience_id: Uuid) -> Vec<ExperienceLink> {
        self.links
            .iter()
            .filter(|l| l.from_id == experience_id)
            .cloned()
            .collect()
    }

    fn links_to(&self, experience_id: Uuid) -> Vec<ExperienceLink> {
        self.links
            .iter()
            .filter(|l| l.to_id == experience_id)
            .cloned()
            .collect()
    }

    fn form_episode(&mut self, experience_ids: Vec<Uuid>, label: impl Into<String>) -> Episode {
        let now = crate::time::now();
        let (started_at, ended_at) = experience_ids
            .iter()
            .filter_map(|id| self.experiences.iter().find(|e| e.id == *id))
            .fold((now, now), |(min, max), e| {
                (min.min(e.occurred_at), max.max(e.occurred_at))
            });
        let episode = Episode::new(experience_ids, started_at, ended_at, label);
        self.episodes.push(episode.clone());
        episode
    }

    fn recall_episode(&self, episode_id: Uuid) -> Option<Episode> {
        self.episodes.iter().find(|ep| ep.id == episode_id).cloned()
    }

    fn episodes(&self) -> Vec<Episode> {
        self.episodes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        experience::Experience,
        link::ExperienceLinkKind,
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
            assert_eq!(s.provenance, Provenance::memory_recall(recalled.id));
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
        assert!(
            frame
                .entries()
                .iter()
                .all(|e| matches!(e, TimelineEntry::Sensation(_)))
        );
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

    // ── LinkedMemory / InMemoryLinked tests ──────────────────────────────────

    #[test]
    fn linked_memory_stores_and_recalls_experiences() {
        let mut mem = InMemoryLinked::new();
        let exp = Experience::new(vec![], now(), now(), "A visitor arrived.");
        mem.store(exp.clone());
        let recalled = mem.recall();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].id, exp.id);
    }

    #[test]
    fn linked_memory_records_causal_link() {
        let mut mem = InMemoryLinked::new();
        let t = now();
        let a = Experience::new(vec![], t, t, "Door opened.");
        let b = Experience::new(vec![], t, t, "Visitor entered.");
        mem.store(a.clone());
        mem.store(b.clone());

        let link = ExperienceLink::new(a.id, b.id, ExperienceLinkKind::Causal);
        mem.link_experiences(link.clone());

        let from_a = mem.links_from(a.id);
        assert_eq!(from_a.len(), 1);
        assert_eq!(from_a[0].to_id, b.id);
        assert_eq!(from_a[0].kind, ExperienceLinkKind::Causal);

        let to_b = mem.links_to(b.id);
        assert_eq!(to_b.len(), 1);
        assert_eq!(to_b[0].from_id, a.id);
    }

    #[test]
    fn linked_memory_records_social_link() {
        let mut mem = InMemoryLinked::new();
        let t = now();
        let a = Experience::new(vec![], t, t, "Alice greeted the system.");
        let b = Experience::new(vec![], t, t, "Alice asked a question.");
        mem.store(a.clone());
        mem.store(b.clone());

        mem.link_experiences(ExperienceLink::new(a.id, b.id, ExperienceLinkKind::Social));

        let social_links = mem.links_from(a.id);
        assert!(
            social_links
                .iter()
                .any(|l| l.kind == ExperienceLinkKind::Social)
        );
    }

    #[test]
    fn linked_memory_forms_and_recalls_episode() {
        let mut mem = InMemoryLinked::new();
        let t0 = now();
        let t1 = t0 + Duration::seconds(30);
        let a = Experience::new(vec![], t0, t0, "System started.");
        let b = Experience::new(vec![], t1, t1, "User logged in.");
        mem.store(a.clone());
        mem.store(b.clone());

        let ep = mem.form_episode(vec![a.id, b.id], "Startup sequence");
        assert_eq!(ep.label, "Startup sequence");
        assert_eq!(ep.experience_ids.len(), 2);
        assert_eq!(ep.started_at, t0);
        assert_eq!(ep.ended_at, t1);

        let recalled = mem.recall_episode(ep.id).expect("episode must be stored");
        assert_eq!(recalled.id, ep.id);
    }

    #[test]
    fn linked_memory_episodes_returns_all_formed_episodes() {
        let mut mem = InMemoryLinked::new();
        let t = now();
        let a = Experience::new(vec![], t, t, "First.");
        let b = Experience::new(vec![], t, t, "Second.");
        mem.store(a.clone());
        mem.store(b.clone());

        mem.form_episode(vec![a.id], "Episode A");
        mem.form_episode(vec![b.id], "Episode B");

        let eps = mem.episodes();
        assert_eq!(eps.len(), 2);
        let labels: Vec<&str> = eps.iter().map(|ep| ep.label.as_str()).collect();
        assert!(labels.contains(&"Episode A"));
        assert!(labels.contains(&"Episode B"));
    }

    #[test]
    fn temporal_clusters_groups_close_experiences_together() {
        let mut mem = InMemoryLinked::new();
        let t0 = now();
        // Three experiences close together, then a gap, then one more.
        let t1 = t0 + Duration::seconds(5);
        let t2 = t0 + Duration::seconds(9);
        let t3 = t0 + Duration::seconds(60);
        let a = Experience::new(vec![], t0, t0, "First.");
        let b = Experience::new(vec![], t1, t1, "Second.");
        let c = Experience::new(vec![], t2, t2, "Third.");
        let d = Experience::new(vec![], t3, t3, "After gap.");
        mem.store(a.clone());
        mem.store(b.clone());
        mem.store(c.clone());
        mem.store(d.clone());

        let clusters = mem.temporal_clusters(Duration::seconds(15));
        assert_eq!(clusters.len(), 2, "two clusters expected: a/b/c and d");
        assert_eq!(clusters[0].experience_ids.len(), 3);
        assert_eq!(clusters[1].experience_ids.len(), 1);

        // temporal_clusters must not persist episodes
        assert!(mem.episodes().is_empty());
    }

    #[test]
    fn temporal_clusters_returns_empty_for_empty_store() {
        let mem = InMemoryLinked::new();
        let clusters = mem.temporal_clusters(Duration::seconds(10));
        assert!(clusters.is_empty());
    }

    #[test]
    fn links_from_and_to_return_empty_when_no_links() {
        let mem = InMemoryLinked::new();
        assert!(mem.links_from(Uuid::new_v4()).is_empty());
        assert!(mem.links_to(Uuid::new_v4()).is_empty());
    }
}
