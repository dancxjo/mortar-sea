use std::collections::HashSet;

use crate::{
    experience::Experience,
    faculty::Faculty,
    memory::{InMemory, Memory},
    sensation::Sensation,
    timeline::{TimelineEntry, TimelineFrame},
    wit::Wit,
};

/// Canonical cognition orchestration model.
///
/// A pipeline receives sensations, invokes faculties, maintains timeline state,
/// invokes wits, stores experiences, and can reintroduce recollections as
/// sensations.
pub struct Pipeline<M: Memory = InMemory> {
    timeline: TimelineFrame,
    memory: M,
    faculties: Vec<Box<dyn Faculty>>,
    wits: Vec<Box<dyn Wit>>,
}

impl Default for Pipeline<InMemory> {
    fn default() -> Self {
        Self::new(InMemory::new())
    }
}

impl<M: Memory> Pipeline<M> {
    pub fn new(memory: M) -> Self {
        Self {
            timeline: TimelineFrame::new(),
            memory,
            faculties: Vec::new(),
            wits: Vec::new(),
        }
    }

    pub fn with_faculty(mut self, faculty: impl Faculty + 'static) -> Self {
        self.faculties.push(Box::new(faculty));
        self
    }

    pub fn with_wit(mut self, wit: impl Wit + 'static) -> Self {
        self.wits.push(Box::new(wit));
        self
    }

    pub fn timeline(&self) -> &TimelineFrame {
        &self.timeline
    }

    pub fn memory(&self) -> &M {
        &self.memory
    }

    /// Present a sensation and run the full cognitive loop:
    /// sensation → faculty outputs → timeline → wit outputs → memory.
    ///
    /// Wits interpret the full accumulated timeline on every call. To avoid
    /// duplicate accumulation across observations, experiences that match an
    /// already-stored experience are skipped before storage.
    pub fn observe(&mut self, sensation: Sensation) -> Vec<Experience> {
        let mut pending = vec![sensation];

        while let Some(sensation) = pending.pop() {
            self.timeline
                .push(TimelineEntry::Sensation(sensation.clone()));

            for faculty in &mut self.faculties {
                let (derived_sensations, impressions) = faculty.process(&sensation);

                for impression in impressions {
                    self.timeline.push(TimelineEntry::Impression(impression));
                }

                pending.extend(derived_sensations);
            }
        }

        let mut experiences = Vec::new();
        let mut known_experience_keys: HashSet<_> =
            self.memory.recall().iter().map(experience_key).collect();
        for wit in &mut self.wits {
            for experience in wit.interpret(&self.timeline) {
                if !known_experience_keys.insert(experience_key(&experience)) {
                    continue;
                }

                self.memory.store(experience.clone());
                self.timeline
                    .push(TimelineEntry::Experience(experience.clone()));
                experiences.push(experience);
            }
        }

        experiences
    }

    /// Re-enter every recalled experience as an ordinary memory sensation.
    pub fn recall_into_timeline(&mut self) -> Vec<Sensation> {
        let sensations: Vec<Sensation> = self
            .memory
            .recall()
            .iter()
            .map(M::experience_to_sensation)
            .collect();

        for sensation in &sensations {
            self.timeline
                .push(TimelineEntry::Sensation(sensation.clone()));
        }

        sensations
    }
}

/// Canonical identity for deduplicating reinterpretation results.
///
/// Two experiences are treated as equivalent when they represent the same
/// meaning (`what`), drawn from the same impression IDs, at the same
/// `occurred_at` instant, regardless of generated UUID or `observed_at`.
fn experience_key(
    experience: &Experience,
) -> (Vec<uuid::Uuid>, chrono::DateTime<chrono::Utc>, String) {
    (
        experience.impression_ids.clone(),
        experience.occurred_at,
        experience.what.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockEmitter, MockFaculty, MockFacultyRule, MockWit, MockWitRule};

    #[test]
    fn pipeline_orchestrates_sensation_to_experience_storage() {
        let mut pipeline = Pipeline::default()
            .with_faculty(MockFaculty::new(
                "speech",
                vec![MockFacultyRule::impression(
                    "audio.utterance",
                    "Heard utterance: {text}",
                )],
            ))
            .with_wit(MockWit::new(
                "intent",
                vec![MockWitRule::new("Heard utterance", "A person spoke.")],
            ));

        let sensation = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        let experiences = pipeline.observe(sensation);

        assert_eq!(experiences.len(), 1);
        assert_eq!(pipeline.memory().recall().len(), 1);
        assert!(pipeline
            .timeline()
            .entries()
            .iter()
            .any(|entry| matches!(entry, TimelineEntry::Experience(_))));
    }
}
