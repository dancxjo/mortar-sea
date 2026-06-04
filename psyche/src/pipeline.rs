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

const MAX_EXPERIENCE_FEEDBACK_LOOPS: usize = 32;

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
    /// sensation → faculty outputs → timeline → wit outputs → memory → sensation.
    ///
    /// Wits interpret the full accumulated timeline on every call. To avoid
    /// duplicate accumulation across observations, experiences that match an
    /// already-stored experience are skipped before storage. Newly created
    /// experiences are also reintroduced as sensations for recursive cognition,
    /// bounded by a feedback loop cap to prevent runaway recursion.
    pub fn observe(&mut self, sensation: Sensation) -> Vec<Experience> {
        let mut pending = vec![sensation];
        let mut feedback_loops = 0;
        let mut experiences = Vec::new();
        let mut known_experience_keys: HashSet<_> =
            self.memory.recall().iter().map(experience_key).collect();

        while !pending.is_empty() && feedback_loops < MAX_EXPERIENCE_FEEDBACK_LOOPS {
            feedback_loops += 1;
            let mut next_pending = Vec::new();

            while let Some(sensation) = pending.pop() {
                self.timeline
                    .push(TimelineEntry::Sensation(sensation.clone()));

                for faculty in &mut self.faculties {
                    let (derived_sensations, impressions) = faculty.process(&sensation);

                    for impression in impressions {
                        self.timeline.push(TimelineEntry::Impression(impression));
                    }

                    next_pending.extend(derived_sensations);
                }
            }

            for wit in &mut self.wits {
                for experience in wit.interpret(&self.timeline) {
                    if !known_experience_keys.insert(experience_key(&experience)) {
                        continue;
                    }

                    self.memory.store(experience.clone());
                    self.timeline
                        .push(TimelineEntry::Experience(experience.clone()));
                    next_pending.push(M::experience_to_sensation(&experience));
                    experiences.push(experience);
                }
            }

            pending = next_pending;
        }

        experiences
    }

    /// Re-enter every currently recalled experience as an ordinary memory sensation.
    ///
    /// Canonical recall semantics:
    /// - Recall is explicit pull, not automatic during [`Self::observe`].
    /// - Selection is a full snapshot of `memory.recall()` at call time.
    /// - Each recalled experience becomes one sensation with
    ///   `kind = "memory.related_experience"` and `source = "memory"`.
    /// - Recalled sensations represent the moment of recollection:
    ///   `occurred_at = observed_at = now`.
    /// - Payload retains historical event metadata (`original_*` fields).
    /// - Recalled sensations are inserted into the [`TimelineFrame`] using normal
    ///   timeline ordering by `occurred_at`.
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
    use crate::impression::Impression;
    use crate::mock::{
        MockEmitter, MockFaculty, MockFacultyRule, MockWit, MockWitRule, ScriptedMemory,
    };
    use crate::time::now;
    use chrono::Duration;
    use serde_json::json;

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
        assert!(
            pipeline
                .timeline()
                .entries()
                .iter()
                .any(|entry| matches!(entry, TimelineEntry::Experience(_)))
        );
    }

    #[test]
    fn recall_is_explicit_pull_not_observe_side_effect() {
        let earlier = now();
        let remembered = Experience::new(vec![], earlier, earlier, "Remembered meaning.");
        let later = earlier + Duration::seconds(5);

        let mut pipeline = Pipeline::new(ScriptedMemory::new(vec![remembered]));
        pipeline.observe(Sensation::new(
            "vision.frame",
            "camera",
            later,
            later,
            json!({}),
        ));

        assert!(!pipeline.timeline().entries().iter().any(|entry| {
            matches!(
                entry,
                TimelineEntry::Sensation(s) if s.kind == "memory.related_experience"
            )
        }));

        let recalled = pipeline.recall_into_timeline();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].kind, "memory.related_experience");
        assert_eq!(recalled[0].source, "memory");
    }

    #[test]
    fn recall_selection_and_timeline_order_are_canonical() {
        let t0 = now() - Duration::seconds(30);
        let t1 = t0 + Duration::seconds(1);
        let t2 = t0 + Duration::seconds(2);

        let late = Experience::new(vec![], t2, t2, "Late meaning.");
        let early = Experience::new(vec![], t0, t0, "Early meaning.");
        let mut pipeline = Pipeline::new(ScriptedMemory::new(vec![late.clone(), early.clone()]));

        let recalled = pipeline.recall_into_timeline();
        assert_eq!(recalled.len(), 2);

        let recalled_experiences: Vec<Experience> = recalled
            .iter()
            .map(|s| serde_json::from_value(s.payload.clone()).expect("experience payload"))
            .collect();
        assert_eq!(recalled_experiences[0].what, late.what);
        assert_eq!(recalled_experiences[1].what, early.what);
        assert_eq!(
            recalled[0].payload["original_experience_id"],
            json!(late.id)
        );
        assert_eq!(
            recalled[0].payload["original_occurred_at"],
            json!(late.occurred_at)
        );
        assert_eq!(
            recalled[1].payload["original_experience_id"],
            json!(early.id)
        );
        assert_eq!(
            recalled[1].payload["original_occurred_at"],
            json!(early.occurred_at)
        );

        pipeline.observe(Sensation::new("vision.frame", "camera", t1, t1, json!({})));

        let memory_sensation_times: Vec<_> = pipeline
            .timeline()
            .entries()
            .iter()
            .filter_map(|entry| match entry {
                TimelineEntry::Sensation(s) if s.kind == "memory.related_experience" => {
                    Some(s.occurred_at)
                }
                _ => None,
            })
            .collect();
        assert_eq!(memory_sensation_times.len(), 2);
        assert!(memory_sensation_times.iter().all(|t| *t > t2));
    }

    #[test]
    fn observe_supports_recursive_experience_feedback() {
        let mut pipeline = Pipeline::default()
            .with_faculty(MockFaculty::new(
                "speech",
                vec![
                    MockFacultyRule::impression("audio.utterance", "Heard utterance: {text}"),
                    MockFacultyRule::impression(
                        "memory.related_experience",
                        "Memory sensation observed.",
                    ),
                ],
            ))
            .with_wit(MockWit::new(
                "semantic",
                vec![
                    MockWitRule::new("Heard utterance", "A person spoke."),
                    MockWitRule::new("Memory sensation observed", "A layered meaning emerged."),
                ],
            ));

        let sensation = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        let experiences = pipeline.observe(sensation);

        assert!(experiences.iter().any(|e| e.what == "A person spoke."));
        assert!(
            experiences
                .iter()
                .any(|e| e.what == "A layered meaning emerged.")
        );
        assert!(pipeline.timeline().entries().iter().any(|entry| {
            matches!(
                entry,
                TimelineEntry::Sensation(s) if s.kind == "memory.related_experience"
            )
        }));
    }

    #[test]
    fn observe_caps_recursive_feedback_loops() {
        #[derive(Default)]
        struct LoopFaculty;

        impl Faculty for LoopFaculty {
            fn process(&mut self, sensation: &Sensation) -> (Vec<Sensation>, Vec<Impression>) {
                if sensation.kind != "audio.utterance"
                    && sensation.kind != "memory.related_experience"
                {
                    return (vec![], vec![]);
                }

                (
                    vec![],
                    vec![Impression::new(
                        vec![sensation.id],
                        sensation.occurred_at,
                        sensation.observed_at,
                        "loop impression",
                    )],
                )
            }
        }

        #[derive(Default)]
        struct LoopWit {
            count: usize,
        }

        impl Wit for LoopWit {
            fn interpret(&mut self, frame: &TimelineFrame) -> Vec<Experience> {
                let Some(impression) = frame.entries().iter().rev().find_map(|entry| match entry {
                    TimelineEntry::Impression(impression) => Some(impression),
                    _ => None,
                }) else {
                    return vec![];
                };

                self.count += 1;
                vec![Experience::new(
                    vec![impression.id],
                    impression.occurred_at,
                    impression.observed_at,
                    format!("loop meaning {}", self.count),
                )]
            }
        }

        let t0 = now();
        let mut pipeline = Pipeline::default()
            .with_faculty(LoopFaculty)
            .with_wit(LoopWit::default());
        let experiences = pipeline.observe(Sensation::new(
            "audio.utterance",
            "mic",
            t0,
            t0,
            json!({ "text": "hello" }),
        ));

        assert_eq!(experiences.len(), MAX_EXPERIENCE_FEEDBACK_LOOPS);
        assert_eq!(
            pipeline.memory().recall().len(),
            MAX_EXPERIENCE_FEEDBACK_LOOPS
        );
    }
}
