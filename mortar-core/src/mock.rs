use serde_json::{json, Value};

use crate::{
    experience::Experience,
    faculty::Faculty,
    impression::Impression,
    memory::InMemory,
    pipeline::Pipeline,
    sensation::Sensation,
    time::now,
    timeline::{TimelineEntry, TimelineFrame},
    wit::Wit,
};

/// A scripted source of sensations for behavior tests and local prototypes.
///
/// `MockEmitter` is deliberately small: it does not sleep, poll hardware, or
/// manage streams. It just lets tests feed known sensations into the same
/// pipeline that real sensors will use later.
#[derive(Debug, Default)]
pub struct MockEmitter {
    sensations: Vec<Sensation>,
}

impl MockEmitter {
    pub fn new(sensations: Vec<Sensation>) -> Self {
        Self { sensations }
    }

    pub fn text(source: impl Into<String>, text: impl Into<String>) -> Self {
        let t = now();
        Self::new(vec![Sensation::new(
            "audio.utterance",
            source,
            t,
            t,
            json!({ "text": text.into() }),
        )])
    }

    pub fn drain(&mut self) -> Vec<Sensation> {
        self.sensations.drain(..).collect()
    }
}

/// A deterministic faculty rule used by [`MockFaculty`].
#[derive(Debug, Clone)]
pub struct MockFacultyRule {
    pub input_kind: String,
    pub impression: String,
    pub derived_sensation: Option<MockDerivedSensation>,
}

impl MockFacultyRule {
    pub fn impression(input_kind: impl Into<String>, impression: impl Into<String>) -> Self {
        Self {
            input_kind: input_kind.into(),
            impression: impression.into(),
            derived_sensation: None,
        }
    }

    pub fn with_derived_sensation(
        input_kind: impl Into<String>,
        impression: impl Into<String>,
        derived_kind: impl Into<String>,
        derived_payload: Value,
    ) -> Self {
        Self {
            input_kind: input_kind.into(),
            impression: impression.into(),
            derived_sensation: Some(MockDerivedSensation {
                kind: derived_kind.into(),
                payload: derived_payload,
            }),
        }
    }
}

/// A sensation emitted by a mock faculty after it notices something.
#[derive(Debug, Clone)]
pub struct MockDerivedSensation {
    pub kind: String,
    pub payload: Value,
}

/// A deterministic faculty backend.
///
/// Rules match by `Sensation.kind`. The impression text supports a few tiny
/// placeholders so behavior tests can stay readable:
///
/// - `{kind}`
/// - `{source}`
/// - `{text}` from `payload.text`
#[derive(Debug, Default)]
pub struct MockFaculty {
    pub name: String,
    pub rules: Vec<MockFacultyRule>,
}

impl MockFaculty {
    pub fn new(name: impl Into<String>, rules: Vec<MockFacultyRule>) -> Self {
        Self {
            name: name.into(),
            rules,
        }
    }
}

impl Faculty for MockFaculty {
    fn process(&mut self, sensation: &Sensation) -> (Vec<Sensation>, Vec<Impression>) {
        let mut sensations = Vec::new();
        let mut impressions = Vec::new();

        for rule in self.rules.iter().filter(|rule| rule.input_kind == sensation.kind) {
            let observed_at = now();
            impressions.push(Impression::new(
                vec![sensation.id],
                sensation.occurred_at,
                observed_at,
                render_template(&rule.impression, sensation),
            ));

            if let Some(derived) = &rule.derived_sensation {
                sensations.push(Sensation::new(
                    derived.kind.clone(),
                    self.name.clone(),
                    sensation.occurred_at,
                    observed_at,
                    derived.payload.clone(),
                ));
            }
        }

        (sensations, impressions)
    }
}

/// A deterministic wit rule used by [`MockWit`].
#[derive(Debug, Clone)]
pub struct MockWitRule {
    pub impression_contains: String,
    pub experience: String,
}

impl MockWitRule {
    pub fn new(impression_contains: impl Into<String>, experience: impl Into<String>) -> Self {
        Self {
            impression_contains: impression_contains.into(),
            experience: experience.into(),
        }
    }
}

/// A deterministic Wit backend.
///
/// It scans the timeline for matching impressions and emits experiences. This
/// is intentionally simple, but it exercises the real `Wit` trait and timeline
/// shape end to end.
#[derive(Debug, Default)]
pub struct MockWit {
    pub name: String,
    pub rules: Vec<MockWitRule>,
}

impl MockWit {
    pub fn new(name: impl Into<String>, rules: Vec<MockWitRule>) -> Self {
        Self {
            name: name.into(),
            rules,
        }
    }
}

impl Wit for MockWit {
    fn interpret(&mut self, frame: &TimelineFrame) -> Vec<Experience> {
        let mut experiences = Vec::new();

        for entry in frame.entries() {
            let TimelineEntry::Impression(impression) = entry else {
                continue;
            };

            for rule in self
                .rules
                .iter()
                .filter(|rule| impression.how.contains(&rule.impression_contains))
            {
                experiences.push(Experience::new(
                    vec![impression.id],
                    impression.occurred_at,
                    now(),
                    rule.experience.clone(),
                ));
            }
        }

        experiences
    }
}

/// A small end-to-end harness backed by deterministic mock components.
#[derive(Default)]
pub struct MockCognition {
    pipeline: Pipeline<InMemory>,
}

impl MockCognition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_faculty(mut self, faculty: impl Faculty + 'static) -> Self {
        self.pipeline = self.pipeline.with_faculty(faculty);
        self
    }

    pub fn with_wit(mut self, wit: impl Wit + 'static) -> Self {
        self.pipeline = self.pipeline.with_wit(wit);
        self
    }

    pub fn timeline(&self) -> &TimelineFrame {
        self.pipeline.timeline()
    }

    pub fn memory(&self) -> &InMemory {
        self.pipeline.memory()
    }

    /// Present a sensation to the mock system and run the full core loop:
    /// sensation → faculty outputs → timeline → wit outputs → memory.
    ///
    /// Wits interpret the full accumulated timeline on every call. To avoid
    /// duplicate accumulation across observations, experiences that match an
    /// already-stored experience are skipped before storage.
    pub fn observe(&mut self, sensation: Sensation) -> Vec<Experience> {
        self.pipeline.observe(sensation)
    }

    /// Re-enter every recalled experience as an ordinary memory sensation.
    pub fn recall_into_timeline(&mut self) -> Vec<Sensation> {
        self.pipeline.recall_into_timeline()
    }
}

fn render_template(template: &str, sensation: &Sensation) -> String {
    let text = sensation
        .payload
        .get("text")
        .and_then(Value::as_str)
        .unwrap_or("");

    template
        .replace("{kind}", &sensation.kind)
        .replace("{source}", &sensation.source)
        .replace("{text}", text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Memory;
    use chrono::Duration;

    #[test]
    fn mock_faculty_renders_text_impression() {
        let mut emitter = MockEmitter::text("mic", "hello");
        let sensation = emitter.drain().pop().expect("scripted sensation");
        let mut faculty = MockFaculty::new(
            "mock_speech",
            vec![MockFacultyRule::impression(
                "audio.utterance",
                "The speaker said {text}.",
            )],
        );

        let (_sensations, impressions) = faculty.process(&sensation);

        assert_eq!(impressions.len(), 1);
        assert_eq!(impressions[0].how, "The speaker said hello.");
    }

    #[test]
    fn behavior_pipeline_sensation_to_memory() {
        let mut cognition = MockCognition::new()
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
        let experiences = cognition.observe(sensation);

        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].what, "A person spoke.");

        let recalled = cognition.memory().recall();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].what, "A person spoke.");

        assert!(cognition
            .timeline()
            .entries()
            .iter()
            .any(|e| matches!(e, TimelineEntry::Sensation(_))));
        assert!(cognition
            .timeline()
            .entries()
            .iter()
            .any(|e| matches!(e, TimelineEntry::Impression(_))));
        assert!(cognition
            .timeline()
            .entries()
            .iter()
            .any(|e| matches!(e, TimelineEntry::Experience(_))));
    }

    #[test]
    fn behavior_experience_recall_becomes_memory_sensation() {
        const TEST_TIME_DELTA_SECONDS: i64 = 10;

        let mut cognition = MockCognition::new()
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

        let earlier_time = now();
        let later_time = earlier_time + Duration::seconds(TEST_TIME_DELTA_SECONDS);

        cognition.observe(Sensation::new(
            "audio.utterance",
            "mic",
            earlier_time,
            earlier_time,
            json!({ "text": "hello" }),
        ));
        cognition.observe(Sensation::new(
            "vision.frame",
            "camera_external",
            later_time,
            later_time,
            json!({}),
        ));

        let sensation_count_before_recall = cognition
            .timeline()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, TimelineEntry::Sensation(_)))
            .count();

        let recalled = cognition.recall_into_timeline();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].kind, "memory.related_experience");
        assert_eq!(recalled[0].source, "memory");
        assert_eq!(recalled[0].occurred_at, earlier_time);

        let recovered: Experience =
            serde_json::from_value(recalled[0].payload.clone()).expect("experience payload");
        assert_eq!(recovered.what, "A person spoke.");

        let entries = cognition.timeline().entries();
        let sensation_count_after_recall = entries
            .iter()
            .filter(|entry| matches!(entry, TimelineEntry::Sensation(_)))
            .count();
        assert_eq!(sensation_count_after_recall, sensation_count_before_recall + 1);

        let memory_sensation_index = entries
            .iter()
            .position(|entry| {
                matches!(
                    entry,
                    TimelineEntry::Sensation(s) if s.kind == "memory.related_experience"
                )
            })
            .expect("memory sensation should be inserted into timeline");
        let external_sensation_index = entries
            .iter()
            .position(|entry| {
                matches!(entry, TimelineEntry::Sensation(s) if s.source == "camera_external")
            })
            .expect("external sensation should be present in timeline");
        assert!(
            memory_sensation_index < external_sensation_index,
            "memory sensation at index {} must appear before external sensation at index {} for chronological ordering",
            memory_sensation_index,
            external_sensation_index
        );
    }

    #[test]
    fn behavior_timeline_orders_heterogeneous_entries() {
        let mut cognition = MockCognition::new()
            .with_faculty(MockFaculty::new(
                "vision",
                vec![MockFacultyRule::impression(
                    "vision.frame",
                    "Observed frame from {source}",
                )],
            ))
            .with_wit(MockWit::new(
                "scene",
                vec![MockWitRule::new("Observed frame", "A frame was observed.")],
            ));

        let earlier_time = now();
        let later_time = earlier_time + Duration::seconds(10);

        cognition.observe(Sensation::new(
            "vision.frame",
            "camera_late",
            later_time,
            later_time,
            json!({}),
        ));
        cognition.observe(Sensation::new(
            "vision.frame",
            "camera_early",
            earlier_time,
            earlier_time,
            json!({}),
        ));

        let entries = cognition.timeline().entries();
        let times: Vec<_> = entries.iter().map(TimelineEntry::occurred_at).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
        assert!(entries.iter().any(|e| matches!(e, TimelineEntry::Sensation(_))));
        assert!(entries.iter().any(|e| matches!(e, TimelineEntry::Impression(_))));
        assert!(entries.iter().any(|e| matches!(e, TimelineEntry::Experience(_))));
    }

    #[test]
    fn behavior_multiple_faculties_process_same_sensation() {
        let mut cognition = MockCognition::new()
            .with_faculty(MockFaculty::new(
                "speech",
                vec![MockFacultyRule::impression(
                    "audio.utterance",
                    "Speech faculty heard {text}",
                )],
            ))
            .with_faculty(MockFaculty::new(
                "context",
                vec![MockFacultyRule::impression(
                    "audio.utterance",
                    "Context faculty noticed source {source}",
                )],
            ));

        let sensation = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        cognition.observe(sensation);

        let impressions: Vec<&Impression> = cognition
            .timeline()
            .entries()
            .iter()
            .filter_map(|entry| match entry {
                TimelineEntry::Impression(impression) => Some(impression),
                _ => None,
            })
            .collect();

        assert_eq!(impressions.len(), 2);
        assert!(
            impressions
                .iter()
                .any(|i| i.how == "Speech faculty heard hello")
        );
        assert!(
            impressions
                .iter()
                .any(|i| i.how == "Context faculty noticed source mic")
        );
    }

    #[test]
    fn behavior_multiple_wits_derive_from_same_timeline() {
        let mut cognition = MockCognition::new()
            .with_faculty(MockFaculty::new(
                "speech",
                vec![MockFacultyRule::impression(
                    "audio.utterance",
                    "Heard utterance: {text}",
                )],
            ))
            .with_wit(MockWit::new(
                "semantic",
                vec![MockWitRule::new("Heard utterance", "A person spoke.")],
            ))
            .with_wit(MockWit::new(
                "social",
                vec![MockWitRule::new("Heard utterance", "Someone attempted interaction.")],
            ));

        let sensation = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        let experiences = cognition.observe(sensation);

        assert_eq!(experiences.len(), 2);
        assert!(experiences.iter().any(|e| e.what == "A person spoke."));
        assert!(
            experiences
                .iter()
                .any(|e| e.what == "Someone attempted interaction.")
        );
        assert_eq!(cognition.memory().recall().len(), 2);
    }

    #[test]
    fn behavior_observe_deduplicates_reinterpreted_experiences() {
        let mut cognition = MockCognition::new()
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

        let first = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        let first_experiences = cognition.observe(first);
        assert_eq!(first_experiences.len(), 1);
        assert_eq!(cognition.memory().recall().len(), 1);

        let second_experiences = cognition.observe(Sensation::new(
            "vision.frame",
            "camera_0",
            now(),
            now(),
            json!({}),
        ));
        assert!(second_experiences.is_empty());
        assert_eq!(cognition.memory().recall().len(), 1);

        let timeline_experience_count = cognition
            .timeline()
            .entries()
            .iter()
            .filter(|entry| matches!(entry, TimelineEntry::Experience(_)))
            .count();
        assert_eq!(timeline_experience_count, 1);
    }
}
