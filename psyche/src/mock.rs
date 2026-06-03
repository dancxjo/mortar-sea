use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    experience::Experience,
    faculty::Faculty,
    impression::Impression,
    memory::{InMemory, Memory},
    pipeline::Pipeline,
    sensation::Sensation,
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
        Self::text_at(source, text, scripted_epoch())
    }

    pub fn text_at(
        source: impl Into<String>,
        text: impl Into<String>,
        occurred_at: DateTime<Utc>,
    ) -> Self {
        let source = source.into();
        let text = text.into();
        Self::new(vec![Sensation {
            id: deterministic_uuid(format!(
                "mock-emitter:text:{}:{}:{}",
                source,
                text,
                occurred_at.to_rfc3339()
            )),
            kind: "audio.utterance".to_owned(),
            source,
            occurred_at,
            observed_at: occurred_at,
            payload: json!({ "text": text }),
        }])
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

        for rule in self
            .rules
            .iter()
            .filter(|rule| rule.input_kind == sensation.kind)
        {
            let how = render_template(&rule.impression, sensation);
            let observed_at = sensation.observed_at;
            impressions.push(Impression {
                id: deterministic_uuid(format!(
                    "mock-faculty:{}:{}:{}",
                    self.name, sensation.id, how
                )),
                sensation_ids: vec![sensation.id],
                occurred_at: sensation.occurred_at,
                observed_at,
                how,
            });

            if let Some(derived) = &rule.derived_sensation {
                sensations.push(Sensation {
                    id: deterministic_uuid(format!(
                        "mock-faculty-derived:{}:{}:{}:{}",
                        self.name, sensation.id, derived.kind, derived.payload
                    )),
                    kind: derived.kind.clone(),
                    source: self.name.clone(),
                    occurred_at: sensation.occurred_at,
                    observed_at,
                    payload: derived.payload.clone(),
                });
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
                experiences.push(Experience {
                    id: deterministic_uuid(format!(
                        "mock-wit:{}:{}:{}",
                        self.name, impression.id, rule.experience
                    )),
                    impression_ids: vec![impression.id],
                    occurred_at: impression.occurred_at,
                    observed_at: impression.observed_at,
                    what: rule.experience.clone(),
                });
            }
        }

        experiences
    }
}

/// A deterministic memory backend for scripted tests and examples.
///
/// `ScriptedMemory` starts with a known set of experiences and appends any
/// newly stored experiences in insertion order.
#[derive(Debug, Default, Clone)]
pub struct ScriptedMemory {
    experiences: Vec<Experience>,
}

impl ScriptedMemory {
    pub fn new(experiences: Vec<Experience>) -> Self {
        Self { experiences }
    }
}

impl Memory for ScriptedMemory {
    fn store(&mut self, experience: Experience) {
        self.experiences.push(experience);
    }

    fn recall(&self) -> Vec<Experience> {
        self.experiences.clone()
    }

    fn experience_to_sensation(experience: &Experience) -> Sensation {
        let payload = serde_json::to_value(experience).unwrap_or_else(|err| {
            panic!("failed to serialize Experience to JSON payload: {}", err)
        });
        Sensation {
            id: deterministic_uuid(format!("mock-memory:{}", experience.id)),
            kind: "memory.related_experience".to_owned(),
            source: "memory".to_owned(),
            occurred_at: experience.occurred_at,
            observed_at: experience.observed_at,
            payload,
        }
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

fn deterministic_uuid(seed: impl AsRef<str>) -> Uuid {
    // Non-cryptographic FNV-1a 128-bit hash to obtain a stable UUID-shaped
    // identifier for deterministic test/example data generation.
    // Not suitable for cryptographic or collision-resistant identity needs.
    // Constants are the standard FNV-1a offset basis and prime for 128-bit.
    const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
    const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

    let mut hash = OFFSET;
    for byte in seed.as_ref().as_bytes() {
        hash ^= u128::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }

    Uuid::from_u128(hash)
}

fn scripted_epoch() -> DateTime<Utc> {
    // Use a fixed timestamp so mock-generated sensations are reproducible.
    DateTime::UNIX_EPOCH
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::now;
    use crate::Memory;
    use chrono::Duration;

    #[test]
    fn mock_emitter_text_is_deterministic() {
        let first = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        let second = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");

        assert_eq!(first.id, second.id);
        assert_eq!(first.occurred_at, second.occurred_at);
        assert_eq!(first.observed_at, second.observed_at);
        assert_eq!(first.payload, second.payload);
    }

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
    fn scripted_memory_recall_and_sensation_are_deterministic() {
        let occurred_at = now();
        let observed_at = occurred_at + Duration::seconds(1);
        let experience = Experience {
            id: deterministic_uuid("seeded-experience"),
            impression_ids: vec![],
            occurred_at,
            observed_at,
            what: "Scripted meaning.".to_owned(),
        };

        let memory = ScriptedMemory::new(vec![experience.clone()]);
        let recalled = memory.recall();
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].id, experience.id);
        assert_eq!(recalled[0].what, experience.what);

        let first = ScriptedMemory::experience_to_sensation(&experience);
        let second = ScriptedMemory::experience_to_sensation(&experience);
        assert_eq!(first.id, second.id);
        assert_eq!(first.observed_at, second.observed_at);
        assert_eq!(first.payload, second.payload);
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
        assert_eq!(
            sensation_count_after_recall,
            sensation_count_before_recall + 1
        );

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
        assert!(entries
            .iter()
            .any(|e| matches!(e, TimelineEntry::Sensation(_))));
        assert!(entries
            .iter()
            .any(|e| matches!(e, TimelineEntry::Impression(_))));
        assert!(entries
            .iter()
            .any(|e| matches!(e, TimelineEntry::Experience(_))));
    }

    #[test]
    fn behavior_timeline_query_and_reference_navigation() {
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

        let t0 = now();
        let t1 = t0 + Duration::seconds(5);
        let t2 = t0 + Duration::seconds(15);

        cognition.observe(Sensation::new(
            "audio.utterance",
            "mic",
            t0,
            t0,
            json!({ "text": "hello" }),
        ));
        cognition.observe(Sensation::new(
            "audio.utterance",
            "mic",
            t1,
            t1,
            json!({ "text": "again" }),
        ));
        cognition.observe(Sensation::new("vision.frame", "camera", t2, t2, json!({})));

        let frame = cognition.timeline();

        let recent = frame.recent_entries(2);
        assert_eq!(recent.len(), 2);
        assert!(recent[0].occurred_at() <= recent[1].occurred_at());
        assert_eq!(recent[1].occurred_at(), t2);

        let impressions: Vec<_> = frame
            .entries_by_kind(crate::timeline::TimelineEntryKind::Impression)
            .collect();
        assert_eq!(impressions.len(), 2);

        let first_sensation = frame
            .entries()
            .iter()
            .find_map(|entry| match entry {
                TimelineEntry::Sensation(sensation)
                    if sensation.kind == "audio.utterance" && sensation.occurred_at == t0 =>
                {
                    Some(sensation)
                }
                _ => None,
            })
            .expect("first utterance sensation should exist");

        let sensation_refs: Vec<_> = frame
            .entries_referencing_sensation(first_sensation.id)
            .collect();
        assert_eq!(sensation_refs.len(), 1);
        let first_impression = match sensation_refs[0] {
            TimelineEntry::Impression(impression) => impression,
            _ => panic!("sensation references should point to impressions"),
        };

        let impression_refs: Vec<_> = frame
            .entries_referencing_impression(first_impression.id)
            .collect();
        assert_eq!(impression_refs.len(), 1);
        assert!(matches!(impression_refs[0], TimelineEntry::Experience(_)));

        let bounded = frame.entries_between(t0, t1);
        assert!(!bounded.is_empty());
        assert!(bounded.iter().all(|entry| entry.occurred_at() <= t1));
        assert!(bounded.iter().any(
            |entry| matches!(entry, TimelineEntry::Sensation(s) if s.id == first_sensation.id)
        ));

        let related = frame.related_entries(first_impression.id);
        assert_eq!(related.len(), 2);
        assert!(related.iter().any(|entry| matches!(
            entry,
            TimelineEntry::Sensation(s) if s.id == first_sensation.id
        )));
        assert!(related
            .iter()
            .any(|entry| matches!(entry, TimelineEntry::Experience(_))));
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
        assert!(impressions
            .iter()
            .any(|i| i.how == "Speech faculty heard hello"));
        assert!(impressions
            .iter()
            .any(|i| i.how == "Context faculty noticed source mic"));
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
                vec![MockWitRule::new(
                    "Heard utterance",
                    "Someone attempted interaction.",
                )],
            ));

        let sensation = MockEmitter::text("mic", "hello")
            .drain()
            .pop()
            .expect("scripted sensation");
        let experiences = cognition.observe(sensation);

        assert_eq!(experiences.len(), 2);
        assert!(experiences.iter().any(|e| e.what == "A person spoke."));
        assert!(experiences
            .iter()
            .any(|e| e.what == "Someone attempted interaction."));
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
