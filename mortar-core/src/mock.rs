use serde_json::{json, Value};

use crate::{
    experience::Experience,
    faculty::Faculty,
    impression::Impression,
    memory::{InMemory, Memory},
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
    timeline: TimelineFrame,
    memory: InMemory,
    faculties: Vec<Box<dyn Faculty>>,
    wits: Vec<Box<dyn Wit>>,
}

impl MockCognition {
    pub fn new() -> Self {
        Self::default()
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

    pub fn memory(&self) -> &InMemory {
        &self.memory
    }

    /// Present a sensation to the mock system and run the full core loop:
    /// sensation → faculty outputs → timeline → wit outputs → memory.
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
        for wit in &mut self.wits {
            for experience in wit.interpret(&self.timeline) {
                self.memory.store(experience.clone());
                self.timeline.push(TimelineEntry::Experience(experience.clone()));
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
            .map(InMemory::experience_to_sensation)
            .collect();

        for sensation in &sensations {
            self.timeline
                .push(TimelineEntry::Sensation(sensation.clone()));
        }

        sensations
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
}
