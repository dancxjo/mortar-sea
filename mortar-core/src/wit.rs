use crate::{
    experience::Experience,
    timeline::{TimelineEntry, TimelineFrame},
};

/// A Wit understands things over time.
///
/// Wits consume a [`TimelineFrame`] and produce [`Experience`]s—meaning
/// extracted from the ordered stream of sensations and impressions. A Wit
/// might, for example, notice that a sequence of face observations followed by
/// a speech observation implies a greeting.
///
/// No concrete implementations are provided here; this trait defines the
/// contract only. In particular, language models and inference engines are
/// explicitly out of scope for this crate.
pub trait Wit {
    /// Given a frame of recent cognitive events, derive zero or more
    /// experiences (interpretations).
    fn interpret(&mut self, frame: &TimelineFrame) -> Vec<Experience>;
}

/// Metadata for a registered wit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredWit {
    pub name: String,
    pub priority: i32,
    pub filter: WitFilter,
    pub cadence: WitCadence,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitRegistryError {
    DuplicateName(String),
}

/// Declares when a wit should be considered for execution.
///
/// Cadence hooks are intentionally lightweight so richer schedulers can be
/// layered in front of this registry later.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WitCadence {
    #[default]
    EveryObserve,
    ScheduledHook(String),
}

/// Declares what timeline content should activate a wit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitFilter {
    Any,
    ImpressionType(String),
    ExperienceType(String),
}

impl WitFilter {
    fn matches(&self, frame: &TimelineFrame) -> bool {
        match self {
            Self::Any => true,
            Self::ImpressionType(expected_type) => {
                frame.entries().iter().any(|entry| match entry {
                    TimelineEntry::Impression(impression) => {
                        timeline_text_type(&impression.how) == expected_type
                    }
                    _ => false,
                })
            }
            Self::ExperienceType(expected_type) => {
                frame.entries().iter().any(|entry| match entry {
                    TimelineEntry::Experience(experience) => {
                        timeline_text_type(&experience.what) == expected_type
                    }
                    _ => false,
                })
            }
        }
    }
}

/// A backend-agnostic registry for declaratively wiring wits by stable name.
#[derive(Default)]
pub struct WitRegistry {
    entries: Vec<WitRegistryEntry>,
}

struct WitRegistryEntry {
    registered: RegisteredWit,
    build: Box<dyn Fn() -> Box<dyn Wit>>,
}

impl WitRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<F>(
        &mut self,
        name: impl Into<String>,
        priority: i32,
        filter: WitFilter,
        cadence: WitCadence,
        build: F,
    ) -> Result<(), WitRegistryError>
    where
        F: Fn() -> Box<dyn Wit> + 'static,
    {
        let name = name.into();
        if self
            .entries
            .iter()
            .any(|entry| entry.registered.name == name)
        {
            return Err(WitRegistryError::DuplicateName(name));
        }

        self.entries.push(WitRegistryEntry {
            registered: RegisteredWit {
                name,
                priority,
                filter,
                cadence,
            },
            build: Box::new(build),
        });
        Ok(())
    }

    pub fn list(&self) -> Vec<RegisteredWit> {
        self.entries
            .iter()
            .map(|entry| entry.registered.clone())
            .collect()
    }

    pub fn select_for_frame(&self, frame: &TimelineFrame) -> Vec<Box<dyn Wit>> {
        let mut selected: Vec<_> = self
            .entries
            .iter()
            .filter(|entry| entry.registered.filter.matches(frame))
            .collect();

        selected.sort_by(|left, right| {
            right
                .registered
                .priority
                .cmp(&left.registered.priority)
                .then_with(|| left.registered.name.cmp(&right.registered.name))
        });

        selected.into_iter().map(|entry| (entry.build)()).collect()
    }
}

fn timeline_text_type(value: &str) -> &str {
    value
        .split_once(':')
        .map(|(left, _)| left)
        .unwrap_or(value)
        .trim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{impression::Impression, sensation::Sensation, time::now};
    use serde_json::json;

    struct DeterministicWit {
        label: String,
    }

    impl Wit for DeterministicWit {
        fn interpret(&mut self, frame: &TimelineFrame) -> Vec<Experience> {
            let occurred_at = frame
                .entries()
                .first()
                .map(TimelineEntry::occurred_at)
                .unwrap_or_else(now);
            vec![Experience::new(
                vec![],
                occurred_at,
                now(),
                format!("{}: understood", self.label),
            )]
        }
    }

    #[test]
    fn registry_registers_and_lists_wits() {
        let mut registry = WitRegistry::new();
        registry
            .register(
                "semantic",
                10,
                WitFilter::Any,
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "semantic".to_owned(),
                    })
                },
            )
            .expect("register semantic");
        registry
            .register(
                "social",
                20,
                WitFilter::ImpressionType("social.intent".to_owned()),
                WitCadence::ScheduledHook("future.social".to_owned()),
                || {
                    Box::new(DeterministicWit {
                        label: "social".to_owned(),
                    })
                },
            )
            .expect("register social");

        assert_eq!(
            registry.list(),
            vec![
                RegisteredWit {
                    name: "semantic".to_owned(),
                    priority: 10,
                    filter: WitFilter::Any,
                    cadence: WitCadence::EveryObserve,
                },
                RegisteredWit {
                    name: "social".to_owned(),
                    priority: 20,
                    filter: WitFilter::ImpressionType("social.intent".to_owned()),
                    cadence: WitCadence::ScheduledHook("future.social".to_owned()),
                },
            ]
        );
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let mut registry = WitRegistry::new();
        registry
            .register(
                "semantic",
                0,
                WitFilter::Any,
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "semantic".to_owned(),
                    })
                },
            )
            .expect("register semantic");

        let duplicate = registry.register(
            "semantic",
            1,
            WitFilter::Any,
            WitCadence::EveryObserve,
            || {
                Box::new(DeterministicWit {
                    label: "semantic-duplicate".to_owned(),
                })
            },
        );
        assert_eq!(
            duplicate,
            Err(WitRegistryError::DuplicateName("semantic".to_owned()))
        );
    }

    #[test]
    fn registry_prioritizes_and_filters_by_timeline_types() {
        let mut registry = WitRegistry::new();
        registry
            .register(
                "fallback",
                1,
                WitFilter::Any,
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "fallback".to_owned(),
                    })
                },
            )
            .expect("register fallback");
        registry
            .register(
                "social",
                30,
                WitFilter::ImpressionType("social.intent".to_owned()),
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "social".to_owned(),
                    })
                },
            )
            .expect("register social");
        registry
            .register(
                "recall",
                20,
                WitFilter::ExperienceType("memory.recall".to_owned()),
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "recall".to_owned(),
                    })
                },
            )
            .expect("register recall");
        registry
            .register(
                "vision",
                40,
                WitFilter::ImpressionType("vision.object".to_owned()),
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "vision".to_owned(),
                    })
                },
            )
            .expect("register vision");

        let t0 = now();
        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(Sensation::new(
            "audio.utterance",
            "mic",
            t0,
            t0,
            json!({ "text": "hello" }),
        )));
        frame.push(TimelineEntry::Impression(Impression::new(
            vec![],
            t0,
            t0,
            "social.intent: greeting",
        )));
        frame.push(TimelineEntry::Experience(Experience::new(
            vec![],
            t0,
            t0,
            "memory.recall: prior greeting",
        )));

        let mut selected = registry.select_for_frame(&frame);
        let outputs: Vec<String> = selected
            .iter_mut()
            .flat_map(|wit| wit.interpret(&frame))
            .map(|experience| experience.what)
            .collect();

        assert_eq!(
            outputs,
            vec![
                "social: understood".to_owned(),
                "recall: understood".to_owned(),
                "fallback: understood".to_owned(),
            ]
        );
    }

    #[test]
    fn behavior_multiple_selected_wits_produce_distinct_experiences() {
        let mut registry = WitRegistry::new();
        registry
            .register(
                "semantic",
                10,
                WitFilter::Any,
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "semantic".to_owned(),
                    })
                },
            )
            .expect("register semantic");
        registry
            .register(
                "social",
                10,
                WitFilter::Any,
                WitCadence::EveryObserve,
                || {
                    Box::new(DeterministicWit {
                        label: "social".to_owned(),
                    })
                },
            )
            .expect("register social");

        let t0 = now();
        let mut frame = TimelineFrame::new();
        frame.push(TimelineEntry::Sensation(Sensation::new(
            "audio.utterance",
            "mic",
            t0,
            t0,
            json!({ "text": "hello" }),
        )));

        let mut selected = registry.select_for_frame(&frame);
        let outputs: Vec<String> = selected
            .iter_mut()
            .flat_map(|wit| wit.interpret(&frame))
            .map(|experience| experience.what)
            .collect();

        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|output| output == "semantic: understood"));
        assert!(outputs.iter().any(|output| output == "social: understood"));
    }
}
