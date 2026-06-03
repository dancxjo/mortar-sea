use crate::{impression::Impression, sensation::Sensation};

/// A Faculty notices things.
///
/// Faculties operate at the boundary between the world and cognition. A Faculty
/// may consume raw [`Sensation`]s (e.g. a vision faculty consuming camera
/// frames) and emit new `Sensation`s or [`Impression`]s back into the pipeline.
///
/// No concrete implementations are provided here; this trait defines the
/// contract only.
pub trait Faculty {
    /// Called when the system presents a sensation to this faculty.
    ///
    /// Returns zero or more sensations and impressions derived from the input.
    fn process(&mut self, sensation: &Sensation) -> (Vec<Sensation>, Vec<Impression>);
}

/// Metadata for a registered faculty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredFaculty {
    pub name: String,
    pub accepted_kinds: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FacultyRegistryError {
    DuplicateName(String),
}

/// A backend-agnostic registry for declaratively wiring faculties by stable name.
#[derive(Default)]
pub struct FacultyRegistry {
    entries: Vec<FacultyRegistryEntry>,
}

struct FacultyRegistryEntry {
    registered: RegisteredFaculty,
    build: Box<dyn Fn() -> Box<dyn Faculty>>,
}

impl FacultyRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<I, K, F>(
        &mut self,
        name: impl Into<String>,
        accepted_kinds: I,
        build: F,
    ) -> Result<(), FacultyRegistryError>
    where
        I: IntoIterator<Item = K>,
        K: Into<String>,
        F: Fn() -> Box<dyn Faculty> + 'static,
    {
        let name = name.into();
        if self
            .entries
            .iter()
            .any(|entry| entry.registered.name == name)
        {
            return Err(FacultyRegistryError::DuplicateName(name));
        }

        let accepted_kinds = accepted_kinds.into_iter().map(Into::into).collect();
        self.entries.push(FacultyRegistryEntry {
            registered: RegisteredFaculty {
                name,
                accepted_kinds,
            },
            build: Box::new(build),
        });
        Ok(())
    }

    pub fn list(&self) -> Vec<RegisteredFaculty> {
        self.entries
            .iter()
            .map(|entry| entry.registered.clone())
            .collect()
    }

    pub fn select_for_kind(&self, sensation_kind: &str) -> Vec<Box<dyn Faculty>> {
        self.entries
            .iter()
            .filter(|entry| {
                entry
                    .registered
                    .accepted_kinds
                    .iter()
                    .any(|kind| kind == sensation_kind)
            })
            .map(|entry| (entry.build)())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::now;
    use serde_json::json;
    use uuid::Uuid;

    struct DeterministicFaculty {
        name: String,
    }

    impl Faculty for DeterministicFaculty {
        fn process(&mut self, sensation: &Sensation) -> (Vec<Sensation>, Vec<Impression>) {
            (
                vec![],
                vec![Impression {
                    id: Uuid::nil(),
                    sensation_ids: vec![sensation.id],
                    occurred_at: sensation.occurred_at,
                    observed_at: sensation.observed_at,
                    how: format!("{} noticed {}", self.name, sensation.kind),
                }],
            )
        }
    }

    #[test]
    fn registry_registers_and_lists_faculties() {
        let mut registry = FacultyRegistry::new();
        registry
            .register("speech", ["audio.utterance"], || {
                Box::new(DeterministicFaculty {
                    name: "speech".to_owned(),
                })
            })
            .expect("register speech");
        registry
            .register("vision", ["vision.frame"], || {
                Box::new(DeterministicFaculty {
                    name: "vision".to_owned(),
                })
            })
            .expect("register vision");

        assert_eq!(
            registry.list(),
            vec![
                RegisteredFaculty {
                    name: "speech".to_owned(),
                    accepted_kinds: vec!["audio.utterance".to_owned()],
                },
                RegisteredFaculty {
                    name: "vision".to_owned(),
                    accepted_kinds: vec!["vision.frame".to_owned()],
                },
            ]
        );
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let mut registry = FacultyRegistry::new();
        registry
            .register("speech", ["audio.utterance"], || {
                Box::new(DeterministicFaculty {
                    name: "speech".to_owned(),
                })
            })
            .expect("register speech");

        let duplicate = registry.register("speech", ["vision.frame"], || {
            Box::new(DeterministicFaculty {
                name: "speech".to_owned(),
            })
        });
        assert_eq!(
            duplicate,
            Err(FacultyRegistryError::DuplicateName("speech".to_owned()))
        );
    }

    #[test]
    fn behavior_multiple_selected_faculties_react_to_one_sensation() {
        let mut registry = FacultyRegistry::new();
        registry
            .register("speech", ["audio.utterance"], || {
                Box::new(DeterministicFaculty {
                    name: "speech".to_owned(),
                })
            })
            .expect("register speech");
        registry
            .register("context", ["audio.utterance"], || {
                Box::new(DeterministicFaculty {
                    name: "context".to_owned(),
                })
            })
            .expect("register context");
        registry
            .register("vision", ["vision.frame"], || {
                Box::new(DeterministicFaculty {
                    name: "vision".to_owned(),
                })
            })
            .expect("register vision");

        let mut faculties = registry.select_for_kind("audio.utterance");
        let sensation = Sensation::new(
            "audio.utterance",
            "mic",
            now(),
            now(),
            json!({ "text": "hello" }),
        );
        let reactions: Vec<String> = faculties
            .iter_mut()
            .flat_map(|faculty| {
                let (_, impressions) = faculty.process(&sensation);
                impressions.into_iter().map(|impression| impression.how)
            })
            .collect();

        assert_eq!(
            reactions,
            vec![
                "speech noticed audio.utterance".to_owned(),
                "context noticed audio.utterance".to_owned(),
            ]
        );
    }
}
