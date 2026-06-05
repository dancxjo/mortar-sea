//! psyche — the foundational cognitive model for the mortar-sea system.
//!
//! The central pipeline is:
//!
//! ```text
//! Sensation → Impression → Experience → Memory → Sensation
//! ```
//!
//! A remembered thing re-enters cognition as a new sensation; there is no
//! separate memory pathway.

pub mod context_frame;
pub mod episode;
pub mod experience;
pub mod faculty;
pub mod impression;
pub mod link;
pub mod llama_cpp;
pub mod llm;
pub mod memory;
pub mod mock;
pub mod pipeline;
pub mod realtime_experience;
pub mod sensation;
pub mod time;
pub mod timeline;
pub mod wit;

pub use context_frame::{ContextFrame, DEFAULT_CONTEXT_FRAME_ITEMS};
pub use episode::Episode;
pub use experience::Experience;
pub use faculty::{Faculty, FacultyRegistry, FacultyRegistryError, RegisteredFaculty};
pub use impression::Impression;
pub use link::{ExperienceLink, ExperienceLinkKind};
pub use llama_cpp::{LlamaCppConfig, LlamaCppEngine};
pub use llm::{
    ChatMessage, GenerationId, GenerationImage, GenerationRequest, LlmEngine, LlmEvent,
    MockLlmEngine,
};
pub use memory::{InMemory, InMemoryLinked, LinkedMemory, Memory};
pub use mock::*;
pub use pipeline::Pipeline;
pub use realtime_experience::{RealTimeExperienceConfig, RealTimeExperienceWit};
pub use sensation::{Provenance, ProvenanceKind, Sensation};
pub use timeline::{EventCluster, TimelineEntry, TimelineEntryKind, TimelineFrame, event_clusters};
pub use wit::{RegisteredWit, Wit, WitCadence, WitFilter, WitRegistry, WitRegistryError};
