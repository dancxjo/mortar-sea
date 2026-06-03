//! mortar-core — the foundational cognitive model for the mortar-sea system.
//!
//! The central pipeline is:
//!
//! ```text
//! Sensation → Impression → Experience → Memory → Sensation
//! ```
//!
//! A remembered thing re-enters cognition as a new sensation; there is no
//! separate memory pathway.

pub mod episode;
pub mod experience;
pub mod faculty;
pub mod impression;
pub mod link;
pub mod memory;
pub mod mock;
pub mod pipeline;
pub mod sensation;
pub mod time;
pub mod timeline;
pub mod wit;

pub use episode::Episode;
pub use experience::Experience;
pub use faculty::{Faculty, FacultyRegistry, FacultyRegistryError, RegisteredFaculty};
pub use impression::Impression;
pub use link::{ExperienceLink, ExperienceLinkKind};
pub use memory::{InMemory, InMemoryLinked, LinkedMemory, Memory};
pub use mock::*;
pub use pipeline::Pipeline;
pub use sensation::Sensation;
pub use timeline::{TimelineEntry, TimelineEntryKind, TimelineFrame};
pub use wit::{RegisteredWit, Wit, WitCadence, WitFilter, WitRegistry, WitRegistryError};
