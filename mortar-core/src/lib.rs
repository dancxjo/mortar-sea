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

pub mod experience;
pub mod faculty;
pub mod impression;
pub mod memory;
pub mod mock;
pub mod pipeline;
pub mod sensation;
pub mod time;
pub mod timeline;
pub mod wit;

pub use experience::Experience;
pub use faculty::{Faculty, FacultyRegistry, FacultyRegistryError, RegisteredFaculty};
pub use impression::Impression;
pub use memory::{InMemory, Memory};
pub use mock::*;
pub use pipeline::Pipeline;
pub use sensation::Sensation;
pub use timeline::{TimelineEntry, TimelineFrame};
pub use wit::Wit;
