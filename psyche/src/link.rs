use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The nature of a directed relationship between two [`Experience`](crate::experience::Experience)s.
///
/// Links are always directed: `from` → `to`. The kind describes why the link
/// was drawn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperienceLinkKind {
    /// The source experience caused or enabled the target.
    ///
    /// Example: "A visitor arrived" caused "The door was opened".
    Causal,

    /// The two experiences are socially connected — they involve the same
    /// people, agent, or social context.
    ///
    /// Example: Two experiences that both reference the same person.
    Social,

    /// The target experience followed the source in an explicit narrative
    /// sequence without implying full causation.
    ///
    /// Example: consecutive steps in an observed routine.
    Sequential,

    /// An application-defined relationship kind not covered by the variants
    /// above. Use this to prototype new link semantics before they are
    /// elevated to first-class variants.
    Custom(String),
}

/// A directed relationship between two [`Experience`](crate::experience::Experience)s.
///
/// Links are the primitive edge type for experience-to-experience graphs. They
/// are backend-independent: any memory store can record, query, and traverse
/// them. Future graph and vector backends are expected to map `ExperienceLink`s
/// to native edge representations.
///
/// ## Directionality
///
/// A link is always directed: `from_id` → `to_id`. This mirrors how causal
/// and narrative relationships work in practice — one event leads to another,
/// rather than them being purely symmetric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceLink {
    /// Unique identifier for this link.
    pub id: Uuid,
    /// The source experience.
    pub from_id: Uuid,
    /// The target experience.
    pub to_id: Uuid,
    /// Why the link was drawn.
    pub kind: ExperienceLinkKind,
    /// When this link was recorded.
    pub created_at: DateTime<Utc>,
}

impl ExperienceLink {
    /// Create a new link with a freshly generated UUID, timestamped now.
    pub fn new(from_id: Uuid, to_id: Uuid, kind: ExperienceLinkKind) -> Self {
        Self {
            id: Uuid::new_v4(),
            from_id,
            to_id,
            kind,
            created_at: crate::time::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::now;

    #[test]
    fn link_records_direction_and_kind() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let link = ExperienceLink::new(a, b, ExperienceLinkKind::Causal);
        assert_eq!(link.from_id, a);
        assert_eq!(link.to_id, b);
        assert_eq!(link.kind, ExperienceLinkKind::Causal);
    }

    #[test]
    fn link_roundtrips_through_json() {
        let link = ExperienceLink::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            ExperienceLinkKind::Custom("test.relation".to_owned()),
        );
        let json = serde_json::to_string(&link).expect("serialize");
        let back: ExperienceLink = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(link.id, back.id);
        assert_eq!(link.from_id, back.from_id);
        assert_eq!(link.to_id, back.to_id);
    }

    #[test]
    fn social_and_sequential_links_are_distinct_kinds() {
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let social = ExperienceLink::new(a, b, ExperienceLinkKind::Social);
        let seq = ExperienceLink::new(a, b, ExperienceLinkKind::Sequential);
        assert_ne!(social.kind, seq.kind);
        assert!(social.created_at <= now());
    }
}
