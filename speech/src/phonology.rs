use serde::{Deserialize, Serialize};

use crate::acoustics::AcousticObservation;
use crate::evidence::EvidenceProvenance;
use crate::feature::FeatureBundle;
use crate::ids::{PhoneId, PhonemeId};
use crate::segment::SegmentStatus;
use crate::spec::Spec;
use crate::time::TimeSpan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phoneme {
    pub id: PhonemeId,
    pub notation: String,
    pub features: FeatureBundle,
    pub default_phone: Option<PhoneId>,
    pub possible_phones: Vec<PhoneId>,
    pub status: SegmentStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct PhonemeInventory {
    pub phonemes: std::collections::HashMap<PhonemeId, Phoneme>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhoneToken {
    pub phone: Spec<PhoneId>,
    pub span: Option<TimeSpan>,
    pub features: FeatureBundle,
    pub acoustic_evidence: Vec<AcousticObservation>,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemeToken {
    pub phoneme: Spec<PhonemeId>,
    pub span: Option<TimeSpan>,
    pub realized_as: Vec<PhoneToken>,
    pub confidence: f32,
    pub provenance: EvidenceProvenance,
}
