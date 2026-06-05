use serde::{Deserialize, Serialize};

use crate::acoustics::AcousticProfile;
use crate::feature::FeatureSystem;
use crate::ids::{LanguageId, VariantId};
use crate::morphology::Morphology;
use crate::orthography::Orthography;
use crate::phonetics::PhoneInventory;
use crate::phonology::PhonemeInventory;
use crate::prosody::ProsodyProfile;
use crate::rules::{AllophoneRule, Phonotactics};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Language {
    pub id: LanguageId,
    pub name: String,
    pub endonym: Option<String>,
    pub iso_639: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinguisticVariant {
    pub id: VariantId,
    pub language: LanguageId,
    pub name: String,
    pub feature_system: FeatureSystem,
    pub phonemes: PhonemeInventory,
    pub phones: PhoneInventory,
    pub allophone_rules: Vec<AllophoneRule>,
    pub phonotactics: Option<Phonotactics>,
    pub orthography: Option<Orthography>,
    pub morphology: Option<Morphology>,
    pub acoustic_profile: Option<AcousticProfile>,
    pub prosody_profile: Option<ProsodyProfile>,
    pub status: VariantStatus,
    pub implementation_status: VariantImplementationStatus,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantStatus {
    Attested,
    Reconstructed,
    Pedagogical,
    Experimental,
    Idiolect,
    SessionLocal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type", content = "data")]
pub enum VariantImplementationStatus {
    Complete,
    StubDerivedFrom(VariantId),
    PermissiveProfile,
}
