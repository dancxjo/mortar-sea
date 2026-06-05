use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{AcousticCueId, FeatureId, PhoneId, PhonemeId};
use crate::spec::Spec;
use crate::time::TimeSpan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticFrame {
    pub span: TimeSpan,
    pub f0_hz: Spec<f32>,
    pub energy_db: Spec<f32>,
    pub voicing_probability: Spec<f32>,
    pub periodicity: Spec<f32>,
    pub harmonicity: Spec<f32>,
    pub formants: Vec<Formant>,
    pub spectral_centroid_hz: Spec<f32>,
    pub spectral_tilt_db_per_octave: Spec<f32>,
    pub zero_crossing_rate: Spec<f32>,
    pub vectors: Vec<AcousticVector>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Formant {
    pub index: u8,
    pub hz: Spec<f32>,
    pub bandwidth_hz: Spec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticVector {
    pub kind: String,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticObservation {
    pub cue: AcousticCueId,
    pub value: Spec<FeatureValue>,
    pub span: Option<TimeSpan>,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcousticCueDef {
    pub id: AcousticCueId,
    pub name: String,
    pub feature: FeatureId,
    pub targets: Vec<CueTarget>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueTarget {
    Phone(PhoneId),
    Phoneme(PhonemeId),
    Feature(FeatureId),
    Boundary,
    Stress,
    Tone,
    Speaker,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcousticProfile {
    pub cues: HashMap<AcousticCueId, AcousticCueDef>,
    pub phone_models: HashMap<PhoneId, AcousticTargetModel>,
    pub phoneme_models: HashMap<PhonemeId, AcousticTargetModel>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AcousticTargetModel {
    pub expected_features: FeatureBundle,
    pub weighted_cues: Vec<WeightedCue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedCue {
    pub cue: AcousticCueId,
    pub weight: f32,
}
