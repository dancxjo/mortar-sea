use serde::{Deserialize, Serialize};

use crate::feature::{FeatureBundle, FeatureValue};
use crate::ids::{FeatureId, PhoneId, PhonemeId};
use crate::prosody::Stress;
use crate::segment::{Environment, SegmentMatcher};
use crate::spec::Spec;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllophoneRule {
    pub id: String,
    pub name: String,
    pub input: PhonemePattern,
    pub environment: Environment,
    #[serde(default)]
    pub conditions: Vec<RuleCondition>,
    pub output: PhonePattern,
    pub confidence: f32,
    pub status: RuleStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpenthesisRule {
    pub id: String,
    pub name: String,
    pub before: Vec<SegmentMatcher>,
    pub after: Vec<SegmentMatcher>,
    pub output: PhonePattern,
    pub confidence: f32,
    pub status: RuleStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonemePattern {
    pub phoneme: Spec<PhonemeId>,
    pub features: FeatureBundle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonePattern {
    pub phone: Spec<PhoneId>,
    pub features: FeatureBundle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleCondition {
    PreviousMatches(SegmentMatcher),
    NextMatches(SegmentMatcher),
    PreviousHasFeature(FeatureId, FeatureValue),
    NextHasFeature(FeatureId, FeatureValue),
    PreviousStress(Stress),
    PreviousStressIn(Vec<Stress>),
    NextStress(Stress),
    NextStressIn(Vec<Stress>),
    NotCarefulStyle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleStatus {
    Productive,
    Lexicalized,
    Optional,
    StyleDependent,
    Experimental,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Phonotactics {
    pub allowed_syllable_shapes: Vec<SyllableShape>,
    pub constraints: Vec<PhonotacticConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyllableShape {
    pub pattern: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhonotacticConstraint {
    pub id: String,
    pub description: String,
    pub matcher: SegmentMatcher,
    pub environment: Environment,
    pub status: RuleStatus,
}
