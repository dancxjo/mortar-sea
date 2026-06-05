use std::collections::HashMap;

use crate::feature::FeatureSystem;
use crate::ids::{LanguageId, PhoneId, PhonemeId, VariantId};
use crate::orthography::Orthography;
use crate::phonetics::{Phone, PhoneInventory};
use crate::phonology::{Phoneme, PhonemeInventory};
use crate::rules::{PhonotacticConstraint, Phonotactics, RuleStatus, SyllableShape};
use crate::segment::{Environment, SegmentMatcher, SegmentStatus, SymbolAlias};
use crate::spec::Spec;
use crate::variant::{LinguisticVariant, VariantImplementationStatus, VariantStatus};

const PHONEMES: &[(&str, &str, &str)] = &[
    ("A", "a", "vowel"),
    ("E", "e", "vowel"),
    ("I", "i", "vowel"),
    ("O", "o", "vowel"),
    ("U", "u", "vowel"),
    ("P", "p", "consonant"),
    ("L", "l", "consonant"),
    ("R", "r", "consonant"),
    ("S", "s", "consonant"),
    ("N", "n", "consonant"),
    ("M", "m", "consonant"),
    ("T", "t", "consonant"),
    ("K", "k", "consonant"),
];

pub fn variant() -> LinguisticVariant {
    let mut phonemes = HashMap::new();
    let mut phones = HashMap::new();
    for (symbol, ipa, _major) in PHONEMES {
        let phone_id = PhoneId(format!("ipa.phone.{ipa}"));
        phones.insert(
            phone_id.clone(),
            Phone {
                id: phone_id.clone(),
                ipa: (*ipa).into(),
                features: Default::default(),
                aliases: vec![SymbolAlias {
                    system: "esperanto".into(),
                    symbol: (*symbol).into(),
                }],
                status: SegmentStatus::Core,
            },
        );
        let phoneme = Phoneme {
            id: PhonemeId(format!("eo.phoneme.{symbol}")),
            notation: format!("/{ipa}/"),
            features: Default::default(),
            default_phone: Some(phone_id.clone()),
            possible_phones: vec![phone_id],
            status: SegmentStatus::Core,
        };
        phonemes.insert(phoneme.id.clone(), phoneme);
    }

    LinguisticVariant {
        id: VariantId("eo".into()),
        language: LanguageId("eo".into()),
        name: "Esperanto (sample)".into(),
        feature_system: FeatureSystem::default(),
        phonemes: PhonemeInventory { phonemes },
        phones: PhoneInventory { phones },
        allophone_rules: Vec::new(),
        phonotactics: Some(Phonotactics {
            allowed_syllable_shapes: vec![
                SyllableShape {
                    pattern: "V".into(),
                },
                SyllableShape {
                    pattern: "CV".into(),
                },
                SyllableShape {
                    pattern: "CVC".into(),
                },
            ],
            constraints: vec![
                cluster_constraint(&["p", "l"]),
                cluster_constraint(&["p", "r"]),
            ],
        }),
        orthography: Some(Orthography {
            name: "Esperanto Latin orthography".into(),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VariantStatus::Attested,
        implementation_status: VariantImplementationStatus::Complete,
    }
}

fn cluster_constraint(cluster: &[&str]) -> PhonotacticConstraint {
    PhonotacticConstraint {
        id: format!("eo.legal_onset.{}", cluster.join("_")),
        description: format!("Legal Esperanto onset cluster {}", cluster.join("")),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster
                .iter()
                .map(|ipa| SegmentMatcher::Phone(PhoneId(format!("ipa.phone.{ipa}"))))
                .collect(),
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn esperanto_sample_loads_expected_phonemes() {
        let eo = variant();
        assert!(
            eo.phonemes
                .phonemes
                .contains_key(&PhonemeId("eo.phoneme.A".into()))
        );
        assert!(
            eo.phonemes
                .phonemes
                .contains_key(&PhonemeId("eo.phoneme.K".into()))
        );
    }
}
