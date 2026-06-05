use std::collections::HashMap;

use crate::data::arpabet::{self, ARPABET};
use crate::feature::FeatureSystem;
use crate::ids::{LanguageId, PhoneId, VariantId};
use crate::orthography::Orthography;
use crate::phonetics::PhoneInventory;
use crate::phonology::PhonemeInventory;
use crate::rules::{
    AllophoneRule, PhonePattern, PhonemePattern, PhonotacticConstraint, Phonotactics, RuleStatus,
    SyllableShape,
};
use crate::segment::{Environment, SegmentMatcher};
use crate::spec::Spec;
use crate::variant::{LinguisticVariant, VariantImplementationStatus, VariantStatus};

#[derive(Debug, Clone, Copy)]
struct EnglishVariantRow {
    id: &'static str,
    name: &'static str,
    implementation_status: ImplementationStatusSpec,
    singing: bool,
}

#[derive(Debug, Clone, Copy)]
enum ImplementationStatusSpec {
    Complete,
    StubDerivedFrom(&'static str),
    PermissiveProfile,
}

const VARIANTS: &[EnglishVariantRow] = &[
    EnglishVariantRow {
        id: "en-US-GA",
        name: "General American English",
        implementation_status: ImplementationStatusSpec::Complete,
        singing: false,
    },
    EnglishVariantRow {
        id: "en-US-singing",
        name: "Permissive Singing Profile",
        implementation_status: ImplementationStatusSpec::PermissiveProfile,
        singing: true,
    },
    EnglishVariantRow {
        id: "en-GB-RP",
        name: "Received Pronunciation (stub)",
        implementation_status: ImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        singing: false,
    },
    EnglishVariantRow {
        id: "en-GB-ScotE",
        name: "Scottish English (stub)",
        implementation_status: ImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        singing: false,
    },
    EnglishVariantRow {
        id: "en-US-AAE",
        name: "African American English (stub)",
        implementation_status: ImplementationStatusSpec::StubDerivedFrom("en-US-GA"),
        singing: false,
    },
];

const LEGAL_ONSETS: &[&[&str]] = &[
    &["p", "l"],
    &["b", "l"],
    &["k", "l"],
    &["ɡ", "l"],
    &["f", "l"],
    &["p", "ɹ"],
    &["b", "ɹ"],
    &["t", "ɹ"],
    &["d", "ɹ"],
    &["k", "ɹ"],
    &["ɡ", "ɹ"],
    &["f", "ɹ"],
    &["θ", "ɹ"],
    &["ʃ", "ɹ"],
    &["s", "p"],
    &["s", "t"],
    &["s", "k"],
    &["s", "l"],
    &["s", "m"],
    &["s", "n"],
    &["s", "w"],
    &["s", "f"],
    &["t", "w"],
    &["k", "w"],
    &["ɡ", "w"],
    &["d", "w"],
    &["ʃ", "w"],
    &["θ", "w"],
    &["s", "p", "l"],
    &["s", "p", "ɹ"],
    &["s", "t", "ɹ"],
    &["s", "k", "ɹ"],
    &["s", "k", "w"],
    &["s", "t", "w"],
];

const SINGING_ONSET_ADDITIONS: &[&[&str]] = &[
    &["t", "l"],
    &["d", "l"],
    &["v", "ɹ"],
    &["v", "l"],
    &["z", "w"],
];

const LEGAL_CODAS: &[&[&str]] = &[
    &["n", "d"],
    &["n", "t"],
    &["n", "z"],
    &["ŋ", "k"],
    &["ŋ", "z"],
    &["m", "p"],
    &["m", "z"],
    &["l", "d"],
    &["l", "t"],
    &["l", "k"],
    &["l", "p"],
    &["l", "f"],
    &["l", "m"],
    &["l", "n"],
    &["l", "z"],
    &["s", "t"],
    &["s", "k"],
    &["s", "p"],
    &["f", "t"],
    &["k", "t"],
    &["k", "s"],
    &["p", "t"],
    &["p", "s"],
    &["t", "s"],
    &["d", "z"],
    &["ɹ", "d"],
    &["ɹ", "t"],
    &["ɹ", "k"],
    &["ɹ", "n"],
    &["ɹ", "m"],
    &["ɹ", "z"],
    &["ɹ", "p"],
    &["ɹ", "f"],
    &["n", "tʃ"],
    &["n", "dʒ"],
    &["l", "tʃ"],
    &["ɹ", "tʃ"],
    &["n", "d", "z"],
    &["n", "t", "s"],
    &["ŋ", "k", "s"],
    &["l", "d", "z"],
    &["l", "t", "s"],
    &["l", "k", "s"],
    &["m", "p", "t"],
    &["m", "p", "s"],
    &["s", "t", "s"],
    &["k", "t", "s"],
    &["ŋ", "θ", "s"],
    &["ŋ", "k", "θ", "s"],
];

pub fn variant(id: &str) -> LinguisticVariant {
    let row = VARIANTS
        .iter()
        .find(|row| row.id == id)
        .unwrap_or(&VARIANTS[0]);
    LinguisticVariant {
        id: VariantId(row.id.into()),
        language: LanguageId("en".into()),
        name: row.name.into(),
        feature_system: FeatureSystem::default(),
        phonemes: phoneme_inventory(row.id),
        phones: phone_inventory(),
        allophone_rules: allophone_rules(row.id),
        phonotactics: Some(phonotactics(row.singing)),
        orthography: Some(Orthography {
            name: "English Latin orthography".into(),
            ..Default::default()
        }),
        morphology: None,
        acoustic_profile: None,
        prosody_profile: None,
        status: VariantStatus::Attested,
        implementation_status: match row.implementation_status {
            ImplementationStatusSpec::Complete => VariantImplementationStatus::Complete,
            ImplementationStatusSpec::StubDerivedFrom(source) => {
                VariantImplementationStatus::StubDerivedFrom(VariantId(source.into()))
            }
            ImplementationStatusSpec::PermissiveProfile => {
                VariantImplementationStatus::PermissiveProfile
            }
        },
    }
}

fn phoneme_inventory(variant_id: &str) -> PhonemeInventory {
    PhonemeInventory {
        phonemes: ARPABET
            .iter()
            .map(|entry| {
                let phoneme = arpabet::phoneme_for_entry(variant_id, entry);
                (phoneme.id.clone(), phoneme)
            })
            .collect(),
    }
}

fn phone_inventory() -> PhoneInventory {
    let mut phones = HashMap::new();
    for entry in ARPABET {
        let phone = arpabet::phone_for_entry(entry);
        phones.insert(phone.id.clone(), phone);
    }
    for ipa in ["ɾ", "|"] {
        let phone = crate::phonetics::Phone {
            id: arpabet::phone_id_for_ipa(ipa),
            ipa: ipa.into(),
            features: Default::default(),
            aliases: Vec::new(),
            status: crate::segment::SegmentStatus::Allophonic,
        };
        phones.insert(phone.id.clone(), phone);
    }
    PhoneInventory { phones }
}

fn allophone_rules(variant_id: &str) -> Vec<AllophoneRule> {
    vec![
        AllophoneRule {
            id: "american_english_intervocalic_flapping".into(),
            name: "American English intervocalic flapping".into(),
            input: PhonemePattern {
                phoneme: Spec::Known(arpabet::phoneme_id(variant_id, "T")),
                features: Default::default(),
            },
            environment: Environment {
                before: vec![SegmentMatcher::FeatureBundle(Default::default())],
                after: vec![SegmentMatcher::FeatureBundle(Default::default())],
                word_position: Spec::Known(crate::segment::WordPosition::Medial),
                stress_context: Spec::Known(crate::prosody::Stress::Unstressed),
                prosodic_context: Spec::Known(crate::prosody::ProsodicContext::CarefulSpeech),
                ..Default::default()
            },
            output: PhonePattern {
                phone: Spec::Known(PhoneId("ipa.phone.ɾ".into())),
                features: Default::default(),
            },
            confidence: 0.95,
            status: RuleStatus::StyleDependent,
        },
        AllophoneRule {
            id: "alveolar_nasal_velar_assimilation".into(),
            name: "Alveolar nasal velar assimilation".into(),
            input: PhonemePattern {
                phoneme: Spec::Known(arpabet::phoneme_id(variant_id, "N")),
                features: Default::default(),
            },
            environment: Environment {
                after: vec![
                    SegmentMatcher::Phoneme(arpabet::phoneme_id(variant_id, "K")),
                    SegmentMatcher::Phoneme(arpabet::phoneme_id(variant_id, "G")),
                ],
                ..Default::default()
            },
            output: PhonePattern {
                phone: Spec::Known(PhoneId("ipa.phone.ŋ".into())),
                features: Default::default(),
            },
            confidence: 0.98,
            status: RuleStatus::Productive,
        },
    ]
}

fn phonotactics(singing: bool) -> Phonotactics {
    let mut constraints = Vec::new();
    constraints.push(PhonotacticConstraint {
        id: "english.illegal_onset.ng".into(),
        description: "Velar nasal is not a legal singleton onset in English".into(),
        matcher: SegmentMatcher::Phone(PhoneId("ipa.phone.ŋ".into())),
        environment: Environment {
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    });

    for cluster in LEGAL_ONSETS {
        constraints.push(cluster_constraint(
            "english.legal_onset",
            cluster,
            RuleStatus::Productive,
        ));
    }
    if singing {
        for cluster in SINGING_ONSET_ADDITIONS {
            constraints.push(cluster_constraint(
                "english.singing_legal_onset",
                cluster,
                RuleStatus::Experimental,
            ));
        }
    }
    for cluster in LEGAL_CODAS {
        constraints.push(cluster_constraint(
            "english.legal_coda",
            cluster,
            RuleStatus::Productive,
        ));
    }

    Phonotactics {
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
            SyllableShape {
                pattern: "CCVC".into(),
            },
            SyllableShape {
                pattern: "CVCC".into(),
            },
        ],
        constraints,
    }
}

fn cluster_constraint(prefix: &str, cluster: &[&str], status: RuleStatus) -> PhonotacticConstraint {
    PhonotacticConstraint {
        id: format!("{prefix}.{}", cluster.join("_")),
        description: format!("Legal cluster {}", cluster.join("")),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster
                .iter()
                .map(|ipa| SegmentMatcher::Phone(arpabet::phone_id_for_ipa(ipa)))
                .collect(),
            syllable_position: if prefix.contains("coda") {
                Spec::Known(crate::segment::SyllablePosition::Coda)
            } else {
                Spec::Known(crate::segment::SyllablePosition::Onset)
            },
            prosodic_context: if prefix.contains("singing") {
                Spec::Known(crate::prosody::ProsodicContext::Emphasized)
            } else {
                Spec::Unspecified
            },
            ..Default::default()
        },
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_cluster(variant: &LinguisticVariant, needle: &str) -> bool {
        variant
            .phonotactics
            .as_ref()
            .unwrap()
            .constraints
            .iter()
            .any(|constraint| constraint.id.ends_with(needle))
    }

    #[test]
    fn singing_adds_tl_without_changing_ga() {
        assert!(!has_cluster(&variant("en-US-GA"), "t_l"));
        assert!(has_cluster(&variant("en-US-singing"), "t_l"));
    }

    #[test]
    fn ga_inventory_contains_arpabet_phonemes_and_ipa_phones() {
        let ga = variant("en-US-GA");
        assert!(
            ga.phonemes
                .phonemes
                .contains_key(&arpabet::phoneme_id("en-US-GA", "AH"))
        );
        assert!(
            ga.phones
                .phones
                .contains_key(&PhoneId("ipa.phone.ʌ".into()))
        );
    }

    #[test]
    fn rules_are_variant_data() {
        let ga = variant("en-US-GA");
        assert!(
            ga.allophone_rules
                .iter()
                .any(|rule| rule.id == "american_english_intervocalic_flapping"
                    && rule.status == RuleStatus::StyleDependent)
        );
    }
}
