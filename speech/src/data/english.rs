use std::collections::HashMap;

use crate::data::arpabet::{self, ARPABET};
use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{FeatureId, LanguageId, PhoneId, PhonemeId, VariantId};
use crate::orthography::Orthography;
use crate::phonetics::PhoneInventory;
use crate::phonology::PhonemeInventory;
use crate::prosody::{ProsodicContext, Stress};
use crate::rules::{
    AllophoneRule, EpenthesisRule, PhonePattern, PhonemePattern, PhonotacticConstraint,
    Phonotactics, RuleCondition, RuleStatus, SyllableShape,
};
use crate::segment::{Environment, SegmentMatcher, SyllablePosition};
use crate::spec::Spec;
use crate::variant::{
    LinguisticVariant, OrthographicUnitKind, OrthographicUnitPronunciation,
    VariantImplementationStatus, VariantStatus, WeakFormFollowingContext, WeakFormRule,
    WeakFormStyleContext,
};

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

const P: PhoneId = PhoneId::borrowed("ipa.phone.p");
const B: PhoneId = PhoneId::borrowed("ipa.phone.b");
const T: PhoneId = PhoneId::borrowed("ipa.phone.t");
const D: PhoneId = PhoneId::borrowed("ipa.phone.d");
const K: PhoneId = PhoneId::borrowed("ipa.phone.k");
const G: PhoneId = PhoneId::borrowed("ipa.phone.ɡ");
const F: PhoneId = PhoneId::borrowed("ipa.phone.f");
const V: PhoneId = PhoneId::borrowed("ipa.phone.v");
const TH: PhoneId = PhoneId::borrowed("ipa.phone.θ");
const SH: PhoneId = PhoneId::borrowed("ipa.phone.ʃ");
const S: PhoneId = PhoneId::borrowed("ipa.phone.s");
const Z: PhoneId = PhoneId::borrowed("ipa.phone.z");
const M: PhoneId = PhoneId::borrowed("ipa.phone.m");
const N: PhoneId = PhoneId::borrowed("ipa.phone.n");
const NG: PhoneId = PhoneId::borrowed("ipa.phone.ŋ");
const L: PhoneId = PhoneId::borrowed("ipa.phone.l");
const R: PhoneId = PhoneId::borrowed("ipa.phone.ɹ");
const W: PhoneId = PhoneId::borrowed("ipa.phone.w");
const Y: PhoneId = PhoneId::borrowed("ipa.phone.j");
const CH: PhoneId = PhoneId::borrowed("ipa.phone.tʃ");
const JH: PhoneId = PhoneId::borrowed("ipa.phone.dʒ");
const TAP: PhoneId = PhoneId::borrowed("ipa.phone.ɾ");
const SCHWA: PhoneId = PhoneId::borrowed("ipa.phone.ə");
const R_COLORED_SCHWA: PhoneId = PhoneId::borrowed("ipa.phone.ɚ");
const SYLLABLE_BREAK: PhoneId = PhoneId::borrowed("ipa.phone.|");

const LEGAL_ONSETS: &[&[PhoneId]] = &[
    &[P, L],
    &[B, L],
    &[K, L],
    &[G, L],
    &[F, L],
    &[P, R],
    &[B, R],
    &[T, R],
    &[D, R],
    &[K, R],
    &[G, R],
    &[F, R],
    &[TH, R],
    &[SH, R],
    &[S, P],
    &[S, T],
    &[S, K],
    &[S, L],
    &[S, M],
    &[S, N],
    &[S, W],
    &[S, F],
    &[T, W],
    &[K, W],
    &[G, W],
    &[D, W],
    &[SH, W],
    &[TH, W],
    &[S, P, L],
    &[S, P, R],
    &[S, T, R],
    &[S, K, R],
    &[S, K, W],
    &[S, T, W],
];

const SINGING_ONSET_ADDITIONS: &[&[PhoneId]] = &[&[T, L], &[D, L], &[V, R], &[V, L], &[Z, W]];

const LEGAL_CODAS: &[&[PhoneId]] = &[
    &[N, D],
    &[N, T],
    &[N, Z],
    &[NG, K],
    &[NG, Z],
    &[M, P],
    &[M, Z],
    &[L, D],
    &[L, T],
    &[L, K],
    &[L, P],
    &[L, F],
    &[L, M],
    &[L, N],
    &[L, Z],
    &[S, T],
    &[S, K],
    &[S, P],
    &[F, T],
    &[K, T],
    &[K, S],
    &[P, T],
    &[P, S],
    &[T, S],
    &[D, Z],
    &[R, D],
    &[R, T],
    &[R, K],
    &[R, N],
    &[R, M],
    &[R, Z],
    &[R, P],
    &[R, F],
    &[N, CH],
    &[N, JH],
    &[L, CH],
    &[R, CH],
    &[N, D, Z],
    &[N, T, S],
    &[NG, K, S],
    &[L, D, Z],
    &[L, T, S],
    &[L, K, S],
    &[M, P, T],
    &[M, P, S],
    &[S, T, S],
    &[K, T, S],
    &[NG, TH, S],
    &[NG, K, TH, S],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClusterScope {
    Onset,
    SingingOnset,
    Coda,
}

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
        epenthesis_rules: epenthesis_rules(),
        weak_forms: weak_forms(row.id),
        orthographic_unit_pronunciations: orthographic_unit_pronunciations(row.id),
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

fn orthographic_unit_pronunciations(variant_id: &str) -> Vec<OrthographicUnitPronunciation> {
    let letters: &[(char, &[&str])] = &[
        ('A', &["EY1"]),
        ('B', &["B", "IY1"]),
        ('C', &["S", "IY1"]),
        ('D', &["D", "IY1"]),
        ('E', &["IY1"]),
        ('F', &["EH1", "F"]),
        ('G', &["JH", "IY1"]),
        ('H', &["EY1", "CH"]),
        ('I', &["AY1"]),
        ('J', &["JH", "EY1"]),
        ('K', &["K", "EY1"]),
        ('L', &["EH1", "L"]),
        ('M', &["EH1", "M"]),
        ('N', &["EH1", "N"]),
        ('O', &["OW1"]),
        ('P', &["P", "IY1"]),
        ('Q', &["K", "Y", "UW1"]),
        ('R', &["AA1", "R"]),
        ('S', &["EH1", "S"]),
        ('T', &["T", "IY1"]),
        ('U', &["Y", "UW1"]),
        ('V', &["V", "IY1"]),
        ('W', &["D", "AH1", "B", "AH0", "L", "Y", "UW0"]),
        ('X', &["EH1", "K", "S"]),
        ('Y', &["W", "AY1"]),
        ('Z', &["Z", "IY1"]),
    ];
    let digits: &[(char, &[&str])] = &[
        ('0', &["Z", "IH1", "R", "OW0"]),
        ('1', &["W", "AH1", "N"]),
        ('2', &["T", "UW1"]),
        ('3', &["TH", "R", "IY1"]),
        ('4', &["F", "AO1", "R"]),
        ('5', &["F", "AY1", "V"]),
        ('6', &["S", "IH1", "K", "S"]),
        ('7', &["S", "EH1", "V", "AH0", "N"]),
        ('8', &["EY1", "T"]),
        ('9', &["N", "AY1", "N"]),
    ];

    letters
        .iter()
        .map(|(letter, symbols)| {
            orthographic_unit(
                variant_id,
                OrthographicUnitKind::LetterName,
                *letter,
                symbols,
            )
        })
        .chain(digits.iter().map(|(digit, symbols)| {
            orthographic_unit(variant_id, OrthographicUnitKind::DigitName, *digit, symbols)
        }))
        .collect()
}

fn orthographic_unit(
    variant_id: &str,
    kind: OrthographicUnitKind,
    unit: char,
    symbols: &[&str],
) -> OrthographicUnitPronunciation {
    OrthographicUnitPronunciation {
        kind,
        unit: unit.to_string(),
        pronunciation: symbols
            .iter()
            .map(|symbol| arpabet::phoneme_id(variant_id, symbol))
            .collect(),
    }
}

fn weak_forms(variant_id: &str) -> Vec<WeakFormRule> {
    [
        weak_form(
            "english_weak_the_before_vowel",
            "the",
            &["DH", "IY0"],
            WeakFormFollowingContext::BeforeVowelish,
            WeakFormStyleContext::Any,
            variant_id,
        ),
        weak_form(
            "english_weak_the_before_consonant",
            "the",
            &["DH", "AH0"],
            WeakFormFollowingContext::BeforeConsonantish,
            WeakFormStyleContext::Any,
            variant_id,
        ),
        weak_form(
            "english_weak_and",
            "and",
            &["AH0", "N", "D"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variant_id,
        ),
        weak_form(
            "english_weak_a",
            "a",
            &["AH0"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variant_id,
        ),
        weak_form(
            "english_weak_an",
            "an",
            &["AH0", "N"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variant_id,
        ),
        weak_form(
            "english_weak_of",
            "of",
            &["AH0", "V"],
            WeakFormFollowingContext::Any,
            WeakFormStyleContext::Any,
            variant_id,
        ),
        weak_form(
            "english_weak_to_before_consonant",
            "to",
            &["T", "AH0"],
            WeakFormFollowingContext::BeforeConsonantish,
            WeakFormStyleContext::CasualOnly,
            variant_id,
        ),
    ]
    .into()
}

fn weak_form(
    id: &str,
    lexical_item: &str,
    symbols: &[&str],
    following: WeakFormFollowingContext,
    style: WeakFormStyleContext,
    variant_id: &str,
) -> WeakFormRule {
    WeakFormRule {
        id: id.into(),
        lexical_item: lexical_item.into(),
        pronunciation: symbols
            .iter()
            .map(|symbol| PhonemeId(format!("{variant_id}.phoneme.{symbol}")))
            .collect(),
        following,
        style,
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
    for (phone_ref, base, ipa) in [(SCHWA, "AH", "ə"), (R_COLORED_SCHWA, "ER", "ɚ")] {
        let mut features = arpabet::entry(base)
            .map(arpabet::feature_bundle)
            .unwrap_or_default();
        features.values.insert(
            FeatureId("phonology.reduced_vowel".into()),
            Spec::Known(FeatureValue::Bool(true)),
        );
        let phone = crate::phonetics::Phone {
            id: phone_ref,
            ipa: ipa.into(),
            features,
            aliases: Vec::new(),
            status: crate::segment::SegmentStatus::Allophonic,
        };
        phones.insert(phone.id.clone(), phone);
    }
    for phone_ref in [TAP, SYLLABLE_BREAK] {
        let ipa = phone_symbol(&phone_ref).into();
        let phone = crate::phonetics::Phone {
            id: phone_ref,
            ipa,
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
                before: vec![SegmentMatcher::FeatureBundle(feature_bundle(&[(
                    "major", "vowel",
                )]))],
                after: vec![SegmentMatcher::FeatureBundle(feature_bundle(&[(
                    "major", "vowel",
                )]))],
                ..Default::default()
            },
            conditions: vec![
                RuleCondition::PreviousMatches(SegmentMatcher::FeatureBundle(feature_bundle(&[(
                    "major", "vowel",
                )]))),
                RuleCondition::PreviousStressIn(vec![Stress::Primary, Stress::Secondary]),
                RuleCondition::NextMatches(SegmentMatcher::FeatureBundle(feature_bundle(&[(
                    "major", "vowel",
                )]))),
                RuleCondition::NextStress(Stress::Unstressed),
                RuleCondition::NotCarefulStyle,
            ],
            output: PhonePattern {
                phone: Spec::Known(TAP),
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
                after: vec![SegmentMatcher::FeatureBundle(feature_bundle(&[
                    ("place", "velar"),
                    ("manner", "stop"),
                ]))],
                ..Default::default()
            },
            conditions: vec![RuleCondition::NextMatches(SegmentMatcher::FeatureBundle(
                feature_bundle(&[("place", "velar"), ("manner", "stop")]),
            ))],
            output: PhonePattern {
                phone: Spec::Known(NG),
                features: Default::default(),
            },
            confidence: 0.98,
            status: RuleStatus::Productive,
        },
    ]
}

fn epenthesis_rules() -> Vec<EpenthesisRule> {
    vec![EpenthesisRule {
        id: "english_letter_name_front_vowel_linking_yod".into(),
        name: "English letter-name front-vowel linking yod".into(),
        before: vec![SegmentMatcher::FeatureBundle(feature_bundle_with_values(
            &[
                ("phonology.major", FeatureValue::Category("vowel".into())),
                (
                    "phonology.vowel_backness",
                    FeatureValue::Category("front".into()),
                ),
                ("orthography.letter_name", FeatureValue::Bool(true)),
            ],
        ))],
        after: vec![SegmentMatcher::FeatureBundle(feature_bundle_with_values(
            &[
                ("phonology.major", FeatureValue::Category("vowel".into())),
                ("orthography.letter_name", FeatureValue::Bool(true)),
            ],
        ))],
        output: PhonePattern {
            phone: Spec::Known(Y),
            features: Default::default(),
        },
        confidence: 0.85,
        status: RuleStatus::Productive,
    }]
}

fn feature_bundle(values: &[(&str, &str)]) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    for (name, value) in values {
        bundle.values.insert(
            FeatureId(format!("phonology.{name}")),
            Spec::Known(FeatureValue::Category((*value).into())),
        );
    }
    bundle
}

fn feature_bundle_with_values(values: &[(&str, FeatureValue)]) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    for (id, value) in values {
        bundle
            .values
            .insert(FeatureId((*id).into()), Spec::Known(value.clone()));
    }
    bundle
}

fn phonotactics(singing: bool) -> Phonotactics {
    let mut constraints = Vec::new();
    constraints.push(PhonotacticConstraint {
        id: "english.illegal_onset.ng".into(),
        description: "Velar nasal is not a legal singleton onset in English".into(),
        matcher: SegmentMatcher::Phone(NG),
        environment: Environment {
            syllable_position: Spec::Known(crate::segment::SyllablePosition::Onset),
            ..Default::default()
        },
        status: RuleStatus::Productive,
    });

    for cluster in LEGAL_ONSETS {
        constraints.push(cluster_constraint(
            ClusterScope::Onset,
            cluster,
            RuleStatus::Productive,
        ));
    }
    if singing {
        for cluster in SINGING_ONSET_ADDITIONS {
            constraints.push(cluster_constraint(
                ClusterScope::SingingOnset,
                cluster,
                RuleStatus::Experimental,
            ));
        }
    }
    for cluster in LEGAL_CODAS {
        constraints.push(cluster_constraint(
            ClusterScope::Coda,
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

fn cluster_constraint(
    scope: ClusterScope,
    cluster: &[PhoneId],
    status: RuleStatus,
) -> PhonotacticConstraint {
    let suffix = cluster_suffix(cluster);
    let label = cluster_label(cluster);
    PhonotacticConstraint {
        id: format!("{}.{}", scope.constraint_prefix(), suffix),
        description: format!("Legal {} cluster {}", scope.label(), label),
        matcher: SegmentMatcher::Any,
        environment: Environment {
            before: cluster.iter().cloned().map(SegmentMatcher::Phone).collect(),
            syllable_position: Spec::Known(scope.syllable_position()),
            prosodic_context: scope.prosodic_context(),
            ..Default::default()
        },
        status,
    }
}

impl ClusterScope {
    fn constraint_prefix(self) -> &'static str {
        match self {
            Self::Onset => "english.legal_onset",
            Self::SingingOnset => "english.singing_legal_onset",
            Self::Coda => "english.legal_coda",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Onset | Self::SingingOnset => "onset",
            Self::Coda => "coda",
        }
    }

    fn syllable_position(self) -> SyllablePosition {
        match self {
            Self::Onset | Self::SingingOnset => SyllablePosition::Onset,
            Self::Coda => SyllablePosition::Coda,
        }
    }

    fn prosodic_context(self) -> Spec<ProsodicContext> {
        match self {
            Self::SingingOnset => Spec::Known(ProsodicContext::Emphasized),
            Self::Onset | Self::Coda => Spec::Unspecified,
        }
    }
}

fn cluster_suffix(cluster: &[PhoneId]) -> String {
    cluster
        .iter()
        .map(phone_symbol)
        .collect::<Vec<_>>()
        .join("_")
}

fn cluster_label(cluster: &[PhoneId]) -> String {
    cluster
        .iter()
        .map(phone_symbol)
        .collect::<Vec<_>>()
        .join("")
}

fn phone_symbol(phone: &PhoneId) -> &str {
    phone
        .as_str()
        .strip_prefix("ipa.phone.")
        .unwrap_or(phone.as_str())
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
        assert!(ga.phones.phones.contains_key(&PhoneId::from("ipa.phone.ʌ")));
    }

    #[test]
    fn rules_are_variant_data() {
        let ga = variant("en-US-GA");
        let flapping = ga
            .allophone_rules
            .iter()
            .find(|rule| rule.id == "american_english_intervocalic_flapping")
            .expect("flapping rule");

        assert_eq!(flapping.status, RuleStatus::StyleDependent);
        assert!(
            flapping
                .conditions
                .contains(&RuleCondition::NotCarefulStyle)
        );
        assert_eq!(flapping.environment.word_position, Spec::Unspecified);
        assert_eq!(flapping.environment.prosodic_context, Spec::Unspecified);
    }

    #[test]
    fn weak_forms_are_variant_data() {
        let ga = variant("en-US-GA");
        let weak_the = ga
            .weak_forms
            .iter()
            .find(|rule| rule.id == "english_weak_the_before_consonant")
            .expect("weak form for the before consonants");

        assert_eq!(weak_the.lexical_item, "the");
        assert_eq!(
            weak_the.pronunciation,
            vec![
                arpabet::phoneme_id("en-US-GA", "DH"),
                arpabet::phoneme_id("en-US-GA", "AH0")
            ]
        );
        assert_eq!(
            weak_the.following,
            WeakFormFollowingContext::BeforeConsonantish
        );
    }
}
