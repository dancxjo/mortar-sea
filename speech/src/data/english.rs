use std::collections::HashMap;

use crate::acoustics::{
    AcousticCueDef, AcousticLandmark, AcousticLandmarkKind, AcousticProfile, AcousticTargetModel,
    CueTarget, LandmarkAnchor, RelativeTimeWindow, WeightedCue,
};
use crate::data::arpabet::{self, ARPABET};
use crate::data::cmudict::CmuPhoneme;
use crate::feature::{FeatureBundle, FeatureSystem, FeatureValue};
use crate::ids::{AcousticCueId, FeatureId, LanguageId, PhoneId, VariantId};
use crate::orthography::Orthography;
use crate::phonetics::PhoneInventory;
use crate::phonology::{PhonemeAllophone, PhonemeInventory};
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
        acoustic_profile: Some(acoustic_profile(row.id)),
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
        cmudict_pronunciation: symbols
            .iter()
            .map(|symbol| CmuPhoneme::parse(symbol))
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
            .map(|symbol| arpabet::phoneme_id(variant_id, symbol))
            .collect(),
        cmudict_pronunciation: symbols
            .iter()
            .map(|symbol| CmuPhoneme::parse(symbol))
            .collect(),
        following,
        style,
    }
}

fn phoneme_inventory(variant_id: &str) -> PhonemeInventory {
    let mut phonemes = ARPABET
        .iter()
        .map(|entry| {
            let phoneme = arpabet::phoneme_for_entry(variant_id, entry);
            (phoneme.id.clone(), phoneme)
        })
        .collect::<HashMap<_, _>>();
    for rule in allophone_rules(variant_id) {
        let Spec::Known(phoneme_id) = &rule.input.phoneme else {
            continue;
        };
        let Spec::Known(phone_id) = &rule.output.phone else {
            continue;
        };
        if let Some(phoneme) = phonemes.get_mut(phoneme_id) {
            if !phoneme.possible_phones.contains(phone_id) {
                phoneme.possible_phones.push(phone_id.clone());
            }
            phoneme.allophones.push(PhonemeAllophone {
                phone: phone_id.clone(),
                environment: rule.environment.clone(),
                conditions: rule.conditions.clone(),
                confidence: rule.confidence,
                status: rule.status.clone(),
                source_rule_id: Some(rule.id.clone()),
            });
        }
    }
    PhonemeInventory { phonemes }
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

fn acoustic_profile(variant_id: &str) -> AcousticProfile {
    let mut cues = HashMap::new();
    for cue in acoustic_cues() {
        cues.insert(cue.id.clone(), cue);
    }

    let mut phone_models = HashMap::new();
    let mut phoneme_models = HashMap::new();
    for entry in ARPABET.iter().filter(|entry| entry.syllabic) {
        let model = vowel_model(entry);
        phone_models.insert(arpabet::phone_id_for_ipa(entry.phone_symbol), model.clone());
        phoneme_models.insert(arpabet::phoneme_id(variant_id, entry.symbol), model);
    }
    phone_models.insert(SCHWA, reduced_central_vowel_model("schwa"));
    phone_models.insert(
        R_COLORED_SCHWA,
        reduced_central_vowel_model("r-colored schwa"),
    );

    let voiceless_bilabial_stop = voiceless_bilabial_stop_model();
    let voiced_bilabial_stop = voiced_bilabial_stop_model();

    phone_models.insert(P, voiceless_bilabial_stop.clone());
    phone_models.insert(B, voiced_bilabial_stop.clone());
    phoneme_models.insert(
        arpabet::phoneme_id(variant_id, "P"),
        voiceless_bilabial_stop,
    );
    phoneme_models.insert(arpabet::phoneme_id(variant_id, "B"), voiced_bilabial_stop);

    AcousticProfile {
        cues,
        phone_models,
        phoneme_models,
    }
}

fn acoustic_cues() -> Vec<AcousticCueDef> {
    vec![
        cue(
            "acoustic.cue.f1_region",
            "first formant region",
            "acoustic.f1_region",
            vec![CueTarget::Feature(FeatureId(
                "phonology.vowel_height".into(),
            ))],
            Some("Low F1 is a common correlate of high vowels.".into()),
        ),
        cue(
            "acoustic.cue.f2_region",
            "second formant region",
            "acoustic.f2_region",
            vec![CueTarget::Feature(FeatureId(
                "phonology.vowel_backness".into(),
            ))],
            Some("High F2 tends to mark front vowels; low F2 tends to mark back/rounded vowels.".into()),
        ),
        cue(
            "acoustic.cue.rounding_resonance",
            "rounding resonance",
            "acoustic.rounding_resonance",
            vec![CueTarget::Feature(FeatureId("phonology.roundedness".into()))],
            Some("Lip rounding usually lowers upper formant energy and reinforces the [u] vs [i] split.".into()),
        ),
        cue(
            "acoustic.cue.periodic_voicing",
            "periodic voicing",
            "acoustic.periodic_voicing",
            vec![
                CueTarget::Feature(FeatureId("phonology.voicing".into())),
                CueTarget::Feature(FeatureId("phonology.syllabic".into())),
            ],
            Some("Regular low-frequency periodicity is a strong cue for voiced sonorants and vowels.".into()),
        ),
        cue(
            "acoustic.cue.sonority_peak",
            "sonority peak",
            "acoustic.sonority_peak",
            vec![
                CueTarget::Feature(FeatureId("phonology.syllabic".into())),
                CueTarget::Stress,
            ],
            Some("Syllable nuclei tend to carry local sonority, energy, and periodicity maxima.".into()),
        ),
        cue(
            "acoustic.cue.vowel_nucleus",
            "vowel nucleus",
            "acoustic.vowel_nucleus",
            vec![CueTarget::Feature(FeatureId("phonology.syllabic".into()))],
            Some("A vowel nucleus is an alignment anchor: stable voicing plus vowel-like formants near the syllable peak.".into()),
        ),
        cue(
            "acoustic.cue.stop_closure",
            "stop closure",
            "acoustic.stop_closure",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("A low-energy closure interval is a core stop landmark.".into()),
        ),
        cue(
            "acoustic.cue.release_burst",
            "release burst",
            "acoustic.release_burst",
            vec![CueTarget::Feature(FeatureId("phonology.manner".into()))],
            Some("Transient burst energy near release helps distinguish stops from continuants.".into()),
        ),
        cue(
            "acoustic.cue.voice_onset_time",
            "voice onset time",
            "acoustic.vot_class",
            vec![CueTarget::Phone(P), CueTarget::Phone(B)],
            Some("VOT separates many English voiced and voiceless stops, but varies with context.".into()),
        ),
        cue(
            "acoustic.cue.closure_voicing",
            "closure voicing",
            "acoustic.voicing_during_closure",
            vec![CueTarget::Phone(B), CueTarget::Phone(P)],
            Some("Periodic low-frequency energy during closure is evidence for a voiced stop.".into()),
        ),
        cue(
            "acoustic.cue.aspiration_noise",
            "aspiration noise",
            "acoustic.aspiration_present",
            vec![CueTarget::Phone(P)],
            Some("Post-release aperiodic breath noise is expected for many English voiceless stops in stressed onsets, but not everywhere.".into()),
        ),
    ]
}

fn vowel_model(entry: &arpabet::ArpabetEntry) -> AcousticTargetModel {
    AcousticTargetModel {
        expected_features: acoustic_feature_bundle(&[
            (
                "f1_region",
                Spec::Known(FeatureValue::Category(f1_region(entry.vowel_height).into())),
            ),
            (
                "f2_region",
                Spec::Known(FeatureValue::Category(
                    f2_region(entry.vowel_backness).into(),
                )),
            ),
            ("rounding_resonance", rounding_resonance(entry.roundedness)),
            ("periodic_voicing", Spec::Known(FeatureValue::Bool(true))),
            ("sonority_peak", Spec::Known(FeatureValue::Bool(true))),
            ("vowel_nucleus", Spec::Known(FeatureValue::Bool(true))),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.f1_region", 0.8),
            ("acoustic.cue.f2_region", 1.0),
            ("acoustic.cue.rounding_resonance", 0.5),
            ("acoustic.cue.periodic_voicing", 0.8),
            ("acoustic.cue.sonority_peak", 0.9),
            ("acoustic.cue.vowel_nucleus", 1.0),
        ]),
        landmarks: vec![vowel_target_landmark(), syllable_nucleus_landmark()],
        notes: Some(format!(
            "Vowel nucleus evidence for ARPABET {}: {:?} height, {:?} backness, {:?} rounding.",
            entry.symbol, entry.vowel_height, entry.vowel_backness, entry.roundedness
        )),
    }
}

fn reduced_central_vowel_model(label: &str) -> AcousticTargetModel {
    AcousticTargetModel {
        expected_features: acoustic_feature_bundle(&[
            (
                "f1_region",
                Spec::Known(FeatureValue::Category("mid".into())),
            ),
            (
                "f2_region",
                Spec::Known(FeatureValue::Category("mid".into())),
            ),
            (
                "rounding_resonance",
                Spec::Known(FeatureValue::Category("absent".into())),
            ),
            ("periodic_voicing", Spec::Known(FeatureValue::Bool(true))),
            (
                "sonority_peak",
                Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)]),
            ),
            ("vowel_nucleus", Spec::Known(FeatureValue::Bool(true))),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.f1_region", 0.7),
            ("acoustic.cue.f2_region", 0.8),
            ("acoustic.cue.periodic_voicing", 0.8),
            ("acoustic.cue.sonority_peak", 0.7),
            ("acoustic.cue.vowel_nucleus", 0.9),
        ]),
        landmarks: vec![vowel_target_landmark(), syllable_nucleus_landmark()],
        notes: Some(format!(
            "Reduced central vowel model for {label}; sonority can be weak in unstressed syllables."
        )),
    }
}

fn f1_region(height: Option<&str>) -> &'static str {
    match height {
        Some("high") => "low",
        Some("mid") | Some("rhotic") => "mid",
        Some("low") => "high",
        _ => "mid",
    }
}

fn f2_region(backness: Option<&str>) -> &'static str {
    match backness {
        Some("front") => "high",
        Some("central") => "mid",
        Some("back") => "low",
        _ => "mid",
    }
}

fn rounding_resonance(roundedness: Option<&str>) -> Spec<FeatureValue> {
    match roundedness {
        Some("rounded") => Spec::Known(FeatureValue::Category("present".into())),
        Some("unrounded") => Spec::Known(FeatureValue::Category("absent".into())),
        _ => Spec::Unspecified,
    }
}

fn voiceless_bilabial_stop_model() -> AcousticTargetModel {
    AcousticTargetModel {
        expected_features: acoustic_feature_bundle(&[
            (
                "stop_closure",
                Spec::Known(FeatureValue::Bool(true)),
            ),
            (
                "release_burst",
                Spec::Known(FeatureValue::Bool(true)),
            ),
            (
                "voicing_during_closure",
                Spec::Known(FeatureValue::Bool(false)),
            ),
            (
                "vot_class",
                Spec::Variable(vec![
                    FeatureValue::Category("short_lag".into()),
                    FeatureValue::Category("long_lag".into()),
                ]),
            ),
            (
                "aspiration_present",
                Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)]),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.stop_closure", 1.0),
            ("acoustic.cue.release_burst", 0.9),
            ("acoustic.cue.voice_onset_time", 0.9),
            ("acoustic.cue.aspiration_noise", 0.6),
            ("acoustic.cue.closure_voicing", 0.5),
        ]),
        landmarks: vec![
            closure_landmark(false),
            release_burst_landmark(),
            aspiration_landmark(),
        ],
        notes: Some("English [p] is a voiceless bilabial stop; aspiration is context-sensitive rather than guaranteed.".into()),
    }
}

fn voiced_bilabial_stop_model() -> AcousticTargetModel {
    AcousticTargetModel {
        expected_features: acoustic_feature_bundle(&[
            (
                "stop_closure",
                Spec::Known(FeatureValue::Bool(true)),
            ),
            (
                "release_burst",
                Spec::Known(FeatureValue::Bool(true)),
            ),
            (
                "voicing_during_closure",
                Spec::Variable(vec![FeatureValue::Bool(true), FeatureValue::Bool(false)]),
            ),
            (
                "vot_class",
                Spec::Variable(vec![
                    FeatureValue::Category("prevoiced".into()),
                    FeatureValue::Category("short_lag".into()),
                ]),
            ),
            (
                "aspiration_present",
                Spec::Known(FeatureValue::Bool(false)),
            ),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.stop_closure", 1.0),
            ("acoustic.cue.release_burst", 0.8),
            ("acoustic.cue.closure_voicing", 0.9),
            ("acoustic.cue.voice_onset_time", 0.8),
            ("acoustic.cue.aspiration_noise", 0.3),
        ]),
        landmarks: vec![closure_landmark(true), release_burst_landmark()],
        notes: Some("English [b] is a voiced bilabial stop; closure voicing can be weak or absent in some positions, so VOT stays variable.".into()),
    }
}

fn vowel_target_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "steady_vowel_target".into(),
        kind: AcousticLandmarkKind::VowelTarget,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.04,
            end_s: 0.04,
        },
        expected_features: FeatureBundle::default(),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.f1_region", 0.8),
            ("acoustic.cue.f2_region", 1.0),
        ]),
        notes: Some(
            "Sample formants near the steady middle of the vowel when the segment is long enough."
                .into(),
        ),
    }
}

fn syllable_nucleus_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "syllable_nucleus_peak".into(),
        kind: AcousticLandmarkKind::VowelTarget,
        anchor: LandmarkAnchor::SegmentCenter,
        window: RelativeTimeWindow {
            start_s: -0.06,
            end_s: 0.06,
        },
        expected_features: acoustic_feature_bundle(&[
            ("vowel_nucleus", Spec::Known(FeatureValue::Bool(true))),
            ("periodic_voicing", Spec::Known(FeatureValue::Bool(true))),
        ]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.vowel_nucleus", 1.0),
            ("acoustic.cue.sonority_peak", 0.9),
            ("acoustic.cue.periodic_voicing", 0.8),
        ]),
        notes: Some("Use this as the preferred anchor when fitting syllable timing.".into()),
    }
}

fn closure_landmark(voiced: bool) -> AcousticLandmark {
    AcousticLandmark {
        id: "stop_closure".into(),
        kind: AcousticLandmarkKind::Closure,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.08,
            end_s: 0.0,
        },
        expected_features: acoustic_feature_bundle(&[(
            "voicing_during_closure",
            Spec::Known(FeatureValue::Bool(voiced)),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.stop_closure", 1.0),
            ("acoustic.cue.closure_voicing", 0.8),
        ]),
        notes: None,
    }
}

fn release_burst_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "release_burst".into(),
        kind: AcousticLandmarkKind::ReleaseBurst,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: -0.005,
            end_s: 0.02,
        },
        expected_features: acoustic_feature_bundle(&[(
            "release_burst",
            Spec::Known(FeatureValue::Bool(true)),
        )]),
        weighted_cues: weighted_cues(&[("acoustic.cue.release_burst", 1.0)]),
        notes: Some("Burst timing is a useful alignment point for oral stops.".into()),
    }
}

fn aspiration_landmark() -> AcousticLandmark {
    AcousticLandmark {
        id: "post_release_aspiration".into(),
        kind: AcousticLandmarkKind::Aspiration,
        anchor: LandmarkAnchor::Release,
        window: RelativeTimeWindow {
            start_s: 0.01,
            end_s: 0.09,
        },
        expected_features: acoustic_feature_bundle(&[(
            "aspiration_present",
            Spec::Variable(vec![FeatureValue::Bool(false), FeatureValue::Bool(true)]),
        )]),
        weighted_cues: weighted_cues(&[
            ("acoustic.cue.aspiration_noise", 1.0),
            ("acoustic.cue.voice_onset_time", 0.7),
        ]),
        notes: Some(
            "Search after release; English aspiration is conditioned by stress and position."
                .into(),
        ),
    }
}

fn acoustic_feature_bundle(values: &[(&str, Spec<FeatureValue>)]) -> FeatureBundle {
    let mut bundle = FeatureBundle::default();
    for (name, value) in values {
        bundle
            .values
            .insert(FeatureId(format!("acoustic.{name}")), value.clone());
    }
    bundle
}

fn weighted_cues(values: &[(&str, f32)]) -> Vec<WeightedCue> {
    values
        .iter()
        .map(|(cue, weight)| WeightedCue {
            cue: AcousticCueId((*cue).into()),
            weight: *weight,
        })
        .collect()
}

fn cue(
    id: &str,
    name: &str,
    feature: &str,
    targets: Vec<CueTarget>,
    notes: Option<String>,
) -> AcousticCueDef {
    AcousticCueDef {
        id: AcousticCueId(id.into()),
        name: name.into(),
        feature: FeatureId(feature.into()),
        targets,
        notes,
    }
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
    use crate::ids::PhonemeId;

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
    fn ga_inventory_contains_canonical_phonemes_and_ipa_phones() {
        let ga = variant("en-US-GA");
        assert!(
            ga.phonemes
                .phonemes
                .contains_key(&PhonemeId("en-US-GA.phoneme.ʌ".into()))
        );
        assert!(ga.phones.phones.contains_key(&PhoneId::from("ipa.phone.ʌ")));
        let ah = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "AH"))
            .expect("AH phoneme decoded to canonical id");
        assert_eq!(ah.id, PhonemeId("en-US-GA.phoneme.ʌ".into()));
        assert!(
            ah.aliases
                .iter()
                .any(|alias| alias.system == "arpabet" && alias.symbol == "AH")
        );
    }

    #[test]
    fn acoustic_profile_distinguishes_high_front_and_back_rounded_vowels() {
        let ga = variant("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let high_front = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("iː"))
            .expect("IY phone fingerprint");
        let high_back = profile
            .phone_models
            .get(&arpabet::phone_id_for_ipa("uː"))
            .expect("UW phone fingerprint");

        assert_acoustic_category(high_front, "f2_region", "high");
        assert_acoustic_category(high_front, "rounding_resonance", "absent");
        assert_acoustic_category(high_back, "f2_region", "low");
        assert_acoustic_category(high_back, "rounding_resonance", "present");
        assert!(
            high_front
                .landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::VowelTarget)
        );
    }

    #[test]
    fn acoustic_profile_marks_bilabial_stop_cues_without_overclaiming_aspiration() {
        let ga = variant("en-US-GA");
        let profile = ga.acoustic_profile.as_ref().expect("acoustic profile");
        let p = profile.phone_models.get(&P).expect("p phone fingerprint");
        let b = profile.phone_models.get(&B).expect("b phone fingerprint");

        assert_acoustic_bool(p, "stop_closure", true);
        assert_acoustic_bool(p, "release_burst", true);
        assert_acoustic_bool(b, "stop_closure", true);
        assert!(
            p.landmarks
                .iter()
                .any(|landmark| landmark.kind == AcousticLandmarkKind::Aspiration)
        );
        assert_eq!(
            acoustic_value(p, "aspiration_present"),
            Some(&Spec::Variable(vec![
                FeatureValue::Bool(false),
                FeatureValue::Bool(true)
            ]))
        );
        assert_eq!(
            acoustic_value(b, "aspiration_present"),
            Some(&Spec::Known(FeatureValue::Bool(false)))
        );
        assert!(
            profile
                .phoneme_models
                .contains_key(&arpabet::phoneme_id("en-US-GA", "P"))
        );
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
    fn phonemes_contain_allophones_with_environments() {
        let ga = variant("en-US-GA");
        let t = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "T"))
            .expect("T phoneme");
        let tap = t
            .allophones
            .iter()
            .find(|allophone| allophone.phone == TAP)
            .expect("tap allophone");

        assert!(t.possible_phones.contains(&TAP));
        assert_eq!(
            tap.source_rule_id.as_deref(),
            Some("american_english_intervocalic_flapping")
        );
        assert_eq!(tap.status, RuleStatus::StyleDependent);
        assert_eq!(tap.environment.before.len(), 1);
        assert_eq!(tap.environment.after.len(), 1);
        assert!(tap.conditions.contains(&RuleCondition::NotCarefulStyle));

        let n = ga
            .phonemes
            .phonemes
            .get(&arpabet::phoneme_id("en-US-GA", "N"))
            .expect("N phoneme");
        assert!(
            n.allophones
                .iter()
                .any(|allophone| allophone.phone == NG && allophone.environment.after.len() == 1)
        );
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

    fn assert_acoustic_category(model: &AcousticTargetModel, name: &str, expected: &str) {
        assert_eq!(
            acoustic_value(model, name),
            Some(&Spec::Known(FeatureValue::Category(expected.into())))
        );
    }

    fn assert_acoustic_bool(model: &AcousticTargetModel, name: &str, expected: bool) {
        assert_eq!(
            acoustic_value(model, name),
            Some(&Spec::Known(FeatureValue::Bool(expected)))
        );
    }

    fn acoustic_value<'a>(
        model: &'a AcousticTargetModel,
        name: &str,
    ) -> Option<&'a Spec<FeatureValue>> {
        model
            .expected_features
            .values
            .get(&FeatureId(format!("acoustic.{name}")))
    }
}
